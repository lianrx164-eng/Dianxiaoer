"""作业运行器:只跑白名单里的 make 目标,argv 写死在模块里,前端只能传作业名。

日志落 log/acceptance/<job>.log,页面轮询回显尾部;停止时连 make → uv → python 一起收。
"""
import os
import signal
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = REPO_ROOT / "log" / "acceptance"


@dataclass(frozen=True)
class JobSpec:
    name: str
    target: str
    heavy: bool = False
    title: str = ""          # 页面按钮上的一行说明
    needs: str = ""          # 前置条件提示(如「需 :8110」)
    args: tuple[str, ...] = ()   # 追加给 make 的参数(如 FORCE=1)


JOBS: dict[str, JobSpec] = {
    spec.name: spec
    for spec in [
        JobSpec("kb-preview", "kb-preview"),
        JobSpec("kb-build", "kb-build"),
        JobSpec("kb-mine", "kb-mine", heavy=True),
        JobSpec("seed-conv", "seed-conv"),
        JobSpec("eval-rag", "eval-rag", heavy=True),
        JobSpec("cost-report", "cost-report"),
        JobSpec("eval-flywheel", "eval-flywheel", heavy=True),
        JobSpec("calibrate-confidence", "calibrate-confidence", heavy=True),
        # ch10 主题分类器:数据→训练→评测→导出→旁路归类
        JobSpec("ch10-golden", "ch10-golden", title="预标黄金样例闸", needs="上游可用"),
        JobSpec("ch10-corpus", "ch10-corpus", heavy=True,
                title="语料流水线", needs="mysql + 上游可用"),
        JobSpec("ch10-dataset", "ch10-dataset", heavy=True,
                title="划分与增强", needs="上游可用"),
        JobSpec("ch10-train", "ch10-train", heavy=True,
                title="全参微调", needs="ml 组已装"),
        JobSpec("ch10-eval", "ch10-eval", title="测试集评测", needs="模型已训练"),
        JobSpec("ch10-export", "ch10-export", title="导出 ONNX", needs="模型已训练"),
        JobSpec("ch10-threshold-scan", "ch10-threshold-scan",
                title="阈值扫描重演", needs="服务 :8110 在跑"),
        JobSpec("classifier-up", "classifier-up", title="拉起分类器", needs="ONNX 已导出"),
        JobSpec("classifier-down", "classifier-down", title="停掉分类器"),
        JobSpec("classify-pool", "classify-pool", title="旁路批量归类",
                needs="服务 :8110 在跑"),
        JobSpec("classify-pool-force", "classify-pool", title="强跑归类(不足一批)",
                needs="服务 :8110 在跑", args=("FORCE=1",)),
    ]
}

# 子进程要打开 Milvus Lite 文件,起之前必须让出进程内单例锁
_MILVUS_JOBS = {"eval-rag", "eval-flywheel", "calibrate-confidence"}


class JobAlreadyRunning(Exception):
    pass


_procs: dict[str, subprocess.Popen] = {}
_handles: dict[str, object] = {}
# 同步 endpoint 跑在线程池里,「检查 + Popen + 登记」必须原子,否则并发双击会起两份
_lock = threading.Lock()


def _log_path(name: str) -> Path:
    return LOG_DIR / f"{name}.log"


def _tail(name: str, lines: int = 40) -> str:
    path = _log_path(name)
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return "\n".join(text.splitlines()[-lines:])


def status(name: str) -> dict:
    spec = JOBS[name]
    proc = _procs.get(name)
    running = bool(proc is not None and proc.poll() is None)
    exit_code = None if running or proc is None else proc.poll()
    return {
        "name": name,
        "target": spec.target,
        "heavy": spec.heavy,
        "running": running,
        "exit_code": exit_code,
        "log_tail": _tail(name),
    }


def status_all() -> list[dict]:
    return [status(name) for name in JOBS]


def start(name: str) -> dict:
    spec = JOBS[name]
    with _lock:
        if name in _procs and _procs[name].poll() is None:
            raise JobAlreadyRunning(name)

        old = _handles.pop(name, None)
        if old is not None:
            old.close()

        if name in _MILVUS_JOBS:
            # 离线评估/校准子进程要打开 Lite 数据目录,先让出进程内单例锁
            from app.kb import milvus_client
            milvus_client.release()

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        handle = open(_log_path(name), "w", encoding="utf-8")
        kwargs: dict = dict(
            cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True,
        )
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        proc = subprocess.Popen(["make", spec.target, *spec.args], **kwargs)
        _procs[name] = proc
        _handles[name] = handle
        return status(name)


def stop(name: str) -> dict:
    JOBS[name]  # 未知作业 → KeyError
    with _lock:
        proc = _procs.get(name)
        if proc is not None and proc.poll() is None:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    capture_output=True,
                )
            else:
                os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
        return status(name)
