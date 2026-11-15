"""app/ 里 import 的每个第三方包都必须在 [project] dependencies 里。

只声明在 dev 组的包(测试工具)一旦被运行时模块引用,换掉传递依赖它的库
就会在 import 阶段崩,而单测发现不了——所以这条用静态检查钉住。
"""
import ast
import re
import sys
import tomllib
from importlib.metadata import packages_distributions
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_REQ_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")
_LOCAL = {"app", "scripts", "tests", "mcp_servers"}


def _declared() -> set[str]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return {_REQ_NAME.match(s).group(0).lower().replace("-", "_")
            for s in data["project"]["dependencies"]}


def _runtime_imports() -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for path in (ROOT / "app").rglob("*.py"):
        where = str(path.relative_to(ROOT))
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                found.setdefault(name.split(".")[0], set()).add(where)
    return found


def test_runtime_imports_are_declared():
    declared = _declared()
    packages = packages_distributions()
    missing = {}
    for mod, where in _runtime_imports().items():
        if mod in sys.stdlib_module_names or mod in _LOCAL:
            continue
        # 一个模块名可能由多个发行版提供(命名空间包),任一显式声明即可
        dists = {d.lower().replace("-", "_") for d in packages.get(mod) or [mod]}
        if not dists & declared:
            missing[mod] = sorted(where)
    assert not missing, f"运行时 import 未声明: {missing}"
