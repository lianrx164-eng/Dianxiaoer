"""ch10 作业运行器:注册表字段、未跑过的状态、强跑参数的 argv。"""
from app.core import jobs


def test_unknown_job_is_not_registered():
    assert "rm-rf" not in jobs.JOBS


def test_unstarted_job_reports_idle():
    st = jobs.status("ch10-eval")
    assert st["running"] is False and st["exit_code"] is None


def test_specs_carry_title_and_needs_defaults():
    spec = jobs.JOBS["ch10-eval"]
    assert spec.title and spec.needs
    assert jobs.JOBS["classifier-down"].needs == ""


def test_force_job_passes_make_args():
    spec = jobs.JOBS["classify-pool-force"]
    assert spec.target == "classify-pool" and spec.args == ("FORCE=1",)
