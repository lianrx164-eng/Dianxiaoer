"""runtime 测试共用打桩:避免每个测试文件复制
get_conversation / list_dialog_messages / maybe_schedule_summary 三件套。"""
from types import SimpleNamespace


def patch_runtime_repo(monkeypatch, runtime, summary="", upto=0, layer1_from=0):
    async def fake_get(cid):
        return SimpleNamespace(summary=summary, summary_upto_msg_id=upto,
                               layer1_from_msg_id=layer1_from)

    async def fake_rows(cid):
        return []

    async def fake_schedule(cid):
        return None

    monkeypatch.setattr(runtime.repository, "get_conversation", fake_get)
    monkeypatch.setattr(runtime.repository, "list_dialog_messages", fake_rows)
    monkeypatch.setattr(runtime.summarizer, "maybe_schedule_summary", fake_schedule)
