from app.db import repository as repo
from app.db.models import Conversation, Faq, Ticket


async def test_create_and_get_conversation(db_session_factory, db_clean):
    cid = await repo.create_conversation("u1")
    conv = await repo.get_conversation(cid)
    assert conv is not None and conv.user_id == "u1"
    assert await repo.get_conversation(999999) is None


async def test_append_and_list_messages_in_order(db_session_factory, db_clean):
    cid = await repo.create_conversation("u1")
    await repo.append_message(cid, "user", content="订单 1001 到哪了")
    await repo.append_message(cid, "assistant", tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}])
    await repo.append_message(cid, "tool", content='{"status":"运输中"}', tool_call_id="c1")
    msgs = await repo.list_messages(cid)
    assert [m.role for m in msgs] == ["user", "assistant", "tool"]
    assert msgs[2].tool_call_id == "c1"


async def test_search_faq_like_hit_and_miss(db_session_factory, db_clean):
    async with db_session_factory() as s:
        s.add(Faq(question="退货政策", answer="7 天无理由退货", category="售后"))
        await s.commit()
    assert len(await repo.search_faq("退货")) == 1          # 命中
    assert await repo.search_faq("鞋子") == []              # 漏召回(字面不含)


async def test_create_ticket_writes_without_flipping_conversation_status(db_session_factory, db_clean):
    cid = await repo.create_conversation("u1")
    no = await repo.create_ticket(cid, "要退货", "售后")
    assert no.startswith("T")
    async with db_session_factory() as s:
        t = await s.get(Ticket, no)
        assert t.ticket_type == "售后" and t.status == "待处理"
        conv = await s.get(Conversation, cid)
        assert conv.status == "进行中"            # 建工单 ≠ 转人工


async def test_conversation_summary_columns_and_layer1(db_session_factory, db_clean):
    cid = await repo.create_conversation("u1")
    conv = await repo.get_conversation(cid)
    assert conv.summary is None
    assert conv.summary_upto_msg_id is None
    assert conv.layer1_from_msg_id is None
    await repo.update_layer1_from(cid, 5)
    assert (await repo.get_conversation(cid)).layer1_from_msg_id == 5


async def test_list_dialog_messages_filters_tool(db_session_factory, db_clean):
    cid = await repo.create_conversation("u1")
    await repo.append_message(cid, "user", content="订单1001到哪了")
    await repo.append_message(cid, "tool", content='{"s":1}', tool_call_id="c1")
    await repo.append_message(cid, "assistant", content="在路上")
    rows = await repo.list_dialog_messages(cid)
    assert [r.role for r in rows] == ["user", "assistant"]      # tool 行过滤,id 升序


async def test_append_summary_appends_and_projects(db_session_factory, db_clean):
    cid = await repo.create_conversation("u1")
    ids = []
    for i in range(4):
        ids.append(await repo.append_message(cid, "user", content=f"q{i} 订单100{i}"))
        ids.append(await repo.append_message(cid, "assistant", content=f"a{i}"))
    assert await repo.append_summary(cid, ids[0], ids[5], "第一段:订单1001/手机13800138000") == 1
    conv = await repo.get_conversation(cid)
    assert conv.summary_upto_msg_id == ids[5] and "订单1001" in conv.summary
    assert await repo.append_summary(cid, ids[6], ids[7], "第二段:补发立柱") == 2
    conv = await repo.get_conversation(cid)
    assert conv.summary_upto_msg_id == ids[7]
    assert "第一段" in conv.summary and "第二段" in conv.summary
    rows = await repo.list_summaries(cid)
    assert [r.seq for r in rows] == [1, 2]
    assert rows[0].content == "第一段:订单1001/手机13800138000"   # 旧段不被重写


async def test_append_summary_projection_keeps_newest(db_session_factory, db_clean, monkeypatch):
    cid = await repo.create_conversation("u1")
    ids = [await repo.append_message(cid, "user", content=f"q{i}") for i in range(4)]
    monkeypatch.setattr(repo.settings, "summary_budget_tokens", 10)
    await repo.append_summary(cid, ids[0], ids[1], "旧段" * 200)
    await repo.append_summary(cid, ids[2], ids[3], "新段")
    conv = await repo.get_conversation(cid)
    assert "旧段" not in conv.summary and "新段" in conv.summary
    assert len(await repo.list_summaries(cid)) == 2              # 表里两段都在


async def test_list_conversations_preview_has_summary_order(db_session_factory, db_clean):
    c1 = await repo.create_conversation("u1")
    await repo.append_message(c1, "user", content="第一问")
    await repo.append_message(c1, "assistant", content="答")
    c2 = await repo.create_conversation("u1")
    m1 = await repo.append_message(c2, "user", content="第二问")
    m2 = await repo.append_message(c2, "assistant", content="答")
    c3 = await repo.create_conversation("u2")
    await repo.append_message(c3, "user", content="别人的会话")
    await repo.append_summary(c2, m1, m2, "摘要")
    items = await repo.list_conversations("u1")
    assert [i["id"] for i in items] == [c2, c1]                  # id desc 新在前
    assert items[0]["preview"] == "第二问"
    assert items[0]["has_summary"] is True and items[1]["has_summary"] is False
    assert all(i["id"] != c3 for i in items)                     # 用户隔离


async def test_append_summary_does_not_regress_boundary(db_session_factory, db_clean):
    cid = await repo.create_conversation("u1")
    ids = [await repo.append_message(cid, "user", content=f"q{i}") for i in range(6)]
    assert await repo.append_summary(cid, ids[0], ids[4], "第一段") == 1
    # 另一进程已推进到更高边界:旧边界调用应放弃,不插重复段、不回退
    assert await repo.append_summary(cid, ids[0], ids[2], "更旧边界") == 0
    conv = await repo.get_conversation(cid)
    assert conv.summary_upto_msg_id == ids[4]
    assert len(await repo.list_summaries(cid)) == 1
