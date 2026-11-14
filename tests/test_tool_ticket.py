from app.db.models import Conversation, Ticket
from app.tools.builtin.tickets import create_ticket


def test_conversation_id_hidden_from_model_schema():
    # InjectedToolArg 的参数不应出现在暴露给 LLM 的 schema 里。
    # 注意:get_input_schema() 含 injected 参数(供执行时传参),
    # 真正发给模型的是 tool_call_schema(bind_tools 用它),这里断言后者。
    schema = create_ticket.tool_call_schema.model_json_schema()
    props = schema.get("properties", {})
    assert "description" in props and "ticket_type" in props
    assert "conversation_id" not in props        # 关键:模型看不到会话主键


async def test_create_ticket_injects_conversation_id_and_writes(db_session_factory, db_clean):
    async with db_session_factory() as s:
        conv = Conversation(user_id="u1")
        s.add(conv)
        await s.commit()
        cid = conv.id
    r = await create_ticket.ainvoke(
        {"description": "商品损坏要退货", "ticket_type": "售后", "conversation_id": cid}
    )
    assert r["ticket_no"].startswith("T")
    assert r["status"] == "已创建"          # 建工单不等于转人工
    async with db_session_factory() as s:
        t = await s.get(Ticket, r["ticket_no"])
        assert t is not None and t.conversation_id == cid


async def test_create_ticket_does_not_mark_conversation_transferred(db_session_factory, db_clean):
    from app.db.models import Conversation

    async with db_session_factory() as s:
        conv = Conversation(user_id="u1")
        s.add(conv)
        await s.commit()
        cid = conv.id
    await create_ticket.ainvoke(
        {"description": "东西坏了", "ticket_type": "售后", "conversation_id": cid}
    )
    async with db_session_factory() as s:
        conv = await s.get(Conversation, cid)
        assert conv.status == "进行中"     # 建工单 ≠ 转人工
