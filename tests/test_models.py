from app.db.models import Conversation, ConversationSummary, Message


async def test_conversation_defaults_and_autoincrement(db_session_factory, db_clean):
    async with db_session_factory() as s:
        conv = Conversation(user_id="u1")
        s.add(conv)
        await s.commit()
        assert conv.id is not None          # BIGINT 自增
        await s.refresh(conv)               # MySQL 8 无 RETURNING,server_default 需读回
        assert conv.status == "进行中"        # DB DEFAULT 读回
        assert conv.created_at is not None


async def test_message_json_tool_calls_roundtrip(db_session_factory, db_clean):
    async with db_session_factory() as s:
        conv = Conversation(user_id="u1")
        s.add(conv)
        await s.flush()
        msg = Message(
            conversation_id=conv.id,
            role="assistant",
            content=None,
            tool_calls=[{"name": "query_order", "args": {"order_id": "1001"}, "id": "c1"}],
        )
        s.add(msg)
        await s.commit()
        got = await s.get(Message, msg.id)
        assert got.tool_calls[0]["name"] == "query_order"   # JSON 往返
        assert got.role == "assistant"


async def test_conversation_summary_row_roundtrip(db_session_factory, db_clean):
    async with db_session_factory() as s:
        row = ConversationSummary(
            conversation_id=1, seq=1, from_msg_id=1, upto_msg_id=2,
            content="用户问过订单1001")
        s.add(row)
        await s.commit()
        await s.refresh(row)
        assert row.id is not None and row.created_at is not None
        assert row.content == "用户问过订单1001"
