import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.graph import nodes
from app.graph.build import build_graph
from app.tools import engine, registry


async def test_sticky_flag_routes_without_classifier(monkeypatch):
    called = []

    async def fake_classify(q, h=""):
        called.append((q, h))
        return {"intent": "售后", "confidence": 0.92}

    monkeypatch.setattr(nodes.intent_mod, "classify", fake_classify)
    out = await nodes.classify_intent({
        "messages": [HumanMessage("猫砂盆漏电")],
        "resolved_query": "猫砂盆漏电", "ticket_followup": True})
    assert out["intent"] == "人工" and out["route"] == "business"
    assert out["ticket_followup"] is False
    assert called == []


async def test_human_intent_sets_next_turn_sticky(monkeypatch):
    async def fake_classify(q, h=""):
        return {"intent": "人工", "confidence": 0.97}

    monkeypatch.setattr(nodes.intent_mod, "classify", fake_classify)
    out = await nodes.classify_intent({
        "messages": [HumanMessage("帮我建个工单")],
        "resolved_query": "帮我建个工单"})
    assert out["route"] == "business" and out["ticket_followup"] is True


async def test_sticky_flag_cleared_after_one_turn(monkeypatch):
    async def fake_classify(q, h=""):
        return {"intent": "售后", "confidence": 0.92}

    monkeypatch.setattr(nodes.intent_mod, "classify", fake_classify)
    out = await nodes.classify_intent({
        "messages": [HumanMessage("那换个问题")],
        "resolved_query": "那换个问题", "ticket_followup": False})
    assert out["route"] == "refund_flow"      # 恢复正常分流
    assert out["ticket_followup"] is False


class _FollowupModel:
    """第 1 轮发缺 description 的 create_ticket;追问后第 3 次调用发完整建单。"""

    def __init__(self):
        self.calls = 0

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, msgs, config=None):
        self.calls += 1
        if self.calls == 1:
            return AIMessage("", tool_calls=[{
                "name": "create_ticket", "id": "t1",
                "args": {"ticket_type": "咨询"}}])
        if self.calls == 2:
            return AIMessage("请问您遇到的具体问题是什么呢?")
        if self.calls == 3:
            return AIMessage("", tool_calls=[{
                "name": "create_ticket", "id": "t2",
                "args": {"description": "猫砂盆漏电", "ticket_type": "售后"}}])
        return AIMessage("请确认工单信息")


def _input(text):
    return {
        "messages": [HumanMessage(text)], "user_id": "u1", "conversation_id": 1,
        "history_messages": [], "summary": "", "summary_upto_msg_id": 0,
        "layer1_from_msg_id": 0, "intent": "", "route": "", "evidence": "",
        "citations": [], "evidence_strong": False, "answer": "", "steps": 0,
        "tokens_used": 0, "suggested_actions": [], "trace": {},
        "resolved_query": "", "intent_confidence": 0.0,
        "order_id": "", "order_data": {},
    }


async def test_followup_turn_reaches_agent_and_interrupts(monkeypatch):
    model = _FollowupModel()
    intent_calls = []

    async def fake_coref(q, h=""):
        return q

    async def fake_intent(q, h=""):
        intent_calls.append(q)
        # 第一句「帮我建个工单」正常判人工;单看补充句「猫砂盆漏电」会被判售后
        return {"intent": "人工" if "工单" in q else "售后", "confidence": 0.92}

    async def only_builtin():
        return registry.builtin_specs()

    async def fake_audit(**kw):
        return None

    async def no_msg(*a, **kw):
        return 1

    monkeypatch.setattr(nodes.coref_mod, "resolve", fake_coref)
    monkeypatch.setattr(nodes.intent_mod, "classify", fake_intent)
    monkeypatch.setattr(nodes, "get_chat_model", lambda **kw: model)
    monkeypatch.setattr(nodes.registry, "get_all_specs", only_builtin)
    monkeypatch.setattr(engine.repository, "insert_tool_audit", fake_audit)
    monkeypatch.setattr(nodes.repository, "append_message", no_msg)

    g = build_graph(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "follow-1"}}

    st1 = await g.ainvoke(_input("帮我建个工单"), cfg)
    assert "__interrupt__" not in st1
    assert st1.get("ticket_followup") is True
    assert intent_calls == ["帮我建个工单"]

    st2 = await g.ainvoke(_input("猫砂盆漏电"), cfg)
    assert intent_calls == ["帮我建个工单"]          # 粘滞放行,没再调分类器
    intr = st2["__interrupt__"][0].value
    assert intr["type"] == "confirm_ticket"
    assert intr["preview"] == {"ticket_type": "售后", "description": "猫砂盆漏电"}
