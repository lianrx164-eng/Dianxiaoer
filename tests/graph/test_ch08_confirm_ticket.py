import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.graph import nodes
from app.graph.build import build_graph
from app.tools import engine, registry


class FakeModel:
    def __init__(self):
        self.calls = 0

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, msgs, config=None):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(content="", tool_calls=[{
                "name": "create_ticket", "id": "tc-1",
                "args": {"description": "猫砂盆漏电", "ticket_type": "售后"}}])
        return AIMessage(content="已为您创建工单")


@pytest.fixture()
def wired(monkeypatch):
    async def fake_coref(q, h=""):
        return q

    async def fake_intent(q, h=""):
        return {"intent": "订单", "confidence": 0.95}

    monkeypatch.setattr(nodes.coref_mod, "resolve", fake_coref)
    monkeypatch.setattr(nodes.intent_mod, "classify", fake_intent)
    monkeypatch.setattr(nodes, "get_chat_model", lambda **kw: FakeModel())

    async def only_builtin():
        return registry.builtin_specs()

    monkeypatch.setattr(nodes.registry, "get_all_specs", only_builtin)

    created, audits = [], []

    async def fake_create(cid, desc, ttype):
        created.append((cid, desc, ttype))
        return "T20261114001"

    async def fake_audit(**kw):
        audits.append(kw)

    monkeypatch.setattr("app.db.repository.create_ticket", fake_create)
    monkeypatch.setattr(engine.repository, "insert_tool_audit", fake_audit)

    async def no_msg(*a, **kw):
        return 1

    monkeypatch.setattr(nodes.repository, "append_message", no_msg)
    return created, audits


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


async def test_confirm_true_creates_ticket(wired):
    created, audits = wired
    g = build_graph(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t1"}}
    st = await g.ainvoke(_input("帮我建个工单,猫砂盆漏电"), cfg)
    intr = st["__interrupt__"][0].value
    assert intr["type"] == "confirm_ticket"
    assert intr["preview"] == {"ticket_type": "售后", "description": "猫砂盆漏电"}
    assert created == []

    st2 = await g.ainvoke(Command(resume={"confirmed": True}), cfg)
    assert created == [(1, "猫砂盆漏电", "售后")]
    assert any(a["status"] == "成功" for a in audits)
    assert "T20261114001" in str(st2["messages"])


async def test_confirm_false_denied_and_audited(wired):
    created, audits = wired
    g = build_graph(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t2"}}
    await g.ainvoke(_input("帮我建个工单,猫砂盆漏电"), cfg)
    await g.ainvoke(Command(resume={"confirmed": False}), cfg)
    assert created == []
    assert any(a["status"] == "权限拒绝" for a in audits)


async def test_resume_value_not_dict_treated_as_cancel(wired):
    created, audits = wired
    g = build_graph(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t3"}}
    await g.ainvoke(_input("帮我建个工单,猫砂盆漏电"), cfg)
    await g.ainvoke(Command(resume="oops"), cfg)
    assert created == []
    assert any(a["status"] == "权限拒绝" for a in audits)


async def test_missing_description_asks_instead_of_interrupt(wired, monkeypatch):
    created, audits = wired

    class NoDescModel(FakeModel):
        async def ainvoke(self, msgs, config=None):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(content="", tool_calls=[{
                    "name": "create_ticket", "id": "tc-1",
                    "args": {"ticket_type": "咨询"}}])
            return AIMessage(content="请问您遇到的具体问题是什么呢?")

    monkeypatch.setattr(nodes, "get_chat_model", lambda **kw: NoDescModel())
    g = build_graph(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t4"}}
    st = await g.ainvoke(_input("帮我建个工单"), cfg)
    assert "__interrupt__" not in st
    assert created == []
    assert any(a["status"] == "校验拦下" for a in audits)
