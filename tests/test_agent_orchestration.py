"""图级编排测试:确定性出口、知识路证据闸、业务路与步数封顶。"""
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.config import settings
from app.graph import nodes
from app.graph.build import build_graph


def _aval(value):
    async def _f(*a, **k):
        return value
    return _f


def _graph(thread="t1"):
    return build_graph(checkpointer=InMemorySaver()), {"configurable": {"thread_id": thread}}


def _input(text="q"):
    return {"messages": [HumanMessage(text)], "conversation_id": 1,
            "steps": 0, "tokens_used": 0, "trace": {}}


@pytest.fixture(autouse=True)
def _quiet_db(monkeypatch):
    async def fake_append(*a, **k):
        return 1

    async def fake_insert(*a, **k):
        return 1

    monkeypatch.setattr(nodes.repository, "append_message", fake_append)
    monkeypatch.setattr(nodes.repository, "insert_low_confidence", fake_insert)


async def test_chitchat_route_returns_fixed_reply(monkeypatch):
    from app.core.prompts import CHITCHAT_REPLY_TEXT

    monkeypatch.setattr(nodes, "classify_intent", _aval(
        {"intent": "闲聊", "route": "fallback_script",
         "trace": {"intent": "闲聊", "route": "fallback_script"}}))
    g, config = _graph()
    final = await g.ainvoke(_input("你好"), config)
    assert final["answer"] == CHITCHAT_REPLY_TEXT
    assert final["route"] == "fallback_script"


async def test_business_route_goes_straight_to_agent(monkeypatch):
    monkeypatch.setattr(nodes, "classify_intent", _aval(
        {"intent": "订单", "route": "business", "trace": {"route": "business"}}))

    async def fake_agent(state, config=None):
        return {"messages": [AIMessage("订单已发货")], "steps": 1, "tokens_used": 5}

    monkeypatch.setattr(nodes, "agent_llm", fake_agent)
    g, config = _graph()
    final = await g.ainvoke(_input("订单1001"), config)
    assert nodes.resolve_answer(final) == "订单已发货"
    assert "forced_rag" not in final.get("trace", {})


async def test_knowledge_weak_evidence_falls_back_and_pools(monkeypatch):
    calls = {}

    async def fake_insert(conversation_id, raw_question, source, reason):
        calls.update(cid=conversation_id, source=source)
        return 1

    monkeypatch.setattr(nodes.repository, "insert_low_confidence", fake_insert)
    monkeypatch.setattr(nodes, "classify_intent", _aval(
        {"intent": "商品咨询", "route": "knowledge", "trace": {"route": "knowledge"}}))
    monkeypatch.setattr(nodes, "forced_rag", _aval(
        {"evidence_strong": False,
         "trace": {"forced_rag": True, "gate": "retrieval_low_conf", "evidence_top": 0.1}}))

    async def boom(state, config=None):
        raise AssertionError("证据弱不应进 Agent")

    monkeypatch.setattr(nodes, "agent_llm", boom)
    g, config = _graph()
    final = await g.ainvoke(_input("怎么注销账号"), config)
    assert final["answer"] == nodes.FALLBACK_REPLY
    assert calls == {"cid": 1, "source": "retrieval_low_conf"}


async def test_knowledge_strong_evidence_reaches_agent_with_citations(monkeypatch):
    citations = [{"n": 1, "id": 1, "section_path": "退货", "question": "能退吗", "answer": "7天"}]
    monkeypatch.setattr(nodes, "classify_intent", _aval(
        {"intent": "商品咨询", "route": "knowledge", "trace": {"route": "knowledge"}}))
    monkeypatch.setattr(nodes, "forced_rag", _aval(
        {"evidence_strong": True, "evidence": "[1] 能退吗: 7天", "citations": citations,
         "trace": {"forced_rag": True, "evidence_top": 0.9}}))

    async def fake_agent(state, config=None):
        assert "已检索到的知识证据" in nodes._agent_messages(state)[0].content
        return {"messages": [AIMessage("7天无理由[1]")], "steps": 1, "tokens_used": 5}

    monkeypatch.setattr(nodes, "agent_llm", fake_agent)
    g, config = _graph()
    final = await g.ainvoke(_input("能退吗"), config)
    assert final["citations"] == citations
    assert nodes.resolve_answer(final) == "7天无理由[1]"


async def test_step_cap_aborts_pending_tools(monkeypatch):
    monkeypatch.setattr(settings, "max_agent_steps", 1)
    monkeypatch.setattr(nodes, "classify_intent", _aval(
        {"intent": "订单", "route": "business", "trace": {"route": "business"}}))

    async def fake_agent(state, config=None):
        return {"messages": [AIMessage("", tool_calls=[
            {"name": "query_order", "args": {"order_id": "1"}, "id": "c1"}])],
            "steps": 1, "tokens_used": 5}

    monkeypatch.setattr(nodes, "agent_llm", fake_agent)
    g, config = _graph()
    final = await g.ainvoke(_input("查订单"), config)
    assert final["answer"] == nodes.FALLBACK_REPLY
    assert final["trace"]["cap"] is True
    tool_msgs = [m for m in final["messages"] if getattr(m, "tool_call_id", None) == "c1"]
    assert tool_msgs and "已中止" in tool_msgs[0].content
