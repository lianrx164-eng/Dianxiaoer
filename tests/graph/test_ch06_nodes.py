import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.graph import nodes


@pytest.mark.asyncio
async def test_classify_intent_writes_confidence_and_route(monkeypatch):
    async def fake_classify(query, history=""):
        return {"intent": "退款退货", "confidence": 0.83}

    monkeypatch.setattr(nodes.intent_mod, "classify", fake_classify)
    out = await nodes.classify_intent({
        "messages": [HumanMessage("这个能退吗")],
        "resolved_query": "蓝牙耳机还能申请退货吗"})
    assert out["intent"] == "退款退货"
    assert out["intent_confidence"] == 0.83
    assert out["route"] == "refund_flow"
    assert out["trace"]["route"] == "refund_flow"
    assert out["trace"]["intent_confidence"] == 0.83
    assert "confidence" not in out["trace"]      # 不占用证据闸的 confidence 键


def test_get_chat_model_honors_model_override():
    from app.config import settings
    from app.core.llm import get_chat_model

    m = get_chat_model(model="glm-4-flash")
    assert m.model_name == "glm-4-flash"
    assert get_chat_model().model_name == settings.chat_model


@pytest.mark.asyncio
async def test_resolve_reference_passthrough_when_complete(monkeypatch):
    async def fake_resolve(q, history=""):
        return q

    monkeypatch.setattr(nodes.coref_mod, "resolve", fake_resolve)
    out = await nodes.coref({"messages": [HumanMessage("蓝牙耳机的保修期多久")]})
    assert out["resolved_query"] == "蓝牙耳机的保修期多久"
    assert out["trace"]["coref"] == "passthrough"


@pytest.mark.asyncio
async def test_resolve_reference_rewrites_with_history(monkeypatch):
    async def fake_resolve(q, history=""):
        assert "蓝牙耳机什么时候到" in history
        return "蓝牙耳机还能申请退货吗"

    monkeypatch.setattr(nodes.coref_mod, "resolve", fake_resolve)
    out = await nodes.coref({"messages": [
        HumanMessage("蓝牙耳机什么时候到"), AIMessage("预计明天"),
        HumanMessage("这个能退吗")]})
    assert out["resolved_query"] == "蓝牙耳机还能申请退货吗"
    assert out["trace"]["coref"] == "rewrite"


def test_extract_order_id():
    assert nodes._extract_order_id("订单1001的物流") == "1001"
    assert nodes._extract_order_id("尾号 20260701 那单") == "20260701"
    assert nodes._extract_order_id("我要退货") is None


@pytest.mark.asyncio
async def test_fetch_order_uses_id_in_query():
    out = await nodes.fetch_order({"resolved_query": "订单1001能退吗", "user_id": "u1"})
    assert out["order_id"] == "1001"
    assert out["order_data"]["order_id"] == "1001"
    assert out["trace"]["fetch_order"]["order_id"] == "1001"


@pytest.mark.asyncio
async def test_fetch_order_interrupts_when_missing(monkeypatch):
    monkeypatch.setattr(nodes.business, "list_user_orders",
                        lambda uid: [{"order_id": "1001", "product": "猫粮",
                                      "status": "已签收", "amount": 99}])
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph

    from app.graph.state import ConversationState

    b = StateGraph(ConversationState)
    b.add_node("fetch_order", nodes.fetch_order)
    b.add_edge(START, "fetch_order")
    b.add_edge("fetch_order", END)
    graph = b.compile(checkpointer=InMemorySaver())
    out = await graph.ainvoke({"resolved_query": "我要退款", "user_id": "u1"},
                              {"configurable": {"thread_id": "t-fetch"}})
    payload = out["__interrupt__"][0].value
    assert payload["type"] == "select_order"
    assert payload["orders"][0]["order_id"] == "1001"


def test_list_user_orders_stable_and_queryable():
    from app.tools import business

    a = business.list_user_orders("u-42")
    b = business.list_user_orders("u-42")
    assert a == b and len(a) >= 2
    snap = business.order_snapshot(a[0]["order_id"])
    assert snap["order_id"] == a[0]["order_id"]
    assert snap["product"] == a[0]["product"]


@pytest.mark.asyncio
async def test_retrieve_policy_expands_dedups_merges(monkeypatch):
    async def fake_expand(q):
        return ["退货政策", "无理由退换货", "退货时限"]

    per_query = {
        "退货政策": [{"id": 1, "question": "退货", "answer": "7天无理由",
                     "rerank_score": 0.7, "section_path": "政策/退货",
                     "content_type": "policy"}],
        "无理由退换货": [
            {"id": 1, "question": "退货", "answer": "7天无理由",
             "rerank_score": 0.9, "section_path": "政策/退货",
             "content_type": "policy"},
            {"id": 2, "question": "运费", "answer": "质量问题商家承担",
             "rerank_score": 0.6, "section_path": "政策/运费",
             "content_type": "policy"}],
        "退货时限": [],
    }

    async def fake_search(q, **k):
        return per_query.get(q, [])

    monkeypatch.setattr(nodes.query_understanding, "expand_queries", fake_expand)
    monkeypatch.setattr(nodes.retrieval, "search_knowledge", fake_search)
    monkeypatch.setattr(nodes.retrieval, "arrange_head_tail", lambda h: h)

    out = await nodes.retrieve_policy({"resolved_query": "这个订单能退吗",
                                       "order_data": {"status": "已签收"}})
    assert [c["id"] for c in out["citations"]] == [1, 2]
    assert "[1]" in out["evidence"] and "[2]" in out["evidence"]
    assert out["trace"]["retrieve_policy"]["hits"] == 2
    assert out["trace"]["retrieve_policy"]["queries"] == [
        "退货政策", "无理由退换货", "退货时限"]


@pytest.mark.asyncio
async def test_agent_tools_intercepts_submit_refund():
    ai = AIMessage(content="", tool_calls=[
        {"id": "r1", "name": "submit_refund",
         "args": {"order_id": "1001", "reason": None}}])
    out = await nodes.agent_tools({"messages": [ai], "order_id": "1001"})
    assert out["suggested_actions"] == [
        {"type": "refund_form",
         "draft": {"order_id": "1001", "reason": None}}]
    tm = out["messages"][0]
    assert tm.name == "submit_refund"
    assert "退款" in tm.content


def test_agent_messages_injects_order_and_policy_on_refund():
    from langchain_core.messages import SystemMessage

    msgs = nodes._agent_messages({
        "route": "refund_flow",
        "order_data": {"order_id": "1001", "status": "已签收",
                       "product": "猫粮 5kg"},
        "evidence": "[1] 退货: 7天无理由",
        "messages": [HumanMessage("这单能退吗")]})
    sys = msgs[0]
    assert isinstance(sys, SystemMessage)
    assert "7天无理由" in sys.content
    assert "猫粮 5kg" in sys.content
    assert "submit_refund" in sys.content


def test_agent_messages_knowledge_path_still_injects_evidence():
    msgs = nodes._agent_messages({
        "route": "knowledge", "evidence": "[1] 运费: 满99包邮",
        "messages": [HumanMessage("运费多少")]})
    assert "满99包邮" in msgs[0].content
