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
