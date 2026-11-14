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
    assert out["trace"]["resolved_query"] == "蓝牙耳机还能申请退货吗"


def test_extract_order_id():
    assert nodes._extract_order_id("订单1001的物流") == "1001"
    assert nodes._extract_order_id("尾号 20260701 那单") == "20260701"
    assert nodes._extract_order_id("我要退货") is None


def test_extract_order_id_rejects_phone_amount_year():
    assert nodes._extract_order_id("我要退款 手机号13812345678") is None
    assert nodes._extract_order_id("退款金额3999.00元") is None
    assert nodes._extract_order_id("我2026年买的猫粮要退款") is None
    # 同句里前一个数字带「订单」线索,不能顺带把后面的金额也认成订单号
    assert nodes._extract_order_id("订单1001 金额3999") == "1001"


def test_extract_order_id_accepts_known_order_without_cue():
    assert nodes._extract_order_id("帮我退 7495", {"7495"}) == "7495"
    assert nodes._extract_order_id("帮我退 7495", set()) is None


@pytest.mark.asyncio
async def test_fetch_order_ignores_phone_number(monkeypatch):
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
    out = await graph.ainvoke(
        {"resolved_query": "我要退款 手机号13812345678", "user_id": "u1"},
        {"configurable": {"thread_id": "t-phone"}})
    assert out["__interrupt__"][0].value["type"] == "select_order"


@pytest.mark.asyncio
async def test_fetch_order_resume_queries_snapshot_once(monkeypatch):
    calls = {"list": 0, "snapshot": 0}
    real_snapshot = nodes.business.order_snapshot

    def fake_list(uid):
        calls["list"] += 1
        return [{"order_id": "1001", "product": "猫粮",
                 "status": "已签收", "amount": 99}]

    def counting_snapshot(oid):
        calls["snapshot"] += 1
        return real_snapshot(oid)

    monkeypatch.setattr(nodes.business, "list_user_orders", fake_list)
    monkeypatch.setattr(nodes.business, "order_snapshot", counting_snapshot)
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command

    from app.graph.state import ConversationState

    b = StateGraph(ConversationState)
    b.add_node("fetch_order", nodes.fetch_order)
    b.add_edge(START, "fetch_order")
    b.add_edge("fetch_order", END)
    graph = b.compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t-resume"}}

    first = await graph.ainvoke({"resolved_query": "我要退款", "user_id": "u1"}, cfg)
    assert first["__interrupt__"][0].value["orders"][0]["order_id"] == "1001"
    assert calls["snapshot"] == 0          # interrupt 之前不查单

    done = await graph.ainvoke(Command(resume="1001"), cfg)
    assert done["order_id"] == "1001"
    assert calls["list"] == 2              # 节点从头重跑;只读可重跑
    assert calls["snapshot"] == 1          # 查单只在 resume 后发生一次


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
    out = await nodes.agent_tools({"messages": [ai], "order_id": "1001",
                                   "route": "refund_flow"})
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


def test_tools_for_hides_submit_refund_outside_refund_flow():
    business_names = {t.name for t in nodes._tools_for({"route": "business"})}
    refund_names = {t.name for t in nodes._tools_for({"route": "refund_flow"})}
    assert "submit_refund" not in business_names
    assert "submit_refund" in refund_names


@pytest.mark.asyncio
async def test_agent_tools_refuses_submit_refund_outside_refund_flow():
    from langchain_core.messages import ToolMessage

    ai = AIMessage(content="", tool_calls=[
        {"id": "r2", "name": "submit_refund",
         "args": {"order_id": "", "reason": None}}])
    out = await nodes.agent_tools({"messages": [ai], "route": "business"})
    assert not out.get("suggested_actions")
    assert isinstance(out["messages"][0], ToolMessage)
    assert "不能发起退款" in out["messages"][0].content


def test_history_text_skips_tool_messages():
    from langchain_core.messages import ToolMessage

    hist = nodes._history_text({"messages": [
        HumanMessage("订单1001到哪了"),
        ToolMessage(content='{"order_id":"1001","status":"已发货"}', tool_call_id="1"),
        AIMessage("已发货"),
        HumanMessage("那它能退吗")]})
    assert "order_id" not in hist
    assert "已发货" in hist and "订单1001到哪了" in hist


@pytest.mark.asyncio
async def test_script_reply_by_intent():
    from app.core.prompts import CHITCHAT_REPLY_TEXT, SCRIPT_REPLY_OTHER

    chit = await nodes.script_reply({"intent": "闲聊",
                                     "trace": {"route": "fallback_script"}})
    assert chit["answer"] == CHITCHAT_REPLY_TEXT
    assert chit["trace"]["route"] == "fallback_script"   # 不覆盖分流出口
    assert chit["trace"]["handler"] == "script"
    assert not chit.get("suggested_actions")

    other = await nodes.script_reply({"intent": "其他"})
    assert other["answer"] == SCRIPT_REPLY_OTHER


@pytest.mark.asyncio
async def test_reply_nodes_preserve_outlet_route(monkeypatch):
    async def fake_insert(*a, **k):
        return 1

    monkeypatch.setattr(nodes.repository, "insert_low_confidence", fake_insert)

    out = await nodes.complaint_reply(
        {"messages": [HumanMessage("我要投诉")], "trace": {"route": "escalate"}})
    assert out["trace"]["route"] == "escalate"
    assert out["trace"]["handler"] == "complaint"

    out = await nodes.fallback_reply({
        "messages": [HumanMessage("q")],
        "trace": {"route": "knowledge", "gate": "retrieval_low_conf"}})
    assert out["trace"]["route"] == "knowledge"
    assert out["trace"]["handler"] == "fallback"

    ai = AIMessage("", tool_calls=[{"name": "query_order", "args": {}, "id": "c1"}])
    out = await nodes.cap_reply({"messages": [HumanMessage("查订单"), ai],
                                 "trace": {"route": "business"}})
    assert out["trace"]["route"] == "business"
    assert out["trace"]["handler"] == "cap"
