import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.core.prompts import AGENT_SYSTEM
from app.graph import nodes


def _state(n_turns=2, summary="", upto=0, layer1_from=0, current="现在的问题",
           evidence="", route="business", extra=()):
    hist, did = [], 1
    for i in range(n_turns):
        hist.append(HumanMessage(f"问题{i}", id=f"db-{did}")); did += 1
        hist.append(AIMessage(f"回答{i}", id=f"db-{did}")); did += 1
    prior = [m for m in hist if int(m.id.split("-")[1]) > upto]
    return {"messages": [*hist, HumanMessage(current), *extra],
            "history_messages": prior, "summary": summary,
            "summary_upto_msg_id": upto, "layer1_from_msg_id": layer1_from,
            "evidence": evidence, "route": route, "conversation_id": 7}


def test_order_with_summary():
    msgs = nodes._agent_messages(_state(
        n_turns=2, summary="用户问过订单1001", evidence="[1] 7天无理由",
        route="knowledge", current="能退吗"))
    assert isinstance(msgs[0], SystemMessage) and msgs[0].content == AGENT_SYSTEM
    assert "7天无理由" not in msgs[0].content             # 证据不进 system(AGENT_SYSTEM 自带 [1] 示例)
    assert isinstance(msgs[1], HumanMessage) and msgs[1].content == "问题0"
    assert msgs[-2].content == "能退吗"                    # 当前用户
    assert "订单1001" in msgs[-1].content and "[1]" in msgs[-1].content


def test_system_is_always_static_for_all_routes():
    for state in (
        _state(route="business"),
        _state(route="knowledge", evidence="[1] 运费: 满99包邮"),
        _state(route="refund_flow", evidence="[1] 7天无理由",
               extra=(AIMessage("", tool_calls=[
                   {"name": "submit_refund", "args": {"order_id": "1001"}, "id": "r1"}]),)),
    ):
        assert nodes._agent_messages(state)[0].content == AGENT_SYSTEM


def test_step1_prompt_is_strict_prefix_of_step2():
    q = HumanMessage("能退吗", id="db-tmp")
    s1 = nodes._agent_messages({"messages": [q], "history_messages": [],
                                "route": "knowledge", "evidence": "[1] 7天"})
    s2 = nodes._agent_messages({"messages": [
        q, AIMessage("", tool_calls=[{"name": "query_order", "args": {}, "id": "c1"}])],
        "history_messages": [], "route": "knowledge", "evidence": "[1] 7天"})
    shape = lambda ms: [(m.type, m.content) for m in ms]      # noqa: E731
    assert shape(s2)[:len(s1)] == shape(s1)


def test_history_text_prepends_summary_and_windows():
    text = nodes._history_text(_state(n_turns=6, summary="用户问过订单1001"))
    assert text.startswith("(早前对话摘要:用户问过订单1001")
    assert "问题0" not in text and "问题3" in text


def test_history_text_legacy_falls_back_to_messages():
    text = nodes._history_text({"messages": [
        HumanMessage("q0"), AIMessage("a0"), HumanMessage("q1")]})
    assert "q0" in text and "a0" in text and "q1" not in text


def test_log_model_context_contains_summary_and_window(caplog):
    state = _state(n_turns=2, summary="用户问过订单1001", evidence="[1] 7天",
                   route="knowledge", current="能退吗")
    msgs = nodes._agent_messages(state)
    with caplog.at_level(logging.INFO, logger="app.graph.nodes"):
        nodes._log_model_context(state, msgs)
    assert "model_ctx" in caplog.text
    assert "用户问过订单1001" in caplog.text
    assert "[human] '问题0" in caplog.text
    assert "tokens=" in caplog.text
