from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.core import context_budget
from app.core.memory import (
    SessionStore, build_layers, db_id, degrade_layer1, render_layer2,
    split_layers, summary_line, trim_history, trim_turns_to_budget)


def test_store_get_unknown_session_returns_empty():
    assert SessionStore().get("nope") == []


def test_store_append_and_get_isolated_by_session():
    store = SessionStore()
    store.append("a", HumanMessage("hi"), AIMessage("hello"))
    store.append("b", HumanMessage("嗨"))
    assert len(store.get("a")) == 2
    assert len(store.get("b")) == 1


def test_trim_keeps_recent_and_starts_on_human():
    msgs = []
    for i in range(20):
        msgs.append(HumanMessage(f"问题{i}:" + "喵" * 50))
        msgs.append(AIMessage(f"回答{i}:" + "喵" * 50))
    trimmed = trim_history(msgs, max_tokens=200)
    assert 0 < len(trimmed) < len(msgs)
    assert trimmed[0].type == "human"
    assert trimmed[-1] == msgs[-1]


def test_store_clear_empties_all_sessions():
    store = SessionStore()
    store.append("a", HumanMessage("hi"), AIMessage("hello"))
    store.clear()
    assert store.get("a") == []


def test_trim_noop_when_under_budget():
    msgs = [HumanMessage("hi"), AIMessage("hello")]
    assert trim_history(msgs, max_tokens=2000) == msgs


def test_trim_keeps_latest_message_when_it_alone_exceeds_budget():
    # 单条消息超预算时不能把整段历史裁空:当前问题(最后一条 human)必须保留
    msgs = [HumanMessage("hi"), AIMessage("hello"), HumanMessage("长" * 9000)]
    trimmed = trim_history(msgs, max_tokens=2000)
    assert trimmed == [msgs[-1]]


def _history(n_turns: int, start_id: int = 1) -> list:
    msgs, did = [], start_id
    for i in range(n_turns):
        msgs.append(HumanMessage(f"问题{i}", id=f"db-{did}")); did += 1
        msgs.append(AIMessage(f"回答{i}", id=f"db-{did}")); did += 1
    return msgs


def _turn_tokens(turn, s=None):
    return context_budget.estimate_messages_tokens(turn, s or context_budget.settings)


def test_split_layers_cuts_at_boundaries():
    h = _history(10)                       # ids 1..20
    layer2, layer1 = split_layers(h, summary_upto=6, layer1_from=14)
    assert [m.id for m in layer2] == [f"db-{i}" for i in range(7, 15)]
    assert [m.id for m in layer1] == [f"db-{i}" for i in range(15, 21)]


def test_split_layers_legacy_without_ids_keeps_all_in_layer1():
    h = [HumanMessage("旧问题"), AIMessage("答")] * 3
    layer2, layer1 = split_layers(h, 0, 0)
    assert layer2 == [] and layer1 == h


def test_degrade_layer1_moves_whole_turns():
    h = _history(10)
    turns = [h[i:i + 2] for i in range(0, len(h), 2)]
    budget = sum(_turn_tokens(t) for t in turns[-4:]) + 1
    new_from = degrade_layer1(h, 0, budget)
    assert new_from == 12                   # 丢第 1-6 轮,边界=第 6 轮 assistant id
    layer2, layer1 = split_layers(h, 0, new_from)
    assert layer1[0].id == "db-13"          # 层 1 从第 7 轮 user 开始


def test_degrade_layer1_keeps_newest_oversized_turn(caplog):
    h = [HumanMessage("超长" + "喵" * 5000, id="db-1"), AIMessage("答", id="db-2")]
    with caplog.at_level("WARNING", logger="app.core.memory"):
        assert degrade_layer1(h, 0, 10) == 0     # 只有一轮:宁可超预算也保留
    assert "layer1 单轮超预算" in caplog.text


def test_degrade_layer1_moves_forward_from_boundary():
    h = _history(4)                         # ids 1..8
    assert degrade_layer1(h, 4, 1) == 6     # 丢第 3 轮,新边界=id6;只向前


def test_render_layer2_truncates_assistant_and_collapses_tools():
    msgs = [
        HumanMessage("完整问题"),
        AIMessage("答" * 100),
        AIMessage("", tool_calls=[{"name": "query_order", "args": {}, "id": "c1"}]),
        ToolMessage("x" * 500, tool_call_id="c1", name="query_order"),
    ]
    out = render_layer2(msgs, assistant_chars=40, tool_keep_chars=120)
    assert out[0].content == "完整问题"                  # 用户原话不动
    assert out[1].content == "答" * 40 + "…"
    assert out[2].content.startswith("[调用工具:query_order")
    assert len(out) == 3                                  # tool 行被折叠,不单独出现


def test_trim_turns_to_budget_keeps_whole_turns():
    h = _history(3)
    budget = _turn_tokens(h[-2:]) + 1
    out = trim_turns_to_budget(h, budget)
    assert [m.content for m in out] == ["问题2", "回答2"]   # 整轮保留,不拆
    assert isinstance(out[0], HumanMessage)


def test_summary_line_flattens_multiline():
    assert summary_line(None) == "" and summary_line("") == ""
    assert summary_line("第一段\n第二段") == "(早前对话摘要:第一段 第二段)"


def test_build_layers_trims_layer2_to_budget():
    h = _history(6)                       # 层2 ids 1..8,层1 ids 9..12
    layer2, layer1 = build_layers(
        h, 0, 8, layer1_budget=10000, layer2_budget=1,
        assistant_chars=40, tool_keep_chars=120)
    assert len(layer2) == 2                # 只留最新整轮
    assert isinstance(layer2[0], HumanMessage)


def test_build_layers_legacy_falls_back_to_trim_history():
    h = [HumanMessage("旧" * 100), AIMessage("答" * 100)] * 5
    layer2, layer1 = build_layers(
        h, 0, 0, layer1_budget=50, layer2_budget=50,
        assistant_chars=40, tool_keep_chars=120)
    assert layer2 == [] and 0 < len(layer1) < len(h)

