import logging

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.config import Settings
from app.core import context_budget as cb


def _settings(**kw):
    base = dict(chat_model="m", chat_base_url="u", chat_api_key="k",
                embed_api_key="k", _env_file=None)
    base.update(kw)
    return Settings(**base)


def test_estimate_text_tokens_uses_chars_per_token():
    s = _settings()
    assert cb.estimate_text_tokens("", s) == 0
    assert cb.estimate_text_tokens("a" * 777, s) == 389
    assert cb.estimate_text_tokens("a" * 2161, s) == 1081


def test_budget_with_demo_config():
    """演示配置下的预算(ch08:工具清单变化后数值已重测,不再是 ch07 验收数字)。"""
    b = cb.compute_budget(_settings(
        model_context_window=18000, max_output_tokens=2000,
        max_user_input_tokens=2000, max_agent_steps=3,
        tool_result_max_tokens=1200, rerank_top_k=5))
    assert (b.window, b.fixed, b.react_peak, b.history, b.layer1, b.layer2) == (
        18000, 9063, 3600, 5337, 3734, 1602)


def test_only_window_set_yields_small_history_and_self_check_warns(caplog):
    s = _settings(model_context_window=18000)      # 步数/rerank 保持默认 6/10
    b = cb.compute_budget(s)
    assert (b.fixed, b.react_peak, b.history, b.layer1, b.layer2) == (
        11063, 7200, 0, 0, 0)
    with caplog.at_level(logging.ERROR, logger="app.core.context_budget"):
        cb.self_check(s)
    assert "上下文预算不足" in caplog.text


def test_default_budget():
    b = cb.compute_budget(_settings())
    assert (b.window, b.fixed, b.react_peak, b.history, b.layer1, b.layer2) == (
        64000, 11063, 7200, 30000, 20999, 9000)


def test_budget_constants_match_prompt_and_tools():
    """提示词或内置工具 schema 变了,这条先红;MCP 工具 schema 用单独的静态预留。"""
    import json

    from langchain_core.utils.function_calling import convert_to_openai_tool

    from app.core.prompts import AGENT_SYSTEM
    from app.tools.registry import builtin_specs

    s = _settings()
    tools_json = json.dumps(
        [convert_to_openai_tool(x.tool) for x in builtin_specs()], ensure_ascii=False)
    assert cb.estimate_text_tokens(AGENT_SYSTEM, s) == s.system_prompt_tokens
    assert cb.estimate_text_tokens(tools_json, s) == s.tool_schema_tokens
    assert s.mcp_tool_schema_tokens > 0           # 三台 MCP 工具的静态预留


def test_estimate_messages_tokens_counts_tool_calls():
    s = _settings()
    msgs = [
        HumanMessage("你好" * 10),
        AIMessage("", tool_calls=[{"name": "query_order",
                                   "args": {"order_id": "1001"}, "id": "c1"}]),
        ToolMessage("x" * 100, tool_call_id="c1", name="query_order"),
    ]
    assert cb.estimate_messages_tokens(msgs, s) > cb.estimate_text_tokens("你好" * 10, s)
