from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

import pytest

from app.graph import nodes
from app.tools.infra import ToolRun


def test_agent_messages_injects_evidence_on_knowledge():
    msgs = nodes._agent_messages(
        {"route": "knowledge", "evidence": "[1] 退货: 7天",
         "messages": [HumanMessage("能退吗")]})
    assert isinstance(msgs[0], SystemMessage)
    assert "[1] 退货: 7天" in msgs[0].content
    assert "query_faq" in msgs[0].content  # 指示别再检索


def test_agent_messages_no_evidence_on_business():
    msgs = nodes._agent_messages({"route": "business", "messages": [HumanMessage("订单1001")]})
    assert isinstance(msgs[0], SystemMessage)
    assert "已检索到的知识证据" not in msgs[0].content


def test_agent_messages_trims_long_history(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "token_budget", 50)
    old = [HumanMessage("旧问题"), AIMessage("长回答" * 5000)]
    msgs = nodes._agent_messages(
        {"route": "business", "messages": [*old, HumanMessage("新问题")]})
    assert msgs[0].type == "system"
    assert msgs[-1].content == "新问题"
    assert all("旧问题" != m.content for m in msgs[1:])


@pytest.mark.asyncio
async def test_agent_llm_accumulates_steps_and_tokens(monkeypatch):
    ai = AIMessage("好的", usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})

    class FakeModel:
        def bind_tools(self, tools):
            return self

        async def ainvoke(self, msgs, config=None):
            return ai

    monkeypatch.setattr(nodes, "get_chat_model", lambda **k: FakeModel())
    out = await nodes.agent_llm({"messages": [HumanMessage("hi")], "steps": 1, "tokens_used": 100})
    assert out["steps"] == 2
    assert out["tokens_used"] == 115
    assert out["messages"][0] is ai


@pytest.mark.asyncio
async def test_agent_tools_executes_normal_tool(monkeypatch):
    async def fake_exec(tc, cid):
        return ToolRun(tool_call_id=tc["id"], name=tc["name"], ok=True,
                       tool_message=ToolMessage(content='{"status":"已发货"}',
                                                tool_call_id=tc["id"], name=tc["name"]))

    monkeypatch.setattr(nodes, "execute_tool_call", fake_exec)
    ai = AIMessage("", tool_calls=[{"name": "query_logistics",
                                    "args": {"tracking_no": "SF1"}, "id": "t1"}])
    out = await nodes.agent_tools({"messages": [ai], "conversation_id": 5})
    assert out["messages"][0].name == "query_logistics"
    assert not out.get("suggested_actions")


@pytest.mark.asyncio
async def test_agent_tools_intercepts_create_ticket(monkeypatch):
    async def fake_exec(tc, cid):
        raise AssertionError("create_ticket 不应被执行(应拦截为提议)")

    monkeypatch.setattr(nodes, "execute_tool_call", fake_exec)
    ai = AIMessage("", tool_calls=[{"name": "create_ticket",
                                    "args": {"description": "屏幕碎了", "ticket_type": "售后"},
                                    "id": "t9"}])
    out = await nodes.agent_tools({"messages": [ai], "conversation_id": 5})
    act = out["suggested_actions"][0]
    assert act["type"] == "create_ticket"
    assert act["draft"]["description"] == "屏幕碎了"
    assert out["messages"][0].tool_call_id == "t9"
    # 拦截只产生选项,必须让模型明确「尚未创建」,不能声称已提交
    assert "尚未创建" in out["messages"][0].content
    assert "不要声称" in out["messages"][0].content
