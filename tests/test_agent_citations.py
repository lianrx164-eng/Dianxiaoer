"""知识路:forced_rag 的编号证据经 SSE 事件送到前端。"""
import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.graph import nodes, runtime
from app.graph.build import build_graph


def _aval(value):
    async def _f(*a, **k):
        return value
    return _f


@pytest.fixture(autouse=True)
def _repo(monkeypatch):
    async def fake_create(uid):
        return 7

    async def fake_append(*a, **k):
        return 1

    monkeypatch.setattr(runtime.repository, "create_conversation", fake_create)
    monkeypatch.setattr(runtime.repository, "append_message", fake_append)


async def _events(monkeypatch):
    citations = [{"n": 1, "id": 1, "section_path": "运费政策",
                  "question": "运费怎么算", "answer": "满99包邮", "content_type": "faq"}]
    monkeypatch.setattr(nodes, "classify_intent", _aval(
        {"intent": "商品咨询", "route": "knowledge", "trace": {"route": "knowledge"}}))
    monkeypatch.setattr(nodes, "forced_rag", _aval(
        {"evidence_strong": True, "evidence": "[1] 运费怎么算: 满99包邮",
         "citations": citations,
         "trace": {"forced_rag": True, "evidence_top": 0.9}}))

    async def fake_agent(state, config=None):
        return {"messages": [AIMessage("满99包邮[1]")], "steps": 1, "tokens_used": 5}

    monkeypatch.setattr(nodes, "agent_llm", fake_agent)
    monkeypatch.setattr(runtime, "get_graph",
                        lambda: build_graph(checkpointer=InMemorySaver()))
    return [e async for e in runtime.stream_turn("u1", "邮费是多少", None)]


async def test_stream_emits_citations_for_knowledge_route(monkeypatch):
    events = await _events(monkeypatch)
    cites = [e for e in events if e["type"] == "citations"]
    assert cites and cites[0]["items"][0]["section_path"] == "运费政策"


async def test_answer_delta_carries_citation_marker(monkeypatch):
    events = await _events(monkeypatch)
    answer = "".join(e["text"] for e in events if e["type"] == "delta")
    assert answer == "满99包邮[1]"
