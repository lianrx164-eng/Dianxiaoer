from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.message import add_messages

from app.graph.state import ConversationState, trace_of


def test_add_messages_reducer_appends():
    merged = add_messages([HumanMessage("hi")], [AIMessage("yo")])
    assert [m.content for m in merged] == ["hi", "yo"]


def test_trace_of_merges_without_mutating():
    old = {"intent": "订单"}
    out = trace_of({"trace": old}, route="business")
    assert out == {"intent": "订单", "route": "business"}
    assert old == {"intent": "订单"}


def test_state_has_required_keys():
    for k in ["messages", "user_id", "conversation_id", "intent", "route",
              "evidence", "citations", "evidence_strong", "answer",
              "steps", "tokens_used", "suggested_actions", "trace"]:
        assert k in ConversationState.__annotations__
