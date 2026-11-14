"""ch06 红线冒烟:interrupt / Command(resume) / astream 在当前 langgraph 上的中断 surface 形状。

只用一个纯 interrupt 节点,不打上游。用法:
    PYTHONUTF8=1 uv run python -m scripts.smoke_interrupt

结论供 fetch_order / runtime 的断言直接引用:
- A: ainvoke 返回 dict 里的 "__interrupt__" 是 list[Interrupt],取 [0].value;
- C: astream(stream_mode=["messages","updates"]) 中断出现在 updates chunk 的 "__interrupt__" 键;
- C': aget_state(config).interrupts / .next 可探未决中断;
- D: 暂停中传新输入会被接受,但会再次触发同一中断,新消息不处理。
"""
import asyncio
from typing import Annotated

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict


class S(TypedDict):
    messages: Annotated[list, add_messages]
    picked: str


async def ask_order(state: S):
    picked = interrupt({"type": "select_order",
                        "orders": [{"order_id": "1001"}, {"order_id": "2002"}]})
    return {"picked": picked}


async def confirm(state: S):
    return {"messages": [("ai", f"已选订单 {state['picked']}")]}


def _build(checkpointer):
    b = StateGraph(S)
    b.add_node("ask_order", ask_order)
    b.add_node("confirm", confirm)
    b.add_edge(START, "ask_order")
    b.add_edge("ask_order", "confirm")
    b.add_edge("confirm", END)
    return b.compile(checkpointer=checkpointer)


async def main():
    async with AsyncSqliteSaver.from_conn_string(":memory:") as cp:
        await cp.setup()
        graph = _build(cp)

        # A) 非流式首跑:中断 surface
        config = {"configurable": {"thread_id": "smoke-int-1"}}
        out = await graph.ainvoke({"messages": [("human", "我要退款")]}, config)
        print("A ainvoke keys=", list(out.keys()))
        print("A __interrupt__=", out.get("__interrupt__"))
        print("A value=", out["__interrupt__"][0].value)

        # B) 非流式 resume
        out2 = await graph.ainvoke(Command(resume="1001"), config)
        print("B picked=", out2.get("picked"),
              "msgs=", [m.content for m in out2["messages"]])

        # C) 流式:中断出现在哪种 chunk
        config2 = {"configurable": {"thread_id": "smoke-int-2"}}
        print("C astream chunks:")
        async for mode, chunk in graph.astream(
                {"messages": [("human", "我要退款")]}, config2,
                stream_mode=["messages", "updates"]):
            if mode == "updates":
                print("  UPD keys=", list(chunk.keys()), "val=", chunk)

        # C') 流式后探 pending
        snap = await graph.aget_state(config2)
        print("C' next=", snap.next, "interrupts=", getattr(snap, "interrupts", None))

        # C'') 流式 resume
        print("C'' astream resume:")
        async for mode, chunk in graph.astream(
                Command(resume="2002"), config2,
                stream_mode=["messages", "updates"]):
            print("  ", mode, repr(chunk)[:240])

        # D) 暂停中传新输入(不是 resume)
        config3 = {"configurable": {"thread_id": "smoke-int-3"}}
        await graph.ainvoke({"messages": [("human", "我要退款")]}, config3)
        snap3 = await graph.aget_state(config3)
        print("D pending before new input: next=", snap3.next)
        out3 = await graph.ainvoke(
            {"messages": [("human", "算了不退了,查物流")]}, config3)
        print("D new-input keys=", list(out3.keys()),
              "picked=", out3.get("picked"))
        print("D messages=", [m.content for m in out3["messages"]])
        print("D 结论:新输入被接受并再次触发中断;新消息不处理(runtime 需显式检测未决中断)")


if __name__ == "__main__":
    asyncio.run(main())
