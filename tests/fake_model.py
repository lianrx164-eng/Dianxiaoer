"""测试用假模型:turn1 返回 scripted[0](可带 tool_calls),非流式收敛返回后续脚本;
astream 按 stream_tokens 逐 token 吐;bind_calls 记录 bind_tools 次数(收敛不 bind 时应为 1)。"""
from langchain_core.messages import AIMessageChunk


class FakeModel:
    def __init__(self, scripted, stream_tokens=None):
        self._scripted = list(scripted)
        self._stream_tokens = list(stream_tokens or [])
        self.bind_calls = 0
        self.invoke_messages = []
        self.astream_messages = []

    def bind_tools(self, tools):
        self.bind_calls += 1
        return self

    async def ainvoke(self, messages):
        self.invoke_messages.append(list(messages))
        return self._scripted.pop(0)

    async def astream(self, messages):
        self.astream_messages.append(list(messages))
        for t in self._stream_tokens:
            yield AIMessageChunk(content=t)
