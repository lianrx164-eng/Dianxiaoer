import pytest

from app.core import intent


class _FakeModel:
    def __init__(self, result=None, exc=None):
        self._result = result
        self._exc = exc

    async def ainvoke(self, messages):
        if self._exc:
            raise self._exc
        return self._result


def _result(value: str):
    return type("R", (), {"intent": value})()


@pytest.mark.asyncio
async def test_classify_returns_schema_value(monkeypatch):
    monkeypatch.setattr(intent, "structured_model", lambda schema: _FakeModel(_result("物流")))
    assert await intent.classify("订单1001的快递到哪了") == "物流"


@pytest.mark.asyncio
async def test_classify_returns_unknown_on_error_so_routing_falls_back_to_agent(monkeypatch):
    monkeypatch.setattr(
        intent, "structured_model", lambda schema: _FakeModel(exc=RuntimeError("boom")))
    assert await intent.classify("订单1001到哪了") == "未知"
