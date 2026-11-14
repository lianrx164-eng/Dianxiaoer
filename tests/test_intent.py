import pytest
from pydantic import ValidationError

from app.core import intent


def test_intent_schema_requires_confidence():
    """confidence 是 spec §6 的必填字段:模型漏输出应解析失败 → 其他/0.0,
    不能被 schema 默认值静默补成合规值。"""
    with pytest.raises(ValidationError):
        intent._Intent(intent="物流")


class _FakeModel:
    def __init__(self, result=None, exc=None):
        self._result = result
        self._exc = exc

    async def ainvoke(self, messages):
        if self._exc:
            raise self._exc
        return self._result


def _result(value: str, conf: float = 0.9):
    return type("R", (), {"intent": value, "confidence": conf})()


@pytest.mark.asyncio
async def test_classify_returns_intent_and_confidence(monkeypatch):
    monkeypatch.setattr(
        intent, "structured_model",
        lambda schema, model=None: _FakeModel(_result("物流", 0.88)))
    r = await intent.classify("订单1001的快递到哪了")
    assert r == {"intent": "物流", "confidence": 0.88}


@pytest.mark.asyncio
async def test_classify_falls_back_to_other_on_error(monkeypatch):
    monkeypatch.setattr(
        intent, "structured_model",
        lambda schema, model=None: _FakeModel(exc=RuntimeError("boom")))
    assert await intent.classify("订单1001到哪了") == {"intent": "其他", "confidence": 0.0}
