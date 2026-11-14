"""ch09 观测挂载:Phoenix 未配置时零副作用;启用时插桩一次并写 trace 属性。"""
from app.core import observability


def test_disabled_when_endpoint_missing(monkeypatch):
    monkeypatch.setattr("app.config.settings.phoenix_endpoint", "")
    assert observability.phoenix_enabled() is False


def test_attach_returns_graph_unchanged_when_disabled(monkeypatch):
    monkeypatch.setattr("app.config.settings.phoenix_endpoint", "")
    sentinel = object()
    assert observability.attach_observability(sentinel) is sentinel


def test_tag_intent_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(observability, "_enabled_impl", False)
    observability.tag_intent("商品咨询", 0.9)   # 不抛即可,不需要 Phoenix 在跑


def test_flush_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(observability, "_tracer_provider", None)
    observability.flush()


def test_attach_instruments_once_and_returns_graph(monkeypatch):
    monkeypatch.setattr("app.config.settings.phoenix_endpoint", "http://localhost:6006/v1/traces")
    monkeypatch.setattr(observability, "_enabled_impl", False)
    calls = []

    def fake_instrument():
        calls.append(1)
        observability._enabled_impl = True

    monkeypatch.setattr(observability, "_instrument", fake_instrument)
    graph = object()
    assert observability.attach_observability(graph) is graph
    assert observability.attach_observability(graph) is graph
    assert calls == [1]                       # 只挂一次


def test_tag_intent_sets_root_span_attributes(monkeypatch):
    class Span:
        attrs = {}

        def is_recording(self):
            return True

        def set_attribute(self, k, v):
            self.attrs[k] = v

    span = Span()
    monkeypatch.setattr(observability, "_enabled_impl", True)
    monkeypatch.setattr(observability, "_root_span", type("CV", (), {"get": staticmethod(lambda: span)})())
    observability.tag_intent("物流", 0.8)
    assert span.attrs == {"intent": "物流", "intent.confidence": 0.8}


def test_tag_intent_failure_is_swallowed(monkeypatch):
    class BadSpan:
        def is_recording(self):
            raise RuntimeError("Phoenix 挂了")

    monkeypatch.setattr(observability, "_enabled_impl", True)
    monkeypatch.setattr(observability, "_root_span",
                        type("CV", (), {"get": staticmethod(lambda: BadSpan())})())
    observability.tag_intent("物流", 0.8)       # 观测失败不许影响业务流


def test_flush_calls_provider_and_swallows(monkeypatch):
    class BadProvider:
        def force_flush(self):
            raise RuntimeError("boom")

    monkeypatch.setattr(observability, "_tracer_provider", BadProvider())
    observability.flush()                        # 不抛即可


async def test_turn_span_disabled_yields_none(monkeypatch):
    monkeypatch.setattr(observability, "_enabled_impl", False)
    async with observability.turn_span(7) as span:
        assert span is None
