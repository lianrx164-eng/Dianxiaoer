import pytest
from pydantic import ValidationError

from app.config import Settings


def test_账号相关的三项必填(monkeypatch):
    # 这三项跟着账号走,给默认值等于替人猜:猜错不在启动时报错,而是第一次调上游时
    # 抛一个看不懂的 401
    for k in ("CHAT_MODEL", "CHAT_BASE_URL", "CHAT_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_settings_env_override(monkeypatch):
    for k, v in {"CHAT_MODEL": "m", "CHAT_BASE_URL": "u", "CHAT_API_KEY": "k"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("MAX_AGENT_STEPS", "4")
    assert Settings(_env_file=None).max_agent_steps == 4


def test_extract_thinking_defaults_to_none(monkeypatch):
    # 默认不发送上游专有的 thinking 参数,换上游不用动代码
    for k, v in {"CHAT_MODEL": "m", "CHAT_BASE_URL": "u", "CHAT_API_KEY": "k"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("EXTRACT_THINKING", raising=False)
    assert Settings(_env_file=None).extract_thinking is None


def test_extract_thinking_env_override(monkeypatch):
    for k, v in {"CHAT_MODEL": "m", "CHAT_BASE_URL": "u", "CHAT_API_KEY": "k"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("EXTRACT_THINKING", "disabled")
    assert Settings(_env_file=None).extract_thinking == "disabled"


def test_ch07_budget_env_override(monkeypatch):
    for k, v in {"CHAT_MODEL": "m", "CHAT_BASE_URL": "u", "CHAT_API_KEY": "k"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("MODEL_CONTEXT_WINDOW", "18000")
    monkeypatch.setenv("MAX_AGENT_STEPS", "3")
    monkeypatch.setenv("RERANK_TOP_K", "5")
    s = Settings(_env_file=None)
    assert s.model_context_window == 18000
    assert s.max_agent_steps == 3 and s.rerank_top_k == 5
