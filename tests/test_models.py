import pytest

from iu_agent.config import Settings
from iu_agent.models import (
    ModelRef,
    ProviderNotConfigured,
    build_chat_model,
    configured_providers,
    default_model_ref,
    list_models,
    parse_model_ref,
)


def _settings(**overrides) -> Settings:
    values = {"anthropic_enabled": False, **overrides}
    return Settings(_env_file=None, **values)


def test_parse_model_ref_forms():
    settings = _settings(moonshot_api_key="k")
    assert parse_model_ref("anthropic:claude-opus-5", settings) == ModelRef("anthropic", "claude-opus-5")
    assert parse_model_ref("kimi", settings) == ModelRef("kimi", "kimi-k3")
    assert parse_model_ref("kimi:", settings) == ModelRef("kimi", "kimi-k3")
    assert parse_model_ref("claude-sonnet-5", settings) == ModelRef("anthropic", "claude-sonnet-5")
    assert parse_model_ref("kimi-k2.6", settings) == ModelRef("kimi", "kimi-k2.6")
    assert parse_model_ref("ollama:qwen3:8b", _settings(ollama_base_url="http://x")) == ModelRef(
        "ollama", "qwen3:8b"
    )
    with pytest.raises(ValueError):
        parse_model_ref("mystery-model", settings)


def test_configured_providers_and_defaults(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    assert configured_providers(_settings()) == []
    assert default_model_ref(_settings()) is None
    settings = _settings(moonshot_api_key="k")
    assert configured_providers(settings) == ["kimi"]
    assert default_model_ref(settings) == ModelRef("kimi", "kimi-k3")
    assert default_model_ref(_settings(moonshot_api_key="k", default_model="kimi:kimi-k2.6")) == ModelRef(
        "kimi", "kimi-k2.6"
    )
    assert configured_providers(_settings(anthropic_enabled=True, anthropic_api_key="a")) == ["anthropic"]


def test_static_model_list():
    models, warnings = list_models(_settings(moonshot_api_key="k"))
    assert warnings == []
    assert [m.ref.name for m in models][:2] == ["kimi-k3", "kimi-k2.7-code"]


def test_build_chat_model_kimi_and_missing_provider():
    settings = _settings(moonshot_api_key="k", kimi_reasoning_effort="low")
    model = build_chat_model(ModelRef("kimi", "kimi-k3"), settings)
    assert model.model_name == "kimi-k3"
    assert model.reasoning_effort == "low"
    with pytest.raises(ProviderNotConfigured):
        build_chat_model(ModelRef("anthropic", "claude-opus-5"), settings)
