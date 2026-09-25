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
    provider_supports_tools,
)


def _settings(**overrides) -> Settings:
    values = {"anthropic_enabled": False, **overrides}
    return Settings(_env_file=None, **values)


def test_parse_model_ref_forms():
    settings = _settings(moonshot_api_key="k", swissai_api_key="h")
    assert parse_model_ref("anthropic:claude-opus-5", settings) == ModelRef("anthropic", "claude-opus-5")
    assert parse_model_ref("kimi", settings) == ModelRef("kimi", "kimi-k3")
    assert parse_model_ref("kimi:", settings) == ModelRef("kimi", "kimi-k3")
    assert parse_model_ref("kimi-k2.6", settings) == ModelRef("kimi", "kimi-k2.6")
    assert parse_model_ref("swissai", settings) == ModelRef("swissai", "swiss-ai/Apertus-v1.5-70B")
    assert parse_model_ref("swissai:swiss-ai/Apertus-8B-Instruct-2509", settings) == ModelRef(
        "swissai", "swiss-ai/Apertus-8B-Instruct-2509"
    )
    assert parse_model_ref("swiss-ai/Apertus-70B-Instruct-2509", settings) == ModelRef(
        "swissai", "swiss-ai/Apertus-70B-Instruct-2509"
    )
    assert parse_model_ref("apertus-8b", settings) == ModelRef("swissai", "apertus-8b")
    assert parse_model_ref("claude-sonnet-5", settings) == ModelRef("anthropic", "claude-sonnet-5")
    assert parse_model_ref("ollama:qwen3:8b", _settings(ollama_base_url="http://x")) == ModelRef(
        "ollama", "qwen3:8b"
    )
    with pytest.raises(ValueError):
        parse_model_ref("mystery-model", settings)


def test_configured_providers_and_defaults(monkeypatch):
    for name in ("ANTHROPIC_API_KEY", "MOONSHOT_API_KEY", "KIMI_API_KEY", "HF_TOKEN", "SWISSAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert configured_providers(_settings()) == []
    assert default_model_ref(_settings()) is None
    assert configured_providers(_settings(moonshot_api_key="k")) == ["kimi"]
    assert default_model_ref(_settings(moonshot_api_key="k")) == ModelRef("kimi", "kimi-k3")
    both = _settings(moonshot_api_key="k", swissai_api_key="h")
    assert configured_providers(both) == ["swissai", "kimi"]
    assert default_model_ref(both) == ModelRef("swissai", "swiss-ai/Apertus-v1.5-70B")
    assert default_model_ref(_settings(moonshot_api_key="k", default_model="kimi:kimi-k2.6")) == ModelRef(
        "kimi", "kimi-k2.6"
    )
    assert configured_providers(_settings(anthropic_enabled=True, anthropic_api_key="a")) == ["anthropic"]


def test_hf_token_env_configures_swissai(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    settings = Settings(_env_file=None, anthropic_enabled=False)
    assert "swissai" in configured_providers(settings)
    assert settings.swissai_api_key.get_secret_value() == "hf_test"


def test_static_model_lists_and_custom_endpoint():
    models, warnings = list_models(_settings(moonshot_api_key="k", swissai_api_key="h"))
    assert warnings == []
    names = [m.ref.name for m in models]
    assert names[:2] == ["swiss-ai/Apertus-v1.5-70B", "swiss-ai/Apertus-70B-Instruct-2509"]
    assert "kimi-k3" in names

    custom = _settings(
        swissai_api_key="k",
        swissai_base_url="https://api.publicai.co/v1",
        swissai_model="swiss-ai/apertus-v1.5-70b",
    )
    models, _ = list_models(custom)
    assert [m.ref.name for m in models] == ["swiss-ai/apertus-v1.5-70b"]


def test_build_chat_model_swissai():
    settings = _settings(swissai_api_key="hf_k", swissai_provider="publicai", swissai_max_output_tokens=1234)
    model = build_chat_model(ModelRef("swissai", "swiss-ai/Apertus-v1.5-70B"), settings)
    assert model.model_name == "swiss-ai/Apertus-v1.5-70B:publicai"
    assert str(model.openai_api_base).rstrip("/") == "https://router.huggingface.co/v1"
    assert model.extra_body == {"max_tokens": 1234}
    assert model.default_headers["User-Agent"].startswith("iu-campus-agent/")
    assert provider_supports_tools(ModelRef("swissai", "x"), settings) is True
    no_tools = _settings(swissai_api_key="k", swissai_tools=False)
    assert provider_supports_tools(ModelRef("swissai", "x"), no_tools) is False

    # provider pinning only applies to the Hugging Face router
    direct = _settings(
        swissai_api_key="k", swissai_provider="publicai", swissai_base_url="https://api.publicai.co/v1"
    )
    built = build_chat_model(ModelRef("swissai", "swiss-ai/apertus-v1.5-70b"), direct)
    assert built.model_name == "swiss-ai/apertus-v1.5-70b"


def test_build_chat_model_kimi_and_missing_provider():
    settings = _settings(moonshot_api_key="k", kimi_reasoning_effort="low")
    model = build_chat_model(ModelRef("kimi", "kimi-k3"), settings)
    assert model.model_name == "kimi-k3"
    assert model.reasoning_effort == "low"
    assert provider_supports_tools(ModelRef("kimi", "kimi-k3"), settings) is True
    with pytest.raises(ProviderNotConfigured):
        build_chat_model(ModelRef("anthropic", "claude-opus-5"), settings)
    with pytest.raises(ProviderNotConfigured):
        build_chat_model(ModelRef("swissai", "swiss-ai/Apertus-v1.5-70B"), settings)
