"""LLM provider registry: Anthropic Claude, Moonshot Kimi and (optionally) a local Ollama server.

A model is addressed as ``<provider>:<name>`` (for example ``anthropic:claude-opus-5`` or
``kimi:kimi-k3``). Bare names are resolved by prefix (``claude-*`` -> anthropic, ``kimi-*`` -> kimi).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import httpx
from langchain_core.language_models import BaseChatModel

from iu_agent.config import Settings

PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_KIMI = "kimi"
PROVIDER_OLLAMA = "ollama"

PROVIDER_LABELS = {
    PROVIDER_ANTHROPIC: "Anthropic Claude",
    PROVIDER_KIMI: "Moonshot Kimi",
    PROVIDER_OLLAMA: "Ollama (local)",
}

PROVIDER_HINTS = {
    PROVIDER_ANTHROPIC: "set ANTHROPIC_API_KEY (console.anthropic.com) or run `ant auth login`",
    PROVIDER_KIMI: "set MOONSHOT_API_KEY (platform.kimi.ai)",
    PROVIDER_OLLAMA: "set OLLAMA_BASE_URL, e.g. http://localhost:11434",
}

# Recommended models per provider (first entry is the provider default). Live lists are fetched
# from the provider APIs on request, see ``list_models``.
STATIC_MODELS: dict[str, list[tuple[str, str]]] = {
    PROVIDER_ANTHROPIC: [
        ("claude-opus-5", "flagship, 1M context, adaptive thinking"),
        ("claude-sonnet-5", "fast and cheaper, 1M context"),
        ("claude-haiku-4-5", "fastest and cheapest, 200K context"),
        ("claude-fable-5-1", "most capable, premium pricing"),
        ("claude-opus-4-8", "previous Opus generation"),
        ("claude-sonnet-4-6", "previous Sonnet generation"),
    ],
    PROVIDER_KIMI: [
        ("kimi-k3", "flagship, 1M context, thinking always on"),
        ("kimi-k2.7-code", "coding model, 256K context"),
        ("kimi-k2.7-code-highspeed", "coding model, high output speed"),
        ("kimi-k2.6", "general model, 256K context"),
    ],
    PROVIDER_OLLAMA: [],
}

# Models where adaptive thinking must be requested explicitly (newer models run it by default).
_EXPLICIT_ADAPTIVE_THINKING = {"claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6", "claude-sonnet-4-6"}


@dataclass(frozen=True)
class ModelRef:
    provider: str
    name: str

    @property
    def id(self) -> str:
        return f"{self.provider}:{self.name}"

    def __str__(self) -> str:
        return self.id


@dataclass(frozen=True)
class ModelInfo:
    ref: ModelRef
    note: str = ""
    source: str = "static"  # "static" | "live"


class ProviderNotConfigured(RuntimeError):
    pass


def anthropic_profile_dir() -> Path:
    """Directory where the Anthropic CLI (`ant auth login`) stores OAuth profiles."""
    return Path.home() / ".config" / "anthropic"


def provider_configured(provider: str, settings: Settings) -> bool:
    if provider == PROVIDER_ANTHROPIC:
        if settings.anthropic_enabled is not None:
            return settings.anthropic_enabled
        return bool(
            settings.anthropic_api_key or settings.anthropic_auth_token or anthropic_profile_dir().exists()
        )
    if provider == PROVIDER_KIMI:
        return settings.moonshot_api_key is not None
    if provider == PROVIDER_OLLAMA:
        return bool(settings.ollama_base_url)
    return False


def configured_providers(settings: Settings) -> list[str]:
    return [
        p for p in (PROVIDER_ANTHROPIC, PROVIDER_KIMI, PROVIDER_OLLAMA) if provider_configured(p, settings)
    ]


def provider_default_model(provider: str, settings: Settings) -> str:
    return {
        PROVIDER_ANTHROPIC: settings.anthropic_model,
        PROVIDER_KIMI: settings.kimi_model,
        PROVIDER_OLLAMA: settings.ollama_model,
    }[provider]


def parse_model_ref(text: str, settings: Settings) -> ModelRef:
    """Parse ``provider:name``, a bare provider name or a bare model name."""
    text = text.strip()
    head, sep, tail = text.partition(":")
    if sep and head.lower() in PROVIDER_LABELS:
        provider = head.lower()
        return ModelRef(provider, tail or provider_default_model(provider, settings))
    lowered = text.lower()
    if lowered in PROVIDER_LABELS:
        return ModelRef(lowered, provider_default_model(lowered, settings))
    if lowered.startswith("claude"):
        return ModelRef(PROVIDER_ANTHROPIC, text)
    if lowered.startswith(("kimi", "moonshot")):
        return ModelRef(PROVIDER_KIMI, text)
    if provider_configured(PROVIDER_OLLAMA, settings):
        return ModelRef(PROVIDER_OLLAMA, text)
    raise ValueError(
        f"Cannot infer the provider for model '{text}'. Use the form provider:model, e.g. "
        "anthropic:claude-opus-5, kimi:kimi-k3 or ollama:qwen3:8b."
    )


def default_model_ref(settings: Settings) -> ModelRef | None:
    if settings.default_model:
        return parse_model_ref(settings.default_model, settings)
    providers = configured_providers(settings)
    if not providers:
        return None
    return ModelRef(providers[0], provider_default_model(providers[0], settings))


# ----------------------------------------------------------------------------- live model lists
def _live_anthropic_models(settings: Settings) -> list[str]:
    import anthropic

    kwargs: dict = {}
    if settings.anthropic_api_key:
        kwargs["api_key"] = settings.anthropic_api_key.get_secret_value()
    if settings.anthropic_auth_token:
        kwargs["auth_token"] = settings.anthropic_auth_token.get_secret_value()
    client = anthropic.Anthropic(timeout=15.0, max_retries=1, **kwargs)
    return [m.id for m in client.models.list()]


def _live_kimi_models(settings: Settings) -> list[str]:
    assert settings.moonshot_api_key is not None
    response = httpx.get(
        f"{settings.moonshot_base_url.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {settings.moonshot_api_key.get_secret_value()}"},
        timeout=15.0,
    )
    response.raise_for_status()
    return [m["id"] for m in response.json().get("data", [])]


def _live_ollama_models(settings: Settings) -> list[str]:
    assert settings.ollama_base_url
    response = httpx.get(f"{settings.ollama_base_url.rstrip('/')}/api/tags", timeout=10.0)
    response.raise_for_status()
    return [m["name"] for m in response.json().get("models", [])]


_LIVE_FETCHERS = {
    PROVIDER_ANTHROPIC: _live_anthropic_models,
    PROVIDER_KIMI: _live_kimi_models,
    PROVIDER_OLLAMA: _live_ollama_models,
}


def list_models(
    settings: Settings, providers: list[str] | None = None, live: bool = False
) -> tuple[list[ModelInfo], list[str]]:
    """Return ``(models, warnings)``.

    The recommended (static) models of every configured provider are listed first. With
    ``live=True`` the provider APIs are queried as well and every additional model they report is
    appended; providers whose live list cannot be fetched produce a warning instead of an error.
    """
    providers = providers if providers is not None else configured_providers(settings)
    result: list[ModelInfo] = []
    warnings: list[str] = []
    for provider in providers:
        live_names: list[str] | None = None
        if live:
            try:
                live_names = _LIVE_FETCHERS[provider](settings)
            except Exception as exc:  # network or auth problems must not break the CLI
                warnings.append(f"{PROVIDER_LABELS[provider]}: live model list unavailable ({exc})")
        seen: set[str] = set()
        for name, note in STATIC_MODELS.get(provider, []):
            if live_names is not None and name not in live_names:
                note = f"{note} (not reported by the API)"
            result.append(
                ModelInfo(ModelRef(provider, name), note=note, source="live" if live_names else "static")
            )
            seen.add(name)
        for name in live_names or []:
            if name not in seen:
                result.append(ModelInfo(ModelRef(provider, name), source="live"))
                seen.add(name)
        if provider == PROVIDER_OLLAMA and settings.ollama_model not in seen:
            result.append(ModelInfo(ModelRef(provider, settings.ollama_model), note="configured default"))
    return result, warnings


# ----------------------------------------------------------------------------- chat model factory
def build_chat_model(ref: ModelRef, settings: Settings) -> BaseChatModel:
    if not provider_configured(ref.provider, settings):
        raise ProviderNotConfigured(
            f"Provider '{ref.provider}' is not configured: {PROVIDER_HINTS.get(ref.provider, '')}."
        )

    if ref.provider == PROVIDER_ANTHROPIC:
        from langchain_anthropic import ChatAnthropic

        kwargs: dict = {
            "model": ref.name,
            "max_tokens": settings.max_output_tokens,
            "timeout": settings.llm_timeout_seconds,
            "max_retries": 2,
        }
        if settings.anthropic_api_key:
            kwargs["api_key"] = settings.anthropic_api_key.get_secret_value()
        if ref.name in _EXPLICIT_ADAPTIVE_THINKING:
            kwargs["thinking"] = {"type": "adaptive"}
        if settings.anthropic_effort:
            kwargs["model_kwargs"] = {"output_config": {"effort": settings.anthropic_effort}}
        return ChatAnthropic(**kwargs)

    if ref.provider == PROVIDER_KIMI:
        from langchain_openai import ChatOpenAI

        assert settings.moonshot_api_key is not None
        kwargs = {
            "model": ref.name,
            "base_url": settings.moonshot_base_url,
            "api_key": settings.moonshot_api_key.get_secret_value(),
            "max_tokens": settings.max_output_tokens,
            "timeout": settings.llm_timeout_seconds,
            "max_retries": 2,
        }
        if settings.kimi_reasoning_effort:
            kwargs["reasoning_effort"] = settings.kimi_reasoning_effort
        return ChatOpenAI(**kwargs)

    if ref.provider == PROVIDER_OLLAMA:
        from langchain_ollama import ChatOllama

        return ChatOllama(model=ref.name, base_url=settings.ollama_base_url, num_ctx=settings.ollama_num_ctx)

    raise ValueError(f"Unknown provider {ref.provider}")
