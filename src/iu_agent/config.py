"""Runtime configuration.

Every value can be set through an environment variable of the same name in upper case
(``ANTHROPIC_API_KEY``, ``QDRANT_URL`` ...) or through a ``.env`` file in the working directory.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_docs_path() -> Path:
    return Path.home() / "OneDrive" / "IU"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------ LLM providers
    anthropic_api_key: SecretStr | None = None
    anthropic_auth_token: SecretStr | None = None
    anthropic_enabled: bool | None = Field(
        default=None,
        description="Force the Anthropic provider on/off, e.g. when credentials come from `ant auth login`.",
    )
    anthropic_model: str = "claude-opus-5"
    anthropic_effort: str | None = Field(default=None, description="low | medium | high | xhigh | max")

    moonshot_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("MOONSHOT_API_KEY", "KIMI_API_KEY")
    )
    moonshot_base_url: str = "https://api.moonshot.ai/v1"
    kimi_model: str = "kimi-k3"
    kimi_reasoning_effort: str | None = Field(default=None, description="low | high | max (Kimi K3)")

    ollama_base_url: str | None = None
    ollama_model: str = "qwen3:8b"
    ollama_num_ctx: int = 32768

    default_model: str | None = Field(
        default=None,
        description="Model to start with, e.g. 'anthropic:claude-opus-5' or 'kimi:kimi-k3'. "
        "If unset the CLI asks which model to use.",
    )
    max_output_tokens: int = 16000
    context_budget_tokens: int = 120_000
    llm_timeout_seconds: float = 600.0

    # ------------------------------------------------------------------ RAG / vector store
    qdrant_url: str | None = Field(
        default=None, description="http://qdrant:6333 in Docker; unset = embedded local mode under DATA_DIR"
    )
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "iu_course_material"
    embedding_model: str = "jinaai/jina-embeddings-v2-base-de"
    sparse_model: str = "Qdrant/bm25"
    hybrid_search: bool = True
    chunk_size: int = 1200
    chunk_overlap: int = 150
    retrieval_k: int = 6
    embedding_threads: int | None = None

    # ------------------------------------------------------------------ sources
    iu_docs_path: Path = Field(default_factory=_default_docs_path)
    iu_exclude_globs: str = Field(
        default="Bill/**,Certificate/**",
        description="Comma separated glob patterns (relative to IU_DOCS_PATH) that are never indexed.",
    )
    iu_include_globs: str = Field(
        default="", description="If set, only paths matching one of these comma separated globs are indexed."
    )
    max_file_mb: int = 200
    data_dir: Path = Path("data")
    workspace_dir: Path = Path(".")
    allow_outside_workspace: bool = False

    # ------------------------------------------------------------------ Moodle / myCampus
    moodle_url: str = "https://mycampus-classic.iu.org"
    moodle_token: SecretStr | None = None
    moodle_service: str = "moodle_mobile_app"
    moodle_sync_forums: str = Field(default="news", description="none | news | all")
    moodle_fetch_urls: bool = False

    # ------------------------------------------------------------------ derived helpers
    @property
    def fastembed_cache_dir(self) -> Path:
        env = os.environ.get("FASTEMBED_CACHE_PATH")
        return Path(env) if env else self.data_dir / "fastembed"

    @property
    def qdrant_local_path(self) -> Path:
        return self.data_dir / "qdrant"

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "manifest.json"

    @property
    def moodle_token_path(self) -> Path:
        return self.data_dir / "moodle_token.json"

    @property
    def moodle_cache_dir(self) -> Path:
        return self.data_dir / "moodle"

    @property
    def history_path(self) -> Path:
        return self.data_dir / "chat_history.txt"

    def exclude_patterns(self) -> list[str]:
        return [p.strip() for p in self.iu_exclude_globs.split(",") if p.strip()]

    def include_patterns(self) -> list[str]:
        return [p.strip() for p in self.iu_include_globs.split(",") if p.strip()]

    def prepare_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.fastembed_cache_dir.mkdir(parents=True, exist_ok=True)
        # fastembed downloads public models from the Hugging Face hub; an expired token stored on the
        # machine would only get in the way, so implicit tokens are disabled.
        os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
        os.environ.setdefault("FASTEMBED_CACHE_PATH", str(self.fastembed_cache_dir))


def load_settings(**overrides) -> Settings:
    settings = Settings(**overrides)
    settings.prepare_dirs()
    return settings
