import math
import re
import zlib

import pytest
from langchain_core.embeddings import Embeddings

from iu_agent.config import Settings
from iu_agent.rag.store import CourseVectorStore


class HashEmbeddings(Embeddings):
    """Deterministic bag-of-words embedding so tests never download a model."""

    dim = 64

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in re.findall(r"\w+", text.lower()):
            vector[zlib.crc32(token.encode("utf-8")) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


@pytest.fixture
def settings(tmp_path, monkeypatch) -> Settings:
    for name in (
        "QDRANT_URL",
        "ANTHROPIC_API_KEY",
        "MOONSHOT_API_KEY",
        "KIMI_API_KEY",
        "SWISSAI_API_KEY",
        "HF_TOKEN",
        "HUGGINGFACE_TOKEN",
        "OLLAMA_BASE_URL",
        "MOODLE_TOKEN",
        "DEFAULT_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    docs = tmp_path / "docs"
    workspace = tmp_path / "ws"
    docs.mkdir()
    workspace.mkdir()
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        hybrid_search=False,
        iu_docs_path=docs,
        workspace_dir=workspace,
        qdrant_url=None,
        anthropic_enabled=False,
    )
    settings.prepare_dirs()
    return settings


@pytest.fixture
def store(settings) -> CourseVectorStore:
    vector_store = CourseVectorStore(settings, embeddings=HashEmbeddings())
    yield vector_store
    vector_store.close()
