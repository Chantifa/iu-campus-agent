"""Dense embeddings computed locally with fastembed (ONNX runtime, CPU friendly, no API key)."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings


class FastEmbedEmbeddings(Embeddings):
    """LangChain ``Embeddings`` adapter around ``fastembed.TextEmbedding``.

    The model is downloaded on first use into ``cache_dir`` (``FASTEMBED_CACHE_PATH``).
    """

    def __init__(
        self,
        model_name: str,
        cache_dir: str | None = None,
        threads: int | None = None,
        batch_size: int = 16,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.threads = threads
        self.batch_size = batch_size
        self._impl = None
        self._dimension: int | None = None

    @staticmethod
    def supported_models() -> dict[str, int]:
        """Map of fastembed model name -> vector dimension."""
        from fastembed import TextEmbedding

        out: dict[str, int] = {}
        for entry in TextEmbedding.list_supported_models():
            if isinstance(entry, dict):
                name, dim = entry.get("model"), entry.get("dim")
            else:  # DenseModelDescription dataclass in newer fastembed versions
                name, dim = getattr(entry, "model", None), getattr(entry, "dim", None)
            if name:
                out[str(name)] = int(dim or 0)
        return out

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            models = self.supported_models()
            if self.model_name not in models:
                known = ", ".join(sorted(models))
                raise ValueError(f"'{self.model_name}' is not a fastembed model. Known models: {known}")
            self._dimension = models[self.model_name]
        return self._dimension

    def _model(self):
        if self._impl is None:
            from fastembed import TextEmbedding

            self._impl = TextEmbedding(
                model_name=self.model_name, cache_dir=self.cache_dir, threads=self.threads
            )
        return self._impl

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model().embed(texts, batch_size=self.batch_size)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model().query_embed(text))).tolist()
