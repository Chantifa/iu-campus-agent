"""Qdrant-backed vector store with hybrid (dense + BM25 sparse) retrieval.

Two deployment modes, chosen by ``QDRANT_URL``:

* unset  -> embedded local mode, data lives under ``DATA_DIR/qdrant`` (no server needed)
* set    -> a Qdrant server (docker-compose / Kubernetes service)
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models

from iu_agent.config import Settings
from iu_agent.rag.embeddings import FastEmbedEmbeddings

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"
PAYLOAD_INDEXES = ("metadata.source", "metadata.course", "metadata.origin", "metadata.course_id")


class CourseVectorStore:
    def __init__(
        self,
        settings: Settings,
        *,
        embeddings: Embeddings | None = None,
        sparse_embeddings: FastEmbedSparse | None = None,
        client: QdrantClient | None = None,
    ) -> None:
        self.settings = settings
        self.collection = settings.qdrant_collection
        self.hybrid = settings.hybrid_search
        self.client = client or self._make_client(settings)
        self.embeddings = embeddings or FastEmbedEmbeddings(
            settings.embedding_model,
            cache_dir=str(settings.fastembed_cache_dir),
            threads=settings.embedding_threads,
        )
        if sparse_embeddings is not None:
            self.sparse = sparse_embeddings
        elif self.hybrid:
            self.sparse = FastEmbedSparse(
                model_name=settings.sparse_model, cache_dir=str(settings.fastembed_cache_dir)
            )
        else:
            self.sparse = None
        self._vs: QdrantVectorStore | None = None

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _make_client(settings: Settings) -> QdrantClient:
        if settings.qdrant_url:
            api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
            return QdrantClient(url=settings.qdrant_url, api_key=api_key, timeout=120)
        settings.qdrant_local_path.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(settings.qdrant_local_path))

    @property
    def location(self) -> str:
        return self.settings.qdrant_url or f"local:{self.settings.qdrant_local_path}"

    def _dimension(self) -> int:
        if isinstance(self.embeddings, FastEmbedEmbeddings):
            return self.embeddings.dimension
        return len(self.embeddings.embed_query("dimension probe"))

    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            return
        vectors_config = {
            DENSE_VECTOR: models.VectorParams(size=self._dimension(), distance=models.Distance.COSINE),
        }
        sparse_config = (
            {SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)} if self.hybrid else None
        )
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_config,
        )
        if not self.settings.qdrant_url:
            return  # the embedded local mode has no payload indexes
        for field in PAYLOAD_INDEXES:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception:  # an index is an optimisation, never a hard requirement
                pass

    @property
    def vectorstore(self) -> QdrantVectorStore:
        if self._vs is None:
            self.ensure_collection()
            self._vs = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection,
                embedding=self.embeddings,
                sparse_embedding=self.sparse,
                retrieval_mode=RetrievalMode.HYBRID if self.hybrid else RetrievalMode.DENSE,
                vector_name=DENSE_VECTOR,
                sparse_vector_name=SPARSE_VECTOR,
            )
        return self._vs

    def reset(self) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self._vs = None

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ writes
    def add_documents(self, documents: Sequence[Document], ids: Sequence[str]) -> None:
        if not documents:
            return
        self.vectorstore.add_documents(list(documents), ids=list(ids), batch_size=32)

    def delete_source(self, source: str) -> None:
        if not self.client.collection_exists(self.collection):
            return
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="metadata.source", match=models.MatchValue(value=source))]
                )
            ),
        )

    # ------------------------------------------------------------------ reads
    def count(self) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        return int(self.client.count(collection_name=self.collection, exact=True).count)

    def search(
        self,
        query: str,
        k: int = 6,
        *,
        course: str | None = None,
        origin: str | None = None,
        source: str | None = None,
    ) -> list[tuple[Document, float]]:
        must: list[models.Condition] = []
        if course:
            must.append(models.FieldCondition(key="metadata.course", match=models.MatchValue(value=course)))
        if origin:
            must.append(models.FieldCondition(key="metadata.origin", match=models.MatchValue(value=origin)))
        if source:
            must.append(models.FieldCondition(key="metadata.source", match=models.MatchValue(value=source)))
        query_filter = models.Filter(must=must) if must else None
        return self.vectorstore.similarity_search_with_score(query, k=k, filter=query_filter)
