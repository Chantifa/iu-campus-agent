# IU Campus Agent - CLI agent image
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TERM=xterm-256color \
    DATA_DIR=/app/data \
    IU_DOCS_PATH=/data/iu \
    WORKSPACE_DIR=/workspace \
    FASTEMBED_CACHE_PATH=/app/data/fastembed \
    HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
    QDRANT_URL=http://qdrant:6333

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) dependencies only (cached as long as pyproject.toml does not change). The stub package and
#    the setuptools build directory are removed again so the real sources are packaged in step 2.
COPY pyproject.toml ./
RUN mkdir -p src/iu_agent && touch src/iu_agent/__init__.py && echo "" > README.md \
    && pip install . && pip uninstall -y iu-campus-agent \
    && rm -rf build src README.md *.egg-info

# 2) the application itself
COPY README.md ./
COPY src ./src
RUN pip install --no-deps . && rm -rf build *.egg-info \
    && iu-agent --version

# Optionally bake the embedding models into the image (build with --build-arg PRELOAD_EMBEDDINGS=true)
ARG PRELOAD_EMBEDDINGS=false
ARG EMBEDDING_MODEL=jinaai/jina-embeddings-v2-base-de
RUN if [ "$PRELOAD_EMBEDDINGS" = "true" ]; then \
      python -c "from fastembed import TextEmbedding, SparseTextEmbedding; \
TextEmbedding('$EMBEDDING_MODEL', cache_dir='/app/data/fastembed'); \
SparseTextEmbedding('Qdrant/bm25', cache_dir='/app/data/fastembed')"; \
    fi

RUN useradd --create-home --uid 1000 agent \
    && mkdir -p /app/data /workspace /data/iu \
    && chown -R agent:agent /app /workspace
USER agent

VOLUME ["/app/data", "/workspace"]
ENTRYPOINT ["iu-agent"]
CMD ["chat"]
