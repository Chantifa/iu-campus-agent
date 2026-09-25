# IU Campus Agent

A **Claude-Code-style software agent for the terminal**, written in Python with
**LangChain + LangGraph**, that knows your **IU (Internationale Hochschule) course material**.

* Chat with an agent that can search your course books, slides, exercises and myCampus pages
  (retrieval-augmented generation over a **Qdrant** vector database), read documents, edit files in
  a workspace, run shell commands (with approval) and fetch web pages.
* Course content comes from **myCampus classic (Moodle) through its web-service API** and/or from
  the local **OneDrive/IU** folder (PDF, DOCX, PPTX, XLSX, HTML, notebooks, Markdown ...).
* Choose the LLM at start-up or at any time with `/model`: **Claude** (Anthropic API),
  **Swiss AI Apertus** (open models by ETH Zurich / EPFL, via the Hugging Face inference router or
  Public AI), **Kimi** (Moonshot API) or a local **Ollama** model.
* Runs locally, in **Docker Compose** or on **Kubernetes**.

```
❯ Was sagt das Skript zu Eigenwerten und wie berechne ich sie?
⚙ search_course_material(query=Eigenwerte berechnen, course=Advanced Maths)
  ↳ ### Result 1 (score 0.842) - course: Advanced Maths | file: DLMDSAM01-01_Session3.pdf | p. 14 (+52 lines)
Eigenwerte λ einer quadratischen Matrix A sind die Lösungen der charakteristischen Gleichung
det(A − λI) = 0 ... (DLMDSAM01-01_Session3.pdf, p. 14)
```

---

## Contents

1. [How it works](#how-it-works)
2. [Requirements](#requirements)
3. [Quick start (local)](#quick-start-local)
4. [Choosing the model](#choosing-the-model)
5. [Indexing the OneDrive/IU folder](#indexing-the-onedriveiu-folder)
6. [myCampus (Moodle) integration](#mycampus-moodle-integration)
7. [Using the agent](#using-the-agent)
8. [Docker Compose](#docker-compose)
9. [Kubernetes](#kubernetes)
10. [Configuration reference](#configuration-reference)
11. [Development](#development)
12. [Limitations and notes](#limitations-and-notes)

---

## How it works

```
                 ┌──────────────────────────── CLI (rich + prompt_toolkit) ────────────────────────────┐
                 │  streaming Markdown, slash commands, model picker, approval prompts for write/shell │
                 └───────────────────────────────────────┬─────────────────────────────────────────────┘
                                                         │
                          ┌──────────────────────────────▼──────────────────────────────┐
                          │              LangGraph StateGraph (ReAct loop)              │
                          │   agent node (LLM + bound tools)  ⇄  ToolNode (tools)       │
                          │   InMemorySaver checkpointer, interrupt() for approvals      │
                          └──────┬──────────────────┬───────────────────┬───────────────┘
                                 │                  │                   │
              ┌──────────────────▼───┐    ┌─────────▼─────────┐   ┌─────▼──────────────────────┐
              │ Claude/Apertus/Kimi   │    │ RAG tools          │   │ software tools             │
              │ (langchain-anthropic, │    │ search_course_...  │   │ read_file, write_file,     │
              │  langchain-openai,    │    │ list_courses,      │   │ list_directory, run_shell, │
              │  langchain-ollama)    │    │ read_document      │   │ fetch_url, moodle_*        │
              └──────────────────────┘    └─────────┬─────────┘   └────────────────────────────┘
                                                    │
                     ┌──────────────────────────────▼───────────────────────────────┐
                     │ Qdrant collection (hybrid: dense Jina-de vectors + BM25)      │
                     │ embedded local mode  or  qdrant server (docker / k8s)         │
                     └──────────────────────────────▲───────────────────────────────┘
                                                    │ incremental ingestion (manifest.json)
                    ┌───────────────────────────────┴────────────────────────────────┐
                    │ OneDrive/IU folder (Semester_N/<Course>/…)                     │
                    │ myCampus Moodle web services (files, pages, books, assignments,│
                    │ announcements) via `iu-agent moodle sync`                      │
                    └────────────────────────────────────────────────────────────────┘
```

* **Agent loop**: an explicit `StateGraph` (`src/iu_agent/agent/graph.py`) with an `agent` node
  and a `ToolNode`. The model is swapped at runtime while the conversation stays in the
  checkpointer. History is trimmed to `CONTEXT_BUDGET_TOKENS`.
* **Approvals**: `write_file` and `run_shell` call LangGraph's `interrupt()`; the CLI shows the
  diff / command and resumes the graph with your decision (`y`, `n` or `a` = always this session).
* **Retrieval**: files are split into ~1200-character chunks with a context header
  (`[Course: … | File: … | Page 12]`), embedded locally with fastembed
  (`jinaai/jina-embeddings-v2-base-de`, German + English, no API key) and stored in Qdrant together
  with BM25 sparse vectors for hybrid search. Course codes such as `DLMDSAM01` are matched exactly
  thanks to the sparse part.
* **Incremental ingestion**: a manifest remembers a fingerprint per document, so re-running
  `ingest`/`moodle sync` only touches new, changed or deleted material.

## Requirements

* Python 3.11 or 3.12 (local run) – or Docker / Kubernetes
* An API key for at least one LLM provider:
  * Claude: `ANTHROPIC_API_KEY` from <https://console.anthropic.com> (a claude.ai subscription
    is not an API key; alternatively install the Anthropic CLI and run `ant auth login`, the SDK
    picks that profile up automatically)
  * Swiss AI Apertus: `HF_TOKEN`, a fine-grained Hugging Face token with the *Inference Providers*
    permission from <https://huggingface.co/settings/tokens> (or a Public AI key, see below)
  * Kimi: `MOONSHOT_API_KEY` from <https://platform.kimi.ai>
  * or a running Ollama server (`OLLAMA_BASE_URL`)
* Embeddings run on the CPU, nothing else is needed. The first start downloads ~330 MB of models.

## Quick start (local)

```bash
git clone https://github.com/Chantifa/iu-campus-agent.git
cd iu-campus-agent
py -3.12 -m venv .venv            # Windows;  python3.12 -m venv .venv on macOS/Linux
.venv\Scripts\activate            # source .venv/bin/activate
pip install -e ".[dev]"
copy .env.example .env            # cp .env.example .env  -> fill in the keys and IU_DOCS_PATH
iu-agent ingest                   # index the OneDrive/IU folder (incremental, safe to repeat)
iu-agent                          # start chatting; the CLI asks which model to use
```

Useful commands:

| Command | What it does |
|---|---|
| `iu-agent` / `iu-agent chat` | interactive agent (asks for the model unless `--model`/`DEFAULT_MODEL` is set) |
| `iu-agent ask "question" [-m kimi:kimi-k3] [-y]` | one-shot question, `-y` auto-approves file writes / shell |
| `iu-agent ingest [PATH] [--force] [--reset] [--dry-run]` | index the IU folder or a sub folder |
| `iu-agent search "query" [--course X] [--k 6]` | raw retrieval without an LLM |
| `iu-agent models [--live]` | list selectable models (live queries the provider APIs) |
| `iu-agent status` | providers, vector store, index statistics |
| `iu-agent moodle check / login / status / courses / sync / logout` | myCampus integration |

## Choosing the model

The CLI shows a numbered list at start-up and you pick a model (Enter takes the default). Inside
the chat use `/model` for the list again, or switch directly, e.g. `/model kimi:kimi-k2.6` or
`/model claude-sonnet-5`. `/models live` asks the provider APIs for every available model.

| Provider | Models (recommended first) | Configuration |
|---|---|---|
| Anthropic Claude | `claude-opus-5` (default), `claude-sonnet-5`, `claude-haiku-4-5`, `claude-fable-5-1`, `claude-opus-4-8`, `claude-sonnet-4-6` | `ANTHROPIC_API_KEY`, optional `ANTHROPIC_EFFORT=low|medium|high|xhigh|max` |
| Swiss AI Apertus | `swiss-ai/Apertus-v1.5-70B` (default), `swiss-ai/Apertus-70B-Instruct-2509`, `swiss-ai/Apertus-8B-Instruct-2509` | `HF_TOKEN` (or `SWISSAI_API_KEY`), optional `SWISSAI_BASE_URL`, `SWISSAI_PROVIDER`, `SWISSAI_TOOLS` |
| Moonshot Kimi | `kimi-k3` (default), `kimi-k2.7-code`, `kimi-k2.7-code-highspeed`, `kimi-k2.6` | `MOONSHOT_API_KEY`, optional `KIMI_REASONING_EFFORT=low|high|max` |
| Ollama | whatever is pulled, e.g. `qwen3:8b` | `OLLAMA_BASE_URL=http://localhost:11434` |

Set `DEFAULT_MODEL=anthropic:claude-opus-5` (or `swissai:swiss-ai/Apertus-v1.5-70B`, `kimi:kimi-k3`)
in `.env` to skip the question. Switching models mid-conversation keeps the conversation.

### Swiss AI Apertus

[Apertus](https://huggingface.co/swiss-ai) is the open, Apache-2.0 model family of the Swiss AI
Initiative (ETH Zurich, EPFL, CSCS): `swiss-ai/Apertus-v1.5-70B` (July 2026, 64K context),
`swiss-ai/Apertus-70B-Instruct-2509` and `swiss-ai/Apertus-8B-Instruct-2509`. The agent talks to
them through OpenAI-compatible endpoints:

| Endpoint | Configuration |
|---|---|
| Hugging Face inference router (default) | `HF_TOKEN=hf_…` (fine-grained token with the *Inference Providers* permission). Optional `SWISSAI_PROVIDER=publicai` or `featherless-ai` pins the provider. |
| Public AI Inference Utility | `SWISSAI_BASE_URL=https://api.publicai.co/v1`, `SWISSAI_API_KEY=…`, `SWISSAI_MODEL=swiss-ai/apertus-v1.5-70b` |
| Your own server (vLLM, SGLang) | `SWISSAI_BASE_URL=http://host:8000/v1`, `SWISSAI_API_KEY=anything` |
| Local through Ollama | pull a GGUF build (`ollama run hf.co/<user>/Apertus-8B-Instruct-2509-GGUF:Q4_K_M`) and use the `ollama` provider |

`iu-agent models --live` lists the Apertus models the router currently serves. Apertus supports
tool use; if an endpoint rejects the `tools` parameter set `SWISSAI_TOOLS=false` and the agent
switches to classic RAG: the best matching chunks are injected into the prompt for every message,
and the file and shell tools are disabled.

## Indexing the OneDrive/IU folder

`iu-agent ingest` walks `IU_DOCS_PATH` (default `~/OneDrive/IU`) and indexes every supported file:
`.pdf .docx .pptx .xlsx .html .ipynb .md .txt .tex .bib .py .csv .rst`.

* The folder layout `Semester_1/Advanced Maths/DLMDSAM01-01_Session1.pdf` is turned into metadata:
  semester, course name, course code, file name and page/slide. The agent can filter searches by
  course.
* `IU_EXCLUDE_GLOBS` (default `Bill/**,Certificate/**`) keeps contracts and certificates out of
  the index. Set it to an empty string to index everything, or add more patterns.
* `iu-agent ingest "C:/Users/X/OneDrive/IU/Semester_3"` indexes only one sub folder (metadata stays
  relative to the IU root).
* Unchanged files are skipped, changed files are re-embedded, deleted files are removed
  (`--no-prune` keeps them). `--reset` drops the collection and starts over, `--force` re-embeds
  everything.
* Scanned PDFs without a text layer are reported as failed (no OCR).

The vector database lives in `data/qdrant` (embedded Qdrant, no server needed). Point `QDRANT_URL`
to a server (`http://localhost:6333`) to share the index with Docker or other processes; the
embedded mode can only be opened by one process at a time.

### Speed and embedding profiles

Embeddings are computed on the CPU. Measured on a ThinkPad (64 chunks of ~1,100 characters,
fastembed, ONNX runtime):

| `EMBEDDING_MODEL` | chunks/s | dim | notes |
|---|---|---|---|
| `jinaai/jina-embeddings-v2-base-de` (default) | 1.3 | 768 | best quality, German + English, 8k-token context |
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | 1.5 | 768 | 50 languages |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 9.8 | 384 | **fast profile**, truncates at 128 tokens, use `CHUNK_SIZE=450` |
| `minishlab/potion-multilingual-128M` | 568 | 256 | static embeddings, instant indexing, weaker semantics (BM25 still exact) |

A real run: the *Advanced Maths* course (15 PDFs incl. a 280-page course book) became 858 chunks
in about 15 minutes with the default model. For the whole IU folder plan on several hours with the
default model, well under an hour with the fast profile:

```bash
# fast profile (put into .env, then `iu-agent ingest --reset` because the vector size changes)
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
CHUNK_SIZE=450
CHUNK_OVERLAP=60
```

Indexing is incremental, so a long first run can also be split by folder
(`iu-agent ingest "…/IU/Semester_1"`, then `Semester_2`, …) or left running in Docker.

## myCampus (Moodle) integration

`https://mycampus-classic.iu.org` is a Moodle site. Its public configuration
(`tool_mobile_get_public_config`, see `iu-agent moodle check`) shows:

| Setting | Value |
|---|---|
| Web services | enabled |
| Mobile web service (`moodle_mobile_app`) | enabled |
| Login type | 3 = SSO through an embedded browser (identity provider `auth.iu.org`, OAuth2) |
| REST endpoint | `/webservice/rest/server.php` (answers *Invalid token* without a token) |

So there **is** an API for the course content, but because the login is SSO, a token cannot be
obtained with a username/password call. The agent uses the same flow as the official Moodle app:

```bash
iu-agent moodle login
```

1. A browser opens `…/admin/tool/mobile/launch.php?service=moodle_mobile_app&passport=…`.
2. Log in with your IU account.
3. Moodle redirects to `moodlemobile://token=<base64>`; the browser cannot open that scheme but
   shows the address (address bar or error page). Copy the whole address.
4. Paste it into the terminal. The CLI decodes `sitehash:::token:::privatetoken`, checks the
   site hash, verifies the token with `core_webservice_get_site_info` and stores it in
   `data/moodle_token.json` (or pass a token directly with `--token`, e.g. one created under
   *Preferences → Security keys* in myCampus).

Then:

```bash
iu-agent moodle courses          # enrolled courses (core_enrol_get_users_courses)
iu-agent moodle sync             # download + index everything, incremental
iu-agent moodle sync --course 12345 --force
```

What `sync` collects per course (`core_course_get_contents` and the `mod_*` functions):
files attached to resources / folders (PDF, DOCX, PPTX …, downloaded with the token to
`data/moodle/<course id>/`), pages, book chapters, labels and module descriptions, assignment
descriptions with due dates, announcement forum posts (`MOODLE_SYNC_FORUMS=news|all|none`) and URL
activities as link records (`MOODLE_FETCH_URLS=true` also fetches their text). Quizzes, SCORM,
videos, H5P and external LTI tools cannot be exported through the API and are listed as skipped.

When a token is stored the agent additionally gets the live tools `moodle_list_courses` and
`moodle_course_contents`.

If the myCampus content is not accessible for you (no token), the agent simply works with the
OneDrive/IU folder – both sources can be combined in the same index.

## Using the agent

```
❯ /help                      commands
❯ Fasse Kapitel 3 des Kursbuchs Advanced Maths zusammen
❯ Which assignments are due in Deep Learning?          (after moodle sync)
❯ Write a python script in workspace/ that plots the gradient descent example from session 4
❯ /search Bayes theorem                                 raw retrieval, no LLM
❯ /model                                                pick another model
```

Tools the model can call: `search_course_material`, `list_courses`, `list_documents`,
`read_document`, `list_directory`, `read_file`, `write_file`*, `run_shell`*, `fetch_url`,
`moodle_list_courses`, `moodle_course_contents` (* = approval prompt). File tools are restricted to
`WORKSPACE_DIR` (default: the current directory) unless `ALLOW_OUTSIDE_WORKSPACE=true`.

Keys: Enter sends, Alt+Enter inserts a newline, Ctrl+C cancels the running answer or clears the
input, Ctrl+D exits, Up/Down browse the history (`data/chat_history.txt`).

## Docker Compose

```bash
cp .env.example .env              # keys + IU_DOCS_HOST_PATH (folder that is mounted read-only)
docker compose build
docker compose run --rm agent ingest          # index the mounted folder into the qdrant service
docker compose run --rm agent                 # interactive chat
docker compose run --rm agent moodle login --no-browser   # prints the login URL, paste the result
docker compose run --rm agent moodle sync
```

* `qdrant` runs as a service (`qdrant/qdrant:v1.19.1`, volume `qdrant_data`, port 6333 also on
  the host so a local `iu-agent` can use `QDRANT_URL=http://localhost:6333`).
* `agent_data` keeps the manifest, cached embedding models, the Moodle token and the chat history.
* `WORKSPACE_HOST_PATH` (default `./workspace`) is the folder the agent may edit.
* `docker compose build --build-arg PRELOAD_EMBEDDINGS=true` bakes the embedding models into the
  image for offline use.

## Kubernetes

The manifests in `k8s/` (kustomize) deploy Qdrant as a StatefulSet, two PVCs (agent data and
the course material), an ingestion Job and an idle agent Deployment you attach to:

```bash
docker build -t iu-campus-agent:latest .                 # image must be reachable by the cluster
make k8s-load-image                                      # Docker Desktop (kind): copy it into the node
kubectl create namespace iu-agent
kubectl -n iu-agent create secret generic iu-agent-secrets --from-env-file=.env   # keys; paths in it are ignored
kubectl apply -k k8s/
scripts/k8s-upload-docs.sh "C:/Users/X/OneDrive/IU"      # copy the documents into the cluster (once)
kubectl -n iu-agent logs -f job/iu-agent-ingest          # indexing starts after the upload
kubectl -n iu-agent exec -it deploy/iu-agent -- iu-agent chat
```

The cluster cannot see your PC's folders (Docker Desktop's kind-based Kubernetes has no host
mount, and a real cluster is on another machine anyway), so the documents live in the
`iu-agent-docs` volume. `scripts/k8s-upload-docs.sh` streams only the indexable files (PDF, DOCX,
PPTX, notebooks, text; `Bill/` and `Certificate/` skipped) into the agent pod with `tar` over
`kubectl exec` and finally writes `/data/iu/.upload-complete`, which releases the ingest Job. Re-run
the script after adding material, then restart the job:

```bash
kubectl -n iu-agent delete job iu-agent-ingest && kubectl apply -k k8s/
```

The secret may simply be your whole `.env`: the pods take `IU_DOCS_PATH`, `DATA_DIR`, `QDRANT_URL`
and `WORKSPACE_DIR` from the manifests, so Windows paths in the file do no harm.
Docker Desktop's kind-based Kubernetes keeps its own image store, so a locally built image has to
be imported into the node (`make k8s-load-image` runs `docker save … | docker exec -i
desktop-control-plane ctr -n k8s.io images import -`); on a real cluster push the image to a
registry and adjust `image:` in the manifests instead. Both PVCs are `ReadWriteOnce`; on a multi-node cluster use an RWX storage class (NFS, CephFS) or
pin the pods to one node. `make k8s-apply`, `make k8s-upload-docs` and `make k8s-delete` wrap the
commands.

## Configuration reference

All settings are environment variables (or `.env`), see `.env.example`.

| Variable | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN` | – | Claude access (`ANTHROPIC_ENABLED=true` when using an `ant auth login` profile) |
| `ANTHROPIC_MODEL` / `ANTHROPIC_EFFORT` | `claude-opus-5` / – | default Claude model, effort level |
| `HF_TOKEN` (or `SWISSAI_API_KEY`) | – | Swiss AI Apertus access (Hugging Face inference router or another OpenAI-compatible endpoint) |
| `SWISSAI_BASE_URL` / `SWISSAI_MODEL` / `SWISSAI_PROVIDER` / `SWISSAI_TOOLS` | `https://router.huggingface.co/v1` / `swiss-ai/Apertus-v1.5-70B` / – / true | Apertus endpoint, default model, provider pin, function calling |
| `MOONSHOT_API_KEY` (or `KIMI_API_KEY`) | – | Kimi access |
| `MOONSHOT_BASE_URL` / `KIMI_MODEL` / `KIMI_REASONING_EFFORT` | `https://api.moonshot.ai/v1` / `kimi-k3` / – | Kimi endpoint and defaults |
| `OLLAMA_BASE_URL` / `OLLAMA_MODEL` / `OLLAMA_NUM_CTX` | – / `qwen3:8b` / 32768 | local models |
| `DEFAULT_MODEL` | – | `provider:model` to start with (otherwise the CLI asks) |
| `MAX_OUTPUT_TOKENS` / `CONTEXT_BUDGET_TOKENS` | 16000 / 120000 | answer length, history budget |
| `QDRANT_URL` / `QDRANT_API_KEY` / `QDRANT_COLLECTION` | – / – / `iu_course_material` | vector store (unset = embedded under `DATA_DIR/qdrant`) |
| `EMBEDDING_MODEL` / `SPARSE_MODEL` / `HYBRID_SEARCH` | `jinaai/jina-embeddings-v2-base-de` / `Qdrant/bm25` / true | retrieval |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` / `RETRIEVAL_K` | 1200 / 150 / 6 | chunking and number of results |
| `IU_DOCS_PATH` / `IU_INCLUDE_GLOBS` / `IU_EXCLUDE_GLOBS` / `MAX_FILE_MB` | `~/OneDrive/IU` / – / `Bill/**,Certificate/**` / 200 | source folder |
| `DATA_DIR` / `WORKSPACE_DIR` / `ALLOW_OUTSIDE_WORKSPACE` | `data` / `.` / false | state folder, agent workspace |
| `MOODLE_URL` / `MOODLE_TOKEN` / `MOODLE_SYNC_FORUMS` / `MOODLE_FETCH_URLS` | `https://mycampus-classic.iu.org` / – / `news` / false | myCampus |

Changing `EMBEDDING_MODEL` requires `iu-agent ingest --reset` (different vector size).

## Development

```bash
pip install -e ".[dev]"
pytest -q            # 33 tests: loaders, chunking, incremental ingest, Moodle client/sync (mock server),
                     # LangGraph loop + interrupts, tools, model registry, CLI
ruff check src tests
```

The tests use a deterministic hash embedding and the embedded Qdrant, so they need no network and
no API key. Open the folder in PyCharm: `.idea/` contains the module definition (sources in `src`,
tests in `tests`, `.venv` interpreter) and run configurations for *iu-agent chat*, *iu-agent ingest*
and *pytest*.

Project layout:

```
src/iu_agent/
  cli.py            typer commands + the interactive ChatSession (streaming, slash commands)
  ui.py             rich rendering: banner, streaming Markdown, tool calls, approval prompts
  config.py         pydantic-settings configuration
  models.py         provider registry (Anthropic / Swiss AI Apertus / Kimi / Ollama), live lists, factory
  agent/graph.py    LangGraph StateGraph (agent ⇄ tools), history trimming
  agent/tools.py    RAG + software tools, interrupt-based approvals
  agent/prompts.py  system prompt
  rag/loaders.py    PDF/DOCX/PPTX/XLSX/HTML/ipynb/text extraction
  rag/chunking.py   chunking with deterministic ids and context headers
  rag/embeddings.py fastembed adapter
  rag/store.py      Qdrant (embedded or server), hybrid retrieval
  rag/ingest.py     manifest, folder scanning, incremental ingestion
  moodle/client.py  Moodle REST web-service client
  moodle/auth.py    launch-URL (SSO) token flow, token storage
  moodle/sync.py    course content -> documents
tests/              pytest suite
k8s/                kustomize manifests
Dockerfile, docker-compose.yml, Makefile, .env.example
```

## Limitations and notes

* Claude models need an Anthropic **API** key (or an `ant auth login` profile); a Claude Pro/Max
  subscription cannot be used by third-party applications.
* Kimi K3 always thinks; if tool calling misbehaves with a thinking model, try
  `KIMI_REASONING_EFFORT=low` or `kimi-k2.6`.
* Apertus on the Hugging Face router is served by third-party providers (Public AI, Featherless);
  if a provider rejects tool calls, set `SWISSAI_TOOLS=false` for prompt-injected retrieval.
* The embedded Qdrant is single-process: run either `ingest` or `chat` at a time, or use the
  Qdrant server from `docker compose up -d qdrant` with `QDRANT_URL=http://localhost:6333`.
* Conversations are kept in memory for the session (`/clear` starts a new thread); the input
  history is persisted.
* Retrieved chunks are sent to the selected LLM provider. Keep sensitive documents out of the index
  with `IU_EXCLUDE_GLOBS` (contracts and certificates are excluded by default).
* When a tool call is cancelled with Ctrl+C the CLI closes the pending tool call so the next
  message is still valid for the provider.

## License

MIT
