# 🎙️ MeetSaransh - Meeting Summarizer

[![CI](https://github.com/R1tulD3v/MeetSaransh/actions/workflows/ci.yml/badge.svg)](https://github.com/R1tulD3v/MeetSaransh/actions/workflows/ci.yml)
![Coverage gate](https://img.shields.io/badge/coverage%20gate-90%25-brightgreen)
![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)

Transcribe meeting audio, generate **action-oriented** summaries, and **ask questions
across all your meetings** with grounded, cited answers.

> _Saransh_ (सारांश) is Hindi for "summary".

**Pipelines:**
`audio -> validate -> transcribe (Whisper) -> summarize (LLM) -> store -> view`
`question -> hybrid retrieval -> grounded answer (LLM) -> cited excerpts`

---

## Features

- **Upload audio** (`.mp3 .wav .m4a .mp4 .webm .ogg .flac ...`) -> transcript + summary.
- **Layered summary** - TL;DR -> key decisions -> action items -> open questions -> topic
  timeline. Depth, not one flat paragraph.
- **Grounded action items** - each with an owner (or `Unassigned`), a due date (or
  `Not specified`), and a `[mm:ss]` timestamp. The prompt is told **not to invent** them.
- **Ask your meetings (RAG)** - semantic + keyword search across every meeting, with a
  written answer and **clickable citations** that jump to the exact transcript moment.
  Honest by design: it says *"I couldn't find anything about that"* when a topic wasn't
  discussed, instead of guessing.
- **Click-to-seek** everywhere - timestamps in the summary, transcript, and chat citations
  all seek the audio / scroll the transcript.
- **Transcript search** with match highlighting.
- **Export** the summary as copy-ready Markdown.
- **Runs with no API key** - a bundled sample meeting demonstrates the whole app offline;
  RAG chat falls back to returning the most relevant excerpts.

### Operational features

- **243 tests, 98% coverage**, every provider call mocked - the suite runs offline.
- **CI on every push**: lint, format, types, tests + coverage gate, dependency audit,
  Docker build and a container smoke test.
- **Containerized**: multi-stage build, non-root user, real health check.
- **Rate limited** per client, with the tightest budget on the endpoints that spend money.
- **Content-validated uploads** - magic bytes, not just the file extension.
- **Structured JSON logs** with request-id correlation, and **Prometheus metrics**.
- **Versioned schema migrations** - the database upgrades in place.

---

## Quick start

**Prerequisites:** Python 3.12+ and a free [Groq API key](https://console.groq.com/keys)
(no credit card). The app also runs **without** a key using the sample meeting.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt

copy .env.example .env            # optional - paste your GROQ_API_KEY (cp on macOS/Linux)

python run.py
```

Open **http://127.0.0.1:8000** -> **Load sample meeting** -> try the **Ask your meetings**
tab. On first RAG use, a ~90 MB embedding model downloads once and is cached.

Interactive API docs (development only): **http://127.0.0.1:8000/docs**

### With Docker

```bash
docker compose up --build
```

Add the metrics stack (Prometheus scraping `/metrics` at http://localhost:9090):

```bash
docker compose --profile observability up --build
```

---

## Development

```bash
pip install -r requirements-dev.txt
pre-commit install          # run the same checks CI runs, before each commit
```

| Task | Command |
| --- | --- |
| Run the tests | `pytest` |
| Tests without the coverage gate | `pytest --no-cov` |
| Lint | `ruff check .` |
| Format | `ruff format .` |
| Type-check | `mypy` |
| Audit dependencies | `pip-audit -r requirements.txt --strict` |

The suite needs **no API key and no network**: Groq calls are mocked with `respx`, and
the embedding model is forced unavailable so no test downloads it. Tests that need the
dense retrieval path use deterministic stand-in vectors instead.

---

## "Ask your meetings" - how the RAG works

The interesting engineering is here, so it's worth spelling out:

- **Chunking** groups consecutive transcript segments into ~130-word chunks with overlap,
  keeping each chunk's `[start, end]` timestamps and its constituent segments - so a
  citation can point at the exact line that answers the question, not just a chunk.
- **Hybrid retrieval** combines a **dense** semantic score (`fastembed` bge-small cosine)
  with a **lexical** BM25 score (pure Python). Research on spoken/transcript content shows
  hybrid beats pure-vector search - and the lexical half means the feature still works if
  the embedding model can't load.
- **Two-layer grounding.** A cheap similarity gate hard-refuses clearly-unrelated
  questions; for everything else the LLM prompt is the real guard - it may answer *only*
  from the retrieved excerpts and must say so when they don't contain the answer.
- **Query-aware citations.** Each source shows the most relevant segment (with its own
  timestamp) and links straight into the transcript at that moment.
- **Storage:** embeddings are stored as `float32` blobs in SQLite; retrieval is
  brute-force cosine in numpy. At this scale that's correct and simple - an ANN index
  (HNSW/IVFFlat) is the scaling path, not a demo requirement.

---

## Architecture

```
Browser (vanilla HTML/JS, no build step) -- Meetings view + Ask (chat) view
        |  fetch()
        v
Middleware -- request id . access log . metrics . rate limit . security headers
        |
        v
FastAPI (app/main.py) -- routes, upload validation, Markdown export
        |
        +-- app/transcription.py  -> Groq Whisper  (ASR, segment timestamps)
        +-- app/summarize.py      -> structured JSON summary
        +-- app/rag.py            -> chunk . hybrid retrieve . grounded answer
        +-- app/embeddings.py     -> fastembed bge-small (ONNX, lazy-loaded)
        +-- app/llm.py            -> shared Groq chat client (retry/backoff)
        +-- app/prompts.py        -> prompt engineering (summary + RAG)
        +-- app/storage.py        -> SQLite: meetings + chunks (stdlib, migrated)
```

```
app/
  main.py          # routes + pipelines + lifespan
  config.py        # env-driven settings + startup validation
  schemas.py       # Pydantic request/response models (the API contract)
  errors.py        # one error envelope + centralized handlers
  middleware.py    # request ids, access logs, metrics, rate limit, headers
  security.py      # magic-byte validation, rate limiter, security headers
  observability.py # JSON logging + Prometheus metrics
  transcription.py # ASR + timestamp formatting
  summarize.py     # summary + defensive JSON parsing
  rag.py           # chunking, BM25, hybrid retrieval, grounded answer
  prompts.py       # graded prompt artifacts (summary + RAG)
  storage.py       # SQLite (meetings + chunks) + versioned migrations
  llm.py           # shared chat client + retry/backoff
  embeddings.py    # local dense embeddings (fastembed)
static/            # index.html, style.css, app.js  (no framework)
tests/             # 243 tests, providers mocked, no network
ops/               # Prometheus scrape config
data/sample/       # bundled sample meeting (offline demo)
```

### API

All routes are versioned under `/api/v1`. The unversioned `/api` prefix still works as a
compatibility alias but is not documented in the OpenAPI schema.

| Method | Route | Purpose |
| --- | --- | --- |
| `GET`  | `/api/v1/health` | status, models, and a real database round-trip |
| `POST` | `/api/v1/meetings` | upload audio -> transcribe -> summarize -> store -> index |
| `POST` | `/api/v1/meetings/sample` | create the bundled sample (no key needed) |
| `GET`  | `/api/v1/meetings` | paginated list (`?limit=&offset=`) |
| `GET`  | `/api/v1/meetings/{id}` | full meeting (transcript, segments, summary) |
| `GET`  | `/api/v1/meetings/{id}/audio` , `/export` | stream audio , Markdown export |
| `DELETE` | `/api/v1/meetings/{id}` | delete meeting + audio + chunks |
| `POST` | `/api/v1/chat` | grounded Q&A (optional `meeting_id` scope) |
| `POST` | `/api/v1/reindex` | index any meetings not yet in the vector store |
| `GET`  | `/api/v1/rag/status` | embeddings availability, indexed meeting/chunk counts |
| `GET`  | `/metrics` | Prometheus exposition (unversioned, by convention) |

Every failure returns the same envelope, so clients branch on a code rather than parsing
prose:

```json
{"error": {"code": "unsupported_content", "message": "...", "request_id": "4990113eef34"}}
```

The `request_id` is also returned as the `X-Request-ID` header and appears on every
server log line for that request.

---

## Configuration

Every setting is an environment variable with a working default, so an empty `.env` is a
valid configuration. See [`.env.example`](.env.example) for the full annotated list.
`app/config.py` validates on startup and refuses to boot on a combination that cannot
work (for example `CHUNK_OVERLAP_WORDS >= CHUNK_TARGET_WORDS`, which would make the
chunker unable to advance), while merely-degraded states such as a missing API key are
logged as warnings and the app starts anyway.

---

## Observability

- **Logs**: one JSON object per line in production (`LOG_JSON=true`), human-readable
  lines in development. Every line carries the request id, and the access log records
  method, route template, status, and duration.
- **Metrics** at `/metrics`: request counts and a latency histogram (bucketed up to 300s,
  because a transcription is not a 10ms request), rate-limit rejections, rejected uploads
  by reason, provider call outcomes and latency by provider, RAG answer modes, and gauges
  for stored meetings and indexed chunks.
- Metric labels use the **route template** (`/api/v1/meetings/{meeting_id}`), never the
  raw path, so meeting ids cannot explode the time-series cardinality.

Useful queries once Prometheus is running:

```promql
histogram_quantile(0.95, sum(rate(meetsaransh_http_request_duration_seconds_bucket[5m])) by (le, route))
sum(rate(meetsaransh_provider_calls_total[5m])) by (provider, outcome)
sum(rate(meetsaransh_rate_limited_total[5m])) by (route)
```

---

## Design decisions (and the trade-offs)

- **Python + FastAPI + vanilla JS, no frontend framework** - ASR/LLM are just HTTP calls,
  so a heavy SPA earns nothing. Zero `node_modules`, no build step. It also lets the CSP
  be genuinely strict, since there is no inline script or style to whitelist.
- **SQLite via the standard library** for both meetings and the vector store - a real
  relational DB, no hosted service, no ORM. Trade-off: single-writer, not built for high
  concurrency - fine for this scope.
- **Hand-rolled migrations on `PRAGMA user_version`** rather than Alembic - Alembic earns
  its weight alongside SQLAlchemy, and there is no ORM here. The mechanism is the point:
  schema changes no longer mean deleting the database.
- **`fastembed` (ONNX) over `sentence-transformers` (torch)** for embeddings -
  CPU-friendly, no heavy `torch` dependency, no extra API key (Groq has no embeddings
  endpoint), works offline. The model downloads once and caches.
- **Hybrid retrieval, not pure-vector** - better recall on conversational text, and it
  degrades to lexical-only if embeddings are unavailable.
- **LLM as the real grounding layer** - the similarity gate can't separate "loosely
  related" from "off-topic" reliably (bge-small compresses those into one score band), so
  the prompt does the honest refusing. Documented rather than pretended-away.
- **Shared `llm.py` with retry-and-backoff on 429** - the free-tier rate limits make this
  a real requirement, not decoration. A 401 is deliberately *not* retried: the key will
  not fix itself.
- **In-process rate limiter, sliding window** - a fixed window would let a client send 2x
  the limit across a boundary, which on an endpoint that triggers a paid transcription is
  a money bug. In-process because the app runs as one process today; the interface is
  what matters, and swapping the deques for a Redis sorted set is a drop-in change.
- **Stdlib `logging` with a JSON formatter, not structlog** - the only thing a logging
  framework would add here is a dependency.
- **Synchronous processing with a loading state**, not a job queue - fewer failure modes
  for a single-user app. Async is a roadmap item, not a pretense.
- **No ffmpeg / no audio chunking in v1** - both mean a heavy dependency or a system
  binary; files are validated and capped at 25 MB (the provider's limit) instead.

---

## Security

- API key is read from `.env` (git-ignored) and used **server-side only** - never sent to
  the browser, never baked into a container image.
- **Uploads are validated by content, not just name.** The magic bytes must identify a
  real audio container *and* match the claimed extension, so a renamed PDF or executable
  is rejected before a single paid API call is made.
- **Uploads stream to disk in slices** with the size cap enforced as they arrive, so an
  oversized body is rejected after ~1 MB rather than after being buffered in memory. A
  rejected or failed upload never leaves a partial file behind.
- **Rate limiting** per client per endpoint, tightest on the endpoints that call paid
  APIs. `X-Forwarded-For` is ignored unless `TRUST_PROXY_HEADERS` is explicitly enabled,
  because otherwise a client could spoof it and mint unlimited identities.
- **Security headers** on every response including errors: a strict CSP with no
  `unsafe-inline`, `frame-ancestors 'none'`, `nosniff`, a referrer policy, and HSTS in
  production only (it would pin a browser to a broken scheme on a local http:// server).
- **Errors never leak internals.** Stack traces go to the logs; the client gets a generic
  message and a request id to quote.
- Caller-supplied `X-Request-ID` is echoed for tracing only if it matches a conservative
  character set - it lands in log files, so newline injection would let a caller forge
  log lines.
- Foreign keys are **enforced** (`PRAGMA foreign_keys=ON`) with `ON DELETE CASCADE`, so
  deleting a meeting cannot orphan its vector chunks.
- `pip-audit` runs in CI and Dependabot opens upgrade PRs that must pass the pipeline.
- Interactive docs and the OpenAPI schema are disabled when `ENVIRONMENT=production`.

---

## Roadmap

Next up, in order:

1. **JWT auth + per-user data isolation** - the last big production-thinking gap.
2. **Async transcription queue** with status polling and progress in the UI.
3. **RAG evaluation harness** - recall@k, MRR, and faithfulness, with an ablation of
   hybrid vs dense-only vs lexical-only.
4. **Postgres + pgvector + an ANN index**, once the corpus outgrows brute-force cosine.
5. **Analytics dashboard** - action items by owner/status, decisions over time.
6. **Long-audio chunking** with overlap to exceed the 25 MB single-file cap.
7. **Speaker diarization** - label "Speaker 1 -> Priya".

---

## Limitations

Stated plainly, because a README that only lists strengths is not informative:

- Diarization is not included, so transcripts aren't speaker-labelled.
- Single file per upload, <= 25 MB (~40 min of typical audio).
- **No authentication yet.** Every meeting is visible to anyone who can reach the server,
  so do not deploy this publicly with real meeting audio until item 1 of the roadmap
  lands.
- Transcription is **synchronous**, so a long upload holds its request open.
- Rate-limiter state is per-process, so it does not survive a restart and would not be
  shared across multiple workers.
- SQLite + local files: single-writer, and the audio lives on the container's volume
  rather than object storage.
- The RAG refusal gate is deliberately lenient; the LLM prompt does the final grounding,
  so a written refusal for off-topic questions requires an API key.

---

## Tech stack

Python . FastAPI . Uvicorn . httpx . Pydantic . SQLite (stdlib) . fastembed (bge-small,
ONNX) . numpy . prometheus-client . vanilla JS/HTML/CSS . Docker . GitHub Actions .
pytest . ruff . mypy . Groq (Whisper `large-v3-turbo` + `openai/gpt-oss-120b`).
