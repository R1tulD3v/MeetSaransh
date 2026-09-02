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

- **Accounts with per-user isolation** - JWT access + revocable refresh tokens, and
  every meeting query filtered by owner in SQL.
- **Background transcription** - uploads return in milliseconds and are processed by a
  worker pool, with live status in the UI and recovery after a restart.
- **376 tests, 98% coverage**, every provider call mocked - the suite runs offline.
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

Open **http://127.0.0.1:8000**, **create an account** (it is local and takes a
second), then **Load sample meeting** -> try the **Ask your meetings** tab. On first RAG
use, a ~90 MB embedding model downloads once and is cached.

If you are upgrading a database that predates accounts, the **first** account you create
claims the meetings already in it, so nothing is stranded.

`JWT_SECRET` is optional in development (one is generated per process, so restarting
signs you out) and **required in production** - the app refuses to start without it.

Interactive API docs (development only): **http://127.0.0.1:8000/docs**

### With Docker

```bash
export JWT_SECRET=$(python -c "import secrets; print(secrets.token_urlsafe(48))")
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
        +-- app/deps.py           -> get_current_user (auth as a dependency)
        +-- app/auth.py           -> scrypt passwords + JWT access/refresh tokens
        +-- app/jobs.py           -> worker pool: transcribe . summarize . index
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
  auth.py          # password hashing (scrypt) + JWT issue/verify
  deps.py          # get_current_user, require_admin
  jobs.py          # background transcription pool + crash recovery
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
tests/             # 376 tests, providers mocked, no network
ops/               # Prometheus scrape config
data/sample/       # bundled sample meeting (offline demo)
```

### API

All routes are versioned under `/api/v1`. The unversioned `/api` prefix still works as a
compatibility alias but is not documented in the OpenAPI schema.

Every route below except `/health` and `/metrics` requires
`Authorization: Bearer <access token>` and only ever returns the caller's own data.

| Method | Route | Purpose |
| --- | --- | --- |
| `GET`  | `/api/v1/health` | status, models, and a real database round-trip (public) |
| `POST` | `/api/v1/auth/register` | create an account, returns a token pair |
| `POST` | `/api/v1/auth/login` | exchange credentials for a token pair |
| `POST` | `/api/v1/auth/refresh` | rotate a refresh token for a fresh pair |
| `POST` | `/api/v1/auth/logout` | revoke every refresh token for the caller |
| `GET`  | `/api/v1/auth/me` | the current account |
| `POST` | `/api/v1/meetings` | upload audio -> **202**, queued for background processing |
| `POST` | `/api/v1/meetings/sample` | create the bundled sample (no key needed) |
| `GET`  | `/api/v1/meetings` | paginated list (`?limit=&offset=`) |
| `GET`  | `/api/v1/meetings/{id}` | full meeting; also the **status polling** endpoint |
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

## Authentication and data isolation

- **Passwords** are hashed with stdlib `hashlib.scrypt` (n=2^15, r=8 - about 32 MB per
  hash). The stored string carries its own parameters, so the cost can be raised later
  and existing hashes are transparently upgraded on the owner's next login.
- **Tokens** are JWTs via PyJWT. A 30-minute access token, and a 7-day refresh token
  whose id is stored hashed so it can be revoked. Refresh **rotates**: a refresh token
  is spendable once, so a stolen one stops working as soon as the real user refreshes.
- **Authentication is a dependency, not middleware.** Middleware would need a list of
  public paths kept in sync with the routes by hand, and the failure mode of that
  drifting is an endpoint that is quietly unauthenticated. As a dependency, a route is
  protected exactly when its signature says so - visible in review and in `/docs`.
- **Isolation is enforced in SQL.** Every storage function takes `user_id` and filters
  on it in the query, so forgetting it is a `TypeError` at import time rather than a
  data leak in production. That includes `get_chunks`, which retrieval reads from
  directly: a missing filter there would not look wrong, it would just start answering
  one user's questions from another user's meetings, complete with citations.
- Someone else's meeting id returns **404, not 403** - a 403 confirms the id exists and
  turns the API into an enumeration oracle. Login failures are likewise indistinguishable
  between "no such account" and "wrong password".

---

## Background processing

Uploading used to run ASR and summarization inside the request, so a 40-minute recording
held the connection open for minutes and any client timeout destroyed work that had
already been paid for at the provider. Now:

```
POST /meetings  ->  202 { id, status: "queued", poll_url }
                     |
   worker pool ->  processing (transcribing -> summarizing -> indexing)  ->  done
                     |
                     +-- on failure  ->  error, with the reason on the meeting
```

The client polls `GET /meetings/{id}` and the UI shows the current stage. Design notes:

- **A thread pool and a status column, not Celery or RQ.** The work is one long network
  call, so it is I/O-bound: threads are the right primitive, and a broker would add an
  operational dependency for no throughput. The database is the queue of record, which
  is what makes crash recovery possible. Moving to a real broker later is
  `submit()` -> `queue.enqueue()`; the state machine and recovery logic do not change.
- **Interrupted jobs are failed at startup, not retried.** The previous attempt may
  already have spent an ASR call, and silently re-spending a user's provider budget on
  every restart is worse than saying plainly that it failed.
- **A failed meeting is kept, not deleted.** A recording that vanishes tells the user
  nothing; a failed row carries the reason. Its audio *is* deleted, since nothing will
  read it again.
- **Per-user job quota**, so one account cannot fill the queue and spend the budget.

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
- **scrypt from the standard library for passwords, but PyJWT for tokens.** Calling a
  vetted KDF correctly is a matter of choosing parameters; verifying a JWT safely is
  not (`alg: none`, algorithm confusion, expiry handling), and auth is the last place
  to save a dependency. One added dependency, drawn where it actually buys safety.
- **Rate limits follow the account, not the IP.** An office behind one NAT shares an
  address, so IP-keyed limits punish colleagues for each other's usage. The bearer
  token is *verified* in the limiter rather than trusted, because an unverified `sub`
  would let anyone choose their own bucket.
- **Background processing with status polling**, not a synchronous request or a broker
  - see the section above for the reasoning and the trade-offs.
- **No ffmpeg / no audio chunking in v1** - both mean a heavy dependency or a system
  binary; files are validated and capped at 25 MB (the provider's limit) instead.

---

## Security

- API key is read from `.env` (git-ignored) and used **server-side only** - never sent to
  the browser, never baked into a container image.
- **No hardcoded default `JWT_SECRET`.** A shipped default is a shipped forgery key:
  anyone who read the source could mint a valid token for any account on every
  deployment that kept it. Production refuses to boot without one; development
  generates a random one per process.
- **Login and registration are rate-limited separately and tightly** - they are
  brute-force targets, and each attempt runs an intentionally slow KDF, so an uncapped
  login endpoint is both a credential risk and a CPU denial-of-service.
- Login failures are **timing- and message-identical** between an unknown account and a
  wrong password, so the form cannot be used to enumerate accounts.
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
  deleting a meeting cannot orphan its vector chunks, and deleting a user takes their
  meetings and chunks with it.
- **Static assets are cache-busted by version** and the app shell is `no-cache`, so a
  deploy cannot leave a browser running last release's JavaScript against a moved-on
  API - a failure that is invisible to anyone testing with an empty cache.
- `pip-audit` runs in CI and Dependabot opens upgrade PRs that must pass the pipeline.
- Interactive docs and the OpenAPI schema are disabled when `ENVIRONMENT=production`.

---

## Roadmap

Next up, in order:

1. **RAG evaluation harness** - recall@k, MRR, and faithfulness, with an ablation of
   hybrid vs dense-only vs lexical-only.
2. **Analytics dashboard** - action items by owner/status, decisions over time.
3. **Postgres + pgvector + an ANN index**, once the corpus outgrows brute-force cosine,
   plus Redis so the rate limiter and job queue are shared across processes.
4. **Long-audio chunking** with overlap to exceed the 25 MB single-file cap.
5. **Speaker diarization** - label "Speaker 1 -> Priya".

Done: [x] auth + per-user isolation &nbsp; [x] background transcription with polling.

---

## Limitations

Stated plainly, because a README that only lists strengths is not informative:

- Diarization is not included, so transcripts aren't speaker-labelled.
- Single file per upload, <= 25 MB (~40 min of typical audio).
- **The job queue and rate limiter hold per-process state.** They do not survive a
  restart and are not shared between workers, which is why the container runs a single
  worker. Redis is the fix, and it is on the roadmap rather than pretended away.
- **Access tokens cannot be revoked before they expire** (at most 30 minutes). That is
  the cost of stateless tokens; logout revokes the refresh token immediately, and the
  short access lifetime is what makes the trade acceptable.
- There is **no password reset flow** and no email verification - both need an email
  provider, which this project does not have.
- **Roles exist but nothing uses them.** `require_admin` is implemented and tested;
  there is simply no admin-only endpoint yet.
- SQLite + local files: single-writer, and the audio lives on the container's volume
  rather than object storage.
- The RAG refusal gate is deliberately lenient; the LLM prompt does the final grounding,
  so a written refusal for off-topic questions requires an API key.

---

## Tech stack

Python . FastAPI . Uvicorn . httpx . Pydantic . SQLite (stdlib) . fastembed (bge-small,
ONNX) . numpy . PyJWT . prometheus-client . vanilla JS/HTML/CSS . Docker .
GitHub Actions . pytest . ruff . mypy . Groq (Whisper `large-v3-turbo` +
`openai/gpt-oss-120b`).
