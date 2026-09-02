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
- **Editable action items** - each extracted with an owner (or `Unassigned`), a due
  date (or `Not specified`), and a `[mm:ss]` timestamp; the prompt is told **not to
  invent** them. Reassign, re-date, reword or tick them off inline, and the dashboard
  follows. Rows a human has touched are marked `edited`, so "the model said Priya owns
  this" stays distinguishable from "we decided Priya owns this".
- **Ask your meetings (RAG)** - semantic + keyword search across every meeting, with the
  question **rewritten into retrieval vocabulary** first, a **streamed** answer, and
  **clickable citations** that jump to the exact transcript moment. Citations arrive
  *before* the first token, so the sources are on screen while the answer is still being
  written. Honest by design: it says *"I couldn't find anything about that"* when a topic
  wasn't discussed, instead of guessing.
- **Click-to-seek** everywhere - timestamps in the summary, transcript, and chat citations
  all seek the audio / scroll the transcript.
- **Transcript search** with match highlighting.
- **Insights dashboard** - action items by owner (and how many actually have a due
  date), meeting cadence over time, recurring topics, and the work nobody has picked up.
  Every figure is a SQL aggregation, and each loose end links back to the meeting it
  came from.
- **Export** the summary as copy-ready Markdown.
- **Runs with no API key** - a bundled sample meeting demonstrates the whole app offline;
  RAG chat falls back to returning the most relevant excerpts.

### Operational features

- **Accounts with per-user isolation** - JWT access + revocable refresh tokens, and
  every meeting query filtered by owner in SQL.
- **Background transcription** - uploads return in milliseconds and are processed by a
  worker pool, with live status in the UI and recovery after a restart.
- **A measured retrieval pipeline** - a labelled evaluation set, an ablation, and a
  regression gate in CI, not just an assertion that the RAG is good.
- **577 tests, 98% coverage**, every provider call mocked - the suite runs offline.
- **CI on every push**: lint, format, types, tests + coverage gate, a retrieval-quality
  regression gate, dependency audit, Docker build and a container smoke test.
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
| Score retrieval quality | `python -m evaluation.run` |

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

### Streaming

`POST /chat/stream` returns Server-Sent Events. Measured against live Groq with query
rewriting off: **citations at 0.11s, first token at 1.28s** - so the sources sit on
screen for over a second before any prose appears, and a reader can start checking the
evidence instead of watching a spinner. A mid-stream failure also still leaves them with
the excerpts.

With rewriting on (the default) both numbers shift about 1.6s later, because retrieval
now waits on an LLM call. That is a real trade the section below spells out rather than
hides: better sources, slower first paint, one environment variable to choose between
them.

- **SSE, not WebSockets.** The traffic is one-directional and short-lived. SSE is plain
  HTTP, so it inherits the same auth dependency, rate limit and error envelope as every
  other endpoint instead of needing a parallel set of them.
- **`fetch` + a stream reader, not `EventSource`.** `EventSource` cannot send an
  `Authorization` header or a POST body, and every route here is authenticated. Parsing
  the frames by hand is a dozen lines.
- **Streams are never retried.** `chat` retries on 429; `chat_stream` does not, because
  the user has already been shown part of the previous attempt and a retry would splice
  two different answers into one paragraph. If the failure happens *before* the first
  token, nothing has been shown, so the client falls back to the buffered call.
- **One refusal gate, shared.** Both paths call the same `_passes_refusal_gate`. Two
  copies would eventually disagree, and the one users noticed would be whichever refused
  a question the other answered. There is a test asserting it stays a single function.

### Query rewriting - built, measured, and shipped on

A question and its answer rarely share vocabulary. People ask about "the basket page"
when the meeting said "cart serializer", or "the slogan" when it said "tagline". An LLM
rewrites the question into retrieval vocabulary before the search runs:

```
"Why was the basket page dragging?"
    -> "basket page dragging slow lag cart page performance issue why"

"What is being done about how often engineers get woken up?"
    -> "engineer wake up frequency actions taken measures steps plan reduce
        interruptions on-call alerts pager duty developers woken up often"
```

The harness says it works, and by a wide margin:

| Configuration | recall@1 | MRR@1 | right meeting @1 | paraphrase recall@1 |
| --- | --- | --- | --- | --- |
| hybrid | 0.827 | 0.846 | 0.885 | 0.500 |
| **hybrid + rewrite** | **0.981** | **1.000** | **1.000** | **1.000** |

Paraphrase recall at rank 1 **doubles**, and the top-ranked chunk now comes from the
right meeting every time. Three things make this safe rather than merely effective:

- **The refusal gate reads the ORIGINAL question, never the rewrite.** A rewriter adds
  synonyms, and synonyms of an off-topic question can collide with the corpus - so a
  gate reading the rewrite could let "what is the capital of France" pick up a keyword
  and stop being refused. The rewrite may improve *ranking*; it may not change whether
  there is an answer at all. There is a test that feeds a deliberately hostile rewriter
  and asserts the refusal still holds.
- **The prompt forbids answering.** A rewriter that adds facts would smuggle a
  hallucination into the retrieval step, where the grounding prompt can no longer catch
  it. Output longer than `QUERY_REWRITE_MAX_CHARS` is discarded on the assumption the
  model started explaining instead of rewriting.
- **Every failure path returns the original question** - no key, provider error, empty
  reply. A rewriter that can break search is worse than no rewriter.

**The cost, stated plainly:** one extra LLM call, measured at a **~1.6s median**. That
moves first-citation latency on the streaming endpoint from ~0.1s to ~1.7s, which
partly spends the win the streaming work bought. It ships on anyway, because answering
from the wrong meeting is a worse failure than answering slowly - but set
`QUERY_REWRITE_ENABLED=false` for latency-sensitive deployments and the app degrades to
plain hybrid retrieval.

### Reranking - built, measured, and shipped off

A cross-encoder reads the question and a chunk *together*, rather than comparing two
precomputed vectors, so it should rank better than the retriever. `app/reranker.py`
implements it with fastembed's ONNX cross-encoder (no torch, consistent with the
embedding model).

The harness says it does not help here:

| Configuration | recall@3 | MRR@3 | paraphrase recall@3 |
| --- | --- | --- | --- |
| **hybrid (shipped)** | **1.000** | **0.923** | **1.000** |
| hybrid + rerank | 0.962 | 0.897 | 0.875 |

So `RERANK_ENABLED` defaults to **false**. The reason is not mysterious once stated: the
eval corpus is 5 chunks, so reranking 5 candidates down to 3 has almost nothing to
reorder, and the cross-encoder's read of a ~130-word conversational chunk is worse than
the hybrid score it overrides. The picture should invert once a corpus is large enough
that the retriever's top-20 contains genuine near-misses - which is exactly why the code
is kept, tested and configurable rather than deleted. Turn it on with
`RERANK_ENABLED=true` and re-run the harness against your own data before believing it
helps.

---

## Measuring the retrieval pipeline

"I built RAG" and "I measured my RAG" are different claims. `evaluation/` holds a
hand-labelled set of **26 questions over 3 deliberately overlapping meetings**, plus 6
off-topic questions that must be refused.

```bash
python -m evaluation.run                 # ablation: hybrid vs dense vs lexical
python -m evaluation.run --alpha-sweep   # tune the hybrid weight
python -m evaluation.run --rerank        # A/B the cross-encoder second stage
python -m evaluation.run --rewrite       # A/B LLM query rewriting (needs a key)
python -m evaluation.run --mode lexical  # deterministic: no model, no network
python -m evaluation.run --judge         # add LLM-as-judge faithfulness (needs a key)
```

**What it reports:** hit rate, recall, precision and MRR at several values of k;
how often the top-ranked chunk came from the *right meeting*; refusal accuracy on the
off-topic set; and the false-refusal rate on the answerable set.

Four design decisions worth defending:

- **Labels are substrings, not chunk ids.** Chunking is one of the things being
  evaluated, so labels that break whenever the chunker changes would be useless.
- **Recall is reported separately for literal and paraphrased questions.** On questions
  that reuse the transcript's own words, BM25 alone already scores near-perfectly, so an
  ablation run only on those shows no difference between configurations and quietly
  justifies whichever one happened to ship.
- **Refusal accuracy is always shown next to the false-refusal rate.** A system that
  refuses everything scores a perfect 1.0 on the first number and is useless.
- **k is swept, not fixed.** With a small corpus a generous k retrieves everything and
  every configuration scores a meaningless 1.000.

### It has already changed three decisions

The hybrid weight shipped at `alpha = 0.65`. The sweep showed it scoring **0.875 recall
on paraphrased questions - identical to lexical-only** - because BM25's confident wrong
matches were outvoting the dense signal. At `0.8` that reaches 1.000 and MRR@3 improves
from 0.904 to 0.923, so **0.8 is what ships now**.

| Configuration | recall@3 | MRR@3 | right meeting @1 | paraphrase recall@3 |
| --- | --- | --- | --- | --- |
| lexical only | 0.885 | 0.865 | 0.885 | 0.625 |
| hybrid, alpha=0.65 | 0.962 | 0.904 | 0.885 | 0.875 |
| **hybrid, alpha=0.80** | **1.000** | **0.923** | 0.885 | **1.000** |
| dense only | 1.000 | 0.955 | 0.962 | 1.000 |

The sweep keeps improving all the way to `alpha = 1.0` (pure dense), and that is
deliberately **not** what ships. Two reasons: 26 questions over 5 chunks is far too small
a sample to justify deleting a retrieval component, and the lexical half does a job this
metric cannot see - it is the only thing that still works when the embedding model fails
to load, and it matches exact tokens (names, ticket ids, error codes) that embeddings
blur together. `0.8` takes the measured win without betting the feature on it.

The other two were **reranking** (built, measured, left disabled) and **query
rewriting** (built, measured, turned on). Both are in the RAG section above. The pair is
the point: the same harness, applied honestly, sent one feature to production and one to
the shelf.

**Honest limits of this harness.** 26 questions over 5 chunks is enough to catch a
regression and to justify a weight change. It is not enough to certify a retrieval
architecture, and the corpus is synthetic. The CI gate runs lexical-only so it stays
deterministic and needs no model download; the dense and hybrid arms are run locally,
where the numbers are read by a person rather than by a threshold.

---

## Action items as state, not text

Action items live in their own table as well as in the summary JSON, and the two are
deliberately **not** duplicates:

- `summary_json` is the **immutable record of what the model extracted** from the
  transcript.
- `action_items` is the **mutable workflow state** on top of it.

Keeping both means "who did the model say owns this" and "who owns it now" stay
separable - which is the first question anyone auditing an AI-generated task list asks.
An `edited` flag marks every row a human has touched, and the UI shows it. Editing a
task never rewrites the summary.

Migration 6 creates the table and **backfills it from every existing summary** with
JSON1, so meetings processed before the feature existed arrive already editable rather
than stranded as text in a blob.

`PATCH` rather than `PUT`: a client that only wants to tick a checkbox should not have
to send the whole row back and risk clobbering a field someone else just changed. The
ownership check lives inside the `UPDATE ... WHERE` clause rather than in a preceding
read, because a check-then-write is a race, and that race would let one user edit
another's task.

---

## The insights dashboard

Every number is a SQL aggregation. Workload and completion read the normalised
`action_items` table, so the charts follow what the team decided rather than only what
the model first extracted; topics and counts use SQLite's JSON1 to walk into
`summary_json` rather than loading every meeting into Python and counting in a loop. That is not only tidier: the loop costs one full transcript read per
meeting, so it degrades exactly as a user accumulates the meetings that make the
dashboard worth looking at.

Three choices worth explaining:

- **`Unassigned` is a bucket, not a gap.** It is usually the most actionable row on the
  page, because it is the work nobody has picked up - so it is charted alongside the
  named owners and given its own panel, rather than filtered out.
- **Days with no meetings are filled in.** Otherwise three meetings three weeks apart
  draw as three adjacent points and read as a busy week.
- **Charts are inline SVG built with element attributes**, not a charting library. It
  keeps the no-build-step, no-`node_modules` story intact and needs no CSP exception.

Scope, stated honestly: this is SQL group-by plus visualisation. It is a real product
feature and a fair "I can model and query data" talking point. It is not data science
and is not presented as such.

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
  analytics.py     # cross-meeting SQL aggregations for the dashboard
  reranker.py      # optional cross-encoder second stage (ships disabled - see below)
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
tests/             # 577 tests, providers mocked, no network
evaluation/        # labelled Q/A set + retrieval metrics + the ablation runner
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
| `GET`  | `/api/v1/meetings/{id}/action-items` | the editable action items |
| `PATCH` | `/api/v1/action-items/{id}` | reassign, re-date, reword, or complete one |
| `GET`  | `/api/v1/meetings` | paginated list (`?limit=&offset=`) |
| `GET`  | `/api/v1/meetings/{id}` | full meeting; also the **status polling** endpoint |
| `GET`  | `/api/v1/meetings/{id}/audio` , `/export` | stream audio , Markdown export |
| `DELETE` | `/api/v1/meetings/{id}` | delete meeting + audio + chunks |
| `POST` | `/api/v1/chat` | grounded Q&A (optional `meeting_id` scope) |
| `POST` | `/api/v1/chat/stream` | the same answer as a Server-Sent Events stream |
| `GET`  | `/api/v1/analytics` | cross-meeting aggregates (`?days=` window) |
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

1. **Postgres + pgvector + an ANN index**, once the corpus outgrows brute-force cosine,
   plus Redis so the rate limiter and job queue are shared across processes. Needs
   Docker or a hosted Postgres to develop against - it is not written blind.
2. **Long-audio chunking** with overlap to exceed the 25 MB single-file cap (needs
   ffmpeg).
3. **Speaker diarization** - label "Speaker 1 -> Priya". Deliberately last: the usable
   options need torch, which would undo the small-footprint story the rest of the
   project is built on.

Done: [x] auth + per-user isolation &nbsp; [x] background transcription with polling
&nbsp; [x] measured retrieval with a CI regression gate &nbsp; [x] insights dashboard
&nbsp; [x] streamed answers &nbsp; [x] reranking (measured, shipped off)
&nbsp; [x] query rewriting (measured, shipped on) &nbsp; [x] editable action items.

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
- **Action items are per-meeting, with no cross-meeting deduplication.** The same task
  agreed in two meetings is two rows. Merging them needs a similarity judgement the app
  does not currently make.
- **`due` is free text**, exactly as the model produced it ("Next Friday", "end of the
  month"). Parsing it into real dates would let the dashboard sort and alert on
  deadlines, and would also be the first place the app started guessing.
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
