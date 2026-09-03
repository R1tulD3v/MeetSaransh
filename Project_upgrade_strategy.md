# MeetSaransh - Project Upgrade Strategy

> A brutally honest, grounded analysis of the current project and a practical roadmap to
> make it interview-strong for SDE, backend, full-stack, and applied-AI roles.
>
> Every claim is tagged **FACT** (verified in this repo), **INFERENCE** (reasoned from the
> code), or **RECOMMENDATION** (a proposed change). Nothing here is invented.

---

## 0. Read this first: the honest framing

**Brutal-honesty item #1 - the outcome may not be about the project.**
Per the earlier research on Unthinkable, their process is heavily weighted toward a live
machine-coding round (DP, greedy, strings/hashmaps on HackerEarth). A project is one input.
Do not assume "not selected" == "project too weak." Fix the project *and* drill DSA.

**Brutal-honesty item #2 - you cannot make one meeting-summarizer credibly serve SDE +
backend + AI + ML + data-analyst + data-engineer all at once.** Attempting that produces a
project that is shallow in six directions. The honest, higher-EV strategy:

- **Make it a standout full-stack + applied-AI (RAG) project.** That is what it already is
  closest to, and that is a hot lane right now.
- Add **one genuine analytics dashboard** for a light "data" talking point.
- Do **NOT** pretend it is an ML project (there is no model training) or a data-analyst
  project (there is no dataset analysis). Claiming those in an interview invites questions
  you cannot answer and reads as padding. If you want ML/data-analyst roles, build a
  *separate* project for them. This document says so repeatedly on purpose.

**Brutal-honesty item #3 - the current biggest weakness is not features, it is the absence
of "production thinking" artifacts:** no tests, no CI, no container, no deployment, no auth,
no observability. That is exactly the signal companies like Unthinkable screen for, and it is
100% within your control to fix. Features are secondary to this.

---

## 1. Current project DNA analysis

### 1.0 What the project is (FACT)

MeetSaransh is a meeting-summarizer web app:
- Upload audio -> transcribe (Groq Whisper `large-v3-turbo`) -> structured summary +
  action items (Groq `openai/gpt-oss-120b`) -> store -> view.
- "Ask your meetings": cross-meeting RAG chat with hybrid retrieval (dense `bge-small` via
  fastembed + pure-Python BM25), grounded answers, deep-linkable citations.
- Backend: FastAPI. Storage: SQLite (stdlib) for meetings + a `chunks` table holding
  `float32` embedding blobs, retrieved with brute-force cosine in numpy.
- Frontend: vanilla HTML/CSS/JS, no framework, no build step.
- Dependencies (FACT, from `requirements.txt`): fastapi, uvicorn, httpx, python-multipart,
  python-dotenv, fastembed, numpy.

### 1.1 Layer-by-layer analysis

Each layer: what exists, what's strong, what's weak/missing, interview questions it already
supports, and the upgrade that raises its value most.

---

#### Frontend

- **Exists (FACT):** Single-page vanilla JS (`static/app.js`, ~350 lines), two views
  (Meetings, Ask), tabs (Summary/Transcript), click-to-seek audio, transcript search with
  highlight, chat UI with clickable citations, toast notifications, light/dark via
  `prefers-color-scheme`, responsive grid.
- **Strong:** No build step, no `node_modules` - clean and fast. Genuinely thoughtful UX
  (citations deep-link into the transcript; grounded-refusal is surfaced honestly). DOM code
  is XSS-careful (uses `textContent`/`el()` helpers, escapes search highlight).
- **Weak / missing (FACT):** No framework (fine, but no React/state-management talking
  point). No component tests. No accessibility pass (ARIA, focus management, keyboard nav).
  No optimistic UI / error boundaries. No loading skeletons beyond a spinner. No
  build/bundle/minify story. No client-side routing. No form-library validation.
- **Interview questions it already supports:** "How do you prevent XSS when injecting user
  content?" (answer: `textContent`, escaping). "How does click-to-seek work?" (timestamp ->
  seconds -> `audio.currentTime`). "Why no framework?" (HTTP-call app, avoid build weight).
- **Highest-value upgrade:** Rebuild the frontend in **React + TypeScript + Vite**, with a
  proper data-fetching layer (TanStack Query), component tests (Vitest + Testing Library),
  and an accessibility pass. This is the single biggest resume gap for full-stack roles.
- **Maturity: intermediate.**

---

#### Backend

- **Exists (FACT):** FastAPI app (`app/main.py`) with ~10 endpoints, a clean module split
  (config, transcription, summarize, rag, embeddings, llm, prompts, storage), Pydantic-typed
  request handling via FastAPI, upload validation (extension + 25 MB size), provider-error
  mapping (401/429/413), a shared `llm.py` chat client with **retry + exponential backoff**
  on 429.
- **Strong:** Genuinely clean separation of concerns. Error handling is above average for a
  student project. Retry/backoff is a real production pattern. Best-effort indexing (a RAG
  failure never fails the main request) shows defensive thinking.
- **Weak / missing (FACT):** Everything is **synchronous** - a long transcription blocks the
  request/worker. No background jobs/queue. No pagination on `/api/meetings`. No request
  validation via explicit Pydantic response models (returns raw dicts). No API versioning.
  No rate limiting. No idempotency on uploads. No config validation on startup (a missing
  key fails at call time, not boot time). `on_event("startup")` is deprecated (should be
  lifespan handlers).
- **Interview questions it already supports:** "How do you handle provider rate limits?"
  (backoff). "What happens if summarization fails after transcription succeeds?" (audio
  cleaned up, 502 returned). "Why a shared llm.py?" (DRY, one place for transport/retry).
- **Highest-value upgrade:** **Async job queue** for transcription (Celery/RQ + Redis, or
  FastAPI `BackgroundTasks` + a status table) with a `status` field and polling endpoint.
  This unlocks the entire "system design" conversation.
- **Maturity: intermediate (leaning strong on code structure, weak on async/scale).**

---

#### Database

- **Exists (FACT):** SQLite via stdlib `sqlite3`. Two tables: `meetings` and `chunks`
  (with a `float32` embedding blob + `segs_json`). Manual schema via `executescript`, a
  per-process init guard, cascade-delete of chunks in application code, an index on
  `chunks.meeting_id`.
- **Strong:** Real relational store, zero external service, sensible indexing, JSON columns
  used pragmatically. Vector storage in SQLite is a legitimately interesting design choice.
- **Weak / missing (FACT):** No migrations (schema changes require deleting the DB - you hit
  this during development). No ORM (fine, but no SQLAlchemy talking point). No connection
  pooling (SQLite is single-writer). No foreign-key enforcement (SQLite needs
  `PRAGMA foreign_keys=ON`; the FK is declared but not enforced). No transactions spanning
  operations. Vector search is O(n) brute force - correct but not scalable.
- **Interview questions it already supports:** "How are embeddings stored?" (`float32` blobs,
  numpy `frombuffer`). "Why SQLite?" (single-user, no service). "Index choice?" (btree on
  meeting_id).
- **Highest-value upgrade:** Migrate to **Postgres + pgvector** with **Alembic migrations**
  and **SQLAlchemy**. This gives you: real migrations, an ANN index (HNSW/IVFFlat) with a
  *defensible reason*, connection pooling, and the "SQL vs NoSQL / index tuning" interview
  surface. Keep SQLite as the default for local dev.
- **Maturity: intermediate.**

---

#### APIs

- **Exists (FACT):** RESTful-ish resource design (`/api/meetings`, `/api/meetings/{id}`,
  `/api/chat`, `/api/reindex`, `/api/rag/status`). FastAPI auto-generates OpenAPI docs at
  `/docs`. Correct-ish status codes (201 on create, 404, 400, 413, 502).
- **Strong:** Free OpenAPI/Swagger. Reasonable resource modeling. Chat endpoint accepts a
  scope param.
- **Weak / missing (FACT):** No explicit Pydantic **response models** (so the OpenAPI schema
  for responses is weak). No pagination/filtering/sorting. No API versioning (`/api/v1`). No
  consistent error envelope. No `PATCH` for editing (e.g., editing action items). No content
  negotiation. `/api/chat` uses a loose `dict` body instead of a typed model.
- **Interview questions it already supports:** "REST design for this domain?" "Status code
  choices?" "How is the API documented?" (OpenAPI).
- **Highest-value upgrade:** Add **typed Pydantic request/response models** everywhere,
  **pagination**, an **error envelope**, and **API versioning**. Cheap, high-polish, very
  visible in `/docs`.
- **Maturity: intermediate.**

---

#### Authentication / Authorization

- **Exists (FACT):** **Nothing.** No login, no users, no sessions, no tokens. Every meeting
  is global; anyone hitting the server sees all meetings.
- **Strong:** N/A.
- **Weak / missing (FACT):** No user model, no auth, no RBAC, no per-user data isolation, no
  password handling, no OAuth, no session/JWT.
- **Interview questions it already supports:** None. This is a hole.
- **Highest-value upgrade:** **JWT auth + user accounts + row-level ownership** so User A
  cannot read User B's meetings. Optionally add **roles** (admin/user) for an RBAC story.
  This is explicitly on Unthinkable's requirement list and is the #1 "production thinking"
  gap. Most candidate demos skip auth - shipping it is a differentiator.
- **Maturity: beginner (absent).**

---

#### Cloud / Infrastructure

- **Exists (FACT):** **Nothing deployed.** Runs locally via `python run.py` on 127.0.0.1.
  No cloud, no container, no reverse proxy, no object storage (audio is on local disk).
- **Weak / missing (FACT):** No Dockerfile, no deployment, no environment separation
  (dev/staging/prod), no secrets manager, no CDN, no object store for audio, no HTTPS story.
- **Interview questions it already supports:** Almost none.
- **Highest-value upgrade:** **Dockerize** (multi-stage), then **deploy** (Render/Fly.io/
  Railway free tier) with a public URL. A live demo link on your resume is worth more than
  three unshipped features. Move audio to **object storage** (S3/R2) with signed URLs.
- **Maturity: beginner (absent).**

---

#### DevOps / CI-CD

- **Exists (FACT):** **Nothing.** No `.github/workflows`, no pipeline, no automation.
- **Weak / missing (FACT):** No CI (lint/test/build on push), no CD, no pre-commit hooks, no
  dependency scanning, no release process.
- **Interview questions it already supports:** None.
- **Highest-value upgrade:** A **GitHub Actions pipeline**: install -> lint (ruff) ->
  type-check (mypy) -> test (pytest) -> build Docker image. Add a **status badge** to the
  README. This is a cheap, extremely legible "I understand CI" signal.
- **Maturity: beginner (absent).**

---

#### Security

- **Exists (FACT):** API key server-side only (read from `.env`, git-ignored - verified not
  leaked). Upload validation (extension + size). XSS-careful DOM code. Provider-error
  messages don't leak internals. **Note (FACT):** a stray `app/.env` duplicate of the key
  exists locally - git-ignored and NOT committed, but should be deleted for hygiene.
- **Strong:** Secret handling and input validation basics are correct.
- **Weak / missing (FACT):** No auth (so no authz). No rate limiting (DoS/cost-abuse risk on
  the paid ASR/LLM endpoints). No CORS policy configured. No security headers (CSP, HSTS).
  No file-content validation (only extension - a `.mp3` could be anything). No request size
  limits at the server layer. No audit logging. No dependency vulnerability scanning.
- **Interview questions it already supports:** "How do you keep API keys out of the client?"
  "How do you validate uploads?"
- **Highest-value upgrade:** **Rate limiting** (slowapi/Redis) + **auth** + **security
  headers** + **magic-byte file validation** + **Dependabot/pip-audit** in CI. Rate limiting
  is especially defensible: "these endpoints call paid APIs, so I cap them per-user/IP."
- **Maturity: intermediate on secrets/validation, beginner on everything else.**

---

#### Testing

- **Exists (FACT):** **No committed tests.** (Smoke tests were run ad hoc via inline Python
  during development but never committed as a suite.) No `tests/` dir, no pytest config.
- **Weak / missing (FACT):** No unit tests, no integration tests, no API tests, no frontend
  tests, no coverage reporting, no test fixtures, no mocking of the Groq provider.
- **Interview questions it already supports:** None (and "do you write tests?" would be
  awkward).
- **Highest-value upgrade:** A **pytest suite** with the Groq calls **mocked** (so tests run
  offline/deterministically): unit tests for chunking, BM25, the refusal gate, JSON
  normalization; integration tests for every endpoint via `TestClient`; a coverage gate in
  CI (e.g., 80%). This is arguably the highest ROI upgrade in the whole document.
- **Maturity: beginner (absent).**

---

#### Observability

- **Exists (FACT):** Effectively none. Errors surface as HTTP responses; stdout goes to a log
  file only because you redirected it. No structured logging, no metrics, no tracing.
- **Weak / missing (FACT):** No structured logs (JSON), no request IDs, no latency metrics,
  no `/metrics` (Prometheus), no error tracking (Sentry), no dashboards, no health-check
  depth (the health endpoint doesn't check the DB or provider).
- **Interview questions it already supports:** None.
- **Highest-value upgrade:** **Structured logging** (structlog) with request IDs + **timing
  middleware** that records per-endpoint latency, plus a **Prometheus `/metrics`** endpoint
  and a tiny Grafana dashboard (or at least logged latency numbers). You already have real
  latency data to show ("60-min audio -> transcript in Xs").
- **Maturity: beginner (absent).**

---

#### AI / ML features

- **Exists (FACT):** Applied AI, done well: ASR (Whisper), structured summarization with a
  grounded, JSON-constrained prompt, and a **RAG pipeline** with hybrid retrieval (dense +
  BM25), segment-aware chunking, contextual-header embedding, a two-layer grounding/refusal
  design, and deep-linkable citations.
- **Strong:** The RAG design is genuinely thoughtful and defensible - hybrid retrieval,
  grounded refusals, citation precision, offline degradation. This is your best interview
  material.
- **Weak / missing (FACT):** **No ML** in the training sense - no model trained, evaluated,
  or served. No RAG **evaluation** (no recall@k, no faithfulness/answer-quality metrics). No
  reranking. No query rewriting/expansion. No streaming responses. No prompt/version
  tracking. No speaker diarization (so answers can't attribute who said what).
- **Interview questions it already supports:** "Explain your chunking strategy." "Dense vs
  lexical - why hybrid?" "How do you prevent hallucination?" "How did you pick the refusal
  threshold?" (you have a real empirical answer). These are strong.
- **Highest-value upgrade:** A **RAG evaluation harness** (a small labelled Q/A set + metrics
  like retrieval recall@k and LLM-as-judge faithfulness) - this converts "I built RAG" into
  "I built RAG and measured it," which is the difference between junior and credible.
  Then **streaming** answers (SSE) and **speaker diarization**.
- **Maturity: strong (applied AI); absent (ML/eval).**

---

#### Data analysis features

- **Exists (FACT):** None. No aggregation, no dashboard, no trends, no analytics.
- **Weak / missing (FACT):** No cross-meeting analytics (action-item completion, decisions
  over time, talk-time, topic frequency), no charts, no exportable reports.
- **Interview questions it already supports:** None.
- **Highest-value upgrade:** An **analytics dashboard**: action items by owner/status,
  meetings over time, most-frequent topics, open vs resolved decisions - computed with SQL
  aggregations and rendered as charts. This is a *light but honest* data story (SQL
  group-bys + visualization), not a fake "data science" claim.
- **Maturity: beginner (absent).**

---

#### Performance

- **Exists (FACT):** Lazy-loaded embedding model (fast startup). Brute-force cosine (fine at
  small scale). Synchronous request handling.
- **Weak / missing (FACT):** No caching (embeddings recomputed per query for the model load;
  transcripts re-fetched). No pagination. No connection pooling. O(n) vector scan. No CDN for
  static assets. No compression. No async I/O for provider calls (httpx is used sync).
- **Interview questions it already supports:** "Where's the bottleneck?" (transcription
  latency; vector scan at scale). "How would you scale retrieval?" (ANN index).
- **Highest-value upgrade:** **Caching** (Redis) for embeddings/query results and repeated
  reads, **async provider calls**, and an **ANN index** once on Postgres/pgvector. Add
  **measured numbers** to the README.
- **Maturity: intermediate.**

---

#### Code quality

- **Exists (FACT):** Clean module boundaries, type hints throughout, docstrings that explain
  *why*, defensive parsing, consistent naming. No linter/formatter/type-checker configured
  (FACT: no ruff/black/mypy/pyproject).
- **Strong:** The code reads well and is well-commented. Separation of concerns is real.
- **Weak / missing (FACT):** No enforced formatting (black/ruff), no linting in CI, no static
  type-checking (mypy), no pre-commit, no docstring/style consistency enforcement, some
  deprecated patterns (`on_event`).
- **Highest-value upgrade:** Add **ruff + black + mypy + pre-commit** and wire them into CI.
  Cheap, and it makes the "I care about code quality" claim verifiable.
- **Maturity: strong (by hand), beginner (by tooling).**

---

#### Scalability

- **Exists (FACT):** None by design - single-user, single-process, SQLite, local disk,
  synchronous, brute-force vector search.
- **Highest-value upgrade:** The whole "scaling story": stateless app + Postgres/pgvector +
  Redis + object storage + job queue + horizontal scaling behind a load balancer. You do not
  have to *build* all of it, but you must be able to *narrate* it, and building 2-3 pieces
  (queue, Postgres, object storage) makes the narration credible.
- **Maturity: beginner.**

---

#### Maintainability

- **Exists (FACT):** Good module structure, good comments, small dependency surface, clear
  README. No migrations, no tests, no CI - so *changing* it safely is hard (you already felt
  this: a schema change meant deleting the DB).
- **Highest-value upgrade:** Tests + migrations + CI. Maintainability is mostly a function of
  those three.
- **Maturity: intermediate.**

### 1.2 Maturity summary (one line per layer)

| Layer | Maturity |
| --- | --- |
| Frontend | intermediate |
| Backend | intermediate (strong structure, weak async/scale) |
| Database | intermediate |
| APIs | intermediate |
| Auth / Authz | beginner (absent) |
| Cloud / Infra | beginner (absent) |
| DevOps / CI-CD | beginner (absent) |
| Security | intermediate (secrets/validation), beginner (rest) |
| Testing | beginner (absent) |
| Observability | beginner (absent) |
| AI / ML | strong (applied AI), beginner (ML/eval) |
| Data analysis | beginner (absent) |
| Performance | intermediate |
| Code quality | strong by hand, beginner by tooling |
| Scalability | beginner |
| Maintainability | intermediate |

**Overall: a strong applied-AI + clean-backend student project, sitting at "intermediate"
overall, held back from "production-like" by the total absence of tests, CI, auth,
deployment, and observability.**

---

## 2. Missing pieces (gap analysis)

Ranked by how much each gap hurts *for this project type* (full-stack + applied AI).

| # | Gap | Present? | Severity for this project | Why it matters |
| --- | --- | --- | --- | --- |
| 1 | **Automated tests** | FACT: none | Critical | "Do you test?" has no good answer. Blocks safe iteration. |
| 2 | **Deployment / live URL** | FACT: none | Critical | An unshipped project is half a project. A link beats features. |
| 3 | **Auth + per-user isolation** | FACT: none | Critical | Explicitly on Unthinkable's list; the top "production thinking" gap. |
| 4 | **CI/CD** | FACT: none | High | Cheap to add, strong signal, gates quality automatically. |
| 5 | **Containerization (Docker)** | FACT: none | High | Table stakes for backend/DevOps credibility; enables deploy. |
| 6 | **Observability (logs/metrics)** | FACT: none | High | "How do you debug prod?" needs structured logs + metrics. |
| 7 | **Rate limiting** | FACT: none | High | Endpoints call *paid* APIs - abuse/cost risk. Very defensible. |
| 8 | **Async job processing** | FACT: sync | High | Long transcriptions block; this is the core system-design story. |
| 9 | **RAG evaluation** | FACT: none | Medium-High | Turns "I built RAG" into "I measured RAG." Rare, credible. |
| 10 | **Analytics/dashboard** | FACT: none | Medium | The only realistic "data" talking point for this project. |
| 11 | **Migrations** | FACT: none | Medium | Safe schema evolution; you already hit this pain. |
| 12 | **Typed API models + pagination** | FACT: partial | Medium | Cheap API polish, visible in `/docs`. |
| 13 | **Caching** | FACT: none | Medium | Performance story; Redis is a resume keyword. |
| 14 | **Frontend framework (React/TS)** | FACT: vanilla | Medium | Biggest gap specifically for full-stack roles. |
| 15 | **Speaker diarization** | FACT: none | Low-Medium | Quality jump; the live answer literally couldn't name Rahul. |

**Which gaps matter most for THIS project type:** 1, 2, 3, 4, 5 (the "production thinking"
cluster) first; then 6, 7, 8 (backend robustness); then 9, 10 (differentiators). Frontend
framework (14) matters only if you are specifically targeting full-stack/frontend roles.

**Explicitly NOT worth chasing for this project:** a real ML training pipeline, a
data-engineering pipeline (Airflow/Spark), or a "data science" narrative. They do not fit a
meeting-summarizer and would read as bolted-on. Build a separate project if you want those.

---

## 3. Best upgrade ideas (detailed)

Each idea includes: what it does, why it matters, role, market relevance, interview value,
complexity, time, dependencies, risk, demo value, and level.

> Scale notes: Complexity/Level in {beginner, medium, advanced}. Time assumes a focused
> student working part-time.

### 3.1 Test suite with mocked providers (pytest)
- **Does:** Unit tests (chunking, BM25, refusal gate, JSON normalization, timestamp fmt) +
  API integration tests via `TestClient`, with Groq HTTP calls mocked (respx/monkeypatch).
  Coverage reported and gated in CI.
- **Why:** Safe iteration, and the single most common "is this person real?" screen.
- **Role:** SDE, Backend, Full-stack, QA. **Market relevance:** universal.
- **Interview value:** Very high. **Complexity:** medium. **Time:** 1-2 days.
- **Dependencies:** pytest, respx/pytest-mock, coverage. **Risk:** low. **Demo value:**
  medium (green CI badge). **Level:** medium.

### 3.2 CI/CD with GitHub Actions
- **Does:** On push/PR: ruff + mypy + pytest + build Docker image; badge in README.
- **Why:** Verifiable quality; DevOps literacy.
- **Role:** SDE, Backend, DevOps. **Market relevance:** universal.
- **Interview value:** High. **Complexity:** beginner-medium. **Time:** 0.5-1 day.
- **Dependencies:** 3.1 (tests), 3.5 (Docker) ideally. **Risk:** low. **Demo:** medium.
  **Level:** medium.

### 3.3 JWT auth + per-user data isolation (+ optional RBAC)
- **Does:** User signup/login, hashed passwords (bcrypt/argon2), JWT access/refresh, and
  row-level ownership so users only see their own meetings. Optional admin role.
- **Why:** #1 production-thinking gap; explicitly on Unthinkable's list.
- **Role:** Backend, Full-stack, Security. **Market relevance:** universal.
- **Interview value:** Very high. **Complexity:** medium. **Time:** 2-3 days.
- **Dependencies:** DB user table, migrations. **Risk:** medium (security correctness).
  **Demo:** high (login flow + isolation). **Level:** medium-advanced.

### 3.4 Async transcription with a job queue + status polling
- **Does:** Upload returns immediately with a job id; a worker (RQ/Celery + Redis, or
  `BackgroundTasks` + a `status` column) processes ASR->summary->index; frontend polls
  `/api/meetings/{id}` for `queued|processing|done|error` and shows progress.
- **Why:** The core system-design story; removes the blocking-request flaw.
- **Role:** Backend, SDE, DevOps. **Market relevance:** high.
- **Interview value:** Very high. **Complexity:** advanced. **Time:** 2-4 days.
- **Dependencies:** Redis (or a DB-backed queue). **Risk:** medium. **Demo:** high (live
  progress bar). **Level:** advanced.

### 3.5 Dockerize + deploy to a public URL
- **Does:** Multi-stage Dockerfile, docker-compose (app + Redis + Postgres), deploy to
  Render/Fly.io/Railway; public HTTPS URL.
- **Why:** A live link is the highest-signal single artifact you can put on a resume.
- **Role:** Backend, DevOps, Full-stack. **Market relevance:** universal.
- **Interview value:** High. **Complexity:** medium. **Time:** 1-2 days.
- **Dependencies:** env/secrets config. **Risk:** medium (cold starts, model download size).
  **Demo:** very high. **Level:** medium.

### 3.6 Postgres + pgvector + Alembic migrations (SQLAlchemy)
- **Does:** Swap SQLite for Postgres; store vectors in pgvector with an HNSW/IVFFlat index;
  Alembic migrations; SQLAlchemy models. Keep SQLite for local dev.
- **Why:** Real DB story: migrations, ANN index choice with a reason, connection pooling.
- **Role:** Backend, Data engineer, SDE. **Market relevance:** high.
- **Interview value:** High. **Complexity:** advanced. **Time:** 2-3 days.
- **Dependencies:** Postgres, SQLAlchemy, Alembic. **Risk:** medium. **Demo:** low-medium.
  **Level:** advanced.

### 3.7 Rate limiting + security headers + magic-byte validation
- **Does:** Per-user/IP rate limits (slowapi/Redis) on the paid endpoints; CSP/HSTS headers;
  validate file *content* (magic bytes), not just extension; CORS policy.
- **Why:** Cost/DoS protection on paid APIs; concrete security posture.
- **Role:** Security, Backend. **Market relevance:** high.
- **Interview value:** High (very defensible narrative). **Complexity:** medium. **Time:**
  1 day. **Dependencies:** Redis for distributed limits (optional). **Risk:** low. **Demo:**
  medium. **Level:** medium.

### 3.8 Structured logging + metrics (observability)
- **Does:** structlog JSON logs with request IDs; latency-timing middleware; Prometheus
  `/metrics`; optional Sentry; a deeper health check (DB + provider reachability).
- **Why:** "How do you debug/operate this in prod?" - the SRE conversation.
- **Role:** DevOps/SRE, Backend. **Market relevance:** high.
- **Interview value:** High. **Complexity:** medium. **Time:** 1-2 days.
- **Dependencies:** structlog, prometheus-client. **Risk:** low. **Demo:** medium (a Grafana
  panel is a nice screenshot). **Level:** medium.

### 3.9 RAG evaluation harness
- **Does:** A small hand-labelled Q/A set over the sample + a few meetings; measure retrieval
  recall@k / MRR and answer faithfulness (LLM-as-judge); a script that prints a scorecard;
  compare hybrid vs dense-only vs lexical-only.
- **Why:** Converts "I built RAG" into "I measured and tuned RAG." Rare at student level.
- **Role:** AI engineer, ML-adjacent, Backend. **Market relevance:** very high (RAG is hot).
- **Interview value:** Very high. **Complexity:** medium-advanced. **Time:** 1-2 days.
- **Dependencies:** a labelled set; an LLM judge. **Risk:** low. **Demo:** high (a metrics
  table + an ablation chart). **Level:** advanced.

### 3.10 Streaming chat responses (SSE) + query rewriting + reranking
- **Does:** Stream tokens to the UI (Server-Sent Events); rewrite/expand the user query
  before retrieval; add a lightweight cross-encoder rerank of top-k.
- **Why:** Modern RAG UX + retrieval-quality depth.
- **Role:** AI engineer, Full-stack. **Market relevance:** high.
- **Interview value:** High. **Complexity:** advanced. **Time:** 2-3 days.
- **Dependencies:** SSE plumbing; a reranker model (adds weight). **Risk:** medium. **Demo:**
  high. **Level:** advanced.

### 3.11 Analytics dashboard
- **Does:** SQL aggregations (action items by owner/status, meetings over time, top topics,
  open vs resolved decisions) rendered as charts; export a report.
- **Why:** The honest "data" talking point (SQL group-bys + viz), plus real product value.
- **Role:** Full-stack, Data analyst (light), Backend. **Market relevance:** medium-high.
- **Interview value:** Medium-high. **Complexity:** medium. **Time:** 1-2 days.
- **Dependencies:** a charting lib; more sample data. **Risk:** low. **Demo:** high. **Level:**
  medium.

### 3.12 React + TypeScript + Vite frontend rebuild
- **Does:** Rebuild the UI as a typed component app with TanStack Query, component tests
  (Vitest + Testing Library), accessibility pass, and a real build pipeline.
- **Why:** The biggest gap specifically for full-stack/frontend roles.
- **Role:** Full-stack, Frontend. **Market relevance:** high.
- **Interview value:** High (for FE/full-stack). **Complexity:** advanced. **Time:** 3-5
  days. **Dependencies:** Node toolchain. **Risk:** medium (scope creep). **Demo:** high.
  **Level:** advanced.

### 3.13 Editable/assignable action items (+ PATCH API + export to email/Markdown)
- **Does:** Make action items editable, assignable, and completable; persist edits; export a
  follow-up email/Markdown. "Close the loop" from summary to action.
- **Why:** Real product thinking; adds a write-path/CRUD story to a mostly read-path app.
- **Role:** Full-stack, Product engineer, Backend. **Market relevance:** medium.
- **Interview value:** Medium. **Complexity:** medium. **Time:** 1-2 days. **Dependencies:**
  a `tasks` table. **Risk:** low. **Demo:** high. **Level:** medium.

### 3.14 Speaker diarization
- **Does:** Label "Speaker 1 -> Priya" so the transcript and answers attribute who said what.
- **Why:** Big perceived-quality jump; your live RAG answer literally couldn't name Rahul.
- **Role:** AI engineer. **Market relevance:** medium. **Interview value:** medium.
  **Complexity:** advanced (heavy dep: pyannote/torch, or an API). **Time:** 2-3 days.
  **Risk:** medium-high (dependency weight, CPU cost). **Demo:** high. **Level:** advanced.

### 3.15 Long-audio chunking (beat the 25 MB cap)
- **Does:** Split long/large audio with overlap, transcribe pieces, stitch timestamps.
- **Why:** Removes the main functional limitation.
- **Role:** Backend, AI. **Market relevance:** medium. **Interview value:** medium.
  **Complexity:** medium (needs ffmpeg/pydub). **Time:** 1-2 days. **Risk:** medium (system
  binary dep). **Demo:** medium. **Level:** medium.

---

## 4. Priority ranking table (scoring model)

**Scoring:** each dimension 1-5. `Value = Interview + Market + Demo + TechDepth + Resume +
LongTerm` (max 30). `Effort` 1-5 (5 = hardest). Verdict blends high Value with low Effort,
but treats **security and testing as mandatory** regardless of raw rank.

| Rank | Category | Upgrade | Interview | Market | Demo | TechDepth | Resume | LongTerm | **Value/30** | Effort/5 | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Testing | Pytest suite + mocks (3.1) | 5 | 5 | 3 | 4 | 5 | 5 | **27** | 2 | **Do now** |
| 2 | DevOps | CI/CD (3.2) | 4 | 5 | 3 | 3 | 5 | 5 | **25** | 2 | **Do now** |
| 3 | Cloud | Docker + deploy (3.5) | 4 | 5 | 5 | 3 | 5 | 4 | **26** | 3 | **Do now** |
| 4 | Security | Rate limit + headers + magic-byte (3.7) | 4 | 4 | 3 | 4 | 5 | 4 | **24** | 2 | **Do now** |
| 5 | Backend/Auth | JWT auth + row-level isolation (3.3) | 5 | 5 | 4 | 4 | 5 | 4 | **27** | 3 | **Do now** |
| 6 | Backend | Async job queue + status (3.4) | 5 | 4 | 5 | 5 | 5 | 4 | **28** | 4 | **Do next** |
| 7 | AI | RAG evaluation harness (3.9) | 5 | 5 | 4 | 5 | 5 | 4 | **28** | 3 | **Do next** |
| 8 | Observability | Structured logs + metrics (3.8) | 4 | 4 | 3 | 4 | 4 | 5 | **24** | 3 | **Do next** |
| 9 | Database | Postgres + pgvector + Alembic (3.6) | 4 | 4 | 2 | 5 | 5 | 5 | **25** | 4 | **Do next** |
| 10 | Data | Analytics dashboard (3.11) | 4 | 4 | 5 | 3 | 4 | 3 | **23** | 3 | **Do next** |
| 11 | AI | Streaming + rewrite + rerank (3.10) | 4 | 5 | 5 | 5 | 4 | 3 | **26** | 4 | **Do later** |
| 12 | Frontend | React + TS rebuild (3.12) | 4 | 4 | 5 | 4 | 4 | 4 | **25** | 5 | **Do later** (if targeting FE) |
| 13 | Product | Editable action items + PATCH (3.13) | 3 | 3 | 5 | 3 | 3 | 3 | **20** | 2 | **Do later** |
| 14 | API | Typed models + pagination + versioning | 3 | 4 | 2 | 3 | 4 | 4 | **20** | 2 | **Do later** |
| 15 | AI | Speaker diarization (3.14) | 3 | 3 | 5 | 4 | 3 | 3 | **21** | 4 | **Do later** |
| 16 | Backend | Long-audio chunking (3.15) | 3 | 3 | 3 | 3 | 3 | 3 | **18** | 3 | **Skip/optional** |

**Effort-vs-impact reading:** rows 1-5 are the highest impact for the lowest effort - do them
first. Rows 6-10 are the "resume builders." Rows 11-16 are polish/differentiators once the
foundation is solid.

---

## 5. Feature bundles

Bundling matters because interviewers probe *coherent stories*, not scattered features.

### Bundle A - "Production-thinking foundation" (do this first)
- **Features:** Pytest suite (3.1) + CI/CD (3.2) + Docker + deploy (3.5) + rate limiting &
  security headers (3.7).
- **Goal:** Turn a local script into a shipped, tested, automated service.
- **Why together:** Tests feed CI; CI builds the Docker image; Docker enables deploy; rate
  limiting protects the deployed paid endpoints. One clean narrative.
- **Makes it interview-ready for:** SDE, Backend, DevOps. **This bundle alone fixes the
  biggest weakness in the project.**

### Bundle B - "Backend robustness"
- **Features:** JWT auth + isolation (3.3) + async job queue (3.4) + Postgres/pgvector +
  Alembic (3.6) + observability (3.8).
- **Goal:** A multi-user, horizontally-scalable, observable backend.
- **Why together:** Auth needs a user table (migrations); the queue needs a status column
  (migrations); observability watches the queue; Postgres/pgvector replaces SQLite for
  concurrency. This is the "system design" bundle.
- **Interview-ready for:** Backend, SDE (system design), DevOps.

### Bundle C - "Applied-AI depth"
- **Features:** RAG evaluation harness (3.9) + streaming + query rewrite + rerank (3.10) +
  speaker diarization (3.14).
- **Goal:** Move from "I used an LLM" to "I engineered and measured a retrieval system."
- **Why together:** Eval tells you whether rewrite/rerank actually help (you can show an
  ablation); diarization improves the excerpts the whole pipeline reasons over.
- **Interview-ready for:** AI engineer, applied-ML, Backend-with-AI.

### Bundle D - "Data & product"
- **Features:** Analytics dashboard (3.11) + editable/assignable action items (3.13) + export.
- **Goal:** Close the loop from "captured" to "actionable," plus a light data story.
- **Why together:** Editable action items generate the state (owner/status) the dashboard
  aggregates. Product + data in one narrative.
- **Interview-ready for:** Full-stack, Product engineer, light Data-analyst.

### Bundle E - "Frontend showcase" (only if targeting FE/full-stack)
- **Features:** React + TS + Vite rebuild (3.12) + component tests + a11y + the dashboard's
  charts.
- **Goal:** A typed, tested, accessible modern frontend.
- **Interview-ready for:** Frontend, Full-stack.

---

## 6. Phase-wise roadmap

### Phase 1 - Quick wins (make it *credible*) - ~4-6 days
- **Build:** Pytest suite (3.1), CI/CD (3.2), Docker + deploy (3.5), rate limiting + security
  headers + magic-byte validation (3.7), delete the stray `app/.env`, add ruff/black/mypy +
  pre-commit.
- **Why this phase:** These are cheap, universally expected, and fix the project's biggest
  weakness (no tests/CI/deploy/security). A live URL + green CI badge changes first
  impressions instantly.
- **Skills learned:** pytest + mocking, GitHub Actions, Docker, deployment, web security
  basics, linting/type-checking.
- **Interview benefit:** You can now answer "do you test / deploy / secure your work?" with
  evidence. **Demo benefit:** a public link and a passing pipeline.

### Phase 2 - Strong resume builders (make it *robust*) - ~1-1.5 weeks
- **Build:** JWT auth + per-user isolation (3.3), async job queue + status polling (3.4),
  structured logging + metrics (3.8).
- **Why:** These create the multi-user + system-design + observability stories that separate
  intermediate from strong.
- **Skills:** auth/security, queues/Redis, background workers, structured logging, metrics.
- **Interview benefit:** system-design conversations you can drive. **Demo benefit:** login +
  isolation + a live progress bar + a metrics panel.

### Phase 3 - Standout features (make it *differentiated*) - ~1-1.5 weeks
- **Build:** RAG evaluation harness (3.9), analytics dashboard (3.11), Postgres + pgvector +
  Alembic (3.6).
- **Why:** RAG-with-metrics is rare and impressive; the dashboard adds the data story;
  Postgres/pgvector gives the real DB + ANN-index conversation.
- **Skills:** retrieval evaluation, LLM-as-judge, SQL aggregation + viz, migrations, vector
  indexing.
- **Interview benefit:** "I measured my RAG and tuned it with an ablation" is a standout
  line. **Demo benefit:** a metrics scorecard + charts.

### Phase 4 - Advanced / production-grade - ~2+ weeks (optional, targeted)
- **Build:** Streaming + query rewrite + rerank (3.10), React + TS frontend (3.12), speaker
  diarization (3.14), editable action items (3.13).
- **Why:** Depth and polish once the foundation is solid; pick based on the role you target
  (FE rebuild only for full-stack/FE; diarization/streaming for AI roles).
- **Skills:** SSE/streaming, rerankers, React/TS/testing, diarization pipelines.
- **Interview benefit:** senior-flavored depth. **Demo benefit:** streaming answers, a polished
  typed UI, who-said-what transcripts.

---

## 7. Security, testing, and code quality per upgrade (mandatory cross-cutting)

For each major upgrade: how to test it, what security risk it adds, how to reduce that risk,
what refactor keeps the code clean, and what to log/measure.

### Auth (3.3)
- **Test:** unit-test password hashing + token issue/verify; integration-test that User A
  gets 403/404 on User B's meeting; test expired/invalid tokens.
- **Security risk:** weak hashing, token leakage, no expiry, enumeration via error messages.
- **Reduce:** argon2/bcrypt, short-lived access + rotating refresh tokens, generic error
  messages, `PRAGMA foreign_keys` / FK constraints, rate-limit login (brute force).
- **Refactor:** a `deps.py` `get_current_user` dependency; a `user_id` column + query filter
  in `storage.py`; never trust a client-supplied user id.
- **Log/measure:** login success/failure counts, 401/403 rates, token-refresh rate.

### Async job queue (3.4)
- **Test:** unit-test the worker task with mocked providers; test state transitions
  (queued->processing->done/error); test a failed job leaves a clean error state.
- **Security risk:** unbounded queue (DoS/cost), a worker that trusts unvalidated input.
- **Reduce:** per-user job quotas, max queue depth, validate before enqueue, dead-letter
  handling.
- **Refactor:** a `jobs` table + a `worker.py`; keep the HTTP handler thin (enqueue + return
  id).
- **Log/measure:** queue depth, job duration, failure rate, retries.

### Deployment / Docker (3.5)
- **Test:** a smoke test that hits `/api/health` in the built container in CI; test that the
  image starts with no `.env` and degrades to sample mode.
- **Security risk:** secrets baked into the image, running as root, exposed debug.
- **Reduce:** secrets via env/secret manager (never in the image), non-root user, minimal
  base image, no `--reload` in prod, pinned deps.
- **Refactor:** multi-stage build; `.dockerignore` (exclude `.venv`, `data/`, `.env`).
- **Log/measure:** container start time, health-check status, cold-start latency.

### Rate limiting / security headers (3.7)
- **Test:** assert the Nth request in a window returns 429; assert headers present; feed a
  fake-extension file and assert rejection by magic bytes.
- **Security risk:** limiter bypass (spoofed IP), overly permissive CORS.
- **Reduce:** key limits by authenticated user (not just IP), strict CORS allowlist, CSP.
- **Refactor:** a middleware layer; centralize limits in config.
- **Log/measure:** 429 counts per user/endpoint, blocked-upload counts.

### RAG evaluation (3.9)
- **Test:** the eval script is itself a test - assert metrics stay above a threshold in CI
  (a regression gate on retrieval quality).
- **Security risk:** low; if using an LLM judge, don't send anything sensitive to it.
- **Reduce:** run eval on the sample/synthetic data only.
- **Refactor:** an `eval/` package with a labelled dataset + metric functions separate from
  app code.
- **Log/measure:** recall@k, MRR, faithfulness, per-config ablation results.

### Analytics dashboard (3.11)
- **Test:** unit-test each aggregation query against a seeded DB with known expected numbers.
- **Security risk:** aggregations leaking across users (if multi-user), SQL injection if any
  raw query building.
- **Reduce:** always filter by `user_id`; parameterized queries only.
- **Refactor:** an `analytics.py` module of pure query functions; charts in the frontend.
- **Log/measure:** dashboard query latency.

### Global code-quality guardrails (apply to everything)
- ruff (lint) + black (format) + mypy (types) + pre-commit, all enforced in CI.
- Replace deprecated `on_event` with lifespan handlers.
- Typed Pydantic request/response models on every endpoint.
- A consistent error envelope + centralized exception handlers.
- Coverage gate (start at 70-80%).

---

## 8. Interview value summary

**What this project lets you talk about *today* (FACT-grounded, already true):**
- RAG internals: hybrid retrieval, chunking with overlap, contextual-header embedding,
  grounded refusal, citation precision, offline degradation, an *empirically tuned* refusal
  threshold (you have the real numbers).
- Pragmatic engineering: shared LLM client with retry/backoff, defensive JSON parsing,
  best-effort indexing, upload validation, secrets kept server-side, XSS-careful DOM code,
  a deliberately small dependency footprint.
- Design trade-offs you can defend: SQLite vs hosted DB, vanilla JS vs framework, sync vs
  queue, fastembed/ONNX vs torch, brute-force cosine vs ANN.

**What you *cannot* yet talk about (and interviewers will ask):**
- Testing strategy, CI/CD, deployment/ops, auth/RBAC, observability, scaling, rate limiting,
  async processing, migrations - because none exist yet.

**The one-sentence pitch after Phases 1-3:** "A multi-user, containerized, tested, and
deployed meeting-intelligence service with an evaluated hybrid-RAG pipeline, background
transcription jobs, per-user data isolation, rate limiting, and a metrics/observability
layer."

---

## 9. Final recommended next 5 upgrades

If you do nothing else, do these five - highest impact for the effort, and they compose into
one coherent "I ship production software" story:

1. **Pytest suite with mocked Groq calls (3.1)** - fixes the most glaring gap; enables safe
   iteration and CI. *~1-2 days.*
2. **CI/CD via GitHub Actions (3.2)** - lint + type-check + test + build on every push; badge
   in README. *~0.5-1 day.*
3. **Docker + deploy to a public URL (3.5)** - a live link is the highest-signal single
   artifact you can add. *~1-2 days.*
4. **JWT auth + per-user data isolation (3.3)** - the top production-thinking gap and
   explicitly on Unthinkable's list. *~2-3 days.*
5. **RAG evaluation harness (3.9)** - converts your strongest existing feature from "I built
   it" into "I measured and tuned it," which is rare and credible. *~1-2 days.*

**Do these and the project moves from "intermediate applied-AI demo" to "production-like,
tested, deployed, multi-user AI service with a measured retrieval system."**

---

## Appendix A - Compact table version

| Priority | Upgrade | Category | Value/30 | Effort/5 | Verdict |
| --- | --- | --- | --- | --- | --- |
| 1 | Pytest suite + mocks | Testing | 27 | 2 | Do now |
| 2 | CI/CD (GitHub Actions) | DevOps | 25 | 2 | Do now |
| 3 | Docker + deploy | Cloud | 26 | 3 | Do now |
| 4 | Rate limit + security headers | Security | 24 | 2 | Do now |
| 5 | JWT auth + isolation | Backend/Auth | 27 | 3 | Do now |
| 6 | Async job queue + status | Backend | 28 | 4 | Do next |
| 7 | RAG evaluation harness | AI | 28 | 3 | Do next |
| 8 | Structured logs + metrics | Observability | 24 | 3 | Do next |
| 9 | Postgres + pgvector + Alembic | Database | 25 | 4 | Do next |
| 10 | Analytics dashboard | Data | 23 | 3 | Do next |
| 11 | Streaming + rewrite + rerank | AI | 26 | 4 | Do later |
| 12 | React + TS rebuild | Frontend | 25 | 5 | Do later (if FE) |
| 13 | Editable action items + PATCH | Product | 20 | 2 | Do later |
| 14 | Typed models + pagination | API | 20 | 2 | Do later |
| 15 | Speaker diarization | AI | 21 | 4 | Do later |
| 16 | Long-audio chunking | Backend | 18 | 3 | Optional |

## Appendix B - One-line "why this project becomes stronger"

> Because it stops being a clever local AI demo and becomes a tested, containerized, deployed,
> multi-user service with an *evaluated* RAG pipeline and real operational concerns - which is
> exactly the "production thinking over feature-count" signal that hiring managers (and
> Unthinkable specifically) screen for.

---

## Appendix C - What to explicitly NOT do (anti-recommendations)

- **Do not** bolt on a fake "ML model" (sentiment, "AI insights") with no training/eval - it
  reads as padding and invites questions you can't answer.
- **Do not** claim a data-analyst/data-engineering story from a single app's tables - build a
  separate project with a real dataset if you want those roles.
- **Do not** add breadth (10 half-features) over depth (3 solid, tested, deployed ones).
  Unthinkable's stated value is "engineering discipline over velocity theatre" - honor it.
- **Do not** rewrite the frontend in React *unless* you are specifically targeting
  frontend/full-stack roles; for backend/AI roles your effort is better spent on Phases 1-2.
