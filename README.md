# 🎙️ MeetSaransh - Meeting Summarizer

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

---

## Quick start

**Prerequisites:** Python 3.10+ and a free [Groq API key](https://console.groq.com/keys)
(no credit card). The app also runs **without** a key using the sample meeting.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt

copy .env.example .env            # optional - paste your GROQ_API_KEY (cp on macOS/Linux)

python run.py
```

Open **http://127.0.0.1:8000** -> **Load sample meeting** -> try the **Ask your meetings** tab.
On first RAG use, a ~90 MB embedding model downloads once and is cached.

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
- **Two-layer grounding.** A cheap similarity gate hard-refuses clearly-unrelated questions;
  for everything else the LLM prompt is the real guard - it may answer *only* from the
  retrieved excerpts and must say so when they don't contain the answer.
- **Query-aware citations.** Each source shows the most relevant segment (with its own
  timestamp) and links straight into the transcript at that moment.
- **Storage:** embeddings are stored as `float32` blobs in SQLite; retrieval is brute-force
  cosine in numpy. At this scale that's correct and simple - an ANN index (HNSW/IVFFlat) is
  the scaling path, not a demo requirement.

---

## Architecture

```
Browser (vanilla HTML/JS, no build step) -- Meetings view + Ask (chat) view
        |  fetch()
        v
FastAPI (app/main.py) -- routes, upload validation, Markdown export
        |
        +-- app/transcription.py  -> Groq Whisper  (ASR, segment timestamps)
        +-- app/summarize.py      -> structured JSON summary
        +-- app/rag.py            -> chunk . hybrid retrieve . grounded answer
        +-- app/embeddings.py     -> fastembed bge-small (ONNX, lazy-loaded)
        +-- app/llm.py            -> shared Groq chat client (retry/backoff)
        +-- app/prompts.py        -> prompt engineering (summary + RAG)
        +-- app/storage.py        -> SQLite: meetings + chunks (stdlib)
```

```
app/
  main.py          # routes + pipelines
  config.py        # env, limits, RAG tunables
  transcription.py # ASR + timestamp formatting
  summarize.py     # summary + defensive JSON parsing
  rag.py           # chunking, BM25, hybrid retrieval, grounded answer
  prompts.py       # graded prompt artifacts (summary + RAG)
  storage.py       # SQLite (meetings + chunks)
  llm.py           # shared chat client + retry/backoff
  embeddings.py    # local dense embeddings (fastembed)
static/            # index.html, style.css, app.js  (no framework)
data/sample/       # bundled sample meeting (offline demo)
```

### API

| Method | Route | Purpose |
| --- | --- | --- |
| `GET`  | `/api/health` | status, model names, whether a key is set |
| `POST` | `/api/meetings` | upload audio -> transcribe -> summarize -> store -> index |
| `POST` | `/api/meetings/sample` | create the bundled sample (no key needed) |
| `GET`  | `/api/meetings` , `/api/meetings/{id}` | list , full meeting |
| `GET`  | `/api/meetings/{id}/audio` , `/export` | stream audio , Markdown export |
| `DELETE` | `/api/meetings/{id}` | delete meeting + audio + chunks |
| `POST` | `/api/chat` | grounded Q&A over meetings (optional `meeting_id` scope) |
| `POST` | `/api/reindex` | index any meetings not yet in the vector store |
| `GET`  | `/api/rag/status` | embeddings availability, indexed meeting/chunk counts |

---

## Design decisions (and the trade-offs)

- **Python + FastAPI + vanilla JS, no frontend framework** - ASR/LLM are just HTTP calls, so
  a heavy SPA earns nothing. Zero `node_modules`, no build step.
- **SQLite via the standard library** for both meetings and the vector store - a real
  relational DB, no hosted service, no ORM. Trade-off: single-writer, not built for high
  concurrency - fine for this scope.
- **`fastembed` (ONNX) over `sentence-transformers` (torch)** for embeddings - CPU-friendly,
  no heavy `torch` dependency, no extra API key (Groq has no embeddings endpoint), works
  offline. The model downloads once and caches.
- **Hybrid retrieval, not pure-vector** - better recall on conversational text, and it
  degrades to lexical-only if embeddings are unavailable.
- **LLM as the real grounding layer** - the similarity gate can't separate "loosely related"
  from "off-topic" reliably (bge-small compresses those into one score band), so the prompt
  does the honest refusing. Documented rather than pretended-away.
- **Shared `llm.py` with retry-and-backoff on 429** - the free-tier rate limits make this a
  real requirement, not decoration.
- **Synchronous processing with a loading state**, not a job queue - fewer failure modes for
  a single-user app. Async is a roadmap item, not a pretense.
- **No ffmpeg / no audio chunking in v1** - both mean a heavy dependency or a system binary;
  files are validated and capped at 25 MB (the provider's limit) instead.

---

## Security & robustness notes

- API key is read from `.env` (git-ignored) and used **server-side only** - never sent to
  the browser.
- Uploads are validated by extension and size before any provider call.
- Provider errors map to actionable messages (401 -> check key, 429 -> rate limit,
  413 -> too large); the LLM client retries transient 429s with exponential backoff.
- LLM JSON is parsed defensively and every summary field is normalized, so a malformed
  response never breaks the UI.
- RAG indexing is best-effort: if it fails, the meeting is still saved and viewable.

---

## Roadmap

- **Async transcription queue** with progress + richer retry semantics.
- **Long-audio chunking** with overlap to exceed the 25 MB single-file cap.
- **Speaker diarization** - label "Speaker 1 -> Priya".
- **Multi-user auth** with per-user meeting/vector isolation.
- **ANN vector index** (HNSW/IVFFlat) once the corpus outgrows brute-force cosine.

---

## Limitations

- Diarization is not included, so transcripts aren't speaker-labelled.
- Single file per upload, <= 25 MB (~40 min of typical audio).
- SQLite + local files: designed for a single-user local run, not a deployed service.
- The RAG refusal gate is deliberately lenient; the LLM prompt does the final grounding, so
  a written refusal for off-topic questions requires an API key.

---

## Tech stack

Python . FastAPI . Uvicorn . httpx . SQLite (stdlib) . fastembed (bge-small, ONNX) . numpy .
vanilla JS/HTML/CSS . Groq (Whisper `large-v3-turbo` + Llama `3.3-70b-versatile`).
