# 🎙️ MeetSaransh - Meeting Summarizer

Transcribe meeting audio and generate **action-oriented** summaries a clean transcript,
a layered summary, and a table of action items with owners, due dates, and timestamps
that jump back to the exact moment in the recording.

> _Saransh_ (सारांश) is Hindi for "summary".

**Pipeline:** `audio → validate → transcribe (Whisper) → summarize (LLM) → store → view`

---

## Why this is scoped the way it is

The assignment asks for one thing done well: **audio -> transcript -> summary + action items**,
graded on _transcription accuracy, summary quality, LLM prompt effectiveness, and code
structure_. The submission guidelines add a hard constraint: **"keep dependencies minimal
and native whenever possible."**

So this project deliberately does **not** ship a vector database, RAG chat, speaker
diarization, or multi-user auth in v1. Those are interesting (see [Roadmap](#roadmap)), but
none appear in the grading criteria and each would drag in heavy dependencies that work
against the guidelines. Instead, the effort goes into the four things that _are_ graded.
Every dependency in [`requirements.txt`](requirements.txt) is annotated with why it's there;
storage and JSON use the Python standard library on purpose.

---

## Features

- **Upload audio** (`.mp3 .wav .m4a .mp4 .webm .ogg .flac …`) and get a transcript + summary.
- **Layered summary** — TL;DR → key decisions → action items → open questions → topic
  timeline. Depth, not one flat paragraph.
- **Grounded action items** — each with an owner (or `Unassigned`), a due date (or
  `Not specified`), and a `[mm:ss]` timestamp. The prompt is instructed **not to invent**
  owners or dates.
- **Click-to-seek** — every timestamp in the summary and transcript seeks the audio player.
- **Transcript search** with match highlighting.
- **Export** the summary as copy-ready Markdown (a follow-up email / notes doc in one click).
- **Runs with no API key** — a bundled sample meeting demonstrates the whole UI offline.

---

## Quick start

**Prerequisites:** Python 3.10+ and a free [Groq API key](https://console.groq.com/keys)
(no credit card). The app also runs **without** a key using the sample meeting.

```bash
# 1. Install dependencies (a virtualenv is recommended)
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS/Linux
pip install -r requirements.txt

# 2. Add your key (optional - skip to run in sample-only mode)
copy .env.example .env            # Windows   (cp on macOS/Linux)
# then paste your key into GROQ_API_KEY=...

# 3. Run
python run.py
```

Open **http://127.0.0.1:8000**. Click **Load sample meeting** to see it work immediately,
or upload your own audio once a key is set.

---

## How it maps to the evaluation criteria

| Graded on | Where it's addressed |
| --- | --- |
| **Transcription accuracy** | Groq `whisper-large-v3-turbo` with segment timestamps ([`transcription.py`](app/transcription.py)) |
| **Summary quality** | Layered structure with decisions, owners, due dates, open questions ([`prompts.py`](app/prompts.py)) |
| **LLM prompt effectiveness** | Prompt is a first-class artifact: strict JSON contract, grounding rules, no-invention guardrails, `temperature=0.2` |
| **Code structure** | Clear separation: config · prompts · ASR · summarize · storage · routes; typed, documented, tiny dependency list |

---

## Architecture

```
Browser (vanilla HTML/JS, no build step)
        │  fetch()
        ▼
FastAPI (app/main.py) - routes, upload validation, Markdown export
        │
        ├── app/transcription.py  → Groq Whisper  (ASR, segment timestamps)
        ├── app/summarize.py      → Groq Llama    (structured JSON summary)
        ├── app/prompts.py        → prompt engineering (isolated & documented)
        └── app/storage.py        → SQLite via stdlib (single-file DB)
```

```
MeetSaransh/
├── app/
│   ├── main.py            # FastAPI routes + processing pipeline
│   ├── config.py          # env/config, limits, paths
│   ├── transcription.py   # ASR provider call + timestamp formatting
│   ├── summarize.py       # LLM call + defensive JSON parsing
│   ├── prompts.py         # the prompt (graded artifact)
│   └── storage.py         # SQLite persistence (stdlib)
├── static/                # index.html, style.css, app.js (no framework)
├── data/sample/           # bundled sample meeting (offline demo)
├── requirements.txt       # 5 deps, each annotated
├── .env.example
└── run.py
```

### API

| Method | Route | Purpose |
| --- | --- | --- |
| `GET`  | `/api/health` | status, model names, whether a key is set |
| `POST` | `/api/meetings` | upload audio → transcribe → summarize → store |
| `POST` | `/api/meetings/sample` | create the bundled sample meeting (no key needed) |
| `GET`  | `/api/meetings` | list meetings |
| `GET`  | `/api/meetings/{id}` | full meeting (transcript + segments + summary) |
| `GET`  | `/api/meetings/{id}/audio` | stream the stored audio |
| `GET`  | `/api/meetings/{id}/export` | summary as Markdown |
| `DELETE` | `/api/meetings/{id}` | delete meeting + its audio |

---

## Design decisions (and the trade-offs)

Each choice below was made by asking "what does the assignment actually reward?" rather than
"what's the most impressive stack?"

- **Python + FastAPI + vanilla JS, no frontend framework.** ASR and the LLM are just HTTP
  calls, so a heavy SPA earns nothing. A framework-free frontend means **zero `node_modules`
  and no build artifacts** - exactly what the guidelines ask for.
- **SQLite via the standard library, not a hosted DB or an ORM.** Satisfies "backend to
  store & process data" with a real relational store, zero extra dependencies, and no
  external service to stand up. Trade-off: single-writer, not built for high concurrency —
  fine for this scope.
- **Synchronous processing with a loading state, not a job queue.** A background worker +
  polling would be more "production", but adds moving parts and failure modes for no benefit
  in a single-user local demo. Groq's Whisper is fast. _(Async is a roadmap item, not a
  pretense.)_
- **Grounding over cleverness in the prompt.** `temperature=0.2`, explicit "don't invent
  owners/dates", and empty-list-if-absent rules. A summary that quietly fabricates an owner
  is worse than one that says `Unassigned`.
- **No ffmpeg / no audio chunking in v1.** Both mean a heavy dependency or a system binary.
  Files are validated and capped at 25 MB (the provider's limit) with a clear message
  instead. Chunking longer audio is a roadmap item.

---

## Security & robustness notes

- API key is read from `.env` (git-ignored) and used **server-side only** — never exposed to
  the browser.
- Uploads are validated by **extension and size** before any provider call.
- Provider errors are mapped to actionable messages (401 → check key, 429 → rate limit,
  413 → file too large) rather than surfaced raw.
- LLM JSON is parsed **defensively** (fence-stripping + outermost-object fallback), and every
  summary field is normalized so the UI never breaks on a malformed response.

---

## Roadmap

Deferred on purpose - out of scope for the graded criteria and/or the "minimal dependencies"
guideline, but the natural next steps:

- **Cross-meeting RAG chat** - "what did we decide about pricing?" across all meetings. Can
  be built lean (an embeddings API + a numpy/SQLite vector store) without a hosted vector DB.
- **Speaker diarization** - label "Speaker 1 -> Priya". High perceived-quality gain; costs a
  heavy model dependency, so deferred.
- **Async transcription queue** with progress + retry-with-backoff on 429s.
- **Long-audio chunking** with overlap to exceed the 25 MB single-file cap.
- **Multi-user auth** with per-user meeting isolation.

---

## Limitations

- Diarization is not included, so the transcript is not speaker-labelled.
- Single file per upload, ≤ 25 MB (~40 min of typical audio).
- SQLite + local file storage: designed for a single-user local run, not a deployed service.

---

## Tech stack

Python · FastAPI · Uvicorn · httpx · SQLite (stdlib) · vanilla JS/HTML/CSS ·
Groq (Whisper `large-v3-turbo` + Llama `3.3-70b-versatile`).
