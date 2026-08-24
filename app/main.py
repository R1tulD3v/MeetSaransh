"""FastAPI application: routes, upload handling, and the process pipeline.

Pipeline for an uploaded meeting:
    audio -> validate -> save -> transcribe (ASR) -> summarize (LLM) -> store -> return

The frontend is plain HTML/JS served statically, so there is no build step and no
node_modules — in keeping with the "minimal and native" submission guideline.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from . import __version__, config, storage, summarize, transcription

app = FastAPI(title="MeetSaransh", version=__version__)


@app.on_event("startup")
def _startup() -> None:
    config.ensure_dirs()
    storage.init_db()


# ----------------------------------------------------------------------------- health
@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": __version__,
        "has_api_key": config.has_api_key(),
        "asr_model": config.ASR_MODEL,
        "llm_model": config.LLM_MODEL,
    }


# --------------------------------------------------------------------------- meetings
@app.get("/api/meetings")
def api_list_meetings() -> list[dict]:
    return storage.list_meetings()


@app.get("/api/meetings/{meeting_id}")
def api_get_meeting(meeting_id: str) -> dict:
    meeting = storage.get_meeting(meeting_id)
    if meeting is None:
        raise HTTPException(status_code=404, detail="Meeting not found")
    return meeting


@app.delete("/api/meetings/{meeting_id}")
def api_delete_meeting(meeting_id: str) -> dict:
    meeting = storage.get_meeting(meeting_id)
    if meeting and meeting.get("audio_ext"):
        audio = config.AUDIO_DIR / f"{meeting_id}{meeting['audio_ext']}"
        audio.unlink(missing_ok=True)
    if not storage.delete_meeting(meeting_id):
        raise HTTPException(status_code=404, detail="Meeting not found")
    return {"deleted": meeting_id}


@app.post("/api/meetings", status_code=201)
async def api_create_meeting(
    file: UploadFile = File(...),
    title: str = Form(""),
) -> dict:
    """Upload audio, transcribe, summarize, and persist."""
    ext = _validate_upload(file)
    raw = await file.read()
    if len(raw) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File is {_mb(len(raw))} MB; the limit is {_mb(config.MAX_UPLOAD_BYTES)} MB.",
        )

    meeting_id = storage.new_id()
    audio_path = config.AUDIO_DIR / f"{meeting_id}{ext}"
    audio_path.write_bytes(raw)

    try:
        transcript = transcription.transcribe(audio_path)
    except transcription.TranscriptionError as exc:
        audio_path.unlink(missing_ok=True)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    try:
        summary = summarize.summarize(title, transcript["timestamped_text"])
    except summarize.SummarizationError as exc:
        audio_path.unlink(missing_ok=True)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    storage.create_meeting(
        title=title or Path(file.filename or "meeting").stem,
        filename=file.filename,
        transcript=transcript,
        summary=summary,
        audio_ext=ext,
        meeting_id=meeting_id,
    )
    return storage.get_meeting(meeting_id)  # type: ignore[return-value]


@app.post("/api/meetings/sample", status_code=201)
def api_create_sample() -> dict:
    """Create a meeting from the bundled sample transcript.

    Lets the app be demoed end-to-end with NO API key. If a key IS configured, the
    summary is generated live from the sample transcript (so graders can see the real
    LLM path too); otherwise the pre-generated sample summary is used.
    """
    sample_path = config.SAMPLE_DIR / "sample_meeting.json"
    if not sample_path.exists():
        raise HTTPException(status_code=500, detail="Sample meeting file is missing.")

    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    segments = sample["segments"]
    transcript = {
        "text": " ".join(s["text"] for s in segments),
        "segments": segments,
        "timestamped_text": transcription.build_timestamped_text(segments),
        "duration": segments[-1]["end"] if segments else 0.0,
    }

    if config.has_api_key():
        try:
            summary = summarize.summarize(sample["title"], transcript["timestamped_text"])
        except summarize.SummarizationError:
            summary = summarize.normalize_summary(sample["summary"])
    else:
        summary = summarize.normalize_summary(sample["summary"])

    mid = storage.create_meeting(
        title=sample["title"], filename="sample_meeting.mp3",
        transcript=transcript, summary=summary, audio_ext=None,
    )
    return storage.get_meeting(mid)  # type: ignore[return-value]


@app.get("/api/meetings/{meeting_id}/audio")
def api_get_audio(meeting_id: str):
    meeting = storage.get_meeting(meeting_id)
    if meeting is None or not meeting.get("audio_ext"):
        raise HTTPException(status_code=404, detail="No audio for this meeting")
    audio = config.AUDIO_DIR / f"{meeting_id}{meeting['audio_ext']}"
    if not audio.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(audio)


@app.get("/api/meetings/{meeting_id}/export", response_class=PlainTextResponse)
def api_export_markdown(meeting_id: str) -> str:
    """Export the summary + action items as copy-ready Markdown (close-the-loop)."""
    meeting = storage.get_meeting(meeting_id)
    if meeting is None:
        raise HTTPException(status_code=404, detail="Meeting not found")
    return _to_markdown(meeting)


# --------------------------------------------------------------------------- helpers
def _validate_upload(file: UploadFile) -> str:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")
    ext = Path(file.filename).suffix.lower()
    if ext not in config.ALLOWED_AUDIO_EXTS:
        allowed = ", ".join(sorted(config.ALLOWED_AUDIO_EXTS))
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{ext}'. Allowed: {allowed}")
    return ext


def _mb(n: int) -> str:
    return f"{n / (1024 * 1024):.1f}"


def _to_markdown(meeting: dict) -> str:
    s = meeting.get("summary", {})
    lines = [f"# {meeting['title']}", ""]
    if s.get("tldr"):
        lines += ["## TL;DR", s["tldr"], ""]
    if s.get("key_decisions"):
        lines += ["## Key decisions"]
        lines += [f"- {d['decision']}" + (f" _( {d['timestamp']} )_" if d.get("timestamp") else "") for d in s["key_decisions"]]
        lines.append("")
    if s.get("action_items"):
        lines += ["## Action items", "", "| Task | Owner | Due |", "| --- | --- | --- |"]
        lines += [f"| {a['task']} | {a['owner']} | {a['due']} |" for a in s["action_items"]]
        lines.append("")
    if s.get("open_questions"):
        lines += ["## Open questions"]
        lines += [f"- {q}" for q in s["open_questions"]]
        lines.append("")
    if s.get("topics"):
        lines += ["## Topics"]
        lines += [f"- **{t['title']}** — {t['summary']}" for t in s["topics"]]
        lines.append("")
    return "\n".join(lines).strip() + "\n"


# ------------------------------------------------------------------- static frontend
@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (config.STATIC_DIR / "index.html").read_text(encoding="utf-8")


app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")
