"""Serveur local : voix YouTube ou fichier audio, sans envoi des données."""

from __future__ import annotations

import logging
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import HF_TOKEN, MAX_UPLOAD_BYTES
from app.jobs import JobNotFound, jobs, redact, to_public
from app.validate import (
    ValidationError,
    audio_extension,
    clean_title,
    parse_num_speakers,
    validate_speaker_id,
    validate_youtube_url,
)
from app.worker import start_extract, start_ingest, start_transcribe

STATIC = Path(__file__).resolve().parent / "static"
LOCAL_CLIENTS = {"127.0.0.1", "::1", "testclient"}
log = logging.getLogger("app")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if shutil.which("ffmpeg") is None:
        log.error("ffmpeg est introuvable.")
    yield


app = FastAPI(title="Extraction de voix", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class SpeakerBody(BaseModel):
    speaker: str = ""
    speakers: list[str] = []
    preroll: bool = True


@app.middleware("http")
async def local_only(request, call_next):
    client = request.client.host if request.client else ""
    if client not in LOCAL_CLIENTS:
        return JSONResponse({"detail": "Accès local uniquement."}, status_code=403)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = "default-src 'self'; media-src 'self'; style-src 'self'; script-src 'self'"
    return response


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "hf_token_configured": bool(HF_TOKEN),
    }


@app.post("/api/jobs", status_code=202)
async def create_job(
    url: str | None = Form(default=None),
    num_speakers: str | None = Form(default=None),
    file: UploadFile | None = File(default=None),
) -> dict:
    has_file = file is not None and bool(file.filename)
    has_url = bool(url and url.strip())
    if has_file and has_url:
        raise HTTPException(status_code=400, detail="Indiquez un lien ou un fichier, pas les deux.")
    if not has_file and not has_url:
        raise HTTPException(
            status_code=400,
            detail="Collez un lien YouTube ou choisissez un fichier audio.",
        )
    try:
        parsed_url = validate_youtube_url(url or "") if has_url else None
        speaker_count = parse_num_speakers(num_speakers)
        extension = audio_extension(file.filename or "") if has_file else ""
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None

    title = clean_title(file.filename or "") if has_file else "Vidéo YouTube"
    job = jobs.create(
        source_type="upload" if has_file else "youtube",
        title=title,
        num_speakers=speaker_count,
        url=parsed_url,
    )
    try:
        if has_file and file is not None:
            dest = jobs.directory(job["id"]) / f"upload{extension}"
            await _save_upload(file, dest)
            jobs.update(job["id"], upload_path=str(dest), title=title)
        start_ingest(job["id"])
    except Exception as exc:
        jobs.delete(job["id"])
        raise HTTPException(status_code=400, detail=redact(str(exc))) from None
    return {"id": job["id"]}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    return to_public(_job_or_404(job_id))


@app.delete("/api/jobs/{job_id}", status_code=204)
def delete_job(job_id: str) -> None:
    try:
        jobs.delete(job_id)
    except JobNotFound:
        raise HTTPException(status_code=404, detail="Travail introuvable.") from None


@app.get("/api/jobs/{job_id}/preview/{speaker_id}")
def preview(job_id: str, speaker_id: str) -> FileResponse:
    return _media(job_id, speaker_id, "preview", inline=True)


@app.post("/api/jobs/{job_id}/extract")
def extract(job_id: str, body: SpeakerBody) -> dict:
    requested = body.speakers or ([body.speaker] if body.speaker else [])
    if not requested:
        raise HTTPException(status_code=400, detail="Choisissez au moins une réplique.")
    if len(requested) > 40:
        raise HTTPException(status_code=400, detail="Quarante répliques au maximum.")
    try:
        speaker_ids = []
        for raw in requested:
            speaker_id = validate_speaker_id(raw)
            if speaker_id not in speaker_ids:
                speaker_ids.append(speaker_id)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    job = _job_or_404(job_id)
    known = {item["id"]: item for item in job["speakers"]}
    chosen = [known[speaker_id] for speaker_id in speaker_ids if speaker_id in known]
    if len(chosen) != len(speaker_ids):
        raise HTTPException(status_code=404, detail="Voix inconnue.")
    if len(chosen) == 1:
        progress = f"Extraction de {chosen[0]['label']}…"
    else:
        progress = "Assemblage des répliques cochées…"
    begun = jobs.try_begin(job_id, {"ready", "done"}, "extracting", progress)
    if begun is None:
        raise HTTPException(status_code=409, detail="Une étape est déjà en cours.")
    start_extract(job_id, speaker_ids, body.preroll)
    return to_public(jobs.get(job_id))


@app.get("/api/jobs/{job_id}/output")
def output(job_id: str, dl: int = 0) -> FileResponse:
    job = _job_or_404(job_id)
    speaker_id = job.get("selected_speaker") or ""
    try:
        validate_speaker_id(speaker_id)
    except ValidationError:
        raise HTTPException(status_code=404, detail="Aucun fichier audio.") from None
    return _media(job_id, speaker_id, "voice", inline=not dl, download_name="voix.mp3")


@app.post("/api/jobs/{job_id}/transcribe")
def transcribe(job_id: str) -> dict:
    job = _job_or_404(job_id)
    if not job["has_output"] or not job.get("selected_speaker"):
        raise HTTPException(status_code=400, detail="Extrayez d'abord une voix.")
    begun = jobs.try_begin(
        job_id,
        {"done"},
        "transcribing",
        "Transcription du fichier final… Le modèle se charge la première fois.",
    )
    if begun is None:
        raise HTTPException(status_code=409, detail="Une étape est déjà en cours.")
    start_transcribe(job_id)
    return to_public(jobs.get(job_id))


@app.get("/api/jobs/{job_id}/transcript")
def transcript(job_id: str) -> FileResponse:
    job = _job_or_404(job_id)
    speaker_id = job.get("selected_speaker") or ""
    try:
        validate_speaker_id(speaker_id)
    except ValidationError:
        raise HTTPException(status_code=404, detail="Aucune transcription.") from None
    path = _contained(job_id, f"transcript_{speaker_id}.txt")
    return FileResponse(path, media_type="text/plain; charset=utf-8", filename="transcription.txt")


def _job_or_404(job_id: str) -> dict:
    try:
        return jobs.get(job_id)
    except JobNotFound:
        raise HTTPException(status_code=404, detail="Travail introuvable.") from None


def _contained(job_id: str, name: str) -> Path:
    try:
        directory = jobs.directory(job_id)
    except JobNotFound:
        raise HTTPException(status_code=404, detail="Travail introuvable.") from None
    path = (directory / name).resolve()
    if path.parent != directory.resolve() or not path.is_file():
        raise HTTPException(status_code=404, detail="Fichier introuvable.")
    return path


def _media(
    job_id: str,
    speaker_id: str,
    kind: str,
    *,
    inline: bool,
    download_name: str | None = None,
) -> FileResponse:
    try:
        validate_speaker_id(speaker_id)
    except ValidationError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    path = _contained(job_id, f"{kind}_{speaker_id}.mp3")
    filename = download_name or path.name
    return FileResponse(
        path,
        media_type="audio/mpeg",
        filename=filename,
        content_disposition_type="inline" if inline else "attachment",
    )


async def _save_upload(upload: UploadFile, dest: Path) -> None:
    total = 0
    try:
        with dest.open("wb") as handle:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise ValueError(
                        f"Fichier trop volumineux (maximum {MAX_UPLOAD_BYTES // (1024 * 1024)} Mo)."
                    )
                handle.write(chunk)
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    if total == 0:
        dest.unlink(missing_ok=True)
        raise ValueError("Le fichier est vide.")
