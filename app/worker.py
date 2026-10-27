"""Enchaînement téléchargement, diarisation, extraction et transcription."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from app.config import PRECISE
from app.jobs import jobs, redact
from app.pipeline.download import download_audio
from app.pipeline.extract import render_preview, render_speaker
from app.pipeline.ffmpeg_util import to_wav
from app.pipeline.timeline import MERGE_GAP_SECONDS, PRECISE_MERGE_GAP_SECONDS

log = logging.getLogger("app")
inference_lock = threading.Lock()


def start_ingest(job_id: str) -> None:
    threading.Thread(
        target=_run_ingest,
        args=(job_id,),
        name=f"ingest-{job_id[:8]}",
        daemon=True,
    ).start()


def start_extract(
    job_id: str,
    speaker_ids: list[str],
    include_preroll: bool = True,
) -> None:
    threading.Thread(
        target=_run_extract,
        args=(job_id, speaker_ids, include_preroll),
        name=f"extract-{job_id[:8]}",
        daemon=True,
    ).start()


def start_transcribe(job_id: str) -> None:
    threading.Thread(
        target=_run_transcribe,
        args=(job_id,),
        name=f"transcribe-{job_id[:8]}",
        daemon=True,
    ).start()


def _run_ingest(job_id: str) -> None:
    try:
        _ingest(job_id)
    except Exception as exc:
        log.exception("ingest %s", job_id)
        jobs.fail(job_id, redact(str(exc)))


def _ingest(job_id: str) -> None:
    job = jobs.get(job_id)
    directory = jobs.directory(job_id)
    if job["source_type"] == "youtube":
        jobs.update(job_id, status="downloading", message="Téléchargement de l'audio…", error="")
        log.info("job %s téléchargement", job_id)
        _source, title = download_audio(job["url"], directory)
        jobs.update(job_id, title=title)
        source = _source
    else:
        jobs.update(job_id, status="downloading", message="Préparation du fichier…", error="")
        source = Path(job["upload_path"])
        log.info("job %s préparation fichier", job_id)

    if not jobs.exists(job_id):
        return
    wav = directory / "work.wav"
    to_wav(source, wav)
    if not jobs.exists(job_id):
        return

    jobs.update(
        job_id,
        status="diarizing",
        message="Analyse des voix… Cela peut prendre plusieurs minutes.",
        error="",
    )
    log.info("job %s diarisation", job_id)
    from app.pipeline.diarize import diarize

    with inference_lock:
        if not jobs.exists(job_id):
            return
        speakers, _grouping = diarize(wav, job["num_speakers"])

    for speaker in speakers:
        if not jobs.exists(job_id):
            return
        start, end = _longest(speaker["turns"])
        render_preview(wav, start, end, directory / f"preview_{speaker['id']}.mp3")

    jobs.update(
        job_id,
        status="ready",
        message="Choisissez une voix.",
        speakers=speakers,
        grouping="voices",
        error="",
    )
    log.info("job %s prêt, %s voix", job_id, len(speakers))


def _run_extract(job_id: str, speaker_ids: list[str], include_preroll: bool = True) -> None:
    job = jobs.get(job_id)
    chosen = [
        item for speaker_id in speaker_ids
        for item in job["speakers"]
        if item["id"] == speaker_id
    ]
    if len(chosen) != len(speaker_ids):
        return
    fallback = "done" if job["has_output"] else "ready"
    output_id = speaker_ids[0] if len(speaker_ids) == 1 else "selection"
    try:
        directory = jobs.directory(job_id)
        wav = directory / "work.wav"
        if not wav.is_file():
            raise RuntimeError("Le fichier audio de travail est introuvable. Relancez l'analyse.")
        turns = [
            (float(start), float(end))
            for speaker in chosen
            for start, end in speaker["turns"]
        ]
        turns.sort()
        dest = directory / f"voice_{output_id}.mp3"
        log.info("job %s extraction %s", job_id, ",".join(speaker_ids))
        render_speaker(
            wav,
            turns,
            dest,
            include_preroll=include_preroll,
            merge_gap=PRECISE_MERGE_GAP_SECONDS if PRECISE else MERGE_GAP_SECONDS,
        )
        transcript = directory / f"transcript_{output_id}.txt"
        transcript.unlink(missing_ok=True)
        if not jobs.exists(job_id):
            return
        if len(chosen) == 1:
            done_message = f"Le fichier de {chosen[0]['label']} est prêt."
        else:
            done_message = "Le fichier des répliques cochées est prêt."
        jobs.update(
            job_id,
            status="done",
            message=done_message,
            selected_speaker=output_id,
            selected_speakers=speaker_ids,
            has_output=True,
            transcript="",
            detected_language="",
            language_label="",
            output_revision=int(job.get("output_revision") or 0) + 1,
            error="",
        )
    except Exception as exc:
        log.exception("extract %s", job_id)
        if not jobs.exists(job_id):
            return
        current = jobs.get(job_id)
        status = "done" if current["has_output"] else fallback
        message = redact(str(exc))
        jobs.update(job_id, status=status, error=message, message="L'extraction a échoué.")


def _run_transcribe(job_id: str) -> None:
    job = jobs.get(job_id)
    speaker_id = job.get("selected_speaker")
    try:
        directory = jobs.directory(job_id)
        audio = directory / f"voice_{speaker_id}.mp3"
        if not audio.is_file():
            raise RuntimeError("Extrayez d'abord une voix.")
        log.info("job %s transcription", job_id)
        from app.pipeline.transcribe import transcribe

        with inference_lock:
            if not jobs.exists(job_id):
                return
            text, language, label = transcribe(audio)
        if not jobs.exists(job_id):
            return
        (directory / f"transcript_{speaker_id}.txt").write_text(text, encoding="utf-8")
        jobs.update(
            job_id,
            status="done",
            message="Transcription prête.",
            transcript=text,
            detected_language=language,
            language_label=label,
            error="",
        )
    except Exception as exc:
        log.exception("transcribe %s", job_id)
        if not jobs.exists(job_id):
            return
        jobs.update(
            job_id,
            status="done",
            error=redact(str(exc)),
            message="La transcription a échoué.",
        )


def _longest(turns: list[list[float]]) -> tuple[float, float]:
    start, end = max(turns, key=lambda turn: float(turn[1]) - float(turn[0]))
    return float(start), float(end)
