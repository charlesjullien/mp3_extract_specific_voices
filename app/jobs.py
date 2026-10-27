"""État des travaux, en mémoire et dans data/."""

from __future__ import annotations

import copy
import json
import re
import shutil
import threading
from pathlib import Path

from app.config import DATA_DIR, HF_TOKEN

_JOB_ID = re.compile(r"^[0-9a-f]{32}$")
_ACTIVE = {"downloading", "diarizing", "extracting", "transcribing"}


class JobNotFound(KeyError):
    pass


def redact(message: str) -> str:
    text = (message or "").strip() or "Une erreur est survenue."
    if HF_TOKEN:
        text = text.replace(HF_TOKEN, "***")
    if len(text) > 400:
        text = text[:400].rstrip() + "…"
    return text


class JobStore:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()
        self.load()

    def load(self) -> None:
        if not self.data_dir.exists():
            return
        for directory in self.data_dir.iterdir():
            manifest = directory / "job.json"
            if not directory.is_dir() or not manifest.is_file():
                continue
            try:
                job = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(job, dict) or not _JOB_ID.fullmatch(str(job.get("id", ""))):
                continue
            if job.get("status") in _ACTIVE:
                voice = directory / f"voice_{job.get('selected_speaker')}.mp3"
                if job.get("has_output") and voice.is_file():
                    job["status"] = "done"
                    job["message"] = (
                        "Le traitement a été interrompu. Le dernier fichier audio est encore disponible."
                    )
                elif job.get("speakers"):
                    job["status"] = "ready"
                    job["message"] = (
                        "Le traitement a été interrompu. Vous pouvez relancer l'extraction."
                    )
                else:
                    job["status"] = "error"
                    job["error"] = "Le traitement a été interrompu. Relancez-le."
                    job["message"] = job["error"]
                self._write(directory / "job.json", job)
            self._jobs[job["id"]] = job

    def create(
        self,
        *,
        source_type: str,
        title: str,
        num_speakers: int | None,
        url: str | None,
    ) -> dict:
        import uuid

        job_id = uuid.uuid4().hex
        message = (
            "Téléchargement de l'audio…"
            if source_type == "youtube"
            else "Préparation du fichier…"
        )
        job = {
            "id": job_id,
            "status": "downloading",
            "message": message,
            "title": title,
            "source_type": source_type,
            "url": url,
            "num_speakers": num_speakers,
            "upload_path": "",
            "speakers": [],
            "grouping": "voices",
            "selected_speaker": None,
            "selected_speakers": [],
            "has_output": False,
            "transcript": "",
            "detected_language": "",
            "language_label": "",
            "output_revision": 0,
            "error": "",
        }
        with self._lock:
            directory = self.data_dir / job_id
            directory.mkdir(parents=True, exist_ok=False)
            self._jobs[job_id] = job
            self._write(directory / "job.json", job)
            return copy.deepcopy(job)

    def exists(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._jobs

    def get(self, job_id: str) -> dict:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFound(job_id)
            return copy.deepcopy(job)

    def directory(self, job_id: str) -> Path:
        if not _JOB_ID.fullmatch(job_id or ""):
            raise JobNotFound(job_id)
        path = (self.data_dir / job_id).resolve()
        if path.parent != self.data_dir.resolve():
            raise JobNotFound(job_id)
        return path

    def update(self, job_id: str, **fields) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.update(fields)
            self._write(self.directory(job_id) / "job.json", job)

    def try_begin(self, job_id: str, allowed: set[str], status: str, message: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job["status"] not in allowed:
                return None
            job["status"] = status
            job["message"] = message
            job["error"] = ""
            self._write(self.directory(job_id) / "job.json", job)
            return copy.deepcopy(job)

    def fail(self, job_id: str, message: str) -> None:
        self.update(
            job_id,
            status="error",
            error=redact(message),
            message=redact(message),
        )

    def delete(self, job_id: str) -> None:
        with self._lock:
            if job_id not in self._jobs:
                raise JobNotFound(job_id)
            self._jobs.pop(job_id, None)
            directory = self.directory(job_id)
        shutil.rmtree(directory, ignore_errors=True)

    def _write(self, path: Path, job: dict) -> None:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)


def to_public(job: dict) -> dict:
    speakers = []
    for speaker in job.get("speakers") or []:
        speakers.append(
            {
                "id": speaker["id"],
                "label": speaker["label"],
                "duration": speaker["duration"],
                "duration_label": speaker["duration_label"],
                "turn_count": speaker["turn_count"],
            }
        )
    return {
        "id": job["id"],
        "status": job["status"],
        "message": job.get("message") or "",
        "title": job.get("title") or "",
        "speakers": speakers,
        "grouping": job.get("grouping") or "voices",
        "selected_speaker": job.get("selected_speaker"),
        "selected_speakers": job.get("selected_speakers") or [],
        "has_output": bool(job.get("has_output")),
        "transcript": job.get("transcript") or "",
        "detected_language": job.get("detected_language") or "",
        "language_label": job.get("language_label") or "",
        "output_revision": job.get("output_revision") or 0,
        "error": job.get("error") or "",
    }


jobs = JobStore(DATA_DIR)
