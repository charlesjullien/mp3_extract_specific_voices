"""Transcription locale du fichier final avec faster-whisper."""

from __future__ import annotations

import threading
from pathlib import Path

from app.config import CPU_THREADS, HF_HOME, WHISPER_MODEL, mark_whisper_ready

_model = None
_model_lock = threading.Lock()

_LANGUAGE_LABELS = {
    "fr": "français",
    "en": "anglais",
    "es": "espagnol",
    "de": "allemand",
    "it": "italien",
    "pt": "portugais",
    "ar": "arabe",
    "zh": "chinois",
    "ja": "japonais",
    "ru": "russe",
    "nl": "néerlandais",
    "pl": "polonais",
    "tr": "turc",
    "uk": "ukrainien",
}


def transcribe(path: Path) -> tuple[str, str, str]:
    model = _get_model()
    segments, info = model.transcribe(
        _load_audio(path),
        vad_filter=True,
        condition_on_previous_text=False,
    )
    lines: list[str] = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        lines.append(f"[{_clock(segment.start)} – {_clock(segment.end)}] {text}")
    if not lines:
        raise RuntimeError("Aucune parole reconnue dans le fichier final.")
    language = getattr(info, "language", "") or ""
    label = _LANGUAGE_LABELS.get(language, language)
    return "\n".join(lines) + "\n", language, label


def _load_audio(path: Path, sampling_rate: int = 16000):
    """Lit l'audio avec ffmpeg. faster-whisper passe sinon par PyAV, incompatible ici."""
    import subprocess

    import numpy as np

    result = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            str(sampling_rate),
            "-",
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout:
        detail = (result.stderr or b"").decode("utf-8", errors="replace").strip()
        raise RuntimeError(detail or "Impossible de lire le fichier audio final.")
    return np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def _get_model():
    global _model
    with _model_lock:
        if _model is not None:
            return _model
        try:
            import torch
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper n'est pas installé. Exécutez : pip install -r requirements.txt"
            ) from exc
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        if device == "cpu":
            torch.set_num_threads(CPU_THREADS)
        offline = __import__("os").environ.get("HF_HUB_OFFLINE", "0") == "1"
        try:
            _model = WhisperModel(
                WHISPER_MODEL,
                device=device,
                compute_type=compute_type,
                cpu_threads=CPU_THREADS if device == "cpu" else 0,
                download_root=str(HF_HOME / "whisper"),
                local_files_only=offline,
            )
        except Exception as exc:
            raise RuntimeError(
                "Impossible de charger le modèle Whisper. Au premier lancement, "
                "laissez HF_HUB_OFFLINE désactivé pour télécharger les poids. "
                f"Détail : {exc}"
            ) from exc
        mark_whisper_ready()
        return _model


def _clock(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
