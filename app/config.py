"""Réglages locaux. La télémétrie est coupée avant tout import des modèles."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HUGGINGFACE_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
HF_HOME = MODEL_DIR / "huggingface"
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_HOME / "hub"))

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
HF_HOME.mkdir(parents=True, exist_ok=True)
_mpl_dir = MODEL_DIR / "matplotlib"
_mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_dir))


def env_bool(raw: str | None, default: bool = False) -> bool:
    """Lit true ou false. Vide : la valeur par défaut."""
    if raw is None or str(raw).strip() == "":
        return default
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "on", "oui"}:
        return True
    if value in {"0", "false", "no", "off", "non"}:
        return False
    return default


def threads_for_percent(cores: int, raw: str | None) -> int:
    """Traduit un pourcentage de cœurs en nombre de fils. Vide = tous les cœurs."""
    cores = max(1, int(cores))
    if raw is None or str(raw).strip() == "":
        return cores
    try:
        percent = int(str(raw).strip())
    except ValueError:
        return cores
    percent = min(max(percent, 1), 100)
    return max(1, round(cores * percent / 100))


HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small").strip() or "small"
DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
CPU_THREADS = threads_for_percent(os.cpu_count() or 1, os.environ.get("CPU_PERCENT"))
# true : vidéos courtes, répliques brèves recollées par personne.
# false : interviews longues, moins de découpes.
PRECISE = env_bool(os.environ.get("PRECISE"), False)
for _thread_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_thread_var, str(CPU_THREADS))
HOST = "127.0.0.1"

try:
    PORT = int(os.environ.get("PORT", "8000"))
except ValueError:
    PORT = 8000

try:
    _upload_mb = int(os.environ.get("MAX_UPLOAD_MB", "500"))
except ValueError:
    _upload_mb = 500
MAX_UPLOAD_BYTES = min(max(_upload_mb, 1), 4096) * 1024 * 1024

_DIARIZATION_MARKER = MODEL_DIR / "diarization.ok"
_WHISPER_MARKER = MODEL_DIR / "whisper.ok"


def _whisper_marker_matches() -> bool:
    if not _WHISPER_MARKER.is_file():
        return False
    try:
        saved = _WHISPER_MARKER.read_text(encoding="utf-8").strip()
    except OSError:
        return False
    return saved == WHISPER_MODEL


_offline_choice = os.environ.get("HF_HUB_OFFLINE")
if _offline_choice == "1" or (
    _offline_choice is None and _DIARIZATION_MARKER.exists() and _whisper_marker_matches()
):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def mark_diarization_ready() -> None:
    _DIARIZATION_MARKER.write_text(DIARIZATION_MODEL + "\n", encoding="utf-8")


def mark_whisper_ready() -> None:
    _WHISPER_MARKER.write_text(WHISPER_MODEL + "\n", encoding="utf-8")
