"""Appels ffmpeg. Les arguments sont passés sans shell."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def run_ffmpeg(args: list[str]) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg est introuvable. Installez-le, puis relancez le programme.")
    command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or "ffmpeg a échoué.").strip()
        raise RuntimeError(detail or "ffmpeg a échoué.")


def to_wav(src: Path, dest: Path) -> None:
    run_ffmpeg(
        [
            "-i",
            str(src),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "44100",
            "-c:a",
            "pcm_s16le",
            str(dest),
        ]
    )


def probe_duration(path: Path) -> float:
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe est introuvable.")
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or "ffprobe a échoué.").strip())
    return float(result.stdout.strip())
