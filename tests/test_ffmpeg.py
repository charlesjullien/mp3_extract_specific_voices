import shutil
import subprocess
from pathlib import Path

import pytest

from app.pipeline.extract import render_speaker
from app.pipeline.ffmpeg_util import probe_duration


def _sine(path: Path, duration: float) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg est introuvable")
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={duration}",
            "-ac",
            "1",
            "-ar",
            "44100",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(result.stderr.strip() or "lavfi indisponible")


def test_render_keeps_five_seconds_of_context(tmp_path: Path):
    wav = tmp_path / "work.wav"
    _sine(wav, 10)
    dest = tmp_path / "voix.mp3"
    render_speaker(wav, [(0, 1), (9, 10)], dest)
    assert dest.is_file()
    assert not (tmp_path / "parts").exists()
    # 1 s de parole + 1 s de jingle + 5 s de contexte + 1 s de parole.
    assert probe_duration(dest) == pytest.approx(8, abs=0.35)
