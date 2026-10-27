"""Découpe et collage de la voix choisie."""

from __future__ import annotations

import shutil
from pathlib import Path

from app.pipeline.ffmpeg_util import run_ffmpeg
from app.pipeline.timeline import (
    JINGLE_SECONDS,
    MERGE_GAP_SECONDS,
    build_sequence,
    preview_bounds,
)


def render_preview(work_wav: Path, start: float, end: float, dest: Path) -> None:
    window_start, window_end = preview_bounds(start, end)
    duration = max(0.1, window_end - window_start)
    run_ffmpeg(
        [
            "-ss",
            f"{window_start:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(work_wav),
            "-c:a",
            "libmp3lame",
            "-q:a",
            "4",
            str(dest),
        ]
    )


def write_jingle(dest: Path, duration: float = JINGLE_SECONDS) -> None:
    """Carillon d'une seconde, généré localement, sans fichier audio externe."""
    tone = (
        "aevalsrc="
        + "'0.22*sin(2*PI*523.25*t)*exp(-3.2*t)"
        + "+0.18*sin(2*PI*659.25*t)*exp(-3.4*max(t-0.16,0))*(gte(t,0.16))"
        + "+0.14*sin(2*PI*783.99*t)*exp(-3.2*max(t-0.34,0))*(gte(t,0.34))'"
        + f":s=44100:d={duration:.3f}"
    )
    run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            tone,
            "-ac",
            "1",
            "-af",
            "afade=t=in:st=0:d=0.02,afade=t=out:st=0.68:d=0.32",
            "-c:a",
            "pcm_s16le",
            str(dest),
        ]
    )


def render_speaker(
    work_wav: Path,
    turns: list[tuple[float, float]],
    dest_mp3: Path,
    include_preroll: bool = True,
    merge_gap: float = MERGE_GAP_SECONDS,
) -> None:
    sequence = build_sequence(
        turns,
        include_preroll=include_preroll,
        merge_gap=merge_gap,
    )
    if not any(piece[0] == "span" for piece in sequence):
        raise RuntimeError("Cette voix ne contient aucun passage exploitable.")

    parts = dest_mp3.parent / "parts"
    if parts.exists():
        shutil.rmtree(parts)
    parts.mkdir(parents=True)
    temporary = dest_mp3.with_suffix(".part.mp3")
    piece_paths: list[Path] = []
    jingle_path = parts / "jingle.wav"
    try:
        if any(piece[0] == "jingle" for piece in sequence):
            write_jingle(jingle_path)
        for index, piece in enumerate(sequence):
            if piece[0] == "jingle":
                piece_paths.append(jingle_path)
                continue
            _kind, start, end = piece
            cut = parts / f"piece_{index:04d}.wav"
            run_ffmpeg(
                [
                    "-ss",
                    f"{start:.3f}",
                    "-t",
                    f"{max(0.05, end - start):.3f}",
                    "-i",
                    str(work_wav),
                    "-c:a",
                    "pcm_s16le",
                    str(cut),
                ]
            )
            piece_paths.append(cut)
        list_path = parts / "list.txt"
        lines = []
        for piece in piece_paths:
            escaped = str(piece.resolve()).replace("'", r"'\''")
            lines.append(f"file '{escaped}'")
        list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        run_ffmpeg(
            [
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-c:a",
                "libmp3lame",
                "-q:a",
                "2",
                str(temporary),
            ]
        )
        temporary.replace(dest_mp3)
    finally:
        shutil.rmtree(parts, ignore_errors=True)
        temporary.unlink(missing_ok=True)
