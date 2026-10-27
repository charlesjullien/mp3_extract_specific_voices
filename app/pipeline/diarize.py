"""Diarisation locale avec pyannote. Aucun envoi d'audio."""

from __future__ import annotations

import threading
from pathlib import Path

from app.config import CPU_THREADS, DIARIZATION_MODEL, HF_TOKEN, PRECISE, mark_diarization_ready
from app.pipeline.timeline import (
    MIN_TURN_SECONDS,
    PRECISE_MERGE_GAP_SECONDS,
    assign_replies,
    format_duration,
    merge_turns,
    replies_from_masks,
)
from app.validate import sanitize_speaker_id

_pipeline = None
_pipeline_lock = threading.Lock()


def get_pipeline():
    global _pipeline
    with _pipeline_lock:
        if _pipeline is not None:
            return _pipeline
        import os

        os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
        if not HF_TOKEN:
            raise RuntimeError(
                "HF_TOKEN est absent du fichier .env. Créez un jeton en lecture sur "
                "Hugging Face après avoir accepté les conditions du modèle "
                f"{DIARIZATION_MODEL}. Ce jeton télécharge les poids du modèle ; "
                "l'audio reste sur cette machine."
            )
        try:
            from pyannote.audio.telemetry import set_telemetry_metrics

            set_telemetry_metrics(False)
        except Exception:
            pass
        try:
            import torch
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise RuntimeError(
                "pyannote.audio n'est pas installé. Exécutez : pip install -r requirements.txt"
            ) from exc

        try:
            try:
                pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, token=HF_TOKEN)
            except TypeError:
                pipeline = Pipeline.from_pretrained(
                    DIARIZATION_MODEL, use_auth_token=HF_TOKEN
                )
        except Exception as exc:
            raise RuntimeError(
                "Impossible de charger le modèle de diarisation. Vérifiez le jeton et "
                f"l'acceptation des conditions de {DIARIZATION_MODEL}. Détail : {exc}"
            ) from exc
        if pipeline is None:
            raise RuntimeError(
                "Impossible de charger le modèle de diarisation. Acceptez les conditions "
                f"de {DIARIZATION_MODEL} sur Hugging Face, puis vérifiez HF_TOKEN."
            )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            torch.set_num_threads(CPU_THREADS)
        pipeline.to(torch.device(device))
        mark_diarization_ready()
        _pipeline = pipeline
        return _pipeline


def diarize(wav_path: Path, num_speakers: int | None) -> tuple[list[dict], str]:
    pipeline = get_pipeline()
    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    captured: dict = {}

    def _capture(step, artefact, **_kwargs):
        if step == "segmentation" and getattr(artefact, "data", None) is not None:
            captured["segmentation"] = artefact

    _set_clustering(pipeline, precise=PRECISE)
    try:
        output = pipeline(str(wav_path), hook=_capture, **kwargs)
    except Exception as exc:
        raise RuntimeError(f"L'analyse des voix a échoué. Détail : {exc}") from exc

    grouped: dict[str, list[tuple[float, float]]] = {}
    annotation = getattr(output, "speaker_diarization", output)
    for start, end, raw_speaker in _iter_turns(annotation):
        grouped.setdefault(str(raw_speaker), []).append((start, end))
    if not grouped:
        raise RuntimeError("Aucune voix détectée dans cet audio.")

    order = sorted(grouped, key=lambda speaker: min(start for start, _ in grouped[speaker]))
    raw_turns: list[list[tuple[float, float]]] = []
    raw_names: list[str] = []
    for raw_speaker in order:
        audible = [
            (start, end)
            for start, end in grouped[raw_speaker]
            if end - start >= MIN_TURN_SECONDS
        ]
        if not audible:
            continue
        raw_turns.append(audible)
        raw_names.append(str(raw_speaker))

    if PRECISE:
        replies = _replies(captured.get("segmentation"))
        if len(replies) >= 2:
            kept_names: list[str] = []
            kept_turns: list[list[tuple[float, float]]] = []
            for name, turns in zip(raw_names, assign_replies(raw_turns, replies)):
                if turns:
                    kept_names.append(name)
                    kept_turns.append(turns)
            if kept_turns:
                raw_names = kept_names
                raw_turns = kept_turns

    used: set[str] = set()
    speakers: list[dict] = []
    merge_gap = PRECISE_MERGE_GAP_SECONDS if PRECISE else None
    for index, (raw_speaker, audible) in enumerate(zip(raw_names, raw_turns), start=1):
        merged = (
            merge_turns(audible, merge_gap=merge_gap)
            if merge_gap is not None
            else merge_turns(audible)
        )
        if not merged:
            continue
        speaker_id = sanitize_speaker_id(raw_speaker, index, used)
        duration = sum(end - start for start, end in merged)
        speakers.append(
            {
                "id": speaker_id,
                "label": f"Voix {len(speakers) + 1}",
                "duration": round(duration, 3),
                "duration_label": format_duration(duration),
                "turn_count": len(merged),
                "turns": [[round(start, 3), round(end, 3)] for start, end in merged],
            }
        )
    if PRECISE:
        kept = [speaker for speaker in speakers if speaker["duration"] >= 1.0]
        if kept:
            speakers = kept
            for index, speaker in enumerate(speakers, start=1):
                speaker["label"] = f"Voix {index}"
    if not speakers:
        raise RuntimeError("Aucune voix détectée dans cet audio.")
    return speakers, "voices"


def _set_clustering(pipeline, precise: bool) -> None:
    """true resserre le regroupement pour les répliques courtes. Le pipeline est réutilisé."""
    clustering = pipeline.clustering
    clustering.threshold = 0.6
    clustering.Fa = 0.07
    clustering.Fb = 0.05 if precise else 0.8


def _replies(segmentation) -> list[tuple[float, float]]:
    if segmentation is None:
        return []
    window = segmentation.sliding_window
    return replies_from_masks(
        segmentation.data > 0.5,
        duration=float(window.duration),
        step=float(window.step),
    )


def _iter_turns(annotation):
    if hasattr(annotation, "itertracks"):
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            yield float(turn.start), float(turn.end), speaker
        return
    for turn, speaker in annotation:
        yield float(turn.start), float(turn.end), speaker
