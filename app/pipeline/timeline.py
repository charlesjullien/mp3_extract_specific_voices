"""Tours de parole et secondes de contexte qui les précèdent."""

from __future__ import annotations

MERGE_GAP_SECONDS = 1.0
PRECISE_MERGE_GAP_SECONDS = 0.4
MIN_TURN_SECONDS = 0.3
MIN_REPLY_SECONDS = 0.5
COLLAPSE_SHARE = 0.85
MIN_REPLY_COUNT = 4
PREROLL_SECONDS = 5.0
PREVIEW_SECONDS = 8.0
JINGLE_SECONDS = 1.0


def merge_turns(
    turns: list[tuple[float, float]],
    merge_gap: float = MERGE_GAP_SECONDS,
) -> list[tuple[float, float]]:
    cleaned: list[tuple[float, float]] = []
    for start, end in turns:
        start = max(0.0, float(start))
        end = float(end)
        if end <= start:
            continue
        cleaned.append((start, end))
    cleaned.sort(key=lambda item: (item[0], item[1]))
    if not cleaned:
        return []

    merged: list[list[float]] = [[cleaned[0][0], cleaned[0][1]]]
    for start, end in cleaned[1:]:
        # Une pause de moins d'une seconde reste dans le même extrait.
        # Au-delà, chaque prise courte (1 s, 3 s, 4 s…) reste un extrait distinct.
        if round(start - merged[-1][1], 3) < merge_gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def speech_blocks(
    turns: list[tuple[float, float]],
    preroll: float = PREROLL_SECONDS,
    merge_gap: float = MERGE_GAP_SECONDS,
    include_preroll: bool = True,
) -> list[list[tuple[float, float]]]:
    """Un bloc par prise de parole.

    Une interruption plus courte que ``merge_gap`` reste dans le bloc.
    Si ``include_preroll`` est vrai, chaque bloc après le premier commence
    par le son original qui précède cette parole, sur ``preroll`` secondes au plus.
    """
    lead = preroll if include_preroll else 0.0
    blocks: list[list[tuple[float, float]]] = []
    prev_end: float | None = None
    for start, end in merge_turns(turns, merge_gap=merge_gap):
        if prev_end is None:
            blocks.append([(start, end)])
            prev_end = end
            continue
        speech_start = max(start, prev_end)
        if speech_start >= end:
            continue
        block: list[tuple[float, float]] = []
        context_start = max(prev_end, speech_start - lead, 0.0)
        if context_start < speech_start:
            block.append((context_start, speech_start))
        block.append((speech_start, end))
        blocks.append(block)
        prev_end = end
    return blocks


def build_timeline(
    turns: list[tuple[float, float]],
    preroll: float = PREROLL_SECONDS,
    merge_gap: float = MERGE_GAP_SECONDS,
    include_preroll: bool = True,
) -> list[tuple[float, float]]:
    """Extraits source à coller, sans le jingle de séparation."""
    segments: list[tuple[float, float]] = []
    for block in speech_blocks(
        turns,
        preroll=preroll,
        merge_gap=merge_gap,
        include_preroll=include_preroll,
    ):
        segments.extend(block)
    return segments


def build_sequence(
    turns: list[tuple[float, float]],
    preroll: float = PREROLL_SECONDS,
    merge_gap: float = MERGE_GAP_SECONDS,
    include_preroll: bool = True,
) -> list[tuple]:
    """Morceaux du fichier final : ``("span", début, fin)`` ou ``("jingle",)``.

    Le jingle d'une seconde est inséré à la fin de chaque prise de parole
    qui est suivie d'une autre. Il n'est pas ajouté après la dernière.
    """
    sequence: list[tuple] = []
    blocks = speech_blocks(
        turns,
        preroll=preroll,
        merge_gap=merge_gap,
        include_preroll=include_preroll,
    )
    for index, block in enumerate(blocks):
        if index:
            sequence.append(("jingle",))
        for start, end in coalesce_segments(block):
            sequence.append(("span", start, end))
    return sequence


def coalesce_segments(segments: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not segments:
        return []
    coalesced: list[list[float]] = [[segments[0][0], segments[0][1]]]
    for start, end in segments[1:]:
        if start <= coalesced[-1][1] + 1e-3:
            coalesced[-1][1] = max(coalesced[-1][1], end)
        else:
            coalesced.append([start, end])
    return [(start, end) for start, end in coalesced]


def preview_bounds(
    start: float,
    end: float,
    length: float = PREVIEW_SECONDS,
) -> tuple[float, float]:
    duration = end - start
    if duration <= length:
        return start, end
    middle = (start + end) / 2
    half = length / 2
    return middle - half, middle + half


def assign_replies(
    speakers: list[list[tuple[float, float]]],
    replies: list[tuple[float, float]],
    min_cover: float = 0.5,
) -> list[list[tuple[float, float]]]:
    """Donne chaque réplique à la personne qui la couvre le plus.

    Le programme recolle ensuite les répliques d'une même personne.
    """
    assigned: list[list[tuple[float, float]]] = [[] for _ in speakers]
    for start, end in replies:
        length = end - start
        if length <= 0 or not speakers:
            continue
        best_index = None
        best_overlap = 0.0
        for index, turns in enumerate(speakers):
            overlap = 0.0
            for turn_start, turn_end in turns:
                overlap += max(0.0, min(end, turn_end) - max(start, turn_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_index = index
        if best_index is not None and best_overlap >= min_cover * length:
            assigned[best_index].append((start, end))
    return assigned


def clustering_collapsed(durations: list[float], share: float = COLLAPSE_SHARE) -> bool:
    """Vrai quand une voix concentre presque toute la parole détectée."""
    total = sum(durations)
    if total <= 0 or not durations:
        return False
    return max(durations) / total >= share


def replies_from_masks(
    masks,
    *,
    duration: float,
    step: float,
    min_turn: float = MIN_REPLY_SECONDS,
) -> list[tuple[float, float]]:
    """Répliques vues dans chaque fenêtre, sans recoller les personnes entre elles.

    Les fenêtres se recouvrent. Seules celles qui se suivent sans recouvrement
    sont gardées, pour ne pas compter deux fois le même passage.
    """
    import numpy as np

    data = np.asarray(masks, dtype=bool)
    n_chunks, n_frames, n_locals = data.shape
    if n_frames == 0 or duration <= 0 or step <= 0:
        return []
    stride = max(1, int(round(duration / step)))
    replies: list[tuple[float, float]] = []
    for chunk in range(0, n_chunks, stride):
        window_start = chunk * step
        for local in range(n_locals):
            active = np.flatnonzero(data[chunk, :, local])
            if active.size == 0:
                continue
            run_start = int(active[0])
            previous = int(active[0])
            for frame in list(active[1:]) + [None]:
                if frame is not None and int(frame) == previous + 1:
                    previous = int(frame)
                    continue
                start = window_start + run_start / n_frames * duration
                end = window_start + (previous + 1) / n_frames * duration
                if end - start >= min_turn:
                    replies.append((round(start, 3), round(end, 3)))
                if frame is None:
                    break
                run_start = previous = int(frame)
    replies.sort()
    return _without_covered_replies(replies)


def _without_covered_replies(
    replies: list[tuple[float, float]],
    cover: float = 0.5,
) -> list[tuple[float, float]]:
    """Retire une réplique déjà en grande partie contenue dans une plus longue."""
    kept: list[tuple[float, float]] = []
    for index, (start, end) in enumerate(replies):
        length = end - start
        covered = False
        for other_index, (other_start, other_end) in enumerate(replies):
            if other_index == index:
                continue
            overlap = min(end, other_end) - max(start, other_start)
            other_length = other_end - other_start
            longer = other_length > length + 1e-3 or (
                abs(other_length - length) <= 1e-3 and other_index < index
            )
            if length > 0 and overlap / length >= cover and longer:
                covered = True
                break
        if not covered:
            kept.append((start, end))
    return kept


def format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    minutes, secs = divmod(total, 60)
    if minutes:
        return f"{minutes} min {secs:02d} s"
    return f"{secs} s"
