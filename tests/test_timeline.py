from app.pipeline.timeline import (
    assign_replies,
    build_sequence,
    build_timeline,
    clustering_collapsed,
    coalesce_segments,
    preview_bounds,
    replies_from_masks,
)


def test_brief_pause_stays_together():
    assert build_timeline([(0, 1), (1.4, 2)]) == [(0, 2)]


def test_short_replies_stay_separate_excerpts():
    assert build_timeline([(10, 15), (18, 22)]) == [(10, 15), (15, 18), (18, 22)]
    assert build_timeline([(0, 5), (7, 10)]) == [(0, 5), (5, 7), (7, 10)]
    assert build_sequence([(10, 15), (18, 22)]) == [
        ("span", 10, 15),
        ("jingle",),
        ("span", 15, 22),
    ]


def test_gap_of_five_seconds_is_cut_and_can_keep_the_lead_in():
    assert build_timeline([(0, 5), (10, 12)]) == [(0, 5), (5, 10), (10, 12)]
    assert build_timeline([(0, 5), (10, 12)], include_preroll=False) == [(0, 5), (10, 12)]


def test_long_gap_keeps_five_seconds_when_requested():
    assert build_timeline([(0, 5), (20, 25)]) == [(0, 5), (15, 20), (20, 25)]
    assert build_sequence([(0, 5), (20, 25)], include_preroll=False) == [
        ("span", 0, 5),
        ("jingle",),
        ("span", 20, 25),
    ]


def test_three_turns():
    assert build_timeline([(0, 2), (20, 22), (40, 41)]) == [
        (0, 2),
        (15, 20),
        (20, 22),
        (35, 40),
        (40, 41),
    ]


def test_first_turn_has_no_leading_context():
    assert build_timeline([(30, 35)]) == [(30, 35)]


def test_empty():
    assert build_timeline([]) == []


def test_unsorted_short_gap_keeps_context_between_excerpts():
    assert build_timeline([(5, 6), (0, 1), (0.2, 2)]) == [(0, 2), (2, 5), (5, 6)]


def test_preview_centers_eight_seconds():
    assert preview_bounds(0, 20) == (6, 14)


def test_preview_keeps_short_turn():
    assert preview_bounds(1, 3) == (1, 3)


def test_coalesce_joins_touching_parts():
    assert coalesce_segments([(10, 15), (15, 18), (18, 22)]) == [(10, 22)]


def test_jingle_marks_the_end_of_each_speech_except_the_last():
    assert build_sequence([(0, 2), (20, 22), (40, 41)]) == [
        ("span", 0, 2),
        ("jingle",),
        ("span", 15, 22),
        ("jingle",),
        ("span", 35, 41),
    ]


def test_single_turn_has_no_jingle():
    assert build_sequence([(30, 35)]) == [("span", 30, 35)]


def test_checked_replies_stay_separate():
    assert build_sequence([(0, 2), (4, 6)], include_preroll=False, merge_gap=0) == [
        ("span", 0, 2),
        ("jingle",),
        ("span", 4, 6),
    ]


def test_one_voice_covering_the_file_counts_as_collapsed():
    assert clustering_collapsed([56.0, 2.5])
    assert clustering_collapsed([30.0, 25.0]) is False


def test_replies_come_from_non_overlapping_windows():
    import numpy as np

    masks = np.zeros((11, 2, 1), dtype=bool)
    masks[0, :, 0] = True
    masks[1, :, 0] = True
    masks[10, :, 0] = True
    assert replies_from_masks(masks, duration=10.0, step=1.0, min_turn=0.5) == [
        (0.0, 10.0),
        (10.0, 20.0),
    ]


def test_a_reply_mostly_inside_another_is_dropped():
    import numpy as np

    masks = np.zeros((1, 10, 2), dtype=bool)
    masks[0, 0:10, 0] = True
    masks[0, 4:6, 1] = True
    assert replies_from_masks(masks, duration=1.0, step=1.0, min_turn=0.1) == [(0.0, 1.0)]


def test_replies_of_one_person_are_grouped_together():
    speakers = [[(0.0, 10.0)], [(12.0, 20.0)]]
    replies = [(0.5, 2.3), (2.4, 6.0), (13.0, 15.0), (16.0, 18.0)]
    assert assign_replies(speakers, replies) == [
        [(0.5, 2.3), (2.4, 6.0)],
        [(13.0, 15.0), (16.0, 18.0)],
    ]


def test_local_speakers_in_one_window_stay_separate_replies():
    import numpy as np

    masks = np.zeros((1, 4, 2), dtype=bool)
    masks[0, 0:2, 0] = True
    masks[0, 2:4, 1] = True
    assert replies_from_masks(masks, duration=1.0, step=1.0, min_turn=0.2) == [
        (0.0, 0.5),
        (0.5, 1.0),
    ]
