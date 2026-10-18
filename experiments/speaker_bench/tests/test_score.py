import pytest
from bench import reference
from bench.align import Item, align, edit_counts
from bench.score import score_case
from bench.text import tokens
from bench.units import Assignment, CaseRun, Unit, case_run

GAP = 0.5


def ref(turns, *, language="en", samples=16000 * 20):
    raw = {
        "case_id": "case",
        "language": language,
        "category": "test",
        "synthetic": True,
        "audio": {"samples": samples, "sample_rate_hz": 16000},
        "turns": turns,
    }
    return reference.build(raw, gap_s=GAP)


def turn(speaker, start, end, text):
    return {"speaker": speaker, "start": start, "end": end, "text": text}


def colored(key, index):
    return Assignment("identified", key, index, False)


GRAY = Assignment("unavailable", None, None, False)


def unit(text, lo, hi, *, readable, final, settled=None, at_readable=None, order=None):
    u = Unit(
        source_id=text,
        order=order or (0, int(lo * 1000), 0),
        text=text,
        language="en",
        admitted_at=readable - 0.05,
        spans=[(lo, hi)],
        first_readable=readable,
        readable_assignment=at_readable if at_readable is not None else GRAY,
        settled_at=settled if settled is not None else readable + 0.2,
        final=final,
        settle_events=1,
    )
    return u


def run(units):
    return CaseRun("case", 100.0, 16000, units, {}, [])


A_B = [turn("A", 0.0, 2.0, "one two three"), turn("B", 2.1, 4.0, "four five six")]


def test_continuous_transfer_released_early_with_correct_colors():
    result = score_case(
        ref(A_B),
        run(
            [
                unit("one two three", 0.0, 2.05, readable=2.6, final=colored("k1", 0)),
                unit("four five six", 2.05, 4.0, readable=4.6, final=colored("k2", 1)),
            ]
        ),
    )
    (transition,) = result["transitions"]
    assert transition["kind"] == "continuous"
    assert transition["separated"] is True
    assert transition["early_release"] is True
    assert transition["boundary_error_s"] == 0.0
    assert transition["predecessor_latency_s"] == pytest.approx(0.6)
    assert transition["edge"] == {"deleted": 0, "misplaced": 0, "duplicated": 0}
    assert (transition["a_label"], transition["b_label"]) == ("correct", "correct")
    assert result["attribution"] == {"colored_correct": 6}
    assert result["mapping"] == {"k1": "A", "k2": "B"}
    assert result["presentation"] == {"late_within_500ms": 2}
    assert result["mixed_units"] == 0


def test_merged_baseline_is_not_an_early_release_and_is_mixed():
    result = score_case(
        ref(A_B),
        run([unit("one two three four five six", 0.0, 4.0, readable=4.6, final=GRAY)]),
    )
    (transition,) = result["transitions"]
    assert transition["outcome"] == "merged"
    assert transition["early_release"] is False
    assert result["attribution"] == {"gray_unavailable": 6}
    assert result["mixed_units"] == 1
    assert result["presentation"] == {"stayed_gray": 1}


def test_late_cut_reports_positive_boundary_error_and_misplaced_successor_word():
    result = score_case(
        ref(A_B),
        run(
            [
                unit("one two three four", 0.0, 2.5, readable=3.0, final=colored("k1", 0)),
                unit("five six", 2.5, 4.0, readable=4.6, final=colored("k2", 1)),
            ]
        ),
    )
    (transition,) = result["transitions"]
    assert transition["outcome"] == "misplaced_boundary"
    assert transition["early_release"] is False
    assert transition["boundary_error_s"] == pytest.approx(0.4)
    assert transition["edge"]["misplaced"] == 1
    assert result["attribution"]["colored_wrong"] == 1


def test_early_cut_reports_negative_boundary_error():
    result = score_case(
        ref(A_B),
        run(
            [
                unit("one two", 0.0, 1.5, readable=2.0, final=colored("k1", 0)),
                unit("three four five six", 1.5, 4.0, readable=4.6, final=colored("k2", 1)),
            ]
        ),
    )
    (transition,) = result["transitions"]
    assert transition["outcome"] == "misplaced_boundary"
    assert transition["boundary_error_s"] == pytest.approx(-0.5)
    assert transition["edge"] == {"deleted": 0, "misplaced": 1, "duplicated": 0}


def test_repeated_predecessor_word_in_successor_counts_as_duplicate():
    result = score_case(
        ref(A_B),
        run(
            [
                unit("one two three", 0.0, 2.05, readable=2.6, final=colored("k1", 0)),
                unit("three four five six", 2.05, 4.0, readable=4.6, final=colored("k2", 1)),
            ]
        ),
    )
    (transition,) = result["transitions"]
    assert transition["outcome"] == "separated"
    assert transition["edge"] == {"deleted": 0, "misplaced": 0, "duplicated": 1}
    assert result["content"]["insertions"] == 1


def test_transition_classes_follow_reference_geometry():
    paused = ref([turn("A", 0.0, 2.0, "a b"), turn("B", 2.8, 4.0, "c d")])
    overlap = ref([turn("A", 0.0, 2.0, "a b"), turn("B", 1.5, 4.0, "c d")])
    contained = ref([turn("A", 0.0, 4.0, "a b c d"), turn("B", 1.5, 2.0, "yeah")])
    assert [t.kind for t in paused.transitions] == ["paused"]
    assert [t.kind for t in overlap.transitions] == ["overlap"]
    assert contained.transitions[0].kind == "contained"
    assert contained.transitions[0].short_successor is True


def test_every_backchannel_inside_a_turn_is_a_contained_transition():
    r = ref(
        [
            turn("A", 0.0, 9.0, "one two three four five six seven eight nine"),
            turn("B", 3.0, 3.4, "yeah"),
            turn("B", 6.5, 7.0, "right"),
        ]
    )
    assert [(t.kind, r.segments[t.before].speaker) for t in r.transitions] == [
        ("contained", "A"),
        ("contained", "A"),
    ]


def test_same_speaker_resuming_after_pause_is_not_a_transition():
    r = ref(
        [
            turn("A", 0.0, 2.0, "one two"),
            turn("B", 2.1, 3.0, "three four"),
            turn("B", 4.0, 5.0, "five six"),
        ]
    )
    assert [t.kind for t in r.transitions] == ["continuous"]


def test_short_aba_marks_returning_short_successor():
    r = ref(
        [
            turn("A", 0.0, 2.0, "one two"),
            turn("B", 2.1, 2.6, "no"),
            turn("A", 2.7, 4.0, "three four"),
        ]
    )
    first = r.transitions[0]
    assert (first.kind, first.short_successor, first.returning) == ("continuous", True, True)


def test_asr_deletion_is_content_loss_not_attribution_error():
    result = score_case(
        ref(A_B),
        run(
            [
                unit("one two", 0.0, 2.05, readable=2.6, final=colored("k1", 0)),
                unit("four five six", 2.05, 4.0, readable=4.6, final=colored("k2", 1)),
            ]
        ),
    )
    assert result["attribution"]["not_recognized"] == 1
    assert result["attribution"]["colored_correct"] == 5
    assert result["transitions"][0]["edge"]["deleted"] == 1
    assert result["content"]["deletions"] == 1


def test_returning_speaker_with_new_key_counts_as_fragmentation():
    r = ref(
        [
            turn("A", 0.0, 2.0, "one two"),
            turn("B", 2.6, 4.0, "three four"),
            turn("A", 14.5, 16.0, "five six"),
        ]
    )
    result = score_case(
        r,
        run(
            [
                unit("one two", 0.0, 2.0, readable=2.5, final=colored("k1", 0)),
                unit("three four", 2.6, 4.0, readable=4.5, final=colored("k2", 1)),
                unit("five six", 14.5, 16.0, readable=16.5, final=colored("k3", 2)),
            ]
        ),
    )
    assert result["returns"] == [{"segment": 2, "gap_s": 12.5, "outcome": "different"}]
    assert result["key_fragmentation"]["A"] == 2
    assert result["attribution"]["colored_wrong"] == 2


def test_same_speaker_split_is_detected():
    r = ref([turn("A", 0.0, 4.0, "one two three four five six seven eight")])
    result = score_case(
        r,
        run(
            [
                unit("one two three four", 0.0, 2.0, readable=2.5, final=colored("k1", 0)),
                unit("five six seven eight", 2.0, 4.0, readable=4.5, final=colored("k1", 0)),
            ]
        ),
    )
    assert [b["kind"] for b in result["boundaries"]] == ["same_speaker_split"]


def test_presentation_violations_are_counted():
    result = score_case(
        ref(A_B),
        run(
            [
                unit(
                    "one two three",
                    0.0,
                    2.05,
                    readable=2.6,
                    at_readable=colored("k1", 0),
                    final=colored("k9", 0),
                ),
                unit("four five six", 2.05, 4.0, readable=4.6, settled=6.0, final=colored("k2", 1)),
            ]
        ),
    )
    assert result["presentation_violations"] == {"identity_changed": 1, "late_after_deadline": 1}


def test_short_interjection_kept_separate_and_identified():
    r = ref(
        [
            turn("A", 0.0, 2.0, "one two"),
            turn("B", 2.1, 2.6, "no"),
            turn("A", 2.7, 4.0, "three four"),
        ]
    )
    result = score_case(
        r,
        run(
            [
                unit("one two", 0.0, 2.05, readable=2.5, final=colored("k1", 0)),
                unit("no", 2.05, 2.65, readable=3.0, final=colored("k2", 1)),
                unit("three four", 2.65, 4.0, readable=4.5, final=colored("k1", 0)),
            ]
        ),
    )
    (short,) = result["short_segments"]
    assert (short["retained"], short["separate"], short["identity"]) == (1.0, True, "correct")


def test_time_window_prevents_matching_distant_repeated_words():
    mapping = align(
        [Item("yes", 0.0, 1.0), Item("yes", 10.0, 11.0)],
        [Item("yes", 10.0, 11.0)],
        tolerance=0.5,
    )
    assert mapping == [None, 0]


def test_character_tokenization_for_korean_and_edit_counts():
    assert tokens("안녕, 하세요!", "ko") == ["안", "녕", "하", "세", "요"]
    assert edit_counts(["a", "b", "c"], ["a", "x", "c", "d"]) == {
        "reference": 3,
        "errors": 2,
        "substitutions": 1,
        "deletions": 0,
        "insertions": 1,
    }


def test_case_run_reads_observer_events():
    events = [
        {"event": "source_open", "case_id": "c", "t": 1.0},
        {
            "event": "admitted",
            "case_id": "c",
            "t": 12.0,
            "source_id": "s1",
            "order": [1, 1, 0],
            "raw_text": "hello",
            "language": "en",
            "origins": [
                {
                    "source_ranges": [
                        {"normalized_start_sample": 16000, "normalized_end_sample": 32000},
                        {"normalized_start_sample": 32000, "normalized_end_sample": 48000},
                    ]
                }
            ],
            "assignment": {"state": "unavailable", "key": None},
        },
        {
            "event": "first_readable",
            "case_id": "c",
            "t": 12.1,
            "source_id": "s1",
            "readable_at": 12.1,
            "assignment": {"state": "unavailable", "key": None},
        },
        {
            "event": "settled",
            "case_id": "c",
            "t": 12.4,
            "source_id": "s1",
            "settled_at": 12.4,
            "assignment": {"state": "identified", "key": {"slot": 0}, "palette_index": 0},
        },
        {
            "event": "source_close",
            "case_id": "c",
            "t": 20.0,
            "record": {
                "case_completed": True,
                "frame_lag_s": {"max": 0.05},
                "lead_in_samples": 16000,
                "case_start_monotonic": 10.0,
            },
        },
    ]
    parsed = case_run(events, "c", max_frame_lag_s=1.0)
    (u,) = parsed.units
    assert u.spans == [(0.0, 2.0)]
    assert u.first_readable == pytest.approx(2.1)
    assert u.settled_at == pytest.approx(2.4)
    assert u.final.colored and u.final.key == "slot=0"
    assert parsed.invalid == []
    lagged = case_run(
        [
            *events[:-1],
            {**events[-1], "record": {**events[-1]["record"], "frame_lag_s": {"max": 2.0}}},
        ],
        "c",
        max_frame_lag_s=1.0,
    )
    assert lagged.invalid == ["frame_lag_exceeded"]
