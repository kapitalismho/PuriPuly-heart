from __future__ import annotations

from dataclasses import dataclass

from experiments.speaker_turn_boundary.turn_episode.phase5_inputs import annotation_views


@dataclass
class Region:
    start_sample: int
    end_sample: int
    speakers: list[str]
    ambiguous: bool = False


@dataclass
class Spec:
    action_kind: str
    acceptable_interval: tuple[int, int]


@dataclass
class Word:
    start_time_s: float | None
    end_time_s: float | None
    ambiguous: bool = False


def test_annotation_views_clip_and_separate_taxonomy() -> None:
    result = annotation_views(
        [
            Region(0, 4000, ["A"]),
            Region(4000, 8000, []),
            Region(8000, 12000, ["A", "B"]),
            Region(12000, 16000, ["B"], True),
        ],
        [Spec("neutral_pause", (4000, 8000)), Spec("unscored", (14000, 17000))],
        [Word(0.10, 0.20), Word(None, None)],
        scored_start=1000,
        scored_end=15000,
    )
    assert result["singleton_intervals"] == [(1000, 4000, "A")]
    assert result["overlap_intervals"] == [(8000, 12000)]
    assert result["pause_intervals"] == [(4000, 8000)]
    assert result["unscored_intervals"] == [(12000, 15000)]
    assert result["word_intervals"] == [(1600, 3200)]
    assert result["word_timing_observable"] is True


def test_missing_word_timing_is_not_observable() -> None:
    result = annotation_views(
        [Region(0, 4000, ["A"])],
        [],
        None,
        scored_start=0,
        scored_end=4000,
    )
    assert result["word_intervals"] is None
    assert result["word_timing_observable"] is False
