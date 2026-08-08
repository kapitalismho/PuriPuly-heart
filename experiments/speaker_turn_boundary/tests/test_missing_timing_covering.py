"""Phase 2 exit-gate remediation fixtures for the reference builder.

Covers the frozen unscored covering-interval rule (bundle P2-030, exit-gate
finding P2-REF-004): a word lacking timing between timed neighbors must produce
an explicit covering span, never an empty dropped interval.
"""

from __future__ import annotations

from experiments.speaker_turn_boundary.turn_episode.build_episodes import (
    RawWord,
    missing_timing_intervals,
)


def _word(
    speaker: str, start: float | None, end: float | None, text: str, path_index: int
) -> RawWord:
    return RawWord(
        speaker=speaker,
        start_time_s=start,
        end_time_s=end,
        text=text,
        ambiguous=False,
        path_index=path_index,
    )


def test_missing_word_between_timed_neighbors_covered():
    words = [
        _word("P1", 10.0, 11.0, "one", 0),
        _word("P1", None, None, "??", 0),
        _word("P1", 13.0, 14.0, "two", 0),
    ]
    intervals = missing_timing_intervals(words, session_end=30 * 16000)
    assert intervals == [(11 * 16000, 13 * 16000)]


def test_missing_run_merges_and_uses_session_bounds():
    words = [
        _word("P1", None, None, "first", 0),
        _word("P1", None, None, "second", 0),
        _word("P1", 20.0, 21.0, "timed", 0),
        _word("P1", None, None, "last", 0),
    ]
    session_end = 25 * 16000
    intervals = missing_timing_intervals(words, session_end=session_end)
    assert intervals == [(0, 20 * 16000), (21 * 16000, session_end)]


def test_missing_run_at_file_end_crosses_other_files():
    words = [
        _word("P1", 1.0, 2.0, "a", 0),
        _word("P1", None, None, "tail", 0),
        _word("P2", 3.0, 4.0, "later", 1),
    ]
    session_end = 10 * 16000
    intervals = missing_timing_intervals(words, session_end=session_end)
    assert intervals == [(2 * 16000, session_end)]


def test_no_missing_words_yields_empty():
    words = [_word("P1", 1.0, 2.0, "a", 0), _word("P1", 3.0, 4.0, "b", 0)]
    assert missing_timing_intervals(words, session_end=10 * 16000) == []
