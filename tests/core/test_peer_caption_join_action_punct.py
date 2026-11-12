from __future__ import annotations

import pytest

from puripuly_heart.core.orchestrator.peer_caption import join_action

VALID_ACTIONS = {"merge", "continuation", "independent"}
PUNCTUATION_ONLY_PREVIOUS = ("...", "!!!", "???", "?!", ",,", ".!?", "。。。", ",，、")


def _join(previous: str, incoming: str, **overrides: object) -> str:
    arguments: dict[str, object] = {
        "language": "en",
        "gap_s": 0.2,
        "expected_lines": 1,
        "provider_id": "soniox",
        "sealed": None,
    }
    arguments.update(overrides)
    return join_action(previous, incoming, **arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize("previous", PUNCTUATION_ONLY_PREVIOUS)
@pytest.mark.parametrize("incoming", ["Next sentence.", "yeah", "..."])
def test_punctuation_only_previous_never_crashes_join_decision(
    previous: str, incoming: str
) -> None:
    assert _join(previous, incoming) in VALID_ACTIONS
    assert _join(previous, incoming, gap_s=None) in VALID_ACTIONS
    assert _join(previous, incoming, gap_s=2.0) in VALID_ACTIONS
    assert _join(previous, incoming, language="ko") in VALID_ACTIONS
    assert _join(previous, incoming, expected_lines=5) in VALID_ACTIONS


def test_punctuation_only_previous_keeps_merge_favoring_endings() -> None:
    assert _join("...", "Next sentence.") == "merge"
    assert _join("...", "Next sentence.", expected_lines=5) == "continuation"
    assert _join("...", "yeah", language="en") == "independent"
    assert _join("...", "Next sentence.", gap_s=2.0) == "independent"


def test_word_endings_keep_existing_join_contract() -> None:
    assert _join("the first,", "continuation", gap_s=None) == "independent"
    assert _join("the first,", "yeah okay", gap_s=0.2) == "independent"
    assert _join("the first,", "continuation", gap_s=1.51) == "independent"
    assert _join("Wait", "there", gap_s=0.2) == "merge"
