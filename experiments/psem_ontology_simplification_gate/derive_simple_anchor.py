from __future__ import annotations

from enum import StrEnum


class SimpleAnchorState(StrEnum):
    NO_SPEECH = "NO_SPEECH"
    ANCHOR_SPEECH = "ANCHOR_SPEECH"
    NON_ANCHOR_SPEECH = "NON_ANCHOR_SPEECH"


def derive_gt_simple_anchor_state(
    *, speech_present: bool, anchor_present: bool
) -> SimpleAnchorState:
    if not speech_present:
        return SimpleAnchorState.NO_SPEECH
    if anchor_present:
        return SimpleAnchorState.ANCHOR_SPEECH
    return SimpleAnchorState.NON_ANCHOR_SPEECH


def derive_model_simple_anchor_state(
    *, speech_present: bool, p_anchor: float, anchor_threshold: float
) -> SimpleAnchorState:
    return derive_gt_simple_anchor_state(
        speech_present=speech_present,
        anchor_present=p_anchor >= anchor_threshold,
    )
