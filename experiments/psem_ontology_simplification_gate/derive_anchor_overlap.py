from __future__ import annotations

from enum import StrEnum


class AnchorOverlapState(StrEnum):
    NO_SPEECH = "NO_SPEECH"
    ANCHOR_ONLY = "ANCHOR_ONLY"
    ANCHOR_WITH_OVERLAP = "ANCHOR_WITH_OVERLAP"
    NON_ANCHOR_SPEECH = "NON_ANCHOR_SPEECH"
    ANCHOR_UNCERTAIN = "ANCHOR_UNCERTAIN"


def anchor_overlap_proxy(p_anchor: float, p_nonanchor_max: float) -> float:
    return min(p_anchor, p_nonanchor_max)


def derive_gt_anchor_overlap_state(
    *, speech_present: bool, anchor_present: bool, anchor_overlap_present: bool
) -> AnchorOverlapState:
    if anchor_overlap_present and not anchor_present:
        raise ValueError("GT anchor overlap cannot be present while the anchor is absent")
    if not speech_present:
        return AnchorOverlapState.NO_SPEECH
    if not anchor_present:
        return AnchorOverlapState.NON_ANCHOR_SPEECH
    if anchor_overlap_present:
        return AnchorOverlapState.ANCHOR_WITH_OVERLAP
    return AnchorOverlapState.ANCHOR_ONLY


def derive_model_anchor_overlap_state(
    *,
    speech_present: bool,
    p_anchor: float,
    p_nonanchor_max: float,
    anchor_threshold: float,
    anchor_overlap_threshold: float,
    strict_inconsistent: bool = False,
) -> AnchorOverlapState:
    if not speech_present:
        return AnchorOverlapState.NO_SPEECH
    anchor_present = p_anchor >= anchor_threshold
    overlap_present = anchor_overlap_proxy(p_anchor, p_nonanchor_max) >= anchor_overlap_threshold
    if not anchor_present:
        if overlap_present and not strict_inconsistent:
            return AnchorOverlapState.ANCHOR_UNCERTAIN
        return AnchorOverlapState.NON_ANCHOR_SPEECH
    if overlap_present:
        return AnchorOverlapState.ANCHOR_WITH_OVERLAP
    return AnchorOverlapState.ANCHOR_ONLY
