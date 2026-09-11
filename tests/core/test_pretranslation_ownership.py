from __future__ import annotations

from uuid import uuid4

from puripuly_heart.core.audio.pretranslation_ownership import (
    PretranslationOwnershipOwner,
    assign_ownership_units,
)
from puripuly_heart.core.audio.psem_receiver import ProspectiveSpeakerHypothesis
from puripuly_heart.core.stt.backend import STTTimedToken


def _token(
    text: str,
    *,
    start: int,
    end: int,
    timing: str = "interval",
    language: str = "en",
) -> STTTimedToken:
    return STTTimedToken(
        text=text,
        language=language,
        start_ms=start,
        end_ms=end,
        timing=timing,  # type: ignore[arg-type]
        source_start_sample=None if timing != "interval" else start,
        source_end_sample=None if timing in {"unmapped", "invalid"} else end,
    )


def _event(
    hypothesis_id: str,
    boundary: int,
    *,
    available: float = 1.0,
    slot: int | None = 1,
    valid: bool = True,
    retracted: bool = False,
    epoch: int = 1,
) -> ProspectiveSpeakerHypothesis:
    return ProspectiveSpeakerHypothesis(
        hypothesis_id=hypothesis_id,
        revision=0,
        capture_epoch=epoch,
        support_start_sample=max(boundary - 100, 0),
        support_end_sample=boundary + 100,
        estimated_transition_sample=boundary,
        observed_frontier_sample=boundary + 200,
        available_at_monotonic_s=available,
        producer_generation=1,
        reference_generation=1,
        producer_valid=valid,
        reference_valid=valid,
        retracted=retracted,
        local_slot=slot,
    )


def test_other_other_keeps_local_boundaries_and_conserves_text() -> None:
    tokens = (
        _token("A ", start=0, end=1000),
        _token("B ", start=1000, end=2000),
        _token("C", start=2000, end=3000),
    )
    events = (
        _event("h1", 1000, slot=1),
        _event("h2", 2000, slot=2),
    )
    units, late, _unknown = assign_ownership_units(
        tokens, events, admitted_at_monotonic_s=2.0, capture_epoch=1
    )
    assert late == ()
    assert [unit.group_id for unit in units] == ["CURRENT-0", "OTHER-1", "OTHER-2"]
    assert "".join(unit.text for unit in units) == "A B C"
    assert units[1].text == "B "
    assert units[2].text == "C"


def test_identical_text_groups_are_not_deduped() -> None:
    tokens = (
        _token("same", start=0, end=1000),
        _token("same", start=1000, end=2000),
    )
    events = (_event("h1", 1000, slot=1),)
    units, _, _ = assign_ownership_units(
        tokens, events, admitted_at_monotonic_s=2.0, capture_epoch=1
    )
    assert [unit.text for unit in units] == ["same", "same"]
    assert units[0].group_id != units[1].group_id


def test_straddle_and_unmapped_stay_unknown_unsplit() -> None:
    tokens = (
        _token("straddle", start=900, end=1100),
        _token("?", start=2000, end=2100, timing="unmapped"),
    )
    events = (_event("h1", 1000, slot=1),)
    units, _, reasons = assign_ownership_units(
        tokens, events, admitted_at_monotonic_s=2.0, capture_epoch=1
    )
    assert all(unit.relation == "UNKNOWN" for unit in units)
    assert "straddle" in reasons
    assert "unmapped" in reasons
    assert "".join(unit.text for unit in units) == "straddle?"


def test_late_after_commit_does_not_mutate() -> None:
    parent = uuid4()
    owner = PretranslationOwnershipOwner(enabled=True)
    owner.observe(_event("h1", 1000, slot=1, available=1.0))
    tokens = (
        _token("A ", start=0, end=1000),
        _token("B", start=1000, end=2000),
    )
    first = owner.assign(
        parent_utterance_id=parent,
        timed_tokens=tokens,
        capture_epoch=1,
        admitted_at_monotonic_s=2.0,
    )
    late = owner.observe_after_commit(parent, _event("h2", 1500, slot=2, available=3.0))
    second = owner.assign(
        parent_utterance_id=parent,
        timed_tokens=tokens,
        capture_epoch=1,
        admitted_at_monotonic_s=4.0,
    )
    assert late == "late"
    assert first.units == second.units
    assert second.disposition == "already_committed"


def test_retracted_and_invalid_are_not_used() -> None:
    tokens = (_token("hello", start=0, end=2000),)
    events = (
        _event("bad", 1000, valid=False),
        _event("gone", 1000, retracted=True, slot=2),
    )
    units, _, _ = assign_ownership_units(
        tokens, events, admitted_at_monotonic_s=2.0, capture_epoch=1
    )
    assert len(units) == 1
    assert units[0].relation == "CURRENT"


def test_gap_epoch_mismatch_does_not_force_assign() -> None:
    tokens = (_token("hello", start=0, end=2000),)
    events = (_event("other-epoch", 1000, epoch=9, slot=2),)
    units, _, _ = assign_ownership_units(
        tokens, events, admitted_at_monotonic_s=2.0, capture_epoch=1
    )
    assert units[0].relation == "CURRENT"


def test_disabled_owner_does_not_split() -> None:
    owner = PretranslationOwnershipOwner(enabled=False)
    owner.observe(_event("h1", 1000, slot=1))
    assignment = owner.assign(
        parent_utterance_id=uuid4(),
        timed_tokens=(_token("hello", start=0, end=2000),),
        capture_epoch=1,
        admitted_at_monotonic_s=2.0,
    )
    assert assignment.disposition == "disabled"
    assert assignment.units == ()


def test_end_only_interval_uncertainty_is_unknown() -> None:
    tokens = (_token("x", start=0, end=1000, timing="end_only"),)
    events = (_event("h1", 500, slot=1),)
    units, _, reasons = assign_ownership_units(
        tokens, events, admitted_at_monotonic_s=2.0, capture_epoch=1
    )
    assert units[0].relation == "UNKNOWN"
    assert "interval_uncertain" in reasons
