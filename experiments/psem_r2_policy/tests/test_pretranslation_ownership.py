from __future__ import annotations

from uuid import uuid4

import pytest

from puripuly_heart.core.audio.pretranslation_ownership import (
    PretranslationEvidence,
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


def _evidence(
    *,
    start: int,
    end: int,
    relation: str,
    available: float = 1.0,
    valid: bool = True,
    epoch: int = 1,
    producer_generation: object = 1,
    reference_generation: object = 1,
) -> PretranslationEvidence:
    return PretranslationEvidence(
        capture_epoch=epoch,
        start_sample=start,
        end_sample=end,
        available_at_monotonic_s=available,
        relation=relation,  # type: ignore[arg-type]
        producer_generation=producer_generation,
        reference_generation=reference_generation,
        reference_valid=valid,
    )


def _observe_evidence(
    owner: PretranslationOwnershipOwner,
    evidence: PretranslationEvidence,
) -> None:
    owner.observe_evidence(
        capture_epoch=evidence.capture_epoch,
        start_sample=evidence.start_sample,
        end_sample=evidence.end_sample,
        available_at_monotonic_s=evidence.available_at_monotonic_s,
        relation=evidence.relation,
        producer_generation=evidence.producer_generation,
        reference_generation=evidence.reference_generation,
        reference_valid=evidence.reference_valid,
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
    evidence = (
        _evidence(start=0, end=1000, relation="CURRENT"),
        _evidence(start=1000, end=2000, relation="OTHER"),
        _evidence(start=2000, end=3000, relation="OTHER"),
    )
    units, late, _unknown = assign_ownership_units(
        tokens,
        events,
        admitted_at_monotonic_s=2.0,
        capture_epoch=1,
        evidence=evidence,
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
    evidence = (
        _evidence(start=0, end=1000, relation="CURRENT"),
        _evidence(start=1000, end=2000, relation="OTHER"),
    )
    units, _, _ = assign_ownership_units(
        tokens,
        events,
        admitted_at_monotonic_s=2.0,
        capture_epoch=1,
        evidence=evidence,
    )
    assert [unit.text for unit in units] == ["same", "same"]
    assert units[0].group_id != units[1].group_id


def test_uncertain_parent_span_abstains_from_partial_partitioning() -> None:
    tokens = (
        _token("straddle", start=900, end=1100),
        _token("?", start=2000, end=2100, timing="unmapped"),
    )
    events = (_event("h1", 1000, slot=1),)
    units, _, reasons = assign_ownership_units(
        tokens, events, admitted_at_monotonic_s=2.0, capture_epoch=1
    )
    assert len(units) == 1
    assert units[0].relation == "UNKNOWN"
    assert units[0].token_indexes == (0, 1)
    assert reasons == ("insufficient_evidence_coverage",)
    assert units[0].text == "straddle?"


def test_late_after_commit_does_not_mutate() -> None:
    parent = uuid4()
    owner = PretranslationOwnershipOwner(enabled=True)
    owner.observe(_event("h1", 1000, slot=1, available=1.0))
    _observe_evidence(owner, _evidence(start=0, end=1000, relation="CURRENT"))
    _observe_evidence(owner, _evidence(start=1000, end=2000, relation="OTHER"))
    tokens = (
        _token("A ", start=0, end=1000),
        _token("B", start=1000, end=2000),
    )
    first = owner.assign(
        parent_utterance_id=parent,
        timed_tokens=tokens,
        capture_epoch=1,
        admitted_at_monotonic_s=2.0,
        parent_text="A B",
    )
    late = owner.observe_after_commit(parent, _event("h2", 1500, slot=2, available=3.0))
    second = owner.assign(
        parent_utterance_id=parent,
        timed_tokens=tokens,
        capture_epoch=1,
        admitted_at_monotonic_s=4.0,
        parent_text="A B",
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
    units, _, reasons = assign_ownership_units(
        tokens, events, admitted_at_monotonic_s=2.0, capture_epoch=1
    )
    assert len(units) == 1
    assert units[0].relation == "UNKNOWN"
    assert "no_reference_evidence" in reasons


def test_gap_epoch_mismatch_does_not_force_assign() -> None:
    tokens = (_token("hello", start=0, end=2000),)
    events = (_event("other-epoch", 1000, epoch=9, slot=2),)
    units, _, _ = assign_ownership_units(
        tokens, events, admitted_at_monotonic_s=2.0, capture_epoch=1
    )
    assert units[0].relation == "UNKNOWN"


def test_disabled_owner_does_not_split() -> None:
    owner = PretranslationOwnershipOwner(enabled=False)
    owner.observe(_event("h1", 1000, slot=1))
    assignment = owner.assign(
        parent_utterance_id=uuid4(),
        timed_tokens=(_token("hello", start=0, end=2000),),
        capture_epoch=1,
        admitted_at_monotonic_s=2.0,
        parent_text="hello",
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


def test_no_events_retain_unknown_not_current() -> None:
    tokens = (_token("hello", start=0, end=2000),)
    units, _, reasons = assign_ownership_units(
        tokens, (), admitted_at_monotonic_s=2.0, capture_epoch=1
    )
    assert units[0].relation == "UNKNOWN"
    assert units[0].group_id == "UNKNOWN-0"
    assert "no_reference_evidence" in reasons


def test_invalid_reference_evidence_does_not_force_current() -> None:
    tokens = (_token("hello", start=0, end=2000),)
    units, _, reasons = assign_ownership_units(
        tokens,
        (_event("stale", 1000, valid=False),),
        admitted_at_monotonic_s=2.0,
        capture_epoch=1,
        evidence=(_evidence(start=0, end=2000, relation="CURRENT", valid=False),),
    )
    assert units[0].relation == "UNKNOWN"
    assert "no_reference_evidence" in reasons


def test_overlapping_evidence_stays_unknown() -> None:
    tokens = (_token("mix", start=0, end=2000),)
    units, _, reasons = assign_ownership_units(
        tokens,
        (),
        admitted_at_monotonic_s=2.0,
        capture_epoch=1,
        evidence=(
            _evidence(start=0, end=2000, relation="CURRENT"),
            _evidence(start=0, end=2000, relation="OTHER"),
        ),
    )
    assert units[0].relation == "UNKNOWN"
    assert "overlap" in reasons


def test_unknown_evidence_stays_unknown() -> None:
    tokens = (_token("mix", start=0, end=2000),)
    units, _, reasons = assign_ownership_units(
        tokens,
        (),
        admitted_at_monotonic_s=2.0,
        capture_epoch=1,
        evidence=(_evidence(start=0, end=2000, relation="UNKNOWN"),),
    )
    assert units[0].relation == "UNKNOWN"
    assert "unknown_evidence" in reasons


def test_unmapped_edges_abstain_instead_of_partially_splitting_middle() -> None:
    tokens = (
        _token("Hi ", start=0, end=500, timing="unmapped"),
        _token("there ", start=500, end=1500),
        _token("Bob ", start=1500, end=2500),
        _token("?", start=2500, end=2600, timing="unmapped"),
    )
    events = (_event("cut", 1500, slot=1),)
    evidence = (
        _evidence(start=500, end=1500, relation="CURRENT"),
        _evidence(start=1500, end=2500, relation="OTHER"),
    )
    units, _, reasons = assign_ownership_units(
        tokens,
        events,
        admitted_at_monotonic_s=2.0,
        capture_epoch=1,
        evidence=evidence,
    )
    assert len(units) == 1
    assert units[0].token_indexes == (0, 1, 2, 3)
    assert units[0].relation == "UNKNOWN"
    assert units[0].text == "Hi there Bob ?"
    assert reasons == ("insufficient_evidence_coverage",)


def test_observe_evidence_labels_covered_intervals() -> None:
    owner = PretranslationOwnershipOwner(enabled=True)
    owner.observe(_event("cut", 1600, slot=1))
    assert (
        owner.observe_evidence(
            capture_epoch=1,
            start_sample=0,
            end_sample=1600,
            available_at_monotonic_s=0.5,
            relation="CURRENT",
            producer_generation=1,
            reference_generation=1,
            reference_valid=True,
        )
        == "observed"
    )
    assert (
        owner.observe_evidence(
            capture_epoch=1,
            start_sample=1600,
            end_sample=3200,
            available_at_monotonic_s=0.5,
            relation="OTHER",
            producer_generation=1,
            reference_generation=1,
            reference_valid=True,
        )
        == "observed"
    )
    assignment = owner.assign(
        parent_utterance_id=uuid4(),
        timed_tokens=(
            _token("Hello ", start=0, end=1600),
            _token("there", start=1600, end=3200),
        ),
        capture_epoch=1,
        admitted_at_monotonic_s=2.0,
        parent_text="Hello there",
    )
    assert assignment.conserved is True
    assert [unit.group_id for unit in assignment.units] == ["CURRENT-0", "OTHER-1"]


def test_mismatched_parent_text_unsplits_instead_of_losing_text() -> None:
    owner = PretranslationOwnershipOwner(enabled=True)
    assignment = owner.assign(
        parent_utterance_id=uuid4(),
        timed_tokens=(
            _token("Hello,", start=0, end=1600),
            _token("world.", start=1600, end=3200),
        ),
        capture_epoch=1,
        admitted_at_monotonic_s=2.0,
        parent_text="Hello, world.",
    )
    assert assignment.disposition == "unsplit"
    assert assignment.units == ()
    assert "unsupported" in assignment.unknown_reasons


def test_observe_evidence_rejects_inverted_bounds() -> None:
    owner = PretranslationOwnershipOwner(enabled=True)
    with pytest.raises(ValueError, match="evidence start follows end"):
        owner.observe_evidence(
            capture_epoch=1,
            start_sample=20,
            end_sample=10,
            available_at_monotonic_s=0.5,
            relation="CURRENT",
            producer_generation=1,
            reference_generation=1,
            reference_valid=True,
        )


def test_late_evidence_is_not_used() -> None:
    tokens = (_token("hello", start=0, end=2000),)
    units, _, reasons = assign_ownership_units(
        tokens,
        (),
        admitted_at_monotonic_s=2.0,
        capture_epoch=1,
        evidence=(_evidence(start=0, end=2000, relation="CURRENT", available=9.0),),
    )
    assert units[0].relation == "UNKNOWN"
    assert "no_reference_evidence" in reasons


def test_partial_reference_support_abstains_and_conserves_parent() -> None:
    tokens = (
        _token("one ", start=0, end=1000),
        _token("two ", start=1000, end=2000),
        _token("three", start=2000, end=3000),
    )
    units, late, reasons = assign_ownership_units(
        tokens,
        (_event("cut", 1000),),
        admitted_at_monotonic_s=2.0,
        capture_epoch=1,
        evidence=(
            _evidence(start=0, end=1000, relation="CURRENT"),
            _evidence(start=1000, end=2000, relation="OTHER"),
        ),
    )
    assert late == ()
    assert len(units) == 1
    assert units[0].token_indexes == (0, 1, 2)
    assert units[0].text == "one two three"
    assert reasons == ("insufficient_evidence_coverage",)


def test_internal_support_hole_does_not_count_as_full_coverage() -> None:
    tokens = (
        _token("before ", start=0, end=1000),
        _token("after", start=1000, end=3000),
    )
    units, _, reasons = assign_ownership_units(
        tokens,
        (_event("cut", 1000),),
        admitted_at_monotonic_s=2.0,
        capture_epoch=1,
        evidence=(
            _evidence(start=0, end=1499, relation="CURRENT"),
            _evidence(start=1500, end=3000, relation="OTHER"),
        ),
    )
    assert len(units) == 1
    assert units[0].text == "before after"
    assert reasons == ("insufficient_evidence_coverage",)


def test_late_transition_cannot_enable_partition_with_full_evidence() -> None:
    tokens = (
        _token("before ", start=0, end=1000),
        _token("after", start=1000, end=2000),
    )
    units, late, reasons = assign_ownership_units(
        tokens,
        (_event("late-cut", 1000, available=3.0),),
        admitted_at_monotonic_s=2.0,
        capture_epoch=1,
        evidence=(
            _evidence(start=0, end=1000, relation="CURRENT"),
            _evidence(start=1000, end=2000, relation="OTHER"),
        ),
    )
    assert len(units) == 1
    assert units[0].text == "before after"
    assert late == ("late-cut",)
    assert reasons == ("no_confirmed_transition",)


def test_coverage_cannot_be_spliced_across_reference_generations() -> None:
    tokens = (
        _token("before ", start=0, end=1000),
        _token("after", start=1000, end=2000),
    )
    units, _, reasons = assign_ownership_units(
        tokens,
        (_event("cut", 1000),),
        admitted_at_monotonic_s=2.0,
        capture_epoch=1,
        evidence=(
            _evidence(start=0, end=1000, relation="CURRENT"),
            _evidence(
                start=1000,
                end=2000,
                relation="OTHER",
                reference_generation=2,
            ),
        ),
    )
    assert len(units) == 1
    assert units[0].text == "before after"
    assert reasons == ("insufficient_evidence_coverage",)
