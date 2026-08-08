"""turn_episode_v1 schemas for the bounded turn-episode speaker-change fusion experiment.

Phase 0 deliverable (approved by the Phase 0 pre-execution review, plan blob
24340f488f1bb46c666a5fc15eef2fc87ef1f826). Implements the design contracts frozen in
``results/turn_episode_v1/reviews/phase_0_review_bundle.md`` Section 6.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

SCHEMA_VERSION = "turn_episode_v1"
PROPOSAL_GENERATION_SCHEMA_VERSION = "turn_episode_v1.proposal_generation"
ACTIONIZATION_SCHEMA_VERSION = "turn_episode_v1.actionization"

FamilyId = Literal["ls_eend", "eres2netv2", "control"]
ProposalKind = Literal[
    "new_track_onset",
    "dominant_replacement",
    "overlap_onset",
    "track_instability",
    "speaker_change_unknown",
]
ClusterOutputKind = Literal[
    "overlap_onset",
    "dominant_replacement",
    "new_track_onset",
    "track_instability",
    "speaker_change_unknown",
]
ReferenceActionKind = Literal[
    "hard_boundary",
    "soft_overlap_marker",
    "state_update",
    "neutral_pause",
    "structural",
    "unscored",
]
EpisodePoolTag = Literal["hard_only", "overlap_present", "negative_only"]
FinalActionKind = Literal[
    "retain_vad",
    "accelerate_or_replace_vad",
    "add_hard_boundary",
    "emit_soft_marker",
    "suppress_detector_duplicate",
    "suppress_vad_duplicate",
    "structural_max_duration",
    "unscored_action",
]
BenefitAttribution = Literal[
    "retained_b0_success",
    "recovered_b0_hard_miss",
    "accelerated_b0_success",
    "correct_soft_marker",
    "late_target_action",
    "hard_miss",
    "soft_miss",
    "none",
]
HarmFlag = Literal[
    "harmful_active_split",
    "lexical_split",
    "same_speaker_pause_split",
    "duplicate_hard_boundary",
    "structural_split",
    "overlap_hard_action",
    "unscored_action",
]
RepresentativeReason = Literal["first", "max_confidence", "fallback_first"]
SuppressionReason = Literal["refractory", "none"]

_FAMILY_IDS: frozenset[str] = frozenset({"ls_eend", "eres2netv2", "control"})
_PROPOSAL_KINDS: frozenset[str] = frozenset(
    {
        "new_track_onset",
        "dominant_replacement",
        "overlap_onset",
        "track_instability",
        "speaker_change_unknown",
    }
)
_REFERENCE_ACTION_KINDS: frozenset[str] = frozenset(
    {
        "hard_boundary",
        "soft_overlap_marker",
        "state_update",
        "neutral_pause",
        "structural",
        "unscored",
    }
)
_EPISODE_POOL_TAGS: frozenset[str] = frozenset({"hard_only", "overlap_present", "negative_only"})
_FINAL_ACTION_KINDS: frozenset[str] = frozenset(
    {
        "retain_vad",
        "accelerate_or_replace_vad",
        "add_hard_boundary",
        "emit_soft_marker",
        "suppress_detector_duplicate",
        "suppress_vad_duplicate",
        "structural_max_duration",
        "unscored_action",
    }
)
_BENEFIT_ATTRIBUTIONS: frozenset[str] = frozenset(
    {
        "retained_b0_success",
        "recovered_b0_hard_miss",
        "accelerated_b0_success",
        "correct_soft_marker",
        "late_target_action",
        "hard_miss",
        "soft_miss",
        "none",
    }
)
_HARM_FLAGS: frozenset[str] = frozenset(
    {
        "harmful_active_split",
        "lexical_split",
        "same_speaker_pause_split",
        "duplicate_hard_boundary",
        "structural_split",
        "overlap_hard_action",
        "unscored_action",
    }
)


class TurnEpisodeSchemaError(ValueError):
    pass


def _require_nonempty(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise TurnEpisodeSchemaError(f"{field_name} must be a non-empty string, got {value!r}")


def _require_epoch(value: Any, field_name: str = "audio_epoch") -> None:
    if not isinstance(value, int) or value < 0:
        raise TurnEpisodeSchemaError(f"{field_name} must be an int >= 0, got {value!r}")


@dataclass(frozen=True, slots=True)
class ProposalEvent:
    proposal_id: str
    family: FamilyId
    checkpoint: str
    profile_id: str
    audio_epoch: int
    source_session_id: str
    proposal_kind: ProposalKind
    boundary_source_sample: int
    observed_source_sample_at_emit: int
    emitted_monotonic_ns: int
    confidence: float | None
    confidence_semantics_id: str
    state_provenance: str
    debug_evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_nonempty(self.proposal_id, "proposal_id")
        if self.family not in _FAMILY_IDS:
            raise TurnEpisodeSchemaError(f"unknown family {self.family!r}")
        _require_nonempty(self.checkpoint, "checkpoint")
        _require_nonempty(self.profile_id, "profile_id")
        _require_epoch(self.audio_epoch)
        _require_nonempty(self.source_session_id, "source_session_id")
        if self.proposal_kind not in _PROPOSAL_KINDS:
            raise TurnEpisodeSchemaError(f"unknown proposal_kind {self.proposal_kind!r}")
        if not isinstance(self.boundary_source_sample, int) or self.boundary_source_sample < 0:
            raise TurnEpisodeSchemaError(
                f"boundary_source_sample must be an int >= 0, got {self.boundary_source_sample!r}"
            )
        if (
            not isinstance(self.observed_source_sample_at_emit, int)
            or self.observed_source_sample_at_emit < 0
        ):
            raise TurnEpisodeSchemaError(
                "observed_source_sample_at_emit must be an int >= 0, got "
                f"{self.observed_source_sample_at_emit!r}"
            )
        if self.boundary_source_sample > self.observed_source_sample_at_emit:
            raise TurnEpisodeSchemaError(
                "observed_source_sample_at_emit must be >= boundary_source_sample "
                f"({self.observed_source_sample_at_emit} < {self.boundary_source_sample})"
            )
        if not isinstance(self.emitted_monotonic_ns, int) or self.emitted_monotonic_ns < 0:
            raise TurnEpisodeSchemaError(
                f"emitted_monotonic_ns must be an int >= 0, got {self.emitted_monotonic_ns!r}"
            )
        if self.confidence is not None and not 0.0 <= float(self.confidence) <= 1.0:
            raise TurnEpisodeSchemaError(
                f"confidence must be in [0, 1] or None, got {self.confidence!r}"
            )
        _require_nonempty(self.confidence_semantics_id, "confidence_semantics_id")
        _require_nonempty(self.state_provenance, "state_provenance")

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "family": self.family,
            "checkpoint": self.checkpoint,
            "profile_id": self.profile_id,
            "audio_epoch": self.audio_epoch,
            "source_session_id": self.source_session_id,
            "proposal_kind": self.proposal_kind,
            "boundary_source_sample": self.boundary_source_sample,
            "observed_source_sample_at_emit": self.observed_source_sample_at_emit,
            "emitted_monotonic_ns": self.emitted_monotonic_ns,
            "confidence": self.confidence,
            "confidence_semantics_id": self.confidence_semantics_id,
            "state_provenance": self.state_provenance,
            "debug_evidence": dict(sorted(self.debug_evidence.items(), key=lambda item: item[0])),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProposalEvent":
        return cls(
            proposal_id=str(data["proposal_id"]),
            family=str(data["family"]),
            checkpoint=str(data["checkpoint"]),
            profile_id=str(data["profile_id"]),
            audio_epoch=int(data["audio_epoch"]),
            source_session_id=str(data["source_session_id"]),
            proposal_kind=str(data["proposal_kind"]),
            boundary_source_sample=int(data["boundary_source_sample"]),
            observed_source_sample_at_emit=int(data["observed_source_sample_at_emit"]),
            emitted_monotonic_ns=int(data["emitted_monotonic_ns"]),
            confidence=data.get("confidence"),
            confidence_semantics_id=str(data["confidence_semantics_id"]),
            state_provenance=str(data["state_provenance"]),
            debug_evidence=dict(data.get("debug_evidence") or {}),
        )


@dataclass(frozen=True, slots=True)
class DetectorProgress:
    audio_epoch: int
    observed_source_sample: int
    safe_boundary_frontier_sample: int

    def __post_init__(self) -> None:
        _require_epoch(self.audio_epoch)
        if not isinstance(self.observed_source_sample, int) or self.observed_source_sample < 0:
            raise TurnEpisodeSchemaError(
                f"observed_source_sample must be an int >= 0, got {self.observed_source_sample!r}"
            )
        if (
            not isinstance(self.safe_boundary_frontier_sample, int)
            or self.safe_boundary_frontier_sample < 0
        ):
            raise TurnEpisodeSchemaError(
                "safe_boundary_frontier_sample must be an int >= 0, got "
                f"{self.safe_boundary_frontier_sample!r}"
            )
        if self.safe_boundary_frontier_sample > self.observed_source_sample:
            raise TurnEpisodeSchemaError(
                "safe_boundary_frontier_sample cannot exceed observed_source_sample "
                f"({self.safe_boundary_frontier_sample} > {self.observed_source_sample})"
            )

    def to_dict(self) -> dict[str, int]:
        return {
            "audio_epoch": self.audio_epoch,
            "observed_source_sample": self.observed_source_sample,
            "safe_boundary_frontier_sample": self.safe_boundary_frontier_sample,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DetectorProgress":
        return cls(
            audio_epoch=int(data["audio_epoch"]),
            observed_source_sample=int(data["observed_source_sample"]),
            safe_boundary_frontier_sample=int(data["safe_boundary_frontier_sample"]),
        )


@dataclass(frozen=True, slots=True)
class ReferenceAction:
    reference_id: str
    audio_epoch: int
    source_session_id: str
    action_kind: ReferenceActionKind
    target_sample: int | None
    acceptable_interval: tuple[int, int]
    evidence_onset_sample: int
    scorable: bool
    primary_case: bool
    episode_pool_tag: EpisodePoolTag

    def __post_init__(self) -> None:
        _require_nonempty(self.reference_id, "reference_id")
        _require_epoch(self.audio_epoch)
        _require_nonempty(self.source_session_id, "source_session_id")
        if self.action_kind not in _REFERENCE_ACTION_KINDS:
            raise TurnEpisodeSchemaError(f"unknown action_kind {self.action_kind!r}")
        if self.episode_pool_tag not in _EPISODE_POOL_TAGS:
            raise TurnEpisodeSchemaError(f"unknown episode_pool_tag {self.episode_pool_tag!r}")
        start, end = self.acceptable_interval
        if not isinstance(start, int) or not isinstance(end, int):
            raise TurnEpisodeSchemaError(
                f"acceptable_interval must be ints, got {self.acceptable_interval!r}"
            )
        if start < 0 or end < start:
            raise TurnEpisodeSchemaError(
                f"acceptable_interval must satisfy 0 <= start <= end, got {self.acceptable_interval!r}"
            )
        if not isinstance(self.evidence_onset_sample, int) or self.evidence_onset_sample < 0:
            raise TurnEpisodeSchemaError(
                f"evidence_onset_sample must be an int >= 0, got {self.evidence_onset_sample!r}"
            )
        if self.target_sample is not None and (
            not isinstance(self.target_sample, int) or self.target_sample < 0
        ):
            raise TurnEpisodeSchemaError(
                f"target_sample must be an int >= 0 or None, got {self.target_sample!r}"
            )
        if self.action_kind in ("hard_boundary", "soft_overlap_marker"):
            if self.target_sample is None:
                raise TurnEpisodeSchemaError(
                    f"{self.action_kind} requires target_sample (B onset), got None"
                )
            if self.evidence_onset_sample != self.target_sample:
                raise TurnEpisodeSchemaError(
                    "detector-evidence onset must equal B onset target_sample "
                    f"({self.evidence_onset_sample} != {self.target_sample})"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_id": self.reference_id,
            "audio_epoch": self.audio_epoch,
            "source_session_id": self.source_session_id,
            "action_kind": self.action_kind,
            "target_sample": self.target_sample,
            "acceptable_interval": list(self.acceptable_interval),
            "evidence_onset_sample": self.evidence_onset_sample,
            "scorable": self.scorable,
            "primary_case": self.primary_case,
            "episode_pool_tag": self.episode_pool_tag,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReferenceAction":
        interval = data["acceptable_interval"]
        return cls(
            reference_id=str(data["reference_id"]),
            audio_epoch=int(data["audio_epoch"]),
            source_session_id=str(data["source_session_id"]),
            action_kind=str(data["action_kind"]),
            target_sample=data.get("target_sample"),
            acceptable_interval=(int(interval[0]), int(interval[1])),
            evidence_onset_sample=int(data["evidence_onset_sample"]),
            scorable=bool(data["scorable"]),
            primary_case=bool(data["primary_case"]),
            episode_pool_tag=str(data["episode_pool_tag"]),
        )


@dataclass(frozen=True, slots=True)
class ReferenceOutcome:
    reference_id: str
    matched_action_id: str | None
    benefit_attribution: BenefitAttribution
    availability_delay_ms: int | None
    localization_error_ms: int | None

    def __post_init__(self) -> None:
        _require_nonempty(self.reference_id, "reference_id")
        if self.matched_action_id is not None:
            _require_nonempty(self.matched_action_id, "matched_action_id")
        if self.benefit_attribution not in _BENEFIT_ATTRIBUTIONS:
            raise TurnEpisodeSchemaError(
                f"unknown benefit_attribution {self.benefit_attribution!r}"
            )
        if self.availability_delay_ms is not None and self.availability_delay_ms < 0:
            raise TurnEpisodeSchemaError(
                f"availability_delay_ms must be >= 0 or None, got {self.availability_delay_ms!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_id": self.reference_id,
            "matched_action_id": self.matched_action_id,
            "benefit_attribution": self.benefit_attribution,
            "availability_delay_ms": self.availability_delay_ms,
            "localization_error_ms": self.localization_error_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReferenceOutcome":
        return cls(
            reference_id=str(data["reference_id"]),
            matched_action_id=data.get("matched_action_id"),
            benefit_attribution=str(data["benefit_attribution"]),
            availability_delay_ms=data.get("availability_delay_ms"),
            localization_error_ms=data.get("localization_error_ms"),
        )


@dataclass(frozen=True, slots=True)
class FinalAction:
    action_id: str
    audio_epoch: int
    source_session_id: str
    action_kind: FinalActionKind
    boundary_source_sample: int | None
    observed_source_sample_at_emit: int | None
    emitted_monotonic_ns: int | None
    availability_source_sample: int
    cluster_id: str | None
    matched_reference_id: str | None
    harm_or_structure_flags: tuple[HarmFlag, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _require_nonempty(self.action_id, "action_id")
        _require_epoch(self.audio_epoch)
        _require_nonempty(self.source_session_id, "source_session_id")
        if self.action_kind not in _FINAL_ACTION_KINDS:
            raise TurnEpisodeSchemaError(f"unknown action_kind {self.action_kind!r}")
        if (
            not isinstance(self.availability_source_sample, int)
            or self.availability_source_sample < 0
        ):
            raise TurnEpisodeSchemaError(
                f"availability_source_sample must be an int >= 0, got {self.availability_source_sample!r}"
            )
        for flag in self.harm_or_structure_flags:
            if flag not in _HARM_FLAGS:
                raise TurnEpisodeSchemaError(f"unknown harm flag {flag!r}")
        if self.matched_reference_id is not None:
            _require_nonempty(self.matched_reference_id, "matched_reference_id")
        hard_kinds = {"retain_vad", "accelerate_or_replace_vad", "add_hard_boundary"}
        if self.action_kind in hard_kinds:
            if self.boundary_source_sample is None:
                raise TurnEpisodeSchemaError(
                    f"hard action kind {self.action_kind} requires boundary_source_sample"
                )
        if self.boundary_source_sample is not None:
            if self.boundary_source_sample < 0:
                raise TurnEpisodeSchemaError(
                    f"boundary_source_sample must be >= 0 or None, got {self.boundary_source_sample!r}"
                )
            if self.availability_source_sample < self.boundary_source_sample:
                raise TurnEpisodeSchemaError(
                    "availability_source_sample cannot precede boundary_source_sample "
                    f"({self.availability_source_sample} < {self.boundary_source_sample})"
                )
            if (
                self.observed_source_sample_at_emit is not None
                and self.observed_source_sample_at_emit < self.boundary_source_sample
            ):
                raise TurnEpisodeSchemaError(
                    "observed_source_sample_at_emit cannot precede boundary_source_sample"
                )
        if (
            self.observed_source_sample_at_emit is not None
            and self.observed_source_sample_at_emit < 0
        ):
            raise TurnEpisodeSchemaError(
                "observed_source_sample_at_emit must be >= 0 or None, got "
                f"{self.observed_source_sample_at_emit!r}"
            )
        if self.emitted_monotonic_ns is not None and self.emitted_monotonic_ns < 0:
            raise TurnEpisodeSchemaError(
                f"emitted_monotonic_ns must be >= 0 or None, got {self.emitted_monotonic_ns!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "audio_epoch": self.audio_epoch,
            "source_session_id": self.source_session_id,
            "action_kind": self.action_kind,
            "boundary_source_sample": self.boundary_source_sample,
            "observed_source_sample_at_emit": self.observed_source_sample_at_emit,
            "emitted_monotonic_ns": self.emitted_monotonic_ns,
            "availability_source_sample": self.availability_source_sample,
            "cluster_id": self.cluster_id,
            "matched_reference_id": self.matched_reference_id,
            "harm_or_structure_flags": list(self.harm_or_structure_flags),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FinalAction":
        return cls(
            action_id=str(data["action_id"]),
            audio_epoch=int(data["audio_epoch"]),
            source_session_id=str(data["source_session_id"]),
            action_kind=str(data["action_kind"]),
            boundary_source_sample=data.get("boundary_source_sample"),
            observed_source_sample_at_emit=data.get("observed_source_sample_at_emit"),
            emitted_monotonic_ns=data.get("emitted_monotonic_ns"),
            availability_source_sample=int(data["availability_source_sample"]),
            cluster_id=data.get("cluster_id"),
            matched_reference_id=data.get("matched_reference_id"),
            harm_or_structure_flags=tuple(data.get("harm_or_structure_flags") or []),
        )


@dataclass(frozen=True, slots=True)
class LogicalBoundaryCluster:
    cluster_id: str
    audio_epoch: int
    source_session_id: str
    member_proposal_ids: tuple[str, ...]
    output_kind: ClusterOutputKind
    compatible_representative_subset: tuple[str, ...]
    representative_proposal_id: str
    representative_reason: RepresentativeReason
    confidence_semantics_id: str
    suppression_reason: SuppressionReason
    open_frontier_sample: int
    close_frontier_sample: int
    availability_source_sample: int
    boundary_spread_samples: int
    confidence_distribution: tuple[float, ...]
    refractory_owner_cluster_id: str | None = None
    tail_closed: bool = False

    def __post_init__(self) -> None:
        _require_nonempty(self.cluster_id, "cluster_id")
        _require_epoch(self.audio_epoch)
        _require_nonempty(self.source_session_id, "source_session_id")
        if not self.member_proposal_ids:
            raise TurnEpisodeSchemaError("member_proposal_ids must not be empty")
        for proposal_id in self.member_proposal_ids:
            _require_nonempty(proposal_id, "member proposal id")
        if len(set(self.member_proposal_ids)) != len(self.member_proposal_ids):
            raise TurnEpisodeSchemaError("member_proposal_ids must not contain duplicates")
        if self.output_kind not in _PROPOSAL_KINDS:
            raise TurnEpisodeSchemaError(f"unknown output_kind {self.output_kind!r}")
        members = set(self.member_proposal_ids)
        if not self.compatible_representative_subset:
            raise TurnEpisodeSchemaError("compatible_representative_subset must not be empty")
        subset = set(self.compatible_representative_subset)
        if not subset <= members:
            raise TurnEpisodeSchemaError(
                "compatible_representative_subset must be a subset of member_proposal_ids"
            )
        if self.representative_proposal_id not in subset:
            raise TurnEpisodeSchemaError(
                "representative_proposal_id must belong to compatible_representative_subset"
            )
        if self.representative_reason not in ("first", "max_confidence", "fallback_first"):
            raise TurnEpisodeSchemaError(
                f"unknown representative_reason {self.representative_reason!r}"
            )
        _require_nonempty(self.confidence_semantics_id, "confidence_semantics_id")
        if self.suppression_reason not in ("refractory", "none"):
            raise TurnEpisodeSchemaError(f"unknown suppression_reason {self.suppression_reason!r}")
        for name, value in (
            ("open_frontier_sample", self.open_frontier_sample),
            ("close_frontier_sample", self.close_frontier_sample),
            ("availability_source_sample", self.availability_source_sample),
            ("boundary_spread_samples", self.boundary_spread_samples),
        ):
            if not isinstance(value, int) or value < 0:
                raise TurnEpisodeSchemaError(f"{name} must be an int >= 0, got {value!r}")
        if self.close_frontier_sample < self.open_frontier_sample:
            raise TurnEpisodeSchemaError(
                "close_frontier_sample cannot precede open_frontier_sample"
            )
        if self.availability_source_sample < self.close_frontier_sample:
            raise TurnEpisodeSchemaError(
                "availability_source_sample cannot precede close_frontier_sample"
            )
        for value in self.confidence_distribution:
            if not 0.0 <= float(value) <= 1.0:
                raise TurnEpisodeSchemaError(
                    f"confidence_distribution values must be in [0, 1], got {value!r}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "audio_epoch": self.audio_epoch,
            "source_session_id": self.source_session_id,
            "member_proposal_ids": list(self.member_proposal_ids),
            "output_kind": self.output_kind,
            "compatible_representative_subset": list(self.compatible_representative_subset),
            "representative_proposal_id": self.representative_proposal_id,
            "representative_reason": self.representative_reason,
            "confidence_semantics_id": self.confidence_semantics_id,
            "suppression_reason": self.suppression_reason,
            "open_frontier_sample": self.open_frontier_sample,
            "close_frontier_sample": self.close_frontier_sample,
            "availability_source_sample": self.availability_source_sample,
            "boundary_spread_samples": self.boundary_spread_samples,
            "confidence_distribution": list(self.confidence_distribution),
            "refractory_owner_cluster_id": self.refractory_owner_cluster_id,
            "tail_closed": self.tail_closed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LogicalBoundaryCluster":
        return cls(
            cluster_id=str(data["cluster_id"]),
            audio_epoch=int(data["audio_epoch"]),
            source_session_id=str(data["source_session_id"]),
            member_proposal_ids=tuple(str(item) for item in data["member_proposal_ids"]),
            output_kind=str(data["output_kind"]),
            compatible_representative_subset=tuple(
                str(item) for item in data["compatible_representative_subset"]
            ),
            representative_proposal_id=str(data["representative_proposal_id"]),
            representative_reason=str(data["representative_reason"]),
            confidence_semantics_id=str(data["confidence_semantics_id"]),
            suppression_reason=str(data["suppression_reason"]),
            open_frontier_sample=int(data["open_frontier_sample"]),
            close_frontier_sample=int(data["close_frontier_sample"]),
            availability_source_sample=int(data["availability_source_sample"]),
            boundary_spread_samples=int(data["boundary_spread_samples"]),
            confidence_distribution=tuple(float(v) for v in data["confidence_distribution"]),
            refractory_owner_cluster_id=data.get("refractory_owner_cluster_id"),
            tail_closed=bool(data.get("tail_closed", False)),
        )
