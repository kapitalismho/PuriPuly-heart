from __future__ import annotations

import hashlib
import json
import numbers
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

CONTRACT_PATH = Path(__file__).with_name("operational_label_contract.json")
EXPECTED_CONTRACT_DOCUMENT_SHA256_BY_VERSION = {
    "psem-handoff-v0": "7cbb831e1513af80827daaf4f63548eb243fd3d842ad5dd2a5c7e8c7a2812fb6",
}


class LabelContractError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class CanonicalInterval:
    start_sample: int
    end_sample: int
    active_speakers: tuple[str, ...]
    ambiguous: bool = False
    speaker_identity_known: bool = True
    source_annotation_ids: tuple[str, ...] = ()

    @property
    def duration_samples(self) -> int:
        return self.end_sample - self.start_sample

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "active_speakers": list(self.active_speakers),
            "ambiguous": self.ambiguous,
            "speaker_identity_known": self.speaker_identity_known,
            "source_annotation_ids": list(self.source_annotation_ids),
        }


@dataclass(frozen=True, slots=True)
class LabelContract:
    schema_version: int
    contract_version: str
    document_sha256: str
    sample_rate_hz: int
    coordinate_convention: str
    grid_mapping: str
    status: str
    reliable_solo_min_duration_ms: int
    annotation_boundary_jitter_ms: int
    gap_topology_min_duration_ms: int
    overlap_topology_min_duration_ms: int
    local_continuity_max_gap_ms: int
    short_backchannel_min_duration_ms: int
    short_backchannel_max_duration_ms: int

    def samples(self, milliseconds: int) -> int:
        numerator = milliseconds * self.sample_rate_hz
        if numerator % 1000:
            raise LabelContractError(
                f"{milliseconds} ms is not sample-exact at {self.sample_rate_hz} Hz"
            )
        return numerator // 1000

    @property
    def reliable_solo_min_duration_samples(self) -> int:
        return self.samples(self.reliable_solo_min_duration_ms)

    @property
    def annotation_boundary_jitter_samples(self) -> int:
        return self.samples(self.annotation_boundary_jitter_ms)

    @property
    def gap_topology_min_duration_samples(self) -> int:
        return self.samples(self.gap_topology_min_duration_ms)

    @property
    def overlap_topology_min_duration_samples(self) -> int:
        return self.samples(self.overlap_topology_min_duration_ms)

    @property
    def local_continuity_max_gap_samples(self) -> int:
        return self.samples(self.local_continuity_max_gap_ms)

    @property
    def short_backchannel_min_duration_samples(self) -> int:
        return self.samples(self.short_backchannel_min_duration_ms)

    @property
    def short_backchannel_max_duration_samples(self) -> int:
        return self.samples(self.short_backchannel_max_duration_ms)


@dataclass(frozen=True, slots=True)
class LabelResult:
    contract_version: str
    contract_document_sha256: str
    sample_rate_hz: int
    intervals: tuple[CanonicalInterval, ...]
    activity_labels: tuple[dict[str, Any], ...]
    transitions: tuple[dict[str, Any], ...]
    topology_episodes: tuple[dict[str, Any], ...]
    exposure: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "contract_document_sha256": self.contract_document_sha256,
            "sample_rate_hz": self.sample_rate_hz,
            "intervals": [interval.to_dict() for interval in self.intervals],
            "activity_labels": [dict(row) for row in self.activity_labels],
            "transitions": [dict(row) for row in self.transitions],
            "topology_episodes": [dict(row) for row in self.topology_episodes],
            "exposure": dict(self.exposure),
        }


def _validate_contract(contract: LabelContract) -> LabelContract:
    integer_fields = {
        "schema_version": contract.schema_version,
        "sample_rate_hz": contract.sample_rate_hz,
        "reliable_solo_min_duration_ms": contract.reliable_solo_min_duration_ms,
        "annotation_boundary_jitter_ms": contract.annotation_boundary_jitter_ms,
        "gap_topology_min_duration_ms": contract.gap_topology_min_duration_ms,
        "overlap_topology_min_duration_ms": contract.overlap_topology_min_duration_ms,
        "local_continuity_max_gap_ms": contract.local_continuity_max_gap_ms,
        "short_backchannel_min_duration_ms": contract.short_backchannel_min_duration_ms,
        "short_backchannel_max_duration_ms": contract.short_backchannel_max_duration_ms,
    }
    for field, value in integer_fields.items():
        _exact_integer(value, field)
    if contract.schema_version != 1:
        raise LabelContractError(f"unsupported contract schema {contract.schema_version}")
    if not re.fullmatch(r"psem-handoff-v[0-9]+", contract.contract_version):
        raise LabelContractError("invalid PSEM contract version")
    if not re.fullmatch(r"[0-9a-f]{64}", contract.document_sha256):
        raise LabelContractError("invalid contract document SHA-256")
    if not contract.status:
        raise LabelContractError("contract status must be non-empty")
    if contract.sample_rate_hz != 16000:
        raise LabelContractError("PSEM source timeline must be 16 kHz")
    if contract.coordinate_convention != "zero_based_half_open_unsnapped_source_samples":
        raise LabelContractError(
            "PSEM source coordinates must be zero-based half-open unsnapped samples"
        )
    if contract.grid_mapping != "forbidden_in_dataset_labels":
        raise LabelContractError("training-grid mapping is forbidden in dataset labels")
    if contract.local_continuity_max_gap_ms != 1200:
        raise LabelContractError("local continuity maximum is the hard #76 value 1200 ms")
    for value in (
        contract.reliable_solo_min_duration_ms,
        contract.annotation_boundary_jitter_ms,
        contract.gap_topology_min_duration_ms,
        contract.overlap_topology_min_duration_ms,
        contract.short_backchannel_min_duration_ms,
        contract.short_backchannel_max_duration_ms,
    ):
        if value <= 0:
            raise LabelContractError("all operational durations must be positive")
        contract.samples(value)
    if contract.annotation_boundary_jitter_ms >= contract.gap_topology_min_duration_ms:
        raise LabelContractError("gap jitter must be below the official gap minimum")
    if contract.annotation_boundary_jitter_ms >= contract.overlap_topology_min_duration_ms:
        raise LabelContractError("overlap jitter must be below the official overlap minimum")
    if contract.short_backchannel_min_duration_ms > contract.short_backchannel_max_duration_ms:
        raise LabelContractError("short backchannel duration bounds are reversed")
    return contract


def load_contract(path: Path | None = None) -> LabelContract:
    installed_document = path is None
    contract_path = path or CONTRACT_PATH
    raw = json.loads(contract_path.read_text(encoding="utf-8"))
    coordinates = raw["source_coordinate_convention"]
    constants = raw["constants_ms"]
    document_sha256 = hashlib.sha256(
        json.dumps(
            raw,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    contract = LabelContract(
        schema_version=_exact_integer(raw["schema_version"], "schema_version"),
        contract_version=_exact_string(raw["contract_version"], "contract_version"),
        document_sha256=document_sha256,
        sample_rate_hz=_exact_integer(coordinates["sample_rate_hz"], "sample_rate_hz"),
        coordinate_convention=_exact_string(coordinates["coordinates"], "coordinates"),
        grid_mapping=_exact_string(coordinates["grid_mapping"], "grid_mapping"),
        status=_exact_string(raw["status"], "status"),
        reliable_solo_min_duration_ms=_exact_integer(
            constants["reliable_solo_min_duration"], "reliable_solo_min_duration"
        ),
        annotation_boundary_jitter_ms=_exact_integer(
            constants["annotation_boundary_jitter"], "annotation_boundary_jitter"
        ),
        gap_topology_min_duration_ms=_exact_integer(
            constants["gap_topology_min_duration"], "gap_topology_min_duration"
        ),
        overlap_topology_min_duration_ms=_exact_integer(
            constants["overlap_topology_min_duration"], "overlap_topology_min_duration"
        ),
        local_continuity_max_gap_ms=_exact_integer(
            constants["local_continuity_max_gap"], "local_continuity_max_gap"
        ),
        short_backchannel_min_duration_ms=_exact_integer(
            constants["short_backchannel_min_duration"], "short_backchannel_min_duration"
        ),
        short_backchannel_max_duration_ms=_exact_integer(
            constants["short_backchannel_max_duration"], "short_backchannel_max_duration"
        ),
    )
    validated = _validate_contract(contract)
    if installed_document:
        expected_sha256 = EXPECTED_CONTRACT_DOCUMENT_SHA256_BY_VERSION.get(
            validated.contract_version
        )
        if expected_sha256 is None or validated.document_sha256 != expected_sha256:
            raise LabelContractError(
                "installed contract document does not match its pinned version identity"
            )
    return validated


def _coerce_interval(value: CanonicalInterval | Mapping[str, Any]) -> CanonicalInterval:
    if isinstance(value, CanonicalInterval):
        start_sample = value.start_sample
        end_sample = value.end_sample
        active_speakers = value.active_speakers
        ambiguous = value.ambiguous
        speaker_identity_known = value.speaker_identity_known
        source_annotation_ids = value.source_annotation_ids
    elif isinstance(value, Mapping):
        start_sample = value["start_sample"]
        end_sample = value["end_sample"]
        active_speakers = value.get("active_speakers", [])
        ambiguous = value.get("ambiguous", False)
        speaker_identity_known = value.get("speaker_identity_known", True)
        source_annotation_ids = value.get("source_annotation_ids", [])
    else:
        raise LabelContractError("canonical interval must be a mapping or CanonicalInterval")
    speakers = tuple(sorted(_exact_string_sequence(active_speakers, "active_speakers")))
    annotation_ids = _exact_string_sequence(source_annotation_ids, "source_annotation_ids")
    return CanonicalInterval(
        start_sample=_exact_sample(start_sample, "start_sample"),
        end_sample=_exact_sample(end_sample, "end_sample"),
        active_speakers=speakers,
        ambiguous=_exact_bool(ambiguous, "ambiguous"),
        speaker_identity_known=_exact_bool(speaker_identity_known, "speaker_identity_known"),
        source_annotation_ids=annotation_ids,
    )


def _exact_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise LabelContractError(f"{field} must be an exact integer")
    return int(value)


def _exact_sample(value: Any, field: str) -> int:
    try:
        return _exact_integer(value, field)
    except LabelContractError as error:
        raise LabelContractError(f"{field} must be an exact integer source sample") from error


def _exact_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise LabelContractError(f"{field} must be a non-empty string")
    return value


def _exact_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise LabelContractError(f"{field} must be a boolean")
    return value


def _exact_string_sequence(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise LabelContractError(f"{field} must be a list or tuple of strings")
    items = tuple(_exact_string(item, field) for item in value)
    if len(set(items)) != len(items):
        raise LabelContractError(f"{field} must not contain duplicates")
    return items


def normalize_intervals(
    values: Iterable[CanonicalInterval | Mapping[str, Any]],
    *,
    scored_start_sample: int | None = None,
    scored_end_sample: int | None = None,
) -> tuple[CanonicalInterval, ...]:
    intervals = [_coerce_interval(value) for value in values]
    if not intervals:
        raise LabelContractError("at least one canonical interval is required")
    if any(
        interval.start_sample < previous.start_sample
        for previous, interval in zip(intervals, intervals[1:])
    ):
        raise LabelContractError("canonical intervals must be provided in source order")
    expected_start = intervals[0].start_sample
    if scored_start_sample is not None:
        expected_start = _exact_sample(scored_start_sample, "scored_start_sample")
    expected_end = intervals[-1].end_sample
    if scored_end_sample is not None:
        expected_end = _exact_sample(scored_end_sample, "scored_end_sample")
    if expected_start < 0 or expected_end <= expected_start:
        raise LabelContractError("invalid scored source range")
    merged: list[CanonicalInterval] = []
    cursor = expected_start
    for interval in intervals:
        _exact_sample(interval.start_sample, "start_sample")
        _exact_sample(interval.end_sample, "end_sample")
        if interval.start_sample != cursor:
            relation = "gap" if interval.start_sample > cursor else "overlap"
            raise LabelContractError(f"canonical timeline has a {relation} at sample {cursor}")
        if interval.end_sample <= interval.start_sample:
            raise LabelContractError("canonical intervals must have positive duration")
        if interval.end_sample > expected_end:
            raise LabelContractError("canonical interval extends beyond scored range")
        if any(not speaker for speaker in interval.active_speakers):
            raise LabelContractError("speaker IDs must be non-empty")
        if (
            merged
            and merged[-1].active_speakers == interval.active_speakers
            and merged[-1].ambiguous == interval.ambiguous
            and merged[-1].speaker_identity_known == interval.speaker_identity_known
        ):
            previous = merged[-1]
            merged[-1] = CanonicalInterval(
                start_sample=previous.start_sample,
                end_sample=interval.end_sample,
                active_speakers=previous.active_speakers,
                ambiguous=previous.ambiguous,
                speaker_identity_known=previous.speaker_identity_known,
                source_annotation_ids=tuple(
                    dict.fromkeys(previous.source_annotation_ids + interval.source_annotation_ids)
                ),
            )
        else:
            merged.append(interval)
        cursor = interval.end_sample
    if cursor != expected_end:
        raise LabelContractError("canonical timeline does not cover the scored range")
    return tuple(merged)


def _reconcile_same_speaker_jitter_gaps(
    intervals: Sequence[CanonicalInterval],
    contract: LabelContract,
) -> tuple[CanonicalInterval, ...]:
    reconciled = list(intervals)
    index = 0
    while index + 2 < len(reconciled):
        left = reconciled[index]
        gap = reconciled[index + 1]
        right = reconciled[index + 2]
        collapsible = (
            not left.ambiguous
            and not gap.ambiguous
            and not right.ambiguous
            and left.speaker_identity_known
            and gap.speaker_identity_known
            and right.speaker_identity_known
            and len(left.active_speakers) == 1
            and not gap.active_speakers
            and right.active_speakers == left.active_speakers
            and gap.duration_samples <= contract.annotation_boundary_jitter_samples
        )
        if not collapsible:
            index += 1
            continue
        reconciled[index : index + 3] = [
            CanonicalInterval(
                start_sample=left.start_sample,
                end_sample=right.end_sample,
                active_speakers=left.active_speakers,
                source_annotation_ids=tuple(
                    dict.fromkeys(
                        left.source_annotation_ids
                        + gap.source_annotation_ids
                        + right.source_annotation_ids
                    )
                ),
            )
        ]
    return tuple(reconciled)


def _activity_state(interval: CanonicalInterval) -> str:
    if interval.ambiguous:
        return "ambiguous"
    if not interval.active_speakers:
        return "silence" if interval.speaker_identity_known else "unknown_speech"
    if len(interval.active_speakers) == 1:
        return "singleton"
    return "overlap"


def _is_complex_overlap_boundary(
    left: CanonicalInterval,
    right: CanonicalInterval,
) -> bool:
    return (
        not left.ambiguous
        and not right.ambiguous
        and len(left.active_speakers) >= 2
        and len(right.active_speakers) >= 2
        and left.active_speakers != right.active_speakers
        and len(set(left.active_speakers) | set(right.active_speakers)) >= 3
    )


def _is_reliable_solo(interval: CanonicalInterval, contract: LabelContract) -> bool:
    return (
        not interval.ambiguous
        and interval.speaker_identity_known
        and len(interval.active_speakers) == 1
        and interval.duration_samples >= contract.reliable_solo_min_duration_samples
    )


def _transition_row(
    *,
    transition_id: str,
    from_interval_index: int | None,
    to_interval_index: int | None,
    from_speaker: str | None,
    to_speaker: str | None,
    source_sample: int | None,
    handoff_target: int | None,
    relation_target: str | None,
    mask_state: str,
    primary_topology: str,
    secondary_tags: Sequence[str] = (),
    coverage_gate_eligible: bool = False,
    gap_samples: int = 0,
    overlap_samples: int = 0,
) -> dict[str, Any]:
    return {
        "transition_id": transition_id,
        "from_interval_index": from_interval_index,
        "to_interval_index": to_interval_index,
        "from_speaker": from_speaker,
        "to_speaker": to_speaker,
        "handoff_confirmed": handoff_target,
        "handoff_source_sample": source_sample,
        "relation_target": relation_target,
        "mask_state": mask_state,
        "primary_topology": primary_topology,
        "secondary_tags": list(secondary_tags),
        "coverage_gate_eligible": coverage_gate_eligible,
        "gap_samples": gap_samples,
        "overlap_samples": overlap_samples,
    }


def _classify_transition(
    intervals: Sequence[CanonicalInterval],
    previous_index: int,
    current_index: int,
    contract: LabelContract,
    transition_id: str,
) -> dict[str, Any]:
    previous = intervals[previous_index]
    current = intervals[current_index]
    previous_speaker = previous.active_speakers[0]
    current_speaker = current.active_speakers[0]
    same_speaker = previous_speaker == current_speaker
    between = intervals[previous_index + 1 : current_index]
    gap_samples = sum(
        interval.duration_samples
        for interval in between
        if not interval.active_speakers and not interval.ambiguous
    )
    overlap_samples = sum(
        interval.duration_samples
        for interval in between
        if len(interval.active_speakers) >= 2 and not interval.ambiguous
    )
    if any(interval.ambiguous for interval in between):
        return _transition_row(
            transition_id=transition_id,
            from_interval_index=previous_index,
            to_interval_index=current_index,
            from_speaker=previous_speaker,
            to_speaker=current_speaker,
            source_sample=None,
            handoff_target=None,
            relation_target=None,
            mask_state="masked",
            primary_topology="ambiguous_annotation_crossing",
            secondary_tags=("ambiguous",),
            gap_samples=gap_samples,
            overlap_samples=overlap_samples,
        )
    if any(not interval.speaker_identity_known for interval in between):
        return _transition_row(
            transition_id=transition_id,
            from_interval_index=previous_index,
            to_interval_index=current_index,
            from_speaker=previous_speaker,
            to_speaker=current_speaker,
            source_sample=None,
            handoff_target=None,
            relation_target=None,
            mask_state="masked",
            primary_topology="unknown_speaker_crossing",
            secondary_tags=("unknown_speaker_identity",),
            gap_samples=gap_samples,
            overlap_samples=overlap_samples,
        )
    if not between:
        return _transition_row(
            transition_id=transition_id,
            from_interval_index=previous_index,
            to_interval_index=current_index,
            from_speaker=previous_speaker,
            to_speaker=current_speaker,
            source_sample=None if same_speaker else current.start_sample,
            handoff_target=0 if same_speaker else 1,
            relation_target="same" if same_speaker else "different",
            mask_state="valid",
            primary_topology=(
                "same_speaker_direct_continuation"
                if same_speaker
                else "clean_direct_different_speaker_handoff"
            ),
            coverage_gate_eligible=not same_speaker,
        )
    if all(not interval.active_speakers for interval in between):
        if gap_samples > contract.local_continuity_max_gap_samples:
            return _transition_row(
                transition_id=transition_id,
                from_interval_index=previous_index,
                to_interval_index=current_index,
                from_speaker=previous_speaker,
                to_speaker=current_speaker,
                source_sample=None,
                handoff_target=None,
                relation_target=None,
                mask_state="masked",
                primary_topology="continuity_unknown",
                secondary_tags=("long_gap",),
                gap_samples=gap_samples,
            )
        if gap_samples <= contract.annotation_boundary_jitter_samples:
            return _transition_row(
                transition_id=transition_id,
                from_interval_index=previous_index,
                to_interval_index=current_index,
                from_speaker=previous_speaker,
                to_speaker=current_speaker,
                source_sample=None if same_speaker else current.start_sample,
                handoff_target=0 if same_speaker else 1,
                relation_target="same" if same_speaker else "different",
                mask_state="valid",
                primary_topology=(
                    "same_speaker_direct_continuation"
                    if same_speaker
                    else "clean_direct_different_speaker_handoff"
                ),
                secondary_tags=("boundary_jitter_gap",),
                coverage_gate_eligible=not same_speaker,
                gap_samples=gap_samples,
            )
        if gap_samples < contract.gap_topology_min_duration_samples:
            return _transition_row(
                transition_id=transition_id,
                from_interval_index=previous_index,
                to_interval_index=current_index,
                from_speaker=previous_speaker,
                to_speaker=current_speaker,
                source_sample=None if same_speaker else current.start_sample,
                handoff_target=0 if same_speaker else 1,
                relation_target="same" if same_speaker else "different",
                mask_state="valid",
                primary_topology=(
                    "micro_gap_same_speaker_resume"
                    if same_speaker
                    else "micro_gap_different_speaker_handoff"
                ),
                secondary_tags=("micro_gap",),
                gap_samples=gap_samples,
            )
        return _transition_row(
            transition_id=transition_id,
            from_interval_index=previous_index,
            to_interval_index=current_index,
            from_speaker=previous_speaker,
            to_speaker=current_speaker,
            source_sample=None if same_speaker else current.start_sample,
            handoff_target=0 if same_speaker else 1,
            relation_target="same" if same_speaker else "different",
            mask_state="valid",
            primary_topology=(
                "same_speaker_silence_gap_resume"
                if same_speaker
                else "silence_gap_different_speaker_handoff"
            ),
            coverage_gate_eligible=True,
            gap_samples=gap_samples,
        )
    if all(len(interval.active_speakers) >= 2 for interval in between):
        if same_speaker:
            competing_speakers = {
                speaker
                for interval in between
                for speaker in interval.active_speakers
                if speaker != previous_speaker
            }
            supported = len(competing_speakers) == 1 and all(
                len(interval.active_speakers) == 2 and previous_speaker in interval.active_speakers
                for interval in between
            )
        else:
            supported = all(
                previous_speaker in interval.active_speakers
                and current_speaker in interval.active_speakers
                and set(interval.active_speakers) <= {previous_speaker, current_speaker}
                for interval in between
            )
        if not supported:
            return _transition_row(
                transition_id=transition_id,
                from_interval_index=previous_index,
                to_interval_index=current_index,
                from_speaker=previous_speaker,
                to_speaker=current_speaker,
                source_sample=None,
                handoff_target=None,
                relation_target=None,
                mask_state="masked",
                primary_topology="complex_overlap_transition",
                secondary_tags=("complex_overlap",),
                overlap_samples=overlap_samples,
            )
        if overlap_samples <= contract.annotation_boundary_jitter_samples:
            return _transition_row(
                transition_id=transition_id,
                from_interval_index=previous_index,
                to_interval_index=current_index,
                from_speaker=previous_speaker,
                to_speaker=current_speaker,
                source_sample=None if same_speaker else current.start_sample,
                handoff_target=0 if same_speaker else 1,
                relation_target="same" if same_speaker else "different",
                mask_state="valid",
                primary_topology=(
                    "same_speaker_direct_continuation"
                    if same_speaker
                    else "clean_direct_different_speaker_handoff"
                ),
                secondary_tags=("boundary_jitter_overlap",),
                coverage_gate_eligible=not same_speaker,
                overlap_samples=overlap_samples,
            )
        if overlap_samples < contract.overlap_topology_min_duration_samples:
            return _transition_row(
                transition_id=transition_id,
                from_interval_index=previous_index,
                to_interval_index=current_index,
                from_speaker=previous_speaker,
                to_speaker=current_speaker,
                source_sample=None if same_speaker else current.start_sample,
                handoff_target=0 if same_speaker else 1,
                relation_target="same" if same_speaker else "different",
                mask_state="valid",
                primary_topology=(
                    "micro_overlap_return" if same_speaker else "micro_overlap_takeover"
                ),
                secondary_tags=("micro_overlap",),
                overlap_samples=overlap_samples,
            )
        return _transition_row(
            transition_id=transition_id,
            from_interval_index=previous_index,
            to_interval_index=current_index,
            from_speaker=previous_speaker,
            to_speaker=current_speaker,
            source_sample=None if same_speaker else current.start_sample,
            handoff_target=0 if same_speaker else 1,
            relation_target="same" if same_speaker else "different",
            mask_state="valid",
            primary_topology="overlap_return" if same_speaker else "overlap_takeover",
            coverage_gate_eligible=True,
            overlap_samples=overlap_samples,
        )
    if (
        gap_samples
        and overlap_samples
        and all(
            not interval.active_speakers or len(interval.active_speakers) >= 2
            for interval in between
        )
    ):
        first_silence = next(
            index for index, interval in enumerate(between) if not interval.active_speakers
        )
        ordered_overlap_then_gap = all(
            len(interval.active_speakers) >= 2 for interval in between[:first_silence]
        ) and all(not interval.active_speakers for interval in between[first_silence:])
        if same_speaker:
            competing_speakers = {
                speaker
                for interval in between[:first_silence]
                for speaker in interval.active_speakers
                if speaker != previous_speaker
            }
            supported = len(competing_speakers) == 1 and all(
                len(interval.active_speakers) == 2 and previous_speaker in interval.active_speakers
                for interval in between[:first_silence]
            )
        else:
            supported = all(
                set(interval.active_speakers) == {previous_speaker, current_speaker}
                for interval in between[:first_silence]
            )
        if ordered_overlap_then_gap and supported:
            if gap_samples > contract.local_continuity_max_gap_samples:
                return _transition_row(
                    transition_id=transition_id,
                    from_interval_index=previous_index,
                    to_interval_index=current_index,
                    from_speaker=previous_speaker,
                    to_speaker=current_speaker,
                    source_sample=None,
                    handoff_target=None,
                    relation_target=None,
                    mask_state="masked",
                    primary_topology="continuity_unknown",
                    secondary_tags=("overlap_then_long_gap",),
                    gap_samples=gap_samples,
                    overlap_samples=overlap_samples,
                )
            return _transition_row(
                transition_id=transition_id,
                from_interval_index=previous_index,
                to_interval_index=current_index,
                from_speaker=previous_speaker,
                to_speaker=current_speaker,
                source_sample=None if same_speaker else current.start_sample,
                handoff_target=0 if same_speaker else 1,
                relation_target="same" if same_speaker else "different",
                mask_state="valid",
                primary_topology=("overlap_gap_return" if same_speaker else "overlap_gap_takeover"),
                secondary_tags=("non_official_mixed_topology",),
                gap_samples=gap_samples,
                overlap_samples=overlap_samples,
            )
    return _transition_row(
        transition_id=transition_id,
        from_interval_index=previous_index,
        to_interval_index=current_index,
        from_speaker=previous_speaker,
        to_speaker=current_speaker,
        source_sample=None,
        handoff_target=None,
        relation_target=None,
        mask_state="masked",
        primary_topology="mixed_unresolved_transition",
        secondary_tags=("mixed_gap_overlap_or_unreliable_solo",),
        gap_samples=gap_samples,
        overlap_samples=overlap_samples,
    )


def _diagnostic_rows(
    intervals: Sequence[CanonicalInterval],
    reliable_indices: Sequence[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    serial = 0
    ambiguous_start: int | None = None
    for index, interval in enumerate(intervals):
        if interval.ambiguous and ambiguous_start is None:
            ambiguous_start = index
        if ambiguous_start is not None and (not interval.ambiguous or index == len(intervals) - 1):
            end_index = index if interval.ambiguous else index - 1
            serial += 1
            rows.append(
                _transition_row(
                    transition_id=f"D{serial:05d}",
                    from_interval_index=ambiguous_start,
                    to_interval_index=end_index,
                    from_speaker=None,
                    to_speaker=None,
                    source_sample=None,
                    handoff_target=None,
                    relation_target=None,
                    mask_state="masked",
                    primary_topology="ambiguous_annotation_region",
                    secondary_tags=("diagnostic",),
                )
            )
            ambiguous_start = None
    for index, interval in enumerate(intervals):
        if (
            not interval.ambiguous
            and interval.speaker_identity_known
            and len(interval.active_speakers) >= 3
        ):
            serial += 1
            rows.append(
                _transition_row(
                    transition_id=f"D{serial:05d}",
                    from_interval_index=index,
                    to_interval_index=index,
                    from_speaker=None,
                    to_speaker=None,
                    source_sample=None,
                    handoff_target=None,
                    relation_target=None,
                    mask_state="masked",
                    primary_topology="complex_overlap_region",
                    secondary_tags=("diagnostic",),
                    overlap_samples=interval.duration_samples,
                )
            )
    for index in range(len(intervals) - 1):
        left = intervals[index]
        right = intervals[index + 1]
        if _is_complex_overlap_boundary(left, right):
            serial += 1
            rows.append(
                _transition_row(
                    transition_id=f"D{serial:05d}",
                    from_interval_index=index,
                    to_interval_index=index + 1,
                    from_speaker=None,
                    to_speaker=None,
                    source_sample=None,
                    handoff_target=None,
                    relation_target=None,
                    mask_state="masked",
                    primary_topology="complex_overlap_transition",
                    secondary_tags=("diagnostic",),
                    overlap_samples=left.duration_samples + right.duration_samples,
                )
            )
    for index, interval in enumerate(intervals):
        if not interval.speaker_identity_known:
            serial += 1
            rows.append(
                _transition_row(
                    transition_id=f"D{serial:05d}",
                    from_interval_index=index,
                    to_interval_index=index,
                    from_speaker=None,
                    to_speaker=None,
                    source_sample=None,
                    handoff_target=None,
                    relation_target=None,
                    mask_state="masked",
                    primary_topology="unknown_speaker_region",
                    secondary_tags=("diagnostic",),
                )
            )
    reliable_set = set(reliable_indices)
    for reliable_index in reliable_indices:
        following: list[CanonicalInterval] = []
        has_later_reliable = False
        for index in range(reliable_index + 1, len(intervals)):
            if index in reliable_set:
                has_later_reliable = True
                break
            following.append(intervals[index])
        if not following:
            continue
        saw_overlap = any(
            len(interval.active_speakers) >= 2 and not interval.ambiguous for interval in following
        )
        first_overlap_index = next(
            (
                index
                for index, interval in enumerate(following)
                if len(interval.active_speakers) >= 2 and not interval.ambiguous
            ),
            None,
        )
        saw_silence_after_overlap = first_overlap_index is not None and any(
            not interval.active_speakers and not interval.ambiguous
            for interval in following[first_overlap_index + 1 :]
        )
        if saw_overlap and saw_silence_after_overlap and not has_later_reliable:
            serial += 1
            rows.append(
                _transition_row(
                    transition_id=f"D{serial:05d}",
                    from_interval_index=reliable_index,
                    to_interval_index=None,
                    from_speaker=intervals[reliable_index].active_speakers[0],
                    to_speaker=None,
                    source_sample=None,
                    handoff_target=None,
                    relation_target=None,
                    mask_state="masked",
                    primary_topology="overlap_to_silence_unresolved",
                    secondary_tags=("diagnostic",),
                    gap_samples=sum(
                        interval.duration_samples
                        for interval in following
                        if not interval.active_speakers
                    ),
                    overlap_samples=sum(
                        interval.duration_samples
                        for interval in following
                        if len(interval.active_speakers) >= 2
                    ),
                )
            )
    return rows


def _topology_episodes(
    transitions: list[dict[str, Any]],
    intervals: Sequence[CanonicalInterval],
    contract: LabelContract,
) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    covered: set[str] = set()
    eligible_short_members = {
        "clean_direct_different_speaker_handoff",
        "silence_gap_different_speaker_handoff",
        "micro_gap_different_speaker_handoff",
        "overlap_takeover",
        "micro_overlap_takeover",
        "overlap_gap_takeover",
    }
    ordered = [
        row
        for row in transitions
        if row["from_interval_index"] is not None and row["to_interval_index"] is not None
    ]
    for first, second in zip(ordered, ordered[1:]):
        if (
            first["transition_id"] in covered
            or second["transition_id"] in covered
            or first["handoff_confirmed"] != 1
            or second["handoff_confirmed"] != 1
            or first["to_interval_index"] != second["from_interval_index"]
            or first["from_speaker"] != second["to_speaker"]
            or first["to_speaker"] != second["from_speaker"]
            or first["primary_topology"] not in eligible_short_members
            or second["primary_topology"] not in eligible_short_members
        ):
            continue
        middle = intervals[first["to_interval_index"]]
        if not (
            contract.short_backchannel_min_duration_samples
            <= middle.duration_samples
            <= contract.short_backchannel_max_duration_samples
        ):
            continue
        episode_id = f"E{len(episodes) + 1:05d}"
        episodes.append(
            {
                "episode_id": episode_id,
                "transition_ids": [
                    first["transition_id"],
                    second["transition_id"],
                ],
                "start_sample": first["handoff_source_sample"],
                "end_sample": second["handoff_source_sample"],
                "primary_topology": "short_backchannel_return",
                "secondary_tags": [
                    f"entry:{first['primary_topology']}",
                    f"return:{second['primary_topology']}",
                ],
                "coverage_gate_eligible": True,
            }
        )
        first["coverage_episode_id"] = episode_id
        second["coverage_episode_id"] = episode_id
        first["coverage_gate_eligible"] = False
        second["coverage_gate_eligible"] = False
        covered.add(first["transition_id"])
        covered.add(second["transition_id"])
    for row in transitions:
        if row["transition_id"] in covered:
            continue
        episode_id = f"E{len(episodes) + 1:05d}"
        row["coverage_episode_id"] = episode_id
        episodes.append(
            {
                "episode_id": episode_id,
                "transition_ids": [row["transition_id"]],
                "start_sample": row["handoff_source_sample"],
                "end_sample": row["handoff_source_sample"],
                "primary_topology": row["primary_topology"],
                "secondary_tags": list(row["secondary_tags"]),
                "coverage_gate_eligible": bool(row["coverage_gate_eligible"]),
            }
        )
    return episodes


def generate_labels(
    values: Iterable[CanonicalInterval | Mapping[str, Any]],
    *,
    contract: LabelContract | None = None,
    scored_start_sample: int | None = None,
    scored_end_sample: int | None = None,
) -> LabelResult:
    installed_contract = load_contract()
    if contract is None:
        active_contract = installed_contract
    else:
        active_contract = _validate_contract(contract)
        if active_contract != installed_contract:
            raise LabelContractError(
                "supplied contract does not match the installed contract version"
            )
    intervals = _reconcile_same_speaker_jitter_gaps(
        normalize_intervals(
            values,
            scored_start_sample=scored_start_sample,
            scored_end_sample=scored_end_sample,
        ),
        active_contract,
    )
    activity_labels: list[dict[str, Any]] = []
    for interval in intervals:
        state = _activity_state(interval)
        activity_labels.append(
            {
                "start_sample": interval.start_sample,
                "end_sample": interval.end_sample,
                "state": state,
                "active_speakers": list(interval.active_speakers),
                "mask_state": (
                    "masked"
                    if interval.ambiguous
                    or (not interval.speaker_identity_known and not interval.active_speakers)
                    else "valid"
                ),
            }
        )
    reliable_indices = [
        index
        for index, interval in enumerate(intervals)
        if _is_reliable_solo(interval, active_contract)
    ]
    transitions: list[dict[str, Any]] = []
    if reliable_indices:
        first_index = reliable_indices[0]
        first = intervals[first_index]
        prefix = intervals[:first_index]
        prefix_requires_mask = any(
            interval.ambiguous
            or not interval.speaker_identity_known
            or len(interval.active_speakers) >= 3
            for interval in prefix
        ) or any(
            _is_complex_overlap_boundary(left, right) for left, right in zip(prefix, prefix[1:])
        )
        transitions.append(
            _transition_row(
                transition_id="T00001",
                from_interval_index=None,
                to_interval_index=first_index,
                from_speaker=None,
                to_speaker=first.active_speakers[0],
                source_sample=None,
                handoff_target=None if prefix_requires_mask else 0,
                relation_target=None,
                mask_state=("masked" if prefix_requires_mask else "handoff_valid_relation_masked"),
                primary_topology="initial_start",
                secondary_tags=(("unresolved_scored_prefix",) if prefix_requires_mask else ()),
            )
        )
        for serial, (previous_index, current_index) in enumerate(
            zip(reliable_indices, reliable_indices[1:]), start=2
        ):
            transitions.append(
                _classify_transition(
                    intervals,
                    previous_index,
                    current_index,
                    active_contract,
                    f"T{serial:05d}",
                )
            )
    transitions.extend(_diagnostic_rows(intervals, reliable_indices))
    transitions.sort(
        key=lambda row: (
            -1 if row["to_interval_index"] is None else int(row["to_interval_index"]),
            row["transition_id"],
        )
    )
    episodes = _topology_episodes(transitions, intervals, active_contract)
    stable_singleton_samples = sum(
        interval.duration_samples
        for interval in intervals
        if _is_reliable_solo(interval, active_contract)
    )
    ongoing_overlap_samples = sum(
        interval.duration_samples
        for interval in intervals
        if not interval.ambiguous and len(interval.active_speakers) >= 2
    )
    ambiguous_samples = sum(
        interval.duration_samples for interval in intervals if interval.ambiguous
    )
    unknown_identity_samples = sum(
        interval.duration_samples for interval in intervals if not interval.speaker_identity_known
    )
    scored_samples = intervals[-1].end_sample - intervals[0].start_sample
    return LabelResult(
        contract_version=active_contract.contract_version,
        contract_document_sha256=active_contract.document_sha256,
        sample_rate_hz=active_contract.sample_rate_hz,
        intervals=intervals,
        activity_labels=tuple(activity_labels),
        transitions=tuple(transitions),
        topology_episodes=tuple(episodes),
        exposure={
            "scored_samples": scored_samples,
            "stable_singleton_samples": stable_singleton_samples,
            "ongoing_overlap_samples": ongoing_overlap_samples,
            "ambiguous_samples": ambiguous_samples,
            "unknown_identity_samples": unknown_identity_samples,
        },
    )
