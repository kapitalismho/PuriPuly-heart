from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from experiments.psem_training_strategy_gate.data.dataset_context import (
    DatasetContext,
    DatasetContextError,
    resolve_dataset_context,
)
from experiments.psem_training_strategy_gate.data.dataset_freeze import (
    AUTHORITY_PIN,
    AUTHORITY_REF,
    CONTRACT_VERSION,
    DATASET_FREEZE_ID,
    EXPECTED_REFERENCE_INTEGRITY_CHECK_IDS,
    NO_MODEL_FIELDS,
    SELECTION_MODEL_EXCLUSION_FIELDS,
    DatasetFreezeError,
    build_v2_dataset_freeze_core,
    validate_checked_dataset_freeze,
)
from experiments.psem_training_strategy_gate.data.evaluator_contract import (
    REQUIRED_VIEW_IDS,
    build_evaluator_contract,
)
from experiments.psem_training_strategy_gate.data.forced_alignment_reference import (
    REFERENCE_COMMIT,
    REFERENCE_REPOSITORY,
)
from experiments.psem_training_strategy_gate.data.identity_components import (
    EXPECTED_V2_SOURCE_IDS,
)
from experiments.psem_training_strategy_gate.data.provenance import (
    canonical_sha256,
    sha256_file,
)
from experiments.psem_training_strategy_gate.data.reference_normalization import (
    EXPECTED_INVENTORY_SHA256,
    ReferenceNormalizationError,
    load_nonlexical_inventory,
)

SAMPLE_RATE_HZ = 16000
GENERATOR_VERSION = "1"
ACCEPTED_FREEZE_MANIFEST_SHA256 = "b600e7050bc3ac92a8393395837452f43711485adfa006547b2d73df1b15fdde"
ACCEPTED_FREEZE_PAYLOAD_SHA256 = "1606f4cc1f497f4bbcd92be3cbf38fe4fb8735edcb454cfd45ab0fddc542261e"
ACCEPTED_CONTRACT_CANONICAL_SHA256 = (
    "74e95d1425498c6743d46fc68b2bedce35c36009646add464b844ce3e5d8464e"
)
ROLE_TRAIN = "PSEM-STRATEGY-TRAIN"
ROLE_DEV = "PSEM-STRATEGY-DEV"
ROLE_EVAL = "PSEM-STRATEGY-EVAL"
PRIMARY_TOPOLOGIES = (
    "clean_direct_different_speaker_handoff",
    "silence_gap_different_speaker_handoff",
    "same_speaker_silence_gap_resume",
    "overlap_return",
    "overlap_takeover",
    "short_backchannel_return",
)
EXPECTED_GATE_SPECS = (
    (f"{ROLE_TRAIN}.scored_samples", 20 * 3600 * SAMPLE_RATE_HZ, "natural_hours.train"),
    (f"{ROLE_DEV}.scored_samples", 5 * 3600 * SAMPLE_RATE_HZ, "natural_hours.dev"),
    (f"{ROLE_EVAL}.scored_samples", 8 * 3600 * SAMPLE_RATE_HZ, "natural_hours.eval"),
    (f"{ROLE_TRAIN}.independent_meetings", 12, "independent_meetings.train"),
    (f"{ROLE_DEV}.independent_meetings", 4, "independent_meetings.dev"),
    (f"{ROLE_EVAL}.independent_meetings", 6, "independent_meetings.eval"),
    (
        "train_dev.primary_topology_counts.clean_direct_different_speaker_handoff",
        100,
        "topology.train_dev.clean_direct_different_speaker_handoff",
    ),
    (
        "eval.primary_topology_counts.clean_direct_different_speaker_handoff",
        20,
        "topology.eval.clean_direct_different_speaker_handoff",
    ),
    (
        "train_dev.primary_topology_counts.silence_gap_different_speaker_handoff",
        200,
        "topology.train_dev.silence_gap_different_speaker_handoff",
    ),
    (
        "eval.primary_topology_counts.silence_gap_different_speaker_handoff",
        40,
        "topology.eval.silence_gap_different_speaker_handoff",
    ),
    (
        "train_dev.primary_topology_counts.same_speaker_silence_gap_resume",
        200,
        "topology.train_dev.same_speaker_silence_gap_resume",
    ),
    (
        "eval.primary_topology_counts.same_speaker_silence_gap_resume",
        40,
        "topology.eval.same_speaker_silence_gap_resume",
    ),
    (
        "train_dev.primary_topology_counts.overlap_return",
        100,
        "topology.train_dev.overlap_return",
    ),
    (
        "eval.primary_topology_counts.overlap_return",
        20,
        "topology.eval.overlap_return",
    ),
    (
        "train_dev.primary_topology_counts.overlap_takeover",
        100,
        "topology.train_dev.overlap_takeover",
    ),
    (
        "eval.primary_topology_counts.overlap_takeover",
        20,
        "topology.eval.overlap_takeover",
    ),
    (
        "train_dev.primary_topology_counts.short_backchannel_return",
        80,
        "topology.train_dev.short_backchannel_return",
    ),
    (
        "eval.primary_topology_counts.short_backchannel_return",
        20,
        "topology.eval.short_backchannel_return",
    ),
    (
        "train_dev.stable_singleton_samples",
        8 * 3600 * SAMPLE_RATE_HZ,
        "negative_exposure.train_dev.stable_singleton",
    ),
    (
        "eval.stable_singleton_samples",
        2 * 3600 * SAMPLE_RATE_HZ,
        "negative_exposure.eval.stable_singleton",
    ),
    (
        "train_dev.ongoing_overlap_samples",
        1 * 3600 * SAMPLE_RATE_HZ,
        "negative_exposure.train_dev.ongoing_overlap",
    ),
    (
        "eval.ongoing_overlap_samples",
        15 * 60 * SAMPLE_RATE_HZ,
        "negative_exposure.eval.ongoing_overlap",
    ),
)
V2_GATE_SPECS = (
    ("minimum", f"{ROLE_TRAIN}.scored_samples", 20 * 3600 * SAMPLE_RATE_HZ, "natural_hours.train"),
    ("minimum", f"{ROLE_TRAIN}.independent_meetings", 12, "independent_meetings.train"),
    (
        "minimum",
        f"{ROLE_TRAIN}.corpus_source_counts.AMI",
        1,
        "corpus_balance.train.source_count.ami",
    ),
    (
        "minimum",
        f"{ROLE_TRAIN}.corpus_source_counts.AliMeeting",
        1,
        "corpus_balance.train.source_count.alimeeting",
    ),
    (
        "minimum",
        f"{ROLE_TRAIN}.corpus_scored_samples.AMI",
        1,
        "corpus_balance.train.scored_samples.ami_present",
    ),
    (
        "minimum",
        f"{ROLE_TRAIN}.corpus_scored_samples.AliMeeting",
        5 * 3600 * SAMPLE_RATE_HZ,
        "corpus_balance.train.scored_hours.alimeeting",
    ),
    (
        "maximum_fraction",
        f"{ROLE_TRAIN}.maximum_corpus_scored_share",
        (4, 5),
        "corpus_balance.train.maximum_corpus_scored_share",
    ),
    ("minimum", f"{ROLE_DEV}.scored_samples", 5 * 3600 * SAMPLE_RATE_HZ, "natural_hours.dev"),
    ("minimum", f"{ROLE_DEV}.independent_meetings", 4, "independent_meetings.dev"),
    ("minimum", f"{ROLE_DEV}.corpus_source_counts.AMI", 2, "corpus_balance.dev.source_count.ami"),
    (
        "minimum",
        f"{ROLE_DEV}.corpus_source_counts.AliMeeting",
        2,
        "corpus_balance.dev.source_count.alimeeting",
    ),
    (
        "minimum",
        f"{ROLE_DEV}.corpus_scored_samples.AMI",
        1 * 3600 * SAMPLE_RATE_HZ,
        "corpus_balance.dev.scored_hours.ami",
    ),
    (
        "minimum",
        f"{ROLE_DEV}.corpus_scored_samples.AliMeeting",
        1 * 3600 * SAMPLE_RATE_HZ,
        "corpus_balance.dev.scored_hours.alimeeting",
    ),
    (
        "maximum_fraction",
        f"{ROLE_DEV}.maximum_corpus_scored_share",
        (3, 4),
        "corpus_balance.dev.maximum_corpus_scored_share",
    ),
    ("minimum", f"{ROLE_EVAL}.scored_samples", 8 * 3600 * SAMPLE_RATE_HZ, "natural_hours.eval"),
    ("minimum", f"{ROLE_EVAL}.independent_meetings", 6, "independent_meetings.eval"),
    ("minimum", f"{ROLE_EVAL}.corpus_source_counts.AMI", 4, "corpus_balance.eval.source_count.ami"),
    (
        "minimum",
        f"{ROLE_EVAL}.corpus_source_counts.AliMeeting",
        4,
        "corpus_balance.eval.source_count.alimeeting",
    ),
    (
        "minimum",
        f"{ROLE_EVAL}.corpus_scored_samples.AMI",
        2 * 3600 * SAMPLE_RATE_HZ,
        "corpus_balance.eval.scored_hours.ami",
    ),
    (
        "minimum",
        f"{ROLE_EVAL}.corpus_scored_samples.AliMeeting",
        2 * 3600 * SAMPLE_RATE_HZ,
        "corpus_balance.eval.scored_hours.alimeeting",
    ),
    (
        "maximum_fraction",
        f"{ROLE_EVAL}.maximum_corpus_scored_share",
        (7, 10),
        "corpus_balance.eval.maximum_corpus_scored_share",
    ),
    *(
        ("minimum", gate_id, required, check_id)
        for gate_id, required, check_id in EXPECTED_GATE_SPECS[6:]
    ),
)
LEAKAGE_FIELDS = (
    ("meeting_session_may_span_roles", "leakage.meeting_session"),
    ("waveform_may_span_roles", "leakage.waveform"),
    ("known_speaker_may_span_roles", "leakage.known_speaker"),
    ("component_may_span_roles", "leakage.connected_component"),
    ("prior_selection_exposed_component_in_eval", "leakage.prior_selection_eval"),
    (
        "exact_known_wavlm_pretraining_overlap_in_eval",
        "leakage.exact_wavlm_pretraining_session_eval",
    ),
)
EXPECTED_MASK_RULES = {
    "unknown_speaker_identity": "retain_activity_and_mask_handoff_and_relation",
    "unknown_active_speech_without_ids": (
        "represent_as_speaker_identity_known_false_with_empty_active_speakers_and_activity_state_unknown_speech"
    ),
    "initial_solo_after_unknown_ambiguous_or_complex_prefix": "mask_handoff_and_relation",
    "ambiguous_annotation_crossing": "mask_handoff_and_relation",
    "complex_overlap_transition": "mask_handoff_and_relation",
    "complex_overlap_region_three_or_more_speakers": (
        "retain_overlap_activity_and_emit_masked_diagnostic_topology"
    ),
    "overlap_to_silence_without_reliable_resolution": "mask_handoff_and_relation",
    "gap_above_local_continuity_maximum": ("continuity_unknown_and_mask_handoff_and_relation"),
    "activity_supervision": "retain_when_interval_activity_is_reliable",
}
V2_EXPECTED_MASK_RULES = {
    **EXPECTED_MASK_RULES,
    "ambiguous_nonlexical_vocalization_crossing": (
        "retain_rttm_activity_and_mask_handoff_and_relation"
    ),
}
EXPECTED_PRECEDENCE = (
    "short_backchannel_return",
    "overlap_takeover",
    "overlap_return",
    "silence_gap_different_speaker_handoff",
    "same_speaker_silence_gap_resume",
    "clean_direct_different_speaker_handoff",
)
EXPECTED_MASKED_TRANSITION_REASONS = frozenset(
    {
        "ambiguous_annotation_crossing",
        "ambiguous_nonlexical_vocalization_crossing",
        "complex_overlap_transition",
        "continuity_unknown",
        "initial_start",
        "mixed_unresolved_transition",
        "unknown_speaker_crossing",
    }
)
EXPECTED_DIAGNOSTIC_MASKED_REGIONS = frozenset(
    {
        "ambiguous_annotation_region",
        "ambiguous_nonlexical_vocalization_region",
        "complex_overlap_region",
        "complex_overlap_transition",
        "overlap_to_silence_unresolved",
        "unknown_speaker_region",
    }
)
REQUIRED_CHECK_IDS = (
    "natural_hours.train",
    "natural_hours.dev",
    "natural_hours.eval",
    "independent_meetings.train",
    "independent_meetings.dev",
    "independent_meetings.eval",
    "topology.train_dev.clean_direct_different_speaker_handoff",
    "topology.eval.clean_direct_different_speaker_handoff",
    "topology.train_dev.silence_gap_different_speaker_handoff",
    "topology.eval.silence_gap_different_speaker_handoff",
    "topology.train_dev.same_speaker_silence_gap_resume",
    "topology.eval.same_speaker_silence_gap_resume",
    "topology.train_dev.overlap_return",
    "topology.eval.overlap_return",
    "topology.train_dev.overlap_takeover",
    "topology.eval.overlap_takeover",
    "topology.train_dev.short_backchannel_return",
    "topology.eval.short_backchannel_return",
    "negative_exposure.train_dev.stable_singleton",
    "negative_exposure.eval.stable_singleton",
    "negative_exposure.train_dev.ongoing_overlap",
    "negative_exposure.eval.ongoing_overlap",
    "leakage.meeting_session",
    "leakage.waveform",
    "leakage.known_speaker",
    "leakage.connected_component",
    "leakage.prior_selection_eval",
    "leakage.exact_wavlm_pretraining_session_eval",
    "annotations.cover_every_scored_range",
    "annotations.unresolved_and_ambiguous_regions_masked",
    "topology.primary_gate_counts_exclusive_and_reproducible",
    "hashes.frozen_artifacts_and_repository_inputs_resolve",
    "hashes.source_annotation_split_identities_resolve",
    "contract.operational_version_frozen",
    "freeze.dataset_freeze_id_present_and_consistent",
    "freeze.current_and_internally_consistent",
    "data.natural_only",
    "split.model_derived_quantities_forbidden",
    "model_boundary.model_predictions_consulted",
    "model_boundary.model_scores_consulted",
    "model_boundary.official_model_results_inspected",
    "model_boundary.official_model_training_performed",
)
V2_REQUIRED_CHECK_IDS = (
    *(spec[3] for spec in V2_GATE_SPECS),
    *REQUIRED_CHECK_IDS[22:31],
    "reference.integrity_and_provenance_current",
    "evaluator.shared_threshold_contract_current",
    *REQUIRED_CHECK_IDS[31:],
)


class DatasetPreflightError(RuntimeError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetPreflightError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise DatasetPreflightError(f"JSON artifact must be an object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetPreflightError(f"invalid JSONL artifact: {path}") from exc
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise DatasetPreflightError(f"JSONL artifact must contain objects: {path}")
    return rows


def _minimum_check(
    check_id: str,
    observed: int,
    required: int,
    unit: str,
    **details: Any,
) -> dict[str, Any]:
    passed = observed >= required
    return {
        "id": check_id,
        "observed": observed,
        "required": required,
        "deficit": max(0, required - observed),
        "unit": unit,
        "passed": passed,
        **details,
    }


def _maximum_check(
    check_id: str,
    observed: int,
    required_maximum: int,
    unit: str,
    **details: Any,
) -> dict[str, Any]:
    passed = observed <= required_maximum
    return {
        "id": check_id,
        "observed": observed,
        "required": {"maximum": required_maximum},
        "deficit": max(0, observed - required_maximum),
        "unit": unit,
        "passed": passed,
        **details,
    }


def _exact_check(
    check_id: str,
    observed: Any,
    required: Any,
    **details: Any,
) -> dict[str, Any]:
    passed = observed == required
    return {
        "id": check_id,
        "observed": observed,
        "required": required,
        "deficit": None,
        "passed": passed,
        **details,
    }


def _sha256_string(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _positive_int(value: Any) -> bool:
    return _nonnegative_int(value) and value > 0


def _nonnegative_int_map(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and all(isinstance(key, str) and key for key in value)
        and all(
            isinstance(count, int) and not isinstance(count, bool) and count >= 0
            for count in value.values()
        )
    )


def _require_check_inventory(
    checks: list[dict[str, Any]],
    required_ids: tuple[str, ...] = REQUIRED_CHECK_IDS,
) -> None:
    observed_ids = tuple(check.get("id") for check in checks)
    if len(required_ids) != len(set(required_ids)) or observed_ids != required_ids:
        label = (
            "fixed 42-check contract"
            if required_ids == REQUIRED_CHECK_IDS
            else "fixed v2 check contract"
        )
        raise DatasetPreflightError(f"preflight check inventory is not the {label}")


def _gate_checks(split: dict[str, Any]) -> list[dict[str, Any]]:
    hard_gates = split.get("hard_gate_results")
    if not isinstance(hard_gates, list):
        raise DatasetPreflightError("split hard gates are missing")
    by_id = {
        gate.get("id"): gate
        for gate in hard_gates
        if isinstance(gate, dict) and isinstance(gate.get("id"), str)
    }
    expected_ids = {gate_id for gate_id, _, _ in EXPECTED_GATE_SPECS}
    if len(by_id) != len(hard_gates) or set(by_id) != expected_ids:
        raise DatasetPreflightError("split hard gate inventory is not exact")
    checks = []
    for gate_id, required, check_id in EXPECTED_GATE_SPECS:
        gate = by_id[gate_id]
        observed = gate.get("observed")
        if isinstance(observed, bool) or not isinstance(observed, int):
            raise DatasetPreflightError(f"split hard gate observation is invalid: {gate_id}")
        if gate.get("required") != required or gate.get("passed") is not (observed >= required):
            raise DatasetPreflightError(f"split hard gate is internally inconsistent: {gate_id}")
        unit = (
            "source_samples_at_16000_hz"
            if gate_id.endswith("scored_samples")
            or gate_id.endswith("stable_singleton_samples")
            or gate_id.endswith("ongoing_overlap_samples")
            else "count"
        )
        details: dict[str, Any] = {"split_gate_id": gate_id}
        if unit.startswith("source_samples"):
            details["observed_hours"] = round(observed / SAMPLE_RATE_HZ / 3600, 6)
            details["required_hours"] = round(required / SAMPLE_RATE_HZ / 3600, 6)
        checks.append(_minimum_check(check_id, observed, required, unit, **details))
    return checks


def _v2_gate_checks(split: dict[str, Any]) -> list[dict[str, Any]]:
    hard_gates = split.get("hard_gate_results")
    if not isinstance(hard_gates, list):
        raise DatasetPreflightError("v2 split hard gates are missing")
    by_id = {
        gate.get("id"): gate
        for gate in hard_gates
        if isinstance(gate, dict) and isinstance(gate.get("id"), str)
    }
    expected_ids = {spec[1] for spec in V2_GATE_SPECS}
    if len(by_id) != len(hard_gates) or set(by_id) != expected_ids:
        raise DatasetPreflightError("v2 split hard gate inventory is not exact")
    checks = []
    for kind, gate_id, required, check_id in V2_GATE_SPECS:
        gate = by_id[gate_id]
        if kind == "minimum":
            observed = gate.get("observed")
            observed_required = gate.get("required")
            if isinstance(observed, bool) or not isinstance(observed, int):
                raise DatasetPreflightError(f"v2 split hard gate observation is invalid: {gate_id}")
            if (
                isinstance(observed_required, bool)
                or not isinstance(observed_required, int)
                or observed_required != required
                or gate.get("passed") is not (observed >= required)
            ):
                raise DatasetPreflightError(
                    f"v2 split hard gate is internally inconsistent: {gate_id}"
                )
            unit = (
                "source_samples_at_16000_hz"
                if "scored_samples" in gate_id
                or gate_id.endswith("stable_singleton_samples")
                or gate_id.endswith("ongoing_overlap_samples")
                else "count"
            )
            details: dict[str, Any] = {"split_gate_id": gate_id}
            if unit.startswith("source_samples"):
                details["observed_hours"] = round(observed / SAMPLE_RATE_HZ / 3600, 6)
                details["required_hours"] = round(required / SAMPLE_RATE_HZ / 3600, 6)
            checks.append(_minimum_check(check_id, observed, required, unit, **details))
            continue
        required_numerator, required_denominator = required
        observed = gate.get("observed")
        required_maximum = gate.get("required_maximum")
        valid = (
            isinstance(observed, dict)
            and isinstance(required_maximum, dict)
            and set(observed) == {"numerator", "denominator", "decimal"}
            and set(required_maximum) == {"numerator", "denominator"}
            and all(
                isinstance(observed.get(field), int) and not isinstance(observed.get(field), bool)
                for field in ("numerator", "denominator")
            )
            and isinstance(observed.get("decimal"), (int, float))
            and not isinstance(observed.get("decimal"), bool)
            and math.isfinite(observed["decimal"])
            and observed["numerator"] >= 0
            and observed["denominator"] > 0
            and observed["numerator"] <= observed["denominator"]
            and observed["decimal"] == round(observed["numerator"] / observed["denominator"], 8)
            and all(
                isinstance(required_maximum.get(field), int)
                and not isinstance(required_maximum.get(field), bool)
                for field in ("numerator", "denominator")
            )
            and required_maximum["numerator"] >= 0
            and required_maximum["denominator"] > 0
            and required_maximum["numerator"] <= required_maximum["denominator"]
            and required_maximum
            == {
                "numerator": required_numerator,
                "denominator": required_denominator,
            }
        )
        if not valid:
            raise DatasetPreflightError(f"v2 split corpus-share gate is invalid: {gate_id}")
        passed = (
            observed["numerator"] * required_denominator
            <= observed["denominator"] * required_numerator
        )
        if gate.get("passed") is not passed:
            raise DatasetPreflightError(
                f"v2 split corpus-share gate is internally inconsistent: {gate_id}"
            )
        checks.append(
            {
                "id": check_id,
                "observed": observed,
                "required": {"maximum": required_maximum},
                "deficit": None,
                "unit": "fraction",
                "passed": passed,
                "split_gate_id": gate_id,
            }
        )
    return checks


def _leakage_checks(split: dict[str, Any]) -> list[dict[str, Any]]:
    leakage = split.get("leakage_audit")
    if not isinstance(leakage, dict):
        raise DatasetPreflightError("split leakage audit is missing")
    checks = []
    for field, check_id in LEAKAGE_FIELDS:
        observed = leakage.get(field)
        if not isinstance(observed, bool):
            raise DatasetPreflightError(f"split leakage field is invalid: {field}")
        checks.append(
            _maximum_check(
                check_id,
                int(observed),
                0,
                "violations",
                split_leakage_field=field,
            )
        )
    return checks


def _annotation_coverage_check(
    source_rows: list[dict[str, Any]],
    annotation_rows: list[dict[str, Any]],
    normalization_rows: list[dict[str, Any]],
    selected_source_ids: set[str],
) -> dict[str, Any]:
    sources = {row.get("source_id"): row for row in source_rows}
    annotations = {row.get("source_id"): row for row in annotation_rows}
    normalizations = {row.get("source_id"): row for row in normalization_rows}
    violations = []
    if (
        len(sources) != len(source_rows)
        or len(annotations) != len(annotation_rows)
        or len(normalizations) != len(normalization_rows)
    ):
        violations.append("duplicate_source_id")
    if set(sources) != selected_source_ids:
        violations.append("source_manifest_coverage")
    if set(annotations) != selected_source_ids:
        violations.append("annotation_manifest_coverage")
    if set(normalizations) != selected_source_ids:
        violations.append("normalization_manifest_coverage")
    covered = 0
    for source_id in sorted(
        selected_source_ids & set(sources) & set(annotations) & set(normalizations)
    ):
        source = sources[source_id]
        annotation = annotations[source_id]
        normalization = normalizations[source_id]
        source_start = source.get("annotation_coverage_start_sample")
        source_end = source.get("annotation_coverage_end_sample")
        annotation_start = annotation.get("coverage_start_sample")
        annotation_end = annotation.get("coverage_end_sample")
        scored_start = normalization.get("scored_start_sample")
        scored_end = normalization.get("scored_end_sample")
        duration = source.get("duration_samples")
        source_coverage_valid = (
            source_start is None
            and source_end is None
            or isinstance(source_start, int)
            and not isinstance(source_start, bool)
            and isinstance(source_end, int)
            and not isinstance(source_end, bool)
            and source_start == annotation_start
            and source_end == annotation_end
        )
        valid = (
            all(
                row.get("schema_version") == 1
                and isinstance(row.get("schema_version"), int)
                and not isinstance(row.get("schema_version"), bool)
                for row in (source, annotation, normalization)
            )
            and isinstance(source.get("sample_rate_hz"), int)
            and not isinstance(source.get("sample_rate_hz"), bool)
            and all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in (
                    annotation_start,
                    annotation_end,
                    scored_start,
                    scored_end,
                    duration,
                )
            )
            and source_coverage_valid
            and 0 <= annotation_start <= scored_start < scored_end <= annotation_end
            and annotation_end <= duration
            and source["sample_rate_hz"] == SAMPLE_RATE_HZ
            and source.get("contract_document_sha256") == ACCEPTED_CONTRACT_CANONICAL_SHA256
            and annotation.get("contract_document_sha256") == ACCEPTED_CONTRACT_CANONICAL_SHA256
            and normalization.get("contract_document_sha256") == ACCEPTED_CONTRACT_CANONICAL_SHA256
            and source.get("contract_version") == CONTRACT_VERSION
            and annotation.get("contract_version") == CONTRACT_VERSION
            and normalization.get("contract_version") == CONTRACT_VERSION
            and normalization.get("exposure", {}).get("scored_samples") == scored_end - scored_start
        )
        if valid:
            covered += 1
        else:
            violations.append(source_id)
    return _minimum_check(
        "annotations.cover_every_scored_range",
        covered,
        len(selected_source_ids),
        "sources",
        violations=violations,
    )


def _v2_annotation_coverage_check(
    context: DatasetContext,
    source_rows: list[dict[str, Any]],
    annotation_rows: list[dict[str, Any]],
    normalization_rows: list[dict[str, Any]],
    selected_source_ids: set[str],
) -> dict[str, Any]:
    sources = {row.get("source_id"): row for row in source_rows}
    annotations = {row.get("source_id"): row for row in annotation_rows}
    normalizations = {row.get("source_id"): row for row in normalization_rows}
    violations = []
    if any(
        len(rows) != len(by_id)
        for rows, by_id in (
            (source_rows, sources),
            (annotation_rows, annotations),
            (normalization_rows, normalizations),
        )
    ):
        violations.append("duplicate_source_id")
    if selected_source_ids != EXPECTED_V2_SOURCE_IDS:
        violations.append("selected_source_scope")
    if set(sources) != selected_source_ids:
        violations.append("source_manifest_coverage")
    if set(annotations) != selected_source_ids:
        violations.append("annotation_manifest_coverage")
    if set(normalizations) != selected_source_ids:
        violations.append("normalization_manifest_coverage")
    covered = 0
    for source_id in sorted(
        selected_source_ids & set(sources) & set(annotations) & set(normalizations)
    ):
        source = sources[source_id]
        annotation = annotations[source_id]
        normalization = normalizations[source_id]
        source_start = source.get("annotation_coverage_start_sample")
        source_end = source.get("annotation_coverage_end_sample")
        annotation_start = annotation.get("coverage_start_sample")
        annotation_end = annotation.get("coverage_end_sample")
        scored_start = normalization.get("scored_start_sample")
        scored_end = normalization.get("scored_end_sample")
        duration = source.get("duration_samples")
        source_coverage_valid = (
            source_start is None
            and source_end is None
            or isinstance(source_start, int)
            and not isinstance(source_start, bool)
            and isinstance(source_end, int)
            and not isinstance(source_end, bool)
            and source_start == annotation_start
            and source_end == annotation_end
        )
        valid = (
            all(
                row.get("schema_version") == 1
                and isinstance(row.get("schema_version"), int)
                and not isinstance(row.get("schema_version"), bool)
                for row in (source, annotation, normalization)
            )
            and isinstance(source.get("sample_rate_hz"), int)
            and not isinstance(source.get("sample_rate_hz"), bool)
            and all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in (
                    annotation_start,
                    annotation_end,
                    scored_start,
                    scored_end,
                    duration,
                )
            )
            and source_coverage_valid
            and 0 <= annotation_start <= scored_start < scored_end <= annotation_end
            and annotation_end <= duration
            and source["sample_rate_hz"] == SAMPLE_RATE_HZ
            and source.get("contract_document_sha256") == context.source_contract.document_sha256
            and annotation.get("contract_document_sha256")
            == context.source_contract.document_sha256
            and normalization.get("contract_document_sha256")
            == context.label_contract.document_sha256
            and source.get("contract_version") == context.source_contract.contract_version
            and annotation.get("contract_version") == context.source_contract.contract_version
            and normalization.get("contract_version") == context.label_contract.contract_version
            and normalization.get("reference_repository") == REFERENCE_REPOSITORY
            and normalization.get("reference_commit") == REFERENCE_COMMIT
            and normalization.get("exposure", {}).get("scored_samples") == scored_end - scored_start
        )
        if valid:
            covered += 1
        else:
            violations.append(source_id)
    return _minimum_check(
        "annotations.cover_every_scored_range",
        covered,
        len(selected_source_ids),
        "sources",
        violations=violations,
    )


def _masking_check(
    contract: dict[str, Any],
    census: dict[str, Any],
    topology_rows: list[dict[str, Any]],
    expected_mask_rules: dict[str, str] = EXPECTED_MASK_RULES,
) -> dict[str, Any]:
    assertions: list[tuple[str, bool]] = [
        (
            f"contract_rule.{name}",
            contract.get("ambiguity_and_mask_rules", {}).get(name) == expected,
        )
        for name, expected in expected_mask_rules.items()
    ]
    row_masked = 0
    row_actual = 0
    row_reasons: Counter[str] = Counter()
    row_diagnostic_regions: Counter[str] = Counter()
    for row in topology_rows:
        source_id = row.get("source_id")
        diagnostics = row.get("mask_diagnostics")
        valid_diagnostics = isinstance(diagnostics, dict)
        masked_count = diagnostics.get("masked_transition_count") if valid_diagnostics else None
        actual_count = diagnostics.get("actual_transition_count") if valid_diagnostics else None
        reasons = diagnostics.get("masked_transition_reasons") if valid_diagnostics else None
        diagnostic_regions = (
            diagnostics.get("diagnostic_masked_region_counts") if valid_diagnostics else None
        )
        valid_counts = all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in (masked_count, actual_count)
        )
        valid_maps = (
            _nonnegative_int_map(reasons)
            and set(reasons) <= EXPECTED_MASKED_TRANSITION_REASONS
            and _nonnegative_int_map(diagnostic_regions)
            and set(diagnostic_regions) <= EXPECTED_DIAGNOSTIC_MASKED_REGIONS
        )
        assertions.append((f"row.{source_id}.mask_diagnostics", valid_counts and valid_maps))
        if valid_counts and valid_maps:
            row_masked += masked_count
            row_actual += actual_count
            row_reasons.update(reasons)
            row_diagnostic_regions.update(diagnostic_regions)
            assertions.append(
                (
                    f"row.{source_id}.masked_reason_total",
                    sum(reasons.values()) == masked_count,
                )
            )
    overall_mask = census.get("overall", {}).get("mask_diagnostics", {})
    overall_reasons = overall_mask.get("masked_transition_reasons")
    overall_diagnostic_regions = overall_mask.get("diagnostic_masked_region_counts")
    overall_maps_valid = (
        _nonnegative_int_map(overall_reasons)
        and set(overall_reasons) <= EXPECTED_MASKED_TRANSITION_REASONS
        and _nonnegative_int_map(overall_diagnostic_regions)
        and set(overall_diagnostic_regions) <= EXPECTED_DIAGNOSTIC_MASKED_REGIONS
    )
    overall_masked = overall_mask.get("masked_transition_count")
    overall_actual = overall_mask.get("actual_transition_count")
    overall_counts_valid = all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in (overall_masked, overall_actual)
    )
    assertions.extend(
        (
            (
                "census.masked_transition_count",
                overall_counts_valid and row_masked == overall_masked,
            ),
            (
                "census.actual_transition_count",
                overall_counts_valid and row_actual == overall_actual,
            ),
            ("census.mask_maps", overall_maps_valid),
            (
                "census.masked_transition_reasons",
                overall_maps_valid and dict(sorted(row_reasons.items())) == overall_reasons,
            ),
            (
                "census.diagnostic_masked_region_counts",
                overall_maps_valid
                and dict(sorted(row_diagnostic_regions.items())) == overall_diagnostic_regions,
            ),
            (
                "census.masked_reason_total",
                overall_maps_valid
                and overall_counts_valid
                and sum(overall_reasons.values()) == overall_masked,
            ),
        )
    )
    violations = [name for name, passed in assertions if not passed]
    return _minimum_check(
        "annotations.unresolved_and_ambiguous_regions_masked",
        len(assertions) - len(violations),
        len(assertions),
        "mask_invariants",
        violations=violations,
        masked_transition_count=overall_mask.get("masked_transition_count"),
        diagnostic_masked_region_counts=overall_mask.get("diagnostic_masked_region_counts"),
    )


def _exclusive_counting_check(
    contract: dict[str, Any],
    census: dict[str, Any],
    topology_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    assertions: list[tuple[str, bool]] = [
        (
            "contract.precedence",
            tuple(contract.get("official_primary_topology_precedence", ())) == EXPECTED_PRECEDENCE,
        ),
        (
            "contract.one_episode_one_gate",
            isinstance(
                contract.get("exclusive_counting", {}).get(
                    "one_episode_may_increment_official_primary_gate_count"
                ),
                int,
            )
            and not isinstance(
                contract.get("exclusive_counting", {}).get(
                    "one_episode_may_increment_official_primary_gate_count"
                ),
                bool,
            )
            and contract.get("exclusive_counting", {}).get(
                "one_episode_may_increment_official_primary_gate_count"
            )
            == 1,
        ),
        (
            "census.exclusive_primary_counting",
            census.get("counting_policy", {}).get("exclusive_primary_counting") is True,
        ),
        (
            "census.short_backchannel_not_double_counted",
            census.get("counting_policy", {}).get(
                "short_backchannel_member_handoffs_counted_separately"
            )
            is False,
        ),
        (
            "census.old_r7_or_r7b_event_counts_forbidden",
            census.get("counting_policy", {}).get("old_r7_or_r7b_event_counts_used") is False,
        ),
        (
            "census.precedence",
            tuple(census.get("counting_policy", {}).get("official_primary_topology_precedence", ()))
            == EXPECTED_PRECEDENCE,
        ),
    ]
    aggregate_counts: Counter[str] = Counter()
    aggregate_episode_count = 0
    for row in topology_rows:
        source_id = row.get("source_id")
        counts = row.get("primary_topology_counts")
        valid_counts = (
            isinstance(counts, dict)
            and set(counts) == set(PRIMARY_TOPOLOGIES)
            and all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in counts.values()
            )
        )
        assertions.append((f"row.{source_id}.topology_inventory", valid_counts))
        if valid_counts:
            row_total = sum(counts.values())
            aggregate_counts.update(counts)
            aggregate_episode_count += row_total
            assertions.append(
                (
                    f"row.{source_id}.exclusive_episode_count",
                    isinstance(row.get("exclusive_primary_episode_count"), int)
                    and not isinstance(row.get("exclusive_primary_episode_count"), bool)
                    and row_total == row["exclusive_primary_episode_count"],
                )
            )
    overall = census.get("overall", {})
    overall_primary_counts = overall.get("primary_topology_counts")
    overall_episode_count = overall.get("exclusive_primary_episode_count")
    overall_counts_valid = (
        _nonnegative_int_map(overall_primary_counts)
        and set(overall_primary_counts) == set(PRIMARY_TOPOLOGIES)
        and isinstance(overall_episode_count, int)
        and not isinstance(overall_episode_count, bool)
        and overall_episode_count >= 0
    )
    assertions.extend(
        (
            (
                "census.aggregate_primary_counts",
                overall_counts_valid
                and dict(sorted(aggregate_counts.items())) == overall_primary_counts,
            ),
            (
                "census.aggregate_exclusive_episode_count",
                overall_counts_valid and aggregate_episode_count == overall_episode_count,
            ),
        )
    )
    violations = [name for name, passed in assertions if not passed]
    return _minimum_check(
        "topology.primary_gate_counts_exclusive_and_reproducible",
        len(assertions) - len(violations),
        len(assertions),
        "counting_invariants",
        violations=violations,
        exclusive_primary_episode_count=overall.get("exclusive_primary_episode_count"),
    )


def _hash_checks(
    data_dir: Path,
    freeze: dict[str, Any],
    source_rows: list[dict[str, Any]],
    annotation_rows: list[dict[str, Any]],
    normalization_rows: list[dict[str, Any]],
    topology_rows: list[dict[str, Any]],
    split_sources: list[dict[str, Any]],
    context: DatasetContext | None = None,
) -> list[dict[str, Any]]:
    repository_root = (
        context.data_dir.parents[3]
        if context is not None and context.is_v2
        else data_dir.parents[2]
    )
    artifact_hashes = freeze.get("artifact_sha256", {})
    inherited_hashes = freeze.get("inherited_artifact_sha256", {})
    repository_hashes = freeze.get("repository_input_sha256", {})
    if not all(
        isinstance(value, dict) for value in (artifact_hashes, inherited_hashes, repository_hashes)
    ):
        raise DatasetPreflightError("frozen file hash inventories are invalid")
    violations = []
    matched = 0
    for name, expected in artifact_hashes.items():
        path = data_dir / name
        if path.is_file() and sha256_file(path) == expected:
            matched += 1
        else:
            violations.append(name)
    for name, expected in inherited_hashes.items():
        path = repository_root / name
        if path.is_file() and sha256_file(path) == expected:
            matched += 1
        else:
            violations.append(name)
    for name, expected in repository_hashes.items():
        path = repository_root / name
        if path.is_file() and sha256_file(path) == expected:
            matched += 1
        else:
            violations.append(name)
    artifact_check = _minimum_check(
        "hashes.frozen_artifacts_and_repository_inputs_resolve",
        matched,
        len(artifact_hashes) + len(inherited_hashes) + len(repository_hashes),
        "files",
        violations=violations,
    )
    sources = {row.get("source_id"): row for row in source_rows}
    annotations = {row.get("source_id"): row for row in annotation_rows}
    normalizations = {row.get("source_id"): row for row in normalization_rows}
    topology = {row.get("source_id"): row for row in topology_rows}
    split = {row.get("source_id"): row for row in split_sources}
    identity_violations = []
    matched_identities = 0
    expected_source_count = freeze.get("source_identity_binding", {}).get("source_count", 0)
    identity_sets = (set(sources), set(annotations), set(normalizations), set(topology), set(split))
    duplicate_free = all(
        len(rows) == len(by_id)
        for rows, by_id in (
            (source_rows, sources),
            (annotation_rows, annotations),
            (normalization_rows, normalizations),
            (topology_rows, topology),
            (split_sources, split),
        )
    )
    exact_coverage = (
        duplicate_free
        and all(source_ids == identity_sets[0] for source_ids in identity_sets[1:])
        and len(identity_sets[0]) == expected_source_count
        and (context is None or not context.is_v2 or identity_sets[0] == EXPECTED_V2_SOURCE_IDS)
    )
    for source_id in sorted(set().union(*identity_sets)):
        source = sources.get(source_id, {})
        annotation = annotations.get(source_id, {})
        normalization = normalizations.get(source_id, {})
        topology_row = topology.get(source_id, {})
        assignment = split.get(source_id, {})
        waveform_sha256 = source.get("waveform_sha256")
        annotation_sha256 = source.get("annotation_sha256")
        label_result_sha256 = normalization.get("label_result_sha256")
        normalization_annotation_sha256 = normalization.get(
            "source_annotation_sha256"
            if context is not None and context.is_v2
            else "annotation_sha256"
        )
        valid = (
            exact_coverage
            and _sha256_string(waveform_sha256)
            and _sha256_string(annotation_sha256)
            and _sha256_string(label_result_sha256)
            and _sha256_string(topology_row.get("topology_episodes_sha256"))
            and waveform_sha256 == assignment.get("waveform_sha256")
            and waveform_sha256 == normalization.get("source_waveform_sha256")
            and waveform_sha256 == topology_row.get("source_waveform_sha256")
            and annotation_sha256 == annotation.get("annotation_sha256")
            and annotation_sha256 == assignment.get("annotation_sha256")
            and annotation_sha256 == normalization_annotation_sha256
            and annotation_sha256 == topology_row.get("annotation_sha256")
            and label_result_sha256 == topology_row.get("label_result_sha256")
            and topology_row.get("normalization_row_sha256") == canonical_sha256(normalization)
            and topology_row.get("scored_start_sample") == normalization.get("scored_start_sample")
            and topology_row.get("scored_end_sample") == normalization.get("scored_end_sample")
            and topology_row.get("scored_samples")
            == normalization.get("exposure", {}).get("scored_samples")
            and topology_row.get("contract_document_sha256")
            == (
                context.label_contract.document_sha256
                if context is not None and context.is_v2
                else ACCEPTED_CONTRACT_CANONICAL_SHA256
            )
            and topology_row.get("contract_version")
            == (
                context.label_contract.contract_version
                if context is not None and context.is_v2
                else CONTRACT_VERSION
            )
        )
        if context is not None and context.is_v2:
            reference_sha256 = normalization.get("reference_sha256")
            scored_start = normalization.get("scored_start_sample")
            scored_end = normalization.get("scored_end_sample")
            scored_samples = normalization.get("exposure", {}).get("scored_samples")
            topology_counts = topology_row.get("primary_topology_counts")
            valid = (
                valid
                and all(
                    row.get("schema_version") == 1
                    and isinstance(row.get("schema_version"), int)
                    and not isinstance(row.get("schema_version"), bool)
                    for row in (source, annotation, normalization, topology_row)
                )
                and _positive_int(source.get("sample_rate_hz"))
                and source["sample_rate_hz"] == SAMPLE_RATE_HZ
                and _positive_int(source.get("duration_samples"))
                and _nonnegative_int(scored_start)
                and _positive_int(scored_end)
                and scored_start < scored_end <= source["duration_samples"]
                and _positive_int(scored_samples)
                and scored_samples == scored_end - scored_start
                and _nonnegative_int(topology_row.get("scored_start_sample"))
                and _positive_int(topology_row.get("scored_end_sample"))
                and _positive_int(topology_row.get("scored_samples"))
                and _nonnegative_int(topology_row.get("exclusive_primary_episode_count"))
                and _nonnegative_int_map(topology_counts)
                and set(topology_counts) == set(PRIMARY_TOPOLOGIES)
                and sum(topology_counts.values()) == topology_row["exclusive_primary_episode_count"]
                and source.get("contract_document_sha256")
                == context.source_contract.document_sha256
                and annotation.get("contract_document_sha256")
                == context.source_contract.document_sha256
                and normalization.get("contract_document_sha256")
                == context.label_contract.document_sha256
                and source.get("contract_version") == context.source_contract.contract_version
                and annotation.get("contract_version") == context.source_contract.contract_version
                and normalization.get("contract_version") == context.label_contract.contract_version
                and normalization.get("source_record_sha256") == canonical_sha256(source)
                and _sha256_string(reference_sha256)
                and source.get("reference_sha256") in (None, reference_sha256)
                and reference_sha256 == assignment.get("reference_sha256")
                and _sha256_string(normalization.get("canonical_intervals_sha256"))
            )
        if valid:
            matched_identities += 1
        else:
            identity_violations.append(source_id)
    identity_check = _minimum_check(
        "hashes.source_annotation_split_identities_resolve",
        matched_identities,
        expected_source_count,
        "sources",
        violations=identity_violations,
        exact_source_coverage=exact_coverage,
    )
    return [artifact_check, identity_check]


def _contract_check(contract: dict[str, Any]) -> dict[str, Any]:
    source_coordinates = contract.get("source_coordinate_convention", {})
    constants = contract.get("constants_ms", {})
    primary_event = contract.get("primary_event", {})
    observed = {
        "canonical_sha256": canonical_sha256(contract),
        "contract_version": contract.get("contract_version"),
        "status": contract.get("status"),
        "sample_rate_hz": source_coordinates.get("sample_rate_hz"),
        "coordinates": source_coordinates.get("coordinates"),
        "grid_mapping": source_coordinates.get("grid_mapping"),
        "local_continuity_max_gap_ms": constants.get("local_continuity_max_gap"),
        "primary_event": primary_event,
    }
    required = {
        "canonical_sha256": ACCEPTED_CONTRACT_CANONICAL_SHA256,
        "contract_version": CONTRACT_VERSION,
        "status": "frozen_after_annotation_only_calibration",
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "coordinates": "zero_based_half_open_unsnapped_source_samples",
        "grid_mapping": "forbidden_in_dataset_labels",
        "local_continuity_max_gap_ms": 1200,
        "primary_event": {
            "id": "handoff_confirmed",
            "boundary": "first_source_sample_of_the_new_reliable_solo_interval",
            "reliability_confirmation_shifts_boundary": False,
            "initial_solo_is_handoff": False,
            "overlap_onset_is_handoff": False,
        },
    }
    return _exact_check("contract.operational_version_frozen", observed, required)


def _v2_contract_check(
    context: DatasetContext,
    contract: dict[str, Any],
) -> dict[str, Any]:
    source_coordinates = contract.get("source_coordinate_convention", {})
    temporal_authority = contract.get("temporal_activity_authority", {})
    nonlexical_mask = contract.get("nonlexical_mask", {})
    observed = {
        "canonical_sha256": canonical_sha256(contract),
        "contract_version": contract.get("contract_version"),
        "status": contract.get("status"),
        "sample_rate_hz": source_coordinates.get("sample_rate_hz"),
        "coordinates": source_coordinates.get("coordinates"),
        "grid_mapping": source_coordinates.get("grid_mapping"),
        "reference_repository": temporal_authority.get("repository"),
        "reference_commit": temporal_authority.get("commit"),
        "model_repair": temporal_authority.get("neural_or_model_prediction_repair"),
        "nonlexical_class": nonlexical_mask.get("class"),
        "nonlexical_scope": nonlexical_mask.get("scope"),
        "primary_event": contract.get("primary_event"),
    }
    required = {
        "canonical_sha256": context.label_contract.document_sha256,
        "contract_version": context.label_contract.contract_version,
        "status": context.label_contract.status,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "coordinates": "zero_based_half_open_unsnapped_source_samples",
        "grid_mapping": "forbidden_in_dataset_labels",
        "reference_repository": REFERENCE_REPOSITORY,
        "reference_commit": REFERENCE_COMMIT,
        "model_repair": "forbidden",
        "nonlexical_class": "ambiguous_nonlexical_vocalization",
        "nonlexical_scope": "handoff_and_relation_supervision_only",
        "primary_event": {
            "id": "handoff_confirmed",
            "boundary": "first_source_sample_of_the_new_reliable_solo_interval",
            "reliability_confirmation_shifts_boundary": False,
            "initial_solo_is_handoff": False,
            "overlap_onset_is_handoff": False,
        },
    }
    return _exact_check("contract.operational_version_frozen", observed, required)


def _v2_reference_integrity_check(
    data_dir: Path,
    freeze: dict[str, Any],
) -> dict[str, Any]:
    receipt = _load_json(data_dir / "reference_artifact_receipt.json")
    integrity = _load_json(data_dir / "reference_integrity_report.json")
    provenance = _load_json(data_dir / "reference_provenance.json")
    artifact_hashes = freeze.get("artifact_sha256", {})
    receipt_hashes = receipt.get("artifact_sha256", {})
    expected_receipt_names = {
        "REFERENCE_MIGRATION.md",
        "reference_integrity_report.json",
        "reference_migration.jsonl",
        "reference_migration_summary.json",
        "reference_provenance.json",
    }
    receipt_current = (
        receipt.get("schema_version") == 1
        and isinstance(receipt.get("schema_version"), int)
        and not isinstance(receipt.get("schema_version"), bool)
        and receipt.get("artifact_role") == "reference_migration_artifact_receipt"
        and _positive_int(receipt.get("source_count"))
        and receipt["source_count"] == len(EXPECTED_V2_SOURCE_IDS)
        and isinstance(receipt_hashes, dict)
        and set(receipt_hashes) == expected_receipt_names
        and all(receipt_hashes.get(name) == artifact_hashes.get(name) for name in receipt_hashes)
        and receipt.get("artifact_set_sha256") == canonical_sha256(receipt_hashes)
    )
    integrity_checks = integrity.get("checks")
    input_policy = integrity.get("input_policy")
    repository_root = data_dir.parents[3]
    selection = _load_json(
        repository_root
        / "experiments/psem_training_strategy_gate/data/alimeeting_train_selection.json"
    )
    forced_alignment = selection.get("source_artifacts", {}).get("forced_alignment")
    selection_model_inputs = selection.get("selection_model_inputs")
    references = provenance.get("references")
    migration_rows = _load_jsonl(data_dir / "reference_migration.jsonl")
    migration_summary = _load_json(data_dir / "reference_migration_summary.json")
    inventory = _load_json(data_dir / "nonlexical_risk_inventory.json")
    normalization_rows = _load_jsonl(data_dir / "normalization_manifest.jsonl")
    try:
        load_nonlexical_inventory(data_dir / "nonlexical_risk_inventory.json")
        inventory_semantics_valid = True
    except ReferenceNormalizationError:
        inventory_semantics_valid = False
    observed = {
        "repository": provenance.get("reference_repository"),
        "commit": provenance.get("reference_commit"),
        "source_count_exact": _positive_int(provenance.get("source_count"))
        and provenance["source_count"] == len(EXPECTED_V2_SOURCE_IDS),
        "reference_count_exact": _positive_int(integrity.get("reference_count"))
        and integrity["reference_count"] == len(EXPECTED_V2_SOURCE_IDS),
        "status": integrity.get("status"),
        "integrity_report_shape_exact": (
            integrity.get("schema_version") == 1
            and isinstance(integrity.get("schema_version"), int)
            and not isinstance(integrity.get("schema_version"), bool)
            and integrity.get("artifact_role") == "reference_integrity_report"
            and integrity.get("scope")
            == "pipeline_correctness_not_independent_acoustic_boundary_accuracy"
            and _positive_int(integrity.get("source_count"))
            and integrity["source_count"] == len(EXPECTED_V2_SOURCE_IDS)
        ),
        "all_integrity_checks_pass": (
            isinstance(integrity_checks, dict)
            and set(integrity_checks) == EXPECTED_REFERENCE_INTEGRITY_CHECK_IDS
            and all(value is True for value in integrity_checks.values())
        ),
        "reference_input_policy_excludes_models": (
            isinstance(input_policy, dict)
            and input_policy.get("model_predictions_or_scores_accepted") is False
            and input_policy.get("selection_receipt_model_inputs") == selection_model_inputs
            and isinstance(selection_model_inputs, dict)
            and all(
                selection_model_inputs.get(field) is False
                for field in SELECTION_MODEL_EXCLUSION_FIELDS
            )
        ),
        "provenance_identity_exact": (
            provenance.get("schema_version") == 1
            and isinstance(provenance.get("schema_version"), int)
            and not isinstance(provenance.get("schema_version"), bool)
            and provenance.get("artifact_role") == "reference_provenance"
            and isinstance(forced_alignment, dict)
            and provenance.get("reference_git_tree") == forced_alignment.get("git_tree")
            and provenance.get("reference_license_ref") == forced_alignment.get("license_ref")
            and provenance.get("reference_license_sha256") == forced_alignment.get("license_sha256")
            and provenance.get("source_license_ids_by_corpus")
            == {"AMI": ["CC-BY-4.0"], "AliMeeting": ["CC-BY-SA-4.0"]}
        ),
        "reference_rows_exact": (
            isinstance(references, list)
            and len(references) == len(EXPECTED_V2_SOURCE_IDS)
            and all(isinstance(row, dict) for row in references)
            and {row.get("source_id") for row in references} == EXPECTED_V2_SOURCE_IDS
            and provenance.get("reference_inventory_sha256") == canonical_sha256(references)
        ),
        "migration_and_provenance_bound": (
            integrity.get("migration_session_manifest_sha256")
            == provenance.get("migration_session_manifest_sha256")
            == canonical_sha256(migration_rows)
            and integrity.get("migration_summary_sha256") == canonical_sha256(migration_summary)
            and integrity.get("reference_provenance_sha256") == canonical_sha256(provenance)
        ),
        "nonlexical_inventory_exact": (
            inventory_semantics_valid
            and canonical_sha256(inventory) == EXPECTED_INVENTORY_SHA256
            and provenance.get("nonlexical_inventory_sha256") == EXPECTED_INVENTORY_SHA256
            and all(
                row.get("nonlexical_inventory_sha256") == EXPECTED_INVENTORY_SHA256
                for row in normalization_rows
            )
        ),
        "reference_inventory_bound": (
            provenance.get("reference_inventory_sha256")
            == integrity.get("reference_inventory_sha256")
            == freeze.get("reference_binding", {}).get("reference_inventory_sha256")
        ),
        "receipt_current": receipt_current,
    }
    return _exact_check(
        "reference.integrity_and_provenance_current",
        observed,
        {
            "repository": REFERENCE_REPOSITORY,
            "commit": REFERENCE_COMMIT,
            "source_count_exact": True,
            "reference_count_exact": True,
            "status": "pass",
            "integrity_report_shape_exact": True,
            "all_integrity_checks_pass": True,
            "reference_input_policy_excludes_models": True,
            "provenance_identity_exact": True,
            "reference_rows_exact": True,
            "migration_and_provenance_bound": True,
            "nonlexical_inventory_exact": True,
            "reference_inventory_bound": True,
            "receipt_current": True,
        },
    )


def _v2_evaluator_contract_check(
    context: DatasetContext,
    data_dir: Path,
) -> dict[str, Any]:
    checked = _load_json(data_dir / "evaluator_contract.json")
    repository_root = context.data_dir.parents[3]
    try:
        rebuilt = build_evaluator_contract(
            data_dir,
            repository_root / "experiments/speaker_representation_scd/models/registry.json",
            repository_root / "experiments/speaker_representation_scd/models/source_registry.json",
        )
    except (OSError, RuntimeError) as exc:
        raise DatasetPreflightError("v2 evaluator contract cannot be rebuilt") from exc
    threshold_policy = checked.get("threshold_policy", {})
    model_policy = checked.get("model_policy", {})
    observed = {
        "current": canonical_sha256(checked) == canonical_sha256(rebuilt),
        "required_views": tuple(sorted(checked.get("required_outputs", {}))),
        "shared_threshold_vector_required": threshold_policy.get(
            "same_threshold_vector_required_for_every_output"
        ),
        "per_corpus_thresholds_allowed": threshold_policy.get("per_corpus_thresholds_allowed"),
        "model_policy_excluded": all(
            model_policy.get(field) is False
            for field in (
                "model_predictions_consulted_for_contract",
                "model_scores_consulted_for_contract",
                "official_model_results_inspected_for_contract",
                "official_model_training_performed_for_contract",
            )
        ),
    }
    return _exact_check(
        "evaluator.shared_threshold_contract_current",
        observed,
        {
            "current": True,
            "required_views": tuple(sorted(REQUIRED_VIEW_IDS)),
            "shared_threshold_vector_required": True,
            "per_corpus_thresholds_allowed": False,
            "model_policy_excluded": True,
        },
    )


def _accepted_freeze_check(data_dir: Path, freeze: dict[str, Any]) -> dict[str, Any]:
    return _exact_check(
        "freeze.current_and_internally_consistent",
        {
            "current": True,
            "manifest_sha256": sha256_file(data_dir / "dataset_freeze.json"),
            "payload_sha256": freeze.get("freeze_payload_sha256"),
        },
        {
            "current": True,
            "manifest_sha256": ACCEPTED_FREEZE_MANIFEST_SHA256,
            "payload_sha256": ACCEPTED_FREEZE_PAYLOAD_SHA256,
        },
    )


def _v2_accepted_freeze_check(data_dir: Path, freeze: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(freeze)
    observed_payload_sha256 = payload.pop("freeze_core_payload_sha256", None)
    expected_payload_sha256 = canonical_sha256(payload)
    return _exact_check(
        "freeze.current_and_internally_consistent",
        {
            "current": observed_payload_sha256 == expected_payload_sha256,
            "freeze_core_payload_sha256": observed_payload_sha256,
        },
        {
            "current": True,
            "freeze_core_payload_sha256": expected_payload_sha256,
        },
    )


def _build_v2_dataset_preflight(
    context: DatasetContext,
    freeze: dict[str, Any],
) -> dict[str, Any]:
    data_dir = context.data_dir
    split = _load_json(data_dir / "split_manifest.json")
    contract = _load_json(data_dir / "operational_label_contract.json")
    census = _load_json(data_dir / "topology_census.json")
    source_rows = _load_jsonl(data_dir / "source_manifest.jsonl")
    annotation_rows = _load_jsonl(data_dir / "annotation_manifest.jsonl")
    normalization_rows = _load_jsonl(data_dir / "normalization_manifest.jsonl")
    topology_rows = _load_jsonl(data_dir / "topology_manifest.jsonl")
    split_sources = split.get("assignments", {}).get("sources")
    if not isinstance(split_sources, list) or any(
        not isinstance(row, dict) for row in split_sources
    ):
        raise DatasetPreflightError("v2 split source assignments are invalid")
    selected_source_ids = {row.get("source_id") for row in split_sources}
    if any(not isinstance(source_id, str) or not source_id for source_id in selected_source_ids):
        raise DatasetPreflightError("v2 split source identifiers are invalid")
    checks = [
        *_v2_gate_checks(split),
        *_leakage_checks(split),
        _v2_annotation_coverage_check(
            context,
            source_rows,
            annotation_rows,
            normalization_rows,
            selected_source_ids,
        ),
        _masking_check(
            contract,
            census,
            topology_rows,
            V2_EXPECTED_MASK_RULES,
        ),
        _exclusive_counting_check(contract, census, topology_rows),
        _v2_reference_integrity_check(data_dir, freeze),
        _v2_evaluator_contract_check(context, data_dir),
        *_hash_checks(
            data_dir,
            freeze,
            source_rows,
            annotation_rows,
            normalization_rows,
            topology_rows,
            split_sources,
            context,
        ),
        _v2_contract_check(context, contract),
        _exact_check(
            "freeze.dataset_freeze_id_present_and_consistent",
            freeze.get("dataset_freeze_id"),
            context.freeze_id,
        ),
        _v2_accepted_freeze_check(data_dir, freeze),
        _exact_check("data.natural_only", split.get("natural_data_only"), True),
        _exact_check(
            "split.model_derived_quantities_forbidden",
            split.get("search", {}).get("model_derived_quantities_allowed"),
            False,
        ),
    ]
    for field in NO_MODEL_FIELDS:
        checks.append(
            _exact_check(
                f"model_boundary.{field}",
                freeze.get("model_policy", {}).get(field),
                False,
            )
        )
    _require_check_inventory(checks, V2_REQUIRED_CHECK_IDS)
    failed_checks = [copy.deepcopy(check) for check in checks if check["passed"] is not True]
    payload = {
        "schema_version": 1,
        "artifact_role": "psem_dataset_preflight",
        "dataset_freeze_id": freeze["dataset_freeze_id"],
        "authority_ref": context.authority_ref,
        "authority_pin": context.authority_pin,
        "contract_version": context.label_contract.contract_version,
        "generator": "experiments.psem_training_strategy_gate.data.dataset_preflight",
        "generator_version": GENERATOR_VERSION,
        "freeze_binding": {
            "freeze_core_payload_sha256": freeze["freeze_core_payload_sha256"],
            "split_manifest_sha256": freeze["artifact_sha256"]["split_manifest.json"],
            "source_manifest_sha256": freeze["artifact_sha256"]["source_manifest.jsonl"],
            "annotation_manifest_sha256": freeze["artifact_sha256"]["annotation_manifest.jsonl"],
            "normalization_manifest_sha256": freeze["artifact_sha256"][
                "normalization_manifest.jsonl"
            ],
            "reference_artifact_receipt_sha256": freeze["reference_binding"][
                "reference_artifact_receipt_sha256"
            ],
            "evaluator_contract_sha256": freeze["evaluator_binding"]["evaluator_contract_sha256"],
        },
        "checks": checks,
        "ready_for_issue_76": not failed_checks,
        "failed_checks": failed_checks,
    }
    return {**payload, "preflight_payload_sha256": canonical_sha256(payload)}


def build_dataset_preflight(data_dir: Path) -> dict[str, Any]:
    try:
        context = resolve_dataset_context(data_dir)
    except DatasetContextError as exc:
        raise DatasetPreflightError("dataset context is invalid") from exc
    if context.is_v2:
        try:
            freeze_core = build_v2_dataset_freeze_core(data_dir)
        except DatasetFreezeError as exc:
            raise DatasetPreflightError("v2 dataset freeze inputs are invalid") from exc
        return _build_v2_dataset_preflight(context, freeze_core)
    try:
        freeze = validate_checked_dataset_freeze(data_dir)
    except DatasetFreezeError as exc:
        raise DatasetPreflightError("accepted dataset freeze is not current") from exc
    split = _load_json(data_dir / "split_manifest.json")
    contract = _load_json(data_dir / "operational_label_contract.json")
    census = _load_json(data_dir / "topology_census.json")
    source_rows = _load_jsonl(data_dir / "source_manifest.jsonl")
    annotation_rows = _load_jsonl(data_dir / "annotation_manifest.jsonl")
    normalization_rows = _load_jsonl(data_dir / "normalization_manifest.jsonl")
    topology_rows = _load_jsonl(data_dir / "topology_manifest.jsonl")
    split_sources = split.get("assignments", {}).get("sources")
    if not isinstance(split_sources, list) or any(
        not isinstance(row, dict) for row in split_sources
    ):
        raise DatasetPreflightError("split source assignments are invalid")
    selected_source_ids = {row.get("source_id") for row in split_sources}
    if any(not isinstance(source_id, str) or not source_id for source_id in selected_source_ids):
        raise DatasetPreflightError("split source identifiers are invalid")
    checks = [
        *_gate_checks(split),
        *_leakage_checks(split),
        _annotation_coverage_check(
            source_rows,
            annotation_rows,
            normalization_rows,
            selected_source_ids,
        ),
        _masking_check(contract, census, topology_rows),
        _exclusive_counting_check(contract, census, topology_rows),
        *_hash_checks(
            data_dir,
            freeze,
            source_rows,
            annotation_rows,
            normalization_rows,
            topology_rows,
            split_sources,
        ),
        _contract_check(contract),
        _exact_check(
            "freeze.dataset_freeze_id_present_and_consistent",
            freeze.get("dataset_freeze_id"),
            DATASET_FREEZE_ID,
        ),
        _accepted_freeze_check(data_dir, freeze),
        _exact_check("data.natural_only", split.get("natural_data_only"), True),
        _exact_check(
            "split.model_derived_quantities_forbidden",
            split.get("search", {}).get("model_derived_quantities_allowed"),
            False,
        ),
    ]
    for field in NO_MODEL_FIELDS:
        checks.append(
            _exact_check(
                f"model_boundary.{field}",
                freeze.get("model_policy", {}).get(field),
                False,
            )
        )
    _require_check_inventory(checks)
    failed_checks = [copy.deepcopy(check) for check in checks if check["passed"] is not True]
    payload = {
        "schema_version": 1,
        "artifact_role": "psem_dataset_preflight",
        "dataset_freeze_id": freeze["dataset_freeze_id"],
        "authority_ref": AUTHORITY_REF,
        "authority_pin": AUTHORITY_PIN,
        "contract_version": CONTRACT_VERSION,
        "generator": "experiments.psem_training_strategy_gate.data.dataset_preflight",
        "generator_version": GENERATOR_VERSION,
        "freeze_binding": {
            "dataset_freeze_manifest_sha256": sha256_file(data_dir / "dataset_freeze.json"),
            "freeze_payload_sha256": freeze["freeze_payload_sha256"],
            "split_manifest_sha256": freeze["artifact_sha256"]["split_manifest.json"],
            "source_manifest_sha256": freeze["artifact_sha256"]["source_manifest.jsonl"],
            "annotation_manifest_sha256": freeze["artifact_sha256"]["annotation_manifest.jsonl"],
        },
        "checks": checks,
        "ready_for_issue_76": not failed_checks,
        "failed_checks": failed_checks,
    }
    return {**payload, "preflight_payload_sha256": canonical_sha256(payload)}


def validate_checked_dataset_preflight(
    data_dir: Path,
    report_path: Path | None = None,
) -> dict[str, Any]:
    checked = _load_json(report_path or data_dir / "preflight_report.json")
    expected = build_dataset_preflight(data_dir)
    if canonical_sha256(checked) != canonical_sha256(expected):
        raise DatasetPreflightError("checked dataset preflight is not current")
    payload = copy.deepcopy(checked)
    observed_digest = payload.pop("preflight_payload_sha256", None)
    if observed_digest != canonical_sha256(payload):
        raise DatasetPreflightError("dataset preflight payload digest is invalid")
    if checked.get("ready_for_issue_76") is not True or checked.get("failed_checks") != []:
        raise DatasetPreflightError("dataset preflight did not pass")
    try:
        context = resolve_dataset_context(data_dir)
    except DatasetContextError as exc:
        raise DatasetPreflightError("dataset context is invalid") from exc
    if context.is_v2:
        try:
            freeze = validate_checked_dataset_freeze(data_dir)
        except DatasetFreezeError as exc:
            raise DatasetPreflightError("final v2 dataset freeze is not current") from exc
        binding = freeze.get("preflight_binding", {})
        if (
            binding.get("preflight_report_sha256")
            != sha256_file(report_path or data_dir / "preflight_report.json")
            or binding.get("preflight_report_canonical_sha256") != canonical_sha256(checked)
            or binding.get("preflight_payload_sha256") != observed_digest
            or binding.get("freeze_core_payload_sha256")
            != expected["freeze_binding"]["freeze_core_payload_sha256"]
            or binding.get("check_count") != len(checked["checks"])
            or binding.get("ready_for_issue_76") is not True
        ):
            raise DatasetPreflightError("final v2 freeze does not bind this preflight")
    return checked


def _write_json_atomic(output_path: Path, value: dict[str, Any]) -> None:
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(
                json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            )
            temporary_path = Path(handle.name)
        temporary_path.replace(output_path)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _incomplete_report(data_dir: Path, reason: str) -> dict[str, Any]:
    freeze_id: Any = None
    authority_ref = AUTHORITY_REF
    authority_pin = AUTHORITY_PIN
    contract_version = CONTRACT_VERSION
    try:
        context = resolve_dataset_context(data_dir)
        authority_ref = context.authority_ref
        authority_pin = context.authority_pin
        contract_version = context.label_contract.contract_version
    except DatasetContextError:
        pass
    freeze_path = data_dir / "dataset_freeze.json"
    try:
        freeze_id = _load_json(freeze_path).get("dataset_freeze_id")
    except DatasetPreflightError:
        pass
    check = _exact_check(
        "preflight.evaluation_completed_against_current_freeze",
        False,
        True,
        reason=reason,
    )
    payload = {
        "schema_version": 1,
        "artifact_role": "psem_dataset_preflight",
        "dataset_freeze_id": freeze_id,
        "authority_ref": authority_ref,
        "authority_pin": authority_pin,
        "contract_version": contract_version,
        "generator": "experiments.psem_training_strategy_gate.data.dataset_preflight",
        "generator_version": GENERATOR_VERSION,
        "freeze_binding": None,
        "checks": [check],
        "ready_for_issue_76": False,
        "failed_checks": [copy.deepcopy(check)],
    }
    return {**payload, "preflight_payload_sha256": canonical_sha256(payload)}


def write_dataset_preflight(data_dir: Path, output_path: Path) -> None:
    _write_json_atomic(
        output_path,
        _incomplete_report(data_dir, "preflight evaluation did not complete"),
    )
    try:
        value = build_dataset_preflight(data_dir)
    except (DatasetPreflightError, DatasetFreezeError) as exc:
        _write_json_atomic(output_path, _incomplete_report(data_dir, str(exc)))
        raise DatasetPreflightError("dataset preflight failed closed") from exc
    _write_json_atomic(output_path, value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_dataset_preflight(args.data_dir.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
