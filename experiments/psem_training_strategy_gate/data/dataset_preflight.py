from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from experiments.psem_training_strategy_gate.data.dataset_freeze import (
    AUTHORITY_PIN,
    AUTHORITY_REF,
    CONTRACT_VERSION,
    DATASET_FREEZE_ID,
    NO_MODEL_FIELDS,
    DatasetFreezeError,
    validate_checked_dataset_freeze,
)
from experiments.psem_training_strategy_gate.data.provenance import (
    canonical_sha256,
    sha256_file,
)

SAMPLE_RATE_HZ = 16000
GENERATOR_VERSION = "1"
ACCEPTED_FREEZE_MANIFEST_SHA256 = (
    "b600e7050bc3ac92a8393395837452f43711485adfa006547b2d73df1b15fdde"
)
ACCEPTED_FREEZE_PAYLOAD_SHA256 = (
    "1606f4cc1f497f4bbcd92be3cbf38fe4fb8735edcb454cfd45ab0fddc542261e"
)
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
    "gap_above_local_continuity_maximum": (
        "continuity_unknown_and_mask_handoff_and_relation"
    ),
    "activity_supervision": "retain_when_interval_activity_is_reliable",
}
EXPECTED_PRECEDENCE = (
    "short_backchannel_return",
    "overlap_takeover",
    "overlap_return",
    "silence_gap_different_speaker_handoff",
    "same_speaker_silence_gap_resume",
    "clean_direct_different_speaker_handoff",
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


def _nonnegative_int_map(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and all(isinstance(key, str) and key for key in value)
        and all(
            isinstance(count, int) and not isinstance(count, bool) and count >= 0
            for count in value.values()
        )
    )


def _require_check_inventory(checks: list[dict[str, Any]]) -> None:
    observed_ids = tuple(check.get("id") for check in checks)
    if (
        len(REQUIRED_CHECK_IDS) != 42
        or len(set(REQUIRED_CHECK_IDS)) != 42
        or observed_ids != REQUIRED_CHECK_IDS
    ):
        raise DatasetPreflightError("preflight check inventory is not the fixed 42-check contract")


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
    for source_id in sorted(selected_source_ids & set(sources) & set(annotations) & set(normalizations)):
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
            or source_start == annotation_start
            and source_end == annotation_end
        )
        valid = (
            all(
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
            and source.get("sample_rate_hz") == SAMPLE_RATE_HZ
            and source.get("contract_document_sha256")
            == ACCEPTED_CONTRACT_CANONICAL_SHA256
            and annotation.get("contract_document_sha256")
            == ACCEPTED_CONTRACT_CANONICAL_SHA256
            and normalization.get("contract_document_sha256")
            == ACCEPTED_CONTRACT_CANONICAL_SHA256
            and source.get("contract_version") == CONTRACT_VERSION
            and annotation.get("contract_version") == CONTRACT_VERSION
            and normalization.get("contract_version") == CONTRACT_VERSION
            and normalization.get("exposure", {}).get("scored_samples")
            == scored_end - scored_start
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
) -> dict[str, Any]:
    assertions: list[tuple[str, bool]] = [
        (
            f"contract_rule.{name}",
            contract.get("ambiguity_and_mask_rules", {}).get(name) == expected,
        )
        for name, expected in EXPECTED_MASK_RULES.items()
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
        valid_maps = _nonnegative_int_map(reasons) and _nonnegative_int_map(
            diagnostic_regions
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
    overall_maps_valid = _nonnegative_int_map(overall_reasons) and _nonnegative_int_map(
        overall_diagnostic_regions
    )
    assertions.extend(
        (
            ("census.masked_transition_count", row_masked == overall_mask.get("masked_transition_count")),
            ("census.actual_transition_count", row_actual == overall_mask.get("actual_transition_count")),
            ("census.mask_maps", overall_maps_valid),
            (
                "census.masked_transition_reasons",
                overall_maps_valid
                and dict(sorted(row_reasons.items())) == overall_reasons,
            ),
            (
                "census.diagnostic_masked_region_counts",
                overall_maps_valid
                and dict(sorted(row_diagnostic_regions.items()))
                == overall_diagnostic_regions,
            ),
            (
                "census.masked_reason_total",
                overall_maps_valid
                and sum(overall_reasons.values())
                == overall_mask.get("masked_transition_count"),
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
            tuple(contract.get("official_primary_topology_precedence", ()))
            == EXPECTED_PRECEDENCE,
        ),
        (
            "contract.one_episode_one_gate",
            contract.get("exclusive_counting", {}).get(
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
            census.get("counting_policy", {}).get("old_r7_or_r7b_event_counts_used")
            is False,
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
            and all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in counts.values())
        )
        assertions.append((f"row.{source_id}.topology_inventory", valid_counts))
        if valid_counts:
            row_total = sum(counts.values())
            aggregate_counts.update(counts)
            aggregate_episode_count += row_total
            assertions.append(
                (
                    f"row.{source_id}.exclusive_episode_count",
                    row_total == row.get("exclusive_primary_episode_count"),
                )
            )
    overall = census.get("overall", {})
    assertions.extend(
        (
            (
                "census.aggregate_primary_counts",
                dict(sorted(aggregate_counts.items())) == overall.get("primary_topology_counts"),
            ),
            (
                "census.aggregate_exclusive_episode_count",
                aggregate_episode_count == overall.get("exclusive_primary_episode_count"),
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
) -> list[dict[str, Any]]:
    repository_root = data_dir.parents[2]
    artifact_hashes = freeze.get("artifact_sha256", {})
    repository_hashes = freeze.get("repository_input_sha256", {})
    violations = []
    matched = 0
    for name, expected in artifact_hashes.items():
        path = data_dir / name
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
        len(artifact_hashes) + len(repository_hashes),
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
            and annotation_sha256 == normalization.get("annotation_sha256")
            and annotation_sha256 == topology_row.get("annotation_sha256")
            and label_result_sha256 == topology_row.get("label_result_sha256")
            and topology_row.get("normalization_row_sha256")
            == canonical_sha256(normalization)
            and topology_row.get("scored_start_sample")
            == normalization.get("scored_start_sample")
            and topology_row.get("scored_end_sample")
            == normalization.get("scored_end_sample")
            and topology_row.get("scored_samples")
            == normalization.get("exposure", {}).get("scored_samples")
            and topology_row.get("contract_document_sha256")
            == ACCEPTED_CONTRACT_CANONICAL_SHA256
            and topology_row.get("contract_version") == CONTRACT_VERSION
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


def build_dataset_preflight(data_dir: Path) -> dict[str, Any]:
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
    if not isinstance(split_sources, list) or any(not isinstance(row, dict) for row in split_sources):
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
    if checked != expected:
        raise DatasetPreflightError("checked dataset preflight is not current")
    payload = copy.deepcopy(checked)
    observed_digest = payload.pop("preflight_payload_sha256", None)
    if observed_digest != canonical_sha256(payload):
        raise DatasetPreflightError("dataset preflight payload digest is invalid")
    if checked.get("ready_for_issue_76") is not True or checked.get("failed_checks") != []:
        raise DatasetPreflightError("dataset preflight did not pass")
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
                json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
            temporary_path = Path(handle.name)
        temporary_path.replace(output_path)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _incomplete_report(data_dir: Path, reason: str) -> dict[str, Any]:
    freeze_id: Any = None
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
        "authority_ref": AUTHORITY_REF,
        "authority_pin": AUTHORITY_PIN,
        "contract_version": CONTRACT_VERSION,
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
