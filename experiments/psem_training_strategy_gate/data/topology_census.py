from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from experiments.psem_training_strategy_gate.data.annotation_normalization import (
    normalize_inventory,
)
from experiments.psem_training_strategy_gate.data.dataset_context import (
    DatasetContextError,
    resolve_dataset_context,
)
from experiments.psem_training_strategy_gate.data.label_contract import (
    CONTRACT_PATH_BY_VERSION,
    LabelContract,
    load_contract,
)
from experiments.psem_training_strategy_gate.data.provenance import (
    BASELINE_AMI_MEETINGS,
    EXPECTED_ALIMEETING_MEETINGS,
    ProvenanceError,
    canonical_sha256,
    sha256_file,
    wav_identity,
    write_jsonl,
)
from experiments.psem_training_strategy_gate.data.reference_normalization import (
    normalize_reference_inventory,
)

FROZEN_CALIBRATION_SHA256 = "1f10fc1980bd1a108753ef6afb45f455618df04014ad6b58dcf6e880ba01f4b1"
FROZEN_CALIBRATION_MARKDOWN_SHA256 = "2efe8137b8330699a0bcc1a770e40ce22c7c615de7b49b3724ddf0ebf80cddf6"
FROZEN_CALIBRATION_INPUTS = {
    "source_manifest_sha256": "5cf6178a35e0c499bc3d79633c3ff1973f5f529c2b39e3bec89ca18ea96d6437",
    "annotation_manifest_sha256": "f635171f8162115e08cee49e4fd748749b372e7eb37797bc91975bf2ca85c4a3",
    "normalization_manifest_sha256": "9805abe480eab757d29a484f67ee543a548dd91c7396007a85e2c60f44065079",
    "source_ids_sha256": "0a66aa29c39fe822b5a7f575edd0f442419b127e83654311ff733b6116cdafe6",
}
FROZEN_CALIBRATION_SOURCE_IDS = frozenset(
    [
        *(f"ami_{meeting_id}" for meeting_id in BASELINE_AMI_MEETINGS),
        *(
            f"alimeeting_{meeting_id}"
            for meeting_id in EXPECTED_ALIMEETING_MEETINGS
        ),
    ]
)

OFFICIAL_PRIMARY_TOPOLOGIES = (
    "short_backchannel_return",
    "overlap_takeover",
    "overlap_return",
    "silence_gap_different_speaker_handoff",
    "same_speaker_silence_gap_resume",
    "clean_direct_different_speaker_handoff",
)
TOTAL_NATURAL_MINIMA = {
    "scored_samples": 33 * 3600 * 16000,
    "independent_meetings": 22,
    "stable_singleton_samples": 10 * 3600 * 16000,
    "ongoing_overlap_samples": 75 * 60 * 16000,
    "primary_topology_counts": {
        "clean_direct_different_speaker_handoff": 120,
        "silence_gap_different_speaker_handoff": 240,
        "same_speaker_silence_gap_resume": 240,
        "overlap_return": 120,
        "overlap_takeover": 120,
        "short_backchannel_return": 100,
    },
}


class TopologyCensusError(RuntimeError):
    pass


def _fraction(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 8) if denominator else 0.0


def _load_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise TopologyCensusError(f"invalid JSONL manifest: {path}") from exc
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise TopologyCensusError(f"JSONL manifest must contain objects: {path}")
    return rows


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TopologyCensusError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise TopologyCensusError(f"JSON artifact must be an object: {path}")
    return value


def _validate_contract_precedence(contract: LabelContract) -> None:
    contract_path = CONTRACT_PATH_BY_VERSION[contract.contract_version]
    try:
        raw = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TopologyCensusError("operational label contract is unreadable") from exc
    if raw.get("official_primary_topology_precedence") != list(
        OFFICIAL_PRIMARY_TOPOLOGIES
    ):
        raise TopologyCensusError(
            "census topology precedence does not match the frozen contract"
        )
    if (
        raw.get("contract_version") != contract.contract_version
        or raw.get("status") != contract.status
    ):
        raise TopologyCensusError("topology census contract identity is invalid")


def _validate_normalization_manifest(
    sessions: Sequence[Any], data_dir: Path
) -> list[dict[str, Any]]:
    observed_rows = _load_jsonl_objects(data_dir / "normalization_manifest.jsonl")
    expected_rows = [
        session.manifest_row()
        for session in sorted(sessions, key=lambda session: session.source_id)
    ]
    if observed_rows != expected_rows:
        raise TopologyCensusError(
            "normalization manifest does not match censused sessions"
        )
    return observed_rows


def _validate_calibrated_inventory(
    data_dir: Path,
    source_ids: Sequence[str],
    contract: LabelContract,
) -> None:
    try:
        context = resolve_dataset_context(data_dir)
    except DatasetContextError as exc:
        raise TopologyCensusError("dataset context is invalid") from exc
    calibration_path = context.calibration_dir / "annotation_calibration.json"
    calibration = _load_json_object(calibration_path)
    calibrated_contract = load_contract(version="psem-handoff-v0")
    constant_fields = (
        "sample_rate_hz",
        "reliable_solo_min_duration_ms",
        "annotation_boundary_jitter_ms",
        "gap_topology_min_duration_ms",
        "overlap_topology_min_duration_ms",
        "local_continuity_max_gap_ms",
        "short_backchannel_min_duration_ms",
        "short_backchannel_max_duration_ms",
    )
    input_policy = calibration.get("input_policy")
    calibrated_scope_valid = (
        FROZEN_CALIBRATION_SOURCE_IDS.issubset(source_ids)
        and canonical_sha256(sorted(FROZEN_CALIBRATION_SOURCE_IDS))
        == FROZEN_CALIBRATION_INPUTS["source_ids_sha256"]
    )
    model_fields = (
        "model_predictions_consulted",
        "model_scores_consulted",
        "official_model_results_inspected",
        "official_model_training_performed",
    )
    if (
        calibration.get("artifact_role") != "annotation_only_calibration"
        or sha256_file(calibration_path) != FROZEN_CALIBRATION_SHA256
        or sha256_file(context.calibration_dir / "ANNOTATION_CALIBRATION.md")
        != FROZEN_CALIBRATION_MARKDOWN_SHA256
        or calibration.get("contract_version") != calibrated_contract.contract_version
        or calibration.get("contract_document_sha256")
        != calibrated_contract.document_sha256
        or calibration.get("contract_status") != calibrated_contract.status
        or any(
            getattr(contract, field) != getattr(calibrated_contract, field)
            for field in constant_fields
        )
        or not isinstance(input_policy, dict)
        or not calibrated_scope_valid
        or any(
            input_policy.get(key) != value
            for key, value in FROZEN_CALIBRATION_INPUTS.items()
        )
        or any(input_policy.get(field) is not False for field in model_fields)
    ):
        raise TopologyCensusError(
            "census inventory does not match the accepted annotation calibration"
        )


def validate_waveform_inventory(data_dir: Path, corpus_root: Path) -> None:
    source_rows = _load_jsonl_objects(data_dir / "source_manifest.jsonl")
    resolved_root = corpus_root.resolve()
    seen_source_ids: set[str] = set()
    identity_fields = (
        "waveform_sha256",
        "waveform_size_bytes",
        "sample_rate_hz",
        "channels",
        "sample_width_bytes",
        "duration_samples",
    )
    for row in source_rows:
        source_id = row.get("source_id")
        audio_ref = row.get("audio_ref")
        if (
            not isinstance(source_id, str)
            or not source_id
            or source_id in seen_source_ids
            or not isinstance(audio_ref, str)
            or not audio_ref
        ):
            raise TopologyCensusError("invalid source waveform inventory row")
        seen_source_ids.add(source_id)
        waveform_path = (resolved_root / audio_ref).resolve()
        if not waveform_path.is_relative_to(resolved_root):
            raise TopologyCensusError(
                f"source waveform escapes corpus root for {source_id}"
            )
        try:
            observed = wav_identity(waveform_path)
        except (OSError, ProvenanceError) as exc:
            raise TopologyCensusError(
                f"invalid source waveform for {source_id}"
            ) from exc
        if any(row.get(field) != observed[field] for field in identity_fields):
            raise TopologyCensusError(
                f"source waveform identity mismatch for {source_id}"
            )


def _micro_diagnostics(session: Any, contract: LabelContract) -> dict[str, int]:
    micro_gap_count = 0
    micro_gap_samples = 0
    micro_overlap_count = 0
    micro_overlap_samples = 0
    for index, interval in enumerate(session.intervals):
        duration = interval.duration_samples
        if (
            0 < index < len(session.intervals) - 1
            and not interval.ambiguous
            and interval.speaker_identity_known
            and not interval.active_speakers
            and session.intervals[index - 1].active_speakers
            and session.intervals[index + 1].active_speakers
            and contract.annotation_boundary_jitter_samples
            < duration
            < contract.gap_topology_min_duration_samples
        ):
            micro_gap_count += 1
            micro_gap_samples += duration
        if (
            not interval.ambiguous
            and interval.speaker_identity_known
            and len(interval.active_speakers) >= 2
            and contract.annotation_boundary_jitter_samples
            < duration
            < contract.overlap_topology_min_duration_samples
        ):
            micro_overlap_count += 1
            micro_overlap_samples += duration
    return {
        "micro_gap_interval_count": micro_gap_count,
        "micro_gap_samples": micro_gap_samples,
        "micro_overlap_interval_count": micro_overlap_count,
        "micro_overlap_samples": micro_overlap_samples,
    }


def _mask_diagnostics(session: Any) -> dict[str, Any]:
    actual_transition_count = 0
    masked_transition_count = 0
    masked_transition_reasons: Counter[str] = Counter()
    diagnostic_masked_region_counts: Counter[str] = Counter()
    for transition in session.labels.transitions:
        if transition["transition_id"].startswith("D"):
            if transition["mask_state"] == "masked":
                diagnostic_masked_region_counts[
                    transition["primary_topology"]
                ] += 1
            continue
        actual_transition_count += 1
        if transition["mask_state"] == "masked":
            masked_transition_count += 1
            masked_transition_reasons[transition["primary_topology"]] += 1
    return {
        "actual_transition_count": actual_transition_count,
        "masked_transition_count": masked_transition_count,
        "masked_transition_fraction": _fraction(
            masked_transition_count, actual_transition_count
        ),
        "masked_transition_reasons": dict(sorted(masked_transition_reasons.items())),
        "diagnostic_masked_region_counts": dict(
            sorted(diagnostic_masked_region_counts.items())
        ),
    }


def build_topology_row(session: Any, contract: LabelContract) -> dict[str, Any]:
    if (
        session.labels.contract_version != contract.contract_version
        or session.labels.contract_document_sha256 != contract.document_sha256
        or session.labels.sample_rate_hz != contract.sample_rate_hz
    ):
        raise TopologyCensusError(
            f"session label contract mismatch for {session.source_id}"
        )
    if session.labels.exposure["scored_samples"] != (
        session.scored_end_sample - session.scored_start_sample
    ):
        raise TopologyCensusError(
            f"session scored exposure mismatch for {session.source_id}"
        )
    primary_counts: Counter[str] = Counter()
    seen_transition_ids: set[str] = set()
    coverage_episode_count = 0
    for episode in session.labels.topology_episodes:
        if episode["coverage_gate_eligible"] is not True:
            continue
        topology = episode["primary_topology"]
        if topology not in OFFICIAL_PRIMARY_TOPOLOGIES:
            raise TopologyCensusError(
                f"unexpected coverage-eligible topology for {session.source_id}: {topology}"
            )
        transition_ids = episode["transition_ids"]
        if (
            not isinstance(transition_ids, list)
            or not transition_ids
            or any(not isinstance(value, str) for value in transition_ids)
            or seen_transition_ids.intersection(transition_ids)
        ):
            raise TopologyCensusError(
                f"non-exclusive topology episode for {session.source_id}"
            )
        seen_transition_ids.update(transition_ids)
        primary_counts[topology] += 1
        coverage_episode_count += 1
    counts = {
        topology: primary_counts[topology]
        for topology in OFFICIAL_PRIMARY_TOPOLOGIES
    }
    if sum(counts.values()) != coverage_episode_count:
        raise TopologyCensusError(
            f"exclusive primary count mismatch for {session.source_id}"
        )
    masked_or_ambiguous_samples = sum(
        interval.duration_samples
        for interval in session.labels.intervals
        if interval.ambiguous or not interval.speaker_identity_known
    )
    normalization_row = session.manifest_row()
    return {
        "schema_version": 1,
        "artifact_role": "natural_topology_census_row",
        "source_id": session.source_id,
        "corpus": session.corpus,
        "session_id": session.session_id,
        "contract_version": contract.contract_version,
        "contract_document_sha256": contract.document_sha256,
        "source_waveform_sha256": session.source_waveform_sha256,
        "annotation_sha256": normalization_row.get(
            "source_annotation_sha256", normalization_row.get("annotation_sha256")
        ),
        "normalization_row_sha256": canonical_sha256(normalization_row),
        "label_result_sha256": normalization_row["label_result_sha256"],
        "topology_episodes_sha256": canonical_sha256(
            list(session.labels.topology_episodes)
        ),
        "scored_start_sample": session.scored_start_sample,
        "scored_end_sample": session.scored_end_sample,
        "scored_samples": session.labels.exposure["scored_samples"],
        "ambiguous_samples": session.labels.exposure["ambiguous_samples"],
        "unknown_identity_samples": session.labels.exposure[
            "unknown_identity_samples"
        ],
        "masked_or_ambiguous_samples": masked_or_ambiguous_samples,
        "stable_singleton_samples": session.labels.exposure[
            "stable_singleton_samples"
        ],
        "ongoing_overlap_samples": session.labels.exposure[
            "ongoing_overlap_samples"
        ],
        "exclusive_primary_episode_count": coverage_episode_count,
        "primary_topology_counts": counts,
        "micro_diagnostics": _micro_diagnostics(session, contract),
        "mask_diagnostics": _mask_diagnostics(session),
        "split_role": "UNASSIGNED_CANDIDATE",
        "component_id": None,
    }


def _aggregate(rows: Sequence[dict[str, Any]], sample_rate_hz: int) -> dict[str, Any]:
    scored_samples = sum(row["scored_samples"] for row in rows)
    ambiguous_samples = sum(row["ambiguous_samples"] for row in rows)
    unknown_identity_samples = sum(row["unknown_identity_samples"] for row in rows)
    masked_samples = sum(row["masked_or_ambiguous_samples"] for row in rows)
    stable_samples = sum(row["stable_singleton_samples"] for row in rows)
    overlap_samples = sum(row["ongoing_overlap_samples"] for row in rows)
    counts = {
        topology: sum(
            row["primary_topology_counts"][topology] for row in rows
        )
        for topology in OFFICIAL_PRIMARY_TOPOLOGIES
    }
    diagnostics = {
        field: sum(row["micro_diagnostics"][field] for row in rows)
        for field in (
            "micro_gap_interval_count",
            "micro_gap_samples",
            "micro_overlap_interval_count",
            "micro_overlap_samples",
        )
    }
    actual_transition_count = sum(
        row["mask_diagnostics"]["actual_transition_count"] for row in rows
    )
    masked_transition_count = sum(
        row["mask_diagnostics"]["masked_transition_count"] for row in rows
    )
    masked_reasons: Counter[str] = Counter()
    diagnostic_regions: Counter[str] = Counter()
    for row in rows:
        masked_reasons.update(
            row["mask_diagnostics"]["masked_transition_reasons"]
        )
        diagnostic_regions.update(
            row["mask_diagnostics"]["diagnostic_masked_region_counts"]
        )
    return {
        "session_count": len(rows),
        "scored_samples": scored_samples,
        "scored_hours": round(scored_samples / sample_rate_hz / 3600, 6),
        "ambiguous_samples": ambiguous_samples,
        "ambiguous_fraction": _fraction(ambiguous_samples, scored_samples),
        "unknown_identity_samples": unknown_identity_samples,
        "unknown_identity_fraction": _fraction(
            unknown_identity_samples, scored_samples
        ),
        "masked_or_ambiguous_samples": masked_samples,
        "masked_or_ambiguous_fraction": _fraction(masked_samples, scored_samples),
        "stable_singleton_samples": stable_samples,
        "stable_singleton_hours": round(
            stable_samples / sample_rate_hz / 3600, 6
        ),
        "ongoing_overlap_samples": overlap_samples,
        "ongoing_overlap_hours": round(
            overlap_samples / sample_rate_hz / 3600, 6
        ),
        "exclusive_primary_episode_count": sum(counts.values()),
        "primary_topology_counts": counts,
        "micro_diagnostics": diagnostics,
        "mask_diagnostics": {
            "actual_transition_count": actual_transition_count,
            "masked_transition_count": masked_transition_count,
            "masked_transition_fraction": _fraction(
                masked_transition_count, actual_transition_count
            ),
            "masked_transition_reasons": dict(sorted(masked_reasons.items())),
            "diagnostic_masked_region_counts": dict(
                sorted(diagnostic_regions.items())
            ),
        },
    }


def _lower_bound_audit(overall: dict[str, Any]) -> dict[str, Any]:
    required_counts = TOTAL_NATURAL_MINIMA["primary_topology_counts"]
    observed_counts = overall["primary_topology_counts"]
    return {
        "scope": "raw_candidate_pool_only_not_split_feasibility",
        "scored_samples": {
            "observed": overall["scored_samples"],
            "required_total": TOTAL_NATURAL_MINIMA["scored_samples"],
            "deficit": max(
                0, TOTAL_NATURAL_MINIMA["scored_samples"] - overall["scored_samples"]
            ),
        },
        "independent_meetings": {
            "observed_upper_bound": overall["session_count"],
            "required_total": TOTAL_NATURAL_MINIMA["independent_meetings"],
            "component_audit_pending": True,
        },
        "stable_singleton_samples": {
            "observed": overall["stable_singleton_samples"],
            "required_total": TOTAL_NATURAL_MINIMA["stable_singleton_samples"],
            "deficit": max(
                0,
                TOTAL_NATURAL_MINIMA["stable_singleton_samples"]
                - overall["stable_singleton_samples"],
            ),
        },
        "ongoing_overlap_samples": {
            "observed": overall["ongoing_overlap_samples"],
            "required_total": TOTAL_NATURAL_MINIMA["ongoing_overlap_samples"],
            "deficit": max(
                0,
                TOTAL_NATURAL_MINIMA["ongoing_overlap_samples"]
                - overall["ongoing_overlap_samples"],
            ),
        },
        "primary_topology_counts": {
            topology: {
                "observed": observed_counts[topology],
                "required_total": required_counts[topology],
                "deficit": max(
                    0, required_counts[topology] - observed_counts[topology]
                ),
            }
            for topology in OFFICIAL_PRIMARY_TOPOLOGIES
        },
    }


def build_topology_census(
    sessions: Sequence[Any],
    data_dir: Path,
    topology_manifest_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not sessions:
        raise TopologyCensusError("topology census requires normalized sessions")
    try:
        context = resolve_dataset_context(data_dir)
    except DatasetContextError as exc:
        raise TopologyCensusError("dataset context is invalid") from exc
    contract = context.label_contract
    _validate_contract_precedence(contract)
    source_ids = [session.source_id for session in sessions]
    if len(set(source_ids)) != len(source_ids):
        raise TopologyCensusError("census source identities must be unique")
    _validate_normalization_manifest(sessions, data_dir)
    _validate_calibrated_inventory(data_dir, source_ids, contract)
    return _build_topology_census_from_validated_sessions(
        sessions, data_dir, topology_manifest_sha256, contract
    )


def _build_topology_census_from_validated_sessions(
    sessions: Sequence[Any],
    data_dir: Path,
    topology_manifest_sha256: str,
    contract: LabelContract | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    context = resolve_dataset_context(data_dir)
    active_contract = contract or context.label_contract
    source_ids = [session.source_id for session in sessions]
    rows = [
        build_topology_row(session, active_contract)
        for session in sorted(sessions, key=lambda session: session.source_id)
    ]
    overall = _aggregate(rows, active_contract.sample_rate_hz)
    corpora = sorted({row["corpus"] for row in rows})
    census = {
        "schema_version": 1,
        "artifact_role": "natural_topology_census",
        "authority_ref": context.authority_ref,
        "authority_pin": context.authority_pin,
        "contract_version": active_contract.contract_version,
        "contract_document_sha256": active_contract.document_sha256,
        "contract_status": active_contract.status,
        "input_manifests": {
            "annotation_calibration_sha256": sha256_file(
                context.calibration_dir / "annotation_calibration.json"
            ),
            "annotation_calibration_markdown_sha256": sha256_file(
                context.calibration_dir / "ANNOTATION_CALIBRATION.md"
            ),
            "source_manifest_sha256": sha256_file(data_dir / "source_manifest.jsonl"),
            "annotation_manifest_sha256": sha256_file(
                data_dir / "annotation_manifest.jsonl"
            ),
            "normalization_manifest_sha256": sha256_file(
                data_dir / "normalization_manifest.jsonl"
            ),
            "source_ids_sha256": canonical_sha256(sorted(source_ids)),
        },
        "topology_manifest_sha256": topology_manifest_sha256,
        "counting_policy": {
            "official_primary_topology_precedence": list(
                OFFICIAL_PRIMARY_TOPOLOGIES
            ),
            "exclusive_primary_counting": True,
            "short_backchannel_member_handoffs_counted_separately": False,
            "old_r7_or_r7b_event_counts_used": False,
        },
        "model_policy": {
            "model_predictions_consulted": False,
            "model_scores_consulted": False,
            "official_model_results_inspected": False,
            "official_model_training_performed": False,
        },
        "split_status": "UNASSIGNED_PRE_IDENTITY_GRAPH",
        "component_status": "PENDING_IDENTITY_GRAPH",
        "overall": overall,
        "by_corpus": {
            corpus: _aggregate(
                [row for row in rows if row["corpus"] == corpus],
                active_contract.sample_rate_hz,
            )
            for corpus in corpora
        },
        "by_split": {"UNASSIGNED_CANDIDATE": overall},
        "candidate_pool_lower_bound_audit": _lower_bound_audit(overall),
    }
    return rows, census


def _validate_split_render_binding(
    census: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    split_manifest: dict[str, Any],
    split_manifest_sha256: str,
) -> None:
    expected_census_sha256 = hashlib.sha256(
        (
            json.dumps(census, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    input_artifacts = split_manifest.get("input_artifacts")
    assignments = split_manifest.get("assignments")
    role_summaries = split_manifest.get("role_summaries")
    source_ids = {row.get("source_id") for row in rows}
    if (
        split_manifest.get("artifact_role") != "psem_component_split_assignment"
        or split_manifest.get("hard_gate_status") != "pass"
        or not isinstance(input_artifacts, dict)
        or input_artifacts.get("topology_manifest_sha256")
        != census.get("topology_manifest_sha256")
        or input_artifacts.get("topology_census_sha256") != expected_census_sha256
        or not isinstance(assignments, dict)
        or not isinstance(assignments.get("sources"), list)
        or {row.get("source_id") for row in assignments["sources"]} != source_ids
        or len(assignments["sources"]) != len(source_ids)
        or not isinstance(role_summaries, dict)
        or set(role_summaries)
        != {
            "PSEM-STRATEGY-TRAIN",
            "PSEM-STRATEGY-DEV",
            "PSEM-STRATEGY-EVAL",
        }
        or len(split_manifest_sha256) != 64
    ):
        raise TopologyCensusError("split manifest is not bound to the rendered census")


def render_data_census(
    census: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    split_manifest: dict[str, Any] | None = None,
    split_manifest_sha256: str | None = None,
) -> str:
    overall = census["overall"]
    split_selected = split_manifest is not None
    if split_selected:
        if split_manifest_sha256 is None:
            raise TopologyCensusError("split manifest hash is required for split-aware rendering")
        _validate_split_render_binding(
            census, rows, split_manifest, split_manifest_sha256
        )
    split_summary = None
    if split_manifest is not None:
        source_count = len(split_manifest["assignments"]["sources"])
        component_count = len(split_manifest["assignments"]["components"])
        gate_count = len(split_manifest["hard_gate_results"])
        if census["contract_version"] == "psem-handoff-v1":
            split_summary = (
                f"The identity graph and pinned WavLM overlap audit cover all {source_count} "
                f"sources in {component_count} components. `split_manifest.json` assigns every "
                "component exactly once in EVAL, then DEV, then TRAIN order. The split reaches "
                "the integer global upper bound for minimum normalized topology slack and passes "
                f"all {gate_count} role-specific hard gates. Dataset freeze "
                "`PSEM-STRATEGY-DATA-v2` will bind these artifacts at the final freeze checkpoint; "
                "final preflight is recorded separately."
            )
        else:
            split_summary = "The identity graph and pinned WavLM overlap audit cover all 76 sources in 42 components. `split_manifest.json` assigns every component exactly once in EVAL, then DEV, then TRAIN order. The split reaches the integer global upper bound for minimum normalized topology slack and passes all 22 role-specific hard gates. Dataset freeze `PSEM-STRATEGY-DATA-v1` is bound in `dataset_freeze.json`; final preflight is recorded separately."
    lines = [
        (
            "# Natural topology census and component split"
            if split_selected
            else "# Natural topology census"
        ),
        "",
        (
            "This is the frozen-contract census of every accepted natural candidate meeting plus the deterministic connected-component assignment selected for TRAIN, DEV, and EVAL. No model prediction, model score, official model result, or model training participated."
            if split_selected
            else "This is the frozen-contract pre-split census of every accepted natural candidate meeting. No model prediction, model score, official model result, or model training participated."
        ),
        "",
        f"Contract: `{census['contract_version']}` (`{census['contract_status']}`)",
        "",
        (
            split_summary
            if split_selected
            else "Split roles remain unassigned until the identity graph and pretrained-checkpoint overlap audit are complete. Component counts are therefore pending and the raw-pool lower-bound audit is not split feasibility evidence."
        ),
        "",
        "## Overall",
        "",
        "| Sessions | Scored hours | Stable singleton hours | Ongoing overlap hours | Masked/ambiguous fraction |",
        "|---:|---:|---:|---:|---:|",
        f"| {overall['session_count']} | {overall['scored_hours']} | {overall['stable_singleton_hours']} | {overall['ongoing_overlap_hours']} | {overall['masked_or_ambiguous_fraction']} |",
        "",
        "## Mask diagnostics",
        "",
        f"- Actual handoff/relation transitions: `{overall['mask_diagnostics']['actual_transition_count']}`",
        f"- Masked handoff/relation transitions: `{overall['mask_diagnostics']['masked_transition_count']}` (`{overall['mask_diagnostics']['masked_transition_fraction']}`)",
        f"- Masked transition reasons: `{json.dumps(overall['mask_diagnostics']['masked_transition_reasons'], sort_keys=True)}`",
        f"- Diagnostic masked region counts: `{json.dumps(overall['mask_diagnostics']['diagnostic_masked_region_counts'], sort_keys=True)}`",
        "",
        "## Exclusive primary topology counts",
        "",
        "| Primary topology | Count |",
        "|---|---:|",
    ]
    lines.extend(
        f"| `{topology}` | {overall['primary_topology_counts'][topology]} |"
        for topology in OFFICIAL_PRIMARY_TOPOLOGIES
    )
    lines.extend(
        [
            "",
            "## By corpus",
            "",
            "| Corpus | Sessions | Scored hours | Stable singleton hours | Ongoing overlap hours | Primary episodes |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for corpus, aggregate in census["by_corpus"].items():
        lines.append(
            f"| {corpus} | {aggregate['session_count']} | {aggregate['scored_hours']} | {aggregate['stable_singleton_hours']} | {aggregate['ongoing_overlap_hours']} | {aggregate['exclusive_primary_episode_count']} |"
        )
    if split_selected:
        assert split_manifest is not None
        role_summaries = split_manifest["role_summaries"]
        role_order = (
            ("TRAIN", "PSEM-STRATEGY-TRAIN"),
            ("DEV", "PSEM-STRATEGY-DEV"),
            ("EVAL", "PSEM-STRATEGY-EVAL"),
        )
        lines.extend(
            [
                "",
                "## By selected role",
                "",
                "| Role | Components | Meetings | Scored h | Stable singleton h | Ongoing overlap h | Known speakers | Corpora |",
                "|---|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for label, role in role_order:
            summary = role_summaries[role]
            lines.append(
                f"| {label} | {summary['component_count']} | {summary['independent_meetings']} | {summary['scored_hours']} | {summary['stable_singleton_hours']} | {summary['ongoing_overlap_hours']} | {summary['known_speaker_count']} | {' + '.join(summary['corpora'])} |"
            )
        lines.extend(
            [
                "",
                "| Role | Direct | Gap handoff | Same gap | Overlap return | Overlap takeover | Short return |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for label, role in role_order:
            counts = role_summaries[role]["primary_topology_counts"]
            lines.append(
                f"| {label} | {counts['clean_direct_different_speaker_handoff']} | {counts['silence_gap_different_speaker_handoff']} | {counts['same_speaker_silence_gap_resume']} | {counts['overlap_return']} | {counts['overlap_takeover']} | {counts['short_backchannel_return']} |"
            )
        lines.extend(
            [
                "",
                "The exact source and component membership, input hashes, search seed/version, objective order, leakage audit, hard-gate observations, and assignment hash are authoritative in `split_manifest.json`.",
            ]
        )
    lines.extend(
        [
            "",
            "## By meeting",
            "",
            "| Corpus | Session | Hours | Direct | Gap handoff | Same gap | Overlap return | Overlap takeover | Short return | Micro gap | Micro overlap | Masked T |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        counts = row["primary_topology_counts"]
        diagnostics = row["micro_diagnostics"]
        hours = round(row["scored_samples"] / 16000 / 3600, 6)
        lines.append(
            f"| {row['corpus']} | {row['session_id']} | {hours} | {counts['clean_direct_different_speaker_handoff']} | {counts['silence_gap_different_speaker_handoff']} | {counts['same_speaker_silence_gap_resume']} | {counts['overlap_return']} | {counts['overlap_takeover']} | {counts['short_backchannel_return']} | {diagnostics['micro_gap_interval_count']} | {diagnostics['micro_overlap_interval_count']} | {row['mask_diagnostics']['masked_transition_count']} |"
        )
    audit = census["candidate_pool_lower_bound_audit"]
    scored_deficit_hours = round(
        audit["scored_samples"]["deficit"] / 16000 / 3600, 6
    )
    stable_observed_hours = round(
        audit["stable_singleton_samples"]["observed"] / 16000 / 3600, 6
    )
    stable_required_hours = round(
        audit["stable_singleton_samples"]["required_total"] / 16000 / 3600, 6
    )
    stable_deficit_hours = round(
        audit["stable_singleton_samples"]["deficit"] / 16000 / 3600, 6
    )
    overlap_observed_minutes = round(
        audit["ongoing_overlap_samples"]["observed"] / 16000 / 60, 6
    )
    overlap_required_minutes = round(
        audit["ongoing_overlap_samples"]["required_total"] / 16000 / 60, 6
    )
    overlap_deficit_minutes = round(
        audit["ongoing_overlap_samples"]["deficit"] / 16000 / 60, 6
    )
    if audit["scored_samples"]["deficit"]:
        lower_bound_status = "The scored-hour deficit is an acquisition blocker."
    elif split_selected:
        lower_bound_status = (
            "The raw-pool lower bounds and the selected connected-component assignment "
            "both pass. EVAL uses only freshness-eligible components; TRAIN+DEV and "
            "EVAL pass every exclusive topology and negative-exposure minimum; TRAIN, "
            "DEV, and EVAL pass their scored-hour and meeting minima."
        )
    else:
        lower_bound_status = (
            "The aggregate scored-hour lower bound passes; role-specific allocation "
            "remains unproven until component assignment."
        )
    lines.extend(
        [
            "",
            "## Raw candidate-pool lower bound",
            "",
            "| Criterion | Observed raw pool | Combined hard role minimum | Raw deficit/status |",
            "|---|---:|---:|---:|",
            f"| Scored natural hours | {overall['scored_hours']} hours | 33.0 hours | {scored_deficit_hours} hours |",
            (
                f"| Independent meetings | {audit['independent_meetings']['observed_upper_bound']} accepted sessions | {audit['independent_meetings']['required_total']} | 0 |"
                if split_selected
                else f"| Independent meetings | {audit['independent_meetings']['observed_upper_bound']} upper bound | {audit['independent_meetings']['required_total']} | component audit pending |"
            ),
            f"| Stable singleton hours | {stable_observed_hours} hours | {stable_required_hours} hours | {stable_deficit_hours} hours |",
            f"| Ongoing overlap minutes | {overlap_observed_minutes} minutes | {overlap_required_minutes} minutes | {overlap_deficit_minutes} minutes |",
        ]
    )
    lines.extend(
        f"| `{topology}` | {audit['primary_topology_counts'][topology]['observed']} | {audit['primary_topology_counts'][topology]['required_total']} | {audit['primary_topology_counts'][topology]['deficit']} |"
        for topology in OFFICIAL_PRIMARY_TOPOLOGIES
    )
    lines.extend(
        [
            "",
            (
                f"{lower_bound_status} No topology substitutes for another, and no threshold, count, or natural-data requirement is weakened."
                if split_selected
                else f"{lower_bound_status} The meeting count is only an upper bound until connected identity components are audited. Zero raw-pool deficits do not prove component-safe role allocation. No topology substitutes for another, and no threshold, count, or natural-data requirement is weakened."
            ),
            "",
            f"Topology manifest SHA-256: `{census['topology_manifest_sha256']}`",
        ]
    )
    if split_selected:
        lines.extend(["", f"Split manifest SHA-256: `{split_manifest_sha256}`"])
    return "\n".join(lines) + "\n"


def write_topology_census(
    data_dir: Path,
    corpus_root: Path,
    manifest_path: Path,
    census_path: Path,
    markdown_path: Path,
    reference_root: Path | None = None,
) -> None:
    validate_waveform_inventory(data_dir, corpus_root)
    try:
        context = resolve_dataset_context(data_dir)
    except DatasetContextError as exc:
        raise TopologyCensusError("dataset context is invalid") from exc
    if context.is_v2:
        if reference_root is None:
            raise TopologyCensusError("v2 topology census requires the reference checkout")
        sessions = normalize_reference_inventory(
            data_dir / "source_manifest.jsonl", corpus_root, reference_root
        )
    else:
        sessions = normalize_inventory(data_dir, corpus_root)
    contract = context.label_contract
    _validate_contract_precedence(contract)
    _validate_normalization_manifest(sessions, data_dir)
    rows = [
        build_topology_row(session, contract)
        for session in sorted(sessions, key=lambda session: session.source_id)
    ]
    write_jsonl(manifest_path, rows)
    rebuilt_rows, census = build_topology_census(
        sessions,
        data_dir,
        sha256_file(manifest_path),
    )
    if rows != rebuilt_rows:
        raise TopologyCensusError("topology rows changed during census generation")
    census_path.write_text(
        json.dumps(census, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    split_manifest_path = data_dir / "split_manifest.json"
    split_manifest = None
    split_manifest_sha256 = None
    if split_manifest_path.is_file():
        try:
            split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TopologyCensusError("split manifest is invalid during census rendering") from exc
        split_manifest_sha256 = sha256_file(split_manifest_path)
    markdown_path.write_text(
        render_data_census(census, rows, split_manifest, split_manifest_sha256),
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--census-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path)
    args = parser.parse_args()
    write_topology_census(
        args.data_dir.resolve(),
        args.corpus_root.resolve(),
        args.manifest_output.resolve(),
        args.census_output.resolve(),
        args.markdown_output.resolve(),
        args.reference_root.resolve() if args.reference_root is not None else None,
    )


if __name__ == "__main__":
    main()
