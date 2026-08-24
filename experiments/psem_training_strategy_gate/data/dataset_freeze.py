from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from experiments.psem_training_strategy_gate.data.dataset_context import (
    DatasetContext,
    DatasetContextError,
    resolve_dataset_context,
)
from experiments.psem_training_strategy_gate.data.evaluator_contract import (
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
    HISTORICAL_CONFIGS,
    canonical_sha256,
    sha256_file,
)
from experiments.psem_training_strategy_gate.data.reference_normalization import (
    EXPECTED_INVENTORY_SHA256,
    ReferenceNormalizationError,
    load_nonlexical_inventory,
)
from experiments.psem_training_strategy_gate.data.split_assignment import (
    validate_checked_split_package,
)

AUTHORITY_REF = "https://github.com/kapitalismho/PuriPuly-heart/issues/77"
AUTHORITY_PIN = "5778025c8aca1ea1cb7cd8fc41645b520ca1f9f749155b5f5daada32e940b559"
DATASET_FREEZE_ID = "PSEM-STRATEGY-DATA-v1"
CONTRACT_VERSION = "psem-handoff-v0"
GENERATOR_VERSION = "1"
FROZEN_ARTIFACTS = (
    "DATASET_PLAN.md",
    "operational_label_contract.json",
    "AMENDMENTS.md",
    "source_manifest.jsonl",
    "prior_exposure_manifest.jsonl",
    "annotation_manifest.jsonl",
    "normalization_manifest.jsonl",
    "annotation_calibration.json",
    "ANNOTATION_CALIBRATION.md",
    "topology_manifest.jsonl",
    "topology_census.json",
    "identity_components.json",
    "wavlm_pretraining_overlap.json",
    "split_manifest.json",
    "split_feasibility.json",
    "DATA_CENSUS.md",
)
V2_FROZEN_ARTIFACTS = (
    "operational_label_contract.json",
    "source_manifest.jsonl",
    "prior_exposure_manifest.jsonl",
    "annotation_manifest.jsonl",
    "normalization_manifest.jsonl",
    "nonlexical_risk_inventory.json",
    "topology_manifest.jsonl",
    "topology_census.json",
    "identity_components.json",
    "wavlm_pretraining_overlap.json",
    "split_manifest.json",
    "split_feasibility.json",
    "DATA_CENSUS.md",
    "evaluator_contract.json",
    "reference_artifact_receipt.json",
    "reference_integrity_report.json",
    "reference_migration.jsonl",
    "reference_migration_summary.json",
    "REFERENCE_MIGRATION.md",
    "reference_provenance.json",
)
MODEL_REGISTRY_PATH = "experiments/speaker_representation_scd/models/registry.json"
MODEL_SOURCE_REGISTRY_PATH = "experiments/speaker_representation_scd/models/source_registry.json"
REPOSITORY_INPUTS = tuple(
    sorted(
        {
            MODEL_REGISTRY_PATH,
            MODEL_SOURCE_REGISTRY_PATH,
            *HISTORICAL_CONFIGS.values(),
        }
    )
)
V2_PIPELINE_INPUTS = (
    "experiments/psem_training_strategy_gate/data/alimeeting_train_materialization.json",
    "experiments/psem_training_strategy_gate/data/alimeeting_train_materialization.py",
    "experiments/psem_training_strategy_gate/data/alimeeting_train_selection.json",
    "experiments/psem_training_strategy_gate/data/alimeeting_train_selection.py",
    "experiments/psem_training_strategy_gate/data/annotation_normalization.py",
    "experiments/psem_training_strategy_gate/data/dataset_context.py",
    "experiments/psem_training_strategy_gate/data/dataset_freeze.py",
    "experiments/psem_training_strategy_gate/data/dataset_preflight.py",
    "experiments/psem_training_strategy_gate/data/evaluator_contract.py",
    "experiments/psem_training_strategy_gate/data/forced_alignment_reference.py",
    "experiments/psem_training_strategy_gate/data/identity_components.py",
    "experiments/psem_training_strategy_gate/data/label_contract.py",
    "experiments/psem_training_strategy_gate/data/pretraining_overlap.py",
    "experiments/psem_training_strategy_gate/data/provenance.py",
    "experiments/psem_training_strategy_gate/data/reference_migration.py",
    "experiments/psem_training_strategy_gate/data/reference_normalization.py",
    "experiments/psem_training_strategy_gate/data/split_assignment.py",
    "experiments/psem_training_strategy_gate/data/split_feasibility.py",
    "experiments/psem_training_strategy_gate/data/topology_census.py",
)
V2_REPOSITORY_INPUTS = tuple(
    sorted({*REPOSITORY_INPUTS, *V2_PIPELINE_INPUTS, "pyproject.toml", "uv.lock"})
)
V2_INHERITED_ARTIFACTS = (
    "experiments/psem_training_strategy_gate/data/annotation_calibration.json",
    "experiments/psem_training_strategy_gate/data/ANNOTATION_CALIBRATION.md",
)
SEARCH_INPUT_ARTIFACTS = (
    "operational_label_contract.json",
    "annotation_calibration.json",
    "ANNOTATION_CALIBRATION.md",
    "source_manifest.jsonl",
    "prior_exposure_manifest.jsonl",
    "annotation_manifest.jsonl",
    "normalization_manifest.jsonl",
    "topology_manifest.jsonl",
    "topology_census.json",
    "identity_components.json",
    "wavlm_pretraining_overlap.json",
)
NO_MODEL_FIELDS = (
    "model_predictions_consulted",
    "model_scores_consulted",
    "official_model_results_inspected",
    "official_model_training_performed",
)
EXPECTED_REFERENCE_INTEGRITY_CHECK_IDS = frozenset(
    {
        "canonical_timeline_deterministic",
        "checked_in_v2_normalization_matches_regeneration",
        "complete_expected_unique_source_inventory",
        "declared_source_tail_receipts_exact",
        "every_reference_hash_present_and_bound",
        "every_reference_identity_unique",
        "exact_upstream_revision",
        "migration_rebuild_deterministic",
        "only_accepted_ami_terminal_tail_clips",
        "rttm_timing_validated",
        "selected_train_rttm_clipping_zero",
        "source_licenses_present_and_bound",
        "speaker_mapping_fail_closed",
        "topology_generation_deterministic",
        "unknown_nonlexical_class_fail_closed",
    }
)
SELECTION_MODEL_EXCLUSION_FIELDS = (
    "audio_features_used",
    "issue_76_outcomes_used",
    "model_scores_used",
    "psem_topology_counts_used",
    "vad_or_diarization_predictions_used",
)
EXPECTED_V2_PREFLIGHT_CHECK_IDS = (
    "natural_hours.train",
    "independent_meetings.train",
    "corpus_balance.train.source_count.ami",
    "corpus_balance.train.source_count.alimeeting",
    "corpus_balance.train.scored_samples.ami_present",
    "corpus_balance.train.scored_hours.alimeeting",
    "corpus_balance.train.maximum_corpus_scored_share",
    "natural_hours.dev",
    "independent_meetings.dev",
    "corpus_balance.dev.source_count.ami",
    "corpus_balance.dev.source_count.alimeeting",
    "corpus_balance.dev.scored_hours.ami",
    "corpus_balance.dev.scored_hours.alimeeting",
    "corpus_balance.dev.maximum_corpus_scored_share",
    "natural_hours.eval",
    "independent_meetings.eval",
    "corpus_balance.eval.source_count.ami",
    "corpus_balance.eval.source_count.alimeeting",
    "corpus_balance.eval.scored_hours.ami",
    "corpus_balance.eval.scored_hours.alimeeting",
    "corpus_balance.eval.maximum_corpus_scored_share",
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
    "reference.integrity_and_provenance_current",
    "evaluator.shared_threshold_contract_current",
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


class DatasetFreezeError(RuntimeError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetFreezeError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise DatasetFreezeError(f"JSON artifact must be an object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetFreezeError(f"invalid JSONL artifact: {path}") from exc
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise DatasetFreezeError(f"JSONL artifact must contain objects: {path}")
    return rows


def _require_model_policy_false(policy: Any, context: str) -> None:
    if not isinstance(policy, dict) or any(
        policy.get(field) is not False for field in NO_MODEL_FIELDS
    ):
        raise DatasetFreezeError(f"model exclusion policy is not frozen for {context}")


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


def _require_hash_bindings(
    observed: Any,
    expected: dict[str, str],
    context: str,
) -> None:
    if not isinstance(observed, dict) or any(
        observed.get(field) != value for field, value in expected.items()
    ):
        raise DatasetFreezeError(f"artifact hash binding is stale for {context}")


def _validate_contract(contract: dict[str, Any]) -> None:
    authority = contract.get("authority")
    if (
        contract.get("schema_version") != 1
        or contract.get("contract_version") != CONTRACT_VERSION
        or contract.get("status") != "frozen_after_annotation_only_calibration"
        or not isinstance(authority, dict)
        or authority.get("dataset_issue") != AUTHORITY_REF
        or authority.get("dataset_issue_sha256") != AUTHORITY_PIN
    ):
        raise DatasetFreezeError("operational label contract is not the pinned frozen contract")


def _validate_split(
    split: dict[str, Any],
    feasibility: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    assignments = split.get("assignments")
    hard_gates = split.get("hard_gate_results")
    leakage = split.get("leakage_audit")
    if (
        split.get("schema_version") != 1
        or split.get("artifact_role") != "psem_component_split_assignment"
        or split.get("authority_ref") != AUTHORITY_REF
        or split.get("authority_pin") != AUTHORITY_PIN
        or split.get("contract_version") != CONTRACT_VERSION
        or split.get("hard_gate_status") != "pass"
        or split.get("natural_data_only") is not True
        or not isinstance(hard_gates, list)
        or len(hard_gates) != 22
        or any(not isinstance(gate, dict) or gate.get("passed") is not True for gate in hard_gates)
        or not isinstance(assignments, dict)
        or not isinstance(assignments.get("components"), list)
        or not isinstance(assignments.get("sources"), list)
        or not isinstance(leakage, dict)
        or leakage.get("exact_source_coverage") is not True
        or any(
            leakage.get(field) is not False
            for field in (
                "component_may_span_roles",
                "exact_known_wavlm_pretraining_overlap_in_eval",
                "known_speaker_may_span_roles",
                "meeting_session_may_span_roles",
                "prior_selection_exposed_component_in_eval",
                "waveform_may_span_roles",
            )
        )
    ):
        raise DatasetFreezeError("split assignment is not frozen at every hard gate")
    _require_model_policy_false(split.get("model_policy"), "split manifest")
    components = assignments["components"]
    sources = assignments["sources"]
    official_roles = {
        "PSEM-STRATEGY-TRAIN",
        "PSEM-STRATEGY-DEV",
        "PSEM-STRATEGY-EVAL",
    }
    component_by_id = {row.get("component_id"): row for row in components if isinstance(row, dict)}
    component_roles = {
        component_id: row.get("role") for component_id, row in component_by_id.items()
    }
    component_source_ids = [
        source_id
        for row in components
        if isinstance(row, dict) and isinstance(row.get("source_ids"), list)
        for source_id in row["source_ids"]
    ]
    source_by_id = {row.get("source_id"): row for row in sources if isinstance(row, dict)}
    if (
        len(components) != 42
        or len(component_by_id) != len(components)
        or any(
            not isinstance(component_id, str)
            or not component_id
            or row.get("role") not in official_roles
            or not isinstance(row.get("eval_eligible"), bool)
            or not isinstance(row.get("source_ids"), list)
            or not row["source_ids"]
            or len(row["source_ids"]) != len(set(row["source_ids"]))
            for component_id, row in component_by_id.items()
        )
        or len(component_source_ids) != len(set(component_source_ids))
        or len(sources) != 76
        or len(source_by_id) != len(sources)
        or set(component_source_ids) != set(source_by_id)
        or any(
            not isinstance(source_id, str)
            or not source_id
            or row.get("component_id") not in component_by_id
            or source_id not in component_by_id[row["component_id"]]["source_ids"]
            or row.get("role") != component_by_id[row["component_id"]]["role"]
            or (
                row.get("role") == "PSEM-STRATEGY-EVAL"
                and component_by_id[row["component_id"]].get("eval_eligible") is not True
            )
            for source_id, row in source_by_id.items()
        )
    ):
        raise DatasetFreezeError("split assignment coverage is incomplete")
    if (
        feasibility.get("artifact_role") != "psem_split_feasibility"
        or feasibility.get("authority_ref") != AUTHORITY_REF
        or feasibility.get("authority_pin") != AUTHORITY_PIN
        or feasibility.get("valid_assignment_exists") is not True
        or feasibility.get("assignment_manifest_emitted") is not True
        or feasibility.get("hard_gate_status") != "pass"
        or feasibility.get("blocking_lower_bounds") != []
        or feasibility.get("assignments") != component_roles
        or feasibility.get("split_manifest_canonical_sha256") != canonical_sha256(split)
    ):
        raise DatasetFreezeError("resolved split feasibility is not bound to the assignment")
    _require_model_policy_false(feasibility.get("model_policy"), "split feasibility")
    return components, sources


def _validate_source_bindings(
    source_rows: list[dict[str, Any]],
    annotation_rows: list[dict[str, Any]],
    split_sources: list[dict[str, Any]],
    split_components: list[dict[str, Any]],
) -> dict[str, Any]:
    source_by_id = {row.get("source_id"): row for row in source_rows}
    annotation_by_id = {row.get("source_id"): row for row in annotation_rows}
    split_by_id = {row.get("source_id"): row for row in split_sources}
    component_by_id = {row.get("component_id"): row for row in split_components}
    if (
        len(source_rows) != 76
        or len(annotation_rows) != 76
        or len(split_sources) != 76
        or len(source_by_id) != 76
        or len(annotation_by_id) != 76
        or len(split_by_id) != 76
        or set(source_by_id) != set(annotation_by_id)
        or set(source_by_id) != set(split_by_id)
    ):
        raise DatasetFreezeError("source, annotation, and split coverage is not exact")
    identities = []
    for source_id in sorted(source_by_id):
        source = source_by_id[source_id]
        annotation = annotation_by_id[source_id]
        split_source = split_by_id[source_id]
        if (
            source.get("contract_version") != CONTRACT_VERSION
            or annotation.get("contract_version") != CONTRACT_VERSION
            or source.get("waveform_sha256") != split_source.get("waveform_sha256")
            or source.get("annotation_sha256") != annotation.get("annotation_sha256")
            or source.get("annotation_sha256") != split_source.get("annotation_sha256")
            or not _sha256_string(source.get("waveform_sha256"))
            or not _sha256_string(source.get("annotation_sha256"))
            or split_source.get("role")
            not in {
                "PSEM-STRATEGY-TRAIN",
                "PSEM-STRATEGY-DEV",
                "PSEM-STRATEGY-EVAL",
            }
        ):
            raise DatasetFreezeError(f"frozen source identity mismatch for {source_id}")
        identities.append(
            {
                "source_id": source_id,
                "waveform_sha256": source["waveform_sha256"],
                "annotation_sha256": source["annotation_sha256"],
                "component_id": split_source["component_id"],
                "role": split_source["role"],
                "final_eval_eligible": (
                    component_by_id[split_source["component_id"]]["eval_eligible"]
                    if split_source["role"] == "PSEM-STRATEGY-EVAL"
                    else None
                ),
            }
        )
    eval_rows = [row for row in identities if row["role"] == "PSEM-STRATEGY-EVAL"]
    return {
        "source_count": len(identities),
        "eval_source_count": len(eval_rows),
        "eval_sources_finally_eligible": all(
            row["final_eval_eligible"] is True for row in eval_rows
        ),
        "source_manifest_eval_eligibility_scope": "acquisition_stage_not_final_role_authority",
        "final_role_and_eval_eligibility_authority": "split_manifest.json",
        "source_ids_sha256": canonical_sha256([row["source_id"] for row in identities]),
        "waveform_identities_sha256": canonical_sha256(
            [
                {"source_id": row["source_id"], "waveform_sha256": row["waveform_sha256"]}
                for row in identities
            ]
        ),
        "annotation_identities_sha256": canonical_sha256(
            [
                {"source_id": row["source_id"], "annotation_sha256": row["annotation_sha256"]}
                for row in identities
            ]
        ),
        "split_source_identities_sha256": canonical_sha256(identities),
    }


def _role_summary(split: dict[str, Any]) -> dict[str, Any]:
    summaries = split.get("role_summaries")
    roles = split.get("official_roles")
    if (
        not isinstance(summaries, dict)
        or not isinstance(roles, list)
        or set(summaries) != set(roles)
    ):
        raise DatasetFreezeError("split role summaries are incomplete")
    return {
        role: {
            "component_count": summaries[role]["component_count"],
            "independent_meetings": summaries[role]["independent_meetings"],
            "scored_samples": summaries[role]["scored_samples"],
            "scored_hours": summaries[role]["scored_hours"],
            "stable_singleton_samples": summaries[role]["stable_singleton_samples"],
            "ongoing_overlap_samples": summaries[role]["ongoing_overlap_samples"],
            "primary_topology_counts": summaries[role]["primary_topology_counts"],
        }
        for role in roles
    }


def _validate_cross_artifact_bindings(
    artifact_hashes: dict[str, str],
    repository_input_hashes: dict[str, str],
    topology_census: dict[str, Any],
    identity_components: dict[str, Any],
    overlap: dict[str, Any],
    split: dict[str, Any],
    feasibility: dict[str, Any],
    source_ids_sha256: str,
) -> None:
    _require_hash_bindings(
        topology_census.get("input_manifests"),
        {
            "annotation_calibration_markdown_sha256": artifact_hashes["ANNOTATION_CALIBRATION.md"],
            "annotation_calibration_sha256": artifact_hashes["annotation_calibration.json"],
            "annotation_manifest_sha256": artifact_hashes["annotation_manifest.jsonl"],
            "normalization_manifest_sha256": artifact_hashes["normalization_manifest.jsonl"],
            "source_manifest_sha256": artifact_hashes["source_manifest.jsonl"],
            "source_ids_sha256": source_ids_sha256,
        },
        "topology census",
    )
    if (
        topology_census.get("topology_manifest_sha256")
        != artifact_hashes["topology_manifest.jsonl"]
    ):
        raise DatasetFreezeError("topology manifest hash binding is stale for topology census")
    _require_hash_bindings(
        identity_components.get("input_artifacts"),
        {
            "census_annotation_calibration_markdown_sha256": artifact_hashes[
                "ANNOTATION_CALIBRATION.md"
            ],
            "census_annotation_calibration_sha256": artifact_hashes["annotation_calibration.json"],
            "prior_exposure_manifest_sha256": artifact_hashes["prior_exposure_manifest.jsonl"],
            "source_manifest_sha256": artifact_hashes["source_manifest.jsonl"],
            "source_ids_sha256": source_ids_sha256,
            "topology_census_sha256": artifact_hashes["topology_census.json"],
            "topology_manifest_sha256": artifact_hashes["topology_manifest.jsonl"],
        },
        "identity components",
    )
    _require_hash_bindings(
        overlap.get("input_artifacts"),
        {
            "identity_components_sha256": artifact_hashes["identity_components.json"],
            "model_registry_sha256": repository_input_hashes[MODEL_REGISTRY_PATH],
            "model_source_registry_sha256": repository_input_hashes[MODEL_SOURCE_REGISTRY_PATH],
            "source_manifest_sha256": artifact_hashes["source_manifest.jsonl"],
            "source_ids_sha256": source_ids_sha256,
        },
        "WavLM overlap audit",
    )
    split_input_bindings = {
        "annotation_manifest_sha256": "annotation_manifest.jsonl",
        "identity_components_sha256": "identity_components.json",
        "normalization_manifest_sha256": "normalization_manifest.jsonl",
        "prior_exposure_manifest_sha256": "prior_exposure_manifest.jsonl",
        "source_manifest_sha256": "source_manifest.jsonl",
        "topology_census_sha256": "topology_census.json",
        "topology_manifest_sha256": "topology_manifest.jsonl",
        "wavlm_pretraining_overlap_sha256": "wavlm_pretraining_overlap.json",
    }
    _require_hash_bindings(
        split.get("input_artifacts"),
        {field: artifact_hashes[name] for field, name in split_input_bindings.items()},
        "split manifest",
    )
    _require_hash_bindings(
        feasibility.get("input_artifacts"),
        {
            "identity_components_sha256": artifact_hashes["identity_components.json"],
            "source_manifest_sha256": artifact_hashes["source_manifest.jsonl"],
            "topology_census_sha256": artifact_hashes["topology_census.json"],
            "wavlm_pretraining_overlap_sha256": artifact_hashes["wavlm_pretraining_overlap.json"],
            "source_ids_sha256": source_ids_sha256,
        },
        "split feasibility",
    )
    expected_search_fingerprint = canonical_sha256(
        {
            "data": {name: artifact_hashes[name] for name in SEARCH_INPUT_ARTIFACTS},
            "historical_prior_exposure_configs": {
                relative_path: repository_input_hashes[relative_path]
                for relative_path in HISTORICAL_CONFIGS.values()
            },
            "registry": repository_input_hashes[MODEL_REGISTRY_PATH],
            "source_registry": repository_input_hashes[MODEL_SOURCE_REGISTRY_PATH],
        }
    )
    if split.get("search", {}).get("input_fingerprint_sha256") != expected_search_fingerprint:
        raise DatasetFreezeError("split search input fingerprint is stale")


def _repository_root(context: DatasetContext) -> Path:
    return context.data_dir.parents[3] if context.is_v2 else context.data_dir.parents[2]


def _validate_v2_contract(contract: dict[str, Any], context: DatasetContext) -> None:
    authority = contract.get("authority")
    schema_version = contract.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != 1
        or canonical_sha256(contract) != context.label_contract.document_sha256
        or contract.get("contract_version") != context.label_contract.contract_version
        or contract.get("status") != context.label_contract.status
        or not isinstance(authority, dict)
        or authority.get("dataset_issue") != context.authority_ref
        or authority.get("dataset_issue_sha256") != context.authority_pin
    ):
        raise DatasetFreezeError("operational label contract is not the pinned v2 contract")


def _validate_v2_split(
    split: dict[str, Any],
    feasibility: dict[str, Any],
    context: DatasetContext,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    assignments = split.get("assignments")
    hard_gates = split.get("hard_gate_results")
    leakage = split.get("leakage_audit")
    false_leakage_fields = {
        "component_may_span_roles",
        "exact_known_wavlm_pretraining_overlap_in_eval",
        "known_speaker_may_span_roles",
        "meeting_session_may_span_roles",
        "prior_selection_exposed_component_in_eval",
        "waveform_may_span_roles",
    }
    if (
        split.get("schema_version") != 1
        or split.get("artifact_role") != "psem_component_split_assignment"
        or split.get("authority_ref") != context.authority_ref
        or split.get("authority_pin") != context.authority_pin
        or split.get("contract_version") != context.label_contract.contract_version
        or split.get("hard_gate_status") != "pass"
        or split.get("natural_data_only") is not True
        or not isinstance(hard_gates, list)
        or len(hard_gates) != 37
        or any(not isinstance(gate, dict) or gate.get("passed") is not True for gate in hard_gates)
        or not isinstance(assignments, dict)
        or not isinstance(assignments.get("components"), list)
        or not isinstance(assignments.get("sources"), list)
        or not isinstance(leakage, dict)
        or set(leakage) != {"exact_source_coverage", *false_leakage_fields}
        or leakage.get("exact_source_coverage") is not True
        or any(leakage.get(field) is not False for field in false_leakage_fields)
    ):
        raise DatasetFreezeError("v2 split assignment is not frozen at every hard gate")
    _require_model_policy_false(split.get("model_policy"), "v2 split manifest")
    components = assignments["components"]
    sources = assignments["sources"]
    official_roles = {
        "PSEM-STRATEGY-TRAIN",
        "PSEM-STRATEGY-DEV",
        "PSEM-STRATEGY-EVAL",
    }
    component_by_id = {row.get("component_id"): row for row in components if isinstance(row, dict)}
    component_roles = {
        component_id: row.get("role") for component_id, row in component_by_id.items()
    }
    component_source_ids = [
        source_id
        for row in components
        if isinstance(row, dict) and isinstance(row.get("source_ids"), list)
        for source_id in row["source_ids"]
    ]
    source_by_id = {row.get("source_id"): row for row in sources if isinstance(row, dict)}
    if (
        len(components) != 57
        or len(component_by_id) != len(components)
        or any(
            not isinstance(component_id, str)
            or not component_id
            or row.get("role") not in official_roles
            or not isinstance(row.get("eval_eligible"), bool)
            or not isinstance(row.get("source_ids"), list)
            or not row["source_ids"]
            or len(row["source_ids"]) != len(set(row["source_ids"]))
            for component_id, row in component_by_id.items()
        )
        or len(component_source_ids) != len(set(component_source_ids))
        or len(sources) != len(EXPECTED_V2_SOURCE_IDS)
        or len(source_by_id) != len(sources)
        or set(component_source_ids) != EXPECTED_V2_SOURCE_IDS
        or set(source_by_id) != EXPECTED_V2_SOURCE_IDS
        or any(
            not isinstance(source_id, str)
            or row.get("component_id") not in component_by_id
            or source_id not in component_by_id[row["component_id"]]["source_ids"]
            or row.get("role") != component_by_id[row["component_id"]]["role"]
            or not _sha256_string(row.get("reference_sha256"))
            or (
                row.get("role") == "PSEM-STRATEGY-EVAL"
                and component_by_id[row["component_id"]].get("eval_eligible") is not True
            )
            for source_id, row in source_by_id.items()
        )
    ):
        raise DatasetFreezeError("v2 split assignment coverage is incomplete")
    if (
        feasibility.get("artifact_role") != "psem_split_feasibility"
        or feasibility.get("authority_ref") != context.authority_ref
        or feasibility.get("authority_pin") != context.authority_pin
        or feasibility.get("contract_version") != context.label_contract.contract_version
        or feasibility.get("valid_assignment_exists") is not True
        or feasibility.get("assignment_manifest_emitted") is not True
        or feasibility.get("hard_gate_status") != "pass"
        or feasibility.get("blocking_lower_bounds") != []
        or feasibility.get("assignments") != component_roles
        or feasibility.get("split_manifest_canonical_sha256") != canonical_sha256(split)
    ):
        raise DatasetFreezeError("v2 split feasibility is not bound to the assignment")
    _require_model_policy_false(feasibility.get("model_policy"), "v2 split feasibility")
    return components, sources


def _validate_v2_source_bindings(
    context: DatasetContext,
    source_rows: list[dict[str, Any]],
    annotation_rows: list[dict[str, Any]],
    normalization_rows: list[dict[str, Any]],
    topology_rows: list[dict[str, Any]],
    split_sources: list[dict[str, Any]],
    split_components: list[dict[str, Any]],
) -> dict[str, Any]:
    inventories = [
        {row.get("source_id"): row for row in rows}
        for rows in (
            source_rows,
            annotation_rows,
            normalization_rows,
            topology_rows,
            split_sources,
        )
    ]
    source_by_id, annotation_by_id, normalization_by_id, topology_by_id, split_by_id = inventories
    if any(
        len(rows) != len(by_id)
        for rows, by_id in zip(
            (
                source_rows,
                annotation_rows,
                normalization_rows,
                topology_rows,
                split_sources,
            ),
            inventories,
            strict=True,
        )
    ) or any(set(by_id) != EXPECTED_V2_SOURCE_IDS for by_id in inventories):
        raise DatasetFreezeError("v2 source identity coverage is not exact")
    component_by_id = {row.get("component_id"): row for row in split_components}
    identities = []
    for source_id in sorted(EXPECTED_V2_SOURCE_IDS):
        source = source_by_id[source_id]
        annotation = annotation_by_id[source_id]
        normalization = normalization_by_id[source_id]
        topology = topology_by_id[source_id]
        split_source = split_by_id[source_id]
        waveform_sha256 = source.get("waveform_sha256")
        annotation_sha256 = source.get("annotation_sha256")
        reference_sha256 = normalization.get("reference_sha256")
        canonical_intervals_sha256 = normalization.get("canonical_intervals_sha256")
        label_result_sha256 = normalization.get("label_result_sha256")
        topology_episodes_sha256 = topology.get("topology_episodes_sha256")
        annotation_start = annotation.get("coverage_start_sample")
        annotation_end = annotation.get("coverage_end_sample")
        scored_start = normalization.get("scored_start_sample")
        scored_end = normalization.get("scored_end_sample")
        source_start = source.get("annotation_coverage_start_sample")
        source_end = source.get("annotation_coverage_end_sample")
        exposure = normalization.get("exposure")
        primary_counts = topology.get("primary_topology_counts")
        source_coverage_valid = (
            source_start is None
            and source_end is None
            or _nonnegative_int(source_start)
            and _positive_int(source_end)
            and source_start == annotation_start
            and source_end == annotation_end
        )
        if (
            any(
                row.get("schema_version") != 1
                or isinstance(row.get("schema_version"), bool)
                or not isinstance(row.get("schema_version"), int)
                for row in (source, annotation, normalization, topology)
            )
            or source.get("contract_version") != context.source_contract.contract_version
            or annotation.get("contract_version") != context.source_contract.contract_version
            or source.get("contract_document_sha256") != context.source_contract.document_sha256
            or annotation.get("contract_document_sha256") != context.source_contract.document_sha256
            or normalization.get("contract_version") != context.label_contract.contract_version
            or topology.get("contract_version") != context.label_contract.contract_version
            or normalization.get("contract_document_sha256")
            != context.label_contract.document_sha256
            or topology.get("contract_document_sha256") != context.label_contract.document_sha256
            or not _positive_int(source.get("sample_rate_hz"))
            or source.get("sample_rate_hz") != 16000
            or not _positive_int(source.get("duration_samples"))
            or not _nonnegative_int(annotation_start)
            or not _positive_int(annotation_end)
            or not _nonnegative_int(scored_start)
            or not _positive_int(scored_end)
            or not (
                0
                <= annotation_start
                <= scored_start
                < scored_end
                <= annotation_end
                <= source["duration_samples"]
            )
            or not source_coverage_valid
            or not isinstance(exposure, dict)
            or any(not _nonnegative_int(value) for value in exposure.values())
            or exposure.get("scored_samples") != scored_end - scored_start
            or not isinstance(primary_counts, dict)
            or any(not _nonnegative_int(value) for value in primary_counts.values())
            or not _nonnegative_int(topology.get("exclusive_primary_episode_count"))
            or sum(primary_counts.values()) != topology.get("exclusive_primary_episode_count")
            or not _nonnegative_int(topology.get("scored_start_sample"))
            or not _positive_int(topology.get("scored_end_sample"))
            or not _positive_int(topology.get("scored_samples"))
            or any(
                not _sha256_string(value)
                for value in (
                    waveform_sha256,
                    annotation_sha256,
                    reference_sha256,
                    canonical_intervals_sha256,
                    label_result_sha256,
                    topology_episodes_sha256,
                )
            )
            or normalization.get("source_record_sha256") != canonical_sha256(source)
            or waveform_sha256 != normalization.get("source_waveform_sha256")
            or waveform_sha256 != topology.get("source_waveform_sha256")
            or waveform_sha256 != split_source.get("waveform_sha256")
            or annotation_sha256 != annotation.get("annotation_sha256")
            or annotation_sha256 != normalization.get("source_annotation_sha256")
            or annotation_sha256 != topology.get("annotation_sha256")
            or annotation_sha256 != split_source.get("annotation_sha256")
            or source.get("reference_sha256") not in (None, reference_sha256)
            or reference_sha256 != split_source.get("reference_sha256")
            or label_result_sha256 != topology.get("label_result_sha256")
            or topology.get("normalization_row_sha256") != canonical_sha256(normalization)
            or topology.get("scored_start_sample") != normalization.get("scored_start_sample")
            or topology.get("scored_end_sample") != normalization.get("scored_end_sample")
            or topology.get("scored_samples") != exposure.get("scored_samples")
            or split_source.get("role")
            not in {
                "PSEM-STRATEGY-TRAIN",
                "PSEM-STRATEGY-DEV",
                "PSEM-STRATEGY-EVAL",
            }
        ):
            raise DatasetFreezeError(f"v2 frozen identity mismatch for {source_id}")
        identities.append(
            {
                "source_id": source_id,
                "waveform_sha256": waveform_sha256,
                "annotation_sha256": annotation_sha256,
                "reference_sha256": reference_sha256,
                "canonical_intervals_sha256": canonical_intervals_sha256,
                "label_result_sha256": label_result_sha256,
                "topology_episodes_sha256": topology_episodes_sha256,
                "component_id": split_source["component_id"],
                "role": split_source["role"],
                "final_eval_eligible": (
                    component_by_id[split_source["component_id"]]["eval_eligible"]
                    if split_source["role"] == "PSEM-STRATEGY-EVAL"
                    else None
                ),
            }
        )
    eval_rows = [row for row in identities if row["role"] == "PSEM-STRATEGY-EVAL"]
    return {
        "source_count": len(identities),
        "eval_source_count": len(eval_rows),
        "eval_sources_finally_eligible": all(
            row["final_eval_eligible"] is True for row in eval_rows
        ),
        "source_manifest_eval_eligibility_scope": "acquisition_stage_not_final_role_authority",
        "final_role_and_eval_eligibility_authority": "split_manifest.json",
        "source_ids_sha256": canonical_sha256([row["source_id"] for row in identities]),
        "waveform_identities_sha256": canonical_sha256(
            [
                {"source_id": row["source_id"], "waveform_sha256": row["waveform_sha256"]}
                for row in identities
            ]
        ),
        "annotation_identities_sha256": canonical_sha256(
            [
                {"source_id": row["source_id"], "annotation_sha256": row["annotation_sha256"]}
                for row in identities
            ]
        ),
        "reference_identities_sha256": canonical_sha256(
            [
                {"source_id": row["source_id"], "reference_sha256": row["reference_sha256"]}
                for row in identities
            ]
        ),
        "canonical_timeline_identities_sha256": canonical_sha256(
            [
                {
                    "source_id": row["source_id"],
                    "canonical_intervals_sha256": row["canonical_intervals_sha256"],
                    "label_result_sha256": row["label_result_sha256"],
                    "topology_episodes_sha256": row["topology_episodes_sha256"],
                }
                for row in identities
            ]
        ),
        "split_source_identities_sha256": canonical_sha256(identities),
    }


def _validate_v2_cross_artifact_bindings(
    context: DatasetContext,
    artifact_hashes: dict[str, str],
    inherited_hashes: dict[str, str],
    repository_input_hashes: dict[str, str],
    topology_census: dict[str, Any],
    identity_components: dict[str, Any],
    overlap: dict[str, Any],
    split: dict[str, Any],
    feasibility: dict[str, Any],
    evaluator: dict[str, Any],
    receipt: dict[str, Any],
    integrity: dict[str, Any],
    provenance: dict[str, Any],
    source_ids_sha256: str,
) -> None:
    calibration_json = "experiments/psem_training_strategy_gate/data/annotation_calibration.json"
    calibration_markdown = "experiments/psem_training_strategy_gate/data/ANNOTATION_CALIBRATION.md"
    _require_hash_bindings(
        topology_census.get("input_manifests"),
        {
            "annotation_calibration_markdown_sha256": inherited_hashes[calibration_markdown],
            "annotation_calibration_sha256": inherited_hashes[calibration_json],
            "annotation_manifest_sha256": artifact_hashes["annotation_manifest.jsonl"],
            "normalization_manifest_sha256": artifact_hashes["normalization_manifest.jsonl"],
            "source_manifest_sha256": artifact_hashes["source_manifest.jsonl"],
            "source_ids_sha256": source_ids_sha256,
        },
        "v2 topology census",
    )
    if (
        topology_census.get("topology_manifest_sha256")
        != artifact_hashes["topology_manifest.jsonl"]
    ):
        raise DatasetFreezeError("v2 topology manifest hash binding is stale")
    _require_hash_bindings(
        identity_components.get("input_artifacts"),
        {
            "census_annotation_calibration_markdown_sha256": inherited_hashes[calibration_markdown],
            "census_annotation_calibration_sha256": inherited_hashes[calibration_json],
            "prior_exposure_manifest_sha256": artifact_hashes["prior_exposure_manifest.jsonl"],
            "reference_artifact_receipt_sha256": artifact_hashes["reference_artifact_receipt.json"],
            "reference_integrity_report_sha256": artifact_hashes["reference_integrity_report.json"],
            "reference_provenance_sha256": artifact_hashes["reference_provenance.json"],
            "source_manifest_sha256": artifact_hashes["source_manifest.jsonl"],
            "source_ids_sha256": source_ids_sha256,
            "topology_census_sha256": artifact_hashes["topology_census.json"],
            "topology_manifest_sha256": artifact_hashes["topology_manifest.jsonl"],
        },
        "v2 identity components",
    )
    _require_hash_bindings(
        overlap.get("input_artifacts"),
        {
            "identity_components_sha256": artifact_hashes["identity_components.json"],
            "model_registry_sha256": repository_input_hashes[MODEL_REGISTRY_PATH],
            "model_source_registry_sha256": repository_input_hashes[MODEL_SOURCE_REGISTRY_PATH],
            "source_manifest_sha256": artifact_hashes["source_manifest.jsonl"],
            "source_ids_sha256": source_ids_sha256,
        },
        "v2 WavLM overlap audit",
    )
    split_input_bindings = {
        "annotation_manifest_sha256": "annotation_manifest.jsonl",
        "identity_components_sha256": "identity_components.json",
        "normalization_manifest_sha256": "normalization_manifest.jsonl",
        "prior_exposure_manifest_sha256": "prior_exposure_manifest.jsonl",
        "source_manifest_sha256": "source_manifest.jsonl",
        "topology_census_sha256": "topology_census.json",
        "topology_manifest_sha256": "topology_manifest.jsonl",
        "wavlm_pretraining_overlap_sha256": "wavlm_pretraining_overlap.json",
    }
    _require_hash_bindings(
        split.get("input_artifacts"),
        {field: artifact_hashes[name] for field, name in split_input_bindings.items()},
        "v2 split manifest",
    )
    _require_hash_bindings(
        feasibility.get("input_artifacts"),
        {
            "identity_components_sha256": artifact_hashes["identity_components.json"],
            "source_manifest_sha256": artifact_hashes["source_manifest.jsonl"],
            "source_ids_sha256": source_ids_sha256,
            "topology_census_sha256": artifact_hashes["topology_census.json"],
            "wavlm_pretraining_overlap_sha256": artifact_hashes["wavlm_pretraining_overlap.json"],
        },
        "v2 split feasibility",
    )
    expected_search_fingerprint = canonical_sha256(
        {
            "data": {
                **{
                    name: artifact_hashes[name]
                    for name in SEARCH_INPUT_ARTIFACTS
                    if name in artifact_hashes
                },
                "annotation_calibration.json": inherited_hashes[calibration_json],
                "ANNOTATION_CALIBRATION.md": inherited_hashes[calibration_markdown],
            },
            "historical_prior_exposure_configs": {
                relative_path: repository_input_hashes[relative_path]
                for relative_path in HISTORICAL_CONFIGS.values()
            },
            "registry": repository_input_hashes[MODEL_REGISTRY_PATH],
            "source_registry": repository_input_hashes[MODEL_SOURCE_REGISTRY_PATH],
        }
    )
    if split.get("search", {}).get("input_fingerprint_sha256") != expected_search_fingerprint:
        raise DatasetFreezeError("v2 split search input fingerprint is stale")
    repository_root = _repository_root(context)
    rebuilt_evaluator = build_evaluator_contract(
        context.data_dir,
        repository_root / MODEL_REGISTRY_PATH,
        repository_root / MODEL_SOURCE_REGISTRY_PATH,
    )
    if canonical_sha256(evaluator) != canonical_sha256(rebuilt_evaluator):
        raise DatasetFreezeError("v2 evaluator contract is not current")
    receipt_artifacts = {
        "REFERENCE_MIGRATION.md",
        "reference_integrity_report.json",
        "reference_migration.jsonl",
        "reference_migration_summary.json",
        "reference_provenance.json",
    }
    selection = _load_json(
        repository_root
        / "experiments/psem_training_strategy_gate/data/alimeeting_train_selection.json"
    )
    forced_alignment = selection.get("source_artifacts", {}).get("forced_alignment")
    selection_model_inputs = selection.get("selection_model_inputs")
    integrity_checks = integrity.get("checks")
    input_policy = integrity.get("input_policy")
    references = provenance.get("references")
    migration_rows = _load_jsonl(context.data_dir / "reference_migration.jsonl")
    migration_summary = _load_json(context.data_dir / "reference_migration_summary.json")
    inventory = _load_json(context.data_dir / "nonlexical_risk_inventory.json")
    normalization_rows = _load_jsonl(context.data_dir / "normalization_manifest.jsonl")
    try:
        load_nonlexical_inventory(context.data_dir / "nonlexical_risk_inventory.json")
    except ReferenceNormalizationError as exc:
        raise DatasetFreezeError("v2 nonlexical inventory is invalid") from exc
    if (
        receipt.get("schema_version") != 1
        or isinstance(receipt.get("schema_version"), bool)
        or receipt.get("artifact_role") != "reference_migration_artifact_receipt"
        or not _positive_int(receipt.get("source_count"))
        or receipt["source_count"] != len(EXPECTED_V2_SOURCE_IDS)
        or not isinstance(receipt.get("artifact_sha256"), dict)
        or set(receipt["artifact_sha256"]) != receipt_artifacts
        or any(
            receipt["artifact_sha256"].get(name) != artifact_hashes[name]
            for name in receipt_artifacts
        )
        or receipt.get("artifact_set_sha256") != canonical_sha256(receipt["artifact_sha256"])
        or integrity.get("schema_version") != 1
        or isinstance(integrity.get("schema_version"), bool)
        or integrity.get("artifact_role") != "reference_integrity_report"
        or integrity.get("scope")
        != "pipeline_correctness_not_independent_acoustic_boundary_accuracy"
        or integrity.get("status") != "pass"
        or not _positive_int(integrity.get("source_count"))
        or integrity["source_count"] != len(EXPECTED_V2_SOURCE_IDS)
        or not _positive_int(integrity.get("reference_count"))
        or integrity["reference_count"] != len(EXPECTED_V2_SOURCE_IDS)
        or not isinstance(integrity_checks, dict)
        or set(integrity_checks) != EXPECTED_REFERENCE_INTEGRITY_CHECK_IDS
        or any(value is not True for value in integrity_checks.values())
        or not isinstance(input_policy, dict)
        or input_policy.get("model_predictions_or_scores_accepted") is not False
        or input_policy.get("selection_receipt_model_inputs") != selection_model_inputs
        or not isinstance(selection_model_inputs, dict)
        or any(
            selection_model_inputs.get(field) is not False
            for field in SELECTION_MODEL_EXCLUSION_FIELDS
        )
        or integrity.get("migration_session_manifest_sha256") != canonical_sha256(migration_rows)
        or integrity.get("migration_summary_sha256") != canonical_sha256(migration_summary)
        or integrity.get("reference_provenance_sha256") != canonical_sha256(provenance)
        or integrity.get("reference_inventory_sha256")
        != provenance.get("reference_inventory_sha256")
        or provenance.get("schema_version") != 1
        or isinstance(provenance.get("schema_version"), bool)
        or provenance.get("artifact_role") != "reference_provenance"
        or provenance.get("reference_repository") != REFERENCE_REPOSITORY
        or provenance.get("reference_commit") != REFERENCE_COMMIT
        or not isinstance(forced_alignment, dict)
        or provenance.get("reference_git_tree") != forced_alignment.get("git_tree")
        or provenance.get("reference_license_ref") != forced_alignment.get("license_ref")
        or provenance.get("reference_license_sha256") != forced_alignment.get("license_sha256")
        or provenance.get("source_license_ids_by_corpus")
        != {"AMI": ["CC-BY-4.0"], "AliMeeting": ["CC-BY-SA-4.0"]}
        or provenance.get("v1_contract_document_sha256") != context.source_contract.document_sha256
        or provenance.get("v2_contract_document_sha256") != context.label_contract.document_sha256
        or provenance.get("nonlexical_inventory_sha256") != EXPECTED_INVENTORY_SHA256
        or canonical_sha256(inventory) != EXPECTED_INVENTORY_SHA256
        or any(
            row.get("nonlexical_inventory_sha256") != EXPECTED_INVENTORY_SHA256
            for row in normalization_rows
        )
        or not _positive_int(provenance.get("source_count"))
        or provenance["source_count"] != len(EXPECTED_V2_SOURCE_IDS)
        or not isinstance(references, list)
        or len(references) != len(EXPECTED_V2_SOURCE_IDS)
        or any(not isinstance(row, dict) for row in references)
        or {row.get("source_id") for row in references} != EXPECTED_V2_SOURCE_IDS
        or provenance.get("reference_inventory_sha256") != canonical_sha256(references)
        or provenance.get("migration_session_manifest_sha256") != canonical_sha256(migration_rows)
    ):
        raise DatasetFreezeError("v2 reference package is not frozen and complete")


def _build_v2_dataset_freeze_core(context: DatasetContext) -> dict[str, Any]:
    data_dir = context.data_dir
    missing = [name for name in V2_FROZEN_ARTIFACTS if not (data_dir / name).is_file()]
    if missing:
        raise DatasetFreezeError(f"required v2 freeze artifacts are missing: {', '.join(missing)}")
    contract = _load_json(data_dir / "operational_label_contract.json")
    split = _load_json(data_dir / "split_manifest.json")
    feasibility = _load_json(data_dir / "split_feasibility.json")
    topology_census = _load_json(data_dir / "topology_census.json")
    identity_components = _load_json(data_dir / "identity_components.json")
    overlap = _load_json(data_dir / "wavlm_pretraining_overlap.json")
    evaluator = _load_json(data_dir / "evaluator_contract.json")
    receipt = _load_json(data_dir / "reference_artifact_receipt.json")
    integrity = _load_json(data_dir / "reference_integrity_report.json")
    provenance = _load_json(data_dir / "reference_provenance.json")
    calibration = _load_json(context.calibration_dir / "annotation_calibration.json")
    source_rows = _load_jsonl(data_dir / "source_manifest.jsonl")
    annotation_rows = _load_jsonl(data_dir / "annotation_manifest.jsonl")
    normalization_rows = _load_jsonl(data_dir / "normalization_manifest.jsonl")
    topology_rows = _load_jsonl(data_dir / "topology_manifest.jsonl")
    _validate_v2_contract(contract, context)
    repository_root = _repository_root(context)
    try:
        rebuilt_split, rebuilt_feasibility = validate_checked_split_package(
            data_dir,
            repository_root / MODEL_REGISTRY_PATH,
            repository_root / MODEL_SOURCE_REGISTRY_PATH,
        )
    except (OSError, RuntimeError) as exc:
        raise DatasetFreezeError("checked v2 split package is not current") from exc
    if canonical_sha256(split) != canonical_sha256(rebuilt_split) or canonical_sha256(
        feasibility
    ) != canonical_sha256(rebuilt_feasibility):
        raise DatasetFreezeError("checked v2 split package is not canonical")
    components, split_sources = _validate_v2_split(split, feasibility, context)
    _require_model_policy_false(topology_census.get("model_policy"), "v2 topology census")
    _require_model_policy_false(overlap.get("model_policy"), "v2 WavLM overlap audit")
    _require_model_policy_false(calibration.get("input_policy"), "annotation calibration")
    identity_summary = identity_components.get("summary")
    if (
        not isinstance(identity_summary, dict)
        or identity_summary.get("component_count") != len(components)
        or identity_summary.get("source_count") != len(split_sources)
        or split.get("search", {}).get("model_derived_quantities_allowed") is not False
    ):
        raise DatasetFreezeError("v2 identity graph coverage does not match the split")
    artifact_hashes = {name: sha256_file(data_dir / name) for name in V2_FROZEN_ARTIFACTS}
    inherited_hashes = {
        name: sha256_file(repository_root / name) for name in V2_INHERITED_ARTIFACTS
    }
    repository_input_hashes = {
        name: sha256_file(repository_root / name) for name in V2_REPOSITORY_INPUTS
    }
    source_binding = _validate_v2_source_bindings(
        context,
        source_rows,
        annotation_rows,
        normalization_rows,
        topology_rows,
        split_sources,
        components,
    )
    _validate_v2_cross_artifact_bindings(
        context,
        artifact_hashes,
        inherited_hashes,
        repository_input_hashes,
        topology_census,
        identity_components,
        overlap,
        split,
        feasibility,
        evaluator,
        receipt,
        integrity,
        provenance,
        source_binding["source_ids_sha256"],
    )
    if source_binding["source_ids_sha256"] != split.get("input_artifacts", {}).get(
        "source_ids_sha256"
    ):
        raise DatasetFreezeError("v2 source identity hash does not match the split")
    payload = {
        "schema_version": 1,
        "artifact_role": "psem_dataset_freeze",
        "dataset_freeze_id": context.freeze_id,
        "freeze_status": "frozen",
        "authority_ref": context.authority_ref,
        "authority_pin": context.authority_pin,
        "contract_version": context.label_contract.contract_version,
        "contract_status": contract["status"],
        "official_roles": split["official_roles"],
        "selection_order": split["selection_order"],
        "source_identity_binding": source_binding,
        "role_summaries": _role_summary(split),
        "artifact_sha256": artifact_hashes,
        "inherited_artifact_sha256": inherited_hashes,
        "repository_input_sha256": repository_input_hashes,
        "reference_binding": {
            "repository": REFERENCE_REPOSITORY,
            "commit": REFERENCE_COMMIT,
            "reference_artifact_receipt_sha256": artifact_hashes["reference_artifact_receipt.json"],
            "reference_integrity_report_sha256": artifact_hashes["reference_integrity_report.json"],
            "reference_provenance_sha256": artifact_hashes["reference_provenance.json"],
            "reference_inventory_sha256": provenance["reference_inventory_sha256"],
            "integrity_status": integrity["status"],
        },
        "evaluator_binding": {
            "evaluator_contract_sha256": artifact_hashes["evaluator_contract.json"],
            "evaluator_contract_canonical_sha256": canonical_sha256(evaluator),
            "split_manifest_sha256": artifact_hashes["split_manifest.json"],
            "shared_threshold_vector_required": True,
            "per_corpus_thresholds_allowed": False,
        },
        "split_binding": {
            "assignment_sha256": split["assignment_sha256"],
            "chosen_assignment_sha256": split["search"]["chosen_assignment_sha256"],
            "input_fingerprint_sha256": split["search"]["input_fingerprint_sha256"],
            "split_manifest_file_sha256": artifact_hashes["split_manifest.json"],
            "split_manifest_canonical_sha256": canonical_sha256(split),
            "hard_gate_count": len(split["hard_gate_results"]),
            "hard_gate_status": split["hard_gate_status"],
        },
        "creation_provenance": {
            "generator": "experiments.psem_training_strategy_gate.data.dataset_freeze",
            "generator_version": GENERATOR_VERSION,
            "source": "repository-bound v2 forced-alignment references and accepted component split",
            "preflight_required": True,
        },
        "model_policy": {field: False for field in NO_MODEL_FIELDS},
    }
    return {**payload, "freeze_core_payload_sha256": canonical_sha256(payload)}


def _validate_v2_preflight_result(
    context: DatasetContext,
    freeze_core: dict[str, Any],
) -> dict[str, Any]:
    report_path = context.data_dir / "preflight_report.json"
    if not report_path.is_file():
        raise DatasetFreezeError("the passing v2 preflight result is missing")
    report = _load_json(report_path)
    payload = copy.deepcopy(report)
    preflight_payload_sha256 = payload.pop("preflight_payload_sha256", None)
    checks = report.get("checks")
    binding = report.get("freeze_binding")
    expected_binding = {
        "freeze_core_payload_sha256": freeze_core["freeze_core_payload_sha256"],
        "split_manifest_sha256": freeze_core["artifact_sha256"]["split_manifest.json"],
        "source_manifest_sha256": freeze_core["artifact_sha256"]["source_manifest.jsonl"],
        "annotation_manifest_sha256": freeze_core["artifact_sha256"]["annotation_manifest.jsonl"],
        "normalization_manifest_sha256": freeze_core["artifact_sha256"][
            "normalization_manifest.jsonl"
        ],
        "reference_artifact_receipt_sha256": freeze_core["reference_binding"][
            "reference_artifact_receipt_sha256"
        ],
        "evaluator_contract_sha256": freeze_core["evaluator_binding"]["evaluator_contract_sha256"],
    }
    if (
        report.get("schema_version") != 1
        or isinstance(report.get("schema_version"), bool)
        or report.get("artifact_role") != "psem_dataset_preflight"
        or report.get("generator")
        != "experiments.psem_training_strategy_gate.data.dataset_preflight"
        or report.get("generator_version") != "1"
        or report.get("dataset_freeze_id") != context.freeze_id
        or report.get("authority_ref") != context.authority_ref
        or report.get("authority_pin") != context.authority_pin
        or report.get("contract_version") != context.label_contract.contract_version
        or canonical_sha256(binding) != canonical_sha256(expected_binding)
        or not isinstance(checks, list)
        or len(checks) != len(EXPECTED_V2_PREFLIGHT_CHECK_IDS)
        or any(not isinstance(check, dict) for check in checks)
        or any(not isinstance(check.get("id"), str) or not check["id"] for check in checks)
        or tuple(check["id"] for check in checks) != EXPECTED_V2_PREFLIGHT_CHECK_IDS
        or any(check.get("passed") is not True for check in checks)
        or report.get("ready_for_issue_76") is not True
        or report.get("failed_checks") != []
        or not _sha256_string(preflight_payload_sha256)
        or preflight_payload_sha256 != canonical_sha256(payload)
    ):
        raise DatasetFreezeError("the v2 preflight result is not current and passing")
    return {
        "preflight_report_sha256": sha256_file(report_path),
        "preflight_report_canonical_sha256": canonical_sha256(report),
        "preflight_payload_sha256": preflight_payload_sha256,
        "freeze_core_payload_sha256": freeze_core["freeze_core_payload_sha256"],
        "check_count": len(checks),
        "ready_for_issue_76": True,
    }


def _build_v2_dataset_freeze(context: DatasetContext) -> dict[str, Any]:
    freeze_core = _build_v2_dataset_freeze_core(context)
    payload = {
        **freeze_core,
        "preflight_binding": _validate_v2_preflight_result(context, freeze_core),
    }
    return {**payload, "freeze_payload_sha256": canonical_sha256(payload)}


def build_v2_dataset_freeze_core(data_dir: Path) -> dict[str, Any]:
    try:
        context = resolve_dataset_context(data_dir)
    except DatasetContextError as exc:
        raise DatasetFreezeError("dataset context is invalid") from exc
    if not context.is_v2:
        raise DatasetFreezeError("the dataset is not the v2 freeze context")
    return _build_v2_dataset_freeze_core(context)


def build_dataset_freeze(data_dir: Path) -> dict[str, Any]:
    try:
        context = resolve_dataset_context(data_dir)
    except DatasetContextError as exc:
        raise DatasetFreezeError("dataset context is invalid") from exc
    if context.is_v2:
        return _build_v2_dataset_freeze(context)
    missing = [name for name in FROZEN_ARTIFACTS if not (data_dir / name).is_file()]
    if missing:
        raise DatasetFreezeError(f"required freeze artifacts are missing: {', '.join(missing)}")
    contract = _load_json(data_dir / "operational_label_contract.json")
    split = _load_json(data_dir / "split_manifest.json")
    feasibility = _load_json(data_dir / "split_feasibility.json")
    topology_census = _load_json(data_dir / "topology_census.json")
    identity_components = _load_json(data_dir / "identity_components.json")
    overlap = _load_json(data_dir / "wavlm_pretraining_overlap.json")
    calibration = _load_json(data_dir / "annotation_calibration.json")
    source_rows = _load_jsonl(data_dir / "source_manifest.jsonl")
    annotation_rows = _load_jsonl(data_dir / "annotation_manifest.jsonl")
    _validate_contract(contract)
    repository_root = data_dir.parents[2]
    registry_path = repository_root / MODEL_REGISTRY_PATH
    source_registry_path = repository_root / MODEL_SOURCE_REGISTRY_PATH
    try:
        rebuilt_split, rebuilt_feasibility = validate_checked_split_package(
            data_dir,
            registry_path,
            source_registry_path,
        )
    except (OSError, RuntimeError) as exc:
        raise DatasetFreezeError("checked split package is not current") from exc
    if split != rebuilt_split or feasibility != rebuilt_feasibility:
        raise DatasetFreezeError("checked split package does not match canonical validation")
    components, split_sources = _validate_split(split, feasibility)
    _require_model_policy_false(topology_census.get("model_policy"), "topology census")
    _require_model_policy_false(overlap.get("model_policy"), "WavLM overlap audit")
    _require_model_policy_false(calibration.get("input_policy"), "annotation calibration")
    identity_summary = identity_components.get("summary")
    if (
        not isinstance(identity_summary, dict)
        or identity_summary.get("component_count") != len(components)
        or identity_summary.get("source_count") != len(split_sources)
        or split.get("search", {}).get("model_derived_quantities_allowed") is not False
    ):
        raise DatasetFreezeError("identity graph coverage does not match the split")
    artifact_hashes = {name: sha256_file(data_dir / name) for name in FROZEN_ARTIFACTS}
    repository_input_hashes = {
        name: sha256_file(repository_root / name) for name in REPOSITORY_INPUTS
    }
    source_binding = _validate_source_bindings(
        source_rows,
        annotation_rows,
        split_sources,
        components,
    )
    _validate_cross_artifact_bindings(
        artifact_hashes,
        repository_input_hashes,
        topology_census,
        identity_components,
        overlap,
        split,
        feasibility,
        source_binding["source_ids_sha256"],
    )
    if source_binding["source_ids_sha256"] != split.get("input_artifacts", {}).get(
        "source_ids_sha256"
    ):
        raise DatasetFreezeError("source identity hash does not match the split input")
    payload = {
        "schema_version": 1,
        "artifact_role": "psem_dataset_freeze",
        "dataset_freeze_id": DATASET_FREEZE_ID,
        "freeze_status": "frozen",
        "authority_ref": AUTHORITY_REF,
        "authority_pin": AUTHORITY_PIN,
        "contract_version": CONTRACT_VERSION,
        "contract_status": contract["status"],
        "official_roles": split["official_roles"],
        "selection_order": split["selection_order"],
        "source_identity_binding": source_binding,
        "role_summaries": _role_summary(split),
        "artifact_sha256": artifact_hashes,
        "repository_input_sha256": repository_input_hashes,
        "split_binding": {
            "assignment_sha256": split["assignment_sha256"],
            "chosen_assignment_sha256": split["search"]["chosen_assignment_sha256"],
            "input_fingerprint_sha256": split["search"]["input_fingerprint_sha256"],
            "split_manifest_file_sha256": artifact_hashes["split_manifest.json"],
            "split_manifest_canonical_sha256": canonical_sha256(split),
            "hard_gate_count": len(split["hard_gate_results"]),
            "hard_gate_status": split["hard_gate_status"],
        },
        "creation_provenance": {
            "generator": "experiments.psem_training_strategy_gate.data.dataset_freeze",
            "generator_version": GENERATOR_VERSION,
            "source": "repository-bound natural meeting manifests and accepted component split",
            "preflight_required": True,
        },
        "model_policy": {field: False for field in NO_MODEL_FIELDS},
    }
    return {**payload, "freeze_payload_sha256": canonical_sha256(payload)}


def validate_checked_dataset_freeze(
    data_dir: Path,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    checked = _load_json(manifest_path or data_dir / "dataset_freeze.json")
    expected = build_dataset_freeze(data_dir)
    if canonical_sha256(checked) != canonical_sha256(expected):
        raise DatasetFreezeError("checked dataset freeze is not current")
    payload = copy.deepcopy(checked)
    observed_digest = payload.pop("freeze_payload_sha256", None)
    if observed_digest != canonical_sha256(payload):
        raise DatasetFreezeError("dataset freeze payload digest is invalid")
    return checked


def write_dataset_freeze(data_dir: Path, output_path: Path) -> None:
    value = build_dataset_freeze(data_dir)
    if output_path.is_file():
        existing = _load_json(output_path)
        if canonical_sha256(existing) != canonical_sha256(value):
            raise DatasetFreezeError(
                "existing dataset freeze is immutable; create a new freeze version"
            )
        return
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_dataset_freeze(args.data_dir.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
