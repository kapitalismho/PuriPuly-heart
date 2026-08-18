from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from experiments.psem_training_strategy_gate.data.provenance import (
    HISTORICAL_CONFIGS,
    canonical_sha256,
    sha256_file,
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
MODEL_REGISTRY_PATH = "experiments/speaker_representation_scd/models/registry.json"
MODEL_SOURCE_REGISTRY_PATH = (
    "experiments/speaker_representation_scd/models/source_registry.json"
)
REPOSITORY_INPUTS = tuple(
    sorted(
        {
            MODEL_REGISTRY_PATH,
            MODEL_SOURCE_REGISTRY_PATH,
            *HISTORICAL_CONFIGS.values(),
        }
    )
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
    if not isinstance(policy, dict) or any(policy.get(field) is not False for field in NO_MODEL_FIELDS):
        raise DatasetFreezeError(f"model exclusion policy is not frozen for {context}")


def _sha256_string(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


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
    component_by_id = {
        row.get("component_id"): row for row in components if isinstance(row, dict)
    }
    component_roles = {
        component_id: row.get("role") for component_id, row in component_by_id.items()
    }
    component_source_ids = [
        source_id
        for row in components
        if isinstance(row, dict) and isinstance(row.get("source_ids"), list)
        for source_id in row["source_ids"]
    ]
    source_by_id = {
        row.get("source_id"): row for row in sources if isinstance(row, dict)
    }
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
            or source_id
            not in component_by_id[row["component_id"]]["source_ids"]
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
    if not isinstance(summaries, dict) or not isinstance(roles, list) or set(summaries) != set(roles):
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
            "annotation_calibration_markdown_sha256": artifact_hashes[
                "ANNOTATION_CALIBRATION.md"
            ],
            "annotation_calibration_sha256": artifact_hashes["annotation_calibration.json"],
            "annotation_manifest_sha256": artifact_hashes["annotation_manifest.jsonl"],
            "normalization_manifest_sha256": artifact_hashes["normalization_manifest.jsonl"],
            "source_manifest_sha256": artifact_hashes["source_manifest.jsonl"],
            "source_ids_sha256": source_ids_sha256,
        },
        "topology census",
    )
    if topology_census.get("topology_manifest_sha256") != artifact_hashes[
        "topology_manifest.jsonl"
    ]:
        raise DatasetFreezeError("topology manifest hash binding is stale for topology census")
    _require_hash_bindings(
        identity_components.get("input_artifacts"),
        {
            "census_annotation_calibration_markdown_sha256": artifact_hashes[
                "ANNOTATION_CALIBRATION.md"
            ],
            "census_annotation_calibration_sha256": artifact_hashes[
                "annotation_calibration.json"
            ],
            "prior_exposure_manifest_sha256": artifact_hashes[
                "prior_exposure_manifest.jsonl"
            ],
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
            "model_source_registry_sha256": repository_input_hashes[
                MODEL_SOURCE_REGISTRY_PATH
            ],
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
        {
            field: artifact_hashes[name]
            for field, name in split_input_bindings.items()
        },
        "split manifest",
    )
    _require_hash_bindings(
        feasibility.get("input_artifacts"),
        {
            "identity_components_sha256": artifact_hashes["identity_components.json"],
            "source_manifest_sha256": artifact_hashes["source_manifest.jsonl"],
            "topology_census_sha256": artifact_hashes["topology_census.json"],
            "wavlm_pretraining_overlap_sha256": artifact_hashes[
                "wavlm_pretraining_overlap.json"
            ],
            "source_ids_sha256": source_ids_sha256,
        },
        "split feasibility",
    )
    expected_search_fingerprint = canonical_sha256(
        {
            "data": {
                name: artifact_hashes[name]
                for name in SEARCH_INPUT_ARTIFACTS
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
        raise DatasetFreezeError("split search input fingerprint is stale")


def build_dataset_freeze(data_dir: Path) -> dict[str, Any]:
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
    if checked != expected:
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
        if existing != value:
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
                json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
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
