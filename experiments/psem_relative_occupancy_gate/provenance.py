from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.psem_relative_occupancy_gate.io_utils import (
    canonical_sha256,
    config,
    data_dir,
    load_json,
    load_jsonl,
    safe_child,
    sha256_file,
)

AUTHORITY_REF = "https://github.com/kapitalismho/PuriPuly-heart/issues/97"
AUTHORITY_PIN = "4a09449425c9990c47228dafa9d48ecefc6d703437d4f92b1d945fa4899b305c"
DATASET_ID = "PSEM-STRATEGY-DATA-v2"
FREEZE_FILE_SHA256 = "bc7e63bb201c2a33a9b2d69b2364fed8f03839278098f0bd175d6833b330a41e"
FREEZE_PAYLOAD_SHA256 = "f9f1882d0de08a4fcd19e63f1da7ae022f940420863be5bbfc14d1d2a7b0f95e"
REFERENCE_REPOSITORY = "https://github.com/nttcslab-sp/diar-forced-alignment"
REFERENCE_COMMIT = "9527b7c64846fb38316a610f32e9d3466bd6d8b7"
REFERENCE_TREE = "ef50279a5041eea70a9a77c4a03446e538cb90bf"
ROLE_SOURCE_COUNTS = {
    "PSEM-STRATEGY-TRAIN": 64,
    "PSEM-STRATEGY-DEV": 10,
    "PSEM-STRATEGY-EVAL": 19,
}
FROZEN_ARTIFACT_NAMES = {
    "DATA_CENSUS.md",
    "REFERENCE_MIGRATION.md",
    "annotation_manifest.jsonl",
    "evaluator_contract.json",
    "identity_components.json",
    "nonlexical_risk_inventory.json",
    "normalization_manifest.jsonl",
    "operational_label_contract.json",
    "prior_exposure_manifest.jsonl",
    "reference_artifact_receipt.json",
    "reference_integrity_report.json",
    "reference_migration.jsonl",
    "reference_migration_summary.json",
    "reference_provenance.json",
    "source_manifest.jsonl",
    "split_feasibility.json",
    "split_manifest.json",
    "topology_census.json",
    "topology_manifest.jsonl",
    "wavlm_pretraining_overlap.json",
}


class ProvenanceError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class FrozenDataset:
    root: Path
    freeze: dict[str, Any]
    split: dict[str, Any]
    sources: dict[str, dict[str, Any]]
    normalizations: dict[str, dict[str, Any]]
    assignments: dict[str, dict[str, Any]]

    def source_ids(self, role: str) -> tuple[str, ...]:
        return tuple(
            sorted(
                source_id
                for source_id, assignment in self.assignments.items()
                if assignment["role"] == role
            )
        )

    def summary(self) -> dict[str, Any]:
        return {
            "dataset_freeze_id": self.freeze["dataset_freeze_id"],
            "freeze_file_sha256": sha256_file(self.root / "dataset_freeze.json"),
            "freeze_payload_sha256": self.freeze["freeze_payload_sha256"],
            "artifact_sha256": self.freeze["artifact_sha256"],
            "source_count": len(self.sources),
            "role_source_counts": dict(ROLE_SOURCE_COUNTS),
            "source_ids_sha256": canonical_sha256(sorted(self.sources)),
            "dev_source_ids": list(self.source_ids("PSEM-STRATEGY-DEV")),
        }


def _rows_by_id(rows: list[dict[str, Any]], field: str, label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(field)
        if not isinstance(value, str) or not value or value in result:
            raise ProvenanceError(f"{label} identities are missing or duplicated")
        result[value] = row
    return result


def load_frozen_dataset() -> FrozenDataset:
    cfg = config()
    dataset_config = cfg.get("dataset")
    if not isinstance(dataset_config, dict):
        raise ProvenanceError("dataset config is missing")
    expected_config = {
        "id": DATASET_ID,
        "freeze_file_sha256": FREEZE_FILE_SHA256,
        "freeze_payload_sha256": FREEZE_PAYLOAD_SHA256,
        "reference_repository": REFERENCE_REPOSITORY,
        "reference_commit": REFERENCE_COMMIT,
        "sample_rate_hz": 16000,
        "source_count": 93,
        "role_source_counts": ROLE_SOURCE_COUNTS,
        "lifecycle_proxy": "gt_speech_non_speech",
        "product_vad_integration": "deferred",
    }
    for field, expected in expected_config.items():
        if dataset_config.get(field) != expected:
            raise ProvenanceError(f"dataset config pin mismatch: {field}")
    if cfg.get("authority") != {"ref": AUTHORITY_REF, "sha256": AUTHORITY_PIN}:
        raise ProvenanceError("issue authority pin mismatch")
    root = data_dir()
    freeze_path = root / "dataset_freeze.json"
    if freeze_path.is_symlink() or sha256_file(freeze_path) != FREEZE_FILE_SHA256:
        raise ProvenanceError("dataset freeze file identity mismatch")
    freeze = load_json(freeze_path)
    if not isinstance(freeze, dict):
        raise ProvenanceError("dataset freeze must be an object")
    payload = copy.deepcopy(freeze)
    observed_payload_sha = payload.pop("freeze_payload_sha256", None)
    if (
        observed_payload_sha != canonical_sha256(payload)
        or observed_payload_sha != FREEZE_PAYLOAD_SHA256
    ):
        raise ProvenanceError("dataset freeze payload identity mismatch")
    if (
        freeze.get("dataset_freeze_id") != DATASET_ID
        or freeze.get("freeze_status") != "frozen"
        or freeze.get("reference_binding", {}).get("repository") != REFERENCE_REPOSITORY
        or freeze.get("reference_binding", {}).get("commit") != REFERENCE_COMMIT
        or freeze.get("source_identity_binding", {}).get("source_count") != 93
        or freeze.get("source_identity_binding", {}).get("eval_source_count") != 19
    ):
        raise ProvenanceError("dataset freeze semantic binding mismatch")
    artifact_hashes = freeze.get("artifact_sha256")
    if not isinstance(artifact_hashes, dict) or set(artifact_hashes) != FROZEN_ARTIFACT_NAMES:
        raise ProvenanceError("dataset freeze artifact set mismatch")
    for name, expected_sha in sorted(artifact_hashes.items()):
        path = safe_child(root, name, f"dataset artifact {name}")
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_sha:
            raise ProvenanceError(f"dataset artifact identity mismatch: {name}")
    source_rows = load_jsonl(root / "source_manifest.jsonl")
    normalization_rows = load_jsonl(root / "normalization_manifest.jsonl")
    split = load_json(root / "split_manifest.json")
    if not isinstance(split, dict):
        raise ProvenanceError("split manifest must be an object")
    assignment_rows = split.get("assignments", {}).get("sources")
    component_rows = split.get("assignments", {}).get("components")
    if not isinstance(assignment_rows, list) or not isinstance(component_rows, list):
        raise ProvenanceError("split assignments are missing")
    sources = _rows_by_id(source_rows, "source_id", "source manifest")
    normalizations = _rows_by_id(
        normalization_rows,
        "source_id",
        "normalization manifest",
    )
    assignments = _rows_by_id(assignment_rows, "source_id", "split source")
    components = _rows_by_id(component_rows, "component_id", "split component")
    expected_ids = set(sources)
    if (
        len(expected_ids) != 93
        or set(normalizations) != expected_ids
        or set(assignments) != expected_ids
    ):
        raise ProvenanceError("frozen source identity sets differ")
    role_counts = Counter(str(row.get("role")) for row in assignments.values())
    if dict(role_counts) != ROLE_SOURCE_COUNTS:
        raise ProvenanceError("frozen role source counts differ")
    component_sources: dict[str, set[str]] = {}
    for component_id, component in components.items():
        source_ids = component.get("source_ids")
        if not isinstance(source_ids, list) or len(source_ids) != len(set(source_ids)):
            raise ProvenanceError(f"component source identities are invalid: {component_id}")
        component_sources[component_id] = {str(value) for value in source_ids}
    for source_id in sorted(expected_ids):
        source = sources[source_id]
        normalization = normalizations[source_id]
        assignment = assignments[source_id]
        component_id = assignment.get("component_id")
        identity_fields = (
            "corpus",
            "session_id",
            "waveform_sha256",
            "annotation_sha256",
        )
        if any(source.get(field) != assignment.get(field) for field in identity_fields):
            raise ProvenanceError(f"source/split identity mismatch: {source_id}")
        if (
            normalization.get("corpus") != source.get("corpus")
            or normalization.get("session_id") != source.get("session_id")
            or normalization.get("source_waveform_sha256") != source.get("waveform_sha256")
            or normalization.get("source_annotation_sha256") != source.get("annotation_sha256")
            or normalization.get("reference_sha256") != assignment.get("reference_sha256")
            or (
                source.get("reference_sha256") is not None
                and normalization.get("reference_sha256") != source.get("reference_sha256")
            )
            or normalization.get("source_record_sha256") != canonical_sha256(source)
        ):
            raise ProvenanceError(f"source/normalization identity mismatch: {source_id}")
        if (
            component_id not in components
            or source_id not in component_sources[str(component_id)]
            or components[str(component_id)].get("role") != assignment.get("role")
        ):
            raise ProvenanceError(f"source/component identity mismatch: {source_id}")
    assigned_component_ids = {
        source_id for values in component_sources.values() for source_id in values
    }
    if assigned_component_ids != expected_ids or sum(
        len(values) for values in component_sources.values()
    ) != len(expected_ids):
        raise ProvenanceError("component membership does not cover the frozen sources")
    return FrozenDataset(root, freeze, split, sources, normalizations, assignments)


def validate_corpus_waveforms(dataset: FrozenDataset, corpus: Path) -> dict[str, Any]:
    rows = []
    for source_id in sorted(dataset.sources):
        source = dataset.sources[source_id]
        waveform = safe_child(corpus, str(source["audio_ref"]), f"waveform {source_id}")
        expected_size = int(source["waveform_size_bytes"])
        expected_sha = str(source["waveform_sha256"])
        if (
            waveform.is_symlink()
            or not waveform.is_file()
            or waveform.stat().st_size != expected_size
            or sha256_file(waveform) != expected_sha
        ):
            raise ProvenanceError(f"waveform identity mismatch: {source_id}")
        rows.append(
            {
                "source_id": source_id,
                "audio_ref": source["audio_ref"],
                "size_bytes": expected_size,
                "sha256": expected_sha,
            }
        )
    return {
        "source_count": len(rows),
        "waveform_identities_sha256": canonical_sha256(rows),
    }
