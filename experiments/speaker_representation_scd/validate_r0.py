from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.speaker_representation_scd.provenance import load_json, sha256_file
from experiments.speaker_representation_scd.schemas import (
    AUTHORITY_SHA256,
    ROLES,
    validate_document,
)

DEFAULT_PATHS = {
    "r0_protocol": "configs/protocol/r0_protocol.json",
    "analysis_contract": "configs/protocol/analysis_contract.json",
    "compute_ceiling": "configs/protocol/compute_ceiling.json",
    "license_disposition": "configs/protocol/license_disposition.json",
    "source_ledger": "data/source_ledger.json",
    "split_contract": "data/split_contract.json",
    "confirmatory_access_policy": "data/confirmatory_access_policy.json",
    "model_registry": "models/registry.json",
}


@dataclass(frozen=True, slots=True)
class BundleValidationResult:
    valid: bool
    errors: tuple[str, ...]
    artifact_hashes: dict[str, str]
    neural_execution_allowed: bool
    neural_execution_blockers: tuple[str, ...]
    confirmatory_access_allowed: bool
    confirmatory_access_blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "artifact_hashes": self.artifact_hashes,
            "neural_execution_allowed": self.neural_execution_allowed,
            "neural_execution_blockers": list(self.neural_execution_blockers),
            "confirmatory_access_allowed": self.confirmatory_access_allowed,
            "confirmatory_access_blockers": list(self.confirmatory_access_blockers),
        }


def _execution_readiness(documents: dict[str, dict[str, Any]]) -> tuple[bool, tuple[str, ...]]:
    blockers: list[str] = []
    compute = documents["compute_ceiling"]
    state = compute["execution_state"]
    guard = compute["legacy_contention_guard"]
    smoke = compute["smoke_gate"]
    if not state["full_extraction_enabled"]:
        blockers.append("full extraction is disabled")
    if guard["release_evidence"] is None:
        blockers.append("legacy experiment release is not evidenced")
    if not smoke["forecast_approved"]:
        blockers.append("smoke forecast is not approved")
    if any(model["acquisition_status"] != "acquired_verified" for model in documents["model_registry"]["models"]):
        blockers.append("model artifacts are not acquired and verified")
    if any(source["acquisition_status"] != "existing_verified" for source in documents["source_ledger"]["sources"]):
        blockers.append("public source artifacts are not acquired and verified")
    if any(model["extraction_status"] != "ready" for model in documents["model_registry"]["models"]):
        blockers.append("one or more extractors are not ready")
    return not blockers, tuple(blockers)


def _confirmatory_readiness(
    documents: dict[str, dict[str, Any]], artifact_hashes: dict[str, str]
) -> tuple[bool, tuple[str, ...]]:
    return False, (
        "r0-1 policy is seal-only",
        "a verified D5 reader gate requires a reviewed protocol amendment",
    )


def load_bundle(root: Path) -> tuple[dict[str, dict[str, Any]], list[str], dict[str, str]]:
    documents: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    hashes: dict[str, str] = {}
    for role, relative in DEFAULT_PATHS.items():
        path = root / relative
        try:
            document = load_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{role}: cannot load {relative}: {exc}")
            continue
        documents[role] = document
        hashes[role] = sha256_file(path)
        errors.extend(validate_document(document, role))
    if set(documents) == ROLES:
        linked = documents["r0_protocol"]["contract_paths"]
        for role, relative in linked.items():
            if DEFAULT_PATHS.get(role) != relative:
                errors.append(f"r0_protocol.contract_paths.{role}: path mismatch")
        registry_ids = {model["model_id"] for model in documents["model_registry"]["models"]}
        license_ids = {model["model_id"] for model in documents["license_disposition"]["models"]}
        if registry_ids != license_ids:
            errors.append("model_registry/license_disposition model IDs differ")
        registry_licenses = {
            model["model_id"]: model["license_id"]
            for model in documents["model_registry"]["models"]
        }
        disposition_licenses = {
            model["model_id"]: model["license_id"]
            for model in documents["license_disposition"]["models"]
        }
        if registry_licenses != disposition_licenses:
            errors.append("model_registry/license_disposition license IDs differ")
        source_ids = {source["source_id"] for source in documents["source_ledger"]["sources"]}
        assigned_source_ids = {
            assignment["source_id"]
            for assignment in documents["split_contract"]["component_assignments"]
        }
        if not assigned_source_ids <= source_ids:
            errors.append("split_contract references unknown source IDs")
    return documents, errors, hashes


def validate_bundle(root: Path | None = None) -> BundleValidationResult:
    base = root or Path(__file__).resolve().parent
    documents, errors, hashes = load_bundle(base)
    authority_path = base.parents[1] / "experiments" / "speaker_representation_scd" / "EXPERIMENT_PLAN.en.md"
    if not authority_path.is_file() or sha256_file(authority_path) != AUTHORITY_SHA256:
        errors.append("authority file identity does not match the pinned SHA-256")
    if set(documents) == ROLES:
        neural_allowed, neural_blockers = _execution_readiness(documents)
        confirmatory_allowed, confirmatory_blockers = _confirmatory_readiness(documents, hashes)
    else:
        neural_allowed, neural_blockers = False, ("R0 contract bundle is incomplete",)
        confirmatory_allowed, confirmatory_blockers = False, ("R0 contract bundle is incomplete",)
    if errors:
        neural_allowed = False
        confirmatory_allowed = False
        neural_blockers = tuple(dict.fromkeys((*neural_blockers, "R0 contract validation failed")))
        confirmatory_blockers = tuple(dict.fromkeys((*confirmatory_blockers, "R0 contract validation failed")))
    return BundleValidationResult(
        valid=not errors,
        errors=tuple(errors),
        artifact_hashes=hashes,
        neural_execution_allowed=neural_allowed,
        neural_execution_blockers=neural_blockers,
        confirmatory_access_allowed=confirmatory_allowed,
        confirmatory_access_blockers=confirmatory_blockers,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args(argv)
    result = validate_bundle(args.root)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0 if result.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
