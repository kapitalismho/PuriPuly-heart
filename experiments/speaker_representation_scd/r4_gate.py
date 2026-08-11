from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.speaker_representation_scd.execution_guard import strict_legacy_scan
from experiments.speaker_representation_scd.provenance import (
    canonical_json_bytes,
    load_json,
    self_sha256_valid,
    sha256_bytes,
    sha256_file,
)
from experiments.speaker_representation_scd.r1_forecast import (
    TECHNICAL_VALIDITY_PATH,
    validate_technical_validity,
)
from experiments.speaker_representation_scd.r1_gate import (
    EXPECTED_ENVIRONMENT,
    EXPERIMENT_ROOT,
    REPOSITORY_ROOT,
)
from experiments.speaker_representation_scd.r2l_gate import AUTHORITY

GATE_PATH = Path("configs/r4/legacy_common_gt_r4_continuous_gate.json")
EXPECTED_ACTIONS = {
    "r4_continuous": True,
    "r4_sensitivity": True,
    "r4_report": True,
    "cache_calibration": False,
    "full_extraction": False,
    "confirmatory_access": False,
    "training": False,
}
EXPECTED_STORAGE = {
    "cache_root_env": "SRSCD_CACHE_ROOT",
    "required_volume": "C:",
    "must_be_outside_repository": True,
    "min_free_gib_before_download": 55,
    "max_source_download_gib": 25,
    "max_derived_cache_gib": 20,
    "max_external_storage_gib": 50,
}
EXPECTED_SUPERVISION = {
    "shared_execution_lease": "control/r1_execution.lock",
    "entrypoint": "python -m experiments.speaker_representation_scd.r4_execute",
    "worker_environment": "environment_venv",
    "direct_worker_execution_allowed": False,
    "legacy_process_scan": "continuous_fail_closed_excluding_phase5_modules",
    "process_inspection_failure": "abort",
    "max_parallel_actions": 1,
    "max_worker_processes": 1,
    "max_cpu_threads": 8,
    "max_process_tree_resident_ram_gib": 24,
    "max_action_wall_hours": 24,
    "max_cumulative_wall_hours": 96,
    "hard_memory_boundary": "windows_job_object_job_memory",
    "action_receipt_authority": "unique_completed_usage_attestation",
    "orphan_receipt_policy": "quarantine_and_retry",
}
EXPECTED_R4 = {
    "primary_context_ms": 300,
    "primary_hop_ms": 100,
    "sensitivity_hop_ms": 50,
    "hop_samples": 1600,
    "sensitivity_hop_samples": 800,
    "stability_gate": 0.25,
    "prototype_k": 3,
    "minimum_event_separation_hops": 2,
    "match_tolerance_ms": 500,
    "false_event_budget_per_hour": 1.0,
    "thresholds": [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    "score_types": ["adjacent", "prototype"],
    "families": ["one_hop", "two_hop", "three_hop", "hysteresis"],
    "bounded_panel": {"maximum_source_hours": 6.0, "maximum_source_count": 80},
    "sensitivity": {"top_encoder_count": 2, "event_stratum_only": True},
}
EXPECTED_EXECUTION_PATHS = {
    "execution_guard.py",
    "provenance.py",
    "r1_forecast.py",
    "r1_gate.py",
    "r2l_gate.py",
    "r3_execute.py",
    "r3_gate.py",
    "r3_probe.py",
    "r4_continuous.py",
    "r4_execute.py",
    "r4_gate.py",
    "run_provenance.py",
    "validate_r4_gate.py",
    "windows_job.py",
}


@dataclass(frozen=True, slots=True)
class R4GateResult:
    valid: bool
    errors: tuple[str, ...]
    allowed_actions: dict[str, bool]
    artifact_hashes: dict[str, str]
    legacy_processes: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "allowed_actions": self.allowed_actions,
            "artifact_hashes": self.artifact_hashes,
            "legacy_processes": list(self.legacy_processes),
        }


class R4GateError(RuntimeError):
    pass


def _exact_keys(value: Any, expected: set[str], label: str, errors: list[str]) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        errors.append(f"{label}: keys differ")


def _validate_reference(
    base: Path,
    reference: Any,
    expected: dict[str, Any],
    label: str,
    errors: list[str],
    artifacts: dict[str, str],
) -> None:
    if reference != expected:
        errors.append(f"{label}: reference differs")
        return
    path = base / expected["path"]
    if not path.is_file() or sha256_file(path) != expected["sha256"]:
        errors.append(f"{label}: byte identity differs")
        return
    artifacts[label] = expected["sha256"]
    expected_self = expected.get("self_sha256")
    if expected_self is not None:
        try:
            document = load_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{label}: unreadable: {exc}")
            return
        if not self_sha256_valid(document) or document.get("self_sha256") != expected_self:
            errors.append(f"{label}: self identity differs")


def _cache_reference(
    cache_root: Path,
    reference: Any,
    expected: dict[str, Any],
    label: str,
    errors: list[str],
) -> None:
    if reference != expected:
        errors.append(f"{label}: reference differs")
        return
    path = cache_root / expected["relative_to_cache_root"]
    if not path.is_file() or sha256_file(path) != expected["sha256"]:
        errors.append(f"{label}: byte identity differs")
        return
    expected_self = expected.get("self_sha256")
    if expected_self is not None:
        try:
            document = load_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{label}: unreadable: {exc}")
            return
        if not self_sha256_valid(document) or document.get("self_sha256") != expected_self:
            errors.append(f"{label}: self identity differs")


def _execution_manifest_errors(base: Path, execution: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(execution, dict):
        return ["r4_gate.execution_code: invalid"]
    files = execution.get("files")
    if not isinstance(files, list) or not files:
        return ["r4_gate.execution_code.files: invalid"]
    observed_paths: set[str] = set()
    for index, row in enumerate(files):
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            errors.append(f"r4_gate.execution_code.files[{index}]: invalid")
            continue
        relative = row.get("path")
        expected_hash = row.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            errors.append(f"r4_gate.execution_code.files[{index}]: invalid")
            continue
        observed_paths.add(relative)
        target = base / relative
        if not target.is_file() or sha256_file(target) != expected_hash:
            errors.append(f"r4_gate.execution_code.files[{index}]: identity differs")
    if observed_paths != EXPECTED_EXECUTION_PATHS:
        errors.append("r4_gate.execution_code.files: inventory differs")
    if execution.get("manifest_sha256") != sha256_bytes(canonical_json_bytes(files)):
        errors.append("r4_gate.execution_code.manifest_sha256: invalid")
    return errors


def validate_r4_gate(
    root: Path | None = None,
    *,
    cache_root: Path | None = None,
    scan_processes: bool = True,
) -> R4GateResult:
    base = (root or EXPERIMENT_ROOT).resolve()
    errors: list[str] = []
    artifacts: dict[str, str] = {}
    try:
        gate = load_json(base / GATE_PATH)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        gate = {}
        errors.append(f"r4_gate: unreadable: {exc}")
    if gate:
        if not self_sha256_valid(gate):
            errors.append("r4_gate.self_sha256: invalid")
        _exact_keys(
            gate,
            {
                "schema_version",
                "artifact_role",
                "experiment_id",
                "protocol_version",
                "authority",
                "accepted_predecessors",
                "dependencies",
                "r4",
                "authorization",
                "storage",
                "supervision",
                "execution_code",
                "self_sha256",
            },
            "r4_gate",
            errors,
        )
        if gate.get("schema_version") != 1:
            errors.append("r4_gate.schema_version: unexpected")
        if gate.get("artifact_role") != "r4_legacy_common_gt_gate":
            errors.append("r4_gate.artifact_role: unexpected")
        if gate.get("experiment_id") != "speaker_representation_scd_v1":
            errors.append("r4_gate.experiment_id: unexpected")
        if gate.get("protocol_version") != "r4-legacy-common-gt-1":
            errors.append("r4_gate.protocol_version: unexpected")
        if gate.get("authority") != AUTHORITY:
            errors.append("r4_gate.authority: differs")
        dependencies = gate.get("dependencies")
        if not isinstance(dependencies, dict) or "environment" not in dependencies:
            errors.append("r4_gate.dependencies: inventory differs")
        else:
            environment = dependencies.get("environment")
            if environment != EXPECTED_ENVIRONMENT:
                errors.append("r4_gate.dependencies.environment: differs")
            for label in (
                "protocol_screen",
                "technical_validity",
                "forecast_contract",
                "source_registry",
            ):
                expected = dependencies.get(label)
                if not isinstance(expected, dict) or "path" not in expected:
                    errors.append(f"r4_gate.dependencies.{label}: invalid")
                    continue
                _validate_reference(
                    base,
                    dependencies.get(label),
                    expected,
                    label,
                    errors,
                    artifacts,
                )
        if gate.get("r4") != EXPECTED_R4:
            errors.append("r4_gate.r4: contract differs")
        if gate.get("authorization") != EXPECTED_ACTIONS:
            errors.append("r4_gate.authorization: differs")
        if gate.get("storage") != EXPECTED_STORAGE:
            errors.append("r4_gate.storage: differs")
        if gate.get("supervision") != EXPECTED_SUPERVISION:
            errors.append("r4_gate.supervision: differs")
        errors.extend(_execution_manifest_errors(base, gate.get("execution_code")))
    if cache_root is not None and not errors:
        try:
            technical = load_json(base / TECHNICAL_VALIDITY_PATH)
            errors.extend(validate_technical_validity(technical, cache_root.resolve()))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"technical_validity: unreadable: {exc}")
        cache_dependencies = gate.get("dependencies", {}).get("cache", {})
        for label, expected in cache_dependencies.items():
            if not isinstance(expected, dict):
                errors.append(f"r4_gate.dependencies.cache.{label}: invalid")
                continue
            _cache_reference(
                cache_root.resolve(),
                cache_dependencies.get(label),
                expected,
                label,
                errors,
            )
    matches: tuple[dict[str, Any], ...] = ()
    if scan_processes:
        matches, failures = strict_legacy_scan()
        if failures:
            errors.append(f"legacy process inspection failed: {failures}")
        if matches:
            errors.append("legacy contention: active speaker_turn_boundary process detected")
    return R4GateResult(
        valid=not errors,
        errors=tuple(errors),
        allowed_actions=dict(gate.get("authorization", {})) if gate else {},
        artifact_hashes=artifacts,
        legacy_processes=matches,
    )


def validated_r4_cache_root(action: str, root: Path | None = None) -> Path:
    value = os.environ.get("SRSCD_CACHE_ROOT")
    if not value:
        raise R4GateError("SRSCD_CACHE_ROOT is required")
    candidate = Path(value)
    if not candidate.is_absolute():
        raise R4GateError("SRSCD_CACHE_ROOT must be absolute")
    resolved = candidate.resolve()
    repository = (root or REPOSITORY_ROOT).resolve()
    if resolved == repository or repository in resolved.parents:
        raise R4GateError("SRSCD_CACHE_ROOT must be outside the repository")
    if resolved.drive.upper() != "C:":
        raise R4GateError("SRSCD_CACHE_ROOT must be on C:")
    result = validate_r4_gate(cache_root=resolved)
    if not result.valid:
        raise R4GateError("; ".join(result.errors))
    if result.allowed_actions.get(action) is not True:
        raise R4GateError(f"R4 action is not authorized: {action}")
    free_gib = shutil.disk_usage(resolved.anchor).free / 1024**3
    if free_gib < EXPECTED_STORAGE["min_free_gib_before_download"]:
        raise R4GateError(f"free disk is below 55 GiB: {free_gib:.3f}")
    return resolved
