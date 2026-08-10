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
from experiments.speaker_representation_scd.validate_r0 import validate_bundle

EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[1]
GATE_PATH = Path("configs/r1/acquisition_smoke_gate.json")
SOURCE_REGISTRY_PATH = Path("models/source_registry.json")
EXPECTED_SOURCE_REGISTRY_SELF_SHA256 = (
    "cc09d72b7f05e0375972f9f427c2dc3b6c6cf6a45399a3d91b18c44d6c1716bf"
)
EXPECTED_RELEASE_SELF_SHA256 = "a00be1e4f07cd119107fbf93a73e6194d21f37ce586f489f467af08b89e14cb1"
EXPECTED_EXECUTION_CODE_PATHS = {
    "__init__.py",
    "acquire_r1.py",
    "execution_guard.py",
    "extraction/__init__.py",
    "extraction/common.py",
    "extraction/eres_prepooling.py",
    "extraction/fixtures.py",
    "extraction/ssl.py",
    "provenance.py",
    "r1_gate.py",
    "r1_execute.py",
    "r1_smoke.py",
    "run_provenance.py",
    "schemas.py",
    "validate_r0.py",
    "validate_r1_gate.py",
    "windows_job.py",
}
EXPECTED_R0_DEPENDENCIES = {
    "r0_protocol": {
        "path": "configs/protocol/r0_protocol.json",
        "sha256": "4a7767a966f27ff85e20b315922b47a8f8d16356a5701cf55c09b292b87bab0d",
    },
    "compute_ceiling": {
        "path": "configs/protocol/compute_ceiling.json",
        "sha256": "7ea9a68238e03a67771b8064c4933149c67dd44c28a8225d25b1af7b94810330",
    },
    "model_registry": {
        "path": "models/registry.json",
        "sha256": "ba5bfa5201b08e41f350ec1c6995eea2688db564b14ea1dedefc3a30e492bbe2",
    },
}
EXPECTED_ENVIRONMENT = {
    "python": "3.12.10",
    "uv": "0.9.17 (2b5d65e61 2025-12-09)",
    "backend": "cpu",
    "project_root": "environment",
    "pyproject": {
        "path": "environment/pyproject.toml",
        "sha256": "4f82330869d8e4dede12ebc6346422a518ebff2245b8875b3a092e5b666bb3c6",
    },
    "lock": {
        "path": "environment/uv.lock",
        "sha256": "2865f5c05352ba1dc99d2899cc583a3844a2ab1a590388aeee3a0eb0644b9af1",
    },
    "runtime_packages": {
        "huggingface-hub": "0.31.4",
        "matplotlib": "3.10.3",
        "numpy": "1.26.4",
        "pandas": "2.2.3",
        "psutil": "7.0.0",
        "pyarrow": "20.0.0",
        "pyyaml": "6.0.2",
        "safetensors": "0.5.3",
        "scikit-learn": "1.6.1",
        "soundfile": "0.13.1",
        "torch": "2.7.1+cpu",
        "torchaudio": "2.7.1+cpu",
        "transformers": "4.52.3",
    },
}
EXPECTED_ACTIONS = {
    "metadata_read": True,
    "environment_sync": True,
    "source_checkout": True,
    "model_artifact_download": True,
    "d0_fixture_materialization": True,
    "neural_smoke": True,
    "corpus_download": False,
    "full_extraction": False,
    "confirmatory_access": False,
    "training": False,
}
EXPECTED_SUPERVISION = {
    "entrypoint": "python -m experiments.speaker_representation_scd.r1_execute",
    "direct_worker_execution_allowed": False,
    "lease_relative_path": "control/r1_execution.lock",
    "lease_owner_validation": "identity_bound_ancestor_depth_2",
    "monitor_interval_seconds": 0.25,
    "legacy_process_scan": "continuous_fail_closed",
    "process_inspection_failure": "abort",
    "max_parallel_actions": 1,
    "max_parallel_models": 1,
    "max_worker_processes": 1,
    "max_cpu_threads": 8,
    "max_process_tree_resident_ram_gib": 24,
    "hard_memory_boundary": "windows_job_object_job_memory",
    "hard_memory_contract_ceiling_bytes": 24 * 1024**3,
    "hard_memory_reserved_headroom_bytes": 1024**3,
    "authoritative_peak_receipt_field": (
        "hard_memory_boundary.authoritative_peak_job_memory_bytes"
    ),
    "sampled_process_tree_rss_role": "diagnostic_fail_closed",
    "max_action_wall_hours": 24,
    "max_cumulative_wall_hours": 96,
    "usage_receipts": "control/usage/*.json",
    "worker_receipt_binding": "execution_id_and_expected_relative_path",
    "action_receipt_authority": "unique_completed_usage_attestation",
    "orphan_receipt_policy": "quarantine_and_retry",
    "downstream_completed_attestation_required": True,
    "environment_sync_receipt": "manifests/r1_environment_sync.json",
    "model_acquisition_receipt": "manifests/r1_model_acquisition.json",
    "evidence_collision_policy": "abort_before_action",
}
EXPECTED_RECEIPT_COMPATIBILITY = {
    "environment_sync_predecessors": [
        {
            "r1_gate_sha256": "ac07d6366cda4ac1a2655363fe66fff3a77eeb7213a81c2f24a97064fc4c636f",
            "r1_gate_self_sha256": "58d848cfbc292f12efa6365aa674de6a098d1e57fb89d5773251b71950a6cd30",
            "execution_code_manifest_sha256": "feca03faef2459cfa0ff34c56d134e42d18d0f1b7f5130e543f688f568d1277e",
            "environment_pyproject_sha256": "4f82330869d8e4dede12ebc6346422a518ebff2245b8875b3a092e5b666bb3c6",
            "environment_lock_sha256": "2865f5c05352ba1dc99d2899cc583a3844a2ab1a590388aeee3a0eb0644b9af1",
        }
    ]
}
EXPECTED_SSL_MODELS = {
    "mhubert-147": "2359b3e9dc6869cb0855119a2866f056aeb400e46252da9cbcc8e9b7aee50c8b",
    "wavlm-base-plus": "3bb273a6ace99408b50cfc81afdbb7ef2de02da2eab0234e18db608ce692fe51",
    "unispeech-sat-base-plus": "0ebc4dd3edc1e10e21a4d16791ad65b9217033d9205317e999a973304b27eda4",
}


@dataclass(frozen=True, slots=True)
class R1GateResult:
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


class R1GateError(RuntimeError):
    pass


def _exact_keys(value: Any, expected: set[str], path: str, errors: list[str]) -> bool:
    if not isinstance(value, dict):
        errors.append(f"{path}: must be an object")
        return False
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        errors.append(f"{path}: missing keys {missing}")
    if unknown:
        errors.append(f"{path}: unknown keys {unknown}")
    return not missing and not unknown


def _legacy_content_valid(document: dict[str, Any]) -> bool:
    expected = document.get("content_sha256")
    body = {key: value for key, value in document.items() if key != "content_sha256"}
    return isinstance(expected, str) and expected == sha256_bytes(canonical_json_bytes(body))


def live_legacy_processes() -> tuple[dict[str, Any], ...]:
    matches, failures = strict_legacy_scan()
    inspection_rows = tuple(
        {
            "pid": row["pid"],
            "name": row["name"],
            "module": None,
            "inspection_error": row["reason"],
        }
        for row in failures
    )
    return tuple(sorted((*matches, *inspection_rows), key=lambda item: item["pid"]))


def _validate_source_registry(path: Path, errors: list[str]) -> dict[str, Any] | None:
    try:
        document = load_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"source_registry: cannot load: {exc}")
        return None
    if not self_sha256_valid(document):
        errors.append("source_registry.self_sha256: invalid")
    if document.get("self_sha256") != EXPECTED_SOURCE_REGISTRY_SELF_SHA256:
        errors.append("source_registry.self_sha256: not the reviewed R1 source identity")
    frontend = document.get("ssl_frontend", {})
    if frontend.get("output_stride_samples") != 320:
        errors.append("source_registry.ssl_frontend.output_stride_samples: must equal 320")
    if frontend.get("output_receptive_field_samples") != 400:
        errors.append("source_registry.ssl_frontend.output_receptive_field_samples: must equal 400")
    models = document.get("models", [])
    if not isinstance(models, list):
        errors.append("source_registry.models: must be a list")
        models = []
    identities: dict[str, str | None] = {}
    for model in models:
        files = model.get("required_files", []) if isinstance(model, dict) else []
        weight = next(
            (
                item.get("sha256")
                for item in files
                if item.get("path") in {"model.safetensors", "pytorch_model.bin"}
            ),
            None,
        )
        identities[str(model.get("model_id"))] = weight
        if model.get("trust_remote_code") is not False:
            errors.append(
                f"source_registry.models.{model.get('model_id')}: remote code must be false"
            )
    if identities != EXPECTED_SSL_MODELS:
        errors.append("source_registry.models: frozen SSL artifact identities differ")
    eres = document.get("eres2netv2", {})
    if eres.get("source_revision") != "707eef4eb9b95fd4a9886776df0022390049a5a6":
        errors.append("source_registry.eres2netv2.source_revision: unexpected")
    if eres.get("checkpoint_file", {}).get("sha256") != (
        "0eb4057106b2573dd7b132cf0c36273ab29afd192c1610f80baa9c556dbb963c"
    ):
        errors.append("source_registry.eres2netv2.checkpoint_file: unexpected")
    taps = eres.get("taps", [])
    if [tap.get("tap_id") for tap in taps if isinstance(tap, dict)] != [
        "S1",
        "S2",
        "S3",
        "S4",
        "FUSED",
    ]:
        errors.append("source_registry.eres2netv2.taps: unexpected tap sequence")
    if not taps or taps[-1].get("official_pool_input") is not True:
        errors.append("source_registry.eres2netv2.taps: FUSED must be the official pool input")
    constraints = document.get("execution_constraints", {})
    if constraints.get("remote_code_allowed") is not False:
        errors.append("source_registry.execution_constraints.remote_code_allowed: must be false")
    if constraints.get("unverified_artifact_execution_allowed") is not False:
        errors.append(
            "source_registry.execution_constraints.unverified_artifact_execution_allowed: must be false"
        )
    return document


def _validate_release(
    path: Path, expected_byte_sha256: str, errors: list[str]
) -> dict[str, Any] | None:
    if not path.is_file():
        errors.append(f"legacy_release: missing {path}")
        return None
    if sha256_file(path) != expected_byte_sha256:
        errors.append("legacy_release: byte identity mismatch")
    try:
        document = load_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"legacy_release: cannot load: {exc}")
        return None
    if not self_sha256_valid(document):
        errors.append("legacy_release.self_sha256: invalid")
    if document.get("self_sha256") != EXPECTED_RELEASE_SELF_SHA256:
        errors.append("legacy_release.self_sha256: not the reviewed release identity")
    if document.get("legacy_process_match_count") != 0:
        errors.append("legacy_release.legacy_process_match_count: must equal zero")
    artifacts = document.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 2:
        errors.append("legacy_release.artifacts: completion and verification are required")
        return document
    if {artifact.get("role") for artifact in artifacts if isinstance(artifact, dict)} != {
        "completion",
        "verification",
    }:
        errors.append("legacy_release.artifacts: roles must be completion and verification")
    for artifact in artifacts:
        relative = artifact.get("path")
        if not isinstance(relative, str):
            errors.append("legacy_release.artifacts.path: invalid")
            continue
        target = REPOSITORY_ROOT / relative
        if not target.is_file():
            errors.append(f"legacy_release.artifacts: missing {relative}")
            continue
        if sha256_file(target) != artifact.get("sha256"):
            errors.append(f"legacy_release.artifacts: byte identity mismatch for {relative}")
            continue
        try:
            payload = load_json(target)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"legacy_release.artifacts: cannot load {relative}: {exc}")
            continue
        if not _legacy_content_valid(payload):
            errors.append(f"legacy_release.artifacts: content identity invalid for {relative}")
        if payload.get("content_sha256") != artifact.get("content_sha256"):
            errors.append(f"legacy_release.artifacts: content identity mismatch for {relative}")
        if artifact.get("role") == "completion":
            if payload.get("schema_version") != "turn_episode_phase4_completion.v1":
                errors.append("legacy_release.artifacts: unexpected completion schema")
            if payload.get("signal_row_count") != 218700 or payload.get("detail_shard_count") != 14:
                errors.append("legacy_release.artifacts: unexpected completion population")
        if artifact.get("role") == "verification":
            if payload.get("schema_version") != "turn_episode_phase4_verification.v1":
                errors.append("legacy_release.artifacts: unexpected verification schema")
            if payload.get("passed") is not True:
                errors.append("legacy_release.artifacts: independent verification did not pass")
    return document


def validate_r1_gate(root: Path | None = None, *, scan_processes: bool = True) -> R1GateResult:
    base = (root or EXPERIMENT_ROOT).resolve()
    errors: list[str] = []
    artifacts: dict[str, str] = {}
    r0 = validate_bundle(base)
    if not r0.valid:
        errors.append("r0_bundle: invalid")
    source_path = base / SOURCE_REGISTRY_PATH
    source = _validate_source_registry(source_path, errors)
    if source_path.is_file():
        artifacts["source_registry"] = sha256_file(source_path)
    gate_path = base / GATE_PATH
    try:
        gate = load_json(gate_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"r1_gate: cannot load: {exc}")
        gate = {}
    if gate_path.is_file():
        artifacts["r1_gate"] = sha256_file(gate_path)
    if gate:
        if not self_sha256_valid(gate):
            errors.append("r1_gate.self_sha256: invalid")
        expected_keys = {
            "schema_version",
            "artifact_role",
            "experiment_id",
            "protocol_version",
            "authority",
            "r0_dependencies",
            "environment",
            "source_registry",
            "execution_code",
            "legacy_release",
            "storage",
            "authorization",
            "supervision",
            "receipt_compatibility",
            "smoke",
            "self_sha256",
        }
        _exact_keys(gate, expected_keys, "r1_gate", errors)
        if gate.get("artifact_role") != "r1_acquisition_smoke_gate":
            errors.append("r1_gate.artifact_role: unexpected")
        if gate.get("schema_version") != 1 or gate.get("protocol_version") != "r1-smoke-1":
            errors.append("r1_gate: schema or protocol version differs")
        if gate.get("experiment_id") != "speaker_representation_scd_v1":
            errors.append("r1_gate.experiment_id: unexpected")
        if gate.get("authority") != {
            "path": "experiments/speaker_representation_scd/EXPERIMENT_PLAN.en.md",
            "sha256": "ca46bce33b90c89597b5c9f2092b952a3f76d638c9c5524d4ca7ba23800e9191",
        }:
            errors.append("r1_gate.authority: differs from the amended owner authority")
        if gate.get("r0_dependencies") != EXPECTED_R0_DEPENDENCIES:
            errors.append("r1_gate.r0_dependencies: differs from the accepted R0 bundle")
        else:
            for name, item in EXPECTED_R0_DEPENDENCIES.items():
                if sha256_file(base / item["path"]) != item["sha256"]:
                    errors.append(f"r1_gate.r0_dependencies.{name}: identity mismatch")
        if gate.get("authorization") != EXPECTED_ACTIONS:
            errors.append("r1_gate.authorization: action boundary differs")
        if gate.get("supervision") != EXPECTED_SUPERVISION:
            errors.append("r1_gate.supervision: execution ceiling enforcement differs")
        if gate.get("receipt_compatibility") != EXPECTED_RECEIPT_COMPATIBILITY:
            errors.append("r1_gate.receipt_compatibility: predecessor identity differs")
        environment = gate.get("environment", {})
        if environment != EXPECTED_ENVIRONMENT:
            errors.append("r1_gate.environment: differs from the reviewed CPU lock")
        for name in ("pyproject", "lock"):
            item = environment.get(name, {})
            relative = item.get("path")
            expected = item.get("sha256")
            if not isinstance(relative, str) or not isinstance(expected, str):
                errors.append(f"r1_gate.environment.{name}: invalid")
                continue
            target = base / relative
            if not target.is_file() or sha256_file(target) != expected:
                errors.append(f"r1_gate.environment.{name}: identity mismatch")
            else:
                artifacts[f"environment_{name}"] = expected
        source_ref = gate.get("source_registry", {})
        if source is not None and source_ref.get("self_sha256") != source.get("self_sha256"):
            errors.append("r1_gate.source_registry: self identity mismatch")
        if source_path.is_file() and source_ref.get("sha256") != sha256_file(source_path):
            errors.append("r1_gate.source_registry: byte identity mismatch")
        if source_ref.get("path") != str(SOURCE_REGISTRY_PATH).replace("\\", "/"):
            errors.append("r1_gate.source_registry: path mismatch")
        code = gate.get("execution_code", {})
        files = code.get("files")
        if not isinstance(files, list) or not files:
            errors.append("r1_gate.execution_code.files: must be a non-empty list")
        else:
            seen: set[str] = set()
            for item in files:
                relative = item.get("path") if isinstance(item, dict) else None
                expected = item.get("sha256") if isinstance(item, dict) else None
                if not isinstance(relative, str) or not isinstance(expected, str):
                    errors.append("r1_gate.execution_code.files: invalid row")
                    continue
                if relative in seen:
                    errors.append(f"r1_gate.execution_code.files: duplicate {relative}")
                seen.add(relative)
                target = base / relative
                if not target.is_file() or sha256_file(target) != expected:
                    errors.append(f"r1_gate.execution_code.files: identity mismatch for {relative}")
            if seen != EXPECTED_EXECUTION_CODE_PATHS:
                errors.append("r1_gate.execution_code.files: path inventory differs")
            manifest_hash = sha256_bytes(canonical_json_bytes(files))
            if code.get("manifest_sha256") != manifest_hash:
                errors.append("r1_gate.execution_code.manifest_sha256: invalid")
        release = gate.get("legacy_release", {})
        release_relative = release.get("path")
        release_hash = release.get("sha256")
        if isinstance(release_relative, str) and isinstance(release_hash, str):
            release_path = base / release_relative
            _validate_release(release_path, release_hash, errors)
            if release_path.is_file():
                artifacts["legacy_release"] = sha256_file(release_path)
        else:
            errors.append("r1_gate.legacy_release: invalid")
        if release_relative != "results/r1/legacy_release.json":
            errors.append("r1_gate.legacy_release: path mismatch")
        storage = gate.get("storage", {})
        if storage != {
            "cache_root_env": "SRSCD_CACHE_ROOT",
            "required_volume": "C:",
            "must_be_outside_repository": True,
            "min_free_gib_before_download": 55,
            "max_source_download_gib": 25,
            "max_derived_cache_gib": 20,
            "max_external_storage_gib": 50,
            "expected_model_files_bytes": 1209138839,
        }:
            errors.append("r1_gate.storage: differs from the R0 ceiling")
        smoke = gate.get("smoke", {})
        if smoke.get("fixtures_per_model") != 10:
            errors.append("r1_gate.smoke.fixtures_per_model: must equal 10")
        if smoke.get("benchmark_windows_per_model") != 100:
            errors.append("r1_gate.smoke.benchmark_windows_per_model: must equal 100")
        if smoke.get("fixture_manifest_sha256") != (
            "5a82813fd5f1b8b40ff1f7ccc4e16fd7cdd05afee6f7df07a6e165e63dedee53"
        ):
            errors.append("r1_gate.smoke.fixture_manifest_sha256: unexpected")
        if smoke.get("batch_modes") != ["single", "batch"]:
            errors.append("r1_gate.smoke.batch_modes: unexpected")
        if smoke.get("required_checks") != [
            "feature_shape",
            "valid_length",
            "independent_input_length_mapping",
            "empirical_source_coordinate_response",
            "exact_availability_frontier",
            "finite_prepool_representation",
            "repeated_run_determinism",
            "future_audio_mutation_invariance",
            "scenario_window_event_straddling",
            "batch_single_parity",
            "eres_fused_to_final_parity",
            "seconds_per_window",
            "peak_ram_gib",
            "supervised_resource_enforcement",
            "hard_job_memory_limit",
            "authoritative_peak_job_memory",
            "completed_usage_action_receipt_binding",
            "aborted_orphan_receipt_rejection_and_retry",
            "exact_run_provenance",
        ]:
            errors.append("r1_gate.smoke.required_checks: unexpected")
        if smoke.get("batch_single_max_abs_tolerance") != 0.0001:
            errors.append("r1_gate.smoke.batch_single_max_abs_tolerance: unexpected")
        if smoke.get("eres_parity_max_abs_tolerance") != 0.000001:
            errors.append("r1_gate.smoke.eres_parity_max_abs_tolerance: unexpected")
        if smoke.get("forecast_required_after_smoke") is not True:
            errors.append("r1_gate.smoke.forecast_required_after_smoke: must be true")
        if smoke.get("forecast_approved") is not False:
            errors.append("r1_gate.smoke.forecast_approved: must remain false")
        if smoke.get("full_extraction_on_pass") is not False:
            errors.append("r1_gate.smoke.full_extraction_on_pass: must remain false")
    processes = live_legacy_processes() if scan_processes else ()
    if processes:
        errors.append("legacy_contention: active speaker_turn_boundary process detected")
    return R1GateResult(
        valid=not errors,
        errors=tuple(errors),
        allowed_actions=dict(gate.get("authorization", {})) if gate else {},
        artifact_hashes=artifacts,
        legacy_processes=processes,
    )


def validated_cache_root(action: str, root: Path | None = None) -> Path:
    result = validate_r1_gate()
    if not result.valid:
        raise R1GateError("; ".join(result.errors))
    if result.allowed_actions.get(action) is not True:
        raise R1GateError(f"R1 action is not authorized: {action}")
    value = os.environ.get("SRSCD_CACHE_ROOT")
    if not value:
        raise R1GateError("SRSCD_CACHE_ROOT is required")
    cache_root = Path(value)
    if not cache_root.is_absolute():
        raise R1GateError("SRSCD_CACHE_ROOT must be absolute")
    resolved = cache_root.resolve()
    repository = (root or REPOSITORY_ROOT).resolve()
    if resolved == repository or repository in resolved.parents:
        raise R1GateError("SRSCD_CACHE_ROOT must be outside the repository")
    if resolved.drive.upper() != "C:":
        raise R1GateError("SRSCD_CACHE_ROOT must be on C:")
    free_gib = shutil.disk_usage(resolved.anchor).free / (1024**3)
    if free_gib < 55:
        raise R1GateError(f"free disk is below 55 GiB: {free_gib:.3f}")
    return resolved
