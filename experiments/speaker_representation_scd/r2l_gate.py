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
from experiments.speaker_representation_scd.r1_gate import EXPERIMENT_ROOT, REPOSITORY_ROOT

GATE_PATH = Path("configs/r2l/legacy_common_gt_gate.json")
AUTHORITY = {
    "path": "experiments/speaker_representation_scd/EXPERIMENT_PLAN.en.md",
    "sha256": "bf1727a62dcd9c8c28cc095c46ebeaaab8e3f723a0c1c6440ed19a9968d590be",
}
SOURCE_ID = "legacy-common-gt-v1"
FORBIDDEN_SOURCE_IDS = (
    "zeroth-korean-development",
    "jvs-development",
    "voxconverse-v03-confirmatory-natural",
    "aishell4-confirmatory-natural-zh",
    "zeroth-korean-confirmatory",
    "jvs-confirmatory",
    "D5",
)
EXPECTED_ACTIONS = {
    "metadata_read": True,
    "legacy_wav_read": True,
    "legacy_validation": True,
    "coordinate_materialization": True,
    "reduced_forecast": True,
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
    "entrypoint": "python -m experiments.speaker_representation_scd.r2l_execute",
    "worker_environment": "repository_root_venv",
    "direct_worker_execution_allowed": False,
    "legacy_process_scan": "continuous_fail_closed",
    "process_inspection_failure": "abort",
    "max_parallel_actions": 1,
    "max_worker_processes": 1,
    "max_cpu_threads": 8,
    "max_process_tree_resident_ram_gib": 24,
    "max_action_wall_hours": 24,
    "max_cumulative_wall_hours": 24,
    "hard_memory_boundary": "windows_job_object_job_memory",
    "action_receipt_authority": "unique_completed_usage_attestation",
    "orphan_receipt_policy": "quarantine_and_retry",
}
EXPECTED_SOURCES = [
    {
        "source_id": "legacy-common-gt-v1",
        "kind": "existing_verified",
        "license": "mixed-upstream-see-legacy-ledger",
        "selection": "diagnostic_dev episodes from exact episode_manifest_dev",
        "expected_episode_count": 804,
        "expected_diagnostic_episode_count": 695,
        "expected_source_identity_count": 616,
        "expected_unique_waveform_count": 600,
        "expected_positive_anchor_count": 450,
        "expected_negative_anchor_count": 360,
        "expected_matched_pair_count": 313,
        "wav_reference_policy": "read_only_in_place",
        "repository_artifacts": [
            {
                "path": "experiments/speaker_turn_boundary/results/turn_episode_v1/episode_manifest_dev.json",
                "sha256": "a5df5e56c2917ba174fca7892fbaa092f2b57705cad7a77bf9a347ba94cddfee",
                "content_sha256": "deb1713cd581c93cc13c47103643041c7551993985379869cfa0ca9a407dff68",
            },
            {
                "path": "experiments/speaker_turn_boundary/results/turn_episode_v1/coverage_inventory.json",
                "sha256": "02a6a118fc90c0d747e9548f07003177b3fc703f33d408d5338427cb6163dd46",
            },
            {
                "path": "experiments/speaker_turn_boundary/results/turn_episode_v1/coverage_inventory_details.jsonl",
                "sha256": "15b2e4f0efa270985c3bbc6d848ee9ed25496089268e561bff921c5c1be3ef8c",
            },
            {
                "path": "experiments/speaker_turn_boundary/results/turn_episode_v1/phase_4_design_ledger.json",
                "sha256": "0a86788a4817d4a205d92b0afb6ee05dc97d11da3e99d4c0501d74be30473691",
                "content_sha256": "c8336c2665b28047b1a169fc9605a6c6a3c400afe553dc3a2d35ca9b20b41536",
            },
            {
                "path": "experiments/speaker_turn_boundary/data/manifests/ls_dev.json",
                "sha256": "14347cdbdb2eff4cc73489f1b59d6755723d9098089dad66ae222984e90370dd",
            },
            {
                "path": "experiments/speaker_turn_boundary/data/manifests/ls_held_out_clean.json",
                "sha256": "c0aabc5ad8c3f00ec53d45f3b372b8ebca7ca9237720a1bb7a70b8de7dda2581",
            },
            {
                "path": "experiments/speaker_turn_boundary/data/manifests/ls_held_out_other.json",
                "sha256": "f0d169394a9fdee9e708bc9cad46c0547946bf967799fa4e2e1a398ddb984079",
            },
        ],
        "legacy_code": {
            "module": "experiments.speaker_turn_boundary.turn_episode.phase4_design",
            "phase4_design_py_sha256": "bbec4e069165dad309c1dd4103521269494bd8c1dcd115a0d2265f632b6edd4a",
        },
    },
]
EXPECTED_MATERIALIZATION = {
    "output_namespace": "legacy-common-gt-v1",
    "canonical_audio": {
        "sample_rate_hz": 16000,
        "channels": 1,
        "sample_width_bytes": 2,
    },
    "waveform_inventory": "data/r2/legacy_common_gt/waveform_inventory.jsonl",
    "source_metadata": "data/r2/legacy_common_gt/source_metadata.jsonl",
    "validation_receipt": "manifests/r2/legacy_common_gt/validation_receipt.json",
    "coordinate_ledger": "manifests/r2/legacy_common_gt/coordinate_ledger.json",
    "reduced_forecast": "manifests/r2/legacy_common_gt/reduced_r3_r4_forecast.json",
    "coordinate_contract": {
        "contexts_ms": [100, 300, 500],
        "hop_samples": 1600,
        "frontier_rule": "eligible_start_plus_context_through_eligible_end",
        "one_shard_per_waveform": True,
        "maximum_jsonl_row_bytes": 1024,
        "r3_primary_frontier_rule": "candidate_coordinate_plus_context",
        "r3_trajectory_offsets_ms": [
            -1000, -750, -500, -300, -200, -100, 0, 100, 200, 300, 500, 750, 1000, 1500, 2000,
        ],
    },
    "r4_panel": {
        "maximum_source_hours": 6,
        "source_order_rule": "stratum_then_sha256",
        "stratum_a": "sources_hosting_at_least_one_matched_pair_anchor",
        "cumulative_cap_rule": "include_while_total_seconds_at_or_below_21600",
    },
    "network_access": False,
    "corpus_download": False,
    "neural_inference": False,
    "training": False,
}
LEGACY_PHASE4_DESIGN_SHA256 = "bbec4e069165dad309c1dd4103521269494bd8c1dcd115a0d2265f632b6edd4a"


@dataclass(frozen=True, slots=True)
class R2LGateResult:
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


class R2LGateError(RuntimeError):
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


def _execution_manifest_errors(base: Path, execution: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(execution, dict):
        return ["r2l_gate.execution_code: invalid"]
    files = execution.get("files")
    if not isinstance(files, list) or not files:
        return ["r2l_gate.execution_code.files: invalid"]
    expected_paths = {
        "execution_guard.py",
        "provenance.py",
        "r1_forecast.py",
        "r1_gate.py",
        "r2l_execute.py",
        "r2l_forecast.py",
        "r2l_gate.py",
        "r2l_materialize.py",
        "run_provenance.py",
        "validate_r2l_gate.py",
        "windows_job.py",
    }
    observed_paths: set[str] = set()
    for index, row in enumerate(files):
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            errors.append(f"r2l_gate.execution_code.files[{index}]: invalid")
            continue
        relative = row.get("path")
        expected_hash = row.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            errors.append(f"r2l_gate.execution_code.files[{index}]: invalid")
            continue
        observed_paths.add(relative)
        target = base / relative
        if not target.is_file() or sha256_file(target) != expected_hash:
            errors.append(f"r2l_gate.execution_code.files[{index}]: identity differs")
    if observed_paths != expected_paths:
        errors.append("r2l_gate.execution_code.files: inventory differs")
    if execution.get("manifest_sha256") != sha256_bytes(canonical_json_bytes(files)):
        errors.append("r2l_gate.execution_code.manifest_sha256: invalid")
    return errors


def validate_r2l_gate(
    root: Path | None = None,
    *,
    cache_root: Path | None = None,
    scan_processes: bool = True,
) -> R2LGateResult:
    base = (root or EXPERIMENT_ROOT).resolve()
    errors: list[str] = []
    artifacts: dict[str, str] = {}
    try:
        gate = load_json(base / GATE_PATH)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        gate = {}
        errors.append(f"r2l_gate: unreadable: {exc}")
    if gate:
        if not self_sha256_valid(gate):
            errors.append("r2l_gate.self_sha256: invalid")
        _exact_keys(
            gate,
            {
                "schema_version",
                "artifact_role",
                "experiment_id",
                "protocol_version",
                "authority",
                "accepted_r1_checkpoint",
                "dependencies",
                "legacy_scope",
                "sources",
                "materialization",
                "storage",
                "authorization",
                "supervision",
                "execution_code",
                "self_sha256",
            },
            "r2l_gate",
            errors,
        )
        if gate.get("schema_version") != 1:
            errors.append("r2l_gate.schema_version: unexpected")
        if gate.get("artifact_role") != "r2l_legacy_common_gt_gate":
            errors.append("r2l_gate.artifact_role: unexpected")
        if gate.get("experiment_id") != "speaker_representation_scd_v1":
            errors.append("r2l_gate.experiment_id: unexpected")
        if gate.get("protocol_version") != "r2l-legacy-common-gt-1":
            errors.append("r2l_gate.protocol_version: unexpected")
        if gate.get("authority") != AUTHORITY:
            errors.append("r2l_gate.authority: differs")
        if gate.get("accepted_r1_checkpoint") != {
            "commit": "f3f234f9bcef00810a1b2eadd5ee2ffcc53e7ede",
            "review_verdict": "accepted",
            "technical_validity_sha256": "fe8195e6b7b64784d35b0a9279bbc13b0c16ece7319d493a894f05376b35d838",
        }:
            errors.append("r2l_gate.accepted_r1_checkpoint: differs")
        expected_dependencies = {
            "source_ledger": {
                "path": "data/source_ledger.json",
                "sha256": "ca81f5eda28b0674b98ead4cb363b1da711fbf33b40c4ddd8ffddb95721177b4",
                "self_sha256": "26a29b8bc8ab7ff7a4d52459d04e2bb112cd60044001ebe74797b27520ad97fc",
            },
            "split_contract": {
                "path": "data/split_contract.json",
                "sha256": "2c68d3b100db099115eb8cfcfb50f70a3c664c4b256c8c87103efd3cb6204d48",
                "self_sha256": "df90e37bca150675bbb4780363fb41a6dd57f64fc505910a35b3cae5b1bf97e8",
            },
            "confirmatory_access_policy": {
                "path": "data/confirmatory_access_policy.json",
                "sha256": "bdd91dcaa5aca3a6e043acfdc00ae6d94e10909a85bf6cc4d14fca529a38d287",
                "self_sha256": "dafae80be50a36f78f116ccf80964e712df5a5869893eba0a0c538e1e38efa8b",
            },
            "compute_ceiling": {
                "path": "configs/protocol/compute_ceiling.json",
                "sha256": "7ea9a68238e03a67771b8064c4933149c67dd44c28a8225d25b1af7b94810330",
                "self_sha256": "29b0225a39f38d9a03584286f932be2722f276be95cfa922bda305155ef312c6",
            },
            "technical_validity": {
                "path": "results/r1/technical_validity.json",
                "sha256": "fe8195e6b7b64784d35b0a9279bbc13b0c16ece7319d493a894f05376b35d838",
                "self_sha256": "9a276d1ced9fb17cdd2f5e0efddf6d8691fcf043b6798ea8081c7ce0b08c7109",
            },
            "forecast_contract": {
                "path": "configs/r1/full_job_forecast_contract.json",
                "sha256": "6a258094a9a75deb09a4d6685b90dd83f2f9170194b3644bb855d40cf1da4980",
                "self_sha256": "3115382954018f54cefa2b691cc56a84fdf5b8998a23d153a2ea3b40e55773e8",
            },
            "protocol_screen": {
                "path": "configs/protocol/reduced_pretraining_screen.json",
                "sha256": "587e9afd3c50f46da7aa022679c5222e8f0437458dd601de6b0a0fe8ca4855b7",
                "self_sha256": "78719c6b10392b8d051f8d90884b73d4ceea845275966cbe8bfa7ae0d2710338",
            },
        }
        dependencies = gate.get("dependencies")
        if not isinstance(dependencies, dict) or set(dependencies) != set(expected_dependencies):
            errors.append("r2l_gate.dependencies: inventory differs")
        else:
            for label, expected in expected_dependencies.items():
                _validate_reference(
                    base,
                    dependencies.get(label),
                    expected,
                    label,
                    errors,
                    artifacts,
                )
        scope = gate.get("legacy_scope")
        if scope != {
            "required_source_ids": [SOURCE_ID],
            "forbidden_source_ids": list(FORBIDDEN_SOURCE_IDS),
            "tier": "development_known",
            "new_corpus_acquisition": False,
            "confirmatory_payload_read_allowed": False,
            "legacy_eres_final_rerun": False,
            "legacy_lseend_rerun": False,
        }:
            errors.append("r2l_gate.legacy_scope: differs")
        if gate.get("authorization") != EXPECTED_ACTIONS:
            errors.append("r2l_gate.authorization: differs")
        if gate.get("storage") != EXPECTED_STORAGE:
            errors.append("r2l_gate.storage: differs")
        if gate.get("supervision") != EXPECTED_SUPERVISION:
            errors.append("r2l_gate.supervision: differs")
        if gate.get("sources") != EXPECTED_SOURCES:
            errors.append("r2l_gate.sources: legacy contract differs")
        else:
            for row in EXPECTED_SOURCES[0]["repository_artifacts"]:
                target = REPOSITORY_ROOT / row["path"]
                if not target.is_file() or sha256_file(target) != row["sha256"]:
                    errors.append(f"r2l_gate.sources.legacy: identity differs for {row['path']}")
                else:
                    artifacts[f"legacy:{row['path']}"] = row["sha256"]
            legacy_code = EXPECTED_SOURCES[0]["legacy_code"]
            module_path = REPOSITORY_ROOT / "experiments" / "speaker_turn_boundary" / "turn_episode" / "phase4_design.py"
            if (
                not module_path.is_file()
                or sha256_file(module_path) != legacy_code["phase4_design_py_sha256"]
            ):
                errors.append("r2l_gate.sources.legacy: phase4_design.py identity differs")
            else:
                artifacts["legacy:phase4_design.py"] = legacy_code["phase4_design_py_sha256"]
        if gate.get("materialization") != EXPECTED_MATERIALIZATION:
            errors.append("r2l_gate.materialization: source boundary differs")
        errors.extend(_execution_manifest_errors(base, gate.get("execution_code")))
    if cache_root is not None and not errors:
        try:
            technical = load_json(base / TECHNICAL_VALIDITY_PATH)
            errors.extend(validate_technical_validity(technical, cache_root.resolve()))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"technical_validity: unreadable: {exc}")
    matches: tuple[dict[str, Any], ...] = ()
    if scan_processes:
        matches, failures = strict_legacy_scan()
        if failures:
            errors.append(f"legacy process inspection failed: {failures}")
        if matches:
            errors.append("legacy contention: active speaker_turn_boundary process detected")
    return R2LGateResult(
        valid=not errors,
        errors=tuple(errors),
        allowed_actions=dict(gate.get("authorization", {})) if gate else {},
        artifact_hashes=artifacts,
        legacy_processes=matches,
    )


def validated_r2l_cache_root(action: str, root: Path | None = None) -> Path:
    value = os.environ.get("SRSCD_CACHE_ROOT")
    if not value:
        raise R2LGateError("SRSCD_CACHE_ROOT is required")
    candidate = Path(value)
    if not candidate.is_absolute():
        raise R2LGateError("SRSCD_CACHE_ROOT must be absolute")
    resolved = candidate.resolve()
    repository = (root or REPOSITORY_ROOT).resolve()
    if resolved == repository or repository in resolved.parents:
        raise R2LGateError("SRSCD_CACHE_ROOT must be outside the repository")
    if resolved.drive.upper() != "C:":
        raise R2LGateError("SRSCD_CACHE_ROOT must be on C:")
    result = validate_r2l_gate(cache_root=resolved)
    if not result.valid:
        raise R2LGateError("; ".join(result.errors))
    if result.allowed_actions.get(action) is not True:
        raise R2LGateError(f"R2-L action is not authorized: {action}")
    free_gib = shutil.disk_usage(resolved.anchor).free / 1024**3
    if free_gib < EXPECTED_STORAGE["min_free_gib_before_download"]:
        raise R2LGateError(f"free disk is below 55 GiB: {free_gib:.3f}")
    return resolved
