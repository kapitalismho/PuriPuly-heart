from __future__ import annotations

import argparse
import json
import math
import os
import sys
import wave
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from experiments.speaker_representation_scd.execution_guard import (
    load_completed_action_receipt,
)
from experiments.speaker_representation_scd.provenance import (
    canonical_json_bytes,
    load_json,
    self_sha256_valid,
    sha256_bytes,
    sha256_file,
    with_self_sha256,
)
from experiments.speaker_representation_scd.r1_gate import EXPERIMENT_ROOT
from experiments.speaker_representation_scd.run_provenance import run_provenance

TECHNICAL_VALIDITY_PATH = Path("results/r1/technical_validity.json")
FORECAST_CONTRACT_PATH = Path("configs/r1/full_job_forecast_contract.json")
DEVELOPMENT_ACQUISITION_PATH = Path(
    "manifests/r2/development/development_acquisition_receipt.json"
)
DEVELOPMENT_LEDGER_PATH = Path(
    "manifests/r2/development/development_coordinate_ledger.json"
)
CACHE_CALIBRATION_PATH = Path(
    "manifests/r2/development/pooled_cache_calibration.json"
)
WAVEFORM_INVENTORY_PATH = Path("data/r2/development/waveform_inventory.jsonl")
MODEL_IDS = (
    "mhubert-147",
    "wavlm-base-plus",
    "unispeech-sat-base-plus",
    "eres2netv2-standard-prepool",
)
AUTHORITY = {
    "path": "experiments/speaker_representation_scd/EXPERIMENT_PLAN.en.md",
    "sha256": "32cfea1779be54b7db2b7094bc35da928b74514d47d2fdd96581ef18d7f84aec",
}
PREDECESSOR_AUTHORITY = {
    "path": "experiments/speaker_representation_scd/EXPERIMENT_PLAN.en.md",
    "sha256": "ca46bce33b90c89597b5c9f2092b952a3f76d638c9c5524d4ca7ba23800e9191",
}
ACCEPTED_IMPLEMENTATION_COMMIT = "f3f234f9bcef00810a1b2eadd5ee2ffcc53e7ede"
DEVELOPMENT_SOURCE_IDS = (
    "legacy-common-gt-v1",
    "zeroth-korean-development",
    "jvs-development",
)
CONFIRMATORY_SOURCE_IDS = (
    "voxconverse-v03-confirmatory-natural",
    "aishell4-confirmatory-natural-zh",
    "zeroth-korean-confirmatory",
    "jvs-confirmatory",
)
POOLING_MS = (100, 300, 500)
HOP_SAMPLES = 1600
SMOKE_WINDOW_MS = (100, 200, 300, 500, 750, 1000, 500, 750, 1000, 300)
FROZEN_INPUTS = {
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
}
EXPECTED_TECHNICAL_REFERENCE = {
    "path": "results/r1/technical_validity.json",
    "sha256": "fe8195e6b7b64784d35b0a9279bbc13b0c16ece7319d493a894f05376b35d838",
    "self_sha256": "9a276d1ced9fb17cdd2f5e0efddf6d8691fcf043b6798ea8081c7ce0b08c7109",
}
EXPECTED_STORAGE = {
    "mhubert-147": (("L1", "L3", "L6", "L9", "L12"), 768, 4),
    "wavlm-base-plus": (("L1", "L3", "L6", "L9", "L12"), 768, 4),
    "unispeech-sat-base-plus": (("L1", "L3", "L6", "L9", "L12"), 768, 4),
    "eres2netv2-standard-prepool": (("S1", "S2", "S3", "S4", "FUSED"), 10240, 4),
}
EXPECTED_CEILINGS = {
    "max_parallel_models": 1,
    "max_worker_processes": 1,
    "max_cpu_threads": 8,
    "max_resident_ram_gib": 24,
    "max_source_download_gib": 25,
    "max_derived_cache_gib": 20,
    "max_external_storage_gib": 50,
    "min_free_disk_gib_before_download": 55,
    "max_total_wall_hours": 24,
    "max_per_model_wall_hours": 8,
}


def _identity_errors(path: Path, reference: dict[str, Any], label: str) -> list[str]:
    errors: list[str] = []
    if not path.is_file():
        return [f"{label}: missing"]
    try:
        if reference.get("sha256") != sha256_file(path):
            errors.append(f"{label}: byte identity mismatch")
        document = load_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [f"{label}: unreadable: {exc}"]
    if not self_sha256_valid(document):
        errors.append(f"{label}: invalid self hash")
    if reference.get("self_sha256") != document.get("self_sha256"):
        errors.append(f"{label}: self identity mismatch")
    return errors


def _frozen_input_errors(references: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if references != FROZEN_INPUTS:
        errors.append("frozen_inputs: identities differ")
        return errors
    for label, reference in FROZEN_INPUTS.items():
        errors.extend(
            _identity_errors(EXPERIMENT_ROOT / reference["path"], reference, label)
        )
    return errors


def _safe_cache_path(
    cache_root: Path,
    relative: str,
    prefix: Path,
    label: str,
) -> tuple[Path | None, list[str]]:
    path_value = Path(relative)
    if path_value.is_absolute() or ".." in path_value.parts:
        return None, [f"{label}: non-canonical path"]
    root = cache_root.resolve()
    expected_parts = prefix.parts
    if path_value.parts[: len(expected_parts)] != expected_parts:
        return None, [f"{label}: outside development namespace"]
    candidate = root / path_value
    try:
        resolved = candidate.resolve(strict=False)
        relative_resolved = resolved.relative_to(root)
    except (OSError, ValueError):
        return None, [f"{label}: path escapes cache root"]
    if relative_resolved.parts[: len(expected_parts)] != expected_parts:
        return None, [f"{label}: resolved path leaves development namespace"]
    return candidate, []


def _load_fixed_cache_document(
    cache_root: Path,
    relative: Path,
    label: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    path, errors = _safe_cache_path(
        cache_root,
        relative.as_posix(),
        Path("manifests/r2/development"),
        label,
    )
    if errors or path is None:
        return None, errors
    if not path.is_file():
        return None, [f"{label}: missing"]
    try:
        return load_json(path), []
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return None, [f"{label}: unreadable: {exc}"]


def validate_technical_validity(
    document: dict[str, Any],
    cache_root: Path | None = None,
) -> list[str]:
    errors: list[str] = []
    if document.get("artifact_role") != "r1_extractor_technical_validity":
        errors.append("technical_validity: unexpected artifact role")
    if document.get("schema_version") != 1:
        errors.append("technical_validity: unexpected schema version")
    if document.get("experiment_id") != "speaker_representation_scd_v1":
        errors.append("technical_validity: unexpected experiment")
    if document.get("authority") != PREDECESSOR_AUTHORITY:
        errors.append("technical_validity: authority differs")
    if document.get("accepted_implementation_commit") != ACCEPTED_IMPLEMENTATION_COMMIT:
        errors.append("technical_validity: implementation commit differs")
    if not self_sha256_valid(document):
        errors.append("technical_validity: invalid self hash")
    technical = document.get("technical_validity")
    expected_technical = {
        "extractor_valid": True,
        "causal_future_mutation_valid": True,
        "timestamp_mapping_valid": True,
        "batch_single_parity_valid": True,
        "eres_reconstruction_valid": True,
    }
    if technical != expected_technical:
        errors.append("technical_validity: incomplete extractor validity")
    phase = document.get("phase_state")
    if phase != {
        "g0_extraction_valid": True,
        "r1_phase_exit": False,
        "forecast_approved": False,
        "full_extraction_enabled": False,
        "blocking_reason": "r2_development_coordinate_ledger_and_cache_calibration_missing",
    }:
        errors.append("technical_validity: phase boundary differs")
    rows = document.get("model_smoke_reports")
    if not isinstance(rows, list) or [
        row.get("model_id") for row in rows if isinstance(row, dict)
    ] != list(MODEL_IDS):
        errors.append("technical_validity: model inventory differs")
        rows = []
    if any(row.get("passed") is not True for row in rows):
        errors.append("technical_validity: a model smoke did not pass")
    if cache_root is None:
        errors.append("technical_validity: external cache root required")
        return errors
    if errors:
        return errors
    action_refs = (
        (document.get("environment_receipt", {}), "sync-environment"),
        (document.get("model_acquisition_receipt", {}), "models"),
    )
    for reference, action in action_refs:
        relative = reference.get("relative_to_cache_root")
        if not isinstance(relative, str):
            errors.append(f"technical_validity: {action} receipt path invalid")
            continue
        path = cache_root / relative
        try:
            receipt = load_completed_action_receipt(cache_root, path, action)
        except (OSError, ValueError, RuntimeError) as exc:
            errors.append(f"technical_validity: {action} receipt invalid: {exc}")
            continue
        if sha256_file(path) != reference.get("sha256"):
            errors.append(f"technical_validity: {action} byte identity mismatch")
        if receipt.get("self_sha256") != reference.get("self_sha256"):
            errors.append(f"technical_validity: {action} self identity mismatch")
    for row in rows:
        relative = row.get("relative_to_cache_root")
        if not isinstance(relative, str):
            errors.append(f"technical_validity: {row['model_id']} path invalid")
            continue
        path = cache_root / relative
        try:
            report = load_completed_action_receipt(cache_root, path, "smoke")
            usage_path = cache_root / "control" / "usage" / f"{row['execution_id']}.json"
            usage = load_json(usage_path)
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
            errors.append(f"technical_validity: {row['model_id']} smoke invalid: {exc}")
            continue
        fixture_windows = tuple(
            fixture.get("window_samples", -1) // 16
            for fixture in report.get("fixtures", [])
            if isinstance(fixture, dict)
        )
        benchmark = report.get("benchmark", {})
        checks = {
            "report_byte": sha256_file(path) == row.get("sha256"),
            "report_self": report.get("self_sha256") == row.get("self_sha256"),
            "model": report.get("model_id") == row.get("model_id"),
            "passed": report.get("passed") is True,
            "execution": report.get("supervision_binding", {}).get("execution_id")
            == row.get("execution_id"),
            "usage_self": self_sha256_valid(usage)
            and usage.get("self_sha256") == row.get("usage_self_sha256"),
            "parameter_count": report.get("parameter_count")
            == row.get("parameter_count"),
            "cold_load": report.get("cold_load_seconds")
            == row.get("cold_load_seconds"),
            "single_cost": benchmark.get("single", {}).get("seconds_per_window")
            == row.get("single_seconds_per_window"),
            "single_count": benchmark.get("single", {}).get("window_count") == 100,
            "fixture_contexts": fixture_windows == SMOKE_WINDOW_MS,
            "peak_job": usage.get("hard_memory_boundary", {}).get(
                "authoritative_peak_job_memory_bytes"
            )
            == row.get("authoritative_peak_job_memory_bytes"),
        }
        errors.extend(
            f"technical_validity: {row['model_id']} {name} mismatch"
            for name, passed in checks.items()
            if not passed
        )
    return errors


def external_validation_binding(document: dict[str, Any]) -> dict[str, Any]:
    return {
        "environment_receipt": document["environment_receipt"],
        "model_acquisition_receipt": document["model_acquisition_receipt"],
        "model_smoke_reports": [
            {
                key: row[key]
                for key in (
                    "model_id",
                    "sha256",
                    "self_sha256",
                    "execution_id",
                    "usage_self_sha256",
                )
            }
            for row in document["model_smoke_reports"]
        ],
    }


def validate_forecast_contract(document: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(document, dict):
        return ["forecast_contract: not an object"]
    if document.get("artifact_role") != "r1_full_job_forecast_contract":
        errors.append("forecast_contract: unexpected artifact role")
    if document.get("schema_version") != 2:
        errors.append("forecast_contract: unexpected schema version")
    if document.get("experiment_id") != "speaker_representation_scd_v1":
        errors.append("forecast_contract: unexpected experiment")
    if document.get("authority") != AUTHORITY:
        errors.append("forecast_contract: authority differs")
    if not self_sha256_valid(document):
        errors.append("forecast_contract: invalid self hash")
    technical_ref = document.get("technical_validity_receipt")
    if technical_ref != EXPECTED_TECHNICAL_REFERENCE:
        errors.append("forecast_contract: technical receipt identity differs")
    else:
        errors.extend(
            _identity_errors(
                EXPERIMENT_ROOT / technical_ref["path"],
                technical_ref,
                "forecast_contract.technical_validity_receipt",
            )
        )
    frozen_inputs = document.get("frozen_inputs")
    if not isinstance(frozen_inputs, dict):
        errors.append("forecast_contract: frozen inputs invalid")
    else:
        errors.extend(_frozen_input_errors(frozen_inputs))
    if document.get("canonical_external_inputs") != {
        "development_acquisition_receipt": DEVELOPMENT_ACQUISITION_PATH.as_posix(),
        "development_coordinate_ledger": DEVELOPMENT_LEDGER_PATH.as_posix(),
        "pooled_cache_calibration": CACHE_CALIBRATION_PATH.as_posix(),
        "development_waveform_inventory": WAVEFORM_INVENTORY_PATH.as_posix(),
        "arbitrary_input_paths_allowed": False,
    }:
        errors.append("forecast_contract: canonical external inputs differ")
    approval = document.get("approval")
    if approval != {
        "forecast_approved": False,
        "full_extraction_enabled": False,
        "required_before_approval": [
            "authoritative_four_model_smoke",
            "verified_r2_development_acquisition",
            "complete_r2_development_coordinate_ledger",
            "r2_pooled_cache_calibration",
            "reviewed_ceiling_decision",
        ],
    }:
        errors.append("forecast_contract: approval boundary differs")
    scope = document.get("development_scope")
    if not isinstance(scope, dict):
        errors.append("forecast_contract: development scope invalid")
        scope = {}
    if scope.get("required_source_ids") != list(DEVELOPMENT_SOURCE_IDS):
        errors.append("forecast_contract: development source scope differs")
    if scope.get("forbidden_source_ids") != list(CONFIRMATORY_SOURCE_IDS):
        errors.append("forecast_contract: confirmatory source scope differs")
    if scope.get("confirmatory_coordinate_access_before_unlock") is not False:
        errors.append("forecast_contract: confirmatory coordinates are not sealed")
    storage_rows = document.get("model_storage_contracts")
    if not isinstance(storage_rows, list) or [
        row.get("model_id") for row in storage_rows if isinstance(row, dict)
    ] != list(MODEL_IDS):
        errors.append("forecast_contract: model storage inventory differs")
    else:
        for row in storage_rows:
            layers, dimension, dtype_bytes = EXPECTED_STORAGE[row["model_id"]]
            expected = {
                "model_id": row["model_id"],
                "retained_layer_ids": list(layers),
                "pooled_dimension_per_layer": dimension,
                "dtype_bytes": dtype_bytes,
            }
            if row != expected:
                errors.append(f"forecast_contract: {row['model_id']} storage contract differs")
    if document.get("coordinate_contract") != {
        "context_mode": "local_trailing_window",
        "pooling_ms": list(POOLING_MS),
        "primary_continuous_hop_ms": 100,
        "ledger_role": "r2_development_coordinate_ledger",
        "complete_source_coverage_required": True,
        "actual_hashed_jsonl_shards_required": True,
        "waveform_inventory_role": "r2_development_waveform_inventory",
        "coordinate_row_schema": "r2_primary_continuous_coordinate_v1",
        "sample_rate_hz": 16000,
        "hop_samples": HOP_SAMPLES,
        "frontier_rule": "eligible_start_plus_context_then_100ms_hops_through_eligible_end",
    }:
        errors.append("forecast_contract: coordinate contract differs")
    if document.get("forecast_method") != {
        "runtime_measurement": "balanced_ten_fixture_worst_case_upper_bound",
        "runtime_upper_bound_factor": 10,
        "runtime_safety_multiplier": 1.25,
        "cold_load_once_per_model": True,
        "cache_calibration_role": "r2_pooled_cache_calibration",
        "actual_npz_and_sample_manifest_required": True,
        "cache_projection_uses_max_measured_or_raw_bytes": True,
        "current_external_root_bytes_included": True,
    }:
        errors.append("forecast_contract: forecast method differs")
    if document.get("ceilings") != EXPECTED_CEILINGS:
        errors.append("forecast_contract: ceilings differ")
    return errors


def _reference_matches(reference: Any, expected: dict[str, Any], label: str) -> list[str]:
    return [] if reference == expected else [f"{label}: frozen input binding differs"]


def _coordinate_row(
    waveform: dict[str, Any], context_ms: int, frontier_sample: int
) -> dict[str, Any]:
    payload = {
        "schema": "r2_primary_continuous_coordinate_v1",
        "source_id": waveform["source_id"],
        "waveform_id": waveform["waveform_id"],
        "waveform_sha256": waveform["artifact_sha256"],
        "context_ms": context_ms,
        "window_start_sample": frontier_sample - context_ms * 16,
        "window_end_sample": frontier_sample,
        "observed_frontier_sample": frontier_sample,
        "hop_samples": HOP_SAMPLES,
    }
    return {
        "coordinate_id": sha256_bytes(canonical_json_bytes(payload)),
        **payload,
    }


def _expected_coordinate_rows(waveform: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    start = waveform["eligible_start_sample"]
    end = waveform["eligible_end_sample"]
    for context_ms in POOLING_MS:
        first_frontier = start + context_ms * 16
        rows.extend(
            _coordinate_row(waveform, context_ms, frontier)
            for frontier in range(first_frontier, end + 1, HOP_SAMPLES)
        )
    return rows


def _coordinate_row_is_expected(
    row: dict[str, Any], waveforms: dict[str, dict[str, Any]]
) -> bool:
    waveform_id = row.get("waveform_id")
    context_ms = row.get("context_ms")
    frontier = row.get("observed_frontier_sample")
    if (
        not isinstance(waveform_id, str)
        or waveform_id not in waveforms
        or context_ms not in POOLING_MS
        or not isinstance(frontier, int)
    ):
        return False
    waveform = waveforms[waveform_id]
    first = waveform["eligible_start_sample"] + context_ms * 16
    if frontier < first or frontier > waveform["eligible_end_sample"]:
        return False
    if (frontier - first) % HOP_SAMPLES:
        return False
    return row == _coordinate_row(waveform, context_ms, frontier)


def _development_acquisition_errors(
    receipt: dict[str, Any],
    contract: dict[str, Any],
    cache_root: Path,
) -> tuple[list[str], int, dict[str, dict[str, Any]]]:
    errors: list[str] = []
    if receipt.get("schema_version") != 1:
        errors.append("development_acquisition: unexpected schema version")
    if receipt.get("artifact_role") != "r2_development_acquisition_receipt":
        errors.append("development_acquisition: unexpected artifact role")
    if receipt.get("experiment_id") != "speaker_representation_scd_v1":
        errors.append("development_acquisition: unexpected experiment")
    if receipt.get("authority") != AUTHORITY:
        errors.append("development_acquisition: authority differs")
    if not self_sha256_valid(receipt):
        errors.append("development_acquisition: invalid self hash")
    errors.extend(
        _reference_matches(
            receipt.get("frozen_inputs"),
            contract.get("frozen_inputs", {}),
            "development_acquisition",
        )
    )
    if receipt.get("development_source_ids") != list(DEVELOPMENT_SOURCE_IDS):
        errors.append("development_acquisition: source inventory differs")
    free_bytes = receipt.get("free_bytes_before_download")
    if not isinstance(free_bytes, int) or free_bytes < 0:
        errors.append("development_acquisition: free disk measurement invalid")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        errors.append("development_acquisition: artifacts missing")
        return errors, 0, {}
    covered: set[str] = set()
    external_bytes = 0
    listed_external_paths: set[str] = set()
    seen_paths: set[tuple[str, str]] = set()
    artifact_by_path: dict[str, dict[str, Any]] = {}
    legacy_expected = "experiments/speaker_turn_boundary/results/turn_episode_v1/episode_manifest_dev.json"
    legacy_sha256 = "a5df5e56c2917ba174fca7892fbaa092f2b57705cad7a77bf9a347ba94cddfee"
    legacy_size_bytes = 2936679
    for index, row in enumerate(artifacts):
        label = f"development_acquisition.artifacts[{index}]"
        if not isinstance(row, dict):
            errors.append(f"{label}: invalid row")
            continue
        source_id = row.get("source_id")
        location = row.get("location")
        relative = row.get("relative_path")
        if source_id not in DEVELOPMENT_SOURCE_IDS or not isinstance(relative, str):
            errors.append(f"{label}: source or path invalid")
            continue
        if source_id in CONFIRMATORY_SOURCE_IDS:
            errors.append(f"{label}: confirmatory source forbidden")
            continue
        key = (str(location), relative)
        if key in seen_paths:
            errors.append(f"{label}: duplicate path")
            continue
        seen_paths.add(key)
        covered.add(source_id)
        if (
            source_id == "legacy-common-gt-v1"
            and location == "repository"
            and relative == legacy_expected
        ):
            if row.get("sha256") != legacy_sha256 or row.get(
                "size_bytes"
            ) != legacy_size_bytes:
                errors.append(f"{label}: frozen legacy identity differs")
            path = EXPERIMENT_ROOT.parents[1] / relative
        else:
            expected_prefix = Path("sources/r2/development") / source_id
            if location != "cache_root":
                errors.append(f"{label}: external source location differs")
                continue
            path, path_errors = _safe_cache_path(
                cache_root, relative, expected_prefix, label
            )
            errors.extend(path_errors)
            if path is None:
                continue
            listed_external_paths.add(relative)
            artifact_by_path[relative] = row
        if not path.is_file():
            errors.append(f"{label}: file missing")
            continue
        actual_size = path.stat().st_size
        if row.get("size_bytes") != actual_size:
            errors.append(f"{label}: size mismatch")
        if row.get("sha256") != sha256_file(path):
            errors.append(f"{label}: sha256 mismatch")
        if location == "cache_root":
            external_bytes += actual_size
    actual_external_paths: set[str] = set()
    source_root = cache_root / "sources" / "r2" / "development"
    if source_root.is_dir():
        for path in source_root.rglob("*"):
            if path.is_symlink():
                errors.append(
                    f"development_acquisition: source symlink forbidden: {path}"
                )
            elif path.is_file():
                actual_external_paths.add(path.relative_to(cache_root).as_posix())
    if listed_external_paths != actual_external_paths:
        errors.append("development_acquisition: source file inventory differs")
    if covered != set(DEVELOPMENT_SOURCE_IDS):
        errors.append("development_acquisition: source coverage incomplete")
    if receipt.get("external_source_download_bytes") != external_bytes:
        errors.append("development_acquisition: external byte total differs")
    inventory_ref = receipt.get("waveform_inventory")
    waveforms: dict[str, dict[str, Any]] = {}
    if not isinstance(inventory_ref, dict) or inventory_ref.get(
        "relative_to_cache_root"
    ) != WAVEFORM_INVENTORY_PATH.as_posix():
        errors.append("development_acquisition: waveform inventory binding differs")
        return errors, external_bytes, waveforms
    inventory_path, inventory_path_errors = _safe_cache_path(
        cache_root,
        WAVEFORM_INVENTORY_PATH.as_posix(),
        Path("data/r2/development"),
        "development_acquisition.waveform_inventory",
    )
    errors.extend(inventory_path_errors)
    if inventory_path is None or not inventory_path.is_file():
        errors.append("development_acquisition: waveform inventory missing")
        return errors, external_bytes, waveforms
    if inventory_ref.get("size_bytes") != inventory_path.stat().st_size or inventory_ref.get(
        "sha256"
    ) != sha256_file(inventory_path):
        errors.append("development_acquisition: waveform inventory identity differs")
    inventory_rows, inventory_row_errors = _read_jsonl(
        inventory_path, "development_acquisition.waveform_inventory"
    )
    errors.extend(inventory_row_errors)
    inventory_sources: set[str] = set()
    inventory_artifact_paths: set[str] = set()
    for index, waveform in enumerate(inventory_rows):
        label = f"development_acquisition.waveform_inventory[{index}]"
        expected_keys = {
            "waveform_id",
            "source_id",
            "artifact_relative_to_cache_root",
            "artifact_sha256",
            "artifact_size_bytes",
            "sample_rate_hz",
            "num_samples",
            "eligible_start_sample",
            "eligible_end_sample",
        }
        waveform_id = waveform.get("waveform_id")
        source_id = waveform.get("source_id")
        relative = waveform.get("artifact_relative_to_cache_root")
        if (
            set(waveform) != expected_keys
            or not isinstance(waveform_id, str)
            or not waveform_id
            or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in waveform_id)
            or waveform_id in waveforms
            or source_id not in DEVELOPMENT_SOURCE_IDS
            or not isinstance(relative, str)
            or relative in inventory_artifact_paths
        ):
            errors.append(f"{label}: row identity invalid")
            continue
        artifact = artifact_by_path.get(relative)
        if (
            artifact is None
            or artifact.get("source_id") != source_id
            or artifact.get("sha256") != waveform.get("artifact_sha256")
            or artifact.get("size_bytes") != waveform.get("artifact_size_bytes")
        ):
            errors.append(f"{label}: acquired artifact binding differs")
            continue
        path, path_errors = _safe_cache_path(
            cache_root,
            relative,
            Path("sources/r2/development") / source_id,
            label,
        )
        errors.extend(path_errors)
        if path is None or not path.is_file():
            errors.append(f"{label}: waveform missing")
            continue
        try:
            with wave.open(str(path), "rb") as handle:
                observed = {
                    "channels": handle.getnchannels(),
                    "sample_width": handle.getsampwidth(),
                    "sample_rate_hz": handle.getframerate(),
                    "num_samples": handle.getnframes(),
                }
        except (OSError, EOFError, wave.Error) as exc:
            errors.append(f"{label}: waveform unreadable: {exc}")
            continue
        start = waveform.get("eligible_start_sample")
        end = waveform.get("eligible_end_sample")
        if (
            observed["channels"] != 1
            or observed["sample_width"] != 2
            or observed["sample_rate_hz"] != 16000
            or waveform.get("sample_rate_hz") != 16000
            or waveform.get("num_samples") != observed["num_samples"]
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start != 0
            or end != observed["num_samples"]
            or end - start < min(POOLING_MS) * 16
        ):
            errors.append(f"{label}: waveform geometry invalid")
            continue
        waveforms[waveform_id] = waveform
        inventory_artifact_paths.add(relative)
        inventory_sources.add(source_id)
    canonical_waveform_paths = {
        relative
        for relative in artifact_by_path
        if Path(relative).suffix.lower() == ".wav"
    }
    if inventory_artifact_paths != canonical_waveform_paths:
        errors.append("development_acquisition: canonical waveform coverage differs")
    if inventory_sources != set(DEVELOPMENT_SOURCE_IDS):
        errors.append("development_acquisition: waveform source coverage incomplete")
    if receipt.get("waveform_count") != len(waveforms):
        errors.append("development_acquisition: waveform count differs")
    return errors, external_bytes, waveforms


def _read_jsonl(path: Path, label: str) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return [], [f"{label}: unreadable: {exc}"]
    if not lines or any(not line.strip() for line in lines):
        errors.append(f"{label}: empty or blank rows")
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{label}: row {index} invalid JSON: {exc}")
            continue
        if not isinstance(value, dict):
            errors.append(f"{label}: row {index} is not an object")
            continue
        rows.append(value)
    return rows, errors


def _ledger_errors(
    ledger: dict[str, Any],
    contract: dict[str, Any],
    cache_root: Path,
    waveforms: dict[str, dict[str, Any]],
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    errors: list[str] = []
    if ledger.get("schema_version") != 1:
        errors.append("development_coordinate_ledger: unexpected schema version")
    if ledger.get("artifact_role") != "r2_development_coordinate_ledger":
        errors.append("development_coordinate_ledger: unexpected artifact role")
    if ledger.get("experiment_id") != "speaker_representation_scd_v1":
        errors.append("development_coordinate_ledger: unexpected experiment")
    if ledger.get("authority") != AUTHORITY:
        errors.append("development_coordinate_ledger: authority differs")
    if not self_sha256_valid(ledger):
        errors.append("development_coordinate_ledger: invalid self hash")
    errors.extend(
        _reference_matches(
            ledger.get("frozen_inputs"),
            contract.get("frozen_inputs", {}),
            "development_coordinate_ledger",
        )
    )
    acquisition_ref = ledger.get("development_acquisition_receipt")
    expected_acquisition_path = DEVELOPMENT_ACQUISITION_PATH.as_posix()
    if not isinstance(acquisition_ref, dict) or acquisition_ref.get(
        "relative_to_cache_root"
    ) != expected_acquisition_path:
        errors.append("development_coordinate_ledger: acquisition binding differs")
    actual_sources = ledger.get("development_source_ids")
    if actual_sources != list(DEVELOPMENT_SOURCE_IDS):
        errors.append("development_coordinate_ledger: source inventory incomplete")
    counts = ledger.get("extraction_windows_by_context_ms")
    source_counts = ledger.get("extraction_windows_by_source_id")
    expected_keys = {str(value) for value in POOLING_MS}
    if not isinstance(counts, dict) or set(counts) != expected_keys or any(
        not isinstance(value, int) or value <= 0 for value in counts.values()
    ):
        errors.append("development_coordinate_ledger: context counts invalid")
        counts = {}
    if not isinstance(source_counts, dict) or set(source_counts) != set(DEVELOPMENT_SOURCE_IDS) or any(
        not isinstance(value, int) or value <= 0 for value in source_counts.values()
    ):
        errors.append("development_coordinate_ledger: source counts invalid")
        source_counts = {}
    shards = ledger.get("coordinate_shards")
    if not isinstance(shards, list) or not shards:
        errors.append("development_coordinate_ledger: coordinate evidence missing")
        return errors, waveforms
    observed_context = {str(value): 0 for value in POOLING_MS}
    observed_source = {value: 0 for value in DEVELOPMENT_SOURCE_IDS}
    seen_paths: set[str] = set()
    seen_waveforms: set[str] = set()
    total_rows = 0
    for index, shard in enumerate(shards):
        label = f"development_coordinate_ledger.shards[{index}]"
        if not isinstance(shard, dict):
            errors.append(f"{label}: invalid row")
            continue
        source_id = shard.get("source_id")
        waveform_id = shard.get("waveform_id")
        relative = shard.get("relative_to_cache_root")
        waveform = waveforms.get(str(waveform_id))
        if (
            source_id not in DEVELOPMENT_SOURCE_IDS
            or waveform is None
            or waveform.get("source_id") != source_id
            or waveform_id in seen_waveforms
            or not isinstance(relative, str)
        ):
            errors.append(f"{label}: source or path invalid")
            continue
        expected_relative = (
            Path("data/r2/development/coordinates")
            / source_id
            / f"{waveform_id}.jsonl"
        ).as_posix()
        if relative != expected_relative:
            errors.append(f"{label}: non-canonical waveform shard path")
            continue
        prefix = Path("data/r2/development/coordinates") / source_id
        path, path_errors = _safe_cache_path(cache_root, relative, prefix, label)
        errors.extend(path_errors)
        if path is None:
            continue
        if relative in seen_paths:
            errors.append(f"{label}: duplicate path")
            continue
        seen_paths.add(relative)
        seen_waveforms.add(str(waveform_id))
        if not path.is_file():
            errors.append(f"{label}: file missing")
            continue
        if shard.get("size_bytes") != path.stat().st_size:
            errors.append(f"{label}: size mismatch")
        if shard.get("sha256") != sha256_file(path):
            errors.append(f"{label}: sha256 mismatch")
        rows, row_errors = _read_jsonl(path, label)
        errors.extend(row_errors)
        expected_rows = _expected_coordinate_rows(waveform)
        if shard.get("row_count") != len(rows):
            errors.append(f"{label}: row count mismatch")
        if rows != expected_rows:
            errors.append(f"{label}: deterministic coordinate set differs")
            continue
        total_rows += len(rows)
        observed_source[source_id] += len(rows)
        for row in rows:
            observed_context[str(row["context_ms"])] += 1
    if seen_waveforms != set(waveforms):
        errors.append("development_coordinate_ledger: waveform shard coverage differs")
    if observed_context != counts:
        errors.append("development_coordinate_ledger: context recount differs")
    if observed_source != source_counts:
        errors.append("development_coordinate_ledger: source recount differs")
    if ledger.get("total_window_count") != total_rows:
        errors.append("development_coordinate_ledger: total recount differs")
    return errors, waveforms


def _calibration_errors(
    calibration: dict[str, Any],
    contract: dict[str, Any],
    cache_root: Path,
    waveforms: dict[str, dict[str, Any]],
) -> tuple[list[str], dict[str, int]]:
    errors: list[str] = []
    measured: dict[str, int] = {}
    if calibration.get("schema_version") != 1:
        errors.append("cache_calibration: unexpected schema version")
    if calibration.get("artifact_role") != "r2_pooled_cache_calibration":
        errors.append("cache_calibration: unexpected artifact role")
    if calibration.get("experiment_id") != "speaker_representation_scd_v1":
        errors.append("cache_calibration: unexpected experiment")
    if calibration.get("authority") != AUTHORITY:
        errors.append("cache_calibration: authority differs")
    if not self_sha256_valid(calibration):
        errors.append("cache_calibration: invalid self hash")
    errors.extend(
        _reference_matches(
            calibration.get("frozen_inputs"),
            contract.get("frozen_inputs", {}),
            "cache_calibration",
        )
    )
    ledger_ref = calibration.get("development_coordinate_ledger")
    if not isinstance(ledger_ref, dict) or ledger_ref.get(
        "relative_to_cache_root"
    ) != DEVELOPMENT_LEDGER_PATH.as_posix():
        errors.append("cache_calibration: coordinate ledger binding differs")
    rows = calibration.get("models")
    if not isinstance(rows, list) or [
        row.get("model_id") for row in rows if isinstance(row, dict)
    ] != list(MODEL_IDS):
        errors.append("cache_calibration: model inventory differs")
        return errors, measured
    storage = {row["model_id"]: row for row in contract["model_storage_contracts"]}
    for row in rows:
        model_id = row["model_id"]
        label = f"cache_calibration.{model_id}"
        sample_count = row.get("sample_coordinate_count")
        if not isinstance(sample_count, int) or sample_count <= 0:
            errors.append(f"{label}: sample count invalid")
            continue
        artifact = row.get("artifact")
        manifest = row.get("sample_manifest")
        if not isinstance(artifact, dict) or not isinstance(manifest, dict):
            errors.append(f"{label}: artifact references invalid")
            continue
        artifact_path, artifact_errors = _safe_cache_path(
            cache_root,
            str(artifact.get("relative_to_cache_root", "")),
            Path("cache/r2/development/calibration") / model_id,
            f"{label}.artifact",
        )
        manifest_path, manifest_errors = _safe_cache_path(
            cache_root,
            str(manifest.get("relative_to_cache_root", "")),
            Path("cache/r2/development/calibration") / model_id,
            f"{label}.sample_manifest",
        )
        errors.extend(artifact_errors)
        errors.extend(manifest_errors)
        if artifact_path is None or manifest_path is None:
            continue
        if not artifact_path.is_file() or not manifest_path.is_file():
            errors.append(f"{label}: artifact or sample manifest missing")
            continue
        actual_bytes = artifact_path.stat().st_size
        if artifact.get("size_bytes") != actual_bytes or artifact.get(
            "sha256"
        ) != sha256_file(artifact_path):
            errors.append(f"{label}: NPZ identity mismatch")
        if manifest.get("size_bytes") != manifest_path.stat().st_size or manifest.get(
            "sha256"
        ) != sha256_file(manifest_path):
            errors.append(f"{label}: sample manifest identity mismatch")
        manifest_rows, manifest_row_errors = _read_jsonl(manifest_path, label)
        errors.extend(manifest_row_errors)
        manifest_ids = [value.get("coordinate_id") for value in manifest_rows]
        manifest_coordinates_valid = all(
            _coordinate_row_is_expected(value, waveforms)
            for value in manifest_rows
        )
        if (
            len(manifest_rows) != sample_count
            or len(set(manifest_ids)) != sample_count
            or not manifest_coordinates_valid
        ):
            errors.append(f"{label}: sample manifest coordinates invalid")
        storage_row = storage[model_id]
        layer_ids = storage_row["retained_layer_ids"]
        dimension = storage_row["pooled_dimension_per_layer"]
        try:
            with np.load(artifact_path, allow_pickle=False) as arrays:
                if set(arrays.files) != set(layer_ids):
                    errors.append(f"{label}: NPZ layer inventory differs")
                for layer_id in layer_ids:
                    if layer_id not in arrays.files:
                        continue
                    value = arrays[layer_id]
                    if value.shape != (sample_count, dimension):
                        errors.append(f"{label}: {layer_id} shape differs")
                    if value.dtype != np.float32:
                        errors.append(f"{label}: {layer_id} dtype differs")
                    if not np.isfinite(value).all():
                        errors.append(f"{label}: {layer_id} contains non-finite values")
        except (OSError, ValueError, EOFError) as exc:
            errors.append(f"{label}: NPZ unreadable: {exc}")
        serialized = math.ceil(actual_bytes / sample_count)
        if row.get("serialized_file_bytes") != actual_bytes or row.get(
            "serialized_bytes_per_coordinate"
        ) != serialized:
            errors.append(f"{label}: serialized measurement differs")
        raw_bytes = len(layer_ids) * dimension * storage_row["dtype_bytes"]
        measured[model_id] = max(serialized, raw_bytes)
    return errors, measured


def _cache_root_size(cache_root: Path) -> tuple[int, list[str]]:
    total = 0
    errors: list[str] = []
    root = cache_root.resolve()
    try:
        for directory, directory_names, file_names in os.walk(root, followlinks=False):
            base = Path(directory)
            for name in tuple(directory_names):
                path = base / name
                if path.is_symlink():
                    errors.append(f"external_storage: symlinked directory forbidden: {path}")
                    directory_names.remove(name)
            for name in file_names:
                path = base / name
                if path.is_symlink():
                    errors.append(f"external_storage: symlinked file forbidden: {path}")
                    continue
                try:
                    path.resolve().relative_to(root)
                    total += path.stat().st_size
                except (OSError, ValueError) as exc:
                    errors.append(f"external_storage: unreadable path {path}: {exc}")
    except OSError as exc:
        errors.append(f"external_storage: root unreadable: {exc}")
    return total, errors


def _forecast_provenance(requested_argv: tuple[str, ...]) -> dict[str, Any]:
    contract_path = EXPERIMENT_ROOT / FORECAST_CONTRACT_PATH
    contract = load_json(contract_path)
    calculator_path = Path(__file__).resolve()
    return {
        "authority": AUTHORITY,
        "forecast_contract": {
            "path": FORECAST_CONTRACT_PATH.as_posix(),
            "sha256": sha256_file(contract_path),
            "self_sha256": contract["self_sha256"],
        },
        "calculator": {
            "path": "r1_forecast.py",
            "sha256": sha256_file(calculator_path),
        },
        "execution_identity": {
            "run_id": uuid4().hex,
            "process_id": os.getpid(),
            "started_at_utc": datetime.now(UTC).isoformat(),
        },
        "run_provenance": run_provenance(
            EXPERIMENT_ROOT.parents[1],
            requested_argv,
            deterministic_seed=0,
            deterministic_kernels=False,
        ),
    }


def _forecast_provenance_errors(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return ["forecast_provenance: missing"]
    errors: list[str] = []
    contract_path = EXPERIMENT_ROOT / FORECAST_CONTRACT_PATH
    contract = load_json(contract_path)
    if value.get("authority") != AUTHORITY:
        errors.append("forecast_provenance: authority differs")
    if value.get("forecast_contract") != {
        "path": FORECAST_CONTRACT_PATH.as_posix(),
        "sha256": sha256_file(contract_path),
        "self_sha256": contract["self_sha256"],
    }:
        errors.append("forecast_provenance: contract identity differs")
    if value.get("calculator") != {
        "path": "r1_forecast.py",
        "sha256": sha256_file(Path(__file__).resolve()),
    }:
        errors.append("forecast_provenance: calculator identity differs")
    execution = value.get("execution_identity")
    if (
        not isinstance(execution, dict)
        or not isinstance(execution.get("run_id"), str)
        or len(execution["run_id"]) != 32
        or any(character not in "0123456789abcdef" for character in execution["run_id"])
        or not isinstance(execution.get("process_id"), int)
        or not isinstance(execution.get("started_at_utc"), str)
    ):
        errors.append("forecast_provenance: execution identity invalid")
    run = value.get("run_provenance")
    if (
        not isinstance(run, dict)
        or not isinstance(run.get("git_commit"), str)
        or len(run["git_commit"]) != 40
        or not isinstance(run.get("git_dirty"), bool)
        or not isinstance(run.get("git_status_porcelain"), list)
        or not isinstance(run.get("requested_argv"), list)
    ):
        errors.append("forecast_provenance: Git/run identity invalid")
    return errors


def build_forecast(
    technical: dict[str, Any],
    contract: dict[str, Any],
    cache_root: Path,
    acquisition: dict[str, Any] | None,
    ledger: dict[str, Any] | None,
    calibration: dict[str, Any] | None,
    provenance: dict[str, Any],
    input_errors: list[str] | None = None,
) -> dict[str, Any]:
    blockers = list(input_errors or [])
    contract_errors = validate_forecast_contract(contract)
    blockers.extend(contract_errors)
    if contract_errors:
        technical_errors = [
            "technical_validity: external validation skipped because contract is invalid"
        ]
    else:
        technical_errors = validate_technical_validity(technical, cache_root)
    blockers.extend(technical_errors)
    blockers.extend(_forecast_provenance_errors(provenance))
    source_download_bytes = 0
    waveforms: dict[str, dict[str, Any]] = {}
    cache_bytes_per_coordinate: dict[str, int] = {}
    if acquisition is None:
        blockers.append("development_acquisition: missing")
    else:
        (
            acquisition_errors,
            source_download_bytes,
            waveforms,
        ) = _development_acquisition_errors(acquisition, contract, cache_root)
        blockers.extend(acquisition_errors)
    if ledger is None:
        blockers.append("development_coordinate_ledger: missing")
    else:
        ledger_errors, waveforms = _ledger_errors(
            ledger, contract, cache_root, waveforms
        )
        blockers.extend(ledger_errors)
        if acquisition is not None:
            reference = ledger.get("development_acquisition_receipt", {})
            acquisition_path = cache_root / DEVELOPMENT_ACQUISITION_PATH
            blockers.extend(
                _identity_errors(
                    acquisition_path,
                    {
                        "sha256": reference.get("sha256"),
                        "self_sha256": reference.get("self_sha256"),
                    },
                    "development_coordinate_ledger.acquisition_receipt",
                )
            )
    if calibration is None:
        blockers.append("cache_calibration: missing")
    elif ledger is None:
        blockers.append("cache_calibration: coordinate ledger unavailable")
    else:
        calibration_errors, cache_bytes_per_coordinate = _calibration_errors(
            calibration, contract, cache_root, waveforms
        )
        blockers.extend(calibration_errors)
        reference = calibration.get("development_coordinate_ledger", {})
        ledger_path = cache_root / DEVELOPMENT_LEDGER_PATH
        blockers.extend(
            _identity_errors(
                ledger_path,
                {
                    "sha256": reference.get("sha256"),
                    "self_sha256": reference.get("self_sha256"),
                },
                "cache_calibration.coordinate_ledger",
            )
        )
    current_root_bytes, root_errors = _cache_root_size(cache_root)
    blockers.extend(root_errors)
    if blockers:
        return with_self_sha256(
            {
                "schema_version": 2,
                "artifact_role": "r1_full_job_forecast",
                "experiment_id": "speaker_representation_scd_v1",
                "status": "not_ready",
                "blockers": sorted(set(blockers)),
                "forecast_provenance": provenance,
                "external_validation_binding": external_validation_binding(technical)
                if not technical_errors
                else None,
                "current_external_root_bytes": current_root_bytes,
                "forecast_approved": False,
                "full_extraction_enabled": False,
            }
        )
    counts = ledger["extraction_windows_by_context_ms"]
    total_windows = sum(counts.values())
    smoke_by_model = {row["model_id"]: row for row in technical["model_smoke_reports"]}
    multiplier = contract["forecast_method"]["runtime_safety_multiplier"]
    upper_factor = contract["forecast_method"]["runtime_upper_bound_factor"]
    models = []
    for model_id in MODEL_IDS:
        smoke = smoke_by_model[model_id]
        worst_case_seconds_per_window = (
            smoke["single_seconds_per_window"] * upper_factor
        )
        projected_seconds = (
            total_windows * worst_case_seconds_per_window * multiplier
            + smoke["cold_load_seconds"]
        )
        models.append(
            {
                "model_id": model_id,
                "inference_window_count": total_windows,
                "measured_balanced_seconds_per_window": smoke[
                    "single_seconds_per_window"
                ],
                "verified_worst_case_seconds_per_window": worst_case_seconds_per_window,
                "projected_wall_hours": projected_seconds / 3600,
                "projected_cache_bytes": total_windows
                * cache_bytes_per_coordinate[model_id],
                "authoritative_peak_job_memory_bytes": smoke[
                    "authoritative_peak_job_memory_bytes"
                ],
            }
        )
    total_wall_hours = sum(row["projected_wall_hours"] for row in models)
    total_cache_bytes = sum(row["projected_cache_bytes"] for row in models)
    total_external_bytes = current_root_bytes + total_cache_bytes
    ceilings = contract["ceilings"]
    gib = 1024**3
    free_bytes = acquisition["free_bytes_before_download"]
    ceiling_checks = {
        "source_download": source_download_bytes
        <= ceilings["max_source_download_gib"] * gib,
        "pre_download_free_disk": free_bytes
        >= ceilings["min_free_disk_gib_before_download"] * gib,
        "per_model_wall_hours": all(
            row["projected_wall_hours"] <= ceilings["max_per_model_wall_hours"]
            for row in models
        ),
        "total_wall_hours": total_wall_hours <= ceilings["max_total_wall_hours"],
        "peak_memory": all(
            row["authoritative_peak_job_memory_bytes"]
            <= ceilings["max_resident_ram_gib"] * gib
            for row in models
        ),
        "derived_cache": total_cache_bytes <= ceilings["max_derived_cache_gib"] * gib,
        "external_storage": total_external_bytes
        <= ceilings["max_external_storage_gib"] * gib,
    }
    passed = all(ceiling_checks.values())
    return with_self_sha256(
        {
            "schema_version": 2,
            "artifact_role": "r1_full_job_forecast",
            "experiment_id": "speaker_representation_scd_v1",
            "status": "ceiling_pass_candidate" if passed else "ceiling_failed",
            "authority": AUTHORITY,
            "forecast_provenance": provenance,
            "external_validation_binding": external_validation_binding(technical),
            "development_acquisition_receipt": {
                "relative_to_cache_root": DEVELOPMENT_ACQUISITION_PATH.as_posix(),
                "sha256": sha256_file(cache_root / DEVELOPMENT_ACQUISITION_PATH),
                "self_sha256": acquisition["self_sha256"],
            },
            "development_coordinate_ledger": {
                "relative_to_cache_root": DEVELOPMENT_LEDGER_PATH.as_posix(),
                "sha256": sha256_file(cache_root / DEVELOPMENT_LEDGER_PATH),
                "self_sha256": ledger["self_sha256"],
            },
            "cache_calibration": {
                "relative_to_cache_root": CACHE_CALIBRATION_PATH.as_posix(),
                "sha256": sha256_file(cache_root / CACHE_CALIBRATION_PATH),
                "self_sha256": calibration["self_sha256"],
            },
            "development_source_ids": ledger["development_source_ids"],
            "inference_window_count_per_model": total_windows,
            "job_count": total_windows * len(MODEL_IDS),
            "models": models,
            "source_download_bytes": source_download_bytes,
            "free_bytes_before_download": free_bytes,
            "current_external_root_bytes": current_root_bytes,
            "total_projected_wall_hours": total_wall_hours,
            "total_projected_cache_bytes": total_cache_bytes,
            "total_projected_external_storage_bytes": total_external_bytes,
            "ceiling_checks": ceiling_checks,
            "forecast_approved": False,
            "full_extraction_enabled": False,
        }
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    args = parser.parse_args(argv)
    cache_root = args.cache_root.resolve()
    technical = load_json(EXPERIMENT_ROOT / TECHNICAL_VALIDITY_PATH)
    contract = load_json(EXPERIMENT_ROOT / FORECAST_CONTRACT_PATH)
    acquisition, acquisition_load_errors = _load_fixed_cache_document(
        cache_root, DEVELOPMENT_ACQUISITION_PATH, "development_acquisition"
    )
    ledger, ledger_load_errors = _load_fixed_cache_document(
        cache_root, DEVELOPMENT_LEDGER_PATH, "development_coordinate_ledger"
    )
    calibration, calibration_load_errors = _load_fixed_cache_document(
        cache_root, CACHE_CALIBRATION_PATH, "cache_calibration"
    )
    forecast = build_forecast(
        technical,
        contract,
        cache_root,
        acquisition,
        ledger,
        calibration,
        _forecast_provenance(
            tuple(sys.argv) if argv is None else ("r1_forecast", *argv)
        ),
        acquisition_load_errors + ledger_load_errors + calibration_load_errors,
    )
    print(json.dumps(forecast, indent=2, sort_keys=True))
    return 0 if forecast["status"] == "ceiling_pass_candidate" else 2


if __name__ == "__main__":
    raise SystemExit(main())
