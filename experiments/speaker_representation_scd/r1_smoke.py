from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from experiments.speaker_representation_scd.acquire_r1 import _runtime_versions
from experiments.speaker_representation_scd.execution_guard import (
    load_completed_action_receipt,
    validate_worker_execution,
    validate_worker_lease,
)
from experiments.speaker_representation_scd.extraction.common import (
    mean_pool_valid,
    trailing_window,
)
from experiments.speaker_representation_scd.extraction.eres_prepooling import (
    ERes2NetV2PrepoolExtractor,
)
from experiments.speaker_representation_scd.extraction.fixtures import (
    EXPECTED_FIXTURE_MANIFEST_SHA256,
    d0_fixtures,
    fixture_window_contract,
    mutate_future,
)
from experiments.speaker_representation_scd.extraction.ssl import SSLExtractor
from experiments.speaker_representation_scd.provenance import (
    load_json,
    sha256_bytes,
    sha256_file,
    verify_file_identity,
    with_self_sha256,
)
from experiments.speaker_representation_scd.r1_gate import (
    EXPERIMENT_ROOT,
    GATE_PATH,
    R1GateError,
    validated_cache_root,
)
from experiments.speaker_representation_scd.run_provenance import run_provenance


def _write_json(path: Path, document: dict[str, Any]) -> None:
    if path.exists():
        raise R1GateError(f"refusing to overwrite an existing R1 artifact: {path}")
    payload = with_self_sha256(document)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def _array_sha256(value: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(value).tobytes())


def _maximum_delta(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return float("inf")
    if left.size == 0:
        return 0.0
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))


def _batch_single_parity(
    single: np.ndarray,
    batched: np.ndarray,
    contract: dict[str, Any],
) -> dict[str, Any]:
    thresholds = {
        "maximum_absolute_delta": float(contract["batch_single_max_abs_tolerance"]),
        "relative_l2_delta": float(contract["batch_single_relative_l2_tolerance"]),
        "pooled_cosine_distance": float(
            contract["batch_single_pooled_cosine_distance_tolerance"]
        ),
        "pooled_unit_max_abs_delta": float(
            contract["batch_single_pooled_unit_max_abs_tolerance"]
        ),
    }
    if single.shape != batched.shape or single.ndim != 2:
        return {
            "maximum_absolute_delta": float("inf"),
            "relative_l2_delta": float("inf"),
            "pooled_cosine_distance": float("inf"),
            "pooled_unit_max_abs_delta": float("inf"),
            "strict_absolute_passed": False,
            "representation_geometry_passed": False,
            "thresholds": thresholds,
            "rule": contract["batch_single_parity_rule"],
            "passed": False,
        }
    left = single.astype(np.float64)
    right = batched.astype(np.float64)
    difference = left - right
    maximum_absolute_delta = (
        float(np.max(np.abs(difference))) if difference.size else 0.0
    )
    scale = max(float(np.linalg.norm(left)), float(np.linalg.norm(right)))
    relative_l2_delta = float(np.linalg.norm(difference) / scale) if scale > 0 else 0.0
    left_pooled = left.mean(axis=0) if left.shape[0] else np.zeros(left.shape[1])
    right_pooled = right.mean(axis=0) if right.shape[0] else np.zeros(right.shape[1])
    left_norm = float(np.linalg.norm(left_pooled))
    right_norm = float(np.linalg.norm(right_pooled))
    if left_norm == 0 and right_norm == 0:
        pooled_cosine_distance = 0.0
        pooled_unit_max_abs_delta = 0.0
    elif left_norm == 0 or right_norm == 0:
        pooled_cosine_distance = float("inf")
        pooled_unit_max_abs_delta = float("inf")
    else:
        left_unit = left_pooled / left_norm
        right_unit = right_pooled / right_norm
        cosine = float(np.clip(np.dot(left_unit, right_unit), -1.0, 1.0))
        pooled_cosine_distance = 1.0 - cosine
        pooled_unit_max_abs_delta = float(np.max(np.abs(left_unit - right_unit)))
    strict_absolute_passed = maximum_absolute_delta <= thresholds["maximum_absolute_delta"]
    representation_geometry_passed = bool(
        relative_l2_delta <= thresholds["relative_l2_delta"]
        and pooled_cosine_distance <= thresholds["pooled_cosine_distance"]
        and pooled_unit_max_abs_delta <= thresholds["pooled_unit_max_abs_delta"]
    )
    passed = bool(strict_absolute_passed or representation_geometry_passed)
    return {
        "maximum_absolute_delta": maximum_absolute_delta,
        "relative_l2_delta": relative_l2_delta,
        "pooled_cosine_distance": pooled_cosine_distance,
        "pooled_unit_max_abs_delta": pooled_unit_max_abs_delta,
        "strict_absolute_passed": strict_absolute_passed,
        "representation_geometry_passed": representation_geometry_passed,
        "thresholds": thresholds,
        "rule": contract["batch_single_parity_rule"],
        "passed": passed,
    }


def _batch_single_parity_rows(
    singles: tuple[np.ndarray, ...],
    batched: np.ndarray,
    contract: dict[str, Any],
) -> dict[str, Any]:
    if batched.ndim < 1 or batched.shape[0] != len(singles):
        rows = [
            _batch_single_parity(single, np.empty((0, 0)), contract)
            for single in singles
        ]
    else:
        rows = [
            _batch_single_parity(single, batched[index], contract)
            for index, single in enumerate(singles)
        ]
    metrics = (
        "maximum_absolute_delta",
        "relative_l2_delta",
        "pooled_cosine_distance",
        "pooled_unit_max_abs_delta",
    )
    return {
        "row_count": len(rows),
        "rows": [
            {"batch_row_index": index, **row} for index, row in enumerate(rows)
        ],
        **{metric: max(float(row[metric]) for row in rows) for metric in metrics},
        "thresholds": rows[0]["thresholds"] if rows else {},
        "rule": rows[0]["rule"] if rows else contract["batch_single_parity_rule"],
        "passed": bool(rows and all(row["passed"] for row in rows)),
    }


def _model_acquisition_gate_identity_accepted(
    receipt: dict[str, Any],
    gate: dict[str, Any],
    gate_sha256: str,
) -> bool:
    receipt_identity = {
        "r1_gate_sha256": receipt.get("r1_gate_sha256"),
        "r1_gate_self_sha256": receipt.get("r1_gate_self_sha256"),
        "execution_code_manifest_sha256": receipt.get("execution_code_manifest_sha256"),
    }
    current_identity = {
        "r1_gate_sha256": gate_sha256,
        "r1_gate_self_sha256": gate.get("self_sha256"),
        "execution_code_manifest_sha256": gate.get("execution_code", {}).get(
            "manifest_sha256"
        ),
    }
    predecessors = gate.get("receipt_compatibility", {}).get(
        "model_acquisition_predecessors", []
    )
    return bool(receipt_identity == current_identity or receipt_identity in predecessors)


def _peak_ram_gib(process: psutil.Process) -> float:
    memory = process.memory_info()
    value = getattr(memory, "peak_wset", memory.rss)
    return float(value / (1024**3))


def _load_acquisition(cache_root: Path) -> dict[str, Any]:
    path = cache_root / "manifests" / "r1_model_acquisition.json"
    document = load_completed_action_receipt(cache_root, path, "models")
    registry = EXPERIMENT_ROOT / "models" / "source_registry.json"
    if document.get("source_registry_sha256") != sha256_file(registry):
        raise R1GateError("R1 model acquisition receipt uses another source registry")
    gate_path = EXPERIMENT_ROOT / GATE_PATH
    gate = load_json(gate_path)
    if not _model_acquisition_gate_identity_accepted(
        document, gate, sha256_file(gate_path)
    ):
        raise R1GateError("R1 model acquisition receipt uses an unapproved acquisition gate")
    sync_path = cache_root / "manifests" / "r1_environment_sync.json"
    sync_receipt = load_completed_action_receipt(
        cache_root,
        sync_path,
        "sync-environment",
    )
    if document.get("environment_sync_receipt_sha256") != sha256_file(sync_path):
        raise R1GateError("R1 model acquisition receipt uses another environment receipt")
    if document.get("environment_sync_receipt_self_sha256") != sync_receipt.get("self_sha256"):
        raise R1GateError("R1 environment receipt self identity differs")
    if document.get("invocation", {}).get("selected_model_ids") != [
        "eres2netv2-standard-prepool",
        "mhubert-147",
        "unispeech-sat-base-plus",
        "wavlm-base-plus",
    ]:
        raise R1GateError("R1 acquisition receipt does not cover all four models")
    return document


def _model_record(receipt: dict[str, Any], model_id: str) -> dict[str, Any]:
    for record in receipt["models"]:
        if record["model_id"] == model_id:
            registry = load_json(EXPERIMENT_ROOT / "models" / "source_registry.json")
            if model_id == "eres2netv2-standard-prepool":
                contract = registry["eres2netv2"]
                expected = [contract["checkpoint_file"], contract["checkpoint_config"]]
                root = Path(record["checkpoint_root"])
                actual = record["checkpoint_files"]
            else:
                contract = next(
                    model for model in registry["models"] if model["model_id"] == model_id
                )
                expected = contract["required_files"]
                root = Path(record["root"])
                actual = record["files"]
            expected_by_name = {row["path"]: row for row in expected}
            actual_by_name = {Path(row["path"]).name: row for row in actual}
            if len(actual) != len(expected) or set(actual_by_name) != set(expected_by_name):
                raise R1GateError(f"acquisition file inventory differs for {model_id}")
            errors: list[str] = []
            for relative, row in expected_by_name.items():
                path = root / relative
                receipt_row = actual_by_name[relative]
                if Path(receipt_row["path"]).resolve() != path.resolve():
                    errors.append(f"receipt path mismatch: {relative}")
                if receipt_row.get("sha256") != row["sha256"]:
                    errors.append(f"receipt hash mismatch: {relative}")
                errors.extend(verify_file_identity(path, row["sha256"], row.get("size_bytes")))
            if errors:
                raise R1GateError("; ".join(errors))
            return record
    raise R1GateError(f"model was not acquired: {model_id}")


def _make_extractor(model_id: str, record: dict[str, Any]):
    if model_id == "eres2netv2-standard-prepool":
        return ERes2NetV2PrepoolExtractor(
            Path(record["checkpoint_root"]),
            Path(record["source_root"]),
            EXPERIMENT_ROOT / "models" / "source_registry.json",
        )
    return SSLExtractor(model_id, Path(record["root"]))


def _expected_eres_lengths(samples: int) -> dict[str, int]:
    fbank = 0 if samples < 400 else 1 + (samples - 400) // 160
    s2 = (fbank + 1) // 2
    s3 = (s2 + 1) // 2
    s4 = (s3 + 1) // 2
    return {"S1": fbank, "S2": s2, "S3": s3, "S4": s4, "FUSED": s4}


def _expected_ssl_length(samples: int) -> int:
    length = samples
    for kernel, stride in zip((10, 3, 3, 3, 3, 2, 2), (5, 2, 2, 2, 2, 2, 2), strict=True):
        if length < kernel:
            return 0
        length = 1 + (length - kernel) // stride
    return length


def _timestamp_mapping(batch, fixture) -> dict[str, Any]:
    if batch.model_id == "eres2netv2-standard-prepool":
        frame_shift = 160
        frame_count = _expected_eres_lengths(fixture.window_samples)["S1"]
        frame_domain = "official_fbank"
    else:
        frame_shift = 320
        frame_count = next(iter(batch.valid_lengths.values()))[0]
        frame_domain = "ssl_convolution"
    frame_count = int(frame_count)
    window_start = fixture.frontier_sample - fixture.window_samples
    first_support = [window_start, window_start + 400]
    last_start = window_start + frame_shift * (frame_count - 1)
    last_support = [last_start, last_start + 400]
    next_support_end = last_start + frame_shift + 400
    observed_exact = bool(
        np.array_equal(
            batch.observed_source_samples,
            np.asarray([fixture.frontier_sample], dtype=np.int64),
        )
    )
    passed = bool(
        frame_count > 0
        and observed_exact
        and last_support[1] <= fixture.frontier_sample
        and next_support_end > fixture.frontier_sample
    )
    return {
        "frame_domain": frame_domain,
        "frame_length_samples": 400,
        "frame_shift_samples": frame_shift,
        "frame_count": frame_count,
        "window_start_sample": window_start,
        "window_end_sample": fixture.frontier_sample,
        "first_frame_support_samples": first_support,
        "last_frame_support_samples": last_support,
        "next_hypothetical_frame_support_end_sample": next_support_end,
        "representation_availability_source_sample": fixture.frontier_sample,
        "post_context_frame_localization_claim": "not_assigned",
        "observed_source_sample_exact": observed_exact,
        "passed": passed,
    }


def _fixture_check(
    extractor,
    fixture,
    batch_single_contract: dict[str, Any],
) -> dict[str, Any]:
    original = trailing_window(fixture.waveform, fixture.frontier_sample, fixture.window_samples)
    changed = trailing_window(
        mutate_future(fixture), fixture.frontier_sample, fixture.window_samples
    )
    if not np.array_equal(original, changed):
        raise RuntimeError("future mutation altered the supplied trailing window")
    first = extractor.extract([original], [fixture.frontier_sample])
    repeated = extractor.extract([original], [fixture.frontier_sample])
    future = extractor.extract([changed], [fixture.frontier_sample])
    paired = extractor.extract([original, changed], [fixture.frontier_sample] * 2)
    layers: dict[str, Any] = {}
    timestamp_mapping = _timestamp_mapping(first, fixture)
    passed = bool(timestamp_mapping["passed"])
    for layer_id, value in first.layers.items():
        repeat_delta = _maximum_delta(value, repeated.layers[layer_id])
        future_delta = _maximum_delta(value, future.layers[layer_id])
        batch_parity = _batch_single_parity_rows(
            (value[0], future.layers[layer_id][0]),
            paired.layers[layer_id],
            batch_single_contract,
        )
        valid_length = int(first.valid_lengths[layer_id][0])
        pooled = mean_pool_valid(value, first.valid_lengths[layer_id])
        finite = bool(np.isfinite(value).all() and np.isfinite(pooled).all())
        if fixture.fixture_id == "silence":
            norm_status = "zero_or_finite"
        else:
            norm_status = "finite" if bool(np.linalg.norm(pooled[0]) > 0) else "zero"
        expected = (
            _expected_eres_lengths(fixture.window_samples)[layer_id]
            if first.model_id == "eres2netv2-standard-prepool"
            else _expected_ssl_length(fixture.window_samples)
        )
        layer_passed = (
            finite
            and valid_length == expected
            and repeat_delta == 0
            and future_delta == 0
            and batch_parity["passed"]
        )
        passed = passed and layer_passed
        layers[layer_id] = {
            "shape": list(value.shape),
            "valid_length": valid_length,
            "expected_length": expected,
            "finite": finite,
            "pooled_norm_status": norm_status,
            "feature_sha256": _array_sha256(value),
            "repeat_max_abs_delta": repeat_delta,
            "future_mutation_max_abs_delta": future_delta,
            "batch_single_max_abs_delta": batch_parity["maximum_absolute_delta"],
            "batch_single_parity": batch_parity,
            "passed": layer_passed,
        }
    parity = None
    if first.model_id == "eres2netv2-standard-prepool":
        flat = first.layers["FUSED"]
        fused = flat.reshape(flat.shape[0], flat.shape[1], 1024, 10).transpose(0, 2, 3, 1)
        reconstructed = extractor.reconstruct_embedding(fused)
        official = first.official_embedding
        finite = bool(np.isfinite(official).all() and np.isfinite(reconstructed).all())
        delta = _maximum_delta(official, reconstructed) if finite else None
        enough_frames = first.layers["FUSED"].shape[1] >= 2
        parity_passed = (not enough_frames and not finite) or (
            enough_frames and finite and delta is not None and delta <= 1e-6
        )
        passed = passed and parity_passed
        parity = {
            "official_finite": bool(np.isfinite(official).all()),
            "reconstructed_finite": bool(np.isfinite(reconstructed).all()),
            "maximum_absolute_delta": delta,
            "minimum_two_frames": enough_frames,
            "passed": parity_passed,
        }
    return {
        "fixture_id": fixture.fixture_id,
        "scenario_kind": fixture.scenario_kind,
        "speaker_segments": [list(row) for row in fixture.speaker_segments],
        "event_samples": list(fixture.event_samples),
        "waveform_sha256": fixture.waveform_sha256,
        "frontier_sample": fixture.frontier_sample,
        "window_samples": fixture.window_samples,
        "window_sha256": _array_sha256(original),
        "scenario_window_contract": fixture_window_contract(fixture),
        "timestamp_mapping": timestamp_mapping,
        "layers": layers,
        "eres_official_embedding_parity": parity,
        "passed": passed,
    }


def _empirical_coordinate_probe(extractor) -> dict[str, Any]:
    samples = 16000
    frontier = 16000
    coordinates = (0, 399, 400, 8000, 15759, 15760, 15999)
    baseline = np.zeros(samples, dtype=np.float32)
    reference = extractor.extract([baseline], [frontier])
    mutations = {}
    for coordinate in coordinates:
        changed = baseline.copy()
        changed[coordinate] = 0.5
        mutations[coordinate] = extractor.extract([changed], [frontier])
    layer_rows: dict[str, Any] = {}
    passed = True
    for layer_id, reference_values in reference.layers.items():
        expected = (
            _expected_eres_lengths(samples)[layer_id]
            if reference.model_id == "eres2netv2-standard-prepool"
            else _expected_ssl_length(samples)
        )
        rows = []
        any_response = False
        for coordinate in coordinates:
            observed = mutations[coordinate]
            values = observed.layers[layer_id]
            delta = np.max(
                np.abs(values[0].astype(np.float64) - reference_values[0].astype(np.float64)),
                axis=1,
            )
            affected = np.flatnonzero(delta > 1e-7).astype(int).tolist()
            any_response = any_response or bool(affected)
            rows.append(
                {
                    "mutated_source_sample": coordinate,
                    "affected_output_indices": affected,
                    "first_affected_output_index": affected[0] if affected else None,
                    "last_affected_output_index": affected[-1] if affected else None,
                    "maximum_absolute_delta": float(delta.max()) if delta.size else 0.0,
                }
            )
        valid_length = int(reference.valid_lengths[layer_id][0])
        layer_passed = bool(
            valid_length == expected
            and reference_values.shape[1] == expected
            and np.isfinite(reference_values).all()
            and any_response
        )
        passed = passed and layer_passed
        layer_rows[layer_id] = {
            "actual_output_length": valid_length,
            "independent_expected_output_length": expected,
            "source_mutations": rows,
            "frame_localization_interpretation": (
                "empirical_changed_index_span_only; representation_available_at_window_end"
            ),
            "passed": layer_passed,
        }
    return {
        "window_start_sample": 0,
        "window_end_sample": frontier,
        "representation_availability_source_sample": frontier,
        "mutated_source_samples": list(coordinates),
        "layers": layer_rows,
        "passed": passed,
    }


def _benchmark(extractor, fixtures) -> dict[str, Any]:
    selected_layer = "FUSED" if extractor.model_id == "eres2netv2-standard-prepool" else "L6"
    process = psutil.Process()
    single_start = time.perf_counter_ns()
    single_count = 0
    for fixture in fixtures:
        window = trailing_window(fixture.waveform, fixture.frontier_sample, fixture.window_samples)
        for _ in range(10):
            if extractor.model_id == "eres2netv2-standard-prepool":
                extractor.extract([window], [fixture.frontier_sample], [selected_layer])
            else:
                extractor.extract([window], [fixture.frontier_sample], [selected_layer])
            single_count += 1
    single_elapsed = (time.perf_counter_ns() - single_start) / 1_000_000_000
    batch_start = time.perf_counter_ns()
    batch_count = 0
    for fixture in fixtures:
        window = trailing_window(fixture.waveform, fixture.frontier_sample, fixture.window_samples)
        rows = [window] * 10
        frontiers = [fixture.frontier_sample] * 10
        if extractor.model_id == "eres2netv2-standard-prepool":
            extractor.extract(rows, frontiers, [selected_layer])
        else:
            extractor.extract(rows, frontiers, [selected_layer])
        batch_count += len(rows)
    batch_elapsed = (time.perf_counter_ns() - batch_start) / 1_000_000_000
    return {
        "selected_layer": selected_layer,
        "single": {
            "window_count": single_count,
            "elapsed_seconds": single_elapsed,
            "seconds_per_window": single_elapsed / single_count,
        },
        "batch": {
            "window_count": batch_count,
            "elapsed_seconds": batch_elapsed,
            "seconds_per_window": batch_elapsed / batch_count,
        },
        "peak_ram_gib": _peak_ram_gib(process),
    }


def run_smoke(
    model_id: str,
    cache_root: Path,
    requested_argv: tuple[str, ...],
) -> dict[str, Any]:
    validated_cache_root("neural_smoke")
    path = cache_root / "results" / "r1" / "smoke" / f"{model_id}.json"
    execution = validate_worker_execution(cache_root, path)
    if execution.requested_argv != requested_argv:
        raise R1GateError("R1 smoke worker invocation differs from its lease")
    if path.exists():
        raise R1GateError(f"refusing to rerun neural smoke: {path}")
    versions = _runtime_versions()
    receipt = _load_acquisition(cache_root)
    record = _model_record(receipt, model_id)
    process = psutil.Process()
    import torch

    np.random.seed(0)
    torch.manual_seed(0)
    started = datetime.now(UTC).isoformat()
    load_start = time.perf_counter_ns()
    extractor = _make_extractor(model_id, record)
    load_seconds = (time.perf_counter_ns() - load_start) / 1_000_000_000
    gate_path = EXPERIMENT_ROOT / GATE_PATH
    gate = load_json(gate_path)
    fixtures = d0_fixtures()
    fixture_rows = [
        _fixture_check(extractor, fixture, gate["smoke"]) for fixture in fixtures
    ]
    empirical_coordinate_probe = _empirical_coordinate_probe(extractor)
    benchmark = _benchmark(extractor, fixtures)
    peak_ram_gib = max(_peak_ram_gib(process), float(benchmark["peak_ram_gib"]))
    resource_limits = {
        "cpu_threads": 8,
        "max_resident_ram_gib": 24,
        "observed_peak_ram_gib": peak_ram_gib,
        "passed": peak_ram_gib <= 24,
    }
    passed = (
        all(row["passed"] for row in fixture_rows)
        and empirical_coordinate_probe["passed"]
        and resource_limits["passed"]
    )
    acquisition_path = cache_root / "manifests" / "r1_model_acquisition.json"
    report = {
        "schema_version": 1,
        "artifact_role": "r1_model_smoke_report",
        "experiment_id": "speaker_representation_scd_v1",
        "model_id": model_id,
        "supervision_binding": {
            "execution_id": execution.execution_id,
            "expected_receipt_relative_path": (execution.expected_receipt_relative_path),
            "authority": "requires_completed_usage_attestation",
        },
        "started_at_utc": started,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "provenance": {
            "r1_gate_sha256": sha256_file(gate_path),
            "r1_gate_self_sha256": gate["self_sha256"],
            "execution_code_manifest_sha256": gate["execution_code"]["manifest_sha256"],
            "source_registry_sha256": receipt["source_registry_sha256"],
            "model_acquisition_receipt_sha256": sha256_file(acquisition_path),
            "model_acquisition_receipt_self_sha256": receipt["self_sha256"],
        },
        "run_provenance": run_provenance(
            EXPERIMENT_ROOT.parents[1],
            requested_argv,
            deterministic_seed=0,
            deterministic_kernels=torch.are_deterministic_algorithms_enabled(),
        ),
        "model_contract": next(
            contract for contract in receipt["model_contracts"] if contract["model_id"] == model_id
        ),
        "model_artifact_record": record,
        "audio_contract": {
            "sample_rate_hz": 16000,
            "context_mode": "local_trailing_window",
            "availability_frontier": "window_end_sample",
            "fixture_manifest_sha256": EXPECTED_FIXTURE_MANIFEST_SHA256,
        },
        "runtime": {
            "python": os.sys.version.split()[0],
            "executable": str(Path(os.sys.executable).resolve()),
            "packages": versions,
        },
        "parameter_count": extractor.parameter_count,
        "cold_load_seconds": load_seconds,
        "peak_ram_gib_after_validation": peak_ram_gib,
        "resource_limits": resource_limits,
        "fixtures": fixture_rows,
        "empirical_coordinate_probe": empirical_coordinate_probe,
        "benchmark": benchmark,
        "full_extraction_enabled": False,
        "passed": passed,
    }
    _write_json(path, report)
    return load_json(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model", required=True)
    args = parser.parse_args(argv)
    if not args.worker:
        raise R1GateError(
            "direct R1 workers are disabled; use experiments.speaker_representation_scd.r1_execute"
        )
    cache_root = validated_cache_root("neural_smoke")
    requested_argv = validate_worker_lease(cache_root)
    report = run_smoke(args.model, cache_root, requested_argv)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
