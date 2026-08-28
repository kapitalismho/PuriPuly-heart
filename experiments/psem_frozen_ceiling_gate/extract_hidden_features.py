from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.psem_frozen_ceiling_gate.build_ceiling_examples import PACKAGE_ROOT
from experiments.psem_frozen_ceiling_gate.evaluate_ceiling import RESULTS_ROOT
from experiments.psem_frozen_ceiling_gate.experiment_support import (
    canonical_sha256,
    load_json,
    path_has_alias,
    posterior_hidden_trigger_gate_passes,
    sha256_file,
    strict_regular_file,
    write_json,
)

REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]
REPRESENTATION_RECEIPT_PATH = PACKAGE_ROOT / "hidden_representation_receipt.json"
HIDDEN_TRIGGER_PATH = PACKAGE_ROOT / "hidden_trigger_revalidation.json"
POSTERIOR_TRAINING_CONFIG_PATH = PACKAGE_ROOT / "posterior_training_config.json"
DECISION_CONFIG_PATH = PACKAGE_ROOT / "decision_config.json"
HIDDEN_CONFIG_PATH = PACKAGE_ROOT / "hidden_config.json"
SPLIT_PATH = PACKAGE_ROOT / "split_manifest.json"
EXTRACTOR_PATH = Path(__file__).resolve()
RUNTIME_ENVIRONMENT = {
    "TRANSCRIBE_DUMP_DIR": None,
    "TRANSCRIBE_SORTFORMER_STREAM_PRESET": "low_latency",
    "TRANSCRIBE_SORTFORMER_TELEMETRY_PATH": None,
    "TRANSCRIBE_SORTFORMER_HIDDEN_PATH": None,
}


def _probability_dump(directory: Path) -> np.ndarray:
    metadata = load_json(directory / "diar.probs.json")
    shape = tuple(map(int, metadata["shape"]))
    values = np.fromfile(directory / "diar.probs.f32", dtype="<f4")
    if shape[1] != 4 or values.size != int(np.prod(shape)):
        raise ValueError("instrumented posterior dump has invalid geometry")
    result = values.reshape(shape)
    if not np.isfinite(result).all():
        raise ValueError("instrumented posterior dump contains non-finite values")
    return result


def _source_input(source_receipt: dict[str, Any]) -> dict[str, Any]:
    inference_audio = source_receipt.get("inference_audio")
    if inference_audio is not None:
        return {
            **inference_audio,
            "expected_sha256": str(inference_audio["sha256"]),
            "expected_size_bytes": int(inference_audio["size_bytes"]),
        }
    frame_count = int(source_receipt["trace"]["frame_count"])
    return {
        "materialization": "authoritative_frozen_source_waveform",
        "path": str(source_receipt["waveform_path"]),
        "expected_sha256": str(source_receipt["waveform_sha256"]),
        "expected_size_bytes": int(source_receipt["waveform_size_bytes"]),
        "native_frame_count": frame_count,
        "retained_frame_count": frame_count,
        "source_start_sample": int(source_receipt["source_start_sample"]),
        "source_end_sample": int(source_receipt["source_end_sample"]),
        "trailing_zero_sample_count": 0,
    }


def _prepare_source(source_id: str, source: dict[str, Any]) -> dict[str, Any]:
    source_receipt = source["receipt"]
    audio_metadata = _source_input(source_receipt)
    audio = strict_regular_file(Path(str(audio_metadata["path"])), "hidden extraction input")
    trace_metadata = source_receipt["trace"]
    trace_path = strict_regular_file(
        Path(str(trace_metadata["trace_path"])), "authoritative posterior trace"
    )
    input_sha256 = sha256_file(audio)
    trace_sha256 = sha256_file(trace_path)
    if input_sha256 != audio_metadata["expected_sha256"] or audio.stat().st_size != int(
        audio_metadata["expected_size_bytes"]
    ):
        raise ValueError(f"authoritative input identity differs: {source_id}")
    if trace_sha256 != trace_metadata["trace_sha256"] or trace_path.stat().st_size != int(
        trace_metadata["trace_size_bytes"]
    ):
        raise ValueError(f"authoritative trace identity differs: {source_id}")
    return {
        "audio": audio,
        "audio_metadata": audio_metadata,
        "input_sha256": input_sha256,
        "trace_path": trace_path,
        "trace_sha256": trace_sha256,
    }


def _runtime_environment(
    dump_root: Path, telemetry_path: Path, hidden_path: Path
) -> tuple[dict[str, str], dict[str, str]]:
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("TRANSCRIBE_")
    }
    effective = {
        **RUNTIME_ENVIRONMENT,
        "TRANSCRIBE_DUMP_DIR": str(dump_root),
        "TRANSCRIBE_SORTFORMER_TELEMETRY_PATH": str(telemetry_path),
        "TRANSCRIBE_SORTFORMER_HIDDEN_PATH": str(hidden_path),
    }
    environment.update({key: str(value) for key, value in effective.items()})
    return environment, {key: str(value) for key, value in effective.items()}


def _authoritative_sources() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for role in ("dev", "eval"):
        path = PACKAGE_ROOT / "frozen_inputs" / f"{role}_sortformer_model_receipt.json"
        receipt = load_json(path)
        for source in receipt["source_receipts"]:
            source_id = str(source["source_id"])
            if source_id in result:
                raise ValueError(f"duplicate authoritative source: {source_id}")
            result[source_id] = {"role": role, "model_receipt_path": path, "receipt": source}
    frozen = load_json(SPLIT_PATH)
    frozen_ids = {str(value["source_id"]) for value in frozen["sources"]}
    if set(result) != frozen_ids:
        raise ValueError("authoritative source receipts differ from frozen split")
    return result


def _resume_source(
    receipt_path: Path,
    *,
    source_id: str,
    contract_sha256: str,
) -> dict[str, Any] | None:
    if not receipt_path.is_file():
        return None
    receipt = load_json(receipt_path)
    feature_path = Path(str(receipt.get("hidden_features_path", "")))
    stable = (
        receipt.get("schema_version") == "psem.hidden_ceiling.source_extraction.v1"
        and receipt.get("source_id") == source_id
        and receipt.get("status") == "complete"
        and receipt.get("extraction_contract_sha256") == contract_sha256
        and receipt.get("posterior_equivalence", {}).get("status") == "equivalent"
        and feature_path.is_file()
        and sha256_file(feature_path) == receipt.get("hidden_features_sha256")
    )
    return receipt if stable else None


def _run_source(
    source_id: str,
    source: dict[str, Any],
    *,
    representation: dict[str, Any],
    prepared: dict[str, Any],
    bench: Path,
    model: Path,
    output_root: Path,
) -> dict[str, Any]:
    source_receipt = source["receipt"]
    audio_metadata = prepared["audio_metadata"]
    audio = prepared["audio"]
    trace_path = prepared["trace_path"]
    source_root = output_root / source_id
    run_root = source_root / "run"
    dump_root = run_root / "dump"
    source_root.mkdir(parents=True, exist_ok=True)
    dump_root.mkdir(parents=True, exist_ok=True)
    bench_sha256 = sha256_file(bench)
    model_sha256 = sha256_file(model)
    input_sha256 = prepared["input_sha256"]
    trace_sha256 = prepared["trace_sha256"]
    representation_sha256 = sha256_file(REPRESENTATION_RECEIPT_PATH)
    hidden_config_sha256 = sha256_file(HIDDEN_CONFIG_PATH)
    split_sha256 = sha256_file(SPLIT_PATH)
    patch_path = REPOSITORY_ROOT / str(representation["runtime"]["hidden_export_patch_path"])
    patch_sha256 = sha256_file(patch_path)
    environment, effective_environment = _runtime_environment(
        dump_root, run_root / "telemetry.jsonl", run_root / "hidden.f32"
    )
    contract = {
        "source_id": source_id,
        "model_receipt_sha256": sha256_file(source["model_receipt_path"]),
        "input_sha256": input_sha256,
        "authoritative_trace_sha256": trace_sha256,
        "instrumented_bench_sha256": bench_sha256,
        "model_sha256": model_sha256,
        "representation_receipt_sha256": representation_sha256,
        "hidden_config_sha256": hidden_config_sha256,
        "split_manifest_sha256": split_sha256,
        "hidden_export_patch_sha256": patch_sha256,
        "extractor_sha256": sha256_file(EXTRACTOR_PATH),
        "hidden_dimension": int(representation["representation"]["dimension"]),
        "native_frame_count": int(audio_metadata["native_frame_count"]),
        "retained_frame_count": int(audio_metadata["retained_frame_count"]),
        "effective_environment": effective_environment,
    }
    contract_sha256 = canonical_sha256(contract)
    receipt_path = source_root / "receipt.json"
    resumed = _resume_source(
        receipt_path,
        source_id=source_id,
        contract_sha256=contract_sha256,
    )
    if resumed is not None:
        return resumed
    for path in (
        dump_root / "diar.probs.f32",
        dump_root / "diar.probs.json",
        run_root / "hidden.f32",
        run_root / "bench.json",
        run_root / "telemetry.jsonl",
    ):
        if path.exists():
            path.unlink()
    hidden_raw = run_root / "hidden.f32"
    bench_receipt_path = run_root / "bench.json"
    log_path = run_root / "run.log"
    telemetry_path = run_root / "telemetry.jsonl"
    command = [
        str(bench),
        "--model",
        str(model),
        "--sample",
        str(audio),
        "--backend",
        "vulkan",
        "--threads",
        "8",
        "--warmup",
        "0",
        "--iters",
        "1",
        "--json-out",
        str(bench_receipt_path),
    ]
    started = time.perf_counter()
    with log_path.open("wb") as log:
        completed = subprocess.run(
            command,
            cwd=run_root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    wall_seconds = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(f"hidden extraction failed for {source_id}: {log_path}")
    bench_receipt = load_json(bench_receipt_path)
    if (
        not str(bench_receipt["backend"]).lower().startswith("vulkan")
        or bench_receipt["warmup"] != 0
        or bench_receipt["iters"] != 1
    ):
        raise ValueError(f"instrumented backend receipt is invalid: {source_id}")
    probabilities = _probability_dump(dump_root)
    with np.load(trace_path, allow_pickle=False) as cached:
        authoritative = np.asarray(cached["probabilities"], dtype=np.float32)
        frame_starts = np.asarray(cached["frame_start_samples"], dtype=np.int64)
        frame_ends = np.asarray(cached["frame_end_samples"], dtype=np.int64)
        frontiers = np.asarray(cached["evidence_frontier_samples"], dtype=np.int64)
    retained_count = int(audio_metadata["retained_frame_count"])
    native_count = int(audio_metadata["native_frame_count"])
    if probabilities.shape != (native_count, 4) or authoritative.shape != (retained_count, 4):
        raise ValueError(f"posterior frame geometry differs: {source_id}")
    delta = np.abs(probabilities[:retained_count] - authoritative)
    if not np.isfinite(authoritative).all() or not np.isfinite(delta).all():
        raise ValueError(f"posterior equivalence contains non-finite values: {source_id}")
    maximum = float(delta.max(initial=0.0))
    tolerance = float(load_json(HIDDEN_CONFIG_PATH)["posterior_equivalence_atol"])
    if maximum > tolerance:
        raise ValueError(f"posterior equivalence failed for {source_id}: {maximum}")
    hidden_dim = int(representation["representation"]["dimension"])
    hidden = np.fromfile(hidden_raw, dtype="<f4")
    if hidden.size != native_count * hidden_dim:
        raise ValueError(f"hidden frame geometry differs: {source_id}")
    hidden = hidden.reshape(native_count, hidden_dim)[:retained_count]
    if not np.isfinite(hidden).all():
        raise ValueError(f"hidden representation contains non-finite values: {source_id}")
    feature_path = source_root / "hidden_features.npz"
    np.savez_compressed(
        feature_path,
        hidden=hidden,
        frame_start_samples=frame_starts,
        frame_end_samples=frame_ends,
        evidence_frontier_samples=frontiers,
    )
    receipt = {
        "schema_version": "psem.hidden_ceiling.source_extraction.v1",
        "status": "complete",
        "source_id": source_id,
        "old_v2_role": source_receipt["role"],
        "model_receipt_path": str(source["model_receipt_path"].resolve()),
        "model_receipt_sha256": sha256_file(source["model_receipt_path"]),
        "authoritative_trace_path": str(trace_path.resolve()),
        "authoritative_trace_sha256": trace_sha256,
        "input_path": str(audio.resolve()),
        "input_sha256": input_sha256,
        "model_path": str(model.resolve()),
        "model_sha256": model_sha256,
        "instrumented_bench_path": str(bench.resolve()),
        "instrumented_bench_sha256": bench_sha256,
        "input_materialization": audio_metadata["materialization"],
        "backend_requested": "vulkan",
        "backend_resolved": bench_receipt["backend"],
        "precision": "Q8_0",
        "preset": "low_latency",
        "threads": 8,
        "hidden_frame_count": retained_count,
        "hidden_dimension": hidden_dim,
        "extraction_contract": contract,
        "extraction_contract_sha256": contract_sha256,
        "hidden_features_path": str(feature_path.resolve()),
        "hidden_features_sha256": sha256_file(feature_path),
        "posterior_equivalence": {
            "status": "equivalent",
            "absolute_tolerance": tolerance,
            "maximum_absolute_error": maximum,
            "mean_absolute_error": float(delta.mean()),
            "exact_element_fraction": float(np.mean(delta == 0.0)),
            "compared_element_count": int(delta.size),
        },
        "execution": {
            "command": command,
            "effective_environment": effective_environment,
            "wall_seconds": wall_seconds,
            "bench_receipt_path": str(bench_receipt_path.resolve()),
            "bench_receipt_sha256": sha256_file(bench_receipt_path),
            "telemetry_path": str(telemetry_path.resolve()),
            "telemetry_sha256": sha256_file(telemetry_path),
            "log_path": str(log_path.resolve()),
            "log_sha256": sha256_file(log_path),
        },
    }
    write_json(receipt_path, receipt)
    return receipt


def run() -> dict[str, Any]:
    representation = load_json(REPRESENTATION_RECEIPT_PATH)
    trigger = load_json(HIDDEN_TRIGGER_PATH)
    posterior_training_cfg = load_json(POSTERIOR_TRAINING_CONFIG_PATH)
    decision_cfg = load_json(DECISION_CONFIG_PATH)
    if sha256_file(HIDDEN_CONFIG_PATH) != representation["hidden_config"]["sha256"]:
        raise ValueError("hidden config identity mismatch")
    runtime = representation["runtime"]
    if representation.get("opened") is not True:
        raise ValueError("hidden stage is not opened")
    trigger_paths = {
        "gt_result_sha256": RESULTS_ROOT / "gt_causal_action_frontier.json",
        "posterior_causal_result_sha256": RESULTS_ROOT / "fullslot_causal_metrics.json",
        "posterior_noncausal_result_sha256": RESULTS_ROOT / "fullslot_noncausal_metrics.json",
        "source_family_result_sha256": RESULTS_ROOT / "source_family_results.json",
    }
    if (
        trigger.get("schema_version") != "psem.hidden_ceiling.trigger_revalidation.v1"
        or trigger.get("status") != "opened"
        or trigger.get("decision") != "hidden_ceiling_remains_required"
        or trigger.get("representation_receipt_sha256") != sha256_file(REPRESENTATION_RECEIPT_PATH)
        or trigger.get("posterior_training_config_sha256")
        != sha256_file(POSTERIOR_TRAINING_CONFIG_PATH)
        or trigger.get("decision_config_sha256") != sha256_file(DECISION_CONFIG_PATH)
        or not posterior_hidden_trigger_gate_passes(
            trigger,
            posterior_training_cfg,
            tuple(map(str, decision_cfg["posterior_train_fit_required_conditions"])),
            int(decision_cfg["posterior_source_family_min_worse_metrics"]),
        )
        or any(sha256_file(path) != trigger.get(field) for field, path in trigger_paths.items())
    ):
        raise ValueError("hidden trigger artifact identity mismatch")
    bench = Path(str(runtime["instrumented_bench_path"]))
    model = Path(str(runtime["model_path"]))
    output_root = Path(str(representation["extraction"]["external_output_root"])).absolute()
    if (
        not output_root.is_absolute()
        or path_has_alias(output_root)
        or output_root == REPOSITORY_ROOT
        or REPOSITORY_ROOT in output_root.parents
    ):
        raise ValueError("hidden output root must be a non-aliased external path")
    if sha256_file(bench) != runtime["instrumented_bench_sha256"]:
        raise ValueError("instrumented bench identity mismatch")
    if sha256_file(model) != runtime["model_sha256"]:
        raise ValueError("hidden extraction model identity mismatch")
    patch_path = REPOSITORY_ROOT / str(runtime["hidden_export_patch_path"])
    if sha256_file(patch_path) != runtime["hidden_export_patch_sha256"]:
        raise ValueError("hidden export patch identity mismatch")
    sources = _authoritative_sources()
    prepared = {
        source_id: _prepare_source(source_id, source) for source_id, source in sources.items()
    }
    rows = [
        _run_source(
            source_id,
            sources[source_id],
            representation=representation,
            prepared=prepared[source_id],
            bench=bench,
            model=model,
            output_root=output_root,
        )
        for source_id in sorted(sources)
    ]
    aggregate = {
        "schema_version": "psem.hidden_ceiling.extraction_receipt.v1",
        "status": "complete",
        "representation_receipt_sha256": sha256_file(REPRESENTATION_RECEIPT_PATH),
        "hidden_config_sha256": sha256_file(HIDDEN_CONFIG_PATH),
        "split_manifest_sha256": sha256_file(SPLIT_PATH),
        "extractor_sha256": sha256_file(EXTRACTOR_PATH),
        "instrumented_bench_sha256": sha256_file(bench),
        "model_sha256": sha256_file(model),
        "hidden_export_patch_sha256": sha256_file(patch_path),
        "source_count": len(rows),
        "source_receipts": [
            {
                "source_id": value["source_id"],
                "receipt_path": str((output_root / value["source_id"] / "receipt.json").resolve()),
                "receipt_sha256": sha256_file(output_root / value["source_id"] / "receipt.json"),
                "hidden_features_path": value["hidden_features_path"],
                "hidden_features_sha256": value["hidden_features_sha256"],
                "extraction_contract_sha256": value["extraction_contract_sha256"],
                "posterior_equivalence": value["posterior_equivalence"],
            }
            for value in rows
        ],
        "posterior_equivalence": {
            "status": "equivalent",
            "absolute_tolerance": float(
                load_json(HIDDEN_CONFIG_PATH)["posterior_equivalence_atol"]
            ),
            "maximum_absolute_error": max(
                value["posterior_equivalence"]["maximum_absolute_error"] for value in rows
            ),
        },
    }
    write_json(RESULTS_ROOT / "hidden_extraction_receipt.json", aggregate)
    return {
        "source_count": len(rows),
        "maximum_absolute_error": aggregate["posterior_equivalence"]["maximum_absolute_error"],
    }


def main() -> None:
    print(run())


if __name__ == "__main__":
    main()
