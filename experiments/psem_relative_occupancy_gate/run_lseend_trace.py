from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import onnxruntime as ort

from experiments.psem_relative_occupancy_gate.contracts import Trace
from experiments.psem_relative_occupancy_gate.io_utils import (
    PACKAGE_ROOT,
    ExperimentError,
    config,
    load_json,
    lseend_root,
    percentile,
    read_pcm16_mono,
    research_root,
    safe_child,
    safe_output_path,
    sha256_file,
    write_json,
)
from experiments.psem_relative_occupancy_gate.trace_io import (
    TRACE_SCHEMA_VERSION,
    TraceIOError,
    validate_trace_receipt,
    write_trace,
)
from experiments.psem_relative_occupancy_gate.trace_runtime import (
    TraceRuntimeError,
    external_trace_root,
    load_trace_manifest,
    source_window,
    trace_run_key,
    validate_full_trace_geometry,
    validate_trace_location,
)
from experiments.speaker_turn_boundary.frontend import (
    output_frame_available_16k_count,
    output_frame_center_16k,
)
from experiments.speaker_turn_boundary.phase3_ls import LSCaptureEpoch, LSEENDCapture

LSEEND_FAMILY = "ls_eend"
LSEEND_FRAME_SAMPLES = 1600
LSEEND_HALF_SUPPORT_SAMPLES = 800
LSEEND_INPUT_CHUNK_SAMPLES = 512
SLOT_IDS = ("slot-0", "slot-1", "slot-2", "slot-3")


class LSEENDTraceError(RuntimeError):
    pass


def _capture_probabilities(capture: LSCaptureEpoch) -> tuple[np.ndarray, np.ndarray]:
    if (
        len(capture.normal_probs) != len(capture.normal_frontiers)
        or len(capture.normal_probs) != len(capture.frame_wall_ns)
    ):
        raise LSEENDTraceError("LS-EEND normal capture geometry is inconsistent")
    values = [*capture.normal_probs, *capture.tail_probs]
    if not values:
        raise LSEENDTraceError("LS-EEND capture emitted no posterior frames")
    probabilities = np.stack(values, axis=0).astype(np.float32, copy=False)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(SLOT_IDS):
        raise LSEENDTraceError("LS-EEND posterior shape is invalid")
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise LSEENDTraceError("LS-EEND posterior values are invalid")
    frontiers = np.asarray(
        [*capture.normal_frontiers, *([capture.epoch_end_count] * len(capture.tail_probs))],
        dtype=np.int64,
    )
    if np.any(frontiers[1:] < frontiers[:-1]):
        raise LSEENDTraceError("LS-EEND evidence frontier regressed")
    return probabilities, frontiers


def lseend_frame_geometry(
    frame_count: int,
    sample_count: int,
    source_start_sample: int,
    local_frontiers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if local_frontiers.shape != (frame_count,):
        raise LSEENDTraceError("LS-EEND frontier geometry mismatch")
    centers = np.asarray(
        [output_frame_center_16k(index) for index in range(frame_count)],
        dtype=np.int64,
    )
    starts = np.maximum(centers - LSEEND_HALF_SUPPORT_SAMPLES, 0)
    ends = np.minimum(centers + LSEEND_HALF_SUPPORT_SAMPLES, sample_count)
    if np.any(ends <= starts):
        raise LSEENDTraceError("LS-EEND posterior support is outside the source")
    normal_count = int(np.count_nonzero(local_frontiers < sample_count))
    for index in range(normal_count):
        if int(local_frontiers[index]) < output_frame_available_16k_count(index):
            raise LSEENDTraceError("LS-EEND frontier precedes required model input")
    if np.any(local_frontiers < ends):
        raise LSEENDTraceError("LS-EEND frontier precedes posterior support")
    return (
        starts + source_start_sample,
        ends + source_start_sample,
        local_frontiers + source_start_sample,
    )


def _trace(
    *,
    row: dict[str, Any],
    capture: LSCaptureEpoch,
    source_start_sample: int,
    source_end_sample: int,
    sidecar: dict[str, Any],
) -> Trace:
    probabilities, local_frontiers = _capture_probabilities(capture)
    sample_count = source_end_sample - source_start_sample
    starts, ends, frontiers = lseend_frame_geometry(
        probabilities.shape[0], sample_count, source_start_sample, local_frontiers
    )
    state_reset = np.zeros(probabilities.shape[0], dtype=np.bool_)
    state_reset[0] = True
    cfg = config()
    metadata = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "source_id": row["source_id"],
        "family": LSEEND_FAMILY,
        "sample_rate_hz": 16000,
        "speaker_slot_ids": list(SLOT_IDS),
        "role": row["role"],
        "manifest_row_sha256": row["row_sha256"],
        "waveform_sha256": row["waveform_sha256"],
        "source_start_sample": source_start_sample,
        "source_end_sample": source_end_sample,
        "model_sha256": cfg["lseend"]["model_sha256"],
        "model_revision": cfg["lseend"]["revision"],
        "sidecar_sha256": cfg["lseend"]["sidecar_sha256"],
        "backend": cfg["lseend"]["backend"],
        "intra_op_threads": cfg["lseend"]["intra_op_threads"],
        "inter_op_threads": cfg["lseend"]["inter_op_threads"],
        "native_frame_samples": LSEEND_FRAME_SAMPLES,
        "native_frame_support_rule": "output_center_plus_or_minus_800_half_open_clipped",
        "input_chunk_samples_16k": LSEEND_INPUT_CHUNK_SAMPLES,
        "evidence_frontier_rule": "actual_stream_chunk_observed_count_or_epoch_end_tail",
        "frontend": {
            "source_sample_rate_hz": 16000,
            "model_sample_rate_hz": sidecar["sample_rate"],
            "resampler_taps": 63,
            "resampler_center_samples_16k": 31,
            "win_length_8k": sidecar["win_length"],
            "fft_size": sidecar["n_fft"],
            "hop_length_8k": sidecar["hop_length"],
            "context": sidecar["context_recp"],
            "subsampling": sidecar["subsampling"],
            "conv_delay": sidecar["conv_delay"],
            "input_dim": sidecar["input_dim"],
            "feature_type": sidecar["feat_type"],
        },
        "slot_validity_metadata_exposed": False,
        "slot_continuity": "fixed_recurrent_output_columns_within_one_uninterrupted_model_epoch",
        "state_reset_rule": "first_frame_only",
        "adapter_code_sha256": sha256_file(Path(__file__)),
        "trace_io_code_sha256": sha256_file(PACKAGE_ROOT / "trace_io.py"),
        "contracts_code_sha256": sha256_file(PACKAGE_ROOT / "contracts.py"),
        "neutral_capture_code_sha256": sha256_file(
            PACKAGE_ROOT.parent / "speaker_turn_boundary/phase3_ls.py"
        ),
        "frontend_code_sha256": sha256_file(
            PACKAGE_ROOT.parent / "speaker_turn_boundary/frontend.py"
        ),
    }
    return Trace(
        source_id=str(row["source_id"]),
        family=LSEEND_FAMILY,
        slot_ids=SLOT_IDS,
        probabilities=probabilities,
        frame_start_samples=starts,
        frame_end_samples=ends,
        evidence_frontier_samples=frontiers,
        slot_alive=np.ones(probabilities.shape, dtype=np.bool_),
        state_reset=state_reset,
        metadata=metadata,
    )


def _runtime_receipt(capture: LSCaptureEpoch, sample_count: int) -> dict[str, Any]:
    chunk_ms = [value * 1000.0 for value in capture.chunk_wall_seconds]
    if not chunk_ms:
        raise LSEENDTraceError("LS-EEND capture emitted no runtime chunks")
    return {
        "input_chunk_samples_16k": LSEEND_INPUT_CHUNK_SAMPLES,
        "input_chunk_count": len(chunk_ms),
        "wall_seconds": capture.wall_seconds,
        "cpu_seconds": capture.cpu_seconds,
        "wall_rtf": capture.wall_seconds / (sample_count / 16000.0),
        "cpu_rtf": capture.cpu_seconds / (sample_count / 16000.0),
        "chunk_compute_ms": {
            "p50": percentile(chunk_ms, 50),
            "p90": percentile(chunk_ms, 90),
            "p99": percentile(chunk_ms, 99),
            "maximum": max(chunk_ms),
            "sum": sum(chunk_ms),
        },
        "normal_frame_count": len(capture.normal_probs),
        "final_tail_frame_count": len(capture.tail_probs),
    }


def _resume_receipt(
    receipt_path: Path,
    *,
    row: dict[str, Any],
    source_start_sample: int,
    source_end_sample: int,
    source_root: Path,
    model: Path,
    sidecar_path: Path,
) -> dict[str, Any] | None:
    if not receipt_path.is_file():
        return None
    receipt = load_json(receipt_path)
    if not isinstance(receipt, dict):
        return None
    trace_path = Path(str(receipt.get("trace", {}).get("trace_path", "")))
    model_cfg = config()["lseend"]
    runtime = receipt.get("runtime")
    expected_usage = (
        "full_frozen_source"
        if source_start_sample == 0
        and source_end_sample == int(row["source_duration_samples"])
        else "non_authoritative_train_adapter_smoke"
    )
    stable = (
        receipt.get("schema_version")
        == "psem.relative_occupancy.model_source_receipt.v1"
        and receipt.get("trace_schema_version") == TRACE_SCHEMA_VERSION
        and receipt.get("family") == LSEEND_FAMILY
        and receipt.get("source_id") == row["source_id"]
        and receipt.get("role") == row["role"]
        and receipt.get("usage") == expected_usage
        and receipt.get("manifest_row_sha256") == row["row_sha256"]
        and receipt.get("waveform_sha256") == row["waveform_sha256"]
        and receipt.get("source_start_sample") == source_start_sample
        and receipt.get("source_end_sample") == source_end_sample
        and receipt.get("model_sha256") == model_cfg["model_sha256"]
        and receipt.get("sidecar_sha256") == model_cfg["sidecar_sha256"]
        and receipt.get("repository") == model_cfg["repository"]
        and receipt.get("revision") == model_cfg["revision"]
        and receipt.get("backend") == model_cfg["backend"]
        and isinstance(runtime, dict)
        and runtime.get("providers") == [model_cfg["backend"]]
        and receipt.get("intra_op_threads") == model_cfg["intra_op_threads"]
        and receipt.get("inter_op_threads") == model_cfg["inter_op_threads"]
        and Path(str(receipt.get("model_path", ""))).resolve() == model
        and Path(str(receipt.get("sidecar_path", ""))).resolve() == sidecar_path
    )
    if not stable:
        return None
    try:
        trace_path = validate_trace_location(
            trace_path,
            family=LSEEND_FAMILY,
            backend=str(model_cfg["backend"]),
            role=str(row["role"]),
            source_id=str(row["source_id"]),
        )
        if trace_path != source_root / "posterior_trace.npz":
            return None
        trace = validate_trace_receipt(trace_path, receipt["trace"])
        validate_full_trace_geometry(
            trace,
            family=LSEEND_FAMILY,
            source_start_sample=source_start_sample,
            source_end_sample=source_end_sample,
        )
    except (ExperimentError, KeyError, TraceIOError, TraceRuntimeError, OSError):
        return None
    expected_metadata = {
        "source_id": row["source_id"],
        "family": LSEEND_FAMILY,
        "role": row["role"],
        "manifest_row_sha256": row["row_sha256"],
        "waveform_sha256": row["waveform_sha256"],
        "source_start_sample": source_start_sample,
        "source_end_sample": source_end_sample,
        "model_sha256": model_cfg["model_sha256"],
        "model_revision": model_cfg["revision"],
        "sidecar_sha256": model_cfg["sidecar_sha256"],
        "backend": model_cfg["backend"],
        "intra_op_threads": model_cfg["intra_op_threads"],
        "inter_op_threads": model_cfg["inter_op_threads"],
        "adapter_code_sha256": sha256_file(Path(__file__)),
        "trace_io_code_sha256": sha256_file(PACKAGE_ROOT / "trace_io.py"),
        "contracts_code_sha256": sha256_file(PACKAGE_ROOT / "contracts.py"),
        "neutral_capture_code_sha256": sha256_file(
            PACKAGE_ROOT.parent / "speaker_turn_boundary/phase3_ls.py"
        ),
        "frontend_code_sha256": sha256_file(
            PACKAGE_ROOT.parent / "speaker_turn_boundary/frontend.py"
        ),
    }
    if any(trace.metadata.get(field) != value for field, value in expected_metadata.items()):
        return None
    return receipt


def run_traces(
    *,
    manifest: Path,
    role: str,
    research: Path,
    ls_root: Path,
    trace_root: Path | None,
    output: Path,
    source_ids: Sequence[str] | None,
    smoke_samples: int | None,
    resume: bool,
) -> dict[str, Any]:
    cfg = config()
    model_cfg = cfg["lseend"]
    research = research_root(research)
    ls_root = lseend_root(ls_root)
    root = external_trace_root(research, trace_root)
    aggregate_output = safe_output_path(output)
    model = safe_child(ls_root, model_cfg["model_relative_path"], "LS-EEND model")
    sidecar_path = safe_child(
        ls_root, model_cfg["sidecar_relative_path"], "LS-EEND sidecar"
    )
    if sha256_file(model) != model_cfg["model_sha256"]:
        raise LSEENDTraceError("LS-EEND model hash mismatch")
    if sha256_file(sidecar_path) != model_cfg["sidecar_sha256"]:
        raise LSEENDTraceError("LS-EEND sidecar hash mismatch")
    if (
        aggregate_output in {manifest.resolve(), model, sidecar_path}
        or aggregate_output == root
        or root in aggregate_output.parents
    ):
        raise LSEENDTraceError("LS-EEND aggregate output aliases an inference input")
    sidecar = load_json(sidecar_path)
    if not isinstance(sidecar, dict):
        raise LSEENDTraceError("LS-EEND sidecar is invalid")
    expected_sidecar = {
        "sample_rate": 8000,
        "frame_hz": 10.0,
        "real_output_dim": 4,
        "full_output_dim": 6,
        "input_dim": 345,
        "context_recp": 7,
        "subsampling": 10,
        "conv_delay": 9,
    }
    if any(sidecar.get(field) != value for field, value in expected_sidecar.items()):
        raise LSEENDTraceError("LS-EEND sidecar semantic contract mismatch")
    rows = load_trace_manifest(manifest, role=role, source_ids=source_ids)
    if role == "PSEM-STRATEGY-DEV" and smoke_samples is not None:
        raise LSEENDTraceError("DEV inference must cover complete frozen sources")
    if role == "PSEM-STRATEGY-TRAIN" and smoke_samples is None:
        raise LSEENDTraceError("TRAIN adapter inference must be explicitly bounded as smoke")
    pending: list[tuple[dict[str, Any], int, int, str, Path, Path]] = []
    source_receipts: list[dict[str, Any]] = []
    for row in rows:
        source_start, source_end, usage = source_window(row, smoke_samples=smoke_samples)
        run_key = trace_run_key(row, source_start, source_end)
        source_root = safe_child(
            root,
            Path(LSEEND_FAMILY) / str(model_cfg["backend"]) / role / run_key,
            "LS-EEND source trace root",
        )
        source_root.mkdir(parents=True, exist_ok=True)
        receipt_path = source_root / "receipt.json"
        if resume:
            previous = _resume_receipt(
                receipt_path,
                row=row,
                source_start_sample=source_start,
                source_end_sample=source_end,
                source_root=source_root,
                model=model,
                sidecar_path=sidecar_path,
            )
            if previous is not None:
                source_receipts.append(previous)
                print(f"{row['source_id']}: resumed verified LS-EEND trace", flush=True)
                continue
        pending.append((row, source_start, source_end, usage, source_root, receipt_path))
    capture_model: LSEENDCapture | None = None
    load_seconds = 0.0
    if pending:
        capture_model = LSEENDCapture(
            model,
            sidecar,
            checkpoint_variant=str(model_cfg["variant"]),
            intra_op_threads=int(model_cfg["intra_op_threads"]),
            inter_op_threads=int(model_cfg["inter_op_threads"]),
        )
        load_seconds = capture_model.load_seconds
        providers = capture_model._session.get_providers()
        if providers != [model_cfg["backend"]]:
            raise LSEENDTraceError(f"LS-EEND provider mismatch: {providers}")
    for row, source_start, source_end, usage, source_root, receipt_path in pending:
        if capture_model is None:
            raise LSEENDTraceError("LS-EEND capture model is unavailable")
        samples = read_pcm16_mono(Path(str(row["audio_path"])))
        samples = samples[source_start:source_end]
        capture = capture_model.run_case(
            samples,
            case_id=str(row["source_id"]),
            audio_epoch=0,
            chunk_samples=LSEEND_INPUT_CHUNK_SAMPLES,
        )
        trace = _trace(
            row=row,
            capture=capture,
            source_start_sample=source_start,
            source_end_sample=source_end,
            sidecar=sidecar,
        )
        trace_path = source_root / "posterior_trace.npz"
        trace_info = write_trace(trace_path, trace)
        receipt = {
            "schema_version": "psem.relative_occupancy.model_source_receipt.v1",
            "trace_schema_version": TRACE_SCHEMA_VERSION,
            "family": LSEEND_FAMILY,
            "source_id": row["source_id"],
            "role": role,
            "usage": usage,
            "manifest_path": str(manifest.resolve()),
            "manifest_sha256": sha256_file(manifest),
            "manifest_row_sha256": row["row_sha256"],
            "waveform_path": str(Path(str(row["audio_path"])).resolve()),
            "waveform_sha256": row["waveform_sha256"],
            "waveform_size_bytes": row["waveform_size_bytes"],
            "source_start_sample": source_start,
            "source_end_sample": source_end,
            "model_path": str(model),
            "model_sha256": sha256_file(model),
            "sidecar_path": str(sidecar_path),
            "sidecar_sha256": sha256_file(sidecar_path),
            "repository": model_cfg["repository"],
            "revision": model_cfg["revision"],
            "backend": model_cfg["backend"],
            "intra_op_threads": model_cfg["intra_op_threads"],
            "inter_op_threads": model_cfg["inter_op_threads"],
            "runtime": {
                **_runtime_receipt(capture, source_end - source_start),
                "providers": providers,
            },
            "trace": trace_info,
        }
        write_json(receipt_path, receipt)
        source_receipts.append(receipt)
        print(
            f"{row['source_id']}: wrote {trace.probabilities.shape[0]} LS-EEND frames",
            flush=True,
        )
    source_receipts.sort(key=lambda value: str(value["source_id"]))
    aggregate = {
        "schema_version": "psem.relative_occupancy.model_receipt.v1",
        "family": LSEEND_FAMILY,
        "role": role,
        "usage": source_receipts[0]["usage"],
        "variant": model_cfg["variant"],
        "repository": model_cfg["repository"],
        "revision": model_cfg["revision"],
        "model_relative_path": model_cfg["model_relative_path"],
        "model_sha256": model_cfg["model_sha256"],
        "sidecar_relative_path": model_cfg["sidecar_relative_path"],
        "sidecar_sha256": model_cfg["sidecar_sha256"],
        "backend": model_cfg["backend"],
        "intra_op_threads": model_cfg["intra_op_threads"],
        "inter_op_threads": model_cfg["inter_op_threads"],
        "onnxruntime_version": ort.__version__,
        "model_load_seconds_for_new_sources": load_seconds,
        "native_frame_ms": model_cfg["native_frame_ms"],
        "slot_count": model_cfg["slot_count"],
        "slot_validity_metadata": model_cfg["slot_validity_metadata"],
        "source_count": len(source_receipts),
        "source_ids": [receipt["source_id"] for receipt in source_receipts],
        "source_receipts": source_receipts,
        "eval_status": "sealed",
    }
    write_json(aggregate_output, aggregate)
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--role",
        choices=("PSEM-STRATEGY-TRAIN", "PSEM-STRATEGY-DEV"),
        required=True,
    )
    parser.add_argument("--research-root", type=Path)
    parser.add_argument("--lseend-root", type=Path)
    parser.add_argument("--trace-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-id", action="append")
    parser.add_argument("--smoke-samples", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        result = run_traces(
            manifest=args.manifest.resolve(),
            role=args.role,
            research=research_root(args.research_root),
            ls_root=lseend_root(args.lseend_root),
            trace_root=args.trace_root.resolve() if args.trace_root else None,
            output=args.output.resolve(),
            source_ids=args.source_id,
            smoke_samples=args.smoke_samples,
            resume=args.resume,
        )
    except (LSEENDTraceError, TraceRuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "source_count": result["source_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
