from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from experiments.psem_relative_occupancy_gate.contracts import Trace
from experiments.psem_relative_occupancy_gate.io_utils import (
    PACKAGE_ROOT,
    canonical_sha256,
    config,
    load_json,
    sha256_file,
    strict_regular_file,
)
from experiments.psem_relative_occupancy_gate.run_sortformer_trace import (
    SORTFORMER_CHUNK_FRAMES,
    SortformerTraceError,
    sortformer_inference_audio_geometry,
    validate_sortformer_telemetry_receipt,
)
from experiments.psem_relative_occupancy_gate.trace_io import validate_trace_receipt
from experiments.psem_relative_occupancy_gate.trace_runtime import (
    TraceRuntimeError,
    backend_resolution_matches,
    validate_full_trace_geometry,
    validate_trace_location,
)


class ModelRunError(RuntimeError):
    pass


def _ends_with(path: Path, relative: str | Path) -> bool:
    suffix = Path(relative)
    return tuple(value.casefold() for value in path.parts[-len(suffix.parts) :]) == tuple(
        value.casefold() for value in suffix.parts
    )


def _root_before_suffix(path: Path, relative: str | Path) -> Path:
    suffix = Path(relative)
    if not _ends_with(path, suffix):
        raise ModelRunError(f"pinned path suffix mismatch: {relative}")
    return path.parents[len(suffix.parts) - 1]


def _sortformer_inference_audio(
    source_receipt: dict[str, Any],
    inference: dict[str, Any],
    source_duration_samples: int,
) -> tuple[Path, dict[str, Any]]:
    binding = source_receipt.get("inference_audio")
    if not isinstance(binding, dict):
        raise ModelRunError("Sortformer inference audio binding is missing")
    path = strict_regular_file(
        Path(str(binding.get("path", ""))), "Sortformer inference audio"
    )
    geometry = sortformer_inference_audio_geometry(source_duration_samples)
    expected = {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        **geometry,
        "source_start_sample": 0,
        "source_end_sample": source_duration_samples,
    }
    if binding != expected or inference.get("audio") != binding:
        raise ModelRunError("Sortformer inference audio binding mismatch")
    return path, binding


def load_model_traces(
    receipt_path: Path,
    *,
    manifest_path: Path,
    manifest: Sequence[dict[str, Any]],
    family: str,
    role: str,
    eval_access_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Trace]]:
    receipt = load_json(receipt_path)
    if not isinstance(receipt, dict):
        raise ModelRunError("model receipt must be an object")
    source_ids = sorted(str(value["source_id"]) for value in manifest)
    eval_status = "opened_once" if role == "PSEM-STRATEGY-EVAL" else "sealed"
    if (
        receipt.get("schema_version") != "psem.relative_occupancy.model_receipt.v1"
        or receipt.get("family") != family
        or receipt.get("role") != role
        or receipt.get("eval_status") != eval_status
        or receipt.get("source_count") != len(source_ids)
        or receipt.get("source_ids") != source_ids
    ):
        raise ModelRunError(f"model receipt contract mismatch: {family}")
    if role == "PSEM-STRATEGY-EVAL":
        selection_hashes = {value.get("eval_selection_sha256") for value in manifest}
        if (
            len(selection_hashes) != 1
            or receipt.get("eval_selection_sha256") != next(iter(selection_hashes))
            or eval_access_path is None
            or receipt.get("eval_access_receipt_sha256")
            != sha256_file(eval_access_path.resolve())
        ):
            raise ModelRunError(f"model EVAL access binding mismatch: {family}")
    elif eval_access_path is not None:
        raise ModelRunError(f"unexpected EVAL access receipt: {family}")
    cfg = config()
    family_config = cfg["sortformer" if family == "streaming_sortformer" else "lseend"]
    expected_family = {
        "model_sha256": family_config["model_sha256"],
        "slot_count": family_config["slot_count"],
        "native_frame_ms": family_config["native_frame_ms"],
        "slot_validity_metadata": family_config["slot_validity_metadata"],
    }
    if family == "streaming_sortformer":
        expected_family.update(
            {
                "model_repository": family_config["model_repository"],
                "model_revision": family_config["model_revision"],
                "model_filename": family_config["model_filename"],
                "source_repository": family_config["source_repository"],
                "source_commit": family_config["source_commit"],
                "telemetry_patch_sha256": family_config["telemetry_patch_sha256"],
                "bench_relative_path": family_config["bench_relative_path"],
                "bench_sha256": family_config["bench_sha256"],
                "backend": family_config["backend"],
                "threads": family_config["threads"],
                "preset": family_config["preset"],
                "chunk_audio_ms": family_config["chunk_audio_ms"],
                "recorded_algorithmic_lookahead_ms": family_config[
                    "algorithmic_lookahead_ms"
                ],
            }
        )
    else:
        expected_family.update(
            {
                "variant": family_config["variant"],
                "repository": family_config["repository"],
                "revision": family_config["revision"],
                "model_relative_path": family_config["model_relative_path"],
                "sidecar_relative_path": family_config["sidecar_relative_path"],
                "sidecar_sha256": family_config["sidecar_sha256"],
                "backend": family_config["backend"],
                "intra_op_threads": family_config["intra_op_threads"],
                "inter_op_threads": family_config["inter_op_threads"],
            }
        )
    if any(receipt.get(field) != value for field, value in expected_family.items()):
        raise ModelRunError(f"model receipt pin mismatch: {family}")
    source_receipts = receipt.get("source_receipts")
    if not isinstance(source_receipts, list) or len(source_receipts) != len(source_ids):
        raise ModelRunError(f"model source receipts are incomplete: {family}")
    rows = {str(value["source_id"]): value for value in manifest}
    traces: dict[str, Trace] = {}
    trace_roots: set[Path] = set()
    manifest_sha256 = sha256_file(manifest_path)
    for source_receipt in source_receipts:
        if not isinstance(source_receipt, dict):
            raise ModelRunError(f"model source receipt is invalid: {family}")
        source_id = str(source_receipt.get("source_id", ""))
        if source_id not in rows or source_id in traces:
            raise ModelRunError(f"model source identity mismatch: {family}:{source_id}")
        row = rows[source_id]
        expected = {
            "schema_version": "psem.relative_occupancy.model_source_receipt.v1",
            "family": family,
            "role": role,
            "manifest_sha256": manifest_sha256,
            "manifest_row_sha256": row["row_sha256"],
            "waveform_sha256": row["waveform_sha256"],
            "source_start_sample": 0,
            "source_end_sample": int(row["source_duration_samples"]),
            "model_sha256": family_config["model_sha256"],
            "trace_schema_version": cfg["trace_schema_version"],
            "usage": "full_frozen_source",
        }
        if family == "streaming_sortformer":
            expected.update(
                {
                    "bench_relative_path": family_config["bench_relative_path"],
                    "bench_sha256": family_config["bench_sha256"],
                    "source_repository": family_config["source_repository"],
                    "source_commit": family_config["source_commit"],
                    "telemetry_patch_sha256": family_config[
                        "telemetry_patch_sha256"
                    ],
                    "backend": family_config["backend"],
                    "threads": family_config["threads"],
                    "preset": family_config["preset"],
                }
            )
        else:
            expected.update(
                {
                    "sidecar_sha256": family_config["sidecar_sha256"],
                    "repository": family_config["repository"],
                    "revision": family_config["revision"],
                    "backend": family_config["backend"],
                    "intra_op_threads": family_config["intra_op_threads"],
                    "inter_op_threads": family_config["inter_op_threads"],
                }
            )
        if any(source_receipt.get(field) != value for field, value in expected.items()):
            raise ModelRunError(f"model source binding mismatch: {family}:{source_id}")
        waveform_path = strict_regular_file(
            Path(str(source_receipt.get("waveform_path", ""))), "frozen waveform"
        )
        if (
            waveform_path != Path(str(row["audio_path"])).resolve()
            or sha256_file(waveform_path) != row["waveform_sha256"]
        ):
            raise ModelRunError(f"model waveform binding mismatch: {family}:{source_id}")
        if family == "streaming_sortformer":
            resolved_backend = source_receipt.get("backend_resolved")
            inference = source_receipt.get("inference")
            bench = inference.get("bench") if isinstance(inference, dict) else None
            if not isinstance(inference, dict):
                raise ModelRunError(
                    f"model inference receipt mismatch: {family}:{source_id}"
                )
            inference_audio_path, inference_audio = _sortformer_inference_audio(
                source_receipt, inference, int(row["source_duration_samples"])
            )
            bench_path = strict_regular_file(
                Path(str(source_receipt.get("bench_path", ""))), "Sortformer bench"
            )
            model_path = strict_regular_file(
                Path(str(source_receipt.get("model_path", ""))), "Sortformer model"
            )
            raw_bench_path = strict_regular_file(
                Path(str(inference.get("bench_path", ""))) if isinstance(inference, dict) else Path(),
                "Sortformer raw bench receipt",
            )
            telemetry_receipt = source_receipt.get("telemetry")
            telemetry_path = strict_regular_file(
                Path(str(telemetry_receipt.get("path", "")))
                if isinstance(telemetry_receipt, dict)
                else Path(),
                "Sortformer telemetry",
            )
            inferred_research_root = _root_before_suffix(
                bench_path, family_config["bench_relative_path"]
            )
            if (
                not backend_resolution_matches(family_config["backend"], resolved_backend)
                or not isinstance(bench, dict)
                or bench.get("backend") != resolved_backend
                or bench.get("iters") != 1
                or bench.get("warmup") != 0
                or inference.get("backend_resolved") != resolved_backend
                or sha256_file(bench_path) != family_config["bench_sha256"]
                or sha256_file(model_path) != family_config["model_sha256"]
                or model_path
                != (
                    inferred_research_root
                    / "models/r8"
                    / family_config["model_filename"]
                ).resolve()
                or inference.get("bench_sha256") != sha256_file(raw_bench_path)
                or load_json(raw_bench_path) != bench
                or bench.get("model_path") != str(model_path)
                or bench.get("sample_path") != str(inference_audio_path)
                or inference.get("raw_probability_frame_count")
                != inference_audio["native_frame_count"]
                or inference.get("retained_probability_frame_count")
                != inference_audio["retained_frame_count"]
                or inference.get("command")
                != [
                    str(bench_path),
                    "--model",
                    str(model_path),
                    "--sample",
                    str(inference_audio_path),
                    "--backend",
                    str(family_config["backend"]),
                    "--threads",
                    str(family_config["threads"]),
                    "--warmup",
                    "0",
                    "--iters",
                    "1",
                    "--json-out",
                    str(raw_bench_path),
                ]
            ):
                raise ModelRunError(
                    f"model backend resolution mismatch: {family}:{source_id}"
                )
        else:
            runtime = source_receipt.get("runtime")
            model_path = strict_regular_file(
                Path(str(source_receipt.get("model_path", ""))), "LS-EEND model"
            )
            sidecar_path = strict_regular_file(
                Path(str(source_receipt.get("sidecar_path", ""))), "LS-EEND sidecar"
            )
            inferred_lseend_root = _root_before_suffix(
                model_path, family_config["model_relative_path"]
            )
            if (
                not isinstance(runtime, dict)
                or runtime.get("providers") != [family_config["backend"]]
                or sha256_file(model_path) != family_config["model_sha256"]
                or sha256_file(sidecar_path) != family_config["sidecar_sha256"]
                or model_path
                != (inferred_lseend_root / family_config["model_relative_path"]).resolve()
                or sidecar_path
                != (inferred_lseend_root / family_config["sidecar_relative_path"]).resolve()
            ):
                raise ModelRunError(f"model provider/root mismatch: {family}:{source_id}")
        trace_receipt = source_receipt.get("trace")
        if not isinstance(trace_receipt, dict):
            raise ModelRunError(f"model trace receipt is missing: {family}:{source_id}")
        try:
            trace_path = validate_trace_location(
                Path(str(trace_receipt.get("trace_path", ""))),
                family=family,
                backend=str(family_config["backend"]),
                role=role,
                source_id=source_id,
            )
            trace = validate_trace_receipt(trace_path, trace_receipt)
            if family == "streaming_sortformer" and raw_bench_path != (
                trace_path.parent / "run" / "bench.json"
            ):
                raise TraceRuntimeError("Sortformer raw bench receipt root mismatch")
            if family == "streaming_sortformer" and telemetry_path != (
                trace_path.parent / "run" / "telemetry.jsonl"
            ):
                raise TraceRuntimeError("Sortformer telemetry root mismatch")
            if family == "streaming_sortformer" and inference_audio_path != (
                trace_path.parent / "input.wav"
            ):
                raise TraceRuntimeError("Sortformer inference audio root mismatch")
            trace_root = trace_path.parents[4]
            if family == "streaming_sortformer" and (
                trace_root == inferred_research_root
                or inferred_research_root not in trace_root.parents
            ):
                raise TraceRuntimeError("posterior trace root is outside the research root")
            if family == "streaming_sortformer":
                validate_sortformer_telemetry_receipt(
                    telemetry_path,
                    telemetry_receipt,
                    expected_chunks=int(inference_audio["native_frame_count"])
                    // SORTFORMER_CHUNK_FRAMES,
                )
            validate_full_trace_geometry(
                trace,
                family=family,
                source_start_sample=0,
                source_end_sample=int(row["source_duration_samples"]),
            )
        except (SortformerTraceError, TraceRuntimeError) as exc:
            raise ModelRunError(f"model trace coverage mismatch: {family}:{source_id}") from exc
        trace_roots.add(trace_path.parents[4])
        expected_metadata = {
            "source_id": source_id,
            "family": family,
            "role": role,
            "source_start_sample": 0,
            "source_end_sample": int(row["source_duration_samples"]),
            "waveform_sha256": row["waveform_sha256"],
            "manifest_row_sha256": row["row_sha256"],
            "model_sha256": family_config["model_sha256"],
            "adapter_code_sha256": sha256_file(
                PACKAGE_ROOT
                / (
                    "run_sortformer_trace.py"
                    if family == "streaming_sortformer"
                    else "run_lseend_trace.py"
                )
            ),
            "contracts_code_sha256": sha256_file(PACKAGE_ROOT / "contracts.py"),
            "trace_io_code_sha256": sha256_file(PACKAGE_ROOT / "trace_io.py"),
        }
        if family == "streaming_sortformer":
            expected_metadata.update(
                {
                    "model_revision": family_config["model_revision"],
                    "source_commit": family_config["source_commit"],
                    "telemetry_patch_sha256": family_config[
                        "telemetry_patch_sha256"
                    ],
                    "inference_audio_path": inference_audio["path"],
                    "inference_audio_sha256": inference_audio["sha256"],
                    "inference_audio_sample_count": inference_audio["sample_count"],
                    "inference_audio_source_sample_count": inference_audio[
                        "source_sample_count"
                    ],
                    "inference_audio_trailing_zero_sample_count": inference_audio[
                        "trailing_zero_sample_count"
                    ],
                    "inference_audio_native_frame_count": inference_audio[
                        "native_frame_count"
                    ],
                    "inference_audio_retained_frame_count": inference_audio[
                        "retained_frame_count"
                    ],
                    "inference_audio_materialization": inference_audio[
                        "materialization"
                    ],
                    "bench_sha256": family_config["bench_sha256"],
                    "bench_relative_path": family_config["bench_relative_path"],
                    "backend": family_config["backend"],
                    "threads": family_config["threads"],
                    "preset": family_config["preset"],
                }
            )
        else:
            expected_metadata.update(
                {
                    "model_revision": family_config["revision"],
                    "sidecar_sha256": family_config["sidecar_sha256"],
                    "backend": family_config["backend"],
                    "intra_op_threads": family_config["intra_op_threads"],
                    "inter_op_threads": family_config["inter_op_threads"],
                    "neutral_capture_code_sha256": sha256_file(
                        PACKAGE_ROOT.parent / "speaker_turn_boundary/phase3_ls.py"
                    ),
                    "frontend_code_sha256": sha256_file(
                        PACKAGE_ROOT.parent / "speaker_turn_boundary/frontend.py"
                    ),
                }
            )
        if any(trace.metadata.get(field) != value for field, value in expected_metadata.items()):
            raise ModelRunError(f"model trace metadata mismatch: {family}:{source_id}")
        traces[source_id] = trace
    if sorted(traces) != source_ids:
        raise ModelRunError(f"model trace source set mismatch: {family}")
    if len(trace_roots) != 1:
        raise ModelRunError(f"model traces span multiple external roots: {family}")
    receipt_identity = {
        "receipt_path": str(receipt_path.resolve()),
        "receipt_sha256": sha256_file(receipt_path),
        "trace_root": str(next(iter(trace_roots))),
        "source_trace_sha256": {
            source_id: next(
                value["trace"]["trace_sha256"]
                for value in source_receipts
                if value["source_id"] == source_id
            )
            for source_id in source_ids
        },
    }
    receipt_identity["identity_sha256"] = canonical_sha256(receipt_identity)
    return receipt_identity, traces
