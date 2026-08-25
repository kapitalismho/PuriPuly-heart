from __future__ import annotations

import wave
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from experiments.psem_relative_occupancy_gate.contracts import Trace
from experiments.psem_relative_occupancy_gate.io_utils import (
    CONFIG_PATH,
    REPOSITORY_ROOT,
    canonical_sha256,
    load_jsonl,
    path_has_alias,
    safe_child,
    sha256_file,
    strict_regular_file,
)
from experiments.psem_relative_occupancy_gate.provenance import load_frozen_dataset
from experiments.speaker_turn_boundary.frontend import output_frame_center_16k


class TraceRuntimeError(RuntimeError):
    pass


def backend_resolution_matches(requested: str, resolved: Any) -> bool:
    requested_normalized = requested.strip().lower()
    resolved_normalized = str(resolved).strip().lower()
    if resolved_normalized == requested_normalized:
        return True
    if requested_normalized not in {"vulkan", "cuda", "metal"}:
        return False
    suffix = resolved_normalized.removeprefix(requested_normalized)
    return resolved_normalized.startswith(requested_normalized) and suffix.isdecimal()


def external_trace_root(research: Path, explicit: Path | None = None) -> Path:
    resolved_research = research.resolve()
    if explicit is None:
        result = safe_child(
            resolved_research,
            "results/issue97/psem_relative_occupancy_gate",
            "posterior trace root",
        )
    else:
        unresolved = explicit.absolute()
        if path_has_alias(unresolved):
            raise TraceRuntimeError("posterior trace root must not use symlinks or junctions")
        result = unresolved.resolve()
        if result == resolved_research or resolved_research not in result.parents:
            raise TraceRuntimeError("posterior trace root must be a child of the research root")
    if result == REPOSITORY_ROOT or REPOSITORY_ROOT in result.parents:
        raise TraceRuntimeError("posterior trace root must remain outside the repository")
    result.mkdir(parents=True, exist_ok=True)
    return result


def validate_trace_location(
    path: Path,
    *,
    family: str,
    backend: str,
    role: str,
    source_id: str,
) -> Path:
    resolved = strict_regular_file(path, "posterior trace")
    suffix = Path(family) / backend / role / source_id / "posterior_trace.npz"
    actual = tuple(value.casefold() for value in resolved.parts[-len(suffix.parts) :])
    expected = tuple(value.casefold() for value in suffix.parts)
    if actual != expected:
        raise TraceRuntimeError("posterior trace path does not match its family/backend role")
    root = resolved.parents[len(suffix.parts) - 1]
    if root == REPOSITORY_ROOT or REPOSITORY_ROOT in root.parents:
        raise TraceRuntimeError("posterior trace root must remain outside the repository")
    return resolved


def validate_full_trace_geometry(
    trace: Trace,
    *,
    family: str,
    source_start_sample: int,
    source_end_sample: int,
) -> None:
    frame_count = int(trace.frame_end_samples.size)
    if frame_count == 0:
        raise TraceRuntimeError("full-source trace contains no frames")
    if trace.metadata.get("source_start_sample") != source_start_sample or trace.metadata.get(
        "source_end_sample"
    ) != source_end_sample:
        raise TraceRuntimeError("full-source trace metadata window mismatch")
    if np.any(trace.frame_start_samples[1:] != trace.frame_end_samples[:-1]):
        raise TraceRuntimeError("full-source trace frame supports are not contiguous")
    if np.count_nonzero(trace.state_reset) != 1 or not bool(trace.state_reset[0]):
        raise TraceRuntimeError("full-source trace model epoch is not uninterrupted")
    sample_count = source_end_sample - source_start_sample
    if family == "streaming_sortformer":
        starts = source_start_sample + np.arange(frame_count, dtype=np.int64) * 1280
        ends = np.minimum(starts + 1280, source_end_sample)
        if frame_count != (sample_count + 1279) // 1280 or int(ends[-1]) != source_end_sample:
            raise TraceRuntimeError("Sortformer full-source frame count is incomplete")
    elif family == "ls_eend":
        centers = np.asarray(
            [output_frame_center_16k(index) for index in range(frame_count)], dtype=np.int64
        )
        starts = source_start_sample + np.maximum(centers - 800, 0)
        ends = source_start_sample + np.minimum(centers + 800, sample_count)
        if int(ends[-1]) <= source_end_sample - 1600:
            raise TraceRuntimeError("LS-EEND full-source tail coverage is incomplete")
    else:
        raise TraceRuntimeError(f"unsupported trace family: {family}")
    if not np.array_equal(trace.frame_start_samples, starts) or not np.array_equal(
        trace.frame_end_samples, ends
    ):
        raise TraceRuntimeError("full-source trace frame geometry mismatch")
    if int(trace.evidence_frontier_samples[-1]) != source_end_sample:
        raise TraceRuntimeError("full-source trace evidence frontier does not reach source end")


def _validate_waveform(row: dict[str, Any]) -> None:
    unresolved = Path(str(row.get("audio_path", "")))
    if path_has_alias(unresolved) or not unresolved.is_file():
        raise TraceRuntimeError(f"waveform is unavailable: {row.get('source_id')}")
    path = unresolved.resolve()
    if path.stat().st_size != int(row.get("waveform_size_bytes", -1)):
        raise TraceRuntimeError(f"waveform size mismatch: {row.get('source_id')}")
    if sha256_file(path) != row.get("waveform_sha256"):
        raise TraceRuntimeError(f"waveform identity mismatch: {row.get('source_id')}")
    with wave.open(str(path), "rb") as reader:
        valid = (
            reader.getframerate() == 16000
            and reader.getnchannels() == 1
            and reader.getsampwidth() == 2
            and reader.getnframes() == int(row.get("source_duration_samples", -1))
        )
    if not valid:
        raise TraceRuntimeError(f"waveform geometry mismatch: {row.get('source_id')}")


def load_trace_manifest(
    path: Path,
    *,
    role: str,
    source_ids: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    if not rows:
        raise TraceRuntimeError("trace manifest is empty")
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        source_id = row.get("source_id")
        if not isinstance(source_id, str) or not source_id or source_id in by_id:
            raise TraceRuntimeError("trace manifest source identities are invalid")
        observed_hash = row.get("row_sha256")
        unhashed = dict(row)
        unhashed.pop("row_sha256", None)
        if observed_hash != canonical_sha256(unhashed):
            raise TraceRuntimeError(f"trace manifest row hash mismatch: {source_id}")
        eval_status_valid = (
            row.get("eval_status") == "opened_once"
            and isinstance(row.get("eval_selection_sha256"), str)
            and bool(row.get("eval_selection_sha256"))
            and isinstance(row.get("eval_authorization_sha256"), str)
            and bool(row.get("eval_authorization_sha256"))
            if role == "PSEM-STRATEGY-EVAL"
            else row.get("eval_status") == "sealed"
            and row.get("eval_selection_sha256") is None
            and row.get("eval_authorization_sha256") is None
        )
        if (
            row.get("schema_version") != "psem.relative_occupancy.manifest_row.v1"
            or row.get("role") != role
            or row.get("sample_rate_hz") != 16000
            or row.get("config_sha256") != sha256_file(CONFIG_PATH)
            or not eval_status_valid
        ):
            raise TraceRuntimeError(f"trace manifest contract mismatch: {source_id}")
        _validate_waveform(row)
        by_id[source_id] = row
    dataset = load_frozen_dataset()
    allowed = set(dataset.source_ids(role))
    if not set(by_id) <= allowed:
        raise TraceRuntimeError("trace manifest contains a source outside its frozen role")
    selected_ids = tuple(sorted(set(source_ids or by_id)))
    if not selected_ids or not set(selected_ids) <= set(by_id):
        raise TraceRuntimeError("requested trace source is absent from the manifest")
    return [by_id[source_id] for source_id in selected_ids]


def trace_run_key(row: dict[str, Any], source_start_sample: int, source_end_sample: int) -> str:
    source_id = str(row["source_id"])
    if source_start_sample == 0 and source_end_sample == int(row["source_duration_samples"]):
        return source_id
    return f"{source_id}__samples_{source_start_sample}_{source_end_sample}"


def waveform_slice(row: dict[str, Any], source_start_sample: int, source_end_sample: int) -> bytes:
    if not 0 <= source_start_sample < source_end_sample <= int(row["source_duration_samples"]):
        raise TraceRuntimeError("waveform slice is outside the frozen source")
    path = Path(str(row["audio_path"]))
    with wave.open(str(path), "rb") as reader:
        reader.setpos(source_start_sample)
        payload = reader.readframes(source_end_sample - source_start_sample)
    expected_bytes = 2 * (source_end_sample - source_start_sample)
    if len(payload) != expected_bytes:
        raise TraceRuntimeError("waveform slice is incomplete")
    return payload


def write_waveform_slice(
    path: Path,
    row: dict[str, Any],
    source_start_sample: int,
    source_end_sample: int,
) -> None:
    payload = waveform_slice(row, source_start_sample, source_end_sample)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(payload)


def source_window(row: dict[str, Any], *, smoke_samples: int | None) -> tuple[int, int, str]:
    duration = int(row["source_duration_samples"])
    if smoke_samples is None:
        return 0, duration, "full_frozen_source"
    if smoke_samples <= 0 or smoke_samples > duration:
        raise TraceRuntimeError("smoke sample count is outside the frozen source")
    return 0, smoke_samples, "non_authoritative_train_adapter_smoke"
