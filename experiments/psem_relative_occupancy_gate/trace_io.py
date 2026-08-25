from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from experiments.psem_relative_occupancy_gate.contracts import Trace
from experiments.psem_relative_occupancy_gate.io_utils import (
    canonical_json,
    sha256_file,
    strict_regular_file,
)

TRACE_SCHEMA_VERSION = "psem.relative_occupancy.trace.v1"
TRACE_ARCHIVE_NAMES = (
    "probabilities.npy",
    "frame_start_samples.npy",
    "frame_end_samples.npy",
    "evidence_frontier_samples.npy",
    "slot_alive.npy",
    "state_reset.npy",
    "slot_ids.npy",
    "metadata_json.npy",
)


class TraceIOError(RuntimeError):
    pass


def _array_bytes(values: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, values, allow_pickle=False)
    return buffer.getvalue()


def _archive_bytes(trace: Trace) -> bytes:
    metadata = dict(trace.metadata)
    expected_identity = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "source_id": trace.source_id,
        "family": trace.family,
        "sample_rate_hz": 16000,
        "speaker_slot_ids": list(trace.slot_ids),
    }
    for field, expected in expected_identity.items():
        if metadata.get(field) != expected:
            raise TraceIOError(f"trace metadata identity mismatch: {field}")
    arrays = {
        "probabilities.npy": np.asarray(trace.probabilities, dtype="<f4"),
        "frame_start_samples.npy": np.asarray(trace.frame_start_samples, dtype="<i8"),
        "frame_end_samples.npy": np.asarray(trace.frame_end_samples, dtype="<i8"),
        "evidence_frontier_samples.npy": np.asarray(
            trace.evidence_frontier_samples, dtype="<i8"
        ),
        "slot_alive.npy": np.asarray(trace.slot_alive, dtype=np.bool_),
        "state_reset.npy": np.asarray(trace.state_reset, dtype=np.bool_),
        "slot_ids.npy": np.asarray(trace.slot_ids, dtype=np.str_),
        "metadata_json.npy": np.frombuffer(
            canonical_json(metadata).encode("utf-8"), dtype=np.uint8
        ),
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name in TRACE_ARCHIVE_NAMES:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100600 << 16
            archive.writestr(info, _array_bytes(arrays[name]), compresslevel=9)
    return buffer.getvalue()


def write_trace(path: Path, trace: Trace) -> dict[str, Any]:
    payload = _archive_bytes(trace)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return trace_receipt(path, trace)


def trace_receipt(path: Path, trace: Trace) -> dict[str, Any]:
    return {
        "schema_version": "psem.relative_occupancy.trace_receipt.v1",
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "source_id": trace.source_id,
        "family": trace.family,
        "speaker_slot_ids": list(trace.slot_ids),
        "frame_count": int(trace.probabilities.shape[0]),
        "slot_count": int(trace.probabilities.shape[1]),
        "trace_path": str(path.resolve()),
        "trace_size_bytes": path.stat().st_size,
        "trace_sha256": sha256_file(path),
    }


def _load_array(archive: zipfile.ZipFile, name: str) -> np.ndarray:
    with archive.open(name, "r") as handle:
        return np.load(io.BytesIO(handle.read()), allow_pickle=False)


def load_trace(path: Path) -> Trace:
    path = strict_regular_file(path, "posterior trace")
    try:
        with zipfile.ZipFile(path, "r") as archive:
            names = tuple(info.filename for info in archive.infolist())
            if names != TRACE_ARCHIVE_NAMES or len(set(names)) != len(names):
                raise TraceIOError("trace archive members differ from the frozen schema")
            arrays = {name: _load_array(archive, name) for name in TRACE_ARCHIVE_NAMES}
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise TraceIOError(f"invalid trace archive: {path}") from exc
    try:
        metadata = json.loads(arrays["metadata_json.npy"].astype(np.uint8).tobytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TraceIOError("invalid trace metadata") from exc
    if not isinstance(metadata, dict):
        raise TraceIOError("trace metadata must be an object")
    slot_ids = tuple(str(value) for value in arrays["slot_ids.npy"].tolist())
    trace = Trace(
        source_id=str(metadata.get("source_id", "")),
        family=str(metadata.get("family", "")),
        slot_ids=slot_ids,
        probabilities=np.asarray(arrays["probabilities.npy"], dtype=np.float32),
        frame_start_samples=np.asarray(arrays["frame_start_samples.npy"], dtype=np.int64),
        frame_end_samples=np.asarray(arrays["frame_end_samples.npy"], dtype=np.int64),
        evidence_frontier_samples=np.asarray(
            arrays["evidence_frontier_samples.npy"], dtype=np.int64
        ),
        slot_alive=np.asarray(arrays["slot_alive.npy"], dtype=np.bool_),
        state_reset=np.asarray(arrays["state_reset.npy"], dtype=np.bool_),
        metadata=metadata,
    )
    if metadata.get("schema_version") != TRACE_SCHEMA_VERSION:
        raise TraceIOError("trace schema version mismatch")
    if metadata.get("sample_rate_hz") != 16000:
        raise TraceIOError("trace sample rate mismatch")
    if metadata.get("speaker_slot_ids") != list(trace.slot_ids):
        raise TraceIOError("trace metadata slot identities differ")
    return trace


def validate_trace_receipt(path: Path, receipt: dict[str, Any]) -> Trace:
    trace = load_trace(path)
    expected = trace_receipt(path, trace)
    for field, value in expected.items():
        if receipt.get(field) != value:
            raise TraceIOError(f"trace receipt mismatch: {field}")
    return trace
