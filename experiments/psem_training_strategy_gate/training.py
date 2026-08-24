from __future__ import annotations

import json
import math
import os
import pickle
import platform
import random
import shutil
import socket
import stat
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import psutil
import torch
import torchaudio

from experiments.psem_training_strategy_gate.augmentation import apply_augmentation
from experiments.psem_training_strategy_gate.evaluator import (
    CandidateEvent,
    PredictionScore,
    ReferenceEvent,
    event_average_precision,
    eventize,
)
from experiments.psem_training_strategy_gate.losses import (
    LossAccumulator,
    LossWeights,
    collate_targets,
    compute_losses,
    loss_statistics,
)
from experiments.psem_training_strategy_gate.models import (
    ARMS,
    ModelContractError,
    PSEMModel,
    build_model,
    build_optimizer,
    model_root,
    optimizer_groups,
    parameter_inventory,
)
from experiments.psem_training_strategy_gate.preflight import (
    CONFIG_PATH,
    CONTRACT_PATH,
    DATA_DIR,
    EXPERIMENT_ROOT,
    LABEL_GENERATOR_PATH,
    SOURCE_MANIFEST_PATH,
    SOURCE_REGISTRY_PATH,
    ExperimentPreflightError,
    PreflightPaths,
    canonical_sha256,
    load_json,
    require_passing_preflight,
    sha256_file,
)
from experiments.psem_training_strategy_gate.sampling import (
    DEV_ROLE,
    EVAL_ROLE,
    MAXIMUM_EPOCHS,
    OFFICIAL_EFFECTIVE_BATCH_SIZE,
    SPLIT_MANIFEST_PATH,
    TRAIN_ROLE,
    WINDOWS_PER_EPOCH,
    RuntimeSession,
    SamplingContractError,
    load_runtime_sessions,
    load_sampling_rows,
    load_waveform_window,
    target_for_row,
)
from experiments.psem_training_strategy_gate.targets import (
    FUTURE_SAMPLES,
    WINDOW_SAMPLES,
    WindowTargets,
    build_window_targets,
    valid_center_samples,
)

OFFICIAL_RUNS = tuple((arm, seed) for arm in ARMS for seed in (7301, 7302))
RUNS_DIRECTORY = "official_runs"
PLAN_FILENAME = "official_run_plan.json"
LOCK_FILENAME = ".official_run.lock"


class TrainingContractError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class OfficialTrainingSettings:
    maximum_epochs: int
    early_stopping_patience: int
    effective_batch_size: int
    checkpoint_interval_batches: int
    checkpoint_matching_collar_ms: int
    warmup_fraction: float
    gradient_clip_norm: float


@dataclass(slots=True)
class TrainingAccumulator:
    batch_count: int = 0
    window_count: int = 0
    total_loss_sum: float = 0.0
    handoff_loss_sum: float = 0.0
    state_loss_sum: float = 0.0
    relation_loss_sum: float = 0.0
    handoff_valid_count: int = 0
    state_valid_count: int = 0
    relation_valid_count: int = 0
    gradient_norm_max: float = 0.0
    elapsed_seconds: float = 0.0
    peak_rss_bytes: int = 0

    def update(
        self,
        losses: Mapping[str, torch.Tensor | int],
        *,
        windows: int,
        gradient_norm: float,
        elapsed_seconds: float,
        rss_bytes: int,
    ) -> None:
        values = {
            "total": float(losses["total"]),
            "handoff": float(losses["handoff"]),
            "state": float(losses["state"]),
            "relation": float(losses["relation"]),
        }
        if any(not math.isfinite(value) or value <= 0.0 for value in values.values()):
            raise TrainingContractError("official training produced a non-positive loss")
        if not math.isfinite(gradient_norm) or gradient_norm <= 0.0:
            raise TrainingContractError("official training produced an invalid gradient norm")
        self.batch_count += 1
        self.window_count += windows
        self.total_loss_sum += values["total"]
        self.handoff_loss_sum += values["handoff"]
        self.state_loss_sum += values["state"]
        self.relation_loss_sum += values["relation"]
        self.handoff_valid_count += int(losses["handoff_valid_count"])
        self.state_valid_count += int(losses["state_valid_count"])
        self.relation_valid_count += int(losses["relation_valid_count"])
        self.gradient_norm_max = max(self.gradient_norm_max, gradient_norm)
        self.elapsed_seconds += elapsed_seconds
        self.peak_rss_bytes = max(self.peak_rss_bytes, rss_bytes)

    def summary(self) -> dict[str, Any]:
        if self.batch_count <= 0 or self.window_count <= 0:
            raise TrainingContractError("official training epoch has no completed batches")
        return {
            "batch_count": self.batch_count,
            "window_count": self.window_count,
            "mean_losses": {
                "total": self.total_loss_sum / self.batch_count,
                "handoff": self.handoff_loss_sum / self.batch_count,
                "state": self.state_loss_sum / self.batch_count,
                "relation": self.relation_loss_sum / self.batch_count,
            },
            "valid_counts": {
                "handoff": self.handoff_valid_count,
                "state": self.state_valid_count,
                "relation": self.relation_valid_count,
            },
            "gradient_norm_max": self.gradient_norm_max,
            "elapsed_seconds": self.elapsed_seconds,
            "peak_rss_bytes": self.peak_rss_bytes,
        }


class _ExclusiveRunLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: Any = None

    def __enter__(self) -> _ExclusiveRunLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+b")
        self.handle.seek(0)
        if self.path.stat().st_size == 0:
            self.handle.write(b"0")
            self.handle.flush()
            self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.handle.close()
            self.handle = None
            raise TrainingContractError(
                "another official training command holds the run lock"
            ) from error
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.handle is None:
            return
        self.handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()
        self.handle = None


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _descriptor(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise TrainingContractError(f"required official artifact is absent: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _descriptor_valid(value: Any) -> bool:
    if not isinstance(value, Mapping) or set(value) != {"path", "size_bytes", "sha256"}:
        return False
    try:
        path = Path(str(value["path"])).resolve()
        return (
            str(path) == value["path"]
            and path.is_file()
            and isinstance(value["size_bytes"], int)
            and not isinstance(value["size_bytes"], bool)
            and path.stat().st_size == value["size_bytes"]
            and sha256_file(path) == value["sha256"]
        )
    except (OSError, TypeError, ValueError):
        return False


def _source_rows() -> dict[str, Mapping[str, Any]]:
    rows = {}
    for line in SOURCE_MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, Mapping) or not isinstance(value.get("source_id"), str):
            raise TrainingContractError("source manifest inventory is invalid")
        source_id = value["source_id"]
        if source_id in rows:
            raise TrainingContractError("source manifest identities are not unique")
        rows[source_id] = value
    return rows


def _role_by_source() -> dict[str, str]:
    split = load_json(SPLIT_MANIFEST_PATH)
    assignments = split.get("assignments") if isinstance(split, Mapping) else None
    sources = assignments.get("sources") if isinstance(assignments, Mapping) else None
    if not isinstance(sources, list):
        raise TrainingContractError("frozen split source inventory is absent")
    result = {}
    for row in sources:
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("source_id"), str)
            or row.get("role") not in {TRAIN_ROLE, DEV_ROLE, EVAL_ROLE}
            or row["source_id"] in result
        ):
            raise TrainingContractError("frozen split source inventory is invalid")
        result[row["source_id"]] = row["role"]
    return result


def _bound_descriptor(path: Path, *, size_bytes: int | None, sha256: str) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise TrainingContractError(f"required official artifact is absent: {resolved}")
    actual_size = resolved.stat().st_size
    if size_bytes is not None and actual_size != size_bytes:
        raise TrainingContractError(f"official input size differs: {resolved}")
    return {"path": str(resolved), "size_bytes": actual_size, "sha256": sha256}


def _materialize_exact_file(
    descriptor: Mapping[str, Any],
    destination: Path,
) -> dict[str, Any]:
    if not _descriptor_valid(descriptor):
        raise TrainingContractError("official material source differs from its bound identity")
    source = Path(str(descriptor["path"])).resolve()
    destination = destination.resolve()
    if destination.is_file():
        observed = _descriptor(destination)
        if (
            observed["size_bytes"] != descriptor["size_bytes"]
            or observed["sha256"] != descriptor["sha256"]
        ):
            raise TrainingContractError("official material snapshot differs from its bound identity")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        try:
            with source.open("rb") as source_handle, temporary.open("xb") as target_handle:
                shutil.copyfileobj(source_handle, target_handle, length=1024 * 1024)
                target_handle.flush()
                os.fsync(target_handle.fileno())
            observed = _descriptor(temporary)
            if (
                observed["size_bytes"] != descriptor["size_bytes"]
                or observed["sha256"] != descriptor["sha256"]
            ):
                raise TrainingContractError(
                    "official material source changed while creating its snapshot"
                )
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    current_mode = destination.stat().st_mode
    writable = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    if current_mode & writable:
        destination.chmod(current_mode & ~writable)
    return _descriptor(destination)


def _materialize_inputs(
    paths: PreflightPaths,
    arm: str,
    output_root: Path,
) -> tuple[dict[str, str], list[Mapping[str, Any]], list[dict[str, Any]]]:
    if paths.corpus_root is None:
        raise TrainingContractError("official corpus root is absent")
    material_root = (output_root / "material_inputs").resolve()
    material_cache_root = material_root / "cache"
    material_corpus_root = material_root / "corpus"
    model_rows = []
    for descriptor in _model_input_descriptors(paths, arm):
        if paths.cache_root is None:
            raise TrainingContractError("official model cache root is absent")
        source = Path(str(descriptor["path"])).resolve()
        relative = source.relative_to(paths.cache_root.resolve())
        model_rows.append(_materialize_exact_file(descriptor, material_cache_root / relative))
    source_rows = []
    for row in _source_input_descriptors(paths):
        source = Path(str(row["waveform"]["path"])).resolve()
        relative = source.relative_to(paths.corpus_root.resolve())
        source_rows.append(
            {
                **row,
                "waveform": _materialize_exact_file(
                    row["waveform"], material_corpus_root / relative
                ),
            }
        )
    return (
        {
            "cache_root": str(material_cache_root.resolve()),
            "corpus_root": str(material_corpus_root.resolve()),
        },
        model_rows,
        source_rows,
    )


def _runtime_input_descriptors(output_root: Path) -> dict[str, Mapping[str, Any]]:
    paths = {
        "sampling_manifest": output_root / "manifests" / "sampling_manifest.jsonl",
        "sampling_summary": output_root / "manifests" / "sampling_summary.json",
        "augmentation_manifest": output_root / "manifests" / "augmentation_manifest.json",
        "model_graphs": output_root / "audits" / "model_graphs.json",
        "parameter_inventory": output_root / "audits" / "parameter_inventory.json",
        "gradient_canary": output_root / "audits" / "gradient_canary.json",
        "weight_update_canary": output_root / "audits" / "weight_update_canary.json",
        "arm_comparability": output_root / "audits" / "arm_comparability.json",
        "metric_contract": output_root / "audits" / "metric_contract.json",
    }
    return {name: _descriptor(path) for name, path in paths.items()}


def _model_input_descriptors(paths: PreflightPaths, arm: str) -> list[Mapping[str, Any]]:
    if arm == "SCRATCH-PSEM":
        return []
    if paths.cache_root is None:
        raise TrainingContractError("official model cache root is absent")
    registry = load_json(SOURCE_REGISTRY_PATH)
    models = registry.get("models") if isinstance(registry, Mapping) else None
    rows = [
        row
        for row in models or []
        if isinstance(row, Mapping) and row.get("model_id") == "wavlm-base-plus"
    ]
    if len(rows) != 1 or not isinstance(rows[0].get("required_files"), list):
        raise TrainingContractError("pinned WavLM input inventory is invalid")
    root = (
        paths.cache_root.resolve() / "models" / "wavlm-base-plus" / str(rows[0]["revision"])
    ).resolve()
    result = []
    for row in rows[0]["required_files"]:
        if not isinstance(row, Mapping):
            raise TrainingContractError("pinned WavLM file inventory is invalid")
        relative = Path(str(row.get("path", "")))
        path = (root / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(root):
            raise TrainingContractError("pinned WavLM file path is invalid")
        result.append(
            _bound_descriptor(
                path,
                size_bytes=row.get("size_bytes"),
                sha256=str(row.get("sha256", "")),
            )
        )
    return result


def _source_input_descriptors(paths: PreflightPaths) -> list[dict[str, Any]]:
    if paths.corpus_root is None:
        raise TrainingContractError("official source roots are absent")
    source_rows = _source_rows()
    roles = _role_by_source()
    split = load_json(SPLIT_MANIFEST_PATH)
    split_rows = {
        row["source_id"]: row
        for row in split["assignments"]["sources"]
        if isinstance(row, Mapping) and isinstance(row.get("source_id"), str)
    }
    result = []
    for source_id, role in sorted(roles.items()):
        if role == EVAL_ROLE:
            continue
        row = source_rows.get(source_id)
        if row is None:
            raise TrainingContractError("frozen source is absent from the source manifest")
        waveform = (paths.corpus_root.resolve() / str(row.get("audio_ref", ""))).resolve()
        split_row = split_rows.get(source_id)
        if not waveform.is_relative_to(paths.corpus_root.resolve()) or split_row is None:
            raise TrainingContractError("official source input path escapes its root")
        result.append(
            {
                "source_id": source_id,
                "role": role,
                "annotation_sha256": row.get("annotation_sha256"),
                "reference_sha256": split_row.get("reference_sha256"),
                "waveform": _bound_descriptor(
                    waveform,
                    size_bytes=row.get("waveform_size_bytes"),
                    sha256=str(row.get("waveform_sha256", "")),
                ),
            }
        )
    return result


def _input_descriptors(inputs: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    descriptors = []

    def collect(value: Any) -> None:
        if isinstance(value, Mapping) and set(value) == {"path", "size_bytes", "sha256"}:
            descriptors.append(value)
            return
        if isinstance(value, Mapping):
            for item in value.values():
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(inputs)
    return descriptors


def _file_snapshots(inputs: Mapping[str, Any]) -> list[dict[str, Any]]:
    descriptors = _input_descriptors(inputs)
    result = []
    for descriptor in descriptors:
        path = Path(str(descriptor["path"])).resolve()
        stat = path.stat()
        result.append(
            {
                "path": str(path),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "ctime_ns": stat.st_ctime_ns,
                "device": stat.st_dev,
                "inode": stat.st_ino,
                "sha256": descriptor["sha256"],
            }
        )
    if len({row["path"] for row in result}) != len(result):
        raise TrainingContractError("official input descriptor paths are not unique")
    return sorted(result, key=lambda row: row["path"])


def _assert_file_snapshots(
    identity: Mapping[str, Any], paths: Sequence[Path] | None = None
) -> None:
    snapshots = identity.get("file_snapshots")
    inputs = identity.get("inputs")
    if not isinstance(snapshots, list) or not isinstance(inputs, Mapping):
        raise TrainingContractError("official input snapshots are absent")
    descriptor_paths = sorted(
        str(Path(str(descriptor["path"])).resolve()) for descriptor in _input_descriptors(inputs)
    )
    if (
        len(descriptor_paths) != len(set(descriptor_paths))
        or len(snapshots) != len(descriptor_paths)
        or any(
            not isinstance(row, Mapping)
            or set(row)
            != {"path", "size_bytes", "mtime_ns", "ctime_ns", "device", "inode", "sha256"}
            or not isinstance(row.get("path"), str)
            or not _strict_int(row.get("size_bytes"))
            or not _strict_int(row.get("mtime_ns"))
            or not _strict_int(row.get("ctime_ns"))
            or not _strict_int(row.get("device"))
            or not _strict_int(row.get("inode"))
            or not isinstance(row.get("sha256"), str)
            or len(row["sha256"]) != 64
            for row in snapshots
        )
        or [row["path"] for row in snapshots] != descriptor_paths
    ):
        raise TrainingContractError("official input snapshot inventory is invalid")
    by_path = {row["path"]: row for row in snapshots}
    selected = list(by_path) if paths is None else [str(path.resolve()) for path in paths]
    for value in selected:
        expected = by_path.get(value)
        if expected is None:
            raise TrainingContractError("official input snapshot inventory is invalid")
        descriptor = next(
            (
                row
                for row in _input_descriptors(inputs)
                if str(Path(str(row["path"])).resolve()) == value
            ),
            None,
        )
        if descriptor is None or expected["sha256"] != descriptor["sha256"]:
            raise TrainingContractError("official input snapshot identity is invalid")
        try:
            stat = Path(value).stat()
        except OSError as error:
            raise TrainingContractError("official input disappeared during execution") from error
        if (
            stat.st_size != expected["size_bytes"]
            or stat.st_mtime_ns != expected["mtime_ns"]
            or stat.st_ctime_ns != expected["ctime_ns"]
            or stat.st_dev != expected["device"]
            or stat.st_ino != expected["inode"]
        ):
            raise TrainingContractError("official input changed during execution")


def _strict_int(value: Any, *, minimum: int = 0) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _finite_number(
    value: Any,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    numeric = float(value)
    return (
        math.isfinite(numeric)
        and (minimum is None or numeric >= minimum)
        and (maximum is None or numeric <= maximum)
    )


def _timestamp_valid(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() == UTC.utcoffset(parsed)


def _timing_valid(value: Any, expected_window_count: int) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value)
        == {"window_count", "per_window_seconds_p50", "per_window_seconds_p95"}
        and _strict_int(value.get("window_count"), minimum=1)
        and value.get("window_count") == expected_window_count
        and _finite_number(value.get("per_window_seconds_p50"), minimum=0.0)
        and value["per_window_seconds_p50"] > 0.0
        and _finite_number(value.get("per_window_seconds_p95"), minimum=0.0)
        and value["per_window_seconds_p95"] >= value["per_window_seconds_p50"]
    )


def _best_valid(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value)
        == {"epoch", "event_average_precision", "total_loss", "collar_ms", "checkpoint"}
        and _strict_int(value.get("epoch"), minimum=1)
        and value["epoch"] <= MAXIMUM_EPOCHS
        and _finite_number(value.get("event_average_precision"), minimum=0.0, maximum=1.0)
        and _finite_number(value.get("total_loss"), minimum=0.0)
        and value.get("collar_ms") == 250
        and _descriptor_valid(value.get("checkpoint"))
    )


def _training_summary_valid(value: Any) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "batch_count",
        "window_count",
        "mean_losses",
        "valid_counts",
        "gradient_norm_max",
        "elapsed_seconds",
        "peak_rss_bytes",
    }:
        return False
    losses = value.get("mean_losses")
    counts = value.get("valid_counts")
    return (
        value.get("batch_count") == WINDOWS_PER_EPOCH // OFFICIAL_EFFECTIVE_BATCH_SIZE
        and value.get("window_count") == WINDOWS_PER_EPOCH
        and isinstance(losses, Mapping)
        and set(losses) == {"total", "handoff", "state", "relation"}
        and all(_finite_number(item, minimum=0.0) for item in losses.values())
        and isinstance(counts, Mapping)
        and set(counts) == {"handoff", "state", "relation"}
        and all(_strict_int(item, minimum=1) for item in counts.values())
        and _finite_number(value.get("gradient_norm_max"), minimum=0.0)
        and value["gradient_norm_max"] > 0.0
        and _finite_number(value.get("elapsed_seconds"), minimum=0.0)
        and value["elapsed_seconds"] > 0.0
        and _strict_int(value.get("peak_rss_bytes"), minimum=1)
    )


def _history_row_valid(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value)
        == {
            "epoch",
            "train",
            "dev_event_average_precision",
            "dev_total_loss",
            "checkpoint_matching_collar_ms",
            "improved",
            "dev_metrics",
        }
        and _strict_int(value.get("epoch"), minimum=1)
        and value["epoch"] <= MAXIMUM_EPOCHS
        and _training_summary_valid(value.get("train"))
        and _finite_number(value.get("dev_event_average_precision"), minimum=0.0, maximum=1.0)
        and _finite_number(value.get("dev_total_loss"), minimum=0.0)
        and value.get("checkpoint_matching_collar_ms") == 250
        and type(value.get("improved")) is bool
        and _descriptor_valid(value.get("dev_metrics"))
    )


def _accumulator_valid(value: Any) -> bool:
    if not isinstance(value, Mapping) or set(value) != set(
        TrainingAccumulator.__dataclass_fields__
    ):
        return False
    integer_fields = {
        "batch_count",
        "window_count",
        "handoff_valid_count",
        "state_valid_count",
        "relation_valid_count",
        "peak_rss_bytes",
    }
    float_fields = set(value) - integer_fields
    if not all(_strict_int(value.get(field)) for field in integer_fields) or not all(
        _finite_number(value.get(field), minimum=0.0) for field in float_fields
    ):
        return False
    batch_count = value["batch_count"]
    if value["window_count"] != batch_count * OFFICIAL_EFFECTIVE_BATCH_SIZE:
        return False
    if batch_count == 0:
        return all(value[field] == 0 for field in set(value) - {"batch_count", "window_count"})
    return all(
        value[field] > 0 for field in integer_fields - {"batch_count", "window_count"}
    ) and all(
        value[field] > 0.0
        for field in {
            "total_loss_sum",
            "handoff_loss_sum",
            "state_loss_sum",
            "relation_loss_sum",
            "gradient_norm_max",
            "elapsed_seconds",
        }
    )


def _payload(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    return {**payload, "payload_sha256": canonical_sha256(payload)}


def _validate_payload(value: Any, role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or value.get("artifact_role") != role:
        raise TrainingContractError(f"{role} artifact identity is invalid")
    payload = dict(value)
    digest = payload.pop("payload_sha256", None)
    if digest != canonical_sha256(payload):
        raise TrainingContractError(f"{role} artifact digest is invalid")
    return value


def _config() -> Mapping[str, Any]:
    value = load_json(EXPERIMENT_ROOT / "config.json")
    if not isinstance(value, Mapping):
        raise TrainingContractError("official config must be an object")
    return value


def official_training_settings() -> OfficialTrainingSettings:
    optimization = _config().get("optimization")
    if not isinstance(optimization, Mapping):
        raise TrainingContractError("official optimization config is absent")
    settings = OfficialTrainingSettings(
        maximum_epochs=int(optimization.get("maximum_epochs", 0)),
        early_stopping_patience=int(optimization.get("early_stopping_patience", 0)),
        effective_batch_size=int(optimization.get("effective_batch_size", 0)),
        checkpoint_interval_batches=int(optimization.get("checkpoint_interval_batches", 0)),
        checkpoint_matching_collar_ms=int(optimization.get("checkpoint_matching_collar_ms", 0)),
        warmup_fraction=float(optimization.get("warmup_fraction", -1.0)),
        gradient_clip_norm=float(optimization.get("gradient_clip_norm", 0.0)),
    )
    if (
        settings.maximum_epochs != MAXIMUM_EPOCHS
        or settings.early_stopping_patience != 4
        or settings.effective_batch_size != OFFICIAL_EFFECTIVE_BATCH_SIZE
        or WINDOWS_PER_EPOCH % settings.effective_batch_size
        or settings.checkpoint_interval_batches != 128
        or settings.checkpoint_matching_collar_ms != 250
        or settings.warmup_fraction != 0.05
        or settings.gradient_clip_norm != 5.0
    ):
        raise TrainingContractError("official optimization config differs from the frozen recipe")
    return settings


def _run_id(arm: str, seed: int) -> str:
    return f"{arm.lower()}-seed-{seed}"


def _runs_root(paths: PreflightPaths) -> Path:
    if paths.output_root is None:
        raise TrainingContractError("official output root is required")
    return paths.output_root.resolve() / RUNS_DIRECTORY


def _guard(paths: PreflightPaths) -> Mapping[str, Any]:
    if any(
        value is None
        for value in (paths.cache_root, paths.corpus_root, paths.reference_root, paths.output_root)
    ):
        raise TrainingContractError("all four official roots are required")
    receipt_path = paths.output_root.resolve() / "preflight" / "experiment_receipt.json"
    try:
        receipt = require_passing_preflight(receipt_path)
    except ExperimentPreflightError as error:
        raise TrainingContractError(str(error)) from error
    expected_paths = {
        "cache_root": str(paths.cache_root.resolve()),
        "corpus_root": str(paths.corpus_root.resolve()),
        "reference_root": str(paths.reference_root.resolve()),
        "output_root": str(paths.output_root.resolve()),
    }
    if receipt.get("paths") != expected_paths:
        raise TrainingContractError("passing preflight belongs to different official roots")
    return receipt


def _initial_plan(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return _payload(
        {
            "schema_version": 1,
            "artifact_role": "psem_official_run_plan",
            "experiment_id": "psem_training_strategy_gate_v1",
            "binding": receipt["binding"],
            "preflight_payload_sha256": receipt["payload_sha256"],
            "execution": _execution_identity(),
            "eval_status": "sealed",
            "created_at": _utc_now(),
            "runs": [
                {
                    "arm": arm,
                    "seed": seed,
                    "run_id": _run_id(arm, seed),
                    "status": "pending",
                    "completion_receipt": None,
                }
                for arm, seed in OFFICIAL_RUNS
            ],
        }
    )


def _validate_plan(plan: Any, receipt: Mapping[str, Any] | None = None) -> dict[str, Any]:
    value = dict(_validate_payload(plan, "psem_official_run_plan"))
    expected_keys = {
        "schema_version",
        "artifact_role",
        "experiment_id",
        "binding",
        "preflight_payload_sha256",
        "execution",
        "eval_status",
        "created_at",
        "runs",
        "payload_sha256",
    }
    if (
        set(value) != expected_keys
        or value.get("schema_version") != 1
        or value.get("experiment_id") != "psem_training_strategy_gate_v1"
        or value.get("eval_status") != "sealed"
        or not isinstance(value.get("created_at"), str)
        or not _execution_identity_valid(value.get("execution"))
    ):
        raise TrainingContractError("official run plan schema is invalid")
    if receipt is not None and (
        value.get("binding") != receipt.get("binding")
        or value.get("preflight_payload_sha256") != receipt.get("payload_sha256")
        or value.get("execution") != _execution_identity()
    ):
        raise TrainingContractError("official run plan belongs to a different preflight")
    rows = value.get("runs")
    if not isinstance(rows, list) or len(rows) != len(OFFICIAL_RUNS):
        raise TrainingContractError("official run plan inventory is invalid")
    for row, expected in zip(rows, OFFICIAL_RUNS, strict=True):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"arm", "seed", "run_id", "status", "completion_receipt"}
            or (row.get("arm"), row.get("seed")) != expected
            or row.get("run_id") != _run_id(*expected)
            or row.get("status") not in {"pending", "running", "completed"}
            or (row.get("status") == "completed")
            is not isinstance(row.get("completion_receipt"), Mapping)
        ):
            raise TrainingContractError("official run plan row is invalid")
    completed = [row["status"] == "completed" for row in rows]
    if any(completed[index] and not all(completed[:index]) for index in range(len(completed))):
        raise TrainingContractError("official run plan completion order is invalid")
    first_incomplete = sum(completed)
    running = [index for index, row in enumerate(rows) if row["status"] == "running"]
    if len(running) > 1 or (running and running[0] != first_incomplete):
        raise TrainingContractError("official run plan active row is invalid")
    if any(row["status"] != "pending" for row in rows[first_incomplete + len(running) :]):
        raise TrainingContractError("official run plan pending suffix is invalid")
    if any(
        row["status"] == "completed" and not _descriptor_valid(row["completion_receipt"])
        for row in rows
    ):
        raise TrainingContractError("official run plan completion artifact is invalid")
    return value


def _load_or_create_plan(paths: PreflightPaths, receipt: Mapping[str, Any]) -> dict[str, Any]:
    path = _runs_root(paths) / PLAN_FILENAME
    if path.is_file():
        return _validate_plan(load_json(path), receipt)
    plan = _initial_plan(receipt)
    _write_json(path, plan)
    return plan


def _write_plan(paths: PreflightPaths, plan: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(plan)
    payload.pop("payload_sha256", None)
    value = _payload(payload)
    _validate_plan(value)
    _write_json(_runs_root(paths) / PLAN_FILENAME, value)
    return value


def _next_run(plan: Mapping[str, Any]) -> Mapping[str, Any] | None:
    return next((row for row in plan["runs"] if row["status"] != "completed"), None)


def _execution_identity() -> dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "device": device,
        "device_name": (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if device == "cuda"
            else platform.processor() or platform.machine()
        ),
        "torch_version": torch.__version__,
        "torchaudio_version": torchaudio.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "float_dtype": "torch.float32",
    }


def _execution_identity_valid(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value)
        == {
            "device",
            "device_name",
            "torch_version",
            "torchaudio_version",
            "python_version",
            "platform",
            "hostname",
            "torch_num_threads",
            "torch_num_interop_threads",
            "float_dtype",
        }
        and value.get("device") in {"cpu", "cuda"}
        and all(
            isinstance(value.get(field), str) and bool(value[field])
            for field in {
                "device_name",
                "torch_version",
                "torchaudio_version",
                "python_version",
                "platform",
                "hostname",
            }
        )
        and _strict_int(value.get("torch_num_threads"), minimum=1)
        and _strict_int(value.get("torch_num_interop_threads"), minimum=1)
        and value.get("float_dtype") == "torch.float32"
    )


def _run_identity(
    paths: PreflightPaths,
    receipt: Mapping[str, Any],
    arm: str,
    seed: int,
    settings: OfficialTrainingSettings,
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    output_root = paths.output_root.resolve()
    repository_inputs = {
        "contract": _descriptor(CONTRACT_PATH),
        "config": _descriptor(CONFIG_PATH),
        "dataset_freeze": _descriptor(DATA_DIR / "dataset_freeze.json"),
        "source_manifest": _descriptor(SOURCE_MANIFEST_PATH),
        "split_manifest": _descriptor(SPLIT_MANIFEST_PATH),
        "source_registry": _descriptor(SOURCE_REGISTRY_PATH),
        "label_generator": _descriptor(LABEL_GENERATOR_PATH),
    }
    expected_binding = {
        "contract": "contract_sha256",
        "config": "config_sha256",
        "dataset_freeze": "dataset_freeze_sha256",
        "source_manifest": "source_manifest_sha256",
        "source_registry": "source_registry_sha256",
        "label_generator": "label_generator_sha256",
    }
    if any(
        repository_inputs[name]["sha256"] != receipt["binding"].get(binding_name)
        for name, binding_name in expected_binding.items()
    ):
        raise TrainingContractError("official repository inputs differ from passing preflight")
    material_roots, model_files, train_dev_sources = _materialize_inputs(
        paths,
        arm,
        output_root,
    )
    inputs = {
        "preflight_receipt": _descriptor(output_root / "preflight" / "experiment_receipt.json"),
        "repository": repository_inputs,
        "runtime": _runtime_input_descriptors(output_root),
        "model_files": model_files,
        "train_dev_sources": train_dev_sources,
    }
    identity = {
        "schema_version": 1,
        "artifact_role": "psem_official_run_contract",
        "experiment_id": "psem_training_strategy_gate_v1",
        "arm": arm,
        "seed": seed,
        "run_id": _run_id(arm, seed),
        "binding": receipt["binding"],
        "preflight_payload_sha256": receipt["payload_sha256"],
        "settings": asdict(settings),
        "execution": dict(execution),
        "material_roots": material_roots,
        "inputs": inputs,
        "file_snapshots": _file_snapshots(inputs),
        "data_roles": {"fit": TRAIN_ROLE, "checkpoint": DEV_ROLE, "eval": "sealed"},
    }
    _assert_identity_inputs(identity)
    return _payload(identity)


def _model_state_schema(model: torch.nn.Module) -> dict[str, Any]:
    rows = [
        {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        for name, value in model.state_dict().items()
    ]
    return {"rows": rows, "sha256": canonical_sha256(rows)}


def _model_state_schema_valid(value: Any) -> bool:
    if not isinstance(value, Mapping) or set(value) != {"rows", "sha256"}:
        return False
    rows = value.get("rows")
    return (
        isinstance(rows, list)
        and bool(rows)
        and value.get("sha256") == canonical_sha256(rows)
        and all(
            isinstance(row, Mapping)
            and set(row) == {"name", "shape", "dtype"}
            and isinstance(row.get("name"), str)
            and bool(row["name"])
            and isinstance(row.get("shape"), list)
            and all(_strict_int(size) for size in row["shape"])
            and isinstance(row.get("dtype"), str)
            and row["dtype"].startswith("torch.")
            for row in rows
        )
        and len({row["name"] for row in rows}) == len(rows)
    )


def _model_state_valid(value: Any, schema: Mapping[str, Any]) -> bool:
    if not isinstance(value, Mapping) or not _model_state_schema_valid(schema):
        return False
    expected = {row["name"]: row for row in schema["rows"]}
    if set(value) != set(expected):
        return False
    for name, tensor in value.items():
        row = expected[name]
        if (
            not isinstance(tensor, torch.Tensor)
            or list(tensor.shape) != row["shape"]
            or str(tensor.dtype) != row["dtype"]
            or (tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()))
        ):
            return False
    return True


def _dev_source_inventory(sessions: Mapping[str, RuntimeSession]) -> list[dict[str, Any]]:
    rows = []
    for source_id, session in sorted(sessions.items()):
        if session.role != DEV_ROLE:
            continue
        start = session.labels.intervals[0].start_sample
        end = session.labels.intervals[-1].end_sample
        centers = tuple(valid_center_samples(start, end))
        scored_mask = [
            build_window_targets(source_id, session.labels, center).handoff_mask
            for center in centers
        ]
        rows.append(
            {
                "source_id": source_id,
                "scored_start_sample": start,
                "scored_end_sample": end,
                "prediction_count": len(centers),
                "scored_prediction_count": sum(scored_mask),
                "excluded_prediction_count": len(scored_mask) - sum(scored_mask),
                "scored_mask_sha256": canonical_sha256(scored_mask),
            }
        )
    if not rows:
        raise TrainingContractError("official DEV source inventory is empty")
    return rows


def _material_identity(
    identity: Mapping[str, Any],
    model: PSEMModel,
    sessions: Mapping[str, RuntimeSession],
) -> dict[str, Any]:
    base = dict(identity)
    base_digest = base.pop("payload_sha256", None)
    inventory = parameter_inventory(model)
    return _payload(
        {
            **base,
            "base_identity_sha256": base_digest,
            "model_state_schema": _model_state_schema(model),
            "parameter_summary": {
                "total_parameters": inventory["total_parameters"],
                "trainable_parameters": inventory["trainable_parameters"],
                "trainable_wavlm_parameters": inventory["trainable_wavlm_parameters"],
            },
            "dev_sources": _dev_source_inventory(sessions),
        }
    )


def _material_contract_matches_base(
    contract: Mapping[str, Any], identity: Mapping[str, Any]
) -> bool:
    if (
        not _model_state_schema_valid(contract.get("model_state_schema"))
        or contract.get("base_identity_sha256") != identity.get("payload_sha256")
        or not isinstance(contract.get("parameter_summary"), Mapping)
        or not isinstance(contract.get("dev_sources"), list)
        or not contract["dev_sources"]
    ):
        return False
    base = dict(contract)
    for key in (
        "payload_sha256",
        "base_identity_sha256",
        "model_state_schema",
        "parameter_summary",
        "dev_sources",
    ):
        base.pop(key, None)
    expected = dict(identity)
    expected.pop("payload_sha256", None)
    return base == expected


def _prepare_run_contract(run_root: Path, identity: Mapping[str, Any]) -> None:
    path = run_root / "run_contract.json"
    if path.is_file():
        current = _validate_payload(load_json(path), "psem_official_run_contract")
        if current != identity:
            raise TrainingContractError("official run contract changed during resume")
        return
    _write_json(path, identity)


def _assert_identity_inputs(identity: Mapping[str, Any]) -> None:
    material_roots = identity.get("material_roots")
    inputs = identity.get("inputs")
    repository = inputs.get("repository") if isinstance(inputs, Mapping) else None
    runtime = inputs.get("runtime") if isinstance(inputs, Mapping) else None
    model_files = inputs.get("model_files") if isinstance(inputs, Mapping) else None
    sources = inputs.get("train_dev_sources") if isinstance(inputs, Mapping) else None
    if (
        not isinstance(inputs, Mapping)
        or not isinstance(material_roots, Mapping)
        or set(material_roots) != {"cache_root", "corpus_root"}
        or any(
            not isinstance(value, str) or str(Path(value).resolve()) != value
            for value in material_roots.values()
        )
        or set(inputs)
        != {"preflight_receipt", "repository", "runtime", "model_files", "train_dev_sources"}
        or not _descriptor_valid(inputs.get("preflight_receipt"))
        or not isinstance(repository, Mapping)
        or set(repository)
        != {
            "contract",
            "config",
            "dataset_freeze",
            "source_manifest",
            "split_manifest",
            "source_registry",
            "label_generator",
        }
        or any(not _descriptor_valid(value) for value in repository.values())
        or not isinstance(runtime, Mapping)
        or set(runtime)
        != {
            "sampling_manifest",
            "sampling_summary",
            "augmentation_manifest",
            "model_graphs",
            "parameter_inventory",
            "gradient_canary",
            "weight_update_canary",
            "arm_comparability",
            "metric_contract",
        }
        or any(not _descriptor_valid(value) for value in runtime.values())
        or not isinstance(model_files, list)
        or len(model_files) != (0 if identity.get("arm") == "SCRATCH-PSEM" else 3)
        or any(not _descriptor_valid(value) for value in model_files)
        or not isinstance(sources, list)
        or not sources
    ):
        raise TrainingContractError("official run inputs changed after the material guard")
    seen_sources = set()
    for row in sources:
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {"source_id", "role", "annotation_sha256", "reference_sha256", "waveform"}
            or not isinstance(row.get("source_id"), str)
            or row["source_id"] in seen_sources
            or row.get("role") not in {TRAIN_ROLE, DEV_ROLE}
            or not isinstance(row.get("annotation_sha256"), str)
            or len(row["annotation_sha256"]) != 64
            or not isinstance(row.get("reference_sha256"), str)
            or len(row["reference_sha256"]) != 64
            or not _descriptor_valid(row.get("waveform"))
        ):
            raise TrainingContractError("official TRAIN/DEV source input binding is invalid")
        seen_sources.add(row["source_id"])
    if seen_sources != {
        source_id for source_id, role in _role_by_source().items() if role in {TRAIN_ROLE, DEV_ROLE}
    }:
        raise TrainingContractError("official TRAIN/DEV source input inventory is incomplete")
    cache_root = Path(material_roots["cache_root"])
    corpus_root = Path(material_roots["corpus_root"])
    if any(
        not Path(str(row["path"])).resolve().is_relative_to(cache_root)
        for row in model_files
    ) or any(
        not Path(str(row["waveform"]["path"])).resolve().is_relative_to(corpus_root)
        for row in sources
    ):
        raise TrainingContractError("official material input escapes its immutable root")
    _assert_file_snapshots(identity)


def _loss_weights(output_root: Path) -> LossWeights:
    summary = load_json(output_root / "manifests" / "sampling_summary.json")
    values = summary.get("loss_weights") if isinstance(summary, Mapping) else None
    if not isinstance(values, Mapping):
        raise TrainingContractError("sampling loss weights are absent")
    weights = LossWeights(
        handoff_positive=float(values["handoff_positive"]),
        state_classes=tuple(float(value) for value in values["state_classes"]),
        relation_classes=tuple(float(value) for value in values["relation_classes"]),
    )
    if len(weights.state_classes) != 3 or len(weights.relation_classes) != 2:
        raise TrainingContractError("sampling loss weights differ from the objective")
    return weights


def _rows_by_epoch(output_root: Path) -> dict[int, list[Mapping[str, Any]]]:
    rows = load_sampling_rows(output_root / "manifests" / "sampling_manifest.jsonl")
    result = {epoch: [] for epoch in range(1, MAXIMUM_EPOCHS + 1)}
    for row in rows:
        epoch = row.get("epoch")
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch not in result:
            raise TrainingContractError("sampling row epoch is invalid")
        result[epoch].append(row)
    for epoch, values in result.items():
        if (
            len(values) != WINDOWS_PER_EPOCH
            or [row.get("epoch_index") for row in values] != list(range(WINDOWS_PER_EPOCH))
            or [row.get("row_id") for row in values]
            != [f"epoch-{epoch:02d}-window-{index:04d}" for index in range(WINDOWS_PER_EPOCH)]
        ):
            raise TrainingContractError("sampling epoch order differs from the frozen manifest")
    return result


def _batch(
    rows: Sequence[Mapping[str, Any]],
    sessions: Mapping[str, RuntimeSession],
    corpus_root: Path,
    identity: Mapping[str, Any],
    *,
    augment: bool,
) -> tuple[torch.Tensor, list[WindowTargets]]:
    waveforms = []
    targets = []
    for row in rows:
        source_id = str(row.get("source_id"))
        session = sessions.get(source_id)
        if session is None:
            raise TrainingContractError("sampling row source is absent from the selected role")
        target = target_for_row(row, session)
        _assert_file_snapshots(identity, ((corpus_root.resolve() / session.audio_ref).resolve(),))
        waveform = load_waveform_window(row, session, corpus_root)
        _assert_file_snapshots(identity, ((corpus_root.resolve() / session.audio_ref).resolve(),))
        if augment:
            waveform = apply_augmentation(waveform, row["augmentation"])
        waveforms.append(waveform)
        targets.append(target)
    return torch.stack(waveforms), targets


def _dev_waveform(
    session: RuntimeSession,
    boundary_sample: int,
    corpus_root: Path,
    identity: Mapping[str, Any],
) -> tuple[torch.Tensor, WindowTargets]:
    target = build_window_targets(session.source_id, session.labels, boundary_sample)
    path = (corpus_root.resolve() / session.audio_ref).resolve()
    if not path.is_relative_to(corpus_root.resolve()):
        raise TrainingContractError("DEV waveform path escapes the bound corpus root")
    _assert_file_snapshots(identity, (path,))
    waveform, sample_rate = torchaudio.load(
        path,
        frame_offset=target.window_start_sample,
        num_frames=WINDOW_SAMPLES,
    )
    _assert_file_snapshots(identity, (path,))
    if sample_rate != 16000 or waveform.shape != (1, WINDOW_SAMPLES):
        raise TrainingContractError("DEV waveform differs from the raw-audio contract")
    return waveform[0], target


def _dev_references(sessions: Mapping[str, RuntimeSession]) -> list[ReferenceEvent]:
    rows = []
    identities: set[tuple[str, int]] = set()
    for source_id, session in sorted(sessions.items()):
        if session.role != DEV_ROLE:
            continue
        for transition in session.labels.transitions:
            sample = transition.get("handoff_source_sample")
            if (
                transition.get("mask_state") != "valid"
                or transition.get("handoff_confirmed") != 1
                or not isinstance(sample, int)
                or isinstance(sample, bool)
            ):
                continue
            identity = (source_id, sample)
            if identity in identities:
                raise TrainingContractError("DEV reference event identity is duplicated")
            identities.add(identity)
            rows.append(
                ReferenceEvent(
                    source_id,
                    sample,
                    str(transition.get("primary_topology") or "handoff_confirmed"),
                )
            )
    if not rows:
        raise TrainingContractError("DEV reference event inventory is empty")
    return rows


def _dev_scored_samples(sessions: Mapping[str, RuntimeSession]) -> int:
    observed = sum(
        session.labels.intervals[-1].end_sample - session.labels.intervals[0].start_sample
        for session in sessions.values()
        if session.role == DEV_ROLE
    )
    split = load_json(SPLIT_MANIFEST_PATH)
    expected = split["role_summaries"][DEV_ROLE]["scored_samples"]
    if observed != expected:
        raise TrainingContractError("DEV runtime exposure differs from the frozen split")
    return observed


def _schedule_factor(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < 0 or total_steps <= 0 or not 0 < warmup_steps < total_steps:
        raise TrainingContractError("cosine schedule geometry is invalid")
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = min((step - warmup_steps) / (total_steps - warmup_steps), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    settings: OfficialTrainingSettings,
) -> torch.optim.lr_scheduler.LambdaLR:
    batches_per_epoch = WINDOWS_PER_EPOCH // settings.effective_batch_size
    total_steps = settings.maximum_epochs * batches_per_epoch
    warmup_steps = math.ceil(total_steps * settings.warmup_fraction)
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _schedule_factor(step, total_steps, warmup_steps),
    )


def _optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _checkpoint_payload(
    *,
    run_identity_sha256: str,
    sequence: int,
    phase: str,
    epoch: int,
    next_batch_index: int,
    model: PSEMModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    accumulator: TrainingAccumulator,
    history: list[dict[str, Any]],
    best: dict[str, Any] | None,
    stale_epochs: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_role": "psem_official_training_checkpoint",
        "run_identity_sha256": run_identity_sha256,
        "sequence": sequence,
        "phase": phase,
        "epoch": epoch,
        "next_batch_index": next_batch_index,
        "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "accumulator": asdict(accumulator),
        "history": history,
        "best": best,
        "stale_epochs": stale_epochs,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python_random_state": random.getstate(),
    }


def _write_checkpoint(run_root: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    slot = "a" if int(payload["sequence"]) % 2 == 0 else "b"
    path = run_root / "checkpoints" / f"latest-{slot}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)
    return _descriptor(path)


def _history_replay(
    history: Any,
) -> tuple[Mapping[str, Any] | None, int] | None:
    if (
        not isinstance(history, list)
        or len(history) > MAXIMUM_EPOCHS
        or [row.get("epoch") for row in history if isinstance(row, Mapping)]
        != list(range(1, len(history) + 1))
    ):
        return None
    best = None
    stale_epochs = 0
    for index, row in enumerate(history):
        if not _history_row_valid(row):
            return None
        improved = _improves(
            float(row["dev_event_average_precision"]),
            float(row["dev_total_loss"]),
            best,
        )
        if row["improved"] is not improved:
            return None
        if improved:
            best = {
                "epoch": row["epoch"],
                "event_average_precision": row["dev_event_average_precision"],
                "total_loss": row["dev_total_loss"],
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= 4 and index != len(history) - 1:
            return None
    return best, stale_epochs


def _rng_state_valid(value: Mapping[str, Any]) -> bool:
    try:
        generator = torch.Generator()
        generator.set_state(value["torch_rng_state"])
        checker = random.Random()
        checker.setstate(value["python_random_state"])
        if torch.cuda.is_available():
            if len(value["cuda_rng_states"]) != torch.cuda.device_count():
                return False
            for index, state in enumerate(value["cuda_rng_states"]):
                torch.Generator(device=f"cuda:{index}").set_state(state)
        elif value["cuda_rng_states"]:
            return False
    except (IndexError, KeyError, RuntimeError, TypeError, ValueError):
        return False
    return True


def _checkpoint_valid(
    value: Any,
    run_identity_sha256: str,
    model_state_schema: Mapping[str, Any] | None = None,
    *,
    slot: str | None = None,
) -> bool:
    expected_keys = {
        "schema_version",
        "artifact_role",
        "run_identity_sha256",
        "sequence",
        "phase",
        "epoch",
        "next_batch_index",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "accumulator",
        "history",
        "best",
        "stale_epochs",
        "torch_rng_state",
        "cuda_rng_states",
        "python_random_state",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != expected_keys
        or value.get("schema_version") != 1
        or value.get("artifact_role") != "psem_official_training_checkpoint"
        or value.get("run_identity_sha256") != run_identity_sha256
        or not isinstance(value.get("sequence"), int)
        or isinstance(value.get("sequence"), bool)
        or value["sequence"] < 0
        or value.get("phase") not in {"train", "dev", "complete"}
        or not isinstance(value.get("epoch"), int)
        or isinstance(value.get("epoch"), bool)
        or not 1 <= value["epoch"] <= MAXIMUM_EPOCHS
        or not isinstance(value.get("next_batch_index"), int)
        or isinstance(value.get("next_batch_index"), bool)
        or not 0 <= value["next_batch_index"] <= WINDOWS_PER_EPOCH
        or value["next_batch_index"] % OFFICIAL_EFFECTIVE_BATCH_SIZE
        or not _model_state_valid(value.get("model_state"), model_state_schema or {})
        or not isinstance(value.get("optimizer_state"), Mapping)
        or set(value["optimizer_state"]) != {"state", "param_groups"}
        or not isinstance(value["optimizer_state"].get("state"), Mapping)
        or not isinstance(value["optimizer_state"].get("param_groups"), list)
        or not value["optimizer_state"]["param_groups"]
        or not isinstance(value.get("scheduler_state"), Mapping)
        or not value["scheduler_state"]
        or not _accumulator_valid(value.get("accumulator"))
        or not isinstance(value.get("history"), list)
        or any(not _history_row_valid(row) for row in value["history"])
        or (value.get("best") is not None and not _best_valid(value.get("best")))
        or not isinstance(value.get("stale_epochs"), int)
        or isinstance(value.get("stale_epochs"), bool)
        or value["stale_epochs"] < 0
        or not isinstance(value.get("torch_rng_state"), torch.Tensor)
        or value["torch_rng_state"].dtype != torch.uint8
        or not isinstance(value.get("cuda_rng_states"), list)
        or any(
            not isinstance(state, torch.Tensor) or state.dtype != torch.uint8
            for state in value["cuda_rng_states"]
        )
        or not isinstance(value.get("python_random_state"), tuple)
        or (slot is not None and slot != ("a" if value["sequence"] % 2 == 0 else "b"))
    ):
        return False
    replay = _history_replay(value["history"])
    if replay is None or not _rng_state_valid(value):
        return False
    expected_best, expected_stale = replay
    if value["stale_epochs"] != expected_stale:
        return False
    if expected_best is None:
        if value["best"] is not None:
            return False
    elif (
        value["best"] is None
        or value["best"].get("epoch") != expected_best["epoch"]
        or value["best"].get("event_average_precision") != expected_best["event_average_precision"]
        or value["best"].get("total_loss") != expected_best["total_loss"]
        or value["best"].get("collar_ms") != 250
    ):
        return False
    if value["phase"] == "train" and value["next_batch_index"] >= WINDOWS_PER_EPOCH:
        return False
    if value["phase"] in {"dev", "complete"} and value["next_batch_index"] != WINDOWS_PER_EPOCH:
        return False
    if value["phase"] == "train":
        if len(value["history"]) != value["epoch"] - 1:
            return False
        if value["accumulator"]["window_count"] != value["next_batch_index"]:
            return False
        if value["stale_epochs"] >= 4:
            return False
    if value["phase"] == "dev":
        if len(value["history"]) != value["epoch"] - 1:
            return False
        if value["accumulator"]["window_count"] != WINDOWS_PER_EPOCH:
            return False
        if value["stale_epochs"] >= 4:
            return False
    if value["phase"] == "complete":
        if len(value["history"]) != value["epoch"] or value["best"] is None:
            return False
        if value["accumulator"] != asdict(TrainingAccumulator()):
            return False
        if value["stale_epochs"] < 4 and value["epoch"] != MAXIMUM_EPOCHS:
            return False
    if value["epoch"] > 1 and value["best"] is None:
        return False
    return True


def _progress_dev_metric_valid(
    run_root: Path,
    row: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> bool:
    try:
        epoch = row["epoch"]
        metrics_path = (run_root / "dev" / f"epoch-{epoch:02d}-metrics.json").resolve()
        if row.get("dev_metrics") != _descriptor(metrics_path):
            return False
        metrics = _validate_payload(load_json(metrics_path), "psem_dev_checkpoint_metrics")
        raw_path = (run_root / "dev" / f"epoch-{epoch:02d}-predictions.jsonl").resolve()
        checkpoint_metric = metrics.get("checkpoint_metric")
        losses = metrics.get("losses")
        expected_prediction_count = sum(
            value["prediction_count"] for value in contract["dev_sources"]
        )
        expected_scored_count = sum(
            value["scored_prediction_count"] for value in contract["dev_sources"]
        )
        expected_excluded_count = sum(
            value["excluded_prediction_count"] for value in contract["dev_sources"]
        )
        return (
            set(metrics)
            == {
                "schema_version",
                "artifact_role",
                "run_identity_sha256",
                "epoch",
                "data_role",
                "eval_opened",
                "source_ids",
                "scored_source_samples",
                "prediction_count",
                "scored_prediction_count",
                "excluded_prediction_count",
                "candidate_count",
                "reference_count",
                "checkpoint_metric",
                "losses",
                "timing",
                "peak_rss_bytes",
                "raw_predictions",
                "generated_at",
                "payload_sha256",
            }
            and metrics.get("run_identity_sha256") == contract["payload_sha256"]
            and metrics.get("epoch") == epoch
            and metrics.get("data_role") == DEV_ROLE
            and metrics.get("eval_opened") is False
            and metrics.get("source_ids")
            == [value["source_id"] for value in contract["dev_sources"]]
            and metrics.get("scored_source_samples")
            == sum(
                value["scored_end_sample"] - value["scored_start_sample"]
                for value in contract["dev_sources"]
            )
            and metrics.get("prediction_count") == expected_prediction_count
            and metrics.get("scored_prediction_count") == expected_scored_count
            and metrics.get("excluded_prediction_count") == expected_excluded_count
            and isinstance(checkpoint_metric, Mapping)
            and checkpoint_metric.get("event_average_precision")
            == row["dev_event_average_precision"]
            and checkpoint_metric.get("collar_ms") == 250
            and isinstance(losses, Mapping)
            and losses.get("total") == row["dev_total_loss"]
            and _timing_valid(metrics.get("timing"), expected_prediction_count)
            and _strict_int(metrics.get("peak_rss_bytes"), minimum=1)
            and _timestamp_valid(metrics.get("generated_at"))
            and metrics.get("raw_predictions") == _descriptor(raw_path)
        )
    except (KeyError, OSError, TrainingContractError, TypeError, ValueError):
        return False


def _progress_best_valid(
    run_root: Path,
    best: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> bool:
    try:
        descriptor = best["checkpoint"]
        path = Path(str(descriptor["path"])).resolve()
        if (
            path.parent != (run_root / "checkpoints").resolve()
            or path.name not in {"best-a.pt", "best-b.pt"}
            or descriptor != _descriptor(path)
        ):
            return False
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        metrics = checkpoint.get("metrics") if isinstance(checkpoint, Mapping) else None
        return (
            isinstance(checkpoint, Mapping)
            and set(checkpoint)
            == {
                "schema_version",
                "artifact_role",
                "run_identity_sha256",
                "arm",
                "seed",
                "epoch",
                "metrics",
                "model_state",
            }
            and checkpoint.get("schema_version") == 1
            and checkpoint.get("artifact_role") == "psem_official_best_checkpoint"
            and checkpoint.get("run_identity_sha256") == contract["payload_sha256"]
            and checkpoint.get("arm") == contract["arm"]
            and checkpoint.get("seed") == contract["seed"]
            and checkpoint.get("epoch") == best["epoch"]
            and metrics
            == {
                "event_average_precision": best["event_average_precision"],
                "total_loss": best["total_loss"],
                "collar_ms": best["collar_ms"],
            }
            and _model_state_valid(checkpoint.get("model_state"), contract["model_state_schema"])
        )
    except (
        EOFError,
        KeyError,
        OSError,
        pickle.UnpicklingError,
        RuntimeError,
        TrainingContractError,
        TypeError,
        ValueError,
    ):
        return False


def _progress_artifacts_valid(
    run_root: Path,
    checkpoint: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> bool:
    if checkpoint.get("run_identity_sha256") != contract.get("payload_sha256"):
        return False
    history = checkpoint.get("history")
    best = checkpoint.get("best")
    return (
        isinstance(history, list)
        and all(
            isinstance(row, Mapping) and _progress_dev_metric_valid(run_root, row, contract)
            for row in history
        )
        and (
            best is None
            or (
                isinstance(best, Mapping)
                and _progress_best_valid(run_root, best, contract)
            )
        )
    )


def _load_checkpoint(
    run_root: Path,
    run_identity_sha256: str,
    model_state_schema: Mapping[str, Any],
    *,
    model: PSEMModel | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
    device: torch.device | None = None,
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    candidates = []
    saw_file = False
    for slot in ("a", "b"):
        path = run_root / "checkpoints" / f"latest-{slot}.pt"
        if not path.is_file():
            continue
        saw_file = True
        try:
            value = torch.load(path, map_location="cpu", weights_only=False)
        except (EOFError, OSError, pickle.UnpicklingError, RuntimeError, TypeError, ValueError):
            continue
        if _checkpoint_valid(
            value, run_identity_sha256, model_state_schema, slot=slot
        ) and (contract is None or _progress_artifacts_valid(run_root, value, contract)):
            candidates.append(value)
    if not saw_file:
        if (run_root / "run_state.json").exists():
            _load_run_state(run_root, run_identity_sha256)
            raise TrainingContractError("official checkpoints are absent for recorded progress")
        return None
    if saw_file and not candidates:
        _load_run_state(run_root, run_identity_sha256)
        raise TrainingContractError("no valid resumable official checkpoint remains")
    candidates.sort(key=lambda value: value["sequence"], reverse=True)
    state = _load_run_state(run_root, run_identity_sha256)

    def validated(candidate: dict[str, Any]) -> dict[str, Any]:
        slot = "a" if candidate["sequence"] % 2 == 0 else "b"
        descriptor = _descriptor(run_root / "checkpoints" / f"latest-{slot}.pt")
        if state["checkpoint_sequence"] == candidate[
            "sequence"
        ] and not _run_state_matches_checkpoint(state, candidate, descriptor):
            raise TrainingContractError("official run state differs from its checkpoint")
        return candidate

    if model is None:
        return validated(candidates[0])
    if optimizer is None or scheduler is None or device is None:
        raise TrainingContractError("deep checkpoint restore dependencies are incomplete")
    initial_optimizer = deepcopy(optimizer.state_dict())
    initial_scheduler = deepcopy(scheduler.state_dict())
    for candidate in candidates:
        try:
            _restore_checkpoint(candidate, model, optimizer, scheduler, device)
            if not _restored_checkpoint_valid(candidate, model, optimizer, scheduler):
                raise TrainingContractError("restored checkpoint state is invalid")
        except (KeyError, RuntimeError, TrainingContractError, TypeError, ValueError):
            optimizer.load_state_dict(initial_optimizer)
            scheduler.load_state_dict(initial_scheduler)
            continue
        return validated(candidate)
    raise TrainingContractError("no deeply restorable official checkpoint remains")


def _restore_checkpoint(
    checkpoint: Mapping[str, Any],
    model: PSEMModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
) -> None:
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    _optimizer_to(optimizer, device)
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
    random.setstate(checkpoint["python_random_state"])


def _restored_checkpoint_valid(
    checkpoint: Mapping[str, Any],
    model: PSEMModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
) -> bool:
    expected_groups = optimizer_groups(model)
    saved_groups = checkpoint["optimizer_state"].get("param_groups")
    if (
        len(optimizer.param_groups) != len(expected_groups)
        or not isinstance(saved_groups, list)
        or len(saved_groups) != len(expected_groups)
    ):
        return False
    expected_by_name = dict(model.named_parameters())
    next_parameter_id = 0
    saved_parameter_ids = []
    for actual, saved, expected in zip(
        optimizer.param_groups, saved_groups, expected_groups, strict=True
    ):
        expected_parameters = [expected_by_name[name] for name in expected.parameter_names]
        expected_ids = list(
            range(next_parameter_id, next_parameter_id + len(expected.parameter_names))
        )
        next_parameter_id += len(expected.parameter_names)
        if (
            not isinstance(saved, Mapping)
            or saved.get("params") != expected_ids
            or any(
                actual_parameter is not expected_parameter
                for actual_parameter, expected_parameter in zip(
                    actual.get("params", []), expected_parameters, strict=False
                )
            )
            or set(actual)
            != {
                "params",
                "lr",
                "group_name",
                "betas",
                "eps",
                "weight_decay",
                "amsgrad",
                "maximize",
                "foreach",
                "capturable",
                "differentiable",
                "fused",
                "decoupled_weight_decay",
                "initial_lr",
            }
            or actual.get("foreach") is not None
            or actual.get("capturable") is not False
            or actual.get("differentiable") is not False
            or actual.get("fused") is not None
            or actual.get("decoupled_weight_decay") is not True
            or actual.get("group_name") != expected.name
            or actual.get("initial_lr") != expected.learning_rate
            or actual.get("betas") != (0.9, 0.999)
            or actual.get("eps") != 1e-8
            or actual.get("weight_decay") != 1e-4
            or actual.get("amsgrad") is not False
            or actual.get("maximize") is not False
            or len(actual.get("params", [])) != len(expected.parameter_names)
        ):
            return False
        saved_parameter_ids.extend(expected_ids)
    saved_state = checkpoint["optimizer_state"].get("state")
    if not isinstance(saved_state, Mapping) or frozenset(saved_state) not in {
        frozenset(),
        frozenset(saved_parameter_ids),
    }:
        return False
    for parameter, state in optimizer.state.items():
        if not isinstance(parameter, torch.nn.Parameter) or not isinstance(state, Mapping):
            return False
        if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            return False
        step = state["step"]
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        if (
            not isinstance(step, torch.Tensor)
            or step.numel() != 1
            or not bool(torch.isfinite(step).all())
            or not isinstance(exp_avg, torch.Tensor)
            or not isinstance(exp_avg_sq, torch.Tensor)
            or exp_avg.shape != parameter.shape
            or exp_avg_sq.shape != parameter.shape
            or exp_avg.dtype != parameter.dtype
            or exp_avg_sq.dtype != parameter.dtype
            or not bool(torch.isfinite(exp_avg).all())
            or not bool(torch.isfinite(exp_avg_sq).all())
        ):
            return False
    batches_per_epoch = WINDOWS_PER_EPOCH // OFFICIAL_EFFECTIVE_BATCH_SIZE
    if checkpoint["phase"] == "train":
        expected_steps = (checkpoint["epoch"] - 1) * batches_per_epoch + (
            checkpoint["next_batch_index"] // OFFICIAL_EFFECTIVE_BATCH_SIZE
        )
    else:
        expected_steps = checkpoint["epoch"] * batches_per_epoch
    expected_parameter_count = sum(len(group.parameter_names) for group in expected_groups)
    if len(optimizer.state) != (0 if expected_steps == 0 else expected_parameter_count):
        return False
    for state in optimizer.state.values():
        if float(state["step"]) != expected_steps:
            return False
    total_steps = MAXIMUM_EPOCHS * batches_per_epoch
    warmup_steps = math.ceil(total_steps * 0.05)
    factor = _schedule_factor(expected_steps, total_steps, warmup_steps)
    expected_lrs = [group.learning_rate * factor for group in expected_groups]
    if [group.get("lr") for group in optimizer.param_groups] != expected_lrs:
        return False
    state = scheduler.state_dict()
    scheduler_keys = {
        "base_lrs",
        "last_epoch",
        "_step_count",
        "_get_lr_called_within_step",
        "_last_lr",
        "lr_lambdas",
    }
    return (
        frozenset(state)
        in {frozenset(scheduler_keys), frozenset((*scheduler_keys, "_is_initial"))}
        and state.get("last_epoch") == expected_steps
        and state.get("_step_count") == expected_steps + 1
        and ("_is_initial" not in state or state["_is_initial"] is False)
        and state.get("_get_lr_called_within_step") is False
        and state.get("base_lrs") == [group.learning_rate for group in expected_groups]
        and state.get("_last_lr") == expected_lrs
        and state.get("lr_lambdas") == [None] * len(expected_groups)
    )


def _run_state(
    identity: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    latest: Mapping[str, Any],
    *,
    status: str,
) -> dict[str, Any]:
    return _payload(
        {
            "schema_version": 1,
            "artifact_role": "psem_official_run_state",
            "run_id": identity["run_id"],
            "run_identity_sha256": identity["payload_sha256"],
            "status": status,
            "phase": checkpoint["phase"],
            "epoch": checkpoint["epoch"],
            "next_batch_index": checkpoint["next_batch_index"],
            "checkpoint_sequence": checkpoint["sequence"],
            "latest_checkpoint": latest,
            "best": checkpoint["best"],
            "history": checkpoint["history"],
            "eval_opened": False,
            "updated_at": _utc_now(),
        }
    )


def _load_run_state(run_root: Path, run_identity_sha256: str) -> dict[str, Any]:
    path = run_root / "run_state.json"
    if not path.is_file():
        raise TrainingContractError("official run state is absent")
    value = dict(_validate_payload(load_json(path), "psem_official_run_state"))
    latest = value.get("latest_checkpoint")
    if (
        set(value)
        != {
            "schema_version",
            "artifact_role",
            "run_id",
            "run_identity_sha256",
            "status",
            "phase",
            "epoch",
            "next_batch_index",
            "checkpoint_sequence",
            "latest_checkpoint",
            "best",
            "history",
            "eval_opened",
            "updated_at",
            "payload_sha256",
        }
        or value.get("schema_version") != 1
        or value.get("run_id") != run_root.name
        or value.get("run_identity_sha256") != run_identity_sha256
        or value.get("status") not in {"running", "completed"}
        or value.get("phase") not in {"train", "dev", "complete"}
        or (value["status"] == "completed") is not (value["phase"] == "complete")
        or not _strict_int(value.get("epoch"), minimum=1)
        or value["epoch"] > MAXIMUM_EPOCHS
        or not _strict_int(value.get("next_batch_index"))
        or value["next_batch_index"] > WINDOWS_PER_EPOCH
        or value["next_batch_index"] % OFFICIAL_EFFECTIVE_BATCH_SIZE
        or not _strict_int(value.get("checkpoint_sequence"))
        or not isinstance(latest, Mapping)
        or set(latest) != {"path", "size_bytes", "sha256"}
        or not isinstance(latest.get("path"), str)
        or Path(latest["path"]).resolve().parent != (run_root / "checkpoints").resolve()
        or Path(latest["path"]).name
        != f"latest-{'a' if value['checkpoint_sequence'] % 2 == 0 else 'b'}.pt"
        or not _strict_int(latest.get("size_bytes"), minimum=1)
        or not isinstance(latest.get("sha256"), str)
        or len(latest["sha256"]) != 64
        or (value.get("best") is not None and not _best_valid(value["best"]))
        or _history_replay(value.get("history")) is None
        or value.get("eval_opened") is not False
        or not isinstance(value.get("updated_at"), str)
        or not value["updated_at"]
    ):
        raise TrainingContractError("official run state is invalid")
    return value


def _run_state_matches_checkpoint(
    state: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    descriptor: Mapping[str, Any],
) -> bool:
    return all(
        (
            state["phase"] == checkpoint["phase"],
            state["epoch"] == checkpoint["epoch"],
            state["next_batch_index"] == checkpoint["next_batch_index"],
            state["checkpoint_sequence"] == checkpoint["sequence"],
            state["latest_checkpoint"] == descriptor,
            state["best"] == checkpoint["best"],
            state["history"] == checkpoint["history"],
            state["status"] == ("completed" if checkpoint["phase"] == "complete" else "running"),
        )
    )


def _assert_pristine_run_root(run_root: Path) -> None:
    allowed = {"run_contract.json"}
    material = [
        path
        for path in run_root.rglob("*")
        if path.is_file() and path.relative_to(run_root).as_posix() not in allowed
    ]
    if material:
        raise TrainingContractError("official run progress exists without a resumable checkpoint")


def _save_progress(
    run_root: Path,
    identity: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    *,
    status: str = "running",
) -> None:
    latest = _write_checkpoint(run_root, checkpoint)
    _write_json(
        run_root / "run_state.json", _run_state(identity, checkpoint, latest, status=status)
    )


def _best_checkpoint(
    run_root: Path,
    identity: Mapping[str, Any],
    model: PSEMModel,
    epoch: int,
    metrics: Mapping[str, Any],
    current_best: Mapping[str, Any] | None,
) -> dict[str, Any]:
    current_name = (
        Path(current_best["checkpoint"]["path"]).name if current_best is not None else None
    )
    slot = "b" if current_name == "best-a.pt" else "a"
    path = run_root / "checkpoints" / f"best-{slot}.pt"
    payload = {
        "schema_version": 1,
        "artifact_role": "psem_official_best_checkpoint",
        "run_identity_sha256": identity["payload_sha256"],
        "arm": identity["arm"],
        "seed": identity["seed"],
        "epoch": epoch,
        "metrics": dict(metrics),
        "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return _descriptor(path)


def _improves(
    event_average_precision_value: float,
    total_loss: float,
    best: Mapping[str, Any] | None,
) -> bool:
    if not math.isfinite(event_average_precision_value) or not math.isfinite(total_loss):
        raise TrainingContractError("DEV checkpoint metrics must be finite")
    if best is None:
        return True
    return event_average_precision_value > best["event_average_precision"] or (
        event_average_precision_value == best["event_average_precision"]
        and total_loss < best["total_loss"]
    )


def _evaluate_dev(
    *,
    model: PSEMModel,
    sessions: Mapping[str, RuntimeSession],
    corpus_root: Path,
    weights: LossWeights,
    settings: OfficialTrainingSettings,
    device: torch.device,
    run_root: Path,
    identity: Mapping[str, Any],
    epoch: int,
) -> dict[str, Any]:
    dev_sessions = {
        source_id: session for source_id, session in sessions.items() if session.role == DEV_ROLE
    }
    if not dev_sessions or any(session.role == EVAL_ROLE for session in dev_sessions.values()):
        raise TrainingContractError("DEV evaluation role selection is invalid")
    predictions: list[PredictionScore] = []
    accumulator = LossAccumulator()
    per_window_seconds: list[float] = []
    process = psutil.Process()
    peak_rss_bytes = process.memory_info().rss
    prediction_path = run_root / "dev" / f"epoch-{epoch:02d}-predictions.jsonl"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = prediction_path.with_name(f".{prediction_path.name}.{os.getpid()}.tmp")
    model.eval()
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        with torch.inference_mode():
            for source_id, session in sorted(dev_sessions.items()):
                centers = list(
                    valid_center_samples(
                        session.labels.intervals[0].start_sample,
                        session.labels.intervals[-1].end_sample,
                    )
                )
                for start in range(0, len(centers), settings.effective_batch_size):
                    selected = centers[start : start + settings.effective_batch_size]
                    waveforms = []
                    targets = []
                    for boundary_sample in selected:
                        waveform, target = _dev_waveform(
                            session,
                            boundary_sample,
                            corpus_root,
                            identity,
                        )
                        waveforms.append(waveform)
                        targets.append(target)
                    values = torch.stack(waveforms).to(device)
                    target_batch = collate_targets(targets).to(device)
                    started = time.perf_counter()
                    outputs = model(values)
                    elapsed = time.perf_counter() - started
                    per_window_seconds.extend([elapsed / len(selected)] * len(selected))
                    accumulator.update(loss_statistics(model, outputs, target_batch, weights))
                    scores = torch.sigmoid(outputs["handoff_logits"]).detach().cpu().tolist()
                    for boundary_sample, target, score in zip(
                        selected, targets, scores, strict=True
                    ):
                        row = PredictionScore(
                            source_id,
                            boundary_sample,
                            boundary_sample + FUTURE_SAMPLES,
                            float(score),
                            target.handoff_mask,
                        )
                        predictions.append(row)
                        handle.write(
                            json.dumps(
                                {
                                    "schema_version": 1,
                                    "artifact_role": "psem_dev_prediction_score",
                                    "source_id": row.source_id,
                                    "boundary_sample": row.boundary_sample,
                                    "observed_frontier_sample": row.observed_frontier_sample,
                                    "score": row.score,
                                    "scored": row.scored,
                                },
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                    peak_rss_bytes = max(peak_rss_bytes, process.memory_info().rss)
    temporary.replace(prediction_path)
    candidates: tuple[CandidateEvent, ...] = eventize(predictions)
    references = _dev_references(dev_sessions)
    checkpoint_metric = event_average_precision(
        candidates,
        references,
        collar_ms=settings.checkpoint_matching_collar_ms,
    )
    losses = accumulator.result()
    ordered_times = sorted(per_window_seconds)
    if not ordered_times:
        raise TrainingContractError("DEV prediction timing inventory is empty")

    def percentile(fraction: float) -> float:
        return ordered_times[math.ceil(fraction * len(ordered_times)) - 1]

    payload = _payload(
        {
            "schema_version": 1,
            "artifact_role": "psem_dev_checkpoint_metrics",
            "run_identity_sha256": identity["payload_sha256"],
            "epoch": epoch,
            "data_role": DEV_ROLE,
            "eval_opened": False,
            "source_ids": sorted(dev_sessions),
            "scored_source_samples": _dev_scored_samples(dev_sessions),
            "prediction_count": len(predictions),
            "scored_prediction_count": sum(row.scored for row in predictions),
            "excluded_prediction_count": sum(not row.scored for row in predictions),
            "candidate_count": len(candidates),
            "reference_count": len(references),
            "checkpoint_metric": checkpoint_metric,
            "losses": losses,
            "timing": {
                "window_count": len(ordered_times),
                "per_window_seconds_p50": percentile(0.5),
                "per_window_seconds_p95": percentile(0.95),
            },
            "peak_rss_bytes": peak_rss_bytes,
            "raw_predictions": _descriptor(prediction_path),
            "generated_at": _utc_now(),
        }
    )
    metrics_path = run_root / "dev" / f"epoch-{epoch:02d}-metrics.json"
    _write_json(metrics_path, payload)
    return {**payload, "artifact": _descriptor(metrics_path)}


def _completion_receipt(
    run_root: Path,
    identity: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    best: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> dict[str, Any]:
    value = _payload(
        {
            "schema_version": 1,
            "artifact_role": "psem_official_training_completion",
            "run_id": identity["run_id"],
            "run_identity_sha256": identity["payload_sha256"],
            "arm": identity["arm"],
            "seed": identity["seed"],
            "status": "completed",
            "eval_opened": False,
            "best": dict(best),
            "history": list(history),
            "parameter_summary": {
                "total_parameters": inventory["total_parameters"],
                "trainable_parameters": inventory["trainable_parameters"],
                "trainable_wavlm_parameters": inventory["trainable_wavlm_parameters"],
            },
            "completed_at": _utc_now(),
        }
    )
    path = run_root / "completion_receipt.json"
    _write_json(path, value)
    return value


def _validated_material_contract(run_root: Path, run_identity_sha256: str) -> dict[str, Any]:
    path = run_root / "run_contract.json"
    if not path.is_file():
        raise TrainingContractError("official run contract is absent")
    value = dict(_validate_payload(load_json(path), "psem_official_run_contract"))
    parameter_summary = value.get("parameter_summary")
    dev_sources = value.get("dev_sources")
    if (
        value.get("payload_sha256") != run_identity_sha256
        or value.get("run_id") != run_root.name
        or (value.get("arm"), value.get("seed")) not in OFFICIAL_RUNS
        or not _model_state_schema_valid(value.get("model_state_schema"))
        or not isinstance(parameter_summary, Mapping)
        or set(parameter_summary)
        != {"total_parameters", "trainable_parameters", "trainable_wavlm_parameters"}
        or not all(_strict_int(item) for item in parameter_summary.values())
        or parameter_summary["total_parameters"] <= 0
        or parameter_summary["trainable_parameters"] <= 0
        or not isinstance(dev_sources, list)
        or not dev_sources
    ):
        raise TrainingContractError("official material run contract is invalid")
    expected_dev_ids = {
        source_id for source_id, role in _role_by_source().items() if role == DEV_ROLE
    }
    seen = set()
    for row in dev_sources:
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {
                "source_id",
                "scored_start_sample",
                "scored_end_sample",
                "prediction_count",
                "scored_prediction_count",
                "excluded_prediction_count",
                "scored_mask_sha256",
            }
            or not isinstance(row.get("source_id"), str)
            or row["source_id"] in seen
            or not _strict_int(row.get("scored_start_sample"))
            or not _strict_int(row.get("scored_end_sample"), minimum=1)
            or row["scored_end_sample"] <= row["scored_start_sample"]
            or row.get("prediction_count")
            != len(valid_center_samples(row["scored_start_sample"], row["scored_end_sample"]))
            or not _strict_int(row.get("scored_prediction_count"))
            or not _strict_int(row.get("excluded_prediction_count"))
            or row["scored_prediction_count"] + row["excluded_prediction_count"]
            != row["prediction_count"]
            or not isinstance(row.get("scored_mask_sha256"), str)
            or len(row["scored_mask_sha256"]) != 64
        ):
            raise TrainingContractError("official DEV source contract is invalid")
        seen.add(row["source_id"])
    if seen != expected_dev_ids:
        raise TrainingContractError("official DEV source contract is incomplete")
    _assert_identity_inputs(value)
    receipt_path = Path(value["inputs"]["preflight_receipt"]["path"])
    try:
        receipt = _validate_payload(load_json(receipt_path), "psem_experiment_preflight")
    except (OSError, TypeError, ValueError) as error:
        raise TrainingContractError("official preflight receipt cannot bind DEV inputs") from error
    roots = receipt.get("paths")
    if (
        not isinstance(roots, Mapping)
        or set(roots) != {"cache_root", "corpus_root", "reference_root", "output_root"}
        or any(
            not isinstance(item, str) or str(Path(item).resolve()) != item
            for item in roots.values()
        )
    ):
        raise TrainingContractError("official preflight roots cannot bind DEV inputs")
    try:
        sessions = load_runtime_sessions(
            Path(roots["corpus_root"]),
            Path(roots["reference_root"]),
            roles=(TRAIN_ROLE, DEV_ROLE),
        )
    except (OSError, SamplingContractError, TypeError, ValueError) as error:
        raise TrainingContractError("official TRAIN/DEV sources cannot be reconstructed") from error
    _assert_file_snapshots(value)
    if value["dev_sources"] != _dev_source_inventory(sessions):
        raise TrainingContractError("official DEV source contract differs from frozen inputs")
    return value


def _verification_runtime(
    contract: Mapping[str, Any],
) -> tuple[
    PSEMModel,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LambdaLR,
    torch.device,
]:
    try:
        settings = OfficialTrainingSettings(**contract["settings"])
    except (KeyError, TypeError, ValueError) as error:
        raise TrainingContractError("official run settings are invalid") from error
    if settings != official_training_settings():
        raise TrainingContractError("official run settings differ from the frozen recipe")
    arm = str(contract.get("arm"))
    if arm == "SCRATCH-PSEM":
        cache_root = Path.cwd().resolve()
    else:
        model_files = contract["inputs"]["model_files"]
        directories = {Path(row["path"]).resolve().parent for row in model_files}
        if len(directories) != 1:
            raise TrainingContractError("official model files do not share the pinned root")
        model_directory = directories.pop()
        try:
            cache_root = model_directory.parents[2]
        except IndexError as error:
            raise TrainingContractError("official model cache root is invalid") from error
        if model_directory != model_root(cache_root):
            raise TrainingContractError("official model files escape the pinned model root")
    try:
        _assert_file_snapshots(contract)
        model = build_model(arm, cache_root=cache_root, seed=int(contract["seed"]))
    except (KeyError, ModelContractError, TypeError, ValueError) as error:
        raise TrainingContractError("official model cannot be reconstructed") from error
    _assert_file_snapshots(contract)
    device = torch.device("cpu")
    model.to(device)
    _validate_material_model(contract, model)
    optimizer = build_optimizer(model)
    scheduler = _build_scheduler(optimizer, settings)
    return model, optimizer, scheduler, device


def _validate_material_model(contract: Mapping[str, Any], model: PSEMModel) -> None:
    inventory = parameter_inventory(model)
    actual_summary = {
        "total_parameters": inventory["total_parameters"],
        "trainable_parameters": inventory["trainable_parameters"],
        "trainable_wavlm_parameters": inventory["trainable_wavlm_parameters"],
    }
    if _model_state_schema(model) != contract.get(
        "model_state_schema"
    ) or actual_summary != contract.get("parameter_summary"):
        raise TrainingContractError("official material model differs from its run contract")


def _validate_dev_predictions(
    run_root: Path,
    epoch: int,
    metrics: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> None:
    path = (run_root / "dev" / f"epoch-{epoch:02d}-predictions.jsonl").resolve()
    expected_descriptor = _descriptor(path)
    if metrics.get("raw_predictions") != expected_descriptor:
        raise TrainingContractError("official DEV predictions escape their run binding")
    expected = (
        (row["source_id"], boundary_sample)
        for row in contract["dev_sources"]
        for boundary_sample in valid_center_samples(
            row["scored_start_sample"], row["scored_end_sample"]
        )
    )
    scored_masks = {row["source_id"]: [] for row in contract["dev_sources"]}
    count = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                target = next(expected)
                value = json.loads(line)
                if (
                    not isinstance(value, Mapping)
                    or set(value)
                    != {
                        "schema_version",
                        "artifact_role",
                        "source_id",
                        "boundary_sample",
                        "observed_frontier_sample",
                        "score",
                        "scored",
                    }
                    or value.get("schema_version") != 1
                    or value.get("artifact_role") != "psem_dev_prediction_score"
                    or (value.get("source_id"), value.get("boundary_sample")) != target
                    or value.get("observed_frontier_sample") != target[1] + FUTURE_SAMPLES
                    or not _finite_number(value.get("score"), minimum=0.0, maximum=1.0)
                    or type(value.get("scored")) is not bool
                ):
                    raise TrainingContractError("official DEV prediction row is invalid")
                scored_masks[target[0]].append(value["scored"])
                count += 1
    except (OSError, StopIteration, TypeError, ValueError) as error:
        raise TrainingContractError("official DEV prediction inventory is invalid") from error
    try:
        next(expected)
    except StopIteration:
        pass
    else:
        raise TrainingContractError("official DEV prediction inventory is incomplete")
    if (
        count != metrics.get("prediction_count")
        or metrics.get("scored_prediction_count")
        != sum(row["scored_prediction_count"] for row in contract["dev_sources"])
        or metrics.get("excluded_prediction_count")
        != sum(row["excluded_prediction_count"] for row in contract["dev_sources"])
        or any(
        len(scored_masks[row["source_id"]]) != row["prediction_count"]
        or sum(scored_masks[row["source_id"]]) != row["scored_prediction_count"]
        or len(scored_masks[row["source_id"]]) - sum(scored_masks[row["source_id"]])
        != row["excluded_prediction_count"]
        or canonical_sha256(scored_masks[row["source_id"]]) != row["scored_mask_sha256"]
        for row in contract["dev_sources"]
        )
    ):
        raise TrainingContractError("official DEV prediction count is invalid")


def verify_completed_run(
    run_root: Path,
    run_identity_sha256: str,
    *,
    model: PSEMModel | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    contract = _validated_material_contract(run_root, run_identity_sha256)
    if model is None:
        if optimizer is not None or scheduler is not None or device is not None:
            raise TrainingContractError("completion verification runtime is incomplete")
        model, optimizer, scheduler, device = _verification_runtime(contract)
    elif optimizer is None or scheduler is None or device is None:
        raise TrainingContractError("completion verification runtime is incomplete")
    else:
        _validate_material_model(contract, model)
    path = run_root / "completion_receipt.json"
    if not path.is_file():
        raise TrainingContractError("official training completion receipt is absent")
    value = dict(_validate_payload(load_json(path), "psem_official_training_completion"))
    best = value.get("best")
    parameter_summary = value.get("parameter_summary")
    arm = value.get("arm")
    seed = value.get("seed")
    expected_keys = {
        "schema_version",
        "artifact_role",
        "run_id",
        "run_identity_sha256",
        "arm",
        "seed",
        "status",
        "eval_opened",
        "best",
        "history",
        "parameter_summary",
        "completed_at",
        "payload_sha256",
    }
    if (
        set(value) != expected_keys
        or value.get("schema_version") != 1
        or value.get("status") != "completed"
        or value.get("eval_opened") is not False
        or value.get("run_identity_sha256") != run_identity_sha256
        or (arm, seed) not in OFFICIAL_RUNS
        or value.get("run_id") != _run_id(arm, seed)
        or run_root.name != value.get("run_id")
        or not _best_valid(best)
        or not isinstance(value.get("history"), list)
        or not value["history"]
        or len(value["history"]) > MAXIMUM_EPOCHS
        or parameter_summary != contract["parameter_summary"]
        or not isinstance(parameter_summary, Mapping)
        or set(parameter_summary)
        != {
            "total_parameters",
            "trainable_parameters",
            "trainable_wavlm_parameters",
        }
        or not all(_strict_int(item, minimum=0) for item in parameter_summary.values())
        or parameter_summary["total_parameters"] <= 0
        or parameter_summary["trainable_parameters"] <= 0
        or parameter_summary["trainable_parameters"] > parameter_summary["total_parameters"]
        or (
            arm in {"FROZEN-WAVLM", "SCRATCH-PSEM"}
            and parameter_summary["trainable_wavlm_parameters"] != 0
        )
        or (arm == "FINETUNE-WAVLM" and parameter_summary["trainable_wavlm_parameters"] <= 0)
        or (
            arm == "SCRATCH-PSEM"
            and not 5_000_000 <= parameter_summary["total_parameters"] <= 10_000_000
        )
    ):
        raise TrainingContractError("official training completion receipt is invalid")
    latest = _load_checkpoint(
        run_root,
        run_identity_sha256,
        contract["model_state_schema"],
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        contract=contract,
    )
    if (
        latest is None
        or latest.get("phase") != "complete"
        or latest.get("history") != value["history"]
        or latest.get("best") != best
    ):
        raise TrainingContractError("official completion state differs from its checkpoint")
    if [row.get("epoch") for row in value["history"] if isinstance(row, Mapping)] != list(
        range(1, len(value["history"]) + 1)
    ):
        raise TrainingContractError("official training epoch history is not contiguous")
    expected_best: Mapping[str, Any] | None = None
    stale_epochs = 0
    for index, row in enumerate(value["history"]):
        if not _history_row_valid(row):
            raise TrainingContractError("official DEV history artifact is invalid")
        improved = _improves(
            float(row["dev_event_average_precision"]),
            float(row["dev_total_loss"]),
            expected_best,
        )
        if row["improved"] is not improved:
            raise TrainingContractError("official DEV checkpoint selection history is invalid")
        if improved:
            expected_best = {
                "epoch": row["epoch"],
                "event_average_precision": row["dev_event_average_precision"],
                "total_loss": row["dev_total_loss"],
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= 4 and index != len(value["history"]) - 1:
            raise TrainingContractError("official training continued after early stopping")
        metrics_path = (run_root / "dev" / f"epoch-{row['epoch']:02d}-metrics.json").resolve()
        if row["dev_metrics"] != _descriptor(metrics_path):
            raise TrainingContractError("official DEV metrics escape their run binding")
        metrics = _validate_payload(load_json(metrics_path), "psem_dev_checkpoint_metrics")
        checkpoint_metric = metrics.get("checkpoint_metric")
        losses = metrics.get("losses")
        if (
            set(metrics)
            != {
                "schema_version",
                "artifact_role",
                "run_identity_sha256",
                "epoch",
                "data_role",
                "eval_opened",
                "source_ids",
                "scored_source_samples",
                "prediction_count",
                "scored_prediction_count",
                "excluded_prediction_count",
                "candidate_count",
                "reference_count",
                "checkpoint_metric",
                "losses",
                "timing",
                "peak_rss_bytes",
                "raw_predictions",
                "generated_at",
                "payload_sha256",
            }
            or metrics.get("run_identity_sha256") != run_identity_sha256
            or metrics.get("epoch") != row["epoch"]
            or metrics.get("data_role") != DEV_ROLE
            or metrics.get("eval_opened") is not False
            or metrics.get("source_ids") != [row["source_id"] for row in contract["dev_sources"]]
            or metrics.get("scored_source_samples")
            != sum(
                row["scored_end_sample"] - row["scored_start_sample"]
                for row in contract["dev_sources"]
            )
            or metrics.get("prediction_count")
            != sum(row["prediction_count"] for row in contract["dev_sources"])
            or metrics.get("scored_prediction_count")
            != sum(row["scored_prediction_count"] for row in contract["dev_sources"])
            or metrics.get("excluded_prediction_count")
            != sum(row["excluded_prediction_count"] for row in contract["dev_sources"])
            or metrics.get("prediction_count")
            != metrics.get("scored_prediction_count") + metrics.get("excluded_prediction_count")
            or not isinstance(checkpoint_metric, Mapping)
            or set(checkpoint_metric)
            != {
                "event_average_precision",
                "maximum_f1",
                "candidate_count",
                "reference_count",
                "threshold_count",
                "collar_ms",
            }
            or checkpoint_metric.get("collar_ms") != 250
            or checkpoint_metric.get("event_average_precision")
            != row["dev_event_average_precision"]
            or not _finite_number(checkpoint_metric.get("maximum_f1"), minimum=0.0, maximum=1.0)
            or not all(
                _strict_int(checkpoint_metric.get(field), minimum=1)
                for field in ("candidate_count", "reference_count", "threshold_count")
            )
            or metrics.get("candidate_count") != checkpoint_metric.get("candidate_count")
            or metrics.get("reference_count") != checkpoint_metric.get("reference_count")
            or not isinstance(losses, Mapping)
            or set(losses) != {"total", "handoff", "state", "relation"}
            or any(not _finite_number(item, minimum=0.0) for item in losses.values())
            or losses.get("total") != row["dev_total_loss"]
            or not _timing_valid(metrics.get("timing"), metrics["prediction_count"])
            or not _strict_int(metrics.get("peak_rss_bytes"), minimum=1)
            or not _timestamp_valid(metrics.get("generated_at"))
            or not _descriptor_valid(metrics.get("raw_predictions"))
        ):
            raise TrainingContractError("official DEV metric binding is invalid")
        _validate_dev_predictions(run_root, row["epoch"], metrics, contract)
    if (
        expected_best is None
        or best.get("epoch") != expected_best["epoch"]
        or best.get("event_average_precision") != expected_best["event_average_precision"]
        or best.get("total_loss") != expected_best["total_loss"]
        or best.get("collar_ms") != 250
        or (len(value["history"]) < MAXIMUM_EPOCHS and stale_epochs < 4)
    ):
        raise TrainingContractError("official best checkpoint and stop boundary are invalid")
    try:
        checkpoint = torch.load(
            Path(best["checkpoint"]["path"]),
            map_location="cpu",
            weights_only=False,
        )
    except (
        EOFError,
        OSError,
        pickle.UnpicklingError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        raise TrainingContractError("official best checkpoint cannot be loaded") from error
    checkpoint_metrics = checkpoint.get("metrics") if isinstance(checkpoint, Mapping) else None
    if (
        not isinstance(checkpoint, Mapping)
        or checkpoint.get("schema_version") != 1
        or checkpoint.get("artifact_role") != "psem_official_best_checkpoint"
        or checkpoint.get("run_identity_sha256") != run_identity_sha256
        or checkpoint.get("arm") != value.get("arm")
        or checkpoint.get("seed") != value.get("seed")
        or checkpoint.get("epoch") != best.get("epoch")
        or not isinstance(checkpoint_metrics, Mapping)
        or set(checkpoint_metrics) != {"event_average_precision", "total_loss", "collar_ms"}
        or checkpoint_metrics.get("event_average_precision") != best.get("event_average_precision")
        or checkpoint_metrics.get("total_loss") != best.get("total_loss")
        or checkpoint_metrics.get("collar_ms") != best.get("collar_ms")
        or not _model_state_valid(checkpoint.get("model_state"), contract["model_state_schema"])
        or Path(best["checkpoint"]["path"]).resolve().parent != (run_root / "checkpoints").resolve()
        or Path(best["checkpoint"]["path"]).name not in {"best-a.pt", "best-b.pt"}
    ):
        raise TrainingContractError("official best checkpoint binding is invalid")
    return value


def _verify_completed_prefix(
    paths: PreflightPaths,
    plan: Mapping[str, Any],
    receipt: Mapping[str, Any],
    settings: OfficialTrainingSettings,
) -> None:
    runs_root = _runs_root(paths)
    for row in plan["runs"]:
        if row["status"] != "completed":
            break
        run_root = runs_root / row["run_id"]
        contract_path = run_root / "run_contract.json"
        if not contract_path.is_file():
            raise TrainingContractError("completed official run contract is absent")
        contract = dict(_validate_payload(load_json(contract_path), "psem_official_run_contract"))
        expected = _run_identity(
            paths,
            receipt,
            row["arm"],
            row["seed"],
            settings,
            plan["execution"],
        )
        if not _material_contract_matches_base(contract, expected):
            raise TrainingContractError("completed official run contract differs from the matrix")
        verify_completed_run(run_root, contract["payload_sha256"])
        if row["completion_receipt"] != _descriptor(run_root / "completion_receipt.json"):
            raise TrainingContractError("official run plan completion binding is stale")


def _execute_run(
    paths: PreflightPaths,
    identity: Mapping[str, Any],
    settings: OfficialTrainingSettings,
) -> dict[str, Any]:
    _assert_identity_inputs(identity)
    output_root = paths.output_root.resolve()
    material_cache_root = Path(identity["material_roots"]["cache_root"])
    material_corpus_root = Path(identity["material_roots"]["corpus_root"])
    rows_by_epoch = _rows_by_epoch(output_root)
    sessions = load_runtime_sessions(
        material_corpus_root,
        paths.reference_root.resolve(),
        roles=(TRAIN_ROLE, DEV_ROLE),
    )
    if {session.role for session in sessions.values()} != {TRAIN_ROLE, DEV_ROLE}:
        raise TrainingContractError("official training roles are incomplete")
    weights = _loss_weights(output_root)
    device = torch.device(identity["execution"]["device"])
    random.seed(identity["seed"])
    torch.manual_seed(identity["seed"])
    torch.use_deterministic_algorithms(True)
    _assert_file_snapshots(identity)
    model = build_model(
        identity["arm"], cache_root=material_cache_root, seed=identity["seed"]
    )
    _assert_file_snapshots(identity)
    model.to(device)
    identity = _material_identity(identity, model, sessions)
    run_root = _runs_root(paths) / identity["run_id"]
    _prepare_run_contract(run_root, identity)
    optimizer = build_optimizer(model)
    scheduler = _build_scheduler(optimizer, settings)
    completion_path = run_root / "completion_receipt.json"
    if completion_path.is_file():
        return verify_completed_run(
            run_root,
            identity["payload_sha256"],
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
    checkpoint = _load_checkpoint(
        run_root,
        identity["payload_sha256"],
        identity["model_state_schema"],
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        contract=identity,
    )
    if checkpoint is None:
        _assert_pristine_run_root(run_root)
        checkpoint = _checkpoint_payload(
            run_identity_sha256=identity["payload_sha256"],
            sequence=0,
            phase="train",
            epoch=1,
            next_batch_index=0,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            accumulator=TrainingAccumulator(),
            history=[],
            best=None,
            stale_epochs=0,
        )
        _save_progress(run_root, identity, checkpoint)
    else:
        _save_progress(
            run_root,
            identity,
            checkpoint,
            status="completed" if checkpoint["phase"] == "complete" else "running",
        )
    while checkpoint["phase"] != "complete":
        _assert_file_snapshots(identity)
        epoch = int(checkpoint["epoch"])
        if epoch < 1 or epoch > settings.maximum_epochs:
            raise TrainingContractError("official checkpoint epoch is invalid")
        accumulator = TrainingAccumulator(**checkpoint["accumulator"])
        history = list(checkpoint["history"])
        best = dict(checkpoint["best"]) if checkpoint["best"] is not None else None
        stale_epochs = int(checkpoint["stale_epochs"])
        sequence = int(checkpoint["sequence"])
        if checkpoint["phase"] == "train":
            model.train()
            epoch_rows = rows_by_epoch[epoch]
            start_index = int(checkpoint["next_batch_index"])
            if start_index % settings.effective_batch_size:
                raise TrainingContractError("official checkpoint batch index is invalid")
            for batch_index in range(
                start_index,
                WINDOWS_PER_EPOCH,
                settings.effective_batch_size,
            ):
                rows = epoch_rows[batch_index : batch_index + settings.effective_batch_size]
                waveforms, targets = _batch(
                    rows,
                    sessions,
                    material_corpus_root,
                    identity,
                    augment=True,
                )
                waveforms = waveforms.to(device)
                target_batch = collate_targets(targets).to(device)
                started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                outputs = model(waveforms)
                losses = compute_losses(model, outputs, target_batch, weights)
                losses["total"].backward()
                gradient_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        [parameter for parameter in model.parameters() if parameter.requires_grad],
                        settings.gradient_clip_norm,
                        error_if_nonfinite=True,
                    )
                )
                optimizer.step()
                scheduler.step()
                accumulator.update(
                    losses,
                    windows=len(rows),
                    gradient_norm=gradient_norm,
                    elapsed_seconds=time.perf_counter() - started,
                    rss_bytes=psutil.Process().memory_info().rss,
                )
                next_index = batch_index + len(rows)
                batches_completed = next_index // settings.effective_batch_size
                if (
                    batches_completed % settings.checkpoint_interval_batches == 0
                    and next_index < WINDOWS_PER_EPOCH
                ):
                    sequence += 1
                    checkpoint = _checkpoint_payload(
                        run_identity_sha256=identity["payload_sha256"],
                        sequence=sequence,
                        phase="train",
                        epoch=epoch,
                        next_batch_index=next_index,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        accumulator=accumulator,
                        history=history,
                        best=best,
                        stale_epochs=stale_epochs,
                    )
                    _save_progress(run_root, identity, checkpoint)
                    print(
                        json.dumps(
                            {
                                "run_id": identity["run_id"],
                                "epoch": epoch,
                                "completed_windows": next_index,
                                "total_windows": WINDOWS_PER_EPOCH,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            sequence += 1
            checkpoint = _checkpoint_payload(
                run_identity_sha256=identity["payload_sha256"],
                sequence=sequence,
                phase="dev",
                epoch=epoch,
                next_batch_index=WINDOWS_PER_EPOCH,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                accumulator=accumulator,
                history=history,
                best=best,
                stale_epochs=stale_epochs,
            )
            _save_progress(run_root, identity, checkpoint)
        if checkpoint["phase"] == "dev":
            dev = _evaluate_dev(
                model=model,
                sessions=sessions,
                corpus_root=material_corpus_root,
                weights=weights,
                settings=settings,
                device=device,
                run_root=run_root,
                identity=identity,
                epoch=epoch,
            )
            event_ap = float(dev["checkpoint_metric"]["event_average_precision"])
            total_loss = float(dev["losses"]["total"])
            improved = _improves(event_ap, total_loss, best)
            if improved:
                best_checkpoint = _best_checkpoint(
                    run_root,
                    identity,
                    model,
                    epoch,
                    {
                        "event_average_precision": event_ap,
                        "total_loss": total_loss,
                        "collar_ms": settings.checkpoint_matching_collar_ms,
                    },
                    best,
                )
                best = {
                    "epoch": epoch,
                    "event_average_precision": event_ap,
                    "total_loss": total_loss,
                    "collar_ms": settings.checkpoint_matching_collar_ms,
                    "checkpoint": best_checkpoint,
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
            history.append(
                {
                    "epoch": epoch,
                    "train": accumulator.summary(),
                    "dev_event_average_precision": event_ap,
                    "dev_total_loss": total_loss,
                    "checkpoint_matching_collar_ms": settings.checkpoint_matching_collar_ms,
                    "improved": improved,
                    "dev_metrics": dev["artifact"],
                }
            )
            stop = (
                stale_epochs >= settings.early_stopping_patience or epoch == settings.maximum_epochs
            )
            sequence = int(checkpoint["sequence"]) + 1
            checkpoint = _checkpoint_payload(
                run_identity_sha256=identity["payload_sha256"],
                sequence=sequence,
                phase="complete" if stop else "train",
                epoch=epoch if stop else epoch + 1,
                next_batch_index=WINDOWS_PER_EPOCH if stop else 0,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                accumulator=TrainingAccumulator(),
                history=history,
                best=best,
                stale_epochs=stale_epochs,
            )
            _save_progress(
                run_root,
                identity,
                checkpoint,
                status="completed" if stop else "running",
            )
            print(
                json.dumps(
                    {
                        "run_id": identity["run_id"],
                        "epoch": epoch,
                        "dev_event_average_precision": event_ap,
                        "dev_total_loss": total_loss,
                        "improved": improved,
                        "early_stop": stop,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if checkpoint["best"] is None:
        raise TrainingContractError("official training completed without a best checkpoint")
    confirmed_receipt = _guard(paths)
    if confirmed_receipt.get("payload_sha256") != identity["preflight_payload_sha256"]:
        raise TrainingContractError("passing preflight changed during official training")
    _assert_identity_inputs(identity)
    _completion_receipt(
        run_root,
        identity,
        checkpoint["history"],
        checkpoint["best"],
        parameter_inventory(model),
    )
    return verify_completed_run(
        run_root,
        identity["payload_sha256"],
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )


def train_official_run(paths: PreflightPaths, arm: str, seed: int) -> dict[str, Any]:
    if (arm, seed) not in OFFICIAL_RUNS:
        raise TrainingContractError("requested run is absent from the frozen official matrix")
    receipt = _guard(paths)
    settings = official_training_settings()
    runs_root = _runs_root(paths)
    with _ExclusiveRunLock(runs_root / LOCK_FILENAME):
        confirmed_receipt = _guard(paths)
        if confirmed_receipt != receipt:
            raise TrainingContractError("passing preflight changed while acquiring the run lock")
        plan = _load_or_create_plan(paths, confirmed_receipt)
        _verify_completed_prefix(paths, plan, confirmed_receipt, settings)
        pending = _next_run(plan)
        if pending is None:
            selected = next(row for row in plan["runs"] if (row["arm"], row["seed"]) == (arm, seed))
            if selected["status"] != "completed":
                raise TrainingContractError("official run plan has no executable row")
            return verify_completed_run(
                runs_root / selected["run_id"],
                load_json(runs_root / selected["run_id"] / "run_contract.json")["payload_sha256"],
            )
        if (pending["arm"], pending["seed"]) != (arm, seed):
            raise TrainingContractError(
                f"official run order requires {pending['arm']} seed {pending['seed']} next"
            )
        identity = _run_identity(
            paths,
            confirmed_receipt,
            arm,
            seed,
            settings,
            plan["execution"],
        )
        pending["status"] = "running"
        pending["completion_receipt"] = None
        plan = _write_plan(paths, plan)
        completion = _execute_run(paths, identity, settings)
        plan = _validate_plan(load_json(runs_root / PLAN_FILENAME), confirmed_receipt)
        pending = _next_run(plan)
        if pending is None or (pending["arm"], pending["seed"]) != (arm, seed):
            raise TrainingContractError("official run plan changed during execution")
        pending["status"] = "completed"
        pending["completion_receipt"] = _descriptor(
            runs_root / pending["run_id"] / "completion_receipt.json"
        )
        _write_plan(paths, plan)
        return completion


def train_all_official_runs(paths: PreflightPaths) -> list[dict[str, Any]]:
    results = []
    for arm, seed in OFFICIAL_RUNS:
        status = training_status(paths)
        row = next(
            (
                value
                for value in status.get("runs", [])
                if (value["arm"], value["seed"]) == (arm, seed)
            ),
            None,
        )
        if row is not None and row["status"] == "completed":
            continue
        results.append(train_official_run(paths, arm, seed))
    train_official_run(paths, *OFFICIAL_RUNS[-1])
    return results


def training_status(paths: PreflightPaths) -> dict[str, Any]:
    if paths.output_root is None:
        raise TrainingContractError("official output root is required")
    path = paths.output_root.resolve() / RUNS_DIRECTORY / PLAN_FILENAME
    if not path.is_file():
        return {
            "artifact_role": "psem_official_run_status",
            "eval_status": "sealed",
            "runs": [],
        }
    receipt = _guard(paths)
    plan = _validate_plan(load_json(path), receipt)
    settings = official_training_settings()
    _verify_completed_prefix(paths, plan, receipt, settings)
    running = next((row for row in plan["runs"] if row["status"] == "running"), None)
    if running is not None:
        run_root = path.parent / running["run_id"]
        contract_path = run_root / "run_contract.json"
        if contract_path.is_file():
            base = _run_identity(
                paths,
                receipt,
                running["arm"],
                running["seed"],
                settings,
                plan["execution"],
            )
            contract = dict(
                _validate_payload(load_json(contract_path), "psem_official_run_contract")
            )
            if not _material_contract_matches_base(contract, base):
                raise TrainingContractError("running official run contract differs from the matrix")
            _assert_identity_inputs(contract)
            model, optimizer, scheduler, device = _verification_runtime(contract)
            checkpoint = _load_checkpoint(
                run_root,
                contract["payload_sha256"],
                contract["model_state_schema"],
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                contract=contract,
            )
            if checkpoint is None:
                _assert_pristine_run_root(run_root)
        elif run_root.exists() and any(run_root.iterdir()):
            raise TrainingContractError("running official run contract is absent")
    return {
        "artifact_role": "psem_official_run_status",
        "eval_status": plan["eval_status"],
        "binding": plan["binding"],
        "runs": plan["runs"],
    }
