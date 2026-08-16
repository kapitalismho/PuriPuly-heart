from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from bisect import bisect_left, bisect_right
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from experiments.online_speaker_memory_handoff.protocol import (
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from experiments.speaker_representation_scd.extraction.ssl import SSLExtractor
from experiments.speaker_representation_scd.r7b_local_segmentation import (
    HOP_SAMPLES,
    SAMPLE_RATE,
    STATE_OVERLAP,
    STATE_SILENCE,
    STATE_SINGLE,
    SessionBundle,
    _load_bundles,
    _metrics,
    _peaks,
    _role_paths,
    _training_centers,
    _valid_centers,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = EXPERIMENT_ROOT / "config.json"
CODE_PATH = Path(__file__).resolve()
SOURCE_REGISTRY_PATH = (
    REPOSITORY_ROOT / "experiments" / "speaker_representation_scd" / "models" / "source_registry.json"
)
R7B_OUTPUT_RELATIVE = "results/r7b/fixed_lag_local_segmentation_v1"
ARMS = (
    "A-FROZEN-DIRECT",
    "B-TRAINABLE-DIRECT",
    "C-FROZEN-STATE",
    "D-TRAINABLE-STATE",
)
DIRECT_ARMS = {"A-FROZEN-DIRECT", "B-TRAINABLE-DIRECT"}
TRAINABLE_ARMS = {"B-TRAINABLE-DIRECT", "D-TRAINABLE-STATE"}


class ExperimentError(RuntimeError):
    pass


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def config() -> dict[str, Any]:
    return load_json(CONFIG_PATH)


def cache_root() -> Path:
    value = os.environ.get("SRSCD_CACHE_ROOT")
    if not value:
        raise ExperimentError("SRSCD_CACHE_ROOT is required")
    root = Path(value).resolve()
    if not root.is_absolute() or root == REPOSITORY_ROOT or REPOSITORY_ROOT in root.parents:
        raise ExperimentError("SRSCD_CACHE_ROOT must be outside the repository")
    return root


def output_root(root: Path) -> Path:
    return root / str(config()["output_relative_path"])


def r7b_root(root: Path) -> Path:
    return root / R7B_OUTPUT_RELATIVE


def _git_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"commit": commit, "dirty": bool(status), "dirty_paths": status}


def _model_identity(root: Path, model_id: str) -> tuple[dict[str, Any], Path]:
    registry = load_json(SOURCE_REGISTRY_PATH)
    matches = [row for row in registry["models"] if row["model_id"] == model_id]
    if matches:
        if len(matches) != 1:
            raise ExperimentError(f"pinned model registry entry is ambiguous: {model_id}")
        identity = matches[0]
        model_root = root / "models" / identity["model_id"] / identity["revision"]
        expected_files = identity["required_files"]
    elif model_id == "eres2netv2-standard-prepool":
        source = registry["eres2netv2"]
        identity = {
            "model_id": source["model_id"],
            "repository": source["checkpoint_repository"],
            "revision": source["checkpoint_revision"],
            "loader_class": "speakerlab.models.eres2net.ERes2NetV2",
        }
        model_root = root / "models" / identity["model_id"] / identity["revision"]
        expected_files = [source["checkpoint_file"], source["checkpoint_config"]]
    else:
        raise ExperimentError(f"pinned model registry entry is unavailable: {model_id}")
    verified_files: list[dict[str, Any]] = []
    for expected in expected_files:
        path = model_root / expected["path"]
        if not path.is_file():
            raise ExperimentError(f"pinned model file is missing: {path}")
        digest = sha256_file(path)
        if digest != expected["sha256"] or path.stat().st_size != int(expected["size_bytes"]):
            raise ExperimentError(f"pinned model file identity differs: {path}")
        verified_files.append(
            {"path": str(path), "sha256": digest, "size_bytes": path.stat().st_size}
        )
    return {**identity, "verified_files": verified_files}, model_root


def _eres_runtime_identity(root: Path) -> tuple[dict[str, Any], Path, Path]:
    registry = load_json(SOURCE_REGISTRY_PATH)
    contract = registry["eres2netv2"]
    acquisition_path = root / "manifests/r1_model_acquisition.json"
    if not acquisition_path.is_file():
        raise ExperimentError("ERes model acquisition receipt is missing")
    acquisition = load_json(acquisition_path)
    matches = [
        row
        for row in acquisition.get("models", [])
        if row.get("model_id") == "eres2netv2-standard-prepool"
    ]
    if len(matches) != 1:
        raise ExperimentError("ERes model acquisition identity is unavailable")
    record = matches[0]
    model_identity, checkpoint_root = _model_identity(root, "eres2netv2-standard-prepool")
    source_root = Path(record["source_root"]).resolve()
    if (
        Path(record["checkpoint_root"]).resolve() != checkpoint_root.resolve()
        or record.get("checkpoint_revision") != contract["checkpoint_revision"]
        or record.get("source_revision") != contract["source_revision"]
    ):
        raise ExperimentError("ERes acquisition roots or revisions differ")
    expected_sources = [*contract["source_files"], contract["source_license"]]
    verified_sources: list[dict[str, Any]] = []
    for expected in expected_sources:
        path = source_root / expected["path"]
        if not path.is_file() or sha256_file(path) != expected["sha256"]:
            raise ExperimentError(f"pinned ERes source file identity differs: {path}")
        verified_sources.append(
            {"path": str(path), "sha256": expected["sha256"], "size_bytes": path.stat().st_size}
        )
    return (
        {
            "checkpoint": model_identity,
            "source_repository": contract["source_repository"],
            "source_revision": contract["source_revision"],
            "verified_source_files": verified_sources,
        },
        checkpoint_root,
        source_root,
    )


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _fold_map(cfg: dict[str, Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    for fold, sessions in enumerate(cfg["folds"]):
        for session_id in sessions:
            if session_id in result:
                raise ExperimentError(f"session appears in multiple folds: {session_id}")
            result[str(session_id)] = fold
    return result


def _upstream_artifact_identities(root: Path, inventory: dict[str, Any]) -> list[dict[str, Any]]:
    identities: list[dict[str, Any]] = []
    for role in ("development", "evaluation"):
        paths = _role_paths(root, role)
        files: dict[str, dict[str, Any]] = {}
        for name in ("dense_vectors", "dense_index", "dense_manifest", "labels"):
            path = paths[name]
            if not path.is_file():
                raise ExperimentError(f"upstream R7-B artifact is missing: {path}")
            files[name] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        manifest = load_json(paths["dense_manifest"])
        if manifest.get("role") != role:
            raise ExperimentError(f"upstream dense manifest role differs: {role}")
        if manifest.get("vectors_sha256") != files["dense_vectors"]["sha256"]:
            raise ExperimentError(f"upstream dense vector identity differs: {role}")
        if manifest.get("index_sha256") != files["dense_index"]["sha256"]:
            raise ExperimentError(f"upstream dense index identity differs: {role}")
        if inventory.get("label_sha256", {}).get(role) != files["labels"]["sha256"]:
            raise ExperimentError(f"upstream partition label identity differs: {role}")
        identities.append({"role": role, "files": files})
    return identities


def prepare(root: Path) -> Path:
    cfg = config()
    inventory_path = r7b_root(root) / "inventory.json"
    if not inventory_path.is_file():
        raise ExperimentError("the completed R7-B inventory is required")
    inventory = load_json(inventory_path)
    fold_map = _fold_map(cfg)
    sessions = {str(row["session_id"]): row for row in inventory["sessions"]}
    if set(fold_map) != set(sessions):
        raise ExperimentError("the issue #72 folds must match the R7-B inventory exactly")
    source_rows: list[dict[str, Any]] = []
    for session_id in sorted(sessions):
        row = sessions[session_id]
        waveform = Path(row["waveform_path"])
        if not waveform.is_file():
            raise ExperimentError(f"source waveform is missing: {waveform}")
        source_rows.append(
            {
                "session_id": session_id,
                "fold": fold_map[session_id],
                "waveform_path": str(waveform.resolve()),
                "waveform_sha256": sha256_file(waveform),
                "scored_hours": float(row["scored_hours"]),
                "event_count": int(row["event_count"]),
            }
        )
    model_identities = []
    for model_id in cfg["model_ids"]:
        identity, _ = _model_identity(root, str(model_id))
        model_identities.append(identity)
    receipt = {
        "schema_version": 1,
        "artifact_role": "psem_trainable_formulation_gate_receipt",
        "experiment_id": cfg["experiment_id"],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "evidence_mode": "development_known_direction_selection",
        "issue": "https://github.com/kapitalismho/PuriPuly-heart/issues/72",
        "config": cfg,
        "config_sha256": sha256_file(CONFIG_PATH),
        "code_sha256": sha256_file(CODE_PATH),
        "source_registry_sha256": sha256_file(SOURCE_REGISTRY_PATH),
        "r7b_inventory_path": str(inventory_path),
        "r7b_inventory_sha256": sha256_file(inventory_path),
        "upstream_r7b_artifacts": _upstream_artifact_identities(root, inventory),
        "model_identities": model_identities,
        "shared_contract": {
            "sample_rate_hz": SAMPLE_RATE,
            "hop_ms": cfg["hop_ms"],
            "encoder_cell_context_ms": cfg["encoder_cell_context_ms"],
            "sequence_frontier_offsets_ms": cfg["sequence_frontier_offsets_ms"],
            "availability_latency_ms": 1000,
            "event_semantics": "R7-B new_speaker_onset",
            "matching_tolerances_ms": [100, 250, 500],
            "false_event_denominator": "continuous scored source hours",
            "duplicate_suppression_ms": cfg["duplicate_suppression_ms"],
            "training_sampling": {
                "natural_background_hop_ms": cfg["training_background_hop_ms"],
                "event_centered_radius_ms": cfg["training_event_radius_ms"],
            },
            "structured_relation": {
                "target": cfg["structured_relation_target"],
                "decoder_source_offsets_ms": [-500, -400, -300, -200, -100, 0],
                "decoder_target_offset_ms": 100,
            },
        },
        "adaptation_recipe": {
            "applies_to": [
                {"model_id": model_id, "arms": ["B-TRAINABLE-DIRECT", "D-TRAINABLE-STATE"]}
                for model_id in cfg["model_ids"]
            ],
            "type": "residual_bottleneck_encoder_adapter",
            "attachment": "pinned_model_final_cell_output",
            "bottleneck_dimension": cfg["adapter_bottleneck_dimension"],
            "pretrained_model_parameters_trainable": 0,
            "adapter_learning_rate": cfg["adapter_learning_rate"],
            "weight_decay": cfg["weight_decay"],
        },
        "arms": [
            {
                "model_id": model_id,
                "id": arm,
                "encoder_adapted": arm in TRAINABLE_ARMS,
                "target": "direct" if arm in DIRECT_ARMS else "structured_state",
            }
            for model_id in cfg["model_ids"]
            for arm in ARMS
        ],
        "sources": source_rows,
        "summary": {
            "source_count": len(source_rows),
            "scored_hours": sum(row["scored_hours"] for row in source_rows),
            "event_count": sum(row["event_count"] for row in source_rows),
        },
        "git": _git_state(),
    }
    directory = output_root(root)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "experiment_receipt.json"
    write_json(path, receipt)
    write_json(directory / "config_used.json", cfg)
    return path


def _feature_path(root: Path, model_id: str, session_id: str) -> Path:
    if model_id == "wavlm-base-plus":
        return output_root(root) / "features" / f"{session_id}.npy"
    return output_root(root) / "features" / model_id / f"{session_id}.npy"


def _feature_manifest_path(root: Path, model_id: str) -> Path:
    if model_id == "wavlm-base-plus":
        return output_root(root) / "feature_manifest.json"
    return output_root(root) / "feature_manifests" / f"{model_id}.json"


def extract(root: Path, model_id: str) -> Path:
    import soundfile as sf

    cfg = config()
    if model_id not in cfg["model_ids"]:
        raise ExperimentError(f"model is outside the frozen comparison: {model_id}")
    receipt_path = output_root(root) / "experiment_receipt.json"
    if not receipt_path.is_file():
        prepare(root)
    receipt = load_json(receipt_path)
    bundles = _load_bundles(root, require_pcm=False)
    runtime_identity = None
    if model_id == "eres2netv2-standard-prepool":
        from experiments.speaker_representation_scd.extraction.eres_prepooling import (
            ERes2NetV2PrepoolExtractor,
        )

        runtime_identity, model_root, source_root = _eres_runtime_identity(root)
        extractor = ERes2NetV2PrepoolExtractor(
            model_root,
            source_root,
            SOURCE_REGISTRY_PATH,
            threads=int(cfg["eres_threads"]),
        )
        dimension = 192
        batch_size = int(cfg["eres_batch_size"])
        encoder_layer = "final_embedding"
        pooling = "official_embedding_then_l2_normalize"
    else:
        _, model_root = _model_identity(root, model_id)
        extractor = SSLExtractor(
            model_id, model_root, threads=int(cfg["wavlm_threads"])
        )
        dimension = int(extractor.model.config.hidden_size)
        batch_size = int(cfg["wavlm_batch_size"])
        encoder_layer = cfg["encoder_layer"]
        pooling = "mean_then_l2_normalize"
    feature_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    context_samples = int(cfg["encoder_cell_context_ms"]) * 16
    for session_id in sorted(bundles):
        path = _feature_path(root, model_id, session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        bundle = bundles[session_id]
        source = next(row for row in receipt["sources"] if row["session_id"] == session_id)
        waveform, sample_rate = sf.read(source["waveform_path"], dtype="float32", always_2d=True)
        if sample_rate != SAMPLE_RATE or waveform.shape[1] != 1:
            raise ExperimentError(f"unexpected waveform geometry: {session_id}")
        mono = waveform[:, 0]
        if int(bundle.frontiers[0]) < context_samples or int(bundle.frontiers[-1]) > len(mono):
            raise ExperimentError(f"WavLM cell windows exceed source audio: {session_id}")
        output = np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=np.float32,
            shape=(len(bundle.frontiers), dimension),
        )
        for batch_start in range(0, len(bundle.frontiers), batch_size):
            frontiers = bundle.frontiers[batch_start : batch_start + batch_size]
            windows = [
                np.ascontiguousarray(mono[int(frontier) - context_samples : int(frontier)])
                for frontier in frontiers
            ]
            if model_id == "eres2netv2-standard-prepool":
                batch = extractor.extract(
                    windows,
                    [int(frontier) for frontier in frontiers],
                    tap_ids=(),
                )
                pooled = np.asarray(batch.official_embedding, dtype=np.float32)
            else:
                batch = extractor.extract(
                    windows,
                    [int(frontier) for frontier in frontiers],
                    layer_ids=[str(cfg["encoder_layer"])],
                )
                values = batch.layers[str(cfg["encoder_layer"])]
                pooled = values.mean(axis=1)
            pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
            output[batch_start : batch_start + len(frontiers)] = pooled
        output.flush()
        del output
        del waveform
        feature_rows.append(
            {
                "session_id": session_id,
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "row_count": len(bundle.frontiers),
                "dimension": dimension,
                "source_waveform_sha256": source["waveform_sha256"],
                "frontiers_sha256": _array_sha256(bundle.frontiers),
            }
        )
        print(
            json.dumps(
                {
                    "stage": "encoder_extract",
                    "model_id": model_id,
                    "session_id": session_id,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    manifest = {
        "schema_version": 1,
        "artifact_role": "encoder_cell_feature_manifest",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model_id": model_id,
        "encoder_layer": encoder_layer,
        "cell_context_ms": cfg["encoder_cell_context_ms"],
        "pooling": pooling,
        "dtype": "float32",
        "code_sha256": sha256_file(CODE_PATH),
        "config_sha256": sha256_file(CONFIG_PATH),
        "receipt_sha256": sha256_file(receipt_path),
        "wall_seconds": time.perf_counter() - started,
        "hardware": {
            "backend": "cpu",
            "cpu_count": os.cpu_count(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "runtime_identity": runtime_identity,
        "sessions": feature_rows,
    }
    path = _feature_manifest_path(root, model_id)
    write_json(path, manifest)
    return path


@dataclass(slots=True)
class FeatureBundle:
    base: SessionBundle
    features: np.ndarray
    direct_labels: np.ndarray


def _direct_labels(bundle: SessionBundle, radius_samples: int) -> np.ndarray:
    labels = np.zeros(len(bundle.frontiers), dtype=np.float32)
    events = np.asarray([int(row["sample"]) for row in bundle.events], dtype=np.int64)
    if not len(events):
        return labels
    for index, frontier in enumerate(bundle.frontiers):
        labels[index] = float(np.min(np.abs(events - int(frontier))) <= radius_samples)
    return labels


def _augment_match(
    prediction_id: int,
    adjacency: dict[int, list[tuple[str, int]]],
    matched_prediction: dict[int, tuple[str, int]],
    matched_reference: dict[tuple[str, int], int],
) -> bool:
    queue = deque([prediction_id])
    seen_predictions = {prediction_id}
    seen_references: set[tuple[str, int]] = set()
    parent: dict[int, tuple[int, tuple[str, int]]] = {}
    while queue:
        current = queue.popleft()
        for reference in adjacency[current]:
            if reference in seen_references:
                continue
            seen_references.add(reference)
            other = matched_reference.get(reference)
            if other is None:
                active_prediction = current
                active_reference = reference
                while True:
                    matched_prediction[active_prediction] = active_reference
                    matched_reference[active_reference] = active_prediction
                    if active_prediction == prediction_id:
                        return True
                    previous_prediction, previous_reference = parent[active_prediction]
                    active_prediction = previous_prediction
                    active_reference = previous_reference
            if other not in seen_predictions:
                seen_predictions.add(other)
                parent[other] = (current, reference)
                queue.append(other)
    return False


def _full_threshold_rows(
    bundles: dict[str, SessionBundle], peaks: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    tolerances = (100, 250, 500)
    references = {
        session_id: sorted(int(event["sample"]) for event in bundle.events)
        for session_id, bundle in bundles.items()
    }
    reference_count = sum(len(values) for values in references.values())
    exposure_hours = sum(bundle.scored_hours for bundle in bundles.values())
    states = {
        tolerance: {
            "adjacency": {},
            "matched_prediction": {},
            "matched_reference": {},
        }
        for tolerance in tolerances
    }
    rows = [
        {
            "threshold": math.nextafter(float(peaks[0]["score"]), math.inf)
            if peaks
            else math.inf,
            "prediction_count": 0,
            "macro_f1": 0.0,
            "tolerances": {
                str(tolerance): {
                    "true_positive_count": 0,
                    "false_event_count": 0,
                    "false_events_per_hour": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                }
                for tolerance in tolerances
            },
        }
    ]
    index = 0
    while index < len(peaks):
        score = float(peaks[index]["score"])
        end = index + 1
        while end < len(peaks) and float(peaks[end]["score"]) == score:
            end += 1
        for prediction_id in range(index, end):
            peak = peaks[prediction_id]
            session_id = str(peak["session_id"])
            boundary = int(peak["boundary_sample"])
            session_references = references[session_id]
            for tolerance in tolerances:
                radius = tolerance * 16
                left = bisect_left(session_references, boundary - radius)
                right = bisect_right(session_references, boundary + radius)
                state = states[tolerance]
                state["adjacency"][prediction_id] = [
                    (session_id, reference_index)
                    for reference_index in range(left, right)
                ]
                _augment_match(
                    prediction_id,
                    state["adjacency"],
                    state["matched_prediction"],
                    state["matched_reference"],
                )
        prediction_count = end
        tolerance_rows: dict[str, dict[str, Any]] = {}
        f1_values: list[float] = []
        for tolerance in tolerances:
            true_positive = len(states[tolerance]["matched_reference"])
            false_events = prediction_count - true_positive
            precision = true_positive / prediction_count if prediction_count else 0.0
            recall = true_positive / reference_count if reference_count else 0.0
            f1 = (
                2.0 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            f1_values.append(f1)
            tolerance_rows[str(tolerance)] = {
                "true_positive_count": true_positive,
                "false_event_count": false_events,
                "false_events_per_hour": false_events / exposure_hours,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        rows.append(
            {
                "threshold": score,
                "prediction_count": prediction_count,
                "macro_f1": float(np.mean(f1_values)),
                "tolerances": tolerance_rows,
            }
        )
        index = end
    return rows


def _sample_curve(
    rows: Sequence[dict[str, Any]], required_indices: set[int], sample_count: int
) -> list[dict[str, Any]]:
    if len(rows) <= sample_count:
        return list(rows)
    indices = set(required_indices)
    indices.add(0)
    indices.add(len(rows) - 1)
    linear = np.linspace(0, len(rows) - 1, sample_count, dtype=np.int64)
    indices.update(int(value) for value in linear)
    return [rows[index] for index in sorted(indices)]


def _curve_and_points(
    bundles: dict[str, SessionBundle], rows: Sequence[dict[str, Any]], cfg: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    peaks = _peaks(rows, int(cfg["duplicate_suppression_ms"]) * 16)
    threshold_rows = _full_threshold_rows(bundles, peaks)
    required_indices: set[int] = set()
    reference_points: dict[str, Any] = {}
    for target in cfg["development_false_event_targets_per_hour"]:
        eligible = [
            (index, row)
            for index, row in enumerate(threshold_rows)
            if float(row["tolerances"]["250"]["false_events_per_hour"]) <= target
        ]
        selected_index, selected_row = max(
            eligible,
            key=lambda value: (
                float(value[1]["tolerances"]["250"]["recall"]),
                -float(value[1]["tolerances"]["250"]["false_events_per_hour"]),
            ),
        )
        required_indices.add(selected_index)
        reference_points[str(target)] = {
            "threshold": selected_row["threshold"],
            "metrics": _metrics(bundles, peaks[: int(selected_row["prediction_count"])]),
        }
    per_tolerance: dict[str, Any] = {}
    for tolerance in (100, 250, 500):
        selected_index, selected_row = max(
            enumerate(threshold_rows),
            key=lambda value: float(value[1]["tolerances"][str(tolerance)]["f1"]),
        )
        required_indices.add(selected_index)
        per_tolerance[str(tolerance)] = {
            "threshold": selected_row["threshold"],
            "metrics": _metrics(bundles, peaks[: int(selected_row["prediction_count"])]),
        }
    macro_index, macro_row = max(
        enumerate(threshold_rows), key=lambda value: float(value[1]["macro_f1"])
    )
    required_indices.add(macro_index)
    unrestricted_index = len(threshold_rows) - 1
    required_indices.add(unrestricted_index)
    frontier = {
        "best_macro_f1_operating_point": {
            "threshold": macro_row["threshold"],
            "macro_f1": macro_row["macro_f1"],
            "metrics": _metrics(bundles, peaks[: int(macro_row["prediction_count"])]),
        },
        "best_f1_by_tolerance": per_tolerance,
        "unrestricted_operating_point": {
            "threshold": threshold_rows[unrestricted_index]["threshold"],
            "metrics": _metrics(bundles, peaks),
        },
    }
    verification = {
        index
        for index in required_indices
        if index == 0 or index == len(threshold_rows) - 1 or index % 97 == 0
    }
    verification.update(required_indices)
    for index in sorted(verification):
        row = threshold_rows[index]
        metrics = _metrics(bundles, peaks[: int(row["prediction_count"])] )
        for tolerance in (100, 250, 500):
            expected = row["tolerances"][str(tolerance)]
            actual = metrics["tolerances"][str(tolerance)]
            if (
                int(expected["true_positive_count"]) != int(actual["true_positive_count"])
                or int(expected["false_event_count"]) != int(actual["false_event_count"])
            ):
                raise ExperimentError("incremental frontier differs from one-to-one matching")
    return (
        _sample_curve(
            threshold_rows,
            required_indices,
            int(cfg["frontier_curve_sample_count"]),
        ),
        reference_points,
        frontier,
    )


def _load_feature_bundles(root: Path, model_id: str) -> dict[str, FeatureBundle]:
    manifest_path = _feature_manifest_path(root, model_id)
    if not manifest_path.is_file():
        raise ExperimentError("WavLM feature manifest is required")
    manifest = load_json(manifest_path)
    base = _load_bundles(root, require_pcm=False)
    rows = {str(row["session_id"]): row for row in manifest["sessions"]}
    result: dict[str, FeatureBundle] = {}
    radius = int(config()["direct_positive_radius_ms"]) * 16
    for session_id, bundle in base.items():
        row = rows.get(session_id)
        if row is None:
            raise ExperimentError(f"encoder feature identity is missing: {model_id}/{session_id}")
        path = _feature_path(root, model_id, session_id)
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise ExperimentError(f"encoder feature identity differs: {model_id}/{session_id}")
        features = np.load(path, mmap_mode="r")
        if len(features) != len(bundle.frontiers):
            raise ExperimentError(f"WavLM feature geometry differs: {session_id}")
        result[session_id] = FeatureBundle(
            base=bundle,
            features=features,
            direct_labels=_direct_labels(bundle, radius),
        )
    return result


def _normalization(
    bundles: dict[str, FeatureBundle], session_ids: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    count = 0
    total: np.ndarray | None = None
    square: np.ndarray | None = None
    for session_id in session_ids:
        values = np.asarray(bundles[session_id].features, dtype=np.float64)
        if total is None:
            total = np.zeros(values.shape[1], dtype=np.float64)
            square = np.zeros(values.shape[1], dtype=np.float64)
        total += values.sum(axis=0)
        square += np.square(values).sum(axis=0)
        count += len(values)
    if total is None or square is None or count == 0:
        raise ExperimentError("normalization set is empty")
    mean = total / count
    variance = np.maximum(square / count - np.square(mean), 1e-8)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def _batch_arrays(
    bundles: dict[str, FeatureBundle],
    references: Sequence[tuple[str, int]],
    mean: np.ndarray,
    scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values: list[np.ndarray] = []
    direct: list[float] = []
    states: list[np.ndarray] = []
    speakers: list[np.ndarray] = []
    for session_id, center in references:
        bundle = bundles[session_id]
        start = center - 5
        end = center + 11
        values.append((np.asarray(bundle.features[start:end]) - mean) / scale)
        direct.append(float(bundle.direct_labels[center]))
        states.append(np.asarray(bundle.base.state[start:end]))
        speakers.append(np.asarray(bundle.base.speaker[start:end]))
    return (
        np.asarray(values, dtype=np.float32),
        np.asarray(direct, dtype=np.float32),
        np.asarray(states, dtype=np.int64),
        np.asarray(speakers, dtype=np.int64),
    )


def _model_class():
    import torch

    class FormulationModel(torch.nn.Module):
        def __init__(self, input_dimension: int, cfg: dict[str, Any], arm: str) -> None:
            super().__init__()
            hidden = int(cfg["hidden_dimension"])
            pair = int(cfg["pair_dimension"])
            self.arm = arm
            self.encoder_adapter = None
            if arm in TRAINABLE_ARMS:
                bottleneck = int(cfg["adapter_bottleneck_dimension"])
                self.encoder_adapter = torch.nn.Sequential(
                    torch.nn.LayerNorm(input_dimension),
                    torch.nn.Linear(input_dimension, bottleneck),
                    torch.nn.GELU(),
                    torch.nn.Linear(bottleneck, input_dimension),
                )
                torch.nn.init.zeros_(self.encoder_adapter[-1].weight)
                torch.nn.init.zeros_(self.encoder_adapter[-1].bias)
            self.input_projection = torch.nn.Sequential(
                torch.nn.Linear(input_dimension, hidden),
                torch.nn.GELU(),
                torch.nn.LayerNorm(hidden),
            )
            self.temporal = torch.nn.GRU(
                hidden,
                hidden // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            if arm in DIRECT_ARMS:
                self.direct_head = torch.nn.Sequential(
                    torch.nn.Linear(hidden * 4, hidden),
                    torch.nn.GELU(),
                    torch.nn.Linear(hidden, 1),
                )
            else:
                self.pair_projection = torch.nn.Sequential(
                    torch.nn.Linear(hidden, pair), torch.nn.GELU()
                )
                self.pair_head = torch.nn.Linear(pair * 2, 1)
                self.state_head = torch.nn.Linear(hidden, 3)

        def forward(self, values):
            if self.encoder_adapter is not None:
                values = values + self.encoder_adapter(values)
            hidden, _ = self.temporal(self.input_projection(values))
            if self.arm in DIRECT_ARMS:
                left = hidden[:, 5]
                right = hidden[:, 6]
                pair = torch.cat([left, right, torch.abs(left - right), left * right], dim=-1)
                return {"direct_logits": self.direct_head(pair).squeeze(-1)}
            pair_values = self.pair_projection(hidden)
            left = pair_values[:, :, None, :]
            right = pair_values[:, None, :, :]
            relation = torch.cat([torch.abs(left - right), left * right], dim=-1)
            return {
                "pair_logits": self.pair_head(relation).squeeze(-1),
                "state_logits": self.state_head(hidden),
            }

    return FormulationModel


def _structured_loss(outputs, states, speakers, state_weight: float):
    import torch

    pair_logits = outputs["pair_logits"]
    state_logits = outputs["state_logits"]
    sequence_length = states.shape[1]
    singleton = states == STATE_SINGLE
    pair_mask = torch.zeros_like(pair_logits, dtype=torch.bool)
    for source in range(sequence_length - 1):
        for target in range(source + 1, sequence_length):
            silence_between = torch.all(
                states[:, source + 1 : target] == STATE_SILENCE, dim=1
            )
            pair_mask[:, source, target] = (
                singleton[:, source] & singleton[:, target] & silence_between
            )
    pair_targets = (speakers[:, :, None] == speakers[:, None, :]).float()
    selected_logits = pair_logits[pair_mask]
    selected_targets = pair_targets[pair_mask]
    if selected_logits.numel() == 0:
        pair_loss = pair_logits.sum() * 0.0
    else:
        positive = torch.clamp(selected_targets.sum(), min=1.0)
        negative = torch.clamp((1.0 - selected_targets).sum(), min=1.0)
        weights = torch.where(
            selected_targets > 0.5,
            0.5 * selected_targets.numel() / positive,
            0.5 * selected_targets.numel() / negative,
        )
        pair_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            selected_logits, selected_targets, weight=weights
        )
    valid_state = states >= 0
    selected_states = states[valid_state]
    selected_state_logits = state_logits[valid_state]
    counts = torch.bincount(selected_states, minlength=3).float().clamp(min=1.0)
    class_weights = counts.sum() / (3.0 * counts)
    state_loss = torch.nn.functional.cross_entropy(
        selected_state_logits, selected_states, weight=class_weights
    )
    return pair_loss + state_weight * state_loss


def _loss(outputs, direct, states, speakers, arm: str, cfg: dict[str, Any], pos_weight):
    import torch

    if arm in DIRECT_ARMS:
        return torch.nn.functional.binary_cross_entropy_with_logits(
            outputs["direct_logits"], direct, pos_weight=pos_weight
        )
    return _structured_loss(outputs, states, speakers, float(cfg["state_loss_weight"]))


def _iterate_batches(
    references: Sequence[tuple[str, int]], batch_size: int, rng: np.random.Generator
) -> Iterable[list[tuple[str, int]]]:
    order = rng.permutation(len(references))
    for start in range(0, len(order), batch_size):
        yield [references[int(index)] for index in order[start : start + batch_size]]


def _validation_loss(
    model,
    bundles: dict[str, FeatureBundle],
    references: Sequence[tuple[str, int]],
    mean: np.ndarray,
    scale: np.ndarray,
    arm: str,
    cfg: dict[str, Any],
    pos_weight,
) -> float:
    import torch

    model.eval()
    losses: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(references), int(cfg["batch_size"])):
            batch = references[start : start + int(cfg["batch_size"])]
            x, direct, states, speakers = _batch_arrays(bundles, batch, mean, scale)
            outputs = model(torch.from_numpy(x))
            value = _loss(
                outputs,
                torch.from_numpy(direct),
                torch.from_numpy(states),
                torch.from_numpy(speakers),
                arm,
                cfg,
                pos_weight,
            )
            losses.append(float(value))
    return float(np.mean(losses)) if losses else math.inf


def _fit_model(
    bundles: dict[str, FeatureBundle],
    train_sessions: Sequence[str],
    validation_sessions: Sequence[str],
    arm: str,
    seed: int,
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, int, float, dict[str, int]]:
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    mean, scale = _normalization(bundles, train_sessions)
    train_references = [
        (session_id, center)
        for session_id in train_sessions
        for center in _training_centers(bundles[session_id].base, cfg)
    ]
    validation_references = [
        (session_id, center)
        for session_id in validation_sessions
        for center in _training_centers(bundles[session_id].base, cfg)
    ]
    if not train_references or not validation_references:
        raise ExperimentError("training or validation examples are empty")
    positive = sum(bundles[session_id].direct_labels[center] for session_id, center in train_references)
    negative = len(train_references) - positive
    pos_weight = torch.tensor(max(float(negative / max(positive, 1.0)), 1.0))
    input_dimension = int(bundles[train_sessions[0]].features.shape[1])
    model = _model_class()(input_dimension, cfg, arm)
    adapter_parameters = (
        list(model.encoder_adapter.parameters()) if model.encoder_adapter is not None else []
    )
    adapter_ids = {id(parameter) for parameter in adapter_parameters}
    head_parameters = [parameter for parameter in model.parameters() if id(parameter) not in adapter_ids]
    parameter_groups = [
        {"params": head_parameters, "lr": float(cfg["head_learning_rate"])}
    ]
    if adapter_parameters:
        parameter_groups.append(
            {"params": adapter_parameters, "lr": float(cfg["adapter_learning_rate"])}
        )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=float(cfg["weight_decay"]),
    )
    best_state: dict[str, Any] | None = None
    best_loss = math.inf
    best_epoch = 0
    stale = 0
    rng = np.random.default_rng(seed)
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        for batch in _iterate_batches(train_references, int(cfg["batch_size"]), rng):
            x, direct, states, speakers = _batch_arrays(bundles, batch, mean, scale)
            outputs = model(torch.from_numpy(x))
            value = _loss(
                outputs,
                torch.from_numpy(direct),
                torch.from_numpy(states),
                torch.from_numpy(speakers),
                arm,
                cfg,
                pos_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        validation = _validation_loss(
            model,
            bundles,
            validation_references,
            mean,
            scale,
            arm,
            cfg,
            pos_weight,
        )
        if validation < best_loss - 1e-5:
            best_loss = validation
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= int(cfg["early_stopping_patience"]):
                break
    if best_state is None:
        raise ExperimentError("training did not produce a finite model")
    counts = {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "encoder_adapter": sum(parameter.numel() for parameter in adapter_parameters),
        "task_head": sum(parameter.numel() for parameter in head_parameters),
        "train_examples": len(train_references),
        "validation_examples": len(validation_references),
    }
    return best_state, mean, scale, best_epoch, best_loss, counts


def _structured_scores(outputs):
    import torch

    pair = torch.sigmoid(outputs["pair_logits"])
    state = torch.softmax(outputs["state_logits"], dim=-1)
    left = 5
    right = 6
    candidates = [
        state[:, left, STATE_SINGLE]
        * state[:, right, STATE_SINGLE]
        * (1.0 - pair[:, left, right])
    ]
    for source in range(left - 1, -1, -1):
        silence = torch.prod(state[:, source + 1 : right, STATE_SILENCE], dim=1)
        candidates.append(
            state[:, source, STATE_SINGLE]
            * silence
            * state[:, right, STATE_SINGLE]
            * (1.0 - pair[:, source, right])
        )
    candidates.append(
        (state[:, left, STATE_SILENCE] + state[:, left, STATE_SINGLE])
        * state[:, right, STATE_OVERLAP]
    )
    return (
        torch.stack(candidates, dim=1).max(dim=1).values,
        state[:, left],
        pair[:, :right, right],
    )


def _score_session(
    bundle: FeatureBundle,
    arm: str,
    states: Sequence[dict[str, Any]],
    means: Sequence[np.ndarray],
    scales: Sequence[np.ndarray],
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    import torch

    references = [(bundle.base.session_id, center) for center in _valid_centers(bundle.base)]
    seed_scores: list[np.ndarray] = []
    seed_states: list[np.ndarray] = []
    seed_relations: list[np.ndarray] = []
    input_dimension = int(bundle.features.shape[1])
    local = {bundle.base.session_id: bundle}
    for state_dict, mean, scale in zip(states, means, scales, strict=True):
        model = _model_class()(input_dimension, cfg, arm)
        model.load_state_dict(state_dict)
        model.eval()
        score_batches: list[np.ndarray] = []
        state_batches: list[np.ndarray] = []
        relation_batches: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(references), int(cfg["batch_size"])):
                batch = references[start : start + int(cfg["batch_size"])]
                x, _, _, _ = _batch_arrays(local, batch, mean, scale)
                outputs = model(torch.from_numpy(x))
                if arm in DIRECT_ARMS:
                    score_batches.append(torch.sigmoid(outputs["direct_logits"]).numpy())
                else:
                    score, state_probability, relation_probability = _structured_scores(outputs)
                    score_batches.append(score.numpy())
                    state_batches.append(state_probability.numpy())
                    relation_batches.append(relation_probability.numpy())
        seed_scores.append(np.concatenate(score_batches))
        if state_batches:
            seed_states.append(np.concatenate(state_batches))
            seed_relations.append(np.concatenate(relation_batches))
    scores = np.mean(seed_scores, axis=0).astype(np.float64)
    state_values = np.mean(seed_states, axis=0).astype(np.float64) if seed_states else None
    relation_values = (
        np.mean(seed_relations, axis=0).astype(np.float64) if seed_relations else None
    )
    return scores, state_values, relation_values


def _prediction_rows(
    bundles: dict[str, FeatureBundle],
    scores: dict[str, np.ndarray],
    state_values: dict[str, np.ndarray],
    relation_values: dict[str, np.ndarray],
    model_id: str,
    arm: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for session_id in sorted(bundles):
        bundle = bundles[session_id]
        centers = list(_valid_centers(bundle.base))
        if len(centers) != len(scores[session_id]):
            raise ExperimentError(f"prediction geometry differs: {session_id}")
        for position, center in enumerate(centers):
            row: dict[str, Any] = {
                "model_id": model_id,
                "arm": arm,
                "session_id": session_id,
                "fold": bundle.base.fold,
                "boundary_sample": int(bundle.base.frontiers[center]),
                "score": float(scores[session_id][position]),
            }
            if session_id in state_values:
                row["state_probabilities"] = [
                    float(value) for value in state_values[session_id][position]
                ]
                source_probabilities = relation_values[session_id][position]
                row["decoder_source_same_probabilities"] = [
                    float(value) for value in source_probabilities
                ]
                row["adjacent_same_probability"] = float(source_probabilities[-1])
            rows.append(row)
    return rows


def _class_metrics(targets: np.ndarray, predictions: np.ndarray, class_count: int) -> dict[str, Any]:
    names = ("silence", "singleton", "overlap")
    per_class: dict[str, Any] = {}
    f1_values: list[float] = []
    for class_id in range(class_count):
        true_positive = int(np.sum((targets == class_id) & (predictions == class_id)))
        false_positive = int(np.sum((targets != class_id) & (predictions == class_id)))
        false_negative = int(np.sum((targets == class_id) & (predictions != class_id)))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[names[class_id]] = {
            "support": int(np.sum(targets == class_id)),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return {
        "count": len(targets),
        "accuracy": float(np.mean(targets == predictions)) if len(targets) else None,
        "macro_f1": float(np.mean(f1_values)),
        "per_class": per_class,
    }


def _binary_metrics(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    predictions = probabilities >= 0.5
    positive = targets == 1
    negative = targets == 0
    true_positive = int(np.sum(predictions & positive))
    true_negative = int(np.sum(~predictions & negative))
    false_positive = int(np.sum(predictions & negative))
    false_negative = int(np.sum(~predictions & positive))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "count": len(targets),
        "same_count": int(np.sum(positive)),
        "different_count": int(np.sum(negative)),
        "accuracy": float(np.mean(predictions == positive)) if len(targets) else None,
        "balanced_accuracy": (recall + specificity) / 2.0,
        "same_speaker_precision": precision,
        "same_speaker_recall": recall,
        "same_speaker_f1": f1,
        "different_speaker_recall": specificity,
    }


def _structured_diagnostics(
    rows: Sequence[dict[str, Any]], bundles: dict[str, FeatureBundle]
) -> dict[str, Any]:
    state_targets: list[int] = []
    state_predictions: list[int] = []
    relation_targets: list[int] = []
    relation_probabilities: list[float] = []
    adjacent_targets: list[int] = []
    adjacent_probabilities: list[float] = []
    silence_gap_targets: list[int] = []
    silence_gap_probabilities: list[float] = []
    index_by_frontier = {
        session_id: {int(value): index for index, value in enumerate(bundle.base.frontiers)}
        for session_id, bundle in bundles.items()
    }
    for row in rows:
        session_id = str(row["session_id"])
        center = index_by_frontier[session_id][int(row["boundary_sample"])]
        bundle = bundles[session_id]
        target = int(bundle.base.state[center])
        if target >= 0:
            state_targets.append(target)
            state_predictions.append(int(np.argmax(row["state_probabilities"])))
        right = center + 1
        source_probabilities = row["decoder_source_same_probabilities"]
        if len(source_probabilities) != 6:
            raise ExperimentError("decoder relation prediction geometry differs")
        for source_position, source in enumerate(range(center - 5, center + 1)):
            if (
                int(bundle.base.state[source]) == STATE_SINGLE
                and int(bundle.base.state[right]) == STATE_SINGLE
                and np.all(bundle.base.state[source + 1 : right] == STATE_SILENCE)
            ):
                target_relation = int(
                    bundle.base.speaker[source] == bundle.base.speaker[right]
                )
                probability = float(source_probabilities[source_position])
                relation_targets.append(target_relation)
                relation_probabilities.append(probability)
                if source < center:
                    silence_gap_targets.append(target_relation)
                    silence_gap_probabilities.append(probability)
                else:
                    adjacent_targets.append(target_relation)
                    adjacent_probabilities.append(probability)
    return {
        "speech_state": _class_metrics(
            np.asarray(state_targets), np.asarray(state_predictions), 3
        ),
        "adjacent_singleton_relation": _binary_metrics(
            np.asarray(adjacent_targets), np.asarray(adjacent_probabilities)
        ),
        "silence_gap_singleton_relation": _binary_metrics(
            np.asarray(silence_gap_targets), np.asarray(silence_gap_probabilities)
        ),
        "decoder_singleton_relation": _binary_metrics(
            np.asarray(relation_targets), np.asarray(relation_probabilities)
        ),
    }


def _false_examples(
    rows: Sequence[dict[str, Any]],
    bundles: dict[str, FeatureBundle],
    threshold: float,
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    peaks = _peaks(rows, int(cfg["duplicate_suppression_ms"]) * 16)
    selected = [row for row in peaks if float(row["score"]) >= threshold]
    result: list[dict[str, Any]] = []
    for row in selected:
        bundle = bundles[str(row["session_id"])].base
        prediction = int(row["boundary_sample"])
        nearest = (
            min(bundle.events, key=lambda event: abs(int(event["sample"]) - prediction))
            if bundle.events
            else None
        )
        distance = (
            abs(int(nearest["sample"]) - prediction) if nearest is not None else None
        )
        if distance is None or distance > 250 * 16:
            result.append(
                {
                    "session_id": row["session_id"],
                    "boundary_sample": prediction,
                    "boundary_seconds": prediction / SAMPLE_RATE,
                    "score": float(row["score"]),
                    "nearest_reference_distance_ms": distance / 16 if distance is not None else None,
                    "nearest_reference_stratum": nearest.get("stratum") if nearest else None,
                }
            )
    return sorted(result, key=lambda row: -float(row["score"]))[:20]


def _arm_paths(root: Path, model_id: str, arm: str) -> dict[str, Path]:
    directory = output_root(root)
    slug = arm.lower().replace("-", "_")
    return {
        "predictions": directory / "predictions" / model_id / f"{slug}.jsonl",
        "metrics": directory / "metrics" / model_id / f"{slug}.json",
        "models": directory / "models" / model_id / slug,
        "false_examples": directory / "false_examples" / model_id / f"{slug}.json",
    }


def develop(root: Path, model_id: str, arm: str) -> Path:
    import torch

    if arm not in ARMS:
        raise ExperimentError(f"unknown arm: {arm}")
    cfg = config()
    if model_id not in cfg["model_ids"]:
        raise ExperimentError(f"model is outside the frozen comparison: {model_id}")
    bundles = _load_feature_bundles(root, model_id)
    all_sessions = sorted(bundles)
    paths = _arm_paths(root, model_id, arm)
    paths["models"].mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    model_receipts: list[dict[str, Any]] = []
    oof_scores: dict[str, np.ndarray] = {}
    oof_states: dict[str, np.ndarray] = {}
    oof_relations: dict[str, np.ndarray] = {}
    parameter_counts: dict[str, int] | None = None
    for fold_index, held_out_sessions in enumerate(cfg["folds"]):
        validation_sessions = cfg["folds"][(fold_index + 1) % len(cfg["folds"])]
        excluded = set(held_out_sessions) | set(validation_sessions)
        train_sessions = [session for session in all_sessions if session not in excluded]
        seed_states: list[dict[str, Any]] = []
        seed_means: list[np.ndarray] = []
        seed_scales: list[np.ndarray] = []
        for seed in cfg["seeds"]:
            state, mean, scale, best_epoch, best_loss, counts = _fit_model(
                bundles,
                train_sessions,
                validation_sessions,
                arm,
                int(seed),
                cfg,
            )
            parameter_counts = counts
            seed_states.append(state)
            seed_means.append(mean)
            seed_scales.append(scale)
            checkpoint = paths["models"] / f"fold_{fold_index + 1}_seed_{seed}.pt"
            torch.save(
                {
                    "schema_version": 1,
                    "arm": arm,
                    "fold": fold_index + 1,
                    "seed": int(seed),
                    "state_dict": state,
                    "mean": mean,
                    "scale": scale,
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_loss,
                    "train_sessions": train_sessions,
                    "validation_sessions": list(validation_sessions),
                    "held_out_sessions": list(held_out_sessions),
                    "code_sha256": sha256_file(CODE_PATH),
                    "config_sha256": sha256_file(CONFIG_PATH),
                },
                checkpoint,
            )
            model_receipts.append(
                {
                    "fold": fold_index + 1,
                    "seed": int(seed),
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_loss,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "train_sessions": train_sessions,
                    "validation_sessions": list(validation_sessions),
                    "held_out_sessions": list(held_out_sessions),
                }
            )
        for session_id in held_out_sessions:
            scores, states, relations = _score_session(
                bundles[session_id],
                arm,
                seed_states,
                seed_means,
                seed_scales,
                cfg,
            )
            oof_scores[session_id] = scores
            if states is not None and relations is not None:
                oof_states[session_id] = states
                oof_relations[session_id] = relations
        print(
            json.dumps(
                {
                    "stage": "development",
                    "model_id": model_id,
                    "arm": arm,
                    "fold": fold_index + 1,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    rows = _prediction_rows(
        bundles, oof_scores, oof_states, oof_relations, model_id, arm
    )
    write_jsonl(paths["predictions"], rows)
    base_bundles = {session_id: bundle.base for session_id, bundle in bundles.items()}
    curve, points, frontier = _curve_and_points(base_bundles, rows, cfg)
    diagnostics = (
        _structured_diagnostics(rows, bundles) if arm not in DIRECT_ARMS else None
    )
    false_examples = _false_examples(
        rows,
        bundles,
        float(points["10"]["threshold"]),
        cfg,
    )
    write_json(paths["false_examples"], false_examples)
    result = {
        "schema_version": 1,
        "model_id": model_id,
        "arm": arm,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "encoder": "adapted" if arm in TRAINABLE_ARMS else "frozen",
        "target": "direct" if arm in DIRECT_ARMS else "structured_state",
        "evidence_mode": "development_known_direction_selection",
        "code_sha256": sha256_file(CODE_PATH),
        "config_sha256": sha256_file(CONFIG_PATH),
        "receipt_sha256": sha256_file(output_root(root) / "experiment_receipt.json"),
        "feature_manifest_sha256": sha256_file(_feature_manifest_path(root, model_id)),
        "wall_seconds": time.perf_counter() - started,
        "parameter_counts": parameter_counts,
        "model_receipts": model_receipts,
        "prediction_path": str(paths["predictions"]),
        "prediction_sha256": sha256_file(paths["predictions"]),
        "curve": curve,
        "selected_operating_points": points,
        "best_macro_f1_operating_point": frontier["best_macro_f1_operating_point"],
        "best_f1_by_tolerance": frontier["best_f1_by_tolerance"],
        "unrestricted_operating_point": frontier["unrestricted_operating_point"],
        "structured_diagnostics": diagnostics,
        "false_examples_path": str(paths["false_examples"]),
        "false_examples_sha256": sha256_file(paths["false_examples"]),
    }
    write_json(paths["metrics"], result)
    return paths["metrics"]


def _point_row(document: dict[str, Any], target: int) -> dict[str, Any]:
    metrics = document["selected_operating_points"][str(target)]["metrics"]
    return {
        "recall_100": float(metrics["tolerances"]["100"]["recall"] or 0.0),
        "recall_250": float(metrics["tolerances"]["250"]["recall"] or 0.0),
        "recall_500": float(metrics["tolerances"]["500"]["recall"] or 0.0),
        "precision_250": float(metrics["tolerances"]["250"]["precision"] or 0.0),
        "f1_250": float(metrics["tolerances"]["250"]["f1"] or 0.0),
        "false_events_per_hour_250": float(
            metrics["tolerances"]["250"]["false_events_per_hour"]
        ),
        "threshold": float(document["selected_operating_points"][str(target)]["threshold"]),
    }


def _frontier_row(document: dict[str, Any]) -> dict[str, Any]:
    point = document["best_macro_f1_operating_point"]
    metrics = point["metrics"]["tolerances"]
    return {
        "threshold": float(point["threshold"]),
        "macro_f1": float(point["macro_f1"]),
        "tolerances": {
            tolerance: {
                name: float(metrics[tolerance][name] or 0.0)
                for name in ("precision", "recall", "f1", "false_events_per_hour")
            }
            for tolerance in ("100", "250", "500")
        },
    }


def _require_artifact(path: Path, expected_sha256: str, expected_size: int | None = None) -> None:
    if not path.is_file():
        raise ExperimentError(f"artifact is missing: {path}")
    if sha256_file(path) != expected_sha256:
        raise ExperimentError(f"artifact identity differs: {path}")
    if expected_size is not None and path.stat().st_size != expected_size:
        raise ExperimentError(f"artifact size differs: {path}")


def _jsonl_row_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def _same_json_value(left: Any, right: Any) -> bool:
    options = {"sort_keys": True, "separators": (",", ":")}
    return json.dumps(left, **options) == json.dumps(right, **options)


def _validate_artifacts(root: Path) -> dict[str, Any]:
    cfg = config()
    current_git = _git_state()
    if current_git["dirty"]:
        raise ExperimentError("artifact verification requires a clean worktree")
    current_code_sha = sha256_file(CODE_PATH)
    current_config_sha = sha256_file(CONFIG_PATH)
    receipt_path = output_root(root) / "experiment_receipt.json"
    if not receipt_path.is_file():
        raise ExperimentError("experiment receipt is missing")
    receipt = load_json(receipt_path)
    if receipt.get("git") != current_git:
        raise ExperimentError("experiment receipt is not bound to the current clean commit")
    if receipt.get("code_sha256") != current_code_sha:
        raise ExperimentError("experiment receipt code identity differs")
    if receipt.get("config_sha256") != current_config_sha or receipt.get("config") != cfg:
        raise ExperimentError("experiment receipt configuration differs")
    if receipt.get("source_registry_sha256") != sha256_file(SOURCE_REGISTRY_PATH):
        raise ExperimentError("experiment receipt model registry identity differs")
    config_used_path = output_root(root) / "config_used.json"
    if not config_used_path.is_file() or load_json(config_used_path) != cfg:
        raise ExperimentError("recorded experiment configuration differs")
    inventory_path = Path(receipt["r7b_inventory_path"])
    _require_artifact(inventory_path, str(receipt["r7b_inventory_sha256"]))
    inventory = load_json(inventory_path)
    if receipt.get("upstream_r7b_artifacts") != _upstream_artifact_identities(root, inventory):
        raise ExperimentError("upstream R7-B artifact identities differ")
    expected_sessions = {str(row["session_id"]) for row in receipt["sources"]}
    if expected_sessions != set(_fold_map(cfg)):
        raise ExperimentError("receipt sessions differ from the frozen folds")
    base_bundles = _load_bundles(root, require_pcm=False)
    if set(base_bundles) != expected_sessions:
        raise ExperimentError("evaluation sessions differ from the frozen receipt")
    expected_prediction_row_count = sum(
        len(_valid_centers(bundle)) for bundle in base_bundles.values()
    )
    for source in receipt["sources"]:
        _require_artifact(Path(source["waveform_path"]), str(source["waveform_sha256"]))
    expected_identities = [
        _model_identity(root, str(model_id))[0] for model_id in cfg["model_ids"]
    ]
    if receipt.get("model_identities") != expected_identities:
        raise ExperimentError("receipt model identities differ from the pinned artifacts")
    expected_arm_pairs = {
        (str(model_id), arm) for model_id in cfg["model_ids"] for arm in ARMS
    }
    receipt_arm_pairs = {
        (str(row["model_id"]), str(row["id"])) for row in receipt.get("arms", [])
    }
    if receipt_arm_pairs != expected_arm_pairs:
        raise ExperimentError("receipt arm matrix differs from the frozen comparison")
    receipt_sha = sha256_file(receipt_path)
    manifest_shas: dict[str, str] = {}
    for model_id in cfg["model_ids"]:
        manifest_path = _feature_manifest_path(root, str(model_id))
        if not manifest_path.is_file():
            raise ExperimentError(f"feature manifest is missing: {model_id}")
        manifest = load_json(manifest_path)
        if manifest.get("model_id") != model_id:
            raise ExperimentError(f"feature manifest model differs: {model_id}")
        if manifest.get("code_sha256") != current_code_sha:
            raise ExperimentError(f"feature manifest code identity differs: {model_id}")
        if manifest.get("config_sha256") != current_config_sha:
            raise ExperimentError(f"feature manifest configuration differs: {model_id}")
        if manifest.get("receipt_sha256") != receipt_sha:
            raise ExperimentError(f"feature manifest receipt identity differs: {model_id}")
        session_rows = {str(row["session_id"]): row for row in manifest.get("sessions", [])}
        if set(session_rows) != expected_sessions or len(session_rows) != len(manifest.get("sessions", [])):
            raise ExperimentError(f"feature manifest sessions differ: {model_id}")
        sources_by_session = {
            str(row["session_id"]): row for row in receipt["sources"]
        }
        base_by_session = _load_bundles(root, require_pcm=False)
        for session_id, row in session_rows.items():
            _require_artifact(
                Path(row["path"]), str(row["sha256"]), int(row["size_bytes"])
            )
            if (
                row.get("source_waveform_sha256")
                != sources_by_session[session_id]["waveform_sha256"]
                or row.get("frontiers_sha256")
                != _array_sha256(base_by_session[session_id].frontiers)
            ):
                raise ExperimentError(f"feature source binding differs: {model_id}/{session_id}")
        expected_runtime = (
            _eres_runtime_identity(root)[0]
            if model_id == "eres2netv2-standard-prepool"
            else None
        )
        if manifest.get("runtime_identity") != expected_runtime:
            raise ExperimentError(f"feature runtime identity differs: {model_id}")
        manifest_shas[str(model_id)] = sha256_file(manifest_path)
    expected_model_receipts = {
        (fold + 1, int(seed))
        for fold in range(len(cfg["folds"]))
        for seed in cfg["seeds"]
    }
    metric_count = 0
    checkpoint_count = 0
    prediction_file_count = 0
    prediction_row_count = 0
    false_example_file_count = 0
    recomputed_metric_count = 0
    feature_bundles = {
        str(model_id): _load_feature_bundles(root, str(model_id))
        for model_id in cfg["model_ids"]
    }
    for model_id, arm in sorted(expected_arm_pairs):
        paths = _arm_paths(root, model_id, arm)
        if not paths["metrics"].is_file():
            raise ExperimentError(f"arm metrics are missing: {model_id}/{arm}")
        metric = load_json(paths["metrics"])
        if metric.get("model_id") != model_id or metric.get("arm") != arm:
            raise ExperimentError(f"arm metric identity differs: {model_id}/{arm}")
        if metric.get("code_sha256") != current_code_sha:
            raise ExperimentError(f"arm metric code identity differs: {model_id}/{arm}")
        if metric.get("config_sha256") != current_config_sha:
            raise ExperimentError(f"arm metric configuration differs: {model_id}/{arm}")
        if metric.get("receipt_sha256") != receipt_sha:
            raise ExperimentError(f"arm metric receipt identity differs: {model_id}/{arm}")
        if metric.get("feature_manifest_sha256") != manifest_shas[model_id]:
            raise ExperimentError(f"arm metric feature identity differs: {model_id}/{arm}")
        if set(metric.get("selected_operating_points", {})) != {
            str(target) for target in cfg["development_false_event_targets_per_hour"]
        }:
            raise ExperimentError(f"arm operating points differ: {model_id}/{arm}")
        diagnostics = metric.get("structured_diagnostics")
        if (arm in DIRECT_ARMS and diagnostics is not None) or (
            arm not in DIRECT_ARMS and diagnostics is None
        ):
            raise ExperimentError(f"arm diagnostics differ: {model_id}/{arm}")
        prediction_path = Path(metric["prediction_path"])
        false_examples_path = Path(metric["false_examples_path"])
        if prediction_path.resolve() != paths["predictions"].resolve():
            raise ExperimentError(f"prediction path differs: {model_id}/{arm}")
        if false_examples_path.resolve() != paths["false_examples"].resolve():
            raise ExperimentError(f"false-example path differs: {model_id}/{arm}")
        _require_artifact(prediction_path, str(metric["prediction_sha256"]))
        _require_artifact(false_examples_path, str(metric["false_examples_sha256"]))
        arm_prediction_rows = _jsonl_row_count(prediction_path)
        if arm_prediction_rows != expected_prediction_row_count:
            raise ExperimentError(f"prediction row count differs: {model_id}/{arm}")
        false_examples = load_json(false_examples_path)
        if not isinstance(false_examples, list):
            raise ExperimentError(f"false examples are malformed: {model_id}/{arm}")
        prediction_rows = read_jsonl(prediction_path)
        if any(
            row.get("model_id") != model_id or row.get("arm") != arm
            for row in prediction_rows
        ):
            raise ExperimentError(f"prediction identity differs: {model_id}/{arm}")
        recomputed_curve, recomputed_points, recomputed_frontier = _curve_and_points(
            base_bundles, prediction_rows, cfg
        )
        if not _same_json_value(metric.get("curve"), recomputed_curve):
            raise ExperimentError(f"recomputed curve differs: {model_id}/{arm}")
        if not _same_json_value(metric.get("selected_operating_points"), recomputed_points):
            raise ExperimentError(f"recomputed operating points differ: {model_id}/{arm}")
        for name in (
            "best_macro_f1_operating_point",
            "best_f1_by_tolerance",
            "unrestricted_operating_point",
        ):
            if not _same_json_value(metric.get(name), recomputed_frontier[name]):
                raise ExperimentError(f"recomputed {name} differs: {model_id}/{arm}")
        recomputed_diagnostics = (
            _structured_diagnostics(prediction_rows, feature_bundles[model_id])
            if arm not in DIRECT_ARMS
            else None
        )
        if not _same_json_value(metric.get("structured_diagnostics"), recomputed_diagnostics):
            raise ExperimentError(f"recomputed diagnostics differ: {model_id}/{arm}")
        recomputed_false_examples = _false_examples(
            prediction_rows,
            feature_bundles[model_id],
            float(recomputed_points["10"]["threshold"]),
            cfg,
        )
        if not _same_json_value(false_examples, recomputed_false_examples):
            raise ExperimentError(f"recomputed false examples differ: {model_id}/{arm}")
        model_receipts = metric.get("model_receipts", [])
        receipt_pairs = {(int(row["fold"]), int(row["seed"])) for row in model_receipts}
        if receipt_pairs != expected_model_receipts or len(model_receipts) != len(receipt_pairs):
            raise ExperimentError(f"checkpoint receipt matrix differs: {model_id}/{arm}")
        for model_receipt in model_receipts:
            _require_artifact(
                Path(model_receipt["checkpoint"]),
                str(model_receipt["checkpoint_sha256"]),
            )
        metric_count += 1
        checkpoint_count += len(model_receipts)
        prediction_file_count += 1
        prediction_row_count += arm_prediction_rows
        false_example_file_count += 1
        recomputed_metric_count += 1
    return {
        "verified_git_commit": current_git["commit"],
        "code_sha256": current_code_sha,
        "config_sha256": current_config_sha,
        "receipt_sha256": receipt_sha,
        "feature_manifest_count": len(manifest_shas),
        "metric_file_count": metric_count,
        "checkpoint_file_count": checkpoint_count,
        "prediction_file_count": prediction_file_count,
        "prediction_row_count": prediction_row_count,
        "false_example_file_count": false_example_file_count,
        "recomputed_metric_count": recomputed_metric_count,
    }


def report(root: Path) -> Path:
    cfg = config()
    artifact_verification = _validate_artifacts(root)
    documents: dict[str, dict[str, dict[str, Any]]] = {}
    frontier_rows: dict[str, dict[str, dict[str, Any]]] = {}
    reference_rows: dict[str, dict[str, dict[str, Any]]] = {}
    deltas: dict[str, dict[str, float]] = {}
    best_by_model: dict[str, str] = {}
    for model_id in cfg["model_ids"]:
        model_documents: dict[str, dict[str, Any]] = {}
        for arm in ARMS:
            path = _arm_paths(root, model_id, arm)["metrics"]
            if not path.is_file():
                raise ExperimentError(f"arm metrics are missing: {model_id}/{arm}")
            model_documents[arm] = load_json(path)
        documents[model_id] = model_documents
        model_frontiers = {
            arm: _frontier_row(document) for arm, document in model_documents.items()
        }
        model_references = {
            arm: {str(target): _point_row(document, target) for target in (1, 5, 10, 20)}
            for arm, document in model_documents.items()
        }
        frontier_rows[model_id] = model_frontiers
        reference_rows[model_id] = model_references
        deltas[model_id] = {
            "A_to_B_adaptation_direct": model_frontiers["B-TRAINABLE-DIRECT"]["macro_f1"]
            - model_frontiers["A-FROZEN-DIRECT"]["macro_f1"],
            "C_to_D_adaptation_structured": model_frontiers["D-TRAINABLE-STATE"]["macro_f1"]
            - model_frontiers["C-FROZEN-STATE"]["macro_f1"],
            "A_to_C_structured_frozen": model_frontiers["C-FROZEN-STATE"]["macro_f1"]
            - model_frontiers["A-FROZEN-DIRECT"]["macro_f1"],
            "B_to_D_structured_adapted": model_frontiers["D-TRAINABLE-STATE"]["macro_f1"]
            - model_frontiers["B-TRAINABLE-DIRECT"]["macro_f1"],
        }
        best_by_model[model_id] = max(
            ARMS, key=lambda arm: model_frontiers[arm]["macro_f1"]
        )
    best_model, best_arm = max(
        ((model_id, arm) for model_id, arm in best_by_model.items()),
        key=lambda value: frontier_rows[value[0]][value[1]]["macro_f1"],
    )
    summary = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "evidence_mode": "development_known_direction_selection",
        "selection_rule": "maximum mean F1 across 100/250/500 ms collars",
        "best_macro_f1_operating_points": frontier_rows,
        "false_event_reference_points": reference_rows,
        "macro_f1_deltas": deltas,
        "best_next_direction_candidate": {"model_id": best_model, "arm": best_arm},
        "best_arm_by_model": best_by_model,
        "artifact_verification": artifact_verification,
        "structured_diagnostics": {
            model_id: {
                arm: documents[model_id][arm]["structured_diagnostics"]
                for arm in ("C-FROZEN-STATE", "D-TRAINABLE-STATE")
            }
            for model_id in cfg["model_ids"]
        },
    }
    summary_path = output_root(root) / "summary.json"
    write_json(summary_path, summary)
    lines = [
        "# PSEM trainable formulation gate results",
        "",
        "Evidence status: **development-known direction-selection evidence only**.",
        "",
        "The three pinned models used the same ten natural continuous meetings, five out-of-fold splits, source-time grid, fixed-lag context, event semantics, duplicate handling, adapter recipe, and evaluation code.",
        "",
        "The primary operating point for each arm maximizes the mean F1 across the 100/250/500 ms collars over the complete score range. FE/h is shown only as context and does not select the threshold.",
        "",
        "## Full-range precision, recall, and F1",
        "",
        "| Model | Arm | Macro F1 | P@100 | R@100 | F1@100 | P@250 | R@250 | F1@250 | P@500 | R@500 | F1@500 | FE/h@250 context |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model_id in cfg["model_ids"]:
        for arm in ARMS:
            row = frontier_rows[model_id][arm]
            m100 = row["tolerances"]["100"]
            m250 = row["tolerances"]["250"]
            m500 = row["tolerances"]["500"]
            lines.append(
                f"| {model_id} | {arm} | {row['macro_f1']:.4f} | {m100['precision']:.4f} | {m100['recall']:.4f} | {m100['f1']:.4f} | {m250['precision']:.4f} | {m250['recall']:.4f} | {m250['f1']:.4f} | {m500['precision']:.4f} | {m500['recall']:.4f} | {m500['f1']:.4f} | {m250['false_events_per_hour']:.2f} |"
            )
    lines.extend(
        [
            "",
            "## FE/h compatibility references",
            "",
            "These rows preserve the issue's requested low-FE reference view. They are annotations of the same score frontier, not a product policy or the basis for choosing the headline threshold.",
            "",
            "| Model | Arm | Target FE/h | R@100 | P@250 | R@250 | F1@250 | R@500 | Actual FE/h@250 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for model_id in cfg["model_ids"]:
        for arm in ARMS:
            for target in (1, 5, 10, 20):
                row = reference_rows[model_id][arm][str(target)]
                lines.append(
                    f"| {model_id} | {arm} | {target} | {row['recall_100']:.4f} | {row['precision_250']:.4f} | {row['recall_250']:.4f} | {row['f1_250']:.4f} | {row['recall_500']:.4f} | {row['false_events_per_hour_250']:.4f} |"
                )
    lines.extend(
        [
            "",
            "## Structured-state diagnostics",
            "",
            "| Model | Arm | State macro F1 | Silence recall | Singleton recall | Overlap recall | Decoder relation balanced accuracy | Decoder different recall | Adjacent different recall | Gap same recall | Gap different recall |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for model_id in cfg["model_ids"]:
        for arm in ("C-FROZEN-STATE", "D-TRAINABLE-STATE"):
            diagnostic = documents[model_id][arm]["structured_diagnostics"]
            state = diagnostic["speech_state"]
            relation = diagnostic["decoder_singleton_relation"]
            adjacent_relation = diagnostic["adjacent_singleton_relation"]
            gap_relation = diagnostic["silence_gap_singleton_relation"]
            lines.append(
                f"| {model_id} | {arm} | {float(state['macro_f1'] or 0.0):.4f} | {float(state['per_class']['silence']['recall'] or 0.0):.4f} | {float(state['per_class']['singleton']['recall'] or 0.0):.4f} | {float(state['per_class']['overlap']['recall'] or 0.0):.4f} | {float(relation['balanced_accuracy'] or 0.0):.4f} | {float(relation['different_speaker_recall'] or 0.0):.4f} | {float(adjacent_relation['different_speaker_recall'] or 0.0):.4f} | {float(gap_relation['same_speaker_recall'] or 0.0):.4f} | {float(gap_relation['different_speaker_recall'] or 0.0):.4f} |"
            )
    lines.extend(["", "## Interpretation", ""])
    for model_id in cfg["model_ids"]:
        model_deltas = deltas[model_id]
        lines.extend(
            [
                f"### {model_id}",
                "",
                f"- Adaptation under direct supervision: A→B Δmacro-F1 = {model_deltas['A_to_B_adaptation_direct']:+.4f}.",
                f"- Adaptation under structured supervision: C→D Δmacro-F1 = {model_deltas['C_to_D_adaptation_structured']:+.4f}.",
                f"- Structured targets with frozen evidence: A→C Δmacro-F1 = {model_deltas['A_to_C_structured_frozen']:+.4f}.",
                f"- Structured targets with adapted evidence: B→D Δmacro-F1 = {model_deltas['B_to_D_structured_adapted']:+.4f}.",
                f"- Best arm for this model: **{best_by_model[model_id]}**.",
                "",
            ]
        )
    lines.extend(
        [
            f"Overall best next-direction candidate: **{best_model} / {best_arm}**.",
            "",
            f"Across the three models, the bounded output adapter did not provide a consistent full-range F1 improvement. The current end-to-end structured-to-event pipeline was worse than direct supervision for every model, but the state and relation diagnostics are non-random and do not establish that the structured representation itself failed. The multiplicative event projection, limited reliable-singleton relation coverage, and component errors are confounded. Carry `{best_model} / {best_arm}` as the current end-to-end candidate, do not carry this adapter recipe as the adaptation answer, and diagnose structured-to-event projection before deciding whether to discard structured representation learning.",
            "",
            "The recommendation is limited to the next PSEM training stage. It is not a release, multilingual-generalization, or production-readiness claim.",
        ]
    )
    path = output_root(root) / "RESULTS.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "extract", "develop", "report"))
    parser.add_argument("--model-id", choices=config()["model_ids"])
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args(argv)
    root = cache_root()
    if args.action == "prepare":
        print(prepare(root))
    elif args.action == "extract":
        if args.model_id is None:
            parser.error("extract requires --model-id")
        print(extract(root, args.model_id))
    elif args.action == "develop":
        if args.model_id is None or args.arm is None:
            parser.error("develop requires --model-id and --arm")
        print(develop(root, args.model_id, args.arm))
    elif args.action == "report":
        print(report(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
