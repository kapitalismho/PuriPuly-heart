from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from experiments.online_speaker_memory_handoff.protocol import (
    _corpus_root,
    _regions_for_source,
    _source_rows,
    input_paths,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    Path(__file__).resolve().parent / "configs" / "r7b" / "fixed_lag_local_segmentation.json"
)
CODE_PATH = Path(__file__).resolve()
SAMPLE_RATE = 16000
HOP_SAMPLES = 1600
R7_RELATIVE = "results/r7/eres_candidate_relation_verifier_v1"
STATE_SILENCE = 0
STATE_SINGLE = 1
STATE_OVERLAP = 2


class R7BError(RuntimeError):
    pass


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def config() -> dict[str, Any]:
    return load_json(CONFIG_PATH)


def cache_root() -> Path:
    value = os.environ.get("SRSCD_CACHE_ROOT")
    if not value:
        raise R7BError("SRSCD_CACHE_ROOT is required")
    root = Path(value).resolve()
    if not root.is_absolute() or root == REPOSITORY_ROOT or REPOSITORY_ROOT in root.parents:
        raise R7BError("SRSCD_CACHE_ROOT must be outside the repository")
    return root


def output_root(root: Path) -> Path:
    return root / str(config()["output_relative_path"])


def r7_root(root: Path) -> Path:
    return root / R7_RELATIVE


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


def _role_paths(root: Path, role: str) -> dict[str, Path]:
    dense = r7_root(root) / "features" / role
    directory = output_root(root) / "features" / role
    return {
        "dense_vectors": dense / "dense_500ms.npy",
        "dense_index": dense / "dense_index.jsonl",
        "dense_manifest": dense / "dense_manifest.json",
        "labels": directory / "partition_labels.npz",
        "pcm": directory / "pcm_features.npy",
        "pcm_manifest": directory / "pcm_manifest.json",
    }


def _fold_map(cfg: dict[str, Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    for fold_index, sessions in enumerate(cfg["folds"]):
        for session_id in sessions:
            if session_id in result:
                raise R7BError(f"session appears in multiple folds: {session_id}")
            result[str(session_id)] = fold_index
    return result


def _label_rows(
    source: dict[str, Any],
    corpus: Path,
    rows: Sequence[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    regions = _regions_for_source(source, corpus)
    speaker_names = sorted({speaker for region in regions for speaker in region.speakers})
    speaker_ids = {speaker: index for index, speaker in enumerate(speaker_names)}
    states = np.full(len(rows), -1, dtype=np.int8)
    speakers = np.full(len(rows), -1, dtype=np.int16)
    region_index = 0
    for row_index, row in enumerate(rows):
        midpoint = int(row["frontier_sample"]) - HOP_SAMPLES // 2
        while region_index + 1 < len(regions) and midpoint >= int(regions[region_index].end_sample):
            region_index += 1
        region = regions[region_index]
        if not (region.start_sample <= midpoint < region.end_sample) or region.ambiguous:
            continue
        if not region.speakers:
            states[row_index] = STATE_SILENCE
        elif len(region.speakers) == 1:
            states[row_index] = STATE_SINGLE
            speakers[row_index] = speaker_ids[next(iter(region.speakers))]
        else:
            states[row_index] = STATE_OVERLAP
    return states, speakers


def prepare(root: Path) -> Path:
    cfg = config()
    r7_inventory_path = r7_root(root) / "inventory.json"
    if not r7_inventory_path.is_file():
        raise R7BError("R7-A inventory is required")
    r7_inventory = load_json(r7_inventory_path)
    fold_map = _fold_map(cfg)
    session_ids = {str(row["session_id"]) for row in r7_inventory["sessions"]}
    if set(fold_map) != session_ids:
        raise R7BError("R7-B folds must contain every R7-A session exactly once")
    paths = input_paths(root)
    sources = _source_rows(paths)
    corpus = _corpus_root(root)
    session_metadata = {str(row["session_id"]): row for row in r7_inventory["sessions"]}
    prepared_sessions: list[dict[str, Any]] = []
    label_hashes: dict[str, str] = {}
    for role in ("development", "evaluation"):
        role_paths = _role_paths(root, role)
        for name in ("dense_vectors", "dense_index", "dense_manifest"):
            if not role_paths[name].is_file():
                raise R7BError(f"R7-A dense artifact is missing: {role_paths[name]}")
        index_rows = read_jsonl(role_paths["dense_index"])
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in index_rows:
            grouped[str(row["session_id"])].append(row)
        states = np.full(len(index_rows), -1, dtype=np.int8)
        speakers = np.full(len(index_rows), -1, dtype=np.int16)
        for session_id, rows in grouped.items():
            row_numbers = [int(row["row"]) for row in rows]
            if row_numbers != list(range(row_numbers[0], row_numbers[-1] + 1)):
                raise R7BError(f"dense session rows are not contiguous: {session_id}")
            source = sources.get(session_id)
            metadata = session_metadata.get(session_id)
            if source is None or metadata is None:
                raise R7BError(f"session metadata is unavailable: {session_id}")
            local_states, local_speakers = _label_rows(source, corpus, rows)
            start_row = row_numbers[0]
            end_row = row_numbers[-1] + 1
            states[start_row:end_row] = local_states
            speakers[start_row:end_row] = local_speakers
            if len(rows) < 16:
                raise R7BError(f"session is too short for R7-B: {session_id}")
            first_boundary = int(rows[5]["frontier_sample"])
            last_boundary = int(rows[-11]["frontier_sample"])
            scored_events = [
                event
                for event in metadata["events"]
                if first_boundary <= int(event["sample"]) < last_boundary + HOP_SAMPLES
            ]
            prepared_sessions.append(
                {
                    **metadata,
                    "dense_role": role,
                    "dense_row_start": start_row,
                    "dense_row_end": end_row,
                    "fold": fold_map[session_id],
                    "first_boundary_sample": first_boundary,
                    "last_boundary_sample": last_boundary,
                    "scored_hours": (last_boundary - first_boundary + HOP_SAMPLES)
                    / SAMPLE_RATE
                    / 3600.0,
                    "events": scored_events,
                    "event_count": len(scored_events),
                }
            )
        role_paths["labels"].parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(role_paths["labels"], state=states, speaker=speakers)
        label_hashes[role] = sha256_file(role_paths["labels"])
    prepared_sessions.sort(key=lambda row: (int(row["fold"]), str(row["session_id"])))
    result = {
        "schema_version": 1,
        "experiment_id": cfg["experiment_id"],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "code_sha256": sha256_file(CODE_PATH),
        "config_sha256": sha256_file(CONFIG_PATH),
        "r7_inventory_sha256": sha256_file(r7_inventory_path),
        "evidence_mode": cfg["evidence_mode"],
        "sessions": prepared_sessions,
        "summary": {
            "scored_hours": sum(float(row["scored_hours"]) for row in prepared_sessions),
            "event_count": sum(int(row["event_count"]) for row in prepared_sessions),
        },
        "label_sha256": label_hashes,
        "git": _git_state(),
    }
    path = output_root(root) / "inventory.json"
    write_json(path, result)
    write_json(output_root(root) / "config_used.json", cfg)
    return path


def _hz_to_mel(value: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(value) / 700.0)


def _mel_to_hz(value: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(value) / 2595.0) - 1.0)


def _mel_filter_bank(fft_size: int, bands: int) -> np.ndarray:
    frequencies = np.linspace(0.0, SAMPLE_RATE / 2.0, fft_size // 2 + 1)
    mel_points = np.linspace(_hz_to_mel(20.0), _hz_to_mel(SAMPLE_RATE / 2.0), bands + 2)
    hz_points = _mel_to_hz(mel_points)
    bank = np.zeros((len(frequencies), bands), dtype=np.float32)
    for index in range(bands):
        left, center, right = hz_points[index : index + 3]
        bank[:, index] = np.maximum(
            0.0,
            np.minimum(
                (frequencies - left) / max(center - left, 1e-9),
                (right - frequencies) / max(right - center, 1e-9),
            ),
        )
    return bank


def _pcm_features(blocks: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    pcm_cfg = cfg["pcm"]
    frame_length = int(pcm_cfg["frame_ms"]) * 16
    frame_hop = int(pcm_cfg["frame_hop_ms"]) * 16
    fft_size = int(pcm_cfg["fft_size"])
    frame_starts = np.arange(0, HOP_SAMPLES - frame_length + 1, frame_hop)
    indices = frame_starts[:, None] + np.arange(frame_length)[None, :]
    frames = blocks[:, indices]
    window = np.hanning(frame_length).astype(np.float32)
    spectra = np.fft.rfft(frames * window, n=fft_size, axis=-1)
    power = np.maximum(np.abs(spectra) ** 2, 1e-12).astype(np.float32)
    bank = _mel_filter_bank(fft_size, int(pcm_cfg["mel_bands"]))
    log_mel = np.log(np.maximum(power @ bank, 1e-10))
    mel_mean = log_mel.mean(axis=1)
    mel_std = log_mel.std(axis=1)
    frame_rms = np.sqrt(np.mean(frames**2, axis=-1) + 1e-12)
    log_rms = np.log(np.sqrt(np.mean(blocks**2, axis=1) + 1e-12))[:, None]
    speech_fraction = (frame_rms > float(pcm_cfg["speech_rms_threshold"])).mean(
        axis=1, keepdims=True
    )
    zcr = np.mean(np.signbit(blocks[:, 1:]) != np.signbit(blocks[:, :-1]), axis=1)[:, None]
    frequencies = np.linspace(0.0, 1.0, power.shape[-1], dtype=np.float32)
    centroid = (
        (power * frequencies[None, None, :]).sum(axis=-1) / np.maximum(power.sum(axis=-1), 1e-12)
    ).mean(axis=1, keepdims=True)
    flatness = (
        np.exp(np.mean(np.log(power), axis=-1)) / np.maximum(np.mean(power, axis=-1), 1e-12)
    ).mean(axis=1, keepdims=True)
    return np.concatenate(
        [mel_mean, mel_std, log_rms, speech_fraction, zcr, centroid, flatness], axis=1
    ).astype(np.float32)


def extract_pcm(root: Path) -> list[Path]:
    import soundfile as sf

    cfg = config()
    inventory_path = output_root(root) / "inventory.json"
    if not inventory_path.is_file():
        prepare(root)
    inventory = load_json(inventory_path)
    results: list[Path] = []
    for role in ("development", "evaluation"):
        paths = _role_paths(root, role)
        index_rows = read_jsonl(paths["dense_index"])
        existing = paths["pcm_manifest"]
        if existing.is_file() and paths["pcm"].is_file():
            manifest = load_json(existing)
            if (
                manifest.get("code_sha256") == sha256_file(CODE_PATH)
                and manifest.get("config_sha256") == sha256_file(CONFIG_PATH)
                and manifest.get("row_count") == len(index_rows)
                and manifest.get("pcm_sha256") == sha256_file(paths["pcm"])
            ):
                results.append(existing)
                continue
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in index_rows:
            grouped[str(row["session_id"])].append(row)
        paths["pcm"].parent.mkdir(parents=True, exist_ok=True)
        feature_dimension = int(cfg["pcm"]["mel_bands"]) * 2 + 5
        output = np.lib.format.open_memmap(
            paths["pcm"], mode="w+", dtype=np.float32, shape=(len(index_rows), feature_dimension)
        )
        started = time.perf_counter()
        for session_id, rows in grouped.items():
            session = next(row for row in inventory["sessions"] if row["session_id"] == session_id)
            waveform, sample_rate = sf.read(
                session["waveform_path"], dtype="float32", always_2d=True
            )
            if sample_rate != SAMPLE_RATE or waveform.shape[1] != 1:
                raise R7BError(f"unexpected waveform geometry: {session_id}")
            mono = waveform[:, 0]
            for start in range(0, len(rows), 512):
                batch_rows = rows[start : start + 512]
                blocks = np.stack(
                    [
                        mono[
                            int(row["frontier_sample"]) - HOP_SAMPLES : int(row["frontier_sample"])
                        ]
                        for row in batch_rows
                    ]
                )
                if blocks.shape[1] != HOP_SAMPLES:
                    raise R7BError(f"incomplete PCM cell: {session_id}")
                row_start = int(batch_rows[0]["row"])
                row_end = row_start + len(batch_rows)
                output[row_start:row_end] = _pcm_features(blocks, cfg)
            del waveform
        output.flush()
        del output
        source_seconds = len(index_rows) * float(cfg["hop_ms"]) / 1000.0
        wall_seconds = time.perf_counter() - started
        manifest = {
            "schema_version": 1,
            "role": role,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "row_count": len(index_rows),
            "feature_dimension": feature_dimension,
            "pcm_sha256": sha256_file(paths["pcm"]),
            "code_sha256": sha256_file(CODE_PATH),
            "config_sha256": sha256_file(CONFIG_PATH),
            "worker_job_id": os.environ.get("ORCA_WORKER_JOB_ID"),
            "wall_seconds": wall_seconds,
            "source_rtf": wall_seconds / source_seconds,
            "hardware": {
                "backend": "cpu",
                "cpu_count": os.cpu_count(),
                "platform": platform.platform(),
                "processor": platform.processor(),
            },
        }
        write_json(existing, manifest)
        results.append(existing)
    return results


@dataclass(slots=True)
class SessionBundle:
    session_id: str
    fold: int
    frontiers: np.ndarray
    eres: np.ndarray
    state: np.ndarray
    speaker: np.ndarray
    pcm: np.ndarray | None
    events: list[dict[str, Any]]
    scored_hours: float
    first_boundary: int
    last_boundary: int


def _load_bundles(root: Path, require_pcm: bool) -> dict[str, SessionBundle]:
    inventory_path = output_root(root) / "inventory.json"
    if not inventory_path.is_file():
        raise R7BError("R7-B inventory is required")
    inventory = load_json(inventory_path)
    metadata = {str(row["session_id"]): row for row in inventory["sessions"]}
    bundles: dict[str, SessionBundle] = {}
    for role in ("development", "evaluation"):
        paths = _role_paths(root, role)
        if require_pcm and not paths["pcm_manifest"].is_file():
            raise R7BError(f"PCM features are missing: {role}")
        dense = np.load(paths["dense_vectors"], mmap_mode="r")
        labels = np.load(paths["labels"])
        pcm = np.load(paths["pcm"], mmap_mode="r") if require_pcm else None
        index_rows = read_jsonl(paths["dense_index"])
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in index_rows:
            grouped[str(row["session_id"])].append(row)
        for session_id, rows in grouped.items():
            info = metadata[session_id]
            start = int(rows[0]["row"])
            end = int(rows[-1]["row"]) + 1
            bundles[session_id] = SessionBundle(
                session_id=session_id,
                fold=int(info["fold"]),
                frontiers=np.asarray([int(row["frontier_sample"]) for row in rows], dtype=np.int64),
                eres=dense[start:end],
                state=labels["state"][start:end],
                speaker=labels["speaker"][start:end],
                pcm=pcm[start:end] if pcm is not None else None,
                events=list(info["events"]),
                scored_hours=float(info["scored_hours"]),
                first_boundary=int(info["first_boundary_sample"]),
                last_boundary=int(info["last_boundary_sample"]),
            )
    return bundles


def _input_values(bundle: SessionBundle, arm: str) -> np.ndarray:
    if arm == "b0":
        return np.asarray(bundle.eres)
    if arm == "b1" and bundle.pcm is not None:
        return np.concatenate([np.asarray(bundle.eres), np.asarray(bundle.pcm)], axis=1)
    raise R7BError(f"invalid or unavailable arm: {arm}")


def _normalization(
    bundles: dict[str, SessionBundle], session_ids: Sequence[str], arm: str
) -> tuple[np.ndarray, np.ndarray]:
    count = 0
    total: np.ndarray | None = None
    total_square: np.ndarray | None = None
    for session_id in session_ids:
        values = _input_values(bundles[session_id], arm).astype(np.float64)
        if total is None:
            total = np.zeros(values.shape[1], dtype=np.float64)
            total_square = np.zeros(values.shape[1], dtype=np.float64)
        total += values.sum(axis=0)
        total_square += np.square(values).sum(axis=0)
        count += len(values)
    if total is None or total_square is None or count == 0:
        raise R7BError("normalization set is empty")
    mean = total / count
    variance = np.maximum(total_square / count - np.square(mean), 1e-8)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def _valid_centers(bundle: SessionBundle) -> range:
    return range(5, len(bundle.frontiers) - 10)


def _training_centers(bundle: SessionBundle, cfg: dict[str, Any]) -> list[int]:
    background_step = int(cfg["training_background_hop_ms"]) // int(cfg["hop_ms"])
    radius = int(cfg["training_event_radius_ms"]) * 16
    event_samples = np.asarray([int(row["sample"]) for row in bundle.events], dtype=np.int64)
    selected: list[int] = []
    for center in _valid_centers(bundle):
        boundary = int(bundle.frontiers[center])
        background = (center - 5) % background_step == 0
        near_event = bool(len(event_samples) and np.min(np.abs(event_samples - boundary)) <= radius)
        if background or near_event:
            selected.append(center)
    return selected


def _batch_arrays(
    bundles: dict[str, SessionBundle],
    references: Sequence[tuple[str, int]],
    arm: str,
    mean: np.ndarray,
    scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_rows: list[np.ndarray] = []
    state_rows: list[np.ndarray] = []
    speaker_rows: list[np.ndarray] = []
    for session_id, center in references:
        bundle = bundles[session_id]
        start = center - 5
        end = center + 11
        x_rows.append((_input_values(bundle, arm)[start:end] - mean) / scale)
        state_rows.append(np.asarray(bundle.state[start:end]))
        speaker_rows.append(np.asarray(bundle.speaker[start:end]))
    return (
        np.asarray(x_rows, dtype=np.float32),
        np.asarray(state_rows, dtype=np.int64),
        np.asarray(speaker_rows, dtype=np.int64),
    )


def _model_class():
    import torch

    class LocalPartitionModel(torch.nn.Module):
        def __init__(self, input_dimension: int, cfg: dict[str, Any]) -> None:
            super().__init__()
            hidden = int(cfg["hidden_dimension"])
            pair = int(cfg["pair_dimension"])
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
            self.pair_projection = torch.nn.Sequential(
                torch.nn.Linear(hidden, pair), torch.nn.GELU()
            )
            self.pair_head = torch.nn.Linear(pair * 2, 1)
            self.state_head = torch.nn.Linear(hidden, 3)

        def forward(self, values):
            hidden, _ = self.temporal(self.input_projection(values))
            pair_values = self.pair_projection(hidden)
            left = pair_values[:, :, None, :]
            right = pair_values[:, None, :, :]
            relation = torch.cat([torch.abs(left - right), left * right], dim=-1)
            pair_logits = self.pair_head(relation).squeeze(-1)
            return pair_logits, self.state_head(hidden)

    return LocalPartitionModel


def _loss_values(pair_logits, state_logits, states, speakers, state_weight: float):
    import torch

    sequence_length = states.shape[1]
    singleton = states == STATE_SINGLE
    pair_mask = singleton[:, :, None] & singleton[:, None, :]
    upper = torch.triu(
        torch.ones((sequence_length, sequence_length), dtype=torch.bool, device=states.device),
        diagonal=1,
    )
    pair_mask &= upper[None, :, :]
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
    return pair_loss + state_weight * state_loss, pair_loss, state_loss


def _iterate_batches(
    references: Sequence[tuple[str, int]], batch_size: int, rng: np.random.Generator
) -> Iterable[list[tuple[str, int]]]:
    order = rng.permutation(len(references))
    for start in range(0, len(order), batch_size):
        yield [references[int(index)] for index in order[start : start + batch_size]]


def _validation_loss(
    model,
    bundles: dict[str, SessionBundle],
    references: Sequence[tuple[str, int]],
    arm: str,
    mean: np.ndarray,
    scale: np.ndarray,
    cfg: dict[str, Any],
) -> float:
    import torch

    model.eval()
    losses: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(references), int(cfg["batch_size"])):
            batch = references[start : start + int(cfg["batch_size"])]
            x, states, speakers = _batch_arrays(bundles, batch, arm, mean, scale)
            pair_logits, state_logits = model(torch.from_numpy(x))
            loss, _, _ = _loss_values(
                pair_logits,
                state_logits,
                torch.from_numpy(states),
                torch.from_numpy(speakers),
                float(cfg["state_loss_weight"]),
            )
            losses.append(float(loss))
    return float(np.mean(losses)) if losses else math.inf


def _fit_model(
    bundles: dict[str, SessionBundle],
    train_sessions: Sequence[str],
    validation_sessions: Sequence[str],
    arm: str,
    seed: int,
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, int, float]:
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    mean, scale = _normalization(bundles, train_sessions, arm)
    train_references = [
        (session_id, center)
        for session_id in train_sessions
        for center in _training_centers(bundles[session_id], cfg)
    ]
    validation_references = [
        (session_id, center)
        for session_id in validation_sessions
        for center in _training_centers(bundles[session_id], cfg)
    ]
    if not train_references or not validation_references:
        raise R7BError("training or validation examples are empty")
    input_dimension = _input_values(bundles[train_sessions[0]], arm).shape[1]
    model = _model_class()(input_dimension, cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
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
            x, states, speakers = _batch_arrays(bundles, batch, arm, mean, scale)
            pair_logits, state_logits = model(torch.from_numpy(x))
            loss, _, _ = _loss_values(
                pair_logits,
                state_logits,
                torch.from_numpy(states),
                torch.from_numpy(speakers),
                float(cfg["state_loss_weight"]),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        validation_loss = _validation_loss(
            model,
            bundles,
            validation_references,
            arm,
            mean,
            scale,
            cfg,
        )
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
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
        raise R7BError("training did not produce a finite model")
    return best_state, mean, scale, best_epoch, best_loss


def _decode_scores(pair_logits, state_logits):
    import torch

    same = torch.sigmoid(pair_logits)
    state = torch.softmax(state_logits, dim=-1)
    left = 5
    right = 6
    candidates = [
        state[:, left, STATE_SINGLE] * state[:, right, STATE_SINGLE] * (1.0 - same[:, left, right])
    ]
    for source in range(left - 1, -1, -1):
        silence = torch.prod(state[:, source + 1 : right, STATE_SILENCE], dim=1)
        candidates.append(
            state[:, source, STATE_SINGLE]
            * silence
            * state[:, right, STATE_SINGLE]
            * (1.0 - same[:, source, right])
        )
    candidates.append(
        (state[:, left, STATE_SILENCE] + state[:, left, STATE_SINGLE])
        * state[:, right, STATE_OVERLAP]
    )
    return torch.stack(candidates, dim=1).max(dim=1).values


def _score_session(
    bundle: SessionBundle,
    arm: str,
    states: Sequence[dict[str, Any]],
    means: Sequence[np.ndarray],
    scales: Sequence[np.ndarray],
    cfg: dict[str, Any],
) -> np.ndarray:
    import torch

    references = [(bundle.session_id, center) for center in _valid_centers(bundle)]
    all_scores: list[np.ndarray] = []
    input_dimension = _input_values(bundle, arm).shape[1]
    for state_dict, mean, scale in zip(states, means, scales, strict=True):
        model = _model_class()(input_dimension, cfg)
        model.load_state_dict(state_dict)
        model.eval()
        score_batches: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(references), int(cfg["batch_size"])):
                batch = references[start : start + int(cfg["batch_size"])]
                x, _, _ = _batch_arrays({bundle.session_id: bundle}, batch, arm, mean, scale)
                pair_logits, state_logits = model(torch.from_numpy(x))
                score_batches.append(_decode_scores(pair_logits, state_logits).numpy())
        all_scores.append(np.concatenate(score_batches))
    return np.mean(all_scores, axis=0).astype(np.float64)


def _prediction_rows(
    bundles: dict[str, SessionBundle], scores: dict[str, np.ndarray], arm: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for session_id, bundle in bundles.items():
        centers = list(_valid_centers(bundle))
        values = scores[session_id]
        if len(centers) != len(values):
            raise R7BError(f"score geometry differs: {session_id}")
        rows.extend(
            {
                "arm": arm,
                "session_id": session_id,
                "fold": bundle.fold,
                "boundary_sample": int(bundle.frontiers[center]),
                "score": float(score),
            }
            for center, score in zip(centers, values, strict=True)
        )
    return sorted(rows, key=lambda row: (str(row["session_id"]), int(row["boundary_sample"])))


def _cosine_baseline_scores(bundle: SessionBundle) -> np.ndarray:
    values = np.asarray(bundle.eres, dtype=np.float64)
    norm = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    return np.asarray(
        [1.0 - float(np.dot(norm[center], norm[center + 1])) for center in _valid_centers(bundle)],
        dtype=np.float64,
    )


def _peaks(rows: Sequence[dict[str, Any]], radius_samples: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["session_id"])].append(row)
    result: list[dict[str, Any]] = []
    for session_rows in grouped.values():
        session_rows.sort(key=lambda row: int(row["boundary_sample"]))
        maxima: list[dict[str, Any]] = []
        for index, row in enumerate(session_rows):
            previous = (
                float(session_rows[index - 1]["score"])
                if index > 0
                and int(row["boundary_sample"]) - int(session_rows[index - 1]["boundary_sample"])
                <= HOP_SAMPLES
                else -math.inf
            )
            following = (
                float(session_rows[index + 1]["score"])
                if index + 1 < len(session_rows)
                and int(session_rows[index + 1]["boundary_sample"]) - int(row["boundary_sample"])
                <= HOP_SAMPLES
                else -math.inf
            )
            if float(row["score"]) >= previous and float(row["score"]) >= following:
                maxima.append(row)
        accepted: list[dict[str, Any]] = []
        for row in sorted(
            maxima, key=lambda value: (-float(value["score"]), int(value["boundary_sample"]))
        ):
            if all(
                abs(int(row["boundary_sample"]) - int(other["boundary_sample"])) > radius_samples
                for other in accepted
            ):
                accepted.append(row)
        result.extend(accepted)
    return sorted(result, key=lambda row: -float(row["score"]))


def _one_to_one(
    predictions: Sequence[int], references: Sequence[int], tolerance: int
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    prediction_values = sorted(int(value) for value in predictions)
    reference_values = sorted(int(value) for value in references)
    matched: list[tuple[int, int]] = []
    false: list[int] = []
    misses: list[int] = []
    prediction_index = 0
    reference_index = 0
    while prediction_index < len(prediction_values) and reference_index < len(reference_values):
        prediction = prediction_values[prediction_index]
        reference = reference_values[reference_index]
        if prediction < reference - tolerance:
            false.append(prediction)
            prediction_index += 1
        elif reference < prediction - tolerance:
            misses.append(reference)
            reference_index += 1
        else:
            matched.append((prediction, reference))
            prediction_index += 1
            reference_index += 1
    false.extend(prediction_values[prediction_index:])
    misses.extend(reference_values[reference_index:])
    return matched, false, misses


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _metrics(
    bundles: dict[str, SessionBundle], selected_peaks: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    predictions_by_session: dict[str, list[int]] = defaultdict(list)
    for row in selected_peaks:
        predictions_by_session[str(row["session_id"])].append(int(row["boundary_sample"]))
    exposure_hours = sum(bundle.scored_hours for bundle in bundles.values())
    result: dict[str, Any] = {
        "exposure_hours": exposure_hours,
        "prediction_count": len(selected_peaks),
        "reference_count": sum(len(bundle.events) for bundle in bundles.values()),
        "availability_latency_ms": {"median": 1000.0, "p90": 1000.0, "p95": 1000.0},
        "tolerances": {},
    }
    primary_matches: list[tuple[str, int, int]] = []
    primary_false: list[tuple[str, int]] = []
    primary_misses: list[tuple[str, int]] = []
    primary_per_meeting: dict[str, Any] = {}
    for tolerance_ms in (100, 250, 500):
        matched_all: list[tuple[str, int, int]] = []
        false_all: list[tuple[str, int]] = []
        miss_all: list[tuple[str, int]] = []
        per_meeting: dict[str, Any] = {}
        for session_id, bundle in bundles.items():
            predictions = predictions_by_session.get(session_id, [])
            references = [int(event["sample"]) for event in bundle.events]
            matched, false, misses = _one_to_one(predictions, references, tolerance_ms * 16)
            matched_all.extend(
                (session_id, prediction, reference) for prediction, reference in matched
            )
            false_all.extend((session_id, prediction) for prediction in false)
            miss_all.extend((session_id, reference) for reference in misses)
            per_meeting[session_id] = {
                "reference_count": len(references),
                "prediction_count": len(predictions),
                "true_positive_count": len(matched),
                "false_event_count": len(false),
                "miss_count": len(misses),
                "recall": _safe_ratio(len(matched), len(references)),
                "false_events_per_hour": len(false) / bundle.scored_hours,
            }
        true_positive_count = len(matched_all)
        precision = _safe_ratio(true_positive_count, len(selected_peaks))
        recall = _safe_ratio(true_positive_count, result["reference_count"])
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall > 0.0
            else None
        )
        result["tolerances"][str(tolerance_ms)] = {
            "true_positive_count": true_positive_count,
            "false_event_count": len(false_all),
            "miss_count": len(miss_all),
            "false_events_per_hour": len(false_all) / exposure_hours,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        if tolerance_ms == 250:
            primary_matches = matched_all
            primary_false = false_all
            primary_misses = miss_all
            primary_per_meeting = per_meeting
    event_by_session = {
        session_id: {int(event["sample"]): event for event in bundle.events}
        for session_id, bundle in bundles.items()
    }
    miss_strata: dict[str, int] = defaultdict(int)
    matched_strata: dict[str, int] = defaultdict(int)
    reference_strata: dict[str, int] = defaultdict(int)
    short_reference_count = 0
    short_matched_count = 0
    for bundle in bundles.values():
        for event in bundle.events:
            reference_strata[str(event["stratum"])] += 1
        short_reference_count += sum(
            bool(event.get("short_backchannel_or_return")) for event in bundle.events
        )
    for session_id, _, reference in primary_matches:
        event = event_by_session[session_id][reference]
        matched_strata[str(event["stratum"])] += 1
        if event.get("short_backchannel_or_return"):
            short_matched_count += 1
    for session_id, reference in primary_misses:
        event = event_by_session[session_id][reference]
        miss_strata[str(event["stratum"])] += 1
    false_strata: dict[str, int] = defaultdict(int)
    for session_id, prediction in primary_false:
        events = bundles[session_id].events
        nearest = (
            min(events, key=lambda event: abs(int(event["sample"]) - prediction))
            if events
            else None
        )
        if nearest is not None and abs(int(nearest["sample"]) - prediction) <= 8000:
            false_strata["ambiguous_distance"] += 1
        else:
            false_strata["same_speaker_false_candidate"] += 1
    meeting_true_positives = [
        int(row["true_positive_count"]) for row in primary_per_meeting.values()
    ]
    result.update(
        {
            "per_meeting": primary_per_meeting,
            "matched_pairs": primary_matches,
            "false_event_samples": primary_false,
            "miss_samples": primary_misses,
            "error_strata": {
                "false_events": dict(sorted(false_strata.items())),
                "misses": dict(sorted(miss_strata.items())),
            },
            "stratum_recall": {
                name: _safe_ratio(matched_strata[name], count)
                for name, count in sorted(reference_strata.items())
            },
            "maximum_meeting_true_positive_share": (
                max(meeting_true_positives) / max(sum(meeting_true_positives), 1)
            ),
            "short_return_recall": _safe_ratio(short_matched_count, short_reference_count),
        }
    )
    return result


def _curve_and_points(
    bundles: dict[str, SessionBundle], rows: Sequence[dict[str, Any]], cfg: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    peaks = _peaks(rows, int(cfg["duplicate_suppression_ms"]) * 16)
    unique_scores = sorted({float(row["score"]) for row in peaks}, reverse=True)
    thresholds = [math.nextafter(unique_scores[0], math.inf)] if unique_scores else [math.inf]
    thresholds.extend(unique_scores[:4096])
    curve: list[dict[str, Any]] = []
    for threshold in thresholds:
        selected = [row for row in peaks if float(row["score"]) >= threshold]
        metrics = _metrics(bundles, selected)
        primary = metrics["tolerances"]["250"]
        curve.append(
            {
                "threshold": threshold,
                "prediction_count": len(selected),
                "true_positive_count": primary["true_positive_count"],
                "false_event_count": primary["false_event_count"],
                "false_events_per_hour": primary["false_events_per_hour"],
                "recall_250": primary["recall"],
            }
        )
        if float(primary["false_events_per_hour"]) > 22.0:
            break
    points: dict[str, Any] = {}
    for target in cfg["development_false_event_targets_per_hour"]:
        eligible = [row for row in curve if float(row["false_events_per_hour"]) <= target]
        selected_row = max(
            eligible,
            key=lambda row: (
                float(row["recall_250"] or 0.0),
                -float(row["false_events_per_hour"]),
            ),
        )
        selected_peaks = [
            row for row in peaks if float(row["score"]) >= float(selected_row["threshold"])
        ]
        points[str(target)] = {
            "threshold": selected_row["threshold"],
            "metrics": _metrics(bundles, selected_peaks),
        }
    return curve, points, peaks


def _fold_recall(point: dict[str, Any], fold_sessions: Sequence[str]) -> float:
    per_meeting = point["metrics"]["per_meeting"]
    true_positive = sum(
        int(per_meeting[session]["true_positive_count"]) for session in fold_sessions
    )
    references = sum(int(per_meeting[session]["reference_count"]) for session in fold_sessions)
    return true_positive / references if references else 0.0


def _fold_false_rate(
    point: dict[str, Any], fold_sessions: Sequence[str], bundles: dict[str, SessionBundle]
) -> float:
    per_meeting = point["metrics"]["per_meeting"]
    false_count = sum(int(per_meeting[session]["false_event_count"]) for session in fold_sessions)
    exposure = sum(bundles[session].scored_hours for session in fold_sessions)
    return false_count / exposure


def _arm_paths(root: Path, arm: str) -> dict[str, Path]:
    directory = output_root(root)
    return {
        "predictions": directory / f"{arm}_oof_predictions.jsonl",
        "metrics": directory / f"{arm}_development_metrics.json",
        "models": directory / "models" / arm,
    }


def _load_predictions(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise R7BError(f"predictions are missing: {path}")
    return read_jsonl(path)


def _gate_result(
    arm: str,
    points: dict[str, Any],
    comparator_points: dict[str, Any],
    bundles: dict[str, SessionBundle],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    point_10 = points["10"]
    point_20 = points["20"]
    primary_10 = point_10["metrics"]["tolerances"]["250"]
    primary_20 = point_20["metrics"]["tolerances"]["250"]
    fold_rows: list[dict[str, Any]] = []
    improved_count = 0
    fold_rate_ok = True
    for fold_index, fold_sessions in enumerate(cfg["folds"]):
        recall = _fold_recall(point_10, fold_sessions)
        comparator_recall = _fold_recall(comparator_points["10"], fold_sessions)
        improved = recall > comparator_recall
        improved_count += int(improved)
        false_rate_10 = _fold_false_rate(point_10, fold_sessions, bundles)
        false_rate_20 = _fold_false_rate(point_20, fold_sessions, bundles)
        fold_ok = false_rate_10 <= 20.0 and false_rate_20 <= 40.0
        fold_rate_ok &= fold_ok
        fold_rows.append(
            {
                "fold": fold_index + 1,
                "sessions": list(fold_sessions),
                "recall_250_at_10": recall,
                "comparator_recall_250_at_10": comparator_recall,
                "improved": improved,
                "false_events_per_hour_at_10": false_rate_10,
                "false_events_per_hour_at_20": false_rate_20,
                "fold_rate_ok": fold_ok,
            }
        )
    strata = point_20["metrics"]["stratum_recall"]
    checks = {
        "recall_at_10": float(primary_10["recall"] or 0.0)
        >= float(cfg["gate"]["recall_at_10_false_events_per_hour"]),
        "recall_at_20": float(primary_20["recall"] or 0.0)
        >= float(cfg["gate"]["recall_at_20_false_events_per_hour"]),
        "aggregate_rate_at_10": float(primary_10["false_events_per_hour"]) <= 10.0,
        "aggregate_rate_at_20": float(primary_20["false_events_per_hour"]) <= 20.0,
        "minimum_improved_folds": improved_count >= int(cfg["gate"]["minimum_improved_folds"]),
        "fold_rate_stability": fold_rate_ok,
        "overlap_recall_nonzero": float(strata.get("overlap_onset") or 0.0) > 0.0,
        "silence_gap_recall_nonzero": float(strata.get("silence_gap_change") or 0.0) > 0.0,
        "meeting_concentration": float(point_20["metrics"]["maximum_meeting_true_positive_share"])
        <= float(cfg["gate"]["maximum_single_meeting_true_positive_share"]),
        "short_return_preserved": float(point_20["metrics"]["short_return_recall"] or 0.0) > 0.0,
    }
    return {
        "arm": arm,
        "passed": all(checks.values()),
        "checks": checks,
        "improved_fold_count": improved_count,
        "folds": fold_rows,
        "comparison": "cosine_baseline" if arm == "b0" else "b0",
    }


def develop(root: Path, arm: str) -> Path:
    import torch

    if arm not in {"b0", "b1"}:
        raise R7BError(f"unknown arm: {arm}")
    cfg = config()
    require_pcm = arm == "b1"
    bundles = _load_bundles(root, require_pcm=require_pcm)
    all_sessions = sorted(bundles)
    started = time.perf_counter()
    arm_paths = _arm_paths(root, arm)
    arm_paths["models"].mkdir(parents=True, exist_ok=True)
    model_receipts: list[dict[str, Any]] = []
    oof_scores: dict[str, np.ndarray] = {}
    parameter_count: int | None = None
    for fold_index, held_out_sessions in enumerate(cfg["folds"]):
        validation_sessions = cfg["folds"][(fold_index + 1) % len(cfg["folds"])]
        excluded = set(held_out_sessions) | set(validation_sessions)
        train_sessions = [session for session in all_sessions if session not in excluded]
        seed_states: list[dict[str, Any]] = []
        seed_means: list[np.ndarray] = []
        seed_scales: list[np.ndarray] = []
        for seed in cfg["seeds"]:
            state, mean, scale, best_epoch, best_loss = _fit_model(
                bundles,
                train_sessions,
                validation_sessions,
                arm,
                int(seed),
                cfg,
            )
            seed_states.append(state)
            seed_means.append(mean)
            seed_scales.append(scale)
            input_dimension = _input_values(bundles[train_sessions[0]], arm).shape[1]
            model = _model_class()(input_dimension, cfg)
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            checkpoint = arm_paths["models"] / f"fold_{fold_index + 1}_seed_{seed}.pt"
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
                }
            )
        for session_id in held_out_sessions:
            oof_scores[session_id] = _score_session(
                bundles[session_id],
                arm,
                seed_states,
                seed_means,
                seed_scales,
                cfg,
            )
        print(
            json.dumps(
                {
                    "stage": "r7b_development",
                    "arm": arm,
                    "fold": fold_index + 1,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    rows = _prediction_rows(bundles, oof_scores, arm)
    write_jsonl(arm_paths["predictions"], rows)
    curve, points, _ = _curve_and_points(bundles, rows, cfg)
    baseline_path = output_root(root) / "cosine_baseline_metrics.json"
    if arm == "b0":
        baseline_scores = {
            session_id: _cosine_baseline_scores(bundle) for session_id, bundle in bundles.items()
        }
        baseline_rows = _prediction_rows(bundles, baseline_scores, "cosine_baseline")
        baseline_curve, baseline_points, _ = _curve_and_points(bundles, baseline_rows, cfg)
        write_json(
            baseline_path,
            {
                "schema_version": 1,
                "curve": baseline_curve,
                "selected_operating_points": baseline_points,
            },
        )
        comparator_points = baseline_points
    else:
        b0_metrics_path = _arm_paths(root, "b0")["metrics"]
        if not b0_metrics_path.is_file():
            raise R7BError("B0 must finish before B1")
        comparator_points = load_json(b0_metrics_path)["selected_operating_points"]
    gate = _gate_result(arm, points, comparator_points, bundles, cfg)
    result = {
        "schema_version": 1,
        "arm": arm,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "evidence_mode": cfg["evidence_mode"],
        "code_sha256": sha256_file(CODE_PATH),
        "config_sha256": sha256_file(CONFIG_PATH),
        "inventory_sha256": sha256_file(output_root(root) / "inventory.json"),
        "worker_job_id": os.environ.get("ORCA_WORKER_JOB_ID"),
        "wall_seconds": time.perf_counter() - started,
        "parameter_count": parameter_count,
        "model_receipts": model_receipts,
        "prediction_sha256": sha256_file(arm_paths["predictions"]),
        "curve": curve,
        "selected_operating_points": points,
        "gate": gate,
    }
    write_json(arm_paths["metrics"], result)
    return arm_paths["metrics"]


def _plot_outputs(root: Path, b0: dict[str, Any], b1: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    baseline = load_json(output_root(root) / "cosine_baseline_metrics.json")
    figure, axis = plt.subplots(figsize=(8, 5))
    for name, document in (("cosine baseline", baseline), ("B0", b0), ("B1", b1)):
        curve = document["curve"]
        axis.plot(
            [float(row["false_events_per_hour"]) for row in curve],
            [float(row["recall_250"] or 0.0) for row in curve],
            label=name,
        )
    axis.set_xlim(0.0, 22.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("out-of-fold false events/hour")
    axis.set_ylabel("Recall@250ms")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_root(root) / "development_recall_false_event_curve.png", dpi=160)
    plt.close(figure)
    rows = _load_predictions(_arm_paths(root, "b1")["predictions"])
    bundles = _load_bundles(root, require_pcm=True)
    peaks = _peaks(rows, int(config()["duplicate_suppression_ms"]) * 16)
    threshold = float(b1["selected_operating_points"]["10"]["threshold"])
    accepted = [row for row in peaks if float(row["score"]) >= threshold][:5]
    rejected = [row for row in peaks if float(row["score"]) < threshold][:5]
    selected = accepted + rejected
    if len(selected) < 10:
        selected.extend(rows[: 10 - len(selected)])
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["session_id"])].append(row)
    figure, axes = plt.subplots(5, 2, figsize=(12, 12))
    for axis, row in zip(axes.reshape(-1), selected[:10], strict=True):
        center = int(row["boundary_sample"])
        nearby = [
            value
            for value in grouped[str(row["session_id"])]
            if abs(int(value["boundary_sample"]) - center) <= 32000
        ]
        axis.plot(
            [(int(value["boundary_sample"]) - center) / SAMPLE_RATE for value in nearby],
            [float(value["score"]) for value in nearby],
            marker=".",
        )
        axis.axvline(0.0, color="tab:orange")
        events = bundles[str(row["session_id"])].events
        if events:
            nearest = min(events, key=lambda event: abs(int(event["sample"]) - center))
            axis.axvline(
                (int(nearest["sample"]) - center) / SAMPLE_RATE,
                color="black",
                linestyle="--",
            )
        axis.set_xlim(-2.0, 2.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_title(
            f"{row['session_id']} @ {center / SAMPLE_RATE:.1f}s\nscore={float(row['score']):.3f}",
            fontsize=8,
        )
    figure.tight_layout()
    figure.savefig(output_root(root) / "representative_partition_timelines.png", dpi=160)
    plt.close(figure)


def report(root: Path) -> Path:
    b0 = load_json(_arm_paths(root, "b0")["metrics"])
    b1 = load_json(_arm_paths(root, "b1")["metrics"])
    inventory = load_json(output_root(root) / "inventory.json")
    _plot_outputs(root, b0, b1)
    lines = [
        "# R7-B Fixed-Lag Local Speaker Segmentation Internal Report",
        "",
        "Evidence status: **development-known internal decision only**.",
        "",
        f"B0 gate: **{'PASS' if b0['gate']['passed'] else 'FAIL'}**.",
        f"B1 gate: **{'PASS' if b1['gate']['passed'] else 'FAIL'}**.",
        "",
        f"Exposure: {float(inventory['summary']['scored_hours']):.3f} hours; {int(inventory['summary']['event_count'])} reference changes.",
        "",
        "| Arm | Target FE/h | Recall@250 | Actual FE/h | Improved folds | Gate |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for arm_name, document in (("B0 frozen ERes", b0), ("B1 ERes + PCM", b1)):
        for target in (10, 20):
            point = document["selected_operating_points"][str(target)]["metrics"]
            primary = point["tolerances"]["250"]
            lines.append(
                f"| {arm_name} | {target} | {float(primary['recall'] or 0.0):.3f} | {float(primary['false_events_per_hour']):.3f} | {int(document['gate']['improved_fold_count'])}/5 | {'PASS' if document['gate']['passed'] else 'FAIL'} |"
            )
    lines.extend(
        [
            "",
            "## Gate checks",
            "",
            *[
                f"- B0 `{name}`: {'pass' if value else 'fail'}"
                for name, value in b0["gate"]["checks"].items()
            ],
            *[
                f"- B1 `{name}`: {'pass' if value else 'fail'}"
                for name, value in b1["gate"]["checks"].items()
            ],
            "",
            "## Decision",
            "",
        ]
    )
    if b1["gate"]["passed"]:
        lines.extend(
            [
                "B1 passed the internal development gate. Stop and request authorization to freeze a new untouched natural evaluation panel before any confirmatory run.",
                "",
                "No confirmatory evaluation was run.",
            ]
        )
    else:
        lines.extend(
            [
                "B1 failed the mandatory internal development gate. R7-B stops. No evaluation is authorized or run.",
                "",
                "A larger decoder, longer latency, or additional representation sweep is not authorized by this addendum.",
            ]
        )
    path = output_root(root) / "REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def smoke() -> dict[str, Any]:
    import torch

    cfg = config()
    model = _model_class()(192, cfg)
    values = torch.randn(3, 16, 192)
    pair_logits, state_logits = model(values)
    scores = _decode_scores(pair_logits, state_logits)
    if pair_logits.shape != (3, 16, 16) or state_logits.shape != (3, 16, 3):
        raise R7BError("model output geometry is invalid")
    if not torch.isfinite(scores).all() or torch.any(scores < 0.0) or torch.any(scores > 1.0):
        raise R7BError("decoded scores are invalid")
    rows = [
        {"session_id": "x", "boundary_sample": index * HOP_SAMPLES, "score": score}
        for index, score in enumerate((0.1, 0.9, 0.2, 0.1, 0.85, 0.1))
    ]
    peaks = _peaks(rows, 200 * 16)
    if [row["boundary_sample"] for row in peaks] != [HOP_SAMPLES, 4 * HOP_SAMPLES]:
        raise R7BError("short return suppression smoke failed")
    matched, false, misses = _one_to_one([100, 900, 1300], [120, 910], 50)
    if len(matched) != 2 or false != [1300] or misses:
        raise R7BError("one-to-one matching smoke failed")
    return {
        "pair_shape": list(pair_logits.shape),
        "state_shape": list(state_logits.shape),
        "score_shape": list(scores.shape),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "short_return_peak_count": len(peaks),
        "code_sha256": sha256_file(CODE_PATH),
        "config_sha256": sha256_file(CONFIG_PATH),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "extract-pcm", "develop", "report", "smoke"))
    parser.add_argument("--arm", choices=("b0", "b1"))
    args = parser.parse_args(argv)
    if args.action == "smoke":
        print(json.dumps(smoke(), indent=2, sort_keys=True))
        return 0
    root = cache_root()
    if args.action == "prepare":
        print(prepare(root))
    elif args.action == "extract-pcm":
        for path in extract_pcm(root):
            print(path)
    elif args.action == "develop":
        if args.arm is None:
            parser.error("develop requires --arm")
        print(develop(root, args.arm))
    elif args.action == "report":
        print(report(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
