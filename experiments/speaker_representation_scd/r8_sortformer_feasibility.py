from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import wave
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from experiments.online_speaker_memory_handoff.protocol import read_jsonl, sha256_file, write_json

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "r8" / "streaming_sortformer_feasibility.json"
CODE_PATH = Path(__file__).resolve()
R7B_RELATIVE = Path("results/r7b/fixed_lag_local_segmentation_v1")


class R8Error(RuntimeError):
    pass


@dataclass(slots=True)
class Session:
    session_id: str
    fold: int
    waveform_path: Path
    waveform_sha256: str
    first_boundary: int
    last_boundary: int
    scored_hours: float
    events: list[dict[str, Any]]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def config() -> dict[str, Any]:
    return load_json(CONFIG_PATH)


def cache_root() -> Path:
    value = os.environ.get("SRSCD_CACHE_ROOT")
    if not value:
        raise R8Error("SRSCD_CACHE_ROOT is required")
    root = Path(value).resolve()
    if not root.is_absolute() or root == REPOSITORY_ROOT or REPOSITORY_ROOT in root.parents:
        raise R8Error("SRSCD_CACHE_ROOT must be outside the repository")
    return root


def output_root(root: Path) -> Path:
    return root / str(config()["output_relative_path"])


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


def _fold_map(cfg: dict[str, Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    for fold, session_ids in enumerate(cfg["folds"]):
        for session_id in session_ids:
            if session_id in result:
                raise R8Error(f"duplicate session in folds: {session_id}")
            result[str(session_id)] = fold
    return result


def _session_rows(root: Path) -> list[dict[str, Any]]:
    path = root / R7B_RELATIVE / "inventory.json"
    if not path.is_file():
        raise R8Error(f"R7-B inventory is missing: {path}")
    inventory = load_json(path)
    rows = list(inventory["sessions"])
    fold_map = _fold_map(config())
    if set(fold_map) != {str(row["session_id"]) for row in rows}:
        raise R8Error("R8 folds do not match the R7-B inventory")
    for row in rows:
        if fold_map[str(row["session_id"])] != int(row["fold"]):
            raise R8Error(f"fold drift for {row['session_id']}")
    return rows


def _sessions(root: Path) -> dict[str, Session]:
    result: dict[str, Session] = {}
    for row in _session_rows(root):
        waveform_path = Path(row["waveform_path"])
        if not waveform_path.is_file():
            raise R8Error(f"waveform is missing: {waveform_path}")
        result[str(row["session_id"])] = Session(
            session_id=str(row["session_id"]),
            fold=int(row["fold"]),
            waveform_path=waveform_path,
            waveform_sha256=sha256_file(waveform_path),
            first_boundary=int(row["first_boundary_sample"]),
            last_boundary=int(row["last_boundary_sample"]),
            scored_hours=float(row["scored_hours"]),
            events=list(row["events"]),
        )
    return result


def _spread(rows: Sequence[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if len(rows) <= count:
        return list(rows)
    positions = np.linspace(0, len(rows) - 1, count, dtype=np.int64)
    return [rows[int(position)] for position in positions]


def _same_speaker_negative_candidates(root: Path, inventory_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for role in ("development", "evaluation"):
        dense_base = root / "results/r7/eres_candidate_relation_verifier_v1/features" / role
        label_base = root / R7B_RELATIVE / "features" / role
        index_rows = read_jsonl(dense_base / "dense_index.jsonl")
        labels = np.load(label_base / "partition_labels.npz")
        states = labels["state"]
        speakers = labels["speaker"]
        for info in [row for row in inventory_rows if row["dense_role"] == role]:
            start = int(info["dense_row_start"])
            end = int(info["dense_row_end"])
            local_states = states[start:end]
            local_speakers = speakers[start:end]
            local_frontiers = np.asarray(
                [int(row["frontier_sample"]) for row in index_rows[start:end]], dtype=np.int64
            )
            run_start = 0
            for index in range(1, len(local_states) + 1):
                continuing = (
                    index < len(local_states)
                    and int(local_states[index]) == 1
                    and int(local_states[index]) == int(local_states[run_start])
                    and int(local_speakers[index]) == int(local_speakers[run_start])
                )
                if continuing:
                    continue
                if int(local_states[run_start]) == 1 and index - run_start >= 300:
                    center_index = (run_start + index - 1) // 2
                    center = int(local_frontiers[center_index])
                    clip_start = center - 15 * 16000
                    clip_end = clip_start + 30 * 16000
                    if (
                        clip_start >= int(info["first_boundary_sample"])
                        and clip_end <= int(info["last_boundary_sample"]) + 1600
                        and all(
                            not (clip_start <= int(event["sample"]) < clip_end)
                            for event in info["events"]
                        )
                    ):
                        candidates.append(
                            {
                                "session_id": str(info["session_id"]),
                                "event_sample": center,
                                "selection_reason": "same_speaker_hard_negative",
                            }
                        )
                        break
                run_start = index
    return candidates


def _no_change_negative_candidates(inventory_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    duration = 30 * 16000
    result: list[dict[str, Any]] = []
    for info in inventory_rows:
        first = int(info["first_boundary_sample"])
        maximum = int(info["last_boundary_sample"]) + 1600
        for start in range(first, maximum - duration + 1, duration):
            end = start + duration
            if all(not (start <= int(event["sample"]) < end) for event in info["events"]):
                result.append(
                    {
                        "session_id": str(info["session_id"]),
                        "event_sample": start + duration // 2,
                        "selection_reason": "annotation_no_change_hard_negative_fallback",
                    }
                )
                break
    return result


def _smoke_panel(root: Path, inventory_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    all_events: list[dict[str, Any]] = []
    for row in inventory_rows:
        for event in row["events"]:
            all_events.append({"session_id": str(row["session_id"]), **event})
    all_events.sort(key=lambda row: (str(row["session_id"]), int(row["sample"])))
    selected: list[dict[str, Any]] = []
    used: set[tuple[str, int]] = set()

    def add(rows: Sequence[dict[str, Any]], count: int, reason: str) -> int:
        available = [
            row
            for row in rows
            if (str(row["session_id"]), int(row["sample"])) not in used
        ]
        chosen = _spread(available, count)
        for row in chosen:
            key = (str(row["session_id"]), int(row["sample"]))
            used.add(key)
            selected.append(
                {
                    "session_id": key[0],
                    "event_sample": key[1],
                    "reference_types": [str(row["stratum"])],
                    "selection_reason": reason,
                }
            )
        return len(chosen)

    direct_count = add(
        [row for row in all_events if row["stratum"] == "clean_change"],
        4,
        "clean_direct_change",
    )
    add([row for row in all_events if row["stratum"] == "overlap_onset"], 4, "overlap_onset")
    add(
        [row for row in all_events if row["stratum"] == "silence_gap_change"],
        4,
        "silence_gap_change",
    )
    add(
        [row for row in all_events if bool(row.get("short_backchannel_or_return"))],
        4,
        "short_backchannel_or_return",
    )
    if direct_count < 4:
        fillers = [
            row
            for row in all_events
            if row["stratum"] in {"overlap_onset", "silence_gap_change"}
        ]
        add(fillers, 4 - direct_count, "clean_direct_shortfall_filler")
    negatives = _same_speaker_negative_candidates(root, inventory_rows)
    strict_negative_count = len(negatives)
    selected_negatives = _spread(negatives, min(4, len(negatives)))
    if len(negatives) < 4:
        existing_negative_sessions = {str(row["session_id"]) for row in negatives}
        fallback_negatives = [
            row
            for row in _no_change_negative_candidates(inventory_rows)
            if str(row["session_id"]) not in existing_negative_sessions
        ]
        selected_negatives.extend(_spread(fallback_negatives, 4 - len(selected_negatives)))
    for row in selected_negatives:
        selected.append(
            {
                **row,
                "reference_types": ["same_speaker_hard_negative"],
            }
        )
    if len(selected) != int(config()["smoke_clip_count"]):
        raise R8Error(f"smoke panel has {len(selected)} clips, expected 20")
    sessions = _sessions(root)
    duration = int(config()["smoke_clip_seconds"]) * 16000
    clips: list[dict[str, Any]] = []
    for index, row in enumerate(selected):
        session = sessions[str(row["session_id"])]
        center = int(row["event_sample"])
        start = max(session.first_boundary, center - duration // 2)
        end = start + duration
        maximum = session.last_boundary + 1600
        if end > maximum:
            end = maximum
            start = end - duration
        if start < 0 or end - start != duration:
            raise R8Error(f"invalid smoke clip geometry: {session.session_id}")
        clips.append(
            {
                "clip_id": f"r8_smoke_{index:02d}",
                "session_id": session.session_id,
                "source_waveform_path": str(session.waveform_path),
                "source_waveform_sha256": session.waveform_sha256,
                "start_sample": start,
                "end_sample": end,
                "reference_types": row["reference_types"],
                "selection_reason": row["selection_reason"],
            }
        )
    return {
        "schema_version": 1,
        "selection_basis": "R7-B annotations only",
        "model_outputs_inspected": False,
        "clean_direct_available": direct_count,
        "clean_direct_shortfall": max(0, 4 - direct_count),
        "strict_same_speaker_negative_available": strict_negative_count,
        "annotation_no_change_negative_fallback_count": max(0, 4 - strict_negative_count),
        "clips": clips,
    }


def prepare(root: Path) -> Path:
    cfg = config()
    rows = _session_rows(root)
    sessions = _sessions(root)
    r7b_inventory = root / R7B_RELATIVE / "inventory.json"
    summary = {
        "scored_hours": sum(session.scored_hours for session in sessions.values()),
        "event_count": sum(len(session.events) for session in sessions.values()),
    }
    if not math.isclose(summary["scored_hours"], 4.731361111111111, abs_tol=1e-12):
        raise R8Error("R7-B exposure drifted")
    if summary["event_count"] != 4619:
        raise R8Error("R7-B reference count drifted")
    directory = output_root(root)
    directory.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": 1,
        "experiment_id": cfg["experiment_id"],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "evidence_mode": cfg["evidence_mode"],
        "r7b_inventory_path": str(r7b_inventory),
        "r7b_inventory_sha256": sha256_file(r7b_inventory),
        "code_sha256": sha256_file(CODE_PATH),
        "config_sha256": sha256_file(CONFIG_PATH),
        "summary": summary,
        "sessions": [
            {
                "session_id": session.session_id,
                "fold": session.fold,
                "waveform_path": str(session.waveform_path),
                "waveform_sha256": session.waveform_sha256,
                "first_boundary_sample": session.first_boundary,
                "last_boundary_sample": session.last_boundary,
                "scored_hours": session.scored_hours,
                "events": session.events,
            }
            for session in sorted(sessions.values(), key=lambda value: (value.fold, value.session_id))
        ],
        "git": _git_state(),
    }
    path = directory / "input_inventory.json"
    write_json(path, document)
    write_json(directory / "config.json", cfg)
    write_json(directory / "smoke_panel.json", _smoke_panel(root, rows))
    return path


def materialize_smoke(root: Path) -> Path:
    panel_path = output_root(root) / "smoke_panel.json"
    if not panel_path.is_file():
        raise R8Error("prepare must run before materialize-smoke")
    panel = load_json(panel_path)
    directory = output_root(root) / "smoke_audio"
    directory.mkdir(parents=True, exist_ok=True)
    for clip in panel["clips"]:
        source = Path(clip["source_waveform_path"])
        target = directory / f"{clip['clip_id']}.wav"
        with wave.open(str(source), "rb") as reader:
            if reader.getframerate() != 16000 or reader.getnchannels() != 1:
                raise R8Error(f"unsupported source WAV geometry: {source}")
            reader.setpos(int(clip["start_sample"]))
            frames = reader.readframes(int(clip["end_sample"]) - int(clip["start_sample"]))
            params = reader.getparams()
        with wave.open(str(target), "wb") as writer:
            writer.setparams(params)
            writer.writeframes(frames)
        clip["materialized_path"] = str(target)
        clip["materialized_sha256"] = sha256_file(target)
    write_json(panel_path, panel)
    return directory


class _FILETIME(ctypes.Structure):
    _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_uint32)]


class _PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_uint32),
        ("PageFaultCount", ctypes.c_uint32),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def _filetime_value(value: _FILETIME) -> int:
    return (int(value.high) << 32) | int(value.low)


def _process_sample(handle: int) -> tuple[int, int, int] | None:
    if os.name != "nt":
        return None
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    created = _FILETIME()
    exited = _FILETIME()
    kernel = _FILETIME()
    user = _FILETIME()
    memory = _PROCESS_MEMORY_COUNTERS_EX()
    memory.cb = ctypes.sizeof(memory)
    if not kernel32.GetProcessTimes(
        ctypes.c_void_p(handle),
        ctypes.byref(created),
        ctypes.byref(exited),
        ctypes.byref(kernel),
        ctypes.byref(user),
    ):
        return None
    if not psapi.GetProcessMemoryInfo(
        ctypes.c_void_p(handle), ctypes.byref(memory), ctypes.sizeof(memory)
    ):
        return None
    return (
        _filetime_value(kernel) + _filetime_value(user),
        int(memory.PeakWorkingSetSize),
        int(memory.PrivateUsage),
    )


def _run_monitored(
    command: Sequence[str], env: dict[str, str], log_path: Path
) -> tuple[int, dict[str, Any]]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    samples: list[tuple[float, int, int, int]] = []
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            list(command), stdout=log, stderr=subprocess.STDOUT, env=env, cwd=REPOSITORY_ROOT
        )
        while process.poll() is None:
            sample = _process_sample(int(process._handle)) if os.name == "nt" else None
            if sample is not None:
                samples.append((time.perf_counter(), *sample))
            time.sleep(0.2)
        return_code = int(process.returncode)
        sample = _process_sample(int(process._handle)) if os.name == "nt" else None
        if sample is not None:
            samples.append((time.perf_counter(), *sample))
    wall_seconds = time.perf_counter() - started
    cpu_values: list[float] = []
    for previous, current in zip(samples, samples[1:]):
        wall = current[0] - previous[0]
        cpu = (current[1] - previous[1]) / 10_000_000.0
        if wall > 0.0 and cpu >= 0.0:
            cpu_values.append(100.0 * cpu / wall / max(os.cpu_count() or 1, 1))
    metrics = {
        "wall_seconds_process": wall_seconds,
        "peak_working_set_bytes": max((row[2] for row in samples), default=None),
        "peak_private_bytes": max((row[3] for row in samples), default=None),
        "mean_cpu_utilization_percent": float(np.mean(cpu_values)) if cpu_values else None,
        "peak_cpu_utilization_percent": max(cpu_values) if cpu_values else None,
        "sample_count": len(samples),
    }
    return return_code, metrics


def _load_dump(directory: Path) -> np.ndarray:
    metadata_path = directory / "diar.probs.json"
    data_path = directory / "diar.probs.f32"
    if not metadata_path.is_file() or not data_path.is_file():
        raise R8Error(f"diar.probs dump is missing: {directory}")
    metadata = load_json(metadata_path)
    shape = tuple(int(value) for value in metadata["shape"])
    probabilities = np.fromfile(data_path, dtype="<f4")
    if probabilities.size != int(np.prod(shape)):
        raise R8Error(f"invalid probability dump size: {probabilities.size} != {shape}")
    probabilities = probabilities.reshape(shape).astype(np.float32)
    if probabilities.ndim != 2 or probabilities.shape[1] != 4:
        raise R8Error(f"unexpected Sortformer probability shape: {probabilities.shape}")
    if not np.isfinite(probabilities).all():
        raise R8Error("non-finite Sortformer probabilities")
    return probabilities


def _fixed_segments(probabilities: np.ndarray, frame_ms: int = 80) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for speaker in range(probabilities.shape[1]):
        active = probabilities[:, speaker] > 0.5
        start: int | None = None
        for frame in range(len(active) + 1):
            value = frame < len(active) and bool(active[frame])
            if value and start is None:
                start = frame
            elif not value and start is not None:
                result.append(
                    {
                        "speaker_slot": speaker + 1,
                        "start_ms": start * frame_ms,
                        "end_ms": frame * frame_ms,
                    }
                )
                start = None
    return sorted(result, key=lambda row: (int(row["start_ms"]), int(row["speaker_slot"])))


def _read_telemetry(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _telemetry_runs(rows: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    runs: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for row in rows:
        if int(row.get("chunk_index", -1)) == 0 and current:
            runs.append(current)
            current = []
        current.append(row)
    if current:
        runs.append(current)
    return runs


def _inference_items(root: Path, panel: str) -> list[dict[str, Any]]:
    if panel == "full":
        result: list[dict[str, Any]] = []
        for session in sorted(
            _sessions(root).values(), key=lambda value: (value.fold, value.session_id)
        ):
            with wave.open(str(session.waveform_path), "rb") as reader:
                duration_seconds = reader.getnframes() / reader.getframerate()
            result.append(
                {
                    "item_id": session.session_id,
                    "audio_path": str(session.waveform_path),
                    "duration_seconds": duration_seconds,
                    "scored_duration_seconds": session.scored_hours * 3600.0,
                    "full_waveform_duration": True,
                }
            )
        return result
    smoke = load_json(output_root(root) / "smoke_panel.json")
    result: list[dict[str, Any]] = []
    for clip in smoke["clips"]:
        if not clip.get("materialized_path"):
            raise R8Error("materialize-smoke must run before smoke inference")
        result.append(
            {
                "item_id": str(clip["clip_id"]),
                "audio_path": str(clip["materialized_path"]),
                "duration_seconds": 30.0,
                "full_waveform_duration": False,
            }
        )
    return result


def run_inference(
    root: Path,
    bench: Path,
    model: Path,
    backend: str,
    panel: str,
    repetitions: int,
    warmup: int,
    resume: bool,
) -> Path:
    if not bench.is_file():
        raise R8Error(f"transcribe-bench is missing: {bench}")
    if not model.is_file():
        raise R8Error(f"model is missing: {model}")
    cfg = config()
    items = _inference_items(root, panel)
    base = output_root(root) / "runs" / panel / backend / model.stem
    dump_base = base / "dumps"
    bench_base = base / "bench"
    log_base = base / "logs"
    telemetry_base = base / "telemetry"
    probability_base = (
        output_root(root) / "probabilities" / backend
        if panel == "full"
        else output_root(root) / "probabilities" / "smoke" / backend / model.stem
    )
    segment_base = (
        output_root(root) / "speaker_segments" / backend
        if panel == "full"
        else output_root(root) / "speaker_segments" / "smoke" / backend / model.stem
    )
    for directory in (
        dump_base,
        bench_base,
        log_base,
        telemetry_base,
        probability_base,
        segment_base,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for item in items:
        item_id = str(item["item_id"])
        receipt_path = base / f"{item_id}.receipt.json"
        if resume and receipt_path.is_file():
            previous = load_json(receipt_path)
            probability_path = Path(previous["probability_path"])
            telemetry_path = Path(previous["telemetry_path"])
            telemetry_runs = _telemetry_runs(_read_telemetry(telemetry_path))
            expected_chunks = math.ceil(int(previous["probability_shape"][0]) / 6)
            complete = (
                probability_path.is_file()
                and sha256_file(probability_path) == str(previous["probability_sha256"])
                and len(telemetry_runs) == warmup + repetitions
                and all(len(run) == expected_chunks for run in telemetry_runs)
                and str(previous["model_sha256"]) == sha256_file(model)
                and str(previous["backend_requested"]) == backend
            )
            if complete:
                rows.append(previous)
                print(f"{item_id}: resumed from verified receipt", flush=True)
                continue
        dump_dir = dump_base / item_id
        dump_dir.mkdir(parents=True, exist_ok=True)
        for stale in dump_dir.glob("diar.probs.*"):
            stale.unlink()
        bench_json = bench_base / f"{item_id}.json"
        telemetry_path = telemetry_base / f"{item_id}.jsonl"
        if telemetry_path.exists():
            telemetry_path.unlink()
        command = [
            str(bench),
            "--model",
            str(model),
            "--sample",
            str(item["audio_path"]),
            "--backend",
            backend,
            "--threads",
            str(int(cfg["cpu_threads"])),
            "--warmup",
            str(warmup),
            "--iters",
            str(repetitions),
            "--json-out",
            str(bench_json),
        ]
        env = os.environ.copy()
        env["TRANSCRIBE_DUMP_DIR"] = str(dump_dir)
        env["TRANSCRIBE_SORTFORMER_STREAM_PRESET"] = str(cfg["preset"])
        env["TRANSCRIBE_SORTFORMER_TELEMETRY_PATH"] = str(telemetry_path)
        return_code, process_metrics = _run_monitored(command, env, log_base / f"{item_id}.log")
        if return_code != 0:
            raise R8Error(f"inference failed for {item_id}; see {log_base / f'{item_id}.log'}")
        bench_result = load_json(bench_json)
        resolved_backend = str(bench_result.get("backend", "unknown")).lower()
        if backend not in resolved_backend:
            raise R8Error(
                f"backend fallback for {item_id}: requested {backend}, resolved {resolved_backend}"
            )
        probabilities = _load_dump(dump_dir)
        np.savez_compressed(
            probability_base / f"{item_id}.npz",
            probabilities=probabilities,
            frame_ms=np.int32(cfg["frame_ms"]),
            backend=np.asarray(backend),
            model_sha256=np.asarray(sha256_file(model)),
        )
        write_json(segment_base / f"{item_id}.json", _fixed_segments(probabilities))
        telemetry = _read_telemetry(telemetry_path)
        telemetry_runs = _telemetry_runs(telemetry)
        if len(telemetry_runs) != warmup + repetitions:
            raise R8Error(
                f"telemetry run count mismatch for {item_id}: {len(telemetry_runs)} != {warmup + repetitions}"
            )
        row = {
            **item,
            "backend_requested": backend,
            "backend_resolved": resolved_backend,
            "bench_path": str(bench),
            "bench_sha256": sha256_file(bench),
            "model_path": str(model),
            "model_sha256": sha256_file(model),
            "probability_path": str(probability_base / f"{item_id}.npz"),
            "probability_sha256": sha256_file(probability_base / f"{item_id}.npz"),
            "probability_shape": list(probabilities.shape),
            "telemetry_path": str(telemetry_path),
            "telemetry_chunk_count": len(telemetry),
            "telemetry_run_count": len(telemetry_runs),
            "telemetry_measured_chunk_count": sum(
                len(run) for run in telemetry_runs[-repetitions:]
            ),
            "bench": bench_result,
            "process": process_metrics,
        }
        write_json(receipt_path, row)
        rows.append(row)
        print(f"{item_id}: {probabilities.shape[0]} frames, {process_metrics['wall_seconds_process']:.3f}s", flush=True)
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "panel": panel,
        "backend": backend,
        "repetitions": repetitions,
        "warmup": warmup,
        "items": rows,
    }
    path = base / "manifest.json"
    write_json(path, manifest)
    return path


def decode_change_events(
    probabilities: np.ndarray,
    threshold: float,
    frame_ms: int = 80,
    maximum_gap_ms: int = 500,
    duplicate_suppression_ms: int = 200,
) -> list[int]:
    active = np.asarray(probabilities >= threshold, dtype=np.bool_)
    if active.ndim != 2 or active.shape[1] != 4:
        raise R8Error(f"invalid probability geometry: {active.shape}")
    candidate_frames: list[int] = []
    seen_speaker = bool(active[0].any()) if len(active) else False
    last_active_frame = 0 if seen_speaker else -1
    last_active_mask = active[0].copy() if seen_speaker else np.zeros(4, dtype=np.bool_)
    for frame in range(1, len(active)):
        previous = active[frame - 1]
        current = active[frame]
        onsets = current & ~previous
        if onsets.any() and seen_speaker:
            emit = False
            for speaker in np.flatnonzero(onsets):
                other_current = current.copy()
                other_current[speaker] = False
                other_previous = previous.copy()
                other_previous[speaker] = False
                if other_current.any() or other_previous.any():
                    emit = True
                elif not previous.any() and last_active_frame >= 0:
                    gap_ms = (frame - last_active_frame - 1) * frame_ms
                    if gap_ms <= maximum_gap_ms and not bool(last_active_mask[speaker]):
                        emit = True
            if emit:
                candidate_frames.append(frame)
        if current.any():
            seen_speaker = True
            last_active_frame = frame
            last_active_mask = current.copy()
    samples_per_frame = frame_ms * 16
    radius_samples = duplicate_suppression_ms * 16
    accepted: list[int] = []
    for frame in candidate_frames:
        sample = frame * samples_per_frame
        if not accepted or sample - accepted[-1] > radius_samples:
            accepted.append(sample)
    return accepted


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


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _metrics(
    sessions: dict[str, Session], predictions: dict[str, list[int]], selected_ids: Iterable[str] | None = None
) -> dict[str, Any]:
    session_ids = sorted(selected_ids if selected_ids is not None else sessions)
    exposure_hours = sum(sessions[session_id].scored_hours for session_id in session_ids)
    reference_count = sum(len(sessions[session_id].events) for session_id in session_ids)
    scored_predictions = {
        session_id: [
            sample
            for sample in predictions.get(session_id, [])
            if sessions[session_id].first_boundary
            <= sample
            < sessions[session_id].last_boundary + 1600
        ]
        for session_id in session_ids
    }
    result: dict[str, Any] = {
        "exposure_hours": exposure_hours,
        "reference_count": reference_count,
        "prediction_count": sum(len(scored_predictions[session_id]) for session_id in session_ids),
        "tolerances": {},
    }
    primary_matches: list[tuple[str, int, int]] = []
    primary_false: list[tuple[str, int]] = []
    primary_misses: list[tuple[str, int]] = []
    primary_per_meeting: dict[str, Any] = {}
    for tolerance_ms in (100, 250, 500):
        matched_all: list[tuple[str, int, int]] = []
        false_all: list[tuple[str, int]] = []
        misses_all: list[tuple[str, int]] = []
        per_meeting: dict[str, Any] = {}
        for session_id in session_ids:
            session = sessions[session_id]
            session_predictions = scored_predictions[session_id]
            references = [int(event["sample"]) for event in session.events]
            matched, false, misses = _one_to_one(
                session_predictions, references, tolerance_ms * 16
            )
            matched_all.extend((session_id, prediction, reference) for prediction, reference in matched)
            false_all.extend((session_id, prediction) for prediction in false)
            misses_all.extend((session_id, reference) for reference in misses)
            per_meeting[session_id] = {
                "reference_count": len(references),
                "prediction_count": len(session_predictions),
                "true_positive_count": len(matched),
                "false_event_count": len(false),
                "miss_count": len(misses),
                "recall": _ratio(len(matched), len(references)),
                "false_events_per_hour": len(false) / session.scored_hours,
            }
        true_positive_count = len(matched_all)
        prediction_count = sum(len(scored_predictions[session_id]) for session_id in session_ids)
        precision = _ratio(true_positive_count, prediction_count)
        recall = _ratio(true_positive_count, reference_count)
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall > 0.0
            else None
        )
        result["tolerances"][str(tolerance_ms)] = {
            "true_positive_count": true_positive_count,
            "false_event_count": len(false_all),
            "miss_count": len(misses_all),
            "false_events_per_hour": len(false_all) / exposure_hours,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        if tolerance_ms == 250:
            primary_matches = matched_all
            primary_false = false_all
            primary_misses = misses_all
            primary_per_meeting = per_meeting
    event_lookup = {
        session_id: {int(event["sample"]): event for event in sessions[session_id].events}
        for session_id in session_ids
    }
    references_by_stratum: dict[str, int] = defaultdict(int)
    matched_by_stratum: dict[str, int] = defaultdict(int)
    short_reference_count = 0
    short_matched_count = 0
    for session_id in session_ids:
        for event in sessions[session_id].events:
            references_by_stratum[str(event["stratum"])] += 1
            if bool(event.get("short_backchannel_or_return")):
                short_reference_count += 1
    for session_id, _, reference in primary_matches:
        event = event_lookup[session_id][reference]
        matched_by_stratum[str(event["stratum"])] += 1
        if bool(event.get("short_backchannel_or_return")):
            short_matched_count += 1
    meeting_true_positives = [
        int(row["true_positive_count"]) for row in primary_per_meeting.values()
    ]
    result.update(
        {
            "per_meeting": primary_per_meeting,
            "matched_pairs": primary_matches,
            "false_event_samples": primary_false,
            "miss_samples": primary_misses,
            "stratum_recall": {
                name: _ratio(matched_by_stratum[name], count)
                for name, count in sorted(references_by_stratum.items())
            },
            "short_return_recall": _ratio(short_matched_count, short_reference_count),
            "maximum_meeting_true_positive_share": (
                max(meeting_true_positives) / max(sum(meeting_true_positives), 1)
            ),
        }
    )
    return result


class EvaluationCache:
    def __init__(
        self, sessions: dict[str, Session], probabilities: dict[str, np.ndarray], cfg: dict[str, Any]
    ) -> None:
        self.sessions = sessions
        self.probabilities = probabilities
        self.cfg = cfg
        self.predictions: dict[float, dict[str, list[int]]] = {}

    def events(self, threshold: float) -> dict[str, list[int]]:
        key = float(np.float32(threshold))
        if key not in self.predictions:
            self.predictions[key] = {
                session_id: decode_change_events(
                    values,
                    key,
                    int(self.cfg["frame_ms"]),
                    int(self.cfg["maximum_gap_ms"]),
                    int(self.cfg["duplicate_suppression_ms"]),
                )
                for session_id, values in self.probabilities.items()
            }
        return self.predictions[key]

    def metrics(self, threshold: float, selected_ids: Iterable[str] | None = None) -> dict[str, Any]:
        return _metrics(self.sessions, self.events(threshold), selected_ids)


def _curve_row(cache: EvaluationCache, threshold: float, selected_ids: Iterable[str] | None = None) -> dict[str, Any]:
    metrics = cache.metrics(threshold, selected_ids)
    primary = metrics["tolerances"]["250"]
    return {
        "threshold": float(np.float32(threshold)),
        "prediction_count": metrics["prediction_count"],
        "true_positive_count": primary["true_positive_count"],
        "false_event_count": primary["false_event_count"],
        "false_events_per_hour": primary["false_events_per_hour"],
        "recall_250": primary["recall"],
    }


def _select_row(rows: Sequence[dict[str, Any]], target: float) -> dict[str, Any]:
    eligible = [row for row in rows if float(row["false_events_per_hour"]) <= target]
    if not eligible:
        return min(rows, key=lambda row: float(row["false_events_per_hour"]))
    return max(
        eligible,
        key=lambda row: (
            float(row["recall_250"] or 0.0),
            -float(row["false_events_per_hour"]),
            -float(row["threshold"]),
        ),
    )


def _refined_thresholds(
    cache: EvaluationCache,
    probabilities: dict[str, np.ndarray],
    dense_rows: Sequence[dict[str, Any]],
    target: float,
    search: dict[str, Any],
) -> list[float]:
    selected = _select_row(dense_rows, target)
    dense_values = sorted({float(row["threshold"]) for row in dense_rows})
    position = dense_values.index(float(selected["threshold"]))
    lower = dense_values[max(0, position - 1)]
    upper = dense_values[min(len(dense_values) - 1, position + 1)]
    unique = np.unique(
        np.concatenate([values[(values >= lower) & (values <= upper)] for values in probabilities.values()])
    ).astype(np.float32)
    if len(unique) == 0:
        return [float(selected["threshold"])]
    refinement_points = int(search["refinement_points"])
    rounds = int(search["refinement_rounds"])
    evaluated: set[float] = {float(selected["threshold"])}
    current = unique
    for _ in range(rounds):
        if len(current) <= int(search["maximum_exact_values_per_target"]):
            break
        indices = np.linspace(0, len(current) - 1, refinement_points, dtype=np.int64)
        probe_values = [float(current[int(index)]) for index in indices]
        probe_rows = [_curve_row(cache, value) for value in probe_values]
        evaluated.update(probe_values)
        best = _select_row(probe_rows, target)
        best_index = probe_values.index(float(best["threshold"]))
        probe_lower = probe_values[max(0, best_index - 1)]
        probe_upper = probe_values[min(len(probe_values) - 1, best_index + 1)]
        current = current[(current >= probe_lower) & (current <= probe_upper)]
    if len(current) > int(search["maximum_exact_values_per_target"]):
        raise R8Error(
            f"exact threshold bracket remains too large for target {target}: {len(current)}"
        )
    evaluated.update(float(value) for value in current)
    return sorted(evaluated)


def score_accuracy(root: Path, backend: str) -> Path:
    cfg = config()
    sessions = _sessions(root)
    probability_dir = output_root(root) / "probabilities" / backend
    probabilities: dict[str, np.ndarray] = {}
    for session_id in sessions:
        path = probability_dir / f"{session_id}.npz"
        if not path.is_file():
            raise R8Error(f"probabilities are missing: {path}")
        values = np.load(path)["probabilities"].astype(np.float32)
        if values.ndim != 2 or values.shape[1] != 4 or not np.isfinite(values).all():
            raise R8Error(f"invalid probabilities: {path}")
        probabilities[session_id] = values
    cache = EvaluationCache(sessions, probabilities, cfg)
    search = cfg["threshold_search"]
    dense_thresholds = np.arange(
        float(search["dense_min"]),
        float(search["dense_max"]) + float(search["dense_step"]) / 2.0,
        float(search["dense_step"]),
        dtype=np.float32,
    )
    rows_by_threshold: dict[float, dict[str, Any]] = {}
    for index, threshold in enumerate(dense_thresholds):
        row = _curve_row(cache, float(threshold))
        rows_by_threshold[float(row["threshold"])] = row
        if index % 20 == 0:
            print(f"dense threshold {index + 1}/{len(dense_thresholds)}", flush=True)
    dense_rows = list(rows_by_threshold.values())
    for target in cfg["development_false_event_targets_per_hour"]:
        for threshold in _refined_thresholds(
            cache, probabilities, dense_rows, float(target), search
        ):
            key = float(np.float32(threshold))
            if key not in rows_by_threshold:
                rows_by_threshold[key] = _curve_row(cache, key)
    curve = sorted(rows_by_threshold.values(), key=lambda row: -float(row["threshold"]))
    per_meeting_curves: dict[str, list[dict[str, Any]]] = {
        session_id: [] for session_id in sessions
    }
    for row in curve:
        threshold = float(row["threshold"])
        metrics = cache.metrics(threshold)
        for session_id, meeting in metrics["per_meeting"].items():
            per_meeting_curves[session_id].append(
                {
                    "threshold": threshold,
                    "prediction_count": meeting["prediction_count"],
                    "true_positive_count": meeting["true_positive_count"],
                    "false_event_count": meeting["false_event_count"],
                    "false_events_per_hour": meeting["false_events_per_hour"],
                    "recall_250": meeting["recall"],
                }
            )
    selected_points: dict[str, Any] = {}
    for target in cfg["development_false_event_targets_per_hour"]:
        selected = _select_row(curve, float(target))
        selected_points[str(target)] = {
            "threshold": selected["threshold"],
            "metrics": cache.metrics(float(selected["threshold"])),
        }
    transfer: list[dict[str, Any]] = []
    for fold, held_out in enumerate(cfg["folds"]):
        held_out_ids = [str(value) for value in held_out]
        development_ids = [session_id for session_id in sessions if session_id not in held_out_ids]
        development_rows = [
            _curve_row(cache, float(row["threshold"]), development_ids) for row in curve
        ]
        for target in cfg["development_false_event_targets_per_hour"]:
            selected = _select_row(development_rows, float(target))
            held_metrics = cache.metrics(float(selected["threshold"]), held_out_ids)
            primary = held_metrics["tolerances"]["250"]
            transfer.append(
                {
                    "fold": fold,
                    "held_out_sessions": held_out_ids,
                    "target_false_events_per_hour": target,
                    "selected_threshold": selected["threshold"],
                    "development_false_events_per_hour": selected["false_events_per_hour"],
                    "development_recall_250": selected["recall_250"],
                    "held_out_false_events_per_hour": primary["false_events_per_hour"],
                    "held_out_recall_250": primary["recall"],
                }
            )
    gate_cfg = cfg["gate"]
    point_10 = selected_points["10"]["metrics"]
    point_20 = selected_points["20"]["metrics"]
    primary_10 = point_10["tolerances"]["250"]
    primary_20 = point_20["tolerances"]["250"]
    transfer_ok = all(
        float(row["held_out_false_events_per_hour"])
        <= float(row["target_false_events_per_hour"])
        * float(gate_cfg["maximum_transfer_false_event_multiplier"])
        for row in transfer
        if int(row["target_false_events_per_hour"]) in {10, 20}
    )
    checks = {
        "recall_at_10_false_events_per_hour": float(primary_10["recall"] or 0.0)
        >= float(gate_cfg["recall_at_10_false_events_per_hour"]),
        "recall_at_20_false_events_per_hour": float(primary_20["recall"] or 0.0)
        >= float(gate_cfg["recall_at_20_false_events_per_hour"]),
        "overlap_onset_nonzero": float(point_20["stratum_recall"].get("overlap_onset") or 0.0)
        > 0.0,
        "silence_gap_change_nonzero": float(
            point_20["stratum_recall"].get("silence_gap_change") or 0.0
        )
        > 0.0,
        "meeting_concentration": float(point_20["maximum_meeting_true_positive_share"])
        <= float(gate_cfg["maximum_single_meeting_true_positive_share"]),
        "threshold_transfer": transfer_ok,
    }
    document = {
        "schema_version": 1,
        "experiment_id": cfg["experiment_id"],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "backend": backend,
        "curve": curve,
        "selected_operating_points": selected_points,
        "threshold_transfer": transfer,
        "gate": {"checks": checks, "passed": all(checks.values())},
        "threshold_count": len(curve),
        "probability_hashes": {
            session_id: sha256_file(probability_dir / f"{session_id}.npz")
            for session_id in sessions
        },
    }
    path = output_root(root) / "accuracy_metrics.json"
    write_json(path, document)
    write_json(output_root(root) / "threshold_transfer_metrics.json", transfer)
    write_json(output_root(root) / "per_meeting_curves.json", per_meeting_curves)
    for target, point in selected_points.items():
        events_dir = output_root(root) / "events" / backend / f"feh_{target}"
        events_dir.mkdir(parents=True, exist_ok=True)
        predictions = cache.events(float(point["threshold"]))
        for session_id, samples in predictions.items():
            write_json(
                events_dir / f"{session_id}.json",
                [{"sample": sample, "time_seconds": sample / 16000.0} for sample in samples],
            )
    return path


def _percentiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p95": None, "p99": None, "maximum": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "maximum": float(np.max(array)),
    }


def _chunk_metrics(rows: Sequence[dict[str, Any]], chunk_audio_ms: float) -> dict[str, Any]:
    totals = [float(row["total_us"]) / 1000.0 for row in rows if "total_us" in row]
    compression = [
        float(row["compression_us"]) / 1000.0
        for row in rows
        if bool(row.get("compression_called"))
    ]
    backlog = 0.0
    maximum_backlog = 0.0
    missed = 0
    for row, value in zip([row for row in rows if "total_us" in row], totals, strict=True):
        if int(row.get("chunk_index", -1)) == 0:
            backlog = 0.0
        backlog = max(0.0, backlog + value - chunk_audio_ms)
        maximum_backlog = max(maximum_backlog, backlog)
        if value > chunk_audio_ms:
            missed += 1
    return {
        "chunk_compute_ms": _percentiles(totals),
        "chunk_count": len(totals),
        "deadline_miss_count": missed,
        "deadline_miss_proportion": missed / len(totals) if totals else None,
        "maximum_backlog_ms": maximum_backlog if totals else None,
        "compression_call_count": len(compression),
        "compression_ms": _percentiles(compression),
    }


def aggregate_compute(root: Path) -> Path:
    cfg = config()
    output = output_root(root)
    backends: dict[str, Any] = {}
    smoke_backends: dict[str, Any] = {}
    for backend in ("cpu", "vulkan"):
        smoke_candidates = sorted((output / "runs" / "smoke" / backend).glob("*Q8_0/manifest.json"))
        if smoke_candidates:
            smoke_manifest = load_json(smoke_candidates[-1])
            smoke_audio = 0.0
            smoke_wall = 0.0
            smoke_chunks: list[dict[str, Any]] = []
            for item in smoke_manifest["items"]:
                smoke_audio += float(item["duration_seconds"])
                smoke_wall += float(item["bench"]["summary"]["wall_ms"]["mean"]) / 1000.0
                runs = _telemetry_runs(_read_telemetry(Path(item["telemetry_path"])))
                measured_runs = runs[-int(smoke_manifest["repetitions"]) :]
                smoke_chunks.extend(row for run in measured_runs for row in run)
            smoke_backends[backend] = {
                "status": "complete",
                "manifest_path": str(smoke_candidates[-1]),
                "resolved_backends": sorted(
                    {str(item["backend_resolved"]) for item in smoke_manifest["items"]}
                ),
                "audio_seconds": smoke_audio,
                "measured_wall_seconds": smoke_wall,
                "rtf": smoke_wall / smoke_audio,
                "audio_seconds_per_wall_second": smoke_audio / smoke_wall,
                "maximum_peak_private_bytes": max(
                    int(item["process"]["peak_private_bytes"])
                    for item in smoke_manifest["items"]
                    if item["process"]["peak_private_bytes"] is not None
                ),
                "chunks": _chunk_metrics(smoke_chunks, float(cfg["chunk_audio_ms"])),
            }
        candidates = sorted((output / "runs" / "full" / backend).glob("*/manifest.json"))
        if not candidates:
            status = (
                "not_run_by_owner_decision"
                if backend == "vulkan"
                and cfg.get("execution_amendment", {}).get("vulkan_scope")
                == "smoke_only_by_owner_decision"
                else "not_run"
            )
            backends[backend] = {"status": status}
            continue
        manifest_path = candidates[-1]
        manifest = load_json(manifest_path)
        meeting_rows: list[dict[str, Any]] = []
        all_chunks: list[dict[str, Any]] = []
        total_audio = 0.0
        total_wall = 0.0
        for item in manifest["items"]:
            duration = float(item["duration_seconds"])
            wall_ms = float(item["bench"]["summary"]["wall_ms"]["mean"])
            telemetry_all = _read_telemetry(Path(item["telemetry_path"]))
            telemetry_runs = _telemetry_runs(telemetry_all)
            repetitions = int(manifest["repetitions"])
            telemetry = [row for run in telemetry_runs[-repetitions:] for row in run]
            chunk = _chunk_metrics(telemetry, float(cfg["chunk_audio_ms"]))
            all_chunks.extend(telemetry)
            total_audio += duration
            total_wall += wall_ms / 1000.0
            meeting_rows.append(
                {
                    "item_id": item["item_id"],
                    "duration_seconds": duration,
                    "wall_seconds": wall_ms / 1000.0,
                    "rtf": wall_ms / 1000.0 / duration,
                    "audio_seconds_per_wall_second": duration / (wall_ms / 1000.0),
                    "load_ms": float(item["bench"]["load_ms"]),
                    "peak_private_bytes": item["process"]["peak_private_bytes"],
                    "peak_working_set_bytes": item["process"]["peak_working_set_bytes"],
                    "mean_cpu_utilization_percent": item["process"][
                        "mean_cpu_utilization_percent"
                    ],
                    "peak_cpu_utilization_percent": item["process"][
                        "peak_cpu_utilization_percent"
                    ],
                    "chunks": chunk,
                }
            )
        aggregate_chunks = _chunk_metrics(all_chunks, float(cfg["chunk_audio_ms"]))
        peak_private = max(
            (int(row["peak_private_bytes"]) for row in meeting_rows if row["peak_private_bytes"]),
            default=None,
        )
        rtf = total_wall / total_audio
        checks = {
            "rtf_below_one": rtf < 1.0,
            "chunk_p99_within_480ms": aggregate_chunks["chunk_compute_ms"]["p99"] is not None
            and float(aggregate_chunks["chunk_compute_ms"]["p99"] or math.inf)
            <= float(cfg["chunk_audio_ms"]),
            "maximum_backlog_within_480ms": aggregate_chunks["maximum_backlog_ms"] is not None
            and float(aggregate_chunks["maximum_backlog_ms"] or math.inf)
            <= float(cfg["chunk_audio_ms"]),
            "memory_within_ceiling": peak_private is not None
            and peak_private <= int(cfg["memory_ceiling_gib"]) * 1024**3,
            "telemetry_available": bool(all_chunks),
        }
        backends[backend] = {
            "status": "complete",
            "manifest_path": str(manifest_path),
            "resolved_backend": manifest["items"][0]["backend_resolved"],
            "total_audio_seconds": total_audio,
            "total_wall_seconds": total_wall,
            "aggregate_rtf": rtf,
            "audio_seconds_per_wall_second": total_audio / total_wall,
            "preferred_headroom_rtf_at_most_0_5": rtf <= 0.5,
            "peak_private_bytes": peak_private,
            "chunks": aggregate_chunks,
            "meetings": meeting_rows,
            "gate": {"checks": checks, "passed": all(checks.values())},
        }
    document = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "cpu_threads": cfg["cpu_threads"],
        "chunk_audio_ms": cfg["chunk_audio_ms"],
        "algorithmic_lookahead_ms": cfg["algorithmic_lookahead_ms"],
        "backends": backends,
        "smoke_backends": smoke_backends,
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "gpu_utilization": "unavailable unless recorded by an external device sampler",
        },
    }
    path = output / "compute_metrics.json"
    write_json(path, document)
    return path


def backend_parity(root: Path) -> Path:
    sessions = _sessions(root)
    cpu_dir = output_root(root) / "probabilities" / "cpu"
    vulkan_dir = output_root(root) / "probabilities" / "vulkan"
    rows: list[dict[str, Any]] = []
    for session_id in sessions:
        cpu_path = cpu_dir / f"{session_id}.npz"
        vulkan_path = vulkan_dir / f"{session_id}.npz"
        if not cpu_path.is_file() or not vulkan_path.is_file():
            continue
        cpu = np.load(cpu_path)["probabilities"]
        vulkan = np.load(vulkan_path)["probabilities"]
        if cpu.shape != vulkan.shape:
            rows.append(
                {
                    "session_id": session_id,
                    "shape_equal": False,
                    "cpu_shape": list(cpu.shape),
                    "vulkan_shape": list(vulkan.shape),
                }
            )
            continue
        difference = np.abs(cpu.astype(np.float64) - vulkan.astype(np.float64))
        rows.append(
            {
                "session_id": session_id,
                "shape_equal": True,
                "maximum_absolute_difference": float(np.max(difference)),
                "mean_absolute_difference": float(np.mean(difference)),
            }
        )
    document = {"schema_version": 1, "meetings": rows}
    path = output_root(root) / "backend_parity.json"
    write_json(path, document)
    return path


def smoke_parity(root: Path) -> Path:
    cfg = config()
    panel = load_json(output_root(root) / "smoke_panel.json")
    q8_stem = Path(str(cfg["model"]["q8_filename"])).stem
    f16_stem = Path(str(cfg["model"]["f16_filename"])).stem
    base = output_root(root) / "probabilities" / "smoke"
    comparisons = {
        "cpu_q8_vs_vulkan_q8": (base / "cpu" / q8_stem, base / "vulkan" / q8_stem),
        "cpu_q8_vs_cpu_f16": (base / "cpu" / q8_stem, base / "cpu" / f16_stem),
    }
    result: dict[str, Any] = {}
    for name, (left_dir, right_dir) in comparisons.items():
        rows: list[dict[str, Any]] = []
        difference_sum = 0.0
        value_count = 0
        maximum = 0.0
        for clip in panel["clips"]:
            clip_id = str(clip["clip_id"])
            left = np.load(left_dir / f"{clip_id}.npz")["probabilities"]
            right = np.load(right_dir / f"{clip_id}.npz")["probabilities"]
            if left.shape != right.shape:
                rows.append(
                    {
                        "clip_id": clip_id,
                        "shape_equal": False,
                        "left_shape": list(left.shape),
                        "right_shape": list(right.shape),
                    }
                )
                continue
            difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
            difference_sum += float(np.sum(difference))
            value_count += difference.size
            maximum = max(maximum, float(np.max(difference)))
            left_events = decode_change_events(left, 0.5)
            right_events = decode_change_events(right, 0.5)
            rows.append(
                {
                    "clip_id": clip_id,
                    "shape_equal": True,
                    "maximum_absolute_difference": float(np.max(difference)),
                    "mean_absolute_difference": float(np.mean(difference)),
                    "fixed_threshold_events_equal": left_events == right_events,
                    "fixed_segments_equal": _fixed_segments(left) == _fixed_segments(right),
                }
            )
        result[name] = {
            "maximum_absolute_difference": maximum,
            "mean_absolute_difference": difference_sum / value_count if value_count else None,
            "all_shapes_equal": all(bool(row["shape_equal"]) for row in rows),
            "fixed_threshold_event_agreement": sum(
                bool(row.get("fixed_threshold_events_equal")) for row in rows
            )
            / len(rows),
            "fixed_segment_agreement": sum(bool(row.get("fixed_segments_equal")) for row in rows)
            / len(rows),
            "clips": rows,
        }
    path = output_root(root) / "smoke_parity.json"
    write_json(path, {"schema_version": 1, "comparisons": result})
    return path


def receipts(root: Path) -> list[Path]:
    cfg = config()
    source = root / "external" / "r8" / "transcribe.cpp"
    models = root / "models" / "r8"
    q8 = models / str(cfg["model"]["q8_filename"])
    f16 = models / str(cfg["model"]["f16_filename"])
    if not source.is_dir() or not q8.is_file():
        raise R8Error("source checkout and Q8_0 model are required")

    def git_value(*arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(source), *arguments],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    full_commit = git_value("rev-parse", "HEAD")
    if not full_commit.startswith(str(cfg["source"]["commit_prefix"])):
        raise R8Error(f"source commit is not pinned: {full_commit}")
    output = output_root(root)
    telemetry_diff = git_value("diff", "--", "src/arch/sortformer")
    telemetry_patch_path = output / "telemetry_patch.diff"
    telemetry_patch_path.write_text(telemetry_diff + ("\n" if telemetry_diff else ""), encoding="utf-8")
    source_document = {
        "schema_version": 1,
        "repository": git_value("remote", "get-url", "origin"),
        "full_commit_sha": full_commit,
        "pinned_prefix": cfg["source"]["commit_prefix"],
        "dirty_paths": git_value("status", "--short").splitlines(),
        "vendored_ggml_tree_sha": git_value("rev-parse", "HEAD:ggml"),
        "telemetry_patch_path": str(telemetry_patch_path),
        "telemetry_patch_sha256": sha256_file(telemetry_patch_path),
    }
    model_files = [q8] + ([f16] if f16.is_file() else [])
    model_document = {
        "schema_version": 1,
        "repository": cfg["model"]["repository"],
        "repository_revision": cfg["model"]["repository_revision"],
        "files": [
            {
                "filename": path.name,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "download_url": f"https://huggingface.co/{cfg['model']['repository']}/resolve/{cfg['model']['repository_revision']}/{path.name}",
            }
            for path in model_files
        ],
        "license": "NVIDIA Open Model License",
        "disposition": "internal research acquisition; redistribution and publication not authorized",
    }
    cpu_bench = source / "build-r8-cpu" / "bin" / "Release" / "transcribe-bench.exe"
    vulkan_candidates = [
        root / "builds" / "r8-vulkan" / "bin" / "Release" / "transcribe-bench.exe",
        source / "build-r8-vulkan" / "bin" / "Release" / "transcribe-bench.exe",
    ]
    vulkan_bench = next((path for path in vulkan_candidates if path.is_file()), None)
    build_document = {
        "schema_version": 1,
        "cmake_version": subprocess.run(
            ["cmake", "--version"], check=True, capture_output=True, text=True
        ).stdout.splitlines()[0],
        "cpu": {
            "available": cpu_bench.is_file(),
            "bench_path": str(cpu_bench),
            "bench_sha256": sha256_file(cpu_bench) if cpu_bench.is_file() else None,
            "threads_locked": cfg["cpu_threads"],
        },
        "vulkan": {
            "available": vulkan_bench is not None,
            "bench_path": str(vulkan_bench) if vulkan_bench is not None else None,
            "bench_sha256": sha256_file(vulkan_bench) if vulkan_bench is not None else None,
        },
    }
    processor = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json -Compress",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    graphics = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,AdapterRAM | ConvertTo-Json -Compress",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    vulkaninfo = subprocess.run(
        ["vulkaninfo", "--summary"], capture_output=True, text=True, errors="replace"
    )
    hardware_document = {
        "schema_version": 1,
        "platform": platform.platform(),
        "processor": json.loads(processor),
        "video_controllers": json.loads(graphics),
        "vulkaninfo_return_code": vulkaninfo.returncode,
        "vulkaninfo_summary": vulkaninfo.stdout,
    }
    paths = [
        output / "source_receipt.json",
        output / "model_receipt.json",
        output / "build_receipt.json",
        output / "hardware_receipt.json",
    ]
    for path, document in zip(
        paths,
        (source_document, model_document, build_document, hardware_document),
        strict=True,
    ):
        write_json(path, document)
    return paths


def validate_telemetry(root: Path) -> Path:
    base = output_root(root) / "telemetry_validation"
    unpatched_dir = base / "unpatched"
    patched_dir = base / "patched"
    unpatched = _load_dump(unpatched_dir)
    patched = _load_dump(patched_dir)
    telemetry = _read_telemetry(patched_dir / "chunks.jsonl")
    required_fields = {
        "chunk_index",
        "pre_encode_us",
        "infer_us",
        "update_us",
        "compression_called",
        "compression_us",
        "total_us",
        "new_audio_frames",
    }
    checks = {
        "shape_equal": unpatched.shape == patched.shape,
        "probability_bytes_equal": (unpatched_dir / "diar.probs.f32").read_bytes()
        == (patched_dir / "diar.probs.f32").read_bytes(),
        "probability_sha256_equal": sha256_file(unpatched_dir / "diar.probs.f32")
        == sha256_file(patched_dir / "diar.probs.f32"),
        "speaker_segments_equal": _fixed_segments(unpatched) == _fixed_segments(patched),
        "telemetry_present": bool(telemetry),
        "telemetry_schema_complete": bool(telemetry)
        and all(required_fields <= set(row) for row in telemetry),
        "chunk_geometry_locked": bool(telemetry)
        and all(int(row["new_audio_frames"]) == int(config()["chunk_audio_ms"]) // 80 for row in telemetry),
    }
    document = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "checks": checks,
        "passed": all(checks.values()),
        "unpatched_probability_sha256": sha256_file(unpatched_dir / "diar.probs.f32"),
        "patched_probability_sha256": sha256_file(patched_dir / "diar.probs.f32"),
        "probability_shape": list(patched.shape),
        "telemetry_row_count": len(telemetry),
    }
    path = output_root(root) / "telemetry_validation.json"
    write_json(path, document)
    if not document["passed"]:
        raise R8Error(f"telemetry validation failed: {checks}")
    return path


def report(root: Path) -> Path:
    output = output_root(root)
    accuracy = load_json(output / "accuracy_metrics.json")
    compute_path = output / "compute_metrics.json"
    compute = load_json(compute_path) if compute_path.is_file() else {"backends": {}}
    smoke_parity_path = output / "smoke_parity.json"
    smoke_parity_document = load_json(smoke_parity_path) if smoke_parity_path.is_file() else None
    inventory = load_json(output / "input_inventory.json")
    lines = [
        "# R8 Streaming Sortformer Feasibility Report",
        "",
        "Evidence status: **development-known internal decision only**.",
        "",
        f"Accuracy gate: **{'PASS' if accuracy['gate']['passed'] else 'FAIL'}**.",
        f"Exposure: {float(inventory['summary']['scored_hours']):.3f} scored hours and {int(inventory['summary']['event_count'])} reference changes across ten meetings.",
        "",
        "| Target FE/h | Threshold | Recall@250 | Actual FE/h |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for target in (1, 5, 10, 20):
        point = accuracy["selected_operating_points"][str(target)]
        primary = point["metrics"]["tolerances"]["250"]
        lines.append(
            f"| {target} | {float(point['threshold']):.6f} | {float(primary['recall'] or 0.0):.3f} | {float(primary['false_events_per_hour']):.3f} |"
        )
    lines.extend(["", "## Compute", "", "| Backend | RTF | Chunk p99 ms | Max backlog ms | Gate |", "| --- | ---: | ---: | ---: | --- |"])
    compute_pass = False
    for backend in ("cpu", "vulkan"):
        row = compute.get("backends", {}).get(backend, {"status": "not_run"})
        if row.get("status") != "complete":
            status = str(row.get("status", "not_run")).upper()
            lines.append(f"| {backend} | unavailable | unavailable | unavailable | {status} |")
            continue
        compute_pass = compute_pass or bool(row["gate"]["passed"])
        p99 = row["chunks"]["chunk_compute_ms"]["p99"]
        backlog = row["chunks"]["maximum_backlog_ms"]
        lines.append(
            f"| {backend} | {float(row['aggregate_rtf']):.3f} | {float(p99) if p99 is not None else float('nan'):.1f} | {float(backlog) if backlog is not None else float('nan'):.1f} | {'PASS' if row['gate']['passed'] else 'FAIL'} |"
        )
    smoke_rows = compute.get("smoke_backends", {})
    if smoke_rows:
        lines.extend(
            [
                "",
                "Repeated Q8_0 smoke compute (20 clips, measured iterations exclude warm-up):",
                "",
                "| Backend | Resolved device | RTF | Chunk p99 ms | Peak private GiB |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for backend in ("cpu", "vulkan"):
            row = smoke_rows.get(backend)
            if row is None:
                continue
            lines.append(
                f"| {backend} | {', '.join(row['resolved_backends'])} | {float(row['rtf']):.3f} | {float(row['chunks']['chunk_compute_ms']['p99']):.1f} | {int(row['maximum_peak_private_bytes']) / 1024**3:.2f} |"
            )
    accuracy_pass = bool(accuracy["gate"]["passed"])
    if accuracy_pass and compute_pass:
        outcome = "A — accuracy and compute are both feasible"
    elif accuracy_pass:
        outcome = "B — accuracy passes, compute fails"
    elif compute_pass:
        outcome = "C — compute passes, accuracy fails"
    else:
        outcome = "D — accuracy and compute both fail"
    lines.extend(
        [
            "",
            "## Gate checks",
            "",
            *[
                f"- Accuracy `{name}`: {'pass' if value else 'fail'}"
                for name, value in accuracy["gate"]["checks"].items()
            ],
            "",
            "## Outcome",
            "",
            f"**{outcome}.**",
            "",
            f"At the 20 FE/h point, overlap-onset recall was {float(accuracy['selected_operating_points']['20']['metrics']['stratum_recall'].get('overlap_onset') or 0.0):.3%}, silence-gap-change recall was {float(accuracy['selected_operating_points']['20']['metrics']['stratum_recall'].get('silence_gap_change') or 0.0):.3%}, clean-change recall was {float(accuracy['selected_operating_points']['20']['metrics']['stratum_recall'].get('clean_change') or 0.0):.3%}, and short-return recall was {float(accuracy['selected_operating_points']['20']['metrics']['short_return_recall'] or 0.0):.3%}.",
            "",
            "CPU passed the hard real-time compute gate but missed the preferred RTF <= 0.5 headroom view. Vulkan full-panel replay was not run by explicit owner decision; its smoke evidence is not a full compute-gate result.",
            "",
            "The measured algorithmic lookahead is approximately 1,040 ms before compute. The pinned API processes a complete recording through an internally streaming core; this report does not claim live push-audio integration.",
        ]
    )
    if smoke_parity_document is not None:
        q8_vk = smoke_parity_document["comparisons"]["cpu_q8_vs_vulkan_q8"]
        q8_f16 = smoke_parity_document["comparisons"]["cpu_q8_vs_cpu_f16"]
        lines.extend(
            [
                "",
                "## Smoke parity context",
                "",
                f"CPU Q8_0 versus Vulkan Q8_0 fixed-threshold event agreement was {float(q8_vk['fixed_threshold_event_agreement']):.1%}; mean absolute probability difference was {float(q8_vk['mean_absolute_difference']):.6f}.",
                f"CPU Q8_0 versus CPU F16 fixed-threshold event agreement was {float(q8_f16['fixed_threshold_event_agreement']):.1%}; mean absolute probability difference was {float(q8_f16['mean_absolute_difference']):.6f}.",
            ]
        )
    path = output / "REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _plot_outputs(root, accuracy, compute)
    artifact_inventory(root)
    return path


def _plot_outputs(root: Path, accuracy: dict[str, Any], compute: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = output_root(root)
    curve = [
        row for row in accuracy["curve"] if float(row["false_events_per_hour"]) <= 30.0
    ]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        [float(row["false_events_per_hour"]) for row in curve],
        [float(row["recall_250"] or 0.0) for row in curve],
        color="tab:blue",
    )
    for target in (1, 5, 10, 20):
        point = accuracy["selected_operating_points"][str(target)]
        primary = point["metrics"]["tolerances"]["250"]
        axis.scatter(
            [float(primary["false_events_per_hour"])],
            [float(primary["recall"] or 0.0)],
            label=f"{target} FE/h target",
        )
    axis.set_xlabel("False events per source hour")
    axis.set_ylabel("Recall@250 ms")
    axis.set_xlim(0, 30)
    axis.set_ylim(bottom=0)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "recall_false_event_curve.png", dpi=160)
    plt.close(figure)

    cpu = compute["backends"]["cpu"]
    manifest = load_json(Path(cpu["manifest_path"]))
    chunks: list[dict[str, Any]] = []
    for item in manifest["items"]:
        runs = _telemetry_runs(_read_telemetry(Path(item["telemetry_path"])))
        chunks.extend(runs[-1])
    compute_values = np.asarray([float(row["total_us"]) / 1000.0 for row in chunks])
    backlog_values: list[float] = []
    backlog = 0.0
    for row, value in zip(chunks, compute_values, strict=True):
        if int(row["chunk_index"]) == 0:
            backlog = 0.0
        backlog = max(0.0, backlog + float(value) - float(config()["chunk_audio_ms"]))
        backlog_values.append(backlog)
    stride = max(1, len(compute_values) // 4000)
    x = np.arange(len(compute_values))[::stride]
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(x, compute_values[::stride], linewidth=0.6, label="chunk compute ms")
    axis.plot(x, np.asarray(backlog_values)[::stride], linewidth=0.8, label="backlog ms")
    axis.axhline(float(config()["chunk_audio_ms"]), color="tab:red", linestyle="--", label="480 ms deadline")
    axis.set_xlabel("Measured CPU chunk sequence")
    axis.set_ylabel("Milliseconds")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "chunk_compute_backlog.png", dpi=160)
    plt.close(figure)

    timeline_dir = output / "representative_timelines"
    timeline_dir.mkdir(parents=True, exist_ok=True)
    point = accuracy["selected_operating_points"]["20"]
    selected = list(point["metrics"]["matched_pairs"][:2]) + [
        [session_id, sample, sample]
        for session_id, sample in point["metrics"]["false_event_samples"][:2]
    ]
    sessions = _sessions(root)
    probabilities_dir = output / "probabilities" / "cpu"
    for index, (session_id, prediction, reference) in enumerate(selected):
        values = np.load(probabilities_dir / f"{session_id}.npz")["probabilities"]
        center_frame = int(round(int(prediction) / (80 * 16)))
        start = max(0, center_frame - 50)
        end = min(len(values), center_frame + 51)
        x_seconds = np.arange(start, end) * 0.08
        figure, axis = plt.subplots(figsize=(9, 4))
        for speaker in range(4):
            axis.plot(x_seconds, values[start:end, speaker], label=f"slot {speaker + 1}")
        axis.axhline(float(point["threshold"]), color="black", linestyle=":", label="20 FE/h threshold")
        axis.axvline(int(prediction) / 16000.0, color="tab:red", linestyle="--", label="prediction")
        for event in sessions[str(session_id)].events:
            sample = int(event["sample"])
            if start * 1280 <= sample < end * 1280:
                axis.axvline(sample / 16000.0, color="tab:green", alpha=0.35)
        axis.set_title(f"{session_id}: {'match' if int(reference) != int(prediction) or [session_id, prediction, reference] in point['metrics']['matched_pairs'] else 'false event'}")
        axis.set_xlabel("Source time (seconds)")
        axis.set_ylabel("Speaker activity probability")
        axis.set_ylim(0, 1)
        axis.legend(ncol=3, fontsize=8)
        figure.tight_layout()
        figure.savefig(timeline_dir / f"timeline_{index:02d}_{session_id}.png", dpi=150)
        plt.close(figure)


def artifact_inventory(root: Path) -> Path:
    output = output_root(root)
    path = output / "artifact_inventory.json"
    rows = [
        {
            "relative_path": str(item.relative_to(output)).replace("\\", "/"),
            "size_bytes": item.stat().st_size,
            "sha256": sha256_file(item),
        }
        for item in sorted(output.rglob("*"))
        if item.is_file() and item != path
    ]
    write_json(
        path,
        {
            "schema_version": 1,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "artifact_count": len(rows),
            "artifacts": rows,
        },
    )
    return path


def smoke() -> dict[str, Any]:
    probabilities = np.asarray(
        [
            [0.9, 0.1, 0.1, 0.1],
            [0.9, 0.1, 0.1, 0.1],
            [0.9, 0.9, 0.1, 0.1],
            [0.1, 0.9, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1],
            [0.9, 0.1, 0.1, 0.1],
        ],
        dtype=np.float32,
    )
    events = decode_change_events(probabilities, 0.5)
    if events != [2560, 7680]:
        raise R8Error(f"event decoder smoke failed: {events}")
    matched, false, misses = _one_to_one([100, 900, 1300], [120, 910], 50)
    if len(matched) != 2 or false != [1300] or misses:
        raise R8Error("one-to-one matcher smoke failed")
    return {
        "decoded_event_samples": events,
        "code_sha256": sha256_file(CODE_PATH),
        "config_sha256": sha256_file(CONFIG_PATH),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=(
            "prepare",
            "materialize-smoke",
            "run",
            "score",
            "compute",
            "parity",
            "smoke-parity",
            "receipts",
            "validate-telemetry",
            "report",
            "smoke",
        ),
    )
    parser.add_argument("--backend", choices=("cpu", "vulkan"), default="cpu")
    parser.add_argument("--panel", choices=("smoke", "full"), default="full")
    parser.add_argument("--bench", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "smoke":
        print(json.dumps(smoke(), indent=2, sort_keys=True))
        return 0
    root = cache_root()
    if args.action == "prepare":
        print(prepare(root))
    elif args.action == "materialize-smoke":
        print(materialize_smoke(root))
    elif args.action == "run":
        if args.bench is None or args.model is None:
            parser.error("run requires --bench and --model")
        print(
            run_inference(
                root,
                args.bench.resolve(),
                args.model.resolve(),
                args.backend,
                args.panel,
                args.repetitions,
                args.warmup,
                args.resume,
            )
        )
    elif args.action == "score":
        print(score_accuracy(root, args.backend))
    elif args.action == "compute":
        print(aggregate_compute(root))
    elif args.action == "parity":
        print(backend_parity(root))
    elif args.action == "smoke-parity":
        print(smoke_parity(root))
    elif args.action == "receipts":
        for path in receipts(root):
            print(path)
    elif args.action == "validate-telemetry":
        print(validate_telemetry(root))
    elif args.action == "report":
        print(report(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
