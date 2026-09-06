from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SAMPLING_ROW_COUNT = 4096
FRAME_SAMPLES = 1280
SAMPLE_RATE_HZ = 16000
CHUNK_SAMPLES = 480000
CHUNK_FRAMES = 375
FRAME_SECONDS = 0.08
GT_CONFIRMATION_SAMPLES = 8000
GT_ENROLLMENT_SAMPLES = 3200
GT_SILENCE_RESET_SAMPLES = 19200
OPTIMIZER_LR = 1e-4
EQUIVALENCE_TOL = 1e-5
IDENTITY_TOL = 1e-6
REQUIRED_HORIZONS = (100, 300, 500)


class MaterialError(RuntimeError):
    pass


class MaterialBlockedError(MaterialError):
    pass


@dataclass(slots=True)
class ResolvedMaterialInputs:
    checkpoint_path: Path
    nemo_checkout: Path
    dependency_lock: Path
    corpus_root: Path
    reference_root: Path
    sampling_manifest: Path
    sampling_sha256: str
    device: str
    ami_source: str
    alimeeting_source: str
    rows_by_source: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_sampling_population(sampling_manifest: Path) -> dict[str, Any]:
    if not sampling_manifest.is_file():
        raise MaterialError(f"sampling manifest missing: {sampling_manifest}")
    rows: list[dict[str, Any]] = []
    for line in sampling_manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise MaterialError("sampling manifest rows must be JSON objects")
            rows.append(value)
    if len(rows) != SAMPLING_ROW_COUNT:
        raise MaterialError(f"sampling manifest has {len(rows)} rows, expected 4096")
    for row in rows:
        if (
            row.get("split_role") != "PSEM-STRATEGY-TRAIN"
            or not isinstance(row.get("source_id"), str)
            or not isinstance(row.get("window_start_sample"), int)
            or not isinstance(row.get("window_end_sample"), int)
            or row["window_end_sample"] != row["window_start_sample"] + CHUNK_SAMPLES
        ):
            raise MaterialError("sampling manifest row identity is invalid")
    rows_by_source: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_source.setdefault(row["source_id"], []).append(row)
    corpora: dict[str, str] = {}
    for row in rows:
        corpus = row.get("corpus")
        if corpus in ("AMI", "AliMeeting"):
            corpora.setdefault(row["source_id"], corpus)
    ami = sorted(s for s, c in corpora.items() if c == "AMI")
    ali = sorted(s for s, c in corpora.items() if c == "AliMeeting")
    if not ami or not ali:
        raise MaterialError("sampling manifest lacks AMI and AliMeeting TRAIN coverage")
    return {
        "rows_by_source": rows_by_source,
        "ami_source": max(ami, key=lambda s: len(rows_by_source[s])),
        "alimeeting_source": max(ali, key=lambda s: len(rows_by_source[s])),
        "sampling_sha256": _sha256_file(sampling_manifest),
    }


def load_source_rows(source_manifest_path: Path) -> dict[str, dict[str, Any]]:
    return {
        json.loads(line)["source_id"]: json.loads(line)
        for line in source_manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def load_source_components(sessions: Any, source_manifest_path: Path) -> dict[str, str]:
    source_rows = load_source_rows(source_manifest_path)
    components: dict[str, str] = {}
    for source_id, session in sessions.items():
        row = source_rows.get(source_id, {})
        components[source_id] = (
            str(row.get("recording_group_id") or row.get("meeting_series") or source_id)
        )
    return components


def resolve_material_inputs(
    checkpoint_path: Path,
    nemo_checkout: Path,
    dependency_lock: Path,
    corpus_root: Path,
    reference_root: Path,
    sampling_manifest: Path,
    nemo_sha256: str,
    device: str = "cuda",
) -> ResolvedMaterialInputs:
    if not checkpoint_path.is_file():
        raise MaterialError(f"checkpoint missing: {checkpoint_path}")
    if _sha256_file(checkpoint_path) != nemo_sha256:
        raise MaterialError("checkpoint identity differs from the frozen .nemo")
    if not nemo_checkout.is_dir():
        raise MaterialError(f"NeMo checkout missing: {nemo_checkout}")
    if not dependency_lock.is_file():
        raise MaterialError(f"dependency lock missing: {dependency_lock}")
    if not corpus_root.is_dir():
        raise MaterialError(f"corpus root missing: {corpus_root}")
    if not reference_root.is_dir():
        raise MaterialError(f"reference root missing: {reference_root}")
    population = resolve_sampling_population(sampling_manifest)
    return ResolvedMaterialInputs(
        checkpoint_path=checkpoint_path,
        nemo_checkout=nemo_checkout,
        dependency_lock=dependency_lock,
        corpus_root=corpus_root,
        reference_root=reference_root,
        sampling_manifest=sampling_manifest,
        sampling_sha256=str(population["sampling_sha256"]),
        device=device,
        ami_source=str(population["ami_source"]),
        alimeeting_source=str(population["alimeeting_source"]),
        rows_by_source=dict(population["rows_by_source"]),
    )


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise MaterialBlockedError("material execution requires torch on the GPU worker") from exc
    return torch


def _detach_state(torch: Any, state: Any) -> Any:
    if isinstance(state, torch.Tensor):
        return state.detach().clone()
    if isinstance(state, tuple) and hasattr(state, "_fields"):
        return type(state)(*(_detach_state(torch, v) for v in state))
    if isinstance(state, tuple):
        return tuple(_detach_state(torch, v) for v in state)
    if isinstance(state, list):
        return [_detach_state(torch, v) for v in state]
    if isinstance(state, dict):
        return {k: _detach_state(torch, v) for k, v in state.items()}
    return state


def _max_abs_diff(torch: Any, first: Any, second: Any) -> float:
    return float((first - second).abs().max())


def _frontier_point(metrics: Any, threshold: float = 0.5) -> Any:
    from experiments.psem_state_corrected_adaptation_gate import frontier as frontier_mod
    active_hours = float(metrics["active_speech_seconds"]) / 3600.0
    if active_hours <= 0:
        raise MaterialError("DEV scorer produced no active speech")
    reference_count = int(metrics["reference_replacement_count"])
    if reference_count <= 0:
        raise MaterialError("DEV scorer produced no reference replacements")
    return frontier_mod.FrontierPoint(
        threshold=float(threshold),
        false_cuts_per_hour=float(metrics["false_cut_count"]) / active_hours,
        contamination=float(
            metrics["exclusive_other_contamination_seconds_per_active_speech_hour"]
        ),
        miss_rate=float(metrics["missed_replacement_count"]) / float(reference_count),
    )


def is_dev_family_session(session: Any, family: str) -> bool:
    return session.source_family == family and "dev" in str(
        getattr(session, "role", "")
    ).lower()

def plan_windows(total_frames: int, window_frames: int = 375) -> list[tuple[int, int]]:
    if total_frames < 2 * window_frames or window_frames <= 0:
        raise MaterialError("source is shorter than two adjacent chunks")
    return [
        (start, min(start + window_frames, total_frames))
        for start in range(0, total_frames, window_frames)
    ]


def microbatch_plan(
    total_frames: int, window_frames: int = 375, accumulation: int = 16
) -> list[dict[str, Any]]:
    windows = plan_windows(total_frames, window_frames)[:2]
    if accumulation % len(windows) != 0:
        raise MaterialError("accumulation must split evenly across windows")
    per_window = accumulation // len(windows)
    plan: list[dict[str, Any]] = []
    for window_index, (window_start, window_end) in enumerate(windows):
        span = window_end - window_start
        base, extra = divmod(span, per_window)
        if base <= 0:
            raise MaterialError("window is too short for the accumulation contract")
        cursor = window_start
        for index in range(per_window):
            length = base + (1 if index < extra else 0)
            plan.append(
                {
                    "start": cursor,
                    "end": cursor + length,
                    "detach_state": index == 0,
                    "window": window_index,
                }
            )
            cursor += length
    return plan


DEFAULT_WORKER_CAP = 24


def resolve_worker_count(requested: int | None) -> int:
    cpus = os.cpu_count() or 1
    cap = max(1, min(DEFAULT_WORKER_CAP, cpus))
    if requested is not None and requested >= 1:
        return max(1, min(int(requested), cap))
    return cap


def _ordered_pool_map(
    worker_fn: Any,
    payloads: list[dict[str, Any]],
    workers: int,
    on_result: Any = None,
) -> list[Any]:
    if workers <= 1 or len(payloads) <= 1:
        results = [worker_fn(payload) for payload in payloads]
        if on_result is not None:
            for result in results:
                on_result(result)
        return results
    import concurrent.futures
    import multiprocessing

    _init_cpu_worker()
    context = multiprocessing.get_context("spawn")
    limit = max(1, min(workers, len(payloads)))
    ordered: list[Any] = [None] * len(payloads)
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_init_cpu_worker,
    ) as pool:
        pending: dict[Any, int] = {}
        queued = iter(enumerate(payloads))
        for _ in range(limit):
            try:
                index, payload = next(queued)
            except StopIteration:
                break
            pending[pool.submit(worker_fn, payload)] = index
        while pending:
            done, _ = concurrent.futures.wait(
                pending, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                index = pending.pop(future)
                result = future.result()
                if on_result is not None:
                    on_result(result)
                    ordered[index] = True
                else:
                    ordered[index] = result
                try:
                    position, payload = next(queued)
                except StopIteration:
                    continue
                pending[pool.submit(worker_fn, payload)] = position
    return ordered


def _build_source_targets_worker(payload: dict[str, Any]) -> dict[str, Any]:
    from experiments.psem_frozen_ceiling_gate.experiment_support import (
        simulate_gt_session,
    )
    result = build_source_targets(
        simulate_gt_session,
        str(payload["source_id"]),
        payload["labels"],
        list(payload["rows"]),
        int(payload["num_frames"]),
    )
    return {
        "source_id": str(payload["source_id"]),
        "authority": result["authority"],
        "multiplicity": list(result["multiplicity"]),
        "episode_ids": list(result["episode_ids"]),
        "intervals": [dict(row) for row in result["intervals"]],
    }


def build_target_payloads(
    sessions: Any,
    rows_by_source: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for source_id in sorted(sessions.keys()):
        session = sessions[source_id]
        rows = list(rows_by_source.get(source_id, []))
        label_end = int(session.labels.intervals[-1].end_sample)
        crop_end = max([r["window_end_sample"] for r in rows] + [label_end])
        payloads.append(
            {
                "source_id": source_id,
                "labels": session.labels,
                "rows": rows,
                "num_frames": canonical_frames(label_end, crop_end),
            }
        )
    return payloads


def build_all_source_targets(
    sessions: Any,
    rows_by_source: dict[str, list[dict[str, Any]]],
    workers: int,
) -> dict[str, dict[str, Any]]:
    results = _ordered_pool_map(
        _build_source_targets_worker,
        build_target_payloads(sessions, rows_by_source),
        workers,
    )
    return {str(item["source_id"]): item for item in results}


def _sweep_points(
    dev: Any, scores: list[float], thresholds: list[float], horizon_ms: int
) -> list[Any]:
    from experiments.psem_frozen_ceiling_gate.evaluate_ceiling import session_metrics
    from experiments.psem_frozen_ceiling_gate.experiment_support import ReplacementEvent
    from experiments.psem_state_corrected_adaptation_gate import frontier_sweep as sweep_mod

    ordered = [float(t) for t in thresholds]
    grid, keys = sweep_mod.sweep_threshold_events(
        dev, [float(s) for s in scores], int(horizon_ms)
    )
    cache: dict[tuple[Any, ...], Any] = {}
    by_threshold: dict[float, Any] = {}
    requested = set(ordered)
    for threshold, key in zip(grid, keys):
        if threshold not in requested:
            continue
        if key not in cache:
            events = tuple(
                ReplacementEvent(
                    source_id=event[0],
                    anchor_episode_id=event[1],
                    anchor_id=event[2],
                    boundary_source_sample=event[3],
                    model_evidence_frontier_sample=event[4],
                    decoder_emit_sample=event[5],
                    compute_lag_ms=None,
                    confirmation_samples=event[6],
                )
                for event in key
            )
            cache[key] = session_metrics(dev, events)
        by_threshold[threshold] = _frontier_point(cache[key], float(threshold))
    missing = [t for t in ordered if t not in by_threshold]
    if missing:
        raise MaterialError("requested threshold outside sweep grid")
    return [by_threshold[t] for t in ordered]

def _frontier_threshold_chunk(payload: dict[str, Any]) -> list[Any]:
    return _sweep_points(
        payload["dev"],
        payload["scores"],
        payload["thresholds"],
        int(payload["horizon_ms"]),
    )


def _frontier_horizon_task(payload: dict[str, Any]) -> list[Any]:
    return _sweep_points(
        payload["dev"],
        payload["scores"],
        payload["thresholds"],
        int(payload["horizon_ms"]),
    )


def candidate_frontier_points(
    dev: Any,
    scores: list[float],
    thresholds: list[float],
    horizon_ms: int,
    workers: int,
) -> list[Any]:
    return _sweep_points(dev, scores, thresholds, horizon_ms)

WORKER_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _init_cpu_worker() -> None:
    import os

    for variable in WORKER_THREAD_VARS:
        os.environ[variable] = "1"



def candidate_frontier_points_multi(
    dev: Any,
    scores: list[float],
    thresholds: list[float],
    horizons_ms: list[int],
    workers: int,
) -> dict[int, list[Any]]:
    import concurrent.futures
    import multiprocessing

    ordered = [float(t) for t in thresholds]
    score_list = [float(s) for s in scores]
    horizons = [int(h) for h in horizons_ms]
    if workers <= 1 or len(ordered) <= 1:
        return {
            horizon_ms: _frontier_threshold_chunk(
                {
                    "dev": dev,
                    "scores": score_list,
                    "thresholds": ordered,
                    "horizon_ms": horizon_ms,
                }
            )
            for horizon_ms in horizons
        }
    _init_cpu_worker()
    context = multiprocessing.get_context("spawn")
    tasks = [
        {
            "dev": dev,
            "scores": score_list,
            "thresholds": ordered,
            "horizon_ms": horizon_ms,
        }
        for horizon_ms in horizons
    ]
    results: dict[int, list[Any]] = {}
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_init_cpu_worker,
    ) as pool:
        merged = list(pool.map(_frontier_horizon_task, tasks))
    for horizon_ms, points in zip(horizons, merged):
        print(
            f"[gate0] frontier horizon={horizon_ms} thresholds={len(ordered)} "
            f"workers={workers}",
            flush=True,
        )
        results[horizon_ms] = points
    return results


def candidate_frontier_points_sessions(
    jobs: list[dict[str, Any]], workers: int
) -> dict[str, dict[int, list[Any]]]:
    import concurrent.futures
    import multiprocessing

    ordered_jobs: list[dict[str, Any]] = []
    for job in jobs:
        ordered_jobs.append(
            {
                "key": str(job["key"]),
                "dev": job["dev"],
                "scores": [float(s) for s in job["scores"]],
                "thresholds": [float(t) for t in job["thresholds"]],
                "horizons": [int(h) for h in job["horizons"]],
            }
        )
    if workers <= 1:
        results: dict[str, dict[int, list[Any]]] = {}
        for job in ordered_jobs:
            points = candidate_frontier_points_multi(
                job["dev"], job["scores"], job["thresholds"], job["horizons"], 1
            )
            results[job["key"]] = points
        return results
    _init_cpu_worker()
    tasks: list[dict[str, Any]] = []
    order: list[tuple[str, int]] = []
    for job in ordered_jobs:
        for horizon_ms in job["horizons"]:
            tasks.append(
                {
                    "key": job["key"],
                    "dev": job["dev"],
                    "scores": job["scores"],
                    "thresholds": job["thresholds"],
                    "horizon_ms": horizon_ms,
                }
            )
            order.append((job["key"], horizon_ms))
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_init_cpu_worker,
    ) as pool:
        merged = list(pool.map(_frontier_horizon_task, tasks))
    grouped: dict[str, dict[int, list[Any]]] = {}
    for (key, horizon_ms), points in zip(order, merged):
        print(
            f"[gate0] frontier session={key} horizon={horizon_ms} "
            f"workers={workers}",
            flush=True,
        )
        grouped.setdefault(key, {})[horizon_ms] = points
    return grouped


class ClassAccumulator:
    def __init__(self) -> None:
        self.replace_pos = 0.0
        self.replace_neg = 0.0
        self.anchor_pos = 0.0
        self.anchor_neg = 0.0

    def add(
        self,
        y_replace: list[float],
        y_anchor: list[float],
        multiplicity: list[int],
        valid: list[bool],
    ) -> None:
        if not (len(y_replace) == len(y_anchor) == len(multiplicity) == len(valid)):
            raise MaterialError("target/multiplicity geometry differs")
        for yr, ya, m, v in zip(y_replace, y_anchor, multiplicity, valid):
            if not v or m <= 0:
                continue
            if yr > 0:
                self.replace_pos += m
            else:
                self.replace_neg += m
            if ya > 0:
                self.anchor_pos += m
            else:
                self.anchor_neg += m

    def weights(self) -> dict[str, float]:
        if min(self.replace_pos, self.replace_neg, self.anchor_pos, self.anchor_neg) <= 0:
            raise MaterialError("partition lacks positive/negative replacement/anchor support")
        return {
            "replacement_positive_weight": self.replace_neg / self.replace_pos,
            "anchor_positive_weight": self.anchor_neg / self.anchor_pos,
        }


def audit_module_modes(
    wrapper_training: bool,
    dropout_training: list[bool],
    head_training: bool,
    wrapper_trainable: list[str],
    head_trainable: list[str],
) -> dict[str, Any]:
    if wrapper_training:
        raise MaterialError("frozen wrapper is not in eval mode")
    if any(dropout_training):
        raise MaterialError("frozen dropout/randomized attention is in train mode")
    if not head_training:
        raise MaterialError("residual head is not in train mode")
    if wrapper_trainable:
        raise MaterialError("frozen backbone exposes trainable parameters")
    if not head_trainable:
        raise MaterialError("residual head exposes no trainable parameters")
    return {
        "sortformer_eval": not wrapper_training,
        "dropout_eval_count": len(dropout_training),
        "psem_head_train": head_training,
        "frozen_trainable_count": 0,
        "head_trainable_count": len(head_trainable),
        "frozen_representation_ok": True,
    }


def oracle_slot_mapping(
    episode_ids: list[str | None],
    anchor_active: list[bool],
    valid: list[bool],
    probabilities: list[list[float]],
) -> tuple[dict[str, int], list[dict[str, Any]], list[int]]:
    if not (len(episode_ids) == len(anchor_active) == len(valid) == len(probabilities)):
        raise MaterialError("mapping geometry differs")
    slot_of: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for episode_id in sorted({e for e in episode_ids if e is not None}):
        support = [
            i
            for i, (e, a, v) in enumerate(zip(episode_ids, anchor_active, valid))
            if e == episode_id and a and v
        ]
        if not support:
            rows.append({"anchor_episode_id": episode_id, "status": "unmapped"})
            continue
        slot_count = len(probabilities[support[0]])
        means = [0.0] * slot_count
        for i in support:
            for s in range(slot_count):
                means[s] += probabilities[i][s]
        means = [v / len(support) for v in means]
        slot = max(range(slot_count), key=lambda s: means[s])
        slot_of[episode_id] = slot
        rows.append(
            {
                "anchor_episode_id": episode_id,
                "status": "mapped",
                "slot_index": slot,
                "support_frame_count": len(support),
            }
        )
    unmapped_frames = [
        i
        for i, e in enumerate(episode_ids)
        if e is None or e not in slot_of
    ]
    return slot_of, rows, unmapped_frames


def canonical_frames(label_end_sample: int, crop_end_sample: int) -> int:
    if label_end_sample <= 0:
        raise MaterialError("label timeline is empty")
    return max(label_end_sample, crop_end_sample) // FRAME_SAMPLES


def slice_waveform_frames(
    total_samples: int, frame_count: int, source_id: str
) -> tuple[int, int]:
    usable = frame_count * FRAME_SAMPLES
    if total_samples < usable:
        raise MaterialError(f"waveform is shorter than the authority: {source_id}")
    return usable, total_samples - usable


def require_frame_alignment(emitted: int, authority_frames: int, source_id: str) -> bool:
    if emitted != authority_frames:
        raise MaterialError(f"evidence/authority frame counts differ: {source_id}")
    return True


def require_frame_vector(shape: tuple[int, ...], name: str) -> bool:
    if len(shape) != 2:
        raise MaterialError(f"{name} is not a framewise [batch, time] tensor")
    return True


def select_fit_slice(
    fit: list[str], rows_by_source: dict[str, list], corpus_of: dict[str, str]
) -> tuple[str, str]:
    ami = sorted(
        (s for s in fit if corpus_of.get(s) == "AMI"),
        key=lambda s: (-len(rows_by_source.get(s, [])), s),
    )
    ali = sorted(
        (s for s in fit if corpus_of.get(s) == "AliMeeting"),
        key=lambda s: (-len(rows_by_source.get(s, [])), s),
    )
    if not ami or not ali:
        raise MaterialError("TRAIN-FIT lacks AMI and AliMeeting slice candidates")
    return ami[0], ali[0]


def mask_calibration(
    targets: list[float], valid: list[bool], mapped: list[bool]
) -> tuple[list[int], dict[str, int]]:
    if not (len(targets) == len(valid) == len(mapped)):
        raise MaterialError("calibration mask geometry differs")
    kept = [i for i, (v, m) in enumerate(zip(valid, mapped)) if v and m]
    if not kept:
        raise MaterialError("calibration has no valid mapped frames")
    coverage = {
        "frames": len(targets),
        "kept": len(kept),
        "positive": sum(1 for i in kept if targets[i] > 0),
    }
    coverage["negative"] = coverage["kept"] - coverage["positive"]
    if coverage["positive"] <= 0 or coverage["negative"] <= 0:
        raise MaterialError("calibration lacks positive/negative support")
    return kept, coverage


def extend_calibration_buffers(
    buffers: dict[str, list[float]],
    f0_all: list[float],
    cand_all: list[float],
    targets_all: list[float],
    kept: list[int],
) -> dict[str, int]:
    if not (len(f0_all) == len(cand_all) == len(targets_all)):
        raise MaterialError("calibration arm geometry differs")
    for i in kept:
        buffers["f0"].append(f0_all[i])
        buffers["cand"].append(cand_all[i])
        buffers["targets"].append(targets_all[i])
    if not (len(buffers["f0"]) == len(buffers["cand"]) == len(buffers["targets"])):
        raise MaterialError("calibration arms diverge")
    if not buffers["f0"]:
        raise MaterialError("TRAIN-CALIB produced no valid mapped frames")
    return {"kept": len(kept)}


def validate_gate0_record(record: dict[str, Any]) -> bool:
    for key in ("verdict", "mode", "checks", "evidence"):
        if key not in record:
            raise MaterialError(f"gate-0 record lacks {key}")
    if record["mode"] != "material" or record["verdict"] != "PASS":
        raise MaterialError("gate-0 record is not a material PASS")
    evidence = record["evidence"]
    for key in (
        "checkpoint_sha256",
        "sampling_sha256",
        "calibration_candidate",
        "calibration_f0",
        "dev",
        "profiler",
        "predictions",
        "partition",
    ):
        if key not in evidence:
            raise MaterialError(f"gate-0 evidence lacks {key}")
    for name in ("calibration_candidate", "calibration_f0"):
        for key in ("slope", "intercept", "nll", "brier", "raw_nll"):
            if key not in evidence[name]:
                raise MaterialError(f"calibration evidence lacks {key}")
        if not evidence[name]["slope"] > 0:
            raise MaterialError("calibration slope is not positive")
    for source_id, horizons in evidence["dev"].items():
        for horizon_ms, result in horizons.items():
            for key in ("budget", "points", "c_envelope", "m_envelope", "raw_ap", "mapping_mapped"):
                if key not in result:
                    raise MaterialError(f"DEV frontier evidence lacks {key}")
            if not result["points"]:
                raise MaterialError("DEV frontier has no exact points")
    profiler = evidence["profiler"]
    for key in ("optimizer_steps", "seconds_per_step", "peak_vram_bytes", "dev_infer_seconds"):
        if key not in profiler:
            raise MaterialError(f"profiler evidence lacks {key}")
    if not 8 <= profiler["optimizer_steps"] <= 16:
        raise MaterialError("profiler did not run 8-16 real optimizer steps")
    return True


def init_source_state(torch: Any, wrapper: Any, batch_size: int = 1) -> Any:
    sortformer = wrapper.sortformer
    return sortformer.sortformer_modules.init_streaming_state(
        batch_size=batch_size,
        async_streaming=sortformer.async_streaming,
        device=sortformer.device,
    )


def prepare_streaming(torch: Any, wrapper: Any, waveform: Any) -> dict[str, Any]:
    sortformer = wrapper.sortformer
    lengths = torch.tensor([waveform.shape[1]], dtype=torch.long, device=waveform.device)
    processed, processed_lengths = sortformer.process_signal(
        audio_signal=waveform, audio_signal_length=lengths
    )
    processed = processed[:, :, : int(processed_lengths.max())]
    offsets = torch.zeros((processed.shape[0],), dtype=torch.long, device=sortformer.device)
    loader = sortformer.sortformer_modules.streaming_feat_loader(
        feat_seq=processed,
        feat_seq_length=processed_lengths,
        feat_seq_offset=offsets,
    )
    return {"loader": loader, "device": sortformer.device}


def advance_streaming_state(
    torch: Any, wrapper: Any, state: Any, chunk: Any, chunk_lengths: Any, left_offset: int,
    right_offset: int,
) -> tuple[Any, Any, Any, Any, dict[str, int]]:
    return wrapper._streaming_step(
        chunk, chunk_lengths, state, left_offset=left_offset, right_offset=right_offset
    )


def run_adjacent_windows(
    torch: Any,
    wrapper: Any,
    waveform: Any,
    window_frames: int = 375,
    detach_between: bool = True,
) -> dict[str, Any]:
    prepared = prepare_streaming(torch, wrapper, waveform)
    state = init_source_state(torch, wrapper)
    windows: list[dict[str, Any]] = []
    current: dict[str, list[Any]] = {"hidden": [], "logits": [], "probabilities": []}
    emitted_in_window = 0
    total_emitted = 0
    boundary_steps: list[int] = []
    step_index = 0
    with torch.no_grad():
        for _, chunk, chunk_lengths, left_offset, right_offset in prepared["loader"]:
            state, hidden, logits, probabilities, trace = advance_streaming_state(
                torch, wrapper, state, chunk, chunk_lengths, left_offset, right_offset
            )
            current["hidden"].append(hidden)
            current["logits"].append(logits)
            current["probabilities"].append(probabilities)
            emitted_in_window += int(hidden.shape[1])
            total_emitted += int(hidden.shape[1])
            step_index += 1
            if emitted_in_window >= window_frames:
                windows.append(
                    {
                        "hidden": torch.cat(current["hidden"], dim=1),
                        "logits": torch.cat(current["logits"], dim=1),
                        "probabilities": torch.cat(current["probabilities"], dim=1),
                        "emitted_frames": emitted_in_window,
                        "steps": step_index,
                    }
                )
                boundary_steps.append(step_index)
                current = {"hidden": [], "logits": [], "probabilities": []}
                emitted_in_window = 0
                if detach_between:
                    state = _detach_state(torch, state)
    if current["hidden"]:
        windows.append(
            {
                "hidden": torch.cat(current["hidden"], dim=1),
                "logits": torch.cat(current["logits"], dim=1),
                "probabilities": torch.cat(current["probabilities"], dim=1),
                "emitted_frames": emitted_in_window,
                "steps": step_index,
            }
        )
    return {"windows": windows, "state_out": state, "boundary_steps": boundary_steps}


def concat_windows(torch: Any, windows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "hidden": torch.cat([w["hidden"] for w in windows], dim=1),
        "logits": torch.cat([w["logits"] for w in windows], dim=1),
        "probabilities": torch.cat([w["probabilities"] for w in windows], dim=1),
        "emitted_frames": sum(w["emitted_frames"] for w in windows),
    }


def full_source_intervals(labels: Any) -> list[dict[str, Any]]:
    intervals = []
    for interval, activity in zip(labels.intervals, labels.activity_labels, strict=True):
        intervals.append(
            {
                "start_sample": int(interval.start_sample),
                "end_sample": int(interval.end_sample),
                "active_speakers": list(interval.active_speakers),
                "masked": bool(
                    activity.get("mask_state") != "valid"
                    or interval.ambiguous
                    or not interval.speaker_identity_known
                ),
            }
        )
    return intervals


def full_source_frame_labels(
    intervals: list[dict[str, Any]], frame_count: int
) -> tuple[list[tuple[str, ...]], list[bool]]:
    active: list[tuple[str, ...]] = [() for _ in range(frame_count)]
    valid: list[bool] = [False for _ in range(frame_count)]
    index = 0
    for frame in range(frame_count):
        center = frame * FRAME_SAMPLES + FRAME_SAMPLES // 2
        while index < len(intervals) and intervals[index]["end_sample"] <= center:
            index += 1
        if index < len(intervals):
            row = intervals[index]
            if row["start_sample"] <= center < row["end_sample"]:
                valid[frame] = not row["masked"]
                active[frame] = tuple(row["active_speakers"]) if valid[frame] else ()
    return active, valid


def build_source_targets(
    simulate: Any, source_id: str, labels: Any, rows: list[dict[str, Any]], num_frames: int
) -> dict[str, Any]:
    from experiments.psem_state_corrected_adaptation_gate import lifecycle as lifecycle_mod
    from experiments.psem_state_corrected_adaptation_gate import multiplicity as multiplicity_mod
    from experiments.psem_sortformer_adaptation_depth.frame_alignment import (
        native_episode_timeline,
        native_frame_coordinates,
    )
    intervals = full_source_intervals(labels)
    reference = simulate(
        {"source_id": source_id, "intervals": intervals},
        replacement_confirmation_samples=GT_CONFIRMATION_SAMPLES,
        enrollment_samples=GT_ENROLLMENT_SAMPLES,
        silence_reset_samples=GT_SILENCE_RESET_SAMPLES,
    )
    episode_ids = native_episode_timeline(reference, num_frames)
    active_by_frame, valid_by_frame = full_source_frame_labels(intervals, num_frames)
    starts, ends = native_frame_coordinates(num_frames)
    centers = starts + (ends - starts) // 2
    episodes = []
    for episode in reference.episodes:
        covered = [
            f
            for f in range(num_frames)
            if episode.anchor_emit_sample <= int(centers[f]) < episode.end_emit_sample
        ]
        if covered:
            episodes.append(
                lifecycle_mod.AnchorEpisode(
                    str(episode.episode_id),
                    str(episode.anchor_speaker),
                    covered[0],
                    covered[-1] + 1,
                )
            )
    authority = lifecycle_mod.build_source_authority(
        source_id, num_frames, episodes, active_by_frame, valid_by_frame
    )
    crops = [
        (row["window_start_sample"] / SAMPLE_RATE_HZ, row["window_end_sample"] / SAMPLE_RATE_HZ)
        for row in rows
    ]
    multiplicity = multiplicity_mod.build_multiplicity(num_frames, crops, authority.valid)
    return {
        "authority": authority,
        "multiplicity": multiplicity,
        "episode_ids": episode_ids,
        "intervals": intervals,
    }


def infer_arm_logits(
    torch: Any,
    wrapper: Any,
    head_module: Any,
    waveform: Any,
    episode_ids: list,
    anchor_active: list[bool],
    valid: list[bool],
    frame_count: int,
    device: Any,
) -> dict[str, Any]:
    passage = run_adjacent_windows(torch, wrapper, waveform, 1 << 30, False)
    evidence = concat_windows(torch, passage["windows"])
    require_frame_alignment(evidence["hidden"].shape[1], frame_count, "inference")
    probabilities = evidence["probabilities"][0].detach().cpu().tolist()
    slot_of, mapping_rows, unmapped_frames = oracle_slot_mapping(
        list(episode_ids), list(anchor_active), list(valid), probabilities
    )
    one_hot = torch.zeros((1, frame_count, 4), dtype=torch.float32, device=device)
    for frame, episode_id in enumerate(episode_ids):
        if episode_id is not None and episode_id in slot_of:
            one_hot[0, frame, slot_of[episode_id]] = 1.0
    selected_logit = (evidence["logits"] * one_hot).sum(dim=-1, keepdim=True).squeeze(-1)
    require_frame_vector(tuple(selected_logit.shape), "selected anchor logit")
    anchor_mask = one_hot.bool()
    neg_inf = torch.full_like(evidence["logits"], float("-inf"))
    non_anchor = torch.where(anchor_mask, neg_inf, evidence["logits"])
    best_non_anchor = non_anchor.max(dim=-1, keepdim=True).values
    best_non_anchor = torch.where(
        torch.isfinite(best_non_anchor), best_non_anchor, torch.zeros_like(best_non_anchor)
    )
    delay = torch.full_like(selected_logit.unsqueeze(-1), 1.04)
    features = torch.cat(
        [
            evidence["hidden"],
            evidence["logits"],
            selected_logit.unsqueeze(-1),
            best_non_anchor,
            delay,
        ],
        dim=-1,
    )
    gru_state: Any = None
    anchor_parts: list[Any] = []
    resid_parts: list[Any] = []
    cursor = 0
    while cursor < frame_count:
        piece = features[:, cursor : cursor + CHUNK_FRAMES]
        outputs, gru_state = head_module(piece, gru_state)
        anchor_parts.append(outputs["anchor_logit"])
        resid_parts.append(outputs["z_residual"])
        cursor += CHUNK_FRAMES
        if cursor < frame_count:
            gru_state = _detach_state(torch, gru_state)
    anchor_logit = torch.cat(anchor_parts, dim=1)
    resid = torch.cat(resid_parts, dim=1)
    require_frame_vector(tuple(resid.shape), "replacement residual")
    f0_logit = torch.logit((1.0 - torch.sigmoid(selected_logit)).clamp(1e-6, 1.0 - 1e-6))
    return {
        "f0_logit": f0_logit,
        "z_residual": resid,
        "anchor_logit": anchor_logit,
        "logits": evidence["logits"],
        "slot_of": slot_of,
        "mapping_rows": mapping_rows,
        "unmapped_frames": unmapped_frames,
        "windows": len(passage["windows"]),
    }


def forward_head_windows(
    torch: Any, head_module: Any, features: Any, windows: list[tuple[int, int]]
) -> tuple[Any, Any]:
    gru_state: Any = None
    anchor_parts: list[Any] = []
    resid_parts: list[Any] = []
    for window_index, (window_start, window_end) in enumerate(windows):
        outputs, gru_state = head_module(features[:, window_start:window_end], gru_state)
        anchor_parts.append(outputs["anchor_logit"])
        resid_parts.append(outputs["z_residual"])
        if window_index < len(windows) - 1:
            gru_state = _detach_state(torch, gru_state)
    return torch.cat(anchor_parts, dim=1), torch.cat(resid_parts, dim=1)


def infer_slice_source_evidence(
    torch: Any,
    wrapper: Any,
    waveform: Any,
    authority: Any,
    mult: Any,
    episode_ids: list[str | None],
    frame_count: int,
    tail_excluded: Any,
    source_id: str,
    device: Any,
) -> dict[str, Any]:
    chunked = run_adjacent_windows(torch, wrapper, waveform, CHUNK_FRAMES, True)
    if len(chunked["windows"]) < 2:
        raise MaterialError(f"slice source is shorter than two chunks: {source_id}")
    oneshot = run_adjacent_windows(torch, wrapper, waveform, 1 << 30, False)
    chunked_all = concat_windows(torch, chunked["windows"])
    oneshot_all = concat_windows(torch, oneshot["windows"])
    require_frame_alignment(chunked_all["hidden"].shape[1], frame_count, source_id)
    equivalence = {
        name: _max_abs_diff(torch, oneshot_all[name], chunked_all[name])
        for name in ("hidden", "logits", "probabilities")
    }
    if any(v > EQUIVALENCE_TOL for v in equivalence.values()):
        raise MaterialError(f"stateful equivalence failed: {source_id}")
    evidence = chunked_all
    anchor_active = [a == 1.0 for a in authority.y_anchor]
    probabilities = evidence["probabilities"][0].detach().cpu().tolist()
    slot_of, mapping_rows, unmapped_frames = oracle_slot_mapping(
        [e for e in episode_ids], anchor_active, list(authority.valid), probabilities
    )
    mapped = sum(1 for r in mapping_rows if r["status"] == "mapped")
    if mapped == 0:
        raise MaterialError(f"oracle mapping is empty: {source_id}")
    unmapped_set = set(unmapped_frames)
    mapped_flags = [i not in unmapped_set for i in range(frame_count)]
    one_hot = torch.zeros((1, frame_count, 4), dtype=torch.float32, device=device)
    for frame, episode_id in enumerate(episode_ids):
        if episode_id is not None and episode_id in slot_of:
            one_hot[0, frame, slot_of[episode_id]] = 1.0
    selected_long = (evidence["logits"] * one_hot).sum(dim=-1, keepdim=True)
    require_frame_vector(tuple(selected_long.squeeze(-1).shape), "selected anchor logit")
    selected_logit = selected_long.squeeze(-1)
    anchor_mask = one_hot.bool()
    neg_inf = torch.full_like(evidence["logits"], float("-inf"))
    non_anchor = torch.where(anchor_mask, neg_inf, evidence["logits"])
    best_non_anchor = non_anchor.max(dim=-1, keepdim=True).values
    best_non_anchor = torch.where(
        torch.isfinite(best_non_anchor), best_non_anchor, torch.zeros_like(best_non_anchor)
    )
    delay = torch.full_like(selected_logit.unsqueeze(-1), 1.04)
    features = torch.cat(
        [
            evidence["hidden"],
            evidence["logits"],
            selected_logit.unsqueeze(-1),
            best_non_anchor,
            delay,
        ],
        dim=-1,
    )
    return {
        "authority": authority,
        "multiplicity": mult,
        "episode_ids": episode_ids,
        "one_hot": one_hot,
        "selected_logit": selected_logit,
        "features": features,
        "mapping_rows": mapping_rows,
        "mapping_mapped": mapped,
        "unmapped_frames": unmapped_frames,
        "mapped_flags": mapped_flags,
        "equivalence": equivalence,
        "boundary_steps": chunked["boundary_steps"],
        "tail_excluded": tail_excluded,
        "frame_count": frame_count,
        "waveform": waveform,
    }


def run_slice_update(
    torch: Any,
    wrapper: Any,
    head_module: Any,
    train: dict[str, Any],
    class_weights: dict[str, float],
    device: Any,
    train_source: str,
) -> dict[str, Any]:
    from experiments.psem_sortformer_adaptation_depth.models import (
        masked_balanced_bce_with_logits,
    )
    train_mult = train["multiplicity"]
    train_mapped = torch.tensor(
        [[1.0 if m else 0.0 for m in train["mapped_flags"]]],
        dtype=torch.float32,
        device=device,
    )
    train_mult_weight = torch.tensor(
        [[float(m) for m in train_mult]], dtype=torch.float32, device=device
    ) * train_mapped
    train_y_replace = torch.tensor(
        [train["authority"].y_replace], dtype=torch.float32, device=device
    )
    train_y_anchor = torch.tensor(
        [train["authority"].y_anchor], dtype=torch.float32, device=device
    )
    if float((train_mult_weight * train_y_replace).sum()) <= 0:
        raise MaterialError(f"slice source lacks replacement support: {train_source}")
    plan = microbatch_plan(train["frame_count"], CHUNK_FRAMES, 16)
    windows = plan_windows(train["frame_count"], CHUNK_FRAMES)[:2]
    head_module.train(True)
    anchor_all, resid_all = forward_head_windows(torch, head_module, train["features"], windows)
    slice_end = windows[1][1]
    f0_all = torch.logit(
        (1.0 - torch.sigmoid(train["selected_logit"][:, :slice_end])).clamp(1e-6, 1.0 - 1e-6)
    )
    product_all = f0_all + resid_all
    identity_diff = float((product_all - f0_all).abs().max())
    if identity_diff > IDENTITY_TOL:
        raise MaterialError(f"zero-residual identity failed: {train_source}")
    train["anchor_logit"] = anchor_all
    train["product_logit"] = product_all
    train["f0_logit"] = f0_all
    frozen_before = [p.detach().clone() for p in wrapper.parameters()]
    head_before = [p.detach().clone() for p in head_module.parameters()]
    print("[gate0] phase=slice-update", flush=True)
    optimizer = torch.optim.AdamW(
        [p for p in head_module.parameters() if p.requires_grad], lr=OPTIMIZER_LR
    )
    optimizer.zero_grad()
    microbatches = [mb for mb in plan if mb["end"] <= windows[1][1]]
    for position, mb in enumerate(microbatches):
        mb_slice = slice(mb["start"], mb["end"])
        mb_mult = train_mult_weight[:, mb_slice]
        if float(mb_mult.sum()) == 0:
            continue
        last_in_window = position == len(microbatches) - 1 or microbatches[position + 1][
            "detach_state"
        ]
        bce_none = torch.nn.functional.binary_cross_entropy_with_logits(
            product_all[:, mb_slice],
            train_y_replace[:, mb_slice],
            pos_weight=torch.as_tensor(
                class_weights["replacement_positive_weight"],
                dtype=product_all.dtype,
                device=device,
            ),
            reduction="none",
        )
        anchor_none = torch.nn.functional.binary_cross_entropy_with_logits(
            anchor_all[:, mb_slice],
            train_y_anchor[:, mb_slice],
            pos_weight=torch.as_tensor(
                class_weights["anchor_positive_weight"],
                dtype=anchor_all.dtype,
                device=device,
            ),
            reduction="none",
        )
        denom = mb_mult.sum().clamp_min(1.0)
        loss = (bce_none * mb_mult).sum() / denom + 0.5 * (anchor_none * mb_mult).sum() / denom
        audit_mask = (mb_mult > 0).to(dtype=product_all.dtype)
        audit_value = masked_balanced_bce_with_logits(
            product_all[:, mb_slice].detach(),
            train_y_replace[:, mb_slice],
            audit_mask,
            class_weights["replacement_positive_weight"],
        )
        if not bool(torch.isfinite(audit_value)):
            raise MaterialError("shared loss primitive is non-finite on real evidence")
        loss.backward(retain_graph=not last_in_window)
    grads = [p.grad for p in head_module.parameters() if p.grad is not None]
    if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):
        raise MaterialError("trainable head gradients are missing or non-finite")
    optimizer.step()
    optimizer.zero_grad()
    if any(not bool(torch.equal(a, b)) for a, b in zip(frozen_before, wrapper.parameters())):
        raise MaterialError("frozen backbone parameters changed")
    if all(bool(torch.equal(a, b)) for a, b in zip(head_before, head_module.parameters())):
        raise MaterialError("trainable head parameters did not update")
    return {
        "anchor_all": anchor_all,
        "resid_all": resid_all,
        "f0_all": f0_all,
        "product_all": product_all,
        "optimizer": optimizer,
        "train_mult_weight": train_mult_weight,
        "train_y_replace": train_y_replace,
        "train_y_anchor": train_y_anchor,
        "microbatches": microbatches,
        "windows": windows,
        "identity_diff": identity_diff,
    }


def run_profiler(
    torch: Any,
    head_module: Any,
    train: dict[str, Any],
    update_ctx: dict[str, Any],
    class_weights: dict[str, float],
    device: Any,
    dev_timings: dict[str, float],
    profile_steps: int = 8,
) -> dict[str, Any]:
    optimizer = update_ctx["optimizer"]
    train_mult_weight = update_ctx["train_mult_weight"]
    train_y_replace = update_ctx["train_y_replace"]
    train_y_anchor = update_ctx["train_y_anchor"]
    microbatches = update_ctx["microbatches"]
    windows = update_ctx["windows"]
    f0_all = update_ctx["f0_all"]
    head_module.train(True)
    print(
        f"[gate0] phase=profile steps={profile_steps}",
        flush=True,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    profile_start = time.perf_counter()
    for _ in range(profile_steps):
        optimizer.zero_grad()
        profile_anchor, profile_resid = forward_head_windows(
            torch, head_module, train["features"], windows
        )
        profile_product = f0_all.detach() + profile_resid
        for position, mb in enumerate(microbatches):
            mb_slice = slice(mb["start"], mb["end"])
            mb_mult = train_mult_weight[:, mb_slice]
            if float(mb_mult.sum()) == 0:
                continue
            last_in_window = position == len(microbatches) - 1 or microbatches[position + 1][
                "detach_state"
            ]
            profile_bce = torch.nn.functional.binary_cross_entropy_with_logits(
                profile_product[:, mb_slice],
                train_y_replace[:, mb_slice],
                pos_weight=torch.as_tensor(
                    class_weights["replacement_positive_weight"],
                    dtype=profile_product.dtype,
                    device=device,
                ),
                reduction="none",
            )
            profile_denom = mb_mult.sum().clamp_min(1.0)
            profile_anchor_none = torch.nn.functional.binary_cross_entropy_with_logits(
                profile_anchor[:, mb_slice],
                train_y_anchor[:, mb_slice],
                pos_weight=torch.as_tensor(
                    class_weights["anchor_positive_weight"],
                    dtype=profile_anchor.dtype,
                    device=device,
                ),
                reduction="none",
            )
            profile_loss = (profile_bce * mb_mult).sum() / profile_denom + 0.5 * (
                profile_anchor_none * mb_mult
            ).sum() / profile_denom
            profile_loss.backward(retain_graph=not last_in_window)
        optimizer.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    profile_seconds = time.perf_counter() - profile_start
    head_module.eval()
    return {
        "optimizer_steps": profile_steps,
        "seconds_per_step": profile_seconds / profile_steps,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device))
        if torch.cuda.is_available()
        else 0,
        "dev_infer_seconds": dict(dev_timings),
        "non_authoritative": True,
    }


def infer_dev_raw_logits(
    torch: Any,
    wrapper: Any,
    head_module: Any,
    dev: Any,
    runtime: Any,
    corpus_root: Path,
    device: Any,
) -> dict[str, Any]:
    import numpy as np
    import torchaudio
    from experiments.psem_sortformer_adaptation_depth.frame_alignment import (
        action_sample_indices,
    )
    dev_relative = Path(runtime.audio_ref)
    dev_path = (corpus_root.resolve() / dev_relative).resolve()
    dev_audio, dev_rate = torchaudio.load(str(dev_path))
    if dev_rate != SAMPLE_RATE_HZ or dev_audio.ndim != 2 or dev_audio.shape[0] != 1:
        raise MaterialError(f"DEV waveform geometry is invalid: {dev.source_id}")
    dev_grid_frames = len(dev.starts)
    dev_usable, dev_tail = slice_waveform_frames(
        int(dev_audio.shape[1]), dev_grid_frames, dev.source_id
    )
    dev_waveform = dev_audio[:, :dev_usable].to(device)
    dev_authority_frames = dev_grid_frames
    dev_start = time.perf_counter()
    dev_passage = run_adjacent_windows(torch, wrapper, dev_waveform, 1 << 30, False)
    dev_evidence = concat_windows(torch, dev_passage["windows"])
    require_frame_alignment(dev_evidence["emitted_frames"], dev_authority_frames, dev.source_id)
    dev_native_ends = np.asarray(
        [(i + 1) * FRAME_SAMPLES for i in range(dev_evidence["emitted_frames"])],
        dtype=np.int64,
    )
    dev_indices = action_sample_indices(dev_native_ends, np.asarray(dev.ends))
    if len(dev_indices) != dev_grid_frames:
        raise MaterialError(f"DEV grid alignment failed: {dev.source_id}")
    dev_probs = dev_evidence["probabilities"][0][dev_indices]
    dev_logits = dev_evidence["logits"][0][dev_indices].unsqueeze(0)
    dev_hidden = dev_evidence["hidden"][0][dev_indices].unsqueeze(0)
    dev_probs_list = np.asarray(dev_probs.detach().cpu(), dtype=np.float64).tolist()
    dev_episode_ids = [None if str(v) in {"", "None"} else str(v) for v in dev.episode_ids]
    dev_anchor_list = [bool(v) for v in np.asarray(dev.anchor_present)]
    dev_valid_list = [bool(v) for v in np.asarray(dev.valid)]
    slots, mapping_rows, dev_unmapped = oracle_slot_mapping(
        dev_episode_ids, dev_anchor_list, dev_valid_list, dev_probs_list
    )
    mapped = sum(1 for r in mapping_rows if r["status"] == "mapped")
    if mapped == 0:
        raise MaterialError(f"DEV oracle mapping is empty: {dev.source_id}")
    dev_unmapped_set = set(dev_unmapped)
    dev_one_hot = torch.zeros((1, len(dev.starts), 4), dtype=torch.float32, device=device)
    for frame, episode_id in enumerate(dev_episode_ids):
        if episode_id is not None and episode_id in slots:
            dev_one_hot[0, frame, slots[episode_id]] = 1.0
    dev_selected = (dev_logits * dev_one_hot).sum(dim=-1, keepdim=True).squeeze(-1)
    require_frame_vector(tuple(dev_selected.shape), "DEV selected anchor logit")
    dev_f0_logit = torch.logit((1.0 - torch.sigmoid(dev_selected)).clamp(1e-6, 1.0 - 1e-6))
    dev_anchor_mask = dev_one_hot.bool()
    dev_neg_inf = torch.full_like(dev_logits, float("-inf"))
    dev_non_anchor = torch.where(dev_anchor_mask, dev_neg_inf, dev_logits)
    dev_best = dev_non_anchor.max(dim=-1, keepdim=True).values
    dev_best = torch.where(torch.isfinite(dev_best), dev_best, torch.zeros_like(dev_best))
    dev_delay = torch.full((1, len(dev.starts), 1), 1.04, device=device)
    dev_features = torch.cat(
        [dev_hidden, dev_logits, dev_selected.unsqueeze(-1), dev_best, dev_delay], dim=-1
    )
    with torch.no_grad():
        dev_outputs, _ = head_module(dev_features)
    require_frame_vector(tuple(dev_outputs["z_residual"].shape), "DEV residual")
    dev_resid = dev_outputs["z_residual"][0].detach().cpu()
    dev_f0_raw = dev_f0_logit.detach().cpu().flatten().tolist()
    dev_cand_raw = (dev_f0_logit + dev_outputs["z_residual"][0]).detach().cpu().flatten().tolist()
    dev_target = [float(v) for v in np.asarray(dev.target)]
    infer_seconds = time.perf_counter() - dev_start
    dev_mapped_flags = [i not in dev_unmapped_set for i in range(dev_grid_frames)]
    dev_kept, dev_coverage = mask_calibration(dev_target, dev_valid_list, dev_mapped_flags)
    return {
        "f0_raw": dev_f0_raw,
        "cand_raw": dev_cand_raw,
        "target": dev_target,
        "valid": dev_valid_list,
        "mapped_flags": dev_mapped_flags,
        "kept": dev_kept,
        "coverage": dev_coverage,
        "mapping_mapped": mapped,
        "mapping_rows": mapping_rows,
        "unmapped_frames": sorted(dev_unmapped_set),
        "grid_frames": dev_grid_frames,
        "infer_seconds": infer_seconds,
    }


def _point_dict(point: Any) -> dict[str, float]:
    return {
        "threshold": float(point.threshold),
        "false_cuts_per_hour": float(point.false_cuts_per_hour),
        "contamination": float(point.contamination),
        "miss_rate": float(point.miss_rate),
    }


def build_horizon_result(
    f0_point: Any,
    candidate_points: list[Any],
    envelopes: dict[str, Any],
    mapping_mapped: int,
    mapping_total: int,
    unmapped_frames: int,
    kept_frames: int,
    raw_ap: float,
    kept_cand_cal: list[float],
    kept_target: list[float],
    kept_f0_cal: list[float],
) -> dict[str, Any]:
    from experiments.psem_state_corrected_adaptation_gate import calibrate as calibrate_mod
    return {
        "f0": f0_point,
        "envelopes": envelopes,
        "points": [
            {
                "threshold": round(p.threshold, 6),
                "false_cuts_per_hour": round(p.false_cuts_per_hour, 6),
                "contamination": round(p.contamination, 6),
                "miss_rate": round(p.miss_rate, 6),
            }
            for p in candidate_points
        ],
        "mapping_mapped": mapping_mapped,
        "mapping_total": mapping_total,
        "unmapped_frames": unmapped_frames,
        "kept_frames": kept_frames,
        "raw_ap": raw_ap,
        "candidate_nll": calibrate_mod.nll_loss(kept_cand_cal, kept_target),
        "candidate_brier": calibrate_mod.brier_score(kept_cand_cal, kept_target),
        "f0_nll": calibrate_mod.nll_loss(kept_f0_cal, kept_target),
        "f0_brier": calibrate_mod.brier_score(kept_f0_cal, kept_target),
    }


def build_gate0_record(
    slice_sources: Any,
    runtime_receipt: dict[str, Any],
    sampling_sha256: str,
    modes: dict[str, Any],
    assignment: dict[str, Any],
    class_weights: dict[str, float],
    calib_coverages: dict[str, Any],
    calibration_candidate: dict[str, Any],
    calibration_f0: dict[str, Any],
    dev_points: dict[str, Any],
    profiler: dict[str, Any],
    predictions_meta: dict[str, dict[str, str]],
    mode: str = "material",
) -> dict[str, Any]:
    from experiments.psem_state_corrected_adaptation_gate import receipts as receipts_mod
    checks = {
        "checkpoint": True,
        "sources": True,
        "lifecycle": True,
        "multiplicity": True,
        "equivalence": True,
        "identity": True,
        "modes": True,
        "update": True,
        "calibration": True,
        "frontier": True,
    }
    evidence = {
        "checkpoint_sha256": runtime_receipt["checkpoint_sha256"],
        "sampling_sha256": sampling_sha256,
        "runtime_receipt_sha256": receipts_mod.canonical_sha256(runtime_receipt),
        "module_modes": modes,
        "partition": {
            "fit": assignment["fit"],
            "calib": assignment["calib"],
            "salt": assignment["salt"],
            "target_frac": assignment["target_frac"],
            "class_weights": class_weights,
        },
        "calibration_coverage": calib_coverages,
        "calibration_candidate": {
            key: float(calibration_candidate[key])
            for key in ("slope", "intercept", "nll", "brier", "raw_nll", "raw_brier")
        } | {"role": calibration_candidate["role"]},
        "calibration_f0": {
            key: float(calibration_f0[key])
            for key in ("slope", "intercept", "nll", "brier", "raw_nll", "raw_brier")
        } | {"role": calibration_f0["role"]},
        "dev": {
            source_id: {
                str(horizon_ms): {
                    "budget": result["envelopes"]["budget"],
                    "useful": result["envelopes"]["useful"],
                    "c_envelope": _point_dict(result["envelopes"]["c_envelope"])
                    if result["envelopes"]["c_envelope"] is not None
                    else None,
                    "m_envelope": _point_dict(result["envelopes"]["m_envelope"])
                    if result["envelopes"]["m_envelope"] is not None
                    else None,
                    "f0": _point_dict(result["f0"]),
                    "points": result["points"],
                    "mapping_mapped": result["mapping_mapped"],
                    "mapping_total": result["mapping_total"],
                    "unmapped_frames": result["unmapped_frames"],
                    "kept_frames": result["kept_frames"],
                    "raw_ap": result["raw_ap"],
                    "candidate_nll": result["candidate_nll"],
                    "candidate_brier": result["candidate_brier"],
                    "f0_nll": result["f0_nll"],
                    "f0_brier": result["f0_brier"],
                }
                for horizon_ms, result in horizons.items()
            }
            for source_id, horizons in dev_points.items()
        },
        "profiler": profiler,
        "predictions": predictions_meta,
    }
    record = {
        **receipts_mod.material_vertical_slice_record(
            slice_sources[0], slice_sources[1], checks, mode
        ),
        "evidence": evidence,
    }
    validate_gate0_record(record)
    return record


def write_gate0_artifacts(
    out_dir: Path,
    record: dict[str, Any],
    assignment: dict[str, Any],
    class_weights: dict[str, float],
    modes: dict[str, Any],
) -> None:
    from experiments.psem_state_corrected_adaptation_gate import receipts as receipts_mod
    out_dir.mkdir(parents=True, exist_ok=True)
    receipts_mod.write_json(out_dir / "experiment_manifest.json", receipts_mod.experiment_manifest())
    receipts_mod.write_json(
        out_dir / "data_sampling_calibration_manifest.json",
        {
            **receipts_mod.sampling_calibration_manifest(
                assignment["fit"], assignment["calib"], assignment["salt"],
                assignment["target_frac"],
            ),
            "class_weights": class_weights,
        },
    )
    receipts_mod.write_json(out_dir / "material_vertical_slice.json", record)
    receipts_mod.write_json(out_dir / "parameter_module_mode_receipt.json", modes)


def run_material_slice(
    resolved: ResolvedMaterialInputs, out_dir: Path, workers: int | None = None
) -> dict[str, Any]:
    worker_count = resolve_worker_count(workers)
    print(f"[gate0] phase=start workers={worker_count}", flush=True)
    torch = _require_torch()
    try:
        from experiments.psem_sortformer_adaptation_depth.nemo_adapter import (
            load_pinned_sortformer,
        )
        from experiments.psem_sortformer_adaptation_depth.sampling import (
            load_training_sessions,
            validate_training_waveform_paths,
        )
        from experiments.psem_sortformer_adaptation_depth.frame_alignment import (
            action_sample_indices,
        )
        from experiments.psem_sortformer_adaptation_depth.preflight import (
            SOURCE_MANIFEST_PATH,
        )
        from experiments.psem_training_strategy_gate.sampling import DEV_ROLE
        from experiments.psem_sortformer_adaptation_depth.execution import (
            load_scoring_sessions,
        )
        from experiments.psem_sortformer_adaptation_depth.models import (
            masked_balanced_bce_with_logits,
        )
        from experiments.psem_frozen_ceiling_gate.build_ceiling_examples import (
            load_sessions,
        )
        from experiments.psem_frozen_ceiling_gate.evaluate_ceiling import (
            decode_scores,
            session_metrics,
        )
        import numpy as np
        import torchaudio
    except ImportError as exc:
        raise MaterialBlockedError(
            "material execution requires the pinned worker runtime"
        ) from exc
    from experiments.psem_state_corrected_adaptation_gate import (
        calibrate,
        frontier,
        head,
    )
    from experiments.psem_state_corrected_adaptation_gate.partition import (
        SourceExposure,
        assign_train_calib,
        validate_partition_support,
    )
    from experiments.psem_state_corrected_adaptation_gate.receipts import (
        canonical_sha256,
        experiment_manifest,
        material_vertical_slice_record,
        sampling_calibration_manifest,
        write_json,
    )

    sessions = load_training_sessions(resolved.corpus_root, resolved.reference_root)
    waveform_paths = validate_training_waveform_paths(sessions, resolved.corpus_root)
    for source_id in (resolved.ami_source, resolved.alimeeting_source):
        if source_id not in sessions:
            raise MaterialError(f"slice source outside the frozen TRAIN split: {source_id}")
    print("[gate0] phase=model-load", flush=True)
    wrapper, runtime_receipt = load_pinned_sortformer(
        resolved.checkpoint_path,
        resolved.nemo_checkout,
        resolved.dependency_lock,
        resolved.device,
    )
    wrapper.eval()
    device = next(wrapper.parameters()).device
    head_module = head.ResidualPSEMHead(199)
    head_module.to(device)
    head_module.train(True)
    for parameter in wrapper.parameters():
        parameter.requires_grad_(False)
    dropout_training = [
        bool(module.training)
        for module in wrapper.modules()
        if isinstance(module, torch.nn.Dropout)
    ]
    modes = audit_module_modes(
        bool(wrapper.training),
        dropout_training,
        bool(head_module.training),
        [name for name, p in wrapper.named_parameters() if p.requires_grad],
        [name for name, p in head_module.named_parameters() if p.requires_grad],
    )
    components = load_source_components(sessions, SOURCE_MANIFEST_PATH)
    source_rows = load_source_rows(SOURCE_MANIFEST_PATH)
    print(
        f"[gate0] phase=target-build sources={len(sessions)} workers={worker_count}",
        flush=True,
    )
    target_cache = build_all_source_targets(sessions, resolved.rows_by_source, worker_count)
    print(f"[gate0] phase=target-build-done sources={len(target_cache)}", flush=True)
    corpus_of: dict[str, str] = {}
    for source_id in sessions:
        manifest_corpus = source_rows.get(source_id, {}).get("corpus")
        row_corpus = next(
            (
                row.get("corpus")
                for row in resolved.rows_by_source.get(source_id, [])
                if row.get("corpus") in ("AMI", "AliMeeting")
            ),
            None,
        )
        corpus = row_corpus or manifest_corpus
        if corpus not in ("AMI", "AliMeeting"):
            raise MaterialError(f"slice corpus is unknown: {source_id}")
        corpus_of[source_id] = corpus
    exposures = [
        SourceExposure(
            source_id=source_id,
            corpus=corpus_of[source_id],
            exposure=float(len(resolved.rows_by_source.get(source_id, []))),
            positive_frames=int(sum(target_cache[source_id]["authority"].y_replace)),
            negative_frames=int(
                sum(1 for v in target_cache[source_id]["authority"].y_replace if v == 0)
            ),
        )
        for source_id in sessions
    ]
    assignment = assign_train_calib(exposures, components=components)
    by_source = {e.source_id: e for e in exposures}
    validate_partition_support(assignment, by_source)
    accumulator = ClassAccumulator()
    for source_id in assignment["fit"]:
        entry = target_cache[source_id]
        accumulator.add(
            list(entry["authority"].y_replace),
            list(entry["authority"].y_anchor),
            list(entry["multiplicity"]),
            list(entry["authority"].valid),
        )
    class_weights = accumulator.weights()
    slice_sources = select_fit_slice(assignment["fit"], resolved.rows_by_source, corpus_of)
    calib_sources = [s for s in assignment["calib"] if resolved.rows_by_source.get(s)]
    if not calib_sources:
        raise MaterialError("partition has no TRAIN-CALIB source with sampling exposure")
    per_source: dict[str, dict[str, Any]] = {}
    print(f"[gate0] phase=slice-inference sources={slice_sources}", flush=True)
    for source_id in slice_sources:
        cached = target_cache[source_id]
        authority = cached["authority"]
        mult = cached["multiplicity"]
        episode_ids = cached["episode_ids"]
        frame_count = authority.num_frames
        if sum(mult) == 0:
            raise MaterialError(f"slice source has no sampling exposure: {source_id}")
        audio, sample_rate = torchaudio.load(str(waveform_paths[source_id]))
        if sample_rate != SAMPLE_RATE_HZ or audio.ndim != 2 or audio.shape[0] != 1:
            raise MaterialError(f"source waveform geometry is invalid: {source_id}")
        usable_samples, tail_excluded = slice_waveform_frames(
            int(audio.shape[1]), frame_count, source_id
        )
        waveform = audio[:, :usable_samples].to(device)
        per_source[source_id] = infer_slice_source_evidence(
            torch,
            wrapper,
            waveform,
            authority,
            mult,
            episode_ids,
            frame_count,
            tail_excluded,
            source_id,
            device,
        )

    train_source = slice_sources[0]
    train = per_source[train_source]
    update_ctx = run_slice_update(
        torch, wrapper, head_module, train, class_weights, device, train_source
    )
    head_module.eval()
    calib_f0_raw: list[float] = []
    calib_cand_raw: list[float] = []
    calib_targets: list[float] = []
    calib_coverages: dict[str, dict[str, int]] = {}
    with torch.no_grad():
        print(f"[gate0] phase=calib sources={calib_sources}", flush=True)
        for source_id in calib_sources:
            cached = target_cache[source_id]
            calib_authority = cached["authority"]
            calib_frames = calib_authority.num_frames
            calib_audio, calib_rate = torchaudio.load(str(waveform_paths[source_id]))
            if calib_rate != SAMPLE_RATE_HZ or calib_audio.ndim != 2 or calib_audio.shape[0] != 1:
                raise MaterialError(f"CALIB waveform geometry is invalid: {source_id}")
            calib_usable, _ = slice_waveform_frames(
                int(calib_audio.shape[1]), calib_frames, source_id
            )
            calib_waveform = calib_audio[:, :calib_usable].to(device)
            calib_anchor = [a == 1.0 for a in calib_authority.y_anchor]
            calib_valid = list(calib_authority.valid)
            calib_out = infer_arm_logits(
                torch,
                wrapper,
                head_module,
                calib_waveform,
                cached["episode_ids"],
                calib_anchor,
                calib_valid,
                calib_frames,
                device,
            )
            calib_unmapped = set(calib_out["unmapped_frames"])
            calib_mapped = [
                i not in calib_unmapped for i in range(calib_frames)
            ]
            calib_targets_all = [float(v) for v in calib_authority.y_replace]
            kept, calib_coverage = mask_calibration(calib_targets_all, calib_valid, calib_mapped)
            calib_f0_all = calib_out["f0_logit"].flatten().tolist()
            calib_cand_all = (
                calib_out["f0_logit"] + calib_out["z_residual"]
            ).flatten().tolist()
            calib_coverages[source_id] = calib_coverage
            extend_calibration_buffers(
                {"f0": calib_f0_raw, "cand": calib_cand_raw, "targets": calib_targets},
                calib_f0_all,
                calib_cand_all,
                calib_targets_all,
                kept,
            )
    if not calib_f0_raw:
        raise MaterialError("TRAIN-CALIB inference produced no frames")
    calibration_f0 = calibrate.fit_affine_calibrator(calib_f0_raw, calib_targets, "TRAIN-CALIB")
    calibration_candidate = calibrate.fit_affine_calibrator(
        calib_cand_raw, calib_targets, "TRAIN-CALIB"
    )

    dev_runtime = load_scoring_sessions(
        resolved.corpus_root, resolved.reference_root, DEV_ROLE
    )
    dev_sessions = load_sessions()
    dev_points: dict[str, dict[str, Any]] = {}
    predictions_meta: dict[str, dict[str, str]] = {}
    dev_timings: dict[str, float] = {}
    for family in ("ami_mix_headset", "alimeeting_far_ch0"):
        print(f"[gate0] phase=dev-inference family={family}", flush=True)
        candidates = [
            s for s in dev_sessions if is_dev_family_session(s, family)
        ]
        if not candidates:
            raise MaterialError(f"DEV snapshot has no {family} session")
        dev = candidates[0]
        if dev.source_id not in dev_runtime:
            raise MaterialError(f"DEV session outside the frozen DEV split: {dev.source_id}")
        runtime = dev_runtime[dev.source_id]
        raw = infer_dev_raw_logits(
            torch, wrapper, head_module, dev, runtime, resolved.corpus_root, device
        )
        dev_f0_raw = raw["f0_raw"]
        dev_cand_raw = raw["cand_raw"]
        dev_target = raw["target"]
        dev_valid_list = raw["valid"]
        dev_mapped_flags = raw["mapped_flags"]
        dev_kept = raw["kept"]
        dev_coverage = raw["coverage"]
        mapped = raw["mapping_mapped"]
        mapping_rows = raw["mapping_rows"]
        dev_unmapped_set = set(raw["unmapped_frames"])
        dev_grid_frames = raw["grid_frames"]
        dev_timings[dev.source_id] = raw["infer_seconds"]
        dev_f0_cal = calibrate.apply_affine(
            dev_f0_raw, float(calibration_f0["slope"]), float(calibration_f0["intercept"])
        )
        dev_cand_cal = calibrate.apply_affine(
            dev_cand_raw,
            float(calibration_candidate["slope"]),
            float(calibration_candidate["intercept"]),
        )
        dev_f0_np = np.asarray([calibrate.sigmoid(z) for z in dev_f0_cal], dtype=np.float64)
        dev_candidate_np = np.asarray(
            [calibrate.sigmoid(z) for z in dev_cand_cal], dtype=np.float64
        )
        dev_f0_np[sorted(dev_unmapped_set)] = float("-inf")
        dev_candidate_np[sorted(dev_unmapped_set)] = float("-inf")
        kept_target = [dev_target[i] for i in dev_kept]
        kept_cand_raw = [[calibrate.sigmoid(z) for z in dev_cand_raw][i] for i in dev_kept]
        kept_cand_cal = [dev_cand_cal[i] for i in dev_kept]
        kept_f0_cal = [dev_f0_cal[i] for i in dev_kept]
        raw_ap = calibrate.average_precision(kept_cand_raw, kept_target)
        horizon_results: dict[int, dict[str, Any]] = {}
        for horizon_ms in REQUIRED_HORIZONS:
            f0_events = decode_scores(dev, dev_f0_np, threshold=0.5, confirmation_ms=horizon_ms)
            f0_metrics = session_metrics(dev, f0_events)
            f0_point = _frontier_point(f0_metrics)
            thresholds = frontier.unique_thresholds(dev_candidate_np.tolist())
            print(
                f"[gate0] phase=frontier source={dev.source_id} horizon={horizon_ms} thresholds={len(thresholds)} workers={worker_count}",
                flush=True,
            )
            candidate_points = candidate_frontier_points(
                dev,
                dev_candidate_np.tolist(),
                thresholds,
                horizon_ms,
                worker_count,
            )
            envelopes = frontier.select_envelopes(f0_point, candidate_points)
            horizon_results[horizon_ms] = build_horizon_result(
                f0_point,
                candidate_points,
                envelopes,
                mapped,
                len(mapping_rows),
                len(dev_unmapped_set),
                dev_coverage["kept"],
                raw_ap,
                kept_cand_cal,
                kept_target,
                kept_f0_cal,
            )
        predictions_path = out_dir / f"raw_predictions_{dev.source_id}.npz"
        print(f"[gate0] phase=artifacts source={dev.source_id}", flush=True)
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            predictions_path,
            f0_logit=np.asarray(dev_f0_raw, dtype=np.float64),
            candidate_logit=np.asarray(dev_cand_raw, dtype=np.float64),
            f0_calibrated=np.asarray(dev_f0_cal, dtype=np.float64),
            candidate_calibrated=np.asarray(dev_cand_cal, dtype=np.float64),
            target=np.asarray(dev_target, dtype=np.float64),
        )
        predictions_meta[dev.source_id] = {
            "path": predictions_path.name,
            "sha256": _sha256_file(predictions_path),
        }
        dev_points[dev.source_id] = horizon_results
    profiler = run_profiler(
        torch, head_module, train, update_ctx, class_weights, device, dev_timings
    )
    record = build_gate0_record(
        slice_sources,
        runtime_receipt,
        resolved.sampling_sha256,
        modes,
        assignment,
        class_weights,
        calib_coverages,
        calibration_candidate,
        calibration_f0,
        dev_points,
        profiler,
        predictions_meta,
    )
    write_gate0_artifacts(out_dir, record, assignment, class_weights, modes)
    return record
