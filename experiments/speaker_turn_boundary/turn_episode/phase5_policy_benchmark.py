from __future__ import annotations

import argparse
import hashlib
import os
import platform
import time
from collections.abc import Iterator, Mapping
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from ..corpus.external import corpus_root
from ..corpus.phase2_schemas import Phase2Manifest
from .build_episodes import (
    build_reference_specs,
    load_ami_raw_words,
    references_for_episode,
)
from .phase4_signal import atomic_write_json
from .phase5_inputs import annotation_views
from .phase5_policy import (
    actionize_clusters,
    canonical_json,
    cluster_proposals,
    derive_fusion_context,
    frequency_matched_control,
    fuse_actions,
)
from .phase5_proposals import anchor_trace
from .phase5_scoring import score_policy_episode
from .schemas import ReferenceAction

AUTHORITY_SHA256 = "e3efdd9410a84bd343da5ba41d634ceec2d54626e1b512f41e410c0668329e36"
CLUSTER_GRID = tuple(product((0, 100, 250), (250, 500), (0, 250, 500), ("first", "max_confidence")))
VAD_GRID = tuple(product((250, 500), (False, True)))
CURRENT_SENTINEL_PROPOSAL_COUNTS = (0, 1, 4, 16, 64, 320)
CURRENT_MAXIMUM_EMITTABLE_POSITIONS = 301
HISTORICAL_MAXIMUM_EMITTABLE_POSITIONS = 11392
MAXIMUM_SOURCE_PREFIX_PROBE_STEPS = 32696
MAXIMUM_SOURCE_TAIL_SAMPLE = 52320000
MAXIMUM_EPISODE_ROUTE_SAMPLES = 480000
TOTAL_PHYSICAL_PROPOSAL_PROBE_STEPS = 660513
TOTAL_LOGICAL_EMITTABLE_POSITIONS = 259876
TOTAL_LOGICAL_IDENTITY_ROWS = 4988898
AUXILIARY_REPETITIONS_PER_WORKER = 10
IDENTITY_BENCHMARK_ROWS = 200000
HISTORICAL_MANIFEST_SHA256 = "1221176c92f50a2b096e4cd64d5da0168527918e3fba539273c614eabf07a398"
HISTORICAL_SCORING_SHAPES = (
    ("ami_ES2003a", 11392),
    ("ami_IS1008a", 9433),
)
HISTORICAL_WORD_FILE_SHA256 = {
    "ami_ES2003a": {
        "ES2003a.A.words.xml": "8ee1e2ba5ab4421e16ebf42d0e221247a83e174fe34aa06e3aadd7a08540e470",
        "ES2003a.B.words.xml": "d9af4424014f9e2ae55e1a33fc7fb95a2352903453e3afb51118845cea4c304d",
        "ES2003a.C.words.xml": "9393f7af2a0cd95da35d306e4861e2513544c54059b11515ff311cd9b051cff4",
        "ES2003a.D.words.xml": "bdcf987020238618bed2941bc906d31aa61aec6732e3104d16f11b7f1a104384",
    },
    "ami_IS1008a": {
        "IS1008a.A.words.xml": "db744e8ca1f45f794e288c5521ab5dfbc2b2f8e5e9f814dd5300053cdab09a2d",
        "IS1008a.B.words.xml": "4a92c5b967c6a32fdfc4a288b7a61064d7259710b6afb32c27e49e4bd146f322",
        "IS1008a.C.words.xml": "d61e501c37877f731058889d799e7ff5a3d19494b8a22f12c7f0f50a87c62758",
        "IS1008a.D.words.xml": "e7f475b606a950de682d275b912821f3d3d612136ce62bd41bc685146767f40b",
    },
}


def sha256_file(path: Path) -> str:
    digest_value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest_value.update(chunk)
    return digest_value.hexdigest()


def proposals(count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(count):
        boundary = 16000 + index * 5600
        rows.append(
            {
                "proposal_id": f"p{index:05d}",
                "family": "eres2netv2",
                "checkpoint": "sentinel",
                "profile_id": "profile_independent_sentinel",
                "audio_epoch": 1,
                "source_session_id": "sentinel_source",
                "proposal_kind": "speaker_change_unknown",
                "boundary_source_sample": boundary,
                "observed_source_sample_at_emit": boundary + 8000 + (index % 3) * 1600,
                "confidence": 0.35 + (index % 11) * 0.05,
                "confidence_semantics_id": "sentinel_change_strength.v1",
            }
        )
    return rows


def vad_actions() -> list[dict[str, Any]]:
    return [
        {
            "action_id": f"v{index:02d}",
            "audio_epoch": 1,
            "source_session_id": "sentinel_source",
            "boundary_source_sample": 24000 + index * 22400,
            "observed_source_sample_at_emit": 28000 + index * 22400,
            "action_kind": "retain_vad",
        }
        for index in range(6)
    ]


def lifecycle_events() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, vad in enumerate(vad_actions()):
        boundary = int(vad["boundary_source_sample"])
        rows.extend(
            [
                {
                    "event_id": f"speech-end-{index}",
                    "audio_epoch": 1,
                    "source_session_id": "sentinel_source",
                    "event_kind": "speech_end",
                    "reason": "silence",
                    "event_source_sample": boundary - 8000,
                    "observed_source_sample_at_emit": boundary - 4000,
                },
                {
                    "event_id": f"speech-start-{index}",
                    "audio_epoch": 1,
                    "source_session_id": "sentinel_source",
                    "event_kind": "speech_start",
                    "reason": "silence",
                    "event_source_sample": boundary,
                    "observed_source_sample_at_emit": int(vad["observed_source_sample_at_emit"]),
                },
            ]
        )
    return rows


def policy_grid_batch(count: int) -> dict[str, Any]:
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    digest_value = hashlib.sha256()
    cluster_executions = 0
    fusion_executions = 0
    proposal_rows = proposals(count)
    vad = vad_actions()
    episode_end = (
        max((int(row["observed_source_sample_at_emit"]) for row in proposal_rows), default=0) + 8000
    )
    started = time.perf_counter()
    for debounce, radius, refractory, representative in CLUSTER_GRID:
        clustered = cluster_proposals(
            proposal_rows,
            cluster_debounce_ms=debounce,
            cluster_boundary_radius_ms=radius,
            refractory_ms=refractory,
            representative=representative,
            episode_observed_end=episode_end,
        )
        detector = actionize_clusters(clustered["clusters"])
        contextual_vad, contextual_detector, _ = derive_fusion_context(
            vad, detector, lifecycle_events()
        )
        cluster_executions += 1
        for vad_radius, silence_association in VAD_GRID:
            fused = fuse_actions(
                contextual_vad,
                contextual_detector,
                detector_vad_radius_ms=vad_radius,
                same_silence_interval_association=silence_association,
            )
            fusion_executions += 1
            digest_value.update(
                canonical_json(
                    [
                        count,
                        debounce,
                        radius,
                        refractory,
                        representative,
                        vad_radius,
                        silence_association,
                        clustered["cluster_count"],
                        clustered["refractory_count"],
                        fused["hard_action_count"],
                    ]
                ).encode("utf-8")
            )
        peak_rss = max(peak_rss, process.memory_info().rss)
    elapsed = time.perf_counter() - started
    return {
        "proposal_count": count,
        "cluster_execution_count": cluster_executions,
        "fusion_execution_count": fusion_executions,
        "elapsed_seconds": elapsed,
        "peak_rss_bytes": peak_rss,
        "trace_sha256": digest_value.hexdigest(),
    }


def benchmark_worker(repetitions: int) -> dict[str, Any]:
    results = [
        policy_grid_batch(
            CURRENT_SENTINEL_PROPOSAL_COUNTS[index % len(CURRENT_SENTINEL_PROPOSAL_COUNTS)]
        )
        for index in range(repetitions)
    ]
    return {
        "batch_count": repetitions,
        "cluster_execution_count": sum(row["cluster_execution_count"] for row in results),
        "fusion_execution_count": sum(row["fusion_execution_count"] for row in results),
        "elapsed_seconds": sum(float(row["elapsed_seconds"]) for row in results),
        "peak_rss_bytes": max(int(row["peak_rss_bytes"]) for row in results),
        "trace_sha256": hashlib.sha256(
            "".join(str(row["trace_sha256"]) for row in results).encode("utf-8")
        ).hexdigest(),
    }


def aggregate(workers: int, results: list[dict[str, Any]], wall_seconds: float) -> dict[str, Any]:
    batch_count = sum(int(row["batch_count"]) for row in results)
    cluster_count = sum(int(row["cluster_execution_count"]) for row in results)
    fusion_count = sum(int(row["fusion_execution_count"]) for row in results)
    return {
        "workers": workers,
        "batch_count": batch_count,
        "cluster_execution_count": cluster_count,
        "fusion_execution_count": fusion_count,
        "wall_seconds": wall_seconds,
        "batches_per_second": batch_count / wall_seconds,
        "policy_nodes_per_second": (cluster_count + fusion_count) / wall_seconds,
        "peak_worker_rss_bytes": max(int(row["peak_rss_bytes"]) for row in results),
        "worker_trace_sha256s": sorted(str(row["trace_sha256"]) for row in results),
    }


class PatternEmbeddings(Mapping[tuple[int, int], np.ndarray]):
    def __init__(self, step: int) -> None:
        self.step = step
        self.vectors = []
        for index in range(4):
            vector = np.zeros(192, dtype=np.float32)
            vector[index] = 1.0
            self.vectors.append(vector)

    def __getitem__(self, key: tuple[int, int]) -> np.ndarray:
        group = (int(key[1]) // self.step // 2) % len(self.vectors)
        return self.vectors[group]

    def __iter__(self) -> Iterator[tuple[int, int]]:
        return iter(())

    def __len__(self) -> int:
        return 0


def state_profile(profile_class: str) -> dict[str, Any]:
    historical = profile_class.startswith("historical_")
    normalized_class = "stable_anchor" if historical else profile_class
    confirmation: int | str = 2 if profile_class == "historical_c2" else 1
    return {
        "proposal_profile_id": f"benchmark:{profile_class}",
        "origin": "historical_phase3_profile" if historical else "accepted_phase4_native_profile",
        "family": "eres2netv2",
        "checkpoint": "E-standard",
        "profile_class": normalized_class,
        "window_samples": 8000,
        "step_samples": 1600,
        "proposal_kind": "speaker_change_unknown",
        "confidence_semantics_id": f"benchmark:{profile_class}",
        "proposal_threshold": {"field": "change_score", "operator": ">", "value": 0.5},
        "confirmation": confirmation,
        "anchor_update": "ema" if profile_class == "historical_c1_ema" else None,
        "anchor_ema_alpha": 0.9,
        "mutual_similarity_threshold": 0.5,
        "scored_state_mode": "source_prefix",
    }


def state_trace_benchmarks() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    embeddings = PatternEmbeddings(1600)
    for profile_class in ("prototype_memory_4",):
        started = time.perf_counter()
        trace = anchor_trace(
            embeddings,
            state_profile(profile_class),
            source_session_id="source",
            audio_epoch=1,
            replay_start=0,
            warm_start=MAXIMUM_SOURCE_TAIL_SAMPLE - MAXIMUM_EPISODE_ROUTE_SAMPLES,
            tail_end=MAXIMUM_SOURCE_TAIL_SAMPLE,
        )
        elapsed = time.perf_counter() - started
        rows.append(
            {
                "profile_shape": profile_class,
                "probe_step_count": MAXIMUM_SOURCE_PREFIX_PROBE_STEPS,
                "elapsed_seconds": elapsed,
                "probe_steps_per_second": MAXIMUM_SOURCE_PREFIX_PROBE_STEPS / elapsed,
                "routed_proposal_count": len(trace["proposals"]),
                "routed_progress_count": len(trace["progress"]),
                "final_state_sha256": trace["final_state_sha256"],
            }
        )
    floor = min(float(row["probe_steps_per_second"]) for row in rows) * 0.75
    return {
        "trace_rows": rows,
        "conservative_probe_steps_per_second_floor": floor,
        "total_physical_proposal_probe_steps": TOTAL_PHYSICAL_PROPOSAL_PROBE_STEPS,
        "forecast_seconds_at_floor": TOTAL_PHYSICAL_PROPOSAL_PROBE_STEPS / floor,
    }


def worst_shape_actions(
    count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    final_actions: list[dict[str, Any]] = []
    energy: list[dict[str, Any]] = []
    for index in range(count):
        boundary = 512 + index * 1024
        observed = boundary + 512
        final_actions.append(
            {
                "final_action_id": f"final:d{index}",
                "event_id": f"d{index}",
                "origin": "detector",
                "action_kind": "add_hard_boundary",
                "boundary_source_sample": boundary,
                "observed_source_sample_at_emit": observed,
                "source_session_id": "source",
                "audio_epoch": 1,
            }
        )
        energy.append(
            {
                "candidate_id": f"energy:{index}",
                "boundary_source_sample": boundary,
                "observed_source_sample": observed,
                "change_strength": (index % 101) / 100.0,
            }
        )
    active = [
        {
            "start": 0,
            "end": count * 1024 + 4096,
            "start_observed_source_sample": 512,
            "end_observed_source_sample": count * 1024 + 4608,
        }
    ]
    return final_actions, energy, active


def control_benchmarks() -> dict[str, Any]:
    final_actions, energy, active = worst_shape_actions(HISTORICAL_MAXIMUM_EMITTABLE_POSITIONS)
    rows: list[dict[str, Any]] = []
    for kind in (
        "uniform_vad_active",
        "causal_energy_change_peak",
        "within_vad_active_position_shuffle",
    ):
        started = time.perf_counter()
        result = frequency_matched_control(
            kind,
            final_actions,
            active,
            energy_candidates=energy,
            forbidden_boundaries=[],
            seed_material="phase5-benchmark",
        )
        elapsed = time.perf_counter() - started
        rows.append(
            {
                "control_kind": kind,
                "required_action_count": HISTORICAL_MAXIMUM_EMITTABLE_POSITIONS,
                "placed_action_count": int(result["placed_hard_action_count"]),
                "status": str(result["status"]),
                "elapsed_seconds": elapsed,
                "actions_per_second": HISTORICAL_MAXIMUM_EMITTABLE_POSITIONS / elapsed,
                "action_trace_sha256": hashlib.sha256(
                    canonical_json(result["actions"]).encode("utf-8")
                ).hexdigest(),
            }
        )
    return {"worst_shape_rows": rows}


def synthetic_scoring_benchmark() -> dict[str, Any]:
    final_actions, _, _ = worst_shape_actions(HISTORICAL_MAXIMUM_EMITTABLE_POSITIONS)
    scored_end = HISTORICAL_MAXIMUM_EMITTABLE_POSITIONS * 1024 + 4096
    references = [
        ReferenceAction(
            reference_id=f"reference:{index}",
            audio_epoch=1,
            source_session_id="source",
            action_kind="hard_boundary",
            target_sample=2048 + index * 16000,
            acceptable_interval=(1536 + index * 16000, 2560 + index * 16000),
            evidence_onset_sample=2048 + index * 16000,
            scorable=True,
            primary_case=True,
            episode_pool_tag="hard_only",
        )
        for index in range(10)
    ]
    started = time.perf_counter()
    result = score_policy_episode(
        final_actions,
        [],
        references,
        [(0, scored_end, "A")],
        [],
        [],
        None,
        [],
        scored_start=0,
        scored_end=scored_end,
        episode_tag="negative_only",
    )
    elapsed = time.perf_counter() - started
    return {
        "shape_id": "synthetic_11392_actions_10_references",
        "action_count": HISTORICAL_MAXIMUM_EMITTABLE_POSITIONS,
        "reference_count": len(references),
        "singleton_interval_count": 1,
        "overlap_interval_count": 0,
        "pause_interval_count": 0,
        "unscored_interval_count": 0,
        "word_interval_count": 0,
        "word_timing_observable": False,
        "elapsed_seconds": elapsed,
        "actions_per_second": HISTORICAL_MAXIMUM_EMITTABLE_POSITIONS / elapsed,
        "score_trace_sha256": hashlib.sha256(canonical_json(result).encode("utf-8")).hexdigest(),
    }


def trusted_word_annotations(case_id: str) -> tuple[list[Any], list[dict[str, Any]], str]:
    meeting_id = case_id.removeprefix("ami_")
    words_dir = corpus_root() / "ami" / "annotations" / "words"
    expected = HISTORICAL_WORD_FILE_SHA256[case_id]
    paths = sorted(words_dir.glob(f"{meeting_id}.*.words.xml"))
    if {path.name for path in paths} != set(expected):
        raise RuntimeError("historical word annotation file set drift")
    receipts = []
    for path in paths:
        actual = sha256_file(path)
        if actual != expected[path.name]:
            raise RuntimeError("historical word annotation byte drift")
        receipts.append({"filename": path.name, "byte_sha256": actual})
    raw_words = load_ami_raw_words(words_dir, meeting_id)
    record_rows = [
        [
            row.speaker,
            row.start_time_s,
            row.end_time_s,
            row.text,
            row.ambiguous,
            row.path_index,
        ]
        for row in raw_words
    ]
    record_sha256 = hashlib.sha256(canonical_json(record_rows).encode("utf-8")).hexdigest()
    return raw_words, receipts, record_sha256


def historical_scoring_fixture(case_id: str, action_count: int) -> dict[str, Any]:
    experiment_dir = Path(__file__).resolve().parents[1]
    manifest_path = experiment_dir / "data" / "manifests" / "mixed_dev_pool.json"
    if sha256_file(manifest_path) != HISTORICAL_MANIFEST_SHA256:
        raise RuntimeError("historical scoring manifest drift")
    manifest = Phase2Manifest.load(manifest_path)
    case = next(row for row in manifest.cases if row.case_id == case_id)
    raw_words, word_file_receipts, word_record_sha256 = trusted_word_annotations(case_id)
    specs = build_reference_specs(list(case.regions), int(case.duration_samples), raw_words)
    references = references_for_episode(
        specs,
        case.case_id,
        "full_session_scoring_benchmark",
        0,
        int(case.duration_samples),
        "overlap_present",
        True,
    )
    views = annotation_views(
        list(case.regions),
        specs,
        raw_words,
        scored_start=0,
        scored_end=int(case.duration_samples),
    )
    final_actions: list[dict[str, Any]] = []
    for index in range(action_count):
        boundary = (index + 1) * int(case.duration_samples) // (action_count + 1)
        final_actions.append(
            {
                "final_action_id": f"score:{case_id}:{index:05d}",
                "action_kind": "add_hard_boundary",
                "origin": "detector",
                "boundary_source_sample": boundary,
                "observed_source_sample_at_emit": min(int(case.duration_samples), boundary + 32000),
                "source_session_id": case.case_id,
                "audio_epoch": 0,
            }
        )
    return {
        "case": case,
        "references": references,
        "views": views,
        "final_actions": final_actions,
        "action_count": action_count,
        "word_file_receipts": word_file_receipts,
        "raw_word_record_count": len(raw_words),
        "word_record_sha256": word_record_sha256,
    }


def score_historical_scoring_fixture(fixture: dict[str, Any]) -> dict[str, Any]:
    case = fixture["case"]
    case_id = str(case.case_id)
    references = fixture["references"]
    views = fixture["views"]
    final_actions = fixture["final_actions"]
    action_count = int(fixture["action_count"])
    started = time.perf_counter()
    result = score_policy_episode(
        final_actions,
        [],
        references,
        views["singleton_intervals"],
        views["pause_intervals"],
        views["overlap_intervals"],
        views["word_intervals"],
        views["unscored_intervals"],
        scored_start=0,
        scored_end=int(case.duration_samples),
        episode_tag="overlap_present",
    )
    elapsed = time.perf_counter() - started
    scorable_references = [
        row
        for row in references
        if row.scorable and row.action_kind in ("hard_boundary", "soft_overlap_marker")
    ]
    return {
        "shape_id": f"{case_id}:{action_count}",
        "case_id": case_id,
        "action_count": action_count,
        "reference_count": len(scorable_references),
        "hard_reference_count": sum(
            row.action_kind == "hard_boundary" for row in scorable_references
        ),
        "soft_reference_count": sum(
            row.action_kind == "soft_overlap_marker" for row in scorable_references
        ),
        "region_count": len(case.regions),
        "singleton_interval_count": len(views["singleton_intervals"]),
        "overlap_interval_count": len(views["overlap_intervals"]),
        "pause_interval_count": len(views["pause_intervals"]),
        "unscored_interval_count": len(views["unscored_intervals"]),
        "word_interval_count": len(views["word_intervals"]),
        "word_timing_observable": bool(views["word_timing_observable"]),
        "raw_word_record_count": int(fixture["raw_word_record_count"]),
        "word_annotation_files": fixture["word_file_receipts"],
        "word_record_sha256": fixture["word_record_sha256"],
        "scored_end_sample": int(case.duration_samples),
        "elapsed_seconds": elapsed,
        "actions_per_second": action_count / elapsed,
        "score_trace_sha256": hashlib.sha256(canonical_json(result).encode("utf-8")).hexdigest(),
    }


def historical_scoring_benchmark(case_id: str, action_count: int) -> dict[str, Any]:
    return score_historical_scoring_fixture(historical_scoring_fixture(case_id, action_count))


def exact_scoring_benchmarks() -> dict[str, Any]:
    exact_rows = [historical_scoring_benchmark(*shape) for shape in HISTORICAL_SCORING_SHAPES]
    envelope = historical_scoring_benchmark("ami_IS1008a", HISTORICAL_MAXIMUM_EMITTABLE_POSITIONS)
    return {
        "source_manifest": "data/manifests/mixed_dev_pool.json",
        "source_manifest_byte_sha256": HISTORICAL_MANIFEST_SHA256,
        "synthetic_sentinel": synthetic_scoring_benchmark(),
        "historical_exact_shapes": exact_rows,
        "joint_forecast_envelope": envelope,
    }


def control_benchmark_worker(repetitions: int) -> dict[str, Any]:
    rows = [control_benchmarks() for _ in range(repetitions)]
    return {
        "action_placement_count": HISTORICAL_MAXIMUM_EMITTABLE_POSITIONS * 3 * repetitions,
        "trace_sha256": hashlib.sha256(canonical_json(rows).encode("utf-8")).hexdigest(),
    }


def scoring_benchmark_worker(repetitions: int) -> dict[str, Any]:
    fixture = historical_scoring_fixture("ami_IS1008a", HISTORICAL_MAXIMUM_EMITTABLE_POSITIONS)
    rows = [score_historical_scoring_fixture(fixture) for _ in range(repetitions)]
    shape = rows[0]
    return {
        "action_count": HISTORICAL_MAXIMUM_EMITTABLE_POSITIONS * repetitions,
        "reference_count": shape["reference_count"],
        "region_count": shape["region_count"],
        "singleton_interval_count": shape["singleton_interval_count"],
        "overlap_interval_count": shape["overlap_interval_count"],
        "pause_interval_count": shape["pause_interval_count"],
        "unscored_interval_count": shape["unscored_interval_count"],
        "word_interval_count": shape["word_interval_count"],
        "trace_sha256": hashlib.sha256(canonical_json(rows).encode("utf-8")).hexdigest(),
    }


def verifier_recompute_unit(_: int = 0) -> dict[str, Any]:
    started = time.perf_counter()
    state = state_trace_benchmarks()
    policy = policy_grid_batch(HISTORICAL_MAXIMUM_EMITTABLE_POSITIONS)
    controls = control_benchmarks()
    scoring = exact_scoring_benchmarks()
    elapsed = time.perf_counter() - started
    state_steps = MAXIMUM_SOURCE_PREFIX_PROBE_STEPS * len(state["trace_rows"])
    control_actions = HISTORICAL_MAXIMUM_EMITTABLE_POSITIONS * 3
    scoring_shape = scoring["joint_forecast_envelope"]
    return {
        "elapsed_seconds": elapsed,
        "state_probe_step_count": state_steps,
        "policy_proposal_position_count": HISTORICAL_MAXIMUM_EMITTABLE_POSITIONS,
        "control_action_placement_count": control_actions,
        "scoring_action_count": HISTORICAL_MAXIMUM_EMITTABLE_POSITIONS,
        "scoring_reference_count": scoring_shape["reference_count"],
        "scoring_region_count": scoring_shape["region_count"],
        "scoring_singleton_interval_count": scoring_shape["singleton_interval_count"],
        "scoring_overlap_interval_count": scoring_shape["overlap_interval_count"],
        "scoring_pause_interval_count": scoring_shape["pause_interval_count"],
        "scoring_unscored_interval_count": scoring_shape["unscored_interval_count"],
        "scoring_word_interval_count": scoring_shape["word_interval_count"],
        "scoring_word_timing_observable": scoring_shape["word_timing_observable"],
        "trace_sha256": hashlib.sha256(
            canonical_json(
                {
                    "state": [row["final_state_sha256"] for row in state["trace_rows"]],
                    "policy": policy["trace_sha256"],
                    "controls": [
                        row["action_trace_sha256"] for row in controls["worst_shape_rows"]
                    ],
                    "scoring": scoring["joint_forecast_envelope"]["score_trace_sha256"],
                }
            ).encode("utf-8")
        ).hexdigest(),
    }


def identity_digest_worker(count: int) -> dict[str, Any]:
    started = time.perf_counter()
    digest_value = hashlib.sha256()
    for index in range(count):
        row = canonical_json(
            [
                f"system:{index % 4611:04d}",
                f"episode:{index % 878:03d}",
                f"node:{index % 2503:04d}",
                f"trace:{hashlib.sha256(str(index).encode('utf-8')).hexdigest()}",
            ]
        ).encode("utf-8")
        digest_value.update(len(row).to_bytes(8, "big"))
        digest_value.update(row)
    return {
        "row_count": count,
        "elapsed_seconds": time.perf_counter() - started,
        "framed_digest_sha256": digest_value.hexdigest(),
    }


def run(worker_count: int, repetitions: int) -> dict[str, Any]:
    serial_started = time.perf_counter()
    serial_result = benchmark_worker(repetitions)
    serial_wall = time.perf_counter() - serial_started
    parallel_started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        parallel_results = list(executor.map(benchmark_worker, [repetitions] * worker_count))
    parallel_wall = time.perf_counter() - parallel_started
    serial = aggregate(1, [serial_result], serial_wall)
    parallel = aggregate(worker_count, parallel_results, parallel_wall)
    historical_worst = policy_grid_batch(HISTORICAL_MAXIMUM_EMITTABLE_POSITIONS)
    historical_parallel_started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        historical_parallel_rows = list(
            executor.map(
                policy_grid_batch,
                [HISTORICAL_MAXIMUM_EMITTABLE_POSITIONS] * worker_count,
            )
        )
    historical_parallel_wall = time.perf_counter() - historical_parallel_started
    historical_parallel = {
        "workers": worker_count,
        "batch_count": worker_count,
        "proposal_position_count": HISTORICAL_MAXIMUM_EMITTABLE_POSITIONS * worker_count,
        "wall_seconds": historical_parallel_wall,
        "proposal_positions_per_second": HISTORICAL_MAXIMUM_EMITTABLE_POSITIONS
        * worker_count
        / historical_parallel_wall,
        "conservative_proposal_positions_per_second": HISTORICAL_MAXIMUM_EMITTABLE_POSITIONS
        * worker_count
        / historical_parallel_wall
        * 0.75,
        "peak_worker_rss_bytes": max(
            int(row["peak_rss_bytes"]) for row in historical_parallel_rows
        ),
        "trace_sha256s": sorted(str(row["trace_sha256"]) for row in historical_parallel_rows),
    }
    controls = control_benchmarks()
    controls_parallel_started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        control_parallel_rows = list(
            executor.map(
                control_benchmark_worker,
                [AUXILIARY_REPETITIONS_PER_WORKER] * worker_count,
            )
        )
    controls_parallel_wall = time.perf_counter() - controls_parallel_started
    control_action_placement_count = sum(
        int(row["action_placement_count"]) for row in control_parallel_rows
    )
    controls["parallel"] = {
        "workers": worker_count,
        "repetitions_per_worker": AUXILIARY_REPETITIONS_PER_WORKER,
        "action_placement_count": control_action_placement_count,
        "wall_seconds": controls_parallel_wall,
        "action_placements_per_second": control_action_placement_count / controls_parallel_wall,
        "conservative_action_placements_per_second": control_action_placement_count
        / controls_parallel_wall
        * 0.75,
    }
    scoring = exact_scoring_benchmarks()
    scoring_parallel_started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        scoring_parallel_rows = list(
            executor.map(
                scoring_benchmark_worker,
                [AUXILIARY_REPETITIONS_PER_WORKER] * worker_count,
            )
        )
    scoring_parallel_wall = time.perf_counter() - scoring_parallel_started
    scoring_dimension_fields = (
        "reference_count",
        "region_count",
        "singleton_interval_count",
        "overlap_interval_count",
        "pause_interval_count",
        "unscored_interval_count",
        "word_interval_count",
    )
    if any(
        any(
            row[field] != scoring["joint_forecast_envelope"][field]
            for field in scoring_dimension_fields
        )
        for row in scoring_parallel_rows
    ):
        raise RuntimeError("parallel scoring shape drift")
    scoring["parallel"] = {
        "workers": worker_count,
        "repetitions_per_worker": AUXILIARY_REPETITIONS_PER_WORKER,
        "action_count": sum(int(row["action_count"]) for row in scoring_parallel_rows),
        "wall_seconds": scoring_parallel_wall,
        "actions_per_second": sum(int(row["action_count"]) for row in scoring_parallel_rows)
        / scoring_parallel_wall,
        "conservative_actions_per_second": sum(
            int(row["action_count"]) for row in scoring_parallel_rows
        )
        / scoring_parallel_wall
        * 0.75,
        "shape_id": scoring["joint_forecast_envelope"]["shape_id"],
        "reference_count": scoring["joint_forecast_envelope"]["reference_count"],
        "region_count": scoring["joint_forecast_envelope"]["region_count"],
        "singleton_interval_count": scoring["joint_forecast_envelope"]["singleton_interval_count"],
        "overlap_interval_count": scoring["joint_forecast_envelope"]["overlap_interval_count"],
        "pause_interval_count": scoring["joint_forecast_envelope"]["pause_interval_count"],
        "unscored_interval_count": scoring["joint_forecast_envelope"]["unscored_interval_count"],
        "word_interval_count": scoring["joint_forecast_envelope"]["word_interval_count"],
        "word_timing_observable": scoring["joint_forecast_envelope"]["word_timing_observable"],
    }
    verifier_serial = verifier_recompute_unit()
    verifier_parallel_started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        verifier_parallel_rows = list(executor.map(verifier_recompute_unit, range(worker_count)))
    verifier_parallel_wall = time.perf_counter() - verifier_parallel_started
    if any(
        row["scoring_word_interval_count"]
        != scoring["joint_forecast_envelope"]["word_interval_count"]
        or not row["scoring_word_timing_observable"]
        for row in verifier_parallel_rows
    ):
        raise RuntimeError("parallel verifier scoring word shape drift")
    verifier_recompute = {
        "algorithm": [
            "for one audit unit, recompute proposal and progress traces from accepted embeddings and coordinates",
            "for one audit unit, recompute clusters, refractory owners, lifecycle fusion actions, controls, matches, harm, contamination, and timing",
            "compare sampled raw and derived traces while exhaustive identity, completeness, pool/block aggregates, and summary arithmetic are verified separately",
            "apply the frozen 2048-unit stratified selection plus all mandatory sentinels and deterministic failure examples",
        ],
        "representative_shapes": list(CURRENT_SENTINEL_PROPOSAL_COUNTS),
        "worst_shapes": {
            "source_prefix_probe_steps": MAXIMUM_SOURCE_PREFIX_PROBE_STEPS,
            "state_profile_shapes": 1,
            "proposal_positions": HISTORICAL_MAXIMUM_EMITTABLE_POSITIONS,
            "control_kinds": 3,
            "scoring_joint_envelope": {
                "actions": HISTORICAL_MAXIMUM_EMITTABLE_POSITIONS,
                "references": scoring["joint_forecast_envelope"]["reference_count"],
                "regions": scoring["joint_forecast_envelope"]["region_count"],
                "singleton_intervals": scoring["joint_forecast_envelope"][
                    "singleton_interval_count"
                ],
                "overlap_intervals": scoring["joint_forecast_envelope"]["overlap_interval_count"],
                "pause_intervals": scoring["joint_forecast_envelope"]["pause_interval_count"],
                "unscored_intervals": scoring["joint_forecast_envelope"]["unscored_interval_count"],
                "word_intervals": scoring["joint_forecast_envelope"]["word_interval_count"],
                "word_timing_observable": scoring["joint_forecast_envelope"][
                    "word_timing_observable"
                ],
            },
        },
        "serial": verifier_serial,
        "parallel": {
            "workers": worker_count,
            "wall_seconds": verifier_parallel_wall,
            "state_probe_step_count": sum(
                int(row["state_probe_step_count"]) for row in verifier_parallel_rows
            ),
            "policy_proposal_position_count": sum(
                int(row["policy_proposal_position_count"]) for row in verifier_parallel_rows
            ),
            "control_action_placement_count": sum(
                int(row["control_action_placement_count"]) for row in verifier_parallel_rows
            ),
            "scoring_action_count": sum(
                int(row["scoring_action_count"]) for row in verifier_parallel_rows
            ),
            "scoring_reference_count": scoring["joint_forecast_envelope"]["reference_count"],
            "scoring_region_count": scoring["joint_forecast_envelope"]["region_count"],
            "scoring_singleton_interval_count": scoring["joint_forecast_envelope"][
                "singleton_interval_count"
            ],
            "scoring_overlap_interval_count": scoring["joint_forecast_envelope"][
                "overlap_interval_count"
            ],
            "scoring_pause_interval_count": scoring["joint_forecast_envelope"][
                "pause_interval_count"
            ],
            "scoring_unscored_interval_count": scoring["joint_forecast_envelope"][
                "unscored_interval_count"
            ],
            "scoring_word_interval_count": scoring["joint_forecast_envelope"][
                "word_interval_count"
            ],
            "scoring_word_timing_observable": scoring["joint_forecast_envelope"][
                "word_timing_observable"
            ],
            "trace_sha256s": sorted(str(row["trace_sha256"]) for row in verifier_parallel_rows),
        },
    }
    identity_serial = identity_digest_worker(IDENTITY_BENCHMARK_ROWS)
    identity_parallel_started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        identity_parallel_rows = list(
            executor.map(identity_digest_worker, [IDENTITY_BENCHMARK_ROWS] * worker_count)
        )
    identity_parallel_wall = time.perf_counter() - identity_parallel_started
    identity_parallel_count = sum(int(row["row_count"]) for row in identity_parallel_rows)
    identity_digest = {
        "framing": "uint64be(canonical_row_byte_count) followed by canonical UTF-8 JSON row bytes",
        "sample_rows_per_worker": IDENTITY_BENCHMARK_ROWS,
        "total_logical_identity_rows": TOTAL_LOGICAL_IDENTITY_ROWS,
        "serial": {
            **identity_serial,
            "rows_per_second": int(identity_serial["row_count"])
            / float(identity_serial["elapsed_seconds"]),
        },
        "parallel": {
            "workers": worker_count,
            "row_count": identity_parallel_count,
            "wall_seconds": identity_parallel_wall,
            "rows_per_second": identity_parallel_count / identity_parallel_wall,
            "conservative_rows_per_second": identity_parallel_count / identity_parallel_wall * 0.75,
            "framed_digest_sha256s": sorted(
                str(row["framed_digest_sha256"]) for row in identity_parallel_rows
            ),
        },
    }
    return {
        "schema_version": "turn_episode_phase5_policy_benchmark.v7",
        "authority_sha256": AUTHORITY_SHA256,
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "physical_core_count": psutil.cpu_count(logical=False),
            "logical_cpu_count": psutil.cpu_count(logical=True),
        },
        "sentinel_contract": {
            "current_proposal_counts": list(CURRENT_SENTINEL_PROPOSAL_COUNTS),
            "current_maximum_emittable_position_count": CURRENT_MAXIMUM_EMITTABLE_POSITIONS,
            "historical_maximum_emittable_position_count": HISTORICAL_MAXIMUM_EMITTABLE_POSITIONS,
            "historical_scoring_shapes": [list(row) for row in HISTORICAL_SCORING_SHAPES],
            "historical_scoring_manifest_sha256": HISTORICAL_MANIFEST_SHA256,
            "historical_word_annotation_file_sha256": HISTORICAL_WORD_FILE_SHA256,
            "maximum_source_prefix_probe_step_count": MAXIMUM_SOURCE_PREFIX_PROBE_STEPS,
            "total_logical_emittable_position_count": TOTAL_LOGICAL_EMITTABLE_POSITIONS,
            "independent_audit_sample_size": 2048,
            "vad_action_count": len(vad_actions()),
            "cluster_grid_count": len(CLUSTER_GRID),
            "vad_grid_count": len(VAD_GRID),
            "nodes_per_batch": len(CLUSTER_GRID) * (1 + len(VAD_GRID)),
            "repetitions_per_worker": repetitions,
        },
        "serial": serial,
        "parallel": parallel,
        "parallel_speedup": parallel["batches_per_second"] / serial["batches_per_second"],
        "selected_policy_workers": worker_count,
        "conservative_parallel_batches_per_second": parallel["batches_per_second"] * 0.75,
        "historical_worst_policy_grid": historical_worst,
        "historical_worst_policy_grid_parallel": historical_parallel,
        "source_prefix_state": state_trace_benchmarks(),
        "frequency_controls": controls,
        "scoring": scoring,
        "independent_verifier_recompute": verifier_recompute,
        "logical_identity_digest": identity_digest,
        "generated_from": {
            "phase5_policy.py": sha256_file(Path(__file__).with_name("phase5_policy.py")),
            "phase5_policy_benchmark.py": sha256_file(Path(__file__).resolve()),
            "phase5_inputs.py": sha256_file(Path(__file__).with_name("phase5_inputs.py")),
            "phase5_proposals.py": sha256_file(Path(__file__).with_name("phase5_proposals.py")),
            "phase5_scoring.py": sha256_file(Path(__file__).with_name("phase5_scoring.py")),
            "build_episodes.py": sha256_file(Path(__file__).with_name("build_episodes.py")),
            "pcm_oracle.py": sha256_file(Path(__file__).with_name("pcm_oracle.py")),
            "scoring.py": sha256_file(Path(__file__).with_name("scoring.py")),
        },
        "environment": {
            "pid": os.getpid(),
            "network": "unused",
            "credentials": "unused",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = args.output or (
        Path(__file__).resolve().parents[1]
        / "results"
        / "turn_episode_v1"
        / "phase_5_policy_benchmark.json"
    )
    payload = run(args.workers, args.repetitions)
    written = atomic_write_json(output, payload)
    print(
        canonical_json(
            {
                "path": str(output),
                "content_sha256": written["content_sha256"],
                "serial_batches_per_second": written["serial"]["batches_per_second"],
                "parallel_batches_per_second": written["parallel"]["batches_per_second"],
                "parallel_speedup": written["parallel_speedup"],
                "source_prefix_probe_steps_per_second_floor": written["source_prefix_state"][
                    "conservative_probe_steps_per_second_floor"
                ],
                "historical_worst_policy_grid_seconds": written["historical_worst_policy_grid"][
                    "elapsed_seconds"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
