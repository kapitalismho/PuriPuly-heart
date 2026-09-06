from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from experiments.psem_frozen_ceiling_gate.build_ceiling_examples import load_sessions
from experiments.psem_frozen_ceiling_gate.evaluate_ceiling import decode_scores, session_metrics
from experiments.psem_frozen_ceiling_gate.experiment_support import (
    ActivityInterval,
    exact_episode_contamination_samples,
)
from experiments.psem_state_corrected_adaptation_gate import cross_frontier as cross_mod
from experiments.psem_state_corrected_adaptation_gate import frontier as frontier_mod
from experiments.psem_state_corrected_adaptation_gate import frontier_sweep as sweep_mod
from experiments.psem_state_corrected_adaptation_gate import arm_runtime
from experiments.psem_state_corrected_adaptation_gate.h_postprocess import (
    ALI_DEV,
    AMI_DEV,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    REQUIRED_CALIB,
    _build_contamination_index,
    _indexed_contamination_samples,
    REQUIRED_DEV,
    PostprocessError,
    _event_tuple_at_threshold,
    _primitive_context,
    _primitive_row,
    _run_sweep_wave,
    build_gate1_diagnostics,
    _bootstrap_interval,
    _point_from_rows,
    _seed_for,
    fit_calib_from_export,
    load_validated_export,
    paired_source_bootstrap_v1,
    partition_hash_for,
    prepare_dev_member,
    run_postprocess,
    union_probability_grid,
    validate_export_manifest,
)
from experiments.psem_state_corrected_adaptation_gate.material import _frontier_point, mask_calibration
from experiments.psem_state_corrected_adaptation_gate.partition import CALIB_SALT
from experiments.psem_state_corrected_adaptation_gate.stages import _write_raw_npz, fit_calibrators, sha256_file

_FIT = ["ami_ES2014a", "alimeeting_R8002_M8005"]


def _partition_fields() -> dict[str, object]:
    return {"fit": list(_FIT), "salt": CALIB_SALT, "target_frac": 0.12}


def _binding() -> dict[str, object]:
    payload = _partition_fields()
    return {
        "arm": "R-H-SC",
        "seed": 7301,
        "input_hash": "a" * 64,
        "checkpoint_hash": "b" * 64,
        "partition_hash": partition_hash_for(
            payload["fit"], REQUIRED_CALIB, payload["salt"], payload["target_frac"]
        ),
        "weights_hash": "d" * 64,
        "code_hash": "e" * 64,
        "optimizer_contract": dict(arm_runtime.OPTIMIZER_CONTRACT),
    }


def _calib_arrays() -> dict[str, list]:
    target = [0.0, 1.0] * 8
    return {
        "f0_raw": [-1.0, 1.2] * 8,
        "cand_raw": [-0.4, 1.8] * 8,
        "target": target,
        "valid": [True] * 16,
        "mapped": [True] * 16,
    }


def _dev_levels(session: object, levels: int = 3) -> dict[str, list]:
    count = int(np.asarray(session.starts).reshape(-1).shape[0])
    rng = np.random.RandomState(7301 + (sum(ord(ch) for ch in session.source_id) % 97))
    picks = rng.choice(np.linspace(-2.0, 2.5, levels), size=count)
    cand = np.where(np.asarray(session.target), picks + 0.6, picks - 0.4)
    return {
        "f0_raw": [float(v) for v in picks],
        "cand_raw": [float(v) for v in cand],
        "target": [float(v) for v in np.asarray(session.target)],
        "valid": [bool(v) for v in np.asarray(session.valid)],
        "mapped": [True] * count,
    }


def _write_export(
    root: Path,
    sessions: dict[str, object],
    *,
    levels: int = 3,
    levels_by_source: dict[str, int] | None = None,
    extra_eval: bool = False,
) -> Path:
    export_dir = Path(root) / "gpu_export"
    export_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    calib_meta: dict[str, object] = {}
    for source_id in REQUIRED_CALIB:
        rel = f"calib_{source_id}.npz"
        path = export_dir / rel
        arrays = _calib_arrays()
        _write_raw_npz(path, arrays["f0_raw"], arrays["cand_raw"], arrays["target"], arrays["valid"], arrays["mapped"])
        digest = sha256_file(path)
        files[rel] = digest
        calib_meta[source_id] = {
            "file": rel,
            "frames": 16,
            "family": "ami_mix_headset" if source_id.startswith("ami_") else "alimeeting_far_ch0",
            "mapping_mapped": 16,
            "mapping_total": 16,
            "unmapped_frames": 0,
            "kept_frames": 16,
            "coverage": {"frames": 16, "kept": 16, "positive": 8, "negative": 8},
            "infer_seconds": 0.1,
            "sha256": digest,
        }
    dev_meta: dict[str, object] = {}
    for source_id in REQUIRED_DEV:
        session = sessions[source_id]
        rel = f"dev_{source_id}.npz"
        path = export_dir / rel
        source_levels = levels if levels_by_source is None else int(levels_by_source.get(source_id, levels))
        arrays = _dev_levels(session, levels=source_levels)
        _write_raw_npz(path, arrays["f0_raw"], arrays["cand_raw"], arrays["target"], arrays["valid"], arrays["mapped"])
        digest = sha256_file(path)
        files[rel] = digest
        frames = len(arrays["target"])
        mapped = sum(arrays["mapped"])
        dev_meta[source_id] = {
            "file": rel,
            "frames": frames,
            "family": "ami_mix_headset" if source_id.startswith("ami_") else "alimeeting_far_ch0",
            "mapping_mapped": mapped,
            "mapping_total": frames,
            "unmapped_frames": frames - mapped,
            "kept_frames": mapped,
            "coverage": {"frames": frames, "kept": mapped},
            "infer_seconds": 0.2,
            "sha256": digest,
        }
    calib_sources = list(REQUIRED_CALIB)
    dev_sources = list(REQUIRED_DEV)
    if extra_eval:
        dev_sources = list(dev_sources) + ["ami_EVAL_fake"]
    manifest = {
        "artifact_role": "issue-121-h-gpu-export",
        "arm": "R-H-SC",
        "seed": 7301,
        "binding": _binding(),
        **_partition_fields(),
        "calib_sources": calib_sources,
        "dev_sources": dev_sources,
        "calib": calib_meta,
        "dev": dev_meta,
        "files": files,
        "frozen_cache_inference_seconds": {},
        "training_metrics": {"path": "training_metrics.json", "sha256": "0" * 64},
    }
    (export_dir / "gpu_export_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return export_dir


def _dev_sessions() -> dict[str, object]:
    return {
        session.source_id: session
        for session in load_sessions()
        if session.source_id in REQUIRED_DEV
    }


class ExportContractTest(unittest.TestCase):
    def test_rejects_incomplete_eval_corrupt_and_truncated(self) -> None:
        sessions = _dev_sessions()
        self.assertEqual(set(sessions), set(REQUIRED_DEV))
        with tempfile.TemporaryDirectory() as tmp:
            export_dir = _write_export(Path(tmp), sessions, levels=2)
            manifest_path = export_dir / "gpu_export_manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["dev_sources"] = list(REQUIRED_DEV)[:-1]
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(PostprocessError):
                validate_export_manifest(export_dir)
        with tempfile.TemporaryDirectory() as tmp:
            export_dir = _write_export(Path(tmp), sessions, levels=2, extra_eval=True)
            with self.assertRaises(PostprocessError):
                validate_export_manifest(export_dir)
        with tempfile.TemporaryDirectory() as tmp:
            export_dir = _write_export(Path(tmp), sessions, levels=2)
            npz = export_dir / f"dev_{REQUIRED_DEV[0]}.npz"
            npz.write_bytes(npz.read_bytes()[:40])
            with self.assertRaises(PostprocessError):
                load_validated_export(export_dir, sessions)
        with tempfile.TemporaryDirectory() as tmp:
            export_dir = _write_export(Path(tmp), sessions, levels=2)
            npz = export_dir / f"dev_{REQUIRED_DEV[0]}.npz"
            digest = "f" * 64
            payload = json.loads((export_dir / "gpu_export_manifest.json").read_text(encoding="utf-8"))
            payload["files"][npz.name] = digest
            payload["dev"][REQUIRED_DEV[0]]["sha256"] = digest
            (export_dir / "gpu_export_manifest.json").write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(PostprocessError):
                load_validated_export(export_dir, sessions)

    def test_rejects_arbitrary_or_unverified_partition_hash(self) -> None:
        sessions = _dev_sessions()
        with tempfile.TemporaryDirectory() as tmp:
            export_dir = _write_export(Path(tmp), sessions, levels=2)
            manifest_path = export_dir / "gpu_export_manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["binding"]["partition_hash"] = "0" * 64
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(PostprocessError):
                validate_export_manifest(export_dir)
        with tempfile.TemporaryDirectory() as tmp:
            export_dir = _write_export(Path(tmp), sessions, levels=2)
            manifest_path = export_dir / "gpu_export_manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            for field in ("fit", "salt", "target_frac"):
                payload["binding"].pop(field, None)
                payload.pop(field, None)
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(PostprocessError) as caught:
                validate_export_manifest(export_dir)
            self.assertIn("fit", str(caught.exception))
            self.assertIn("salt", str(caught.exception))
            self.assertIn("target_frac", str(caught.exception))


def _diag_metrics(tag: str) -> dict[str, object]:
    return {
        "active_speech_seconds": 3600.0,
        "reference_replacement_count": 10,
        "exclusive_other_contamination_seconds_per_active_speech_hour": 1.0,
        "exclusive_other_contamination_seconds": 1.0,
        "missed_replacement_count": 1,
        "false_cut_count": 1,
        "predicted_cut_count": 2,
        "matched_replacement_count": 1,
        "replacement_emit_delay_ms": {"p50": 0.0, "p90": 0.0},
        "topology": {"tag": tag},
    }


def _diag_point(threshold: float = 0.4) -> dict[str, float]:
    return {
        "threshold": float(threshold),
        "false_cuts_per_hour": 1.0,
        "contamination": 0.1,
        "miss_rate": 0.1,
    }


def _diag_block() -> dict[str, object]:
    return {
        "reference": {"threshold": 0.5, "false_cuts_per_hour": 2.0, "contamination": 0.2, "miss_rate": 0.2},
        "budget": 2.0,
        "useful": True,
        "c_envelope": _diag_point(0.4),
        "m_envelope": _diag_point(0.3),
        "points": [_diag_point(0.4), _diag_point(0.3)],
    }


def _diag_fixture() -> tuple[dict, dict, dict, dict, dict]:
    members = {}
    selected = {}
    frontier = {"horizons": {}}
    for source_id in REQUIRED_DEV:
        members[source_id] = {
            "sidecar": {},
            "mapped": [True],
            "frames": 1,
            "kept": [0],
            "raw_ap": 0.0,
            "f0_cal_nll": 0.0,
            "f0_cal_brier": 0.0,
            "candidate_cal_nll": 0.0,
            "candidate_cal_brier": 0.0,
        }
    for horizon_ms in (100, 300, 500):
        horizon_key = str(horizon_ms)
        selected[horizon_key] = {}
        frontier["horizons"][horizon_key] = {}
        for group in ("macro", "ami", "alimeeting", "pooled"):
            frontier["horizons"][horizon_key][group] = {
                "raw": _diag_block(),
                "calibrated": _diag_block(),
            }
        for kind in ("raw", "calibrated"):
            sources = {}
            for source_id in REQUIRED_DEV:
                sources[source_id] = {
                    "f0": {"threshold": 0.5, "metrics": _diag_metrics("f0")},
                    "c_envelope": {"threshold": 0.4, "metrics": _diag_metrics("c")},
                    "m_envelope": {"threshold": 0.3, "metrics": _diag_metrics("m")},
                }
            selected[horizon_key][kind] = {
                "sources": sources,
                "topology": {
                    "c_envelope": {"from": "c"},
                    "m_envelope": {},
                },
            }
    calibration = {
        "f0": {"slope": 1.0, "intercept": 0.0, "nll": 0.0, "brier": 0.0, "ap": 0.0, "raw_nll": 0.0, "raw_brier": 0.0},
        "candidate": {
            "slope": 1.0,
            "intercept": 0.0,
            "nll": 0.0,
            "brier": 0.0,
            "ap": 0.0,
            "raw_nll": 0.0,
            "raw_brier": 0.0,
        },
        "frames": 1,
        "sources": list(REQUIRED_CALIB),
    }
    export_meta = {"binding": {}, "seed": 7301}
    return members, frontier, selected, calibration, export_meta


class EnvelopeIdentityTest(unittest.TestCase):
    def test_missing_m_metrics_raise_and_empty_m_topology_is_not_c(self) -> None:
        members, frontier, selected, calibration, export_meta = _diag_fixture()
        ok = build_gate1_diagnostics(members, frontier, selected, calibration, export_meta)
        empty_m = ok["horizons"]["100"]["raw"]["envelopes"]["m_envelope"]["topology"]
        self.assertEqual(empty_m, {})
        self.assertNotEqual(empty_m, ok["horizons"]["100"]["raw"]["envelopes"]["c_envelope"]["topology"])
        missing = selected["100"]["raw"]["sources"][REQUIRED_DEV[0]]
        del missing["m_envelope"]
        with self.assertRaises(PostprocessError) as caught:
            build_gate1_diagnostics(members, frontier, selected, calibration, export_meta)
        message = str(caught.exception)
        self.assertIn(REQUIRED_DEV[0], message)
        self.assertIn("m_envelope", message)


class BootstrapIdentityTest(unittest.TestCase):
    def test_paired_source_bootstrap_seed_and_resamples(self) -> None:
        deltas = {source_id: float(index - 5) / 10.0 for index, source_id in enumerate(REQUIRED_DEV)}
        seed = _seed_for("calibrated", 500, 0, 0, None)
        self.assertEqual(seed, int(BOOTSTRAP_SEED) + 200 + 2 * 40)
        got = _bootstrap_interval(deltas, seed)
        want = paired_source_bootstrap_v1(dict(sorted(deltas.items())), seed=seed, resamples=BOOTSTRAP_RESAMPLES)
        self.assertEqual(got["lower"], want["lower"])
        self.assertEqual(got["upper"], want["upper"])
        self.assertEqual(got["replicate_estimates_sha256"], want["replicate_estimates_sha256"])
        self.assertEqual(got["resamples"], 2000)
        self.assertEqual(got["seed"], seed)
        self.assertEqual(got["algorithm"], "paired_source_bootstrap_v1")
        self.assertEqual(got["aggregation"], "source_mean_not_pooled_rate")
        ami_seed = _seed_for("calibrated", 500, 0, 0, 0)
        ali_seed = _seed_for("calibrated", 500, 0, 0, 1)
        self.assertEqual(ami_seed, seed + 20)
        self.assertEqual(ali_seed, seed + 30)


class PrimitiveParityTest(unittest.TestCase):
    def test_scalar_f0_cm_parity_on_truncated_real_session(self) -> None:
        full = next(session for session in load_sessions() if session.source_id == "ami_ES2009a")
        count = 900
        dev = replace(
            full,
            starts=full.starts[:count],
            ends=full.ends[:count],
            episode_ids=full.episode_ids[:count],
            episode_speakers=full.episode_speakers[:count],
            frontiers=full.frontiers[:count],
            probabilities=full.probabilities[:count],
            alive=full.alive[:count],
            reset=full.reset[:count],
            valid=full.valid[:count],
            masked=full.masked[:count],
            speech_present=full.speech_present[:count],
            anchor_present=full.anchor_present[:count],
            overlap=full.overlap[:count],
        )
        arrays = _dev_levels(dev, levels=4)
        arrays["session"] = dev
        kept, _ = mask_calibration(arrays["target"], arrays["valid"], arrays["mapped"])
        f0_fit, cand_fit = fit_calibrators(
            [arrays["f0_raw"][i] for i in kept],
            [arrays["cand_raw"][i] for i in kept],
            [arrays["target"][i] for i in kept],
        )
        member = prepare_dev_member(dev.source_id, arrays, f0_fit, cand_fit)
        scores = np.asarray(member["cand_raw_prob"], dtype=np.float64)
        f0_scores = np.asarray(member["f0_prob"], dtype=np.float64)
        grid = union_probability_grid([member["cand_raw_prob"]])
        horizon_ms = 300
        f0_events = decode_scores(dev, f0_scores, threshold=0.5, confirmation_ms=horizon_ms)
        f0_metrics = session_metrics(dev, f0_events)
        f0_point = _frontier_point(f0_metrics, 0.5)
        points = []
        primitive_rows = []
        for threshold in grid:
            events = decode_scores(dev, scores, threshold=float(threshold), confirmation_ms=horizon_ms)
            metrics = session_metrics(dev, events)
            points.append(_frontier_point(metrics, float(threshold)))
            primitive_rows.append(
                {
                    "threshold": float(threshold),
                    "false_cut_count": int(metrics["false_cut_count"]),
                    "active_speech_seconds": float(metrics["active_speech_seconds"]),
                    "reference_replacement_count": int(metrics["reference_replacement_count"]),
                    "missed_replacement_count": int(metrics["missed_replacement_count"]),
                    "exclusive_other_contamination_seconds": float(
                        metrics["exclusive_other_contamination_seconds"]
                    ),
                }
            )
        envelopes = frontier_mod.select_envelopes(f0_point, points)
        primitive_f0 = _point_from_rows(
            [
                {
                    "false_cut_count": int(f0_metrics["false_cut_count"]),
                    "active_speech_seconds": float(f0_metrics["active_speech_seconds"]),
                    "reference_replacement_count": int(f0_metrics["reference_replacement_count"]),
                    "missed_replacement_count": int(f0_metrics["missed_replacement_count"]),
                    "exclusive_other_contamination_seconds": float(
                        f0_metrics["exclusive_other_contamination_seconds"]
                    ),
                }
            ],
            0.5,
        )
        self.assertAlmostEqual(primitive_f0["contamination"], f0_point.contamination)
        self.assertAlmostEqual(primitive_f0["miss_rate"], f0_point.miss_rate)
        self.assertAlmostEqual(primitive_f0["false_cuts_per_hour"], f0_point.false_cuts_per_hour)
        indexed = {float(row["threshold"]): row for row in primitive_rows}
        for name in ("c_envelope", "m_envelope"):
            chosen = envelopes[name]
            if chosen is None:
                continue
            derived = _point_from_rows([indexed[float(chosen.threshold)]], float(chosen.threshold))
            self.assertAlmostEqual(derived["contamination"], chosen.contamination)
            self.assertAlmostEqual(derived["miss_rate"], chosen.miss_rate)
            self.assertAlmostEqual(derived["false_cuts_per_hour"], chosen.false_cuts_per_hour)


    def test_swept_six_primitive_rows_match_direct_metrics(self) -> None:
        full = next(session for session in load_sessions() if session.source_id == "ami_ES2009a")
        count = 900
        dev = replace(
            full,
            starts=full.starts[:count],
            ends=full.ends[:count],
            episode_ids=full.episode_ids[:count],
            episode_speakers=full.episode_speakers[:count],
            frontiers=full.frontiers[:count],
            probabilities=full.probabilities[:count],
            alive=full.alive[:count],
            reset=full.reset[:count],
            valid=full.valid[:count],
            masked=full.masked[:count],
            speech_present=full.speech_present[:count],
            anchor_present=full.anchor_present[:count],
            overlap=full.overlap[:count],
        )
        scores = (np.arange(count, dtype=np.float64) % 7.0) / 6.0
        scores[0] = float("-inf")
        grid, event_rows = sweep_mod.sweep_threshold_events(dev, scores.tolist(), 300)
        context = _primitive_context(dev)
        cache: dict = {}
        for threshold, events in zip(grid, event_rows):
            got = _primitive_row(context, cache, 300, events, float(threshold))
            direct = session_metrics(
                dev,
                decode_scores(dev, scores, threshold=float(threshold), confirmation_ms=300),
            )
            self.assertEqual(
                got,
                {
                    "threshold": float(threshold),
                    "false_cut_count": int(direct["false_cut_count"]),
                    "active_speech_seconds": float(direct["active_speech_seconds"]),
                    "reference_replacement_count": int(direct["reference_replacement_count"]),
                    "missed_replacement_count": int(direct["missed_replacement_count"]),
                    "exclusive_other_contamination_seconds": float(
                        direct["exclusive_other_contamination_seconds"]
                    ),
                },
            )
        f0_grid, f0_events = sweep_mod.sweep_threshold_events(dev, scores.tolist(), 300)
        f0_tuple = _event_tuple_at_threshold(f0_grid, f0_events, 0.5)
        got_f0 = _primitive_row(context, cache, 300, f0_tuple, 0.5)
        direct_f0 = session_metrics(dev, decode_scores(dev, scores, threshold=0.5, confirmation_ms=300))
        self.assertEqual(got_f0["false_cut_count"], int(direct_f0["false_cut_count"]))
        self.assertEqual(got_f0["reference_replacement_count"], int(direct_f0["reference_replacement_count"]))
        self.assertEqual(got_f0["missed_replacement_count"], int(direct_f0["missed_replacement_count"]))
        self.assertEqual(got_f0["active_speech_seconds"], float(direct_f0["active_speech_seconds"]))
        self.assertEqual(
            got_f0["exclusive_other_contamination_seconds"],
            float(direct_f0["exclusive_other_contamination_seconds"]),
        )


    def test_wave_primitives_match_legacy_exact_rows(self) -> None:
        full = next(session for session in load_sessions() if session.source_id == "ami_ES2009a")
        count = 240
        dev = replace(
            full,
            starts=full.starts[:count],
            ends=full.ends[:count],
            episode_ids=full.episode_ids[:count],
            episode_speakers=full.episode_speakers[:count],
            frontiers=full.frontiers[:count],
            probabilities=full.probabilities[:count],
            alive=full.alive[:count],
            reset=full.reset[:count],
            valid=full.valid[:count],
            masked=full.masked[:count],
            speech_present=full.speech_present[:count],
            anchor_present=full.anchor_present[:count],
            overlap=full.overlap[:count],
        )
        scores = ((np.arange(count, dtype=np.float64) % 5.0) / 4.0).tolist()
        member_scores = {
            "ami_ES2009a": {
                "dev": dev,
                "scores": {"raw": scores, "calibrated": scores, "f0": scores},
            }
        }
        wave_grids = {
            "ami_ES2009a": {
                "raw": sorted(set(scores), reverse=True),
                "calibrated": sorted(set(scores), reverse=True),
                "f0": [0.5],
            }
        }
        tasks = cross_mod.plan_exact_tasks(wave_grids, list(frontier_mod.HORIZONS_MS))
        swept, _ = _run_sweep_wave(member_scores, wave_grids["ami_ES2009a"], 1)
        legacy, _ = cross_mod.run_exact_wave(member_scores, tasks, 1)
        self.assertEqual(swept, legacy)

    def test_global_union_rows_match_legacy_for_distinct_members(self) -> None:
        sessions = _dev_sessions()
        source_ids = ("ami_ES2009a", "ami_ES2009b")
        members = {}
        local_scores = {
            source_ids[0]: ((np.arange(240, dtype=np.float64) % 5.0) / 4.0).tolist(),
            source_ids[1]: ((np.arange(240, dtype=np.float64) % 3.0) / 2.0 + 0.1).tolist(),
        }
        for source_id in source_ids:
            full = sessions[source_id]
            count = 240
            dev = replace(
                full,
                starts=full.starts[:count],
                ends=full.ends[:count],
                episode_ids=full.episode_ids[:count],
                episode_speakers=full.episode_speakers[:count],
                frontiers=full.frontiers[:count],
                probabilities=full.probabilities[:count],
                alive=full.alive[:count],
                reset=full.reset[:count],
                valid=full.valid[:count],
                masked=full.masked[:count],
                speech_present=full.speech_present[:count],
                anchor_present=full.anchor_present[:count],
                overlap=full.overlap[:count],
            )
            scores = local_scores[source_id]
            members[source_id] = {
                "dev": dev,
                "scores": {"raw": scores, "calibrated": scores, "f0": scores},
            }
        global_grid = sorted(set(local_scores[source_ids[0]] + local_scores[source_ids[1]]), reverse=True)
        wave_grids = {
            source_id: {"raw": global_grid, "calibrated": global_grid, "f0": [0.5]}
            for source_id in source_ids
        }
        tasks = cross_mod.plan_exact_tasks(wave_grids, list(frontier_mod.HORIZONS_MS))
        swept, _ = _run_sweep_wave(members, wave_grids[source_ids[0]], 1)
        legacy, _ = cross_mod.run_exact_wave(members, tasks, 1)
        self.assertEqual(swept, legacy)
        for source_id in source_ids:
            for kind in ("raw", "calibrated"):
                for horizon_ms in frontier_mod.HORIZONS_MS:
                    self.assertEqual(
                        [float(row["threshold"]) for row in swept[source_id][kind][horizon_ms]],
                        global_grid,
                    )



    def test_indexed_contamination_matches_exact_interval_oracle(self) -> None:
        intervals = (
            ActivityInterval(0, 100, ("A",), False),
            ActivityInterval(100, 220, ("B",), False),
            ActivityInterval(220, 300, (), False),
            ActivityInterval(300, 420, ("B", "C"), True),
            ActivityInterval(420, 500, ("C",), False),
            ActivityInterval(500, 500, ("B",), False),
            ActivityInterval(500, 620, ("B",), False),
        )
        context = {"contamination_index": _build_contamination_index(intervals, ("A", "B", "C"))}
        cases = (
            ("A", -20, 50),
            ("A", 50, 180),
            ("A", 180, 430),
            ("A", 430, 700),
            ("B", 0, 100),
            ("B", 100, 620),
            ("C", 0, 620),
            ("C", 300, 420),
            ("A", 620, 620),
            ("B", 620, 500),
        )
        for anchor, start, end in cases:
            expected = exact_episode_contamination_samples(
                intervals,
                anchor_speaker=anchor,
                start_sample=start,
                end_sample=end,
            )
            self.assertEqual(
                _indexed_contamination_samples(context, anchor, start, end),
                expected,
                (anchor, start, end),
            )


class FullPostprocessTest(unittest.TestCase):
    def test_run_postprocess_full10_small_levels_and_determinism(self) -> None:
        sessions = _dev_sessions()
        self.assertEqual(len(sessions), 10)
        self.assertEqual(len(AMI_DEV), 7)
        self.assertEqual(len(ALI_DEV), 3)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            export_dir = _write_export(
                root,
                sessions,
                levels=2,
                levels_by_source={"ami_ES2009a": 2, "ami_ES2009b": 3},
            )
            out_a = root / "out-a"
            out_b = root / "out-b"
            result_a = run_postprocess(export_dir, out_a, workers=1)
            result_b = run_postprocess(export_dir, out_b, workers=2)
            self.assertFalse(result_a["gate_receipt_emitted"])
            self.assertEqual(result_a["dev_sources"], list(REQUIRED_DEV))
            self.assertEqual(result_a["calib_sources"], list(REQUIRED_CALIB))
            for name in ("calibration_metrics.json", "dev_frontier.json", "gate1_diagnostics.json", "gate1_decision_evidence.md"):
                self.assertTrue((out_a / name).is_file(), name)
            calib = json.loads((out_a / "calibration_metrics.json").read_text(encoding="utf-8"))
            packed = {source_id: {"target": _calib_arrays()["target"], "valid": _calib_arrays()["valid"], "mapped": _calib_arrays()["mapped"], "f0_raw": _calib_arrays()["f0_raw"], "cand_raw": _calib_arrays()["cand_raw"]} for source_id in REQUIRED_CALIB}
            f0_fit, cand_fit, _ = fit_calib_from_export(packed)
            direct_f0, direct_cand = fit_calibrators(
                _calib_arrays()["f0_raw"] * 11,
                _calib_arrays()["cand_raw"] * 11,
                _calib_arrays()["target"] * 11,
            )
            self.assertEqual(float(f0_fit["slope"]), float(direct_f0["slope"]))
            self.assertEqual(float(cand_fit["slope"]), float(direct_cand["slope"]))
            self.assertEqual(float(calib["f0"]["slope"]), float(direct_f0["slope"]))
            self.assertGreater(float(calib["f0"]["slope"]), 0.0)
            self.assertGreater(float(calib["candidate"]["slope"]), 0.0)
            frontier = json.loads((out_a / "dev_frontier.json").read_text(encoding="utf-8"))
            cross_mod.validate_canonical(frontier)
            self.assertEqual(frontier["arm"], "R-H-SC")
            self.assertEqual(sorted(frontier["sources"]), sorted(REQUIRED_DEV))
            self.assertNotIn("OPEN-T2", json.dumps(frontier))
            diagnostics = json.loads((out_a / "gate1_diagnostics.json").read_text(encoding="utf-8"))
            self.assertTrue(diagnostics["human_adjudication_required"])
            self.assertFalse(diagnostics["gate_receipt_emitted"])
            self.assertFalse(diagnostics["t2_opened"])
            self.assertFalse(diagnostics["eval_opened"])
            self.assertEqual(diagnostics["timing_criterion_ms"], 80.0)
            report = (out_a / "gate1_decision_evidence.md").read_text(encoding="utf-8")
            self.assertIn("not a gate receipt", report)
            self.assertIn("80 ms", report)
            for horizon_ms in frontier_mod.HORIZONS_MS:
                for kind in ("raw", "calibrated"):
                    node = diagnostics["horizons"][str(horizon_ms)][kind]
                    for envelope_name in ("c_envelope", "m_envelope"):
                        env = node["envelopes"].get(envelope_name)
                        if not env:
                            continue
                        self.assertEqual(set(env["meetings"]), set(REQUIRED_DEV))
                        self.assertEqual(set(env["leave_one_meeting_out"]), set(REQUIRED_DEV))
                        boot = env["bootstrap"]["contamination"]["pooled_source_mean"]
                        self.assertEqual(boot["seed"], _seed_for(kind, int(horizon_ms), 0 if envelope_name == "c_envelope" else 1, 0, None))
                        self.assertEqual(boot["resamples"], 2000)
                        self.assertEqual(boot["algorithm"], "paired_source_bootstrap_v1")
                        self.assertIn("source_mean_not_pooled_rate", boot["aggregation"])
                        self.assertIn("topology", env)
                        points = diagnostics["horizons"][str(horizon_ms)][kind]
                        self.assertIn("ranking", points)
            frontier_b = json.loads((out_b / "dev_frontier.json").read_text(encoding="utf-8"))
            diagnostics_b = json.loads((out_b / "gate1_diagnostics.json").read_text(encoding="utf-8"))
            self.assertEqual(frontier["horizons"], frontier_b["horizons"])
            self.assertEqual(frontier["sources"], frontier_b["sources"])
            self.assertEqual(diagnostics["horizons"], diagnostics_b["horizons"])
            self.assertEqual(
                (out_a / "calibration_metrics.json").read_text(encoding="utf-8"),
                (out_b / "calibration_metrics.json").read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
