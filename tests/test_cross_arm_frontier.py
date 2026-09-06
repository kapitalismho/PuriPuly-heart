from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path


def _double_worker(payload):
    return {"key": payload["key"], "value": payload["value"] * 2}


def _pt(threshold, cuts, cont, miss):
    return {
        "threshold": float(threshold),
        "false_cuts_per_hour": float(cuts),
        "contamination": float(cont),
        "miss_rate": float(miss),
    }


def _h_inputs():
    from experiments.psem_state_corrected_adaptation_gate import h_arm

    calibrators = {"f0": {"slope": 1.0, "intercept": 0.0}, "candidate": {"slope": 1.0, "intercept": 0.0}}
    dev_scores = {
        "dev-ami": {
            "f0": [0.0] * 4, "candidate": [-1.0, 0.0, 1.0, 2.0],
            "target": [0.0, 0.0, 1.0, 1.0], "mapped": [True] * 4,
        },
        "dev-ali": {
            "f0": [0.0] * 4, "candidate": [-0.5, 0.5, 1.5, 2.5],
            "target": [0.0, 1.0, 1.0, 0.0], "mapped": [True] * 4,
        },
    }
    tables = {}
    for source, entry in dev_scores.items():
        conv = h_arm.dev_frontier_inputs(entry["f0"], entry["candidate"], entry["mapped"], calibrators)
        tables[source] = {}
        for horizon in (100, 300, 500):
            tables[source][horizon] = {
                "f0": (4.0, 0.2, 0.3),
                "by_threshold_raw": {t: (4.5, 0.21, 0.29) for t in conv["thresholds_raw"]},
                "by_threshold_calibrated": {t: (5.0, 0.2, 0.3) for t in conv["thresholds_calibrated"]},
            }
    grid = [0.9, 0.5]
    group_tables = {
        name: {
            hz: {
                "kinds": {
                    "raw": {
                        "thresholds": list(grid),
                        "points": [[t, 3.0, 0.21, 0.29] for t in grid],
                        "f0": [4.0, 0.2, 0.3],
                    },
                    "calibrated": {
                        "thresholds": list(grid),
                        "points": [[t, 3.5, 0.25, 0.35] for t in grid],
                        "f0": [4.0, 0.2, 0.3],
                    },
                },
            }
            for hz in (100, 300, 500)
        }
        for name in ("AMI", "AliMeeting", "pooled")
    }
    return calibrators, dev_scores, tables, group_tables


def _write_h_file(run_dir):
    from experiments.psem_state_corrected_adaptation_gate import h_arm

    calibrators, dev_scores, tables, group_tables = _h_inputs()
    doc = h_arm.run_dev_frontier(
        run_dir, {"seed": 7301}, dev_scores, tables,
        {"dev-ami": "AMI", "dev-ali": "AliMeeting"}, calibrators, group_tables, workers=1,
    )
    return run_dir / h_arm.DEV_FRONTIER_NAME, doc


def _t2_groups_obj():
    from experiments.psem_state_corrected_adaptation_gate import frontier as frontier_mod

    def _fp(t, c, co, m):
        return frontier_mod.FrontierPoint(
            threshold=float(t), false_cuts_per_hour=float(c),
            contamination=float(co), miss_rate=float(m),
        )

    groups_obj = {}
    for horizon in ("100", "300", "500"):
        groups_obj[horizon] = {}
        for kind in ("calibrated", "raw"):
            groups_obj[horizon][kind] = {}
            for name in ("macro", "ami", "alimeeting", "pooled"):
                groups_obj[horizon][kind][name] = {
                    "points": [_fp(0.8, 6.0, 0.2, 0.3), _fp(0.4, 9.0, 0.18, 0.28)],
                    "f0": _fp(0.5, 10.0, 0.25, 0.35),
                    "metrics": {"frames": 8, "kept_frames": 8},
                    "sources": ["s-a"],
                }
    return groups_obj


class HToT2FileTest(unittest.TestCase):
    def test_h_dev_file_is_canonical(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir(parents=True)
            path, doc = _write_h_file(run_dir)
            self.assertTrue(path.is_file())
            self.assertEqual(doc["artifact_role"], "issue-121-cross-arm-dev-frontier")
            self.assertEqual(doc["arm"], "R-H-SC")
            raw_pts = doc["horizons"]["100"]["ami"]["raw"]["points"]
            cal_pts = doc["horizons"]["100"]["ami"]["calibrated"]["points"]
            self.assertNotEqual(
                [p["false_cuts_per_hour"] for p in raw_pts],
                [p["false_cuts_per_hour"] for p in cal_pts],
            )

    def test_t2_reads_h_file_and_compares(self):
        from experiments.psem_state_corrected_adaptation_gate import temporal_train

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir(parents=True)
            path, _ = _write_h_file(run_dir)
            baseline = temporal_train.read_baseline_frontier(path, "R-H-SC")
        self.assertIn("macro", baseline["horizons"]["100"])
        raw_ref = baseline["horizons"]["100"]["ami"]["raw"]["reference"]
        h_points = baseline["horizons"]["100"]["ami"]["calibrated"]["points"]
        t2_points = [_pt(0.8, 3.0, 0.19, 0.28), _pt(0.4, 4.0, 0.18, 0.27)]
        corpora = {"horizon_ms": 100, "kind": "calibrated", "group": "ami", "sources": ["s-a"]}
        compared = temporal_train.compare_t2_to_h_f0(t2_points, raw_ref, corpora, h_points)
        self.assertEqual(compared["arm"], "R-T2-SC")
        self.assertEqual(compared["baseline"], "R-H-SC+F0")
        self.assertEqual(compared["budget"], 4.0)
        self.assertEqual(compared["budget_kind"], "raw_f0_at_0.5")
        self.assertIsNotNone(compared["c_envelope"])
        self.assertIsNotNone(compared["baseline_depth"])
        self.assertIsNotNone(compared["baseline_depth"]["c_envelope"])
        self.assertIsNotNone(compared["delta_vs_baseline"])
        self.assertIsNotNone(compared["delta_vs_baseline"]["c_envelope"])

    def test_wrong_preceding_arm_rejected(self):
        from experiments.psem_state_corrected_adaptation_gate import temporal_train

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir(parents=True)
            path, _ = _write_h_file(run_dir)
            with self.assertRaises(temporal_train.TemporalArmError):
                temporal_train.read_baseline_frontier(path, "R-T2-SC")
            bad = json.loads(path.read_text(encoding="utf-8"))
            bad["artifact_role"] = "issue-121-h-dev-frontier"
            path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(temporal_train.TemporalArmError):
                temporal_train.read_baseline_frontier(path, "R-H-SC")


class T2ToTAFileTest(unittest.TestCase):
    def test_t2_dev_file_is_canonical_and_feeds_ta(self):
        from experiments.psem_state_corrected_adaptation_gate import temporal_train

        groups_obj = _t2_groups_obj()
        comparisons = {
            h: {
                g: {
                    k: temporal_train.compare_t2_to_h_f0(
                        [_pt(0.8, 6.0, 0.2, 0.3)],
                        _pt(0.5, 10.0, 0.25, 0.35),
                        {"horizon_ms": int(h), "kind": k, "group": g, "sources": ["s-a"]},
                        [_pt(0.8, 7.0, 0.28, 0.38)],
                    )
                    for k in ("calibrated", "raw")
                }
                for g in ("macro", "ami", "alimeeting", "pooled")
            }
            for h in ("100", "300", "500")
        }
        gate_evidence = {"first": "macro", "horizons": {}}
        per_source = {"s-a": {"family": "ami_mix_headset", "frames": 8}}
        document = temporal_train.assemble_temporal_dev_document(
            "R-T2-SC", groups_obj, per_source, comparisons, gate_evidence, "/tmp/h.json",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dev_frontier.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            baseline = temporal_train.read_baseline_frontier(path, "R-T2-SC")
        t2_ref = baseline["horizons"]["300"]["pooled"]["raw"]["reference"]
        t2_points = baseline["horizons"]["300"]["pooled"]["raw"]["points"]
        ta_points = [_pt(0.8, 5.0, 0.19, 0.27), _pt(0.4, 8.0, 0.17, 0.26)]
        corpora = {"horizon_ms": 300, "kind": "raw", "group": "pooled", "sources": ["s-a"]}
        compared = temporal_train.compare_ta_to_t2_f0(ta_points, t2_ref, corpora, t2_points)
        self.assertEqual(compared["arm"], "R-TA-SC")
        self.assertEqual(compared["baseline"], "R-T2-SC+F0")
        self.assertEqual(compared["budget"], 10.0)
        self.assertIsNotNone(compared["baseline_depth"])
        self.assertIsNotNone(compared["delta_vs_baseline"])


class BudgetRejectionTest(unittest.TestCase):
    def test_non_half_reference_rejected(self):
        from experiments.psem_state_corrected_adaptation_gate import cross_frontier as cross_mod

        with self.assertRaises(cross_mod.CrossFrontierError):
            cross_mod.build_block([_pt(0.5, 9.0, 0.2, 0.3)], _pt(0.4, 10.0, 0.25, 0.35))

    def test_candidate_rate_as_budget_rejected(self):
        from experiments.psem_state_corrected_adaptation_gate import cross_frontier as cross_mod

        block = cross_mod.build_block([_pt(0.5, 9.0, 0.2, 0.3)], _pt(0.5, 10.0, 0.25, 0.35))
        block["budget"] = 9.0
        with self.assertRaises(cross_mod.CrossFrontierError):
            cross_mod.validate_block(block, "ami/raw@100")

    def test_compare_with_non_half_budget_rejected(self):
        from experiments.psem_state_corrected_adaptation_gate import temporal_train

        with self.assertRaises(temporal_train.TemporalArmError):
            temporal_train.compare_t2_to_h_f0(
                [_pt(0.4, 9.0, 0.2, 0.3)], _pt(0.4, 10.0, 0.25, 0.35),
                {"group": "ami"}, [_pt(0.4, 8.0, 0.2, 0.3)],
            )

    def test_missing_groups_kinds_horizons_rejected(self):
        from experiments.psem_state_corrected_adaptation_gate import cross_frontier as cross_mod
        from experiments.psem_state_corrected_adaptation_gate import temporal_train

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir(parents=True)
            path, _ = _write_h_file(run_dir)
            base = json.loads(path.read_text(encoding="utf-8"))
            for variant in ("macro", "raw", "horizon"):
                bad = json.loads(json.dumps(base))
                if variant == "macro":
                    del bad["horizons"]["100"]["macro"]
                elif variant == "raw":
                    del bad["horizons"]["300"]["ami"]["raw"]
                else:
                    del bad["horizons"]["500"]
                with self.assertRaises(cross_mod.CrossFrontierError, msg=variant):
                    cross_mod.validate_canonical(bad)
                bad_path = Path(tmp) / f"bad_{variant}.json"
                bad_path.write_text(json.dumps(bad), encoding="utf-8")
                with self.assertRaises(temporal_train.TemporalArmError, msg=variant):
                    temporal_train.read_baseline_frontier(bad_path, "R-H-SC")


class AggregationTruthTest(unittest.TestCase):
    def test_pooled_sums_primitives_not_averages(self):
        from experiments.psem_state_corrected_adaptation_gate import cross_frontier as cross_mod
        from experiments.psem_state_corrected_adaptation_gate import temporal_train

        members = [
            {
                "false_cut_count": 10,
                "active_speech_seconds": 3600.0,
                "reference_replacement_count": 10,
                "missed_replacement_count": 5,
                "exclusive_other_contamination_seconds": 360.0,
            },
            {
                "false_cut_count": 10,
                "active_speech_seconds": 32400.0,
                "reference_replacement_count": 90,
                "missed_replacement_count": 45,
                "exclusive_other_contamination_seconds": 3240.0,
            },
        ]
        summed = temporal_train._aggregate_dev_metrics(members)
        point = temporal_train._pooled_frontier_point(summed, 0.5)
        self.assertAlmostEqual(point.false_cuts_per_hour, 2.0)
        self.assertNotAlmostEqual(point.false_cuts_per_hour, (10.0 + 10.0 / 9.0) / 2.0)
        cross_point = cross_mod.pooled_point_from_sums(
            summed["false_cut_count"],
            summed["active_speech_hours"] * 3600.0,
            summed["reference_replacement_count"],
            summed["missed_replacement_count"],
            summed["contamination_seconds"],
            0.5,
        )
        self.assertAlmostEqual(cross_point["false_cuts_per_hour"], point.false_cuts_per_hour)
        self.assertAlmostEqual(cross_point["miss_rate"], point.miss_rate)

    def test_macro_requires_identical_grids(self):
        from experiments.psem_state_corrected_adaptation_gate import cross_frontier as cross_mod

        ami = [_pt(0.9, 1.0, 0.5, 0.1), _pt(0.5, 2.0, 0.2, 0.4)]
        ali = [_pt(0.9, 1.0, 0.1, 0.6), _pt(0.5, 2.0, 0.4, 0.05)]
        averaged = cross_mod.macro_average_points(ami, ali)
        self.assertEqual([p["threshold"] for p in averaged], [0.9, 0.5])
        self.assertAlmostEqual(averaged[0]["contamination"], 0.3)
        with self.assertRaises(cross_mod.CrossFrontierError):
            cross_mod.macro_average_points(ami, [_pt(0.8, 1.0, 0.1, 0.6), _pt(0.5, 2.0, 0.4, 0.05)])


class CpuContractTest(unittest.TestCase):
    def test_ordered_process_map_preserves_order(self):
        from experiments.psem_state_corrected_adaptation_gate import arm_runtime

        payloads = [{"key": f"k{i}", "value": i} for i in range(16)]
        expected = [_double_worker(p) for p in payloads]
        for workers in (1, 2, 8):
            self.assertEqual(arm_runtime.ordered_process_map(_double_worker, payloads, workers), expected)

    def test_worker_cap_defaults_to_physical_and_honors_explicit(self):
        from experiments.psem_state_corrected_adaptation_gate import arm_runtime

        self.assertEqual(
            arm_runtime.resolve_workers(None), arm_runtime.default_worker_limit()
        )
        self.assertLessEqual(
            arm_runtime.resolve_workers(None), os.cpu_count() or 1
        )
        self.assertEqual(
            arm_runtime.resolve_workers(10**6),
            max(1, min(24, os.cpu_count() or 1)),
        )
        self.assertEqual(arm_runtime.resolve_workers(1), 1)

    def test_backfill_payload_is_plain_data(self):
        payload = {
            "source_id": "s-a",
            "labels": None,
            "rows": [{"window_start_sample": 0, "window_end_sample": 480000}],
            "audio_ref": "s-a.wav",
            "waveform_sha256": "a" * 64,
            "corpus_root": "/tmp/corpus",
            "sampling_sha256": "b" * 64,
        }
        json.dumps(payload)


if __name__ == "__main__":
    unittest.main()
