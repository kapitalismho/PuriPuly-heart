from __future__ import annotations

import importlib.util
import json
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments.psem_state_corrected_adaptation_gate import (
    calibrate as calibrate_mod,
)
from experiments.psem_state_corrected_adaptation_gate import (
    frontier as frontier_mod,
)
from experiments.psem_state_corrected_adaptation_gate.material import (
    MaterialBlockedError,
    MaterialError,
    build_horizon_result,
    mask_calibration,
)
from experiments.psem_state_corrected_adaptation_gate.stages import (
    fit_calibrators,
    load_stage_targets,
    resolve_dev_session,
    restore_authority,
    run_stage_b,
    score_dev_frontiers,
    serialize_authority,
    sha256_file,
    verify_bundle_manifest,
)
from experiments.psem_state_corrected_adaptation_gate.receipts import write_json
from experiments.psem_state_corrected_adaptation_gate.lifecycle import AnchorEpisode

HAS_TORCH = importlib.util.find_spec("torch") is not None


def _snapshot_sessions():
    from experiments.psem_frozen_ceiling_gate.build_ceiling_examples import (
        load_sessions,
    )

    return load_sessions()


def _snapshot_available() -> bool:
    try:
        sessions = _snapshot_sessions()
    except Exception:
        return False
    return any(
        s.source_family == "ami_mix_headset" and s.role == "dev" for s in sessions
    )


class BundleVerifyTest(unittest.TestCase):
    def test_valid_manifest_verifies(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "targets").mkdir()
            payload = root / "targets" / "s.json"
            payload.write_text(json.dumps({"a": 1}), encoding="utf-8")
            manifest = {
                "artifact_role": "probe",
                "files": {"targets/s.json": sha256_file(payload)},
            }
            write_json(root / "stage_a_manifest.json", manifest)
            verified = verify_bundle_manifest(root, "stage_a_manifest.json")
            self.assertEqual(verified["artifact_role"], "probe")

    def test_tampered_file_rejected(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "targets").mkdir()
            payload = root / "targets" / "s.json"
            payload.write_text(json.dumps({"a": 1}), encoding="utf-8")
            manifest = {
                "artifact_role": "probe",
                "files": {"targets/s.json": sha256_file(payload)},
            }
            write_json(root / "stage_a_manifest.json", manifest)
            payload.write_text(json.dumps({"a": 2}), encoding="utf-8")
            with self.assertRaises(MaterialError):
                verify_bundle_manifest(root, "stage_a_manifest.json")

    def test_tampered_manifest_rejected(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {"artifact_role": "probe", "files": {}}
            path = root / "stage_a_manifest.json"
            write_json(path, manifest)
            text = path.read_text(encoding="utf-8").replace("probe", "probes")
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(MaterialError):
                verify_bundle_manifest(root, "stage_a_manifest.json")


class AuthorityRoundTripTest(unittest.TestCase):
    def test_serialize_restore_exact(self):
        authority = SimpleNamespace(
            num_frames=4,
            episodes=(AnchorEpisode("ep1", "spkA", 0, 4),),
            y_anchor=(1.0, 1.0, 0.0, 0.0),
            y_replace=(0.0, 0.0, 1.0, 0.0),
            valid=(True, True, True, False),
            ledger={"opportunities": [{"id": "ep1"}]},
        )
        entry = {
            "authority": authority,
            "multiplicity": [0, 0, 2, 0],
            "episode_ids": ["ep1", "ep1", None, "ep1"],
            "intervals": [{"start_sample": 0}],
        }
        payload = serialize_authority("src-x", entry)
        text = json.dumps(payload, sort_keys=True)
        restored = restore_authority(json.loads(text))
        self.assertEqual(restored.num_frames, 4)
        self.assertEqual(list(restored.y_anchor), [1.0, 1.0, 0.0, 0.0])
        self.assertEqual(list(restored.y_replace), [0.0, 0.0, 1.0, 0.0])
        self.assertEqual(list(restored.valid), [True, True, True, False])
        self.assertEqual(payload["episode_ids"], ["ep1", "ep1", None, "ep1"])
        self.assertEqual(payload["multiplicity"], [0, 0, 2, 0])
        self.assertEqual(payload["episodes"][0]["anchor_speaker"], "spkA")


class StageBHashGateTest(unittest.TestCase):
    def _payload(self) -> dict:
        return {
            "source_id": "s",
            "num_frames": 2,
            "y_anchor": [1.0, 0.0],
            "y_replace": [0.0, 1.0],
            "valid": [True, True],
            "multiplicity": [0, 1],
            "episode_ids": ["ep1", None],
            "audio_ref": "a.wav",
            "waveform_sha256": "0" * 64,
        }

    def _bundle(self, root: Path) -> Path:
        (root / "targets").mkdir(exist_ok=True)
        payload = root / "targets" / "s.json"
        payload.write_text(json.dumps(self._payload(), sort_keys=True), encoding="utf-8")
        manifest = {
            "artifact_role": "probe",
            "slice_sources": ["s"],
            "calib_sources": [],
            "class_weights": {},
            "targets": {
                "s": {
                    "file": "targets/s.json",
                    "sha256": sha256_file(payload),
                    "num_frames": 2,
                    "audio_ref": "a.wav",
                    "waveform_sha256": "0" * 64,
                }
            },
            "files": {"targets/s.json": sha256_file(payload)},
        }
        write_json(root / "stage_a_manifest.json", manifest)
        return root

    def test_tampered_bundle_rejected_before_torch(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._bundle(root)
            (root / "targets" / "s.json").write_text(
                json.dumps({"a": 9}), encoding="utf-8"
            )
            with self.assertRaises(MaterialError):
                run_stage_b(
                    root,
                    Path("nope"),
                    Path("nope"),
                    Path("nope"),
                    Path("nope"),
                    Path("nope"),
                    "cuda",
                    Path(tmp) / "out",
                    1,
                )

    @unittest.skipIf(HAS_TORCH, "torch-less ordering requires missing torch")
    def test_valid_bundle_without_torch_blocked(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._bundle(root)
            with self.assertRaises(MaterialBlockedError):
                run_stage_b(
                    root,
                    Path("nope"),
                    Path("nope"),
                    Path("nope"),
                    Path("nope"),
                    Path("nope"),
                    "cuda",
                    Path(tmp) / "out",
                    1,
                )
    def test_bad_geometry_rejected_after_verify(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "targets").mkdir(exist_ok=True)
            bad = dict(self._payload())
            bad["y_anchor"] = [1.0]
            payload = root / "targets" / "s.json"
            payload.write_text(json.dumps(bad, sort_keys=True), encoding="utf-8")
            manifest = {
                "artifact_role": "probe",
                "slice_sources": ["s"],
                "calib_sources": [],
                "class_weights": {},
                "targets": {
                    "s": {
                        "file": "targets/s.json",
                        "sha256": sha256_file(payload),
                        "num_frames": 2,
                        "audio_ref": "a.wav",
                        "waveform_sha256": "0" * 64,
                    }
                },
                "files": {"targets/s.json": sha256_file(payload)},
            }
            write_json(root / "stage_a_manifest.json", manifest)
            with self.assertRaises(MaterialError):
                run_stage_b(
                    root,
                    Path("nope"),
                    Path("nope"),
                    Path("nope"),
                    Path("nope"),
                    Path("nope"),
                    "cuda",
                    Path(tmp) / "out",
                    1,
                )


class StageBTargetLoaderTest(unittest.TestCase):
    def _case(self, root: Path, payload_obj: Any, meta: dict | None = None) -> dict:
        (root / "targets").mkdir(exist_ok=True)
        payload = root / "targets" / "s.json"
        payload.write_text(json.dumps(payload_obj, sort_keys=True), encoding="utf-8")
        entry = {
            "file": "targets/s.json",
            "sha256": sha256_file(payload),
            "num_frames": 2,
            "audio_ref": "a.wav",
            "waveform_sha256": "0" * 64,
        }
        if meta is not None:
            entry.update(meta)
        return {"targets": {"s": entry}}

    def _full_payload(self) -> dict:
        return {
            "source_id": "s",
            "num_frames": 2,
            "y_anchor": [1.0, 0.0],
            "y_replace": [0.0, 1.0],
            "valid": [True, True],
            "multiplicity": [0, 1],
            "episode_ids": ["ep1", None],
            "audio_ref": "a.wav",
            "waveform_sha256": "0" * 64,
        }

    def test_resolves_full_payloads_from_metadata_bundle(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._case(root, self._full_payload())
            self.assertNotIn("multiplicity", manifest["targets"]["s"])
            resolved = load_stage_targets(root, manifest, ["s"])
            self.assertEqual(resolved["s"]["multiplicity"], [0, 1])
            self.assertEqual(resolved["s"]["episode_ids"], ["ep1", None])
            self.assertEqual(len(resolved["s"]["y_anchor"]), 2)

    def test_escape_missing_and_hash_rejected(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._case(root, self._full_payload())
            escaped = dict(manifest)
            escaped["targets"] = {
                "s": {**manifest["targets"]["s"], "file": "../evil.json"}
            }
            with self.assertRaises(MaterialError):
                load_stage_targets(root, escaped, ["s"])
            with self.assertRaises(MaterialError):
                load_stage_targets(root, {"targets": {}}, ["s"])
            (root / "targets" / "s.json").write_text(
                json.dumps({"other": True}), encoding="utf-8"
            )
            with self.assertRaises(MaterialError):
                load_stage_targets(root, manifest, ["s"])

    def test_object_metadata_and_geometry_rejected(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._case(root, [1, 2])
            with self.assertRaises(MaterialError):
                load_stage_targets(root, manifest, ["s"])
            manifest = self._case(root, self._full_payload(), {"num_frames": 3})
            with self.assertRaises(MaterialError):
                load_stage_targets(root, manifest, ["s"])
            bad = self._full_payload()
            bad["y_replace"] = [0.0]
            manifest = self._case(root, bad)
            with self.assertRaises(MaterialError):
                load_stage_targets(root, manifest, ["s"])


class StageCTamperTest(unittest.TestCase):
    def _dirs(self, root: Path) -> tuple[Path, Path]:
        bundle = root / "bundle"
        staged = root / "staged"
        (bundle / "targets").mkdir(parents=True)
        payload = bundle / "targets" / "s.json"
        payload.write_text(json.dumps({"a": 1}), encoding="utf-8")
        write_json(
            bundle / "stage_a_manifest.json",
            {
                "artifact_role": "probe-a",
                "files": {"targets/s.json": sha256_file(payload)},
            },
        )
        arrays = staged / "stage_b_arrays"
        arrays.mkdir(parents=True)
        npz = arrays / "x.npz"
        np.savez_compressed(npz, f0_raw=np.asarray([0.0], dtype=np.float64))
        write_json(
            staged / "stage_b_manifest.json",
            {
                "artifact_role": "probe-b",
                "files": {"stage_b_arrays/x.npz": sha256_file(npz)},
            },
        )
        return bundle, staged

    def test_tampered_npz_rejected(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            bundle, staged = self._dirs(Path(tmp))
            np.savez_compressed(
                staged / "stage_b_arrays" / "x.npz",
                f0_raw=np.asarray([1.0], dtype=np.float64),
            )
            from experiments.psem_state_corrected_adaptation_gate.stages import (
                run_stage_c,
            )

            with self.assertRaises(MaterialError):
                run_stage_c(bundle, staged, Path(tmp) / "out", 1)

    def test_missing_file_rejected(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            bundle, staged = self._dirs(Path(tmp))
            (staged / "stage_b_arrays" / "x.npz").unlink()
            from experiments.psem_state_corrected_adaptation_gate.stages import (
                run_stage_c,
            )

            with self.assertRaises(MaterialError):
                run_stage_c(bundle, staged, Path(tmp) / "out", 1)


@unittest.skipUnless(_snapshot_available(), "stage-C match requires frozen snapshot")
class StageCExactMatchTest(unittest.TestCase):
    def test_matches_direct_primitives(self):
        from experiments.psem_frozen_ceiling_gate.evaluate_ceiling import (
            decode_scores,
            session_metrics,
        )

        sessions = _snapshot_sessions()
        full = next(
            s
            for s in sessions
            if s.source_family == "ami_mix_headset" and s.role == "dev"
        )
        count = 800
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
        rng = np.random.RandomState(7301)
        levels = rng.rand(60)
        scores = rng.choice(levels, count).tolist()
        target = [float(v) for v in np.asarray(dev.target)]
        valid = [bool(v) for v in np.asarray(dev.valid)]
        mapped = [True] * count
        kept, _ = mask_calibration(target, valid, mapped)
        kept_f0 = [scores[i] for i in kept]
        kept_cand = [[1.0 - s for s in scores][i] for i in kept]
        kept_target = [target[i] for i in kept]
        cand_raw = [1.0 - s for s in scores]
        stage_f0, stage_cand = fit_calibrators(kept_f0, kept_cand, kept_target)
        direct_f0 = calibrate_mod.fit_affine_calibrator(kept_f0, kept_target, "TRAIN-CALIB")
        direct_cand = calibrate_mod.fit_affine_calibrator(kept_cand, kept_target, "TRAIN-CALIB")
        self.assertEqual(stage_f0, direct_f0)
        self.assertEqual(stage_cand, direct_cand)
        unmapped: list[int] = []
        stage_horizons, _ = score_dev_frontiers(
            dev, scores, cand_raw, target, valid, mapped, unmapped, stage_f0, stage_cand, 1
        )
        f0_cal = calibrate_mod.apply_affine(scores, float(direct_f0["slope"]), float(direct_f0["intercept"]))
        f0_np = np.asarray([calibrate_mod.sigmoid(z) for z in f0_cal], dtype=np.float64)
        kept_f0_cal_d = [f0_cal[i] for i in kept]
        cand_cal_d = calibrate_mod.apply_affine(cand_raw, float(direct_cand["slope"]), float(direct_cand["intercept"]))
        cand_np_d = np.asarray([calibrate_mod.sigmoid(z) for z in cand_cal_d], dtype=np.float64)
        kept_cand_cal_d = [cand_cal_d[i] for i in kept]
        for horizon_ms in (100, 300, 500):
            thresholds = frontier_mod.unique_thresholds(cand_np_d.tolist())
            events = decode_scores(dev, f0_np, threshold=0.5, confirmation_ms=horizon_ms)
            from experiments.psem_state_corrected_adaptation_gate.material import (
                _frontier_point,
            )
            f0_point = _frontier_point(session_metrics(dev, events))
            points = []
            for threshold in thresholds:
                happenings = decode_scores(
                    dev, cand_np_d, threshold=float(threshold), confirmation_ms=horizon_ms
                )
                metrics = session_metrics(dev, happenings)
                points.append(
                    frontier_mod.FrontierPoint(
                        threshold=float(threshold),
                        false_cuts_per_hour=float(metrics["false_cut_count"])
                        / (float(metrics["active_speech_seconds"]) / 3600.0),
                        contamination=float(
                            metrics["exclusive_other_contamination_seconds_per_active_speech_hour"]
                        ),
                        miss_rate=float(metrics["missed_replacement_count"])
                        / float(metrics["reference_replacement_count"]),
                    )
                )
            envelopes = frontier_mod.select_envelopes(f0_point, points)
            expected = build_horizon_result(
                f0_point,
                points,
                envelopes,
                0,
                0,
                0,
                len(kept),
                0.0,
                kept_cand_cal_d,
                kept_target,
                kept_f0_cal_d,
            )
            got = build_horizon_result(
                stage_horizons[horizon_ms]["f0_point"],
                stage_horizons[horizon_ms]["candidate_points"],
                stage_horizons[horizon_ms]["envelopes"],
                0,
                0,
                0,
                len(kept),
                0.0,
                stage_horizons[horizon_ms]["kept_cand_cal"],
                stage_horizons[horizon_ms]["kept_target"],
                stage_horizons[horizon_ms]["kept_f0_cal"],
            )
            self.assertEqual(got, expected)


@unittest.skipUnless(_snapshot_available(), "dev filter requires frozen snapshot")
class DevExcludesEvalTest(unittest.TestCase):
    def test_dev_filter_excludes_eval(self):
        sessions = _snapshot_sessions()
        ami_dev = [
            s.source_id
            for s in sessions
            if s.source_family == "ami_mix_headset" and s.role == "dev"
        ]
        self.assertTrue(ami_dev)
        roles = {
            s.role for s in sessions if s.source_id in ami_dev
        }
        self.assertEqual(roles, {"dev"})
        from experiments.psem_state_corrected_adaptation_gate.stages import (
            resolve_dev_session,
        )

        by_id = {s.source_id: s for s in sessions}
        session = resolve_dev_session(by_id, ami_dev[0])
        self.assertEqual(session.role, "dev")
        eval_id = next(s.source_id for s in sessions if s.role == "eval")
        with self.assertRaises(MaterialError):
            resolve_dev_session(by_id, eval_id)
        with self.assertRaises(MaterialError):
            resolve_dev_session(by_id, "no-such-source")


class SpoolRetentionTest(unittest.TestCase):
    def _entry(self, frames=4):
        from types import SimpleNamespace

        from experiments.psem_state_corrected_adaptation_gate.lifecycle import AnchorEpisode

        return {
            "authority": SimpleNamespace(
                num_frames=frames,
                episodes=(AnchorEpisode("ep1", "spkA", 0, frames),),
                y_anchor=tuple([1.0] * frames),
                y_replace=tuple([0.0] * frames),
                valid=tuple([True] * frames),
                ledger={"opportunities": []},
            ),
            "multiplicity": [1] * frames,
            "episode_ids": ["ep1"] * frames,
            "intervals": [],
        }

    def test_prune_keeps_needed_only(self):
        import tempfile
        from types import SimpleNamespace

        from experiments.psem_state_corrected_adaptation_gate.stages import (
            prune_spooled_targets,
            sha256_file,
            write_spooled_target,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entries = {}
            for sid in ("a", "b", "c"):
                session = SimpleNamespace(audio_ref=f"{sid}.wav", waveform_sha256="0" * 64)
                entries[sid] = write_spooled_target(root, sid, self._entry(), session)
            removed = prune_spooled_targets(root, ["b"])
            self.assertEqual(removed, ["a", "c"])
            remaining = sorted(p.name for p in root.glob("*.json"))
            self.assertEqual(remaining, ["b.json"])
            self.assertEqual(
                entries["b"]["sha256"], sha256_file(root / "b.json")
            )


if __name__ == "__main__":
    unittest.main()
