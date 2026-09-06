from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from experiments.psem_state_corrected_adaptation_gate.calibrate import average_precision
from experiments.psem_state_corrected_adaptation_gate.material import (
    ClassAccumulator,
    MaterialBlockedError,
    MaterialError,
    audit_module_modes,
    microbatch_plan,
    oracle_slot_mapping,
    plan_windows,
    resolve_material_inputs,
    run_material_slice,
    run_slice_update,
    validate_gate0_record,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None
from experiments.psem_state_corrected_adaptation_gate.receipts import NEMO_SHA256

PACKAGE = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE.parents[1]


def _manifest_rows(per_source: int = 2048) -> list[dict]:
    rows = []
    for source, corpus in (("ami-slice-00", "AMI"), ("ali-slice-00", "AliMeeting")):
        for index in range(per_source):
            start = index * 480000
            rows.append(
                {
                    "split_role": "PSEM-STRATEGY-TRAIN",
                    "source_id": source,
                    "corpus": corpus,
                    "window_start_sample": start,
                    "window_end_sample": start + 480000,
                }
            )
    return rows


class ResolveTest(unittest.TestCase):
    def _paths(self, directory: Path) -> dict[str, Path]:
        checkpoint = directory / "model.nemo"
        checkpoint.write_bytes(b"not the frozen checkpoint")
        manifest = directory / "sampling.jsonl"
        manifest.write_text(
            "\n".join(json.dumps(r) for r in _manifest_rows()), encoding="utf-8"
        )
        roots = {}
        for name in ("nemo", "corpus", "reference"):
            path = directory / name
            path.mkdir()
            roots[name] = path
        lock = directory / "lock.json"
        lock.write_text("{}", encoding="utf-8")
        return {"checkpoint": checkpoint, "manifest": manifest, "lock": lock, **roots}

    def test_wrong_checkpoint_hash_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._paths(Path(tmp))
            with self.assertRaises(MaterialError):
                resolve_material_inputs(
                    paths["checkpoint"], paths["nemo"], paths["lock"], paths["corpus"],
                    paths["reference"], paths["manifest"], NEMO_SHA256,
                )

    def test_short_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._paths(Path(tmp))
            paths["manifest"].write_text("{}\n", encoding="utf-8")
            with self.assertRaises(MaterialError):
                resolve_material_inputs(
                    paths["checkpoint"], paths["nemo"], paths["lock"], paths["corpus"],
                    paths["reference"], paths["manifest"], NEMO_SHA256,
                )

    def test_valid_shape_resolves_both_corpora(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._paths(Path(tmp))
            paths["checkpoint"].write_bytes(b"placeholder")
            digest = hashlib.sha256(b"placeholder").hexdigest()
            resolved = resolve_material_inputs(
                paths["checkpoint"], paths["nemo"], paths["lock"], paths["corpus"],
                paths["reference"], paths["manifest"], digest,
            )
            self.assertEqual(resolved.ami_source, "ami-slice-00")
            self.assertEqual(resolved.alimeeting_source, "ali-slice-00")

    def test_material_never_passes_without_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            from unittest.mock import patch

            with patch(
                "experiments.psem_state_corrected_adaptation_gate.material._require_torch",
                side_effect=MaterialBlockedError("worker runtime unavailable"),
            ):
                with self.assertRaises(MaterialBlockedError):
                    run_material_slice(object(), out_dir)
            verdicts = []
            if out_dir.is_dir():
                for path in out_dir.glob("*.json"):
                    verdicts.append(json.loads(path.read_text(encoding="utf-8")).get("verdict"))
            self.assertNotIn("PASS", verdicts)

    def test_frozen_checkpoint_identity_matches_107(self):
        models = (
            REPO_ROOT / "experiments" / "psem_sortformer_adaptation_depth" / "models.py"
        ).read_text(encoding="utf-8")
        self.assertIn(NEMO_SHA256, models)


class Gate0BehaviorTest(unittest.TestCase):
    def test_windows_cover_two_adjacent_chunks(self):
        windows = plan_windows(800)[:2]
        self.assertEqual(windows, [(0, 375), (375, 750)])
        with self.assertRaises(MaterialError):
            plan_windows(700)

    def test_microbatches_align_to_window_edge(self):
        plan = microbatch_plan(750, 375, 16)
        self.assertEqual(len(plan), 16)
        self.assertEqual(plan[-1]["end"], 750)
        detaches = [(m["start"], m["window"]) for m in plan if m["detach_state"]]
        self.assertEqual(detaches, [(0, 0), (375, 1)])
        self.assertEqual(sum(m["end"] - m["start"] for m in plan), 750)

    def test_class_weights_need_full_support(self):
        accumulator = ClassAccumulator()
        accumulator.add([1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2, 1, 1], [True, True, True])
        weights = accumulator.weights()
        self.assertAlmostEqual(weights["replacement_positive_weight"], 2.0 / 2.0)
        self.assertAlmostEqual(weights["anchor_positive_weight"], 1.0 / 3.0)
        empty = ClassAccumulator()
        with self.assertRaises(MaterialError):
            empty.weights()
        one_sided = ClassAccumulator()
        one_sided.add([0.0], [1.0], [1], [True])
        with self.assertRaises(MaterialError):
            one_sided.weights()

    def test_mode_audit_rejects_train_mode_backbone(self):
        receipt = audit_module_modes(False, [False, False], True, [], ["h.gru.weight"])
        self.assertTrue(receipt["frozen_representation_ok"])
        with self.assertRaises(MaterialError):
            audit_module_modes(True, [False], True, [], ["h"])
        with self.assertRaises(MaterialError):
            audit_module_modes(False, [True], True, [], ["h"])
        with self.assertRaises(MaterialError):
            audit_module_modes(False, [], False, [], ["h"])
        with self.assertRaises(MaterialError):
            audit_module_modes(False, [], True, ["s.layer"], ["h"])

    def test_mapping_needs_no_validity_mask(self):
        episode_ids = ["ep1", "ep1", "ep1", "ep2", None]
        anchor_active = [True, True, True, True, False]
        valid = [True, True, False, True, False]
        probabilities = [[0.1, 0.9], [0.2, 0.8], [0.9, 0.1], [0.4, 0.6], [0.0, 0.0]]
        slot_of, rows, unmapped = oracle_slot_mapping(
            episode_ids, anchor_active, valid, probabilities
        )
        self.assertEqual(slot_of, {"ep1": 1, "ep2": 1})
        self.assertEqual(rows[0]["support_frame_count"], 2)
        self.assertIn(4, unmapped)
        self.assertNotIn(2, unmapped)
        with self.assertRaises(MaterialError):
            oracle_slot_mapping(["ep1"], [True], [True, False], probabilities)

    def test_average_precision_ranking(self):
        self.assertAlmostEqual(average_precision([0.9, 0.1, 0.8], [1.0, 0.0, 1.0]), 1.0)
        self.assertLess(average_precision([0.1, 0.9, 0.2], [1.0, 0.0, 0.0]), 1.0)
        with self.assertRaises(Exception):
            average_precision([0.5], [0.0])

    def test_record_validator_rejects_thin_record(self):
        with self.assertRaises(MaterialError):
            validate_gate0_record({"verdict": "PASS", "mode": "material"})
        thin = {
            "verdict": "PASS",
            "mode": "material",
            "checks": {},
            "evidence": {
                "checkpoint_sha256": "x",
                "sampling_sha256": "x",
                "partition": {"fit": ["a"], "calib": ["b"]},
                "calibration_candidate": {
                    "slope": 1.0, "intercept": 0.0, "nll": 1.0, "brier": 0.25,
                    "raw_nll": 1.0,
                },
                "calibration_f0": {
                    "slope": 1.0, "intercept": 0.0, "nll": 1.0, "brier": 0.25,
                    "raw_nll": 1.0,
                },
                "dev": {
                    "src": {
                        "100": {
                            "budget": 1.0,
                            "points": [{"threshold": 0.5}],
                            "c_envelope": None,
                            "m_envelope": None,
                            "raw_ap": 0.5,
                            "mapping_mapped": 1,
                        }
                    }
                },
                "profiler": {
                    "optimizer_steps": 8,
                    "seconds_per_step": 1.0,
                    "peak_vram_bytes": 1,
                    "dev_infer_seconds": {},
                },
                "predictions": {"src": {"path": "p", "sha256": "s"}},
            },
        }
        self.assertTrue(validate_gate0_record(thin))

    def test_canonical_geometry_and_alignment(self):
        from experiments.psem_state_corrected_adaptation_gate.material import (
            canonical_frames,
            mask_calibration,
            require_frame_alignment,
            require_frame_vector,
            select_fit_slice,
            slice_waveform_frames,
        )
        self.assertEqual(canonical_frames(960000, 480000), 750)
        self.assertEqual(slice_waveform_frames(960000, 750, "src"), (960000, 0))
        self.assertEqual(slice_waveform_frames(961000, 750, "src"), (960000, 1000))
        with self.assertRaises(MaterialError):
            slice_waveform_frames(1000, 750, "src")
        self.assertTrue(require_frame_alignment(750, 750, "src"))
        with self.assertRaises(MaterialError):
            require_frame_alignment(749, 750, "src")
        self.assertTrue(require_frame_vector((1, 750), "logit"))
        with self.assertRaises(MaterialError):
            require_frame_vector((1, 750, 750), "logit")
        fit = ["ami-a", "ami-b", "ali-a"]
        rows = {"ami-a": [1, 2, 3], "ami-b": [1], "ali-a": [1, 2]}
        corpus = {"ami-a": "AMI", "ami-b": "AMI", "ali-a": "AliMeeting"}
        self.assertEqual(select_fit_slice(fit, rows, corpus), ("ami-a", "ali-a"))
        with self.assertRaises(MaterialError):
            select_fit_slice(["ami-a"], rows, corpus)
        kept, coverage = mask_calibration([1.0, 0.0, 1.0, 0.0], [True, True, False, True], [True, True, True, False])
        self.assertEqual(kept, [0, 1])
        self.assertEqual(coverage["positive"], 1)
        with self.assertRaises(MaterialError):
            mask_calibration([0.0, 0.0], [True, True], [True, True])

    def test_calibration_buffers_accumulate_all_arms(self):
        from experiments.psem_state_corrected_adaptation_gate.material import (
            extend_calibration_buffers,
        )
        buffers = {"f0": [], "cand": [], "targets": []}
        extend_calibration_buffers(
            buffers, [0.1, 0.2, 0.3, 0.4], [1.1, 1.2, 1.3, 1.4], [0.0, 1.0, 0.0, 1.0], [1, 3]
        )
        self.assertEqual(buffers["f0"], [0.2, 0.4])
        self.assertEqual(buffers["cand"], [1.2, 1.4])
        self.assertEqual(buffers["targets"], [1.0, 1.0])
        with self.assertRaises(MaterialError):
            extend_calibration_buffers(
                {"f0": [], "cand": [], "targets": []}, [0.1], [0.2], [0.0, 1.0], [0]
            )
        with self.assertRaises(MaterialError):
            extend_calibration_buffers(
                {"f0": [], "cand": [], "targets": []}, [0.1], [0.2], [0.0], []
            )


@unittest.skipUnless(HAS_TORCH, "slice update requires torch runtime")
class SliceUpdateLossPrimitiveTest(unittest.TestCase):
    def test_update_runs_shared_audit_to_parameter_change(self):
        import torch
        from types import SimpleNamespace

        from experiments.psem_state_corrected_adaptation_gate import head as head_mod

        self.assertIsNotNone(head_mod.ResidualPSEMHead)
        frames = 750
        generator = torch.Generator().manual_seed(7301)
        train = {
            "multiplicity": [0] * 375 + [1] * 375,
            "mapped_flags": [True] * frames,
            "authority": SimpleNamespace(
                y_replace=[1.0 if i % 10 == 0 else 0.0 for i in range(frames)],
                y_anchor=[1.0 if i % 3 == 0 else 0.0 for i in range(frames)],
            ),
            "frame_count": frames,
            "features": torch.randn(1, frames, 199, generator=generator),
            "selected_logit": torch.randn(1, frames, generator=generator),
        }
        frozen = torch.nn.Parameter(torch.zeros(4))
        frozen.requires_grad_(False)
        wrapper = SimpleNamespace(parameters=lambda: [frozen])
        head = head_mod.ResidualPSEMHead(199)
        device = (
            torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
        )
        head.to(device)
        before = [p.detach().clone() for p in head.parameters()]
        train["features"] = train["features"].to(device)
        train["selected_logit"] = train["selected_logit"].to(device)
        ctx = run_slice_update(
            torch,
            wrapper,
            head,
            train,
            {"replacement_positive_weight": 9.0, "anchor_positive_weight": 1.5},
            device,
            "probe",
        )
        self.assertEqual(tuple(ctx["product_all"].shape), (1, frames))
        self.assertLessEqual(ctx["identity_diff"], 1e-6)
        self.assertTrue(
            any(
                not bool(torch.equal(a, b))
                for a, b in zip(before, head.parameters())
            )
        )
        self.assertTrue(bool(torch.equal(frozen, torch.zeros(4))))


@unittest.skipUnless(HAS_TORCH, "DEV geometry requires torch runtime")
class DevBatchGeometryTest(unittest.TestCase):
    def _fixture(self, directory, frames=96, emit_frames=None):
        import torch
        import wave
        from types import SimpleNamespace

        import numpy as np

        samples = frames * 1280
        path = Path(directory) / "probe.wav"
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(b"\x00\x00" * samples)
        generator = torch.Generator().manual_seed(7301)
        emitted = frames if emit_frames is None else emit_frames
        passage = {
            "windows": [
                {
                    "hidden": torch.randn(1, emitted, 8, generator=generator),
                    "logits": torch.randn(1, emitted, 4, generator=generator),
                    "probabilities": torch.softmax(
                        torch.randn(1, emitted, 4, generator=generator), dim=-1
                    ),
                    "emitted_frames": emitted,
                }
            ]
        }
        grid = np.arange(frames, dtype=np.int64)
        dev = SimpleNamespace(
            source_id="probe-dev",
            starts=grid * 1280,
            ends=(grid + 1) * 1280,
            episode_ids=["ep1"] * frames,
            anchor_present=[i < frames // 2 for i in range(frames)],
            valid=[True] * frames,
            target=[1.0 if i % 5 == 0 else 0.0 for i in range(frames)],
        )
        runtime = SimpleNamespace(audio_ref="probe.wav")
        return passage, dev, runtime

    def test_batched_evidence_yields_aligned_logits(self):
        import math
        import tempfile

        import torch
        from unittest import mock

        from experiments.psem_state_corrected_adaptation_gate import head as head_mod
        from experiments.psem_state_corrected_adaptation_gate import material as material_mod

        frames = 96
        with tempfile.TemporaryDirectory() as tmp:
            passage, dev, runtime = self._fixture(tmp, frames)
            head = head_mod.ResidualPSEMHead(15)
            device = (
            torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
        )
            head.to(device)
            for window in passage["windows"]:
                for key in ("hidden", "logits", "probabilities"):
                    window[key] = window[key].to(device)
            waveform = (torch.zeros(1, frames * 1280), 16000)
            with mock.patch.object(
                material_mod, "run_adjacent_windows", return_value=passage
            ), mock.patch("torchaudio.load", return_value=waveform):
                out = material_mod.infer_dev_raw_logits(
                    torch, None, head, dev, runtime, Path(tmp), device
                )
        self.assertEqual(len(out["f0_raw"]), frames)
        self.assertEqual(len(out["cand_raw"]), frames)
        self.assertEqual(len(out["target"]), frames)
        self.assertEqual(out["grid_frames"], frames)
        self.assertGreaterEqual(out["mapping_mapped"], 1)
        self.assertTrue(all(math.isfinite(v) for v in out["f0_raw"]))
        self.assertTrue(all(math.isfinite(v) for v in out["cand_raw"]))
        self.assertLessEqual(len(out["kept"]), frames)

    def test_mismatched_evidence_still_fails(self):
        import tempfile

        import torch
        from unittest import mock

        from experiments.psem_state_corrected_adaptation_gate import head as head_mod
        from experiments.psem_state_corrected_adaptation_gate import material as material_mod
        from experiments.psem_state_corrected_adaptation_gate.material import MaterialError

        frames = 96
        with tempfile.TemporaryDirectory() as tmp:
            passage, dev, runtime = self._fixture(tmp, frames, emit_frames=frames + 8)
            head = head_mod.ResidualPSEMHead(15)
            device = (
            torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
        )
            waveform = (torch.zeros(1, frames * 1280), 16000)
            with mock.patch.object(
                material_mod, "run_adjacent_windows", return_value=passage
            ), mock.patch("torchaudio.load", return_value=waveform):
                with self.assertRaises(MaterialError):
                    material_mod.infer_dev_raw_logits(
                        torch, None, head, dev, runtime, Path(tmp), device
                    )


@unittest.skipUnless(HAS_TORCH, "profiler mode requires torch runtime")
class ProfilerTrainModeTest(unittest.TestCase):
    def test_profiler_runs_backward_in_train_mode_then_restores(self):
        import torch
        from types import SimpleNamespace

        from experiments.psem_state_corrected_adaptation_gate import head as head_mod
        from experiments.psem_state_corrected_adaptation_gate.material import (
            microbatch_plan,
            plan_windows,
            run_profiler,
        )

        frames = 750
        generator = torch.Generator().manual_seed(7301)
        features = torch.randn(1, frames, 199, generator=generator)
        f0_all = torch.randn(1, frames, generator=generator)
        mult = torch.zeros(1, frames)
        mult[:, 375:] = 1.0
        train = {"features": features}
        head = head_mod.ResidualPSEMHead(199)
        head.eval()
        optimizer = torch.optim.AdamW(
            [p for p in head.parameters() if p.requires_grad], lr=1e-4
        )
        windows = plan_windows(frames, 375)[:2]
        update_ctx = {
            "optimizer": optimizer,
            "train_mult_weight": mult,
            "train_y_replace": torch.zeros(1, frames),
            "train_y_anchor": torch.zeros(1, frames),
            "microbatches": [
                mb for mb in microbatch_plan(frames, 375, 16) if mb["end"] <= windows[1][1]
            ],
            "windows": windows,
            "f0_all": f0_all,
        }
        device = (
            torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
        )
        head.to(device)
        before = [p.detach().clone() for p in head.parameters()]
        train["features"] = train["features"].to(device)
        update_ctx["f0_all"] = update_ctx["f0_all"].to(device)
        update_ctx["train_mult_weight"] = update_ctx["train_mult_weight"].to(device)
        update_ctx["train_y_replace"] = update_ctx["train_y_replace"].to(device)
        update_ctx["train_y_anchor"] = update_ctx["train_y_anchor"].to(device)
        modes = []
        handle = head.gru.register_forward_hook(
            lambda module, _inputs, _output: modes.append(bool(module.training))
        )
        try:
            profiler = run_profiler(
                torch,
                head,
                train,
                update_ctx,
                {
                    "replacement_positive_weight": 9.0,
                    "anchor_positive_weight": 1.5,
                },
                device,
                {},
                profile_steps=2,
            )
        finally:
            handle.remove()
        self.assertTrue(modes)
        self.assertTrue(all(modes))
        self.assertFalse(head.training)
        self.assertEqual(profiler["optimizer_steps"], 2)
        self.assertGreaterEqual(profiler["seconds_per_step"], 0.0)
        self.assertTrue(
            any(
                not bool(torch.equal(a, b))
                for a, b in zip(before, head.parameters())
            )
        )


@unittest.skipUnless(HAS_TORCH, "profiler record link requires torch runtime")
class ProfilerRecordLinkTest(unittest.TestCase):
    def test_record_receives_profiler_result(self):
        import torch

        from experiments.psem_state_corrected_adaptation_gate import head as head_mod
        from experiments.psem_state_corrected_adaptation_gate.material import (
            build_gate0_record,
            microbatch_plan,
            plan_windows,
            run_profiler,
        )

        frames = 750
        generator = torch.Generator().manual_seed(7301)
        features = torch.randn(1, frames, 199, generator=generator)
        f0_all = torch.randn(1, frames, generator=generator)
        mult = torch.zeros(1, frames)
        mult[:, 375:] = 1.0
        train = {"features": features}
        head = head_mod.ResidualPSEMHead(199)
        optimizer = torch.optim.AdamW(
            [p for p in head.parameters() if p.requires_grad], lr=1e-4
        )
        windows = plan_windows(frames, 375)[:2]
        update_ctx = {
            "optimizer": optimizer,
            "train_mult_weight": mult,
            "train_y_replace": torch.zeros(1, frames),
            "train_y_anchor": torch.zeros(1, frames),
            "microbatches": [
                mb for mb in microbatch_plan(frames, 375, 16) if mb["end"] <= windows[1][1]
            ],
            "windows": windows,
            "f0_all": f0_all,
        }
        device = (
            torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
        )
        head.to(device)
        train["features"] = train["features"].to(device)
        update_ctx["f0_all"] = update_ctx["f0_all"].to(device)
        update_ctx["train_mult_weight"] = update_ctx["train_mult_weight"].to(device)
        update_ctx["train_y_replace"] = update_ctx["train_y_replace"].to(device)
        update_ctx["train_y_anchor"] = update_ctx["train_y_anchor"].to(device)
        profiler = run_profiler(
            torch,
            head,
            train,
            update_ctx,
                {
                    "replacement_positive_weight": 9.0,
                    "anchor_positive_weight": 1.5,
                },
                device,
                {},
                profile_steps=8,
        )
        calibration = {
            "slope": 1.0,
            "intercept": 0.0,
            "nll": 0.5,
            "brier": 0.25,
            "raw_nll": 0.6,
            "raw_brier": 0.3,
            "role": "probe",
        }
        record = build_gate0_record(
            ("ami-x", "ali-y"),
            {"checkpoint_sha256": "abc"},
            "sasha",
            {},
            {"fit": [], "calib": [], "salt": "s", "target_frac": 0.12},
            {
                "replacement_positive_weight": 9.0,
                "anchor_positive_weight": 1.5,
            },
            {},
            dict(calibration),
            dict(calibration),
            {},
            profiler,
            {},
        )
        self.assertEqual(
            record["evidence"]["profiler"]["optimizer_steps"],
            profiler["optimizer_steps"],
        )
        self.assertEqual(record["evidence"]["profiler"]["seconds_per_step"], profiler["seconds_per_step"])
        self.assertIn("verdict", record)


if __name__ == "__main__":
    unittest.main()
