from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from experiments.psem_state_corrected_adaptation_gate import arm_runtime
from experiments.psem_state_corrected_adaptation_gate import temporal_train


def _h(tag: str) -> str:
    return hashlib.sha256(tag.encode("utf-8")).hexdigest()


def _config(tmp: Path, arm: str = arm_runtime.ARM_R_T2_SC, seed: int = 7301):
    return arm_runtime.ArmRunConfig(
        arm=arm,
        seed=seed,
        root=tmp / "arms",
        input_hash=_h("input"),
        checkpoint_hash=_h("ckpt"),
        partition_hash=_h("part"),
        weights_hash=_h("w"),
        code_hash=_h("code"),
    )


def _param_names() -> list[str]:
    names = ["psem_head.gru.weight_ih_l0", "psem_head.residual_out.weight"]
    for layer in range(18):
        base = f"sortformer.transformer_encoder.layers.{layer}"
        names += [
            f"{base}.self_attn.weight",
            f"{base}.norm1.weight",
            f"{base}.norm2.bias",
            f"{base}.mlp.weight",
        ]
    names += [
        "sortformer.sortformer_modules.first_hidden_to_hidden.weight",
        "sortformer.sortformer_modules.single_hidden_to_spks.weight",
        "sortformer.encoder.conformer.weight",
        "sortformer.frontend_encoder.conv.weight",
        "sortformer.decoder.other.weight",
    ]
    return names


class FakeParam:
    def __init__(self, name: str) -> None:
        self.name = name
        self.requires_grad = False

    def requires_grad_(self, mode: bool):
        self.requires_grad = bool(mode)
        return self


class FakeModule:
    def __init__(self, training: bool = False) -> None:
        self.training = training

    def train(self, mode: bool = True):
        self.training = bool(mode)
        return self


class FakeModel:
    def __init__(self, names: list[str]) -> None:
        self._params = [(n, FakeParam(n)) for n in names]
        self._modules = {n.rsplit(".", 1)[0]: FakeModule() for n in names}

    def named_parameters(self):
        return iter(self._params)

    def named_modules(self):
        yield ("", FakeModule())
        for path, module in self._modules.items():
            yield (path, module)


def _authority(frames: int):
    half = frames // 2
    y_replace = [1.0 if (f // 120) % 2 else 0.0 for f in range(frames)]
    y_anchor = [1.0] * frames
    valid = [True] * frames
    return SimpleNamespace(y_replace=y_replace, y_anchor=y_anchor, valid=valid)


class TemporalPolicyTest(unittest.TestCase):
    def test_h_arm_rejected(self):
        with self.assertRaises(arm_runtime.AuthorizationError):
            temporal_train.require_temporal_arm(arm_runtime.ARM_R_H_SC)
        with self.assertRaises(arm_runtime.AuthorizationError):
            temporal_train.legacy_policy_arm(arm_runtime.ARM_R_H_SC)
        with self.assertRaises(arm_runtime.AuthorizationError):
            temporal_train.apply_temporal_parameter_policy(
                FakeModel(_param_names()), arm_runtime.ARM_R_H_SC
            )

    def test_t2_trains_only_final_two_blocks(self):
        names = _param_names()
        trainable = set(temporal_train.trainable_names(names, arm_runtime.ARM_R_T2_SC))
        for layer in (16, 17):
            self.assertIn(
                f"sortformer.transformer_encoder.layers.{layer}.self_attn.weight",
                trainable,
            )
            self.assertIn(
                f"sortformer.transformer_encoder.layers.{layer}.norm1.weight",
                trainable,
            )
        for layer in range(16):
            self.assertNotIn(
                f"sortformer.transformer_encoder.layers.{layer}.self_attn.weight",
                trainable,
            )
        self.assertIn(
            "sortformer.sortformer_modules.first_hidden_to_hidden.weight", trainable
        )
        self.assertIn(
            "sortformer.sortformer_modules.single_hidden_to_spks.weight", trainable
        )
        self.assertIn("psem_head.gru.weight_ih_l0", trainable)

    def test_ta_trains_all_blocks(self):
        names = _param_names()
        trainable = set(temporal_train.trainable_names(names, arm_runtime.ARM_R_TA_SC))
        for layer in range(18):
            self.assertIn(
                f"sortformer.transformer_encoder.layers.{layer}.self_attn.weight",
                trainable,
            )
        self.assertIn(
            "sortformer.sortformer_modules.single_hidden_to_spks.weight", trainable
        )

    def test_acoustic_nest_encoder_always_frozen(self):
        names = _param_names()
        for arm in (arm_runtime.ARM_R_T2_SC, arm_runtime.ARM_R_TA_SC):
            trainable = set(temporal_train.trainable_names(names, arm))
            self.assertNotIn("sortformer.encoder.conformer.weight", trainable)
            self.assertNotIn("sortformer.frontend_encoder.conv.weight", trainable)
            audit = temporal_train.audit_temporal_trainability(names, arm)
            self.assertTrue(audit["acoustic_encoder_frozen"])
        self.assertTrue(
            temporal_train.is_frozen_encoder_param(
                "sortformer.encoder.transformer.layers.16.weight"
            )
        )

    def test_module_modes_follow_policy(self):
        for arm in (arm_runtime.ARM_R_T2_SC, arm_runtime.ARM_R_TA_SC):
            model = FakeModel(_param_names())
            temporal_train.apply_temporal_parameter_policy(model, arm)
            temporal_train.set_temporal_module_modes(model, arm)
            receipt = temporal_train.audit_temporal_module_modes(model, arm)
            self.assertTrue(receipt["modes_ok"])
            model._modules["sortformer.encoder.conformer"].train(True)
            with self.assertRaises(temporal_train.TemporalArmError):
                temporal_train.audit_temporal_module_modes(model, arm)

    def test_optimizer_group_split(self):
        for arm in (arm_runtime.ARM_R_T2_SC, arm_runtime.ARM_R_TA_SC):
            model = FakeModel(_param_names())
            head = FakeModel(["gru.weight", "residual_out.weight"])
            temporal_train.apply_temporal_parameter_policy(model, arm)
            for _, param in head.named_parameters():
                param.requires_grad_(True)
            groups = temporal_train.optimizer_param_groups(model, head, arm)
            self.assertEqual(
                sorted(groups["head"]),
                [
                    "psem_head.gru.weight",
                    "psem_head.gru.weight_ih_l0",
                    "psem_head.residual_out.weight",
                ],
            )
            self.assertTrue(
                all(
                    "first_hidden_to_hidden" in n or "single_hidden_to_spks" in n
                    for n in groups["activity"]
                )
            )
            self.assertTrue(
                all("transformer_encoder.layers" in n for n in groups["temporal"])
            )

    def test_no_eval(self):
        with self.assertRaises(temporal_train.TemporalArmError):
            temporal_train.assert_no_eval(["PSEM-STRATEGY-TRAIN", "PSEM-STRATEGY-EVAL"])
        temporal_train.assert_no_eval(["PSEM-STRATEGY-TRAIN", "PSEM-STRATEGY-DEV"])

    def test_single_stream_guard(self):
        temporal_train.check_single_stream(arm_runtime.ARM_R_T2_SC)
        try:
            with self.assertRaises(temporal_train.TemporalArmError):
                temporal_train.check_single_stream(arm_runtime.ARM_R_TA_SC)
        finally:
            temporal_train.release_stream(arm_runtime.ARM_R_T2_SC)
        temporal_train.check_single_stream(arm_runtime.ARM_R_TA_SC)
        temporal_train.release_stream(arm_runtime.ARM_R_TA_SC)

    def test_chunk_spans_with_tail(self):
        self.assertEqual(
            temporal_train.chunk_spans(800), [(0, 375), (375, 750), (750, 800)]
        )
        self.assertEqual(temporal_train.chunk_spans(375), [(0, 375)])
        with self.assertRaises(temporal_train.TemporalArmError):
            temporal_train.chunk_spans(0)

    def test_mult_excludes_unmapped_and_invalid(self):
        authority = _authority(800)
        authority.valid[100] = False
        multiplicity = [2.0] * 800
        episode_ids = ["e1"] * 800
        weights = temporal_train.frame_mult_weights(
            authority, multiplicity, episode_ids, {"e1": 0}, 0, 800
        )
        self.assertEqual(weights[50], 2.0)
        self.assertEqual(weights[100], 0.0)
        self.assertFalse(
            temporal_train.chunk_has_support(
                authority, multiplicity, episode_ids, {}, 0, 800
            )
        )
        self.assertTrue(
            temporal_train.chunk_has_support(
                authority, multiplicity, episode_ids, {"e1": 0}, 700, 100
            )
        )
    def test_loss_chunk_schedule(self):
        entries = {
            "s-a": {
                "num_frames": 800,
                "authority": _authority(800),
                "multiplicity": [1] * 800,
                "episode_ids": ["e1"] * 800,
            }
        }
        mapping = temporal_train.PassMapping(
            source_to_mapping={"s-a": {"slot_of": {"e1": 0}, "rows": [], "unmapped": []}},
            manifest_hash="0" * 64,
            arm=arm_runtime.ARM_R_T2_SC,
            seed=7301,
        )
        self.assertEqual(temporal_train.count_loss_chunks(entries, mapping, ["s-a"]), 3)
        self.assertEqual(temporal_train.schedule_total_steps(3, 16), 1)
        self.assertEqual(temporal_train.schedule_total_steps(17, 16), 2)
        with self.assertRaises(temporal_train.TemporalArmError):
            temporal_train.schedule_total_steps(0, 16)
    def test_group_scale_exact_mean(self):
        self.assertEqual(temporal_train.group_scale(16, 16), 1.0 / 16.0)
        self.assertEqual(temporal_train.group_scale(32, 16), 1.0 / 16.0)
        self.assertEqual(temporal_train.group_scale(5, 16), 1.0 / 5.0)
        self.assertEqual(temporal_train.group_scale(1, 16), 1.0)
        with self.assertRaises(temporal_train.TemporalArmError):
            temporal_train.group_scale(0, 16)


    def test_authorization_runs_before_backend(self):
        import tempfile
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as tmp:
            config = _config(Path(tmp))
            store = Path(tmp) / "store"
            store.mkdir()
            with self.assertRaises(arm_runtime.AuthorizationError):
                temporal_train.run_profile_command(config, store, SimpleNamespace())

    def test_execution_binding_mismatch_fails_closed(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            checkpoint = tmp_path / "model.nemo"
            checkpoint.write_bytes(b"checkpoint-bytes")
            manifest = {
                "sampling_sha256": _h("input"),
                "nemo_sha256": hashlib.sha256(b"checkpoint-bytes").hexdigest(),
                "fit": ["s-a"],
                "calib": ["s-b"],
                "salt": "salt",
                "target_frac": 0.12,
                "class_weights": {
                    "replacement_positive_weight": 9.0,
                    "anchor_positive_weight": 1.0,
                },
            }
            partition_hash = arm_runtime.canonical_sha256(
                {
                    "fit": ["s-a"],
                    "calib": ["s-b"],
                    "salt": "salt",
                    "target_frac": 0.12,
                }
            )
            _, weights_hash = arm_runtime.bind_class_weights(
                dict(manifest["class_weights"])
            )
            base = {
                "arm": arm_runtime.ARM_R_T2_SC,
                "seed": 7301,
                "root": tmp_path / "arms",
                "input_hash": _h("input"),
                "checkpoint_hash": hashlib.sha256(b"checkpoint-bytes").hexdigest(),
                "partition_hash": partition_hash,
                "weights_hash": weights_hash,
                "code_hash": temporal_train._code_identity(),
            }
            config = arm_runtime.config_from_dict(dict(base))
            binding = temporal_train.verify_execution_binding(
                config, manifest, checkpoint
            )
            self.assertEqual(binding["fit"], ["s-a"])
            for key, value in (
                ("input_hash", _h("other")),
                ("partition_hash", "0" * 64),
                ("weights_hash", "0" * 64),
                ("code_hash", "0" * 64),
                ("checkpoint_hash", "0" * 64),
            ):
                tampered = arm_runtime.config_from_dict({**base, key: value})
                with self.assertRaises(temporal_train.TemporalArmError):
                    temporal_train.verify_execution_binding(
                        tampered, manifest, checkpoint
                    )

    def test_code_identity_covers_load_bearing_files(self):
        import tempfile

        required = {
            "arm_runtime.py",
            "gates.py",
            "temporal_train.py",
            "run_temporal_arm.py",
            "material.py",
            "head.py",
            "stages.py",
            "frontier.py",
            "cross_frontier.py",
            "multiplicity.py",
            "lifecycle.py",
            "calibrate.py",
            "../psem_sortformer_adaptation_depth/parameter_policy.py",
        }
        self.assertEqual(set(temporal_train.CODE_IDENTITY_FILES), required)
        for name in temporal_train.CODE_IDENTITY_FILES:
            self.assertTrue(
                (Path(temporal_train.__file__).resolve().parent / name).is_file(),
                name,
            )
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "a.py"
            second = Path(tmp) / "b.py"
            first.write_bytes(b"content-a")
            second.write_bytes(b"content-b")
            intact = temporal_train._digest_named_files(
                [("a.py", first), ("b.py", second)]
            )
            second.write_bytes(b"content-c")
            self.assertNotEqual(
                temporal_train._digest_named_files(
                    [("a.py", first), ("b.py", second)]
                ),
                intact,
            )
            self.assertNotEqual(
                temporal_train._digest_named_files([("a.py", first)]), intact
            )

    def test_stored_mapping_load_and_tamper(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            config = _config(Path(tmp))
            mapping = temporal_train.PassMapping(
                source_to_mapping={"s-a": {"slot_of": {"e1": 0}, "rows": [], "unmapped": []}},
                manifest_hash="",
                arm=config.arm,
                seed=config.seed,
            )
            mapping.manifest_hash = arm_runtime.canonical_sha256(
                {sid: mapping.source_to_mapping[sid] for sid in sorted(mapping.source_to_mapping)}
            )
            run_dir = config.run_dir()
            temporal_train.write_mapping_files(run_dir, mapping, config.config_hash)
            loaded = temporal_train.load_frozen_mapping(run_dir, config)
            self.assertEqual(loaded.manifest_hash, mapping.manifest_hash)
            self.assertEqual(loaded.for_source("s-a")["slot_of"], {"e1": 0})
            other = _config(Path(tmp), arm=arm_runtime.ARM_R_TA_SC)
            with self.assertRaises(temporal_train.TemporalArmError):
                temporal_train.load_frozen_mapping(run_dir, other)

    def test_arm_artifact_names(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            paths = temporal_train.write_arm_artifacts(
                Path(tmp),
                {
                    "experiment_manifest.json": {"a": 1},
                    "data_sampling_calibration_manifest.json": {"b": 2},
                    "parameter_module_mode_receipt.json": {"c": 3},
                    "training_metrics.json": {"d": 4},
                    "calibration_metrics.json": {"e": 5},
                    "dev_frontier.json": {"f": 6},
                },
            )
            self.assertEqual(
                sorted(p.name for p in paths),
                [
                    "calibration_metrics.json",
                    "data_sampling_calibration_manifest.json",
                    "dev_frontier.json",
                    "experiment_manifest.json",
                    "parameter_module_mode_receipt.json",
                    "training_metrics.json",
                ],
            )
            for path in paths:
                self.assertTrue(path.is_file())

    def test_profile_gate(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            config = _config(Path(tmp))
            with self.assertRaises(temporal_train.TemporalArmError):
                temporal_train.require_profile_receipt(None, config)
            bad = {
                "optimizer_steps": 4,
                "seconds_per_step": 1.0,
                "peak_vram_bytes": 8,
                "dev_infer_seconds": {},
                "arm": config.arm,
                "seed": config.seed,
                "config_hash": config.config_hash,
            }
            with self.assertRaises(temporal_train.TemporalArmError):
                temporal_train.require_profile_receipt(bad, config)
            good = dict(bad, optimizer_steps=8)
            self.assertEqual(
                temporal_train.require_profile_receipt(good, config)["optimizer_steps"],
                8,
            )
            wrong_arm = dict(good, arm=arm_runtime.ARM_R_TA_SC)
            with self.assertRaises(temporal_train.TemporalArmError):
                temporal_train.require_profile_receipt(wrong_arm, config)

    def test_frontier_comparison_hooks(self):
        from experiments.psem_state_corrected_adaptation_gate import frontier

        def _points() -> list[frontier.FrontierPoint]:
            return [
                frontier.FrontierPoint(0.9, 10.0, 5.0, 0.4),
                frontier.FrontierPoint(0.5, 20.0, 9.0, 0.6),
                frontier.FrontierPoint(0.1, 40.0, 12.0, 0.2),
            ]

        corpora = {"horizon_ms": 500, "group": "pooled", "sources": ["s-a"]}
        budget = frontier.FrontierPoint(0.5, 20.0, 9.0, 0.6)
        t2 = temporal_train.compare_t2_to_h_f0(_points(), budget, corpora, _points())
        self.assertEqual(t2["arm"], "R-T2-SC")
        self.assertEqual(t2["baseline"], "R-H-SC+F0")
        self.assertEqual(t2["budget"], 20.0)
        self.assertIsNotNone(t2["baseline_depth"])
        self.assertIsNotNone(t2["delta_vs_baseline"])
        ta = temporal_train.compare_ta_to_t2_f0(_points(), budget, corpora, _points())
        self.assertEqual(ta["arm"], "R-TA-SC")
        self.assertEqual(ta["baseline"], "R-T2-SC+F0")

    def test_baseline_without_half_point_raises(self):
        from experiments.psem_state_corrected_adaptation_gate import frontier

        points = [
            frontier.FrontierPoint(0.9, 10.0, 5.0, 0.4),
            frontier.FrontierPoint(0.1, 40.0, 12.0, 0.2),
        ]
        corpora = {"horizon_ms": 500, "group": "pooled", "sources": ["s-a"]}
        with self.assertRaises(temporal_train.TemporalArmError):
            temporal_train.compare_t2_to_h_f0(
                points, frontier.FrontierPoint(0.4, 20.0, 9.0, 0.6), corpora, points
            )

    def test_baseline_frontier_strict_validation(self):
        import tempfile

        from experiments.psem_state_corrected_adaptation_gate import cross_frontier as cross_mod

        def _block(threshold=0.5):
            return cross_mod.build_block(
                [
                    {"threshold": 0.4, "false_cuts_per_hour": 10.0, "contamination": 1.0, "miss_rate": 0.5},
                    {"threshold": 0.5, "false_cuts_per_hour": 20.0, "contamination": 2.0, "miss_rate": 0.6},
                ],
                {"threshold": threshold, "false_cuts_per_hour": 20.0, "contamination": 2.0, "miss_rate": 0.6},
            )

        def _doc(arm="R-H-SC"):
            return {
                "artifact_role": cross_mod.ARTIFACT_ROLE,
                "version": cross_mod.CANONICAL_VERSION,
                "arm": arm,
                "horizons_ms": [100, 300, 500],
                "group_order": list(cross_mod.GROUP_ORDER),
                "horizons": {
                    str(h): {
                        group: {kind: _block() for kind in ("calibrated", "raw")}
                        for group in ("macro", "ami", "alimeeting", "pooled")
                    }
                    for h in (100, 300, 500)
                },
                "sources": {},
            }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "base.json"
            path.write_text(json.dumps(_doc()), encoding="utf-8")
            parsed = temporal_train.read_baseline_frontier(path, "R-H-SC")
            self.assertEqual(
                parsed["horizons"]["100"]["ami"]["calibrated"]["reference"].threshold, 0.5
            )
            self.assertIn("macro", parsed["horizons"]["100"])
            with self.assertRaises(temporal_train.TemporalArmError):
                temporal_train.read_baseline_frontier(path, "R-T2-SC")
            bad = _doc(arm="R-T2-SC")
            path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(temporal_train.TemporalArmError):
                temporal_train.read_baseline_frontier(path, "R-H-SC")
            bad = _doc()
            del bad["horizons"]["300"]
            path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(temporal_train.TemporalArmError):
                temporal_train.read_baseline_frontier(path, "R-H-SC")
            bad = _doc()
            bad["horizons"]["100"]["pooled"]["calibrated"]["reference"]["threshold"] = 0.4
            path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(temporal_train.TemporalArmError):
                temporal_train.read_baseline_frontier(path, "R-H-SC")
            bad = _doc()
            del bad["artifact_role"]
            path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(temporal_train.TemporalArmError):
                temporal_train.read_baseline_frontier(path, "R-H-SC")
            path.write_text("{truncated", encoding="utf-8")
            with self.assertRaises(temporal_train.TemporalArmError):
                temporal_train.read_baseline_frontier(path, "R-H-SC")

    def test_cli_help_runs_standalone(self):
        import subprocess
        import sys

        from experiments.psem_state_corrected_adaptation_gate import run_temporal_arm

        proc = subprocess.run(
            [sys.executable, str(Path(run_temporal_arm.__file__)) , "--help"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("--arm", proc.stdout)
        self.assertIn("R-T2-SC", proc.stdout)

    def test_calibration_role_enforced(self):
        from experiments.psem_state_corrected_adaptation_gate import calibrate

        fit = calibrate.fit_affine_calibrator(
            [0.2, -0.4, 0.6, -0.1], [1.0, 0.0, 1.0, 0.0], "TRAIN-CALIB"
        )
        self.assertEqual(fit["role"], "TRAIN-CALIB")
        with self.assertRaises(calibrate.CalibrationError):
            calibrate.fit_affine_calibrator([0.1], [1.0], "DEV")
if __name__ == "__main__":
    unittest.main()
