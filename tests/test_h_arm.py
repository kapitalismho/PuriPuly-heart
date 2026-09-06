from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.psem_state_corrected_adaptation_gate import arm_runtime, h_arm
from experiments.psem_state_corrected_adaptation_gate.arm_runtime import (
    ARM_R_H_SC,
    ARM_R_T2_SC,
    ArmRunConfig,
    AuthorizationError,
)
from experiments.psem_state_corrected_adaptation_gate.calibrate import CalibrationError


def _h(tag: str) -> str:
    return hashlib.sha256(tag.encode("utf-8")).hexdigest()


def _config(tmp, arm=ARM_R_H_SC, seed=7301):
    return ArmRunConfig(
        arm=arm,
        seed=seed,
        root=Path(tmp) / "arms",
        input_hash=_h("input"),
        checkpoint_hash=_h("checkpoint"),
        partition_hash=_h("partition"),
        weights_hash=_h("weights"),
        code_hash=_h("code"),
    )


def _p0(store, config):
    store = Path(store)
    store.mkdir(parents=True, exist_ok=True)
    (store / "p0_pass.json").write_text(
        json.dumps(
            {
                "verdict": "PASS",
                "input_hash": config.input_hash,
                "checkpoint_hash": config.checkpoint_hash,
                "partition_hash": config.partition_hash,
            }
        ),
        encoding="utf-8",
    )


def _stage_manifest(fit=("s-fit-1", "s-fit-2"), calib=("s-cal-1",)):
    return {
        "fit": sorted(fit),
        "calib": sorted(calib),
        "class_weights": {"replacement_positive_weight": 2.0, "anchor_positive_weight": 1.5},
        "sampling_sha256": _h("sampling"),
        "salt": "issue-121-train-calib-v1",
        "target_frac": 0.12,
        "targets": {},
    }


def _target_payload(source_id, frames=800, corpus="AMI"):
    import numpy as np

    rng = np.random.default_rng(abs(hash(source_id)) % (2**31))
    return {
        "source_id": source_id,
        "num_frames": frames,
        "y_anchor": [1.0 if v > 0.5 else 0.0 for v in rng.random(frames)],
        "y_replace": [1.0 if v > 0.85 else 0.0 for v in rng.random(frames)],
        "valid": [True] * frames,
        "multiplicity": [1] * frames,
        "episode_ids": [f"ep-{i % 3}" for i in range(frames)],
        "sampling_sha256": _h("sampling"),
        "corpus": corpus,
    }


def _double_worker(payload):
    return {"key": payload["key"], "value": payload["value"] * 2}


class FakeWrapper:
    def __init__(self, torch):
        self._linear = torch.nn.Linear(4, 4)
        for p in self._linear.parameters():
            p.requires_grad_(False)
        self.training = False
        self.eval_calls = 0

    def eval(self):
        self.training = False
        self.eval_calls += 1
        return self

    def parameters(self):
        return self._linear.parameters()

    def named_parameters(self):
        return self._linear.named_parameters()


class FakeHead:
    def __init__(self, torch, calls):
        self._torch = torch
        self.proj = torch.nn.Linear(199, 8)
        self.anchor = torch.nn.Linear(8, 1)
        self.resid = torch.nn.Linear(8, 1)
        torch.nn.init.zeros_(self.resid.weight)
        torch.nn.init.zeros_(self.resid.bias)
        self.calls = calls
        self.training = False
        self._mods = [self.proj, self.anchor, self.resid]

    def __call__(self, features, state=None):
        self.calls.append({"none": state is None, "id": None if state is None else id(state)})
        h = self._torch.tanh(self.proj(features))
        out = {
            "anchor_logit": self.anchor(h).squeeze(-1),
            "z_residual": self.resid(h).squeeze(-1),
        }
        return out, features.detach().mean(dim=1)

    def parameters(self):
        for m in self._mods:
            yield from m.parameters()

    def named_parameters(self):
        for i, m in enumerate(self._mods):
            yield from ((f"m{i}.{n}", p) for n, p in m.named_parameters())

    def state_dict(self):
        out = {}
        for i, m in enumerate(self._mods):
            for n, p in m.state_dict().items():
                out[f"m{i}.{n}"] = p
        return out

    def load_state_dict(self, state):
        for i, m in enumerate(self._mods):
            m.load_state_dict({n: state[f"m{i}.{n}"] for n in m.state_dict()})

    def train(self, mode=True):
        self.training = bool(mode)
        return self


def _tensor_targets(torch, payload, zero_ranges=()):
    frames = int(payload["num_frames"])
    mult = [float(m) for m in payload["multiplicity"]]
    for start, end in zero_ranges:
        for i in range(start, min(end, frames)):
            mult[i] = 0.0
    device = "cpu"
    return {
        "num_frames": frames,
        "y_replace": torch.as_tensor([payload["y_replace"]], dtype=torch.float32, device=device),
        "y_anchor": torch.as_tensor([payload["y_anchor"]], dtype=torch.float32, device=device),
        "mult_weight": torch.as_tensor([mult], dtype=torch.float32, device=device),
        "f0": torch.zeros((1, frames), dtype=torch.float32, device=device),
    }


def _tensor_features(torch, payload):
    import numpy as np

    frames = int(payload["num_frames"])
    hidden = np.random.default_rng(7).normal(size=(frames, 192)).astype(np.float32) * 0.1
    logits = np.random.default_rng(9).normal(size=(frames, 4)).astype(np.float32) * 0.1
    selected = logits[:, 0].tolist()
    best = logits.max(axis=1).tolist()
    feats = h_arm.assemble_features(hidden, logits, selected, best, [1.04] * frames)
    return torch.as_tensor(np.asarray(feats, dtype=np.float32)).unsqueeze(0)


def _profile_test_batch(torch, head):
    frames = 2 * h_arm.CHUNK_FRAMES
    device = next(head.parameters()).device
    generator = torch.Generator(device="cpu")
    generator.manual_seed(7301)
    features = (torch.randn((1, frames, 199), generator=generator) * 0.1).to(device)
    targets = ((torch.rand((1, frames), generator=generator) > 0.9).to(torch.float32)).to(device)
    return {
        "source_id": "profile-test",
        "features": features,
        "y_replace": targets,
        "y_anchor": torch.zeros((1, frames), dtype=torch.float32, device=device),
        "mult_weight": torch.ones((1, frames), dtype=torch.float32, device=device),
        "f0": torch.zeros((1, frames), dtype=torch.float32, device=device),
        "windows": [(0, h_arm.CHUNK_FRAMES), (h_arm.CHUNK_FRAMES, frames)],
        "io_bytes": 0,
    }


AMI_DEV = tuple(f"ami-dev-{i}" for i in range(7))
ALI_DEV = tuple(f"ali-dev-{i}" for i in range(3))
FULL_DEV = AMI_DEV + ALI_DEV


def _export_gpu(config, calib=("s-cal-1",), dev=FULL_DEV, fail=False, frames=8, call_head=True):
    def _export(head, wrap, tch):
        if fail:
            raise h_arm.HArmError("export boom")
        if call_head:
            feat = tch.zeros((1, frames, 199))
            out, _ = head(feat)
            resid = [float(v) for v in out["z_residual"].detach().cpu().reshape(-1).tolist()]
            cand = (resid + [0.0] * frames)[:frames]
        else:
            cand = [0.1] * frames
        f0 = [0.0] * frames
        target = [0.0] * (frames // 2) + [1.0] * (frames - frames // 2)
        valid = [True] * frames
        mapped = [True] * frames
        coverage = {"frames": frames, "kept": frames, "positive": frames - frames // 2, "negative": frames // 2}
        calib_table = {}
        for sid in calib:
            h_arm.write_aligned_export_npz(
                h_arm.gpu_export_dir(config.run_dir()) / h_arm.export_npz_name("calib", sid),
                f0, cand, target, valid, mapped,
            )
            calib_table[sid] = h_arm.export_source_entry(
                "calib", sid, frames, "ami_mix_headset", frames, frames, 0, frames, coverage, 0.01,
            )
        dev_table = {}
        for sid in dev:
            family = "ami_mix_headset" if sid in AMI_DEV or str(sid).startswith("ami") else "alimeeting_far_ch0"
            h_arm.write_aligned_export_npz(
                h_arm.gpu_export_dir(config.run_dir()) / h_arm.export_npz_name("dev", sid),
                f0, cand, target, valid, mapped,
            )
            dev_table[sid] = h_arm.export_source_entry(
                "dev", sid, frames, family, frames, frames, 0, frames, coverage, 0.02,
            )
        metrics = config.run_dir() / h_arm.TRAINING_METRICS_NAME
        if not metrics.is_file():
            arm_runtime.atomic_write_json(metrics, {"artifact_role": "issue-121-h-training-metrics"})
        return h_arm.write_gpu_export_manifest(
            config.run_dir(), config, calib_table, dev_table, {"s-fit-1": 0.5}, metrics,
            fit=("s-fit-1", "s-fit-2"),
            salt="issue-121-train-calib-v1",
            target_frac=0.12,
            trained_head_sha256="aa" * 32,
        )


    return _export



class AuthorizationOrderTest(unittest.TestCase):
    def test_h_arm_rejects_other_arms(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp, arm=ARM_R_T2_SC)
            with self.assertRaises(AuthorizationError):
                h_arm.require_h_arm(config)

    def test_blocked_run_never_calls_loaders(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            calls: list = []
            deps = h_arm.HArmDeps(
                load_bundle_manifest=lambda: calls.append("bundle") or {},
                load_wrapper_head=lambda: calls.append("models") or (None, None),
                load_torch=lambda: calls.append("torch"),
            )
            with self.assertRaises(AuthorizationError):
                h_arm.run_h_arm(config, Path(tmp) / "store", deps)
            self.assertEqual(calls, [])

    def test_eval_sources_are_forbidden(self):
        with self.assertRaises(h_arm.HArmError):
            h_arm.forbid_eval(["ami_eval_session"])
        with self.assertRaises(h_arm.HArmError):
            h_arm.forbid_eval(["s1"], {"s1": "EVAL"})


class FreezeTest(unittest.TestCase):
    def test_backbone_eval_head_train(self):
        import torch

        wrapper = FakeWrapper(torch)
        head = FakeHead(torch, [])
        receipt = h_arm.freeze_backbone_train_head(wrapper, head)
        self.assertTrue(receipt["frozen_representation_ok"])
        self.assertEqual(wrapper.eval_calls, 1)
        self.assertTrue(head.training)
        self.assertFalse(any(p.requires_grad for p in wrapper.parameters()))
        self.assertTrue(any(p.requires_grad for p in head.parameters()))

    def test_head_without_trainable_params_is_rejected(self):
        import torch

        class FrozenHead(FakeHead):
            def parameters(self):
                return iter(())

            def named_parameters(self):
                return iter(())

        with self.assertRaises(h_arm.HArmError):
            h_arm.freeze_backbone_train_head(FakeWrapper(torch), FrozenHead(torch, []))


class CacheGeometryTest(unittest.TestCase):
    def test_199_geometry_and_coverage(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            binding = h_arm.h_binding(config, _stage_manifest())
            run_dir = config.run_dir()
            hidden = np.zeros((10, 192), dtype=np.float32)
            logits = np.zeros((10, 4), dtype=np.float32)
            feats = h_arm.assemble_features(hidden, logits, [0.1] * 10, [0.2] * 10, [1.04] * 10)
            self.assertEqual(tuple(feats.shape), (10, 199))
            self.assertEqual(str(feats.dtype), "float32")
            self.assertEqual(sum(arm_runtime.HEAD_INPUT_PARTS.values()), 199)
            with self.assertRaises(h_arm.HArmError):
                h_arm.assemble_features(np.zeros((10, 191), dtype=np.float32), logits, [0.1] * 10, [0.2] * 10, [1.04] * 10)
            with self.assertRaises(h_arm.HArmError):
                h_arm.assemble_features(hidden, logits, [0.1] * 9, [0.2] * 10, [1.04] * 10)
            for source in ("s-fit-1", "s-fit-2", "s-cal-1"):
                h_arm.write_source_cache(
                    run_dir, source, hidden, logits, {"ep-0": 1},
                    [{"anchor_episode_id": "ep-0", "status": "mapped"}], {"t": 1.0}, binding
                )
            records = {
                s: {"file": f"cache/{s}.npz", "sha256": "x", "frames": 10}
                for s in ("s-fit-1", "s-fit-2", "s-cal-1")
            }
            h_arm.write_cache_manifest(run_dir, records, binding)
            h_arm.require_cache_coverage(run_dir, ["s-fit-1", "s-fit-2"], ["s-cal-1"], binding)
            with self.assertRaises(h_arm.HArmError):
                h_arm.require_cache_coverage(run_dir, ["s-fit-1", "s-missing"], ["s-cal-1"], binding)

    def test_hash_mismatch_rejected(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            binding = h_arm.h_binding(config, _stage_manifest(fit=("a",), calib=("b",)))
            run_dir = config.run_dir()
            hidden = np.zeros((5, 192), dtype=np.float32)
            logits = np.zeros((5, 4), dtype=np.float32)
            h_arm.write_source_cache(run_dir, "a", hidden, logits, {}, [], {}, binding)
            reused = h_arm.read_source_cache(run_dir, "a", {**binding, "seed": 9999})
            self.assertEqual(reused["meta"]["source_id"], "a")
            with self.assertRaises(h_arm.HArmError):
                h_arm.read_source_cache(run_dir, "a", {**binding, "checkpoint_hash": "0" * 64})
            path = h_arm.cache_npz_path(run_dir, "a")
            with open(path, "ab") as handle:
                handle.write(b"\x00")
            with self.assertRaises(h_arm.HArmError):
                h_arm.read_source_cache(run_dir, "a", binding)


class ScheduleTest(unittest.TestCase):
    def test_acc16_partial_group_and_zero_loss_exclusion(self):
        flags = {"s1": [True] * 20}
        plan = h_arm.plan_fit_schedule(["s1"], flags)
        self.assertEqual(plan["total_steps"], 2)
        self.assertEqual(plan["warmup_steps"], 1)
        boundaries = [c for c in plan["chunks"] if c["is_step_boundary"]]
        self.assertEqual([c["chunk_index"] for c in boundaries], [15, 19])
        self.assertEqual(boundaries[0]["optimizer_step"], 0)
        self.assertEqual(boundaries[1]["optimizer_step"], 1)
        plan2 = h_arm.plan_fit_schedule(["s1"], {"s1": [True, False, True]})
        self.assertEqual(plan2["loss_chunks"], 2)
        self.assertEqual(plan2["total_steps"], 1)
        self.assertIsNone(plan2["chunks"][1]["accum_position"])

    def test_state_carry_reset_and_zero_chunk_advance(self):
        import torch

        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            binding = h_arm.h_binding(config, _stage_manifest(fit=("s-fit-1", "s-fit-2"), calib=("s-cal-1",)))
            run_dir = config.run_dir()
            calls: list = []
            head = FakeHead(torch, calls)
            wrapper = FakeWrapper(torch)
            payloads = {s: _target_payload(s) for s in ("s-fit-1", "s-fit-2")}
            features = {s: _tensor_features(torch, payloads[s]) for s in payloads}
            targets = {
                "s-fit-1": _tensor_targets(torch, payloads["s-fit-1"]),
                "s-fit-2": _tensor_targets(torch, payloads["s-fit-2"], zero_ranges=((375, 750),)),
            }
            weights = {"replacement_positive_weight": 2.0, "anchor_positive_weight": 1.5}
            sched = h_arm.plan_fit_schedule(
                ["s-fit-1", "s-fit-2"],
                {s: h_arm.loss_flags_for_source(800, targets[s]["mult_weight"]) for s in targets},
            )
            report = h_arm.run_fit_pass(torch, wrapper, head, features, targets, weights, run_dir, binding, sched)
            self.assertEqual(len(calls), 6)
            self.assertTrue(calls[0]["none"])
            self.assertTrue(calls[3]["none"])
            self.assertFalse(calls[1]["none"])
            self.assertFalse(calls[4]["none"])
            self.assertEqual(report["per_source"]["s-fit-2"]["loss_chunks"], 2)
            self.assertEqual(report["steps_taken"], report["total_steps"])
            self.assertEqual(report["per_source"]["s-fit-1"]["steps"], 0)
            self.assertEqual(report["completed_sources"], ["s-fit-1", "s-fit-2"])

    def test_carrier_detach_semantics(self):
        import torch

        from experiments.psem_state_corrected_adaptation_gate import streaming as streaming_mod

        state = (torch.ones(2, 2) * 2).requires_grad_(True)
        carrier = streaming_mod.StateCarrier(state, "s")
        detached = carrier.detach()
        self.assertEqual(carrier.detached_steps, 1)
        self.assertIsNone(detached.grad_fn)
        self.assertTrue(bool((detached == 2).all()))
        carrier.reset(None, "t")
        self.assertIsNone(carrier.carry())
        self.assertEqual(carrier.detached_steps, 0)


class AccumulationTest(unittest.TestCase):
    def _two_chunk_io(self, torch):
        frames = 750
        gen = torch.Generator().manual_seed(5)
        feats = torch.randn(1, frames, 199, generator=gen) * 0.1
        gen2 = torch.Generator().manual_seed(6)
        yr = (torch.rand(1, frames, generator=gen2) > 0.8).float()
        yr[0, 0] = 1.0
        ya = (torch.rand(1, frames, generator=gen2) > 0.5).float()
        return feats, {
            "num_frames": frames,
            "y_replace": yr,
            "y_anchor": ya,
            "mult_weight": torch.ones(1, frames),
            "f0": torch.zeros(1, frames),
        }

    def test_group_mean_matches_explicit_mean(self):
        import torch

        weights = {"replacement_positive_weight": 2.0, "anchor_positive_weight": 1.5}
        sched = h_arm.plan_fit_schedule(["s"], {"s": [True, True]})
        self.assertEqual(sched["total_steps"], 1)
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            binding = h_arm.h_binding(config, _stage_manifest(fit=("s",), calib=("c",)))
            run_dir = config.run_dir()
            torch.manual_seed(11)
            feats, tgt = self._two_chunk_io(torch)
            torch.manual_seed(11)
            engine_head = FakeHead(torch, [])
            h_arm.run_fit_pass(
                torch, FakeWrapper(torch), engine_head, {"s": feats},
                {"s": {k: (v.clone() if hasattr(v, "clone") else v) for k, v in tgt.items()}},
                weights, run_dir, binding, sched,
            )
            torch.manual_seed(11)
            manual_head = FakeHead(torch, [])
            opt = h_arm.build_head_optimizer(torch, manual_head)
            sch = h_arm.build_warmup_scheduler(torch, opt, 1)
            manual_head.train(True)
            opt.zero_grad()
            out0, st0 = manual_head(feats[:, 0:375], None)
            loss0 = h_arm.chunk_loss_value(
                torch, out0["z_residual"] + tgt["f0"][:, 0:375], out0["anchor_logit"],
                tgt["y_replace"][:, 0:375], tgt["y_anchor"][:, 0:375],
                tgt["mult_weight"][:, 0:375], weights,
            ) / 2
            loss0.backward()
            st = st0.detach().clone()
            out1, _ = manual_head(feats[:, 375:750], st)
            loss1 = h_arm.chunk_loss_value(
                torch, out1["z_residual"] + tgt["f0"][:, 375:750], out1["anchor_logit"],
                tgt["y_replace"][:, 375:750], tgt["y_anchor"][:, 375:750],
                tgt["mult_weight"][:, 375:750], weights,
            ) / 2
            loss1.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in manual_head.parameters() if p.requires_grad], h_arm.HEAD_CLIP_NORM
            )
            opt.step()
            sch.step()
            for a, b in zip(engine_head.parameters(), manual_head.parameters()):
                self.assertTrue(bool(torch.equal(a, b)))
            torch.manual_seed(11)
            sum_head = FakeHead(torch, [])
            opt2 = h_arm.build_head_optimizer(torch, sum_head)
            sch2 = h_arm.build_warmup_scheduler(torch, opt2, 1)
            sum_head.train(True)
            opt2.zero_grad()
            o0, t0 = sum_head(feats[:, 0:375], None)
            h_arm.chunk_loss_value(
                torch, o0["z_residual"] + tgt["f0"][:, 0:375], o0["anchor_logit"],
                tgt["y_replace"][:, 0:375], tgt["y_anchor"][:, 0:375],
                tgt["mult_weight"][:, 0:375], weights,
            ).backward()
            tt = t0.detach().clone()
            o1, _ = sum_head(feats[:, 375:750], tt)
            h_arm.chunk_loss_value(
                torch, o1["z_residual"] + tgt["f0"][:, 375:750], o1["anchor_logit"],
                tgt["y_replace"][:, 375:750], tgt["y_anchor"][:, 375:750],
                tgt["mult_weight"][:, 375:750], weights,
            ).backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in sum_head.parameters() if p.requires_grad], h_arm.HEAD_CLIP_NORM
            )
            opt2.step()
            sch2.step()
            self.assertTrue(
                any(not bool(torch.equal(a, b)) for a, b in zip(engine_head.parameters(), sum_head.parameters()))
            )

    def test_group_spans_source_boundary_without_forced_step(self):
        import torch

        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            binding = h_arm.h_binding(config, _stage_manifest(fit=("g1", "g2"), calib=("c",)))
            run_dir = config.run_dir()
            payloads = {s: _target_payload(s, frames=3750) for s in ("g1", "g2")}
            features = {s: _tensor_features(torch, payloads[s]) for s in payloads}
            targets = {s: _tensor_targets(torch, payloads[s]) for s in payloads}
            weights = {"replacement_positive_weight": 2.0, "anchor_positive_weight": 1.5}
            sched = h_arm.plan_fit_schedule(
                ["g1", "g2"],
                {s: h_arm.loss_flags_for_source(3750, targets[s]["mult_weight"]) for s in targets},
            )
            self.assertEqual(sched["total_steps"], 2)
            head = FakeHead(torch, [])
            report = h_arm.run_fit_pass(
                torch, FakeWrapper(torch), head, features, targets, weights, run_dir, binding, sched
            )
            self.assertEqual(report["steps_taken"], 2)
            self.assertEqual(report["per_source"]["g1"]["steps"], 0)
            self.assertEqual(report["per_source"]["g2"]["steps"], 2)
            self.assertEqual(report["accum_count"], 0)


class ProfileGateTest(unittest.TestCase):
    def test_full_run_refuses_without_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            _p0(Path(tmp) / "store", config)
            deps = h_arm.HArmDeps(
                load_bundle_manifest=lambda: _stage_manifest(),
                bundle_dir=Path(tmp),
                load_wrapper_head=lambda: (None, None),
                load_torch=lambda: __import__("torch"),
            )
            with self.assertRaises(h_arm.HArmError):
                h_arm.run_h_arm(config, Path(tmp) / "store", deps)

    def test_profile_runs_real_optimizer_steps(self):
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            _p0(Path(tmp) / "store", config)
            manifest = _stage_manifest()
            binding = h_arm.h_binding(config, manifest)
            head = FakeHead(torch, [])
            before = [p.detach().clone() for p in head.parameters()]
            deps = h_arm.HArmDeps(
                load_bundle_manifest=lambda: manifest,
                load_wrapper_head=lambda: (FakeWrapper(torch), head),
                load_torch=lambda: torch,
                profile_batch=lambda hm, w, t: _profile_test_batch(t, hm),
                profile_dev_sample=lambda hm, w, t: {"infer_seconds": 1.0, "io_bytes": 8},
            )
            out = h_arm.run_profile_command(config, Path(tmp) / "store", deps)
            self.assertEqual(out["profile"]["optimizer_steps"], 8)
            self.assertGreater(out["profile"]["seconds_per_step"], 0)
            self.assertIn("peak_vram_bytes", out["profile"])
            self.assertIn("projected_train_seconds", out["profile"])
            self.assertEqual(out["profile"]["dev_infer_seconds"], 1.0)
            self.assertFalse(out["profile"]["authoritative"])
            self.assertEqual(out["profile"]["scope"], "profile-only")
            self.assertTrue(out["profile"]["stateful"])
            self.assertEqual(out["profile"]["windows"], 2)
            self.assertEqual(len(head.calls), 8)
            carried = [c for c in head.calls if not c["none"]]
            self.assertGreater(len(carried), 0)
            self.assertGreater(len({c["id"] for c in carried}), 1)
            for a, b in zip(before, head.parameters()):
                self.assertTrue(bool(torch.equal(a, b)))
            required = h_arm.require_profile(config.run_dir(), binding)
            self.assertEqual(required["optimizer_steps"], 8)
            with self.assertRaises(h_arm.HArmError):
                h_arm.run_profile(config.run_dir(), binding, lambda: {}, lambda: {}, 0, steps=4)

    def test_profile_restores_state_on_dev_failure(self):
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            _p0(Path(tmp) / "store", config)
            manifest = _stage_manifest()
            wrapper = FakeWrapper(torch)
            head = FakeHead(torch, [])
            before = [p.detach().clone() for p in head.parameters()]
            head_mode_before = bool(head.training)
            wrapper_mode_before = bool(wrapper.training)

            def _boom(hm, w, t):
                raise h_arm.HArmError("dev boom")

            deps = h_arm.HArmDeps(
                load_bundle_manifest=lambda: manifest,
                load_wrapper_head=lambda: (wrapper, head),
                load_torch=lambda: torch,
                profile_batch=lambda hm, w, t: _profile_test_batch(t, hm),
                profile_dev_sample=_boom,
            )
            with self.assertRaises(h_arm.HArmError):
                h_arm.run_profile_command(config, Path(tmp) / "store", deps)
            for a, b in zip(before, head.parameters()):
                self.assertTrue(bool(torch.equal(a, b)))
            self.assertEqual(bool(head.training), head_mode_before)
            self.assertEqual(bool(wrapper.training), wrapper_mode_before)

    def test_profile_refuses_without_real_batch_or_dev(self):
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            _p0(Path(tmp) / "store", config)
            manifest = _stage_manifest()
            head = FakeHead(torch, [])
            no_batch = h_arm.HArmDeps(
                load_bundle_manifest=lambda: manifest,
                load_wrapper_head=lambda: (FakeWrapper(torch), head),
                load_torch=lambda: torch,
                profile_dev_sample=lambda hm, w, t: {"infer_seconds": 1.0, "io_bytes": 0},
            )
            with self.assertRaises(h_arm.HArmError):
                h_arm.run_profile_command(config, Path(tmp) / "store", no_batch)
            no_dev = h_arm.HArmDeps(
                load_bundle_manifest=lambda: manifest,
                load_wrapper_head=lambda: (FakeWrapper(torch), head),
                load_torch=lambda: torch,
                profile_batch=lambda hm, w, t: _profile_test_batch(t, hm),
            )
            with self.assertRaises(h_arm.HArmError):
                h_arm.run_profile_command(config, Path(tmp) / "store", no_dev)


class CalibrationFrontierTest(unittest.TestCase):
    def test_calib_only_and_dev_frontier(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            binding = h_arm.h_binding(config, _stage_manifest())
            run_dir = config.run_dir()
            with self.assertRaises(CalibrationError):
                from experiments.psem_state_corrected_adaptation_gate.calibrate import (
                    fit_affine_calibrator,
                )

                fit_affine_calibrator([0.0, 1.0], [0.0, 1.0], "DEV")
            calib_raw = {
                "s-cal-1": {
                    "f0": [-2.0, -1.0, 0.0, 1.0, 2.0, -1.5, 0.5, 1.5],
                    "candidate": [-1.8, -0.9, 0.2, 1.2, 2.2, -1.2, 0.7, 1.7],
                    "target": [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
                }
            }
            metrics = h_arm.run_calibration_stage(run_dir, binding, calib_raw, workers=1)
            self.assertGreater(float(metrics["f0"]["slope"]), 0)
            self.assertGreater(float(metrics["candidate"]["slope"]), 0)
            self.assertLessEqual(float(metrics["candidate"]["nll"]), float(metrics["candidate"]["raw_nll"]))
            loaded = arm_runtime.load_source_predictions(run_dir, "s-cal-1", binding)
            self.assertEqual(len(loaded["candidate_logit"]), 8)
            with self.assertRaises(arm_runtime.CheckpointError):
                arm_runtime.load_source_predictions(run_dir, "s-cal-1", {**binding, "seed": 9999})
            calibrators = {
                "f0": {"slope": 1.5, "intercept": 0.25},
                "candidate": {"slope": 0.75, "intercept": -0.5},
            }
            dev_scores = {
                "dev-ami": {
                    "f0": [0.2 * i - 1.0 for i in range(10)],
                    "candidate": [0.3 * i - 1.2 for i in range(10)],
                    "target": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    "mapped": [True] * 10,
                },
                "dev-ali": {
                    "f0": [0.25 * i - 0.8 for i in range(10)],
                    "candidate": [0.2 * i - 0.9 for i in range(10)],
                    "target": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                    "mapped": [True] * 8 + [False, False],
                },
            }
            tables = {}
            for source, entry in dev_scores.items():
                conv = h_arm.dev_frontier_inputs(
                    entry["f0"], entry["candidate"], entry["mapped"], calibrators
                )
                tables[source] = {}
                for horizon in (100, 300, 500):
                    tables[source][horizon] = {
                        "f0": (4.0, 0.2, 0.3),
                        "by_threshold_raw": {t: (7.0 - i * 0.4, 0.22 - i * 0.01, 0.32 - i * 0.01) for i, t in enumerate(conv["thresholds_raw"])},
                        "by_threshold_calibrated": {t: (8.0 - i * 0.5, 0.25 - i * 0.01, 0.35 - i * 0.01) for i, t in enumerate(conv["thresholds_calibrated"])},
                    }
            doc = h_arm.run_dev_frontier(
                run_dir, binding, dev_scores, tables,
                {"dev-ami": "AMI", "dev-ali": "AliMeeting"}, calibrators,
                {
                    name: {
                        hz: {
                            "kinds": {
                                kind: {
                                    "thresholds": list(tables["dev-ami"][hz][f"by_threshold_{kind}"]),
                                    "points": [[t, m[0], m[1], m[2]] for t, m in tables["dev-ami"][hz][f"by_threshold_{kind}"].items()],
                                    "f0": list(tables["dev-ami"][hz]["f0"]),
                                }
                                for kind in ("raw", "calibrated")
                            },
                        }
                        for hz in (100, 300, 500)
                    }
                    for name in ("AMI", "AliMeeting", "pooled")
                },
                workers=1,
            )
            self.assertEqual(doc["artifact_role"], "issue-121-cross-arm-dev-frontier")
            self.assertEqual(doc["arm"], "R-H-SC")
            self.assertEqual(sorted(doc["sources"]), ["dev-ali", "dev-ami"])
            for source in ("dev-ami", "dev-ali"):
                for horizon in ("100", "300", "500"):
                    for kind in ("raw", "calibrated"):
                        node = doc["sources"][source][horizon][kind]
                        self.assertEqual(node["reference"]["threshold"], 0.5)
                        self.assertEqual(node["budget"], 4.0)
                        finite = [p["threshold"] for p in node["points"] if p["threshold"] != float("-inf")]
                        self.assertTrue(finite)
                        for threshold in finite:
                            self.assertGreaterEqual(threshold, 0.0)
                            self.assertLessEqual(threshold, 1.0)
            for horizon in ("100", "300", "500"):
                groups = doc["horizons"][horizon]
                self.assertEqual(sorted(groups), ["alimeeting", "ami", "macro", "pooled"])
                for group in ("macro", "ami", "alimeeting", "pooled"):
                    for kind in ("raw", "calibrated"):
                        self.assertEqual(groups[group][kind]["reference"]["threshold"], 0.5)
                        self.assertEqual(groups[group][kind]["budget"], groups[group][kind]["reference"]["false_cuts_per_hour"])
                        self.assertTrue(groups[group][kind]["points"])
            for source in ("dev-ami", "dev-ali"):
                saved = arm_runtime.load_source_predictions(run_dir, f"dev_{source}", binding)
                for key in ("f0_logit", "candidate_logit", "f0_prob", "candidate_prob",
                            "f0_cal_logit", "candidate_cal_logit", "target", "mapped"):
                    self.assertIn(key, saved)
            self.assertIn("raw_ap", doc["sources"]["dev-ami"]["100"]["raw"]["diagnostics"])
            self.assertIn("candidate_cal_nll", doc["sources"]["dev-ami"]["100"]["raw"]["diagnostics"])
            with self.assertRaises(h_arm.HArmError):
                h_arm.run_dev_frontier(run_dir, binding, {"x-eval": dev_scores["dev-ami"]}, {"x-eval": tables["dev-ami"]}, {}, calibrators)

    def test_worker_parity_through_shared_helper(self):
        payloads = [{"key": f"k{i}", "value": i} for i in range(12)]
        seq = arm_runtime.ordered_process_map(_double_worker, payloads, 1)
        par = arm_runtime.ordered_process_map(_double_worker, payloads, 2)
        self.assertEqual(seq, par)
        self.assertEqual(arm_runtime.resolve_workers(None, cap=8), min(8, __import__("os").cpu_count() or 1))


class EndToEndTest(unittest.TestCase):
    def _deps(self, torch, manifest, payloads, config, export_fail=False):
        wrapper = FakeWrapper(torch)
        head = FakeHead(torch, [])

        def _features(cached):
            import numpy as np

            target = cached["target"]
            hidden = np.asarray(cached["hidden192"])
            logits = np.asarray(cached["slot_logits4"])
            feats = h_arm.assemble_features(hidden, logits, logits[:, 0].tolist(), logits.max(axis=1).tolist(), [1.04] * len(hidden))
            _ = target
            return torch.as_tensor(np.asarray(feats, dtype=np.float32)).unsqueeze(0)

        return (
            h_arm.HArmDeps(
                load_bundle_manifest=lambda: manifest,
                bundle_dir=Path("/none"),
                build_missing_targets=lambda s: dict(payloads[s]),
                build_evidence=lambda s, p: {
                    "hidden192": __import__("numpy").random.default_rng(3).normal(size=(p["num_frames"], 192)).astype("float32"),
                    "slot_logits4": __import__("numpy").zeros((p["num_frames"], 4), dtype="float32"),
                    "slot_of": {"ep-0": 0},
                    "mapping_rows": [{"anchor_episode_id": "ep-0", "status": "mapped"}],
                    "timing": {},
                },
                build_features=_features,
                build_targets=lambda s, p: _tensor_targets(torch, p),
                load_wrapper_head=lambda: (wrapper, head),
                load_torch=lambda: torch,
                export_gpu_evidence=_export_gpu(config, fail=export_fail),
                workers=1,
            ),
            head,
        )


    def test_full_run_writes_final_manifest_last(self):
        import torch

        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            _p0(Path(tmp) / "store", config)
            manifest = _stage_manifest()
            binding = h_arm.h_binding(config, manifest)
            h_arm.run_profile(
                config.run_dir(), binding, lambda: {"io_bytes": 0},
                lambda: {"infer_seconds": 1.0, "io_bytes": 0}, 0, torch=torch, device=None, steps=8,
            )
            payloads = {s: _target_payload(s, frames=800) for s in ("s-fit-1", "s-fit-2", "s-cal-1")}
            deps, head = self._deps(torch, manifest, payloads, config)
            before = [p.detach().clone() for p in head.parameters()]
            out = h_arm.run_h_arm(config, Path(tmp) / "store", deps)
            run_dir = Path(out["run_dir"])
            for name in (
                "experiment_manifest.json",
                "data_sampling_calibration_manifest.json",
                "parameter_module_mode_receipt.json",
                "training_metrics.json",
                "gpu_export/gpu_export_manifest.json",
                "cache_manifest.json",
                "profile_receipt.json",
                "final_manifest.json",
            ):
                self.assertTrue((run_dir / name).is_file(), name)
            self.assertFalse((run_dir / "calibration_metrics.json").is_file())
            self.assertFalse((run_dir / "dev_frontier.json").is_file())
            final = json.loads((run_dir / "final_manifest.json").read_text(encoding="utf-8"))
            paths = [a["path"] for a in final["artifacts"]]
            normed = [p.replace("\\", "/") for p in paths]
            self.assertEqual(sum(1 for p in paths if p.endswith(".npz") and "gpu_export" in p.replace("\\", "/")), 1 + len(FULL_DEV))
            self.assertEqual(sum(1 for p in paths if p.endswith(".npz") and "/cache/" in p.replace("\\", "/")), 3)
            self.assertEqual(sum(1 for p in normed if "/cache/" in p and p.endswith(".json")), 3)
            self.assertEqual(sum(1 for p in paths if p.endswith(".pt")), 8)

            self.assertTrue(any(p.endswith("checkpoints/checkpoint.json") for p in normed))
            export_doc = json.loads((run_dir / "gpu_export" / "gpu_export_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(export_doc["artifact_role"], "issue-121-h-gpu-export")
            self.assertEqual(export_doc["binding"], config.binding)

            self.assertEqual(sorted(export_doc["dev_sources"]), sorted(FULL_DEV))
            self.assertEqual(len([s for s in export_doc["dev_sources"] if s in AMI_DEV]), 7)
            self.assertEqual(len([s for s in export_doc["dev_sources"] if s in ALI_DEV]), 3)
            ledger = json.loads(
                (run_dir / "checkpoints" / "checkpoint.json").read_text(encoding="utf-8")
            )
            self.assertEqual(ledger["completed_sources"], ["s-fit-1", "s-fit-2"])
            self.assertIn("s-fit-2", ledger["sources"])
            self.assertIn("s-fit-1", ledger["sources"])
            after = list(head.parameters())
            self.assertTrue(any(not bool(torch.equal(a, b)) for a, b in zip(before, after)))


    def test_partial_failure_writes_no_final_manifest(self):
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            _p0(Path(tmp) / "store", config)
            manifest = _stage_manifest()
            binding = h_arm.h_binding(config, manifest)
            h_arm.run_profile(
                config.run_dir(), binding, lambda: {"io_bytes": 0},
                lambda: {"infer_seconds": 1.0, "io_bytes": 0}, 0, torch=torch, device=None, steps=8,
            )
            payloads = {s: _target_payload(s, frames=800) for s in ("s-fit-1", "s-fit-2", "s-cal-1")}
            deps, _ = self._deps(torch, manifest, payloads, config, export_fail=True)

            with self.assertRaises(h_arm.HArmError):
                h_arm.run_h_arm(config, Path(tmp) / "store", deps)
            self.assertFalse((config.run_dir() / "final_manifest.json").is_file())

    def test_missing_targets_materialized_without_mutating_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            manifest = _stage_manifest()
            run_dir = config.run_dir()
            payloads = h_arm.ensure_full_source_targets(
                Path(tmp) / "bundle",
                manifest,
                run_dir,
                lambda s: _target_payload(s, frames=100, corpus="AliMeeting" if s == "s-cal-1" else "AMI"),
            )
            self.assertEqual(sorted(payloads), ["s-cal-1", "s-fit-1", "s-fit-2"])
            self.assertFalse((Path(tmp) / "bundle").exists())
            self.assertTrue((run_dir / h_arm.H_TARGETS_DIRNAME / "s-fit-1.json").is_file())
            reused = h_arm.ensure_full_source_targets(Path(tmp) / "bundle", manifest, run_dir, None)
            self.assertEqual(sorted(reused), ["s-cal-1", "s-fit-1", "s-fit-2"])
            empty = config.run_dir().parent / "empty"
            with self.assertRaises(h_arm.HArmError):
                h_arm.ensure_full_source_targets(Path(tmp) / "bundle", manifest, empty, None)



class PodEntryTest(unittest.TestCase):
    def _inputs(self, tmp):
        return h_arm.HArmPodInputs(
            bundle_dir=Path(tmp) / "bundle",
            checkpoint=Path(tmp) / "ckpt",
            nemo_checkout=Path(tmp) / "nemo",
            dependency_lock=Path(tmp) / "lock",
            corpus_root=Path(tmp) / "corpus",
            reference_root=Path(tmp) / "reference",
            sampling_manifest=Path(tmp) / "sampling.jsonl",
        )

    def test_pod_entries_authorize_before_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            inputs = self._inputs(tmp)
            with self.assertRaises(AuthorizationError):
                h_arm.run_h_arm_pod(config, Path(tmp) / "store", inputs)
            with self.assertRaises(AuthorizationError):
                h_arm.run_profile_pod(config, Path(tmp) / "store", inputs)

    def test_pure_pod_helpers(self):
        selected, best = h_arm.pod_selected_best([[0.5, 0.1, 0.2, 0.3]], ["e"], {"e": 2})
        self.assertEqual(selected, [0.2])
        self.assertEqual(best, [0.5])
        selected, best = h_arm.pod_selected_best([[0.5, 0.1, 0.2, 0.3]], [None], {})
        self.assertEqual(selected, [0.5])
        self.assertEqual(best, [0.3])
        self.assertAlmostEqual(h_arm.pod_f0_from_selected([0.0])[0], 0.0)
        manifest = {"fit": ["a", "b"], "calib": ["c"], "targets": {}}
        payloads = {
            "a": _target_payload("a", frames=800),
            "b": _target_payload("b", frames=400),
        }
        self.assertEqual(h_arm.pod_planned_train_steps(manifest, payloads), 1)
        with self.assertRaises(h_arm.HArmError):
            h_arm.pod_planned_train_steps(manifest, {"a": payloads["a"]})

    def test_pod_stage_manifest_uses_concrete_verifier(self):
        from experiments.psem_state_corrected_adaptation_gate import receipts as receipts_mod

        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "bundle"
            bundle.mkdir(parents=True)
            body = {"fit": ["a"], "calib": ["c"], "targets": {"a": {"num_frames": 800}}, "files": {},
                    "class_weights": {"replacement_positive_weight": 2.0, "anchor_positive_weight": 1.5},
                    "sampling_sha256": _h("sampling"), "salt": "s", "target_frac": 0.12}
            receipts_mod.write_json(bundle / "stage_a_manifest.json", body)
            inputs = self._inputs(tmp)
            manifest = h_arm.pod_stage_manifest(inputs)
            self.assertEqual(manifest["fit"], ["a"])
            deps = h_arm.pod_deps(_config(tmp), inputs, manifest, {"a": _target_payload("a", frames=800)})
            for field in ("load_bundle_manifest", "build_evidence", "build_features", "build_targets", "load_wrapper_head", "load_torch", "export_gpu_evidence", "profile_batch", "profile_dev_sample"):

                self.assertIsNotNone(getattr(deps, field), field)
            self.assertEqual(deps.total_profile_steps, 1)

    def test_cli_dispatch_blocked_without_callbacks(self):
        from experiments.psem_state_corrected_adaptation_gate import run_h_arm as cli

        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({
                "arm": config.arm, "seed": config.seed, "root": str(config.root),
                "input_hash": config.input_hash, "checkpoint_hash": config.checkpoint_hash,
                "partition_hash": config.partition_hash, "weights_hash": config.weights_hash,
                "code_hash": config.code_hash,
            }), encoding="utf-8")
            base = ["--command", "run", "--config", str(path), "--store", str(Path(tmp) / "store"),
                    "--bundle-dir", str(Path(tmp) / "bundle"), "--checkpoint", "c", "--nemo-checkout", "n",
                    "--dependency-lock", "l", "--corpus-root", "r", "--reference-root", "f",
                    "--sampling-manifest", "s"]
            self.assertEqual(cli.main(base), 3)
            self.assertEqual(cli.main(["--command", "profile"] + base[2:]), 3)


class CacheSinglePassTest(unittest.TestCase):
    def test_cache_uses_one_continuous_pass(self):
        from unittest import mock

        import torch

        from experiments.psem_state_corrected_adaptation_gate import material as material_mod
        from experiments.psem_state_corrected_adaptation_gate import stages as stages_mod

        frames = 800
        audio = torch.zeros(1, frames * 1280)
        hidden = torch.randn(1, frames, 192) * 0.1
        logits = torch.randn(1, frames, 4) * 0.1
        probs = torch.softmax(logits, dim=-1)
        canned = {
            "windows": [{
                "hidden": hidden, "logits": logits, "probabilities": probs,
                "emitted_frames": frames, "steps": 3,
            }],
            "state_out": None,
            "boundary_steps": [3],
        }
        payload = {
            "source_id": "cs", "num_frames": frames, "audio_ref": "a", "waveform_sha256": "w",
            "y_anchor": [1.0] * frames, "y_replace": [0.0] * frames, "valid": [True] * frames,
            "multiplicity": [1] * frames, "episode_ids": ["ep-0"] * frames,
        }
        with tempfile.TemporaryDirectory() as tmp:
            inputs = h_arm.HArmPodInputs(
                bundle_dir=Path(tmp), checkpoint=Path(tmp), nemo_checkout=Path(tmp),
                dependency_lock=Path(tmp), corpus_root=Path(tmp), reference_root=Path(tmp),
                sampling_manifest=Path(tmp),
            )
            ctx = {"torch": torch, "wrapper": object(), "head": None, "device": "cpu"}
            with mock.patch.object(stages_mod, "load_waveform_bytes", return_value=(audio, 16000)), mock.patch.object(
                material_mod, "run_adjacent_windows", return_value=canned
            ) as adj, mock.patch.object(
                material_mod, "infer_slice_source_evidence", side_effect=AssertionError("duplicate")
            ):
                ev = h_arm.pod_source_evidence(inputs, ctx, "cs", payload)
            self.assertEqual(adj.call_count, 1)
            self.assertEqual(tuple(ev["hidden192"].shape), (frames, 192))
            self.assertEqual(tuple(ev["slot_logits4"].shape), (frames, 4))
            self.assertTrue(any(r.get("status") == "mapped" for r in ev["mapping_rows"]))
            self.assertEqual(ev["timing"]["windows"], 1)
            self.assertEqual(ev["timing"]["unmapped_frames"], [])


if __name__ == "__main__":
    unittest.main()
class PlannedStepsExactTest(unittest.TestCase):
    def _payload(self, source_id, frames, valid_zero=(), mult_one=True):
        return {
            "source_id": source_id,
            "num_frames": frames,
            "y_anchor": [0.0] * frames,
            "y_replace": [1.0 if i % 375 == 0 else 0.0 for i in range(frames)],
            "valid": [False if any(s <= i < e for s, e in valid_zero) else True for i in range(frames)],
            "multiplicity": [1 if mult_one else 0] * frames,
            "episode_ids": [None] * frames,
        }

    def test_valid_masking_drives_flags(self):
        frames = 800
        payload = self._payload("a", frames, valid_zero=((0, 375),))
        flags = h_arm.loss_flags_for_source(frames, h_arm.target_mult_list(payload, ()))
        self.assertEqual(flags, [False, True, True])
        manifest = {"fit": ["a"], "calib": ["c"]}
        self.assertEqual(h_arm.pod_planned_train_steps(manifest, {"a": payload}), 1)
        with self.assertRaises(h_arm.HArmError):
            h_arm.pod_planned_train_steps(manifest, {})

    def test_mapping_eligibility_counts_exactly(self):
        import numpy as np

        frames = 17 * 375
        payload = self._payload("a", frames)
        manifest = {
            "fit": ["a"], "calib": ["c"],
            "class_weights": {"replacement_positive_weight": 2.0, "anchor_positive_weight": 1.5},
            "sampling_sha256": _h("sampling"), "salt": "s", "target_frac": 0.12, "targets": {},
        }
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp, seed=7301)
            binding = h_arm.h_binding(config, manifest)
            run_dir = config.run_dir()
            hidden = np.zeros((frames, 192), dtype=np.float32)
            logits = np.zeros((frames, 4), dtype=np.float32)
            h_arm.write_source_cache(
                run_dir, "a", hidden, logits, {}, [],
                {"unmapped_frames": list(range(0, 375))}, binding,
            )
            self.assertEqual(h_arm.pod_planned_train_steps(manifest, {"a": payload}), 2)
            self.assertEqual(
                h_arm.pod_planned_train_steps(manifest, {"a": payload}, run_dir, binding), 1
            )
            bare = h_arm.pod_planned_train_steps(
                {"fit": ["a"], "calib": ["c"]}, {"a": dict(payload, valid=[True] * frames)}
            )
            self.assertEqual(bare, 2)


class ProfileBatchSelectionTest(unittest.TestCase):
    def _fixtures(self, torch):
        import numpy as np

        frames = 800
        hidden = np.zeros((frames, 192), dtype=np.float32)
        logits = np.zeros((frames, 4), dtype=np.float32)
        evidence = {
            "hidden192": hidden,
            "slot_logits4": logits,
            "slot_of": {"ep-0": 0},
            "mapping_rows": [{"anchor_episode_id": "ep-0", "status": "mapped"}],
            "timing": {"unmapped_frames": []},
        }
        manifest = {"fit": ["p1", "p2"], "calib": ["c"]}
        p1 = {
            "source_id": "p1", "num_frames": frames,
            "y_anchor": [0.0] * frames, "y_replace": [0.0] * frames,
            "valid": [True] * frames, "multiplicity": [0] * frames,
            "episode_ids": ["ep-0"] * frames,
        }
        replace = [0.0] * frames
        replace[400] = 1.0
        p2 = dict(p1, source_id="p2", multiplicity=[1] * frames, y_replace=replace)
        return manifest, {"p1": p1, "p2": p2, "c": dict(p1, source_id="c")}, evidence


    def test_first_supported_window_wins(self):
        from unittest import mock

        import torch

        tmp = tempfile.mkdtemp()
        manifest, payloads, evidence = self._fixtures(torch)
        inputs = h_arm.HArmPodInputs(
            bundle_dir="b", checkpoint="c", nemo_checkout="n", dependency_lock="l",
            corpus_root="r", reference_root="f", sampling_manifest="s",
        )
        ctx = {"torch": torch, "device": "cpu"}
        seen: list = []

        def _evidence(i, c, sid, p):
            seen.append(sid)
            return dict(evidence)

        with mock.patch.object(h_arm, "pod_source_evidence", side_effect=_evidence), mock.patch.object(
            h_arm, "pod_payload_for", side_effect=lambda i, r, m, sid: dict(payloads[sid])
        ):
            batch = h_arm.pod_profile_batch(inputs, ctx, Path(tmp) / "run", {}, manifest, Path(tmp))
        self.assertEqual(seen, ["p1", "p2"])
        self.assertEqual(batch["source_id"], "p2")
        self.assertEqual((batch["chunk_start"], batch["chunk_end"]), (375, 750))
        self.assertEqual(tuple(batch["features"].shape), (1, 800, 199))
        self.assertEqual(batch["windows"], [(0, 375), (375, 750), (750, 800)])
        self.assertGreater(float(batch["mult_weight"].sum()), 0)

    def test_refusal_without_support(self):
        from unittest import mock

        import torch

        manifest, payloads, evidence = self._fixtures(torch)
        payloads["p2"] = dict(payloads["p2"], multiplicity=[0] * 800, y_replace=[0.0] * 800)
        inputs = h_arm.HArmPodInputs(
            bundle_dir="b", checkpoint="c", nemo_checkout="n", dependency_lock="l",
            corpus_root="r", reference_root="f", sampling_manifest="s",
        )
        ctx = {"torch": torch, "device": "cpu"}
        with mock.patch.object(h_arm, "pod_source_evidence", return_value=dict(evidence)), mock.patch.object(
            h_arm, "pod_payload_for", side_effect=lambda i, r, m, sid: dict(payloads[sid])
        ):
            with self.assertRaises(h_arm.HArmError):
                h_arm.pod_profile_batch(inputs, ctx, Path(tempfile.mkdtemp()) / "run", {}, manifest, Path(tempfile.mkdtemp()))


class ValidMaskingTest(unittest.TestCase):
    def test_valid_and_mapped_mask_multiply(self):
        import numpy as np
        import torch

        frames = 10
        payload = {
            "source_id": "v", "num_frames": frames,
            "y_anchor": [0.0] * frames, "y_replace": [1.0] * frames,
            "valid": [True, False] + [True] * (frames - 2),
            "multiplicity": [1] * frames,
            "episode_ids": ["ep-0"] * frames,
        }
        manifest = {
            "fit": ["v"], "calib": ["c"],
            "class_weights": {"replacement_positive_weight": 2.0, "anchor_positive_weight": 1.5},
            "sampling_sha256": _h("sampling"), "salt": "s", "target_frac": 0.12, "targets": {},
        }
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            binding = h_arm.h_binding(config, manifest)
            run_dir = config.run_dir()
            h_arm.write_source_cache(
                run_dir, "v", np.zeros((frames, 192), dtype=np.float32),
                np.zeros((frames, 4), dtype=np.float32), {"ep-0": 0},
                [{"anchor_episode_id": "ep-0", "status": "mapped"}],
                {"unmapped_frames": [0]}, binding,
            )
            ctx = {"torch": torch, "device": "cpu"}
            target = h_arm.pod_target_tensors(ctx, run_dir, binding, "v", payload)
            mult = target["mult_weight"][0].tolist()
            self.assertEqual(mult[0], 0.0)
            self.assertEqual(mult[1], 0.0)
            self.assertEqual(mult[2], 1.0)
            self.assertEqual(sum(mult), frames - 2)


class CacheResumeTest(unittest.TestCase):
    def _deps(self, torch, manifest, payloads, calls, config):
        wrapper = FakeWrapper(torch)
        head = FakeHead(torch, [])

        def _features(cached):
            import numpy as np

            hidden = np.asarray(cached["hidden192"])
            logits = np.asarray(cached["slot_logits4"])
            feats = h_arm.assemble_features(hidden, logits, logits[:, 0].tolist(), logits.max(axis=1).tolist(), [1.04] * len(hidden))
            return torch.as_tensor(np.asarray(feats, dtype=np.float32)).unsqueeze(0)

        def _evidence(source_id, payload):
            calls.append(source_id)
            import numpy as np

            frames = int(payload["num_frames"])
            return {
                "hidden192": np.zeros((frames, 192), dtype=np.float32),
                "slot_logits4": np.zeros((frames, 4), dtype=np.float32),
                "slot_of": {"ep-0": 0},
                "mapping_rows": [{"anchor_episode_id": "ep-0", "status": "mapped"}],
                "timing": {},
            }

        return h_arm.HArmDeps(
            load_bundle_manifest=lambda: manifest,
            bundle_dir=Path("/none"),
            build_missing_targets=lambda s: dict(payloads[s]),
            build_evidence=_evidence,
            build_features=_features,
            build_targets=lambda s, p: _tensor_targets(torch, p),
            load_wrapper_head=lambda: (wrapper, head),
            load_torch=lambda: torch,
            export_gpu_evidence=_export_gpu(config, calib=("kc",)),
            workers=1,
        )


    def _manifest(self):
        return _stage_manifest(fit=("k1", "k2"), calib=("kc",))

    def test_resume_runs_inference_only_for_missing(self):
        import numpy as np
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            _p0(Path(tmp) / "store", config)
            manifest = self._manifest()
            binding = h_arm.h_binding(config, manifest)
            h_arm.run_profile(
                config.run_dir(), binding, lambda: {"io_bytes": 0},
                lambda: {"infer_seconds": 1.0, "io_bytes": 0}, 0, torch=torch, device=None, steps=8,
            )
            run_dir = config.run_dir()
            h_arm.write_source_cache(
                run_dir, "k1", np.zeros((400, 192), dtype=np.float32),
                np.zeros((400, 4), dtype=np.float32), {"ep-0": 0},
                [{"anchor_episode_id": "ep-0", "status": "mapped"}], {}, binding,
            )
            payloads = {s: _target_payload(s, frames=400) for s in ("k1", "k2", "kc")}
            calls: list = []
            out = h_arm.run_h_arm(config, Path(tmp) / "store", self._deps(torch, manifest, payloads, calls, config))

            self.assertEqual(sorted(calls), ["k2", "kc"])
            self.assertTrue((run_dir / "final_manifest.json").is_file())
            _ = out

    def test_corrupt_entry_fails_closed(self):
        import numpy as np
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            _p0(Path(tmp) / "store", config)
            manifest = self._manifest()
            binding = h_arm.h_binding(config, manifest)
            h_arm.run_profile(
                config.run_dir(), binding, lambda: {"io_bytes": 0},
                lambda: {"infer_seconds": 1.0, "io_bytes": 0}, 0, torch=torch, device=None, steps=8,
            )
            run_dir = config.run_dir()
            h_arm.write_source_cache(
                run_dir, "k1", np.zeros((400, 192), dtype=np.float32),
                np.zeros((400, 4), dtype=np.float32), {"ep-0": 0},
                [{"anchor_episode_id": "ep-0", "status": "mapped"}], {}, binding,
            )
            npz = h_arm.cache_npz_path(run_dir, "k1")
            before = npz.read_bytes()
            with open(npz, "ab") as handle:
                handle.write(b"\x00")
            payloads = {s: _target_payload(s, frames=400) for s in ("k1", "k2", "kc")}
            calls: list = []
            with self.assertRaises(h_arm.HArmError):
                h_arm.run_h_arm(config, Path(tmp) / "store", self._deps(torch, manifest, payloads, calls, config))

            self.assertEqual(npz.read_bytes(), before + b"\x00")
            self.assertNotIn("k1", calls)
            self.assertFalse((run_dir / "final_manifest.json").is_file())


class PodBindingTest(unittest.TestCase):
    def _bundle(self, tmp, fit=("a",), calib=("c",), weights=None, salt="s", frac=0.12):
        from experiments.psem_state_corrected_adaptation_gate import receipts as receipts_mod

        weights = weights or {"replacement_positive_weight": 2.0, "anchor_positive_weight": 1.5}
        bundle = Path(tmp) / "bundle"
        bundle.mkdir(parents=True)
        entries = {}
        for sid in sorted(set(fit) | set(calib)):
            target = {
                "source_id": sid, "num_frames": 400, "audio_ref": f"{sid}.wav", "waveform_sha256": _h("wav"),
                "y_anchor": [0.0] * 400, "y_replace": [1.0 if i == 0 else 0.0 for i in range(400)],
                "valid": [True] * 400, "multiplicity": [1] * 400, "episode_ids": [None] * 400,
                "corpus": "AMI",
            }

            target_path = bundle / "targets" / f"{sid}.json"
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(json.dumps(target, sort_keys=True), encoding="utf-8")
            entries[sid] = {
                "file": f"targets/{sid}.json",
                "sha256": arm_runtime.sha256_file(target_path),
                "num_frames": 400, "audio_ref": f"{sid}.wav", "waveform_sha256": _h("wav"),
            }
        body = {
            "fit": sorted(fit), "calib": sorted(calib),
            "class_weights": dict(weights), "sampling_sha256": _h("sampling"),
            "salt": salt, "target_frac": frac,
            "targets": entries,
            "files": {},
        }
        manifest_path = receipts_mod.write_json(bundle / "stage_a_manifest.json", body)
        ckpt = Path(tmp) / "ckpt.pt"
        ckpt.write_bytes(b"checkpoint-bytes")
        return bundle, manifest_path, entries, ckpt
    def _expected(self, manifest_path, ckpt, fit=("a",), calib=("c",), salt="s", frac=0.12, weights=None):
        import hashlib as _hl

        weights = weights or {"replacement_positive_weight": 2.0, "anchor_positive_weight": 1.5}
        _, weights_hash = arm_runtime.bind_class_weights(dict(weights))
        partition = _hl.sha256(
            json.dumps(
                {"fit": sorted(fit), "calib": sorted(calib), "salt": salt, "target_frac": float(frac)},
                sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return {
            "input_hash": _hl.sha256(manifest_path.read_bytes()).hexdigest(),
            "checkpoint_hash": _hl.sha256(ckpt.read_bytes()).hexdigest(),
            "partition_hash": partition,
            "weights_hash": weights_hash,
        }

    def _config(self, tmp, hashes):
        return ArmRunConfig(
            arm=ARM_R_H_SC, seed=7301, root=Path(tmp) / "arms",
            input_hash=hashes["input_hash"], checkpoint_hash=hashes["checkpoint_hash"],
            partition_hash=hashes["partition_hash"], weights_hash=hashes["weights_hash"],
            code_hash=h_arm.h_code_digest(),
        )

    def _inputs(self, tmp, bundle, ckpt):
        return h_arm.HArmPodInputs(
            bundle_dir=bundle, checkpoint=ckpt, nemo_checkout=Path(tmp) / "n",
            dependency_lock=Path(tmp) / "l", corpus_root=Path(tmp) / "r",
            reference_root=Path(tmp) / "f", sampling_manifest=Path(tmp) / "s",
        )

    def _p0(self, tmp, config):
        store = Path(tmp) / "store"
        store.mkdir(parents=True, exist_ok=True)
        (store / "p0_pass.json").write_text(json.dumps({
            "verdict": "PASS", "input_hash": config.input_hash,
            "checkpoint_hash": config.checkpoint_hash, "partition_hash": config.partition_hash,
        }), encoding="utf-8")
        return store

    def test_binding_verifies_before_model_load(self):
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp:
            bundle, manifest_path, _, ckpt = self._bundle(tmp)
            hashes = self._expected(manifest_path, ckpt)
            config = self._config(tmp, hashes)
            store = self._p0(tmp, config)
            inputs = self._inputs(tmp, bundle, ckpt)
            report = h_arm.verify_config_binding(config, json.loads(manifest_path.read_text(encoding="utf-8")), inputs)
            self.assertEqual(report["partition_hash"], hashes["partition_hash"])
            with mock.patch.object(h_arm, "pod_model_context", side_effect=AssertionError("model")) as loader:
                with self.assertRaises(h_arm.HArmError):
                    h_arm.run_h_arm_pod(config, store, inputs)
                loader.assert_not_called()

    def test_each_mismatch_blocks_before_model(self):
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp:
            bundle, manifest_path, _, ckpt = self._bundle(tmp)
            good = self._expected(manifest_path, ckpt)
            inputs = self._inputs(tmp, bundle, ckpt)
            for key in ("code_hash", "partition_hash", "weights_hash", "input_hash", "checkpoint_hash"):
                bad = dict(good)
                bad[key] = _h(f"wrong-{key}") if key != "code_hash" else _h("wrong-code")
                if key == "code_hash":
                    config = ArmRunConfig(
                        arm=ARM_R_H_SC, seed=7301, root=Path(tmp) / "arms",
                        input_hash=good["input_hash"], checkpoint_hash=good["checkpoint_hash"],
                        partition_hash=good["partition_hash"], weights_hash=good["weights_hash"],
                        code_hash=_h("wrong-code"),
                    )
                else:
                    config = self._config(tmp, bad)
                store = self._p0(tmp, config)
                with mock.patch.object(h_arm, "pod_model_context", side_effect=AssertionError("model")) as loader:
                    with self.assertRaises(AuthorizationError, msg=key):
                        h_arm.run_h_arm_pod(config, store, inputs)
                    loader.assert_not_called()

    def test_auth_receipt_checked_first(self):
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp:
            bundle, manifest_path, _, ckpt = self._bundle(tmp)
            hashes = self._expected(manifest_path, ckpt)
            config = self._config(tmp, hashes)
            inputs = self._inputs(tmp, bundle, ckpt)
            store = Path(tmp) / "empty-store"
            store.mkdir(parents=True)
            with mock.patch.object(h_arm, "pod_model_context", side_effect=AssertionError("model")) as loader:
                with self.assertRaises(AuthorizationError) as ctx:
                    h_arm.run_h_arm_pod(config, store, inputs)
                self.assertIn("H gate", str(ctx.exception))
                loader.assert_not_called()
class GpuExportContractTest(unittest.TestCase):
    def test_dev_population_uses_all_scoring_sessions(self):
        from types import SimpleNamespace


        snapshots = [
            SimpleNamespace(source_id=sid, source_family="ami_mix_headset", role="dev")
            for sid in AMI_DEV
        ] + [
            SimpleNamespace(source_id=sid, source_family="alimeeting_far_ch0", role="dev")
            for sid in ALI_DEV
        ]
        runtime = {
            snap.source_id: SimpleNamespace(source_id=snap.source_id, role="PSEM-STRATEGY-DEV")
            for snap in snapshots
        }
        members = h_arm.join_dev_export_population(runtime, snapshots)
        self.assertEqual([m["source_id"] for m in members], sorted(FULL_DEV))
        self.assertEqual(sum(1 for m in members if m["family"] == "ami_mix_headset"), 7)
        self.assertEqual(sum(1 for m in members if m["family"] == "alimeeting_far_ch0"), 3)


    def test_complete_calib_dev_export_and_corruption_refusal(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            calib = tuple(f"cal-{i}" for i in range(11))
            doc = _export_gpu(config, calib=calib, dev=FULL_DEV, call_head=False)(None, None, None)
            self.assertEqual(doc["artifact_role"], "issue-121-h-gpu-export")
            self.assertEqual(doc["binding"], config.binding)
            self.assertEqual(doc["fit"], ["s-fit-1", "s-fit-2"])
            self.assertEqual(doc["salt"], "issue-121-train-calib-v1")
            self.assertEqual(doc["target_frac"], 0.12)
            self.assertEqual(len(doc["calib_sources"]), 11)
            self.assertEqual(len(doc["dev_sources"]), 10)
            export_dir = h_arm.gpu_export_dir(config.run_dir())
            for sid in calib:
                loaded = h_arm.load_aligned_export_npz(export_dir / f"calib_{sid}.npz")
                self.assertEqual(loaded["frames"], 8)
            for sid in FULL_DEV:
                loaded = h_arm.load_aligned_export_npz(export_dir / f"dev_{sid}.npz")
                self.assertEqual(len(loaded["f0_raw"]), len(loaded["mapped"]))
            target = export_dir / f"dev_{FULL_DEV[0]}.npz"
            target.write_bytes(b"not-an-npz")
            with self.assertRaises(h_arm.HArmError):
                h_arm.load_aligned_export_npz(target)
            with self.assertRaises(h_arm.HArmError):
                h_arm.write_gpu_export_manifest(
                    config.run_dir(), config, doc["calib"], doc["dev"], {}, config.run_dir() / h_arm.TRAINING_METRICS_NAME,
                    fit=doc["fit"], salt=doc["salt"], target_frac=doc["target_frac"],
                    trained_head_sha256=doc["trained_head_sha256"],
                )



    def test_partial_export_refuses_binding_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            other = _config(tmp, seed=7302)
            arm_runtime.atomic_write_json(
                h_arm.gpu_export_progress_path(config.run_dir()),
                {
                    "identity": h_arm.gpu_export_identity(config.binding, "aa" * 32, "bb" * 32),
                    "completed": {},
                },
            )
            with self.assertRaises(h_arm.HArmError):
                h_arm.load_export_progress(config.run_dir(), other.binding, "aa" * 32, "bb" * 32)

    def test_completed_export_rejects_wrong_head_and_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            doc = _export_gpu(config, calib=("c0",), dev=("ami-dev-0",), call_head=False)(None, None, None)
            expected = h_arm.gpu_export_identity(
                config.binding, doc["trained_head_sha256"], doc["training_metrics"]["sha256"]
            )
            h_arm.require_gpu_export_identity(doc, expected, "completed GPU export")
            with self.assertRaises(h_arm.HArmError):
                h_arm.require_gpu_export_identity(
                    doc,
                    h_arm.gpu_export_identity(config.binding, "cc" * 32, doc["training_metrics"]["sha256"]),
                    "completed GPU export",
                )
            with self.assertRaises(h_arm.HArmError):
                h_arm.require_gpu_export_identity(
                    doc,
                    h_arm.gpu_export_identity(config.binding, doc["trained_head_sha256"], "dd" * 32),
                    "completed GPU export",
                )
            missing = dict(doc)
            missing.pop("trained_head_sha256")
            with self.assertRaises(h_arm.HArmError):
                h_arm.require_gpu_export_identity(missing, expected, "completed GPU export")
            arm_runtime.atomic_write_json(
                h_arm.gpu_export_progress_path(config.run_dir()),
                {"identity": expected, "completed": {}},
            )
            loaded = h_arm.load_export_progress(
                config.run_dir(), config.binding, doc["trained_head_sha256"], doc["training_metrics"]["sha256"]
            )
            self.assertEqual(loaded["identity"], expected)
            with self.assertRaises(h_arm.HArmError):
                h_arm.load_export_progress(
                    config.run_dir(), config.binding, "cc" * 32, doc["training_metrics"]["sha256"]
                )
            with self.assertRaises(h_arm.HArmError):
                h_arm.load_export_progress(
                    config.run_dir(), config.binding, doc["trained_head_sha256"], "dd" * 32
                )
            bare = dict(expected)
            del bare["training_metrics_sha256"]
            arm_runtime.atomic_write_json(
                h_arm.gpu_export_progress_path(config.run_dir()),
                {"identity": bare, "completed": {}},
            )
            with self.assertRaises(h_arm.HArmError):
                h_arm.load_export_progress(
                    config.run_dir(), config.binding, doc["trained_head_sha256"], doc["training_metrics"]["sha256"]
                )





    def test_cli_postprocess_skips_gpu_args(self):
        from unittest import mock

        from experiments.psem_state_corrected_adaptation_gate import run_h_arm as cli

        with mock.patch.object(h_arm, "run_postprocess_command", return_value={"ok": True}) as posted:
            code = cli.main(["--command", "postprocess", "--export-dir", "e", "--out-dir", "o", "--workers", "4"])
        self.assertEqual(code, 0)
        posted.assert_called_once()
        self.assertEqual(cli.main(["--command", "postprocess"]), 2)
        self.assertEqual(cli.main(["--command", "run"]), 2)
        self.assertEqual(cli.main(["--command", "profile"]), 2)



class SourceCorpusMetadataTest(unittest.TestCase):
    def test_empty_corpus_string_is_unknown(self):
        with self.assertRaises(h_arm.HArmError) as ctx:
            h_arm.corpus_family("")
        self.assertIn("export corpus is unknown:", str(ctx.exception))

    def test_stage_a_payload_without_corpus_uses_source_rows(self):
        from unittest import mock

        source_id = "alimeeting_R0005_M0035"
        payload = {"source_id": source_id}
        with mock.patch(
            "experiments.psem_state_corrected_adaptation_gate.material.load_source_rows",
            return_value={source_id: {"corpus": "AliMeeting"}},
        ):
            self.assertEqual(h_arm.resolve_source_corpus(source_id, payload), "AliMeeting")
            filled = h_arm.with_authoritative_corpus(source_id, payload)
        self.assertEqual(filled["corpus"], "AliMeeting")
        self.assertEqual(filled["source_family"], "alimeeting_far_ch0")

    def test_payload_corpus_is_authoritative_when_present(self):
        self.assertEqual(h_arm.resolve_source_corpus("any", {"corpus": "AMI"}), "AMI")

    def test_unknown_source_without_corpus_fails_closed(self):
        from unittest import mock

        with mock.patch(
            "experiments.psem_state_corrected_adaptation_gate.material.load_source_rows",
            return_value={},
        ):
            with self.assertRaises(h_arm.HArmError) as ctx:
                h_arm.resolve_source_corpus("missing-source", {})
        self.assertIn("export corpus is unknown:", str(ctx.exception))



class SerialResidencyTest(unittest.TestCase):
    def test_one_live_source_and_cross_boundary_accumulation(self):
        import weakref

        import torch

        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            binding = h_arm.h_binding(config, _stage_manifest(fit=("g1", "g2"), calib=("c",)))
            run_dir = config.run_dir()
            payloads = {s: _target_payload(s, frames=3750) for s in ("g1", "g2")}
            targets = {s: _tensor_targets(torch, payloads[s]) for s in payloads}
            weights = {"replacement_positive_weight": 2.0, "anchor_positive_weight": 1.5}
            sched = h_arm.plan_fit_schedule(
                ["g1", "g2"],
                {s: h_arm.loss_flags_for_source(3750, targets[s]["mult_weight"]) for s in targets},
            )
            self.assertEqual(sched["total_steps"], 2)
            calls: list = []
            head = FakeHead(torch, calls)
            events: list = []
            refs: list = []

            def _loader(source_id):
                tensor = _tensor_features(torch, payloads[source_id])
                refs.append(weakref.ref(
                    tensor,
                    lambda _ref, sid=source_id: events.append(("dead", sid, len(calls))),
                ))
                return tensor
            report = h_arm.run_fit_pass(
                torch, FakeWrapper(torch), head, {}, targets, weights, run_dir, binding, sched,
                feature_loader=_loader,
            )
            self.assertEqual(len(calls), 20)
            deaths = [e for e in events if e[0] == "dead"]
            self.assertEqual([(d[1], d[2]) for d in deaths], [("g1", 10), ("g2", 20)])
            self.assertEqual(report["steps_taken"], 2)
            self.assertEqual(report["per_source"]["g1"]["steps"], 0)
            self.assertEqual(report["per_source"]["g2"]["steps"], 2)
            self.assertEqual(report["accum_count"], 0)


class ThresholdDomainTest(unittest.TestCase):
    def test_probability_domain_and_masking(self):
        calibrators = {
            "f0": {"slope": 1.5, "intercept": 0.25},
            "candidate": {"slope": 0.75, "intercept": -0.5},
        }
        conv = h_arm.dev_frontier_inputs([0.0, 2.0, -2.0], [1.0, -1.0, 0.5], [True, True, False], calibrators)
        self.assertAlmostEqual(conv["f0_prob"][0], 0.5)
        self.assertAlmostEqual(conv["candidate_prob"][0], h_arm.calibrate_mod.sigmoid(1.0))
        self.assertEqual(conv["candidate_cal_masked"][-1], float("-inf"))
        self.assertEqual(conv["candidate_raw_masked"][-1], float("-inf"))
        self.assertEqual(conv["f0_masked"][-1], float("-inf"))
        finite = [t for t in conv["thresholds_calibrated"] if t != float("-inf")]
        self.assertIn(float("-inf"), conv["thresholds_calibrated"])
        self.assertIn(float("-inf"), conv["thresholds_raw"])
        for threshold in finite:
            self.assertGreaterEqual(threshold, 0.0)
            self.assertLessEqual(threshold, 1.0)
        self.assertEqual(conv["f0_reference"], 0.5)
        raw_half = h_arm.calibrate_mod.sigmoid(0.5)
        self.assertNotAlmostEqual(raw_half, 0.5)
        with self.assertRaises(h_arm.HArmError):
            h_arm.dev_frontier_inputs([0.0], [0.0, 1.0], [True], calibrators)
        with self.assertRaises(h_arm.HArmError):
            h_arm.dev_calibrator_pair({"f0": {"slope": 0.0, "intercept": 0.0}}, "f0")


class SeedRngTest(unittest.TestCase):
    def test_seed_determinism_and_rng_resume(self):
        import random

        import torch

        report_a = h_arm.seed_all_from_config(torch, 7301)
        torch.manual_seed(11)
        head_a = FakeHead(torch, [])
        params_a = [p.detach().clone() for p in head_a.parameters()]
        report_b = h_arm.seed_all_from_config(torch, 7301)
        torch.manual_seed(11)
        head_b = FakeHead(torch, [])
        for a, b in zip(params_a, head_b.parameters()):
            self.assertTrue(bool(torch.equal(a, b)))
        self.assertEqual(report_a["seed"], 7301)
        self.assertEqual(report_b["seed"], 7301)
        h_arm.seed_all_from_config(torch, 7301)
        snap = h_arm.snapshot_rng(torch)
        first_torch = torch.randn(4)
        first_py = random.random()
        torch.randn(4)
        random.random()
        h_arm.restore_rng(torch, snap)
        self.assertTrue(bool(torch.equal(first_torch, torch.randn(4))))
        self.assertEqual(first_py, random.random())
        blob = h_arm.serialize_blob(torch, snap)
        revived = h_arm.deserialize_blob(torch, blob)
        h_arm.restore_rng(torch, revived)
        self.assertTrue(bool(torch.equal(first_torch, torch.randn(4))))
        self.assertEqual(first_py, random.random())

    def test_checkpoint_carries_all_rng_families(self):
        import random

        import torch

        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            binding = h_arm.h_binding(config, _stage_manifest(fit=("r1",), calib=("c",)))
            run_dir = config.run_dir()
            head = FakeHead(torch, [])
            opt = h_arm.build_head_optimizer(torch, head)
            sch = h_arm.build_warmup_scheduler(torch, opt, 1)
            h_arm.seed_all_from_config(torch, 7301)
            snap = h_arm.snapshot_rng(torch)
            torch.randn(8)
            random.random()
            h_arm.checkpoint_after_source(torch, run_dir, "r1", [], binding, head, opt, sch)
            torch.randn(8)
            random.random()
            fresh = FakeHead(torch, [])
            fresh_opt = h_arm.build_head_optimizer(torch, fresh)
            fresh_sch = h_arm.build_warmup_scheduler(torch, fresh_opt, 1)
            h_arm.restore_head_state(torch, run_dir, binding, fresh, fresh_opt, fresh_sch)
            h_arm.restore_rng(torch, snap)
            self.assertTrue(bool(torch.equal(snap["torch_cpu"], torch.get_rng_state())))


class EmptyMappingTest(unittest.TestCase):
    def test_unmapped_source_trains_zero_loss_without_abort(self):
        from unittest import mock

        import numpy as np
        import torch

        from experiments.psem_state_corrected_adaptation_gate import material as material_mod
        from experiments.psem_state_corrected_adaptation_gate import stages as stages_mod

        frames = 800
        audio = torch.zeros(1, frames * 1280)
        hidden = torch.randn(1, frames, 192) * 0.1
        logits = torch.randn(1, frames, 4) * 0.1
        probs = torch.softmax(logits, dim=-1)
        canned = {
            "windows": [{
                "hidden": hidden, "logits": logits, "probabilities": probs,
                "emitted_frames": frames, "steps": 3,
            }],
            "state_out": None,
            "boundary_steps": [3],
        }
        payload = {
            "source_id": "cs", "num_frames": frames, "audio_ref": "a", "waveform_sha256": "w",
            "y_anchor": [0.0] * frames, "y_replace": [1.0] * frames, "valid": [True] * frames,
            "multiplicity": [1] * frames, "episode_ids": ["ep-9"] * frames,
        }
        with tempfile.TemporaryDirectory() as tmp:
            inputs = h_arm.HArmPodInputs(
                bundle_dir=Path(tmp), checkpoint=Path(tmp), nemo_checkout=Path(tmp),
                dependency_lock=Path(tmp), corpus_root=Path(tmp), reference_root=Path(tmp),
                sampling_manifest=Path(tmp),
            )
            ctx = {"torch": torch, "wrapper": object(), "head": None, "device": "cpu"}
            with mock.patch.object(stages_mod, "load_waveform_bytes", return_value=(audio, 16000)), mock.patch.object(
                material_mod, "run_adjacent_windows", return_value=canned
            ):
                ev = h_arm.pod_source_evidence(inputs, ctx, "cs", payload)
            self.assertEqual(ev["slot_of"], {})
            self.assertEqual(len(ev["timing"]["unmapped_frames"]), frames)
            self.assertTrue(all(r.get("status") == "unmapped" for r in ev["mapping_rows"]))
            config = _config(tmp)
            binding = h_arm.h_binding(config, _stage_manifest(fit=("cs",), calib=("c",)))
            run_dir = config.run_dir()
            meta = h_arm.write_source_cache(
                run_dir, "cs", np.asarray(ev["hidden192"]), np.asarray(ev["slot_logits4"]),
                ev["slot_of"], ev["mapping_rows"], ev["timing"], binding,
            )
            self.assertEqual(meta["frames"], frames)
            target = h_arm.pod_target_tensors(
                {"torch": torch, "device": "cpu"}, run_dir, binding, "cs", payload
            )
            self.assertEqual(float(target["mult_weight"].sum()), 0.0)
            flags = h_arm.loss_flags_for_source(frames, target["mult_weight"])
            self.assertEqual(flags, [False, False, False])
            sched = h_arm.plan_fit_schedule(["cs"], {"cs": flags})
            self.assertEqual(sched["total_steps"], 0)
            calls: list = []
            head = FakeHead(torch, calls)
            feats = torch.randn(1, frames, 199) * 0.1
            report = h_arm.run_fit_pass(
                torch, FakeWrapper(torch), head, {"cs": feats},
                {"cs": {**target, "num_frames": frames}}, {"replacement_positive_weight": 2.0, "anchor_positive_weight": 1.5},
                run_dir, binding, sched,
            )
            self.assertEqual(len(calls), 3)
            self.assertEqual(report["steps_taken"], 0)
            self.assertEqual(report["completed_sources"], ["cs"])


class CodeIdentityTest(unittest.TestCase):
    def test_digest_stable_sensitive_and_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in h_arm.H_IDENTITY_FILES:
                (root / name).write_text("x", encoding="utf-8")
            first = h_arm.h_code_digest(root)
            self.assertEqual(first, h_arm.h_code_digest(root))
            self.assertEqual(len(h_arm.H_IDENTITY_FILES), 13)
            self.assertIn("cross_frontier.py", h_arm.H_IDENTITY_FILES)
            (root / "h_arm.py").write_text("y", encoding="utf-8")
            self.assertNotEqual(first, h_arm.h_code_digest(root))
            (root / "material.py").write_text("y", encoding="utf-8")
            self.assertNotEqual(first, h_arm.h_code_digest(root))

    def test_real_digest_verifies(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "bundle"
            bundle.mkdir(parents=True)
            from experiments.psem_state_corrected_adaptation_gate import receipts as receipts_mod

            body = {"fit": ["a"], "calib": ["c"], "targets": {}, "files": {},
                    "class_weights": {"replacement_positive_weight": 2.0, "anchor_positive_weight": 1.5},
                    "sampling_sha256": _h("sampling"), "salt": "s", "target_frac": 0.12}
            receipts_mod.write_json(bundle / "stage_a_manifest.json", body)
            ckpt = Path(tmp) / "ckpt.pt"
            ckpt.write_bytes(b"checkpoint-bytes")
            inputs = h_arm.HArmPodInputs(
                bundle_dir=bundle, checkpoint=ckpt, nemo_checkout=Path(tmp) / "n",
                dependency_lock=Path(tmp) / "l", corpus_root=Path(tmp) / "r",
                reference_root=Path(tmp) / "f", sampling_manifest=Path(tmp) / "s",
            )
            manifest = h_arm.pod_stage_manifest(inputs)
            expected = h_arm.h_code_digest()
            config = ArmRunConfig(
                arm=ARM_R_H_SC, seed=7301, root=Path(tmp) / "arms",
                input_hash=hashlib.sha256((bundle / "stage_a_manifest.json").read_bytes()).hexdigest(),
                checkpoint_hash=hashlib.sha256(b"checkpoint-bytes").hexdigest(),
                partition_hash=h_arm.partition_hash_for(manifest),
                weights_hash=arm_runtime.bind_class_weights(dict(manifest["class_weights"]))[1],
                code_hash=expected,
            )
            report = h_arm.verify_config_binding(config, manifest, inputs)
            self.assertEqual(report["partition_hash"], config.partition_hash)
class ModeRestoreTest(unittest.TestCase):
    def test_calib_and_dev_restore_modes_on_error(self):
        from unittest import mock

        import torch

        class FlakyHead(FakeHead):
            def __init__(self, torch, calls, fail_at):
                super().__init__(torch, calls)
                self.fail_at = fail_at
                self.seen = 0

            def eval(self):
                self.training = False
                return self

            def __call__(self, features, state=None):
                self.seen += 1
                if self.seen >= self.fail_at:
                    raise h_arm.HArmError("head boom")
                return super().__call__(features, state)

        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            manifest = {"fit": ["f"], "calib": ["cc"]}
            binding = h_arm.h_binding(config, {
                **manifest,
                "class_weights": {"replacement_positive_weight": 2.0, "anchor_positive_weight": 1.5},
                "sampling_sha256": _h("sampling"), "salt": "s", "target_frac": 0.12, "targets": {},
            })
            run_dir = config.run_dir()
            import numpy as np

            h_arm.write_source_cache(
                run_dir, "cc", np.zeros((800, 192), dtype=np.float32),
                np.zeros((800, 4), dtype=np.float32), {"ep-0": 0},
                [{"anchor_episode_id": "ep-0", "status": "mapped"}], {}, binding,
            )
            payload = _target_payload("cc", frames=800)
            inputs = h_arm.HArmPodInputs(
                bundle_dir=Path(tmp), checkpoint=Path(tmp), nemo_checkout=Path(tmp),
                dependency_lock=Path(tmp), corpus_root=Path(tmp), reference_root=Path(tmp),
                sampling_manifest=Path(tmp),
            )
            head = FlakyHead(torch, [], fail_at=2)
            head.train(True)
            ctx = {"torch": torch, "wrapper": object(), "head": head, "device": "cpu"}
            with mock.patch.object(h_arm, "pod_stage_manifest", return_value=dict(manifest)), mock.patch.object(
                h_arm, "pod_payload_for", return_value=dict(payload)
            ):
                with self.assertRaises(h_arm.HArmError):
                    h_arm.pod_calib_raw(inputs, ctx, run_dir, binding)
                self.assertTrue(bool(head.training))

    def test_dev_tables_restores_mode_on_error(self):
        from types import SimpleNamespace
        from unittest import mock

        import torch

        from experiments.psem_frozen_ceiling_gate import build_ceiling_examples as ceiling_mod
        from experiments.psem_sortformer_adaptation_depth import execution as execution_mod
        from experiments.psem_state_corrected_adaptation_gate import material as material_mod

        dev = SimpleNamespace(source_id="d", source_family="ami_mix_headset", role="PSEM-STRATEGY-DEV")

        class FlakyHead(FakeHead):
            def eval(self):
                self.training = False
                return self

        head = FlakyHead(torch, [])
        head.train(True)
        ctx = {"torch": torch, "wrapper": object(), "head": head, "device": "cpu"}
        inputs = h_arm.HArmPodInputs(
            bundle_dir="b", checkpoint="c", nemo_checkout="n", dependency_lock="l",
            corpus_root="r", reference_root="f", sampling_manifest="s",
        )
        calibrators = {"f0": {"slope": 1.0, "intercept": 0.0}, "candidate": {"slope": 1.0, "intercept": 0.0}}

        def _boom(*args, **kwargs):
            raise h_arm.HArmError("infer boom")

        with mock.patch.object(ceiling_mod, "load_sessions", return_value=[dev]), mock.patch.object(
            execution_mod, "load_scoring_sessions", return_value={"d": object()}
        ), mock.patch.object(material_mod, "infer_dev_raw_logits", side_effect=_boom):
            with self.assertRaises(h_arm.HArmError):
                h_arm.pod_dev_tables(inputs, ctx, calibrators)
            self.assertTrue(bool(head.training))


class GroupAggregationTest(unittest.TestCase):
    def _rows(self, hours, ref, miss, false_cuts=0):
        return {
            "active_speech_seconds": float(hours) * 3600.0,
            "reference_replacement_count": int(ref),
            "false_cut_count": int(false_cuts),
            "missed_replacement_count": int(miss),
            "exclusive_other_contamination_seconds": 0.0,
        }

    def test_summed_primitives_beat_averaged_rates(self):
        from experiments.psem_state_corrected_adaptation_gate import frontier as frontier_mod
        from experiments.psem_state_corrected_adaptation_gate import material as material_mod

        budget = frontier_mod.FrontierPoint(
            threshold=0.5, false_cuts_per_hour=100.0, contamination=0.0, miss_rate=0.0
        )
        first = {
            "m1": self._rows(1.0, 10, 0),
            "m2": self._rows(9.0, 90, 90),
        }
        second = {
            "m1": self._rows(1.0, 10, 6),
            "m2": self._rows(9.0, 90, 40),
        }
        mean_first = sum(r["missed_replacement_count"] / r["reference_replacement_count"] for r in first.values()) / 2
        mean_second = sum(r["missed_replacement_count"] / r["reference_replacement_count"] for r in second.values()) / 2
        self.assertLess(mean_first, mean_second)
        pooled_first = material_mod._frontier_point(h_arm.sum_session_metrics(list(first.values())), 0.9)
        pooled_second = material_mod._frontier_point(h_arm.sum_session_metrics(list(second.values())), 0.1)
        self.assertAlmostEqual(pooled_first.miss_rate, 0.9)
        self.assertAlmostEqual(pooled_second.miss_rate, 0.46)
        self.assertNotAlmostEqual(
            pooled_first.miss_rate,
            sum(r["missed_replacement_count"] / r["reference_replacement_count"] for r in first.values()) / 2,
        )
        envelopes = frontier_mod.select_envelopes(budget, [pooled_first, pooled_second])
        self.assertIsNotNone(envelopes["m_envelope"])
        self.assertAlmostEqual(envelopes["m_envelope"].threshold, 0.1)
        mean_points = [
            frontier_mod.FrontierPoint(threshold=0.9, false_cuts_per_hour=0.0, contamination=0.0, miss_rate=mean_first),
            frontier_mod.FrontierPoint(threshold=0.1, false_cuts_per_hour=0.0, contamination=0.0, miss_rate=mean_second),
        ]
        mean_envelopes = frontier_mod.select_envelopes(budget, mean_points)
        self.assertAlmostEqual(mean_envelopes["m_envelope"].threshold, 0.9)

    def test_run_frontier_uses_true_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            binding = h_arm.h_binding(config, _stage_manifest())
            run_dir = config.run_dir()
            calibrators = {
                "f0": {"slope": 1.0, "intercept": 0.0},
                "candidate": {"slope": 1.0, "intercept": 0.0},
            }
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
            group_tables = {}
            for name, f0_triple in (("AMI", (3.0, 0.1, 0.2)), ("AliMeeting", (5.0, 0.3, 0.4)), ("pooled", (4.0, 0.2, 0.25))):
                group_tables[name] = {}
                for horizon in (100, 300, 500):
                    grid = [0.9, 0.7, 0.5, 0.3]
                    group_tables[name][horizon] = {
                        "thresholds": list(grid),
                        "points": [[t, 8.0 - i, 0.3 - 0.05 * i, 0.4 - 0.05 * i] for i, t in enumerate(grid)],
                        "f0": list(f0_triple),
                        "kinds": {
                            "raw": {
                                "thresholds": list(grid),
                                "points": [[t, 7.0 - i, 0.28 - 0.05 * i, 0.38 - 0.05 * i] for i, t in enumerate(grid)],
                                "f0": list(f0_triple),
                            },
                            "calibrated": {
                                "thresholds": list(grid),
                                "points": [[t, 8.0 - i, 0.3 - 0.05 * i, 0.4 - 0.05 * i] for i, t in enumerate(grid)],
                                "f0": list(f0_triple),
                            },
                        },
                    }
            doc = h_arm.run_dev_frontier(
                run_dir, binding, dev_scores, tables,
                {"dev-ami": "AMI", "dev-ali": "AliMeeting"}, calibrators, group_tables, workers=1,
            )
            self.assertEqual(doc["group_order"], ["macro", "ami", "alimeeting", "pooled"])
            pooled = doc["horizons"]["100"]["pooled"]["calibrated"]
            self.assertEqual(pooled["budget"], 4.0)
            self.assertEqual(pooled["reference"]["contamination"], 0.2)
            macro = doc["horizons"]["100"]["macro"]["calibrated"]
            self.assertAlmostEqual(macro["budget"], 4.0)
            self.assertAlmostEqual(macro["reference"]["contamination"], 0.2)
            self.assertEqual(sorted(doc["horizons"]["100"]), ["alimeeting", "ami", "macro", "pooled"])

    def test_macro_selects_realizable_common_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            binding = h_arm.h_binding(config, _stage_manifest())
            run_dir = config.run_dir()
            calibrators = {
                "f0": {"slope": 1.0, "intercept": 0.0},
                "candidate": {"slope": 1.0, "intercept": 0.0},
            }
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
            ami_points = [[0.9, 1.0, 0.5, 0.1], [0.5, 2.0, 0.2, 0.4]]
            ali_points = [[0.9, 1.0, 0.1, 0.6], [0.5, 2.0, 0.4, 0.05]]
            pooled_points = [[0.9, 1.0, 0.3, 0.35], [0.5, 2.0, 0.3, 0.225]]

            def _kinds(points, f0):
                return {
                    "kinds": {
                        kind: {
                            "thresholds": [p[0] for p in points],
                            "points": [list(p) for p in points],
                            "f0": list(f0),
                        }
                        for kind in ("raw", "calibrated")
                    },
                }

            group_tables = {
                "AMI": {hz: _kinds(ami_points, [5.0, 0.3, 0.5]) for hz in (100, 300, 500)},
                "AliMeeting": {hz: _kinds(ali_points, [5.0, 0.5, 0.3]) for hz in (100, 300, 500)},
                "pooled": {hz: _kinds(pooled_points, [5.0, 0.4, 0.4]) for hz in (100, 300, 500)},
            }
            doc = h_arm.run_dev_frontier(
                run_dir, binding, dev_scores, tables,
                {"dev-ami": "AMI", "dev-ali": "AliMeeting"}, calibrators, group_tables, workers=1,
            )
            macro = doc["horizons"]["100"]["macro"]["calibrated"]
            c_env = macro["c_envelope"]
            self.assertIsNotNone(c_env)
            self.assertIn(c_env["threshold"], (0.9, 0.5))
            self.assertAlmostEqual(c_env["contamination"], 0.3)
            self.assertNotAlmostEqual(c_env["contamination"], 0.15)
            self.assertAlmostEqual(macro["reference"]["contamination"], 0.4)
            self.assertAlmostEqual(macro["reference"]["miss_rate"], 0.4)
            self.assertTrue(macro["useful"])
            self.assertEqual(
                doc["horizons"]["100"]["macro"]["calibrated"]["c_envelope"]["threshold"],
                c_env["threshold"],
            )
            mismatched = dict(group_tables)
            mismatched["AliMeeting"] = {
                hz: _kinds([[0.8, 1.0, 0.1, 0.6], [0.5, 2.0, 0.4, 0.05]], [5.0, 0.5, 0.3])
                for hz in (100, 300, 500)
            }
            with self.assertRaises(h_arm.HArmError):
                h_arm.run_dev_frontier(
                    run_dir, binding, dev_scores, tables,
                    {"dev-ami": "AMI", "dev-ali": "AliMeeting"}, calibrators, mismatched, workers=1,
                )


class FrozenReuseTest(unittest.TestCase):
    def test_byte_copy_reuse_skips_recompression(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            binding = h_arm.h_binding(config, _stage_manifest())
            run_dir = config.run_dir()
            payload = _target_payload("r1", frames=64)
            identity = h_arm.frozen_evidence_identity(binding, payload, "r1")
            hidden = np.zeros((64, 192), dtype=np.float32)
            logits = np.zeros((64, 4), dtype=np.float32)
            frozen_meta = h_arm.write_frozen_evidence(
                config.root, identity, hidden, logits, {"ep-0": 0},
                [{"anchor_episode_id": "ep-0", "status": "mapped"}], {},
            )
            self.assertIsNotNone(h_arm.frozen_hit_meta(config.root, identity))
            reused = h_arm.reuse_frozen_to_cache(
                run_dir, "r1", h_arm.frozen_evidence_paths(config.root, identity)[0],
                frozen_meta, binding,
            )
            self.assertEqual(reused["sha256"], frozen_meta["sha256"])
            self.assertEqual(
                h_arm.cache_npz_path(run_dir, "r1").read_bytes(),
                h_arm.frozen_evidence_paths(config.root, identity)[0].read_bytes(),
            )
            manifest = h_arm.write_cache_manifest(run_dir, {"r1": {
                "file": reused["file"], "sha256": reused["sha256"], "frames": reused["frames"],
            }}, binding)
            self.assertTrue(manifest.is_file())
            covered = h_arm.require_cache_coverage(run_dir, ["r1"], [], binding)
            self.assertIn("r1", covered["sources"])

    def test_tamper_fails_closed(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            binding = h_arm.h_binding(config, _stage_manifest())
            run_dir = config.run_dir()
            payload = _target_payload("r1", frames=64)
            identity = h_arm.frozen_evidence_identity(binding, payload, "r1")
            frozen_meta = h_arm.write_frozen_evidence(
                config.root, identity, np.zeros((64, 192), dtype=np.float32),
                np.zeros((64, 4), dtype=np.float32), {"ep-0": 0},
                [{"anchor_episode_id": "ep-0", "status": "mapped"}], {},
            )
            h_arm.reuse_frozen_to_cache(
                run_dir, "r1", h_arm.frozen_evidence_paths(config.root, identity)[0],
                frozen_meta, binding,
            )
            with open(h_arm.cache_npz_path(run_dir, "r1"), "ab") as handle:
                handle.write(b"\x00")
            with self.assertRaises(h_arm.HArmError):
                h_arm.verify_source_cache_file(run_dir, "r1", binding)
            with self.assertRaises(h_arm.HArmError):
                h_arm.read_source_cache(run_dir, "r1", binding)
            self.assertIsNone(h_arm.frozen_hit_meta(config.root, dict(identity, num_frames=65)))

    def test_binding_mismatch_refuses_coverage(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            binding = h_arm.h_binding(config, _stage_manifest())
            run_dir = config.run_dir()
            meta = h_arm.write_source_cache(
                run_dir, "r1", np.zeros((16, 192), dtype=np.float32),
                np.zeros((16, 4), dtype=np.float32), {"ep-0": 0},
                [{"anchor_episode_id": "ep-0", "status": "mapped"}], {}, binding,
            )
            h_arm.write_cache_manifest(run_dir, {"r1": {
                "file": meta["file"], "sha256": meta["sha256"], "frames": meta["frames"],
            }}, binding)
            bad = dict(binding)
            bad["checkpoint_hash"] = "0" * 64
            with self.assertRaises(h_arm.HArmError):
                h_arm.require_cache_coverage(run_dir, ["r1"], [], bad)
            with self.assertRaises(h_arm.HArmError):
                h_arm.require_cache_coverage(run_dir, ["r1", "missing"], [], binding)


class BoundedMemoryTest(unittest.TestCase):
    def _mini_deps(self, torch, manifest, payloads, calls, config):
        wrapper = FakeWrapper(torch)
        head = FakeHead(torch, [])

        def _features(cached):
            import numpy as np

            hidden = np.asarray(cached["hidden192"])
            logits = np.asarray(cached["slot_logits4"])
            feats = h_arm.assemble_features(hidden, logits, logits[:, 0].tolist(), logits.max(axis=1).tolist(), [1.04] * len(hidden))
            return torch.as_tensor(np.asarray(feats, dtype=np.float32)).unsqueeze(0)

        def _evidence(source_id, payload):
            calls.append(source_id)
            import numpy as np

            frames = int(payload["num_frames"])
            return {
                "hidden192": np.zeros((frames, 192), dtype=np.float32),
                "slot_logits4": np.zeros((frames, 4), dtype=np.float32),
                "slot_of": {"ep-0": 0},
                "mapping_rows": [{"anchor_episode_id": "ep-0", "status": "mapped"}],
                "timing": {},
            }

        return h_arm.HArmDeps(
            load_bundle_manifest=lambda: manifest,
            bundle_dir=Path("/none"),
            build_missing_targets=lambda s: dict(payloads[s]),
            build_evidence=_evidence,
            build_features=_features,
            build_targets=lambda s, p: _tensor_targets(torch, p),
            load_wrapper_head=lambda: (wrapper, head),
            load_torch=lambda: torch,
            export_gpu_evidence=_export_gpu(config, calib=("bc",)),
            workers=1,
        )


    def test_run_holds_no_global_evidence_table(self):
        import numpy as np
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            _p0(Path(tmp) / "store", config)
            manifest = _stage_manifest(fit=("b1", "b2"), calib=("bc",))
            binding = h_arm.h_binding(config, manifest)
            h_arm.run_profile(
                config.run_dir(), binding, lambda: {"io_bytes": 0},
                lambda: {"infer_seconds": 1.0, "io_bytes": 0}, 0, torch=torch, device=None, steps=8,
            )
            payloads = {s: _target_payload(s, frames=400) for s in ("b1", "b2", "bc")}
            prefrozen = _target_payload("b1", frames=400)
            identity = h_arm.frozen_evidence_identity(binding, prefrozen, "b1")
            frozen_meta = h_arm.write_frozen_evidence(
                config.root, identity, np.zeros((400, 192), dtype=np.float32),
                np.zeros((400, 4), dtype=np.float32), {"ep-0": 0},
                [{"anchor_episode_id": "ep-0", "status": "mapped"}], {},
            )
            calls: list = []
            out = h_arm.run_h_arm(config, Path(tmp) / "store", self._mini_deps(torch, manifest, payloads, calls, config))

            self.assertNotIn("b1", calls)
            self.assertEqual(sorted(calls), ["b2", "bc"])
            self.assertEqual(
                h_arm.cache_npz_path(config.run_dir(), "b1").read_bytes(),
                h_arm.frozen_evidence_paths(config.root, identity)[0].read_bytes(),
            )
            self.assertEqual(frozen_meta["sha256"], json.loads(
                h_arm.cache_meta_path(config.run_dir(), "b1").read_text(encoding="utf-8")
            )["sha256"])
            carried = h_arm._CARRIED_CACHE
            self.assertTrue(carried is None or "table" not in dict(carried))
            run_dir = Path(out["run_dir"])
            self.assertTrue((run_dir / "final_manifest.json").is_file())
            ledger = json.loads((run_dir / "checkpoints" / "checkpoint.json").read_text(encoding="utf-8"))
            self.assertEqual(ledger["completed_sources"], ["b1", "b2"])


class ProfileSectionsTest(unittest.TestCase):
    def test_sections_match_legacy_keys(self):
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            binding = h_arm.h_binding(config, _stage_manifest())
            total = 40
            receipt = h_arm.run_profile(
                config.run_dir(), binding, lambda: {"io_bytes": 4},
                lambda: {"infer_seconds": 2.0, "io_bytes": 8,
                         "frontier_slice": {"horizon_ms": 100, "sampled_thresholds": 3,
                                            "total_thresholds": 5, "points": [],
                                            "seconds": 0.5, "projected_seconds": 1.0}},
                total, torch=torch, device=None, steps=8, hourly_cost_usd=2.0,
            )
            self.assertEqual(receipt["train"]["seconds_per_step"], receipt["seconds_per_step"])
            self.assertEqual(receipt["train"]["projected_train_seconds"], receipt["projected_train_seconds"])
            self.assertEqual(receipt["train"]["projected_cost_usd"], receipt["projected_cost_usd"])
            self.assertEqual(receipt["train"]["total_steps"], total)
            self.assertEqual(receipt["dev_inference"]["seconds"], 2.0)
            self.assertEqual(receipt["dev_inference"]["seconds"], receipt["dev_infer_seconds"])
            self.assertEqual(receipt["dev_inference"]["io_bytes"], 8)
            self.assertEqual(receipt["measured"]["representative_dev_infer_seconds"], 2.0)
            self.assertEqual(receipt["estimated"]["cpu_tail"]["kind"], "estimated")
            self.assertEqual(receipt["estimated"]["cpu_tail"]["projected_seconds"], 1.0)
            self.assertEqual(receipt["dev_inference"]["kind"], "measured")
            self.assertEqual(receipt["assembly_projection"]["kind"], "estimated")
            self.assertEqual(receipt["assembly_projection"]["io_bytes"], receipt["io_bytes"])
            self.assertEqual(receipt["assembly_projection"]["train_io_bytes"], 8 * 4)

            required = h_arm.require_profile(config.run_dir(), binding)
            self.assertEqual(required["train"]["total_steps"], total)


class LossCacheTest(unittest.TestCase):
    def test_cached_weights_match_fresh(self):
        import torch

        torch.manual_seed(0)
        product = torch.randn(1, 32, requires_grad=True)
        anchor = torch.randn(1, 32, requires_grad=True)
        y_replace = (torch.rand(1, 32) > 0.5).float()
        y_anchor = (torch.rand(1, 32) > 0.5).float()
        mult = torch.ones(1, 32)
        weights = {"replacement_positive_weight": 2.0, "anchor_positive_weight": 1.5}
        fresh = h_arm.chunk_loss_value(torch, product, anchor, y_replace, y_anchor, mult, weights)
        cached = h_arm.build_loss_weight_tensors(torch, weights, product.dtype, product.device)
        reused = h_arm.chunk_loss_value(torch, product, anchor, y_replace, y_anchor, mult, weights, cached_weights=cached)
        self.assertAlmostEqual(float(fresh.detach()), float(reused.detach()), places=6)
        with self.assertRaises(KeyError):
            h_arm.chunk_loss_value(torch, product, anchor, y_replace, y_anchor, mult, {})
        with self.assertRaises(KeyError):
            h_arm.build_loss_weight_tensors(torch, {}, product.dtype, product.device)

    def test_flag_paths_agree(self):
        import torch

        mult = [1.0, 0.0, 2.0] * 300
        tensor = torch.as_tensor([mult], dtype=torch.float32)
        self.assertEqual(
            h_arm.loss_flags_from_mult_list(900, mult),
            h_arm.loss_flags_for_source(900, tensor),
        )
