from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from experiments.psem_state_corrected_adaptation_gate import arm_runtime, h_arm
from experiments.psem_state_corrected_adaptation_gate.arm_runtime import (
    ARM_R_H_SC,
    ArmRunConfig,
    CheckpointError,
)


def _h(tag: str) -> str:
    return hashlib.sha256(tag.encode("utf-8")).hexdigest()


def _config(tmp):
    return ArmRunConfig(
        arm=ARM_R_H_SC,
        seed=7301,
        root=Path(tmp) / "arms",
        input_hash=_h("input"),
        checkpoint_hash=_h("checkpoint"),
        partition_hash=_h("partition"),
        weights_hash=_h("weights"),
        code_hash=_h("code"),
    )


def _manifest():
    return {
        "fit": ["rs-1", "rs-2"],
        "calib": ["rs-c"],
        "class_weights": {"replacement_positive_weight": 2.0, "anchor_positive_weight": 1.5},
        "sampling_sha256": _h("sampling"),
        "salt": "issue-121-train-calib-v1",
        "target_frac": 0.12,
        "targets": {},
    }


class FakeWrapper:
    def __init__(self, torch):
        self._linear = torch.nn.Linear(4, 4)
        for p in self._linear.parameters():
            p.requires_grad_(False)
        self.training = False

    def eval(self):
        self.training = False
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
        self.calls.append({"none": state is None})
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


def _frames(torch, payload, seed):
    torch.manual_seed(seed)
    feats = torch.randn(1, payload["num_frames"], 199) * 0.1
    target = {
        "num_frames": payload["num_frames"],
        "y_replace": (torch.rand(1, payload["num_frames"]) > 0.85).float(),
        "y_anchor": (torch.rand(1, payload["num_frames"]) > 0.5).float(),
        "mult_weight": torch.ones(1, payload["num_frames"]),
        "f0": torch.zeros(1, payload["num_frames"]),
    }
    target["y_replace"][0, 0] = 1.0
    return feats, target


def _payload(source_id, frames=800):
    return {
        "source_id": source_id,
        "num_frames": frames,
        "y_anchor": [0.0] * frames,
        "y_replace": [0.0] * frames,
        "valid": [True] * frames,
        "multiplicity": [1] * frames,
        "episode_ids": [None] * frames,
    }


def _assert_nested_equal(case, first, second, path=""):
    if isinstance(first, dict):
        case.assertIsInstance(second, dict, path)
        case.assertEqual(sorted(first), sorted(second), path)
        for key in first:
            _assert_nested_equal(case, first[key], second[key], f"{path}.{key}")
        return
    if isinstance(first, (list, tuple)):
        case.assertEqual(len(first), len(second), path)
        for i, (a, b) in enumerate(zip(first, second)):
            _assert_nested_equal(case, a, b, f"{path}[{i}]")
        return
    try:
        import torch

        if isinstance(first, torch.Tensor):
            case.assertTrue(bool(torch.equal(first, second)), path)
            return
    except ImportError:
        pass
    case.assertEqual(first, second, path)


class SourceBoundaryResumeTest(unittest.TestCase):
    def test_mid_group_resume_equals_uninterrupted(self):
        import torch

        weights = {"replacement_positive_weight": 2.0, "anchor_positive_weight": 1.5}
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            binding = h_arm.h_binding(config, _manifest())
            run_dir = config.run_dir()
            torch.manual_seed(7)
            feats_a, tgt_a = _frames(torch, _payload("rs-1"), seed=1)
            feats_b, tgt_b = _frames(torch, _payload("rs-2"), seed=2)
            flags = {
                "rs-1": h_arm.loss_flags_for_source(800, tgt_a["mult_weight"]),
                "rs-2": h_arm.loss_flags_for_source(800, tgt_b["mult_weight"]),
            }
            schedule = h_arm.plan_fit_schedule(["rs-1", "rs-2"], flags)
            self.assertEqual(schedule["loss_chunks"], 6)
            self.assertEqual(schedule["total_steps"], 1)
            self.assertEqual(schedule["warmup_steps"], arm_runtime.compute_warmup_steps(1))
            torch.manual_seed(11)
            head = FakeHead(torch, [])
            wrapper = FakeWrapper(torch)
            opt = h_arm.build_head_optimizer(torch, head)
            sch = h_arm.build_warmup_scheduler(torch, opt, int(schedule["warmup_steps"]))
            leg1 = h_arm.run_fit_pass(
                torch, wrapper, head, {"rs-1": feats_a}, {"rs-1": tgt_a},
                weights, run_dir, binding, schedule, optimizer=opt, scheduler=sch,
                limit_sources=["rs-1"],
            )
            self.assertEqual(leg1["completed_sources"], ["rs-1"])
            self.assertEqual(leg1["per_source"]["rs-1"]["steps"], 0)
            self.assertEqual(leg1["accum_count"], 3)
            self.assertEqual(leg1["steps_taken"], 0)
            torch.manual_seed(11)
            fresh = FakeHead(torch, [])
            fresh_opt = h_arm.build_head_optimizer(torch, fresh)
            fresh_sch = h_arm.build_warmup_scheduler(torch, fresh_opt, int(schedule["warmup_steps"]))
            restored = h_arm.restore_head_state(torch, run_dir, binding, fresh, fresh_opt, fresh_sch)
            self.assertEqual(restored["completed_sources"], ["rs-1"])
            self.assertEqual(restored["pending"], {"accum_count": 3, "steps_taken": 0})
            for (n1, p1), (n2, p2) in zip(head.named_parameters(), fresh.named_parameters()):
                self.assertEqual(n1, n2)
                self.assertTrue(bool(torch.equal(p1.grad, p2.grad)), n1)
            torch.manual_seed(7)
            feats_b2, tgt_b2 = _frames(torch, _payload("rs-2"), seed=2)
            leg2 = h_arm.run_fit_pass(
                torch, wrapper, fresh, {"rs-2": feats_b2}, {"rs-2": tgt_b2},
                weights, run_dir, binding, schedule, completed=["rs-1"],
                optimizer=fresh_opt, scheduler=fresh_sch, pending=restored["pending"],
            )
            self.assertEqual(leg2["completed_sources"], ["rs-1", "rs-2"])
            self.assertEqual(leg2["steps_taken"], 1)
            self.assertEqual(leg2["accum_count"], 0)
            self.assertEqual(leg2["schedule"]["total_steps"], schedule["total_steps"])
            self.assertEqual(leg2["schedule"]["warmup_steps"], schedule["warmup_steps"])
            torch.manual_seed(7)
            feats_au, tgt_au = _frames(torch, _payload("rs-1"), seed=1)
            feats_bu, tgt_bu = _frames(torch, _payload("rs-2"), seed=2)
            torch.manual_seed(11)
            head_u = FakeHead(torch, [])
            opt_u = h_arm.build_head_optimizer(torch, head_u)
            sch_u = h_arm.build_warmup_scheduler(torch, opt_u, int(schedule["warmup_steps"]))
            h_arm.run_fit_pass(
                torch, FakeWrapper(torch), head_u,
                {"rs-1": feats_au, "rs-2": feats_bu}, {"rs-1": tgt_au, "rs-2": tgt_bu},
                weights, Path(tmp) / "other", binding, schedule, optimizer=opt_u, scheduler=sch_u,
            )
            for (n1, p1), (n2, p2) in zip(fresh.named_parameters(), head_u.named_parameters()):
                self.assertEqual(n1, n2)
                self.assertTrue(bool(torch.equal(p1, p2)), n1)
                self.assertIsNone(p1.grad, n1)
                self.assertIsNone(p2.grad, n1)
            _assert_nested_equal(self, fresh_opt.state_dict(), opt_u.state_dict(), "optimizer")
            self.assertEqual(fresh_sch.state_dict(), sch_u.state_dict())

    def test_binding_mismatch_blocks_resume(self):
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            binding = h_arm.h_binding(config, _manifest())
            run_dir = config.run_dir()
            head = FakeHead(torch, [])
            optimizer = h_arm.build_head_optimizer(torch, head)
            scheduler = h_arm.build_warmup_scheduler(torch, optimizer, 1)
            feats, tgt = _frames(torch, _payload("rs-1", frames=400), seed=3)
            weights = {"replacement_positive_weight": 2.0, "anchor_positive_weight": 1.5}
            schedule = h_arm.plan_fit_schedule(
                ["rs-1"], {"rs-1": h_arm.loss_flags_for_source(400, tgt["mult_weight"])}
            )
            h_arm.run_fit_pass(
                torch, FakeWrapper(torch), head, {"rs-1": feats}, {"rs-1": tgt},
                weights, run_dir, binding, schedule, optimizer=optimizer, scheduler=scheduler,
            )
            bad = dict(binding)
            bad["seed"] = 7302
            with self.assertRaises(CheckpointError):
                arm_runtime.load_source_checkpoint(run_dir, bad)
            with self.assertRaises(CheckpointError):
                h_arm.restore_head_state(torch, run_dir, bad, head, optimizer, scheduler)

    def test_checkpoint_requires_four_blobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            binding = h_arm.h_binding(config, _manifest())
            with self.assertRaises(CheckpointError):
                arm_runtime.save_source_checkpoint(
                    config.run_dir(), "rs-1", ["rs-1"], binding, blobs={"model": b"x"}
                )
            with self.assertRaises(CheckpointError):
                arm_runtime.save_source_checkpoint(
                    config.run_dir(), "rs-1", ["rs-1"], binding,
                    blobs={r: b"" for r in arm_runtime.CHECKPOINT_ROLES},
                )

    def test_resume_plan_is_source_granular(self):
        with self.assertRaises(CheckpointError):
            h_arm.remaining_fit_sources(["a", "b", "c"], ["b"])
        self.assertEqual(h_arm.remaining_fit_sources(["a", "b"], ["a", "b"]), [])




    def test_chronological_order_beats_lexicographic(self):
        import torch

        weights = {"replacement_positive_weight": 2.0, "anchor_positive_weight": 1.5}
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            binding = h_arm.h_binding(config, _manifest())
            run_dir = config.run_dir()
            feats_a, tgt_a = _frames(torch, _payload("s-a"), seed=1)
            feats_b, tgt_b = _frames(torch, _payload("s-b"), seed=2)
            chunks = []
            position = 0
            for source_id in ("s-b", "s-a"):
                for chunk_index in range(3):
                    chunks.append({
                        "source": source_id, "chunk_index": chunk_index, "contributes": True,
                        "accum_position": position, "optimizer_step": 0,
                        "is_step_boundary": position == 5,
                    })
                    position += 1
            schedule = {
                "sources": ["s-b", "s-a"], "chunks": chunks, "loss_chunks": 6,
                "total_steps": 1, "warmup_steps": 1, "accumulation": 16,
            }
            torch.manual_seed(11)
            head = FakeHead(torch, [])
            opt = h_arm.build_head_optimizer(torch, head)
            sch = h_arm.build_warmup_scheduler(torch, opt, int(schedule["warmup_steps"]))
            leg1 = h_arm.run_fit_pass(
                torch, FakeWrapper(torch), head, {"s-b": feats_b}, {"s-b": tgt_b},
                weights, run_dir, binding, schedule, optimizer=opt, scheduler=sch,
                limit_sources=["s-b"],
            )
            self.assertEqual(leg1["completed_sources"], ["s-b"])
            self.assertEqual(leg1["steps_taken"], 0)
            self.assertEqual(leg1["accum_count"], 3)
            torch.manual_seed(11)
            fresh = FakeHead(torch, [])
            fresh_opt = h_arm.build_head_optimizer(torch, fresh)
            fresh_sch = h_arm.build_warmup_scheduler(torch, fresh_opt, int(schedule["warmup_steps"]))
            leg2 = h_arm.run_fit_pass(
                torch, FakeWrapper(torch), fresh, {"s-a": feats_a}, {"s-a": tgt_a},
                weights, run_dir, binding, schedule, completed=["s-b"],
                optimizer=fresh_opt, scheduler=fresh_sch,
                pending=h_arm.restore_head_state(
                    torch, run_dir, binding, fresh, fresh_opt, fresh_sch,
                    source_order=["s-b", "s-a"],
                )["pending"],
            )
            self.assertEqual(leg2["completed_sources"], ["s-b", "s-a"])
            self.assertEqual(leg2["steps_taken"], 1)
            record = arm_runtime.load_source_checkpoint(run_dir, binding)
            self.assertEqual(record["completed_sources"], ["s-b", "s-a"])
            final_params = {k: v.clone() for k, v in fresh.state_dict().items()}
            torch.manual_seed(11)
            check = FakeHead(torch, [])
            check_opt = h_arm.build_head_optimizer(torch, check)
            check_sch = h_arm.build_warmup_scheduler(torch, check_opt, int(schedule["warmup_steps"]))
            h_arm.restore_head_state(
                torch, run_dir, binding, check, check_opt, check_sch,
                source_order=["s-b", "s-a"],
            )
            for key, value in final_params.items():
                self.assertTrue(bool(torch.equal(value, check.state_dict()[key])), key)
            torch.manual_seed(11)
            other = FakeHead(torch, [])
            other_opt = h_arm.build_head_optimizer(torch, other)
            other_sch = h_arm.build_warmup_scheduler(torch, other_opt, int(schedule["warmup_steps"]))
            h_arm.restore_head_state(
                torch, run_dir, binding, other, other_opt, other_sch,
                source_order=["s-b"],
            )
            self.assertTrue(
                any(not bool(torch.equal(a, b)) for a, b in zip(final_params.values(), other.state_dict().values()))
            )
if __name__ == "__main__":
    unittest.main()
