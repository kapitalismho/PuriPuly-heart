from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from experiments.psem_state_corrected_adaptation_gate import arm_runtime
from experiments.psem_state_corrected_adaptation_gate.arm_runtime import (
    ACCUMULATION,
    ARM_R_H_SC,
    ARM_R_T2_SC,
    ARM_R_TA_SC,
    CHECKPOINT_ROLES,
    ArmRunConfig,
    AuthorizationError,
    CheckpointError,
)


def _h(tag: str) -> str:
    return hashlib.sha256(tag.encode("utf-8")).hexdigest()


def _double(payload):
    return {"key": payload["key"], "value": payload["value"] * 2}


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


def _blobs(tag: str) -> dict[str, bytes]:
    return {role: f"{tag}-{role}-state".encode("utf-8") for role in CHECKPOINT_ROLES}


class ScheduleTest(unittest.TestCase):
    def test_deterministic_source_order(self):
        plan = arm_runtime.plan_schedule(
            ["b", "a", "c"], {"b": [True], "a": [True], "c": [True]}
        )
        self.assertEqual(plan["sources"], ["a", "b", "c"])
        self.assertEqual([c["source"] for c in plan["chunks"]], ["a", "b", "c"])

    def test_step_math_zero_loss_and_partial(self):
        plan = arm_runtime.plan_schedule(
            ["s"],
            {"s": [True, False, True, True] + [False] * 3 + [True] * 30},
        )
        loss = 3 + 30
        total = -(-loss // ACCUMULATION)
        self.assertEqual(plan["loss_chunks"], loss)
        self.assertEqual(plan["total_steps"], total)
        import math

        self.assertEqual(plan["warmup_steps"], max(1, math.ceil(0.05 * total)))
        zero = [c for c in plan["chunks"] if not c["contributes"]]
        self.assertTrue(zero)
        self.assertTrue(all(c["optimizer_step"] is None for c in zero))
        steps = sorted({c["optimizer_step"] for c in plan["chunks"] if c["contributes"]})
        self.assertEqual(steps, list(range(total)))
        boundary = [c for c in plan["chunks"] if c.get("is_step_boundary")]
        self.assertEqual(len(boundary), total)

    def test_partial_final_accumulation_single_step(self):
        plan = arm_runtime.plan_schedule(["s"], {"s": [True] * 20})
        self.assertEqual(plan["total_steps"], 2)
        last = [c for c in plan["chunks"] if c["contributes"]][-1]
        self.assertTrue(last["is_step_boundary"])
        self.assertEqual(last["optimizer_step"], 1)

    def test_empty_loss_plan(self):
        plan = arm_runtime.plan_schedule(["s"], {"s": [False, False]})
        self.assertEqual(plan["total_steps"], 0)
        self.assertEqual(plan["warmup_steps"], 0)


class WorkerTest(unittest.TestCase):
    def test_cap_and_parity(self):
        self.assertLessEqual(arm_runtime.resolve_workers(None), 24)
        self.assertLessEqual(arm_runtime.resolve_workers(64), 24)
        self.assertEqual(arm_runtime.resolve_workers(2), min(2, os.cpu_count() or 1))
        self.assertEqual(
            arm_runtime.resolve_workers(None), arm_runtime.default_worker_limit()
        )
        self.assertLessEqual(
            arm_runtime.resolve_workers(None), int(os.cpu_count() or 1)
        )
        payloads = [{"key": i, "value": i} for i in range(12)]
        seq = arm_runtime.ordered_process_map(_double, payloads, 1)
        par = arm_runtime.ordered_process_map(_double, payloads, 2)
        self.assertEqual(seq, par)
        self.assertEqual([r["value"] for r in par], [i * 2 for i in range(12)])

    def test_default_physical_explicit_honored_and_threads_stay_one(self):
        logical = int(os.cpu_count() or 1)
        physical = arm_runtime._physical_cpu_count() or logical
        self.assertEqual(
            arm_runtime.default_worker_limit(), max(1, min(24, int(physical)))
        )
        self.assertEqual(
            arm_runtime.resolve_workers(None), arm_runtime.default_worker_limit()
        )
        self.assertEqual(arm_runtime.resolve_workers(1), 1)
        self.assertEqual(
            arm_runtime.resolve_workers(10**6), max(1, min(24, logical))
        )
        receipt = arm_runtime.thread_cap_receipt()
        self.assertEqual(receipt["omp_num_threads"], "1")

    def test_prefetch_bounded_independent_of_workers(self):
        self.assertEqual(arm_runtime.resolve_prefetch_depth(24, 100), 4)
        self.assertEqual(arm_runtime.resolve_prefetch_depth(10**6, 10**6), 4)
        self.assertEqual(arm_runtime.resolve_prefetch_depth(2, 100), 2)
        self.assertEqual(arm_runtime.resolve_prefetch_depth(24, 3), 3)
        self.assertEqual(arm_runtime.resolve_prefetch_depth(1, 1), 1)
        self.assertEqual(arm_runtime.resolve_prefetch_depth(4, 0), 0)


class CheckpointBlobTest(unittest.TestCase):
    def test_four_blob_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            run_dir = config.run_dir()
            binding = config.binding
            blobs_a = _blobs("a")
            arm_runtime.save_source_checkpoint(run_dir, "a", [], binding, blobs_a)
            loaded = arm_runtime.load_source_checkpoint(run_dir, binding)
            self.assertEqual(loaded["completed_sources"], ["a"])
            self.assertFalse(loaded["fresh"])
            self.assertEqual(set(loaded["blobs"]["a"]), set(CHECKPOINT_ROLES))
            for role, path in loaded["blobs"]["a"].items():
                self.assertTrue(str(path).startswith(str(run_dir)))
                self.assertEqual(path.read_bytes(), blobs_a[role])
            blobs_b = _blobs("b")
            arm_runtime.save_source_checkpoint(
                run_dir, "b", loaded["completed_sources"], binding, blobs_b
            )
            loaded2 = arm_runtime.load_source_checkpoint(run_dir, binding)
            self.assertEqual(loaded2["completed_sources"], ["a", "b"])
            self.assertEqual(loaded2["blobs"]["b"]["rng"].read_bytes(), blobs_b["rng"])
            self.assertEqual(
                arm_runtime.resume_plan(["c", "a", "b"], loaded2["completed_sources"]), ["c"]
            )

    def test_completion_order_preserved_not_sorted(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            run_dir = config.run_dir()
            binding = config.binding
            blobs_s2 = _blobs("s2")
            blobs_s10 = _blobs("s10")
            arm_runtime.save_source_checkpoint(run_dir, "s2", [], binding, blobs_s2)
            arm_runtime.save_source_checkpoint(run_dir, "s10", ["s2"], binding, blobs_s10)
            loaded = arm_runtime.load_source_checkpoint(run_dir, binding)
            self.assertEqual(loaded["completed_sources"], ["s2", "s10"])
            latest = loaded["completed_sources"][-1]
            self.assertEqual(latest, "s10")
            for role, path in loaded["blobs"][latest].items():
                self.assertEqual(path.read_bytes(), blobs_s10[role])

    def test_every_committed_source_stays_restorable_out_of_lexical_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            run_dir = config.run_dir()
            binding = config.binding
            blobs_b = _blobs("s-b")
            blobs_a = _blobs("s-a")
            arm_runtime.save_source_checkpoint(run_dir, "s-b", [], binding, blobs_b)
            arm_runtime.save_source_checkpoint(
                run_dir, "s-a", ["s-b"], binding, blobs_a
            )
            loaded = arm_runtime.load_source_checkpoint(run_dir, binding)
            self.assertEqual(loaded["completed_sources"], ["s-b", "s-a"])
            for source_id, expected in (("s-b", blobs_b), ("s-a", blobs_a)):
                self.assertIn(source_id, loaded["blobs"])
                for role, path in loaded["blobs"][source_id].items():
                    self.assertEqual(path.read_bytes(), expected[role])
            order = ["s-b", "s-a"]
            in_order = [s for s in order if s in set(loaded["completed_sources"])]
            latest = in_order[-1] if in_order else loaded["completed_sources"][-1]
            self.assertEqual(latest, "s-a")
            in_order = [s for s in ["s-b"] if s in set(loaded["completed_sources"])]
            earliest = in_order[-1] if in_order else loaded["completed_sources"][-1]
            self.assertEqual(earliest, "s-b")
            for role, path in loaded["blobs"][earliest].items():
                self.assertEqual(path.read_bytes(), blobs_b[role])

    def test_save_rejects_duplicate_and_stale_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            run_dir = config.run_dir()
            binding = config.binding
            arm_runtime.save_source_checkpoint(run_dir, "a", [], binding, _blobs("a"))
            with self.assertRaises(CheckpointError):
                arm_runtime.save_source_checkpoint(run_dir, "a", ["a"], binding, _blobs("a"))
            with self.assertRaises(CheckpointError):
                arm_runtime.save_source_checkpoint(run_dir, "b", [], binding, _blobs("b"))
            with self.assertRaises(CheckpointError):
                arm_runtime.save_source_checkpoint(
                    run_dir, "b", ["a", "a"], binding, _blobs("b")
                )

    def test_resume_requires_exact_chronological_prefix(self):
        self.assertEqual(arm_runtime.resume_plan(["a", "b", "c"], []), ["a", "b", "c"])
        self.assertEqual(arm_runtime.resume_plan(["a", "b", "c"], ["a", "b"]), ["c"])
        self.assertEqual(arm_runtime.resume_plan(["a", "b"], ["a", "b"]), [])
        with self.assertRaises(CheckpointError):
            arm_runtime.resume_plan(["a", "b", "c"], ["a", "c"])
        with self.assertRaises(CheckpointError):
            arm_runtime.resume_plan(["a", "b"], ["b", "a"])
        with self.assertRaises(CheckpointError):
            arm_runtime.resume_plan(["a", "b", "c"], ["b"])
        with self.assertRaises(CheckpointError):
            arm_runtime.resume_plan(["a", "b"], ["a", "a"])
        with self.assertRaises(CheckpointError):
            arm_runtime.resume_plan(["a", "b"], ["a", "zzz"])


    def test_manifest_records_path_hash_size_and_is_last(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            run_dir = config.run_dir()
            blobs = _blobs("a")
            out = arm_runtime.save_source_checkpoint(run_dir, "a", [], config.binding, blobs)
            manifest = json.loads(out.read_text(encoding="utf-8"))
            entry = manifest["sources"]["a"]
            self.assertEqual(set(entry), set(CHECKPOINT_ROLES))
            for role in CHECKPOINT_ROLES:
                record = entry[role]
                self.assertEqual(record["path"], f"a.{role}.pt")
                self.assertEqual(record["sha256"], hashlib.sha256(blobs[role]).hexdigest())
                self.assertEqual(record["size"], len(blobs[role]))
            manifest_mtime = out.stat().st_mtime_ns
            for role in CHECKPOINT_ROLES:
                blob_mtime = (run_dir / "checkpoints" / f"a.{role}.pt").stat().st_mtime_ns
                self.assertGreaterEqual(manifest_mtime, blob_mtime)

    def test_blob_roles_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            run_dir = config.run_dir()
            with self.assertRaises(CheckpointError):
                arm_runtime.save_source_checkpoint(run_dir, "a", [], config.binding, None)
            partial = {"model": b"m"}
            with self.assertRaises(CheckpointError):
                arm_runtime.save_source_checkpoint(run_dir, "a", [], config.binding, partial)
            empty = {role: b"" for role in CHECKPOINT_ROLES}
            with self.assertRaises(CheckpointError):
                arm_runtime.save_source_checkpoint(run_dir, "a", [], config.binding, empty)

    def test_tamper_missing_and_orphan_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            run_dir = config.run_dir()
            arm_runtime.save_source_checkpoint(run_dir, "a", [], config.binding, _blobs("a"))
            victim = run_dir / "checkpoints" / "a.model.pt"
            victim.write_bytes(b"tampered-state")
            with self.assertRaises(CheckpointError):
                arm_runtime.load_source_checkpoint(run_dir, config.binding)
            victim.unlink()
            with self.assertRaises(CheckpointError):
                arm_runtime.load_source_checkpoint(run_dir, config.binding)

    def test_blobs_without_manifest_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            run_dir = config.run_dir()
            orphan_dir = run_dir / "checkpoints"
            orphan_dir.mkdir(parents=True, exist_ok=True)
            for role in CHECKPOINT_ROLES:
                (orphan_dir / f"a.{role}.pt").write_bytes(b"orphan")
            loaded = arm_runtime.load_source_checkpoint(run_dir, config.binding)
            self.assertTrue(loaded["fresh"])
            self.assertEqual(loaded["completed_sources"], [])
            self.assertEqual(loaded["blobs"], {})

    def test_orphan_manifest_reference_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            run_dir = config.run_dir()
            binding = config.binding
            arm_runtime.save_source_checkpoint(run_dir, "a", [], binding, _blobs("a"))
            manifest_path = run_dir / "checkpoints" / "checkpoint.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["sources"]["a"]["model"]["path"] = "ghost.model.pt"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(CheckpointError):
                arm_runtime.load_source_checkpoint(run_dir, binding)

    def test_seed_and_contract_binding_distinction(self):
        with tempfile.TemporaryDirectory() as tmp:
            screen = _config(tmp, seed=7301)
            arm_runtime.save_source_checkpoint(
                screen.run_dir(), "a", [], screen.binding, _blobs("a")
            )
            confirm = _config(tmp, seed=7302)
            self.assertNotEqual(screen.config_hash, confirm.config_hash)
            with self.assertRaises(CheckpointError):
                arm_runtime.load_source_checkpoint(screen.run_dir(), confirm.binding)
            drifted = dict(screen.binding)
            drifted["optimizer_contract"] = {
                **drifted["optimizer_contract"],
                "weight_decay": 0.01,
            }
            with self.assertRaises(CheckpointError):
                arm_runtime.load_source_checkpoint(screen.run_dir(), drifted)

    def test_binding_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            run_dir = config.run_dir()
            arm_runtime.save_source_checkpoint(
                run_dir, "a", [], config.binding, _blobs("a")
            )
            bad = dict(config.binding)
            bad["input_hash"] = _h("other-input")
            with self.assertRaises(CheckpointError):
                arm_runtime.load_source_checkpoint(run_dir, bad)


class PredictionManifestTest(unittest.TestCase):
    def test_predictions_reuse_only_on_binding_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            run_dir = config.run_dir()
            binding = config.binding
            arm_runtime.save_source_predictions(
                run_dir, "s1", {"raw": [0.1, 0.2]}, binding
            )
            self.assertEqual(
                arm_runtime.load_source_predictions(run_dir, "s1", binding),
                {"raw": [0.1, 0.2]},
            )
            bad = dict(binding)
            bad["weights_hash"] = _h("other-weights")
            with self.assertRaises(CheckpointError):
                arm_runtime.load_source_predictions(run_dir, "s1", bad)
            with self.assertRaises(CheckpointError):
                arm_runtime.load_source_predictions(run_dir, "missing", binding)


class ManifestTest(unittest.TestCase):
    def test_manifest_last_with_hash_verify(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            artifact = run_dir / "predictions" / "s1.json"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text("{}", encoding="utf-8")
            self.assertFalse((run_dir / "final_manifest.json").exists())
            with self.assertRaises(arm_runtime.ArmError):
                arm_runtime.write_final_manifest(
                    run_dir, {"role": "final"}, [run_dir / "absent.json"]
                )
            out = arm_runtime.write_final_manifest(
                run_dir, {"role": "final"}, [artifact]
            )
            self.assertTrue(out.is_file())
            body = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(body["artifacts"][0]["sha256"], arm_runtime.sha256_file(artifact))

    def test_head_dim_and_weights_binding(self):
        self.assertEqual(arm_runtime.check_head_input_dim(199), 199)
        with self.assertRaises(arm_runtime.ArmError):
            arm_runtime.check_head_input_dim(200)
        bound, digest = arm_runtime.bind_class_weights(
            {"replacement_positive_weight": 9.5, "anchor_positive_weight": 0.5}
        )
        self.assertEqual(bound["replacement_positive_weight"], 9.5)
        self.assertEqual(digest, arm_runtime.canonical_sha256(bound))
        with self.assertRaises(arm_runtime.ArmError):
            arm_runtime.bind_class_weights({"replacement_positive_weight": 1.0})

    def test_hash_fields_require_lowercase_64_hex(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = {
                "arm": ARM_R_H_SC,
                "seed": 7301,
                "root": str(Path(tmp)),
                "input_hash": _h("input"),
                "checkpoint_hash": _h("checkpoint"),
                "partition_hash": _h("partition"),
                "weights_hash": _h("weights"),
                "code_hash": _h("code"),
            }
            arm_runtime.config_from_dict(dict(base))
            for key in (
                "input_hash",
                "checkpoint_hash",
                "partition_hash",
                "weights_hash",
                "code_hash",
            ):
                bad = dict(base)
                bad[key] = "short"
                with self.assertRaises(arm_runtime.ArmError):
                    arm_runtime.config_from_dict(bad)
                upper = dict(base)
                upper[key] = base[key].upper()
                with self.assertRaises(arm_runtime.ArmError):
                    arm_runtime.config_from_dict(upper)


if __name__ == "__main__":
    unittest.main()
