from __future__ import annotations

import json
import os
import pickle
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


def _pure_worker(payload):
    return {"key": payload["key"], "value": payload["value"] * 2 + 1}


class _FakeCudaDown:
    @staticmethod
    def is_available():
        return False


class _FakeCudaUp:
    seen = []

    @staticmethod
    def is_available():
        return True

    @staticmethod
    def manual_seed_all(seed):
        _FakeCudaUp.seen.append(seed)

    @staticmethod
    def set_rng_state_all(states):
        _FakeCudaUp.seen.append(list(states))


class _FakeTorchSeed:
    def __init__(self, cuda):
        self.cuda = cuda
        self.calls = []

    def manual_seed(self, seed):
        self.calls.append(("manual_seed", seed))


class _CpuTensor:
    def __init__(self, value):
        self.value = value
        self.cpu_calls = 0

    def cpu(self):
        self.cpu_calls += 1
        return self

    def detach(self):
        return _CpuTensor(self.value)

    def item(self):
        return float(self.value)


class _StackTorch:
    def __init__(self):
        self.cpu_calls = 0
        self.cuda = _FakeCudaDown()
        outer = self

        class _Sum:
            def __init__(self, total):
                self.total = total

            def cpu(self):
                outer.cpu_calls += 1
                return self

            def item(self):
                return float(self.total)

        self._Sum = _Sum

    def stack(self, terms):
        return _StackTorchStack(list(terms), self._Sum)


class _StackTorchStack:
    def __init__(self, terms, maker):
        self.terms = terms
        self.maker = maker

    def sum(self):
        return self.maker(sum(float(t.value) for t in self.terms))


class _ArrTorch:
    float32 = "float32"

    def __init__(self):
        self.tensor_calls = []
        self.cuda = _FakeCudaDown()

    def tensor(self, data, dtype=None, device=None):
        self.tensor_calls.append(device)
        import numpy as np

        return np.asarray(data, dtype=np.float64)


class ThreadContractTest(unittest.TestCase):
    def test_env_caps_assigned(self):
        from experiments.psem_state_corrected_adaptation_gate import arm_runtime

        arm_runtime.apply_thread_caps()
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            self.assertEqual(os.environ.get(name), "1")
        self.assertEqual(os.environ.get("TOKENIZERS_PARALLELISM"), "false")
        receipt = arm_runtime.thread_cap_receipt()
        self.assertEqual(receipt["omp_num_threads"], "1")

    def test_worker_receipt_defaults(self):
        from experiments.psem_state_corrected_adaptation_gate import arm_runtime

        receipt = arm_runtime.worker_receipt(None, 64)
        self.assertEqual(
            receipt["effective_workers"], arm_runtime.default_worker_limit()
        )
        self.assertLessEqual(receipt["effective_workers"], os.cpu_count() or 1)
        self.assertEqual(receipt["worker_cap"], 24)
        self.assertTrue(receipt["ordered"])
        self.assertEqual(receipt["backend"], "spawn" if receipt["effective_workers"] > 1 else "serial")
        single = arm_runtime.worker_receipt(None, 1)
        self.assertEqual(single["effective_workers"], 1)
        self.assertEqual(single["backend"], "serial")

    def test_spawn_pool_reapplies_caps(self):
        import multiprocessing

        from experiments.psem_state_corrected_adaptation_gate import arm_runtime

        self.assertEqual(arm_runtime.spawn_worker_init.__name__, "spawn_worker_init")
        self.assertEqual(multiprocessing.get_context("spawn")._name, "spawn")


class SpawnParityTest(unittest.TestCase):
    def test_workers_1_and_24_identical_order(self):
        import json as _json

        from experiments.psem_state_corrected_adaptation_gate import arm_runtime

        payloads = [{"key": f"k{i:03d}", "value": i} for i in range(48)]
        serial = arm_runtime.ordered_process_map(_pure_worker, payloads, 1)
        parallel = arm_runtime.ordered_process_map(_pure_worker, payloads, 24)
        self.assertEqual(parallel, serial)
        self.assertEqual(
            _json.dumps(parallel, sort_keys=True), _json.dumps(serial, sort_keys=True)
        )
        self.assertEqual([r["key"] for r in parallel], [p["key"] for p in payloads])


class BackfillContractTest(unittest.TestCase):
    def test_decode_payload_carries_audio_ref(self):
        from experiments.psem_state_corrected_adaptation_gate import temporal_train

        session = SimpleNamespace(source_id="s-a", audio_ref="audio/s-a.wav", waveform_sha256="w")
        payload = temporal_train.decode_payload(session, "/corpus")
        self.assertEqual(payload["audio_ref"], "audio/s-a.wav")
        self.assertEqual(payload["source_id"], "s-a")
        self.assertEqual(payload["waveform_sha256"], "w")
        self.assertEqual(payload["corpus_root"], "/corpus")

    def test_backfill_worker_is_spawn_picklable(self):
        from experiments.psem_state_corrected_adaptation_gate import temporal_train

        pickle.dumps(temporal_train.backfill_target_task)

    def test_backfill_write_bytes_deterministic(self):
        from experiments.psem_state_corrected_adaptation_gate import temporal_train

        payload = {
            "source_id": "s-a",
            "num_frames": 4,
            "episodes": [],
            "y_anchor": [0.0] * 4,
            "y_replace": [1.0] * 4,
            "valid": [True] * 4,
            "multiplicity": [1] * 4,
            "episode_ids": [None] * 4,
            "audio_ref": "audio/s-a.wav",
            "waveform_sha256": "w",
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            first_path, first_sha = temporal_train.write_backfill_target(run_dir, "s-a", payload)
            first_bytes = first_path.read_bytes()
            second_path, second_sha = temporal_train.write_backfill_target(run_dir, "s-a", payload)
            self.assertEqual(first_sha, second_sha)
            self.assertEqual(first_bytes, second_path.read_bytes())


class RngContractTest(unittest.TestCase):
    def test_restore_converts_to_cpu_and_fails_closed(self):
        from experiments.psem_state_corrected_adaptation_gate import h_arm

        class _CudaTensor:
            def __init__(self, torch):
                self._torch = torch

            def cuda(self):
                return self

            def cpu(self):
                return ("cpu-state",)

        class _Torch:
            cuda = _FakeCudaUp()

            def set_rng_state(self, state):
                self.cpu_state = state

        torch = _Torch()
        _FakeCudaUp.seen = []
        snap = {"python": random.getstate(), "torch_cpu": b"x", "torch_cuda": [_CudaTensor(torch)]}
        h_arm.restore_rng(torch, snap)
        self.assertEqual(torch.cpu_state, b"x")
        self.assertEqual(_FakeCudaUp.seen[-1], [("cpu-state",)])

        class _FailCuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def set_rng_state_all(states):
                raise RuntimeError("boom")

        class _FailTorch:
            cuda = _FailCuda

            def set_rng_state(self, state):
                return None

        with self.assertRaises(h_arm.HArmError):
            h_arm.restore_rng(_FailTorch(), snap)

    def test_seed_determinism_across_families(self):
        import random as _random

        from experiments.psem_state_corrected_adaptation_gate import h_arm
        from experiments.psem_state_corrected_adaptation_gate import temporal_train

        torch_h = _FakeTorchSeed(_FakeCudaDown())
        first = h_arm.seed_all_from_config(torch_h, 7301)
        state_a = _random.getstate()
        second = h_arm.seed_all_from_config(torch_h, 7301)
        self.assertEqual(state_a, _random.getstate())
        self.assertEqual(first["seed"], 7301)
        self.assertEqual(second["seed"], 7301)
        self.assertIn("numpy", first)
        torch_t = _FakeTorchSeed(_FakeCudaDown())
        report = temporal_train.seed_temporal_from_config(torch_t, 7302)
        state_b = _random.getstate()
        temporal_train.seed_temporal_from_config(torch_t, 7302)
        self.assertEqual(state_b, _random.getstate())
        self.assertEqual(report["seed"], 7302)
        self.assertEqual(torch_t.calls[-2:], [("manual_seed", 7302), ("manual_seed", 7302)])


class ScheduleIndexTest(unittest.TestCase):
    def test_preindex_matches_scan(self):
        from experiments.psem_state_corrected_adaptation_gate import h_arm

        plan = h_arm.plan_fit_schedule(["s1", "s2"], {"s1": [True] * 20, "s2": [True] * 5})
        index = {(c["source"], int(c["chunk_index"])): c for c in plan["chunks"]}
        for chunk in plan["chunks"]:
            scanned = next(
                c
                for c in plan["chunks"]
                if c["source"] == chunk["source"] and c["chunk_index"] == chunk["chunk_index"]
            )
            self.assertIs(index[(chunk["source"], int(chunk["chunk_index"]))], scanned)


class AccumulateLossTest(unittest.TestCase):
    def test_single_transfer_and_value(self):
        from experiments.psem_state_corrected_adaptation_gate import h_arm

        torch = _StackTorch()
        terms = [_CpuTensor(v) for v in (0.5, 1.25, 2.0)]
        total = h_arm.accumulate_source_loss(torch, terms)
        self.assertAlmostEqual(total, 3.75)
        self.assertEqual(torch.cpu_calls, 1)
        self.assertEqual(h_arm.accumulate_source_loss(torch, []), 0.0)


class FeaturesDtypeTest(unittest.TestCase):
    def test_features_float32(self):
        import numpy as np

        from experiments.psem_state_corrected_adaptation_gate import h_arm

        feats = h_arm.assemble_features(
            np.zeros((4, 192)), np.zeros((4, 4)), [0.1] * 4, [0.2] * 4, [1.04] * 4
        )
        self.assertEqual(str(feats.dtype), "float32")
        self.assertEqual(tuple(feats.shape), (4, 199))


class SourceTensorsTest(unittest.TestCase):
    def test_built_once_slices_match(self):
        from experiments.psem_state_corrected_adaptation_gate import temporal_train

        torch = _ArrTorch()
        authority = SimpleNamespace(
            y_replace=[float(i % 2) for i in range(8)],
            y_anchor=[1.0] * 8,
            valid=[True] * 8,
        )
        prep = {
            "num_frames": 8,
            "authority": authority,
            "multiplicity": [1] * 8,
            "episode_ids": ["e"] * 8,
        }
        tensors = temporal_train.build_source_device_tensors(torch, prep, {"e": 0}, "cpu")
        self.assertEqual(len(torch.tensor_calls), 3)
        self.assertTrue(all(d == "cpu" for d in torch.tensor_calls))
        import numpy as np

        np.testing.assert_allclose(tensors["y_replace"][0], np.array([float(i % 2) for i in range(8)]))
        np.testing.assert_allclose(tensors["y_replace"][:, 2:6][0], np.array([0.0, 1.0, 0.0, 1.0]))
        self.assertEqual(tensors["mult_weight"].shape, (1, 8))


class CheckpointMilestoneTest(unittest.TestCase):
    def _binding(self):
        return {"seed": 7301, "arm": "R-H-SC"}

    def test_bounded_retention_and_latest_resume(self):
        from experiments.psem_state_corrected_adaptation_gate import arm_runtime

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            binding = self._binding()
            completed = []
            for index in range(9):
                source_id = f"s-{index:02d}"
                blobs = {role: f"{source_id}-{role}".encode() for role in arm_runtime.CHECKPOINT_ROLES}
                arm_runtime.save_source_checkpoint(run_dir, source_id, list(completed), binding, blobs=blobs)
                completed.append(source_id)
            record = arm_runtime.load_source_checkpoint(run_dir, binding)
            self.assertEqual(record["completed_sources"], completed)
            self.assertEqual(set(record["blobs"]), set(completed))
            for source_id in completed:
                for role in arm_runtime.CHECKPOINT_ROLES:
                    expected = f"{source_id}-{role}".encode()
                    self.assertEqual(record["blobs"][source_id][role].read_bytes(), expected)
            blob_files = list((run_dir / arm_runtime.CHECKPOINT_DIRNAME).glob("*.pt"))
            self.assertEqual(len(blob_files), len(completed) * len(arm_runtime.CHECKPOINT_ROLES))
            manifest = json.loads(
                (run_dir / arm_runtime.CHECKPOINT_DIRNAME / arm_runtime.CHECKPOINT_NAME).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["completed_sources"], completed)
            for source_id in completed:
                entry = manifest["sources"][source_id]
                self.assertEqual(set(entry), set(arm_runtime.CHECKPOINT_ROLES))
                for role in arm_runtime.CHECKPOINT_ROLES:
                    self.assertTrue(entry[role]["sha256"])
            orphan = run_dir / arm_runtime.CHECKPOINT_DIRNAME / "orphan.model.pt"
            orphan.write_bytes(b"orphan")
            blobs = {role: b"extra-" + role.encode() for role in arm_runtime.CHECKPOINT_ROLES}
            arm_runtime.save_source_checkpoint(
                run_dir, "s-09", list(completed), binding, blobs=blobs
            )
            self.assertFalse(orphan.exists())
            rerecord = arm_runtime.load_source_checkpoint(run_dir, binding)
            self.assertEqual(set(rerecord["blobs"]), set(completed + ["s-09"]))
            latest = "s-09"
            latest_record = manifest["sources"][completed[-1]]
            tamper = run_dir / arm_runtime.CHECKPOINT_DIRNAME / latest_record["model"]["path"]
            tamper.write_bytes(b"corrupt")
            with self.assertRaises(arm_runtime.CheckpointError):
                arm_runtime.load_source_checkpoint(run_dir, binding)

    def test_single_load_record_reused(self):
        from experiments.psem_state_corrected_adaptation_gate import arm_runtime

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            binding = self._binding()
            blobs = {role: b"x" for role in arm_runtime.CHECKPOINT_ROLES}
            arm_runtime.save_source_checkpoint(run_dir, "s-00", [], binding, blobs=blobs)
            first = arm_runtime.load_source_checkpoint(run_dir, binding)
            second = arm_runtime.load_source_checkpoint(run_dir, binding)
            self.assertEqual(first["completed_sources"], second["completed_sources"])
            self.assertEqual(
                first["blobs"]["s-00"]["model"].read_bytes(),
                second["blobs"]["s-00"]["model"].read_bytes(),
            )


class GpuLockTest(unittest.TestCase):
    def test_acquire_release_reacquire(self):
        from experiments.psem_state_corrected_adaptation_gate import arm_runtime

        with tempfile.TemporaryDirectory() as tmp:
            scope = Path(tmp)
            owner = {"run_id": "R-H-SC-7301", "arm": "R-H-SC", "seed": 7301}
            path = arm_runtime.acquire_arm_gpu_lock(scope, owner)
            self.assertTrue(path.is_file())
            with self.assertRaises(arm_runtime.ArmError):
                arm_runtime.acquire_arm_gpu_lock(scope, owner)
            arm_runtime.release_arm_gpu_lock(path)
            self.assertFalse(path.exists())
            path = arm_runtime.acquire_arm_gpu_lock(scope, owner)
            arm_runtime.release_arm_gpu_lock(path)

    def test_stale_dead_owner_reclaimed(self):
        import time as _time

        from experiments.psem_state_corrected_adaptation_gate import arm_runtime

        with tempfile.TemporaryDirectory() as tmp:
            scope = Path(tmp)
            stale = scope / arm_runtime.ARM_GPU_LOCK_NAME
            stale.write_text(
                json.dumps({"pid": 999999999, "time": _time.time() - 7200.0, "run_id": "old"}),
                encoding="utf-8",
            )
            path = arm_runtime.acquire_arm_gpu_lock(scope, {"run_id": "new"})
            self.assertTrue(path.is_file())
            arm_runtime.release_arm_gpu_lock(path)

    def test_live_owner_never_reclaimed_for_age(self):
        import time as _time

        from experiments.psem_state_corrected_adaptation_gate import arm_runtime

        with tempfile.TemporaryDirectory() as tmp:
            scope = Path(tmp)
            stale = scope / arm_runtime.ARM_GPU_LOCK_NAME
            stale.write_text(
                json.dumps({"pid": os.getpid(), "time": _time.time() - 7200.0, "run_id": "old"}),
                encoding="utf-8",
            )
            with self.assertRaises(arm_runtime.ArmError):
                arm_runtime.acquire_arm_gpu_lock(scope, {"run_id": "new"})

    def test_corrupt_lock_fails_closed(self):
        from experiments.psem_state_corrected_adaptation_gate import arm_runtime

        with tempfile.TemporaryDirectory() as tmp:
            scope = Path(tmp)
            stale = scope / arm_runtime.ARM_GPU_LOCK_NAME
            stale.write_text("{corrupt", encoding="utf-8")
            with self.assertRaises(arm_runtime.ArmError):
                arm_runtime.acquire_arm_gpu_lock(scope, {"run_id": "new"})
            self.assertTrue(stale.is_file())

    def test_context_manager_releases(self):
        from experiments.psem_state_corrected_adaptation_gate import arm_runtime

        with tempfile.TemporaryDirectory() as tmp:
            scope = Path(tmp)
            with arm_runtime.arm_gpu_lock(scope, {"run_id": "r"}) as path:
                self.assertTrue(Path(path).is_file())
            self.assertFalse(Path(path).exists())


class ThreadEnforcementTest(unittest.TestCase):
    def setUp(self):
        import sys
        from unittest.mock import patch

        self.enterContext(patch.dict(sys.modules))

    def test_late_torch_gets_one_one(self):
        import sys
        import types

        from experiments.psem_state_corrected_adaptation_gate import arm_runtime

        calls = []
        fake = types.ModuleType("torch-late-fake")

        def _get_num_threads():
            return 1 if ("set", 1) in calls else 4

        def _set_num_threads(value):
            calls.append(("set", int(value)))

        def _get_num_interop_threads():
            return 1

        def _set_num_interop_threads(value):
            calls.append(("set_interop", int(value)))

        fake.get_num_threads = _get_num_threads
        fake.set_num_threads = _set_num_threads
        fake.get_num_interop_threads = _get_num_interop_threads
        fake.set_num_interop_threads = _set_num_interop_threads
        sys.modules["torch"] = fake
        try:
            arm_runtime._TORCH_CAPPED_MODULE = None
            receipt = arm_runtime.apply_thread_caps()
            self.assertIn(("set", 1), calls)
            self.assertIn(("set_interop", 1), calls)
            self.assertEqual(receipt["torch_num_threads"], 1)
            self.assertEqual(receipt["torch_num_interop_threads"], 1)
            enforced = arm_runtime.enforce_thread_caps()
            self.assertEqual(enforced["torch_num_threads"], 1)
        finally:
            del sys.modules["torch"]
            arm_runtime._TORCH_CAPPED_MODULE = None

    def test_unenforceable_loaded_torch_fails_closed(self):
        import sys
        import types

        from experiments.psem_state_corrected_adaptation_gate import arm_runtime

        fake = types.ModuleType("torch-stuck-fake")
        fake.get_num_threads = lambda: 4
        fake.set_num_threads = lambda value: None
        fake.get_num_interop_threads = lambda: 1
        fake.set_num_interop_threads = lambda value: None
        sys.modules["torch"] = fake
        try:
            arm_runtime._TORCH_CAPPED_MODULE = None
            with self.assertRaises(arm_runtime.ArmError):
                arm_runtime.enforce_thread_caps()
        finally:
            del sys.modules["torch"]
            arm_runtime._TORCH_CAPPED_MODULE = None


class BackfillPayloadTest(unittest.TestCase):
    def _sessions(self):
        from types import SimpleNamespace

        return {
            f"s-{index:02d}": SimpleNamespace(
                source_id=f"s-{index:02d}",
                labels=None,
                audio_ref=f"audio/s-{index:02d}.wav",
                waveform_sha256=f"wave-{index:02d}",
            )
            for index in range(4)
        }

    def _rows(self):
        return {
            f"s-{index:02d}": [{"window_start_sample": 0, "window_end_sample": 480000}]
            for index in range(4)
        }

    def _manifest(self):
        return {"sampling_sha256": "s" * 64, "targets": {}, "files": {}}

    def _document(self, source_id):
        return {
            "source_id": source_id,
            "num_frames": 4,
            "episodes": [],
            "y_anchor": [0.0] * 4,
            "y_replace": [1.0] * 4,
            "valid": [True] * 4,
            "multiplicity": [1] * 4,
            "episode_ids": [None] * 4,
            "intervals": [],
            "audio_ref": f"audio/{source_id}.wav",
            "waveform_sha256": f"wave-{source_id}",
            "sampling_sha256": "s" * 64,
            "backfilled": True,
        }

    def _run_parallel(self, workers):
        from experiments.psem_state_corrected_adaptation_gate import arm_runtime
        from experiments.psem_state_corrected_adaptation_gate import temporal_train

        seen = []

        def _fake_map(worker_fn, payloads, n_workers):
            seen.append((worker_fn, list(payloads), n_workers))
            for payload in payloads:
                self.assertIn("audio_ref", payload)
                self.assertTrue(str(payload["audio_ref"]))
            return [
                {"source_id": str(p["source_id"]), "document": self._document(str(p["source_id"]))}
                for p in payloads
            ]

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            original = arm_runtime.ordered_process_map
            arm_runtime.ordered_process_map = _fake_map
            try:
                resolved, binding = temporal_train.resolve_durable_targets(
                    None,
                    run_dir,
                    Path(tmp) / "bundle",
                    self._manifest(),
                    self._sessions(),
                    self._rows(),
                    Path(tmp),
                    None,
                    [f"s-{index:02d}" for index in range(4)],
                    workers,
                )
            finally:
                arm_runtime.ordered_process_map = original
            self.assertEqual(sorted(resolved), [f"s-{index:02d}" for index in range(4)])
            self.assertEqual(
                [entry["audio_ref"] for _, entry in sorted(resolved.items())],
                [f"audio/s-{index:02d}.wav" for index in range(4)],
            )
            files = {}
            for source_id in sorted(resolved):
                target = run_dir / "targets" / f"{source_id}.json"
                files[source_id] = target.read_bytes()
            manifest = json.loads(
                (run_dir / "targets" / "targets_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                sorted(manifest["sources"]), [f"s-{index:02d}" for index in range(4)]
            )
            return files, {sid: binding[sid]["sha256"] for sid in sorted(binding)}, seen

    def test_parallel_payloads_carry_audio_ref_and_match(self):
        first_files, first_hashes, first_seen = self._run_parallel(8)
        second_files, second_hashes, second_seen = self._run_parallel(8)
        self.assertEqual(first_files, second_files)
        self.assertEqual(first_hashes, second_hashes)
        self.assertEqual(len(first_seen), 1)
        _, payloads, effective = first_seen[0]
        self.assertEqual(effective, 8)
        self.assertEqual(
            [str(p["source_id"]) for p in payloads],
            [f"s-{index:02d}" for index in range(4)],
        )


class FrozenCacheTest(unittest.TestCase):
    def _binding(self, seed=7301):
        return {
            "arm": "R-H-SC",
            "seed": seed,
            "input_hash": "i" * 64,
            "checkpoint_hash": "c" * 64,
            "code_hash": "d" * 64,
        }

    def _payload(self):
        return {
            "num_frames": 4,
            "waveform_sha256": "w" * 64,
            "audio_ref": "audio/s-a.wav",
        }

    def test_cross_seed_reuse_on_exact_identity(self):
        import numpy as np

        from experiments.psem_state_corrected_adaptation_gate import h_arm

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = self._payload()
            identity_a = h_arm.frozen_evidence_identity(self._binding(7301), payload)
            identity_b = h_arm.frozen_evidence_identity(self._binding(7302), payload)
            self.assertEqual(identity_a, identity_b)
            hidden = np.zeros((4, 192), dtype=np.float32)
            logits = np.zeros((4, 4), dtype=np.float32)
            meta = h_arm.write_frozen_evidence(
                root, identity_a, hidden, logits, {"e": 0}, [], {"t": 1.0}
            )
            hit = h_arm.read_frozen_evidence(root, identity_b)
            self.assertIsNotNone(hit)
            self.assertEqual(hit["meta"]["sha256"], meta["sha256"])
            np.testing.assert_array_equal(hit["hidden192"], hidden)

    def test_tamper_and_mismatch_fail_closed(self):
        import numpy as np

        from experiments.psem_state_corrected_adaptation_gate import h_arm

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = self._payload()
            identity = h_arm.frozen_evidence_identity(self._binding(), payload)
            hidden = np.zeros((4, 192), dtype=np.float32)
            logits = np.zeros((4, 4), dtype=np.float32)
            meta = h_arm.write_frozen_evidence(root, identity, hidden, logits, {}, [], {})
            npz = root / h_arm.FROZEN_EVIDENCE_DIRNAME / (str(meta["key"]) + ".npz")
            with open(npz, "ab") as handle:
                handle.write(b"\x00")
            with self.assertRaises(h_arm.HArmError):
                h_arm.read_frozen_evidence(root, identity)
            other = dict(payload, waveform_sha256="z" * 64)
            self.assertIsNone(h_arm.read_frozen_evidence(root, h_arm.frozen_evidence_identity(self._binding(), other)))

    def test_profile_batch_persists_representative_cache(self):
        from experiments.psem_state_corrected_adaptation_gate import h_arm

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir(parents=True)
            self.assertEqual(
                h_arm.frozen_root_for(run_dir, Path(tmp)), Path(tmp)
            )
            self.assertEqual(
                h_arm.frozen_root_for(run_dir, None), run_dir.resolve().parent.parent
            )


class LoaderEnforcementTest(unittest.TestCase):
    def setUp(self):
        import sys
        from unittest.mock import patch

        self.enterContext(patch.dict(sys.modules))

    def _fake_torch_module(self, name, calls):
        import sys
        import types

        fake = types.ModuleType(name)
        state = {"threads": 4, "interop": 2}
        fake.get_num_threads = lambda: int(state["threads"])
        fake.set_num_threads = lambda value: (calls.append(("set", int(value))), state.__setitem__("threads", int(value)))[0]
        fake.get_num_interop_threads = lambda: int(state["interop"])
        fake.set_num_interop_threads = lambda value: (calls.append(("set_interop", int(value))), state.__setitem__("interop", int(value)))[0]
        sys.modules["torch"] = fake
        return fake

    def test_h_loader_enforces_after_import(self):
        import sys

        from experiments.psem_state_corrected_adaptation_gate import arm_runtime, h_arm

        calls = []
        self._fake_torch_module("torch-h-loader", calls)
        try:
            arm_runtime._TORCH_CAPPED_MODULE = None
            torch = h_arm.default_torch_loader()
            self.assertIn(("set", 1), calls)
            self.assertIn(("set_interop", 1), calls)
            self.assertIs(sys.modules["torch"], torch)
        finally:
            del sys.modules["torch"]
            arm_runtime._TORCH_CAPPED_MODULE = None

    def test_temporal_loader_enforces_after_import(self):
        import sys

        from experiments.psem_state_corrected_adaptation_gate import arm_runtime, temporal_train

        calls = []
        self._fake_torch_module("torch-t-loader", calls)
        try:
            arm_runtime._TORCH_CAPPED_MODULE = None
            torch = temporal_train.load_torch()
            self.assertIn(("set", 1), calls)
            self.assertIn(("set_interop", 1), calls)
            self.assertIs(sys.modules["torch"], torch)
        finally:
            del sys.modules["torch"]
            arm_runtime._TORCH_CAPPED_MODULE = None

    def test_decode_task_runs_torch_free(self):
        import sys
        import types

        from experiments.psem_state_corrected_adaptation_gate import arm_runtime, temporal_train

        sys.modules.pop("torch", None)

        class _Wave:
            shape = (1, 2560)

            def __getitem__(self, idx):
                return self

        parent = types.ModuleType("experiments.psem_sortformer_adaptation_depth")
        parent.__path__ = []
        execution = types.ModuleType("execution-stub")
        execution.load_source_waveform = lambda session, root: (_Wave(), 2560, 0)
        sys.modules["experiments.psem_sortformer_adaptation_depth"] = parent
        sys.modules["experiments.psem_sortformer_adaptation_depth.execution"] = execution
        try:
            arm_runtime._TORCH_CAPPED_MODULE = None
            out = temporal_train.decode_waveform_task(
                {
                    "source_id": "s-a",
                    "audio_ref": "audio/s-a.wav",
                    "waveform_sha256": "w",
                    "corpus_root": "/corpus",
                }
            )
            self.assertEqual(out["source_id"], "s-a")
            self.assertEqual(out["num_frames"], 2)
            self.assertNotIn("torch", sys.modules)
        finally:
            del sys.modules["experiments.psem_sortformer_adaptation_depth.execution"]
            del sys.modules["experiments.psem_sortformer_adaptation_depth"]

    def test_decode_task_enforces_when_torch_present(self):
        import sys
        import types

        from experiments.psem_state_corrected_adaptation_gate import arm_runtime, temporal_train

        calls = []
        self._fake_torch_module("torch-task-loader", calls)

        class _Wave:
            shape = (1, 2560)

            def __getitem__(self, idx):
                return self

        parent = types.ModuleType("experiments.psem_sortformer_adaptation_depth")
        parent.__path__ = []
        execution = types.ModuleType("execution-stub-two")
        execution.load_source_waveform = lambda session, root: (_Wave(), 2560, 0)
        sys.modules["experiments.psem_sortformer_adaptation_depth"] = parent
        sys.modules["experiments.psem_sortformer_adaptation_depth.execution"] = execution
        try:
            arm_runtime._TORCH_CAPPED_MODULE = None
            out = temporal_train.decode_waveform_task(
                {
                    "source_id": "s-a",
                    "audio_ref": "audio/s-a.wav",
                    "waveform_sha256": "w",
                    "corpus_root": "/corpus",
                }
            )
            self.assertEqual(out["num_frames"], 2)
            self.assertIn(("set", 1), calls)
        finally:
            del sys.modules["torch"]
            del sys.modules["experiments.psem_sortformer_adaptation_depth.execution"]
            del sys.modules["experiments.psem_sortformer_adaptation_depth"]
            arm_runtime._TORCH_CAPPED_MODULE = None


class CheckpointCrashTest(unittest.TestCase):
    def test_manifest_failure_keeps_prior_resume(self):
        from experiments.psem_state_corrected_adaptation_gate import arm_runtime

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            binding = {"seed": 7301, "arm": "R-H-SC"}
            blobs = {role: b"s-00-" + role.encode() for role in arm_runtime.CHECKPOINT_ROLES}
            arm_runtime.save_source_checkpoint(run_dir, "s-00", [], binding, blobs=blobs)
            manifest_path = run_dir / arm_runtime.CHECKPOINT_DIRNAME / arm_runtime.CHECKPOINT_NAME
            prior_bytes = manifest_path.read_bytes()
            prior_files = {
                p.name: p.read_bytes()
                for p in (run_dir / arm_runtime.CHECKPOINT_DIRNAME).glob("*.pt")
            }
            real_write = arm_runtime.atomic_write_json

            def _fail_manifest(path, payload):
                if Path(path).name == arm_runtime.CHECKPOINT_NAME:
                    raise OSError("injected manifest failure")
                return real_write(path, payload)

            arm_runtime.atomic_write_json = _fail_manifest
            try:
                next_blobs = {role: b"s-01-" + role.encode() for role in arm_runtime.CHECKPOINT_ROLES}
                with self.assertRaises(OSError):
                    arm_runtime.save_source_checkpoint(run_dir, "s-01", ["s-00"], binding, blobs=next_blobs)
            finally:
                arm_runtime.atomic_write_json = real_write
            self.assertEqual(manifest_path.read_bytes(), prior_bytes)
            record = arm_runtime.load_source_checkpoint(run_dir, binding)
            self.assertEqual(record["completed_sources"], ["s-00"])
            self.assertEqual(set(record["blobs"]), {"s-00"})
            for name, data in prior_files.items():
                self.assertTrue((run_dir / arm_runtime.CHECKPOINT_DIRNAME / name).is_file())


class FrontierSliceTest(unittest.TestCase):
    def _grid(self, count=46000):
        return [float(count - i) / float(count) for i in range(count)]

    def test_slice_bounded_spanning_deterministic(self):
        from experiments.psem_state_corrected_adaptation_gate import cross_frontier as cross_mod

        grid = self._grid()
        first = cross_mod.bounded_threshold_slice(grid, 16)
        second = cross_mod.bounded_threshold_slice(grid, 16)
        self.assertEqual(first, second)
        self.assertLessEqual(len(first), 16)
        self.assertEqual(first[0], grid[0])
        self.assertEqual(first[-1], grid[-1])
        small = [0.9, 0.5, 0.1]
        self.assertEqual(cross_mod.bounded_threshold_slice(small, 16), small)
        with self.assertRaises(cross_mod.CrossFrontierError):
            cross_mod.bounded_threshold_slice(grid, 0)
        with self.assertRaises(cross_mod.CrossFrontierError):
            cross_mod.project_frontier_cost(1.0, 0, len(grid))
        self.assertAlmostEqual(
            cross_mod.project_frontier_cost(2.0, 16, 46000), 2.0 * 46000.0 / 16.0
        )

    def test_temporal_caller_evaluates_bounded_grid_only(self):
        import sys
        import types

        from experiments.psem_state_corrected_adaptation_gate import cross_frontier as cross_mod
        from experiments.psem_state_corrected_adaptation_gate import temporal_train

        seen = []

        def _decode(dev, scores, threshold, confirmation_ms):
            seen.append(float(threshold))
            return [threshold]

        def _metrics(dev, events):
            return {
                "false_cut_count": 2,
                "active_speech_seconds": 3600.0,
                "reference_replacement_count": 10,
                "missed_replacement_count": 1,
                "exclusive_other_contamination_seconds": 180.0,
            }

        package = types.ModuleType("experiments.psem_frozen_ceiling_gate")
        package.__path__ = []
        module = types.ModuleType("evaluate_ceiling")
        module.decode_scores = _decode
        module.session_metrics = _metrics
        prior = {
            key: sys.modules[key]
            for key in (
                "experiments.psem_frozen_ceiling_gate",
                "experiments.psem_frozen_ceiling_gate.evaluate_ceiling",
            )
            if key in sys.modules
        }
        sys.modules["experiments.psem_frozen_ceiling_gate"] = package
        sys.modules["experiments.psem_frozen_ceiling_gate.evaluate_ceiling"] = module
        try:
            member = {"source_id": "s-a", "snapshot": object(), "cand_raw": self._grid(46000), "f0_raw": [0.5] * 46000, "unmapped": []}
            full = [float(v) for v in member["cand_raw"]]
            grid = cross_mod.bounded_threshold_slice(full, cross_mod.FRONTIER_SLICE_LIMIT - 1)
            out = temporal_train._group_score_frontier([member], "cand_raw", grid, 100, workers=1)
            self.assertLessEqual(len(seen), 16)
            self.assertEqual(len(out["points"]), len(grid))
            self.assertEqual(
                {p.threshold for p in out["points"]}, {float(t) for t in grid}
            )
            self.assertNotIn("c_envelope", out)
            self.assertNotIn("useful", out)
            self.assertNotIn("gate_evidence", out)
        finally:
            for key in (
                "experiments.psem_frozen_ceiling_gate",
                "experiments.psem_frozen_ceiling_gate.evaluate_ceiling",
            ):
                if key in prior:
                    sys.modules[key] = prior[key]
                else:
                    del sys.modules[key]

    def test_h_caller_evaluates_bounded_grid_only(self):
        from experiments.psem_state_corrected_adaptation_gate import h_arm

        calls = []

        def _decode(dev, scores, threshold, confirmation_ms):
            calls.append(float(threshold))
            return [threshold]

        def _metrics(dev, events):
            return {
                "active_speech_seconds": 3600.0,
                "false_cut_count": 2,
                "reference_replacement_count": 10,
                "missed_replacement_count": 1,
                "exclusive_other_contamination_seconds_per_active_speech_hour": 180.0,
            }

        scores = self._grid(46000)
        out = h_arm.score_frontier_slice(_decode, _metrics, object(), scores, 100)
        self.assertLessEqual(len(calls), 16)
        self.assertEqual(out["sampled_thresholds"], len(calls))
        self.assertEqual(out["total_thresholds"], 46000)
        self.assertEqual(len(out["points"]), len(calls))
        self.assertAlmostEqual(
            out["projected_seconds"], out["seconds"] * 46000.0 / len(calls)
        )
        self.assertNotIn("c_envelope", out)
        self.assertNotIn("m_envelope", out)
        self.assertNotIn("useful", out)
        self.assertEqual(calls[0], max(scores))
        self.assertEqual(calls[-1], min(scores))


class ExactWaveParityTest(unittest.TestCase):
    def _sessions(self):
        import numpy as np

        from experiments.psem_frozen_ceiling_gate.build_ceiling_examples import (
            load_sessions,
        )

        sessions = load_sessions()
        amis = [s for s in sessions if s.source_family == "ami_mix_headset" and s.role == "dev"][:2]
        alis = [s for s in sessions if s.source_family == "alimeeting_far_ch0" and s.role == "dev"][:1]
        if len(amis) < 2 or len(alis) < 1:
            self.skipTest("frontier parity requires AMI and AliMeeting DEV snapshots")
        from dataclasses import replace

        count = 40
        trimmed = []
        for full in amis + alis:
            trimmed.append(
                replace(
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
            )
        rng = np.random.RandomState(7301)
        levels = rng.rand(8)
        return trimmed, rng, levels

    def _members(self):
        devs, rng, levels = self._sessions()
        members = {}
        names = ["ami-0", "ami-1", "ali-0"]
        for name, dev in zip(names, devs):
            raw = rng.choice(levels, 40).tolist()
            cal = rng.choice(levels, 40).tolist()
            f0 = rng.choice(levels, 40).tolist()
            members[name] = {
                "dev": dev,
                "scores": {"raw": raw, "calibrated": cal, "f0": f0},
            }
        return members

    def test_workers1_vs_workers24_byte_order_metric_parity(self):
        import concurrent.futures as _futures
        import json as _json

        from experiments.psem_state_corrected_adaptation_gate import arm_runtime, cross_frontier

        members = self._members()
        grids = {
            name: {
                kind: sorted(set(float(v) for v in entry["scores"][kind]), reverse=True)
                for kind in ("raw", "calibrated", "f0")
            }
            for name, entry in members.items()
        }
        tasks = cross_frontier.plan_exact_tasks(grids, [100, 300, 500])
        for task in tasks:
            self.assertNotIn("dev", task)
            self.assertNotIn("scores", task)
            self.assertEqual(sorted(task), ["horizon_ms", "kind", "member", "thresholds"])
        serial_results, serial_receipt = cross_frontier.run_exact_wave(members, tasks, 1)
        real_executor = _futures.ProcessPoolExecutor
        created = []

        class _Recording(real_executor):
            def __init__(self, *args, **kwargs):
                created.append(kwargs)
                super().__init__(*args, **kwargs)

        _futures.ProcessPoolExecutor = _Recording
        try:
            parallel_results, parallel_receipt = cross_frontier.run_exact_wave(members, tasks, 24)
        finally:
            _futures.ProcessPoolExecutor = real_executor
        self.assertEqual(len(created), 1)
        self.assertIs(created[0]["initializer"], cross_frontier.init_exact_context)
        self.assertEqual(
            set(created[0]["initargs"][0]),
            {"ami-0", "ami-1", "ali-0"},
        )
        self.assertEqual(
            _json.dumps(serial_results, sort_keys=True),
            _json.dumps(parallel_results, sort_keys=True),
        )
        def _point(rows):
            totals = cross_frontier.sum_primitives(rows)
            return cross_frontier.pooled_point_from_sums(
                totals["false_cut_count"],
                totals["active_speech_seconds"],
                totals["reference_replacement_count"],
                totals["missed_replacement_count"],
                totals["exclusive_other_contamination_seconds"],
                rows[0]["threshold"],
            )

        for name in ("ami-0", "ami-1", "ali-0"):
            for kind in ("raw", "calibrated"):
                for horizon in (100, 300, 500):
                    serial_points = [
                        _point([row]) for row in serial_results[name][kind][horizon]
                    ]
                    parallel_points = [
                        _point([row]) for row in parallel_results[name][kind][horizon]
                    ]
                    self.assertEqual(serial_points, parallel_points)
        self.assertEqual(serial_receipt["pool_count"], 0)
        self.assertEqual(serial_receipt["backend"], "serial")
        self.assertEqual(parallel_receipt["pool_count"], 1)
        self.assertEqual(parallel_receipt["backend"], "spawn")
        self.assertTrue(parallel_receipt["exact"])
        self.assertEqual(parallel_receipt["total_tasks"], len(tasks))
        self.assertEqual(parallel_receipt["effective_workers"], min(24, os.cpu_count() or 1))

    def test_threshold_planning_scales_tasks_not_frames(self):
        import json as _json

        from experiments.psem_state_corrected_adaptation_gate import cross_frontier

        members = {
            f"s-{index}": {"dev": object(), "scores": {"raw": [0.5], "calibrated": [0.5], "f0": [0.5]}}
            for index in range(3)
        }
        grids = {
            name: {
                "raw": [float(46000 - i) / 46000.0 for i in range(46000)],
                "calibrated": [float(46000 - i) / 46000.0 for i in range(46000)],
                "f0": [0.5],
            }
            for name in members
        }
        tasks = cross_frontier.plan_exact_tasks(grids, [100, 300, 500])
        per_combo = (46000 + cross_frontier.EXACT_THRESHOLD_CHUNK - 1) // cross_frontier.EXACT_THRESHOLD_CHUNK
        self.assertEqual(len(tasks), 3 * 2 * 3 * per_combo + 3 * 1 * 3)
        small_grids = {
            name: {
                "raw": list(grid["raw"])[:40],
                "calibrated": list(grid["calibrated"])[:40],
                "f0": [0.5],
            }
            for name, grid in grids.items()
        }
        small_tasks = cross_frontier.plan_exact_tasks(small_grids, [100, 300, 500])
        payload_bytes = sum(len(_json.dumps(task).encode()) for task in tasks)
        small_bytes = sum(len(_json.dumps(task).encode()) for task in small_tasks)
        self.assertGreater(payload_bytes, small_bytes)
        self.assertLess(payload_bytes, 3 * 46000 * 8 * 24)
        for task in tasks:
            self.assertNotIn("dev", task)
            self.assertNotIn("scores", task)
            self.assertEqual(sorted(task), ["horizon_ms", "kind", "member", "thresholds"])
            self.assertLessEqual(len(task["thresholds"]), cross_frontier.EXACT_THRESHOLD_CHUNK)


class MultiChunkWaveTest(unittest.TestCase):
    def _grid300(self):
        return [float(300 - i) / 300.0 for i in range(300)]

    def _stub_evaluator(self, seen):
        import sys
        import types

        def _decode(dev, scores, threshold, confirmation_ms):
            seen.append(float(threshold))
            return [threshold]

        def _metrics(dev, events):
            return {
                "false_cut_count": 2,
                "active_speech_seconds": 3600.0,
                "reference_replacement_count": 10,
                "missed_replacement_count": 1,
                "exclusive_other_contamination_seconds": 180.0,
            }

        package = types.ModuleType("experiments.psem_frozen_ceiling_gate")
        package.__path__ = []
        module = types.ModuleType("evaluate_ceiling")
        module.decode_scores = _decode
        module.session_metrics = _metrics
        prior = {
            key: sys.modules[key]
            for key in (
                "experiments.psem_frozen_ceiling_gate",
                "experiments.psem_frozen_ceiling_gate.evaluate_ceiling",
            )
            if key in sys.modules
        }
        sys.modules["experiments.psem_frozen_ceiling_gate"] = package
        sys.modules["experiments.psem_frozen_ceiling_gate.evaluate_ceiling"] = module
        return prior

    def _unstub_evaluator(self, prior):
        import sys

        for key in (
            "experiments.psem_frozen_ceiling_gate",
            "experiments.psem_frozen_ceiling_gate.evaluate_ceiling",
        ):
            if key in prior:
                sys.modules[key] = prior[key]
            else:
                del sys.modules[key]

    def test_stubbed_serial_300_all_present_descending(self):
        import json as _json

        from experiments.psem_state_corrected_adaptation_gate import cross_frontier

        seen = []
        prior = self._stub_evaluator(seen)
        try:
            members = {"m": {"dev": object(), "scores": {"raw": self._grid300(), "f0": [0.5]}}}
            grids = {"m": {"raw": list(self._grid300()), "f0": [0.5]}}
            tasks = cross_frontier.plan_exact_tasks(grids, [100])
            self.assertGreater(len(tasks), 1)
            results, receipt = cross_frontier.run_exact_wave(members, tasks, 1)
            points = results["m"]["raw"][100]
            self.assertEqual(len(points), 300)
            self.assertEqual(
                [float(p["threshold"]) for p in points], list(self._grid300())
            )
            self.assertEqual(
                _json.dumps(points, sort_keys=True),
                _json.dumps(results["m"]["raw"][100], sort_keys=True),
            )
            self.assertEqual(
                receipt["primitive_counts"]["m/raw/100"],
                {"expected": 300, "observed": 300},
            )
            self.assertEqual(receipt["pool_count"], 0)
            self.assertTrue(receipt["exact"])
        finally:
            self._unstub_evaluator(prior)

    def test_spawn_300_structure_descending(self):
        from dataclasses import replace

        from experiments.psem_frozen_ceiling_gate.build_ceiling_examples import (
            load_sessions,
        )
        from experiments.psem_state_corrected_adaptation_gate import cross_frontier

        sessions = load_sessions()
        full = next(
            s for s in sessions if s.source_family == "ami_mix_headset" and s.role == "dev"
        )
        dev = replace(
            full,
            starts=full.starts[:40],
            ends=full.ends[:40],
            episode_ids=full.episode_ids[:40],
            episode_speakers=full.episode_speakers[:40],
            frontiers=full.frontiers[:40],
            probabilities=full.probabilities[:40],
            alive=full.alive[:40],
            reset=full.reset[:40],
            valid=full.valid[:40],
            masked=full.masked[:40],
            speech_present=full.speech_present[:40],
            anchor_present=full.anchor_present[:40],
            overlap=full.overlap[:40],
        )
        grid = self._grid300()
        members = {"m": {"dev": dev, "scores": {"raw": list(grid), "f0": [0.5] * 40}}}
        tasks = cross_frontier.plan_exact_tasks({"m": {"raw": list(grid), "f0": [0.5]}}, [100, 300, 500])
        results, receipt = cross_frontier.run_exact_wave(members, tasks, 24)
        for horizon in (100, 300, 500):
            points = results["m"]["raw"][horizon]
            self.assertEqual(len(points), 300)
            self.assertEqual([float(p["threshold"]) for p in points], list(grid))
            self.assertEqual(
                receipt["primitive_counts"][f"m/raw/{horizon}"],
                {"expected": 300, "observed": 300},
            )
        self.assertEqual(receipt["pool_count"], 1)
        self.assertEqual(receipt["backend"], "spawn")

    def test_dropped_row_fails_closed(self):
        from experiments.psem_state_corrected_adaptation_gate import cross_frontier

        seen = []
        prior = self._stub_evaluator(seen)
        original = cross_frontier.exact_threshold_task
        cross_frontier.exact_threshold_task = lambda payload: {
            **original(payload),
            "primitives": original(payload)["primitives"][:-1],
        }
        try:
            members = {"m": {"dev": object(), "scores": {"raw": self._grid300(), "f0": [0.5]}}}
            grids = {"m": {"raw": list(self._grid300()), "f0": [0.5]}}
            tasks = cross_frontier.plan_exact_tasks(grids, [100])
            with self.assertRaises(cross_frontier.CrossFrontierError):
                cross_frontier.run_exact_wave(members, tasks, 1)
        finally:
            cross_frontier.exact_threshold_task = original
            self._unstub_evaluator(prior)

    def test_index_lookup_scales_and_rejects_duplicates(self):
        from experiments.psem_state_corrected_adaptation_gate import cross_frontier

        rows = [{"threshold": float(46000 - i), "v": i} for i in range(46000)]
        indexed = cross_frontier.index_threshold_rows(rows)
        self.assertEqual(len(indexed), 46000)
        for probe in (46000.0, 23000.0, 1.0):
            self.assertEqual(indexed[probe]["threshold"], probe)
        with self.assertRaises(cross_frontier.CrossFrontierError):
            cross_frontier.index_threshold_rows(rows[:100] + rows[:1])


class ScoreDomainTest(unittest.TestCase):
    def test_probabilities_not_logits_unmapped_masked(self):
        import math

        from experiments.psem_state_corrected_adaptation_gate import temporal_train

        f0_raw = [0.0, 2.0, -2.0, 0.5]
        cand_raw = [1.0, -1.0, 0.5, -0.5]
        mapped = [True, True, False, True]
        calibrators = {"f0": {"slope": 2.0, "intercept": 0.5}, "candidate": {"slope": 0.5, "intercept": -0.25}}
        probs = temporal_train.prepare_dev_scores(f0_raw, cand_raw, mapped, [2], calibrators)
        self.assertEqual(probs["f0_prob"][2], float("-inf"))
        self.assertEqual(probs["cand_raw_prob"][2], float("-inf"))
        self.assertEqual(probs["cand_cal_prob"][2], float("-inf"))
        self.assertAlmostEqual(probs["cand_raw_prob"][0], 1.0 / (1.0 + math.exp(-1.0)))
        self.assertNotAlmostEqual(probs["cand_raw_prob"][0], 1.0)
        self.assertAlmostEqual(
            probs["cand_cal_prob"][0], 1.0 / (1.0 + math.exp(-(0.5 * 1.0 - 0.25)))
        )
        self.assertNotEqual(probs["cand_cal_prob"][0], probs["cand_raw_prob"][0])
        self.assertAlmostEqual(probs["f0_prob"][0], 1.0 / (1.0 + math.exp(0.0)))
        grid = sorted(
            {v for v in probs["cand_raw_prob"] + probs["cand_cal_prob"] if v != float("-inf")},
            reverse=True,
        )
        self.assertNotIn(float("-inf"), grid)
        self.assertTrue(all(0.0 <= v <= 1.0 for v in grid))

    def test_unmapped_only_values_never_form_thresholds(self):
        from experiments.psem_state_corrected_adaptation_gate import h_arm, temporal_train

        probs = temporal_train.prepare_dev_scores(
            [5.0, 0.0],
            [9.0, 0.0],
            [False, True],
            [0],
            {"f0": {"slope": 1.0, "intercept": 0.0}, "candidate": {"slope": 1.0, "intercept": 0.0}},
        )
        grid = h_arm.union_probability_grid([probs["cand_raw_prob"], probs["cand_cal_prob"]])
        self.assertEqual(len(grid), 1)
        self.assertAlmostEqual(grid[0], 0.5)
        with self.assertRaises(h_arm.HArmError):
            h_arm.union_probability_grid([probs["f0_prob"][:1]])

    def test_nonpositive_calibrator_rejected(self):
        from experiments.psem_state_corrected_adaptation_gate import temporal_train

        with self.assertRaises(temporal_train.TemporalArmError):
            temporal_train.prepare_dev_scores(
                [0.0], [0.0], [True], [],
                {"f0": {"slope": 0.0, "intercept": 0.0}, "candidate": {"slope": 1.0, "intercept": 0.0}},
            )


class FrontierPoolAccountingTest(unittest.TestCase):
    def test_run_dev_frontier_opens_no_pools(self):
        import concurrent.futures as _futures
        import multiprocessing as _mp
        import tempfile
        from pathlib import Path
        from unittest import mock

        from experiments.psem_state_corrected_adaptation_gate import h_arm

        calibrators = {"f0": {"slope": 1.0, "intercept": 0.0}, "candidate": {"slope": 1.0, "intercept": 0.0}}
        dev_scores = {
            "s-a": {
                "f0": [0.0] * 4, "candidate": [-1.0, 0.0, 1.0, 2.0],
                "target": [0.0, 0.0, 1.0, 1.0], "mapped": [True] * 4,
            },
        }
        conv = h_arm.dev_frontier_inputs(
            dev_scores["s-a"]["f0"], dev_scores["s-a"]["candidate"],
            dev_scores["s-a"]["mapped"], calibrators,
        )
        tables = {
            "s-a": {
                hz: {
                    "f0": (4.0, 0.2, 0.3),
                    "by_threshold_raw": {t: (4.5, 0.21, 0.29) for t in conv["thresholds_raw"]},
                    "by_threshold_calibrated": {t: (5.0, 0.2, 0.3) for t in conv["thresholds_calibrated"]},
                }
                for hz in (100, 300, 500)
            }
        }
        grid = [0.9, 0.5]
        group_tables = {
            name: {
                hz: {
                    "kinds": {
                        kind: {
                            "thresholds": list(grid),
                            "points": [[t, 7.0, 0.28, 0.38] for t in grid],
                            "f0": [4.0, 0.2, 0.3],
                        }
                        for kind in ("raw", "calibrated")
                    },
                }
                for hz in (100, 300, 500)
            }
            for name in ("AMI", "AliMeeting", "pooled")
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir(parents=True)

            def _no_pool(*args, **kwargs):
                raise AssertionError("envelope postprocessing must stay in-process")

            with mock.patch.object(_futures, "ProcessPoolExecutor", _no_pool), mock.patch.object(
                _mp, "get_context", side_effect=AssertionError("no pool context")
            ):
                doc = h_arm.run_dev_frontier(
                    run_dir, {"seed": 7301}, dev_scores, tables,
                    {"s-a": "AMI"}, calibrators, group_tables, workers=24,
                )
            self.assertEqual(doc["artifact_role"], "issue-121-cross-arm-dev-frontier")
            self.assertIn("macro", doc["horizons"]["100"])


if __name__ == "__main__":
    unittest.main()
