from __future__ import annotations

import importlib.util
import unittest
from types import SimpleNamespace

import numpy as np

from experiments.psem_state_corrected_adaptation_gate.material import (
    build_all_source_targets,
    candidate_frontier_points,
    is_dev_family_session,
    resolve_worker_count,
)
from experiments.psem_training_strategy_gate.data.label_contract import (
    CanonicalInterval,
    LabelResult,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None


def _snapshot_available() -> bool:
    try:
        from experiments.psem_frozen_ceiling_gate.build_ceiling_examples import (
            load_sessions,
        )
    except Exception:
        return False
    try:
        sessions = load_sessions()
    except Exception:
        return False
    return any(
        s.source_family == "ami_mix_headset" and s.role == "dev" for s in sessions
    )


def _synthetic_labels() -> LabelResult:
    intervals = (
        CanonicalInterval(0, 25600, ("spkA",)),
        CanonicalInterval(25600, 51200, ("spkB",)),
    )
    activity = ({"mask_state": "valid"}, {"mask_state": "valid"})
    return LabelResult(
        "psem-handoff-v1", "0" * 64, 16000, intervals, activity, (), (), {}
    )


def _synthetic_sessions() -> dict[str, SimpleNamespace]:
    rows = [{"window_start_sample": 0, "window_end_sample": 51200}]
    return {
        "src-a": SimpleNamespace(labels=_synthetic_labels(), rows=rows),
        "src-b": SimpleNamespace(labels=_synthetic_labels(), rows=rows),
    }


class DevRoleFilterTest(unittest.TestCase):
    def test_lowercase_snapshot_roles(self):
        sessions = [
            SimpleNamespace(
                source_family="ami_mix_headset", role="dev", source_id="a"
            ),
            SimpleNamespace(
                source_family="ami_mix_headset", role="eval", source_id="b"
            ),
            SimpleNamespace(
                source_family="alimeeting_far_ch0", role="dev", source_id="c"
            ),
            SimpleNamespace(source_family="other", role="dev", source_id="d"),
        ]
        ami = [
            s.source_id
            for s in sessions
            if is_dev_family_session(s, "ami_mix_headset")
        ]
        ali = [
            s.source_id
            for s in sessions
            if is_dev_family_session(s, "alimeeting_far_ch0")
        ]
        self.assertEqual(ami, ["a"])
        self.assertEqual(ali, ["c"])


class WorkerCountTest(unittest.TestCase):
    def test_bounds(self):
        import os

        cap = max(1, min(24, os.cpu_count() or 1))
        self.assertEqual(resolve_worker_count(1), 1)
        self.assertEqual(resolve_worker_count(10**6), cap)
        self.assertEqual(resolve_worker_count(24), min(24, cap))
        self.assertTrue(1 <= resolve_worker_count(None) <= cap)
        self.assertTrue(1 <= resolve_worker_count(0) <= cap)
        self.assertLessEqual(resolve_worker_count(4), cap)


@unittest.skipUnless(HAS_TORCH, "target construction requires torch runtime")
class TargetParityTest(unittest.TestCase):
    def test_sequential_matches_parallel(self):
        sessions = _synthetic_sessions()
        rows_by_source = {
            sid: list(entry.rows) for sid, entry in sessions.items()
        }
        plain = {
            sid: SimpleNamespace(labels=entry.labels)
            for sid, entry in sessions.items()
        }
        sequential = build_all_source_targets(plain, rows_by_source, 1)
        parallel = build_all_source_targets(plain, rows_by_source, 3)
        self.assertEqual(sorted(sequential.keys()), ["src-a", "src-b"])
        self.assertEqual(sorted(parallel.keys()), ["src-a", "src-b"])
        for source_id in ("src-a", "src-b"):
            first, second = sequential[source_id], parallel[source_id]
            self.assertEqual(first["authority"], second["authority"])
            self.assertEqual(first["multiplicity"], second["multiplicity"])
            self.assertEqual(first["episode_ids"], second["episode_ids"])
            self.assertEqual(first["intervals"], second["intervals"])
        self.assertTrue(sum(sequential["src-a"]["multiplicity"]) > 0)


@unittest.skipUnless(_snapshot_available(), "frontier parity requires frozen snapshot")
class FrontierParityTest(unittest.TestCase):
    def test_threshold_points_match_and_order(self):
        from experiments.psem_frozen_ceiling_gate.build_ceiling_examples import (
            load_sessions,
        )

        sessions = load_sessions()
        dev = next(
            s
            for s in sessions
            if s.source_family == "ami_mix_headset" and s.role == "dev"
        )
        rng = np.random.RandomState(7301)
        scores = rng.rand(len(dev.starts)).tolist()
        thresholds = sorted(set(scores[::6000]), reverse=True)[:8]
        sequential = candidate_frontier_points(dev, scores, thresholds, 100, 1)
        parallel = candidate_frontier_points(dev, scores, thresholds, 100, 3)
        self.assertEqual(len(sequential), len(thresholds))
        self.assertEqual(
            [p.threshold for p in parallel], [p.threshold for p in sequential]
        )
        for first, second in zip(sequential, parallel):
            self.assertEqual(first.threshold, second.threshold)
            self.assertEqual(
                first.false_cuts_per_hour, second.false_cuts_per_hour
            )
            self.assertEqual(first.contamination, second.contamination)
            self.assertEqual(first.miss_rate, second.miss_rate)
        ordered = [p.threshold for p in parallel]
        self.assertEqual(ordered, sorted(ordered, reverse=True))

@unittest.skipUnless(_snapshot_available(), "frontier parity requires frozen snapshot")
class MultiHorizonParityTest(unittest.TestCase):
    def _fixture(self):
        from dataclasses import replace

        from experiments.psem_frozen_ceiling_gate.build_ceiling_examples import (
            load_sessions,
        )

        sessions = load_sessions()
        full = next(
            s
            for s in sessions
            if s.source_family == "ami_mix_headset" and s.role == "dev"
        )
        count = 2000
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
        thresholds = sorted(set(scores), reverse=True)
        return dev, scores, thresholds

    def test_workers1_matches_workers24(self):
        from experiments.psem_state_corrected_adaptation_gate.material import (
            candidate_frontier_points_multi,
        )

        dev, scores, thresholds = self._fixture()
        horizons = [100, 300, 500]
        sequential = candidate_frontier_points_multi(dev, scores, thresholds, horizons, 1)
        parallel = candidate_frontier_points_multi(dev, scores, thresholds, horizons, 24)
        self.assertEqual(sorted(sequential.keys()), horizons)
        self.assertEqual(sorted(parallel.keys()), horizons)
        for horizon_ms in horizons:
            first, second = sequential[horizon_ms], parallel[horizon_ms]
            self.assertEqual(len(first), len(second))
            for a, b in zip(first, second):
                self.assertEqual(a.threshold, b.threshold)
                self.assertEqual(a.false_cuts_per_hour, b.false_cuts_per_hour)
                self.assertEqual(a.contamination, b.contamination)
                self.assertEqual(a.miss_rate, b.miss_rate)


class WorkerThreadCapTest(unittest.TestCase):
    def test_initializer_forces_single_thread_without_torch(self):
        import os
        import sys
        from unittest import mock

        from experiments.psem_state_corrected_adaptation_gate.material import (
            WORKER_THREAD_VARS,
            _init_cpu_worker,
        )

        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.dict(sys.modules):
                sys.modules.pop("torch", None)
                _init_cpu_worker()
                self.assertNotIn("torch", sys.modules)
            for variable in WORKER_THREAD_VARS:
                self.assertEqual(os.environ[variable], "1")




def _parity_square(payload):
    index = int(payload["index"])
    return {"index": index, "value": index * index}


class PoolStreamingTest(unittest.TestCase):
    def test_one_pool_bounded_streaming_ordered(self):
        import concurrent.futures
        from unittest import mock

        from experiments.psem_state_corrected_adaptation_gate.material import (
            _ordered_pool_map,
        )

        created = []
        seen = []
        peaks = []
        real_wait = concurrent.futures.wait

        class CountingPool:
            def __init__(self, *args, **kwargs):
                created.append(kwargs)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def submit(self, fn, payload):
                future = concurrent.futures.Future()
                try:
                    future.set_result(fn(payload))
                except Exception as exc:
                    future.set_exception(exc)
                return future

        def spy_wait(pending, **kwargs):
            peaks.append(len(list(pending)))
            return real_wait(pending, **kwargs)

        payloads = [{"index": i} for i in range(64)]
        with mock.patch(
            "concurrent.futures.ProcessPoolExecutor", CountingPool
        ), mock.patch("concurrent.futures.wait", spy_wait):
            results = _ordered_pool_map(
                _parity_square, payloads, 24, on_result=seen.append
            )
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0].get("max_workers"), 24)
        self.assertTrue(peaks)
        self.assertLessEqual(max(peaks), 24)
        self.assertEqual(len(seen), 64)
        self.assertTrue(all(result is True for result in results))
        self.assertEqual(sorted(r["index"] for r in seen), list(range(64)))

    def test_ordered_results_without_callback(self):
        import concurrent.futures
        from unittest import mock

        from experiments.psem_state_corrected_adaptation_gate.material import (
            _ordered_pool_map,
        )

        created = []

        class CountingPool:
            def __init__(self, *args, **kwargs):
                created.append(kwargs)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def submit(self, fn, payload):
                future = concurrent.futures.Future()
                try:
                    future.set_result(fn(payload))
                except Exception as exc:
                    future.set_exception(exc)
                return future

        payloads = [{"index": i} for i in range(64)]
        with mock.patch(
            "concurrent.futures.ProcessPoolExecutor", CountingPool
        ):
            results = _ordered_pool_map(_parity_square, payloads, 24)
        self.assertEqual(len(created), 1)
        self.assertEqual([r["index"] for r in results], list(range(64)))
        self.assertEqual(
            [r["value"] for r in results], [i * i for i in range(64)]
        )
if __name__ == "__main__":
    unittest.main()
