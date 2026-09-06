from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from experiments.psem_state_corrected_adaptation_gate import frontier as frontier_mod
from experiments.psem_state_corrected_adaptation_gate import frontier_sweep as sweep_mod
from experiments.psem_state_corrected_adaptation_gate.material import _frontier_point
from experiments.psem_frozen_ceiling_gate.evaluate_ceiling import (
    decode_scores,
    session_metrics,
)
from experiments.psem_frozen_ceiling_gate.experiment_support import ReplacementEvent


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


def _real_dev():
    from experiments.psem_frozen_ceiling_gate.build_ceiling_examples import (
        load_sessions,
    )

    sessions = load_sessions()
    return next(
        s
        for s in sessions
        if s.source_family == "ami_mix_headset" and s.role == "dev"
    )


def _brute(dev, scores, horizon_ms):
    thresholds = frontier_mod.unique_thresholds(scores)
    points = []
    for threshold in thresholds:
        events = decode_scores(
            dev, np.asarray(scores), threshold=threshold, confirmation_ms=horizon_ms
        )
        points.append(_frontier_point(session_metrics(dev, events), threshold))
    return thresholds, points


def _swept(dev, scores, horizon_ms):
    from experiments.psem_state_corrected_adaptation_gate.material import (
        _sweep_points,
    )

    thresholds = frontier_mod.unique_thresholds(scores)
    return thresholds, _sweep_points(dev, scores, thresholds, horizon_ms)


def _assert_same(test, first, second):
    test.assertEqual(len(first), len(second))
    for a, b in zip(first, second):
        test.assertEqual(a.threshold, b.threshold)
        test.assertEqual(a.false_cuts_per_hour, b.false_cuts_per_hour)
        test.assertEqual(a.contamination, b.contamination)
        test.assertEqual(a.miss_rate, b.miss_rate)


class SweepRejectionTest(unittest.TestCase):
    def test_nan_and_posinf_rejected(self):
        with self.assertRaises(ValueError):
            sweep_mod.sweep_threshold_events(None, [0.1, float("nan")], 100)
        with self.assertRaises(ValueError):
            sweep_mod.sweep_threshold_events(None, [0.1, float("inf")], 100)


def _micro_session(frames_per_episode=30, valid=None, masked=None, speech=None):
    from experiments.psem_frozen_ceiling_gate.experiment_support import (
        AnchorEpisode,
        GTSessionResult,
    )

    frames = frames_per_episode * 2
    starts = np.arange(frames, dtype=np.int64) * 1280
    total = frames * 1280
    half = frames_per_episode * 1280
    return SimpleNamespace(
        source_id="micro",
        starts=starts,
        ends=starts + 1280,
        episode_ids=np.array(["e1"] * frames_per_episode + ["e2"] * frames_per_episode),
        episode_speakers=np.array(["spkA"] * frames_per_episode + ["spkB"] * frames_per_episode),
        valid=np.ones(frames, dtype=bool) if valid is None else np.array(valid, dtype=bool),
        masked=np.zeros(frames, dtype=bool) if masked is None else np.array(masked, dtype=bool),
        speech_present=np.ones(frames, dtype=bool) if speech is None else np.array(speech, dtype=bool),
        frontiers=starts + 1280,
        anchor_present=np.ones(frames, dtype=bool),
        reference=GTSessionResult(
            source_id="micro",
            confirmation_samples=1600,
            enrollment_samples=0,
            silence_reset_samples=0,
            events=(
                ReplacementEvent("micro", "e1", "spkA", 5000, 6400, 8000, None, 1600),
            ),
            episodes=(
                AnchorEpisode("e1", "micro", "spkA", 0, 1280, half, None),
                AnchorEpisode("e2", "micro", "spkB", half, half + 1280, total, None),
            ),
        ),
        manifest={
            "intervals": [
                {"start_sample": 0, "end_sample": half, "active_speakers": ["spkA"]},
                {"start_sample": half, "end_sample": total, "active_speakers": ["spkB"]},
            ],
            "topology_windows": [
                {"start_sample": 0, "end_sample": half, "primary_topology": "solo"},
                {"start_sample": half, "end_sample": total, "primary_topology": "solo"},
            ],
        },
    )


class SweepMicroParityTest(unittest.TestCase):
    def test_basic_two_episodes(self):
        dev = _micro_session()
        rng = np.random.RandomState(7301)
        scores = rng.rand(60).tolist()
        for horizon_ms in (99, 100, 101):
            _, expected = _brute(dev, scores, horizon_ms)
            _, got = _swept(dev, scores, horizon_ms)
            _assert_same(self, expected, got)

    def test_equal_scores(self):
        dev = _micro_session()
        scores = [0.5] * 60
        for horizon_ms in (100, 300):
            _, expected = _brute(dev, scores, horizon_ms)
            _, got = _swept(dev, scores, horizon_ms)
            _assert_same(self, expected, got)

    def test_silence_and_mask_transparency(self):
        dev = _micro_session(
            speech=[i % 3 != 0 for i in range(60)],
            masked=[i % 7 == 0 for i in range(60)],
        )
        rng = np.random.RandomState(7302)
        scores = rng.rand(60).tolist()
        for horizon_ms in (100, 500):
            _, expected = _brute(dev, scores, horizon_ms)
            _, got = _swept(dev, scores, horizon_ms)
            _assert_same(self, expected, got)

    def test_invalid_reset(self):
        dev = _micro_session(valid=[i < 20 or i >= 26 for i in range(60)])
        scores = [0.9] * 60
        _, expected = _brute(dev, scores, 100)
        _, got = _swept(dev, scores, 100)
        _assert_same(self, expected, got)

    def test_inf_floor(self):
        dev = _micro_session()
        scores = [0.9] * 60
        scores[10] = float("-inf")
        _, expected = _brute(dev, scores, 100)
        _, got = _swept(dev, scores, 100)
        _assert_same(self, expected, got)

    def test_random_grids(self):
        for seed in (7301, 7302, 7303):
            rng = np.random.RandomState(seed)
            count = 80
            episodes = [f"ep{i // 20}" for i in range(count)]
            starts = np.arange(count, dtype=np.int64) * 1280
            total = count * 1280
            dev = SimpleNamespace(
                source_id="rand",
                starts=starts,
                ends=starts + 1280,
                episode_ids=np.array(episodes),
                episode_speakers=np.array(["spk"] * count),
                valid=np.array(list(rng.rand(count) < 0.9)),
                masked=np.array(list(rng.rand(count) < 0.1)),
                speech_present=np.array(list(rng.rand(count) < 0.85)),
                frontiers=starts + 1280,
                anchor_present=np.array(list(rng.rand(count) < 0.5)),
                reference=_micro_session(count // 2).reference,
                manifest=_micro_session(count // 2).manifest,
            )
            _ = total
            scores = rng.rand(count).tolist()
            for horizon_ms in (99, 100, 101, 300):
                _, expected = _brute(dev, scores, horizon_ms)
                _, got = _swept(dev, scores, horizon_ms)
                _assert_same(self, expected, got)

    def test_envelope_equality(self):
        dev = _micro_session()
        rng = np.random.RandomState(7301)
        scores = rng.rand(60).tolist()
        _, brute_points = _brute(dev, scores, 100)
        _, swept_points = _swept(dev, scores, 100)
        f0 = brute_points[-1]
        brute_env = frontier_mod.select_envelopes(f0, brute_points)
        swept_env = frontier_mod.select_envelopes(f0, swept_points)
        self.assertEqual(brute_env, swept_env)


@unittest.skipUnless(_snapshot_available(), "sweep parity requires frozen snapshot")
class SweepRealSessionTest(unittest.TestCase):
    def _dev(self, count=2000):
        full = _real_dev()
        return replace(
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

    def test_truncated_full_grid_parity(self):
        from experiments.psem_state_corrected_adaptation_gate.material import (
            candidate_frontier_points,
        )

        dev = self._dev(2000)
        rng = np.random.RandomState(7301)
        levels = rng.rand(60)
        scores = rng.choice(levels, 2000).tolist()
        for horizon_ms in (100, 300, 500):
            _, expected = _brute(dev, scores, horizon_ms)
            got = candidate_frontier_points(dev, scores, frontier_mod.unique_thresholds(scores), horizon_ms, 1)
            _assert_same(self, expected, got)


    def test_workers1_matches_workers8(self):
        from experiments.psem_state_corrected_adaptation_gate.material import (
            candidate_frontier_points_multi,
        )
        dev = self._dev(2000)
        rng = np.random.RandomState(7301)
        levels = rng.rand(60)
        scores = rng.choice(levels, 2000).tolist()
        grid = frontier_mod.unique_thresholds(scores)
        horizons = [100, 300, 500]
        sequential = candidate_frontier_points_multi(dev, scores, grid, horizons, 1)
        parallel = candidate_frontier_points_multi(dev, scores, grid, horizons, 8)
        for horizon_ms in horizons:
            _assert_same(self, sequential[horizon_ms], parallel[horizon_ms])
    def test_stratified_full_session_parity(self):
        from experiments.psem_frozen_ceiling_gate.evaluate_ceiling import (
            decode_scores,
            session_metrics,
        )
        from experiments.psem_frozen_ceiling_gate.experiment_support import (
            ReplacementEvent,
        )
        from experiments.psem_state_corrected_adaptation_gate import (
            frontier_sweep as sweep_mod,
        )
        from experiments.psem_state_corrected_adaptation_gate.material import (
            _frontier_point,
        )

        dev = _real_dev()
        rng = np.random.RandomState(7301)
        scores = rng.rand(len(dev.starts)).tolist()
        scored = np.asarray(scores)
        for horizon_ms in (100, 500):
            grid, keys = sweep_mod.sweep_threshold_events(dev, scores, horizon_ms)
            stratum = sorted(set([grid[0], grid[-1]] + grid[::4000]), reverse=True)
            by_threshold = dict(zip(grid, keys))
            for threshold in stratum:
                events = decode_scores(
                    dev, scored, threshold=threshold, confirmation_ms=horizon_ms
                )
                expected = _frontier_point(session_metrics(dev, events), threshold)
                rebuilt = tuple(
                    ReplacementEvent(
                        source_id=e[0],
                        anchor_episode_id=e[1],
                        anchor_id=e[2],
                        boundary_source_sample=e[3],
                        model_evidence_frontier_sample=e[4],
                        decoder_emit_sample=e[5],
                        compute_lag_ms=None,
                        confirmation_samples=e[6],
                    )
                    for e in by_threshold[threshold]
                )
                got = _frontier_point(session_metrics(dev, rebuilt), threshold)
                _assert_same(self, [expected], [got])


    def test_perf_reduction(self):
        import time

        from experiments.psem_frozen_ceiling_gate.evaluate_ceiling import (
            decode_scores,
        )
        from experiments.psem_state_corrected_adaptation_gate import (
            frontier_sweep as sweep_mod,
        )

        dev = self._dev(2000)
        rng = np.random.RandomState(7301)
        scores = rng.rand(2000).tolist()
        grid = frontier_mod.unique_thresholds(scores)
        scored = np.asarray(scores)
        start = time.perf_counter()
        for threshold in grid:
            decode_scores(dev, scored, threshold=threshold, confirmation_ms=100)
        brute_seconds = time.perf_counter() - start
        start = time.perf_counter()
        _, keys = sweep_mod.sweep_threshold_events(dev, scores, 100)
        sweep_seconds = time.perf_counter() - start
        self.assertEqual(len(keys), len(grid))
        self.assertLess(sweep_seconds, 0.5 * brute_seconds)
        rep_levels = rng.rand(20)
        rep_scores = rng.choice(rep_levels, 2000).tolist()
        _, rep_keys = sweep_mod.sweep_threshold_events(dev, rep_scores, 100)
        self.assertLessEqual(len(set(rep_keys)), 60)


if __name__ == "__main__":
    unittest.main()
