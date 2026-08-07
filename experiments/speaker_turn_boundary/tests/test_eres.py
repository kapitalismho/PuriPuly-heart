from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.speaker_turn_boundary.adapters.eres2netv2 import (
    EresAdjacentProfile,
    EresStableAnchorProfile,
    cosine_similarity,
    eres_adjacent_profiles,
    eres_anchor_profiles,
    eres_profile_matrix,
    kaldi_fbank_numpy,
    threshold_range,
)


class _FakeRunner:
    def __init__(self, embed_fn) -> None:
        self._embed_fn = embed_fn
        self.calls: list[tuple[int, int]] = []
        self._audio_epoch = 0

    def embed_cached(self, samples: np.ndarray, start: int, end: int) -> np.ndarray:
        self.calls.append((int(start), int(end)))
        return self._embed_fn(samples, start, end)

    def _fake_embed(self, samples: np.ndarray) -> np.ndarray:
        window = int(samples.size)
        return np.full(4, float(window) / 16000.0, dtype=np.float32)


def test_eres_artifact_filename_keeps_json_suffix(tmp_path) -> None:
    from experiments.speaker_turn_boundary.run_eres_sweep import (
        eres_artifact_filename,
    )

    filename = eres_artifact_filename(
        "phase1_dev",
        "E-standard",
        "adjacent",
        "adjacent-W0.5-s0.10-thr0.30-c1",
    )
    assert filename.endswith(".json")
    assert "pjson" not in filename
    assert Path(filename).suffix == ".json"
    assert filename.count(".") == 1
    payload = {"profile_id": "adjacent-W0.5-s0.10-thr0.30-c1"}
    artifact = tmp_path / filename
    artifact.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    assert json.loads(artifact.read_text(encoding="utf-8")) == payload


def test_profile_matrix_matches_issue():
    matrix = dict(eres_profile_matrix())
    assert matrix[0.50] == [0.10, 0.25]
    assert matrix[0.75] == [0.10, 0.25]
    assert matrix[1.00] == [0.10, 0.25, 0.50]
    assert matrix[1.50] == [0.25, 0.50]
    assert matrix[2.00] == [0.50]
    total_windows = sum(len(steps) for steps in matrix.values())
    assert total_windows == 10
    assert threshold_range() == [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def test_adjacent_profile_expansion():
    profiles = eres_adjacent_profiles()
    assert len(profiles) == 10 * 9 * 2
    identifiers = {profile.profile_id for profile in profiles}
    assert len(identifiers) == len(profiles)


def test_anchor_profile_expansion():
    profiles = eres_anchor_profiles(windows=(0.50, 0.75, 1.00, 1.50))
    assert len(profiles) == 4 * 2 * 9 * 2 * 2
    identifiers = {profile.profile_id for profile in profiles}
    assert len(identifiers) == len(profiles)


def test_cosine_similarity():
    left = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    right = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    assert cosine_similarity(left, right) == pytest.approx(0.0)
    assert cosine_similarity(left, left) == pytest.approx(1.0)
    assert cosine_similarity(np.zeros(3), right) == pytest.approx(0.0)


def test_fbank_deterministic_shape():
    rng = np.random.default_rng(1)
    samples = rng.normal(0, 0.1, 16000).astype(np.float32)
    first = kaldi_fbank_numpy(samples)
    second = kaldi_fbank_numpy(samples)
    assert first.shape == (98, 80)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()


def test_fbank_short_input():
    assert kaldi_fbank_numpy(np.zeros(100, dtype=np.float32)).shape == (0, 80)


def _run_adjacent(runtime, samples, profile, utterance):
    from experiments.speaker_turn_boundary.run_eres_sweep import _adjacent_builder

    builder = _adjacent_builder(runtime, profile, 0)
    boundaries, progress = builder(samples, utterance)
    return boundaries, progress


def test_adjacent_skip_insufficient_speech():
    profile = EresAdjacentProfile(
        window_seconds=1.0, step_seconds=0.25, threshold=0.5, confirmation=1
    )
    samples = np.zeros(16000 * 3, dtype=np.float32)
    boundaries, progress = _run_adjacent(
        _FakeRunner(lambda s, a, b: s[a:b]), samples, profile, (0, 16000)
    )
    assert boundaries == []
    assert progress == []


def test_adjacent_confirmation_semantics():
    samples = np.zeros(16000 * 3, dtype=np.float32)
    profile = EresAdjacentProfile(
        window_seconds=0.5, step_seconds=0.25, threshold=0.5, confirmation=2
    )
    runtime = _FakeRunner(lambda s, a, b: s[a:b])
    boundaries, _ = _run_adjacent(runtime, samples, profile, (0, 48000))
    assert len(boundaries) >= 1
    assert boundaries[0].boundary_sample == 8000
    assert boundaries[0].observed_sample == 8000 + 8000
    assert boundaries[0].confidence == pytest.approx(1.0 - 0.0)
    assert all(b.observed_sample == b.boundary_sample + 8000 for b in boundaries)


def test_adjacent_window_positioning():
    samples = np.zeros(16000 * 4, dtype=np.float32)
    profile = EresAdjacentProfile(
        window_seconds=1.0, step_seconds=0.5, threshold=0.5, confirmation=1
    )
    runtime = _FakeRunner(lambda s, a, b: s[a:b])
    boundaries, _ = _run_adjacent(runtime, samples, profile, (16000, 48000))
    assert boundaries[0].boundary_sample == 16000 + 16000
    assert boundaries[0].observed_sample == 16000 + 16000 + 16000


def test_anchor_lifecycle_and_promotion():
    samples = np.zeros(16000 * 5, dtype=np.float32)
    profile = EresStableAnchorProfile(
        window_seconds=0.5,
        step_seconds=0.25,
        threshold=0.5,
        confirmation=1,
        mutual_similarity_threshold=0.5,
        anchor_update="none",
    )
    from experiments.speaker_turn_boundary.run_eres_sweep import _anchor_builder

    runtime = _FakeRunner(lambda s, a, b: s[a:b])
    builder = _anchor_builder(runtime, profile, 0)
    boundaries, _ = builder(samples, (0, 80000))
    assert len(boundaries) >= 1
    assert boundaries[0].boundary_sample == 8000


def test_anchor_mutual_similarity_rule():
    profile = EresStableAnchorProfile(
        window_seconds=0.5,
        step_seconds=0.25,
        threshold=0.5,
        confirmation=2,
        mutual_similarity_threshold=0.5,
        anchor_update="none",
    )

    class _ConstrainedRuntime:
        def __init__(self) -> None:
            self.calls = 0
            self._audio_epoch = 0
            self._vectors = [
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
                np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            ]

        def embed_cached(self, samples: np.ndarray, start: int, end: int) -> np.ndarray:
            self.calls += 1
            return self._vectors[(self.calls - 1) % len(self._vectors)]

    from experiments.speaker_turn_boundary.run_eres_sweep import _anchor_builder

    samples = np.zeros(16000 * 5, dtype=np.float32)
    boundaries, _ = _anchor_builder(_ConstrainedRuntime(), profile, 0)(samples, (0, 80000))
    assert boundaries == []


def test_anchor_vad_reset_reinitializes():
    from experiments.speaker_turn_boundary.run_eres_sweep import _anchor_builder

    profile = EresStableAnchorProfile(
        window_seconds=0.5,
        step_seconds=0.25,
        threshold=0.5,
        confirmation=1,
        mutual_similarity_threshold=0.5,
        anchor_update="none",
    )
    samples = np.zeros(16000 * 8, dtype=np.float32)
    runtime = _FakeRunner(lambda s, a, b: s[a:b])
    builder = _anchor_builder(runtime, profile, 0)
    first, _ = builder(samples, (0, 40000))
    second, _ = builder(samples, (48000, 80000))
    assert first[0].boundary_sample == 8000
    assert second[0].boundary_sample == 48000 + 8000


def test_eres_progress_epoch_scoped_across_reused_builder_and_cache():
    from experiments.speaker_turn_boundary.run_eres_sweep import (
        EresDetectorRunner,
        _adjacent_builder,
        _anchor_builder,
    )

    class _FakeRuntime:
        def embed(self, samples: np.ndarray) -> np.ndarray:
            return np.full(4, 0.5, dtype=np.float32)

    samples = np.zeros(16000 * 4, dtype=np.float32)
    profiles = [
        (
            "adjacent",
            _adjacent_builder,
            EresAdjacentProfile(
                window_seconds=0.5, step_seconds=0.25, threshold=0.5, confirmation=1
            ),
        ),
        (
            "anchor",
            _anchor_builder,
            EresStableAnchorProfile(
                window_seconds=0.5,
                step_seconds=0.25,
                threshold=0.5,
                confirmation=1,
                mutual_similarity_threshold=0.5,
                anchor_update="none",
            ),
        ),
    ]
    for kind, factory, profile in profiles:
        runner = EresDetectorRunner(_FakeRuntime(), "E-standard")
        builder = factory(runner, profile, 0)
        for epoch in (0, 1):
            runner.start_epoch(epoch)
            _, progress = runner.run_case(samples, [(0, samples.size)], builder)
            assert progress, f"{kind} epoch {epoch}: expected progress snapshots"
            assert {snapshot.audio_epoch for snapshot in progress} == {epoch}, (
                f"{kind} epoch {epoch}: progress records leaked a foreign epoch "
                f"(got {sorted({snapshot.audio_epoch for snapshot in progress})})"
            )
        assert len(runner._embedding_cache) > 0, "cached/reused embedding path not exercised"


def test_eres_profile_validation():
    with pytest.raises(ValueError):
        EresAdjacentProfile(window_seconds=1.0, step_seconds=0.25, threshold=1.5, confirmation=1)
    with pytest.raises(ValueError):
        EresStableAnchorProfile(
            window_seconds=0.5,
            step_seconds=0.25,
            threshold=0.5,
            confirmation=1,
            mutual_similarity_threshold=0.5,
            anchor_update="unknown",
        )
