from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.speaker_turn_boundary.turn_episode import phase4_signal
from experiments.speaker_turn_boundary.turn_episode.phase4_signal import (
    ACOUSTIC_WINDOWS_SHA256,
    COORDINATE_ROWS_SHA256,
    EMBEDDING_WINDOWS_SHA256,
    AudioSource,
    LSCaptureEpoch,
    Phase4Inputs,
    _advance_anchor_state,
    _eres_state_trace,
    _eres_trace_comparison,
    _ls_profile_trace,
    _scored_trace,
    atomic_write_json,
    auc,
    deterministic_gzip,
    eer,
    load_eres_embeddings,
    load_inputs,
    read_json,
    run_eres_cache,
    run_ls_cache,
    save_eres_embeddings,
    signal_registry,
)
from experiments.speaker_turn_boundary.turn_episode.verify_phase4_signal import (
    MUTATIONS,
    _reference_eres_comparison,
    _reference_eres_state_trace,
    _reference_ls_trace,
    _reference_scored_trace,
    expected_registry,
    independent_auc,
    independent_eer,
    load_input_contract,
    verify_mutation_fixture,
)


@pytest.fixture(scope="module")
def phase4_inputs():
    experiment_dir = Path(__file__).resolve().parents[1]
    return load_inputs(experiment_dir)


def test_phase4_inputs_recompute_accepted_design(phase4_inputs) -> None:
    coordinate = phase4_inputs.design_ledger["coordinate_ledger"]
    assert coordinate["coordinate_rows_sha256"] == COORDINATE_ROWS_SHA256
    assert coordinate["unique_embedding_windows_sha256"] == EMBEDDING_WINDOWS_SHA256
    assert coordinate["unique_acoustic_windows_sha256"] == ACOUSTIC_WINDOWS_SHA256
    assert len(phase4_inputs.episodes) == 695
    assert len(phase4_inputs.candidates) == 810
    assert len(phase4_inputs.pairs) == 313


def test_signal_registry_is_complete_and_unique() -> None:
    registry = signal_registry()
    ids = [row["signal_extractor_id"] for row in registry]
    assert len(ids) == len(set(ids))
    assert len(registry) == 270
    assert all(row["sign"] == "higher_means_change" for row in registry)
    assert {row["causal_horizon_ms"] for row in registry} == {250, 500, 1000}


def test_auc_ties_and_eer_are_deterministic() -> None:
    assert auc([1.0, 1.0], [0.0, 1.0]) == pytest.approx(0.75)
    assert auc([0.0], [1.0]) == 0.0
    assert eer([0.9, 0.8], [0.2, 0.1]) == 0.0


def test_deterministic_gzip_has_stable_bytes() -> None:
    payload = b'{"a":1}\n'
    assert deterministic_gzip(payload) == deterministic_gzip(payload)


def test_eres_cache_v2_shards_are_bounded_deterministic_and_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(phase4_signal, "ERES_CACHE_SHARD_TARGET_BYTES", 2_500)
    windows = [(index * 100, index * 100 + 8000) for index in range(12)]
    generator = np.random.default_rng(7)
    embeddings = generator.normal(size=(12, 192)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    shadows = generator.normal(size=(12, 80)).astype(np.float32)
    shadows /= np.linalg.norm(shadows, axis=1, keepdims=True)
    rms = generator.normal(size=12).astype(np.float32)
    source = AudioSource("fixture", "a" * 64, tmp_path / "unused.wav", 10_000, False)
    contract = {"contract_sha256": "b" * 64}
    first = save_eres_embeddings(
        tmp_path,
        contract,
        "fixture-checkpoint",
        source,
        windows,
        embeddings,
        shadows,
        rms,
        [0.1] * len(windows),
    )
    first_bytes = [Path(path).read_bytes() for path in first["paths"]]
    second = save_eres_embeddings(
        tmp_path,
        contract,
        "fixture-checkpoint",
        source,
        windows,
        embeddings,
        shadows,
        rms,
        [0.1] * len(windows),
    )
    assert first["payload_sha256"] == second["payload_sha256"]
    assert first_bytes == [Path(path).read_bytes() for path in second["paths"]]
    assert all(Path(path).stat().st_size <= 20 * 1024 * 1024 for path in second["paths"])
    loaded = load_eres_embeddings(tmp_path, contract, "fixture-checkpoint", source, windows)
    assert loaded is not None
    loaded_embeddings, loaded_shadows, evidence = loaded
    assert list(loaded_embeddings) == windows
    assert np.allclose(np.stack(list(loaded_embeddings.values())), embeddings)
    assert np.allclose(np.stack([value[0] for value in loaded_shadows.values()]), shadows)
    assert evidence["shard_count"] > 1
    first_shard = Path(evidence["paths"][0])
    mutated = bytearray(first_shard.read_bytes())
    mutated[-1] ^= 1
    first_shard.write_bytes(mutated)
    with pytest.raises(Exception, match="shard byte hash mismatch"):
        load_eres_embeddings(tmp_path, contract, "fixture-checkpoint", source, windows)


def test_eres_state_equivalence_executes_and_detects_prefix_state() -> None:
    windows = [(end - 8000, end) for end in range(8000, 32001, 1600)]
    left = np.zeros(192, dtype=np.float32)
    left[0] = 1.0
    right = np.zeros(192, dtype=np.float32)
    right[1] = 1.0
    embeddings = {
        window: (left.copy() if window[1] < 24000 else right.copy()) for window in windows
    }
    shadow = np.zeros(80, dtype=np.float32)
    shadow[0] = 1.0
    shadows = {window: (shadow.copy(), -1.0) for window in windows}
    source = _eres_state_trace(
        embeddings,
        shadows,
        window=8000,
        step=1600,
        replay_start=0,
        scored_start=24000,
        scored_end=32000,
        snapshot_frontier=16000,
        mode="stable_no_update",
    )
    reset = _eres_state_trace(
        embeddings,
        shadows,
        window=8000,
        step=1600,
        replay_start=16000,
        scored_start=24000,
        scored_end=32000,
        snapshot_frontier=16000,
        mode="stable_no_update",
    )
    comparison = _eres_trace_comparison(source, reset)
    assert comparison["aligned_window_count"] > 0
    assert comparison["aligned_window_cosine_min"] == 1.0
    assert not comparison["passed"]
    assert not comparison["exact_trace_fields"]["proposals"]


@pytest.mark.parametrize("mode", ["confirmed_anchor", "prototype_memory_4"])
def test_eres_exact_half_cosine_is_not_a_change_candidate(mode: str) -> None:
    anchor = np.zeros(192, dtype=np.float32)
    anchor[0] = 1.0
    probe = np.zeros(192, dtype=np.float32)
    probe[0] = 0.5
    probe[1] = np.sqrt(0.75)
    shadow = np.zeros(80, dtype=np.float32)
    shadow[0] = 1.0
    state: dict[str, object] = {"anchor": None, "pending": None}
    _advance_anchor_state(state, mode, anchor, (shadow, -1.0), (0, 8000))
    _advance_anchor_state(state, mode, probe, (shadow, -1.0), (1600, 9600))
    assert state["pending"] is None
    trace = _eres_state_trace(
        {(0, 8000): anchor, (1600, 9600): probe},
        {(0, 8000): (shadow, -1.0), (1600, 9600): (shadow, -1.0)},
        window=8000,
        step=1600,
        replay_start=0,
        scored_start=9600,
        scored_end=9600,
        snapshot_frontier=8000,
        mode=mode,
    )
    assert trace["scores"][0]["change_score"] == pytest.approx(0.50)
    assert trace["proposals"] == []


def test_run_eres_cache_supports_all_hit_and_mixed_retry_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Node:
        def __init__(self, name: str, shape: list[object]) -> None:
            self.name = name
            self.type = "tensor(float)"
            self.shape = shape

    class Session:
        def get_inputs(self):
            return [Node("fbank", [1, "time", 80])]

        def get_outputs(self):
            return [Node("embedding", [1, 192])]

        def run(self, output_names, feed):
            vector = np.arange(1, 193, dtype=np.float32)
            return [vector.reshape(1, -1)]

    class Runtime:
        def __init__(self, path: str) -> None:
            self._session = Session()
            self._output_names = ["embedding"]

    monkeypatch.setattr(phase4_signal, "EresEmbeddingRuntime", Runtime)
    experiment_dir = Path(__file__).resolve().parents[1]
    sources: dict[str, AudioSource] = {}
    windows_by_wav: dict[str, set[tuple[int, int]]] = {}
    for index in range(2):
        path = tmp_path / f"source-{index}.wav"
        samples = (np.sin((np.arange(9000) + index) / 13.0) * 1000).astype("<i2")
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(samples.tobytes())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        source = AudioSource(f"source-{index}", digest, path, 9000, False)
        sources[source.source_id] = source
        windows_by_wav[digest] = {(0, 8000)}
    inputs = Phase4Inputs(
        experiment_dir=experiment_dir,
        result_dir=tmp_path,
        episodes=[],
        candidates=[],
        pairs=[],
        sources=sources,
        source_by_episode={},
        embedding_windows=windows_by_wav,
        acoustic_windows={},
        design_ledger={},
    )
    args = SimpleNamespace(cache_root=tmp_path / "cache", eres_onnx_root=tmp_path)
    _, first = run_eres_cache(inputs, args, source_limit=2, window_limit=1)
    assert all(row["cache_hit_count"] == 0 for row in first.values())
    _, all_hit = run_eres_cache(inputs, args, source_limit=2, window_limit=1)
    assert all(row["cache_hit_count"] == 2 for row in all_hit.values())
    for section in all_hit.values():
        Path(section["sources"][0]["metadata_path"]).unlink()
    _, mixed = run_eres_cache(inputs, args, source_limit=2, window_limit=1)
    assert all(row["cache_hit_count"] == 1 for row in mixed.values())


def test_run_ls_cache_supports_all_hit_and_mixed_retry_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Node:
        def __init__(self, name: str, shape: list[object]) -> None:
            self.name = name
            self.type = "tensor(float)"
            self.shape = shape

    class Session:
        def get_inputs(self):
            return [Node("speech", [1, "time"])]

        def get_outputs(self):
            return [Node("posterior", [1, "frames", 2])]

    class Runtime:
        def __init__(self, *args, **kwargs) -> None:
            self._session = Session()

        def run_case(self, samples, *, case_id, audio_epoch):
            return LSCaptureEpoch(
                case_id=case_id,
                audio_epoch=audio_epoch,
                normal_probs=[np.asarray([0.2, 0.8], dtype=np.float32)],
                normal_frontiers=[len(samples)],
                frame_wall_ns=[0],
                epoch_end_count=len(samples),
                finalize_wall_ns=0,
                length_samples=len(samples),
            )

    monkeypatch.setattr(phase4_signal, "LSEENDCapture", Runtime)
    monkeypatch.setattr(phase4_signal, "load_sidecar_metadata", lambda path: {})
    experiment_dir = Path(__file__).resolve().parents[1]
    sources: dict[str, AudioSource] = {}
    for index in range(2):
        path = tmp_path / f"ls-source-{index}.wav"
        samples = (np.sin((np.arange(9000) + index) / 13.0) * 1000).astype("<i2")
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(samples.tobytes())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        source = AudioSource(f"source-{index}", digest, path, 9000, False)
        sources[source.source_id] = source
    inputs = Phase4Inputs(
        experiment_dir=experiment_dir,
        result_dir=tmp_path,
        episodes=[],
        candidates=[],
        pairs=[],
        sources=sources,
        source_by_episode={},
        embedding_windows={},
        acoustic_windows={},
        design_ledger={},
    )
    args = SimpleNamespace(cache_root=tmp_path / "cache", hf_root=tmp_path)
    _, first = run_ls_cache(inputs, args, source_limit=2)
    assert all(row["cache_hit_count"] == 0 for row in first.values())
    _, all_hit = run_ls_cache(inputs, args, source_limit=2)
    assert all(row["cache_hit_count"] == 2 for row in all_hit.values())
    for section in all_hit.values():
        Path(section["sources"][0]["metadata_path"]).unlink()
    _, mixed = run_ls_cache(inputs, args, source_limit=2)
    assert all(row["cache_hit_count"] == 1 for row in mixed.values())


@pytest.mark.parametrize(
    "profile_class",
    ["new_track_onset", "dominant_replacement", "hysteretic_activity_state"],
)
def test_ls_state_sentinel_trace_is_exact_for_identical_input(profile_class: str) -> None:
    probabilities = np.asarray(
        [[0.8, 0.1], [0.8, 0.1], [0.1, 0.8], [0.1, 0.8], [0.8, 0.1]],
        dtype=np.float32,
    )
    trace = _ls_profile_trace(
        probabilities,
        offset=0,
        epoch_length=100_000,
        profile_class=profile_class,
    )
    bounds = {"scored_start": 0, "scored_end": 100_000}
    assert _scored_trace(trace, bounds) == _scored_trace(trace, bounds)
    reference = _reference_ls_trace(
        probabilities,
        offset=0,
        epoch_length=100_000,
        profile_class=profile_class,
    )
    assert _reference_scored_trace(reference, bounds) == _scored_trace(trace, bounds)


def test_independent_eres_state_replay_matches_runner_fixture() -> None:
    windows = [(end - 8000, end) for end in range(8000, 16001, 1600)]
    generator = np.random.default_rng(17)
    embeddings = {
        window: vector
        for window, vector in zip(
            windows,
            (
                row / np.linalg.norm(row)
                for row in generator.normal(size=(len(windows), 192)).astype(np.float32)
            ),
        )
    }
    shadow = np.zeros(80, dtype=np.float32)
    shadow[0] = 1.0
    shadows = {window: (shadow.copy(), -1.0) for window in windows}
    runner = _eres_state_trace(
        embeddings,
        shadows,
        window=8000,
        step=1600,
        replay_start=0,
        scored_start=9600,
        scored_end=16000,
        snapshot_frontier=8000,
        mode="prototype_memory_4",
    )
    reference = _reference_eres_state_trace(
        embeddings,
        shadows,
        window=8000,
        step=1600,
        replay_start=0,
        scored_start=9600,
        scored_end=16000,
        snapshot_frontier=8000,
        mode="prototype_memory_4",
    )
    assert reference == runner
    assert _reference_eres_comparison(reference, runner)["passed"]


def test_atomic_self_hashed_json_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    written = atomic_write_json(path, {"schema_version": "fixture", "value": 3})
    loaded = read_json(path)
    assert loaded == written
    mutated = json.loads(path.read_text(encoding="utf-8"))
    mutated["value"] = 4
    path.write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(Exception, match="content hash mismatch"):
        read_json(path)


def test_independent_verifier_rebuilds_inputs_and_registry() -> None:
    experiment_dir = Path(__file__).resolve().parents[1]
    inputs = load_input_contract(experiment_dir / "results" / "turn_episode_v1")
    assert len(inputs["episodes"]) == 695
    assert len(inputs["candidates"]) == 810
    assert len(inputs["pairs"]) == 313
    assert sum(len(values) for values in inputs["embedding_windows"].values()) == 895_656
    assert expected_registry() == signal_registry()


def test_independent_auc_and_eer_match_frozen_tie_rules() -> None:
    positives = [1.0, 1.0]
    negatives = [0.0, 1.0]
    assert independent_auc(positives, negatives) == auc(positives, negatives)
    assert independent_eer(positives, negatives) == eer(positives, negatives)


@pytest.mark.parametrize("mutation", MUTATIONS)
def test_public_mutation_fixture_rejects_required_mutation(mutation: str) -> None:
    result = verify_mutation_fixture(mutation)
    assert not result["passed"]
    assert result["mismatches"]
