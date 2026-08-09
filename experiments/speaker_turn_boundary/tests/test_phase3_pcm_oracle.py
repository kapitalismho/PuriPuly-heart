from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from experiments.speaker_turn_boundary.tests.helpers import SequenceVadEngine
from experiments.speaker_turn_boundary.turn_episode.pcm_oracle import (
    EXPECTED_CLAMP_SHA256,
    EXPECTED_POPULATION_SHA256,
    B0LifecycleReplay,
    CanonicalPCMAssembler,
    Phase3OracleError,
    canonical_sha256,
    clamp_identity,
    compact_detail_row,
    population_identity,
    run_pcm_fixtures,
    simulate_case,
    span_cover_flags,
    turn_spans,
)
from experiments.speaker_turn_boundary.turn_episode.verify_pcm_oracle import (
    IndependentAccumulator,
    _hydrate_row,
    _mutation_fixtures,
    _register_definitions,
    _row_mismatches,
)


def test_pcm_fixture_and_property_matrix_passes() -> None:
    result = run_pcm_fixtures()
    assert result["fixtures_passed"]
    assert result["property_checks_passed"]
    assert result["property_cases"] == 756


def test_absolute_origin_late_action_conserves_pcm() -> None:
    assembler = CanonicalPCMAssembler(3, 4096, 5120, 0)
    assembler.append_chunk(4096, 4608)
    assembler.update_progress(4608, 4200)
    assembler.ordinary_release()
    assembler.append_chunk(4608, 5120)
    assembler.update_progress(5120, 4300)
    record = assembler.apply_action(
        action_id="late",
        action_epoch=3,
        boundary_sample=4400,
        availability_sample=5000,
        owner="oracle",
    )
    assembler.terminal(64)
    assert record["recoverability"] == "late_unrecoverable"
    assert record["unrecoverable_span"] == [4400, 4608]
    assert all(span_cover_flags(assembler.realized_spans(), 4096, 5120).values())


def test_stale_epoch_rejection_is_state_invariant() -> None:
    assembler = CanonicalPCMAssembler(9, 1000, 1512, 512)
    assembler.append_chunk(1000, 1512)
    assembler.update_progress(1512, 1200)
    before = assembler.state_digest()
    record = assembler.apply_action(
        action_id="stale",
        action_epoch=8,
        boundary_sample=1300,
        availability_sample=1512,
        owner="oracle",
    )
    assert record["rejection"] == "stale_epoch"
    assert record["state_unchanged"]
    assert assembler.state_digest() == before


def test_safe_drain_uses_scheduler_deadline_without_pcm() -> None:
    assembler = CanonicalPCMAssembler(0, 1000, 1512, 512)
    assembler.append_chunk(1000, 1512)
    assembler.update_progress(1512, 1100)
    assembler.arm_drain("drain", 1400, 7)
    assembler.resolve_drains(2006)
    assert not assembler.drain_records
    assembler.resolve_drains(2007)
    assert assembler.drain_records[0]["outcome"] == "safe_drain_timeout_fallback"
    assert assembler.drain_records[0]["scheduler_latency_ms"] == 2000


def test_drain_fifo_and_regression_contract() -> None:
    assembler = CanonicalPCMAssembler(0, 1000, 1512, 512)
    assembler.append_chunk(1000, 1512)
    assembler.update_progress(1512, 1350)
    assembler.arm_drain("a", 1200, 0)
    assembler.arm_drain("b", 1300, 100)
    assembler.arm_drain("c", 1400, 200)
    assembler.resolve_drains(0)
    assembler.resolve_drains(2200)
    assert [record["drain_id"] for record in assembler.drain_records] == ["a", "b", "c"]
    assert [record["outcome"] for record in assembler.drain_records] == [
        "safe_complete",
        "safe_complete",
        "safe_drain_timeout_fallback",
    ]
    regressed = CanonicalPCMAssembler(0, 1000, 1512, 512)
    regressed.append_chunk(1000, 1512)
    regressed.update_progress(1512, 1000)
    regressed.arm_drain("a", 1300, 0)
    with pytest.raises(Phase3OracleError):
        regressed.arm_drain("b", 1200, 0)


def test_turn_span_cover_rejects_duplication() -> None:
    spans = turn_spans(1000, 2000, [1250, 1500, 1500, 1750])
    assert all(span_cover_flags(spans, 1000, 2000).values())
    duplicated = [*spans, dict(spans[0])]
    assert not span_cover_flags(duplicated, 1000, 2000)["no_duplication"]


def test_population_and_clamp_identities_match_reviewed_bundle() -> None:
    results = Path(__file__).resolve().parent.parent / "results" / "turn_episode_v1"
    manifest = json.loads((results / "episode_manifest_dev.json").read_text(encoding="utf-8"))
    scoring = json.loads((results / "scoring_fixture_report.json").read_text(encoding="utf-8"))
    selected = {str(row["episode_id"]) for row in scoring["baseline_smoke"]["rows"]}
    episodes = [
        episode
        for episode in manifest["episodes"]
        if episode["episode_id"] in selected and episode["status"] == "scorable"
    ]
    processed_ends = {
        str(episode["episode_id"]): int(episode["bounds"]["scored_end"]) for episode in episodes
    }
    assert population_identity(episodes)["sha256"] == EXPECTED_POPULATION_SHA256
    assert clamp_identity(episodes, processed_ends)["sha256"] == EXPECTED_CLAMP_SHA256


def test_lifecycle_replay_emits_structural_and_terminal_events(tmp_path: Path) -> None:
    wav_path = tmp_path / "lifecycle.wav"
    samples = np.zeros(512 * 224, dtype="<i2")
    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(samples.tobytes())
    probabilities = [1.0] * 224
    replay = B0LifecycleReplay(lambda: SequenceVadEngine(probabilities)).replay(wav_path, "fixture")
    reasons = [event["reason"] for event in replay["events"]]
    assert "max_duration" in reasons
    assert reasons[-1] == "end_of_input"
    assert replay["events"][-1]["event_source_sample"] == samples.size


def _synthetic_case() -> dict[str, object]:
    singleton = [[4096, 8000, "A"], [8000, 16384, "B"]]
    episode = {
        "episode_content_sha256": "ab" * 32,
        "session_id": "session",
        "episode_id": "episode",
        "pool": "diagnostic_dev",
        "tag": "hard_only",
        "status": "scorable",
        "audio_epoch": 0,
    }
    reference = {
        "reference_id": "reference",
        "action_kind": "hard_boundary",
        "scorable": True,
        "audio_epoch": 0,
        "target_sample": 8000,
        "acceptable_interval": [8000, 8000],
        "evidence_onset_sample": 8000,
    }
    return {
        "episode": episode,
        "start": 4096,
        "end": 16384,
        "hard_references": [reference],
        "b0_actions": [],
        "lifecycle_events": [],
        "regions": [],
        "regions_digest": canonical_sha256([]),
        "singleton_intervals": singleton,
        "singleton_digest": canonical_sha256(singleton),
    }


def test_independent_row_recomputation_and_mutations() -> None:
    row = simulate_case(_synthetic_case(), 250, 0, 500)
    assert _row_mismatches(row) == []
    accumulator = IndependentAccumulator(250, 0, 500)
    accumulator.add(row)
    mutations = _mutation_fixtures(row, accumulator.finish())
    assert {fixture["fixture_id"] for fixture in mutations} == {
        "missing_row",
        "duplicated_span",
        "altered_ownership",
        "altered_contamination_numerator",
        "altered_quantile",
    }
    assert all(fixture["rejected"] for fixture in mutations)


def test_compact_detail_round_trip_preserves_independent_evidence() -> None:
    original = simulate_case(_synthetic_case(), 250, 0, 500)
    compacted = compact_detail_row(json.loads(json.dumps(original)), set())
    cache: dict[str, dict[str, object]] = {}
    assert _register_definitions(compacted, cache) == []
    hydrated, errors = _hydrate_row(compacted, cache)
    assert errors == []
    assert hydrated is not None
    assert _row_mismatches(hydrated) == []
    assert hydrated["progress_rows"] == original["progress_rows"]
    assert hydrated["baseline_turn_spans"] == original["baseline_turn_spans"]
    assert hydrated["singleton_intervals"] == original["singleton_intervals"]
