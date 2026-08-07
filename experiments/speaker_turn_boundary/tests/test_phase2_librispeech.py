from __future__ import annotations

import json

import numpy as np
import pytest

from experiments.speaker_turn_boundary.corpus import librispeech as ls
from experiments.speaker_turn_boundary.corpus.librispeech import (
    SplitIndex,
    UtteranceInfo,
    _apply_opus,
    _duration_target,
    _valid_overlaps,
    _zero_gap_evidence,
    build_librispeech_manifest,
    build_splice,
    trim_energy,
)
from experiments.speaker_turn_boundary.corpus.phase2_schemas import (
    validate_phase2_manifest,
)
from experiments.speaker_turn_boundary.ground_truth import (
    classify_active_speaker_transitions,
)
from experiments.speaker_turn_boundary.tests.phase2_helpers import (
    speech_like,
    write_pcm16_flac,
)


def _speech(seconds: float, seed: int = 1) -> np.ndarray:
    return speech_like(seconds, seed)


def test_trim_energy_removes_leading_trailing_silence():
    speech = _speech(1.0, seed=3)
    samples = np.concatenate(
        [np.zeros(3200, dtype=np.float32), speech, np.zeros(4800, dtype=np.float32)]
    )
    start, end = trim_energy(samples)
    assert start <= 3200 + 640
    assert end >= 16000 + 3200 - 640
    assert end <= samples.size


def test_trim_energy_returns_none_for_silence():
    assert trim_energy(np.zeros(32000, dtype=np.float32)) is None


def test_build_splice_gap_tiles_regions():
    audio, regions, splice = build_splice(
        _speech(0.5, seed=1),
        _speech(0.5, seed=2),
        gap_ms=300,
        overlap_ms=None,
        lead_samples=1600,
        tail_samples=1600,
    )
    assert splice.a_end_sample == 1600 + 8000
    assert splice.b_onset_sample == 1600 + 8000 + 4800
    assert splice.gap_samples == 4800
    assert sum(r.end_sample - r.start_sample for r in regions) == audio.size
    assert [r.speakers for r in regions] == [frozenset(), {"A"}, set(), {"B"}, set()]
    changes, _ = classify_active_speaker_transitions(regions)
    assert len(changes) == 1
    assert changes[0].kind == "gap_speaker_change"
    assert changes[0].change_sample == splice.b_onset_sample


def test_build_splice_zero_gap_clean_handoff():
    audio, regions, splice = build_splice(
        _speech(0.5, seed=1),
        _speech(0.5, seed=2),
        gap_ms=0,
        overlap_ms=None,
        lead_samples=1600,
        tail_samples=1600,
    )
    assert splice.b_onset_sample == splice.a_end_sample
    assert [r.speakers for r in regions] == [frozenset(), {"A"}, {"B"}, set()]
    changes, _ = classify_active_speaker_transitions(regions)
    assert len(changes) == 1
    assert changes[0].kind == "clean_handoff"
    assert changes[0].change_sample == splice.a_end_sample


def test_build_splice_overlap_interruption_onset():
    audio, regions, splice = build_splice(
        _speech(0.6, seed=1),
        _speech(0.6, seed=2),
        gap_ms=None,
        overlap_ms=300,
        lead_samples=1600,
        tail_samples=1600,
    )
    assert splice.b_onset_sample == splice.a_end_sample - 4800
    assert [r.speakers for r in regions] == [frozenset(), {"A"}, {"A", "B"}, {"B"}, set()]
    changes, _ = classify_active_speaker_transitions(regions)
    assert len(changes) == 1
    assert changes[0].kind == "interruption_onset"
    assert changes[0].change_sample == splice.b_onset_sample


def test_build_splice_rejects_overlap_larger_than_duration():
    with pytest.raises(Exception):
        build_splice(
            _speech(0.2, seed=1),
            _speech(0.2, seed=2),
            gap_ms=None,
            overlap_ms=500,
            lead_samples=1600,
            tail_samples=1600,
        )


def test_zero_gap_evidence_b_onset_is_a_end():
    a = _speech(0.5, seed=1)
    b = _speech(0.5, seed=2)
    audio, _, splice = build_splice(
        a, b, gap_ms=0, overlap_ms=None, lead_samples=1600, tail_samples=1600
    )
    evidence = _zero_gap_evidence(audio, splice.a_end_sample)
    assert evidence.b_onset_is_a_end
    assert evidence.pre_junction_rms > 1e-3
    assert evidence.post_junction_rms > 1e-3
    assert evidence.junction_peak_abs > 1e-4


def test_duration_target_stress_bucket():
    import random

    rng = random.Random(42)
    for _ in range(50):
        target = _duration_target(rng, "stress")
        assert 0.30 <= target <= 0.50
    assert _duration_target(rng, "1.0") == 1.0


def test_valid_overlaps_respect_duration():
    assert _valid_overlaps("2.0") == [100, 300, 500]
    assert _valid_overlaps("0.5") == [100, 300]
    assert _valid_overlaps("stress") == [100]


def test_apply_opus_round_trip_length():
    speech = _speech(1.0, seed=5)
    decoded = _apply_opus(speech)
    assert decoded.size == speech.size
    assert np.sqrt(np.mean((decoded - speech) ** 2)) < 0.05


def _tiny_index(tmp_path) -> SplitIndex:
    index = SplitIndex(split="dev-clean")
    speaker_meta = [
        ("100", "1211", 2.8),
        ("100", "1212", 3.0),
        ("101", "1234", 2.6),
        ("101", "1235", 3.2),
        ("102", "9999", 2.9),
    ]
    for speaker, chapter, duration in speaker_meta:
        for utt_index in range(3):
            utterance_id = f"{speaker}-{chapter}-{utt_index:04d}"
            samples = speech_like(duration, seed=int(speaker) + utt_index)
            samples = np.concatenate(
                [np.zeros(4000, dtype=np.float32), samples, np.zeros(6000, dtype=np.float32)]
            )
            flac_path = tmp_path / f"LibriSpeech/dev-clean/{speaker}/{chapter}/{utterance_id}.flac"
            flac_path.parent.mkdir(parents=True, exist_ok=True)
            write_pcm16_flac(flac_path, samples)
            index.utterances.append(
                UtteranceInfo(
                    split="dev-clean",
                    speaker=speaker,
                    chapter=chapter,
                    utterance_id=utterance_id,
                    path=flac_path,
                    transcript="hello",
                    duration_seconds=duration + 0.625,
                )
            )
            index.by_id[utterance_id] = index.utterances[-1]
            index.by_speaker.setdefault(speaker, []).append(index.utterances[-1])
    index.utterances.sort(key=lambda u: u.utterance_id)
    index.speakers = sorted(index.by_speaker)
    return index


def test_tiny_manifest_build_deterministic_and_valid(tmp_path, monkeypatch):
    monkeypatch.setattr(ls, "DURATION_TARGETS", [1.0, "stress"])
    monkeypatch.setattr(ls, "GAPS_MS", [800, 0])
    monkeypatch.setattr(ls, "OVERLAPS_MS", [100, 300])
    monkeypatch.setattr(
        ls,
        "CASE_COUNTS",
        {
            **ls.CASE_COUNTS,
            "positive_per_combo": 1,
            "same_speaker_per_combo": 1,
            "gain_per_combo": 1,
            "stress_per_combo": 1,
            "silence": 1,
            "noise_only": 1,
            "bandlimit": 1,
        },
    )
    index = _tiny_index(tmp_path)
    first = build_librispeech_manifest(
        split="dev-clean", manifest_id="tiny", out_dir=tmp_path / "a", index=index, seed=99
    )
    second = build_librispeech_manifest(
        split="dev-clean", manifest_id="tiny", out_dir=tmp_path / "b", index=index, seed=99
    )
    assert first.hash == second.hash
    assert (tmp_path / "a" / "manifests" / "tiny.json").read_bytes() == (
        tmp_path / "b" / "manifests" / "tiny.json"
    ).read_bytes()
    assert validate_phase2_manifest(first, tmp_path / "a") == []
    zero_gap = [c for c in first.cases if c.condition.get("gap_ms") == 0]
    assert zero_gap
    for case in zero_gap:
        assert case.zero_gap_evidence is not None
        assert case.zero_gap_evidence.b_onset_is_a_end
    positive = [c for c in first.cases if c.kind == "different_speaker_gap"]
    assert positive
    assert all(case.splice is not None for case in positive)
    assert all(
        case.splice.b_onset_sample
        == case.splice.a_end_sample
        + (case.splice.gap_samples or 0)
        - (case.splice.overlap_samples or 0)
        for case in first.cases
        if case.splice is not None
    )
    assert len({source.speaker for case in first.cases for source in case.sources}) >= 3


def test_manifest_json_has_source_hashes(tmp_path, monkeypatch):
    monkeypatch.setattr(ls, "DURATION_TARGETS", [1.0])
    monkeypatch.setattr(ls, "GAPS_MS", [800])
    monkeypatch.setattr(ls, "OVERLAPS_MS", [100])
    monkeypatch.setattr(
        ls,
        "CASE_COUNTS",
        {
            **ls.CASE_COUNTS,
            "positive_per_combo": 1,
            "same_speaker_per_combo": 1,
            "gain_per_combo": 1,
            "stress_per_combo": 1,
            "silence": 1,
            "noise_only": 1,
            "bandlimit": 1,
        },
    )
    index = _tiny_index(tmp_path)
    build_librispeech_manifest(
        split="dev-clean", manifest_id="tiny2", out_dir=tmp_path, index=index, seed=5
    )
    data = json.loads((tmp_path / "manifests" / "tiny2.json").read_text(encoding="utf-8"))
    first = data["cases"][0]
    assert first["sources"][0]["file_sha256"]
    assert first["sources"][0]["cut_start_sample"] <= first["sources"][0]["cut_end_sample"]
    assert first["transforms"] is not None
    assert first["active_speech_samples"] > 0
    assert data["build"]["config_hash"]
