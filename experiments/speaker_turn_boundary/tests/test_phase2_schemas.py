from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.speaker_turn_boundary.corpus.phase2_schemas import (
    PHASE2_MANIFEST_SCHEMA,
    Phase2Case,
    Phase2Manifest,
    SourceRef,
    SpliceSpec,
    TransformSpec,
    ZeroGapEvidence,
    expected_change_kind,
    make_phase2_manifest,
    validate_phase2_manifest,
)
from experiments.speaker_turn_boundary.ground_truth import SpeakerRegion


def _case(case_id: str = "case0") -> Phase2Case:
    return Phase2Case(
        case_id=case_id,
        wav_relative_path="generated/case0.wav",
        duration_samples=16000,
        wav_sha256="00" * 32,
        seed=7,
        regions=[SpeakerRegion(0, 0, 16000, frozenset())],
        kind="different_speaker_gap",
        condition={"duration_target_s": 1.0, "transition": "gap", "gap_ms": 0, "overlap_ms": None},
        sources=[
            SourceRef(
                role="A",
                speaker="s1",
                session="s1-c1",
                utterance="s1-c1-0001",
                file_sha256="11" * 32,
                original_start_sample=0,
                original_end_sample=16000,
                trimmed_start_sample=0,
                trimmed_end_sample=16000,
                cut_start_sample=0,
                cut_end_sample=16000,
                gain=1.0,
            )
        ],
        splice=SpliceSpec(
            a_end_sample=8000, b_onset_sample=8000, gap_samples=0, overlap_samples=None
        ),
        transforms=[TransformSpec("opus", {"bitrate_kbps": 32})],
        zero_gap_evidence=ZeroGapEvidence(
            b_onset_is_a_end=True,
            pre_junction_rms=0.01,
            post_junction_rms=0.01,
            junction_peak_abs=0.5,
        ),
        active_speech_samples=16000,
    )


def _manifest(tmp_path: Path, cases: list[Phase2Case] | None = None) -> Phase2Manifest:
    return make_phase2_manifest(
        manifest_id="test_manifest",
        split_role="dev",
        corpus={"name": "librispeech", "version": "1", "license": "CC BY 4.0"},
        build={"script": "test", "config_hash": "x"},
        disjointness_groups=["g1"],
        generator={"script": "test"},
        cases=cases or [_case()],
    )


def test_phase2_manifest_round_trip_and_hash(tmp_path):
    manifest = _manifest(tmp_path)
    restored = Phase2Manifest.from_dict(json.loads(json.dumps(manifest.to_dict())))
    assert restored.to_dict() == manifest.to_dict()
    assert restored.hash == manifest.hash
    assert manifest.schema_version == PHASE2_MANIFEST_SCHEMA
    assert manifest.baseline_sha


def test_phase2_manifest_self_describing_fields(tmp_path):
    manifest = _manifest(tmp_path)
    data = manifest.to_dict()
    assert data["split_role"] == "dev"
    assert data["corpus"]["license"] == "CC BY 4.0"
    assert data["build"]["config_hash"]
    assert data["disjointness_groups"] == ["g1"]
    assert data["cases"][0]["seed"] == 7
    assert data["cases"][0]["condition"]["gap_ms"] == 0
    assert data["cases"][0]["transforms"][0]["name"] == "opus"
    assert data["cases"][0]["zero_gap_evidence"]["b_onset_is_a_end"] is True


def test_expected_change_kind_mapping():
    assert (
        expected_change_kind({"gap_ms": 0, "transition": "gap"}, "different_speaker_gap")
        == "clean_handoff"
    )
    assert (
        expected_change_kind({"gap_ms": 300, "transition": "gap"}, "different_speaker_gap")
        == "gap_speaker_change"
    )
    assert (
        expected_change_kind(
            {"transition": "overlap", "overlap_ms": 300}, "different_speaker_overlap"
        )
        == "interruption_onset"
    )
    assert expected_change_kind({"gap_ms": 0}, "same_speaker") is None
    assert expected_change_kind({}, "silence") is None
    assert expected_change_kind({}, "real_meeting") is None


def test_validate_phase2_manifest_rejects_missing_wav(tmp_path):
    manifest = _manifest(tmp_path)
    problems = validate_phase2_manifest(manifest, tmp_path)
    assert any("wav missing" in problem for problem in problems)


def test_validate_phase2_manifest_accepts_valid_wav(tmp_path):
    manifest = _manifest(tmp_path)
    from experiments.speaker_turn_boundary.tests.phase2_helpers import write_pcm16_wav

    wav_path = tmp_path / "generated" / "case0.wav"
    write_pcm16_wav(wav_path, np.zeros(16000, dtype=np.float32))
    import hashlib

    cases = [
        Phase2Case(
            case_id=case.case_id,
            wav_relative_path=case.wav_relative_path,
            duration_samples=case.duration_samples,
            wav_sha256=hashlib.sha256(wav_path.read_bytes()).hexdigest(),
            seed=case.seed,
            regions=case.regions,
            kind=case.kind,
            condition=case.condition,
            sources=case.sources,
            splice=case.splice,
            transforms=case.transforms,
            zero_gap_evidence=case.zero_gap_evidence,
            active_speech_samples=case.active_speech_samples,
        )
        for case in manifest.cases
    ]
    manifest = _manifest(tmp_path, cases=cases)
    problems = validate_phase2_manifest(manifest, tmp_path)
    assert problems == []


def test_validate_phase2_manifest_rejects_duplicate_ids(tmp_path):
    manifest = _manifest(tmp_path, cases=[_case("dup"), _case("dup")])
    from experiments.speaker_turn_boundary.tests.phase2_helpers import write_pcm16_wav

    for case in manifest.cases:
        write_pcm16_wav(
            tmp_path / case.wav_relative_path,
            np.zeros(case.duration_samples, dtype=np.float32),
        )
    problems = validate_phase2_manifest(manifest, tmp_path)
    assert any("duplicate case_id" in problem for problem in problems)


def test_phase2_manifest_write_reproducible(tmp_path):
    first = _manifest(tmp_path)
    second = _manifest(tmp_path)
    assert first.hash == second.hash
    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    first.write(path_a)
    second.write(path_b)
    assert path_a.read_bytes() == path_b.read_bytes()
