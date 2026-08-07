from __future__ import annotations

import json

import numpy as np
import pytest

from experiments.speaker_turn_boundary.build_synthetic_cases import build
from experiments.speaker_turn_boundary.config import (
    BASELINE_SHA,
    MANIFEST_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
)
from experiments.speaker_turn_boundary.schemas import (
    DatasetManifest,
    ManifestCase,
    RunResult,
    SchemaValidationError,
    canonical_json,
    sha256_hex,
    validate_manifest,
)
from experiments.speaker_turn_boundary.tests.helpers import write_pcm16_wav


def test_canonical_json_is_key_order_independent():
    first = canonical_json({"b": 2, "a": {"d": 1, "c": 2}})
    second = canonical_json({"a": {"c": 2, "d": 1}, "b": 2})
    assert first == second
    assert sha256_hex({"b": 2, "a": 1}) == sha256_hex({"a": 1, "b": 2})


def test_build_manifest_is_deterministic(tmp_dir):
    first = build(tmp_dir / "first", seed=7, manifest_id="det")
    second = build(tmp_dir / "second", seed=7, manifest_id="det")
    first_path = tmp_dir / "first" / "manifests" / "det.json"
    second_path = tmp_dir / "second" / "manifests" / "det.json"
    assert first_path.read_bytes() == second_path.read_bytes()
    assert first.hash == second.hash


def test_build_manifest_records_baseline_and_schema(tmp_dir):
    manifest = build(tmp_dir, seed=7, manifest_id="det")
    assert manifest.schema_version == MANIFEST_SCHEMA_VERSION
    assert manifest.baseline_sha == BASELINE_SHA
    assert manifest.canonical_sample_rate_hz == 16000
    assert manifest.generator == {
        "script": "build_synthetic_cases.py",
        "seed": 7,
    }
    case_ids = {case.case_id for case in manifest.cases}
    assert case_ids == {
        "golden_two_utterance_gap400",
        "golden_single_utterance",
        "golden_silence",
    }


def test_built_manifest_validates_against_generated_wavs(tmp_dir):
    manifest = build(tmp_dir, seed=7, manifest_id="det")
    validate_manifest(manifest, tmp_dir)


def test_validate_manifest_rejects_missing_wav(tmp_dir):
    manifest = build(tmp_dir, seed=7, manifest_id="det")
    missing = DatasetManifest(
        manifest_id="x",
        schema_version=MANIFEST_SCHEMA_VERSION,
        baseline_sha=BASELINE_SHA,
        canonical_sample_rate_hz=16000,
        generator={"script": "x", "seed": 1},
        cases=[
            ManifestCase.from_dict(
                _replace_path(manifest.cases[0].to_dict(), "generated/missing.wav")
            )
        ],
    )
    with pytest.raises(SchemaValidationError):
        validate_manifest(missing, tmp_dir)


def _replace_path(case_dict: dict, new_path: str) -> dict:
    updated = dict(case_dict)
    updated["wav_relative_path"] = new_path
    return updated


def test_validate_manifest_rejects_wrong_sample_rate_wav(tmp_dir):
    manifest = build(tmp_dir, seed=7, manifest_id="det")
    wrong = tmp_dir / "wrong_rate.wav"
    write_pcm16_wav(wrong, np.zeros(1000, dtype=np.float32), sample_rate_hz=8000)
    modified = DatasetManifest(
        manifest_id=manifest.manifest_id,
        schema_version=manifest.schema_version,
        baseline_sha=manifest.baseline_sha,
        canonical_sample_rate_hz=manifest.canonical_sample_rate_hz,
        generator=manifest.generator,
        cases=[
            ManifestCase.from_dict(_replace_path(manifest.cases[0].to_dict(), "wrong_rate.wav"))
        ],
    )
    with pytest.raises(SchemaValidationError):
        validate_manifest(modified, tmp_dir)


def test_validate_manifest_rejects_duration_mismatch(tmp_dir):
    manifest = build(tmp_dir, seed=7, manifest_id="det")
    modified = DatasetManifest(
        manifest_id=manifest.manifest_id,
        schema_version=manifest.schema_version,
        baseline_sha=manifest.baseline_sha,
        canonical_sample_rate_hz=manifest.canonical_sample_rate_hz,
        generator=manifest.generator,
        cases=[
            ManifestCase.from_dict(_replace_path(manifest.cases[0].to_dict(), "golden_silence.wav"))
        ],
    )
    with pytest.raises(SchemaValidationError):
        validate_manifest(modified, tmp_dir)


def test_validate_manifest_rejects_tampered_wav(tmp_dir):
    manifest = build(tmp_dir, seed=7, manifest_id="det")
    wav_path = tmp_dir / "generated" / "golden_silence.wav"
    write_pcm16_wav(wav_path, np.full(32000, 0.001, dtype=np.float32))
    with pytest.raises(SchemaValidationError):
        validate_manifest(manifest, tmp_dir)


def test_manifest_round_trip_via_json(tmp_dir):
    manifest = build(tmp_dir, seed=7, manifest_id="det")
    restored = DatasetManifest.from_dict(json.loads(canonical_json(manifest.to_dict())))
    assert restored.to_dict() == manifest.to_dict()
    assert restored.hash == manifest.hash


def test_manifest_regions_match_case_structure(tmp_dir):
    manifest = build(tmp_dir, seed=7, manifest_id="det")
    by_id = {case.case_id: case for case in manifest.cases}
    two_utterance = by_id["golden_two_utterance_gap400"]
    labels = [region.speakers for region in two_utterance.regions]
    assert labels == [
        frozenset(),
        frozenset({"A"}),
        frozenset(),
        frozenset({"B"}),
        frozenset(),
    ]
    assert (
        sum(region.end_sample - region.start_sample for region in two_utterance.regions)
        == two_utterance.duration_samples
    )


def test_result_round_trip_and_self_hash(tmp_dir):
    result = RunResult(
        result_id="abc",
        schema_version=RESULT_SCHEMA_VERSION,
        baseline_sha=BASELINE_SHA,
        profile_id="b0",
        manifest_id="det",
        manifest_sha256="mhash",
        seed=7,
        runtime_metadata={"k": 1},
        started_at_utc="2026-01-01T00:00:00+00:00",
        finished_at_utc="2026-01-01T00:00:01+00:00",
        epochs=[],
        coalescing={"report": {"a": 1}, "cuts": [], "detections": []},
    )
    hashed = result.with_self_hash()
    assert hashed.verify_self_hash()
    restored = RunResult.from_dict({**hashed.to_dict(), "result_sha256": hashed.result_sha256})
    assert restored.to_dict() == hashed.to_dict()
    assert restored.verify_self_hash()


def test_result_write_and_hash(tmp_dir):
    result = RunResult(
        result_id="abc",
        schema_version=RESULT_SCHEMA_VERSION,
        baseline_sha=BASELINE_SHA,
        profile_id="b0",
        manifest_id="det",
        manifest_sha256="mhash",
        seed=7,
        runtime_metadata={},
        started_at_utc="x",
        finished_at_utc="y",
        epochs=[],
        coalescing={"report": {}, "cuts": [], "detections": []},
    )
    out_path = tmp_dir / "result.json"
    written_hash = result.write(out_path)
    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert loaded["result_sha256"] == written_hash
    restored = RunResult.from_dict(loaded)
    assert restored.verify_self_hash()
