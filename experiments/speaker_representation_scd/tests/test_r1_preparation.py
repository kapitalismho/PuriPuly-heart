from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from experiments.speaker_representation_scd.acquire_r1 import (
    _write_json as write_acquisition_json,
)
from experiments.speaker_representation_scd.extraction.common import (
    ExtractionBatch,
    l2_normalize,
    mean_pool_valid,
    trailing_window,
)
from experiments.speaker_representation_scd.extraction.fixtures import (
    EXPECTED_FIXTURE_MANIFEST_SHA256,
    EXPECTED_FIXTURE_SHA256,
    d0_fixtures,
    fixture_manifest,
    fixture_window_contract,
    mutate_future,
)
from experiments.speaker_representation_scd.provenance import (
    canonical_json_bytes,
    load_json,
    self_sha256_valid,
    sha256_bytes,
    sha256_file,
    with_self_sha256,
)
from experiments.speaker_representation_scd.r1_gate import (
    EXPECTED_ACTIONS,
    EXPECTED_EXECUTION_CODE_PATHS,
    GATE_PATH,
    SOURCE_REGISTRY_PATH,
    R1GateError,
    _validate_source_registry,
    validate_r1_gate,
)
from experiments.speaker_representation_scd.r1_smoke import (
    _expected_eres_lengths,
    _expected_ssl_length,
    _timestamp_mapping,
)
from experiments.speaker_representation_scd.r1_smoke import (
    _write_json as write_smoke_json,
)
from experiments.speaker_representation_scd.validate_r0 import DEFAULT_PATHS

ROOT = Path(__file__).resolve().parents[1]


def test_r1_source_registry_is_self_hashed_and_bridges_r0_artifacts() -> None:
    source = load_json(ROOT / "models" / "source_registry.json")
    r0 = load_json(ROOT / "models" / "registry.json")
    assert self_sha256_valid(source)
    assert source["r0_registry"]["sha256"] == sha256_file(ROOT / "models" / "registry.json")
    r0_hashes = {model["model_id"]: model["artifact"]["sha256"] for model in r0["models"]}
    source_hashes = {
        model["model_id"]: next(
            row["sha256"]
            for row in model["required_files"]
            if row["path"] in {"model.safetensors", "pytorch_model.bin"}
        )
        for model in source["models"]
    }
    source_hashes[source["eres2netv2"]["model_id"]] = source["eres2netv2"]["checkpoint_file"][
        "sha256"
    ]
    assert source_hashes == r0_hashes


def test_eres_taps_are_explicit_and_fused_is_official_pool_input() -> None:
    source = load_json(ROOT / "models" / "source_registry.json")
    taps = source["eres2netv2"]["taps"]
    assert [tap["tap_id"] for tap in taps] == ["S1", "S2", "S3", "S4", "FUSED"]
    assert [tap["flattened_dimension"] for tap in taps] == [10240] * 5
    assert [tap["temporal_stride_ms"] for tap in taps] == [10, 20, 40, 80, 80]
    assert [tap["official_pool_input"] for tap in taps] == [False, False, False, False, True]


def test_r1_model_file_inventory_matches_the_gate_byte_forecast() -> None:
    source = load_json(ROOT / "models" / "source_registry.json")
    total = sum(row["size_bytes"] for model in source["models"] for row in model["required_files"])
    total += source["eres2netv2"]["checkpoint_file"]["size_bytes"]
    total += source["eres2netv2"]["checkpoint_config"]["size_bytes"]
    assert total == 1_209_138_839


def test_r1_acquisition_gate_is_valid_without_a_live_process_scan() -> None:
    result = validate_r1_gate(scan_processes=False)
    assert result.valid, result.errors
    assert result.allowed_actions == EXPECTED_ACTIONS
    gate = load_json(ROOT / GATE_PATH)
    release = load_json(ROOT / "results" / "r1" / "legacy_release.json")
    assert self_sha256_valid(gate)
    assert self_sha256_valid(release)


def test_rehashed_r1_action_expansion_is_rejected(tmp_path: Path) -> None:
    paths = set(DEFAULT_PATHS.values()) | {
        str(GATE_PATH).replace("\\", "/"),
        str(SOURCE_REGISTRY_PATH).replace("\\", "/"),
        "environment/pyproject.toml",
        "environment/uv.lock",
        "results/r1/legacy_release.json",
        *EXPECTED_EXECUTION_CODE_PATHS,
    }
    for relative in paths:
        source = ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    gate_path = tmp_path / GATE_PATH
    gate = load_json(gate_path)
    gate["authorization"]["corpus_download"] = True
    gate = with_self_sha256(gate)
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    result = validate_r1_gate(tmp_path, scan_processes=False)
    assert not result.valid
    assert "r1_gate.authorization: action boundary differs" in result.errors


def test_rehashed_remote_code_or_source_identity_mutation_is_rejected(tmp_path: Path) -> None:
    source = load_json(ROOT / "models" / "source_registry.json")
    source["models"][0]["trust_remote_code"] = True
    source = with_self_sha256(source)
    path = tmp_path / "source_registry.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    errors: list[str] = []
    _validate_source_registry(path, errors)
    assert any("not the reviewed R1 source identity" in error for error in errors)
    assert any("remote code must be false" in error for error in errors)


def test_d0_fixtures_have_frozen_identities_and_future_mutation_is_after_frontier() -> None:
    fixtures = d0_fixtures()
    assert len(fixtures) == 10
    assert {fixture.fixture_id: fixture.waveform_sha256 for fixture in fixtures} == (
        EXPECTED_FIXTURE_SHA256
    )
    assert sha256_bytes(canonical_json_bytes(fixture_manifest(fixtures))) == (
        EXPECTED_FIXTURE_MANIFEST_SHA256
    )
    assert {fixture.scenario_kind for fixture in fixtures} == {
        "silence",
        "one_speaker",
        "clean_a_to_b",
        "gap_a_to_b",
        "overlap_a_to_b",
        "backchannel_b_to_a",
        "gain_step_same_speaker",
        "noise_step_same_speaker",
        "timestamp_impulses",
        "channel_chirp_same_speaker",
    }
    for fixture in fixtures:
        contract = fixture_window_contract(fixture)
        assert contract["passed"], contract["errors"]
        changed = mutate_future(fixture)
        assert np.array_equal(
            fixture.waveform[: fixture.frontier_sample],
            changed[: fixture.frontier_sample],
        )
        assert not np.array_equal(
            fixture.waveform[fixture.frontier_sample :],
            changed[fixture.frontier_sample :],
        )
        assert np.array_equal(
            trailing_window(
                fixture.waveform,
                fixture.frontier_sample,
                fixture.window_samples,
            ),
            trailing_window(
                changed,
                fixture.frontier_sample,
                fixture.window_samples,
            ),
        )


def test_d0_transition_windows_contain_the_named_scenario() -> None:
    fixtures = {fixture.fixture_id: fixture for fixture in d0_fixtures()}

    def active(fixture_id: str, sample: int) -> set[str]:
        return {
            speaker
            for speaker, start, end in fixtures[fixture_id].speaker_segments
            if start <= sample < end
        }

    clean = fixtures["clean_a_to_b"]
    clean_event = clean.event_samples[0]
    assert clean.frontier_sample - clean.window_samples < clean_event < clean.frontier_sample
    assert active(clean.fixture_id, clean_event - 1) == {"A"}
    assert active(clean.fixture_id, clean_event) == {"B"}

    gap = fixtures["gap_a_to_b"]
    gap_event = gap.event_samples[0]
    gap_start = max(end for speaker, _, end in gap.speaker_segments if speaker == "A")
    assert gap.frontier_sample - gap.window_samples < gap_start < gap_event < gap.frontier_sample
    assert active(gap.fixture_id, gap_start - 1) == {"A"}
    assert active(gap.fixture_id, gap_start) == set()
    assert active(gap.fixture_id, gap_event) == {"B"}

    overlap = fixtures["overlap_a_to_b"]
    overlap_onset, overlap_exclusive = overlap.event_samples
    assert (
        overlap.frontier_sample - overlap.window_samples
        < overlap_onset
        < overlap_exclusive
        < overlap.frontier_sample
    )
    assert active(overlap.fixture_id, overlap_onset - 1) == {"A"}
    assert active(overlap.fixture_id, overlap_onset) == {"A", "B"}
    assert active(overlap.fixture_id, overlap_exclusive) == {"B"}

    backchannel = fixtures["backchannel_b_to_a"]
    backchannel_onset, backchannel_offset = backchannel.event_samples
    assert (
        backchannel.frontier_sample - backchannel.window_samples
        < backchannel_onset
        < backchannel_offset
        < backchannel.frontier_sample
    )
    assert active(backchannel.fixture_id, backchannel_onset - 1) == {"A"}
    assert active(backchannel.fixture_id, backchannel_onset) == {"A", "B"}
    assert active(backchannel.fixture_id, backchannel_offset) == {"A"}

    for fixture_id in (
        "gain_step_same_speaker",
        "noise_step_same_speaker",
        "channel_chirp_same_speaker",
    ):
        fixture = fixtures[fixture_id]
        event = fixture.event_samples[0]
        window_start = fixture.frontier_sample - fixture.window_samples
        assert window_start < event < fixture.frontier_sample
        assert fixture.nuisance_reference is not None
        assert np.array_equal(
            fixture.waveform[window_start:event],
            fixture.nuisance_reference[window_start:event],
        )
        assert not np.array_equal(
            fixture.waveform[event : fixture.frontier_sample],
            fixture.nuisance_reference[event : fixture.frontier_sample],
        )
        assert active(fixture_id, event - 1) == {"A"}
        assert active(fixture_id, event) == {"A"}


def test_valid_mean_pool_excludes_tail_and_l2_marks_zero_norm() -> None:
    values = np.asarray(
        [
            [[1.0, 3.0], [3.0, 5.0], [100.0, 100.0]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    pooled = mean_pool_valid(values, np.asarray([2, 3], dtype=np.int64))
    assert np.array_equal(pooled[0], np.asarray([2.0, 4.0], dtype=np.float32))
    normalized, valid = l2_normalize(pooled)
    assert valid.tolist() == [True, False]
    assert np.isclose(np.linalg.norm(normalized[0]), 1.0)
    assert np.isnan(normalized[1]).all()


def test_eres_analytic_lengths_cover_the_primary_window_grid() -> None:
    expected = {
        1600: {"S1": 8, "S2": 4, "S3": 2, "S4": 1, "FUSED": 1},
        3200: {"S1": 18, "S2": 9, "S3": 5, "S4": 3, "FUSED": 3},
        4800: {"S1": 28, "S2": 14, "S3": 7, "S4": 4, "FUSED": 4},
        8000: {"S1": 48, "S2": 24, "S3": 12, "S4": 6, "FUSED": 6},
        12000: {"S1": 73, "S2": 37, "S3": 19, "S4": 10, "FUSED": 10},
        16000: {"S1": 98, "S2": 49, "S3": 25, "S4": 13, "FUSED": 13},
    }
    assert {samples: _expected_eres_lengths(samples) for samples in expected} == expected


def test_ssl_independent_length_geometry_covers_the_primary_window_grid() -> None:
    assert {
        samples: _expected_ssl_length(samples) for samples in (1600, 3200, 4800, 8000, 12000, 16000)
    } == {1600: 4, 3200: 9, 4800: 14, 8000: 24, 12000: 37, 16000: 49}


def test_timestamp_mapping_uses_exact_window_frontier() -> None:
    fixture = d0_fixtures()[2]
    batch = ExtractionBatch(
        model_id="wavlm-base-plus",
        layers={"L6": np.zeros((1, 14, 2), dtype=np.float32)},
        valid_lengths={"L6": np.asarray([14], dtype=np.int64)},
        observed_source_samples=np.asarray([fixture.frontier_sample], dtype=np.int64),
    )
    mapping = _timestamp_mapping(batch, fixture)
    assert mapping["window_end_sample"] == fixture.frontier_sample
    assert mapping["representation_availability_source_sample"] == fixture.frontier_sample
    assert mapping["last_frame_support_samples"][1] <= fixture.frontier_sample
    assert mapping["next_hypothetical_frame_support_end_sample"] > fixture.frontier_sample
    assert mapping["passed"] is True


@pytest.mark.parametrize("writer", [write_acquisition_json, write_smoke_json])
def test_r1_evidence_writers_refuse_overwrite(tmp_path: Path, writer) -> None:
    path = tmp_path / "receipt.json"
    writer(path, {"schema_version": 1})
    with pytest.raises(R1GateError, match="refusing to overwrite"):
        writer(path, {"schema_version": 1})
