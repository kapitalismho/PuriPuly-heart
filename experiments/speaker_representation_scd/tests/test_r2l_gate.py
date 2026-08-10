from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

import experiments.speaker_representation_scd.r2l_forecast as forecast_module
import experiments.speaker_representation_scd.r2l_gate as gate_module
from experiments.speaker_representation_scd.provenance import with_self_sha256
from experiments.speaker_representation_scd.r1_gate import EXPERIMENT_ROOT, REPOSITORY_ROOT
from experiments.speaker_representation_scd.r2l_forecast import build_reduced_forecast
from experiments.speaker_representation_scd.r2l_gate import (
    EXPECTED_ACTIONS,
    GATE_PATH,
    validate_r2l_gate,
)
from experiments.speaker_representation_scd.r2l_materialize import (
    MAX_R4_SOURCE_SECONDS,
    R2LValidationError,
    _classify_window,
    _coordinate_row,
    freeze_r4_panel,
    generate_coordinates,
    load_legacy_documents,
)


def test_r2l_gate_is_valid_without_external_execution() -> None:
    result = validate_r2l_gate(scan_processes=False)
    assert result.valid, result.errors
    assert result.allowed_actions == EXPECTED_ACTIONS
    assert result.allowed_actions["legacy_validation"] is True
    assert result.allowed_actions["coordinate_materialization"] is True
    assert result.allowed_actions["full_extraction"] is False
    assert result.allowed_actions["confirmatory_access"] is False
    assert result.allowed_actions["training"] is False


def test_rehashed_semantic_gate_mutation_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    original_loader = gate_module.load_json
    gate = original_loader(EXPERIMENT_ROOT / GATE_PATH)
    mutated = deepcopy(gate)
    mutated["authorization"]["confirmatory_access"] = True
    mutated = with_self_sha256(mutated)

    def load(path: Path) -> dict:
        if path.resolve() == (EXPERIMENT_ROOT / GATE_PATH).resolve():
            return mutated
        return original_loader(path)

    monkeypatch.setattr(gate_module, "load_json", load)
    result = validate_r2l_gate(scan_processes=False)
    assert not result.valid
    assert "r2l_gate.authorization: differs" in result.errors


def test_manifest_byte_identity_drift_fails_closed(tmp_path: Path) -> None:
    manifest = (
        tmp_path
        / "experiments"
        / "speaker_turn_boundary"
        / "results"
        / "turn_episode_v1"
        / "episode_manifest_dev.json"
    )
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")
    with pytest.raises(R2LValidationError, match="byte identity differs"):
        load_legacy_documents(tmp_path)


def test_derive_candidates_reproduces_frozen_digests() -> None:
    pytest.importorskip("janus")
    documents = load_legacy_documents(REPOSITORY_ROOT)
    derived = None
    try:
        derived = __import__(
            "experiments.speaker_representation_scd.r2l_materialize", fromlist=["derive_candidates"]
        ).derive_candidates(documents)
    finally:
        assert derived is not None
    assert len(derived["positives"]) == 450
    assert len(derived["negatives"]) == 360
    assert len(derived["pairs"]) == 313
    assert derived["exclusions"] == documents["design_ledger"]["matching"]["exclusions"]
    assert len(derived["episodes"]) == 695


def test_freeze_r4_panel_respects_cap_and_deterministic_order() -> None:
    import hashlib

    rows = [
        {
            "session_id": "b-session",
            "stratum": "b",
            "duration_seconds": 3 * 3600,
        },
        {
            "session_id": "a-session",
            "stratum": "a",
            "duration_seconds": 3 * 3600,
        },
        {
            "session_id": "c-session",
            "stratum": "b",
            "duration_seconds": 3 * 3600,
        },
    ]

    def key(row: dict) -> tuple[int, str]:
        return (
            0 if row["stratum"] == "a" else 1,
            hashlib.sha256(str(row["session_id"]).encode("utf-8")).hexdigest(),
        )

    expected_order = [row["session_id"] for row in sorted(rows, key=key)]
    panel, excluded, total_seconds = freeze_r4_panel(rows)
    assert [row["session_id"] for row in panel] == expected_order[:2]
    assert [row["session_id"] for row in excluded] == expected_order[2:]
    assert total_seconds <= MAX_R4_SOURCE_SECONDS
    assert excluded[0]["reason"] == "r4_source_hour_cap"


def test_coordinate_rows_are_bounded_and_deterministic() -> None:
    row = _coordinate_row(
        "ls_dev:case_00",
        "a" * 64,
        300,
        24000,
        "r3_primary",
        "positive:test",
        None,
        "entirely_new",
    )
    assert len(json.dumps(row, sort_keys=True).encode("utf-8")) <= 1024
    repeat = _coordinate_row(
        "ls_dev:case_00",
        "a" * 64,
        300,
        24000,
        "r3_primary",
        "positive:test",
        None,
        "entirely_new",
    )
    assert row["coordinate_id"] == repeat["coordinate_id"]
    assert row["scope"] == "legacy-common-gt-v1"
    assert row["observed_frontier_sample"] == 24000
    assert row["window_start_sample"] == 24000 - 4800


def test_classify_window_labels() -> None:
    regions = [
        {"start_sample": 0, "end_sample": 16000, "speakers": [], "ambiguous": False},
        {"start_sample": 16000, "end_sample": 48000, "speakers": ["A"], "ambiguous": False},
        {"start_sample": 48000, "end_sample": 64000, "speakers": ["A", "B"], "ambiguous": False},
    ]
    assert _classify_window(4000, 8000, 32000, regions, "positive") == "silence"
    assert _classify_window(16000, 24000, 32000, regions, "positive") == "entirely_old"
    assert _classify_window(16000, 32000, 32000, regions, "positive") == "entirely_old"
    assert _classify_window(32000, 40000, 32000, regions, "positive") == "entirely_new"
    assert _classify_window(30000, 34000, 32000, regions, "positive") == "boundary_straddling"
    assert _classify_window(48000, 56000, 32000, regions, "positive") == "overlap"
    assert _classify_window(30000, 34000, 32000, regions, "negative") == "stable_same_speaker"
    assert _classify_window(48000, 56000, 32000, regions, "negative") == "overlap"


def test_generate_coordinates_roles_and_exclusions() -> None:
    documents = {"cases": {}}
    derived = {
        "positives": [
            {
                "candidate_id": "positive:p1",
                "class": "positive",
                "session_id": "ls_dev:case_00",
                "wav_sha256": "w" * 64,
                "coordinate": 16000,
            }
        ],
        "negatives": [],
    }
    waveforms = {
        "w" * 64: {
            "waveform_id": "w" * 64,
            "eligible_start_sample": 0,
            "eligible_end_sample": 64000,
            "session_ids": ["ls_dev:case_00"],
        }
    }
    source_rows = [
        {
            "session_id": "ls_dev:case_00",
            "waveform_id": "w" * 64,
            "eligible_start_sample": 0,
            "eligible_end_sample": 64000,
        }
    ]
    regions = {"ls_dev:case_00": []}
    rows_by_waveform, counts, excluded = generate_coordinates(
        documents, derived, waveforms, source_rows, regions
    )
    assert counts["r3_primary"] == 3
    assert counts["r4_counts"] == {"100": 40, "300": 38, "500": 36}
    assert len(excluded) == 5
    assert all(row["reason"] == "trajectory_window_out_of_eligible_range" for row in excluded)
    roles = {row["coordinate_role"] for row in rows_by_waveform["w" * 64]}
    assert roles == {"r3_primary", "r3_trajectory", "r4_continuous"}
    trajectory = [
        row for row in rows_by_waveform["w" * 64] if row["coordinate_role"] == "r3_trajectory"
    ]
    assert all(row["trajectory_offset_ms"] is not None for row in trajectory)


def _fake_technical() -> dict:
    rows = []
    for index, model_id in enumerate(
        ("mhubert-147", "wavlm-base-plus", "unispeech-sat-base-plus", "eres2netv2-standard-prepool")
    ):
        rows.append(
            {
                "model_id": model_id,
                "sha256": "a" * 64,
                "self_sha256": "b" * 64,
                "execution_id": f"{index:032d}",
                "usage_self_sha256": "c" * 64,
                "single_seconds_per_window": 0.02 + index * 0.001,
                "batch_seconds_per_window": 0.01 + index * 0.0005,
                "cold_load_seconds": 3.0,
                "authoritative_peak_job_memory_bytes": 1_600_000_000,
            }
        )
    return {
        "environment_receipt": {"relative_to_cache_root": "manifests/r1_environment_sync.json"},
        "model_acquisition_receipt": {"relative_to_cache_root": "manifests/r1_model_acquisition.json"},
        "model_smoke_reports": rows,
    }


def _fake_ledger() -> dict:
    return with_self_sha256(
        {
            "schema_version": 1,
            "artifact_role": "r2l_legacy_common_gt_coordinate_ledger",
            "experiment_id": "speaker_representation_scd_v1",
            "authority": forecast_module.AUTHORITY,
            "scope": "legacy-common-gt-v1",
            "r3": {
                "positive_anchor_count": 450,
                "negative_anchor_count": 360,
                "primary_window_count": 2379,
                "trajectory_window_count": 31567,
            },
            "r4": {
                "panel_source_count": 1,
                "panel_total_source_hours": 0.5,
                "panel_sources": [
                    {
                        "session_id": "ls_dev:case_00",
                        "synthetic_manifest": "ls_dev",
                        "stratum": "a",
                        "eligible_start_sample": 0,
                        "eligible_end_sample": 32000,
                    }
                ],
                "windows_by_context_ms": {"100": 190, "300": 170, "500": 150},
            },
        }
    )


def test_reduced_forecast_rejects_invalid_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(forecast_module, "validate_technical_validity", lambda *args: [])
    monkeypatch.setattr(forecast_module, "_cache_root_size", lambda *args: (0, []))
    monkeypatch.setattr(forecast_module, "sha256_file", lambda *args: "f" * 64)
    contract = forecast_module.load_json(EXPERIMENT_ROOT / forecast_module.FORECAST_CONTRACT_PATH)
    ledger = _fake_ledger()
    ledger["r3"]["positive_anchor_count"] = 100
    ledger = with_self_sha256(ledger)
    provenance = forecast_module.forecast_provenance(("r2l_forecast", "--cache-root", "x"))
    forecast = build_reduced_forecast(_fake_technical(), contract, ledger, EXPERIMENT_ROOT, provenance)
    assert forecast["status"] == "not_ready"
    assert any("r3 anchor counts differ" in blocker for blocker in forecast["blockers"])


def test_reduced_forecast_ceiling_math_and_supervision_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(forecast_module, "validate_technical_validity", lambda *args: [])
    monkeypatch.setattr(forecast_module, "_cache_root_size", lambda *args: (0, []))
    monkeypatch.setattr(forecast_module, "sha256_file", lambda *args: "f" * 64)
    contract = forecast_module.load_json(EXPERIMENT_ROOT / forecast_module.FORECAST_CONTRACT_PATH)
    ledger = _fake_ledger()
    provenance = forecast_module.forecast_provenance(("r2l_forecast", "--cache-root", "x"))
    binding = {
        "execution_id": "a" * 32,
        "expected_receipt_relative_path": "manifests/r2/legacy_common_gt/validation_receipt.json",
        "authority": "requires_completed_usage_attestation",
    }
    forecast = build_reduced_forecast(
        _fake_technical(),
        contract,
        ledger,
        EXPERIMENT_ROOT,
        provenance,
        supervision_binding=binding,
    )
    assert forecast["status"] == "ceiling_pass_candidate_conservative"
    assert forecast["supervision_binding"] == binding
    assert forecast["r3"]["primary_window_count"] == 2379
    assert forecast["r4"]["primary_window_count"] == 170
    assert forecast["sensitivity"]["top_two_model_ids"]
    assert forecast["scope"] == "legacy-common-gt-v1"
    assert forecast["coordinate_ledger"]["self_sha256"] == ledger["self_sha256"]
    assert forecast["forecast_approved"] is False


def test_gate_file_has_valid_json_identity() -> None:
    gate = json.loads((EXPERIMENT_ROOT / GATE_PATH).read_text(encoding="utf-8"))
    assert gate["self_sha256"] != "pending"
