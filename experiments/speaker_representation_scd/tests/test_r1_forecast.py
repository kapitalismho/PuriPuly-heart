from __future__ import annotations

import json
import wave
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

import experiments.speaker_representation_scd.r1_forecast as forecast_module
from experiments.speaker_representation_scd.provenance import (
    load_json,
    sha256_file,
    with_self_sha256,
)
from experiments.speaker_representation_scd.r1_forecast import (
    CACHE_CALIBRATION_PATH,
    DEVELOPMENT_ACQUISITION_PATH,
    DEVELOPMENT_LEDGER_PATH,
    FORECAST_CONTRACT_PATH,
    FROZEN_INPUTS,
    MODEL_IDS,
    TECHNICAL_VALIDITY_PATH,
    WAVEFORM_INVENTORY_PATH,
    _calibration_errors,
    _development_acquisition_errors,
    _expected_coordinate_rows,
    _forecast_provenance,
    _ledger_errors,
    _safe_cache_path,
    build_forecast,
    main,
    validate_forecast_contract,
)
from experiments.speaker_representation_scd.r1_gate import EXPERIMENT_ROOT


def _technical() -> dict:
    return load_json(EXPERIMENT_ROOT / TECHNICAL_VALIDITY_PATH)


def _contract() -> dict:
    return load_json(EXPERIMENT_ROOT / FORECAST_CONTRACT_PATH)


def _write_json(path: Path, value: dict) -> dict:
    document = with_self_sha256(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    return document


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _materialize_development_inputs(
    cache_root: Path, eligible_samples: int = 16000
) -> tuple[dict, dict, dict]:
    contract = _contract()
    external_artifacts = []
    waveform_rows = []
    for source_id in forecast_module.DEVELOPMENT_SOURCE_IDS:
        path = cache_root / "sources" / "r2" / "development" / source_id / "archive.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source_id.encode("utf-8"))
        external_artifacts.append(
            {
                "source_id": source_id,
                "location": "cache_root",
                "relative_path": path.relative_to(cache_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
        waveform_path = path.with_name("audio.wav")
        with wave.open(str(waveform_path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(b"\0\0" * eligible_samples)
        external_artifacts.append(
            {
                "source_id": source_id,
                "location": "cache_root",
                "relative_path": waveform_path.relative_to(cache_root).as_posix(),
                "size_bytes": waveform_path.stat().st_size,
                "sha256": sha256_file(waveform_path),
            }
        )
        waveform_rows.append(
            {
                "waveform_id": source_id.replace("-", "_"),
                "source_id": source_id,
                "artifact_relative_to_cache_root": waveform_path.relative_to(
                    cache_root
                ).as_posix(),
                "artifact_sha256": sha256_file(waveform_path),
                "artifact_size_bytes": waveform_path.stat().st_size,
                "sample_rate_hz": 16000,
                "num_samples": eligible_samples,
                "eligible_start_sample": 0,
                "eligible_end_sample": eligible_samples,
            }
        )
    legacy = EXPERIMENT_ROOT.parents[1] / (
        "experiments/speaker_turn_boundary/results/turn_episode_v1/episode_manifest_dev.json"
    )
    artifacts = [
        {
            "source_id": "legacy-common-gt-v1",
            "location": "repository",
            "relative_path": (
                "experiments/speaker_turn_boundary/results/turn_episode_v1/episode_manifest_dev.json"
            ),
            "size_bytes": legacy.stat().st_size,
            "sha256": sha256_file(legacy),
        },
        *external_artifacts,
    ]
    waveform_inventory_path = cache_root / WAVEFORM_INVENTORY_PATH
    _write_rows(waveform_inventory_path, waveform_rows)
    acquisition = _write_json(
        cache_root / DEVELOPMENT_ACQUISITION_PATH,
        {
            "schema_version": 1,
            "artifact_role": "r2_development_acquisition_receipt",
            "experiment_id": "speaker_representation_scd_v1",
            "authority": forecast_module.AUTHORITY,
            "frozen_inputs": FROZEN_INPUTS,
            "development_source_ids": list(forecast_module.DEVELOPMENT_SOURCE_IDS),
            "free_bytes_before_download": 60 * 1024**3,
            "external_source_download_bytes": sum(
                row["size_bytes"] for row in external_artifacts
            ),
            "waveform_inventory": {
                "relative_to_cache_root": WAVEFORM_INVENTORY_PATH.as_posix(),
                "size_bytes": waveform_inventory_path.stat().st_size,
                "sha256": sha256_file(waveform_inventory_path),
            },
            "waveform_count": len(waveform_rows),
            "artifacts": artifacts,
        },
    )
    source_ids = list(forecast_module.DEVELOPMENT_SOURCE_IDS)
    coordinate_rows: list[dict] = []
    shards = []
    for waveform_row in waveform_rows:
        source_id = waveform_row["source_id"]
        rows = _expected_coordinate_rows(waveform_row)
        coordinate_rows.extend(rows)
        path = (
            cache_root
            / "data"
            / "r2"
            / "development"
            / "coordinates"
            / source_id
            / f"{waveform_row['waveform_id']}.jsonl"
        )
        _write_rows(path, rows)
        shards.append(
            {
                "source_id": source_id,
                "waveform_id": waveform_row["waveform_id"],
                "relative_to_cache_root": path.relative_to(cache_root).as_posix(),
                "row_count": len(rows),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    acquisition_path = cache_root / DEVELOPMENT_ACQUISITION_PATH
    ledger = _write_json(
        cache_root / DEVELOPMENT_LEDGER_PATH,
        {
            "schema_version": 1,
            "artifact_role": "r2_development_coordinate_ledger",
            "experiment_id": "speaker_representation_scd_v1",
            "authority": forecast_module.AUTHORITY,
            "frozen_inputs": FROZEN_INPUTS,
            "development_acquisition_receipt": {
                "relative_to_cache_root": DEVELOPMENT_ACQUISITION_PATH.as_posix(),
                "sha256": sha256_file(acquisition_path),
                "self_sha256": acquisition["self_sha256"],
            },
            "development_source_ids": source_ids,
            "extraction_windows_by_context_ms": {
                str(context): sum(
                    row["context_ms"] == context for row in coordinate_rows
                )
                for context in forecast_module.POOLING_MS
            },
            "extraction_windows_by_source_id": {
                source_id: sum(
                    row["source_id"] == source_id for row in coordinate_rows
                )
                for source_id in source_ids
            },
            "total_window_count": len(coordinate_rows),
            "coordinate_shards": shards,
        },
    )
    calibration_rows = []
    sample_rows = coordinate_rows[:2]
    for storage in contract["model_storage_contracts"]:
        model_id = storage["model_id"]
        root = cache_root / "cache" / "r2" / "development" / "calibration" / model_id
        root.mkdir(parents=True, exist_ok=True)
        npz_path = root / "pooled-vectors.npz"
        np.savez(
            npz_path,
            **{
                layer_id: np.zeros(
                    (len(sample_rows), storage["pooled_dimension_per_layer"]),
                    dtype=np.float32,
                )
                for layer_id in storage["retained_layer_ids"]
            },
        )
        manifest_path = root / "sample-manifest.jsonl"
        _write_rows(manifest_path, sample_rows)
        calibration_rows.append(
            {
                "model_id": model_id,
                "sample_coordinate_count": len(sample_rows),
                "artifact": {
                    "relative_to_cache_root": npz_path.relative_to(cache_root).as_posix(),
                    "size_bytes": npz_path.stat().st_size,
                    "sha256": sha256_file(npz_path),
                },
                "sample_manifest": {
                    "relative_to_cache_root": manifest_path.relative_to(cache_root).as_posix(),
                    "size_bytes": manifest_path.stat().st_size,
                    "sha256": sha256_file(manifest_path),
                },
                "serialized_file_bytes": npz_path.stat().st_size,
                "serialized_bytes_per_coordinate": (
                    npz_path.stat().st_size + len(sample_rows) - 1
                )
                // len(sample_rows),
            }
        )
    ledger_path = cache_root / DEVELOPMENT_LEDGER_PATH
    calibration = _write_json(
        cache_root / CACHE_CALIBRATION_PATH,
        {
            "schema_version": 1,
            "artifact_role": "r2_pooled_cache_calibration",
            "experiment_id": "speaker_representation_scd_v1",
            "authority": forecast_module.AUTHORITY,
            "frozen_inputs": FROZEN_INPUTS,
            "development_coordinate_ledger": {
                "relative_to_cache_root": DEVELOPMENT_LEDGER_PATH.as_posix(),
                "sha256": sha256_file(ledger_path),
                "self_sha256": ledger["self_sha256"],
            },
            "models": calibration_rows,
        },
    )
    return acquisition, ledger, calibration


def test_forecast_contract_is_valid() -> None:
    assert validate_forecast_contract(_contract()) == []


def test_public_cli_requires_cache_root_and_rejects_arbitrary_ledger() -> None:
    with pytest.raises(SystemExit):
        main([])
    with pytest.raises(SystemExit):
        main(["--cache-root", "C:/cache", "--ledger", "D:/d5.json"])


def test_public_cli_missing_inputs_is_not_ready(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    result = main(["--cache-root", str(tmp_path)])
    output = json.loads(capsys.readouterr().out)
    assert result == 2
    assert output["status"] == "not_ready"
    assert "development_acquisition: missing" in output["blockers"]
    assert "development_coordinate_ledger: missing" in output["blockers"]
    assert "cache_calibration: missing" in output["blockers"]


def test_public_cli_external_smoke_error_is_blocking(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        forecast_module,
        "validate_technical_validity",
        lambda document, cache_root: ["technical_validity: smoke byte identity mismatch"],
    )
    result = main(["--cache-root", str(tmp_path)])
    output = json.loads(capsys.readouterr().out)
    assert result == 2
    assert "technical_validity: smoke byte identity mismatch" in output["blockers"]


def test_development_inputs_require_actual_files_and_recounts(tmp_path: Path) -> None:
    acquisition, ledger, calibration = _materialize_development_inputs(tmp_path)
    acquisition_errors, _, waveforms = _development_acquisition_errors(
        acquisition, _contract(), tmp_path
    )
    ledger_errors, waveforms = _ledger_errors(
        ledger, _contract(), tmp_path, waveforms
    )
    calibration_errors, measured = _calibration_errors(
        calibration, _contract(), tmp_path, waveforms
    )
    assert acquisition_errors == []
    assert ledger_errors == []
    assert calibration_errors == []
    assert set(measured) == set(MODEL_IDS)


def test_coordinate_shard_tamper_is_rejected(tmp_path: Path) -> None:
    acquisition, ledger, _ = _materialize_development_inputs(tmp_path)
    _, _, waveforms = _development_acquisition_errors(
        acquisition, _contract(), tmp_path
    )
    shard = ledger["coordinate_shards"][0]
    path = tmp_path / shard["relative_to_cache_root"]
    path.write_text(path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    errors, _ = _ledger_errors(ledger, _contract(), tmp_path, waveforms)
    assert any("sha256 mismatch" in error for error in errors)
    assert any("row count mismatch" in error for error in errors)
    assert any("deterministic coordinate set differs" in error for error in errors)


def test_self_consistent_missing_coordinate_and_frontier_mutation_are_rejected(
    tmp_path: Path,
) -> None:
    acquisition, ledger, _ = _materialize_development_inputs(tmp_path)
    _, _, waveforms = _development_acquisition_errors(
        acquisition, _contract(), tmp_path
    )
    shard = ledger["coordinate_shards"][0]
    path = tmp_path / shard["relative_to_cache_root"]
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    removed = rows.pop()
    _write_rows(path, rows)
    shortened = deepcopy(ledger)
    shortened_shard = shortened["coordinate_shards"][0]
    shortened_shard["row_count"] = len(rows)
    shortened_shard["size_bytes"] = path.stat().st_size
    shortened_shard["sha256"] = sha256_file(path)
    shortened["extraction_windows_by_context_ms"][str(removed["context_ms"])] -= 1
    shortened["extraction_windows_by_source_id"][removed["source_id"]] -= 1
    shortened["total_window_count"] -= 1
    shortened = with_self_sha256(shortened)
    missing_errors, _ = _ledger_errors(
        shortened, _contract(), tmp_path, waveforms
    )
    assert any("deterministic coordinate set differs" in error for error in missing_errors)
    original = _materialize_development_inputs(tmp_path / "frontier")
    frontier_acquisition, frontier_ledger, _ = original
    _, _, frontier_waveforms = _development_acquisition_errors(
        frontier_acquisition, _contract(), tmp_path / "frontier"
    )
    frontier_shard = frontier_ledger["coordinate_shards"][0]
    frontier_path = tmp_path / "frontier" / frontier_shard["relative_to_cache_root"]
    frontier_rows = [
        json.loads(line)
        for line in frontier_path.read_text(encoding="utf-8").splitlines()
    ]
    first = frontier_rows[0]
    waveform = frontier_waveforms[first["waveform_id"]]
    frontier_rows[0] = forecast_module._coordinate_row(
        waveform, first["context_ms"], first["observed_frontier_sample"] + 1
    )
    _write_rows(frontier_path, frontier_rows)
    mutated = deepcopy(frontier_ledger)
    mutated_shard = mutated["coordinate_shards"][0]
    mutated_shard["size_bytes"] = frontier_path.stat().st_size
    mutated_shard["sha256"] = sha256_file(frontier_path)
    mutated = with_self_sha256(mutated)
    frontier_errors, _ = _ledger_errors(
        mutated, _contract(), tmp_path / "frontier", frontier_waveforms
    )
    assert any("deterministic coordinate set differs" in error for error in frontier_errors)


def test_unlisted_source_file_and_legacy_identity_mutation_are_rejected(
    tmp_path: Path,
) -> None:
    acquisition, _, _ = _materialize_development_inputs(tmp_path)
    extra = (
        tmp_path
        / "sources"
        / "r2"
        / "development"
        / "jvs-development"
        / "unlisted.bin"
    )
    extra.write_bytes(b"unlisted")
    mutated = deepcopy(acquisition)
    mutated["artifacts"][0]["sha256"] = "0" * 64
    mutated = with_self_sha256(mutated)
    errors, _, _ = _development_acquisition_errors(mutated, _contract(), tmp_path)
    assert "development_acquisition: source file inventory differs" in errors
    assert any("frozen legacy identity differs" in error for error in errors)


def test_acquired_canonical_waveform_cannot_be_omitted_from_inventory(
    tmp_path: Path,
) -> None:
    acquisition, _, _ = _materialize_development_inputs(tmp_path)
    source_id = "jvs-development"
    original = (
        tmp_path / "sources" / "r2" / "development" / source_id / "audio.wav"
    )
    extra = original.with_name("omitted.wav")
    extra.write_bytes(original.read_bytes())
    mutated = deepcopy(acquisition)
    mutated["artifacts"].append(
        {
            "source_id": source_id,
            "location": "cache_root",
            "relative_path": extra.relative_to(tmp_path).as_posix(),
            "size_bytes": extra.stat().st_size,
            "sha256": sha256_file(extra),
        }
    )
    mutated["external_source_download_bytes"] += extra.stat().st_size
    mutated = with_self_sha256(mutated)
    errors, _, _ = _development_acquisition_errors(mutated, _contract(), tmp_path)
    assert "development_acquisition: canonical waveform coverage differs" in errors


def test_npz_tamper_and_manifest_tamper_are_rejected(tmp_path: Path) -> None:
    acquisition, ledger, calibration = _materialize_development_inputs(tmp_path)
    _, _, waveforms = _development_acquisition_errors(
        acquisition, _contract(), tmp_path
    )
    _, waveforms = _ledger_errors(ledger, _contract(), tmp_path, waveforms)
    first = calibration["models"][0]
    npz_path = tmp_path / first["artifact"]["relative_to_cache_root"]
    npz_path.write_bytes(npz_path.read_bytes() + b"tamper")
    manifest_path = tmp_path / first["sample_manifest"]["relative_to_cache_root"]
    manifest_coordinate = json.loads(
        manifest_path.read_text(encoding="utf-8").splitlines()[0]
    )["coordinate_id"]
    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8").replace(
            manifest_coordinate, "unknown-coordinate"
        ),
        encoding="utf-8",
    )
    errors, _ = _calibration_errors(calibration, _contract(), tmp_path, waveforms)
    assert any("NPZ identity mismatch" in error for error in errors)
    assert any("sample manifest identity mismatch" in error for error in errors)
    assert any("sample manifest coordinates invalid" in error for error in errors)


def test_d5_and_alias_paths_are_rejected_before_read(tmp_path: Path) -> None:
    _, errors = _safe_cache_path(
        tmp_path,
        "data/r2/confirmatory/secret.jsonl",
        Path("data/r2/development/coordinates/legacy-common-gt-v1"),
        "ledger",
    )
    assert errors == ["ledger: outside development namespace"]


def test_forecast_candidate_uses_conservative_runtime_and_full_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    acquisition, ledger, calibration = _materialize_development_inputs(tmp_path)
    monkeypatch.setattr(
        forecast_module, "validate_technical_validity", lambda document, cache_root: []
    )
    forecast = build_forecast(
        _technical(),
        _contract(),
        tmp_path,
        acquisition,
        ledger,
        calibration,
        _forecast_provenance(("test",)),
    )
    assert forecast["status"] == "ceiling_pass_candidate"
    assert forecast["inference_window_count_per_model"] == ledger["total_window_count"]
    assert forecast["job_count"] == ledger["total_window_count"] * 4
    assert all(
        row["verified_worst_case_seconds_per_window"]
        == row["measured_balanced_seconds_per_window"] * 10
        for row in forecast["models"]
    )
    assert forecast["total_projected_external_storage_bytes"] == (
        forecast["current_external_root_bytes"]
        + forecast["total_projected_cache_bytes"]
    )
    assert all(forecast["ceiling_checks"].values())
    assert not forecast["forecast_approved"]
    assert not forecast["full_extraction_enabled"]
    assert forecast["forecast_provenance"]["calculator"]["sha256"] == sha256_file(
        EXPERIMENT_ROOT / "r1_forecast.py"
    )


def test_forecast_rejects_mutated_calculator_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    acquisition, ledger, calibration = _materialize_development_inputs(tmp_path)
    monkeypatch.setattr(
        forecast_module, "validate_technical_validity", lambda document, cache_root: []
    )
    provenance = _forecast_provenance(("test",))
    provenance["calculator"]["sha256"] = "0" * 64
    result = build_forecast(
        _technical(),
        _contract(),
        tmp_path,
        acquisition,
        ledger,
        calibration,
        provenance,
    )
    assert result["status"] == "not_ready"
    assert "forecast_provenance: calculator identity differs" in result["blockers"]


def test_context_skew_does_not_reduce_runtime_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline_root = tmp_path / "baseline"
    skewed_root = tmp_path / "skewed"
    baseline_inputs = _materialize_development_inputs(baseline_root)
    skewed_inputs = _materialize_development_inputs(skewed_root, 80000)
    monkeypatch.setattr(
        forecast_module, "validate_technical_validity", lambda document, cache_root: []
    )
    baseline = build_forecast(
        _technical(),
        _contract(),
        baseline_root,
        *baseline_inputs,
        _forecast_provenance(("baseline",)),
    )
    skewed = build_forecast(
        _technical(),
        _contract(),
        skewed_root,
        *skewed_inputs,
        _forecast_provenance(("skewed",)),
    )
    assert [
        row["verified_worst_case_seconds_per_window"] for row in baseline["models"]
    ] == [
        row["verified_worst_case_seconds_per_window"] for row in skewed["models"]
    ]
    assert skewed["inference_window_count_per_model"] > baseline[
        "inference_window_count_per_model"
    ]


def test_rehashed_scope_ceiling_or_frozen_input_mutation_is_rejected() -> None:
    scope_mutation = deepcopy(_contract())
    scope_mutation["development_scope"]["required_source_ids"] = [
        "legacy-common-gt-v1"
    ]
    assert "forecast_contract: development source scope differs" in validate_forecast_contract(
        with_self_sha256(scope_mutation)
    )
    ceiling_mutation = deepcopy(_contract())
    ceiling_mutation["ceilings"]["max_total_wall_hours"] = 960
    assert "forecast_contract: ceilings differ" in validate_forecast_contract(
        with_self_sha256(ceiling_mutation)
    )
    frozen_mutation = deepcopy(_contract())
    frozen_mutation["frozen_inputs"]["source_ledger"]["sha256"] = "0" * 64
    assert "frozen_inputs: identities differ" in validate_forecast_contract(
        with_self_sha256(frozen_mutation)
    )
    technical_path_mutation = deepcopy(_contract())
    technical_path_mutation["technical_validity_receipt"]["path"] = (
        "../../sealed/d5.json"
    )
    technical_errors = validate_forecast_contract(
        with_self_sha256(technical_path_mutation)
    )
    assert "forecast_contract: technical receipt identity differs" in technical_errors
    assert not any("sealed" in error for error in technical_errors)
