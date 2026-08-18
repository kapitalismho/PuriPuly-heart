from __future__ import annotations

import json
import wave
from dataclasses import replace
from pathlib import Path

import pytest

from experiments.psem_training_strategy_gate.data import topology_census
from experiments.psem_training_strategy_gate.data.annotation_normalization import (
    NormalizedSession,
)
from experiments.psem_training_strategy_gate.data.label_contract import (
    CanonicalInterval,
    generate_labels,
    load_contract,
)
from experiments.psem_training_strategy_gate.data.provenance import (
    EXPECTED_ALIMEETING_MEETINGS,
    EXPECTED_AMI_MEETINGS,
    canonical_sha256,
    sha256_file,
    wav_identity,
    write_jsonl,
)
from experiments.psem_training_strategy_gate.data.topology_census import (
    FROZEN_CALIBRATION_MARKDOWN_SHA256,
    FROZEN_CALIBRATION_SOURCE_IDS,
    OFFICIAL_PRIMARY_TOPOLOGIES,
    TopologyCensusError,
    _build_topology_census_from_validated_sessions,
    _validate_calibrated_inventory,
    build_topology_census,
    build_topology_row,
    render_data_census,
    validate_waveform_inventory,
    write_topology_census,
)

DATA_DIR = Path(__file__).resolve().parents[1]
EXPECTED_SOURCE_COUNT = len(EXPECTED_AMI_MEETINGS) + len(EXPECTED_ALIMEETING_MEETINGS)


def _sessions(intervals: tuple[CanonicalInterval, ...]) -> list[NormalizedSession]:
    labels = generate_labels(
        intervals,
        scored_start_sample=intervals[0].start_sample,
        scored_end_sample=intervals[-1].end_sample,
    )
    return [
        NormalizedSession(
            source_id=f"{prefix}_session",
            corpus=corpus,
            session_id=f"{prefix}_session",
            scored_start_sample=intervals[0].start_sample,
            scored_end_sample=intervals[-1].end_sample,
            source_waveform_sha256="a" * 64,
            annotation_sha256="b" * 64,
            raw_speech_span_count=len(intervals),
            clipped_span_count=0,
            intervals=intervals,
            labels=labels,
        )
        for prefix, corpus in (("ami", "AMI"), ("alimeeting", "AliMeeting"))
    ]


def _write_inputs(data_dir: Path, sessions: list[NormalizedSession]) -> None:
    ordered = sorted(sessions, key=lambda session: session.source_id)
    source_ids = [session.source_id for session in ordered]
    write_jsonl(
        data_dir / "source_manifest.jsonl",
        ({"source_id": session.source_id} for session in ordered),
    )
    write_jsonl(
        data_dir / "annotation_manifest.jsonl",
        ({"source_id": session.source_id} for session in ordered),
    )
    write_jsonl(
        data_dir / "normalization_manifest.jsonl",
        (session.manifest_row() for session in ordered),
    )
    contract = load_contract()
    calibration = {
        "artifact_role": "annotation_only_calibration",
        "contract_version": contract.contract_version,
        "contract_document_sha256": contract.document_sha256,
        "contract_status": contract.status,
        "input_policy": {
            "source_manifest_sha256": sha256_file(
                data_dir / "source_manifest.jsonl"
            ),
            "annotation_manifest_sha256": sha256_file(
                data_dir / "annotation_manifest.jsonl"
            ),
            "normalization_manifest_sha256": sha256_file(
                data_dir / "normalization_manifest.jsonl"
            ),
            "source_ids_sha256": canonical_sha256(source_ids),
            "model_predictions_consulted": False,
            "model_scores_consulted": False,
            "official_model_results_inspected": False,
            "official_model_training_performed": False,
        },
    }
    (data_dir / "annotation_calibration.json").write_text(
        json.dumps(calibration, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (data_dir / "ANNOTATION_CALIBRATION.md").write_text(
        "fixture calibration\n", encoding="utf-8", newline="\n"
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_checked_in_topology_census_binds_the_frozen_inventory() -> None:
    rows = _read_jsonl(DATA_DIR / "topology_manifest.jsonl")
    census = json.loads(
        (DATA_DIR / "topology_census.json").read_text(encoding="utf-8")
    )
    contract = load_contract()
    assert len(rows) == EXPECTED_SOURCE_COUNT
    assert len({row["source_id"] for row in rows}) == EXPECTED_SOURCE_COUNT
    assert {row["corpus"] for row in rows} == {"AMI", "AliMeeting"}
    assert census["contract_version"] == contract.contract_version
    assert census["contract_document_sha256"] == contract.document_sha256
    assert census["contract_status"] == "frozen_after_annotation_only_calibration"
    assert census["topology_manifest_sha256"] == sha256_file(
        DATA_DIR / "topology_manifest.jsonl"
    )
    assert census["input_manifests"][
        "annotation_calibration_sha256"
    ] == sha256_file(DATA_DIR / "annotation_calibration.json")
    assert census["input_manifests"][
        "annotation_calibration_markdown_sha256"
    ] == FROZEN_CALIBRATION_MARKDOWN_SHA256
    assert census["input_manifests"]["source_manifest_sha256"] == sha256_file(
        DATA_DIR / "source_manifest.jsonl"
    )
    assert census["input_manifests"]["annotation_manifest_sha256"] == sha256_file(
        DATA_DIR / "annotation_manifest.jsonl"
    )
    assert census["input_manifests"][
        "normalization_manifest_sha256"
    ] == sha256_file(DATA_DIR / "normalization_manifest.jsonl")
    assert census["input_manifests"]["source_ids_sha256"] == canonical_sha256(
        sorted(row["source_id"] for row in rows)
    )
    assert census["overall"]["session_count"] == EXPECTED_SOURCE_COUNT
    assert census["overall"]["scored_samples"] == sum(
        row["scored_samples"] for row in rows
    )
    assert census["overall"]["exclusive_primary_episode_count"] == sum(
        census["overall"]["primary_topology_counts"].values()
    )
    assert census["counting_policy"]["exclusive_primary_counting"] is True
    assert census["counting_policy"]["old_r7_or_r7b_event_counts_used"] is False
    assert census["model_policy"]["model_predictions_consulted"] is False
    assert census["model_policy"]["model_scores_consulted"] is False
    assert census["model_policy"]["official_model_training_performed"] is False
    assert census["overall"]["mask_diagnostics"][
        "masked_transition_fraction"
    ] == 0.33906608
    markdown = (DATA_DIR / "DATA_CENSUS.md").read_text(encoding="utf-8")
    split_manifest = json.loads(
        (DATA_DIR / "split_manifest.json").read_text(encoding="utf-8")
    )
    assert markdown == render_data_census(
        census,
        rows,
        split_manifest,
        sha256_file(DATA_DIR / "split_manifest.json"),
    )
    assert "component audit pending" not in markdown
    assert "passes all 22 role-specific hard gates" in markdown
    assert "No topology substitutes for another" in markdown
    assert "Stable singleton hours" in markdown
    assert "Ongoing overlap minutes" in markdown
    assert all(topology in markdown for topology in OFFICIAL_PRIMARY_TOPOLOGIES)


def test_short_backchannel_is_one_exclusive_primary_episode(tmp_path: Path) -> None:
    intervals = (
        CanonicalInterval(0, 3200, ("A",), source_annotation_ids=("a",)),
        CanonicalInterval(3200, 9600, ("B",), source_annotation_ids=("b",)),
        CanonicalInterval(9600, 12800, ("A",), source_annotation_ids=("c",)),
    )
    sessions = _sessions(intervals)
    _write_inputs(tmp_path, sessions)
    rows, census = _build_topology_census_from_validated_sessions(
        sessions, tmp_path, "c" * 64
    )
    for row in rows:
        assert row["exclusive_primary_episode_count"] == 1
        assert row["primary_topology_counts"]["short_backchannel_return"] == 1
        assert (
            row["primary_topology_counts"][
                "clean_direct_different_speaker_handoff"
            ]
            == 0
        )
        assert sum(row["primary_topology_counts"].values()) == 1
    assert census["overall"]["primary_topology_counts"][
        "short_backchannel_return"
    ] == 2
    assert census["counting_policy"][
        "short_backchannel_member_handoffs_counted_separately"
    ] is False


def test_rendered_census_does_not_report_a_zero_hour_deficit_as_blocking() -> None:
    census = json.loads(
        (DATA_DIR / "topology_census.json").read_text(encoding="utf-8")
    )
    rows = _read_jsonl(DATA_DIR / "topology_manifest.jsonl")
    census["candidate_pool_lower_bound_audit"]["scored_samples"]["deficit"] = 0
    markdown = render_data_census(census, rows)
    assert "aggregate scored-hour lower bound passes" in markdown
    assert "scored-hour deficit is an acquisition blocker" not in markdown


def test_frozen_calibration_scope_allows_expansion_but_not_omission() -> None:
    contract = load_contract()
    expanded = sorted([*FROZEN_CALIBRATION_SOURCE_IDS, "ami_new_session"])
    _validate_calibrated_inventory(DATA_DIR, expanded, contract)
    missing = sorted(FROZEN_CALIBRATION_SOURCE_IDS)[1:]
    with pytest.raises(TopologyCensusError, match="accepted annotation calibration"):
        _validate_calibrated_inventory(DATA_DIR, missing, contract)


def test_micro_diagnostics_are_not_official_primary_counts() -> None:
    intervals = (
        CanonicalInterval(0, 3200, ("A",), source_annotation_ids=("a",)),
        CanonicalInterval(3200, 4320, (), source_annotation_ids=()),
        CanonicalInterval(4320, 7520, ("B",), source_annotation_ids=("b",)),
        CanonicalInterval(
            7520, 8640, ("B", "C"), source_annotation_ids=("c",)
        ),
        CanonicalInterval(8640, 11840, ("B",), source_annotation_ids=("d",)),
    )
    row = build_topology_row(_sessions(intervals)[0], load_contract())
    assert row["micro_diagnostics"] == {
        "micro_gap_interval_count": 1,
        "micro_gap_samples": 1120,
        "micro_overlap_interval_count": 1,
        "micro_overlap_samples": 1120,
    }
    assert set(row["primary_topology_counts"]) == set(
        OFFICIAL_PRIMARY_TOPOLOGIES
    )
    assert "micro_gap_different_speaker_handoff" not in row[
        "primary_topology_counts"
    ]
    assert "micro_overlap_return" not in row["primary_topology_counts"]


def test_masked_transitions_and_diagnostic_regions_are_separate() -> None:
    intervals = (
        CanonicalInterval(0, 3200, ("A",), source_annotation_ids=("a",)),
        CanonicalInterval(
            3200, 4800, ("A", "B", "C"), source_annotation_ids=("b",)
        ),
        CanonicalInterval(4800, 8000, ("B",), source_annotation_ids=("c",)),
    )
    diagnostics = build_topology_row(
        _sessions(intervals)[0], load_contract()
    )["mask_diagnostics"]
    assert diagnostics["actual_transition_count"] == 2
    assert diagnostics["masked_transition_count"] == 1
    assert diagnostics["masked_transition_fraction"] == 0.5
    assert diagnostics["masked_transition_reasons"] == {
        "complex_overlap_transition": 1
    }
    assert diagnostics["diagnostic_masked_region_counts"] == {
        "complex_overlap_region": 1
    }


def test_census_rejects_a_stale_normalization_manifest(tmp_path: Path) -> None:
    intervals = (
        CanonicalInterval(0, 3200, ("A",), source_annotation_ids=("a",)),
        CanonicalInterval(3200, 6400, ("B",), source_annotation_ids=("b",)),
    )
    sessions = _sessions(intervals)
    _write_inputs(tmp_path, sessions)
    stale_rows = [
        session.manifest_row()
        for session in sorted(sessions, key=lambda session: session.source_id)
    ]
    stale_rows[0]["label_result_sha256"] = "0" * 64
    write_jsonl(tmp_path / "normalization_manifest.jsonl", stale_rows)
    with pytest.raises(
        TopologyCensusError,
        match="normalization manifest does not match censused sessions",
    ):
        build_topology_census(sessions, tmp_path, "c" * 64)


def test_census_rejects_a_partial_calibrated_inventory(tmp_path: Path) -> None:
    intervals = (
        CanonicalInterval(0, 3200, ("A",), source_annotation_ids=("a",)),
        CanonicalInterval(3200, 6400, ("B",), source_annotation_ids=("b",)),
    )
    sessions = _sessions(intervals)
    _write_inputs(tmp_path, sessions)
    selected = sessions[:1]
    write_jsonl(
        tmp_path / "source_manifest.jsonl",
        ({"source_id": selected[0].source_id},),
    )
    write_jsonl(
        tmp_path / "annotation_manifest.jsonl",
        ({"source_id": selected[0].source_id},),
    )
    write_jsonl(
        tmp_path / "normalization_manifest.jsonl",
        (selected[0].manifest_row(),),
    )
    with pytest.raises(
        TopologyCensusError,
        match="accepted annotation calibration",
    ):
        build_topology_census(selected, tmp_path, "c" * 64)


def test_waveform_inventory_rejects_replaced_audio(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    waveform_path = corpus_root / "audio.wav"
    with wave.open(str(waveform_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * 100)
    identity = wav_identity(waveform_path)
    write_jsonl(
        tmp_path / "source_manifest.jsonl",
        ({"source_id": "session", "audio_ref": "audio.wav", **identity},),
    )
    validate_waveform_inventory(tmp_path, corpus_root)
    with wave.open(str(waveform_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\1\0" * 100)
    with pytest.raises(TopologyCensusError, match="identity mismatch"):
        validate_waveform_inventory(tmp_path, corpus_root)


def test_topology_row_rejects_a_stale_label_contract() -> None:
    intervals = (
        CanonicalInterval(0, 3200, ("A",), source_annotation_ids=("a",)),
        CanonicalInterval(3200, 6400, ("B",), source_annotation_ids=("b",)),
    )
    session = _sessions(intervals)[0]
    stale_session = replace(
        session,
        labels=replace(session.labels, contract_document_sha256="0" * 64),
    )
    with pytest.raises(TopologyCensusError, match="session label contract mismatch"):
        build_topology_row(stale_session, load_contract())


def test_census_outputs_are_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intervals = (
        CanonicalInterval(0, 3200, ("A",), source_annotation_ids=("a",)),
        CanonicalInterval(3200, 6400, ("B",), source_annotation_ids=("b",)),
    )
    sessions = _sessions(intervals)
    _write_inputs(tmp_path, sessions)
    monkeypatch.setattr(topology_census, "normalize_inventory", lambda *_: sessions)
    monkeypatch.setattr(
        topology_census, "validate_waveform_inventory", lambda *_: None
    )
    monkeypatch.setattr(
        topology_census, "_validate_calibrated_inventory", lambda *_: None
    )
    outputs = (
        tmp_path / "topology_manifest.jsonl",
        tmp_path / "topology_census.json",
        tmp_path / "DATA_CENSUS.md",
    )
    write_topology_census(
        tmp_path,
        tmp_path,
        *outputs,
    )
    first = tuple(path.read_bytes() for path in outputs)
    write_topology_census(
        tmp_path,
        tmp_path,
        *outputs,
    )
    second = tuple(path.read_bytes() for path in outputs)
    assert first == second
