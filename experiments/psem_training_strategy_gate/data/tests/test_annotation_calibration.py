from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.psem_training_strategy_gate.data.annotation_calibration import (
    AnnotationCalibrationError,
    build_calibration_report,
    duration_summary,
    render_markdown,
)
from experiments.psem_training_strategy_gate.data.annotation_normalization import (
    NormalizedSession,
)
from experiments.psem_training_strategy_gate.data.label_contract import (
    CanonicalInterval,
    generate_labels,
    load_contract,
)
from experiments.psem_training_strategy_gate.data.provenance import (
    sha256_file,
    write_jsonl,
)

DATA_DIR = Path(__file__).resolve().parents[1]
CALIBRATION_JSON_SHA256 = "1f10fc1980bd1a108753ef6afb45f455618df04014ad6b58dcf6e880ba01f4b1"
CALIBRATION_MARKDOWN_SHA256 = "2efe8137b8330699a0bcc1a770e40ce22c7c615de7b49b3724ddf0ebf80cddf6"
CALIBRATION_SOURCE_MANIFEST_SHA256 = "5cf6178a35e0c499bc3d79633c3ff1973f5f529c2b39e3bec89ca18ea96d6437"
CALIBRATION_ANNOTATION_MANIFEST_SHA256 = "f635171f8162115e08cee49e4fd748749b372e7eb37797bc91975bf2ca85c4a3"
CALIBRATION_NORMALIZATION_MANIFEST_SHA256 = "9805abe480eab757d29a484f67ee543a548dd91c7396007a85e2c60f44065079"


def _synthetic_sessions(
    intervals: tuple[CanonicalInterval, ...],
) -> list[NormalizedSession]:
    labels = generate_labels(
        intervals,
        scored_start_sample=intervals[0].start_sample,
        scored_end_sample=intervals[-1].end_sample,
    )
    return [
        NormalizedSession(
            source_id=f"{prefix}_session",
            corpus=corpus,
            session_id="session",
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


def _write_calibration_inputs(
    data_dir: Path, sessions: list[NormalizedSession]
) -> None:
    ordered_sessions = sorted(sessions, key=lambda session: session.source_id)
    write_jsonl(
        data_dir / "source_manifest.jsonl",
        ({"source_id": session.source_id} for session in ordered_sessions),
    )
    write_jsonl(
        data_dir / "annotation_manifest.jsonl",
        ({"source_id": session.source_id} for session in ordered_sessions),
    )
    write_jsonl(
        data_dir / "normalization_manifest.jsonl",
        (session.manifest_row() for session in ordered_sessions),
    )


def test_duration_summary_uses_deterministic_sample_quantiles() -> None:
    summary = duration_summary([16, 160, 1600], 16000)
    assert summary["count"] == 3
    assert summary["total_samples"] == 1776
    assert summary["quantiles_ms"]["p00"] == 1.0
    assert summary["quantiles_ms"]["p50"] == 10.0
    assert summary["quantiles_ms"]["p100"] == 100.0


def test_checked_in_calibration_report_binds_the_frozen_annotation_only_inputs() -> None:
    report = json.loads(
        (DATA_DIR / "annotation_calibration.json").read_text(encoding="utf-8")
    )
    contract = load_contract()
    assert contract.status == "frozen_after_annotation_only_calibration"
    assert report["contract_version"] == contract.contract_version
    assert report["contract_document_sha256"] == contract.document_sha256
    assert report["contract_status"] == contract.status
    assert report["overall"]["session_count"] == 28
    assert set(report["by_corpus"]) == {"AMI", "AliMeeting"}
    assert sha256_file(DATA_DIR / "annotation_calibration.json") == CALIBRATION_JSON_SHA256
    assert (
        sha256_file(DATA_DIR / "ANNOTATION_CALIBRATION.md")
        == CALIBRATION_MARKDOWN_SHA256
    )
    assert (
        report["input_policy"]["source_manifest_sha256"]
        == CALIBRATION_SOURCE_MANIFEST_SHA256
    )
    assert (
        report["input_policy"]["annotation_manifest_sha256"]
        == CALIBRATION_ANNOTATION_MANIFEST_SHA256
    )
    assert (
        report["input_policy"]["normalization_manifest_sha256"]
        == CALIBRATION_NORMALIZATION_MANIFEST_SHA256
    )
    assert report["input_policy"]["model_predictions_consulted"] is False
    assert report["input_policy"]["model_scores_consulted"] is False
    assert report["decision"]["version_bump_required"] is False


def test_calibration_report_is_annotation_only_and_retains_the_contract(
    tmp_path: Path,
) -> None:
    intervals = (
        CanonicalInterval(0, 3200, ("A",), source_annotation_ids=("a",)),
        CanonicalInterval(3200, 4000, (), source_annotation_ids=()),
        CanonicalInterval(4000, 5600, ("B",), source_annotation_ids=("b",)),
        CanonicalInterval(5600, 6400, (), source_annotation_ids=()),
        CanonicalInterval(6400, 9600, ("A",), source_annotation_ids=("c",)),
        CanonicalInterval(9600, 11200, ("A", "B"), source_annotation_ids=("d",)),
    )
    sessions = _synthetic_sessions(intervals)
    _write_calibration_inputs(tmp_path, sessions)
    report = build_calibration_report(sessions, tmp_path)
    assert report["input_policy"]["model_predictions_consulted"] is False
    assert report["input_policy"]["model_scores_consulted"] is False
    assert report["input_policy"]["official_model_results_inspected"] is False
    assert report["decision"]["action"] == (
        "retain_all_provisional_constants_and_freeze_contract"
    )
    assert report["overall"]["session_count"] == 2
    assert report["overall"]["silence_gap_bins"]["jitter_at_or_below_50ms"] == 4
    assert report["overall"]["intervening_speaker_bins"]["below_200ms"] == 2
    markdown = render_markdown(report)
    assert "No model prediction" in markdown
    assert "All provisional constants are retained" in markdown


def test_calibration_rejects_a_stale_normalization_manifest(tmp_path: Path) -> None:
    intervals = (
        CanonicalInterval(0, 3200, ("A",), source_annotation_ids=("a",)),
        CanonicalInterval(3200, 6400, ("B",), source_annotation_ids=("b",)),
    )
    sessions = _synthetic_sessions(intervals)
    _write_calibration_inputs(tmp_path, sessions)
    stale_rows = [
        session.manifest_row()
        for session in sorted(sessions, key=lambda session: session.source_id)
    ]
    stale_rows[0]["contract_document_sha256"] = "0" * 64
    write_jsonl(tmp_path / "normalization_manifest.jsonl", stale_rows)
    with pytest.raises(
        AnnotationCalibrationError,
        match="normalization manifest does not match calibrated sessions",
    ):
        build_calibration_report(sessions, tmp_path)


def test_masked_transition_fraction_excludes_diagnostic_rows(tmp_path: Path) -> None:
    intervals = (
        CanonicalInterval(0, 3200, ("A",), source_annotation_ids=("a",)),
        CanonicalInterval(
            3200, 4800, ("A", "B", "C"), source_annotation_ids=("b",)
        ),
        CanonicalInterval(4800, 8000, ("B",), source_annotation_ids=("c",)),
    )
    sessions = _synthetic_sessions(intervals)
    _write_calibration_inputs(tmp_path, sessions)
    overall = build_calibration_report(sessions, tmp_path)["overall"]
    assert overall["masked_transition_fraction"] == 0.5
    assert overall["masked_transition_reasons"] == {
        "complex_overlap_transition": 2
    }
    assert overall["diagnostic_masked_region_counts"] == {
        "complex_overlap_region": 2
    }


def test_boundary_granularity_uses_spacing_independent_of_grid_phase(
    tmp_path: Path,
) -> None:
    intervals = (
        CanonicalInterval(0, 80, (), source_annotation_ids=()),
        CanonicalInterval(80, 240, ("A",), source_annotation_ids=("a",)),
        CanonicalInterval(240, 400, (), source_annotation_ids=()),
        CanonicalInterval(400, 3600, ("A",), source_annotation_ids=("b",)),
    )
    sessions = _synthetic_sessions(intervals)
    _write_calibration_inputs(tmp_path, sessions)
    granularity = build_calibration_report(sessions, tmp_path)["overall"][
        "boundary_granularity"
    ]
    assert granularity["gcd_quantum_samples"] == 160
    assert granularity["gcd_quantum_ms"] == 10.0
    assert granularity["minimum_positive_step_samples"] == 160
