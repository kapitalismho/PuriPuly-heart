from __future__ import annotations

import json
from pathlib import Path

from experiments.psem_training_strategy_gate.data.label_contract import (
    CanonicalInterval,
    generate_labels,
    load_contract,
)
from experiments.psem_training_strategy_gate.data.provenance import (
    canonical_sha256,
    sha256_file,
)
from experiments.psem_training_strategy_gate.data.reference_migration import (
    TopologyEvent,
    _handoff_comparison,
    _mask_comparison,
    _match_identity_events,
    _speech_correspondence,
    _topology_comparison,
)

DATA_DIR = Path(__file__).resolve().parents[1]
V2_DIR = DATA_DIR / "v2"
EXPECTED_ARTIFACT_SHA256 = {
    "reference_migration.jsonl": "9a1b03ee8bb3684d749a4d005dd01e4525712c1f155091eea5e3c67cd763c558",
    "reference_migration_summary.json": "1a3c2fd4ae76fd3bd9fdad98d0716493034f1b3157e177eec0b5048653cad478",
    "reference_provenance.json": "4d3f6a06d080bd7fd0da008912993899bc1dd9ea4e95f616df03a21946ea3c61",
    "reference_integrity_report.json": "bc3c0a4a2158084fbf12595bdb8e59ee909bd5d92446cca2a7b63eb95d29ea54",
    "REFERENCE_MIGRATION.md": "85e3cf38541b53011d0478b5b2f63a1f919005e47ff1515654cfd4b4f2d326e9",
    "reference_artifact_receipt.json": "bd896219d4646e341e35377459c51c052373d74010235ea47c0615bab4358885",
}


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _labels(intervals: tuple[CanonicalInterval, ...], version: str = "psem-handoff-v0"):
    return generate_labels(
        intervals,
        contract=load_contract(version=version),
        scored_start_sample=intervals[0].start_sample,
        scored_end_sample=intervals[-1].end_sample,
    )


def test_checked_in_reference_migration_artifacts_are_bound() -> None:
    rows = _jsonl(V2_DIR / "reference_migration.jsonl")
    summary = _json(V2_DIR / "reference_migration_summary.json")
    provenance = _json(V2_DIR / "reference_provenance.json")
    integrity = _json(V2_DIR / "reference_integrity_report.json")
    receipt = _json(V2_DIR / "reference_artifact_receipt.json")
    sources = _jsonl(V2_DIR / "source_manifest.jsonl")
    markdown = (V2_DIR / "REFERENCE_MIGRATION.md").read_text(encoding="utf-8")
    normalized = {row["source_id"]: row for row in _jsonl(V2_DIR / "normalization_manifest.jsonl")}

    assert {name: sha256_file(V2_DIR / name) for name in EXPECTED_ARTIFACT_SHA256} == (
        EXPECTED_ARTIFACT_SHA256
    )
    assert len(rows) == len(sources) == len(normalized) == 93
    assert {row["source_id"] for row in rows} == {row["source_id"] for row in sources}
    assert summary["session_manifest_sha256"] == canonical_sha256(rows)
    assert provenance["migration_session_manifest_sha256"] == canonical_sha256(rows)
    assert integrity["migration_session_manifest_sha256"] == canonical_sha256(rows)
    assert summary["overall"]["session_count"] == 93
    assert summary["by_corpus"]["AMI"]["session_count"] == 68
    assert summary["by_corpus"]["AliMeeting"]["session_count"] == 25
    assert integrity["status"] == "pass"
    assert all(integrity["checks"].values())
    assert integrity["reference_inventory_sha256"] == provenance["reference_inventory_sha256"]
    assert integrity["reference_provenance_sha256"] == canonical_sha256(provenance)
    assert integrity["migration_summary_sha256"] == canonical_sha256(summary)
    assert provenance["reference_inventory_sha256"] == canonical_sha256(
        provenance["references"]
    )
    assert receipt["artifact_sha256"] == {
        name: sha256_file(V2_DIR / name)
        for name in receipt["artifact_sha256"]
    }
    assert receipt["artifact_set_sha256"] == canonical_sha256(
        receipt["artifact_sha256"]
    )
    assert "does not perform additional manual boundary adjudication" in markdown
    assert all(f"`{row['source_id']}`" in markdown for row in rows)
    assert "## Corpus diagnostic detail" in markdown
    assert "## Meeting diagnostic detail" in markdown
    assert "| 100 ms |" in markdown
    assert "| 200 ms |" in markdown
    for artifact in provenance["input_artifacts"]:
        assert sha256_file(DATA_DIR / artifact["ref"]) == artifact["sha256"]
    assert provenance["source_license_ids_by_corpus"] == {
        "AMI": ["CC-BY-4.0"],
        "AliMeeting": ["CC-BY-SA-4.0"],
    }
    source_by_id = {row["source_id"]: row for row in sources}
    reference_by_id = {
        row["source_id"]: row for row in provenance["references"]
    }
    expected_confusion_keys = {
        f"{old}->{new}"
        for old in ("direct", "gap", "overlap")
        for new in ("direct", "gap", "overlap")
    }
    for row in rows:
        normalization = normalized[row["source_id"]]
        reference = reference_by_id[row["source_id"]]
        source = source_by_id[row["source_id"]]
        assert row["reference_sha256"] == normalization["reference_sha256"]
        assert row["reference_sha256"] == reference["reference_sha256"]
        assert row["reference_ref"] == normalization["reference_ref"]
        assert row["reference_ref"] == reference["reference_ref"]
        assert row["source_waveform_sha256"] == normalization["source_waveform_sha256"]
        assert row["source_waveform_sha256"] == reference["source_waveform_sha256"]
        assert row["source_annotation_sha256"] == normalization["source_annotation_sha256"]
        assert row["source_annotation_sha256"] == reference["source_annotation_sha256"]
        assert reference["source_record_sha256"] == canonical_sha256(source)
        assert reference["canonical_intervals_sha256"] == normalization[
            "canonical_intervals_sha256"
        ]
        assert reference["label_result_sha256"] == normalization["label_result_sha256"]
        assert reference["speaker_mapping_sha256"] == normalization[
            "speaker_mapping_sha256"
        ]
        assert reference["reference_metadata_sha256"] == normalization[
            "reference_metadata_sha256"
        ]
        assert set(row["topology"]["direct_gap_overlap_confusion"]) == (
            expected_confusion_keys
        )
        assert row["v1_exposure"]["scored_samples"] == row["v2_exposure"]["scored_samples"]
        assert (
            row["change_classification"]["topology_changing_episode_count"]
            == row["topology"]["topology_changing_total_count"]
        )
    assert set(summary["overall"]["topology"]["direct_gap_overlap_confusion"]) == (
        expected_confusion_keys
    )
    selected_train = [row for row in rows if "Train_Ali_far" in row["reference_ref"]]
    assert len(selected_train) == 17
    assert all(row["v1_exposure"]["reliable_solo_samples"] > 0 for row in selected_train)
    assert all(row["topology"]["v1_episode_count"] > 0 for row in selected_train)
    assert all(
        row["speech_correspondence"]["deterministic_old_segment_count"] > 0
        for row in selected_train
    )


def test_speech_correspondence_distinguishes_internal_pause_and_outer_padding() -> None:
    old = (CanonicalInterval(0, 10000, ("A",)),)
    new = (
        CanonicalInterval(0, 1000, ()),
        CanonicalInterval(1000, 3000, ("A",)),
        CanonicalInterval(3000, 4000, ()),
        CanonicalInterval(4000, 9000, ("A",)),
        CanonicalInterval(9000, 10000, ()),
    )

    result, starts, ends, absolute = _speech_correspondence(old, new)

    assert result["deterministic_old_segment_count"] == 1
    assert result["removed_internal_pause_samples"] == 1000
    assert result["removed_outer_padding_samples"] == 2000
    assert starts == (1000,)
    assert ends == (-1000,)
    assert absolute == (1000, 1000)


def test_speech_correspondence_marks_shared_new_span_ambiguous() -> None:
    old = (
        CanonicalInterval(0, 3000, ("A",)),
        CanonicalInterval(3000, 4000, ()),
        CanonicalInterval(4000, 7000, ("A",)),
    )
    new = (CanonicalInterval(0, 7000, ("A",)),)

    result, starts, ends, absolute = _speech_correspondence(old, new)

    assert result["ambiguous_old_segment_count"] == 2
    assert result["deterministic_old_segment_count"] == 0
    assert starts == ()
    assert ends == ()
    assert absolute == ()


def test_event_matching_maximizes_identity_matches_then_minimizes_displacement() -> None:
    old = (
        TopologyEvent(("A", "B"), "overlap_takeover", 0, 0),
        TopologyEvent(("A", "B"), "overlap_takeover", 7000, 7000),
    )
    new = (
        TopologyEvent(("A", "B"), "overlap_takeover", 5000, 5000),
        TopologyEvent(("A", "B"), "overlap_takeover", 9000, 9000),
    )

    pairs, removed, added = _match_identity_events(old, new)

    assert pairs == ((0, 0), (1, 1))
    assert removed == ()
    assert added == ()


def test_topology_comparison_separates_timing_only_from_topology_change() -> None:
    old_intervals = (
        CanonicalInterval(0, 3200, ("A",)),
        CanonicalInterval(3200, 6400, ("B",)),
    )
    new_intervals = (
        CanonicalInterval(0, 3200, ("A",)),
        CanonicalInterval(3200, 4800, ()),
        CanonicalInterval(4800, 8000, ("B",)),
    )

    result = _topology_comparison(_labels(old_intervals), _labels(new_intervals))

    assert result["matched_identity_within_500ms_count"] == 1
    assert result["timing_only_change_count"] == 0
    assert result["topology_changing_match_count"] == 1
    assert result["direct_gap_overlap_confusion"] == {
        "direct->direct": 0,
        "direct->gap": 1,
        "direct->overlap": 0,
        "gap->direct": 0,
        "gap->gap": 0,
        "gap->overlap": 0,
        "overlap->direct": 0,
        "overlap->gap": 0,
        "overlap->overlap": 0,
    }
    assert result["retention"]["within_50ms"]["count"] == 0
    assert result["retention"]["within_100ms"]["count"] == 1


def test_topology_comparison_counts_same_topology_time_shift_as_timing_only() -> None:
    old_intervals = (
        CanonicalInterval(0, 3200, ("A",)),
        CanonicalInterval(3200, 6400, ("B",)),
    )
    new_intervals = (
        CanonicalInterval(0, 4000, ("A",)),
        CanonicalInterval(4000, 7200, ("B",)),
    )

    result = _topology_comparison(_labels(old_intervals), _labels(new_intervals))

    assert result["unchanged_episode_count"] == 0
    assert result["timing_only_change_count"] == 1
    assert result["topology_changing_total_count"] == 0
    assert result["retention"]["within_50ms"]["count"] == 1


def test_same_speaker_gap_uses_reliable_solo_start_as_diagnostic_event_time() -> None:
    old_intervals = (
        CanonicalInterval(0, 3200, ("A",)),
        CanonicalInterval(3200, 4800, ()),
        CanonicalInterval(4800, 8000, ("A",)),
    )
    new_intervals = (
        CanonicalInterval(0, 3200, ("A",)),
        CanonicalInterval(3200, 5600, ()),
        CanonicalInterval(5600, 8800, ("A",)),
    )

    result = _topology_comparison(_labels(old_intervals), _labels(new_intervals))

    assert result["timing_only_change_count"] == 1
    assert result["retention"]["within_50ms"]["count"] == 1


def test_handoff_and_nonlexical_mask_changes_are_reported() -> None:
    old_intervals = (
        CanonicalInterval(0, 3200, ("A",)),
        CanonicalInterval(3200, 6400, ("B",)),
    )
    new_intervals = (
        CanonicalInterval(0, 3200, ("A",)),
        CanonicalInterval(
            3200,
            4800,
            (),
            handoff_relation_mask_classes=("ambiguous_nonlexical_vocalization",),
            mask_annotation_ids=("mask-1",),
        ),
        CanonicalInterval(4800, 8000, ("B",)),
    )
    old_labels = _labels(old_intervals)
    new_labels = _labels(new_intervals, "psem-handoff-v1")

    handoffs = _handoff_comparison(old_labels, new_labels)
    masks = _mask_comparison(old_labels, new_labels)

    assert handoffs["removed_handoff_count"] == 1
    assert handoffs["added_handoff_count"] == 0
    assert masks["v1_masked_transition_count"] == 0
    assert masks["v2_masked_transition_count"] == 1
    assert masks["v2_reasons"] == {"ambiguous_nonlexical_vocalization_crossing": 1}
