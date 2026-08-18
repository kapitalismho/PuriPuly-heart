from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest

from experiments.psem_training_strategy_gate.data.provenance import (
    EXPECTED_ALIMEETING_MEETINGS,
    EXPECTED_AMI_MEETINGS,
    HISTORICAL_CONFIGS,
    REQUIRED_PRIOR_SOURCE_IDS,
    ProvenanceError,
    _ami_metadata,
    _historical_identities,
    _parse_alimeeting_textgrid,
    _validate_ami_segments,
    _wav_identity,
    collect_prior_exposure,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_historical_configs_reconstruct_the_mandatory_prior_set() -> None:
    prior = collect_prior_exposure(REPO_ROOT)
    assert set(prior) == REQUIRED_PRIOR_SOURCE_IDS
    for source_id in REQUIRED_PRIOR_SOURCE_IDS:
        assert prior[source_id]["prior_uses"] == sorted(HISTORICAL_CONFIGS)
        assert len(prior[source_id]["evidence"]) == len(HISTORICAL_CONFIGS)


def test_checked_in_provenance_inventory_is_session_specific_and_complete() -> None:
    sources = read_jsonl(DATA_DIR / "source_manifest.jsonl")
    annotations = read_jsonl(DATA_DIR / "annotation_manifest.jsonl")
    prior = read_jsonl(DATA_DIR / "prior_exposure_manifest.jsonl")
    assert len(sources) == 28
    assert len(annotations) == 28
    assert len(prior) == 10
    assert {row["session_id"] for row in sources if row["corpus"] == "AMI"} == EXPECTED_AMI_MEETINGS
    assert {
        row["session_id"] for row in sources if row["corpus"] == "AliMeeting"
    } == EXPECTED_ALIMEETING_MEETINGS
    assert {row["source_id"] for row in sources} == {
        row["source_id"] for row in annotations
    }
    assert {row["source_id"] for row in prior} == REQUIRED_PRIOR_SOURCE_IDS
    exposed = {row["source_id"] for row in sources if row["selection_exposed"]}
    assert exposed == REQUIRED_PRIOR_SOURCE_IDS
    reconstructed = collect_prior_exposure(REPO_ROOT)
    prior_by_id = {row["source_id"]: row for row in prior}
    for row in sources:
        assert len(row["waveform_sha256"]) == 64
        assert len(row["annotation_sha256"]) == 64
        assert row["sample_rate_hz"] == 16000
        assert row["channels"] == 1
        assert row["sample_width_bytes"] == 2
        if row["source_id"] in REQUIRED_PRIOR_SOURCE_IDS:
            assert row["eval_eligible"] is False
        else:
            assert row["eval_eligible"] is None
        assert row["corpus_version"]
    for source_id, facts in reconstructed.items():
        assert prior_by_id[source_id]["prior_uses"] == facts["prior_uses"]
        assert prior_by_id[source_id]["evidence"] == facts["evidence"]


def test_source_waveforms_match_the_preexisting_materialization_ledgers() -> None:
    sources = {
        row["source_id"]: row for row in read_jsonl(DATA_DIR / "source_manifest.jsonl")
    }
    checked = set()
    phase2_dir = REPO_ROOT / "experiments" / "speaker_turn_boundary" / "data" / "manifests"
    for name in (
        "alimeeting_eval_pilot.json",
        "ami_dev_pilot.json",
        "ami_held_out_pilot.json",
    ):
        raw = json.loads((phase2_dir / name).read_text(encoding="utf-8"))
        for case in raw["cases"]:
            source = sources.get(case["case_id"])
            if source is None:
                continue
            assert source["waveform_sha256"] == case["wav_sha256"]
            assert source["duration_samples"] == case["duration_samples"]
            checked.add(case["case_id"])
    materialization_path = (
        REPO_ROOT
        / "experiments"
        / "speaker_turn_boundary"
        / "results"
        / "turn_episode_v1"
        / "ami_materialization_manifest.json"
    )
    materialization = json.loads(materialization_path.read_text(encoding="utf-8"))
    for meeting_id, existing in materialization["meetings"].items():
        source_id = f"ami_{meeting_id}"
        assert sources[source_id]["waveform_sha256"] == existing["sha256"]
        checked.add(source_id)
    assert checked == set(sources)


def test_waveform_validation_fails_closed_for_missing_source(tmp_path: Path) -> None:
    with pytest.raises(ProvenanceError, match="missing source waveform"):
        _wav_identity(tmp_path / "missing.wav")


def test_historical_config_structure_is_required() -> None:
    with pytest.raises(ProvenanceError, match="must be an object"):
        _historical_identities("r6", sorted(REQUIRED_PRIOR_SOURCE_IDS))
    with pytest.raises(ProvenanceError, match="non-empty list"):
        _historical_identities("r6", {"development_sessions": []})


def test_waveform_validation_rejects_a_truncated_payload(tmp_path: Path) -> None:
    path = tmp_path / "truncated.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * 100)
    path.write_bytes(path.read_bytes()[:-20])
    with pytest.raises(ProvenanceError, match="truncated or empty"):
        _wav_identity(path)


def test_ami_segment_bundles_require_valid_xml_and_speaker_files(tmp_path: Path) -> None:
    with pytest.raises(ProvenanceError, match="missing"):
        _validate_ami_segments([], "ES2003a", ["A"], 16000)
    path = tmp_path / "ES2003a.A.segments.xml"
    path.write_text("not xml", encoding="utf-8")
    with pytest.raises(ProvenanceError, match="invalid AMI segment XML"):
        _validate_ami_segments([path], "ES2003a", ["A"], 16000)


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [
        ("nan", "0.5", "invalid AMI segment bounds"),
        ("1.25", "1.5", "exceed the source timeline"),
    ],
)
def test_ami_segment_bounds_must_be_finite_and_begin_within_the_waveform(
    tmp_path: Path, start: str, end: str, message: str
) -> None:
    path = tmp_path / "ES2003a.A.segments.xml"
    path.write_text(
        f'<root><segment transcriber_start="{start}" transcriber_end="{end}"/></root>',
        encoding="utf-8",
    )
    with pytest.raises(ProvenanceError, match=message):
        _validate_ami_segments([path], "ES2003a", ["A"], 16000)


def test_ami_metadata_rejects_duplicates_and_exposes_missing_speakers(tmp_path: Path) -> None:
    resources = tmp_path / "corpusResources"
    resources.mkdir()
    meetings = resources / "meetings.xml"
    meetings.write_text(
        '<root><meeting observation="ES2003a"/><meeting observation="ES2003a"/></root>',
        encoding="utf-8",
    )
    with pytest.raises(ProvenanceError, match="duplicate"):
        _ami_metadata(tmp_path)
    meetings.write_text('<root><meeting observation="ES2003a"/></root>', encoding="utf-8")
    metadata = _ami_metadata(tmp_path)["ES2003a"]
    assert metadata["speaker_ids"] == []
    assert metadata["unknown_speaker_count"] == 1


def alimeeting_textgrid(
    *,
    tier_count: str = "1",
    item_index: str = "1",
    timeline_start: str = "0",
    interval_start: str = "0",
) -> str:
    return "\n".join(
        [
            'File type = "ooTextFile"',
            'Object class = "TextGrid"',
            "",
            f"xmin = {timeline_start}",
            "xmax = 1",
            "tiers? <exists>",
            f"size = {tier_count}",
            "item []:",
            f"  item [{item_index}]:",
            '    class = "IntervalTier"',
            '    name = "N_SPK1"',
            "    xmin = 0",
            "    xmax = 1",
            "    intervals: size = 1",
            "    intervals [1]:",
            f"      xmin = {interval_start}",
            "      xmax = 1",
            '      text = "speech"',
        ]
    )


def test_alimeeting_textgrid_requires_real_interval_tiers(tmp_path: Path) -> None:
    path = tmp_path / "invalid.TextGrid"
    path.write_text('name = "N_SPK1"', encoding="utf-8")
    with pytest.raises(ProvenanceError, match="header"):
        _parse_alimeeting_textgrid(path)
    path.write_text(alimeeting_textgrid(), encoding="utf-8")
    parsed = _parse_alimeeting_textgrid(path)
    assert parsed["speaker_ids"] == ["SPK1"]
    assert parsed["timeline_end_sample"] == 16000


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (alimeeting_textgrid(tier_count="2"), "tier count mismatch"),
        (alimeeting_textgrid(item_index="2"), "tier count mismatch"),
        (alimeeting_textgrid(timeline_start="nan"), "invalid AliMeeting TextGrid timeline"),
        (alimeeting_textgrid(interval_start="nan"), "interval exceeds tier bounds"),
    ],
)
def test_alimeeting_textgrid_rejects_invalid_cardinality_and_nonfinite_bounds(
    tmp_path: Path, payload: str, message: str
) -> None:
    path = tmp_path / "invalid.TextGrid"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ProvenanceError, match=message):
        _parse_alimeeting_textgrid(path)
