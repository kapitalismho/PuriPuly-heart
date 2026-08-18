from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest

from experiments.psem_training_strategy_gate.data import provenance
from experiments.psem_training_strategy_gate.data.provenance import (
    AMI_EXPANSION_EXCLUSIONS,
    BASELINE_AMI_MEETINGS,
    EXPECTED_ALIMEETING_MEETINGS,
    EXPECTED_AMI_MEETINGS,
    HISTORICAL_CONFIGS,
    REQUIRED_PRIOR_SOURCE_IDS,
    ProvenanceError,
    _ami_metadata,
    _ami_rows,
    _historical_identities,
    _parse_alimeeting_textgrid,
    _validate_ami_audio_inventory,
    _validate_ami_expansion_components,
    _validate_ami_segments,
    _validate_excluded_waveform,
    _validate_reversed_segment,
    collect_prior_exposure,
    wav_identity,
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
    assert len(sources) == len(EXPECTED_AMI_MEETINGS) + len(EXPECTED_ALIMEETING_MEETINGS)
    assert len(annotations) == len(sources)
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
    assert checked == {
        *(f"ami_{meeting_id}" for meeting_id in BASELINE_AMI_MEETINGS),
        *(f"alimeeting_{meeting_id}" for meeting_id in EXPECTED_ALIMEETING_MEETINGS),
    }


def test_expansion_component_validation_rejects_a_missing_baseline_annotation() -> None:
    available = set(EXPECTED_AMI_MEETINGS) | set(AMI_EXPANSION_EXCLUSIONS)
    meetings = {
        meeting_id: {
            "speaker_ids": [f"speaker-{meeting_id}"],
            "unknown_speaker_agents": [],
        }
        for meeting_id in available
    }
    assert len(_validate_ami_expansion_components(meetings, available)) == 14
    available.remove(next(iter(BASELINE_AMI_MEETINGS)))
    with pytest.raises(ProvenanceError, match="baseline annotation inventory"):
        _validate_ami_expansion_components(meetings, available)


def test_ami_audio_inventory_rejects_noncanonical_completed_wav(tmp_path: Path) -> None:
    for meeting_id in EXPECTED_AMI_MEETINGS:
        path = tmp_path / meeting_id / f"{meeting_id}.Mix-Headset.wav"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"")
    _validate_ami_audio_inventory(tmp_path)
    extra = tmp_path / next(iter(EXPECTED_AMI_MEETINGS)) / "recording.wav"
    extra.write_bytes(b"")
    with pytest.raises(ProvenanceError, match="extra=.*recording.wav"):
        _validate_ami_audio_inventory(tmp_path)


def test_exclusion_evidence_is_content_bound(tmp_path: Path) -> None:
    waveform_path = tmp_path / "excluded.wav"
    with wave.open(str(waveform_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * 100)
    waveform_record = wav_identity(waveform_path)
    _validate_excluded_waveform(waveform_path, waveform_record)
    waveform_record["waveform_sha256"] = "0" * 64
    with pytest.raises(ProvenanceError, match="evidence changed"):
        _validate_excluded_waveform(waveform_path, waveform_record)

    segment_path = tmp_path / "segments.xml"
    segment_path.write_text(
        '<root><segment transcriber_start="2.0" transcriber_end="1.0"/></root>',
        encoding="utf-8",
        newline="\n",
    )
    segment_record = {
        "annotation_file_size_bytes": segment_path.stat().st_size,
        "annotation_file_sha256": provenance.sha256_file(segment_path),
        "invalid_segment_index": 0,
        "transcriber_start": "2.0",
        "transcriber_end": "1.0",
    }
    _validate_reversed_segment(segment_path, segment_record)
    segment_record["transcriber_end"] = "3.0"
    with pytest.raises(ProvenanceError, match="segment evidence changed"):
        _validate_reversed_segment(segment_path, segment_record)


def test_provenance_reruns_expansion_selection_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject_selection(_: Path) -> dict:
        raise ProvenanceError("selection drift")

    monkeypatch.setattr(
        provenance, "validate_ami_expansion_selection", reject_selection
    )
    with pytest.raises(ProvenanceError, match="selection drift"):
        _ami_rows(tmp_path, {})


def test_waveform_validation_fails_closed_for_missing_source(tmp_path: Path) -> None:
    with pytest.raises(ProvenanceError, match="missing source waveform"):
        wav_identity(tmp_path / "missing.wav")


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
        wav_identity(path)


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
