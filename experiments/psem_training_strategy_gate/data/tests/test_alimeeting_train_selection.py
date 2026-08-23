from __future__ import annotations

import hashlib
import io
import json
import struct
import tarfile
import wave
from dataclasses import replace
from pathlib import Path

import pytest

from experiments.psem_training_strategy_gate.data import (
    alimeeting_train_materialization,
)
from experiments.psem_training_strategy_gate.data.alimeeting_train_materialization import (
    AliMeetingTrainMaterializationError,
    _materialize_channel_zero,
    _materialize_validated_raw_members,
    validate_materialization_receipt,
)
from experiments.psem_training_strategy_gate.data.alimeeting_train_selection import (
    SAMPLE_RATE_HZ,
    AliMeetingTrainSelectionError,
    ArchiveMember,
    CandidateComponent,
    CandidateSession,
    _accepted_scored_samples,
    _recording_group,
    _waveform_duration_samples,
    build_components,
    select_components,
    validate_selection_receipt,
)
from experiments.psem_training_strategy_gate.data.forced_alignment_reference import (
    canonical_sha256,
    parse_rttm,
)
from experiments.psem_training_strategy_gate.data.reference_normalization import (
    ReferenceNormalizationError,
    alimeeting_speaker_ids,
    parse_alimeeting_nonlexical_masks,
)


def _component(index: int) -> CandidateComponent:
    return CandidateComponent(
        component_id=f"component-{index:064x}",
        session_ids=(f"R{index:04d}_M{index:04d}",),
        scored_samples=SAMPLE_RATE_HZ * 1800,
        participant_buckets=(2, 3, 4),
        room_ids=(f"R{index:04d}",),
        recording_group_ids=(f"MS{index:03d}",),
        speaker_ids=(f"SPK{index:04d}",),
        shared_speaker_ids=(),
        selection_eligible=True,
        excluded_session_ids=(),
    )


def _session(
    session_id: str,
    speakers: tuple[str, ...],
    room_id: str,
    recording_group_id: str,
) -> CandidateSession:
    return CandidateSession(
        session_id=session_id,
        room_id=room_id,
        meeting_id=session_id.split("_", 1)[1],
        recording_group_id=recording_group_id,
        scored_samples=SAMPLE_RATE_HZ * 1800,
        textgrid_timeline_samples=SAMPLE_RATE_HZ * 1800,
        annotation_tail_excess_samples=0,
        reference_end_sample=SAMPLE_RATE_HZ * 1800,
        selection_eligible=True,
        selection_exclusion_reasons=(),
        participant_count=len(speakers),
        speaker_ids=speakers,
        waveform_member=ArchiveMember(
            "wave.wav",
            45,
            None,
            SAMPLE_RATE_HZ * 1800,
        ),
        textgrid_member=ArchiveMember("text.TextGrid", 1, "1" * 64),
        reference_ref=f"AliMeeting/Train_Ali_far/{session_id}.rttm",
        reference_sha256="2" * 64,
    )


def _textgrid(tiers: list[tuple[str, str]]) -> str:
    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        "xmin = 0",
        "xmax = 1",
        "tiers? <exists>",
        f"size = {len(tiers)}",
        "item []:",
    ]
    for index, (tier_name, text) in enumerate(tiers, 1):
        lines.extend(
            [
                f"  item [{index}]:",
                '    class = "IntervalTier"',
                f'    name = "{tier_name}"',
                "    xmin = 0",
                "    xmax = 1",
                "    intervals: size = 1",
                "    intervals [1]:",
                "      xmin = 0",
                "      xmax = 1",
                f'      text = "{text}"',
            ]
        )
    return "\n".join(lines)


def test_selection_requires_unique_first_three_objectives() -> None:
    components = tuple(_component(index) for index in range(19))
    with pytest.raises(
        AliMeetingTrainSelectionError,
        match="multiple AliMeeting duration-optimal candidates",
    ):
        select_components(components)


def test_selection_excludes_a_whole_ineligible_component() -> None:
    components = [_component(index) for index in range(19)]
    components[0] = replace(
        components[0],
        selection_eligible=False,
        excluded_session_ids=components[0].session_ids,
    )
    result = select_components(tuple(components))
    assert result.optimal_candidate_count == 1
    assert components[0].component_id not in result.component_ids
    assert len(result.component_ids) == 18


def test_components_connect_only_shared_official_speakers() -> None:
    sessions = (
        _session("R0001_M0001", ("SPK1", "SPK2"), "R0001", "MS001"),
        _session("R0001_M0002", ("SPK2", "SPK3"), "R0001", "MS001"),
        _session("R0001_M0003", ("SPK4", "SPK5"), "R0001", "MS001"),
    )
    components = build_components(sessions)
    assert [component.session_ids for component in components] == [
        ("R0001_M0001", "R0001_M0002"),
        ("R0001_M0003",),
    ]
    assert components[0].shared_speaker_ids == ("SPK2",)
    assert components[1].shared_speaker_ids == ()


def test_train_textgrid_tiers_and_silence_marker_are_fail_closed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "R0003_M0046.TextGrid"
    path.write_text(
        _textgrid(
            [
                ("R0003_M0046_F_SPK0093", "(sil)"),
                ("M_SPK0094", "词"),
            ]
        ),
        encoding="utf-8",
    )
    assert alimeeting_speaker_ids(path) == ("SPK0093", "SPK0094")
    parsed = parse_alimeeting_nonlexical_masks(
        "alimeeting_R0003_M0046",
        path,
        speaker_map={"SPK0093": "SPK0093", "SPK0094": "SPK0094"},
        scored_start_sample=0,
        scored_end_sample=SAMPLE_RATE_HZ,
    )
    assert parsed.masks == ()
    assert parsed.observed_marker_counts == {"(sil)": 1}
    assert parsed.observed_class_counts == {
        "text:lexical_text": 1,
        "text:nonhuman_noise_marker_only": 1,
    }
    path.write_text(
        _textgrid([("R9999_M9999_F_SPK0093", "词")]),
        encoding="utf-8",
    )
    with pytest.raises(ReferenceNormalizationError, match="invalid.*structure"):
        alimeeting_speaker_ids(path)


def test_textgrid_only_tail_is_bounded_by_the_waveform_rule(
    tmp_path: Path,
) -> None:
    path = tmp_path / "R0003_M0046.TextGrid"
    path.write_text(
        _textgrid([("F_SPK0093", "词")]),
        encoding="utf-8",
    )
    parsed = parse_alimeeting_nonlexical_masks(
        "alimeeting_R0003_M0046",
        path,
        speaker_map={"SPK0093": "SPK0093"},
        scored_start_sample=0,
        scored_end_sample=SAMPLE_RATE_HZ - 159,
    )
    assert parsed.masks == ()
    with pytest.raises(
        ReferenceNormalizationError,
        match="timeline does not match scored range",
    ):
        parse_alimeeting_nonlexical_masks(
            "alimeeting_R0003_M0046",
            path,
            speaker_map={"SPK0093": "SPK0093"},
            scored_start_sample=0,
            scored_end_sample=SAMPLE_RATE_HZ - 161,
        )


def test_train_rttm_gender_prefixes_map_to_canonical_speakers(
    tmp_path: Path,
) -> None:
    path = tmp_path / "R0003_M0046.rttm"
    path.write_text(
        "SPEAKER R0003_M0046 1 0.1 0.2 <NA> <NA> F_SPK0093 <NA> <NA>\n"
        "SPEAKER R0003_M0046 1 0.4 0.2 <NA> <NA> M_SPK0094 <NA> <NA>\n",
        encoding="utf-8",
    )
    parsed = parse_rttm(
        path,
        corpus="AliMeeting",
        session_id="R0003_M0046",
        speaker_map={"SPK0093": "SPK0093", "SPK0094": "SPK0094"},
        scored_start_sample=0,
        scored_end_sample=SAMPLE_RATE_HZ,
    )
    assert [span.speaker_id for span in parsed.spans] == [
        "SPK0093",
        "SPK0094",
    ]


def test_waveform_member_identity_rejects_cross_session_alias() -> None:
    assert (
        _recording_group(
            "Train_Ali_far/audio_dir/R0003_M0046_MS002.wav",
            "R0003_M0046",
        )
        == "MS002"
    )
    with pytest.raises(AliMeetingTrainSelectionError, match="invalid"):
        _recording_group(
            "Train_Ali_far/audio_dir/R0003_M0047_MS002.wav",
            "R0003_M0046",
        )


def test_committed_selection_receipt_reproduces_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    receipt_path = Path(__file__).parents[1] / "alimeeting_train_selection.json"
    receipt = validate_selection_receipt(receipt_path)
    assert receipt["selected_meeting_count"] == 17
    assert receipt["objective_values"]["optimal_candidate_count"] == 1
    tampered = {**receipt, "selection_hash": "0" * 64}
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(AliMeetingTrainSelectionError, match="does not reproduce"):
        validate_selection_receipt(tampered_path)
    for field, value in (
        ("candidate_summary", {**receipt["candidate_summary"], "scored_samples": 0}),
        (
            "selection_model_inputs",
            {**receipt["selection_model_inputs"], "model_scores_used": True},
        ),
        (
            "selection_policy",
            {**receipt["selection_policy"], "annotation_and_model_score_blind": False},
        ),
    ):
        changed = {**receipt, field: value}
        changed_path = tmp_path / f"tampered-{field}.json"
        changed_path.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(
            AliMeetingTrainSelectionError,
            match="receipt file identity changed",
        ):
            validate_selection_receipt(changed_path)


def _tiny_train_archive(
    path: Path,
    waveform_payload: bytes,
    textgrid_payload: bytes,
) -> None:
    with tarfile.open(path, "w:gz") as bundle:
        for name in sorted(alimeeting_train_materialization.ARCHIVE_ROOTS):
            member = tarfile.TarInfo(name)
            member.type = tarfile.DIRTYPE
            bundle.addfile(member)
        for name, payload in (
            (
                "Train_Ali_far/audio_dir/R0001_M0001_MS001.wav",
                waveform_payload,
            ),
            (
                "Train_Ali_far/textgrid_dir/R0001_M0001.TextGrid",
                textgrid_payload,
            ),
        ):
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            bundle.addfile(member, io.BytesIO(payload))


def test_archive_validation_stages_raw_and_binds_selected_textgrid(
    tmp_path: Path,
    monkeypatch,
) -> None:
    raw_path = tmp_path / "source.wav"
    with wave.open(str(raw_path), "wb") as handle:
        handle.setnchannels(8)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE_HZ)
        handle.writeframes(struct.pack("<16h", *range(16)))
    waveform_payload = raw_path.read_bytes()
    textgrid_payload = b"textgrid-bytes"
    archive_path = tmp_path / "Train_Ali_far.tar.gz"
    _tiny_train_archive(archive_path, waveform_payload, textgrid_payload)
    archive_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    monkeypatch.setattr(alimeeting_train_materialization, "ARCHIVE_SESSION_COUNT", 1)
    monkeypatch.setattr(
        alimeeting_train_materialization,
        "TRAIN_ARCHIVE_SIZE_BYTES",
        archive_path.stat().st_size,
    )
    monkeypatch.setattr(
        alimeeting_train_materialization,
        "TRAIN_ARCHIVE_SHA256",
        archive_sha256,
    )
    row = {
        "session_id": "R0001_M0001",
        "waveform_archive_member_ref": (
            "Train_Ali_far/audio_dir/R0001_M0001_MS001.wav"
        ),
        "waveform_archive_member_size_bytes": len(waveform_payload),
        "textgrid_archive_member_ref": (
            "Train_Ali_far/textgrid_dir/R0001_M0001.TextGrid"
        ),
        "textgrid_archive_member_size_bytes": len(textgrid_payload),
        "textgrid_sha256": hashlib.sha256(textgrid_payload).hexdigest(),
    }
    corrupt_root = tmp_path / "corrupt-corpus"
    monkeypatch.setattr(
        alimeeting_train_materialization,
        "TRAIN_ARCHIVE_SHA256",
        "0" * 64,
    )
    with pytest.raises(
        AliMeetingTrainMaterializationError,
        match="archive inventory changed",
    ):
        _materialize_validated_raw_members(
            archive_path,
            corrupt_root,
            {row["session_id"]: row},
            {row["session_id"]: row},
        )
    assert not (
        corrupt_root
        / "alimeeting/Train_Ali/Train_Ali_far/audio_dir/"
        / "R0001_M0001_MS001.wav"
    ).exists()
    monkeypatch.setattr(
        alimeeting_train_materialization,
        "TRAIN_ARCHIVE_SHA256",
        archive_sha256,
    )
    with pytest.raises(
        AliMeetingTrainMaterializationError,
        match="TextGrid archive bytes changed",
    ):
        _materialize_validated_raw_members(
            archive_path,
            tmp_path / "textgrid-corpus",
            {row["session_id"]: row},
            {
                row["session_id"]: {
                    **row,
                    "textgrid_sha256": "0" * 64,
                }
            },
        )


def test_channel_zero_materialization_is_exact_and_conflict_safe(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "raw.wav"
    target_path = tmp_path / "channel-zero.wav"
    frames = (
        (11, 12, 13, 14, 15, 16, 17, 18),
        (-21, -22, -23, -24, -25, -26, -27, -28),
        (301, 302, 303, 304, 305, 306, 307, 308),
    )
    with wave.open(str(raw_path), "wb") as handle:
        handle.setnchannels(8)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE_HZ)
        handle.writeframes(
            struct.pack("<24h", *(sample for frame in frames for sample in frame))
        )
    payload = raw_path.read_bytes()
    assert _waveform_duration_samples(payload, len(payload)) == 3
    assert _accepted_scored_samples(3, 3) == (3, 0)
    assert _accepted_scored_samples(162, 3) == (3, 159)
    identity, status, source_identity = _materialize_channel_zero(
        raw_path,
        target_path,
    )
    assert status == "materialized"
    assert source_identity["duration_samples"] == 3
    assert identity["duration_samples"] == 3
    with wave.open(str(target_path), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == SAMPLE_RATE_HZ
        assert struct.unpack("<3h", handle.readframes(3)) == (11, -21, 301)
    repeated_identity, repeated_status, _ = _materialize_channel_zero(
        raw_path,
        target_path,
    )
    assert repeated_status == "existing"
    assert repeated_identity == identity
    target_path.write_bytes(b"conflict")
    with pytest.raises(
        AliMeetingTrainMaterializationError,
        match="existing target conflicts",
    ):
        _materialize_channel_zero(raw_path, target_path)


def test_committed_materialization_receipt_is_structural_and_deterministic(
    tmp_path: Path,
) -> None:
    data_root = Path(__file__).parents[1]
    receipt_path = data_root / "alimeeting_train_materialization.json"
    selection_path = data_root / "alimeeting_train_selection.json"
    receipt = validate_materialization_receipt(
        receipt_path,
        selection_path,
        None,
    )
    assert receipt["selected_session_count"] == 17
    assert all(
        not any(key.endswith("_status") for key in row)
        for row in receipt["materialized_sessions"]
    )
    tampered = {**receipt, "archive_sha256": "0" * 64}
    tampered_path = tmp_path / "tampered-materialization.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(
        AliMeetingTrainMaterializationError,
        match="materialization authority changed",
    ):
        validate_materialization_receipt(
            tampered_path,
            selection_path,
            None,
        )
    changed = json.loads(json.dumps(receipt))
    changed["materialized_sessions"][0]["waveform_sha256"] = "0" * 64
    changed_path = tmp_path / "tampered-waveform-identity.json"
    changed_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(
        AliMeetingTrainMaterializationError,
        match="receipt file identity changed",
    ):
        validate_materialization_receipt(
            changed_path,
            selection_path,
            None,
        )


def test_committed_v2_inventory_contains_all_normalized_train_sources() -> None:
    data_root = Path(__file__).parents[1]
    selection = validate_selection_receipt(
        data_root / "alimeeting_train_selection.json"
    )
    def load_rows(path: Path) -> list[dict]:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        ]
    source_rows = load_rows(data_root / "v2" / "source_manifest.jsonl")
    annotation_rows = load_rows(
        data_root / "v2" / "annotation_manifest.jsonl"
    )
    normalization_rows = load_rows(
        data_root / "v2" / "normalization_manifest.jsonl"
    )
    source_by_id = {row["source_id"]: row for row in source_rows}
    normalized_by_id = {
        row["source_id"]: row for row in normalization_rows
    }
    expected_source_ids = set(source_by_id)
    assert len(expected_source_ids) == 93
    assert {row["source_id"] for row in annotation_rows} == expected_source_ids
    assert set(normalized_by_id) == expected_source_ids
    assert {row["corpus"] for row in source_rows} == {"AMI", "AliMeeting"}
    for session_id in selection["selected_session_ids"]:
        source_id = f"alimeeting_{session_id}"
        source = source_by_id[source_id]
        normalized = normalized_by_id[source_id]
        assert source["speaker_identity_scope"] == "corpus_global"
        assert source["meeting_type"] == "natural_meeting_train_partition"
        assert normalized["contract_version"] == "psem-handoff-v1"
        assert normalized["scored_end_sample"] == source[
            "annotation_coverage_end_sample"
        ]
        assert normalized["source_record_sha256"] == canonical_sha256(source)
