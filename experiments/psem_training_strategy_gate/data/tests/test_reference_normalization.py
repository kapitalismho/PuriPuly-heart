from __future__ import annotations

import json
import wave
from dataclasses import replace
from pathlib import Path

import pytest

from experiments.psem_training_strategy_gate.data import reference_normalization
from experiments.psem_training_strategy_gate.data.forced_alignment_reference import (
    REFERENCE_COMMIT,
    REFERENCE_REPOSITORY,
    ForcedAlignmentReferenceError,
    ReferenceSpan,
    canonical_sha256,
    sha256_file,
)
from experiments.psem_training_strategy_gate.data.label_contract import (
    LabelContractError,
    generate_labels,
    load_contract,
    normalize_intervals,
)
from experiments.psem_training_strategy_gate.data.reference_normalization import (
    INVENTORY_PATH,
    MASK_CLASS,
    AmbiguityMask,
    ReferenceNormalizationError,
    classify_alimeeting_text,
    compose_reference_timeline,
    load_nonlexical_inventory,
    normalize_reference_inventory,
    normalize_reference_session,
    open_reference_checkout,
    parse_alimeeting_nonlexical_masks,
    parse_ami_nonlexical_masks,
)


def _inventory_with_markers(tmp_path: Path):
    raw = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    raw["alimeeting"]["marker_tokens"] = {
        "[laugh]": "human_vocal",
        "[noise]": "nonhuman_noise",
    }
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    return load_nonlexical_inventory(path)


def _textgrid(intervals: list[tuple[str, str, str]]) -> str:
    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        "xmin = 0",
        "xmax = 1",
        "tiers? <exists>",
        "size = 1",
        "item []:",
        "  item [1]:",
        '    class = "IntervalTier"',
        '    name = "N_SPK1"',
        "    xmin = 0",
        "    xmax = 1",
        f"    intervals: size = {len(intervals)}",
    ]
    for index, (start, end, text) in enumerate(intervals, 1):
        lines.extend(
            [
                f"    intervals [{index}]:",
                f"      xmin = {start}",
                f"      xmax = {end}",
                f'      text = "{text}"',
            ]
        )
    return "\n".join(lines)


def _alimeeting_fixture(tmp_path: Path, monkeypatch):
    corpus_root = tmp_path / "corpus"
    textgrid = (
        corpus_root
        / "alimeeting"
        / "Eval_Ali"
        / "Eval_Ali_far"
        / "textgrid_dir"
        / "R1.TextGrid"
    )
    textgrid.parent.mkdir(parents=True)
    textgrid.write_text(_textgrid([("0", "1", "lexical")]), encoding="utf-8")
    audio = corpus_root / "alimeeting" / "far_ch0" / "R1.wav"
    audio.parent.mkdir(parents=True)
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 16000)
    reference_root = tmp_path / "reference"
    rttm = reference_root / "AliMeeting" / "Eval_Ali_far" / "R1.rttm"
    rttm.parent.mkdir(parents=True)
    rttm.write_text(
        "SPEAKER R1 1 0.2 0.2 <NA> <NA> N_SPK1 <NA> <NA>\n",
        encoding="utf-8",
    )
    provenance = {
        "repository": REFERENCE_REPOSITORY,
        "commit": REFERENCE_COMMIT,
        "git_tree": "1" * 40,
        "git_object_format": "sha1",
        "tracked_file_count": 1,
        "license_ref": "LICENSE",
        "license_sha256": "2" * 64,
        "readme_ref": "README.md",
        "readme_sha256": "3" * 64,
    }
    monkeypatch.setattr(
        reference_normalization,
        "validate_reference_checkout",
        lambda _: provenance,
    )
    contract = load_contract()
    annotation_files = [
        {
            "ref": textgrid.relative_to(corpus_root).as_posix(),
            "size_bytes": textgrid.stat().st_size,
            "sha256": sha256_file(textgrid),
        }
    ]
    source_row = {
        "schema_version": 1,
        "source_id": "alimeeting_R1",
        "corpus": "AliMeeting",
        "session_id": "R1",
        "speaker_ids": ["SPK1"],
        "audio_ref": audio.relative_to(corpus_root).as_posix(),
        "waveform_sha256": sha256_file(audio),
        "waveform_size_bytes": audio.stat().st_size,
        "duration_samples": 16000,
        "sample_rate_hz": 16000,
        "channels": 1,
        "sample_width_bytes": 2,
        "annotation_ref": textgrid.relative_to(corpus_root).as_posix(),
        "annotation_sha256": canonical_sha256(annotation_files),
        "annotation_coverage_start_sample": 0,
        "annotation_coverage_end_sample": 16000,
        "contract_version": contract.contract_version,
        "contract_document_sha256": contract.document_sha256,
    }
    return corpus_root, reference_root, textgrid, source_row, provenance


def test_v1_contract_is_pinned_and_preserves_v0_constants() -> None:
    v0 = load_contract()
    v1 = load_contract(version="psem-handoff-v1")
    assert v0.contract_version == "psem-handoff-v0"
    assert v1.contract_version == "psem-handoff-v1"
    assert v1.document_sha256 == (
        "3915ab5d6fe3c8e2eb0933ce619f425eb9a08cf6bc3a46eacf370647956772d2"
    )
    assert (
        v1.reliable_solo_min_duration_ms,
        v1.annotation_boundary_jitter_ms,
        v1.gap_topology_min_duration_ms,
        v1.overlap_topology_min_duration_ms,
        v1.local_continuity_max_gap_ms,
        v1.short_backchannel_min_duration_ms,
        v1.short_backchannel_max_duration_ms,
    ) == (200, 50, 100, 100, 1200, 200, 1000)
    with pytest.raises(LabelContractError, match="unsupported installed"):
        load_contract(version="psem-handoff-v2")


def test_relation_mask_preserves_activity_and_masks_crossing_transition() -> None:
    spans = [
        ReferenceSpan(0, 6000, "A", ("fixture.rttm#L1",)),
        ReferenceSpan(6000, 12000, "B", ("fixture.rttm#L2",)),
    ]
    masks = [
        AmbiguityMask(
            5500,
            6500,
            MASK_CLASS,
            "A",
            "fixture.words.xml#v1",
            "ami_nonzero_duration_vocalsound",
        )
    ]
    intervals = compose_reference_timeline(
        spans,
        masks,
        scored_start_sample=0,
        scored_end_sample=12000,
    )
    assert intervals == compose_reference_timeline(
        reversed(spans),
        reversed(masks),
        scored_start_sample=0,
        scored_end_sample=12000,
    )
    result = generate_labels(
        intervals,
        contract=load_contract(version="psem-handoff-v1"),
        scored_start_sample=0,
        scored_end_sample=12000,
    )
    assert [row["state"] for row in result.activity_labels] == [
        "singleton",
        "singleton",
        "singleton",
        "singleton",
    ]
    assert all(row["mask_state"] == "valid" for row in result.activity_labels)
    transition = next(
        row
        for row in result.transitions
        if row["primary_topology"]
        == "ambiguous_nonlexical_vocalization_crossing"
    )
    assert transition["handoff_confirmed"] is None
    assert transition["relation_target"] is None
    assert transition["secondary_tags"] == [MASK_CLASS]
    assert result.exposure["ambiguous_samples"] == 0
    assert result.exposure["ambiguous_nonlexical_vocalization_samples"] == 1000
    assert result.exposure["stable_singleton_samples"] == 12000


def test_relation_mask_metadata_is_fail_closed_and_v0_serialization_is_stable() -> None:
    with pytest.raises(LabelContractError, match="must both be present"):
        normalize_intervals(
            [
                {
                    "start_sample": 0,
                    "end_sample": 10,
                    "active_speakers": [],
                    "handoff_relation_mask_classes": [MASK_CLASS],
                }
            ]
        )
    plain = normalize_intervals(
        [{"start_sample": 0, "end_sample": 10, "active_speakers": []}]
    )[0]
    assert "handoff_relation_mask_classes" not in plain.to_dict()
    assert "mask_annotation_ids" not in plain.to_dict()
    masked = {
        "start_sample": 0,
        "end_sample": 10,
        "active_speakers": [],
        "handoff_relation_mask_classes": [MASK_CLASS],
        "mask_annotation_ids": ["fixture#mask"],
    }
    with pytest.raises(LabelContractError, match="v0 does not accept"):
        generate_labels([masked])
    with pytest.raises(LabelContractError, match="unsupported handoff/relation"):
        normalize_intervals(
            [
                {
                    **masked,
                    "handoff_relation_mask_classes": ["made_up_mask"],
                }
            ]
        )


def test_ami_inventory_masks_nonzero_point_reversed_and_unlocalized_events(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ES2003a.A.words.xml"
    path.write_text(
        """<nite:root xmlns:nite="http://nite.sourceforge.net/">
<w nite:id="w1" starttime="0" endtime="0.05">hello</w>
<vocalsound nite:id="v1" starttime="0.1" endtime="0.2" type="laugh"/>
<disfmarker nite:id="d1" starttime="0.2" endtime="0.2"/>
<vocalsound nite:id="v2" starttime="0.5" type="cough"/>
<gap nite:id="g1" starttime="0.6" endtime="0.7"/>
<vocalsound nite:id="v3" starttime="0.8" endtime="0.7" type="laugh"/>
<w nite:id="w2" starttime="0.9" endtime="0.95">left</w>
<vocalsound nite:id="v4" type="other"/>
<w nite:id="w3" starttime="1.1" endtime="1.2">right</w>
<transformerror nite:id="t1"/>
</nite:root>""",
        encoding="utf-8",
    )
    parsed = parse_ami_nonlexical_masks(
        "ES2003a",
        [path],
        speaker_map={"A": "MEE001"},
        scored_start_sample=0,
        scored_end_sample=32000,
    )
    assert parsed.observed_class_counts == {
        "element:disfmarker": 1,
        "element:gap": 1,
        "element:transformerror": 1,
        "element:vocalsound": 4,
        "element:w": 3,
        "mask:ami_nonzero_duration_vocalsound": 1,
        "mask:ami_point_or_zero_duration_vocalsound": 1,
        "mask:ami_reversed_vocalsound_bounds": 1,
        "mask:ami_unlocalized_vocalsound_neighbor_bounded": 1,
    }
    assert parsed.observed_marker_counts == {"cough": 1, "laugh": 2, "other": 1}
    assert [
        (mask.start_sample, mask.end_sample, mask.annotation_class)
        for mask in parsed.masks
    ] == [
        (800, 4000, "ami_nonzero_duration_vocalsound"),
        (4000, 12000, "ami_point_or_zero_duration_vocalsound"),
        (10400, 13600, "ami_reversed_vocalsound_bounds"),
        (14400, 18400, "ami_unlocalized_vocalsound_neighbor_bounded"),
    ]


def test_ami_inventory_rejects_unknown_elements_and_unbounded_vocalsounds(
    tmp_path: Path,
) -> None:
    unknown = tmp_path / "ES2003a.A.words.xml"
    unknown.write_text(
        '<nite:root xmlns:nite="http://nite.sourceforge.net/"><breath nite:id="x"/></nite:root>',
        encoding="utf-8",
    )
    with pytest.raises(ReferenceNormalizationError, match="unseen AMI"):
        parse_ami_nonlexical_masks(
            "ES2003a",
            [unknown],
            speaker_map={"A": "MEE001"},
            scored_start_sample=0,
            scored_end_sample=16000,
        )
    unknown.write_text(
        '<nite:root xmlns:nite="http://nite.sourceforge.net/"><vocalsound nite:id="x" type="laugh"/></nite:root>',
        encoding="utf-8",
    )
    with pytest.raises(ReferenceNormalizationError, match="no deterministic localization"):
        parse_ami_nonlexical_masks(
            "ES2003a",
            [unknown],
            speaker_map={"A": "MEE001"},
            scored_start_sample=0,
            scored_end_sample=16000,
        )


def test_ami_inventory_binds_unmapped_empty_speaker_placeholders(tmp_path: Path) -> None:
    mapped = tmp_path / "EN2002c.A.words.xml"
    mapped.write_text(
        '<nite:root xmlns:nite="http://nite.sourceforge.net/"><w nite:id="w1" starttime="0" endtime="0.1">word</w></nite:root>',
        encoding="utf-8",
    )
    placeholder = tmp_path / "EN2002c.D.words.xml"
    placeholder.write_text(
        '<nite:root xmlns:nite="http://nite.sourceforge.net/"/>',
        encoding="utf-8",
    )
    parsed = parse_ami_nonlexical_masks(
        "EN2002c",
        [mapped, placeholder],
        speaker_map={"A": "MEE001"},
        scored_start_sample=0,
        scored_end_sample=16000,
    )
    assert parsed.observed_class_counts == {
        "element:w": 1,
        "file:unmapped_empty_speaker_placeholder": 1,
    }
    placeholder.write_text(
        '<nite:root xmlns:nite="http://nite.sourceforge.net/"><w nite:id="w2" starttime="0" endtime="0.1">word</w></nite:root>',
        encoding="utf-8",
    )
    with pytest.raises(ReferenceNormalizationError, match="no official identity"):
        parse_ami_nonlexical_masks(
            "EN2002c",
            [mapped, placeholder],
            speaker_map={"A": "MEE001"},
            scored_start_sample=0,
            scored_end_sample=16000,
        )
def test_alimeeting_inventory_classifies_every_markup_action_and_masks_human_risk(
    tmp_path: Path,
) -> None:
    inventory = _inventory_with_markers(tmp_path)
    assert classify_alimeeting_text(" 词 [laugh] ", inventory) == (
        "mixed_lexical_human_vocal",
        ("[laugh]",),
    )
    assert classify_alimeeting_text("[noise]", inventory) == (
        "nonhuman_noise_marker_only",
        ("[noise]",),
    )
    with pytest.raises(ReferenceNormalizationError, match="unseen AliMeeting"):
        classify_alimeeting_text("[breath]", inventory)
    with pytest.raises(ReferenceNormalizationError, match="malformed"):
        classify_alimeeting_text("word [laugh", inventory)
    path = tmp_path / "R1.TextGrid"
    path.write_text(
        _textgrid(
            [
                ("0", "0.2", "词"),
                ("0.2", "0.4", "[laugh]"),
                ("0.4", "0.6", "词[laugh]"),
                ("0.6", "0.8", "[noise]"),
                ("0.8", "1", ""),
            ]
        ),
        encoding="utf-8",
    )
    parsed = parse_alimeeting_nonlexical_masks(
        "alimeeting_R1",
        path,
        speaker_map={"SPK1": "SPK1"},
        scored_start_sample=0,
        scored_end_sample=16000,
        inventory=inventory,
    )
    assert parsed.observed_class_counts == {
        "text:empty": 1,
        "text:human_vocal_marker_only": 1,
        "text:lexical_text": 1,
        "text:mixed_lexical_human_vocal": 1,
        "text:nonhuman_noise_marker_only": 1,
    }
    assert parsed.observed_marker_counts == {"[laugh]": 2, "[noise]": 1}
    assert [(mask.start_sample, mask.end_sample) for mask in parsed.masks] == [
        (3200, 6400),
        (6400, 9600),
    ]


def test_alimeeting_textgrid_identity_and_interval_structure_are_fail_closed(
    tmp_path: Path,
) -> None:
    inventory = _inventory_with_markers(tmp_path)
    path = tmp_path / "R1.TextGrid"
    path.write_text(_textgrid([("0", "0.6", "a"), ("0.5", "1", "b")]), encoding="utf-8")
    with pytest.raises(ReferenceNormalizationError, match="exceeds scored range"):
        parse_alimeeting_nonlexical_masks(
            "alimeeting_R1",
            path,
            speaker_map={"SPK1": "SPK1"},
            scored_start_sample=0,
            scored_end_sample=16000,
            inventory=inventory,
        )
    path.write_text(
        _textgrid([("0", "0.5", "a"), ("0.5", "1", "b")]).replace(
            "intervals [2]:", "intervals [1]:"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ReferenceNormalizationError, match="structure"):
        parse_alimeeting_nonlexical_masks(
            "alimeeting_R1",
            path,
            speaker_map={"SPK1": "SPK1"},
            scored_start_sample=0,
            scored_end_sample=16000,
            inventory=inventory,
        )
    with pytest.raises(ReferenceNormalizationError, match="session identity"):
        parse_alimeeting_nonlexical_masks(
            "alimeeting_R2",
            path,
            speaker_map={"SPK1": "SPK1"},
            scored_start_sample=0,
            scored_end_sample=16000,
            inventory=inventory,
        )


def test_inventory_dataclass_injection_is_rejected(tmp_path: Path) -> None:
    inventory = replace(
        _inventory_with_markers(tmp_path),
        _validation_token=object(),
    )
    with pytest.raises(ReferenceNormalizationError, match="was not validated"):
        classify_alimeeting_text("[laugh]", inventory)


def test_reference_session_uses_rttm_not_textgrid_as_activity_authority(
    tmp_path: Path,
    monkeypatch,
) -> None:
    corpus_root, reference_root, textgrid, source_row, provenance = (
        _alimeeting_fixture(tmp_path, monkeypatch)
    )
    checkout = open_reference_checkout(reference_root)
    with pytest.raises(TypeError):
        checkout.provenance["git_tree"] = "4" * 40
    normalized = normalize_reference_session(source_row, corpus_root, checkout)
    assert normalized.labels.contract_version == "psem-handoff-v1"
    assert [row["state"] for row in normalized.labels.activity_labels] == [
        "silence",
        "singleton",
        "silence",
    ]
    assert all(row["mask_state"] == "valid" for row in normalized.labels.activity_labels)
    assert normalized.manifest_row()["reference_ref"] == (
        "AliMeeting/Eval_Ali_far/R1.rttm"
    )
    assert normalized.manifest_row()["nonlexical_mask_count"] == 0
    assert normalized.manifest_row()["source_id"] == "alimeeting_R1"
    assert normalized.manifest_row()["source_record_sha256"] == canonical_sha256(
        source_row
    )
    assert normalized.manifest_row()["reference_git_tree"] == provenance["git_tree"]
    with pytest.raises(TypeError):
        normalized.reference_checkout_provenance["git_tree"] = "4" * 40
    assert normalized.manifest_row()["reference_metadata_files"] == [
        {
            "role": "speaker_and_nonlexical_annotation",
            "ref": "alimeeting/Eval_Ali/Eval_Ali_far/textgrid_dir/R1.TextGrid",
            "sha256": normalized.metadata_files[0].sha256,
            "size_bytes": textgrid.stat().st_size,
        }
    ]
    assert len(normalized.manifest_row()["speaker_mapping_sha256"]) == 64

    forged = replace(
        checkout,
        provenance={**provenance, "git_tree": "4" * 40},
    )
    with pytest.raises(ReferenceNormalizationError, match="was not validated"):
        normalize_reference_session(source_row, corpus_root, forged)
    forged_root = replace(checkout, root=tmp_path / "replacement")
    with pytest.raises(ReferenceNormalizationError, match="was not validated"):
        normalize_reference_session(source_row, corpus_root, forged_root)

    bad_source = {**source_row, "annotation_sha256": "0" * 64}
    with pytest.raises(ReferenceNormalizationError, match="annotation receipt"):
        normalize_reference_session(bad_source, corpus_root, checkout)
    reference_path = reference_root / "AliMeeting/Eval_Ali_far/R1.rttm"
    train_source = {
        **source_row,
        "meeting_type": "natural_meeting_train_partition",
        "reference_ref": "AliMeeting/Eval_Ali_far/R1.rttm",
        "reference_sha256": sha256_file(reference_path),
    }
    normalize_reference_session(train_source, corpus_root, checkout)
    with pytest.raises(ReferenceNormalizationError, match="source reference receipt"):
        normalize_reference_session(
            {**train_source, "reference_sha256": "0" * 64},
            corpus_root,
            checkout,
        )

    for field, value in (
        ("sample_rate_hz", 16000.0),
        ("waveform_size_bytes", float(source_row["waveform_size_bytes"])),
        ("annotation_coverage_start_sample", False),
    ):
        with pytest.raises(ReferenceNormalizationError, match="identity is invalid"):
            normalize_reference_session(
                {**source_row, field: value},
                corpus_root,
                checkout,
            )


def test_reference_session_binds_declared_textgrid_tail_to_observed_bytes(
    tmp_path: Path, monkeypatch
) -> None:
    corpus_root, reference_root, textgrid, source_row, _ = _alimeeting_fixture(
        tmp_path, monkeypatch
    )
    textgrid.write_text(
        _textgrid([("0", "1.01", "lexical")]).replace(
            "xmax = 1\n", "xmax = 1.01\n"
        ),
        encoding="utf-8",
    )
    annotation_files = [
        {
            "ref": textgrid.relative_to(corpus_root).as_posix(),
            "size_bytes": textgrid.stat().st_size,
            "sha256": sha256_file(textgrid),
        }
    ]
    source_row = {
        **source_row,
        "annotation_sha256": canonical_sha256(annotation_files),
        "annotation_tail_excess_samples": 160,
    }
    checkout = open_reference_checkout(reference_root)

    normalize_reference_session(source_row, corpus_root, checkout)
    with pytest.raises(ReferenceNormalizationError, match="tail receipt"):
        normalize_reference_session(
            {**source_row, "annotation_tail_excess_samples": 0},
            corpus_root,
            checkout,
        )


def test_reference_inventory_binds_manifest_and_resolves_every_session(
    tmp_path: Path,
    monkeypatch,
) -> None:
    corpus_root, reference_root, _, source_row, _ = _alimeeting_fixture(
        tmp_path, monkeypatch
    )
    source_manifest = tmp_path / "source_manifest.jsonl"
    source_manifest.write_text(
        json.dumps(source_row, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    sessions = normalize_reference_inventory(
        source_manifest,
        corpus_root,
        reference_root,
    )
    assert [session.source_id for session in sessions] == ["alimeeting_R1"]


def test_reference_inventory_rejects_checkout_mutation_during_normalization(
    tmp_path: Path,
    monkeypatch,
) -> None:
    corpus_root, reference_root, _, source_row, provenance = _alimeeting_fixture(
        tmp_path,
        monkeypatch,
    )
    source_manifest = tmp_path / "source_manifest.jsonl"
    source_manifest.write_text(
        json.dumps(source_row, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    reference_path = reference_root / "AliMeeting/Eval_Ali_far/R1.rttm"
    expected_sha256 = sha256_file(reference_path)

    def validate_checkout(_: Path) -> dict:
        if sha256_file(reference_path) != expected_sha256:
            raise ForcedAlignmentReferenceError("checkout bytes changed")
        return provenance

    original_normalize = reference_normalization.normalize_reference_session

    def mutate_after_normalize(*args, **kwargs):
        session = original_normalize(*args, **kwargs)
        reference_path.write_text("mutated\n", encoding="utf-8")
        return session

    monkeypatch.setattr(
        reference_normalization,
        "validate_reference_checkout",
        validate_checkout,
    )
    monkeypatch.setattr(
        reference_normalization,
        "normalize_reference_session",
        mutate_after_normalize,
    )
    with pytest.raises(
        ReferenceNormalizationError,
        match="changed during normalization",
    ):
        normalize_reference_inventory(
            source_manifest,
            corpus_root,
            reference_root,
        )
