from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.psem_training_strategy_gate.data.annotation_normalization import (
    AnnotationNormalizationError,
    AnnotationSpan,
    normalize_source,
    parse_alimeeting_annotations,
    parse_ami_annotations,
    spans_to_canonical_intervals,
    timestamp_to_sample,
)
from experiments.psem_training_strategy_gate.data.label_contract import load_contract
from experiments.psem_training_strategy_gate.data.provenance import (
    EXPECTED_ALIMEETING_MEETINGS,
    EXPECTED_AMI_MEETINGS,
    canonical_sha256,
    sha256_file,
)

DATA_DIR = Path(__file__).resolve().parents[1]
EXPECTED_SOURCE_COUNT = len(EXPECTED_AMI_MEETINGS) + len(EXPECTED_ALIMEETING_MEETINGS)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_timestamp_conversion_is_sample_exact_and_rejects_nonfinite_values() -> None:
    assert timestamp_to_sample("1.00003125", "timestamp") == 16000
    assert timestamp_to_sample("1.00009375", "timestamp") == 16002
    with pytest.raises(AnnotationNormalizationError, match="non-finite"):
        timestamp_to_sample("NaN", "timestamp")


def test_checked_in_normalization_manifest_binds_every_source_and_annotation() -> None:
    sources = {row["source_id"]: row for row in read_jsonl(DATA_DIR / "source_manifest.jsonl")}
    annotations = {
        row["source_id"]: row for row in read_jsonl(DATA_DIR / "annotation_manifest.jsonl")
    }
    normalized = {
        row["source_id"]: row
        for row in read_jsonl(DATA_DIR / "normalization_manifest.jsonl")
    }
    assert len(normalized) == EXPECTED_SOURCE_COUNT
    assert set(normalized) == set(sources) == set(annotations)
    for source_id, row in normalized.items():
        source = sources[source_id]
        annotation = annotations[source_id]
        assert row["corpus"] == source["corpus"] == annotation["corpus"]
        assert row["session_id"] == source["session_id"] == annotation["session_id"]
        assert row["source_waveform_sha256"] == source["waveform_sha256"]
        assert row["annotation_sha256"] == annotation["annotation_sha256"]
        assert row["scored_start_sample"] == annotation["coverage_start_sample"]
        assert row["scored_end_sample"] == annotation["coverage_end_sample"]
        assert row["exposure"]["scored_samples"] == (
            row["scored_end_sample"] - row["scored_start_sample"]
        )
        assert row["canonical_interval_count"] > 0
        assert row["activity_label_count"] > 0
        assert row["raw_speech_span_count"] > 0
        assert len(row["canonical_intervals_sha256"]) == 64
        assert len(row["label_result_sha256"]) == 64


def test_span_sweep_builds_complete_silence_overlap_and_unknown_timeline() -> None:
    intervals = spans_to_canonical_intervals(
        [
            AnnotationSpan(0, 100, "A", True, "a1"),
            AnnotationSpan(50, 150, "B", True, "b1"),
            AnnotationSpan(125, 175, "unknown:meeting:C", False, "c1"),
        ],
        scored_start_sample=0,
        scored_end_sample=200,
    )
    assert [
        (
            row.start_sample,
            row.end_sample,
            row.active_speakers,
            row.speaker_identity_known,
            row.source_annotation_ids,
        )
        for row in intervals
    ] == [
        (0, 50, ("A",), True, ("a1",)),
        (50, 100, ("A", "B"), True, ("a1", "b1")),
        (100, 125, ("B",), True, ("b1",)),
        (125, 150, ("B", "unknown:meeting:C"), False, ("b1", "c1")),
        (150, 175, ("unknown:meeting:C",), False, ("c1",)),
        (175, 200, (), True, ()),
    ]


def test_ami_parser_maps_global_identities_and_clips_only_the_waveform_tail(
    tmp_path: Path,
) -> None:
    meetings = tmp_path / "meetings.xml"
    meetings.write_text(
        """<root><meeting observation="ES2003a"><speaker nxt_agent="A" global_name="MEE001"/><speaker nxt_agent="B"/></meeting></root>""",
        encoding="utf-8",
    )
    a_path = tmp_path / "ES2003a.A.segments.xml"
    b_path = tmp_path / "ES2003a.B.segments.xml"
    a_path.write_text(
        """<root><segment id="s1" transcriber_start="0.1" transcriber_end="0.4"/></root>""",
        encoding="utf-8",
    )
    b_path.write_text(
        """<root><segment id="s2" transcriber_start="0.8" transcriber_end="1.1"/></root>""",
        encoding="utf-8",
    )
    parsed = parse_ami_annotations(
        "ES2003a",
        meetings,
        [a_path, b_path],
        scored_start_sample=0,
        scored_end_sample=16000,
    )
    assert parsed.raw_speech_span_count == 2
    assert parsed.clipped_span_count == 1
    assert parsed.spans[0] == AnnotationSpan(
        1600, 6400, "MEE001", True, "ES2003a.A.segments.xml#s1"
    )
    assert parsed.spans[1] == AnnotationSpan(
        12800,
        16000,
        "unknown:ES2003a:B",
        False,
        "ES2003a.B.segments.xml#s2",
    )


def test_ami_parser_rejects_annotations_beyond_the_tail_tolerance(tmp_path: Path) -> None:
    meetings = tmp_path / "meetings.xml"
    meetings.write_text(
        '<root><meeting observation="ES2003a"><speaker nxt_agent="A" global_name="MEE001"/></meeting></root>',
        encoding="utf-8",
    )
    path = tmp_path / "ES2003a.A.segments.xml"
    path.write_text(
        '<root><segment id="s1" transcriber_start="0.8" transcriber_end="3.1"/></root>',
        encoding="utf-8",
    )
    with pytest.raises(AnnotationNormalizationError, match="tail tolerance"):
        parse_ami_annotations(
            "ES2003a",
            meetings,
            [path],
            scored_start_sample=0,
            scored_end_sample=16000,
        )


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [
        ("-0.00001", "0.5", "begins outside"),
        ("0.8", "3.00001", "tail tolerance"),
    ],
)
def test_ami_parser_validates_raw_bounds_before_sample_rounding(
    tmp_path: Path, start: str, end: str, message: str
) -> None:
    meetings = tmp_path / "meetings.xml"
    meetings.write_text(
        '<root><meeting observation="ES2003a"><speaker nxt_agent="A" global_name="MEE001"/></meeting></root>',
        encoding="utf-8",
    )
    path = tmp_path / "ES2003a.A.segments.xml"
    path.write_text(
        f'<root><segment id="s1" transcriber_start="{start}" transcriber_end="{end}"/></root>',
        encoding="utf-8",
    )
    with pytest.raises(AnnotationNormalizationError, match=message):
        parse_ami_annotations(
            "ES2003a",
            meetings,
            [path],
            scored_start_sample=0,
            scored_end_sample=16000,
        )


def test_ami_parser_counts_a_subsample_annotation_tail_as_clipped(tmp_path: Path) -> None:
    meetings = tmp_path / "meetings.xml"
    meetings.write_text(
        '<root><meeting observation="ES2003a"><speaker nxt_agent="A" global_name="MEE001"/></meeting></root>',
        encoding="utf-8",
    )
    path = tmp_path / "ES2003a.A.segments.xml"
    path.write_text(
        '<root><segment id="s1" transcriber_start="0.8" transcriber_end="1.000001"/></root>',
        encoding="utf-8",
    )
    parsed = parse_ami_annotations(
        "ES2003a",
        meetings,
        [path],
        scored_start_sample=0,
        scored_end_sample=16000,
    )
    assert parsed.clipped_span_count == 1
    assert parsed.spans[0].end_sample == 16000


def textgrid_payload() -> str:
    return "\n".join(
        [
            'File type = "ooTextFile"',
            'Object class = "TextGrid"',
            "",
            "xmin = 0",
            "xmax = 1",
            "tiers? <exists>",
            "size = 2",
            "item []:",
            "  item [1]:",
            '    class = "IntervalTier"',
            '    name = "N_SPK1"',
            "    xmin = 0",
            "    xmax = 1",
            "    intervals: size = 2",
            "    intervals [1]:",
            "      xmin = 0",
            "      xmax = 0.25",
            '      text = ""',
            "    intervals [2]:",
            "      xmin = 0.25",
            "      xmax = 0.75",
            '      text = "speech"',
            "  item [2]:",
            '    class = "IntervalTier"',
            '    name = "unresolved"',
            "    xmin = 0",
            "    xmax = 1",
            "    intervals: size = 1",
            "    intervals [1]:",
            "      xmin = 0.5",
            "      xmax = 1",
            '      text = "speech"',
        ]
    )


def test_alimeeting_parser_preserves_tier_interval_ids_and_unknown_identity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "R8001_M8004.TextGrid"
    path.write_text(textgrid_payload(), encoding="utf-8")
    parsed = parse_alimeeting_annotations(
        "alimeeting_R8001_M8004",
        path,
        scored_start_sample=0,
        scored_end_sample=16000,
    )
    assert parsed.raw_speech_span_count == 2
    assert parsed.clipped_span_count == 0
    assert parsed.spans == (
        AnnotationSpan(
            4000,
            12000,
            "SPK1",
            True,
            "R8001_M8004.TextGrid#item[1].intervals[2]",
        ),
        AnnotationSpan(
            8000,
            16000,
            "unknown:alimeeting_R8001_M8004:tier:2",
            False,
            "R8001_M8004.TextGrid#item[2].intervals[1]",
        ),
    )


def test_alimeeting_parser_requires_manifest_coverage_to_match_textgrid(
    tmp_path: Path,
) -> None:
    path = tmp_path / "R8001_M8004.TextGrid"
    path.write_text(textgrid_payload(), encoding="utf-8")
    with pytest.raises(AnnotationNormalizationError, match="scored range"):
        parse_alimeeting_annotations(
            "alimeeting_R8001_M8004",
            path,
            scored_start_sample=0,
            scored_end_sample=15999,
        )


def test_normalize_source_binds_manifest_bytes_and_applies_the_label_generator(
    tmp_path: Path,
) -> None:
    contract = load_contract()
    ref = "alimeeting/R8001_M8004.TextGrid"
    path = tmp_path / ref
    path.parent.mkdir()
    path.write_text(textgrid_payload(), encoding="utf-8")
    file_row = {
        "ref": ref,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    annotation_sha256 = canonical_sha256([file_row])
    source_row = {
        "source_id": "alimeeting_R8001_M8004",
        "corpus": "AliMeeting",
        "session_id": "R8001_M8004",
        "duration_samples": 16000,
        "sample_rate_hz": 16000,
        "annotation_coverage_start_sample": 0,
        "annotation_coverage_end_sample": 16000,
        "annotation_ref": ref,
        "waveform_sha256": "a" * 64,
        "annotation_sha256": annotation_sha256,
        "contract_version": contract.contract_version,
        "contract_document_sha256": contract.document_sha256,
    }
    annotation_row = {
        "source_id": "alimeeting_R8001_M8004",
        "corpus": "AliMeeting",
        "session_id": "R8001_M8004",
        "coverage_start_sample": 0,
        "coverage_end_sample": 16000,
        "annotation_files": [file_row],
        "annotation_sha256": annotation_sha256,
        "contract_version": contract.contract_version,
        "contract_document_sha256": contract.document_sha256,
    }
    normalized = normalize_source(source_row, annotation_row, tmp_path)
    assert normalized.labels.contract_version == contract.contract_version
    assert normalized.labels.intervals == normalized.intervals
    assert normalized.labels.exposure["scored_samples"] == 16000
    assert normalized.manifest_row()["label_result_sha256"]
    with pytest.raises(AnnotationNormalizationError, match="coverage range"):
        normalize_source({**source_row, "sample_rate_hz": 8000}, annotation_row, tmp_path)
    with pytest.raises(AnnotationNormalizationError, match="bundle structure"):
        normalize_source(
            {**source_row, "annotation_coverage_end_sample": 15999},
            annotation_row,
            tmp_path,
        )
    with pytest.raises(AnnotationNormalizationError, match="bundle structure"):
        normalize_source(
            {**source_row, "annotation_ref": "alimeeting/other.TextGrid"},
            annotation_row,
            tmp_path,
        )
    with pytest.raises(AnnotationNormalizationError, match="source identity"):
        normalize_source(
            {**source_row, "source_id": "bad-source"},
            {**annotation_row, "source_id": "bad-source"},
            tmp_path,
        )


def test_normalize_source_rejects_wrong_session_and_extra_annotation_files(
    tmp_path: Path,
) -> None:
    contract = load_contract()
    ref = "alimeeting/WRONG_SESSION.TextGrid"
    path = tmp_path / ref
    path.parent.mkdir()
    path.write_text(textgrid_payload(), encoding="utf-8")
    file_row = {
        "ref": ref,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    annotation_sha256 = canonical_sha256([file_row])
    source_row = {
        "source_id": "alimeeting_R8001_M8004",
        "corpus": "AliMeeting",
        "session_id": "R8001_M8004",
        "duration_samples": 16000,
        "sample_rate_hz": 16000,
        "annotation_coverage_start_sample": 0,
        "annotation_coverage_end_sample": 16000,
        "annotation_ref": ref,
        "waveform_sha256": "a" * 64,
        "annotation_sha256": annotation_sha256,
        "contract_version": contract.contract_version,
        "contract_document_sha256": contract.document_sha256,
    }
    annotation_row = {
        "source_id": "alimeeting_R8001_M8004",
        "corpus": "AliMeeting",
        "session_id": "R8001_M8004",
        "coverage_start_sample": 0,
        "coverage_end_sample": 16000,
        "annotation_files": [file_row],
        "annotation_sha256": annotation_sha256,
        "contract_version": contract.contract_version,
        "contract_document_sha256": contract.document_sha256,
    }
    with pytest.raises(AnnotationNormalizationError, match="bundle structure"):
        normalize_source(source_row, annotation_row, tmp_path)
    correct_ref = "alimeeting/R8001_M8004.TextGrid"
    correct_path = tmp_path / correct_ref
    correct_path.write_text(textgrid_payload(), encoding="utf-8")
    correct_row = {
        "ref": correct_ref,
        "sha256": sha256_file(correct_path),
        "size_bytes": correct_path.stat().st_size,
    }
    extra_path = tmp_path / "alimeeting" / "extra.xml"
    extra_path.write_text("<root/>", encoding="utf-8")
    extra_row = {
        "ref": "alimeeting/extra.xml",
        "sha256": sha256_file(extra_path),
        "size_bytes": extra_path.stat().st_size,
    }
    files = [correct_row, extra_row]
    extra_sha256 = canonical_sha256(files)
    with pytest.raises(AnnotationNormalizationError, match="bundle structure"):
        normalize_source(
            {
                **source_row,
                "annotation_ref": correct_ref,
                "annotation_sha256": extra_sha256,
            },
            {
                **annotation_row,
                "annotation_files": files,
                "annotation_sha256": extra_sha256,
            },
            tmp_path,
        )


def test_normalize_source_requires_full_waveform_coverage_for_ami(tmp_path: Path) -> None:
    contract = load_contract()
    meeting_ref = "ami/annotations/corpusResources/meetings.xml"
    segment_ref = "ami/annotations/segments/ES2003a.A.segments.xml"
    meeting_path = tmp_path / meeting_ref
    segment_path = tmp_path / segment_ref
    segment_path.parent.mkdir(parents=True)
    meeting_path.parent.mkdir(parents=True, exist_ok=True)
    meeting_path.write_text(
        '<root><meeting observation="ES2003a"><speaker nxt_agent="A" global_name="MEE001"/></meeting></root>',
        encoding="utf-8",
    )
    segment_path.write_text(
        '<root><segment id="s1" transcriber_start="0.1" transcriber_end="0.4"/></root>',
        encoding="utf-8",
    )
    file_rows = [
        {
            "ref": ref,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for ref, path in ((meeting_ref, meeting_path), (segment_ref, segment_path))
    ]
    annotation_sha256 = canonical_sha256(file_rows)
    source_row = {
        "source_id": "ami_ES2003a",
        "corpus": "AMI",
        "session_id": "ES2003a",
        "duration_samples": 16000,
        "sample_rate_hz": 16000,
        "annotation_ref": "ami/annotations/segments/ES2003a.*.segments.xml",
        "waveform_sha256": "a" * 64,
        "annotation_sha256": annotation_sha256,
        "contract_version": contract.contract_version,
        "contract_document_sha256": contract.document_sha256,
    }
    annotation_row = {
        "source_id": "ami_ES2003a",
        "corpus": "AMI",
        "session_id": "ES2003a",
        "coverage_start_sample": 0,
        "coverage_end_sample": 15000,
        "annotation_files": file_rows,
        "annotation_sha256": annotation_sha256,
        "contract_version": contract.contract_version,
        "contract_document_sha256": contract.document_sha256,
    }
    with pytest.raises(AnnotationNormalizationError, match="bundle structure"):
        normalize_source(source_row, annotation_row, tmp_path)
