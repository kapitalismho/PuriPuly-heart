from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from experiments.psem_training_strategy_gate.data.label_contract import (
    CanonicalInterval,
    LabelResult,
    generate_labels,
    load_contract,
    normalize_intervals,
)
from experiments.psem_training_strategy_gate.data.provenance import (
    SOURCE_ID_PATTERN,
    _parse_alimeeting_textgrid,
    canonical_sha256,
    sha256_file,
    write_jsonl,
)

SAMPLE_RATE_HZ = 16000
AMI_ANNOTATION_TAIL_TOLERANCE_SAMPLES = 32000
AMI_SEGMENT_NAME = re.compile(
    r"^(?P<meeting>[A-Za-z0-9]+)\.(?P<agent>[A-Za-z0-9]+)\.segments\.xml$"
)
ALIMEETING_TIER_NAME = re.compile(r"^N_(SPK\d+)$")
TEXTGRID_ITEM = re.compile(r"\s*item \[(\d+)\]:\s*")
TEXTGRID_INTERVAL = re.compile(r"\s*intervals \[(\d+)\]:\s*")


class AnnotationNormalizationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class AnnotationSpan:
    start_sample: int
    end_sample: int
    speaker_id: str
    speaker_identity_known: bool
    source_annotation_id: str


@dataclass(frozen=True, slots=True)
class ParsedAnnotations:
    spans: tuple[AnnotationSpan, ...]
    raw_speech_span_count: int
    clipped_span_count: int


@dataclass(frozen=True, slots=True)
class NormalizedSession:
    source_id: str
    corpus: str
    session_id: str
    scored_start_sample: int
    scored_end_sample: int
    source_waveform_sha256: str
    annotation_sha256: str
    raw_speech_span_count: int
    clipped_span_count: int
    intervals: tuple[CanonicalInterval, ...]
    labels: LabelResult

    def manifest_row(self) -> dict[str, Any]:
        interval_rows = [interval.to_dict() for interval in self.intervals]
        label_payload = self.labels.to_dict()
        return {
            "schema_version": 1,
            "source_id": self.source_id,
            "corpus": self.corpus,
            "session_id": self.session_id,
            "contract_version": self.labels.contract_version,
            "contract_document_sha256": self.labels.contract_document_sha256,
            "source_waveform_sha256": self.source_waveform_sha256,
            "annotation_sha256": self.annotation_sha256,
            "scored_start_sample": self.scored_start_sample,
            "scored_end_sample": self.scored_end_sample,
            "raw_speech_span_count": self.raw_speech_span_count,
            "clipped_span_count": self.clipped_span_count,
            "canonical_interval_count": len(self.intervals),
            "canonical_intervals_sha256": canonical_sha256(interval_rows),
            "activity_label_count": len(self.labels.activity_labels),
            "transition_count": len(self.labels.transitions),
            "topology_episode_count": len(self.labels.topology_episodes),
            "label_result_sha256": canonical_sha256(label_payload),
            "exposure": dict(self.labels.exposure),
        }


def _parse_timestamp(value: str, field: str) -> Decimal:
    try:
        timestamp = Decimal(value)
    except InvalidOperation as exc:
        raise AnnotationNormalizationError(f"invalid timestamp for {field}") from exc
    if not timestamp.is_finite():
        raise AnnotationNormalizationError(f"non-finite timestamp for {field}")
    return timestamp


def timestamp_to_sample(value: str, field: str) -> int:
    timestamp = _parse_timestamp(value, field)
    sample = timestamp * SAMPLE_RATE_HZ
    return int(sample.to_integral_value(rounding=ROUND_HALF_EVEN))


def _annotation_id(element: ET.Element, path: Path) -> str:
    value = next(
        (
            attribute_value
            for attribute_name, attribute_value in element.attrib.items()
            if attribute_name == "id" or attribute_name.endswith("}id")
        ),
        None,
    )
    if not value:
        raise AnnotationNormalizationError(f"AMI segment lacks an annotation ID: {path}")
    return f"{path.name}#{value}"


def _require_unique_annotation_ids(spans: Sequence[AnnotationSpan]) -> None:
    annotation_ids = [span.source_annotation_id for span in spans]
    if len(set(annotation_ids)) != len(annotation_ids):
        raise AnnotationNormalizationError("source annotation IDs must be unique")


def _ami_speaker_map(meetings_path: Path, meeting_id: str) -> dict[str, tuple[str, bool]]:
    try:
        root = ET.parse(meetings_path).getroot()
    except ET.ParseError as exc:
        raise AnnotationNormalizationError(
            f"invalid AMI meetings metadata: {meetings_path}"
        ) from exc
    matches = [element for element in root.iter("meeting") if element.get("observation") == meeting_id]
    if len(matches) != 1:
        raise AnnotationNormalizationError(
            f"AMI meeting metadata cardinality mismatch: {meeting_id}"
        )
    speakers: dict[str, tuple[str, bool]] = {}
    for element in matches[0].findall("speaker"):
        agent = element.get("nxt_agent")
        if not agent or agent in speakers:
            raise AnnotationNormalizationError(
                f"invalid AMI speaker agent metadata: {meeting_id}"
            )
        global_name = element.get("global_name")
        if global_name:
            speakers[agent] = (global_name, True)
        else:
            speakers[agent] = (f"unknown:{meeting_id}:{agent}", False)
    if not speakers:
        raise AnnotationNormalizationError(f"AMI meeting has no speaker metadata: {meeting_id}")
    return speakers


def parse_ami_annotations(
    meeting_id: str,
    meetings_path: Path,
    segment_paths: Sequence[Path],
    *,
    scored_start_sample: int,
    scored_end_sample: int,
) -> ParsedAnnotations:
    speaker_map = _ami_speaker_map(meetings_path, meeting_id)
    spans: list[AnnotationSpan] = []
    clipped = 0
    found_agents = set()
    scored_start_time = Decimal(scored_start_sample) / SAMPLE_RATE_HZ
    scored_end_time = Decimal(scored_end_sample) / SAMPLE_RATE_HZ
    annotation_tail_end_time = (
        Decimal(scored_end_sample + AMI_ANNOTATION_TAIL_TOLERANCE_SAMPLES)
        / SAMPLE_RATE_HZ
    )
    for path in sorted(segment_paths):
        match = AMI_SEGMENT_NAME.fullmatch(path.name)
        if match is None or match.group("meeting") != meeting_id:
            raise AnnotationNormalizationError(f"unexpected AMI segment filename: {path.name}")
        agent = match.group("agent")
        found_agents.add(agent)
        identity = speaker_map.get(agent)
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError as exc:
            raise AnnotationNormalizationError(f"invalid AMI segment XML: {path}") from exc
        segments = list(root.iter("segment"))
        if segments and identity is None:
            raise AnnotationNormalizationError(
                f"AMI segment speaker has no metadata identity: {meeting_id}.{agent}"
            )
        for element in segments:
            try:
                start_time = _parse_timestamp(
                    element.attrib["transcriber_start"], "transcriber_start"
                )
                end_time = _parse_timestamp(
                    element.attrib["transcriber_end"], "transcriber_end"
                )
            except KeyError as exc:
                raise AnnotationNormalizationError(
                    f"AMI segment lacks timestamp bounds: {path}"
                ) from exc
            if start_time < scored_start_time or start_time >= scored_end_time:
                raise AnnotationNormalizationError(
                    f"AMI segment begins outside the scored range: {path}"
                )
            if end_time <= start_time:
                raise AnnotationNormalizationError(f"AMI segment has invalid bounds: {path}")
            if end_time > annotation_tail_end_time:
                raise AnnotationNormalizationError(
                    f"AMI segment exceeds the annotation tail tolerance: {path}"
                )
            start_sample = timestamp_to_sample(str(start_time), "transcriber_start")
            end_sample = timestamp_to_sample(str(end_time), "transcriber_end")
            if end_time > scored_end_time:
                end_sample = scored_end_sample
                clipped += 1
            if end_sample <= start_sample:
                raise AnnotationNormalizationError(f"AMI segment has invalid bounds: {path}")
            speaker_id, identity_known = identity
            spans.append(
                AnnotationSpan(
                    start_sample=start_sample,
                    end_sample=end_sample,
                    speaker_id=speaker_id,
                    speaker_identity_known=identity_known,
                    source_annotation_id=_annotation_id(element, path),
                )
            )
    if not set(speaker_map).issubset(found_agents):
        raise AnnotationNormalizationError(
            f"AMI segment bundle is missing speaker files: {meeting_id}"
        )
    if not spans:
        raise AnnotationNormalizationError(f"AMI segment bundle is empty: {meeting_id}")
    _require_unique_annotation_ids(spans)
    return ParsedAnnotations(
        spans=tuple(sorted(spans, key=_span_order)),
        raw_speech_span_count=len(spans),
        clipped_span_count=clipped,
    )


def _textgrid_value(lines: Sequence[str], key: str) -> str:
    prefix = f"{key} ="
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(prefix):
            value = stripped.split("=", 1)[1].strip()
            if len(value) >= 2 and value.startswith('"') and value.endswith('"'):
                return value[1:-1]
            return value
    raise AnnotationNormalizationError(f"TextGrid field missing: {key}")


def parse_alimeeting_annotations(
    source_id: str,
    path: Path,
    *,
    scored_start_sample: int,
    scored_end_sample: int,
) -> ParsedAnnotations:
    metadata = _parse_alimeeting_textgrid(path)
    if (
        metadata["timeline_start_sample"] != scored_start_sample
        or metadata["timeline_end_sample"] != scored_end_sample
    ):
        raise AnnotationNormalizationError(
            f"AliMeeting TextGrid timeline does not match the scored range: {path}"
        )
    lines = path.read_text(encoding="utf-8").splitlines()
    item_matches = [
        (index, match)
        for index, line in enumerate(lines)
        if (match := TEXTGRID_ITEM.fullmatch(line))
    ]
    spans: list[AnnotationSpan] = []
    for position, (start, item_match) in enumerate(item_matches):
        end = item_matches[position + 1][0] if position + 1 < len(item_matches) else len(lines)
        block = lines[start:end]
        tier_name = _textgrid_value(block, "name")
        tier_match = ALIMEETING_TIER_NAME.fullmatch(tier_name)
        if tier_match is None:
            speaker_id = f"unknown:{source_id}:tier:{item_match.group(1)}"
            identity_known = False
        else:
            speaker_id = tier_match.group(1)
            identity_known = True
        interval_matches = [
            (index, match)
            for index, line in enumerate(block)
            if (match := TEXTGRID_INTERVAL.fullmatch(line))
        ]
        for interval_position, (interval_start, interval_match) in enumerate(interval_matches):
            interval_end = (
                interval_matches[interval_position + 1][0]
                if interval_position + 1 < len(interval_matches)
                else len(block)
            )
            interval = block[interval_start:interval_end]
            text = _textgrid_value(interval, "text")
            if not text.strip():
                continue
            start_sample = timestamp_to_sample(_textgrid_value(interval, "xmin"), "xmin")
            end_sample = timestamp_to_sample(_textgrid_value(interval, "xmax"), "xmax")
            if (
                start_sample < scored_start_sample
                or end_sample > scored_end_sample
                or end_sample <= start_sample
            ):
                raise AnnotationNormalizationError(
                    f"AliMeeting speech interval exceeds the scored range: {path}"
                )
            spans.append(
                AnnotationSpan(
                    start_sample=start_sample,
                    end_sample=end_sample,
                    speaker_id=speaker_id,
                    speaker_identity_known=identity_known,
                    source_annotation_id=(
                        f"{path.name}#item[{item_match.group(1)}]."
                        f"intervals[{interval_match.group(1)}]"
                    ),
                )
            )
    if not spans:
        raise AnnotationNormalizationError(f"AliMeeting TextGrid has no speech: {path}")
    _require_unique_annotation_ids(spans)
    return ParsedAnnotations(
        spans=tuple(sorted(spans, key=_span_order)),
        raw_speech_span_count=len(spans),
        clipped_span_count=0,
    )


def _span_order(span: AnnotationSpan) -> tuple[int, int, str, str]:
    return (
        span.start_sample,
        span.end_sample,
        span.speaker_id,
        span.source_annotation_id,
    )


def spans_to_canonical_intervals(
    spans: Iterable[AnnotationSpan],
    *,
    scored_start_sample: int,
    scored_end_sample: int,
) -> tuple[CanonicalInterval, ...]:
    if (
        isinstance(scored_start_sample, bool)
        or not isinstance(scored_start_sample, int)
        or isinstance(scored_end_sample, bool)
        or not isinstance(scored_end_sample, int)
        or scored_start_sample < 0
        or scored_end_sample <= scored_start_sample
    ):
        raise AnnotationNormalizationError("invalid scored source range")
    ordered_spans = tuple(sorted(spans, key=_span_order))
    if not ordered_spans:
        raise AnnotationNormalizationError("at least one speech annotation span is required")
    starts: dict[int, list[int]] = defaultdict(list)
    ends: dict[int, list[int]] = defaultdict(list)
    boundaries = {scored_start_sample, scored_end_sample}
    for index, span in enumerate(ordered_spans):
        if (
            isinstance(span.start_sample, bool)
            or not isinstance(span.start_sample, int)
            or isinstance(span.end_sample, bool)
            or not isinstance(span.end_sample, int)
            or span.start_sample < scored_start_sample
            or span.end_sample > scored_end_sample
            or span.end_sample <= span.start_sample
            or not span.speaker_id
            or not span.source_annotation_id
        ):
            raise AnnotationNormalizationError("invalid annotation span")
        starts[span.start_sample].append(index)
        ends[span.end_sample].append(index)
        boundaries.add(span.start_sample)
        boundaries.add(span.end_sample)
    ordered_boundaries = sorted(boundaries)
    active: set[int] = set()
    intervals: list[CanonicalInterval] = []
    for position, boundary in enumerate(ordered_boundaries[:-1]):
        active.difference_update(ends.get(boundary, ()))
        active.update(starts.get(boundary, ()))
        next_boundary = ordered_boundaries[position + 1]
        active_spans = [ordered_spans[index] for index in sorted(active)]
        intervals.append(
            CanonicalInterval(
                start_sample=boundary,
                end_sample=next_boundary,
                active_speakers=tuple(
                    sorted({span.speaker_id for span in active_spans})
                ),
                ambiguous=False,
                speaker_identity_known=all(
                    span.speaker_identity_known for span in active_spans
                ),
                source_annotation_ids=tuple(
                    sorted({span.source_annotation_id for span in active_spans})
                ),
            )
        )
    active.difference_update(ends.get(scored_end_sample, ()))
    if active:
        raise AnnotationNormalizationError("annotation sweep did not close at scored end")
    return normalize_intervals(
        intervals,
        scored_start_sample=scored_start_sample,
        scored_end_sample=scored_end_sample,
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError) as exc:
        raise AnnotationNormalizationError(f"invalid JSONL manifest: {path}") from exc
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise AnnotationNormalizationError(f"JSONL manifest must contain objects: {path}")
    return rows


def _manifest_file_paths(
    corpus_root: Path, annotation_row: Mapping[str, Any]
) -> list[Path]:
    corpus_root = corpus_root.resolve()
    files = annotation_row.get("annotation_files")
    if not isinstance(files, list) or not files:
        raise AnnotationNormalizationError("annotation manifest file list is missing")
    paths = []
    observed_rows = []
    for row in files:
        if not isinstance(row, Mapping):
            raise AnnotationNormalizationError("annotation file identity must be an object")
        ref = row.get("ref")
        if not isinstance(ref, str) or not ref:
            raise AnnotationNormalizationError("annotation file ref is invalid")
        path = (corpus_root / ref).resolve()
        if not path.is_relative_to(corpus_root) or not path.is_file():
            raise AnnotationNormalizationError(f"annotation file is unavailable: {ref}")
        observed = {
            "ref": path.relative_to(corpus_root).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        if dict(row) != observed:
            raise AnnotationNormalizationError(f"annotation file identity mismatch: {ref}")
        paths.append(path)
        observed_rows.append(observed)
    if len(set(paths)) != len(paths):
        raise AnnotationNormalizationError("annotation file refs must be unique")
    if canonical_sha256(observed_rows) != annotation_row.get("annotation_sha256"):
        raise AnnotationNormalizationError("annotation bundle identity mismatch")
    return paths


def normalize_source(
    source_row: Mapping[str, Any],
    annotation_row: Mapping[str, Any],
    corpus_root: Path,
) -> NormalizedSession:
    contract = load_contract()
    source_id = source_row.get("source_id")
    if not isinstance(source_id, str) or source_id != annotation_row.get("source_id"):
        raise AnnotationNormalizationError("source and annotation identities do not match")
    if (
        source_row.get("contract_version") != contract.contract_version
        or annotation_row.get("contract_version") != contract.contract_version
        or source_row.get("contract_document_sha256") != contract.document_sha256
        or annotation_row.get("contract_document_sha256") != contract.document_sha256
    ):
        raise AnnotationNormalizationError("manifest contract identity mismatch")
    corpus = source_row.get("corpus")
    session_id = source_row.get("session_id")
    if (
        not isinstance(corpus, str)
        or corpus != annotation_row.get("corpus")
        or not isinstance(session_id, str)
        or session_id != annotation_row.get("session_id")
    ):
        raise AnnotationNormalizationError("source and annotation metadata do not match")
    source_prefix = {"AMI": "ami", "AliMeeting": "alimeeting"}.get(corpus)
    if (
        source_prefix is None
        or SOURCE_ID_PATTERN.fullmatch(source_id) is None
        or source_id != f"{source_prefix}_{session_id}"
    ):
        raise AnnotationNormalizationError("invalid canonical source identity")
    scored_start_sample = annotation_row.get("coverage_start_sample")
    scored_end_sample = annotation_row.get("coverage_end_sample")
    duration_samples = source_row.get("duration_samples")
    if (
        isinstance(scored_start_sample, bool)
        or not isinstance(scored_start_sample, int)
        or isinstance(scored_end_sample, bool)
        or not isinstance(scored_end_sample, int)
        or isinstance(duration_samples, bool)
        or not isinstance(duration_samples, int)
        or scored_start_sample != 0
        or scored_end_sample <= scored_start_sample
        or scored_end_sample > duration_samples
        or source_row.get("sample_rate_hz") != SAMPLE_RATE_HZ
    ):
        raise AnnotationNormalizationError("invalid manifest coverage range")
    if source_row.get("annotation_sha256") != annotation_row.get("annotation_sha256"):
        raise AnnotationNormalizationError("source and annotation bundle hashes differ")
    paths = _manifest_file_paths(corpus_root, annotation_row)
    if corpus == "AMI":
        meetings = [path for path in paths if path.name == "meetings.xml"]
        segments = [path for path in paths if AMI_SEGMENT_NAME.fullmatch(path.name)]
        if (
            scored_end_sample != duration_samples
            or source_row.get("annotation_ref")
            != f"ami/annotations/segments/{session_id}.*.segments.xml"
            or len(meetings) != 1
            or not segments
            or len(meetings) + len(segments) != len(paths)
        ):
            raise AnnotationNormalizationError("AMI annotation bundle structure mismatch")
        parsed = parse_ami_annotations(
            session_id,
            meetings[0],
            segments,
            scored_start_sample=scored_start_sample,
            scored_end_sample=scored_end_sample,
        )
    elif corpus == "AliMeeting":
        textgrids = [path for path in paths if path.suffix == ".TextGrid"]
        actual_ref = (
            textgrids[0].resolve().relative_to(corpus_root.resolve()).as_posix()
            if len(textgrids) == 1
            else None
        )
        if (
            source_row.get("annotation_coverage_start_sample")
            != scored_start_sample
            or source_row.get("annotation_coverage_end_sample") != scored_end_sample
            or len(textgrids) != 1
            or len(paths) != 1
            or textgrids[0].stem != session_id
            or source_row.get("annotation_ref") != actual_ref
        ):
            raise AnnotationNormalizationError(
                "AliMeeting annotation bundle structure mismatch"
            )
        parsed = parse_alimeeting_annotations(
            source_id,
            textgrids[0],
            scored_start_sample=scored_start_sample,
            scored_end_sample=scored_end_sample,
        )
    else:
        raise AnnotationNormalizationError(f"unsupported corpus: {corpus}")
    intervals = spans_to_canonical_intervals(
        parsed.spans,
        scored_start_sample=scored_start_sample,
        scored_end_sample=scored_end_sample,
    )
    labels = generate_labels(
        intervals,
        scored_start_sample=scored_start_sample,
        scored_end_sample=scored_end_sample,
    )
    waveform_sha256 = source_row.get("waveform_sha256")
    annotation_sha256 = annotation_row.get("annotation_sha256")
    if (
        not isinstance(waveform_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", waveform_sha256)
        or not isinstance(annotation_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", annotation_sha256)
    ):
        raise AnnotationNormalizationError("invalid source or annotation hash")
    return NormalizedSession(
        source_id=source_id,
        corpus=corpus,
        session_id=session_id,
        scored_start_sample=scored_start_sample,
        scored_end_sample=scored_end_sample,
        source_waveform_sha256=waveform_sha256,
        annotation_sha256=annotation_sha256,
        raw_speech_span_count=parsed.raw_speech_span_count,
        clipped_span_count=parsed.clipped_span_count,
        intervals=intervals,
        labels=labels,
    )


def normalize_inventory(data_dir: Path, corpus_root: Path) -> list[NormalizedSession]:
    source_rows = _load_jsonl(data_dir / "source_manifest.jsonl")
    annotation_rows = _load_jsonl(data_dir / "annotation_manifest.jsonl")
    sources = {row.get("source_id"): row for row in source_rows}
    annotations = {row.get("source_id"): row for row in annotation_rows}
    if (
        len(sources) != len(source_rows)
        or len(annotations) != len(annotation_rows)
        or set(sources) != set(annotations)
    ):
        raise AnnotationNormalizationError("source and annotation inventories differ")
    return [
        normalize_source(sources[source_id], annotations[source_id], corpus_root)
        for source_id in sorted(sources)
    ]


def write_normalization_manifest(
    data_dir: Path, corpus_root: Path, output_path: Path
) -> None:
    sessions = normalize_inventory(data_dir, corpus_root)
    write_jsonl(output_path, (session.manifest_row() for session in sessions))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_normalization_manifest(
        args.data_dir.resolve(), args.corpus_root.resolve(), args.output.resolve()
    )


if __name__ == "__main__":
    main()
