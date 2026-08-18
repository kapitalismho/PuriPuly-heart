from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import wave
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from experiments.psem_training_strategy_gate.data.label_contract import load_contract

HISTORICAL_CONFIGS = {
    "r6": "experiments/online_speaker_memory_handoff/config.json",
    "r7": "experiments/speaker_representation_scd/configs/r7/eres_candidate_relation_verifier.json",
    "r7b": "experiments/speaker_representation_scd/configs/r7b/fixed_lag_local_segmentation.json",
    "r8": "experiments/speaker_representation_scd/configs/r8/streaming_sortformer_feasibility.json",
    "r9": "experiments/speaker_representation_scd/configs/r9/sortformer_change_verification_upper_bound.json",
    "issue_72": "experiments/psem_trainable_formulation_gate/config.json",
}
HISTORICAL_ID_FIELDS = {
    "r6": ("development_sessions", "evaluation_sessions"),
    "r7": ("development_sessions", "evaluation_sessions"),
    "r7b": ("folds",),
    "r8": ("folds",),
    "r9": ("folds",),
    "issue_72": ("folds",),
}
REQUIRED_PRIOR_SOURCE_IDS = frozenset(
    {
        "alimeeting_R8001_M8004",
        "alimeeting_R8008_M8013",
        "alimeeting_R8009_M8019",
        "alimeeting_R8007_M8010",
        "ami_IS1009a",
        "ami_EN2001d",
        "ami_TS3006a",
        "ami_ES2003a",
        "ami_TS3009b",
        "ami_ES2015d",
    }
)
AMI_AUDIO_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror//amicorpus/{meeting}/audio/{meeting}.Mix-Headset.wav"
AMI_ANNOTATION_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
ALIMEETING_SOURCE_URL = "https://www.openslr.org/119/"
ALIMEETING_ARCHIVE_URL = "https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Eval_Ali.tar.gz"
ALIMEETING_ARCHIVE_SIZE_BYTES = 3673718355
SOURCE_ID_PATTERN = re.compile(r"^(ami|alimeeting)_[A-Za-z0-9_]+$")
ALIMEETING_TIER_PATTERN = re.compile(r'^\s*name\s*=\s*"N_(SPK\d+)"\s*$')
EXPECTED_AMI_MEETINGS = frozenset(
    {
        "EN2001d", "EN2002c", "EN2006a", "EN2009d", "ES2002b",
        "ES2003a", "ES2004a", "ES2014a", "ES2015d", "ES2016a",
        "IS1008a", "IS1009a", "TS3003b", "TS3004a", "TS3005b",
        "TS3006a", "TS3007a", "TS3008b", "TS3009b", "TS3012c",
    }
)
EXPECTED_ALIMEETING_MEETINGS = frozenset(
    {
        "R8001_M8004", "R8003_M8001", "R8007_M8010", "R8007_M8011",
        "R8008_M8013", "R8009_M8018", "R8009_M8019", "R8009_M8020",
    }
)


class ProvenanceError(RuntimeError):
    pass


def sha256_file(path: Path, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_strings(item)


def _historical_identities(prior_use: str, raw: Any) -> list[str]:
    if not isinstance(raw, dict):
        raise ProvenanceError(f"historical config must be an object: {prior_use}")
    identities = []
    for field in HISTORICAL_ID_FIELDS[prior_use]:
        value = raw.get(field)
        if not isinstance(value, list) or not value:
            raise ProvenanceError(f"historical config field must be a non-empty list: {prior_use}.{field}")
        items = list(_walk_strings(value))
        if not items or any(not SOURCE_ID_PATTERN.fullmatch(item) for item in items):
            raise ProvenanceError(f"invalid historical source identities: {prior_use}.{field}")
        identities.extend(items)
    return sorted(set(identities))


def collect_prior_exposure(repo_root: Path) -> dict[str, dict[str, Any]]:
    prior_uses: dict[str, set[str]] = defaultdict(set)
    evidence: dict[str, list[dict[str, str]]] = defaultdict(list)
    for prior_use, relative_path in HISTORICAL_CONFIGS.items():
        path = repo_root / relative_path
        if not path.is_file():
            raise ProvenanceError(f"missing historical config: {relative_path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        identities = _historical_identities(prior_use, raw)
        config_sha256 = sha256_file(path)
        for source_id in identities:
            prior_uses[source_id].add(prior_use)
            evidence[source_id].append(
                {
                    "prior_use": prior_use,
                    "ref": relative_path.replace("\\", "/"),
                    "sha256": config_sha256,
                }
            )
    if not REQUIRED_PRIOR_SOURCE_IDS.issubset(prior_uses):
        missing = sorted(REQUIRED_PRIOR_SOURCE_IDS - prior_uses.keys())
        raise ProvenanceError(f"mandatory prior-exposure identities missing: {missing}")
    return {
        source_id: {
            "prior_uses": sorted(uses),
            "evidence": sorted(evidence[source_id], key=lambda row: row["prior_use"]),
        }
        for source_id, uses in sorted(prior_uses.items())
    }


def _wav_identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ProvenanceError(f"missing source waveform: {path}")
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width_bytes = handle.getsampwidth()
        sample_rate_hz = handle.getframerate()
        duration_samples = handle.getnframes()
        compression = handle.getcomptype()
        payload_size_bytes = 0
        while payload := handle.readframes(1 << 19):
            payload_size_bytes += len(payload)
    if channels != 1 or sample_width_bytes != 2 or sample_rate_hz != 16000 or compression != "NONE":
        raise ProvenanceError(f"source waveform is not 16 kHz mono PCM16: {path}")
    if (
        duration_samples <= 0
        or payload_size_bytes != duration_samples * channels * sample_width_bytes
    ):
        raise ProvenanceError(f"source waveform payload is truncated or empty: {path}")
    return {
        "waveform_sha256": sha256_file(path),
        "waveform_size_bytes": path.stat().st_size,
        "sample_rate_hz": sample_rate_hz,
        "channels": channels,
        "sample_width_bytes": sample_width_bytes,
        "duration_samples": duration_samples,
    }


def _file_rows(paths: Iterable[Path], root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(paths):
        if not path.is_file():
            raise ProvenanceError(f"missing annotation source: {path}")
        rows.append(
            {
                "ref": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not rows:
        raise ProvenanceError("annotation bundle is empty")
    return rows


def _ami_metadata(annotations_root: Path) -> dict[str, dict[str, Any]]:
    meetings_path = annotations_root / "corpusResources" / "meetings.xml"
    if not meetings_path.is_file():
        raise ProvenanceError(f"missing AMI meetings metadata: {meetings_path}")
    meetings: dict[str, dict[str, Any]] = {}
    for element in ET.parse(meetings_path).getroot().iter("meeting"):
        meeting_id = element.get("observation")
        if not meeting_id:
            continue
        if meeting_id in meetings:
            raise ProvenanceError(f"duplicate AMI meeting metadata: {meeting_id}")
        speakers = []
        unknown_agents = []
        speaker_agents = []
        speaker_count = 0
        for speaker in element.iter("speaker"):
            speaker_count += 1
            agent = speaker.get("nxt_agent")
            global_name = speaker.get("global_name")
            if agent:
                speaker_agents.append(agent)
            if global_name:
                speakers.append(global_name)
            else:
                unknown_agents.append(agent or "unknown")
        if speaker_count == 0:
            unknown_agents.append("unknown")
        meetings[meeting_id] = {
            "speaker_ids": sorted(set(speakers)),
            "unknown_speaker_agents": sorted(set(unknown_agents)),
            "unknown_speaker_count": len(unknown_agents),
            "speaker_agents": sorted(set(speaker_agents)),
            "meeting_type": element.get("type", "unknown"),
        }
    return meetings


def _validate_ami_segments(
    paths: list[Path], meeting_id: str, expected_agents: list[str], duration_samples: int
) -> None:
    if not paths:
        raise ProvenanceError(f"AMI segment annotations missing for {meeting_id}")
    found_agents = set()
    segment_counts: dict[str, int] = {}
    total_segments = 0
    for path in paths:
        parts = path.name.split(".")
        if len(parts) != 4 or parts[0] != meeting_id or parts[2:] != ["segments", "xml"]:
            raise ProvenanceError(f"unexpected AMI segment filename: {path.name}")
        agent = parts[1]
        found_agents.add(agent)
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError as exc:
            raise ProvenanceError(f"invalid AMI segment XML: {path}") from exc
        segments = list(root.iter("segment"))
        segment_counts[agent] = len(segments)
        total_segments += len(segments)
        for segment in segments:
            try:
                start = float(segment.attrib["transcriber_start"])
                end = float(segment.attrib["transcriber_end"])
            except (KeyError, ValueError) as exc:
                raise ProvenanceError(f"invalid AMI segment bounds: {path}") from exc
            if not math.isfinite(start) or not math.isfinite(end):
                raise ProvenanceError(f"invalid AMI segment bounds: {path}")
            start_sample = round(start * 16000)
            end_sample = round(end * 16000)
            if (
                start_sample < 0
                or start_sample >= duration_samples
                or end_sample <= start_sample
                or end_sample > duration_samples + 32000
            ):
                raise ProvenanceError(f"AMI segment bounds exceed the source timeline: {path}")
    expected = set(expected_agents)
    if expected and not expected.issubset(found_agents):
        raise ProvenanceError(f"AMI segment speaker files do not match metadata: {meeting_id}")
    if any(segment_counts[agent] for agent in found_agents - expected):
        raise ProvenanceError(f"AMI segment speaker has no metadata identity: {meeting_id}")
    if total_segments == 0:
        raise ProvenanceError(f"AMI segment annotation bundle is empty: {meeting_id}")


def _prior_fields(source_id: str, prior: dict[str, dict[str, Any]]) -> dict[str, Any]:
    exposed = source_id in prior
    return {
        "selection_exposed": exposed,
        "prior_uses": prior[source_id]["prior_uses"] if exposed else [],
        "eval_eligible": False if exposed else None,
        "eval_eligibility_reason": (
            "forbidden_prior_selection_exposure"
            if exposed
            else "pending_identity_component_and_pretraining_overlap_audit"
        ),
    }


def _ami_rows(
    corpus_root: Path,
    prior: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    audio_root = corpus_root / "ami" / "audio"
    annotations_root = corpus_root / "ami" / "annotations"
    meetings_meta = _ami_metadata(annotations_root)
    meetings_file = annotations_root / "corpusResources" / "meetings.xml"
    source_rows = []
    annotation_rows = []
    discovered = {path.name for path in audio_root.iterdir() if path.is_dir()}
    if discovered != EXPECTED_AMI_MEETINGS:
        raise ProvenanceError(
            f"AMI inventory mismatch; missing={sorted(EXPECTED_AMI_MEETINGS - discovered)} "
            f"extra={sorted(discovered - EXPECTED_AMI_MEETINGS)}"
        )
    for meeting_dir in sorted(audio_root.iterdir()):
        if not meeting_dir.is_dir():
            continue
        meeting_id = meeting_dir.name
        source_id = f"ami_{meeting_id}"
        wav_path = meeting_dir / f"{meeting_id}.Mix-Headset.wav"
        audio = _wav_identity(wav_path)
        segment_paths = sorted((annotations_root / "segments").glob(f"{meeting_id}.*.segments.xml"))
        metadata = meetings_meta.get(meeting_id)
        if metadata is None:
            raise ProvenanceError(f"AMI meeting metadata missing for {meeting_id}")
        _validate_ami_segments(
            segment_paths,
            meeting_id,
            metadata["speaker_agents"],
            audio["duration_samples"],
        )
        annotation_files = _file_rows([meetings_file, *segment_paths], corpus_root)
        annotation_sha256 = canonical_sha256(annotation_files)
        prior_fields = _prior_fields(source_id, prior)
        source_rows.append(
            {
                "schema_version": 1,
                "source_id": source_id,
                "corpus": "AMI",
                "session_id": meeting_id,
                "meeting_series": meeting_id[:-1],
                "meeting_type": metadata["meeting_type"],
                "corpus_version": "AMI Meeting Corpus public Mix-Headset",
                "speaker_ids": metadata["speaker_ids"],
                "unknown_speaker_agents": metadata["unknown_speaker_agents"],
                "unknown_speaker_count": metadata["unknown_speaker_count"],
                "speaker_identity_status": (
                    "known"
                    if not metadata["unknown_speaker_agents"]
                    else "partially_or_fully_unknown"
                ),
                "audio_ref": wav_path.relative_to(corpus_root).as_posix(),
                "audio_source_url": AMI_AUDIO_URL.format(meeting=meeting_id),
                "recording_recipe": "Mix-Headset original 16 kHz mono PCM16",
                "license_id": "CC-BY-4.0",
                "use_authorization": "public_research_under_source_license",
                "redistribution_status": "source_license_governs_raw_audio_not_committed",
                "annotation_ref": f"ami/annotations/segments/{meeting_id}.*.segments.xml",
                "annotation_sha256": annotation_sha256,
                **audio,
                **prior_fields,
            }
        )
        annotation_rows.append(
            {
                "schema_version": 1,
                "source_id": source_id,
                "corpus": "AMI",
                "session_id": meeting_id,
                "annotation_format": "AMI NXT manual segments XML",
                "annotation_version": "ami_public_manual_1.6.2",
                "annotation_source_url": AMI_ANNOTATION_URL,
                "annotation_files": annotation_files,
                "annotation_sha256": annotation_sha256,
                "coverage_start_sample": 0,
                "coverage_end_sample": audio["duration_samples"],
                "coverage_status": "full_source_timeline_under_explicit_absence_is_silence_rule",
                "speaker_identity_source": "corpusResources/meetings.xml global_name",
            }
        )
    return source_rows, annotation_rows


def _textgrid_value(lines: list[str], key: str) -> str:
    prefix = f"{key} ="
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(prefix):
            return stripped.split("=", 1)[1].strip().strip('"')
    raise ProvenanceError(f"TextGrid field missing: {key}")


def _parse_alimeeting_textgrid(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 8 or lines[0].strip() != 'File type = "ooTextFile"' or lines[1].strip() != 'Object class = "TextGrid"':
        raise ProvenanceError(f"invalid AliMeeting TextGrid header: {path}")
    item_matches = [
        (index, match)
        for index, line in enumerate(lines)
        if (match := re.fullmatch(r"\s*item \[(\d+)\]:\s*", line))
    ]
    item_indexes = [index for index, _ in item_matches]
    if not item_indexes:
        raise ProvenanceError(f"AliMeeting TextGrid contains no tiers: {path}")
    header = lines[: item_indexes[0]]
    try:
        timeline_start_s = float(_textgrid_value(header, "xmin"))
        timeline_end_s = float(_textgrid_value(header, "xmax"))
        declared_tier_count = int(_textgrid_value(header, "size"))
    except ValueError as exc:
        raise ProvenanceError(f"invalid AliMeeting TextGrid timeline: {path}") from exc
    item_numbers = [int(match.group(1)) for _, match in item_matches]
    if (
        not math.isfinite(timeline_start_s)
        or not math.isfinite(timeline_end_s)
        or timeline_start_s != 0
        or timeline_end_s <= timeline_start_s
    ):
        raise ProvenanceError(f"invalid AliMeeting TextGrid timeline: {path}")
    if (
        declared_tier_count <= 0
        or declared_tier_count != len(item_indexes)
        or item_numbers != list(range(1, declared_tier_count + 1))
    ):
        raise ProvenanceError(f"AliMeeting TextGrid tier count mismatch: {path}")
    speakers = []
    unknown_tiers = []
    interval_count = 0
    for position, start in enumerate(item_indexes):
        end = item_indexes[position + 1] if position + 1 < len(item_indexes) else len(lines)
        block = lines[start:end]
        if _textgrid_value(block, "class") != "IntervalTier":
            raise ProvenanceError(f"unsupported AliMeeting TextGrid tier class: {path}")
        tier_name = _textgrid_value(block, "name")
        match = re.fullmatch(r"N_(SPK\d+)", tier_name)
        if match:
            speakers.append(match.group(1))
        else:
            unknown_tiers.append(tier_name or "unknown")
        declared_line = next(
            (line.strip() for line in block if line.strip().startswith("intervals: size =")),
            None,
        )
        if declared_line is None:
            raise ProvenanceError(f"AliMeeting TextGrid tier lacks interval count: {path}")
        try:
            declared_count = int(declared_line.split("=", 1)[1].strip())
        except ValueError as exc:
            raise ProvenanceError(f"invalid AliMeeting TextGrid interval count: {path}") from exc
        interval_matches = [
            (index, match)
            for index, line in enumerate(block)
            if (match := re.fullmatch(r"\s*intervals \[(\d+)\]:\s*", line))
        ]
        interval_indexes = [index for index, _ in interval_matches]
        interval_numbers = [int(match.group(1)) for _, match in interval_matches]
        if (
            declared_count != len(interval_indexes)
            or declared_count <= 0
            or interval_numbers != list(range(1, declared_count + 1))
        ):
            raise ProvenanceError(f"AliMeeting TextGrid interval count mismatch: {path}")
        for interval_position, interval_start in enumerate(interval_indexes):
            interval_end = (
                interval_indexes[interval_position + 1]
                if interval_position + 1 < len(interval_indexes)
                else len(block)
            )
            interval = block[interval_start:interval_end]
            try:
                start_s = float(_textgrid_value(interval, "xmin"))
                end_s = float(_textgrid_value(interval, "xmax"))
                _textgrid_value(interval, "text")
            except ValueError as exc:
                raise ProvenanceError(f"invalid AliMeeting TextGrid interval bounds: {path}") from exc
            if (
                not math.isfinite(start_s)
                or not math.isfinite(end_s)
                or start_s < timeline_start_s
                or end_s <= start_s
                or end_s > timeline_end_s
            ):
                raise ProvenanceError(f"AliMeeting TextGrid interval exceeds tier bounds: {path}")
        interval_count += declared_count
    if not speakers and not unknown_tiers:
        raise ProvenanceError(f"no AliMeeting speaker tiers found: {path}")
    return {
        "speaker_ids": sorted(set(speakers)),
        "unknown_speaker_tiers": sorted(set(unknown_tiers)),
        "interval_count": interval_count,
        "timeline_start_sample": round(timeline_start_s * 16000),
        "timeline_end_sample": round(timeline_end_s * 16000),
    }


def _alimeeting_rows(
    corpus_root: Path,
    prior: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    audio_root = corpus_root / "alimeeting" / "far_ch0"
    textgrid_root = corpus_root / "alimeeting" / "Eval_Ali" / "Eval_Ali_far" / "textgrid_dir"
    source_rows = []
    annotation_rows = []
    discovered = {path.stem for path in audio_root.glob("R*_M*.wav")}
    if discovered != EXPECTED_ALIMEETING_MEETINGS:
        raise ProvenanceError(
            f"AliMeeting inventory mismatch; missing={sorted(EXPECTED_ALIMEETING_MEETINGS - discovered)} "
            f"extra={sorted(discovered - EXPECTED_ALIMEETING_MEETINGS)}"
        )
    for wav_path in sorted(audio_root.glob("R*_M*.wav")):
        meeting_id = wav_path.stem
        source_id = f"alimeeting_{meeting_id}"
        textgrid_path = textgrid_root / f"{meeting_id}.TextGrid"
        audio = _wav_identity(wav_path)
        annotation_files = _file_rows([textgrid_path], corpus_root)
        annotation_sha256 = canonical_sha256(annotation_files)
        textgrid = _parse_alimeeting_textgrid(textgrid_path)
        if textgrid["timeline_start_sample"] != 0 or textgrid["timeline_end_sample"] > audio["duration_samples"]:
            raise ProvenanceError(f"AliMeeting TextGrid bounds exceed source waveform: {meeting_id}")
        prior_fields = _prior_fields(source_id, prior)
        source_rows.append(
            {
                "schema_version": 1,
                "source_id": source_id,
                "corpus": "AliMeeting",
                "session_id": meeting_id,
                "meeting_series": None,
                "meeting_type": "natural_meeting_eval_partition",
                "corpus_version": "M2MeT Eval_Ali",
                "speaker_ids": textgrid["speaker_ids"],
                "unknown_speaker_agents": textgrid["unknown_speaker_tiers"],
                "unknown_speaker_count": len(textgrid["unknown_speaker_tiers"]),
                "speaker_identity_status": (
                    "known_corpus_speaker_ids"
                    if not textgrid["unknown_speaker_tiers"]
                    else "partially_or_fully_unknown"
                ),
                "audio_ref": wav_path.relative_to(corpus_root).as_posix(),
                "audio_source_url": ALIMEETING_SOURCE_URL,
                "source_archive_url": ALIMEETING_ARCHIVE_URL,
                "source_archive_size_bytes": ALIMEETING_ARCHIVE_SIZE_BYTES,
                "recording_recipe": "far-field array channel 0 materialized as 16 kHz mono PCM16",
                "license_id": "CC-BY-SA-4.0",
                "use_authorization": "public_research_under_source_license",
                "redistribution_status": "source_license_governs_raw_audio_not_committed",
                "annotation_ref": textgrid_path.relative_to(corpus_root).as_posix(),
                "annotation_sha256": annotation_sha256,
                "annotation_coverage_start_sample": textgrid["timeline_start_sample"],
                "annotation_coverage_end_sample": textgrid["timeline_end_sample"],
                **audio,
                **prior_fields,
            }
        )
        annotation_rows.append(
            {
                "schema_version": 1,
                "source_id": source_id,
                "corpus": "AliMeeting",
                "session_id": meeting_id,
                "annotation_format": "Praat TextGrid participant IntervalTiers",
                "annotation_version": "M2MeT Eval_Ali",
                "annotation_source_url": ALIMEETING_SOURCE_URL,
                "annotation_files": annotation_files,
                "annotation_sha256": annotation_sha256,
                "coverage_start_sample": 0,
                "coverage_end_sample": textgrid["timeline_end_sample"],
                "coverage_status": "textgrid_bounded_timeline_trailing_audio_unscored",
                "speaker_identity_source": "N_SPKxxxx tier names",
                "interval_count": textgrid["interval_count"],
            }
        )
    return source_rows, annotation_rows


def build_provenance(
    repo_root: Path,
    corpus_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    prior = collect_prior_exposure(repo_root)
    ami_sources, ami_annotations = _ami_rows(corpus_root, prior)
    ali_sources, ali_annotations = _alimeeting_rows(corpus_root, prior)
    source_rows = sorted([*ami_sources, *ali_sources], key=lambda row: row["source_id"])
    annotation_rows = sorted(
        [*ami_annotations, *ali_annotations], key=lambda row: row["source_id"]
    )
    sources_by_id = {row["source_id"]: row for row in source_rows}
    if not REQUIRED_PRIOR_SOURCE_IDS.issubset(sources_by_id):
        missing = sorted(REQUIRED_PRIOR_SOURCE_IDS - sources_by_id.keys())
        raise ProvenanceError(f"prior-exposed source material is unavailable: {missing}")
    prior_rows = []
    for source_id, facts in sorted(prior.items()):
        source = sources_by_id.get(source_id)
        if source is None:
            raise ProvenanceError(f"historical source is missing from candidate inventory: {source_id}")
        prior_rows.append(
            {
                "schema_version": 1,
                "source_id": source_id,
                "corpus": source["corpus"],
                "session_id": source["session_id"],
                "meeting_series": source["meeting_series"],
                "speaker_ids": source["speaker_ids"],
                "waveform_sha256": source["waveform_sha256"],
                "annotation_sha256": source["annotation_sha256"],
                "prior_uses": facts["prior_uses"],
                "selection_exposed": True,
                "eval_eligible": False,
                "reason": "prior experimental selection exposure",
                "evidence": facts["evidence"],
            }
        )
    if {row["source_id"] for row in prior_rows} != REQUIRED_PRIOR_SOURCE_IDS:
        unexpected = sorted({row["source_id"] for row in prior_rows} - REQUIRED_PRIOR_SOURCE_IDS)
        raise ProvenanceError(f"unexpected reconstructed historical sources: {unexpected}")
    contract = load_contract()
    for row in [*source_rows, *annotation_rows, *prior_rows]:
        row["contract_version"] = contract.contract_version
        row["contract_document_sha256"] = contract.document_sha256
    return source_rows, annotation_rows, prior_rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    path.write_text(payload, encoding="utf-8", newline="\n")


def write_provenance(repo_root: Path, corpus_root: Path, output_dir: Path) -> None:
    source_rows, annotation_rows, prior_rows = build_provenance(repo_root, corpus_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "source_manifest.jsonl", source_rows)
    write_jsonl(output_dir / "annotation_manifest.jsonl", annotation_rows)
    write_jsonl(output_dir / "prior_exposure_manifest.jsonl", prior_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    write_provenance(args.repo_root.resolve(), args.corpus_root.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
