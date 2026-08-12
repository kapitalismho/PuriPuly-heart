from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = EXPERIMENT_ROOT / "config.json"
SAMPLE_RATE = 16000


class R6Error(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SpeakerRegion:
    audio_epoch: int
    start_sample: int
    end_sample: int
    speakers: frozenset[str]
    ambiguous: bool = False


@dataclass(frozen=True, slots=True)
class SpeechInterval:
    speaker: str
    start_time_s: float
    end_time_s: float
    ambiguous: bool = False


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def cache_root() -> Path:
    value = os.environ.get("SRSCD_CACHE_ROOT")
    if not value:
        raise R6Error("SRSCD_CACHE_ROOT is required")
    root = Path(value).resolve()
    if not root.is_absolute() or REPOSITORY_ROOT == root or REPOSITORY_ROOT in root.parents:
        raise R6Error("SRSCD_CACHE_ROOT must be an absolute path outside the repository")
    return root


def output_root(config: dict[str, Any], root: Path) -> Path:
    return root / str(config["output_relative_path"])


def input_paths(root: Path) -> dict[str, Path]:
    return {
        "source_metadata.jsonl": root / "data/r2/legacy_common_gt/source_metadata.jsonl",
        "waveform_inventory.jsonl": root / "data/r2/legacy_common_gt/waveform_inventory.jsonl",
        "anchor_index.jsonl": root / "data/r3/legacy_common_gt/anchor_index.jsonl",
        "reduced_r3_r4_forecast.json": root
        / "manifests/r2/legacy_common_gt/reduced_r3_r4_forecast.json",
        "promotion_ledger.json": root / "manifests/r3/legacy_common_gt/promotion_ledger.json",
    }


def validate_inputs(config: dict[str, Any], root: Path) -> dict[str, Path]:
    paths = input_paths(root)
    for name, expected in config["input_identities"].items():
        path = paths[name]
        if not path.is_file():
            raise R6Error(f"required input is missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise R6Error(f"input identity mismatch for {name}: {actual}")
    return paths


def _corpus_root(root: Path) -> Path:
    receipt = load_json(root / "manifests/r2/legacy_common_gt/validation_receipt.json")
    path = Path(str(receipt["corpus_root"])).resolve()
    if not path.is_dir():
        raise R6Error(f"corpus root is unavailable: {path}")
    return path


def _source_rows(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(paths["source_metadata.jsonl"])
    return {str(row["session_id"]): row for row in rows}


def _waveform_paths(paths: dict[str, Path], corpus: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for row in read_jsonl(paths["waveform_inventory.jsonl"]):
        result[str(row["waveform_id"])] = corpus / Path(
            str(row["artifact_relative_path"]).replace("\\", "/")
        )
    return result


def _parse_textgrid(path: Path) -> list[tuple[str, list[tuple[float, float, str]]]]:
    tiers: list[tuple[str, list[tuple[float, float, str]]]] = []
    name: str | None = None
    intervals: list[tuple[float, float, str]] = []
    start: float | None = None
    end: float | None = None

    def flush() -> None:
        nonlocal name, intervals
        if name is not None:
            tiers.append((name, intervals))
        name = None
        intervals = []

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("item ["):
            flush()
        elif line.startswith("name ="):
            value = line.split("=", 1)[1].strip()
            name = value[1:-1] if value.startswith('"') and value.endswith('"') else value
        elif line.startswith("xmin ="):
            start = float(line.split("=", 1)[1].strip())
        elif line.startswith("xmax ="):
            end = float(line.split("=", 1)[1].strip())
        elif line.startswith("text =") and name is not None:
            value = line.split("=", 1)[1].strip()
            text = value[1:-1] if value.startswith('"') and value.endswith('"') else value
            if start is not None and end is not None:
                intervals.append((start, end, text))
            start = None
            end = None
    flush()
    return tiers


def _intervals_to_regions(
    intervals: Sequence[SpeechInterval], duration: int
) -> list[SpeakerRegion]:
    spans: list[tuple[int, int, str, bool]] = []
    boundaries = {0, duration}
    for interval in intervals:
        start = max(0, int(round(interval.start_time_s * SAMPLE_RATE)))
        end = min(duration, int(round(interval.end_time_s * SAMPLE_RATE)))
        if end <= start:
            continue
        spans.append((start, end, interval.speaker, interval.ambiguous))
        boundaries.update((start, end))
    ordered = sorted(boundaries)
    regions: list[SpeakerRegion] = []
    for left, right in zip(ordered[:-1], ordered[1:], strict=True):
        speakers: set[str] = set()
        ambiguous = False
        for start, end, speaker, span_ambiguous in spans:
            if end <= left or start >= right:
                continue
            speakers.add(speaker)
            ambiguous = ambiguous or span_ambiguous
        value = SpeakerRegion(0, left, right, frozenset(speakers), ambiguous)
        if (
            regions
            and regions[-1].speakers == value.speakers
            and regions[-1].ambiguous == value.ambiguous
        ):
            previous = regions[-1]
            regions[-1] = SpeakerRegion(
                0, previous.start_sample, right, previous.speakers, previous.ambiguous
            )
        else:
            regions.append(value)
    return regions


def _ami_intervals(meeting_id: str, annotations: Path) -> list[SpeechInterval]:
    rows: list[SpeechInterval] = []
    for path in sorted((annotations / "words").glob(f"{meeting_id}.*.words.xml")):
        parts = path.name.split(".")
        if len(parts) < 4:
            raise R6Error(f"unexpected AMI word annotation: {path.name}")
        speaker = f"{meeting_id}.Participant{parts[1]}"
        tree = ET.parse(path)
        for element in tree.getroot().iter():
            if element.tag.lower() != "w":
                continue
            start = element.get("starttime")
            end = element.get("endtime")
            if start is None or end is None:
                continue
            text = "".join(element.itertext()).strip()
            rows.append(SpeechInterval(speaker, float(start), float(end), "%" in text))
    rows.sort(key=lambda row: (row.start_time_s, row.end_time_s, row.speaker))
    return rows


def _regions_for_source(source: dict[str, Any], corpus: Path) -> list[SpeakerRegion]:
    session_id = str(source["session_id"])
    duration = int(source["eligible_end_sample"])
    if source["corpus"] == "ami":
        meeting_id = session_id.removeprefix("ami_")
        intervals = _ami_intervals(meeting_id, corpus / "ami" / "annotations")
        if not intervals:
            raise R6Error(f"AMI annotations are unavailable: {meeting_id}")
        return _intervals_to_regions(intervals, duration)
    if source["corpus"] == "alimeeting":
        meeting_id = session_id.removeprefix("alimeeting_")
        textgrid = (
            corpus
            / "alimeeting"
            / "Eval_Ali"
            / "Eval_Ali_far"
            / "textgrid_dir"
            / f"{meeting_id}.TextGrid"
        )
        if not textgrid.is_file():
            raise R6Error(f"AliMeeting annotations are unavailable: {meeting_id}")
        intervals: list[SpeechInterval] = []
        for tier_name, tier_intervals in _parse_textgrid(textgrid):
            speaker = re.sub(r"^N_", "", tier_name)
            for start, end, text in tier_intervals:
                if text.strip():
                    intervals.append(SpeechInterval(speaker, start, end))
        intervals.sort(key=lambda row: (row.start_time_s, row.end_time_s, row.speaker))
        return _intervals_to_regions(intervals, duration)
    raise R6Error(f"R6 natural protocol does not support corpus: {source['corpus']}")


def _overlap_samples(
    regions: list[SpeakerRegion], start: int, end: int
) -> tuple[int, set[str], bool, bool]:
    active = 0
    speakers: set[str] = set()
    overlap = False
    ambiguous = False
    for region in regions:
        if region.end_sample <= start:
            continue
        if region.start_sample >= end:
            break
        amount = min(end, region.end_sample) - max(start, region.start_sample)
        if amount <= 0:
            continue
        if region.ambiguous:
            ambiguous = True
        if region.speakers:
            active += amount
            speakers.update(region.speakers)
            if len(region.speakers) > 1:
                overlap = True
    return active, speakers, overlap, ambiguous


def _eligible_enrollment(
    regions: list[SpeakerRegion], start: int, end: int, minimum_coverage: float
) -> str | None:
    active, speakers, overlap, ambiguous = _overlap_samples(regions, start, end)
    if overlap or ambiguous or len(speakers) != 1:
        return None
    if active < int((end - start) * minimum_coverage):
        return None
    tail_start = max(start, end - 3200)
    tail_active, tail_speakers, tail_overlap, tail_ambiguous = _overlap_samples(
        regions, tail_start, end
    )
    if tail_active == 0 or tail_overlap or tail_ambiguous or tail_speakers != speakers:
        return None
    return next(iter(speakers))


def _first_other_onset(
    regions: list[SpeakerRegion], speaker: str, start: int, end: int
) -> tuple[int, set[str], str] | None:
    for region in regions:
        if region.end_sample <= start:
            continue
        if region.start_sample >= end:
            break
        if region.ambiguous:
            continue
        others = set(region.speakers) - {speaker}
        if others:
            onset = max(start, region.start_sample)
            kind = "overlap" if speaker in region.speakers else "clean"
            return onset, others, kind
    return None


def _event_views(
    regions: list[SpeakerRegion], speaker: str, onset: int, end: int
) -> dict[str, Any]:
    exclusive_new = None
    current_return = None
    other_seen = False
    other_speakers: set[str] = set()
    for region in regions:
        if region.end_sample <= onset:
            continue
        if region.start_sample >= end:
            break
        if region.ambiguous:
            continue
        active = set(region.speakers)
        others = active - {speaker}
        if others:
            other_seen = True
            other_speakers.update(others)
            if speaker not in active and exclusive_new is None:
                exclusive_new = max(onset, region.start_sample)
        elif other_seen and speaker in active and current_return is None:
            current_return = max(onset, region.start_sample)
    return {
        "exclusive_new_onset_sample": exclusive_new,
        "current_returns_sample": current_return,
        "other_speakers": sorted(other_speakers),
    }


def build_units(
    session_id: str,
    regions: list[SpeakerRegion],
    eligible_start: int,
    eligible_end: int,
    enrollment_ms: int,
    policy: dict[str, Any],
) -> list[dict[str, Any]]:
    hop = 1600
    enrollment = enrollment_ms * 16
    maximum_horizon = int(policy["maximum_horizon_ms"]) * 16
    post_event = int(policy["post_event_tail_ms"]) * 16
    minimum_query = int(policy["minimum_query_ms"]) * 16
    coverage = float(policy["minimum_speech_coverage"])
    cursor = eligible_start
    units: list[dict[str, Any]] = []
    sequence = 0
    while cursor + enrollment + minimum_query <= eligible_end:
        frontier = ((cursor + enrollment + hop - 1) // hop) * hop
        speaker = None
        while frontier + minimum_query <= eligible_end:
            speaker = _eligible_enrollment(regions, frontier - enrollment, frontier, coverage)
            if speaker is not None:
                break
            frontier += hop
        if speaker is None:
            break
        horizon_end = min(eligible_end, frontier + maximum_horizon)
        event = _first_other_onset(regions, speaker, frontier, horizon_end)
        if event is None:
            stream_end = horizon_end
            event_sample = None
            event_kind = "negative"
            event_speakers: list[str] = []
            views = {
                "exclusive_new_onset_sample": None,
                "current_returns_sample": None,
                "other_speakers": [],
            }
        else:
            event_sample, event_speaker_set, event_kind = event
            stream_end = min(eligible_end, event_sample + post_event)
            event_speakers = sorted(event_speaker_set)
            views = _event_views(regions, speaker, event_sample, stream_end)
        if stream_end - frontier < minimum_query:
            cursor = frontier + hop
            continue
        unit_id = f"{session_id}|e{enrollment_ms}|{sequence:04d}"
        units.append(
            {
                "unit_id": unit_id,
                "session_id": session_id,
                "enrollment_ms": enrollment_ms,
                "current_speaker": speaker,
                "enrollment_start_sample": frontier - enrollment,
                "enrollment_end_sample": frontier,
                "stream_start_sample": frontier,
                "stream_end_sample": stream_end,
                "new_speaker_onset_sample": event_sample,
                "event_kind": event_kind,
                "event_speakers": event_speakers,
                **views,
            }
        )
        sequence += 1
        cursor = max(frontier + hop, stream_end)
    return units


def _git_state() -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    return {"commit": head, "dirty": bool(status), "dirty_path_count": len(status)}


def prepare(root: Path) -> Path:
    config = load_json(CONFIG_PATH)
    paths = validate_inputs(config, root)
    corpus = _corpus_root(root)
    sources = _source_rows(paths)
    waveforms = _waveform_paths(paths, corpus)
    all_sessions = config["development_sessions"] + config["evaluation_sessions"]
    expected = set(all_sessions)
    if set(config["development_sessions"]) & set(config["evaluation_sessions"]):
        raise R6Error("development and evaluation meetings overlap")
    if not expected <= set(sources):
        raise R6Error(f"configured sessions are missing: {sorted(expected - set(sources))}")
    forecast = load_json(paths["reduced_r3_r4_forecast.json"])
    frozen_evaluation = set(forecast["bounded_variant"]["included_session_ids"])
    anchor_counts: dict[str, int] = defaultdict(int)
    for row in read_jsonl(paths["anchor_index.jsonl"]):
        if row.get("class") == "positive":
            anchor_counts[str(row["session_id"])] += 1
    if not set(config["evaluation_sessions"]) <= frozen_evaluation:
        raise R6Error("evaluation sessions differ from the frozen R4 panel")
    if set(config["development_sessions"]) & frozen_evaluation:
        raise R6Error("a development session occurs in the frozen R4 panel")
    rows: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    for role, session_ids in (
        ("development", config["development_sessions"]),
        ("evaluation", config["evaluation_sessions"]),
    ):
        for session_id in session_ids:
            source = sources[session_id]
            waveform_id = str(source["waveform_id"])
            waveform = waveforms.get(waveform_id)
            if waveform is None or not waveform.is_file():
                raise R6Error(f"waveform is unavailable for {session_id}")
            regions = _regions_for_source(source, corpus)
            if not regions or regions[0].start_sample != 0:
                raise R6Error(f"speaker regions are incomplete for {session_id}")
            session_units: list[dict[str, Any]] = []
            for enrollment_ms in (1500, 2000):
                generated = build_units(
                    session_id,
                    regions,
                    int(source["eligible_start_sample"]),
                    int(source["eligible_end_sample"]),
                    enrollment_ms,
                    config["unit_policy"],
                )
                for row in generated:
                    row["role"] = role
                    row["corpus"] = source["corpus"]
                session_units.extend(generated)
            rows.extend(session_units)
            positives = sum(row["new_speaker_onset_sample"] is not None for row in session_units)
            inventory.append(
                {
                    "session_id": session_id,
                    "role": role,
                    "corpus": source["corpus"],
                    "language": source["language"],
                    "eligible_start_sample": int(source["eligible_start_sample"]),
                    "eligible_end_sample": int(source["eligible_end_sample"]),
                    "eligible_hours": round(
                        (int(source["eligible_end_sample"]) - int(source["eligible_start_sample"]))
                        / SAMPLE_RATE
                        / 3600,
                        9,
                    ),
                    "waveform_id": waveform_id,
                    "waveform_path": str(waveform),
                    "annotation_sha256": source["annotation_sha256"],
                    "unit_count": len(session_units),
                    "positive_unit_count": positives,
                    "negative_unit_count": len(session_units) - positives,
                    "shared_r4_anchor_count": anchor_counts.get(session_id, 0),
                }
            )
    out = output_root(config, root)
    units_path = out / "protocol/units.jsonl"
    write_jsonl(units_path, rows)
    write_json(
        out / "protocol/inventory.json",
        {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "created_at_utc": datetime.now(UTC).isoformat(),
            "config_path": str(CONFIG_PATH.relative_to(REPOSITORY_ROOT)).replace("\\", "/"),
            "config_sha256": sha256_file(CONFIG_PATH),
            "input_identities": config["input_identities"],
            "corpus_root": str(corpus),
            "sessions": inventory,
            "summary": {
                "development_hours": round(
                    sum(row["eligible_hours"] for row in inventory if row["role"] == "development"),
                    9,
                ),
                "evaluation_hours": round(
                    sum(row["eligible_hours"] for row in inventory if row["role"] == "evaluation"),
                    9,
                ),
                "unit_count": len(rows),
                "positive_unit_count": sum(
                    row["new_speaker_onset_sample"] is not None for row in rows
                ),
                "negative_unit_count": sum(row["new_speaker_onset_sample"] is None for row in rows),
                "shared_r4_evaluation_anchor_count": sum(
                    anchor_counts.get(session_id, 0) for session_id in config["evaluation_sessions"]
                ),
            },
            "event_scope_note": (
                "The 86 shared R4 anchors are retained as a compatibility count. R6-A1 uses "
                "chronologically generated first-handoff units from the complete speaker "
                "annotations so genuine unanchored speaker activity is not counted as false."
            ),
            "git": _git_state(),
        },
    )
    return units_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "validate"))
    args = parser.parse_args(argv)
    root = cache_root()
    if args.action == "prepare":
        print(prepare(root))
    else:
        config = load_json(CONFIG_PATH)
        validate_inputs(config, root)
        print(output_root(config, root) / "protocol/inventory.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
