from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from decimal import Decimal, DecimalException, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REFERENCE_REPOSITORY = "https://github.com/nttcslab-sp/diar-forced-alignment"
REFERENCE_COMMIT = "9527b7c64846fb38316a610f32e9d3466bd6d8b7"
REFERENCE_LICENSE = "LICENSE"
REFERENCE_README = "README.md"
SAMPLE_RATE_HZ = 16000
AMI_SOURCE_TAIL_MAX_EXCESS_SAMPLES = 16
ALIMEETING_TEXTGRID_ONLY_TAIL_MAX_EXCESS_SAMPLES = 160
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
AMI_SPEAKER_PATTERN = re.compile(r"(?P<session>[A-Za-z0-9]+)\.(?P<agent>[A-Za-z0-9]+)")
ALIMEETING_SPEAKER_PATTERN = re.compile(r"(?:N|[FM])_(SPK[0-9]+)")


class ForcedAlignmentReferenceError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ReferenceSpan:
    start_sample: int
    end_sample: int
    speaker_id: str
    source_annotation_ids: tuple[str, ...]

    @property
    def duration_samples(self) -> int:
        return self.end_sample - self.start_sample

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "speaker_id": self.speaker_id,
            "source_annotation_ids": list(self.source_annotation_ids),
        }


@dataclass(frozen=True, slots=True)
class ParsedRttm:
    spans: tuple[ReferenceSpan, ...]
    raw_row_count: int
    raw_speaker_ids: tuple[str, ...]
    clipped_tail_row_count: int


def sha256_file(path: Path, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _decimal(value: str, field: str, path: Path, line_number: int) -> Decimal:
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise ForcedAlignmentReferenceError(
            f"invalid {field} in {path.name} line {line_number}"
        ) from exc
    if not result.is_finite():
        raise ForcedAlignmentReferenceError(
            f"non-finite {field} in {path.name} line {line_number}"
        )
    if abs(result.adjusted()) > 18:
        raise ForcedAlignmentReferenceError(
            f"out-of-range {field} in {path.name} line {line_number}"
        )
    return result


def timestamp_to_sample(value: Decimal) -> int:
    try:
        return round(Fraction(value) * SAMPLE_RATE_HZ)
    except (DecimalException, OverflowError, ValueError) as exc:
        raise ForcedAlignmentReferenceError("timestamp is not sample-convertible") from exc


def _exact_sample(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ForcedAlignmentReferenceError(f"{field} must be an integer sample")
    return value


def _normalized_repository_url(value: str) -> str:
    normalized = value.strip().rstrip("/")
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    return normalized.lower()


def _git(checkout_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(checkout_root), *arguments],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or "git command failed"
        raise ForcedAlignmentReferenceError(detail)
    return result.stdout.strip()


def _git_optional(checkout_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(checkout_root), *arguments],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode not in (0, 1):
        detail = result.stderr.strip() or result.stdout.strip() or "git command failed"
        raise ForcedAlignmentReferenceError(detail)
    return result.stdout.strip() if result.returncode == 0 else ""


def _git_bytes(checkout_root: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(checkout_root), *arguments],
        check=False,
        capture_output=True,
    )
    if result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ForcedAlignmentReferenceError(detail or "git command failed")
    return result.stdout


def _is_regular_non_alias_file(path: Path) -> bool:
    try:
        details = path.lstat()
    except OSError:
        return False
    attributes = getattr(details, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return (
        stat.S_ISREG(details.st_mode)
        and not path.is_symlink()
        and not (reparse_flag and attributes & reparse_flag)
    )


def _git_blob_sha1(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _tracked_tree(checkout_root: Path) -> list[tuple[str, str, str]]:
    raw = _git_bytes(checkout_root, "ls-tree", "-rz", "--full-tree", "HEAD")
    rows = []
    for entry in raw.rstrip(b"\0").split(b"\0") if raw else []:
        try:
            metadata, encoded_path = entry.split(b"\t", 1)
            mode, kind, object_id = metadata.decode("ascii").split(" ")
            relative_path = encoded_path.decode("utf-8")
        except (UnicodeError, ValueError) as exc:
            raise ForcedAlignmentReferenceError("invalid forced-alignment Git tree") from exc
        if (
            kind != "blob"
            or mode not in {"100644", "100755"}
            or not re.fullmatch(r"[0-9a-f]{40}", object_id)
            or not relative_path
        ):
            raise ForcedAlignmentReferenceError(
                f"unsupported forced-alignment Git tree entry: {relative_path}"
            )
        rows.append((relative_path, object_id, mode))
    if not rows:
        raise ForcedAlignmentReferenceError("forced-alignment Git tree is empty")
    return rows


def _validate_complete_materialization(checkout_root: Path, tree: str) -> int:
    if _git(checkout_root, "rev-parse", "--show-object-format") != "sha1":
        raise ForcedAlignmentReferenceError("unsupported forced-alignment Git object format")
    if _git(checkout_root, "write-tree") != tree:
        raise ForcedAlignmentReferenceError("forced-alignment Git index differs from HEAD")
    if _git_optional(checkout_root, "config", "--bool", "--get", "core.sparseCheckout") == "true":
        raise ForcedAlignmentReferenceError("sparse forced-alignment checkout is forbidden")
    config_names = {
        name.lower()
        for name in _git(checkout_root, "config", "--name-only", "--list").splitlines()
    }
    partial_config = any(
        name == "extensions.partialclone"
        or (
            name.startswith("remote.")
            and name.endswith((".promisor", ".partialclonefilter"))
        )
        for name in config_names
    )
    promisor_markers = tuple((checkout_root / ".git" / "objects" / "pack").glob("*.promisor"))
    if partial_config or promisor_markers:
        raise ForcedAlignmentReferenceError("partial forced-alignment checkout is forbidden")
    index_flags = _git(checkout_root, "ls-files", "-v").splitlines()
    if any(not line.startswith("H ") for line in index_flags):
        raise ForcedAlignmentReferenceError(
            "forced-alignment checkout has noncanonical index flags"
        )
    untracked = _git_bytes(
        checkout_root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
    )
    ignored = _git_bytes(
        checkout_root,
        "ls-files",
        "--others",
        "--ignored",
        "--exclude-standard",
        "-z",
    )
    if untracked or ignored:
        raise ForcedAlignmentReferenceError(
            "forced-alignment checkout contains untracked or ignored files"
        )
    rows = _tracked_tree(checkout_root)
    for relative_path, object_id, _ in rows:
        path = checkout_root / relative_path
        if not _is_regular_non_alias_file(path):
            raise ForcedAlignmentReferenceError(
                f"forced-alignment tracked file is missing or aliased: {relative_path}"
            )
        if _git_blob_sha1(path) != object_id:
            raise ForcedAlignmentReferenceError(
                f"forced-alignment tracked bytes differ from Git: {relative_path}"
            )
    return len(rows)


def validate_reference_checkout(reference_root: Path) -> dict[str, Any]:
    root = reference_root.resolve()
    if not root.is_dir() or not (root / ".git").is_dir():
        raise ForcedAlignmentReferenceError("forced-alignment reference is not a Git checkout")
    head = _git(root, "rev-parse", "HEAD^{commit}")
    if head != REFERENCE_COMMIT:
        raise ForcedAlignmentReferenceError(
            f"forced-alignment reference commit mismatch: {head}"
        )
    remote = _git(root, "remote", "get-url", "origin")
    if _normalized_repository_url(remote) != _normalized_repository_url(
        REFERENCE_REPOSITORY
    ):
        raise ForcedAlignmentReferenceError(
            f"forced-alignment reference repository mismatch: {remote}"
        )
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    if not re.fullmatch(r"[0-9a-f]{40}", tree):
        raise ForcedAlignmentReferenceError("forced-alignment reference tree is invalid")
    tracked_file_count = _validate_complete_materialization(root, tree)
    for name in (REFERENCE_LICENSE, REFERENCE_README):
        if not (root / name).is_file():
            raise ForcedAlignmentReferenceError(
                f"forced-alignment reference is missing {name}"
            )
    return {
        "repository": REFERENCE_REPOSITORY,
        "commit": head,
        "git_tree": tree,
        "git_object_format": "sha1",
        "tracked_file_count": tracked_file_count,
        "license_ref": REFERENCE_LICENSE,
        "license_sha256": sha256_file(root / REFERENCE_LICENSE),
        "readme_ref": REFERENCE_README,
        "readme_sha256": sha256_file(root / REFERENCE_README),
    }


def acquire_reference(reference_root: Path) -> dict[str, Any]:
    target = reference_root.resolve()
    if target.exists():
        return validate_reference_checkout(target)
    parent = target.parent
    if not parent.is_dir():
        raise ForcedAlignmentReferenceError(
            "forced-alignment reference parent directory does not exist"
        )
    with tempfile.TemporaryDirectory(prefix="diar-forced-alignment-", dir=parent) as value:
        temporary = Path(value) / "checkout"
        clone = subprocess.run(
            ["git", "clone", "--no-checkout", REFERENCE_REPOSITORY, str(temporary)],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if clone.returncode:
            detail = clone.stderr.strip() or clone.stdout.strip() or "git clone failed"
            raise ForcedAlignmentReferenceError(detail)
        _git(temporary, "config", "core.autocrlf", "false")
        _git(temporary, "config", "core.eol", "lf")
        _git(temporary, "checkout", "--detach", REFERENCE_COMMIT)
        validate_reference_checkout(temporary)
        temporary.replace(target)
    return validate_reference_checkout(target)


def _reference_roots(corpus: str) -> tuple[Path, ...]:
    if corpus == "AMI":
        return (Path("AMI/train"), Path("AMI/dev"), Path("AMI/test"))
    if corpus == "AliMeeting":
        return (Path("AliMeeting/Train_Ali_far"), Path("AliMeeting/Eval_Ali_far"))
    raise ForcedAlignmentReferenceError(f"unsupported reference corpus: {corpus}")


def resolve_reference_path(
    reference_root: Path,
    *,
    corpus: str,
    session_id: str,
) -> Path:
    if not isinstance(session_id, str) or not re.fullmatch(r"[A-Za-z0-9_]+", session_id):
        raise ForcedAlignmentReferenceError("invalid reference session identity")
    root = reference_root.resolve()
    candidates = []
    for relative_root in _reference_roots(corpus):
        candidate = root / relative_root / f"{session_id}.rttm"
        try:
            candidate.lstat()
        except OSError:
            continue
        if not _is_regular_non_alias_file(candidate):
            raise ForcedAlignmentReferenceError(
                f"RTTM reference is not a regular non-alias file: {candidate.name}"
            )
        candidates.append(candidate)
    if len(candidates) != 1:
        raise ForcedAlignmentReferenceError(
            f"expected exactly one RTTM for {corpus}/{session_id}, found {len(candidates)}"
        )
    selected = candidates[0]
    if selected.parent.resolve() != selected.parent or not selected.is_relative_to(root):
        raise ForcedAlignmentReferenceError("RTTM path escapes the reference checkout")
    return selected


def build_ami_speaker_map(meetings_path: Path, session_id: str) -> dict[str, str]:
    try:
        root = ET.parse(meetings_path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise ForcedAlignmentReferenceError(
            f"invalid AMI meetings metadata: {meetings_path}"
        ) from exc
    meetings = [
        element
        for element in root.iter("meeting")
        if element.get("observation") == session_id
    ]
    if len(meetings) != 1:
        raise ForcedAlignmentReferenceError(
            f"AMI meeting metadata cardinality mismatch: {session_id}"
        )
    speaker_map: dict[str, str] = {}
    for speaker in meetings[0].findall("speaker"):
        agent = speaker.get("nxt_agent")
        identity = speaker.get("global_name")
        if (
            not agent
            or not identity
            or agent.strip() != agent
            or identity.strip() != identity
            or agent in speaker_map
            or identity in speaker_map.values()
        ):
            raise ForcedAlignmentReferenceError(
                f"AMI speaker identity is unresolved or duplicated: {session_id}"
            )
        speaker_map[agent] = identity
    if not speaker_map:
        raise ForcedAlignmentReferenceError(
            f"AMI meeting has no official speaker identities: {session_id}"
        )
    return speaker_map


def build_alimeeting_speaker_map(speaker_ids: Sequence[str]) -> dict[str, str]:
    if isinstance(speaker_ids, (str, bytes)):
        raise ForcedAlignmentReferenceError("AliMeeting speaker inventory is invalid")
    identities = list(speaker_ids)
    if (
        not identities
        or any(
            not isinstance(identity, str)
            or ALIMEETING_SPEAKER_PATTERN.fullmatch(f"N_{identity}") is None
            for identity in identities
        )
        or len(set(identities)) != len(identities)
    ):
        raise ForcedAlignmentReferenceError("AliMeeting speaker inventory is invalid")
    return {identity: identity for identity in identities}


def _mapped_speaker(
    raw_speaker: str,
    *,
    corpus: str,
    session_id: str,
    speaker_map: Mapping[str, str],
    path: Path,
    line_number: int,
) -> str:
    if corpus == "AMI":
        match = AMI_SPEAKER_PATTERN.fullmatch(raw_speaker)
        if match is None or match.group("session") != session_id:
            raise ForcedAlignmentReferenceError(
                f"invalid AMI RTTM speaker in {path.name} line {line_number}"
            )
        source_speaker = match.group("agent")
    elif corpus == "AliMeeting":
        match = ALIMEETING_SPEAKER_PATTERN.fullmatch(raw_speaker)
        if match is None:
            raise ForcedAlignmentReferenceError(
                f"invalid AliMeeting RTTM speaker in {path.name} line {line_number}"
            )
        source_speaker = match.group(1)
    else:
        raise ForcedAlignmentReferenceError(f"unsupported reference corpus: {corpus}")
    mapped = speaker_map.get(source_speaker)
    if not isinstance(mapped, str) or not mapped:
        raise ForcedAlignmentReferenceError(
            f"unmapped RTTM speaker {raw_speaker} in {path.name} line {line_number}"
        )
    if corpus == "AliMeeting" and mapped != source_speaker:
        raise ForcedAlignmentReferenceError(
            f"noncanonical AliMeeting speaker mapping in {path.name} line {line_number}"
        )
    return mapped


def _union_same_speaker(spans: Iterable[ReferenceSpan]) -> tuple[ReferenceSpan, ...]:
    by_speaker: dict[str, list[ReferenceSpan]] = {}
    for span in spans:
        by_speaker.setdefault(span.speaker_id, []).append(span)
    merged: list[ReferenceSpan] = []
    for speaker_id, speaker_spans in sorted(by_speaker.items()):
        ordered = sorted(
            speaker_spans,
            key=lambda span: (
                span.start_sample,
                span.end_sample,
                span.source_annotation_ids,
            ),
        )
        current = ordered[0]
        for span in ordered[1:]:
            if span.start_sample <= current.end_sample:
                current = ReferenceSpan(
                    start_sample=current.start_sample,
                    end_sample=max(current.end_sample, span.end_sample),
                    speaker_id=speaker_id,
                    source_annotation_ids=tuple(
                        dict.fromkeys(
                            current.source_annotation_ids + span.source_annotation_ids
                        )
                    ),
                )
            else:
                merged.append(current)
                current = span
        merged.append(current)
    return tuple(
        sorted(
            merged,
            key=lambda span: (
                span.start_sample,
                span.end_sample,
                span.speaker_id,
                span.source_annotation_ids,
            ),
        )
    )


def parse_rttm(
    path: Path,
    *,
    corpus: str,
    session_id: str,
    speaker_map: Mapping[str, str],
    scored_start_sample: int,
    scored_end_sample: int,
) -> ParsedRttm:
    start_bound = _exact_sample(scored_start_sample, "scored_start_sample")
    end_bound = _exact_sample(scored_end_sample, "scored_end_sample")
    if start_bound < 0 or end_bound <= start_bound:
        raise ForcedAlignmentReferenceError("invalid scored reference range")
    if not speaker_map or any(
        not isinstance(source, str)
        or not source
        or source.strip() != source
        or not isinstance(target, str)
        or not target
        or target.strip() != target
        for source, target in speaker_map.items()
    ):
        raise ForcedAlignmentReferenceError("reference speaker map is invalid")
    if len(set(speaker_map.values())) != len(speaker_map):
        raise ForcedAlignmentReferenceError("reference speaker map is not one-to-one")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ForcedAlignmentReferenceError(f"RTTM is unavailable or invalid: {path}") from exc
    spans: list[ReferenceSpan] = []
    raw_speakers: set[str] = set()
    clipped_tail_rows = 0
    for line_number, line in enumerate(lines, 1):
        fields = line.split()
        if len(fields) != 10:
            raise ForcedAlignmentReferenceError(
                f"RTTM row must have ten fields in {path.name} line {line_number}"
            )
        if (
            fields[0] != "SPEAKER"
            or fields[1] != session_id
            or fields[2] != "1"
            or fields[5] != "<NA>"
            or fields[6] != "<NA>"
            or fields[8] != "<NA>"
            or fields[9] != "<NA>"
        ):
            raise ForcedAlignmentReferenceError(
                f"unexpected RTTM row identity in {path.name} line {line_number}"
            )
        start = _decimal(fields[3], "RTTM start", path, line_number)
        duration = _decimal(fields[4], "RTTM duration", path, line_number)
        if start < 0 or duration <= 0:
            raise ForcedAlignmentReferenceError(
                f"invalid RTTM bounds in {path.name} line {line_number}"
            )
        try:
            start_position = Fraction(start) * SAMPLE_RATE_HZ
            end_position = (Fraction(start) + Fraction(duration)) * SAMPLE_RATE_HZ
            start_sample = round(start_position)
            end_sample = round(end_position)
        except (DecimalException, OverflowError, ValueError) as exc:
            raise ForcedAlignmentReferenceError(
                f"unconvertible RTTM bounds in {path.name} line {line_number}"
            ) from exc
        if (
            start_position < start_bound
            or start_position >= end_bound
            or end_sample <= start_sample
        ):
            raise ForcedAlignmentReferenceError(
                f"RTTM row exceeds the scored range in {path.name} line {line_number}"
            )
        if end_position > end_bound:
            excess = end_position - end_bound
            if (
                corpus != "AMI"
                or excess > AMI_SOURCE_TAIL_MAX_EXCESS_SAMPLES
                or start_sample >= end_bound
            ):
                raise ForcedAlignmentReferenceError(
                    f"RTTM row exceeds the scored range in {path.name} line {line_number}"
                )
            end_sample = end_bound
            clipped_tail_rows += 1
        raw_speaker = fields[7]
        speaker_id = _mapped_speaker(
            raw_speaker,
            corpus=corpus,
            session_id=session_id,
            speaker_map=speaker_map,
            path=path,
            line_number=line_number,
        )
        raw_speakers.add(raw_speaker)
        spans.append(
            ReferenceSpan(
                start_sample=start_sample,
                end_sample=end_sample,
                speaker_id=speaker_id,
                source_annotation_ids=(f"{path.name}#L{line_number}",),
            )
        )
    if not spans:
        raise ForcedAlignmentReferenceError(f"RTTM contains no speech rows: {path}")
    return ParsedRttm(
        spans=_union_same_speaker(spans),
        raw_row_count=len(spans),
        raw_speaker_ids=tuple(sorted(raw_speakers)),
        clipped_tail_row_count=clipped_tail_rows,
    )


def build_reference_inventory(
    reference_root: Path,
    selected_sources: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    checkout = validate_reference_checkout(reference_root)
    source_ids: set[str] = set()
    sessions: set[tuple[str, str]] = set()
    reference_paths: set[Path] = set()
    rows = []
    for source in selected_sources:
        source_id = source.get("source_id")
        corpus = source.get("corpus")
        session_id = source.get("session_id")
        speaker_map = source.get("speaker_map")
        if (
            not isinstance(source_id, str)
            or not source_id
            or source_id in source_ids
            or not isinstance(corpus, str)
            or not isinstance(session_id, str)
            or not isinstance(speaker_map, Mapping)
        ):
            raise ForcedAlignmentReferenceError("reference selection row is invalid")
        source_ids.add(source_id)
        session_key = (corpus, session_id)
        if session_key in sessions:
            raise ForcedAlignmentReferenceError(
                f"duplicate reference selection for {corpus}/{session_id}"
            )
        sessions.add(session_key)
        path = resolve_reference_path(
            reference_root,
            corpus=corpus,
            session_id=session_id,
        )
        if path in reference_paths:
            raise ForcedAlignmentReferenceError(
                f"duplicate consumed RTTM reference: {path.name}"
            )
        reference_paths.add(path)
        parsed = parse_rttm(
            path,
            corpus=corpus,
            session_id=session_id,
            speaker_map=speaker_map,
            scored_start_sample=source.get("scored_start_sample"),
            scored_end_sample=source.get("scored_end_sample"),
        )
        span_rows = [span.to_dict() for span in parsed.spans]
        rows.append(
            {
                "source_id": source_id,
                "corpus": corpus,
                "session_id": session_id,
                "reference_ref": path.relative_to(reference_root.resolve()).as_posix(),
                "reference_sha256": sha256_file(path),
                "reference_size_bytes": path.stat().st_size,
                "raw_row_count": parsed.raw_row_count,
                "clipped_tail_row_count": parsed.clipped_tail_row_count,
                "canonical_span_count": len(parsed.spans),
                "raw_speaker_ids": list(parsed.raw_speaker_ids),
                "mapped_speaker_ids": sorted({span.speaker_id for span in parsed.spans}),
                "speaker_map": dict(sorted(speaker_map.items())),
                "speaker_map_sha256": canonical_sha256(dict(sorted(speaker_map.items()))),
                "canonical_spans_sha256": canonical_sha256(span_rows),
            }
        )
    if not rows:
        raise ForcedAlignmentReferenceError("reference selection is empty")
    ordered_rows = sorted(rows, key=lambda row: row["source_id"])
    return {
        "schema_version": 1,
        "artifact_role": "psem_forced_alignment_reference_provenance",
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "source_coordinate_convention": "zero_based_half_open_unsnapped_source_samples",
        "source_tail_rule": {
            "corpus": "AMI",
            "action": "clip_terminal_rttm_end_to_scored_waveform_end",
            "maximum_excess_samples": AMI_SOURCE_TAIL_MAX_EXCESS_SAMPLES,
            "basis": "one 1 ms RTTM timestamp quantum at 16 kHz",
            "row_start_must_precede_scored_end": True,
        },
        "upstream": checkout,
        "source_count": len(ordered_rows),
        "sources_sha256": canonical_sha256(ordered_rows),
        "sources": ordered_rows,
    }


def _load_selection(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ForcedAlignmentReferenceError(f"invalid reference selection: {path}") from exc
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ForcedAlignmentReferenceError("reference selection must be a JSON array")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    path.write_text(payload + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--acquire", action="store_true")
    args = parser.parse_args()
    if args.acquire:
        acquire_reference(args.reference_root)
    inventory = build_reference_inventory(
        args.reference_root,
        _load_selection(args.selection),
    )
    _write_json(args.output, inventory)


if __name__ == "__main__":
    main()
