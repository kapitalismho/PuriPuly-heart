from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import tarfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from experiments.psem_training_strategy_gate.data.forced_alignment_reference import (
    ALIMEETING_TEXTGRID_ONLY_TAIL_MAX_EXCESS_SAMPLES,
    REFERENCE_COMMIT,
    REFERENCE_REPOSITORY,
    canonical_sha256,
    parse_rttm,
    resolve_reference_path,
    sha256_file,
)
from experiments.psem_training_strategy_gate.data.reference_normalization import (
    alimeeting_speaker_ids,
    alimeeting_textgrid_range_samples,
    open_reference_checkout,
)

AUTHORITY_REF = "https://github.com/kapitalismho/PuriPuly-heart/issues/86"
AUTHORITY_PIN = "90078d66026f1374b065a5b9022788c40fac076cd4cf307df87b5027ea3fcb63"
OPENSLR_RESOURCE_URL = "https://www.openslr.org/119/"
TRAIN_ARCHIVE_URL = (
    "https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/"
    "AliMeeting/openlr/Train_Ali_far.tar.gz"
)
TRAIN_ARCHIVE_SIZE_BYTES = 78639309701
TRAIN_ARCHIVE_SHA256 = (
    "98a5b2840704c7fb2ab3fdc7e5c63c25bd453528e0e32443938279bae8e200e7"
)
ROOM_WORKBOOK_URL = (
    "https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/"
    "AliMeeting/AliMeeting_Trainset_Room.xlsx"
)
ROOM_WORKBOOK_SIZE_BYTES = 10276
ROOM_WORKBOOK_SHA256 = (
    "c475dc78ad9d3c86676ca7e8eecbd64c00059f7e9ed4bb47c82c97c515bab390"
)
SELECTION_RECEIPT_SHA256 = (
    "cce8d9a60595df7ba31b3a2a1cf70fe6ee4abd373c91c833fe5750250e8fc5fd"
)
SELECTION_SALT = "PSEM-STRATEGY-DATA-v2"
SAMPLE_RATE_HZ = 16000
MIN_MEETINGS = 14
MAX_MEETINGS = 18
MIN_SCORED_SAMPLES = 7 * 3600 * SAMPLE_RATE_HZ
MAX_SCORED_SAMPLES = 9 * 3600 * SAMPLE_RATE_HZ
TARGET_SCORED_SAMPLES = 8 * 3600 * SAMPLE_RATE_HZ
PARTICIPANT_BUCKETS = (2, 3, 4)
CATALOG_SESSION_COUNT = 212
CATALOG_SCORED_HOURS = 104.75
ARCHIVE_SESSION_COUNT = 209
ARCHIVE_ROOTS = frozenset(
    {
        "Train_Ali_far",
        "Train_Ali_far/audio_dir",
        "Train_Ali_far/textgrid_dir",
    }
)
SESSION_PATTERN = re.compile(r"^(?P<room>R[0-9]+)_(?P<meeting>M[0-9]+)$")
WAVEFORM_MEMBER_PATTERN = re.compile(
    r"^Train_Ali_far/audio_dir/"
    r"(?P<session>R[0-9]+_M[0-9]+)_(?P<recording_group>MS[0-9]+)\.wav$"
)
TEXTGRID_MEMBER_PATTERN = re.compile(
    r"^Train_Ali_far/textgrid_dir/"
    r"(?P<session>R[0-9]+_M[0-9]+)\.TextGrid$"
)


class AliMeetingTrainSelectionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ArchiveMember:
    ref: str
    size_bytes: int
    sha256: str | None
    duration_samples: int | None = None


@dataclass(frozen=True, slots=True)
class CandidateSession:
    session_id: str
    room_id: str
    meeting_id: str
    recording_group_id: str
    scored_samples: int
    textgrid_timeline_samples: int
    annotation_tail_excess_samples: int
    reference_end_sample: int
    selection_eligible: bool
    selection_exclusion_reasons: tuple[str, ...]
    participant_count: int
    speaker_ids: tuple[str, ...]
    waveform_member: ArchiveMember
    textgrid_member: ArchiveMember
    reference_ref: str
    reference_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "room_id": self.room_id,
            "meeting_id": self.meeting_id,
            "recording_group_id": self.recording_group_id,
            "scored_samples": self.scored_samples,
            "scored_hours": round(
                self.scored_samples / SAMPLE_RATE_HZ / 3600,
                9,
            ),
            "waveform_duration_samples": self.waveform_member.duration_samples,
            "textgrid_timeline_samples": self.textgrid_timeline_samples,
            "annotation_tail_excess_samples": (
                self.annotation_tail_excess_samples
            ),
            "reference_end_sample": self.reference_end_sample,
            "reference_tail_excess_samples": max(
                0,
                self.reference_end_sample
                - (self.waveform_member.duration_samples or 0),
            ),
            "selection_eligible": self.selection_eligible,
            "selection_exclusion_reasons": list(
                self.selection_exclusion_reasons
            ),
            "participant_count": self.participant_count,
            "speaker_ids": list(self.speaker_ids),
            "waveform_archive_member_ref": self.waveform_member.ref,
            "waveform_archive_member_size_bytes": (
                self.waveform_member.size_bytes
            ),
            "textgrid_archive_member_ref": self.textgrid_member.ref,
            "textgrid_archive_member_size_bytes": self.textgrid_member.size_bytes,
            "textgrid_sha256": self.textgrid_member.sha256,
            "reference_ref": self.reference_ref,
            "reference_sha256": self.reference_sha256,
        }


@dataclass(frozen=True, slots=True)
class CandidateComponent:
    component_id: str
    session_ids: tuple[str, ...]
    scored_samples: int
    participant_buckets: tuple[int, ...]
    room_ids: tuple[str, ...]
    recording_group_ids: tuple[str, ...]
    speaker_ids: tuple[str, ...]
    shared_speaker_ids: tuple[str, ...]
    selection_eligible: bool
    excluded_session_ids: tuple[str, ...]

    @property
    def meeting_count(self) -> int:
        return len(self.session_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "session_ids": list(self.session_ids),
            "meeting_count": self.meeting_count,
            "scored_samples": self.scored_samples,
            "participant_buckets": list(self.participant_buckets),
            "room_ids": list(self.room_ids),
            "recording_group_ids": list(self.recording_group_ids),
            "speaker_ids": list(self.speaker_ids),
            "shared_identity_reasons": [
                {"axis": "known_speaker_identity", "value": speaker_id}
                for speaker_id in self.shared_speaker_ids
            ],
            "selection_eligible": self.selection_eligible,
            "excluded_session_ids": list(self.excluded_session_ids),
        }


@dataclass(frozen=True, slots=True)
class SelectionResult:
    participant_bucket_coverage: int
    metadata_identity_coverage: int
    scored_samples: int
    target_distance_samples: int
    optimal_candidate_count: int
    component_ids: tuple[str, ...]
    selection_hash: str


class _DigestingReader:
    def __init__(self, handle: BinaryIO) -> None:
        self._handle = handle
        self._digest = hashlib.sha256()
        self.size_bytes = 0

    def read(self, size: int = -1) -> bytes:
        data = self._handle.read(size)
        self._digest.update(data)
        self.size_bytes += len(data)
        return data

    @property
    def sha256(self) -> str:
        return self._digest.hexdigest()


class _Components:
    def __init__(self, values: Iterable[str]) -> None:
        self._parent = {value: value for value in values}

    def find(self, value: str) -> str:
        root = value
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[value] != value:
            following = self._parent[value]
            self._parent[value] = root
            value = following
        return root

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self._parent[max(left_root, right_root)] = min(
                left_root,
                right_root,
            )


def _regular_file_identity(
    path: Path,
    expected_size_bytes: int,
    expected_sha256: str,
) -> None:
    try:
        resolved = path.resolve(strict=True)
        size_bytes = resolved.stat().st_size
    except OSError as exc:
        raise AliMeetingTrainSelectionError(
            f"required AliMeeting artifact is unavailable: {path}"
        ) from exc
    if (
        not resolved.is_file()
        or resolved.is_symlink()
        or size_bytes != expected_size_bytes
        or sha256_file(resolved) != expected_sha256
    ):
        raise AliMeetingTrainSelectionError(
            f"AliMeeting artifact identity changed: {path.name}"
        )


def _waveform_duration_samples(payload: bytes, member_size: int) -> int:
    if (
        len(payload) < 44
        or payload[:4] != b"RIFF"
        or payload[8:12] != b"WAVE"
        or struct.unpack_from("<I", payload, 4)[0] + 8 != member_size
    ):
        raise AliMeetingTrainSelectionError(
            "AliMeeting waveform RIFF identity is invalid"
        )
    offset = 12
    format_payload: bytes | None = None
    fact_samples: int | None = None
    data_size: int | None = None
    data_offset: int | None = None
    while offset + 8 <= len(payload):
        chunk_id = payload[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", payload, offset + 4)[0]
        chunk_start = offset + 8
        if chunk_id == b"data":
            data_size = chunk_size
            data_offset = chunk_start
            break
        chunk_end = chunk_start + chunk_size
        if chunk_end > len(payload):
            raise AliMeetingTrainSelectionError(
                "AliMeeting waveform header exceeds the inspected prefix"
            )
        if chunk_id == b"fmt ":
            if format_payload is not None:
                raise AliMeetingTrainSelectionError(
                    "AliMeeting waveform format chunk is duplicated"
                )
            format_payload = payload[chunk_start:chunk_end]
        elif chunk_id == b"fact":
            if fact_samples is not None or chunk_size != 4:
                raise AliMeetingTrainSelectionError(
                    "AliMeeting waveform fact chunk is invalid"
                )
            fact_samples = struct.unpack_from("<I", payload, chunk_start)[0]
        offset = chunk_end + (chunk_size & 1)
    if (
        format_payload is None
        or len(format_payload) < 16
        or data_size is None
        or data_offset is None
        or data_offset + data_size + (data_size & 1) != member_size
    ):
        raise AliMeetingTrainSelectionError(
            "AliMeeting waveform chunk inventory is invalid"
        )
    (
        format_tag,
        channels,
        sample_rate_hz,
        byte_rate,
        block_align,
        bits_per_sample,
    ) = struct.unpack_from("<HHIIHH", format_payload)
    pcm_subformat = bytes.fromhex("0100000000001000800000aa00389b71")
    if format_tag == 0xFFFE:
        if (
            len(format_payload) != 40
            or struct.unpack_from("<H", format_payload, 16)[0] != 22
            or struct.unpack_from("<H", format_payload, 18)[0] != 16
            or format_payload[24:40] != pcm_subformat
        ):
            raise AliMeetingTrainSelectionError(
                "AliMeeting waveform extensible PCM format is invalid"
            )
    elif format_tag != 1 or len(format_payload) != 16:
        raise AliMeetingTrainSelectionError(
            "AliMeeting waveform encoding is unsupported"
        )
    if (
        channels != 8
        or sample_rate_hz != SAMPLE_RATE_HZ
        or bits_per_sample != 16
        or block_align != 16
        or byte_rate != SAMPLE_RATE_HZ * block_align
        or data_size <= 0
        or data_size % block_align
    ):
        raise AliMeetingTrainSelectionError(
            "AliMeeting waveform is not complete 16 kHz 8-channel PCM16"
        )
    duration_samples = data_size // block_align
    if fact_samples is not None and fact_samples != duration_samples:
        raise AliMeetingTrainSelectionError(
            "AliMeeting waveform fact duration is inconsistent"
        )
    return duration_samples


def _accepted_scored_samples(
    textgrid_timeline_samples: int,
    waveform_duration_samples: int,
) -> tuple[int, int]:
    excess = max(0, textgrid_timeline_samples - waveform_duration_samples)
    return min(textgrid_timeline_samples, waveform_duration_samples), excess


def _selection_exclusion_reasons(
    *,
    textgrid_timeline_samples: int,
    waveform_duration_samples: int,
    reference_end_sample: int,
) -> tuple[str, ...]:
    reasons = []
    if reference_end_sample > waveform_duration_samples:
        reasons.append("forced_alignment_extends_beyond_waveform")
    if (
        textgrid_timeline_samples - waveform_duration_samples
        > ALIMEETING_TEXTGRID_ONLY_TAIL_MAX_EXCESS_SAMPLES
    ):
        reasons.append("textgrid_tail_exceeds_documented_source_tail_rule")
    return tuple(reasons)


def _scan_archive(
    archive_path: Path,
) -> tuple[dict[str, ArchiveMember], dict[str, ArchiveMember]]:
    try:
        archive_size = archive_path.resolve(strict=True).stat().st_size
    except OSError as exc:
        raise AliMeetingTrainSelectionError(
            f"AliMeeting Train archive is unavailable: {archive_path}"
        ) from exc
    if archive_size != TRAIN_ARCHIVE_SIZE_BYTES:
        raise AliMeetingTrainSelectionError(
            "AliMeeting Train archive size changed"
        )
    waveform_members: dict[str, ArchiveMember] = {}
    textgrid_members: dict[str, ArchiveMember] = {}
    directory_members: set[str] = set()
    member_count = 0
    try:
        with archive_path.open("rb") as raw_handle:
            digesting_reader = _DigestingReader(raw_handle)
            with tarfile.open(fileobj=digesting_reader, mode="r|gz") as bundle:
                for member in bundle:
                    member_count += 1
                    normalized_name = member.name.rstrip("/")
                    if member.isdir():
                        directory_members.add(normalized_name)
                        continue
                    if not member.isfile():
                        raise AliMeetingTrainSelectionError(
                            f"unsupported AliMeeting archive member: {member.name}"
                        )
                    waveform_match = WAVEFORM_MEMBER_PATTERN.fullmatch(
                        member.name
                    )
                    if waveform_match is not None:
                        session_id = waveform_match.group("session")
                        if session_id in waveform_members or member.size <= 44:
                            raise AliMeetingTrainSelectionError(
                                "AliMeeting waveform archive members are invalid"
                            )
                        extracted = bundle.extractfile(member)
                        if extracted is None:
                            raise AliMeetingTrainSelectionError(
                                f"AliMeeting waveform member is unreadable: {member.name}"
                            )
                        header = extracted.read(min(member.size, 1 << 20))
                        waveform_members[session_id] = ArchiveMember(
                            ref=member.name,
                            size_bytes=member.size,
                            sha256=None,
                            duration_samples=_waveform_duration_samples(
                                header,
                                member.size,
                            ),
                        )
                        continue
                    textgrid_match = TEXTGRID_MEMBER_PATTERN.fullmatch(
                        member.name
                    )
                    if textgrid_match is None:
                        raise AliMeetingTrainSelectionError(
                            f"unexpected AliMeeting archive member: {member.name}"
                        )
                    session_id = textgrid_match.group("session")
                    if session_id in textgrid_members:
                        raise AliMeetingTrainSelectionError(
                            "AliMeeting TextGrid archive members are duplicated"
                        )
                    extracted = bundle.extractfile(member)
                    if extracted is None:
                        raise AliMeetingTrainSelectionError(
                            f"AliMeeting TextGrid member is unreadable: {member.name}"
                        )
                    payload = extracted.read()
                    if len(payload) != member.size:
                        raise AliMeetingTrainSelectionError(
                            f"AliMeeting TextGrid member is incomplete: {member.name}"
                        )
                    textgrid_members[session_id] = ArchiveMember(
                        ref=member.name,
                        size_bytes=member.size,
                        sha256=hashlib.sha256(payload).hexdigest(),
                    )
            while digesting_reader.read(1 << 20):
                pass
            archive_sha256 = digesting_reader.sha256
            observed_archive_size = digesting_reader.size_bytes
    except (OSError, EOFError, tarfile.TarError) as exc:
        raise AliMeetingTrainSelectionError(
            "AliMeeting Train archive is invalid"
        ) from exc
    if (
        observed_archive_size != TRAIN_ARCHIVE_SIZE_BYTES
        or archive_sha256 != TRAIN_ARCHIVE_SHA256
        or member_count != 2 * ARCHIVE_SESSION_COUNT + len(ARCHIVE_ROOTS)
        or directory_members != ARCHIVE_ROOTS
        or len(waveform_members) != ARCHIVE_SESSION_COUNT
        or len(textgrid_members) != ARCHIVE_SESSION_COUNT
        or set(waveform_members) != set(textgrid_members)
    ):
        raise AliMeetingTrainSelectionError(
            "AliMeeting Train archive inventory changed"
        )
    return waveform_members, textgrid_members


def _recording_group(member_ref: str, session_id: str) -> str:
    match = WAVEFORM_MEMBER_PATTERN.fullmatch(member_ref)
    if match is None or match.group("session") != session_id:
        raise AliMeetingTrainSelectionError(
            f"AliMeeting waveform member identity is invalid: {session_id}"
        )
    return match.group("recording_group")


def _candidate_sessions(
    waveform_members: Mapping[str, ArchiveMember],
    textgrid_members: Mapping[str, ArchiveMember],
    textgrid_root: Path,
    reference_root: Path,
) -> tuple[CandidateSession, ...]:
    checkout = open_reference_checkout(reference_root)
    root = textgrid_root.resolve()
    if not root.is_dir():
        raise AliMeetingTrainSelectionError(
            "AliMeeting Train TextGrid root is unavailable"
        )
    observed_textgrids = {
        path.stem: path for path in sorted(root.glob("*.TextGrid"))
    }
    if (
        len(observed_textgrids) != ARCHIVE_SESSION_COUNT
        or set(observed_textgrids) != set(textgrid_members)
    ):
        raise AliMeetingTrainSelectionError(
            "extracted AliMeeting TextGrid inventory changed"
        )
    rows = []
    meeting_ids: set[str] = set()
    for session_id in sorted(waveform_members):
        session_match = SESSION_PATTERN.fullmatch(session_id)
        if session_match is None:
            raise AliMeetingTrainSelectionError(
                f"invalid AliMeeting Train session identity: {session_id}"
            )
        meeting_id = session_match.group("meeting")
        if meeting_id in meeting_ids:
            raise AliMeetingTrainSelectionError(
                f"duplicate AliMeeting meeting identity: {meeting_id}"
            )
        meeting_ids.add(meeting_id)
        textgrid_path = observed_textgrids[session_id]
        archived_textgrid = textgrid_members[session_id]
        if (
            textgrid_path.stat().st_size != archived_textgrid.size_bytes
            or sha256_file(textgrid_path) != archived_textgrid.sha256
        ):
            raise AliMeetingTrainSelectionError(
                f"extracted AliMeeting TextGrid changed: {session_id}"
            )
        scored_start_sample, scored_end_sample = (
            alimeeting_textgrid_range_samples(textgrid_path)
        )
        if scored_start_sample != 0:
            raise AliMeetingTrainSelectionError(
                f"AliMeeting Train scored range does not start at zero: {session_id}"
            )
        waveform_duration_samples = waveform_members[
            session_id
        ].duration_samples
        if waveform_duration_samples is None:
            raise AliMeetingTrainSelectionError(
                f"AliMeeting waveform duration is unresolved: {session_id}"
            )
        textgrid_timeline_samples = scored_end_sample
        speakers = tuple(sorted(alimeeting_speaker_ids(textgrid_path)))
        if len(speakers) not in PARTICIPANT_BUCKETS:
            raise AliMeetingTrainSelectionError(
                f"unsupported AliMeeting participant count: {session_id}"
            )
        reference_path = resolve_reference_path(
            checkout.root,
            corpus="AliMeeting",
            session_id=session_id,
        )
        parsed_reference = parse_rttm(
            reference_path,
            corpus="AliMeeting",
            session_id=session_id,
            speaker_map={speaker: speaker for speaker in speakers},
            scored_start_sample=scored_start_sample,
            scored_end_sample=textgrid_timeline_samples,
        )
        reference_end_sample = max(
            (span.end_sample for span in parsed_reference.spans),
            default=0,
        )
        scored_end_sample, annotation_tail_excess_samples = (
            _accepted_scored_samples(
                textgrid_timeline_samples,
                waveform_duration_samples,
            )
        )
        selection_exclusion_reasons = _selection_exclusion_reasons(
            textgrid_timeline_samples=textgrid_timeline_samples,
            waveform_duration_samples=waveform_duration_samples,
            reference_end_sample=reference_end_sample,
        )
        mapped_reference_speakers = {
            raw.split("_", 1)[1]
            for raw in parsed_reference.raw_speaker_ids
        }
        if mapped_reference_speakers != set(speakers):
            raise AliMeetingTrainSelectionError(
                f"AliMeeting reference speaker inventory changed: {session_id}"
            )
        rows.append(
            CandidateSession(
                session_id=session_id,
                room_id=session_match.group("room"),
                meeting_id=meeting_id,
                recording_group_id=_recording_group(
                    waveform_members[session_id].ref,
                    session_id,
                ),
                scored_samples=scored_end_sample,
                textgrid_timeline_samples=textgrid_timeline_samples,
                annotation_tail_excess_samples=(
                    annotation_tail_excess_samples
                ),
                reference_end_sample=reference_end_sample,
                selection_eligible=not selection_exclusion_reasons,
                selection_exclusion_reasons=selection_exclusion_reasons,
                participant_count=len(speakers),
                speaker_ids=speakers,
                waveform_member=waveform_members[session_id],
                textgrid_member=archived_textgrid,
                reference_ref=reference_path.relative_to(checkout.root).as_posix(),
                reference_sha256=sha256_file(reference_path),
            )
        )
    return tuple(rows)


def build_components(
    sessions: Sequence[CandidateSession],
) -> tuple[CandidateComponent, ...]:
    by_id = {session.session_id: session for session in sessions}
    if len(by_id) != len(sessions) or not sessions:
        raise AliMeetingTrainSelectionError(
            "AliMeeting candidate session identities are invalid"
        )
    components = _Components(by_id)
    sessions_by_speaker: dict[str, list[str]] = defaultdict(list)
    for session in sessions:
        for speaker_id in session.speaker_ids:
            sessions_by_speaker[speaker_id].append(session.session_id)
    for connected_sessions in sessions_by_speaker.values():
        for session_id in connected_sessions[1:]:
            components.union(connected_sessions[0], session_id)
    sessions_by_root: dict[str, list[CandidateSession]] = defaultdict(list)
    for session_id, session in by_id.items():
        sessions_by_root[components.find(session_id)].append(session)
    rows = []
    for members in sorted(
        (
            tuple(sorted(values, key=lambda value: value.session_id))
            for values in sessions_by_root.values()
        ),
        key=lambda values: values[0].session_id,
    ):
        session_ids = tuple(value.session_id for value in members)
        shared_speakers = tuple(
            sorted(
                speaker_id
                for speaker_id in {
                    speaker
                    for member in members
                    for speaker in member.speaker_ids
                }
                if sum(
                    speaker_id in member.speaker_ids for member in members
                )
                > 1
            )
        )
        rows.append(
            CandidateComponent(
                component_id=f"component-{canonical_sha256(list(session_ids))}",
                session_ids=session_ids,
                scored_samples=sum(value.scored_samples for value in members),
                participant_buckets=tuple(
                    sorted({value.participant_count for value in members})
                ),
                room_ids=tuple(sorted({value.room_id for value in members})),
                recording_group_ids=tuple(
                    sorted({value.recording_group_id for value in members})
                ),
                speaker_ids=tuple(
                    sorted(
                        {
                            speaker_id
                            for value in members
                            for speaker_id in value.speaker_ids
                        }
                    )
                ),
                shared_speaker_ids=shared_speakers,
                selection_eligible=all(
                    member.selection_eligible for member in members
                ),
                excluded_session_ids=tuple(
                    member.session_id
                    for member in members
                    if not member.selection_eligible
                ),
            )
        )
    return tuple(rows)


def _coverage_axes(
    components: Sequence[CandidateComponent],
) -> tuple[tuple[int, ...], tuple[tuple[str, str], ...]]:
    buckets = tuple(PARTICIPANT_BUCKETS)
    identities = tuple(
        [
            *(('room_ids', value) for value in sorted(
                {item for component in components for item in component.room_ids}
            )),
            *(('recording_group_ids', value) for value in sorted(
                {
                    item
                    for component in components
                    for item in component.recording_group_ids
                }
            )),
        ]
    )
    return buckets, identities


def _base_constraints(
    components: Sequence[CandidateComponent],
    *,
    include_distance: bool,
) -> tuple[
    int,
    int,
    int,
    list[dict[int, float]],
    list[float],
    list[float],
    int,
]:
    buckets, identities = _coverage_axes(components)
    component_count = len(components)
    bucket_offset = component_count
    identity_offset = bucket_offset + len(buckets)
    distance_index = identity_offset + len(identities)
    scale = math.gcd(
        TARGET_SCORED_SAMPLES,
        MIN_SCORED_SAMPLES,
        MAX_SCORED_SAMPLES,
        *(component.scored_samples for component in components),
    )
    rows: list[dict[int, float]] = []
    lower: list[float] = []
    upper: list[float] = []

    def add(values: dict[int, float], low: float, high: float) -> None:
        rows.append(values)
        lower.append(low)
        upper.append(high)

    add(
        {
            index: component.meeting_count
            for index, component in enumerate(components)
        },
        MIN_MEETINGS,
        MAX_MEETINGS,
    )
    for index, component in enumerate(components):
        if not component.selection_eligible:
            add({index: 1}, 0, 0)
    duration_row = {
        index: component.scored_samples // scale
        for index, component in enumerate(components)
    }
    add(
        duration_row,
        MIN_SCORED_SAMPLES // scale,
        MAX_SCORED_SAMPLES // scale,
    )
    for offset, bucket in enumerate(buckets):
        covering = [
            index
            for index, component in enumerate(components)
            if bucket in component.participant_buckets
        ]
        variable = bucket_offset + offset
        add(
            {variable: 1, **{index: -1 for index in covering}},
            -np.inf,
            0,
        )
        add(
            {variable: -len(covering), **{index: 1 for index in covering}},
            -np.inf,
            0,
        )
    for offset, (field, value) in enumerate(identities):
        covering = [
            index
            for index, component in enumerate(components)
            if value in getattr(component, field)
        ]
        variable = identity_offset + offset
        add(
            {variable: 1, **{index: -1 for index in covering}},
            -np.inf,
            0,
        )
        add(
            {variable: -len(covering), **{index: 1 for index in covering}},
            -np.inf,
            0,
        )
    if include_distance:
        add(
            {**duration_row, distance_index: -1},
            -np.inf,
            TARGET_SCORED_SAMPLES // scale,
        )
        add(
            {
                **{index: -value for index, value in duration_row.items()},
                distance_index: -1,
            },
            -np.inf,
            -(TARGET_SCORED_SAMPLES // scale),
        )
    return (
        bucket_offset,
        identity_offset,
        distance_index,
        rows,
        lower,
        upper,
        scale,
    )


def _solve(
    variable_count: int,
    rows: Sequence[Mapping[int, float]],
    lower: Sequence[float],
    upper: Sequence[float],
    objective: np.ndarray,
    *,
    unbounded_last: bool = False,
) -> np.ndarray | None:
    matrix = lil_matrix((len(rows), variable_count), dtype=float)
    for row_index, values in enumerate(rows):
        for column, value in values.items():
            matrix[row_index, column] = value
    result = milp(
        objective,
        integrality=np.ones(variable_count),
        bounds=Bounds(
            np.zeros(variable_count),
            np.array([1] * (variable_count - 1) + [np.inf])
            if variable_count and unbounded_last
            else np.ones(variable_count),
        ),
        constraints=LinearConstraint(
            matrix.tocsr(),
            np.array(lower),
            np.array(upper),
        ),
        options={"mip_rel_gap": 0.0, "presolve": True},
    )
    if result.x is None:
        if result.status == 2:
            return None
        raise AliMeetingTrainSelectionError(
            f"AliMeeting Train optimizer failed: {result.message}"
        )
    if not result.success:
        raise AliMeetingTrainSelectionError(
            f"AliMeeting Train optimizer did not prove optimality: {result.message}"
        )
    return result.x


def _selected_indices(
    result: np.ndarray,
    component_count: int,
) -> tuple[int, ...]:
    selected = tuple(
        index
        for index, value in enumerate(result[:component_count])
        if value > 0.5
    )
    if any(
        abs(value - round(value)) > 1e-7
        for value in result[:component_count]
    ):
        raise AliMeetingTrainSelectionError(
            "AliMeeting Train optimizer returned a fractional component"
        )
    return selected


def _selection_hash(component_ids: Sequence[str]) -> str:
    payload = SELECTION_SALT + "\n" + "\n".join(sorted(component_ids)) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_components(
    components: Sequence[CandidateComponent],
) -> SelectionResult:
    if not components or len({value.component_id for value in components}) != len(
        components
    ):
        raise AliMeetingTrainSelectionError(
            "AliMeeting component inventory is invalid"
        )
    (
        bucket_offset,
        identity_offset,
        distance_index,
        rows,
        lower,
        upper,
        scale,
    ) = _base_constraints(components, include_distance=True)
    component_count = len(components)
    bucket_count = len(PARTICIPANT_BUCKETS)
    identity_count = distance_index - identity_offset
    variable_count = distance_index + 1
    first_objective = np.zeros(variable_count)
    first_objective[bucket_offset:identity_offset] = -1
    first = _solve(
        variable_count,
        rows,
        lower,
        upper,
        first_objective,
        unbounded_last=True,
    )
    if first is None:
        raise AliMeetingTrainSelectionError(
            "no component-safe AliMeeting Train tranche is feasible"
        )
    bucket_coverage = int(
        round(sum(first[bucket_offset:identity_offset]))
    )
    rows.append(
        {
            bucket_offset + index: 1
            for index in range(bucket_count)
        }
    )
    lower.append(bucket_coverage)
    upper.append(bucket_coverage)
    second_objective = np.zeros(variable_count)
    second_objective[identity_offset:distance_index] = -1
    second = _solve(
        variable_count,
        rows,
        lower,
        upper,
        second_objective,
        unbounded_last=True,
    )
    if second is None:
        raise AliMeetingTrainSelectionError(
            "AliMeeting identity-coverage optimum is unavailable"
        )
    metadata_identity_coverage = int(
        round(sum(second[identity_offset:distance_index]))
    )
    rows.append(
        {
            identity_offset + index: 1
            for index in range(identity_count)
        }
    )
    lower.append(metadata_identity_coverage)
    upper.append(metadata_identity_coverage)
    third_objective = np.zeros(variable_count)
    third_objective[distance_index] = 1
    third = _solve(
        variable_count,
        rows,
        lower,
        upper,
        third_objective,
        unbounded_last=True,
    )
    if third is None:
        raise AliMeetingTrainSelectionError(
            "AliMeeting duration-distance optimum is unavailable"
        )
    third_indices = _selected_indices(third, component_count)
    scored_samples = sum(
        components[index].scored_samples for index in third_indices
    )
    target_distance_samples = abs(
        scored_samples - TARGET_SCORED_SAMPLES
    )
    if int(round(third[distance_index])) * scale != target_distance_samples:
        raise AliMeetingTrainSelectionError(
            "AliMeeting duration optimum failed exact validation"
        )
    (
        enumeration_bucket_offset,
        enumeration_identity_offset,
        enumeration_distance_index,
        enumeration_rows,
        enumeration_lower,
        enumeration_upper,
        enumeration_scale,
    ) = _base_constraints(components, include_distance=False)
    if enumeration_scale != scale:
        raise AliMeetingTrainSelectionError(
            "AliMeeting optimizer scale is inconsistent"
        )
    enumeration_rows.extend(
        [
            {
                enumeration_bucket_offset + index: 1
                for index in range(bucket_count)
            },
            {
                enumeration_identity_offset + index: 1
                for index in range(identity_count)
            },
        ]
    )
    enumeration_lower.extend(
        [bucket_coverage, metadata_identity_coverage]
    )
    enumeration_upper.extend(
        [bucket_coverage, metadata_identity_coverage]
    )
    enumeration_variable_count = enumeration_distance_index
    objective = np.zeros(enumeration_variable_count)
    component_ids = tuple(
        sorted(components[index].component_id for index in third_indices)
    )
    for total_samples in sorted(
        {
            TARGET_SCORED_SAMPLES - target_distance_samples,
            TARGET_SCORED_SAMPLES + target_distance_samples,
        }
    ):
        if not MIN_SCORED_SAMPLES <= total_samples <= MAX_SCORED_SAMPLES:
            continue
        total_row = {
            index: component.scored_samples // scale
            for index, component in enumerate(components)
        }
        candidate_rows = [*enumeration_rows, total_row]
        candidate_lower = [
            *enumeration_lower,
            total_samples // scale,
        ]
        candidate_upper = [
            *enumeration_upper,
            total_samples // scale,
        ]
        if total_samples == scored_samples:
            selected_set = set(third_indices)
            candidate_rows.append(
                {
                    index: 1 if index in selected_set else -1
                    for index in range(component_count)
                }
            )
            candidate_lower.append(-np.inf)
            candidate_upper.append(len(third_indices) - 1)
        alternative = _solve(
            enumeration_variable_count,
            candidate_rows,
            candidate_lower,
            candidate_upper,
            objective,
        )
        if alternative is not None:
            _selected_indices(alternative, component_count)
            raise AliMeetingTrainSelectionError(
                "multiple AliMeeting duration-optimal candidates require the "
                "salted SHA-256 tie-break"
            )
    return SelectionResult(
        participant_bucket_coverage=bucket_coverage,
        metadata_identity_coverage=metadata_identity_coverage,
        scored_samples=sum(
            component.scored_samples
            for component in components
            if component.component_id in component_ids
        ),
        target_distance_samples=target_distance_samples,
        optimal_candidate_count=1,
        component_ids=component_ids,
        selection_hash=_selection_hash(component_ids),
    )


def build_selection_receipt(
    archive_path: Path,
    room_workbook_path: Path,
    textgrid_root: Path,
    reference_root: Path,
) -> dict[str, Any]:
    _regular_file_identity(
        room_workbook_path,
        ROOM_WORKBOOK_SIZE_BYTES,
        ROOM_WORKBOOK_SHA256,
    )
    waveform_members, textgrid_members = _scan_archive(archive_path)
    sessions = _candidate_sessions(
        waveform_members,
        textgrid_members,
        textgrid_root,
        reference_root,
    )
    components = build_components(sessions)
    selection = select_components(components)
    selected_component_ids = set(selection.component_ids)
    selected_components = tuple(
        component
        for component in components
        if component.component_id in selected_component_ids
    )
    selected_session_ids = tuple(
        sorted(
            session_id
            for component in selected_components
            for session_id in component.session_ids
        )
    )
    session_rows = [session.to_dict() for session in sessions]
    component_rows = [component.to_dict() for component in components]
    checkout = open_reference_checkout(reference_root)
    room_ids = sorted({session.room_id for session in sessions})
    recording_groups = sorted(
        {session.recording_group_id for session in sessions}
    )
    return {
        "schema_version": 1,
        "artifact_role": "alimeeting_train_component_selection_receipt",
        "authority_ref": AUTHORITY_REF,
        "authority_pin": AUTHORITY_PIN,
        "selection_salt": SELECTION_SALT,
        "selection_hash_encoding": (
            "utf8(salt + LF + LF_join(sorted_component_ids) + LF)"
        ),
        "source_artifacts": {
            "openslr_resource_url": OPENSLR_RESOURCE_URL,
            "train_archive": {
                "url": TRAIN_ARCHIVE_URL,
                "size_bytes": TRAIN_ARCHIVE_SIZE_BYTES,
                "sha256": TRAIN_ARCHIVE_SHA256,
            },
            "room_workbook": {
                "url": ROOM_WORKBOOK_URL,
                "size_bytes": ROOM_WORKBOOK_SIZE_BYTES,
                "sha256": ROOM_WORKBOOK_SHA256,
                "selection_identity_use": (
                    "binds the official room-configuration artifact; discrete "
                    "room identity is the released session R field"
                ),
            },
            "forced_alignment": dict(checkout.provenance),
        },
        "published_inventory_discrepancy": {
            "openslr_catalog_session_count": CATALOG_SESSION_COUNT,
            "openslr_catalog_scored_hours": CATALOG_SCORED_HOURS,
            "archive_session_count": ARCHIVE_SESSION_COUNT,
            "pinned_reference_session_count": len(sessions),
            "policy": (
                "use the exact shared 209-session intersection of the official "
                "far-field archive and pinned reference; do not invent or infer "
                "three absent sessions"
            ),
        },
        "component_policy": {
            "known_speaker_identity": (
                "connect every session sharing an official SPK identity"
            ),
            "explicit_session_series": (
                "no separate series field is present in the released far-field "
                "session, waveform, or TextGrid identities; room and recording-"
                "group identities are diversity axes and are not inferred as "
                "session-series links"
            ),
            "room_identity": "session_id R field",
            "meeting_identity": "session_id M field",
            "recording_group_identity": "waveform filename MS field",
            "scored_exposure": (
                "intersection of the TextGrid timeline and validated waveform "
                "duration"
            ),
            "textgrid_tail_rule": {
                "maximum_excess_samples": (
                    ALIMEETING_TEXTGRID_ONLY_TAIL_MAX_EXCESS_SAMPLES
                ),
                "action": (
                    "clip only when the forced alignment ends within the "
                    "waveform; otherwise exclude the whole component"
                ),
            },
            "selection_eligibility": (
                "exclude every whole component containing a session whose "
                "forced alignment extends beyond the waveform or whose "
                "annotation-only tail exceeds the documented rule"
            ),
        },
        "selection_policy": {
            "meeting_range": [MIN_MEETINGS, MAX_MEETINGS],
            "scored_sample_range": [
                MIN_SCORED_SAMPLES,
                MAX_SCORED_SAMPLES,
            ],
            "target_scored_samples": TARGET_SCORED_SAMPLES,
            "participant_count_buckets": list(PARTICIPANT_BUCKETS),
            "metadata_identity_axes": ["room", "recording_group"],
            "lexicographic_objectives": [
                "maximize_participant_bucket_coverage",
                "maximize_distinct_typed_room_and_recording_group_identities",
                "minimize_absolute_distance_from_target_scored_samples",
                "minimum_salted_sha256_of_sorted_component_ids",
            ],
            "tie_break_status": (
                "not_exercised_unique_after_duration_objective"
            ),
            "selection_hash_use": (
                "integrity receipt for the unique selected component list"
            ),
            "optimizer": "scipy.optimize.milp exact-integer branch-and-bound",
            "annotation_and_model_score_blind": True,
        },
        "candidate_summary": {
            "session_count": len(sessions),
            "component_count": len(components),
            "singleton_component_count": sum(
                component.meeting_count == 1 for component in components
            ),
            "multi_session_component_count": sum(
                component.meeting_count > 1 for component in components
            ),
            "scored_samples": sum(session.scored_samples for session in sessions),
            "participant_count_distribution": dict(
                sorted(Counter(session.participant_count for session in sessions).items())
            ),
            "room_ids": room_ids,
            "recording_group_ids": recording_groups,
            "annotation_tail_session_count": sum(
                session.annotation_tail_excess_samples > 0
                for session in sessions
            ),
            "maximum_annotation_tail_excess_samples": max(
                session.annotation_tail_excess_samples
                for session in sessions
            ),
            "selection_eligible_session_count": sum(
                session.selection_eligible for session in sessions
            ),
            "selection_excluded_session_count": sum(
                not session.selection_eligible for session in sessions
            ),
            "selection_exclusion_reason_counts": dict(
                sorted(
                    Counter(
                        reason
                        for session in sessions
                        for reason in session.selection_exclusion_reasons
                    ).items()
                )
            ),
            "candidate_sessions_sha256": canonical_sha256(session_rows),
            "candidate_components_sha256": canonical_sha256(component_rows),
        },
        "candidate_sessions": session_rows,
        "candidate_components": component_rows,
        "objective_values": {
            "participant_bucket_coverage": (
                selection.participant_bucket_coverage
            ),
            "metadata_identity_coverage": selection.metadata_identity_coverage,
            "scored_samples": selection.scored_samples,
            "scored_hours": round(
                selection.scored_samples / SAMPLE_RATE_HZ / 3600,
                9,
            ),
            "target_distance_samples": selection.target_distance_samples,
            "target_distance_hours": round(
                selection.target_distance_samples / SAMPLE_RATE_HZ / 3600,
                9,
            ),
            "optimal_candidate_count": selection.optimal_candidate_count,
        },
        "selected_component_ids": list(selection.component_ids),
        "selected_session_ids": list(selected_session_ids),
        "selected_meeting_count": len(selected_session_ids),
        "selection_hash": selection.selection_hash,
        "selection_model_inputs": {
            "participant_count": "TextGrid tier count",
            "scored_samples": (
                "TextGrid timeline intersected with validated waveform duration"
            ),
            "speaker_identity": "TextGrid tier SPK identity",
            "room_identity": "session filename R field",
            "recording_group_identity": "waveform filename MS field",
            "psem_topology_counts_used": False,
            "audio_features_used": False,
            "vad_or_diarization_predictions_used": False,
            "model_scores_used": False,
            "issue_76_outcomes_used": False,
        },
        "reference_identity": {
            "repository": REFERENCE_REPOSITORY,
            "commit": REFERENCE_COMMIT,
        },
    }


def _exact_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AliMeetingTrainSelectionError(f"{field} must be an integer")
    return value


def _string_tuple(value: Any, field: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or any(
            not isinstance(item, str)
            or not item
            or item.strip() != item
            for item in value
        )
        or len(value) != len(set(value))
    ):
        raise AliMeetingTrainSelectionError(f"{field} is invalid")
    return tuple(value)


def validate_selection_receipt(path: Path) -> dict[str, Any]:
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AliMeetingTrainSelectionError(
            f"AliMeeting Train selection receipt is invalid: {path}"
        ) from exc
    if not isinstance(receipt, dict):
        raise AliMeetingTrainSelectionError(
            "AliMeeting Train selection receipt must be an object"
        )
    source_artifacts = receipt.get("source_artifacts")
    archive = source_artifacts.get("train_archive") if isinstance(
        source_artifacts,
        dict,
    ) else None
    workbook = source_artifacts.get("room_workbook") if isinstance(
        source_artifacts,
        dict,
    ) else None
    discrepancy = receipt.get("published_inventory_discrepancy")
    component_policy = receipt.get("component_policy")
    tail_rule = component_policy.get("textgrid_tail_rule") if isinstance(
        component_policy,
        dict,
    ) else None
    if (
        receipt.get("schema_version") != 1
        or receipt.get("artifact_role")
        != "alimeeting_train_component_selection_receipt"
        or receipt.get("authority_ref") != AUTHORITY_REF
        or receipt.get("authority_pin") != AUTHORITY_PIN
        or receipt.get("selection_salt") != SELECTION_SALT
        or receipt.get("selection_hash_encoding")
        != "utf8(salt + LF + LF_join(sorted_component_ids) + LF)"
        or not isinstance(archive, dict)
        or archive.get("url") != TRAIN_ARCHIVE_URL
        or archive.get("size_bytes") != TRAIN_ARCHIVE_SIZE_BYTES
        or archive.get("sha256") != TRAIN_ARCHIVE_SHA256
        or not isinstance(workbook, dict)
        or workbook.get("url") != ROOM_WORKBOOK_URL
        or workbook.get("size_bytes") != ROOM_WORKBOOK_SIZE_BYTES
        or workbook.get("sha256") != ROOM_WORKBOOK_SHA256
        or not isinstance(discrepancy, dict)
        or discrepancy.get("archive_session_count") != ARCHIVE_SESSION_COUNT
        or discrepancy.get("pinned_reference_session_count")
        != ARCHIVE_SESSION_COUNT
        or not isinstance(tail_rule, dict)
        or tail_rule.get("maximum_excess_samples")
        != ALIMEETING_TEXTGRID_ONLY_TAIL_MAX_EXCESS_SAMPLES
        or tail_rule.get("action")
        != (
            "clip only when the forced alignment ends within the waveform; "
            "otherwise exclude the whole component"
        )
        or receipt.get("reference_identity")
        != {"repository": REFERENCE_REPOSITORY, "commit": REFERENCE_COMMIT}
    ):
        raise AliMeetingTrainSelectionError(
            "AliMeeting Train selection authority or source identity changed"
        )
    raw_sessions = receipt.get("candidate_sessions")
    if not isinstance(raw_sessions, list) or len(raw_sessions) != ARCHIVE_SESSION_COUNT:
        raise AliMeetingTrainSelectionError(
            "AliMeeting candidate session census is incomplete"
        )
    sessions = []
    for row in raw_sessions:
        if not isinstance(row, dict):
            raise AliMeetingTrainSelectionError(
                "AliMeeting candidate session row is invalid"
            )
        session_id = row.get("session_id")
        match = SESSION_PATTERN.fullmatch(session_id) if isinstance(
            session_id,
            str,
        ) else None
        speaker_ids = _string_tuple(row.get("speaker_ids"), "speaker_ids")
        participant_count = _exact_integer(
            row.get("participant_count"),
            "participant_count",
        )
        waveform_ref = row.get("waveform_archive_member_ref")
        textgrid_ref = row.get("textgrid_archive_member_ref")
        textgrid_match = TEXTGRID_MEMBER_PATTERN.fullmatch(
            textgrid_ref
        ) if isinstance(textgrid_ref, str) else None
        textgrid_sha256 = row.get("textgrid_sha256")
        reference_ref = row.get("reference_ref")
        reference_sha256 = row.get("reference_sha256")
        waveform_duration_samples = _exact_integer(
            row.get("waveform_duration_samples"),
            "waveform_duration_samples",
        )
        textgrid_timeline_samples = _exact_integer(
            row.get("textgrid_timeline_samples"),
            "textgrid_timeline_samples",
        )
        scored_samples, annotation_tail_excess_samples = (
            _accepted_scored_samples(
                textgrid_timeline_samples,
                waveform_duration_samples,
            )
        )
        reference_end_sample = _exact_integer(
            row.get("reference_end_sample"),
            "reference_end_sample",
        )
        selection_exclusion_reasons = _selection_exclusion_reasons(
            textgrid_timeline_samples=textgrid_timeline_samples,
            waveform_duration_samples=waveform_duration_samples,
            reference_end_sample=reference_end_sample,
        )
        if (
            match is None
            or row.get("room_id") != match.group("room")
            or row.get("meeting_id") != match.group("meeting")
            or participant_count != len(speaker_ids)
            or participant_count not in PARTICIPANT_BUCKETS
            or not isinstance(waveform_ref, str)
            or _recording_group(waveform_ref, session_id)
            != row.get("recording_group_id")
            or textgrid_match is None
            or textgrid_match.group("session") != session_id
            or not isinstance(textgrid_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", textgrid_sha256) is None
            or not isinstance(reference_ref, str)
            or reference_ref
            != f"AliMeeting/Train_Ali_far/{session_id}.rttm"
            or not isinstance(reference_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", reference_sha256) is None
            or row.get("scored_samples") != scored_samples
            or row.get("annotation_tail_excess_samples")
            != annotation_tail_excess_samples
            or row.get("reference_tail_excess_samples")
            != max(0, reference_end_sample - waveform_duration_samples)
            or row.get("selection_eligible")
            is not (not selection_exclusion_reasons)
            or row.get("selection_exclusion_reasons")
            != list(selection_exclusion_reasons)
        ):
            raise AliMeetingTrainSelectionError(
                f"AliMeeting candidate session identity is invalid: {session_id}"
            )
        sessions.append(
            CandidateSession(
                session_id=session_id,
                room_id=match.group("room"),
                meeting_id=match.group("meeting"),
                recording_group_id=row["recording_group_id"],
                scored_samples=scored_samples,
                textgrid_timeline_samples=textgrid_timeline_samples,
                annotation_tail_excess_samples=(
                    annotation_tail_excess_samples
                ),
                reference_end_sample=reference_end_sample,
                selection_eligible=not selection_exclusion_reasons,
                selection_exclusion_reasons=selection_exclusion_reasons,
                participant_count=participant_count,
                speaker_ids=speaker_ids,
                waveform_member=ArchiveMember(
                    ref=waveform_ref,
                    size_bytes=_exact_integer(
                        row.get("waveform_archive_member_size_bytes"),
                        "waveform_archive_member_size_bytes",
                    ),
                    sha256=None,
                    duration_samples=waveform_duration_samples,
                ),
                textgrid_member=ArchiveMember(
                    ref=textgrid_ref,
                    size_bytes=_exact_integer(
                        row.get("textgrid_archive_member_size_bytes"),
                        "textgrid_archive_member_size_bytes",
                    ),
                    sha256=textgrid_sha256,
                ),
                reference_ref=reference_ref,
                reference_sha256=reference_sha256,
            )
        )
    expected_components = build_components(sessions)
    expected_session_rows = [session.to_dict() for session in sessions]
    expected_component_rows = [
        component.to_dict() for component in expected_components
    ]
    summary = receipt.get("candidate_summary")
    if (
        raw_sessions != expected_session_rows
        or receipt.get("candidate_components") != expected_component_rows
        or not isinstance(summary, dict)
        or summary.get("session_count") != len(sessions)
        or summary.get("component_count") != len(expected_components)
        or summary.get("annotation_tail_session_count")
        != sum(
            session.annotation_tail_excess_samples > 0
            for session in sessions
        )
        or summary.get("maximum_annotation_tail_excess_samples")
        != max(
            session.annotation_tail_excess_samples
            for session in sessions
        )
        or summary.get("selection_eligible_session_count")
        != sum(session.selection_eligible for session in sessions)
        or summary.get("selection_excluded_session_count")
        != sum(not session.selection_eligible for session in sessions)
        or summary.get("selection_exclusion_reason_counts")
        != dict(
            sorted(
                Counter(
                    reason
                    for session in sessions
                    for reason in session.selection_exclusion_reasons
                ).items()
            )
        )
        or summary.get("candidate_sessions_sha256")
        != canonical_sha256(expected_session_rows)
        or summary.get("candidate_components_sha256")
        != canonical_sha256(expected_component_rows)
    ):
        raise AliMeetingTrainSelectionError(
            "AliMeeting candidate census does not reproduce"
        )
    selection = select_components(expected_components)
    selected_component_ids = list(selection.component_ids)
    selected_session_ids = sorted(
        session_id
        for component in expected_components
        if component.component_id in selection.component_ids
        for session_id in component.session_ids
    )
    objective_values = receipt.get("objective_values")
    expected_objectives = {
        "participant_bucket_coverage": selection.participant_bucket_coverage,
        "metadata_identity_coverage": selection.metadata_identity_coverage,
        "scored_samples": selection.scored_samples,
        "scored_hours": round(
            selection.scored_samples / SAMPLE_RATE_HZ / 3600,
            9,
        ),
        "target_distance_samples": selection.target_distance_samples,
        "target_distance_hours": round(
            selection.target_distance_samples / SAMPLE_RATE_HZ / 3600,
            9,
        ),
        "optimal_candidate_count": selection.optimal_candidate_count,
    }
    if (
        objective_values != expected_objectives
        or receipt.get("selected_component_ids") != selected_component_ids
        or receipt.get("selected_session_ids") != selected_session_ids
        or receipt.get("selected_meeting_count") != len(selected_session_ids)
        or receipt.get("selection_hash") != selection.selection_hash
    ):
        raise AliMeetingTrainSelectionError(
            "AliMeeting selection receipt does not reproduce"
        )
    if sha256_file(path) != SELECTION_RECEIPT_SHA256:
        raise AliMeetingTrainSelectionError(
            "AliMeeting Train selection receipt file identity changed"
        )
    return receipt


def write_selection_receipt(
    archive_path: Path,
    room_workbook_path: Path,
    textgrid_root: Path,
    reference_root: Path,
    output_path: Path,
) -> None:
    receipt = build_selection_receipt(
        archive_path,
        room_workbook_path,
        textgrid_root,
        reference_root,
    )
    output_path.write_text(
        json.dumps(
            receipt,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--room-workbook", type=Path, required=True)
    parser.add_argument("--textgrid-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_selection_receipt(
        args.archive.resolve(),
        args.room_workbook.resolve(),
        args.textgrid_root.resolve(),
        args.reference_root.resolve(),
        args.output.resolve(),
    )


if __name__ == "__main__":
    main()
