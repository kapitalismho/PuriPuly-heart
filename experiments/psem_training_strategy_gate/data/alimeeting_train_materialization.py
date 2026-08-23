from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
import wave
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from experiments.psem_training_strategy_gate.data.alimeeting_train_selection import (
    ARCHIVE_ROOTS,
    ARCHIVE_SESSION_COUNT,
    SAMPLE_RATE_HZ,
    TEXTGRID_MEMBER_PATTERN,
    TRAIN_ARCHIVE_SHA256,
    TRAIN_ARCHIVE_SIZE_BYTES,
    WAVEFORM_MEMBER_PATTERN,
    validate_selection_receipt,
)
from experiments.psem_training_strategy_gate.data.forced_alignment_reference import (
    sha256_file,
)

MATERIALIZATION_RECEIPT_SHA256 = (
    "1490458de43d8cec5d2200009cf35f00ba73f33555a6744e9700818809818933"
)


class AliMeetingTrainMaterializationError(RuntimeError):
    pass


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


class _MemberReader:
    def __init__(self, handle: BinaryIO) -> None:
        self._handle = handle
        self._digest = hashlib.sha256()
        self.size_bytes = 0

    def copy_to(self, output: BinaryIO) -> None:
        while chunk := self._handle.read(1 << 20):
            output.write(chunk)
            self._digest.update(chunk)
            self.size_bytes += len(chunk)

    @property
    def sha256(self) -> str:
        return self._digest.hexdigest()


def _temporary_path(parent: Path, suffix: str) -> Path:
    handle = tempfile.NamedTemporaryFile(
        dir=parent,
        prefix=".alimeeting-train-",
        suffix=suffix,
        delete=False,
    )
    path = Path(handle.name)
    handle.close()
    return path


def _install_identical_or_new(
    temporary_path: Path,
    target_path: Path,
    *,
    expected_size_bytes: int,
    expected_sha256: str,
) -> str:
    if (
        temporary_path.stat().st_size != expected_size_bytes
        or sha256_file(temporary_path) != expected_sha256
    ):
        raise AliMeetingTrainMaterializationError(
            f"materialized source identity changed: {target_path.name}"
        )
    if target_path.exists():
        if (
            not target_path.is_file()
            or target_path.is_symlink()
            or target_path.stat().st_size != expected_size_bytes
            or sha256_file(target_path) != expected_sha256
        ):
            raise AliMeetingTrainMaterializationError(
                f"existing target conflicts with source bytes: {target_path}"
            )
        temporary_path.unlink()
        return "existing"
    os.replace(temporary_path, target_path)
    return "materialized"


def _stage_raw_member(
    bundle: tarfile.TarFile,
    member: tarfile.TarInfo,
    target_parent: Path,
) -> tuple[Path, str]:
    extracted = bundle.extractfile(member)
    if extracted is None:
        raise AliMeetingTrainMaterializationError(
            f"selected waveform archive member is unreadable: {member.name}"
        )
    target_parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _temporary_path(target_parent, ".wav")
    try:
        reader = _MemberReader(extracted)
        with temporary_path.open("wb") as output:
            reader.copy_to(output)
        if reader.size_bytes != member.size:
            raise AliMeetingTrainMaterializationError(
                f"selected waveform archive member is incomplete: {member.name}"
            )
        return temporary_path, reader.sha256
    except BaseException:
        if temporary_path.exists():
            temporary_path.unlink()
        raise


def _materialize_channel_zero(
    raw_path: Path,
    target_path: Path,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _temporary_path(target_path.parent, ".wav")
    try:
        try:
            with wave.open(str(raw_path), "rb") as source:
                source_identity = {
                    "channels": source.getnchannels(),
                    "sample_width_bytes": source.getsampwidth(),
                    "sample_rate_hz": source.getframerate(),
                    "duration_samples": source.getnframes(),
                    "compression": source.getcomptype(),
                }
                if (
                    source_identity["channels"] != 8
                    or source_identity["sample_width_bytes"] != 2
                    or source_identity["sample_rate_hz"] != SAMPLE_RATE_HZ
                    or source_identity["duration_samples"] <= 0
                    or source_identity["compression"] != "NONE"
                ):
                    raise AliMeetingTrainMaterializationError(
                        f"selected AliMeeting source is not 8-channel PCM16: {raw_path}"
                    )
                observed_frames = 0
                with wave.open(str(temporary_path), "wb") as output:
                    output.setnchannels(1)
                    output.setsampwidth(2)
                    output.setframerate(SAMPLE_RATE_HZ)
                    while payload := source.readframes(1 << 16):
                        samples = np.frombuffer(payload, dtype="<i2")
                        if len(samples) % 8:
                            raise AliMeetingTrainMaterializationError(
                                f"selected waveform frame payload is truncated: {raw_path}"
                            )
                        frames = samples.reshape((-1, 8))
                        output.writeframesraw(frames[:, 0].tobytes())
                        observed_frames += len(frames)
                if observed_frames != source_identity["duration_samples"]:
                    raise AliMeetingTrainMaterializationError(
                        f"selected waveform frame count changed: {raw_path}"
                    )
        except (OSError, EOFError, wave.Error, ValueError) as exc:
            raise AliMeetingTrainMaterializationError(
                f"selected AliMeeting waveform is invalid: {raw_path}"
            ) from exc
        try:
            identity = _mono_wav_identity(temporary_path)
        except (OSError, EOFError, wave.Error) as exc:
            raise AliMeetingTrainMaterializationError(
                f"materialized AliMeeting channel zero is invalid: {target_path}"
            ) from exc
        status = _install_identical_or_new(
            temporary_path,
            target_path,
            expected_size_bytes=identity["waveform_size_bytes"],
            expected_sha256=identity["waveform_sha256"],
        )
        return identity, status, source_identity
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _mono_wav_identity(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width_bytes = handle.getsampwidth()
        sample_rate_hz = handle.getframerate()
        duration_samples = handle.getnframes()
        compression = handle.getcomptype()
        payload_size_bytes = 0
        while payload := handle.readframes(1 << 19):
            payload_size_bytes += len(payload)
    if (
        channels != 1
        or sample_width_bytes != 2
        or sample_rate_hz != SAMPLE_RATE_HZ
        or duration_samples <= 0
        or compression != "NONE"
        or payload_size_bytes != duration_samples * sample_width_bytes
    ):
        raise AliMeetingTrainMaterializationError(
            f"materialized waveform is not complete 16 kHz mono PCM16: {path}"
        )
    return {
        "waveform_sha256": sha256_file(path),
        "waveform_size_bytes": path.stat().st_size,
        "sample_rate_hz": sample_rate_hz,
        "channels": channels,
        "sample_width_bytes": sample_width_bytes,
        "duration_samples": duration_samples,
    }


def _copy_textgrid(
    source_path: Path,
    target_path: Path,
    *,
    expected_size_bytes: int,
    expected_sha256: str,
) -> str:
    if (
        not source_path.is_file()
        or source_path.stat().st_size != expected_size_bytes
        or sha256_file(source_path) != expected_sha256
    ):
        raise AliMeetingTrainMaterializationError(
            f"selected TextGrid source identity changed: {source_path.name}"
        )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _temporary_path(target_path.parent, ".TextGrid")
    try:
        shutil.copyfile(source_path, temporary_path)
        return _install_identical_or_new(
            temporary_path,
            target_path,
            expected_size_bytes=expected_size_bytes,
            expected_sha256=expected_sha256,
        )
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _materialize_validated_raw_members(
    archive_path: Path,
    corpus_root: Path,
    candidate_by_id: dict[str, dict[str, Any]],
    selected_rows: dict[str, dict[str, Any]],
) -> dict[str, str]:
    selected_ids = set(selected_rows)
    raw_sha256: dict[str, str] = {}
    staged_raw: dict[str, Path] = {}
    selected_textgrid_sha256: dict[str, str] = {}
    waveform_sessions: set[str] = set()
    textgrid_sessions: set[str] = set()
    directory_members: set[str] = set()
    member_count = 0
    raw_root = (
        corpus_root
        / "alimeeting"
        / "Train_Ali"
        / "Train_Ali_far"
        / "audio_dir"
    )
    try:
        try:
            if archive_path.resolve(strict=True).stat().st_size != TRAIN_ARCHIVE_SIZE_BYTES:
                raise AliMeetingTrainMaterializationError(
                    "AliMeeting Train archive size changed"
                )
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
                            raise AliMeetingTrainMaterializationError(
                                f"unsupported archive member: {member.name}"
                            )
                        waveform_match = WAVEFORM_MEMBER_PATTERN.fullmatch(member.name)
                        if waveform_match is not None:
                            session_id = waveform_match.group("session")
                            if session_id in waveform_sessions:
                                raise AliMeetingTrainMaterializationError(
                                    f"duplicate waveform member: {session_id}"
                                )
                            waveform_sessions.add(session_id)
                            if session_id not in selected_ids:
                                continue
                            expected_row = selected_rows[session_id]
                            if (
                                member.name
                                != expected_row["waveform_archive_member_ref"]
                                or member.size
                                != expected_row["waveform_archive_member_size_bytes"]
                            ):
                                raise AliMeetingTrainMaterializationError(
                                    f"selected waveform member identity changed: {session_id}"
                                )
                            staged_path, digest = _stage_raw_member(
                                bundle,
                                member,
                                raw_root,
                            )
                            staged_raw[session_id] = staged_path
                            raw_sha256[session_id] = digest
                            continue
                        textgrid_match = TEXTGRID_MEMBER_PATTERN.fullmatch(member.name)
                        if textgrid_match is None:
                            raise AliMeetingTrainMaterializationError(
                                f"unexpected archive member: {member.name}"
                            )
                        session_id = textgrid_match.group("session")
                        if session_id in textgrid_sessions:
                            raise AliMeetingTrainMaterializationError(
                                f"duplicate TextGrid member: {session_id}"
                            )
                        textgrid_sessions.add(session_id)
                        if session_id not in selected_ids:
                            continue
                        expected_row = selected_rows[session_id]
                        if (
                            member.name
                            != expected_row["textgrid_archive_member_ref"]
                            or member.size
                            != expected_row["textgrid_archive_member_size_bytes"]
                        ):
                            raise AliMeetingTrainMaterializationError(
                                f"selected TextGrid member identity changed: {session_id}"
                            )
                        extracted = bundle.extractfile(member)
                        if extracted is None:
                            raise AliMeetingTrainMaterializationError(
                                f"selected TextGrid archive member is unreadable: {member.name}"
                            )
                        digest = hashlib.sha256()
                        observed_size = 0
                        while payload := extracted.read(1 << 20):
                            digest.update(payload)
                            observed_size += len(payload)
                        observed_sha256 = digest.hexdigest()
                        if (
                            observed_size != member.size
                            or observed_sha256 != expected_row["textgrid_sha256"]
                        ):
                            raise AliMeetingTrainMaterializationError(
                                f"selected TextGrid archive bytes changed: {session_id}"
                            )
                        selected_textgrid_sha256[session_id] = observed_sha256
                while digesting_reader.read(1 << 20):
                    pass
                archive_sha256 = digesting_reader.sha256
                observed_archive_size = digesting_reader.size_bytes
        except (OSError, EOFError, tarfile.TarError) as exc:
            raise AliMeetingTrainMaterializationError(
                "AliMeeting Train archive materialization failed"
            ) from exc
        expected_candidate_ids = set(candidate_by_id)
        if (
            observed_archive_size != TRAIN_ARCHIVE_SIZE_BYTES
            or archive_sha256 != TRAIN_ARCHIVE_SHA256
            or member_count != 2 * ARCHIVE_SESSION_COUNT + len(ARCHIVE_ROOTS)
            or directory_members != ARCHIVE_ROOTS
            or waveform_sessions != expected_candidate_ids
            or textgrid_sessions != expected_candidate_ids
            or set(raw_sha256) != selected_ids
            or set(selected_textgrid_sha256) != selected_ids
        ):
            raise AliMeetingTrainMaterializationError(
                "AliMeeting Train archive inventory changed during materialization"
            )
        for session_id in sorted(selected_ids):
            selected = selected_rows[session_id]
            raw_target = raw_root / Path(
                selected["waveform_archive_member_ref"]
            ).name
            _install_identical_or_new(
                staged_raw[session_id],
                raw_target,
                expected_size_bytes=selected[
                    "waveform_archive_member_size_bytes"
                ],
                expected_sha256=raw_sha256[session_id],
            )
        return raw_sha256
    finally:
        for staged_path in staged_raw.values():
            if staged_path.exists():
                staged_path.unlink()


def materialize_selected_train(
    archive_path: Path,
    selection_path: Path,
    extracted_textgrid_root: Path,
    corpus_root: Path,
) -> dict[str, Any]:
    selection = validate_selection_receipt(selection_path)
    selected_ids = set(selection["selected_session_ids"])
    candidate_by_id = {
        row["session_id"]: row for row in selection["candidate_sessions"]
    }
    selected_rows = {value: candidate_by_id[value] for value in selected_ids}
    if any(
        not row.get("selection_eligible")
        for row in selected_rows.values()
    ):
        raise AliMeetingTrainMaterializationError(
            "selection receipt contains an ineligible selected session"
        )
    raw_sha256 = _materialize_validated_raw_members(
        archive_path,
        corpus_root,
        candidate_by_id,
        selected_rows,
    )
    materialized_rows = []
    for session_id in sorted(selected_ids):
        selected = selected_rows[session_id]
        raw_name = Path(selected["waveform_archive_member_ref"]).name
        raw_path = (
            corpus_root
            / "alimeeting"
            / "Train_Ali"
            / "Train_Ali_far"
            / "audio_dir"
            / raw_name
        )
        ch0_path = corpus_root / "alimeeting" / "far_ch0" / f"{session_id}.wav"
        ch0_identity, _, raw_wave_identity = _materialize_channel_zero(
            raw_path,
            ch0_path,
        )
        if (
            raw_wave_identity["duration_samples"]
            != selected["waveform_duration_samples"]
            or ch0_identity["duration_samples"]
            != selected["waveform_duration_samples"]
            or selected["scored_samples"] > ch0_identity["duration_samples"]
        ):
            raise AliMeetingTrainMaterializationError(
                f"selected waveform duration changed: {session_id}"
            )
        source_textgrid = extracted_textgrid_root / f"{session_id}.TextGrid"
        target_textgrid = (
            corpus_root
            / "alimeeting"
            / "Train_Ali"
            / "Train_Ali_far"
            / "textgrid_dir"
            / f"{session_id}.TextGrid"
        )
        _copy_textgrid(
            source_textgrid,
            target_textgrid,
            expected_size_bytes=selected["textgrid_archive_member_size_bytes"],
            expected_sha256=selected["textgrid_sha256"],
        )
        materialized_rows.append(
            {
                "session_id": session_id,
                "source_archive_member_ref": selected[
                    "waveform_archive_member_ref"
                ],
                "source_archive_member_size_bytes": selected[
                    "waveform_archive_member_size_bytes"
                ],
                "source_archive_member_sha256": raw_sha256[session_id],
                "source_waveform_ref": raw_path.relative_to(corpus_root).as_posix(),
                "source_waveform_identity": raw_wave_identity,
                "audio_ref": ch0_path.relative_to(corpus_root).as_posix(),
                **ch0_identity,
                "annotation_ref": target_textgrid.relative_to(
                    corpus_root
                ).as_posix(),
                "annotation_size_bytes": selected[
                    "textgrid_archive_member_size_bytes"
                ],
                "annotation_file_sha256": selected["textgrid_sha256"],
                "annotation_coverage_start_sample": 0,
                "annotation_coverage_end_sample": selected["scored_samples"],
                "textgrid_timeline_samples": selected[
                    "textgrid_timeline_samples"
                ],
                "annotation_tail_excess_samples": selected[
                    "annotation_tail_excess_samples"
                ],
                "speaker_ids": selected["speaker_ids"],
                "room_id": selected["room_id"],
                "meeting_id": selected["meeting_id"],
                "recording_group_id": selected["recording_group_id"],
                "reference_ref": selected["reference_ref"],
                "reference_sha256": selected["reference_sha256"],
            }
        )
    return {
        "schema_version": 1,
        "artifact_role": "alimeeting_train_selected_audio_materialization",
        "selection_receipt_sha256": sha256_file(selection_path),
        "selection_hash": selection["selection_hash"],
        "archive_size_bytes": TRAIN_ARCHIVE_SIZE_BYTES,
        "archive_sha256": TRAIN_ARCHIVE_SHA256,
        "selected_session_count": len(materialized_rows),
        "selected_session_ids": sorted(selected_ids),
        "materialized_sessions": materialized_rows,
    }


def validate_materialization_receipt(
    path: Path,
    selection_path: Path,
    corpus_root: Path | None,
) -> dict[str, Any]:
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AliMeetingTrainMaterializationError(
            f"AliMeeting Train materialization receipt is invalid: {path}"
        ) from exc
    selection = validate_selection_receipt(selection_path)
    selected_ids = selection["selected_session_ids"]
    candidate_by_id = {
        row["session_id"]: row for row in selection["candidate_sessions"]
    }
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema_version") != 1
        or receipt.get("artifact_role")
        != "alimeeting_train_selected_audio_materialization"
        or receipt.get("selection_receipt_sha256")
        != sha256_file(selection_path)
        or receipt.get("selection_hash") != selection["selection_hash"]
        or receipt.get("archive_size_bytes") != TRAIN_ARCHIVE_SIZE_BYTES
        or receipt.get("archive_sha256") != TRAIN_ARCHIVE_SHA256
        or receipt.get("selected_session_count") != len(selected_ids)
        or receipt.get("selected_session_ids") != selected_ids
    ):
        raise AliMeetingTrainMaterializationError(
            "AliMeeting Train materialization authority changed"
        )
    rows = receipt.get("materialized_sessions")
    if (
        not isinstance(rows, list)
        or [row.get("session_id") for row in rows if isinstance(row, dict)]
        != selected_ids
    ):
        raise AliMeetingTrainMaterializationError(
            "AliMeeting Train materialization inventory changed"
        )
    expected_keys = {
        "session_id",
        "source_archive_member_ref",
        "source_archive_member_size_bytes",
        "source_archive_member_sha256",
        "source_waveform_ref",
        "source_waveform_identity",
        "audio_ref",
        "waveform_sha256",
        "waveform_size_bytes",
        "sample_rate_hz",
        "channels",
        "sample_width_bytes",
        "duration_samples",
        "annotation_ref",
        "annotation_size_bytes",
        "annotation_file_sha256",
        "annotation_coverage_start_sample",
        "annotation_coverage_end_sample",
        "textgrid_timeline_samples",
        "annotation_tail_excess_samples",
        "speaker_ids",
        "room_id",
        "meeting_id",
        "recording_group_id",
        "reference_ref",
        "reference_sha256",
    }
    root = corpus_root.resolve() if corpus_root is not None else None
    for row in rows:
        if not isinstance(row, dict) or set(row) != expected_keys:
            raise AliMeetingTrainMaterializationError(
                "AliMeeting Train materialization row shape changed"
            )
        session_id = row["session_id"]
        selected = candidate_by_id[session_id]
        raw_name = Path(selected["waveform_archive_member_ref"]).name
        raw_ref = (
            Path("alimeeting")
            / "Train_Ali"
            / "Train_Ali_far"
            / "audio_dir"
            / raw_name
        ).as_posix()
        audio_ref = (
            Path("alimeeting") / "far_ch0" / f"{session_id}.wav"
        ).as_posix()
        annotation_ref = (
            Path("alimeeting")
            / "Train_Ali"
            / "Train_Ali_far"
            / "textgrid_dir"
            / f"{session_id}.TextGrid"
        ).as_posix()
        source_identity = {
            "channels": 8,
            "sample_width_bytes": 2,
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "duration_samples": selected["waveform_duration_samples"],
            "compression": "NONE",
        }
        if (
            row["source_archive_member_ref"]
            != selected["waveform_archive_member_ref"]
            or row["source_archive_member_size_bytes"]
            != selected["waveform_archive_member_size_bytes"]
            or not isinstance(row["source_archive_member_sha256"], str)
            or re.fullmatch(
                r"[0-9a-f]{64}",
                row["source_archive_member_sha256"],
            )
            is None
            or row["source_waveform_ref"] != raw_ref
            or row["source_waveform_identity"] != source_identity
            or row["audio_ref"] != audio_ref
            or row["sample_rate_hz"] != SAMPLE_RATE_HZ
            or row["channels"] != 1
            or row["sample_width_bytes"] != 2
            or row["duration_samples"]
            != selected["waveform_duration_samples"]
            or not isinstance(row["waveform_size_bytes"], int)
            or isinstance(row["waveform_size_bytes"], bool)
            or not isinstance(row["waveform_sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", row["waveform_sha256"])
            is None
            or row["annotation_ref"] != annotation_ref
            or row["annotation_size_bytes"]
            != selected["textgrid_archive_member_size_bytes"]
            or row["annotation_file_sha256"] != selected["textgrid_sha256"]
            or row["annotation_coverage_start_sample"] != 0
            or row["annotation_coverage_end_sample"]
            != selected["scored_samples"]
            or row["textgrid_timeline_samples"]
            != selected["textgrid_timeline_samples"]
            or row["annotation_tail_excess_samples"]
            != selected["annotation_tail_excess_samples"]
            or row["speaker_ids"] != selected["speaker_ids"]
            or row["room_id"] != selected["room_id"]
            or row["meeting_id"] != selected["meeting_id"]
            or row["recording_group_id"] != selected["recording_group_id"]
            or row["reference_ref"] != selected["reference_ref"]
            or row["reference_sha256"] != selected["reference_sha256"]
        ):
            raise AliMeetingTrainMaterializationError(
                f"AliMeeting Train materialization row changed: {session_id}"
            )
        if root is None:
            continue
        raw_path = root / raw_ref
        audio_path = root / audio_ref
        annotation_path = root / annotation_ref
        try:
            if (
                not raw_path.is_file()
                or raw_path.is_symlink()
                or raw_path.stat().st_size
                != row["source_archive_member_size_bytes"]
                or sha256_file(raw_path)
                != row["source_archive_member_sha256"]
                or not annotation_path.is_file()
                or annotation_path.is_symlink()
                or annotation_path.stat().st_size
                != row["annotation_size_bytes"]
                or sha256_file(annotation_path)
                != row["annotation_file_sha256"]
                or not audio_path.is_file()
                or audio_path.is_symlink()
            ):
                raise AliMeetingTrainMaterializationError(
                    f"AliMeeting Train materialized source changed: {session_id}"
                )
            with wave.open(str(raw_path), "rb") as source:
                observed_source_identity = {
                    "channels": source.getnchannels(),
                    "sample_width_bytes": source.getsampwidth(),
                    "sample_rate_hz": source.getframerate(),
                    "duration_samples": source.getnframes(),
                    "compression": source.getcomptype(),
                }
            audio_identity = _mono_wav_identity(audio_path)
        except (OSError, EOFError, wave.Error) as exc:
            raise AliMeetingTrainMaterializationError(
                f"AliMeeting Train materialized bytes are invalid: {session_id}"
            ) from exc
        expected_audio_identity = {
            key: row[key]
            for key in (
                "waveform_sha256",
                "waveform_size_bytes",
                "sample_rate_hz",
                "channels",
                "sample_width_bytes",
                "duration_samples",
            )
        }
        if (
            observed_source_identity != source_identity
            or audio_identity != expected_audio_identity
        ):
            raise AliMeetingTrainMaterializationError(
                f"AliMeeting Train waveform identity changed: {session_id}"
            )
    if sha256_file(path) != MATERIALIZATION_RECEIPT_SHA256:
        raise AliMeetingTrainMaterializationError(
            "AliMeeting Train materialization receipt file identity changed"
        )
    return receipt


def write_materialization_receipt(
    archive_path: Path,
    selection_path: Path,
    extracted_textgrid_root: Path,
    corpus_root: Path,
    output_path: Path,
) -> None:
    receipt = materialize_selected_train(
        archive_path,
        selection_path,
        extracted_textgrid_root,
        corpus_root,
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
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--textgrid-root", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_materialization_receipt(
        args.archive.resolve(),
        args.selection.resolve(),
        args.textgrid_root.resolve(),
        args.corpus_root.resolve(),
        args.output.resolve(),
    )


if __name__ == "__main__":
    main()
