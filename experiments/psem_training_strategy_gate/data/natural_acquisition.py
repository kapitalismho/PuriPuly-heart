from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import wave
from pathlib import Path
from typing import Any, BinaryIO, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from experiments.psem_training_strategy_gate.data.provenance import (
    ADDITIONAL_AMI_MEETINGS,
    AMI_AUDIO_URL,
    ProvenanceError,
    _ami_identity_components,
    validate_ami_expansion_selection,
    wav_identity,
)

CONTENT_RANGE_PATTERN = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")


class NaturalAcquisitionError(RuntimeError):
    pass


def _identity_components(
    meetings: Mapping[str, Mapping[str, Any]], available: set[str]
) -> list[tuple[str, ...]]:
    try:
        normalized = {
            meeting_id: dict(metadata) for meeting_id, metadata in meetings.items()
        }
        return _ami_identity_components(normalized, available)
    except ProvenanceError as exc:
        raise NaturalAcquisitionError(str(exc)) from exc


def validate_expansion_selection(corpus_root: Path) -> dict[str, Any]:
    try:
        return validate_ami_expansion_selection(corpus_root)
    except ProvenanceError as exc:
        raise NaturalAcquisitionError(str(exc)) from exc


def _accepted_additional_identities() -> dict[str, dict[str, Any]]:
    manifest_path = Path(__file__).with_name("source_manifest.jsonl")
    try:
        rows = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise NaturalAcquisitionError(
            f"accepted source manifest is unreadable: {manifest_path}"
        ) from exc
    identity_fields = (
        "waveform_sha256",
        "waveform_size_bytes",
        "sample_rate_hz",
        "channels",
        "sample_width_bytes",
        "duration_samples",
    )
    identities: dict[str, dict[str, Any]] = {}
    for row in rows:
        meeting_id = row.get("session_id")
        if row.get("corpus") != "AMI" or meeting_id not in ADDITIONAL_AMI_MEETINGS:
            continue
        if (
            row.get("source_id") != f"ami_{meeting_id}"
            or row.get("audio_source_url") != AMI_AUDIO_URL.format(meeting=meeting_id)
            or any(field not in row for field in identity_fields)
            or meeting_id in identities
        ):
            raise NaturalAcquisitionError(
                f"accepted AMI source identity is invalid: {meeting_id}"
            )
        identities[meeting_id] = {field: row[field] for field in identity_fields}
    if set(identities) != ADDITIONAL_AMI_MEETINGS:
        raise NaturalAcquisitionError(
            "accepted AMI source identities do not cover the expansion inventory"
        )
    return identities


def _validate_expected_waveform(
    path: Path, expected_identity: Mapping[str, Any]
) -> None:
    try:
        observed = wav_identity(path)
    except (OSError, EOFError, ProvenanceError, wave.Error) as exc:
        raise NaturalAcquisitionError(f"invalid accepted AMI waveform: {path}") from exc
    if observed != dict(expected_identity):
        raise NaturalAcquisitionError(f"accepted AMI waveform identity changed: {path}")


def _response_status(response: BinaryIO) -> int:
    status = getattr(response, "status", None)
    if isinstance(status, int):
        return status
    return int(response.getcode())


def _remote_size(url: str, timeout_seconds: float) -> int:
    request = Request(
        url,
        headers={
            "Accept-Encoding": "identity",
            "User-Agent": "PuriPuly-heart-issue-77",
        },
        method="HEAD",
    )
    try:
        response = urlopen(request, timeout=timeout_seconds)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise NaturalAcquisitionError(f"AMI metadata request failed: {url}: {exc}") from exc
    with response:
        status = _response_status(response)
        content_length = response.headers.get("Content-Length")
        if status != 200 or content_length is None:
            raise NaturalAcquisitionError(
                f"AMI metadata response is incomplete: {url}: HTTP {status}"
            )
        size = int(content_length)
        if size <= 44:
            raise NaturalAcquisitionError(f"AMI remote waveform size is invalid: {url}: {size}")
        return size


def _prepare_partial(
    part_path: Path,
    remote_size: int,
    expected_identity: Mapping[str, Any] | None = None,
) -> int:
    existing_size = part_path.stat().st_size if part_path.is_file() else 0
    if existing_size > remote_size:
        part_path.unlink()
        return 0
    if existing_size == remote_size:
        try:
            if expected_identity is None:
                wav_identity(part_path)
            else:
                _validate_expected_waveform(part_path, expected_identity)
        except (
            OSError,
            EOFError,
            NaturalAcquisitionError,
            ProvenanceError,
            wave.Error,
        ):
            part_path.unlink()
            return 0
    return existing_size


def _download_to_part(
    url: str,
    part_path: Path,
    timeout_seconds: float,
    expected_identity: Mapping[str, Any],
) -> None:
    remote_size = _remote_size(url, timeout_seconds)
    expected_size_bytes = int(expected_identity["waveform_size_bytes"])
    if remote_size != expected_size_bytes:
        raise NaturalAcquisitionError(
            f"AMI remote waveform identity changed: {url}: {remote_size}/{expected_size_bytes}"
        )
    existing_size = _prepare_partial(part_path, remote_size, expected_identity)
    if existing_size == remote_size:
        return
    headers = {"Accept-Encoding": "identity", "User-Agent": "PuriPuly-heart-issue-77"}
    if existing_size:
        headers["Range"] = f"bytes={existing_size}-"
    request = Request(url, headers=headers)
    try:
        response = urlopen(request, timeout=timeout_seconds)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise NaturalAcquisitionError(f"AMI download failed: {url}: {exc}") from exc
    with response:
        status = _response_status(response)
        mode = "wb"
        if existing_size and status == 206:
            content_range = response.headers.get("Content-Range")
            match = CONTENT_RANGE_PATTERN.fullmatch(content_range or "")
            if (
                match is None
                or int(match.group(1)) != existing_size
                or int(match.group(3)) != remote_size
            ):
                raise NaturalAcquisitionError(
                    f"AMI resume response is invalid: {url}: {content_range}"
                )
            mode = "ab"
        elif status == 200:
            existing_size = 0
            content_length = response.headers.get("Content-Length")
            if content_length is None or int(content_length) != remote_size:
                raise NaturalAcquisitionError(
                    f"AMI download size metadata changed: {url}: {content_length}/{remote_size}"
                )
        else:
            raise NaturalAcquisitionError(f"AMI download returned HTTP {status}: {url}")
        transferred = 0
        with part_path.open(mode) as output:
            while chunk := response.read(
                min(1 << 20, remote_size - existing_size - transferred + 1)
            ):
                if existing_size + transferred + len(chunk) > remote_size:
                    raise NaturalAcquisitionError(
                        f"AMI download exceeds remote size: {url}: "
                        f"{existing_size + transferred + len(chunk)}/{remote_size}"
                    )
                output.write(chunk)
                transferred += len(chunk)
        content_length = response.headers.get("Content-Length")
        if content_length is not None and transferred != int(content_length):
            raise NaturalAcquisitionError(
                f"AMI download payload is incomplete: {url}: {transferred}/{content_length}"
            )
        if part_path.stat().st_size != remote_size:
            raise NaturalAcquisitionError(
                f"AMI download size is incomplete: {url}: {part_path.stat().st_size}/{remote_size}"
            )


@contextlib.contextmanager
def _materialization_lock(corpus_root: Path):
    lock_path = corpus_root / "ami" / "audio" / ".issue-77-materialization.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        if lock_path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise NaturalAcquisitionError(
                f"another AMI materializer is active: {lock_path}"
            ) from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _materialize_one(
    corpus_root: Path,
    meeting_id: str,
    timeout_seconds: float,
    expected_identity: Mapping[str, Any],
) -> str:
    target_dir = corpus_root / "ami" / "audio" / meeting_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{meeting_id}.Mix-Headset.wav"
    if target.is_file():
        _validate_expected_waveform(target, expected_identity)
        return "existing"
    part_path = target.with_suffix(target.suffix + ".part")
    _download_to_part(
        AMI_AUDIO_URL.format(meeting=meeting_id),
        part_path,
        timeout_seconds,
        expected_identity,
    )
    _validate_expected_waveform(part_path, expected_identity)
    os.replace(part_path, target)
    return "downloaded"


def materialize_additional_ami(
    corpus_root: Path, timeout_seconds: float = 120.0
) -> dict[str, Any]:
    with _materialization_lock(corpus_root):
        selection = validate_expansion_selection(corpus_root)
        expected_identities = _accepted_additional_identities()
        statuses = {
            meeting_id: _materialize_one(
                corpus_root,
                meeting_id,
                timeout_seconds,
                expected_identities[meeting_id],
            )
            for meeting_id in sorted(ADDITIONAL_AMI_MEETINGS)
        }
    return {
        **selection,
        "downloaded_count": sum(status == "downloaded" for status in statuses.values()),
        "existing_count": sum(status == "existing" for status in statuses.values()),
        "statuses": statuses,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    args = parser.parse_args()
    result = materialize_additional_ami(
        args.corpus_root.resolve(), timeout_seconds=args.timeout_seconds
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
