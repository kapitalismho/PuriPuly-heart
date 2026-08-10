from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import shutil
import sys
import tarfile
import urllib.request
import wave
import zipfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterable

import numpy as np
import soundfile as sf

from experiments.speaker_representation_scd.execution_guard import (
    load_completed_action_receipt,
    validate_worker_execution,
)
from experiments.speaker_representation_scd.provenance import (
    load_json,
    sha256_bytes,
    sha256_file,
    with_self_sha256,
)
from experiments.speaker_representation_scd.r1_forecast import (
    AUTHORITY,
    DEVELOPMENT_ACQUISITION_PATH,
    DEVELOPMENT_LEDGER_PATH,
    FROZEN_INPUTS,
    WAVEFORM_INVENTORY_PATH,
    _expected_coordinate_rows,
)
from experiments.speaker_representation_scd.r1_gate import EXPERIMENT_ROOT, REPOSITORY_ROOT
from experiments.speaker_representation_scd.r2_gate import (
    CONFIRMATORY_SOURCE_IDS,
    DEVELOPMENT_SOURCE_IDS,
    GATE_PATH,
    R2GateError,
    validated_r2_cache_root,
)
from experiments.speaker_representation_scd.run_provenance import run_provenance

ARCHIVE_RECEIPT = Path("manifests/r2/development/development_archive_receipt.json")
MATERIALIZATION_RECEIPT = Path(
    "manifests/r2/development/development_materialization_receipt.json"
)
SOURCE_METADATA_PATH = Path("data/r2/development/source_metadata.jsonl")
ZEROTH_URL = "https://storage.googleapis.com/zeroth_project/zeroth_korean.tar.gz"
JVS_URL = (
    "https://drive.usercontent.google.com/download?"
    "id=19oAw8wWn3Y7z6CKChRdAyGOB9yupL_Xt&export=download&confirm=t"
)
JVS_DEVELOPMENT = (
    "jvs046",
    "jvs095",
    "jvs089",
    "jvs081",
    "jvs064",
    "jvs060",
    "jvs028",
    "jvs009",
    "jvs068",
    "jvs015",
    "jvs030",
    "jvs053",
    "jvs047",
    "jvs078",
    "jvs032",
    "jvs055",
    "jvs048",
    "jvs022",
    "jvs024",
    "jvs097",
)
JVS_CONFIRMATORY = (
    "jvs050",
    "jvs003",
    "jvs094",
    "jvs011",
    "jvs052",
    "jvs023",
    "jvs002",
    "jvs016",
    "jvs013",
    "jvs025",
    "jvs093",
    "jvs019",
    "jvs066",
    "jvs058",
    "jvs051",
    "jvs086",
    "jvs059",
    "jvs029",
    "jvs033",
    "jvs077",
)
ZEROTH_TRAIN_SPEAKER_COUNT = 105
ZEROTH_TEST_SPEAKER_COUNT = 10
JVS_RELEASE_SPEAKER_COUNT = 100
JVS_CONDITION_COUNTS = {
    "parallel100": 100,
    "nonpara30": 30,
    "whisper10": 10,
    "falsetto10": 10,
}
MAX_DERIVED_BYTES = 20 * 1024**3
MAX_EXTERNAL_BYTES = 50 * 1024**3
MAX_COORDINATE_ROW_BYTES = 1024
AUXILIARY_OUTPUT_RESERVE_BYTES = 64 * 1024**2
CONTEXTS_MS = (100, 200, 300, 500, 750, 1000)


class MaterializationBudget:
    def __init__(self, cache_root: Path) -> None:
        self.cache_root = cache_root
        self.external_baseline = _tree_size(cache_root)
        self.derived_baseline = _derived_tree_size(cache_root)
        self.reserved = 0

    def ensure_projection(self, size_bytes: int) -> None:
        if size_bytes < 0:
            raise R2GateError("materialization projection is invalid")
        if self.derived_baseline + size_bytes > MAX_DERIVED_BYTES:
            raise R2GateError("materialization projection exceeds the 20 GiB derived ceiling")
        if self.external_baseline + size_bytes > MAX_EXTERNAL_BYTES:
            raise R2GateError("materialization projection exceeds the 50 GiB external ceiling")
        if shutil.disk_usage(self.cache_root.anchor).free < size_bytes:
            raise R2GateError("free disk is below the materialization projection")

    def reserve(self, size_bytes: int) -> None:
        self.ensure_projection(self.reserved + size_bytes)
        self.reserved += size_bytes


def _write_json(
    path: Path,
    document: dict[str, Any],
    budget: MaterializationBudget | None = None,
) -> dict[str, Any]:
    if path.exists():
        raise R2GateError(f"refusing to overwrite an existing R2 artifact: {path}")
    payload = with_self_sha256(document)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if budget is not None:
        budget.reserve(len(encoded.encode("utf-8")))
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(encoded, encoding="utf-8", newline="\n")
    temporary.replace(path)
    return payload


def _write_jsonl(
    path: Path,
    rows: Iterable[dict[str, Any]],
    staging: Path,
    budget: MaterializationBudget,
    *,
    maximum_bytes: int | None = None,
) -> None:
    temporary = staging / sha256_bytes(path.as_posix().encode("utf-8"))
    temporary.parent.mkdir(parents=True, exist_ok=True)
    encoded = [json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows]
    size_bytes = sum(len(row.encode("utf-8")) for row in encoded)
    if maximum_bytes is not None and size_bytes > maximum_bytes:
        raise R2GateError(f"materialized JSONL exceeds its projection: {path}")
    budget.reserve(size_bytes)
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.writelines(encoded)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(path)


def _download(
    url: str,
    target: Path,
    partial: Path,
    *,
    max_bytes: int,
) -> dict[str, Any]:
    if target.exists() or partial.exists():
        raise R2GateError("archive acquisition requires empty fixed target and partial paths")
    partial.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "speaker-representation-scd-v1-r2"}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        status = int(getattr(response, "status", response.getcode()))
        if status != 200:
            raise R2GateError(f"archive endpoint returned HTTP {status}")
        content_type = str(response.headers.get("Content-Type") or "").lower()
        if "text/html" in content_type:
            raise R2GateError("archive endpoint returned HTML")
        with partial.open("xb") as handle:
            while True:
                block = response.read(1 << 20)
                if not block:
                    break
                handle.write(block)
                if handle.tell() > max_bytes:
                    raise R2GateError("archive download exceeded its source ceiling")
        result = {
            "requested_url": url,
            "final_url": response.geturl(),
            "resumed_from_bytes": 0,
            "response_headers": {
                key.lower(): value
                for key, value in response.headers.items()
                if key.lower() in {"content-length", "content-range", "etag", "last-modified"}
            },
        }
    target.parent.mkdir(parents=True, exist_ok=True)
    partial.replace(target)
    return result


def _archive_magic(path: Path, archive_type: str) -> None:
    with path.open("rb") as handle:
        prefix = handle.read(4)
    if archive_type == "tar.gz" and prefix[:2] != b"\x1f\x8b":
        raise R2GateError("Zeroth archive gzip magic differs")
    if archive_type == "zip" and prefix != b"PK\x03\x04":
        raise R2GateError("JVS archive zip magic differs")


def _tree_size(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise R2GateError(f"external cache symlink is forbidden: {path}")
        if path.is_file():
            total += path.stat().st_size
    return total


def _derived_tree_size(cache_root: Path) -> int:
    roots = [
        cache_root / "data" / "r2" / "development",
        cache_root / "manifests" / "r2" / "development",
    ]
    source_root = cache_root / "sources" / "r2" / "development"
    if source_root.exists():
        roots.extend(path / "waveforms" for path in source_root.iterdir() if path.is_dir())
    return sum(_tree_size(root) for root in roots if root.exists())


def acquire_archives(cache_root: Path, requested_argv: tuple[str, ...]) -> dict[str, Any]:
    validated_r2_cache_root("development_archive_download")
    receipt_path = cache_root / ARCHIVE_RECEIPT
    execution = validate_worker_execution(cache_root, receipt_path)
    if execution.requested_argv != requested_argv:
        raise R2GateError("R2 archive worker invocation differs from its lease")
    if receipt_path.exists():
        raise R2GateError(f"refusing to rerun R2 archive acquisition: {receipt_path}")
    free_bytes = shutil.disk_usage(cache_root.anchor).free
    source_root = cache_root / "sources" / "r2" / "development"
    if source_root.exists() and any(source_root.rglob("*")):
        raise R2GateError("R2 development source root must be empty before archive acquisition")
    current_external_bytes = _tree_size(cache_root)
    if current_external_bytes + 17 * 1024**3 > 50 * 1024**3:
        raise R2GateError("projected archive acquisition exceeds the 50 GiB external ceiling")
    sources = (
        (
            "zeroth-korean-development",
            ZEROTH_URL,
            "zeroth_korean.tar.gz",
            "tar.gz",
            12 * 1024**3,
        ),
        (
            "jvs-development",
            JVS_URL,
            "jvs_ver1.zip",
            "zip",
            5 * 1024**3,
        ),
    )
    artifacts: list[dict[str, Any]] = []
    for source_id, url, name, archive_type, maximum in sources:
        target = cache_root / "sources" / "r2" / "development" / source_id / name
        partial = cache_root / "control" / "downloads" / "r2" / f"{name}.part"
        transfer = _download(url, target, partial, max_bytes=maximum)
        _archive_magic(target, archive_type)
        population = (
            _zeroth_release_population(target)
            if source_id == "zeroth-korean-development"
            else _jvs_release_population(target)
        )
        artifacts.append(
            {
                "source_id": source_id,
                "relative_to_cache_root": target.relative_to(cache_root).as_posix(),
                "archive_type": archive_type,
                "size_bytes": target.stat().st_size,
                "sha256": sha256_file(target),
                "transfer": transfer,
                "release_population": population,
            }
        )
    if sum(row["size_bytes"] for row in artifacts) > 25 * 1024**3:
        raise R2GateError("development archives exceed the 25 GiB source ceiling")
    gate_path = EXPERIMENT_ROOT / GATE_PATH
    gate = load_json(gate_path)
    receipt = {
        "schema_version": 1,
        "artifact_role": "r2_development_archive_receipt",
        "experiment_id": "speaker_representation_scd_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "authority": AUTHORITY,
        "frozen_inputs": FROZEN_INPUTS,
        "development_source_ids": list(DEVELOPMENT_SOURCE_IDS),
        "confirmatory_source_ids_sealed": list(CONFIRMATORY_SOURCE_IDS),
        "free_bytes_before_download": free_bytes,
        "external_bytes_before_download": current_external_bytes,
        "artifacts": artifacts,
        "r2_gate_sha256": sha256_file(gate_path),
        "r2_gate_self_sha256": gate["self_sha256"],
        "execution_code_manifest_sha256": gate["execution_code"]["manifest_sha256"],
        "supervision_binding": {
            "execution_id": execution.execution_id,
            "expected_receipt_relative_path": execution.expected_receipt_relative_path,
            "authority": "requires_completed_usage_attestation",
        },
        "run_provenance": run_provenance(
            REPOSITORY_ROOT,
            requested_argv,
            deterministic_seed=0,
            deterministic_kernels=False,
        ),
    }
    return _write_json(receipt_path, receipt)


def _validated_archive_map(cache_root: Path, receipt: dict[str, Any]) -> dict[str, Path]:
    gate_path = EXPERIMENT_ROOT / GATE_PATH
    gate = load_json(gate_path)
    if (
        receipt.get("authority") != AUTHORITY
        or receipt.get("frozen_inputs") != FROZEN_INPUTS
        or receipt.get("development_source_ids") != list(DEVELOPMENT_SOURCE_IDS)
        or receipt.get("confirmatory_source_ids_sealed") != list(CONFIRMATORY_SOURCE_IDS)
        or receipt.get("r2_gate_sha256") != sha256_file(gate_path)
        or receipt.get("r2_gate_self_sha256") != gate.get("self_sha256")
        or receipt.get("execution_code_manifest_sha256")
        != gate.get("execution_code", {}).get("manifest_sha256")
    ):
        raise R2GateError("R2 archive receipt gate identity differs")
    expected = {
        "zeroth-korean-development": (
            "sources/r2/development/zeroth-korean-development/zeroth_korean.tar.gz",
            "tar.gz",
        ),
        "jvs-development": (
            "sources/r2/development/jvs-development/jvs_ver1.zip",
            "zip",
        ),
    }
    rows = receipt.get("artifacts")
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise R2GateError("R2 archive receipt artifact inventory differs")
    result: dict[str, Path] = {}
    listed: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise R2GateError("R2 archive receipt artifact row is invalid")
        source_id = row.get("source_id")
        if source_id not in expected:
            raise R2GateError("R2 archive receipt source differs")
        relative, archive_type = expected[source_id]
        if row.get("relative_to_cache_root") != relative or row.get("archive_type") != archive_type:
            raise R2GateError("R2 archive receipt path or type differs")
        path = cache_root / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or row.get("size_bytes") != path.stat().st_size
            or row.get("sha256") != sha256_file(path)
        ):
            raise R2GateError("R2 archive receipt byte identity differs")
        _archive_magic(path, archive_type)
        population = (
            _zeroth_release_population(path)
            if source_id == "zeroth-korean-development"
            else _jvs_release_population(path)
        )
        if row.get("release_population") != population:
            raise R2GateError("R2 archive receipt release population differs")
        result[source_id] = path
        listed.add(relative)
    actual = {
        path.relative_to(cache_root).as_posix()
        for path in (cache_root / "sources" / "r2" / "development").rglob("*")
        if path.is_file()
    }
    if listed != actual:
        raise R2GateError("R2 archive source tree contains unregistered files")
    return result


def _member_parts(name: str) -> tuple[str, ...]:
    if "\\" in name or "\x00" in name:
        raise R2GateError(f"archive member name is non-canonical: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise R2GateError(f"archive member path is unsafe: {name!r}")
    if ":" in path.parts[0]:
        raise R2GateError(f"archive member drive path is unsafe: {name!r}")
    return path.parts


def _zeroth_member_inventory(archive: Path) -> tuple[list[tarfile.TarInfo], tuple[str, ...], tuple[str, ...]]:
    audio: list[tarfile.TarInfo] = []
    train_speakers: set[str] = set()
    test_speakers: set[str] = set()
    names: set[str] = set()
    folded: set[str] = set()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            parts = _member_parts(member.name.rstrip("/"))
            normalized = "/".join(parts)
            if normalized in names or normalized.casefold() in folded:
                raise R2GateError("Zeroth archive has duplicate or case-colliding members")
            names.add(normalized)
            folded.add(normalized.casefold())
            if member.issym() or member.islnk() or member.isdev():
                raise R2GateError("Zeroth archive contains a forbidden link or device member")
            for partition, speakers in (("train_data_01", train_speakers), ("test_data_01", test_speakers)):
                if partition not in parts:
                    continue
                index = parts.index(partition)
                if len(parts) >= index + 4 and parts[-1].lower().endswith(".flac"):
                    speakers.add(parts[index + 2])
                    if partition == "train_data_01" and member.isfile():
                        audio.append(member)
    return audio, tuple(sorted(train_speakers)), tuple(sorted(test_speakers))


def select_zeroth_development_members(archive: Path) -> tuple[list[tarfile.TarInfo], tuple[str, ...]]:
    audio, train_speakers, test_speakers = _zeroth_member_inventory(archive)
    if (
        len(train_speakers) != ZEROTH_TRAIN_SPEAKER_COUNT
        or len(test_speakers) != ZEROTH_TEST_SPEAKER_COUNT
        or set(train_speakers) & set(test_speakers)
    ):
        raise R2GateError("Zeroth train/test speaker metadata is invalid")
    selected = tuple(
        sorted(train_speakers, key=lambda value: (hashlib.sha256(value.encode()).hexdigest(), value))[:20]
    )
    if len(selected) != 20:
        raise R2GateError("Zeroth development selection does not contain 20 speakers")
    rows: list[tarfile.TarInfo] = []
    for member in audio:
        parts = _member_parts(member.name)
        index = parts.index("train_data_01")
        if parts[index + 2] in selected:
            rows.append(member)
    rows.sort(key=lambda member: member.name)
    if not rows:
        raise R2GateError("Zeroth development selection has no audio")
    return rows, selected


def _zeroth_release_population(archive: Path) -> dict[str, Any]:
    _, train_speakers, test_speakers = _zeroth_member_inventory(archive)
    if (
        len(train_speakers) != ZEROTH_TRAIN_SPEAKER_COUNT
        or len(test_speakers) != ZEROTH_TEST_SPEAKER_COUNT
        or set(train_speakers) & set(test_speakers)
    ):
        raise R2GateError("Zeroth train/test speaker metadata is invalid")
    selected = tuple(
        sorted(train_speakers, key=lambda value: (hashlib.sha256(value.encode()).hexdigest(), value))[:20]
    )
    return {
        "train_speaker_count": len(train_speakers),
        "test_speaker_count": len(test_speakers),
        "train_speakers": list(train_speakers),
        "test_speakers": list(test_speakers),
        "selected_development_speakers": list(selected),
    }


def _jvs_member_inventory(
    archive: Path,
) -> tuple[list[zipfile.ZipInfo], dict[str, dict[str, int]]]:
    selected: list[zipfile.ZipInfo] = []
    names: set[str] = set()
    folded: set[str] = set()
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            parts = _member_parts(member.filename.rstrip("/"))
            normalized = "/".join(parts)
            if normalized in names or normalized.casefold() in folded:
                raise R2GateError("JVS archive has duplicate or case-colliding members")
            names.add(normalized)
            folded.add(normalized.casefold())
            mode = member.external_attr >> 16
            if (mode & 0o170000) == 0o120000:
                raise R2GateError("JVS archive contains a forbidden symlink")
            speakers = [
                part
                for part in parts
                if part.startswith("jvs") and len(part) == 6 and part[3:].isdigit()
            ]
            if not speakers or member.is_dir() or not member.filename.lower().endswith(".wav"):
                continue
            speaker = speakers[0]
            conditions = [condition for condition in JVS_CONDITION_COUNTS if condition in parts]
            if not conditions:
                continue
            condition = conditions[0]
            counts[speaker][condition] += 1
            if speaker in JVS_CONFIRMATORY:
                continue
            if speaker not in JVS_DEVELOPMENT:
                continue
            selected.append(member)
    selected.sort(key=lambda member: member.filename)
    return selected, {
        speaker: dict(sorted(condition_counts.items()))
        for speaker, condition_counts in sorted(counts.items())
    }


def _jvs_release_population(archive: Path) -> dict[str, Any]:
    _, counts = _jvs_member_inventory(archive)
    expected_speakers = {f"jvs{index:03d}" for index in range(1, JVS_RELEASE_SPEAKER_COUNT + 1)}
    if set(counts) != expected_speakers:
        raise R2GateError("JVS release speaker population differs")
    for speaker in sorted(expected_speakers):
        if counts[speaker] != JVS_CONDITION_COUNTS:
            raise R2GateError(f"JVS condition coverage differs for {speaker}")
    return {
        "speaker_count": len(counts),
        "condition_member_counts": dict(JVS_CONDITION_COUNTS),
        "speaker_condition_member_counts": counts,
        "development_speakers": list(JVS_DEVELOPMENT),
        "confirmatory_speakers_sealed": list(JVS_CONFIRMATORY),
    }


def select_jvs_development_members(archive: Path) -> list[zipfile.ZipInfo]:
    selected, counts = _jvs_member_inventory(archive)
    expected_speakers = {f"jvs{index:03d}" for index in range(1, JVS_RELEASE_SPEAKER_COUNT + 1)}
    if set(counts) != expected_speakers:
        raise R2GateError("JVS release speaker population differs")
    for speaker in sorted(expected_speakers):
        if counts[speaker] != JVS_CONDITION_COUNTS:
            raise R2GateError(f"JVS condition coverage differs for {speaker}")
    observed = {
        next(part for part in _member_parts(member.filename) if part.startswith("jvs") and len(part) == 6)
        for member in selected
    }
    if observed != set(JVS_DEVELOPMENT):
        raise R2GateError("JVS development speaker coverage differs")
    return selected


def _pcm16(data: np.ndarray, source_rate: int, target_rate: int = 16000) -> np.ndarray:
    if data.ndim != 1:
        raise R2GateError("development audio must be mono")
    if source_rate == target_rate:
        if data.dtype == np.int16:
            return data
        return np.clip(np.rint(data.astype(np.float64) * 32768.0), -32768, 32767).astype(np.int16)
    if source_rate != 24000:
        raise R2GateError(f"unsupported development source sample rate: {source_rate}")
    import torch
    import torchaudio.functional as audio_functional

    values = torch.from_numpy(data.astype(np.float32) / 32768.0)
    output = audio_functional.resample(
        values,
        source_rate,
        target_rate,
        lowpass_filter_width=64,
        rolloff=0.9475937167399596,
        resampling_method="sinc_interp_kaiser",
        beta=14.769656459379492,
    )
    return np.clip(np.rint(output.numpy().astype(np.float64) * 32768.0), -32768, 32767).astype(np.int16)


def _write_wav(
    path: Path,
    samples: np.ndarray,
    staging: Path,
    budget: MaterializationBudget,
) -> None:
    temporary = staging / f"{path.stem}.wav"
    temporary.parent.mkdir(parents=True, exist_ok=True)
    expected_bytes = 44 + samples.size * 2
    budget.reserve(expected_bytes)
    with wave.open(str(temporary), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(samples.astype("<i2", copy=False).tobytes())
    if temporary.stat().st_size != expected_bytes:
        raise R2GateError(f"canonical waveform size differs from projection: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if sha256_file(path) != sha256_file(temporary):
            raise R2GateError(f"existing materialized waveform differs: {path}")
        temporary.unlink()
    else:
        temporary.replace(path)


def _waveform_row(path: Path, cache_root: Path, waveform_id: str, source_id: str) -> dict[str, Any]:
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2 or handle.getframerate() != 16000:
            raise R2GateError(f"canonical waveform contract differs: {path}")
        samples = handle.getnframes()
    if samples < 1600:
        raise R2GateError(f"canonical waveform is shorter than 100 ms: {path}")
    return {
        "waveform_id": waveform_id,
        "source_id": source_id,
        "artifact_relative_to_cache_root": path.relative_to(cache_root).as_posix(),
        "artifact_sha256": sha256_file(path),
        "artifact_size_bytes": path.stat().st_size,
        "sample_rate_hz": 16000,
        "num_samples": samples,
        "eligible_start_sample": 0,
        "eligible_end_sample": samples,
    }


def _legacy_sources() -> list[dict[str, Any]]:
    legacy = REPOSITORY_ROOT / "experiments" / "speaker_turn_boundary"
    results = legacy / "results" / "turn_episode_v1"
    manifest = load_json(results / "episode_manifest_dev.json")
    episodes = [row for row in manifest["episodes"] if row["pool"] == "diagnostic_dev"]
    if len(episodes) != 695:
        raise R2GateError("legacy diagnostic episode population differs")
    inventory = load_json(results / "coverage_inventory.json")
    details = {
        row["session_id"]: row
        for row in (
            json.loads(line)
            for line in (results / "coverage_inventory_details.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        )
    }
    manifests = {
        name: {
            row["case_id"]: row
            for row in load_json(legacy / "data" / "manifests" / f"{name}.json")["cases"]
        }
        for name in ("ls_dev", "ls_held_out_clean", "ls_held_out_other")
    }
    corpus_root = Path(inventory["corpus_root"]).resolve()
    rows: dict[str, dict[str, Any]] = {}
    for episode in episodes:
        session_id = str(episode["session_id"])
        if ":" in session_id and session_id.split(":", 1)[0] in manifests:
            manifest_name, case_id = session_id.split(":", 1)
            case = manifests[manifest_name][case_id]
            relative = Path(case["wav_relative_path"])
            candidates = (
                legacy / "data" / relative,
                corpus_root / "phase2_build" / relative,
                corpus_root / relative,
            )
            paths = [candidate.resolve() for candidate in candidates if candidate.is_file()]
            if len(paths) != 1:
                raise R2GateError(f"legacy synthetic source resolution differs: {session_id}")
            path = paths[0]
        else:
            detail = details[session_id]
            path = (corpus_root / detail["wav_path"]).resolve()
        if not path.is_file() or path.is_symlink():
            raise R2GateError(f"legacy waveform is missing or linked: {session_id}")
        expected = str(episode["wav_sha256"])
        if sha256_file(path) != expected:
            raise R2GateError(f"legacy waveform hash differs: {session_id}")
        row = rows.setdefault(expected, {"path": path, "session_ids": []})
        if row["path"] != path:
            if sha256_file(row["path"]) != expected:
                raise R2GateError(f"legacy duplicate waveform differs: {session_id}")
        row["session_ids"].append(session_id)
    if len(rows) != 600:
        raise R2GateError("legacy diagnostic waveform population differs")
    return [
        {"wav_sha256": digest, **rows[digest]}
        for digest in sorted(rows)
    ]


def _canonical_sample_count(frames: int, sample_rate: int, channels: int) -> int:
    if channels != 1 or frames < 0:
        raise R2GateError("development audio metadata is not mono PCM-compatible")
    if sample_rate == 16000:
        return frames
    if sample_rate == 24000:
        return math.ceil(frames * 16000 / 24000)
    raise R2GateError(f"unsupported development source sample rate: {sample_rate}")


def _planned_waveform(
    waveform_id: str,
    source_id: str,
    frames: int,
    sample_rate: int,
    channels: int,
) -> dict[str, Any]:
    samples = _canonical_sample_count(frames, sample_rate, channels)
    if samples < 1600:
        raise R2GateError("planned canonical waveform is shorter than 100 ms")
    return {
        "waveform_id": waveform_id,
        "source_id": source_id,
        "artifact_sha256": "0" * 64,
        "eligible_start_sample": 0,
        "eligible_end_sample": samples,
        "num_samples": samples,
    }


def _coordinate_count(waveform: dict[str, Any]) -> int:
    end = int(waveform["num_samples"])
    return sum(max(0, (end - context_ms * 16) // 800 + 1) for context_ms in CONTEXTS_MS)


def _materialization_projection(
    archives: dict[str, Path],
) -> tuple[int, dict[str, Any]]:
    planned: list[dict[str, Any]] = []
    selected_source_bytes = 0
    for source in _legacy_sources():
        path = source["path"]
        with wave.open(str(path), "rb") as handle:
            planned.append(
                _planned_waveform(
                    f"legacy_{source['wav_sha256'][:24]}",
                    "legacy-common-gt-v1",
                    handle.getnframes(),
                    handle.getframerate(),
                    handle.getnchannels(),
                )
            )
        selected_source_bytes += path.stat().st_size
    zeroth_members, _ = select_zeroth_development_members(
        archives["zeroth-korean-development"]
    )
    selected_source_bytes += sum(member.size for member in zeroth_members)
    with tarfile.open(archives["zeroth-korean-development"], "r:gz") as handle:
        for member in zeroth_members:
            source = handle.extractfile(member)
            if source is None:
                raise R2GateError(f"cannot inspect selected Zeroth member: {member.name}")
            info = sf.info(source)
            identity = sha256_bytes(member.name.encode("utf-8"))[:24]
            planned.append(
                _planned_waveform(
                    f"zeroth_{identity}",
                    "zeroth-korean-development",
                    int(info.frames),
                    int(info.samplerate),
                    int(info.channels),
                )
            )
    jvs_members = select_jvs_development_members(archives["jvs-development"])
    selected_source_bytes += sum(member.file_size for member in jvs_members)
    with zipfile.ZipFile(archives["jvs-development"]) as handle:
        for member in jvs_members:
            with handle.open(member) as source:
                info = sf.info(source)
            identity = sha256_bytes(member.filename.encode("utf-8"))[:24]
            planned.append(
                _planned_waveform(
                    f"jvs_{identity}",
                    "jvs-development",
                    int(info.frames),
                    int(info.samplerate),
                    int(info.channels),
                )
            )
    if selected_source_bytes > MAX_DERIVED_BYTES:
        raise R2GateError("selected uncompressed source payloads exceed 20 GiB")
    waveform_bytes = sum(44 + int(row["num_samples"]) * 2 for row in planned)
    coordinate_count = sum(_coordinate_count(row) for row in planned)
    coordinate_bytes = coordinate_count * MAX_COORDINATE_ROW_BYTES
    projected = waveform_bytes + coordinate_bytes + AUXILIARY_OUTPUT_RESERVE_BYTES
    return projected, {
        "waveform_count": len(planned),
        "selected_source_payload_bytes": selected_source_bytes,
        "canonical_waveform_bytes": waveform_bytes,
        "coordinate_count": coordinate_count,
        "maximum_coordinate_bytes": coordinate_bytes,
        "auxiliary_output_reserve_bytes": AUXILIARY_OUTPUT_RESERVE_BYTES,
        "projected_derived_bytes": projected,
    }


def _materialize_legacy(
    cache_root: Path,
    staging: Path,
    budget: MaterializationBudget,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    waveforms: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    root = cache_root / "sources" / "r2" / "development" / "legacy-common-gt-v1" / "waveforms"
    for source in _legacy_sources():
        waveform_id = f"legacy_{source['wav_sha256'][:24]}"
        target = root / f"{waveform_id}.wav"
        if target.exists():
            if sha256_file(target) != source["wav_sha256"]:
                raise R2GateError(f"existing legacy waveform differs: {target}")
        else:
            temporary = staging / target.name
            budget.reserve(source["path"].stat().st_size)
            shutil.copyfile(source["path"], temporary)
            if sha256_file(temporary) != source["wav_sha256"]:
                raise R2GateError("legacy copy hash differs")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary.replace(target)
        waveforms.append(_waveform_row(target, cache_root, waveform_id, "legacy-common-gt-v1"))
        metadata.append(
            {
                "waveform_id": waveform_id,
                "source_id": "legacy-common-gt-v1",
                "legacy_session_ids": sorted(set(source["session_ids"])),
                "original_wav_sha256": source["wav_sha256"],
            }
        )
    return waveforms, metadata


def _read_audio(stream: BinaryIO) -> tuple[np.ndarray, int]:
    values, sample_rate = sf.read(stream, dtype="int16", always_2d=False)
    return np.asarray(values), int(sample_rate)


def _materialize_zeroth(
    cache_root: Path,
    archive: Path,
    staging: Path,
    budget: MaterializationBudget,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    members, speakers = select_zeroth_development_members(archive)
    waveforms: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    root = cache_root / "sources" / "r2" / "development" / "zeroth-korean-development" / "waveforms"
    with tarfile.open(archive, "r:gz") as handle:
        for member in members:
            parts = _member_parts(member.name)
            index = parts.index("train_data_01")
            speaker = parts[index + 2]
            source = handle.extractfile(member)
            if source is None:
                raise R2GateError(f"cannot open selected Zeroth member: {member.name}")
            payload = source.read()
            values, rate = _read_audio(io.BytesIO(payload))
            samples = _pcm16(values, rate)
            identity = sha256_bytes(member.name.encode("utf-8"))[:24]
            waveform_id = f"zeroth_{identity}"
            target = root / f"{waveform_id}.wav"
            _write_wav(target, samples, staging, budget)
            waveforms.append(_waveform_row(target, cache_root, waveform_id, "zeroth-korean-development"))
            metadata.append(
                {
                    "waveform_id": waveform_id,
                    "source_id": "zeroth-korean-development",
                    "speaker_id": speaker,
                    "source_member": member.name,
                    "source_member_sha256": sha256_bytes(payload),
                    "source_sample_rate_hz": rate,
                }
            )
    if {row["speaker_id"] for row in metadata} != set(speakers):
        raise R2GateError("Zeroth materialized speaker coverage differs")
    return waveforms, metadata


def _materialize_jvs(
    cache_root: Path,
    archive: Path,
    staging: Path,
    budget: MaterializationBudget,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    members = select_jvs_development_members(archive)
    waveforms: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    root = cache_root / "sources" / "r2" / "development" / "jvs-development" / "waveforms"
    with zipfile.ZipFile(archive) as handle:
        for member in members:
            parts = _member_parts(member.filename)
            speaker = next(part for part in parts if part in JVS_DEVELOPMENT)
            condition = next(
                part
                for part in parts
                if part in {"parallel100", "nonpara30", "whisper10", "falsetto10"}
            )
            payload = handle.read(member)
            values, rate = _read_audio(io.BytesIO(payload))
            samples = _pcm16(values, rate)
            identity = sha256_bytes(member.filename.encode("utf-8"))[:24]
            waveform_id = f"jvs_{identity}"
            target = root / f"{waveform_id}.wav"
            _write_wav(target, samples, staging, budget)
            waveforms.append(_waveform_row(target, cache_root, waveform_id, "jvs-development"))
            metadata.append(
                {
                    "waveform_id": waveform_id,
                    "source_id": "jvs-development",
                    "speaker_id": speaker,
                    "condition": condition,
                    "source_member": member.filename,
                    "source_member_sha256": sha256_bytes(payload),
                    "source_sample_rate_hz": rate,
                }
            )
    return waveforms, metadata


def _artifact_rows(cache_root: Path) -> list[dict[str, Any]]:
    root = cache_root / "sources" / "r2" / "development"
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise R2GateError(f"development source symlink is forbidden: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(cache_root).as_posix()
        parts = Path(relative).parts
        source_id = parts[3] if len(parts) > 3 else None
        if source_id not in DEVELOPMENT_SOURCE_IDS:
            raise R2GateError(f"development source namespace differs: {relative}")
        rows.append(
            {
                "source_id": source_id,
                "location": "cache_root",
                "relative_path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    legacy = REPOSITORY_ROOT / "experiments" / "speaker_turn_boundary" / "results" / "turn_episode_v1" / "episode_manifest_dev.json"
    rows.insert(
        0,
        {
            "source_id": "legacy-common-gt-v1",
            "location": "repository",
            "relative_path": "experiments/speaker_turn_boundary/results/turn_episode_v1/episode_manifest_dev.json",
            "size_bytes": legacy.stat().st_size,
            "sha256": sha256_file(legacy),
        },
    )
    return rows


def _coordinate_ledger(
    cache_root: Path,
    waveforms: list[dict[str, Any]],
    acquisition_path: Path,
    acquisition: dict[str, Any],
    staging: Path,
    budget: MaterializationBudget,
) -> dict[str, Any]:
    counts_context: dict[str, int] = defaultdict(int)
    counts_source: dict[str, int] = defaultdict(int)
    shards: list[dict[str, Any]] = []
    total = 0
    for waveform in waveforms:
        rows = _expected_coordinate_rows(waveform)
        source_id = waveform["source_id"]
        path = (
            cache_root
            / "data"
            / "r2"
            / "development"
            / "coordinates"
            / source_id
            / f"{waveform['waveform_id']}.jsonl"
        )
        _write_jsonl(
            path,
            rows,
            staging,
            budget,
            maximum_bytes=len(rows) * MAX_COORDINATE_ROW_BYTES,
        )
        for row in rows:
            counts_context[str(row["context_ms"])] += 1
        counts_source[source_id] += len(rows)
        total += len(rows)
        shards.append(
            {
                "source_id": source_id,
                "waveform_id": waveform["waveform_id"],
                "relative_to_cache_root": path.relative_to(cache_root).as_posix(),
                "row_count": len(rows),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    document = {
        "schema_version": 1,
        "artifact_role": "r2_development_coordinate_ledger",
        "experiment_id": "speaker_representation_scd_v1",
        "authority": AUTHORITY,
        "frozen_inputs": FROZEN_INPUTS,
        "development_acquisition_receipt": {
            "relative_to_cache_root": acquisition_path.relative_to(cache_root).as_posix(),
            "sha256": sha256_file(acquisition_path),
            "self_sha256": acquisition["self_sha256"],
        },
        "development_source_ids": list(DEVELOPMENT_SOURCE_IDS),
        "extraction_windows_by_context_ms": dict(sorted(counts_context.items())),
        "extraction_windows_by_source_id": {
            source_id: counts_source[source_id] for source_id in DEVELOPMENT_SOURCE_IDS
        },
        "total_window_count": total,
        "coordinate_shards": shards,
    }
    return _write_json(cache_root / DEVELOPMENT_LEDGER_PATH, document, budget)


def materialize_development(cache_root: Path, requested_argv: tuple[str, ...]) -> dict[str, Any]:
    validated_r2_cache_root("development_waveform_materialization")
    receipt_path = cache_root / MATERIALIZATION_RECEIPT
    execution = validate_worker_execution(cache_root, receipt_path)
    if execution.requested_argv != requested_argv:
        raise R2GateError("R2 materialization worker invocation differs from its lease")
    if receipt_path.exists() or (cache_root / DEVELOPMENT_ACQUISITION_PATH).exists() or (cache_root / DEVELOPMENT_LEDGER_PATH).exists():
        raise R2GateError("refusing to overwrite an existing R2 materialization result")
    archive_path = cache_root / ARCHIVE_RECEIPT
    archives = load_completed_action_receipt(cache_root, archive_path, "r2-archives")
    archive_map = _validated_archive_map(cache_root, archives)
    budget = MaterializationBudget(cache_root)
    projected_bytes, projection = _materialization_projection(archive_map)
    budget.ensure_projection(projected_bytes)
    staging = cache_root / "control" / "staging" / "r2" / execution.execution_id
    staging.mkdir(parents=True, exist_ok=False)
    legacy_waveforms, legacy_metadata = _materialize_legacy(cache_root, staging, budget)
    zeroth_waveforms, zeroth_metadata = _materialize_zeroth(
        cache_root,
        archive_map["zeroth-korean-development"],
        staging,
        budget,
    )
    jvs_waveforms, jvs_metadata = _materialize_jvs(
        cache_root,
        archive_map["jvs-development"],
        staging,
        budget,
    )
    waveforms = sorted(
        legacy_waveforms + zeroth_waveforms + jvs_waveforms,
        key=lambda row: row["waveform_id"],
    )
    metadata = sorted(
        legacy_metadata + zeroth_metadata + jvs_metadata,
        key=lambda row: row["waveform_id"],
    )
    if len({row["waveform_id"] for row in waveforms}) != len(waveforms):
        raise R2GateError("R2 waveform IDs are not unique")
    _write_jsonl(cache_root / WAVEFORM_INVENTORY_PATH, waveforms, staging, budget)
    _write_jsonl(cache_root / SOURCE_METADATA_PATH, metadata, staging, budget)
    artifacts = _artifact_rows(cache_root)
    inventory_path = cache_root / WAVEFORM_INVENTORY_PATH
    acquisition_path = cache_root / DEVELOPMENT_ACQUISITION_PATH
    acquisition = _write_json(
        acquisition_path,
        {
            "schema_version": 1,
            "artifact_role": "r2_development_acquisition_receipt",
            "experiment_id": "speaker_representation_scd_v1",
            "authority": AUTHORITY,
            "frozen_inputs": FROZEN_INPUTS,
            "development_source_ids": list(DEVELOPMENT_SOURCE_IDS),
            "free_bytes_before_download": archives["free_bytes_before_download"],
            "external_source_download_bytes": sum(
                row["size_bytes"] for row in artifacts if row["location"] == "cache_root"
            ),
            "waveform_inventory": {
                "relative_to_cache_root": WAVEFORM_INVENTORY_PATH.as_posix(),
                "size_bytes": inventory_path.stat().st_size,
                "sha256": sha256_file(inventory_path),
            },
            "waveform_count": len(waveforms),
            "artifacts": artifacts,
            "release_populations": {
                row["source_id"]: row["release_population"]
                for row in archives["artifacts"]
            },
            "materialization_projection": projection,
        },
        budget,
    )
    ledger = _coordinate_ledger(
        cache_root,
        waveforms,
        acquisition_path,
        acquisition,
        staging,
        budget,
    )
    gate_path = EXPERIMENT_ROOT / GATE_PATH
    gate = load_json(gate_path)
    receipt = {
        "schema_version": 1,
        "artifact_role": "r2_development_materialization_receipt",
        "experiment_id": "speaker_representation_scd_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "authority": AUTHORITY,
        "development_source_ids": list(DEVELOPMENT_SOURCE_IDS),
        "confirmatory_payload_read": False,
        "confirmatory_member_extraction": False,
        "archive_receipt": {
            "relative_to_cache_root": ARCHIVE_RECEIPT.as_posix(),
            "sha256": sha256_file(archive_path),
            "self_sha256": archives["self_sha256"],
        },
        "development_acquisition_receipt": {
            "relative_to_cache_root": DEVELOPMENT_ACQUISITION_PATH.as_posix(),
            "sha256": sha256_file(acquisition_path),
            "self_sha256": acquisition["self_sha256"],
        },
        "development_coordinate_ledger": {
            "relative_to_cache_root": DEVELOPMENT_LEDGER_PATH.as_posix(),
            "sha256": sha256_file(cache_root / DEVELOPMENT_LEDGER_PATH),
            "self_sha256": ledger["self_sha256"],
        },
        "source_metadata": {
            "relative_to_cache_root": SOURCE_METADATA_PATH.as_posix(),
            "sha256": sha256_file(cache_root / SOURCE_METADATA_PATH),
            "size_bytes": (cache_root / SOURCE_METADATA_PATH).stat().st_size,
        },
        "waveform_count": len(waveforms),
        "coordinate_count": ledger["total_window_count"],
        "release_populations": acquisition["release_populations"],
        "materialization_projection": projection,
        "materialized_derived_bytes": budget.reserved,
        "r2_gate_sha256": sha256_file(gate_path),
        "r2_gate_self_sha256": gate["self_sha256"],
        "execution_code_manifest_sha256": gate["execution_code"]["manifest_sha256"],
        "supervision_binding": {
            "execution_id": execution.execution_id,
            "expected_receipt_relative_path": execution.expected_receipt_relative_path,
            "authority": "requires_completed_usage_attestation",
        },
        "run_provenance": run_provenance(
            REPOSITORY_ROOT,
            requested_argv,
            deterministic_seed=0,
            deterministic_kernels=True,
        ),
    }
    result = _write_json(receipt_path, receipt, budget)
    if _derived_tree_size(cache_root) > MAX_DERIVED_BYTES:
        raise R2GateError("materialized outputs exceed the 20 GiB derived ceiling")
    if _tree_size(cache_root) > MAX_EXTERNAL_BYTES:
        raise R2GateError("external cache exceeds the 50 GiB ceiling")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True, choices=("archives", "materialize"))
    args = parser.parse_args(argv)
    action = (
        "development_archive_download"
        if args.worker == "archives"
        else "development_waveform_materialization"
    )
    cache_root = validated_r2_cache_root(action)
    requested = tuple(json.loads(os.environ.get("SRSCD_REQUESTED_ARGV", "[]")))
    if args.worker == "archives":
        receipt = acquire_archives(cache_root, requested)
    else:
        receipt = materialize_development(cache_root, requested)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except R2GateError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error
