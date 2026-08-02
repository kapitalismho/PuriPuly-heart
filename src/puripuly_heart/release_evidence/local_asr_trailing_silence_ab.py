from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from puripuly_heart import __version__
from puripuly_heart.app.adapters.gpu_worker_process import DefaultGpuWorkerProcessFactory
from puripuly_heart.app.wiring.wiring_local_asr_provider_runtime import (
    _create_shared_gpu_asr_runtime,
)
from puripuly_heart.core.language import get_local_qwen_language_hint
from puripuly_heart.core.local_gpu_assets import inspect_local_gpu_install, local_gpu_model_path
from puripuly_heart.core.local_stt_assets import (
    LOCAL_QWEN_GPU_MODEL_ID,
    load_local_stt_asset_manifest,
)
from puripuly_heart.core.local_stt_catalog import inspect_required_cpu_model_installs
from puripuly_heart.core.runtime.gpu_asr import GpuASRDiagnostic
from puripuly_heart.core.stt.backend import STTBackendSession, STTBackendTranscriptEvent
from puripuly_heart.providers.stt.local_cpu import create_local_cpu_backend
from puripuly_heart.providers.stt.local_gpu import LocalGpuSTTBackend
from puripuly_heart.release_evidence.local_cpu_real_decode import (
    DECODE_CASES,
    DecodeCase,
    _attempt_payload,
    _read_audio,
    _resample,
    _sha256,
)

REPORT_SCHEMA = "puripuly-heart/local-asr-trailing-silence-ab/v2"
MODULE_NAME = "puripuly_heart.release_evidence.local_asr_trailing_silence_ab"
REPORTED_TRAILING_SILENCE_MS = 400
SAFETY_TAIL_MS = 128
EXPECTED_TRIMMED_MS = REPORTED_TRAILING_SILENCE_MS - SAFETY_TAIL_MS
AUTHORITY_PIN_PATTERN = re.compile(r"[0-9a-f]{64}")
GIT_TREE_PATTERN = re.compile(r"[0-9a-f]{40}")
SOURCE_PATHS = (
    "src/puripuly_heart/app/wiring/wiring_local_asr_provider_runtime.py",
    "src/puripuly_heart/core/local_asr/trailing_silence.py",
    "src/puripuly_heart/core/runtime/gpu_asr.py",
    "src/puripuly_heart/providers/stt/local_cpu.py",
    "src/puripuly_heart/providers/stt/local_gpu.py",
    "src/puripuly_heart/providers/stt/local_parakeet_sherpa.py",
    "src/puripuly_heart/providers/stt/local_qwen_sherpa.py",
    "src/puripuly_heart/release_evidence/local_asr_trailing_silence_ab.py",
)
NATIVE_RUNTIME_MODULES = (
    ("sherpa_onnx", "sherpa-onnx", "sherpa_onnx.lib._sherpa_onnx"),
    (
        "onnxruntime",
        "onnxruntime",
        "onnxruntime.capi.onnxruntime_pybind11_state",
    ),
    ("soxr", "soxr", "soxr.soxr_ext"),
)
TRIM_PATTERN = re.compile(
    r"\[LocalASR\]\[Trim\] channel=(?P<channel>\S+) "
    r"model=(?P<model>\S+) backend=(?P<backend>\S+) "
    r"audio_before_seconds=(?P<audio_before_seconds>\d+\.\d+) "
    r"reported_trailing_silence_seconds=(?P<reported>none|\d+\.\d+) "
    r"actual_trimmed_seconds=(?P<actual_trimmed_seconds>\d+\.\d+) "
    r"submitted_audio_seconds=(?P<submitted_audio_seconds>\d+\.\d+)"
)


class _LocalASRDiagnosticHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.attempt_messages: list[str] = []
        self.trim_messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if "[LocalASR][Attempt]" in message:
            self.attempt_messages.append(message)
        if "[LocalASR][Trim]" in message:
            self.trim_messages.append(message)


@dataclass(frozen=True, slots=True)
class _ObservedDecode:
    text: str
    payload: dict[str, object]


class _AsyncClosable(Protocol):
    async def close(self) -> None: ...


def _diagnostics_enabled() -> bool:
    return True


async def _close_resources(
    resources: Sequence[_AsyncClosable | None],
    *,
    primary_error: BaseException | None,
) -> None:
    first_cleanup_error: BaseException | None = None
    for resource in resources:
        if resource is None:
            continue
        try:
            await resource.close()
        except BaseException as exc:
            if first_cleanup_error is None:
                first_cleanup_error = exc
    if first_cleanup_error is None:
        return
    if primary_error is not None:
        primary_error.add_note(
            f"resource cleanup also failed: {type(first_cleanup_error).__name__}: "
            f"{first_cleanup_error}"
        )
        return
    raise first_cleanup_error


def _samples_sha256(samples_f32: np.ndarray) -> str:
    samples = np.ascontiguousarray(samples_f32, dtype=np.float32)
    return hashlib.sha256(samples.tobytes()).hexdigest()


def _git(
    repo_root: Path | None,
    *args: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root or Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "git command failed"
        raise RuntimeError(message)
    return result


def _source_file_identity(
    repo_root: Path,
    candidate_tree: str,
    relative_path: str,
) -> dict[str, object]:
    source_path = repo_root / relative_path
    if not source_path.is_file():
        raise RuntimeError(f"required evidence source is missing: {relative_path}")
    blob = _git(repo_root, "rev-parse", f"{candidate_tree}:{relative_path}").stdout.strip()
    return {
        "path": relative_path,
        "git_blob_sha1": blob,
        "sha256": _sha256(source_path),
    }


def _source_identity(candidate_tree: str) -> dict[str, object]:
    normalized_tree = candidate_tree.strip().lower()
    if GIT_TREE_PATTERN.fullmatch(normalized_tree) is None:
        raise ValueError("candidate tree must be a lowercase 40-character Git tree ID")
    repo_result = _git(None, "rev-parse", "--show-toplevel")
    repo_root = Path(repo_result.stdout.strip()).resolve()
    unstaged = _git(
        repo_root,
        "diff",
        "--quiet",
        "--no-ext-diff",
        "--submodule=diff",
        "--",
        check=False,
    )
    if unstaged.returncode == 1:
        raise RuntimeError("tracked worktree differs from the staged candidate")
    if unstaged.returncode != 0:
        raise RuntimeError(unstaged.stderr.strip() or "unable to validate tracked worktree")
    actual_tree = _git(repo_root, "write-tree").stdout.strip().lower()
    if actual_tree != normalized_tree:
        raise RuntimeError(
            f"staged candidate tree mismatch: expected {normalized_tree}, actual {actual_tree}"
        )
    return {
        "repository_root": str(repo_root),
        "head_commit": _git(repo_root, "rev-parse", "HEAD").stdout.strip(),
        "candidate_tree": actual_tree,
        "tracked_worktree_matches_index": True,
        "staged_paths": _git(
            repo_root,
            "diff",
            "--cached",
            "--name-only",
            "--diff-filter=ACDMRTUXB",
        ).stdout.splitlines(),
        "execution_sources": [
            _source_file_identity(repo_root, actual_tree, relative_path)
            for relative_path in SOURCE_PATHS
        ],
    }


def _authority_identity(
    authority_ref: str,
    authority_pin: str,
    snapshot_path: Path,
) -> dict[str, object]:
    if not snapshot_path.is_file():
        raise FileNotFoundError(snapshot_path)
    content = snapshot_path.read_text(encoding="utf-8")
    sections = content.split("---", 2)
    if len(sections) != 3 or sections[0].strip():
        raise RuntimeError("authority snapshot has invalid frontmatter")
    frontmatter = sections[1]

    def value(key: str) -> str:
        match = re.search(rf"(?m)^{re.escape(key)}:\s*(.+?)\s*$", frontmatter)
        if match is None:
            raise RuntimeError(f"authority snapshot is missing {key}")
        return match.group(1)

    snapshot_ref = value("authority_ref")
    document_ref = value("document_ref")
    snapshot_pin = value("authority_pin").lower()
    document_pin = value("document_sha256").lower()
    if snapshot_ref != authority_ref or document_ref != authority_ref:
        raise RuntimeError("authority snapshot reference does not match invocation")
    if snapshot_pin != authority_pin or document_pin != authority_pin:
        raise RuntimeError("authority snapshot pin does not match invocation")
    return {
        "ref": authority_ref,
        "pin_sha256": authority_pin,
        "snapshot_path": str(snapshot_path),
        "snapshot_sha256": _sha256(snapshot_path),
        "document_kind": value("document_kind"),
        "document_node_id": value("document_node_id"),
        "document_updated_at": value("document_updated_at"),
    }


def _is_native_runtime_artifact(path: Path) -> bool:
    name = path.name.lower()
    return (
        path.suffix.lower() in {".dll", ".pyd", ".dylib"} or name.endswith(".so") or ".so." in name
    )


def _native_runtime_identity(
    package_module: str,
    distribution_name: str,
    loaded_module: str,
) -> dict[str, object]:
    package = importlib.import_module(package_module)
    package_origin = Path(str(package.__file__)).resolve()
    loaded = importlib.import_module(loaded_module)
    loaded_origin = Path(str(loaded.__file__)).resolve()
    native_artifacts = sorted(
        path
        for path in package_origin.parent.rglob("*")
        if path.is_file() and _is_native_runtime_artifact(path)
    )
    if not native_artifacts or loaded_origin not in native_artifacts:
        raise RuntimeError(f"native runtime artifacts are incomplete: {package_module}")
    return {
        "module": package_module,
        "distribution": distribution_name,
        "version": importlib.metadata.version(distribution_name),
        "package_origin": str(package_origin),
        "package_origin_sha256": _sha256(package_origin),
        "loaded_module": loaded_module,
        "loaded_artifact": str(loaded_origin),
        "loaded_artifact_sha256": _sha256(loaded_origin),
        "native_artifacts": [
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in native_artifacts
        ],
    }


def _invocation_payload(
    *,
    model_root: Path,
    audio_root: Path,
    gpu_worker_path: Path,
    gpu_device_id: str,
    report_path: Path,
    repetitions: int,
    authority_ref: str,
    authority_pin: str,
    authority_snapshot_path: Path,
    candidate_tree: str,
) -> dict[str, object]:
    command = [
        str(Path(sys.executable).resolve()),
        "-m",
        MODULE_NAME,
        "--model-root",
        str(model_root),
        "--audio-root",
        str(audio_root),
        "--gpu-worker",
        str(gpu_worker_path),
        "--gpu-device-id",
        gpu_device_id,
        "--report",
        str(report_path),
        "--repetitions",
        str(repetitions),
        "--authority-ref",
        authority_ref,
        "--authority-pin",
        authority_pin,
        "--authority-snapshot",
        str(authority_snapshot_path),
        "--candidate-tree",
        candidate_tree,
    ]
    return {
        "module": MODULE_NAME,
        "command": command,
        "parameters": {
            "model_root": str(model_root),
            "audio_root": str(audio_root),
            "gpu_worker": str(gpu_worker_path),
            "gpu_device_id": gpu_device_id,
            "report": str(report_path),
            "repetitions": repetitions,
            "authority_ref": authority_ref,
            "authority_pin": authority_pin,
            "authority_snapshot": str(authority_snapshot_path),
            "candidate_tree": candidate_tree,
        },
    }


def _publish_report_atomic(report_path: Path, report: Mapping[str, object]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=report_path.parent,
            prefix=f".{report_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, report_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _with_fixed_trailing_silence(samples_f32: np.ndarray) -> np.ndarray:
    trailing_samples = REPORTED_TRAILING_SILENCE_MS * 16_000 // 1000
    return np.concatenate(
        (
            np.asarray(samples_f32, dtype=np.float32).reshape(-1),
            np.zeros(trailing_samples, dtype=np.float32),
        )
    )


def _trim_payload(message: str, expected_model_id: str) -> dict[str, object]:
    match = TRIM_PATTERN.search(message)
    if match is None:
        raise RuntimeError("local ASR trim diagnostic was not emitted")
    payload: dict[str, object] = dict(match.groupdict())
    for key in (
        "audio_before_seconds",
        "actual_trimmed_seconds",
        "submitted_audio_seconds",
    ):
        payload[key] = float(str(payload[key]))
    reported = payload.pop("reported")
    payload["reported_trailing_silence_seconds"] = (
        None if reported == "none" else float(str(reported))
    )
    if payload["model"] != expected_model_id:
        raise RuntimeError("trim diagnostic model identity mismatch")
    return payload


def _text_payload(text: str) -> dict[str, object]:
    encoded = text.encode("utf-8")
    ending = text[-1:] if text else ""
    return {
        "empty": not bool(text),
        "text_length": len(text),
        "text_sha256": hashlib.sha256(encoded).hexdigest(),
        "ending_sha256": hashlib.sha256(ending.encode("utf-8")).hexdigest(),
    }


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload[key]
    if not isinstance(value, Mapping):
        raise RuntimeError(f"evidence payload field is not a mapping: {key}")
    return value


def _comparison(
    baseline: _ObservedDecode,
    trimmed: _ObservedDecode,
) -> dict[str, object]:
    baseline_text = baseline.text
    trimmed_text = trimmed.text
    transcript_equal = baseline_text == trimmed_text
    ending_preserved = (
        bool(baseline_text) and bool(trimmed_text) and (baseline_text[-1] == trimmed_text[-1])
    )
    terminal_deletion = (
        bool(baseline_text)
        and bool(trimmed_text)
        and len(trimmed_text) < len(baseline_text)
        and baseline_text.startswith(trimmed_text)
    )
    baseline_submitted = float(_mapping(baseline.payload, "trim")["submitted_audio_seconds"])
    trimmed_submitted = float(_mapping(trimmed.payload, "trim")["submitted_audio_seconds"])
    return {
        "transcript_equal": transcript_equal,
        "ending_preserved": ending_preserved,
        "terminal_deletion_regression": terminal_deletion,
        "submitted_audio_reduction_seconds": baseline_submitted - trimmed_submitted,
    }


def _summarize_pairs(pairs: Sequence[dict[str, object]]) -> dict[str, object]:
    comparisons = [_mapping(pair, "comparison") for pair in pairs]
    baseline_results = [_mapping(pair, "baseline") for pair in pairs]
    trimmed_results = [_mapping(pair, "trimmed") for pair in pairs]
    baseline_empty_count = sum(
        bool(_mapping(result, "transcript")["empty"]) for result in baseline_results
    )
    trimmed_empty_count = sum(
        bool(_mapping(result, "transcript")["empty"]) for result in trimmed_results
    )
    reductions = [
        float(comparison["submitted_audio_reduction_seconds"]) for comparison in comparisons
    ]
    baseline_decode = [
        float(_mapping(result, "attempt")["decode_seconds"]) for result in baseline_results
    ]
    trimmed_decode = [
        float(_mapping(result, "attempt")["decode_seconds"]) for result in trimmed_results
    ]
    baseline_queue = [
        float(_mapping(result, "attempt")["queue_wait_seconds"]) for result in baseline_results
    ]
    trimmed_queue = [
        float(_mapping(result, "attempt")["queue_wait_seconds"]) for result in trimmed_results
    ]
    all_transcripts_equal = all(bool(comparison["transcript_equal"]) for comparison in comparisons)
    all_endings_preserved = all(bool(comparison["ending_preserved"]) for comparison in comparisons)
    terminal_deletion_regressions = sum(
        bool(comparison["terminal_deletion_regression"]) for comparison in comparisons
    )
    submitted_reduction_matches_policy = all(
        math.isclose(
            reduction,
            EXPECTED_TRIMMED_MS / 1000.0,
            abs_tol=0.001,
        )
        for reduction in reductions
    )
    passed = (
        all_transcripts_equal
        and all_endings_preserved
        and terminal_deletion_regressions == 0
        and baseline_empty_count == 0
        and trimmed_empty_count == 0
        and submitted_reduction_matches_policy
    )
    return {
        "status": "passed" if passed else "failed",
        "repetitions": len(pairs),
        "all_transcripts_equal": all_transcripts_equal,
        "all_endings_preserved": all_endings_preserved,
        "terminal_deletion_regressions": terminal_deletion_regressions,
        "baseline_empty_transcript_rate": baseline_empty_count / len(pairs),
        "trimmed_empty_transcript_rate": trimmed_empty_count / len(pairs),
        "submitted_audio_reduction_seconds": reductions,
        "submitted_reduction_matches_policy": submitted_reduction_matches_policy,
        "baseline_median_decode_seconds": statistics.median(baseline_decode),
        "trimmed_median_decode_seconds": statistics.median(trimmed_decode),
        "baseline_median_queue_wait_seconds": statistics.median(baseline_queue),
        "trimmed_median_queue_wait_seconds": statistics.median(trimmed_queue),
    }


def _pair_payload(
    repetition: int,
    baseline: _ObservedDecode,
    trimmed: _ObservedDecode,
) -> dict[str, object]:
    return {
        "repetition": repetition,
        "baseline": baseline.payload,
        "trimmed": trimmed.payload,
        "comparison": _comparison(baseline, trimmed),
    }


async def _decode_cpu_mode(
    *,
    session: STTBackendSession,
    events: AsyncIterator[STTBackendTranscriptEvent],
    samples_f32: np.ndarray,
    model_id: str,
    trailing_silence_ms: int | None,
    handler: _LocalASRDiagnosticHandler,
) -> _ObservedDecode:
    attempt_offset = len(handler.attempt_messages)
    trim_offset = len(handler.trim_messages)
    await session.send_audio_f32(samples_f32)
    await session.on_speech_end(trailing_silence_ms=trailing_silence_ms)
    event = await asyncio.wait_for(anext(events), timeout=300)
    attempt_message = next(
        message
        for message in handler.attempt_messages[attempt_offset:]
        if f"model={model_id}" in message
    )
    trim_message = next(
        message for message in handler.trim_messages[trim_offset:] if f"model={model_id}" in message
    )
    return _ObservedDecode(
        text=event.text,
        payload={
            "transcript": _text_payload(event.text),
            "attempt": _attempt_payload(attempt_message, model_id),
            "trim": _trim_payload(trim_message, model_id),
        },
    )


async def _wait_for_gpu_attempt(
    diagnostics: list[GpuASRDiagnostic],
    offset: int,
) -> GpuASRDiagnostic:
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        for diagnostic in diagnostics[offset:]:
            if diagnostic.kind == "decode_attempt":
                return diagnostic
        await asyncio.sleep(0.01)
    raise RuntimeError("GPU decode attempt diagnostic was not emitted")


async def _decode_gpu_mode(
    *,
    session: STTBackendSession,
    events: AsyncIterator[STTBackendTranscriptEvent],
    samples_f32: np.ndarray,
    model_id: str,
    trailing_silence_ms: int | None,
    handler: _LocalASRDiagnosticHandler,
    diagnostics: list[GpuASRDiagnostic],
) -> _ObservedDecode:
    trim_offset = len(handler.trim_messages)
    diagnostic_offset = len(diagnostics)
    await session.send_audio_f32(samples_f32)
    await session.on_speech_end(trailing_silence_ms=trailing_silence_ms)
    event = await asyncio.wait_for(anext(events), timeout=300)
    attempt_diagnostic = await _wait_for_gpu_attempt(diagnostics, diagnostic_offset)
    trim_message = next(
        message for message in handler.trim_messages[trim_offset:] if f"model={model_id}" in message
    )
    attempt = dict(attempt_diagnostic.fields)
    for key in ("audio_seconds", "decode_seconds", "rtf", "queue_wait_seconds"):
        attempt[key] = float(attempt[key])
    if attempt.get("model") != model_id or attempt.get("backend") != "Vulkan":
        raise RuntimeError("GPU attempt diagnostic identity mismatch")
    return _ObservedDecode(
        text=event.text,
        payload={
            "transcript": _text_payload(event.text),
            "attempt": attempt,
            "trim": _trim_payload(trim_message, model_id),
        },
    )


async def _run_cpu_case(
    case: DecodeCase,
    *,
    model_root: Path,
    audio_root: Path,
    repetitions: int,
    handler: _LocalASRDiagnosticHandler,
) -> dict[str, object]:
    audio_path = (audio_root / case.audio_filename).resolve()
    source_samples, source_rate_hz = _read_audio(audio_path)
    samples_16k = _resample(source_samples, source_rate_hz)
    fixed_samples = _with_fixed_trailing_silence(samples_16k)
    backend = create_local_cpu_backend(
        case.model_id,
        model_root=model_root,
        source_language=case.source_language,
        sample_rate_hz=16_000,
        stream_label="evidence",
        diagnostics_enabled=_diagnostics_enabled,
    )
    session = None
    load_started = time.perf_counter()
    try:
        session = await backend.open_session()
        load_seconds = time.perf_counter() - load_started
        events = session.events()
        pairs: list[dict[str, object]] = []
        for index in range(repetitions):
            if index % 2 == 0:
                baseline = await _decode_cpu_mode(
                    session=session,
                    events=events,
                    samples_f32=fixed_samples,
                    model_id=case.model_id,
                    trailing_silence_ms=None,
                    handler=handler,
                )
                trimmed = await _decode_cpu_mode(
                    session=session,
                    events=events,
                    samples_f32=fixed_samples,
                    model_id=case.model_id,
                    trailing_silence_ms=REPORTED_TRAILING_SILENCE_MS,
                    handler=handler,
                )
            else:
                trimmed = await _decode_cpu_mode(
                    session=session,
                    events=events,
                    samples_f32=fixed_samples,
                    model_id=case.model_id,
                    trailing_silence_ms=REPORTED_TRAILING_SILENCE_MS,
                    handler=handler,
                )
                baseline = await _decode_cpu_mode(
                    session=session,
                    events=events,
                    samples_f32=fixed_samples,
                    model_id=case.model_id,
                    trailing_silence_ms=None,
                    handler=handler,
                )
            pairs.append(_pair_payload(index + 1, baseline, trimmed))
        return {
            "model_id": case.model_id,
            "execution_backend": "CPU",
            "source_language": case.source_language,
            "model_load_seconds": load_seconds,
            "fixture": _fixture_payload(
                case,
                audio_path=audio_path,
                source_rate_hz=source_rate_hz,
                samples_16k=samples_16k,
                fixed_samples=fixed_samples,
            ),
            "pairs": pairs,
            "summary": _summarize_pairs(pairs),
        }
    finally:
        await _close_resources(
            (session, backend),
            primary_error=sys.exc_info()[1],
        )


def _fixture_payload(
    case: DecodeCase,
    *,
    audio_path: Path,
    source_rate_hz: int,
    samples_16k: np.ndarray,
    fixed_samples: np.ndarray,
) -> dict[str, object]:
    return {
        "filename": case.audio_filename,
        "source_url": case.source_url,
        "file_sha256": _sha256(audio_path),
        "source_sample_rate_hz": source_rate_hz,
        "decode_sample_rate_hz": 16_000,
        "source_audio_seconds": samples_16k.size / 16_000.0,
        "appended_trailing_silence_ms": REPORTED_TRAILING_SILENCE_MS,
        "fixed_input_seconds": fixed_samples.size / 16_000.0,
        "fixed_input_f32_sha256": _samples_sha256(fixed_samples),
    }


async def _run_gpu_case(
    case: DecodeCase,
    *,
    model_root: Path,
    audio_root: Path,
    gpu_worker_path: Path,
    gpu_device_id: str,
    repetitions: int,
    handler: _LocalASRDiagnosticHandler,
) -> dict[str, object]:
    audio_path = (audio_root / case.audio_filename).resolve()
    source_samples, source_rate_hz = _read_audio(audio_path)
    samples_16k = _resample(source_samples, source_rate_hz)
    fixed_samples = _with_fixed_trailing_silence(samples_16k)
    diagnostics: list[GpuASRDiagnostic] = []
    runtime = _create_shared_gpu_asr_runtime(
        process_factory=DefaultGpuWorkerProcessFactory(executable_path=gpu_worker_path),
        diagnostic_sink=diagnostics.append,
    )
    devices = await runtime.discover_devices()
    backend = LocalGpuSTTBackend(
        runtime=runtime,
        channel="self",
        model_path=local_gpu_model_path(model_root),
        model_id=LOCAL_QWEN_GPU_MODEL_ID,
        device_id=gpu_device_id,
        sample_rate_hz=16_000,
        source_mode="manual",
        language_hint=get_local_qwen_language_hint(case.source_language),
        diagnostics_enabled=_diagnostics_enabled,
    )
    session = None
    load_started = time.perf_counter()
    try:
        session = await backend.open_session()
        load_seconds = time.perf_counter() - load_started
        activation = next(
            diagnostic for diagnostic in diagnostics if diagnostic.kind == "activation_ready"
        )
        events = session.events()
        pairs: list[dict[str, object]] = []
        for index in range(repetitions):
            if index % 2 == 0:
                baseline = await _decode_gpu_mode(
                    session=session,
                    events=events,
                    samples_f32=fixed_samples,
                    model_id=LOCAL_QWEN_GPU_MODEL_ID,
                    trailing_silence_ms=None,
                    handler=handler,
                    diagnostics=diagnostics,
                )
                trimmed = await _decode_gpu_mode(
                    session=session,
                    events=events,
                    samples_f32=fixed_samples,
                    model_id=LOCAL_QWEN_GPU_MODEL_ID,
                    trailing_silence_ms=REPORTED_TRAILING_SILENCE_MS,
                    handler=handler,
                    diagnostics=diagnostics,
                )
            else:
                trimmed = await _decode_gpu_mode(
                    session=session,
                    events=events,
                    samples_f32=fixed_samples,
                    model_id=LOCAL_QWEN_GPU_MODEL_ID,
                    trailing_silence_ms=REPORTED_TRAILING_SILENCE_MS,
                    handler=handler,
                    diagnostics=diagnostics,
                )
                baseline = await _decode_gpu_mode(
                    session=session,
                    events=events,
                    samples_f32=fixed_samples,
                    model_id=LOCAL_QWEN_GPU_MODEL_ID,
                    trailing_silence_ms=None,
                    handler=handler,
                    diagnostics=diagnostics,
                )
            pairs.append(_pair_payload(index + 1, baseline, trimmed))
        return {
            "model_id": LOCAL_QWEN_GPU_MODEL_ID,
            "execution_backend": "Vulkan",
            "source_language": case.source_language,
            "model_load_seconds": load_seconds,
            "device_request": gpu_device_id,
            "devices": [
                {
                    "device_id": device.device_id,
                    "name": device.name,
                    "device_type": device.device_type,
                    "memory_total_bytes": device.memory_total_bytes,
                    "memory_free_bytes": device.memory_free_bytes,
                }
                for device in devices
            ],
            "activation": dict(activation.fields),
            "fixture": _fixture_payload(
                case,
                audio_path=audio_path,
                source_rate_hz=source_rate_hz,
                samples_16k=samples_16k,
                fixed_samples=fixed_samples,
            ),
            "pairs": pairs,
            "summary": _summarize_pairs(pairs),
        }
    finally:
        await _close_resources(
            (session, backend, runtime),
            primary_error=sys.exc_info()[1],
        )


def _validated_installs(model_root: Path) -> list[dict[str, object]]:
    cpu_snapshot = inspect_required_cpu_model_installs(model_root, verify_checksums=True)
    if not cpu_snapshot.cpu_auto_available:
        raise RuntimeError("strict validation did not accept all required CPU models")
    gpu_snapshot = inspect_local_gpu_install(
        explicit_opt_in=True,
        model_root=model_root,
        verify_checksums=True,
    )
    if not gpu_snapshot.activation_allowed:
        raise RuntimeError("strict validation did not accept the GPU model")
    states = {model.model_id: model.state for model in cpu_snapshot.models}
    if gpu_snapshot.state is None:
        raise RuntimeError("strict validation returned no GPU install state")
    states[LOCAL_QWEN_GPU_MODEL_ID] = gpu_snapshot.state
    installs: list[dict[str, object]] = []
    for model_id, state in states.items():
        manifest = load_local_stt_asset_manifest(model_id)
        installed = state.installed_manifest
        if installed is None:
            raise RuntimeError("strict validation returned no installed manifest")
        model_dir = model_root / manifest.install_dirname
        files: list[dict[str, object]] = []
        for item in manifest.files:
            path = model_dir / item.relative_path
            actual_sha256 = _sha256(path)
            actual_size = path.stat().st_size
            if actual_sha256 != item.sha256:
                raise RuntimeError(f"model checksum changed after validation: {model_id}")
            if item.size_bytes is not None and actual_size != item.size_bytes:
                raise RuntimeError(f"model size changed after validation: {model_id}")
            files.append(
                {
                    "relative_path": item.relative_path,
                    "size_bytes": actual_size,
                    "sha256": actual_sha256,
                }
            )
        installs.append(
            {
                "model_id": model_id,
                "engine": manifest.engine,
                "upstream_repo": manifest.upstream_repo,
                "install_dir": str(model_dir.resolve()),
                "installed_manifest": installed.to_dict(),
                "files": files,
            }
        )
    return installs


async def run_evidence(
    *,
    model_root: Path,
    audio_root: Path,
    gpu_worker_path: Path,
    gpu_device_id: str,
    report_path: Path,
    repetitions: int,
    authority_ref: str,
    authority_pin: str,
    authority_snapshot_path: Path,
    candidate_tree: str,
) -> int:
    resolved_report_path = report_path.resolve()
    resolved_report_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_report_path.unlink(missing_ok=True)
    try:
        return await _execute_evidence(
            model_root=model_root,
            audio_root=audio_root,
            gpu_worker_path=gpu_worker_path,
            gpu_device_id=gpu_device_id,
            report_path=resolved_report_path,
            repetitions=repetitions,
            authority_ref=authority_ref,
            authority_pin=authority_pin,
            authority_snapshot_path=authority_snapshot_path,
            candidate_tree=candidate_tree,
        )
    except BaseException:
        resolved_report_path.unlink(missing_ok=True)
        raise


async def _execute_evidence(
    *,
    model_root: Path,
    audio_root: Path,
    gpu_worker_path: Path,
    gpu_device_id: str,
    report_path: Path,
    repetitions: int,
    authority_ref: str,
    authority_pin: str,
    authority_snapshot_path: Path,
    candidate_tree: str,
) -> int:
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    normalized_authority_ref = authority_ref.strip()
    normalized_authority_pin = authority_pin.strip().lower()
    normalized_candidate_tree = candidate_tree.strip().lower()
    if not normalized_authority_ref:
        raise ValueError("authority reference must be nonempty")
    if AUTHORITY_PIN_PATTERN.fullmatch(normalized_authority_pin) is None:
        raise ValueError("authority pin must be a lowercase 64-character SHA-256 digest")
    resolved_model_root = model_root.resolve()
    resolved_audio_root = audio_root.resolve()
    resolved_gpu_worker_path = gpu_worker_path.resolve()
    resolved_authority_snapshot_path = authority_snapshot_path.resolve()
    if not resolved_gpu_worker_path.is_file():
        raise FileNotFoundError(resolved_gpu_worker_path)
    authority_identity = _authority_identity(
        normalized_authority_ref,
        normalized_authority_pin,
        resolved_authority_snapshot_path,
    )
    source_identity = _source_identity(normalized_candidate_tree)
    gpu_worker_sha256 = _sha256(resolved_gpu_worker_path)
    started_at = time.time()
    validation_started = time.perf_counter()
    installs = _validated_installs(resolved_model_root)
    validation_seconds = time.perf_counter() - validation_started
    handler = _LocalASRDiagnosticHandler()
    cpu_logger = logging.getLogger("puripuly_heart.providers.stt.local_qwen_sherpa")
    gpu_logger = logging.getLogger("puripuly_heart.providers.stt.local_gpu")
    previous_levels = (cpu_logger.level, gpu_logger.level)
    cpu_logger.addHandler(handler)
    gpu_logger.addHandler(handler)
    cpu_logger.setLevel(logging.INFO)
    gpu_logger.setLevel(logging.INFO)
    try:
        model_results = [
            await _run_cpu_case(
                case,
                model_root=resolved_model_root,
                audio_root=resolved_audio_root,
                repetitions=repetitions,
                handler=handler,
            )
            for case in DECODE_CASES
        ]
        model_results.append(
            await _run_gpu_case(
                DECODE_CASES[-1],
                model_root=resolved_model_root,
                audio_root=resolved_audio_root,
                gpu_worker_path=resolved_gpu_worker_path,
                gpu_device_id=gpu_device_id,
                repetitions=repetitions,
                handler=handler,
            )
        )
    finally:
        cpu_logger.removeHandler(handler)
        gpu_logger.removeHandler(handler)
        cpu_logger.setLevel(previous_levels[0])
        gpu_logger.setLevel(previous_levels[1])
    if _source_identity(normalized_candidate_tree) != source_identity:
        raise RuntimeError("candidate source identity changed during evidence execution")
    if (
        _authority_identity(
            normalized_authority_ref,
            normalized_authority_pin,
            resolved_authority_snapshot_path,
        )
        != authority_identity
    ):
        raise RuntimeError("authority identity changed during evidence execution")
    if _sha256(resolved_gpu_worker_path) != gpu_worker_sha256:
        raise RuntimeError("GPU worker identity changed during evidence execution")
    passed = all(_mapping(result, "summary")["status"] == "passed" for result in model_results)
    executable = Path(sys.executable).resolve()
    report = {
        "schema": REPORT_SCHEMA,
        "status": "passed" if passed else "failed",
        "started_unix_seconds": started_at,
        "completed_unix_seconds": time.time(),
        "authority": authority_identity,
        "source": source_identity,
        "invocation": _invocation_payload(
            model_root=resolved_model_root,
            audio_root=resolved_audio_root,
            gpu_worker_path=resolved_gpu_worker_path,
            gpu_device_id=gpu_device_id,
            report_path=report_path,
            repetitions=repetitions,
            authority_ref=normalized_authority_ref,
            authority_pin=normalized_authority_pin,
            authority_snapshot_path=resolved_authority_snapshot_path,
            candidate_tree=normalized_candidate_tree,
        ),
        "application": {
            "version": __version__,
            "executable": str(executable),
            "executable_sha256": _sha256(executable),
            "frozen": bool(getattr(sys, "frozen", False)),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "native_runtimes": [
            _native_runtime_identity(package, distribution, loaded)
            for package, distribution, loaded in NATIVE_RUNTIME_MODULES
        ],
        "gpu_worker": {
            "path": str(resolved_gpu_worker_path),
            "size_bytes": resolved_gpu_worker_path.stat().st_size,
            "sha256": gpu_worker_sha256,
        },
        "model_root": str(resolved_model_root),
        "audio_root": str(resolved_audio_root),
        "strict_validation_seconds": validation_seconds,
        "policy": {
            "reported_trailing_silence_ms": REPORTED_TRAILING_SILENCE_MS,
            "safety_tail_ms": SAFETY_TAIL_MS,
            "expected_trimmed_ms": EXPECTED_TRIMMED_MS,
        },
        "repetitions": repetitions,
        "model_installs": installs,
        "models": model_results,
    }
    _publish_report_atomic(report_path, report)
    return 0 if passed else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--gpu-worker", type=Path, required=True)
    parser.add_argument("--gpu-device-id", default="auto")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--authority-ref", required=True)
    parser.add_argument("--authority-pin", required=True)
    parser.add_argument("--authority-snapshot", type=Path, required=True)
    parser.add_argument("--candidate-tree", required=True)
    args = parser.parse_args(argv)
    return asyncio.run(
        run_evidence(
            model_root=args.model_root,
            audio_root=args.audio_root,
            gpu_worker_path=args.gpu_worker,
            gpu_device_id=args.gpu_device_id,
            report_path=args.report,
            repetitions=args.repetitions,
            authority_ref=args.authority_ref,
            authority_pin=args.authority_pin,
            authority_snapshot_path=args.authority_snapshot,
            candidate_tree=args.candidate_tree,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
