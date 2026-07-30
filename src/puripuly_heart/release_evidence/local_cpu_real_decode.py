from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import importlib
import json
import logging
import math
import platform
import re
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from puripuly_heart import __version__
from puripuly_heart.core.local_stt_assets import (
    LOCAL_STT_MODEL_ID,
    PARAKEET_JAPANESE_MODEL_ID,
    PARAKEET_V3_MODEL_ID,
    load_local_stt_asset_manifest,
)
from puripuly_heart.core.local_stt_catalog import inspect_required_cpu_model_installs
from puripuly_heart.providers.stt.local_cpu import create_local_cpu_backend

REPORT_SCHEMA = "puripuly-heart/local-cpu-real-decode/v1"
ATTEMPT_PATTERN = re.compile(
    r"\[LocalASR\]\[Attempt\] channel=(?P<channel>\S+) "
    r"model=(?P<model>\S+) backend=(?P<backend>\S+) "
    r"audio_seconds=(?P<audio_seconds>\d+\.\d+) "
    r"decode_seconds=(?P<decode_seconds>\d+\.\d+) "
    r"rtf=(?P<rtf>\d+\.\d+) result=(?P<result>\S+) "
    r"queue_wait_seconds=(?P<queue_wait_seconds>\d+\.\d+)"
)


@dataclass(frozen=True, slots=True)
class DecodeCase:
    model_id: str
    source_language: str
    audio_filename: str
    source_url: str


DECODE_CASES = (
    DecodeCase(
        model_id=PARAKEET_V3_MODEL_ID,
        source_language="en",
        audio_filename="parakeet-v3-en.wav",
        source_url=(
            "https://huggingface.co/csukuangfj/"
            "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/resolve/"
            "2bda32ec70b097a55adaa07d9a7173915b43cc78/test_wavs/en.wav"
        ),
    ),
    DecodeCase(
        model_id=PARAKEET_JAPANESE_MODEL_ID,
        source_language="ja",
        audio_filename="parakeet-ja.wav",
        source_url=(
            "https://huggingface.co/csukuangfj/"
            "sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8/resolve/"
            "bef18eb066808c90bd0f5df5be685767b0732de8/test_wavs/test_ja_1.wav"
        ),
    ),
    DecodeCase(
        model_id=LOCAL_STT_MODEL_ID,
        source_language="de",
        audio_filename="qwen-de.wav",
        source_url=(
            "https://huggingface.co/csukuangfj2/"
            "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/resolve/"
            "2cc50d1abfe4d4f2df8d71f536d108bb40f943d2/test_wavs/de.wav"
        ),
    ),
)


class _AttemptHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if "[LocalASR][Attempt]" in message:
            self.messages.append(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        sample_width = source.getsampwidth()
        sample_rate_hz = source.getframerate()
        frames = source.readframes(source.getnframes())
    if sample_width != 2:
        raise RuntimeError(f"unsupported PCM sample width: {sample_width}")
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, sample_rate_hz


def _resample(samples: np.ndarray, source_rate_hz: int) -> np.ndarray:
    if source_rate_hz == 16000:
        return np.asarray(samples, dtype=np.float32)
    soxr = importlib.import_module("soxr")
    return np.asarray(soxr.resample(samples, source_rate_hz, 16000), dtype=np.float32)


def _runtime_module_identity(module_name: str) -> dict[str, object]:
    module = importlib.import_module(module_name)
    raw_origin = getattr(module, "__file__", None)
    if not raw_origin:
        return {"module": module_name, "origin": None, "sha256": None}
    origin = Path(str(raw_origin)).resolve()
    artifact = origin if origin.is_file() else Path(sys.executable).resolve()
    return {
        "module": module_name,
        "origin": str(origin),
        "artifact": str(artifact),
        "sha256": _sha256(artifact),
    }


def _evidence_diagnostics_enabled() -> bool:
    return True


def _attempt_payload(message: str, expected_model_id: str) -> dict[str, object]:
    match = ATTEMPT_PATTERN.search(message)
    if match is None:
        raise RuntimeError("local CPU attempt diagnostic was not emitted")
    payload: dict[str, object] = dict(match.groupdict())
    for key in ("audio_seconds", "decode_seconds", "rtf", "queue_wait_seconds"):
        payload[key] = float(str(payload[key]))
    if payload["model"] != expected_model_id:
        raise RuntimeError("attempt diagnostic model identity mismatch")
    if payload["backend"] != "CPU" or payload["result"] != "success":
        raise RuntimeError("attempt diagnostic did not record CPU success")
    audio_seconds = float(payload["audio_seconds"])
    decode_seconds = float(payload["decode_seconds"])
    rtf = float(payload["rtf"])
    if audio_seconds <= 0 or decode_seconds <= 0:
        raise RuntimeError("attempt diagnostic timing is not positive")
    if not math.isclose(rtf, decode_seconds / audio_seconds, rel_tol=0.002, abs_tol=0.002):
        raise RuntimeError("attempt diagnostic RTF is inconsistent")
    return payload


async def _decode_case(
    case: DecodeCase,
    *,
    model_root: Path,
    audio_root: Path,
    handler: _AttemptHandler,
) -> dict[str, object]:
    audio_path = (audio_root / case.audio_filename).resolve()
    samples, source_rate_hz = _read_audio(audio_path)
    samples_16k = _resample(samples, source_rate_hz)
    message_offset = len(handler.messages)
    backend = create_local_cpu_backend(
        case.model_id,
        model_root=model_root,
        source_language=case.source_language,
        sample_rate_hz=16000,
        stream_label="evidence",
        diagnostics_enabled=_evidence_diagnostics_enabled,
    )
    session = None
    load_started = time.perf_counter()
    try:
        session = await backend.open_session()
        load_seconds = time.perf_counter() - load_started
        send_audio_f32 = getattr(session, "send_audio_f32")
        await send_audio_f32(samples_16k)
        await session.on_speech_end()
        event = await asyncio.wait_for(anext(session.events()), timeout=300)
        if not event.is_final or not event.text:
            raise RuntimeError("real local CPU decode did not produce a nonempty final result")
        attempt_message = next(
            message
            for message in handler.messages[message_offset:]
            if f"model={case.model_id}" in message
        )
        attempt = _attempt_payload(attempt_message, case.model_id)
        return {
            "model_id": case.model_id,
            "provider_id": str(getattr(backend, "provider_id")),
            "source_language": case.source_language,
            "execution_backend": "CPU",
            "audio_fixture": {
                "filename": case.audio_filename,
                "source_url": case.source_url,
                "sha256": _sha256(audio_path),
                "source_sample_rate_hz": source_rate_hz,
                "decode_sample_rate_hz": 16000,
                "audio_seconds": samples_16k.size / 16000.0,
            },
            "model_load_seconds": load_seconds,
            "decode_result": {
                "status": "nonempty_final",
                "text_length": len(event.text),
            },
            "attempt": attempt,
        }
    finally:
        if session is not None:
            await session.close()
        close_backend = getattr(backend, "close", None)
        if callable(close_backend):
            await close_backend()
        gc.collect()


async def run_evidence(
    *,
    model_root: Path,
    audio_root: Path,
    report_path: Path,
) -> int:
    resolved_model_root = model_root.resolve()
    resolved_audio_root = audio_root.resolve()
    resolved_report_path = report_path.resolve()
    executable = Path(sys.executable).resolve()
    started_at = time.time()
    validation_started = time.perf_counter()
    snapshot = inspect_required_cpu_model_installs(
        resolved_model_root,
        verify_checksums=True,
    )
    validation_seconds = time.perf_counter() - validation_started
    if not snapshot.cpu_auto_available:
        raise RuntimeError("strict validation did not accept all required CPU models")
    installs: list[dict[str, object]] = []
    for model in snapshot.models:
        manifest = load_local_stt_asset_manifest(model.model_id)
        installed = model.state.installed_manifest
        if installed is None:
            raise RuntimeError("strict validation returned no installed manifest")
        installs.append(
            {
                "model_id": model.model_id,
                "status": model.state.status,
                "selected_source": installed.selected_source,
                "selected_revision": installed.selected_revision,
                "file_count": len(manifest.files),
                "expected_total_bytes": sum(item.size_bytes or 0 for item in manifest.files),
            }
        )
    logger = logging.getLogger("puripuly_heart.providers.stt.local_qwen_sherpa")
    previous_level = logger.level
    handler = _AttemptHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        decodes = [
            await _decode_case(
                case,
                model_root=resolved_model_root,
                audio_root=resolved_audio_root,
                handler=handler,
            )
            for case in DECODE_CASES
        ]
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "passed",
        "started_unix_seconds": started_at,
        "completed_unix_seconds": time.time(),
        "application": {
            "version": __version__,
            "executable": str(executable),
            "executable_sha256": _sha256(executable),
            "frozen": bool(getattr(sys, "frozen", False)),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "runtime_modules": [
            _runtime_module_identity("sherpa_onnx"),
            _runtime_module_identity("onnxruntime"),
            _runtime_module_identity("soxr"),
        ],
        "model_root": str(resolved_model_root),
        "strict_validation_seconds": validation_seconds,
        "cpu_auto_available": snapshot.cpu_auto_available,
        "model_installs": installs,
        "decodes": decodes,
    }
    resolved_report_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    return asyncio.run(
        run_evidence(
            model_root=args.model_root,
            audio_root=args.audio_root,
            report_path=args.report,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
