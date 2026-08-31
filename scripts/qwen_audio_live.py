from __future__ import annotations

import argparse
import array
import asyncio
import hashlib
import json
import math
import os
import re
import struct
import sys
import time
import wave
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from puripuly_heart.providers.stt.qwen_audio import QWEN_AUDIO_MODEL, QwenAudioStreamingSTTBackend

SAMPLE_RATE_HZ = 16000
CHUNK_MS = 100
REGION_ENDPOINTS = {
    "beijing": "wss://dashscope.aliyuncs.com/api-ws/v1/inference",
    "singapore": "wss://dashscope-intl.aliyuncs.com/api-ws/v1/inference",
}
REGION_ENV_KEYS = {
    "beijing": ("ALIBABA_API_KEY_BEIJING", "ALIBABA_API_KEY", "DASHSCOPE_API_KEY"),
    "singapore": ("ALIBABA_API_KEY_SINGAPORE", "ALIBABA_API_KEY", "DASHSCOPE_API_KEY"),
}
REGION_KEYRING_KEYS = {
    "beijing": ("alibaba_api_key_beijing", "alibaba_api_key"),
    "singapore": ("alibaba_api_key_singapore", "alibaba_api_key"),
}
TASK_ID_RE = re.compile(r"\b(?:task|session|run)[_-][0-9a-f]{8,}\b", re.IGNORECASE)
UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f-]{27,}\b", re.IGNORECASE)
SAFE_EXCEPTION_TYPES = {
    "TimeoutError",
    "ValueError",
    "TypeError",
    "RuntimeError",
    "ConnectionError",
    "QwenAudioProtocolError",
    "QwenAudioTaskFailedError",
}
TRANSCRIPT_KEYS = {"terminal_text", "terminal_texts", "fresh_session_terminal_text"}
SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "secret",
    "token",
    "password",
    "authorization",
    "credential",
    "credentials",
    "header",
    "task_id",
    "event_id",
    "item_id",
    "uuid",
}
CREDENTIAL_VALUE_RE = re.compile(
    r"(?:\bbearer\s+\S+|\bsk-[A-Za-z0-9_-]{8,}\b|\bAIza[A-Za-z0-9_-]{20,}\b)",
    re.IGNORECASE,
)


@dataclass(slots=True)
class TaskRecord:
    sequence: int
    task_id: str
    requested_duration_s: float | None = None
    pcm_duration_s: float | None = None
    pcm_bytes_expected: int | None = None
    pcm_chunks_sent: int = 0
    pcm_bytes_sent: int = 0
    pcm_expected_sha256: str | None = None
    pcm_sent_sha256: str | None = None
    pcm_expected_frame_sha256: list[str] | None = field(default=None, repr=False)
    pcm_sent_frame_sha256: list[str] = field(default_factory=list, repr=False)
    pcm_sent_hasher: Any = field(default_factory=hashlib.sha256, repr=False)
    run_task_sent_at: str | None = None
    task_started_at: str | None = None
    finish_task_sent_at: str | None = None
    task_finished_at: str | None = None
    actual_task_duration_s: float | None = None
    transition_latency_s: float | None = None
    usage_duration: float | None = None
    terminal_text: str = ""
    terminal_count: int = 0
    pending_terminal_texts: list[str] = field(default_factory=list, repr=False)
    scenario: str | None = None
    sent_monotonic: float | None = field(default=None, repr=False)
    finish_monotonic: float | None = field(default=None, repr=False)
    started_monotonic: float | None = field(default=None, repr=False)
    finished_monotonic: float | None = field(default=None, repr=False)

    def as_dict(self, *, retain_transcripts: bool = False) -> dict[str, object]:
        result: dict[str, object] = {
            "sequence": self.sequence,
            "requested_duration_s": self.requested_duration_s,
            "pcm_duration_s": self.pcm_duration_s,
            "pcm_bytes_expected": self.pcm_bytes_expected,
            "pcm_chunks_sent": self.pcm_chunks_sent,
            "pcm_bytes_sent": self.pcm_bytes_sent,
            "pcm_expected_sha256": self.pcm_expected_sha256,
            "pcm_sent_sha256": self.pcm_sent_sha256,
            "pcm_expected_frame_sha256": self.pcm_expected_frame_sha256,
            "pcm_sent_frame_sha256": self.pcm_sent_frame_sha256,
            "run_task_sent_at": self.run_task_sent_at,
            "task_started_at": self.task_started_at,
            "finish_task_sent_at": self.finish_task_sent_at,
            "task_finished_at": self.task_finished_at,
            "actual_task_duration_s": self.actual_task_duration_s,
            "transition_latency_s": self.transition_latency_s,
            "usage_duration": self.usage_duration,
            "terminal_count": self.terminal_count,
            "scenario": self.scenario,
        }
        if retain_transcripts:
            result["terminal_text"] = self.terminal_text
        return result


@dataclass(slots=True)
class RecordingWebSocket:
    inner: Any
    sent: list[tuple[float, float, object]] = field(default_factory=list)
    received: list[tuple[float, float, object]] = field(default_factory=list)

    async def send(self, value: object) -> None:
        monotonic = time.monotonic()
        wall = time.time()
        await self.inner.send(value)
        self.sent.append((monotonic, wall, value))

    async def recv(self) -> object:
        value = await self.inner.recv()
        self.received.append((time.monotonic(), time.time(), value))
        return value

    async def close(self) -> None:
        await self.inner.close()


@dataclass(slots=True)
class RecordingFactory:
    socket: RecordingWebSocket | None = None

    async def connect(self, endpoint: str, **kwargs: object) -> RecordingWebSocket:
        import websockets

        inner = await websockets.connect(endpoint, **kwargs)
        self.socket = RecordingWebSocket(inner)
        return self.socket


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def _json_mapping(value: object) -> Mapping[str, object] | None:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, Mapping) else None


def _event_type(message: Mapping[str, object]) -> str:
    header = message.get("header")
    header = header if isinstance(header, Mapping) else {}
    return str(header.get("event") or message.get("event") or "")


def _task_id(message: Mapping[str, object]) -> str:
    header = message.get("header")
    header = header if isinstance(header, Mapping) else {}
    return str(header.get("task_id") or message.get("task_id") or "").strip()


def _sentence_end_texts(message: Mapping[str, object]) -> list[str]:
    found: list[str] = []
    seen_sentence_ids: set[str] = set()

    def visit(value: object) -> None:
        if not isinstance(value, Mapping):
            return
        sentence = value.get("sentence")
        if isinstance(sentence, Mapping) and sentence.get("sentence_end") is True:
            sentence_id = str(sentence.get("sentence_id") or "").strip()
            duplicate = bool(sentence_id and sentence_id in seen_sentence_ids)
            if sentence_id:
                seen_sentence_ids.add(sentence_id)
            text = str(sentence.get("text") or "").strip()
            if text and not duplicate:
                found.append(text)
        for key, child in value.items():
            if key != "header":
                visit(child)

    visit(message)
    return found


def _provider_text(message: Mapping[str, object]) -> str:
    return " ".join(_sentence_end_texts(message))


def _usage_duration(message: Mapping[str, object]) -> tuple[float | None, object | None]:
    def visit(value: object) -> tuple[float | None, object | None]:
        if not isinstance(value, Mapping):
            return None, None
        usage = value.get("usage")
        if isinstance(usage, Mapping) and "duration" in usage:
            raw = usage.get("duration")
            try:
                return float(raw), raw
            except (TypeError, ValueError):
                return None, None
        for key, child in value.items():
            if key == "header":
                continue
            result = visit(child)
            if result[1] is not None:
                return result
        return None, None

    return visit(message)


def _pcm_for_duration(duration_s: float, *, sample_rate_hz: int = SAMPLE_RATE_HZ) -> bytes:
    sample_count = max(int(round(duration_s * sample_rate_hz)), 1)
    return b"".join(
        struct.pack("<h", int(4096 * math.sin(2 * math.pi * 440 * index / sample_rate_hz)))
        for index in range(sample_count)
    )



def _pcm_identity(frames: list[bytes]) -> dict[str, object]:
    digest = hashlib.sha256()
    frame_digests: list[str] = []
    byte_count = 0
    for frame in frames:
        digest.update(frame)
        byte_count += len(frame)
        frame_digests.append(hashlib.sha256(frame).hexdigest())
    return {
        "bytes": byte_count,
        "chunks": len(frames),
        "sha256": digest.hexdigest(),
        "frame_sha256": frame_digests,
    }


def _record_sent_identity(record: TaskRecord) -> dict[str, object]:
    return {
        "bytes": record.pcm_bytes_sent,
        "chunks": record.pcm_chunks_sent,
        "sha256": record.pcm_sent_sha256,
        "frame_sha256": list(record.pcm_sent_frame_sha256),
    }

def _set_expected_identity(record: TaskRecord, pcm: bytes) -> None:
    identity = _pcm_identity(_chunks(pcm))
    record.pcm_bytes_expected = int(identity["bytes"])
    record.pcm_expected_sha256 = str(identity["sha256"])
    record.pcm_expected_frame_sha256 = list(identity["frame_sha256"])



def _chunks(pcm: bytes, *, chunk_ms: int = CHUNK_MS) -> list[bytes]:
    chunk_size = SAMPLE_RATE_HZ * 2 * chunk_ms // 1000
    return [pcm[offset : offset + chunk_size] for offset in range(0, len(pcm), chunk_size)]


def _read_fixture(path: Path) -> bytes:
    with wave.open(str(path), "rb") as audio:
        if audio.getnchannels() != 1 or audio.getsampwidth() != 2:
            raise ValueError("audio fixture must be mono 16-bit PCM")
        sample_rate = audio.getframerate()
        pcm = audio.readframes(audio.getnframes())
    if sample_rate == SAMPLE_RATE_HZ:
        return pcm
    source = array.array("h")
    source.frombytes(pcm)
    if not source:
        raise ValueError("audio fixture is empty")
    target_count = max(int(round(len(source) * SAMPLE_RATE_HZ / sample_rate)), 1)
    target = array.array("h")
    for index in range(target_count):
        position = index * sample_rate / SAMPLE_RATE_HZ
        lower = min(int(position), len(source) - 1)
        upper = min(lower + 1, len(source) - 1)
        fraction = position - lower
        value = round(source[lower] + (source[upper] - source[lower]) * fraction)
        target.append(max(-32768, min(32767, value)))
    return target.tobytes()


def _fit_pcm(pcm: bytes, duration_s: float) -> bytes:
    target = max(int(round(duration_s * SAMPLE_RATE_HZ)) * 2, 2)
    if len(pcm) >= target:
        return pcm[:target]
    repeats = (target + len(pcm) - 1) // len(pcm)
    return (pcm * repeats)[:target]


def _comparison_pcm(fixture: bytes | None) -> bytes:
    return _fit_pcm(fixture, 0.8) if fixture else _pcm_for_duration(0.8)


def _secret_for_region(region: str) -> tuple[str | None, str]:
    for env_name in REGION_ENV_KEYS[region]:
        if os.getenv(env_name):
            return os.environ[env_name], f"env:{env_name}"
    try:
        from puripuly_heart.app.wiring.wiring_secrets_factory import STABLE_KEYRING_SERVICE_NAME
        from puripuly_heart.core.storage.secrets import KeyringSecretStore

        store = KeyringSecretStore(service_name=STABLE_KEYRING_SERVICE_NAME)
        for key_name in REGION_KEYRING_KEYS[region]:
            value = store.get(key_name)
            if value:
                return value, f"keyring:{key_name}"
    except Exception as exc:
        return None, f"keyring-unavailable:{type(exc).__name__}"
    return None, "missing"


def build_cases() -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for duration_s in (0.3, 0.8, 1.2):
        for repetition in (1, 2):
            cases.append(
                {
                    "name": f"short_{duration_s:.1f}s_repeat_{repetition}",
                    "duration_s": duration_s,
                }
            )
    cases.extend(
        [
            {"name": "conversational_2s", "duration_s": 2.0},
            {"name": "conversational_4s", "duration_s": 4.0},
            {"name": "conversational_7s", "duration_s": 7.0},
        ]
    )
    return cases


def _task_records(socket: RecordingWebSocket) -> list[TaskRecord]:
    records: list[TaskRecord] = []
    by_id: dict[str, TaskRecord] = {}
    current: TaskRecord | None = None
    for timestamp, wall_timestamp, value in socket.sent:
        message = _json_mapping(value)
        if message is not None:
            header = message.get("header")
            header = header if isinstance(header, Mapping) else {}
            action = str(header.get("action") or "")
            task_id = _task_id(message)
            if action == "run-task" and task_id:
                current = TaskRecord(
                    sequence=len(records) + 1,
                    task_id=task_id,
                    run_task_sent_at=_iso(wall_timestamp),
                    sent_monotonic=timestamp,
                )
                records.append(current)
                by_id[task_id] = current
            elif action == "finish-task" and task_id in by_id:
                current = by_id[task_id]
                current.finish_task_sent_at = _iso(wall_timestamp)
                current.finish_monotonic = timestamp
        elif isinstance(value, bytes) and current is not None:
            current.pcm_chunks_sent += 1
            current.pcm_bytes_sent += len(value)
            current.pcm_sent_hasher.update(value)
            current.pcm_sent_frame_sha256.append(hashlib.sha256(value).hexdigest())
    for record in records:
        record.pcm_sent_sha256 = record.pcm_sent_hasher.hexdigest()
    for timestamp, wall_timestamp, value in socket.received:
        message = _json_mapping(value)
        if message is None:
            continue
        record = by_id.get(_task_id(message))
        if record is None:
            continue
        event_type = _event_type(message)
        if event_type == "task-started":
            record.task_started_at = _iso(wall_timestamp)
            record.started_monotonic = timestamp
        elif event_type == "result-generated":
            if record.finished_monotonic is not None:
                continue
            usage, raw = _usage_duration(message)
            if raw is not None:
                record.usage_duration = usage
            record.pending_terminal_texts.extend(_sentence_end_texts(message))
        elif event_type == "task-finished":
            record.task_finished_at = _iso(wall_timestamp)
            record.finished_monotonic = timestamp
            if record.started_monotonic is not None:
                record.actual_task_duration_s = timestamp - record.started_monotonic
            if record.pending_terminal_texts:
                record.terminal_text = " ".join(record.pending_terminal_texts).strip()
                record.terminal_count = len(record.pending_terminal_texts)
                record.pending_terminal_texts.clear()
            usage, raw = _usage_duration(message)
            if raw is not None:
                record.usage_duration = usage
    for previous, current in zip(records, records[1:]):
        if current.started_monotonic is not None and previous.finish_monotonic is not None:
            current.transition_latency_s = current.started_monotonic - previous.finish_monotonic
    return records


def _request_vocabulary(socket: RecordingWebSocket | None) -> dict[str, object] | None:
    if socket is None:
        return None
    for _, _, value in socket.sent:
        message = _json_mapping(value)
        if message is None:
            continue
        header = message.get("header")
        header = header if isinstance(header, Mapping) else {}
        if header.get("action") != "run-task":
            continue
        payload = message.get("payload")
        payload = payload if isinstance(payload, Mapping) else {}
        parameters = payload.get("parameters")
        parameters = parameters if isinstance(parameters, Mapping) else {}
        vocabulary = parameters.get("vocabulary")
        return dict(vocabulary) if isinstance(vocabulary, Mapping) else None
    return None


def _event_shapes(socket: RecordingWebSocket | None) -> dict[str, object]:
    if socket is None:
        return {}

    def paths(value: object, prefix: str = "") -> set[str]:
        if not isinstance(value, Mapping):
            return {prefix} if prefix else set()
        result: set[str] = set()
        for key, child in value.items():
            normalized_key = str(key)
            if normalized_key == "header" or "id" in normalized_key.casefold():
                continue
            child_path = f"{prefix}.{normalized_key}" if prefix else normalized_key
            result.add(child_path)
            result.update(paths(child, child_path))
        return result

    def usage_paths(value: object, prefix: str = "") -> set[str]:
        if not isinstance(value, Mapping):
            return set()
        result: set[str] = set()
        usage = value.get("usage")
        if isinstance(usage, Mapping) and "duration" in usage:
            result.add(f"{prefix}.usage.duration" if prefix else "usage.duration")
        for key, child in value.items():
            if key != "header":
                child_path = f"{prefix}.{key}" if prefix else str(key)
                result.update(usage_paths(child, child_path))
        return result

    grouped: dict[str, dict[str, object]] = {}
    for _, _, value in socket.received:
        message = _json_mapping(value)
        if message is None:
            continue
        event_type = _event_type(message) or "unknown"
        summary = grouped.setdefault(
            event_type,
            {
                "count": 0,
                "field_paths": set(),
                "usage_duration_paths": set(),
                "sentence_end_count": 0,
            },
        )
        summary["count"] += 1
        summary["field_paths"].update(paths(message))
        summary["usage_duration_paths"].update(usage_paths(message))
        summary["sentence_end_count"] += len(_sentence_end_texts(message))
    for summary in grouped.values():
        summary["field_paths"] = sorted(summary["field_paths"])
        summary["usage_duration_paths"] = sorted(summary["usage_duration_paths"])
    return grouped


def _safe_error(exc: BaseException) -> dict[str, object]:
    error_type = type(exc).__name__
    result: dict[str, object] = {"type": error_type if error_type in SAFE_EXCEPTION_TYPES else "Exception"}
    code = getattr(exc, "error_code", None)
    if isinstance(code, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", code):
        result["provider_code"] = code
    return result


def _redacted_command(argv: list[str]) -> list[str]:
    sensitive_flags = {
        "--api-key",
        "--apikey",
        "--secret",
        "--token",
        "--password",
        "--passwd",
        "--header",
        "--authorization",
        "--auth",
    }
    result: list[str] = []
    redact_next = False
    for item in argv:
        if redact_next:
            result.append("<redacted>")
            redact_next = False
            continue
        lowered = item.casefold()
        flag_name = lowered.split("=", 1)[0]
        if flag_name in sensitive_flags:
            result.append("<redacted>")
            if "=" not in item:
                redact_next = True
        elif any(
            lowered.startswith(prefix)
            for prefix in (
                "api_key=",
                "apikey=",
                "password=",
                "passwd=",
                "secret=",
                "token=",
                "header=",
                "authorization=",
                "auth=",
            )
        ):
            result.append("<redacted>")
        else:
            result.append(item)
    return result


def sanitize_report(value: object, *, retain_transcripts: bool = False) -> object:
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for raw_key, child in value.items():
            key = str(raw_key)
            normalized = key.casefold()
            if (
                normalized in SENSITIVE_KEYS
                or normalized.endswith("_id")
                or normalized == "id"
                or normalized == "message"
                or normalized == "detail"
            ):
                continue
            if normalized in TRANSCRIPT_KEYS or "transcript" in normalized:
                if retain_transcripts and normalized in TRANSCRIPT_KEYS:
                    result[key] = sanitize_report(child, retain_transcripts=True)
                continue
            if normalized == "text":
                continue
            if normalized == "error":
                if isinstance(child, Mapping):
                    safe_error = {}
                    if child.get("type") in SAFE_EXCEPTION_TYPES:
                        safe_error["type"] = child["type"]
                    if isinstance(child.get("provider_code"), str):
                        safe_error["provider_code"] = child["provider_code"]
                    result[key] = safe_error
                else:
                    result[key] = {"type": "Exception"}
                continue
            if normalized == "failures":
                result[key] = [
                    sanitize_report(item, retain_transcripts=retain_transcripts)
                    for item in child
                ] if isinstance(child, list) else []
                continue
            result[key] = sanitize_report(child, retain_transcripts=retain_transcripts)
        return result
    if isinstance(value, list):
        return [sanitize_report(item, retain_transcripts=retain_transcripts) for item in value]
    if isinstance(value, tuple):
        return [sanitize_report(item, retain_transcripts=retain_transcripts) for item in value]
    if isinstance(value, str):
        if UUID_RE.search(value) or TASK_ID_RE.search(value) or CREDENTIAL_VALUE_RE.search(value):
            return "<redacted>"
        return value
    return value


def _safe_fixture_error(exc: BaseException) -> dict[str, object]:
    return _safe_error(exc)


def _case_result(
    name: str,
    duration_s: float,
    pcm: bytes,
    boundary_latency_s: float,
    event: object,
    *,
    retain_transcripts: bool,
) -> dict[str, object]:
    result: dict[str, object] = {
        "name": name,
        "requested_duration_s": duration_s,
        "pcm_duration_s": len(pcm) / (SAMPLE_RATE_HZ * 2),
        "pcm_bytes": len(pcm),
        "common_local_boundary_to_terminal_s": boundary_latency_s,
        "terminal_count": 1,
        "status": "measured",
    }
    if retain_transcripts:
        result["terminal_text"] = str(getattr(event, "text", "") or "")
    return result


async def _send_pcm(session: object, pcm: bytes, *, realtime_delay_s: float) -> None:
    for chunk in _chunks(pcm):
        await session.send_audio(chunk)
        if realtime_delay_s > 0:
            await asyncio.sleep(realtime_delay_s)


async def _next_event(iterator: Any, timeout_s: float) -> object:
    return await asyncio.wait_for(iterator.__anext__(), timeout=timeout_s)

async def _measure_boundary_to_terminal(boundary: Any, next_event: Any) -> tuple[object, float]:
    started = time.monotonic()
    await boundary()
    event = await next_event()
    return event, time.monotonic() - started


async def _wait_for_task_started(factory: RecordingFactory, count: int, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        socket = factory.socket
        if socket is not None:
            started = sum(
                1
                for _, _, value in socket.received
                if (message := _json_mapping(value)) is not None
                and _event_type(message) == "task-started"
            )
            if started >= count:
                return
        await asyncio.sleep(0.005)
    raise TimeoutError(f"task-started count did not reach {count}")


async def _run_audio_region(
    region: str,
    api_key: str,
    *,
    fixture: bytes | None,
    hotword: str,
    task_timeout_s: float,
    realtime_delay_s: float,
    retain_transcripts: bool,
) -> dict[str, object]:
    factory = RecordingFactory()
    backend = QwenAudioStreamingSTTBackend(
        api_key=api_key,
        language="ko",
        model=QWEN_AUDIO_MODEL,
        endpoint=REGION_ENDPOINTS[region],
        sample_rate_hz=SAMPLE_RATE_HZ,
        task_start_timeout_s=task_timeout_s,
        task_finish_timeout_s=task_timeout_s,
        websocket_factory=factory.connect,
    )
    started_at = _iso(time.time())
    session = None
    iterator = None
    failures: list[object] = []
    case_results: list[dict[str, object]] = []
    comparison_pcm = _comparison_pcm(fixture)
    split_first = _fit_pcm(fixture, 7.0) if fixture else _pcm_for_duration(7.0)
    split_second = _fit_pcm(fixture, 0.2) if fixture else _pcm_for_duration(0.2)
    drain_pcm = _fit_pcm(fixture, 0.3) if fixture else _pcm_for_duration(0.3)
    comparison_latency: float | None = None
    try:
        session = await backend.open_session()
        iterator = session.events().__aiter__()
        cases = build_cases()
        for case_index, case in enumerate(cases):
            duration_s = float(case["duration_s"])
            name = str(case["name"])
            pcm = _fit_pcm(fixture, duration_s) if fixture else _pcm_for_duration(duration_s)
            if name == "short_0.8s_repeat_1":
                comparison_pcm = pcm
            try:
                await _send_pcm(session, pcm, realtime_delay_s=realtime_delay_s)
                event, boundary_latency = await _measure_boundary_to_terminal(
                    session.on_speech_end,
                    lambda: _next_event(iterator, task_timeout_s),
                )
                if name == "short_0.8s_repeat_1":
                    comparison_latency = boundary_latency
                case_results.append(
                    _case_result(
                        name,
                        duration_s,
                        pcm,
                        boundary_latency,
                        event,
                        retain_transcripts=retain_transcripts,
                    )
                )
                await _wait_for_task_started(factory, case_index + 2, task_timeout_s)
            except Exception as exc:
                failures.append(_safe_error(exc))
                case_results.append(
                    {
                        "name": name,
                        "requested_duration_s": duration_s,
                        "status": "failed",
                        "error": _safe_error(exc),
                    }
                )
        split_first = _fit_pcm(fixture, 7.0) if fixture else _pcm_for_duration(7.0)
        split_second = _fit_pcm(fixture, 0.2) if fixture else _pcm_for_duration(0.2)
        split_started = time.monotonic()
        await _send_pcm(session, split_first, realtime_delay_s=realtime_delay_s)
        await session.on_speech_end()
        await _send_pcm(session, split_second, realtime_delay_s=realtime_delay_s)
        first_split_event = await _next_event(iterator, task_timeout_s)
        await _wait_for_task_started(factory, len(cases) + 2, task_timeout_s)
        await session.on_speech_end()
        second_split_event = await _next_event(iterator, task_timeout_s)
        await _wait_for_task_started(factory, len(cases) + 3, task_timeout_s)
        split_result: dict[str, object] = {
            "name": "continuous_crossing_7s_split",
            "requested_duration_s": 7.2,
            "pcm_duration_s": 7.2,
            "intended_chunks_before_fence": len(_chunks(split_first)),
            "intended_chunks_after_fence": len(_chunks(split_second)),
            "intended_bytes_before_fence": len(split_first),
            "intended_bytes_after_fence": len(split_second),
            "common_local_boundary_to_terminal_s": time.monotonic() - split_started,
            "terminal_count": 2,
            "terminal_counts_by_boundary": [1, 1],
            "status": "measured",
        }
        if retain_transcripts:
            split_result["terminal_texts"] = [
                str(getattr(first_split_event, "text", "") or ""),
                str(getattr(second_split_event, "text", "") or ""),
            ]
        case_results.append(split_result)
        drain_pcm = _fit_pcm(fixture, 0.3) if fixture else _pcm_for_duration(0.3)
        await _send_pcm(session, drain_pcm, realtime_delay_s=realtime_delay_s)
        drain_started = time.monotonic()
        await session.stop()
        drain_event = await _next_event(iterator, task_timeout_s)
        drain_result = _case_result(
            "final_drain_before_stop",
            0.3,
            drain_pcm,
            time.monotonic() - drain_started,
            drain_event,
            retain_transcripts=retain_transcripts,
        )
        case_results.append(drain_result)
    except Exception as exc:
        failures.append(_safe_error(exc))
    finally:
        if session is not None:
            try:
                await session.close()
            except Exception as exc:
                failures.append(_safe_error(exc))
    socket = factory.socket
    records = _task_records(socket) if socket is not None else []
    cases = build_cases()
    for record, case in zip(records[: len(cases)], cases):
        duration_s = float(case["duration_s"])
        pcm = _fit_pcm(fixture, duration_s) if fixture else _pcm_for_duration(duration_s)
        record.requested_duration_s = duration_s
        record.pcm_duration_s = duration_s
        _set_expected_identity(record, pcm)
        record.scenario = str(case["name"])
    split_index = len(cases)
    if len(records) > split_index:
        records[split_index].scenario = "continuous_crossing_7s_split_first"
        records[split_index].requested_duration_s = 7.0
        records[split_index].pcm_duration_s = 7.0
        _set_expected_identity(records[split_index], split_first)
    if len(records) > split_index + 1:
        records[split_index + 1].scenario = "continuous_crossing_7s_split_second"
        records[split_index + 1].requested_duration_s = 0.2
        records[split_index + 1].pcm_duration_s = 0.2
        _set_expected_identity(records[split_index + 1], split_second)
    if len(records) > split_index + 2:
        records[split_index + 2].scenario = "final_drain_before_stop"
        records[split_index + 2].requested_duration_s = 0.3
        records[split_index + 2].pcm_duration_s = 0.3
        _set_expected_identity(records[split_index + 2], drain_pcm)
    split_records = records[split_index : split_index + 2]
    expected_split = [_pcm_identity(_chunks(split_first)), _pcm_identity(_chunks(split_second))]
    sent_split = [_record_sent_identity(record) for record in split_records]
    split_conserved = expected_split == sent_split
    split_accounting = {
        "expected": expected_split,
        "sent": sent_split,
        "expected_bytes": [len(split_first), len(split_second)],
        "sent_bytes": [record.pcm_bytes_sent for record in split_records],
        "expected_chunks": [len(_chunks(split_first)), len(_chunks(split_second))],
        "sent_chunks": [record.pcm_chunks_sent for record in split_records],
        "expected_sha256": [item["sha256"] for item in expected_split],
        "sent_sha256": [item["sha256"] for item in sent_split],
        "expected_frame_sha256": [item["frame_sha256"] for item in expected_split],
        "sent_frame_sha256": [item["frame_sha256"] for item in sent_split],
        "conserved": split_conserved,
    }
    if not split_conserved:
        failures.append({"type": "SplitConservationError"})
    result: dict[str, object] = {
        "region": region,
        "endpoint": REGION_ENDPOINTS[region],
        "model": QWEN_AUDIO_MODEL,
        "language": "ko",
        "started_at": started_at,
        "finished_at": _iso(time.time()),
        "status": "measured" if not failures else "failed",
        "cases": case_results,
        "tasks": [record.as_dict(retain_transcripts=retain_transcripts) for record in records],
        "provider_event_shapes": _event_shapes(socket),
        "usage_duration_sum": (
            sum(record.usage_duration for record in records if record.usage_duration is not None)
            if any(record.usage_duration is not None for record in records)
            else None
        ),
        "usage_duration_observed_count": sum(
            record.usage_duration is not None for record in records
        ),
        "split_byte_accounting": split_accounting,
        "comparison_pcm_bytes": len(comparison_pcm),
        "qwen_audio_common_boundary_to_terminal_s": comparison_latency,
        "failures": failures,
    }
    return result


async def _run_split_only(
    region: str,
    api_key: str,
    *,
    fixture: bytes | None,
    task_timeout_s: float,
) -> dict[str, object]:
    factory = RecordingFactory()
    backend = QwenAudioStreamingSTTBackend(
        api_key=api_key,
        language="ko",
        model=QWEN_AUDIO_MODEL,
        endpoint=REGION_ENDPOINTS[region],
        sample_rate_hz=SAMPLE_RATE_HZ,
        task_start_timeout_s=task_timeout_s,
        task_finish_timeout_s=task_timeout_s,
        websocket_factory=factory.connect,
    )
    session = None
    try:
        session = await backend.open_session()
        iterator = session.events().__aiter__()
        first = _fit_pcm(fixture, 7.0) if fixture else _pcm_for_duration(7.0)
        second = _fit_pcm(fixture, 0.2) if fixture else _pcm_for_duration(0.2)
        await _send_pcm(session, first, realtime_delay_s=0.001)
        await session.on_speech_end()
        await _send_pcm(session, second, realtime_delay_s=0.001)
        await _next_event(iterator, task_timeout_s)
        await _wait_for_task_started(factory, 2, task_timeout_s)
        await session.on_speech_end()
        await _next_event(iterator, task_timeout_s)
        await session.close()
        records = _task_records(factory.socket) if factory.socket is not None else []
        records = records[:2]
        expected = [_pcm_identity(_chunks(first)), _pcm_identity(_chunks(second))]
        sent = [_record_sent_identity(record) for record in records]
        conserved = expected == sent
        return {
            "region": region,
            "status": "measured" if conserved else "failed",
            "intended_bytes": [len(first), len(second)],
            "sent_bytes": [record.pcm_bytes_sent for record in records],
            "intended_chunks": [len(_chunks(first)), len(_chunks(second))],
            "sent_chunks": [record.pcm_chunks_sent for record in records],
            "expected": expected,
            "sent": sent,
            "conserved": conserved,
            "event_shapes": _event_shapes(factory.socket),
        }
    except Exception as exc:
        return {"region": region, "status": "failed", "error": _safe_error(exc)}
    finally:
        if session is not None:
            try:
                await session.close()
            except Exception:
                pass


async def _run_hotword_transport_case(
    region: str,
    api_key: str,
    *,
    fixture: bytes | None,
    hotwords: tuple[str, ...],
    task_timeout_s: float,
    retain_transcripts: bool,
) -> dict[str, object]:
    factory = RecordingFactory()
    backend = QwenAudioStreamingSTTBackend(
        api_key=api_key,
        language="ko",
        model=QWEN_AUDIO_MODEL,
        endpoint=REGION_ENDPOINTS[region],
        sample_rate_hz=SAMPLE_RATE_HZ,
        task_start_timeout_s=task_timeout_s,
        task_finish_timeout_s=task_timeout_s,
        hotwords=hotwords,
        websocket_factory=factory.connect,
    )
    session = None
    try:
        session = await backend.open_session()
        pcm = fixture if fixture else _pcm_for_duration(0.8)
        iterator = session.events().__aiter__()
        await _send_pcm(session, pcm, realtime_delay_s=0.001)
        await session.on_speech_end()
        event = await _next_event(iterator, task_timeout_s)
        result: dict[str, object] = {
            "status": "measured_transport",
            "requested_hotwords": list(hotwords),
            "request_vocabulary": _request_vocabulary(factory.socket),
            "pcm_duration_s": len(pcm) / (SAMPLE_RATE_HZ * 2),
            "terminal_count": 1,
            "usage_duration": _task_records(factory.socket)[0].usage_duration
            if factory.socket is not None and _task_records(factory.socket)
            else None,
        }
        if retain_transcripts:
            result["terminal_text"] = str(getattr(event, "text", "") or "")
        return result
    except Exception as exc:
        return {"status": "failed", "error": _safe_error(exc)}
    finally:
        if session is not None:
            try:
                await session.close()
            except Exception:
                pass


async def _run_reconnect_probe(
    region: str,
    api_key: str,
    *,
    fixture: bytes | None,
    task_timeout_s: float,
    retain_transcripts: bool,
) -> dict[str, object]:
    pcm = _comparison_pcm(fixture)
    first_factory = RecordingFactory()
    first_backend = QwenAudioStreamingSTTBackend(
        api_key=api_key,
        language="ko",
        model=QWEN_AUDIO_MODEL,
        endpoint=REGION_ENDPOINTS[region],
        sample_rate_hz=SAMPLE_RATE_HZ,
        task_start_timeout_s=task_timeout_s,
        task_finish_timeout_s=task_timeout_s,
        websocket_factory=first_factory.connect,
    )
    first_session = None
    abort_started = time.monotonic()
    try:
        first_session = await first_backend.open_session()
        await first_session.send_audio(pcm)
        await first_session.abort_for_toggle_off()
        first_outcome = "aborted_mid_lifecycle"
    except Exception as exc:
        first_outcome = _safe_error(exc)
    abort_elapsed = time.monotonic() - abort_started
    if first_session is not None:
        try:
            await first_session.close()
        except Exception:
            pass
    second_factory = RecordingFactory()
    second_backend = QwenAudioStreamingSTTBackend(
        api_key=api_key,
        language="ko",
        model=QWEN_AUDIO_MODEL,
        endpoint=REGION_ENDPOINTS[region],
        sample_rate_hz=SAMPLE_RATE_HZ,
        task_start_timeout_s=task_timeout_s,
        task_finish_timeout_s=task_timeout_s,
        websocket_factory=second_factory.connect,
    )
    second_session = None
    recovery_started = time.monotonic()
    try:
        second_session = await second_backend.open_session()
        iterator = second_session.events().__aiter__()
        await _send_pcm(second_session, pcm, realtime_delay_s=0.001)
        await second_session.on_speech_end()
        event = await _next_event(iterator, task_timeout_s)
        abort_ok = first_outcome == "aborted_mid_lifecycle"
        result: dict[str, object] = {
            "status": "measured" if abort_ok else "failed",
            "abort_status": "measured" if abort_ok else "failed",
            "first_outcome": first_outcome,
            "abort_elapsed_s": abort_elapsed,
            "recovery_elapsed_s": time.monotonic() - recovery_started,
            "pcm_duration_s": len(pcm) / (SAMPLE_RATE_HZ * 2),
            "terminal_count": 1,
        }
        if retain_transcripts:
            result["fresh_session_terminal_text"] = str(getattr(event, "text", "") or "")
        return result
    except Exception as exc:
        return {
            "status": "failed",
            "abort_status": "measured" if first_outcome == "aborted_mid_lifecycle" else "failed",
            "first_outcome": first_outcome,
            "abort_elapsed_s": abort_elapsed,
            "recovery_elapsed_s": time.monotonic() - recovery_started,
            "error": _safe_error(exc),
        }
    finally:
        if second_session is not None:
            try:
                await second_session.close()
            except Exception:
                pass


async def _run_qwen_audio_compare(
    region: str,
    api_key: str,
    *,
    pcm: bytes,
    task_timeout_s: float,
    retain_transcripts: bool,
) -> dict[str, object]:
    started = time.monotonic()
    factory = RecordingFactory()
    backend = QwenAudioStreamingSTTBackend(
        api_key=api_key,
        language="ko",
        model=QWEN_AUDIO_MODEL,
        endpoint=REGION_ENDPOINTS[region],
        sample_rate_hz=SAMPLE_RATE_HZ,
        task_start_timeout_s=task_timeout_s,
        task_finish_timeout_s=task_timeout_s,
        websocket_factory=factory.connect,
    )
    session = None
    try:
        session = await backend.open_session()
        iterator = session.events().__aiter__()
        await session.send_audio(pcm)
        event, boundary_latency = await _measure_boundary_to_terminal(
            session.on_speech_end,
            lambda: _next_event(iterator, task_timeout_s),
        )
        result: dict[str, object] = {
            "status": "measured",
            "endpoint": backend.endpoint,
            "pcm_bytes": len(pcm),
            "common_local_boundary_to_terminal_s": boundary_latency,
            "session_elapsed_s": time.monotonic() - started,
            "terminal_count": 1,
        }
        if retain_transcripts:
            result["terminal_text"] = str(getattr(event, "text", "") or "")
        return result
    except Exception as exc:
        return {"status": "blocked", "error": _safe_error(exc)}
    finally:
        if session is not None:
            try:
                await session.close()
            except Exception:
                pass


async def _run_qwen3_compare(
    region: str,
    api_key: str,
    *,
    pcm: bytes,
    qwen_audio_latency_s: float | None,
    task_timeout_s: float,
    retain_transcripts: bool,
) -> dict[str, object]:
    started = time.monotonic()
    session = None
    try:
        from puripuly_heart.providers.stt.qwen_asr import QwenASRRealtimeSTTBackend

        backend = QwenASRRealtimeSTTBackend(
            api_key=api_key,
            language="ko",
            model="qwen3-asr-flash-realtime",
            endpoint=REGION_ENDPOINTS[region].replace("/inference", "/realtime"),
            sample_rate_hz=SAMPLE_RATE_HZ,
            connect_timeout_s=task_timeout_s,
            finish_timeout_s=task_timeout_s,
        )
        session = await backend.open_session()
        iterator = session.events().__aiter__()
        event, qwen3_latency = await _measure_boundary_to_terminal(
            session.on_speech_end,
            lambda: _next_event(iterator, task_timeout_s),
        )
        result: dict[str, object] = {
            "status": "measured",
            "endpoint": backend.endpoint,
            "pcm_bytes": len(pcm),
            "common_local_boundary_to_terminal_s": qwen3_latency,
            "common_metric_delta_audio_minus_qwen3_s": (
                qwen_audio_latency_s - qwen3_latency
                if qwen_audio_latency_s is not None
                else None
            ),
            "session_elapsed_s": time.monotonic() - started,
            "terminal_count": 1,
        }
        if retain_transcripts:
            result["terminal_text"] = str(getattr(event, "text", "") or "")
        return result
    except Exception as exc:
        return {"status": "blocked", "error": _safe_error(exc)}
    finally:
        if session is not None:
            try:
                await session.close()
            except Exception:
                pass



def _nested_blockers(value: object, path: str = "") -> list[str]:
    blockers: list[str] = []
    if isinstance(value, Mapping):
        status = value.get("status")
        if path and path != "status" and status in {"blocked", "failed"}:
            blockers.append(f"{path} status={status}")
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if key in {"failures", "blocked"} and child:
                blockers.append(f"{child_path} present")
            if key == "abort_status" and child != "measured":
                blockers.append(f"{child_path}={child}")
            if key == "split_byte_accounting" and isinstance(child, Mapping):
                if child.get("conserved") is not True:
                    blockers.append(f"{child_path}.conserved={child.get('conserved')}")
            blockers.extend(_nested_blockers(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            blockers.extend(_nested_blockers(child, f"{path}[{index}]"))
    return blockers


def _required_region_complete(
    value: object,
    *,
    require_split: bool,
    require_qwen3: bool,
    require_hotword: bool,
) -> bool:
    if not isinstance(value, Mapping) or value.get("status") != "measured":
        return False
    if _nested_blockers(value):
        return False
    if require_split:
        split = value.get("split_byte_accounting")
        if not isinstance(split, Mapping) or split.get("conserved") is not True:
            return False
    if require_qwen3:
        comparison = value.get("qwen3_comparison")
        if not isinstance(comparison, Mapping) or comparison.get("status") != "measured":
            return False
    if require_hotword:
        for key in ("hotword_on", "hotword_off"):
            transport = value.get(key)
            if not isinstance(transport, Mapping) or transport.get("status") != "measured_transport":
                return False
    return True


def _base_report(args: argparse.Namespace, regions: tuple[str, ...]) -> dict[str, object]:
    return {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "live" if args.live else "dry-run",
        "command": _redacted_command(sys.argv if args._argv is None else [sys.argv[0], *args._argv]),
        "transcript_retention": {
            "enabled": bool(args.retain_transcripts),
            "mode": "explicit_opt_in" if args.retain_transcripts else "redacted_by_default",
        },
        "acceptance": {
            "smoke": {"status": "not_run"},
            "billing": {"status": "blocked", "reason_code": "console_access_unavailable"},
            "hotword_recognition_delta": {
                "status": "blocked",
                "reason_code": "fixture_not_distinguishing",
            },
            "qwen3_comparison": {"status": "not_run"},
            "hotword_transport": {"status": "not_run"},
            "reconnect": {"status": "not_run"},
            "split_byte_conservation": {"status": "not_run"},
        },
        "fixture": {"status": "not_read", "path": str(args.audio) if args.audio else None},
        "regions": {},
        "blocked": [
            "Model Studio console/account access unavailable; cost and billing rounding are not measured.",
            "Hotword recognition delta is not established by the available fixture.",
        ],
        "planned_regions": regions,
        "planned_cases": build_cases(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--region", choices=("beijing", "singapore", "both"), default="both")
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--hotword", default="PuriPuly")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--task-timeout", type=float, default=20.0)
    parser.add_argument("--realtime-delay", type=float, default=0.005)
    parser.add_argument("--compare-qwen3", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--comparison-only", action="store_true")
    parser.add_argument("--split-only", action="store_true")
    parser.add_argument("--reconnect-only", action="store_true")
    parser.add_argument("--hotword-only", action="store_true")
    parser.add_argument("--retain-transcripts", action="store_true")
    return parser


async def run(argv: list[str] | None = None) -> dict[str, object]:
    parser = _parser()
    args = parser.parse_args(argv)
    args._argv = argv
    regions = ("beijing", "singapore") if args.region == "both" else (args.region,)
    report = _base_report(args, regions)
    if not args.live:
        report["credential_sources"] = {region: "not_read" for region in regions}
        result = sanitize_report(report, retain_transcripts=False)
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result
    fixture = None
    if args.audio is not None:
        try:
            fixture = _read_fixture(args.audio)
            report["fixture"] = {
                "status": "available",
                "path": str(args.audio),
                "sample_rate_hz": SAMPLE_RATE_HZ,
                "channels": 1,
                "sample_width_bytes": 2,
            }
        except Exception as exc:
            report["fixture"] = {
                "status": "blocked",
                "path": str(args.audio),
                "error": _safe_fixture_error(exc),
            }
            report["blocked"].append("Korean/hotword fixture could not be loaded.")
    else:
        report["fixture"] = {"status": "synthetic_pcm", "path": None}
        report["blocked"].append("Korean utterance and hotword recognition require an explicit local SAPI fixture.")
    report["acceptance"]["smoke"] = {
        "status": (
            "not_run"
            if args.comparison_only or args.split_only or args.reconnect_only or args.hotword_only
            else "measured"
        )
    }
    for region in regions:
        if args.live:
            api_key, credential_source = _secret_for_region(region)
        else:
            api_key, credential_source = None, "not_read"
        if api_key is None:
            report["regions"][region] = {
                "status": "blocked",
                "credential_source": credential_source,
                "endpoint": REGION_ENDPOINTS[region],
                "blocked": ["region credential unavailable"],
            }
            report["blocked"].append(f"{region}: region credential unavailable.")
            continue
        try:
            if args.comparison_only:
                pcm = _comparison_pcm(fixture)
                qwen_audio_comparison = await _run_qwen_audio_compare(
                    region,
                    api_key,
                    pcm=pcm,
                    task_timeout_s=args.task_timeout,
                    retain_transcripts=args.retain_transcripts,
                )
                qwen3_comparison = await _run_qwen3_compare(
                    region,
                    api_key,
                    pcm=pcm,
                    qwen_audio_latency_s=qwen_audio_comparison.get(
                        "common_local_boundary_to_terminal_s"
                    ),
                    task_timeout_s=args.task_timeout,
                    retain_transcripts=args.retain_transcripts,
                )
                result = {
                    "region": region,
                    "status": (
                        "measured"
                        if qwen_audio_comparison.get("status") == "measured"
                        and qwen3_comparison.get("status") == "measured"
                        else "blocked"
                    ),
                    "comparison_pcm_bytes": len(pcm),
                    "qwen_audio_comparison": qwen_audio_comparison,
                    "qwen3_comparison": qwen3_comparison,
                }
            elif args.split_only:
                result = await _run_split_only(
                    region, api_key, fixture=fixture, task_timeout_s=args.task_timeout
                )
            elif args.reconnect_only:
                result = await _run_reconnect_probe(
                    region,
                    api_key,
                    fixture=fixture,
                    task_timeout_s=args.task_timeout,
                    retain_transcripts=args.retain_transcripts,
                )
            elif args.hotword_only:
                hotword_on = await _run_hotword_transport_case(
                    region,
                    api_key,
                    fixture=fixture,
                    hotwords=(args.hotword,),
                    task_timeout_s=args.task_timeout,
                    retain_transcripts=args.retain_transcripts,
                )
                hotword_off = await _run_hotword_transport_case(
                    region,
                    api_key,
                    fixture=fixture,
                    hotwords=(),
                    task_timeout_s=args.task_timeout,
                    retain_transcripts=args.retain_transcripts,
                )
                result = {
                    "region": region,
                    "status": (
                        "measured"
                        if hotword_on.get("status") == "measured_transport"
                        and hotword_off.get("status") == "measured_transport"
                        else "blocked"
                    ),
                    "hotword_on": hotword_on,
                    "hotword_off": hotword_off,
                }
            else:
                result = await _run_audio_region(
                    region,
                    api_key,
                    fixture=fixture,
                    hotword=args.hotword,
                    task_timeout_s=args.task_timeout,
                    realtime_delay_s=args.realtime_delay,
                    retain_transcripts=args.retain_transcripts,
                )
                result["qwen3_comparison"] = (
                    await _run_qwen3_compare(
                        region,
                        api_key,
                        pcm=_comparison_pcm(fixture),
                        qwen_audio_latency_s=result.get("qwen_audio_common_boundary_to_terminal_s"),
                        task_timeout_s=args.task_timeout,
                        retain_transcripts=args.retain_transcripts,
                    )
                    if args.compare_qwen3
                    else {"status": "skipped"}
                )
                result["hotword_on"] = await _run_hotword_transport_case(
                    region,
                    api_key,
                    fixture=fixture,
                    hotwords=(args.hotword,),
                    task_timeout_s=args.task_timeout,
                    retain_transcripts=args.retain_transcripts,
                )
                result["hotword_off"] = await _run_hotword_transport_case(
                    region,
                    api_key,
                    fixture=fixture,
                    hotwords=(),
                    task_timeout_s=args.task_timeout,
                    retain_transcripts=args.retain_transcripts,
                )
                required_children: list[tuple[object, str]] = [
                    (result.get("hotword_on"), "measured_transport"),
                    (result.get("hotword_off"), "measured_transport"),
                ]
                if args.compare_qwen3:
                    required_children.append((result.get("qwen3_comparison"), "measured"))
                if result.get("status") == "measured" and not all(
                    isinstance(child, Mapping) and child.get("status") == expected_status
                    for child, expected_status in required_children
                ):
                    result["status"] = "blocked"
            report["regions"][region] = result
            if result.get("status") != "measured":
                report["blocked"].append(f"{region}: region status is {result.get('status')}.")
            for detail in _nested_blockers(result):
                report["blocked"].append(f"{region}: {detail}.")
        except Exception as exc:
            result = {
                "status": "blocked",
                "endpoint": REGION_ENDPOINTS[region],
                "credential_source": credential_source,
                "error": _safe_error(exc),
                "blocked": ["region live operation failed"],
            }
            report["regions"][region] = result
            report["blocked"].append(f"{region}: region live operation failed.")
            for detail in _nested_blockers(result):
                report["blocked"].append(f"{region}: {detail}.")
    region_values = list(report["regions"].values())
    require_split = args.split_only or not (
        args.comparison_only or args.reconnect_only or args.hotword_only
    )
    require_qwen3 = args.comparison_only or (
        args.compare_qwen3
        and not (args.split_only or args.reconnect_only or args.hotword_only)
    )
    require_hotword = args.hotword_only or not (
        args.comparison_only or args.split_only or args.reconnect_only
    )
    complete = [
        _required_region_complete(
            value,
            require_split=require_split,
            require_qwen3=require_qwen3,
            require_hotword=require_hotword,
        )
        for value in region_values
    ]
    if args.comparison_only or require_qwen3:
        report["acceptance"]["qwen3_comparison"] = {
            "status": "measured" if complete and all(complete) else "blocked"
        }
    if args.hotword_only or require_hotword:
        report["acceptance"]["hotword_transport"] = {
            "status": "measured" if complete and all(complete) else "blocked"
        }
    if args.reconnect_only:
        report["acceptance"]["reconnect"] = {
            "status": "measured" if complete and all(complete) else "blocked"
        }
    if require_split:
        report["acceptance"]["split_byte_conservation"] = {
            "status": "measured" if complete and all(complete) else "blocked"
        }
    if not (
        args.comparison_only
        or args.split_only
        or args.reconnect_only
        or args.hotword_only
    ):
        report["acceptance"]["smoke"] = {
            "status": "measured" if complete and all(complete) else "blocked"
        }
    result = sanitize_report(report, retain_transcripts=args.retain_transcripts)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    print(json.dumps(asyncio.run(run()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
