from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from collections import Counter, deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from puripuly_heart.config.paths import user_config_dir
from puripuly_heart.core.diagnostic_validation import (
    BROKER_RAW_MESSAGE_REDACTION_MARKER,
    DIAGNOSTIC_REDACTION_MARKER,
    DIAGNOSTIC_SINK_FAILURE_JSONL,
    DIAGNOSTIC_VALIDATION_STATUS_ACCEPTED,
    LOCAL_LLM_EXTRA_BODY_REDACTION_MARKER,
    PROVIDER_RESPONSE_BODY_REDACTION_MARKER,
    redact_text_for_sink,
)
from puripuly_heart.core.overlay.manifest import normalize_overlay_logging_mode
from puripuly_heart.core.runtime_logging import SessionLoggingMode

_PROCESS_EVENT_LIMIT = 256
_CHILD_LINE_LIMIT = 100
_PRESENTER_SNAPSHOT_LIMIT = 30
_PRESENTER_REMOVAL_LIMIT = 50
_BRIDGE_EVENT_LIMIT = 30
_TRANSLATION_EVENT_LIMIT = 50
_CHATBOX_EVENT_LIMIT = 50
_STT_EVENT_LIMIT = 50
_NATIVE_EVENT_LIMIT = 50
_MEASUREMENT_PHASE_EVENT_LIMIT = 128
_PRESENTATION_DIAGNOSTICS_MARKER = "presentation_diagnostics "
_MAX_NATIVE_RECORDS_PER_LINE = 8
_MAX_DIAGNOSTIC_LINE_BYTES = 4 * 1024
_MAX_DIAGNOSTIC_DUMP_BYTES = 1024 * 1024
_DIAGNOSTIC_DUMP_DEADLINE_SECONDS = 1.0
_NATIVE_SAFE_FIELDS = frozenset(
    {
        "logical_revision",
        "scene_generation",
        "logical_causes",
        "render_generation",
        "submission_attempt",
        "outcome",
        "visibility",
        "observed_at_ms",
        "reason",
        "lease_disposition",
        "handoff_mode",
        "content_identity",
        "readiness_us",
        "observed_runtime_visible",
        "desired_visible",
        "physical_hmd_visibility",
    }
)
_SENSITIVE_DIAGNOSTIC_FIELD_KEYS = {
    "authorization",
    "auth_token",
    "authtoken",
    "bearer",
    "broker_raw_message",
    "content",
    "cookie",
    "credential",
    "exception",
    "file_contents",
    "id_token",
    "idtoken",
    "local_llm_extra_body",
    "note",
    "path_contents",
    "provider_response_body",
    "raw_exception",
    "secret",
    "session_token",
    "stack_trace",
    "text",
    "token",
    "transcript",
}


def default_overlay_diagnostics_dir() -> Path:
    return user_config_dir() / "diagnostics" / "overlay"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return _json_safe_fields(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _json_safe_fields(fields: dict[Any, Any]) -> dict[str, Any]:
    safe_fields: dict[str, Any] = {}
    redactions: list[str] = []
    for key, value in fields.items():
        normalized_key = str(key)
        compact_key = normalized_key.lower().replace("-", "_")
        if normalized_key == "content_identity":
            safe_fields[normalized_key] = _json_safe(value)
            continue
        if compact_key in _SENSITIVE_DIAGNOSTIC_FIELD_KEYS or any(
            sensitive in compact_key
            for sensitive in (
                "token",
                "secret",
                "authorization",
                "credential",
                "content",
                "text",
                "transcript",
                "stack_trace",
                "path",
            )
        ):
            marker = {
                "broker_raw_message": BROKER_RAW_MESSAGE_REDACTION_MARKER,
                "provider_response_body": PROVIDER_RESPONSE_BODY_REDACTION_MARKER,
                "local_llm_extra_body": LOCAL_LLM_EXTRA_BODY_REDACTION_MARKER,
            }.get(compact_key)
            if marker is not None:
                redactions.append(marker)
            elif isinstance(value, str):
                redacted = _redact_failure_jsonl_text(value)
                if redacted != value:
                    redactions.append(redacted)
                else:
                    redactions.append(DIAGNOSTIC_REDACTION_MARKER)
            continue
        if isinstance(value, str):
            safe_fields[normalized_key] = _redact_failure_jsonl_text(value)
        else:
            safe_fields[normalized_key] = _json_safe(value)
    if redactions:
        safe_fields["redactions"] = redactions
    return safe_fields


def _redact_failure_jsonl_text(text: str) -> str:
    result = redact_text_for_sink(text, DIAGNOSTIC_SINK_FAILURE_JSONL)
    if result.status == DIAGNOSTIC_VALIDATION_STATUS_ACCEPTED and result.text is not None:
        return result.text
    return DIAGNOSTIC_REDACTION_MARKER


@dataclass(slots=True)
class OverlayDiagnosticsRecorder:
    overlay_instance_id: str
    diagnostics_dir: Path = field(default_factory=default_overlay_diagnostics_dir)
    logging_mode: str = SessionLoggingMode.BASIC.value

    process_events: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=_PROCESS_EVENT_LIMIT)
    )
    child_stdout_lines: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=_CHILD_LINE_LIMIT)
    )
    child_stderr_lines: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=_CHILD_LINE_LIMIT)
    )
    presenter_events: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=_PRESENTER_SNAPSHOT_LIMIT)
    )
    presenter_removal_events: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=_PRESENTER_REMOVAL_LIMIT)
    )
    bridge_events: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=_BRIDGE_EVENT_LIMIT)
    )
    translation_events: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=_TRANSLATION_EVENT_LIMIT)
    )
    chatbox_events: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=_CHATBOX_EVENT_LIMIT)
    )
    stt_events: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=_STT_EVENT_LIMIT)
    )
    native_events: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=_NATIVE_EVENT_LIMIT)
    )
    measurement_phase_events: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=_MEASUREMENT_PHASE_EVENT_LIMIT)
    )
    last_dump_path: Path | None = None
    last_dump_receipt: dict[str, Any] | None = None

    _sequence: int = field(init=False, default=0)
    _started_at: float = field(init=False, default_factory=time.monotonic)
    _memory_dropped: Counter[str] = field(init=False, default_factory=Counter)
    _input_rejected: Counter[str] = field(init=False, default_factory=Counter)
    _dump_abandoned: int = field(init=False, default=0)
    _phase_native_cursor: int = field(init=False, default=0)
    _phase_native_drop_cursor: int = field(init=False, default=0)
    _writer_active: bool = field(init=False, default=False)
    _writer_disabled: bool = field(init=False, default=False)
    _dump_attempt: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.set_logging_mode(self.logging_mode)

    def set_logging_mode(self, mode: SessionLoggingMode | str | bool | object) -> None:
        normalized = normalize_overlay_logging_mode(mode)
        if normalized != SessionLoggingMode.DETAILED.value:
            self._clear_stage_events()
        self.logging_mode = normalized

    def record_process(self, event: str, **fields: Any) -> dict[str, Any]:
        return self._append(self.process_events, category="process", event=event, **fields)

    def record_child_line(self, stream: str, line: str) -> dict[str, Any]:
        target = self.child_stderr_lines if stream == "stderr" else self.child_stdout_lines
        return self._append(
            target, category="child_line", event="child_line", stream=stream, line=line
        )

    def note_input_rejected(self, reason: str, *, count: int = 1) -> None:
        self._input_rejected[reason] += max(0, count)

    def record_presenter(self, event: str, **fields: Any) -> dict[str, Any]:
        return self._append_stage(
            self.presenter_events, category="presenter", event=event, **fields
        )

    def record_presenter_removal(
        self, event: str = "entry_removed", **fields: Any
    ) -> dict[str, Any]:
        return self._append_stage(
            self.presenter_removal_events,
            category="presenter_removal",
            event=event,
            **fields,
        )

    def record_bridge(self, event: str, **fields: Any) -> dict[str, Any]:
        return self._append_stage(self.bridge_events, category="bridge", event=event, **fields)

    def record_translation(self, event: str, **fields: Any) -> dict[str, Any]:
        return self._append_stage(
            self.translation_events, category="translation", event=event, **fields
        )

    def record_chatbox(self, event: str, **fields: Any) -> dict[str, Any]:
        return self._append_stage(self.chatbox_events, category="chatbox", event=event, **fields)

    def record_stt(self, event: str, **fields: Any) -> dict[str, Any]:
        return self._append_stage(self.stt_events, category="stt", event=event, **fields)

    def record_native(self, event: str, **fields: Any) -> dict[str, Any]:
        return self._append_stage(self.native_events, category="native", event=event, **fields)

    def ingest_native_child_line(self, line: str) -> bool:
        if not self._stage_recording_enabled():
            return False
        marker_at = line.find(_PRESENTATION_DIAGNOSTICS_MARKER)
        if marker_at < 0:
            return False
        raw = line[marker_at + len(_PRESENTATION_DIAGNOSTICS_MARKER) :].strip()
        try:
            records = json.loads(raw)
        except json.JSONDecodeError:
            self.note_input_rejected("native_invalid_json")
            return True
        if not isinstance(records, list):
            self.note_input_rejected("native_non_list")
            return True
        if len(records) > _MAX_NATIVE_RECORDS_PER_LINE:
            self.note_input_rejected(
                "native_batch_overflow", count=len(records) - _MAX_NATIVE_RECORDS_PER_LINE
            )
            records = records[:_MAX_NATIVE_RECORDS_PER_LINE]
        ingested = False
        for record in records:
            if not isinstance(record, dict):
                self.note_input_rejected("native_non_record")
                continue
            stage = record.get("stage")
            native_sequence = record.get("sequence")
            safe = {key: record.get(key) for key in _NATIVE_SAFE_FIELDS if key in record}
            for key in ("reason", "lease_disposition", "handoff_mode", "content_identity"):
                value = safe.get(key)
                if isinstance(value, str):
                    safe[key] = value[:128]
            self.record_native(
                str(stage or "presentation")[:128],
                native_sequence=native_sequence,
                **safe,
                actual_visibility="not_queried",
            )
            ingested = True
        return True if records else ingested

    def capture_measurement_phase(self, phase: str, *, scene_revision: int | None) -> None:
        if not self._stage_recording_enabled():
            return
        candidates = [
            event
            for event in self.native_events
            if int(event.get("sequence", 0)) > self._phase_native_cursor
        ]
        dropped_now = self._memory_dropped["native"]
        unavailable = max(0, dropped_now - self._phase_native_drop_cursor)
        omitted = max(0, len(candidates) - _MAX_NATIVE_RECORDS_PER_LINE)
        selected = candidates[-_MAX_NATIVE_RECORDS_PER_LINE:]
        correlation = "not_observed"
        if selected:
            correlation = "partial" if unavailable or omitted else "observed"
        self._append(
            self.measurement_phase_events,
            category="measurement_phase",
            event="checkpoint",
            phase=phase,
            scene_revision=scene_revision,
            native_record_count=len(selected),
            native_records_omitted=omitted,
            native_records_unavailable=unavailable,
            native_correlation=correlation,
        )
        for native in selected:
            retained = {
                key: value
                for key, value in native.items()
                if key
                in {
                    "native_sequence",
                    "logical_revision",
                    "scene_generation",
                    "logical_causes",
                    "render_generation",
                    "submission_attempt",
                    "event",
                    "outcome",
                    "visibility",
                    "source_monotonic_ms",
                    "observed_at_ms",
                    "reason",
                    "lease_disposition",
                    "handoff_mode",
                    "content_identity",
                }
            }
            if "event" in retained:
                retained["native_event"] = retained.pop("event")
            self._append(
                self.measurement_phase_events,
                category="measurement_phase_native",
                event=str(native.get("event", "presentation")),
                phase=phase,
                checkpoint_scene_revision=scene_revision,
                **retained,
            )
        if candidates:
            self._phase_native_cursor = max(int(event.get("sequence", 0)) for event in candidates)
        self._phase_native_drop_cursor = dropped_now

    def evidence_summary(self) -> dict[str, Any]:
        events = [
            *self.native_events,
            *(
                event
                for event in self.measurement_phase_events
                if event.get("category") == "measurement_phase_native"
            ),
        ]
        handoff_modes = Counter(
            str(event["handoff_mode"]) for event in events if event.get("handoff_mode") is not None
        )
        stages = Counter(str(event.get("event", "unknown")) for event in events)
        outcomes = Counter(
            str(event["outcome"]) for event in events if event.get("outcome") is not None
        )
        return {
            "native_records_retained": len(self.native_events),
            "phase_records_retained": len(self.measurement_phase_events),
            "native_handoff_modes": dict(sorted(handoff_modes.items())),
            "native_stage_counts": dict(sorted(stages.items())),
            "native_outcome_counts": dict(sorted(outcomes.items())),
            "cache_hit_observed": any(
                event.get("event") == "submission_returned"
                and event.get("outcome") == "success"
                and event.get("handoff_mode") == "cached_frame_rehandoff"
                and event.get("content_identity") is not None
                for event in events
            ),
            "real_render_observed": any(
                event.get("event") == "render_returned"
                and event.get("outcome") == "success"
                and event.get("handoff_mode") == "off"
                for event in events
            ),
            "cached_frame_rehandoff_observed": any(
                event.get("event") == "submission_returned"
                and event.get("outcome") == "success"
                and event.get("handoff_mode") == "cached_frame_rehandoff"
                and event.get("reason") == "cached_completed_frame_rehandoff"
                and event.get("submission_attempt") is not None
                for event in events
            ),
            "content_identity_observed": any(
                event.get("content_identity") is not None for event in events
            ),
            "memory_dropped": dict(sorted(self._memory_dropped.items())),
            "input_rejected": dict(sorted(self._input_rejected.items())),
            "dump_abandoned": self._dump_abandoned,
        }

    async def dump_evidence(self, *, outcome: str, **summary_fields: Any) -> dict[str, Any]:
        self._dump_attempt += 1
        attempt = self._dump_attempt
        if self._writer_disabled:
            return self._record_dump_abandonment("writer_disabled_after_abandonment")
        if self._writer_active:
            return self._record_dump_abandonment("writer_busy")

        path, temporary, content, receipt = self._prepare_dump(
            outcome,
            _json_safe_fields(dict(summary_fields)),
            self._sorted_events(),
        )
        completed = threading.Event()
        write_dump_file = self._write_dump_file
        failure: list[BaseException] = []

        def write_owned_snapshot() -> None:
            try:
                write_dump_file(temporary, path, content)
            except BaseException as error:
                failure.append(error)
            finally:
                completed.set()

        self._writer_active = True
        writer = threading.Thread(
            target=write_owned_snapshot,
            name=f"OverlayDiagnosticsWriter:{self.overlay_instance_id}",
            daemon=True,
        )
        try:
            writer.start()
        except RuntimeError:
            self._writer_active = False
            return self._record_dump_abandonment("writer_start_failed")
        deadline = asyncio.get_running_loop().time() + _DIAGNOSTIC_DUMP_DEADLINE_SECONDS
        try:
            while not completed.is_set():
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    self._writer_active = False
                    self._writer_disabled = True
                    return self._record_dump_abandonment("dump_deadline_exceeded")
                await asyncio.sleep(min(0.01, remaining))
        except asyncio.CancelledError:
            self._writer_active = False
            self._writer_disabled = True
            self._record_dump_abandonment("dump_cancelled")
            raise

        self._writer_active = False
        if failure:
            return self._record_dump_abandonment("dump_io_failed")
        if attempt == self._dump_attempt:
            self.last_dump_path = path
            self.last_dump_receipt = receipt
        return receipt

    def _record_dump_abandonment(self, reason: str) -> dict[str, Any]:
        self._dump_abandoned += 1
        receipt = {
            "outcome": "abandoned",
            "reason": reason,
            "deadline_seconds": _DIAGNOSTIC_DUMP_DEADLINE_SECONDS,
            "byte_ceiling": _MAX_DIAGNOSTIC_DUMP_BYTES,
            "line_byte_ceiling": _MAX_DIAGNOSTIC_LINE_BYTES,
            "dump_abandoned": self._dump_abandoned,
            "complete": False,
            **self.evidence_summary(),
        }
        self.last_dump_receipt = receipt
        return receipt

    def _prepare_dump(
        self,
        outcome: str,
        summary_fields: dict[str, Any],
        events: list[dict[str, Any]],
    ) -> tuple[Path, Path, bytes, dict[str, Any]]:
        timestamp = (
            f"{time.strftime('%Y%m%d-%H%M%S', time.localtime())}-"
            f"{time.time_ns() % 1_000_000_000:09d}"
        )
        path = (
            self.diagnostics_dir
            / f"overlay-diagnostics-{outcome}-{timestamp}-{self.overlay_instance_id}.jsonl"
        )
        writer_identity = hashlib.sha256(
            self.overlay_instance_id.encode("utf-8", errors="replace")
        ).hexdigest()[:16]
        temporary = self.diagnostics_dir / f".overlay-diagnostics-{writer_identity}.tmp"
        encoded_events: list[bytes] = []
        truncated_records = 0
        for event in events:
            encoded = self._encode_line(event)
            if len(encoded) > _MAX_DIAGNOSTIC_LINE_BYTES:
                truncated_records += 1
                encoded = self._encode_line(
                    {
                        "sequence": event.get("sequence"),
                        "category": event.get("category"),
                        "event": event.get("event"),
                        "record_truncated": True,
                        "original_bytes": len(encoded),
                    }
                )
            encoded_events.append(encoded)

        base_summary = {
            "category": "summary",
            "event": f"{outcome}_summary",
            "overlay_instance_id": self.overlay_instance_id,
            **summary_fields,
            **self.evidence_summary(),
            "record_line_byte_ceiling": _MAX_DIAGNOSTIC_LINE_BYTES,
            "dump_byte_ceiling": _MAX_DIAGNOSTIC_DUMP_BYTES,
            "dump_deadline_seconds": _DIAGNOSTIC_DUMP_DEADLINE_SECONDS,
            "records_available": len(encoded_events),
            "records_truncated": truncated_records,
        }
        summary_line = self._encode_line({**base_summary, "records_omitted": len(encoded_events)})
        available = _MAX_DIAGNOSTIC_DUMP_BYTES - len(summary_line)
        retained: list[bytes] = []
        retained_bytes = 0
        for encoded in encoded_events:
            if retained_bytes + len(encoded) > available:
                break
            retained.append(encoded)
            retained_bytes += len(encoded)
        omitted = len(encoded_events) - len(retained)
        completeness = (
            omitted == 0
            and truncated_records == 0
            and not self._memory_dropped
            and not self._input_rejected
            and self._dump_abandoned == 0
        )
        summary_line = self._encode_line(
            {
                **base_summary,
                "records_retained": len(retained),
                "records_omitted": omitted,
                "complete": completeness,
            }
        )
        while retained and len(summary_line) + sum(map(len, retained)) > _MAX_DIAGNOSTIC_DUMP_BYTES:
            retained.pop()
            omitted += 1
            completeness = False
            summary_line = self._encode_line(
                {
                    **base_summary,
                    "records_retained": len(retained),
                    "records_omitted": omitted,
                    "complete": False,
                }
            )
        content = b"".join((summary_line, *retained))
        receipt = {
            "outcome": "written",
            "file_name": path.name,
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
            "byte_ceiling": _MAX_DIAGNOSTIC_DUMP_BYTES,
            "line_byte_ceiling": _MAX_DIAGNOSTIC_LINE_BYTES,
            "deadline_seconds": _DIAGNOSTIC_DUMP_DEADLINE_SECONDS,
            "records_retained": len(retained),
            "records_omitted": omitted,
            "records_truncated": truncated_records,
            "complete": completeness,
            **self.evidence_summary(),
        }
        return path, temporary, content, receipt

    @staticmethod
    def _write_dump_file(temporary: Path, path: Path, content: bytes) -> None:
        temporary.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("wb") as handle:
            handle.write(content)
        temporary.replace(path)

    @staticmethod
    def _encode_line(payload: dict[str, Any]) -> bytes:
        raw = json.dumps(payload, ensure_ascii=True, default=str, sort_keys=True)
        return raw.encode("utf-8", errors="replace") + b"\n"

    def _stage_recording_enabled(self) -> bool:
        return self.logging_mode == SessionLoggingMode.DETAILED.value

    def _clear_stage_events(self) -> None:
        self.presenter_events.clear()
        self.presenter_removal_events.clear()
        self.bridge_events.clear()
        self.translation_events.clear()
        self.chatbox_events.clear()
        self.stt_events.clear()
        self.native_events.clear()
        self.measurement_phase_events.clear()
        self._phase_native_cursor = 0
        self._phase_native_drop_cursor = self._memory_dropped["native"]

    def _append_stage(
        self,
        target: deque[dict[str, Any]],
        *,
        category: str,
        event: str,
        **fields: Any,
    ) -> dict[str, Any]:
        if not self._stage_recording_enabled():
            return {}
        return self._append(target, category=category, event=event, **fields)

    def _append(
        self,
        target: deque[dict[str, Any]],
        *,
        category: str,
        event: str,
        **fields: Any,
    ) -> dict[str, Any]:
        payload = self._event(category=category, event=event, **fields)
        if target.maxlen is not None and len(target) >= target.maxlen:
            self._memory_dropped[category] += 1
        target.append(payload)
        return payload

    def _event(self, *, category: str, event: str, **fields: Any) -> dict[str, Any]:
        self._sequence += 1
        safe_fields = _json_safe_fields(dict(fields))
        source_monotonic_ms = safe_fields.pop("monotonic_ms", None)
        payload: dict[str, Any] = {
            "sequence": self._sequence,
            "recorded_at": time.time(),
            "monotonic_ms": round((time.monotonic() - self._started_at) * 1000, 3),
            "overlay_instance_id": self.overlay_instance_id,
            "category": category,
            "event": event,
        }
        if source_monotonic_ms is not None:
            payload["source_monotonic_ms"] = source_monotonic_ms
        payload.update(safe_fields)
        return payload

    def _sorted_events(self) -> list[dict[str, Any]]:
        return sorted(
            self._iter_all_events(),
            key=lambda event: int(event.get("sequence", 0)),
        )

    def _iter_all_events(self) -> Iterable[dict[str, Any]]:
        yield from self.process_events
        yield from self.child_stdout_lines
        yield from self.child_stderr_lines
        yield from self.presenter_events
        yield from self.presenter_removal_events
        yield from self.bridge_events
        yield from self.translation_events
        yield from self.chatbox_events
        yield from self.stt_events
        yield from self.native_events
        yield from self.measurement_phase_events
