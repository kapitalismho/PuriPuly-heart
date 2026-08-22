from __future__ import annotations

import json
import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from puripuly_heart.config.paths import user_config_dir
from puripuly_heart.core.diagnostic_validation import (
    DIAGNOSTIC_REDACTION_MARKER,
    DIAGNOSTIC_SINK_FAILURE_JSONL,
    DIAGNOSTIC_VALIDATION_STATUS_ACCEPTED,
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
_PRESENTATION_DIAGNOSTICS_MARKER = "presentation_diagnostics "
_SENSITIVE_DIAGNOSTIC_FIELD_KEYS = {
    "accesstoken",
    "apikey",
    "audio",
    "audiobytes",
    "authorization",
    "credentials",
    "idtoken",
    "refreshtoken",
    "sessiontoken",
    "subtitlecontent",
    "text",
    "transcript",
}


def default_overlay_diagnostics_dir() -> Path:
    return user_config_dir() / "diagnostics" / "overlay"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return _redact_failure_jsonl_text(str(value))
    if isinstance(value, dict):
        return _json_safe_fields(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, str):
        return _redact_failure_jsonl_text(value)
    return value


def _json_safe_fields(fields: dict[Any, Any]) -> dict[str, Any]:
    safe_fields: dict[str, Any] = {}
    for key, value in fields.items():
        key_text = str(key)
        normalized_key = "".join(character for character in key_text.lower() if character.isalnum())
        if normalized_key in _SENSITIVE_DIAGNOSTIC_FIELD_KEYS:
            safe_fields[f"redacted_field_{len(safe_fields) + 1}"] = DIAGNOSTIC_REDACTION_MARKER
            continue
        raw_assignment = f"{key_text}={value}"
        redacted_assignment = _redact_failure_jsonl_text(raw_assignment)
        if redacted_assignment != raw_assignment:
            safe_fields[f"redacted_field_{len(safe_fields) + 1}"] = redacted_assignment
            continue
        safe_fields[key_text] = _json_safe(value)
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
    last_dump_path: Path | None = None

    _sequence: int = field(init=False, default=0)
    _started_at: float = field(init=False, default_factory=time.monotonic)

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
            return False
        if not isinstance(records, list):
            return False
        ingested = False
        for record in records:
            if not isinstance(record, dict):
                continue
            self.record_native(
                str(record.get("stage") or "presentation"),
                logical_revision=record.get("logical_revision"),
                outcome=record.get("outcome"),
                readiness_us=record.get("readiness_us"),
                cached_visibility=record.get("observed_runtime_visible"),
                desired_visible=record.get("desired_visible"),
                actual_visibility="not_queried",
                physical_hmd_visibility=record.get("physical_hmd_visibility"),
            )
            ingested = True
        return ingested

    def dump_failure(self, **summary_fields: Any) -> Path:
        self.diagnostics_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        path = (
            self.diagnostics_dir
            / f"overlay-diagnostics-{timestamp}-{self.overlay_instance_id}.jsonl"
        )
        events = [
            self._event(category="summary", event="failure_summary", **summary_fields),
            *self._sorted_events(),
        ]
        with path.open("w", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(event, ensure_ascii=True, default=str))
                handle.write("\n")
        self.last_dump_path = path
        return path

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
