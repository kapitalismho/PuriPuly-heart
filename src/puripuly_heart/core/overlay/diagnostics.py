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

_PROCESS_EVENT_LIMIT = 50
_CHILD_LINE_LIMIT = 100
_PRESENTER_SNAPSHOT_LIMIT = 30
_PRESENTER_REMOVAL_LIMIT = 50
_BRIDGE_EVENT_LIMIT = 30
_TRANSLATION_EVENT_LIMIT = 50


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
    last_dump_path: Path | None = None

    _sequence: int = field(init=False, default=0)

    def record_process(self, event: str, **fields: Any) -> dict[str, Any]:
        _ = (event, fields)
        return {}

    def record_child_line(self, stream: str, line: str) -> dict[str, Any]:
        target = self.child_stderr_lines if stream == "stderr" else self.child_stdout_lines
        return self._append(
            target, category="child_line", event="child_line", stream=stream, line=line
        )

    def record_presenter(self, event: str, **fields: Any) -> dict[str, Any]:
        _ = (event, fields)
        return {}

    def record_presenter_removal(
        self, event: str = "entry_removed", **fields: Any
    ) -> dict[str, Any]:
        _ = (event, fields)
        return {}

    def record_bridge(self, event: str, **fields: Any) -> dict[str, Any]:
        _ = (event, fields)
        return {}

    def record_translation(self, event: str, **fields: Any) -> dict[str, Any]:
        _ = (event, fields)
        return {}

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
        payload: dict[str, Any] = {
            "sequence": self._sequence,
            "recorded_at": time.time(),
            "overlay_instance_id": self.overlay_instance_id,
            "category": category,
            "event": event,
        }
        payload.update(_json_safe_fields(dict(fields)))
        return payload

    def _sorted_events(self) -> list[dict[str, Any]]:
        return sorted(
            self._iter_all_events(),
            key=lambda event: int(event.get("sequence", 0)),
        )

    def _iter_all_events(self) -> Iterable[dict[str, Any]]:
        yield from self.child_stdout_lines
        yield from self.child_stderr_lines
