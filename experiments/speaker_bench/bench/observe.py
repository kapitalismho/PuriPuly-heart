from __future__ import annotations

import dataclasses
import enum
import json
import logging
import threading
import time
from pathlib import Path
from uuid import UUID

current_case: dict[str, object] = {"case_id": None}
activity: dict[str, float] = {"last": 0.0}
ACTIVITY_EVENTS = frozenset({"admitted", "first_readable", "settled", "log"})
_lock = threading.Lock()
_path: Path | None = None


def serial(value, depth=0):
    if depth > 8:
        return repr(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, enum.Enum):
        return value.value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: serial(getattr(value, field.name), depth + 1)
            for field in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {str(k): serial(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [serial(v, depth + 1) for v in value]
    return repr(value)


def emit(kind: str, **payload) -> None:
    if _path is None:
        return
    now = time.monotonic()
    if kind in ACTIVITY_EVENTS:
        activity["last"] = now
    record = {"event": kind, "t": now, "case_id": current_case["case_id"], **payload}
    line = json.dumps(record, ensure_ascii=False, default=repr)
    with _lock, _path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def _span(span):
    return {
        "capture_epoch": span.capture_epoch,
        "normalized_start_sample": span.normalized_start_sample,
        "normalized_end_sample": span.normalized_end_sample,
        "source_start_sample": span.source_start_sample,
        "source_end_sample": span.source_end_sample,
        "source_start_monotonic_s": span.source_start_monotonic_s,
        "source_end_monotonic_s": span.source_end_monotonic_s,
    }


def _origin(origin):
    return {
        "text_start": origin.text_start,
        "text_end": origin.text_end,
        "request_id": origin.request_id,
        "contribution_id": origin.contribution_id,
        "source_binding": origin.source_binding,
        "ownership": origin.ownership,
        "source_ranges": [_span(s) for s in origin.source_ranges],
        "context_ranges": [_span(s) for s in origin.context_ranges],
        "source_available_at_monotonic_s": origin.source_available_at_monotonic_s,
        "speaker_attribution": serial(origin.speaker_attribution),
        "speaker_proof_id": origin.speaker_proof_id,
    }


def _assignment(assignment):
    if assignment is None:
        return None
    attribution = assignment.attribution
    return {
        "state": attribution.state,
        "key": serial(attribution.key),
        "source": attribution.source,
        "speaker_scope_id": attribution.speaker_scope_id,
        "palette_index": assignment.palette_index,
        "palette_overflow": assignment.palette_overflow,
        "settled": assignment.settled,
        "revision": assignment.revision,
    }


class _LogRecorder(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            emit("log", logger=record.name, level=record.levelname, message=record.getMessage())
        except Exception:
            pass


def set_path(path: Path) -> None:
    global _path
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        _path = path


def install(path: Path) -> None:
    set_path(path)

    from puripuly_heart.core import speaker_identity

    allocator = speaker_identity.PeerSpeakerIdentityAllocator
    original_observe = allocator.observe
    original_mark_readable = allocator.mark_readable
    original_settle = allocator._settle
    original_reset = allocator.reset
    original_retire_source = allocator.retire_source

    def observe(self, transcript, *, child_sequence, defer_settlement=False):
        assignment = original_observe(
            self, transcript, child_sequence=child_sequence, defer_settlement=defer_settlement
        )
        source = transcript.source_child
        emit(
            "admitted",
            allocator=id(self),
            utterance_id=str(transcript.utterance_id),
            source_id=None if source is None else str(source.source_id),
            child_sequence=child_sequence,
            order=None if source is None else list(source.order),
            raw_text=transcript.text if source is None else source.raw_text,
            language=None if source is None else source.language,
            text_range=None if source is None else list(source.text_range),
            display_group_id=None if source is None else serial(source.display_group_id),
            origins=[] if source is None else [_origin(o) for o in source.origins],
            final_speaker_runs=serial(transcript.final_speaker_runs),
            assignment=_assignment(assignment),
        )
        return assignment

    def mark_readable(self, source, *, now):
        record = self._sources.get(source.source_id)
        first = record is not None and record.first_readable is None
        assignment = original_mark_readable(self, source, now=now)
        if first:
            emit(
                "first_readable",
                allocator=id(self),
                source_id=str(source.source_id),
                readable_at=now,
                assignment=_assignment(assignment),
            )
        return assignment

    def _settle(self, record, *, now):
        was_settled = record.settled
        original_settle(self, record, now=now)
        if was_settled:
            return
        source_id = next((k for k, v in self._sources.items() if v is record), None)
        emit(
            "settled",
            allocator=id(self),
            source_id=None if source_id is None else str(source_id),
            settled_at=now,
            first_readable=record.first_readable,
            assignment=_assignment(record.assignment),
            presented=_assignment(record.presented),
        )

    def reset(self):
        emit("allocator_reset", allocator=id(self))
        return original_reset(self)

    def retire_source(self, source_id):
        record = self._sources.get(source_id)
        if record is not None:
            emit(
                "retired",
                allocator=id(self),
                source_id=str(source_id),
                settled=record.settled,
                assignment=_assignment(record.assignment),
            )
        return original_retire_source(self, source_id)

    allocator.observe = observe
    allocator.mark_readable = mark_readable
    allocator._settle = _settle
    allocator.reset = reset
    allocator.retire_source = retire_source

    handler = _LogRecorder(level=logging.INFO)
    for name in (
        "puripuly_heart.core.stt.batch_speaker",
        "puripuly_heart.core.stt.streaming_speaker",
    ):
        logger = logging.getLogger(name)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
