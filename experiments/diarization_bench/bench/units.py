from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

SAMPLE_RATE_HZ = 16_000
_SLICE = re.compile(
    r"\[BatchSpeaker\].*?reason=(?P<reason>\S+) source_boundary=(?P<boundary>-?\d+) "
    r"evidence_ready=(?P<ready>\S+) submitted_at=(?P<submitted>\S+)"
)


@dataclass(slots=True)
class Assignment:
    state: str
    key: str | None
    palette_index: int | None
    palette_overflow: bool

    @property
    def identified(self) -> bool:
        return self.key is not None and self.state == "identified"

    @property
    def colored(self) -> bool:
        return self.identified and self.palette_index is not None and not self.palette_overflow

    @property
    def displayed(self) -> str:
        if self.colored:
            return "colored"
        if self.identified and self.palette_overflow:
            return "overflow"
        if self.identified:
            return "unallocated"
        return self.state or "unknown"


@dataclass(slots=True)
class Unit:
    source_id: str
    order: tuple
    text: str
    language: str | None
    admitted_at: float
    spans: list[tuple[float, float]]
    first_readable: float | None = None
    readable_assignment: Assignment | None = None
    settled_at: float | None = None
    final: Assignment | None = None
    settle_events: int = 0
    events: list[dict] = field(default_factory=list)

    @property
    def lo(self) -> float:
        return min(lo for lo, _ in self.spans) if self.spans else float("nan")

    @property
    def hi(self) -> float:
        return max(hi for _, hi in self.spans) if self.spans else float("nan")


@dataclass(slots=True)
class Slice:
    reason: str
    boundary_s: float
    evidence_ready_s: float | None
    submitted_s: float


@dataclass(slots=True)
class CaseRun:
    case_id: str
    case_start: float
    lead_in_samples: int
    units: list[Unit]
    source_record: dict
    invalid: list[str]
    slices: list[Slice] = field(default_factory=list)
    evidence: list[tuple[float, float, float, tuple[float, ...]]] = field(default_factory=list)


def _key(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return "|".join(f"{k}={value[k]}" for k in sorted(value))
    return str(value)


def _assignment(value: dict | None) -> Assignment | None:
    if value is None:
        return None
    return Assignment(
        value.get("state") or "unknown",
        _key(value.get("key")),
        value.get("palette_index"),
        bool(value.get("palette_overflow")),
    )


def _merge(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for lo, hi in sorted(spans):
        if merged and lo <= merged[-1][1] + 1e-6:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [(lo, hi) for lo, hi in merged]


def load_events(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def case_run(events: list[dict], case_id: str, *, max_frame_lag_s: float | None) -> CaseRun:
    own = [e for e in events if e.get("case_id") == case_id]
    closes = [e for e in own if e["event"] == "source_close"]
    invalid: list[str] = []
    if not closes:
        raise ValueError(f"no source record for case {case_id}")
    record = closes[-1]["record"]
    if not record.get("case_completed"):
        invalid.append("case_audio_incomplete")
    if max_frame_lag_s is not None and record["frame_lag_s"]["max"] > max_frame_lag_s:
        invalid.append("frame_lag_exceeded")
    lead = int(record["lead_in_samples"])
    start = float(record["case_start_monotonic"])
    units: dict[str, Unit] = {}
    slices: list[Slice] = []
    frames: list[tuple[float, float, float, tuple[float, ...]]] = []
    seen_frames: set = set()
    for event in own:
        kind = event["event"]
        source_id = event.get("source_id")
        if kind == "admitted" and source_id:
            if source_id in units:
                continue
            spans = []
            for origin in event["origins"]:
                for span in origin["source_ranges"]:
                    a, b = span["normalized_start_sample"], span["normalized_end_sample"]
                    if a is None or b is None:
                        continue
                    spans.append(((a - lead) / SAMPLE_RATE_HZ, (b - lead) / SAMPLE_RATE_HZ))
            units[source_id] = Unit(
                source_id,
                tuple(event.get("order") or ()),
                event.get("raw_text") or "",
                event.get("language"),
                float(event["t"]) - start,
                _merge(spans),
            )
            units[source_id].events.append(event)
        elif kind == "first_readable" and source_id in units:
            unit = units[source_id]
            if unit.first_readable is None:
                unit.first_readable = float(event["readable_at"]) - start
                unit.readable_assignment = _assignment(event.get("assignment"))
        elif kind == "settled" and source_id in units:
            unit = units[source_id]
            unit.settle_events += 1
            unit.settled_at = float(event["settled_at"]) - start
            unit.final = _assignment(event.get("assignment"))
            unit.events.append(event)
        elif kind == "observation_error":
            invalid.append("observer_failed")
        elif kind == "evidence":
            for ready, lo, hi, probabilities, _epoch in event["frames"]:
                key = (lo, hi)
                if key in seen_frames:
                    continue
                seen_frames.add(key)
                frames.append(
                    (
                        float(ready) - start,
                        (lo - lead) / SAMPLE_RATE_HZ,
                        (hi - lead) / SAMPLE_RATE_HZ,
                        tuple(probabilities),
                    )
                )
        elif kind == "log":
            match = _SLICE.search(event.get("message", ""))
            if match:
                ready = match.group("ready")
                slices.append(
                    Slice(
                        match.group("reason"),
                        (int(match.group("boundary")) - lead) / SAMPLE_RATE_HZ,
                        None if ready == "None" else float(ready) - start,
                        float(match.group("submitted")) - start,
                    )
                )
    ordered = sorted(units.values(), key=lambda u: (u.order, u.admitted_at))
    for unit in ordered:
        if unit.final is None:
            unit.final = _assignment(unit.events[0].get("assignment"))
    frames.sort(key=lambda f: f[1])
    return CaseRun(case_id, start, lead, ordered, record, invalid, slices, frames)
