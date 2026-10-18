from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from bench.text import tokens as tokenize

OVERLAP_MARGIN_S = 0.1
SHORT_TURN_S = 1.0


@dataclass(slots=True)
class RefToken:
    index: int
    speaker: str
    token: str
    start: float
    end: float
    window: tuple[float, float]
    turn: int


@dataclass(slots=True)
class Segment:
    index: int
    speaker: str
    start: float
    end: float
    tokens: list[int] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(slots=True)
class Transition:
    index: int
    before: int
    after: int
    kind: str
    gap: float
    short_successor: bool
    returning: bool

    @property
    def key(self) -> str:
        return f"t{self.index}"


@dataclass(slots=True)
class Reference:
    case_id: str
    language: str
    category: str
    synthetic: bool
    duration: float
    tokens: list[RefToken]
    segments: list[Segment]
    transitions: list[Transition]
    regions: list[tuple[float, float]]
    gap_s: float
    raw: dict

    def region_end(self, time: float) -> float:
        for lo, hi in self.regions:
            if lo - 1e-9 <= time <= hi + 1e-9:
                return hi
        return time

    def speech_seconds(self) -> float:
        return sum(hi - lo for lo, hi in self.regions)


def load(path: Path, *, gap_s: float) -> Reference:
    return build(json.loads(Path(path).read_text(encoding="utf-8")), gap_s=gap_s)


def build(raw: dict, *, gap_s: float) -> Reference:
    language = raw["language"]
    items: list[RefToken] = []
    for turn_index, turn in enumerate(raw["turns"]):
        words = turn.get("words")
        if words:
            for word in words:
                for token in tokenize(word["text"], language):
                    items.append(
                        RefToken(
                            0,
                            turn["speaker"],
                            token,
                            float(word["start"]),
                            float(word["end"]),
                            (float(word["start"]), float(word["end"])),
                            turn_index,
                        )
                    )
            continue
        parts = tokenize(turn["text"], language)
        start, end = float(turn["start"]), float(turn["end"])
        width = (end - start) / max(1, len(parts))
        for k, token in enumerate(parts):
            items.append(
                RefToken(
                    0,
                    turn["speaker"],
                    token,
                    start + k * width,
                    start + (k + 1) * width,
                    (start, end),
                    turn_index,
                )
            )
    items.sort(key=lambda t: (t.start, t.end, t.speaker, t.turn))
    for index, item in enumerate(items):
        item.index = index
    segments = _segments(items, gap_s)
    transitions = _transitions(segments, gap_s)
    regions = _regions(items, gap_s)
    return Reference(
        case_id=raw["case_id"],
        language=language,
        category=raw.get("category", ""),
        synthetic=bool(raw.get("synthetic")),
        duration=float(raw["audio"]["samples"]) / float(raw["audio"]["sample_rate_hz"]),
        tokens=items,
        segments=segments,
        transitions=transitions,
        regions=regions,
        gap_s=gap_s,
        raw=raw,
    )


def _segments(items: list[RefToken], gap_s: float) -> list[Segment]:
    by_speaker: dict[str, list[RefToken]] = {}
    for item in items:
        by_speaker.setdefault(item.speaker, []).append(item)
    segments: list[Segment] = []
    for speaker, owned in by_speaker.items():
        current: Segment | None = None
        for item in owned:
            if current is not None and item.start - current.end < gap_s:
                current.end = max(current.end, item.end)
                current.tokens.append(item.index)
                continue
            current = Segment(0, speaker, item.start, item.end, [item.index])
            segments.append(current)
    segments.sort(key=lambda s: (s.start, s.end, s.speaker))
    for index, segment in enumerate(segments):
        segment.index = index
    return segments


def _transitions(segments: list[Segment], gap_s: float) -> list[Transition]:
    transitions: list[Transition] = []
    for after in segments:
        earlier = [
            s
            for s in segments
            if s.start < after.start or (s.start == after.start and s.index < after.index)
        ]
        others = [s for s in earlier if s.speaker != after.speaker]
        if not others:
            continue
        before = max(others, key=lambda s: (s.end, s.start))
        own = [s for s in earlier if s.speaker == after.speaker]
        if own and max(s.end for s in own) > before.end:
            continue
        gap = after.start - before.end
        if after.end <= before.end:
            kind = "contained"
        elif gap >= gap_s:
            kind = "paused"
        elif gap >= -OVERLAP_MARGIN_S:
            kind = "continuous"
        else:
            kind = "overlap"
        resumed = next(
            (
                s
                for s in segments[after.index + 1 :]
                if s.speaker == before.speaker and s.start - after.end < gap_s
            ),
            None,
        )
        transitions.append(
            Transition(
                len(transitions),
                before.index,
                after.index,
                kind,
                gap,
                after.duration <= SHORT_TURN_S,
                resumed is not None,
            )
        )
    return transitions


def _regions(items: list[RefToken], gap_s: float) -> list[tuple[float, float]]:
    spans = sorted((item.start, item.end) for item in items)
    regions: list[list[float]] = []
    for lo, hi in spans:
        if regions and lo - regions[-1][1] < gap_s:
            regions[-1][1] = max(regions[-1][1], hi)
        else:
            regions.append([lo, hi])
    return [(lo, hi) for lo, hi in regions]
