from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .schemas import ReferenceAction


class Phase5InputError(RuntimeError):
    pass


def field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value[name]
    return getattr(value, name)


def clip_interval(start: int, end: int, low: int, high: int) -> tuple[int, int] | None:
    left = max(start, low)
    right = min(end, high)
    return (left, right) if left < right else None


def merge_intervals(intervals: Sequence[Sequence[Any]]) -> list[tuple[int, int]]:
    ordered = sorted((int(row[0]), int(row[1])) for row in intervals if int(row[0]) < int(row[1]))
    merged: list[tuple[int, int]] = []
    for start, end in ordered:
        if not merged or merged[-1][1] < start:
            merged.append((start, end))
            continue
        merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def annotation_views(
    regions: Sequence[Any],
    specs: Sequence[Any],
    raw_words: Sequence[Any] | None,
    *,
    scored_start: int,
    scored_end: int,
) -> dict[str, Any]:
    singleton: list[tuple[int, int, str]] = []
    overlap: list[tuple[int, int]] = []
    unscored: list[tuple[int, int]] = []
    for region in regions:
        clipped = clip_interval(
            int(field(region, "start_sample")),
            int(field(region, "end_sample")),
            scored_start,
            scored_end,
        )
        if clipped is None:
            continue
        speakers = sorted(str(value) for value in field(region, "speakers"))
        ambiguous = bool(field(region, "ambiguous"))
        if ambiguous:
            unscored.append(clipped)
        elif len(speakers) == 1:
            singleton.append((clipped[0], clipped[1], speakers[0]))
        elif len(speakers) > 1:
            overlap.append(clipped)
    pauses: list[tuple[int, int]] = []
    for spec in specs:
        action_kind = str(field(spec, "action_kind"))
        interval = field(spec, "acceptable_interval")
        clipped = clip_interval(int(interval[0]), int(interval[1]), scored_start, scored_end)
        if clipped is None:
            continue
        if action_kind == "neutral_pause":
            pauses.append(clipped)
        elif action_kind == "unscored":
            unscored.append(clipped)
    words: list[tuple[int, int]] = []
    for word in raw_words or []:
        start_seconds = field(word, "start_time_s")
        end_seconds = field(word, "end_time_s")
        if start_seconds is None or end_seconds is None or bool(field(word, "ambiguous")):
            continue
        clipped = clip_interval(
            round(float(start_seconds) * 16000),
            round(float(end_seconds) * 16000),
            scored_start,
            scored_end,
        )
        if clipped is not None:
            words.append(clipped)
    singleton.sort()
    overlap.sort()
    pauses.sort()
    words.sort()
    unscored = merge_intervals(unscored)
    if any(left[1] > right[0] for left, right in zip(singleton, singleton[1:])):
        raise Phase5InputError("singleton annotation intervals overlap")
    return {
        "singleton_intervals": singleton,
        "overlap_intervals": overlap,
        "pause_intervals": pauses,
        "word_intervals": words if raw_words is not None else None,
        "unscored_intervals": unscored,
        "word_timing_observable": raw_words is not None,
    }


def episode_references(episode: dict[str, Any]) -> list[ReferenceAction]:
    references = [ReferenceAction.from_dict(row) for row in episode["references"]]
    if any(
        reference.source_session_id != str(episode["session_id"])
        or reference.audio_epoch != int(episode["audio_epoch"])
        for reference in references
    ):
        raise Phase5InputError("episode reference identity drift")
    return references
