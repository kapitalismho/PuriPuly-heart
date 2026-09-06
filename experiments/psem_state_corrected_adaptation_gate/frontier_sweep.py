from __future__ import annotations

from typing import Any

from experiments.psem_state_corrected_adaptation_gate.frontier import unique_thresholds


class SweepError(ValueError):
    pass


def checked_scores(scores: list[float]) -> list[float]:
    values = [float(s) for s in scores]
    for value in values:
        if value != value:
            raise SweepError("NaN score rejected")
        if value == float("inf"):
            raise SweepError("+inf score rejected")
    return values


def episode_runs(episode_ids: list[Any]) -> list[tuple[str, list[int]]]:
    runs: list[tuple[str, list[int]]] = []
    index = 0
    count = len(episode_ids)
    while index < count:
        key = str(episode_ids[index])
        start = index
        while index < count and episode_ids[index] == key:
            index += 1
        runs.append((key, list(range(start, index))))
    return runs


def simulate_episode(
    frames: list[int],
    episode_key: str,
    source_id: str,
    speakers: list[str],
    starts: list[int],
    ends: list[int],
    valid: list[bool],
    masked: list[bool],
    speech: list[bool],
    scores: list[float],
    frontiers: list[int],
    threshold: float,
    confirmation: int,
) -> tuple[Any, ...] | None:
    pending_boundary: int | None = None
    pending_samples = 0
    previous_end: int | None = None
    for i in frames:
        if not valid[i]:
            pending_boundary = None
            pending_samples = 0
            continue
        if masked[i]:
            continue
        start = starts[i]
        end = ends[i]
        if not speech[i]:
            continue
        if scores[i] < threshold:
            pending_boundary = None
            pending_samples = 0
            continue
        if previous_end is not None and start != previous_end:
            pending_boundary = None
            pending_samples = 0
        if pending_boundary is None:
            pending_boundary = start
        duration = end - start
        needed = confirmation - pending_samples
        if duration >= needed:
            qualifying = start + needed
            frontier = frontiers[i]
            emit = qualifying if qualifying >= frontier else frontier
            return (
                source_id,
                episode_key,
                str(speakers[i]),
                int(pending_boundary),
                int(frontier),
                int(emit),
                int(confirmation),
            )
        pending_samples += duration
        previous_end = end
    return None


def sweep_threshold_events(
    dev: Any, scores: list[float], confirmation_ms: int
) -> tuple[list[float], list[tuple[Any, ...]]]:
    values = checked_scores(scores)
    count = len(values)
    starts = [int(v) for v in dev.starts]
    ends = [int(v) for v in dev.ends]
    episode_ids = list(dev.episode_ids)
    valid = [bool(v) for v in dev.valid]
    masked = [bool(v) for v in dev.masked]
    speech = [bool(v) for v in dev.speech_present]
    frontiers = [int(v) for v in dev.frontiers]
    speakers = [str(v) for v in dev.episode_speakers]
    source_id = str(dev.source_id)
    for name, seq in (
        ("starts", starts),
        ("ends", ends),
        ("episode_ids", episode_ids),
        ("valid", valid),
        ("masked", masked),
        ("speech_present", speech),
        ("frontiers", frontiers),
        ("episode_speakers", speakers),
    ):
        if len(seq) != count:
            raise SweepError(f"session {name} length differs from scores")
    confirmation = int(confirmation_ms) * 16
    if confirmation <= 0:
        raise SweepError("confirmation must be positive")
    grid = unique_thresholds(values)
    runs = episode_runs(episode_ids)
    distinct: dict[str, list[float]] = {}
    for key, frames in runs:
        seen: list[float] = []
        for i in frames:
            value = values[i]
            if value not in seen:
                seen.append(value)
        seen.sort(reverse=True)
        distinct[key + "\x00" + str(frames[0])] = seen
    behaviors: dict[str, dict[float, tuple[Any, ...] | None]] = {}
    ordered_runs: list[tuple[list[int], str]] = []
    for key, frames in runs:
        rkey = key + "\x00" + str(frames[0])
        ordered_runs.append((frames, rkey))
        table: dict[float, tuple[Any, ...] | None] = {}
        for level in distinct[rkey]:
            table[level] = simulate_episode(
                frames,
                key,
                source_id,
                speakers,
                starts,
                ends,
                valid,
                masked,
                speech,
                values,
                frontiers,
                level,
                confirmation,
            )
        behaviors[rkey] = table
    keys_per_threshold: list[tuple[Any, ...]] = []
    pointers = {rkey: -1 for rkey in distinct}
    levels_of = {rkey: sorted(table.keys(), reverse=True) for rkey, table in behaviors.items()}
    for threshold in grid:
        events: list[Any] = []
        for frames, rkey in ordered_runs:
            levels = levels_of[rkey]
            pointer = pointers[rkey]
            while pointer + 1 < len(levels) and levels[pointer + 1] >= threshold:
                pointer += 1
            pointers[rkey] = pointer
            if pointer >= 0:
                event = behaviors[rkey][levels[pointer]]
                if event is not None:
                    events.append(event)
        keys_per_threshold.append(tuple(events))
    return grid, keys_per_threshold
