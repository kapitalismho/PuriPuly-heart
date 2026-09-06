from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class AnchorEpisode:
    episode_id: str
    anchor_speaker: str
    start_frame: int
    end_frame: int


@dataclass(frozen=True, slots=True)
class SourceAuthority:
    source_id: str
    num_frames: int
    episodes: tuple[AnchorEpisode, ...]
    y_anchor: tuple[float, ...]
    y_replace: tuple[float, ...]
    valid: tuple[bool, ...]
    ledger: dict[str, Any]


class LifecycleError(RuntimeError):
    pass


def _episode_at(episodes: tuple[AnchorEpisode, ...], frame: int) -> AnchorEpisode | None:
    for episode in episodes:
        if episode.start_frame <= frame < episode.end_frame:
            return episode
    return None


def build_source_authority(
    source_id: str,
    num_frames: int,
    episodes: Sequence[AnchorEpisode],
    active_by_frame: Sequence[Sequence[str]] | None = None,
    valid_by_frame: Sequence[bool] | None = None,
) -> SourceAuthority:
    ordered = tuple(sorted(tuple(episodes), key=lambda e: (e.start_frame, e.end_frame)))
    if num_frames <= 0:
        raise LifecycleError("num_frames must be positive")
    for episode in ordered:
        if not (0 <= episode.start_frame < episode.end_frame <= num_frames):
            raise LifecycleError("episode out of source range")
    for first, second in zip(ordered, ordered[1:]):
        if second.start_frame < first.end_frame:
            raise LifecycleError("episodes must not overlap")
    if active_by_frame is None:
        active_by_frame = [() for _ in range(num_frames)]
    if valid_by_frame is None:
        valid_by_frame = [True for _ in range(num_frames)]
    if len(active_by_frame) != num_frames or len(valid_by_frame) != num_frames:
        raise LifecycleError("activity/validity geometry differs from source frames")
    y_anchor: list[float] = [0.0] * num_frames
    y_replace: list[float] = [0.0] * num_frames
    valid: list[bool] = [bool(v) for v in valid_by_frame]
    opportunities: list[dict[str, Any]] = []
    previous_replace = 0.0
    for frame in range(num_frames):
        active = tuple(active_by_frame[frame])
        episode = _episode_at(ordered, frame)
        current_replace = 0.0
        if valid[frame] and episode is not None:
            anchor_active = episode.anchor_speaker in active
            y_anchor[frame] = 1.0 if anchor_active else 0.0
            current_replace = 1.0 if (len(active) > 0 and not anchor_active) else 0.0
            y_replace[frame] = current_replace
        if current_replace == 1.0 and previous_replace == 0.0 and episode is not None:
            opportunities.append({"episode_id": episode.episode_id, "frame": frame})
        previous_replace = current_replace
    ledger: dict[str, Any] = {
        "source_id": source_id,
        "num_frames": num_frames,
        "episodes": [
            {
                "episode_id": e.episode_id,
                "anchor_speaker": e.anchor_speaker,
                "start_frame": e.start_frame,
                "end_frame": e.end_frame,
            }
            for e in ordered
        ],
        "opportunities": opportunities,
        "positive_frames": int(sum(y_replace)),
        "anchor_frames": int(sum(y_anchor)),
        "valid_frames": int(sum(1 for v in valid if v)),
    }
    return SourceAuthority(
        source_id=source_id,
        num_frames=num_frames,
        episodes=ordered,
        y_anchor=tuple(y_anchor),
        y_replace=tuple(y_replace),
        valid=tuple(valid),
        ledger=ledger,
    )


def authority_slice(authority: SourceAuthority, start: int, end: int) -> dict[str, tuple]:
    if not (0 <= start <= end <= authority.num_frames):
        raise LifecycleError("slice outside source range")
    return {
        "y_anchor": authority.y_anchor[start:end],
        "y_replace": authority.y_replace[start:end],
        "valid": authority.valid[start:end],
    }
