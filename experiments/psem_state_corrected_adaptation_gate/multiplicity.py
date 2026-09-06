from __future__ import annotations

from collections.abc import Sequence


FRAME_SECONDS = 0.08
SELECT_BLIND_SECONDS = 2.0
WARMUP_FRAMES = int(round(SELECT_BLIND_SECONDS / FRAME_SECONDS))


class MultiplicityError(RuntimeError):
    pass


def crop_frame_range(num_frames: int, crop_start_s: float, crop_end_s: float) -> range:
    if not (crop_end_s > crop_start_s + SELECT_BLIND_SECONDS):
        raise MultiplicityError("crop must extend beyond the 2s selection blind")
    select_start = crop_start_s + SELECT_BLIND_SECONDS
    first = int(select_start / FRAME_SECONDS) if select_start > 0 else 0
    last_exclusive = int((crop_end_s - 1e-9) / FRAME_SECONDS) + 1
    first = max(first, 0)
    last_exclusive = min(last_exclusive, num_frames)
    if first >= last_exclusive:
        return range(0, 0)
    return range(first, last_exclusive)


def build_multiplicity(
    num_frames: int,
    crops: Sequence[tuple[float, float]],
    valid: Sequence[bool] | None = None,
) -> list[int]:
    if num_frames <= 0:
        raise MultiplicityError("num_frames must be positive")
    if valid is not None and len(valid) != num_frames:
        raise MultiplicityError("validity geometry differs from source frames")
    delta = [0] * (num_frames + 1)
    for crop_start_s, crop_end_s in crops:
        span = crop_frame_range(num_frames, crop_start_s, crop_end_s)
        if span.start < span.stop:
            delta[span.start] += 1
            delta[span.stop] -= 1
    multiplicity = [0] * num_frames
    depth = 0
    if valid is None:
        for frame in range(num_frames):
            depth += delta[frame]
            multiplicity[frame] = depth
    else:
        for frame in range(num_frames):
            depth += delta[frame]
            multiplicity[frame] = depth if valid[frame] else 0
    return multiplicity


def expected_post_warmup_units(crops: Sequence[tuple[float, float]]) -> int:
    total = 0
    for crop_start_s, crop_end_s in crops:
        duration = crop_end_s - (crop_start_s + SELECT_BLIND_SECONDS)
        if duration > 0:
            total += int((duration - 1e-9) / FRAME_SECONDS) + 1
    return total
