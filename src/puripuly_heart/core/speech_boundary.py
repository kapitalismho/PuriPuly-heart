from __future__ import annotations

from typing import Literal, TypeAlias

SpeechBoundaryReason: TypeAlias = Literal["silence", "soft_pause", "max_duration"]


def boundary_wait_ms(
    reason: SpeechBoundaryReason | None,
    *,
    observed_tail_ms: int,
) -> int | None:
    return observed_tail_ms if reason is not None else None
