from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

try:
    import torch
except ImportError:
    torch = None


FRAME_SECONDS = 0.08
CHUNK_SECONDS = 30.0
CHUNK_FRAMES = int(round(CHUNK_SECONDS / FRAME_SECONDS))


class StreamingError(RuntimeError):
    pass


def chunk_boundaries(num_frames: int, chunk_frames: int = CHUNK_FRAMES) -> list[tuple[int, int]]:
    if num_frames <= 0 or chunk_frames <= 0:
        raise StreamingError("frame counts must be positive")
    return [(s, min(s + chunk_frames, num_frames)) for s in range(0, num_frames, chunk_frames)]


@dataclass(slots=True)
class StateCarrier:
    state: Any
    source_id: str
    detached_steps: int = 0

    def carry(self) -> Any:
        return self.state

    def detach(self) -> Any:
        state = self.state
        if torch is not None and isinstance(state, torch.Tensor):
            self.state = state.detach().clone()
        elif torch is not None and isinstance(state, (list, tuple)) and any(
            isinstance(s, torch.Tensor) for s in state
        ):
            self.state = type(state)(
                s.detach().clone() if isinstance(s, torch.Tensor) else s for s in state
            )
        elif torch is not None and isinstance(state, dict) and any(
            isinstance(v, torch.Tensor) for v in state.values()
        ):
            self.state = {
                k: v.detach().clone() if isinstance(v, torch.Tensor) else v
                for k, v in state.items()
            }
        elif isinstance(state, list):
            self.state = list(state)
        elif isinstance(state, tuple):
            self.state = tuple(state)
        elif isinstance(state, dict):
            self.state = dict(state)
        self.detached_steps += 1
        return self.state

    def reset(self, state: Any, source_id: str) -> None:
        self.state = state
        self.source_id = source_id
        self.detached_steps = 0


def max_abs_diff(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) != len(second):
        raise StreamingError("equivalence geometry differs")
    return max((abs(a - b) for a, b in zip(first, second)), default=0.0)


def chunk_equivalence(
    one_shot: Sequence[float], chunked: Sequence[float], tol: float
) -> dict[str, Any]:
    diff = max_abs_diff(one_shot, chunked)
    passed = diff <= tol
    return {"max_abs_diff": diff, "tol": tol, "passed": passed}
