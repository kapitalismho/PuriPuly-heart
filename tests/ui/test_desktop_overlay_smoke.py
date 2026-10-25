from __future__ import annotations

from pathlib import Path

import pytest

from puripuly_heart.ui.desktop_overlay_smoke import _capture_caption_frames


@pytest.mark.asyncio
@pytest.mark.parametrize("frames", [(4792,), (0, 4792), (0, 0)])
async def test_caption_observation_retains_missing_frames_and_bounds_extra_wait(
    tmp_path: Path, frames: tuple[int, ...]
) -> None:
    class Desktop:
        def __init__(self) -> None:
            self.captures: list[tuple[int, Path]] = []
            self.waits: list[float] = []

        def capture(self, hwnd: int, path: Path) -> int:
            self.captures.append((hwnd, path))
            return frames[len(self.captures) - 1]

        async def wait(self, seconds: float) -> None:
            self.waits.append(seconds)

    desktop = Desktop()

    observation = await _capture_caption_frames(desktop, 101, tmp_path, "relocked")

    assert observation["initial_pixels"] == frames[0]
    assert observation["final_pixels"] == frames[-1]
    assert desktop.captures[0] == (101, tmp_path / "relocked.png")
    assert len(desktop.captures) == len(frames)
    if len(frames) == 2:
        assert desktop.captures[1] == (101, tmp_path / "relocked-subsequent.png")
        assert desktop.waits == [0.3]
        assert observation["subsequent_frame_after_s"] == 0.3
    else:
        assert desktop.waits == []
