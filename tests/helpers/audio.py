from __future__ import annotations

from dataclasses import dataclass

from puripuly_heart.core.audio.format import AudioFrameF32


@dataclass(slots=True)
class FakeAudioSource:
    frames_list: list[AudioFrameF32]

    async def frames(self):
        for item in self.frames_list:
            yield item

    async def close(self) -> None:
        return
