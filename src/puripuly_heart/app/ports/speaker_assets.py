from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class SpeakerAssetSnapshot:
    ready: bool
    artifact_path: Path
    runtime_dir: Path
    reason: str | None = None
    metadata: dict[str, object] | None = None


class SpeakerAssets(Protocol):
    async def inspect(self) -> SpeakerAssetSnapshot: ...
    async def install(
        self, *, download_consent: bool, source_path: Path | None = None
    ) -> SpeakerAssetSnapshot: ...
    async def cancel(self) -> None: ...
    async def close(self) -> None: ...
