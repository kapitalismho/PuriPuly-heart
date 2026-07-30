from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol


class LocalSTTDownloadPortError(RuntimeError):
    pass


class LocalSTTDownloadPortCancelled(LocalSTTDownloadPortError):
    pass


@dataclass(frozen=True, slots=True)
class HuggingFaceDownloadRequest:
    repo_id: str
    revision: str
    remote_path: str
    local_dir: Path
    expected_size_bytes: int | None


@dataclass(frozen=True, slots=True)
class HuggingFaceDownloadProgress:
    downloaded_bytes: int
    total_bytes: int | None


HuggingFaceProgressCallback = Callable[[HuggingFaceDownloadProgress], None]


class HuggingFaceDownloadPort(Protocol):
    async def download(
        self,
        request: HuggingFaceDownloadRequest,
        *,
        cancel_event: threading.Event | None,
        on_progress: HuggingFaceProgressCallback | None,
    ) -> Path: ...


__all__ = [
    "HuggingFaceDownloadPort",
    "HuggingFaceDownloadProgress",
    "HuggingFaceDownloadRequest",
    "LocalSTTDownloadPortCancelled",
    "LocalSTTDownloadPortError",
]
