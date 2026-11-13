from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from puripuly_heart.core.messages import ErrorDiagnostics


class LocalSTTDownloadPortError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        failure_code: str = "download_failed",
        cause_type: str | None = None,
        worker_exit_code: int | None = None,
        status_code: int | None = None,
        os_error_code: int | None = None,
        failure_diagnostics: ErrorDiagnostics | None = None,
    ) -> None:
        super().__init__(message)
        from puripuly_heart.core.diagnostics import sanitize_diagnostic_fields

        fields, _ = sanitize_diagnostic_fields(
            {
                "asset_failure_code": failure_code,
                "exception_class": cause_type,
                "exit_code": worker_exit_code,
                "errno": os_error_code,
            }
        )
        self.failure_code = str(fields.get("asset_failure_code", "worker_protocol_failed"))
        self.cause_type = fields.get("exception_class")
        self.worker_exit_code = fields.get("exit_code")
        self.status_code = (
            status_code if type(status_code) is int and 100 <= status_code <= 599 else None
        )
        self.os_error_code = fields.get("errno")
        self.failure_diagnostics = failure_diagnostics
        self.secondary_diagnostics: tuple[ErrorDiagnostics, ...] = ()


class LocalSTTDownloadPortCancelled(LocalSTTDownloadPortError):
    pass


@dataclass(frozen=True, slots=True)
class HuggingFaceDownloadRequest:
    repo_id: str
    revision: str
    remote_path: str
    local_dir: Path
    expected_size_bytes: int | None
    endpoint: str | None = None
    on_recovery: Callable[[ErrorDiagnostics], None] | None = None


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
