from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from puripuly_heart.runtime_layout import current_runtime_layout

from .local_stt_download_port import (
    HuggingFaceDownloadProgress,
    HuggingFaceDownloadRequest,
    HuggingFaceProgressCallback,
    LocalSTTDownloadPortCancelled,
    LocalSTTDownloadPortError,
)

WorkerCommandFactory = Callable[[HuggingFaceDownloadRequest, Path, Path], Sequence[str]]
_XET_TRANSFER_LOCK = threading.Lock()
_WORKER_STOP_TIMEOUT_S = 5.0
_WORKER_EVENT_LOCK = threading.Lock()
_WORKER_EVENT_PATH: Path | None = None
_SSL_PATH_ENV = (("SSL_CERT_FILE", "file"), ("SSL_CERT_DIR", "directory"))


def _default_worker_command(
    _request: HuggingFaceDownloadRequest,
    request_path: Path,
    event_path: Path,
) -> Sequence[str]:
    layout = current_runtime_layout()
    command = [str(layout.host_executable)]
    if layout.host_kind == "source":
        command.extend(["-m", "puripuly_heart.main"])
    elif layout.host_kind == "native":
        command.append("--headless")
    command.extend(
        [
            "hf-xet-download-worker",
            "--request-file",
            str(request_path),
            "--event-file",
            str(event_path),
        ]
    )
    return command


def _worker_payload(request: HuggingFaceDownloadRequest) -> dict[str, object]:
    return {
        "repo_id": request.repo_id,
        "revision": request.revision,
        "remote_path": request.remote_path,
        "local_dir": str(request.local_dir),
        "expected_size_bytes": request.expected_size_bytes,
    }


def _worker_environment(*, disable_xet: bool) -> dict[str, str]:
    environment = os.environ.copy()
    for name, path_type in _SSL_PATH_ENV:
        value = environment.get(name)
        if not value:
            continue
        path = Path(value)
        try:
            valid = path.is_file() if path_type == "file" else path.is_dir()
        except OSError:
            valid = False
        if not valid:
            environment.pop(name, None)
    if disable_xet:
        environment["HF_HUB_DISABLE_XET"] = "1"
    return environment


async def _stop_worker(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        await process.wait()
        return
    if os.name == "nt":
        taskkill = await asyncio.create_subprocess_exec(
            "taskkill",
            "/PID",
            str(process.pid),
            "/T",
            "/F",
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        try:
            await asyncio.wait_for(taskkill.wait(), timeout=_WORKER_STOP_TIMEOUT_S)
        except asyncio.TimeoutError:
            taskkill.kill()
            await taskkill.wait()
        try:
            await asyncio.wait_for(process.wait(), timeout=_WORKER_STOP_TIMEOUT_S)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
        return
    try:
        process.terminate()
    except ProcessLookupError:
        await process.wait()
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=_WORKER_STOP_TIMEOUT_S)
    except asyncio.TimeoutError:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        await process.wait()


class HuggingFaceXetDownloadAdapter:
    def __init__(self, *, worker_command_factory: WorkerCommandFactory | None = None) -> None:
        self._worker_command_factory = worker_command_factory or _default_worker_command

    async def download(
        self,
        request: HuggingFaceDownloadRequest,
        *,
        cancel_event: threading.Event | None,
        on_progress: HuggingFaceProgressCallback | None,
    ) -> Path:
        acquired = False
        try:
            while not acquired:
                if cancel_event is not None and cancel_event.is_set():
                    raise LocalSTTDownloadPortCancelled("Hugging Face/Xet download cancelled")
                acquired = _XET_TRANSFER_LOCK.acquire(blocking=False)
                if not acquired:
                    await asyncio.sleep(0.05)

            request.local_dir.mkdir(parents=True, exist_ok=True)
            try:
                return await self._download_attempt(
                    request,
                    cancel_event=cancel_event,
                    on_progress=on_progress,
                    disable_xet=False,
                )
            except LocalSTTDownloadPortError as exc:
                if exc.failure_code != "download_failed":
                    raise
            shutil.rmtree(request.local_dir / ".cache", ignore_errors=True)
            if cancel_event is not None and cancel_event.is_set():
                raise LocalSTTDownloadPortCancelled("Hugging Face/Xet download cancelled")
            return await self._download_attempt(
                request,
                cancel_event=cancel_event,
                on_progress=on_progress,
                disable_xet=True,
            )
        finally:
            if acquired:
                _XET_TRANSFER_LOCK.release()

    async def _download_attempt(
        self,
        request: HuggingFaceDownloadRequest,
        *,
        cancel_event: threading.Event | None,
        on_progress: HuggingFaceProgressCallback | None,
        disable_xet: bool,
    ) -> Path:
        process: asyncio.subprocess.Process | None = None
        ipc_id = uuid4().hex
        request_path = request.local_dir / f".hf-xet-request-{ipc_id}.json"
        event_path = request.local_dir / f".hf-xet-events-{ipc_id}.jsonl"
        try:
            request_path.write_text(json.dumps(_worker_payload(request)), encoding="utf-8")
            event_path.touch()
            command = tuple(self._worker_command_factory(request, request_path, event_path))
            creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            try:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                    creationflags=creationflags,
                    env=_worker_environment(disable_xet=disable_xet),
                )
            except Exception as exc:
                raise LocalSTTDownloadPortError(
                    "Hugging Face/Xet worker could not start",
                    failure_code="worker_spawn_failed",
                    cause_type=type(exc).__name__,
                    os_error_code=_os_error_code(exc),
                ) from exc
            error_message: str | None = None
            error_failure_code: str | None = None
            error_cause_type: str | None = None
            error_status_code: int | None = None
            error_os_code: int | None = None
            completed_path: Path | None = None

            with event_path.open("r", encoding="utf-8") as event_stream:
                while True:
                    if cancel_event is not None and cancel_event.is_set():
                        raise LocalSTTDownloadPortCancelled("Hugging Face/Xet download cancelled")
                    line = event_stream.readline()
                    if not line:
                        if process.returncode is not None:
                            break
                        await asyncio.sleep(0.05)
                        continue
                    try:
                        message = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise LocalSTTDownloadPortError(
                            "invalid Hugging Face/Xet worker response",
                            failure_code="worker_protocol_failed",
                            cause_type=type(exc).__name__,
                        ) from exc
                    message_type = message.get("type")
                    if message_type == "progress" and on_progress is not None:
                        on_progress(
                            HuggingFaceDownloadProgress(
                                downloaded_bytes=max(0, int(message["downloaded_bytes"])),
                                total_bytes=(
                                    int(message["total_bytes"])
                                    if message.get("total_bytes") is not None
                                    else None
                                ),
                            )
                        )
                    elif message_type == "complete":
                        completed_path = Path(str(message["path"]))
                    elif message_type == "error":
                        error_message = str(
                            message.get("message") or "Hugging Face/Xet worker failed"
                        )
                        error_failure_code = str(
                            message.get("failure_code") or "worker_reported_failure"
                        )
                        error_cause_type = str(message.get("error_type") or "unavailable")
                        error_status_code = _optional_int(message.get("status_code"))
                        error_os_code = _optional_int(message.get("os_error_code"))

            return_code = await process.wait()
            if return_code != 0 or completed_path is None:
                detail = error_message or f"worker exited with code {return_code}"
                raise LocalSTTDownloadPortError(
                    detail,
                    failure_code=error_failure_code or "worker_exited",
                    cause_type=error_cause_type,
                    worker_exit_code=return_code,
                    status_code=error_status_code,
                    os_error_code=error_os_code,
                )
            return completed_path
        except asyncio.CancelledError:
            if process is not None:
                await _stop_worker(process)
            raise
        except LocalSTTDownloadPortCancelled:
            if process is not None:
                await _stop_worker(process)
            raise
        finally:
            if process is not None and process.returncode is None:
                await _stop_worker(process)
            request_path.unlink(missing_ok=True)
            event_path.unlink(missing_ok=True)


class _WorkerProgress:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.total = kwargs.get("total")
        self.n = int(kwargs.get("initial", 0) or 0)
        self._lock = threading.Lock()

    def __enter__(self) -> _WorkerProgress:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def update(self, size_bytes: int | float | None = 1) -> None:
        increment = int(size_bytes or 0)
        if increment <= 0:
            return
        with self._lock:
            self.n += increment
            _write_worker_message(
                {
                    "type": "progress",
                    "downloaded_bytes": self.n,
                    "total_bytes": self.total,
                }
            )

    def set_postfix_str(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def refresh(self) -> None:
        return None

    def close(self) -> None:
        return None


def _write_worker_message(message: dict[str, object]) -> None:
    if _WORKER_EVENT_PATH is None:
        raise RuntimeError("Hugging Face/Xet worker event path is not configured")
    with _WORKER_EVENT_LOCK:
        with _WORKER_EVENT_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(message) + "\n")
            handle.flush()


def _optional_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _os_error_code(exc: BaseException) -> int | None:
    return _optional_int(getattr(exc, "winerror", None)) or _optional_int(
        getattr(exc, "errno", None)
    )


def _http_status_code(exc: BaseException) -> int | None:
    response = getattr(exc, "response", None)
    return _optional_int(getattr(response, "status_code", None))


def run_huggingface_xet_worker(*, request_path: Path, event_path: Path) -> int:
    global _WORKER_EVENT_PATH
    previous_xet_cache = os.environ.get("HF_XET_CACHE")
    _WORKER_EVENT_PATH = event_path
    failure_code = "worker_request_failed"
    try:
        payload = json.loads(request_path.read_text(encoding="utf-8"))
        local_dir = Path(str(payload["local_dir"])).resolve()
        xet_cache_dir = local_dir / ".cache" / "xet"
        os.environ["HF_XET_CACHE"] = str(xet_cache_dir)
        failure_code = "worker_import_failed"
        from huggingface_hub import hf_hub_download

        failure_code = "download_failed"
        downloaded_path = Path(
            hf_hub_download(
                repo_id=str(payload["repo_id"]),
                filename=str(payload["remote_path"]),
                revision=str(payload["revision"]),
                token=False,
                local_dir=local_dir,
                tqdm_class=_WorkerProgress,
            )
        )
        shutil.rmtree(local_dir / ".cache", ignore_errors=True)
        _write_worker_message({"type": "complete", "path": str(downloaded_path)})
        return 0
    except Exception as exc:
        message: dict[str, object] = {
            "type": "error",
            "failure_code": failure_code,
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        status_code = _http_status_code(exc)
        if status_code is not None:
            message["status_code"] = status_code
        os_error_code = _os_error_code(exc)
        if os_error_code is not None:
            message["os_error_code"] = os_error_code
        _write_worker_message(message)
        return 1
    finally:
        if previous_xet_cache is None:
            os.environ.pop("HF_XET_CACHE", None)
        else:
            os.environ["HF_XET_CACHE"] = previous_xet_cache
        _WORKER_EVENT_PATH = None


__all__ = ["HuggingFaceXetDownloadAdapter", "run_huggingface_xet_worker"]
