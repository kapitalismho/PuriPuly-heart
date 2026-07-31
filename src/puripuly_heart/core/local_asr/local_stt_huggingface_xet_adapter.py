from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

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


def _default_worker_command(
    _request: HuggingFaceDownloadRequest,
    request_path: Path,
    event_path: Path,
) -> Sequence[str]:
    command = [sys.executable]
    if not getattr(sys, "frozen", False):
        command.extend(["-m", "puripuly_heart.main"])
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
        process: asyncio.subprocess.Process | None = None
        request_path: Path | None = None
        event_path: Path | None = None
        try:
            while not acquired:
                if cancel_event is not None and cancel_event.is_set():
                    raise LocalSTTDownloadPortCancelled("Hugging Face/Xet download cancelled")
                acquired = _XET_TRANSFER_LOCK.acquire(blocking=False)
                if not acquired:
                    await asyncio.sleep(0.05)

            request.local_dir.mkdir(parents=True, exist_ok=True)
            ipc_id = uuid4().hex
            request_path = request.local_dir / f".hf-xet-request-{ipc_id}.json"
            event_path = request.local_dir / f".hf-xet-events-{ipc_id}.jsonl"
            request_path.write_text(json.dumps(_worker_payload(request)), encoding="utf-8")
            event_path.touch()
            command = tuple(self._worker_command_factory(request, request_path, event_path))
            creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                creationflags=creationflags,
            )
            error_message: str | None = None
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
                            "invalid Hugging Face/Xet worker response"
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

            return_code = await process.wait()
            if return_code != 0 or completed_path is None:
                detail = error_message or f"worker exited with code {return_code}"
                raise LocalSTTDownloadPortError(detail)
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
            if request_path is not None:
                request_path.unlink(missing_ok=True)
            if event_path is not None:
                event_path.unlink(missing_ok=True)
            if acquired:
                _XET_TRANSFER_LOCK.release()


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


def run_huggingface_xet_worker(*, request_path: Path, event_path: Path) -> int:
    global _WORKER_EVENT_PATH
    previous_xet_cache = os.environ.get("HF_XET_CACHE")
    _WORKER_EVENT_PATH = event_path
    try:
        payload = json.loads(request_path.read_text(encoding="utf-8"))
        local_dir = Path(str(payload["local_dir"])).resolve()
        xet_cache_dir = local_dir / ".cache" / "xet"
        os.environ["HF_XET_CACHE"] = str(xet_cache_dir)
        from huggingface_hub import hf_hub_download

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
        _write_worker_message({"type": "error", "message": str(exc)})
        return 1
    finally:
        if previous_xet_cache is None:
            os.environ.pop("HF_XET_CACHE", None)
        else:
            os.environ["HF_XET_CACHE"] = previous_xet_cache
        _WORKER_EVENT_PATH = None


__all__ = ["HuggingFaceXetDownloadAdapter", "run_huggingface_xet_worker"]
