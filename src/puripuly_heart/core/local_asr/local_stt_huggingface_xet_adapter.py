from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from puripuly_heart.core.diagnostics import sanitize_diagnostics
from puripuly_heart.core.external_network import ProxyPolicy, select_tls
from puripuly_heart.core.messages import ErrorDiagnostics
from puripuly_heart.core.model_asset_diagnostics import model_asset_failure_diagnostics
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
_WORKER_PROXY_POLICY_ENV = "PURIPULY_HEART_HF_PROXY_POLICY"
_NATIVE_EMPTY_BYPASS = ","
_NATIVE_ALL_BYPASS = "*,0.0.0.0/0,::/0"


def _default_worker_command(
    _request: HuggingFaceDownloadRequest,
    request_path: Path,
    event_path: Path,
) -> Sequence[str]:
    layout = current_runtime_layout()
    command = [str(layout.host_executable)]
    if layout.host_kind == "source":
        command.extend(["-m", "puripuly_heart.main"])
    else:
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
        "endpoint": request.endpoint,
    }


def _worker_policy(environment: dict[str, str]) -> ProxyPolicy:
    snapshot = environment.get(_WORKER_PROXY_POLICY_ENV)
    if snapshot is None:
        return ProxyPolicy(environment=environment)
    selected = json.loads(snapshot)
    return ProxyPolicy(
        environment={f"{name}_proxy": value for name, value in selected["environment"].items()},
        system=selected["system"],
        system_bypass=selected["system_bypass"],
    )


def _worker_environment() -> dict[str, str]:
    environment = os.environ.copy()
    tls = select_tls(environment=environment)
    policy = _worker_policy(environment).snapshot()
    environment[_WORKER_PROXY_POLICY_ENV] = json.dumps(
        {
            "environment": policy.environment,
            "system": policy.system,
            "system_bypass": policy.system_bypass,
        }
    )
    environment["HF_XET_TELEMETRY_ENABLED"] = "0"
    environment["HF_HUB_DISABLE_TELEMETRY"] = "1"
    proxy_urls = [
        policy.environment.get(scheme)
        or policy.environment.get("all")
        or policy.system.get(scheme)
        or policy.system.get("all")
        for scheme in ("http", "https")
    ]
    uses_system = any(
        not (policy.environment.get(scheme) or policy.environment.get("all"))
        and (policy.system.get(scheme) or policy.system.get("all"))
        for scheme in ("http", "https")
    )
    incompatible_proxy = (
        bool(any(proxy_urls) and policy.environment.get("no"))
        or bool(uses_system and policy.system_bypass)
        or bool(any(proxy_urls) and not all(proxy_urls))
        or any(
            urlsplit(value).scheme not in ("http", "https", "socks5", "socks5h")
            for value in proxy_urls
            if value
        )
    )
    if tls.source in ("explicit_file", "explicit_directory") or incompatible_proxy:
        environment["HF_HUB_DISABLE_XET"] = "1"
        return environment
    for name in tuple(environment):
        if name.lower().endswith("_proxy"):
            environment.pop(name)
    environment["NO_PROXY"] = _NATIVE_EMPTY_BYPASS if any(proxy_urls) else _NATIVE_ALL_BYPASS
    for scheme, proxy_url in zip(("http", "https"), proxy_urls):
        if proxy_url:
            urlsplit(proxy_url).port
            environment[f"{scheme.upper()}_PROXY"] = proxy_url
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
        self._active_workers: dict[int, str] = {}
        self._recovery_diagnostics: deque[ErrorDiagnostics] = deque(maxlen=8)

    @property
    def recovery_diagnostics(self) -> tuple[ErrorDiagnostics, ...]:
        return tuple(self._recovery_diagnostics)

    @property
    def child_states(self) -> tuple[str, ...]:
        return tuple(
            f"hf-xet-worker:pid={pid}:{state}"
            for pid, state in sorted(self._active_workers.items())
        )

    async def download(
        self,
        request: HuggingFaceDownloadRequest,
        *,
        cancel_event: threading.Event | None,
        on_progress: HuggingFaceProgressCallback | None,
    ) -> Path:
        acquired = False
        first_failure: LocalSTTDownloadPortError | None = None
        try:
            while not acquired:
                if cancel_event is not None and cancel_event.is_set():
                    raise LocalSTTDownloadPortCancelled("Hugging Face/Xet download cancelled")
                acquired = _XET_TRANSFER_LOCK.acquire(blocking=False)
                if not acquired:
                    await asyncio.sleep(0.05)

            try:
                environment = _worker_environment()
                if request.endpoint is not None:
                    if request.endpoint not in {
                        "https://huggingface.co",
                        "https://hf-mirror.net",
                        "https://hf-mirror.com",
                    }:
                        raise ValueError("Unsupported public model endpoint")
                    environment["HF_ENDPOINT"] = request.endpoint
                    environment["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
                    environment["HF_HUB_ETAG_TIMEOUT"] = "15"
                    environment["HF_HUB_DOWNLOAD_TIMEOUT"] = "30"
                    if request.endpoint != "https://huggingface.co":
                        environment["HF_HUB_DISABLE_XET"] = "1"
                    huggingface_tokens = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
                    for name in huggingface_tokens:
                        environment.pop(name, None)
            except Exception as exc:
                raise LocalSTTDownloadPortError(
                    "Hugging Face download network configuration failed",
                    failure_code="network_configuration_failed",
                    cause_type=type(exc).__name__,
                    os_error_code=_os_error_code(exc),
                ) from exc
            http_only = environment.get("HF_HUB_DISABLE_XET", "").upper() in {
                "1",
                "ON",
                "YES",
                "TRUE",
            }
            request.local_dir.mkdir(parents=True, exist_ok=True)
            try:
                return await self._download_attempt(
                    request,
                    cancel_event=cancel_event,
                    on_progress=on_progress,
                    environment=environment,
                )
            except LocalSTTDownloadPortError as exc:
                if http_only or exc.failure_code != "download_failed":
                    raise
                first_failure = exc
            shutil.rmtree(request.local_dir / ".cache", ignore_errors=True)
            if cancel_event is not None and cancel_event.is_set():
                raise LocalSTTDownloadPortCancelled("Hugging Face/Xet download cancelled")
            try:
                result = await self._download_attempt(
                    request,
                    cancel_event=cancel_event,
                    on_progress=on_progress,
                    environment={**environment, "HF_HUB_DISABLE_XET": "1"},
                )
            except LocalSTTDownloadPortError as last_failure:
                first = model_asset_failure_diagnostics(
                    first_failure,
                    operation="download",
                    fields={"source_kind": "huggingface", "source_attempt": 1},
                )
                last = model_asset_failure_diagnostics(
                    last_failure,
                    operation="download",
                    fields={"source_kind": "huggingface", "source_attempt": 2},
                )
                last_failure.failure_diagnostics = sanitize_diagnostics(
                    ErrorDiagnostics(
                        component="model_assets",
                        operation="download",
                        code="local_asr.download_failed",
                        category="lifecycle",
                        visibility="diagnostic_only",
                        content_policy="metadata_only",
                        status_code=None,
                        retry_after_ms=None,
                        fields={},
                        causes=(first, last),
                    )
                )
                raise
            recovered = model_asset_failure_diagnostics(
                first_failure,
                operation="download",
                fields={"source_kind": "huggingface", "source_attempt": 1},
            )
            self._recovery_diagnostics.append(recovered)
            if request.on_recovery is not None:
                request.on_recovery(recovered)
            return result
        except BaseException as failure:
            if isinstance(failure, (asyncio.CancelledError, LocalSTTDownloadPortCancelled)):
                if first_failure is not None:
                    failure.secondary_diagnostics = (
                        model_asset_failure_diagnostics(
                            first_failure,
                            operation="download",
                            fields={"source_kind": "huggingface", "source_attempt": 1},
                        ),
                    )
            elif isinstance(failure, (Exception, BaseExceptionGroup)):
                failure.failure_diagnostics = model_asset_failure_diagnostics(
                    failure,
                    operation="download",
                    code="local_asr.download_failed",
                    fields={
                        "source_kind": (
                            "huggingface_mirror" if request.endpoint is not None else "huggingface"
                        ),
                    },
                )
            raise
        finally:
            if acquired:
                _XET_TRANSFER_LOCK.release()

    async def _download_attempt(
        self,
        request: HuggingFaceDownloadRequest,
        *,
        cancel_event: threading.Event | None,
        on_progress: HuggingFaceProgressCallback | None,
        environment: dict[str, str],
    ) -> Path:
        process: asyncio.subprocess.Process | None = None
        ipc_id = uuid4().hex
        request_path = request.local_dir / f".hf-xet-request-{ipc_id}.json"
        event_path = request.local_dir / f".hf-xet-events-{ipc_id}.jsonl"
        primary_failure: BaseException | None = None
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
                    env=environment,
                )
                if process.pid is not None:
                    self._active_workers[int(process.pid)] = "running"
            except Exception as exc:
                raise LocalSTTDownloadPortError(
                    "Hugging Face/Xet worker could not start",
                    failure_code="worker_spawn_failed",
                    cause_type=type(exc).__name__,
                    os_error_code=_os_error_code(exc),
                ) from exc
            error_failure_code: str | None = None
            error_cause_type: str | None = None
            error_status_code: int | None = None
            error_os_code: int | None = None
            completed_path: Path | None = None

            with event_path.open("r", encoding="utf-8") as event_stream:
                last_progress = time.monotonic()
                last_bytes = -1
                while True:
                    if cancel_event is not None and cancel_event.is_set():
                        raise LocalSTTDownloadPortCancelled("Hugging Face/Xet download cancelled")
                    line = event_stream.readline()
                    if not line:
                        if process.returncode is not None:
                            break
                        if time.monotonic() - last_progress > 90:
                            raise LocalSTTDownloadPortError(
                                "Download made no progress; retry or change source",
                                failure_code="download_timeout",
                            )
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
                    if not isinstance(message, dict):
                        raise LocalSTTDownloadPortError(
                            "Invalid model download worker response",
                            failure_code="worker_protocol_failed",
                        )
                    message_type = message.get("type")
                    if message_type == "progress" and on_progress is not None:
                        downloaded_bytes = message.get("downloaded_bytes")
                        total_bytes = message.get("total_bytes")
                        if (
                            type(downloaded_bytes) is not int
                            or not 0 <= downloaded_bytes < 2**63
                            or (
                                total_bytes is not None
                                and (type(total_bytes) is not int or not 0 <= total_bytes < 2**63)
                            )
                        ):
                            raise LocalSTTDownloadPortError(
                                "Invalid model download worker progress",
                                failure_code="worker_protocol_failed",
                            )
                        if downloaded_bytes > last_bytes:
                            last_progress = time.monotonic()
                            last_bytes = downloaded_bytes
                        on_progress(
                            HuggingFaceDownloadProgress(
                                downloaded_bytes=downloaded_bytes,
                                total_bytes=total_bytes,
                            )
                        )
                    elif message_type == "complete":
                        completed_path = Path(str(message["path"]))
                    elif message_type == "error":
                        native = LocalSTTDownloadPortError(
                            "Model download worker reported failure",
                            failure_code=message.get("failure_code"),
                            cause_type=message.get("error_type"),
                            status_code=message.get("status_code"),
                            os_error_code=message.get("os_error_code"),
                        )
                        error_failure_code = native.failure_code
                        error_cause_type = native.cause_type
                        error_status_code = native.status_code
                        error_os_code = native.os_error_code

            return_code = await process.wait()
            if return_code != 0 or completed_path is None:
                detail = "Model download worker failed"
                raise LocalSTTDownloadPortError(
                    detail,
                    failure_code=error_failure_code or "worker_exited",
                    cause_type=error_cause_type,
                    worker_exit_code=return_code,
                    status_code=error_status_code,
                    os_error_code=error_os_code,
                )
            return completed_path
        except BaseException as exc:
            primary_failure = exc
            raise
        finally:
            cleanup_failures: list[BaseException] = []
            interrupted: asyncio.CancelledError | None = None
            if process is not None and process.returncode is None:
                stop_task = asyncio.create_task(_stop_worker(process))
                while not stop_task.done():
                    try:
                        await asyncio.shield(stop_task)
                    except asyncio.CancelledError as cancellation:
                        interrupted = cancellation
                    except BaseException:
                        break
                try:
                    stop_task.result()
                except BaseException as cleanup:
                    cleanup_failures.append(cleanup)
            if process is not None and process.pid is not None and process.returncode is not None:
                self._active_workers.pop(int(process.pid), None)
            for path in (request_path, event_path):
                try:
                    path.unlink(missing_ok=True)
                except BaseException as cleanup:
                    cleanup_failures.append(cleanup)
            if cleanup_failures:
                if primary_failure is not None or interrupted is not None:
                    original = primary_failure or interrupted
                    raise BaseExceptionGroup(
                        "Model download and worker cleanup failed",
                        (original, *cleanup_failures),
                    ) from original
                raise BaseExceptionGroup("Model download worker cleanup failed", cleanup_failures)
            if interrupted is not None and primary_failure is None:
                raise interrupted


class _WorkerProgress:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.total = kwargs.get("total")
        self.n = int(kwargs.get("initial", 0) or 0)
        self._lock = threading.Lock()
        self._last_emit = 0.0

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
            if time.monotonic() - self._last_emit < 0.25 and self.n != self.total:
                return
            self._last_emit = time.monotonic()
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
    return value if type(value) is int and -(2**31) <= value < 2**31 else None


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
    previous_network_environment = {
        name: value
        for name, value in os.environ.items()
        if name.lower().endswith("_proxy")
        or name
        in (
            "HF_HUB_DISABLE_XET",
            "HF_XET_TELEMETRY_ENABLED",
            "HF_HUB_DISABLE_TELEMETRY",
            _WORKER_PROXY_POLICY_ENV,
        )
    }
    _WORKER_EVENT_PATH = event_path
    failure_code = "worker_request_failed"
    try:
        payload = json.loads(request_path.read_text(encoding="utf-8"))
        local_dir = Path(str(payload["local_dir"])).resolve()
        xet_cache_dir = local_dir / ".cache" / "xet"
        os.environ["HF_XET_CACHE"] = str(xet_cache_dir)
        failure_code = "network_configuration_failed"
        environment = _worker_environment()
        for name in tuple(os.environ):
            if name.lower().endswith("_proxy") and name not in environment:
                os.environ.pop(name)
        os.environ.update(environment)
        failure_code = "worker_import_failed"
        if payload.get("endpoint") in {"https://hf-mirror.net", "https://hf-mirror.com"}:
            from puripuly_heart.core.local_translation.public_download import download_public_file

            downloaded_path = download_public_file(payload, on_progress=_write_worker_message)
            _write_worker_message({"type": "complete", "path": str(downloaded_path)})
            return 0
        from huggingface_hub import hf_hub_download, set_client_factory
        from huggingface_hub.utils._http import hf_request_event_hook

        from .local_stt_huggingface_client import huggingface_client

        set_client_factory(
            partial(
                huggingface_client,
                policy=_worker_policy(environment),
                follow_redirects=True,
                timeout=30.0 if payload.get("endpoint") else None,
                event_hooks={"request": [hf_request_event_hook]},
            )
        )

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
            "message": (
                "Hugging Face download network configuration failed"
                if failure_code == "network_configuration_failed"
                else "Public model download failed"
            ),
        }
        status_code = _http_status_code(exc)
        if isinstance(exc, LocalSTTDownloadPortError):
            message["failure_code"] = exc.failure_code
            status_code = exc.status_code
        if status_code is not None:
            message["status_code"] = status_code
        os_error_code = _os_error_code(exc)
        if os_error_code is not None:
            message["os_error_code"] = os_error_code
        _write_worker_message(message)
        return 1
    finally:
        for name in tuple(os.environ):
            if name.lower().endswith("_proxy") or name in (
                "HF_HUB_DISABLE_XET",
                "HF_XET_TELEMETRY_ENABLED",
                "HF_HUB_DISABLE_TELEMETRY",
                _WORKER_PROXY_POLICY_ENV,
            ):
                os.environ.pop(name)
        os.environ.update(previous_network_environment)
        if previous_xet_cache is None:
            os.environ.pop("HF_XET_CACHE", None)
        else:
            os.environ["HF_XET_CACHE"] = previous_xet_cache
        _WORKER_EVENT_PATH = None


__all__ = ["HuggingFaceXetDownloadAdapter", "run_huggingface_xet_worker"]
