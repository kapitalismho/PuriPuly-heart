from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import secrets
import sys
import tempfile
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast
from uuid import uuid4

from puripuly_heart.app.ports.gpu_worker import (
    GpuWorkerActivation,
    GpuWorkerClientPort,
    GpuWorkerClosedError,
    GpuWorkerDevice,
    GpuWorkerEvent,
    GpuWorkerMode,
    GpuWorkerProcessFactoryPort,
    GpuWorkerRequestError,
    GpuWorkerTranscription,
)
from puripuly_heart.core.lifecycle import LifecycleScope, start_lifecycle_task

GPU_WORKER_CONTRACT_VERSION = 2
GPU_WORKER_EXECUTABLE_NAME = "PuriPulyHeartGpuWorker.exe"
_MAX_FRAME_BYTES = 4 * 1024 * 1024
_MAX_STDERR_LINES = 64
_MAX_STDERR_LINE_CHARS = 2_000
_STDERR_FAILURE_FLUSH_SECONDS = 0.05

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DefaultGpuWorkerProcessFactory(GpuWorkerProcessFactoryPort):
    executable_path: Path | None = None
    command_prefix: tuple[str, ...] = ()
    startup_timeout_s: float = 10.0
    request_timeout_s: float = 300.0
    heartbeat_interval_ms: int = 500
    heartbeat_timeout_s: float = 3.0
    cooperative_shutdown_s: float = 2.0
    terminate_grace_s: float = 1.0

    async def start(self, *, mode: GpuWorkerMode) -> GpuWorkerClientPort:
        executable = self.executable_path or self.resolve_default_executable()
        if not executable.is_file():
            raise FileNotFoundError(executable)
        session_id = uuid4().hex
        auth_token = secrets.token_hex(32)
        temporary_directory = tempfile.TemporaryDirectory(prefix="puripuly-gpu-worker-")
        manifest_path = Path(temporary_directory.name) / "launch.json"
        connection_queue: asyncio.Queue[tuple[asyncio.StreamReader, asyncio.StreamWriter]] = (
            asyncio.Queue()
        )

        def accept_connection(
            reader: asyncio.StreamReader,
            writer: asyncio.StreamWriter,
        ) -> None:
            connection_queue.put_nowait((reader, writer))

        server = await asyncio.start_server(
            accept_connection,
            host="127.0.0.1",
            port=0,
            limit=_MAX_FRAME_BYTES + 1,
        )
        socket = server.sockets[0] if server.sockets else None
        if socket is None:
            server.close()
            await server.wait_closed()
            temporary_directory.cleanup()
            raise GpuWorkerClosedError("worker listener did not bind")
        port = int(socket.getsockname()[1])
        manifest = {
            "contract_version": GPU_WORKER_CONTRACT_VERSION,
            "session_id": session_id,
            "auth_token": auth_token,
            "connect_host": "127.0.0.1",
            "connect_port": port,
            "heartbeat_interval_ms": self.heartbeat_interval_ms,
            "mode": mode,
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=True, separators=(",", ":")),
            encoding="utf-8",
        )
        os.chmod(manifest_path, 0o600)
        process: asyncio.subprocess.Process | None = None
        try:
            process = await asyncio.create_subprocess_exec(
                *self.command_prefix,
                str(executable),
                "--config",
                str(manifest_path),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            reader, writer = await self._accept_authenticated_connection(
                connection_queue,
                session_id=session_id,
                auth_token=auth_token,
                process=process,
            )
        except BaseException:
            server.close()
            await server.wait_closed()
            if process is not None:
                await _terminate_process(
                    process,
                    terminate_grace_s=self.terminate_grace_s,
                )
            temporary_directory.cleanup()
            raise
        server.close()
        return _DefaultGpuWorkerClient(
            process=process,
            server=server,
            reader=reader,
            writer=writer,
            session_id=session_id,
            temporary_directory=temporary_directory,
            request_timeout_s=self.request_timeout_s,
            heartbeat_timeout_s=self.heartbeat_timeout_s,
            cooperative_shutdown_s=self.cooperative_shutdown_s,
            terminate_grace_s=self.terminate_grace_s,
        )

    async def _accept_authenticated_connection(
        self,
        queue: asyncio.Queue[tuple[asyncio.StreamReader, asyncio.StreamWriter]],
        *,
        session_id: str,
        auth_token: str,
        process: asyncio.subprocess.Process,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.startup_timeout_s
        scope = LifecycleScope(f"gpu-worker-startup-{session_id}")
        process_wait = start_lifecycle_task(scope, process.wait(), name="process-wait")
        attempt = 0
        try:
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise GpuWorkerClosedError("worker authentication timed out")
                attempt += 1
                connection_wait = start_lifecycle_task(
                    scope,
                    queue.get(),
                    name=f"connection-wait-{attempt}",
                )
                done, _pending = await asyncio.wait(
                    {connection_wait, process_wait},
                    timeout=remaining,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if process_wait in done:
                    raise GpuWorkerClosedError(
                        f"worker exited before authentication with code {process_wait.result()}"
                    )
                if connection_wait not in done:
                    raise GpuWorkerClosedError("worker authentication timed out")
                reader, writer = connection_wait.result()
                remaining = deadline - loop.time()
                if remaining <= 0:
                    await _close_writer(writer)
                    raise GpuWorkerClosedError("worker authentication timed out")
                frame_wait = start_lifecycle_task(
                    scope,
                    reader.readline(),
                    name=f"authentication-frame-{attempt}",
                )
                done, _pending = await asyncio.wait(
                    {frame_wait, process_wait},
                    timeout=remaining,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if process_wait in done:
                    await _close_writer(writer)
                    raise GpuWorkerClosedError(
                        f"worker exited before authentication with code {process_wait.result()}"
                    )
                if frame_wait not in done:
                    await _close_writer(writer)
                    raise GpuWorkerClosedError("worker authentication timed out")
                raw = frame_wait.result()
                try:
                    payload = _decode_frame(raw)
                except GpuWorkerClosedError:
                    await _close_writer(writer)
                    continue
                if _authenticated(
                    payload,
                    session_id=session_id,
                    auth_token=auth_token,
                ):
                    return reader, writer
                await _close_writer(writer)
        finally:
            await scope.close()

    @classmethod
    def default_executable_candidates(
        cls,
        *,
        sys_executable: Path | None = None,
        repo_root: Path | None = None,
    ) -> tuple[Path, Path]:
        executable = (sys_executable or Path(sys.executable)).resolve()
        root = repo_root or Path(__file__).resolve().parents[4]
        return (
            executable.with_name(GPU_WORKER_EXECUTABLE_NAME),
            root / "build" / "gpu_worker" / GPU_WORKER_EXECUTABLE_NAME,
        )

    @classmethod
    def resolve_default_executable(
        cls,
        *,
        sys_executable: Path | None = None,
        repo_root: Path | None = None,
    ) -> Path:
        packaged, staged = cls.default_executable_candidates(
            sys_executable=sys_executable,
            repo_root=repo_root,
        )
        if packaged.is_file():
            return packaged
        if staged.is_file():
            return staged
        return packaged


@dataclass(slots=True)
class _DefaultGpuWorkerClient(GpuWorkerClientPort):
    process: asyncio.subprocess.Process
    server: asyncio.AbstractServer
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    session_id: str
    temporary_directory: tempfile.TemporaryDirectory[str]
    request_timeout_s: float
    heartbeat_timeout_s: float
    cooperative_shutdown_s: float
    terminate_grace_s: float
    _scope: LifecycleScope = field(init=False, repr=False)
    _pending: dict[str, asyncio.Future[dict[str, object]]] = field(
        init=False, default_factory=dict, repr=False
    )
    _events: asyncio.Queue[GpuWorkerEvent | None] = field(init=False, repr=False)
    _write_lock: asyncio.Lock = field(init=False, repr=False)
    _close_lock: asyncio.Lock = field(init=False, repr=False)
    _close_complete: asyncio.Event = field(init=False, repr=False)
    _last_heartbeat: float = field(init=False, repr=False)
    _terminal_error: GpuWorkerClosedError | None = field(init=False, default=None, repr=False)
    _stderr_tail: deque[str] = field(init=False, repr=False)
    _stderr_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _closed: bool = field(init=False, default=False, repr=False)
    _closing: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        self._scope = LifecycleScope(f"gpu-worker-client-{self.session_id}")
        self._events = asyncio.Queue()
        self._write_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._close_complete = asyncio.Event()
        self._last_heartbeat = time.monotonic()
        self._stderr_tail = deque(maxlen=_MAX_STDERR_LINES)
        start_lifecycle_task(self._scope, self._read_frames(), name="frame-reader")
        start_lifecycle_task(self._scope, self._monitor_process(), name="process-monitor")
        start_lifecycle_task(self._scope, self._monitor_heartbeat(), name="heartbeat-monitor")
        if self.process.stdout is not None:
            start_lifecycle_task(
                self._scope,
                self._drain_stream(self.process.stdout),
                name="stdout-drain",
            )
        if self.process.stderr is not None:
            self._stderr_task = start_lifecycle_task(
                self._scope,
                self._capture_stderr(self.process.stderr),
                name="stderr-capture",
            )

    @property
    def pid(self) -> int | None:
        return self.process.pid

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def temporary_directory_path(self) -> Path:
        return Path(self.temporary_directory.name)

    @property
    def returncode(self) -> int | None:
        return self.process.returncode

    async def discover(self) -> tuple[GpuWorkerDevice, ...]:
        payload = await self._request("discover")
        raw_devices = payload.get("devices")
        if not isinstance(raw_devices, list):
            raise GpuWorkerClosedError("worker discovery response is invalid")
        return tuple(_device_from_payload(item) for item in raw_devices)

    async def activate(self, *, model_path: Path, device_id: str) -> GpuWorkerActivation:
        payload = await self._request(
            "activate",
            model_path=str(model_path.resolve()),
            device_id=device_id,
        )
        raw_activation = payload.get("activation")
        if not isinstance(raw_activation, dict):
            raise GpuWorkerClosedError("worker activation response is invalid")
        raw_device = raw_activation.get("device")
        if not isinstance(raw_device, dict):
            raise GpuWorkerClosedError("worker activation device is invalid")
        return GpuWorkerActivation(
            device=_device_from_payload(raw_device),
            model_load_seconds=_required_float(raw_activation, "model_load_seconds"),
            warmup_seconds=_required_float(raw_activation, "warmup_seconds"),
        )

    async def transcribe(
        self,
        *,
        request_id: str,
        channel: Literal["self", "peer"],
        audio_path: Path,
        language_hint: str | None = None,
        on_request_sent: Callable[[], None] | None = None,
    ) -> GpuWorkerTranscription:
        payload = await self._request(
            "transcribe",
            request_id=request_id,
            on_request_sent=on_request_sent,
            channel=channel,
            audio_path=str(audio_path.resolve()),
            language_hint=language_hint,
        )
        raw_transcription = payload.get("transcription")
        if not isinstance(raw_transcription, dict):
            raise GpuWorkerClosedError("worker transcription response is invalid")
        text = raw_transcription.get("text")
        detected_language = raw_transcription.get("detected_language")
        if not isinstance(text, str) or not (
            detected_language is None or isinstance(detected_language, str)
        ):
            raise GpuWorkerClosedError("worker transcription fields are invalid")
        return GpuWorkerTranscription(
            text=text,
            detected_language=detected_language,
            audio_seconds=_required_float(raw_transcription, "audio_seconds"),
            decode_seconds=_required_float(raw_transcription, "decode_seconds"),
            rtf=_required_float(raw_transcription, "rtf"),
        )

    async def cancel(self, target_request_id: str) -> None:
        await self._send(
            {
                "type": "cancel",
                "request_id": uuid4().hex,
                "target_request_id": target_request_id,
            }
        )

    async def next_event(self) -> GpuWorkerEvent:
        event = await self._events.get()
        if event is None:
            raise self._terminal_error or GpuWorkerClosedError("worker event stream closed")
        return event

    async def close(self) -> None:
        if self._close_complete.is_set():
            return
        try:
            await asyncio.shield(self._close_once())
        except asyncio.CancelledError:
            await self._close_once()
            raise

    async def force_close(self) -> None:
        self._closing = True
        self._closed = True
        self._fail_pending(GpuWorkerClosedError("worker client force-closed"))
        if self.process.returncode is None:
            await _terminate_process(
                self.process,
                terminate_grace_s=self.terminate_grace_s,
            )
        await _close_writer(self.writer)
        self.server.close()
        await self.server.wait_closed()
        await self._scope.close()
        self.temporary_directory.cleanup()
        self._closing = False
        self._close_complete.set()

    async def _close_once(self) -> None:
        async with self._close_lock:
            if self._close_complete.is_set():
                return
            self._closing = True
            self._closed = True
            try:
                self._fail_pending(GpuWorkerClosedError("worker client closed"))
                if self.process.returncode is None:
                    try:
                        await self._send(
                            {
                                "type": "shutdown",
                                "request_id": uuid4().hex,
                            },
                            allow_closing=True,
                        )
                    except GpuWorkerClosedError:
                        pass
                    try:
                        await asyncio.wait_for(
                            self.process.wait(),
                            timeout=self.cooperative_shutdown_s,
                        )
                    except TimeoutError:
                        await _terminate_process(
                            self.process,
                            terminate_grace_s=self.terminate_grace_s,
                        )
                await _close_writer(self.writer)
                await self.server.wait_closed()
                await self._scope.close()
            finally:
                self.temporary_directory.cleanup()
                self._closing = False
                self._close_complete.set()

    async def _request(
        self,
        request_type: str,
        *,
        request_id: str | None = None,
        on_request_sent: Callable[[], None] | None = None,
        **fields: object,
    ) -> dict[str, object]:
        resolved_request_id = request_id or uuid4().hex
        if resolved_request_id in self._pending:
            raise ValueError("GPU worker request_id is already active")
        future: asyncio.Future[dict[str, object]] = asyncio.get_running_loop().create_future()
        self._pending[resolved_request_id] = future
        try:
            await self._send(
                {
                    "type": request_type,
                    "request_id": resolved_request_id,
                    **fields,
                }
            )
            if on_request_sent is not None:
                on_request_sent()
            return await asyncio.wait_for(future, timeout=self.request_timeout_s)
        finally:
            self._pending.pop(resolved_request_id, None)

    async def _send(
        self,
        payload: dict[str, object],
        *,
        allow_closing: bool = False,
    ) -> None:
        if self._closed and not (allow_closing and self._closing):
            raise GpuWorkerClosedError("worker client is closed")
        frame = {
            **payload,
            "contract_version": GPU_WORKER_CONTRACT_VERSION,
            "session_id": self.session_id,
        }
        encoded = json.dumps(frame, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        if len(encoded) > _MAX_FRAME_BYTES:
            raise ValueError("GPU worker frame exceeds maximum size")
        async with self._write_lock:
            try:
                self.writer.write(encoded + b"\n")
                await self.writer.drain()
            except (ConnectionError, OSError) as exc:
                raise GpuWorkerClosedError("worker connection write failed") from exc

    async def _read_frames(self) -> None:
        try:
            while True:
                raw = await self.reader.readline()
                if not raw:
                    if not self._closing and self._terminal_error is None:
                        self._terminal_error = GpuWorkerClosedError(
                            "worker event stream closed",
                            code="event_stream_closed",
                            exit_code=self.process.returncode,
                        )
                        await asyncio.sleep(_STDERR_FAILURE_FLUSH_SECONDS)
                        self._log_failure_stderr(
                            failure_code="event_stream_closed",
                            exit_code=self.process.returncode,
                        )
                    return
                payload = _decode_frame(raw)
                if not _valid_session_frame(payload, self.session_id):
                    raise GpuWorkerClosedError("worker frame session contract is invalid")
                frame_type = payload.get("type")
                if frame_type == "heartbeat":
                    self._last_heartbeat = time.monotonic()
                    continue
                if frame_type == "response":
                    await self._handle_response(payload)
                    continue
                if frame_type == "event":
                    await self._handle_event(payload)
                    continue
                raise GpuWorkerClosedError("worker frame type is invalid")
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            error = GpuWorkerClosedError(
                "worker frame reader failed",
                code="frame_reader_failed",
                failure_type=type(exc).__name__,
            )
            self._log_failure_stderr(failure_code="frame_reader_failed")
            self._terminal_error = error
            self._fail_pending(error)
        finally:
            await self._events.put(None)

    async def _handle_response(self, payload: dict[str, object]) -> None:
        request_id = payload.get("request_id")
        if not isinstance(request_id, str):
            return
        future = self._pending.get(request_id)
        if future is None or future.done():
            return
        if payload.get("status") == "ok":
            self._stderr_tail.clear()
            response_payload = payload.get("payload")
            if isinstance(response_payload, dict):
                future.set_result(cast(dict[str, object], response_payload))
            else:
                future.set_exception(GpuWorkerClosedError("worker response payload is invalid"))
            return
        code = payload.get("error_code")
        response_payload = payload.get("payload")
        failure_code = code if isinstance(code, str) else "worker_failure"
        if failure_code == "cancelled":
            self._stderr_tail.clear()
        else:
            await asyncio.sleep(_STDERR_FAILURE_FLUSH_SECONDS)
            self._log_failure_stderr(
                failure_code=failure_code,
                request_id=request_id,
            )
        future.set_exception(
            GpuWorkerRequestError(
                failure_code,
                (
                    cast(dict[str, object], response_payload)
                    if isinstance(response_payload, dict)
                    else None
                ),
                attempt_started=payload.get("attempt_started") is True,
            )
        )

    async def _handle_event(self, payload: dict[str, object]) -> None:
        name = payload.get("event")
        request_id = payload.get("request_id")
        fields = payload.get("fields")
        if not isinstance(name, str) or not (request_id is None or isinstance(request_id, str)):
            raise GpuWorkerClosedError("worker event is invalid")
        if not isinstance(fields, dict):
            fields = {}
        await self._events.put(
            GpuWorkerEvent(
                name=name,
                request_id=request_id,
                fields=cast(dict[str, object], fields),
            )
        )

    async def _monitor_process(self) -> None:
        await self.process.wait()
        if not self._closing:
            if self._stderr_task is not None:
                await asyncio.gather(self._stderr_task, return_exceptions=True)
            error = GpuWorkerClosedError(
                "worker process exited",
                code="worker_process_exited",
                exit_code=self.process.returncode,
            )
            self._log_failure_stderr(
                failure_code="worker_process_exited",
                exit_code=self.process.returncode,
            )
            self._terminal_error = error
            self._fail_pending(error)

    async def _monitor_heartbeat(self) -> None:
        interval = max(0.1, min(self.heartbeat_timeout_s / 3.0, 1.0))
        while not self._closed:
            await asyncio.sleep(interval)
            if time.monotonic() - self._last_heartbeat <= self.heartbeat_timeout_s:
                continue
            error = GpuWorkerClosedError(
                "worker heartbeat timed out",
                code="heartbeat_timeout",
            )
            self._log_failure_stderr(failure_code="heartbeat_timeout")
            self._terminal_error = error
            self._fail_pending(error)
            if self.process.returncode is None:
                self.process.terminate()
            return

    async def _drain_stream(self, stream: asyncio.StreamReader) -> None:
        while await stream.readline():
            pass

    async def _capture_stderr(self, stream: asyncio.StreamReader) -> None:
        while raw_line := await stream.readline():
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line:
                continue
            if len(line) > _MAX_STDERR_LINE_CHARS:
                line = f"{line[:_MAX_STDERR_LINE_CHARS]}…"
            self._stderr_tail.append(line)

    def _log_failure_stderr(
        self,
        *,
        failure_code: str,
        request_id: str | None = None,
        exit_code: int | None = None,
    ) -> None:
        stderr_tail = tuple(self._stderr_tail)
        self._stderr_tail.clear()
        if not stderr_tail:
            return
        logger.error(
            "[GPUWorker][Failure] failure_code=%s request_id=%s exit_code=%s stderr_tail=%s",
            failure_code,
            request_id or "none",
            exit_code if exit_code is not None else "none",
            json.dumps(stderr_tail, ensure_ascii=False),
        )

    def _fail_pending(self, error: BaseException) -> None:
        for future in tuple(self._pending.values()):
            if not future.done():
                future.set_exception(error)


def _decode_frame(raw: bytes) -> dict[str, object]:
    if not raw or len(raw) > _MAX_FRAME_BYTES:
        raise GpuWorkerClosedError("worker frame size is invalid")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GpuWorkerClosedError("worker frame JSON is invalid") from exc
    if not isinstance(payload, dict):
        raise GpuWorkerClosedError("worker frame must be an object")
    return cast(dict[str, object], payload)


def _authenticated(
    payload: dict[str, object],
    *,
    session_id: str,
    auth_token: str,
) -> bool:
    token = payload.get("auth_token")
    return (
        payload.get("type") == "authenticate"
        and payload.get("contract_version") == GPU_WORKER_CONTRACT_VERSION
        and payload.get("session_id") == session_id
        and payload.get("worker") == "PuriPulyHeartGpuWorker"
        and isinstance(token, str)
        and hmac.compare_digest(token, auth_token)
    )


def _valid_session_frame(payload: dict[str, object], session_id: str) -> bool:
    return (
        payload.get("contract_version") == GPU_WORKER_CONTRACT_VERSION
        and payload.get("session_id") == session_id
    )


def _device_from_payload(payload: object) -> GpuWorkerDevice:
    if not isinstance(payload, dict):
        raise GpuWorkerClosedError("worker device payload is invalid")
    for field_name in ("device_id", "name", "description", "device_type"):
        if not isinstance(payload.get(field_name), str):
            raise GpuWorkerClosedError("worker device string field is invalid")
    return GpuWorkerDevice(
        device_id=cast(str, payload["device_id"]),
        registry_index=_required_int(payload, "registry_index"),
        name=cast(str, payload["name"]),
        description=cast(str, payload["description"]),
        device_type=cast(str, payload["device_type"]),
        memory_total_bytes=_required_int(payload, "memory_total_bytes"),
        memory_free_bytes=_required_int(payload, "memory_free_bytes"),
    )


def _required_int(payload: dict[object, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise GpuWorkerClosedError(f"worker numeric field is invalid: {key}")
    return value


def _required_float(payload: dict[object, object], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GpuWorkerClosedError(f"worker numeric field is invalid: {key}")
    return float(value)


async def _close_writer(writer: asyncio.StreamWriter) -> None:
    writer.close()
    try:
        await writer.wait_closed()
    except (ConnectionError, OSError):
        pass


async def _terminate_process(
    process: asyncio.subprocess.Process,
    *,
    terminate_grace_s: float,
) -> None:
    if process.returncode is None:
        try:
            process.terminate()
        except ProcessLookupError:
            pass
    try:
        await asyncio.wait_for(process.wait(), timeout=terminate_grace_s)
    except TimeoutError:
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        await process.wait()


__all__ = [
    "DefaultGpuWorkerProcessFactory",
    "GPU_WORKER_CONTRACT_VERSION",
    "GPU_WORKER_EXECUTABLE_NAME",
]
