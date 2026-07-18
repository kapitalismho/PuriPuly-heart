from __future__ import annotations

import asyncio
import heapq
import inspect
import tempfile
import wave
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal
from uuid import uuid4

import numpy as np

from puripuly_heart.app.ports.gpu_worker import (
    GpuWorkerActivation,
    GpuWorkerClientPort,
    GpuWorkerClosedError,
    GpuWorkerDevice,
    GpuWorkerEvent,
    GpuWorkerProcessFactoryPort,
    GpuWorkerRequestError,
    GpuWorkerTranscription,
)
from puripuly_heart.core.clock import Clock, SystemClock
from puripuly_heart.core.lifecycle import LifecycleScope, start_lifecycle_task
from puripuly_heart.core.owned_thread import run_owned_thread_call

GPU_PENDING_TTL_SECONDS = 12.0
GPU_DISCOVERY_PENDING_SECONDS = 2.0
GPU_SAMPLE_RATE_HZ = 16_000
GPU_CHANNEL_CANCEL_SECONDS = 1.0
GPU_FORCE_CLOSE_SECONDS = 2.0
GpuASRChannel = Literal["self", "peer"]
GpuASRDiagnosticSink = Callable[["GpuASRDiagnostic"], Awaitable[None] | None]


class GpuASRRuntimeState(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    READY = "ready"
    FAILED = "failed"
    STOPPING = "stopping"
    CLOSED = "closed"


class GpuDiscoveryState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"


class GpuASRRuntimeError(RuntimeError):
    pass


class GpuASRManualRetryRequired(GpuASRRuntimeError):
    pass


class GpuASRDecodeDropped(GpuASRRuntimeError):
    pass


class GpuASRWorkExpired(GpuASRRuntimeError):
    pass


class GpuASRWorkDiscarded(GpuASRRuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class GpuASRDiagnostic:
    kind: str
    fields: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _ActivationConfig:
    model_path: Path
    model_id: str
    device_id: str


@dataclass(frozen=True, slots=True)
class _ActivationOutcome:
    activation: GpuWorkerActivation | None = None
    failure_code: str | None = None


@dataclass(order=True, slots=True)
class _PendingWork:
    speech_end_at: float
    sequence: int
    request_id: str = field(compare=False)
    channel: GpuASRChannel = field(compare=False)
    samples_f32: np.ndarray = field(compare=False)
    language_hint: str | None = field(compare=False)
    future: asyncio.Future[GpuWorkerTranscription] = field(compare=False)
    request_sent: asyncio.Event = field(
        compare=False,
        default_factory=asyncio.Event,
    )
    settled: asyncio.Event = field(
        compare=False,
        default_factory=asyncio.Event,
    )
    discard_reason: str | None = field(compare=False, default=None)


class SharedGpuASRRuntime:
    def __init__(
        self,
        *,
        process_factory: GpuWorkerProcessFactoryPort,
        clock: Clock | None = None,
        diagnostic_sink: GpuASRDiagnosticSink | None = None,
        pending_ttl_seconds: float = GPU_PENDING_TTL_SECONDS,
        discovery_pending_seconds: float = GPU_DISCOVERY_PENDING_SECONDS,
        channel_cancel_seconds: float = GPU_CHANNEL_CANCEL_SECONDS,
        force_close_seconds: float = GPU_FORCE_CLOSE_SECONDS,
        pending_reaper_interval_seconds: float | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._process_factory = process_factory
        self._clock = clock or SystemClock()
        self._diagnostic_sink = diagnostic_sink
        self._pending_ttl_seconds = pending_ttl_seconds
        self._discovery_pending_seconds = discovery_pending_seconds
        self._channel_cancel_seconds = channel_cancel_seconds
        self._force_close_seconds = force_close_seconds
        self._pending_reaper_interval_seconds = (
            min(0.25, max(0.01, pending_ttl_seconds / 2.0))
            if pending_reaper_interval_seconds is None
            else pending_reaper_interval_seconds
        )
        if self._pending_reaper_interval_seconds <= 0:
            raise ValueError("pending_reaper_interval_seconds must be > 0")
        self._sleep = sleep
        self._scope = LifecycleScope("shared-gpu-asr-runtime")
        self._lock = asyncio.Lock()
        self._discovery_lock = asyncio.Lock()
        self._queue_event = asyncio.Event()
        self._queue: list[_PendingWork] = []
        self._active_channels: set[GpuASRChannel] = set()
        self._client: GpuWorkerClientPort | None = None
        self._activation: GpuWorkerActivation | None = None
        self._config: _ActivationConfig | None = None
        self._active_work: _PendingWork | None = None
        self._sequence = 0
        self._generation = 0
        self._dispatcher_task: asyncio.Task[None] | None = None
        self._event_task: asyncio.Task[None] | None = None
        self._reaper_task: asyncio.Task[None] | None = None
        self._activation_task: asyncio.Task[_ActivationOutcome] | None = None
        self._temporary_directory: tempfile.TemporaryDirectory[str] | None = None
        self._state = GpuASRRuntimeState.STOPPED
        self._discovery_state = GpuDiscoveryState.IDLE
        self._last_failure_code: str | None = None
        self._decode_recovery_armed = False
        self._closed = False

    @property
    def state(self) -> GpuASRRuntimeState:
        return self._state

    @property
    def discovery_state(self) -> GpuDiscoveryState:
        return self._discovery_state

    @property
    def active_channels(self) -> frozenset[GpuASRChannel]:
        return frozenset(self._active_channels)

    @property
    def pending_count(self) -> int:
        return len(self._queue)

    @property
    def worker_pid(self) -> int | None:
        return self._client.pid if self._client is not None else None

    @property
    def last_failure_code(self) -> str | None:
        return self._last_failure_code

    @property
    def configured_device_id(self) -> str | None:
        return self._config.device_id if self._config is not None else None

    async def discover_devices(self) -> tuple[GpuWorkerDevice, ...]:
        async with self._discovery_lock:
            self._ensure_open()
            self._discovery_state = GpuDiscoveryState.RUNNING
            client = await self._process_factory.start(mode="discovery")
            task = start_lifecycle_task(
                self._scope,
                client.discover(),
                name=f"discovery-{uuid4().hex}",
            )
            try:
                done, _pending = await asyncio.wait({task}, timeout=self._discovery_pending_seconds)
                if not done:
                    self._discovery_state = GpuDiscoveryState.PENDING
                    await self._emit("discovery_pending", {})
                devices = await task
                self._discovery_state = GpuDiscoveryState.READY
                await self._emit("discovery_ready", {"device_count": len(devices)})
                return devices
            except BaseException as exc:
                if isinstance(exc, asyncio.CancelledError):
                    raise
                self._discovery_state = GpuDiscoveryState.FAILED
                await self._emit("discovery_failed", {"failure": _failure_code(exc)})
                raise
            finally:
                await client.close()

    async def activate_channel(
        self,
        channel: GpuASRChannel,
        *,
        model_path: Path,
        model_id: str,
        device_id: str,
    ) -> GpuWorkerActivation:
        config = _ActivationConfig(
            model_path=model_path.resolve(),
            model_id=model_id,
            device_id=device_id,
        )
        async with self._lock:
            self._ensure_open()
            if self._config is not None and self._config != config:
                raise GpuASRRuntimeError("active GPU channels must share one model and device")
            self._active_channels.add(channel)
            self._config = config
            if self._state == GpuASRRuntimeState.READY:
                if self._activation is None:
                    raise GpuASRRuntimeError("GPU activation state is inconsistent")
                return self._activation
            if self._state == GpuASRRuntimeState.FAILED:
                raise GpuASRManualRetryRequired(self._last_failure_code or "manual_retry_required")
            if self._state == GpuASRRuntimeState.STOPPED:
                task = self._begin_activation_locked(config)
            elif self._state == GpuASRRuntimeState.STARTING and self._activation_task is not None:
                task = self._activation_task
            else:
                raise GpuASRRuntimeError(f"GPU runtime is {self._state.value}")
        return await self._await_activation(task)

    async def retry(self) -> GpuWorkerActivation:
        async with self._lock:
            self._ensure_open()
            if self._state != GpuASRRuntimeState.FAILED:
                raise GpuASRRuntimeError("GPU runtime is not awaiting manual retry")
            if not self._active_channels or self._config is None:
                raise GpuASRRuntimeError("no active GPU channel remains")
            task = self._begin_activation_locked(self._config)
        return await self._await_activation(task)

    async def submit(
        self,
        channel: GpuASRChannel,
        samples_f32: np.ndarray,
        *,
        speech_end_at: float,
        language_hint: str | None = None,
    ) -> GpuWorkerTranscription:
        samples = np.asarray(samples_f32, dtype=np.float32)
        if samples.ndim != 1 or samples.size == 0:
            raise ValueError("GPU ASR audio must be a non-empty mono array")
        samples = np.ascontiguousarray(samples).copy()
        async with self._lock:
            self._ensure_open()
            if channel not in self._active_channels:
                raise GpuASRRuntimeError(f"GPU channel is not active: {channel}")
            if self._state == GpuASRRuntimeState.FAILED:
                raise GpuASRManualRetryRequired(self._last_failure_code or "manual_retry_required")
            recovery_starting = (
                self._state == GpuASRRuntimeState.STARTING
                and self._decode_recovery_armed
                and self._activation_task is not None
            )
            if self._state != GpuASRRuntimeState.READY and not recovery_starting:
                raise GpuASRRuntimeError(f"GPU runtime is {self._state.value}")
            self._sequence += 1
            future: asyncio.Future[GpuWorkerTranscription] = (
                asyncio.get_running_loop().create_future()
            )
            work = _PendingWork(
                speech_end_at=speech_end_at,
                sequence=self._sequence,
                request_id=uuid4().hex,
                channel=channel,
                samples_f32=samples,
                language_hint=language_hint,
                future=future,
            )
            heapq.heappush(self._queue, work)
            self._queue_event.set()
        return await future

    async def deactivate_channel(self, channel: GpuASRChannel) -> None:
        tasks: tuple[asyncio.Task[None], ...] = ()
        stop_error: Exception | None = None
        stopped = False
        active: _PendingWork | None = None
        client: GpuWorkerClientPort | None = None
        pending_discarded = 0
        async with self._lock:
            self._active_channels.discard(channel)
            pending_discarded = self._discard_channel_pending_locked(
                channel,
                "channel_disabled",
            )
            if self._active_channels or self._state in {
                GpuASRRuntimeState.STOPPED,
                GpuASRRuntimeState.CLOSED,
            }:
                candidate = self._active_work
                if candidate is not None and candidate.channel == channel:
                    candidate.discard_reason = "channel_disabled"
                    if not candidate.future.done():
                        candidate.future.set_exception(GpuASRWorkDiscarded("channel_disabled"))
                    active = candidate
                    client = self._client
            else:
                stopped = True
                tasks, stop_error, _terminal = await self._stop_locked(
                    final_state=GpuASRRuntimeState.STOPPED,
                    reason="last_channel_disabled",
                )
                self._config = None
                self._last_failure_code = None
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if stop_error is not None:
            raise stop_error
        if stopped:
            return
        active_cancelled = False
        if active is not None and client is not None:
            active_cancelled = await self._cancel_channel_work(active, client)
        await self._emit(
            "channel_deactivated",
            {
                "channel": channel,
                "pending_discarded": pending_discarded,
                "active_cancelled": active_cancelled,
                "worker_retained": bool(self._active_channels),
            },
        )

    async def close(self) -> None:
        tasks: tuple[asyncio.Task[None], ...] = ()
        stop_error: Exception | None = None
        terminal = True
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            self._active_channels.clear()
            tasks, stop_error, terminal = await self._stop_locked(
                final_state=GpuASRRuntimeState.CLOSED,
                reason="application_shutdown",
            )
            self._config = None
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if stop_error is not None:
            if not terminal:
                async with self._lock:
                    self._closed = False
            raise stop_error
        await self._scope.close()

    def _begin_activation_locked(
        self,
        config: _ActivationConfig,
    ) -> asyncio.Task[_ActivationOutcome]:
        self._state = GpuASRRuntimeState.STARTING
        self._last_failure_code = None
        self._decode_recovery_armed = False
        self._generation += 1
        generation = self._generation
        task = start_lifecycle_task(
            self._scope,
            self._run_activation(generation, config),
            name=f"activation-{generation}",
        )
        self._activation_task = task
        return task

    async def _await_activation(
        self,
        task: asyncio.Task[_ActivationOutcome],
    ) -> GpuWorkerActivation:
        outcome = await asyncio.shield(task)
        if outcome.activation is not None:
            return outcome.activation
        raise GpuASRManualRetryRequired(outcome.failure_code or "manual_retry_required")

    async def _run_activation(
        self,
        generation: int,
        config: _ActivationConfig,
    ) -> _ActivationOutcome:
        client: GpuWorkerClientPort | None = None
        ownership_transferred = False
        event_task: asyncio.Task[None] | None = None
        temporary_directory = tempfile.TemporaryDirectory(prefix="puripuly-gpu-asr-audio-")
        try:
            client = await self._process_factory.start(mode="persistent")
            async with self._lock:
                if generation != self._generation or self._state != GpuASRRuntimeState.STARTING:
                    raise GpuASRRuntimeError("GPU activation was superseded")
                self._client = client
                ownership_transferred = True
                event_task = start_lifecycle_task(
                    self._scope,
                    self._consume_events(generation, client),
                    name=f"worker-events-{generation}",
                )
                self._event_task = event_task
            activation = await client.activate(
                model_path=config.model_path,
                device_id=config.device_id,
            )
        except BaseException as exc:
            if event_task is not None:
                event_task.cancel()
            if client is not None and not client.is_closed:
                if not ownership_transferred:
                    await self._close_unowned_client(client)
                elif self._client is client and generation == self._generation:
                    await self._close_unowned_client(client)
            if event_task is not None:
                await asyncio.gather(event_task, return_exceptions=True)
            await run_owned_thread_call(temporary_directory.cleanup)
            if isinstance(exc, asyncio.CancelledError):
                raise
            failure_code = _failure_code(exc)
            async with self._lock:
                if generation != self._generation:
                    return _ActivationOutcome(failure_code=failure_code)
                self._activation_task = None
                self._client = None
                self._event_task = None
                self._state = GpuASRRuntimeState.FAILED
                self._last_failure_code = failure_code
            await self._emit(
                "activation_failed",
                {
                    "model": config.model_id,
                    "backend": "Vulkan",
                    **_failure_diagnostic_fields(exc),
                },
            )
            return _ActivationOutcome(failure_code=failure_code)
        async with self._lock:
            if (
                generation != self._generation
                or self._state != GpuASRRuntimeState.STARTING
                or not self._active_channels
            ):
                stale = True
            else:
                stale = False
                self._client = client
                self._activation = activation
                self._temporary_directory = temporary_directory
                self._activation_task = None
                self._state = GpuASRRuntimeState.READY
                self._last_failure_code = None
                self._dispatcher_task = start_lifecycle_task(
                    self._scope,
                    self._dispatch(generation, client),
                    name=f"dispatcher-{generation}",
                )
                self._reaper_task = start_lifecycle_task(
                    self._scope,
                    self._reap_pending_work(generation, client),
                    name=f"pending-ttl-reaper-{generation}",
                )
        if stale:
            if event_task is not None:
                event_task.cancel()
            if not client.is_closed:
                await self._close_unowned_client(client)
            if event_task is not None:
                await asyncio.gather(event_task, return_exceptions=True)
            await run_owned_thread_call(temporary_directory.cleanup)
            return _ActivationOutcome(failure_code="activation_cancelled")
        await self._emit(
            "activation_ready",
            {
                "model": config.model_id,
                "backend": "Vulkan",
                "device": activation.device.device_id,
                "model_load_seconds": activation.model_load_seconds,
                "warmup_seconds": activation.warmup_seconds,
            },
        )
        return _ActivationOutcome(activation=activation)

    async def _close_unowned_client(self, client: GpuWorkerClientPort) -> None:
        try:
            await asyncio.wait_for(client.close(), timeout=self._force_close_seconds)
        except BaseException:
            if not client.is_closed:
                await asyncio.wait_for(client.force_close(), timeout=self._force_close_seconds)

    async def _reap_pending_work(
        self,
        generation: int,
        client: GpuWorkerClientPort,
    ) -> None:
        while True:
            await self._sleep(self._pending_reaper_interval_seconds)
            expired: list[tuple[_PendingWork, float, str]] = []
            async with self._lock:
                if not self._is_current_ready(generation, client):
                    return
                now = self._clock.now()
                model_id = self._config.model_id if self._config is not None else "gpu"
                retained: list[_PendingWork] = []
                while self._queue:
                    work = heapq.heappop(self._queue)
                    queue_wait = max(0.0, now - work.speech_end_at)
                    if queue_wait >= self._pending_ttl_seconds:
                        work.discard_reason = "speech_end_ttl"
                        if not work.future.done():
                            work.future.set_exception(GpuASRWorkExpired("speech_end_ttl"))
                        work.settled.set()
                        expired.append((work, queue_wait, model_id))
                    else:
                        retained.append(work)
                self._queue = retained
                heapq.heapify(self._queue)
            for work, queue_wait, model_id in expired:
                await self._emit_work_expired(
                    work,
                    model_id=model_id,
                    queue_wait=queue_wait,
                )

    async def _stop_locked(
        self,
        *,
        final_state: GpuASRRuntimeState,
        reason: str,
    ) -> tuple[tuple[asyncio.Task[None], ...], Exception | None, bool]:
        self._state = GpuASRRuntimeState.STOPPING
        self._generation += 1
        self._discard_pending_locked(reason)
        active = self._active_work
        if active is not None and not active.future.done():
            active.future.set_exception(GpuASRWorkDiscarded(reason))
        client = self._client
        tasks = tuple(
            task
            for task in (
                self._activation_task,
                self._dispatcher_task,
                self._event_task,
                self._reaper_task,
            )
            if task is not None and task is not asyncio.current_task()
        )
        for task in tasks:
            task.cancel()
        stop_error: Exception | None = None
        terminal = True
        forced = False
        if client is not None:
            stop_error, terminal, forced = await self._stop_client_bounded(
                client,
                request_id=active.request_id if active is not None else None,
            )
        if not terminal:
            self._state = GpuASRRuntimeState.FAILED
            self._last_failure_code = "worker_shutdown_failed"
            self._queue_event.set()
            await self._emit(
                "worker_failed",
                {
                    "backend": "Vulkan",
                    "failure": self._last_failure_code,
                    "retry": "manual",
                    "fallback": "none",
                },
            )
            return tasks, stop_error, False
        await self._cleanup_temporary_directory_locked()
        self._client = None
        self._activation = None
        self._active_work = None
        self._dispatcher_task = None
        self._event_task = None
        self._activation_task = None
        self._reaper_task = None
        self._decode_recovery_armed = False
        self._queue_event.set()
        self._state = final_state
        await self._emit("worker_stopped", {"outcome": reason, "forced": forced})
        return tasks, stop_error, True

    async def _stop_client_bounded(
        self,
        client: GpuWorkerClientPort,
        *,
        request_id: str | None,
    ) -> tuple[Exception | None, bool, bool]:
        errors: list[Exception] = []
        pending_tasks: list[asyncio.Task[None]] = []
        force_required = False
        already_closed = False
        if request_id is not None:
            cancel_task = start_lifecycle_task(
                self._scope,
                client.cancel(request_id),
                name=f"shutdown-cancel:{request_id}",
            )
            pending_tasks.append(cancel_task)
            done, _pending = await asyncio.wait(
                {cancel_task},
                timeout=self._channel_cancel_seconds,
            )
            if not done:
                force_required = True
            else:
                try:
                    await cancel_task
                except GpuWorkerClosedError:
                    already_closed = True
                except Exception as exc:
                    errors.append(exc)
                    force_required = True
        if not force_required and not already_closed:
            close_task = start_lifecycle_task(
                self._scope,
                client.close(),
                name=f"shutdown-close:{id(client)}",
            )
            pending_tasks.append(close_task)
            done, _pending = await asyncio.wait(
                {close_task},
                timeout=self._force_close_seconds,
            )
            if not done:
                force_required = True
            else:
                try:
                    await close_task
                except GpuWorkerClosedError:
                    already_closed = True
                except Exception as exc:
                    errors.append(exc)
                    force_required = True
        terminal = True
        if force_required and not already_closed:
            force_task = start_lifecycle_task(
                self._scope,
                client.force_close(),
                name=f"shutdown-force-close:{id(client)}",
            )
            done, _pending = await asyncio.wait(
                {force_task},
                timeout=self._force_close_seconds,
            )
            if not done:
                force_task.cancel()
                errors.append(GpuASRRuntimeError("GPU worker force close timed out"))
                terminal = False
            else:
                try:
                    await force_task
                except Exception as exc:
                    errors.append(exc)
                    terminal = False
        for task in pending_tasks:
            if not task.done():
                task.cancel()
        if terminal and pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)
        if not errors:
            return None, terminal, force_required
        if len(errors) == 1:
            return errors[0], terminal, force_required
        return ExceptionGroup("GPU worker shutdown failed", errors), terminal, force_required

    async def _dispatch(self, generation: int, client: GpuWorkerClientPort) -> None:
        while True:
            await self._queue_event.wait()
            self._queue_event.clear()
            while True:
                initial_expiry: tuple[_PendingWork, float, str] | None = None
                async with self._lock:
                    if not self._is_current_ready(generation, client):
                        return
                    if not self._queue:
                        break
                    work = heapq.heappop(self._queue)
                    queue_wait = max(0.0, self._clock.now() - work.speech_end_at)
                    if queue_wait >= self._pending_ttl_seconds:
                        if not work.future.done():
                            work.future.set_exception(GpuASRWorkExpired("speech_end_ttl"))
                        config = self._config
                        initial_expiry = (
                            work,
                            queue_wait,
                            config.model_id if config else "gpu",
                        )
                    else:
                        self._active_work = work
                        config = self._config
                        temporary_directory = self._temporary_directory
                if initial_expiry is not None:
                    expired_work, expired_wait, expired_model = initial_expiry
                    await self._emit_work_expired(
                        expired_work,
                        model_id=expired_model,
                        queue_wait=expired_wait,
                    )
                    expired_work.settled.set()
                    continue
                if config is None or temporary_directory is None:
                    await self._fail_worker(
                        generation, client, GpuASRRuntimeError("GPU resources missing")
                    )
                    return
                audio_path = Path(temporary_directory.name) / f"{work.request_id}.wav"
                final_queue_wait = queue_wait
                transcription: GpuWorkerTranscription | None = None
                expired_after_staging = False
                restart_failure: GpuWorkerRequestError | None = None
                terminal_failure: BaseException | None = None
                try:
                    await run_owned_thread_call(
                        lambda: _write_pcm16_wav(audio_path, work.samples_f32)
                    )
                    final_queue_wait = max(
                        0.0,
                        self._clock.now() - work.speech_end_at,
                    )
                    if work.discard_reason is not None:
                        continue
                    if final_queue_wait >= self._pending_ttl_seconds:
                        expired_after_staging = True
                        await self._emit_work_expired(
                            work,
                            model_id=config.model_id,
                            queue_wait=final_queue_wait,
                        )
                        continue
                    transcription = await client.transcribe(
                        request_id=work.request_id,
                        channel=work.channel,
                        audio_path=audio_path,
                        language_hint=work.language_hint,
                        on_request_sent=work.request_sent.set,
                    )
                except BaseException as exc:
                    if isinstance(exc, asyncio.CancelledError):
                        if not work.future.done():
                            work.future.set_exception(GpuASRWorkDiscarded("dispatcher_cancelled"))
                        raise
                    expected_discard = (
                        work.discard_reason is not None
                        and isinstance(exc, GpuWorkerRequestError)
                        and exc.code == "cancelled"
                    )
                    if isinstance(exc, GpuWorkerRequestError) and exc.attempt_started:
                        await self._emit_attempt_failure(
                            work=work,
                            model_id=config.model_id,
                            queue_wait=final_queue_wait,
                            exception=exc,
                        )
                    elif work.discard_reason is None:
                        await self._emit(
                            "work_prestart_failed",
                            {
                                "channel": work.channel,
                                "model": config.model_id,
                                "provider": "gpu_qwen",
                                **_failure_diagnostic_fields(exc),
                                "queue_wait_seconds": final_queue_wait,
                            },
                        )
                    recoverable_decode_failure = (
                        not expected_discard
                        and isinstance(exc, GpuWorkerRequestError)
                        and exc.attempt_started
                        and exc.code == "decode_failure"
                        and not self._decode_recovery_armed
                    )
                    if recoverable_decode_failure:
                        restart_failure = exc
                    elif not expected_discard:
                        terminal_failure = exc
                    elif not work.future.done():
                        work.future.set_exception(exc)
                finally:
                    await run_owned_thread_call(lambda: audio_path.unlink(missing_ok=True))
                    async with self._lock:
                        if transcription is not None:
                            self._decode_recovery_armed = False
                        if transcription is not None and not work.future.done():
                            if (
                                work.discard_reason is None
                                and work.channel in self._active_channels
                            ):
                                work.future.set_result(transcription)
                            else:
                                work.future.set_exception(
                                    GpuASRWorkDiscarded(work.discard_reason or "channel_disabled")
                                )
                        if expired_after_staging and not work.future.done():
                            work.future.set_exception(GpuASRWorkExpired("speech_end_ttl"))
                        if (
                            self._active_work is work
                            and restart_failure is None
                            and terminal_failure is None
                        ):
                            self._active_work = None
                    work.settled.set()
                if restart_failure is not None:
                    await self._restart_after_decode_failure(
                        generation,
                        client,
                        work,
                        restart_failure,
                    )
                    return
                if terminal_failure is not None:
                    await self._fail_worker(generation, client, terminal_failure)
                    return
                if transcription is not None:
                    await self._emit(
                        "decode_attempt",
                        {
                            "channel": work.channel,
                            "model": config.model_id,
                            "backend": "Vulkan",
                            "audio_seconds": transcription.audio_seconds,
                            "decode_seconds": transcription.decode_seconds,
                            "rtf": transcription.rtf,
                            "result": work.discard_reason or "success",
                            "queue_wait_seconds": final_queue_wait,
                        },
                    )

    async def _restart_after_decode_failure(
        self,
        generation: int,
        client: GpuWorkerClientPort,
        work: _PendingWork,
        exception: GpuWorkerRequestError,
    ) -> None:
        async with self._lock:
            if not self._is_current_generation(generation, client):
                return
            config = self._config
            temporary_directory = self._temporary_directory
            if config is None:
                terminal_failure: BaseException | None = GpuASRRuntimeError(
                    "GPU recovery configuration is missing"
                )
            else:
                terminal_failure = None
                self._state = GpuASRRuntimeState.STARTING
                self._last_failure_code = exception.code
                self._decode_recovery_armed = True
                self._generation += 1
                recovery_generation = self._generation
                stale_tasks = tuple(
                    task
                    for task in (self._event_task, self._reaper_task)
                    if task is not None and task is not asyncio.current_task()
                )
                for task in stale_tasks:
                    task.cancel()
                self._activation = None
                self._active_work = None
                self._dispatcher_task = None
                self._event_task = None
                self._reaper_task = None
                self._temporary_directory = None
                recovery_task = start_lifecycle_task(
                    self._scope,
                    self._run_decode_recovery(
                        recovery_generation,
                        config,
                        client,
                        stale_tasks,
                        temporary_directory,
                        exception,
                    ),
                    name=f"decode-recovery-{recovery_generation}",
                )
                self._activation_task = recovery_task
                if not work.future.done():
                    work.future.set_exception(GpuASRDecodeDropped(exception.code))
                if self._active_work is work:
                    self._active_work = None
        if terminal_failure is not None:
            await self._fail_worker(generation, client, terminal_failure)
            return
        await self._emit(
            "worker_recovery_started",
            {
                "backend": "Vulkan",
                **_failure_diagnostic_fields(exception),
                "retry": "restart_only",
                "utterance_retry": False,
            },
        )

    async def _run_decode_recovery(
        self,
        generation: int,
        config: _ActivationConfig,
        previous_client: GpuWorkerClientPort,
        stale_tasks: tuple[asyncio.Task[None], ...],
        temporary_directory: tempfile.TemporaryDirectory[str] | None,
        exception: GpuWorkerRequestError,
    ) -> _ActivationOutcome:
        try:
            await self._close_unowned_client(previous_client)
            if stale_tasks:
                await asyncio.gather(*stale_tasks, return_exceptions=True)
            if temporary_directory is not None:
                await run_owned_thread_call(temporary_directory.cleanup)
        except asyncio.CancelledError:
            raise
        except BaseException as cleanup_exception:
            return await self._finish_decode_recovery_failure(
                generation,
                config,
                cleanup_exception,
            )
        outcome = await self._run_activation(generation, config)
        if outcome.activation is None:
            await self._finish_decode_recovery_failure(
                generation,
                config,
                GpuASRRuntimeError(outcome.failure_code or "worker_recovery_failed"),
                failure_code=outcome.failure_code,
            )
            return outcome
        await self._emit(
            "worker_recovery_ready",
            {
                "backend": "Vulkan",
                "failure": exception.code,
                "device": outcome.activation.device.device_id,
                "utterance_retry": False,
            },
        )
        return outcome

    async def _finish_decode_recovery_failure(
        self,
        generation: int,
        config: _ActivationConfig,
        exception: BaseException,
        *,
        failure_code: str | None = None,
    ) -> _ActivationOutcome:
        resolved_failure_code = failure_code or _failure_code(exception)
        async with self._lock:
            if generation != self._generation:
                return _ActivationOutcome(failure_code=resolved_failure_code)
            self._activation_task = None
            self._client = None
            self._activation = None
            self._event_task = None
            self._dispatcher_task = None
            self._reaper_task = None
            self._state = GpuASRRuntimeState.FAILED
            self._last_failure_code = resolved_failure_code
            self._discard_pending_locked("worker_recovery_failed")
            self._queue_event.set()
        diagnostic_fields = _failure_diagnostic_fields(exception)
        diagnostic_fields["failure"] = resolved_failure_code
        await self._emit(
            "worker_recovery_failed",
            {
                "model": config.model_id,
                "backend": "Vulkan",
                **diagnostic_fields,
                "retry": "manual",
                "fallback": "none",
                "utterance_retry": False,
            },
        )
        await self._emit(
            "worker_failed",
            {
                "backend": "Vulkan",
                **diagnostic_fields,
                "retry": "manual",
                "fallback": "none",
            },
        )
        return _ActivationOutcome(failure_code=resolved_failure_code)

    async def _consume_events(self, generation: int, client: GpuWorkerClientPort) -> None:
        try:
            while self._is_current_generation(generation, client):
                event = await client.next_event()
                await self._emit_worker_event(event)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            await self._fail_worker(generation, client, exc)

    async def _fail_worker(
        self,
        generation: int,
        client: GpuWorkerClientPort,
        exception: BaseException,
    ) -> None:
        async with self._lock:
            if not self._is_current_generation(generation, client):
                return
            self._state = GpuASRRuntimeState.FAILED
            self._last_failure_code = _failure_code(exception)
            self._discard_pending_locked("worker_failed")
            active = self._active_work
            if active is not None and not active.future.done():
                active.future.set_exception(exception)
            self._active_work = None
            self._client = None
            self._activation = None
            await client.close()
            await self._cleanup_temporary_directory_locked()
            self._queue_event.set()
            await self._emit(
                "worker_failed",
                {
                    "backend": "Vulkan",
                    **_failure_diagnostic_fields(exception),
                    "retry": "manual",
                    "fallback": "none",
                },
            )

    async def _emit_attempt_failure(
        self,
        *,
        work: _PendingWork,
        model_id: str,
        queue_wait: float,
        exception: BaseException,
    ) -> None:
        if not isinstance(exception, GpuWorkerRequestError):
            return
        fields: dict[str, object] = {
            "channel": work.channel,
            "model": model_id,
            "backend": "Vulkan",
            "result": _failure_code(exception),
            "queue_wait_seconds": queue_wait,
        }
        for key in ("audio_seconds", "decode_seconds", "rtf"):
            value = exception.fields.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                fields[key] = float(value)
        await self._emit("decode_attempt", fields)

    async def _emit_work_expired(
        self,
        work: _PendingWork,
        *,
        model_id: str,
        queue_wait: float,
    ) -> None:
        await self._emit(
            "work_expired",
            {
                "channel": work.channel,
                "model": model_id,
                "provider": "gpu_qwen",
                "expiry_reason": "speech_end_ttl",
                "queue_wait_seconds": queue_wait,
            },
        )

    async def _cancel_channel_work(
        self,
        work: _PendingWork,
        client: GpuWorkerClientPort,
    ) -> bool:
        if work.settled.is_set():
            return False
        request_sent = start_lifecycle_task(
            self._scope,
            work.request_sent.wait(),
            name=f"cancel-request-sent:{work.request_id}",
        )
        settled = start_lifecycle_task(
            self._scope,
            work.settled.wait(),
            name=f"cancel-settled:{work.request_id}",
        )
        try:
            done, _pending = await asyncio.wait(
                {request_sent, settled},
                timeout=self._channel_cancel_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                await self._escalate_channel_cancel_timeout(client)
                return True
        finally:
            for task in (request_sent, settled):
                if not task.done():
                    task.cancel()
            await asyncio.gather(request_sent, settled, return_exceptions=True)
        if work.settled.is_set():
            return False
        try:
            await asyncio.wait_for(
                client.cancel(work.request_id),
                timeout=self._channel_cancel_seconds,
            )
        except GpuWorkerClosedError:
            pass
        except TimeoutError:
            await self._escalate_channel_cancel_timeout(client)
            return True
        try:
            await asyncio.wait_for(
                work.settled.wait(),
                timeout=self._channel_cancel_seconds,
            )
        except TimeoutError:
            await self._escalate_channel_cancel_timeout(client)
        return True

    async def _escalate_channel_cancel_timeout(
        self,
        client: GpuWorkerClientPort,
    ) -> None:
        async with self._lock:
            if self._client is not client:
                return
            self._generation += 1
            self._state = GpuASRRuntimeState.FAILED
            self._last_failure_code = "channel_cancel_timeout"
            self._discard_pending_locked("channel_cancel_timeout")
            active = self._active_work
            if active is not None and not active.future.done():
                active.future.set_exception(GpuASRWorkDiscarded("channel_cancel_timeout"))
            tasks = tuple(
                task
                for task in (
                    self._activation_task,
                    self._dispatcher_task,
                    self._event_task,
                    self._reaper_task,
                )
                if task is not None and task is not asyncio.current_task()
            )
            for task in tasks:
                task.cancel()
            temporary_directory = self._temporary_directory
            self._client = None
            self._activation = None
            self._active_work = None
            self._dispatcher_task = None
            self._event_task = None
            self._activation_task = None
            self._reaper_task = None
            self._temporary_directory = None
            self._queue_event.set()
        close_task = start_lifecycle_task(
            self._scope,
            client.close(),
            name=f"channel-escalation-close:{id(client)}",
        )
        done, _pending = await asyncio.wait(
            {close_task},
            timeout=self._force_close_seconds,
        )
        if not done:
            force_close = getattr(client, "force_close", None)
            if callable(force_close):
                await asyncio.wait_for(force_close(), timeout=self._force_close_seconds)
            close_task.cancel()
        await asyncio.gather(close_task, *tasks, return_exceptions=True)
        if temporary_directory is not None:
            await run_owned_thread_call(temporary_directory.cleanup)
        await self._emit(
            "worker_failed",
            {
                "backend": "Vulkan",
                "failure": "channel_cancel_timeout",
                "retry": "manual",
                "fallback": "none",
            },
        )

    async def _emit_worker_event(self, event: GpuWorkerEvent) -> None:
        fields: dict[str, object] = {"event": event.name}
        for key in (
            "phase",
            "progress",
            "channel",
            "backend",
            "outcome",
            "active",
        ):
            value = event.fields.get(key)
            if isinstance(value, (str, int, float, bool)) or value is None:
                fields[key] = value
        await self._emit("worker_lifecycle", fields)

    def _discard_pending_locked(self, reason: str) -> None:
        while self._queue:
            work = heapq.heappop(self._queue)
            work.discard_reason = reason
            if not work.future.done():
                work.future.set_exception(GpuASRWorkDiscarded(reason))
            work.settled.set()

    def _discard_channel_pending_locked(
        self,
        channel: GpuASRChannel,
        reason: str,
    ) -> int:
        discarded = [work for work in self._queue if work.channel == channel]
        if not discarded:
            return 0
        self._queue = [work for work in self._queue if work.channel != channel]
        heapq.heapify(self._queue)
        for work in discarded:
            work.discard_reason = reason
            if not work.future.done():
                work.future.set_exception(GpuASRWorkDiscarded(reason))
            work.settled.set()
        return len(discarded)

    async def _cleanup_temporary_directory_locked(self) -> None:
        temporary_directory = self._temporary_directory
        self._temporary_directory = None
        if temporary_directory is not None:
            await run_owned_thread_call(temporary_directory.cleanup)

    def _is_current_generation(self, generation: int, client: GpuWorkerClientPort) -> bool:
        return generation == self._generation and self._client is client

    def _is_current_ready(self, generation: int, client: GpuWorkerClientPort) -> bool:
        return (
            self._is_current_generation(generation, client)
            and self._state == GpuASRRuntimeState.READY
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise GpuASRRuntimeError("GPU runtime is closed")

    async def _emit(self, kind: str, fields: Mapping[str, object]) -> None:
        if self._diagnostic_sink is None:
            return
        result = self._diagnostic_sink(GpuASRDiagnostic(kind=kind, fields=fields))
        if inspect.isawaitable(result):
            await result


def _write_pcm16_wav(path: Path, samples_f32: np.ndarray) -> None:
    pcm16 = np.rint(np.clip(samples_f32, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(GPU_SAMPLE_RATE_HZ)
        wav_file.writeframes(pcm16.tobytes())


def _failure_code(exception: BaseException) -> str:
    if isinstance(exception, GpuWorkerRequestError):
        return exception.code
    if isinstance(exception, GpuWorkerClosedError):
        return exception.code
    return type(exception).__name__


def _failure_diagnostic_fields(exception: BaseException) -> dict[str, object]:
    fields: dict[str, object] = {"failure": _failure_code(exception)}
    if isinstance(exception, GpuWorkerClosedError):
        if exception.exit_code is not None:
            fields["exit_code"] = exception.exit_code
        if exception.failure_type is not None:
            fields["failure_type"] = exception.failure_type
    return fields


__all__ = [
    "GPU_DISCOVERY_PENDING_SECONDS",
    "GPU_PENDING_TTL_SECONDS",
    "GPU_SAMPLE_RATE_HZ",
    "GpuASRChannel",
    "GpuASRDecodeDropped",
    "GpuASRDiagnostic",
    "GpuASRManualRetryRequired",
    "GpuASRRuntimeError",
    "GpuASRRuntimeState",
    "GpuASRWorkDiscarded",
    "GpuASRWorkExpired",
    "GpuDiscoveryState",
    "SharedGpuASRRuntime",
]
