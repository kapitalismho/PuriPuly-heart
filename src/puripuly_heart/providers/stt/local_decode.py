from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import numpy as np

from puripuly_heart.core.lifecycle import LifecycleScope, start_lifecycle_task

DEFAULT_LOCAL_DECODE_BACKLOG_WARN_SIZE = 8
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LocalDecodeJob:
    sequence: int
    samples_f32: np.ndarray
    audio_ms: float


@dataclass(frozen=True, slots=True)
class LocalDecodeCompletion:
    job: LocalDecodeJob
    text: str
    inference_ms: float


@dataclass(frozen=True, slots=True)
class LocalDecodeBacklog:
    pending_jobs: int
    buffered_audio_ms: float
    warning_threshold: int


@dataclass(frozen=True, slots=True)
class LocalDecodeFailure:
    job: LocalDecodeJob
    error: Exception
    discarded_jobs: tuple[LocalDecodeJob, ...]


@dataclass(slots=True)
class LocalDecodeCoordinator:
    owner_name: str
    sample_rate_hz: int
    decode: Callable[[np.ndarray], Awaitable[str]]
    on_completion: Callable[[LocalDecodeCompletion], Awaitable[None]]
    on_failure: Callable[[LocalDecodeFailure], Awaitable[None]]
    on_backlog_warning: Callable[[LocalDecodeBacklog], object] | None = None
    start_after: asyncio.Event | None = None
    backlog_warn_size: int = DEFAULT_LOCAL_DECODE_BACKLOG_WARN_SIZE
    clock: Callable[[], float] = time.perf_counter
    _queue: deque[LocalDecodeJob] = field(init=False, repr=False)
    _scope: LifecycleScope = field(init=False, repr=False)
    _worker_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _active_job: LocalDecodeJob | None = field(init=False, default=None, repr=False)
    _next_sequence: int = field(init=False, default=1, repr=False)
    _worker_generation: int = field(init=False, default=0, repr=False)
    _queued_audio_ms: float = field(init=False, default=0.0, repr=False)
    _backlog_warning_active: bool = field(init=False, default=False, repr=False)
    _accepting: bool = field(init=False, default=True, repr=False)
    _closed: bool = field(init=False, default=False, repr=False)
    _failed: bool = field(init=False, default=False, repr=False)
    _close_complete: asyncio.Event = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be > 0")
        if self.backlog_warn_size <= 0:
            raise ValueError("backlog_warn_size must be > 0")
        self._queue = deque()
        self._scope = LifecycleScope(f"{self.owner_name}-{id(self):x}")
        self._close_complete = asyncio.Event()

    @property
    def accepting(self) -> bool:
        return self._accepting and not self._closed and not self._failed

    @property
    def pending_jobs(self) -> int:
        return len(self._queue) + (1 if self._active_job is not None else 0)

    @property
    def buffered_audio_ms(self) -> float:
        active_audio_ms = self._active_job.audio_ms if self._active_job is not None else 0.0
        return self._queued_audio_ms + active_audio_ms

    def enqueue(self, samples_f32: np.ndarray) -> bool:
        if not self.accepting:
            return False
        samples = np.asarray(samples_f32, dtype=np.float32).reshape(-1).copy()
        audio_ms = samples.size * 1000.0 / float(self.sample_rate_hz)
        job = LocalDecodeJob(
            sequence=self._next_sequence,
            samples_f32=samples,
            audio_ms=audio_ms,
        )
        self._next_sequence += 1
        self._queue.append(job)
        self._queued_audio_ms += audio_ms
        self._update_backlog_warning()
        self._ensure_worker()
        return True

    async def stop(self) -> None:
        self._accepting = False
        if self._closed:
            await asyncio.shield(self._close_complete.wait())
            return
        worker_task = self._worker_task
        if worker_task is not None:
            try:
                await asyncio.shield(worker_task)
            except asyncio.CancelledError:
                current_task = asyncio.current_task()
                if current_task is not None and current_task.cancelling():
                    raise
                if not self._closed:
                    raise
                await asyncio.shield(self._close_complete.wait())

    async def close(self) -> None:
        if self._closed:
            await asyncio.shield(self._close_complete.wait())
            return
        self._accepting = False
        self._closed = True
        self._queue.clear()
        self._queued_audio_ms = 0.0
        try:
            try:
                await self._scope.close()
            except asyncio.CancelledError:
                await self._scope.close()
                raise
        finally:
            self._active_job = None
            self._backlog_warning_active = False
            self._close_complete.set()

    def _ensure_worker(self) -> None:
        if self._worker_task is not None and not self._worker_task.done():
            return
        self._worker_generation += 1
        self._worker_task = start_lifecycle_task(
            self._scope,
            self._run_worker(),
            name=f"decode-worker-{self._worker_generation}",
        )

    async def _run_worker(self) -> None:
        try:
            if self.start_after is not None:
                await self.start_after.wait()
            while self._queue:
                job = self._queue.popleft()
                self._queued_audio_ms = max(0.0, self._queued_audio_ms - job.audio_ms)
                self._active_job = job
                try:
                    if job.samples_f32.size == 0:
                        text = ""
                        inference_ms = 0.0
                    else:
                        started_at = self.clock()
                        text = await self.decode(job.samples_f32)
                        inference_ms = (self.clock() - started_at) * 1000.0
                    await self.on_completion(
                        LocalDecodeCompletion(
                            job=job,
                            text=str(text),
                            inference_ms=inference_ms,
                        )
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self._failed = True
                    self._accepting = False
                    failure = LocalDecodeFailure(
                        job=job,
                        error=exc,
                        discarded_jobs=tuple(self._queue),
                    )
                    self._queue.clear()
                    self._queued_audio_ms = 0.0
                    await self._notify_failure(failure)
                    return
                finally:
                    self._active_job = None
                    self._update_backlog_warning()
        finally:
            self._active_job = None

    async def _notify_failure(self, failure: LocalDecodeFailure) -> None:
        try:
            await self.on_failure(failure)
        except asyncio.CancelledError:
            raise
        except Exception as callback_exc:
            logger.exception(
                "%s decode failure callback failed: callback_exception=%s",
                self.owner_name,
                type(callback_exc).__name__,
            )

    def _update_backlog_warning(self) -> None:
        pending_jobs = self.pending_jobs
        if pending_jobs <= self.backlog_warn_size:
            self._backlog_warning_active = False
            return
        if self._backlog_warning_active:
            return
        self._backlog_warning_active = True
        if self.on_backlog_warning is None:
            return
        try:
            self.on_backlog_warning(
                LocalDecodeBacklog(
                    pending_jobs=pending_jobs,
                    buffered_audio_ms=self.buffered_audio_ms,
                    warning_threshold=self.backlog_warn_size,
                )
            )
        except Exception as exc:
            logger.exception(
                "%s decode backlog callback failed: callback_exception=%s",
                self.owner_name,
                type(exc).__name__,
            )
