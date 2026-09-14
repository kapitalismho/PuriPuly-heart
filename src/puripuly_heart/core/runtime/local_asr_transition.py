from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal

from puripuly_heart.core.lifecycle import LifecycleScope, start_lifecycle_task

LocalASRTransitionChannel = Literal["self", "peer"]
LocalASRTransitionStatus = Literal["applied", "superseded", "failed", "closed"]


@dataclass(frozen=True, slots=True)
class LocalASRSessionOptions:
    source_language: str
    source_mode: str = "manual"
    language_hint: str | None = None


@dataclass(frozen=True, slots=True)
class LocalASRTransitionRequest:
    channel: LocalASRTransitionChannel
    requested_provider: str
    actual_provider: str
    model_id: str | None
    session_options: LocalASRSessionOptions
    trigger: str


@dataclass(frozen=True, slots=True)
class PreparedLocalASRTransition:
    request: LocalASRTransitionRequest
    provider: object
    generation: int
    validation_ms: int = 0
    load_ms: int = 0


@dataclass(frozen=True, slots=True)
class LocalASRTransitionOutcome:
    status: LocalASRTransitionStatus
    generation: int
    request: LocalASRTransitionRequest
    prepared: PreparedLocalASRTransition | None = None


PrepareLocalASRTransition = Callable[
    [LocalASRTransitionRequest, int], Awaitable[PreparedLocalASRTransition]
]
CommitLocalASRTransition = Callable[[PreparedLocalASRTransition], Awaitable[None]]
LocalASRTransitionDiagnosticSink = Callable[[dict[str, object]], None]


@dataclass(slots=True)
class _QueuedTransition:
    request: LocalASRTransitionRequest
    generation: int
    prepare: PrepareLocalASRTransition
    commit: CommitLocalASRTransition
    submitted_at: float
    future: asyncio.Future[LocalASRTransitionOutcome]


@dataclass(slots=True)
class LocalASRTransitionCoordinator:
    channel: LocalASRTransitionChannel
    stabilization_s: float = 0.15
    diagnostic_sink: LocalASRTransitionDiagnosticSink | None = None
    clock: Callable[[], float] = time.monotonic
    _generation: int = field(init=False, default=0)
    _active_generation: int = field(init=False, default=0)
    _phase: str = field(init=False, default="idle")
    _pending: _QueuedTransition | None = field(init=False, default=None, repr=False)
    _worker_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _prepared_candidates: dict[int, PreparedLocalASRTransition] = field(
        init=False,
        default_factory=dict,
        repr=False,
    )
    _discard_tasks: set[asyncio.Task[None]] = field(
        init=False,
        default_factory=set,
        repr=False,
    )
    _wake_event: asyncio.Event = field(init=False, repr=False)
    _scope: LifecycleScope = field(init=False, repr=False)
    _discard_sequence: int = field(init=False, default=0)
    _closed: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        if self.stabilization_s < 0:
            raise ValueError("stabilization_s must be >= 0")
        self._wake_event = asyncio.Event()
        self._scope = LifecycleScope(f"local-asr-transition:{self.channel}")

    async def request_transition(
        self,
        request: LocalASRTransitionRequest,
        *,
        prepare: PrepareLocalASRTransition,
        commit: CommitLocalASRTransition,
    ) -> LocalASRTransitionOutcome:
        if request.channel != self.channel:
            raise ValueError("request channel does not match coordinator channel")
        self._generation += 1
        generation = self._generation
        loop = asyncio.get_running_loop()
        future: asyncio.Future[LocalASRTransitionOutcome] = loop.create_future()
        queued = _QueuedTransition(
            request=request,
            generation=generation,
            prepare=prepare,
            commit=commit,
            submitted_at=self.clock(),
            future=future,
        )
        if self._closed:
            future.set_result(self._outcome(queued, "closed"))
            return await future
        replaced = self._pending
        self._pending = queued
        if replaced is not None:
            self._resolve(replaced, "superseded")
            self._emit(replaced, "superseded")
        self._wake_event.set()
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = start_lifecycle_task(
                self._scope,
                self._run(),
                name=f"worker-{generation}",
            )
        return await asyncio.shield(future)

    def lifecycle_snapshot(self) -> dict[str, object]:
        return {
            "owner": f"LocalASRTransitionCoordinator:{self.channel}",
            "channel": self.channel,
            "phase": self._phase,
            "active_generation": self._active_generation,
            "preparation_task_present": self._worker_task is not None
            and not self._worker_task.done(),
            "temporary_candidate_count": len(self._prepared_candidates),
        }

    async def cancel_current(self) -> None:
        self._generation += 1
        pending = self._pending
        self._pending = None
        if pending is not None:
            self._resolve(pending, "superseded")
            self._emit(pending, "superseded")
        worker = self._worker_task
        if worker is not None and worker is not asyncio.current_task():
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
        self._worker_task = None
        if not self._closed:
            self._phase = "idle"

    async def close(self) -> None:
        if self._closed and self._worker_task is None and not self._prepared_candidates:
            return
        self._closed = True
        pending = self._pending
        self._pending = None
        if pending is not None:
            self._resolve(pending, "closed")
        self._wake_event.set()
        worker = self._worker_task
        if worker is not None and worker is not asyncio.current_task():
            if self._phase == "waiting_boundary" and not worker.done():
                worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
        self._worker_task = None
        candidates = tuple(self._prepared_candidates.values())
        self._prepared_candidates.clear()
        for prepared in candidates:
            self._schedule_discard(prepared.provider)
        if self._discard_tasks:
            await asyncio.gather(*tuple(self._discard_tasks), return_exceptions=True)
        await self._scope.close()
        self._phase = "closed"

    async def _run(self) -> None:
        try:
            while True:
                queued = self._pending
                if queued is None:
                    if self._closed:
                        return
                    self._phase = "idle"
                    self._wake_event.clear()
                    await self._wake_event.wait()
                    continue
                self._phase = "stabilizing"
                remaining = self.stabilization_s - (self.clock() - queued.submitted_at)
                if remaining > 0:
                    self._wake_event.clear()
                    try:
                        await asyncio.wait_for(self._wake_event.wait(), timeout=remaining)
                    except asyncio.TimeoutError:
                        pass
                    if self._pending is not queued:
                        continue
                if self._pending is not queued:
                    continue
                self._pending = None
                self._phase = "preparing"
                prepared: PreparedLocalASRTransition | None = None
                prepare_started_at = self.clock()
                try:
                    prepared = await queued.prepare(queued.request, queued.generation)
                    if prepared.generation != queued.generation:
                        raise ValueError("prepared transition generation does not match request")
                    if prepared.request != queued.request:
                        raise ValueError("prepared transition request does not match request")
                    self._prepared_candidates[queued.generation] = prepared
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self._resolve(queued, "failed")
                    self._emit(
                        queued,
                        "failed",
                        exception=exc,
                        load_ms=max(
                            0,
                            int(round((self.clock() - prepare_started_at) * 1000)),
                        ),
                    )
                    continue
                if (
                    self._closed
                    or self._pending is not None
                    or queued.generation != self._generation
                ):
                    self._prepared_candidates.pop(queued.generation, None)
                    self._schedule_discard(prepared.provider)
                    status: LocalASRTransitionStatus = "closed" if self._closed else "superseded"
                    self._resolve(queued, status, prepared)
                    self._emit(queued, status, prepared)
                    continue
                self._phase = "waiting_boundary"
                try:
                    await queued.commit(prepared)
                except asyncio.CancelledError:
                    self._prepared_candidates.pop(queued.generation, None)
                    self._schedule_discard(prepared.provider)
                    self._resolve(queued, "closed", prepared)
                    self._emit(queued, "closed", prepared)
                    if self._closed:
                        return
                    raise
                except Exception as exc:
                    self._prepared_candidates.pop(queued.generation, None)
                    self._schedule_discard(prepared.provider)
                    self._resolve(queued, "failed", prepared)
                    self._emit(queued, "failed", prepared, exception=exc)
                    continue
                self._prepared_candidates.pop(queued.generation, None)
                self._active_generation = queued.generation
                self._phase = "idle"
                self._resolve(queued, "applied", prepared)
                self._emit(queued, "applied", prepared)
        finally:
            if self._worker_task is asyncio.current_task():
                self._worker_task = None

    def _outcome(
        self,
        queued: _QueuedTransition,
        status: LocalASRTransitionStatus,
        prepared: PreparedLocalASRTransition | None = None,
    ) -> LocalASRTransitionOutcome:
        return LocalASRTransitionOutcome(
            status=status,
            generation=queued.generation,
            request=queued.request,
            prepared=prepared,
        )

    def _resolve(
        self,
        queued: _QueuedTransition,
        status: LocalASRTransitionStatus,
        prepared: PreparedLocalASRTransition | None = None,
    ) -> None:
        if not queued.future.done():
            queued.future.set_result(self._outcome(queued, status, prepared))

    def _emit(
        self,
        queued: _QueuedTransition,
        status: LocalASRTransitionStatus,
        prepared: PreparedLocalASRTransition | None = None,
        *,
        exception: Exception | None = None,
        load_ms: int | None = None,
    ) -> None:
        if self.diagnostic_sink is None:
            return
        validation_ms = prepared.validation_ms if prepared is not None else 0
        resolved_load_ms = prepared.load_ms if prepared is not None else (load_ms or 0)
        fields = {
            "channel": queued.request.channel,
            "requested_provider": queued.request.requested_provider,
            "actual_provider": queued.request.actual_provider,
            "model_id": queued.request.model_id,
            "generation": queued.generation,
            "trigger": queued.request.trigger,
            "validation_ms": validation_ms,
            "load_ms": resolved_load_ms,
            "boundary_wait_ms": max(
                0,
                int(round((self.clock() - queued.submitted_at) * 1000))
                - validation_ms
                - resolved_load_ms,
            ),
            "switch_ms": 0,
            "outcome": status,
        }
        if exception is not None:
            fields["failure_type"] = type(exception).__name__
        try:
            self.diagnostic_sink(fields)
        except Exception:
            pass

    def _schedule_discard(self, provider: object) -> None:
        self._discard_sequence += 1
        task = start_lifecycle_task(
            self._scope,
            self._close_provider(provider),
            name=f"discard-{self._discard_sequence}",
        )
        self._discard_tasks.add(task)
        task.add_done_callback(self._discard_tasks.discard)

    @staticmethod
    async def _close_provider(provider: object) -> None:
        close_backend = getattr(provider, "close_backend", None)
        close = close_backend if callable(close_backend) else getattr(provider, "close", None)
        if not callable(close):
            return
        result = close()
        if inspect.isawaitable(result):
            await result


__all__ = [
    "LocalASRSessionOptions",
    "LocalASRTransitionCoordinator",
    "LocalASRTransitionOutcome",
    "LocalASRTransitionRequest",
    "PreparedLocalASRTransition",
]
