from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

DEFAULT_DIALOGUE_CONCURRENCY = 6
DEFAULT_JUDGE_CONCURRENCY = 3


@dataclass(frozen=True)
class AttemptJob:
    identity: str
    group: str
    execute: Callable[[Any], Awaitable[dict]]


def _raise_failures(failures: list[BaseException]) -> None:
    if len(failures) == 1:
        raise failures[0]
    if failures:
        raise BaseExceptionGroup("Attempt execution failed", failures)


class _AttemptModels:
    def __init__(self, models, judge_slots, release_dialogue, started: float) -> None:
        self._models = models
        self._judge_slots = judge_slots
        self._release_dialogue = release_dialogue
        self._started = started
        self._judgment_started: float | None = None
        self._judgment_finished: float | None = None
        self._calls: list[dict] = []

    @property
    def attempts(self) -> list[dict]:
        return deepcopy(getattr(self._models, "attempts", []))

    async def prepare(self) -> None:
        await self._models.prepare()

    async def complete(self, *, role: str, instructions: str, payload: dict, schema: dict) -> dict:
        requested = perf_counter() - self._started
        if role != "judge" and self._judgment_started is not None:
            raise RuntimeError("Dialogue completion requested after judgment began")
        if role == "judge" and self._judgment_started is None:
            self._judgment_started = requested
            self._release_dialogue()
        call = {
            "role": role,
            "requested_seconds": requested,
            "started_seconds": None,
            "finished_seconds": None,
            "queue_wait_seconds": 0.0,
            "completion_seconds": None,
            "status": "queued",
        }
        self._calls.append(call)
        acquired = False
        try:
            if role == "judge":
                await self._judge_slots.acquire()
                acquired = True
            call["started_seconds"] = perf_counter() - self._started
            if role == "judge":
                call["queue_wait_seconds"] = call["started_seconds"] - requested
            call["status"] = "running"
            response = await self._models.complete(
                role=role, instructions=instructions, payload=payload, schema=schema
            )
            call["status"] = "completed"
            return response
        except BaseException as error:
            call["status"] = "cancelled" if isinstance(error, asyncio.CancelledError) else "failed"
            call["error_type"] = type(error).__name__
            raise
        finally:
            finished = perf_counter() - self._started
            call["finished_seconds"] = finished
            if call["started_seconds"] is not None:
                call["completion_seconds"] = finished - call["started_seconds"]
            else:
                call["queue_wait_seconds"] = finished - requested
            if role == "judge":
                self._judgment_finished = finished
            if acquired:
                self._judge_slots.release()

    def timing(self, *, started_at: str, execute_finished: float, total: float) -> dict:
        dialogue_finished = (
            execute_finished if self._judgment_started is None else self._judgment_started
        )
        return {
            "clock": "monotonic",
            "basis": "Local orchestration wall time; not native provider or first-token timing",
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "dialogue_started_seconds": 0.0,
            "dialogue_finished_seconds": dialogue_finished,
            "dialogue_seconds": dialogue_finished,
            "judgment_started_seconds": self._judgment_started,
            "judgment_finished_seconds": self._judgment_finished,
            "judgment_seconds": (
                self._judgment_finished - self._judgment_started
                if self._judgment_finished is not None and self._judgment_started is not None
                else 0.0
            ),
            "execute_finished_seconds": execute_finished,
            "total_seconds": total,
            "calls": deepcopy(self._calls),
        }


class AttemptScheduler:
    def __init__(
        self,
        model_factory: Callable[[AttemptJob], Any],
        *,
        dialogue_concurrency: int = DEFAULT_DIALOGUE_CONCURRENCY,
        judge_concurrency: int = DEFAULT_JUDGE_CONCURRENCY,
        group_concurrency: dict[str, int] | None = None,
    ) -> None:
        limits = (dialogue_concurrency, judge_concurrency, *(group_concurrency or {}).values())
        if any(type(limit) is not int or limit < 1 for limit in limits):
            raise ValueError("Concurrency limits must be positive integers")
        self._model_factory = model_factory
        self._dialogue_concurrency = dialogue_concurrency
        self._judge_concurrency = judge_concurrency
        self._group_concurrency = dict(group_concurrency or {})
        self._running = False

    async def run(
        self,
        jobs: Iterable[AttemptJob],
        *,
        checkpoint: Callable[[AttemptJob, dict], Awaitable[None]],
        stop_requested: Callable[[], bool] | None = None,
    ) -> list[dict]:
        if self._running:
            raise RuntimeError("AttemptScheduler is already running")
        pending: dict[str, deque[tuple[int, AttemptJob]]] = {}
        identities: set[str] = set()
        for index, job in enumerate(jobs):
            if job.identity in identities:
                raise ValueError(f"Duplicate attempt identity: {job.identity}")
            identities.add(job.identity)
            pending.setdefault(job.group, deque()).append((index, job))
        groups = deque(pending)
        group_active: dict[str, int] = dict.fromkeys(pending, 0)
        dialogue_active = 0
        active: dict[asyncio.Task, int] = {}
        results: dict[int, dict] = {}
        failures: list[BaseException] = []
        changed = asyncio.Event()
        judge_slots = asyncio.Semaphore(self._judge_concurrency)
        checkpoint_lock = asyncio.Lock()
        stopped = False
        self._running = True

        def should_stop() -> bool:
            nonlocal stopped
            if not stopped and stop_requested is not None and stop_requested():
                stopped = True
            return stopped

        def next_job() -> tuple[int, AttemptJob] | None:
            for _ in range(len(groups)):
                group = groups.popleft()
                queue = pending[group]
                if not queue:
                    continue
                groups.append(group)
                limit = self._group_concurrency.get(group, self._dialogue_concurrency)
                if group_active[group] < limit:
                    return queue.popleft()
            return None

        async def execute(job: AttemptJob) -> dict | None:
            nonlocal dialogue_active, stopped
            released = False
            models = None
            wrapped = None
            result = None
            errors: list[BaseException] = []
            started = perf_counter()
            started_at = datetime.now(timezone.utc).isoformat()

            def release_dialogue() -> None:
                nonlocal dialogue_active, released
                if not released:
                    released = True
                    dialogue_active -= 1
                    group_active[job.group] -= 1
                    changed.set()

            try:
                if not should_stop():
                    models = self._model_factory(job)
                    if not should_stop():
                        wrapped = _AttemptModels(models, judge_slots, release_dialogue, started)
                        completed = await job.execute(wrapped)
                        if not isinstance(completed, dict):
                            raise TypeError("Attempt execute must return a result dictionary")
                        result = completed
            except BaseException as error:
                stopped = True
                errors.append(error)
            finally:
                execute_finished = perf_counter() - started
                release_dialogue()
                close = getattr(models, "close", None)
                if close is not None:
                    try:
                        await close()
                    except BaseException as error:
                        stopped = True
                        errors.append(error)
                if result is not None:
                    result["execution_timing"] = wrapped.timing(
                        started_at=started_at,
                        execute_finished=execute_finished,
                        total=perf_counter() - started,
                    )
                    try:
                        async with checkpoint_lock:
                            await checkpoint(job, result)
                    except BaseException as error:
                        stopped = True
                        errors.append(error)
            _raise_failures(errors)
            return result

        def collect_finished() -> None:
            for task, index in tuple(active.items()):
                if not task.done():
                    continue
                del active[task]
                try:
                    result = task.result()
                except BaseException as error:
                    failures.append(error)
                else:
                    if result is not None:
                        results[index] = result

        try:
            while groups or active:
                changed.clear()
                collect_finished()
                if not failures and not should_stop():
                    while (
                        dialogue_active < self._dialogue_concurrency
                        and len(active) < self._dialogue_concurrency + self._judge_concurrency
                        and not should_stop()
                    ):
                        candidate = next_job()
                        if candidate is None:
                            break
                        index, job = candidate
                        dialogue_active += 1
                        group_active[job.group] += 1
                        task = asyncio.create_task(execute(job), name=f"attempt:{job.identity}")
                        active[task] = index
                        task.add_done_callback(lambda _: changed.set())
                if not active:
                    break
                await changed.wait()
        except BaseException as error:
            stopped = True
            failures.append(error)
        finally:
            if active:
                drain = asyncio.gather(*active, return_exceptions=True)
                while not drain.done():
                    try:
                        await asyncio.shield(drain)
                    except asyncio.CancelledError as error:
                        failures.append(error)
                collect_finished()
            self._running = False
        _raise_failures(failures)
        return [results[index] for index in sorted(results)]
