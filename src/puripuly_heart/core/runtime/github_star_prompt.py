from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from typing import Any

GithubStarPromptDiagnosticsSink = Callable[[str, Mapping[str, object]], None]
GithubStarPromptRuntimeStateChanged = Callable[["GithubStarPromptRuntime"], None]
GithubStarPromptRunner = Callable[[int], Awaitable[bool]]


def _raise_cleanup_failures(message: str, failures: list[Exception]) -> None:
    if not failures:
        return
    if len(failures) == 1:
        raise failures[0]
    raise ExceptionGroup(message, failures)


class GithubStarPromptRuntime:
    """Owns GitHub-star prompt timer and persistence background tasks."""

    resource_fields = (
        "_launch_prompt_task",
        "_translation_success_task",
        "_generation",
    )
    stop_ingress_policy = "cancel scheduled prompt"
    shutdown_policy = "cancel/gather timer task and owned prompt persistence tasks"
    late_callback_rule = "late prompt callback checks current prompt state"

    def __init__(
        self,
        *,
        cancel_timeout_s: float = 2.0,
        diagnostics_sink: GithubStarPromptDiagnosticsSink | None = None,
        state_changed: GithubStarPromptRuntimeStateChanged | None = None,
    ) -> None:
        self._cancel_timeout_s = max(0.0, float(cancel_timeout_s))
        self._diagnostics_sink = diagnostics_sink
        self._state_changed = state_changed
        self._launch_prompt_task: asyncio.Task[bool] | None = None
        self._translation_success_task: asyncio.Task[bool] | None = None
        self._generation = 0
        self._closing = False
        self._closed = False
        self._close_lock = asyncio.Lock()

    @property
    def owner_name(self) -> str:
        return "GithubStarPromptRuntime"

    @property
    def launch_prompt_task(self) -> asyncio.Task[bool] | None:
        return self._launch_prompt_task

    @property
    def translation_success_task(self) -> asyncio.Task[bool] | None:
        return self._translation_success_task

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def is_closing(self) -> bool:
        return self._closing

    def lifecycle_owner_snapshot(self) -> dict[str, object]:
        return {
            "owner": self.owner_name,
            "resource_fields": self.resource_fields,
            "stop_ingress": self.stop_ingress_policy,
            "shutdown_policy": self.shutdown_policy,
            "late_callback_rule": self.late_callback_rule,
        }

    def is_current_generation(self, generation: int) -> bool:
        return not self._closing and not self._closed and generation == self._generation

    def start_launch_prompt(self, run_prompt: GithubStarPromptRunner) -> asyncio.Task[bool]:
        if self._closing or self._closed:
            state = "closing" if self._closing else "closed"
            raise RuntimeError(f"{self.owner_name} is {state} to new launch prompts")
        self._collect_done_task(self._launch_prompt_task)
        if self._launch_prompt_task is not None and not self._launch_prompt_task.done():
            raise RuntimeError(f"{self.owner_name} already owns a launch prompt task")

        self._generation += 1
        generation = self._generation
        task = asyncio.create_task(
            self._run_launch_prompt_guarded(run_prompt, generation),
            name=f"{self.owner_name}:launch-prompt",
        )
        self._launch_prompt_task = task
        task.add_done_callback(self._on_launch_prompt_done)
        self._notify_state_changed()
        return task

    def start_translation_success_observation(
        self,
        coroutine: Coroutine[Any, Any, bool],
    ) -> asyncio.Task[bool]:
        if self._closing or self._closed:
            coroutine.close()
            state = "closing" if self._closing else "closed"
            raise RuntimeError(f"{self.owner_name} is {state} to translation success work")
        self._collect_done_task(self._translation_success_task)
        if self._translation_success_task is not None and not self._translation_success_task.done():
            coroutine.close()
            raise RuntimeError(f"{self.owner_name} already owns a translation success task")

        task = asyncio.create_task(
            coroutine,
            name=f"{self.owner_name}:translation-success",
        )
        self._translation_success_task = task
        task.add_done_callback(self._on_translation_success_done)
        self._notify_state_changed()
        return task

    async def drain_translation_success_task(self) -> None:
        task = self._translation_success_task
        if task is None:
            return
        await asyncio.gather(task, return_exceptions=True)
        if self._translation_success_task is task:
            self._translation_success_task = None
            self._notify_state_changed()

    def stop_ingress(self) -> None:
        if self._closed:
            return
        self._closing = True
        self._closed = True
        self._generation += 1
        for task in (self._launch_prompt_task, self._translation_success_task):
            if task is not None and not task.done():
                task.cancel()
        self._notify_state_changed()

    async def close(self) -> None:
        if (
            self._closed
            and self._launch_prompt_task is None
            and self._translation_success_task is None
        ):
            return
        async with self._close_lock:
            if (
                self._closed
                and self._launch_prompt_task is None
                and self._translation_success_task is None
            ):
                return
            self.stop_ingress()
            try:
                cleanup_failures: list[Exception] = []
                launch_task = self._launch_prompt_task
                translation_task = self._translation_success_task
                cleanup_failures.extend(
                    await self._cancel_task_bounded(
                        launch_task,
                        task_name="launch_prompt",
                    )
                )
                cleanup_failures.extend(
                    await self._cancel_task_bounded(
                        translation_task,
                        task_name="translation_success",
                    )
                )
                self._clear_task_reference_if_done(self._launch_prompt_task)
                self._clear_task_reference_if_done(self._translation_success_task)
                self._notify_state_changed()
                _raise_cleanup_failures(
                    f"{self.owner_name} close cleanup failed",
                    cleanup_failures,
                )
            finally:
                self._closing = False
                self._notify_state_changed()

    async def _run_launch_prompt_guarded(
        self,
        run_prompt: GithubStarPromptRunner,
        generation: int,
    ) -> bool:
        if not self.is_current_generation(generation):
            return False
        try:
            result = await run_prompt(generation)
        except asyncio.CancelledError:
            return False
        except Exception as exc:
            self._emit(
                "github_star_prompt_launch_failed",
                {"error_type": type(exc).__name__},
            )
            raise
        if not self.is_current_generation(generation):
            self._emit("github_star_prompt_late_callback_dropped", {})
            return False
        return bool(result)

    async def _cancel_task_bounded(
        self,
        task: asyncio.Task[Any] | None,
        *,
        task_name: str,
    ) -> list[Exception]:
        if task is None or task is asyncio.current_task():
            return []
        if not task.done() and task.cancelling() == 0:
            task.cancel()
        done, pending = await asyncio.wait({task}, timeout=self._cancel_timeout_s)
        for completed in done:
            self._observe_task_exception(completed, task_name=task_name)
        if pending:
            self._emit(
                "github_star_prompt_task_cancel_timeout",
                {"task_name": task_name},
            )
            return [
                TimeoutError(
                    f"{self.owner_name} timed out cancelling {task_name} task during close"
                )
            ]
        return []

    def _clear_task_reference_if_done(self, task: asyncio.Task[Any] | None) -> None:
        if task is None or not task.done():
            return
        if task is self._launch_prompt_task:
            self._launch_prompt_task = None
        if task is self._translation_success_task:
            self._translation_success_task = None

    def _on_launch_prompt_done(self, task: asyncio.Task[bool]) -> None:
        self._observe_task_exception(task, task_name="launch_prompt")
        if self._launch_prompt_task is task:
            self._launch_prompt_task = None
            self._notify_state_changed()

    def _on_translation_success_done(self, task: asyncio.Task[bool]) -> None:
        self._observe_task_exception(task, task_name="translation_success")
        if self._translation_success_task is task:
            self._translation_success_task = None
            self._notify_state_changed()

    def _collect_done_task(self, task: asyncio.Task[Any] | None) -> None:
        if task is None or not task.done():
            return
        if task is self._launch_prompt_task:
            self._launch_prompt_task = None
            self._observe_task_exception(task, task_name="launch_prompt")
        if task is self._translation_success_task:
            self._translation_success_task = None
            self._observe_task_exception(task, task_name="translation_success")

    def _observe_task_exception(self, task: asyncio.Task[Any], *, task_name: str) -> None:
        if task.cancelled():
            return
        try:
            exception = task.exception()
        except asyncio.CancelledError:
            return
        if exception is not None:
            self._emit(
                "github_star_prompt_task_failed",
                {"task_name": task_name, "error_type": type(exception).__name__},
            )

    def _emit(self, event: str, metadata: Mapping[str, object]) -> None:
        if self._diagnostics_sink is None:
            return
        try:
            self._diagnostics_sink(event, metadata)
        except Exception:
            return

    def _notify_state_changed(self) -> None:
        if self._state_changed is not None:
            with contextlib.suppress(Exception):
                self._state_changed(self)


__all__ = ["GithubStarPromptRuntime"]
