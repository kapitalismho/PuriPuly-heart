from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from puripuly_heart.app.ports.manual_typing import SelfChatboxTypingPort

MANUAL_INPUT_TYPING_REASON = "manual_input"


@dataclass(slots=True)
class ManualTypingOwner:
    output_provider: Callable[[], SelfChatboxTypingPort | None]
    completion_provider: Callable[[object], object | None]
    log_detailed: Callable[[str], object]
    log_error: Callable[[str], object]
    idle_timeout_seconds: float
    submit_timeout_seconds: float
    _idle_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _submit_generation: int = field(init=False, default=0, repr=False)

    @property
    def idle_task(self) -> asyncio.Task[None] | None:
        return self._idle_task

    @property
    def submit_generation(self) -> int:
        return self._submit_generation

    async def submit(self, submit: Callable[[], Awaitable[object]] | None) -> None:
        self.clear_input()
        if submit is None:
            return
        reason = self.begin_submit()
        try:
            utterance_id = await submit()
            await self._wait_for_completion(utterance_id)
        except Exception as exc:
            self.log_error(f"Submit failed: {exc}")
        finally:
            self.clear_submit(reason)

    def set_input_activity(self, has_text: bool) -> None:
        if has_text:
            self._set_reason(MANUAL_INPUT_TYPING_REASON, True)
            self._reschedule_idle_timeout()
            return
        self.clear_input()

    async def release(self) -> None:
        task = self._cancel_idle_task()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        output = self.output_provider()
        if output is not None:
            output.clear_self_chatbox_typing_reasons()
        self.log_detailed("[ManualTyping] release status=cleared")

    def begin_submit(self) -> str:
        self._submit_generation += 1
        reason = f"manual_submit:{self._submit_generation}"
        self._set_reason(reason, True)
        return reason

    def clear_submit(self, reason: str) -> None:
        self._set_reason(reason, False)

    def clear_input(self) -> None:
        self._cancel_idle_task()
        self._set_reason(MANUAL_INPUT_TYPING_REASON, False)

    async def _wait_for_completion(self, utterance_id: object) -> None:
        completion = self.completion_provider(utterance_id)
        if completion is None:
            return
        if isinstance(completion, asyncio.Task):
            awaitable: object = asyncio.gather(completion, return_exceptions=True)
        elif inspect.isawaitable(completion):
            awaitable = completion
        else:
            return
        try:
            await asyncio.wait_for(
                asyncio.shield(awaitable),
                timeout=self.submit_timeout_seconds,
            )
        except asyncio.TimeoutError:
            self.log_detailed("[ManualTyping] submit output wait timed out")
        except Exception as exc:
            self.log_error(f"Manual submit output wait failed: {exc}")

    def _set_reason(self, reason: str, active: bool) -> None:
        output = self.output_provider()
        if output is None:
            return
        try:
            output.set_self_chatbox_typing_reason(reason, active)
        except Exception as exc:
            self.log_error(f"Manual typing output update failed: {exc}")

    def _reschedule_idle_timeout(self) -> None:
        self._cancel_idle_task()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self.log_error("Manual typing idle timeout scheduling failed: no running loop")
            return
        self._idle_task = loop.create_task(self._idle_timeout())

    def _cancel_idle_task(self) -> asyncio.Task[None] | None:
        task = self._idle_task
        self._idle_task = None
        if task is not None and not task.done():
            task.cancel()
        return task

    async def _idle_timeout(self) -> None:
        try:
            await asyncio.sleep(self.idle_timeout_seconds)
            self._set_reason(MANUAL_INPUT_TYPING_REASON, False)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.log_error(f"Manual typing idle timeout failed: {exc}")
        finally:
            if self._idle_task is asyncio.current_task():
                self._idle_task = None
