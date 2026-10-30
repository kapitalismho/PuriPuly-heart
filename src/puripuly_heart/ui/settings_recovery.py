from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace

from puripuly_heart.app.ports.settings_view import (
    AudioSettingsIntent,
    CustomVocabularySettingsIntent,
    PromptApplyIntent,
    ProviderApplyIntent,
    SettingsApplyOutcome,
    SystemPromptEdit,
)


@dataclass
class _PendingChange:
    intent: object
    attempts: int = 0
    result: SettingsApplyOutcome | None = None
    delay: float | None = None


class SettingsRecovery:
    def __init__(
        self,
        apply: Callable[[object], Awaitable[SettingsApplyOutcome]],
        committed: Callable[[object], None],
        changed: Callable[[str], None],
        *,
        retry_delays: tuple[float, ...] = (1.0, 3.0),
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._apply = apply
        self._committed = committed
        self._changed = changed
        self._retry_delays = retry_delays
        self._sleep = sleep
        self._pending: dict[object, _PendingChange] = {}
        self._closed = False
        self._running = False
        self.phase = "idle"

    @property
    def has_pending(self) -> bool:
        return bool(self._pending)

    def submit(self, intent: object) -> None:
        if self._closed:
            return
        key: object = type(intent) if intent is not None else None
        if isinstance(intent, CustomVocabularySettingsIntent):
            key = (type(intent), intent.source_language)
        if isinstance(intent, PromptApplyIntent):
            provider = self._pending.get(ProviderApplyIntent)
            if provider is not None:
                self.submit(ProviderApplyIntent((SystemPromptEdit(intent.value),)))
                return
        if isinstance(intent, ProviderApplyIntent):
            edits = {}
            previous = self._pending.get(key)
            if previous is not None:
                edits.update((type(edit), edit) for edit in previous.intent.edits)
            prompt = self._pending.pop(PromptApplyIntent, None)
            if prompt is not None:
                edits[SystemPromptEdit] = SystemPromptEdit(prompt.intent.value)
            edits.update((type(edit), edit) for edit in intent.edits)
            intent = ProviderApplyIntent(tuple(edits.values()))
        elif isinstance(intent, AudioSettingsIntent):
            previous = self._pending.get(key)
            changes = (
                {type(change): change for change in previous.intent.changes}
                if previous is not None
                else {}
            )
            changes.update((type(change), change) for change in intent.changes)
            intent = AudioSettingsIntent(tuple(changes.values()))
        self._pending[key] = _PendingChange(intent)

    def retry(self) -> None:
        if self._closed:
            return
        for key, pending in tuple(self._pending.items()):
            self._pending[key] = replace(pending, attempts=0, delay=None)

    def stop(self) -> None:
        self._closed = True
        self._pending.clear()

    async def drain(self) -> None:
        if self._running or self._closed:
            return
        self._running = True
        try:
            while not self._closed:
                candidates = [
                    (key, pending)
                    for key, pending in self._pending.items()
                    if pending.attempts <= len(self._retry_delays)
                ]
                if not candidates:
                    self._set_phase(self._failure_phase())
                    return
                key, pending = min(candidates, key=lambda item: item[1].delay is not None)
                if pending.delay is not None:
                    self._set_phase("retrying_apply" if pending.intent is None else "retrying_save")
                    delay = pending.delay
                    pending.delay = None
                    await self._sleep(delay)
                    continue
                self._set_phase("applying" if pending.intent is None else "saving")
                result = await self._apply(pending.intent)
                if self._closed:
                    return
                pending.attempts += 1
                pending.result = result
                if result.saved and pending.intent is not None:
                    self._committed(pending.intent)
                if self._pending.get(key) is not pending:
                    if result.saved and not result.applied:
                        self._pending.setdefault(None, _PendingChange(None, result=result))
                    continue
                if result.applied or result.saved and pending.intent is not None:
                    del self._pending[key]
                if result.applied:
                    if key is not None and None in self._pending:
                        self._pending[None] = _PendingChange(None)
                    continue
                if result.saved and pending.intent is not None:
                    key = None
                    pending = _PendingChange(None, attempts=pending.attempts, result=result)
                    self._pending[key] = pending
                if result.saved is None:
                    pending.attempts = len(self._retry_delays) + 1
                elif pending.attempts <= len(self._retry_delays):
                    pending.delay = self._retry_delays[pending.attempts - 1]
        finally:
            self._running = False

    def _failure_phase(self) -> str:
        failures = {
            (
                "unknown_failed"
                if pending.result is None or pending.result.saved is None
                else "apply_failed" if pending.result.saved else "save_failed"
            )
            for pending in self._pending.values()
        }
        return (
            next(iter(failures)) if len(failures) == 1 else "mixed_failed" if failures else "idle"
        )

    def _set_phase(self, phase: str) -> None:
        self.phase = phase
        self._changed(phase)
