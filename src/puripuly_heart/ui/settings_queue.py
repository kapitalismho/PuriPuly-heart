from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

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
    attempted: bool = False


class SettingsQueue:
    def __init__(
        self,
        apply: Callable[[object], Awaitable[SettingsApplyOutcome]],
        committed: Callable[[object], None],
    ) -> None:
        self._apply = apply
        self._committed = committed
        self._pending: dict[object, _PendingChange] = {}
        self._closed = False
        self._running = False

    @property
    def has_pending(self) -> bool:
        return bool(self._pending)

    def submit(self, intent: object, *, resubmit: bool = True) -> bool:
        if self._closed:
            return False
        key: object = type(intent) if intent is not None else None
        if isinstance(intent, CustomVocabularySettingsIntent):
            key = (type(intent), intent.source_language)
        if isinstance(intent, PromptApplyIntent):
            provider = self._pending.get(ProviderApplyIntent)
            if provider is not None:
                return self.submit(
                    ProviderApplyIntent((SystemPromptEdit(intent.value),)), resubmit=resubmit
                )
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
        previous = self._pending.get(key)
        if not resubmit and previous is not None and previous.intent == intent:
            return False
        self._pending[key] = _PendingChange(intent)
        return True

    def replace_pending_prompt(self, intent: PromptApplyIntent) -> bool:
        if self._closed:
            return False
        prompt = self._pending.get(PromptApplyIntent)
        if prompt is not None:
            previous = prompt.intent.value
        else:
            provider = self._pending.get(ProviderApplyIntent)
            if provider is None:
                return False
            previous = next(
                (
                    edit.value
                    for edit in provider.intent.edits
                    if isinstance(edit, SystemPromptEdit)
                ),
                None,
            )
        if previous is None or previous == intent.value:
            return False
        self.submit(intent)
        return True

    def stop(self) -> None:
        self._closed = True
        self._pending.clear()

    async def drain(self) -> None:
        if self._running or self._closed:
            return
        self._running = True
        try:
            while not self._closed:
                candidate = next(
                    (
                        (key, pending)
                        for key, pending in self._pending.items()
                        if not pending.attempted
                    ),
                    None,
                )
                if candidate is None:
                    return
                key, pending = candidate
                pending.attempted = True
                result = await self._apply(pending.intent)
                if self._closed:
                    return
                if result.saved and pending.intent is not None:
                    self._committed(pending.intent)
                if self._pending.get(key) is pending and (result.applied or result.saved):
                    del self._pending[key]
        finally:
            self._running = False
