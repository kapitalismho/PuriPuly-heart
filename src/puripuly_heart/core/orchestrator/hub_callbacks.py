from __future__ import annotations

import asyncio
from collections.abc import Callable
from uuid import UUID

from puripuly_heart.core.orchestrator.hub import ClientHub
from puripuly_heart.core.orchestrator.translation_turn import (
    TranslationOutputSubmission,
    TranslationTurnChild,
    TranslationTurnOutcome,
    TranslationTurnProcessResult,
)
from puripuly_heart.core.runtime.stt_session_projection import SttSessionStateProjection


class ClientHubDurableOwnerCallbacks:
    __slots__ = ("_hub", "_stt_sessions")

    def __init__(self, stt_sessions: SttSessionStateProjection) -> None:
        self._hub: ClientHub | None = None
        self._stt_sessions = stt_sessions

    def bind(self, hub: ClientHub) -> None:
        if self._hub is not None and self._hub is not hub:
            raise RuntimeError("durable owner callbacks are already bound")
        self._hub = hub

    async def self_event_handler(self, event: object) -> None:
        self._stt_sessions.record(event)
        await self._require()._handle_stt_event(event)

    async def peer_event_handler(self, event: object) -> None:
        self._stt_sessions.record(event)
        await self._require()._handle_stt_event(event)

    async def retired_event_handler(self, event: object) -> None:
        self._stt_sessions.record(event)
        await self._require()._handle_retired_stt_event(event)

    async def self_exception_handler(self, exc: Exception) -> None:
        await self._require()._handle_stt_event_loop_exception(exc, channel="self")

    async def peer_exception_handler(self, exc: Exception) -> None:
        await self._require()._handle_stt_event_loop_exception(exc, channel="peer")

    async def child_created(self, child: TranslationTurnChild) -> None:
        await self._require()._on_peer_final_run_child_created(child)

    async def child_started(
        self,
        child: TranslationTurnChild,
        task: asyncio.Task[TranslationTurnProcessResult],
    ) -> None:
        await self._require()._on_peer_final_run_child_started(child, task)

    async def process_child(
        self,
        child: TranslationTurnChild,
        cancellation_requested: Callable[[], bool],
    ) -> TranslationTurnProcessResult:
        return await self._require()._process_peer_final_run_child(
            child,
            cancellation_requested,
        )

    async def child_terminal(
        self,
        child: TranslationTurnChild,
        outcome: TranslationTurnOutcome,
    ) -> None:
        await self._require()._on_peer_final_run_child_terminal(child, outcome)

    async def parent_closed(self, parent_utterance_id: UUID) -> None:
        await self._require()._on_peer_final_run_parent_closed(parent_utterance_id)

    async def parent_rejected(self, parent_utterance_id: UUID) -> None:
        await self._require()._on_peer_final_run_parent_rejected(parent_utterance_id)

    async def submit_translation_output(self, submission: TranslationOutputSubmission) -> None:
        await self._require().submit_translation_output(submission)

    def _require(self) -> ClientHub:
        if self._hub is None:
            raise RuntimeError("durable owner callbacks are not bound")
        return self._hub


__all__ = ["ClientHubDurableOwnerCallbacks"]
