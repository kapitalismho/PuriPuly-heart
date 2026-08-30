from __future__ import annotations

import asyncio
from collections.abc import Callable
from uuid import UUID

from puripuly_heart.core.orchestrator.peer_translation_channel import (
    PeerTranslationChannelOwner,
)
from puripuly_heart.core.orchestrator.self_translation_channel import (
    SelfTranslationChannelOwner,
)
from puripuly_heart.core.orchestrator.translation_turn import (
    TranslationOutputSubmission,
    TranslationTurnChild,
    TranslationTurnOutcome,
    TranslationTurnProcessResult,
)
from puripuly_heart.core.runtime.stt_session_projection import SttSessionStateProjection


class TranslationChannelOwnerCallbacks:
    __slots__ = ("_peer", "_self", "_stt_sessions")

    def __init__(self, stt_sessions: SttSessionStateProjection) -> None:
        self._self: SelfTranslationChannelOwner | None = None
        self._peer: PeerTranslationChannelOwner | None = None
        self._stt_sessions = stt_sessions

    def bind_self(self, owner: SelfTranslationChannelOwner) -> None:
        if self._self is not None and self._self is not owner:
            raise RuntimeError("Self durable owner callbacks are already bound")
        self._self = owner

    def bind_peer(self, owner: PeerTranslationChannelOwner) -> None:
        if self._peer is not None and self._peer is not owner:
            raise RuntimeError("Peer durable owner callbacks are already bound")
        self._peer = owner

    async def self_event_handler(self, event: object) -> None:
        self._stt_sessions.record(event)
        await self._require_self().handle_stt_event(event)

    async def peer_event_handler(self, event: object) -> None:
        self._stt_sessions.record(event)
        await self._require_peer().handle_stt_event(event)

    async def retired_event_handler(self, event: object) -> None:
        self._stt_sessions.record(event)
        if getattr(event, "channel", None) == "self":
            await self._require_self().handle_retired_stt_event(event)
            return
        await self._require_peer().handle_retired_stt_event(event)

    async def self_exception_handler(self, exc: Exception) -> None:
        await self._require_self().handle_stt_event_loop_exception(exc)

    async def peer_exception_handler(self, exc: Exception) -> None:
        await self._require_peer().handle_stt_event_loop_exception(exc, channel="peer")

    async def child_created(self, child: TranslationTurnChild) -> None:
        if child.channel == "self":
            await self._require_self().on_child_created(child)
            return
        await self._require_peer().on_child_created(child)

    async def child_started(
        self,
        child: TranslationTurnChild,
        task: asyncio.Task[TranslationTurnProcessResult],
    ) -> None:
        if child.channel == "self":
            await self._require_self().on_child_started(child, task)
            return
        await self._require_peer().on_child_started(child, task)

    async def parent_admitted(self, children: tuple[TranslationTurnChild, ...]) -> None:
        if children and children[0].channel == "self":
            await self._require_self().on_parent_admitted(children)

    async def process_child(
        self,
        child: TranslationTurnChild,
        cancellation_requested: Callable[[], bool],
    ) -> TranslationTurnProcessResult:
        if child.channel == "self":
            return await self._require_self().process_child(
                child,
                cancellation_requested,
            )
        return await self._require_peer().process_child(
            child,
            cancellation_requested,
        )

    async def child_terminal(
        self,
        child: TranslationTurnChild,
        outcome: TranslationTurnOutcome,
    ) -> None:
        if child.channel == "self":
            await self._require_self().on_child_terminal(child, outcome)
            return
        await self._require_peer().on_child_terminal(child, outcome)

    async def parent_closed(self, parent_utterance_id: UUID) -> None:
        await self._require_peer().on_parent_closed(parent_utterance_id)

    async def parent_rejected(self, parent_utterance_id: UUID) -> None:
        await self._require_peer().on_parent_rejected(parent_utterance_id)

    async def submit_translation_output(self, submission: TranslationOutputSubmission) -> None:
        if submission.channel == "self":
            await self._require_self().submit_translation_output(submission)
            return
        await self._require_peer().submit_translation_output(submission)

    def _require_self(self) -> SelfTranslationChannelOwner:
        if self._self is None:
            raise RuntimeError("Self durable owner callbacks are not bound")
        return self._self

    def _require_peer(self) -> PeerTranslationChannelOwner:
        if self._peer is None:
            raise RuntimeError("Peer durable owner callbacks are not bound")
        return self._peer


__all__ = ["TranslationChannelOwnerCallbacks"]
