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
from puripuly_heart.core.runtime.peer_channel import PeerCaptureSessionOwner
from puripuly_heart.core.runtime.self_capture import SelfCaptureSessionOwner
from puripuly_heart.core.runtime.stt_session_projection import SttSessionStateProjection
from puripuly_heart.core.stt.backend import (
    STTProviderEpochEnded,
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
)
from puripuly_heart.domain.events import STTSessionState, STTSessionStateEvent


class TranslationChannelOwnerCallbacks:
    __slots__ = ("_peer", "_peer_capture", "_self", "_self_capture", "_stt_sessions")

    def __init__(self, stt_sessions: SttSessionStateProjection) -> None:
        self._self: SelfTranslationChannelOwner | None = None
        self._peer: PeerTranslationChannelOwner | None = None
        self._peer_capture: PeerCaptureSessionOwner | None = None
        self._self_capture: SelfCaptureSessionOwner | None = None
        self._stt_sessions = stt_sessions

    def bind_self(self, owner: SelfTranslationChannelOwner) -> None:
        if self._self is not None and self._self is not owner:
            raise RuntimeError("Self durable owner callbacks are already bound")
        self._self = owner

    def bind_peer(self, owner: PeerTranslationChannelOwner) -> None:
        if self._peer is not None and self._peer is not owner:
            raise RuntimeError("Peer durable owner callbacks are already bound")
        self._peer = owner

    def bind_peer_capture(self, owner: PeerCaptureSessionOwner) -> None:
        if self._peer_capture is not None and self._peer_capture is not owner:
            raise RuntimeError("Peer source owner callbacks are already bound")
        self._peer_capture = owner

    def bind_self_capture(self, owner: SelfCaptureSessionOwner) -> None:
        if self._self_capture is not None and self._self_capture is not owner:
            raise RuntimeError("Self source owner callbacks are already bound")
        self._self_capture = owner

    async def self_event_handler(self, event: object) -> None:
        await self._before_self_event(event)
        self._stt_sessions.record(event)
        await self._require_self().handle_stt_event(event)
        await self._after_self_event(event)

    async def _before_self_event(self, event: object) -> None:
        if isinstance(event, STTProviderTurnUpdate) or (
            isinstance(event, STTProviderTurnTerminal)
            and event.outcome in ("final", "empty", "degraded", "suppressed")
        ):
            await self._publish_self_session_state(STTSessionState.STREAMING)
        if isinstance(event, STTProviderTurnTerminal) and self._self_capture is not None:
            self._self_capture.note_recognition_terminal(event)

    async def _after_self_event(self, event: object) -> None:
        if isinstance(event, STTProviderTurnTerminal) and (
            event.outcome in ("failed", "expired", "cancelled") or event.failure_reason is not None
        ):
            await self._publish_self_session_state(STTSessionState.DISCONNECTED)

    async def _publish_self_session_state(self, state: STTSessionState) -> None:
        if self._stt_sessions.state("self") is state:
            return
        event = STTSessionStateEvent(state=state, channel="self")
        self._stt_sessions.record(event)
        await self._require_self().handle_stt_event(event)

    async def peer_event_handler(self, event: object) -> None:
        if isinstance(event, STTProviderTurnUpdate):
            return
        if isinstance(event, STTProviderEpochEnded):
            return
        if isinstance(event, STTProviderTurnTerminal):
            source_owner = self._require_peer_capture()
            admissions = source_owner.admit_provider_terminal(event)
            peer = self._require_peer()
            for receipt, terminal in admissions:
                await peer.handle_provider_turn_terminal(receipt, terminal)
            return
        self._stt_sessions.record(event)
        await self._require_peer().handle_stt_event(event)

    async def retired_event_handler(self, event: object) -> None:
        self._stt_sessions.record(event)
        if getattr(event, "channel", None) == "self" or isinstance(
            event,
            STTProviderTurnUpdate | STTProviderTurnTerminal,
        ):
            await self._before_self_event(event)
            await self._require_self().handle_retired_stt_event(event)
            await self._after_self_event(event)
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
        if not children:
            return
        if children[0].channel == "self":
            await self._require_self().on_parent_admitted(children)
            return
        await self._require_peer().on_parent_admitted(children)

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

    async def submit_translation_output(
        self,
        submission: TranslationOutputSubmission,
    ) -> object | None:
        if submission.channel == "self":
            return await self._require_self().submit_translation_output(submission)
        return await self._require_peer().submit_translation_output(submission)

    def _require_self(self) -> SelfTranslationChannelOwner:
        if self._self is None:
            raise RuntimeError("Self durable owner callbacks are not bound")
        return self._self

    def _require_peer(self) -> PeerTranslationChannelOwner:
        if self._peer is None:
            raise RuntimeError("Peer durable owner callbacks are not bound")
        return self._peer

    def _require_peer_capture(self) -> PeerCaptureSessionOwner:
        if self._peer_capture is None:
            raise RuntimeError("Peer source owner callbacks are not bound")
        return self._peer_capture


__all__ = ["TranslationChannelOwnerCallbacks"]
