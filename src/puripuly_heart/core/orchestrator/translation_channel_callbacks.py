from __future__ import annotations

import asyncio
from collections import OrderedDict
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
    STTProviderInputTerminal,
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
    STTRecognitionUnitTerminal,
    scoped_event_for_delivery,
)
from puripuly_heart.core.stt.recognition_units import RecognitionConsumptionLedger
from puripuly_heart.domain.events import STTSessionState, STTSessionStateEvent
from puripuly_heart.domain.recognition import RecognitionStreamIdentity


class TranslationChannelOwnerCallbacks:
    __slots__ = (
        "_peer",
        "_peer_capture",
        "_self",
        "_self_capture",
        "_stt_sessions",
        "_recognition_consumption",
        "_recognition_locks",
        "_self_ready_stream",
        "_self_last_input_stream",
        "_self_obsolete_streams",
    )

    def __init__(self, stt_sessions: SttSessionStateProjection) -> None:
        self._self: SelfTranslationChannelOwner | None = None
        self._peer: PeerTranslationChannelOwner | None = None
        self._peer_capture: PeerCaptureSessionOwner | None = None
        self._self_capture: SelfCaptureSessionOwner | None = None
        self._stt_sessions = stt_sessions
        self._recognition_consumption = RecognitionConsumptionLedger()
        self._recognition_locks = {"self": asyncio.Lock(), "peer": asyncio.Lock()}
        self._self_ready_stream: RecognitionStreamIdentity | None = None
        self._self_last_input_stream: RecognitionStreamIdentity | None = None
        self._self_obsolete_streams: OrderedDict[RecognitionStreamIdentity, None] = OrderedDict()

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
        event = scoped_event_for_delivery(event)
        if event is None:
            return
        if isinstance(event, STTProviderInputTerminal):
            if event.channel != "self":
                raise ValueError("Self input terminal belongs to another channel")
            if self._self_capture is not None:
                self._self_capture.note_input_terminal(event)
            self._require_self().release_input_segment(event.identity.segment.segment_id)
            await self._after_self_input_terminal(event)
            return
        if isinstance(event, STTProviderEpochEnded):
            await self._after_self_epoch_ended(event)
            return
        if isinstance(event, STTRecognitionUnitTerminal):
            await self._admit_recognition_unit(event, channel="self")
            return
        await self._before_self_event(event)
        event = scoped_event_for_delivery(event)
        if event is None:
            return
        self._stt_sessions.record(event)
        await self._require_self().handle_stt_event(event)
        await self._after_self_event(event)

    async def _before_self_event(self, event: object) -> None:
        if isinstance(event, STTProviderTurnUpdate | STTProviderTurnTerminal) and (
            self._self_scoped_event_is_current(event)
            and (
                isinstance(event, STTProviderTurnUpdate)
                or event.outcome in ("final", "empty", "degraded", "suppressed")
            )
        ):
            await self._publish_self_session_state(STTSessionState.STREAMING)
        if isinstance(event, STTProviderTurnTerminal) and self._self_capture is not None:
            self._self_capture.note_recognition_terminal(event)

    async def _after_self_event(self, event: object) -> None:
        if (
            isinstance(event, STTProviderTurnTerminal)
            and self._self_scoped_event_is_current(event)
            and (
                event.outcome in ("failed", "expired", "cancelled")
                or event.failure_reason is not None
            )
        ):
            await self._publish_self_session_state(STTSessionState.DISCONNECTED)

    def _self_scoped_event_is_current(
        self, event: STTProviderTurnUpdate | STTProviderTurnTerminal
    ) -> bool:
        return (
            self._self_capture is None
            or event.identity.segment.activation_generation
            == self._self_capture.snapshot.generation
        )

    async def _publish_self_session_state(self, state: STTSessionState) -> None:
        if self._stt_sessions.state("self") is state:
            return
        event = STTSessionStateEvent(state=state, channel="self")
        self._stt_sessions.record(event)
        await self._require_self().handle_stt_event(event)

    def _self_input_stream(self, event: STTProviderInputTerminal) -> RecognitionStreamIdentity:
        identity = event.identity
        return RecognitionStreamIdentity(
            "self",
            identity.segment.activation_generation,
            identity.segment.capture_epoch,
            identity.provider_epoch_id,
            identity.settings_scope,
        )

    def _self_stream_scope_is_current(self, stream: RecognitionStreamIdentity) -> bool:
        capture = self._self_capture
        if capture is None:
            return True
        if stream.activation_generation != capture.snapshot.generation:
            return False
        is_current = getattr(capture, "is_current_recognition_stream", None)
        return not callable(is_current) or is_current(stream)

    def _supersede_self_stream(self, stream: RecognitionStreamIdentity | None) -> None:
        if stream is None:
            return
        self._self_obsolete_streams[stream] = None
        self._self_obsolete_streams.move_to_end(stream)
        if len(self._self_obsolete_streams) > 4096:
            self._self_obsolete_streams.popitem(last=False)

    async def _after_self_input_terminal(self, event: STTProviderInputTerminal) -> None:
        stream = self._self_input_stream(event)
        if not self._self_stream_scope_is_current(stream) or stream in self._self_obsolete_streams:
            return
        if event.outcome == "submitted":
            if self._self_last_input_stream != stream:
                self._supersede_self_stream(self._self_last_input_stream)
                if self._self_ready_stream != stream:
                    self._supersede_self_stream(self._self_ready_stream)
                self._self_last_input_stream = stream
            return
        if event.outcome in ("failed", "expired", "cancelled") or event.failure_reason:
            self._supersede_self_stream(stream)
            await self._publish_self_session_state(STTSessionState.DISCONNECTED)

    async def _after_self_epoch_ended(self, event: STTProviderEpochEnded) -> None:
        for stream in (self._self_ready_stream, self._self_last_input_stream):
            if (
                stream is not None
                and stream.provider_epoch_id == event.provider_epoch_id
                and stream not in self._self_obsolete_streams
                and self._self_stream_scope_is_current(stream)
            ):
                self._supersede_self_stream(stream)
                await self._publish_self_session_state(STTSessionState.DISCONNECTED)

    async def _admit_recognition_unit(
        self,
        event: STTRecognitionUnitTerminal,
        *,
        channel: str,
    ) -> None:
        stream = event.unit.identity.stream
        if stream.channel != channel:
            raise ValueError("Recognition unit belongs to another channel")
        async with self._recognition_locks[channel]:
            if not self._recognition_consumption.consume(event):
                return
            if not self._recognition_stream_is_current(stream):
                return
            if (
                channel == "self"
                and event.outcome in ("final", "empty")
                and stream not in self._self_obsolete_streams
            ):
                if self._self_ready_stream != stream:
                    self._supersede_self_stream(self._self_ready_stream)
                    if self._self_last_input_stream != stream:
                        self._supersede_self_stream(self._self_last_input_stream)
                    self._self_ready_stream = stream
                await self._publish_self_session_state(STTSessionState.STREAMING)
            if event.outcome != "final" or not event.unit.text:
                return
            if channel == "self":
                await self._require_self().handle_recognition_unit(event.unit)
            else:
                await self._require_peer().handle_recognition_unit(event.unit)

    def _recognition_stream_is_current(self, stream: RecognitionStreamIdentity) -> bool:
        if stream.channel == "self":
            capture = self._self_capture
            owner = self._require_self()
        else:
            capture = self._peer_capture
            owner = self._require_peer()
        return (
            capture is not None
            and capture.is_current_recognition_stream(stream)
            and owner.local_asr_runtime.is_current_recognition_stream(stream.channel, stream)
        )

    async def peer_event_handler(self, event: object) -> None:
        if isinstance(event, STTProviderInputTerminal):
            if event.channel != "peer":
                raise ValueError("Peer input terminal belongs to another channel")
            admissions = self._require_peer_capture().note_input_terminal(event)
            self._require_peer().release_input_segment(event.identity.segment.segment_id)
            if not admissions:
                return
            async with self._recognition_locks["peer"]:
                peer = self._require_peer()
                for receipt, terminal in admissions:
                    await peer.handle_provider_turn_terminal(receipt, terminal)
            return
        if isinstance(event, STTRecognitionUnitTerminal):
            await self._admit_recognition_unit(event, channel="peer")
            return
        if isinstance(event, STTProviderTurnUpdate):
            return
        if isinstance(event, STTProviderEpochEnded):
            return
        if isinstance(event, STTProviderTurnTerminal):
            async with self._recognition_locks["peer"]:
                admissions = self._require_peer_capture().admit_provider_terminal(event)
                peer = self._require_peer()
                for receipt, terminal in admissions:
                    await peer.handle_provider_turn_terminal(receipt, terminal)
            return
        self._stt_sessions.record(event)
        await self._require_peer().handle_stt_event(event)

    async def retired_event_handler(self, event: object) -> None:
        if isinstance(event, STTProviderInputTerminal):
            if event.channel == "self":
                if self._self_capture is not None:
                    self._self_capture.note_input_terminal(event)
                self._require_self().release_input_segment(event.identity.segment.segment_id)
                await self._after_self_input_terminal(event)
            else:
                admissions = self._require_peer_capture().note_input_terminal(event)
                self._require_peer().release_input_segment(event.identity.segment.segment_id)
                if not admissions:
                    return
                async with self._recognition_locks["peer"]:
                    for receipt, terminal in admissions:
                        await self._require_peer().handle_provider_turn_terminal(receipt, terminal)
            return
        if isinstance(event, STTRecognitionUnitTerminal):
            await self._admit_recognition_unit(event, channel=event.unit.identity.stream.channel)
            return
        if isinstance(event, STTProviderEpochEnded):
            await self._after_self_epoch_ended(event)
            return
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
