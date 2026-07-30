from __future__ import annotations

from dataclasses import dataclass, field, replace
from uuid import uuid4

import pytest

from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.orchestrator.configuration import (
    TranslationRuntimeConfig,
    TranslationRuntimeConfigurationOwner,
)
from puripuly_heart.core.orchestrator.translation_diagnostics import (
    TranslationLatencyDiagnosticsOwner,
)
from puripuly_heart.core.orchestrator.translation_output_projection import (
    ActiveSelfProjection,
    TranslationOutputProjectionOwner,
)
from puripuly_heart.core.orchestrator.translation_turn import TranslationOutputSubmission
from puripuly_heart.core.overlay.state import ActiveSelfOverlayMetadata
from puripuly_heart.core.runtime.output import OutputRuntime
from puripuly_heart.domain.events import UIEvent, UIEventType
from puripuly_heart.domain.models import OSCMessage, Translation


@dataclass(slots=True)
class RecordingChatbox:
    messages: list[OSCMessage] = field(default_factory=list)

    def enqueue(self, message: OSCMessage) -> None:
        self.messages.append(message)

    def send_immediate(self, text: str) -> bool:
        _ = text
        return True

    def send_typing(self, is_typing: bool) -> None:
        _ = is_typing

    def set_typing_reason(self, reason: str, active: bool) -> None:
        _ = reason, active

    def clear_typing_reasons(self) -> None:
        return None

    def process_due(self) -> None:
        return None


@dataclass(slots=True)
class RecordingUiMessages:
    events: list[UIEvent] = field(default_factory=list)

    async def publish(self, event: UIEvent) -> None:
        self.events.append(event)


@dataclass(slots=True)
class RecordingOverlay:
    events: list[object] = field(default_factory=list)
    metadata: ActiveSelfOverlayMetadata | None = None
    fail: bool = False

    async def emit(self, event: object) -> None:
        if self.fail:
            raise RuntimeError("overlay failed")
        self.events.append(event)

    def active_self_overlay_metadata(self) -> ActiveSelfOverlayMetadata | None:
        return self.metadata


def make_owner(
    *,
    configuration: TranslationRuntimeConfig | None = None,
    overlay: RecordingOverlay | None = None,
) -> tuple[
    TranslationOutputProjectionOwner,
    RecordingChatbox,
    RecordingUiMessages,
    TranslationRuntimeConfigurationOwner,
]:
    clock = FakeClock(_now=10.0)
    config_owner = TranslationRuntimeConfigurationOwner(configuration or TranslationRuntimeConfig())
    chatbox = RecordingChatbox()
    ui_messages = RecordingUiMessages()
    diagnostics = TranslationLatencyDiagnosticsOwner(
        clock=clock,
        config_snapshot=config_owner.snapshot,
    )
    owner = TranslationOutputProjectionOwner(
        output_runtime=OutputRuntime(
            chatbox=chatbox,
            clock=clock,
            overlay_sink=overlay,
        ),
        ui_messages=ui_messages,
        diagnostics=diagnostics,
        clock=clock,
    )
    return owner, chatbox, ui_messages, config_owner


def submission(
    config_owner: TranslationRuntimeConfigurationOwner,
    *,
    channel: str,
    outcome: str,
    translation: Translation | None = None,
    failure_code: str | None = None,
) -> TranslationOutputSubmission:
    utterance_id = translation.utterance_id if translation is not None else uuid4()
    return TranslationOutputSubmission(
        parent_utterance_id=utterance_id,
        child_utterance_id=utterance_id,
        sequence=0,
        channel=channel,
        source="Peer" if channel == "peer" else "Mic",
        source_text="source text",
        source_language="ja",
        target_language="en",
        outcome=outcome,
        config_snapshot=config_owner.snapshot(),
        translation=translation,
        failure_code=failure_code,
    )


@pytest.mark.asyncio
async def test_translated_self_projects_ui_overlay_and_chatbox_once() -> None:
    overlay = RecordingOverlay()
    owner, chatbox, ui_messages, config_owner = make_owner(overlay=overlay)
    translation = Translation(
        utterance_id=uuid4(),
        text="translated",
        source_text="source text",
        source_language="ja",
        target_language="en",
        channel="self",
    )

    receipt = await owner.project_translation_result(
        submission(
            config_owner,
            channel="self",
            outcome="translated",
            translation=translation,
        )
    )

    assert receipt.clear_runtime_latency_bookkeeping
    assert not receipt.complete_peer_logical_turn
    assert [event.type for event in ui_messages.events] == [
        UIEventType.TRANSLATION_DONE,
        UIEventType.OSC_SENT,
    ]
    assert [getattr(event, "type") for event in overlay.events] == [
        "translation_final",
        "utterance_closed",
    ]
    assert [message.text for message in chatbox.messages] == ["source text (translated)"]


@pytest.mark.asyncio
async def test_translated_peer_projects_overlay_and_ui_but_hard_denies_chatbox() -> None:
    overlay = RecordingOverlay()
    owner, chatbox, ui_messages, config_owner = make_owner(overlay=overlay)
    translation = Translation(
        utterance_id=uuid4(),
        text="translated",
        source_text="source text",
        source_language="ja",
        target_language="en",
        channel="peer",
    )

    await owner.project_translation_result(
        submission(
            config_owner,
            channel="peer",
            outcome="translated",
            translation=translation,
        )
    )

    assert [event.type for event in ui_messages.events] == [UIEventType.TRANSLATION_DONE]
    assert [getattr(event, "type") for event in overlay.events] == [
        "translation_final",
        "utterance_closed",
    ]
    assert chatbox.messages == []
    peer_decisions = [
        decision
        for decision in owner.routing_decisions
        if decision.route == "self_chatbox" and decision.publication_kind == "peer_subtitle"
    ]
    assert len(peer_decisions) == 1
    assert peer_decisions[0].reason == "peer_chatbox_denied"


@pytest.mark.asyncio
async def test_peer_source_only_projects_transcript_and_denial_once() -> None:
    overlay = RecordingOverlay()
    owner, chatbox, ui_messages, config_owner = make_owner(overlay=overlay)

    receipt = await owner.project_translation_result(
        submission(config_owner, channel="peer", outcome="source_only")
    )

    assert receipt.clear_runtime_latency_bookkeeping
    assert [getattr(event, "type") for event in overlay.events] == [
        "peer_transcript_final",
        "utterance_closed",
    ]
    assert ui_messages.events == []
    assert chatbox.messages == []
    assert [decision.decision for decision in owner.routing_decisions] == [
        "published",
        "published",
        "denied",
    ]
    assert owner.routing_decisions[-1].reason == "peer_chatbox_denied"


@pytest.mark.asyncio
async def test_self_failure_with_fallback_projects_incomplete_close_and_source_chatbox() -> None:
    overlay = RecordingOverlay()
    owner, chatbox, ui_messages, config_owner = make_owner(
        configuration=replace(
            TranslationRuntimeConfig(),
            fallback_transcript_only=True,
        ),
        overlay=overlay,
    )

    await owner.project_translation_result(
        submission(config_owner, channel="self", outcome="failed")
    )

    assert getattr(overlay.events[0], "type") == "utterance_closed"
    assert not getattr(overlay.events[0], "is_final")
    assert [message.text for message in chatbox.messages] == ["source text"]
    assert [event.type for event in ui_messages.events] == [UIEventType.OSC_SENT]


@pytest.mark.asyncio
async def test_stale_peer_completion_only_denies_chatbox_and_requests_turn_cleanup() -> None:
    overlay = RecordingOverlay()
    owner, chatbox, ui_messages, config_owner = make_owner(overlay=overlay)

    receipt = await owner.project_translation_result(
        submission(
            config_owner,
            channel="peer",
            outcome="failed",
            failure_code="stale_provider_completion",
        )
    )

    assert receipt.complete_peer_logical_turn
    assert overlay.events == []
    assert chatbox.messages == []
    assert ui_messages.events == []
    assert [decision.reason for decision in owner.routing_decisions] == ["peer_chatbox_denied"]


@pytest.mark.asyncio
async def test_overlay_failure_does_not_suppress_self_ui_or_chatbox() -> None:
    overlay = RecordingOverlay(fail=True)
    owner, chatbox, ui_messages, config_owner = make_owner(overlay=overlay)
    translation = Translation(
        utterance_id=uuid4(),
        text="translated",
        source_text="source text",
        channel="self",
    )

    await owner.project_translation_result(
        submission(
            config_owner,
            channel="self",
            outcome="translated",
            translation=translation,
        )
    )

    assert [event.type for event in ui_messages.events] == [
        UIEventType.TRANSLATION_DONE,
        UIEventType.OSC_SENT,
    ]
    assert [message.text for message in chatbox.messages] == ["source text (translated)"]
    assert any(
        decision.reason == "destination_publish_failed" for decision in owner.routing_decisions
    )


@pytest.mark.asyncio
async def test_active_self_projection_owns_soft_reuse_and_sticky_secondary_decision() -> None:
    merge_id = uuid4()
    overlay = RecordingOverlay(
        metadata=ActiveSelfOverlayMetadata(
            text="hello",
            secondary_text="sticky",
            utterance_id=merge_id,
            occupant_key=f"self:{merge_id}",
            update_id=None,
            origin_wall_clock_ms=None,
            session_scope=None,
            source_text_hash=None,
            source_text_len=None,
            logical_turn_key=None,
            primary_language="en",
            secondary_language="ko",
        )
    )
    owner, _chatbox, _ui_messages, _config_owner = make_owner(overlay=overlay)

    receipt = await owner.sync_active_self(
        ActiveSelfProjection(
            merge_id=merge_id,
            active_text="hello?",
            spec_text="hello",
            spec_translation=Translation(
                utterance_id=merge_id,
                text="spec",
            ),
            source_language="en",
            target_language="ko",
            resume_pending=False,
            resume_confirmed=False,
        )
    )

    assert owner.soft_reuse_mode(" hello ", "hello...") == "soft_boundary"
    assert owner.soft_reuse_mode("hello", "hello?") is None
    assert receipt.source == "sticky_cache"
    assert receipt.secondary_text == "sticky"
    assert receipt.emitted
