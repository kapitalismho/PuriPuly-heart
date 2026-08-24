from __future__ import annotations

import asyncio
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
from puripuly_heart.core.orchestrator.translation_turn import (
    TranslationOutputSubmission,
    TranslationTurnChild,
    TranslationTurnOutcome,
)
from puripuly_heart.core.overlay.state import ActiveSelfOverlayMetadata
from puripuly_heart.core.runtime.output import OutputRuntime
from puripuly_heart.core.runtime_logging import SessionLoggingMode
from puripuly_heart.domain.events import UIEvent, UIEventType
from puripuly_heart.domain.models import OSCMessage, Transcript, Translation
from tests.core.test_translation_owner_branch_coverage import (
    _make_runtime_logging_capture,
    _runtime_log_messages,
)


@dataclass(slots=True)
class RecordingChatbox:
    messages: list[OSCMessage] = field(default_factory=list)
    fail: bool = False

    def enqueue(self, message: OSCMessage) -> None:
        if self.fail:
            raise RuntimeError("chatbox failed")
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
    chatbox: RecordingChatbox | None = None,
) -> tuple[
    TranslationOutputProjectionOwner,
    RecordingChatbox,
    RecordingUiMessages,
    TranslationRuntimeConfigurationOwner,
]:
    clock = FakeClock(_now=10.0)
    config_owner = TranslationRuntimeConfigurationOwner(configuration or TranslationRuntimeConfig())
    chatbox = chatbox or RecordingChatbox()
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


def self_children(
    config_owner: TranslationRuntimeConfigurationOwner,
    *,
    parent_id=None,
    turn_generation: int = 0,
    turn_order: int = 0,
    source_parts: tuple[str, ...] = ("source text",),
) -> tuple[TranslationTurnChild, ...]:
    parent_id = parent_id or uuid4()
    snapshot = config_owner.snapshot()
    children: list[TranslationTurnChild] = []
    sequence = 0
    for source_part in source_parts:
        for target_index, target_language in enumerate(snapshot.value.self_target_languages):
            child_id = uuid4()
            children.append(
                TranslationTurnChild(
                    parent_utterance_id=parent_id,
                    utterance_id=child_id,
                    sequence=sequence,
                    target_index=target_index,
                    turn_generation=turn_generation,
                    turn_order=turn_order,
                    transcript=Transcript(
                        utterance_id=child_id,
                        text=source_part,
                        is_final=True,
                        channel="self",
                    ),
                    detected_language="en",
                    target_language=target_language,
                    source="Mic",
                    turn_kind="self",
                    context_policy="integrated_preferred",
                    config_snapshot=snapshot,
                )
            )
            sequence += 1
    return tuple(children)


def self_submission(
    child: TranslationTurnChild,
    *,
    outcome: TranslationTurnOutcome = "translated",
    text: str = "translated",
) -> TranslationOutputSubmission:
    translation = (
        Translation(
            utterance_id=child.utterance_id,
            text=text,
            source_text=child.transcript.text,
            source_language="en",
            target_language=child.target_language,
            channel="self",
            logical_turn_key=f"self:{child.parent_utterance_id}",
        )
        if outcome == "translated"
        else None
    )
    return TranslationOutputSubmission(
        parent_utterance_id=child.parent_utterance_id,
        child_utterance_id=child.utterance_id,
        sequence=child.sequence,
        channel="self",
        source=child.source,
        source_text=child.transcript.text,
        source_language=child.detected_language,
        target_language=child.target_language,
        outcome=outcome,
        config_snapshot=child.config_snapshot,
        translation=translation,
        target_index=child.target_index,
        turn_generation=child.turn_generation,
        turn_order=child.turn_order,
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
    assert chatbox.messages[0].self_turn_key is None


@pytest.mark.asyncio
async def test_lifecycle_single_target_chatbox_carries_turn_identity_without_revision() -> None:
    configuration = TranslationRuntimeConfig(
        target_language="zh-CN",
        self_target_languages=("zh-CN",),
    )
    owner, chatbox, _ui_messages, config_owner = make_owner(configuration=configuration)
    child = self_children(config_owner, turn_generation=2, turn_order=7)[0]
    assert owner.admit_self_turn((child,))

    await owner.project_translation_result(self_submission(child, text="single"))

    assert [message.text for message in chatbox.messages] == ["source text (single)"]
    assert chatbox.messages[0].self_turn_key == (2, 7)
    assert chatbox.messages[0].presentation_revision == 0
    assert chatbox.messages[0].target_indexes == (0,)
    assert chatbox.messages[0].target_languages == ("zh-CN",)


@pytest.mark.asyncio
async def test_dual_target_projection_publishes_first_completion_then_ordered_snapshot() -> None:
    configuration = TranslationRuntimeConfig(
        target_language="zh-CN",
        self_target_languages=("zh-CN", "ja"),
    )
    owner, chatbox, _ui_messages, config_owner = make_owner(configuration=configuration)
    children = self_children(config_owner, turn_order=4)
    primary, secondary = children
    assert owner.admit_self_turn(children)

    await owner.project_translation_result(self_submission(secondary, text="こんにちは"))
    await owner.complete_self_target(secondary, "translated")
    await owner.project_translation_result(self_submission(primary, text="你好"))
    await owner.complete_self_target(primary, "translated")

    assert [message.utterance_id for message in chatbox.messages] == [
        primary.parent_utterance_id,
        primary.parent_utterance_id,
    ]
    assert [message.text for message in chatbox.messages] == [
        "こんにちは",
        "你好\nこんにちは",
    ]
    assert [message.self_turn_key for message in chatbox.messages] == [(0, 4), (0, 4)]
    assert [message.presentation_revision for message in chatbox.messages] == [1, 2]
    assert [message.target_indexes for message in chatbox.messages] == [(1,), (0, 1)]
    assert [message.target_languages for message in chatbox.messages] == [
        ("ja",),
        ("zh-CN", "ja"),
    ]
    assert [
        decision.metadata["presentation_revision"]
        for decision in owner.routing_decisions
        if decision.decision == "published"
        and decision.route == "self_chatbox"
        and "presentation_revision" in decision.metadata
    ] == [1, 2]
    assert [
        decision.metadata["target_indexes"]
        for decision in owner.routing_decisions
        if decision.decision == "published"
        and decision.route == "self_chatbox"
        and "target_indexes" in decision.metadata
    ] == ["1", "0,1"]
    assert owner.self_turn_aggregate_count == 0
    assert owner.self_turn_tombstone_count == 1


@pytest.mark.asyncio
async def test_duplicate_or_terminal_child_output_cannot_create_another_revision() -> None:
    configuration = TranslationRuntimeConfig(
        target_language="zh-CN",
        self_target_languages=("zh-CN", "ja"),
    )
    owner, chatbox, ui_messages, config_owner = make_owner(configuration=configuration)
    children = self_children(config_owner)
    assert owner.admit_self_turn(children)

    first = await owner.project_translation_result(self_submission(children[0], text="first"))
    duplicate = await owner.project_translation_result(
        self_submission(children[0], text="changed duplicate")
    )
    await owner.complete_self_target(children[1], "cancelled")
    late = await owner.project_translation_result(self_submission(children[1], text="late success"))

    assert first.record_runtime_translation
    assert not duplicate.record_runtime_translation
    assert not late.record_runtime_translation
    assert [message.text for message in chatbox.messages] == ["first"]
    assert sum(event.type == UIEventType.TRANSLATION_DONE for event in ui_messages.events) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcomes", "expected"),
    [
        (("translated", "failed"), ["successful"]),
        (("failed", "translated"), ["successful"]),
        (("failed", "failed"), []),
    ],
)
async def test_dual_target_failure_never_exposes_source_text(
    outcomes: tuple[TranslationTurnOutcome, TranslationTurnOutcome],
    expected: list[str],
) -> None:
    configuration = TranslationRuntimeConfig(
        target_language="zh-CN",
        self_target_languages=("zh-CN", "ja"),
        fallback_transcript_only=True,
    )
    owner, chatbox, _ui_messages, config_owner = make_owner(configuration=configuration)
    children = self_children(config_owner)
    assert owner.admit_self_turn(children)

    for child, outcome in zip(children, outcomes, strict=True):
        await owner.project_translation_result(
            self_submission(child, outcome=outcome, text="successful")
        )
        await owner.complete_self_target(child, outcome)

    assert [message.text for message in chatbox.messages] == expected
    assert all("source text" not in message.text for message in chatbox.messages)


@pytest.mark.asyncio
async def test_newer_visible_turn_suppresses_older_late_complete_revision() -> None:
    configuration = TranslationRuntimeConfig(
        target_language="zh-CN",
        self_target_languages=("zh-CN", "ja"),
    )
    owner, chatbox, ui_messages, config_owner = make_owner(configuration=configuration)
    older = self_children(config_owner, turn_order=10)
    newer = self_children(config_owner, turn_order=11)
    assert owner.admit_self_turn(older)
    assert owner.admit_self_turn(newer)

    await owner.project_translation_result(self_submission(older[0], text="old primary"))
    await owner.complete_self_target(older[0], "translated")
    await owner.project_translation_result(self_submission(newer[0], text="new primary"))
    await owner.complete_self_target(newer[0], "translated")
    older_late = await owner.project_translation_result(
        self_submission(older[1], text="old secondary")
    )
    await owner.complete_self_target(older[1], "translated")
    await owner.project_translation_result(self_submission(newer[1], text="new secondary"))
    await owner.complete_self_target(newer[1], "translated")

    assert older_late.record_runtime_translation
    assert [message.text for message in chatbox.messages] == [
        "old primary",
        "new primary",
        "new primary\nnew secondary",
    ]
    assert sum(event.type == UIEventType.TRANSLATION_DONE for event in ui_messages.events) == 4


@pytest.mark.asyncio
async def test_near_simultaneous_target_completions_never_finish_with_partial_revision() -> None:
    configuration = TranslationRuntimeConfig(
        target_language="zh-CN",
        self_target_languages=("zh-CN", "ja"),
    )
    owner, chatbox, _ui_messages, config_owner = make_owner(configuration=configuration)
    children = self_children(config_owner)
    assert owner.admit_self_turn(children)

    await asyncio.gather(
        owner.project_translation_result(self_submission(children[0], text="primary")),
        owner.project_translation_result(self_submission(children[1], text="secondary")),
    )
    await asyncio.gather(
        owner.complete_self_target(children[0], "translated"),
        owner.complete_self_target(children[1], "translated"),
    )

    assert len(chatbox.messages) == 2
    assert chatbox.messages[-1].text == "primary\nsecondary"


@pytest.mark.asyncio
async def test_new_generation_rejects_retired_turn_output_and_bounds_aggregate_state() -> None:
    configuration = TranslationRuntimeConfig(
        target_language="zh-CN",
        self_target_languages=("zh-CN", "ja"),
    )
    owner, chatbox, _ui_messages, config_owner = make_owner(configuration=configuration)
    retired = self_children(config_owner, turn_generation=0, turn_order=3)
    current = self_children(config_owner, turn_generation=1, turn_order=0)
    assert owner.admit_self_turn(retired)
    assert owner.admit_self_turn(current)

    retired_receipt = await owner.project_translation_result(
        self_submission(retired[0], text="retired")
    )
    await owner.project_translation_result(self_submission(current[0], text="current"))

    assert not retired_receipt.record_runtime_translation
    assert [message.text for message in chatbox.messages] == ["current"]
    assert owner.self_turn_aggregate_count == 1
    assert owner.self_turn_tombstone_count == 1


@pytest.mark.asyncio
async def test_lifecycle_generation_retirement_rejects_output_without_new_admission() -> None:
    configuration = TranslationRuntimeConfig(
        target_language="zh-CN",
        self_target_languages=("zh-CN", "ja"),
    )
    owner, chatbox, _ui_messages, config_owner = make_owner(configuration=configuration)
    runtime_logging, log_stream = _make_runtime_logging_capture()
    runtime_logging.set_mode(SessionLoggingMode.DETAILED)
    owner.diagnostics.runtime_logging = runtime_logging
    retired = self_children(config_owner, turn_generation=0, turn_order=3)
    assert owner.admit_self_turn(retired)

    owner.retire_turn_generation("self", 1)
    receipt = await owner.project_translation_result(self_submission(retired[0], text="retired"))

    assert not receipt.record_runtime_translation
    assert chatbox.messages == []
    assert owner.self_turn_aggregate_count == 0
    assert owner.self_turn_tombstone_count == 1
    stale = next(
        message
        for message in _runtime_log_messages(log_stream)
        if "translation_result_suppressed_stale_turn" in message
    )
    assert "turn_generation=0" in stale
    assert "target_indexes=(0,)" in stale
    assert "target_languages=('zh-CN',)" in stale
    assert "revision=0" in stale


@pytest.mark.asyncio
async def test_generation_retirement_wins_before_waiting_snapshot_publication() -> None:
    configuration = TranslationRuntimeConfig(
        target_language="zh-CN",
        self_target_languages=("zh-CN", "ja"),
    )
    owner, chatbox, _ui_messages, config_owner = make_owner(configuration=configuration)
    retired = self_children(config_owner, turn_generation=0, turn_order=3)
    current = self_children(config_owner, turn_generation=1, turn_order=0)
    assert owner.admit_self_turn(retired)

    await owner._self_publish_lock.acquire()
    retired_output = asyncio.create_task(
        owner.project_translation_result(self_submission(retired[0], text="retired"))
    )
    await asyncio.sleep(0)
    assert owner.admit_self_turn(current)
    owner._self_publish_lock.release()
    receipt = await retired_output

    assert not receipt.record_runtime_translation
    assert chatbox.messages == []


@pytest.mark.asyncio
async def test_target_with_multiple_source_runs_publishes_only_after_all_runs_complete() -> None:
    configuration = TranslationRuntimeConfig(
        target_language="zh-CN",
        self_target_languages=("zh-CN", "ja"),
    )
    owner, chatbox, _ui_messages, config_owner = make_owner(configuration=configuration)
    children = self_children(config_owner, source_parts=("first", "second"))
    assert owner.admit_self_turn(children)

    await owner.project_translation_result(self_submission(children[0], text="primary one"))
    await owner.complete_self_target(children[0], "translated")
    assert chatbox.messages == []
    await owner.project_translation_result(self_submission(children[2], text="primary two"))
    await owner.complete_self_target(children[2], "translated")
    assert [message.text for message in chatbox.messages] == ["primary one primary two"]
    await owner.project_translation_result(self_submission(children[1], text="secondary one"))
    await owner.complete_self_target(children[1], "translated")
    await owner.project_translation_result(self_submission(children[3], text="secondary two"))
    await owner.complete_self_target(children[3], "translated")

    assert chatbox.messages[-1].text == ("primary one primary two\nsecondary one secondary two")


@pytest.mark.asyncio
async def test_dual_target_publication_denial_records_identity_and_attempt_latency() -> None:
    configuration = TranslationRuntimeConfig(
        target_language="zh-CN",
        self_target_languages=("zh-CN", "ja"),
    )
    owner, chatbox, _ui_messages, config_owner = make_owner(
        configuration=configuration,
        chatbox=RecordingChatbox(fail=True),
    )
    runtime_logging, log_stream = _make_runtime_logging_capture()
    runtime_logging.set_mode(SessionLoggingMode.DETAILED)
    owner.diagnostics.runtime_logging = runtime_logging
    children = self_children(config_owner, turn_generation=3, turn_order=4)
    assert owner.admit_self_turn(children)

    await owner.project_translation_result(self_submission(children[0], text="private primary"))
    await owner.complete_self_target(children[0], "translated")
    await owner.project_translation_result(self_submission(children[1], text="private secondary"))
    await owner.complete_self_target(children[1], "translated")

    messages = _runtime_log_messages(log_stream)
    first_denied = next(
        message for message in messages if "translation_first_result_publication_denied" in message
    )
    complete_denied = next(
        message
        for message in messages
        if "translation_complete_result_publication_denied" in message
    )
    assert chatbox.messages == []
    assert "turn_generation=3 turn_order=4" in first_denied
    assert "target_indexes=(0,)" in first_denied
    assert "target_languages=('zh-CN',)" in first_denied
    assert "revision=1" in first_denied
    assert "reason=destination_publish_failed" in first_denied
    assert "publication_attempt_elapsed_ms=0" in first_denied
    assert "first_visible_elapsed_ms=None" in first_denied
    assert "target_indexes=(0, 1)" in complete_denied
    assert "revision=2" in complete_denied
    assert "complete_visible_elapsed_ms=None" in complete_denied
    combined = "\n".join(messages)
    assert "private primary" not in combined
    assert "private secondary" not in combined


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
