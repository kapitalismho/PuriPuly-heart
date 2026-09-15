from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from uuid import UUID, uuid4

import pytest

from puripuly_heart.config.overlay_calibration import OverlayCalibration
from puripuly_heart.core.overlay.bridge import OverlayBridge
from puripuly_heart.core.overlay.presenter import OverlayPresenter
from puripuly_heart.domain.events import UIEventType
from puripuly_heart.domain.models import Translation
from tests.helpers.translation_owners import compose_translation_test_harness


@dataclass
class ConcurrentRecordingProvider:
    started: asyncio.Queue[dict[str, object]] = field(default_factory=asyncio.Queue)
    release: asyncio.Event = field(default_factory=asyncio.Event)

    async def translate(
        self,
        *,
        utterance_id: UUID,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
        scene_participant_count: int | None = None,
    ) -> Translation:
        await self.started.put(
            {
                "utterance_id": utterance_id,
                "text": text,
                "system_prompt": system_prompt,
                "source_language": source_language,
                "target_language": target_language,
                "context": context,
            }
        )
        await self.release.wait()
        return Translation(
            utterance_id=utterance_id,
            text=f"translated-{target_language}",
            source_text=text,
            source_language=source_language,
            target_language=target_language,
        )

    async def close(self) -> None:
        return None


@dataclass
class TargetControlledProvider:
    started: asyncio.Queue[tuple[str, str]] = field(default_factory=asyncio.Queue)
    releases: dict[tuple[str, str], asyncio.Event] = field(default_factory=dict)
    failed_targets: set[str] = field(default_factory=set)

    async def translate(
        self,
        *,
        utterance_id: UUID,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
        scene_participant_count: int | None = None,
    ) -> Translation:
        _ = system_prompt, context
        key = (text, target_language)
        release = self.releases.setdefault(key, asyncio.Event())
        await self.started.put(key)
        await release.wait()
        if target_language in self.failed_targets:
            raise RuntimeError("target failed")
        return Translation(
            utterance_id=utterance_id,
            text=f"translated-{target_language}",
            source_text=text,
            source_language=source_language,
            target_language=target_language,
        )

    async def close(self) -> None:
        return None


class RecordingOsc:
    def __init__(self) -> None:
        self.messages: list[object] = []

    def enqueue(self, message: object) -> None:
        self.messages.append(message)

    def send_immediate(self, _text: str) -> bool:
        return True

    def send_typing(self, _is_typing: bool) -> None:
        return None

    def set_typing_reason(self, _reason: str, _active: bool) -> None:
        return None

    def clear_typing_reasons(self) -> None:
        return None

    def process_due(self) -> None:
        return None


@dataclass
class RecordingOverlay:
    events: list[object] = field(default_factory=list)

    async def emit(self, event: object) -> None:
        self.events.append(event)

    def active_self_overlay_metadata(self) -> None:
        return None


class RecordingRuntimeLogging:

    def __init__(self) -> None:
        self.messages: list[str] = []

    def emit_basic(self, message: str, *, level: int = 20) -> None:
        _ = level
        self.messages.append(message)

    def emit_diagnostic(self, message: str, *, level: int = 20) -> bool:
        _ = level
        self.messages.append(message)
        return True

    def emit_diagnostic_lazy(self, build_message, *, level: int = 20) -> bool:
        _ = level
        self.messages.append(build_message())
        return True


class StalledBridgeConnection:
    def __init__(self) -> None:
        self.send_started = asyncio.Event()
        self.release_send = asyncio.Event()
        self.sent_payloads: list[str] = []

    async def send(self, payload: str) -> None:
        self.send_started.set()
        await self.release_send.wait()
        self.sent_payloads.append(payload)

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_dual_target_turns_admit_in_order_and_execute_without_provider_barriers() -> None:
    provider = ConcurrentRecordingProvider()
    harness = compose_translation_test_harness(
        stt=None,
        llm=provider,
        osc=RecordingOsc(),
        source_language="en",
        target_language="zh-CN",
        self_target_languages=("zh-CN", "ja"),
    )

    try:
        first_parent = await harness.self_owner.submit_text("first source")
        first_calls = [
            await asyncio.wait_for(provider.started.get(), timeout=1),
            await asyncio.wait_for(provider.started.get(), timeout=1),
        ]
        second_parent = await harness.self_owner.submit_text("second source")
        second_calls = [
            await asyncio.wait_for(provider.started.get(), timeout=1),
            await asyncio.wait_for(provider.started.get(), timeout=1),
        ]

        assert {call["target_language"] for call in first_calls} == {"zh-CN", "ja"}
        assert {call["target_language"] for call in second_calls} == {"zh-CN", "ja"}
        assert len({call["utterance_id"] for call in first_calls}) == 2
        assert len({call["utterance_id"] for call in second_calls}) == 2
        assert first_parent != second_parent
        assert all(call["context"] == "" for call in first_calls)
        assert all(call["context"] == '- [self] "first source"' for call in second_calls)
    finally:
        provider.release.set()
        await harness.translation_turns.wait_for_idle()
        await harness.translation_turns.close()


@pytest.mark.asyncio
async def test_failed_child_creation_does_not_retain_prepared_self_request() -> None:
    provider = ConcurrentRecordingProvider()
    provider.release.set()
    harness = compose_translation_test_harness(
        stt=None,
        llm=provider,
        osc=RecordingOsc(),
        source_language="en",
        target_language="zh-CN",
        self_target_languages=("zh-CN", "ja"),
    )
    original_child_created = harness.translation_turns.on_child_created

    async def fail_primary_creation(child) -> None:
        if child.channel == "self" and child.target_index == 0:
            raise RuntimeError("primary child creation failed")
        await original_child_created(child)

    harness.translation_turns.on_child_created = fail_primary_creation
    try:
        await harness.self_owner.submit_text("source")
        await harness.translation_turns.wait_for_idle()
        assert harness.self_owner._admitted_requests == {}
        assert harness.translation_turns.has_resources is False
    finally:
        await harness.translation_turns.close()


@pytest.mark.asyncio
async def test_end_to_end_secondary_first_publishes_progressive_parent_snapshots() -> None:
    provider = TargetControlledProvider()
    osc = RecordingOsc()
    overlay = RecordingOverlay()
    harness = compose_translation_test_harness(
        stt=None,
        llm=provider,
        osc=osc,
        overlay_sink=overlay,
        source_language="en",
        target_language="zh-CN",
        self_target_languages=("zh-CN", "ja"),
    )

    try:
        parent_id = await harness.self_owner.submit_text("source text")
        started = {
            await asyncio.wait_for(provider.started.get(), timeout=1),
            await asyncio.wait_for(provider.started.get(), timeout=1),
        }
        assert started == {("source text", "zh-CN"), ("source text", "ja")}

        provider.releases[("source text", "ja")].set()
        for _ in range(100):
            if len(osc.messages) == 1:
                break
            await asyncio.sleep(0)
        assert [(message.utterance_id, message.text) for message in osc.messages] == [
            (parent_id, "translated-ja")
        ]
        assert osc.messages[0].target_indexes == (1,)
        assert osc.messages[0].target_languages == ("ja",)
        first_ui_events = []
        while not harness.ui_events.empty():
            first_ui_events.append(harness.ui_events.get_nowait())
        assert sum(event.type == UIEventType.TRANSCRIPT_FINAL for event in first_ui_events) == 1
        assert sum(event.type == UIEventType.TRANSLATION_DONE for event in first_ui_events) == 0
        assert sum(event.type == UIEventType.OSC_SENT for event in first_ui_events) == 1
        assert [event.type for event in overlay.events] == ["self_transcript_final"]

        provider.releases[("source text", "zh-CN")].set()
        await harness.translation_turns.wait_for_idle()
        assert [(message.utterance_id, message.text) for message in osc.messages] == [
            (parent_id, "translated-ja"),
            (parent_id, "translated-zh-CN\ntranslated-ja"),
        ]
        assert osc.messages[1].target_indexes == (0, 1)
        assert osc.messages[1].target_languages == ("zh-CN", "ja")
        final_ui_events = []
        while not harness.ui_events.empty():
            final_ui_events.append(harness.ui_events.get_nowait())
        translation_events = [
            event for event in final_ui_events if event.type == UIEventType.TRANSLATION_DONE
        ]
        assert [event.utterance_id for event in translation_events] == [parent_id]
        assert [event.payload.target_language for event in translation_events] == ["zh-CN"]
        assert [event.type for event in overlay.events] == [
            "self_transcript_final",
            "translation_final",
            "utterance_closed",
        ]
        assert all(event.utterance_id == parent_id for event in overlay.events)
        assert len(harness.self_runtime.utterances) == 2
        assert all(
            bundle.final is not None and bundle.final.text == "source text"
            for bundle in harness.self_runtime.utterances.values()
        )
        assert harness.output_projection.self_turn_aggregate_count == 0
    finally:
        for release in provider.releases.values():
            release.set()
        await harness.translation_turns.close()


@pytest.mark.asyncio
async def test_two_dual_target_parents_complete_secondary_first_through_stalled_bridge() -> None:
    provider = TargetControlledProvider()
    osc = RecordingOsc()
    bridge = OverlayBridge(session_token="dual-target-token")
    connection = StalledBridgeConnection()
    bridge._authenticated_connections.add(connection)  # type: ignore[arg-type]
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        bridge=bridge,
    )
    harness = compose_translation_test_harness(
        stt=None,
        llm=provider,
        osc=osc,
        overlay_sink=presenter,
        source_language="en",
        target_language="zh-CN",
        self_target_languages=("zh-CN", "ja"),
    )

    try:
        first_parent = await harness.self_owner.submit_text("first parent")
        first_started = {
            await asyncio.wait_for(provider.started.get(), timeout=1) for _ in range(2)
        }
        assert first_started == {
            ("first parent", "zh-CN"),
            ("first parent", "ja"),
        }
        await connection.send_started.wait()
        provider.releases[("first parent", "ja")].set()
        for _ in range(100):
            if len(osc.messages) == 1:
                break
            await asyncio.sleep(0)
        assert osc.messages[0].utterance_id == first_parent
        assert osc.messages[0].target_indexes == (1,)
        provider.releases[("first parent", "zh-CN")].set()
        await harness.translation_turns.wait_for_parent(first_parent)

        second_parent = await harness.self_owner.submit_text("second parent")
        second_started = {
            await asyncio.wait_for(provider.started.get(), timeout=1) for _ in range(2)
        }
        assert second_started == {
            ("second parent", "zh-CN"),
            ("second parent", "ja"),
        }
        provider.releases[("second parent", "ja")].set()
        for _ in range(100):
            if len(osc.messages) == 3:
                break
            await asyncio.sleep(0)
        assert osc.messages[2].utterance_id == second_parent
        assert osc.messages[2].target_indexes == (1,)
        provider.releases[("second parent", "zh-CN")].set()
        await harness.translation_turns.wait_for_idle()
        parent_ids = [first_parent, second_parent]

        by_parent = {
            parent_id: [
                message.target_indexes
                for message in osc.messages
                if message.utterance_id == parent_id
            ]
            for parent_id in parent_ids
        }
        assert by_parent == {
            parent_ids[0]: [(1,), (0, 1)],
            parent_ids[1]: [(1,), (0, 1)],
        }
        assert harness.output_runtime.overlay_admission_snapshot() == {
            "active": 0,
            "unsent": 0,
            "batches": 0,
            "reserved_bytes": 0,
            "scopes": {},
        }
        assert connection.sent_payloads == []
    finally:
        for release in provider.releases.values():
            release.set()
        connection.release_send.set()
        await harness.translation_turns.close()
        await harness.output_runtime.close()
        await presenter.close()
        await bridge.stop()


@pytest.mark.asyncio
async def test_overlapping_dual_target_parent_projects_ready_surfaces_before_chatbox() -> None:
    provider = TargetControlledProvider()
    osc = RecordingOsc()
    bridge = OverlayBridge(session_token="dual-target-overlap-token")
    connection = StalledBridgeConnection()
    bridge._authenticated_connections.add(connection)
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        bridge=bridge,
    )
    harness = compose_translation_test_harness(
        stt=None,
        llm=provider,
        osc=osc,
        overlay_sink=presenter,
        source_language="en",
        target_language="zh-CN",
        self_target_languages=("zh-CN", "ja"),
    )
    observed_ui = []

    async def wait_for_second_primary_surface(parent_id: UUID) -> None:
        while True:
            while not harness.ui_events.empty():
                observed_ui.append(harness.ui_events.get_nowait())
            second_translation_visible = any(
                event.type == UIEventType.TRANSLATION_DONE and event.utterance_id == parent_id
                for event in observed_ui
            )
            block = next(
                (item for item in bridge.snapshot().blocks if item.id == f"self:{parent_id}"),
                None,
            )
            if (
                second_translation_visible
                and block is not None
                and block.secondary_text == "translated-zh-CN"
            ):
                return
            await asyncio.sleep(0)

    try:
        first_parent = await harness.self_owner.submit_text("first overlap")
        first_started = {
            await asyncio.wait_for(provider.started.get(), timeout=1) for _ in range(2)
        }
        assert first_started == {
            ("first overlap", "zh-CN"),
            ("first overlap", "ja"),
        }
        second_parent = await harness.self_owner.submit_text("second overlap")
        second_started = {
            await asyncio.wait_for(provider.started.get(), timeout=1) for _ in range(2)
        }
        assert second_started == {
            ("second overlap", "zh-CN"),
            ("second overlap", "ja"),
        }
        await connection.send_started.wait()
        provider.releases[("first overlap", "zh-CN")].set()

        provider.releases[("second overlap", "zh-CN")].set()
        await asyncio.wait_for(
            wait_for_second_primary_surface(second_parent),
            timeout=1,
        )

        assert not harness.translation_turns.is_parent_closed(first_parent)
        assert not harness.translation_turns.is_parent_closed(second_parent)
        admission = harness.output_runtime.overlay_admission_snapshot()
        chatbox_scope = admission["scopes"]["manual:chatbox"]
        assert chatbox_scope["active"] == 1
        assert chatbox_scope["unsent"] == 1
        assert 0 < chatbox_scope["reserved_bytes"] <= 2 * 1024 * 1024
        assert connection.sent_payloads == []

        provider.releases[("first overlap", "ja")].set()
        provider.releases[("second overlap", "ja")].set()
        await harness.translation_turns.wait_for_idle()
        assert harness.output_runtime.overlay_admission_snapshot()["reserved_bytes"] == 0
    finally:
        for release in provider.releases.values():
            release.set()
        connection.release_send.set()
        await harness.translation_turns.close()
        await harness.output_runtime.close()
        await presenter.close()
        await bridge.stop()


@pytest.mark.asyncio
async def test_newer_transcript_visibility_suppresses_older_primary_latest_surfaces() -> None:
    provider = TargetControlledProvider()
    osc = RecordingOsc()
    overlay = RecordingOverlay()
    harness = compose_translation_test_harness(
        stt=None,
        llm=provider,
        osc=osc,
        overlay_sink=overlay,
        source_language="en",
        target_language="zh-CN",
        self_target_languages=("zh-CN", "ja"),
    )

    try:
        older_parent_id = await harness.self_owner.submit_text("older source")
        await asyncio.wait_for(provider.started.get(), timeout=1)
        await asyncio.wait_for(provider.started.get(), timeout=1)
        newer_parent_id = await harness.self_owner.submit_text("newer source")
        await asyncio.wait_for(provider.started.get(), timeout=1)
        await asyncio.wait_for(provider.started.get(), timeout=1)

        provider.releases[("older source", "zh-CN")].set()
        for _ in range(100):
            older_bundle = harness.self_runtime.utterances.get(older_parent_id)
            if older_bundle is not None and older_bundle.translation is not None:
                break
            await asyncio.sleep(0)

        ui_events = []
        while not harness.ui_events.empty():
            ui_events.append(harness.ui_events.get_nowait())
        assert [
            event.utterance_id for event in ui_events if event.type == UIEventType.TRANSCRIPT_FINAL
        ] == [older_parent_id, newer_parent_id]
        assert not any(event.type == UIEventType.TRANSLATION_DONE for event in ui_events)
        assert not any(event.type == UIEventType.OSC_SENT for event in ui_events)
        assert osc.messages == []
        assert [event.type for event in overlay.events] == [
            "self_transcript_final",
            "self_transcript_final",
        ]
        assert [event.utterance_id for event in overlay.events] == [
            older_parent_id,
            newer_parent_id,
        ]
        assert harness.self_runtime.utterances[older_parent_id].translation is not None
    finally:
        for release in provider.releases.values():
            release.set()
        await harness.translation_turns.wait_for_idle()
        await harness.translation_turns.close()


@pytest.mark.asyncio
async def test_direct_self_translation_uses_dual_target_turn_lifecycle() -> None:
    provider = TargetControlledProvider()
    osc = RecordingOsc()
    harness = compose_translation_test_harness(
        stt=None,
        llm=provider,
        osc=osc,
        source_language="en",
        target_language="zh-CN",
        self_target_languages=("zh-CN", "ja"),
    )
    parent_id = uuid4()
    task = asyncio.create_task(harness.self_owner.translate_and_enqueue(parent_id, "source text"))

    try:
        started = {
            await asyncio.wait_for(provider.started.get(), timeout=1),
            await asyncio.wait_for(provider.started.get(), timeout=1),
        }
        assert started == {("source text", "zh-CN"), ("source text", "ja")}

        provider.releases[("source text", "ja")].set()
        for _ in range(100):
            if len(osc.messages) == 1:
                break
            await asyncio.sleep(0)
        assert [(message.utterance_id, message.text) for message in osc.messages] == [
            (parent_id, "translated-ja")
        ]

        provider.releases[("source text", "zh-CN")].set()
        await asyncio.wait_for(task, timeout=1)
        assert [(message.utterance_id, message.text) for message in osc.messages] == [
            (parent_id, "translated-ja"),
            (parent_id, "translated-zh-CN\ntranslated-ja"),
        ]
    finally:
        for release in provider.releases.values():
            release.set()
        await asyncio.gather(task, return_exceptions=True)
        await harness.translation_turns.close()
