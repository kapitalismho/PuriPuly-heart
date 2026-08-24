from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

import pytest

from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.messages import SEVERITY_INFO, UserMessageRef
from puripuly_heart.core.osc.chatbox_paginator import (
    ChatboxPaginator,
    ChatboxPaginatorOutputAdapter,
)
from puripuly_heart.core.output.models import (
    PeerSubtitlePublication,
    SelfUtterancePublication,
    SystemDisclosurePublication,
)
from puripuly_heart.core.overlay.sink import (
    OverlayEventAdapter,
    SubtitleOverlayOutputAdapter,
)


@dataclass(slots=True)
class RecordingOscSender:
    chatbox_texts: list[str] = field(default_factory=list)
    typing_values: list[bool] = field(default_factory=list)

    def send_chatbox(self, text: str) -> None:
        self.chatbox_texts.append(text)

    def send_typing(self, is_typing: bool) -> None:
        self.typing_values.append(is_typing)


@dataclass(slots=True)
class RecordingOverlaySink:
    events: list[object] = field(default_factory=list)

    async def emit(self, event: object) -> None:
        self.events.append(event)


@pytest.mark.asyncio
async def test_chatbox_paginator_output_adapter_preserves_self_and_system_paths() -> None:
    sender = RecordingOscSender()
    paginator = ChatboxPaginator(sender=sender, clock=FakeClock(_now=123.0))
    adapter = ChatboxPaginatorOutputAdapter(
        paginator=paginator,
        include_source=False,
        render_system_disclosure=lambda publication: publication.message.key,
    )
    self_publication = SelfUtterancePublication(
        utterance_id=str(uuid4()),
        transcript_text="self source text",
        translation_text="self translated text",
        source_language="ko",
        target_language="en",
        is_final=True,
        metadata={},
    )
    system_publication = SystemDisclosurePublication(
        disclosure_id=str(uuid4()),
        message=UserMessageRef(key="runtime.disclosure", params={}, severity=SEVERITY_INFO),
        metadata={},
    )

    await adapter.publish_self_utterance(self_publication)
    await adapter.publish_system_disclosure(system_publication)

    assert sender.chatbox_texts == ["self translated text", "runtime.disclosure"]
    assert sender.typing_values == [False]
    assert not hasattr(adapter, "publish_peer_subtitle")
    assert "self source text" not in repr(system_publication)


@pytest.mark.asyncio
async def test_chatbox_adapter_propagates_turn_revision_metadata_to_paginator() -> None:
    sender = RecordingOscSender()
    clock = FakeClock(_now=123.0)
    paginator = ChatboxPaginator(sender=sender, clock=clock, max_chars=4)
    adapter = ChatboxPaginatorOutputAdapter(
        paginator=paginator,
        include_source=False,
        render_system_disclosure=lambda publication: publication.message.key,
    )
    parent_id = str(uuid4())

    await adapter.publish_self_utterance(
        SelfUtterancePublication(
            utterance_id=parent_id,
            transcript_text="source",
            translation_text="abcdefgh",
            source_language="ko",
            target_language="en",
            is_final=True,
            metadata={"turn_generation": 2, "turn_order": 3, "presentation_revision": 1},
        )
    )
    await adapter.publish_self_utterance(
        SelfUtterancePublication(
            utterance_id=parent_id,
            transcript_text="source",
            translation_text="12345678",
            source_language="ko",
            target_language="en",
            is_final=True,
            metadata={"turn_generation": 2, "turn_order": 3, "presentation_revision": 2},
        )
    )
    clock.advance(3.0)
    paginator.process_due()

    assert sender.chatbox_texts == ["abcd", "1234", "5678"]


@pytest.mark.asyncio
async def test_chatbox_adapter_rejects_partial_turn_revision_metadata() -> None:
    sender = RecordingOscSender()
    adapter = ChatboxPaginatorOutputAdapter(
        paginator=ChatboxPaginator(sender=sender, clock=FakeClock(_now=123.0)),
        include_source=False,
        render_system_disclosure=lambda publication: publication.message.key,
    )
    publication = SelfUtterancePublication(
        utterance_id=str(uuid4()),
        transcript_text="source",
        translation_text="translation",
        source_language="ko",
        target_language="en",
        is_final=True,
        metadata={"presentation_revision": 1},
    )

    with pytest.raises(ValueError, match="self turn identity"):
        await adapter.publish_self_utterance(publication)

    assert sender.chatbox_texts == []


@pytest.mark.asyncio
async def test_chatbox_paginator_output_adapter_redacts_system_disclosure_text() -> None:
    sender = RecordingOscSender()
    paginator = ChatboxPaginator(sender=sender, clock=FakeClock(_now=123.0))
    adapter = ChatboxPaginatorOutputAdapter(
        paginator=paginator,
        render_system_disclosure=lambda _publication: (
            "broker_raw_message=eligibility token=chatbox-adapter-secret"
        ),
    )
    system_publication = SystemDisclosurePublication(
        disclosure_id=str(uuid4()),
        message=UserMessageRef(key="runtime.disclosure", params={}, severity=SEVERITY_INFO),
        metadata={},
    )

    await adapter.publish_system_disclosure(system_publication)

    assert sender.chatbox_texts == ["[broker-raw-message-redacted]"]


@pytest.mark.asyncio
async def test_subtitle_overlay_output_adapter_emits_peer_translation_final_event() -> None:
    utterance_id = uuid4()
    sink = RecordingOverlaySink()
    adapter = SubtitleOverlayOutputAdapter(
        sink=sink,
        event_adapter=OverlayEventAdapter(clock=FakeClock(_now=456.0)),
    )
    publication = PeerSubtitlePublication(
        utterance_id=str(utterance_id),
        transcript_text="peer source text",
        translation_text="peer translated text",
        source_language="ja",
        target_language="en",
        is_final=True,
        metadata={},
    )

    await adapter.publish_peer_subtitle(publication)

    assert len(sink.events) == 1
    event = sink.events[0]
    assert getattr(event, "type") == "translation_final"
    assert event.utterance_id == utterance_id
    assert event.channel == "peer"
    assert event.text == "peer translated text"
    assert event.source_text == "peer source text"
    assert event.source_language == "ja"
    assert event.target_language == "en"


@pytest.mark.asyncio
async def test_subtitle_overlay_output_adapter_preserves_peer_translation_protocol_metadata() -> (
    None
):
    utterance_id = uuid4()
    sink = RecordingOverlaySink()
    adapter = SubtitleOverlayOutputAdapter(
        sink=sink,
        event_adapter=OverlayEventAdapter(clock=FakeClock(_now=456.0)),
    )
    publication = PeerSubtitlePublication(
        utterance_id=str(utterance_id),
        transcript_text="peer source text",
        translation_text="peer translated text",
        source_language="ja",
        target_language="en",
        is_final=True,
        metadata={
            "created_at": 321.5,
            "update_id": "review-update-id",
            "origin_wall_clock_ms": 1712345678901,
            "session_scope": "session-42",
            "source_text_hash": "src-hash-42",
            "source_text_len": 17,
            "logical_turn_key": "peer:turn-42",
        },
    )

    await adapter.publish_peer_subtitle(publication)

    event = sink.events[0]
    assert getattr(event, "type") == "translation_final"
    assert event.created_at == 321.5
    assert event.update_id == "review-update-id"
    assert event.origin_wall_clock_ms == 1712345678901
    assert event.session_scope == "session-42"
    assert event.source_text_hash == "src-hash-42"
    assert event.source_text_len == 17
    assert event.logical_turn_key == "peer:turn-42"


@pytest.mark.asyncio
async def test_subtitle_overlay_output_adapter_emits_peer_source_only_event() -> None:
    utterance_id = uuid4()
    sink = RecordingOverlaySink()
    adapter = SubtitleOverlayOutputAdapter(
        sink=sink,
        event_adapter=OverlayEventAdapter(clock=FakeClock(_now=789.0)),
    )
    publication = PeerSubtitlePublication(
        utterance_id=str(utterance_id),
        transcript_text="peer source only",
        translation_text=None,
        source_language="ja",
        target_language="en",
        is_final=True,
        metadata={
            "created_at": 654.5,
            "update_id": "source-only-update",
            "origin_wall_clock_ms": 1712345678902,
            "session_scope": "session-source",
            "source_text_hash": "src-hash-source",
            "source_text_len": 16,
            "logical_turn_key": "peer:source-turn",
        },
    )

    await adapter.publish_peer_subtitle(publication)

    assert len(sink.events) == 1
    event = sink.events[0]
    assert getattr(event, "type") == "peer_transcript_final"
    assert event.utterance_id == utterance_id
    assert event.channel == "peer"
    assert event.text == "peer source only"
    assert event.source_language == "ja"
    assert event.target_language == "en"
    assert event.created_at == 654.5
    assert event.update_id == "source-only-update"
    assert event.origin_wall_clock_ms == 1712345678902
    assert event.session_scope == "session-source"
    assert event.source_text_hash == "src-hash-source"
    assert event.source_text_len == 16
    assert event.logical_turn_key == "peer:source-turn"
