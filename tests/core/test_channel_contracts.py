from __future__ import annotations

from uuid import uuid4

from puripuly_heart.domain.events import STTFinalEvent, UIEvent, UIEventType
from puripuly_heart.domain.models import Transcript, Translation


def test_transcript_and_translation_models_preserve_channel_metadata() -> None:
    transcript = Transcript(
        utterance_id=uuid4(),
        text="hello there",
        is_final=True,
        created_at=123.0,
        channel="peer",
    )
    translation = Translation(
        utterance_id=transcript.utterance_id,
        source_text=transcript.text,
        translated_text="안녕하세요",
        source_language="en",
        target_language="ko",
        channel="peer",
    )

    assert transcript.channel == "peer"
    assert translation.text == "안녕하세요"
    assert translation.translated_text == "안녕하세요"


def test_ui_and_stt_events_can_reference_self_or_peer_channels() -> None:
    transcript = Transcript(
        utterance_id=uuid4(),
        text="self text",
        is_final=True,
        created_at=456.0,
        channel="self",
    )
    stt_event = STTFinalEvent(utterance_id=transcript.utterance_id, transcript=transcript)
    ui_event = UIEvent(
        type=UIEventType.TRANSCRIPT_FINAL,
        utterance_id=transcript.utterance_id,
        payload=transcript,
        source="Peer",
        channel="peer",
    )

    assert stt_event.channel == "self"
    assert stt_event.transcript.channel == "self"
    assert ui_event.channel == "peer"


def test_ui_event_adopts_channel_from_payload_when_not_explicitly_provided() -> None:
    transcript = Transcript(
        utterance_id=uuid4(),
        text="peer text",
        is_final=True,
        created_at=789.0,
        channel="peer",
    )

    ui_event = UIEvent(
        type=UIEventType.TRANSCRIPT_FINAL,
        utterance_id=transcript.utterance_id,
        payload=transcript,
        source="Peer",
    )

    assert ui_event.channel == "peer"
