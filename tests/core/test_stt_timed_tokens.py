from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from puripuly_heart.core.audio.format import AudioCaptureSpan
from puripuly_heart.core.audio.ownership import (
    AudioSegmentIdentity,
    AudioSegmentSettingsSnapshot,
)
from puripuly_heart.core.stt.backend import (
    LEGACY_STT_SESSION_PROJECTION,
    STTNativeProvenance,
    STTProviderTurnIdentity,
    STTProviderTurnRequest,
    STTProviderTurnTerminal,
    STTSessionProjection,
    STTTimedToken,
)
from puripuly_heart.core.stt.scoped_normalizer import STTScopedTurnNormalizer
from puripuly_heart.domain.models import FinalLanguageRun
from puripuly_heart.providers.stt.deepgram import _DeepgramSDKSession, _ScopedWord


def _identity() -> STTProviderTurnIdentity:
    return STTProviderTurnIdentity(
        segment=AudioSegmentIdentity(
            activation_generation=1,
            segment_order=1,
            segment_id=uuid4(),
            capture_epoch=1,
        ),
        provider_epoch_id="epoch",
        provider_turn_id="turn",
    )


def _session() -> _DeepgramSDKSession:
    return _DeepgramSDKSession(
        api_key="k",
        model="nova-3",
        language="en",
        sample_rate_hz=16000,
        connect_timeout_s=5.0,
        keyterms=[],
        drain_timeout_s=0.05,
        projection=STTSessionProjection(mode="scoped", provider_epoch_id="epoch"),
    )


def _span(start: int, end: int, epoch: int = 1) -> AudioCaptureSpan:
    return AudioCaptureSpan(
        capture_epoch=epoch,
        callback_sequence=1,
        source_sample_rate_hz=16000,
        source_start_sample=start,
        source_end_sample=end,
        source_start_monotonic_s=start / 16000,
        source_end_monotonic_s=end / 16000,
        normalized_sample_rate_hz=16000,
        normalized_start_sample=start,
        normalized_end_sample=end,
    )


def test_normalizer_strip_preserves_punctuation_language_and_times() -> None:
    identity = _identity()
    normalizer = STTScopedTurnNormalizer(identity)
    tokens = (
        STTTimedToken(
            " ",
            language="en",
            start_ms=0,
            end_ms=10,
            timing="interval",
            source_start_sample=0,
            source_end_sample=160,
        ),
        STTTimedToken(
            "Hi",
            language="en",
            start_ms=10,
            end_ms=20,
            timing="interval",
            source_start_sample=160,
            source_end_sample=320,
        ),
        STTTimedToken(
            "!",
            language="en",
            start_ms=20,
            end_ms=30,
            timing="interval",
            source_start_sample=320,
            source_end_sample=480,
        ),
        STTTimedToken(
            " ",
            language="ja",
            start_ms=30,
            end_ms=40,
            timing="interval",
            source_start_sample=480,
            source_end_sample=640,
        ),
    )
    terminal = normalizer.apply_terminal(
        STTProviderTurnTerminal(
            identity=identity,
            outcome="final",
            text=" Hi! ",
            final_language_runs=(
                FinalLanguageRun(" Hi!", "en"),
                FinalLanguageRun(" ", "ja"),
            ),
            timed_tokens=tokens,
        )
    )
    assert terminal.text == "Hi!"
    assert "".join(token.text for token in terminal.timed_tokens) == "Hi!"
    assert [token.language for token in terminal.timed_tokens] == ["en", "en"]
    assert terminal.timed_tokens[1].text == "!"
    assert terminal.timed_tokens[0].start_ms == 10
    assert terminal.timed_tokens[1].end_ms == 30


def test_equal_ends_are_valid_decreasing_is_invalid() -> None:
    session = _session()
    session._record_origin(b"\x00\x00" * 16000, (_span(0, 16000),), False)
    session._scoped_words = [
        _ScopedWord(text="a", start_s=0.0, end_s=0.1, language="en"),
        _ScopedWord(text="b", start_s=0.05, end_s=0.1, language="en"),
        _ScopedWord(text="c", start_s=0.06, end_s=0.08, language="en"),
    ]
    timed = session._timed_tokens_from_scoped()
    assert timed[0].timing == "interval"
    assert timed[1].timing == "interval"
    assert timed[2].timing == "invalid"


def test_origin_mapping_keeps_context_only_on_session_clock() -> None:
    session = _session()
    context = b"\x00\x00" * 1600
    content = b"\x00\x00" * 1600
    session._record_origin(context, (_span(0, 1600),), True)
    session._record_origin(content, (_span(1600, 3200),), False)
    assert session._session_ms == 200
    mapped_context = session._map_session_ms(50)
    mapped_content = session._map_session_ms(150)
    assert mapped_context == 800
    assert mapped_content == 2400


def test_missing_start_is_end_only_not_chained() -> None:
    session = _session()
    session._record_origin(b"\x00\x00" * 16000, (_span(0, 16000),), False)
    session._scoped_words = [
        _ScopedWord(text="a", start_s=None, end_s=0.1, language="en"),
        _ScopedWord(text="b", start_s=None, end_s=0.2, language="en"),
    ]
    timed = session._timed_tokens_from_scoped()
    assert timed[0].timing == "end_only"
    assert timed[0].start_ms is None
    assert timed[1].start_ms is None
    assert timed[1].source_start_sample is None


def test_other_providers_omit_timing_by_default() -> None:
    identity = _identity()
    terminal = STTProviderTurnTerminal(identity=identity, outcome="final", text="hello")
    assert terminal.timed_tokens == ()
    normalized = STTScopedTurnNormalizer(identity).apply_terminal(terminal)
    assert normalized.timed_tokens == ()
    _ = LEGACY_STT_SESSION_PROJECTION
    _ = AudioSegmentSettingsSnapshot
    _ = STTNativeProvenance


def _deepgram_result(text: str, *, words=(), from_finalize: bool = False, is_final: bool = True):
    from deepgram.extensions.types.sockets import ListenV1ResultsEvent
    from deepgram.extensions.types.sockets.listen_v1_results_event import (
        ListenV1Alternative,
        ListenV1Channel,
        ListenV1ModelInfo,
        ListenV1ResultsMetadata,
        ListenV1Word,
    )

    return ListenV1ResultsEvent(
        type="Results",
        channel_index=[0, 1],
        duration=0.2,
        start=0.0,
        is_final=is_final,
        speech_final=False,
        channel=ListenV1Channel(
            alternatives=[
                ListenV1Alternative(
                    transcript=text,
                    confidence=1.0,
                    words=[
                        ListenV1Word(
                            word=item["word"],
                            start=item["start"],
                            end=item["end"],
                            confidence=1.0,
                            language=item.get("language", "en"),
                            punctuated_word=item.get("punctuated_word", item["word"]),
                        )
                        for item in words
                    ],
                )
            ]
        ),
        metadata=ListenV1ResultsMetadata(
            request_id="session-request",
            model_info=ListenV1ModelInfo(name="nova-3", version="1", arch="nova"),
            model_uuid="model",
        ),
        from_finalize=from_finalize,
    )


@pytest.mark.asyncio
async def test_deepgram_scoped_terminal_preserves_word_times() -> None:
    session = _session()
    session._loop = asyncio.get_running_loop()

    async def write(_payload):
        return None

    session._write_thread_payload = write
    identity = _identity()
    await session.begin_turn(
        STTProviderTurnRequest(
            identity=identity,
            settings=AudioSegmentSettingsSnapshot(
                provider_id="deepgram",
                provider_signature=("deepgram",),
                runtime_signature=("deepgram",),
                source_mode="desktop",
                source_language="en",
                expected_languages=("en",),
                target_sample_rate_hz=16000,
                vad_speech_threshold=0.4,
                vad_hangover_ms=800,
                vad_pre_roll_ms=500,
            ),
        )
    )
    pcm = b"\x00\x00" * 3200
    await session.send_turn_audio(
        identity,
        pcm,
        payload_sequence=1,
        source_ranges=(_span(0, 3200),),
        context_only=False,
    )
    session._build_transcript_event(
        _deepgram_result(
            "Hello there",
            words=(
                {
                    "word": "Hello",
                    "punctuated_word": "Hello ",
                    "start": 0.0,
                    "end": 0.1,
                },
                {
                    "word": "there",
                    "punctuated_word": "there",
                    "start": 0.1,
                    "end": 0.2,
                },
            ),
        )
    )
    await session.seal_turn(
        identity,
        sealed_content_ranges=(_span(0, 3200),),
        seal_reason="vad",
        observed_trailing_silence_ms=0,
    )
    session._build_transcript_event(_deepgram_result("", from_finalize=True))
    await asyncio.sleep(0)
    terminal = None
    async for event in session.turn_events():
        if isinstance(event, STTProviderTurnTerminal):
            terminal = event
            break
    assert terminal is not None
    assert terminal.text == "Hello there"
    assert len(terminal.timed_tokens) == 2
    assert "".join(token.text for token in terminal.timed_tokens) == "Hello there"
    assert terminal.timed_tokens[0].start_ms == 0
    assert terminal.timed_tokens[0].end_ms == 100
    assert terminal.timed_tokens[1].start_ms == 100
    assert terminal.timed_tokens[0].source_start_sample == 0
    assert terminal.timed_tokens[0].source_end_sample == 1600
    assert terminal.timed_tokens[1].source_start_sample == 1600
