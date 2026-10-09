from __future__ import annotations

import asyncio
import json
import logging
import sys
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import pytest
from websockets.asyncio.server import serve
from websockets.exceptions import InvalidStatus, ProtocolError

from puripuly_heart.core.audio.format import AudioCaptureSpan
from puripuly_heart.core.audio.ownership import AudioSegmentIdentity, AudioSegmentSettingsSnapshot
from puripuly_heart.core.stt.backend import (
    PermanentSTTScopedSessionError,
    STTBackendTranscriptEvent,
    STTProviderEpochEnded,
    STTProviderTurnIdentity,
    STTProviderTurnRequest,
    STTProviderTurnTerminal,
    STTSessionProjection,
)
from puripuly_heart.providers.stt import soniox as soniox_module
from puripuly_heart.providers.stt.soniox import (
    _STOP,
    SonioxRealtimeSTTBackend,
    _FinalizeRequest,
    _SonioxSession,
)


def _make_session(
    *,
    context_terms: list[str] | None = None,
    enable_language_identification: bool = False,
    projection: STTSessionProjection | None = None,
) -> _SonioxSession:
    return _SonioxSession(
        api_key="k",
        model="m",
        endpoint="wss://example",
        sample_rate_hz=16000,
        language_hints=["en"],
        context_terms=context_terms or [],
        keepalive_interval_s=10.0,
        trailing_silence_ms=100,
        connect_timeout_s=5.0,
        enable_language_identification=enable_language_identification,
        projection=projection or STTSessionProjection(),
    )


def _scoped_request(*, channel: str = "peer") -> STTProviderTurnRequest:
    settings = AudioSegmentSettingsSnapshot(
        provider_id="soniox",
        provider_signature=("soniox",),
        runtime_signature=("soniox",),
        source_mode="desktop",
        source_language="en",
        expected_languages=("en",),
        target_sample_rate_hz=16000,
        vad_speech_threshold=0.4,
        vad_hangover_ms=800,
        vad_pre_roll_ms=500,
    )
    return STTProviderTurnRequest(
        identity=STTProviderTurnIdentity(
            segment=AudioSegmentIdentity(
                activation_generation=1,
                segment_order=1,
                segment_id=uuid4(),
                capture_epoch=1,
            ),
            provider_epoch_id="epoch-1",
            provider_turn_id="turn-1",
        ),
        settings=settings,
        channel=channel,
    )


def _content_span(duration_ms: int) -> tuple[AudioCaptureSpan, ...]:
    sample_count = 16000 * duration_ms // 1000
    return (
        AudioCaptureSpan(
            capture_epoch=1,
            callback_sequence=1,
            source_sample_rate_hz=16000,
            source_start_sample=1000,
            source_end_sample=1000 + sample_count,
            source_start_monotonic_s=10.0,
            source_end_monotonic_s=10.0 + duration_ms / 1000,
            normalized_sample_rate_hz=16000,
            normalized_start_sample=2000,
            normalized_end_sample=2000 + sample_count,
        ),
    )


@pytest.mark.asyncio
async def test_soniox_backend_open_cancellation_closes_started_session(monkeypatch) -> None:
    started = asyncio.Event()
    closed = asyncio.Event()
    sessions = []

    class PartialSession:
        def __init__(self, **_kwargs) -> None:
            self.websocket_open = True
            sessions.append(self)

        async def start(self) -> None:
            started.set()
            await asyncio.Event().wait()

        async def close(self) -> None:
            self.websocket_open = False
            closed.set()

    monkeypatch.setattr(soniox_module, "_SonioxSession", PartialSession)
    backend = SonioxRealtimeSTTBackend(api_key="k", language_hints=["en"])
    open_task = asyncio.create_task(backend.open_session())
    await started.wait()

    open_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await open_task

    assert closed.is_set()
    assert sessions and sessions[0].websocket_open is False


async def _request_finalize(
    session: _SonioxSession,
    *,
    observed_tail_ms: int = 500,
) -> _FinalizeRequest:
    await session.on_speech_end(trailing_silence_ms=observed_tail_ms)
    finalize = await session._audio_q.get()
    assert isinstance(finalize, _FinalizeRequest)
    return finalize


@pytest.mark.asyncio
async def test_soniox_backend_validates_params() -> None:
    backend = SonioxRealtimeSTTBackend(api_key="", language_hints=["en"])
    with pytest.raises(PermanentSTTScopedSessionError, match="api_key"):
        await backend.open_session()

    backend = SonioxRealtimeSTTBackend(api_key="k", language_hints=["en"], endpoint="")
    with pytest.raises(PermanentSTTScopedSessionError, match="endpoint"):
        await backend.open_session()

    backend = SonioxRealtimeSTTBackend(
        api_key="k",
        language_hints=["en"],
        keepalive_interval_s=0.0,
    )
    with pytest.raises(PermanentSTTScopedSessionError, match="keepalive_interval_s"):
        await backend.open_session()

    backend = SonioxRealtimeSTTBackend(
        api_key="k",
        language_hints=[],
        language_hints_strict=True,
    )
    with pytest.raises(PermanentSTTScopedSessionError, match="language_hints_strict"):
        await backend.open_session()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "permanent"),
    [
        (400, True),
        (401, True),
        (403, True),
        (418, True),
        (True, True),
        (408, False),
        (413, False),
        (429, False),
        (500, False),
        (503, False),
    ],
)
async def test_soniox_rejected_open_only_retries_known_server_statuses(
    monkeypatch, status: int, permanent: bool
) -> None:
    async def rejected_connect(*_args, **_kwargs):
        raise InvalidStatus(SimpleNamespace(status_code=status))

    monkeypatch.setattr("puripuly_heart.core.network_clients.external_websocket_connect", rejected_connect)
    backend = SonioxRealtimeSTTBackend(api_key="k", language_hints=["en"])
    error_type = PermanentSTTScopedSessionError if permanent else InvalidStatus
    with pytest.raises(error_type):
        await backend.open_session(
            projection=STTSessionProjection(mode="scoped", provider_epoch_id="e")
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "permanent"),
    [(ProtocolError("invalid protocol"), True), (ConnectionResetError("reset"), False)],
)
async def test_soniox_open_config_send_distinguishes_protocol_from_transport(
    monkeypatch, failure: Exception, permanent: bool
) -> None:
    class WebSocket:
        async def send(self, _payload: object) -> None:
            raise failure

        async def close(self) -> None:
            pass

    async def connect(*_args, **_kwargs):
        return WebSocket()

    monkeypatch.setattr("puripuly_heart.core.network_clients.external_websocket_connect", connect)
    backend = SonioxRealtimeSTTBackend(api_key="k", language_hints=["en"])
    error_type = PermanentSTTScopedSessionError if permanent else ConnectionResetError
    with pytest.raises(error_type):
        await backend.open_session()


def test_soniox_backend_defaults_to_realtime_v5_model() -> None:
    backend = SonioxRealtimeSTTBackend(api_key="k", language_hints=["en"])

    assert backend.model == "stt-rt-v5"


@pytest.mark.asyncio
async def test_soniox_session_handles_message_errors() -> None:
    session = _make_session()

    session._handle_message("not-json")
    assert session._event_projection._legacy_events.empty()

    session._handle_message(json.dumps({"error": "bad"}))
    event = session._event_projection._legacy_events.get_nowait()
    assert isinstance(event, RuntimeError)


@pytest.mark.asyncio
async def test_soniox_session_collects_final_tokens() -> None:
    session = _make_session()
    await _request_finalize(session)

    message = {
        "tokens": [
            {"text": "Hello", "is_final": True, "end_ms": 100},
            {"text": " ", "is_final": True, "end_ms": 110},
            {"text": "world", "is_final": True, "end_ms": 120},
            {"text": "<fin>", "is_final": True},
        ]
    }
    session._handle_message(json.dumps(message))
    event = session._event_projection._legacy_events.get_nowait()

    assert isinstance(event, STTBackendTranscriptEvent)
    assert event.text == "Hello world"
    assert event.final_language_runs == ()


@pytest.mark.asyncio
async def test_soniox_session_emits_ordered_adjacent_final_language_runs() -> None:
    session = _make_session(enable_language_identification=True)
    await _request_finalize(session)

    session._handle_message(
        json.dumps(
            {
                "tokens": [
                    {"text": "안", "language": "ko", "is_final": True, "end_ms": 100},
                    {"text": "녕", "language": "ko", "is_final": True, "end_ms": 110},
                    {"text": "こんにちは", "language": "ja", "is_final": True, "end_ms": 120},
                    {"text": "你", "language": "zh", "is_final": True, "end_ms": 130},
                    {"text": "好", "language": "zh", "is_final": True, "end_ms": 140},
                    {"text": "世界", "language": "ja", "is_final": True, "end_ms": 150},
                    {"text": "<fin>", "is_final": True},
                ]
            }
        )
    )

    event = session._event_projection._legacy_events.get_nowait()
    assert event.text == "안녕こんにちは你好世界"
    assert [(run.text, run.language) for run in event.final_language_runs] == [
        ("안녕", "ko"),
        ("こんにちは", "ja"),
        ("你好", "zh"),
        ("世界", "ja"),
    ]


@pytest.mark.asyncio
async def test_soniox_preserves_present_and_missing_speakers_in_session_scope() -> None:
    first = _make_session(enable_language_identification=True)
    second = _make_session(enable_language_identification=True)
    await _request_finalize(first)
    first._handle_message(
        json.dumps(
            {
                "tokens": [
                    {
                        "text": "one ",
                        "language": "en",
                        "speaker": "1",
                        "confidence": 0.01,
                        "is_final": True,
                    },
                    {
                        "text": "unknown ",
                        "language": "en",
                        "speaker": False,
                        "is_final": True,
                    },
                    {
                        "text": "again",
                        "language": "en",
                        "speaker": "1",
                        "confidence": 0.99,
                        "is_final": True,
                    },
                    {"text": "<fin>", "is_final": True},
                ]
            }
        )
    )

    event = first._event_projection._legacy_events.get_nowait()
    assert [(run.text, run.speaker_id) for run in event.final_speaker_runs] == [
        ("one ", "1"),
        ("unknown ", None),
        ("again", "1"),
    ]
    assert {run.session_scope for run in event.final_speaker_runs} == {first.speaker_session_scope}
    assert [run.attribution.state for run in event.final_speaker_runs] == [
        "identified",
        "malformed",
        "identified",
    ]
    assert (
        event.final_speaker_runs[0].attribution.key == event.final_speaker_runs[2].attribution.key
    )
    assert event.final_speaker_runs[1].attribution.key is None
    await _request_finalize(first)
    first._handle_message(
        json.dumps(
            {
                "tokens": [
                    {"text": "next", "speaker": "1", "is_final": True},
                    {"text": "<fin>", "is_final": True},
                ]
            }
        )
    )
    next_event = first._event_projection._legacy_events.get_nowait()
    assert {run.session_scope for run in next_event.final_speaker_runs} == {
        first.speaker_session_scope
    }
    assert second.speaker_session_scope != first.speaker_session_scope


@pytest.mark.asyncio
async def test_soniox_preserves_missing_and_malformed_attribution_as_distinct_text_runs() -> None:
    session = _make_session(enable_language_identification=True)
    await _request_finalize(session)
    session._handle_message(
        json.dumps(
            {
                "tokens": [
                    {"text": "before ", "is_final": True},
                    {"text": "invalid ", "speaker": False, "is_final": True},
                    {"text": "after", "is_final": True},
                    {"text": "<fin>", "is_final": True},
                ]
            }
        )
    )
    terminal = session._event_projection._legacy_events.get_nowait()
    assert terminal.text == "before invalid after"
    assert [(run.text, run.attribution.state) for run in terminal.final_speaker_runs] == [
        ("before ", "missing"),
        ("invalid ", "malformed"),
        ("after", "missing"),
    ]


def test_soniox_preserves_timing_and_overlapping_speaker_runs() -> None:
    session = _make_session(enable_language_identification=True)
    tokens = [
        soniox_module._FinalToken("a ", 100, 250, "en", "A", "identified"),
        soniox_module._FinalToken("b ", 200, 350, "en", "B", "identified"),
        soniox_module._FinalToken("b2", 350, 450, "en", "B", "identified"),
    ]

    runs = session._speaker_runs_for_tokens(tokens)

    assert [
        (run.text, run.speaker_id, run.source_start_ms, run.source_end_ms, run.overlaps_previous)
        for run in runs
    ] == [
        ("a ", "A", 100, 250, False),
        ("b b2", "B", 200, 450, True),
    ]
    assert [run.attribution.state for run in runs] == ["identified", "identified"]


@pytest.mark.asyncio
async def test_soniox_terminal_cleanup_keeps_final_runs_equal_to_emitted_text() -> None:
    session = _make_session(enable_language_identification=True)
    await _request_finalize(session)

    session._handle_message(
        json.dumps(
            {
                "tokens": [
                    {"text": ". ", "language": "ja", "is_final": True, "end_ms": 100},
                    {"text": "あ", "language": "ja", "is_final": True, "end_ms": 110},
                    {"text": "你", "language": "zh", "is_final": True, "end_ms": 120},
                    {"text": "<fin>", "is_final": True},
                ]
            }
        )
    )

    event = session._event_projection._legacy_events.get_nowait()
    assert event.text == ". あ你"
    assert [(token.text, token.language) for token in session._final_tokens] == [
        (". ", "ja"),
        ("あ", "ja"),
        ("你", "zh"),
    ]
    assert [(run.text, run.language) for run in event.final_language_runs] == [
        (". あ", "ja"),
        ("你", "zh"),
    ]
    assert "".join(run.text for run in event.final_language_runs) == event.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fixture_name", "tokens", "expected_runs"),
    [
        (
            "korean-only",
            [("가", "ko", 100), ("나", "ko", 110)],
            [("가나", "ko")],
        ),
        (
            "japanese-only",
            [("あ", "ja", 100), ("い", "ja", 110)],
            [("あい", "ja")],
        ),
        (
            "generic-chinese",
            [("你", "zh", 100), ("好", "zh", 110)],
            [("你好", "zh")],
        ),
        (
            "japanese-to-chinese",
            [("あ", "ja", 100), ("你", "zh", 110), ("好", "zh", 120)],
            [("あ", "ja"), ("你好", "zh")],
        ),
        (
            "chinese-to-japanese-to-korean",
            [("你", "zh", 100), ("あ", "ja", 110), ("가", "ko", 120)],
            [("你", "zh"), ("あ", "ja"), ("가", "ko")],
        ),
    ],
)
async def test_soniox_controlled_final_token_fixtures_preserve_each_token_and_adjacent_runs(
    fixture_name: str,
    tokens: list[tuple[str, str, int]],
    expected_runs: list[tuple[str, str]],
) -> None:
    session = _make_session(enable_language_identification=True)
    fixture_tokens = [
        {"text": text, "language": language, "is_final": True, "end_ms": end_ms}
        for text, language, end_ms in tokens
    ]
    fixture_tokens.append({"text": "<fin>", "is_final": True})

    await _request_finalize(session)
    session._handle_message(json.dumps({"tokens": fixture_tokens}))

    event = session._event_projection._legacy_events.get_nowait()
    assert fixture_name
    assert [(token.text, token.language, token.end_ms) for token in session._final_tokens] == tokens
    assert event.text == "".join(text for text, _, _ in tokens)
    assert [(run.text, run.language) for run in event.final_language_runs] == expected_runs


@pytest.mark.asyncio
async def test_soniox_controlled_finalize_boundaries_remain_independent_and_append_only() -> None:
    merged = _make_session(enable_language_identification=True)
    await _request_finalize(merged)
    merged._handle_message(
        json.dumps(
            {
                "tokens": [
                    {"text": "あ", "language": "ja", "is_final": True, "end_ms": 100},
                    {"text": "<fin>", "is_final": True},
                ]
            }
        )
    )
    first_event = merged._event_projection._legacy_events.get_nowait()
    await _request_finalize(merged)
    merged._handle_message(
        json.dumps(
            {
                "tokens": [
                    {"text": "你", "language": "zh", "is_final": True, "end_ms": 200},
                    {"text": "<fin>", "is_final": True},
                ]
            }
        )
    )
    merged_event = merged._event_projection._legacy_events.get_nowait()

    assert first_event.text == "あ"
    assert [(token.text, token.language, token.end_ms) for token in merged._final_tokens] == [
        ("你", "zh", 200),
    ]
    assert [(run.text, run.language) for run in merged_event.final_language_runs] == [
        ("你", "zh"),
    ]

    replaced = _make_session(enable_language_identification=True)
    await _request_finalize(replaced)
    replaced._handle_message(
        json.dumps(
            {
                "tokens": [
                    {"text": "你", "language": "zh", "is_final": True, "end_ms": 100},
                    {"text": "旧", "language": "ja", "is_final": True, "end_ms": 200},
                    {"text": "旧", "language": "ko", "is_final": True, "end_ms": 300},
                    {"text": "<fin>", "is_final": True},
                ]
            }
        )
    )
    original_event = replaced._event_projection._legacy_events.get_nowait()
    await _request_finalize(replaced)
    replaced._handle_message(
        json.dumps(
            {
                "tokens": [
                    {"text": "あ", "language": "ja", "is_final": True, "end_ms": 200},
                    {"text": "가", "language": "ko", "is_final": True, "end_ms": 300},
                    {"text": "<fin>", "is_final": True},
                ]
            }
        )
    )
    replaced_event = replaced._event_projection._legacy_events.get_nowait()

    assert original_event.text == "你旧旧"
    assert [(token.text, token.language, token.end_ms) for token in replaced._final_tokens] == [
        ("あ", "ja", 200),
        ("가", "ko", 300),
    ]
    assert replaced_event.text == "あ가"
    assert [(run.text, run.language) for run in replaced_event.final_language_runs] == [
        ("あ", "ja"),
        ("가", "ko"),
    ]


@pytest.mark.asyncio
async def test_soniox_session_retains_unknown_detected_language_for_safe_fallback() -> None:
    session = _make_session(enable_language_identification=True)
    await _request_finalize(session)

    session._handle_message(
        json.dumps(
            {
                "tokens": [
                    {"text": "bonjour", "language": "xx", "is_final": True, "end_ms": 100},
                    {"text": "<fin>", "is_final": True},
                ]
            }
        )
    )

    event = session._event_projection._legacy_events.get_nowait()
    assert [(run.text, run.language) for run in event.final_language_runs] == [("bonjour", "xx")]


@pytest.mark.asyncio
async def test_soniox_session_appends_final_tokens_across_messages_in_receive_order() -> None:
    session = _make_session()
    await _request_finalize(session)

    session._handle_message(
        json.dumps(
            {
                "tokens": [
                    {"text": "Hello", "is_final": True, "end_ms": 100},
                    {"text": " world", "is_final": True, "end_ms": 200},
                ]
            }
        )
    )
    assert session._event_projection._legacy_events.empty()

    session._handle_message(
        json.dumps(
            {
                "tokens": [
                    {"text": ". ", "is_final": True, "end_ms": 150},
                    {"text": "world", "is_final": True, "end_ms": 200},
                    {"text": "!", "is_final": True, "end_ms": 260},
                    {"text": "<fin>", "is_final": True},
                ]
            }
        )
    )
    event = session._event_projection._legacy_events.get_nowait()
    assert isinstance(event, STTBackendTranscriptEvent)
    assert event.text == "Hello world. world!"


@pytest.mark.asyncio
async def test_soniox_session_preserves_equal_and_regressing_timestamp_tokens() -> None:
    session = _make_session()
    await _request_finalize(session)

    session._handle_message(
        json.dumps(
            {
                "tokens": [
                    {"text": "A", "is_final": True, "end_ms": 100},
                    {"text": "B", "is_final": True, "end_ms": 100},
                ]
            }
        )
    )
    session._handle_message(
        json.dumps(
            {
                "tokens": [
                    {"text": "C", "is_final": True, "end_ms": 90},
                    {"text": "<fin>", "is_final": True},
                ]
            }
        )
    )
    event = session._event_projection._legacy_events.get_nowait()

    assert isinstance(event, STTBackendTranscriptEvent)
    assert event.text == "ABC"


@pytest.mark.asyncio
async def test_soniox_session_on_speech_end_enqueues_finalize(caplog) -> None:
    session = _make_session()

    with caplog.at_level(logging.INFO):
        await session.on_speech_end(trailing_silence_ms=240, reason="soft_pause")

    finalize = await session._audio_q.get()
    assert isinstance(finalize, _FinalizeRequest)
    assert session._audio_q.empty()
    assert caplog.text == ""

    caplog.clear()
    with caplog.at_level(logging.INFO):
        await session.on_speech_end()

    finalize = await session._audio_q.get()
    assert isinstance(finalize, _FinalizeRequest)
    assert session._audio_q.empty()
    assert caplog.text == ""


@pytest.mark.asyncio
async def test_soniox_session_repeated_finalize_boundaries_clear_each_final_segment() -> None:
    session = _make_session()

    await session.on_speech_end(trailing_silence_ms=0)
    await session.on_speech_end(trailing_silence_ms=0)

    first_finalize = await session._audio_q.get()
    second_finalize = await session._audio_q.get()
    assert isinstance(first_finalize, _FinalizeRequest)
    assert isinstance(second_finalize, _FinalizeRequest)

    session._handle_message(
        json.dumps(
            {
                "tokens": [
                    {"text": "First", "is_final": True, "end_ms": 100},
                    {"text": "<fin>", "is_final": True},
                ]
            }
        )
    )
    session._handle_message(
        json.dumps(
            {
                "tokens": [
                    {"text": "Second", "is_final": True, "end_ms": 200},
                    {"text": "<fin>", "is_final": True},
                ]
            }
        )
    )

    await session.on_speech_end(trailing_silence_ms=0)
    third_finalize = await session._audio_q.get()
    assert isinstance(third_finalize, _FinalizeRequest)
    session._handle_message(
        json.dumps(
            {
                "tokens": [
                    {"text": "Third", "is_final": True, "end_ms": 300},
                    {"text": "<fin>", "is_final": True},
                ]
            }
        )
    )

    events = [session._event_projection._legacy_events.get_nowait() for _ in range(3)]
    assert [event.text for event in events] == ["First", "Second", "Third"]


@pytest.mark.asyncio
async def test_soniox_session_duplicate_finalize_marker_emits_one_terminal() -> None:
    session = _make_session()
    await _request_finalize(session)

    session._handle_message(
        json.dumps(
            {
                "tokens": [
                    {"text": "Only", "is_final": True, "end_ms": 100},
                    {"text": "<fin>", "is_final": True},
                    {"text": "<fin>", "is_final": True},
                ]
            }
        )
    )

    event = session._event_projection._legacy_events.get_nowait()
    assert event.text == "Only"
    assert session._event_projection._legacy_events.empty()


@pytest.mark.asyncio
async def test_soniox_unmatched_finalize_marker_retains_tokens_for_next_request() -> None:
    session = _make_session()

    session._handle_message(
        json.dumps(
            {
                "tokens": [
                    {"text": "Kept", "is_final": True, "end_ms": 100},
                    {"text": "<fin>", "is_final": True},
                ]
            }
        )
    )
    assert session._event_projection._legacy_events.empty()
    assert [token.text for token in session._pending_tokens] == ["Kept"]

    await _request_finalize(session)
    session._handle_message(
        json.dumps(
            {
                "tokens": [
                    {"text": "More", "is_final": True, "end_ms": 200},
                    {"text": "<fin>", "is_final": True},
                ]
            }
        )
    )
    event = session._event_projection._legacy_events.get_nowait()
    assert event.text == "KeptMore"
    assert session._event_projection._legacy_events.empty()


@pytest.mark.asyncio
async def test_soniox_extra_tokens_after_consumed_fin_are_retained() -> None:
    session = _make_session()
    await _request_finalize(session)

    session._handle_message(
        json.dumps(
            {
                "tokens": [
                    {"text": "A", "is_final": True, "end_ms": 100},
                    {"text": "<fin>", "is_final": True},
                    {"text": "B", "is_final": True, "end_ms": 200},
                    {"text": "<fin>", "is_final": True},
                ]
            }
        )
    )
    first = session._event_projection._legacy_events.get_nowait()
    assert first.text == "A"
    assert session._event_projection._legacy_events.empty()
    assert [token.text for token in session._pending_tokens] == ["B"]

    await _request_finalize(session)
    session._handle_message(json.dumps({"tokens": [{"text": "<fin>", "is_final": True}]}))
    second = session._event_projection._legacy_events.get_nowait()
    assert second.text == "B"
    assert session._event_projection._legacy_events.empty()


@pytest.mark.asyncio
async def test_soniox_empty_final_boundary_clears_previous_segment_before_next_final() -> None:
    session = _make_session()

    await session.on_speech_end(trailing_silence_ms=0)
    first_finalize = await session._audio_q.get()
    assert isinstance(first_finalize, _FinalizeRequest)
    session._handle_message(
        json.dumps(
            {
                "tokens": [
                    {"text": "First", "is_final": True, "end_ms": 100},
                    {"text": "<fin>", "is_final": True},
                ]
            }
        )
    )
    first = session._event_projection._legacy_events.get_nowait()
    assert first.text == "First"

    await session.on_speech_end(trailing_silence_ms=0)
    empty_finalize = await session._audio_q.get()
    assert isinstance(empty_finalize, _FinalizeRequest)
    session._handle_message(json.dumps({"tokens": [{"text": "<fin>", "is_final": True}]}))
    empty_boundary = session._event_projection._legacy_events.get_nowait()
    assert empty_boundary.text == ""
    assert empty_boundary.is_final is True

    await session.on_speech_end(trailing_silence_ms=0)
    next_finalize = await session._audio_q.get()
    assert isinstance(next_finalize, _FinalizeRequest)
    session._handle_message(
        json.dumps(
            {
                "tokens": [
                    {"text": "Second", "is_final": True, "end_ms": 200},
                    {"text": "<fin>", "is_final": True},
                ]
            }
        )
    )
    second = session._event_projection._legacy_events.get_nowait()
    assert second.text == "Second"


@pytest.mark.asyncio
async def test_soniox_whitespace_final_boundary_emits_empty_final_ack() -> None:
    session = _make_session()

    await session.on_speech_end(trailing_silence_ms=0)
    finalize = await session._audio_q.get()
    assert isinstance(finalize, _FinalizeRequest)

    session._handle_message(
        json.dumps(
            {
                "tokens": [
                    {"text": "   ", "is_final": True, "end_ms": 100},
                    {"text": "<fin>", "is_final": True},
                ]
            }
        )
    )

    event = session._event_projection._legacy_events.get_nowait()
    assert event.text == ""
    assert event.is_final is True


@pytest.mark.asyncio
async def test_soniox_session_send_audio_and_stop() -> None:
    session = _make_session()

    await session.send_audio(b"abc")
    assert await session._audio_q.get() == b"abc"

    await session.stop()
    assert session._stopped is True
    assert await session._audio_q.get() is _STOP


@pytest.mark.asyncio
async def test_soniox_send_loop_preserves_finalize_before_stream_end() -> None:
    class RecordingWebSocket:
        def __init__(self) -> None:
            self.sent: list[object] = []

        async def send(self, payload: object) -> None:
            self.sent.append(payload)

    session = _make_session()
    websocket = RecordingWebSocket()
    session._ws = websocket

    await session.send_audio(b"abc")
    await session.on_speech_end(trailing_silence_ms=240)
    await session.stop()
    await session._send_loop()

    assert websocket.sent[0] == b"abc"
    assert json.loads(websocket.sent[1]) == {"type": "finalize"}
    assert websocket.sent[2] == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("channel", "reason", "duration_ms", "expected_padding"),
    [
        ("peer", "delivery_pause", 4000, True),
        ("peer", "delivery_pause", 6999, True),
        ("peer", "delivery_deadline", 7000, True),
        ("peer", "delivery_pause", 3999, False),
        ("peer", "delivery_pause", 7000, False),
        ("peer", "silence", 1000, False),
        ("peer", "source_eof", 6000, False),
        ("self", "delivery_pause", 6000, False),
        ("self", "delivery_deadline", 7000, False),
    ],
)
async def test_soniox_scoped_finalize_applies_s200_only_to_listen_fixed_boundaries(
    channel: str,
    reason: str,
    duration_ms: int,
    expected_padding: bool,
) -> None:
    class RecordingWebSocket:
        def __init__(self) -> None:
            self.sent: list[object] = []

        async def send(self, payload: object) -> None:
            self.sent.append(payload)

    session = _make_session(
        projection=STTSessionProjection(mode="scoped", provider_epoch_id="epoch-1")
    )
    websocket = RecordingWebSocket()
    session._ws = websocket
    request = _scoped_request(channel=channel)
    await session.begin_turn(request)
    writer = asyncio.create_task(session._send_loop())

    await session.seal_turn(
        request.identity,
        sealed_content_ranges=_content_span(duration_ms),
        seal_reason=reason,
        observed_trailing_silence_ms=224,
    )
    await session.stop()
    await writer

    finalize_index = next(
        index
        for index, payload in enumerate(websocket.sent)
        if isinstance(payload, str) and payload and json.loads(payload).get("type") == "finalize"
    )
    preceding = websocket.sent[finalize_index - 1] if finalize_index else None
    if expected_padding:
        assert preceding == bytes(16000 * 200 // 1000 * 2)
    else:
        assert not isinstance(preceding, bytes) or preceding != bytes(16000 * 200 // 1000 * 2)


@pytest.mark.asyncio
async def test_soniox_s200_is_atomic_before_finalize_and_does_not_claim_next_audio() -> None:
    padding_started = asyncio.Event()
    release_padding = asyncio.Event()

    class BlockingWebSocket:
        def __init__(self) -> None:
            self.sent: list[object] = []

        async def send(self, payload: object) -> None:
            self.sent.append(payload)
            if payload == bytes(16000 * 200 // 1000 * 2):
                padding_started.set()
                await release_padding.wait()

    session = _make_session(
        projection=STTSessionProjection(mode="scoped", provider_epoch_id="epoch-1")
    )
    websocket = BlockingWebSocket()
    session._ws = websocket
    request = _scoped_request()
    ranges = _content_span(4500)
    await session.begin_turn(request)
    writer = asyncio.create_task(session._send_loop())
    await session.send_turn_audio(
        request.identity,
        b"current",
        payload_sequence=1,
        source_ranges=ranges,
        context_only=False,
    )

    seal = asyncio.create_task(
        session.seal_turn(
            request.identity,
            sealed_content_ranges=ranges,
            seal_reason="delivery_pause",
            observed_trailing_silence_ms=224,
        )
    )
    await asyncio.wait_for(padding_started.wait(), timeout=1)
    await session.send_audio(b"next")
    release_padding.set()
    await seal
    await session.stop()
    await writer

    assert websocket.sent == [
        b"current",
        bytes(16000 * 200 // 1000 * 2),
        json.dumps({"type": "finalize"}),
        b"next",
        "",
    ]
    assert ranges == _content_span(4500)


@pytest.mark.asyncio
async def test_soniox_session_events_yield_and_raise() -> None:
    session = _make_session()

    session._event_projection.put_legacy(STTBackendTranscriptEvent(text="hi", is_final=True))
    session._event_projection.put_legacy(None)

    gen = session.events()
    event = await gen.__anext__()
    assert event.text == "hi"
    with pytest.raises(StopAsyncIteration):
        await gen.__anext__()

    session._event_projection.put_legacy(RuntimeError("boom"))
    gen = session.events()
    with pytest.raises(RuntimeError, match="boom"):
        await gen.__anext__()


@pytest.mark.asyncio
async def test_soniox_verify_api_key_handles_timeout(monkeypatch):
    seen: dict[str, object] = {}

    class FakeWebSocket:
        def __init__(self):
            self.sent = []

        async def send(self, payload):
            self.sent.append(payload)
            seen["config"] = payload

        async def recv(self):
            raise asyncio.TimeoutError

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class FakeWebsockets:
        @staticmethod
        def connect(*_args, **_kwargs):
            return FakeWebSocket()

    monkeypatch.setattr(
        "puripuly_heart.core.network_clients.external_websocket_connect", FakeWebsockets.connect
    )

    assert await SonioxRealtimeSTTBackend.verify_api_key("secret") is True
    config = json.loads(str(seen["config"]))
    assert config["model"] == "stt-rt-v5"


@pytest.mark.parametrize("speaker_diarization", [True, False])
@pytest.mark.asyncio
async def test_soniox_session_start_send_recv_and_close(
    monkeypatch, speaker_diarization: bool
) -> None:
    recv_queue: asyncio.Queue[object] = asyncio.Queue()

    class FakeWebSocket:
        def __init__(self):
            self.sent: list[object] = []
            self.closed = False

        async def send(self, payload):
            self.sent.append(payload)

        async def recv(self):
            return await recv_queue.get()

        async def close(self):
            self.closed = True

    ws = FakeWebSocket()

    async def connect(*_args, **_kwargs):
        return ws

    fake_websockets = SimpleNamespace(
        connect=connect,
        exceptions=SimpleNamespace(ConnectionClosedOK=type("ConnectionClosedOK", (Exception,), {})),
    )
    monkeypatch.setitem(sys.modules, "websockets", fake_websockets)
    monkeypatch.setattr(
        "puripuly_heart.core.network_clients.external_websocket_connect", connect
    )

    session = _SonioxSession(
        api_key="k",
        model="m",
        endpoint="wss://example",
        sample_rate_hz=16000,
        language_hints=["en"],
        context_terms=["Puripuly", "VRChat"],
        keepalive_interval_s=0.01,
        trailing_silence_ms=50,
        connect_timeout_s=5.0,
        language_hints_strict=True,
        enable_speaker_diarization=speaker_diarization,
    )

    await session.start()

    await session.send_audio(b"abc")
    await session.on_speech_end()

    await recv_queue.put(
        json.dumps(
            {"tokens": [{"text": "Hi", "is_final": True}, {"text": "<fin>", "is_final": True}]}
        )
    )

    event = await session._event_projection._legacy_events.get()
    assert event.text == "Hi"

    await asyncio.sleep(0.02)
    await recv_queue.put(None)
    await session.close()

    config = json.loads(ws.sent[0])
    assert config["context"]["terms"] == ["Puripuly", "VRChat"]
    assert config["language_hints"] == ["en"]
    assert config["language_hints_strict"] is True
    assert config["enable_speaker_diarization"] is speaker_diarization

    payloads = [
        payload
        for payload in ws.sent
        if isinstance(payload, str) and payload.strip().startswith("{")
    ]
    assert any(json.loads(p).get("type") == "finalize" for p in payloads)
    assert any(json.loads(p).get("type") == "keepalive" for p in payloads)
    assert b"abc" in ws.sent
    assert ws.closed is True


@pytest.mark.asyncio
async def test_soniox_session_local_server_preserves_finalize_and_remote_close() -> None:
    wire_events: list[object] = []
    keepalive_seen = asyncio.Event()
    finalize_seen = asyncio.Event()
    server_closed = asyncio.Event()

    async def handler(connection) -> None:
        wire_events.append(json.loads(await connection.recv()))
        try:
            while True:
                message = await connection.recv()
                if isinstance(message, bytes):
                    wire_events.append(("audio", len(message)))
                    continue
                payload = json.loads(message)
                wire_events.append(payload)
                if payload.get("type") == "keepalive":
                    keepalive_seen.set()
                if payload.get("type") == "finalize":
                    finalize_seen.set()
                    await connection.send(
                        json.dumps(
                            {
                                "tokens": [
                                    {"text": "hello", "is_final": True, "end_ms": 100},
                                    {"text": "<fin>", "is_final": True},
                                ]
                            }
                        )
                    )
                    await connection.close(code=1000, reason="fake-complete")
                    return
        finally:
            server_closed.set()

    server = await serve(handler, "127.0.0.1", 0, ping_interval=None)
    host, port = server.sockets[0].getsockname()[:2]
    session = _SonioxSession(
        api_key="fake-key",
        model="fake-model",
        endpoint=f"ws://{host}:{port}",
        sample_rate_hz=16000,
        language_hints=["en"],
        context_terms=[],
        keepalive_interval_s=0.01,
        trailing_silence_ms=0,
        connect_timeout_s=2,
    )

    try:
        await session.start()
        await asyncio.wait_for(keepalive_seen.wait(), timeout=1)
        await session.send_audio(b"abc")
        await session.on_speech_end(trailing_silence_ms=0)
        event = await asyncio.wait_for(session._event_projection._legacy_events.get(), timeout=1)
        assert event.text == "hello"
        await asyncio.wait_for(finalize_seen.wait(), timeout=1)
        assert session._recv_task is not None
        await asyncio.wait_for(session._recv_task, timeout=1)
        assert session._stopped is True
    finally:
        await session.close()
        server.close()
        await server.wait_closed()

    await asyncio.wait_for(server_closed.wait(), timeout=1)
    assert wire_events[0]["model"] == "fake-model"
    assert ("audio", 3) in wire_events
    assert any(
        isinstance(event, dict) and event.get("type") == "keepalive" for event in wire_events
    )
    assert any(isinstance(event, dict) and event.get("type") == "finalize" for event in wire_events)


@pytest.mark.asyncio
async def test_soniox_session_start_omits_context_when_no_terms(monkeypatch) -> None:
    class FakeWebSocket:
        def __init__(self):
            self.sent: list[object] = []
            self.closed = False

        async def send(self, payload):
            self.sent.append(payload)

        async def recv(self):
            return None

        async def close(self):
            self.closed = True

    ws = FakeWebSocket()

    async def connect(*_args, **_kwargs):
        return ws

    fake_websockets = SimpleNamespace(
        connect=connect,
        exceptions=SimpleNamespace(ConnectionClosedOK=type("ConnectionClosedOK", (Exception,), {})),
    )
    monkeypatch.setitem(sys.modules, "websockets", fake_websockets)
    monkeypatch.setattr(
        "puripuly_heart.core.network_clients.external_websocket_connect", connect
    )

    session = _make_session()
    await session.start()
    await session.close()

    config = json.loads(ws.sent[0])
    assert "context" not in config
    assert "language_hints_strict" not in config


def _soniox_events(caplog) -> list[dict[str, str]]:
    events = []
    for record in caplog.records:
        if not record.getMessage().startswith("[Soniox] "):
            continue
        parts = record.getMessage().split()
        events.append({"event": parts[1], **dict(part.split("=", 1) for part in parts[2:])})
    return events


@pytest.mark.asyncio
async def test_soniox_scoped_success_logs_one_written_and_accepted_summary(caplog) -> None:
    class WebSocket:
        async def send(self, payload: object) -> None:
            pass

    session = _make_session(
        projection=STTSessionProjection(mode="scoped", provider_epoch_id="epoch-1")
    )
    session._ws = WebSocket()
    request = _scoped_request(channel="self")
    with caplog.at_level(logging.INFO):
        await session.begin_turn(request)
        writer = asyncio.create_task(session._send_loop())
        await session.seal_turn(
            request.identity,
            sealed_content_ranges=(),
            seal_reason="silence",
            observed_trailing_silence_ms=None,
        )
        session._handle_message(json.dumps({"tokens": [{"text": "<fin>", "is_final": True}]}))
        await session.stop()
        await writer
    events = _soniox_events(caplog)
    assert len(events) == 1
    assert events[0]["event"] == "turn_end"
    assert events[0]["channel"] == "self"
    assert events[0]["utterance_id"] == str(request.identity.segment.segment_id)
    assert events[0]["trigger"] == "fin"
    assert events[0]["reason"] == "none"
    assert events[0]["finalize_requested"] == "true"
    assert events[0]["finalize_written"] == "true"
    assert events[0]["finalize_queue_ms"] != "none"
    assert events[0]["since_finalize_requested_ms"] != "none"
    assert events[0]["since_finalize_written_ms"] != "none"
    assert events[0]["fin_received"] == "true"
    assert events[0]["fin_accepted"] == "true"
    assert events[0]["last_rx_age_ms"] != "none"


@pytest.mark.asyncio
async def test_soniox_stop_logs_queued_finalize_before_writer_can_send(caplog) -> None:
    session = _make_session(
        projection=STTSessionProjection(mode="scoped", provider_epoch_id="epoch-1")
    )
    session._ws = SimpleNamespace(send=None)
    request = _scoped_request()
    with caplog.at_level(logging.INFO):
        await session.begin_turn(request)
        seal = asyncio.create_task(
            session.seal_turn(
                request.identity,
                sealed_content_ranges=(),
                seal_reason="silence",
                observed_trailing_silence_ms=None,
            )
        )
        while session._audio_q.empty():
            await asyncio.sleep(0)
        await session.stop()
        seal.cancel()
        with pytest.raises(asyncio.CancelledError):
            await seal
    events = _soniox_events(caplog)
    assert len(events) == 1
    assert events[0]["trigger"] == "local_stop"
    assert events[0]["finalize_requested"] == "true"
    assert events[0]["finalize_written"] == "false"
    assert events[0]["finalize_queue_ms"] == "none"
    assert events[0]["fin_received"] == "false"
    assert events[0]["since_finalize_requested_ms"] != "none"
    assert events[0]["since_finalize_written_ms"] == "none"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "detail", "fin_received"),
    [
        ('{"tokens": [}', "invalid_json", "false"),
        ('["not", "a message"]', "invalid_message_shape", "false"),
        ('{"tokens": "malformed"}', "invalid_tokens_shape", "false"),
        ('{"tokens": [17]}', "invalid_token_shape", "false"),
        ('{"tokens": [{"text": "<fin>", "is_final": true}]}', "fin_before_seal", "true"),
    ],
)
async def test_soniox_protocol_rejection_logs_safe_detail_once(
    caplog, message: str, detail: str, fin_received: str
) -> None:
    session = _make_session(
        projection=STTSessionProjection(mode="scoped", provider_epoch_id="epoch-1")
    )
    session._ws = SimpleNamespace(close_code=None)
    request = _scoped_request()
    with caplog.at_level(logging.INFO):
        await session.begin_turn(request)
        session._handle_message(message)
        session._scoped_transport_failure("soniox_connection_ended", orderly=True)
        await session.close()
    events = _soniox_events(caplog)
    assert len(events) == 1
    assert events[0]["trigger"] == "protocol_error"
    assert events[0]["reason"] == "soniox_protocol_ambiguity"
    assert events[0]["protocol_detail"] == detail
    assert events[0]["fin_received"] == fin_received
    assert events[0]["fin_accepted"] == "false"
    assert events[0]["finalize_requested"] == "false"
    assert events[0]["last_rx_age_ms"] != "none"


@pytest.mark.asyncio
async def test_soniox_idle_fin_is_epoch_fault_not_previous_turn(caplog) -> None:
    session = _make_session(
        projection=STTSessionProjection(mode="scoped", provider_epoch_id="epoch-1")
    )
    session._ws = SimpleNamespace(close_code=None)
    request = _scoped_request()
    with caplog.at_level(logging.INFO):
        await session.begin_turn(request)
        session._event_projection.seal(request.identity)
        session._pending_finalize_requests = 1
        session._handle_message('{"tokens": [{"text": "<fin>", "is_final": true}]}')
        session._handle_message('{"tokens": [{"text": "<fin>", "is_final": true}]}')
        session._handle_message('{"tokens": [{"text": "<fin>", "is_final": true}]}')
    events = _soniox_events(caplog)
    assert [event["event"] for event in events] == ["turn_end", "session_fault"]
    assert events[0]["trigger"] == "fin"
    assert events[1]["turn"] == "none"
    assert events[1]["utterance_id"] == "none"
    assert events[1]["epoch"] == "epoch-1"
    assert events[1]["protocol_detail"] == "fin_without_turn"
    assert events[1]["fin_received"] == "true"


@pytest.mark.asyncio
async def test_soniox_fin_with_wrong_pending_count_logs_rejection(caplog) -> None:
    session = _make_session(
        projection=STTSessionProjection(mode="scoped", provider_epoch_id="epoch-1")
    )
    session._ws = SimpleNamespace(close_code=None)
    request = _scoped_request()
    with caplog.at_level(logging.INFO):
        await session.begin_turn(request)
        session._event_projection.seal(request.identity)
        session._handle_message('{"tokens": [{"text": "<fin>", "is_final": true}]}')
    events = _soniox_events(caplog)
    assert len(events) == 1
    assert events[0]["trigger"] == "protocol_error"
    assert events[0]["protocol_detail"] == "fin_pending_count_mismatch"
    assert events[0]["fin_received"] == "true"
    assert events[0]["fin_accepted"] == "false"
    assert events[0]["finalize_requested"] == "false"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (403, "403"),
        ("481", "481"),
        (True, "unclassified"),
        ("secret-token", "unclassified"),
        ({"secret": "value"}, "unclassified"),
    ],
)
async def test_soniox_server_error_code_never_logs_arbitrary_content(
    caplog, code: object, expected: str
) -> None:
    session = _make_session(
        projection=STTSessionProjection(mode="scoped", provider_epoch_id="epoch-1")
    )
    session._ws = SimpleNamespace(close_code="private-close-reason")
    request = _scoped_request()
    with caplog.at_level(logging.INFO):
        await session.begin_turn(request)
        session._handle_message(json.dumps({"error": "secret-error-content", "error_code": code}))
        await session.close()
    events = _soniox_events(caplog)
    assert len(events) == 1
    assert events[0]["trigger"] == "provider_error"
    assert events[0]["reason"] == "soniox_request_failed"
    assert events[0]["server_error_code"] == expected
    assert events[0]["ws_close_code"] == "unclassified"
    assert events[0]["exception_type"] == "none"
    assert "secret" not in caplog.text
    assert "private-close-reason" not in caplog.text


@pytest.mark.asyncio
async def test_soniox_delayed_finalize_write_never_claims_next_turn(caplog) -> None:
    class WebSocket:
        async def send(self, payload: object) -> None:
            pass

    session = _make_session(
        projection=STTSessionProjection(mode="scoped", provider_epoch_id="epoch-1")
    )
    session._ws = WebSocket()
    first = _scoped_request()
    second = replace(
        first,
        identity=replace(
            first.identity,
            segment=replace(first.identity.segment, segment_id=uuid4()),
            provider_turn_id="turn-2",
        ),
    )
    with caplog.at_level(logging.INFO):
        await session.begin_turn(first)
        seal = asyncio.create_task(
            session.seal_turn(
                first.identity,
                sealed_content_ranges=(),
                seal_reason="silence",
                observed_trailing_silence_ms=None,
            )
        )
        while session._audio_q.empty():
            await asyncio.sleep(0)
        session._handle_message('{"tokens": [{"text": "<fin>", "is_final": true}]}')
        seal.cancel()
        with pytest.raises(asyncio.CancelledError):
            await seal
        await session.begin_turn(second)
        writer = asyncio.create_task(session._send_loop())
        while not session._audio_q.empty():
            await asyncio.sleep(0)
        await session.stop()
        await writer
    first_log, second_log = _soniox_events(caplog)
    assert first_log["turn"] == "turn-1"
    assert first_log["finalize_requested"] == "true"
    assert first_log["finalize_written"] == "false"
    assert second_log["turn"] == "turn-2"
    assert second_log["trigger"] == "local_stop"
    assert second_log["finalize_requested"] == "false"
    assert second_log["finalize_written"] == "false"


@pytest.mark.asyncio
async def test_soniox_write_failure_preserves_original_error_before_close(caplog) -> None:
    class FailingWebSocket:
        close_code = 1006

        async def send(self, payload: object) -> None:
            if payload != "":
                raise ConnectionResetError("private-websocket-message")

        async def close(self) -> None:
            pass

    session = _make_session(
        projection=STTSessionProjection(mode="scoped", provider_epoch_id="epoch-1")
    )
    session._ws = FailingWebSocket()
    request = _scoped_request()
    with caplog.at_level(logging.INFO):
        await session.begin_turn(request)
        writer = asyncio.create_task(session._send_loop())
        with pytest.raises(ConnectionResetError):
            await session.seal_turn(
                request.identity,
                sealed_content_ranges=(),
                seal_reason="silence",
                observed_trailing_silence_ms=None,
            )
        await writer
        terminal = await asyncio.wait_for(anext(session.turn_events()), timeout=1)
        ended = await asyncio.wait_for(anext(session.turn_events()), timeout=1)
        assert isinstance(terminal, STTProviderTurnTerminal)
        assert terminal.failure_retryable is True
        assert isinstance(ended, STTProviderEpochEnded)
        assert ended.failure_retryable is True
        await session.close()
    events = _soniox_events(caplog)
    assert len(events) == 1
    assert events[0]["trigger"] == "transport_error"
    assert events[0]["reason"] == "soniox_write_failed"
    assert events[0]["finalize_requested"] == "true"
    assert events[0]["finalize_written"] == "false"
    assert events[0]["ws_close_code"] == "1006"
    assert events[0]["exception_type"] == "ConnectionResetError"
    assert "private-websocket-message" not in caplog.text


@pytest.mark.asyncio
async def test_soniox_close_snapshots_unresolved_turn_before_await(caplog) -> None:
    closing = asyncio.Event()
    resume = asyncio.Event()

    class WebSocket:
        async def close(self) -> None:
            closing.set()
            await resume.wait()

    session = _make_session(
        projection=STTSessionProjection(mode="scoped", provider_epoch_id="epoch-1")
    )
    session._ws = WebSocket()
    request = _scoped_request()
    with caplog.at_level(logging.INFO):
        await session.begin_turn(request)
        closing_task = asyncio.create_task(session.close())
        await asyncio.wait_for(closing.wait(), timeout=1)
        events = _soniox_events(caplog)
        assert len(events) == 1
        assert events[0]["trigger"] == "local_close"
        assert events[0]["turn"] == "turn-1"
        assert events[0]["finalize_requested"] == "false"
        resume.set()
        await closing_task
    assert len(_soniox_events(caplog)) == 1


@pytest.mark.asyncio
async def test_soniox_abort_logs_safe_reason_before_turn_cleanup(caplog) -> None:
    session = _make_session(
        projection=STTSessionProjection(mode="scoped", provider_epoch_id="epoch-1")
    )
    session._ws = SimpleNamespace(close_code=None)
    request = _scoped_request()
    with caplog.at_level(logging.INFO):
        await session.begin_turn(request)
        await session.abort_turn(request.identity, reason="toggle_off:private arbitrary detail")
        await session.close()
    events = _soniox_events(caplog)
    assert len(events) == 1
    assert events[0]["trigger"] == "abort"
    assert events[0]["reason"] == "toggle_off"
    assert "private arbitrary detail" not in caplog.text
