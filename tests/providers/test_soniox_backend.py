from __future__ import annotations

import asyncio
import json
import logging
import sys
from types import SimpleNamespace

import pytest
from websockets.asyncio.server import serve

from puripuly_heart.core.stt.backend import STTBackendTranscriptEvent
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
    with pytest.raises(ValueError, match="api_key"):
        await backend.open_session()

    backend = SonioxRealtimeSTTBackend(api_key="k", language_hints=["en"], endpoint="")
    with pytest.raises(ValueError, match="endpoint"):
        await backend.open_session()

    backend = SonioxRealtimeSTTBackend(
        api_key="k",
        language_hints=["en"],
        keepalive_interval_s=0.0,
    )
    with pytest.raises(ValueError, match="keepalive_interval_s"):
        await backend.open_session()

    backend = SonioxRealtimeSTTBackend(
        api_key="k",
        language_hints=[],
        language_hints_strict=True,
    )
    with pytest.raises(ValueError, match="language_hints_strict"):
        await backend.open_session()


def test_soniox_backend_defaults_to_realtime_v5_model() -> None:
    backend = SonioxRealtimeSTTBackend(api_key="k", language_hints=["en"])

    assert backend.model == "stt-rt-v5"


@pytest.mark.asyncio
async def test_soniox_session_handles_message_errors() -> None:
    session = _make_session()

    session._handle_message("not-json")
    assert session._events.empty()

    session._handle_message(json.dumps({"error": "bad"}))
    event = session._events.get_nowait()
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
    event = session._events.get_nowait()

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

    event = session._events.get_nowait()
    assert event.text == "안녕こんにちは你好世界"
    assert [(run.text, run.language) for run in event.final_language_runs] == [
        ("안녕", "ko"),
        ("こんにちは", "ja"),
        ("你好", "zh"),
        ("世界", "ja"),
    ]


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

    event = session._events.get_nowait()
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

    event = session._events.get_nowait()
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
    first_event = merged._events.get_nowait()
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
    merged_event = merged._events.get_nowait()

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
    original_event = replaced._events.get_nowait()
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
    replaced_event = replaced._events.get_nowait()

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

    event = session._events.get_nowait()
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
    assert session._events.empty()

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
    event = session._events.get_nowait()
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
    event = session._events.get_nowait()

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
    assert "boundary_reason=soft_pause" in caplog.text
    assert "observed_tail_ms=240" in caplog.text
    assert "boundary_wait_ms=240" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.INFO):
        await session.on_speech_end()

    finalize = await session._audio_q.get()
    assert isinstance(finalize, _FinalizeRequest)
    assert session._audio_q.empty()
    assert "boundary_reason=None" in caplog.text
    assert "boundary_wait_ms=None" in caplog.text


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

    events = [session._events.get_nowait() for _ in range(3)]
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

    event = session._events.get_nowait()
    assert event.text == "Only"
    assert session._events.empty()


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
    assert session._events.empty()
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
    event = session._events.get_nowait()
    assert event.text == "KeptMore"
    assert session._events.empty()


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
    first = session._events.get_nowait()
    assert first.text == "A"
    assert session._events.empty()
    assert [token.text for token in session._pending_tokens] == ["B"]

    await _request_finalize(session)
    session._handle_message(json.dumps({"tokens": [{"text": "<fin>", "is_final": True}]}))
    second = session._events.get_nowait()
    assert second.text == "B"
    assert session._events.empty()


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
    first = session._events.get_nowait()
    assert first.text == "First"

    await session.on_speech_end(trailing_silence_ms=0)
    empty_finalize = await session._audio_q.get()
    assert isinstance(empty_finalize, _FinalizeRequest)
    session._handle_message(json.dumps({"tokens": [{"text": "<fin>", "is_final": True}]}))
    empty_boundary = session._events.get_nowait()
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
    second = session._events.get_nowait()
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

    event = session._events.get_nowait()
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
async def test_soniox_session_events_yield_and_raise() -> None:
    session = _make_session()

    session._events.put_nowait(STTBackendTranscriptEvent(text="hi", is_final=True))
    session._events.put_nowait(None)

    gen = session.events()
    event = await gen.__anext__()
    assert event.text == "hi"
    with pytest.raises(StopAsyncIteration):
        await gen.__anext__()

    session._events.put_nowait(RuntimeError("boom"))
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

    monkeypatch.setitem(__import__("sys").modules, "websockets", FakeWebsockets)

    assert await SonioxRealtimeSTTBackend.verify_api_key("secret") is True
    config = json.loads(str(seen["config"]))
    assert config["model"] == "stt-rt-v5"


@pytest.mark.asyncio
async def test_soniox_session_start_send_recv_and_close(monkeypatch) -> None:
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
    )

    await session.start()

    await session.send_audio(b"abc")
    await session.on_speech_end()

    await recv_queue.put(
        json.dumps(
            {"tokens": [{"text": "Hi", "is_final": True}, {"text": "<fin>", "is_final": True}]}
        )
    )

    event = await session._events.get()
    assert event.text == "Hi"

    await asyncio.sleep(0.02)
    await recv_queue.put(None)
    await session.close()

    config = json.loads(ws.sent[0])
    assert config["context"]["terms"] == ["Puripuly", "VRChat"]
    assert config["language_hints"] == ["en"]
    assert config["language_hints_strict"] is True

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
        event = await asyncio.wait_for(session._events.get(), timeout=1)
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

    session = _make_session()
    await session.start()
    await session.close()

    config = json.loads(ws.sent[0])
    assert "context" not in config
    assert "language_hints_strict" not in config
