from __future__ import annotations

import asyncio
import ssl
from dataclasses import replace

import pytest

from puripuly_heart.core.stt.backend import (
    STTProviderEpochEnded,
    STTProviderTurnTerminal,
    STTSessionProjection,
)
from tests.helpers.scoped_provider_sessions import _deepgram_session, _request, _scribe_session
from tests.providers.test_custom_stt import (
    _backend,
    _FakeAsyncClient,
    _FakeResponse,
    _scoped_request,
)


@pytest.mark.parametrize("channel", ["self", "peer"])
@pytest.mark.parametrize("error,retryable", [
    (ConnectionError("controlled closed connection"), True),
    (TimeoutError(), True),
    (ssl.SSLError("controlled TLS failure"), False),
    (PermissionError("controlled permission failure"), False),
    (RuntimeError("network transient retry decode_failure"), False),
    (ValueError("controlled protocol failure"), False),
])
async def test_deepgram_transport_classification_uses_typed_native_cause(channel, error, retryable):
    session = _deepgram_session()
    request = replace(_request("deepgram"), channel=channel)
    try:
        await session.begin_turn(request)
        session._scoped_transport_end(False, "deepgram_transport_error", error)
        events = session.turn_events()
        terminal = await asyncio.wait_for(anext(events), 1)
        ended = await asyncio.wait_for(anext(events), 1)
        assert isinstance(terminal, STTProviderTurnTerminal)
        assert isinstance(ended, STTProviderEpochEnded)
        assert terminal.outcome == "failed" and terminal.text == ""
        assert terminal.failure_retryable is retryable
        assert ended.failure_retryable is retryable
        assert terminal.failure_diagnostics.channel == channel
    finally:
        await session.close()


@pytest.mark.parametrize("channel", ["self", "peer"])
@pytest.mark.parametrize("reason,retryable", [
    ("scribe_session_time_limit_exceeded", True),
    ("scribe_insufficient_audio_activity", True),
    ("scribe_resource_exhausted", True),
    ("scribe_rate_limited", True),
    ("scribe_auth_error", False),
    ("scribe_quota_exceeded", False),
    ("scribe_invalid_request", False),
    ("scribe_committed_transcript_missing_text", False),
    ("network transient retry", False),
])
async def test_scribe_native_event_policy_is_explicit(channel, reason, retryable):
    session, _ = _scribe_session()
    request = replace(_request("elevenlabs_scribe"), channel=channel)
    try:
        await session.begin_turn(request)
        session._scoped_transport_failure(reason, orderly=False)
        events = session.turn_events()
        terminal = await asyncio.wait_for(anext(events), 1)
        ended = await asyncio.wait_for(anext(events), 1)
        assert terminal.outcome == "failed" and terminal.text == ""
        assert terminal.failure_retryable is retryable
        assert ended.failure_retryable is retryable
        assert terminal.failure_diagnostics.channel == channel
    finally:
        await session.close()


@pytest.mark.parametrize("channel", ["self", "peer"])
@pytest.mark.parametrize("status,retryable", [(503, True), (401, False), (403, False), (404, False), (400, False)])
async def test_custom_offline_http_error_classification_reaches_native_terminal(channel, status, retryable):
    requests = []

    async def response(_url, **options):
        requests.append(options)
        return _FakeResponse(status, {"error": "controlled rejection"})

    backend = _backend(http_client_factory=lambda **_: _FakeAsyncClient(response))
    session = await backend.open_session(projection=STTSessionProjection("scoped", "epoch-1"))
    request = _scoped_request(channel=channel)
    try:
        await session.begin_turn(request)
        await session.send_turn_audio(request.identity, b"\x00\x10" * 160,
                                      payload_sequence=1, source_ranges=(), context_only=False)
        await session.seal_turn(request.identity, sealed_content_ranges=(),
                                seal_reason="silence", observed_trailing_silence_ms=800)
        events = session.turn_events()
        terminal = await asyncio.wait_for(anext(events), 1)
        ended = await asyncio.wait_for(anext(events), 1)
        assert terminal.outcome == "failed" and terminal.text == ""
        assert terminal.failure_retryable is retryable
        assert ended.failure_retryable is retryable
        assert len(requests) == 1
        assert terminal.failure_diagnostics.channel == channel
    finally:
        await session.close()
        await backend.close()
