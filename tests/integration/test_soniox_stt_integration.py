from __future__ import annotations

import os

import pytest

from tests.integration.helpers import (
    drain_and_close,
    integration_mark,
    open_session,
    require_env,
    stream_silence,
)

pytestmark = integration_mark()


@pytest.mark.asyncio
async def test_soniox_realtime_streaming_smoke():
    api_key = require_env("SONIOX_API_KEY")

    from puripuly_heart.providers.stt.soniox import SonioxRealtimeSTTBackend

    backend = SonioxRealtimeSTTBackend(
        api_key=api_key,
        model=os.getenv("SONIOX_STT_MODEL", "stt-rt-v5"),
        endpoint=os.getenv("SONIOX_STT_ENDPOINT", "wss://stt-rt.soniox.com/transcribe-websocket"),
        language_hints=[os.getenv("SONIOX_STT_LANGUAGE", "ko")],
        sample_rate_hz=int(os.getenv("SONIOX_STT_SAMPLE_RATE", "16000")),
        keepalive_interval_s=float(os.getenv("SONIOX_STT_KEEPALIVE", "10")),
        trailing_silence_ms=int(os.getenv("SONIOX_STT_TRAILING_SILENCE_MS", "100")),
    )

    session = await open_session(backend)

    # Send a short silence stream just to validate connectivity/stream lifecycle.
    await stream_silence(session)

    await session.stop()
    await drain_and_close(session)
