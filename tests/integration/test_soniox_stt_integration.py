from __future__ import annotations

import asyncio
import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("INTEGRATION") != "1", reason="set INTEGRATION=1 to run integration tests"
)


@pytest.mark.asyncio
async def test_soniox_realtime_streaming_smoke():
    api_key = os.getenv("SONIOX_API_KEY")
    if not api_key:
        pytest.skip("missing env var SONIOX_API_KEY")

    from puripuly_heart.providers.stt.soniox import SonioxRealtimeSTTBackend

    backend = SonioxRealtimeSTTBackend(
        api_key=api_key,
        model=os.getenv("SONIOX_STT_MODEL", "stt-rt-v3"),
        endpoint=os.getenv("SONIOX_STT_ENDPOINT", "wss://stt-rt.soniox.com/transcribe-websocket"),
        language_hints=[os.getenv("SONIOX_STT_LANGUAGE", "ko")],
        sample_rate_hz=int(os.getenv("SONIOX_STT_SAMPLE_RATE", "16000")),
        keepalive_interval_s=float(os.getenv("SONIOX_STT_KEEPALIVE", "10")),
        trailing_silence_ms=int(os.getenv("SONIOX_STT_TRAILING_SILENCE_MS", "100")),
    )

    session = await backend.open_session()

    # Send a short silence stream just to validate connectivity/stream lifecycle.
    silence = b"\0" * 1024
    for _ in range(10):
        await session.send_audio(silence)
        await asyncio.sleep(0.032)

    await session.stop()

    async def _drain():
        async for _ev in session.events():
            pass

    await asyncio.wait_for(_drain(), timeout=30.0)
    await asyncio.wait_for(session.close(), timeout=5.0)
