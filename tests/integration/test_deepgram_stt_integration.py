from __future__ import annotations

import asyncio
import os

import pytest


pytestmark = pytest.mark.skipif(os.getenv("INTEGRATION") != "1", reason="set INTEGRATION=1 to run integration tests")


@pytest.mark.asyncio
async def test_deepgram_realtime_streaming_smoke():
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        pytest.skip("missing env var DEEPGRAM_API_KEY")

    try:
        import websocket  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            "websocket-client is required for this integration test; install with pip install websocket-client"
        ) from exc

    from ajebal_daera_translator.providers.stt.deepgram import DeepgramRealtimeSTTBackend

    backend = DeepgramRealtimeSTTBackend(
        api_key=api_key,
        model=os.getenv("DEEPGRAM_STT_MODEL", "nova-3"),
        language=os.getenv("DEEPGRAM_STT_LANGUAGE", "ko"),
        sample_rate_hz=int(os.getenv("DEEPGRAM_STT_SAMPLE_RATE", "16000")),
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
