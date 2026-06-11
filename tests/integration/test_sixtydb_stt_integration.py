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
async def test_sixtydb_realtime_streaming_smoke():
    api_key = require_env("SIXTYDB_API_KEY")

    from puripuly_heart.providers.stt.sixtydb import SixtyDBRealtimeSTTBackend

    backend = SixtyDBRealtimeSTTBackend(
        api_key=api_key,
        endpoint=os.getenv("SIXTYDB_STT_ENDPOINT", "wss://api.60db.ai/ws/stt"),
        language_codes=[os.getenv("SIXTYDB_STT_LANGUAGE", "en")],
        sample_rate_hz=int(os.getenv("SIXTYDB_STT_SAMPLE_RATE", "16000")),
        utterance_end_ms=int(os.getenv("SIXTYDB_STT_UTTERANCE_END_MS", "300")),
        trailing_silence_ms=int(os.getenv("SIXTYDB_STT_TRAILING_SILENCE_MS", "400")),
    )

    session = await open_session(backend)

    # Send a short silence stream just to validate connectivity/stream lifecycle.
    await stream_silence(session)

    await session.stop()
    await drain_and_close(session)
