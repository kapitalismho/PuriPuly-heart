from __future__ import annotations

import asyncio
import os

import pytest


pytestmark = pytest.mark.skipif(os.getenv("INTEGRATION") != "1", reason="set INTEGRATION=1 to run integration tests")


@pytest.mark.asyncio
async def test_google_speech_v2_streaming_smoke():
    recognizer = os.getenv("GOOGLE_SPEECH_RECOGNIZER")
    if not recognizer:
        pytest.skip("missing env var GOOGLE_SPEECH_RECOGNIZER")

    try:
        import google.cloud.speech_v2  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            "google-cloud-speech is required for this integration test; install project dependencies."
        ) from exc

    from ajebal_daera_translator.providers.stt.google_speech_v2 import GoogleSpeechV2Backend

    backend = GoogleSpeechV2Backend(
        recognizer=recognizer,
        endpoint=os.getenv("GOOGLE_SPEECH_ENDPOINT", "speech.googleapis.com"),
        sample_rate_hz=16000,
        language_codes=(os.getenv("GOOGLE_SPEECH_LANGUAGE", "en-US"),),
    )

    session = await backend.open_session()

    # 32ms chunks @ 16kHz => 512 samples => 1024 bytes PCM16LE.
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
