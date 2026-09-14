from __future__ import annotations

import pytest

from tests.integration.helpers import (
    drain_and_close,
    get_qwen_audio_endpoint,
    integration_mark,
    open_session,
    require_env,
    stream_silence,
)

pytestmark = integration_mark()


@pytest.mark.asyncio
async def test_qwen_audio_streaming_smoke():
    api_key = require_env("ALIBABA_API_KEY")

    from puripuly_heart.providers.stt.qwen_audio import QwenAudioStreamingSTTBackend

    backend = QwenAudioStreamingSTTBackend(
        api_key=api_key,
        endpoint=get_qwen_audio_endpoint(),
        language_hints=("ko",),
    )

    session = await open_session(backend)

    await stream_silence(session)

    await session.stop()
    await drain_and_close(session)
