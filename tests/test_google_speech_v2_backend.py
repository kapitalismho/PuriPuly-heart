from __future__ import annotations

import pytest

from ajebal_daera_translator.providers.stt.deepgram import DeepgramRealtimeSTTBackend


@pytest.mark.asyncio
async def test_deepgram_backend_requires_api_key() -> None:
    backend = DeepgramRealtimeSTTBackend(
        api_key="",
        language="en",
        model="nova-3",
        sample_rate_hz=16000,
    )

    with pytest.raises(ValueError):
        await backend.open_session()


@pytest.mark.asyncio
async def test_deepgram_backend_requires_valid_sample_rate() -> None:
    backend = DeepgramRealtimeSTTBackend(
        api_key="k",
        language="en",
        model="nova-3",
        sample_rate_hz=44100,
    )

    with pytest.raises(ValueError):
        await backend.open_session()
