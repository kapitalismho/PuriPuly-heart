from __future__ import annotations

import pytest

from puripuly_heart.providers.llm.qwen import QwenLLMProvider
from puripuly_heart.providers.stt.qwen_asr import QwenASRRealtimeSTTBackend


@pytest.mark.asyncio
async def test_qwen_asr_backend_requires_api_key() -> None:
    backend = QwenASRRealtimeSTTBackend(
        api_key="",
        model="qwen3-asr-flash-realtime",
        endpoint="wss://example",
        language="en",
        sample_rate_hz=16000,
    )

    with pytest.raises(ValueError):
        await backend.open_session()


@pytest.mark.asyncio
async def test_qwen_asr_backend_requires_valid_sample_rate() -> None:
    backend = QwenASRRealtimeSTTBackend(
        api_key="k",
        model="qwen3-asr-flash-realtime",
        endpoint="wss://example",
        language="en",
        sample_rate_hz=44100,
    )

    with pytest.raises(ValueError):
        await backend.open_session()


@pytest.mark.asyncio
async def test_qwen_asr_backend_verify_api_key_delegates(monkeypatch) -> None:
    seen: dict[str, str] = {}

    async def fake_verify(api_key: str) -> bool:
        seen["api_key"] = api_key
        return True

    monkeypatch.setattr(QwenLLMProvider, "verify_api_key", staticmethod(fake_verify))

    ok = await QwenASRRealtimeSTTBackend.verify_api_key("secret")

    assert ok is True
    assert seen == {"api_key": "secret"}
