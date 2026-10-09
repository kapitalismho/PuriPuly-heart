from __future__ import annotations

import httpx

import pytest

from puripuly_heart.core import network_clients

from puripuly_heart.providers.stt.deepgram import DeepgramRealtimeSTTBackend


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


@pytest.mark.asyncio
async def test_deepgram_backend_requires_positive_connect_timeout() -> None:
    backend = DeepgramRealtimeSTTBackend(
        api_key="k",
        language="en",
        model="nova-3",
        sample_rate_hz=16000,
        connect_timeout_s=0.0,
    )

    with pytest.raises(ValueError):
        await backend.open_session()


@pytest.mark.asyncio
async def test_deepgram_backend_verify_api_key_handles_empty_and_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = []
    def respond(request):
        seen.append(request)
        return httpx.Response(200)
    monkeypatch.setattr(
        network_clients, "external_client",
        lambda **kwargs: httpx.Client(transport=httpx.MockTransport(respond)),
    )

    assert await DeepgramRealtimeSTTBackend.verify_api_key("") is False
    assert await DeepgramRealtimeSTTBackend.verify_api_key("secret") is True
    assert len(seen) == 1
    assert seen[0].headers["Authorization"] == "Token secret"


@pytest.mark.asyncio
async def test_deepgram_backend_verify_api_key_raises_on_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        network_clients, "external_client",
        lambda **kwargs: httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(401))),
    )

    with pytest.raises(httpx.HTTPStatusError) as caught:
        await DeepgramRealtimeSTTBackend.verify_api_key("secret")
    assert caught.value.response.status_code == 401
