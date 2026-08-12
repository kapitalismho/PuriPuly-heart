from __future__ import annotations

import urllib.error

import pytest

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
    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(_request, timeout=0):
        assert timeout == 5
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    assert await DeepgramRealtimeSTTBackend.verify_api_key("") is False
    assert await DeepgramRealtimeSTTBackend.verify_api_key("secret") is True


@pytest.mark.asyncio
async def test_deepgram_backend_verify_api_key_raises_on_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(_request, timeout=0):
        _ = timeout
        raise urllib.error.HTTPError(
            url="https://api.deepgram.com/v1/projects",
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(Exception, match="HTTP 401"):
        await DeepgramRealtimeSTTBackend.verify_api_key("secret")
