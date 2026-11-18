from __future__ import annotations

import asyncio
from uuid import uuid4

import httpx
import pytest

from puripuly_heart.providers.llm import index_translate


class HeldCloseClient:
    def __init__(self, outcome="success", *, close_error=None):
        self.outcome = outcome
        self.close_error = close_error
        self.post_started = asyncio.Event()
        self.close_started = asyncio.Event()
        self.close_release = asyncio.Event()
        self.close_finished = asyncio.Event()
        self.close_calls = 0
        self.events = []

    async def post(self, url, **kwargs):
        self.post_started.set()
        if self.outcome == "pending":
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.events.append("request_cancelled")
                raise
        if self.outcome == "failure":
            return httpx.Response(503)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "Hola."},
                    }
                ]
            },
        )

    async def aclose(self):
        self.close_calls += 1
        self.events.append("close_started")
        self.close_started.set()
        try:
            await self.close_release.wait()
        except asyncio.CancelledError:
            self.events.append("close_cancelled")
            raise
        if self.close_error is not None:
            self.events.append("close_failed")
            raise self.close_error
        self.events.append("close_finished")
        self.close_finished.set()


@pytest.fixture
def install_client(monkeypatch):
    monkeypatch.setattr(index_translate, "select_tls", lambda: object())
    monkeypatch.setattr(index_translate, "ProxyPolicy", lambda: object())

    def install(client):
        creations = []

        def create_client(**kwargs):
            creations.append(kwargs)
            return client

        monkeypatch.setattr(index_translate.network_clients, "external_async_client", create_client)
        return creations

    return install


async def translate(provider):
    return await provider.translate(
        utterance_id=uuid4(),
        text="Hello.",
        system_prompt="Translate from English to Spanish. Return only the translation.",
        source_language="en",
        target_language="es",
    )


async def test_concurrent_close_waiters_share_cleanup_and_preserve_repeated_cancellation(
    install_client,
):
    client = HeldCloseClient()
    creations = install_client(client)
    provider = index_translate.IndexTranslateProvider()
    await translate(provider)
    cancelled_waiter = asyncio.create_task(provider.close())
    normal_waiter = None
    try:
        await asyncio.wait_for(client.close_started.wait(), 1)
        normal_waiter = asyncio.create_task(provider.close())
        for _ in range(3):
            cancelled_waiter.cancel()
            await asyncio.sleep(0)
        assert not cancelled_waiter.done()
        assert not normal_waiter.done()
        assert provider._client is client
        assert client.events == ["close_started"]
        client.close_release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(cancelled_waiter, 1)
        assert await asyncio.wait_for(normal_waiter, 1) is None
        assert client.events == ["close_started", "close_finished"]
        assert provider._client is None
        assert len(creations) == 1
        await provider.close()
        with pytest.raises(asyncio.CancelledError):
            await translate(provider)
        assert client.close_calls == 1
        assert len(creations) == 1
    finally:
        client.close_release.set()
        await asyncio.gather(cancelled_waiter, return_exceptions=True)
        if normal_waiter is not None:
            await asyncio.gather(normal_waiter, return_exceptions=True)
        await provider.close()


async def test_failed_close_retains_client_and_observed_terminal_failure_without_duplicate_cleanup(
    install_client,
):
    failure = OSError("synthetic close failure")
    client = HeldCloseClient(close_error=failure)
    install_client(client)
    provider = index_translate.IndexTranslateProvider()
    await translate(provider)
    client.close_release.set()
    for _ in range(2):
        with pytest.raises(OSError) as observed:
            await provider.close()
        assert observed.value is failure
        assert provider._client is client
    assert client.events == ["close_started", "close_failed"]
    assert client.close_calls == 1
    assert not client.close_finished.is_set()


async def test_close_without_client_is_idempotent_and_does_not_create_transport(
    install_client,
):
    creations = install_client(HeldCloseClient())
    provider = index_translate.IndexTranslateProvider()
    await provider.close()
    await provider.close()
    with pytest.raises(asyncio.CancelledError):
        await translate(provider)
    assert creations == []
