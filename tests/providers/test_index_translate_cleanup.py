from __future__ import annotations

import asyncio

import httpx
import pytest

from puripuly_heart.app.services.index_selection import IndexSelectionOwner
from puripuly_heart.app.wiring import wiring_index_selection
from puripuly_heart.config.settings_vnext.defaults import new_settings_for_first_run
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.core.llm.index_translate import IndexServiceUnavailableError
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
def install_probe_client(monkeypatch):
    monkeypatch.setattr(index_translate, "select_tls", lambda: object())
    monkeypatch.setattr(index_translate, "ProxyPolicy", lambda: object())

    def install(client):
        providers = []
        creations = []

        def create_client(**kwargs):
            creations.append(kwargs)
            return client

        def create_provider():
            provider = index_translate.IndexTranslateProvider()
            providers.append(provider)
            return provider

        monkeypatch.setattr(index_translate.network_clients, "external_async_client", create_client)
        monkeypatch.setattr(wiring_index_selection, "IndexTranslateProvider", create_provider)
        return providers, creations

    return install


@pytest.mark.parametrize("retire", ["shutdown", "selection_change"])
async def test_selection_owner_shutdown_drains_probe_close_despite_repeated_cancels(
    install_probe_client, retire
):
    client = HeldCloseClient("pending")
    providers, creations = install_probe_client(client)
    current = new_settings_for_first_run("en-US")
    messages = []
    diagnostics = []

    async def unexpected_observation():
        pytest.fail("A cancelled probe must not observe allowance")

    async def unexpected_commit(*args):
        pytest.fail("A cancelled probe must not commit fallback")

    owner = IndexSelectionOwner(
        settings_provider=lambda: current,
        probe=wiring_index_selection.check_index_availability,
        observe_allowance=unexpected_observation,
        commit_fallback=unexpected_commit,
        message_sink=messages.append,
        diagnostic_sink=diagnostics.append,
    )
    shutdown = None
    owner.start()
    try:
        await asyncio.wait_for(client.post_started.wait(), 1)
        work = owner._work_task
        assert work is not None
        if retire == "selection_change":
            current = AppSettingsVNext()
            owner.selection_changed(current)
        else:
            owner.stop_ingress()
        await asyncio.wait_for(client.close_started.wait(), 1)
        provider = providers[0]
        shutdown = asyncio.create_task(owner.close())
        for _ in range(3):
            owner.stop_ingress()
            await asyncio.sleep(0)
        assert not shutdown.done()
        assert not work.done()
        assert work in owner._tasks
        assert provider._client is client
        assert provider._close_task is not None
        assert not provider._close_task.done()
        assert not client.close_finished.is_set()
        assert client.events == ["request_cancelled", "close_started"]
        assert messages == []
        assert diagnostics == []
        client.close_release.set()
        await asyncio.wait_for(shutdown, 1)
        assert work.cancelled()
        assert owner._tasks == set()
        assert client.events == ["request_cancelled", "close_started", "close_finished"]
        assert client.close_finished.is_set()
        assert provider._client is None
        assert len(creations) == 1
        await owner.close()
        await provider.close()
        assert client.close_calls == 1
    finally:
        client.close_release.set()
        await owner.close()
        if shutdown is not None:
            await asyncio.gather(shutdown, return_exceptions=True)


@pytest.mark.parametrize("outcome", ["success", "failure"])
async def test_probe_wrapper_settles_client_close_before_returning_success_or_failure(
    install_probe_client, outcome
):
    client = HeldCloseClient(outcome)
    providers, creations = install_probe_client(client)
    probe = asyncio.create_task(wiring_index_selection.check_index_availability())
    try:
        await asyncio.wait_for(client.close_started.wait(), 1)
        assert not probe.done()
        assert providers[0]._client is client
        client.close_release.set()
        if outcome == "failure":
            with pytest.raises(IndexServiceUnavailableError) as failure:
                await asyncio.wait_for(probe, 1)
            assert failure.value.status_code == 503
        else:
            assert await asyncio.wait_for(probe, 1) is None
        assert client.events == ["close_started", "close_finished"]
        assert client.close_finished.is_set()
        assert providers[0]._client is None
        assert len(creations) == 1
        await providers[0].close()
        await providers[0].close()
        assert client.close_calls == 1
    finally:
        client.close_release.set()
        await asyncio.gather(probe, return_exceptions=True)


async def test_concurrent_close_waiters_share_cleanup_and_preserve_repeated_cancellation(
    install_probe_client,
):
    client = HeldCloseClient()
    _, creations = install_probe_client(client)
    provider = index_translate.IndexTranslateProvider()
    await provider.check_availability()
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
            await provider.check_availability()
        assert client.close_calls == 1
        assert len(creations) == 1
    finally:
        client.close_release.set()
        await asyncio.gather(cancelled_waiter, return_exceptions=True)
        if normal_waiter is not None:
            await asyncio.gather(normal_waiter, return_exceptions=True)
        await provider.close()


async def test_failed_close_retains_client_and_observed_terminal_failure_without_duplicate_cleanup(
    install_probe_client,
):
    failure = OSError("synthetic close failure")
    client = HeldCloseClient(close_error=failure)
    install_probe_client(client)
    provider = index_translate.IndexTranslateProvider()
    await provider.check_availability()
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
    install_probe_client,
):
    _, creations = install_probe_client(HeldCloseClient())
    provider = index_translate.IndexTranslateProvider()
    await provider.close()
    await provider.close()
    with pytest.raises(asyncio.CancelledError):
        await provider.check_availability()
    assert creations == []
