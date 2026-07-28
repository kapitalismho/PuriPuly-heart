from __future__ import annotations

import asyncio
from typing import cast

import pytest

from puripuly_heart.app.services.openrouter_pkce_flow import OpenRouterPkceFlowOwner
from puripuly_heart.core.openrouter_pkce import OpenRouterPKCEExchangeResult
from puripuly_heart.core.runtime.oauth import OAuthRuntime


@pytest.mark.asyncio
async def test_owner_runs_tracks_reopens_and_clears_active_pkce_flow() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    class Client:
        def __init__(self) -> None:
            self.reopen_calls = 0

        async def run_desktop_flow(self) -> OpenRouterPKCEExchangeResult:
            entered.set()
            await release.wait()
            return OpenRouterPKCEExchangeResult(api_key="key", user_id="user")

        def reopen_authorization_url(self) -> bool:
            self.reopen_calls += 1
            return True

    client = Client()
    owner = OpenRouterPkceFlowOwner(client_factory=lambda: client)

    task = asyncio.create_task(owner.run_flow())
    await entered.wait()

    assert owner.active_client is client
    assert owner.reopen_authorization_url() is True
    assert client.reopen_calls == 1

    release.set()
    result = await task

    assert result == OpenRouterPKCEExchangeResult(api_key="key", user_id="user")
    assert owner.active_client is None
    assert owner.get_runtime().active_task_names == ()


def test_owner_reopens_compatibility_client_without_runtime() -> None:
    reopen_calls = 0

    def reopen() -> bool:
        nonlocal reopen_calls
        reopen_calls += 1
        return True

    owner = OpenRouterPkceFlowOwner(client_factory=lambda: object())
    owner.active_client = type("Client", (), {"reopen_authorization_url": staticmethod(reopen)})()

    assert owner.reopen_authorization_url() is True
    assert reopen_calls == 1


@pytest.mark.asyncio
async def test_owner_close_cancels_active_flow_and_clears_client() -> None:
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    class Client:
        async def run_desktop_flow(self) -> OpenRouterPKCEExchangeResult:
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

    owner = OpenRouterPkceFlowOwner(client_factory=Client)
    task = asyncio.create_task(owner.run_flow())
    await entered.wait()

    await owner.close()
    await asyncio.gather(task, return_exceptions=True)

    assert cancelled.is_set()
    assert owner.active_client is None
    assert owner.get_runtime().is_closed is True
    assert task.cancelled()


@pytest.mark.asyncio
async def test_owner_close_clears_client_when_runtime_close_fails() -> None:
    class FailingRuntime:
        async def close(self) -> None:
            raise RuntimeError("close failed")

    owner = OpenRouterPkceFlowOwner(client_factory=lambda: object())
    owner.runtime = cast(OAuthRuntime, FailingRuntime())
    owner.active_client = object()

    with pytest.raises(RuntimeError, match="close failed"):
        await owner.close()

    assert owner.active_client is None
