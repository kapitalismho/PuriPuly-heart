from __future__ import annotations

import asyncio
from urllib.parse import parse_qs

import httpx
import pytest

from puripuly_heart.core.chatgpt.oauth import CHATGPT_TOKEN_URL
from puripuly_heart.core.chatgpt.session import CHATGPT_REFRESH_TOKEN_SECRET
from puripuly_heart.core.storage.secrets import InMemorySecretStore
from tests.app.test_chatgpt_account import _Flow, _owner


class _ExchangeGate(_Flow):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.authorization_exchanges = 0

    async def handler(self, request: httpx.Request) -> httpx.Response:
        if str(request.url) == CHATGPT_TOKEN_URL:
            form = parse_qs(request.content.decode())
            if form.get("grant_type") == ["authorization_code"]:
                self.authorization_exchanges += 1
                self.started.set()
                await self.release.wait()
        return super().handler(request)


@pytest.mark.asyncio
async def test_cancel_after_callback_cancels_whole_connect_before_credential_store() -> None:
    store = InMemorySecretStore()
    flow = _ExchangeGate()
    owner = _owner(store, flow)
    connecting = asyncio.create_task(owner.connect())
    try:
        await asyncio.wait_for(flow.started.wait(), timeout=2)
        progress_during_exchange = owner.in_progress
        owner.cancel()
        flow.release.set()
        result = await asyncio.wait_for(connecting, timeout=2)
        assert (
            progress_during_exchange,
            result.succeeded,
            result.failure_code,
            store.get(CHATGPT_REFRESH_TOKEN_SECRET),
            owner.in_progress,
        ) == (True, False, "cancelled", None, False)
    finally:
        flow.release.set()
        await asyncio.gather(connecting, return_exceptions=True)
        await owner.close()


@pytest.mark.asyncio
async def test_callback_does_not_reopen_connect_admission_before_prior_settlement() -> None:
    store = InMemorySecretStore()
    flow = _ExchangeGate()
    owner = _owner(store, flow)
    first = asyncio.create_task(owner.connect())
    second = None
    try:
        await asyncio.wait_for(flow.started.wait(), timeout=2)
        progress_during_exchange = owner.in_progress
        second = asyncio.create_task(owner.connect())
        for _ in range(100):
            if second.done() or flow.authorization_exchanges >= 2:
                break
            await asyncio.sleep(0.01)
        flow.release.set()
        first_result, second_result = await asyncio.gather(first, second)
        assert (
            progress_during_exchange,
            first_result.succeeded,
            second_result.succeeded,
            second_result.failure_code,
            flow.authorization_exchanges,
        ) == (True, True, False, "in_progress", 1)
    finally:
        flow.release.set()
        tasks = [first] if second is None else [first, second]
        await asyncio.gather(*tasks, return_exceptions=True)
        await owner.close()


@pytest.mark.asyncio
async def test_cancel_after_store_start_drains_and_reports_actual_success(monkeypatch) -> None:
    import threading

    store = InMemorySecretStore()
    flow = _Flow()
    owner = _owner(store, flow)
    original_store = owner.session.store_sign_in
    store_started = threading.Event()
    release_store = threading.Event()

    def held_store(_session, **kwargs) -> None:
        store_started.set()
        if not release_store.wait(timeout=2):
            raise TimeoutError("test store gate was not released")
        original_store(**kwargs)

    monkeypatch.setattr(type(owner.session), "store_sign_in", held_store)
    connecting = asyncio.create_task(owner.connect())
    try:
        for _ in range(200):
            if store_started.is_set():
                break
            await asyncio.sleep(0.01)
        assert store_started.is_set()
        owner.cancel()
        await asyncio.sleep(0)
        assert not connecting.done()
        release_store.set()
        result = await asyncio.wait_for(connecting, timeout=2)
        assert result.succeeded
        assert result.failure_code is None
        assert store.get(CHATGPT_REFRESH_TOKEN_SECRET) == "refresh"
    finally:
        release_store.set()
        await asyncio.gather(connecting, return_exceptions=True)
        await owner.close()


@pytest.mark.asyncio
async def test_close_cancels_exchange_before_closing_http_session() -> None:
    store = InMemorySecretStore()
    flow = _ExchangeGate()
    owner = _owner(store, flow)
    connecting = asyncio.create_task(owner.connect())
    closing = None
    try:
        await asyncio.wait_for(flow.started.wait(), timeout=2)
        closing = asyncio.create_task(owner.close())
        await asyncio.sleep(0)
        flow.release.set()
        result = await asyncio.wait_for(connecting, timeout=2)
        await asyncio.wait_for(closing, timeout=2)
        assert not result.succeeded
        assert result.failure_code == "cancelled"
        assert store.get(CHATGPT_REFRESH_TOKEN_SECRET) is None
        assert not owner.in_progress
    finally:
        flow.release.set()
        await asyncio.gather(connecting, return_exceptions=True)
        if closing is None:
            await owner.close()
        else:
            await asyncio.gather(closing, return_exceptions=True)
