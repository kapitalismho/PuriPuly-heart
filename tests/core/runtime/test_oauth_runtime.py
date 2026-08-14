from __future__ import annotations

import asyncio
import contextlib
import threading
from dataclasses import dataclass

import pytest
from puripuly_heart.core.openrouter_pkce import OpenRouterPKCEExchangeResult

from puripuly_heart.core.runtime import OAuthRuntime


class BlockingListener:
    redirect_uri = "http://127.0.0.1:62187/discord/callback"

    def __init__(self) -> None:
        self.wait_started = threading.Event()
        self.wait_finished = threading.Event()
        self.closed = threading.Event()
        self.close_calls = 0

    def wait(self, timeout: float | None = None) -> object:
        _ = timeout
        self.wait_started.set()
        self.closed.wait(timeout=5.0)
        self.wait_finished.set()
        raise RuntimeError("listener closed")

    def close(self) -> None:
        self.close_calls += 1
        self.closed.set()


class RetriableFailingListener:
    def __init__(self) -> None:
        self.close_calls = 0
        self.closed = False

    def close(self) -> None:
        self.close_calls += 1
        if self.close_calls == 1:
            raise RuntimeError("listener close failed")
        self.closed = True


class ClosingListener:
    def __init__(self) -> None:
        self.close_calls = 0
        self.closed = False

    def close(self) -> None:
        self.close_calls += 1
        self.closed = True


class ExternalTaskHandle:
    def __init__(self) -> None:
        self.cancel_calls = 0
        self.cancelled = False

    def cancel(self) -> None:
        self.cancel_calls += 1
        self.cancelled = True


@pytest.mark.asyncio
async def test_oauth_runtime_close_unblocks_attached_loopback_listener_wait() -> None:
    runtime = OAuthRuntime(auth_task_timeout_s=0.05)
    listener = BlockingListener()
    runtime.attach_loopback_listener(listener, listener_name="discord")

    async def wait_for_listener() -> None:
        await asyncio.to_thread(listener.wait, 10.0)

    task = runtime.create_auth_task(wait_for_listener(), task_name="discord-prepare")
    assert task.get_name() == "OAuthRuntime:discord-prepare"
    assert await asyncio.to_thread(listener.wait_started.wait, 1.0) is True

    await runtime.close()

    assert listener.close_calls == 1
    assert await asyncio.to_thread(listener.wait_finished.wait, 1.0) is True
    assert runtime.active_task_names == ()
    assert runtime.active_listener_names == ()


@pytest.mark.asyncio
async def test_oauth_runtime_close_failure_keeps_retry_and_continues_cleanup() -> None:
    runtime = OAuthRuntime(auth_task_timeout_s=0.01)
    failing_listener = RetriableFailingListener()
    closing_listener = ClosingListener()
    external_handle = ExternalTaskHandle()
    auth_cancelled = asyncio.Event()

    async def auth_task() -> None:
        try:
            await asyncio.sleep(999)
        except asyncio.CancelledError:
            auth_cancelled.set()
            raise

    runtime.attach_loopback_listener(failing_listener, listener_name="failing")
    runtime.attach_loopback_listener(closing_listener, listener_name="closing")
    runtime.create_auth_task(auth_task(), task_name="auth")
    runtime.start_external_task(
        task_runner=lambda _task_factory: external_handle,
        task_factory=lambda: asyncio.sleep(0),
        task_name="discord-ui",
    )

    with pytest.raises(RuntimeError, match="listener close failed"):
        await runtime.close()

    assert failing_listener.close_calls == 1
    assert closing_listener.closed is True
    assert external_handle.cancel_calls == 1
    assert auth_cancelled.is_set() is True
    assert runtime.active_task_names == ()
    assert runtime.external_task_names == ()
    assert runtime.active_listener_names == ("failing",)
    assert runtime.is_closed is True
    assert runtime.is_closing is False

    await runtime.close()

    assert failing_listener.close_calls == 2
    assert failing_listener.closed is True
    assert runtime.active_listener_names == ()


@pytest.mark.asyncio
async def test_oauth_runtime_close_returns_when_auth_task_suppresses_cancellation() -> None:
    runtime = OAuthRuntime(auth_task_timeout_s=0.01)
    auth_started = asyncio.Event()
    cancel_seen = asyncio.Event()
    release_stubborn_auth = asyncio.Event()

    async def stubborn_auth_task() -> None:
        auth_started.set()
        while not release_stubborn_auth.is_set():
            try:
                await release_stubborn_auth.wait()
            except asyncio.CancelledError:
                cancel_seen.set()

    runtime.create_auth_task(stubborn_auth_task(), task_name="stubborn")
    await auth_started.wait()

    close_task = asyncio.create_task(runtime.close())
    try:
        await asyncio.wait_for(asyncio.shield(close_task), timeout=0.2)
    finally:
        release_stubborn_auth.set()
        with contextlib.suppress(Exception):
            await close_task

    assert cancel_seen.is_set() is True
    assert runtime.active_task_names == ()
    assert runtime.is_closed is True
    assert runtime.is_closing is False


@pytest.mark.asyncio
async def test_oauth_runtime_rejects_new_auth_tasks_after_close() -> None:
    runtime = OAuthRuntime()
    await runtime.close()

    async def never_started() -> None:
        await asyncio.sleep(999)

    with pytest.raises(RuntimeError, match="closed"):
        runtime.create_auth_task(never_started(), task_name="late-auth")


@dataclass
class FakePKCEClient:
    started: asyncio.Event
    release: asyncio.Event
    reopen_calls: int = 0

    async def run_desktop_flow(self) -> OpenRouterPKCEExchangeResult:
        self.started.set()
        await self.release.wait()
        return OpenRouterPKCEExchangeResult(api_key="pkce-key", user_id="user-1")

    def reopen_authorization_url(self) -> bool:
        self.reopen_calls += 1
        return True


@pytest.mark.asyncio
async def test_openrouter_pkce_flow_is_owned_and_reopenable_only_while_active() -> None:
    runtime = OAuthRuntime()
    client = FakePKCEClient(started=asyncio.Event(), release=asyncio.Event())

    task = asyncio.create_task(runtime.run_openrouter_pkce_flow(client))
    await client.started.wait()

    assert runtime.active_task_names == ("openrouter-pkce",)
    assert runtime.reopen_openrouter_pkce_authorization_url() is True
    assert client.reopen_calls == 1

    client.release.set()
    result = await task

    assert result == OpenRouterPKCEExchangeResult(api_key="pkce-key", user_id="user-1")
    assert runtime.active_task_names == ()
    assert runtime.reopen_openrouter_pkce_authorization_url() is False


@pytest.mark.asyncio
async def test_openrouter_pkce_rejection_does_not_leak_active_client() -> None:
    runtime = OAuthRuntime()
    await runtime.close()
    client = FakePKCEClient(started=asyncio.Event(), release=asyncio.Event())

    with pytest.raises(RuntimeError, match="closed"):
        await runtime.run_openrouter_pkce_flow(client)

    assert runtime.reopen_openrouter_pkce_authorization_url() is False
    assert client.reopen_calls == 0


@pytest.mark.asyncio
async def test_openrouter_pkce_duplicate_rejection_preserves_existing_active_client() -> None:
    runtime = OAuthRuntime()
    first_client = FakePKCEClient(started=asyncio.Event(), release=asyncio.Event())
    second_client = FakePKCEClient(started=asyncio.Event(), release=asyncio.Event())

    first_task = asyncio.create_task(runtime.run_openrouter_pkce_flow(first_client))
    await first_client.started.wait()
    try:
        with pytest.raises(RuntimeError, match="already owns auth task"):
            await runtime.run_openrouter_pkce_flow(second_client)

        assert runtime.reopen_openrouter_pkce_authorization_url() is True
        assert first_client.reopen_calls == 1
        assert second_client.reopen_calls == 0
    finally:
        first_client.release.set()
        await first_task


def test_oauth_runtime_rejects_duplicate_external_task_without_orphaning_first() -> None:
    runtime = OAuthRuntime()
    first_handle = ExternalTaskHandle()
    second_handle = ExternalTaskHandle()

    runtime.start_external_task(
        task_runner=lambda _task_factory: first_handle,
        task_factory=lambda: asyncio.sleep(0),
        task_name="discord-managed-auth-dialog",
    )

    with pytest.raises(RuntimeError, match="already owns external task"):
        runtime.start_external_task(
            task_runner=lambda _task_factory: second_handle,
            task_factory=lambda: asyncio.sleep(0),
            task_name="discord-managed-auth-dialog",
        )

    assert runtime.external_task_names == ("discord-managed-auth-dialog",)
    assert first_handle.cancel_calls == 0
    assert second_handle.cancel_calls == 0


def test_oauth_runtime_task_runner_failure_does_not_mutate_external_state() -> None:
    runtime = OAuthRuntime()

    def failing_runner(_task_factory):  # type: ignore[no-untyped-def]
        raise RuntimeError("runner failed")

    with pytest.raises(RuntimeError, match="runner failed"):
        runtime.start_external_task(
            task_runner=failing_runner,
            task_factory=lambda: asyncio.sleep(0),
            task_name="discord-managed-auth-dialog",
        )

    assert runtime.external_task_names == ()
