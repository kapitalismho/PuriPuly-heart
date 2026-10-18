from __future__ import annotations

import asyncio
import threading
from datetime import UTC, datetime

import pytest
from puripuly_heart.core.discord_managed_oauth import (
    DiscordOAuthBrowserLaunchError,
    DiscordOAuthStateError,
    _timeout_seconds,
    register_discord_oauth_expected_state,
    run_discord_oauth_callback_flow,
)
from puripuly_heart.core.discord_oauth_loopback import (
    DiscordOAuthCallbackError,
    DiscordOAuthCallbackResult,
)


class FakeListener:
    def __init__(self, result: DiscordOAuthCallbackResult | None = None) -> None:
        self.result = result or DiscordOAuthCallbackResult(
            code="discord-code-1",
            state="discord-state-1",
        )
        self.wait_timeout: float | None = None
        self.wait_thread_id: int | None = None
        self.error: BaseException | None = None
        self.closed = False
        self.expected_state = None

    def close(self) -> None:
        self.closed = True

    def expect_state(self, state: str) -> None:
        self.expected_state = state

    def wait(self, timeout: float | None = None) -> DiscordOAuthCallbackResult:
        self.wait_timeout = timeout
        self.wait_thread_id = threading.get_ident()
        if self.error is not None:
            raise self.error
        return self.result


@pytest.mark.asyncio
async def test_browser_opens_authorization_url_and_returns_code_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listener = FakeListener()
    opened_urls: list[str] = []
    event_loop_thread_id = threading.get_ident()

    monkeypatch.setattr(
        "puripuly_heart.core.discord_managed_oauth.webbrowser.open",
        lambda url: opened_urls.append(url) or True,
    )

    code, state = await run_discord_oauth_callback_flow(
        listener,
        "https://discord.com/oauth2/authorize?client_id=client-1",
        "2099-04-08T06:10:00.000Z",
    )

    assert opened_urls == ["https://discord.com/oauth2/authorize?client_id=client-1"]
    assert (code, state) == ("discord-code-1", "discord-state-1")
    assert listener.wait_timeout == 300
    assert listener.wait_thread_id is not None
    assert listener.wait_thread_id != event_loop_thread_id
    assert listener.closed


def test_timeout_is_clamped_between_one_and_three_hundred_seconds() -> None:
    def now() -> datetime:
        return datetime(2026, 4, 8, 6, 0, 0, tzinfo=UTC)

    assert _timeout_seconds("2026-04-08T06:00:45.000Z", now_provider=now) == 45
    assert _timeout_seconds("2026-04-08T06:10:00.000Z", now_provider=now) == 300
    assert _timeout_seconds("2026-04-08T05:59:59.000Z", now_provider=now) == 1


def test_timeout_invalid_expiry_falls_back_to_three_hundred_seconds() -> None:
    assert _timeout_seconds("not-a-timestamp") == 300


@pytest.mark.asyncio
async def test_listener_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    listener = FakeListener()
    listener.error = DiscordOAuthCallbackError("access_denied", "discord-state-2")
    monkeypatch.setattr(
        "puripuly_heart.core.discord_managed_oauth.webbrowser.open",
        lambda _url: True,
    )

    with pytest.raises(DiscordOAuthCallbackError) as exc_info:
        await run_discord_oauth_callback_flow(
            listener,
            "https://discord.com/oauth2/authorize?client_id=client-1",
            "2026-04-08T06:10:00.000Z",
        )

    assert exc_info.value.error == "access_denied"
    assert exc_info.value.state == "discord-state-2"
    assert listener.closed


@pytest.mark.asyncio
async def test_browser_launch_failure_closes_without_waiting(monkeypatch) -> None:
    listener = FakeListener()
    monkeypatch.setattr(
        "puripuly_heart.core.discord_managed_oauth.webbrowser.open", lambda _: False
    )
    with pytest.raises(DiscordOAuthBrowserLaunchError):
        await run_discord_oauth_callback_flow(listener, "https://discord.invalid/", "invalid")
    assert listener.wait_thread_id is None
    assert listener.closed


@pytest.mark.parametrize("query", ["", "state=", "state=one&state=two"])
def test_invalid_authorization_state_cannot_register(query) -> None:
    listener = FakeListener()
    with pytest.raises(DiscordOAuthStateError):
        register_discord_oauth_expected_state(listener, f"https://discord.invalid/?{query}")
    assert listener.expected_state is None


def test_authorization_state_registers_exact_value() -> None:
    listener = FakeListener()
    register_discord_oauth_expected_state(listener, "https://discord.invalid/?state=valid%2Bstate")
    assert listener.expected_state == "valid+state"


@pytest.mark.asyncio
async def test_timeout_closes_listener_without_masking_error() -> None:
    listener = FakeListener()
    listener.error = TimeoutError("callback did not arrive")
    with pytest.raises(TimeoutError):
        await run_discord_oauth_callback_flow(
            listener, "https://discord.invalid/", "invalid", open_browser=False
        )
    assert listener.closed


@pytest.mark.asyncio
async def test_cancel_settles_callback_wait_and_listener() -> None:
    from puripuly_heart.core.discord_oauth_loopback import bind_first_available

    listener = bind_first_available()
    task = asyncio.create_task(
        run_discord_oauth_callback_flow(
            listener, "https://discord.invalid/", "invalid", open_browser=False
        )
    )
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not listener._thread.is_alive()
    assert listener._server.socket.fileno() == -1


@pytest.mark.asyncio
async def test_actual_callback_flow_ignores_stale_state_and_completes_current_state(monkeypatch):
    import urllib.error
    import urllib.request

    from puripuly_heart.core.discord_oauth_loopback import bind_first_available

    listener = bind_first_available()
    authorization_url = "https://discord.invalid/?state=current"
    register_discord_oauth_expected_state(listener, authorization_url)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    opened = []

    def browser(url):
        opened.append(url)
        with pytest.raises(urllib.error.HTTPError) as stale:
            opener.open(f"{listener.redirect_uri}?code=old&state=stale", timeout=2)
        assert stale.value.code == 400
        with opener.open(
            f"{listener.redirect_uri}?code=valid-code&state=current", timeout=2
        ) as response:
            assert response.status == 200
        return True

    monkeypatch.setattr("puripuly_heart.core.discord_managed_oauth.webbrowser.open", browser)
    try:
        assert await run_discord_oauth_callback_flow(listener, authorization_url, "invalid") == (
            "valid-code",
            "current",
        )
        assert opened == [authorization_url]
        assert not listener._thread.is_alive()
        assert listener._server.socket.fileno() == -1
    finally:
        listener.close()
