from __future__ import annotations

import threading
from datetime import UTC, datetime

import pytest
from puripuly_heart.core.discord_managed_oauth import (
    _timeout_seconds,
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
