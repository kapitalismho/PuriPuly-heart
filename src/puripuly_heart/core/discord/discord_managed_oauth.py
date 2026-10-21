from __future__ import annotations

import asyncio
import webbrowser
from collections.abc import Callable
from datetime import UTC, datetime
from urllib.parse import parse_qs, urlsplit

from puripuly_heart.core.owned_thread import run_owned_thread_call

from .discord_oauth_loopback import DiscordOAuthLoopbackListener

MAX_DISCORD_OAUTH_WAIT_SECONDS = 300.0
MIN_DISCORD_OAUTH_WAIT_SECONDS = 1.0


class DiscordOAuthBrowserLaunchError(RuntimeError):
    pass


class DiscordOAuthStateError(ValueError):
    pass


def register_discord_oauth_expected_state(
    listener: DiscordOAuthLoopbackListener, authorization_url: str
) -> None:
    try:
        states = parse_qs(urlsplit(authorization_url).query, keep_blank_values=True).get("state")
        if states is None or len(states) != 1 or not states[0]:
            raise DiscordOAuthStateError("Discord OAuth authorization state is missing or invalid")
        listener.expect_state(states[0])
    except ValueError as exc:
        raise DiscordOAuthStateError(
            "Discord OAuth authorization state is missing or invalid"
        ) from exc


async def run_discord_oauth_callback_flow(
    listener: DiscordOAuthLoopbackListener,
    authorization_url: str,
    expires_at: str,
    *,
    open_browser: bool = True,
    authorization_url_sink: Callable[[str], None] | None = None,
) -> tuple[str, str]:
    callback_future = None
    try:
        if authorization_url_sink is not None:
            authorization_url_sink(authorization_url)
        if open_browser:
            try:
                opened = webbrowser.open(authorization_url)
            except (webbrowser.Error, OSError) as exc:
                raise DiscordOAuthBrowserLaunchError(
                    "Discord authorization browser could not open"
                ) from exc
            if not opened:
                raise DiscordOAuthBrowserLaunchError("Discord authorization browser could not open")
        callback_future = asyncio.get_running_loop().run_in_executor(
            None, listener.wait, _timeout_seconds(expires_at)
        )
        callback_result = await asyncio.shield(callback_future)
        return callback_result.code, callback_result.state
    finally:
        try:
            await run_owned_thread_call(listener.close)
        finally:
            if callback_future is not None:
                while not callback_future.done():
                    try:
                        await asyncio.shield(callback_future)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
                if not callback_future.cancelled():
                    callback_future.exception()


def _timeout_seconds(
    expires_at: str,
    *,
    now_provider: Callable[[], datetime] | None = None,
) -> float:
    expires_at_utc = _parse_expiry(expires_at)
    if expires_at_utc is None:
        return MAX_DISCORD_OAUTH_WAIT_SECONDS

    now = (now_provider or (lambda: datetime.now(UTC)))()
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    else:
        now = now.astimezone(UTC)

    remaining_seconds = (expires_at_utc - now).total_seconds()
    return min(
        MAX_DISCORD_OAUTH_WAIT_SECONDS,
        max(MIN_DISCORD_OAUTH_WAIT_SECONDS, remaining_seconds),
    )


def _parse_expiry(value: str) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
