from __future__ import annotations

import asyncio
import webbrowser
from collections.abc import Callable
from datetime import UTC, datetime

from .discord_oauth_loopback import DiscordOAuthLoopbackListener

MAX_DISCORD_OAUTH_WAIT_SECONDS = 300.0
MIN_DISCORD_OAUTH_WAIT_SECONDS = 1.0


async def run_discord_oauth_callback_flow(
    listener: DiscordOAuthLoopbackListener,
    authorization_url: str,
    expires_at: str,
) -> tuple[str, str]:
    webbrowser.open(authorization_url)
    timeout = _timeout_seconds(expires_at)
    callback_result = await asyncio.to_thread(listener.wait, timeout)
    return callback_result.code, callback_result.state


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
