import sys as _sys

from puripuly_heart import core as _parent

from . import discord_managed_oauth as discord_managed_oauth
from . import discord_oauth_loopback as discord_oauth_loopback
from . import oauth_callback_page as oauth_callback_page
from .discord_managed_oauth import (
    MAX_DISCORD_OAUTH_WAIT_SECONDS,
    MIN_DISCORD_OAUTH_WAIT_SECONDS,
    run_discord_oauth_callback_flow,
)
from .discord_oauth_loopback import (
    DISCORD_OAUTH_LOOPBACK_HOST,
    DISCORD_OAUTH_LOOPBACK_PATH,
    DISCORD_OAUTH_LOOPBACK_PORTS,
    DiscordOAuthCallbackError,
    DiscordOAuthCallbackResult,
    DiscordOAuthLoopbackClosedError,
    DiscordOAuthLoopbackListener,
    bind_first_available,
    render_discord_oauth_callback_completion_page,
)
from .oauth_callback_page import (
    OAUTH_CALLBACK_COMPLETION_FALLBACK_LINES,
    OAUTH_CALLBACK_COMPLETION_LINE_KEYS,
    OAUTH_CALLBACK_FONT_FAMILIES,
    OAUTH_CALLBACK_TITLE_KEY,
    render_oauth_callback_completion_page,
    resolve_oauth_callback_locale,
)

__all__ = [
    "DISCORD_OAUTH_LOOPBACK_HOST",
    "DISCORD_OAUTH_LOOPBACK_PATH",
    "DISCORD_OAUTH_LOOPBACK_PORTS",
    "DiscordOAuthCallbackError",
    "DiscordOAuthCallbackResult",
    "DiscordOAuthLoopbackClosedError",
    "DiscordOAuthLoopbackListener",
    "MAX_DISCORD_OAUTH_WAIT_SECONDS",
    "MIN_DISCORD_OAUTH_WAIT_SECONDS",
    "OAUTH_CALLBACK_COMPLETION_FALLBACK_LINES",
    "OAUTH_CALLBACK_COMPLETION_LINE_KEYS",
    "OAUTH_CALLBACK_FONT_FAMILIES",
    "OAUTH_CALLBACK_TITLE_KEY",
    "bind_first_available",
    "discord_managed_oauth",
    "discord_oauth_loopback",
    "oauth_callback_page",
    "render_discord_oauth_callback_completion_page",
    "render_oauth_callback_completion_page",
    "resolve_oauth_callback_locale",
    "run_discord_oauth_callback_flow",
]

_COMPAT_MODULES = {
    "discord_managed_oauth": discord_managed_oauth,
    "discord_oauth_loopback": discord_oauth_loopback,
    "oauth_callback_page": oauth_callback_page,
}
for _name, _module in _COMPAT_MODULES.items():
    _sys.modules[f"puripuly_heart.core.{_name}"] = _module
    setattr(_parent, _name, _module)
del _COMPAT_MODULES, _module, _name
