from importlib import import_module as _import_module

_SUBMODULES = frozenset(
    {
        "discord_managed_oauth",
        "discord_oauth_loopback",
        "oauth_callback_page",
    }
)
_EXPORT_SOURCES = {
    "bind_first_available": "discord_oauth_loopback",
    "DISCORD_OAUTH_LOOPBACK_HOST": "discord_oauth_loopback",
    "DISCORD_OAUTH_LOOPBACK_PATH": "discord_oauth_loopback",
    "DISCORD_OAUTH_LOOPBACK_PORTS": "discord_oauth_loopback",
    "DiscordOAuthCallbackError": "discord_oauth_loopback",
    "DiscordOAuthCallbackResult": "discord_oauth_loopback",
    "DiscordOAuthLoopbackClosedError": "discord_oauth_loopback",
    "DiscordOAuthLoopbackListener": "discord_oauth_loopback",
    "MAX_DISCORD_OAUTH_WAIT_SECONDS": "discord_managed_oauth",
    "MIN_DISCORD_OAUTH_WAIT_SECONDS": "discord_managed_oauth",
    "OAUTH_CALLBACK_COMPLETION_FALLBACK_LINES": "oauth_callback_page",
    "OAUTH_CALLBACK_COMPLETION_LINE_KEYS": "oauth_callback_page",
    "OAUTH_CALLBACK_FONT_FAMILIES": "oauth_callback_page",
    "OAUTH_CALLBACK_TITLE_KEY": "oauth_callback_page",
    "render_discord_oauth_callback_completion_page": "discord_oauth_loopback",
    "render_oauth_callback_completion_page": "oauth_callback_page",
    "resolve_oauth_callback_locale": "oauth_callback_page",
    "run_discord_oauth_callback_flow": "discord_managed_oauth",
}

__all__ = [
    "bind_first_available",
    "discord_managed_oauth",
    "discord_oauth_loopback",
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
    "oauth_callback_page",
    "OAUTH_CALLBACK_TITLE_KEY",
    "render_discord_oauth_callback_completion_page",
    "render_oauth_callback_completion_page",
    "resolve_oauth_callback_locale",
    "run_discord_oauth_callback_flow",
]


def __getattr__(name: str) -> object:
    if name in _SUBMODULES:
        return _import_module(f".{name}", __name__)
    source = _EXPORT_SOURCES.get(name)
    if source is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(_import_module(f".{source}", __name__), name)
