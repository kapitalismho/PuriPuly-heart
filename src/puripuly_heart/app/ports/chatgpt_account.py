from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from puripuly_heart.core.messages import ErrorDiagnostics

CHATGPT_USAGE_SETTINGS_URL: Final = "https://chatgpt.com/settings/usage"


@dataclass(frozen=True, slots=True)
class ChatGptAccountSnapshot:
    signed_in: bool | None
    email: str | None = None
    in_progress: bool = False
    diagnostics: ErrorDiagnostics | None = None


@dataclass(frozen=True, slots=True)
class ChatGptConnectResult:
    succeeded: bool
    first_sign_in: bool = False
    email: str | None = None
    failure_code: str | None = None
    diagnostics: ErrorDiagnostics | None = None
    secondary_diagnostics: tuple[ErrorDiagnostics, ...] = ()


@dataclass(frozen=True, slots=True)
class ChatGptSignOutResult:
    remote_revoked: bool
    local_credentials_removed: bool | None = None
    diagnostics: ErrorDiagnostics | None = None
    secondary_diagnostics: tuple[ErrorDiagnostics, ...] = ()


__all__ = [
    "CHATGPT_USAGE_SETTINGS_URL",
    "ChatGptAccountSnapshot",
    "ChatGptConnectResult",
    "ChatGptSignOutResult",
]
