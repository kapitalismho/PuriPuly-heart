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
    secondary_omitted_count: int = 0

    def __post_init__(self) -> None:
        excess = max(0, len(self.secondary_diagnostics) - 4)
        if excess:
            object.__setattr__(self, "secondary_diagnostics", self.secondary_diagnostics[:4])
            object.__setattr__(
                self, "secondary_omitted_count", self.secondary_omitted_count + excess
            )


@dataclass(frozen=True, slots=True)
class ChatGptSignOutResult:
    remote_revoked: bool
    local_credentials_removed: bool | None = None
    diagnostics: ErrorDiagnostics | None = None
    secondary_diagnostics: tuple[ErrorDiagnostics, ...] = ()
    secondary_omitted_count: int = 0

    def __post_init__(self) -> None:
        excess = max(0, len(self.secondary_diagnostics) - 4)
        if excess:
            object.__setattr__(self, "secondary_diagnostics", self.secondary_diagnostics[:4])
            object.__setattr__(
                self, "secondary_omitted_count", self.secondary_omitted_count + excess
            )


__all__ = [
    "CHATGPT_USAGE_SETTINGS_URL",
    "ChatGptAccountSnapshot",
    "ChatGptConnectResult",
    "ChatGptSignOutResult",
]
