from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, Literal, Protocol

from puripuly_heart.core.messages import (
    DiagnosticFieldValue,
    ErrorDiagnostics,
    UserMessageRef,
)
from puripuly_heart.core.openrouter_metadata import OpenRouterKeyMetadata

ProviderVerificationStatus = Literal["verified", "failed", "skipped"]
PROVIDER_VERIFICATION_STATUS_VERIFIED: Final[ProviderVerificationStatus] = "verified"
PROVIDER_VERIFICATION_STATUS_FAILED: Final[ProviderVerificationStatus] = "failed"
PROVIDER_VERIFICATION_STATUS_SKIPPED: Final[ProviderVerificationStatus] = "skipped"
PROVIDER_VERIFICATION_STATUSES: Final[tuple[ProviderVerificationStatus, ...]] = (
    PROVIDER_VERIFICATION_STATUS_VERIFIED,
    PROVIDER_VERIFICATION_STATUS_FAILED,
    PROVIDER_VERIFICATION_STATUS_SKIPPED,
)


def _freeze_fields(
    values: Mapping[str, DiagnosticFieldValue],
) -> Mapping[str, DiagnosticFieldValue]:
    return MappingProxyType(dict(values))


@dataclass(frozen=True, slots=True)
class ProviderVerificationRequest:
    provider: str
    secret_key: str
    secret_value: str = field(repr=False)
    secret_revision: str | None
    context: Mapping[str, DiagnosticFieldValue]

    def __post_init__(self) -> None:
        object.__setattr__(self, "context", _freeze_fields(self.context))


@dataclass(frozen=True, slots=True)
class ProviderVerificationResult:
    status: ProviderVerificationStatus
    provider: str
    secret_key: str
    secret_revision: str | None
    evidence: Mapping[str, DiagnosticFieldValue]
    message: UserMessageRef | None
    diagnostics: ErrorDiagnostics | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence", _freeze_fields(self.evidence))


class ProviderVerifierPort(Protocol):
    async def verify_api_key(
        self,
        provider: str,
        api_key: str,
        *,
        model: str | None = None,
        base_url: str | None = None,
        low_latency: bool = False,
    ) -> bool: ...

    async def verify_qwen_llm_api_key(
        self,
        api_key: str,
        *,
        base_url: str,
        model: str | None,
        low_latency: bool,
    ) -> bool: ...

    async def fetch_openrouter_key_metadata(
        self,
        api_key: str,
    ) -> OpenRouterKeyMetadata | None: ...

    async def verify_provider_secret(
        self,
        request: ProviderVerificationRequest,
    ) -> ProviderVerificationResult: ...


__all__ = [
    "PROVIDER_VERIFICATION_STATUSES",
    "PROVIDER_VERIFICATION_STATUS_FAILED",
    "PROVIDER_VERIFICATION_STATUS_SKIPPED",
    "PROVIDER_VERIFICATION_STATUS_VERIFIED",
    "ProviderVerificationRequest",
    "ProviderVerificationResult",
    "ProviderVerificationStatus",
    "ProviderVerifierPort",
]
