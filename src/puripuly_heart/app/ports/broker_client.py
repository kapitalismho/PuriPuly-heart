from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Protocol

from puripuly_heart.core.messages import (
    DiagnosticFieldValue,
    ErrorDiagnostics,
    UserMessageRef,
)


def _freeze_fields(
    values: Mapping[str, DiagnosticFieldValue],
) -> Mapping[str, DiagnosticFieldValue]:
    return MappingProxyType(dict(values))


@dataclass(frozen=True, slots=True)
class BrokerIssueRequest:
    discord_user_id: str | None
    local_public_key: str
    local_identity_revision: str | None
    metadata: Mapping[str, DiagnosticFieldValue]
    authorization_code: str | None = field(default=None, repr=False)
    oauth_state: str | None = field(default=None, repr=False)
    redirect_uri: str | None = None
    issue_nonce: str | None = field(default=None, repr=False)
    hardware_hash: str | None = field(default=None, repr=False)
    hardware_hash_salt_version: int | None = None
    operation_id: str | None = None
    resume_token: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_fields(self.metadata))


@dataclass(frozen=True, slots=True)
class BrokerIssueResult:
    succeeded: bool
    broker_connection_id: str | None
    managed_secret_key: str | None = field(repr=False)
    remote_key_revision: str | None
    message: UserMessageRef | None
    diagnostics: ErrorDiagnostics | None
    managed_credential_ref: str | None = None
    expires_at: str | None = None
    openrouter_user_id: str | None = None
    referral_id: str | None = None
    referral_bonus_applied: bool = False
    pass_status: object | None = field(default=None, repr=False)
    delivery_ack: ManagedKeyDeliveryAckMetadata | None = field(default=None, repr=False)
    unknown_outcome: bool = False


@dataclass(frozen=True, slots=True)
class ManagedKeyDeliveryAckMetadata:
    source: str
    delivery_id: str
    managed_credential_ref: str
    expires_at: str | None
    delivery_ack_token: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class ManagedKeyDeliveryAckRequest:
    delivery_id: str
    managed_credential_ref: str
    delivery_ack_token: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class ManagedKeyDeliveryAckResult:
    succeeded: bool
    status: str
    message: UserMessageRef | None = None
    diagnostics: ErrorDiagnostics | None = None
    referral_bonus_applied: bool = False
    referral_id: str | None = None
    pass_status: object | None = field(default=None, repr=False)
    referral_status: str | None = None
    referral_settlement: str | None = None


QqManagedAssertionFailureSubcode = Literal[
    "invalid_credential",
    "mismatch",
    "lifetime_used",
    "rate_limited",
    "key_unavailable",
    "broker_unavailable",
]


@dataclass(frozen=True, slots=True)
class QqManagedAssertionRequest:
    qq_identity: str = field(repr=False)
    credential: str = field(repr=False)
    asserted_at: str
    metadata: Mapping[str, DiagnosticFieldValue]
    referral_id: str | None = None
    installation_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_fields(self.metadata))


@dataclass(frozen=True, slots=True)
class QqManagedEntitlementSnapshot:
    qq_subject_ref: str
    managed_credential_ref: str | None
    expires_at: str | None
    openrouter_user_id: str | None = None


@dataclass(frozen=True, slots=True)
class QqManagedAssertionResult:
    succeeded: bool
    managed_secret_key: str | None = field(repr=False)
    entitlement: QqManagedEntitlementSnapshot | None
    failure_subcode: QqManagedAssertionFailureSubcode | None
    retry_after_ms: int | None
    message: UserMessageRef | None
    diagnostics: ErrorDiagnostics | None
    referral_bonus_applied: bool = False
    referral_id: str | None = None
    pass_status: object | None = field(default=None, repr=False)
    delivery_ack: ManagedKeyDeliveryAckMetadata | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class QqManagedStatusRequest:
    qq_identity: str = field(repr=False)
    credential: str = field(repr=False)
    installation_id: str | None = None


@dataclass(frozen=True, slots=True)
class QqManagedStatusResult:
    referral_id: str | None
    pass_status: object | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class ManagedOperationAttemptSnapshot:
    attempt_index: int | None = None
    provider_key_name: str | None = None
    managed_credential_ref: str | None = None
    outcome: str | None = None


@dataclass(frozen=True, slots=True)
class ManagedOperationDeliverySnapshot:
    delivery_id: str | None = None
    status: str | None = None
    expires_at: str | None = None


@dataclass(frozen=True, slots=True)
class ManagedOperationReferralSnapshot:
    status: str | None = None
    settlement: str | None = None


@dataclass(frozen=True, slots=True)
class ManagedOperationStatusRequest:
    operation_id: str
    installation_id: str | None = None
    resume_token: str | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class ManagedOperationResumeRequest:
    operation_id: str
    installation_id: str | None = None
    resume_token: str | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class ManagedOperationStatusResult:
    succeeded: bool
    operation_status: str
    client_action: str
    message: UserMessageRef | None = None
    diagnostics: ErrorDiagnostics | None = None
    attempt: ManagedOperationAttemptSnapshot | None = None
    delivery: ManagedOperationDeliverySnapshot | None = None
    referral: ManagedOperationReferralSnapshot | None = None
    failed_reason: str | None = None
    managed_secret_key: str | None = field(default=None, repr=False)
    managed_credential_ref: str | None = None
    expires_at: str | None = None
    openrouter_user_id: str | None = None
    referral_id: str | None = None
    referral_bonus_applied: bool = False
    pass_status: object | None = field(default=None, repr=False)
    delivery_ack: ManagedKeyDeliveryAckMetadata | None = field(default=None, repr=False)



class BrokerClientPort(Protocol):
    async def issue_managed_connection(
        self,
        request: BrokerIssueRequest,
    ) -> BrokerIssueResult: ...

    async def assert_qq_managed_identity(
        self,
        request: QqManagedAssertionRequest,
    ) -> QqManagedAssertionResult: ...

    async def get_qq_managed_status(
        self,
        request: QqManagedStatusRequest,
    ) -> QqManagedStatusResult: ...

    async def acknowledge_managed_key_delivery(
        self,
        request: ManagedKeyDeliveryAckRequest,
    ) -> ManagedKeyDeliveryAckResult: ...

    async def get_managed_operation_status(
        self,
        request: ManagedOperationStatusRequest,
    ) -> ManagedOperationStatusResult: ...

    async def resume_managed_operation(
        self,
        request: ManagedOperationResumeRequest,
    ) -> ManagedOperationStatusResult: ...


__all__ = [
    "BrokerClientPort",
    "BrokerIssueRequest",
    "BrokerIssueResult",
    "ManagedKeyDeliveryAckMetadata",
    "ManagedKeyDeliveryAckRequest",
    "ManagedKeyDeliveryAckResult",
    "ManagedOperationAttemptSnapshot",
    "ManagedOperationDeliverySnapshot",
    "ManagedOperationReferralSnapshot",
    "ManagedOperationResumeRequest",
    "ManagedOperationStatusRequest",
    "ManagedOperationStatusResult",
    "QqManagedAssertionFailureSubcode",
    "QqManagedAssertionRequest",
    "QqManagedAssertionResult",
    "QqManagedEntitlementSnapshot",
    "QqManagedStatusRequest",
    "QqManagedStatusResult",
]
