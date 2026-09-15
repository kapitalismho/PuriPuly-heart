from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

SafeMessageParam: TypeAlias = str | int | float | bool | None
DiagnosticFieldValue: TypeAlias = str | int | float | bool | None

SAFE_MESSAGE_PARAM_KEY_MAX_LENGTH: Final = 64
SAFE_MESSAGE_PARAM_VALUE_MAX_LENGTH: Final = 256
SAFE_MESSAGE_PARAM_MAX_ITEMS: Final = 16

DIAGNOSTIC_FIELD_KEY_MAX_LENGTH: Final = 64
DIAGNOSTIC_FIELD_VALUE_MAX_LENGTH: Final = 1024
DIAGNOSTIC_FIELD_MAX_ITEMS: Final = 32

Severity: TypeAlias = Literal["info", "warning", "error"]
SEVERITY_INFO: Final[Severity] = "info"
SEVERITY_WARNING: Final[Severity] = "warning"
SEVERITY_ERROR: Final[Severity] = "error"
SEVERITIES: Final[tuple[Severity, ...]] = (
    SEVERITY_INFO,
    SEVERITY_WARNING,
    SEVERITY_ERROR,
)

DiagnosticCategory: TypeAlias = Literal[
    "auth",
    "network",
    "timeout",
    "quota",
    "rate_limit",
    "invalid_response",
    "service_unavailable",
    "lifecycle",
    "transaction",
    "unknown",
]
DIAGNOSTIC_CATEGORY_AUTH: Final[DiagnosticCategory] = "auth"
DIAGNOSTIC_CATEGORY_NETWORK: Final[DiagnosticCategory] = "network"
DIAGNOSTIC_CATEGORY_TIMEOUT: Final[DiagnosticCategory] = "timeout"
DIAGNOSTIC_CATEGORY_QUOTA: Final[DiagnosticCategory] = "quota"
DIAGNOSTIC_CATEGORY_RATE_LIMIT: Final[DiagnosticCategory] = "rate_limit"
DIAGNOSTIC_CATEGORY_INVALID_RESPONSE: Final[DiagnosticCategory] = "invalid_response"
DIAGNOSTIC_CATEGORY_SERVICE_UNAVAILABLE: Final[DiagnosticCategory] = "service_unavailable"
DIAGNOSTIC_CATEGORY_LIFECYCLE: Final[DiagnosticCategory] = "lifecycle"
DIAGNOSTIC_CATEGORY_TRANSACTION: Final[DiagnosticCategory] = "transaction"
DIAGNOSTIC_CATEGORY_UNKNOWN: Final[DiagnosticCategory] = "unknown"
DIAGNOSTIC_CATEGORIES: Final[tuple[DiagnosticCategory, ...]] = (
    DIAGNOSTIC_CATEGORY_AUTH,
    DIAGNOSTIC_CATEGORY_NETWORK,
    DIAGNOSTIC_CATEGORY_TIMEOUT,
    DIAGNOSTIC_CATEGORY_QUOTA,
    DIAGNOSTIC_CATEGORY_RATE_LIMIT,
    DIAGNOSTIC_CATEGORY_INVALID_RESPONSE,
    DIAGNOSTIC_CATEGORY_SERVICE_UNAVAILABLE,
    DIAGNOSTIC_CATEGORY_LIFECYCLE,
    DIAGNOSTIC_CATEGORY_TRANSACTION,
    DIAGNOSTIC_CATEGORY_UNKNOWN,
)

DiagnosticVisibility: TypeAlias = Literal[
    "basic",
    "diagnostic_only",
    "persisted_failure_only",
]
DIAGNOSTIC_VISIBILITY_BASIC: Final[DiagnosticVisibility] = "basic"
DIAGNOSTIC_VISIBILITY_DIAGNOSTIC_ONLY: Final[DiagnosticVisibility] = "diagnostic_only"
DIAGNOSTIC_VISIBILITY_PERSISTED_FAILURE_ONLY: Final[DiagnosticVisibility] = "persisted_failure_only"
DIAGNOSTIC_VISIBILITIES: Final[tuple[DiagnosticVisibility, ...]] = (
    DIAGNOSTIC_VISIBILITY_BASIC,
    DIAGNOSTIC_VISIBILITY_DIAGNOSTIC_ONLY,
    DIAGNOSTIC_VISIBILITY_PERSISTED_FAILURE_ONLY,
)

ContentPolicy: TypeAlias = Literal[
    "metadata_only",
    "redacted",
    "raw_user_text_allowed",
    "secret_forbidden",
]
CONTENT_POLICY_METADATA_ONLY: Final[ContentPolicy] = "metadata_only"
CONTENT_POLICY_REDACTED: Final[ContentPolicy] = "redacted"
CONTENT_POLICY_RAW_USER_TEXT_ALLOWED: Final[ContentPolicy] = "raw_user_text_allowed"
CONTENT_POLICY_SECRET_FORBIDDEN: Final[ContentPolicy] = "secret_forbidden"
CONTENT_POLICIES: Final[tuple[ContentPolicy, ...]] = (
    CONTENT_POLICY_METADATA_ONLY,
    CONTENT_POLICY_REDACTED,
    CONTENT_POLICY_RAW_USER_TEXT_ALLOWED,
    CONTENT_POLICY_SECRET_FORBIDDEN,
)

RuntimeApplyStatus: TypeAlias = Literal["applied", "degraded", "failed"]
RUNTIME_APPLY_STATUS_APPLIED: Final[RuntimeApplyStatus] = "applied"
RUNTIME_APPLY_STATUS_DEGRADED: Final[RuntimeApplyStatus] = "degraded"
RUNTIME_APPLY_STATUS_FAILED: Final[RuntimeApplyStatus] = "failed"
RUNTIME_APPLY_RESULT_STATUSES: Final[tuple[RuntimeApplyStatus, ...]] = (
    RUNTIME_APPLY_STATUS_APPLIED,
    RUNTIME_APPLY_STATUS_DEGRADED,
    RUNTIME_APPLY_STATUS_FAILED,
)

TransactionStatus: TypeAlias = Literal[
    "settings_commit_failed",
    "settings_commit_success_runtime_applied",
    "settings_commit_success_runtime_degraded",
    "secret_write_failed",
    "provider_verification_failed",
    "settings_commit_failed_secret_restored",
    "settings_commit_failed_secret_restore_failed",
    "remote_active_local_missing",
    "remote_delivery_ack_pending",
]
TRANSACTION_STATUS_SETTINGS_COMMIT_FAILED: Final[TransactionStatus] = "settings_commit_failed"
TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED: Final[TransactionStatus] = (
    "settings_commit_success_runtime_applied"
)
TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED: Final[TransactionStatus] = (
    "settings_commit_success_runtime_degraded"
)
TRANSACTION_STATUS_SECRET_WRITE_FAILED: Final[TransactionStatus] = "secret_write_failed"
TRANSACTION_STATUS_PROVIDER_VERIFICATION_FAILED: Final[TransactionStatus] = (
    "provider_verification_failed"
)
TRANSACTION_STATUS_SETTINGS_COMMIT_FAILED_SECRET_RESTORED: Final[TransactionStatus] = (
    "settings_commit_failed_secret_restored"
)
TRANSACTION_STATUS_SETTINGS_COMMIT_FAILED_SECRET_RESTORE_FAILED: Final[TransactionStatus] = (
    "settings_commit_failed_secret_restore_failed"
)
TRANSACTION_STATUS_REMOTE_ACTIVE_LOCAL_MISSING: Final[TransactionStatus] = (
    "remote_active_local_missing"
)
TRANSACTION_STATUS_REMOTE_DELIVERY_ACK_PENDING: Final[TransactionStatus] = (
    "remote_delivery_ack_pending"
)
TRANSACTION_RESULT_STATUSES: Final[tuple[TransactionStatus, ...]] = (
    TRANSACTION_STATUS_SETTINGS_COMMIT_FAILED,
    TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
    TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED,
    TRANSACTION_STATUS_SECRET_WRITE_FAILED,
    TRANSACTION_STATUS_PROVIDER_VERIFICATION_FAILED,
    TRANSACTION_STATUS_SETTINGS_COMMIT_FAILED_SECRET_RESTORED,
    TRANSACTION_STATUS_SETTINGS_COMMIT_FAILED_SECRET_RESTORE_FAILED,
    TRANSACTION_STATUS_REMOTE_ACTIVE_LOCAL_MISSING,
    TRANSACTION_STATUS_REMOTE_DELIVERY_ACK_PENDING,
)


def _freeze_safe_mapping(
    values: Mapping[str, SafeMessageParam],
) -> Mapping[str, SafeMessageParam]:
    return MappingProxyType(dict(values))


@dataclass(frozen=True, slots=True)
class UserMessageRef:
    key: str
    params: Mapping[str, SafeMessageParam]
    severity: Severity

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", _freeze_safe_mapping(self.params))


@dataclass(frozen=True, slots=True)
class ErrorDiagnostics:
    component: str
    operation: str | None
    code: str | None
    category: DiagnosticCategory
    visibility: DiagnosticVisibility
    content_policy: ContentPolicy
    status_code: int | None
    retry_after_ms: int | None
    fields: Mapping[str, DiagnosticFieldValue]

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", _freeze_safe_mapping(self.fields))


@dataclass(frozen=True, slots=True)
class UserErrorReport:
    message: UserMessageRef
    diagnostics: ErrorDiagnostics

    def __str__(self) -> str:
        code = self.diagnostics.code or "unknown"
        return f"{self.message.key} (category={self.diagnostics.category}, code={code})"


@dataclass(frozen=True, slots=True)
class RuntimeApplyResult:
    status: RuntimeApplyStatus
    message: UserMessageRef | None
    diagnostics: ErrorDiagnostics | None


@dataclass(frozen=True, slots=True)
class TransactionResult:
    status: TransactionStatus
    message: UserMessageRef | None
    diagnostics: ErrorDiagnostics | None


__all__ = [
    "CONTENT_POLICIES",
    "CONTENT_POLICY_METADATA_ONLY",
    "CONTENT_POLICY_RAW_USER_TEXT_ALLOWED",
    "CONTENT_POLICY_REDACTED",
    "CONTENT_POLICY_SECRET_FORBIDDEN",
    "ContentPolicy",
    "DIAGNOSTIC_CATEGORIES",
    "DIAGNOSTIC_CATEGORY_AUTH",
    "DIAGNOSTIC_CATEGORY_INVALID_RESPONSE",
    "DIAGNOSTIC_CATEGORY_LIFECYCLE",
    "DIAGNOSTIC_CATEGORY_NETWORK",
    "DIAGNOSTIC_CATEGORY_QUOTA",
    "DIAGNOSTIC_CATEGORY_RATE_LIMIT",
    "DIAGNOSTIC_CATEGORY_SERVICE_UNAVAILABLE",
    "DIAGNOSTIC_CATEGORY_TIMEOUT",
    "DIAGNOSTIC_CATEGORY_TRANSACTION",
    "DIAGNOSTIC_CATEGORY_UNKNOWN",
    "DIAGNOSTIC_FIELD_KEY_MAX_LENGTH",
    "DIAGNOSTIC_FIELD_MAX_ITEMS",
    "DIAGNOSTIC_FIELD_VALUE_MAX_LENGTH",
    "DIAGNOSTIC_VISIBILITIES",
    "DIAGNOSTIC_VISIBILITY_BASIC",
    "DIAGNOSTIC_VISIBILITY_DIAGNOSTIC_ONLY",
    "DIAGNOSTIC_VISIBILITY_PERSISTED_FAILURE_ONLY",
    "DiagnosticCategory",
    "DiagnosticFieldValue",
    "DiagnosticVisibility",
    "ErrorDiagnostics",
    "RUNTIME_APPLY_RESULT_STATUSES",
    "RUNTIME_APPLY_STATUS_APPLIED",
    "RUNTIME_APPLY_STATUS_DEGRADED",
    "RUNTIME_APPLY_STATUS_FAILED",
    "RuntimeApplyResult",
    "RuntimeApplyStatus",
    "SAFE_MESSAGE_PARAM_KEY_MAX_LENGTH",
    "SAFE_MESSAGE_PARAM_MAX_ITEMS",
    "SAFE_MESSAGE_PARAM_VALUE_MAX_LENGTH",
    "SEVERITIES",
    "SEVERITY_ERROR",
    "SEVERITY_INFO",
    "SEVERITY_WARNING",
    "TRANSACTION_RESULT_STATUSES",
    "TRANSACTION_STATUS_REMOTE_ACTIVE_LOCAL_MISSING",
    "TRANSACTION_STATUS_REMOTE_DELIVERY_ACK_PENDING",
    "TRANSACTION_STATUS_PROVIDER_VERIFICATION_FAILED",
    "TRANSACTION_STATUS_SECRET_WRITE_FAILED",
    "TRANSACTION_STATUS_SETTINGS_COMMIT_FAILED",
    "TRANSACTION_STATUS_SETTINGS_COMMIT_FAILED_SECRET_RESTORE_FAILED",
    "TRANSACTION_STATUS_SETTINGS_COMMIT_FAILED_SECRET_RESTORED",
    "TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED",
    "TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED",
    "TransactionResult",
    "TransactionStatus",
    "SafeMessageParam",
    "Severity",
    "UserMessageRef",
    "UserErrorReport",
]
