from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Protocol

from puripuly_heart.app.ports._settings_values import freeze_settings_values
from puripuly_heart.app.ports.provider_verifier import (
    PROVIDER_VERIFICATION_STATUS_VERIFIED,
    ProviderVerificationRequest,
    ProviderVerifierPort,
)
from puripuly_heart.app.ports.secret_store import SecretSnapshot, SecretStorePort
from puripuly_heart.app.ports.settings_repository import (
    SettingsCommitRequest,
    SettingsRepositoryPort,
)
from puripuly_heart.app.services.operational_state_payloads import (
    ProviderVerificationEvidence,
    provider_verification_state_values,
)
from puripuly_heart.core.diagnostics import safe_exception_fields, sanitize_diagnostics
from puripuly_heart.core.messages import (
    CONTENT_POLICY_METADATA_ONLY,
    DIAGNOSTIC_CATEGORY_TRANSACTION,
    DIAGNOSTIC_VISIBILITY_BASIC,
    TRANSACTION_STATUS_PROVIDER_VERIFICATION_FAILED,
    TRANSACTION_STATUS_SECRET_WRITE_FAILED,
    TRANSACTION_STATUS_SETTINGS_COMMIT_FAILED,
    TRANSACTION_STATUS_SETTINGS_COMMIT_FAILED_SECRET_RESTORE_FAILED,
    TRANSACTION_STATUS_SETTINGS_COMMIT_FAILED_SECRET_RESTORED,
    TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
    DiagnosticFieldValue,
    ErrorDiagnostics,
    TransactionResult,
    UserMessageRef,
)

_VERIFICATION_EVIDENCE_SENSITIVE_KEY_FRAGMENTS = (
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "body",
    "credential_value",
    "password",
    "payload",
    "private_key",
    "raw",
    "refresh_token",
    "secret",
    "token",
)


@dataclass(frozen=True, slots=True)
class DashboardNeedsKeySnapshot:
    translation_needs_key: bool | None
    stt_needs_key: bool | None
    settings_revision: str | None = None


@dataclass(frozen=True, slots=True)
class SecretSetRequest:
    secret_key: str
    secret_value: str = field(repr=False)
    settings_values: Mapping[str, object]
    expected_settings_revision: str | None
    reason: str | None
    correlation_id: str | None
    dashboard_needs_key: DashboardNeedsKeySnapshot | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "settings_values", freeze_settings_values(self.settings_values))


@dataclass(frozen=True, slots=True)
class SecretClearRequest:
    secret_key: str
    settings_values: Mapping[str, object]
    expected_settings_revision: str | None
    reason: str | None
    correlation_id: str | None
    dashboard_needs_key: DashboardNeedsKeySnapshot | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "settings_values", freeze_settings_values(self.settings_values))


@dataclass(frozen=True, slots=True)
class ProviderSecretVerificationRequest:
    provider: str
    secret_key: str
    secret_value: str = field(repr=False)
    secret_revision: str | None = None
    verifier_context: Mapping[str, DiagnosticFieldValue] = field(default_factory=dict)
    expected_settings_revision: str | None = None
    reason: str | None = None
    correlation_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "verifier_context",
            MappingProxyType(
                _sanitize_verification_fields(
                    self.verifier_context,
                    secret_value=self.secret_value,
                )
            ),
        )


class DashboardNeedsKeySnapshotPublisher(Protocol):
    async def publish_dashboard_needs_key_snapshot(
        self,
        snapshot: DashboardNeedsKeySnapshot,
        *,
        correlation_id: str | None,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class SecretSettingsTransaction:
    secret_store: SecretStorePort
    settings_repository: SettingsRepositoryPort
    dashboard_needs_key_publisher: DashboardNeedsKeySnapshotPublisher | None = None
    provider_verifier: ProviderVerifierPort | None = None

    async def set_provider_secret(self, request: SecretSetRequest) -> TransactionResult:
        try:
            snapshot = await self.secret_store.snapshot_secret(request.secret_key)
        except Exception as exc:
            return _secret_write_failed_result(
                secret_key=request.secret_key,
                action="set",
                operation="snapshot_secret",
                exception=exc,
            )

        try:
            write_result = await self.secret_store.set_secret(
                request.secret_key,
                request.secret_value,
            )
        except Exception as exc:
            return _secret_write_failed_result(
                secret_key=request.secret_key,
                action="set",
                operation="set_secret",
                exception=exc,
            )

        if not write_result.succeeded:
            return _secret_write_failed_result(
                secret_key=request.secret_key,
                action="set",
                operation="set_secret",
                diagnostics=write_result.diagnostics,
            )

        return await self._commit_settings_or_restore_secret(
            request=request,
            snapshot=snapshot,
            action="set",
        )

    async def clear_provider_secret(self, request: SecretClearRequest) -> TransactionResult:
        try:
            snapshot = await self.secret_store.snapshot_secret(request.secret_key)
        except Exception as exc:
            return _secret_write_failed_result(
                secret_key=request.secret_key,
                action="clear",
                operation="snapshot_secret",
                exception=exc,
            )

        try:
            write_result = await self.secret_store.clear_secret(request.secret_key)
        except Exception as exc:
            return _secret_write_failed_result(
                secret_key=request.secret_key,
                action="clear",
                operation="clear_secret",
                exception=exc,
            )

        if not write_result.succeeded:
            return _secret_write_failed_result(
                secret_key=request.secret_key,
                action="clear",
                operation="clear_secret",
                diagnostics=write_result.diagnostics,
            )

        return await self._commit_settings_or_restore_secret(
            request=request,
            snapshot=snapshot,
            action="clear",
        )

    async def verify_provider_secret(
        self,
        request: ProviderSecretVerificationRequest,
    ) -> TransactionResult:
        if self.provider_verifier is None:
            return _provider_verification_failed_result(
                provider=request.provider,
                secret_key=request.secret_key,
                code="provider_verifier_missing",
                phase="verify_provider_secret",
            )
        if not request.verifier_context:
            return _provider_verification_failed_result(
                provider=request.provider,
                secret_key=request.secret_key,
                code="provider_verification_context_missing",
                phase="verify_provider_secret",
            )

        try:
            verification_result = await self.provider_verifier.verify_provider_secret(
                ProviderVerificationRequest(
                    provider=request.provider,
                    secret_key=request.secret_key,
                    secret_value=request.secret_value,
                    secret_revision=request.secret_revision,
                    context=request.verifier_context,
                )
            )
        except Exception as exc:
            return _provider_verification_failed_result(
                provider=request.provider,
                secret_key=request.secret_key,
                code="provider_verifier_exception",
                phase="verify_provider_secret",
                exception=exc,
            )

        if (
            verification_result.provider != request.provider
            or verification_result.secret_key != request.secret_key
        ):
            return _provider_verification_failed_result(
                provider=request.provider,
                secret_key=request.secret_key,
                code="provider_verifier_result_mismatch",
                phase="verify_provider_secret",
                verifier_status=verification_result.status,
            )

        settings_values = provider_verification_state_values(
            ProviderVerificationEvidence(
                status=verification_result.status,
                provider=request.provider,
                secret_key=request.secret_key,
                secret_revision=verification_result.secret_revision or request.secret_revision,
                secret_fingerprint=_secret_fingerprint(request.secret_value),
                verifier_context=dict(request.verifier_context),
                verifier_evidence=_sanitize_verification_fields(
                    verification_result.evidence,
                    secret_value=request.secret_value,
                ),
            )
        )
        try:
            commit_result = await self.settings_repository.save(
                SettingsCommitRequest(
                    values=settings_values,
                    expected_revision=request.expected_settings_revision,
                    reason=request.reason,
                )
            )
        except Exception as exc:
            return _settings_commit_failed_result(
                provider=request.provider,
                secret_key=request.secret_key,
                message=None,
                exception=exc,
            )

        if not commit_result.succeeded or commit_result.snapshot is None:
            return TransactionResult(
                status=TRANSACTION_STATUS_SETTINGS_COMMIT_FAILED,
                message=commit_result.message,
                diagnostics=(
                    sanitize_diagnostics(commit_result.diagnostics)
                    if commit_result.diagnostics
                    else None
                )
                or _settings_commit_failed_diagnostics(
                    provider=request.provider,
                    secret_key=request.secret_key,
                ),
            )

        if verification_result.status == PROVIDER_VERIFICATION_STATUS_VERIFIED:
            return TransactionResult(
                status=TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
                message=verification_result.message or commit_result.message,
                diagnostics=commit_result.diagnostics,
            )

        return TransactionResult(
            status=TRANSACTION_STATUS_PROVIDER_VERIFICATION_FAILED,
            message=verification_result.message or commit_result.message,
            diagnostics=(
                sanitize_diagnostics(verification_result.diagnostics)
                if verification_result.diagnostics is not None
                else None
            ),
        )

    async def _commit_settings_or_restore_secret(
        self,
        *,
        request: SecretSetRequest | SecretClearRequest,
        snapshot: SecretSnapshot,
        action: str,
    ) -> TransactionResult:
        try:
            commit_result = await self.settings_repository.save(
                SettingsCommitRequest(
                    values=request.settings_values,
                    expected_revision=request.expected_settings_revision,
                    reason=request.reason,
                )
            )
        except Exception as exc:
            return await self._restore_after_settings_commit_failure(
                snapshot=snapshot,
                action=action,
                commit_message=None,
                primary=_settings_commit_failed_diagnostics(
                    provider="unknown", secret_key=request.secret_key, exception=exc
                ),
            )

        if commit_result.succeeded and commit_result.snapshot is not None:
            notification = await self._publish_dashboard_needs_key_snapshot(
                request.dashboard_needs_key,
                settings_revision=commit_result.snapshot.revision,
                correlation_id=request.correlation_id,
            )
            return TransactionResult(
                status=TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
                message=commit_result.message,
                diagnostics=commit_result.diagnostics,
                secondary_diagnostics=(
                    *getattr(self.settings_repository, "secondary_diagnostics", ()),
                    *((notification,) if notification is not None else ()),
                ),
                secondary_omitted_count=getattr(
                    self.settings_repository, "secondary_omitted_count", 0
                ),
            )

        return await self._restore_after_settings_commit_failure(
            snapshot=snapshot,
            action=action,
            commit_message=commit_result.message,
            primary=commit_result.diagnostics
            or _settings_commit_failed_diagnostics(
                provider="unknown", secret_key=request.secret_key
            ),
            upstream_secondaries=getattr(self.settings_repository, "secondary_diagnostics", ()),
            upstream_omitted_count=getattr(self.settings_repository, "secondary_omitted_count", 0),
        )

    async def _restore_after_settings_commit_failure(
        self,
        *,
        snapshot: SecretSnapshot,
        action: str,
        commit_message: UserMessageRef | None,
        primary: ErrorDiagnostics,
        upstream_secondaries: tuple[ErrorDiagnostics, ...] = (),
        upstream_omitted_count: int = 0,
    ) -> TransactionResult:
        primary = sanitize_diagnostics(primary)
        restore_cause = None
        try:
            restore_result = await self.secret_store.restore_secret(snapshot)
            restore_succeeded = restore_result.succeeded
            restore_cause = (
                sanitize_diagnostics(restore_result.diagnostics)
                if restore_result.diagnostics
                else None
            )
        except Exception as exc:
            restore_succeeded = False
            restore_cause = ErrorDiagnostics(
                component="secret_settings_transaction",
                operation="restore_secret",
                code="secret_restore_failed",
                category=DIAGNOSTIC_CATEGORY_TRANSACTION,
                visibility=DIAGNOSTIC_VISIBILITY_BASIC,
                content_policy=CONTENT_POLICY_METADATA_ONLY,
                status_code=None,
                retry_after_ms=None,
                fields={"phase": "restore_secret", **safe_exception_fields(exc)},
            )
        outcome = _compensation_diagnostics(
            code=(
                "settings_commit_failed_secret_restored"
                if restore_succeeded
                else "settings_commit_failed_secret_restore_failed"
            ),
            secret_key=snapshot.key,
            action=action,
            previous_secret_existed=snapshot.existed,
            secret_restore_succeeded=restore_succeeded,
        )
        secondary = [*upstream_secondaries, replace(outcome, cause_id=primary.diagnostic_id)]
        if restore_cause is not None:
            secondary.append(replace(restore_cause, cause_id=primary.diagnostic_id))
        return TransactionResult(
            status=(
                TRANSACTION_STATUS_SETTINGS_COMMIT_FAILED_SECRET_RESTORED
                if restore_succeeded
                else TRANSACTION_STATUS_SETTINGS_COMMIT_FAILED_SECRET_RESTORE_FAILED
            ),
            message=commit_message,
            diagnostics=primary,
            secondary_diagnostics=tuple(secondary),
            secondary_omitted_count=upstream_omitted_count,
        )

    async def _publish_dashboard_needs_key_snapshot(
        self,
        snapshot: DashboardNeedsKeySnapshot | None,
        *,
        settings_revision: str | None,
        correlation_id: str | None,
    ) -> ErrorDiagnostics | None:
        if snapshot is None or self.dashboard_needs_key_publisher is None:
            return
        try:
            await self.dashboard_needs_key_publisher.publish_dashboard_needs_key_snapshot(
                replace(snapshot, settings_revision=settings_revision),
                correlation_id=correlation_id,
            )
        except Exception as exc:
            return ErrorDiagnostics(
                component="secret_settings_transaction",
                operation="dashboard_notification",
                code="dashboard_notification_failed",
                category=DIAGNOSTIC_CATEGORY_TRANSACTION,
                visibility=DIAGNOSTIC_VISIBILITY_BASIC,
                content_policy=CONTENT_POLICY_METADATA_ONLY,
                status_code=None,
                retry_after_ms=None,
                fields={"phase": "dashboard_notification", **safe_exception_fields(exc)},
            )
        return None


def _secret_write_failed_result(
    *,
    secret_key: str,
    action: str,
    operation: str,
    exception: BaseException | None = None,
    diagnostics: ErrorDiagnostics | None = None,
) -> TransactionResult:
    cause = sanitize_diagnostics(diagnostics) if diagnostics is not None else None
    return TransactionResult(
        status=TRANSACTION_STATUS_SECRET_WRITE_FAILED,
        message=None,
        diagnostics=ErrorDiagnostics(
            component="secret_settings_transaction",
            operation=operation,
            code="secret_write_failed",
            category=DIAGNOSTIC_CATEGORY_TRANSACTION,
            visibility=DIAGNOSTIC_VISIBILITY_BASIC,
            content_policy=CONTENT_POLICY_METADATA_ONLY,
            status_code=cause.status_code if cause else None,
            retry_after_ms=cause.retry_after_ms if cause else None,
            fields={
                **(cause.fields if cause else {}),
                "secret_key": secret_key,
                "action": action,
                "phase": operation,
                **(safe_exception_fields(exception) if exception is not None else {}),
                "secret_write_succeeded": False,
            },
            cause_id=cause.diagnostic_id if cause else None,
        ),
        secondary_diagnostics=(cause,) if cause else (),
    )


def _normalized_evidence_key(key: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in key)


def _is_sensitive_evidence_key(key: str) -> bool:
    normalized = _normalized_evidence_key(key)
    return any(
        fragment in normalized for fragment in _VERIFICATION_EVIDENCE_SENSITIVE_KEY_FRAGMENTS
    )


def _is_safe_diagnostic_field_value(value: object) -> bool:
    return value is None or isinstance(value, str | int | float | bool)


def _sanitize_verification_fields(
    values: Mapping[str, DiagnosticFieldValue],
    *,
    secret_value: str,
) -> dict[str, DiagnosticFieldValue]:
    sanitized: dict[str, DiagnosticFieldValue] = {}
    for raw_key, raw_value in values.items():
        if not isinstance(raw_key, str):
            continue
        if secret_value and secret_value in raw_key:
            continue
        if _is_sensitive_evidence_key(raw_key):
            continue
        if not _is_safe_diagnostic_field_value(raw_value):
            continue
        if isinstance(raw_value, str) and secret_value and secret_value in raw_value:
            continue
        sanitized[raw_key] = raw_value
    return sanitized


def _secret_fingerprint(secret_value: str) -> str:
    digest = hashlib.sha256(secret_value.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _provider_verification_failed_result(
    *,
    provider: str,
    secret_key: str,
    code: str,
    phase: str,
    verifier_status: str | None = None,
    message: UserMessageRef | None = None,
    exception: BaseException | None = None,
    diagnostics: ErrorDiagnostics | None = None,
) -> TransactionResult:
    fields: dict[str, DiagnosticFieldValue] = {
        "provider": provider,
        "secret_key": secret_key,
        "phase": phase,
        "verification_succeeded": False,
    }
    if verifier_status is not None:
        fields["verifier_status"] = verifier_status
    fields.update(safe_exception_fields(exception) if exception is not None else {})
    return TransactionResult(
        status=TRANSACTION_STATUS_PROVIDER_VERIFICATION_FAILED,
        message=message,
        diagnostics=(
            sanitize_diagnostics(diagnostics)
            if diagnostics is not None
            else ErrorDiagnostics(
                component="secret_settings_transaction",
                operation="verify_provider_secret",
                code=code,
                category=DIAGNOSTIC_CATEGORY_TRANSACTION,
                visibility=DIAGNOSTIC_VISIBILITY_BASIC,
                content_policy=CONTENT_POLICY_METADATA_ONLY,
                status_code=None,
                retry_after_ms=None,
                fields=fields,
            )
        ),
    )


def _settings_commit_failed_result(
    *,
    provider: str,
    secret_key: str,
    message: UserMessageRef | None,
    exception: BaseException | None = None,
) -> TransactionResult:
    return TransactionResult(
        status=TRANSACTION_STATUS_SETTINGS_COMMIT_FAILED,
        message=message,
        diagnostics=_settings_commit_failed_diagnostics(
            provider=provider,
            secret_key=secret_key,
            exception=exception,
        ),
    )


def _settings_commit_failed_diagnostics(
    *,
    provider: str,
    secret_key: str,
    exception: BaseException | None = None,
) -> ErrorDiagnostics:
    return ErrorDiagnostics(
        component="secret_settings_transaction",
        operation="commit_provider_verification",
        code="settings_commit_failed",
        category=DIAGNOSTIC_CATEGORY_TRANSACTION,
        visibility=DIAGNOSTIC_VISIBILITY_BASIC,
        content_policy=CONTENT_POLICY_METADATA_ONLY,
        status_code=None,
        retry_after_ms=None,
        fields={
            "provider": provider,
            "secret_key": secret_key,
            "settings_commit_succeeded": False,
            "phase": "settings_save",
            **(safe_exception_fields(exception) if exception is not None else {}),
        },
    )


def _compensation_diagnostics(
    *,
    code: str,
    secret_key: str,
    action: str,
    previous_secret_existed: bool,
    secret_restore_succeeded: bool,
) -> ErrorDiagnostics:
    return ErrorDiagnostics(
        component="secret_settings_transaction",
        operation="restore_secret",
        code=code,
        category=DIAGNOSTIC_CATEGORY_TRANSACTION,
        visibility=DIAGNOSTIC_VISIBILITY_BASIC,
        content_policy=CONTENT_POLICY_METADATA_ONLY,
        status_code=None,
        retry_after_ms=None,
        fields={
            "secret_key": secret_key,
            "action": action,
            "previous_secret_existed": previous_secret_existed,
            "settings_commit_succeeded": False,
            "secret_restore_succeeded": secret_restore_succeeded,
        },
    )


__all__ = [
    "DashboardNeedsKeySnapshot",
    "DashboardNeedsKeySnapshotPublisher",
    "ProviderSecretVerificationRequest",
    "SecretClearRequest",
    "SecretSetRequest",
    "SecretSettingsTransaction",
]
