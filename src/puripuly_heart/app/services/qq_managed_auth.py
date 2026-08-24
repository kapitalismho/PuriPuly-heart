from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from puripuly_heart.app.ports.broker_client import (
    BrokerClientPort,
    QqManagedAssertionFailureSubcode,
    QqManagedAssertionRequest,
    QqManagedAssertionResult,
    QqManagedEntitlementSnapshot,
)
from puripuly_heart.app.ports.managed_identity_state import (
    ManagedIdentitySnapshot,
    ManagedIdentityStatePort,
)
from puripuly_heart.app.ports.secret_store import SecretSnapshot, SecretStorePort
from puripuly_heart.app.services.managed_auth_claims import (
    MANAGED_AUTH_CLAIM_SOURCE_QQ,
    ManagedAuthClaimGuard,
)
from puripuly_heart.app.services.managed_key_delivery_ack import (
    ACK_SOURCE_QQ,
    ManagedKeyDeliveryAckService,
    ManagedKeyDeliveryAckTokenStoreError,
)
from puripuly_heart.core.messages import (
    CONTENT_POLICY_METADATA_ONLY,
    DIAGNOSTIC_CATEGORY_AUTH,
    DIAGNOSTIC_CATEGORY_QUOTA,
    DIAGNOSTIC_CATEGORY_RATE_LIMIT,
    DIAGNOSTIC_CATEGORY_SERVICE_UNAVAILABLE,
    DIAGNOSTIC_CATEGORY_TRANSACTION,
    DIAGNOSTIC_VISIBILITY_BASIC,
    SEVERITY_ERROR,
    SEVERITY_INFO,
    SEVERITY_WARNING,
    TRANSACTION_STATUS_PROVIDER_VERIFICATION_FAILED,
    TRANSACTION_STATUS_REMOTE_ACTIVE_LOCAL_MISSING,
    TRANSACTION_STATUS_REMOTE_DELIVERY_ACK_PENDING,
    TRANSACTION_STATUS_SECRET_WRITE_FAILED,
    TRANSACTION_STATUS_SETTINGS_COMMIT_FAILED_SECRET_RESTORE_FAILED,
    TRANSACTION_STATUS_SETTINGS_COMMIT_FAILED_SECRET_RESTORED,
    TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
    DiagnosticCategory,
    DiagnosticFieldValue,
    ErrorDiagnostics,
    Severity,
    TransactionResult,
    UserMessageRef,
)

OPENROUTER_MANAGED_QQ_API_KEY_SECRET = "openrouter_managed_qq_api_key"
QQ_MANAGED_AUTH_FAILURE_MESSAGE_KEYS = (
    "qq_managed_auth.invalid_credential",
    "qq_managed_auth.mismatch",
    "qq_managed_auth.lifetime_used",
    "qq_managed_auth.rate_limited",
    "qq_managed_auth.key_unavailable",
    "qq_managed_auth.broker_unavailable",
)


def _freeze_fields(
    values: Mapping[str, DiagnosticFieldValue],
) -> Mapping[str, DiagnosticFieldValue]:
    return MappingProxyType(dict(values))


@dataclass(frozen=True, slots=True)
class QqManagedAuthRequest:
    qq_identity: str = field(repr=False)
    credential: str = field(repr=False)
    asserted_at: str
    correlation_id: str | None = None
    metadata: Mapping[str, DiagnosticFieldValue] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_fields(self.metadata))


@dataclass(frozen=True, slots=True)
class QqManagedAuthService:
    broker_client: BrokerClientPort
    secret_store: SecretStorePort
    managed_state: ManagedIdentityStatePort
    claim_guard: ManagedAuthClaimGuard | None = None
    delivery_ack_service: ManagedKeyDeliveryAckService | None = None

    async def authenticate(self, request: QqManagedAuthRequest) -> TransactionResult:
        recovery_result = await self._recover_pending_delivery_ack()
        if recovery_result is not None:
            return recovery_result

        claim_result = await self._preflight_claim_source()
        if claim_result is not None:
            return claim_result

        broker_result = await self._assert_qq_identity(request)
        if isinstance(broker_result, TransactionResult):
            return broker_result

        if not broker_result.managed_secret_key or broker_result.entitlement is None:
            return _remote_active_local_missing_result(
                operation="assert_qq_managed_identity",
                code="qq_assertion_missing_managed_key",
                phase="broker_assertion",
                failure_subcode="key_unavailable",
                secret_write_succeeded=False,
                settings_commit_succeeded=False,
                rollback_succeeded=None,
                compensation_succeeded=None,
                retry_after_ms=broker_result.retry_after_ms,
            )

        secret_snapshot = await self._snapshot_secret()
        if isinstance(secret_snapshot, TransactionResult):
            return secret_snapshot
        state_snapshot = self.managed_state.snapshot()

        if broker_result.delivery_ack is not None:
            try:
                await self._delivery_ack_service().store_pending(broker_result.delivery_ack)
            except ManagedKeyDeliveryAckTokenStoreError:
                return _remote_active_local_missing_result(
                    operation="store_qq_delivery_ack_token",
                    code="qq_delivery_ack_token_store_failed_before_local_key_write",
                    phase="delivery_ack_token_store",
                    failure_subcode="key_unavailable",
                    secret_write_succeeded=False,
                    settings_commit_succeeded=False,
                    rollback_succeeded=None,
                    compensation_succeeded=None,
                    retry_after_ms=None,
                )
            self._apply_entitlement_snapshot(broker_result.entitlement)
            if self.claim_guard is not None:
                self.claim_guard.record_success(MANAGED_AUTH_CLAIM_SOURCE_QQ)
            try:
                self.managed_state.persist()
            except Exception:
                return _remote_active_local_missing_result(
                    operation="persist_qq_pending_delivery_ack",
                    code="qq_pending_delivery_ack_persist_failed_before_local_key_write",
                    phase="delivery_ack_pending_persist",
                    failure_subcode="key_unavailable",
                    secret_write_succeeded=False,
                    settings_commit_succeeded=False,
                    rollback_succeeded=None,
                    compensation_succeeded=None,
                    retry_after_ms=None,
                )
            secret_write = await self._write_managed_secret(broker_result.managed_secret_key)
            if isinstance(secret_write, TransactionResult):
                return secret_write
            ack_result = await self._delivery_ack_service().retry_pending()
            if not ack_result.succeeded:
                return _delivery_ack_pending_result(
                    ack_status=ack_result.status,
                    diagnostics_present=ack_result.diagnostics is not None,
                )
            return TransactionResult(
                status=TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
                message=_message("qq_managed_auth.success", severity=SEVERITY_INFO),
                diagnostics=_diagnostics(
                    operation="persist_qq_managed_auth",
                    code="qq_managed_auth_succeeded",
                    category=DIAGNOSTIC_CATEGORY_TRANSACTION,
                    fields={
                        "phase": "settings_commit",
                        "secret_key": OPENROUTER_MANAGED_QQ_API_KEY_SECRET,
                        "secret_write_succeeded": True,
                        "settings_commit_succeeded": True,
                        "entitlement_ref_present": bool(
                            broker_result.entitlement.managed_credential_ref
                        ),
                        "entitlement_expires_at_present": bool(
                            broker_result.entitlement.expires_at
                        ),
                    },
                ),
            )

        secret_write = await self._write_managed_secret(broker_result.managed_secret_key)
        if isinstance(secret_write, TransactionResult):
            return secret_write

        self._apply_entitlement_snapshot(broker_result.entitlement)
        if self.claim_guard is not None:
            self.claim_guard.record_success(MANAGED_AUTH_CLAIM_SOURCE_QQ)
        try:
            self.managed_state.persist()
        except Exception:
            return await self._rollback_after_persist_failure(
                secret_snapshot=secret_snapshot,
                state_snapshot=state_snapshot,
            )

        return TransactionResult(
            status=TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
            message=_message("qq_managed_auth.success", severity=SEVERITY_INFO),
            diagnostics=_diagnostics(
                operation="persist_qq_managed_auth",
                code="qq_managed_auth_succeeded",
                category=DIAGNOSTIC_CATEGORY_TRANSACTION,
                fields={
                    "phase": "settings_commit",
                    "secret_key": OPENROUTER_MANAGED_QQ_API_KEY_SECRET,
                    "secret_write_succeeded": True,
                    "settings_commit_succeeded": True,
                    "entitlement_ref_present": bool(
                        broker_result.entitlement.managed_credential_ref
                    ),
                    "entitlement_expires_at_present": bool(broker_result.entitlement.expires_at),
                },
            ),
        )

    async def _preflight_claim_source(self) -> TransactionResult | None:
        if self.claim_guard is None:
            return None
        return await self.claim_guard.preflight(MANAGED_AUTH_CLAIM_SOURCE_QQ)

    def _delivery_ack_service(self) -> ManagedKeyDeliveryAckService:
        if self.delivery_ack_service is not None:
            return self.delivery_ack_service
        return ManagedKeyDeliveryAckService(
            broker_client=self.broker_client,
            secret_store=self.secret_store,
            managed_state=self.managed_state,
        )

    async def _recover_pending_delivery_ack(self) -> TransactionResult | None:
        if self.managed_state.pending_delivery_ack_source != ACK_SOURCE_QQ:
            return None
        result = await self._delivery_ack_service().recover_pending(
            source=ACK_SOURCE_QQ,
            managed_secret_key=OPENROUTER_MANAGED_QQ_API_KEY_SECRET,
        )
        if not result.succeeded:
            return _delivery_ack_pending_result(
                ack_status=result.status,
                diagnostics_present=result.diagnostics is not None,
            )
        if result.status == "none":
            return None
        return TransactionResult(
            status=TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
            message=_message("qq_managed_auth.success", severity=SEVERITY_INFO),
            diagnostics=_diagnostics(
                operation="recover_qq_managed_key_delivery",
                code="qq_delivery_ack_recovered",
                category=DIAGNOSTIC_CATEGORY_TRANSACTION,
                fields={
                    "phase": "remote_delivery_ack_recovery",
                    "delivery_ack_status": result.status,
                    "secret_write_succeeded": True,
                    "settings_commit_succeeded": True,
                },
            ),
        )

    async def _assert_qq_identity(
        self,
        request: QqManagedAuthRequest,
    ) -> QqManagedAssertionResult | TransactionResult:
        try:
            result = await self.broker_client.assert_qq_managed_identity(
                QqManagedAssertionRequest(
                    qq_identity=request.qq_identity,
                    credential=request.credential,
                    asserted_at=request.asserted_at,
                    metadata=request.metadata,
                )
            )
        except Exception:
            return _broker_failure_result(
                failure_subcode="broker_unavailable",
                code="qq_assertion_exception",
                retry_after_ms=None,
            )

        if not result.succeeded:
            return _broker_failure_result(
                failure_subcode=result.failure_subcode or "broker_unavailable",
                code="qq_assertion_failed",
                retry_after_ms=result.retry_after_ms,
                diagnostics_present=result.diagnostics is not None,
            )

        return result

    async def _snapshot_secret(self) -> SecretSnapshot | TransactionResult:
        try:
            return await self.secret_store.snapshot_secret(OPENROUTER_MANAGED_QQ_API_KEY_SECRET)
        except Exception:
            return _secret_failure_result(
                operation="snapshot_qq_managed_secret",
                code="qq_secret_snapshot_failed",
                phase="secret_snapshot",
            )

    async def _write_managed_secret(self, managed_secret_key: str) -> TransactionResult | None:
        try:
            result = await self.secret_store.set_secret(
                OPENROUTER_MANAGED_QQ_API_KEY_SECRET,
                managed_secret_key,
            )
        except Exception:
            return _secret_failure_result(
                operation="set_qq_managed_secret",
                code="qq_secret_write_exception",
                phase="secret_write",
            )

        if not result.succeeded:
            return _secret_failure_result(
                operation="set_qq_managed_secret",
                code="qq_secret_write_failed",
                phase="secret_write",
                diagnostics_present=result.diagnostics is not None,
                message=result.message,
            )
        return None

    def _apply_entitlement_snapshot(self, entitlement: QqManagedEntitlementSnapshot) -> None:
        current_ref = self.managed_state.active_managed_credential_ref
        next_ref = (
            _normalize_optional_text(entitlement.managed_credential_ref)
            or _normalize_optional_text(entitlement.qq_subject_ref)
            or current_ref
            or self.managed_state.installation_id
        )
        if current_ref != next_ref:
            self.managed_state.founder_letter_seen_credential_ref = None
        self.managed_state.active_managed_credential_ref = next_ref
        self.managed_state.active_managed_expires_at = _normalize_optional_text(
            entitlement.expires_at
        )

    async def _rollback_after_persist_failure(
        self,
        *,
        secret_snapshot: SecretSnapshot,
        state_snapshot: ManagedIdentitySnapshot,
    ) -> TransactionResult:
        state_rollback_succeeded = self._restore_state_snapshot(state_snapshot)
        secret_rollback_succeeded = await self._restore_secret_snapshot(secret_snapshot)
        compensation_succeeded = state_rollback_succeeded and secret_rollback_succeeded
        status = (
            TRANSACTION_STATUS_SETTINGS_COMMIT_FAILED_SECRET_RESTORED
            if compensation_succeeded
            else TRANSACTION_STATUS_SETTINGS_COMMIT_FAILED_SECRET_RESTORE_FAILED
        )
        return TransactionResult(
            status=status,
            message=_message(
                "qq_managed_auth.settings_commit_failed",
                severity=SEVERITY_ERROR,
            ),
            diagnostics=_diagnostics(
                operation="persist_qq_managed_auth",
                code=(
                    "qq_settings_commit_failed_rolled_back"
                    if compensation_succeeded
                    else "qq_settings_commit_failed_rollback_failed"
                ),
                category=DIAGNOSTIC_CATEGORY_TRANSACTION,
                fields={
                    "phase": "settings_commit",
                    "secret_key": OPENROUTER_MANAGED_QQ_API_KEY_SECRET,
                    "secret_write_succeeded": True,
                    "settings_commit_succeeded": False,
                    "state_rollback_succeeded": state_rollback_succeeded,
                    "secret_rollback_succeeded": secret_rollback_succeeded,
                    "compensation_succeeded": compensation_succeeded,
                },
            ),
        )

    async def _rollback_after_delivery_ack_token_store_failure(
        self,
        *,
        secret_snapshot: SecretSnapshot,
        state_snapshot: ManagedIdentitySnapshot,
        metadata: object,
    ) -> TransactionResult:
        state_rollback_succeeded = self._restore_state_snapshot(state_snapshot)
        secret_rollback_succeeded = await self._restore_secret_snapshot(secret_snapshot)
        if not secret_rollback_succeeded:
            self.managed_state.pending_delivery_ack_source = getattr(metadata, "source", None)
            self.managed_state.pending_delivery_ack_delivery_id = getattr(
                metadata, "delivery_id", None
            )
            self.managed_state.pending_delivery_ack_managed_credential_ref = getattr(
                metadata,
                "managed_credential_ref",
                None,
            )
            self.managed_state.pending_delivery_ack_expires_at = getattr(
                metadata, "expires_at", None
            )
            try:
                self.managed_state.persist()
            except Exception:
                pass
        return _remote_active_local_missing_result(
            operation="store_qq_delivery_ack_token",
            code=(
                "qq_delivery_ack_token_store_failed_local_key_rolled_back"
                if state_rollback_succeeded and secret_rollback_succeeded
                else "qq_delivery_ack_token_store_failed_local_key_rollback_failed"
            ),
            phase="delivery_ack_token_store",
            failure_subcode="key_unavailable",
            secret_write_succeeded=not secret_rollback_succeeded,
            settings_commit_succeeded=False,
            rollback_succeeded=state_rollback_succeeded,
            compensation_succeeded=state_rollback_succeeded and secret_rollback_succeeded,
            retry_after_ms=None,
        )

    def _restore_state_snapshot(self, snapshot: ManagedIdentitySnapshot) -> bool:
        try:
            self.managed_state.restore(snapshot)
            self.managed_state.persist()
        except Exception:
            self.managed_state.restore(snapshot)
            return False
        return True

    async def _restore_secret_snapshot(self, snapshot: SecretSnapshot) -> bool:
        try:
            result = await self.secret_store.restore_secret(snapshot)
        except Exception:
            return False
        return result.succeeded


def _broker_failure_result(
    *,
    failure_subcode: QqManagedAssertionFailureSubcode,
    code: str,
    retry_after_ms: int | None,
    diagnostics_present: bool = False,
) -> TransactionResult:
    category = _category_for_failure_subcode(failure_subcode)
    return TransactionResult(
        status=TRANSACTION_STATUS_PROVIDER_VERIFICATION_FAILED,
        message=_message(
            f"qq_managed_auth.{failure_subcode}",
            severity=SEVERITY_WARNING if failure_subcode == "rate_limited" else SEVERITY_ERROR,
            params={"retry_after_ms": retry_after_ms} if retry_after_ms is not None else {},
        ),
        diagnostics=_diagnostics(
            operation="assert_qq_managed_identity",
            code=code,
            category=category,
            retry_after_ms=retry_after_ms,
            fields={
                "phase": "broker_assertion",
                "qq_failure_subcode": failure_subcode,
                "diagnostics_present": diagnostics_present,
                "secret_write_succeeded": False,
                "settings_commit_succeeded": False,
            },
        ),
    )


def _secret_failure_result(
    *,
    operation: str,
    code: str,
    phase: str,
    diagnostics_present: bool = False,
    message: UserMessageRef | None = None,
) -> TransactionResult:
    return TransactionResult(
        status=TRANSACTION_STATUS_SECRET_WRITE_FAILED,
        message=message or _message("qq_managed_auth.secret_write_failed", severity=SEVERITY_ERROR),
        diagnostics=_diagnostics(
            operation=operation,
            code=code,
            category=DIAGNOSTIC_CATEGORY_TRANSACTION,
            fields={
                "phase": phase,
                "secret_key": OPENROUTER_MANAGED_QQ_API_KEY_SECRET,
                "diagnostics_present": diagnostics_present,
                "secret_write_succeeded": False,
                "settings_commit_succeeded": False,
            },
        ),
    )


def _remote_active_local_missing_result(
    *,
    operation: str,
    code: str,
    phase: str,
    failure_subcode: QqManagedAssertionFailureSubcode,
    secret_write_succeeded: bool,
    settings_commit_succeeded: bool,
    rollback_succeeded: bool | None,
    compensation_succeeded: bool | None,
    retry_after_ms: int | None,
) -> TransactionResult:
    fields: dict[str, DiagnosticFieldValue] = {
        "phase": phase,
        "qq_failure_subcode": failure_subcode,
        "remote_active": True,
        "secret_key": OPENROUTER_MANAGED_QQ_API_KEY_SECRET,
        "secret_write_succeeded": secret_write_succeeded,
        "settings_commit_succeeded": settings_commit_succeeded,
    }
    if rollback_succeeded is not None:
        fields["rollback_succeeded"] = rollback_succeeded
    if compensation_succeeded is not None:
        fields["compensation_succeeded"] = compensation_succeeded
    return TransactionResult(
        status=TRANSACTION_STATUS_REMOTE_ACTIVE_LOCAL_MISSING,
        message=_message("qq_managed_auth.key_unavailable", severity=SEVERITY_ERROR),
        diagnostics=_diagnostics(
            operation=operation,
            code=code,
            category=DIAGNOSTIC_CATEGORY_SERVICE_UNAVAILABLE,
            retry_after_ms=retry_after_ms,
            fields=fields,
        ),
    )


def _delivery_ack_pending_result(
    *,
    ack_status: str,
    diagnostics_present: bool,
) -> TransactionResult:
    return TransactionResult(
        status=TRANSACTION_STATUS_REMOTE_DELIVERY_ACK_PENDING,
        message=_message("qq_managed_auth.key_unavailable", severity=SEVERITY_WARNING),
        diagnostics=_diagnostics(
            operation="acknowledge_qq_managed_key_delivery",
            code="qq_delivery_ack_pending",
            category=DIAGNOSTIC_CATEGORY_TRANSACTION,
            fields={
                "phase": "remote_delivery_ack",
                "secret_key": OPENROUTER_MANAGED_QQ_API_KEY_SECRET,
                "secret_write_succeeded": True,
                "settings_commit_succeeded": True,
                "delivery_ack_status": ack_status,
                "diagnostics_present": diagnostics_present,
            },
        ),
    )


def _category_for_failure_subcode(
    failure_subcode: QqManagedAssertionFailureSubcode,
) -> DiagnosticCategory:
    if failure_subcode in {"invalid_credential", "mismatch"}:
        return DIAGNOSTIC_CATEGORY_AUTH
    if failure_subcode == "lifetime_used":
        return DIAGNOSTIC_CATEGORY_QUOTA
    if failure_subcode == "rate_limited":
        return DIAGNOSTIC_CATEGORY_RATE_LIMIT
    return DIAGNOSTIC_CATEGORY_SERVICE_UNAVAILABLE


def _diagnostics(
    *,
    operation: str,
    code: str,
    category: DiagnosticCategory,
    fields: Mapping[str, DiagnosticFieldValue],
    retry_after_ms: int | None = None,
) -> ErrorDiagnostics:
    return ErrorDiagnostics(
        component="qq_managed_auth",
        operation=operation,
        code=code,
        category=category,
        visibility=DIAGNOSTIC_VISIBILITY_BASIC,
        content_policy=CONTENT_POLICY_METADATA_ONLY,
        status_code=None,
        retry_after_ms=retry_after_ms,
        fields=fields,
    )


def _message(
    key: str,
    *,
    severity: Severity,
    params: Mapping[str, object] | None = None,
) -> UserMessageRef:
    safe_params: dict[str, str | int | float | bool | None] = {}
    for param_key, value in (params or {}).items():
        if isinstance(value, str | int | float | bool) or value is None:
            safe_params[param_key] = value
    return UserMessageRef(key=key, params=safe_params, severity=severity)


def _normalize_optional_text(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


__all__ = [
    "OPENROUTER_MANAGED_QQ_API_KEY_SECRET",
    "QqManagedAuthRequest",
    "QqManagedAuthService",
]
