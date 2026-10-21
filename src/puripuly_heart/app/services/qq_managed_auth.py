from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType

from puripuly_heart.app.ports.broker_client import (
    BrokerClientPort,
    ManagedKeyDeliveryAckMetadata,
    ManagedKeyDeliveryAckResult,
    ManagedOperationResumeRequest,
    ManagedOperationStatusRequest,
    ManagedOperationStatusResult,
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
from puripuly_heart.app.services.managed.managed_operation import (
    DEFAULT_MAX_STATUS_POLL_INTERVAL_MS,
    DEFAULT_MAX_STATUS_POLLS,
    DEFAULT_STATUS_POLL_INTERVAL_MS,
    MANAGED_OPERATION_MAX_CONSECUTIVE_PROBE_FAILURES,
    MANAGED_OPERATION_SOURCE_QQ,
    MANAGED_OPERATION_UNKNOWN_OPERATION_STATUS,
    ManagedOperationIdentity,
    ManagedOperationTokenReadError,
    ManagedOperationTokenStoreError,
    ProgressSink,
    clear_pending_operation_if_source,
    clear_resume_token,
    emit_progress,
    new_managed_operation_id,
    new_managed_operation_resume_token,
    other_source_pending_operation,
    read_pending_operation,
    read_resume_token,
    status_poll_delay_ms,
    store_resume_token,
    update_pending_operation_state,
    write_pending_operation,
)
from puripuly_heart.app.services.managed_auth_claims import (
    MANAGED_AUTH_CLAIM_SOURCE_QQ,
    ManagedAuthClaimGuard,
)
from puripuly_heart.app.services.managed_key_delivery_ack import (
    ACK_RECONCILIATION_STATUSES,
    ACK_SOURCE_QQ,
    ManagedKeyDeliveryAckService,
    ManagedKeyDeliveryAckTokenStoreError,
    other_source_pending_delivery_ack,
    read_any_pending_delivery_ack,
)
from puripuly_heart.core.diagnostics import (
    emit_failure,
    safe_exception_fields,
    sanitize_diagnostics,
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
    TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
    DiagnosticCategory,
    DiagnosticFieldValue,
    ErrorDiagnostics,
    Severity,
    TransactionResult,
    UserMessageRef,
)
from puripuly_heart.core.openrouter.openrouter_credentials import (
    OPENROUTER_MANAGED_QQ_API_KEY_SECRET,
)
from puripuly_heart.core.qq_auth_diagnostics import (
    exception_fields,
    log_qq_diagnostics,
    safe_fields,
)


class QqManagedAuthFailure(RuntimeError):
    def __init__(self, diagnostics: ErrorDiagnostics) -> None:
        self.diagnostics = diagnostics
        super().__init__(diagnostics.code)


def _qq_exception_diagnostics(
    phase: str, exception: BaseException, *, code: str = "qq_application_failed"
) -> ErrorDiagnostics:
    diagnostics = getattr(exception, "failure_diagnostics", None) or getattr(
        exception, "diagnostics", None
    )
    if isinstance(diagnostics, ErrorDiagnostics):
        return sanitize_diagnostics(diagnostics)
    fields = {"phase": phase, **exception_fields(exception), **safe_exception_fields(exception)}
    kind = fields.get("network_error_kind")
    transport = kind in {"timeout", "proxy", "tls", "dns", "network"}
    return _diagnostics(
        operation=phase,
        code=f"broker_{kind}" if phase == "broker_issue" and transport else code,
        category=(
            "timeout"
            if kind == "timeout"
            else (
                "network"
                if transport
                else (
                    "unknown"
                    if isinstance(exception, (TypeError, ValueError, KeyError, RuntimeError))
                    else DIAGNOSTIC_CATEGORY_TRANSACTION
                )
            )
        ),
        fields=fields,
    )


def _observe_qq_failure(phase: str, exception: BaseException) -> None:
    emit_failure(
        logging.getLogger(__name__),
        _qq_exception_diagnostics(phase, exception),
        severity="warning",
        disposition="degraded",
    )


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
    progress_sink: ProgressSink | None = field(default=None, repr=False)
    max_status_polls: int | None = None
    status_poll_interval_ms: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_fields(self.metadata))


@dataclass(frozen=True, slots=True)
class QqManagedAuthService:
    broker_client: BrokerClientPort
    secret_store: SecretStorePort
    managed_state: ManagedIdentityStatePort
    claim_guard: ManagedAuthClaimGuard | None = None
    delivery_ack_service: ManagedKeyDeliveryAckService | None = None
    assertion_result_sink: Callable[[QqManagedAssertionResult], None] | None = field(
        default=None,
        repr=False,
    )
    ack_result_sink: Callable[[ManagedKeyDeliveryAckResult], None] | None = field(
        default=None,
        repr=False,
    )

    async def authenticate(self, request: QqManagedAuthRequest) -> TransactionResult:
        try:
            result = await self._authenticate(request)
        except Exception as exc:
            diagnostic = _qq_exception_diagnostics("qq_authentication", exc)
            emit_failure(logging.getLogger(__name__), diagnostic)
            raise QqManagedAuthFailure(diagnostic) from exc
        if result.status == TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED:
            log_qq_diagnostics(
                result.diagnostics,
                fallback_code="authentication_succeeded",
                level=logging.INFO,
            )
        else:
            if result.diagnostics is not None:
                emit_failure(
                    logging.getLogger(__name__),
                    result.diagnostics,
                    severity=(
                        "warning"
                        if result.status == TRANSACTION_STATUS_REMOTE_DELIVERY_ACK_PENDING
                        else "error"
                    ),
                    disposition=(
                        "degraded"
                        if result.status == TRANSACTION_STATUS_REMOTE_DELIVERY_ACK_PENDING
                        else "failed"
                    ),
                )
        for secondary in result.secondary_diagnostics:
            emit_failure(
                logging.getLogger(__name__), secondary, severity="warning", disposition="degraded"
            )
        return result

    async def _authenticate(self, request: QqManagedAuthRequest) -> TransactionResult:
        other_source_refusal = _refuse_other_source_pending(self.managed_state)
        if other_source_refusal is not None:
            return other_source_refusal
        recovery_result = await self._recover_pending_delivery_ack()
        if recovery_result is not None:
            return recovery_result

        recovered_result = await self._recover_pending_operation(request)
        if recovered_result is not None:
            return recovered_result

        claim_result = await self._preflight_claim_source()
        if claim_result is not None:
            return claim_result

        emit_progress(request.progress_sink, "preparing")
        assured_operation = await self._ensure_pending_operation(request)
        if isinstance(assured_operation, TransactionResult):
            return assured_operation
        operation, resume_token = assured_operation
        broker_result = await self._assert_qq_identity(
            request,
            operation_id=operation.operation_id,
            resume_token=resume_token,
        )
        if isinstance(broker_result, TransactionResult):
            return await self._reconcile_assert_result(
                request,
                assertion=broker_result,
                operation=operation,
                resume_token=resume_token,
            )

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

        if broker_result.delivery_ack is None or (
            broker_result.delivery_ack.source != ACK_SOURCE_QQ
            or broker_result.delivery_ack.managed_credential_ref
            != broker_result.entitlement.managed_credential_ref
            or not broker_result.delivery_ack.managed_credential_ref
            or not broker_result.delivery_ack.delivery_id
            or not broker_result.delivery_ack.delivery_ack_token
        ):
            return TransactionResult(
                status=TRANSACTION_STATUS_REMOTE_ACTIVE_LOCAL_MISSING,
                message=_qq_operation_message("qq_managed_auth.error.action_required"),
                diagnostics=_diagnostics(
                    operation="validate_managed_delivery",
                    code=(
                        "managed_delivery_contract_missing"
                        if broker_result.delivery_ack is None
                        else "managed_delivery_contract_mismatch"
                    ),
                    category=DIAGNOSTIC_CATEGORY_TRANSACTION,
                    fields={
                        "phase": "broker_assertion",
                        "secret_write_succeeded": False,
                        "settings_commit_succeeded": False,
                    },
                ),
            )
        self._emit_assertion_result(broker_result)
        secret_snapshot = await self._snapshot_secret()
        if isinstance(secret_snapshot, TransactionResult):
            return secret_snapshot
        state_snapshot = self.managed_state.snapshot()

        if broker_result.delivery_ack is not None:
            secret_write = await self._write_managed_secret(broker_result.managed_secret_key)
            if isinstance(secret_write, TransactionResult):
                return secret_write
            try:
                await self._delivery_ack_service().bind_credential(
                    broker_result.delivery_ack, broker_result.managed_secret_key
                )
            except Exception as exc:
                return _delivery_ack_pending_result(
                    ack_status="credential_binding_store_failed",
                    diagnostics=_qq_exception_diagnostics("credential_binding", exc),
                )
        if broker_result.delivery_ack is not None:
            try:
                await self._delivery_ack_service().store_pending(broker_result.delivery_ack)
            except ManagedKeyDeliveryAckTokenStoreError as exc:
                return _remote_active_local_missing_result(
                    operation="store_qq_delivery_ack_token",
                    code="qq_delivery_ack_token_store_failed",
                    phase="delivery_ack_token_store",
                    failure_subcode="key_unavailable",
                    secret_write_succeeded=True,
                    settings_commit_succeeded=False,
                    rollback_succeeded=None,
                    compensation_succeeded=None,
                    retry_after_ms=None,
                    diagnostics=_qq_exception_diagnostics(
                        "delivery_ack_token_store",
                        exc,
                        code="qq_delivery_ack_token_store_failed",
                    ),
                )
            self._apply_entitlement_snapshot(broker_result.entitlement)
            if self.claim_guard is not None:
                self.claim_guard.record_success(MANAGED_AUTH_CLAIM_SOURCE_QQ)
            try:
                self.managed_state.persist()
            except Exception as exc:
                primary = _qq_exception_diagnostics(
                    "delivery_ack_pending_persist",
                    exc,
                    code="qq_pending_delivery_ack_persist_failed",
                )
                secondary = tuple(
                    sanitize_diagnostics(cause)
                    for cause in getattr(exc, "secondary_diagnostics", ())
                    if isinstance(cause, ErrorDiagnostics)
                )
                _, restore_causes = self._restore_state_snapshot(state_snapshot)
                secondary += tuple(
                    replace(cause, cause_id=cause.cause_id or primary.diagnostic_id)
                    for cause in restore_causes
                )
                return _remote_active_local_missing_result(
                    operation="persist_qq_pending_delivery_ack",
                    code="qq_pending_delivery_ack_persist_failed",
                    phase="delivery_ack_pending_persist",
                    failure_subcode="key_unavailable",
                    secret_write_succeeded=True,
                    settings_commit_succeeded=False,
                    rollback_succeeded=None,
                    compensation_succeeded=None,
                    retry_after_ms=None,
                    diagnostics=primary,
                    secondary_diagnostics=secondary,
                    secondary_omitted_count=getattr(exc, "secondary_omitted_count", 0),
                )
            ack_result = await self._delivery_ack_service().retry_pending()
            self._consume_ack_result(ack_result.ack_result)
            if not ack_result.succeeded:
                return _delivery_ack_pending_result(
                    ack_status=ack_result.status,
                    diagnostics=ack_result.diagnostics,
                    secondary_diagnostics=ack_result.secondary_diagnostics,
                    secondary_omitted_count=ack_result.secondary_omitted_count,
                    secret_write_succeeded=True,
                    settings_commit_succeeded=True,
                )
            await self._clear_qq_operation(preserve_active_credential=True)
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
        self._consume_ack_result(result.ack_result)
        if not result.succeeded:
            if (
                result.status in ACK_RECONCILIATION_STATUSES
                and read_pending_operation(self.managed_state, source=MANAGED_OPERATION_SOURCE_QQ)
                is not None
            ):
                return None
            if result.status in ACK_RECONCILIATION_STATUSES:
                return TransactionResult(
                    status=TRANSACTION_STATUS_REMOTE_ACTIVE_LOCAL_MISSING,
                    message=_qq_operation_message("qq_managed_auth.error.action_required"),
                    diagnostics=result.diagnostics,
                )
            return _delivery_ack_pending_result(
                ack_status=result.status,
                diagnostics=result.diagnostics,
                secondary_diagnostics=result.secondary_diagnostics,
                secondary_omitted_count=result.secondary_omitted_count,
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
                    "ack_status": result.status,
                    "secret_write_succeeded": True,
                    "settings_commit_succeeded": True,
                },
            ),
        )

    async def _assert_qq_identity(
        self,
        request: QqManagedAuthRequest,
        *,
        operation_id: str | None = None,
        resume_token: str | None = None,
    ) -> QqManagedAssertionResult | TransactionResult:
        try:
            result = await self.broker_client.assert_qq_managed_identity(
                QqManagedAssertionRequest(
                    qq_identity=request.qq_identity,
                    credential=request.credential,
                    asserted_at=request.asserted_at,
                    metadata=request.metadata,
                    installation_id=_normalize_optional_text(self.managed_state.installation_id),
                    operation_id=operation_id,
                    resume_token=resume_token,
                )
            )
        except Exception as exc:
            primary = _qq_exception_diagnostics("broker_issue", exc, code="qq_assertion_exception")
            result = _broker_failure_result(
                failure_subcode="broker_unavailable",
                code="qq_assertion_exception",
                retry_after_ms=None,
                diagnostics=primary,
            )
            return result

        if not result.succeeded:
            failure = _broker_failure_result(
                failure_subcode=result.failure_subcode or "broker_unavailable",
                code="qq_assertion_failed",
                retry_after_ms=result.retry_after_ms,
                diagnostics=result.diagnostics,
                diagnostic_fields=(
                    result.diagnostics.fields if result.diagnostics is not None else {}
                ),
                status_code=(
                    result.diagnostics.status_code if result.diagnostics is not None else None
                ),
            )
            return failure

        return result

    def _installation_id(self) -> str | None:
        installation_id = _normalize_optional_text(self.managed_state.installation_id)
        return installation_id or None

    def _max_polls(self, request: QqManagedAuthRequest) -> int:
        if request.max_status_polls is not None and request.max_status_polls >= 0:
            return request.max_status_polls
        return DEFAULT_MAX_STATUS_POLLS

    def _poll_interval_ms(self, request: QqManagedAuthRequest) -> int:
        if request.status_poll_interval_ms is not None and request.status_poll_interval_ms >= 0:
            return request.status_poll_interval_ms
        return DEFAULT_STATUS_POLL_INTERVAL_MS

    async def _clear_qq_operation(
        self,
        *,
        primary: ErrorDiagnostics | None = None,
        preserve_active_credential: bool = False,
    ) -> tuple[ErrorDiagnostics, ...]:
        causes = ()
        cleared = False
        previous = self.managed_state.snapshot()
        try:
            if not preserve_active_credential:
                await self._delivery_ack_service().clear_failed_credential_projection(ACK_SOURCE_QQ)
            cleared = clear_pending_operation_if_source(
                self.managed_state, MANAGED_OPERATION_SOURCE_QQ
            )
            if cleared:
                if self.managed_state.pending_delivery_ack_source == ACK_SOURCE_QQ:
                    self.managed_state.pending_delivery_ack_source = None
                    self.managed_state.pending_delivery_ack_delivery_id = None
                    self.managed_state.pending_delivery_ack_managed_credential_ref = None
                    self.managed_state.pending_delivery_ack_expires_at = None
                self.managed_state.persist()
        except Exception as exc:
            self.managed_state.restore(previous)
            cleared = False
            cause = _qq_exception_diagnostics("managed_operation_clear", exc)
            causes = (
                replace(
                    cause, cause_id=primary.diagnostic_id if primary is not None else cause.cause_id
                ),
            )
        if cleared:
            try:
                await clear_resume_token(self.secret_store)
                if previous.pending_delivery_ack_source == ACK_SOURCE_QQ:
                    await self.secret_store.clear_secret("openrouter_managed_qq_delivery_ack_token")
            except Exception as exc:
                cause = _qq_exception_diagnostics("managed_operation_clear", exc)
                causes += (
                    replace(
                        cause,
                        cause_id=(
                            primary.diagnostic_id
                            if primary is not None
                            else causes[0].diagnostic_id if causes else cause.cause_id
                        ),
                    ),
                )
        for cause in causes:
            emit_failure(
                logging.getLogger(__name__), cause, severity="warning", disposition="degraded"
            )
        return causes

    async def _ensure_pending_operation(
        self,
        request: QqManagedAuthRequest,
    ) -> tuple[ManagedOperationIdentity, str] | TransactionResult:
        installation_id = self._installation_id()
        if not installation_id:
            return _local_failure_result(
                operation="ensure_qq_managed_operation",
                code="qq_managed_operation_installation_unavailable",
                phase="persist_managed_operation",
            )
        pending = read_pending_operation(self.managed_state, source=MANAGED_OPERATION_SOURCE_QQ)
        try:
            existing_token = await read_resume_token(self.secret_store)
        except ManagedOperationTokenReadError as exc:
            return _local_failure_result(
                operation="read_resume_token",
                code="managed_operation_token_read_failed",
                phase="secret_observation",
                diagnostics=exc.diagnostics,
            )
        if (
            pending is not None
            and pending.installation_id == installation_id
            and existing_token is not None
        ):
            return (pending, existing_token)
        if pending is not None or existing_token is not None:
            await self._clear_qq_operation()
        resume_token = new_managed_operation_resume_token()
        try:
            await store_resume_token(self.secret_store, resume_token)
        except ManagedOperationTokenStoreError as exc:
            return _local_failure_result(
                operation="ensure_qq_managed_operation",
                code="qq_managed_operation_token_store_failed_before_assert",
                phase="persist_managed_operation",
                diagnostics=_qq_exception_diagnostics("persist_managed_operation", exc),
            )
        operation = ManagedOperationIdentity(
            operation_id=new_managed_operation_id(),
            source=MANAGED_OPERATION_SOURCE_QQ,
            installation_id=installation_id,
            last_known_state=None,
        )
        write_pending_operation(self.managed_state, operation)
        try:
            self.managed_state.persist()
        except Exception as exc:
            primary = _qq_exception_diagnostics("persist_managed_operation", exc)
            secondary = await self._clear_qq_operation(primary=primary)
            return _local_failure_result(
                operation="ensure_qq_managed_operation",
                code="qq_managed_operation_persist_failed_before_assert",
                phase="persist_managed_operation",
                diagnostics=primary,
                secondary_diagnostics=secondary,
            )
        return (operation, resume_token)

    async def _recover_pending_operation(
        self,
        request: QqManagedAuthRequest,
    ) -> TransactionResult | None:
        pending = read_pending_operation(self.managed_state, source=MANAGED_OPERATION_SOURCE_QQ)
        if pending is None:
            return None
        installation_id = self._installation_id()
        if not installation_id or pending.installation_id != installation_id:
            return _qq_operation_action_required_result(
                request,
                operation=pending,
                probe=ManagedOperationStatusResult(
                    succeeded=False,
                    operation_status="CREATE_UNKNOWN",
                    client_action="action_required",
                ),
                code="qq_managed_operation_installation_mismatch",
            )
        try:
            resume_token = await read_resume_token(self.secret_store)
        except ManagedOperationTokenReadError as exc:
            return replace(
                _qq_operation_recovery_pending_result(request, operation=pending, polls=0),
                diagnostics=exc.diagnostics,
            )
        if resume_token is None:
            return _qq_operation_action_required_result(
                request,
                operation=pending,
                probe=ManagedOperationStatusResult(
                    succeeded=False,
                    operation_status="CREATE_UNKNOWN",
                    client_action="action_required",
                ),
                code="qq_managed_operation_resume_token_missing",
            )
        emit_progress(request.progress_sink, "recovering")
        return await self._drive_qq_operation_recovery(
            request,
            operation=pending,
            resume_token=resume_token,
            first_probe=None,
            unknown_outcome=True,
            original_failure=None,
        )

    async def _reconcile_assert_result(
        self,
        request: QqManagedAuthRequest,
        *,
        assertion: TransactionResult,
        operation: ManagedOperationIdentity,
        resume_token: str,
    ) -> TransactionResult:
        unknown = _assertion_failure_unknown_outcome(assertion)
        if assertion.diagnostics is not None and (
            assertion.diagnostics.fields.get("dispatch_certainty") == "not_dispatched"
        ):
            await self._clear_qq_operation()
            return assertion
        try:
            first_probe = await self._fetch_operation_status(
                operation.operation_id,
                resume_token,
                operation.installation_id,
            )
        except Exception as exc:
            _observe_qq_failure("managed_recovery", exc)
            first_probe = None
        if first_probe is not None and (
            first_probe.operation_status == "FAILED"
            or first_probe.client_action == "action_required"
        ):
            if first_probe.operation_status == "FAILED":
                await self._clear_qq_operation()
            return _qq_operation_action_required_result(
                request, operation=operation, probe=first_probe
            )
        if first_probe is None or (
            not first_probe.succeeded
            and first_probe.operation_status != MANAGED_OPERATION_UNKNOWN_OPERATION_STATUS
        ):
            if unknown:
                emit_progress(request.progress_sink, "recovering")
                recovering = await self._drive_qq_operation_recovery(
                    request,
                    operation=operation,
                    resume_token=resume_token,
                    first_probe=first_probe,
                    unknown_outcome=True,
                    original_failure=None,
                    initial_probe_failures=0,
                )
                return (
                    _with_network_context(recovering, assertion)
                    if recovering is not None
                    else assertion
                )
            return assertion
        if (
            not unknown
            and not first_probe.succeeded
            and first_probe.operation_status == MANAGED_OPERATION_UNKNOWN_OPERATION_STATUS
        ):
            await self._clear_qq_operation()
            return assertion
        emit_progress(request.progress_sink, "recovering")
        recovered = await self._drive_qq_operation_recovery(
            request,
            operation=operation,
            resume_token=resume_token,
            first_probe=first_probe,
            unknown_outcome=unknown,
            original_failure=None if unknown else assertion,
        )
        return _with_network_context(recovered, assertion) if recovered is not None else assertion

    async def _fetch_operation_status(
        self,
        operation_id: str,
        resume_token: str,
        installation_id: str,
    ) -> ManagedOperationStatusResult | None:
        try:
            result = await self.broker_client.get_managed_operation_status(
                ManagedOperationStatusRequest(
                    operation_id=operation_id,
                    installation_id=installation_id,
                    resume_token=resume_token,
                    source=MANAGED_OPERATION_SOURCE_QQ,
                )
            )
            if not result.succeeded and result.diagnostics is not None:
                emit_failure(
                    logging.getLogger(__name__),
                    result.diagnostics,
                    severity="warning",
                    disposition="degraded",
                )
            return result
        except Exception as exc:
            _observe_qq_failure("managed_operation_status", exc)
            return ManagedOperationStatusResult(
                succeeded=False,
                operation_status="CREATE_UNKNOWN",
                client_action="wait",
                diagnostics=_qq_exception_diagnostics("managed_operation_status", exc),
            )

    async def _call_resume(
        self,
        operation: ManagedOperationIdentity,
        resume_token: str,
    ) -> ManagedOperationStatusResult | None:
        try:
            result = await self.broker_client.resume_managed_operation(
                ManagedOperationResumeRequest(
                    operation_id=operation.operation_id,
                    installation_id=operation.installation_id,
                    resume_token=resume_token,
                    source=MANAGED_OPERATION_SOURCE_QQ,
                )
            )
            if not result.succeeded and result.diagnostics is not None:
                emit_failure(
                    logging.getLogger(__name__),
                    result.diagnostics,
                    severity="warning",
                    disposition="degraded",
                )
            return result
        except Exception as exc:
            _observe_qq_failure("managed_operation_resume", exc)
            return ManagedOperationStatusResult(
                succeeded=False,
                operation_status="CREATE_UNKNOWN",
                client_action="wait",
                diagnostics=_qq_exception_diagnostics("managed_operation_resume", exc),
            )

    async def _drive_qq_operation_recovery(
        self,
        request: QqManagedAuthRequest,
        *,
        operation: ManagedOperationIdentity,
        resume_token: str,
        first_probe: ManagedOperationStatusResult | None,
        unknown_outcome: bool,
        original_failure: TransactionResult | None,
        initial_probe_failures: int = 0,
    ) -> TransactionResult | None:
        max_polls = self._max_polls(request)
        base_interval = self._poll_interval_ms(request)
        polls = 0
        consecutive_probe_failures = initial_probe_failures
        last_diagnostics = first_probe.diagnostics if first_probe is not None else None
        pending_probe: ManagedOperationStatusResult | None | str = first_probe or "fetch"
        first_iteration = True
        while True:
            if pending_probe == "fetch":
                if polls >= max_polls:
                    return _qq_operation_recovery_pending_result(
                        request, operation=operation, polls=polls, cause=last_diagnostics
                    )
                if polls > 0:
                    await asyncio.sleep(
                        status_poll_delay_ms(
                            polls - 1, base_interval, DEFAULT_MAX_STATUS_POLL_INTERVAL_MS
                        )
                        / 1000
                    )
                probe = await self._fetch_operation_status(
                    operation.operation_id, resume_token, operation.installation_id
                )
                polls += 1
            else:
                probe = pending_probe
                pending_probe = "fetch"
            assert isinstance(probe, ManagedOperationStatusResult) or probe is None
            if probe is not None and (
                probe.operation_status == "FAILED" or probe.client_action == "action_required"
            ):
                if probe.operation_status == "FAILED":
                    await self._clear_qq_operation()
                return _qq_operation_action_required_result(
                    request, operation=operation, probe=probe
                )
            if probe is None or not probe.succeeded:
                if (
                    probe is not None
                    and probe.operation_status == MANAGED_OPERATION_UNKNOWN_OPERATION_STATUS
                ):
                    if original_failure is not None and first_iteration:
                        await self._clear_qq_operation()
                        return original_failure
                    await self._clear_qq_operation()
                    return None
                consecutive_probe_failures += 1
                if probe is not None and probe.diagnostics is not None:
                    last_diagnostics = probe.diagnostics
                if consecutive_probe_failures >= MANAGED_OPERATION_MAX_CONSECUTIVE_PROBE_FAILURES:
                    return _qq_operation_recovery_pending_result(
                        request, operation=operation, polls=polls, cause=last_diagnostics
                    )
                if original_failure is not None and first_iteration and not unknown_outcome:
                    return original_failure
                first_iteration = False
                continue
            first_iteration = False
            consecutive_probe_failures = 0
            update_pending_operation_state(self.managed_state, probe.operation_status)
            try:
                self.managed_state.persist()
            except Exception as exc:
                _observe_qq_failure("managed_recovery", exc)
                return _qq_operation_recovery_pending_result(
                    request, operation=operation, polls=polls
                )
            if probe.operation_status == "FAILED":
                await self._clear_qq_operation()
                return _qq_operation_action_required_result(
                    request, operation=operation, probe=probe
                )
            if probe.operation_status == "ACTIVE":
                return await self._recover_active_qq_operation(
                    request,
                    operation=operation,
                    probe=probe,
                )
            converted = _status_result_to_qq_credential(probe)
            if converted is not None:
                return await self._finalize_qq_resumed_credential(
                    request,
                    operation=operation,
                    credential=converted,
                )
            if probe.client_action == "acknowledge_delivery":
                acknowledged = await self._recover_qq_acknowledge_delivery(
                    request,
                    operation=operation,
                    resume_token=resume_token,
                    probe=probe,
                )
                if acknowledged is not None:
                    return acknowledged
                pending_probe = "fetch"
                continue
            if probe.client_action == "retry_authorized":
                resumed = await self._call_resume(operation, resume_token)
                if resumed is not None and resumed.diagnostics is not None:
                    last_diagnostics = resumed.diagnostics
                if resumed is not None and (
                    resumed.operation_status == "FAILED"
                    or resumed.client_action == "action_required"
                ):
                    if resumed.operation_status == "FAILED":
                        await self._clear_qq_operation()
                    return _qq_operation_action_required_result(
                        request, operation=operation, probe=resumed
                    )
                if resumed is not None and resumed.succeeded:
                    if resumed.operation_status == "FAILED":
                        await self._clear_qq_operation()
                        return _qq_operation_action_required_result(
                            request, operation=operation, probe=resumed
                        )
                    resumed_converted = _status_result_to_qq_credential(resumed)
                    if resumed_converted is not None:
                        return await self._finalize_qq_resumed_credential(
                            request,
                            operation=operation,
                            credential=resumed_converted,
                        )
                pending_probe = "fetch"
                continue
            if probe.client_action == "action_required":
                return _qq_operation_action_required_result(
                    request, operation=operation, probe=probe
                )
            emit_progress(request.progress_sink, "recovering")
            pending_probe = "fetch"

    async def _recover_active_qq_operation(
        self,
        request: QqManagedAuthRequest,
        *,
        operation: ManagedOperationIdentity,
        probe: ManagedOperationStatusResult,
    ) -> TransactionResult:
        try:
            binding, credential = await self._delivery_ack_service().credential_binding(
                ACK_SOURCE_QQ
            )
        except Exception as exc:
            return replace(
                _qq_operation_recovery_pending_result(request, operation=operation, polls=0),
                diagnostics=_qq_exception_diagnostics("secret_observation", exc),
            )
        remote_ref = probe.managed_credential_ref or (
            probe.attempt.managed_credential_ref if probe.attempt is not None else None
        )
        if (
            not remote_ref
            or binding is None
            or not binding.matches(
                credential=credential,
                source=ACK_SOURCE_QQ,
                installation_id=operation.installation_id,
                operation_id=operation.operation_id,
                managed_credential_ref=remote_ref,
            )
        ):
            return _qq_operation_action_required_result(
                request,
                operation=operation,
                probe=probe,
                code="qq_managed_operation_active_key_unbound",
            )
        self.managed_state.active_managed_credential_ref = binding.managed_credential_ref
        self.managed_state.persist()
        if self.claim_guard is not None:
            self.claim_guard.record_success(MANAGED_AUTH_CLAIM_SOURCE_QQ)
            try:
                self.claim_guard.managed_state.persist()
            except Exception as exc:
                _observe_qq_failure("managed_recovery", exc)
                return _qq_operation_recovery_pending_result(request, operation=operation, polls=0)
        await self._clear_qq_operation(preserve_active_credential=True)
        return _qq_operation_active_recovered_result(request, operation=operation, probe=probe)

    async def _recover_qq_acknowledge_delivery(
        self,
        request: QqManagedAuthRequest,
        *,
        operation: ManagedOperationIdentity,
        resume_token: str,
        probe: ManagedOperationStatusResult,
    ) -> TransactionResult | None:
        service = self.delivery_ack_service
        if service is None:
            service = ManagedKeyDeliveryAckService(
                broker_client=self.broker_client,
                secret_store=self.secret_store,
                managed_state=self.managed_state,
            )
        recovered = await service.recover_operation_delivery(ACK_SOURCE_QQ, probe)
        if recovered is not None:
            self._consume_ack_result(recovered.ack_result)
            if recovered.succeeded:
                if self.claim_guard is not None:
                    self.claim_guard.record_success(MANAGED_AUTH_CLAIM_SOURCE_QQ)
                    try:
                        self.claim_guard.managed_state.persist()
                    except Exception as exc:
                        _observe_qq_failure("managed_recovery", exc)
                        return _qq_operation_recovery_pending_result(
                            request, operation=operation, polls=0
                        )
                await self._clear_qq_operation(preserve_active_credential=True)
                return TransactionResult(
                    status=TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
                    message=(
                        recovered.ack_result.message if recovered.ack_result is not None else None
                    ),
                    diagnostics=_diagnostics(
                        operation="recover_qq_managed_operation",
                        code="qq_managed_operation_delivery_ack_recovered",
                        category=DIAGNOSTIC_CATEGORY_TRANSACTION,
                        fields={
                            "phase": "remote_managed_operation_recovery",
                            "secret_key": OPENROUTER_MANAGED_QQ_API_KEY_SECRET,
                            "managed_operation_id": operation.operation_id,
                            "operation_status": probe.operation_status,
                            "secret_write_succeeded": True,
                            "settings_commit_succeeded": True,
                        },
                    ),
                )
            if recovered.status not in ACK_RECONCILIATION_STATUSES:
                return _qq_operation_recovery_pending_result(
                    request, operation=operation, polls=0, cause=recovered.diagnostics
                )
        resumed = await self._call_resume(operation, resume_token)
        if resumed is not None and (
            resumed.operation_status == "FAILED" or resumed.client_action == "action_required"
        ):
            if resumed.operation_status == "FAILED":
                await self._clear_qq_operation()
            return _qq_operation_action_required_result(request, operation=operation, probe=resumed)
        if resumed is not None and resumed.succeeded:
            converted = _status_result_to_qq_credential(resumed)
            if converted is not None:
                return await self._finalize_qq_resumed_credential(
                    request,
                    operation=operation,
                    credential=converted,
                )
        return None

    async def _finalize_qq_resumed_credential(
        self,
        request: QqManagedAuthRequest,
        *,
        operation: ManagedOperationIdentity,
        credential: _QqResumedCredential,
    ) -> TransactionResult:
        if credential.delivery_ack.source != ACK_SOURCE_QQ or (
            credential.delivery_ack.managed_credential_ref != credential.managed_credential_ref
        ):
            return _local_failure_result(
                operation="validate_managed_delivery",
                code="managed_delivery_contract_mismatch",
                phase="managed_recovery",
            )
        secret_snapshot = await self._snapshot_secret()
        if isinstance(secret_snapshot, TransactionResult):
            return secret_snapshot
        state_snapshot = self.managed_state.snapshot()
        delivery_ack = credential.delivery_ack
        try:
            await self._delivery_ack_service().store_pending(delivery_ack)
        except ManagedKeyDeliveryAckTokenStoreError as exc:
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
                diagnostics=_qq_exception_diagnostics(
                    "delivery_ack_token_store",
                    exc,
                    code="qq_delivery_ack_token_store_failed_before_local_key_write",
                ),
            )
        secret_write = await self._write_managed_secret(credential.managed_secret_key)
        if isinstance(secret_write, TransactionResult):
            return secret_write
        try:
            await self._delivery_ack_service().bind_credential(
                delivery_ack, credential.managed_secret_key
            )
        except Exception as exc:
            return _delivery_ack_pending_result(
                ack_status="credential_binding_store_failed",
                diagnostics=_qq_exception_diagnostics("credential_binding", exc),
            )
        self._apply_entitlement_snapshot(
            QqManagedEntitlementSnapshot(
                qq_subject_ref=credential.qq_subject_ref or request.qq_identity,
                managed_credential_ref=credential.managed_credential_ref,
                expires_at=credential.expires_at,
                openrouter_user_id=credential.openrouter_user_id,
            )
        )
        if self.claim_guard is not None:
            self.claim_guard.record_success(MANAGED_AUTH_CLAIM_SOURCE_QQ)
        try:
            self.managed_state.persist()
        except Exception as exc:
            return self._restore_after_delivery_persist_failure(
                state_snapshot=state_snapshot,
                primary=_qq_exception_diagnostics(
                    "local_settings_commit", exc, code="settings_commit_failed"
                ),
                upstream_secondaries=tuple(
                    sanitize_diagnostics(cause)
                    for cause in getattr(exc, "secondary_diagnostics", ())
                    if isinstance(cause, ErrorDiagnostics)
                ),
                upstream_omitted_count=getattr(exc, "secondary_omitted_count", 0),
            )
        ack_result = await self._delivery_ack_service().retry_pending()
        self._consume_ack_result(ack_result.ack_result)
        if not ack_result.succeeded:
            return _delivery_ack_pending_result(
                ack_status=ack_result.status,
                diagnostics=ack_result.diagnostics,
                secondary_diagnostics=ack_result.secondary_diagnostics,
                secondary_omitted_count=ack_result.secondary_omitted_count,
            )
        await self._clear_qq_operation(preserve_active_credential=True)
        return TransactionResult(
            status=TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
            message=_message("qq_managed_auth.success", severity=SEVERITY_INFO),
            diagnostics=_diagnostics(
                operation="persist_qq_managed_auth",
                code="qq_managed_auth_resumed_credential_acknowledged",
                category=DIAGNOSTIC_CATEGORY_TRANSACTION,
                fields={
                    "phase": "remote_managed_operation_recovery",
                    "secret_key": OPENROUTER_MANAGED_QQ_API_KEY_SECRET,
                    "managed_operation_id": operation.operation_id,
                    "secret_write_succeeded": True,
                    "settings_commit_succeeded": True,
                },
            ),
        )

    async def _snapshot_secret(self) -> SecretSnapshot | TransactionResult:
        try:
            return await self.secret_store.snapshot_secret(OPENROUTER_MANAGED_QQ_API_KEY_SECRET)
        except Exception as exc:
            diagnostics = _qq_exception_diagnostics(
                "snapshot_secret", exc, code="qq_secret_snapshot_failed"
            )
            return _secret_failure_result(
                operation="snapshot_qq_managed_secret",
                code="qq_secret_snapshot_failed",
                phase="secret_snapshot",
                diagnostics=diagnostics,
            )

    async def _write_managed_secret(self, managed_secret_key: str) -> TransactionResult | None:
        try:
            result = await self.secret_store.set_secret(
                OPENROUTER_MANAGED_QQ_API_KEY_SECRET,
                managed_secret_key,
            )
        except Exception as exc:
            diagnostics = _qq_exception_diagnostics(
                "set_secret", exc, code="qq_secret_write_exception"
            )
            return _secret_failure_result(
                operation="set_qq_managed_secret",
                code="qq_secret_write_exception",
                phase="secret_write",
                diagnostics=diagnostics,
            )

        if not result.succeeded:
            return _secret_failure_result(
                operation="set_qq_managed_secret",
                code="qq_secret_write_failed",
                phase="secret_write",
                diagnostics=result.diagnostics,
                message=result.message,
            )
        return None

    def _emit_assertion_result(self, result: QqManagedAssertionResult) -> None:
        if self.assertion_result_sink is None:
            return
        try:
            self.assertion_result_sink(result)
        except Exception as exc:
            _observe_qq_failure("auth_notification", exc)
            return

    def _consume_ack_result(self, result: ManagedKeyDeliveryAckResult | None) -> None:
        if result is None:
            return
        if not result.succeeded and result.diagnostics is not None:
            emit_failure(
                logging.getLogger(__name__),
                result.diagnostics,
                severity="warning",
                disposition="degraded",
            )
        if self.ack_result_sink is not None:
            try:
                self.ack_result_sink(result)
            except Exception as exc:
                _observe_qq_failure("auth_notification", exc)
                return

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

    def _restore_after_delivery_persist_failure(
        self,
        *,
        state_snapshot: ManagedIdentitySnapshot,
        primary: ErrorDiagnostics,
        upstream_secondaries: tuple[ErrorDiagnostics, ...] = (),
        upstream_omitted_count: int = 0,
    ) -> TransactionResult:
        state_rollback_succeeded, state_causes = self._restore_state_snapshot(state_snapshot)
        return TransactionResult(
            status=TRANSACTION_STATUS_REMOTE_ACTIVE_LOCAL_MISSING,
            message=_message(
                "qq_managed_auth.settings_commit_failed",
                severity=SEVERITY_ERROR,
            ),
            diagnostics=replace(
                primary,
                fields={
                    **primary.fields,
                    "secret_key": OPENROUTER_MANAGED_QQ_API_KEY_SECRET,
                    "secret_write_succeeded": True,
                    "settings_commit_succeeded": False,
                    "state_rollback_succeeded": state_rollback_succeeded,
                },
            ),
            secondary_diagnostics=tuple(
                replace(cause, cause_id=cause.cause_id or primary.diagnostic_id)
                for cause in (
                    *upstream_secondaries,
                    *state_causes,
                )
            ),
            secondary_omitted_count=upstream_omitted_count,
        )

    def _restore_state_snapshot(
        self, snapshot: ManagedIdentitySnapshot
    ) -> tuple[bool, tuple[ErrorDiagnostics, ...]]:
        try:
            self.managed_state.restore(snapshot)
        except Exception as exc:
            return False, (
                _qq_exception_diagnostics(
                    "restore_memory", exc, code="settings_memory_restore_failed"
                ),
            )
        return True, ()


@dataclass(frozen=True, slots=True)
class _QqResumedCredential:
    managed_secret_key: str
    delivery_ack: ManagedKeyDeliveryAckMetadata
    managed_credential_ref: str | None
    expires_at: str | None
    openrouter_user_id: str | None
    qq_subject_ref: str | None


def _status_result_to_qq_credential(
    probe: ManagedOperationStatusResult,
) -> _QqResumedCredential | None:
    if not probe.succeeded:
        return None
    if not probe.managed_secret_key:
        return None
    if probe.delivery_ack is None:
        return None
    return _QqResumedCredential(
        managed_secret_key=probe.managed_secret_key,
        delivery_ack=probe.delivery_ack,
        managed_credential_ref=probe.managed_credential_ref,
        expires_at=probe.expires_at,
        openrouter_user_id=probe.openrouter_user_id,
        qq_subject_ref=probe.qq_subject_ref,
    )


def _assertion_failure_unknown_outcome(assertion: TransactionResult) -> bool:
    diagnostics = assertion.diagnostics
    if diagnostics is None:
        return False
    if diagnostics.fields.get("dispatch_certainty") == "not_dispatched":
        return False
    if diagnostics.code == "qq_assertion_exception":
        return True
    return (
        isinstance(diagnostics.fields, Mapping)
        and diagnostics.fields.get("qq_failure_subcode") == "broker_unavailable"
    )


def _local_failure_result(
    *,
    operation: str,
    code: str,
    phase: str,
    diagnostics: ErrorDiagnostics | None = None,
    secondary_diagnostics: tuple[ErrorDiagnostics, ...] = (),
    secondary_omitted_count: int = 0,
) -> TransactionResult:
    return TransactionResult(
        status=TRANSACTION_STATUS_PROVIDER_VERIFICATION_FAILED,
        message=_message(
            (
                "qq_managed_auth.configuration_unavailable"
                if code == "qq_managed_operation_installation_unavailable"
                else "qq_managed_auth.local_storage_unavailable"
            ),
            severity=SEVERITY_ERROR,
        ),
        diagnostics=_diagnostics(
            operation=operation,
            code=code,
            category=DIAGNOSTIC_CATEGORY_TRANSACTION,
            cause=diagnostics,
            fields={
                "phase": phase,
                "secret_key": OPENROUTER_MANAGED_QQ_API_KEY_SECRET,
                "secret_write_succeeded": False,
                "settings_commit_succeeded": False,
            },
        ),
        secondary_diagnostics=secondary_diagnostics,
        secondary_omitted_count=secondary_omitted_count,
    )


def _other_source_refusal_result(
    *,
    other_source: str,
    pending_kind: str,
) -> TransactionResult:
    return TransactionResult(
        status=TRANSACTION_STATUS_PROVIDER_VERIFICATION_FAILED,
        message=_message("qq_managed_auth.error.other_source_pending", severity=SEVERITY_ERROR),
        diagnostics=_diagnostics(
            operation="authenticate_qq_managed_identity",
            code=f"qq_other_source_pending_{pending_kind}",
            category=DIAGNOSTIC_CATEGORY_TRANSACTION,
            fields={
                "phase": "other_source_pending",
                "secret_key": OPENROUTER_MANAGED_QQ_API_KEY_SECRET,
                "other_source": other_source,
                "pending_kind": pending_kind,
                "secret_write_succeeded": False,
                "settings_commit_succeeded": False,
            },
        ),
    )


def _refuse_other_source_pending(
    managed_state: ManagedIdentityStatePort,
) -> TransactionResult | None:
    other_ack = other_source_pending_delivery_ack(managed_state, source=ACK_SOURCE_QQ)
    if other_ack is not None:
        return _other_source_refusal_result(
            other_source=other_ack.source, pending_kind="delivery_ack"
        )
    other_operation = other_source_pending_operation(
        managed_state, source=MANAGED_OPERATION_SOURCE_QQ
    )
    if other_operation is not None:
        own_ack = read_any_pending_delivery_ack(managed_state)
        if own_ack is not None and own_ack.source == ACK_SOURCE_QQ:
            return None
        return _other_source_refusal_result(
            other_source=other_operation.source, pending_kind="operation"
        )
    return None


def _qq_operation_message(key: str) -> UserMessageRef:
    return UserMessageRef(key=key, params={}, severity=SEVERITY_ERROR)


def _qq_operation_recovery_pending_result(
    request: QqManagedAuthRequest,
    operation: ManagedOperationIdentity,
    polls: int,
    cause: ErrorDiagnostics | None = None,
) -> TransactionResult:
    del request
    return TransactionResult(
        status=TRANSACTION_STATUS_REMOTE_DELIVERY_ACK_PENDING,
        message=_qq_operation_message("qq_managed_auth.error.recovery_pending"),
        diagnostics=_diagnostics(
            operation="recover_qq_managed_operation",
            code="qq_managed_operation_recovery_pending",
            category=DIAGNOSTIC_CATEGORY_TRANSACTION,
            cause=cause,
            fields={
                "phase": "remote_managed_operation_recovery",
                "secret_key": OPENROUTER_MANAGED_QQ_API_KEY_SECRET,
                "managed_operation_id": operation.operation_id,
                "managed_operation_state": operation.last_known_state,
                "status_polls": polls,
                "secret_write_succeeded": None,
                "settings_commit_succeeded": None,
            },
        ),
    )


def _qq_operation_action_required_result(
    request: QqManagedAuthRequest,
    operation: ManagedOperationIdentity,
    probe: ManagedOperationStatusResult,
    code: str | None = None,
) -> TransactionResult:
    del request
    failed_reason = probe.failed_reason
    if probe.operation_status == "FAILED" and failed_reason == "authorization_expired":
        resolved_code = "qq_managed_operation_authorization_expired"
        message_key = "qq_managed_auth.error.authorization_expired"
    else:
        resolved_code = code or "qq_managed_operation_action_required"
        message_key = "qq_managed_auth.error.action_required"
    fields: dict[str, DiagnosticFieldValue] = {
        "phase": "remote_managed_operation_recovery",
        "secret_key": OPENROUTER_MANAGED_QQ_API_KEY_SECRET,
        "managed_operation_id": operation.operation_id,
        "operation_status": probe.operation_status,
        "client_action": probe.client_action,
        "secret_write_succeeded": None,
        "settings_commit_succeeded": None,
    }
    if failed_reason is not None:
        fields["failed_reason"] = failed_reason
    return TransactionResult(
        status=TRANSACTION_STATUS_REMOTE_ACTIVE_LOCAL_MISSING,
        message=_qq_operation_message(message_key),
        diagnostics=_diagnostics(
            operation="recover_qq_managed_operation",
            code=resolved_code,
            category=DIAGNOSTIC_CATEGORY_TRANSACTION,
            fields=fields,
            cause=probe.diagnostics,
        ),
    )


def _qq_operation_active_recovered_result(
    request: QqManagedAuthRequest,
    operation: ManagedOperationIdentity,
    probe: ManagedOperationStatusResult,
) -> TransactionResult:
    del request
    fields: dict[str, DiagnosticFieldValue] = {
        "phase": "remote_managed_operation_recovery",
        "secret_key": OPENROUTER_MANAGED_QQ_API_KEY_SECRET,
        "managed_operation_id": operation.operation_id,
        "operation_status": probe.operation_status,
        "secret_write_succeeded": True,
        "settings_commit_succeeded": True,
    }
    return TransactionResult(
        status=TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
        message=None,
        diagnostics=_diagnostics(
            operation="recover_qq_managed_operation",
            code="qq_managed_operation_active_recovered",
            category=DIAGNOSTIC_CATEGORY_TRANSACTION,
            fields=fields,
        ),
    )


def _broker_failure_result(
    *,
    failure_subcode: QqManagedAssertionFailureSubcode,
    code: str,
    retry_after_ms: int | None,
    diagnostics: ErrorDiagnostics | None = None,
    diagnostic_fields: Mapping[str, object] | None = None,
    status_code: int | None = None,
) -> TransactionResult:
    category = _category_for_failure_subcode(failure_subcode)
    detail = safe_fields(
        diagnostic_fields or (diagnostics.fields if diagnostics is not None else {})
    )
    if type(status_code) is int:
        detail["http_status"] = status_code
    network_kind = detail.get("network_error_kind")
    if diagnostics is not None:
        diagnostics = replace(
            diagnostics,
            code=(
                f"broker_{network_kind}"
                if network_kind in {"timeout", "proxy", "tls", "dns", "network"}
                else diagnostics.code
            ),
            category=(
                "timeout"
                if network_kind == "timeout"
                else "network" if network_kind in {"proxy", "tls", "dns", "network"} else category
            ),
        )
    message_key = (
        f"qq_managed_auth.network_{network_kind}"
        if network_kind in {"timeout", "proxy", "tls", "dns", "network"}
        else f"qq_managed_auth.{failure_subcode}"
    )
    if detail.get("configuration_error") in {
        "unsupported_proxy_configuration",
        "invalid_proxy_configuration",
        "invalid_tls_configuration",
    }:
        message_key = "qq_managed_auth.network_configuration"
    if detail.get("broker_code") == "internal_error":
        message_key = "qq_managed_auth.broker_unavailable"
    return TransactionResult(
        status=TRANSACTION_STATUS_PROVIDER_VERIFICATION_FAILED,
        message=_message(
            message_key,
            severity=SEVERITY_WARNING if failure_subcode == "rate_limited" else SEVERITY_ERROR,
            params={"retry_after_ms": retry_after_ms} if retry_after_ms is not None else {},
        ),
        diagnostics=_diagnostics(
            operation="assert_qq_managed_identity",
            cause=diagnostics,
            code=(
                f"broker_{network_kind}"
                if network_kind in {"timeout", "proxy", "tls", "dns", "network"}
                else code
            ),
            category=category,
            retry_after_ms=retry_after_ms,
            fields={
                **detail,
                "phase": "broker_assertion",
                "qq_failure_subcode": failure_subcode,
                "secret_write_succeeded": False,
                "settings_commit_succeeded": False,
            },
        ),
    )


def _with_network_context(
    result: TransactionResult, assertion: TransactionResult
) -> TransactionResult:
    if (
        result.status != TRANSACTION_STATUS_REMOTE_DELIVERY_ACK_PENDING
        or assertion.diagnostics is None
    ):
        return result
    detail = safe_fields(assertion.diagnostics.fields)
    if detail.get("network_error_kind") in {"timeout", "proxy", "tls", "dns", "network"}:
        secondary = result.secondary_diagnostics
        if (
            result.diagnostics is not None
            and result.diagnostics.diagnostic_id != assertion.diagnostics.diagnostic_id
        ):
            secondary = (
                replace(result.diagnostics, cause_id=assertion.diagnostics.diagnostic_id),
                *secondary,
            )
        return replace(
            result,
            message=assertion.message,
            diagnostics=assertion.diagnostics,
            secondary_diagnostics=secondary,
        )
    return result


def _secret_failure_result(
    *,
    operation: str,
    code: str,
    phase: str,
    diagnostics: ErrorDiagnostics | None = None,
    message: UserMessageRef | None = None,
) -> TransactionResult:
    return TransactionResult(
        status=TRANSACTION_STATUS_SECRET_WRITE_FAILED,
        message=message or _message("qq_managed_auth.secret_write_failed", severity=SEVERITY_ERROR),
        diagnostics=_diagnostics(
            operation=operation,
            code=code,
            category=DIAGNOSTIC_CATEGORY_TRANSACTION,
            cause=diagnostics,
            fields={
                "phase": phase,
                "secret_key": OPENROUTER_MANAGED_QQ_API_KEY_SECRET,
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
    diagnostics: ErrorDiagnostics | None = None,
    secondary_diagnostics: tuple[ErrorDiagnostics, ...] = (),
    secondary_omitted_count: int = 0,
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
            cause=diagnostics,
            retry_after_ms=retry_after_ms,
            fields=fields,
        ),
        secondary_diagnostics=secondary_diagnostics,
        secondary_omitted_count=secondary_omitted_count,
    )


def _delivery_ack_pending_result(
    *,
    ack_status: str,
    diagnostics: ErrorDiagnostics | None,
    secondary_diagnostics: tuple[ErrorDiagnostics, ...] = (),
    secondary_omitted_count: int = 0,
    secret_write_succeeded: bool | None = None,
    settings_commit_succeeded: bool | None = None,
) -> TransactionResult:
    managed_secret_missing = ack_status == "managed_secret_missing"
    return TransactionResult(
        status=TRANSACTION_STATUS_REMOTE_DELIVERY_ACK_PENDING,
        message=_message(
            (
                "qq_managed_auth.managed_secret_missing"
                if managed_secret_missing
                else "qq_managed_auth.delivery_ack_pending"
            ),
            severity=SEVERITY_ERROR if managed_secret_missing else SEVERITY_WARNING,
        ),
        diagnostics=_diagnostics(
            operation="acknowledge_qq_managed_key_delivery",
            code=(
                "qq_delivery_ack_managed_secret_missing"
                if managed_secret_missing
                else "qq_delivery_ack_pending"
            ),
            category=DIAGNOSTIC_CATEGORY_TRANSACTION,
            cause=diagnostics,
            fields={
                "phase": "secret_write" if managed_secret_missing else "remote_delivery_ack",
                "secret_key": OPENROUTER_MANAGED_QQ_API_KEY_SECRET,
                "secret_write_succeeded": (
                    False if managed_secret_missing else secret_write_succeeded
                ),
                "settings_commit_succeeded": settings_commit_succeeded,
                "ack_status": ack_status,
            },
        ),
        secondary_diagnostics=secondary_diagnostics,
        secondary_omitted_count=secondary_omitted_count,
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
    cause: ErrorDiagnostics | None = None,
) -> ErrorDiagnostics:
    if cause is not None:
        safe = sanitize_diagnostics(cause)
        return replace(
            safe, visibility=DIAGNOSTIC_VISIBILITY_BASIC, fields={**fields, **safe.fields}
        )
    return ErrorDiagnostics(
        component="qq_managed_auth",
        operation=operation,
        code=code,
        category=category,
        visibility=DIAGNOSTIC_VISIBILITY_BASIC,
        content_policy=CONTENT_POLICY_METADATA_ONLY,
        status_code=fields.get("http_status") if type(fields.get("http_status")) is int else None,
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
