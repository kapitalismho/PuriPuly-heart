from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from puripuly_heart.app.ports._settings_values import freeze_settings_values
from puripuly_heart.app.ports.broker_client import (
    BrokerClientPort,
    BrokerIssueRequest,
    BrokerIssueResult,
    ManagedKeyDeliveryAckRequest,
    ManagedKeyDeliveryAckResult,
    ManagedOperationResumeRequest,
    ManagedOperationStatusRequest,
    ManagedOperationStatusResult,
)
from puripuly_heart.app.ports.discord_auth import (
    DiscordAuthPort,
    DiscordAuthRequest,
    DiscordAuthResult,
)
from puripuly_heart.app.ports.managed_identity import (
    ManagedIdentityPort,
    ManagedIdentityPreflightRequest,
    ManagedIdentityPreflightResult,
)
from puripuly_heart.app.ports.managed_identity_state import ManagedIdentityStatePort
from puripuly_heart.app.ports.secret_store import SecretStorePort, SecretWriteResult
from puripuly_heart.app.ports.settings_repository import (
    SettingsCommitRequest,
    SettingsRepositoryPort,
)
from puripuly_heart.core.messages import (
    CONTENT_POLICY_METADATA_ONLY,
    DIAGNOSTIC_CATEGORY_TRANSACTION,
    DIAGNOSTIC_VISIBILITY_BASIC,
    SEVERITY_ERROR,
    TRANSACTION_STATUS_PROVIDER_VERIFICATION_FAILED,
    TRANSACTION_STATUS_REMOTE_ACTIVE_LOCAL_MISSING,
    TRANSACTION_STATUS_REMOTE_DELIVERY_ACK_PENDING,
    TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
    DiagnosticFieldValue,
    ErrorDiagnostics,
    TransactionResult,
    UserMessageRef,
)

from .managed_auth_claims import (
    MANAGED_AUTH_CLAIM_SOURCE_DISCORD,
    OPENROUTER_MANAGED_USER_ID_MAX_LENGTH,
    OPENROUTER_MANAGED_USER_ID_SECRET,
    OPENROUTER_MANAGED_USER_INSTALLATION_ID_SECRET,
    ManagedAuthClaimGuard,
)
from .managed_key_delivery_ack import (
    ACK_SOURCE_DISCORD,
    ManagedKeyDeliveryAckService,
    ManagedKeyDeliveryAckTokenStoreError,
    apply_ack_referral_to_managed_state,
    clear_pending_ack_in_settings_values,
    secret_key_for_ack_source,
    store_pending_ack_in_settings_values,
)
from .managed_operation import (
    DEFAULT_MAX_STATUS_POLLS,
    DEFAULT_MAX_STATUS_POLL_INTERVAL_MS,
    DEFAULT_STATUS_POLL_INTERVAL_MS,
    MANAGED_OPERATION_MAX_CONSECUTIVE_PROBE_FAILURES,
    MANAGED_OPERATION_SOURCE_DISCORD,
    MANAGED_OPERATION_UNKNOWN_OPERATION_STATUS,
    ManagedOperationIdentity,
    ManagedOperationTokenStoreError,
    ProgressSink,
    clear_pending_operation,
    clear_resume_token,
    emit_progress,
    new_managed_operation_id,
    new_managed_operation_resume_token,
    read_pending_operation,
    read_resume_token,
    status_poll_delay_ms,
    store_resume_token,
    update_pending_operation_state,
    write_pending_operation,
)

_SETTINGS_SENSITIVE_KEY_FRAGMENTS = (
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "credential_value",
    "credentialvalue",
    "managed_secret_key",
    "managedsecretkey",
    "password",
    "private_key",
    "privatekey",
    "raw",
    "secret",
    "token",
)


def _freeze_fields(
    values: Mapping[str, DiagnosticFieldValue],
) -> Mapping[str, DiagnosticFieldValue]:
    return MappingProxyType(dict(values))


@dataclass(frozen=True, slots=True)
class ManagedConnectionAuthRequest:
    local_secret_key: str
    settings_values: Mapping[str, object] = field(repr=False)
    expected_settings_revision: str | None
    reason: str | None
    correlation_id: str | None
    broker_metadata: Mapping[str, DiagnosticFieldValue] = field(default_factory=dict, repr=False)
    progress_sink: ProgressSink | None = field(default=None, repr=False)
    max_status_polls: int | None = None
    status_poll_interval_ms: int | None = None
    ack_result_sink: object | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "settings_values", freeze_settings_values(self.settings_values))
        object.__setattr__(self, "broker_metadata", _freeze_fields(self.broker_metadata))


@dataclass(frozen=True, slots=True)
class ManagedConnectionAuthService:
    local_identity: ManagedIdentityPort
    discord_auth: DiscordAuthPort
    broker_client: BrokerClientPort
    secret_store: SecretStorePort
    settings_repository: SettingsRepositoryPort
    claim_guard: ManagedAuthClaimGuard | None = None
    delivery_ack_service: ManagedKeyDeliveryAckService | None = None
    managed_state: ManagedIdentityStatePort | None = None

    async def authorize(self, request: ManagedConnectionAuthRequest) -> TransactionResult:
        if _caller_settings_values_are_unsafe(request.settings_values):
            return _unsafe_settings_values_result(request)

        recovery_result = await self._recover_pending_delivery_ack(request)
        if recovery_result is not None:
            return recovery_result

        claim_result = await self._preflight_claim_source()
        if claim_result is not None:
            return claim_result

        identity_result = await self._preflight_local_identity(request)
        if isinstance(identity_result, TransactionResult):
            return identity_result

        emit_progress(request.progress_sink, "preparing")
        resumed_result = await self._recover_pending_operation(
            request,
            identity_result=identity_result,
        )
        if resumed_result is not None:
            return resumed_result
        discord_result = await self._start_discord_auth(request)
        if isinstance(discord_result, TransactionResult):
            return discord_result

        operation_assurance = await self._ensure_pending_operation(
            request,
            identity_result=identity_result,
        )
        if isinstance(operation_assurance, TransactionResult):
            return operation_assurance
        operation, resume_token = operation_assurance
        settings_values = pending_operation_settings_values(request.settings_values, operation)
        broker_result = await self._issue_managed_connection(
            request=request,
            identity_result=identity_result,
            discord_result=discord_result,
            operation_id=operation.operation_id,
            resume_token=resume_token,
        )
        if isinstance(broker_result, TransactionResult):
            return await self._reconcile_issue_result(
                request,
                identity_result=identity_result,
                discord_result=discord_result,
                issue_result=broker_result,
                operation=operation,
                resume_token=resume_token,
                settings_values=settings_values,
            )
        return await self._finalize_issued_credential(
            request=request,
            identity_result=identity_result,
            broker_result=broker_result,
            settings_values=settings_values,
        )


    async def _finalize_issued_credential(
        self,
        *,
        request: ManagedConnectionAuthRequest,
        identity_result: ManagedIdentityPreflightResult,
        broker_result: BrokerIssueResult,
        settings_values: Mapping[str, object],
    ) -> TransactionResult:
        if not broker_result.managed_secret_key:
            return _remote_active_local_missing_result(
                request=request,
                broker_result=broker_result,
                operation="set_managed_secret",
                code="remote_active_managed_secret_missing",
                phase="local_secret_write",
                secret_write_succeeded=False,
                settings_commit_succeeded=False,
                diagnostics_present=broker_result.diagnostics is not None,
                message=broker_result.message,
            )

        if _settings_values_contain_raw_secret(
            request.settings_values,
            secret_value=broker_result.managed_secret_key,
        ):
            return _remote_active_unsafe_settings_values_result(
                request=request,
                broker_result=broker_result,
                message=broker_result.message,
            )

        settings_values = _settings_values_with_broker_issue(
            settings_values,
            broker_result,
        )
        commit_result: TransactionResult | None = None
        if broker_result.delivery_ack is not None:
            try:
                settings_values = await store_pending_ack_in_settings_values(
                    settings_values=settings_values,
                    secret_store=self.secret_store,
                    metadata=broker_result.delivery_ack,
                )
            except ManagedKeyDeliveryAckTokenStoreError:
                return _remote_active_local_missing_result(
                    request=request,
                    broker_result=broker_result,
                    operation="store_delivery_ack_token",
                    code="delivery_ack_token_store_failed_before_local_key_write",
                    phase="delivery_ack_token_store",
                    secret_write_succeeded=False,
                    settings_commit_succeeded=False,
                    diagnostics_present=broker_result.diagnostics is not None,
                    message=broker_result.message,
                )
            commit_result = await self._commit_settings(
                request=request,
                broker_result=broker_result,
                settings_values=settings_values,
                secret_message=None,
                secret_diagnostics_present=False,
            )
            if commit_result.status != TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED:
                return commit_result

        secret_write_result = await self._write_local_managed_secret(
            request=request,
            broker_result=broker_result,
        )
        if isinstance(secret_write_result, TransactionResult):
            if broker_result.delivery_ack is not None:
                with contextlib.suppress(Exception):
                    await self.secret_store.clear_secret(
                        secret_key_for_ack_source(broker_result.delivery_ack.source)
                    )
            return secret_write_result

        await self._store_managed_user_identifier(
            identity_result=identity_result,
            broker_result=broker_result,
        )

        if broker_result.delivery_ack is None:
            commit_result = await self._commit_settings(
                request=request,
                broker_result=broker_result,
                settings_values=_settings_values_without_pending_operation(settings_values),
                secret_message=secret_write_result.message,
                secret_diagnostics_present=secret_write_result.diagnostics is not None,
            )
            if commit_result.status != TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED:
                return commit_result
        assert commit_result is not None

        if broker_result.delivery_ack is not None:
            ack_result = await self.broker_client.acknowledge_managed_key_delivery(
                ManagedKeyDeliveryAckRequest(
                    delivery_id=broker_result.delivery_ack.delivery_id,
                    managed_credential_ref=broker_result.delivery_ack.managed_credential_ref,
                    delivery_ack_token=broker_result.delivery_ack.delivery_ack_token,
                )
            )
            self._emit_ack_result(request, ack_result)
            if not ack_result.succeeded:
                return _delivery_ack_pending_result(
                    request=request,
                    broker_result=broker_result,
                    ack_status=ack_result.status,
                    diagnostics_present=ack_result.diagnostics is not None,
                    message=ack_result.message or commit_result.message,
                )
            try:
                clear_result = await self.secret_store.clear_secret(
                    secret_key_for_ack_source(broker_result.delivery_ack.source)
                )
            except Exception:
                return _delivery_ack_pending_result(
                    request=request,
                    broker_result=broker_result,
                    ack_status="token_clear_failed",
                    diagnostics_present=False,
                    message=commit_result.message,
                )
            if not clear_result.succeeded:
                return _delivery_ack_pending_result(
                    request=request,
                    broker_result=broker_result,
                    ack_status="token_clear_failed",
                    diagnostics_present=clear_result.diagnostics is not None,
                    message=clear_result.message or commit_result.message,
                )
            self._apply_ack_referral(ack_result, broker_result.delivery_ack.source)
            settings_values = _settings_values_with_ack_referral(
                settings_values, ack_result, broker_result.delivery_ack.source
            )
            cleared_values = _settings_values_without_pending_operation(
                clear_pending_ack_in_settings_values(settings_values)
            )
            clear_commit = await self._commit_settings(
                request=request,
                broker_result=broker_result,
                settings_values=cleared_values,
                secret_message=commit_result.message,
                secret_diagnostics_present=commit_result.diagnostics is not None,
            )
            if clear_commit.status != TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED:
                try:
                    restore_result = await self.secret_store.set_secret(
                        secret_key_for_ack_source(broker_result.delivery_ack.source),
                        broker_result.delivery_ack.delivery_ack_token,
                    )
                    token_restored = restore_result.succeeded
                except Exception:
                    token_restored = False
                return _delivery_ack_pending_result(
                    request=request,
                    broker_result=broker_result,
                    ack_status=(
                        "metadata_clear_failed" if token_restored else "token_restore_failed"
                    ),
                    diagnostics_present=clear_commit.diagnostics is not None,
                    message=clear_commit.message or commit_result.message,
                )
            commit_result = clear_commit

        claim_persist_result = self._record_successful_claim_after_commit(
            request=request,
            broker_result=broker_result,
            message=commit_result.message,
        )
        if claim_persist_result is not None:
            return claim_persist_result
        await self._clear_terminal_operation_safe()
        return commit_result

    def _managed_state(self) -> ManagedIdentityStatePort | None:
        if self.managed_state is not None:
            return self.managed_state
        if self.delivery_ack_service is not None:
            return self.delivery_ack_service.managed_state
        if self.claim_guard is not None:
            return self.claim_guard.managed_state
        return None

    def _emit_ack_result(
        self,
        request: ManagedConnectionAuthRequest,
        ack_result: ManagedKeyDeliveryAckResult,
    ) -> None:
        sink = request.ack_result_sink
        if sink is None:
            return
        append = getattr(sink, "append", None)
        if callable(append):
            append(ack_result)
            return
        if callable(sink):
            sink(ack_result)

    def _apply_ack_referral(
        self,
        ack_result: ManagedKeyDeliveryAckResult,
        source: str,
    ) -> None:
        state = self._managed_state()
        if state is None:
            return
        apply_ack_referral_to_managed_state(state, ack_result, source)

    def _max_polls(self, request: ManagedConnectionAuthRequest) -> int:
        if request.max_status_polls is not None and request.max_status_polls >= 0:
            return request.max_status_polls
        return DEFAULT_MAX_STATUS_POLLS

    def _poll_interval_ms(self, request: ManagedConnectionAuthRequest) -> int:
        if request.status_poll_interval_ms is not None and request.status_poll_interval_ms >= 0:
            return request.status_poll_interval_ms
        return DEFAULT_STATUS_POLL_INTERVAL_MS

    async def _local_secret_present(self, request: ManagedConnectionAuthRequest) -> bool:
        try:
            stored = await self.secret_store.get_secret(request.local_secret_key)
        except Exception:
            return False
        return stored is not None and bool(stored.value)

    async def _clear_terminal_operation(self, state: ManagedIdentityStatePort) -> None:
        try:
            clear_pending_operation(state)
            state.persist()
        except Exception:
            pass
        try:
            await clear_resume_token(self.secret_store)
        except Exception:
            pass

    async def _ensure_pending_operation(
        self,
        request: ManagedConnectionAuthRequest,
        *,
        identity_result: ManagedIdentityPreflightResult,
    ) -> tuple[ManagedOperationIdentity, str] | TransactionResult:
        state = self._managed_state()
        if state is None:
            return _pre_issue_failed_result(
                request=request,
                operation="ensure_managed_operation",
                code="managed_operation_state_unavailable",
                phase="persist_managed_operation",
                message=None,
                diagnostics_present=False,
            )
        installation_id = identity_result.local_identity_revision or state.installation_id
        installation_id = installation_id.strip() if isinstance(installation_id, str) else ""
        if not installation_id:
            return _pre_issue_failed_result(
                request=request,
                operation="ensure_managed_operation",
                code="managed_operation_installation_unavailable",
                phase="persist_managed_operation",
                message=None,
                diagnostics_present=False,
            )
        pending = read_pending_operation(state)
        existing_token = await read_resume_token(self.secret_store)
        if (
            pending is not None
            and pending.installation_id == installation_id
            and existing_token is not None
        ):
            return (pending, existing_token)
        if pending is not None or existing_token is not None:
            await self._clear_terminal_operation(state)
        resume_token = new_managed_operation_resume_token()
        try:
            await store_resume_token(self.secret_store, resume_token)
        except ManagedOperationTokenStoreError:
            return _pre_issue_failed_result(
                request=request,
                operation="ensure_managed_operation",
                code="managed_operation_token_store_failed_before_issue",
                phase="persist_managed_operation",
                message=None,
                diagnostics_present=False,
            )
        operation = ManagedOperationIdentity(
            operation_id=new_managed_operation_id(),
            source=MANAGED_OPERATION_SOURCE_DISCORD,
            installation_id=installation_id,
            last_known_state=None,
        )
        write_pending_operation(state, operation)
        try:
            state.persist()
        except Exception:
            await self._clear_terminal_operation(state)
            return _pre_issue_failed_result(
                request=request,
                operation="ensure_managed_operation",
                code="managed_operation_persist_failed_before_issue",
                phase="persist_managed_operation",
                message=None,
                diagnostics_present=False,
            )
        return (operation, resume_token)

    async def _recover_pending_operation(
        self,
        request: ManagedConnectionAuthRequest,
        *,
        identity_result: ManagedIdentityPreflightResult,
    ) -> TransactionResult | None:
        state = self._managed_state()
        if state is None:
            return None
        pending = read_pending_operation(state)
        if pending is None:
            return None
        installation_id = identity_result.local_identity_revision or state.installation_id
        installation_id = installation_id.strip() if isinstance(installation_id, str) else ""
        if not installation_id or pending.installation_id != installation_id:
            await self._clear_terminal_operation(state)
            return None
        resume_token = await read_resume_token(self.secret_store)
        if resume_token is None:
            await self._clear_terminal_operation(state)
            return None
        emit_progress(request.progress_sink, "recovering")
        return await self._drive_operation_recovery(
            request,
            identity_result=identity_result,
            discord_result=None,
            operation=pending,
            resume_token=resume_token,
            settings_values=pending_operation_settings_values(
                request.settings_values, pending
            ),
            first_probe=None,
            unknown_outcome=True,
            original_failure=None,
        )

    async def _reconcile_issue_result(
        self,
        request: ManagedConnectionAuthRequest,
        *,
        identity_result: ManagedIdentityPreflightResult,
        discord_result: DiscordAuthResult,
        issue_result: TransactionResult,
        operation: ManagedOperationIdentity,
        resume_token: str,
        settings_values: Mapping[str, object],
    ) -> TransactionResult:
        unknown = _issue_failure_unknown_outcome(issue_result)
        try:
            first_probe = await self._fetch_operation_status(
                operation.operation_id,
                resume_token,
                operation.installation_id,
            )
        except Exception:
            first_probe = None
        if first_probe is None or (
            not first_probe.succeeded
            and first_probe.operation_status != MANAGED_OPERATION_UNKNOWN_OPERATION_STATUS
        ):
            if unknown:
                emit_progress(request.progress_sink, "recovering")
                return await self._drive_operation_recovery(
                    request,
                    identity_result=identity_result,
                    discord_result=discord_result,
                    operation=operation,
                    resume_token=resume_token,
                    settings_values=settings_values,
                    first_probe=None,
                    unknown_outcome=True,
                    original_failure=None,
                    initial_probe_failures=1,
                )
            return issue_result
        if (
            not unknown
            and not first_probe.succeeded
            and first_probe.operation_status == MANAGED_OPERATION_UNKNOWN_OPERATION_STATUS
        ):
            await self._clear_terminal_operation_safe()
            return issue_result
        emit_progress(request.progress_sink, "recovering")
        recovered = await self._drive_operation_recovery(
            request,
            identity_result=identity_result,
            discord_result=discord_result,
            operation=operation,
            resume_token=resume_token,
            settings_values=settings_values,
            first_probe=first_probe,
            unknown_outcome=unknown,
            original_failure=None if unknown else issue_result,
        )
        return recovered if recovered is not None else issue_result

    async def _fetch_operation_status(
        self,
        operation_id: str,
        resume_token: str,
        installation_id: str,
    ) -> ManagedOperationStatusResult | None:
        try:
            return await self.broker_client.get_managed_operation_status(
                ManagedOperationStatusRequest(
                    operation_id=operation_id,
                    installation_id=installation_id,
                    resume_token=resume_token,
                )
            )
        except Exception:
            return None

    async def _call_resume(
        self,
        operation: ManagedOperationIdentity,
        resume_token: str,
    ) -> ManagedOperationStatusResult | None:
        try:
            return await self.broker_client.resume_managed_operation(
                ManagedOperationResumeRequest(
                    operation_id=operation.operation_id,
                    installation_id=operation.installation_id,
                    resume_token=resume_token,
                )
            )
        except Exception:
            return None

    async def _drive_operation_recovery(
        self,
        request: ManagedConnectionAuthRequest,
        *,
        identity_result: ManagedIdentityPreflightResult,
        discord_result: DiscordAuthResult | None,
        operation: ManagedOperationIdentity,
        resume_token: str,
        settings_values: Mapping[str, object],
        first_probe: ManagedOperationStatusResult | None,
        unknown_outcome: bool,
        original_failure: TransactionResult | None,
        initial_probe_failures: int = 0,
    ) -> TransactionResult | None:
        max_polls = self._max_polls(request)
        base_interval = self._poll_interval_ms(request)
        polls = 0
        consecutive_probe_failures = initial_probe_failures
        pending_probe: ManagedOperationStatusResult | None | str = first_probe or "fetch"
        reissue_used = False
        resume_authorized_seen = False
        first_iteration = True
        while True:
            if pending_probe == "fetch":
                if polls >= max_polls:
                    return _operation_recovery_pending_result(
                        request, operation=operation, polls=polls
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
            if probe is None or not probe.succeeded:
                if (
                    probe is not None
                    and probe.operation_status == MANAGED_OPERATION_UNKNOWN_OPERATION_STATUS
                ):
                    if discord_result is not None and not reissue_used and (
                        unknown_outcome or resume_authorized_seen
                    ):
                        reissue_used = True
                        reissued = await self._issue_managed_connection(
                            request=request,
                            identity_result=identity_result,
                            discord_result=discord_result,
                            operation_id=operation.operation_id,
                            resume_token=resume_token,
                        )
                        if not isinstance(reissued, TransactionResult):
                            return await self._finalize_issued_credential(
                                request=request,
                                identity_result=identity_result,
                                broker_result=reissued,
                                settings_values=settings_values,
                            )
                        if not _issue_failure_unknown_outcome(reissued):
                            await self._clear_terminal_operation_safe()
                            return reissued
                        continue
                    if original_failure is not None and first_iteration:
                        await self._clear_terminal_operation_safe()
                        return original_failure
                    if discord_result is None:
                        await self._clear_terminal_operation_safe()
                        return None
                    continue
                consecutive_probe_failures += 1
                if (
                    consecutive_probe_failures
                    >= MANAGED_OPERATION_MAX_CONSECUTIVE_PROBE_FAILURES
                ):
                    return _operation_recovery_pending_result(
                        request, operation=operation, polls=polls
                    )
                if original_failure is not None and first_iteration and not unknown_outcome:
                    return original_failure
                first_iteration = False
                continue
            first_iteration = False
            consecutive_probe_failures = 0
            state = self._managed_state()
            if state is not None:
                update_pending_operation_state(state, probe.operation_status)
                try:
                    state.persist()
                except Exception:
                    return _operation_recovery_pending_result(
                        request, operation=operation, polls=polls
                    )
            if probe.operation_status == "FAILED":
                await self._clear_terminal_operation_safe()
                return _operation_action_required_result(request, operation=operation, probe=probe)
            if probe.operation_status == "ACTIVE":
                return await self._recover_active_operation(
                    request,
                    identity_result=identity_result,
                    operation=operation,
                    probe=probe,
                )
            converted = _status_result_to_issue_result(probe)
            if converted is not None:
                return await self._finalize_issued_credential(
                    request=request,
                    identity_result=identity_result,
                    broker_result=converted,
                    settings_values=settings_values,
                )
            if probe.client_action == "acknowledge_delivery":
                acknowledged = await self._recover_acknowledge_delivery(
                    request,
                    identity_result=identity_result,
                    operation=operation,
                    resume_token=resume_token,
                    settings_values=settings_values,
                    probe=probe,
                )
                if acknowledged is not None:
                    return acknowledged
                pending_probe = "fetch"
                continue
            if probe.client_action == "retry_authorized":
                resume_authorized_seen = True
                resumed = await self._call_resume(operation, resume_token)
                if resumed is not None and resumed.succeeded:
                    if resumed.operation_status == "FAILED":
                        await self._clear_terminal_operation_safe()
                        return _operation_action_required_result(
                            request, operation=operation, probe=resumed
                        )
                    resumed_converted = _status_result_to_issue_result(resumed)
                    if resumed_converted is not None:
                        return await self._finalize_issued_credential(
                            request=request,
                            identity_result=identity_result,
                            broker_result=resumed_converted,
                            settings_values=settings_values,
                        )
                    if (
                        resumed.operation_status == "RETRY_READY"
                        and discord_result is not None
                        and not reissue_used
                    ):
                        reissue_used = True
                        reissued = await self._issue_managed_connection(
                            request=request,
                            identity_result=identity_result,
                            discord_result=discord_result,
                            operation_id=operation.operation_id,
                            resume_token=resume_token,
                        )
                        if not isinstance(reissued, TransactionResult):
                            return await self._finalize_issued_credential(
                                request=request,
                                identity_result=identity_result,
                                broker_result=reissued,
                                settings_values=settings_values,
                            )
                        if not _issue_failure_unknown_outcome(reissued):
                            await self._clear_terminal_operation_safe()
                            return reissued
                pending_probe = "fetch"
                continue
            if probe.client_action == "action_required":
                await self._clear_terminal_operation_safe()
                return _operation_action_required_result(request, operation=operation, probe=probe)
            emit_progress(request.progress_sink, "recovering")
            pending_probe = "fetch"

    async def _clear_terminal_operation_safe(self) -> None:
        state = self._managed_state()
        if state is None:
            return
        await self._clear_terminal_operation(state)

    async def _recover_active_operation(
        self,
        request: ManagedConnectionAuthRequest,
        *,
        identity_result: ManagedIdentityPreflightResult,
        operation: ManagedOperationIdentity,
        probe: ManagedOperationStatusResult,
    ) -> TransactionResult:
        if not await self._local_secret_present(request):
            await self._clear_terminal_operation_safe()
            return _operation_action_required_result(
                request, operation=operation, probe=probe, code="managed_operation_active_key_missing"
            )
        self._apply_status_referral(probe)
        if self.claim_guard is not None:
            self.claim_guard.record_success(MANAGED_AUTH_CLAIM_SOURCE_DISCORD)
            try:
                self.claim_guard.managed_state.persist()
            except Exception:
                return _operation_recovery_pending_result(
                    request, operation=operation, polls=0
                )
        await self._clear_terminal_operation_safe()
        return TransactionResult(
            status=TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
            message=None,
            diagnostics=_metadata_diagnostics(
                operation="recover_managed_operation",
                code="managed_operation_active_recovered",
                fields={
                    "phase": "remote_managed_operation_recovery",
                    "local_secret_key": request.local_secret_key,
                    "managed_operation_id": operation.operation_id,
                    "operation_status": probe.operation_status,
                    "secret_write_succeeded": True,
                    "settings_commit_succeeded": True,
                },
            ),
        )

    def _apply_status_referral(self, probe: ManagedOperationStatusResult) -> None:
        state = self._managed_state()
        if state is None:
            return
        normalized_referral_id = _normalize_optional_text(probe.referral_id)
        if normalized_referral_id is not None:
            state.referral_id = normalized_referral_id
            state.referral_source = MANAGED_OPERATION_SOURCE_DISCORD

    async def _recover_acknowledge_delivery(
        self,
        request: ManagedConnectionAuthRequest,
        *,
        identity_result: ManagedIdentityPreflightResult,
        operation: ManagedOperationIdentity,
        resume_token: str,
        settings_values: Mapping[str, object],
        probe: ManagedOperationStatusResult,
    ) -> TransactionResult | None:
        service = self.delivery_ack_service
        if (
            service is not None
            and service.managed_state.pending_delivery_ack_source == ACK_SOURCE_DISCORD
            and await self._local_secret_present(request)
        ):
            recovered = await service.retry_pending()
            if recovered.ack_result is not None:
                self._emit_ack_result(request, recovered.ack_result)
            if recovered.succeeded:
                if self.claim_guard is not None:
                    self.claim_guard.record_success(MANAGED_AUTH_CLAIM_SOURCE_DISCORD)
                    try:
                        self.claim_guard.managed_state.persist()
                    except Exception:
                        return _operation_recovery_pending_result(
                            request, operation=operation, polls=0
                        )
                await self._clear_terminal_operation_safe()
                return TransactionResult(
                    status=TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
                    message=(
                        recovered.ack_result.message if recovered.ack_result is not None else None
                    ),
                    diagnostics=_metadata_diagnostics(
                        operation="recover_managed_operation",
                        code="managed_operation_delivery_ack_recovered",
                        fields={
                            "phase": "remote_managed_operation_recovery",
                            "local_secret_key": request.local_secret_key,
                            "managed_operation_id": operation.operation_id,
                            "operation_status": probe.operation_status,
                            "secret_write_succeeded": True,
                            "settings_commit_succeeded": True,
                        },
                    ),
                )
            return _operation_recovery_pending_result(request, operation=operation, polls=0)
        resumed = await self._call_resume(operation, resume_token)
        if resumed is not None and resumed.succeeded:
            converted = _status_result_to_issue_result(resumed)
            if converted is not None:
                return await self._finalize_issued_credential(
                    request=request,
                    identity_result=identity_result,
                    broker_result=converted,
                    settings_values=settings_values,
                )
        return None

    async def _recover_pending_delivery_ack(
        self,
        request: ManagedConnectionAuthRequest,
    ) -> TransactionResult | None:
        service = self.delivery_ack_service
        if (
            service is None
            or service.managed_state.pending_delivery_ack_source != ACK_SOURCE_DISCORD
        ):
            return None
        result = await service.recover_pending(
            source=ACK_SOURCE_DISCORD,
            managed_secret_key=request.local_secret_key,
        )
        if result.ack_result is not None:
            self._emit_ack_result(request, result.ack_result)
        message = result.ack_result.message if result.ack_result is not None else None
        if not result.succeeded:
            return _delivery_ack_recovery_pending_result(
                request=request,
                ack_status=result.status,
                diagnostics_present=result.diagnostics is not None,
                message=message,
            )
        if result.status == "none":
            return None
        if self.claim_guard is not None:
            self.claim_guard.record_success(MANAGED_AUTH_CLAIM_SOURCE_DISCORD)
            try:
                self.claim_guard.managed_state.persist()
            except Exception:
                return TransactionResult(
                    status=TRANSACTION_STATUS_REMOTE_ACTIVE_LOCAL_MISSING,
                    message=message,
                    diagnostics=_metadata_diagnostics(
                        operation="persist_recovered_managed_claim_source",
                        code="recovered_delivery_ack_claim_persist_failed",
                        fields={
                            "phase": "remote_delivery_ack_recovery",
                            "local_secret_key": request.local_secret_key,
                            "secret_write_succeeded": True,
                            "settings_commit_succeeded": False,
                        },
                    ),
                )
        return TransactionResult(
            status=TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
            message=message,
            diagnostics=_metadata_diagnostics(
                operation="recover_managed_key_delivery",
                code="delivery_ack_recovered",
                fields={
                    "phase": "remote_delivery_ack_recovery",
                    "local_secret_key": request.local_secret_key,
                    "delivery_ack_status": result.status,
                    "secret_write_succeeded": True,
                    "settings_commit_succeeded": True,
                },
            ),
        )

    async def _preflight_claim_source(self) -> TransactionResult | None:
        if self.claim_guard is None:
            return None
        return await self.claim_guard.preflight(MANAGED_AUTH_CLAIM_SOURCE_DISCORD)

    async def _preflight_local_identity(
        self,
        request: ManagedConnectionAuthRequest,
    ) -> ManagedIdentityPreflightResult | TransactionResult:
        try:
            identity_result = await self.local_identity.preflight_managed_identity(
                ManagedIdentityPreflightRequest(
                    local_secret_key=request.local_secret_key,
                    correlation_id=request.correlation_id,
                    metadata=request.broker_metadata,
                )
            )
        except Exception:
            return _pre_issue_failed_result(
                request=request,
                operation="preflight_managed_identity",
                code="local_identity_preflight_exception",
                phase="local_identity_preflight",
                message=None,
                diagnostics_present=False,
            )

        if not identity_result.succeeded or not identity_result.local_public_key:
            return _pre_issue_failed_result(
                request=request,
                operation="preflight_managed_identity",
                code="local_identity_preflight_failed",
                phase="local_identity_preflight",
                message=identity_result.message,
                diagnostics_present=identity_result.diagnostics is not None,
            )

        return identity_result

    async def _start_discord_auth(
        self,
        request: ManagedConnectionAuthRequest,
    ) -> DiscordAuthResult | TransactionResult:
        try:
            discord_result = await self.discord_auth.start_discord_auth(
                DiscordAuthRequest(
                    correlation_id=request.correlation_id,
                    metadata=request.broker_metadata,
                )
            )
        except Exception:
            return _pre_issue_failed_result(
                request=request,
                operation="start_discord_auth",
                code="discord_auth_exception",
                phase="discord_auth",
                message=None,
                diagnostics_present=False,
            )

        if not discord_result.succeeded or not _discord_auth_issue_material_present(discord_result):
            return _pre_issue_failed_result(
                request=request,
                operation="start_discord_auth",
                code="discord_auth_failed",
                phase="discord_auth",
                message=discord_result.message,
                diagnostics_present=discord_result.diagnostics is not None,
            )

        return discord_result

    async def _issue_managed_connection(
        self,
        *,
        request: ManagedConnectionAuthRequest,
        identity_result: ManagedIdentityPreflightResult,
        discord_result: DiscordAuthResult,
        operation_id: str | None = None,
        resume_token: str | None = None,
    ) -> BrokerIssueResult | TransactionResult:
        assert identity_result.local_public_key is not None
        assert _discord_auth_issue_material_present(discord_result)
        try:
            broker_result = await self.broker_client.issue_managed_connection(
                BrokerIssueRequest(
                    discord_user_id=discord_result.discord_user_id,
                    local_public_key=identity_result.local_public_key,
                    local_identity_revision=identity_result.local_identity_revision,
                    authorization_code=discord_result.authorization_code,
                    oauth_state=discord_result.oauth_state,
                    redirect_uri=discord_result.redirect_uri,
                    issue_nonce=discord_result.issue_nonce,
                    hardware_hash=discord_result.hardware_hash,
                    hardware_hash_salt_version=discord_result.hardware_hash_salt_version,
                    operation_id=operation_id,
                    resume_token=resume_token,
                    metadata=request.broker_metadata,
                )
            )
        except Exception:
            return _pre_issue_failed_result(
                request=request,
                operation="issue_managed_connection",
                code="broker_issue_exception",
                phase="broker_issue",
                message=None,
                diagnostics_present=False,
            )

        if not broker_result.succeeded:
            if broker_result.unknown_outcome:
                return _pre_issue_failed_result(
                    request=request,
                    operation="issue_managed_connection",
                    code="broker_issue_unknown_outcome",
                    phase="broker_issue",
                    message=broker_result.message,
                    diagnostics_present=broker_result.diagnostics is not None,
                )
            return _pre_issue_failed_result(
                request=request,
                operation="issue_managed_connection",
                code="broker_issue_failed",
                phase="broker_issue",
                message=broker_result.message,
                diagnostics_present=broker_result.diagnostics is not None,
            )

        return broker_result

    async def _write_local_managed_secret(
        self,
        *,
        request: ManagedConnectionAuthRequest,
        broker_result: BrokerIssueResult,
    ) -> SecretWriteResult | TransactionResult:
        assert broker_result.managed_secret_key is not None
        try:
            secret_write_result = await self.secret_store.set_secret(
                request.local_secret_key,
                broker_result.managed_secret_key,
            )
        except Exception:
            return _remote_active_local_missing_result(
                request=request,
                broker_result=broker_result,
                operation="set_managed_secret",
                code="remote_active_local_secret_write_failed",
                phase="local_secret_write",
                secret_write_succeeded=False,
                settings_commit_succeeded=False,
                diagnostics_present=False,
                message=None,
            )

        if not secret_write_result.succeeded:
            return _remote_active_local_missing_result(
                request=request,
                broker_result=broker_result,
                operation="set_managed_secret",
                code="remote_active_local_secret_write_failed",
                phase="local_secret_write",
                secret_write_succeeded=False,
                settings_commit_succeeded=False,
                diagnostics_present=secret_write_result.diagnostics is not None,
                message=secret_write_result.message,
            )

        return secret_write_result

    async def _commit_settings(
        self,
        *,
        request: ManagedConnectionAuthRequest,
        broker_result: BrokerIssueResult,
        settings_values: Mapping[str, object],
        secret_message: UserMessageRef | None,
        secret_diagnostics_present: bool,
    ) -> TransactionResult:
        try:
            settings_commit_result = await self.settings_repository.save(
                SettingsCommitRequest(
                    values=settings_values,
                    expected_revision=request.expected_settings_revision,
                    reason=request.reason,
                )
            )
        except Exception:
            return _remote_active_local_missing_result(
                request=request,
                broker_result=broker_result,
                operation="commit_settings",
                code="remote_active_local_settings_commit_failed",
                phase="local_settings_commit",
                secret_write_succeeded=True,
                settings_commit_succeeded=False,
                diagnostics_present=False,
                message=secret_message,
            )

        if not settings_commit_result.succeeded or settings_commit_result.snapshot is None:
            return _remote_active_local_missing_result(
                request=request,
                broker_result=broker_result,
                operation="commit_settings",
                code="remote_active_local_settings_commit_failed",
                phase="local_settings_commit",
                secret_write_succeeded=True,
                settings_commit_succeeded=False,
                diagnostics_present=settings_commit_result.diagnostics is not None,
                message=settings_commit_result.message or secret_message,
            )

        return TransactionResult(
            status=TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
            message=settings_commit_result.message or secret_message,
            diagnostics=(
                settings_commit_result.diagnostics
                if settings_commit_result.diagnostics is not None
                else (
                    _metadata_diagnostics(
                        operation="commit_settings",
                        code="settings_commit_succeeded_after_managed_issue",
                        fields={
                            "phase": "local_settings_commit",
                            "secret_write_succeeded": True,
                            "settings_commit_succeeded": True,
                            "secret_diagnostics_present": secret_diagnostics_present,
                        },
                    )
                    if secret_diagnostics_present
                    else None
                )
            ),
        )

    async def _store_managed_user_identifier(
        self,
        *,
        identity_result: ManagedIdentityPreflightResult,
        broker_result: BrokerIssueResult,
    ) -> None:
        user_id = _normalize_managed_user_identifier(broker_result.openrouter_user_id)
        installation_id = _normalize_optional_text(identity_result.local_identity_revision)
        if user_id is None or installation_id is None:
            return
        try:
            user_write = await self.secret_store.set_secret(
                OPENROUTER_MANAGED_USER_ID_SECRET,
                user_id,
            )
            installation_write = await self.secret_store.set_secret(
                OPENROUTER_MANAGED_USER_INSTALLATION_ID_SECRET,
                installation_id,
            )
        except Exception:
            await self._clear_managed_user_identifier()
            return
        if not user_write.succeeded or not installation_write.succeeded:
            await self._clear_managed_user_identifier()

    async def _clear_managed_user_identifier(self) -> None:
        for key in (
            OPENROUTER_MANAGED_USER_ID_SECRET,
            OPENROUTER_MANAGED_USER_INSTALLATION_ID_SECRET,
        ):
            try:
                await self.secret_store.clear_secret(key)
            except Exception:
                pass

    def _record_successful_claim_after_commit(
        self,
        *,
        request: ManagedConnectionAuthRequest,
        broker_result: BrokerIssueResult,
        message: UserMessageRef | None,
    ) -> TransactionResult | None:
        if self.claim_guard is None:
            return None
        self.claim_guard.record_success(MANAGED_AUTH_CLAIM_SOURCE_DISCORD)
        try:
            self.claim_guard.managed_state.persist()
        except Exception:
            return _remote_active_local_missing_result(
                request=request,
                broker_result=broker_result,
                operation="persist_managed_claim_source",
                code="remote_active_local_claim_persist_failed",
                phase="local_claim_persist",
                secret_write_succeeded=True,
                settings_commit_succeeded=True,
                diagnostics_present=False,
                message=message,
            )
        return None


def _normalized_settings_key(key: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in key)


def _discord_auth_issue_material_present(result: DiscordAuthResult) -> bool:
    if result.discord_user_id:
        return True
    return bool(
        result.authorization_code
        and result.oauth_state
        and result.redirect_uri
        and result.issue_nonce
        and result.hardware_hash
        and result.hardware_hash_salt_version is not None
    )


def _normalize_optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _normalize_managed_user_identifier(value: object) -> str | None:
    normalized = _normalize_optional_text(value)
    if normalized is None or len(normalized) > OPENROUTER_MANAGED_USER_ID_MAX_LENGTH:
        return None
    return normalized


def _settings_key_is_unsafe(key: object) -> bool:
    if not isinstance(key, str):
        return True
    normalized = _normalized_settings_key(key)
    compacted = normalized.replace("_", "")
    return any(
        fragment in normalized or fragment in compacted
        for fragment in _SETTINGS_SENSITIVE_KEY_FRAGMENTS
    )


def _caller_settings_values_are_unsafe(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            _settings_key_is_unsafe(key) or _caller_settings_values_are_unsafe(nested_value)
            for key, nested_value in value.items()
        )

    if isinstance(value, list | tuple):
        return any(_caller_settings_values_are_unsafe(item) for item in value)

    return False


def _settings_values_contain_raw_secret(
    value: object,
    *,
    secret_value: str,
) -> bool:
    if isinstance(value, Mapping):
        return any(
            (isinstance(key, str) and bool(secret_value and secret_value in key))
            or _settings_values_contain_raw_secret(nested_value, secret_value=secret_value)
            for key, nested_value in value.items()
        )

    if isinstance(value, list | tuple):
        return any(
            _settings_values_contain_raw_secret(item, secret_value=secret_value) for item in value
        )

    return isinstance(value, str) and bool(secret_value and secret_value in value)


def _pre_issue_failed_result(
    *,
    request: ManagedConnectionAuthRequest,
    operation: str,
    code: str,
    phase: str,
    message: UserMessageRef | None,
    diagnostics_present: bool,
) -> TransactionResult:
    return TransactionResult(
        status=TRANSACTION_STATUS_PROVIDER_VERIFICATION_FAILED,
        message=message,
        diagnostics=_metadata_diagnostics(
            operation=operation,
            code=code,
            fields={
                "phase": phase,
                "local_secret_key": request.local_secret_key,
                "remote_active": False,
                "diagnostics_present": diagnostics_present,
            },
        ),
    )


def _unsafe_settings_values_result(
    request: ManagedConnectionAuthRequest,
) -> TransactionResult:
    return TransactionResult(
        status=TRANSACTION_STATUS_PROVIDER_VERIFICATION_FAILED,
        message=None,
        diagnostics=_metadata_diagnostics(
            operation="validate_settings_values",
            code="unsafe_settings_values",
            fields={
                "phase": "validate_settings_values",
                "local_secret_key": request.local_secret_key,
                "remote_active": False,
                "settings_values_accepted": False,
            },
        ),
    )


def _remote_active_unsafe_settings_values_result(
    *,
    request: ManagedConnectionAuthRequest,
    broker_result: BrokerIssueResult,
    message: UserMessageRef | None,
) -> TransactionResult:
    fields: dict[str, DiagnosticFieldValue] = {
        "phase": "validate_settings_values_after_broker",
        "local_secret_key": request.local_secret_key,
        "remote_active": True,
        "broker_issue_succeeded": True,
        "settings_values_accepted": False,
        "secret_write_succeeded": False,
        "settings_commit_succeeded": False,
    }
    if broker_result.broker_connection_id is not None:
        fields["broker_connection_id"] = broker_result.broker_connection_id
    if broker_result.remote_key_revision is not None:
        fields["remote_key_revision"] = broker_result.remote_key_revision
    return TransactionResult(
        status=TRANSACTION_STATUS_REMOTE_ACTIVE_LOCAL_MISSING,
        message=message,
        diagnostics=_metadata_diagnostics(
            operation="validate_settings_values",
            code="remote_active_unsafe_settings_values",
            fields=fields,
        ),
    )


def _remote_active_local_missing_result(
    *,
    request: ManagedConnectionAuthRequest,
    broker_result: BrokerIssueResult,
    operation: str,
    code: str,
    phase: str,
    secret_write_succeeded: bool,
    settings_commit_succeeded: bool,
    diagnostics_present: bool,
    message: UserMessageRef | None,
) -> TransactionResult:
    fields: dict[str, DiagnosticFieldValue] = {
        "phase": phase,
        "local_secret_key": request.local_secret_key,
        "remote_active": True,
        "broker_issue_succeeded": True,
        "secret_write_succeeded": secret_write_succeeded,
        "settings_commit_succeeded": settings_commit_succeeded,
        "diagnostics_present": diagnostics_present,
    }
    if broker_result.broker_connection_id is not None:
        fields["broker_connection_id"] = broker_result.broker_connection_id
    if broker_result.remote_key_revision is not None:
        fields["remote_key_revision"] = broker_result.remote_key_revision
    return TransactionResult(
        status=TRANSACTION_STATUS_REMOTE_ACTIVE_LOCAL_MISSING,
        message=message,
        diagnostics=_metadata_diagnostics(
            operation=operation,
            code=code,
            fields=fields,
        ),
    )


def _delivery_ack_pending_result(
    *,
    request: ManagedConnectionAuthRequest,
    broker_result: BrokerIssueResult,
    ack_status: str,
    diagnostics_present: bool,
    message: UserMessageRef | None,
) -> TransactionResult:
    fields: dict[str, DiagnosticFieldValue] = {
        "phase": "remote_delivery_ack",
        "local_secret_key": request.local_secret_key,
        "remote_active": False,
        "broker_issue_succeeded": True,
        "secret_write_succeeded": True,
        "settings_commit_succeeded": True,
        "delivery_ack_status": ack_status,
        "diagnostics_present": diagnostics_present,
    }
    if broker_result.managed_credential_ref is not None:
        fields["managed_credential_ref"] = broker_result.managed_credential_ref
    return TransactionResult(
        status=TRANSACTION_STATUS_REMOTE_DELIVERY_ACK_PENDING,
        message=message,
        diagnostics=_metadata_diagnostics(
            operation="acknowledge_managed_key_delivery",
            code="remote_delivery_ack_pending",
            fields=fields,
        ),
    )


def _delivery_ack_recovery_pending_result(
    *,
    request: ManagedConnectionAuthRequest,
    ack_status: str,
    diagnostics_present: bool,
    message: UserMessageRef | None,
) -> TransactionResult:
    return TransactionResult(
        status=TRANSACTION_STATUS_REMOTE_DELIVERY_ACK_PENDING,
        message=message,
        diagnostics=_metadata_diagnostics(
            operation="recover_managed_key_delivery",
            code="delivery_ack_recovery_pending",
            fields={
                "phase": "remote_delivery_ack_recovery",
                "local_secret_key": request.local_secret_key,
                "delivery_ack_status": ack_status,
                "diagnostics_present": diagnostics_present,
            },
        ),
    )


def pending_operation_settings_values(
    settings_values: Mapping[str, object],
    operation: ManagedOperationIdentity,
) -> dict[str, object]:
    values = _copy_settings_values(settings_values)
    state = values.setdefault("state", {})
    if not isinstance(state, dict):
        state = {}
        values["state"] = state
    managed = state.setdefault("managed_connection", {})
    if not isinstance(managed, dict):
        managed = {}
        state["managed_connection"] = managed
    managed["pending_managed_operation_id"] = operation.operation_id
    managed["pending_managed_operation_source"] = operation.source
    managed["pending_managed_operation_installation_id"] = operation.installation_id
    if operation.last_known_state is not None:
        managed["pending_managed_operation_state"] = operation.last_known_state
    return values


def _settings_values_without_pending_operation(
    settings_values: Mapping[str, object],
) -> dict[str, object]:
    values = _copy_settings_values(settings_values)
    state = values.get("state")
    if not isinstance(state, dict):
        return values
    managed = state.get("managed_connection")
    if not isinstance(managed, dict):
        return values
    for key in (
        "pending_managed_operation_id",
        "pending_managed_operation_source",
        "pending_managed_operation_installation_id",
        "pending_managed_operation_state",
    ):
        managed.pop(key, None)
    return values


def _settings_values_with_ack_referral(
    settings_values: Mapping[str, object],
    ack_result: ManagedKeyDeliveryAckResult,
    source: str,
) -> dict[str, object]:
    values = _copy_settings_values(settings_values)
    state = values.setdefault("state", {})
    if not isinstance(state, dict):
        state = {}
        values["state"] = state
    managed = state.setdefault("managed_connection", {})
    if not isinstance(managed, dict):
        managed = {}
        state["managed_connection"] = managed
    if source in {ACK_SOURCE_DISCORD}:
        managed["referral_source"] = source
        if _normalize_optional_text(ack_result.referral_id) is not None:
            managed["referral_id"] = _normalize_optional_text(ack_result.referral_id)
    return values


def _issue_failure_unknown_outcome(issue_result: TransactionResult) -> bool:
    diagnostics = issue_result.diagnostics
    if diagnostics is None:
        return False
    return diagnostics.code in {"broker_issue_exception", "broker_issue_unknown_outcome"}


def _operation_message(key: str) -> UserMessageRef:
    return UserMessageRef(key=key, params={}, severity=SEVERITY_ERROR)


def _operation_recovery_pending_result(
    request: ManagedConnectionAuthRequest,
    operation: ManagedOperationIdentity,
    polls: int,
) -> TransactionResult:
    return TransactionResult(
        status=TRANSACTION_STATUS_REMOTE_DELIVERY_ACK_PENDING,
        message=_operation_message("discord_auth.error.recovery_pending"),
        diagnostics=_metadata_diagnostics(
            operation="recover_managed_operation",
            code="managed_operation_recovery_pending",
            fields={
                "phase": "remote_managed_operation_recovery",
                "local_secret_key": request.local_secret_key,
                "managed_operation_id": operation.operation_id,
                "managed_operation_state": operation.last_known_state,
                "status_polls": polls,
                "secret_write_succeeded": True,
                "settings_commit_succeeded": True,
            },
        ),
    )


def _operation_action_required_result(
    request: ManagedConnectionAuthRequest,
    operation: ManagedOperationIdentity,
    probe: ManagedOperationStatusResult,
    code: str | None = None,
) -> TransactionResult:
    failed_reason = probe.failed_reason
    if probe.operation_status == "FAILED" and failed_reason == "authorization_expired":
        resolved_code = "managed_operation_authorization_expired"
        message_key = "discord_auth.error.authorization_expired"
    else:
        resolved_code = code or "managed_operation_action_required"
        message_key = "discord_auth.error.action_required"
    fields: dict[str, DiagnosticFieldValue] = {
        "phase": "remote_managed_operation_recovery",
        "local_secret_key": request.local_secret_key,
        "managed_operation_id": operation.operation_id,
        "operation_status": probe.operation_status,
        "client_action": probe.client_action,
        "secret_write_succeeded": True,
        "settings_commit_succeeded": True,
    }
    if failed_reason is not None:
        fields["failed_reason"] = failed_reason
    return TransactionResult(
        status=TRANSACTION_STATUS_REMOTE_ACTIVE_LOCAL_MISSING,
        message=_operation_message(message_key),
        diagnostics=_metadata_diagnostics(
            operation="recover_managed_operation",
            code=resolved_code,
            fields=fields,
        ),
    )


def _status_result_to_issue_result(
    probe: ManagedOperationStatusResult,
) -> BrokerIssueResult | None:
    if not probe.succeeded or not probe.managed_secret_key:
        return None
    return BrokerIssueResult(
        succeeded=True,
        broker_connection_id=probe.managed_credential_ref,
        managed_secret_key=probe.managed_secret_key,
        remote_key_revision=probe.managed_credential_ref,
        message=None,
        diagnostics=None,
        managed_credential_ref=probe.managed_credential_ref,
        expires_at=probe.expires_at,
        openrouter_user_id=probe.openrouter_user_id,
        referral_id=probe.referral_id,
        referral_bonus_applied=probe.referral_bonus_applied,
        pass_status=probe.pass_status,
        delivery_ack=probe.delivery_ack,
        unknown_outcome=False,
    )


def _settings_values_with_broker_issue(
    settings_values: Mapping[str, object],
    broker_result: BrokerIssueResult,
) -> dict[str, object]:
    values = _copy_settings_values(settings_values)
    state = values.setdefault("state", {})
    if not isinstance(state, dict):
        state = {}
        values["state"] = state
    managed = state.setdefault("managed_connection", {})
    if not isinstance(managed, dict):
        managed = {}
        state["managed_connection"] = managed
    managed_credential_ref = broker_result.managed_credential_ref
    if managed_credential_ref:
        managed["active_managed_credential_ref"] = managed_credential_ref
    if broker_result.expires_at:
        managed["active_managed_expires_at"] = broker_result.expires_at
    if broker_result.referral_id:
        managed["referral_id"] = broker_result.referral_id
    return values


def _copy_settings_values(values: Mapping[str, object]) -> dict[str, object]:
    return {key: _copy_settings_value(value) for key, value in values.items()}


def _copy_settings_value(value: object) -> object:
    if isinstance(value, Mapping):
        return _copy_settings_values(value)
    if isinstance(value, tuple | list):
        return [_copy_settings_value(item) for item in value]
    return value


def _metadata_diagnostics(
    *,
    operation: str,
    code: str,
    fields: Mapping[str, DiagnosticFieldValue],
) -> ErrorDiagnostics:
    return ErrorDiagnostics(
        component="managed_connection_auth",
        operation=operation,
        code=code,
        category=DIAGNOSTIC_CATEGORY_TRANSACTION,
        visibility=DIAGNOSTIC_VISIBILITY_BASIC,
        content_policy=CONTENT_POLICY_METADATA_ONLY,
        status_code=None,
        retry_after_ms=None,
        fields=fields,
    )


__all__ = [
    "ManagedConnectionAuthRequest",
    "ManagedConnectionAuthService",
]
