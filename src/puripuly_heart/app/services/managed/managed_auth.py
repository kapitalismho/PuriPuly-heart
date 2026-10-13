from __future__ import annotations

import contextlib
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field, replace

from puripuly_heart.core.diagnostics import emit_failure, safe_exception_fields
from puripuly_heart.core.managed_openrouter_release import TalkTogetherPassStatus
from puripuly_heart.core.messages import (
    TRANSACTION_STATUS_PROVIDER_VERIFICATION_FAILED,
    TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED,
    ErrorDiagnostics,
    TransactionResult,
)
from puripuly_heart.core.qq_auth_diagnostics import log_qq_auth


def _managed_auth_failure(
    phase: str,
    exception: BaseException,
    code: str,
    *,
    cause_id: str | None = None,
    disposition: str = "failed",
) -> ErrorDiagnostics:
    diagnostics = getattr(exception, "diagnostics", None)
    if not isinstance(diagnostics, ErrorDiagnostics):
        diagnostics = ErrorDiagnostics(
            component="managed_auth",
            operation=phase,
            code=code,
            category="lifecycle",
            visibility="basic",
            content_policy="metadata_only",
            status_code=None,
            retry_after_ms=None,
            fields={"phase": phase, **safe_exception_fields(exception)},
            cause_id=cause_id,
        )
    emit_failure(
        logging.getLogger(__name__),
        diagnostics,
        severity="warning" if disposition == "degraded" else "error",
        disposition=disposition,
    )
    return diagnostics


QQ_AUTH_DIALOG_MESSAGE_KEY_BY_SERVICE_KEY = {
    **{
        f"qq_managed_auth.{name}": f"qq_auth.error.{name}"
        for name in (
            "configuration_unavailable",
            "local_storage_unavailable",
            "delivery_ack_pending",
            "managed_secret_missing",
            "network_timeout",
            "network_proxy",
            "network_tls",
            "network_dns",
            "network_network",
        )
    },
    "qq_managed_auth.already_claimed_discord": "qq_auth.error.already_claimed_discord",
    "qq_managed_auth.invalid_credential": "qq_auth.error.credential_mismatch",
    "qq_managed_auth.mismatch": "qq_auth.error.credential_mismatch",
    "qq_managed_auth.lifetime_used": "qq_auth.error.lifetime_used",
    "qq_managed_auth.rate_limited": "qq_auth.error.rate_limited",
    "qq_managed_auth.key_unavailable": "qq_auth.error.key_unavailable",
    "qq_managed_auth.broker_unavailable": "qq_auth.error.broker_unavailable",
    "qq_managed_auth.settings_commit_failed": "qq_auth.error.settings_commit_failed",
    "qq_managed_auth.secret_write_failed": "qq_auth.error.secret_write_failed",
    "qq_managed_auth.error.retry": "qq_auth.error.retry",
    "qq_managed_auth.error.recovery_pending": "qq_auth.error.recovery_pending",
    "qq_managed_auth.error.action_required": "qq_auth.error.action_required",
    "qq_managed_auth.error.authorization_expired": "qq_auth.error.authorization_expired",
    "qq_managed_auth.error.other_source_pending": "qq_auth.error.other_source_pending",
}


@dataclass(frozen=True, slots=True)
class ManagedAuthState:
    settings_available: bool
    managed_selected: bool
    managed_china: bool
    local_key_available: bool
    release_service_available: bool
    runtime_available: bool
    ingress_frozen: bool


@dataclass(frozen=True, slots=True)
class ManagedAuthExecutionResult:
    succeeded: bool
    transaction_result: TransactionResult | None = None
    delivery_ack_pending: bool = False
    referral_bonus_applied: bool = False
    referral_id: str | None = None
    pass_status: TalkTogetherPassStatus | None = None
    message_key: str = "discord_auth.error.retry"
    message_kwargs: Mapping[str, object] = field(default_factory=dict)
    error_class: str | None = None
    runtime_rebuild: str = "never"
    log_message: str | None = None
    log_failure: bool = True


ManagedAuthStateProvider = Callable[[], ManagedAuthState]
ManagedAuthPendingSink = Callable[[bool], None]
ManagedAuthQqExecutor = Callable[[str, str, str | None], Awaitable[ManagedAuthExecutionResult]]
ManagedAuthDiscordExecutor = Callable[
    [str | None, Callable[[], None] | None, Callable[[], None] | None],
    Awaitable[ManagedAuthExecutionResult],
]
ManagedAuthRuntimeEnsurer = Callable[[str], Awaitable[bool]]
ManagedAuthUsageViewSink = Callable[
    [str | None, TalkTogetherPassStatus | None],
    None,
]
ManagedAuthUsageRefreshSink = Callable[[], None]
ManagedAuthMessageSink = Callable[[str, Mapping[str, object]], None]
ManagedAuthResultSink = Callable[[TransactionResult], None]
ManagedAuthLogSink = Callable[[str], None]


def _failure_kind_for_message_key(message_key: str) -> str:
    if message_key in {
        "discord_auth.error.action_required",
        "discord_auth.error.authorization_expired",
    }:
        return "action_required"
    if message_key == "discord_auth.error.recovery_pending":
        return "recovering"
    return "failed"


@dataclass(slots=True)
class ManagedAuthOwner:
    state_provider: ManagedAuthStateProvider
    pending_sink: ManagedAuthPendingSink
    qq_executor: ManagedAuthQqExecutor
    discord_executor: ManagedAuthDiscordExecutor
    runtime_ensurer: ManagedAuthRuntimeEnsurer
    usage_view_sink: ManagedAuthUsageViewSink
    usage_refresh_sink: ManagedAuthUsageRefreshSink
    message_sink: ManagedAuthMessageSink
    result_sink: ManagedAuthResultSink
    log_sink: ManagedAuthLogSink
    pending: bool = False
    discord_in_progress: bool = False
    last_referral_bonus_applied: bool = False
    last_failure_kind: str = "failed"
    last_message_key: str | None = None
    callback_received_hook: Callable[[], None] | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _ingress_stopped: bool = field(init=False, default=False, repr=False)

    def set_pending(self, pending: bool) -> None:
        self.pending = bool(pending)
        try:
            self.pending_sink(self.pending)
        except Exception as exc:
            _managed_auth_failure(
                "auth_notification", exc, "managed_auth_notification_failed", disposition="degraded"
            )

    def clear_pending(self) -> None:
        self.set_pending(False)

    def dashboard_action(self) -> str:
        state = self.state_provider()
        if not state.managed_selected:
            return "continue"
        if self.discord_in_progress or self.pending:
            return "in_progress"
        if state.local_key_available:
            return "continue"
        return "prompt"

    def dashboard_prompt_kind(self) -> str:
        return "qq" if self.state_provider().managed_china else "discord"

    async def start_qq(
        self,
        *,
        qq_identity: str,
        credential: str,
        referral_id: str | None = None,
    ) -> bool | tuple[str, dict[str, object]]:
        self.last_referral_bonus_applied = False
        state = self.state_provider()
        if (
            self._ingress_stopped
            or state.ingress_frozen
            or not state.settings_available
            or not state.managed_china
            or not state.release_service_available
        ):
            code = (
                "configuration_unavailable"
                if not state.settings_available or not state.release_service_available
                else "flow_not_ready"
            )
            log_qq_auth("preflight", code)
            return "qq_auth.error.configuration_unavailable", {}
        try:
            result = await self.qq_executor(qq_identity, credential, referral_id)
        except Exception as exc:
            diagnostic = _managed_auth_failure(
                "managed_authentication", exc, "managed_authentication_failed"
            )
            self._report_result(
                TransactionResult(TRANSACTION_STATUS_PROVIDER_VERIFICATION_FAILED, None, diagnostic)
            )
            return "qq_auth.error.retry", {}
        if result.transaction_result is not None:
            self._report_result(result.transaction_result)
        if result.succeeded:
            self.clear_pending()
            self.last_referral_bonus_applied = result.referral_bonus_applied
            if result.runtime_rebuild != "never":
                await self._ensure_auth_runtime(result, propagate=False)
            self._notify_usage(result.referral_id, result.pass_status)
            return True
        if result.delivery_ack_pending and result.referral_id is not None:
            self._notify_usage(result.referral_id, result.pass_status)
        message_key = QQ_AUTH_DIALOG_MESSAGE_KEY_BY_SERVICE_KEY.get(
            result.message_key,
            result.message_key,
        )
        return message_key, dict(result.message_kwargs)

    def _report_result(self, result: TransactionResult) -> None:
        try:
            self.result_sink(result)
        except Exception as exc:
            _managed_auth_failure(
                "auth_notification",
                exc,
                "managed_auth_notification_failed",
                cause_id=(
                    result.diagnostics.diagnostic_id if result.diagnostics is not None else None
                ),
                disposition="degraded",
            )

    async def _ensure_auth_runtime(
        self, result: ManagedAuthExecutionResult, *, propagate: bool
    ) -> bool:
        try:
            applied = await self.runtime_ensurer(result.runtime_rebuild)
        except Exception as exc:
            diagnostic = _managed_auth_failure(
                "runtime_apply", exc, "provider_runtime_apply_exception", disposition="degraded"
            )
            previous = result.transaction_result
            if previous is not None:
                self._report_result(
                    replace(
                        previous,
                        status=TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED,
                        diagnostics=diagnostic,
                    )
                )
            if propagate:
                raise
            return False
        if not applied and result.transaction_result is not None:
            diagnostic = ErrorDiagnostics(
                component="managed_auth",
                operation="runtime_apply",
                code="provider_runtime_apply_unavailable",
                category="lifecycle",
                visibility="basic",
                content_policy="metadata_only",
                status_code=None,
                retry_after_ms=None,
                fields={"phase": "runtime_apply", "observation_succeeded": False},
            )
            self._report_result(
                replace(
                    result.transaction_result,
                    status=TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED,
                    diagnostics=diagnostic,
                )
            )
        return applied

    def _notify_usage(
        self, referral_id: str | None, pass_status: TalkTogetherPassStatus | None
    ) -> None:
        try:
            self.usage_view_sink(referral_id, pass_status)
        except Exception as exc:
            _managed_auth_failure(
                "auth_notification", exc, "managed_auth_notification_failed", disposition="degraded"
            )
        try:
            self.usage_refresh_sink()
        except Exception as exc:
            _managed_auth_failure(
                "auth_notification", exc, "managed_auth_notification_failed", disposition="degraded"
            )

    async def start_discord(
        self,
        *,
        on_callback_received: Callable[[], None] | None = None,
        on_recovery_started: Callable[[], None] | None = None,
        referral_id: str | None = None,
        authorization_url_sink: Callable[[str], None] | None = None,
        open_browser: bool = True,
    ) -> bool:
        self.last_referral_bonus_applied = False
        self.last_failure_kind = "failed"
        self.last_message_key = None
        state = self.state_provider()
        if (
            self._ingress_stopped
            or state.ingress_frozen
            or not state.settings_available
            or not state.release_service_available
        ):
            self.discord_in_progress = False
            self.clear_pending()
            self.message_sink("discord_auth.error.retry", {})
            return False
        previous_callback = self.callback_received_hook
        self.callback_received_hook = on_callback_received
        self.discord_in_progress = True
        self.set_pending(True)
        try:
            if authorization_url_sink is None and open_browser:
                result = await self.discord_executor(
                    referral_id, on_callback_received, on_recovery_started
                )
            else:
                result = await self.discord_executor(
                    referral_id,
                    on_callback_received,
                    on_recovery_started,
                    authorization_url_sink=authorization_url_sink,
                    open_browser=open_browser,
                )
            if result.transaction_result is not None:
                self._report_result(result.transaction_result)
            if result.succeeded:
                self.last_referral_bonus_applied = result.referral_bonus_applied
                if not await self._ensure_auth_runtime(result, propagate=True):
                    self.message_sink("discord_auth.error.retry", {})
                    return False
                self._notify_usage(result.referral_id, result.pass_status)
                return True
            if result.log_message is not None:
                self.log_sink(result.log_message)
            if result.log_failure:
                self.log_sink(
                    "[ManagedAuth] Discord auth failed: "
                    f"message_key={result.message_key} "
                    f"class={result.error_class or 'unknown'}"
                )
            if result.delivery_ack_pending and result.referral_id is not None:
                self._notify_usage(result.referral_id, result.pass_status)
            self.last_message_key = result.message_key
            self.last_failure_kind = _failure_kind_for_message_key(result.message_key)
            self.message_sink(result.message_key, result.message_kwargs)
            return False
        finally:
            if self.callback_received_hook is on_callback_received:
                self.callback_received_hook = previous_callback
            self.discord_in_progress = False
            self.clear_pending()

    def on_callback_received(self) -> None:
        hook = self.callback_received_hook
        if callable(hook):
            try:
                hook()
            except Exception as exc:
                _managed_auth_failure(
                    "auth_notification",
                    exc,
                    "managed_auth_notification_failed",
                    disposition="degraded",
                )

    def stop_ingress(self) -> None:
        self._ingress_stopped = True
        self.discord_in_progress = False
        self.callback_received_hook = None
        with contextlib.suppress(Exception):
            self.clear_pending()

    async def close(self) -> None:
        self.stop_ingress()


__all__ = [
    "ManagedAuthExecutionResult",
    "ManagedAuthOwner",
    "ManagedAuthState",
]
