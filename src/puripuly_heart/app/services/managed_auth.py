from __future__ import annotations

import contextlib
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field

from puripuly_heart.core.managed_openrouter_release import TalkTogetherPassStatus
from puripuly_heart.core.messages import TransactionResult

QQ_AUTH_DIALOG_MESSAGE_KEY_BY_SERVICE_KEY = {
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
ManagedAuthQqExecutor = Callable[[str, str], Awaitable[ManagedAuthExecutionResult]]
ManagedAuthDiscordExecutor = Callable[
    [str | None, Callable[[], None] | None],
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
    callback_received_hook: Callable[[], None] | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _ingress_stopped: bool = field(init=False, default=False, repr=False)

    def set_pending(self, pending: bool) -> None:
        self.pending = bool(pending)
        self.pending_sink(self.pending)

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
    ) -> bool | tuple[str, dict[str, object]]:
        state = self.state_provider()
        if (
            self._ingress_stopped
            or state.ingress_frozen
            or not state.settings_available
            or not state.managed_china
            or not state.release_service_available
        ):
            return "qq_auth.error.retry", {}
        result = await self.qq_executor(qq_identity, credential)
        if result.succeeded:
            self.clear_pending()
            if result.runtime_rebuild != "never":
                await self.runtime_ensurer(result.runtime_rebuild)
            else:
                self.usage_refresh_sink()
            return True
        message_key = QQ_AUTH_DIALOG_MESSAGE_KEY_BY_SERVICE_KEY.get(
            result.message_key,
            result.message_key,
        )
        return message_key, dict(result.message_kwargs)

    async def start_discord(
        self,
        *,
        on_callback_received: Callable[[], None] | None = None,
        referral_id: str | None = None,
    ) -> bool:
        self.last_referral_bonus_applied = False
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
            result = await self.discord_executor(referral_id, on_callback_received)
            if result.transaction_result is not None:
                self.result_sink(result.transaction_result)
            if result.succeeded:
                self.last_referral_bonus_applied = result.referral_bonus_applied
                if not await self.runtime_ensurer(result.runtime_rebuild):
                    self.message_sink("discord_auth.error.retry", {})
                    return False
                self.usage_view_sink(result.referral_id, result.pass_status)
                self.usage_refresh_sink()
                return True
            if result.log_message is not None:
                self.log_sink(result.log_message)
            if result.log_failure:
                self.log_sink(
                    "[ManagedAuth] Discord auth failed: "
                    f"message_key={result.message_key} "
                    f"class={result.error_class or 'unknown'}"
                )
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
            hook()

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
