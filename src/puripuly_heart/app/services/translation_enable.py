from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field

from puripuly_heart.core.messages import TransactionResult


@dataclass(frozen=True, slots=True)
class TranslationEnableState:
    runtime_available: bool
    translation_enabled: bool
    llm_available: bool
    settings_available: bool
    provider_name: str | None
    qwen_region: str | None
    managed_selected: bool
    managed_china: bool
    managed_local_key_available: bool
    managed_release_service_available: bool
    ingress_frozen: bool


@dataclass(frozen=True, slots=True)
class ManagedTranslationPreparation:
    ready: bool
    transaction_result: TransactionResult | None = None
    message_key: str | None = None
    message_kwargs: Mapping[str, object] = field(default_factory=dict)
    diagnostics_text: str | None = None
    show_qq_dialog: bool = False


TranslationEnableStateProvider = Callable[[], TranslationEnableState]
ManagedTranslationPrepare = Callable[[], Awaitable[ManagedTranslationPreparation]]
TranslationEnableFounderRoute = Callable[[], Awaitable[bool]]
TranslationEnablePendingSink = Callable[[bool], None]
TranslationEnableRuntimeEnsurer = Callable[[str], Awaitable[bool]]
TranslationEnableUsageRefreshSink = Callable[[], None]
TranslationEnableUsageRefreshNow = Callable[[], Awaitable[None]]
TranslationEnableRuntimeSink = Callable[[bool], None]
TranslationEnableDashboardSink = Callable[[bool], None]
TranslationEnableClearContext = Callable[[], None]
TranslationEnableWarmup = Callable[[], Awaitable[None]]
TranslationEnableTeardown = Callable[[], Awaitable[None]]
TranslationEnableMessageSink = Callable[[str, Mapping[str, object]], None]
TranslationEnableQqDialogSink = Callable[[], None]
TranslationEnableResultSink = Callable[[TransactionResult], None]
TranslationEnableLogSink = Callable[[str], None]
TranslationEnableFounderLetterSink = Callable[[], None]
TranslationEnableStartingSink = Callable[[bool], None]


@dataclass(slots=True)
class TranslationEnableOwner:
    state_provider: TranslationEnableStateProvider
    managed_prepare: ManagedTranslationPrepare
    founder_route: TranslationEnableFounderRoute
    pending_sink: TranslationEnablePendingSink
    runtime_ensurer: TranslationEnableRuntimeEnsurer
    usage_refresh_sink: TranslationEnableUsageRefreshSink
    usage_refresh_now: TranslationEnableUsageRefreshNow
    runtime_sink: TranslationEnableRuntimeSink
    dashboard_sink: TranslationEnableDashboardSink
    clear_context: TranslationEnableClearContext
    warmup: TranslationEnableWarmup
    message_sink: TranslationEnableMessageSink
    qq_dialog_sink: TranslationEnableQqDialogSink
    result_sink: TranslationEnableResultSink
    log_basic: TranslationEnableLogSink
    log_detailed: TranslationEnableLogSink
    log_error: TranslationEnableLogSink
    founder_letter_sink: TranslationEnableFounderLetterSink
    teardown: TranslationEnableTeardown | None = None
    starting_sink: TranslationEnableStartingSink | None = None
    intent_enabled: bool = False
    generation: int = 0
    _ingress_stopped: bool = field(init=False, default=False, repr=False)

    def record_intent(self, enabled: bool) -> int:
        self.intent_enabled = bool(enabled)
        self.generation += 1
        return self.generation

    def intent_matches(self, *, enabled: bool, generation: int) -> bool:
        return generation == self.generation and self.intent_enabled == bool(enabled)

    def _publish_starting(self, starting: bool) -> None:
        if self.starting_sink is not None:
            self.starting_sink(bool(starting))

    async def set_enabled(self, enabled: bool) -> bool:
        request_generation = self.record_intent(enabled)
        if not enabled:
            self.pending_sink(False)
            self._publish_starting(False)
        state = self.state_provider()
        if self._ingress_stopped or state.ingress_frozen or not state.runtime_available:
            if not enabled:
                await self._teardown()
            self._publish_starting(False)
            return False
        self.log_basic(f"[Translation] Toggle request: enabled={enabled}")
        self.log_detailed(
            "[Translation] Toggle detail: "
            f"current_enabled={state.translation_enabled} "
            f"llm_available={state.llm_available}"
        )
        if enabled:
            self._publish_starting(True)
        try:
            if enabled and state.managed_selected:
                if not await self._prepare_managed(request_generation, state):
                    return False
            if enabled and not self.intent_matches(
                enabled=True,
                generation=request_generation,
            ):
                self.log_detailed(
                    "[Translation] Skipping stale enable request after newer toggle intent"
                )
                return False
            state = self.state_provider()
            if enabled and not state.llm_available:
                self.runtime_sink(False)
                self.dashboard_sink(False)
                self.log_error("Translation is ON but LLM provider is not configured.")
                return False
            if enabled and state.settings_available and state.provider_name is not None:
                self.log_basic(f"[Translation] Enabled with provider: {state.provider_name}")
                if state.provider_name == "qwen" and state.qwen_region is not None:
                    self.log_detailed(
                        "[Translation] Provider detail: "
                        f"provider={state.provider_name} region={state.qwen_region}"
                    )
            self.clear_context()
            self.runtime_sink(enabled)
            if enabled:
                await self.warmup()
            else:
                await self._teardown()
            return self.state_provider().translation_enabled
        finally:
            if self.generation == request_generation:
                self._publish_starting(False)

    async def _prepare_managed(
        self,
        request_generation: int,
        state: TranslationEnableState,
    ) -> bool:
        if await self.founder_route():
            return False
        if not state.managed_release_service_available:
            return True
        self.pending_sink(not state.managed_local_key_available)
        try:
            result = await self.managed_prepare()
        except Exception:
            self.pending_sink(False)
            raise
        self.pending_sink(False)
        if not self.intent_matches(enabled=True, generation=request_generation):
            self.log_detailed(
                "[Translation] Skipping stale managed enable result after newer toggle intent"
            )
            return False
        if result.transaction_result is not None:
            self.result_sink(result.transaction_result)
        if result.ready:
            current = self.state_provider()
            if not current.llm_available:
                await self.runtime_ensurer("if_missing")
            else:
                self.usage_refresh_sink()
            return True
        if result.diagnostics_text:
            self.log_error(f"[ManagedAuth] {result.diagnostics_text}")
        await self.usage_refresh_now()
        self.runtime_sink(False)
        self.dashboard_sink(False)
        if result.show_qq_dialog:
            self.qq_dialog_sink()
            return False
        if result.message_key is not None:
            self.message_sink(result.message_key, result.message_kwargs)
        return False

    async def _teardown(self) -> None:
        if self.teardown is None:
            return
        await self.teardown()

    def disable_for_managed_exhaustion(self, *, reopen_founder_letter: bool) -> None:
        self.record_intent(False)
        self.pending_sink(False)
        self._publish_starting(False)
        if reopen_founder_letter:
            self.founder_letter_sink()
        self.runtime_sink(False)
        self.dashboard_sink(False)

    def stop_ingress(self) -> None:
        self._ingress_stopped = True
        self.record_intent(False)
        self.pending_sink(False)
        self._publish_starting(False)

    async def close(self) -> None:
        self.stop_ingress()


__all__ = [
    "ManagedTranslationPreparation",
    "TranslationEnableOwner",
    "TranslationEnableState",
]
