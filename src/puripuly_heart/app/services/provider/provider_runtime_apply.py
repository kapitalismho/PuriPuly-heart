"""Application-owned provider and settings runtime orchestration."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field

from puripuly_heart.app.ports.runtime_apply import RuntimeApplyRequest
from puripuly_heart.app.ports.settings_runtime_effects import SettingsRuntimeState
from puripuly_heart.core.diagnostics import safe_exception_fields
from puripuly_heart.core.messages import (
    CONTENT_POLICY_METADATA_ONLY,
    DIAGNOSTIC_CATEGORY_LIFECYCLE,
    DIAGNOSTIC_CATEGORY_TRANSACTION,
    DIAGNOSTIC_VISIBILITY_BASIC,
    RUNTIME_APPLY_STATUS_APPLIED,
    RUNTIME_APPLY_STATUS_FAILED,
    SEVERITY_WARNING,
    TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED,
    ErrorDiagnostics,
    RuntimeApplyResult,
    TransactionResult,
    UserMessageRef,
)
from puripuly_heart.core.runtime.logging import (
    LIVE_AUDIENCE_BASIC,
    LIVE_AUDIENCE_RECORD_ATTRIBUTE,
)
from puripuly_heart.core.runtime.provider_rebuild import ProviderRuntimeRebuildService

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ProviderRuntimeApplyPlan:
    should_rebuild_llm: bool
    should_refresh_peer: bool
    should_refresh_self_stt: bool
    coordinated_gpu_restart: bool = False


@dataclass(frozen=True, slots=True)
class ProviderRuntimeState:
    runtime_available: bool
    llm_available: bool
    self_stt_available: bool
    peer_stt_available: bool
    self_stt_desired: bool
    peer_stt_desired: bool
    self_failure_diagnostics: ErrorDiagnostics | None = None
    peer_failure_diagnostics: ErrorDiagnostics | None = None


ProviderRuntimeStateProvider = Callable[[object], ProviderRuntimeState]
ProviderRuntimeCommonEffect = Callable[[object], None]
ProviderRuntimeAsyncEffect = Callable[[], Awaitable[None]]
ProviderRuntimeGpuEffect = Callable[
    [object, ProviderRuntimeApplyPlan],
    Awaitable[None],
]
ProviderRuntimeSignatureSink = Callable[[object], None]
ProviderRuntimeRetrySink = Callable[[], None]
ProviderRuntimeSettingsProvider = Callable[[], object | None]
ProviderRuntimeSignatureCacheProvider = Callable[
    [],
    tuple[object | None, object | None, object | None],
]
ProviderRuntimeSignatureBuilder = Callable[[object], object]
ProviderRuntimePeerSignatureBuilder = Callable[[object, object | None], object]
ProviderRuntimeGpuRestartDecision = Callable[[object, object], bool]
ProviderRuntimeSelfConvergenceProvider = Callable[[object], bool | None]
ProviderRuntimePeerConvergenceProvider = Callable[[object], bool | None]
LlmProviderReplace = Callable[[object | None], Awaitable[object | None]]
LlmProviderFactory = Callable[[object], object | Awaitable[object | None] | None]
LlmProviderRebuildContextProvider = Callable[[], "LlmProviderRebuildContext | None"]
LlmProviderAvailabilitySink = Callable[[bool], None]
LlmProviderMessageSink = Callable[[str], None]


def _unknown_self_runtime_convergence(_settings: object) -> bool | None:
    return None


def _unknown_peer_runtime_convergence(_settings: object) -> bool | None:
    return None


class ProviderRuntimeConvergenceError(RuntimeError):
    code = "stt_runtime_apply_not_converged"

    def __init__(
        self, message: str, *, failure_diagnostics: ErrorDiagnostics | None = None
    ) -> None:
        self.failure_diagnostics = failure_diagnostics
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class LlmProviderRebuildContext:
    settings: object
    replace_provider: LlmProviderReplace
    requires_secret: bool
    resource_label: str = "LLM provider"


@dataclass(slots=True)
class LlmProviderRebuildOwner:
    context_provider: LlmProviderRebuildContextProvider
    provider_factory: LlmProviderFactory
    availability_sink: LlmProviderAvailabilitySink
    usage_refresh: ProviderRuntimeAsyncEffect
    failure_sink: LlmProviderMessageSink
    success_sink: LlmProviderMessageSink
    runtime: ProviderRuntimeRebuildService = dataclass_field(
        default_factory=ProviderRuntimeRebuildService,
    )

    async def rebuild(self) -> bool:
        context = self.context_provider()
        if context is None:
            return False
        outcome = await self.runtime.rebuild_llm_provider(
            replace_provider=context.replace_provider,
            create_provider=lambda: self.provider_factory(context.settings),
        )
        provider = outcome.provider
        self.availability_sink(provider is None and context.requires_secret)
        await self.usage_refresh()
        if provider is None:
            self.failure_sink(f"{context.resource_label} not available")
            return False
        self.success_sink(f"[Settings] {context.resource_label} rebuilt successfully")
        return True


@dataclass(slots=True)
class ProviderRuntimeOwner:
    state_provider: ProviderRuntimeStateProvider
    common_effect: ProviderRuntimeCommonEffect
    rebuild_llm: ProviderRuntimeAsyncEffect
    recover_gpu: ProviderRuntimeGpuEffect
    refresh_peer: ProviderRuntimeAsyncEffect
    refresh_self_stt: ProviderRuntimeAsyncEffect
    signature_sink: ProviderRuntimeSignatureSink
    llm_retry_sink: ProviderRuntimeRetrySink
    current_settings_provider: ProviderRuntimeSettingsProvider
    signature_cache_provider: ProviderRuntimeSignatureCacheProvider
    self_signature_builder: ProviderRuntimeSignatureBuilder
    peer_signature_builder: ProviderRuntimePeerSignatureBuilder
    llm_signature_builder: ProviderRuntimeSignatureBuilder
    gpu_restart_decision: ProviderRuntimeGpuRestartDecision
    self_runtime_convergence: ProviderRuntimeSelfConvergenceProvider = (
        _unknown_self_runtime_convergence
    )
    peer_runtime_convergence: ProviderRuntimePeerConvergenceProvider = (
        _unknown_peer_runtime_convergence
    )

    def build_plan(
        self,
        next_settings: object,
        *,
        force_rebuild_llm: bool,
        canonical_settings: object | None = None,
    ) -> ProviderRuntimeApplyPlan:
        current_settings = self.current_settings_provider()
        self_signature, peer_signature, llm_signature = self.signature_cache_provider()
        if current_settings is not None:
            if self_signature is None:
                self_signature = self.self_signature_builder(current_settings)
            if peer_signature is None:
                peer_signature = self.peer_signature_builder(current_settings, None)
            if llm_signature is None:
                llm_signature = self.llm_signature_builder(current_settings)
        next_self_signature = self.self_signature_builder(next_settings)
        next_peer_signature = self.peer_signature_builder(
            next_settings,
            canonical_settings,
        )
        next_llm_signature = self.llm_signature_builder(next_settings)
        return ProviderRuntimeApplyPlan(
            should_rebuild_llm=(
                force_rebuild_llm or llm_signature is None or next_llm_signature != llm_signature
            ),
            should_refresh_peer=(
                self.state_provider(next_settings).peer_stt_desired
                and (
                    peer_signature is None
                    or next_peer_signature != peer_signature
                    or self.peer_runtime_convergence(next_settings) is False
                )
            ),
            should_refresh_self_stt=(
                self_signature is None
                or next_self_signature != self_signature
                or self.self_runtime_convergence(next_settings) is False
            ),
            coordinated_gpu_restart=(
                current_settings is not None
                and self.gpu_restart_decision(current_settings, next_settings)
            ),
        )

    async def apply(
        self,
        settings: object,
        plan: ProviderRuntimeApplyPlan,
    ) -> None:
        self.common_effect(settings)
        if plan.should_rebuild_llm:
            await self.rebuild_llm()
        if plan.coordinated_gpu_restart:
            await self.recover_gpu(settings, plan)
            self._require_stt_runtime_convergence(settings)
            self.signature_sink(settings)
            if plan.should_rebuild_llm and not self.state_provider(settings).llm_available:
                self.llm_retry_sink()
            return
        if plan.should_refresh_peer:
            await self.refresh_peer()
        if plan.should_refresh_self_stt:
            await self.refresh_self_stt()
        self._require_stt_runtime_convergence(settings)
        self.signature_sink(settings)
        if plan.should_rebuild_llm and not self.state_provider(settings).llm_available:
            self.llm_retry_sink()

    def _require_stt_runtime_convergence(self, settings: object) -> None:
        if self.self_runtime_convergence(settings) is False:
            raise ProviderRuntimeConvergenceError(
                "Self STT runtime did not converge to requested settings",
                failure_diagnostics=self.state_provider(settings).self_failure_diagnostics,
            )
        if self.peer_runtime_convergence(settings) is False:
            raise ProviderRuntimeConvergenceError(
                "Peer STT runtime did not converge to requested settings",
                failure_diagnostics=self.state_provider(settings).peer_failure_diagnostics,
            )

    def unavailable_result(
        self,
        settings: object,
        plan: ProviderRuntimeApplyPlan,
        *,
        operation: str,
        surface: str,
    ) -> RuntimeApplyResult | None:
        state = self.state_provider(settings)
        if not state.runtime_available:
            return None
        if plan.should_rebuild_llm and not state.llm_available:
            return _runtime_apply_failed_result(
                operation=operation,
                code="provider_runtime_apply_unavailable",
                surface=surface,
            )
        if plan.should_refresh_self_stt and state.self_stt_desired and not state.self_stt_available:
            return _runtime_apply_failed_result(
                operation=operation,
                code="stt_runtime_apply_unavailable",
                surface=surface,
            )
        if plan.should_refresh_peer and state.peer_stt_desired and not state.peer_stt_available:
            return _runtime_apply_failed_result(
                operation=operation,
                code="peer_stt_runtime_apply_unavailable",
                surface=surface,
            )
        return None


SettingsRuntimeApplyEffect = Callable[[object, bool], Awaitable[None]]
SettingsRuntimeStateProvider = Callable[[object], SettingsRuntimeState]


def _settings_mutation_diagnostics(
    *,
    component: str,
    operation: str,
    code: str,
    category=DIAGNOSTIC_CATEGORY_TRANSACTION,
    surface: str = "translation_provider",
    exception: BaseException | None = None,
) -> ErrorDiagnostics:
    current = exception
    for _ in range(4):
        if current is None:
            break
        diagnostics = getattr(current, "failure_diagnostics", None)
        if not isinstance(diagnostics, ErrorDiagnostics):
            diagnostics = getattr(current, "diagnostics", None)
        if isinstance(diagnostics, ErrorDiagnostics):
            return replace(diagnostics, fields={**diagnostics.fields, "surface": surface})
        current = current.__cause__
    return ErrorDiagnostics(
        component=component,
        operation=operation,
        code=code,
        category=category,
        visibility=DIAGNOSTIC_VISIBILITY_BASIC,
        content_policy=CONTENT_POLICY_METADATA_ONLY,
        status_code=None,
        retry_after_ms=None,
        fields={
            "surface": surface,
            "phase": (
                "settings_save"
                if code == "settings_save_failed"
                else (
                    operation
                    if operation
                    in {
                        "rollback",
                        "commit_notification",
                        "failure_notification",
                        "compensate_local_asr",
                        "resync_committed_runtime",
                        "restore_memory",
                    }
                    else "runtime_apply"
                )
            ),
            **(safe_exception_fields(exception) if exception is not None else {}),
        },
    )


def _exception_code(exc: BaseException, fallback: str) -> str:
    from puripuly_heart.app.services.capture.self_capture_application import (
        SelfCaptureRuntimeApplyError,
    )

    if isinstance(exc, ProviderRuntimeConvergenceError):
        return ProviderRuntimeConvergenceError.code
    if isinstance(exc, SelfCaptureRuntimeApplyError):
        return SelfCaptureRuntimeApplyError.code
    return fallback


def _runtime_apply_failed_result(
    *,
    operation: str,
    code: str,
    surface: str,
    exception: BaseException | None = None,
) -> RuntimeApplyResult:
    return RuntimeApplyResult(
        status=RUNTIME_APPLY_STATUS_FAILED,
        message=UserMessageRef(
            key="settings.mutation.runtime_apply_failed",
            params={"phase": "runtime_apply"},
            severity=SEVERITY_WARNING,
        ),
        diagnostics=_settings_mutation_diagnostics(
            component="settings_application",
            operation=operation,
            code=code,
            category=DIAGNOSTIC_CATEGORY_LIFECYCLE,
            surface=surface,
            exception=exception,
        ),
    )


def _runtime_apply_result_as_degraded_transaction(
    runtime_result: RuntimeApplyResult,
) -> TransactionResult:
    return TransactionResult(
        status=TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED,
        message=runtime_result.message,
        diagnostics=runtime_result.diagnostics,
        secondary_diagnostics=runtime_result.secondary_diagnostics,
        secondary_omitted_count=runtime_result.secondary_omitted_count,
    )


def _provider_runtime_apply_unavailable_result(
    *,
    owner: ProviderRuntimeOwner,
    settings: object,
    plan: ProviderRuntimeApplyPlan,
    operation: str,
    surface: str,
) -> RuntimeApplyResult | None:
    return owner.unavailable_result(
        settings,
        plan,
        operation=operation,
        surface=surface,
    )


def _stt_language_audio_runtime_unavailable_result(
    *,
    state: SettingsRuntimeState,
    settings: object,
) -> RuntimeApplyResult | None:
    _ = settings
    if not state.runtime_available:
        return None
    if state.self_stt_desired and not state.self_stt_available:
        return _runtime_apply_failed_result(
            operation="apply_stt_language_audio_runtime",
            code="stt_language_audio_runtime_unavailable",
            surface="stt_language_audio",
        )
    if state.peer_stt_desired and not state.peer_stt_available:
        return _runtime_apply_failed_result(
            operation="apply_stt_language_audio_runtime",
            code="peer_stt_language_audio_runtime_unavailable",
            surface="stt_language_audio",
        )
    if state.qwen_llm_desired and not state.llm_available:
        return _runtime_apply_failed_result(
            operation="apply_stt_language_audio_runtime",
            code="llm_stt_language_audio_runtime_unavailable",
            surface="stt_language_audio",
        )
    return None


def _stt_language_audio_runtime_degraded_transaction_result(
    *, code: str | None = None, exception: BaseException | None = None
) -> TransactionResult:
    return _runtime_apply_result_as_degraded_transaction(
        _runtime_apply_failed_result(
            operation="apply_stt_language_audio_runtime",
            code=code or "stt_language_audio_runtime_apply_exception",
            surface="stt_language_audio",
            exception=exception,
        )
    )


def _translation_provider_save_failed_transaction_result(
    *, operation: str, exception: BaseException | None = None
) -> TransactionResult:
    return TransactionResult(
        status=TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED,
        message=UserMessageRef(
            key="settings.mutation.runtime_apply_failed",
            params={"phase": "settings_save"},
            severity=SEVERITY_WARNING,
        ),
        diagnostics=_settings_mutation_diagnostics(
            component="settings_application",
            operation=operation,
            code="settings_save_failed",
            category=DIAGNOSTIC_CATEGORY_TRANSACTION,
            surface="translation_provider",
            exception=exception,
        ),
    )


def _stt_language_audio_save_failed_transaction_result(
    *, operation: str, exception: BaseException | None = None
) -> TransactionResult:
    return TransactionResult(
        status=TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED,
        message=UserMessageRef(
            key="settings.mutation.runtime_apply_failed",
            params={"phase": "settings_save"},
            severity=SEVERITY_WARNING,
        ),
        diagnostics=_settings_mutation_diagnostics(
            component="settings_application",
            operation=operation,
            code="settings_save_failed",
            category=DIAGNOSTIC_CATEGORY_TRANSACTION,
            surface="stt_language_audio",
            exception=exception,
        ),
    )


def _overlay_osc_output_runtime_degraded_transaction_result(
    *, code: str | None = None, exception: BaseException | None = None
) -> TransactionResult:
    return _runtime_apply_result_as_degraded_transaction(
        _runtime_apply_failed_result(
            operation="apply_overlay_osc_output_runtime",
            code=code or "overlay_osc_output_runtime_apply_exception",
            surface="overlay_osc_output",
            exception=exception,
        )
    )


def _overlay_osc_output_save_failed_transaction_result(
    *, operation: str, exception: BaseException | None = None
) -> TransactionResult:
    return TransactionResult(
        status=TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED,
        message=UserMessageRef(
            key="settings.mutation.runtime_apply_failed",
            params={"phase": "settings_save"},
            severity=SEVERITY_WARNING,
        ),
        diagnostics=_settings_mutation_diagnostics(
            component="settings_application",
            operation=operation,
            code="settings_save_failed",
            category=DIAGNOSTIC_CATEGORY_TRANSACTION,
            surface="overlay_osc_output",
            exception=exception,
        ),
    )


def _ui_prompt_clipboard_state_runtime_degraded_transaction_result(
    *, code: str | None = None, exception: BaseException | None = None
) -> TransactionResult:
    return _runtime_apply_result_as_degraded_transaction(
        _runtime_apply_failed_result(
            operation="apply_ui_prompt_clipboard_state_runtime",
            code=code or "ui_prompt_clipboard_state_runtime_apply_exception",
            surface="ui_prompt_clipboard_state",
            exception=exception,
        )
    )


def _ui_prompt_clipboard_state_save_failed_transaction_result(
    *, operation: str, exception: BaseException | None = None
) -> TransactionResult:
    return TransactionResult(
        status=TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED,
        message=UserMessageRef(
            key="settings.mutation.runtime_apply_failed",
            params={"phase": "settings_save"},
            severity=SEVERITY_WARNING,
        ),
        diagnostics=_settings_mutation_diagnostics(
            component="settings_application",
            operation=operation,
            code="settings_save_failed",
            category=DIAGNOSTIC_CATEGORY_TRANSACTION,
            surface="ui_prompt_clipboard_state",
            exception=exception,
        ),
    )


@dataclass(slots=True)
class ProviderRuntimeApplyAdapter:
    owner: ProviderRuntimeOwner
    settings: object
    plan: ProviderRuntimeApplyPlan
    surface: str = "translation_provider"
    operation: str = "apply_provider_runtime"

    async def apply_runtime(self, request: RuntimeApplyRequest) -> RuntimeApplyResult:
        _ = request
        try:
            await self.owner.apply(self.settings, self.plan)
        except Exception as exc:
            return RuntimeApplyResult(
                status=RUNTIME_APPLY_STATUS_FAILED,
                message=UserMessageRef(
                    key="settings.mutation.runtime_apply_failed",
                    params={"phase": "runtime_apply"},
                    severity=SEVERITY_WARNING,
                ),
                diagnostics=_settings_mutation_diagnostics(
                    component="provider_application",
                    operation=self.operation,
                    code=_exception_code(exc, "provider_runtime_apply_exception"),
                    category=DIAGNOSTIC_CATEGORY_LIFECYCLE,
                    surface=self.surface,
                    exception=exc,
                ),
            )
        unavailable_result = _provider_runtime_apply_unavailable_result(
            owner=self.owner,
            settings=self.settings,
            plan=self.plan,
            operation=self.operation,
            surface=self.surface,
        )
        if unavailable_result is not None:
            return unavailable_result
        logger.log(
            logging.INFO,
            "[Settings] provider_apply_completed operation=%s refresh_self=%s refresh_peer=%s",
            self.operation,
            self.plan.should_refresh_self_stt,
            self.plan.should_refresh_peer,
            extra={LIVE_AUDIENCE_RECORD_ATTRIBUTE: LIVE_AUDIENCE_BASIC},
        )
        return RuntimeApplyResult(
            status=RUNTIME_APPLY_STATUS_APPLIED,
            message=None,
            diagnostics=None,
        )


@dataclass(slots=True)
class SttLanguageAudioRuntimeApplyAdapter:
    apply_settings: SettingsRuntimeApplyEffect
    state_provider: SettingsRuntimeStateProvider
    settings: object
    reload_settings_view: bool = True

    async def apply_runtime(self, request: RuntimeApplyRequest) -> RuntimeApplyResult:
        _ = request
        try:
            await self.apply_settings(
                self.settings,
                self.reload_settings_view,
            )
        except Exception as exc:
            return RuntimeApplyResult(
                status=RUNTIME_APPLY_STATUS_FAILED,
                message=UserMessageRef(
                    key="settings.mutation.runtime_apply_failed",
                    params={"phase": "runtime_apply"},
                    severity=SEVERITY_WARNING,
                ),
                diagnostics=_settings_mutation_diagnostics(
                    component="settings_application",
                    operation="apply_stt_language_audio_runtime",
                    code=_exception_code(
                        exc,
                        "stt_language_audio_runtime_apply_exception",
                    ),
                    category=DIAGNOSTIC_CATEGORY_LIFECYCLE,
                    surface="stt_language_audio",
                    exception=exc,
                ),
            )
        unavailable_result = _stt_language_audio_runtime_unavailable_result(
            state=self.state_provider(self.settings),
            settings=self.settings,
        )
        if unavailable_result is not None:
            return unavailable_result
        return RuntimeApplyResult(
            status=RUNTIME_APPLY_STATUS_APPLIED,
            message=None,
            diagnostics=None,
        )


@dataclass(slots=True)
class OverlayOscOutputRuntimeApplyAdapter:
    apply_settings: SettingsRuntimeApplyEffect
    settings: object

    async def apply_runtime(self, request: RuntimeApplyRequest) -> RuntimeApplyResult:
        _ = request
        try:
            await self.apply_settings(
                self.settings,
                True,
            )
        except Exception as exc:
            return _runtime_apply_failed_result(
                operation="apply_overlay_osc_output_runtime",
                code=_exception_code(exc, "overlay_osc_output_runtime_apply_exception"),
                surface="overlay_osc_output",
                exception=exc,
            )
        return RuntimeApplyResult(
            status=RUNTIME_APPLY_STATUS_APPLIED,
            message=None,
            diagnostics=None,
        )


@dataclass(slots=True)
class UiPromptClipboardStateRuntimeApplyAdapter:
    apply_settings: SettingsRuntimeApplyEffect
    settings: object

    async def apply_runtime(self, request: RuntimeApplyRequest) -> RuntimeApplyResult:
        _ = request
        try:
            await self.apply_settings(
                self.settings,
                True,
            )
        except Exception as exc:
            return _runtime_apply_failed_result(
                operation="apply_ui_prompt_clipboard_state_runtime",
                code=_exception_code(
                    exc,
                    "ui_prompt_clipboard_state_runtime_apply_exception",
                ),
                surface="ui_prompt_clipboard_state",
                exception=exc,
            )
        return RuntimeApplyResult(
            status=RUNTIME_APPLY_STATUS_APPLIED,
            message=None,
            diagnostics=None,
        )


@dataclass(slots=True)
class NoopRuntimeApply:
    async def apply_runtime(self, request: RuntimeApplyRequest) -> RuntimeApplyResult:
        _ = request
        return RuntimeApplyResult(
            status=RUNTIME_APPLY_STATUS_APPLIED,
            message=None,
            diagnostics=None,
        )


__all__ = [
    "LlmProviderRebuildContext",
    "LlmProviderRebuildOwner",
    "NoopRuntimeApply",
    "OverlayOscOutputRuntimeApplyAdapter",
    "ProviderRuntimeApplyAdapter",
    "ProviderRuntimeApplyPlan",
    "ProviderRuntimeConvergenceError",
    "ProviderRuntimeOwner",
    "ProviderRuntimePeerConvergenceProvider",
    "ProviderRuntimeSelfConvergenceProvider",
    "ProviderRuntimeState",
    "SettingsRuntimeState",
    "SttLanguageAudioRuntimeApplyAdapter",
    "UiPromptClipboardStateRuntimeApplyAdapter",
]
