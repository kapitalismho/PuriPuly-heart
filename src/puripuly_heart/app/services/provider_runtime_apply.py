"""Runtime-apply port adapters owned by the app-service boundary.

The ``RuntimeApplyPort`` implementations live here so the GuiController
dispatches provider/settings runtime apply *intent* instead of owning the
apply orchestration adapters inline. Each adapter delegates the actual runtime
mutation to a narrow ``ControllerRuntimeApplyBoundary`` protocol implemented
(structurally) by the controller, and returns a ``RuntimeApplyResult`` with
metadata-only diagnostics.

This module owns the runtime-apply result-construction family (degraded/save-
failed transaction builders and post-apply availability checks) so the
controller no longer carries runtime-apply orchestration state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from puripuly_heart.app.ports.runtime_apply import RuntimeApplyRequest
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


@dataclass(frozen=True, slots=True)
class _ProviderRuntimeApplyPlan:
    should_rebuild_llm: bool
    should_refresh_peer: bool
    should_refresh_self_stt: bool
    coordinated_gpu_restart: bool = False


class ControllerRuntimeApplyBoundary(Protocol):
    """Narrow controller surface used by the runtime-apply port adapters.

    The controller implements this protocol structurally. The adapters call
    these methods to mutate runtime state and to query post-apply
    availability; the controller keeps its settings-shape and hub knowledge
    while the app-service module owns the apply/result orchestration.
    """

    _stt_desired: bool
    hub: object | None

    async def _apply_provider_runtime_plan(
        self,
        settings: object,
        plan: "_ProviderRuntimeApplyPlan",
    ) -> None: ...

    async def _apply_settings_direct(
        self,
        settings: object,
        *,
        persist: bool = True,
        strict_runtime_errors: bool = False,
        reload_settings_view: bool = True,
    ) -> None: ...

    def _peer_runtime_should_be_active(self, settings: object) -> bool: ...

    def _is_qwen_llm(self, settings: object) -> bool: ...


def _settings_mutation_diagnostics(
    *,
    component: str,
    operation: str,
    code: str,
    category=DIAGNOSTIC_CATEGORY_TRANSACTION,
    surface: str = "translation_provider",
) -> ErrorDiagnostics:
    return ErrorDiagnostics(
        component=component,
        operation=operation,
        code=code,
        category=category,
        visibility=DIAGNOSTIC_VISIBILITY_BASIC,
        content_policy=CONTENT_POLICY_METADATA_ONLY,
        status_code=None,
        retry_after_ms=None,
        fields={"surface": surface},
    )


def _runtime_apply_failed_result(
    *,
    operation: str,
    code: str,
    surface: str,
) -> RuntimeApplyResult:
    return RuntimeApplyResult(
        status=RUNTIME_APPLY_STATUS_FAILED,
        message=UserMessageRef(
            key="settings.mutation.runtime_apply_failed",
            params={"phase": "runtime_apply"},
            severity=SEVERITY_WARNING,
        ),
        diagnostics=_settings_mutation_diagnostics(
            component="gui_controller",
            operation=operation,
            code=code,
            category=DIAGNOSTIC_CATEGORY_LIFECYCLE,
            surface=surface,
        ),
    )


def _runtime_apply_result_as_degraded_transaction(
    runtime_result: RuntimeApplyResult,
) -> TransactionResult:
    return TransactionResult(
        status=TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED,
        message=runtime_result.message,
        diagnostics=runtime_result.diagnostics,
    )


def _provider_runtime_apply_unavailable_result(
    *,
    controller: ControllerRuntimeApplyBoundary,
    settings: object,
    plan: _ProviderRuntimeApplyPlan,
    operation: str,
    surface: str,
) -> RuntimeApplyResult | None:
    if controller.hub is None:
        return None
    if plan.should_rebuild_llm and controller.hub.llm is None:
        return _runtime_apply_failed_result(
            operation=operation,
            code="provider_runtime_apply_unavailable",
            surface=surface,
        )
    if (
        plan.should_refresh_self_stt
        and controller._stt_desired
        and not controller.hub.has_stt_provider("self")
    ):
        return _runtime_apply_failed_result(
            operation=operation,
            code="stt_runtime_apply_unavailable",
            surface=surface,
        )
    if (
        plan.should_refresh_peer
        and controller._peer_runtime_should_be_active(settings)
        and not controller.hub.has_stt_provider("peer")
    ):
        return _runtime_apply_failed_result(
            operation=operation,
            code="peer_stt_runtime_apply_unavailable",
            surface=surface,
        )
    return None


def _stt_language_audio_runtime_unavailable_result(
    *,
    controller: ControllerRuntimeApplyBoundary,
    settings: object,
) -> RuntimeApplyResult | None:
    if controller.hub is None:
        return None
    if controller._stt_desired and not controller.hub.has_stt_provider("self"):
        return _runtime_apply_failed_result(
            operation="apply_stt_language_audio_runtime",
            code="stt_language_audio_runtime_unavailable",
            surface="stt_language_audio",
        )
    if controller._peer_runtime_should_be_active(settings) and not controller.hub.has_stt_provider(
        "peer"
    ):
        return _runtime_apply_failed_result(
            operation="apply_stt_language_audio_runtime",
            code="peer_stt_language_audio_runtime_unavailable",
            surface="stt_language_audio",
        )
    if controller._is_qwen_llm(settings) and controller.hub.llm is None:
        return _runtime_apply_failed_result(
            operation="apply_stt_language_audio_runtime",
            code="llm_stt_language_audio_runtime_unavailable",
            surface="stt_language_audio",
        )
    return None


def _stt_language_audio_runtime_degraded_transaction_result() -> TransactionResult:
    return _runtime_apply_result_as_degraded_transaction(
        _runtime_apply_failed_result(
            operation="apply_stt_language_audio_runtime",
            code="stt_language_audio_runtime_apply_exception",
            surface="stt_language_audio",
        )
    )


def _translation_provider_save_failed_transaction_result(*, operation: str) -> TransactionResult:
    return TransactionResult(
        status=TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED,
        message=UserMessageRef(
            key="settings.mutation.runtime_apply_failed",
            params={"phase": "settings_save"},
            severity=SEVERITY_WARNING,
        ),
        diagnostics=_settings_mutation_diagnostics(
            component="gui_controller",
            operation=operation,
            code="settings_save_failed",
            category=DIAGNOSTIC_CATEGORY_TRANSACTION,
            surface="translation_provider",
        ),
    )


def _stt_language_audio_save_failed_transaction_result(*, operation: str) -> TransactionResult:
    return TransactionResult(
        status=TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED,
        message=UserMessageRef(
            key="settings.mutation.runtime_apply_failed",
            params={"phase": "settings_save"},
            severity=SEVERITY_WARNING,
        ),
        diagnostics=_settings_mutation_diagnostics(
            component="gui_controller",
            operation=operation,
            code="settings_save_failed",
            category=DIAGNOSTIC_CATEGORY_TRANSACTION,
            surface="stt_language_audio",
        ),
    )


def _overlay_osc_output_runtime_degraded_transaction_result() -> TransactionResult:
    return _runtime_apply_result_as_degraded_transaction(
        _runtime_apply_failed_result(
            operation="apply_overlay_osc_output_runtime",
            code="overlay_osc_output_runtime_apply_exception",
            surface="overlay_osc_output",
        )
    )


def _overlay_osc_output_save_failed_transaction_result(*, operation: str) -> TransactionResult:
    return TransactionResult(
        status=TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED,
        message=UserMessageRef(
            key="settings.mutation.runtime_apply_failed",
            params={"phase": "settings_save"},
            severity=SEVERITY_WARNING,
        ),
        diagnostics=_settings_mutation_diagnostics(
            component="gui_controller",
            operation=operation,
            code="settings_save_failed",
            category=DIAGNOSTIC_CATEGORY_TRANSACTION,
            surface="overlay_osc_output",
        ),
    )


def _ui_prompt_clipboard_state_runtime_degraded_transaction_result() -> TransactionResult:
    return _runtime_apply_result_as_degraded_transaction(
        _runtime_apply_failed_result(
            operation="apply_ui_prompt_clipboard_state_runtime",
            code="ui_prompt_clipboard_state_runtime_apply_exception",
            surface="ui_prompt_clipboard_state",
        )
    )


def _ui_prompt_clipboard_state_save_failed_transaction_result(
    *, operation: str
) -> TransactionResult:
    return TransactionResult(
        status=TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED,
        message=UserMessageRef(
            key="settings.mutation.runtime_apply_failed",
            params={"phase": "settings_save"},
            severity=SEVERITY_WARNING,
        ),
        diagnostics=_settings_mutation_diagnostics(
            component="gui_controller",
            operation=operation,
            code="settings_save_failed",
            category=DIAGNOSTIC_CATEGORY_TRANSACTION,
            surface="ui_prompt_clipboard_state",
        ),
    )


@dataclass(slots=True)
class _ControllerProviderRuntimeApply:
    controller: ControllerRuntimeApplyBoundary
    settings: object
    plan: _ProviderRuntimeApplyPlan
    surface: str = "translation_provider"
    operation: str = "apply_provider_runtime"

    async def apply_runtime(self, request: RuntimeApplyRequest) -> RuntimeApplyResult:
        _ = request
        try:
            await self.controller._apply_provider_runtime_plan(self.settings, self.plan)
        except Exception:
            return RuntimeApplyResult(
                status=RUNTIME_APPLY_STATUS_FAILED,
                message=UserMessageRef(
                    key="settings.mutation.runtime_apply_failed",
                    params={"phase": "runtime_apply"},
                    severity=SEVERITY_WARNING,
                ),
                diagnostics=_settings_mutation_diagnostics(
                    component="gui_controller",
                    operation=self.operation,
                    code="provider_runtime_apply_exception",
                    category=DIAGNOSTIC_CATEGORY_LIFECYCLE,
                    surface=self.surface,
                ),
            )
        unavailable_result = _provider_runtime_apply_unavailable_result(
            controller=self.controller,
            settings=self.settings,
            plan=self.plan,
            operation=self.operation,
            surface=self.surface,
        )
        if unavailable_result is not None:
            return unavailable_result
        return RuntimeApplyResult(
            status=RUNTIME_APPLY_STATUS_APPLIED,
            message=None,
            diagnostics=None,
        )


@dataclass(slots=True)
class _ControllerSttLanguageAudioRuntimeApply:
    controller: ControllerRuntimeApplyBoundary
    settings: object
    reload_settings_view: bool = True

    async def apply_runtime(self, request: RuntimeApplyRequest) -> RuntimeApplyResult:
        _ = request
        try:
            await self.controller._apply_settings_direct(
                self.settings,
                persist=False,
                strict_runtime_errors=True,
                reload_settings_view=self.reload_settings_view,
            )
        except Exception:
            return RuntimeApplyResult(
                status=RUNTIME_APPLY_STATUS_FAILED,
                message=UserMessageRef(
                    key="settings.mutation.runtime_apply_failed",
                    params={"phase": "runtime_apply"},
                    severity=SEVERITY_WARNING,
                ),
                diagnostics=_settings_mutation_diagnostics(
                    component="gui_controller",
                    operation="apply_stt_language_audio_runtime",
                    code="stt_language_audio_runtime_apply_exception",
                    category=DIAGNOSTIC_CATEGORY_LIFECYCLE,
                    surface="stt_language_audio",
                ),
            )
        unavailable_result = _stt_language_audio_runtime_unavailable_result(
            controller=self.controller,
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
class _ControllerOverlayOscOutputRuntimeApply:
    controller: ControllerRuntimeApplyBoundary
    settings: object

    async def apply_runtime(self, request: RuntimeApplyRequest) -> RuntimeApplyResult:
        _ = request
        try:
            await self.controller._apply_settings_direct(
                self.settings,
                persist=False,
                strict_runtime_errors=True,
            )
        except Exception:
            return _runtime_apply_failed_result(
                operation="apply_overlay_osc_output_runtime",
                code="overlay_osc_output_runtime_apply_exception",
                surface="overlay_osc_output",
            )
        return RuntimeApplyResult(
            status=RUNTIME_APPLY_STATUS_APPLIED,
            message=None,
            diagnostics=None,
        )


@dataclass(slots=True)
class _ControllerUiPromptClipboardStateRuntimeApply:
    controller: ControllerRuntimeApplyBoundary
    settings: object

    async def apply_runtime(self, request: RuntimeApplyRequest) -> RuntimeApplyResult:
        _ = request
        try:
            await self.controller._apply_settings_direct(
                self.settings,
                persist=False,
                strict_runtime_errors=True,
            )
        except Exception:
            return _runtime_apply_failed_result(
                operation="apply_ui_prompt_clipboard_state_runtime",
                code="ui_prompt_clipboard_state_runtime_apply_exception",
                surface="ui_prompt_clipboard_state",
            )
        return RuntimeApplyResult(
            status=RUNTIME_APPLY_STATUS_APPLIED,
            message=None,
            diagnostics=None,
        )


@dataclass(slots=True)
class _ControllerNoopRuntimeApply:
    async def apply_runtime(self, request: RuntimeApplyRequest) -> RuntimeApplyResult:
        _ = request
        return RuntimeApplyResult(
            status=RUNTIME_APPLY_STATUS_APPLIED,
            message=None,
            diagnostics=None,
        )


__all__ = [
    "ControllerRuntimeApplyBoundary",
]
