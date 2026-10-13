from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import ClassVar, Final, Protocol

from puripuly_heart.app.ports._settings_values import freeze_settings_values
from puripuly_heart.app.ports.runtime_apply import RuntimeApplyPort, RuntimeApplyRequest
from puripuly_heart.app.ports.settings_repository import (
    SettingsCommitRequest,
    SettingsCommitResult,
    SettingsRepositoryPort,
    SettingsSnapshot,
)
from puripuly_heart.core.diagnostics import safe_exception_fields
from puripuly_heart.core.messages import (
    CONTENT_POLICY_METADATA_ONLY,
    DIAGNOSTIC_CATEGORY_LIFECYCLE,
    DIAGNOSTIC_VISIBILITY_BASIC,
    RUNTIME_APPLY_STATUS_APPLIED,
    SEVERITY_WARNING,
    TRANSACTION_STATUS_SETTINGS_COMMIT_FAILED,
    TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
    TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED,
    TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_INTERRUPTED,
    ErrorDiagnostics,
    RuntimeApplyResult,
    TransactionResult,
    UserMessageRef,
)


class SettingsMutationInterrupted(asyncio.CancelledError):
    def __init__(self, transaction_result: TransactionResult) -> None:
        self.transaction_result = transaction_result
        super().__init__("Settings runtime application interrupted after commit")


SETTINGS_MUTATION_SURFACE_TRANSLATION_PROVIDER: Final = "settings.translation_provider"
SETTINGS_MUTATION_SURFACE_STT_LANGUAGE_AUDIO: Final = "settings.stt_language_audio"
SETTINGS_MUTATION_SURFACE_OVERLAY_OSC_OUTPUT: Final = "settings.overlay_osc_output"
SETTINGS_MUTATION_SURFACE_UI_PROMPT_CLIPBOARD_STATE: Final = "settings.ui_prompt_clipboard_state"


@dataclass(frozen=True, slots=True)
class SettingsMutationRequest:
    values: Mapping[str, object]
    expected_revision: str | None
    reason: str | None
    correlation_id: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", freeze_settings_values(self.values))


@dataclass(frozen=True, slots=True)
class SettingsMutationValidationResult:
    succeeded: bool
    message: UserMessageRef | None
    diagnostics: ErrorDiagnostics | None


class SettingsMutationCommand(Protocol):
    values: Mapping[str, object]
    surface: ClassVar[str]

    def to_mutation_request(
        self,
        *,
        expected_revision: str | None,
        correlation_id: str | None,
    ) -> SettingsMutationRequest: ...


@dataclass(frozen=True, slots=True)
class TranslationProviderSettingsMutation:
    values: Mapping[str, object]
    surface: ClassVar[str] = SETTINGS_MUTATION_SURFACE_TRANSLATION_PROVIDER

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", freeze_settings_values(self.values))

    def to_mutation_request(
        self,
        *,
        expected_revision: str | None,
        correlation_id: str | None,
    ) -> SettingsMutationRequest:
        return SettingsMutationRequest(
            values=self.values,
            expected_revision=expected_revision,
            reason=SETTINGS_MUTATION_SURFACE_TRANSLATION_PROVIDER,
            correlation_id=correlation_id,
        )


@dataclass(frozen=True, slots=True)
class SttLanguageAudioSettingsMutation:
    values: Mapping[str, object]
    surface: ClassVar[str] = SETTINGS_MUTATION_SURFACE_STT_LANGUAGE_AUDIO

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", freeze_settings_values(self.values))

    def to_mutation_request(
        self,
        *,
        expected_revision: str | None,
        correlation_id: str | None,
    ) -> SettingsMutationRequest:
        return SettingsMutationRequest(
            values=self.values,
            expected_revision=expected_revision,
            reason=SETTINGS_MUTATION_SURFACE_STT_LANGUAGE_AUDIO,
            correlation_id=correlation_id,
        )


@dataclass(frozen=True, slots=True)
class OverlayOscOutputSettingsMutation:
    values: Mapping[str, object]
    surface: ClassVar[str] = SETTINGS_MUTATION_SURFACE_OVERLAY_OSC_OUTPUT

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", freeze_settings_values(self.values))

    def to_mutation_request(
        self,
        *,
        expected_revision: str | None,
        correlation_id: str | None,
    ) -> SettingsMutationRequest:
        return SettingsMutationRequest(
            values=self.values,
            expected_revision=expected_revision,
            reason=SETTINGS_MUTATION_SURFACE_OVERLAY_OSC_OUTPUT,
            correlation_id=correlation_id,
        )


@dataclass(frozen=True, slots=True)
class UiPromptClipboardStateSettingsMutation:
    values: Mapping[str, object]
    surface: ClassVar[str] = SETTINGS_MUTATION_SURFACE_UI_PROMPT_CLIPBOARD_STATE

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", freeze_settings_values(self.values))

    def to_mutation_request(
        self,
        *,
        expected_revision: str | None,
        correlation_id: str | None,
    ) -> SettingsMutationRequest:
        return SettingsMutationRequest(
            values=self.values,
            expected_revision=expected_revision,
            reason=SETTINGS_MUTATION_SURFACE_UI_PROMPT_CLIPBOARD_STATE,
            correlation_id=correlation_id,
        )


class SettingsMutationValidator(Protocol):
    async def validate(
        self,
        request: SettingsMutationRequest,
    ) -> SettingsMutationValidationResult: ...


class SettingsSnapshotPublisher(Protocol):
    async def publish_settings_snapshot(
        self,
        snapshot: SettingsSnapshot,
        *,
        correlation_id: str | None,
    ) -> None: ...


class RuntimeApplyResultPublisher(Protocol):
    async def publish_runtime_apply_result(
        self,
        result: RuntimeApplyResult,
        *,
        correlation_id: str | None,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class SettingsMutationService:
    settings_repository: SettingsRepositoryPort
    runtime_apply: RuntimeApplyPort
    validator: SettingsMutationValidator
    snapshot_publisher: SettingsSnapshotPublisher | None = None
    runtime_result_publisher: RuntimeApplyResultPublisher | None = None

    async def mutate(self, request: SettingsMutationRequest) -> TransactionResult:
        validation_result = await self.validator.validate(request)
        if not validation_result.succeeded:
            return TransactionResult(
                status=TRANSACTION_STATUS_SETTINGS_COMMIT_FAILED,
                message=validation_result.message,
                diagnostics=validation_result.diagnostics,
            )

        try:
            commit_result = await self.settings_repository.save(
                SettingsCommitRequest(
                    values=request.values,
                    expected_revision=request.expected_revision,
                    reason=request.reason,
                )
            )
        except Exception as exc:
            return TransactionResult(
                status=TRANSACTION_STATUS_SETTINGS_COMMIT_FAILED,
                message=None,
                diagnostics=_phase_diagnostics("settings_save_failed", "settings_save", exc),
            )
        if not commit_result.succeeded or commit_result.snapshot is None:
            return TransactionResult(
                status=TRANSACTION_STATUS_SETTINGS_COMMIT_FAILED,
                message=commit_result.message,
                diagnostics=commit_result.diagnostics,
                secondary_diagnostics=getattr(
                    self.settings_repository, "secondary_diagnostics", ()
                ),
                secondary_omitted_count=getattr(
                    self.settings_repository, "secondary_omitted_count", 0
                ),
            )

        snapshot = commit_result.snapshot
        secondary = list(getattr(self.settings_repository, "secondary_diagnostics", ()))
        secondary_omitted_count = getattr(self.settings_repository, "secondary_omitted_count", 0)
        if self.snapshot_publisher is not None:
            try:
                await self.snapshot_publisher.publish_settings_snapshot(
                    snapshot,
                    correlation_id=request.correlation_id,
                )
            except Exception as exc:
                secondary.append(
                    _phase_diagnostics(
                        "settings_snapshot_notification_failed", "snapshot_notification", exc
                    )
                )

        try:
            runtime_result = await self.runtime_apply.apply_runtime(
                RuntimeApplyRequest(
                    settings_values=snapshot.values,
                    reason=request.reason,
                    correlation_id=request.correlation_id,
                )
            )
        except asyncio.CancelledError:
            runtime_result = RuntimeApplyResult(
                status="degraded",
                message=None,
                diagnostics=_phase_diagnostics("runtime_apply_interrupted", "runtime_apply", None),
            )
            if self.runtime_result_publisher is not None:
                try:
                    await self.runtime_result_publisher.publish_runtime_apply_result(
                        runtime_result, correlation_id=request.correlation_id
                    )
                except Exception as exc:
                    secondary.append(
                        _phase_diagnostics(
                            "runtime_result_notification_failed", "runtime_notification", exc
                        )
                    )
            transaction_result = TransactionResult(
                status=TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_INTERRUPTED,
                message=None,
                diagnostics=runtime_result.diagnostics,
                secondary_diagnostics=tuple(secondary),
                secondary_omitted_count=secondary_omitted_count,
            )
            raise SettingsMutationInterrupted(transaction_result)
        except Exception as exc:
            runtime_result = RuntimeApplyResult(
                status="failed",
                message=UserMessageRef(
                    key="settings.mutation.runtime_apply_failed",
                    params={"phase": "runtime_apply"},
                    severity=SEVERITY_WARNING,
                ),
                diagnostics=_phase_diagnostics("runtime_apply_exception", "runtime_apply", exc),
            )

        transaction_result = _transaction_result_for_commit_and_runtime(
            commit_result=commit_result,
            runtime_result=runtime_result,
        )
        secondary.extend(runtime_result.secondary_diagnostics)
        if self.runtime_result_publisher is not None:
            try:
                await self.runtime_result_publisher.publish_runtime_apply_result(
                    runtime_result,
                    correlation_id=request.correlation_id,
                )
            except Exception as exc:
                secondary.append(
                    _phase_diagnostics(
                        "runtime_result_notification_failed", "runtime_notification", exc
                    )
                )

        return replace(
            transaction_result,
            secondary_diagnostics=tuple(secondary),
            secondary_omitted_count=secondary_omitted_count
            + runtime_result.secondary_omitted_count,
        )


def _transaction_result_for_commit_and_runtime(
    *,
    commit_result: SettingsCommitResult,
    runtime_result: RuntimeApplyResult,
) -> TransactionResult:
    transaction_status = (
        TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED
        if runtime_result.status == RUNTIME_APPLY_STATUS_APPLIED
        else TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED
    )
    return TransactionResult(
        status=transaction_status,
        message=(
            runtime_result.message if runtime_result.message is not None else commit_result.message
        ),
        diagnostics=(
            runtime_result.diagnostics
            if runtime_result.diagnostics is not None
            else commit_result.diagnostics
        ),
    )


def _phase_diagnostics(code: str, phase: str, exception: BaseException | None) -> ErrorDiagnostics:
    return ErrorDiagnostics(
        component="settings_mutation",
        operation=phase,
        code=code,
        category=DIAGNOSTIC_CATEGORY_LIFECYCLE,
        visibility=DIAGNOSTIC_VISIBILITY_BASIC,
        content_policy=CONTENT_POLICY_METADATA_ONLY,
        status_code=None,
        retry_after_ms=None,
        fields={
            "phase": phase,
            **(safe_exception_fields(exception) if exception is not None else {}),
        },
    )


__all__ = [
    "OverlayOscOutputSettingsMutation",
    "SettingsMutationInterrupted",
    "RuntimeApplyResultPublisher",
    "SETTINGS_MUTATION_SURFACE_OVERLAY_OSC_OUTPUT",
    "SETTINGS_MUTATION_SURFACE_STT_LANGUAGE_AUDIO",
    "SETTINGS_MUTATION_SURFACE_TRANSLATION_PROVIDER",
    "SETTINGS_MUTATION_SURFACE_UI_PROMPT_CLIPBOARD_STATE",
    "SettingsMutationCommand",
    "SettingsMutationRequest",
    "SettingsMutationService",
    "SettingsMutationValidationResult",
    "SettingsMutationValidator",
    "SettingsSnapshotPublisher",
    "SttLanguageAudioSettingsMutation",
    "TranslationProviderSettingsMutation",
    "UiPromptClipboardStateSettingsMutation",
]
