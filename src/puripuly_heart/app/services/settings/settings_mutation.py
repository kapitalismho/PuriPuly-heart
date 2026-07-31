from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar, Final, Protocol

from puripuly_heart.app.ports._settings_values import freeze_settings_values
from puripuly_heart.app.ports.runtime_apply import RuntimeApplyPort, RuntimeApplyRequest
from puripuly_heart.app.ports.settings_repository import (
    SettingsCommitRequest,
    SettingsCommitResult,
    SettingsRepositoryPort,
    SettingsSnapshot,
)
from puripuly_heart.core.messages import (
    CONTENT_POLICY_METADATA_ONLY,
    DIAGNOSTIC_CATEGORY_LIFECYCLE,
    DIAGNOSTIC_VISIBILITY_BASIC,
    RUNTIME_APPLY_STATUS_APPLIED,
    SEVERITY_WARNING,
    TRANSACTION_STATUS_SETTINGS_COMMIT_FAILED,
    TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
    TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED,
    ErrorDiagnostics,
    RuntimeApplyResult,
    TransactionResult,
    UserMessageRef,
)

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

        commit_result = await self.settings_repository.save(
            SettingsCommitRequest(
                values=request.values,
                expected_revision=request.expected_revision,
                reason=request.reason,
            )
        )
        if not commit_result.succeeded or commit_result.snapshot is None:
            return TransactionResult(
                status=TRANSACTION_STATUS_SETTINGS_COMMIT_FAILED,
                message=commit_result.message,
                diagnostics=commit_result.diagnostics,
            )

        snapshot = commit_result.snapshot
        if self.snapshot_publisher is not None:
            try:
                await self.snapshot_publisher.publish_settings_snapshot(
                    snapshot,
                    correlation_id=request.correlation_id,
                )
            except Exception:
                pass

        try:
            runtime_result = await self.runtime_apply.apply_runtime(
                RuntimeApplyRequest(
                    settings_values=snapshot.values,
                    reason=request.reason,
                    correlation_id=request.correlation_id,
                )
            )
        except Exception:
            return _runtime_apply_exception_transaction_result()

        transaction_result = _transaction_result_for_commit_and_runtime(
            commit_result=commit_result,
            runtime_result=runtime_result,
        )
        if self.runtime_result_publisher is not None:
            try:
                await self.runtime_result_publisher.publish_runtime_apply_result(
                    runtime_result,
                    correlation_id=request.correlation_id,
                )
            except Exception:
                pass

        return transaction_result


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


def _runtime_apply_exception_transaction_result() -> TransactionResult:
    return TransactionResult(
        status=TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED,
        message=UserMessageRef(
            key="settings.mutation.runtime_apply_failed",
            params={"phase": "runtime_apply"},
            severity=SEVERITY_WARNING,
        ),
        diagnostics=ErrorDiagnostics(
            component="settings_mutation",
            operation="runtime_apply",
            code="runtime_apply_exception",
            category=DIAGNOSTIC_CATEGORY_LIFECYCLE,
            visibility=DIAGNOSTIC_VISIBILITY_BASIC,
            content_policy=CONTENT_POLICY_METADATA_ONLY,
            status_code=None,
            retry_after_ms=None,
            fields={"phase": "runtime_apply"},
        ),
    )


__all__ = [
    "OverlayOscOutputSettingsMutation",
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
