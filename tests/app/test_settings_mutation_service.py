from __future__ import annotations

import ast
import importlib
import inspect
from collections.abc import Mapping
from dataclasses import FrozenInstanceError, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import get_type_hints

import pytest

from puripuly_heart.app.ports import runtime_apply, settings_repository
from puripuly_heart.core import messages

SERVICE_MODULE = "puripuly_heart.app.services.settings_mutation"

FORBIDDEN_SERVICE_IMPORT_PREFIXES = (
    "flet",
    "keyring",
    "puripuly_heart.app.adapters",
    "puripuly_heart.app.wiring",
    "puripuly_heart.config.settings",
    "puripuly_heart.config.settings_vnext",
    "puripuly_heart.config.runtime_resolution",
    "puripuly_heart.core.managed_openrouter_broker_client",
    "puripuly_heart.core.orchestrator",
    "puripuly_heart.core.runtime",
    "puripuly_heart.core.storage",
    "puripuly_heart.providers",
    "puripuly_heart.ui",
)


@dataclass(slots=True)
class RecordingSettingsRepository:
    result: settings_repository.SettingsCommitResult
    saved_requests: list[settings_repository.SettingsCommitRequest] = field(default_factory=list)

    async def load(self) -> settings_repository.SettingsSnapshot:
        raise AssertionError("SettingsMutationService should not load in these scenarios")

    async def save(
        self,
        request: settings_repository.SettingsCommitRequest,
    ) -> settings_repository.SettingsCommitResult:
        self.saved_requests.append(request)
        return self.result


@dataclass(slots=True)
class RecordingRuntimeApply:
    result: messages.RuntimeApplyResult
    requests: list[runtime_apply.RuntimeApplyRequest] = field(default_factory=list)

    async def apply_runtime(
        self,
        request: runtime_apply.RuntimeApplyRequest,
    ) -> messages.RuntimeApplyResult:
        self.requests.append(request)
        return self.result


@dataclass(slots=True)
class RaisingRuntimeApply:
    exception: Exception
    requests: list[runtime_apply.RuntimeApplyRequest] = field(default_factory=list)

    async def apply_runtime(
        self,
        request: runtime_apply.RuntimeApplyRequest,
    ) -> messages.RuntimeApplyResult:
        self.requests.append(request)
        raise self.exception


@dataclass(slots=True)
class RecordingSettingsSnapshotPublisher:
    publications: list[tuple[settings_repository.SettingsSnapshot, str | None]] = field(
        default_factory=list
    )

    async def publish_settings_snapshot(
        self,
        snapshot: settings_repository.SettingsSnapshot,
        *,
        correlation_id: str | None,
    ) -> None:
        self.publications.append((snapshot, correlation_id))


@dataclass(slots=True)
class RaisingSettingsSnapshotPublisher:
    publications: list[tuple[settings_repository.SettingsSnapshot, str | None]] = field(
        default_factory=list
    )

    async def publish_settings_snapshot(
        self,
        snapshot: settings_repository.SettingsSnapshot,
        *,
        correlation_id: str | None,
    ) -> None:
        self.publications.append((snapshot, correlation_id))
        raise RuntimeError("raw snapshot publisher failure should not leak")


@dataclass(slots=True)
class RecordingRuntimeResultPublisher:
    publications: list[tuple[messages.RuntimeApplyResult, str | None]] = field(default_factory=list)

    async def publish_runtime_apply_result(
        self,
        result: messages.RuntimeApplyResult,
        *,
        correlation_id: str | None,
    ) -> None:
        self.publications.append((result, correlation_id))


@dataclass(slots=True)
class RaisingRuntimeResultPublisher:
    publications: list[tuple[messages.RuntimeApplyResult, str | None]] = field(default_factory=list)

    async def publish_runtime_apply_result(
        self,
        result: messages.RuntimeApplyResult,
        *,
        correlation_id: str | None,
    ) -> None:
        self.publications.append((result, correlation_id))
        raise RuntimeError("raw runtime result publisher failure should not leak")


@dataclass(slots=True)
class RecordingSettingsMutationValidator:
    result: object
    requests: list[object] = field(default_factory=list)

    async def validate(self, request: object) -> object:
        self.requests.append(request)
        return self.result


def _service_module():
    return importlib.import_module(SERVICE_MODULE)


def _message(
    key: str,
    *,
    severity: messages.Severity = messages.SEVERITY_ERROR,
) -> messages.UserMessageRef:
    return messages.UserMessageRef(
        key=key,
        params={"phase": "settings_mutation"},
        severity=severity,
    )


def _diagnostics(code: str, *, operation: str) -> messages.ErrorDiagnostics:
    return messages.ErrorDiagnostics(
        component="settings_mutation",
        operation=operation,
        code=code,
        category=messages.DIAGNOSTIC_CATEGORY_TRANSACTION,
        visibility=messages.DIAGNOSTIC_VISIBILITY_BASIC,
        content_policy=messages.CONTENT_POLICY_METADATA_ONLY,
        status_code=None,
        retry_after_ms=None,
        fields={"phase": operation},
    )


def _service(
    *,
    repository: RecordingSettingsRepository,
    runtime: RecordingRuntimeApply | RaisingRuntimeApply,
    snapshot_publisher: (
        RecordingSettingsSnapshotPublisher | RaisingSettingsSnapshotPublisher | None
    ) = None,
    runtime_result_publisher: (
        RecordingRuntimeResultPublisher | RaisingRuntimeResultPublisher | None
    ) = None,
    validator: RecordingSettingsMutationValidator | None = None,
):
    settings_mutation = _service_module()
    if validator is None:
        validator = RecordingSettingsMutationValidator(
            settings_mutation.SettingsMutationValidationResult(
                succeeded=True,
                message=None,
                diagnostics=None,
            )
        )
    kwargs = {
        "settings_repository": repository,
        "runtime_apply": runtime,
        "snapshot_publisher": snapshot_publisher,
        "runtime_result_publisher": runtime_result_publisher,
        "validator": validator,
    }
    return settings_mutation.SettingsMutationService(**kwargs)


def test_settings_mutation_request_is_frozen_slotted_and_deep_freezes_payload() -> None:
    settings_mutation = _service_module()

    values = {
        "provider": {
            "aliases": ["openrouter"],
            "options": {"streaming": True},
        }
    }
    request = settings_mutation.SettingsMutationRequest(
        values=values,
        expected_revision="settings-r1",
        reason="user_patch",
        correlation_id="corr-1",
    )

    values["provider"]["aliases"].append("qwen")
    values["provider"]["options"]["streaming"] = False

    assert is_dataclass(request)
    assert not hasattr(request, "__dict__")
    assert {field.name for field in fields(request)} == {
        "values",
        "expected_revision",
        "reason",
        "correlation_id",
    }

    with pytest.raises(FrozenInstanceError):
        request.reason = "other"  # type: ignore[misc]
    with pytest.raises(TypeError):
        request.values["provider"] = "qwen"  # type: ignore[index]

    provider = request.values["provider"]
    assert isinstance(provider, Mapping)
    assert provider["aliases"] == ("openrouter",)
    assert isinstance(provider["options"], Mapping)
    assert provider["options"]["streaming"] is True

    hints = get_type_hints(settings_mutation.SettingsMutationRequest)
    assert hints["values"] == Mapping[str, object]
    assert hints["expected_revision"] == str | None
    assert hints["reason"] == str | None
    assert hints["correlation_id"] == str | None


def test_settings_mutation_validation_contract_is_frozen_slotted_async_protocol() -> None:
    settings_mutation = _service_module()
    validation_message = _message("settings.validation.failed")
    validation_diagnostics = _diagnostics("validation_failed", operation="validate")

    result = settings_mutation.SettingsMutationValidationResult(
        succeeded=False,
        message=validation_message,
        diagnostics=validation_diagnostics,
    )

    assert is_dataclass(result)
    assert not hasattr(result, "__dict__")
    assert {field.name for field in fields(result)} == {
        "succeeded",
        "message",
        "diagnostics",
    }

    with pytest.raises(FrozenInstanceError):
        result.succeeded = True  # type: ignore[misc]

    result_hints = get_type_hints(settings_mutation.SettingsMutationValidationResult)
    assert result_hints["succeeded"] is bool
    assert result_hints["message"] == messages.UserMessageRef | None
    assert result_hints["diagnostics"] == messages.ErrorDiagnostics | None

    validator = settings_mutation.SettingsMutationValidator
    assert getattr(validator, "_is_protocol", False)
    assert inspect.iscoroutinefunction(validator.validate)

    validator_hints = get_type_hints(validator.validate)
    assert validator_hints["request"] == settings_mutation.SettingsMutationRequest
    assert validator_hints["return"] == settings_mutation.SettingsMutationValidationResult


def test_settings_mutation_service_requires_validation_owner_dependency() -> None:
    settings_mutation = _service_module()

    signature = inspect.signature(settings_mutation.SettingsMutationService)

    assert "validator" in signature.parameters
    assert signature.parameters["validator"].default is inspect.Signature.empty


@pytest.mark.asyncio
async def test_validation_failure_returns_transaction_failure_without_save_runtime_or_publications() -> (
    None
):
    settings_mutation = _service_module()
    validation_message = _message("settings.validation.failed")
    validation_diagnostics = _diagnostics("validation_failed", operation="validate")
    validator = RecordingSettingsMutationValidator(
        settings_mutation.SettingsMutationValidationResult(
            succeeded=False,
            message=validation_message,
            diagnostics=validation_diagnostics,
        )
    )
    committed_snapshot = settings_repository.SettingsSnapshot(
        values={"provider": "openrouter"},
        revision="settings-r2",
    )
    repository = RecordingSettingsRepository(
        settings_repository.SettingsCommitResult(
            succeeded=True,
            snapshot=committed_snapshot,
            message=None,
            diagnostics=None,
        )
    )
    runtime = RecordingRuntimeApply(
        messages.RuntimeApplyResult(
            status=messages.RUNTIME_APPLY_STATUS_APPLIED,
            message=None,
            diagnostics=None,
        )
    )
    snapshot_publisher = RecordingSettingsSnapshotPublisher()
    runtime_result_publisher = RecordingRuntimeResultPublisher()
    request = settings_mutation.SettingsMutationRequest(
        values={"provider": "openrouter"},
        expected_revision="settings-r1",
        reason="user_patch",
        correlation_id="corr-validation",
    )

    result = await _service(
        repository=repository,
        runtime=runtime,
        snapshot_publisher=snapshot_publisher,
        runtime_result_publisher=runtime_result_publisher,
        validator=validator,
    ).mutate(request)

    assert result == messages.TransactionResult(
        status=messages.TRANSACTION_STATUS_SETTINGS_COMMIT_FAILED,
        message=validation_message,
        diagnostics=validation_diagnostics,
    )
    assert validator.requests == [request]
    assert repository.saved_requests == []
    assert runtime.requests == []
    assert snapshot_publisher.publications == []
    assert runtime_result_publisher.publications == []


@pytest.mark.asyncio
async def test_commit_failure_returns_transaction_failure_without_runtime_or_publications() -> None:
    settings_mutation = _service_module()
    commit_message = _message("settings.commit.failed")
    commit_diagnostics = _diagnostics("commit_failed", operation="save")
    repository = RecordingSettingsRepository(
        settings_repository.SettingsCommitResult(
            succeeded=False,
            snapshot=None,
            message=commit_message,
            diagnostics=commit_diagnostics,
        )
    )
    runtime = RecordingRuntimeApply(
        messages.RuntimeApplyResult(
            status=messages.RUNTIME_APPLY_STATUS_APPLIED,
            message=None,
            diagnostics=None,
        )
    )
    snapshot_publisher = RecordingSettingsSnapshotPublisher()
    runtime_result_publisher = RecordingRuntimeResultPublisher()

    result = await _service(
        repository=repository,
        runtime=runtime,
        snapshot_publisher=snapshot_publisher,
        runtime_result_publisher=runtime_result_publisher,
    ).mutate(
        settings_mutation.SettingsMutationRequest(
            values={"provider": "openrouter"},
            expected_revision="settings-r1",
            reason="user_patch",
            correlation_id="corr-1",
        )
    )

    assert result == messages.TransactionResult(
        status=messages.TRANSACTION_STATUS_SETTINGS_COMMIT_FAILED,
        message=commit_message,
        diagnostics=commit_diagnostics,
    )
    assert len(repository.saved_requests) == 1
    assert repository.saved_requests[0].expected_revision == "settings-r1"
    assert repository.saved_requests[0].reason == "user_patch"
    assert runtime.requests == []
    assert snapshot_publisher.publications == []
    assert runtime_result_publisher.publications == []


@pytest.mark.asyncio
async def test_commit_success_with_runtime_applied_publishes_snapshot_and_runtime_result() -> None:
    settings_mutation = _service_module()
    committed_snapshot = settings_repository.SettingsSnapshot(
        values={"provider": {"aliases": ["committed"]}},
        revision="settings-r2",
    )
    repository = RecordingSettingsRepository(
        settings_repository.SettingsCommitResult(
            succeeded=True,
            snapshot=committed_snapshot,
            message=_message("settings.commit.applied", severity=messages.SEVERITY_INFO),
            diagnostics=None,
        )
    )
    runtime_message = _message("runtime.apply.applied", severity=messages.SEVERITY_INFO)
    runtime_result = messages.RuntimeApplyResult(
        status=messages.RUNTIME_APPLY_STATUS_APPLIED,
        message=runtime_message,
        diagnostics=None,
    )
    runtime = RecordingRuntimeApply(runtime_result)
    snapshot_publisher = RecordingSettingsSnapshotPublisher()
    runtime_result_publisher = RecordingRuntimeResultPublisher()

    result = await _service(
        repository=repository,
        runtime=runtime,
        snapshot_publisher=snapshot_publisher,
        runtime_result_publisher=runtime_result_publisher,
    ).mutate(
        settings_mutation.SettingsMutationRequest(
            values={"provider": {"aliases": ["draft"]}},
            expected_revision="settings-r1",
            reason="user_patch",
            correlation_id="corr-2",
        )
    )

    assert result == messages.TransactionResult(
        status=messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
        message=runtime_message,
        diagnostics=None,
    )
    assert len(repository.saved_requests) == 1
    assert repository.saved_requests[0].values["provider"]["aliases"] == ("draft",)
    assert repository.saved_requests[0].expected_revision == "settings-r1"
    assert repository.saved_requests[0].reason == "user_patch"
    assert snapshot_publisher.publications == [(committed_snapshot, "corr-2")]
    assert runtime.requests == [
        runtime_apply.RuntimeApplyRequest(
            settings_values=committed_snapshot.values,
            reason="user_patch",
            correlation_id="corr-2",
        )
    ]
    assert runtime_result_publisher.publications == [(runtime_result, "corr-2")]


@pytest.mark.parametrize(
    "runtime_status",
    [messages.RUNTIME_APPLY_STATUS_DEGRADED, messages.RUNTIME_APPLY_STATUS_FAILED],
)
@pytest.mark.asyncio
async def test_commit_success_with_runtime_degraded_or_failed_returns_degraded_transaction(
    runtime_status: messages.RuntimeApplyStatus,
) -> None:
    settings_mutation = _service_module()
    committed_snapshot = settings_repository.SettingsSnapshot(
        values={"provider": "openrouter"},
        revision="settings-r2",
    )
    repository = RecordingSettingsRepository(
        settings_repository.SettingsCommitResult(
            succeeded=True,
            snapshot=committed_snapshot,
            message=None,
            diagnostics=None,
        )
    )
    runtime_message = _message("runtime.apply.degraded", severity=messages.SEVERITY_WARNING)
    runtime_diagnostics = _diagnostics(f"runtime_{runtime_status}", operation="apply")
    runtime_result = messages.RuntimeApplyResult(
        status=runtime_status,
        message=runtime_message,
        diagnostics=runtime_diagnostics,
    )
    runtime = RecordingRuntimeApply(runtime_result)
    snapshot_publisher = RecordingSettingsSnapshotPublisher()
    runtime_result_publisher = RecordingRuntimeResultPublisher()

    result = await _service(
        repository=repository,
        runtime=runtime,
        snapshot_publisher=snapshot_publisher,
        runtime_result_publisher=runtime_result_publisher,
    ).mutate(
        settings_mutation.SettingsMutationRequest(
            values={"provider": "openrouter"},
            expected_revision=None,
            reason="user_patch",
            correlation_id="corr-3",
        )
    )

    assert result == messages.TransactionResult(
        status=messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED,
        message=runtime_message,
        diagnostics=runtime_diagnostics,
    )
    assert runtime.requests == [
        runtime_apply.RuntimeApplyRequest(
            settings_values=committed_snapshot.values,
            reason="user_patch",
            correlation_id="corr-3",
        )
    ]
    assert snapshot_publisher.publications == [(committed_snapshot, "corr-3")]
    assert runtime_result_publisher.publications == [(runtime_result, "corr-3")]


@pytest.mark.asyncio
async def test_snapshot_publisher_failure_still_applies_runtime_and_returns_runtime_result() -> (
    None
):
    settings_mutation = _service_module()
    committed_snapshot = settings_repository.SettingsSnapshot(
        values={"provider": "openrouter"},
        revision="settings-r2",
    )
    repository = RecordingSettingsRepository(
        settings_repository.SettingsCommitResult(
            succeeded=True,
            snapshot=committed_snapshot,
            message=_message("settings.commit.applied", severity=messages.SEVERITY_INFO),
            diagnostics=None,
        )
    )
    runtime_message = _message("runtime.apply.applied", severity=messages.SEVERITY_INFO)
    runtime_diagnostics = _diagnostics("runtime_applied", operation="apply")
    runtime_result = messages.RuntimeApplyResult(
        status=messages.RUNTIME_APPLY_STATUS_APPLIED,
        message=runtime_message,
        diagnostics=runtime_diagnostics,
    )
    runtime = RecordingRuntimeApply(runtime_result)
    snapshot_publisher = RaisingSettingsSnapshotPublisher()
    runtime_result_publisher = RecordingRuntimeResultPublisher()

    result = await _service(
        repository=repository,
        runtime=runtime,
        snapshot_publisher=snapshot_publisher,
        runtime_result_publisher=runtime_result_publisher,
    ).mutate(
        settings_mutation.SettingsMutationRequest(
            values={"provider": "draft"},
            expected_revision="settings-r1",
            reason="user_patch",
            correlation_id="corr-snapshot-publisher",
        )
    )

    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED
    assert result.message == runtime_message
    assert result.diagnostics is runtime_diagnostics
    assert result.secondary_diagnostics[0].code == "settings_snapshot_notification_failed"
    assert result.secondary_diagnostics[0].fields["exception_class"] == "RuntimeError"
    assert snapshot_publisher.publications == [(committed_snapshot, "corr-snapshot-publisher")]
    assert runtime.requests == [
        runtime_apply.RuntimeApplyRequest(
            settings_values=committed_snapshot.values,
            reason="user_patch",
            correlation_id="corr-snapshot-publisher",
        )
    ]
    assert runtime_result_publisher.publications == [(runtime_result, "corr-snapshot-publisher")]


@pytest.mark.asyncio
async def test_runtime_result_publisher_failure_returns_known_runtime_result() -> None:
    settings_mutation = _service_module()
    committed_snapshot = settings_repository.SettingsSnapshot(
        values={"provider": "openrouter"},
        revision="settings-r2",
    )
    repository = RecordingSettingsRepository(
        settings_repository.SettingsCommitResult(
            succeeded=True,
            snapshot=committed_snapshot,
            message=None,
            diagnostics=None,
        )
    )
    runtime_message = _message("runtime.apply.degraded", severity=messages.SEVERITY_WARNING)
    runtime_diagnostics = _diagnostics("runtime_degraded", operation="apply")
    runtime_result = messages.RuntimeApplyResult(
        status=messages.RUNTIME_APPLY_STATUS_DEGRADED,
        message=runtime_message,
        diagnostics=runtime_diagnostics,
    )
    runtime = RecordingRuntimeApply(runtime_result)
    snapshot_publisher = RecordingSettingsSnapshotPublisher()
    runtime_result_publisher = RaisingRuntimeResultPublisher()

    result = await _service(
        repository=repository,
        runtime=runtime,
        snapshot_publisher=snapshot_publisher,
        runtime_result_publisher=runtime_result_publisher,
    ).mutate(
        settings_mutation.SettingsMutationRequest(
            values={"provider": "draft"},
            expected_revision="settings-r1",
            reason="user_patch",
            correlation_id="corr-runtime-publisher",
        )
    )

    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED
    assert result.message == runtime_message
    assert result.diagnostics is runtime_diagnostics
    assert result.secondary_diagnostics[0].code == "runtime_result_notification_failed"
    assert snapshot_publisher.publications == [(committed_snapshot, "corr-runtime-publisher")]
    assert runtime.requests == [
        runtime_apply.RuntimeApplyRequest(
            settings_values=committed_snapshot.values,
            reason="user_patch",
            correlation_id="corr-runtime-publisher",
        )
    ]
    assert runtime_result_publisher.publications == [(runtime_result, "corr-runtime-publisher")]


@pytest.mark.asyncio
async def test_runtime_apply_exception_returns_controlled_degraded_result_without_runtime_publish() -> (
    None
):
    settings_mutation = _service_module()
    committed_snapshot = settings_repository.SettingsSnapshot(
        values={"provider": "openrouter"},
        revision="settings-r2",
    )
    repository = RecordingSettingsRepository(
        settings_repository.SettingsCommitResult(
            succeeded=True,
            snapshot=committed_snapshot,
            message=None,
            diagnostics=None,
        )
    )
    runtime = RaisingRuntimeApply(RuntimeError("raw runtime secret-token failure"))
    snapshot_publisher = RecordingSettingsSnapshotPublisher()
    runtime_result_publisher = RecordingRuntimeResultPublisher()

    result = await _service(
        repository=repository,
        runtime=runtime,
        snapshot_publisher=snapshot_publisher,
        runtime_result_publisher=runtime_result_publisher,
    ).mutate(
        settings_mutation.SettingsMutationRequest(
            values={"provider": "draft"},
            expected_revision="settings-r1",
            reason="user_patch",
            correlation_id="corr-runtime-exception",
        )
    )

    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED
    assert result.diagnostics is not None
    assert result.diagnostics.code == "runtime_apply_exception"
    assert result.diagnostics.fields == {
        "phase": "runtime_apply",
        "exception_class": "RuntimeError",
    }
    assert "raw runtime secret-token failure" not in repr(result)
    assert snapshot_publisher.publications == [(committed_snapshot, "corr-runtime-exception")]
    assert runtime.requests == [
        runtime_apply.RuntimeApplyRequest(
            settings_values=committed_snapshot.values,
            reason="user_patch",
            correlation_id="corr-runtime-exception",
        )
    ]
    assert runtime_result_publisher.publications[0][0].diagnostics is result.diagnostics
    assert runtime_result_publisher.publications[0][0].status == "failed"


@pytest.mark.asyncio
async def test_commit_success_message_and_diagnostics_are_runtime_fallbacks() -> None:
    settings_mutation = _service_module()
    committed_snapshot = settings_repository.SettingsSnapshot(
        values={"provider": "openrouter"},
        revision="settings-r2",
    )
    commit_message = _message("settings.commit.applied", severity=messages.SEVERITY_INFO)
    commit_diagnostics = _diagnostics("settings_commit_applied", operation="save")
    repository = RecordingSettingsRepository(
        settings_repository.SettingsCommitResult(
            succeeded=True,
            snapshot=committed_snapshot,
            message=commit_message,
            diagnostics=commit_diagnostics,
        )
    )
    runtime = RecordingRuntimeApply(
        messages.RuntimeApplyResult(
            status=messages.RUNTIME_APPLY_STATUS_APPLIED,
            message=None,
            diagnostics=None,
        )
    )

    result = await _service(repository=repository, runtime=runtime).mutate(
        settings_mutation.SettingsMutationRequest(
            values={"provider": "draft"},
            expected_revision="settings-r1",
            reason="user_patch",
            correlation_id="corr-fallback",
        )
    )

    assert result == messages.TransactionResult(
        status=messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
        message=commit_message,
        diagnostics=commit_diagnostics,
    )
    assert runtime.requests == [
        runtime_apply.RuntimeApplyRequest(
            settings_values=committed_snapshot.values,
            reason="user_patch",
            correlation_id="corr-fallback",
        )
    ]


@pytest.mark.asyncio
async def test_runtime_message_and_diagnostics_take_precedence_over_commit_values() -> None:
    settings_mutation = _service_module()
    committed_snapshot = settings_repository.SettingsSnapshot(
        values={"provider": "openrouter"},
        revision="settings-r2",
    )
    repository = RecordingSettingsRepository(
        settings_repository.SettingsCommitResult(
            succeeded=True,
            snapshot=committed_snapshot,
            message=_message("settings.commit.applied", severity=messages.SEVERITY_INFO),
            diagnostics=_diagnostics("settings_commit_applied", operation="save"),
        )
    )
    runtime_message = _message("runtime.apply.degraded", severity=messages.SEVERITY_WARNING)
    runtime_diagnostics = _diagnostics("runtime_degraded", operation="apply")
    runtime = RecordingRuntimeApply(
        messages.RuntimeApplyResult(
            status=messages.RUNTIME_APPLY_STATUS_DEGRADED,
            message=runtime_message,
            diagnostics=runtime_diagnostics,
        )
    )

    result = await _service(repository=repository, runtime=runtime).mutate(
        settings_mutation.SettingsMutationRequest(
            values={"provider": "draft"},
            expected_revision="settings-r1",
            reason="user_patch",
            correlation_id="corr-runtime-precedence",
        )
    )

    assert result == messages.TransactionResult(
        status=messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED,
        message=runtime_message,
        diagnostics=runtime_diagnostics,
    )


def test_typed_stt_language_audio_mutation_builds_request_for_order22_surface() -> None:
    settings_mutation = _service_module()

    mutation = settings_mutation.SttLanguageAudioSettingsMutation(
        values={
            "languages.source_language": "ja",
            "stt.low_latency_mode": False,
            "audio.input_device": "Headset Mic",
        }
    )

    request = mutation.to_mutation_request(
        expected_revision="settings-r2",
        correlation_id="corr-order22",
    )

    assert request == settings_mutation.SettingsMutationRequest(
        values={
            "languages.source_language": "ja",
            "stt.low_latency_mode": False,
            "audio.input_device": "Headset Mic",
        },
        expected_revision="settings-r2",
        reason=settings_mutation.SETTINGS_MUTATION_SURFACE_STT_LANGUAGE_AUDIO,
        correlation_id="corr-order22",
    )


def test_typed_overlay_osc_output_mutation_builds_request_for_order23_surface() -> None:
    settings_mutation = _service_module()

    mutation = settings_mutation.OverlayOscOutputSettingsMutation(
        values={
            "overlay.show_translation": False,
            "overlay.desktop_flet.size_preset": "large",
            "osc.chatbox_max_chars": 120,
        }
    )

    request = mutation.to_mutation_request(
        expected_revision="settings-r3",
        correlation_id="corr-order23",
    )

    assert request == settings_mutation.SettingsMutationRequest(
        values={
            "overlay.show_translation": False,
            "overlay.desktop_flet.size_preset": "large",
            "osc.chatbox_max_chars": 120,
        },
        expected_revision="settings-r3",
        reason=settings_mutation.SETTINGS_MUTATION_SURFACE_OVERLAY_OSC_OUTPUT,
        correlation_id="corr-order23",
    )


def test_typed_ui_prompt_clipboard_state_mutation_builds_request_for_order24_surface() -> None:
    settings_mutation = _service_module()

    mutation = settings_mutation.UiPromptClipboardStateSettingsMutation(
        values={
            "ui.locale": "ja",
            "ui.clipboard_auto_translate_enabled": True,
            "system_prompt": "custom translation style",
        }
    )

    request = mutation.to_mutation_request(
        expected_revision="settings-r4",
        correlation_id="corr-order24",
    )

    assert request == settings_mutation.SettingsMutationRequest(
        values={
            "ui.locale": "ja",
            "ui.clipboard_auto_translate_enabled": True,
            "system_prompt": "custom translation style",
        },
        expected_revision="settings-r4",
        reason=settings_mutation.SETTINGS_MUTATION_SURFACE_UI_PROMPT_CLIPBOARD_STATE,
        correlation_id="corr-order24",
    )


def test_typed_settings_mutation_command_freezes_values_payload() -> None:
    settings_mutation = _service_module()

    values: dict[str, object] = {
        "languages.source_language": "ja",
        "nested": {"enabled": True},
    }
    mutation = settings_mutation.SttLanguageAudioSettingsMutation(values=values)
    values["languages.source_language"] = "en"
    values["nested"] = {"enabled": False}

    assert mutation.values["languages.source_language"] == "ja"
    nested = mutation.values["nested"]
    assert isinstance(nested, Mapping)
    assert nested["enabled"] is True


def test_settings_mutation_service_module_avoids_concrete_ui_provider_and_i18n_imports() -> None:
    module = _service_module()
    tree = ast.parse(Path(module.__file__ or "").read_text(encoding="utf-8"))
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)

    assert not {
        imported
        for imported in imported_modules
        for forbidden in FORBIDDEN_SERVICE_IMPORT_PREFIXES
        if imported == forbidden or imported.startswith(f"{forbidden}.")
    }


@pytest.mark.asyncio
async def test_combined_secondary_causes_preserve_order_and_known_omission_count():
    repository_causes = tuple(
        _diagnostics("settings_save_failed", operation="save") for _ in range(3)
    )
    runtime_causes = tuple(
        _diagnostics("runtime_apply_exception", operation="runtime_apply") for _ in range(2)
    )

    class Repository:
        secondary_diagnostics = repository_causes
        secondary_omitted_count = 0

        async def save(self, request):
            return settings_repository.SettingsCommitResult(
                True, settings_repository.SettingsSnapshot({}, "settings-r2"), None, None
            )

    runtime = RecordingRuntimeApply(
        messages.RuntimeApplyResult(
            messages.RUNTIME_APPLY_STATUS_FAILED,
            None,
            _diagnostics("runtime_apply_exception", operation="runtime_apply"),
            secondary_diagnostics=runtime_causes,
        )
    )
    module = _service_module()
    result = await _service(
        repository=Repository(),
        runtime=runtime,
        snapshot_publisher=RaisingSettingsSnapshotPublisher(),
        runtime_result_publisher=RaisingRuntimeResultPublisher(),
    ).mutate(module.SettingsMutationRequest({}, "settings-r1", "user_patch", "corr-omission"))
    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED
    assert len(result.secondary_diagnostics) == 4
    assert result.secondary_diagnostics[:3] == repository_causes
    assert result.secondary_diagnostics[3].code == "settings_snapshot_notification_failed"
    assert result.secondary_omitted_count == 3


def test_runtime_result_bounds_secondary_causes_without_silent_loss():
    causes = tuple(
        _diagnostics("runtime_apply_exception", operation="runtime_apply") for _ in range(6)
    )
    result = messages.RuntimeApplyResult(messages.RUNTIME_APPLY_STATUS_FAILED, None, None, causes)
    assert result.secondary_diagnostics == causes[:4]
    assert result.secondary_omitted_count == 2
