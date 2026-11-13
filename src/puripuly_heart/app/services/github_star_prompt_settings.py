from __future__ import annotations

from collections.abc import Callable, Mapping

from puripuly_heart.app.ports.runtime_apply import RuntimeApplyRequest
from puripuly_heart.app.services.canonical_settings_persistence import SettingsOwner
from puripuly_heart.app.services.github_star_prompt import GithubStarPromptOwner
from puripuly_heart.app.services.settings.settings_mutation import (
    SettingsMutationService,
    UiPromptClipboardStateSettingsMutation,
)
from puripuly_heart.app.services.settings.settings_mutation_legacy import (
    build_ui_prompt_clipboard_state_settings_path_patch,
    settings_path_mutation_validator_for_command,
)
from puripuly_heart.core.messages import (
    RUNTIME_APPLY_STATUS_APPLIED,
    TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
    TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED,
    RuntimeApplyResult,
    TransactionResult,
)

GithubStarPromptTransactionResultSink = Callable[[TransactionResult], None]
GithubStarPromptSaveFailureSink = Callable[[str, Exception], None]
GithubStarPromptRuntimeDiagnosticsSink = Callable[[str, Mapping[str, object]], None]
GithubStarPromptSettingsMutationServiceProvider = Callable[
    [],
    SettingsMutationService | None,
]


class _GithubStarPromptNoopRuntimeApply:
    async def apply_runtime(self, request: RuntimeApplyRequest) -> RuntimeApplyResult:
        _ = request
        return RuntimeApplyResult(
            status=RUNTIME_APPLY_STATUS_APPLIED,
            message=None,
            diagnostics=None,
        )


def compose_github_star_prompt_owner(
    *,
    settings: SettingsOwner,
    transaction_result_sink: GithubStarPromptTransactionResultSink,
    save_failure_sink: GithubStarPromptSaveFailureSink,
    runtime_diagnostics_sink: GithubStarPromptRuntimeDiagnosticsSink,
    mutation_service_provider: GithubStarPromptSettingsMutationServiceProvider,
) -> GithubStarPromptOwner:

    async def persist_state(base: object, committed: object) -> bool:
        patch_values = build_ui_prompt_clipboard_state_settings_path_patch(base, committed)
        if not patch_values:
            return True
        repository = settings.create_canonical_patch_repository(
            base_settings=base,
            committed_settings=committed,
            surface="ui_prompt_clipboard_state",
        )
        command = UiPromptClipboardStateSettingsMutation(values=patch_values)
        service = mutation_service_provider() or SettingsMutationService(
            settings_repository=repository,
            runtime_apply=_GithubStarPromptNoopRuntimeApply(),
            validator=settings_path_mutation_validator_for_command(command),
        )
        transaction_result = await service.mutate(
            command.to_mutation_request(
                expected_revision=None,
                correlation_id=None,
            )
        )
        transaction_result_sink(transaction_result)
        return transaction_result.status in {
            TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
            TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED,
        }

    return GithubStarPromptOwner(
        settings_provider=lambda: settings.canonical,
        persist_settings_state=persist_state,
        log_save_failure=save_failure_sink,
        runtime_diagnostics_sink=runtime_diagnostics_sink,
    )


__all__ = [
    "GithubStarPromptRuntimeDiagnosticsSink",
    "GithubStarPromptSaveFailureSink",
    "GithubStarPromptSettingsMutationServiceProvider",
    "GithubStarPromptTransactionResultSink",
    "compose_github_star_prompt_owner",
]
