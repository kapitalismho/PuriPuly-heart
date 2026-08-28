from __future__ import annotations

from collections.abc import Callable, Mapping

from puripuly_heart.app.ports.runtime_apply import RuntimeApplyRequest
from puripuly_heart.app.services.canonical_settings_persistence import SettingsOwner
from puripuly_heart.app.services.github_star_prompt import GithubStarPromptOwner
from puripuly_heart.app.services.settings_mutation import (
    SettingsMutationService,
    UiPromptClipboardStateSettingsMutation,
)
from puripuly_heart.app.services.settings_mutation_legacy import (
    build_ui_prompt_clipboard_state_settings_path_patch,
    settings_path_mutation_validator_for_command,
)
from puripuly_heart.config.translation_values import TranslationConnection
from puripuly_heart.core.messages import (
    RUNTIME_APPLY_STATUS_APPLIED,
    TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
    TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED,
    RuntimeApplyResult,
    TransactionResult,
)

GITHUB_STAR_PROMPT_MANAGED_REMAINING_PERCENT_THRESHOLD = 60
_MANAGED_CONNECTIONS = frozenset(
    {
        TranslationConnection.MANAGED,
        TranslationConnection.MANAGED_CHINA,
    }
)
_USER_OWNED_CLOUD_CONNECTIONS = frozenset(
    {
        TranslationConnection.OPENROUTER,
        TranslationConnection.OFFICIAL_BYOK,
    }
)

GithubStarPromptRemainingPercentProvider = Callable[[], int | None]
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
    managed_remaining_percent: GithubStarPromptRemainingPercentProvider,
    transaction_result_sink: GithubStarPromptTransactionResultSink,
    save_failure_sink: GithubStarPromptSaveFailureSink,
    runtime_diagnostics_sink: GithubStarPromptRuntimeDiagnosticsSink,
    mutation_service_provider: GithubStarPromptSettingsMutationServiceProvider,
) -> GithubStarPromptOwner:
    def report_settings_save_failure(_message: str) -> None:
        try:
            runtime_diagnostics_sink(
                "github_star_prompt_settings_persistence_failed",
                {
                    "component": "settings_repository",
                    "operation": "save",
                    "code": "settings_save_failed",
                    "surface": "ui_prompt_clipboard_state",
                },
            )
        except Exception:
            return

    async def persist_state(base: object, committed: object) -> bool:
        patch_values = build_ui_prompt_clipboard_state_settings_path_patch(base, committed)
        if not patch_values:
            return True
        repository = settings.create_legacy_patch_repository(
            base_settings=base,
            committed_settings=committed,
            surface="ui_prompt_clipboard_state",
            save_failure_sink=report_settings_save_failure,
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
        settings.complete()
        transaction_result_sink(transaction_result)
        return transaction_result.status in {
            TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
            TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED,
        }

    def connection_for(value: object | None) -> TranslationConnection | None:
        if value is None:
            return None
        translation = getattr(value, "translation", None)
        raw_connection = getattr(translation, "connection", None)
        if raw_connection is None:
            intent = getattr(value, "intent", None)
            raw_connection = getattr(
                getattr(intent, "translation", None),
                "connection",
                None,
            )
        if isinstance(raw_connection, TranslationConnection):
            return raw_connection
        try:
            return TranslationConnection(raw_connection)
        except (TypeError, ValueError):
            return None

    def has_user_owned_cloud_connection(value: object | None) -> bool:
        return connection_for(value) in _USER_OWNED_CLOUD_CONNECTIONS

    def is_eligible() -> bool:
        current = settings.current
        connection = connection_for(current)
        if connection in _MANAGED_CONNECTIONS:
            remaining_percent = managed_remaining_percent()
            return (
                remaining_percent is not None
                and remaining_percent <= GITHUB_STAR_PROMPT_MANAGED_REMAINING_PERCENT_THRESHOLD
            )
        if connection in _USER_OWNED_CLOUD_CONNECTIONS and current is not None:
            ui = getattr(current, "ui", None)
            return bool(getattr(ui, "github_star_prompt_translation_success_observed", False))
        return False

    return GithubStarPromptOwner(
        settings_provider=lambda: settings.current,
        persist_settings_state=persist_state,
        is_eligible=is_eligible,
        has_user_owned_cloud_connection=has_user_owned_cloud_connection,
        log_save_failure=save_failure_sink,
        runtime_diagnostics_sink=runtime_diagnostics_sink,
    )


__all__ = [
    "GITHUB_STAR_PROMPT_MANAGED_REMAINING_PERCENT_THRESHOLD",
    "GithubStarPromptRemainingPercentProvider",
    "GithubStarPromptRuntimeDiagnosticsSink",
    "GithubStarPromptSaveFailureSink",
    "GithubStarPromptSettingsMutationServiceProvider",
    "GithubStarPromptTransactionResultSink",
    "compose_github_star_prompt_owner",
]
