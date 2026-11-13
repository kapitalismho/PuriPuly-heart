from __future__ import annotations

from types import SimpleNamespace

from puripuly_heart.app.adapters.ui_runtime import UiEngagementRuntimeAdapter
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.core.telemetry import AppActiveDayTelemetryService


class RecordingTelemetryClient:
    def __init__(self) -> None:
        self.requests: list[tuple[str, str, str]] = []

    async def record_app_active_day(
        self,
        identifier: str,
        active_date_utc: str,
        *,
        base_url: str,
    ) -> bool:
        self.requests.append((identifier, active_date_utc, base_url))
        return True


class CommittedResults:
    current = None

    def committed(self) -> bool:
        return True


class RecordingSettingsApplication:
    def __init__(self, settings: SimpleNamespace) -> None:
        self.settings = settings
        self.applied: list[AppSettingsVNext] = []
        self.results = CommittedResults()

    async def apply(self, next_settings: AppSettingsVNext) -> bool:
        self.applied.append(next_settings)
        self.settings.canonical = next_settings
        return True


async def test_active_day_success_persists_date_in_canonical_telemetry_state() -> None:
    initial = AppSettingsVNext()
    settings = SimpleNamespace(canonical=initial)
    settings_application = RecordingSettingsApplication(settings)
    telemetry_client = RecordingTelemetryClient()
    adapter = UiEngagementRuntimeAdapter(
        settings=settings,
        settings_application=settings_application,
        github_prompt=None,
        telemetry=AppActiveDayTelemetryService(telemetry_client),
        after_launch=None,
    )

    result = await adapter.record_app_active_day("2026-09-03")

    assert result.status == "sent"
    assert result.persisted is True
    assert len(telemetry_client.requests) == 1
    assert len(settings_application.applied) == 1
    assert initial.state.telemetry.last_sent_date_utc is None
    assert settings.canonical.state.telemetry.last_sent_date_utc == "2026-09-03"


async def test_active_day_persistence_receipt_preserves_original_diagnostics():
    from puripuly_heart.core.diagnostics import exception_diagnostics
    from puripuly_heart.core.messages import (
        TRANSACTION_STATUS_SETTINGS_COMMIT_FAILED,
        TransactionResult,
    )

    initial = AppSettingsVNext()
    settings = SimpleNamespace(canonical=initial)
    diagnostics = exception_diagnostics(
        PermissionError(13, "synthetic-private-path"),
        component="settings_repository",
        operation="save",
        code="settings_save_failed",
        category="transaction",
    )

    async def fail_apply(_settings):
        return False

    application = SimpleNamespace(
        apply=fail_apply,
        results=SimpleNamespace(
            committed=lambda: False,
            current=TransactionResult(
                status=TRANSACTION_STATUS_SETTINGS_COMMIT_FAILED,
                message=None,
                diagnostics=diagnostics,
            ),
        ),
    )
    adapter = UiEngagementRuntimeAdapter(
        settings=settings,
        settings_application=application,
        github_prompt=None,
        telemetry=AppActiveDayTelemetryService(RecordingTelemetryClient()),
        after_launch=None,
    )

    result = await adapter.record_app_active_day("2026-09-03")

    assert result.status == "persist_failed"
    assert result.diagnostics is diagnostics
    assert settings.canonical.state.telemetry.last_sent_date_utc is None
    assert "synthetic-private" not in str(result)
