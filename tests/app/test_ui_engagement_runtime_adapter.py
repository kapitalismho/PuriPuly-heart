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
    def committed(self) -> bool:
        return True


class RecordingSettingsApplication:
    def __init__(self, settings: SimpleNamespace) -> None:
        self.settings = settings
        self.applied: list[AppSettingsVNext] = []
        self.results = CommittedResults()

    async def apply(self, next_settings: AppSettingsVNext, **_options: object) -> bool:
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


def _engagement_adapter(
    settings: SimpleNamespace,
    settings_application: RecordingSettingsApplication,
) -> UiEngagementRuntimeAdapter:
    return UiEngagementRuntimeAdapter(
        settings=settings,
        settings_application=settings_application,
        github_prompt=None,
        telemetry=None,
        after_launch=None,
    )


async def test_main_input_first_launch_is_claimed_once_and_persisted() -> None:
    settings = SimpleNamespace(canonical=AppSettingsVNext())
    settings_application = RecordingSettingsApplication(settings)
    adapter = _engagement_adapter(settings, settings_application)

    assert await adapter.claim_main_input_first_launch() is True
    assert settings.canonical.state.main_input_hint.first_launch_completed is True

    assert await adapter.claim_main_input_first_launch() is False
    assert len(settings_application.applied) == 1


async def test_main_input_launch_without_settings_is_treated_as_first_launch() -> None:
    settings = SimpleNamespace(canonical=None)
    settings_application = RecordingSettingsApplication(settings)
    adapter = _engagement_adapter(settings, settings_application)

    assert await adapter.claim_main_input_first_launch() is True
    assert settings_application.applied == []
