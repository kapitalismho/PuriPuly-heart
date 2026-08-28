from __future__ import annotations

import json

import httpx
import pytest

from puripuly_heart.config.settings import AppSettings, with_telemetry_enabled
from puripuly_heart.core.telemetry import (
    AppActiveDayTelemetryService,
    HttpAppActiveDayTelemetryClient,
)


class FakeTelemetryClient:
    def __init__(self, *, result: bool = True, exc: Exception | None = None) -> None:
        self.result = result
        self.exc = exc
        self.calls: list[tuple[str, str, str]] = []

    async def record_app_active_day(
        self,
        identifier: str,
        active_date_utc: str,
        *,
        base_url: str,
    ) -> bool:
        self.calls.append((identifier, active_date_utc, base_url))
        if self.exc is not None:
            raise self.exc
        return self.result


class PersistRecorder:
    def __init__(self, *, result: bool = True) -> None:
        self.result = result
        self.calls: list[AppSettings] = []

    async def __call__(self, settings: AppSettings) -> bool:
        self.calls.append(settings)
        return self.result


def _enabled_settings(identifier: str = "anon-id") -> AppSettings:
    settings = AppSettings()
    settings.telemetry_state.anonymous_id = identifier
    return with_telemetry_enabled(settings, True)


@pytest.mark.asyncio
async def test_http_client_posts_only_anonymous_id_and_utc_date() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/telemetry/app-active-day"
        assert json.loads(request.content) == {
            "anonymous_id": "anon-telemetry-id-123456",
            "active_date_utc": "2026-08-28",
        }
        return httpx.Response(200, json={"ok": True})

    client = HttpAppActiveDayTelemetryClient(transport=httpx.MockTransport(handler))

    result = await client.record_app_active_day(
        "anon-telemetry-id-123456",
        "2026-08-28",
        base_url="https://broker.example.test",
    )

    assert result is True


@pytest.mark.asyncio
async def test_disabled_reporting_skips_without_client_call() -> None:
    settings = with_telemetry_enabled(AppSettings(), False)
    client = FakeTelemetryClient()
    persist = PersistRecorder()
    events: list[tuple[str, dict[str, object]]] = []
    service = AppActiveDayTelemetryService(
        client,
        diagnostics_sink=lambda event, metadata: events.append((event, dict(metadata))),
    )

    result = await service.record_app_active_day(
        settings, active_date_utc="2026-08-28", persist_sent_date=persist
    )

    assert result.status == "skipped_disabled"
    assert result.attempted_send is False
    assert client.calls == []
    assert persist.calls == []
    assert events == [("skipped_disabled", {"reason": "reporting_disabled"})]


@pytest.mark.asyncio
async def test_enabled_missing_identifier_skips_without_client_call() -> None:
    settings = AppSettings()
    settings.telemetry_state.anonymous_id = None
    client = FakeTelemetryClient()
    persist = PersistRecorder()
    service = AppActiveDayTelemetryService(client)

    result = await service.record_app_active_day(
        settings, active_date_utc="2026-08-28", persist_sent_date=persist
    )

    assert result.status == "skipped_missing_identifier"
    assert client.calls == []
    assert persist.calls == []


@pytest.mark.asyncio
async def test_already_sent_date_skips_without_client_call() -> None:
    settings = _enabled_settings()
    settings.telemetry_state.last_sent_date_utc = "2026-08-28"
    client = FakeTelemetryClient()
    persist = PersistRecorder()
    service = AppActiveDayTelemetryService(client)

    result = await service.record_app_active_day(
        settings, active_date_utc="2026-08-28", persist_sent_date=persist
    )

    assert result.status == "skipped_already_sent"
    assert client.calls == []
    assert persist.calls == []


@pytest.mark.asyncio
async def test_success_sends_once_to_current_broker_and_persists_utc_date() -> None:
    settings = _enabled_settings()
    settings.openrouter.broker_base_url = "https://broker.example.test"
    client = FakeTelemetryClient(result=True)
    persist = PersistRecorder()
    service = AppActiveDayTelemetryService(client)

    result = await service.record_app_active_day(
        settings, active_date_utc="2026-08-28", persist_sent_date=persist
    )

    assert result.status == "sent"
    assert result.attempted_send is True
    assert result.persisted is True
    assert client.calls == [("anon-id", "2026-08-28", "https://broker.example.test")]
    assert len(persist.calls) == 1
    assert persist.calls[0].telemetry_state.last_sent_date_utc == "2026-08-28"
    assert settings.telemetry_state.last_sent_date_utc is None


@pytest.mark.asyncio
async def test_failed_send_does_not_persist_or_mark_date() -> None:
    settings = _enabled_settings()
    client = FakeTelemetryClient(result=False)
    persist = PersistRecorder()
    service = AppActiveDayTelemetryService(client)

    result = await service.record_app_active_day(
        settings, active_date_utc="2026-08-28", persist_sent_date=persist
    )

    assert result.status == "send_failed"
    assert result.attempted_send is True
    assert len(client.calls) == 1
    assert persist.calls == []
    assert settings.telemetry_state.last_sent_date_utc is None


@pytest.mark.asyncio
async def test_client_exception_returns_safe_diagnostics_without_persisting() -> None:
    settings = _enabled_settings()
    client = FakeTelemetryClient(exc=RuntimeError("secret payload should not appear"))
    persist = PersistRecorder()
    service = AppActiveDayTelemetryService(client)

    result = await service.record_app_active_day(
        settings, active_date_utc="2026-08-28", persist_sent_date=persist
    )

    assert result.status == "send_failed"
    assert result.diagnostics == {
        "reason": "client_exception",
        "error_type": "RuntimeError",
    }
    assert "secret payload" not in str(result.diagnostics)
    assert persist.calls == []


@pytest.mark.asyncio
async def test_successful_send_with_failed_persistence_does_not_report_persisted() -> None:
    settings = _enabled_settings()
    client = FakeTelemetryClient(result=True)
    persist = PersistRecorder(result=False)
    service = AppActiveDayTelemetryService(client)

    result = await service.record_app_active_day(
        settings, active_date_utc="2026-08-28", persist_sent_date=persist
    )

    assert result.status == "persist_failed"
    assert result.attempted_send is True
    assert result.persisted is False
    assert persist.calls[0].telemetry_state.last_sent_date_utc == "2026-08-28"
