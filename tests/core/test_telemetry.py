from __future__ import annotations

import json

import httpx
import pytest

from puripuly_heart.core.telemetry import (
    AppActiveDayTelemetryService,
    AppActiveDayTelemetryState,
    HttpAppActiveDayTelemetryClient,
    TelemetryPersistenceResult,
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
        self.calls: list[AppActiveDayTelemetryState] = []

    async def __call__(self, settings: AppActiveDayTelemetryState) -> TelemetryPersistenceResult:
        self.calls.append(settings)
        return TelemetryPersistenceResult(self.result)


def _enabled_settings(identifier: str | None = "anon-id") -> AppActiveDayTelemetryState:
    return AppActiveDayTelemetryState(
        enabled=True,
        anonymous_id=identifier,
        last_sent_date_utc=None,
        broker_base_url="https://broker.example.test",
    )


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
    settings = AppActiveDayTelemetryState(
        enabled=False,
        anonymous_id=None,
        last_sent_date_utc=None,
        broker_base_url="https://broker.example.test",
    )
    client = FakeTelemetryClient()
    persist = PersistRecorder()
    events: list[tuple[str, dict[str, object]]] = []
    service = AppActiveDayTelemetryService(
        client,
        diagnostics_sink=lambda event, metadata: events.append((event, dict(metadata))),
    )

    result = await service.record_app_active_day(
        settings,
        active_date_utc="2026-08-28",
        persist_sent_date=persist,
    )

    assert result.status == "skipped_disabled"
    assert result.attempted_send is False
    assert client.calls == []
    assert persist.calls == []
    assert events == [("skipped_disabled", {"reason": "reporting_disabled"})]


@pytest.mark.asyncio
async def test_enabled_missing_identifier_skips_without_client_call() -> None:
    settings = _enabled_settings(None)
    client = FakeTelemetryClient()
    persist = PersistRecorder()
    service = AppActiveDayTelemetryService(client)

    result = await service.record_app_active_day(
        settings,
        active_date_utc="2026-08-28",
        persist_sent_date=persist,
    )

    assert result.status == "skipped_missing_identifier"
    assert client.calls == []
    assert persist.calls == []


@pytest.mark.asyncio
async def test_already_sent_date_skips_without_client_call() -> None:
    settings = AppActiveDayTelemetryState(
        enabled=True,
        anonymous_id="anon-id",
        last_sent_date_utc="2026-08-28",
        broker_base_url="https://broker.example.test",
    )
    client = FakeTelemetryClient()
    persist = PersistRecorder()
    service = AppActiveDayTelemetryService(client)

    result = await service.record_app_active_day(
        settings,
        active_date_utc="2026-08-28",
        persist_sent_date=persist,
    )

    assert result.status == "skipped_already_sent"
    assert client.calls == []
    assert persist.calls == []


@pytest.mark.asyncio
async def test_success_sends_once_to_current_broker_and_persists_utc_date() -> None:
    settings = _enabled_settings()
    client = FakeTelemetryClient(result=True)
    persist = PersistRecorder()
    service = AppActiveDayTelemetryService(client)

    result = await service.record_app_active_day(
        settings,
        active_date_utc="2026-08-28",
        persist_sent_date=persist,
    )

    assert result.status == "sent"
    assert result.attempted_send is True
    assert result.persisted is True
    assert client.calls == [("anon-id", "2026-08-28", "https://broker.example.test")]
    assert len(persist.calls) == 1
    assert persist.calls[0].last_sent_date_utc == "2026-08-28"
    assert settings.last_sent_date_utc is None


@pytest.mark.asyncio
async def test_failed_send_does_not_persist_or_mark_date() -> None:
    settings = _enabled_settings()
    client = FakeTelemetryClient(result=False)
    persist = PersistRecorder()
    service = AppActiveDayTelemetryService(client)

    result = await service.record_app_active_day(
        settings,
        active_date_utc="2026-08-28",
        persist_sent_date=persist,
    )

    assert result.status == "send_failed"
    assert result.attempted_send is True
    assert len(client.calls) == 1
    assert persist.calls == []
    assert settings.last_sent_date_utc is None


@pytest.mark.asyncio
async def test_client_exception_returns_safe_diagnostics_without_persisting() -> None:
    settings = _enabled_settings()
    client = FakeTelemetryClient(exc=RuntimeError("secret payload should not appear"))
    persist = PersistRecorder()
    service = AppActiveDayTelemetryService(client)

    result = await service.record_app_active_day(
        settings,
        active_date_utc="2026-08-28",
        persist_sent_date=persist,
    )

    assert result.status == "send_failed"
    assert result.diagnostics.code == "telemetry_send_failed"
    assert result.diagnostics.fields["exception_class"] == "RuntimeError"
    assert "secret payload" not in str(result.diagnostics)
    assert persist.calls == []


@pytest.mark.asyncio
async def test_successful_send_with_failed_persistence_does_not_report_persisted() -> None:
    settings = _enabled_settings()
    client = FakeTelemetryClient(result=True)
    persist = PersistRecorder(result=False)
    service = AppActiveDayTelemetryService(client)

    result = await service.record_app_active_day(
        settings,
        active_date_utc="2026-08-28",
        persist_sent_date=persist,
    )

    assert result.status == "persist_failed"
    assert result.attempted_send is True
    assert result.persisted is False
    assert persist.calls[0].last_sent_date_utc == "2026-08-28"


@pytest.mark.asyncio
async def test_transport_failure_survives_throwing_diagnostics_sink() -> None:
    failure = httpx.ConnectError("synthetic-secret-url")
    failure.connection_diagnostics = {
        "transport": "https",
        "proxy_source": "env",
        "stage": "request_send",
        "dispatch_certainty": "possible",
    }

    def fail_sink(*_args):
        raise OSError(5, "synthetic-private-path")

    result = await AppActiveDayTelemetryService(
        FakeTelemetryClient(exc=failure), diagnostics_sink=fail_sink
    ).record_app_active_day(
        _enabled_settings(), active_date_utc="2026-08-28", persist_sent_date=PersistRecorder()
    )

    assert result.status == "send_failed"
    assert result.diagnostics.category == "network"
    assert result.diagnostics.fields["proxy_source"] == "env"
    assert result.diagnostics.fields["dispatch_certainty"] == "possible"
    assert result.secondary_diagnostics[0].code == "telemetry_notification_failed"
    assert "synthetic" not in str(result)


@pytest.mark.asyncio
async def test_sent_telemetry_retains_failed_persistence_cause() -> None:
    async def persist(_settings):
        raise PermissionError(13, "synthetic-private-path")

    result = await AppActiveDayTelemetryService(FakeTelemetryClient()).record_app_active_day(
        _enabled_settings(), active_date_utc="2026-08-28", persist_sent_date=persist
    )

    assert result.status == "persist_failed"
    assert result.attempted_send and not result.persisted
    assert result.diagnostics.code == "telemetry_persist_failed"
    assert result.diagnostics.fields["errno"] == 13
    assert "synthetic" not in str(result)


@pytest.mark.asyncio
async def test_malformed_success_retains_received_status_without_body() -> None:
    client = HttpAppActiveDayTelemetryClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, text="synthetic-secret"))
    )
    result = await AppActiveDayTelemetryService(client).record_app_active_day(
        _enabled_settings(), active_date_utc="2026-08-28", persist_sent_date=PersistRecorder()
    )
    assert result.status == "send_failed"
    assert result.diagnostics.causes[0].status_code == 200
    assert result.diagnostics.causes[0].fields["response_shape"] == "non_json"
    assert "synthetic-secret" not in str(result)
