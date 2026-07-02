from __future__ import annotations

import json
from datetime import datetime, timezone

import httpx
import pytest

from puripuly_heart.config.settings import AppSettings, TranslationConnection
from puripuly_heart.core.telemetry import (
    TELEMETRY_TRANSLATION_SUCCESS_DAY_SIGNAL,
    TelemetryDeliveryClient,
    TranslationSuccessTelemetryService,
)


class FakeTelemetryClient:
    def __init__(self, base_url: str, failures: list[Exception] | None = None) -> None:
        self.base_url = base_url
        self.failures = failures or []
        self.calls: list[dict[str, str]] = []
        self.closed = False

    async def send_translation_success_day(
        self,
        *,
        telemetry_identifier: str,
        active_date_utc: str,
    ) -> None:
        self.calls.append(
            {
                "telemetry_identifier": telemetry_identifier,
                "active_date_utc": active_date_utc,
            }
        )
        if self.failures:
            raise self.failures.pop(0)

    async def close(self) -> None:
        self.closed = True


def _allowed_settings() -> AppSettings:
    settings = AppSettings()
    settings.telemetry.allow()
    settings.openrouter.broker_base_url = "https://broker.example"
    return settings


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "connection",
    [
        TranslationConnection.MANAGED,
        TranslationConnection.MANAGED_CHINA,
        TranslationConnection.OPENROUTER,
        TranslationConnection.OFFICIAL_BYOK,
        TranslationConnection.OLLAMA,
    ],
)
async def test_translation_success_telemetry_is_connection_type_independent(
    connection: TranslationConnection,
) -> None:
    settings = _allowed_settings()
    settings.translation.connection = connection
    client = FakeTelemetryClient(settings.openrouter.broker_base_url)

    async def persist(_settings: AppSettings) -> None:
        return None

    service = TranslationSuccessTelemetryService(
        read_settings=lambda: settings,
        persist_settings=persist,
        client_factory=lambda _base_url: client,
        now=lambda: datetime(2026, 7, 2, tzinfo=timezone.utc),
    )

    assert await service.record_translation_success() is True
    assert client.calls[0]["active_date_utc"] == "2026-07-02"


@pytest.mark.asyncio
async def test_translation_success_telemetry_skips_unknown_and_declined_without_identifier() -> (
    None
):
    unknown = AppSettings()
    declined = AppSettings()
    declined.telemetry.decline()
    clients: list[FakeTelemetryClient] = []

    async def persist(_settings: AppSettings) -> None:
        raise AssertionError("ineligible telemetry state must not persist")

    for settings in (unknown, declined):
        service = TranslationSuccessTelemetryService(
            read_settings=lambda settings=settings: settings,
            persist_settings=persist,
            client_factory=lambda base_url: clients.append(FakeTelemetryClient(base_url))
            or clients[-1],
            now=lambda: datetime(2026, 7, 2, tzinfo=timezone.utc),
        )

        assert await service.record_translation_success() is False
        assert settings.telemetry.identifier is None

    assert clients == []


@pytest.mark.asyncio
async def test_translation_success_telemetry_sends_once_per_utc_date_and_next_day() -> None:
    settings = _allowed_settings()
    client = FakeTelemetryClient(settings.openrouter.broker_base_url)
    saved_dates: list[list[str]] = []
    current_now = datetime(2026, 7, 2, 23, 59, tzinfo=timezone.utc)

    async def persist(settings: AppSettings) -> None:
        saved_dates.append(list(settings.telemetry.sent_utc_dates))

    service = TranslationSuccessTelemetryService(
        read_settings=lambda: settings,
        persist_settings=persist,
        client_factory=lambda _base_url: client,
        now=lambda: current_now,
    )

    assert await service.record_translation_success() is True
    assert await service.record_translation_success() is False

    current_now = datetime(2026, 7, 3, 0, 1, tzinfo=timezone.utc)
    assert await service.record_translation_success() is True

    assert client.calls == [
        {
            "telemetry_identifier": settings.telemetry.identifier,
            "active_date_utc": "2026-07-02",
        },
        {
            "telemetry_identifier": settings.telemetry.identifier,
            "active_date_utc": "2026-07-03",
        },
    ]
    assert saved_dates == [["2026-07-02"], ["2026-07-02", "2026-07-03"]]


@pytest.mark.asyncio
async def test_translation_success_telemetry_delivery_failure_does_not_mark_date() -> None:
    settings = _allowed_settings()
    client = FakeTelemetryClient(
        settings.openrouter.broker_base_url,
        failures=[RuntimeError("broker unavailable")],
    )
    persisted = 0

    async def persist(_settings: AppSettings) -> None:
        nonlocal persisted
        persisted += 1

    service = TranslationSuccessTelemetryService(
        read_settings=lambda: settings,
        persist_settings=persist,
        client_factory=lambda _base_url: client,
        now=lambda: datetime(2026, 7, 2, tzinfo=timezone.utc),
    )

    assert await service.record_translation_success() is False
    assert settings.telemetry.sent_utc_dates == []

    assert await service.record_translation_success() is True
    assert settings.telemetry.sent_utc_dates == ["2026-07-02"]
    assert len(client.calls) == 2
    assert persisted == 1


@pytest.mark.asyncio
async def test_telemetry_delivery_client_payload_minimality() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(204)

    client = TelemetryDeliveryClient(
        "https://broker.example",
        transport=httpx.MockTransport(handler),
    )

    await client.send_translation_success_day(
        telemetry_identifier="telemetry-id-1234567890",
        active_date_utc="2026-07-02",
    )
    await client.close()

    assert requests[0].url == "https://broker.example/v1/telemetry/translation-success-day"
    assert requests[0].method == "POST"
    assert requests[0].headers["content-type"] == "application/json"
    payload = json.loads(requests[0].content.decode("utf-8"))
    assert payload == {
        "signal": TELEMETRY_TRANSLATION_SUCCESS_DAY_SIGNAL,
        "telemetry_identifier": "telemetry-id-1234567890",
        "active_date_utc": "2026-07-02",
    }
