from __future__ import annotations

from datetime import date

import pytest

from puripuly_heart.core.telemetry import (
    TranslationSuccessTelemetryService,
    TranslationSuccessTelemetryState,
)


class FakeTelemetryClient:
    def __init__(self, *, result: bool = True, exc: Exception | None = None) -> None:
        self.result = result
        self.exc = exc
        self.calls: list[tuple[str, str]] = []

    async def record_translation_success_day(self, identifier: str, active_date_utc: str) -> bool:
        self.calls.append((identifier, active_date_utc))
        if self.exc is not None:
            raise self.exc
        return self.result


class PersistRecorder:
    def __init__(self, *, result: bool = True) -> None:
        self.result = result
        self.calls: list[TranslationSuccessTelemetryState] = []

    async def __call__(self, settings: TranslationSuccessTelemetryState) -> bool:
        self.calls.append(settings)
        return self.result


def _date_provider() -> date:
    return date(2026, 7, 3)


def _allowed_settings(identifier: str = "anon-id") -> TranslationSuccessTelemetryState:
    return TranslationSuccessTelemetryState(
        consent="allow",
        anonymous_id=identifier,
    )


@pytest.mark.asyncio
async def test_declined_consent_skips_without_client_call() -> None:
    settings = TranslationSuccessTelemetryState(consent="decline", anonymous_id=None)
    client = FakeTelemetryClient()
    persist = PersistRecorder()
    events: list[tuple[str, dict[str, object]]] = []
    service = TranslationSuccessTelemetryService(
        client,
        utc_date_provider=_date_provider,
        diagnostics_sink=lambda event, metadata: events.append((event, dict(metadata))),
    )

    result = await service.record_translation_success_day(settings, persist_sent_date=persist)

    assert result.status == "skipped_consent"
    assert result.attempted_send is False
    assert client.calls == []
    assert persist.calls == []
    assert events == [("skipped_consent", {"reason": "consent_declined"})]


@pytest.mark.asyncio
async def test_unknown_consent_is_treated_as_allow_when_identifier_present() -> None:
    settings = TranslationSuccessTelemetryState(
        consent="unknown",
        anonymous_id="anon-unknown",
    )
    client = FakeTelemetryClient()
    persist = PersistRecorder()
    service = TranslationSuccessTelemetryService(client, utc_date_provider=_date_provider)

    result = await service.record_translation_success_day(settings, persist_sent_date=persist)

    assert result.status == "sent"
    assert client.calls == [("anon-unknown", "2026-07-03")]
    assert persist.calls


@pytest.mark.asyncio
async def test_allowed_missing_identifier_skips_without_client_call() -> None:
    settings = TranslationSuccessTelemetryState(consent="allow", anonymous_id=None)
    client = FakeTelemetryClient()
    persist = PersistRecorder()
    service = TranslationSuccessTelemetryService(client, utc_date_provider=_date_provider)

    result = await service.record_translation_success_day(settings, persist_sent_date=persist)

    assert result.status == "skipped_missing_identifier"
    assert client.calls == []
    assert persist.calls == []


@pytest.mark.asyncio
async def test_already_sent_date_skips_without_client_call() -> None:
    settings = _allowed_settings()
    settings = TranslationSuccessTelemetryState(
        consent=settings.consent,
        anonymous_id=settings.anonymous_id,
        sent_translation_success_dates_utc=("2026-07-03",),
    )
    client = FakeTelemetryClient()
    persist = PersistRecorder()
    service = TranslationSuccessTelemetryService(client, utc_date_provider=_date_provider)

    result = await service.record_translation_success_day(settings, persist_sent_date=persist)

    assert result.status == "skipped_already_sent"
    assert client.calls == []
    assert persist.calls == []


@pytest.mark.asyncio
async def test_success_sends_once_and_persists_current_utc_date() -> None:
    settings = _allowed_settings()
    client = FakeTelemetryClient(result=True)
    persist = PersistRecorder()
    service = TranslationSuccessTelemetryService(client, utc_date_provider=_date_provider)

    result = await service.record_translation_success_day(settings, persist_sent_date=persist)

    assert result.status == "sent"
    assert result.attempted_send is True
    assert result.persisted is True
    assert client.calls == [("anon-id", "2026-07-03")]
    assert len(persist.calls) == 1
    assert persist.calls[0].sent_translation_success_dates_utc == ("2026-07-03",)
    assert settings.sent_translation_success_dates_utc == ()


@pytest.mark.asyncio
async def test_failed_send_does_not_persist_or_mark_date() -> None:
    settings = _allowed_settings()
    client = FakeTelemetryClient(result=False)
    persist = PersistRecorder()
    service = TranslationSuccessTelemetryService(client, utc_date_provider=_date_provider)

    result = await service.record_translation_success_day(settings, persist_sent_date=persist)

    assert result.status == "send_failed"
    assert result.attempted_send is True
    assert client.calls == [("anon-id", "2026-07-03")]
    assert persist.calls == []
    assert settings.sent_translation_success_dates_utc == ()


@pytest.mark.asyncio
async def test_client_exception_returns_safe_diagnostics_without_persisting() -> None:
    settings = _allowed_settings()
    client = FakeTelemetryClient(exc=RuntimeError("secret payload should not appear"))
    persist = PersistRecorder()
    service = TranslationSuccessTelemetryService(client, utc_date_provider=_date_provider)

    result = await service.record_translation_success_day(settings, persist_sent_date=persist)

    assert result.status == "send_failed"
    assert result.diagnostics == {
        "reason": "client_exception",
        "error_type": "RuntimeError",
    }
    assert "secret payload" not in str(result.diagnostics)
    assert persist.calls == []


@pytest.mark.asyncio
async def test_successful_send_with_failed_persistence_does_not_report_persisted() -> None:
    settings = _allowed_settings()
    client = FakeTelemetryClient(result=True)
    persist = PersistRecorder(result=False)
    service = TranslationSuccessTelemetryService(client, utc_date_provider=_date_provider)

    result = await service.record_translation_success_day(settings, persist_sent_date=persist)

    assert result.status == "persist_failed"
    assert result.attempted_send is True
    assert result.persisted is False
    assert persist.calls[0].sent_translation_success_dates_utc == ("2026-07-03",)
