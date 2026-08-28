from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Protocol


@dataclass(frozen=True, slots=True)
class TranslationSuccessTelemetryState:
    consent: str
    anonymous_id: str | None
    sent_translation_success_dates_utc: tuple[str, ...] = ()

    def with_sent_date(self, active_date_utc: str) -> TranslationSuccessTelemetryState:
        return TranslationSuccessTelemetryState(
            consent=self.consent,
            anonymous_id=self.anonymous_id,
            sent_translation_success_dates_utc=tuple(
                sorted({*self.sent_translation_success_dates_utc, active_date_utc})
            ),
        )


class TranslationSuccessTelemetryClientPort(Protocol):
    async def record_translation_success_day(
        self,
        identifier: str,
        active_date_utc: str,
    ) -> bool: ...


TelemetryPersistSentDate = Callable[[TranslationSuccessTelemetryState], Awaitable[bool]]
TelemetryDiagnosticsSink = Callable[[str, Mapping[str, object]], None]


@dataclass(frozen=True, slots=True)
class TranslationSuccessTelemetryResult:
    status: str
    attempted_send: bool = False
    persisted: bool = False
    active_date_utc: str | None = None
    diagnostics: Mapping[str, object] | None = None


class NoopTranslationSuccessTelemetryClient:
    async def record_translation_success_day(
        self,
        identifier: str,
        active_date_utc: str,
    ) -> bool:
        _ = (identifier, active_date_utc)
        return False


class TranslationSuccessTelemetryService:
    def __init__(
        self,
        client: TranslationSuccessTelemetryClientPort | None,
        *,
        utc_date_provider: Callable[[], date] | None = None,
        diagnostics_sink: TelemetryDiagnosticsSink | None = None,
    ) -> None:
        self._client = client
        self._utc_date_provider = utc_date_provider or _current_utc_date
        self._diagnostics_sink = diagnostics_sink

    async def record_translation_success_day(
        self,
        settings: TranslationSuccessTelemetryState,
        *,
        persist_sent_date: TelemetryPersistSentDate,
    ) -> TranslationSuccessTelemetryResult:
        active_date_utc = self._active_date_utc()
        consent = settings.consent
        identifier = settings.anonymous_id
        sent_dates = set(settings.sent_translation_success_dates_utc)

        if consent == "decline":
            return self._result(
                "skipped_consent",
                active_date_utc=active_date_utc,
                reason="consent_declined",
            )
        if consent not in {"allow", "unknown"}:
            return self._result(
                "skipped_consent",
                active_date_utc=active_date_utc,
                reason="consent_not_allow",
            )
        if not identifier:
            return self._result(
                "skipped_missing_identifier",
                active_date_utc=active_date_utc,
                reason="missing_identifier",
            )
        if active_date_utc in sent_dates:
            return self._result(
                "skipped_already_sent",
                active_date_utc=active_date_utc,
                reason="already_sent",
            )
        if self._client is None:
            return self._result(
                "skipped_no_client",
                active_date_utc=active_date_utc,
                reason="no_client",
            )

        try:
            sent = await self._client.record_translation_success_day(identifier, active_date_utc)
        except Exception as exc:
            return self._result(
                "send_failed",
                attempted_send=True,
                active_date_utc=active_date_utc,
                reason="client_exception",
                error_type=type(exc).__name__,
            )
        if not sent:
            return self._result(
                "send_failed",
                attempted_send=True,
                active_date_utc=active_date_utc,
                reason="client_returned_false",
            )

        persisted = await persist_sent_date(settings.with_sent_date(active_date_utc))
        return self._result(
            "sent" if persisted else "persist_failed",
            attempted_send=True,
            persisted=persisted,
            active_date_utc=active_date_utc,
        )

    def _active_date_utc(self) -> str:
        return self._utc_date_provider().strftime("%Y-%m-%d")

    def _result(
        self,
        status: str,
        *,
        attempted_send: bool = False,
        persisted: bool = False,
        active_date_utc: str | None = None,
        **diagnostics: object,
    ) -> TranslationSuccessTelemetryResult:
        safe_diagnostics = {
            key: value
            for key, value in diagnostics.items()
            if value is None or isinstance(value, str | int | float | bool)
        }
        if self._diagnostics_sink is not None:
            try:
                self._diagnostics_sink(status, safe_diagnostics)
            except Exception:
                pass
        return TranslationSuccessTelemetryResult(
            status=status,
            attempted_send=attempted_send,
            persisted=persisted,
            active_date_utc=active_date_utc,
            diagnostics=safe_diagnostics,
        )


def _current_utc_date() -> date:
    return datetime.now(timezone.utc).date()


__all__ = [
    "NoopTranslationSuccessTelemetryClient",
    "TranslationSuccessTelemetryClientPort",
    "TranslationSuccessTelemetryResult",
    "TranslationSuccessTelemetryService",
    "TranslationSuccessTelemetryState",
]
