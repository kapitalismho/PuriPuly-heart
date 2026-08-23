from __future__ import annotations

import copy
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from puripuly_heart.config.settings import AppSettings


class TranslationSuccessTelemetryClientPort(Protocol):
    async def record_translation_success_day(
        self,
        identifier: str,
        active_date_utc: str,
    ) -> bool: ...


TelemetryPersistSentDate = Callable[[AppSettings], Awaitable[bool]]
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
        diagnostics_sink: TelemetryDiagnosticsSink | None = None,
    ) -> None:
        self._client = client
        self._diagnostics_sink = diagnostics_sink

    async def record_translation_success_day(
        self,
        settings: AppSettings,
        *,
        active_date_utc: str,
        persist_sent_date: TelemetryPersistSentDate,
    ) -> TranslationSuccessTelemetryResult:
        enabled = settings.telemetry.enabled
        identifier = settings.telemetry_state.anonymous_id
        last_sent_date_utc = settings.telemetry_state.last_sent_date_utc

        if not enabled:
            return self._result(
                "skipped_disabled",
                active_date_utc=active_date_utc,
                reason="reporting_disabled",
            )
        if not identifier:
            return self._result(
                "skipped_missing_identifier",
                active_date_utc=active_date_utc,
                reason="missing_identifier",
            )
        if active_date_utc == last_sent_date_utc:
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

        updated = copy.deepcopy(settings)
        updated.telemetry_state.last_sent_date_utc = active_date_utc
        updated.validate()
        persisted = await persist_sent_date(updated)
        return self._result(
            "sent" if persisted else "persist_failed",
            attempted_send=True,
            persisted=persisted,
            active_date_utc=active_date_utc,
        )

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


__all__ = [
    "NoopTranslationSuccessTelemetryClient",
    "TranslationSuccessTelemetryClientPort",
    "TranslationSuccessTelemetryResult",
    "TranslationSuccessTelemetryService",
]
