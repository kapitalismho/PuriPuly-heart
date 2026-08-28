from __future__ import annotations

import copy
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlsplit

import httpx

from puripuly_heart.config.settings import AppSettings


class AppActiveDayTelemetryClientPort(Protocol):
    async def record_app_active_day(
        self,
        identifier: str,
        active_date_utc: str,
        *,
        base_url: str,
    ) -> bool: ...


TelemetryPersistSentDate = Callable[[AppSettings], Awaitable[bool]]
TelemetryDiagnosticsSink = Callable[[str, Mapping[str, object]], None]


@dataclass(frozen=True, slots=True)
class AppActiveDayTelemetryResult:
    status: str
    attempted_send: bool = False
    persisted: bool = False
    active_date_utc: str | None = None
    diagnostics: Mapping[str, object] | None = None


@dataclass(slots=True)
class HttpAppActiveDayTelemetryClient:
    timeout: float = 10.0
    transport: httpx.AsyncBaseTransport | None = None

    async def record_app_active_day(
        self,
        identifier: str,
        active_date_utc: str,
        *,
        base_url: str,
    ) -> bool:
        normalized_base_url = _normalize_base_url(base_url)
        async with httpx.AsyncClient(
            base_url=normalized_base_url,
            timeout=self.timeout,
            transport=self.transport,
        ) as client:
            response = await client.post(
                "/v1/telemetry/app-active-day",
                json={
                    "anonymous_id": identifier,
                    "active_date_utc": active_date_utc,
                },
            )
        response.raise_for_status()
        payload = response.json()
        return isinstance(payload, Mapping) and payload.get("ok") is True


class AppActiveDayTelemetryService:
    def __init__(
        self,
        client: AppActiveDayTelemetryClientPort,
        *,
        diagnostics_sink: TelemetryDiagnosticsSink | None = None,
    ) -> None:
        self._client = client
        self._diagnostics_sink = diagnostics_sink

    async def record_app_active_day(
        self,
        settings: AppSettings,
        *,
        active_date_utc: str,
        persist_sent_date: TelemetryPersistSentDate,
    ) -> AppActiveDayTelemetryResult:
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

        try:
            sent = await self._client.record_app_active_day(
                identifier,
                active_date_utc,
                base_url=settings.openrouter.broker_base_url,
            )
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
    ) -> AppActiveDayTelemetryResult:
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
        return AppActiveDayTelemetryResult(
            status=status,
            attempted_send=attempted_send,
            persisted=persisted,
            active_date_utc=active_date_utc,
            diagnostics=safe_diagnostics,
        )


def _normalize_base_url(base_url: str) -> str:
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("telemetry base_url must be a non-empty string")
    normalized = base_url.strip().rstrip("/")
    parsed = urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("telemetry base_url must be an absolute HTTP URL")
    if parsed.path not in {"", "/"}:
        raise ValueError("telemetry base_url must not include a path prefix")
    return normalized


__all__ = [
    "AppActiveDayTelemetryClientPort",
    "AppActiveDayTelemetryResult",
    "AppActiveDayTelemetryService",
    "HttpAppActiveDayTelemetryClient",
]
