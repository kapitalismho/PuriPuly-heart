from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from typing import Protocol
from urllib.parse import urlsplit

import httpx

from puripuly_heart.core import network_clients
from puripuly_heart.core.diagnostics import exception_diagnostics
from puripuly_heart.core.messages import ErrorDiagnostics


@dataclass(frozen=True, slots=True)
class AppActiveDayTelemetryState:
    enabled: bool
    anonymous_id: str | None
    last_sent_date_utc: str | None
    broker_base_url: str

    def with_sent_date(self, active_date_utc: str) -> AppActiveDayTelemetryState:
        return replace(self, last_sent_date_utc=active_date_utc)


class AppActiveDayTelemetryClientPort(Protocol):
    async def record_app_active_day(
        self,
        identifier: str,
        active_date_utc: str,
        *,
        base_url: str,
    ) -> bool: ...


@dataclass(frozen=True, slots=True)
class TelemetryPersistenceResult:
    persisted: bool
    diagnostics: ErrorDiagnostics | None = None
    secondary_diagnostics: tuple[ErrorDiagnostics, ...] = ()
    secondary_omitted_count: int = 0


TelemetryPersistSentDate = Callable[
    [AppActiveDayTelemetryState], Awaitable[TelemetryPersistenceResult]
]
TelemetryDiagnosticsSink = Callable[[str, Mapping[str, object]], None]


@dataclass(frozen=True, slots=True)
class AppActiveDayTelemetryResult:
    status: str
    attempted_send: bool = False
    persisted: bool = False
    active_date_utc: str | None = None
    diagnostics: ErrorDiagnostics | None = None
    secondary_diagnostics: tuple[ErrorDiagnostics, ...] = ()
    secondary_omitted_count: int = 0


class TelemetryRequestError(RuntimeError):
    def __init__(self, diagnostics: ErrorDiagnostics) -> None:
        self.diagnostics = diagnostics
        super().__init__(diagnostics.code)


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
        async with network_clients.external_async_client(
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
        try:
            payload = response.json()
        except ValueError as exc:
            cause = exception_diagnostics(
                exc,
                component="telemetry",
                operation="telemetry_send",
                code="telemetry_response_rejected",
                category="invalid_response",
                fields={"response_shape": "non_json", "dispatch_certainty": "response_received"},
            )
            raise TelemetryRequestError(
                replace(cause, status_code=response.status_code)
            ) from exc
        if not isinstance(payload, Mapping) or payload.get("ok") is not True:
            raise TelemetryRequestError(
                ErrorDiagnostics(
                    component="telemetry",
                    operation="telemetry_send",
                    code="telemetry_response_rejected",
                    category="invalid_response",
                    visibility="basic",
                    content_policy="metadata_only",
                    status_code=response.status_code,
                    retry_after_ms=None,
                    fields={
                        "response_shape": "malformed_success" if isinstance(payload, Mapping) else "non_object",
                        "dispatch_certainty": "response_received",
                    },
                )
            )
        return True


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
        settings: AppActiveDayTelemetryState,
        *,
        active_date_utc: str,
        persist_sent_date: TelemetryPersistSentDate,
    ) -> AppActiveDayTelemetryResult:
        if not settings.enabled:
            return self._result(
                "skipped_disabled",
                active_date_utc=active_date_utc,
                reason="reporting_disabled",
            )
        if not settings.anonymous_id:
            return self._result(
                "skipped_missing_identifier",
                active_date_utc=active_date_utc,
                reason="missing_identifier",
            )
        if active_date_utc == settings.last_sent_date_utc:
            return self._result(
                "skipped_already_sent",
                active_date_utc=active_date_utc,
                reason="already_sent",
            )

        try:
            sent = await self._client.record_app_active_day(
                settings.anonymous_id,
                active_date_utc,
                base_url=settings.broker_base_url,
            )
        except Exception as exc:
            return self._result(
                "send_failed",
                attempted_send=True,
                active_date_utc=active_date_utc,
                diagnostics=exception_diagnostics(
                    exc,
                    component="telemetry",
                    operation="telemetry_send",
                    code="telemetry_send_failed",
                ),
            )
        if not sent:
            return self._result(
                "send_failed",
                attempted_send=True,
                active_date_utc=active_date_utc,
                diagnostics=_telemetry_failure("telemetry_send", "telemetry_response_rejected"),
            )

        try:
            persistence = await persist_sent_date(settings.with_sent_date(active_date_utc))
        except Exception as exc:
            persistence = TelemetryPersistenceResult(
                False,
                exception_diagnostics(
                    exc,
                    component="telemetry",
                    operation="telemetry_persist",
                    code="telemetry_persist_failed",
                    category="transaction",
                ),
            )
        return self._result(
            "sent" if persistence.persisted else "persist_failed",
            attempted_send=True,
            persisted=persistence.persisted,
            active_date_utc=active_date_utc,
            diagnostics=(
                persistence.diagnostics
                if persistence.persisted or persistence.diagnostics is not None
                else _telemetry_failure("telemetry_persist", "telemetry_persist_failed")
            ),
            secondary_diagnostics=persistence.secondary_diagnostics,
            secondary_omitted_count=persistence.secondary_omitted_count,
        )

    def _result(
        self,
        status: str,
        *,
        attempted_send: bool = False,
        persisted: bool = False,
        active_date_utc: str | None = None,
        diagnostics: ErrorDiagnostics | None = None,
        reason: str | None = None,
        secondary_diagnostics: tuple[ErrorDiagnostics, ...] = (),
        secondary_omitted_count: int = 0,
    ) -> AppActiveDayTelemetryResult:
        metadata = {"diagnostics": diagnostics} if diagnostics is not None else (
            {"reason": reason} if reason is not None else {}
        )
        if self._diagnostics_sink is not None:
            try:
                self._diagnostics_sink(status, metadata)
            except Exception as exc:
                secondary_diagnostics += (
                    exception_diagnostics(
                        exc,
                        component="telemetry",
                        operation="telemetry_notification",
                        code="telemetry_notification_failed",
                        fields={"notification_delivered": False},
                    ),
                )
        excess = max(0, len(secondary_diagnostics) - 4)
        return AppActiveDayTelemetryResult(
            status=status,
            attempted_send=attempted_send,
            persisted=persisted,
            active_date_utc=active_date_utc,
            diagnostics=diagnostics,
            secondary_diagnostics=secondary_diagnostics[:4],
            secondary_omitted_count=secondary_omitted_count + excess,
        )


def _telemetry_failure(operation: str, code: str) -> ErrorDiagnostics:
    return ErrorDiagnostics(
        component="telemetry",
        operation=operation,
        code=code,
        category="transaction" if operation == "telemetry_persist" else "invalid_response",
        visibility="basic",
        content_policy="metadata_only",
        status_code=None,
        retry_after_ms=None,
        fields={},
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
    "AppActiveDayTelemetryState",
    "HttpAppActiveDayTelemetryClient",
    "TelemetryPersistenceResult",
]
