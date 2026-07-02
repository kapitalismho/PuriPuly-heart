from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol

import httpx

from puripuly_heart.config.settings import AppSettings, TelemetryConsent

logger = logging.getLogger(__name__)

TELEMETRY_TRANSLATION_SUCCESS_DAY_PATH = "/v1/telemetry/translation-success-day"
TELEMETRY_TRANSLATION_SUCCESS_DAY_SIGNAL = "translation_success_day"


class TelemetrySettingsReader(Protocol):
    def __call__(self) -> AppSettings | None: ...


class TelemetrySettingsPersister(Protocol):
    def __call__(self, settings: AppSettings) -> Awaitable[None]: ...


def utc_active_date(value: datetime | None = None) -> str:
    resolved = value or datetime.now(timezone.utc)
    if resolved.tzinfo is None:
        resolved = resolved.replace(tzinfo=timezone.utc)
    return resolved.astimezone(timezone.utc).date().isoformat()


@dataclass(slots=True)
class TelemetryDeliveryClient:
    base_url: str
    timeout: float = 5.0
    transport: httpx.AsyncBaseTransport | None = None
    _client: httpx.AsyncClient | None = field(init=False, default=None, repr=False)
    _client_lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock, repr=False)

    def __post_init__(self) -> None:
        self.base_url = self.base_url.strip().rstrip("/")
        if not self.base_url:
            raise ValueError("telemetry base_url must be a non-empty string")

    async def send_translation_success_day(
        self,
        *,
        telemetry_identifier: str,
        active_date_utc: str,
    ) -> None:
        client = await self._get_http_client()
        response = await client.post(
            TELEMETRY_TRANSLATION_SUCCESS_DAY_PATH,
            json={
                "signal": TELEMETRY_TRANSLATION_SUCCESS_DAY_SIGNAL,
                "telemetry_identifier": telemetry_identifier,
                "active_date_utc": active_date_utc,
            },
        )
        response.raise_for_status()

    async def close(self) -> None:
        async with self._client_lock:
            client = self._client
            self._client = None
        if client is not None:
            await client.aclose()

    async def _get_http_client(self) -> httpx.AsyncClient:
        if self._client is not None:
            return self._client
        async with self._client_lock:
            if self._client is None:
                self._client = httpx.AsyncClient(
                    base_url=self.base_url,
                    timeout=self.timeout,
                    transport=self.transport,
                    follow_redirects=False,
                )
            return self._client


@dataclass(slots=True)
class TranslationSuccessTelemetryService:
    read_settings: TelemetrySettingsReader
    persist_settings: TelemetrySettingsPersister
    client_factory: Callable[[str], TelemetryDeliveryClient] = TelemetryDeliveryClient
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)
    _lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock, repr=False)
    _client: TelemetryDeliveryClient | None = field(init=False, default=None, repr=False)
    _client_base_url: str | None = field(init=False, default=None, repr=False)
    _inflight_dates: set[str] = field(init=False, default_factory=set, repr=False)

    async def record_translation_success(self) -> bool:
        active_date_utc = utc_active_date(self.now())
        async with self._lock:
            settings = self.read_settings()
            if settings is None:
                return False
            telemetry = settings.telemetry
            if telemetry.consent != TelemetryConsent.ALLOW or telemetry.identifier is None:
                return False
            if active_date_utc in telemetry.sent_utc_dates:
                return False
            if active_date_utc in self._inflight_dates:
                return False
            telemetry_identifier = telemetry.identifier
            broker_base_url = settings.openrouter.broker_base_url
            self._inflight_dates.add(active_date_utc)

        try:
            client = await self._client_for(broker_base_url)
            await client.send_translation_success_day(
                telemetry_identifier=telemetry_identifier,
                active_date_utc=active_date_utc,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.info("Translation success-day telemetry delivery skipped: %s", exc)
            return False
        finally:
            async with self._lock:
                self._inflight_dates.discard(active_date_utc)

        async with self._lock:
            settings = self.read_settings()
            if settings is None:
                return True
            telemetry = settings.telemetry
            if (
                telemetry.consent != TelemetryConsent.ALLOW
                or telemetry.identifier != telemetry_identifier
            ):
                return True
            if active_date_utc not in telemetry.sent_utc_dates:
                telemetry.sent_utc_dates.append(active_date_utc)
                telemetry.sent_utc_dates = sorted(dict.fromkeys(telemetry.sent_utc_dates))
                try:
                    await self.persist_settings(settings)
                except Exception as exc:
                    logger.info(
                        "Translation success-day telemetry dedupe persistence skipped: %s", exc
                    )
            return True

    async def close(self) -> None:
        client = self._client
        self._client = None
        self._client_base_url = None
        if client is not None:
            await client.close()

    async def _client_for(self, base_url: str) -> TelemetryDeliveryClient:
        if self._client is not None and self._client_base_url == base_url:
            return self._client
        previous = self._client
        self._client = self.client_factory(base_url)
        self._client_base_url = base_url
        if previous is not None:
            await previous.close()
        return self._client
