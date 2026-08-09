from __future__ import annotations

import asyncio
import contextvars
import logging
import ssl
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx

from puripuly_heart.core.http_extensions.keys import http_extension_secret_key
from puripuly_heart.core.http_extensions.schema import (
    HttpExtension,
    HttpExtensionConfigurationError,
    HttpExtensionResponseError,
    extract_translation_text,
    render_translation_request,
)
from puripuly_heart.core.messages import (
    DIAGNOSTIC_CATEGORY_AUTH,
    DIAGNOSTIC_CATEGORY_INVALID_RESPONSE,
    DIAGNOSTIC_CATEGORY_NETWORK,
    DIAGNOSTIC_CATEGORY_TIMEOUT,
    DiagnosticCategory,
)
from puripuly_heart.core.translation_backend import (
    TranslationBackend,
    TranslationBackendRequest,
    TranslationSecretResolver,
)
from puripuly_heart.domain.models import Translation

_HTTP_LOG_SUPPRESSION = contextvars.ContextVar(
    "puripuly_heart_http_log_suppression",
    default=False,
)


def _is_http_logger_name(name: str) -> bool:
    return (
        name == "httpx"
        or name.startswith("httpx.")
        or name == "httpcore"
        or name.startswith("httpcore.")
    )


class _HttpPrivacyFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not (_HTTP_LOG_SUPPRESSION.get() and _is_http_logger_name(record.name))


_HTTP_PRIVACY_FILTER = _HttpPrivacyFilter()


def _install_http_privacy_filters() -> None:
    logger_objects: list[logging.Logger] = [
        logging.getLogger("httpx"),
        logging.getLogger("httpcore"),
    ]
    for logger_name, logger in logging.Logger.manager.loggerDict.items():
        if _is_http_logger_name(logger_name) and isinstance(logger, logging.Logger):
            logger_objects.append(logger)
    handlers: list[logging.Handler] = list(logging.getLogger().handlers)
    seen_logger_ids: set[int] = set()
    seen_handler_ids: set[int] = {id(handler) for handler in handlers}
    for logger in logger_objects:
        if id(logger) not in seen_logger_ids:
            seen_logger_ids.add(id(logger))
            if not any(item is _HTTP_PRIVACY_FILTER for item in logger.filters):
                logger.addFilter(_HTTP_PRIVACY_FILTER)
        for handler in logger.handlers:
            if id(handler) not in seen_handler_ids:
                seen_handler_ids.add(id(handler))
                handlers.append(handler)
    for handler in handlers:
        if not any(item is _HTTP_PRIVACY_FILTER for item in handler.filters):
            handler.addFilter(_HTTP_PRIVACY_FILTER)


_install_http_privacy_filters()


class HttpExtensionTranslationError(RuntimeError):
    diagnostic_provider = "custom_http"

    def __init__(
        self,
        category: str,
        *,
        status_code: int | None = None,
        diagnostic_category: DiagnosticCategory | None = None,
    ) -> None:
        self.category = category
        self.status_code = status_code
        self.diagnostic_category = diagnostic_category or _diagnostic_category(category)
        detail = f"{category} ({status_code})" if status_code is not None else category
        super().__init__(detail)


def _diagnostic_category(category: str) -> DiagnosticCategory:
    if category == "timeout":
        return DIAGNOSTIC_CATEGORY_TIMEOUT
    if category in {"connect error", "TLS error", "transport error", "HTTP request error"}:
        return DIAGNOSTIC_CATEGORY_NETWORK
    return DIAGNOSTIC_CATEGORY_INVALID_RESPONSE


def _is_tls_connect_error(error: BaseException) -> bool:
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ssl.SSLError):
            return True
        current = current.__cause__ or current.__context__
    return False


@dataclass(slots=True)
class HttpExtensionTranslationBackend(TranslationBackend):
    extension: HttpExtension
    secret_store: TranslationSecretResolver
    timeout: float = 10.0
    concurrency_limit: int = 5
    client_factory: Callable[..., httpx.AsyncClient] = httpx.AsyncClient
    _client: httpx.AsyncClient | None = field(init=False, default=None, repr=False)
    _semaphore: asyncio.Semaphore = field(init=False, repr=False)
    _closed: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        if self.concurrency_limit <= 0:
            raise ValueError("concurrency_limit must be > 0")
        self._semaphore = asyncio.Semaphore(self.concurrency_limit)

    async def translate(self, request: TranslationBackendRequest) -> Translation:
        if self._closed:
            raise HttpExtensionConfigurationError("translation backend is closed")
        secret_values = self._secret_values()
        url, headers, query, body = render_translation_request(
            self.extension,
            request,
            secrets=secret_values,
        )
        async with self._semaphore:
            if self._closed:
                raise HttpExtensionConfigurationError("translation backend is closed")
            client = await self._get_client()
            request_kwargs: dict[str, object] = {
                "headers": headers,
                "params": query,
            }
            body_type = self.extension.request.body.type
            if body_type == "json":
                request_kwargs["json"] = body
            elif body_type == "form":
                request_kwargs["data"] = body
            _install_http_privacy_filters()
            log_token = _HTTP_LOG_SUPPRESSION.set(True)
            transport_error_category: str | None = None
            try:
                response = await client.post(url, **request_kwargs)
            except asyncio.CancelledError:
                raise
            except httpx.TimeoutException:
                transport_error_category = "timeout"
            except httpx.ConnectError as error:
                transport_error_category = (
                    "TLS error" if _is_tls_connect_error(error) else "connect error"
                )
            except httpx.TransportError:
                transport_error_category = "transport error"
            except httpx.HTTPError:
                transport_error_category = "HTTP request error"
            finally:
                _HTTP_LOG_SUPPRESSION.reset(log_token)
            if transport_error_category is not None:
                raise HttpExtensionTranslationError(transport_error_category)
            if not 200 <= response.status_code < 300:
                raise HttpExtensionTranslationError(
                    "HTTP status error",
                    status_code=response.status_code,
                )
            try:
                translated = extract_translation_text(self.extension, response.text)
            except HttpExtensionResponseError:
                raise
            except UnicodeError:
                translated = None
            if translated is None:
                raise HttpExtensionTranslationError("response decoding error")
            return Translation(
                utterance_id=request.utterance_id,
                text=translated,
                source_text=request.text,
                source_language=request.source_language,
                target_language=request.target_language,
            )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        client = self._client
        self._client = None
        if client is not None:
            await client.aclose()

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = self.client_factory(
                timeout=self.timeout,
                follow_redirects=False,
                trust_env=False,
            )
        return self._client

    def _secret_values(self) -> dict[str, str]:
        values: dict[str, str] = {}
        for secret in self.extension.secrets:
            value = self.secret_store.get(http_extension_secret_key(self.extension.id, secret.id))
            if value is None or not value.strip():
                raise HttpExtensionConfigurationError(
                    f"missing required credential: {secret.label}",
                    diagnostic_category=DIAGNOSTIC_CATEGORY_AUTH,
                )
            values[secret.id] = value
        return values
