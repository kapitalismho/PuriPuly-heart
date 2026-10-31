from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
from websockets.exceptions import ConnectionClosed, InvalidStatus

from puripuly_heart.core import network_clients
from puripuly_heart.core.chatgpt.oauth import ChatGptAuthError
from puripuly_heart.core.chatgpt.session import ChatGptAccessTokenPort
from puripuly_heart.core.photo.image import PreparedPhoto
from puripuly_heart.core.photo.prompt import render_photo_prompt
from puripuly_heart.core.photo.protocol import (
    PhotoProviderConnection,
    PhotoProviderError,
    PhotoRequestSent,
    PhotoStreamCompleted,
    PhotoStreamEvent,
    PhotoTextDelta,
)

logger = logging.getLogger(__name__)


def _status_error(status: int) -> PhotoProviderError:
    if status in (401, 403):
        return PhotoProviderError("photo_authentication_failed", retryable=False)
    if status == 429:
        return PhotoProviderError("photo_quota_exhausted", quota=True)
    if status >= 500 or status in (408, 409):
        return PhotoProviderError("photo_provider_unavailable")
    return PhotoProviderError("photo_request_rejected", retryable=False)


def _event_error(error: Any, *, incomplete: bool = False) -> PhotoProviderError:
    if not isinstance(error, dict):
        return PhotoProviderError(
            "photo_stream_incomplete" if incomplete else "photo_stream_invalid"
        )
    code = error.get("code")
    if code in (
        "subscription_sharing_usage_limit_exceeded",
        "insufficient_quota",
        "billing_hard_limit_reached",
    ):
        return PhotoProviderError("photo_usage_limit", retryable=False)
    if code in ("invalid_api_key", "invalid_access_token", "token_expired", "authentication_error"):
        return PhotoProviderError("photo_authentication_failed", retryable=False)
    status = error.get("status")
    if status in ("UNAUTHENTICATED", "PERMISSION_DENIED"):
        return PhotoProviderError("photo_authentication_failed", retryable=False)
    if (
        code == 429
        or status == "RESOURCE_EXHAUSTED"
        or code in ("rate_limit_exceeded", "resource_exhausted")
    ):
        return PhotoProviderError("photo_quota_exhausted", quota=True)
    if type(code) is int:
        return _status_error(code)
    if code in ("content_filter", "content_policy_violation", "safety"):
        return PhotoProviderError("photo_content_rejected", retryable=False)
    return PhotoProviderError(
        "photo_stream_incomplete" if incomplete else "photo_provider_unavailable"
    )


async def _http_error(response: httpx.Response) -> PhotoProviderError:
    default = _status_error(response.status_code)
    try:
        payload = json.loads(await response.aread())
    except ValueError, UnicodeDecodeError:
        return default
    error = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(error, dict):
        return default
    for detail in error.get("details") or []:
        if isinstance(detail, dict) and detail.get("reason") in (
            "API_KEY_INVALID",
            "API_KEY_EXPIRED",
        ):
            return PhotoProviderError("photo_authentication_failed", retryable=False)
    if error.get("status") in ("UNAUTHENTICATED", "PERMISSION_DENIED", "RESOURCE_EXHAUSTED"):
        return _event_error(error)
    return default


def _classified(error: Exception) -> PhotoProviderError:
    if isinstance(error, PhotoProviderError):
        return error
    if isinstance(error, ChatGptAuthError):
        return PhotoProviderError("photo_authentication_failed", retryable=False)
    if isinstance(error, InvalidStatus):
        return _status_error(error.response.status_code)
    if isinstance(error, httpx.HTTPStatusError):
        return _status_error(error.response.status_code)
    if isinstance(error, (TimeoutError, httpx.TimeoutException)):
        return PhotoProviderError("photo_timeout")
    if isinstance(error, (ConnectionClosed, httpx.HTTPError, OSError)):
        return PhotoProviderError("photo_transport_failed")
    return PhotoProviderError("photo_stream_invalid", retryable=False)


def _report(error: PhotoProviderError) -> None:
    logger.warning(
        "photo_provider_failure code=%s retryable=%s quota=%s",
        error.code,
        error.retryable,
        error.quota,
    )


async def _settle(task: asyncio.Task[Any]) -> None:
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
        except BaseException:
            break
    with contextlib.suppress(BaseException):
        task.result()


class _PhotoConnection(ABC):
    google: bool
    model: str

    def __init__(self) -> None:
        self._closed = False
        self._producer: asyncio.Task[None] | None = None
        self._cleanup: asyncio.Task[None] | None = None
        self._queue: asyncio.Queue[PhotoStreamEvent | PhotoProviderError | None] | None = None
        self._preopen_lock = asyncio.Lock()
        self.connect_ms = 0.0

    @abstractmethod
    def _exchange(
        self, image: PreparedPhoto, target_language: str
    ) -> AsyncIterator[PhotoStreamEvent]: ...

    @abstractmethod
    async def _close_transport(self) -> None: ...

    @abstractmethod
    async def preopen(self) -> None: ...

    async def stream(
        self, image: PreparedPhoto, target_language: str
    ) -> AsyncIterator[PhotoStreamEvent]:
        if self._closed:
            raise PhotoProviderError("photo_connection_closed", retryable=False)
        if self._producer is not None:
            raise PhotoProviderError("photo_request_in_progress", retryable=False)
        queue: asyncio.Queue[PhotoStreamEvent | PhotoProviderError | None] = asyncio.Queue(32)
        self._queue = queue

        async def produce() -> None:
            try:
                async for event in self._exchange(image, target_language):
                    await queue.put(event)
            except Exception as exc:
                error = _classified(exc)
                _report(error)
                await queue.put(error)
            await queue.put(None)

        producer = asyncio.create_task(produce(), name="photo-provider-stream")
        self._producer = producer
        primary: BaseException | None = None
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                if isinstance(event, PhotoProviderError):
                    raise event from None
                yield event
        except BaseException as exc:
            primary = exc
            raise
        finally:
            try:
                await self.close()
            except Exception:
                if primary is None:
                    raise

    def _sent(self) -> None:
        if self._queue is None:
            raise RuntimeError("photo request has no owner")
        self._queue.put_nowait(PhotoRequestSent(time.monotonic()))

    async def close(self) -> None:
        self._closed = True
        producer = self._producer
        if producer is not None:
            producer.cancel()
            await _settle(producer)
            self._producer = None
        if self._cleanup is None:

            async def close_transport() -> None:
                async with self._preopen_lock:
                    await self._close_transport()

            self._cleanup = asyncio.create_task(close_transport(), name="photo-provider-close")
        await _settle(self._cleanup)
        try:
            self._cleanup.result()
        except Exception as exc:
            error = _classified(exc)
            _report(error)
            raise error from None
        if self._queue is not None:
            with contextlib.suppress(asyncio.QueueFull):
                self._queue.put_nowait(None)


class LunaPhotoConnection(_PhotoConnection):
    google = False

    def __init__(
        self,
        session: ChatGptAccessTokenPort,
        *,
        model: str = "gpt-6-luna",
        endpoint: str = "wss://api.openai.com/v1/responses",
        timeout: float = 240.0,
    ) -> None:
        super().__init__()
        self.model = model
        self._session = session
        self._endpoint = endpoint
        self._timeout = timeout
        self._socket: Any = None
        self._token: str | None = None

    async def preopen(self) -> None:
        async with self._preopen_lock:
            if self._closed:
                raise PhotoProviderError("photo_connection_closed", retryable=False)
            if self._socket is not None:
                return
            try:
                started = time.monotonic()
                token = await self._session.access_token()
                self._token = token
                socket = await network_clients.external_websocket_connect(
                    self._endpoint,
                    additional_headers={"Authorization": f"Bearer {token}"},
                    max_size=4 * 1024 * 1024,
                    open_timeout=15,
                )
                if self._closed:
                    await socket.close()
                    raise PhotoProviderError("photo_connection_closed", retryable=False)
                self._socket = socket
                self.connect_ms = (time.monotonic() - started) * 1000
            except Exception as exc:
                error = _classified(exc)
                if error.code == "photo_authentication_failed" and self._token is not None:
                    self._session.invalidate_access_token(self._token)
                _report(error)
                raise error from None

    async def _exchange(
        self, image: PreparedPhoto, target_language: str
    ) -> AsyncIterator[PhotoStreamEvent]:
        await self.preopen()
        prompt = render_photo_prompt(target_language)
        payload = json.dumps(
            {
                "type": "response.create",
                "model": self.model,
                "store": False,
                "reasoning": {"effort": "none"},
                "instructions": prompt.instructions,
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{image.jpeg_base64}",
                            },
                            {"type": "input_text", "text": prompt.user_message},
                        ],
                    }
                ],
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "blocks",
                        "schema": prompt.schema,
                        "strict": True,
                    }
                },
            }
        )
        async with asyncio.timeout(self._timeout):
            self._sent()
            await self._socket.send(payload)
            while True:
                event = json.loads(await self._socket.recv())
                if not isinstance(event, dict):
                    raise PhotoProviderError("photo_stream_invalid", retryable=False)
                kind = event.get("type")
                if kind == "response.output_text.delta":
                    delta = event.get("delta")
                    if not isinstance(delta, str):
                        raise PhotoProviderError("photo_stream_invalid", retryable=False)
                    if delta:
                        yield PhotoTextDelta(delta)
                elif kind == "response.completed":
                    response = event.get("response") or {}
                    if response.get("status") not in (None, "completed"):
                        raise PhotoProviderError("photo_stream_incomplete")
                    for output in response.get("output") or []:
                        for part in output.get("content") or []:
                            if part.get("type") == "refusal":
                                raise PhotoProviderError("photo_content_rejected", retryable=False)
                    yield PhotoStreamCompleted(
                        usage=response.get("usage") or {}, model=response.get("model") or self.model
                    )
                    return
                elif kind in ("response.failed", "response.incomplete", "error"):
                    response = event.get("response") or {}
                    error = _event_error(
                        event.get("error") or response.get("error"),
                        incomplete=kind == "response.incomplete",
                    )
                    if error.code == "photo_authentication_failed" and self._token is not None:
                        self._session.invalidate_access_token(self._token)
                    raise error
                elif kind in ("response.refusal.delta", "response.refusal.done"):
                    raise PhotoProviderError("photo_content_rejected", retryable=False)

    async def _close_transport(self) -> None:
        if self._socket is not None:
            await self._socket.close()
            self._socket = None
        self._token = None


class GooglePhotoConnection(_PhotoConnection):
    google = True

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "gemini-3.5-flash-lite",
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/models",
        timeout: float = 240.0,
        http_factory: Callable[[], httpx.AsyncClient] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._http_factory = http_factory or (
            lambda: network_clients.external_async_client(
                timeout=httpx.Timeout(timeout, connect=20),
                limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
            )
        )
        self._http: httpx.AsyncClient | None = None
        self._preopened = False

    async def preopen(self) -> None:
        async with self._preopen_lock:
            if self._closed:
                raise PhotoProviderError("photo_connection_closed", retryable=False)
            if self._preopened:
                return
            try:
                started = time.monotonic()
                if self._http is None:
                    self._http = self._http_factory()
                response = await self._http.get(
                    f"{self._base_url}/{self.model}",
                    headers={"x-goog-api-key": self._api_key},
                )
                if response.status_code != 200:
                    raise await _http_error(response)
                metadata = response.json()
                if metadata.get("name") != f"models/{self.model}":
                    raise PhotoProviderError("photo_request_rejected", retryable=False)
                if self._closed:
                    raise PhotoProviderError("photo_connection_closed", retryable=False)
                self._preopened = True
                self.connect_ms = (time.monotonic() - started) * 1000
            except Exception as exc:
                error = _classified(exc)
                _report(error)
                raise error from None

    async def _exchange(
        self, image: PreparedPhoto, target_language: str
    ) -> AsyncIterator[PhotoStreamEvent]:
        await self.preopen()
        prompt = render_photo_prompt(target_language, google=True)
        body = {
            "systemInstruction": {"parts": [{"text": prompt.instructions}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"inline_data": {"mime_type": "image/jpeg", "data": image.jpeg_base64}},
                        {"text": prompt.user_message},
                    ],
                }
            ],
            "generationConfig": {
                "maxOutputTokens": 8192,
                "thinkingConfig": {"thinkingLevel": "MINIMAL"},
                "mediaResolution": "MEDIA_RESOLUTION_HIGH",
                "responseMimeType": "application/json",
                "responseJsonSchema": prompt.schema,
            },
        }
        sent = False

        async def trace(name: str, info: dict[str, Any]) -> None:
            nonlocal sent
            if name.endswith("send_request_headers.started") and not sent:
                sent = True
                self._sent()

        usage: dict[str, Any] = {}
        returned_model = self.model
        finished = False
        async with asyncio.timeout(self._timeout):
            assert self._http is not None
            async with self._http.stream(
                "POST",
                f"{self._base_url}/{self.model}:streamGenerateContent?alt=sse",
                headers={"x-goog-api-key": self._api_key},
                json=body,
                extensions={"trace": trace},
            ) as response:
                if response.status_code != 200:
                    raise await _http_error(response)
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if not raw or raw == "[DONE]":
                        continue
                    chunk = json.loads(raw)
                    if not isinstance(chunk, dict):
                        raise PhotoProviderError("photo_stream_invalid", retryable=False)
                    if chunk.get("error"):
                        raise _event_error(chunk["error"])
                    feedback = chunk.get("promptFeedback") or {}
                    if feedback.get("blockReason"):
                        raise PhotoProviderError("photo_content_rejected", retryable=False)
                    usage = chunk.get("usageMetadata") or usage
                    returned_model = chunk.get("modelVersion") or returned_model
                    candidates = chunk.get("candidates") or []
                    if len(candidates) > 1:
                        raise PhotoProviderError("photo_stream_invalid", retryable=False)
                    for candidate in candidates:
                        if any(
                            rating.get("blocked") for rating in candidate.get("safetyRatings") or []
                        ):
                            raise PhotoProviderError("photo_content_rejected", retryable=False)
                        reason = candidate.get("finishReason")
                        if reason and reason != "STOP":
                            if reason in (
                                "SAFETY",
                                "RECITATION",
                                "BLOCKLIST",
                                "PROHIBITED_CONTENT",
                                "SPII",
                                "IMAGE_SAFETY",
                            ):
                                raise PhotoProviderError("photo_content_rejected", retryable=False)
                            raise PhotoProviderError("photo_stream_incomplete")
                        for part in (candidate.get("content") or {}).get("parts") or []:
                            if part.get("thought") is True:
                                continue
                            text = part.get("text")
                            if text is not None:
                                if not isinstance(text, str) or finished:
                                    raise PhotoProviderError(
                                        "photo_stream_invalid", retryable=False
                                    )
                                if text:
                                    yield PhotoTextDelta(text)
                        if reason == "STOP":
                            finished = True
        if not finished:
            raise PhotoProviderError("photo_stream_incomplete")
        yield PhotoStreamCompleted(usage=usage, model=returned_model)

    async def _close_transport(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None


def make_photo_connection(
    *,
    route: str,
    chatgpt_session: ChatGptAccessTokenPort | None = None,
    gemini_api_key: str | None = None,
    model: str | None = None,
) -> PhotoProviderConnection:
    if route == "luna" and chatgpt_session is not None and model in (None, "gpt-6-luna"):
        return LunaPhotoConnection(chatgpt_session)
    if (
        route == "google"
        and gemini_api_key
        and model in (None, "gemini-3.5-flash-lite", "gemini-3.1-flash-lite")
    ):
        return GooglePhotoConnection(gemini_api_key, model=model or "gemini-3.5-flash-lite")
    raise PhotoProviderError("photo_request_rejected", retryable=False)
