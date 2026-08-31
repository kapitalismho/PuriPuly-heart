from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from typing import Protocol
from uuid import UUID

import httpx

from puripuly_heart.core.error_messages import format_error_report_for_log, provider_failure_report
from puripuly_heart.core.observability import ProviderObservationPort
from puripuly_heart.domain.models import Translation
from puripuly_heart.providers.llm.messages import build_translation_user_message

logger = logging.getLogger(__name__)
_QWEN_PROBE_MODEL = "qwen3.8-flash"

def _normalize_qwen_model(value: object) -> str:
    if not isinstance(value, str):
        return ""
    normalized = value.strip().lower()
    return "qwen3.8-flash" if normalized == "qwen3.5-plus" else normalized


def _log_basic_request(
    *,
    runtime_logging: ProviderObservationPort | None,
    operation: str,
    text: str,
    source_language: str,
    target_language: str,
    context: str,
) -> None:
    message = "[Basic][LLM] Qwen request [%s][context=%s] %s -> %s: %r" % (
        operation,
        "yes" if context else "no",
        source_language,
        target_language,
        text,
    )
    if runtime_logging is not None:
        runtime_logging.emit_basic(message)
        return
    logger.info(message)


def _log_basic_response(
    *, runtime_logging: ProviderObservationPort | None, operation: str, text: str
) -> None:
    message = "[Basic][LLM] Qwen response [%s]: %r" % (operation, text)
    if runtime_logging is not None:
        runtime_logging.emit_basic(message)
        return
    logger.info(message)


def _log_basic_request_failure(
    *,
    runtime_logging: ProviderObservationPort | None,
    operation: str,
    status: int,
    message: str,
) -> None:
    report = provider_failure_report(
        RuntimeError(f"status={status} message={message}"),
        provider="qwen",
        operation=operation,
    )
    rendered = "[Basic][LLM] Qwen request failed [%s]: %s" % (
        operation,
        format_error_report_for_log(report),
    )
    if runtime_logging is not None:
        runtime_logging.emit_basic(rendered, level=logging.ERROR)
        return
    logger.error(rendered)


def _build_system_prompt(
    *,
    system_prompt: str,
    source_language: str,
    target_language: str,
) -> str:
    formatted = (
        system_prompt.format(
            source_language=source_language,
            target_language=target_language,
        )
        if "{source_language}" in system_prompt
        else system_prompt
    )
    return formatted


def _build_user_message(*, text: str, context: str) -> str:
    return build_translation_user_message(text=text, context=context)


def _extract_message_content(content: object) -> str:
    if isinstance(content, str):
        result = content.strip()
        if result:
            return result
        raise RuntimeError("DashScope response contained empty message content")

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        if parts:
            return "\n".join(parts)

    raise RuntimeError("DashScope response did not contain message content")


def _extract_error_message(data: object) -> str:
    if not isinstance(data, dict):
        return ""
    message = data.get("message")
    if isinstance(message, str) and message.strip():
        return message.strip()
    error = data.get("error")
    if isinstance(error, dict):
        nested = error.get("message")
        if isinstance(nested, str) and nested.strip():
            return nested.strip()
    if isinstance(error, str) and error.strip():
        return error.strip()
    return ""


class AsyncQwenClient(Protocol):
    async def translate(
        self,
        *,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
    ) -> str: ...

    async def close(self) -> None: ...


@dataclass(slots=True)
class AsyncQwenLLMProvider:
    """httpx 기반 비동기 Qwen 클라이언트 (저지연 모드용)

    DashScope OpenAI 호환 API를 사용하여 즉시 취소 가능한 번역을 제공합니다.
    """

    api_key: str
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "qwen3.8-flash"
    timeout: float = 30.0
    runtime_logging: ProviderObservationPort | None = None
    client: AsyncQwenClient | None = None
    _internal_client: AsyncQwenClient | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self.model = _normalize_qwen_model(self.model)

    def _get_client(self) -> AsyncQwenClient:
        if self.client is not None:
            return self.client
        if self._internal_client is None:
            self._internal_client = HttpxQwenClient(
                api_key=self.api_key,
                model=self.model,
                base_url=self.base_url,
                timeout=self.timeout,
                runtime_logging=self.runtime_logging,
            )
        return self._internal_client

    async def translate(
        self,
        *,
        utterance_id: UUID,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
    ) -> Translation:
        client = self._get_client()
        translated = await client.translate(
            text=text,
            system_prompt=system_prompt,
            source_language=source_language,
            target_language=target_language,
            context=context,
        )
        return Translation(utterance_id=utterance_id, text=translated)

    async def close(self) -> None:
        if self._internal_client is not None:
            await self._internal_client.close()
            self._internal_client = None

    async def warmup(self) -> None:
        # Warmup probes the default model.
        await self.verify_api_key(self.api_key, base_url=self.base_url, model=_QWEN_PROBE_MODEL)

    @staticmethod
    async def verify_api_key(
        api_key: str,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model: str = _QWEN_PROBE_MODEL,
    ) -> bool:
        if not api_key:
            return False
        model = _normalize_qwen_model(model)
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": "ping"}],
                        "enable_thinking": False,
                        "max_tokens": 1,
                    },
                )
                return response.status_code == 200
        except Exception:
            return False


@dataclass(slots=True)
class HttpxQwenClient:
    """httpx를 사용하는 DashScope OpenAI 호환 클라이언트"""

    api_key: str
    model: str
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    timeout: float = 30.0
    runtime_logging: ProviderObservationPort | None = None
    _client: httpx.AsyncClient | None = field(init=False, default=None, repr=False)
    _client_lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock, repr=False)

    @staticmethod
    def _normalize_language_code(code: str) -> str:
        if not code:
            return "auto"
        normalized = code.lower()
        if normalized in {"auto"}:
            return "auto"
        if normalized in {"zh-cn", "zh-hans", "zh"}:
            return "zh"
        if normalized in {"zh-tw", "zh-hant", "zh_tw"}:
            return "zh_tw"
        return normalized.split("-")[0]

    async def _get_http_client(self) -> httpx.AsyncClient:
        if self._client is not None:
            return self._client

        async with self._client_lock:
            if self._client is None:
                self._client = httpx.AsyncClient(timeout=self.timeout)
            return self._client

    def _build_request_body(
        self,
        *,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str,
    ) -> dict[str, object]:
        system_content = _build_system_prompt(
            system_prompt=system_prompt,
            source_language=source_language,
            target_language=target_language,
        )
        user_message = _build_user_message(text=text, context=context)

        request_body: dict[str, object] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_message},
            ],
            "enable_thinking": False,
            "temperature": 0.6,
        }
        return request_body

    async def translate(
        self,
        *,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
    ) -> str:
        _log_basic_request(
            runtime_logging=self.runtime_logging,
            operation="translate",
            text=text,
            source_language=source_language,
            target_language=target_language,
            context=context,
        )

        request_body = self._build_request_body(
            text=text,
            system_prompt=system_prompt,
            source_language=source_language,
            target_language=target_language,
            context=context,
        )

        client = await self._get_http_client()
        response = await client.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=request_body,
        )
        if response.status_code != 200:
            error_message = ""
            with contextlib.suppress(Exception):
                error_message = _extract_error_message(response.json())
            if not error_message:
                with contextlib.suppress(Exception):
                    error_message = response.text[:200]
            _log_basic_request_failure(
                runtime_logging=self.runtime_logging,
                operation="translate",
                status=response.status_code,
                message=error_message or "unknown error",
            )
            raise RuntimeError(
                "DashScope compatible-mode request failed "
                f"(status={response.status_code}, message={error_message or 'unknown error'})"
            )

        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("DashScope response did not contain choices")

        message = choices[0].get("message", {})
        result = _extract_message_content(message.get("content"))
        _log_basic_response(
            runtime_logging=self.runtime_logging,
            operation="translate",
            text=result,
        )
        return result

    async def close(self) -> None:
        async with self._client_lock:
            client = self._client
            self._client = None
        if client is not None:
            await client.aclose()
