from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Protocol
from uuid import UUID

import httpx

from puripuly_heart.core.error_messages import format_error_report_for_log, provider_failure_report
from puripuly_heart.core.observability import ProviderObservationPort
from puripuly_heart.domain.models import Translation
from puripuly_heart.providers.llm.error_details import extract_provider_error_detail
from puripuly_heart.providers.llm.messages import build_translation_user_message

logger = logging.getLogger(__name__)

CEREBRAS_DEFAULT_BASE_URL = "https://api.cerebras.ai/v1"
CEREBRAS_DEFAULT_MODEL = "gemma-4-31b"


def _log_basic_request(
    *,
    runtime_logging: ProviderObservationPort | None,
    operation: str,
    text: str,
    source_language: str,
    target_language: str,
    context: str,
) -> None:
    message = "[Basic][LLM] Cerebras request [%s][context=%s] %s -> %s: text_length=%d" % (
        operation,
        "yes" if context else "no",
        source_language,
        target_language,
        len(text),
    )
    if runtime_logging is not None:
        runtime_logging.emit_basic(message)
        return
    logger.info(message)


def _log_basic_response(
    *, runtime_logging: ProviderObservationPort | None, operation: str, text: str
) -> None:
    message = "[Basic][LLM] Cerebras response [%s]: %r" % (operation, text)
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
        provider="cerebras",
        operation=operation,
    )
    rendered = "[Basic][LLM] Cerebras request failed [%s]: %s" % (
        operation,
        format_error_report_for_log(report),
    )
    rendered = f"{rendered} message={message}"
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
    return (
        system_prompt.format(
            source_language=source_language,
            target_language=target_language,
        )
        if "{source_language}" in system_prompt
        else system_prompt
    )


def _build_user_message(*, text: str, context: str) -> str:
    return build_translation_user_message(text=text, context=context)


def _extract_message_content(content: object) -> str:
    if isinstance(content, str):
        result = content.strip()
        if result:
            return result
        raise RuntimeError("Cerebras response contained empty message content")

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        if parts:
            return "\n".join(parts)

    raise RuntimeError("Cerebras response did not contain message content")


def _has_length_finish_reason(data: object) -> bool:
    if not isinstance(data, dict):
        return False

    choices = data.get("choices")
    if not isinstance(choices, list):
        return False

    for choice in choices:
        if isinstance(choice, dict) and choice.get("finish_reason") == "length":
            return True
    return False


class CerebrasClient(Protocol):
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
class CerebrasLLMProvider:
    api_key: str
    base_url: str = CEREBRAS_DEFAULT_BASE_URL
    model: str = CEREBRAS_DEFAULT_MODEL
    max_completion_tokens: int = 100
    timeout: float = 30.0
    runtime_logging: ProviderObservationPort | None = None
    client: CerebrasClient | None = None
    _internal_client: CerebrasClient | None = field(init=False, default=None, repr=False)

    def _get_client(self) -> CerebrasClient:
        if self.client is not None:
            return self.client
        if self._internal_client is None:
            self._internal_client = HttpxCerebrasClient(
                api_key=self.api_key,
                model=self.model,
                base_url=self.base_url,
                max_completion_tokens=self.max_completion_tokens,
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
        translated = await self._get_client().translate(
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

    @staticmethod
    async def verify_api_key(
        api_key: str,
        base_url: str = CEREBRAS_DEFAULT_BASE_URL,
        model: str = CEREBRAS_DEFAULT_MODEL,
    ) -> bool:
        if not api_key:
            return False
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
                        "reasoning_effort": "none",
                        "max_completion_tokens": 1,
                        "stream": False,
                    },
                )
                return response.status_code == 200
        except Exception:
            return False


@dataclass(slots=True)
class HttpxCerebrasClient:
    api_key: str
    model: str
    base_url: str = CEREBRAS_DEFAULT_BASE_URL
    max_completion_tokens: int = 100
    timeout: float = 30.0
    runtime_logging: ProviderObservationPort | None = None
    _client: httpx.AsyncClient | None = field(init=False, default=None, repr=False)
    _client_lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock, repr=False)

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

        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
            "reasoning_effort": "none",
            "temperature": 0.6,
            "max_completion_tokens": self.max_completion_tokens,
        }

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

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
            headers=self._headers(),
            json=request_body,
        )
        if response.status_code != 200:
            detail = extract_provider_error_detail(response, sensitive_values=(self.api_key,))
            _log_basic_request_failure(
                runtime_logging=self.runtime_logging,
                operation="translate",
                status=response.status_code,
                message=detail,
            )
            raise RuntimeError(
                f"Cerebras request failed (status={response.status_code} message={detail})"
            )

        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("Cerebras response did not contain choices")
        if _has_length_finish_reason(data):
            raise RuntimeError("Cerebras response was truncated by max_completion_tokens limit")

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
