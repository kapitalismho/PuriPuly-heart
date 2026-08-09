from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Protocol
from uuid import UUID

import httpx

from puripuly_heart.config.llm_profiles import OPENROUTER_MODEL_DEEPSEEK_V4_FLASH
from puripuly_heart.config.settings import OpenRouterProviderRouting, OpenRouterRoutingMode
from puripuly_heart.core.error_messages import format_error_report_for_log, provider_failure_report
from puripuly_heart.core.openrouter_credentials import normalize_managed_openrouter_user_identifier
from puripuly_heart.core.openrouter_metadata import OpenRouterKeyMetadata
from puripuly_heart.core.runtime_logging import SessionRuntimeLoggingService
from puripuly_heart.domain.models import Translation
from puripuly_heart.providers.llm.messages import build_translation_user_message

logger = logging.getLogger(__name__)
_OPENROUTER_KEY_URL = "https://openrouter.ai/api/v1/key"


def _log_basic_request(
    *,
    runtime_logging: SessionRuntimeLoggingService | None,
    operation: str,
    text: str,
    source_language: str,
    target_language: str,
    context: str,
) -> None:
    message = "[Basic][LLM] OpenRouter request [%s][context=%s] %s -> %s: %r" % (
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
    *, runtime_logging: SessionRuntimeLoggingService | None, operation: str, text: str
) -> None:
    message = "[Basic][LLM] OpenRouter response [%s]: %r" % (operation, text)
    if runtime_logging is not None:
        runtime_logging.emit_basic(message)
        return
    logger.info(message)


def _log_basic_request_failure(
    *,
    runtime_logging: SessionRuntimeLoggingService | None,
    operation: str,
    status: int,
    message: str,
) -> None:
    report = provider_failure_report(
        RuntimeError(f"status={status} message={message}"),
        provider="openrouter",
        operation=operation,
    )
    rendered = "[Basic][LLM] OpenRouter request failed [%s]: %s" % (
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
        raise RuntimeError("OpenRouter response contained empty message content")

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        if parts:
            return "\n".join(parts)

    raise RuntimeError("OpenRouter response did not contain message content")


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


def _build_provider_preferences(
    provider_routing: OpenRouterProviderRouting = OpenRouterProviderRouting.DEFAULT,
    *,
    model: str | None = None,
    models: tuple[str, ...] = (),
) -> dict[str, object]:
    if provider_routing == OpenRouterProviderRouting.GEMMA4_26B_31B_LATENCY:
        return {
            "only": ["cloudflare", "coreweave/bf16", "friendli"],
            "sort": {"by": "latency", "partition": "none"},
            "allow_fallbacks": True,
        }
    if provider_routing == OpenRouterProviderRouting.GEMMA4_31B_LATENCY:
        return {
            "only": ["coreweave/bf16", "friendli"],
            "sort": {"by": "latency"},
            "allow_fallbacks": True,
        }
    if provider_routing == OpenRouterProviderRouting.GEMMA4_26B_LATENCY:
        return {
            "only": ["cloudflare", "parasail/bf16"],
            "sort": {"by": "latency"},
            "allow_fallbacks": True,
        }
    if provider_routing == OpenRouterProviderRouting.GEMMA4_31B_CEREBRAS_ONLY:
        return {
            "only": ["cerebras/fp16"],
            "allow_fallbacks": False,
        }
    if provider_routing == OpenRouterProviderRouting.DEEPSEEK_ONLY:
        return {
            "only": ["baidu/fp8", "deepseek/fp8", "siliconflow/fp8"],
            "sort": {"by": "latency"},
            "allow_fallbacks": True,
        }
    if provider_routing == OpenRouterProviderRouting.DEEPSEEK_V4_FLASH_LATENCY:
        return {
            "only": [
                "coreweave/fp8",
                "baidu/fp8",
                "deepseek/fp8",
                "cloudflare/fp8",
            ],
            "sort": {"by": "latency"},
            "allow_fallbacks": True,
        }
    if provider_routing == OpenRouterProviderRouting.GOOGLE_GEMINI_LATENCY:
        return {
            "sort": "latency",
            "only": ["google-vertex", "google-ai-studio"],
            "allow_fallbacks": True,
            "data_collection": "deny",
        }
    if model == "google/gemma-4-26b-a4b-it" and len(models) <= 1:
        return {
            "order": ["wafer", "cloudflare", "deepinfra"],
            "only": ["wafer", "cloudflare", "deepinfra"],
            "allow_fallbacks": True,
        }
    if model == OPENROUTER_MODEL_DEEPSEEK_V4_FLASH:
        return {
            "only": [
                "coreweave/fp8",
                "baidu/fp8",
                "deepseek/fp8",
                "cloudflare/fp8",
            ],
            "sort": {"by": "latency"},
            "allow_fallbacks": True,
        }
    return {
        "sort": "latency",
        "allow_fallbacks": True,
        "ignore": ["venice", "deepinfra", "google-vertex"],
    }


class OpenRouterClient(Protocol):
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


def _optional_number(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


@dataclass(slots=True)
class OpenRouterLLMProvider:
    api_key: str
    user_identifier: str | None = None
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "google/gemma-4-26b-a4b-it"
    models: tuple[str, ...] = ()
    routing_mode: OpenRouterRoutingMode = OpenRouterRoutingMode.LATENCY
    provider_routing: OpenRouterProviderRouting = OpenRouterProviderRouting.DEFAULT
    max_tokens: int = 100
    timeout: float = 30.0
    runtime_logging: SessionRuntimeLoggingService | None = None
    client: OpenRouterClient | None = None
    _internal_client: OpenRouterClient | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        models = tuple(self.models) if self.models else (self.model,)
        if not models or models[0] != self.model or any(not model for model in models):
            raise ValueError(
                "models must start with the selected model and contain no empty values"
            )
        self.models = models

    def _get_client(self) -> OpenRouterClient:
        if self.client is not None:
            return self.client
        if self._internal_client is None:
            self._internal_client = HttpxOpenRouterClient(
                api_key=self.api_key,
                user_identifier=self.user_identifier,
                model=self.model,
                models=self.models,
                base_url=self.base_url,
                routing_mode=self.routing_mode,
                provider_routing=self.provider_routing,
                max_tokens=self.max_tokens,
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

    @staticmethod
    async def verify_api_key(api_key: str) -> bool:
        if not api_key:
            return False
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    _OPENROUTER_KEY_URL,
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                return response.status_code == 200
        except Exception:
            return False

    @staticmethod
    async def fetch_key_metadata(api_key: str) -> OpenRouterKeyMetadata | None:
        if not api_key:
            return None
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    _OPENROUTER_KEY_URL,
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                response.raise_for_status()
                payload = response.json()
        except Exception:
            return None

        if not isinstance(payload, dict):
            return None
        data = payload.get("data")
        if not isinstance(data, dict):
            return None
        return OpenRouterKeyMetadata(
            limit_usd=_optional_number(data.get("limit")),
            remaining_usd=_optional_number(data.get("limit_remaining")),
            usage_usd=_optional_number(data.get("usage")),
        )


@dataclass(slots=True)
class HttpxOpenRouterClient:
    api_key: str
    model: str
    models: tuple[str, ...] = ()
    user_identifier: str | None = None
    base_url: str = "https://openrouter.ai/api/v1"
    routing_mode: OpenRouterRoutingMode = OpenRouterRoutingMode.LATENCY
    provider_routing: OpenRouterProviderRouting = OpenRouterProviderRouting.DEFAULT
    max_tokens: int = 100
    timeout: float = 30.0
    runtime_logging: SessionRuntimeLoggingService | None = None
    _client: httpx.AsyncClient | None = field(init=False, default=None, repr=False)
    _client_lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock, repr=False)

    def __post_init__(self) -> None:
        models = tuple(self.models) if self.models else (self.model,)
        if not models or models[0] != self.model or any(not model for model in models):
            raise ValueError(
                "models must start with the selected model and contain no empty values"
            )
        self.models = models

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
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_message},
            ],
            "reasoning": {"effort": "none"},
            "provider": _build_provider_preferences(
                self.provider_routing,
                model=self.model,
                models=self.models,
            ),
            "max_tokens": self.max_tokens,
        }
        if len(self.models) == 1:
            request_body["model"] = self.models[0]
        else:
            request_body["models"] = list(self.models)
        user_identifier = normalize_managed_openrouter_user_identifier(self.user_identifier)
        if user_identifier is not None:
            request_body["user"] = user_identifier
        return request_body

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
            _log_basic_request_failure(
                runtime_logging=self.runtime_logging,
                operation="translate",
                status=response.status_code,
                message="provider returned non-success status",
            )
            raise RuntimeError(f"OpenRouter request failed (status={response.status_code})")

        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("OpenRouter response did not contain choices")
        if _has_length_finish_reason(data):
            raise RuntimeError("OpenRouter response was truncated by max_tokens limit")

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
