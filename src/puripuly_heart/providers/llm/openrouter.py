from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Protocol
from uuid import UUID

import httpx

from puripuly_heart.config.llm_profiles import (
    OPENROUTER_MODEL_DEEPSEEK_V4_FLASH,
    OPENROUTER_MODEL_DEEPSEEK_V4_FLASH_41,
    OPENROUTER_MODEL_GPT_6_LUNA,
)
from puripuly_heart.core.error_messages import format_error_report_for_log, provider_failure_report
from puripuly_heart.core.llm.latency import current_attempt
from puripuly_heart.core.observability import ProviderObservationPort
from puripuly_heart.core.openrouter.authentication import OpenRouterAuthenticationError
from puripuly_heart.core.openrouter_credentials import normalize_managed_openrouter_user_identifier
from puripuly_heart.core.openrouter_metadata import OpenRouterKeyMetadata
from puripuly_heart.core.openrouter_routing import (
    OpenRouterProviderRouting,
    OpenRouterRoutingMode,
)
from puripuly_heart.domain.models import Translation
from puripuly_heart.providers.llm.messages import build_translation_user_message

logger = logging.getLogger(__name__)
_OPENROUTER_KEY_URL = "https://openrouter.ai/api/v1/key"
_LIMIT_SOURCES = frozenset(
    {"openrouter_credits", "openrouter_key_limit", "openrouter_in_flight_budget"}
)
_LIMIT_REASONS = frozenset({"in_flight_budget_exhausted", "weight_exceeds_budget"})
_MAX_RETRY_AFTER_SECONDS = 86_400


class OpenRouterResponseError(RuntimeError):
    diagnostic_provider = "openrouter"

    def __init__(
        self,
        status_code: int,
        *,
        limit_source: str | None = None,
        limit_reason: str | None = None,
        retry_after_ms: int | None = None,
    ) -> None:
        self.status_code = status_code
        self.limit_source = limit_source
        self.limit_reason = limit_reason
        self.retry_after_ms = retry_after_ms
        super().__init__(f"OpenRouter request failed (status={status_code})")


def _payment_limit_metadata(response: httpx.Response) -> tuple[str | None, str | None]:
    try:
        data = response.json()
    except Exception:
        return None, None
    if not isinstance(data, dict):
        return None, None
    error = data.get("error")
    if not isinstance(error, dict):
        return None, None
    metadata = error.get("metadata")
    if not isinstance(metadata, dict):
        return None, None
    source = metadata.get("limit_source")
    reason = metadata.get("reason")
    if (source is not None and (not isinstance(source, str) or source not in _LIMIT_SOURCES)) or (
        reason is not None and (not isinstance(reason, str) or reason not in _LIMIT_REASONS)
    ):
        return None, None
    return source, reason


def _retry_after_ms(response: httpx.Response) -> int | None:
    raw = response.headers.get("Retry-After")
    if raw is None:
        return None
    value = raw.strip()
    if value.isascii() and value.isdecimal():
        seconds = int(value) if len(value) <= 10 else _MAX_RETRY_AFTER_SECONDS + 1
    else:
        try:
            when = parsedate_to_datetime(value)
        except ValueError, TypeError, IndexError, OverflowError:
            return None
        if when.tzinfo is None:
            return None
        seconds = (when - datetime.now(timezone.utc)).total_seconds()
    if 0 <= seconds <= _MAX_RETRY_AFTER_SECONDS:
        return int(seconds * 1000)
    return None


def _log_basic_request_failure(
    *,
    runtime_logging: ProviderObservationPort | None,
    operation: str,
    error: OpenRouterResponseError,
) -> None:
    report = provider_failure_report(error, provider="openrouter", operation=operation)
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


def _build_user_message(
    *, text: str, context: str, scene_participant_count: int | None = None
) -> str:
    return build_translation_user_message(
        text=text, context=context, scene_participant_count=scene_participant_count
    )


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


def _reasoning_token_count(data: dict[str, object]) -> int | None:
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return None
    details = usage.get("completion_tokens_details")
    token_count = (
        details.get("reasoning_tokens")
        if isinstance(details, dict)
        else usage.get("reasoning_tokens")
    )
    if isinstance(token_count, int) and not isinstance(token_count, bool) and token_count >= 0:
        return token_count
    return None


def _build_provider_preferences(
    provider_routing: OpenRouterProviderRouting = OpenRouterProviderRouting.DEFAULT,
    *,
    model: str | None = None,
    models: tuple[str, ...] = (),
) -> dict[str, object]:
    if (
        model == OPENROUTER_MODEL_DEEPSEEK_V4_FLASH
        and provider_routing == OpenRouterProviderRouting.DEEPSEEK_V4_FLASH_CHINA
    ):
        return {
            "only": ["baidu/fp8"],
            "allow_fallbacks": False,
        }
    if model == OPENROUTER_MODEL_DEEPSEEK_V4_FLASH_41:
        return {
            "only": ["deepseek", "wafer"],
            "allow_fallbacks": False,
        }
    if model == OPENROUTER_MODEL_DEEPSEEK_V4_FLASH:
        return {
            "only": [
                "makora",
                "together",
                "wafer/fast",
                "baidu/fp8",
            ],
            "sort": {"by": "latency", "partition": "none"},
            "allow_fallbacks": True,
        }
    if provider_routing == OpenRouterProviderRouting.GEMMA4_26B_31B_LATENCY:
        return {
            "only": [
                "cloudflare",
                "coreweave/fp4",
                "deepinfra/turbo",
                "dekallm/bf16",
                "nextbit/bf16",
                "makora",
            ],
            "sort": {"by": "latency", "partition": "none"},
            "allow_fallbacks": True,
        }
    if provider_routing == OpenRouterProviderRouting.GEMMA4_31B_MODELRUN_ONLY:
        return {
            "only": ["modelrun/fp4"],
            "allow_fallbacks": False,
        }
    if provider_routing in (
        OpenRouterProviderRouting.DEEPSEEK_ONLY,
        OpenRouterProviderRouting.DEEPSEEK_V4_FLASH_LATENCY,
    ):
        return {
            "only": [
                "makora",
                "together",
                "wafer/fast",
                "baidu/fp8",
            ],
            "sort": {"by": "latency", "partition": "none"},
            "allow_fallbacks": True,
        }
    if provider_routing == OpenRouterProviderRouting.DEEPSEEK_V4_FLASH_41_STRICT:
        return {
            "only": ["deepseek", "wafer"],
            "allow_fallbacks": False,
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
            "sort": {"by": "latency"},
            "only": ["cloudflare", "dekallm/bf16", "nextbit/bf16", "makora"],
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
        scene_participant_count: int | None = None,
        max_output_tokens: int | None = None,
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
    runtime_logging: ProviderObservationPort | None = None
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
        scene_participant_count: int | None = None,
        max_output_tokens: int | None = None,
    ) -> Translation:
        client = self._get_client()
        kwargs = {
            "text": text,
            "system_prompt": system_prompt,
            "source_language": source_language,
            "target_language": target_language,
            "context": context,
            "scene_participant_count": scene_participant_count,
        }
        if max_output_tokens is not None:
            kwargs["max_output_tokens"] = max_output_tokens
        translated = await client.translate(**kwargs)  # type: ignore[arg-type]
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
        except Exception as exc:
            raise OpenRouterAuthenticationError.from_exception(
                exc, stage="key_verification"
            ) from None
        if response.status_code == 200:
            return True
        if response.status_code == 401:
            return False
        raise OpenRouterAuthenticationError.from_status(
            response.status_code, stage="key_verification"
        )

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
    runtime_logging: ProviderObservationPort | None = None
    _client: httpx.AsyncClient | None = field(init=False, default=None, repr=False)
    _client_lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock, repr=False)
    _last_reasoning_tokens: int | None = field(init=False, default=None, repr=False)

    @property
    def last_reasoning_tokens(self) -> int | None:
        return self._last_reasoning_tokens

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
        scene_participant_count: int | None = None,
        max_output_tokens: int | None = None,
    ) -> dict[str, object]:
        explicit_cache = all(model == OPENROUTER_MODEL_GPT_6_LUNA for model in self.models)
        rendered_system_prompt = _build_system_prompt(
            system_prompt=system_prompt,
            source_language=source_language,
            target_language=target_language,
        )
        system_content: str | list[dict[str, object]] = rendered_system_prompt
        if explicit_cache:
            system_content = [
                {
                    "type": "text",
                    "text": rendered_system_prompt,
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                }
            ]
        user_message = _build_user_message(
            text=text, context=context, scene_participant_count=scene_participant_count
        )

        request_body: dict[str, object] = {
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_message},
            ],
            "reasoning": {"effort": "none"},
            "temperature": 0.6,
            "provider": _build_provider_preferences(
                self.provider_routing,
                model=self.model,
                models=self.models,
            ),
            "max_tokens": max_output_tokens or self.max_tokens,
        }
        if explicit_cache:
            request_body["prompt_cache_options"] = {"mode": "explicit", "ttl": "30m"}
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
        scene_participant_count: int | None = None,
        max_output_tokens: int | None = None,
    ) -> str:
        self._last_reasoning_tokens = None
        request_body = self._build_request_body(
            text=text,
            system_prompt=system_prompt,
            source_language=source_language,
            target_language=target_language,
            context=context,
            scene_participant_count=scene_participant_count,
            max_output_tokens=max_output_tokens,
        )

        client = await self._get_http_client()
        observation = current_attempt()
        if observation is not None:
            observation.transport = "http_json"
            observation.mark_sent()
        response = await client.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=request_body,
        )
        if response.status_code != 200:
            source, reason = (
                _payment_limit_metadata(response) if response.status_code == 402 else (None, None)
            )
            error = OpenRouterResponseError(
                response.status_code,
                limit_source=source,
                limit_reason=reason,
                retry_after_ms=_retry_after_ms(response),
            )
            _log_basic_request_failure(
                runtime_logging=self.runtime_logging,
                operation="translate",
                error=error,
            )
            raise error

        try:
            data = response.json()
        except ValueError:
            raise RuntimeError("OpenRouter response was not valid JSON") from None
        if not isinstance(data, dict):
            raise RuntimeError("OpenRouter response did not contain a valid payload")
        if observation is not None:
            observation.record_openai_response(data)
        self._last_reasoning_tokens = _reasoning_token_count(data)
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("OpenRouter response did not contain choices")
        if _has_length_finish_reason(data):
            raise RuntimeError("OpenRouter response was truncated by max_tokens limit")
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise RuntimeError("OpenRouter response did not contain a valid choice")
        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise RuntimeError("OpenRouter response did not contain message content")
        return _extract_message_content(message.get("content"))

    async def close(self) -> None:
        async with self._client_lock:
            client = self._client
            self._client = None
        if client is not None:
            await client.aclose()
