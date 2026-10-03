from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Protocol
from uuid import UUID

import httpx

from puripuly_heart.config.runtime_resolution import OPENAI_MODEL_GPT_6_LUNA
from puripuly_heart.core.error_messages import format_error_report_for_log, provider_failure_report
from puripuly_heart.core.llm.latency import current_attempt
from puripuly_heart.core.observability import ProviderObservationPort
from puripuly_heart.domain.models import Translation
from puripuly_heart.providers.llm.messages import build_translation_user_message

logger = logging.getLogger(__name__)
_OPENAI_BASE_URL = "https://api.openai.com/v1"


class OpenAIResponseError(RuntimeError):
    diagnostic_provider = "openai"

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(f"OpenAI request failed (status={status_code})")


def _log_basic_request_failure(
    *,
    runtime_logging: ProviderObservationPort | None,
    operation: str,
    error: OpenAIResponseError,
) -> None:
    report = provider_failure_report(error, provider="openai", operation=operation)
    rendered = "[Basic][LLM] OpenAI request failed [%s]: %s" % (
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
    if "{source_language}" not in system_prompt:
        return system_prompt
    return system_prompt.format(
        source_language=source_language,
        target_language=target_language,
    )


def _extract_message_content(content: object) -> str:
    if isinstance(content, str):
        result = content.strip()
        if result:
            return result
        raise RuntimeError("OpenAI response contained empty message content")

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        if parts:
            return "\n".join(parts)

    raise RuntimeError("OpenAI response did not contain message content")


def _has_length_finish_reason(data: object) -> bool:
    if not isinstance(data, dict):
        return False
    choices = data.get("choices")
    if not isinstance(choices, list):
        return False
    return any(
        isinstance(choice, dict) and choice.get("finish_reason") == "length" for choice in choices
    )


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


class OpenAIClient(Protocol):
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


@dataclass(slots=True)
class OpenAILLMProvider:
    api_key: str = field(repr=False)
    base_url: str = _OPENAI_BASE_URL
    model: str = OPENAI_MODEL_GPT_6_LUNA
    timeout: float = 30.0
    runtime_logging: ProviderObservationPort | None = None
    client: OpenAIClient | None = None
    _internal_client: OpenAIClient | None = field(init=False, default=None, repr=False)

    def _get_client(self) -> OpenAIClient:
        if self.client is not None:
            return self.client
        if self._internal_client is None:
            self._internal_client = HttpxOpenAIClient(
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
        scene_participant_count: int | None = None,
        max_output_tokens: int | None = None,
    ) -> Translation:
        client = self._get_client()
        translated = await client.translate(
            text=text,
            system_prompt=system_prompt,
            source_language=source_language,
            target_language=target_language,
            context=context,
            scene_participant_count=scene_participant_count,
            max_output_tokens=max_output_tokens,
        )
        return Translation(utterance_id=utterance_id, text=translated)

    async def close(self) -> None:
        if self._internal_client is not None:
            await self._internal_client.close()
            self._internal_client = None

    @staticmethod
    async def verify_api_key(
        api_key: str,
        *,
        model: str = OPENAI_MODEL_GPT_6_LUNA,
    ) -> bool:
        if not api_key:
            return False
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{_OPENAI_BASE_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": "Reply with OK."}],
                        "reasoning_effort": "none",
                        "temperature": 0.6,
                        "max_completion_tokens": 1,
                    },
                )
                return response.status_code == 200
        except Exception:
            return False


@dataclass(slots=True)
class HttpxOpenAIClient:
    api_key: str
    model: str = OPENAI_MODEL_GPT_6_LUNA
    base_url: str = _OPENAI_BASE_URL
    timeout: float = 30.0
    runtime_logging: ProviderObservationPort | None = None
    _client: httpx.AsyncClient | None = field(init=False, default=None, repr=False)
    _client_lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock, repr=False)
    _last_reasoning_tokens: int | None = field(init=False, default=None, repr=False)

    @property
    def last_reasoning_tokens(self) -> int | None:
        return self._last_reasoning_tokens

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
        rendered_system_prompt = _build_system_prompt(
            system_prompt=system_prompt,
            source_language=source_language,
            target_language=target_language,
        )
        system_content: str | list[dict[str, object]] = rendered_system_prompt
        if self.model == OPENAI_MODEL_GPT_6_LUNA:
            system_content = [
                {
                    "type": "text",
                    "text": rendered_system_prompt,
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                }
            ]
        user_message = build_translation_user_message(
            text=text,
            context=context,
            scene_participant_count=scene_participant_count,
        )
        request_body: dict[str, object] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_message},
            ],
            "reasoning_effort": "none",
            "temperature": 0.6,
        }
        if self.model == OPENAI_MODEL_GPT_6_LUNA:
            request_body["prompt_cache_options"] = {"mode": "explicit", "ttl": "30m"}
        if max_output_tokens is not None:
            request_body["max_completion_tokens"] = max_output_tokens
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
            f"{self.base_url.rstrip('/')}/chat/completions",
            headers=self._headers(),
            json=request_body,
        )
        if response.status_code != 200:
            error = OpenAIResponseError(response.status_code)
            _log_basic_request_failure(
                runtime_logging=self.runtime_logging,
                operation="translate",
                error=error,
            )
            raise error

        try:
            data = response.json()
        except ValueError:
            raise RuntimeError("OpenAI response was not valid JSON") from None
        if not isinstance(data, dict):
            raise RuntimeError("OpenAI response did not contain a valid payload")
        if observation is not None:
            observation.record_openai_response(data)
        self._last_reasoning_tokens = _reasoning_token_count(data)
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("OpenAI response did not contain choices")
        if _has_length_finish_reason(data):
            raise RuntimeError("OpenAI response was truncated by max_completion_tokens limit")
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise RuntimeError("OpenAI response did not contain a valid choice")
        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise RuntimeError("OpenAI response did not contain message content")
        return _extract_message_content(message.get("content"))

    async def close(self) -> None:
        async with self._client_lock:
            client = self._client
            self._client = None
        if client is not None:
            await client.aclose()
