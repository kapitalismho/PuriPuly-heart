from __future__ import annotations

import asyncio
import contextlib
import re
from dataclasses import dataclass, field
from uuid import UUID

import httpx

from puripuly_heart.core import network_clients
from puripuly_heart.core.external_network import ProxyPolicy, select_tls
from puripuly_heart.core.llm.index_translate import (
    IndexRequestLimitError,
    IndexServiceUnavailableError,
    IndexTranslateError,
)
from puripuly_heart.core.llm.latency import current_attempt
from puripuly_heart.core.llm.provider import LLMProvider
from puripuly_heart.core.observability import ProviderObservationPort
from puripuly_heart.core.owned_thread import run_owned_thread_call
from puripuly_heart.domain.models import Translation
from puripuly_heart.providers.llm.messages import build_translation_user_message

INDEX_TRANSLATE_ENDPOINT = "https://index-translate.bilibili.com/v1/chat/completions"
INDEX_TRANSLATE_MODEL = "Index-Translate-35B-A3B"
INDEX_DEFAULT_OUTPUT_TOKENS = 2048
_THOUGHT_MARKER = re.compile(r"</?(?:think|analysis|reasoning)(?:\s|>)|<\|", re.IGNORECASE)


def _parse_translation(data: object) -> str:
    if not isinstance(data, dict):
        raise IndexServiceUnavailableError("invalid_response")
    choices = data.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise IndexServiceUnavailableError("invalid_response")
    choice = choices[0]
    if choice.get("finish_reason") == "length":
        raise IndexRequestLimitError("output_limit")
    if choice.get("finish_reason") != "stop":
        raise IndexServiceUnavailableError("invalid_response")
    message = choice.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise IndexServiceUnavailableError("invalid_response")
    if message.get("tool_calls") or message.get("function_call") or message.get("refusal"):
        raise IndexServiceUnavailableError("invalid_response")
    content = message.get("content")
    if isinstance(content, list):
        if not content or any(
            not isinstance(part, dict)
            or part.get("type") != "text"
            or not isinstance(part.get("text"), str)
            for part in content
        ):
            raise IndexServiceUnavailableError("invalid_response")
        content = "\n".join(part["text"] for part in content)
    if not isinstance(content, str) or not content.strip() or _THOUGHT_MARKER.search(content):
        raise IndexServiceUnavailableError("invalid_response")
    return content.strip()


@dataclass(slots=True)
class IndexTranslateProvider(LLMProvider):
    model: str = INDEX_TRANSLATE_MODEL
    timeout: float = 20.0
    availability_timeout: float = 10.0
    runtime_logging: ProviderObservationPort | None = None
    _client: httpx.AsyncClient | None = field(init=False, default=None, repr=False)
    _client_lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock, repr=False)
    _closed: bool = field(init=False, default=False, repr=False)
    _close_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)

    async def _get_http_client(self) -> httpx.AsyncClient:
        async with self._client_lock:
            if self._closed:
                raise asyncio.CancelledError
            if self._client is None:
                tls, policy = await run_owned_thread_call(lambda: (select_tls(), ProxyPolicy()))
                self._client = network_clients.external_async_client(
                    tls=tls,
                    policy=policy,
                    timeout=self.timeout,
                    headers={"User-Agent": "PuriPulyHeart", "Content-Type": "application/json"},
                )
            return self._client

    async def _complete(
        self,
        *,
        system_prompt: str,
        user_message: str,
        max_output_tokens: int,
        timeout: float,
    ) -> str:
        if isinstance(max_output_tokens, bool) or max_output_tokens <= 0:
            raise IndexRequestLimitError("output_limit")
        try:
            async with asyncio.timeout(timeout):
                client = await self._get_http_client()
                observation = current_attempt()
                if observation is not None:
                    observation.transport = "http_json"
                    observation.mark_sent()
                response = await client.post(
                    INDEX_TRANSLATE_ENDPOINT,
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message},
                        ],
                        "temperature": 0,
                        "chat_template_kwargs": {"enable_thinking": False},
                        "max_tokens": max_output_tokens,
                        "stream": False,
                    },
                )
                if response.status_code != 200:
                    if response.status_code == 413:
                        raise IndexRequestLimitError("input_limit", status_code=413)
                    if response.status_code in (400, 422):
                        raise IndexTranslateError(
                            "invalid_request", status_code=response.status_code
                        )
                    raise IndexServiceUnavailableError(
                        "http_status", status_code=response.status_code
                    )
                try:
                    data = response.json()
                except ValueError as exc:
                    raise IndexServiceUnavailableError("invalid_response") from exc
                if observation is not None and isinstance(data, dict):
                    observation.record_openai_response(data)
                return _parse_translation(data)
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise IndexServiceUnavailableError("timeout") from exc
        except httpx.TransportError as exc:
            raise IndexServiceUnavailableError("transport") from exc

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
        translated = await self._complete(
            system_prompt=system_prompt,
            user_message=build_translation_user_message(
                text=text, context=context, scene_participant_count=scene_participant_count
            ),
            max_output_tokens=(
                INDEX_DEFAULT_OUTPUT_TOKENS if max_output_tokens is None else max_output_tokens
            ),
            timeout=self.timeout,
        )
        return Translation(utterance_id=utterance_id, text=translated)

    async def check_availability(self) -> None:
        await self._complete(
            system_prompt="Translate from English to Spanish. Return only the translation.",
            user_message=build_translation_user_message(text="Hello.", context=""),
            max_output_tokens=64,
            timeout=self.availability_timeout,
        )

    async def close(self) -> None:
        if self._close_task is None:
            self._closed = True
            self._close_task = asyncio.create_task(
                self._close_owned_client(), name="index-translate-client-close"
            )
        try:
            await asyncio.shield(self._close_task)
        except asyncio.CancelledError:
            while not self._close_task.done():
                try:
                    await asyncio.shield(self._close_task)
                except asyncio.CancelledError:
                    current_task = asyncio.current_task()
                    if current_task is None or not current_task.cancelling():
                        raise
                except BaseException:
                    break
            with contextlib.suppress(BaseException):
                self._close_task.result()
            raise asyncio.CancelledError

    async def _close_owned_client(self) -> None:
        async with self._client_lock:
            if self._client is not None:
                await self._client.aclose()
                self._client = None
