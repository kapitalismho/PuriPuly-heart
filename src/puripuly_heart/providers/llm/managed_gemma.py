from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from uuid import UUID

import httpx

from puripuly_heart.core.error_messages import format_error_report_for_log, provider_failure_report
from puripuly_heart.core.http_client_logging import suppress_http_client_logs
from puripuly_heart.core.local_translation.runtime import (
    ManagedGemmaMetrics,
    ManagedGemmaResponse,
    ManagedGemmaRuntimeOwner,
)
from puripuly_heart.core.local_translation.runtime_profile import GemmaBackend
from puripuly_heart.domain.models import Translation
from puripuly_heart.providers.llm.messages import build_translation_user_message

logger = logging.getLogger(__name__)


def _log_basic_request(
    *,
    runtime_logging: object | None,
    operation: str,
    text: str,
    source_language: str,
    target_language: str,
    context: str,
) -> None:
    message = "[Basic][LLM] Gemma request [%s][context=%s] %s -> %s: %r" % (
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


def _log_basic_response(*, runtime_logging: object | None, operation: str, text: str) -> None:
    message = "[Basic][LLM] Gemma response [%s]: %r" % (operation, text)
    if runtime_logging is not None:
        runtime_logging.emit_basic(message)
        return
    logger.info(message)


def _log_basic_request_failure(
    *,
    runtime_logging: object | None,
    operation: str,
    exc: BaseException,
) -> None:
    report = provider_failure_report(
        exc,
        provider="managed_gemma",
        operation=operation,
    )
    rendered = "[Basic][LLM] Gemma request failed [%s]: %s" % (
        operation,
        format_error_report_for_log(report),
    )
    if runtime_logging is not None:
        runtime_logging.emit_basic(rendered, level=logging.ERROR)
        return
    logger.error(rendered)


def _number(value: object) -> float:
    return float(value) if isinstance(value, int | float) else 0.0


def _integer(value: object) -> int:
    return int(value) if isinstance(value, int | float) else 0


def _optional_integer(value: object) -> int | None:
    return int(value) if isinstance(value, int | float) else None


def _response_text(payload: object) -> str:
    if not isinstance(payload, dict):
        raise RuntimeError("managed Gemma response was not an object")
    content = payload.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("managed Gemma response contained empty content")
    return content.strip()


def _response_metrics(payload: object) -> ManagedGemmaMetrics:
    if not isinstance(payload, dict):
        raise RuntimeError("managed Gemma response was not an object")
    timings = payload.get("timings") if isinstance(payload.get("timings"), dict) else {}
    prompt_tokens = _integer(timings.get("prompt_n"))
    completion_tokens = _integer(timings.get("predicted_n"))
    prompt_ms = _number(timings.get("prompt_ms"))
    generation_ms = _number(timings.get("predicted_ms", timings.get("generation_ms")))
    generation_tps = _number(timings.get("predicted_per_second", timings.get("generation_tps")))
    if generation_tps <= 0 and generation_ms > 0:
        generation_tps = completion_tokens * 1000.0 / generation_ms
    return ManagedGemmaMetrics(
        prompt_tokens=prompt_tokens,
        cached_prompt_tokens=_optional_integer(timings.get("cache_n")),
        completion_tokens=completion_tokens,
        prompt_ms=prompt_ms,
        generation_ms=generation_ms,
        generation_tps=generation_tps,
        drafted_tokens=_optional_integer(timings.get("draft_n")),
        accepted_tokens=_optional_integer(timings.get("draft_n_accepted")),
    )


@dataclass(slots=True)
class HttpxManagedGemmaTransport:
    base_url: str
    timeout: httpx.Timeout | float = field(
        default_factory=lambda: httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0)
    )
    poll_interval_s: float = 1.0
    _client: httpx.AsyncClient | None = field(init=False, default=None, repr=False)

    def _require_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                trust_env=False,
                follow_redirects=False,
            )
        return self._client

    async def wait_until_ready(self, *, timeout_s: float) -> None:
        deadline = asyncio.get_running_loop().time() + timeout_s
        client = self._require_client()
        last_error: Exception | None = None
        with suppress_http_client_logs():
            while asyncio.get_running_loop().time() < deadline:
                try:
                    response = await client.get("/health")
                    if response.status_code == 200:
                        return
                except httpx.HTTPError as exc:
                    last_error = exc
                await asyncio.sleep(self.poll_interval_s)
        raise TimeoutError("managed llama.cpp server did not become ready") from last_error

    async def prepare_prefix(self, *, system_prompt: str, slot_id: int) -> None:
        await self._completion(
            messages=(
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": " "},
            ),
            max_tokens=1,
            slot_id=slot_id,
        )

    async def restore_prefix(self, *, filename: str, slot_id: int) -> bool:
        try:
            response = await self._require_client().post(
                f"/slots/{slot_id}?action=restore",
                json={"filename": filename},
            )
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    async def save_prefix(self, *, filename: str, slot_id: int) -> None:
        response = await self._require_client().post(
            f"/slots/{slot_id}?action=save",
            json={"filename": filename},
        )
        response.raise_for_status()

    async def translate(
        self,
        *,
        system_prompt: str,
        user_message: str,
        slot_id: int,
    ) -> ManagedGemmaResponse:
        payload = await self._completion(
            messages=(
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ),
            slot_id=slot_id,
        )
        return ManagedGemmaResponse(
            text=_response_text(payload),
            metrics=_response_metrics(payload),
        )

    async def _completion(
        self,
        *,
        messages: tuple[dict[str, str], ...],
        slot_id: int,
        max_tokens: int | None = None,
    ) -> object:
        template_response = await self._require_client().post(
            "/apply-template",
            json={
                "messages": list(messages),
                "add_generation_prompt": True,
            },
        )
        template_response.raise_for_status()
        template_payload = template_response.json()
        if not isinstance(template_payload, dict):
            raise RuntimeError("managed Gemma template response was not an object")
        prompt = template_payload.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise RuntimeError("managed Gemma template response did not contain a prompt")
        body: dict[str, object] = {
            "prompt": prompt,
            "stream": False,
            "temperature": 0.2,
            "cache_prompt": True,
            "id_slot": slot_id,
        }
        if max_tokens is not None:
            body["n_predict"] = max_tokens
        response = await self._require_client().post("/completion", json=body)
        response.raise_for_status()
        return response.json()

    async def close(self) -> None:
        client = self._client
        if client is not None:
            await client.aclose()
            if self._client is client:
                self._client = None


@dataclass(slots=True)
class ManagedGemmaLLMProvider:
    runtime: ManagedGemmaRuntimeOwner
    backend: GemmaBackend
    vulkan_device: str = "Vulkan0"
    release_runtime: Callable[[], Awaitable[None]] | None = None
    runtime_logging: object | None = None
    _closed: bool = field(init=False, default=False, repr=False)
    _released: bool = field(init=False, default=False, repr=False)

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
        if self._closed:
            raise RuntimeError("managed Gemma provider is closed")
        _log_basic_request(
            runtime_logging=self.runtime_logging,
            operation="translate",
            text=text,
            source_language=source_language,
            target_language=target_language,
            context=context,
        )
        try:
            response = await self.runtime.translate(
                backend=self.backend,
                source_language=source_language,
                target_language=target_language,
                system_prompt=system_prompt,
                user_message=build_translation_user_message(text=text, context=context),
                vulkan_device=self.vulkan_device,
            )
        except Exception as exc:
            _log_basic_request_failure(
                runtime_logging=self.runtime_logging,
                operation="translate",
                exc=exc,
            )
            raise
        _log_basic_response(
            runtime_logging=self.runtime_logging,
            operation="translate",
            text=response.text,
        )
        return Translation(
            utterance_id=utterance_id,
            text=response.text,
            source_text=text,
            source_language=source_language,
            target_language=target_language,
        )

    async def close(self) -> None:
        if self._released:
            return
        self._closed = True
        if self.release_runtime is None:
            await self.runtime.release()
        else:
            await self.release_runtime()
        self._released = True


__all__ = ["HttpxManagedGemmaTransport", "ManagedGemmaLLMProvider"]
