from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from uuid import UUID

import httpx

from puripuly_heart.core.http_client_logging import suppress_http_client_logs
from puripuly_heart.core.llm.latency import (
    AttemptLatency,
    current_attempt,
    current_request,
    observe_attempt,
)
from puripuly_heart.core.local_translation.assets import GEMMA_26B_MODEL_ID, GemmaModelSpec
from puripuly_heart.core.local_translation.runtime import (
    ManagedGemmaMetrics,
    ManagedGemmaResponse,
    ManagedGemmaRuntimeOwner,
)
from puripuly_heart.core.local_translation.runtime_profile import (
    MANAGED_GEMMA_26B_MODEL_ALIAS,
    MANAGED_GEMMA_MODEL_ALIAS,
    GemmaBackend,
)
from puripuly_heart.core.observability import ProviderObservationPort
from puripuly_heart.domain.models import Translation
from puripuly_heart.providers.llm.error_details import LLMResponseError
from puripuly_heart.providers.llm.messages import build_translation_user_message

logger = logging.getLogger(__name__)


def _number(value: object) -> float:
    return float(value) if isinstance(value, int | float) else 0.0


def _integer(value: object) -> int:
    return int(value) if isinstance(value, int | float) else 0


def _optional_integer(value: object) -> int | None:
    return int(value) if isinstance(value, int | float) else None


def _response_text(payload: object) -> str:
    if not isinstance(payload, dict):
        raise LLMResponseError("managed Gemma response was not an object", provider="managed_gemma")
    content = payload.get("content")
    if not isinstance(content, str) or not content.strip():
        raise LLMResponseError(
            "managed Gemma response contained empty content", provider="managed_gemma"
        )
    return content.strip()


def _response_metrics(payload: object) -> ManagedGemmaMetrics:
    if not isinstance(payload, dict):
        raise LLMResponseError("managed Gemma response was not an object", provider="managed_gemma")
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

    async def wait_slot_idle(self, *, slot_id: int) -> None:
        timeout = self.timeout.read if isinstance(self.timeout, httpx.Timeout) else self.timeout
        async with asyncio.timeout(timeout or 120.0):
            while True:
                response = await self._require_client().get("/slots")
                response.raise_for_status()
                slots = response.json()
                if isinstance(slots, list):
                    for slot in slots:
                        if isinstance(slot, dict) and slot.get("id") == slot_id:
                            if slot.get("is_processing") is False:
                                return
                            break
                    else:
                        raise LLMResponseError(
                            "managed Gemma slot was missing during settlement",
                            provider="managed_gemma",
                        )
                else:
                    raise LLMResponseError(
                        "managed Gemma slot state was invalid", provider="managed_gemma"
                    )
                await asyncio.sleep(min(self.poll_interval_s, 0.1))

    async def restore_prefix(self, *, filename: str, slot_id: int) -> bool:
        try:
            response = await self._require_client().post(
                f"/slots/{slot_id}?action=restore",
                json={"filename": filename},
            )
        except httpx.HTTPError:
            await self.wait_slot_idle(slot_id=slot_id)
            return False
        return response.status_code == 200

    async def save_prefix(self, *, filename: str, slot_id: int) -> None:
        try:
            response = await self._require_client().post(
                f"/slots/{slot_id}?action=save",
                json={"filename": filename},
            )
            response.raise_for_status()
        except httpx.HTTPError:
            await self.wait_slot_idle(slot_id=slot_id)
            raise

    async def translate(
        self,
        *,
        system_prompt: str,
        user_message: str,
        slot_id: int,
        max_output_tokens: int | None = None,
    ) -> ManagedGemmaResponse:
        payload = await self._completion(
            messages=(
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ),
            slot_id=slot_id,
            max_tokens=max_output_tokens,
            observation=current_attempt(),
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
        observation: AttemptLatency | None = None,
    ) -> object:
        template_response = await self._require_client().post(
            "/apply-template",
            json={
                "messages": list(messages),
                "add_generation_prompt": True,
            },
        )
        template_response.raise_for_status()
        try:
            template_payload = template_response.json()
        except ValueError as exc:
            raise LLMResponseError(
                "managed Gemma template response was not valid JSON", provider="managed_gemma"
            ) from exc
        if not isinstance(template_payload, dict):
            raise LLMResponseError(
                "managed Gemma template response was not an object", provider="managed_gemma"
            )
        prompt = template_payload.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise LLMResponseError(
                "managed Gemma template response did not contain a prompt", provider="managed_gemma"
            )
        body: dict[str, object] = {
            "prompt": prompt,
            "stream": False,
            "temperature": 0.6,
            "cache_prompt": True,
            "id_slot": slot_id,
        }
        if max_tokens is not None:
            body["n_predict"] = max_tokens
        client = self._require_client()
        if observation is not None:
            observation.transport = "http_json"
            observation.mark_sent()
        response = await client.post("/completion", json=body)
        response.raise_for_status()
        try:
            payload = response.json()
        except ValueError as exc:
            raise LLMResponseError(
                "managed Gemma response was not valid JSON", provider="managed_gemma"
            ) from exc
        if observation is not None and isinstance(payload, dict):
            timings = payload.get("timings")
            if isinstance(timings, dict):
                observation.record_usage(
                    input_tokens=timings.get("prompt_n"),
                    output_tokens=timings.get("predicted_n"),
                    cached_input_tokens=timings.get("cache_n"),
                )
                observation.record_server_timings(
                    prompt_ms=timings.get("prompt_ms"),
                    generation_ms=timings.get("predicted_ms", timings.get("generation_ms")),
                    generation_tps=timings.get(
                        "predicted_per_second", timings.get("generation_tps")
                    ),
                )
            model = payload.get("model")
            observation.actual_model = model if isinstance(model, str) else None
        return payload

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
    runtime_logging: ProviderObservationPort | None = None
    spec: GemmaModelSpec | None = None
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
        scene_participant_count: int | None = None,
        max_output_tokens: int | None = None,
    ) -> Translation:
        if self._closed:
            raise LLMResponseError(
                "managed Gemma provider is closed",
                provider="managed_gemma",
                diagnostic_category="lifecycle",
            )
        runtime_kwargs = {
            "backend": self.backend,
            "source_language": source_language,
            "target_language": target_language,
            "system_prompt": system_prompt,
            "user_message": build_translation_user_message(
                text=text,
                context=context,
                scene_participant_count=scene_participant_count,
            ),
            "vulkan_device": self.vulkan_device,
        }
        if max_output_tokens is not None:
            runtime_kwargs["max_output_tokens"] = max_output_tokens
        if self.spec is not None:
            runtime_kwargs["spec"] = self.spec
        model = (
            MANAGED_GEMMA_26B_MODEL_ALIAS
            if self.spec is not None and self.spec.model_id == GEMMA_26B_MODEL_ID
            else MANAGED_GEMMA_MODEL_ALIAS
        )
        with observe_attempt(provider="managed_gemma", model=model):
            response = await self.runtime.translate(**runtime_kwargs)  # type: ignore[arg-type]
            request = current_request()
            if request is not None:
                request.winner_attempt = 0

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
