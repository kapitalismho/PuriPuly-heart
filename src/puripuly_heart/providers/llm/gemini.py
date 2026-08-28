from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol
from uuid import UUID

from puripuly_heart.core.observability import ProviderObservationPort
from puripuly_heart.domain.models import Translation
from puripuly_heart.providers.llm.messages import build_translation_user_message

logger = logging.getLogger(__name__)

GEMINI_31_FLASH_LITE_GA_MODEL = "gemini-3.1-flash-lite"


def _normalized_model_id(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().rsplit("/", 1)[-1]


def _model_entry_matches(entry: object, requested_model: str) -> bool:
    for attr in ("name", "id", "model", "model_id", "base_model_id", "baseModelId"):
        if _normalized_model_id(getattr(entry, attr, None)) == requested_model:
            return True
    if _normalized_model_id(entry) == requested_model:
        return True
    return False


def _log_basic_request(
    *,
    runtime_logging: ProviderObservationPort | None,
    operation: str,
    text: str,
    source_language: str,
    target_language: str,
    context: str,
) -> None:
    message = "[Basic][LLM] Gemini request [%s][context=%s] %s -> %s: %r" % (
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
    message = "[Basic][LLM] Gemini response [%s]: %r" % (operation, text)
    if runtime_logging is not None:
        runtime_logging.emit_basic(message)
        return
    logger.info(message)


def _log_basic_missing_text(
    *, runtime_logging: ProviderObservationPort | None, operation: str
) -> None:
    message = "[Basic][LLM] Gemini response missing text [%s]" % operation
    if runtime_logging is not None:
        runtime_logging.emit_basic(message, level=logging.ERROR)
        return
    logger.error(message)


class GeminiClient(Protocol):
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
class GeminiLLMProvider:
    api_key: str
    model: str = GEMINI_31_FLASH_LITE_GA_MODEL
    runtime_logging: ProviderObservationPort | None = None
    client: GeminiClient | None = None
    _internal_client: GeminiClient | None = field(init=False, default=None, repr=False)

    def _get_client(self) -> GeminiClient:
        if self.client is not None:
            return self.client
        if self._internal_client is None:
            self._internal_client = GoogleGenaiGeminiClient(
                api_key=self.api_key,
                model=self.model,
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

    async def warmup(self) -> None:
        client = self._get_client()
        await client.translate(
            text="warmup",
            system_prompt="Reply with OK only.",
            source_language="en",
            target_language="en",
            context="",
        )

    async def close(self) -> None:
        if self._internal_client is not None:
            await self._internal_client.close()
            self._internal_client = None

    @staticmethod
    async def verify_api_key(
        api_key: str,
        *,
        model: str = GEMINI_31_FLASH_LITE_GA_MODEL,
    ) -> bool:
        if not api_key:
            return False
        try:
            from google import genai  # type: ignore

            client = genai.Client(api_key=api_key)
            requested_model = _normalized_model_id(model)
            async for entry in await client.aio.models.list(config={"page_size": 1000}):
                if not requested_model or _model_entry_matches(entry, requested_model):
                    return True
            return False
        except Exception:
            return False


@dataclass(slots=True)
class GoogleGenaiGeminiClient:
    api_key: str
    model: str
    runtime_logging: ProviderObservationPort | None = None
    _client: Any = field(init=False, default=None, repr=False)

    def _get_client(self) -> Any:
        if self._client is None:
            from google import genai  # type: ignore

            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _build_request(
        self,
        *,
        operation: str,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str,
    ) -> tuple[str, str]:
        formatted_system_prompt = (
            system_prompt.format(
                source_language=source_language,
                target_language=target_language,
            )
            if "{source_language}" in system_prompt
            else system_prompt
        )

        _log_basic_request(
            runtime_logging=self.runtime_logging,
            operation=operation,
            text=text,
            source_language=source_language,
            target_language=target_language,
            context=context,
        )

        return formatted_system_prompt, build_translation_user_message(text=text, context=context)

    async def translate(
        self,
        *,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
    ) -> str:
        from google.genai import types  # type: ignore

        formatted_system_prompt, user_message = self._build_request(
            operation="translate",
            text=text,
            system_prompt=system_prompt,
            source_language=source_language,
            target_language=target_language,
            context=context,
        )

        client = self._get_client()
        thinking_level = (
            types.ThinkingLevel.MINIMAL
            if _normalized_model_id(self.model) == GEMINI_31_FLASH_LITE_GA_MODEL
            else types.ThinkingLevel.LOW
        )
        response = await client.aio.models.generate_content(
            model=self.model,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=formatted_system_prompt,
                temperature=0.6,
                thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
        if getattr(response, "text", None):
            result = str(response.text).strip()
            _log_basic_response(
                runtime_logging=self.runtime_logging,
                operation="translate",
                text=result,
            )
            return result
        _log_basic_missing_text(runtime_logging=self.runtime_logging, operation="translate")
        raise RuntimeError("Gemini response did not contain text")

    async def close(self) -> None:
        self._client = None
