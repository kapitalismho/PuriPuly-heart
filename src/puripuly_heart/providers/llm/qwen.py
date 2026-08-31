from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

import httpx

from puripuly_heart.core.error_messages import format_error_report_for_log, provider_failure_report
from puripuly_heart.core.observability import ProviderObservationPort
from puripuly_heart.domain.models import Translation
from puripuly_heart.providers.llm.messages import build_translation_user_message

logger = logging.getLogger(__name__)
_QWEN_COMPATIBLE_MODELS = frozenset({"qwen3.5-flash", "qwen3.5-plus", "qwen3.8-flash"})
_QWEN_PROBE_MODEL = "qwen3.8-flash"


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
    status: int | None,
    message: str,
) -> None:
    status_text = "unknown" if status is None else str(status)
    report = provider_failure_report(
        RuntimeError(f"status={status_text} message={message}"),
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


def _normalize_qwen_model(value: object) -> str:
    if not isinstance(value, str):
        return ""
    normalized = value.strip().lower()
    return "qwen3.8-flash" if normalized == "qwen3.5-plus" else normalized


def _is_qwen_compatible_model(model: str) -> bool:
    return _normalize_qwen_model(model) in _QWEN_COMPATIBLE_MODELS


def _to_compatible_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/compatible-mode/v1"):
        return normalized
    if normalized.endswith("/api/v1"):
        return normalized[: -len("/api/v1")] + "/compatible-mode/v1"
    return normalized + "/compatible-mode/v1"


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


class QwenClient(Protocol):
    async def translate(
        self,
        *,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
    ) -> str: ...


@dataclass(slots=True)
class QwenLLMProvider:
    api_key: str
    base_url: str = "https://dashscope.aliyuncs.com/api/v1"
    model: str = "qwen3.8-flash"
    runtime_logging: ProviderObservationPort | None = None
    client: QwenClient | None = None

    def __post_init__(self) -> None:
        self.model = _normalize_qwen_model(self.model)

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
        client = self.client or DashScopeQwenClient(
            api_key=self.api_key,
            model=self.model,
            base_url=self.base_url,
            runtime_logging=self.runtime_logging,
        )
        translated = await client.translate(
            text=text,
            system_prompt=system_prompt,
            source_language=source_language,
            target_language=target_language,
            context=context,
        )
        return Translation(utterance_id=utterance_id, text=translated)

    async def close(self) -> None:
        pass

    async def warmup(self) -> None:
        await self.verify_api_key(self.api_key, base_url=self.base_url, model=self.model)

    @staticmethod
    async def verify_api_key(
        api_key: str,
        base_url: str = "https://dashscope.aliyuncs.com/api/v1",
        model: str = _QWEN_PROBE_MODEL,
    ) -> bool:
        if not api_key:
            return False
        model = _normalize_qwen_model(model)
        if _is_qwen_compatible_model(model):
            compatible_base_url = _to_compatible_base_url(base_url)
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(
                        f"{compatible_base_url}/chat/completions",
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

        try:
            import dashscope  # type: ignore

            def _check():
                try:
                    dashscope.api_key = api_key
                    dashscope.base_http_api_url = base_url
                    response = dashscope.Generation.call(
                        model=model,
                        messages=[{"role": "user", "content": "ping"}],
                        max_tokens=1,
                        result_format="message",
                    )
                    return response.status_code == 200
                except Exception:
                    return False

            return await asyncio.to_thread(_check)
        except Exception:
            return False


@dataclass(slots=True)
class DashScopeQwenClient:
    api_key: str
    model: str
    base_url: str = "https://dashscope.aliyuncs.com/api/v1"
    runtime_logging: ProviderObservationPort | None = None
    def __post_init__(self) -> None:
        self.model = _normalize_qwen_model(self.model)

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

    def _build_messages(
        self,
        *,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str,
    ) -> list[dict[str, str]]:
        system_content = _build_system_prompt(
            system_prompt=system_prompt,
            source_language=source_language,
            target_language=target_language,
        )
        user_message = _build_user_message(text=text, context=context)
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_message},
        ]

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

        def _call() -> str:
            messages = self._build_messages(
                text=text,
                system_prompt=system_prompt,
                source_language=source_language,
                target_language=target_language,
                context=context,
            )
            if _is_qwen_compatible_model(self.model):
                compatible_base_url = _to_compatible_base_url(self.base_url)
                response = httpx.post(
                    f"{compatible_base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": messages,
                        "enable_thinking": False,
                    },
                    timeout=30.0,
                )
                if response.status_code != 200:
                    error_message = ""
                    with contextlib.suppress(Exception):
                        error_message = _extract_error_message(response.json())
                    if not error_message:
                        error_message = response.text[:200]
                    _log_basic_request_failure(
                        runtime_logging=self.runtime_logging,
                        operation="translate",
                        status=response.status_code,
                        message=error_message or "unknown error",
                    )
                    raise RuntimeError(
                        "DashScope compatible-mode request failed "
                        f"(status={response.status_code}, message={error_message})"
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

            import dashscope  # type: ignore

            dashscope.api_key = self.api_key
            dashscope.base_http_api_url = self.base_url
            response = dashscope.Generation.call(
                model=self.model,
                messages=messages,
                result_format="message",
            )
            output = getattr(response, "output", None)
            if not output:
                status = getattr(response, "status_code", None)
                code = getattr(response, "code", None)
                message = getattr(response, "message", None)
                _log_basic_request_failure(
                    runtime_logging=self.runtime_logging,
                    operation="translate",
                    status=status,
                    message=str(message or code or "missing output"),
                )
                raise RuntimeError(
                    "DashScope response did not contain output "
                    f"(status={status}, code={code}, message={message})"
                )
            choice = output.get("choices", [{}])[0]
            message = choice.get("message", {})
            result = _extract_message_content(message.get("content"))
            _log_basic_response(
                runtime_logging=self.runtime_logging,
                operation="translate",
                text=result,
            )
            return result

        return await asyncio.to_thread(_call)
