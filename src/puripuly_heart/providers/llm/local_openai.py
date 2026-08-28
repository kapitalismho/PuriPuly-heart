from __future__ import annotations

import ast
import asyncio
import contextlib
import json
import re
from dataclasses import dataclass, field
from typing import Mapping, Protocol
from urllib.parse import urlsplit, urlunsplit
from uuid import UUID

import httpx

from puripuly_heart.core.runtime_logging import SessionRuntimeLoggingService
from puripuly_heart.domain.models import Translation
from puripuly_heart.providers.llm.messages import build_translation_user_message

LOCAL_OPENAI_RESERVED_EXTRA_BODY_KEYS = frozenset(
    {
        "model",
        "messages",
        "stream",
        "tools",
        "tool_choice",
        "functions",
        "function_call",
        "max_tokens",
    }
)
LOCAL_OPENAI_SENSITIVE_EXTRA_BODY_KEYS = frozenset(
    {"api_key", "authorization", "headers", "token", "secret", "password"}
)
_DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"
_MODEL_TOKEN_CHARS = r"A-Za-z0-9_./:+-"


class LocalOpenAIReservedExtraBodyKeyError(ValueError):
    pass


class LocalOpenAISensitiveExtraBodyKeyError(ValueError):
    pass


def _default_timeout() -> httpx.Timeout:
    return httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0)


def _normalize_base_url(value: str) -> str:
    try:
        parsed = urlsplit(value.strip())
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("invalid local LLM base URL") from exc
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("invalid local LLM base URL")
    if not parsed.hostname:
        raise ValueError("invalid local LLM base URL")
    if (
        "@" in parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("invalid local LLM base URL")
    path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _assert_extra_body_is_safe(extra_body: Mapping[str, object]) -> None:
    for key, value in extra_body.items():
        if not isinstance(key, str):
            raise ValueError("local LLM extra_body keys must be strings")
        normalized = key.lower()
        if normalized in LOCAL_OPENAI_RESERVED_EXTRA_BODY_KEYS:
            raise LocalOpenAIReservedExtraBodyKeyError(f"reserved local LLM extra_body key: {key}")
        if normalized in LOCAL_OPENAI_SENSITIVE_EXTRA_BODY_KEYS:
            raise LocalOpenAISensitiveExtraBodyKeyError(
                f"sensitive local LLM extra_body key: {key}"
            )
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"local LLM extra_body value for {key!r} is not JSON serializable"
            ) from exc


def _build_system_prompt(*, system_prompt: str, source_language: str, target_language: str) -> str:
    return (
        system_prompt.format(source_language=source_language, target_language=target_language)
        if "{source_language}" in system_prompt
        else system_prompt
    )


def _extract_message_content(content: object) -> str:
    if isinstance(content, str):
        stripped = content.strip()
        if stripped:
            return stripped
        raise RuntimeError("Local LLM response contained empty message content")
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        if parts:
            return "\n".join(parts)
    raise RuntimeError("Local LLM response did not contain message content")


def _extract_error_message(data: object) -> str:
    if isinstance(data, dict):
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


def _redact_extra_body_fragments(text: str, extra_body: Mapping[str, object] | None) -> str:
    if not extra_body:
        return text
    sanitized = text
    for rendered in (
        json.dumps(dict(extra_body), ensure_ascii=False, sort_keys=True),
        json.dumps(dict(extra_body), ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    ):
        sanitized = sanitized.replace(rendered, "[extra-body-redacted]")
    sanitized = _redact_extra_body_json_objects(sanitized, extra_body)
    sanitized = _redact_extra_body_key_value_fragments(sanitized, extra_body)
    return sanitized


def _contains_extra_body_pair(value: object, extra_body: Mapping[str, object]) -> bool:
    if isinstance(value, dict):
        if any(key in value and value[key] == expected for key, expected in extra_body.items()):
            return True
        return any(_contains_extra_body_pair(child, extra_body) for child in value.values())
    if isinstance(value, list):
        return any(_contains_extra_body_pair(item, extra_body) for item in value)
    return False


def _redact_extra_body_json_objects(text: str, extra_body: Mapping[str, object]) -> str:
    decoder = json.JSONDecoder()
    parts: list[str] = []
    index = 0
    while index < len(text):
        start = text.find("{", index)
        if start < 0:
            parts.append(text[index:])
            break
        parts.append(text[index:start])
        try:
            parsed, end = decoder.raw_decode(text[start:])
        except ValueError:
            parts.append(text[start])
            index = start + 1
            continue
        if _contains_extra_body_pair(parsed, extra_body):
            parts.append("[extra-body-redacted]")
        else:
            parts.append(text[start : start + end])
        index = start + end
    return "".join(parts)


def _extra_body_value_patterns(value: object) -> list[str]:
    rendered: set[str] = set()
    for kwargs in ({}, {"separators": (",", ":")}):
        with contextlib.suppress(TypeError, ValueError):
            rendered.add(json.dumps(value, ensure_ascii=False, allow_nan=False, **kwargs))
    if isinstance(value, bool):
        rendered.add(str(value))
    elif value is None:
        rendered.add("None")
    elif isinstance(value, str) and value:
        rendered.add(value)
        rendered.add(f"'{value}'")
    return [re.escape(item) for item in sorted(rendered, key=len, reverse=True)]


def _redact_extra_body_key_value_fragments(text: str, extra_body: Mapping[str, object]) -> str:
    sanitized = _redact_extra_body_structured_key_value_fragments(text, extra_body)
    for key, value in sorted(extra_body.items(), key=lambda item: len(item[0]), reverse=True):
        value_pattern = "|".join(_extra_body_value_patterns(value))
        if not value_pattern:
            continue
        escaped_key = re.escape(key)
        key_pattern = rf"(?:[\"']{escaped_key}[\"']|(?<![\w.-]){escaped_key}(?![\w.-]))"
        sanitized = re.sub(
            rf"{key_pattern}\s*[:=]\s*(?:{value_pattern})(?=$|[\s,;.)\]}}])",
            "[extra-body-redacted]",
            sanitized,
            flags=re.IGNORECASE,
        )
    return sanitized


def _redact_extra_body_structured_key_value_fragments(
    text: str, extra_body: Mapping[str, object]
) -> str:
    sanitized = text
    for key, value in sorted(extra_body.items(), key=lambda item: len(item[0]), reverse=True):
        if not isinstance(value, dict):
            continue
        escaped_key = re.escape(key)
        key_pattern = rf"(?:[\"']{escaped_key}[\"']|(?<![\w.-]){escaped_key}(?![\w.-]))"
        sanitized = _redact_matching_structured_value(sanitized, key_pattern, value)
    return sanitized


def _redact_matching_structured_value(text: str, key_pattern: str, expected: object) -> str:
    pattern = re.compile(rf"{key_pattern}\s*[:=]\s*", flags=re.IGNORECASE)
    parts: list[str] = []
    index = 0
    while index < len(text):
        match = pattern.search(text, index)
        if match is None:
            parts.append(text[index:])
            break
        parts.append(text[index : match.start()])
        literal_start = match.end()
        literal_end = _balanced_literal_end(text, literal_start)
        if literal_end is None:
            parts.append(text[match.start() : match.end()])
            index = match.end()
            continue
        literal = text[literal_start:literal_end]
        if _loads_expected_literal(literal, expected):
            parts.append("[extra-body-redacted]")
        else:
            parts.append(text[match.start() : literal_end])
        index = literal_end
    return "".join(parts)


def _balanced_literal_end(text: str, start: int) -> int | None:
    if start >= len(text) or text[start] != "{":
        return None
    stack = ["}"]
    quote: str | None = None
    escaped = False
    for index in range(start + 1, len(text)):
        char = text[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char == "{":
            stack.append("}")
        elif char == "}":
            stack.pop()
            if not stack:
                return index + 1
    return None


def _loads_expected_literal(literal: str, expected: object) -> bool:
    with contextlib.suppress(json.JSONDecodeError, TypeError):
        return json.loads(literal) == expected
    with contextlib.suppress(ValueError, SyntaxError):
        return ast.literal_eval(literal) == expected
    return False


def _redact_model_name(text: str, model: str) -> str:
    model_name = model.strip()
    if not model_name:
        return text
    escaped = re.escape(model_name)
    suffix_boundary = r"(?=$|[\s,;)\]}]|[.:](?:$|[\s,;)\]}'\"]))"
    sanitized = text
    for pattern in (
        rf"(?P<prefix>[\"']model[\"']\s*:\s*[\"'])(?P<value>{escaped})(?P<suffix>[\"'])",
        rf"(?P<prefix>\bmodel\s*[:=]\s*[\"']?)(?P<value>{escaped})(?P<suffix>[\"']?){suffix_boundary}",
        rf"(?P<prefix>\bmodel\s+[\"']?)(?P<value>{escaped})(?P<suffix>[\"']?){suffix_boundary}",
    ):
        sanitized = re.sub(
            pattern,
            lambda match: f"{match.group('prefix')}[model-redacted]{match.group('suffix')}",
            sanitized,
            flags=re.IGNORECASE,
        )
    sanitized = re.sub(
        rf"(?<![{_MODEL_TOKEN_CHARS}]){escaped}(?:(?![{_MODEL_TOKEN_CHARS}])|(?=[.:](?:$|[\s,;)\]}}'\"])))",
        "[model-redacted]",
        sanitized,
        flags=re.IGNORECASE,
    )
    return sanitized


def _redact_request_body_fragments(text: str) -> str:
    sanitized = re.sub(
        r"\b(?:body|json|request_body|payload|messages)\s*[:=]\s*.*",
        "[request-body-redacted]",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return re.sub(
        r"\{(?=[\s\S]*[\"']messages[\"']\s*:)[\s\S]*\}",
        "[request-body-redacted]",
        sanitized,
        flags=re.IGNORECASE,
    )


def _sanitize_error_text(
    text: str,
    *,
    api_key: str = "",
    model: str = "",
    extra_body: Mapping[str, object] | None = None,
    source_text: str = "",
) -> str:
    sanitized = text
    sanitized = re.sub(r"Bearer\s+[^\s,}]+", "Bearer [redacted]", sanitized, flags=re.IGNORECASE)
    if api_key:
        sanitized = sanitized.replace(api_key, "[redacted]")
    if source_text:
        sanitized = sanitized.replace(source_text, "[source-text-redacted]")
    sanitized = _redact_model_name(sanitized, model)
    sanitized = re.sub(
        r"Authorization\s*[:=]\s*(?!Bearer\s+\[redacted\])[^\s,}]+",
        "Authorization: [redacted]",
        sanitized,
        flags=re.IGNORECASE,
    )
    sanitized = re.sub(r"https?://[^\s'\"]+", "[url-redacted]", sanitized)
    sanitized = _redact_request_body_fragments(sanitized)
    sanitized = _redact_extra_body_fragments(sanitized, extra_body)
    return sanitized[:300]


def _has_length_finish_reason(data: object) -> bool:
    if not isinstance(data, dict):
        return False
    choices = data.get("choices")
    if not isinstance(choices, list):
        return False
    return any(
        isinstance(choice, dict) and choice.get("finish_reason") == "length" for choice in choices
    )


class LocalOpenAIClient(Protocol):
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
class HttpxLocalOpenAIClient:
    base_url: str = _DEFAULT_BASE_URL
    model: str = "llama3.1:8b"
    api_key: str = ""
    extra_body: Mapping[str, object] = field(
        default_factory=lambda: {"reasoning_effort": "none", "temperature": 0.6}
    )
    max_tokens: int | None = None
    timeout: httpx.Timeout | float = field(default_factory=_default_timeout)
    runtime_logging: SessionRuntimeLoggingService | None = None
    _client: httpx.AsyncClient | None = field(init=False, default=None, repr=False)
    _client_lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock, repr=False)

    def __post_init__(self) -> None:
        self.base_url = _normalize_base_url(self.base_url)
        if not self.model.strip():
            raise ValueError("invalid local LLM model")
        self.api_key = self.api_key.strip()
        _assert_extra_body_is_safe(self.extra_body)

    def _build_http_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=self.timeout, trust_env=False, follow_redirects=False)

    async def _get_http_client(self) -> httpx.AsyncClient:
        if self._client is not None:
            return self._client
        async with self._client_lock:
            if self._client is None:
                self._client = self._build_http_client()
            return self._client

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        key = self.api_key.strip()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def _build_request_body(
        self,
        *,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str,
    ) -> dict[str, object]:
        _assert_extra_body_is_safe(self.extra_body)
        body: dict[str, object] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": _build_system_prompt(
                        system_prompt=system_prompt,
                        source_language=source_language,
                        target_language=target_language,
                    ),
                },
                {
                    "role": "user",
                    "content": build_translation_user_message(text=text, context=context),
                },
            ],
            "stream": False,
        }
        if self.max_tokens is not None:
            body["max_tokens"] = self.max_tokens
        body.update(dict(self.extra_body))
        return body

    async def translate(
        self,
        *,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
    ) -> str:
        client = await self._get_http_client()
        try:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=self._build_request_body(
                    text=text,
                    system_prompt=system_prompt,
                    source_language=source_language,
                    target_language=target_language,
                    context=context,
                ),
            )
        except asyncio.CancelledError:
            raise
        except httpx.HTTPError as exc:
            message = _sanitize_error_text(
                str(exc),
                api_key=self.api_key,
                model=self.model,
                extra_body=self.extra_body,
                source_text=text,
            )
            raise RuntimeError(f"Local LLM request failed: {message}") from exc

        if response.status_code != 200:
            error_message = ""
            with contextlib.suppress(Exception):
                error_message = _extract_error_message(response.json())
            if not error_message:
                error_message = response.text
            message = _sanitize_error_text(
                error_message or "unknown error",
                api_key=self.api_key,
                model=self.model,
                extra_body=self.extra_body,
                source_text=text,
            )
            raise RuntimeError(
                f"Local LLM request failed (status={response.status_code}, message={message})"
            )

        try:
            data = response.json()
        except Exception as exc:
            raise RuntimeError("Local LLM response was not valid JSON") from exc
        if not isinstance(data, dict):
            raise RuntimeError("Local LLM response was not a JSON object")
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("Local LLM response did not contain choices")
        if _has_length_finish_reason(data):
            raise RuntimeError("Local LLM response was truncated by max_tokens limit")
        first = choices[0]
        if not isinstance(first, dict):
            raise RuntimeError("Local LLM response choice was malformed")
        message = first.get("message")
        if not isinstance(message, dict):
            raise RuntimeError("Local LLM response did not contain message")
        return _extract_message_content(message.get("content"))

    async def close(self) -> None:
        async with self._client_lock:
            client = self._client
            self._client = None
        if client is not None:
            await client.aclose()


@dataclass(slots=True)
class LocalOpenAICompatibleLLMProvider:
    base_url: str = _DEFAULT_BASE_URL
    model: str = "llama3.1:8b"
    api_key: str = ""
    extra_body: Mapping[str, object] = field(
        default_factory=lambda: {"reasoning_effort": "none", "temperature": 0.6}
    )
    max_tokens: int | None = None
    timeout: httpx.Timeout | float = field(default_factory=_default_timeout)
    runtime_logging: SessionRuntimeLoggingService | None = None
    client: LocalOpenAIClient | None = None
    _internal_client: HttpxLocalOpenAIClient | None = field(init=False, default=None, repr=False)
    _external_client_closed: bool = field(init=False, default=False, repr=False)

    def _client_for_call(self) -> LocalOpenAIClient:
        if self.client is not None:
            return self.client
        if self._internal_client is None:
            self._internal_client = HttpxLocalOpenAIClient(
                base_url=self.base_url,
                model=self.model,
                api_key=self.api_key,
                extra_body=self.extra_body,
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
        result = await self._client_for_call().translate(
            text=text,
            system_prompt=system_prompt,
            source_language=source_language,
            target_language=target_language,
            context=context,
        )
        return Translation(
            utterance_id=utterance_id,
            text=result,
            source_text=text,
            source_language=source_language,
            target_language=target_language,
        )

    async def close(self) -> None:
        internal = self._internal_client
        self._internal_client = None
        if internal is not None:
            await internal.close()
        if self.client is not None and not self._external_client_closed:
            self._external_client_closed = True
            await self.client.close()

    @staticmethod
    async def verify_connection(
        *,
        base_url: str = _DEFAULT_BASE_URL,
        model: str = "llama3.1:8b",
        api_key: str = "",
        extra_body: Mapping[str, object] | None = None,
    ) -> bool:
        client: HttpxLocalOpenAIClient | None = None
        try:
            client = HttpxLocalOpenAIClient(
                base_url=base_url,
                model=model,
                api_key=api_key,
                extra_body=(
                    extra_body
                    if extra_body is not None
                    else {"reasoning_effort": "none", "temperature": 0.6}
                ),
                max_tokens=1,
                timeout=httpx.Timeout(connect=3.0, read=10.0, write=5.0, pool=3.0),
            )
            await client.translate(
                text="ping", system_prompt="", source_language="", target_language=""
            )
            return True
        except asyncio.CancelledError:
            raise
        except Exception:
            return False
        finally:
            if client is not None:
                await client.close()
