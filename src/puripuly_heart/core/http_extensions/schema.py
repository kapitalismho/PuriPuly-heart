from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

from puripuly_heart.core.messages import (
    DIAGNOSTIC_CATEGORY_AUTH,
    DIAGNOSTIC_CATEGORY_INVALID_RESPONSE,
    DIAGNOSTIC_CATEGORY_LIFECYCLE,
    DiagnosticCategory,
)
from puripuly_heart.core.translation_backend import TranslationBackendRequest

type JSONValue = None | bool | int | float | str | list[JSONValue] | dict[str, JSONValue]

_IDENTIFIER_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")
_PLACEHOLDER_PATTERN = re.compile(r"\{\{([^{}]*)\}\}")
_DANGEROUS_HEADERS = {
    "connection",
    "content-length",
    "host",
    "transfer-encoding",
}


class HttpExtensionError(ValueError):
    diagnostic_provider = "custom_http"


class HttpExtensionValidationError(HttpExtensionError):
    pass


class HttpExtensionConfigurationError(HttpExtensionError):
    def __init__(
        self,
        message: str,
        *,
        diagnostic_category: DiagnosticCategory = DIAGNOSTIC_CATEGORY_LIFECYCLE,
    ) -> None:
        self.diagnostic_category = diagnostic_category
        super().__init__(message)


class HttpExtensionResponseError(HttpExtensionError):
    diagnostic_category = DIAGNOSTIC_CATEGORY_INVALID_RESPONSE


@dataclass(frozen=True, slots=True)
class HttpExtensionSecret:
    id: str
    label: str


@dataclass(frozen=True, slots=True)
class HttpExtensionBody:
    type: str
    value: JSONValue | None = None


@dataclass(frozen=True, slots=True)
class HttpExtensionRequest:
    query: dict[str, str]
    body: HttpExtensionBody


@dataclass(frozen=True, slots=True)
class HttpExtensionResponse:
    type: str
    pointer: str | None = None


@dataclass(frozen=True, slots=True)
class HttpExtensionLanguageMap:
    source: dict[str, str]
    target: dict[str, str]


@dataclass(frozen=True, slots=True)
class HttpExtension:
    schema_version: int
    id: str
    name: str
    description: str | None
    url: str
    headers: dict[str, str]
    request: HttpExtensionRequest
    response: HttpExtensionResponse
    secrets: tuple[HttpExtensionSecret, ...]
    language_map: HttpExtensionLanguageMap

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def secret_ids(self) -> frozenset[str]:
        return frozenset(secret.id for secret in self.secrets)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def parse_http_extension(
    value: object,
    *,
    source_path: Path | None = None,
) -> HttpExtension:
    prefix = f"{source_path}: " if source_path is not None else ""
    data = _require_mapping(value, f"{prefix}extension must be a JSON object")
    allowed = {
        "schema_version",
        "id",
        "name",
        "description",
        "url",
        "headers",
        "request",
        "response",
        "secrets",
        "language_map",
    }
    unknown = set(data) - allowed
    if unknown:
        raise HttpExtensionValidationError(
            f"{prefix}unsupported extension field: {sorted(unknown)[0]}"
        )
    schema_version = data.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        raise HttpExtensionValidationError(f"{prefix}schema_version must be 1")

    http_extension_id = _required_text(data, "id", prefix)
    _validate_identifier(http_extension_id, f"{prefix}id")
    name = _required_text(data, "name", prefix)
    description = data.get("description")
    if description is not None and not isinstance(description, str):
        raise HttpExtensionValidationError(f"{prefix}description must be a string")
    url = _required_text(data, "url", prefix)
    _validate_url(url, prefix)

    headers = _string_map(data.get("headers", {}), f"{prefix}headers")
    for header_name in headers:
        if header_name.strip().lower() in _DANGEROUS_HEADERS:
            raise HttpExtensionValidationError(f"{prefix}header is transport-managed")

    request_data = _required_mapping(data, "request", prefix)
    request_unknown = set(request_data) - {"query", "body"}
    if request_unknown:
        raise HttpExtensionValidationError(
            f"{prefix}unsupported request field: {sorted(request_unknown)[0]}"
        )
    query = _string_map(request_data.get("query", {}), f"{prefix}request.query")
    body_data = _required_mapping(request_data, "body", prefix)
    body_unknown = set(body_data) - {"type", "value"}
    if body_unknown:
        raise HttpExtensionValidationError(
            f"{prefix}unsupported request.body field: {sorted(body_unknown)[0]}"
        )
    body_type = body_data.get("type")
    if not isinstance(body_type, str) or body_type not in {"json", "form", "none"}:
        raise HttpExtensionValidationError(f"{prefix}request.body.type must be json, form, or none")
    body_value = body_data.get("value")
    if body_type in {"json", "form"} and "value" not in body_data:
        raise HttpExtensionValidationError(
            f"{prefix}request.body.value is required for {body_type}"
        )
    if body_type == "json":
        _validate_json_value(body_value, f"{prefix}request.body.value")
    elif body_type == "form":
        body_value = _string_map(body_value, f"{prefix}request.body.value")
    else:
        body_value = None

    response_data = _required_mapping(data, "response", prefix)
    response_unknown = set(response_data) - {"type", "pointer"}
    if response_unknown:
        raise HttpExtensionValidationError(
            f"{prefix}unsupported response field: {sorted(response_unknown)[0]}"
        )
    response_type = response_data.get("type")
    if not isinstance(response_type, str) or response_type not in {"json", "text"}:
        raise HttpExtensionValidationError(f"{prefix}response.type must be json or text")
    pointer = response_data.get("pointer")
    if response_type == "json":
        if not isinstance(pointer, str):
            raise HttpExtensionValidationError(
                f"{prefix}response.pointer is required for json responses"
            )
        _validate_json_pointer(pointer, f"{prefix}response.pointer")
    elif pointer is not None:
        raise HttpExtensionValidationError(
            f"{prefix}response.pointer is not used for text responses"
        )

    secrets = _parse_secrets(data.get("secrets", []), prefix)
    language_map = _parse_language_map(data.get("language_map"), prefix)
    extension = HttpExtension(
        schema_version=1,
        id=http_extension_id,
        name=name,
        description=description,
        url=url,
        headers=headers,
        request=HttpExtensionRequest(
            query=query,
            body=HttpExtensionBody(type=body_type, value=body_value),
        ),
        response=HttpExtensionResponse(type=response_type, pointer=pointer),
        secrets=secrets,
        language_map=language_map,
    )
    _validate_placeholders(extension, prefix)
    return extension


def render_translation_request(
    extension: HttpExtension,
    request: TranslationBackendRequest,
    *,
    secrets: Mapping[str, str],
) -> tuple[str, dict[str, str], dict[str, str], JSONValue | None]:
    values = {
        "text": request.text,
        "source_language": extension.language_map.source.get(
            request.source_language,
            request.source_language,
        ),
        "target_language": extension.language_map.target.get(
            request.target_language,
            request.target_language,
        ),
    }
    secret_values: dict[str, str] = {}
    for secret in extension.secrets:
        value = secrets.get(secret.id)
        if value is None or not value.strip():
            raise HttpExtensionConfigurationError(
                f"missing required credential: {secret.label}",
                diagnostic_category=DIAGNOSTIC_CATEGORY_AUTH,
            )
        secret_values[secret.id] = value

    def replace(value: str) -> str:
        def replacement(match: re.Match[str]) -> str:
            token = match.group(1)
            if token.startswith("secret:"):
                return secret_values[token.removeprefix("secret:")]
            return values[token]

        return _PLACEHOLDER_PATTERN.sub(replacement, value)

    rendered_headers = {key: replace(value) for key, value in extension.headers.items()}
    rendered_query = {key: replace(value) for key, value in extension.request.query.items()}
    rendered_body = _render_value(extension.request.body.value, replace)
    return extension.url, rendered_headers, rendered_query, rendered_body


def extract_translation_text(extension: HttpExtension, response_text: str) -> str:
    if extension.response.type == "text":
        translated = response_text.strip()
    else:
        invalid_json = False
        payload: object = None
        try:
            payload = json.loads(response_text)
        except (TypeError, ValueError):
            invalid_json = True
        if invalid_json:
            raise HttpExtensionResponseError("invalid response JSON")
        translated_value = _resolve_json_pointer(payload, extension.response.pointer or "")
        if not isinstance(translated_value, str):
            raise HttpExtensionResponseError("response pointer did not select a string")
        translated = translated_value.strip()
    if not translated:
        raise HttpExtensionResponseError("empty translation response")
    return translated


def resolve_json_pointer(value: object, pointer: str) -> object:
    _validate_json_pointer(pointer, "JSON Pointer")
    return _resolve_json_pointer(value, pointer)


def _resolve_json_pointer(value: object, pointer: str) -> object:
    if pointer == "":
        return value
    current = value
    for raw_token in pointer.split("/")[1:]:
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict):
            if token not in current:
                raise HttpExtensionResponseError("response pointer path is missing")
            current = current[token]
            continue
        if isinstance(current, list):
            if (
                token == "-"
                or not token
                or any(character < "0" or character > "9" for character in token)
                or (len(token) > 1 and token[0] == "0")
            ):
                raise HttpExtensionResponseError("response pointer array index is invalid")
            index = int(token)
            if index >= len(current):
                raise HttpExtensionResponseError("response pointer path is missing")
            current = current[index]
            continue
        raise HttpExtensionResponseError("response pointer path is missing")
    return current


def _parse_secrets(value: object, prefix: str) -> tuple[HttpExtensionSecret, ...]:
    if not isinstance(value, list):
        raise HttpExtensionValidationError(f"{prefix}secrets must be an array")
    result: list[HttpExtensionSecret] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        data = _require_mapping(item, f"{prefix}secrets[{index}] must be an object")
        if set(data) != {"id", "label"}:
            raise HttpExtensionValidationError(
                f"{prefix}secrets[{index}] must contain only id and label"
            )
        secret_id = data.get("id")
        label = data.get("label")
        if not isinstance(secret_id, str) or not secret_id:
            raise HttpExtensionValidationError(f"{prefix}secrets[{index}].id must be a string")
        _validate_identifier(secret_id, f"{prefix}secrets[{index}].id")
        if secret_id in seen:
            raise HttpExtensionValidationError(f"{prefix}duplicate secret id")
        if not isinstance(label, str) or not label.strip():
            raise HttpExtensionValidationError(
                f"{prefix}secrets[{index}].label must be a non-empty string"
            )
        seen.add(secret_id)
        result.append(HttpExtensionSecret(id=secret_id, label=label))
    return tuple(result)


def _parse_language_map(value: object, prefix: str) -> HttpExtensionLanguageMap:
    if value is None:
        return HttpExtensionLanguageMap(source={}, target={})
    data = _require_mapping(value, f"{prefix}language_map must be an object")
    unknown = set(data) - {"source", "target"}
    if unknown:
        raise HttpExtensionValidationError(
            f"{prefix}unsupported language_map field: {sorted(unknown)[0]}"
        )
    return HttpExtensionLanguageMap(
        source=_string_map(data.get("source", {}), f"{prefix}language_map.source"),
        target=_string_map(data.get("target", {}), f"{prefix}language_map.target"),
    )


def _validate_placeholders(extension: HttpExtension, prefix: str) -> None:
    declared = extension.secret_ids
    referenced: set[str] = set()

    def visit(value: object) -> None:
        if isinstance(value, str):
            matches = list(_PLACEHOLDER_PATTERN.finditer(value))
            match_starts = {match.start() for match in matches}
            match_ends = {match.end() - 2 for match in matches}
            if any(
                value[index : index + 2] == "{{"
                and index not in match_starts
                or value[index : index + 2] == "}}"
                and index not in match_ends
                for index in range(len(value) - 1)
            ):
                raise HttpExtensionValidationError(f"{prefix}invalid placeholder syntax")
            for match in _PLACEHOLDER_PATTERN.finditer(value):
                token = match.group(1)
                if token in {"text", "source_language", "target_language"}:
                    continue
                if token.startswith("secret:"):
                    secret_id = token.removeprefix("secret:")
                    if not _IDENTIFIER_PATTERN.fullmatch(secret_id):
                        raise HttpExtensionValidationError(f"{prefix}invalid secret placeholder")
                    if secret_id not in declared:
                        raise HttpExtensionValidationError(
                            f"{prefix}secret placeholder is not declared"
                        )
                    referenced.add(secret_id)
                    continue
                raise HttpExtensionValidationError(f"{prefix}unsupported placeholder")
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)

    visit(extension.headers)
    visit(extension.request.query)
    visit(extension.request.body.value)
    unused = declared - referenced
    if unused:
        raise HttpExtensionValidationError(
            f"{prefix}declared secret is unused: {sorted(unused)[0]}"
        )


def _validate_url(value: str, prefix: str) -> None:
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
    except ValueError as exc:
        raise HttpExtensionValidationError(f"{prefix}url is invalid") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or not hostname:
        raise HttpExtensionValidationError(f"{prefix}url must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise HttpExtensionValidationError(f"{prefix}url must not contain credentials")
    if "{{" in parsed.scheme or "}}" in parsed.scheme or "{{" in parsed.netloc:
        raise HttpExtensionValidationError(f"{prefix}url host cannot contain placeholders")
    if "{{" in value or "}}" in value:
        raise HttpExtensionValidationError(f"{prefix}url cannot contain placeholders")


def _validate_identifier(value: str, field: str) -> None:
    if _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise HttpExtensionValidationError(f"{field} has an invalid identifier")


def _validate_json_pointer(value: str, field: str) -> None:
    if value and not value.startswith("/"):
        raise HttpExtensionValidationError(f"{field} must start with / or be empty")
    for token in value.split("/")[1:]:
        for index, character in enumerate(token):
            if character == "~" and (index + 1 >= len(token) or token[index + 1] not in {"0", "1"}):
                raise HttpExtensionValidationError(f"{field} contains invalid escape")


def _validate_json_value(value: object, field: str) -> None:
    if value is None or isinstance(value, str | bool | int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise HttpExtensionValidationError(f"{field} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{field}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise HttpExtensionValidationError(f"{field} keys must be strings")
            _validate_json_value(item, f"{field}.{key}")
        return
    raise HttpExtensionValidationError(f"{field} contains an unsupported JSON value")


def _render_value(value: JSONValue | None, replace: Any) -> JSONValue | None:
    if isinstance(value, str):
        return replace(value)
    if isinstance(value, list):
        return [_render_value(item, replace) for item in value]
    if isinstance(value, dict):
        return {key: _render_value(item, replace) for key, item in value.items()}
    return value


def _required_text(data: Mapping[str, object], key: str, prefix: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise HttpExtensionValidationError(f"{prefix}{key} must be a non-empty string")
    return value


def _required_mapping(data: Mapping[str, object], key: str, prefix: str) -> dict[str, object]:
    if key not in data:
        raise HttpExtensionValidationError(f"{prefix}{key} is required")
    return _require_mapping(data[key], f"{prefix}{key} must be an object")


def _require_mapping(value: object, message: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise HttpExtensionValidationError(message)
    return dict(value)


def _string_map(value: object, field: str) -> dict[str, str]:
    data = _require_mapping(value, f"{field} must be an object")
    result: dict[str, str] = {}
    for key, item in data.items():
        if not key or not isinstance(item, str):
            raise HttpExtensionValidationError(f"{field} must be a string-to-string map")
        result[key] = item
    return result


__all__ = [
    "JSONValue",
    "HttpExtension",
    "HttpExtensionBody",
    "HttpExtensionConfigurationError",
    "HttpExtensionError",
    "HttpExtensionLanguageMap",
    "HttpExtensionRequest",
    "HttpExtensionResponse",
    "HttpExtensionResponseError",
    "HttpExtensionSecret",
    "HttpExtensionValidationError",
    "extract_translation_text",
    "parse_http_extension",
    "render_translation_request",
    "resolve_json_pointer",
]
