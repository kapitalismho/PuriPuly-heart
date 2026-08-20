from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Final, Mapping
from urllib.parse import quote, urlsplit, urlunsplit

CUSTOM_STT_PROVIDER: Final = "custom"
_CUSTOM_STT_SECRET_GENERATION = 0
CUSTOM_STT_MODE_OFFLINE: Final = "offline"
CUSTOM_STT_MODE_REALTIME: Final = "realtime"
CUSTOM_STT_MODES: Final[tuple[str, ...]] = (
    CUSTOM_STT_MODE_OFFLINE,
    CUSTOM_STT_MODE_REALTIME,
)

CUSTOM_STT_COMPAT_OPENAI_TRANSCRIPTION: Final = "openai_transcription"
CUSTOM_STT_COMPAT_OPENAI_REALTIME: Final = "openai_realtime"
CUSTOM_STT_COMPATIBILITIES: Final[tuple[str, ...]] = (
    CUSTOM_STT_COMPAT_OPENAI_TRANSCRIPTION,
    CUSTOM_STT_COMPAT_OPENAI_REALTIME,
)

CUSTOM_STT_CAPABILITY_LANGUAGE_HINT: Final = "language_hint"

CUSTOM_STT_VALIDATION_UNREACHABLE: Final = "unreachable"
CUSTOM_STT_VALIDATION_AUTH_FAILURE: Final = "auth_failure"
CUSTOM_STT_VALIDATION_COMPATIBILITY_MISMATCH: Final = "compatibility_mismatch"
CUSTOM_STT_VALIDATION_MODEL_UNAVAILABLE: Final = "model_unavailable"
CUSTOM_STT_VALIDATION_TRANSCRIPTION_UNVERIFIED: Final = "transcription_unverified"
CUSTOM_STT_VALIDATION_READY: Final = "ready"

CUSTOM_STT_VALIDATION_STATUSES: Final[tuple[str, ...]] = (
    CUSTOM_STT_VALIDATION_UNREACHABLE,
    CUSTOM_STT_VALIDATION_AUTH_FAILURE,
    CUSTOM_STT_VALIDATION_COMPATIBILITY_MISMATCH,
    CUSTOM_STT_VALIDATION_MODEL_UNAVAILABLE,
    CUSTOM_STT_VALIDATION_TRANSCRIPTION_UNVERIFIED,
    CUSTOM_STT_VALIDATION_READY,
)

CUSTOM_STT_VALIDATION_I18N_KEYS: Final[Mapping[str, str]] = {
    CUSTOM_STT_VALIDATION_UNREACHABLE: "settings.custom_stt.validation.unreachable",
    CUSTOM_STT_VALIDATION_AUTH_FAILURE: "settings.custom_stt.validation.auth_failure",
    CUSTOM_STT_VALIDATION_COMPATIBILITY_MISMATCH: (
        "settings.custom_stt.validation.compatibility_mismatch"
    ),
    CUSTOM_STT_VALIDATION_MODEL_UNAVAILABLE: "settings.custom_stt.validation.model_unavailable",
    CUSTOM_STT_VALIDATION_TRANSCRIPTION_UNVERIFIED: (
        "settings.custom_stt.validation.transcription_unverified"
    ),
    CUSTOM_STT_VALIDATION_READY: "settings.custom_stt.validation.ready",
}

_DEFAULT_COMPATIBILITY_BY_MODE: Final[Mapping[str, str]] = {
    CUSTOM_STT_MODE_OFFLINE: CUSTOM_STT_COMPAT_OPENAI_TRANSCRIPTION,
    CUSTOM_STT_MODE_REALTIME: CUSTOM_STT_COMPAT_OPENAI_REALTIME,
}
_SUPPORTED_COMPATIBILITIES_BY_MODE: Final[Mapping[str, frozenset[str]]] = {
    CUSTOM_STT_MODE_OFFLINE: frozenset({CUSTOM_STT_COMPAT_OPENAI_TRANSCRIPTION}),
    CUSTOM_STT_MODE_REALTIME: frozenset({CUSTOM_STT_COMPAT_OPENAI_REALTIME}),
}
_CAPABILITIES_BY_COMPATIBILITY: Final[Mapping[str, frozenset[str]]] = {
    CUSTOM_STT_COMPAT_OPENAI_TRANSCRIPTION: frozenset({CUSTOM_STT_CAPABILITY_LANGUAGE_HINT}),
    CUSTOM_STT_COMPAT_OPENAI_REALTIME: frozenset({CUSTOM_STT_CAPABILITY_LANGUAGE_HINT}),
}

_SECRET_REDACTION: Final = "[redacted]"
_BEARER_RE: Final = re.compile(r"(?i)\bBearer\s+\S+")
_QUERY_SECRET_RE: Final = re.compile(
    r"(?i)([?&](?:api[_-]?key|token|access_token|auth|authorization|secret)=)[^&]+"
)
_OPENAI_STYLE_SECRET_RE: Final = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b")
_EMBEDDED_USERINFO_RE: Final = re.compile(r"(?i)((?:https?|wss?)://)[^/\s:@]+(?::[^/\s@]*)?@")

CUSTOM_STT_RESERVED_EXTRA_KEYS: Final[frozenset[str]] = frozenset(
    {
        # 전송 구조와 충돌하는 키 (offline form / realtime JSON)
        "file",
        "type",
        "session",
        "audio",
        "input_audio_format",
        "input_audio_transcription",
        "turn_detection",
    }
)
CUSTOM_STT_SENSITIVE_EXTRA_KEYS: Final[frozenset[str]] = frozenset(
    {"api_key", "authorization", "headers", "token", "secret", "password"}
)


class CustomSTTConfigurationError(ValueError):
    """Raised when Custom STT settings violate the supported product contract."""


@dataclass(frozen=True, slots=True)
class CustomSTTConnectionValidation:
    status: str
    message: str
    endpoint: str

    def __post_init__(self) -> None:
        if self.status not in CUSTOM_STT_VALIDATION_STATUSES:
            raise ValueError(f"unsupported Custom STT validation status: {self.status}")


def bump_custom_stt_secret_generation() -> int:
    global _CUSTOM_STT_SECRET_GENERATION
    _CUSTOM_STT_SECRET_GENERATION += 1
    return _CUSTOM_STT_SECRET_GENERATION


def custom_stt_secret_generation() -> int:
    return _CUSTOM_STT_SECRET_GENERATION


def default_compatibility_for_mode(mode: str) -> str:
    try:
        return _DEFAULT_COMPATIBILITY_BY_MODE[mode]
    except KeyError as exc:
        raise CustomSTTConfigurationError(f"unsupported Custom STT mode: {mode}") from exc


def supported_compatibilities_for_mode(mode: str) -> frozenset[str]:
    try:
        return _SUPPORTED_COMPATIBILITIES_BY_MODE[mode]
    except KeyError as exc:
        raise CustomSTTConfigurationError(f"unsupported Custom STT mode: {mode}") from exc


def capabilities_for_compatibility(compatibility: str) -> frozenset[str]:
    try:
        return _CAPABILITIES_BY_COMPATIBILITY[compatibility]
    except KeyError as exc:
        raise CustomSTTConfigurationError(
            f"unsupported Custom STT compatibility: {compatibility}"
        ) from exc


def compatibility_supports(compatibility: str, capability: str) -> bool:
    return capability in capabilities_for_compatibility(compatibility)


def validate_mode_compatibility(mode: str, compatibility: str) -> None:
    allowed = supported_compatibilities_for_mode(mode)
    if compatibility not in allowed:
        raise CustomSTTConfigurationError(
            f"Custom STT compatibility {compatibility} is not supported in {mode} mode"
        )


def normalize_custom_stt_mode(value: object, *, default: str = CUSTOM_STT_MODE_OFFLINE) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"streaming", "realtime", "real-time"}:
            return CUSTOM_STT_MODE_REALTIME
        if normalized in CUSTOM_STT_MODES:
            return normalized
    return default


def normalize_custom_stt_compatibility(
    value: object,
    *,
    mode: str,
    default: str | None = None,
) -> str:
    allowed = supported_compatibilities_for_mode(mode)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in allowed:
            return normalized
        if normalized:
            raise CustomSTTConfigurationError(
                f"Custom STT compatibility {normalized} is not supported in {mode} mode"
            )
    return default if default is not None else default_compatibility_for_mode(mode)


def normalize_custom_stt_endpoint(value: object) -> str:
    if not isinstance(value, str):
        return ""
    raw = value.strip()
    if not raw:
        return ""
    try:
        parsed = urlsplit(raw)
        _ = parsed.port
    except ValueError:
        return ""
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        if not parsed.scheme or not parsed.hostname:
            return ""
        netloc = f"{parsed.hostname}:{parsed.port}" if parsed.port else parsed.hostname
        return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
    return raw


def normalize_custom_stt_model(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def normalize_custom_stt_extra(value: object) -> dict[str, object]:
    """Normalize the free-form Custom STT extra JSON mapping.

    Values must be JSON-serializable scalars, mappings, or lists. Keys must be
    strings and must not collide with reserved transport keys or carry secrets.
    """
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise CustomSTTConfigurationError("Custom STT extra must be a JSON object")
    normalized: dict[str, object] = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str) or not raw_key.strip():
            raise CustomSTTConfigurationError("Custom STT extra keys must be non-empty strings")
        key = raw_key.strip().lower()
        if key in CUSTOM_STT_SENSITIVE_EXTRA_KEYS:
            raise CustomSTTConfigurationError(f"sensitive Custom STT extra key: {key}")
        if key in CUSTOM_STT_RESERVED_EXTRA_KEYS:
            raise CustomSTTConfigurationError(f"reserved Custom STT extra key: {key}")
        normalized[raw_key.strip()] = _copy_custom_stt_extra_value(raw_value)
    try:
        json.dumps(normalized, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise CustomSTTConfigurationError("Custom STT extra must be JSON serializable") from exc
    return normalized


def _copy_custom_stt_extra_value(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        copied: dict[str, object] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise CustomSTTConfigurationError("Custom STT extra mapping keys must be strings")
            copied[key] = _copy_custom_stt_extra_value(child)
        return copied
    if isinstance(value, list):
        return [_copy_custom_stt_extra_value(item) for item in value]
    raise CustomSTTConfigurationError(
        "Custom STT extra values must be JSON-like scalars, mappings, or lists"
    )


def language_hint_for_source(source_language: str) -> str | None:
    normalized = source_language.strip()
    if not normalized:
        return None
    primary = normalized.split("-", 1)[0].split("_", 1)[0].lower()
    if not re.fullmatch(r"[a-z]{2,3}", primary):
        return None
    return primary


def sanitize_endpoint_for_display(endpoint: str) -> str:
    raw = endpoint.strip()
    if not raw:
        return ""
    try:
        parsed = urlsplit(raw)
        port = parsed.port
    except ValueError:
        return sanitize_custom_stt_text(raw) or _SECRET_REDACTION
    if parsed.username or parsed.password:
        if parsed.scheme and parsed.hostname:
            netloc = f"{parsed.hostname}:{port}" if port else parsed.hostname
            return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
        return sanitize_custom_stt_text(raw) or _SECRET_REDACTION
    if not parsed.scheme or not parsed.hostname:
        return sanitize_custom_stt_text(raw)
    host = parsed.hostname
    netloc = f"{host}:{port}" if port else host
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def sanitize_custom_stt_text(value: str, *, secret: str = "") -> str:
    text = value.replace("\r\n", "\n").replace("\r", "\n")
    if secret:
        text = text.replace(secret, _SECRET_REDACTION)
    text = _BEARER_RE.sub(f"Bearer {_SECRET_REDACTION}", text)
    text = _OPENAI_STYLE_SECRET_RE.sub(_SECRET_REDACTION, text)
    text = _QUERY_SECRET_RE.sub(rf"\1{_SECRET_REDACTION}", text)
    text = _EMBEDDED_USERINFO_RE.sub(r"\1", text)
    try:
        parsed = urlsplit(text)
    except ValueError:
        parsed = None
    if parsed is not None and parsed.hostname and (parsed.username or parsed.password):
        host = parsed.hostname
        netloc = f"{host}:{parsed.port}" if parsed.port else host
        text = urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))
    return text


def classify_http_failure(status_code: int | None, body_excerpt: str = "") -> str:
    if status_code is None:
        return CUSTOM_STT_VALIDATION_UNREACHABLE
    if status_code in {401, 403}:
        return CUSTOM_STT_VALIDATION_AUTH_FAILURE
    lowered = body_excerpt.lower()
    if status_code == 400 and "model" in lowered:
        return CUSTOM_STT_VALIDATION_MODEL_UNAVAILABLE
    if "unknown model" in lowered or "model_not_found" in lowered:
        return CUSTOM_STT_VALIDATION_MODEL_UNAVAILABLE
    if status_code == 404 and "model" in lowered:
        return CUSTOM_STT_VALIDATION_MODEL_UNAVAILABLE
    if status_code == 404:
        return CUSTOM_STT_VALIDATION_COMPATIBILITY_MISMATCH
    if status_code >= 500:
        return CUSTOM_STT_VALIDATION_UNREACHABLE
    return CUSTOM_STT_VALIDATION_TRANSCRIPTION_UNVERIFIED


def resolve_openai_transcription_url(endpoint: str) -> str:
    parsed = _require_http_endpoint(endpoint, allowed_schemes=("http", "https"))
    path = parsed.path.rstrip("/")
    if path.endswith("/audio/transcriptions"):
        resolved_path = path
    elif path.endswith("/v1"):
        resolved_path = f"{path}/audio/transcriptions"
    elif path:
        resolved_path = f"{path}/v1/audio/transcriptions"
    else:
        resolved_path = "/v1/audio/transcriptions"
    return urlunsplit((parsed.scheme, parsed.netloc, resolved_path, parsed.query, ""))


def resolve_openai_realtime_url(endpoint: str) -> str:
    parsed = _require_http_endpoint(endpoint, allowed_schemes=("http", "https", "ws", "wss"))
    scheme = {"http": "ws", "https": "wss"}.get(parsed.scheme, parsed.scheme)
    path = parsed.path.rstrip("/")
    if path.endswith("/realtime"):
        resolved_path = path
    elif path.endswith("/v1"):
        resolved_path = f"{path}/realtime"
    elif path:
        resolved_path = f"{path}/v1/realtime"
    else:
        resolved_path = "/v1/realtime"
    return urlunsplit((scheme, parsed.netloc, resolved_path, parsed.query, ""))


def append_custom_stt_query(endpoint: str, extra: Mapping[str, object]) -> str:
    """Append free-form extra entries as query parameters to an endpoint URL.

    String values are appended as plain text; other JSON values are serialized
    so structured values survive round-trips. Empty extra mappings return the
    endpoint unchanged.
    """
    if not extra:
        return endpoint
    parsed = urlsplit(endpoint)
    pairs: list[str] = []
    for key, value in extra.items():
        if isinstance(value, str):
            serialized = value
        else:
            serialized = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        pairs.append(f"{quote(str(key), safe='')}={quote(serialized, safe='')}")
    existing = parsed.query
    joined = "&".join(pairs)
    query = f"{existing}&{joined}" if existing else joined
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, query, parsed.fragment))


def _require_http_endpoint(endpoint: str, *, allowed_schemes: tuple[str, ...]) -> urlsplit:
    raw = endpoint.strip()
    if not raw:
        raise CustomSTTConfigurationError("Custom STT endpoint is required")
    try:
        parsed = urlsplit(raw)
        _ = parsed.port
    except ValueError as exc:
        raise CustomSTTConfigurationError("Custom STT endpoint is invalid") from exc
    if parsed.scheme not in allowed_schemes or not parsed.hostname:
        raise CustomSTTConfigurationError("Custom STT endpoint is invalid")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise CustomSTTConfigurationError("Custom STT endpoint is invalid")
    return parsed
