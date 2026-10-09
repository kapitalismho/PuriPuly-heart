from __future__ import annotations

import ssl
from collections import deque
from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

import httpx
from requests import exceptions as requests_errors
from urllib3 import exceptions as urllib3_errors

from puripuly_heart.core.messages import DiagnosticFieldValue

TransportErrorKind: TypeAlias = Literal["tls", "proxy", "timeout", "network"]
TRANSPORT_ENUM_FIELDS: Final = MappingProxyType({
    "transport": frozenset({"http", "https", "ws", "wss", "unknown"}),
    "tls_source": frozenset({"windows", "explicit_file", "explicit_directory", "requests_bundle", "curl_bundle", "none", "unknown"}),
    "proxy_source": frozenset({"env", "system", "direct", "bypass", "explicit", "unknown"}),
    "tls_backend": frozenset({"native", "openssl", "none", "unknown"}),
})
TRANSPORT_NUMERIC_FIELDS: Final = ("tls_verify_code", "os_errno", "winerror")
TRANSPORT_FIELD_KEYS: Final = (*TRANSPORT_ENUM_FIELDS, *TRANSPORT_NUMERIC_FIELDS)


def _attribute(value: object, name: str) -> object:
    try:
        return getattr(value, name, None)
    except Exception:
        return None


def _mapping_value(value: Mapping[object, object], name: str) -> object:
    try:
        return value.get(name)
    except Exception:
        return None


def transport_exception_chain(exc: BaseException | None) -> Iterator[BaseException]:
    if not isinstance(exc, BaseException):
        return
    pending = deque([exc])
    seen = {id(exc)}
    while pending:
        current = pending.popleft()
        yield current
        children = (
            _attribute(current, "__cause__"),
            _attribute(current, "__context__"),
            _attribute(current, "reason"),
        )
        args = _attribute(current, "args")
        if type(args) is tuple:
            children += args[:32]
        for child in children:
            if isinstance(child, BaseException) and id(child) not in seen and len(seen) < 32:
                seen.add(id(child))
                pending.append(child)


def classify_transport_error(exc: BaseException) -> TransportErrorKind | None:
    fallback: TransportErrorKind | None = None
    for item in transport_exception_chain(exc):
        if isinstance(item, (ssl.SSLError, requests_errors.SSLError, urllib3_errors.SSLError)):
            return "tls"
        if isinstance(item, (httpx.ProxyError, requests_errors.ProxyError, urllib3_errors.ProxyError)):
            fallback = "proxy"
        elif isinstance(item, urllib3_errors.NewConnectionError):
            if fallback is None:
                fallback = "network"
        elif isinstance(item, (TimeoutError, httpx.TimeoutException, requests_errors.Timeout, urllib3_errors.TimeoutError)):
            if fallback != "proxy":
                fallback = "timeout"
        elif isinstance(item, (httpx.TransportError, ConnectionError, OSError, urllib3_errors.ProtocolError)):
            if fallback is None:
                fallback = "network"
    return fallback


def safe_transport_fields(exc: BaseException) -> dict[str, DiagnosticFieldValue]:
    fields: dict[str, DiagnosticFieldValue] = {key: "unknown" for key in TRANSPORT_ENUM_FIELDS}
    for item in transport_exception_chain(exc):
        metadata = _attribute(item, "connection_diagnostics")
        if isinstance(metadata, Mapping):
            for key, allowed in TRANSPORT_ENUM_FIELDS.items():
                value = _mapping_value(metadata, key)
                if type(value) is str and value in allowed and fields[key] == "unknown":
                    fields[key] = value
        normalized = _attribute(item, "diagnostic_transport_fields")
        if isinstance(normalized, Mapping):
            for key in TRANSPORT_FIELD_KEYS:
                value = _mapping_value(normalized, key)
                if key in TRANSPORT_ENUM_FIELDS:
                    if type(value) is str and value in TRANSPORT_ENUM_FIELDS[key] and fields[key] == "unknown":
                        fields[key] = value
                elif type(value) is int:
                    fields.setdefault(key, value)
        if isinstance(item, OSError) and not isinstance(
            item, (ssl.SSLError, requests_errors.SSLError)
        ):
            for source, key in (("errno", "os_errno"), ("winerror", "winerror")):
                value = _attribute(item, source)
                if type(value) is int:
                    fields.setdefault(key, value)
        if isinstance(item, ssl.SSLCertVerificationError):
            value = _attribute(item, "verify_code")
            if type(value) is int:
                fields.setdefault("tls_verify_code", value)
    return fields
