from __future__ import annotations

import os
import ssl
import sys
import urllib.request
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit


class NetworkConfigurationError(OSError):
    def __init__(self, code: str, *, fields: Mapping[str, str] | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.diagnostic_transport_fields = {
            "configuration_error": code,
            "stage": "client_configuration",
            "dispatch_certainty": "not_dispatched",
            **(fields or {}),
        }


@dataclass(frozen=True)
class TLSSelection:
    context: ssl.SSLContext
    source: str
    backend: str


def select_tls(
    *, requests: bool = False, environment: Mapping[str, str] | None = None
) -> TLSSelection:
    environment = os.environ if environment is None else environment
    choices = (
        (("REQUESTS_CA_BUNDLE", "requests_bundle"), ("CURL_CA_BUNDLE", "curl_bundle"))
        if requests
        else (("SSL_CERT_FILE", "explicit_file"), ("SSL_CERT_DIR", "explicit_directory"))
    )
    for name, source in choices:
        value = environment.get(name)
        if not value:
            continue
        try:
            path = Path(value)
            directory = source == "explicit_directory" or (requests and path.is_dir())
            if directory:
                if not path.is_dir():
                    raise FileNotFoundError
                context = ssl.create_default_context(capath=value)
            else:
                context = ssl.create_default_context(cafile=value)
        except (OSError, ValueError) as exc:
            raise NetworkConfigurationError(
                "invalid_tls_configuration",
                fields={"tls_source": source, "tls_backend": "openssl"},
            ) from exc
        return TLSSelection(context, source, "openssl")
    if sys.platform == "win32":
        import truststore

        return TLSSelection(truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT), "windows", "native")
    return TLSSelection(ssl.create_default_context(), "unknown", "openssl")


def _windows_proxy_settings() -> tuple[dict[str, str], str]:
    if sys.platform != "win32":
        return {}, ""
    proxies = urllib.request.getproxies_registry()
    import winreg

    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Internet Settings",
        ) as key:
            if not winreg.QueryValueEx(key, "ProxyEnable")[0]:
                return {}, ""
            try:
                bypass = str(winreg.QueryValueEx(key, "ProxyOverride")[0])
            except FileNotFoundError:
                bypass = ""
    except OSError:
        return proxies, ""
    return proxies, bypass


@dataclass(frozen=True)
class ProxyRoute:
    url: str | None
    source: str


class ProxyPolicy:
    def __init__(
        self,
        *,
        environment: Mapping[str, str] | None = None,
        system: Mapping[str, str] | None = None,
        system_bypass: str | None = None,
    ) -> None:
        if environment is None:
            self.environment = urllib.request.getproxies_environment()
        else:
            proxies: dict[str, str] = {}
            for name, value in environment.items():
                if name.lower().endswith("_proxy") and value:
                    proxies[name[:-6].lower()] = value
            for name, value in environment.items():
                if name.endswith("_proxy"):
                    if value:
                        proxies[name[:-6]] = value
                    else:
                        proxies.pop(name[:-6], None)
            self.environment = proxies
        if system is None:
            self.system, bypass = _windows_proxy_settings()
        else:
            self.system, bypass = dict(system), ""
        self.system_bypass = bypass if system_bypass is None else system_bypass
        self.environment = {
            name: value if name == "no" or "://" in value else f"http://{value}"
            for name, value in self.environment.items()
        }
        self.system = {
            name: value if "://" in value else f"http://{value}"
            for name, value in self.system.items()
        }

    def route(self, url: str) -> ProxyRoute:
        parsed = urlsplit(url)
        scheme = parsed.scheme.lower()
        host = parsed.hostname or ""
        port = parsed.port or {"http": 80, "https": 443, "ws": 80, "wss": 443}.get(scheme)
        authority = f"{host}:{port}" if port else host
        if urllib.request.proxy_bypass_environment(authority, self.environment):
            return ProxyRoute(None, "bypass")
        protocol = "https" if scheme == "wss" else "http" if scheme == "ws" else scheme
        keys = (scheme, "socks", protocol, "all") if scheme in ("ws", "wss") else (protocol, "all")
        for key in keys:
            if self.environment.get(key):
                return ProxyRoute(self.environment[key], "env")
        proxy = self.system.get(protocol) or self.system.get("all")
        if proxy:
            for pattern in self.system_bypass.split(";"):
                pattern = pattern.strip().lower()
                if pattern == "<local>" and "." not in host:
                    return ProxyRoute(None, "bypass")
                if pattern and fnmatch(host.lower(), pattern):
                    return ProxyRoute(None, "bypass")
            return ProxyRoute(proxy, "system")
        return ProxyRoute(None, "direct")


def validate_http_proxy(route: ProxyRoute) -> None:
    if route.url is None:
        return
    try:
        proxy = urlsplit(route.url)
        if proxy.scheme.lower() not in {"http", "https", "socks5", "socks5h"}:
            raise NetworkConfigurationError("unsupported_proxy_configuration")
        if not proxy.hostname or proxy.port is None and proxy.netloc.endswith(":"):
            raise ValueError
    except ValueError as exc:
        raise NetworkConfigurationError("invalid_proxy_configuration") from exc


def connection_diagnostic_fields(
    *, url: str, tls: TLSSelection, route: ProxyRoute
) -> dict[str, str]:
    scheme = urlsplit(url).scheme.lower()
    secure = scheme in ("https", "wss") or bool(
        route.url and route.url.lower().startswith("https:")
    )
    return {
        "transport": scheme if scheme in ("http", "https", "ws", "wss") else "unknown",
        "tls_source": tls.source if secure else "none",
        "tls_backend": tls.backend if secure else "none",
        "proxy_source": route.source,
    }


def annotate_connection_error(
    exc: Exception,
    *,
    url: str,
    tls: TLSSelection,
    route: ProxyRoute,
    stage: str = "request_send",
    dispatch_certainty: str = "possible",
    http_status: int | None = None,
) -> None:
    exc.connection_diagnostics = connection_diagnostic_fields(url=url, tls=tls, route=route)
    fields = dict(getattr(exc, "diagnostic_transport_fields", {}))
    fields.setdefault("stage", stage)
    fields.setdefault("dispatch_certainty", dispatch_certainty)
    if http_status is not None:
        fields["http_status"] = http_status
    exc.diagnostic_transport_fields = fields
