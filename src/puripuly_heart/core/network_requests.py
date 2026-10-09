from __future__ import annotations

from typing import Any

import requests
from requests.adapters import HTTPAdapter
from requests.auth import _basic_auth_str
from requests.utils import get_auth_from_url

from .external_network import ProxyPolicy, TLSSelection, annotate_connection_error, select_tls


class _TLSAdapter(HTTPAdapter):
    def __init__(self, tls: TLSSelection) -> None:
        self.tls = tls
        super().__init__()

    def build_connection_pool_key_attributes(self, request, verify, cert=None):
        host, options = super().build_connection_pool_key_attributes(request, True, cert)
        options.pop("ca_certs", None)
        options.pop("ca_cert_dir", None)
        options["ssl_context"] = self.tls.context
        return host, options

    def cert_verify(self, conn, url, verify, cert):
        if cert:
            super().cert_verify(conn, url, True, cert)
            conn.ca_certs = None
            conn.ca_cert_dir = None

    def proxy_manager_for(self, proxy, **kwargs):
        if proxy.startswith("https:"):
            kwargs["proxy_ssl_context"] = self.tls.context
        return super().proxy_manager_for(proxy, **kwargs)


class ExternalRequestsSession(requests.Session):
    def __init__(self) -> None:
        super().__init__()
        self.tls = select_tls(requests=True)
        self.policy = ProxyPolicy()
        self.mount("https://", _TLSAdapter(self.tls))
        self.mount("http://", _TLSAdapter(self.tls))

    def merge_environment_settings(self, url, proxies, stream, verify, cert):
        settings = super().merge_environment_settings(url, proxies, stream, verify, cert)
        route = self.policy.route(url)
        settings["proxies"] = {"all": route.url} if route.url else {}
        settings["verify"] = True
        return settings

    def rebuild_proxies(self, prepared_request, proxies):
        route = self.policy.route(prepared_request.url)
        prepared_request.headers.pop("Proxy-Authorization", None)
        if route.url and not prepared_request.url.startswith("https:"):
            username, password = get_auth_from_url(route.url)
            if username and password:
                prepared_request.headers["Proxy-Authorization"] = _basic_auth_str(username, password)
        return {"all": route.url} if route.url else {}

    def send(self, request, **kwargs: Any):
        route = self.policy.route(request.url)
        kwargs["proxies"] = {"all": route.url} if route.url else {}
        kwargs["verify"] = True
        try:
            return super().send(request, **kwargs)
        except Exception as exc:
            annotate_connection_error(exc, url=request.url, tls=self.tls, route=route)
            raise
