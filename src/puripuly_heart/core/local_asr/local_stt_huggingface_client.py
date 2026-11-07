from __future__ import annotations

import threading
from typing import Any

import httpx2

from puripuly_heart.core.external_network import (
    ProxyPolicy,
    TLSSelection,
    annotate_connection_error,
    connection_diagnostic_fields,
    select_tls,
    validate_http_proxy,
)


class _ResponseStream(httpx2.SyncByteStream):
    def __init__(self, stream, *, url, tls, route, status) -> None:
        self.stream = stream
        self.url, self.tls, self.route, self.status = url, tls, route, status

    def __iter__(self):
        try:
            yield from self.stream
        except Exception as exc:
            annotate_connection_error(
                exc,
                url=self.url,
                tls=self.tls,
                route=self.route,
                stage="response_body",
                dispatch_certainty="response_received",
                http_status=self.status,
            )
            raise

    def close(self) -> None:
        self.stream.close()


class _HubTransport(httpx2.BaseTransport):
    def __init__(self, tls: TLSSelection, policy: ProxyPolicy) -> None:
        self.tls = tls
        self.policy = policy
        self.transports: dict[str | None, httpx2.HTTPTransport] = {}
        self.lock = threading.Lock()

    def handle_request(self, request: httpx2.Request) -> httpx2.Response:
        url = str(request.url)
        route = self.policy.route(url)
        try:
            validate_http_proxy(route)
            with self.lock:
                transport = self.transports.get(route.url)
                if transport is None:
                    proxy = (
                        httpx2.Proxy(
                            route.url,
                            ssl_context=(
                                self.tls.context if route.url.startswith("https:") else None
                            ),
                        )
                        if route.url
                        else None
                    )
                    transport = httpx2.HTTPTransport(
                        verify=self.tls.context, proxy=proxy, trust_env=False
                    )
                    self.transports[route.url] = transport
            response = transport.handle_request(request)
            response.stream = _ResponseStream(
                response.stream, url=url, tls=self.tls, route=route, status=response.status_code
            )
            response.extensions["connection_diagnostics"] = {
                **connection_diagnostic_fields(url=url, tls=self.tls, route=route),
                "stage": "response_headers",
                "dispatch_certainty": "response_received",
                "http_status": response.status_code,
            }
            return response
        except Exception as exc:
            annotate_connection_error(exc, url=url, tls=self.tls, route=route)
            raise

    def close(self) -> None:
        for transport in self.transports.values():
            transport.close()


def huggingface_client(*, policy: ProxyPolicy, **kwargs: Any) -> httpx2.Client:
    return httpx2.Client(transport=_HubTransport(select_tls(), policy), trust_env=False, **kwargs)
