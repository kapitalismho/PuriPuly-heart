from __future__ import annotations

import threading
from typing import Any

import httpx
from websockets.asyncio.client import connect

from .external_network import (
    ProxyPolicy,
    TLSSelection,
    annotate_connection_error,
    connection_diagnostic_fields,
    select_tls,
    validate_http_proxy,
)


class _HTTPResponseStream(httpx.SyncByteStream):
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


class _AsyncHTTPResponseStream(httpx.AsyncByteStream):
    def __init__(self, stream, *, url, tls, route, status) -> None:
        self.stream = stream
        self.url, self.tls, self.route, self.status = url, tls, route, status

    async def __aiter__(self):
        try:
            async for chunk in self.stream:
                yield chunk
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

    async def aclose(self) -> None:
        await self.stream.aclose()


def _annotate_response(response, *, request, tls, route, stream_type):
    response.stream = stream_type(
        response.stream, url=str(request.url), tls=tls, route=route, status=response.status_code
    )
    response.extensions["connection_diagnostics"] = {
        **connection_diagnostic_fields(url=str(request.url), tls=tls, route=route),
        "stage": "response_headers",
        "dispatch_certainty": "response_received",
        "http_status": response.status_code,
    }
    return response


class _HTTPTransport(httpx.BaseTransport):
    def __init__(self, tls: TLSSelection, policy: ProxyPolicy, **kwargs: Any) -> None:
        self.tls = tls
        self.policy = policy
        self.kwargs = kwargs
        self.transports: dict[str | None, httpx.HTTPTransport] = {}
        self.lock = threading.Lock()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        route = self.policy.route(str(request.url))
        try:
            validate_http_proxy(route)
            with self.lock:
                transport = self.transports.get(route.url)
                if transport is None:
                    proxy = (
                        httpx.Proxy(
                            route.url,
                            ssl_context=(
                                self.tls.context if route.url.startswith("https:") else None
                            ),
                        )
                        if route.url
                        else None
                    )
                    transport = httpx.HTTPTransport(
                        verify=self.tls.context, proxy=proxy, trust_env=False, **self.kwargs
                    )
                    self.transports[route.url] = transport
            return _annotate_response(
                transport.handle_request(request),
                request=request,
                tls=self.tls,
                route=route,
                stream_type=_HTTPResponseStream,
            )
        except Exception as exc:
            annotate_connection_error(exc, url=str(request.url), tls=self.tls, route=route)
            raise

    def close(self) -> None:
        for transport in self.transports.values():
            transport.close()


class _AsyncHTTPTransport(httpx.AsyncBaseTransport):
    def __init__(self, tls: TLSSelection, policy: ProxyPolicy, **kwargs: Any) -> None:
        self.tls = tls
        self.policy = policy
        self.kwargs = kwargs
        self.transports: dict[str | None, httpx.AsyncHTTPTransport] = {}

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        route = self.policy.route(str(request.url))
        try:
            validate_http_proxy(route)
            transport = self.transports.get(route.url)
            if transport is None:
                proxy = (
                    httpx.Proxy(
                        route.url,
                        ssl_context=self.tls.context if route.url.startswith("https:") else None,
                    )
                    if route.url
                    else None
                )
                transport = httpx.AsyncHTTPTransport(
                    verify=self.tls.context, proxy=proxy, trust_env=False, **self.kwargs
                )
                self.transports[route.url] = transport
            return _annotate_response(
                await transport.handle_async_request(request),
                request=request,
                tls=self.tls,
                route=route,
                stream_type=_AsyncHTTPResponseStream,
            )
        except Exception as exc:
            annotate_connection_error(exc, url=str(request.url), tls=self.tls, route=route)
            raise

    async def aclose(self) -> None:
        for transport in self.transports.values():
            await transport.aclose()


def _client_options(
    kwargs: dict[str, Any],
    *,
    asynchronous: bool,
    tls: TLSSelection | None,
    policy: ProxyPolicy | None,
) -> dict[str, Any]:
    if kwargs.get("transport") is not None or kwargs.get("trust_env") is False:
        return kwargs
    tls = tls or select_tls()
    policy = policy or ProxyPolicy()
    kwargs["trust_env"] = False
    transport_options = {
        key: kwargs.pop(key) for key in ("limits", "http1", "http2") if key in kwargs
    }
    transport_type = _AsyncHTTPTransport if asynchronous else _HTTPTransport
    kwargs["transport"] = transport_type(tls, policy, **transport_options)
    return kwargs


def external_async_client(
    *, tls: TLSSelection | None = None, policy: ProxyPolicy | None = None, **kwargs: Any
) -> httpx.AsyncClient:
    options = _client_options(kwargs, asynchronous=True, tls=tls, policy=policy)
    return httpx.AsyncClient(**options)


def external_client(
    *, tls: TLSSelection | None = None, policy: ProxyPolicy | None = None, **kwargs: Any
) -> httpx.Client:
    options = _client_options(kwargs, asynchronous=False, tls=tls, policy=policy)
    return httpx.Client(**options)


class _PolicyWebSocketConnect(connect):
    def __init__(self, url: str, tls: TLSSelection, policy: ProxyPolicy, **kwargs: Any) -> None:
        self.tls = tls
        self.policy = policy
        self.route = policy.route(url)
        kwargs["proxy"] = self.route.url
        super().__init__(url, **kwargs)

    async def create_connection(self) -> Any:
        self.route = self.policy.route(self.uri)
        self.proxy = self.route.url
        if self.uri.startswith("wss:"):
            self.connection_kwargs["ssl"] = self.tls.context
        if self.route.url and self.route.url.startswith("https:"):
            self.connection_kwargs["proxy_ssl"] = self.tls.context
        else:
            self.connection_kwargs.pop("proxy_ssl", None)
        return await super().create_connection()


class ExternalWebSocketConnect:
    def __init__(self, url: str, **kwargs: Any) -> None:
        self.tls = select_tls()
        self.connect = _PolicyWebSocketConnect(url, self.tls, ProxyPolicy(), **kwargs)

    async def _open(self) -> Any:
        try:
            return await self.connect
        except Exception as exc:
            annotate_connection_error(
                exc, url=self.connect.uri, tls=self.tls, route=self.connect.route
            )
            raise

    def __await__(self):
        return self._open().__await__()

    async def __aenter__(self) -> Any:
        self.connection = await self._open()
        return self.connection

    async def __aexit__(self, *args: Any) -> None:
        await self.connection.close()


def external_websocket_connect(url: str, **kwargs: Any) -> ExternalWebSocketConnect:
    return ExternalWebSocketConnect(url, **kwargs)


def genai_http_options(
    *,
    sync_transport: Any = None,
    async_transport: Any = None,
) -> dict[str, Any]:
    transport = getattr(sync_transport, "_transport", None)
    if isinstance(transport, _HTTPTransport):
        tls, policy = transport.tls, transport.policy
    else:
        tls, policy = select_tls(), ProxyPolicy()
    owns_sync = sync_transport is None
    if sync_transport is None:
        sync_transport = external_client(
            tls=tls, policy=policy, timeout=None, follow_redirects=True
        )
    try:
        if async_transport is None:
            async_transport = external_async_client(
                tls=tls, policy=policy, timeout=None, follow_redirects=True
            )
    except BaseException:
        if owns_sync:
            sync_transport.close()
        raise
    return {
        "client_args": {"verify": tls.context, "trust_env": False},
        "async_client_args": {"verify": tls.context, "ssl": tls.context, "trust_env": False},
        "httpx_client": sync_transport,
        "httpx_async_client": async_transport,
    }
