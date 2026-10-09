from __future__ import annotations

from typing import Any

from puripuly_heart.core.external_network import annotate_connection_error


def configure_live_network(client: Any, http_options: Any) -> None:
    transport = http_options.client_args["transport"]
    url = client._api_client._websocket_base_url()
    route = transport.policy.route(url)
    contexts = client._api_client._websocket_ssl_ctx
    contexts["ssl"] = transport.tls.context
    contexts["proxy"] = route.url
    if route.url and route.url.startswith("https:"):
        contexts["proxy_ssl"] = transport.tls.context
    client._puripuly_live_network = (url, transport.tls, route)


def annotate_live_error(exc: Exception, client: Any) -> None:
    context = getattr(client, "_puripuly_live_network", None)
    if context is not None:
        url, tls, route = context
        annotate_connection_error(exc, url=url, tls=tls, route=route)
