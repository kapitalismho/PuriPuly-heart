from __future__ import annotations

from functools import partial
from typing import Any

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import InvalidStatus, SecurityError

from puripuly_heart.core.external_network import ProxyPolicy, annotate_connection_error


class _LiveRedirectGuard(ClientConnection):
    def __init__(self, *args: Any, policy: ProxyPolicy, **kwargs: Any) -> None:
        self._policy = policy
        super().__init__(*args, **kwargs)

    async def handshake(self, *args: Any, **kwargs: Any) -> None:
        try:
            await super().handshake(*args, **kwargs)
        except InvalidStatus as exc:
            uri = self.protocol.uri
            host = f"[{uri.host}]" if ":" in uri.host else uri.host
            url = f"{'wss' if uri.secure else 'ws'}://{host}:{uri.port}{uri.resource_name}"
            redirected = connect(url).process_redirect(exc)
            if (
                isinstance(redirected, str)
                and self._policy.route(redirected).url != self._policy.route(url).url
            ):
                raise SecurityError("WebSocket redirect changes the selected proxy route") from None
            raise


def configure_live_network(client: Any, http_options: Any) -> None:
    transport = http_options.httpx_client._transport
    url = client._api_client._websocket_base_url()
    route = transport.policy.route(url)
    contexts = client._api_client._websocket_ssl_ctx
    contexts["ssl"] = transport.tls.context
    contexts["proxy"] = route.url
    contexts["create_connection"] = partial(_LiveRedirectGuard, policy=transport.policy)
    if route.url and route.url.startswith("https:"):
        contexts["proxy_ssl"] = transport.tls.context
    client._puripuly_live_network = (url, transport.tls, route)


def annotate_live_error(exc: Exception, client: Any) -> None:
    context = getattr(client, "_puripuly_live_network", None)
    if context is not None:
        url, tls, route = context
        annotate_connection_error(exc, url=url, tls=tls, route=route)
