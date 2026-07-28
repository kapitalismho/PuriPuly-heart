from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from puripuly_heart.core.openrouter_pkce import OpenRouterPKCEExchangeResult
from puripuly_heart.core.runtime.oauth import OAuthRuntime


@dataclass(slots=True)
class OpenRouterPkceFlowOwner:
    client_factory: Callable[[], object]
    runtime_factory: Callable[[], OAuthRuntime] = OAuthRuntime
    _runtime: OAuthRuntime | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _active_client: object | None = field(
        init=False,
        default=None,
        repr=False,
    )

    @property
    def runtime(self) -> OAuthRuntime | None:
        return self._runtime

    @runtime.setter
    def runtime(self, runtime: OAuthRuntime | None) -> None:
        self._runtime = runtime

    @property
    def active_client(self) -> object | None:
        return self._active_client

    @active_client.setter
    def active_client(self, client: object | None) -> None:
        self._active_client = client

    def get_runtime(self) -> OAuthRuntime:
        if self._runtime is None:
            self._runtime = self.runtime_factory()
        return self._runtime

    async def run_flow(self) -> OpenRouterPKCEExchangeResult:
        client = self.client_factory()
        self._active_client = client
        try:
            return await self.get_runtime().run_openrouter_pkce_flow(client)
        finally:
            if self._active_client is client:
                self._active_client = None

    def reopen_authorization_url(self) -> bool:
        runtime = self._runtime
        if runtime is not None and runtime.reopen_openrouter_pkce_authorization_url():
            return True
        client = self._active_client
        if client is None:
            return False
        reopen = getattr(client, "reopen_authorization_url", None)
        return bool(reopen()) if callable(reopen) else False

    async def close(self) -> None:
        runtime = self._runtime
        try:
            if runtime is not None:
                await runtime.close()
        finally:
            self._active_client = None


__all__ = ["OpenRouterPkceFlowOwner"]
