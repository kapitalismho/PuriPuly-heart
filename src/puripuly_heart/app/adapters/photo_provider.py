from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass

from puripuly_heart.app.ports.settings_secrets import SettingsSecretStorePort
from puripuly_heart.core.chatgpt.session import ChatGptSession
from puripuly_heart.core.owned_thread import run_owned_thread_call
from puripuly_heart.core.photo.lifecycle import PhotoRoute
from puripuly_heart.providers.photo import make_photo_connection


@dataclass(slots=True)
class PhotoProviderAdapter:
    session: ChatGptSession
    secret_store: Callable[[], SettingsSecretStorePort]

    async def resolve_route(self) -> PhotoRoute | None:
        return await run_owned_thread_call(self._observe_route)

    def _observe_route(self) -> PhotoRoute | None:
        if self.session.status().signed_in is True:
            return PhotoRoute(
                "luna",
                lambda model: make_photo_connection(
                    route="luna", chatgpt_session=self.session, model=model
                ),
            )
        try:
            key = self.secret_store().get("gemini_api_key")
        except Exception:
            return None
        if not key:
            return None
        return PhotoRoute(
            "google-" + hashlib.sha256(key.encode()).hexdigest(),
            lambda model: make_photo_connection(route="google", gemini_api_key=key, model=model),
        )
