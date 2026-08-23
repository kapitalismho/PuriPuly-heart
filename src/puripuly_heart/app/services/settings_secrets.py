from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass, replace

from puripuly_heart.app.ports.settings_secrets import (
    OpenRouterPkceSecretSnapshot,
    SettingsSecretLoadResult,
    SettingsSecretMutation,
    SettingsSecretMutationResult,
    SettingsSecretSnapshot,
    SettingsSecretStorePort,
)

SettingsSecretStoreFactory = Callable[[], SettingsSecretStorePort]


def _load_value(
    store: SettingsSecretStorePort,
    key: str,
    *,
    legacy_keys: tuple[str, ...] = (),
) -> str:
    value = store.get(key) or ""
    if value or not legacy_keys:
        return value
    for legacy_key in legacy_keys:
        legacy_value = store.get(legacy_key) or ""
        if legacy_value:
            with contextlib.suppress(Exception):
                store.set(key, legacy_value)
            return legacy_value
    return ""


@dataclass(frozen=True, slots=True)
class SettingsSecretsOwner:
    secret_store_factory: SettingsSecretStoreFactory

    def load(self) -> SettingsSecretLoadResult[SettingsSecretSnapshot]:
        try:
            store = self.secret_store_factory()
        except Exception as exc:
            return SettingsSecretLoadResult(
                snapshot=None,
                error_message=f"Failed to load secrets: {exc}",
            )

        snapshot = SettingsSecretSnapshot()
        try:
            snapshot = replace(
                snapshot,
                google_api_key=store.get("google_api_key") or "",
            )
            snapshot = replace(
                snapshot,
                openrouter_api_key=store.get("openrouter_api_key") or "",
            )
            snapshot = replace(
                snapshot,
                deepseek_api_key=store.get("deepseek_api_key") or "",
            )
            snapshot = replace(
                snapshot,
                cerebras_api_key=store.get("cerebras_api_key") or "",
            )
            snapshot = replace(
                snapshot,
                deepgram_api_key=store.get("deepgram_api_key") or "",
            )
            snapshot = replace(
                snapshot,
                soniox_api_key=store.get("soniox_api_key") or "",
            )
            snapshot = replace(
                snapshot,
                local_llm_api_key=store.get("local_llm_api_key") or "",
            )
            snapshot = replace(
                snapshot,
                custom_stt_api_key=store.get("custom_stt_api_key") or "",
            )
            snapshot = replace(
                snapshot,
                alibaba_api_key_beijing=_load_value(
                    store,
                    "alibaba_api_key_beijing",
                    legacy_keys=("alibaba_api_key",),
                ),
            )
            snapshot = replace(
                snapshot,
                alibaba_api_key_singapore=_load_value(
                    store,
                    "alibaba_api_key_singapore",
                    legacy_keys=("alibaba_api_key",),
                ),
            )
        except Exception as exc:
            return SettingsSecretLoadResult(snapshot=snapshot, read_error=exc)
        return SettingsSecretLoadResult(snapshot=snapshot)

    def load_openrouter_pkce(
        self,
    ) -> SettingsSecretLoadResult[OpenRouterPkceSecretSnapshot]:
        try:
            store = self.secret_store_factory()
        except Exception as exc:
            return SettingsSecretLoadResult(
                snapshot=None,
                error_message=f"Failed to load secrets: {exc}",
            )

        snapshot = OpenRouterPkceSecretSnapshot()
        try:
            snapshot = replace(
                snapshot,
                openrouter_api_key=store.get("openrouter_api_key") or "",
            )
            snapshot = replace(
                snapshot,
                deepseek_api_key=store.get("deepseek_api_key") or "",
            )
            snapshot = replace(
                snapshot,
                cerebras_api_key=store.get("cerebras_api_key") or "",
            )
        except Exception as exc:
            return SettingsSecretLoadResult(snapshot=snapshot, read_error=exc)
        return SettingsSecretLoadResult(snapshot=snapshot)

    def mutate(self, mutation: SettingsSecretMutation) -> SettingsSecretMutationResult:
        try:
            store = self.secret_store_factory()
            if mutation.value:
                store.set(mutation.key, mutation.value)
            else:
                store.delete(mutation.key)
        except Exception as exc:
            return SettingsSecretMutationResult(
                succeeded=False,
                error_type=type(exc).__name__,
            )
        return SettingsSecretMutationResult(succeeded=True)


__all__ = ["SettingsSecretStoreFactory", "SettingsSecretsOwner"]
