from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, replace

from puripuly_heart.app.ports.settings_secrets import (
    OpenRouterPkceSecretSnapshot,
    SettingsSecretLoadResult,
    SettingsSecretMutation,
    SettingsSecretMutationResult,
    SettingsSecretSnapshot,
    SettingsSecretStorePort,
    SettingsSecretValuesSnapshot,
)
from puripuly_heart.core.diagnostics import emit_failure, safe_exception_fields
from puripuly_heart.core.messages import ErrorDiagnostics

logger = logging.getLogger(__name__)


def _secret_diagnostics(phase: str, exception: BaseException) -> ErrorDiagnostics:
    diagnostics = ErrorDiagnostics(
        component="settings_secrets",
        operation=phase,
        code="settings_secret_failed",
        category="transaction",
        visibility="basic",
        content_policy="metadata_only",
        status_code=None,
        retry_after_ms=None,
        fields={"phase": phase, **safe_exception_fields(exception)},
    )
    emit_failure(logger, diagnostics, disposition="degraded", severity="warning", stage=phase)
    return diagnostics


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
            try:
                store.set(key, legacy_value)
            except Exception as exc:
                _secret_diagnostics("legacy_secret_migration", exc)
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
                error_message="Secret storage is unavailable.",
                diagnostics=_secret_diagnostics("store_creation", exc),
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
                openai_api_key=store.get("openai_api_key") or "",
            )
            snapshot = replace(
                snapshot,
                deepseek_api_key=store.get("deepseek_api_key") or "",
            )
            snapshot = replace(
                snapshot,
                deepgram_api_key=store.get("deepgram_api_key") or "",
            )
            snapshot = replace(
                snapshot,
                gemini_transcribe_api_key=store.get("gemini_transcribe_api_key") or "",
            )
            snapshot = replace(
                snapshot,
                elevenlabs_scribe_api_key=store.get("elevenlabs_scribe_api_key") or "",
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
            return SettingsSecretLoadResult(
                snapshot=snapshot, diagnostics=_secret_diagnostics("secret_read", exc)
            )
        return SettingsSecretLoadResult(snapshot=snapshot)

    def load_values(
        self,
        keys: tuple[str, ...],
    ) -> SettingsSecretLoadResult[SettingsSecretValuesSnapshot]:
        try:
            store = self.secret_store_factory()
        except Exception as exc:
            return SettingsSecretLoadResult(
                snapshot=None,
                error_message="Secret storage is unavailable.",
                diagnostics=_secret_diagnostics("store_creation", exc),
            )

        values: list[tuple[str, str]] = []
        snapshot = SettingsSecretValuesSnapshot()
        try:
            for key in keys:
                values.append((key, store.get(key) or ""))
                snapshot = SettingsSecretValuesSnapshot(tuple(values))
        except Exception as exc:
            return SettingsSecretLoadResult(
                snapshot=snapshot, diagnostics=_secret_diagnostics("secret_read", exc)
            )
        return SettingsSecretLoadResult(snapshot=snapshot)

    def load_openrouter_pkce(
        self,
    ) -> SettingsSecretLoadResult[OpenRouterPkceSecretSnapshot]:
        try:
            store = self.secret_store_factory()
        except Exception as exc:
            return SettingsSecretLoadResult(
                snapshot=None,
                error_message="Secret storage is unavailable.",
                diagnostics=_secret_diagnostics("store_creation", exc),
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
        except Exception as exc:
            return SettingsSecretLoadResult(
                snapshot=snapshot, diagnostics=_secret_diagnostics("secret_read", exc)
            )
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
                diagnostics=_secret_diagnostics("secret_mutation", exc),
            )
        return SettingsSecretMutationResult(succeeded=True)


__all__ = ["SettingsSecretStoreFactory", "SettingsSecretsOwner"]
