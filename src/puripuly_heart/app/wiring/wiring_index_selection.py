from __future__ import annotations

import asyncio
import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from puripuly_heart.app.ports.provider_verifier import ProviderVerifierPort
from puripuly_heart.app.services.canonical_settings_persistence import (
    SettingsOwner,
    materialize_canonical_translation_settings,
)
from puripuly_heart.app.services.index_selection import IndexManagedAllowanceObservation
from puripuly_heart.app.wiring.wiring_managed_auth_factory import (
    build_openrouter_credential_runtime_config_from_vnext,
)
from puripuly_heart.app.wiring.wiring_secrets_factory import create_secret_store
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.core.messages import ErrorDiagnostics
from puripuly_heart.core.openrouter_credentials import resolve_openrouter_credentials
from puripuly_heart.core.openrouter_handoff import is_effectively_exhausted
from puripuly_heart.core.owned_thread import run_owned_thread_call
from puripuly_heart.providers.llm.index_translate import IndexTranslateProvider


def gemma_fallback_settings(settings: AppSettingsVNext, connection: str) -> AppSettingsVNext:
    translation = settings.intent.translation
    history = dict(translation.connection_history)
    history["gemma4_26b_31b"] = connection
    return materialize_canonical_translation_settings(
        replace(
            settings,
            intent=replace(
                settings.intent,
                translation=replace(
                    translation,
                    model="gemma4_26b_31b",
                    connection=connection,
                    connection_history=history,
                ),
            ),
        )
    )


async def check_index_availability() -> None:
    provider = IndexTranslateProvider()
    try:
        await provider.check_availability()
    finally:
        await provider.close()


def unknown_managed_allowance(reason: str) -> IndexManagedAllowanceObservation:
    return IndexManagedAllowanceObservation(
        "unknown",
        ErrorDiagnostics(
            component="index_selection",
            operation="managed_allowance",
            code="index_managed_allowance_unknown",
            category="unknown",
            visibility="basic",
            content_policy="metadata_only",
            status_code=None,
            retry_after_ms=None,
            fields={"reason": reason},
        ),
    )


@dataclass(slots=True)
class IndexManagedAllowanceObserver:
    settings: SettingsOwner
    config_path: Path
    verifier: ProviderVerifierPort
    ingress_frozen: Callable[[], bool]
    metadata_timeout_seconds: float = 5.0

    async def destination_credential_available(self) -> bool:
        current = self.settings.canonical
        if current is None or self.ingress_frozen():
            return False

        def resolve() -> bool:
            secrets = create_secret_store(current.intent.secrets, config_path=self.config_path)
            config = build_openrouter_credential_runtime_config_from_vnext(current)
            return bool(resolve_openrouter_credentials(config, secrets=secrets).api_key)

        try:
            available = await run_owned_thread_call(resolve)
        except Exception:
            return False
        live = self.settings.canonical
        return bool(
            available
            and not self.ingress_frozen()
            and live is not None
            and live.intent.translation == current.intent.translation
            and live.state.managed_connection == current.state.managed_connection
            and live.intent.secrets == current.intent.secrets
        )

    async def observe(self) -> IndexManagedAllowanceObservation:
        current = self.settings.canonical
        if current is None or self.ingress_frozen():
            return unknown_managed_allowance("settings_unavailable")
        target = gemma_fallback_settings(current, "managed")
        identity = target.state.managed_connection
        config = build_openrouter_credential_runtime_config_from_vnext(target)

        def read_key() -> str | None:
            secrets = create_secret_store(target.intent.secrets, config_path=self.config_path)
            return resolve_openrouter_credentials(config, secrets=secrets).api_key

        def current_identity() -> bool:
            live = self.settings.canonical
            return (
                live is not None
                and live.state.managed_connection == identity
                and live.intent.secrets == target.intent.secrets
                and not self.ingress_frozen()
            )

        try:
            key = await run_owned_thread_call(read_key)
        except Exception:
            return unknown_managed_allowance("credential_unavailable")
        if not current_identity():
            return unknown_managed_allowance("credential_not_current")
        if not key:
            return IndexManagedAllowanceObservation("missing")
        try:
            async with asyncio.timeout(self.metadata_timeout_seconds):
                metadata = await self.verifier.fetch_openrouter_key_metadata(key)
            latest_key = await run_owned_thread_call(read_key)
        except Exception:
            return unknown_managed_allowance("metadata_unavailable")
        if not current_identity() or latest_key != key:
            return unknown_managed_allowance("credential_not_current")
        remaining = metadata.remaining_usd if metadata is not None else None
        if (
            remaining is None
            or isinstance(remaining, bool)
            or not isinstance(remaining, (int, float))
            or not math.isfinite(remaining)
        ):
            return unknown_managed_allowance("metadata_invalid")

        async def confirm_current() -> bool:
            if not current_identity():
                return False
            try:
                return await run_owned_thread_call(read_key) == key and current_identity()
            except Exception:
                return False

        return IndexManagedAllowanceObservation(
            "exhausted" if is_effectively_exhausted(metadata) else "remaining",
            confirm_current=confirm_current,
        )
