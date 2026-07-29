from __future__ import annotations

from pathlib import Path

from puripuly_heart.app.ports.ui_application import UiApplicationPort
from puripuly_heart.app.ports.ui_presentation import UiPresentationPort
from puripuly_heart.app.ports.vrchat_osc_presence import VrchatOscPresencePort
from puripuly_heart.app.services.canonical_settings_persistence import (
    compose_settings_owner,
)
from puripuly_heart.app.services.provider_settings import (
    ProviderSettingsOwner,
    provider_verification_context,
)
from puripuly_heart.app.services.provider_verification_binding import (
    ProviderVerificationBindingOwner,
)
from puripuly_heart.app.services.ui_application import UiApplicationBoundary
from puripuly_heart.app.wiring import create_secret_store, create_sync_secret_store_adapter
from puripuly_heart.core.runtime_logging import RuntimeLoggingSinks
from puripuly_heart.core.translation_policy import FIXED_TRANSLATION_POLICY
from puripuly_heart.ui.controller import GuiController


def compose_ui_application(
    *,
    presentation: UiPresentationPort,
    config_path: Path,
    allow_stable_settings_import: bool = False,
    runtime_logging_sinks: RuntimeLoggingSinks | None = None,
    vrchat_osc_presence: VrchatOscPresencePort | None = None,
) -> UiApplicationPort:
    settings_owner = compose_settings_owner(config_path)
    provider_settings_owner = ProviderSettingsOwner(
        settings=settings_owner,
        binding=ProviderVerificationBindingOwner(
            context_provider=lambda provider: provider_verification_context(
                settings_owner.current,
                provider,
                low_latency=FIXED_TRANSLATION_POLICY.fast_translation_enabled,
            ),
        ),
        secret_store_factory=lambda settings: create_sync_secret_store_adapter(
            create_secret_store(settings.secrets, config_path=config_path)
        ),
        active_secret_provider=lambda settings, secret_key: create_secret_store(
            settings.secrets,
            config_path=config_path,
        ).get(secret_key),
    )
    backend = GuiController(
        page=None,
        app=presentation,
        config_path=config_path,
        allow_stable_settings_import=allow_stable_settings_import,
        runtime_logging_sinks=runtime_logging_sinks,
        vrchat_osc_presence=vrchat_osc_presence,
        settings_owner=settings_owner,
        provider_settings_owner=provider_settings_owner,
    )
    save_failure_sink = getattr(backend, "_log_error", None)
    if callable(save_failure_sink):
        provider_settings_owner.save_failure_sink = save_failure_sink
    return UiApplicationBoundary(backend)
