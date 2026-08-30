from __future__ import annotations

from dataclasses import replace

import pytest
from puripuly_heart.app.services.settings_application import (
    materialize_immediate_settings_intent,
    materialize_prompt_apply_intent,
    materialize_provider_apply_intent,
    settings_view_surface_snapshots,
)

from puripuly_heart.app.ports.settings_view import (
    DesktopOverlayPositionResetIntent,
    DesktopOverlaySizeIntent,
    PromptApplyIntent,
)
from puripuly_heart.app.services.canonical_settings_persistence import (
    materialize_canonical_translation_settings,
)
from puripuly_heart.config.settings import AppSettings
from puripuly_heart.config.settings_vnext.migration import from_legacy_app_settings
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.ui.i18n import get_locale, set_locale
from puripuly_heart.ui.views.settings import SettingsView


def _canonical_settings(settings: object) -> object:
    if isinstance(settings, AppSettingsVNext):
        return settings
    if isinstance(settings, AppSettings):
        return from_legacy_app_settings(
            settings,
            preserve_provider_verification=True,
        )
    return settings


def _with_resolved_prompt(canonical: object, prompt_snapshot: object) -> object:
    if not isinstance(canonical, AppSettingsVNext) or prompt_snapshot is None:
        return canonical
    system_prompt = getattr(prompt_snapshot, "system_prompt", None)
    if not isinstance(system_prompt, str):
        return canonical
    return replace(
        canonical,
        intent=replace(
            canonical.intent,
            prompts=replace(canonical.intent.prompts, system_prompt=system_prompt),
        ),
    )


@pytest.fixture(autouse=True)
def restore_locale_after_test():
    previous_locale = get_locale()
    try:
        yield
    finally:
        set_locale(previous_locale)


@pytest.fixture(autouse=True)
def settings_view_typed_boundary_adapter(monkeypatch: pytest.MonkeyPatch):
    load = SettingsView.load_from_settings
    refresh = SettingsView.refresh_after_openrouter_pkce_success
    build_provider = SettingsView.build_provider_apply_settings
    consume_prompt = SettingsView.consume_prompt_apply_settings

    def set_compatibility_settings(view, settings):
        canonical = _canonical_settings(settings)
        view._test_compatibility_settings = canonical
        if canonical is not None:
            provider, general, prompt, overlay = settings_view_surface_snapshots(canonical)
            view._provider_snapshot = provider
            view._provider_draft = None
            view._provider_edits = {}
            view._general_snapshot = general
            view._prompt_snapshot = prompt
            view._overlay_snapshot = overlay
            if hasattr(view, "_prompt_editor") and prompt.system_prompt:
                view._prompt_editor.value = prompt.system_prompt

    def get_compatibility_settings(view):
        return getattr(view, "_test_compatibility_settings", None)

    def set_provider_draft(view, settings):
        if settings is None:
            view._provider_draft = None
            return
        provider, _general, _prompt, _overlay = settings_view_surface_snapshots(
            _canonical_settings(settings)
        )
        view._provider_draft = provider

    def get_provider_draft(view):
        current = get_compatibility_settings(view)
        intent = build_provider(view)
        if current is None or intent is None:
            return None
        return materialize_provider_apply_intent(
            current,
            intent,
            materialize_translation=materialize_canonical_translation_settings,
        )

    def load_adapter(
        view,
        settings=None,
        *,
        provider=None,
        general=None,
        prompt=None,
        overlay=None,
        config_path,
        preserve_custom_vocab_draft=False,
    ):
        if settings is not None:
            canonical = _canonical_settings(settings)
            provider, general, prompt, overlay = settings_view_surface_snapshots(canonical)
            view._test_compatibility_settings = canonical
        result = load(
            view,
            provider=provider,
            general=general,
            prompt=prompt,
            overlay=overlay,
            config_path=config_path,
            preserve_custom_vocab_draft=preserve_custom_vocab_draft,
        )
        if settings is not None:
            view._test_compatibility_settings = _with_resolved_prompt(
                view._test_compatibility_settings,
                view._prompt_snapshot,
            )
        return result

    def refresh_adapter(
        view,
        settings=None,
        *,
        provider=None,
        prompt=None,
        config_path,
    ):
        if settings is not None:
            canonical = _canonical_settings(settings)
            provider, _general, prompt, _overlay = settings_view_surface_snapshots(canonical)
            view._test_compatibility_settings = canonical
        return refresh(
            view,
            provider=provider,
            prompt=prompt,
            config_path=config_path,
        )

    def build_provider_adapter(view):
        current = get_compatibility_settings(view)
        intent = build_provider(view)
        if current is None or intent is None:
            return None
        return materialize_provider_apply_intent(
            current,
            intent,
            materialize_translation=materialize_canonical_translation_settings,
        )

    def consume_prompt_adapter(view):
        current = get_compatibility_settings(view)
        intent = consume_prompt(view)
        if current is None or intent is None:
            return None
        updated = materialize_prompt_apply_intent(current, intent)
        view._test_compatibility_settings = updated
        return updated

    def emit_settings_adapter(view, intent):
        current = get_compatibility_settings(view)
        if current is None:
            return
        pending_size = getattr(view, "_desktop_overlay_pending_size_preset", None)
        if pending_size is not None:
            current = materialize_immediate_settings_intent(
                current,
                DesktopOverlaySizeIntent(pending_size),
            )
        if getattr(view, "_desktop_overlay_pending_position_reset", False):
            current = materialize_immediate_settings_intent(
                current,
                DesktopOverlayPositionResetIntent(),
            )
        updated = materialize_immediate_settings_intent(current, intent)
        view._test_compatibility_settings = updated
        if view.on_settings_changed:
            view.on_settings_changed(updated)

    def emit_prompt_adapter(view, intent):
        current = get_compatibility_settings(view)
        if current is None:
            return
        if isinstance(intent, PromptApplyIntent):
            updated = materialize_prompt_apply_intent(current, intent)
        elif isinstance(intent, AppSettingsVNext):
            updated = intent
        else:
            updated = materialize_prompt_apply_intent(current, intent)
        view._test_compatibility_settings = updated
        if view.on_prompt_apply_settings:
            view.on_prompt_apply_settings(updated)
        elif view.on_settings_changed:
            view.on_settings_changed(updated)

    monkeypatch.setattr(SettingsView, "load_from_settings", load_adapter)
    monkeypatch.setattr(SettingsView, "refresh_after_openrouter_pkce_success", refresh_adapter)
    monkeypatch.setattr(
        SettingsView,
        "_settings",
        property(get_compatibility_settings, set_compatibility_settings),
        raising=False,
    )
    monkeypatch.setattr(
        SettingsView,
        "_provider_settings_draft",
        property(get_provider_draft, set_provider_draft),
        raising=False,
    )
    monkeypatch.setattr(SettingsView, "build_provider_apply_settings", build_provider_adapter)
    monkeypatch.setattr(SettingsView, "consume_prompt_apply_settings", consume_prompt_adapter)
    monkeypatch.setattr(SettingsView, "_emit_settings_changed", emit_settings_adapter)
    monkeypatch.setattr(SettingsView, "_emit_prompt_apply_settings", emit_prompt_adapter)
