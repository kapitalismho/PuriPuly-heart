from __future__ import annotations

import ast
import pathlib

import flet as ft
import pytest

pytest.importorskip("flet")

import puripuly_heart
from puripuly_heart.ui.foundation.tokens import FOUNDATION_DESIGN_TOKENS
from puripuly_heart.ui.settings.contract import (
    SettingsApiSurfaceSlots,
    SettingsProviderIntents,
    SettingsSurfaceIntents,
)
from puripuly_heart.ui.settings.renderer import (
    SETTINGS_API_GPU_PLACEHOLDER_COUNT,
    SETTINGS_ROW_SPACING,
    compose_settings_api_surface,
)
from puripuly_heart.ui.views import settings as settings_view_module

SOURCE_ROOT = pathlib.Path(puripuly_heart.__file__).resolve().parent

GUI_CONTROLLER_ONLY_PUSHES = (
    "set_managed_trial_usage_state",
    "set_local_cpu_auto_available",
    "refresh_loopback_capture_target",
)

DUAL_DRIVER_PUSHES = (
    "load_from_settings",
    "set_managed_key_state",
    "set_gpu_devices",
    "set_overlay_calibration",
)


class _SlotProvider:
    def __init__(self) -> None:
        self.controls = {
            name: ft.Text(name)
            for name in (
                "self_stt",
                "peer_stt",
                "translation_provider",
                "translation_connection",
                "translation_fallback",
                "gpu_device",
                "local_llm_connection",
                "managed_key",
                "peer_expected_language",
                "api_keys",
            )
        }

    def self_stt_control(self) -> ft.Control:
        return self.controls["self_stt"]

    def peer_stt_control(self) -> ft.Control:
        return self.controls["peer_stt"]

    def translation_provider_control(self) -> ft.Control:
        return self.controls["translation_provider"]

    def translation_connection_control(self) -> ft.Control:
        return self.controls["translation_connection"]

    def translation_fallback_control(self) -> ft.Control:
        return self.controls["translation_fallback"]

    def gpu_device_control(self) -> ft.Control:
        return self.controls["gpu_device"]

    def local_llm_connection_control(self) -> ft.Control:
        return self.controls["local_llm_connection"]

    def managed_key_control(self) -> ft.Control:
        return self.controls["managed_key"]

    def peer_expected_language_control(self) -> ft.Control:
        return self.controls["peer_expected_language"]

    def api_keys_control(self) -> ft.Control:
        return self.controls["api_keys"]


def _compose() -> tuple[object, _SlotProvider, list[ft.Control]]:
    provider = _SlotProvider()
    placeholders: list[ft.Control] = []

    def placeholder_factory() -> ft.Control:
        control = ft.Container()
        placeholders.append(control)
        return control

    surface = compose_settings_api_surface(
        SettingsApiSurfaceSlots.from_slot_provider(provider),
        placeholder_factory=placeholder_factory,
    )
    return surface, provider, placeholders


def test_settings_api_rows_use_the_shared_page_spacing_token() -> None:
    assert SETTINGS_ROW_SPACING == FOUNDATION_DESIGN_TOKENS.spacing.page

    surface, _, _ = _compose()
    for row in (
        surface.provider_controls,
        surface.translation_connection_controls,
        surface.gpu_device_controls,
    ):
        assert row.spacing == FOUNDATION_DESIGN_TOKENS.spacing.page
        assert row.expand is True


def test_settings_api_surface_preserves_the_accepted_row_order() -> None:
    surface, provider, _ = _compose()

    assert surface.rows == (
        surface.provider_row,
        surface.translation_connection_row,
        surface.gpu_device_row,
        provider.controls["local_llm_connection"],
        provider.controls["managed_key"],
        provider.controls["peer_expected_language"],
        provider.controls["api_keys"],
    )


def test_settings_api_surface_places_every_slot_in_the_accepted_position() -> None:
    surface, provider, placeholders = _compose()

    assert len(placeholders) == 1 + SETTINGS_API_GPU_PLACEHOLDER_COUNT
    assert surface.provider_controls.controls == [
        provider.controls["self_stt"],
        provider.controls["peer_stt"],
        provider.controls["translation_provider"],
    ]
    assert surface.translation_connection_controls.controls == [
        surface.translation_connection_leading_placeholder,
        provider.controls["translation_connection"],
        provider.controls["translation_fallback"],
    ]
    assert surface.gpu_device_controls.controls == [
        provider.controls["gpu_device"],
        *surface.gpu_device_placeholders,
    ]


def test_settings_api_surface_preserves_accepted_initial_row_visibility() -> None:
    surface, _, _ = _compose()

    assert surface.provider_row.visible is True
    assert surface.translation_connection_row.visible is True
    assert surface.gpu_device_row.visible is False


def test_bind_settings_intents_carries_every_previously_ad_hoc_g14_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings_view_module.SettingsView, "_populate_host_apis", lambda self: None)
    monkeypatch.setattr(
        settings_view_module.SettingsView, "_refresh_microphones", lambda self: None
    )
    monkeypatch.setattr(settings_view_module.SettingsView, "update", lambda self: None)
    monkeypatch.setattr(
        settings_view_module,
        "create_secret_store",
        lambda *_args, **_kwargs: None,
    )
    view = settings_view_module.SettingsView()

    def make(tag: str):
        def _sentinel(*_args, **_kwargs):
            return tag

        return _sentinel

    surface = SettingsSurfaceIntents(
        settings_changed=make("settings_changed"),
        show_snackbar=make("show_snackbar"),
        runtime_log_basic=make("runtime_log_basic"),
        runtime_log_detailed=make("runtime_log_detailed"),
    )
    provider = SettingsProviderIntents(
        providers_changed=make("providers_changed"),
        request_openrouter_pkce=make("request_openrouter_pkce"),
        verify_api_key=make("verify_api_key"),
        provider_secret_change=make("provider_secret_change"),
        secret_cleared=make("secret_cleared"),
        local_llm_secret_changed=make("local_llm_secret_changed"),
        gpu_discovery_requested=make("gpu_discovery_requested"),
    )

    view.bind_settings_intents(surface=surface, provider=provider)

    assert view.on_settings_changed is surface.settings_changed
    assert view.show_snackbar is surface.show_snackbar
    assert view.runtime_log_basic is surface.runtime_log_basic
    assert view.runtime_log_detailed is surface.runtime_log_detailed
    assert view.on_providers_changed is provider.providers_changed
    assert view.on_request_openrouter_pkce is provider.request_openrouter_pkce
    assert view.on_verify_api_key is provider.verify_api_key
    assert view.on_provider_secret_change is provider.provider_secret_change
    assert view.on_secret_cleared is provider.secret_cleared
    assert view.on_local_llm_secret_changed is provider.local_llm_secret_changed
    assert view.on_gpu_discovery_requested is provider.gpu_discovery_requested


def test_bind_settings_intents_keeps_optional_presentation_sinks_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings_view_module.SettingsView, "_populate_host_apis", lambda self: None)
    monkeypatch.setattr(
        settings_view_module.SettingsView, "_refresh_microphones", lambda self: None
    )
    monkeypatch.setattr(settings_view_module.SettingsView, "update", lambda self: None)
    monkeypatch.setattr(
        settings_view_module,
        "create_secret_store",
        lambda *_args, **_kwargs: None,
    )
    view = settings_view_module.SettingsView()
    existing_basic = view.runtime_log_basic
    existing_detailed = view.runtime_log_detailed

    view.bind_settings_intents(
        surface=SettingsSurfaceIntents(
            settings_changed=lambda *_a, **_k: None,
            show_snackbar=lambda *_a, **_k: None,
        ),
        provider=SettingsProviderIntents(
            providers_changed=lambda *_a, **_k: None,
            request_openrouter_pkce=lambda *_a, **_k: None,
            verify_api_key=lambda *_a, **_k: None,
            provider_secret_change=lambda *_a, **_k: None,
            secret_cleared=lambda *_a, **_k: None,
            local_llm_secret_changed=lambda *_a, **_k: None,
            gpu_discovery_requested=lambda *_a, **_k: None,
        ),
    )

    assert view.runtime_log_basic is existing_basic
    assert view.runtime_log_detailed is existing_detailed


def _attribute_calls(source: str, owner_names: tuple[str, ...]) -> set[str]:
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            value = node.value
            if isinstance(value, ast.Name) and value.id in owner_names:
                names.add(node.attr)
            elif isinstance(value, ast.Attribute) and value.attr in owner_names:
                names.add(node.attr)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            owner = node.args[0]
            if (isinstance(owner, ast.Name) and owner.id in owner_names) or (
                isinstance(owner, ast.Attribute) and owner.attr in owner_names
            ):
                names.add(node.args[1].value)
    return names


def test_both_drivers_reach_the_settings_view_without_an_orphaned_push() -> None:
    controller_source = (SOURCE_ROOT / "ui" / "controller.py").read_text(encoding="utf-8")
    app_source = (SOURCE_ROOT / "ui" / "app.py").read_text(encoding="utf-8")
    driver_names = ("view_settings", "settings_view")

    controller_attrs = _attribute_calls(controller_source, driver_names)
    app_attrs = _attribute_calls(app_source, driver_names)

    for name in GUI_CONTROLLER_ONLY_PUSHES:
        assert name in controller_attrs, f"{name} lost its GuiController push site"
        assert callable(getattr(settings_view_module.SettingsView, name, None)), name

    for name in DUAL_DRIVER_PUSHES:
        assert name in controller_attrs, f"{name} lost its GuiController push site"

    assert "load_from_settings" in app_attrs
    assert "consume_provider_apply_settings" in app_attrs
    assert "has_provider_changes" in app_attrs
