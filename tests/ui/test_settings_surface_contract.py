from __future__ import annotations

import ast

import flet as ft
import pytest

pytest.importorskip("flet")

from puripuly_heart.ui.foundation.tokens import FOUNDATION_DESIGN_TOKENS
from puripuly_heart.ui.settings.contract import (
    SettingsApiSurfaceSlots,
    SettingsGeneralIntents,
    SettingsGeneralSurfaceSlots,
    SettingsOverlayIntents,
    SettingsOverlaySurfaceSlots,
    SettingsPromptIntents,
    SettingsPromptSurfaceSlots,
    SettingsProviderIntents,
    SettingsSurfaceIntents,
)
from puripuly_heart.ui.settings.renderer import (
    SETTINGS_ROW_SPACING,
    compose_settings_api_surface,
    compose_settings_general_surface,
    compose_settings_overlay_surface,
    compose_settings_prompt_surface,
)
from puripuly_heart.ui.views import settings as settings_view_module
from tests.helpers.paths import SOURCE_ROOT

SETTINGS_ADAPTER_PUSHES = (
    "set_managed_trial_usage_state",
    "set_local_cpu_auto_available",
    "refresh_loopback_capture_target",
    "load_from_settings",
    "refresh_after_openrouter_pkce_success",
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
                "gpu_llm",
                "gpu_refresh",
                "local_llm_connection",
                "custom_stt_connection",
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

    def gpu_llm_control(self) -> ft.Control:
        return self.controls["gpu_llm"]

    def gpu_refresh_control(self) -> ft.Control:
        return self.controls["gpu_refresh"]

    def local_llm_connection_control(self) -> ft.Control:
        return self.controls["local_llm_connection"]

    def custom_stt_connection_control(self) -> ft.Control:
        return self.controls["custom_stt_connection"]

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
        provider.controls["custom_stt_connection"],
        provider.controls["managed_key"],
        provider.controls["peer_expected_language"],
        provider.controls["api_keys"],
    )


def test_settings_api_surface_places_every_slot_in_the_accepted_position() -> None:
    surface, provider, placeholders = _compose()

    assert len(placeholders) == 1
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
        provider.controls["gpu_llm"],
        provider.controls["gpu_refresh"],
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
    general = SettingsGeneralIntents(
        start_microphone_test=make("start_microphone_test"),
        telemetry_consent_change=make("telemetry_consent_change"),
        list_loopback_capture_options=make("list_loopback_capture_options"),
        list_loopback_process_options=make("list_loopback_process_options"),
        list_loopback_device_options=make("list_loopback_device_options"),
        current_loopback_capture_option=make("current_loopback_capture_option"),
        apply_loopback_capture_option=make("apply_loopback_capture_option"),
        loopback_capture_summary=make("loopback_capture_summary"),
    )
    prompt = SettingsPromptIntents(prompt_apply_settings=make("prompt_apply_settings"))
    overlay = SettingsOverlayIntents(
        desktop_overlay_lock_change=make("desktop_overlay_lock_change"),
        desktop_overlay_size_change=make("desktop_overlay_size_change"),
        desktop_overlay_recovery_action=make("desktop_overlay_recovery_action"),
        desktop_overlay_position_reset=make("desktop_overlay_position_reset"),
        view_logs=make("view_logs"),
        calibration_begin=make("calibration_begin"),
        calibration_change=make("calibration_change"),
        calibration_apply=make("calibration_apply"),
        calibration_cancel=make("calibration_cancel"),
    )

    view.bind_settings_intents(
        surface=surface,
        provider=provider,
        general=general,
        prompt=prompt,
        overlay=overlay,
    )

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
    assert view.on_start_microphone_test is general.start_microphone_test
    assert view.on_telemetry_consent_change is general.telemetry_consent_change
    assert view.on_list_loopback_capture_options is general.list_loopback_capture_options
    assert view.on_list_loopback_process_options is general.list_loopback_process_options
    assert view.on_list_loopback_device_options is general.list_loopback_device_options
    assert view.on_current_loopback_capture_option is general.current_loopback_capture_option
    assert view.on_apply_loopback_capture_option is general.apply_loopback_capture_option
    assert view.on_loopback_capture_summary is general.loopback_capture_summary
    assert view.on_prompt_apply_settings is prompt.prompt_apply_settings
    assert view.on_desktop_overlay_lock_change is overlay.desktop_overlay_lock_change
    assert view.on_desktop_overlay_size_change is overlay.desktop_overlay_size_change
    assert view.on_desktop_overlay_recovery_action is overlay.desktop_overlay_recovery_action
    assert view.on_desktop_overlay_position_reset is overlay.desktop_overlay_position_reset
    assert view.on_view_logs is overlay.view_logs
    assert view.on_overlay_calibration_begin is overlay.calibration_begin
    assert view.on_overlay_calibration_change is overlay.calibration_change
    assert view.on_overlay_calibration_apply is overlay.calibration_apply
    assert view.on_overlay_calibration_cancel is overlay.calibration_cancel


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
    existing_calibration_begin = view.on_overlay_calibration_begin

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
        general=SettingsGeneralIntents(
            start_microphone_test=lambda *_a, **_k: None,
            telemetry_consent_change=lambda *_a, **_k: None,
            list_loopback_capture_options=lambda *_a, **_k: None,
            list_loopback_process_options=lambda *_a, **_k: None,
            list_loopback_device_options=lambda *_a, **_k: None,
            current_loopback_capture_option=lambda *_a, **_k: "",
            apply_loopback_capture_option=lambda *_a, **_k: None,
            loopback_capture_summary=lambda *_a, **_k: "",
        ),
        prompt=SettingsPromptIntents(prompt_apply_settings=lambda *_a, **_k: None),
        overlay=SettingsOverlayIntents(
            desktop_overlay_lock_change=lambda *_a, **_k: None,
            desktop_overlay_size_change=lambda *_a, **_k: None,
            desktop_overlay_recovery_action=lambda *_a, **_k: None,
            desktop_overlay_position_reset=lambda *_a, **_k: None,
            view_logs=lambda *_a, **_k: None,
        ),
    )

    assert view.runtime_log_basic is existing_basic
    assert view.runtime_log_detailed is existing_detailed
    assert view.on_overlay_calibration_begin is existing_calibration_begin


def _general_slots() -> SettingsGeneralSurfaceSlots:
    return SettingsGeneralSurfaceSlots(
        **{
            name: ft.Text(name)
            for name in SettingsGeneralSurfaceSlots.__dataclass_fields__  # noqa: F821
        }
    )


def _overlay_slots() -> SettingsOverlaySurfaceSlots:
    return SettingsOverlaySurfaceSlots(
        **{name: ft.Text(name) for name in SettingsOverlaySurfaceSlots.__dataclass_fields__}
    )


def test_general_surface_preserves_the_accepted_row_order_and_spacing() -> None:
    placeholders: list[ft.Control] = []
    slots = _general_slots()
    surface = compose_settings_general_surface(
        slots,
        placeholder_factory=lambda: _track(placeholders),
    )

    assert len(placeholders) == 1
    assert surface.rows == (
        surface.primary_row,
        surface.audio_row,
        surface.vad_row,
        surface.clipboard_row,
    )
    assert surface.primary_row.content.controls == [
        slots.ui,
        slots.chatbox_source,
        surface.primary_row_placeholder,
    ]
    assert surface.audio_row.content.controls == [
        slots.audio_host_api,
        slots.microphone,
        slots.loopback,
    ]
    assert surface.vad_row.content.controls == [
        slots.microphone_test,
        slots.self_vad,
        slots.peer_vad,
    ]
    assert surface.clipboard_row.content.controls == [
        slots.clipboard_auto_translate,
        slots.vrchat_mic_intercept,
        slots.telemetry_consent,
    ]
    for row in surface.rows:
        assert row.visible is True
        assert row.content.spacing == FOUNDATION_DESIGN_TOKENS.spacing.page
        assert row.content.expand is True


def test_prompt_surface_preserves_the_accepted_two_card_order() -> None:
    vocabulary = ft.Text("vocabulary")
    persona = ft.Text("persona")
    surface = compose_settings_prompt_surface(
        SettingsPromptSurfaceSlots(custom_vocabulary=vocabulary, persona=persona)
    )
    assert surface.rows == (vocabulary, persona)


def test_overlay_surface_preserves_the_accepted_six_rows_and_recovery_visibility() -> None:
    placeholders: list[ft.Control] = []
    slots = _overlay_slots()
    surface = compose_settings_overlay_surface(
        slots,
        placeholder_factory=lambda: _track(placeholders),
    )

    assert len(surface.rows) == 6
    assert len(placeholders) == 1
    assert surface.rows[0] is surface.target_row
    assert surface.vr_rows == (surface.rows[1], surface.rows[2])
    assert surface.desktop_rows == (surface.rows[3], surface.rows[4])
    assert surface.desktop_controls_row is surface.rows[3]
    assert surface.recovery_row is surface.rows[5]
    assert surface.recovery_row.visible is False
    for row in surface.rows[:5]:
        assert row.visible is True

    assert surface.target_row.content.controls == [
        slots.overlay_target,
        slots.overlay_translation,
        slots.overlay_peer_original,
    ]
    assert surface.rows[1].content.controls == [slots.anchor, slots.distance, slots.offset_x]
    assert surface.rows[2].content.controls == [slots.offset_y, slots.text_scale, slots.vr_reset]
    assert surface.rows[3].content.controls == [
        slots.desktop_size,
        slots.desktop_lock,
        slots.desktop_background_alpha,
    ]
    assert surface.rows[4].content.controls == [
        slots.desktop_swap_caption_languages,
        slots.desktop_reset,
        slots.desktop_reset_spacer,
    ]
    assert surface.recovery_row.content.controls == [
        slots.desktop_status,
        surface.recovery_row_placeholder,
        slots.desktop_status_trailing,
    ]


def _track(bucket: list[ft.Control]) -> ft.Control:
    control = ft.Container()
    bucket.append(control)
    return control


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


def test_settings_projection_is_the_only_full_settings_view_pusher() -> None:
    adapter_source = (SOURCE_ROOT / "ui" / "presentation_adapter.py").read_text(encoding="utf-8")
    app_source = (SOURCE_ROOT / "ui" / "app.py").read_text(encoding="utf-8")
    projection_source = (
        SOURCE_ROOT / "app" / "services" / "settings" / "settings_projection.py"
    ).read_text(encoding="utf-8")
    driver_names = ("view_settings", "settings_view")

    adapter_attrs = _attribute_calls(adapter_source, driver_names)
    app_attrs = _attribute_calls(app_source, driver_names)
    projection_attrs = _attribute_calls(projection_source, ("presentation",))

    for name in SETTINGS_ADAPTER_PUSHES:
        assert name in adapter_attrs, f"{name} lost its presentation adapter push site"
        assert callable(getattr(settings_view_module.SettingsView, name, None)), name

    assert "render_settings" in projection_attrs
    assert "refresh_settings_after_openrouter_pkce_success" in projection_attrs
    assert "load_from_settings" not in app_attrs
    assert "refresh_after_openrouter_pkce_success" not in app_attrs
    assert "consume_provider_apply_settings" in app_attrs
    assert "has_provider_changes" in app_attrs
