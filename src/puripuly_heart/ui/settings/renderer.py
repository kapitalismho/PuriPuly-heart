from __future__ import annotations

from collections.abc import Callable

import flet as ft

from puripuly_heart.ui.foundation.tokens import FOUNDATION_DESIGN_TOKENS
from puripuly_heart.ui.settings.contract import (
    SettingsApiSurfaceRegions,
    SettingsApiSurfaceSlots,
    SettingsGeneralSurfaceRegions,
    SettingsGeneralSurfaceSlots,
    SettingsOverlaySurfaceRegions,
    SettingsOverlaySurfaceSlots,
    SettingsPromptSurfaceRegions,
    SettingsPromptSurfaceSlots,
)

SETTINGS_ROW_SPACING = FOUNDATION_DESIGN_TOKENS.spacing.page


def compose_settings_api_surface(
    slots: SettingsApiSurfaceSlots,
    *,
    placeholder_factory: Callable[[], ft.Control],
) -> SettingsApiSurfaceRegions:
    provider_controls = ft.Row(
        [slots.self_stt, slots.peer_stt, slots.translation_provider],
        spacing=SETTINGS_ROW_SPACING,
        expand=True,
    )
    provider_row = ft.Container(content=provider_controls)

    translation_connection_leading_placeholder = placeholder_factory()
    translation_connection_controls = ft.Row(
        [
            translation_connection_leading_placeholder,
            slots.translation_connection,
            slots.translation_fallback,
        ],
        spacing=SETTINGS_ROW_SPACING,
        expand=True,
    )
    translation_connection_row = ft.Container(
        content=translation_connection_controls,
        visible=True,
    )

    gpu_device_controls = ft.Row(
        [slots.gpu_device, slots.gpu_llm, slots.gpu_refresh],
        spacing=SETTINGS_ROW_SPACING,
        expand=True,
    )
    gpu_device_row = ft.Container(content=gpu_device_controls)
    gpu_device_row.visible = False

    rows: list[ft.Control] = [
        provider_row,
        translation_connection_row,
        gpu_device_row,
        slots.local_llm_connection,
        slots.custom_stt_connection,
        slots.managed_key,
        slots.peer_expected_language,
        slots.api_keys,
    ]
    if slots.http_extension is not None:
        rows.insert(2, slots.http_extension)

    return SettingsApiSurfaceRegions(
        rows=tuple(rows),
        provider_row=provider_row,
        provider_controls=provider_controls,
        translation_connection_row=translation_connection_row,
        translation_connection_controls=translation_connection_controls,
        translation_connection_leading_placeholder=translation_connection_leading_placeholder,
        gpu_device_row=gpu_device_row,
        gpu_device_controls=gpu_device_controls,
    )


def compose_settings_triple_row(
    first: ft.Control,
    second: ft.Control,
    third: ft.Control,
    *,
    visible: bool = True,
) -> ft.Container:
    row = ft.Container(
        content=ft.Row(
            [first, second, third],
            spacing=SETTINGS_ROW_SPACING,
            expand=True,
        ),
    )
    row.visible = visible
    return row


def compose_settings_general_surface(
    slots: SettingsGeneralSurfaceSlots,
    *,
    placeholder_factory: Callable[[], ft.Control],
) -> SettingsGeneralSurfaceRegions:
    primary_row_placeholder = placeholder_factory()
    primary_row = compose_settings_triple_row(
        slots.ui,
        slots.chatbox_source,
        primary_row_placeholder,
    )
    audio_row = compose_settings_triple_row(
        slots.audio_host_api,
        slots.microphone,
        slots.loopback,
    )
    vad_row = compose_settings_triple_row(
        slots.microphone_test,
        slots.self_vad,
        slots.peer_vad,
    )
    clipboard_row = compose_settings_triple_row(
        slots.clipboard_auto_translate,
        slots.vrchat_mic_intercept,
        slots.telemetry_consent,
    )
    return SettingsGeneralSurfaceRegions(
        rows=(primary_row, audio_row, vad_row, clipboard_row),
        primary_row=primary_row,
        audio_row=audio_row,
        vad_row=vad_row,
        clipboard_row=clipboard_row,
        primary_row_placeholder=primary_row_placeholder,
    )


def compose_settings_prompt_surface(
    slots: SettingsPromptSurfaceSlots,
) -> SettingsPromptSurfaceRegions:
    return SettingsPromptSurfaceRegions(rows=(slots.custom_vocabulary, slots.persona))


def compose_settings_overlay_surface(
    slots: SettingsOverlaySurfaceSlots,
    *,
    placeholder_factory: Callable[[], ft.Control],
) -> SettingsOverlaySurfaceRegions:
    target_row = compose_settings_triple_row(
        slots.overlay_target,
        slots.overlay_translation,
        slots.overlay_peer_original,
    )
    vr_anchor_row = compose_settings_triple_row(slots.anchor, slots.distance, slots.offset_x)
    vr_offset_row = compose_settings_triple_row(slots.offset_y, slots.text_scale, slots.vr_reset)
    desktop_controls_row = compose_settings_triple_row(
        slots.desktop_size,
        slots.desktop_lock,
        slots.desktop_background_alpha,
    )
    desktop_reset_row = compose_settings_triple_row(
        slots.desktop_swap_caption_languages,
        slots.desktop_reset,
        slots.desktop_reset_spacer,
    )
    recovery_row_placeholder = placeholder_factory()
    recovery_row = compose_settings_triple_row(
        slots.desktop_status,
        recovery_row_placeholder,
        slots.desktop_status_trailing,
        visible=False,
    )
    return SettingsOverlaySurfaceRegions(
        rows=(
            target_row,
            vr_anchor_row,
            vr_offset_row,
            desktop_controls_row,
            desktop_reset_row,
            recovery_row,
        ),
        target_row=target_row,
        vr_rows=(vr_anchor_row, vr_offset_row),
        desktop_rows=(desktop_controls_row, desktop_reset_row),
        desktop_controls_row=desktop_controls_row,
        recovery_row=recovery_row,
        recovery_row_placeholder=recovery_row_placeholder,
    )


__all__ = [
    "SETTINGS_ROW_SPACING",
    "compose_settings_api_surface",
    "compose_settings_general_surface",
    "compose_settings_overlay_surface",
    "compose_settings_prompt_surface",
    "compose_settings_triple_row",
]
