from __future__ import annotations

from collections.abc import Callable

import flet as ft

from puripuly_heart.ui.foundation.tokens import FOUNDATION_DESIGN_TOKENS
from puripuly_heart.ui.settings.contract import (
    SettingsApiSurfaceRegions,
    SettingsApiSurfaceSlots,
)

SETTINGS_ROW_SPACING = FOUNDATION_DESIGN_TOKENS.spacing.page
SETTINGS_API_GPU_PLACEHOLDER_COUNT = 2


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

    gpu_device_placeholders = tuple(
        placeholder_factory() for _ in range(SETTINGS_API_GPU_PLACEHOLDER_COUNT)
    )
    gpu_device_controls = ft.Row(
        [slots.gpu_device, *gpu_device_placeholders],
        spacing=SETTINGS_ROW_SPACING,
        expand=True,
    )
    gpu_device_row = ft.Container(content=gpu_device_controls)
    gpu_device_row.visible = False

    return SettingsApiSurfaceRegions(
        rows=(
            provider_row,
            translation_connection_row,
            gpu_device_row,
            slots.local_llm_connection,
            slots.managed_key,
            slots.peer_expected_language,
            slots.api_keys,
        ),
        provider_row=provider_row,
        provider_controls=provider_controls,
        translation_connection_row=translation_connection_row,
        translation_connection_controls=translation_connection_controls,
        translation_connection_leading_placeholder=translation_connection_leading_placeholder,
        gpu_device_row=gpu_device_row,
        gpu_device_controls=gpu_device_controls,
        gpu_device_placeholders=gpu_device_placeholders,
    )


__all__ = [
    "SETTINGS_API_GPU_PLACEHOLDER_COUNT",
    "SETTINGS_ROW_SPACING",
    "compose_settings_api_surface",
]
