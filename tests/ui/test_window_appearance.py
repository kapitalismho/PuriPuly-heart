import asyncio
from pathlib import Path
from types import SimpleNamespace

import flet as ft
import pytest

from puripuly_heart.ui.app import TranslatorApp
from puripuly_heart.ui.components.title_bar import TitleBar
from puripuly_heart.ui.foundation.tokens import FOUNDATION_DESIGN_TOKENS, WINDOWS_DARK_PALETTE
from puripuly_heart.ui.foundation.window_appearance import load_dark_mode, save_dark_mode
from puripuly_heart.ui.i18n import t
from puripuly_heart.ui.theme import WINDOWS_THEME_SUPPORTED, get_app_theme


def test_theme_preference_preserves_existing_settings(tmp_path: Path) -> None:
    config = tmp_path / "settings.json"
    existing = b'{"legacy_setting": "unchanged"}'
    config.write_bytes(existing)

    assert load_dark_mode(config) is False
    save_dark_mode(config, True)
    assert load_dark_mode(config) is True
    assert load_dark_mode(tmp_path / "other.json") is False
    save_dark_mode(config, False)
    assert load_dark_mode(config) is False
    assert config.read_bytes() == existing


def test_invalid_theme_preference_falls_back_to_light(tmp_path: Path) -> None:
    config = tmp_path / "settings.json"
    preference = tmp_path / "settings.json.theme"
    preference.write_bytes(b"\xff")
    assert load_dark_mode(config) is False
    preference.write_text("unrecognized", encoding="utf-8")
    assert load_dark_mode(config) is False


@pytest.mark.skipif(not WINDOWS_THEME_SUPPORTED, reason="Windows main-window theme")
def test_native_themes_resolve_the_main_window_colors() -> None:
    light = get_app_theme().color_scheme
    dark = get_app_theme(dark=True).color_scheme
    assert light.surface_container_lowest == FOUNDATION_DESIGN_TOKENS.palette.background
    assert dark.surface_container_lowest == WINDOWS_DARK_PALETTE.background
    assert light.surface == FOUNDATION_DESIGN_TOKENS.palette.surface
    assert dark.surface == WINDOWS_DARK_PALETTE.surface
    assert light.primary == dark.primary == FOUNDATION_DESIGN_TOKENS.palette.primary
    assert dark.on_primary != light.on_primary
    assert dark.surface_container_highest != light.surface_container_highest


@pytest.mark.skipif(not WINDOWS_THEME_SUPPORTED, reason="Windows main-window theme")
async def test_toggle_changes_mounted_window_without_replacing_controls(tmp_path: Path) -> None:
    config = tmp_path / "settings.json"
    updates = []
    page = SimpleNamespace(theme_mode=ft.ThemeMode.LIGHT, update=lambda: updates.append(1))
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = page
    app._theme_config_path = config
    app._theme_preference_lock = asyncio.Lock()
    app.title_bar = TitleBar(page, on_close=lambda: None, on_toggle_theme=lambda: None)
    text_input = ft.TextField(value="Draft text")
    app.content_area = ft.Container(content=text_input)
    controls = app.title_bar.content.controls[1].controls
    button, minimize, _close = controls
    assert button.icon == ft.Icons.DARK_MODE_OUTLINED
    assert button.tooltip == t("theme.switch_to_dark")
    assert button.width == minimize.width

    await app._toggle_window_theme()
    assert page.theme_mode == ft.ThemeMode.DARK
    assert button.icon == ft.Icons.LIGHT_MODE_OUTLINED
    assert button.tooltip == t("theme.switch_to_light")
    assert page.bgcolor == WINDOWS_DARK_PALETTE.background
    assert load_dark_mode(config) is True
    assert app.content_area.content is text_input
    assert text_input.value == "Draft text"

    await app._toggle_window_theme()
    assert page.theme_mode == ft.ThemeMode.LIGHT
    assert button.icon == ft.Icons.DARK_MODE_OUTLINED
    assert load_dark_mode(config) is False
    assert len(updates) == 2
