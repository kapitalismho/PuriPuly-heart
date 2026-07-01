from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("flet")

import puripuly_heart.ui.app as app_module
from puripuly_heart.ui.app import TranslatorApp


class DummyDashboard:
    def __init__(self) -> None:
        self.translation_calls: list[tuple[bool, bool]] = []
        self.stt_calls: list[tuple[bool, bool]] = []

    def set_translation_needs_key(self, value: bool, *, update_ui: bool = True) -> None:
        self.translation_calls.append((value, update_ui))

    def set_stt_needs_key(self, value: bool, *, update_ui: bool = True) -> None:
        self.stt_calls.append((value, update_ui))


def _make_app_with_verified_state() -> TranslatorApp:
    app = TranslatorApp.__new__(TranslatorApp)
    app.controller = SimpleNamespace(
        settings=SimpleNamespace(
            api_key_verified=SimpleNamespace(
                deepgram=True,
                soniox=True,
                google=True,
                openrouter=True,
                deepseek=True,
                cerebras=True,
                alibaba_beijing=True,
                alibaba_singapore=True,
            )
        ),
        config_path=Path("settings.json"),
    )
    app.view_dashboard = DummyDashboard()
    return app


def test_on_secret_cleared_resets_alibaba_beijing_for_new_secret_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _make_app_with_verified_state()
    saves: list[tuple[Path, object]] = []

    def fake_save(path: Path, settings: object) -> None:
        saves.append((path, settings))

    monkeypatch.setattr(app_module, "save_settings", fake_save)

    app._on_secret_cleared("alibaba_api_key_beijing")

    assert app.controller.settings.api_key_verified.alibaba_beijing is False
    assert app.view_dashboard.translation_calls == [(True, False)]
    assert app.view_dashboard.stt_calls == []
    assert len(saves) == 1


def test_on_secret_cleared_resets_alibaba_singapore_for_new_secret_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _make_app_with_verified_state()
    saves: list[tuple[Path, object]] = []

    def fake_save(path: Path, settings: object) -> None:
        saves.append((path, settings))

    monkeypatch.setattr(app_module, "save_settings", fake_save)

    app._on_secret_cleared("alibaba_api_key_singapore")

    assert app.controller.settings.api_key_verified.alibaba_singapore is False
    assert app.view_dashboard.translation_calls == [(True, False)]
    assert app.view_dashboard.stt_calls == []
    assert len(saves) == 1


def test_on_secret_cleared_ignores_unknown_key(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _make_app_with_verified_state()
    saves: list[tuple[Path, object]] = []

    def fake_save(path: Path, settings: object) -> None:
        saves.append((path, settings))

    monkeypatch.setattr(app_module, "save_settings", fake_save)

    app._on_secret_cleared("unknown_key")

    assert app.controller.settings.api_key_verified.alibaba_beijing is True
    assert app.controller.settings.api_key_verified.alibaba_singapore is True
    assert app.controller.settings.api_key_verified.openrouter is True
    assert app.controller.settings.api_key_verified.cerebras is True
    assert app.view_dashboard.translation_calls == []
    assert app.view_dashboard.stt_calls == []
    assert saves == []


def test_on_secret_cleared_resets_openrouter_for_new_secret_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _make_app_with_verified_state()
    saves: list[tuple[Path, object]] = []

    def fake_save(path: Path, settings: object) -> None:
        saves.append((path, settings))

    monkeypatch.setattr(app_module, "save_settings", fake_save)

    app._on_secret_cleared("openrouter_api_key")

    assert app.controller.settings.api_key_verified.openrouter is False
    assert app.view_dashboard.translation_calls == [(True, False)]
    assert app.view_dashboard.stt_calls == []
    assert len(saves) == 1


def test_on_secret_cleared_resets_deepseek_for_new_secret_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _make_app_with_verified_state()
    saves: list[tuple[Path, object]] = []

    def fake_save(path: Path, settings: object) -> None:
        saves.append((path, settings))

    monkeypatch.setattr(app_module, "save_settings", fake_save)

    app._on_secret_cleared("deepseek_api_key")

    assert app.controller.settings.api_key_verified.deepseek is False
    assert app.view_dashboard.translation_calls == [(True, False)]
    assert app.view_dashboard.stt_calls == []
    assert len(saves) == 1


def test_on_secret_cleared_resets_cerebras_for_new_secret_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _make_app_with_verified_state()
    saves: list[tuple[Path, object]] = []

    def fake_save(path: Path, settings: object) -> None:
        saves.append((path, settings))

    monkeypatch.setattr(app_module, "save_settings", fake_save)

    app._on_secret_cleared("cerebras_api_key")

    assert app.controller.settings.api_key_verified.cerebras is False
    assert app.view_dashboard.translation_calls == [(True, False)]
    assert app.view_dashboard.stt_calls == []
    assert len(saves) == 1
