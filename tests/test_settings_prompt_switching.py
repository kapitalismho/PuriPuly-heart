from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("flet")

from ajebal_daera_translator.config.prompts import load_prompt_for_provider
from ajebal_daera_translator.config.settings import AppSettings, LLMProviderName
from ajebal_daera_translator.ui.views import settings as settings_view


class DummySecretStore:
    def get(self, _key: str) -> str | None:
        return None


def _make_settings_view(monkeypatch):
    monkeypatch.setattr(settings_view.SettingsView, "_populate_host_apis", lambda self: None)
    monkeypatch.setattr(settings_view.SettingsView, "_refresh_microphones", lambda self: None)
    monkeypatch.setattr(settings_view.SettingsView, "update", lambda self: None)
    monkeypatch.setattr(
        settings_view, "create_secret_store", lambda *args, **kwargs: DummySecretStore()
    )
    return settings_view.SettingsView()


def test_settings_view_loads_qwen_prompt(monkeypatch) -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.QWEN

    view = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))

    assert view.system_prompt.value == load_prompt_for_provider("qwen")
    assert "Qwen" in view.prompt_provider_label.value
    assert settings.system_prompt == view.system_prompt.value


def test_settings_view_switches_prompt_on_llm_change(monkeypatch) -> None:
    settings = AppSettings()

    view = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))

    assert view.system_prompt.value == load_prompt_for_provider("gemini")

    view.llm_provider.value = "Alibaba Qwen"
    view._on_provider_change(None)

    assert view.system_prompt.value == load_prompt_for_provider("qwen")
    assert settings.provider.llm == LLMProviderName.QWEN

    view.llm_provider.value = "Google Gemini"
    view._on_provider_change(None)

    assert view.system_prompt.value == load_prompt_for_provider("gemini")
    assert settings.provider.llm == LLMProviderName.GEMINI
