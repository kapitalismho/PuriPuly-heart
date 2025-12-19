from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Callable

import flet as ft
from flet.core.colors import Colors as colors
from flet.core.icons import Icons as icons

from ajebal_daera_translator.app.wiring import create_secret_store
from ajebal_daera_translator.config.settings import (
    AppSettings,
    LLMProviderName,
    SecretsBackend,
    STTProviderName,
)
from ajebal_daera_translator.ui.components.bento_card import BentoCard
from ajebal_daera_translator.ui.theme import COLOR_PRIMARY

logger = logging.getLogger(__name__)


class SettingsView(ft.ListView):
    def __init__(self):
        super().__init__(expand=True, spacing=15, padding=10)

        self.on_settings_changed: Callable[[AppSettings], None] | None = None
        self.on_providers_changed: Callable[[], None] | None = None

        self._settings: AppSettings | None = None
        self._config_path: Path | None = None

        self.stt_provider = ft.Dropdown(
            label="STT Provider",
            options=[
                ft.dropdown.Option("Alibaba Model Studio"),
                ft.dropdown.Option("Deepgram"),
            ],
            on_change=self._on_provider_change,
            border_radius=8,
        )
        self.llm_provider = ft.Dropdown(
            label="LLM Provider",
            options=[
                ft.dropdown.Option("Google Gemini"),
                ft.dropdown.Option("Alibaba Qwen"),
            ],
            on_change=self._on_provider_change,
            border_radius=8,
        )

        self.google_api_key = ft.TextField(
            label="Google API Key (Gemini)",
            password=True,
            can_reveal_password=True,
            on_change=lambda e: self._on_secret_change("google_api_key", self.google_api_key.value),
            border_radius=8,
        )
        self.alibaba_api_key = ft.TextField(
            label="Alibaba API Key (Qwen + Alibaba STT)",
            password=True,
            can_reveal_password=True,
            on_change=lambda e: self._on_secret_change("alibaba_api_key", self.alibaba_api_key.value),
            border_radius=8,
        )



        self.alibaba_stt_model = ft.TextField(
            label="Alibaba STT Model",
            on_change=self._on_setting_change,
            border_radius=8,
        )
        self.alibaba_stt_endpoint = ft.TextField(
            label="Alibaba STT Endpoint",
            on_change=self._on_setting_change,
            border_radius=8,
        )

        self.deepgram_api_key = ft.TextField(
            label="Deepgram API Key",
            password=True,
            can_reveal_password=True,
            on_change=lambda e: self._on_secret_change("deepgram_api_key", self.deepgram_api_key.value),
            border_radius=8,
        )
        self.deepgram_stt_model = ft.TextField(
            label="Deepgram Model",
            hint_text="nova-3",
            on_change=self._on_setting_change,
            border_radius=8,
        )

        self.audio_host_api = ft.Dropdown(
            label="Audio Host API",
            options=[
                ft.dropdown.Option("(Default)"),
                ft.dropdown.Option("MME"),
                ft.dropdown.Option("DirectSound"),
                ft.dropdown.Option("WASAPI"),
                ft.dropdown.Option("ASIO"),
            ],
            on_change=self._on_audio_change,
            border_radius=8,
        )
        self.microphone = ft.Dropdown(
            label="Microphone",
            options=[ft.dropdown.Option("(Default)")],
            on_change=self._on_audio_change,
            border_radius=8,
        )

        self.vad_sensitivity = ft.Slider(
            min=0.0,
            max=1.0,
            divisions=20,
            value=0.5,
            label="{value}",
            active_color=COLOR_PRIMARY,
            on_change_end=self._on_audio_change,
        )

        self.system_prompt = ft.TextField(
            label="System Prompt",
            multiline=True,
            min_lines=3,
            on_change=self._on_setting_change,
            border_radius=8,
        )

        self.apply_providers_btn = ft.ElevatedButton(
            text="Apply Provider Changes (Restart Pipeline)",
            icon=icons.PLAY_CIRCLE_FILL_ROUNDED,
            style=ft.ButtonStyle(
                color=colors.WHITE,
                bgcolor=COLOR_PRIMARY,
                shape=ft.RoundedRectangleBorder(radius=8),
            ),
            on_click=self._on_apply_providers,
        )

        self.controls = [
            ft.Text("Settings", size=20, weight=ft.FontWeight.BOLD, color=colors.WHITE),
            self._build_section(
                "Providers",
                [
                    ft.Text("Speech-to-Text (STT)", size=12, color=colors.GREY_400),
                    self.stt_provider,
                    self.alibaba_stt_model,
                    self.alibaba_stt_endpoint,
                    ft.Divider(height=5, color=colors.TRANSPARENT),
                    self.deepgram_api_key,
                    self.deepgram_stt_model,
                    ft.Divider(height=10, color=colors.TRANSPARENT),
                    ft.Text("Translation (LLM)", size=12, color=colors.GREY_400),
                    self.llm_provider,
                    ft.Divider(height=5, color=colors.TRANSPARENT),
                    self.google_api_key,
                    self.alibaba_api_key,
                    ft.Divider(height=10, color=colors.TRANSPARENT),
                    self.apply_providers_btn,
                ],
            ),
            self._build_section(
                "Audio",
                [
                    self.audio_host_api,
                    self.microphone,
                    ft.Text("VAD Sensitivity (Speech Detection)", size=12, color=colors.GREY_400),
                    self.vad_sensitivity,
                ],
            ),
            self._build_section(
                "Persona",
                [
                    self.system_prompt,
                ],
            ),
        ]

    def load_from_settings(self, settings: AppSettings, *, config_path: Path) -> None:
        self._settings = settings
        self._config_path = config_path

        if settings.provider.stt == STTProviderName.ALIBABA:
            self.stt_provider.value = "Alibaba Model Studio"
        else:
            self.stt_provider.value = "Deepgram"
        self.llm_provider.value = "Google Gemini" if settings.provider.llm == LLMProviderName.GEMINI else "Alibaba Qwen"

        self.alibaba_stt_model.value = settings.alibaba_stt.model
        self.alibaba_stt_endpoint.value = settings.alibaba_stt.endpoint
        self.deepgram_stt_model.value = settings.deepgram_stt.model

        self.audio_host_api.value = settings.audio.input_host_api or "(Default)"
        self._refresh_microphones()
        self.microphone.value = settings.audio.input_device or "(Default)"
        self.vad_sensitivity.value = settings.stt.vad_speech_threshold

        self.system_prompt.value = settings.system_prompt

        with contextlib.suppress(Exception):
            store = create_secret_store(settings.secrets, config_path=config_path)
            self.google_api_key.value = store.get("google_api_key") or ""
            self.alibaba_api_key.value = store.get("alibaba_api_key") or ""
            self.deepgram_api_key.value = store.get("deepgram_api_key") or ""

        self._update_provider_visibility()
        self.update()

    def _build_section(self, title: str, controls: list[ft.Control]) -> ft.Control:
        return BentoCard(
            content=ft.Column(
                [
                    ft.Text(title, size=12, weight=ft.FontWeight.BOLD, color=colors.GREY_500),
                    ft.Column(controls, spacing=15),
                ]
            )
        )

    def _on_secret_change(self, key: str, value: str) -> None:
        if self._settings is None or self._config_path is None:
            return
        with contextlib.suppress(Exception):
            store = create_secret_store(self._settings.secrets, config_path=self._config_path)
            if value:
                store.set(key, value)
            else:
                store.delete(key)

    def _on_provider_change(self, e) -> None:
        _ = e
        if self._settings is None:
            return

        stt_value = self.stt_provider.value
        if stt_value == "Alibaba Model Studio":
            self._settings.provider.stt = STTProviderName.ALIBABA
        else:
            self._settings.provider.stt = STTProviderName.DEEPGRAM

        self._settings.provider.llm = (
            LLMProviderName.GEMINI if self.llm_provider.value == "Google Gemini" else LLMProviderName.QWEN
        )
        self._update_provider_visibility()
        self._emit_settings_changed()

    def _on_apply_providers(self, e) -> None:
        _ = e
        self._emit_settings_changed()
        if self.on_providers_changed:
            self.on_providers_changed()

    def _update_provider_visibility(self) -> None:
        if self._settings is None:
            return

        stt_provider = self._settings.provider.stt

        is_alibaba_stt = stt_provider == STTProviderName.ALIBABA
        self.alibaba_stt_model.visible = is_alibaba_stt
        self.alibaba_stt_endpoint.visible = is_alibaba_stt

        is_deepgram_stt = stt_provider == STTProviderName.DEEPGRAM
        self.deepgram_api_key.visible = is_deepgram_stt
        self.deepgram_stt_model.visible = is_deepgram_stt

        self.google_api_key.visible = True
        self.alibaba_api_key.visible = True

    def _on_setting_change(self, e) -> None:
        _ = e
        if self._settings is None:
            return

        self._settings.alibaba_stt.model = self.alibaba_stt_model.value or self._settings.alibaba_stt.model
        self._settings.alibaba_stt.endpoint = self.alibaba_stt_endpoint.value or self._settings.alibaba_stt.endpoint
        self._settings.deepgram_stt.model = self.deepgram_stt_model.value or self._settings.deepgram_stt.model
        self._settings.system_prompt = self.system_prompt.value or ""

        self._emit_settings_changed()

    def _on_audio_change(self, e) -> None:
        _ = e
        if self._settings is None:
            return

        host_api = self.audio_host_api.value or "(Default)"
        if host_api == "(Default)":
            host_api = ""
        self._settings.audio.input_host_api = host_api

        device = self.microphone.value or "(Default)"
        if device == "(Default)":
            device = ""
        self._settings.audio.input_device = device

        self._settings.stt.vad_speech_threshold = float(self.vad_sensitivity.value or 0.5)

        self._refresh_microphones()
        self._emit_settings_changed()

    def _refresh_microphones(self) -> None:
        host_api = self.audio_host_api.value or "(Default)"
        if host_api == "(Default)":
            host_api = ""

        devices = ["(Default)"]
        with contextlib.suppress(Exception):
            import sounddevice as sd  # type: ignore

            hostapi_index: int | None = None
            if host_api:
                for idx, item in enumerate(sd.query_hostapis()):
                    name = str(item.get("name", "") or "")
                    if name.lower() == host_api.lower():
                        hostapi_index = idx
                        break

            for dev in sd.query_devices():
                if int(dev.get("max_input_channels", 0) or 0) <= 0:
                    continue
                if hostapi_index is not None and int(dev.get("hostapi", -1) or -1) != hostapi_index:
                    continue
                name = str(dev.get("name", "") or "").strip()
                if name:
                    devices.append(name)

        # Preserve selection if possible.
        current = self.microphone.value
        self.microphone.options = [ft.dropdown.Option(d) for d in devices]
        if current in devices:
            self.microphone.value = current
        else:
            self.microphone.value = "(Default)"

    def _emit_settings_changed(self) -> None:
        if self._settings is None:
            return
        if self.on_settings_changed:
            self.on_settings_changed(self._settings)
