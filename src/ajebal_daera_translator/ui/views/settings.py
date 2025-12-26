from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path
from typing import Callable

import flet as ft
from flet import Colors as colors
from flet import Icons as icons

from ajebal_daera_translator.app.wiring import create_secret_store
from ajebal_daera_translator.config.settings import (
    AppSettings,
    LLMProviderName,
    SecretsBackend,
    STTProviderName,
)
from ajebal_daera_translator.config.prompts import load_prompt_for_provider
from ajebal_daera_translator.core.language import get_stt_compatibility_warning
from ajebal_daera_translator.ui.components.bento_card import BentoCard
from ajebal_daera_translator.ui.theme import COLOR_PRIMARY

logger = logging.getLogger(__name__)


class SettingsView(ft.ListView):
    def __init__(self):
        super().__init__(expand=True, spacing=15, padding=10)

        self.on_settings_changed: Callable[[AppSettings], None] | None = None
        self.on_providers_changed: Callable[[], None] | None = None
        self.on_verify_api_key: Callable[[str, str], object] | None = None  # Returns Awaitable[tuple[bool, str]]

        self._settings: AppSettings | None = None
        self._config_path: Path | None = None

        self.stt_provider = ft.Dropdown(
            label="STT Provider",
            options=[
                ft.dropdown.Option("Deepgram"),
                ft.dropdown.Option("Qwen ASR"),
            ],
            on_select=self._on_provider_change,
            border_radius=8,
        )
        self.llm_provider = ft.Dropdown(
            label="LLM Provider",
            options=[
                ft.dropdown.Option("Google Gemini"),
                ft.dropdown.Option("Alibaba Qwen"),
            ],
            on_select=self._on_provider_change,
            border_radius=8,
        )

        self.google_api_key = ft.TextField(
            label="Google API Key (Gemini)",
            password=True,
            can_reveal_password=True,
            on_change=lambda e: self._on_secret_change("google_api_key", self.google_api_key.value),
            border_radius=8,
            expand=True,
        )
        self.verify_google_btn = ft.IconButton(
            icon=icons.CHECK_CIRCLE_OUTLINE_ROUNDED,
            icon_color=colors.GREY_400,
            tooltip="Verify Key",
            on_click=lambda e: self._on_verify_req("google", self.google_api_key.value, e.control),
        )

        self.alibaba_api_key = ft.TextField(
            label="Alibaba API Key (Qwen + Alibaba STT)",
            password=True,
            can_reveal_password=True,
            on_change=lambda e: self._on_secret_change("alibaba_api_key", self.alibaba_api_key.value),
            border_radius=8,
            expand=True,
        )
        self.verify_alibaba_btn = ft.IconButton(
            icon=icons.CHECK_CIRCLE_OUTLINE_ROUNDED,
            icon_color=colors.GREY_400,
            tooltip="Verify Key",
            on_click=lambda e: self._on_verify_req("alibaba", self.alibaba_api_key.value, e.control),
        )



        self.deepgram_api_key = ft.TextField(
            label="Deepgram API Key",
            password=True,
            can_reveal_password=True,
            on_change=lambda e: self._on_secret_change("deepgram_api_key", self.deepgram_api_key.value),
            border_radius=8,
            expand=True,
        )
        self.verify_deepgram_btn = ft.IconButton(
            icon=icons.CHECK_CIRCLE_OUTLINE_ROUNDED,
            icon_color=colors.GREY_400,
            tooltip="Verify Key",
            on_click=lambda e: self._on_verify_req("deepgram", self.deepgram_api_key.value, e.control),
        )

        self.deepgram_stt_model = ft.TextField(
            label="Deepgram Model",
            hint_text="nova-3",
            on_change=self._on_setting_change,
            border_radius=8,
        )

        self.audio_host_api = ft.Dropdown(
            label="Audio Host API",
            options=[ft.dropdown.Option("(Default)")],  # Will be populated dynamically
            on_select=self._on_audio_change,
            border_radius=8,
        )
        self._populate_host_apis()
        
        self.microphone = ft.Dropdown(
            label="Microphone",
            options=[ft.dropdown.Option("(Default)")],
            on_select=self._on_audio_change,
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

        self.prompt_provider_label = ft.Text(
            "Prompt for: Gemini",
            size=12,
            color=colors.GREY_400,
        )
        self.system_prompt = ft.TextField(
            label="System Prompt",
            multiline=True,
            min_lines=3,
            on_change=self._on_setting_change,
            border_radius=8,
        )

        self.apply_providers_btn = ft.ElevatedButton(
            content=ft.Text("Apply Provider Changes (Restart Pipeline)"),
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
                    ft.Divider(height=5, color=colors.TRANSPARENT),
                    ft.Row([self.deepgram_api_key, self.verify_deepgram_btn]),
                    self.deepgram_stt_model,
                    ft.Divider(height=10, color=colors.TRANSPARENT),
                    ft.Text("Translation (LLM)", size=12, color=colors.GREY_400),
                    self.llm_provider,
                    ft.Divider(height=5, color=colors.TRANSPARENT),
                    ft.Row([self.google_api_key, self.verify_google_btn]),
                    ft.Row([self.alibaba_api_key, self.verify_alibaba_btn]),
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
                    self.prompt_provider_label,
                    self.system_prompt,
                ],
            ),
        ]

    def load_from_settings(self, settings: AppSettings, *, config_path: Path) -> None:
        self._settings = settings
        self._config_path = config_path

        if settings.provider.stt == STTProviderName.QWEN_ASR:
            self.stt_provider.value = "Qwen ASR"
        else:
            # Default to Deepgram (handles DEEPGRAM and legacy ALIBABA)
            self.stt_provider.value = "Deepgram"
        self.llm_provider.value = "Google Gemini" if settings.provider.llm == LLMProviderName.GEMINI else "Alibaba Qwen"

        self.deepgram_stt_model.value = settings.deepgram_stt.model

        self.audio_host_api.value = settings.audio.input_host_api or "(Default)"
        self._refresh_microphones()
        self.microphone.value = settings.audio.input_device or "(Default)"
        self.vad_sensitivity.value = settings.stt.vad_speech_threshold

        # Load prompt for current LLM provider
        provider_name = "gemini" if settings.provider.llm == LLMProviderName.GEMINI else "qwen"
        self.prompt_provider_label.value = f"Prompt for: {provider_name.capitalize()}"
        saved_prompt = settings.system_prompt or ""
        if saved_prompt.strip():
            self.system_prompt.value = saved_prompt
        else:
            self.system_prompt.value = load_prompt_for_provider(provider_name)
        if self.prompt_provider_label.page:
            self.prompt_provider_label.update()
        if self.system_prompt.page:
            self.system_prompt.update()
        if self._settings and not saved_prompt.strip():
            self._settings.system_prompt = self.system_prompt.value

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
                    ft.Column(
                        controls,
                        spacing=15,
                        horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
                    ),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
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
        if stt_value == "Qwen ASR":
            self._settings.provider.stt = STTProviderName.QWEN_ASR
        else:
            self._settings.provider.stt = STTProviderName.DEEPGRAM

        # Check STT provider compatibility with current source language
        source_lang = self._settings.languages.source_language
        stt_provider = self._settings.provider.stt.value
        warning = get_stt_compatibility_warning(source_lang, stt_provider)
        if warning and self.page:
            self.page.show_dialog(ft.SnackBar(
                ft.Text(warning),
                bgcolor=colors.ORANGE_700,
                duration=4000,
            ))

        new_llm = LLMProviderName.GEMINI if self.llm_provider.value == "Google Gemini" else LLMProviderName.QWEN

        # Load provider-specific prompt when LLM provider changes
        if self._settings.provider.llm != new_llm:
            self._settings.provider.llm = new_llm
            provider_name = "gemini" if new_llm == LLMProviderName.GEMINI else "qwen"
            self.prompt_provider_label.value = f"Prompt for: {provider_name.capitalize()}"
            self.system_prompt.value = load_prompt_for_provider(provider_name)
            self._settings.system_prompt = self.system_prompt.value
            if self.prompt_provider_label.page:
                self.prompt_provider_label.update()
            if self.system_prompt.page:
                self.system_prompt.update()
        else:
            self._settings.provider.llm = new_llm

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

        is_deepgram_stt = stt_provider == STTProviderName.DEEPGRAM
        self.deepgram_api_key.visible = is_deepgram_stt
        self.deepgram_stt_model.visible = is_deepgram_stt
        self.verify_deepgram_btn.visible = is_deepgram_stt

        # Qwen ASR uses alibaba_api_key
        is_qwen_asr = stt_provider == STTProviderName.QWEN_ASR

        self.google_api_key.visible = True
        self.verify_google_btn.visible = True
        # Show Alibaba key for Qwen ASR or Qwen LLM
        self.alibaba_api_key.visible = True
        self.verify_alibaba_btn.visible = True

    def _on_verify_req(self, provider: str, key: str, btn_control: ft.Control) -> None:
        if not self.on_verify_api_key:
            return
        
        if not key:
            self.page.show_dialog(ft.SnackBar(ft.Text("API Key is empty!"), bgcolor=colors.RED_400))
            return

        async def _run():
            original_icon = btn_control.icon
            original_color = btn_control.icon_color
            
            btn_control.icon = icons.HOURGLASS_TOP_ROUNDED
            btn_control.icon_color = colors.BLUE_400
            if btn_control.page:
                btn_control.update()
            
            try:
                success, msg = await self.on_verify_api_key(provider, key)
                if success:
                    self.page.show_dialog(ft.SnackBar(ft.Text(f"{provider.capitalize()} Verified!"), bgcolor=colors.GREEN_400))
                    btn_control.icon = icons.CHECK_CIRCLE_ROUNDED
                    btn_control.icon_color = colors.GREEN_400
                else:
                    logger.error(f"Verification failed for {provider}: {msg}")
                    # Also write to app logs UI
                    self.page.show_dialog(ft.SnackBar(ft.Text(f"Failed: {msg}"), bgcolor=colors.RED_400))
                    btn_control.icon = icons.ERROR_OUTLINE_ROUNDED
                    btn_control.icon_color = colors.RED_400
            except Exception as e:
                self.page.show_dialog(ft.SnackBar(ft.Text(f"Error: {e}"), bgcolor=colors.RED_400))
                btn_control.icon = icons.ERROR_OUTLINE_ROUNDED
                btn_control.icon_color = colors.RED_400
            
            if btn_control.page:
                btn_control.update()
            
            await asyncio.sleep(3)
            
            btn_control.icon = original_icon
            btn_control.icon_color = original_color
            if btn_control.page:
                btn_control.update()

        self.page.run_task(_run)

    def _on_setting_change(self, e) -> None:
        _ = e
        if self._settings is None:
            return

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
        try:
            import sounddevice as sd  # type: ignore

            hostapi_index: int | None = None
            if host_api:
                for idx, item in enumerate(sd.query_hostapis()):
                    name = str(item.get("name", "") or "")
                    # Exact match since dropdown options now come from sounddevice directly
                    if name == host_api:
                        hostapi_index = idx
                        break
                
                # If user selected a host API but it wasn't found, skip all devices
                if hostapi_index is None:
                    devices = ["(Default)"]  # No matching host API

            for dev in sd.query_devices():
                if int(dev.get("max_input_channels", 0) or 0) <= 0:
                    continue
                if hostapi_index is not None and int(dev.get("hostapi", -1) or -1) != hostapi_index:
                    continue
                name = str(dev.get("name", "") or "").strip()
                if name:
                    devices.append(name)
        except Exception as e:
            logger.warning(f"Failed to enumerate microphones: {e}")

        # Preserve selection if possible.
        current = self.microphone.value
        self.microphone.options = [ft.dropdown.Option(d) for d in devices]
        if current in devices:
            self.microphone.value = current
        else:
            self.microphone.value = "(Default)"
        
        # Force UI update if component is attached to page
        if self.microphone.page:
            self.microphone.update()

    def _populate_host_apis(self) -> None:
        """Populate audio host API dropdown with available APIs from the system.

        Only shows DirectSound and WASAPI to avoid user confusion:
        - MME: High latency, can be unstable
        - ASIO: Exclusive mode, prevents microphone sharing with other apps (e.g., VRChat)
        """
        options = [ft.dropdown.Option("(Default)")]
        # Only allow these host APIs for better compatibility
        allowed_apis = {"windows directsound", "windows wasapi"}
        try:
            import sounddevice as sd  # type: ignore
            for api in sd.query_hostapis():
                name = str(api.get("name", "") or "").strip()
                if name and name.lower() in allowed_apis:
                    options.append(ft.dropdown.Option(name))
        except Exception as e:
            logger.warning(f"Failed to enumerate host APIs: {e}")

        self.audio_host_api.options = options

    def _emit_settings_changed(self) -> None:
        if self._settings is None:
            return
        if self.on_settings_changed:
            self.on_settings_changed(self._settings)

    def refresh_prompt_if_empty(self) -> None:
        if self._settings is None:
            return
        current = (self.system_prompt.value or "").strip()
        if current:
            return
        provider_name = "gemini" if self._settings.provider.llm == LLMProviderName.GEMINI else "qwen"
        self.system_prompt.value = load_prompt_for_provider(provider_name)
        if self.system_prompt.page:
            self.system_prompt.update()
