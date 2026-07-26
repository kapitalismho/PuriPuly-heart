"""Audio settings component with Host API and Microphone."""

from __future__ import annotations

import logging
from typing import Callable

import flet as ft

from puripuly_heart.config.audio_host_api import (
    WINDOWS_DIRECTSOUND_HOST_API,
    WINDOWS_MME_HOST_API,
    WINDOWS_WASAPI_COMPATIBILITY_HOST_API,
    WINDOWS_WASAPI_HOST_API,
    normalize_input_host_api,
)
from puripuly_heart.ui.components.settings.settings_modal import OptionItem, SettingsModal
from puripuly_heart.ui.flet_runtime import is_control_mounted
from puripuly_heart.ui.i18n import t
from puripuly_heart.ui.theme import COLOR_ON_BACKGROUND, COLOR_PRIMARY

logger = logging.getLogger(__name__)
_CENTER_ALIGNMENT = ft.Alignment(0, 0)


class AudioSettings(ft.Column):
    """Audio settings for microphone and desktop loopback capture."""

    def __init__(
        self,
        on_change: Callable[[], None] | None = None,
    ):
        self._on_change = on_change
        self._default_option_label = t("settings.default_option")

        # Current selections
        self._current_host_api = ""
        self._current_microphone = ""
        self._current_desktop_output_device = ""
        self._current_desktop_vad_threshold = 0.5
        self._current_desktop_hangover_ms = 700
        self._current_desktop_pre_roll_ms = 500

        self._host_api_label = self._build_section_label(t("settings.audio_host_api"))
        self._microphone_label = self._build_section_label(t("settings.microphone"))
        self._desktop_output_label = self._build_section_label(
            t("settings.desktop_audio.output_device")
        )

        # Clickable text for Host API
        self._host_api_text = self._build_clickable_text(
            self._default_option_label,
            self._on_host_api_click,
        )

        # Clickable text for Microphone
        self._mic_text = self._build_clickable_text(
            self._default_option_label,
            self._on_mic_click,
        )

        self._desktop_output_text = self._build_clickable_text(
            self._default_option_label,
            self._on_desktop_output_click,
        )

        super().__init__(
            controls=[
                self._host_api_label,
                self._host_api_text,
                ft.Container(height=8),
                self._microphone_label,
                self._mic_text,
                ft.Container(height=12),
                self._desktop_output_label,
                self._desktop_output_text,
            ],
            spacing=8,
            expand=True,
            alignment=ft.MainAxisAlignment.START,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        )

    def _build_section_label(self, text: str) -> ft.Text:
        return ft.Text(text, size=15, color=COLOR_PRIMARY)

    def _build_clickable_text(self, text: str, on_click) -> ft.Container:
        """Build a clickable centered text with hover effect."""
        text_control = ft.Text(
            text,
            size=28,
            color=COLOR_ON_BACKGROUND,
            text_align=ft.TextAlign.CENTER,
            max_lines=2,
            overflow=ft.TextOverflow.ELLIPSIS,
        )
        return ft.Container(
            content=text_control,
            alignment=_CENTER_ALIGNMENT,
            expand=True,
            on_click=on_click,
            on_hover=self._on_text_hover,
        )

    def _build_numeric_field(self, *, label: str, value: str, on_change_end) -> ft.TextField:
        return ft.TextField(
            label=label,
            value=value,
            dense=True,
            expand=True,
            text_align=ft.TextAlign.CENTER,
            on_blur=on_change_end,
            on_submit=on_change_end,
        )

    def _host_api_label_for(self, value: str) -> str:
        """Return the localized display label for a persisted host API value."""
        host_api = str(value or "").strip()
        if not host_api:
            return self._default_option_label

        label_key_by_value = {
            WINDOWS_MME_HOST_API: "settings.audio_host_api.option.windows_mme",
            WINDOWS_WASAPI_HOST_API: "settings.audio_host_api.option.windows_wasapi",
            WINDOWS_WASAPI_COMPATIBILITY_HOST_API: (
                "settings.audio_host_api.option.windows_wasapi_compatibility"
            ),
            WINDOWS_DIRECTSOUND_HOST_API: "settings.audio_host_api.option.windows_directsound",
        }
        label_key = label_key_by_value.get(host_api)
        if label_key is None:
            return host_api
        return t(label_key)

    @property
    def host_api_display_label(self) -> str:
        return self._host_api_label_for(self._current_host_api)

    def _on_text_hover(self, e: ft.ControlEvent) -> None:
        """Handle hover effect on clickable text."""
        container = e.control
        text_control = container.content
        if e.data == "true":
            text_control.color = COLOR_PRIMARY
        else:
            text_control.color = COLOR_ON_BACKGROUND
        container.update()

    @property
    def host_api(self) -> str:
        """Get selected host API (empty string for default)."""
        return self._current_host_api

    @host_api.setter
    def host_api(self, val: str) -> None:
        self._current_host_api = val
        display = self._host_api_label_for(val)
        self._host_api_text.content.value = display
        if is_control_mounted(self._host_api_text):
            self._host_api_text.update()

    @property
    def microphone(self) -> str:
        """Get selected microphone (empty string for default)."""
        return self._current_microphone

    @microphone.setter
    def microphone(self, val: str) -> None:
        self._current_microphone = val
        display = val or self._default_option_label
        self._mic_text.content.value = display
        if is_control_mounted(self._mic_text):
            self._mic_text.update()

    @property
    def desktop_output_device(self) -> str:
        return self._current_desktop_output_device

    @desktop_output_device.setter
    def desktop_output_device(self, val: str) -> None:
        self._current_desktop_output_device = val
        display = val or self._default_option_label
        self._desktop_output_text.content.value = display
        if is_control_mounted(self._desktop_output_text):
            self._desktop_output_text.update()

    @property
    def desktop_vad_threshold(self) -> float:
        return self._current_desktop_vad_threshold

    @desktop_vad_threshold.setter
    def desktop_vad_threshold(self, val: float) -> None:
        self._current_desktop_vad_threshold = float(val)
        field = getattr(self, "_desktop_vad_field", None)
        if field is not None:
            field.value = f"{self._current_desktop_vad_threshold:.2f}"
            if is_control_mounted(field):
                field.update()

    @property
    def desktop_hangover_ms(self) -> int:
        return self._current_desktop_hangover_ms

    @desktop_hangover_ms.setter
    def desktop_hangover_ms(self, val: int) -> None:
        self._current_desktop_hangover_ms = int(val)
        field = getattr(self, "_desktop_hangover_field", None)
        if field is not None:
            field.value = str(self._current_desktop_hangover_ms)
            if is_control_mounted(field):
                field.update()

    @property
    def desktop_pre_roll_ms(self) -> int:
        return self._current_desktop_pre_roll_ms

    @desktop_pre_roll_ms.setter
    def desktop_pre_roll_ms(self, val: int) -> None:
        self._current_desktop_pre_roll_ms = int(val)
        field = getattr(self, "_desktop_pre_roll_field", None)
        if field is not None:
            field.value = str(self._current_desktop_pre_roll_ms)
            if is_control_mounted(field):
                field.update()

    def _get_host_api_options(self) -> list[OptionItem]:
        """Get available host API options."""
        options = [OptionItem(value="", label=self._default_option_label)]

        try:
            import sounddevice as sd

            available_host_apis = {
                str(api.get("name", "") or "").strip().casefold() for api in sd.query_hostapis()
            }
        except Exception as e:
            logger.warning(f"Failed to enumerate host APIs: {e}")
            return options

        if WINDOWS_MME_HOST_API.casefold() in available_host_apis:
            options.append(
                OptionItem(
                    value=WINDOWS_MME_HOST_API,
                    label=self._host_api_label_for(WINDOWS_MME_HOST_API),
                )
            )

        if WINDOWS_WASAPI_HOST_API.casefold() in available_host_apis:
            options.append(
                OptionItem(
                    value=WINDOWS_WASAPI_HOST_API,
                    label=self._host_api_label_for(WINDOWS_WASAPI_HOST_API),
                )
            )
            options.append(
                OptionItem(
                    value=WINDOWS_WASAPI_COMPATIBILITY_HOST_API,
                    label=self._host_api_label_for(WINDOWS_WASAPI_COMPATIBILITY_HOST_API),
                )
            )

        if WINDOWS_DIRECTSOUND_HOST_API.casefold() in available_host_apis:
            options.append(
                OptionItem(
                    value=WINDOWS_DIRECTSOUND_HOST_API,
                    label=self._host_api_label_for(WINDOWS_DIRECTSOUND_HOST_API),
                )
            )

        return options

    def _get_microphone_options(self) -> list[OptionItem]:
        """Get available microphone options based on selected host API."""
        options = [OptionItem(value="", label=self._default_option_label)]

        try:
            import sounddevice as sd

            hostapi_index: int | None = None
            profile = normalize_input_host_api(self._current_host_api)
            actual_host_api = profile.actual_host_api
            if actual_host_api:
                for idx, item in enumerate(sd.query_hostapis()):
                    name = str(item.get("name", "") or "")
                    if name == actual_host_api:
                        hostapi_index = idx
                        break

            for dev in sd.query_devices():
                if int(dev.get("max_input_channels", 0) or 0) <= 0:
                    continue
                device_hostapi = dev.get("hostapi", -1)
                if device_hostapi is None:
                    device_hostapi = -1
                if hostapi_index is not None and int(device_hostapi) != hostapi_index:
                    continue
                name = str(dev.get("name", "") or "").strip()
                if name:
                    options.append(OptionItem(value=name, label=name))
        except Exception as e:
            logger.warning(f"Failed to enumerate microphones: {e}")

        return options

    def _get_desktop_output_options(self) -> list[OptionItem]:
        options = [OptionItem(value="", label=self._default_option_label)]

        manager = None
        try:
            import pyaudiowpatch as pyaudio  # type: ignore

            manager = pyaudio.PyAudio()
            seen: set[str] = set()
            for info in manager.get_loopback_device_info_generator():
                name = str(info.get("name", "") or "").strip()
                if not name or name in seen:
                    continue
                seen.add(name)
                options.append(OptionItem(value=name, label=name))
        except Exception as e:
            logger.warning(f"Failed to enumerate desktop loopback outputs: {e}")
        finally:
            if manager is not None:
                try:
                    manager.terminate()
                except Exception:
                    pass

        return options

    def _on_host_api_click(self, e) -> None:
        """Open Host API selection modal."""
        if not is_control_mounted(self):
            return
        options = self._get_host_api_options()
        modal = SettingsModal(
            self.page,
            t("settings.audio_host_api"),
            options,
            self._on_host_api_selected,
            show_description=False,
        )
        modal.open(self._current_host_api)

    def _on_host_api_selected(self, value: str) -> None:
        """Handle host API selection from modal."""
        self.host_api = value
        # Reset microphone when host API changes
        self.microphone = ""
        self._emit_change()

    def _on_mic_click(self, e) -> None:
        """Open Microphone selection modal."""
        if not is_control_mounted(self):
            return
        options = self._get_microphone_options()
        modal = SettingsModal(
            self.page,
            t("settings.microphone"),
            options,
            self._on_mic_selected,
            show_description=False,
        )
        modal.open(self._current_microphone)

    def _on_mic_selected(self, value: str) -> None:
        """Handle microphone selection from modal."""
        self.microphone = value
        self._emit_change()

    def _on_desktop_output_click(self, e) -> None:
        """Open desktop loopback output selection modal."""
        if not is_control_mounted(self):
            return
        options = self._get_desktop_output_options()
        modal = SettingsModal(
            self.page,
            t("settings.desktop_audio.output_device"),
            options,
            self._on_desktop_output_selected,
            show_description=False,
        )
        modal.open(self._current_desktop_output_device)

    def _on_desktop_output_selected(self, value: str) -> None:
        self.desktop_output_device = value
        self._emit_change()

    def _on_desktop_vad_threshold_change(self, e) -> None:
        self.desktop_vad_threshold = self._parse_float(
            e.control.value,
            fallback=self._current_desktop_vad_threshold,
            minimum=0.0,
            maximum=1.0,
        )
        self._emit_change()

    def _on_desktop_hangover_change(self, e) -> None:
        self.desktop_hangover_ms = self._parse_int(
            e.control.value,
            fallback=self._current_desktop_hangover_ms,
            minimum=0,
        )
        self._emit_change()

    def _on_desktop_pre_roll_change(self, e) -> None:
        self.desktop_pre_roll_ms = self._parse_int(
            e.control.value,
            fallback=self._current_desktop_pre_roll_ms,
            minimum=0,
        )
        self._emit_change()

    def _parse_float(
        self,
        raw_value: str,
        *,
        fallback: float,
        minimum: float,
        maximum: float | None = None,
    ) -> float:
        try:
            parsed = float(raw_value)
        except (TypeError, ValueError):
            parsed = fallback
        if parsed < minimum:
            parsed = minimum
        if maximum is not None and parsed > maximum:
            parsed = maximum
        return parsed

    def _parse_int(
        self,
        raw_value: str,
        *,
        fallback: int,
        minimum: int,
    ) -> int:
        try:
            parsed = int(raw_value)
        except (TypeError, ValueError):
            parsed = fallback
        return max(minimum, parsed)

    def _emit_change(self) -> None:
        if self._on_change:
            self._on_change()

    def apply_locale(self) -> None:
        """Update labels when locale changes."""
        self._default_option_label = t("settings.default_option")
        self._host_api_label.value = t("settings.audio_host_api")
        self._microphone_label.value = t("settings.microphone")
        self._desktop_output_label.value = t("settings.desktop_audio.output_device")

        self._host_api_text.content.value = self._host_api_label_for(self._current_host_api)
        self._mic_text.content.value = self._current_microphone or self._default_option_label
        self._desktop_output_text.content.value = (
            self._current_desktop_output_device or self._default_option_label
        )

        if is_control_mounted(self):
            self.update()
