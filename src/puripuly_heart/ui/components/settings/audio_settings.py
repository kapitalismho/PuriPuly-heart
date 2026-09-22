"""Audio settings state and option enumeration for Host API, Microphone and loopback output."""

from __future__ import annotations

import logging

from puripuly_heart.config.audio_host_api import (
    WINDOWS_DIRECTSOUND_HOST_API,
    WINDOWS_MME_HOST_API,
    WINDOWS_WASAPI_COMPATIBILITY_HOST_API,
    WINDOWS_WASAPI_HOST_API,
    normalize_input_host_api,
)
from puripuly_heart.ui.components.settings.settings_modal import OptionItem
from puripuly_heart.ui.i18n import t

logger = logging.getLogger(__name__)


class AudioSettings:
    """Holds the audio selections and enumerates their available options."""

    def __init__(self) -> None:
        self._default_option_label = t("settings.default_option")

        self._current_host_api = ""
        self._current_microphone = ""
        self._current_desktop_output_device = ""

    @property
    def host_api(self) -> str:
        """Get selected host API (empty string for default)."""
        return self._current_host_api

    @host_api.setter
    def host_api(self, val: str) -> None:
        self._current_host_api = val

    @property
    def microphone(self) -> str:
        """Get selected microphone (empty string for default)."""
        return self._current_microphone

    @microphone.setter
    def microphone(self, val: str) -> None:
        self._current_microphone = val

    @property
    def desktop_output_device(self) -> str:
        return self._current_desktop_output_device

    @desktop_output_device.setter
    def desktop_output_device(self, val: str) -> None:
        self._current_desktop_output_device = val

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

    def apply_locale(self) -> None:
        """Update option labels when locale changes."""
        self._default_option_label = t("settings.default_option")
