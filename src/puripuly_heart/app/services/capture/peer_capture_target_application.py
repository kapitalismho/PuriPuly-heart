from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass, field

from puripuly_heart.app.ports.peer_capture_target import (
    LoopbackDeviceInventoryPort,
    PeerCaptureTargetRuntimeEffectsPort,
    ProcessCaptureInventoryPort,
)
from puripuly_heart.app.ports.settings_view import GeneralSettingsSnapshot
from puripuly_heart.app.ports.ui_models import OptionItem
from puripuly_heart.app.services.canonical_settings_persistence import SettingsOwner
from puripuly_heart.app.services.settings_application import settings_view_surface_snapshots
from puripuly_heart.config.capture_target_resolution import (
    resolve_desktop_audio_capture_target,
)
from puripuly_heart.config.resolved import ResolvedDesktopAudioCaptureTarget
from puripuly_heart.config.settings_vnext.schema import (
    CaptureTargetIntent,
    ProcessCaptureTargetIntent,
)

from .peer_capture_target import PeerCaptureTargetResolutionService

Localizer = Callable[[str], str]
SettingsPresentationSink = Callable[[GeneralSettingsSnapshot], None]
WarningReset = Callable[[], None]


@dataclass(slots=True)
class PeerCaptureTargetApplicationOwner:
    settings: SettingsOwner
    localize: Localizer
    processes: ProcessCaptureInventoryPort
    devices: LoopbackDeviceInventoryPort
    runtime_effects: PeerCaptureTargetRuntimeEffectsPort
    settings_presentation_sink: SettingsPresentationSink
    warning_reset: WarningReset
    _target_resolution: PeerCaptureTargetResolutionService = field(
        init=False,
        default_factory=PeerCaptureTargetResolutionService,
        repr=False,
    )

    def summary(self) -> str:
        target = self._resolve_target()
        if target is None:
            return self.localize("settings.default_option")
        if target.kind == "named_output_device":
            return target.device_name or self.localize("settings.default_option")
        if target.kind == "process":
            return self._process_option_label(
                self._process_target_from_resolved(target),
                "",
            )
        return self.localize("settings.default_option")

    def options(self) -> list[OptionItem]:
        return [*self.process_options(), *self.device_options()]

    def process_options(self) -> list[OptionItem]:
        section = self.localize("settings.desktop_audio.section.process")
        options: list[OptionItem] = []
        seen_values: set[str] = set()
        for candidate in self.processes.candidates():
            value = self.encode_process_option(candidate.target)
            seen_values.add(value)
            options.append(
                OptionItem(
                    value=value,
                    label=self._process_option_label(candidate.target, candidate.name),
                    disabled=not candidate.enabled,
                    section=section,
                )
            )
        current_value = self.current_value()
        if current_value.startswith("process:") and current_value not in seen_values:
            process = self.decode_option(current_value).process
            if process is not None:
                options.insert(
                    0,
                    OptionItem(
                        value=current_value,
                        label=self._process_option_label(process, ""),
                        section=section,
                    ),
                )
        options.sort(key=lambda option: option.disabled)
        return options

    def device_options(self) -> list[OptionItem]:
        section = self.localize("settings.desktop_audio.section.device")
        return [
            OptionItem(
                value="device:",
                label=self.localize("settings.default_option"),
                section=section,
            ),
            *[
                OptionItem(
                    value=f"device:{device}",
                    label=device,
                    section=section,
                )
                for device in self.devices.names()
            ],
        ]

    def current_value(self) -> str:
        target = self._resolve_target()
        if target is None:
            return "device:"
        if target.kind == "process":
            return self.encode_process_option(self._process_target_from_resolved(target))
        if target.kind == "named_output_device":
            return f"device:{target.device_name or ''}"
        return "device:"

    def _resolve_target(self) -> ResolvedDesktopAudioCaptureTarget | None:
        canonical = self.settings.canonical
        if canonical is None:
            return None
        return resolve_desktop_audio_capture_target(canonical.intent.desktop_audio.capture_target)

    async def apply(self, value: str) -> None:
        current = self.settings.canonical
        if current is None:
            return
        apply_capture_target = getattr(self.settings, "apply_capture_target", None)
        if apply_capture_target is not None:
            next_settings = apply_capture_target(self.decode_option(value))
        else:
            next_settings = self.settings.update_capture_target(
                current,
                self.decode_option(value),
            )
        self.settings.canonical = next_settings
        self.settings.authoritative = True
        self.settings.remember_projection(next_settings)
        self.warning_reset()
        await self.runtime_effects.apply_capture_target(next_settings)
        _provider, general, _prompt, _overlay = settings_view_surface_snapshots(next_settings)
        with contextlib.suppress(Exception):
            self.settings_presentation_sink(general)

    @staticmethod
    def encode_process_option(target: ProcessCaptureTargetIntent) -> str:
        if target.kind == "discord":
            return f"process:discord:{target.discord_channel}"
        if target.kind == "vrchat":
            return f"process:vrchat:{target.executable_identity}"
        return f"process:generic:{target.executable_identity}"

    @staticmethod
    def decode_option(value: str) -> CaptureTargetIntent:
        if value.startswith("process:"):
            payload = value[len("process:") :]
            kind, _, rest = payload.partition(":")
            if kind == "discord":
                return CaptureTargetIntent.process_target(ProcessCaptureTargetIntent.discord(rest))
            if kind == "vrchat":
                return CaptureTargetIntent.process_target(ProcessCaptureTargetIntent.vrchat(rest))
            return CaptureTargetIntent.process_target(
                ProcessCaptureTargetIntent.generic_executable(rest)
            )
        device_name = value[len("device:") :] if value.startswith("device:") else value
        if device_name:
            return CaptureTargetIntent.named_output_device(device_name)
        return CaptureTargetIntent.default_output_device()

    @staticmethod
    def _process_target_from_resolved(
        target: ResolvedDesktopAudioCaptureTarget,
    ) -> ProcessCaptureTargetIntent:
        if target.process_kind == "discord":
            return ProcessCaptureTargetIntent.discord(target.discord_channel or "")
        if target.process_kind == "vrchat":
            return ProcessCaptureTargetIntent.vrchat(target.executable_identity or "")
        return ProcessCaptureTargetIntent.generic_executable(target.executable_identity or "")

    def _process_option_label(
        self,
        target: ProcessCaptureTargetIntent,
        fallback_name: str,
    ) -> str:
        if target.kind == "vrchat":
            base = self.localize("settings.desktop_audio.process.vrchat")
        elif target.kind == "discord":
            channel = target.discord_channel or "stable"
            if channel == "ptb":
                base = self.localize("settings.desktop_audio.process.discord_ptb")
            elif channel == "canary":
                base = self.localize("settings.desktop_audio.process.discord_canary")
            else:
                base = self.localize("settings.desktop_audio.process.discord_stable")
        elif fallback_name:
            return fallback_name
        else:
            path = target.executable_identity or ""
            basename = path.rsplit("\\", 1)[-1]
            if basename.lower().endswith(".exe"):
                basename = basename[:-4]
            base = basename or self.localize("settings.default_option")
        if target.kind in {"vrchat", "discord"} and fallback_name:
            count_suffix = fallback_name.rsplit(" (", 1)
            if len(count_suffix) == 2 and count_suffix[1].endswith(")"):
                count = count_suffix[1][:-1]
                if count.isdigit():
                    return f"{base} ({count})"
        return base


__all__ = ["PeerCaptureTargetApplicationOwner"]
