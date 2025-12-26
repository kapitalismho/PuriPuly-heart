from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from pathlib import Path

import flet as ft

from puripuly_heart.app.wiring import (
    create_llm_provider,
    create_secret_store,
    create_stt_backend,
)
from puripuly_heart.config.settings import (
    AppSettings,
    STTProviderName,
    load_settings,
    save_settings,
)
from puripuly_heart.core.audio.source import (
    SoundDeviceAudioSource,
    resolve_sounddevice_input_device,
)
from puripuly_heart.core.clock import SystemClock
from puripuly_heart.core.orchestrator.hub import ClientHub
from puripuly_heart.core.osc.smart_queue import SmartOscQueue
from puripuly_heart.core.osc.udp_sender import VrchatOscUdpSender
from puripuly_heart.core.stt.controller import ManagedSTTProvider
from puripuly_heart.core.vad.bundled import SILERO_VAD_VERSION, ensure_silero_vad_onnx
from puripuly_heart.core.vad.gating import VadGating
from puripuly_heart.core.vad.silero import SileroVadOnnx
from puripuly_heart.providers.llm.gemini import GeminiLLMProvider
from puripuly_heart.providers.llm.qwen import QwenLLMProvider
from puripuly_heart.providers.stt.deepgram import DeepgramRealtimeSTTBackend
from puripuly_heart.providers.stt.qwen_asr import QwenASRRealtimeSTTBackend
from puripuly_heart.ui.event_bridge import UIEventBridge

logger = logging.getLogger(__name__)

# Hardcoded STT session reset deadline (not configurable via settings)
STT_RESET_DEADLINE_S = 180.0


@dataclass(slots=True)
class GuiController:
    page: ft.Page
    app: object
    config_path: Path

    settings: AppSettings | None = None
    clock: SystemClock = SystemClock()

    sender: VrchatOscUdpSender | None = None
    osc: SmartOscQueue | None = None
    hub: ClientHub | None = None

    _bridge_task: asyncio.Task[None] | None = None
    _mic_task: asyncio.Task[None] | None = None
    _audio_source: SoundDeviceAudioSource | None = None
    _vad: VadGating | None = None

    async def start(self) -> None:
        self.settings = self._load_or_init_settings(self.config_path)
        self._sync_ui_from_settings()

        # Attach log handler to LogsView for GUI log display
        logs_view = getattr(self.app, "view_logs", None)
        if logs_view is not None:
            logs_view.attach_log_handler()

        await self._init_pipeline()

        assert self.hub is not None

        dash = getattr(self.app, "view_dashboard", None)
        if dash is not None:
            # Set needs_key flags (used when user tries to toggle)
            dash.translation_needs_key = self.hub.llm is None
            dash.stt_needs_key = self.hub.stt is None
            # Set initial enabled states (all start as off/gray)
            dash.set_translation_enabled(False)
            dash.set_stt_enabled(False)
            self.hub.translation_enabled = False

        await self.hub.start(auto_flush_osc=True)

        bridge = UIEventBridge(app=self.app, event_queue=self.hub.ui_events)
        self._bridge_task = asyncio.create_task(bridge.run())

    async def stop(self) -> None:
        await self.set_stt_enabled(False)

        if self._bridge_task:
            self._bridge_task.cancel()
            await asyncio.gather(self._bridge_task, return_exceptions=True)
            self._bridge_task = None

        if self.hub is not None:
            with contextlib.suppress(Exception):
                await self.hub.stop()
            self.hub = None

        if self.sender is not None:
            with contextlib.suppress(Exception):
                self.sender.close()
            self.sender = None
        self.osc = None

    async def set_translation_enabled(self, enabled: bool) -> None:
        if self.hub is None:
            return
        if enabled and self.hub.llm is None:
            self.hub.translation_enabled = False
            dash = getattr(self.app, "view_dashboard", None)
            if dash is not None:
                dash.set_translation_enabled(False)
            self._log_error("Translation is ON but LLM provider is not configured.")
            return

        # Clear context history when toggling translation
        self.hub.clear_context()
        self.hub.translation_enabled = bool(enabled)

    async def set_stt_enabled(self, enabled: bool) -> None:
        if not enabled:
            await self._stop_mic_loop()
            if self.hub is not None:
                with contextlib.suppress(Exception):
                    await self.hub.stt.close()
            return

        await self._start_mic_loop()
        # Pre-warm STT session for faster first response
        if self.hub is not None and self.hub.stt is not None:
            with contextlib.suppress(Exception):
                await self.hub.stt.warmup()

    async def submit_text(self, text: str) -> None:
        if self.hub is None:
            return
        try:
            await self.hub.submit_text(text, source="You")
        except Exception as exc:
            self._log_error(f"Submit failed: {exc}")

    async def apply_settings(self, settings: AppSettings) -> None:
        self.settings = settings
        self._save_settings()

        if self.hub is not None:
            self.hub.source_language = settings.languages.source_language
            self.hub.target_language = settings.languages.target_language
            self.hub.system_prompt = settings.system_prompt

        # Audio/VAD changes apply on next STT start; if STT is running, restart mic loop.
        if self._mic_task is not None:
            await self._stop_mic_loop()
            await self._start_mic_loop()

    async def verify_api_key(self, provider: str, key: str) -> tuple[bool, str]:
        """Verify API key using the respective provider's static check. Returns (success, error_msg)."""
        if not key:
            return False, "API Key is empty"

        try:
            success = False
            if provider == "google":
                success = await GeminiLLMProvider.verify_api_key(key)
            elif provider == "alibaba_beijing":
                # Beijing region endpoint
                success = await QwenLLMProvider.verify_api_key(
                    key, base_url="https://dashscope.aliyuncs.com/api/v1"
                )
            elif provider == "alibaba_singapore":
                # Singapore region endpoint
                success = await QwenLLMProvider.verify_api_key(
                    key, base_url="https://dashscope-intl.aliyuncs.com/api/v1"
                )
            elif provider == "deepgram":
                success = await DeepgramRealtimeSTTBackend.verify_api_key(key)
            else:
                return False, f"Unknown provider: {provider}"

            if success:
                return True, "Verification successful"
            else:
                return False, "Verification failed (check logs/console for details)"
        except Exception as exc:
            msg = f"Verification error for {provider}: {exc}"
            self._log_error(msg)
            return False, str(exc)

    async def apply_providers(self) -> None:
        if self.settings is None:
            return
        # Rebuild LLM provider on the fly; STT provider changes require full pipeline rebuild.
        await self._rebuild_pipeline(rebuild_stt=True)

    def _load_or_init_settings(self, path: Path) -> AppSettings:
        if path.exists():
            return load_settings(path)
        settings = AppSettings()
        path.parent.mkdir(parents=True, exist_ok=True)
        save_settings(path, settings)
        return settings

    async def _rebuild_pipeline(self, *, rebuild_stt: bool) -> None:
        _ = rebuild_stt
        if self._bridge_task:
            self._bridge_task.cancel()
            await asyncio.gather(self._bridge_task, return_exceptions=True)
            self._bridge_task = None

        await self.set_stt_enabled(False)
        if self.hub is not None:
            with contextlib.suppress(Exception):
                await self.hub.stop()
        if self.sender is not None:
            with contextlib.suppress(Exception):
                self.sender.close()
        self.sender = None
        self.osc = None
        self.hub = None
        await self._init_pipeline()
        assert self.hub is not None

        dash = getattr(self.app, "view_dashboard", None)
        if dash is not None:
            dash.set_translation_needs_key(self.hub.llm is None)
            dash.set_stt_needs_key(self.hub.stt is None)

            self.hub.translation_enabled = (
                bool(getattr(dash, "is_translation_on", True)) and self.hub.llm is not None
            )
            dash.set_translation_enabled(self.hub.translation_enabled)

        await self.hub.start(auto_flush_osc=True)

        bridge = UIEventBridge(app=self.app, event_queue=self.hub.ui_events)
        self._bridge_task = asyncio.create_task(bridge.run())

        # Trigger background verification to sync button colors
        asyncio.create_task(self._verify_and_update_status())

    async def _init_pipeline(self) -> None:
        assert self.settings is not None
        secrets = create_secret_store(self.settings.secrets, config_path=self.config_path)

        llm = None
        with contextlib.suppress(Exception):
            llm = create_llm_provider(self.settings, secrets=secrets)

        stt = None
        try:
            backend = create_stt_backend(self.settings, secrets=secrets)
            stt = ManagedSTTProvider(
                backend=backend,
                sample_rate_hz=self.settings.audio.internal_sample_rate_hz,
                clock=self.clock,
                reset_deadline_s=STT_RESET_DEADLINE_S,
                drain_timeout_s=self.settings.stt.drain_timeout_s,
                bridging_ms=self.settings.audio.ring_buffer_ms,
            )
        except Exception as exc:
            self._log_error(f"STT backend not available: {exc}")

        sender = VrchatOscUdpSender(
            host=self.settings.osc.host,
            port=self.settings.osc.port,
            chatbox_address=self.settings.osc.chatbox_address,
            chatbox_send=self.settings.osc.chatbox_send,
            chatbox_clear=self.settings.osc.chatbox_clear,
        )
        osc = SmartOscQueue(
            sender=sender,
            clock=self.clock,
            max_chars=self.settings.osc.chatbox_max_chars,
            cooldown_s=self.settings.osc.cooldown_s,
            ttl_s=self.settings.osc.ttl_s,
        )

        hub = ClientHub(
            stt=stt,
            llm=llm,
            osc=osc,
            clock=self.clock,
            source_language=self.settings.languages.source_language,
            target_language=self.settings.languages.target_language,
            system_prompt=self.settings.system_prompt,
            fallback_transcript_only=True,
            translation_enabled=True,
            hangover_s=1.1,  # Match VadGating.hangover_ms (1100ms)
        )

        self.sender = sender
        self.osc = osc
        self.hub = hub

    async def _start_mic_loop(self) -> None:
        if self._mic_task is not None:
            return
        assert self.settings is not None
        assert self.hub is not None

        try:
            model_path = ensure_silero_vad_onnx()
        except Exception as exc:
            self._log_error(f"Failed to prepare Silero VAD model ({SILERO_VAD_VERSION}): {exc}")
            return

        vad = VadGating(
            engine=SileroVadOnnx(model_path=model_path),
            sample_rate_hz=self.settings.audio.internal_sample_rate_hz,
            ring_buffer_ms=self.settings.audio.ring_buffer_ms,
            speech_threshold=self.settings.stt.vad_speech_threshold,
        )

        device_idx = None
        with contextlib.suppress(Exception):
            device_idx = resolve_sounddevice_input_device(
                host_api=self.settings.audio.input_host_api,
                device=self.settings.audio.input_device,
            )

        source = SoundDeviceAudioSource(
            sample_rate_hz=None,  # Use device default; resampled later to internal rate
            channels=self.settings.audio.internal_channels,
            device=device_idx,
        )

        self._vad = vad
        self._audio_source = source
        self._mic_task = asyncio.create_task(self._run_mic_loop())

    async def _stop_mic_loop(self) -> None:
        if self._mic_task is None:
            return
        self._mic_task.cancel()
        await asyncio.gather(self._mic_task, return_exceptions=True)
        self._mic_task = None

        if self._audio_source is not None:
            with contextlib.suppress(Exception):
                await self._audio_source.close()
            self._audio_source = None
        self._vad = None

    async def _run_mic_loop(self) -> None:
        assert self.hub is not None
        assert self._audio_source is not None
        assert self._vad is not None

        from puripuly_heart.app.headless_mic import run_audio_vad_loop

        try:
            await run_audio_vad_loop(
                source=self._audio_source,
                vad=self._vad,
                hub=self.hub,
                target_sample_rate_hz=self.settings.audio.internal_sample_rate_hz,  # type: ignore[union-attr]
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._log_error(f"Mic loop error: {exc}")

    def _save_settings(self) -> None:
        assert self.settings is not None
        try:
            save_settings(self.config_path, self.settings)
        except Exception as exc:
            self._log_error(f"Failed to save settings: {exc}")

    def _sync_ui_from_settings(self) -> None:
        settings = self.settings
        if settings is None:
            return

        # Dashboard language dropdowns are initialized by the view; set values if present.
        with contextlib.suppress(Exception):
            dash = getattr(self.app, "view_dashboard", None)
            if dash is not None:
                dash.set_languages_from_codes(
                    settings.languages.source_language, settings.languages.target_language
                )

        with contextlib.suppress(Exception):
            view_settings = getattr(self.app, "view_settings", None)
            if view_settings is not None:
                view_settings.load_from_settings(settings, config_path=self.config_path)

    def _log_error(self, message: str) -> None:
        logger.error(message)
        with contextlib.suppress(Exception):
            logs = getattr(self.app, "view_logs", None)
            if logs is not None:
                logs.append_log(f"ERROR: {message}")

    async def _verify_and_update_status(self) -> None:
        """Background task to verify keys and update dashboard status."""
        if self.settings is None:
            return

        dash = getattr(self.app, "view_dashboard", None)
        if dash is None:
            return

        # 1. Verify LLM
        llm_valid = False
        if self.hub and self.hub.llm:
            # It was created, but is the key valid?
            try:
                secrets = create_secret_store(self.settings.secrets, config_path=self.config_path)
                provider_name = self.settings.provider.llm
                key = ""
                if provider_name == "gemini":
                    key = secrets.get("google_api_key") or ""
                    llm_valid = await GeminiLLMProvider.verify_api_key(key)
                elif provider_name == "qwen":
                    key = secrets.get("alibaba_api_key") or ""
                    llm_valid = await QwenLLMProvider.verify_api_key(key)
                else:
                    # Assume valid for others or if no key usage known
                    llm_valid = True
            except Exception:
                llm_valid = False

        # If LLM verification failed, force needs_key = True even if provider object exists
        if not llm_valid:
            dash.set_translation_needs_key(True)
            # If it was enabled, we potentially disable it or just let the warning show on next interaction
            # User request: "Validation Fail -> Orange". Implicitly, if it's ON and fails, maybe we should turn it OFF?
            # For now, setting needs_key=True ensures that if they try to toggle, it warns.
            # If it is currently ON, we might want to flag it.
            if self.hub:
                self.hub.translation_enabled = False  # Disable internally
            dash.set_translation_enabled(False)  # Visually turn off
        else:
            dash.set_translation_needs_key(False)

        # 2. Verify STT
        stt_valid = False
        if self.hub and self.hub.stt:
            try:
                secrets = create_secret_store(self.settings.secrets, config_path=self.config_path)
                provider_name = self.settings.provider.stt

                if provider_name == STTProviderName.DEEPGRAM:
                    key = secrets.get("deepgram_api_key") or ""
                    stt_valid = await DeepgramRealtimeSTTBackend.verify_api_key(key)
                elif provider_name == STTProviderName.QWEN_ASR:
                    key = secrets.get("alibaba_api_key") or ""
                    stt_valid = await QwenASRRealtimeSTTBackend.verify_api_key(key)
                else:
                    stt_valid = True
            except Exception:
                stt_valid = False

        if not stt_valid:
            dash.set_stt_needs_key(True)
            if self.hub:
                # Close STT backend?
                pass
            dash.set_stt_enabled(False)
        else:
            dash.set_stt_needs_key(False)
