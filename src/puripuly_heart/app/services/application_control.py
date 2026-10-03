from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING
from uuid import uuid4

from puripuly_heart.app.language_selection import LanguageSelectionChange
from puripuly_heart.app.ports.managed_gemma_translation import ManagedGemmaTranslationSelection
from puripuly_heart.app.ports.settings_view import (
    ActivationNoticeSettingsIntent,
    AudioInputSettingsIntent,
    AudioSettingsIntent,
    ChatboxSourceSettingsIntent,
    ClipboardSettingsIntent,
    CloudFreeTierProvidersEdit,
    CustomSttEndpointEdit,
    CustomSttExtraEdit,
    CustomSttModelEdit,
    CustomVocabularySettingsIntent,
    DesktopAudioOutputSettingsIntent,
    DesktopOverlayBackgroundAlphaIntent,
    DesktopOverlaySizeIntent,
    DesktopOverlaySwapCaptionLanguagesIntent,
    LlmGpuDeviceEdit,
    LocaleSettingsIntent,
    LocalLlmBaseUrlEdit,
    LocalLlmExtraBodyEdit,
    LocalLlmModelEdit,
    ManagedReferralEdit,
    OscConnectionSettingsIntent,
    OverlayPeerOriginalSettingsIntent,
    OverlayTargetSettingsIntent,
    OverlayTranslationSettingsIntent,
    PeerExpectedLanguagesIntent,
    PeerSttProviderEdit,
    PeerVadHangoverIntent,
    PeerVadPreRollIntent,
    PeerVadSpeechThresholdIntent,
    PromptApplyIntent,
    ProviderApplyIntent,
    QwenBeijingApiHostEdit,
    QwenRegionEdit,
    QwenSingaporeApiHostEdit,
    SelfSttProviderEdit,
    SelfVadSettingsIntent,
    SonioxSpeakerDiarizationEdit,
    SttGpuDeviceEdit,
    SystemPromptEdit,
    TranslationHttpExtensionEdit,
    TranslationSelectionEdit,
    VrcMicInterceptSettingsIntent,
)
from puripuly_heart.app.services.application_control_events import ControlEvents
from puripuly_heart.app.services.canonical_settings_persistence import canonical_snapshot_values
from puripuly_heart.app.services.capture.peer_capture_target_application import (
    PeerCaptureTargetUnavailable,
)
from puripuly_heart.config.provider_values import QwenRegion, STTProviderName
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.config.translation_values import (
    TranslationConnection,
    TranslationModel,
    provider_llm_for_translation,
    supported_translation_connections,
)
from puripuly_heart.core.messages import (
    TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
    TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED,
    TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_INTERRUPTED,
    TransactionResult,
)

if TYPE_CHECKING:
    from puripuly_heart.app.services.ui_application import UiApplicationBoundary

TERMINAL = frozenset(
    {
        "applied",
        "degraded",
        "rejected",
        "persistence_failed",
        "failed",
        "action_required",
        "cancelled",
        "interrupted",
    }
)
COMMANDS = {
    "app.stop": "Request ordered host shutdown",
    "capture.set": "Set self or peer capture explicitly",
    "translation.set": "Set translation explicitly",
    "text.submit": "Submit manual text through the self pipeline",
    "microphone.test": "Start or stop microphone test",
    "audio.target.set": "Set peer loopback target",
    "audio.retry": "Retry lost peer process target",
    "settings.apply": "Apply focused typed settings changes",
    "provider.apply": "Apply self and peer STT changes together",
    "models.install": "Install or repair selected CPU/GPU ASR assets",
    "models.prepare": "Inspect CPU or GPU model readiness and integrity",
    "models.cancel": "Cancel an active CPU/GPU model installation",
    "gemma.prepare": "Prepare managed local Gemma model assets",
    "gemma.cancel": "Cancel managed local Gemma preparation",
    "models.retry": "Retry GPU activation after installation",
    "gpu.discover": "Discover available GPU devices",
    "secrets.set": "Store provider credential (value never returned)",
    "secrets.delete": "Delete provider credential",
    "secrets.verify": "Verify provider credential separately from storage",
    "auth.login": "Authorize a QQ, Discord, OpenRouter, or ChatGPT account with explicit browser consent",
    "auth.logout": "Remove locally stored account authorization (ChatGPT also requests remote revocation)",
    "overlay.set": "Set overlay enabled state",
    "overlay.lock": "Lock/unlock desktop caption positioning",
    "overlay.size": "Select desktop overlay size",
    "overlay.position.reset": "Reset desktop overlay position",
    "overlay.calibrate": "Begin, change, apply or cancel overlay calibration",
}
QUERIES = {
    "app.status",
    "settings.current",
    "settings.choices",
    "capture.channels",
    "audio.devices",
    "audio.processes",
    "audio.target",
    "providers.status",
    "gpu.status",
    "models.status",
    "secrets.presence",
    "auth.status",
    "overlay.status",
    "osc.status",
    "consent.peer_translation",
}
MODEL_BACKENDS = ("cpu", "gpu")
COMMAND_ARGUMENTS = {
    "app.stop": {},
    "capture.set": {
        "channel": ["self", "peer"],
        "enabled": "boolean",
        "accept_terms": "optional boolean (true only for peer enabled; defaults false)",
    },
    "translation.set": {"enabled": "boolean"},
    "text.submit": {"text": "non-empty string (protected content)"},
    "microphone.test": {"enabled": "boolean"},
    "audio.target.set": {"value": "available audio.target option value"},
    "audio.retry": {},
    "settings.apply": {
        "changes": "object of settings_fields to typed values (or direct typed fields)"
    },
    "provider.apply": {
        "channel": ["self", "peer", "both"],
        "provider": "settings.choices stt_providers",
        "self": "optional settings.choices stt_providers",
        "peer": "optional settings.choices stt_providers",
    },
    "models.install": {
        "backend": "cpu or gpu (default gpu)",
        "model_ids": "optional list of backend-compatible models.status IDs; empty means backend default",
    },
    "models.prepare": {
        "backend": "cpu or gpu (default cpu)",
        "verify_checksums": "optional boolean (default false)",
    },
    "models.retry": {"backend": "gpu (optional; default gpu)"},
    "gpu.discover": {},
    "secrets.set": {
        "name": "declared secrets.presence key",
        "value": "non-empty secret (stdin/hidden input only)",
    },
    "secrets.delete": {"name": "declared secrets.presence key"},
    "secrets.verify": {"name": "secret key", "value": "non-empty secret (stdin/hidden input only)"},
    "overlay.set": {"enabled": "boolean"},
    "overlay.lock": {"locked": "boolean"},
    "overlay.size": {"preset": "settings.choices overlay size"},
    "auth.login": {
        "provider": ["qq", "discord", "openrouter", "chatgpt"],
        "qq_identity": "required non-empty string for QQ",
        "credential": "required non-empty secret for QQ (stdin/hidden input only)",
        "referral_id": "optional referral string for QQ or Discord",
        "open_browser": "optional boolean, default false (OAuth only)",
    },
    "auth.logout": {"provider": ["qq", "discord", "openrouter", "chatgpt"]},
    "models.cancel": {"backend": "cpu or gpu (optional; default gpu)"},
    "gemma.prepare": {},
    "gemma.cancel": {},
    "overlay.position.reset": {},
    "overlay.calibrate": {
        "action": ["begin", "change", "apply", "cancel"],
        "field": "required for change: anchor, offset_x, offset_y, distance, text_scale, background_alpha",
        "value": "required for change; type/range depends on field",
    },
}

IMMEDIATE = {
    "locale": lambda v: LocaleSettingsIntent(v),
    "clipboard.enabled": lambda v: ClipboardSettingsIntent(v),
    "chatbox.source.enabled": lambda v: ChatboxSourceSettingsIntent(v),
    "chatbox.activation_notice.enabled": lambda v: ActivationNoticeSettingsIntent(v),
    "vrc_mic_intercept.enabled": lambda v: VrcMicInterceptSettingsIntent(v),
    "overlay.target": lambda v: OverlayTargetSettingsIntent(v),
    "overlay.translation.enabled": lambda v: OverlayTranslationSettingsIntent(v),
    "overlay.peer_original.enabled": lambda v: OverlayPeerOriginalSettingsIntent(v),
    "overlay.background_alpha": lambda v: DesktopOverlayBackgroundAlphaIntent(v),
    "overlay.swap_caption_languages": lambda v: DesktopOverlaySwapCaptionLanguagesIntent(v),
    "overlay.desktop_size": lambda v: DesktopOverlaySizeIntent(v),
    "self_vad.speech_threshold": lambda v: SelfVadSettingsIntent(v),
    "peer_vad.speech_threshold": lambda v: PeerVadSpeechThresholdIntent(v),
    "peer_vad.hangover_ms": lambda v: PeerVadHangoverIntent(v),
    "peer_vad.pre_roll_ms": lambda v: PeerVadPreRollIntent(v),
    "peer_expected_languages": lambda v: PeerExpectedLanguagesIntent(tuple(v)),
}
PROVIDER_EDITS = {
    "stt.gpu_device_id": SttGpuDeviceEdit,
    "translation.gpu_device_id": LlmGpuDeviceEdit,
    "translation.qwen.region": QwenRegionEdit,
    "translation.qwen.beijing.api_host": QwenBeijingApiHostEdit,
    "translation.qwen.singapore.api_host": QwenSingaporeApiHostEdit,
    "translation.http_extension_id": TranslationHttpExtensionEdit,
    "local_llm.base_url": LocalLlmBaseUrlEdit,
    "local_llm.model": LocalLlmModelEdit,
    "local_llm.extra_body_json": LocalLlmExtraBodyEdit,
    "stt.custom.endpoint": CustomSttEndpointEdit,
    "stt.custom.model": CustomSttModelEdit,
    "stt.custom.extra_json": CustomSttExtraEdit,
    "stt.soniox.enable_speaker_diarization": SonioxSpeakerDiarizationEdit,
    "prompts.system_prompt_override": SystemPromptEdit,
}
SETTINGS_FIELDS = (
    frozenset(IMMEDIATE)
    | frozenset(PROVIDER_EDITS)
    | frozenset(
        {
            "stt.provider",
            "peer_stt.provider",
            "stt.cloud_free_tier_providers",
            "audio.input_host_api",
            "audio.input_device",
            "audio.output_device",
            "translation.model",
            "translation.connection",
            "translation.connection_history",
            "translation.previous_llm_model",
            "managed.referral_id",
            "prompts.value",
            "osc.connection",
            "custom_vocabulary",
            "languages",
            "telemetry.enabled",
        }
    )
)
BOOLEAN_FIELDS = frozenset(
    {
        "clipboard.enabled",
        "chatbox.source.enabled",
        "chatbox.activation_notice.enabled",
        "vrc_mic_intercept.enabled",
        "overlay.translation.enabled",
        "overlay.peer_original.enabled",
        "overlay.swap_caption_languages",
        "stt.soniox.enable_speaker_diarization",
        "telemetry.enabled",
    }
)
NUMBER_FIELDS = frozenset(
    {
        "overlay.background_alpha",
        "self_vad.speech_threshold",
        "peer_vad.speech_threshold",
    }
)
INTEGER_FIELDS = frozenset({"peer_vad.hangover_ms", "peer_vad.pre_roll_ms"})
OBJECT_FIELDS = frozenset(
    {
        "languages",
        "osc.connection",
        "custom_vocabulary",
        "translation.connection_history",
    }
)
LIST_FIELDS = frozenset({"peer_expected_languages", "stt.cloud_free_tier_providers"})
OPTIONAL_STRING_FIELDS = frozenset({"managed.referral_id", "translation.previous_llm_model"})

FREE_FORM_SETTINGS_FIELDS = frozenset(
    {
        "prompts.value",
        "local_llm.base_url",
        "local_llm.model",
        "local_llm.extra_body_json",
        "stt.custom.endpoint",
        "stt.custom.model",
        "stt.custom.extra_json",
    }
)


def _extension_ids(application: UiApplicationBoundary) -> tuple[str, ...]:
    registry = application.http_extension_registry()
    if registry is None:
        return ()
    return tuple(sorted(loaded.definition.id for loaded in registry.snapshot.extensions))


_VERIFIABLE_SECRET_PROVIDERS = {
    "google_api_key": "google",
    "openrouter_api_key": "openrouter",
    "openai_api_key": "openai",
    "deepseek_api_key": "deepseek",
    "deepgram_api_key": "deepgram",
    "gemini_transcribe_api_key": "gemini_transcribe",
    "elevenlabs_scribe_api_key": "elevenlabs_scribe",
    "soniox_api_key": "soniox",
    "alibaba_api_key_beijing": "alibaba_beijing",
    "alibaba_api_key_singapore": "alibaba_singapore",
}


def _secret_keys(application: UiApplicationBoundary) -> tuple[str, ...]:
    from puripuly_heart.app.ports.settings_secrets import SettingsSecretKey
    from puripuly_heart.core.http_extensions import http_extension_secret_key

    keys = [item.value for item in SettingsSecretKey]
    registry = application.http_extension_registry()
    if registry is not None:
        for loaded in registry.snapshot.extensions:
            keys.extend(
                http_extension_secret_key(loaded.definition.id, secret.id)
                for secret in loaded.definition.secrets
            )
    return tuple(keys)


def _known_secret(application: UiApplicationBoundary, name: str) -> str:
    if name not in _secret_keys(application):
        raise ValueError("unknown or undeclared secret")
    return name


def _json(value: object) -> object:
    from dataclasses import asdict, is_dataclass
    from enum import Enum

    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _json(asdict(value))
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json(v) for v in value]
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    return str(value)


def _redact(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: (
                "<redacted>"
                if key.lower() == "custom_terms"
                or any(
                    s in key.lower()
                    for s in (
                        "secret",
                        "api_key",
                        "token",
                        "password",
                        "credential",
                        "prompt",
                        "vocabulary",
                        "referral",
                    )
                )
                else _redact(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


@dataclass(slots=True)
class _Operation:
    receipt: dict
    request_signature: str = ""
    task: asyncio.Task | None = None
    backend: str | None = None
    install_started: bool = False
    interrupt_requested: bool = False


class UnknownOperationError(ValueError):
    pass


@dataclass(slots=True)
class ApplicationControlOwner:
    application: UiApplicationBoundary
    settings: object
    pipeline: object
    results: object
    events: ControlEvents
    provisioning: Callable[[], object | None]
    gpu: Callable[[], object | None]
    peer: Callable[[], object | None]
    calibration: Callable[[], object | None]
    gemma: Callable[[], object | None]
    sync_ui: Callable[[], None]
    locale_choices: tuple[str, ...]
    localize: Callable[[str], str]
    gemma_selection: Callable[[AppSettingsVNext], ManagedGemmaTranslationSelection]
    instance_id: str | None = None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _resource_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _stop_requested: asyncio.Event = field(default_factory=asyncio.Event)
    _operations: OrderedDict[str, _Operation] = field(default_factory=OrderedDict)
    _requests: OrderedDict[str, str] = field(default_factory=OrderedDict)
    _revision: int = 0
    _fingerprint: str | None = None
    _open: bool = True
    _lock_owner: asyncio.Task | None = None
    _auth_challenge: dict | None = None

    def lock_owned_by_current_task(self) -> bool:
        return self._lock_owner is asyncio.current_task()

    def bind_instance(self, instance_id: str) -> None:
        if not instance_id or (self.instance_id is not None and self.instance_id != instance_id):
            raise ValueError("instance identity cannot change within a host")
        self.instance_id = instance_id
        self.settings.observe_commits(self._publish_committed)

    async def wait_for_stop_request(self) -> None:
        await self._stop_requested.wait()

    def freeze_ingress(self) -> None:
        if not self._open:
            return
        self._open = False
        for operation_id, operation in self._operations.items():
            if operation.task is not None and not operation.task.done():
                operation.interrupt_requested = True
                if operation.receipt["status"] == "accepted":
                    operation.receipt.update(status="interrupted", terminal=True)
                    self.events.publish(
                        {
                            "topic": "operation",
                            "operation_id": operation_id,
                            "status": "interrupted",
                            "terminal": True,
                            "revision": self._revision,
                        }
                    )
                operation.task.cancel()

    async def drain_operations(self) -> None:
        tasks = tuple(
            operation.task for operation in self._operations.values() if operation.task is not None
        )
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _publish_committed(self, canonical: object) -> None:
        fingerprint = hashlib.sha256(
            json.dumps(canonical_snapshot_values(canonical), sort_keys=True, default=str).encode()
        ).hexdigest()
        if fingerprint == self._fingerprint:
            return
        initial = self._fingerprint is None
        self._fingerprint = fingerprint
        self._revision += 1
        if not initial:
            self.events.publish({"topic": "settings", "revision": self._revision})

    def _current_revision(self) -> int:
        return self._revision

    async def apply_mutation(
        self, mutator: Callable[[object], object]
    ) -> tuple[object, object, object]:
        """OSC focused edit: obtain canonical under the shared mutation lock."""
        import copy

        async with self._resource_lock:
            async with self._lock:
                current = self.application.compatibility_settings()
                if current is None:
                    raise RuntimeError("settings not loaded")
                previous = copy.deepcopy(current)
                updated = mutator(copy.deepcopy(current))
                result = await self.application._settings.apply_settings(updated)
                self._current_revision()
                return result, previous, updated

    def capabilities(self) -> dict:
        return {
            "version": 1,
            "instance_id": self.instance_id,
            "commands": {
                key: {
                    "description": value,
                    "arguments": COMMAND_ARGUMENTS[key],
                    "cancellable": key in {"models.install", "gemma.prepare"},
                }
                for key, value in COMMANDS.items()
            },
            "queries": sorted(QUERIES),
            "settings_fields": sorted(SETTINGS_FIELDS),
            "secret_keys": list(_secret_keys(self.application)),
            "topics": [
                "operation",
                "settings",
                "auth",
                "capture",
                "transcript",
                "translation",
                "session_state_changed",
                "error",
                "logs",
                "gap",
            ],
        }

    def _channels(self) -> dict:
        canonical = self.application.compatibility_settings()
        runtime = getattr(self.pipeline, "local_asr_runtime", None)
        snapshot = runtime.snapshot if runtime is not None else None
        runtime_channels = (
            {item.channel: item for item in snapshot.channels} if snapshot is not None else {}
        )
        components = getattr(self.pipeline, "current", None)
        peer_capture = components.peer_capture if components is not None else None
        captures = {"self": getattr(self.pipeline, "self_capture", None), "peer": peer_capture}
        channels = {}
        for channel, capture in captures.items():
            state = capture.snapshot if capture is not None else None
            provider = runtime_channels.get(channel)
            selected = (
                (
                    canonical.intent.stt.provider
                    if channel == "self"
                    else canonical.intent.peer_stt.provider
                )
                if canonical is not None
                else None
            )
            channels[channel] = {
                "selected_provider": selected,
                "runtime_provider": provider.provider_id if provider is not None else None,
                "capture_attached_provider": (
                    state.provider_id if state is not None and state.has_source else None
                ),
                "desired_active": state.desired_active if state is not None else False,
                "effective_active": state.effective_active if state is not None else False,
                "pending_handoff": provider.pending_handoff if provider is not None else False,
                "failure_reason": _json(state.failure_reason) if state is not None else None,
                "provider_status": _json(state.provider_status) if state is not None else None,
                "capture": _json(state) if state is not None else None,
                "provider": _json(provider) if provider is not None else None,
            }
        return channels

    async def _settings_choice_catalog(self, base: dict, canonical: object | None) -> dict:
        from puripuly_heart.app.ports.osc_control import OSC_CONNECTION_MODES
        from puripuly_heart.app.services.application_audio_devices import (
            enumerate_audio_devices,
        )
        from puripuly_heart.config.audio_host_api import (
            WINDOWS_WASAPI_COMPATIBILITY_HOST_API,
            WINDOWS_WASAPI_HOST_API,
            normalize_input_host_api,
        )
        from puripuly_heart.config.desktop_overlay_values import (
            DESKTOP_FLET_SIZE_PRESET_ORDER,
        )
        from puripuly_heart.config.overlay_calibration import (
            OVERLAY_CALIBRATION_ANCHORS,
        )
        from puripuly_heart.config.provider_values import (
            CLOUD_FREE_TIER_STT_PROVIDERS,
            MAX_CUSTOM_VOCAB_TERMS,
            DeepSeekLLMModel,
            GeminiLLMModel,
            LLMProviderName,
            LocalLLMBackend,
            OpenRouterCredentialSource,
            OpenRouterLLMModel,
            OpenRouterSelectionAlias,
            QwenLLMModel,
        )
        from puripuly_heart.config.resolved import OVERLAY_TARGETS
        from puripuly_heart.config.runtime_resolution import (
            DEEPGRAM_STT_MODEL_NOVA_3,
            ELEVENLABS_SCRIBE_STT_MODEL,
            GEMINI_TRANSCRIBE_STT_MODEL,
            QWEN_AUDIO_STT_MODEL,
            SONIOX_STT_MODEL_RT_V5,
        )
        from puripuly_heart.core.language import get_all_language_options

        settings = getattr(canonical, "intent", None)
        language_codes = [code for code, _name in get_all_language_options()]
        audio = await asyncio.to_thread(enumerate_audio_devices, canonical)
        host_api_values = ["", *audio["host_apis"]]
        if WINDOWS_WASAPI_HOST_API in audio["host_apis"]:
            host_api_values.append(WINDOWS_WASAPI_COMPATIBILITY_HOST_API)
        selected_host_api = getattr(getattr(settings, "audio", None), "input_host_api", "")
        if selected_host_api and selected_host_api not in host_api_values:
            host_api_values.append(selected_host_api)
        input_hosts = [
            {
                "value": value,
                "available": value == ""
                or value in audio["host_apis"]
                or (
                    value == WINDOWS_WASAPI_COMPATIBILITY_HOST_API
                    and WINDOWS_WASAPI_HOST_API in audio["host_apis"]
                ),
            }
            for value in host_api_values
        ]
        selected_input_device = getattr(getattr(settings, "audio", None), "input_device", "")
        selected_profile = normalize_input_host_api(selected_host_api)
        input_devices = [
            {
                "value": item["name"],
                "host_api": item["host_api"],
                "available": (
                    not selected_profile.actual_host_api
                    or item["host_api"] == selected_profile.actual_host_api
                ),
            }
            for item in audio["microphones"]
        ]
        if selected_input_device and not any(
            item["value"] == selected_input_device for item in input_devices
        ):
            input_devices.append(
                {
                    "value": selected_input_device,
                    "host_api": selected_host_api,
                    "available": False,
                }
            )
        selected_output_device = getattr(
            getattr(settings, "desktop_audio", None), "output_device", ""
        )
        output_devices = [
            {"value": "", "available": True},
            *({"value": name, "available": True} for name in audio["loopback_outputs"]),
        ]
        if selected_output_device and not any(
            item["value"] == selected_output_device for item in output_devices
        ):
            output_devices.append(
                {
                    "value": selected_output_device,
                    "available": False,
                }
            )
        target_options = self.application.list_loopback_capture_options()
        capture_targets = [
            {
                "value": str(getattr(option, "value", "")),
                "available": not bool(getattr(option, "disabled", False)),
            }
            for option in target_options
        ]
        selected_capture_target = self.application.current_loopback_capture_option_value()
        if selected_capture_target and not any(
            item["value"] == selected_capture_target for item in capture_targets
        ):
            capture_targets.append(
                {
                    "value": str(selected_capture_target),
                    "available": False,
                }
            )
        gpu_owner = self.gpu()
        gpu_snapshot = getattr(gpu_owner, "snapshot", None)
        gpu_devices = [
            {"value": str(device.device_id), "available": True}
            for device in getattr(gpu_snapshot, "devices", ())
        ]
        gpu_device_ids = {item["value"] for item in gpu_devices}
        selected_gpu_ids = {
            getattr(getattr(settings, "stt", None), "gpu_device_id", "auto"),
            getattr(getattr(settings, "translation", None), "gpu_device_id", "auto"),
        }
        gpu_choices = [{"value": "auto", "available": True}, *gpu_devices]
        for selected in sorted(selected_gpu_ids - {"auto"} - gpu_device_ids):
            gpu_choices.append({"value": selected, "available": False})
        provisioning_owner = self.provisioning()
        model_snapshot = getattr(provisioning_owner, "snapshot", None)
        local_models = [
            {
                "id": model.model_id,
                "backend": model.backend,
                "status": model.status,
                "available": model.available,
            }
            for model in getattr(model_snapshot, "models", ())
        ]
        registry = self.application.http_extension_registry()
        extension_ids = _extension_ids(self.application)
        extension_errors = len(registry.snapshot.errors) if registry is not None else 0
        finite_choices = {
            "locale": list(self.locale_choices),
            "languages": {
                "codes": language_codes,
                "secondary_target_unset": "",
                "peer_source_modes": ["manual", "auto"],
            },
            "peer_expected_languages": language_codes,
            "stt.provider": [item.value for item in STTProviderName],
            "peer_stt.provider": [item.value for item in STTProviderName],
            "stt.cloud_free_tier_providers": [item.value for item in CLOUD_FREE_TIER_STT_PROVIDERS],
            "translation.model": [item.value for item in TranslationModel],
            "translation.connection": {
                model.value: [
                    connection.value for connection in supported_translation_connections(model)
                ]
                for model in TranslationModel
            },
            "translation.previous_llm_model": [item.value for item in TranslationModel],
            "translation.connection_history": {
                model.value: [
                    connection.value for connection in supported_translation_connections(model)
                ]
                for model in TranslationModel
            },
            "translation.qwen.region": [item.value for item in QwenRegion],
            "translation.http_extension_id": ["", *extension_ids],
            "stt.gpu_device_id": gpu_choices,
            "translation.gpu_device_id": gpu_choices,
            "audio.input_host_api": input_hosts,
            "audio.input_device": input_devices,
            "audio.output_device": output_devices,
            "overlay.target": list(OVERLAY_TARGETS),
            "overlay.desktop_size": list(DESKTOP_FLET_SIZE_PRESET_ORDER),
            "osc.connection.mode": list(OSC_CONNECTION_MODES),
        }
        field_types = {
            name: (
                "boolean"
                if name in BOOLEAN_FIELDS
                else (
                    "number"
                    if name in NUMBER_FIELDS
                    else (
                        "integer"
                        if name in INTEGER_FIELDS
                        else (
                            "object"
                            if name in OBJECT_FIELDS
                            else (
                                "array<string>"
                                if name in LIST_FIELDS
                                else (
                                    "optional-string"
                                    if name in OPTIONAL_STRING_FIELDS
                                    else "string"
                                )
                            )
                        )
                    )
                )
            )
            for name in sorted(SETTINGS_FIELDS)
        }
        field_schemas = {
            name: {
                "type": field_types[name],
                "free_form": name in FREE_FORM_SETTINGS_FIELDS,
                **({"choices": f"/choices/{name}"} if name in finite_choices else {}),
                **({"nullable": True} if name in OPTIONAL_STRING_FIELDS else {}),
            }
            for name in sorted(SETTINGS_FIELDS)
        }
        field_schemas.update(
            {
                "peer_expected_languages": {
                    "type": "array<string>",
                    "items": "/choices/peer_expected_languages",
                    "empty_allowed": True,
                },
                "stt.cloud_free_tier_providers": {
                    "type": "array<string>",
                    "items": "/choices/stt.cloud_free_tier_providers",
                    "min_items": 1,
                    "unique": True,
                },
                "languages": {
                    "type": "object",
                    "required": [],
                    "properties": {
                        "source": {"type": "string", "choices": "/choices/languages/codes"},
                        "target": {"type": "string", "choices": "/choices/languages/codes"},
                        "secondary_target": {
                            "type": "string",
                            "choices": "/choices/languages/codes",
                            "empty_allowed": True,
                        },
                        "peer_source": {
                            "type": "string",
                            "choices": "/choices/languages/codes",
                            "empty_allowed": True,
                        },
                        "peer_target": {
                            "type": "string",
                            "choices": "/choices/languages/codes",
                            "empty_allowed": True,
                        },
                        "peer_source_mode": {
                            "type": "string",
                            "choices": "/choices/languages/peer_source_modes",
                        },
                    },
                },
                "custom_vocabulary": {
                    "type": "object",
                    "required": ["source_language", "terms"],
                    "properties": {
                        "source_language": {
                            "type": "string",
                            "choices": "/choices/languages/codes",
                        },
                        "terms": {
                            "type": "array<string>",
                            "free_form": True,
                            "max_items": MAX_CUSTOM_VOCAB_TERMS,
                        },
                    },
                },
                "translation.connection_history": {
                    "type": "object",
                    "key_choices": "/choices/translation.model",
                    "value_choices_by_key": "/choices/translation.connection",
                },
                "osc.connection": {
                    "type": "object",
                    "required": ["mode", "receive_port"],
                    "properties": {
                        "mode": {"type": "string", "choices": "/choices/osc.connection.mode"},
                        "send_port": {"type": "integer", "minimum": 1, "maximum": 65535},
                        "receive_port": {"type": "integer", "minimum": 1, "maximum": 65535},
                    },
                },
                "overlay.background_alpha": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                },
                "self_vad.speech_threshold": {
                    "type": "number",
                    "minimum": 0.1,
                    "maximum": 1,
                },
                "peer_vad.speech_threshold": {
                    "type": "number",
                    "minimum": 0.1,
                    "maximum": 1,
                },
                "peer_vad.hangover_ms": {
                    "type": "integer",
                    "minimum": 0,
                },
                "peer_vad.pre_roll_ms": {
                    "type": "integer",
                    "minimum": 0,
                },
                "overlay.calibration": {
                    "type": "object",
                    "required": [
                        "anchor",
                        "offset_x",
                        "offset_y",
                        "distance",
                        "text_scale",
                        "background_alpha",
                    ],
                    "properties": {
                        "anchor": {
                            "type": "string",
                            "choices": "/overlay/calibration_fields/anchor",
                        },
                        "offset_x": {"type": "number"},
                        "offset_y": {"type": "number"},
                        "distance": {"type": "number", "exclusiveMinimum": 0},
                        "text_scale": {"type": "number", "exclusiveMinimum": 0},
                        "background_alpha": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                    },
                },
            }
        )
        return {
            **base,
            "fields": sorted(SETTINGS_FIELDS),
            "field_types": field_types,
            "field_schemas": field_schemas,
            "command_choices": {
                "capture.set.channel": ["self", "peer"],
                "provider.apply.channel": ["self", "peer", "both"],
                "models.backend": list(MODEL_BACKENDS),
                "auth.provider": COMMAND_ARGUMENTS["auth.login"]["provider"],
                "overlay.calibration.action": COMMAND_ARGUMENTS["overlay.calibrate"]["action"],
                "overlay.calibration.fields": {
                    "anchor": list(OVERLAY_CALIBRATION_ANCHORS),
                    "numeric": [
                        "offset_x",
                        "offset_y",
                        "distance",
                        "text_scale",
                        "background_alpha",
                    ],
                },
                "secrets.names": _secret_keys(self.application),
                "audio.target.options": _json(capture_targets),
                "models.install.ids": local_models,
            },
            "free_form_fields": sorted(FREE_FORM_SETTINGS_FIELDS),
            "choices": finite_choices,
            "providers": {
                "stt": [item.value for item in STTProviderName],
                "llm": [item.value for item in LLMProviderName],
                "translation_models": finite_choices["translation.connection"],
                "qwen_llm_models": [item.value for item in QwenLLMModel],
                "openrouter_models": [item.value for item in OpenRouterLLMModel],
                "openrouter_selection_aliases": [item.value for item in OpenRouterSelectionAlias],
                "openrouter_credential_sources": [
                    item.value for item in OpenRouterCredentialSource
                ],
                "gemini_llm_models": [item.value for item in GeminiLLMModel],
                "deepseek_llm_models": [item.value for item in DeepSeekLLMModel],
                "local_llm_backends": [item.value for item in LocalLLMBackend],
            },
            "audio": {
                **audio,
                "input_host_apis": input_hosts,
                "input_devices": input_devices,
                "output_devices": output_devices,
                "capture_targets": capture_targets,
                "selected_capture_target": selected_capture_target,
            },
            "gpu": {
                "backends": list(MODEL_BACKENDS),
                "devices": gpu_choices,
                "discovery": {
                    "attempted": getattr(gpu_snapshot, "discovery_attempted", False),
                    "failed": getattr(gpu_snapshot, "discovery_failed", False),
                    "state": getattr(gpu_snapshot, "ui_state", None),
                },
            },
            "models": {
                "local_asr": local_models,
                "local_asr_owner_available": provisioning_owner is not None,
                "stt": {
                    "deepgram": [DEEPGRAM_STT_MODEL_NOVA_3],
                    "gemini_transcribe": [GEMINI_TRANSCRIBE_STT_MODEL],
                    "elevenlabs_scribe": [ELEVENLABS_SCRIBE_STT_MODEL],
                    "qwen_audio": [QWEN_AUDIO_STT_MODEL],
                    "soniox": [SONIOX_STT_MODEL_RT_V5],
                },
            },
            "extensions": {
                "translation_http_ids": list(extension_ids),
                "load_error_count": extension_errors,
            },
            "overlay": {
                "targets": list(OVERLAY_TARGETS),
                "desktop_sizes": list(DESKTOP_FLET_SIZE_PRESET_ORDER),
                "calibration_fields": {
                    "anchor": list(OVERLAY_CALIBRATION_ANCHORS),
                    "offset_x": {"type": "number"},
                    "offset_y": {"type": "number"},
                    "distance": {"type": "number", "exclusiveMinimum": 0},
                    "text_scale": {"type": "number", "exclusiveMinimum": 0},
                    "background_alpha": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                },
            },
            "osc_modes": list(OSC_CONNECTION_MODES),
            "hardware_status": {
                "audio_errors": audio["errors"],
                "gpu_discovery_attempted": getattr(gpu_snapshot, "discovery_attempted", False),
                "gpu_discovery_failed": getattr(gpu_snapshot, "discovery_failed", False),
                "extension_load_error_count": extension_errors,
            },
        }

    async def query(self, name: str, arguments: dict) -> dict:
        if name not in QUERIES:
            raise ValueError("unknown application query")
        if not isinstance(arguments, dict):
            raise ValueError("query arguments must be an object")
        if arguments:
            raise ValueError("query does not accept arguments")
        revision = self._current_revision()
        canonical = self.settings.committed_settings
        base = {"instance_id": self.instance_id, "revision": revision}
        if name == "settings.current":
            return {
                **base,
                "settings": (
                    _redact(canonical_snapshot_values(canonical)) if canonical is not None else None
                ),
            }
        if name == "settings.choices":
            return await self._settings_choice_catalog(base, canonical)
        if name in {"capture.channels", "providers.status"}:
            return {**base, "channels": self._channels()}
        if name == "app.status":
            return {
                **base,
                "host_pid": os.getpid(),
                "channels": self._channels(),
                "microphone_test": self.application.microphone_test_snapshot(),
                "translation_enabled": self.application.state().translation_enabled,
                "translation_runtime_ready": self.application.state().translation_runtime_ready,
                "overlay_target": self.application.state().overlay_target,
                "stopping": self._stop_requested.is_set(),
            }
        if name in {"audio.target", "audio.processes"}:
            options = (
                self.application.list_loopback_process_options()
                if name == "audio.processes"
                else self.application.list_loopback_capture_options()
            )
            return {
                **base,
                "options": _json(options),
                "selected": _json(self.application.current_loopback_capture_option_value()),
                "effective": _json(self.application.loopback_capture_summary()),
            }
        if name == "audio.devices":
            from puripuly_heart.app.services.application_audio_devices import (
                enumerate_audio_devices,
            )

            return {**base, **await asyncio.to_thread(enumerate_audio_devices, canonical)}
        if name == "gpu.status":
            owner = self.gpu()
            return {
                **base,
                "state": _json(owner.snapshot) if owner is not None else None,
                "runtime": (
                    _json(self.pipeline.local_asr_runtime.snapshot.gpu)
                    if self.pipeline.local_asr_runtime is not None
                    else None
                ),
            }
        if name == "models.status":
            owner = self.provisioning()
            gemma = self.gemma()
            return {
                **base,
                "state": _json(getattr(owner, "snapshot", None)) if owner is not None else None,
                "gemma": _json(gemma.snapshot) if gemma is not None else None,
            }
        if name == "overlay.status":
            output = self.application.output_status()
            return {
                **base,
                "target": self.application.state().overlay_target,
                "locked": self.application.state().desktop_overlay_captions_locked,
                "settings": _json(self.application.settings_overlay_snapshot()),
                "output": _json(output["overlay"]),
            }
        if name == "osc.status":
            output = self.application.output_status()
            return {
                **base,
                "ports": self.application.effective_osc_ports(),
                "output": _json(output["osc"]),
            }
        if name == "auth.status":
            return {
                **base,
                "action": self.application.dashboard_managed_auth_action(),
                "prompt_kind": self.application.dashboard_managed_auth_prompt_kind(),
                "failure_kind": self.application.managed_auth_last_failure_kind(),
                "consent_accepted": self.application.state().peer_translation_eula_accepted,
                "challenge": (
                    dict(self._auth_challenge) if self._auth_challenge is not None else None
                ),
                "chatgpt": _chatgpt_status(self.application),
            }
        if name == "consent.peer_translation":
            return {
                **base,
                "accepted": self.application.state().peer_translation_eula_accepted,
                "terms": self.localize("peer_translation_eula.body"),
            }
        if name == "secrets.presence":
            keys = _secret_keys(self.application)
            result = self.application.settings_secrets().load_values(keys)
            if result.snapshot is None or result.read_error is not None:
                return {
                    **base,
                    "status": "failed",
                    "error": (
                        type(result.read_error).__name__
                        if result.read_error is not None
                        else "secret_store_unavailable"
                    ),
                }
            return {
                **base,
                "status": "applied",
                "presence": {key: bool(result.snapshot.get(key)) for key in keys},
            }
        raise AssertionError(name)

    async def submit(
        self,
        command: str,
        arguments: dict,
        *,
        request_id: str,
        expected_revision: int | None = None,
    ) -> dict:
        if self.instance_id is None:
            raise RuntimeError("host must bind instance before accepting control")
        if not self._open:
            raise RuntimeError("application ingress is closing")
        if (
            command not in COMMANDS
            or not isinstance(arguments, dict)
            or not request_id
            or len(request_id) > 128
        ):
            raise ValueError("invalid application command or request identity")
        signature = hashlib.sha256(
            json.dumps(
                [command, arguments, expected_revision], sort_keys=True, ensure_ascii=False
            ).encode("utf-8")
        ).hexdigest()
        previous = self._requests.get(request_id)
        if previous is not None:
            prior = self._operations[previous]
            if prior.request_signature != signature:
                raise ValueError("request identity reused with different mutation")
            return dict(prior.receipt)
        if len(self._operations) >= 256 and not self._evict_terminal():
            raise RuntimeError("application operation capacity reached")
        operation_id = uuid4().hex
        receipt = {
            "instance_id": self.instance_id,
            "operation_id": operation_id,
            "request_id": request_id,
            "status": "accepted",
            "terminal": False,
            "revision": self._current_revision(),
        }
        self._operations[operation_id] = _Operation(
            receipt,
            backend=arguments.get("backend", "gpu") if command == "models.install" else None,
            request_signature=signature,
        )
        self._requests[request_id] = operation_id
        task = asyncio.create_task(
            self._execute(command, arguments, expected_revision, operation_id),
            name=f"control:{command}:{operation_id}",
        )
        self._operations[operation_id].task = task
        return dict(receipt)

    def _evict_terminal(self) -> bool:
        for key, operation in self._operations.items():
            if operation.task is not None and operation.task.done():
                self._operations.pop(key)
                self._requests.pop(operation.receipt["request_id"], None)
                return True
        return False

    def _trim(self) -> None:
        while len(self._operations) > 256 and self._evict_terminal():
            pass

    async def _execute(
        self, command: str, arguments: dict, expected_revision: int | None, operation_id: str
    ) -> None:
        operation = self._operations[operation_id]
        captured = None
        completed = False
        acquired = False
        try:
            operation.receipt["status"] = "running"
            changes = arguments.get("changes", arguments) if command == "settings.apply" else None
            safe_capture_off = command == "capture.set" and arguments.get("enabled") is False
            safe_locale = (
                command == "settings.apply"
                and isinstance(changes, dict)
                and len(changes) == 1
                and "locale" in changes
            )
            independent_stop = command in {"app.stop", "models.cancel", "gemma.cancel"} or (
                command == "microphone.test" and arguments.get("enabled") is False
            )
            resource_lock = (
                None
                if independent_stop
                or safe_capture_off
                or safe_locale
                or command == "secrets.verify"
                else self._resource_lock
            )
            if resource_lock is not None:
                await resource_lock.acquire()
                acquired = True
            with self.results.capture(revision=self._current_revision) as captured:
                deferred = None
                async with asyncio.Lock() if independent_stop else self._lock:
                    if not self._open and command != "app.stop":
                        raise asyncio.CancelledError
                    if not independent_stop:
                        self._lock_owner = asyncio.current_task()
                    revision = self._current_revision()
                    if expected_revision is not None and expected_revision != revision:
                        operation.receipt.update(
                            status="rejected",
                            error={"code": "revision_conflict", "current_revision": revision},
                        )
                    elif command == "models.install":
                        _validate_command_args(
                            command, arguments, locale_choices=self.locale_choices
                        )
                        if self.provisioning() is None:
                            result = {
                                "status": "failed",
                                "error": {"code": "model_owner_unavailable"},
                            }
                        else:
                            deferred = (
                                "install",
                                self._start_model_install(arguments, operation_id),
                            )
                    elif command == "text.submit":
                        _validate_command_args(
                            command, arguments, locale_choices=self.locale_choices
                        )
                        if not isinstance(arguments["text"], str) or not arguments["text"].strip():
                            raise ValueError("text must be non-empty")
                        deferred = ("text", arguments["text"])
                    else:
                        result = await self._dispatch(command, arguments, operation_id=operation_id)
                    if self._lock_owner is asyncio.current_task():
                        self._lock_owner = None
                if operation.receipt["status"] != "rejected":
                    if deferred is not None:
                        if not self._open:
                            raise asyncio.CancelledError
                        if deferred[0] == "install":
                            result = await self._install_result(deferred[1])
                        else:
                            await self.application.submit_text(deferred[1])
                            result = True
                    transaction = captured.current
                    status = "applied"
                    if transaction is not None and command in {
                        "settings.apply",
                        "provider.apply",
                        "secrets.set",
                        "secrets.delete",
                        "overlay.lock",
                        "overlay.size",
                        "overlay.position.reset",
                    }:
                        status = _transaction_status(transaction)
                        operation.receipt["transaction"] = _transaction_data(transaction)
                    elif result is False:
                        status = "failed"
                    elif isinstance(result, dict) and "status" in result:
                        status = result["status"]
                    if status == "applied" and command in {"provider.apply", "settings.apply"}:
                        mismatch = {
                            channel: state
                            for channel, state in self._channels().items()
                            if state["selected_provider"] != state["runtime_provider"]
                            and state["failure_reason"] is not None
                            and not state["pending_handoff"]
                        }
                        if mismatch:
                            status = "degraded"
                            operation.receipt["runtime_observation"] = {
                                "status": "degraded",
                                "reason": "selected_provider_not_effective",
                                "channels": mismatch,
                            }
                    operation.receipt.update(status=status, revision=self._current_revision())
                    if command in {"settings.apply", "provider.apply"} and status in {
                        "applied",
                        "degraded",
                    }:
                        try:
                            self.sync_ui()
                        except Exception:
                            self.events.publish(
                                {
                                    "topic": "error",
                                    "operation_id": operation_id,
                                    "code": "presentation_sync_failed",
                                }
                            )
                    if isinstance(result, dict):
                        operation.receipt.update({k: v for k, v in result.items() if k != "status"})
                        if "status" in result and result["status"] != "applied":
                            operation.receipt["status"] = result["status"]
                    completed = True
        except asyncio.CancelledError:
            operation.receipt["status"] = "interrupted" if not self._open else "cancelled"
        except PeerCaptureTargetUnavailable:
            operation.receipt.update(
                status="action_required", error={"code": "capture_target_unavailable"}
            )
        except (ValueError, TypeError, KeyError) as exc:
            operation.receipt.update(
                status="rejected", error={"code": "invalid_arguments", "type": type(exc).__name__}
            )
        except Exception as exc:
            operation.receipt.update(
                status="failed",
                error={"code": "application_operation_failed", "type": type(exc).__name__},
            )
        finally:
            if (
                operation.interrupt_requested
                and command != "app.stop"
                and operation.receipt["status"] in {"applied", "degraded"}
            ):
                operation.receipt["status"] = "interrupted"
            elif (
                not self._open
                and command != "app.stop"
                and not completed
                and operation.receipt["status"]
                not in {"interrupted", "cancelled", "rejected", "persistence_failed"}
            ):
                operation.receipt["status"] = "interrupted"
            if (
                operation.receipt["status"] in {"interrupted", "cancelled"}
                and captured is not None
                and captured.committed_revision is not None
            ):
                if "transaction" not in operation.receipt:
                    transaction = captured.current
                    if transaction is None or transaction.status not in {
                        TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
                        TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED,
                    }:
                        transaction = TransactionResult(
                            TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_INTERRUPTED,
                            None,
                            None,
                        )
                    operation.receipt["transaction"] = _transaction_data(transaction)
                operation.receipt["revision"] = captured.committed_revision
            operation.receipt["terminal"] = operation.receipt["status"] in TERMINAL
            if self._lock_owner is asyncio.current_task():
                self._lock_owner = None
            if (
                self._auth_challenge is not None
                and self._auth_challenge.get("operation_id") == operation_id
            ):
                self._auth_challenge = None
                self.events.publish(
                    {"topic": "auth", "operation_id": operation_id, "phase": "finished"}
                )
            self.events.publish(
                {
                    "topic": "operation",
                    "operation_id": operation_id,
                    "status": operation.receipt["status"],
                    "terminal": operation.receipt["terminal"],
                    "revision": self._current_revision(),
                }
            )
            self._trim()
            if acquired:
                resource_lock.release()

    def _start_model_install(self, args: dict, operation_id: str) -> asyncio.Task:
        from puripuly_heart.core.local_asr_provisioning import LocalASRInstallRequest

        owner = self.provisioning()
        if owner is None:
            raise RuntimeError("model owner is unavailable")
        backend = args.get("backend", "gpu")
        snapshot = owner.snapshot
        requested_model_ids = args.get("model_ids")
        if requested_model_ids:
            model_ids = tuple(requested_model_ids)
        elif backend == "cpu":
            model_ids = snapshot.required_cpu_model_ids
        else:
            model_ids = (snapshot.gpu_model_id,)
        model_backends = {item.model_id: item.backend for item in snapshot.models}
        if not model_ids or any(model_backends.get(model_id) != backend for model_id in model_ids):
            raise ValueError("unsupported model ids")
        settings = self.application.compatibility_settings()
        request = LocalASRInstallRequest(
            backend=backend,
            model_ids=model_ids,
            locale=settings.intent.ui.locale if settings is not None else None,
            origin="application_control",
            explicit_gpu_intent=backend == "gpu",
        )
        install_task = owner.start_install(request)
        self._operations[operation_id].install_started = True
        return install_task

    @staticmethod
    async def _install_result(install_task: asyncio.Task) -> dict:
        result = await install_task
        return {
            "status": (
                "cancelled"
                if result.cancelled
                else "degraded" if result.failed_model_ids else "applied"
            ),
            "models": _json(result.snapshot),
            "failed_model_ids": list(result.failed_model_ids),
        }

    async def _dispatch(self, command: str, args: dict, *, operation_id: str) -> object:
        app = self.application
        _validate_command_args(command, args, locale_choices=self.locale_choices)
        if command == "app.stop":
            self._stop_requested.set()
            return True
        if command == "capture.set":
            enabled = _boolean(args, "enabled")
            channel = args["channel"]
            accept_terms = args.get("accept_terms", False)
            consent_transaction = None
            consent_status = "applied"
            if channel == "peer" and enabled:
                if accept_terms:
                    before = getattr(self.results, "current", None)
                    await app.accept_peer_translation_eula_and_enable()
                    current = getattr(self.results, "current", None)
                    if current is not None and current is not before:
                        consent_transaction = _transaction_data(current)
                        consent_status = _transaction_status(current)
                        if consent_status in {"persistence_failed", "failed"}:
                            return {
                                "status": consent_status,
                                "consent_accepted": app.state().peer_translation_eula_accepted,
                                "transaction": consent_transaction,
                            }
                elif app.state().peer_translation_eula_accepted is not True:
                    return {"status": "action_required", "action": "accept_peer_translation_terms"}
                else:
                    await app.set_peer_translation_enabled(True)
            elif channel == "self":
                await app.set_stt_enabled(enabled)
            else:
                await app.set_peer_translation_enabled(False)
            cursor = self.events.sequence
            self.events.publish({"topic": "capture", "channel": channel})
            async for event in self.events.subscribe(
                topics=["capture"],
                channel=channel,
                include_transcripts=False,
                include_translations=False,
                after=cursor,
            ):
                state = self._channels()[channel]
                capture = state["capture"]
                transaction_data = (
                    {"transaction": consent_transaction} if consent_transaction is not None else {}
                )
                if not enabled:
                    if not state["desired_active"] and not state["effective_active"]:
                        return {"status": "applied", "channel": state, **transaction_data}
                elif state["effective_active"] and (
                    state["selected_provider"]
                    == state["runtime_provider"]
                    == state["capture_attached_provider"]
                    and not state["pending_handoff"]
                    and state["provider_status"] == "ready"
                ):
                    return {
                        "status": "degraded" if consent_status == "degraded" else "applied",
                        "channel": state,
                        **transaction_data,
                    }
                elif capture is not None and (
                    state["failure_reason"] is not None
                    or capture.get("state") in {"faulted", "failed"}
                ):
                    if capture.get("target_reason") is not None:
                        return {
                            "status": "action_required",
                            "action": "select_available_capture_target",
                            "channel": state,
                            **transaction_data,
                        }
                    return {
                        "status": "degraded",
                        "channel": state,
                        **transaction_data,
                    }
                if event["sequence"] <= cursor:
                    continue
        if command == "translation.set":
            enabled = _boolean(args, "enabled")
            await app.set_translation_enabled(enabled, allow_authorization=False)
            state = app.state()
            matched = state.translation_enabled == enabled and (
                not enabled or state.translation_runtime_ready is True
            )
            if matched:
                return {"status": "applied", "translation_enabled": state.translation_enabled}
            if enabled and app.dashboard_managed_auth_prompt_kind() not in ("none", ""):
                return {
                    "status": "action_required",
                    "action": app.dashboard_managed_auth_prompt_kind(),
                }
            return {
                "status": "degraded",
                "translation_enabled": state.translation_enabled,
                "runtime_ready": state.translation_runtime_ready,
            }
        if command == "text.submit":
            text = args["text"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError("text must be non-empty")
            await app.submit_text(text)
            return True
        if command == "microphone.test":
            if not _boolean(args, "enabled"):
                await app.stop_microphone_test()
                snapshot = app.microphone_test_snapshot()
                return {
                    "status": (
                        "applied"
                        if snapshot["state"] == "off" and not snapshot["effective_active"]
                        else "failed"
                    ),
                    "microphone_test": snapshot,
                }
            started = await app.start_microphone_test()
            ready = await app.wait_microphone_test_ready() if started else False
            snapshot = app.microphone_test_snapshot()
            if ready and snapshot["failure_reason"] is None:
                return {"status": "applied", "microphone_test": snapshot}
            if (
                started
                and snapshot["state"] in {"off", "stopping"}
                and snapshot["failure_reason"] is None
            ):
                return {"status": "cancelled", "microphone_test": snapshot}
            if snapshot["state"] in {"pending", "ready", "stopping"}:
                return {"status": "degraded", "microphone_test": snapshot}
            return {
                "status": "action_required",
                "action": "select_available_microphone",
                "microphone_test": snapshot,
            }
        if command == "audio.target.set":
            value = args["value"]
            options = app.list_loopback_capture_options()
            selected = next(
                (item for item in options if item.value == value),
                None,
            )
            if selected is None:
                raise ValueError("unknown audio target option")
            if selected.disabled:
                raise PeerCaptureTargetUnavailable("audio target is unavailable")
            return await app.apply_loopback_capture_option(value)
        if command == "audio.retry":
            return await app.retry_peer_process_capture()
        if command == "settings.apply":
            return await self._apply_fields(args)
        if command == "provider.apply":
            return await self._apply_provider(args)
        if command == "models.prepare":
            owner = self.provisioning()
            if owner is None:
                return {"status": "failed", "error": {"code": "model_owner_unavailable"}}
            backend = args.get("backend", "cpu")
            verify = args.get("verify_checksums", False)
            if not isinstance(verify, bool):
                raise ValueError("verify_checksums must be boolean")
            if backend == "cpu":
                snapshot = await owner.inspect_cpu(verify_checksums=verify)
            elif backend == "gpu":
                snapshot = await owner.inspect_gpu(explicit_intent=True, verify_checksums=verify)
            else:
                raise ValueError("unsupported model backend")
            return {"status": "applied", "models": _json(snapshot)}
        if command == "models.cancel":
            owner = self.provisioning()
            if owner is None:
                return {"status": "failed", "error": {"code": "model_owner_unavailable"}}
            backend = args.get("backend", "gpu")
            if backend not in ("cpu", "gpu"):
                raise ValueError("unsupported model backend")
            active = owner.snapshot.activity_for(backend) is not None
            if active:
                await owner.cancel_install(backend)
            return {
                "status": "cancelled" if active else "rejected",
                "backend": backend,
                "reason": None if active else "no_active_install",
            }
        if command == "gemma.prepare":
            owner = self.gemma()
            settings = app.compatibility_settings()
            if owner is None or settings is None:
                return {"status": "failed", "error": {"code": "gemma_owner_unavailable"}}
            activation = await owner.prepare(self.gemma_selection(settings))
            await activation.release()
            return {"status": "applied", "gemma": _json(owner.snapshot)}
        if command == "gemma.cancel":
            owner = self.gemma()
            if owner is None:
                return {"status": "failed", "error": {"code": "gemma_owner_unavailable"}}
            cancelled = owner.cancel()
            return {
                "status": "cancelled" if cancelled else "rejected",
                "reason": None if cancelled else "no_active_preparation",
            }
        if command == "models.retry":
            if args.get("backend", "gpu") != "gpu":
                raise ValueError("only GPU activation retry is supported")
            owner = self.gpu()
            if owner is None:
                return {"status": "failed", "error": {"code": "gpu_owner_unavailable"}}
            await owner.retry_activation()
            return {
                "status": "applied" if not owner.snapshot.discovery_failed else "degraded",
                "gpu": _json(owner.snapshot),
            }
        if command == "gpu.discover":
            owner = self.gpu()
            if owner is None:
                return {"status": "action_required", "reason": "gpu_owner_unavailable"}
            await app.ensure_gpu_device_discovery()
            snapshot = owner.snapshot
            if snapshot.discovery_failure_state == "unsupported" or (
                snapshot.discovery_attempted
                and not snapshot.devices
                and not snapshot.discovery_failed
            ):
                status = "action_required"
            elif snapshot.discovery_failed:
                status = "failed"
            else:
                status = "applied"
            return {"status": status, "gpu": _json(snapshot)}
        if command == "auth.login":
            return await self._login(args, operation_id)
        if command == "auth.logout":
            return await self._logout(args)
        if command in {"secrets.set", "secrets.delete"}:
            key = _known_secret(app, _string(args, "name"))
            value = _string(args, "value") if command == "secrets.set" else ""
            return await app.persist_provider_secret_change(key, value)
        if command == "secrets.verify":
            key = _string(args, "value")
            name = _known_secret(app, _string(args, "name"))
            provider = _VERIFIABLE_SECRET_PROVIDERS.get(name)
            if provider is None:
                return {
                    "status": "action_required",
                    "verification": "unavailable",
                    "reason": "secret_has_no_verification_protocol",
                }
            verified, reason = await app.verify_api_key(provider, key)
            return {
                "status": "applied" if verified else "rejected",
                "verification": "verified" if verified else "failed",
                "reason": reason if verified else "verification_failed",
            }
        if command == "overlay.set":
            enabled = _boolean(args, "enabled")
            await app.set_overlay_enabled(enabled)
            output = (
                await app.wait_overlay_transition() if enabled else app.output_status()["overlay"]
            )
            status = "degraded"
            reason = "overlay_runtime_not_ready"
            if not enabled:
                if (
                    output.get("desired_enabled") is False
                    and output.get("lifecycle") == "off"
                    and output.get("runtime_active") is False
                    and output.get("process_state") is None
                    and output.get("effective_target") is None
                ):
                    status = "applied"
                    reason = None
            elif output.get("ingress_stopped") is True:
                status = "cancelled"
                reason = "overlay_ingress_stopped"
            elif output.get("settings_available") is not True:
                status = "action_required"
                reason = "overlay_settings_unavailable"
            elif output.get("desired_enabled") is not True:
                status = "degraded"
                reason = "overlay_enable_not_requested"
            elif (
                output.get("lifecycle") == "connected"
                and output.get("runtime_active") is True
                and output.get("presentation_ready") is True
                and output.get("effective_target") is not None
            ):
                status = "applied"
                reason = None
            elif output.get("failure_reason") == "output_unavailable":
                status = "action_required"
                reason = "output_unavailable"
            transaction = self.results.current
            result = {
                "status": status,
                "intent": {
                    "requested_enabled": enabled,
                    "desired_enabled": output.get("desired_enabled"),
                },
                "output": _json(output),
            }
            if reason is not None:
                result["reason"] = reason
            if transaction is not None:
                result["transaction"] = {
                    **_transaction_data(transaction),
                    "effect": "overlay_intent_only",
                }
                transaction_status = _transaction_status(transaction)
                if transaction_status != "applied":
                    result["status"] = transaction_status
            return result
        if command == "overlay.lock":
            return await app.set_desktop_overlay_captions_locked(_boolean(args, "locked"))
        if command == "overlay.size":
            return await app.set_desktop_overlay_size_preset(_string(args, "preset"))
        if command == "overlay.position.reset":
            return await app.reset_desktop_overlay_position()
        if command == "overlay.calibrate":
            action = args["action"]
            if action == "begin":
                return _json(app.begin_overlay_calibration())
            if action == "change":
                return _json(
                    app.set_overlay_calibration_field(_string(args, "field"), args["value"])
                )
            if action == "apply":
                before = getattr(self.results, "current", None)
                value = app.apply_overlay_calibration()
                calibration = self.calibration()
                if calibration is None:
                    raise RuntimeError("calibration owner is unavailable")
                await calibration.wait_pending()
                result = getattr(self.results, "current", None)
                if result is not None and result is not before:
                    return {
                        "status": _transaction_status(result),
                        "calibration": _json(value),
                        "transaction": _transaction_data(result),
                    }
                return {"status": "applied", "calibration": _json(value)}
            if action == "cancel":
                return _json(app.cancel_overlay_calibration())
            raise ValueError("unknown calibration action")
        raise ValueError("unsupported command")

    async def _login(self, args: dict, operation_id: str) -> dict:
        provider = _string(args, "provider")
        if provider not in {"qq", "discord", "openrouter", "chatgpt"}:
            raise ValueError("unsupported account provider")
        app = self.application
        before = getattr(self.results, "current", None)
        if provider == "qq":
            identity = _string(args, "qq_identity")
            credential = _string(args, "credential")
            if not identity or not credential:
                raise ValueError("QQ account and credential are required")
            result = await app.start_qq_managed_auth_from_dialog(
                qq_identity=identity, credential=credential, referral_id=args.get("referral_id")
            )
        else:
            open_browser = args.get("open_browser", False)
            if not isinstance(open_browser, bool):
                raise ValueError("open_browser must be boolean")

            def authorization_url(url: str) -> None:
                self._auth_challenge = {
                    "operation_id": operation_id,
                    "provider": provider,
                    "authorization_url": url,
                }
                self.events.publish(
                    {
                        "topic": "auth",
                        "operation_id": operation_id,
                        "phase": "authorization_required",
                        "provider": provider,
                        "authorization_url": url,
                    }
                )

            if provider == "chatgpt":
                connect = await app.connect_chatgpt(
                    open_browser=open_browser,
                    authorization_url_sink=authorization_url,
                )
                self.sync_ui()
                if connect.succeeded:
                    return {
                        "status": "applied",
                        "provider": provider,
                        "authorization": "complete",
                        "first_sign_in": connect.first_sign_in,
                    }
                return {
                    "status": (
                        "rejected"
                        if connect.failure_code in {"access_denied", "plan_scope_missing"}
                        else "failed"
                    ),
                    "provider": provider,
                    "reason": connect.failure_code or "authorization_failed",
                }
            if provider == "discord":
                result = await app.start_discord_managed_auth_from_dialog(
                    referral_id=args.get("referral_id"),
                    open_browser=open_browser,
                    authorization_url_sink=authorization_url,
                    on_callback_received=lambda: self.events.publish(
                        {
                            "topic": "auth",
                            "operation_id": operation_id,
                            "phase": "callback_received",
                            "provider": provider,
                        }
                    ),
                    on_recovery_started=lambda: self.events.publish(
                        {
                            "topic": "auth",
                            "operation_id": operation_id,
                            "phase": "recovery_started",
                            "provider": provider,
                        }
                    ),
                )
            else:
                if "referral_id" in args:
                    raise ValueError("referral_id is not valid for OpenRouter authorization")
                target = app.build_managed_openrouter_byok_target()
                if target is None:
                    from puripuly_heart.app.ports.settings_view import OpenRouterPkceTarget
                    from puripuly_heart.config.provider_values import OpenRouterSelectionAlias

                    current = app.compatibility_settings()
                    alias = current.intent.translation.openrouter_selection_alias
                    if alias is None:
                        return {
                            "status": "action_required",
                            "action": "select_openrouter_byok_model",
                            "provider": provider,
                        }
                    target = OpenRouterPkceTarget(OpenRouterSelectionAlias(alias))
                result = await app.connect_openrouter_via_pkce(
                    target=target,
                    launch_source="cli",
                    open_browser=open_browser,
                    authorization_url_sink=authorization_url,
                )
        transaction = getattr(self.results, "current", None)
        transaction_data = (
            {"transaction": _transaction_data(transaction)}
            if transaction is not None and transaction is not before
            else {}
        )
        if result is True:
            status = _transaction_status(transaction) if transaction_data else "applied"
            return {
                "status": status,
                "provider": provider,
                "authorization": "complete",
                **transaction_data,
            }
        if transaction_data:
            status = _transaction_status(transaction)
            if status in {"degraded", "persistence_failed"}:
                return {"status": status, "provider": provider, **transaction_data}
        if isinstance(result, tuple) and result:
            reason = result[0]
            action = isinstance(reason, str) and reason.endswith(
                (".action_required", ".recovery_pending")
            )
            return {
                "status": "action_required" if action else "rejected",
                "provider": provider,
                "reason": "resolve_pending_authorization" if action else "authorization_failed",
                **transaction_data,
            }
        kind = app.managed_auth_last_failure_kind() if provider == "discord" else None
        return {
            "status": "action_required" if kind in {"action_required", "recovering"} else "failed",
            "provider": provider,
            "reason": "authorization_failed",
            **transaction_data,
        }

    async def _logout(self, args: dict) -> dict:
        provider = _string(args, "provider")
        if provider not in {"qq", "discord", "openrouter", "chatgpt"}:
            raise ValueError("unsupported account provider")
        app = self.application
        if provider == "chatgpt":
            signed_out = await app.sign_out_chatgpt()
            self.sync_ui()
            return {
                "status": "applied",
                "provider": provider,
                "scope": "remote_revoked" if signed_out.remote_revoked else "local_only",
            }
        current = app.compatibility_settings()
        managed = current.state.managed_connection
        if provider in {"qq", "discord"} and (
            managed.pending_delivery_ack_source or managed.pending_managed_operation_id
        ):
            return {"status": "action_required", "action": "resolve_pending_managed_authorization"}
        translation = current.intent.translation
        connection = translation.connection
        active_route = provider_llm_for_translation(
            translation.model, connection
        ) == "openrouter" and (
            (
                provider == "qq"
                and connection == "managed_china"
                and translation.openrouter_selected_source == "managed"
            )
            or (
                provider == "discord"
                and connection == "managed"
                and translation.openrouter_selected_source == "managed"
            )
            or (
                provider == "openrouter"
                and connection == "openrouter"
                and translation.openrouter_selected_source == "byok"
            )
        )
        stopped = active_route and app.state().translation_enabled
        if stopped:
            await app.set_translation_enabled(False)
            if app.state().translation_enabled:
                return {"status": "degraded", "reason": "translation_stop_failed"}
        if provider == "openrouter":
            before = getattr(self.results, "current", None)
            succeeded = await app.persist_provider_secret_change("openrouter_api_key", "")
            transaction = getattr(self.results, "current", None)
            if transaction is not None and transaction is not before:
                status = _transaction_status(transaction)
                result = {"status": status, "transaction": _transaction_data(transaction)}
            else:
                result = {"status": "applied" if succeeded else "failed"}
            result.update(provider=provider, scope="local_only")
        else:
            result = await app.logout_local_managed(provider)
        if result["status"] != "applied":
            if stopped:
                await app.set_translation_enabled(True)
                if not app.state().translation_enabled:
                    result["reason"] = "translation_restore_failed"
                    result["status"] = "degraded"
            return result
        if active_route:
            runtime = await app.apply_providers(force_rebuild_llm=True, persist_settings=False)
            if runtime is False:
                result["status"] = "degraded"
                result["reason"] = "runtime_rebuild_failed"
        self.sync_ui()
        return result

    async def _apply_provider(self, args: dict) -> object:
        edits = []
        if "channel" in args:
            provider = STTProviderName(args["provider"])
            channel = args["channel"]
            if channel not in ("self", "peer", "both"):
                raise ValueError("invalid provider channel")
            edits.extend(
                (SelfSttProviderEdit if target == "self" else PeerSttProviderEdit)(provider)
                for target in ("self", "peer")
                if channel in (target, "both")
            )
        else:
            for channel in ("self", "peer"):
                if channel in args:
                    provider = STTProviderName(args[channel])
                    edits.append(
                        (SelfSttProviderEdit if channel == "self" else PeerSttProviderEdit)(
                            provider
                        )
                    )
        if not edits:
            raise ValueError("provider.apply requires self or peer")
        return await self.application.apply_provider_intent(ProviderApplyIntent(tuple(edits)))

    async def _validate_settings_changes(
        self,
        changes: dict,
        canonical: object,
    ) -> None:
        dynamic_choices: dict[str, set[str]] = {}
        audio_fields = {
            "audio.input_host_api",
            "audio.input_device",
            "audio.output_device",
        }
        if changes.keys() & audio_fields:
            from puripuly_heart.app.services.application_audio_devices import (
                enumerate_audio_devices,
            )
            from puripuly_heart.config.audio_host_api import (
                WINDOWS_WASAPI_COMPATIBILITY_HOST_API,
                WINDOWS_WASAPI_HOST_API,
                normalize_input_host_api,
            )

            devices = await asyncio.to_thread(enumerate_audio_devices, canonical)
            host_apis = set(devices["host_apis"])
            dynamic_choices["audio.input_host_api"] = {""} | host_apis
            if WINDOWS_WASAPI_HOST_API in host_apis:
                dynamic_choices["audio.input_host_api"].add(WINDOWS_WASAPI_COMPATIBILITY_HOST_API)
            host_api = changes.get("audio.input_host_api", canonical.intent.audio.input_host_api)
            actual_host_api = normalize_input_host_api(host_api).actual_host_api
            dynamic_choices["audio.input_device"] = {""} | {
                item["name"]
                for item in devices["microphones"]
                if not actual_host_api or item["host_api"] == actual_host_api
            }
            dynamic_choices["audio.output_device"] = {""} | set(devices["loopback_outputs"])
        if changes.keys() & {"stt.gpu_device_id", "translation.gpu_device_id"}:
            owner = self.gpu()
            snapshot = getattr(owner, "snapshot", None)
            available_gpu_ids = {
                str(device.device_id) for device in getattr(snapshot, "devices", ())
            }
            for field in ("stt.gpu_device_id", "translation.gpu_device_id"):
                dynamic_choices[field] = available_gpu_ids
        if "translation.http_extension_id" in changes:
            dynamic_choices["translation.http_extension_id"] = set(_extension_ids(self.application))

        for name, value in changes.items():
            _validate_settings_field(
                name, value, locale_choices=self.locale_choices, choices=dynamic_choices
            )
        if "translation.connection" in changes:
            from puripuly_heart.config.translation_values import (
                TranslationModel,
                supported_translation_connections,
            )

            model = TranslationModel(
                changes.get("translation.model", canonical.intent.translation.model)
            )
            connection = changes["translation.connection"]
            if connection not in {item.value for item in supported_translation_connections(model)}:
                raise ValueError("unsupported translation model and connection combination")

    async def _apply_fields(self, args: dict) -> object:
        changes = args.get("changes", args)
        if type(changes) is not dict or not changes:
            raise ValueError("settings.apply requires nonempty changes object")
        unsupported = changes.keys() - SETTINGS_FIELDS
        if unsupported:
            raise ValueError("unknown settings fields")
        canonical = self.application.compatibility_settings()
        if canonical is None:
            raise RuntimeError("settings not loaded")
        await self._validate_settings_changes(changes, canonical)
        from puripuly_heart.app.services.settings.settings_application import (
            materialize_immediate_settings_intent,
            materialize_language_selection,
            materialize_prompt_apply_intent,
            materialize_provider_apply_intent,
            settings_view_surface_snapshots,
        )

        updated = canonical
        translation_order = {
            "translation.model": 0,
            "translation.connection": 1,
            "translation.connection_history": 2,
            "translation.previous_llm_model": 3,
        }
        for key, value in sorted(
            changes.items(),
            key=lambda item: translation_order.get(item[0], -1),
        ):
            if key in IMMEDIATE:
                updated = materialize_immediate_settings_intent(updated, IMMEDIATE[key](value))
            elif key == "audio.output_device":
                updated = materialize_immediate_settings_intent(
                    updated, AudioSettingsIntent((DesktopAudioOutputSettingsIntent(value),))
                )
            elif key in {"audio.input_host_api", "audio.input_device"}:
                updated = materialize_immediate_settings_intent(
                    updated,
                    AudioSettingsIntent(
                        (
                            AudioInputSettingsIntent(
                                changes.get(
                                    "audio.input_host_api", updated.intent.audio.input_host_api
                                ),
                                changes.get(
                                    "audio.input_device", updated.intent.audio.input_device
                                ),
                            ),
                        )
                    ),
                )
            elif key in {
                "translation.model",
                "translation.connection",
                "translation.connection_history",
                "translation.previous_llm_model",
            }:
                selection = settings_view_surface_snapshots(updated)[0].translation
                model = TranslationModel(value) if key == "translation.model" else selection.model
                if key == "translation.connection":
                    connection = TranslationConnection(value)
                elif key == "translation.model" and model != selection.model:
                    from puripuly_heart.config.translation_values import (
                        default_translation_connection,
                    )

                    saved = dict(selection.connection_history).get(model)
                    connection = (
                        saved
                        if saved in supported_translation_connections(model)
                        else default_translation_connection(model)
                    )
                else:
                    connection = selection.connection
                if connection not in supported_translation_connections(model):
                    raise ValueError("unsupported translation model and connection combination")
                history = dict(selection.connection_history)
                updates = ()
                if key == "translation.connection_history":
                    updates = tuple(
                        (TranslationModel(k), TranslationConnection(v)) for k, v in value.items()
                    )
                    for historic_model, historic_connection in updates:
                        if historic_connection not in supported_translation_connections(
                            historic_model
                        ):
                            raise ValueError("unsupported translation connection history")
                    history.update(updates)
                elif key in {"translation.model", "translation.connection"}:
                    updates = ((model, connection),)
                    history[model] = connection
                previous = (
                    (TranslationModel(value) if value is not None else None)
                    if key == "translation.previous_llm_model"
                    else selection.previous_llm_model
                )
                edit = TranslationSelectionEdit(
                    replace(
                        selection,
                        model=model,
                        connection=connection,
                        connection_history=tuple(history.items()),
                        previous_llm_model=previous,
                    ),
                    updates,
                )
                updated = materialize_provider_apply_intent(
                    updated,
                    ProviderApplyIntent((edit,)),
                    materialize_translation=self.settings.materialize_translation,
                )
            elif key == "managed.referral_id":
                updated = materialize_provider_apply_intent(
                    updated,
                    ProviderApplyIntent((ManagedReferralEdit(value),)),
                    materialize_translation=self.settings.materialize_translation,
                )
            elif key == "prompts.value":
                updated = materialize_prompt_apply_intent(updated, PromptApplyIntent(value))
            elif key == "custom_vocabulary":
                updated = materialize_immediate_settings_intent(
                    updated,
                    CustomVocabularySettingsIntent(value["source_language"], tuple(value["terms"])),
                )
            elif key == "osc.connection":
                updated = materialize_immediate_settings_intent(
                    updated,
                    OscConnectionSettingsIntent(
                        value["mode"],
                        value.get("send_port", updated.intent.osc.send_port),
                        value["receive_port"],
                    ),
                )
            elif key == "languages":
                if not isinstance(value, dict) or value.keys() - {
                    "source",
                    "target",
                    "secondary_target",
                    "peer_source",
                    "peer_target",
                    "peer_source_mode",
                }:
                    raise ValueError("languages requires known language choices")
                languages = updated.intent.languages
                updated = materialize_language_selection(
                    updated,
                    LanguageSelectionChange(
                        source_code=value.get("source", languages.source_language),
                        target_code=value.get("target", languages.target_language),
                        secondary_target_code=value.get(
                            "secondary_target", languages.secondary_target_language
                        ),
                        peer_source_code=value.get("peer_source", languages.peer_source_language),
                        peer_target_code=value.get("peer_target", languages.peer_target_language),
                        peer_source_mode=value.get("peer_source_mode", languages.peer_source_mode),
                        recent_source_codes=tuple(languages.recent_source_languages),
                        recent_target_codes=tuple(languages.recent_target_languages),
                    ),
                )
            elif key == "telemetry.enabled":
                if not isinstance(value, bool):
                    raise ValueError("telemetry.enabled must be boolean")
                updated = self.settings.with_telemetry_enabled(updated, value)
            else:
                if key in {"stt.provider", "peer_stt.provider"}:
                    edit = (SelfSttProviderEdit if key == "stt.provider" else PeerSttProviderEdit)(
                        STTProviderName(value)
                    )
                elif key == "stt.cloud_free_tier_providers":
                    edit = CloudFreeTierProvidersEdit(
                        tuple(STTProviderName(item) for item in value)
                    )
                else:
                    edit = PROVIDER_EDITS[key](value)
                updated = materialize_provider_apply_intent(
                    updated,
                    ProviderApplyIntent((edit,)),
                    materialize_translation=self.settings.materialize_translation,
                )
        if any(
            key in changes
            for key in (
                set(PROVIDER_EDITS)
                | {
                    "stt.provider",
                    "peer_stt.provider",
                    "stt.cloud_free_tier_providers",
                    "translation.model",
                    "translation.connection",
                    "translation.connection_history",
                    "translation.previous_llm_model",
                    "managed.referral_id",
                }
            )
        ):
            return await self.application.apply_providers(updated)
        return await self.application.apply_settings(updated)

    async def operation(self, operation_id: str) -> dict:
        return dict(self._get(operation_id).receipt)

    def _get(self, operation_id: str) -> _Operation:
        try:
            return self._operations[operation_id]
        except KeyError:
            raise UnknownOperationError("unknown or expired operation identity") from None

    async def wait(self, operation_id: str, timeout: float | None = None) -> dict:
        operation = self._get(operation_id)
        if operation.task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(operation.task), timeout=timeout)
            except TimeoutError:
                pass
        return dict(operation.receipt)

    async def cancel(self, operation_id: str) -> dict:
        operation = self._get(operation_id)
        if operation.task is None or operation.task.done():
            return dict(operation.receipt)
        if operation.task.get_name().startswith("control:models.install:"):
            owner = self.provisioning()
            if (
                owner is not None
                and operation.install_started
                and operation.backend in ("cpu", "gpu")
            ):
                await owner.cancel_install(operation.backend)
        elif operation.task.get_name().startswith("control:gemma.prepare:"):
            owner = self.gemma()
            if owner is not None:
                owner.cancel()
        else:
            return {**operation.receipt, "cancellation": "unsupported"}
        operation.task.cancel()
        await operation.task
        return dict(operation.receipt)

    def subscribe(
        self,
        *,
        topics: list[str],
        channel: str | None = None,
        include_transcripts: bool = False,
        include_translations: bool = False,
        after: int | None = None,
    ) -> AsyncIterator[dict]:
        if channel not in (None, "self", "peer"):
            raise ValueError("unsupported channel")
        if set(topics) - set(self.capabilities()["topics"]):
            raise ValueError("unsupported event topic")
        return self.events.subscribe(
            topics=topics,
            channel=channel,
            include_transcripts=include_transcripts,
            include_translations=include_translations,
            after=after,
        )


def _validate_command_args(command: str, args: dict, *, locale_choices: tuple[str, ...]) -> None:
    if type(args) is not dict:
        raise ValueError("command arguments must be an object")
    if command not in COMMAND_ARGUMENTS:
        raise ValueError("unknown application command")
    if command == "settings.apply":
        if "changes" in args:
            if args.keys() != {"changes"}:
                raise ValueError("settings changes cannot be mixed with direct fields")
            changes = args["changes"]
            if type(changes) is not dict or not changes:
                raise ValueError("settings.apply requires a nonempty changes object")
        else:
            changes = args
            if not changes:
                raise ValueError("settings.apply requires changes")
        if changes.keys() - SETTINGS_FIELDS:
            raise ValueError("unknown settings field")
        for name, value in changes.items():
            _validate_settings_field(name, value, locale_choices=locale_choices)
        return
    if command == "provider.apply":
        if "channel" in args:
            if args.keys() != {"channel", "provider"}:
                raise ValueError("channel provider selection requires channel and provider only")
            if type(args["channel"]) is not str or args["channel"] not in {
                "self",
                "peer",
                "both",
            }:
                raise ValueError("invalid provider channel")
            _validate_stt_provider(args["provider"])
            return
        if not args or args.keys() - {"self", "peer"}:
            raise ValueError("provider.apply requires self and/or peer provider values")
        for provider in args.values():
            _validate_stt_provider(provider)
        return

    allowed = COMMAND_ARGUMENTS[command].keys()
    if args.keys() - allowed:
        raise ValueError("unknown command arguments")
    required = {
        "capture.set": {"channel", "enabled"},
        "translation.set": {"enabled"},
        "text.submit": {"text"},
        "microphone.test": {"enabled"},
        "audio.target.set": {"value"},
        "secrets.set": {"name", "value"},
        "secrets.delete": {"name"},
        "secrets.verify": {"name", "value"},
        "overlay.set": {"enabled"},
        "overlay.lock": {"locked"},
        "overlay.size": {"preset"},
        "overlay.calibrate": {"action"},
        "auth.login": {"provider"},
        "auth.logout": {"provider"},
    }.get(command, set())
    if required - args.keys():
        raise ValueError("missing command arguments")

    if command == "capture.set":
        if type(args["channel"]) is not str or args["channel"] not in {"self", "peer"}:
            raise ValueError("channel must be self or peer")
        if type(args["enabled"]) is not bool:
            raise ValueError("enabled must be a boolean")
        accept_terms = args.get("accept_terms", False)
        if type(accept_terms) is not bool:
            raise ValueError("accept_terms must be a boolean")
        if accept_terms and (args["channel"] != "peer" or not args["enabled"]):
            raise ValueError("accept_terms is only valid when enabling peer capture")
    elif command in {"translation.set", "microphone.test", "overlay.set"}:
        if type(args["enabled"]) is not bool:
            raise ValueError("enabled must be a boolean")
    elif command == "overlay.lock":
        if type(args["locked"]) is not bool:
            raise ValueError("locked must be a boolean")
    elif command == "text.submit":
        if type(args["text"]) is not str or not args["text"].strip():
            raise ValueError("text must be a non-empty string")
    elif command == "audio.target.set":
        if type(args["value"]) is not str or not args["value"]:
            raise ValueError("audio target value must be a non-empty string")
    elif command in {"secrets.set", "secrets.verify"}:
        if type(args["name"]) is not str or not args["name"]:
            raise ValueError("secret name must be a non-empty string")
        if type(args["value"]) is not str or not args["value"]:
            raise ValueError("secret value must be a non-empty string")
    elif command == "secrets.delete":
        if type(args["name"]) is not str or not args["name"]:
            raise ValueError("secret name must be a non-empty string")
    elif command == "overlay.size":
        from puripuly_heart.config.desktop_overlay_values import (
            DESKTOP_FLET_SIZE_PRESET_ORDER,
        )

        if type(args["preset"]) is not str or args["preset"] not in DESKTOP_FLET_SIZE_PRESET_ORDER:
            raise ValueError("unsupported overlay size preset")
    elif command in {"models.install", "models.prepare", "models.cancel", "models.retry"}:
        backend = args.get("backend", "cpu" if command == "models.prepare" else "gpu")
        if type(backend) is not str or backend not in MODEL_BACKENDS:
            raise ValueError("unsupported model backend")
        if command == "models.retry" and backend != "gpu":
            raise ValueError("only GPU activation retry is supported")
        if command == "models.install" and "model_ids" in args:
            model_ids = args["model_ids"]
            if (
                type(model_ids) is not list
                or any(type(model_id) is not str or not model_id for model_id in model_ids)
                or len(set(model_ids)) != len(model_ids)
            ):
                raise ValueError("model_ids must be a list of unique non-empty strings")
        if command == "models.prepare" and (
            "verify_checksums" in args and type(args["verify_checksums"]) is not bool
        ):
            raise ValueError("verify_checksums must be a boolean")
    elif command in {"auth.login", "auth.logout"}:
        provider = args["provider"]
        if type(provider) is not str or provider not in {"qq", "discord", "openrouter", "chatgpt"}:
            raise ValueError("unsupported account provider")
        if command == "auth.login":
            _validate_auth_login_args(args, provider)
    elif command == "overlay.calibrate":
        _validate_calibration_args(args)


def _validate_auth_login_args(args: dict, provider: str) -> None:
    from puripuly_heart.config.provider_values import normalize_owned_referral_id

    open_browser = args.get("open_browser", False)
    if type(open_browser) is not bool:
        raise ValueError("open_browser must be a boolean")
    referral_id = args.get("referral_id")
    if referral_id is not None and (
        type(referral_id) is not str
        or not referral_id
        or normalize_owned_referral_id(referral_id) is None
    ):
        raise ValueError("invalid referral ID")
    if provider == "qq":
        if args.keys() - {
            "provider",
            "qq_identity",
            "credential",
            "referral_id",
            "open_browser",
        }:
            raise ValueError("QQ login has unsupported arguments")
        if (
            type(args.get("qq_identity")) is not str
            or not args["qq_identity"]
            or type(args.get("credential")) is not str
            or not args["credential"]
        ):
            raise ValueError("QQ identity and credential are required")
        if open_browser:
            raise ValueError("QQ login does not support browser opening")
        return
    if args.keys() - {"provider", "referral_id", "open_browser"}:
        raise ValueError("OAuth login does not accept QQ credentials")
    if provider == "openrouter" and referral_id is not None:
        raise ValueError("referral_id is not valid for OpenRouter authorization")
    if provider == "chatgpt" and referral_id is not None:
        raise ValueError("referral_id is not valid for ChatGPT authorization")


def _chatgpt_status(application: object) -> dict[str, bool]:
    snapshot_provider = getattr(application, "chatgpt_account_snapshot", None)
    if not callable(snapshot_provider):
        return {"signed_in": False, "in_progress": False, "sign_in_required": False}
    snapshot = snapshot_provider()
    required = getattr(application, "chatgpt_sign_in_required", None)
    return {
        "signed_in": bool(snapshot.signed_in),
        "in_progress": bool(snapshot.in_progress),
        "sign_in_required": bool(required()) if callable(required) else False,
    }


def _validate_calibration_args(args: dict) -> None:
    from puripuly_heart.config.overlay_calibration import (
        OVERLAY_CALIBRATION_ANCHORS,
    )

    action = args["action"]
    if type(action) is not str or action not in {"begin", "change", "apply", "cancel"}:
        raise ValueError("unsupported overlay calibration action")
    if action != "change":
        if args.keys() != {"action"}:
            raise ValueError("calibration field and value only apply to change")
        return
    if args.keys() != {"action", "field", "value"}:
        raise ValueError("calibration change requires field and value only")
    field = args["field"]
    value = args["value"]
    if type(field) is not str:
        raise ValueError("calibration field must be a string")
    if field == "anchor":
        if type(value) is not str or value not in OVERLAY_CALIBRATION_ANCHORS:
            raise ValueError("unsupported calibration anchor")
        return
    if field not in {"offset_x", "offset_y", "distance", "text_scale", "background_alpha"}:
        raise ValueError("unsupported calibration field")
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("calibration value must be a finite number")
    if field in {"distance", "text_scale"} and value <= 0:
        raise ValueError(f"{field} must be greater than zero")
    if field == "background_alpha" and not 0 <= value <= 1:
        raise ValueError("background_alpha must be between zero and one")


def _validate_stt_provider(value: object) -> None:
    if type(value) is not str or value not in {item.value for item in STTProviderName}:
        raise ValueError("unsupported STT provider")


def _supported_language_codes() -> frozenset[str]:
    from puripuly_heart.core.language import SUPPORTED_LANGUAGES

    return frozenset(SUPPORTED_LANGUAGES)


def _valid_json_value(value: object) -> bool:
    if value is None or type(value) in (bool, int, str):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is list:
        return all(_valid_json_value(item) for item in value)
    if type(value) is dict:
        return all(type(key) is str and _valid_json_value(item) for key, item in value.items())
    return False


def _validate_settings_field(
    name: str,
    value: object,
    *,
    locale_choices: tuple[str, ...],
    choices: dict[str, set[str]] | None = None,
) -> None:
    if name not in SETTINGS_FIELDS:
        raise ValueError("unknown settings field")
    if name in BOOLEAN_FIELDS:
        valid = type(value) is bool
    elif name in NUMBER_FIELDS:
        valid = type(value) in (int, float) and math.isfinite(value)
    elif name in INTEGER_FIELDS:
        valid = type(value) is int and value >= 0
    elif name in OBJECT_FIELDS:
        valid = type(value) is dict
    elif name in LIST_FIELDS:
        valid = type(value) is list and all(type(item) is str for item in value)
    elif name in OPTIONAL_STRING_FIELDS:
        valid = value is None or type(value) is str
    else:
        valid = type(value) is str
    if not valid:
        raise ValueError("settings field has an invalid value type")

    from puripuly_heart.config.desktop_overlay_values import (
        DESKTOP_FLET_SIZE_PRESET_ORDER,
    )
    from puripuly_heart.config.provider_values import (
        CLOUD_FREE_TIER_STT_PROVIDERS,
        QwenRegion,
        normalize_local_llm_base_url,
        normalize_owned_referral_id,
    )
    from puripuly_heart.config.resolved import (
        OVERLAY_TARGETS,
        is_valid_vad_onset_threshold,
    )
    from puripuly_heart.config.translation_values import (
        TranslationConnection,
        TranslationModel,
        supported_translation_connections,
    )

    language_codes = _supported_language_codes()
    if name == "locale":
        valid = value in locale_choices
    elif name in {"stt.provider", "peer_stt.provider"}:
        valid = value in {item.value for item in STTProviderName}
    elif name == "stt.cloud_free_tier_providers":
        allowed = {item.value for item in CLOUD_FREE_TIER_STT_PROVIDERS}
        valid = bool(value) and len(value) == len(set(value)) and set(value) <= allowed
    elif name == "translation.model":
        valid = value in {item.value for item in TranslationModel}
    elif name == "translation.connection":
        valid = value in {item.value for item in TranslationConnection}
    elif name == "translation.previous_llm_model":
        valid = value is None or value in {item.value for item in TranslationModel}
    elif name == "translation.connection_history":
        valid = all(
            type(model) is str
            and type(connection) is str
            and model in {item.value for item in TranslationModel}
            and connection
            in {item.value for item in supported_translation_connections(TranslationModel(model))}
            for model, connection in value.items()
        )
    elif name == "translation.qwen.region":
        valid = value in {item.value for item in QwenRegion}
    elif name in {"translation.qwen.beijing.api_host", "translation.qwen.singapore.api_host"}:
        from puripuly_heart.config.alibaba_connection import workspace_api_host_region

        parsed = workspace_api_host_region(value)
        valid = value == "" or (parsed is not None and parsed[0] == name.split(".")[2])
    elif name == "overlay.target":
        valid = value in OVERLAY_TARGETS
    elif name == "overlay.desktop_size":
        valid = value in DESKTOP_FLET_SIZE_PRESET_ORDER
    elif name == "overlay.background_alpha":
        valid = 0 <= value <= 1
    elif name in {"self_vad.speech_threshold", "peer_vad.speech_threshold"}:
        valid = is_valid_vad_onset_threshold(value)
    elif name == "peer_expected_languages":
        valid = all(item in language_codes for item in value)
    elif name == "languages":
        language_fields = {
            "source",
            "target",
            "secondary_target",
            "peer_source",
            "peer_target",
            "peer_source_mode",
        }
        valid = bool(value) and not value.keys() - language_fields
        if valid:
            for field, code in value.items():
                if field == "peer_source_mode":
                    if code not in {"manual", "auto"}:
                        valid = False
                        break
                elif field in {"secondary_target", "peer_source", "peer_target"}:
                    if code != "" and code not in language_codes:
                        valid = False
                        break
                elif code not in language_codes:
                    valid = False
                    break
    elif name == "custom_vocabulary":
        from puripuly_heart.config.provider_values import MAX_CUSTOM_VOCAB_TERMS

        valid = (
            value.keys() == {"source_language", "terms"}
            and type(value.get("source_language")) is str
            and value["source_language"] in language_codes
            and type(value.get("terms")) is list
            and len(value["terms"]) <= MAX_CUSTOM_VOCAB_TERMS
            and all(type(term) is str for term in value["terms"])
        )
    elif name == "osc.connection":
        from puripuly_heart.app.ports.osc_control import OSC_CONNECTION_MODES

        valid = (
            value.keys()
            in (
                {"mode", "receive_port"},
                {"mode", "send_port", "receive_port"},
            )
            and type(value.get("mode")) is str
            and value["mode"] in OSC_CONNECTION_MODES
            and type(value.get("receive_port")) is int
            and 1 <= value["receive_port"] <= 65535
            and (
                "send_port" not in value
                or (type(value["send_port"]) is int and 1 <= value["send_port"] <= 65535)
            )
        )
    elif name == "managed.referral_id":
        valid = value is None or not value or normalize_owned_referral_id(value) is not None
    elif name == "local_llm.base_url":
        valid = bool(value)
        if valid:
            normalize_local_llm_base_url(value)
    elif name == "local_llm.model":
        valid = bool(value)
    elif name in {"local_llm.extra_body_json", "stt.custom.extra_json"}:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            valid = False
        else:
            valid = type(parsed) is dict and _valid_json_value(parsed)
    elif name == "translation.http_extension_id":
        valid = value == "" or choices is None or value in choices.get(name, set())
    elif name in {"stt.gpu_device_id", "translation.gpu_device_id"}:
        valid = value == "auto" or choices is None or value in choices.get(name, set())
    elif name in {"audio.input_host_api", "audio.input_device", "audio.output_device"}:
        valid = choices is None or value in choices.get(name, set())
    if not valid:
        raise ValueError("unsupported settings field value")


def _string(args: dict, key: str) -> str:
    value = args[key]
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _boolean(args: dict, key: str) -> bool:
    value = args[key]
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _transaction_status(result: TransactionResult) -> str:
    status = result.status
    if status == "settings_commit_success_runtime_applied":
        return "applied"
    if status == "settings_commit_success_runtime_degraded":
        return "degraded"
    if status == TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_INTERRUPTED:
        return "interrupted"
    if status.startswith("settings_commit_failed") or status == "secret_write_failed":
        return "persistence_failed"
    return "failed"


def _transaction_data(result: TransactionResult) -> dict:
    return {
        "status": result.status,
        "message": result.message.key if result.message is not None else None,
        "diagnostics": (
            {
                "code": result.diagnostics.code,
                "category": result.diagnostics.category,
                "component": result.diagnostics.component,
            }
            if result.diagnostics is not None
            else None
        ),
    }
