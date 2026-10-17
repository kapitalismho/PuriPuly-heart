from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pytest
from puripuly_heart.app.services.self_capture_application import (
    SelfCaptureApplicationOwner,
    SelfCaptureApplicationSettings,
)
from puripuly_heart.app.wiring_stt_factory import (
    build_self_capture_session_config,
    build_self_capture_vad_signature,
    build_self_stt_provider_request,
    build_self_stt_provider_signature,
    build_self_stt_runtime_signature,
)
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from puripuly_heart.app.wiring.wiring_local_asr_provider_runtime import SharedSTTProviderFactory
from puripuly_heart.config.provider_values import STTProviderName
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.core import network_clients
from puripuly_heart.core.audio.format import float32_to_pcm16le_bytes
from puripuly_heart.core.clock import SystemClock
from puripuly_heart.core.runtime.self_capture import SelfCaptureSessionOwner
from puripuly_heart.core.self_capture import (
    SelfCaptureAdmission,
    SelfCaptureAdmissionStatus,
    SelfCaptureProviderMutation,
    SelfCaptureProviderMutationStatus,
    SelfCaptureProviderStatus,
    SelfCaptureSessionConfig,
    SelfCaptureSessionState,
)
from puripuly_heart.core.storage.secrets import InMemorySecretStore
from puripuly_heart.domain.events import UIEventType
from tests.core.test_stt_scoped_engine import wait_until
from tests.providers.test_deepgram_reuse import _installed_callback_payload
from tests.providers.test_qwen_audio import control, native
from tests.providers.test_qwen_audio_delivery_recovery import (
    final_sources,
    require_final_source,
    self_capture_pipeline,
    spoken_turn,
)

_CLOUD_PROVIDERS = (
    STTProviderName.DEEPGRAM,
    STTProviderName.QWEN_AUDIO,
    STTProviderName.SONIOX,
)


class _Admission:
    async def admit(self, config: SelfCaptureSessionConfig) -> SelfCaptureAdmission:
        _ = config
        return SelfCaptureAdmission(SelfCaptureAdmissionStatus.ADMITTED)


class _RecordingProvider:
    def __init__(self) -> None:
        self.ready = False
        self.replace_result = SelfCaptureProviderMutation(SelfCaptureProviderMutationStatus.APPLIED)
        self.handoff_result = SelfCaptureProviderMutation(SelfCaptureProviderMutationStatus.APPLIED)
        self.replace_calls: list[tuple[object, bool]] = []
        self.handoff_calls: list[tuple[object, bool]] = []
        self.reconfigure_calls: list[object] = []
        self.release_calls: list[tuple[str, float | None]] = []
        self.start_calls = 0
        self.warmup_calls = 0
        self.terminal_failure_handler = None

    def is_ready(self, config: SelfCaptureSessionConfig) -> bool:
        _ = config
        return self.ready

    async def replace(
        self,
        request: object,
        *,
        start: bool,
        on_terminal_failure: Callable[[Exception], Awaitable[None]],
    ) -> SelfCaptureProviderMutation:
        self.replace_calls.append((request, start))
        self.terminal_failure_handler = on_terminal_failure
        if self.replace_result.status is SelfCaptureProviderMutationStatus.APPLIED:
            self.ready = True
        return self.replace_result

    async def handoff(
        self,
        request: object,
        *,
        start: bool,
        on_terminal_failure: Callable[[Exception], Awaitable[None]],
    ) -> SelfCaptureProviderMutation:
        self.handoff_calls.append((request, start))
        self.terminal_failure_handler = on_terminal_failure
        return self.handoff_result

    async def cancel_handoff(self) -> bool:
        return True

    async def start_ingress(self) -> None:
        self.start_calls += 1

    async def warmup(self) -> None:
        self.warmup_calls += 1

    async def reconfigure(self, session_options: object) -> None:
        self.reconfigure_calls.append(session_options)

    async def release(
        self,
        *,
        mode: Literal["drain", "abort"],
        release_backend_after: float | None = None,
    ) -> None:
        self.release_calls.append((mode, release_backend_after))
        self.ready = False


class _Source:
    async def close(self) -> None:
        return None


class _Loop:
    def __init__(self) -> None:
        self.release = asyncio.Event()

    async def run(self, **_kwargs: object) -> None:
        await self.release.wait()


def _settings(provider: STTProviderName, language: str) -> AppSettingsVNext:
    settings = AppSettingsVNext()
    return replace(
        settings,
        intent=replace(
            settings.intent,
            stt=replace(settings.intent.stt, provider=provider.value),
            languages=replace(settings.intent.languages, source_language=language),
        ),
    )


def _with_custom_terms(
    settings: AppSettingsVNext,
    terms: dict[str, list[str]],
) -> AppSettingsVNext:
    return replace(
        settings,
        intent=replace(
            settings.intent,
            stt=replace(
                settings.intent.stt,
                custom_vocabulary_enabled=True,
                custom_terms=terms,
            ),
        ),
    )


def _build_owner(
    provider: _RecordingProvider,
    settings_holder: dict[str, AppSettingsVNext],
) -> SelfCaptureSessionOwner:
    loop = _Loop()
    return SelfCaptureSessionOwner(
        admission=_Admission(),
        provider=provider,
        provider_request_factory=lambda _config, warmup: build_self_stt_provider_request(
            settings_holder["settings"],
            warmup=warmup,
        ),
        source_factory=lambda _config: _Source(),
        vad_factory=lambda _config: object(),
        run_audio_loop=loop.run,
        vad_sink=object(),
    )


def _application_owner(
    *,
    session: SelfCaptureSessionOwner,
    settings_holder: dict[str, AppSettingsVNext],
    basic_logs: list[str] | None = None,
) -> SelfCaptureApplicationOwner:
    return SelfCaptureApplicationOwner(
        settings_provider=lambda: SelfCaptureApplicationSettings(
            config=build_self_capture_session_config(settings_holder["settings"]),
            provider_id=settings_holder["settings"].intent.stt.provider,
            qwen_region=None,
        ),
        runtime_available=lambda: True,
        capture_owner=lambda: session,
        capture_owner_if_created=lambda: session,
        persist_manual_fallback=lambda: True,
        reset_local_pending=lambda: None,
        clear_gpu_pending=lambda: None,
        overlay_state_provider=lambda: "off",
        mark_promo_eligible=lambda: None,
        dashboard_enabled_sink=lambda _enabled: None,
        dashboard_needs_key_sink=lambda _needs_key: None,
        dashboard_needs_key=lambda _available: False,
        state_sink=lambda _snapshot: None,
        sync_effective_flags=lambda: None,
        sync_local_notice=lambda: None,
        log_basic=(
            (lambda message: basic_logs.append(message))
            if basic_logs is not None
            else (lambda _message: None)
        ),
        log_diagnostic=lambda _message, _level: None,
    )


def test_self_capture_signature_excludes_next_episode_vad_policy() -> None:
    initial = _settings(STTProviderName.LOCAL_QWEN_GPU, "ko")
    endpoint_updated = replace(
        initial,
        intent=replace(
            initial.intent,
            audio=replace(initial.intent.audio, ring_buffer_ms=750),
            stt=replace(
                initial.intent.stt,
                vad_speech_threshold=0.7,
                low_latency_vad_hangover_ms=800,
            ),
        ),
    )

    assert build_self_capture_vad_signature(endpoint_updated) == (
        build_self_capture_vad_signature(initial)
    )
    assert build_self_stt_runtime_signature(endpoint_updated) != (
        build_self_stt_runtime_signature(initial)
    )

    gpu_updated = replace(
        endpoint_updated,
        intent=replace(
            endpoint_updated.intent,
            stt=replace(endpoint_updated.intent.stt, gpu_device_id="1"),
        ),
    )
    assert build_self_capture_vad_signature(gpu_updated) != (
        build_self_capture_vad_signature(endpoint_updated)
    )


@pytest.mark.parametrize("provider", _CLOUD_PROVIDERS)
def test_cloud_source_language_is_owned_by_self_provider_identity(
    provider: STTProviderName,
) -> None:
    korean = _settings(provider, "ko")
    japanese = _settings(provider, "ja")

    assert build_self_stt_runtime_signature(korean) != build_self_stt_runtime_signature(japanese)
    assert build_self_stt_provider_signature(korean) != build_self_stt_provider_signature(japanese)
    assert build_self_capture_vad_signature(korean) == build_self_capture_vad_signature(japanese)
    assert build_self_capture_session_config(korean).session_options is None
    assert build_self_stt_provider_request(japanese).config.source_language == "ja"


@pytest.mark.parametrize(
    "provider",
    (STTProviderName.LOCAL_QWEN, STTProviderName.LOCAL_QWEN_GPU),
)
def test_local_source_language_is_owned_by_session_options(
    provider: STTProviderName,
) -> None:
    korean = _settings(provider, "ko")
    japanese = _settings(provider, "ja")
    korean_config = build_self_capture_session_config(korean)
    japanese_config = build_self_capture_session_config(japanese)

    assert build_self_stt_provider_signature(korean) == build_self_stt_provider_signature(japanese)
    assert korean_config.session_options is not None
    assert japanese_config.session_options is not None
    assert korean_config.session_options != japanese_config.session_options
    assert japanese_config.session_options.source_language == "ja"


def test_local_cpu_auto_model_identity_changes_with_source_language() -> None:
    english = _settings(STTProviderName.LOCAL_CPU_AUTO, "en")
    japanese = _settings(STTProviderName.LOCAL_CPU_AUTO, "ja")
    korean = _settings(STTProviderName.LOCAL_QWEN, "ko")
    japanese_qwen = _settings(STTProviderName.LOCAL_QWEN, "ja")

    assert build_self_stt_provider_signature(english) != build_self_stt_provider_signature(japanese)
    assert build_self_stt_provider_signature(korean) == build_self_stt_provider_signature(
        japanese_qwen
    )


@pytest.mark.parametrize(
    "provider",
    (
        STTProviderName.DEEPGRAM,
        STTProviderName.SONIOX,
        STTProviderName.QWEN_AUDIO,
    ),
)
def test_custom_vocabulary_changes_supported_provider_signature(
    provider: STTProviderName,
) -> None:
    initial = _settings(provider, "ko")
    initial = _with_custom_terms(initial, {"ko": ["Puripuly"]})
    updated = _with_custom_terms(initial, {"ko": ["Puripuly", "VRChat"]})

    assert build_self_stt_runtime_signature(updated) != build_self_stt_runtime_signature(initial)
    assert build_self_stt_provider_signature(updated) != build_self_stt_provider_signature(initial)


def test_qwen_audio_capture_provider_identity_matches_runtime_provider() -> None:
    settings = _settings(STTProviderName.QWEN_AUDIO, "ko")

    capture = build_self_capture_session_config(settings)
    request = build_self_stt_provider_request(settings)

    assert capture.provider_id == STTProviderName.QWEN_AUDIO.value
    assert capture.provider_id == request.config.provider


def test_custom_vocabulary_does_not_change_unsupported_provider_signature() -> None:
    initial = _with_custom_terms(_settings(STTProviderName.LOCAL_QWEN, "ko"), {"ko": ["Puripuly"]})
    updated = _with_custom_terms(initial, {"ko": ["Puripuly", "VRChat"]})

    assert build_self_stt_runtime_signature(updated) != build_self_stt_runtime_signature(initial)
    assert build_self_stt_provider_signature(updated) == build_self_stt_provider_signature(initial)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider",
    (
        STTProviderName.DEEPGRAM,
        STTProviderName.SONIOX,
        STTProviderName.QWEN_AUDIO,
    ),
)
async def test_custom_vocabulary_change_reaches_native_tasks_and_fresh_consumer(
    provider: STTProviderName, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initial = _with_custom_terms(_settings(provider, "en"), {"en": ["Puripuly"]})
    updated = _with_custom_terms(initial, {"en": ["Puripuly", "VRChat"]})
    connections = []

    async def handle(socket):
        record = {
            "socket": socket,
            "terms": parse_qs(urlsplit(socket.request.path).query).get("keyterm", []),
            "pcm": bytearray(),
            "finished": False,
            "task_id": None,
        }
        connections.append(record)
        async for message in socket:
            if isinstance(message, bytes):
                record["pcm"].extend(message)
                continue
            packet = json.loads(message)
            if provider is STTProviderName.QWEN_AUDIO:
                if packet["header"]["action"] == "run-task":
                    record["task_id"] = packet["header"]["task_id"]
                    record["terms"] = list(packet["payload"]["parameters"].get("vocabulary", {}))
                    await socket.send(json.dumps(control(record["task_id"], "task-started")))
                else:
                    record["finished"] = True
            elif provider is STTProviderName.SONIOX:
                if "model" in packet:
                    record["terms"] = packet.get("context", {}).get("terms", [])
                elif packet.get("type") == "finalize":
                    record["finished"] = True
            elif packet["type"] == "Finalize":
                record["finished"] = True
            elif packet["type"] == "CloseStream":
                await socket.close()

    async def complete(record, text, tail=""):
        socket = record["socket"]
        if provider is STTProviderName.QWEN_AUDIO:
            await socket.send(json.dumps(native(record["task_id"], 1, text)))
            await socket.send(json.dumps(native(record["task_id"], 2, tail)))
            await socket.send(json.dumps(control(record["task_id"], "task-finished")))
        elif provider is STTProviderName.SONIOX:
            await socket.send(
                json.dumps(
                    {
                        "tokens": [
                            {"text": text, "is_final": True, "end_ms": 40},
                            {"text": "<fin>", "is_final": True},
                        ]
                    }
                )
            )
        else:
            await socket.send(
                _installed_callback_payload(
                    {"alternatives": [{"transcript": text, "confidence": 1.0, "words": []}]},
                    finalize=True,
                )
            )

    secrets = InMemorySecretStore()
    for key in ("deepgram_api_key", "soniox_api_key", "alibaba_api_key_beijing"):
        secrets.set(key, "offline")
    factory = SharedSTTProviderFactory(
        secrets=secrets, clock=SystemClock(), reset_deadline_s=1, gpu_model_path=tmp_path
    )
    async with serve(handle, "127.0.0.1", 0) as server:
        endpoint = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"

        def loopback_connect(uri, **kwargs):
            return connect(endpoint + "/?" + urlsplit(uri).query, **{**kwargs, "proxy": None})

        monkeypatch.setattr(network_clients, "external_websocket_connect", loopback_connect)
        async with self_capture_pipeline(None, provider_factory=factory, settings=initial) as env:
            await spoken_turn(env)
            await wait_until(lambda: len(connections) == 1 and connections[0]["finished"])
            assert connections[0]["terms"] == ["Puripuly"]
            assert connections[0]["pcm"] == float32_to_pcm16le_bytes(
                np.full(640, 0.1, dtype=np.float32)
            )
            old_dispatch = env.guarded
            if provider is STTProviderName.QWEN_AUDIO:
                await spoken_turn(env, 0.2)
                await wait_until(lambda: len(connections) == 2 and connections[1]["finished"])
                second = connections[1]
                await second["socket"].send(
                    json.dumps(native(second["task_id"], 1, "RETIRED PENDING SOURCE"))
                )
                engine = env.harness.self_owner.local_asr_runtime.current_provider("self")
                await wait_until(
                    lambda: any(turn.pending_update is not None for turn in engine._turns.values())
                )
                await spoken_turn(env, 0.25)
            else:
                await complete(connections[0], "Initial source.")
                await require_final_source(env, "Initial source.")
            old_connections = list(connections)
            env.settings_box[0] = updated
            applied = await env.owner.apply_intent(
                build_self_capture_session_config(updated), enabled=True
            )
            assert applied.state is SelfCaptureSessionState.RUNNING
            assert applied.effective_active
            env.guarded = env.owner._vad_dispatch
            if provider is STTProviderName.QWEN_AUDIO:
                for record in old_connections:
                    await asyncio.wait_for(record["socket"].wait_closed(), timeout=1)
                    with pytest.raises(ConnectionClosed):
                        await complete(record, "RETIRED LATE SOURCE")
                assert old_dispatch.retention_budget.used_bytes == 0
                assert len(connections) == 2
            offset = len(connections)
            await spoken_turn(env, 0.3, epoch=2)
            await wait_until(lambda: len(connections) == offset + 1 and connections[-1]["finished"])
            first = connections[-1]
            if provider is STTProviderName.QWEN_AUDIO:
                await spoken_turn(env, 0.4, epoch=2)
                await wait_until(
                    lambda: len(connections) == offset + 2 and connections[-1]["finished"]
                )
                await complete(connections[-1], " Fresh B.", " Tail B.")
                await complete(first, "Fresh A.", " Tail A.")
                expected = "Fresh A. Tail A. Fresh B. Tail B."
            else:
                await complete(first, "Fresh source.")
                expected = "Fresh source."
            transcript = await require_final_source(env, expected)
            completed = [
                event
                for event in env.harness.ui_events._queue
                if event.type is UIEventType.TRANSLATION_DONE
                and event.utterance_id == transcript.utterance_id
            ]
            assert len(completed) == 1
            assert completed[0].payload.text == env.llm.response_text
            assert all(record["terms"] == ["Puripuly", "VRChat"] for record in connections[offset:])
            values = (0.3, 0.4) if provider is STTProviderName.QWEN_AUDIO else (0.3,)
            assert [record["pcm"] for record in connections[offset:]] == [
                float32_to_pcm16le_bytes(np.full(640, value, dtype=np.float32)) for value in values
            ]
            assert all("RETIRED" not in item.text for item in final_sources(env))
            assert all("RETIRED" not in call["text"] for call in env.llm.calls)
            if provider is STTProviderName.QWEN_AUDIO:
                assert len(transcript.recognition_origins) == 4
                assert len(connections) == 4
        for record in connections:
            await asyncio.wait_for(record["socket"].wait_closed(), timeout=1)
        print(
            "STT_CUSTOM_VOCABULARY_REPLACEMENT "
            + json.dumps(
                {
                    "provider": provider.value,
                    "native_terms": [record["terms"] for record in connections],
                    "pcm_bytes": [len(record["pcm"]) for record in connections],
                    "fresh_source": transcript.text,
                    "completed_translation": completed[0].payload.text,
                    "recognition_origins": len(transcript.recognition_origins),
                    "old_pending_and_late_source_rejected": provider is STTProviderName.QWEN_AUDIO,
                    "all_connections_closed": all(
                        record["socket"].close_code is not None for record in connections
                    ),
                },
                sort_keys=True,
            )
        )


@pytest.mark.asyncio
async def test_running_cloud_source_language_change_handoffs_new_backend() -> None:
    korean = _settings(STTProviderName.DEEPGRAM, "ko")
    japanese = _settings(STTProviderName.DEEPGRAM, "ja")
    settings_holder = {"settings": korean}
    provider = _RecordingProvider()
    owner = _build_owner(provider, settings_holder)

    await owner.apply_intent(build_self_capture_session_config(korean), enabled=True)
    settings_holder["settings"] = japanese
    snapshot = await owner.apply_intent(build_self_capture_session_config(japanese), enabled=True)

    assert snapshot.state is SelfCaptureSessionState.RUNNING
    assert snapshot.runtime_signature == build_self_stt_runtime_signature(japanese)
    assert provider.reconfigure_calls == []
    assert len(provider.handoff_calls) == 1
    request, started = provider.handoff_calls[0]
    assert started is True
    assert request.config.source_language == "ja"
    await owner.close()


@pytest.mark.asyncio
async def test_running_local_source_language_change_handoffs_scoped_backend() -> None:
    korean = _settings(STTProviderName.LOCAL_QWEN, "ko")
    japanese = _settings(STTProviderName.LOCAL_QWEN, "ja")
    settings_holder = {"settings": korean}
    provider = _RecordingProvider()
    owner = _build_owner(provider, settings_holder)

    await owner.apply_intent(build_self_capture_session_config(korean), enabled=True)
    settings_holder["settings"] = japanese
    snapshot = await owner.apply_intent(build_self_capture_session_config(japanese), enabled=True)

    assert snapshot.state is SelfCaptureSessionState.RUNNING
    assert snapshot.runtime_signature == build_self_stt_runtime_signature(japanese)
    assert provider.reconfigure_calls == []
    assert len(provider.handoff_calls) == 1
    request, started = provider.handoff_calls[0]
    assert started is True
    assert request.config.source_language == "ja"
    assert request.session_options is not None
    assert request.session_options.language_hint == "ja"
    await owner.close()


@pytest.mark.asyncio
async def test_running_local_cpu_auto_language_change_handoffs_when_model_changes() -> None:
    english = _settings(STTProviderName.LOCAL_CPU_AUTO, "en")
    japanese = _settings(STTProviderName.LOCAL_CPU_AUTO, "ja")
    settings_holder = {"settings": english}
    provider = _RecordingProvider()
    owner = _build_owner(provider, settings_holder)

    await owner.apply_intent(build_self_capture_session_config(english), enabled=True)
    settings_holder["settings"] = japanese
    snapshot = await owner.apply_intent(build_self_capture_session_config(japanese), enabled=True)

    assert snapshot.state is SelfCaptureSessionState.RUNNING
    assert snapshot.runtime_signature == build_self_stt_runtime_signature(japanese)
    assert provider.reconfigure_calls == []
    assert len(provider.handoff_calls) == 1
    request, started = provider.handoff_calls[0]
    assert started is True
    assert request.config.source_language == "ja"
    await owner.close()


@pytest.mark.asyncio
async def test_language_change_while_stt_off_prepares_new_language() -> None:
    korean = _settings(STTProviderName.DEEPGRAM, "ko")
    japanese = _settings(STTProviderName.DEEPGRAM, "ja")
    settings_holder = {"settings": korean}
    provider = _RecordingProvider()
    owner = _build_owner(provider, settings_holder)

    await owner.prepare_provider(build_self_capture_session_config(korean))
    settings_holder["settings"] = japanese
    snapshot = await owner.prepare_provider(build_self_capture_session_config(japanese))
    started = await owner.apply_intent(build_self_capture_session_config(japanese), enabled=True)

    assert snapshot.runtime_signature == build_self_stt_runtime_signature(japanese)
    assert started.runtime_signature == build_self_stt_runtime_signature(japanese)
    assert [
        cast(Any, request).config.source_language for request, _start in provider.replace_calls
    ][-1] == "ja"
    await owner.close()


@pytest.mark.asyncio
async def test_replace_provider_hot_cloud_language_change_uses_application_path() -> None:
    korean = _settings(STTProviderName.DEEPGRAM, "ko")
    japanese = _settings(STTProviderName.DEEPGRAM, "ja")
    settings_holder = {"settings": korean}
    provider = _RecordingProvider()
    session = _build_owner(provider, settings_holder)
    application = _application_owner(session=session, settings_holder=settings_holder)

    await session.apply_intent(build_self_capture_session_config(korean), enabled=True)
    settings_holder["settings"] = japanese
    await application.replace_provider(smooth_local=True)

    assert session.snapshot.runtime_signature == build_self_stt_runtime_signature(japanese)
    assert provider.handoff_calls
    assert provider.handoff_calls[-1][0].config.source_language == "ja"
    await session.close()


@pytest.mark.parametrize(
    "provider_name",
    [STTProviderName.DEEPGRAM, STTProviderName.LOCAL_QWEN],
)
@pytest.mark.asyncio
async def test_failed_scoped_language_handoff_does_not_look_applied(
    provider_name: STTProviderName,
) -> None:
    korean = _settings(provider_name, "ko")
    japanese = _settings(provider_name, "ja")
    settings_holder = {"settings": korean}
    provider = _RecordingProvider()
    session = _build_owner(provider, settings_holder)
    diagnostics: list[str] = []
    application = _application_owner(
        session=session,
        settings_holder=settings_holder,
        basic_logs=diagnostics,
    )

    await session.apply_intent(build_self_capture_session_config(korean), enabled=True)
    provider.handoff_result = SelfCaptureProviderMutation(SelfCaptureProviderMutationStatus.FAILED)
    settings_holder["settings"] = japanese

    with pytest.raises(RuntimeError, match="did not apply the requested configuration"):
        await application.replace_provider(smooth_local=True)

    assert diagnostics == [
        "[STT] Provider replacement not applied: failure_reason=provider_failed "
        "provider_status=ready state=running signature_mismatch=True required=True"
    ]
    assert session.snapshot.runtime_signature == build_self_stt_runtime_signature(korean)
    assert session.snapshot.provider_status is SelfCaptureProviderStatus.READY
    await session.close()
