from __future__ import annotations

import asyncio
import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from puripuly_heart.app.adapters.peer_capture.peer_capture_provider import (
    PeerCaptureProviderAdapter,
)
from puripuly_heart.app.adapters.peer_capture.peer_capture_vad_sink import (
    PeerCaptureVadSinkAdapter,
)
from puripuly_heart.app.adapters.self_capture.self_capture_provider import (
    SelfCaptureProviderAdapter,
)
from puripuly_heart.app.adapters.ui_runtime import (
    UiProviderRuntimeAdapter,
    UiSettingsRuntimeAdapter,
)
from puripuly_heart.app.ports.settings_view import ProviderApplyIntent
from puripuly_heart.app.services.canonical_settings_persistence import compose_settings_owner
from puripuly_heart.app.services.capture.self_capture_application import (
    SelfCaptureApplicationOwner,
    SelfCaptureApplicationSettings,
)
from puripuly_heart.app.services.provider.provider_settings import ProviderApplicationOwner
from puripuly_heart.app.services.settings.settings_application import (
    settings_view_surface_snapshots,
)
from puripuly_heart.app.services.settings.settings_transaction_result import (
    SettingsTransactionResultOwner,
)
from puripuly_heart.app.services.ui_application import UiApplicationBoundary
from puripuly_heart.app.wiring.wiring_provider_runtime import compose_provider_runtime
from puripuly_heart.app.wiring.wiring_stt_factory import (
    build_peer_capture_session_config_from_vnext,
    build_peer_stt_provider_request,
    build_self_capture_session_config_from_vnext,
    build_self_stt_provider_request_from_vnext,
)
from puripuly_heart.config.provider_values import STTProviderName
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.core.audio.format import AudioCaptureSpan
from puripuly_heart.core.clock import SystemClock
from puripuly_heart.core.local_asr.local_asr_provider_runtime import (
    LocalASRProviderRuntimeCallbacks,
)
from puripuly_heart.core.messages import (
    TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED,
)
from puripuly_heart.core.peer_capture import (
    PeerCaptureAdmission,
    PeerCaptureAdmissionStatus,
    PeerCaptureResolvedTarget,
    PeerCaptureTargetResolution,
    PeerCaptureTargetStatus,
)
from puripuly_heart.core.runtime.local_asr_provider_runtime import (
    LocalASRProviderRuntimeOwner,
)
from puripuly_heart.core.runtime.peer_channel import PeerCaptureSessionOwner
from puripuly_heart.core.runtime.provider_handle import ProviderRuntimeHandle
from puripuly_heart.core.runtime.self_capture import SelfCaptureSessionOwner
from puripuly_heart.core.self_capture import (
    SelfCaptureAdmission,
    SelfCaptureAdmissionStatus,
)
from puripuly_heart.core.stt.backend import (
    STTProviderTurnIdentity,
    STTProviderTurnRequest,
    STTProviderTurnTerminal,
)
from puripuly_heart.core.stt.rolling import RollingProviderDefinition, RollingSTTBackend
from puripuly_heart.core.stt.scoped_engine import ScopedRecognitionEngine
from puripuly_heart.core.vad.gating import SpeechChunk, SpeechEnd, SpeechStart
from puripuly_heart.ui.views.settings import SettingsView
from tests.core.runtime.test_local_asr_provider_runtime import (
    FakeGpuRuntimeFactory,
    FakeProvisioningPort,
)
from tests.helpers.fakes import RecordingOscQueue
from tests.helpers.translation_owners import compose_translation_test_harness


class _TransportSession:
    def __init__(self, *, terminal_text: str = "") -> None:
        self.closed = asyncio.Event()
        self.speech_ends: list[object] = []
        self.terminal_text = terminal_text
        self._events: asyncio.Queue[object] = asyncio.Queue()

    async def begin_turn(self, _request: STTProviderTurnRequest) -> None:
        return None

    async def send_turn_audio(self, *_args, **_kwargs) -> None:
        return None

    async def seal_turn(self, identity: STTProviderTurnIdentity, **kwargs) -> None:
        self.speech_ends.append((kwargs.get("trailing_silence_ms"), kwargs.get("reason")))
        await self._events.put(
            STTProviderTurnTerminal(
                identity,
                "final" if self.terminal_text else "empty",
                text=self.terminal_text,
                text_authority="authoritative",
            )
        )

    async def turn_events(self):
        while True:
            event = await self._events.get()
            if event is None:
                return
            yield event

    async def abort_turn(self, _identity, *, reason) -> None:
        return None

    async def stop(self) -> None:
        await self.close()

    async def close(self) -> None:
        self.closed.set()
        await self._events.put(None)


class _TransportBackend:
    def __init__(self, *, terminal_text: str = "") -> None:
        self.sessions: list[_TransportSession] = []
        self.terminal_text = terminal_text

    async def open_session(self, **_kwargs) -> _TransportSession:
        session = _TransportSession(terminal_text=self.terminal_text)
        self.sessions.append(session)
        return session


class _ScopedProviderFactory:
    def __init__(self) -> None:
        self.providers: list[ScopedRecognitionEngine] = []
        self.transports: list[_TransportBackend] = []
        self.backends: list[object] = []
        self.terminal_text = ""

    async def create(self, request, *, gpu_runtime, on_terminal_failure=None):
        _ = gpu_runtime
        transport = _TransportBackend(terminal_text=self.terminal_text)
        self.transports.append(transport)
        if request.provider_id == STTProviderName.ROLLING_FREE.value:
            backend = RollingSTTBackend(
                providers=(
                    RollingProviderDefinition(
                        name=STTProviderName.GEMINI_TRANSCRIBE,
                        build_backend=lambda transport=transport: transport,
                        is_configured=lambda: True,
                    ),
                )
            )
        else:
            backend = transport
        self.backends.append(backend)

        async def open_session(_settings, epoch_id):
            return await backend.open_session(
                projection=SimpleNamespace(mode="scoped", provider_epoch_id=epoch_id)
            )

        provider = ScopedRecognitionEngine(
            session_factory=open_session,
            channel=request.channel,
            terminal_failure_sink=on_terminal_failure,
            accepted_settings_scope=(
                request.provider_id,
                request.provider_signature,
                request.runtime_signature,
            ),
            event_drain_timeout_s=0.2,
        )
        self.providers.append(provider)
        return provider


class _RuntimeFactory:
    def __init__(self) -> None:
        self.provider_factory = _ScopedProviderFactory()
        self.runtime: LocalASRProviderRuntimeOwner | None = None
        self.callbacks: LocalASRProviderRuntimeCallbacks | None = None

    def create(self, callbacks: LocalASRProviderRuntimeCallbacks):
        self.runtime = LocalASRProviderRuntimeOwner(
            provider_factory=self.provider_factory,
            gpu_runtime_factory=FakeGpuRuntimeFactory(),
            provisioning=FakeProvisioningPort(),
            self_event_handler=callbacks.self_event_handler,
            peer_event_handler=callbacks.peer_event_handler,
            retired_event_handler=callbacks.retired_event_handler,
            self_exception_handler=callbacks.self_exception_handler,
            peer_exception_handler=callbacks.peer_exception_handler,
        )
        self.callbacks = callbacks
        return self.runtime


class _Admission:
    async def admit(self, _config):
        return SelfCaptureAdmission(SelfCaptureAdmissionStatus.ADMITTED)


class _PeerAdmission:
    async def admit(self, _config):
        return PeerCaptureAdmission(PeerCaptureAdmissionStatus.ADMITTED)


class _PeerTargetResolver:
    async def resolve(self, target):
        return PeerCaptureTargetResolution(
            PeerCaptureTargetStatus.RESOLVED,
            target=PeerCaptureResolvedTarget(intent=target),
        )


class _Source:
    def __init__(self) -> None:
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1


class _ChannelReset:
    async def reset_provider_channel(self, _channel: str) -> None:
        return None


class _Peer:
    last_provider_signature = None
    last_runtime_signature = None
    last_intent_enabled = False
    last_activation_requested = False

    def effective_enabled(self) -> bool:
        return False

    def activation_requested(self, *, intent_enabled: bool, eula_accepted: bool) -> bool:
        return bool(intent_enabled and eula_accepted)

    def capture_runtime_convergence(self, _config) -> None:
        return None


class _ManagedRelease:
    service = None

    async def rebuild(self, **_kwargs) -> None:
        return None


class _NoopPort:
    def __getattr__(self, _name):
        return _noop


async def _noop() -> None:
    return None


def _settings(provider: str) -> AppSettingsVNext:
    base = AppSettingsVNext()
    return replace(
        base,
        intent=replace(
            base.intent,
            stt=replace(base.intent.stt, provider=provider),
            languages=replace(base.intent.languages, source_language="ko"),
        ),
    )


def _peer_settings(provider: str) -> AppSettingsVNext:
    base = AppSettingsVNext()
    return replace(
        base,
        intent=replace(
            base.intent,
            peer_stt=replace(base.intent.peer_stt, provider=provider),
            languages=replace(
                base.intent.languages,
                peer_source_mode="manual",
                peer_source_language="ko",
                peer_expected_languages=("ko",),
            ),
        ),
    )


@pytest.mark.asyncio
async def test_local_scope_change_and_capture_restart_replace_engines_without_losing_recognition() -> (
    None
):
    settings = _settings(STTProviderName.LOCAL_QWEN.value)
    settings_holder = {"settings": settings}
    runtime_factory = _RuntimeFactory()
    runtime_factory.provider_factory.terminal_text = "recognized"
    osc = RecordingOscQueue()
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=osc,
        local_asr_provider_runtime_factory=runtime_factory,
    )
    runtime = runtime_factory.runtime
    assert runtime is not None
    sources: list[_Source] = []

    async def source_factory(_config):
        source = _Source()
        sources.append(source)
        return source

    async def run_audio_loop(**_kwargs):
        await asyncio.Event().wait()

    capture = SelfCaptureSessionOwner(
        admission=_Admission(),
        provider=SelfCaptureProviderAdapter(runtime, _ChannelReset()),
        provider_request_factory=lambda _config, _warmup: (
            build_self_stt_provider_request_from_vnext(settings_holder["settings"])
        ),
        source_factory=source_factory,
        vad_factory=lambda _config: object(),
        run_audio_loop=run_audio_loop,
        vad_sink=harness.self_owner,
    )
    initial_config = build_self_capture_session_config_from_vnext(settings)
    await capture.apply_intent(initial_config, enabled=True)
    initial_provider = runtime.current_provider("self")
    assert isinstance(initial_provider, ScopedRecognitionEngine)

    japanese = replace(
        settings,
        intent=replace(
            settings.intent,
            languages=replace(settings.intent.languages, source_language="ja"),
        ),
    )
    settings_holder["settings"] = japanese
    japanese_request = build_self_stt_provider_request_from_vnext(japanese)
    japanese_config = build_self_capture_session_config_from_vnext(japanese)
    await capture.apply_intent(japanese_config, enabled=True)
    configured_provider = runtime.current_provider("self")
    assert isinstance(configured_provider, ScopedRecognitionEngine)
    assert configured_provider is not initial_provider
    assert configured_provider.accepted_settings_scope is not None
    assert configured_provider.accepted_settings_scope[2] == japanese_request.runtime_signature

    first_dispatch = capture._vad_dispatch
    assert first_dispatch is not None
    frame = np.zeros(512, dtype=np.float32)
    first_id = uuid4()
    await first_dispatch.handle_vad_event(SpeechStart(first_id, frame, frame))
    await first_dispatch.handle_vad_event(SpeechEnd(first_id))

    async def wait_for_completed_turn(
        transport: _TransportBackend,
        provider: ScopedRecognitionEngine,
    ) -> None:
        while (
            not transport.sessions
            or not transport.sessions[-1].speech_ends
            or not provider.is_at_utterance_boundary
        ):
            await asyncio.sleep(0)

    await asyncio.wait_for(
        wait_for_completed_turn(
            runtime_factory.provider_factory.transports[1],
            configured_provider,
        ),
        timeout=1.0,
    )

    restarted_config = replace(
        japanese_config,
        capture_signature=("changed-device",),
    )
    await capture.apply_intent(restarted_config, enabled=True, restart=True)

    restarted_provider = runtime.current_provider("self")
    assert isinstance(restarted_provider, ScopedRecognitionEngine)
    assert restarted_provider is not configured_provider
    assert configured_provider.is_live is False
    assert runtime.snapshot.channel_for("self").provider_live is True
    assert len(sources) == 2
    assert sources[0].close_calls == 1
    second_dispatch = capture._vad_dispatch

    assert second_dispatch is not None
    second_id = uuid4()
    await second_dispatch.handle_vad_event(SpeechStart(second_id, frame, frame))
    await second_dispatch.handle_vad_event(SpeechEnd(second_id))
    await asyncio.wait_for(
        wait_for_completed_turn(
            runtime_factory.provider_factory.transports[2],
            restarted_provider,
        ),
        timeout=1.0,
    )
    await capture.close()
    await runtime.close()


@pytest.mark.asyncio
async def test_provider_apply_intent_full_vertical_rolling_gemini_soniox_reverse_and_speech_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings_owner = compose_settings_owner(tmp_path / "settings.json")
    settings_owner.start()
    settings_owner.canonical = _settings(STTProviderName.ROLLING_FREE.value)
    harness_factory = _RuntimeFactory()
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        local_asr_provider_runtime_factory=harness_factory,
    )
    runtime = harness_factory.runtime
    assert runtime is not None
    peer = _Peer()
    llm_runtime = ProviderRuntimeHandle(name="llm", provider=object())
    sources: list[_Source] = []

    async def source_factory(_config):
        source = _Source()
        sources.append(source)
        return source

    async def run_audio_loop(**_kwargs):
        await asyncio.Event().wait()

    def request_factory(config, _warmup):
        current = settings_owner.canonical
        assert current is not None
        return build_self_stt_provider_request_from_vnext(current)

    config_a = build_self_capture_session_config_from_vnext(settings_owner.canonical)
    capture_owner = SelfCaptureSessionOwner(
        admission=_Admission(),
        provider=SelfCaptureProviderAdapter(runtime, _ChannelReset()),
        provider_request_factory=request_factory,
        source_factory=source_factory,
        vad_factory=lambda _config: object(),
        run_audio_loop=run_audio_loop,
        vad_sink=harness.self_owner,
    )

    async def replace_self_stt(smooth: bool) -> None:
        await self_application.replace_provider(smooth_local=smooth)

    self_application = SelfCaptureApplicationOwner(
        settings_provider=lambda: SelfCaptureApplicationSettings(
            config=build_self_capture_session_config_from_vnext(settings_owner.canonical),
            provider_id=settings_owner.canonical.intent.stt.provider,
            qwen_region=None,
        ),
        runtime_available=lambda: True,
        capture_owner=lambda: capture_owner,
        capture_owner_if_created=lambda: capture_owner,
        persist_manual_fallback=lambda: True,
        reset_local_pending=lambda: None,
        clear_gpu_pending=lambda: None,
        overlay_state_provider=lambda: "off",
        mark_promo_eligible=lambda: None,
        dashboard_enabled_sink=lambda _value: None,
        dashboard_needs_key_sink=lambda _value: None,
        dashboard_needs_key=lambda _value: False,
        state_sink=lambda _snapshot: None,
        sync_effective_flags=lambda: None,
        sync_local_notice=lambda: None,
        log_basic=lambda _message: None,
        log_diagnostic=lambda _message, _level: None,
    )
    await capture_owner.apply_intent(config_a, enabled=True, explicit_toggle_off=False)
    source = capture_owner.source
    loop_task = capture_owner.loop_task
    assert source is sources[0]
    assert loop_task is not None
    initial_provider = runtime.current_provider("self")
    assert isinstance(initial_provider, ScopedRecognitionEngine)
    initial_backend = harness_factory.provider_factory.backends[-1]
    assert isinstance(initial_backend, RollingSTTBackend)
    assert initial_backend.providers[0].name is STTProviderName.GEMINI_TRANSCRIBE

    components = compose_provider_runtime(
        config_path=tmp_path / "settings.json",
        settings=settings_owner,
        llm_runtime_provider=lambda: llm_runtime,
        local_asr_runtime_provider=lambda: runtime,
        translation_runtime_configuration_provider=lambda: None,
        self_capture_provider=lambda: capture_owner,
        self_capture_owner=lambda: capture_owner,
        peer=lambda: peer,
        peer_desired=lambda _settings: False,
        canonical_settings=lambda value: value,
        clear_local_pending=lambda: None,
        sync_local_notice=lambda: None,
        managed_pending_sink=lambda _value: None,
        managed_pending_provider=lambda: False,
        dashboard_managed_pending_sink=lambda _value: None,
        sync_effective_flags=lambda _settings: None,
        refresh_overlay=lambda: None,
        refresh_peer_runtime=_noop,
        replace_self_stt=replace_self_stt,
        self_state_sink=lambda _snapshot: None,
        self_availability=lambda snapshot: snapshot.provider_status.value == "ready",
        gpu_recovery=lambda _settings, _plan: _noop(),
        managed_release=lambda: _ManagedRelease(),
        managed_delegate_ready=lambda: None,
        runtime_logging=SimpleNamespace(),
        translation_needs_key_sink=lambda _value: None,
        usage_refresh=_noop,
        failure_sink=lambda _message: None,
        success_sink=lambda _message: None,
        additional_signature_sink=lambda _settings: None,
    )
    components.sync_signatures(settings_owner.canonical)

    results = SettingsTransactionResultOwner()
    provider_application = ProviderApplicationOwner(
        settings=settings_owner,
        runtime=components.runtime,
        merge_settings=copy.deepcopy,
        preserve_before_replace=lambda value: _preserve(value),
        sync_ui=lambda: None,
        order24_patch_provider=lambda _value: None,
        apply_order24=lambda _value: _false(),
        remember_order22=lambda _value: None,
        mutation_service_provider=lambda: None,
        save_failure_sink=lambda _message: None,
        results=results,
        sync_memory=lambda _value: None,
        capture_runtime_signatures=components.capture_signatures_before_canonical_mutation,
        sync_signatures=components.sync_signatures,
        consume_superseded_settings=lambda _value: False,
        active_local_asr_change=lambda _base, _next: False,
        compensate_local_asr=lambda **_kwargs: _noop(),
        llm_retry_pending=lambda: False,
        mark_llm_retry=lambda: None,
    )
    ui_settings = UiSettingsRuntimeAdapter(
        settings=settings_owner,
        projection=SimpleNamespace(),
        application=SimpleNamespace(),
        merge_provider_settings=lambda value: value,
        telemetry_enabled_settings=lambda value, _enabled: value,
    )
    ui_provider = UiProviderRuntimeAdapter(
        settings=settings_owner,
        provider_application=provider_application,
        gpu=SimpleNamespace(),
        managed=SimpleNamespace(),
        credential_verification=SimpleNamespace(),
        provider_settings=SimpleNamespace(),
        build_byok_target_settings=lambda _settings: None,
    )
    boundary = UiApplicationBoundary(
        startup=SimpleNamespace(),
        input_runtime=SimpleNamespace(),
        peer_capture=SimpleNamespace(),
        settings=ui_settings,
        provider=ui_provider,
        microphone=SimpleNamespace(),
        overlay=SimpleNamespace(),
        managed=SimpleNamespace(),
        engagement=SimpleNamespace(),
        diagnostics=SimpleNamespace(),
        state=SimpleNamespace(snapshot=SimpleNamespace()),
        runtime_shutdown=_NoopPort(),
        runtime_logging=SimpleNamespace(),
        settings_secrets=SimpleNamespace(),
        osc_state_publisher=lambda: None,
    )

    monkeypatch.setattr(SettingsView, "_populate_host_apis", lambda self: None)
    monkeypatch.setattr(SettingsView, "_refresh_microphones", lambda self: None)
    monkeypatch.setattr(SettingsView, "update", lambda self: None)
    monkeypatch.setattr(SettingsView, "_load_secrets", lambda self, *_args: None)
    settings_view = SettingsView()
    baseline_settings = settings_owner.canonical
    assert baseline_settings is not None
    (
        provider_snapshot,
        general_snapshot,
        prompt_snapshot,
        overlay_snapshot,
    ) = settings_view_surface_snapshots(baseline_settings)
    settings_view.load_from_settings(
        provider=provider_snapshot,
        general=general_snapshot,
        prompt=prompt_snapshot,
        overlay=overlay_snapshot,
        config_path=tmp_path / "settings.json",
    )
    settings_view._on_stt_selected(STTProviderName.SONIOX.value)
    assert settings_view._provider_draft is not None
    assert settings_view._provider_draft.stt_provider is STTProviderName.SONIOX
    assert settings_owner.canonical.intent.stt.provider == STTProviderName.ROLLING_FREE.value
    assert runtime.current_provider("self") is initial_provider
    assert runtime.snapshot.channel_for("self").provider_id == STTProviderName.ROLLING_FREE.value
    assert capture_owner.snapshot.provider_id == STTProviderName.ROLLING_FREE.value
    soniox_intent = settings_view.build_provider_apply_settings()
    assert isinstance(soniox_intent, ProviderApplyIntent)

    original_refresh_self_stt = components.runtime.refresh_self_stt
    original_self_runtime_convergence = components.runtime.self_runtime_convergence
    components.runtime.refresh_self_stt = _noop
    components.runtime.self_runtime_convergence = lambda _settings: False
    await boundary.apply_provider_intent(soniox_intent)
    stale_result = results.current
    assert stale_result is not None
    assert stale_result.status == TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED
    assert stale_result.diagnostics is not None
    assert stale_result.diagnostics.code == "stt_runtime_apply_not_converged"
    assert settings_owner.canonical.intent.stt.provider == STTProviderName.SONIOX.value
    assert runtime.current_provider("self") is initial_provider
    assert capture_owner.snapshot.provider_id == STTProviderName.ROLLING_FREE.value
    assert runtime.snapshot.channel_for("self").provider_id == STTProviderName.ROLLING_FREE.value
    components.runtime.refresh_self_stt = original_refresh_self_stt
    components.runtime.self_runtime_convergence = original_self_runtime_convergence
    await boundary.apply_provider_intent(soniox_intent)
    assert settings_owner.canonical.intent.stt.provider == STTProviderName.SONIOX.value
    soniox_provider = runtime.current_provider("self")
    assert isinstance(soniox_provider, ScopedRecognitionEngine)
    assert runtime.snapshot.channel_for("self").provider_id == STTProviderName.SONIOX.value
    expected_soniox = build_self_capture_session_config_from_vnext(settings_owner.canonical)
    assert capture_owner.snapshot.runtime_signature == expected_soniox.runtime_signature
    assert capture_owner.snapshot.desired_active is True
    assert capture_owner.snapshot.effective_active is True
    assert capture_owner.source is source
    assert capture_owner.loop_task is loop_task

    settings_view._on_stt_selected(STTProviderName.ROLLING_FREE.value)
    rolling_intent = settings_view.build_provider_apply_settings()
    assert isinstance(rolling_intent, ProviderApplyIntent)
    utterance_id = uuid4()
    frame = np.zeros(512, dtype=np.float32)
    vad_dispatch = capture_owner._vad_dispatch
    assert vad_dispatch is not None
    await vad_dispatch.handle_vad_event(SpeechStart(utterance_id, frame, frame))
    await vad_dispatch.handle_vad_event(SpeechChunk(utterance_id, frame))
    while soniox_provider.is_at_turn_boundary:
        await asyncio.sleep(0)
    old_transport = harness_factory.provider_factory.backends[-1]
    assert isinstance(old_transport, _TransportBackend)
    apply_rolling = asyncio.create_task(boundary.apply_provider_intent(rolling_intent))

    async def _wait_until_handoff_pending_or_apply_done() -> None:
        while not (runtime.snapshot.channel_for("self").pending_handoff or apply_rolling.done()):
            await asyncio.sleep(0)

    await asyncio.wait_for(_wait_until_handoff_pending_or_apply_done(), timeout=2.0)
    assert not apply_rolling.done()
    assert runtime.snapshot.channel_for("self").pending_handoff is True
    assert runtime.snapshot.channel_for("self").provider_id == STTProviderName.SONIOX.value
    assert capture_owner.snapshot.provider_id == STTProviderName.SONIOX.value
    await vad_dispatch.handle_vad_event(SpeechEnd(utterance_id))
    await apply_rolling
    assert old_transport.sessions
    assert old_transport.sessions[-1].speech_ends

    restored = runtime.current_provider("self")
    assert isinstance(restored, ScopedRecognitionEngine)
    restored_backend = harness_factory.provider_factory.backends[-1]
    assert isinstance(restored_backend, RollingSTTBackend)
    assert restored_backend.providers[0].name is STTProviderName.GEMINI_TRANSCRIBE
    assert runtime.snapshot.channel_for("self").provider_id == STTProviderName.ROLLING_FREE.value
    assert settings_owner.canonical is not None
    assert settings_owner.canonical.intent.stt.provider == STTProviderName.ROLLING_FREE.value
    expected = build_self_capture_session_config_from_vnext(settings_owner.canonical)
    assert capture_owner.snapshot.runtime_signature == expected.runtime_signature
    assert capture_owner.snapshot.desired_active is True
    assert capture_owner.snapshot.effective_active is True
    assert capture_owner.source is source
    assert capture_owner.loop_task is loop_task
    provider_count_before_disable = len(harness_factory.provider_factory.providers)
    await capture_owner.apply_intent(
        expected,
        enabled=False,
        explicit_toggle_off=True,
    )
    assert capture_owner.snapshot.desired_active is False
    detached_channel = runtime.snapshot.channel_for("self")
    unrelated_settings = replace(
        settings_owner.canonical,
        intent=replace(
            settings_owner.canonical.intent,
            translation=replace(
                settings_owner.canonical.intent.translation,
                concurrency_limit=(
                    settings_owner.canonical.intent.translation.concurrency_limit + 1
                ),
            ),
        ),
    )
    unrelated_plan = components.runtime.build_plan(
        unrelated_settings,
        force_rebuild_llm=False,
    )
    assert unrelated_plan.should_refresh_self_stt is False
    await components.runtime.apply(unrelated_settings, unrelated_plan)
    assert len(harness_factory.provider_factory.providers) == provider_count_before_disable
    assert runtime.snapshot.channel_for("self").provider_id == detached_channel.provider_id
    assert capture_owner.snapshot.desired_active is False

    await capture_owner.close()

    await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("initial_provider_id", "next_provider_id", "active_turn"),
    (
        (
            STTProviderName.SONIOX.value,
            STTProviderName.GEMINI_TRANSCRIBE.value,
            True,
        ),
        (
            STTProviderName.GEMINI_TRANSCRIBE.value,
            STTProviderName.SONIOX.value,
            True,
        ),
        (
            STTProviderName.SONIOX.value,
            STTProviderName.GEMINI_TRANSCRIBE.value,
            False,
        ),
    ),
    ids=("soniox-to-gemini-active", "gemini-to-soniox-active", "idle"),
)
async def test_peer_provider_handoff_commits_from_owned_speech_end_and_routes_next_turn(
    initial_provider_id: str,
    next_provider_id: str,
    active_turn: bool,
) -> None:
    harness_factory = _RuntimeFactory()
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        local_asr_provider_runtime_factory=harness_factory,
    )
    runtime = harness_factory.runtime
    assert runtime is not None
    sources: list[_Source] = []

    async def source_factory(_config, _target):
        source = _Source()
        sources.append(source)
        return source

    async def run_audio_loop(**_kwargs):
        await asyncio.Event().wait()

    capture = PeerCaptureSessionOwner(
        admission=_PeerAdmission(),
        target_resolver=_PeerTargetResolver(),
        provider=PeerCaptureProviderAdapter(runtime, harness.peer_owner),
        clock=SystemClock(),
        provider_request_factory=lambda config, warmup: build_peer_stt_provider_request(
            config,
            gpu_device_id="auto",
            warmup=warmup,
        ),
        source_factory=source_factory,
        vad_factory=lambda _config: object(),
        run_audio_loop=run_audio_loop,
        vad_sink=PeerCaptureVadSinkAdapter(lambda: harness.peer_owner),
    )
    callbacks = harness_factory.callbacks
    assert callbacks is not None
    callback_owner = callbacks.peer_exception_handler.__self__
    callback_owner.bind_peer_capture(capture)
    capture.bind_publication_generation_observer(
        activated=harness.output_runtime.activate_peer_generation,
        retired=harness.output_runtime.retire_peer_generation,
    )
    initial_config = build_peer_capture_session_config_from_vnext(
        _peer_settings(initial_provider_id)
    )
    next_config = build_peer_capture_session_config_from_vnext(_peer_settings(next_provider_id))
    sequence = 0

    def owned(event):
        nonlocal sequence
        ledger = capture.segment_ledger
        assert ledger is not None
        result = ledger.observe_vad_event(
            event,
            now_monotonic_s=asyncio.get_running_loop().time(),
        )
        sequence += 1
        return result

    def frame_event(utterance_id, *, start: bool):
        frame = np.ones(512, dtype=np.float32)
        source_start = sequence * 512
        span = AudioCaptureSpan(
            capture_epoch=1,
            callback_sequence=sequence,
            source_sample_rate_hz=16000,
            source_start_sample=source_start,
            source_end_sample=source_start + 512,
            source_start_monotonic_s=source_start / 16000,
            source_end_monotonic_s=(source_start + 512) / 16000,
            normalized_sample_rate_hz=16000,
            normalized_start_sample=source_start,
            normalized_end_sample=source_start + 512,
        )
        if start:
            return SpeechStart(
                utterance_id,
                np.empty((0,), dtype=np.float32),
                frame,
                chunk_capture=(span,),
            )
        return SpeechChunk(utterance_id, frame, chunk_capture=(span,))

    async def wait_until(predicate) -> None:
        async def wait_loop() -> None:
            while not predicate():
                await asyncio.sleep(0)

        await asyncio.wait_for(wait_loop(), timeout=2.0)

    guarded_sink = None

    try:
        started = await capture.apply_intent(initial_config, enabled=True)
        assert started.provider_id == initial_provider_id
        assert runtime.snapshot.channel_for("peer").provider_id == initial_provider_id
        assert capture.source is sources[0]
        initial_engine = runtime.current_provider("peer")
        assert isinstance(initial_engine, ScopedRecognitionEngine)
        old_transport = harness_factory.provider_factory.transports[-1]
        guarded_sink = capture.guard_vad_sink()

        if active_turn:
            old_utterance_id = uuid4()
            await guarded_sink.handle_owned_vad_event(
                owned(frame_event(old_utterance_id, start=True))
            )
            await guarded_sink.handle_owned_vad_event(
                owned(frame_event(old_utterance_id, start=False))
            )
            await wait_until(
                lambda: bool(old_transport.sessions) and not initial_engine.is_at_turn_boundary
            )

        apply_next = asyncio.create_task(capture.apply_intent(next_config, enabled=True))
        if active_turn:
            await wait_until(
                lambda: runtime.snapshot.channel_for("peer").pending_handoff or apply_next.done()
            )
            assert apply_next.done() is False
            assert runtime.snapshot.channel_for("peer").pending_handoff is True
            assert capture.snapshot.provider_id == initial_provider_id
            await guarded_sink.handle_owned_vad_event(owned(SpeechEnd(old_utterance_id)))

        applied = await asyncio.wait_for(apply_next, timeout=2.0)
        assert applied.provider_id == next_provider_id
        assert applied.runtime_signature == next_config.runtime_signature
        assert runtime.snapshot.channel_for("peer").provider_id == next_provider_id
        assert runtime.snapshot.channel_for("peer").pending_handoff is False
        if active_turn:
            assert old_transport.sessions[-1].speech_ends

        new_engine = runtime.current_provider("peer")
        assert isinstance(new_engine, ScopedRecognitionEngine)
        assert new_engine is not initial_engine
        new_transport = harness_factory.provider_factory.transports[-1]
        assert new_transport is not old_transport
        assert new_transport.sessions == []
        next_utterance_id = uuid4()
        await guarded_sink.handle_owned_vad_event(owned(frame_event(next_utterance_id, start=True)))
        await guarded_sink.handle_owned_vad_event(
            owned(frame_event(next_utterance_id, start=False))
        )
        await wait_until(lambda: bool(new_transport.sessions))
        assert len(old_transport.sessions) == (1 if active_turn else 0)
        await guarded_sink.handle_owned_vad_event(owned(SpeechEnd(next_utterance_id)))
        await wait_until(lambda: bool(new_transport.sessions[-1].speech_ends))
    finally:
        if guarded_sink is not None:
            await guarded_sink.abort()
        await capture.close()
        await runtime.close()


async def _preserve(value):
    return value


async def _false() -> bool:
    return False
