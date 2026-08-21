from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from puripuly_heart.app.wiring_runtime_pipeline import (
    RuntimePipelineLauncher,
    RuntimePipelineResourceOwner,
    compose_runtime_pipeline,
)

from puripuly_heart.app import wiring_runtime_pipeline as runtime_pipeline_module
from puripuly_heart.config.settings import (
    AppSettings,
    TranslationConnection,
    TranslationModel,
    materialize_translation_settings,
)
from puripuly_heart.core.audio.gate import VrcMicAudioGate
from puripuly_heart.core.clock import SystemClock
from puripuly_heart.core.orchestrator.peer_translation_channel import (
    PeerTranslationChannelOwner,
)
from puripuly_heart.core.osc.receiver import VrcMicState
from puripuly_heart.core.runtime.prebuilt_local_asr_provider_runtime import (
    PrebuiltLocalASRProviderRuntimeFactory,
)
from puripuly_heart.domain.events import STTFinalEvent
from puripuly_heart.domain.models import Transcript


class ManagedRelease:
    service = None

    async def rebuild(self, *, secrets: object) -> None:
        _ = secrets


class RecordingSender:
    def __init__(self, events: list[str] | None = None) -> None:
        self.events = events
        self.closed = False

    def close(self) -> None:
        self.closed = True
        if self.events is not None:
            self.events.append("sender")


class RecordingChatbox:
    def __init__(self) -> None:
        self.messages: list[object] = []

    def enqueue(self, message: object) -> None:
        self.messages.append(message)

    def send_typing(self, is_typing: bool) -> None:
        _ = is_typing

    def set_typing_reason(self, reason: str, active: bool) -> None:
        _ = reason, active

    def clear_typing_reasons(self) -> None:
        return

    def process_due(self) -> None:
        return

    def send_immediate(self, text: str) -> bool:
        _ = text
        return True


class CaptureOwner:
    def __init__(self, label: str, events: list[str] | None = None) -> None:
        self.label = label
        self.events = events

    async def prepare_provider(self, config: object) -> object:
        _ = config
        return SimpleNamespace(provider_status=SimpleNamespace(value="ready"))

    async def close(self) -> None:
        if self.events is not None:
            self.events.append(self.label)


@pytest.mark.asyncio
async def test_pipeline_composes_each_durable_owner_once_and_injects_same_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime_pipeline_module, "create_secret_store", lambda *_a, **_k: object())
    monkeypatch.setattr(
        runtime_pipeline_module,
        "create_translation_backend",
        lambda *_a, **_k: None,
    )
    sender = RecordingSender()
    chatbox = RecordingChatbox()
    monkeypatch.setattr(runtime_pipeline_module, "VrchatOscUdpSender", lambda *_a, **_k: sender)
    monkeypatch.setattr(runtime_pipeline_module, "ChatboxPaginator", lambda *_a, **_k: chatbox)
    captured: dict[str, tuple[object, ...]] = {}

    def self_capture(*args: object) -> CaptureOwner:
        captured["self"] = args
        return CaptureOwner("self")

    def peer_capture(*args: object) -> CaptureOwner:
        captured["peer"] = args
        return CaptureOwner("peer")

    pipeline = await compose_runtime_pipeline(
        settings=AppSettings(),
        config_path=Path("settings.json"),
        clock=SystemClock(),
        runtime_logging=None,
        managed_release=ManagedRelease(),
        managed_delegate_ready=lambda: None,
        local_asr_factory=lambda _secrets: PrebuiltLocalASRProviderRuntimeFactory(
            self_provider=None,
            peer_provider=None,
        ),
        self_capture_factory=self_capture,
        peer_capture_factory=peer_capture,
        vrc_mic_state=None,
        vrc_mic_audio_gate=None,
        receiver_active=False,
        stt_failure_sink=lambda _message: None,
    )

    assert pipeline.translation_output_projection is not None
    assert pipeline.translation_output_projection.output_runtime is pipeline.output_runtime
    assert (
        pipeline.peer_translation_channel.output_projection
        is pipeline.translation_output_projection
    )
    assert pipeline.self_translation_channel.runtime is pipeline.self_runtime
    assert pipeline.peer_translation_channel.runtime is pipeline.peer_runtime
    assert pipeline.peer_translation_channel.translation_turns is pipeline.translation_turns
    assert pipeline.translation_requests.context_resolver is pipeline.context_resolver
    assert pipeline.peer_translation_channel.diagnostics is pipeline.translation_diagnostics
    assert pipeline.peer_translation_channel.local_asr_runtime is pipeline.local_asr_runtime
    assert pipeline.translation_requests.provider_runtime is pipeline.llm_runtime
    assert pipeline.peer_translation_channel.translation_requests is pipeline.translation_requests
    assert not hasattr(pipeline.peer_translation_channel, "_llm_provider_runtime")
    assert pipeline.resource_owner.output_runtime is pipeline.output_runtime
    assert pipeline.resource_owner.local_asr_runtime is pipeline.local_asr_runtime
    assert pipeline.resource_owner.self_translation_channel is pipeline.self_translation_channel
    assert pipeline.resource_owner.peer_translation_channel is pipeline.peer_translation_channel
    assert captured["self"] == (
        pipeline.self_translation_channel,
        pipeline.local_asr_runtime,
        pipeline.self_translation_channel,
        pipeline.vrc_mic_audio_gate,
    )
    assert captured["peer"] == (
        pipeline.peer_translation_channel,
        pipeline.local_asr_runtime,
        pipeline.peer_translation_channel,
    )
    assert "llm" not in PeerTranslationChannelOwner.__dataclass_fields__
    assert "stt" not in PeerTranslationChannelOwner.__dataclass_fields__
    assert "peer_stt" not in PeerTranslationChannelOwner.__dataclass_fields__


@pytest.mark.asyncio
async def test_managed_gemma_startup_constructs_provider_without_preparing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    activation_runtime = object()

    class ManagedGemma:
        runtime = activation_runtime

        async def prepare(self, _selection: object) -> object:
            events.append("prepare")
            raise AssertionError("compose must not prepare managed Gemma")

        async def deactivate(self) -> None:
            events.append("deactivate")

    class Provider:
        async def close(self) -> None:
            events.append("close")

    def create_backend(_settings: AppSettings, **kwargs: object) -> Provider:
        events.append("create")
        assert kwargs["managed_gemma_runtime"] is activation_runtime
        assert kwargs["managed_gemma_release"] is runtime_pipeline_module.noop_managed_gemma_release
        return Provider()

    monkeypatch.setattr(runtime_pipeline_module, "create_secret_store", lambda *_a, **_k: object())
    monkeypatch.setattr(runtime_pipeline_module, "create_translation_backend", create_backend)
    monkeypatch.setattr(
        runtime_pipeline_module,
        "VrchatOscUdpSender",
        lambda *_a, **_k: RecordingSender(),
    )
    monkeypatch.setattr(
        runtime_pipeline_module,
        "ChatboxPaginator",
        lambda *_a, **_k: RecordingChatbox(),
    )
    settings = AppSettings()
    settings.translation.model = TranslationModel.MANAGED_GEMMA
    settings.translation.connection = TranslationConnection.GPU
    materialize_translation_settings(settings)

    pipeline = await compose_runtime_pipeline(
        settings=settings,
        config_path=Path("settings.json"),
        clock=SystemClock(),
        runtime_logging=None,
        managed_release=ManagedRelease(),
        managed_delegate_ready=lambda: None,
        managed_gemma=ManagedGemma(),
        local_asr_factory=lambda _secrets: PrebuiltLocalASRProviderRuntimeFactory(
            self_provider=None,
            peer_provider=None,
        ),
        self_capture_factory=lambda *_args: CaptureOwner("self"),
        peer_capture_factory=lambda *_args: CaptureOwner("peer"),
        vrc_mic_state=None,
        vrc_mic_audio_gate=None,
        receiver_active=False,
        stt_failure_sink=lambda _message: None,
    )

    assert events == ["create"]
    await pipeline.resource_owner.close_llm()
    assert events[-1] == "close"
    assert not hasattr(PeerTranslationChannelOwner, "start")
    assert not hasattr(PeerTranslationChannelOwner, "stop")
    assert not pipeline.translation_turns.channel_ingress_open("self")
    assert not pipeline.translation_turns.channel_ingress_open("peer")
    await pipeline.start_callbacks.start_output(False)
    await pipeline.start_callbacks.open_self_ingress()
    await pipeline.start_callbacks.open_peer_ingress()
    await pipeline.start_callbacks.start_translation_turns()
    await pipeline.start_callbacks.start_local_asr()
    assert pipeline.translation_turns.channel_ingress_open("self")
    assert pipeline.translation_turns.channel_ingress_open("peer")
    await pipeline.resource_owner.close()
    assert not pipeline.translation_turns.channel_ingress_open("self")
    assert not pipeline.translation_turns.channel_ingress_open("peer")
    with pytest.raises(RuntimeError, match="closed"):
        await pipeline.translation_turns.start()


@pytest.mark.asyncio
async def test_pipeline_composition_preserves_runtime_configuration_and_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm_kwargs: dict[str, object] = {}
    osc_kwargs: dict[str, object] = {}
    runtime_logging = object()
    monkeypatch.setattr(runtime_pipeline_module, "create_secret_store", lambda *_a, **_k: object())

    def create_llm(*_args: object, **kwargs: object) -> None:
        llm_kwargs.update(kwargs)
        return None

    def create_osc(*_args: object, **kwargs: object) -> RecordingChatbox:
        osc_kwargs.update(kwargs)
        return RecordingChatbox()

    monkeypatch.setattr(runtime_pipeline_module, "create_translation_backend", create_llm)
    monkeypatch.setattr(
        runtime_pipeline_module,
        "VrchatOscUdpSender",
        lambda *_a, **_k: RecordingSender(),
    )
    monkeypatch.setattr(runtime_pipeline_module, "ChatboxPaginator", create_osc)

    settings = AppSettings()
    settings.osc.chatbox_include_source = False
    settings.osc.vrc_mic_intercept = True
    settings.languages.peer_source_language = "ja"
    settings.languages.peer_target_language = "en"
    original_state = VrcMicState(muted=False)
    gate = VrcMicAudioGate(state=original_state, enabled=False)

    pipeline = await compose_runtime_pipeline(
        settings=settings,
        config_path=Path("settings.json"),
        clock=SystemClock(),
        runtime_logging=runtime_logging,
        managed_release=ManagedRelease(),
        managed_delegate_ready=lambda: None,
        local_asr_factory=lambda _secrets: PrebuiltLocalASRProviderRuntimeFactory(
            self_provider=None,
            peer_provider=None,
        ),
        self_capture_factory=lambda *_args: CaptureOwner("self"),
        peer_capture_factory=lambda *_args: CaptureOwner("peer"),
        vrc_mic_state=None,
        vrc_mic_audio_gate=gate,
        receiver_active=True,
        stt_failure_sink=lambda _message: None,
    )

    config = pipeline.translation_runtime_configuration.snapshot().value
    assert config.chatbox_include_source is False
    assert config.peer_source_language == "ja"
    assert config.peer_target_language == "en"
    assert pipeline.translation_diagnostics.runtime_logging is runtime_logging
    assert llm_kwargs["runtime_logging"] is runtime_logging
    assert osc_kwargs["runtime_logging"] is runtime_logging
    assert pipeline.vrc_mic_audio_gate is gate
    assert pipeline.vrc_mic_state is not original_state
    assert gate.state is pipeline.vrc_mic_state
    assert gate.enabled is True
    assert gate.receiver_active is True
    await pipeline.resource_owner.close()


@pytest.mark.asyncio
async def test_pipeline_launcher_replaces_owned_runtime_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []

    class PriorSelf:
        async def close(self) -> None:
            events.append("close_previous")

    class PeerApplication:
        last_intent_enabled = False

        async def replace_runtime(self, runtime: object) -> None:
            events.append(("replace_peer", runtime))

    new_self = CaptureOwner("self")
    new_peer = CaptureOwner("peer")
    pipeline = SimpleNamespace(
        prepare_self_provider=True,
        self_capture=new_self,
        peer_capture=new_peer,
    )

    async def compose(**_kwargs: object) -> object:
        events.append("compose")
        return pipeline

    monkeypatch.setattr(runtime_pipeline_module, "compose_runtime_pipeline", compose)
    peer_application = PeerApplication()

    async def configure_vrc_mic(**kwargs: object) -> None:
        events.append(("configure_vrc_mic", kwargs["enabled"]))

    launcher = RuntimePipelineLauncher(
        config_path=Path("settings.json"),
        clock=SystemClock(),
        runtime_logging=object(),
        managed_release=ManagedRelease(),
        managed_delegate_ready=lambda: None,
        local_asr_factory=lambda _secrets: object(),
        self_capture_factory=lambda *_args: new_self,
        peer_capture_factory=lambda *_args: new_peer,
        previous_self_capture=lambda: PriorSelf(),
        component_sink=lambda components: events.append(("apply", components)),
        peer_application=lambda: peer_application,
        configure_vrc_mic=configure_vrc_mic,
        stt_failure_sink=lambda message: events.append(("failure", message)),
        cleanup_failure_sink=lambda _message, _exception: None,
    )
    settings = AppSettings()
    settings.ui.peer_translation_enabled = True

    result = await launcher.launch(
        settings,
        vrc_mic_state=None,
        vrc_mic_audio_gate=None,
        receiver_active=False,
    )

    assert result is pipeline
    assert events[0] == "compose"
    assert events[1] == "close_previous"
    assert events[2:4] == [
        ("apply", pipeline),
        ("replace_peer", new_peer),
    ]
    assert events[4:] == [
        ("configure_vrc_mic", settings.osc.vrc_mic_intercept),
    ]
    assert peer_application.last_intent_enabled is True


@pytest.mark.asyncio
async def test_pipeline_output_keeps_peer_off_chatbox_and_channels_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RecordingOverlay:
        def __init__(self) -> None:
            self.events: list[object] = []

        async def emit(self, event: object) -> None:
            self.events.append(event)

        def active_self_overlay_metadata(self) -> None:
            return None

    chatbox = RecordingChatbox()
    overlay = RecordingOverlay()
    monkeypatch.setattr(runtime_pipeline_module, "create_secret_store", lambda *_a, **_k: object())
    monkeypatch.setattr(
        runtime_pipeline_module,
        "create_translation_backend",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        runtime_pipeline_module,
        "VrchatOscUdpSender",
        lambda *_a, **_k: RecordingSender(),
    )
    monkeypatch.setattr(runtime_pipeline_module, "ChatboxPaginator", lambda *_a, **_k: chatbox)

    pipeline = await compose_runtime_pipeline(
        settings=AppSettings(),
        config_path=Path("settings.json"),
        clock=SystemClock(),
        runtime_logging=None,
        managed_release=ManagedRelease(),
        managed_delegate_ready=lambda: None,
        local_asr_factory=lambda _secrets: PrebuiltLocalASRProviderRuntimeFactory(
            self_provider=None,
            peer_provider=None,
        ),
        self_capture_factory=lambda *_args: CaptureOwner("self"),
        peer_capture_factory=lambda *_args: CaptureOwner("peer"),
        vrc_mic_state=None,
        vrc_mic_audio_gate=None,
        receiver_active=False,
        stt_failure_sink=lambda _message: None,
    )
    peer_owner = pipeline.peer_translation_channel
    output_projection = pipeline.translation_output_projection
    assert output_projection is not None
    await pipeline.start_callbacks.start_output(False)
    await pipeline.start_callbacks.open_self_ingress()
    await pipeline.start_callbacks.open_peer_ingress()
    await pipeline.start_callbacks.start_translation_turns()
    await pipeline.start_callbacks.start_local_asr()
    await output_projection.replace_overlay_sink(overlay)

    self_id = await pipeline.self_translation_channel.submit_text(
        "manual self text",
        source="You",
    )
    parent_id = uuid4()
    existing_peer_ids = set(peer_owner.runtime.utterances)
    await peer_owner.handle_stt_event(
        STTFinalEvent(
            utterance_id=parent_id,
            transcript=Transcript(
                utterance_id=parent_id,
                text="peer presentation text",
                is_final=True,
                created_at=peer_owner.clock.now(),
                channel="peer",
            ),
        )
    )
    await peer_owner.translation_turns.wait_for_idle()
    peer_id = next(
        utterance_id
        for utterance_id, bundle in peer_owner.runtime.utterances.items()
        if utterance_id not in existing_peer_ids
        and bundle.final is not None
        and bundle.final.text == "peer presentation text"
    )
    output_projection.publish_system_disclosure("system disclosure")

    assert [getattr(message, "text") for message in chatbox.messages] == [
        "manual self text",
        "system disclosure",
    ]
    assert [getattr(event, "channel") for event in overlay.events] == [
        "self",
        "self",
        "peer",
        "peer",
    ]
    assert [getattr(event, "utterance_id") for event in overlay.events] == [
        self_id,
        self_id,
        peer_id,
        peer_id,
    ]
    peer_denials = [
        decision
        for decision in pipeline.output_runtime.routing_decisions
        if decision.publication_kind == "peer_subtitle" and decision.route == "self_chatbox"
    ]
    assert len(peer_denials) == 1
    assert peer_denials[0].reason == "peer_chatbox_denied"
    await pipeline.resource_owner.close()


class RecordingAsyncOwner:
    def __init__(
        self,
        label: str,
        events: list[str],
        *,
        failures: int = 0,
    ) -> None:
        self.label = label
        self.events = events
        self.failures = failures

    async def close(self) -> None:
        self.events.append(self.label)
        if self.failures:
            self.failures -= 1
            raise RuntimeError(f"{self.label} failed")

    async def reset_runtime_state(self) -> None:
        await self.close()


class RecordingTurns(RecordingAsyncOwner):
    async def open_channel_ingress(self, channel: str) -> None:
        self.events.append(f"open_{channel}")

    async def close_channel_ingress(self, channel: str) -> None:
        self.events.append(f"close_{channel}")

    async def start(self) -> None:
        self.events.append("start_turns")


class RecordingSelfChannel(RecordingAsyncOwner):
    def __init__(self, runtime: RecordingAsyncOwner, events: list[str]) -> None:
        super().__init__("self_channel", events)
        self.runtime = runtime

    async def open_ingress(self) -> None:
        self.events.append("open_self_owner")

    async def close_ingress(self) -> None:
        self.events.append("close_self_owner")


class RecordingPeerChannel(RecordingAsyncOwner):
    def __init__(self, runtime: RecordingAsyncOwner, events: list[str]) -> None:
        super().__init__("peer_channel", events)
        self.runtime = runtime

    async def open_ingress(self) -> None:
        self.events.append("open_peer_owner")

    async def close_ingress(self) -> None:
        self.events.append("close_peer_owner")


class RecordingOutput(RecordingAsyncOwner):
    async def start(self, *, auto_flush_chatbox: bool) -> None:
        self.events.append(f"start_output_{auto_flush_chatbox}")


class RecordingLocalAsr(RecordingAsyncOwner):
    async def start(self) -> None:
        self.events.append("start_local_asr")


@pytest.mark.asyncio
async def test_resource_owner_uses_one_named_start_and_close_inventory() -> None:
    events: list[str] = []
    self_runtime = RecordingAsyncOwner("self_runtime", events)
    peer_runtime = RecordingAsyncOwner("peer_runtime", events)
    resources = RuntimePipelineResourceOwner(
        sender=RecordingSender(events),
        output_runtime=RecordingOutput("output", events),
        self_runtime=self_runtime,
        peer_runtime=peer_runtime,
        self_translation_channel=RecordingSelfChannel(self_runtime, events),
        peer_translation_channel=RecordingPeerChannel(peer_runtime, events),
        translation_turns=RecordingTurns("turns", events),
        local_asr_runtime=RecordingLocalAsr("local_asr", events),
        llm_runtime=RecordingAsyncOwner("llm", events),
        self_capture=RecordingAsyncOwner("self_capture", events),
        peer_capture=RecordingAsyncOwner("peer_capture", events),
    )

    await resources.start_callbacks.start_output(True)
    await resources.start_callbacks.open_self_ingress()
    await resources.start_callbacks.open_peer_ingress()
    await resources.start_callbacks.start_translation_turns()
    await resources.start_callbacks.start_local_asr()
    assert events == [
        "start_output_True",
        "open_self_owner",
        "open_self",
        "open_peer_owner",
        "open_peer",
        "start_turns",
        "start_local_asr",
    ]

    events.clear()
    await resources.close()
    assert events == [
        "self_capture",
        "peer_capture",
        "close_self_owner",
        "close_self",
        "close_peer_owner",
        "close_peer",
        "turns",
        "output",
        "self_channel",
        "peer_channel",
        "local_asr",
        "llm",
        "sender",
    ]
    assert not resources.has_resources


@pytest.mark.parametrize("acquired_count", range(1, 12))
@pytest.mark.asyncio
async def test_resource_owner_rolls_back_every_partial_acquisition_boundary(
    acquired_count: int,
) -> None:
    events: list[str] = []
    self_runtime = RecordingAsyncOwner("self_runtime", events)
    peer_runtime = RecordingAsyncOwner("peer_runtime", events)
    acquisition_order = (
        ("sender", RecordingSender(events)),
        ("output_runtime", RecordingOutput("output", events)),
        ("self_runtime", self_runtime),
        ("peer_runtime", peer_runtime),
        ("self_translation_channel", RecordingSelfChannel(self_runtime, events)),
        ("peer_translation_channel", RecordingPeerChannel(peer_runtime, events)),
        ("translation_turns", RecordingTurns("turns", events)),
        ("local_asr_runtime", RecordingLocalAsr("local_asr", events)),
        ("llm_runtime", RecordingAsyncOwner("llm", events)),
        ("self_capture", RecordingAsyncOwner("self_capture", events)),
        ("peer_capture", RecordingAsyncOwner("peer_capture", events)),
    )
    resources = RuntimePipelineResourceOwner(
        **dict(acquisition_order[:acquired_count]),
    )

    await resources.close()

    acquired = {name for name, _owner in acquisition_order[:acquired_count]}
    self_event = "self_channel" if "self_translation_channel" in acquired else "self_runtime"
    peer_event = "peer_channel" if "peer_translation_channel" in acquired else "peer_runtime"
    expected = [
        event
        for field_name, event in (
            ("self_capture", "self_capture"),
            ("peer_capture", "peer_capture"),
            ("translation_turns", "turns"),
            ("output_runtime", "output"),
            ("self_runtime", self_event),
            (
                (
                    "peer_translation_channel"
                    if "peer_translation_channel" in acquired
                    else "peer_runtime"
                ),
                peer_event,
            ),
            ("local_asr_runtime", "local_asr"),
            ("llm_runtime", "llm"),
            ("sender", "sender"),
        )
        if field_name in acquired
    ]
    assert events == expected
    assert not resources.has_resources


@pytest.mark.asyncio
async def test_resource_owner_continues_after_failure_and_retries_only_retained_owner() -> None:
    events: list[str] = []
    failed_self = RecordingAsyncOwner("self_capture", events, failures=1)
    resources = RuntimePipelineResourceOwner(
        sender=RecordingSender(events),
        output_runtime=RecordingOutput("output", events),
        self_runtime=RecordingAsyncOwner("self_channel", events),
        peer_runtime=RecordingAsyncOwner("peer_channel", events),
        translation_turns=RecordingTurns("turns", events),
        local_asr_runtime=RecordingLocalAsr("local_asr", events),
        llm_runtime=RecordingAsyncOwner("llm", events),
        self_capture=failed_self,
        peer_capture=RecordingAsyncOwner("peer_capture", events),
    )
    resources.self_ingress_open = True
    resources.peer_ingress_open = True

    with pytest.raises(BaseExceptionGroup, match="runtime pipeline cleanup failed"):
        await resources.close()

    assert events == [
        "self_capture",
        "peer_capture",
        "close_self",
        "close_peer",
        "turns",
        "output",
        "self_channel",
        "peer_channel",
        "local_asr",
        "llm",
        "sender",
    ]
    assert resources.self_capture is failed_self
    assert resources.has_resources

    events.clear()
    await resources.close()
    assert events == ["self_capture"]
    assert not resources.has_resources


@pytest.mark.parametrize("failure_stage", ("sender", "peer_owner"))
@pytest.mark.asyncio
async def test_pipeline_composition_closes_pending_llm_on_early_failure(
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    events: list[str] = []

    class Llm:
        async def close(self) -> None:
            events.append("llm")

    def create_sender(*_args: object, **_kwargs: object) -> RecordingSender:
        if failure_stage == "sender":
            raise RuntimeError("sender construction failed")
        return RecordingSender(events)

    def create_peer_owner(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("peer_owner construction failed")

    monkeypatch.setattr(runtime_pipeline_module, "create_secret_store", lambda *_a, **_k: object())
    monkeypatch.setattr(
        runtime_pipeline_module,
        "create_translation_backend",
        lambda *_a, **_k: Llm(),
    )
    monkeypatch.setattr(runtime_pipeline_module, "VrchatOscUdpSender", create_sender)
    monkeypatch.setattr(
        runtime_pipeline_module,
        "ChatboxPaginator",
        lambda *_a, **_k: RecordingChatbox(),
    )
    monkeypatch.setattr(
        runtime_pipeline_module,
        "PeerTranslationChannelOwner",
        create_peer_owner,
    )

    with pytest.raises(RuntimeError, match=f"{failure_stage} construction failed"):
        await compose_runtime_pipeline(
            settings=AppSettings(),
            config_path=Path("settings.json"),
            clock=SystemClock(),
            runtime_logging=None,
            managed_release=ManagedRelease(),
            managed_delegate_ready=lambda: None,
            local_asr_factory=lambda _secrets: PrebuiltLocalASRProviderRuntimeFactory(
                self_provider=None,
                peer_provider=None,
            ),
            self_capture_factory=lambda *_args: CaptureOwner("self"),
            peer_capture_factory=lambda *_args: CaptureOwner("peer"),
            vrc_mic_state=None,
            vrc_mic_audio_gate=None,
            receiver_active=False,
            stt_failure_sink=lambda _message: None,
        )

    if failure_stage == "sender":
        assert events == ["llm"]
    else:
        assert events[-2:] == ["llm", "sender"]


@pytest.mark.asyncio
async def test_pipeline_launcher_retains_failed_cleanup_for_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    failed_self = RecordingAsyncOwner("new_self", events, failures=1)
    resources: RuntimePipelineResourceOwner | None = None

    class PriorSelf:
        async def close(self) -> None:
            events.append("prior_self")
            raise RuntimeError("prior self close failed")

    async def compose(**kwargs: object) -> object:
        nonlocal resources
        resources = kwargs["resources"]
        resources.sender = RecordingSender(events)
        resources.self_capture = failed_self
        resources.peer_capture = RecordingAsyncOwner("new_peer", events)
        return SimpleNamespace(
            prepare_self_provider=False,
            self_capture=failed_self,
            peer_capture=resources.peer_capture,
        )

    monkeypatch.setattr(runtime_pipeline_module, "compose_runtime_pipeline", compose)
    launcher = RuntimePipelineLauncher(
        config_path=Path("settings.json"),
        clock=SystemClock(),
        runtime_logging=object(),
        managed_release=ManagedRelease(),
        managed_delegate_ready=lambda: None,
        local_asr_factory=lambda _secrets: object(),
        self_capture_factory=lambda *_args: failed_self,
        peer_capture_factory=lambda *_args: object(),
        previous_self_capture=lambda: PriorSelf(),
        component_sink=lambda _components: events.append("applied"),
        peer_application=lambda: pytest.fail("peer runtime must not be replaced"),
        configure_vrc_mic=lambda **_kwargs: pytest.fail("VRC mic must not be configured"),
        stt_failure_sink=lambda _message: None,
        cleanup_failure_sink=lambda message, _exception: events.append(message),
    )

    with pytest.raises(BaseExceptionGroup) as raised:
        await launcher.launch(
            AppSettings(),
            vrc_mic_state=None,
            vrc_mic_audio_gate=None,
            receiver_active=False,
        )

    assert "prior self close failed" in str(raised.value.exceptions[0])
    assert launcher.failed_resources is resources
    assert events == [
        "prior_self",
        "new_self",
        "new_peer",
        "sender",
        "Runtime pipeline launch cleanup failed",
    ]

    events.clear()
    await launcher.close()
    assert events == ["new_self"]
    assert launcher.failed_resources is None
