from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from puripuly_heart.app import wiring_runtime_pipeline as runtime_pipeline_module
from puripuly_heart.app.wiring_runtime_pipeline import (
    RuntimePipelineComponents,
    RuntimePipelineLauncher,
    RuntimePipelineResources,
    compose_runtime_pipeline,
)
from puripuly_heart.config.settings import AppSettings
from puripuly_heart.core.audio.gate import VrcMicAudioGate
from puripuly_heart.core.clock import SystemClock
from puripuly_heart.core.osc.receiver import VrcMicState
from puripuly_heart.core.runtime.prebuilt_local_asr_provider_runtime import (
    PrebuiltLocalASRProviderRuntimeFactory,
)


class ManagedRelease:
    service = None

    async def rebuild(self, *, secrets: object) -> None:
        _ = secrets


@pytest.mark.asyncio
async def test_pipeline_composition_preserves_runtime_configuration_and_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    llm_kwargs: dict[str, object] = {}
    osc_kwargs: dict[str, object] = {}
    runtime_logging = object()

    monkeypatch.setattr(runtime_pipeline_module, "create_secret_store", lambda *_a, **_k: object())

    def create_llm(*_args: object, **kwargs: object) -> object:
        llm_kwargs.update(kwargs)
        return object()

    def create_hub(*_args: object, **kwargs: object) -> object:
        captured.update(kwargs)
        return SimpleNamespace()

    def create_osc(*_args: object, **kwargs: object) -> object:
        osc_kwargs.update(kwargs)
        return object()

    monkeypatch.setattr(runtime_pipeline_module, "create_llm_provider", create_llm)
    monkeypatch.setattr(runtime_pipeline_module, "VrchatOscUdpSender", lambda *_a, **_k: object())
    monkeypatch.setattr(runtime_pipeline_module, "ChatboxPaginator", create_osc)
    monkeypatch.setattr(runtime_pipeline_module, "ClientHub", create_hub)

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
        local_asr_factory=lambda _secrets: object(),
        self_capture_factory=lambda _hub, _gate: object(),
        peer_capture_factory=lambda _hub: object(),
        vrc_mic_state=None,
        vrc_mic_audio_gate=gate,
        receiver_active=True,
        stt_failure_sink=lambda _message: None,
    )

    assert captured["chatbox_include_source"] is False
    assert captured["peer_source_language"] == "ja"
    assert captured["peer_target_language"] == "en"
    assert llm_kwargs["runtime_logging"] is runtime_logging
    assert osc_kwargs["runtime_logging"] is runtime_logging
    assert pipeline.vrc_mic_audio_gate is gate
    assert pipeline.vrc_mic_state is not original_state
    assert gate.state is pipeline.vrc_mic_state
    assert gate.enabled is True
    assert gate.receiver_active is True


@pytest.mark.asyncio
async def test_pipeline_launcher_replaces_owned_runtime_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []

    class PriorSelf:
        async def close(self) -> None:
            events.append("close_previous")

    class NewSelf:
        async def prepare_provider(self, config: object) -> object:
            events.append(("prepare_self", config))
            return SimpleNamespace(provider_status=SimpleNamespace(value="ready"))

    class PeerApplication:
        last_intent_enabled = False

        async def replace_runtime(self, runtime: object) -> None:
            events.append(("replace_peer", runtime))

    new_self = NewSelf()
    new_peer = object()
    pipeline = RuntimePipelineComponents(
        sender=object(),
        osc=object(),
        hub=object(),
        self_capture=new_self,
        peer_capture=new_peer,
        vrc_mic_state=VrcMicState(),
        vrc_mic_audio_gate=VrcMicAudioGate(state=VrcMicState()),
        prepare_self_provider=True,
        resources=RuntimePipelineResources(),
    )

    async def compose(**_kwargs: object) -> RuntimePipelineComponents:
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
        self_capture_factory=lambda _hub, _gate: new_self,
        peer_capture_factory=lambda _hub: new_peer,
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
    assert events[1][0] == "prepare_self"
    assert events[2:4] == [
        "close_previous",
        ("apply", pipeline),
    ]
    assert events[4:] == [
        ("replace_peer", new_peer),
        ("configure_vrc_mic", settings.osc.vrc_mic_intercept),
    ]
    assert peer_application.last_intent_enabled is True


@pytest.mark.asyncio
async def test_pipeline_output_keeps_peer_off_chatbox_and_channels_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    monkeypatch.setattr(runtime_pipeline_module, "create_llm_provider", lambda *_a, **_k: None)
    monkeypatch.setattr(runtime_pipeline_module, "VrchatOscUdpSender", lambda *_a, **_k: object())
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
        self_capture_factory=lambda _hub, _gate: object(),
        peer_capture_factory=lambda _hub: object(),
        vrc_mic_state=None,
        vrc_mic_audio_gate=None,
        receiver_active=False,
        stt_failure_sink=lambda _message: None,
    )
    hub = pipeline.hub
    await hub.replace_overlay_sink(overlay)

    self_id = await hub.submit_text("manual self text", source="You")
    peer_id = await hub.handle_peer_transcript_final_for_test("peer presentation text")
    hub.enqueue_peer_translation_disclosure("system disclosure")

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
        for decision in hub.output_runtime.routing_decisions
        if decision.publication_kind == "peer_subtitle" and decision.route == "self_chatbox"
    ]
    assert len(peer_denials) == 1
    assert peer_denials[0].reason == "peer_chatbox_denied"

    await hub.stop()


@pytest.mark.asyncio
async def test_pipeline_composition_rolls_back_partial_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class Sender:
        def close(self) -> None:
            events.append("sender")

    class Hub:
        async def stop(self) -> None:
            events.append("hub")

    class SelfCapture:
        async def close(self) -> None:
            events.append("self")

    monkeypatch.setattr(
        runtime_pipeline_module,
        "create_secret_store",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        runtime_pipeline_module,
        "create_llm_provider",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runtime_pipeline_module,
        "VrchatOscUdpSender",
        lambda *_args, **_kwargs: Sender(),
    )
    monkeypatch.setattr(
        runtime_pipeline_module,
        "ChatboxPaginator",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        runtime_pipeline_module,
        "ClientHub",
        lambda *_args, **_kwargs: Hub(),
    )

    with pytest.raises(RuntimeError, match="peer construction failed"):
        await compose_runtime_pipeline(
            settings=AppSettings(),
            config_path=Path("settings.json"),
            clock=SystemClock(),
            runtime_logging=None,
            managed_release=ManagedRelease(),
            managed_delegate_ready=lambda: None,
            local_asr_factory=lambda _secrets: object(),
            self_capture_factory=lambda _hub, _gate: SelfCapture(),
            peer_capture_factory=lambda _hub: (_ for _ in ()).throw(
                RuntimeError("peer construction failed")
            ),
            vrc_mic_state=None,
            vrc_mic_audio_gate=None,
            receiver_active=False,
            stt_failure_sink=lambda _message: None,
        )

    assert events == ["self", "hub", "sender"]


@pytest.mark.parametrize("failure_stage", ("sender", "hub"))
@pytest.mark.asyncio
async def test_pipeline_composition_closes_llm_before_hub_ownership(
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    events: list[str] = []

    class Llm:
        async def close(self) -> None:
            events.append("llm")

    class Sender:
        def close(self) -> None:
            events.append("sender")

    def create_sender(*_args: object, **_kwargs: object) -> Sender:
        if failure_stage == "sender":
            raise RuntimeError("sender construction failed")
        return Sender()

    def create_hub(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("hub construction failed")

    monkeypatch.setattr(
        runtime_pipeline_module,
        "create_secret_store",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        runtime_pipeline_module,
        "create_llm_provider",
        lambda *_args, **_kwargs: Llm(),
    )
    monkeypatch.setattr(
        runtime_pipeline_module,
        "VrchatOscUdpSender",
        create_sender,
    )
    monkeypatch.setattr(
        runtime_pipeline_module,
        "ChatboxPaginator",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(runtime_pipeline_module, "ClientHub", create_hub)

    with pytest.raises(RuntimeError, match=f"{failure_stage} construction failed"):
        await compose_runtime_pipeline(
            settings=AppSettings(),
            config_path=Path("settings.json"),
            clock=SystemClock(),
            runtime_logging=None,
            managed_release=ManagedRelease(),
            managed_delegate_ready=lambda: None,
            local_asr_factory=lambda _secrets: object(),
            self_capture_factory=lambda _hub, _gate: object(),
            peer_capture_factory=lambda _hub: object(),
            vrc_mic_state=None,
            vrc_mic_audio_gate=None,
            receiver_active=False,
            stt_failure_sink=lambda _message: None,
        )

    assert events == (["llm"] if failure_stage == "sender" else ["llm", "sender"])


@pytest.mark.asyncio
async def test_pipeline_launcher_retains_failed_cleanup_for_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class NewSelf:
        close_calls = 0

        async def prepare_provider(self, _config: object) -> object:
            return SimpleNamespace(provider_status=SimpleNamespace(value="ready"))

        async def close(self) -> None:
            self.close_calls += 1
            events.append(f"new_self_{self.close_calls}")
            if self.close_calls == 1:
                raise RuntimeError("new self close failed")

    class NewPeer:
        async def close(self) -> None:
            events.append("new_peer")

    class Hub:
        async def stop(self) -> None:
            events.append("hub")

    class Sender:
        def close(self) -> None:
            events.append("sender")

    class PriorSelf:
        async def close(self) -> None:
            events.append("prior_self")
            raise RuntimeError("prior self close failed")

    new_self = NewSelf()
    new_peer = NewPeer()

    async def compose(**kwargs: object) -> RuntimePipelineComponents:
        resources = kwargs["resources"]
        assert isinstance(resources, RuntimePipelineResources)
        resources.sender = Sender()
        resources.hub = Hub()
        resources.self_capture = new_self
        resources.peer_capture = new_peer
        return RuntimePipelineComponents(
            sender=resources.sender,
            osc=object(),
            hub=resources.hub,
            self_capture=new_self,
            peer_capture=new_peer,
            vrc_mic_state=VrcMicState(),
            vrc_mic_audio_gate=VrcMicAudioGate(state=VrcMicState()),
            prepare_self_provider=True,
            resources=resources,
        )

    monkeypatch.setattr(runtime_pipeline_module, "compose_runtime_pipeline", compose)
    launcher = RuntimePipelineLauncher(
        config_path=Path("settings.json"),
        clock=SystemClock(),
        runtime_logging=object(),
        managed_release=ManagedRelease(),
        managed_delegate_ready=lambda: None,
        local_asr_factory=lambda _secrets: object(),
        self_capture_factory=lambda _hub, _gate: new_self,
        peer_capture_factory=lambda _hub: new_peer,
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
    assert launcher.failed_resources is not None
    assert events == [
        "prior_self",
        "new_self_1",
        "new_peer",
        "hub",
        "sender",
        "Runtime pipeline launch cleanup failed",
    ]

    await launcher.close()

    assert events[-1] == "new_self_2"
    assert launcher.failed_resources is None
