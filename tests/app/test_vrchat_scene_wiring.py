from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

from puripuly_heart.app import wiring_runtime_pipeline as runtime_pipeline_module
from puripuly_heart.app.wiring.wiring_runtime_pipeline import (
    RuntimePipelineLauncher,
    RuntimePipelineResourceOwner,
    compose_runtime_pipeline,
)
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.core.clock import SystemClock
from puripuly_heart.core.runtime.prebuilt_local_asr_provider_runtime import (
    PrebuiltLocalASRProviderRuntimeFactory,
)
from puripuly_heart.core.vrchat_scene import VrchatSceneSnapshot
from tests.helpers.runtime_pipeline import pipeline_inputs_from_vnext


class ManagedRelease:
    service = None

    async def rebuild(self, *, secrets: object) -> None:
        _ = secrets


class RecordingSender:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


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
    async def prepare_provider(self, config: object) -> object:
        _ = config
        return SimpleNamespace(provider_status=SimpleNamespace(value="ready"))

    def bind_publication_generation_observer(self, *, activated, retired) -> None:
        _ = activated, retired

    async def close(self) -> None:
        return


@dataclass(slots=True)
class FakeSceneService:
    started: bool = False
    closed: bool = False
    snapshot_value: VrchatSceneSnapshot = field(default_factory=VrchatSceneSnapshot)

    def snapshot(self) -> VrchatSceneSnapshot:
        return self.snapshot_value

    async def start(self) -> None:
        self.started = True

    async def close(self) -> None:
        self.closed = True


def _compose_kwargs() -> dict:
    return {
        "inputs": pipeline_inputs_from_vnext(AppSettingsVNext()),
        "secrets": object(),
        "config_path": Path("settings.json"),
        "clock": SystemClock(),
        "runtime_logging": None,
        "managed_release": ManagedRelease(),
        "managed_delegate_ready": lambda: None,
        "local_asr_factory": lambda _secrets: PrebuiltLocalASRProviderRuntimeFactory(
            self_provider=None,
            peer_provider=None,
        ),
        "self_capture_factory": lambda *args: CaptureOwner(),
        "peer_capture_factory": lambda *args: CaptureOwner(),
        "vrc_mic_state": None,
        "vrc_mic_audio_gate": None,
        "receiver_active": False,
        "stt_failure_sink": lambda _message: None,
    }


def _patch_backend(monkeypatch) -> None:  # type: ignore[no-untyped-def]
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
    monkeypatch.setattr(
        runtime_pipeline_module,
        "ChatboxPaginator",
        lambda *_a, **_k: RecordingChatbox(),
    )


async def test_compose_supplies_scene_provider_to_request_owner(monkeypatch) -> None:
    _patch_backend(monkeypatch)

    pipeline = await compose_runtime_pipeline(**_compose_kwargs())

    try:
        assert pipeline.vrchat_scene is not None
        assert pipeline.translation_requests.scene_provider is pipeline.vrchat_scene
        assert pipeline.resource_owner.vrchat_scene is pipeline.vrchat_scene
    finally:
        await pipeline.resource_owner.close()


async def test_scene_start_and_close_callbacks_drive_owned_service(monkeypatch) -> None:
    _patch_backend(monkeypatch)
    scene = FakeSceneService()

    pipeline = await compose_runtime_pipeline(**_compose_kwargs(), vrchat_scene=scene)

    try:
        assert scene.started is False
        await pipeline.start_callbacks.start_vrchat_scene()
        assert scene.started is True
    finally:
        await pipeline.resource_owner.close()
    assert scene.closed is False


async def test_shared_scene_survives_pipeline_resource_close(monkeypatch) -> None:
    _patch_backend(monkeypatch)
    scene = FakeSceneService()

    pipeline = await compose_runtime_pipeline(**_compose_kwargs(), vrchat_scene=scene)
    await pipeline.resource_owner.close()

    assert scene.closed is False
    assert pipeline.resource_owner.vrchat_scene is None


async def test_owned_scene_closes_with_pipeline_resources(monkeypatch) -> None:
    _patch_backend(monkeypatch)

    pipeline = await compose_runtime_pipeline(**_compose_kwargs())
    scene = pipeline.vrchat_scene
    assert scene is not None
    await pipeline.resource_owner.close()

    assert pipeline.resource_owner.vrchat_scene is None


async def test_launcher_reuses_scene_across_rebuilds_and_closes_on_shutdown(
    monkeypatch,
) -> None:
    created: list[FakeSceneService] = []

    def scene_factory() -> FakeSceneService:
        scene = FakeSceneService()
        created.append(scene)
        return scene

    composed_kwargs: list[dict] = []

    async def fake_compose(**kwargs):  # type: ignore[no-untyped-def]
        composed_kwargs.append(kwargs)
        return SimpleNamespace(prepare_self_provider=False, peer_capture=None)

    monkeypatch.setattr(runtime_pipeline_module, "compose_runtime_pipeline", fake_compose)

    class PeerApplication:
        async def replace_runtime(self, runtime: object) -> None:
            _ = runtime

    launcher = RuntimePipelineLauncher(
        config_path=Path("settings.json"),
        clock=SystemClock(),
        runtime_logging=object(),
        managed_release=ManagedRelease(),
        managed_delegate_ready=lambda: None,
        local_asr_factory=lambda _secrets: object(),
        self_capture_factory=lambda *args: CaptureOwner(),
        peer_capture_factory=lambda *args: CaptureOwner(),
        previous_self_capture=lambda: None,
        component_sink=lambda _components: None,
        peer_application=PeerApplication,
        configure_vrc_mic=lambda **kwargs: _noop_async(),
        stt_failure_sink=lambda _message: None,
        cleanup_failure_sink=lambda _message, _exc: None,
        vrchat_scene_factory=scene_factory,
    )
    inputs = pipeline_inputs_from_vnext(AppSettingsVNext())

    await launcher.launch(
        inputs, secrets=object(), vrc_mic_state=None, vrc_mic_audio_gate=None, receiver_active=False
    )
    await launcher.launch(
        inputs, secrets=object(), vrc_mic_state=None, vrc_mic_audio_gate=None, receiver_active=False
    )

    assert len(created) == 1
    assert created[0].started is True
    assert composed_kwargs[0]["vrchat_scene"] is created[0]
    assert composed_kwargs[1]["vrchat_scene"] is created[0]

    await launcher.close()

    assert created[0].closed is True
    assert launcher.vrchat_scene is None


async def _noop_async() -> None:
    return


async def test_prepare_reads_fresh_snapshot_per_request(monkeypatch) -> None:
    _patch_backend(monkeypatch)

    pipeline = await compose_runtime_pipeline(**_compose_kwargs())
    try:
        owner = pipeline.translation_requests

        class RecordingProvider:
            def __init__(self) -> None:
                self.calls = 0

            def snapshot(self) -> VrchatSceneSnapshot:
                self.calls += 1
                return VrchatSceneSnapshot(status="ready", participant_count=self.calls)

        provider = RecordingProvider()
        owner.scene_provider = provider  # type: ignore[assignment]

        first = owner.prepare("hello", channel="self")
        second = owner.prepare("hello again", channel="self")

        assert provider.calls == 2
        assert first.scene_snapshot == VrchatSceneSnapshot(status="ready", participant_count=1)
        assert second.scene_snapshot == VrchatSceneSnapshot(status="ready", participant_count=2)
    finally:
        await pipeline.resource_owner.close()


async def test_prepare_defaults_to_unavailable_without_ready_scene(monkeypatch) -> None:
    _patch_backend(monkeypatch)

    pipeline = await compose_runtime_pipeline(**_compose_kwargs())
    try:
        prepared = pipeline.translation_requests.prepare("hello", channel="self")

        assert prepared.scene_snapshot == VrchatSceneSnapshot()
    finally:
        await pipeline.resource_owner.close()


async def test_shutdown_adapter_closes_pipeline_scene() -> None:
    from puripuly_heart.app.adapters.application_runtime_shutdown import (
        ApplicationRuntimeShutdownAdapter,
    )

    scene = FakeSceneService()
    resources = RuntimePipelineResourceOwner()
    resources.vrchat_scene = scene
    pipeline = SimpleNamespace(
        current=SimpleNamespace(
            close_callbacks=resources.close_callbacks,
            resource_owner=resources,
        ),
        vrchat_scene=scene,
    )
    adapter = ApplicationRuntimeShutdownAdapter.__new__(ApplicationRuntimeShutdownAdapter)
    adapter.pipeline = pipeline

    await adapter.close_vrchat_scene_runtime()

    assert scene.closed is True
    assert pipeline.vrchat_scene is None
