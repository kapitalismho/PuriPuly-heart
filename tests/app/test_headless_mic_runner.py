from __future__ import annotations

import logging

import pytest

import puripuly_heart.app.headless_mic as headless_mic
from puripuly_heart.config.audio_host_api import (
    WINDOWS_WASAPI_COMPATIBILITY_HOST_API,
    WINDOWS_WASAPI_HOST_API,
)
from puripuly_heart.config.settings import (
    AppSettings,
    LLMProviderName,
    OpenRouterCredentialSource,
    OpenRouterSettings,
    ProviderSettings,
    STTProviderName,
)
from puripuly_heart.config.vad_defaults import (
    DEFAULT_LOW_LATENCY_VAD_HANGOVER_MS,
    DEFAULT_STABLE_VAD_HANGOVER_MS,
)
from puripuly_heart.core.storage.secrets import InMemorySecretStore


def _patch_headless_mic_startup(
    monkeypatch: pytest.MonkeyPatch,
    vad_path,
    *,
    resolve_device,
    source_factory,
) -> None:
    class FakeSender:
        def close(self):
            return None

    class FakeHub:
        def __init__(self, *args, **kwargs):
            self.peer_stt = kwargs.get("peer_stt")

        async def start(self, *args, **kwargs):
            return None

        async def stop(self):
            return None

    async def fake_run_audio_vad_loop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(headless_mic, "default_vad_model_path", lambda: vad_path)
    monkeypatch.setattr(headless_mic, "ensure_silero_vad_onnx", lambda target_path: vad_path)
    monkeypatch.setattr(headless_mic, "create_secret_store", lambda *_a, **_k: "secrets")
    monkeypatch.setattr(headless_mic, "create_llm_provider", lambda *_a, **_k: "llm")
    monkeypatch.setattr(headless_mic, "create_stt_backend", lambda *_a, **_k: "backend")
    monkeypatch.setattr(headless_mic, "ManagedSTTProvider", lambda *a, **k: object())
    monkeypatch.setattr(headless_mic, "VrchatOscUdpSender", lambda *a, **k: FakeSender())
    monkeypatch.setattr(headless_mic, "ChatboxPaginator", lambda *a, **k: object())
    monkeypatch.setattr(headless_mic, "ClientHub", FakeHub)
    monkeypatch.setattr(headless_mic, "SileroVadOnnx", lambda *a, **k: object())
    monkeypatch.setattr(headless_mic, "VadGating", lambda *a, **k: object())
    monkeypatch.setattr(headless_mic, "SoundDeviceAudioSource", source_factory)
    monkeypatch.setattr(headless_mic, "run_audio_vad_loop", fake_run_audio_vad_loop)
    monkeypatch.setattr(headless_mic, "resolve_sounddevice_input_device", resolve_device)


@pytest.mark.asyncio
async def test_headless_mic_runner_handles_keyboard_interrupt(monkeypatch, tmp_path) -> None:
    settings = AppSettings()
    settings.osc.vrc_mic_intercept = False
    config_path = tmp_path / "settings.json"
    vad_path = tmp_path / "vad.onnx"
    vad_path.write_text("dummy", encoding="utf-8")

    sender_ref: dict[str, object] = {}

    class FakeSender:
        def __init__(self, *args, **kwargs):
            sender_ref["instance"] = self
            self.closed = False

        def close(self):
            self.closed = True

    class FakeHub:
        def __init__(self, *args, **kwargs):
            return None

        async def start(self, *args, **kwargs):
            return None

        async def stop(self):
            return None

    class FakeSource:
        async def close(self):
            return None

    async def fake_run_audio_vad_loop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(headless_mic, "default_vad_model_path", lambda: vad_path)
    monkeypatch.setattr(headless_mic, "ensure_silero_vad_onnx", lambda target_path: vad_path)
    monkeypatch.setattr(headless_mic, "create_secret_store", lambda *_a, **_k: "secrets")
    monkeypatch.setattr(headless_mic, "create_llm_provider", lambda *_a, **_k: "llm")
    monkeypatch.setattr(headless_mic, "create_stt_backend", lambda *_a, **_k: "backend")
    monkeypatch.setattr(headless_mic, "ManagedSTTProvider", lambda *a, **k: object())
    monkeypatch.setattr(headless_mic, "VrchatOscUdpSender", FakeSender)
    monkeypatch.setattr(headless_mic, "ChatboxPaginator", lambda *a, **k: object())
    monkeypatch.setattr(headless_mic, "ClientHub", FakeHub)
    monkeypatch.setattr(headless_mic, "SileroVadOnnx", lambda *a, **k: object())
    monkeypatch.setattr(headless_mic, "VadGating", lambda *a, **k: object())
    monkeypatch.setattr(headless_mic, "SoundDeviceAudioSource", lambda *a, **k: FakeSource())
    monkeypatch.setattr(headless_mic, "run_audio_vad_loop", fake_run_audio_vad_loop)
    monkeypatch.setattr(headless_mic, "resolve_sounddevice_input_device", lambda *a, **k: None)

    runner = headless_mic.HeadlessMicRunner(
        settings=settings,
        config_path=config_path,
        vad_model_path=vad_path,
        use_llm=True,
    )
    result = await runner.run()

    assert result == 0
    assert sender_ref["instance"].closed is True


@pytest.mark.asyncio
async def test_headless_mic_runner_normalizes_wasapi_compatibility_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    settings = AppSettings()
    settings.audio.input_host_api = WINDOWS_WASAPI_COMPATIBILITY_HOST_API
    settings.audio.input_device = "Compat Mic"
    settings.osc.vrc_mic_intercept = False
    config_path = tmp_path / "settings.json"
    vad_path = tmp_path / "vad.onnx"
    vad_path.write_text("dummy", encoding="utf-8")
    resolve_calls: list[dict[str, object]] = []
    source_calls: list[dict[str, object]] = []

    class FakeSource:
        async def close(self):
            return None

    def fake_resolve(*, host_api: str, device: str) -> int:
        resolve_calls.append({"host_api": host_api, "device": device})
        return 9

    def fake_source(*_args, **kwargs) -> FakeSource:
        source_calls.append(dict(kwargs))
        return FakeSource()

    _patch_headless_mic_startup(
        monkeypatch,
        vad_path,
        resolve_device=fake_resolve,
        source_factory=fake_source,
    )

    runner = headless_mic.HeadlessMicRunner(
        settings=settings,
        config_path=config_path,
        vad_model_path=vad_path,
        use_llm=True,
    )
    result = await runner.run()

    assert result == 0
    assert resolve_calls == [{"host_api": WINDOWS_WASAPI_HOST_API, "device": "Compat Mic"}]
    assert source_calls[0]["device"] == 9
    assert source_calls[0].get("wasapi_auto_convert") is True
    assert source_calls[0].get("wasapi_exclusive") is False


@pytest.mark.asyncio
async def test_headless_mic_runner_does_not_apply_wasapi_flags_to_name_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    settings = AppSettings()
    settings.audio.input_host_api = WINDOWS_WASAPI_COMPATIBILITY_HOST_API
    settings.audio.input_device = "Compat Mic"
    settings.osc.vrc_mic_intercept = False
    config_path = tmp_path / "settings.json"
    vad_path = tmp_path / "vad.onnx"
    vad_path.write_text("dummy", encoding="utf-8")
    resolve_calls: list[dict[str, object]] = []
    source_calls: list[dict[str, object]] = []

    class FakeSource:
        async def close(self):
            return None

    def fake_resolve(*, host_api: str, device: str) -> int:
        resolve_calls.append({"host_api": host_api, "device": device})
        if host_api == WINDOWS_WASAPI_HOST_API:
            return 9
        if host_api == "":
            return 10
        return 99

    def fake_source(*_args, **kwargs) -> FakeSource:
        source_calls.append(dict(kwargs))
        if len(source_calls) == 1:
            raise RuntimeError("first open failed")
        return FakeSource()

    _patch_headless_mic_startup(
        monkeypatch,
        vad_path,
        resolve_device=fake_resolve,
        source_factory=fake_source,
    )

    runner = headless_mic.HeadlessMicRunner(
        settings=settings,
        config_path=config_path,
        vad_model_path=vad_path,
        use_llm=True,
    )
    result = await runner.run()

    assert result == 0
    assert resolve_calls == [
        {"host_api": WINDOWS_WASAPI_HOST_API, "device": "Compat Mic"},
        {"host_api": "", "device": "Compat Mic"},
    ]
    assert source_calls[0].get("wasapi_auto_convert") is True
    assert source_calls[1]["device"] == 10
    assert source_calls[1].get("wasapi_auto_convert") is False
    assert source_calls[1].get("wasapi_exclusive") is False


@pytest.mark.asyncio
async def test_headless_mic_runner_retries_same_device_name_fallback_without_wasapi_flags(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    settings = AppSettings()
    settings.audio.input_host_api = WINDOWS_WASAPI_COMPATIBILITY_HOST_API
    settings.audio.input_device = "Compat Mic"
    settings.osc.vrc_mic_intercept = False
    config_path = tmp_path / "settings.json"
    vad_path = tmp_path / "vad.onnx"
    vad_path.write_text("dummy", encoding="utf-8")
    resolve_calls: list[dict[str, object]] = []
    source_calls: list[dict[str, object]] = []

    class FakeSource:
        async def close(self):
            return None

    def fake_resolve(*, host_api: str, device: str) -> int:
        resolve_calls.append({"host_api": host_api, "device": device})
        return 9

    def fake_source(*_args, **kwargs) -> FakeSource:
        source_calls.append(dict(kwargs))
        if len(source_calls) == 1:
            raise RuntimeError("first open failed")
        return FakeSource()

    _patch_headless_mic_startup(
        monkeypatch,
        vad_path,
        resolve_device=fake_resolve,
        source_factory=fake_source,
    )

    runner = headless_mic.HeadlessMicRunner(
        settings=settings,
        config_path=config_path,
        vad_model_path=vad_path,
        use_llm=True,
    )
    result = await runner.run()

    assert result == 0
    assert resolve_calls == [
        {"host_api": WINDOWS_WASAPI_HOST_API, "device": "Compat Mic"},
        {"host_api": "", "device": "Compat Mic"},
    ]
    assert len(source_calls) == 2
    assert source_calls[0]["device"] == 9
    assert source_calls[0].get("wasapi_auto_convert") is True
    assert source_calls[0].get("wasapi_exclusive") is False
    assert source_calls[1]["device"] == 9
    assert source_calls[1].get("wasapi_auto_convert") is False
    assert source_calls[1].get("wasapi_exclusive") is False


@pytest.mark.asyncio
async def test_headless_mic_runner_does_not_apply_wasapi_flags_to_system_default_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    settings = AppSettings()
    settings.audio.input_host_api = WINDOWS_WASAPI_COMPATIBILITY_HOST_API
    settings.audio.input_device = ""
    settings.osc.vrc_mic_intercept = False
    config_path = tmp_path / "settings.json"
    vad_path = tmp_path / "vad.onnx"
    vad_path.write_text("dummy", encoding="utf-8")
    resolve_calls: list[dict[str, object]] = []
    source_calls: list[dict[str, object]] = []

    class FakeSource:
        async def close(self):
            return None

    def fake_resolve(*, host_api: str, device: str) -> int:
        resolve_calls.append({"host_api": host_api, "device": device})
        if host_api == WINDOWS_WASAPI_HOST_API:
            return 9
        return 99

    def fake_source(*_args, **kwargs) -> FakeSource:
        source_calls.append(dict(kwargs))
        if len(source_calls) == 1:
            raise RuntimeError("first open failed")
        return FakeSource()

    _patch_headless_mic_startup(
        monkeypatch,
        vad_path,
        resolve_device=fake_resolve,
        source_factory=fake_source,
    )

    runner = headless_mic.HeadlessMicRunner(
        settings=settings,
        config_path=config_path,
        vad_model_path=vad_path,
        use_llm=True,
    )
    result = await runner.run()

    assert result == 0
    assert resolve_calls == [{"host_api": WINDOWS_WASAPI_HOST_API, "device": ""}]
    assert source_calls[0].get("wasapi_auto_convert") is True
    assert source_calls[1]["device"] is None
    assert source_calls[1].get("wasapi_auto_convert") is False
    assert source_calls[1].get("wasapi_exclusive") is False


@pytest.mark.asyncio
async def test_headless_mic_runner_rejects_managed_openrouter_without_release_service(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    settings = AppSettings(
        provider=ProviderSettings(llm=LLMProviderName.OPENROUTER),
        openrouter=OpenRouterSettings(selected_source=OpenRouterCredentialSource.MANAGED),
    )
    config_path = tmp_path / "settings.json"
    vad_path = tmp_path / "vad.onnx"
    vad_path.write_text("dummy", encoding="utf-8")

    monkeypatch.setattr(
        headless_mic,
        "create_secret_store",
        lambda *_a, **_k: InMemorySecretStore(),
    )
    monkeypatch.setattr(
        headless_mic,
        "create_stt_backend",
        lambda *_a, **_k: pytest.fail("STT backend should not initialize"),
    )

    runner = headless_mic.HeadlessMicRunner(
        settings=settings,
        config_path=config_path,
        vad_model_path=vad_path,
        use_llm=True,
    )

    with pytest.raises(
        headless_mic.HeadlessMicInitializationError, match="managed release service"
    ):
        await runner.run()


@pytest.mark.asyncio
async def test_headless_mic_runner_starts_and_stops_vrc_receiver_when_enabled(
    monkeypatch, tmp_path
) -> None:
    settings = AppSettings()
    settings.osc.vrc_mic_intercept = True
    config_path = tmp_path / "settings.json"
    vad_path = tmp_path / "vad.onnx"
    vad_path.write_text("dummy", encoding="utf-8")

    receiver_events: list[str] = []
    run_kwargs: dict[str, object] = {}

    class FakeReceiver:
        def __init__(self, *args, **kwargs):
            _ = (args, kwargs)

        async def start(self):
            receiver_events.append("start")

        def stop(self):
            receiver_events.append("stop")

    class FakeSender:
        def close(self):
            return None

    class FakeHub:
        def __init__(self, *args, **kwargs):
            return None

        async def start(self, *args, **kwargs):
            return None

        async def stop(self):
            return None

    class FakeSource:
        async def close(self):
            return None

    async def fake_run_audio_vad_loop(*_args, **_kwargs):
        run_kwargs.update(_kwargs)
        return None

    monkeypatch.setattr(headless_mic, "default_vad_model_path", lambda: vad_path)
    monkeypatch.setattr(headless_mic, "ensure_silero_vad_onnx", lambda target_path: vad_path)
    monkeypatch.setattr(headless_mic, "create_secret_store", lambda *_a, **_k: "secrets")
    monkeypatch.setattr(headless_mic, "create_llm_provider", lambda *_a, **_k: "llm")
    monkeypatch.setattr(headless_mic, "create_stt_backend", lambda *_a, **_k: "backend")
    monkeypatch.setattr(headless_mic, "ManagedSTTProvider", lambda *a, **k: object())
    monkeypatch.setattr(headless_mic, "VrchatOscUdpSender", lambda *a, **k: FakeSender())
    monkeypatch.setattr(headless_mic, "ChatboxPaginator", lambda *a, **k: object())
    monkeypatch.setattr(headless_mic, "ClientHub", FakeHub)
    monkeypatch.setattr(headless_mic, "SileroVadOnnx", lambda *a, **k: object())
    monkeypatch.setattr(headless_mic, "VadGating", lambda *a, **k: object())
    monkeypatch.setattr(headless_mic, "SoundDeviceAudioSource", lambda *a, **k: FakeSource())
    monkeypatch.setattr(headless_mic, "run_audio_vad_loop", fake_run_audio_vad_loop)
    monkeypatch.setattr(headless_mic, "resolve_sounddevice_input_device", lambda *a, **k: None)
    monkeypatch.setattr(headless_mic, "VrcOscReceiver", FakeReceiver)

    runner = headless_mic.HeadlessMicRunner(
        settings=settings,
        config_path=config_path,
        vad_model_path=vad_path,
        use_llm=True,
    )
    result = await runner.run()

    assert result == 0
    assert receiver_events == ["start", "stop"]
    assert run_kwargs["audio_gate"] is not None


@pytest.mark.asyncio
async def test_headless_mic_runner_continues_when_vrc_receiver_start_raises_oserror(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = AppSettings()
    settings.osc.vrc_mic_intercept = True
    config_path = tmp_path / "settings.json"
    vad_path = tmp_path / "vad.onnx"
    vad_path.write_text("dummy", encoding="utf-8")

    receiver_events: list[str] = []
    run_kwargs: dict[str, object] = {}

    class FakeReceiver:
        def __init__(self, *args, **kwargs):
            _ = (args, kwargs)

        async def start(self):
            receiver_events.append("start")
            raise OSError("busy")

        def stop(self):
            receiver_events.append("stop")

    class FakeSender:
        def close(self):
            return None

    class FakeHub:
        def __init__(self, *args, **kwargs):
            return None

        async def start(self, *args, **kwargs):
            return None

        async def stop(self):
            return None

    class FakeSource:
        async def close(self):
            return None

    async def fake_run_audio_vad_loop(*_args, **_kwargs):
        run_kwargs.update(_kwargs)
        return None

    monkeypatch.setattr(headless_mic, "default_vad_model_path", lambda: vad_path)
    monkeypatch.setattr(headless_mic, "ensure_silero_vad_onnx", lambda target_path: vad_path)
    monkeypatch.setattr(headless_mic, "create_secret_store", lambda *_a, **_k: "secrets")
    monkeypatch.setattr(headless_mic, "create_llm_provider", lambda *_a, **_k: "llm")
    monkeypatch.setattr(headless_mic, "create_stt_backend", lambda *_a, **_k: "backend")
    monkeypatch.setattr(headless_mic, "ManagedSTTProvider", lambda *a, **k: object())
    monkeypatch.setattr(headless_mic, "VrchatOscUdpSender", lambda *a, **k: FakeSender())
    monkeypatch.setattr(headless_mic, "ChatboxPaginator", lambda *a, **k: object())
    monkeypatch.setattr(headless_mic, "ClientHub", FakeHub)
    monkeypatch.setattr(headless_mic, "SileroVadOnnx", lambda *a, **k: object())
    monkeypatch.setattr(headless_mic, "VadGating", lambda *a, **k: object())
    monkeypatch.setattr(headless_mic, "SoundDeviceAudioSource", lambda *a, **k: FakeSource())
    monkeypatch.setattr(headless_mic, "run_audio_vad_loop", fake_run_audio_vad_loop)
    monkeypatch.setattr(headless_mic, "resolve_sounddevice_input_device", lambda *a, **k: None)
    monkeypatch.setattr(headless_mic, "VrcOscReceiver", FakeReceiver)

    runner = headless_mic.HeadlessMicRunner(
        settings=settings,
        config_path=config_path,
        vad_model_path=vad_path,
        use_llm=True,
    )

    with caplog.at_level(logging.WARNING, logger="puripuly_heart.app.headless_mic"):
        result = await runner.run()

    gate = run_kwargs["audio_gate"]
    assert result == 0
    assert receiver_events == ["start"]
    assert gate.enabled is True
    assert gate.receiver_active is False
    assert any("VRChat mic sync receiver unavailable" in message for message in caplog.messages)


@pytest.mark.asyncio
async def test_headless_mic_runner_starts_peer_desktop_loop_when_peer_translation_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    settings = AppSettings()
    settings.ui.peer_translation_enabled = True
    settings.ui.integrated_context_enabled = True
    settings.desktop_audio.output_device = "Headphones (Loopback)"
    config_path = tmp_path / "settings.json"
    vad_path = tmp_path / "vad.onnx"
    vad_path.write_text("dummy", encoding="utf-8")

    created_hub: dict[str, object] = {}
    run_calls: list[dict[str, object]] = []

    class FakeSender:
        def close(self):
            return None

    class FakeHub:
        def __init__(self, *args, **kwargs):
            created_hub.update(kwargs)
            self.peer_stt = kwargs.get("peer_stt")

        async def start(self, *args, **kwargs):
            return None

        async def stop(self):
            return None

    class FakeSource:
        async def close(self):
            return None

    class FakeDesktopSource(FakeSource):
        pass

    async def fake_run_audio_vad_loop(*_args, **kwargs):
        run_calls.append(kwargs)
        return None

    monkeypatch.setattr(headless_mic, "default_vad_model_path", lambda: vad_path)
    monkeypatch.setattr(headless_mic, "ensure_silero_vad_onnx", lambda target_path: vad_path)
    monkeypatch.setattr(headless_mic, "create_secret_store", lambda *_a, **_k: "secrets")
    monkeypatch.setattr(headless_mic, "create_llm_provider", lambda *_a, **_k: "llm")
    monkeypatch.setattr(headless_mic, "create_stt_backend", lambda *_a, **_k: "backend")
    monkeypatch.setattr(headless_mic, "create_peer_stt_backend", lambda *_a, **_k: "peer-backend")
    monkeypatch.setattr(headless_mic, "ManagedSTTProvider", lambda *a, **k: object())
    monkeypatch.setattr(headless_mic, "VrchatOscUdpSender", lambda *a, **k: FakeSender())
    monkeypatch.setattr(headless_mic, "ChatboxPaginator", lambda *a, **k: object())
    monkeypatch.setattr(headless_mic, "ClientHub", FakeHub)
    monkeypatch.setattr(headless_mic, "SileroVadOnnx", lambda *a, **k: object())
    monkeypatch.setattr(headless_mic, "VadGating", lambda *a, **k: object())
    monkeypatch.setattr(headless_mic, "SoundDeviceAudioSource", lambda *a, **k: FakeSource())
    monkeypatch.setattr(
        headless_mic,
        "DesktopLoopbackAudioSource",
        lambda *a, **k: FakeDesktopSource(),
    )
    monkeypatch.setattr(
        headless_mic,
        "DesktopPeerPipeline",
        lambda *a, **k: FakeDesktopSource(),
    )
    monkeypatch.setattr(headless_mic, "run_audio_vad_loop", fake_run_audio_vad_loop)
    monkeypatch.setattr(headless_mic, "resolve_sounddevice_input_device", lambda *a, **k: None)

    runner = headless_mic.HeadlessMicRunner(
        settings=settings,
        config_path=config_path,
        vad_model_path=vad_path,
        use_llm=True,
    )
    result = await runner.run()

    assert result == 0
    assert created_hub["peer_stt"] is not None
    assert created_hub["peer_translation_enabled"] is True
    assert created_hub["integrated_context_enabled"] is True
    assert len(run_calls) == 2
    assert {call["sink"].channel for call in run_calls} == {"self", "peer"}


@pytest.mark.asyncio
async def test_headless_mic_runner_isolates_peer_loop_runtime_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = AppSettings()
    settings.ui.peer_translation_enabled = True
    settings.desktop_audio.output_device = "Headphones (Loopback)"
    config_path = tmp_path / "settings.json"
    vad_path = tmp_path / "vad.onnx"
    vad_path.write_text("dummy", encoding="utf-8")

    class FakeSender:
        def close(self):
            return None

    class FakeHub:
        def __init__(self, *args, **kwargs):
            self.peer_stt = kwargs.get("peer_stt")

        async def start(self, *args, **kwargs):
            return None

        async def stop(self):
            return None

    class FakeSource:
        async def close(self):
            return None

    class FakeDesktopSource(FakeSource):
        pass

    async def fake_run_audio_vad_loop(*_args, **kwargs):
        if kwargs["sink"].channel == "peer":
            raise RuntimeError("peer loop boom")
        return None

    monkeypatch.setattr(headless_mic, "default_vad_model_path", lambda: vad_path)
    monkeypatch.setattr(headless_mic, "ensure_silero_vad_onnx", lambda target_path: vad_path)
    monkeypatch.setattr(headless_mic, "create_secret_store", lambda *_a, **_k: "secrets")
    monkeypatch.setattr(headless_mic, "create_llm_provider", lambda *_a, **_k: "llm")
    monkeypatch.setattr(headless_mic, "create_stt_backend", lambda *_a, **_k: "backend")
    monkeypatch.setattr(headless_mic, "create_peer_stt_backend", lambda *_a, **_k: "peer-backend")
    monkeypatch.setattr(headless_mic, "ManagedSTTProvider", lambda *a, **k: object())
    monkeypatch.setattr(headless_mic, "VrchatOscUdpSender", lambda *a, **k: FakeSender())
    monkeypatch.setattr(headless_mic, "ChatboxPaginator", lambda *a, **k: object())
    monkeypatch.setattr(headless_mic, "ClientHub", FakeHub)
    monkeypatch.setattr(headless_mic, "SileroVadOnnx", lambda *a, **k: object())
    monkeypatch.setattr(headless_mic, "VadGating", lambda *a, **k: object())
    monkeypatch.setattr(headless_mic, "SoundDeviceAudioSource", lambda *a, **k: FakeSource())
    monkeypatch.setattr(
        headless_mic,
        "DesktopLoopbackAudioSource",
        lambda *a, **k: FakeDesktopSource(),
    )
    monkeypatch.setattr(
        headless_mic,
        "DesktopPeerPipeline",
        lambda *a, **k: FakeDesktopSource(),
    )
    monkeypatch.setattr(headless_mic, "run_audio_vad_loop", fake_run_audio_vad_loop)
    monkeypatch.setattr(headless_mic, "resolve_sounddevice_input_device", lambda *a, **k: None)

    runner = headless_mic.HeadlessMicRunner(
        settings=settings,
        config_path=config_path,
        vad_model_path=vad_path,
        use_llm=True,
    )

    with caplog.at_level(logging.ERROR, logger="puripuly_heart.app.headless_mic"):
        result = await runner.run()

    assert result == 0
    assert any("Peer desktop loop failed" in message for message in caplog.messages)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("low_latency_mode", "expected_self_hangover_ms", "expected_hub_hangover_s"),
    [
        (True, DEFAULT_LOW_LATENCY_VAD_HANGOVER_MS, 0.5),
        (False, DEFAULT_STABLE_VAD_HANGOVER_MS, 1.0),
    ],
)
async def test_headless_mic_runner_uses_shared_peer_vad_policy_helper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    low_latency_mode: bool,
    expected_self_hangover_ms: int,
    expected_hub_hangover_s: float,
) -> None:
    settings = AppSettings()
    settings.stt.low_latency_mode = low_latency_mode
    settings.ui.peer_translation_enabled = True
    settings.desktop_audio.output_device = "Headphones (Loopback)"
    settings.desktop_audio.vad_speech_threshold = 0.72
    settings.desktop_audio.vad_hangover_ms = 950
    settings.desktop_audio.vad_pre_roll_ms = 420
    config_path = tmp_path / "settings.json"
    vad_path = tmp_path / "vad.onnx"
    vad_path.write_text("dummy", encoding="utf-8")

    helper_calls: list[dict[str, object]] = []
    self_vad_calls: list[dict[str, object]] = []
    hub_calls: list[dict[str, object]] = []
    engine = object()

    class FakeSender:
        def close(self):
            return None

    class FakeHub:
        def __init__(self, *args, **kwargs):
            hub_calls.append(dict(kwargs))
            self.peer_stt = kwargs.get("peer_stt")

        async def start(self, *args, **kwargs):
            return None

        async def stop(self):
            return None

    class FakeSource:
        async def close(self):
            return None

    class FakeDesktopSource(FakeSource):
        pass

    async def fake_run_audio_vad_loop(*_args, **_kwargs):
        return None

    def fake_create_peer_vad_gating(
        *, engine, sample_rate_hz, ring_buffer_ms, speech_threshold, hangover_ms
    ):
        helper_calls.append(
            {
                "engine": engine,
                "sample_rate_hz": sample_rate_hz,
                "ring_buffer_ms": ring_buffer_ms,
                "speech_threshold": speech_threshold,
                "hangover_ms": hangover_ms,
            }
        )
        return object()

    def fake_self_vad_gating(*_args, **kwargs):
        self_vad_calls.append(dict(kwargs))
        return object()

    monkeypatch.setattr(headless_mic, "default_vad_model_path", lambda: vad_path)
    monkeypatch.setattr(headless_mic, "ensure_silero_vad_onnx", lambda target_path: vad_path)
    monkeypatch.setattr(headless_mic, "create_secret_store", lambda *_a, **_k: "secrets")
    monkeypatch.setattr(headless_mic, "create_llm_provider", lambda *_a, **_k: "llm")
    monkeypatch.setattr(headless_mic, "create_stt_backend", lambda *_a, **_k: "backend")
    monkeypatch.setattr(headless_mic, "create_peer_stt_backend", lambda *_a, **_k: "peer-backend")
    monkeypatch.setattr(headless_mic, "ManagedSTTProvider", lambda *a, **k: object())
    monkeypatch.setattr(headless_mic, "VrchatOscUdpSender", lambda *a, **k: FakeSender())
    monkeypatch.setattr(headless_mic, "ChatboxPaginator", lambda *a, **k: object())
    monkeypatch.setattr(headless_mic, "ClientHub", FakeHub)
    monkeypatch.setattr(headless_mic, "SileroVadOnnx", lambda *a, **k: engine)
    monkeypatch.setattr(headless_mic, "VadGating", fake_self_vad_gating)
    monkeypatch.setattr(headless_mic, "create_peer_vad_gating", fake_create_peer_vad_gating)
    monkeypatch.setattr(headless_mic, "SoundDeviceAudioSource", lambda *a, **k: FakeSource())
    monkeypatch.setattr(
        headless_mic,
        "DesktopLoopbackAudioSource",
        lambda *a, **k: FakeDesktopSource(),
    )
    monkeypatch.setattr(
        headless_mic,
        "DesktopPeerPipeline",
        lambda *a, **k: FakeDesktopSource(),
    )
    monkeypatch.setattr(headless_mic, "run_audio_vad_loop", fake_run_audio_vad_loop)
    monkeypatch.setattr(headless_mic, "resolve_sounddevice_input_device", lambda *a, **k: None)

    runner = headless_mic.HeadlessMicRunner(
        settings=settings,
        config_path=config_path,
        vad_model_path=vad_path,
        use_llm=True,
    )
    result = await runner.run()

    assert result == 0
    assert self_vad_calls[0].get("max_segment_ms") is None
    assert self_vad_calls[0]["hangover_ms"] == expected_self_hangover_ms
    assert hub_calls[0]["hangover_s"] == expected_hub_hangover_s
    assert helper_calls == [
        {
            "engine": engine,
            "sample_rate_hz": 16000,
            "ring_buffer_ms": 420,
            "speech_threshold": 0.72,
            "hangover_ms": 950,
        }
    ]


@pytest.mark.asyncio
async def test_headless_runner_uses_selected_peer_provider_configuration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    settings = AppSettings()
    settings.ui.peer_translation_enabled = True
    settings.provider.peer_stt = STTProviderName.SONIOX
    settings.peer_soniox_stt.model = "peer-soniox"
    settings.peer_soniox_stt.endpoint = "wss://peer-soniox.example/realtime"
    settings.peer_soniox_stt.keepalive_interval_s = 8.0
    settings.peer_soniox_stt.trailing_silence_ms = 300
    calls: list[AppSettings] = []
    created_hub: dict[str, object] = {}
    peer_backend = object()
    config_path = tmp_path / "settings.json"
    vad_path = tmp_path / "vad.onnx"
    vad_path.write_text("dummy", encoding="utf-8")

    class FakeManagedSTTProvider:
        def __init__(self, *, backend, sample_rate_hz, channel=None, **kwargs):
            self.backend = backend
            self.sample_rate_hz = sample_rate_hz
            self.channel = channel
            self.kwargs = kwargs

    def fake_create_peer_stt_backend(settings: AppSettings, *, secrets):
        _ = secrets
        calls.append(settings)
        return peer_backend

    class FakeHub:
        def __init__(self, *args, **kwargs):
            created_hub.update(kwargs)
            self.peer_stt = kwargs.get("peer_stt")

        async def start(self, *args, **kwargs):
            return None

        async def stop(self):
            return None

    class FakeSender:
        def close(self):
            return None

    class FakeSource:
        async def close(self):
            return None

    async def fake_run_audio_vad_loop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(headless_mic, "create_peer_stt_backend", fake_create_peer_stt_backend)
    monkeypatch.setattr(headless_mic, "create_secret_store", lambda *_a, **_k: "secrets")
    monkeypatch.setattr(headless_mic, "create_llm_provider", lambda *_a, **_k: "llm")
    monkeypatch.setattr(headless_mic, "create_stt_backend", lambda *_a, **_k: "backend")
    monkeypatch.setattr(headless_mic, "ManagedSTTProvider", FakeManagedSTTProvider)
    monkeypatch.setattr(headless_mic, "VrchatOscUdpSender", lambda *a, **k: FakeSender())
    monkeypatch.setattr(headless_mic, "ClientHub", FakeHub)
    monkeypatch.setattr(headless_mic, "SileroVadOnnx", lambda *a, **k: object())
    monkeypatch.setattr(headless_mic, "VadGating", lambda *a, **k: object())
    monkeypatch.setattr(headless_mic, "SoundDeviceAudioSource", lambda *a, **k: FakeSource())
    monkeypatch.setattr(headless_mic, "DesktopLoopbackAudioSource", lambda *a, **k: FakeSource())
    monkeypatch.setattr(headless_mic, "DesktopPeerPipeline", lambda *a, **k: FakeSource())
    monkeypatch.setattr(headless_mic, "run_audio_vad_loop", fake_run_audio_vad_loop)
    monkeypatch.setattr(headless_mic, "resolve_sounddevice_input_device", lambda *a, **k: None)

    runner = headless_mic.HeadlessMicRunner(
        settings=settings,
        config_path=config_path,
        vad_model_path=vad_path,
        use_llm=False,
    )

    result = await runner.run()

    assert result == 0
    assert len(calls) == 1
    peer_settings = calls[0]
    assert peer_settings.provider.peer_stt == STTProviderName.SONIOX
    assert peer_settings.peer_soniox_stt.model == "peer-soniox"
    assert peer_settings.peer_soniox_stt.endpoint == "wss://peer-soniox.example/realtime"
    assert peer_settings.peer_soniox_stt.keepalive_interval_s == 8.0
    assert peer_settings.peer_soniox_stt.trailing_silence_ms == 300
    assert isinstance(created_hub["peer_stt"], FakeManagedSTTProvider)
    assert isinstance(created_hub["stt"], FakeManagedSTTProvider)
    assert created_hub["stt"].kwargs["stt_provider_name"] == settings.provider.stt
    assert created_hub["peer_stt"].backend is peer_backend
    assert created_hub["peer_stt"].channel == "peer"
    assert created_hub["peer_stt"].kwargs["stt_provider_name"] == STTProviderName.SONIOX
