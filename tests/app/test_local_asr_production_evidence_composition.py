from pathlib import Path
from types import SimpleNamespace

import pytest

import puripuly_heart.composition.local_asr_production_evidence as composition_module
from puripuly_heart.config.settings import AppSettings


class FakeOwner:
    pass


class FakeLlmRuntime:
    pass


class FakeSelfVad:
    pass


class FakePeerVad:
    pass


@pytest.mark.asyncio
async def test_composition_delegates_the_evidence_specific_access_contract(
    monkeypatch,
) -> None:
    events: list[object] = []
    settings = AppSettings()
    owner = FakeOwner()
    start_callbacks = SimpleNamespace(
        start_output=lambda auto_flush: _record_async(
            events,
            ("start-output", auto_flush),
        ),
        open_self_ingress=lambda: _record_async(events, "open-self"),
        open_peer_ingress=lambda: _record_async(events, "open-peer"),
        start_translation_turns=lambda: _record_async(events, "start-turns"),
        start_local_asr=lambda: _record_async(events, "start-local-asr"),
    )
    access = SimpleNamespace(
        config_path=Path("settings.json"),
        load_compatibility_settings=lambda: (
            events.append(("load", Path("settings.json"))) or settings
        ),
        initialize=lambda value: _record_async(
            events,
            ("initialize", value),
        ),
        owner=owner,
        llm_runtime=FakeLlmRuntime(),
        translation_runtime_configuration=object(),
        self_vad=FakeSelfVad(),
        peer_vad=FakePeerVad(),
        channel_reset=object(),
        start_callbacks=start_callbacks,
        retry_gpu_activation=lambda: _record_async(events, "retry"),
    )

    class FakeApplication:
        async def stop(self) -> None:
            events.append("close")

    captured: dict[str, object] = {}

    def compose_runtime(**kwargs):
        captured.update(kwargs)
        kwargs["local_asr_evidence_sink"](access)
        return FakeApplication()

    monkeypatch.setattr(
        composition_module,
        "compose_application_runtime",
        compose_runtime,
    )
    monkeypatch.setattr(
        composition_module,
        "LocalASRProviderRuntimeOwner",
        FakeOwner,
    )
    monkeypatch.setattr(
        composition_module,
        "build_self_stt_provider_request",
        lambda current, warmup: (
            events.append(("self-request", current, warmup)) or ("self-request", warmup)
        ),
    )
    monkeypatch.setattr(
        composition_module,
        "build_peer_capture_session_config",
        lambda current: events.append(("peer-config", current)) or "peer-config",
    )
    monkeypatch.setattr(
        composition_module,
        "build_peer_stt_provider_request",
        lambda config, gpu_device_id, warmup: (
            events.append(("peer-request", config, gpu_device_id, warmup))
            or ("peer-request", warmup)
        ),
    )

    evidence = composition_module.compose_local_asr_production_evidence(
        config_path=Path("settings.json"),
    )

    assert evidence.load_compatibility_settings() is settings
    await evidence.initialize(settings)
    assert evidence.owner is owner
    assert evidence.composition_facts() == {
        "application": "FakeApplication",
        "factory": "LocalASRProviderRuntimeFactory",
        "owner": "FakeOwner",
        "llm_owner": "FakeLlmRuntime",
        "self_vad": "FakeSelfVad",
        "peer_vad": "FakePeerVad",
    }
    await evidence.start_runtime()
    assert evidence.build_self_provider_request(settings, warmup=True) == (
        "self-request",
        True,
    )
    assert evidence.build_peer_provider_request(settings, warmup=False) == (
        "peer-request",
        False,
    )
    await evidence.retry_gpu_activation()
    await evidence.close()

    assert captured["config_path"] == Path("settings.json")
    assert captured["presentation"].debug_ui_preview is False
    assert events == [
        ("load", Path("settings.json")),
        ("initialize", settings),
        ("start-output", False),
        "open-self",
        "open-peer",
        "start-turns",
        "start-local-asr",
        ("self-request", settings, True),
        ("peer-config", settings),
        ("peer-request", "peer-config", settings.stt.gpu_device_id, False),
        "retry",
        "close",
    ]


@pytest.mark.asyncio
async def test_initialize_preserves_canonical_owner_failure_contract(
    monkeypatch,
) -> None:
    access = SimpleNamespace(
        config_path=Path("settings.json"),
        load_compatibility_settings=lambda: object(),
        initialize=lambda _value: _record_async([], None),
        owner=object(),
        retry_gpu_activation=lambda: _record_async([], None),
    )

    def compose_runtime(**kwargs):
        kwargs["local_asr_evidence_sink"](access)
        return SimpleNamespace(stop=lambda: _record_async([], None))

    monkeypatch.setattr(
        composition_module,
        "compose_application_runtime",
        compose_runtime,
    )
    evidence = composition_module.compose_local_asr_production_evidence(
        config_path=Path("settings.json"),
    )

    with pytest.raises(
        RuntimeError,
        match="production application did not compose the canonical owner",
    ):
        await evidence.initialize(object())


async def _record_async(events: list[object], value: object) -> None:
    events.append(value)
