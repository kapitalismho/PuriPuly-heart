from pathlib import Path

import pytest

import puripuly_heart.composition.local_asr_production_evidence as composition_module
from puripuly_heart.config.settings import AppSettings


class FakeOwner:
    pass


class FakeHub:
    def __init__(self, owner: object) -> None:
        self.local_asr_provider_runtime = owner


@pytest.mark.asyncio
async def test_composition_is_page_free_and_delegates_the_complete_evidence_contract(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    events: list[object] = []
    settings = AppSettings()
    owner = FakeOwner()

    class FakeController:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)
            self.config_path = kwargs["config_path"]
            self.settings = None
            self.hub = FakeHub(owner)

        def _load_or_init_settings(self, path: Path) -> object:
            events.append(("load", path))
            return settings

        async def _init_pipeline(self) -> None:
            events.append(("initialize", self.settings))

        async def retry_gpu_activation(self) -> None:
            events.append("retry")

        async def stop(self) -> None:
            events.append("close")

    monkeypatch.setattr(composition_module, "GuiController", FakeController)
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

    application = composition_module.compose_local_asr_production_evidence(
        config_path=Path("settings.json"),
    )

    assert application.config_path == Path("settings.json")
    assert application.load_compatibility_settings() is settings
    await application.initialize(settings)
    assert application.hub.local_asr_provider_runtime is owner
    assert application.owner is owner
    assert application.composition_facts() == {
        "controller": "FakeController",
        "hub": "FakeHub",
        "factory": "LocalASRProviderRuntimeFactory",
        "owner": "FakeOwner",
    }
    assert application.build_self_provider_request(settings, warmup=True) == (
        "self-request",
        True,
    )
    assert application.build_peer_provider_request(settings, warmup=False) == (
        "peer-request",
        False,
    )
    await application.retry_gpu_activation()
    await application.close()

    assert captured["page"] is None
    assert captured["config_path"] == Path("settings.json")
    assert captured["app"].debug_ui_preview is False
    assert events == [
        ("load", Path("settings.json")),
        ("initialize", settings),
        ("self-request", settings, True),
        ("peer-config", settings),
        ("peer-request", "peer-config", settings.stt.gpu_device_id, False),
        "retry",
        "close",
    ]


@pytest.mark.asyncio
async def test_initialize_preserves_canonical_owner_failure_contract(monkeypatch) -> None:
    class FakeController:
        def __init__(self, **kwargs) -> None:
            self.config_path = kwargs["config_path"]
            self.settings = None
            self.hub = FakeHub(object())

        async def _init_pipeline(self) -> None:
            return None

    monkeypatch.setattr(composition_module, "GuiController", FakeController)
    application = composition_module.compose_local_asr_production_evidence(
        config_path=Path("settings.json"),
    )

    with pytest.raises(
        RuntimeError,
        match="production controller did not compose the canonical owner",
    ):
        await application.initialize(object())
