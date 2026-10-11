from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest

from puripuly_heart.app.services.speaker_application import (
    SpeakerApplicationOwner,
    SpeakerAssetSnapshot,
)
from puripuly_heart.composition.headless_application import compose_headless_application
from puripuly_heart.config.settings_vnext.schema import (
    AppSettingsVNext,
    SecretsIntent,
    SpeakerIntent,
)
from puripuly_heart.config.settings_vnext.serialization import from_dict, to_dict
from puripuly_heart.core.speaker import BackendReady, BackendUpdate, SpeakerRuntime


class Assets:
    def __init__(self, ready=True):
        self.ready = ready
        self.inspections = 0
        self.installs = []
        self.closed = False

    async def inspect(self):
        self.inspections += 1
        return SpeakerAssetSnapshot(
            self.ready, Path("model.gguf"), Path("runtime"), None if self.ready else "missing"
        )

    async def install(self, *, download_consent, source_path):
        self.installs.append((download_consent, source_path))
        return await self.inspect()

    async def cancel(self):
        pass

    async def close(self):
        self.closed = True


class Backend:
    def __init__(self):
        self.starts = 0
        self.stops = 0
        self.resets = []
        self.start_gate = None

    async def start(self):
        self.starts += 1
        if self.start_gate is not None:
            await self.start_gate.wait()
        return BackendReady(160, 160)

    async def reset(self, epoch):
        self.resets.append(epoch)

    async def feed(self, pcm, compact_start_sample):
        return BackendUpdate(compact_start_sample + len(pcm), compact_start_sample + len(pcm))

    async def pause(self):
        return BackendUpdate(0, 0)

    async def stop(self):
        self.stops += 1


def test_intent_has_no_profile_device_or_consent_default():
    intent = AppSettingsVNext().intent.speaker
    assert intent == SpeakerIntent()
    assert from_dict(to_dict(AppSettingsVNext())).intent.speaker == intent
    for invalid in (
        {"enabled": True},
        {"vulkan_device_id": "auto"},
        {"vulkan_device_id": "-1"},
        {"profile": "default"},
    ):
        with pytest.raises(ValueError):
            SpeakerIntent(**invalid)


@pytest.mark.asyncio
async def test_disabled_and_unavailable_never_activate_or_download():
    assets = Assets(ready=False)
    owner = SpeakerApplicationOwner(
        assets=assets, runtime_factory=lambda *_: pytest.fail("must not activate")
    )
    await owner.apply(SpeakerIntent())
    assert assets.inspections == 0
    result = await owner.apply(SpeakerIntent(enabled=True, profile="1.04", vulkan_device_id="0"))
    assert result["state"] == "unavailable"
    assert result["reason"] == "missing"
    assert assets.installs == []
    await owner.close()


@pytest.mark.asyncio
async def test_local_install_consent_is_independent_and_requires_disable():
    assets = Assets()
    backend = Backend()
    owner = SpeakerApplicationOwner(
        assets=assets, runtime_factory=lambda *_: SpeakerRuntime(backend)
    )
    await owner.apply(SpeakerIntent(download_consent=True))
    await owner.install(source_path=Path("verified-local.gguf"))
    assert assets.installs == [(True, Path("verified-local.gguf"))]
    await owner.apply(SpeakerIntent(enabled=True, profile="0.64", vulkan_device_id="0"))
    with pytest.raises(ValueError, match="disable"):
        await owner.install(source_path=Path("another.gguf"))
    owner.reset()
    await asyncio.sleep(0)
    assert backend.starts == 1
    await owner.apply(SpeakerIntent())
    assert backend.stops == 1
    assert owner.snapshot()["state"] == "disabled"
    await owner.close()


@pytest.mark.asyncio
async def test_disable_fences_startup_without_requiring_gpu_completion():
    backend = Backend()
    backend.start_gate = asyncio.Event()
    owner = SpeakerApplicationOwner(
        assets=Assets(), runtime_factory=lambda *_: SpeakerRuntime(backend)
    )
    startup = asyncio.create_task(
        owner.apply(SpeakerIntent(enabled=True, profile="1.04", vulkan_device_id="0"))
    )
    while backend.starts == 0:
        await asyncio.sleep(0)
    result = await asyncio.wait_for(owner.apply(SpeakerIntent()), 1)
    assert result["state"] == "disabled"
    await asyncio.gather(startup, return_exceptions=True)
    assert backend.stops >= 1
    await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["disable", "close"])
async def test_failed_native_cleanup_retains_debt_and_retries_without_reload(action):
    class RetryStopBackend(Backend):
        async def stop(self):
            self.stops += 1
            if self.stops == 1:
                raise RuntimeError("retryable native cleanup")

    backend = RetryStopBackend()
    assets = Assets()
    owner = SpeakerApplicationOwner(
        assets=assets, runtime_factory=lambda *_: SpeakerRuntime(backend)
    )
    await owner.apply(SpeakerIntent(enabled=True, profile="1.04", vulkan_device_id="0"))

    async def stop():
        if action == "close":
            await owner.close()
        else:
            await owner.apply(SpeakerIntent())

    with pytest.raises(RuntimeError, match="retryable native cleanup"):
        await stop()
    failed = owner.snapshot()
    assert failed["state"] == "cleanup_pending"
    assert failed["cleanup_pending"]
    assert failed["runtime"].state == "cleanup_pending"
    assert not failed["runtime"].ready
    assert failed["reason"] == "backend_stop_failed: retryable native cleanup"
    assert assets.closed is (action == "close")
    with pytest.raises(ValueError, match="disable"):
        await owner.install(source_path=Path("replacement.gguf"))
    await stop()
    recovered = owner.snapshot()
    assert recovered["state"] == ("closed" if action == "close" else "disabled")
    assert not recovered["cleanup_pending"]
    assert recovered["runtime"] is None
    assert backend.starts == 1
    assert backend.stops == 2
    await owner.close()


@pytest.mark.asyncio
async def test_headless_authenticated_owner_config_persists_and_missing_runtime_is_gray(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PURIPULY_HEART_SECRETS_PASSPHRASE", "isolated-speaker-test")
    from puripuly_heart.composition import application_runtime

    monkeypatch.setattr(application_runtime, "NativeSpeakerAssets", lambda: Assets(ready=False))
    settings = AppSettingsVNext()
    settings = replace(
        settings,
        intent=replace(
            settings.intent,
            osc=replace(settings.intent.osc, connection_mode="off"),
            secrets=SecretsIntent(backend="encrypted_file", encrypted_file_path="secrets.json"),
        ),
    )
    config = tmp_path / "settings.json"
    config.write_text(json.dumps(to_dict(settings)), encoding="utf-8")
    app = compose_headless_application(config)
    try:
        await app.start()
        control = app.control()
        control.bind_instance("speaker-test")
        initial = await control.query("speaker.status", {})
        assert initial["speaker"]["state"] == "disabled"
        assert initial["speaker"]["intent"]["profile"] is None
        bad = await control.submit(
            "speaker.configure", {"enabled": True}, request_id="speaker-invalid"
        )
        assert (await control.wait(bad["operation_id"], timeout=5))["status"] == "rejected"
        missing_consent = await control.submit(
            "speaker.install", {}, request_id="speaker-no-consent"
        )
        required = await control.wait(missing_consent["operation_id"], timeout=5)
        assert required["status"] == "action_required"
        assert required["action"] == "speaker_download_consent"
        good = await control.submit(
            "speaker.configure",
            {"profile": "0.64", "vulkan_device_id": "0", "download_consent": True},
            request_id="speaker-consent",
        )
        assert (await control.wait(good["operation_id"], timeout=5))["status"] == "applied"
        current = await control.query("settings.current", {})
        assert current["settings"]["intent"]["speaker"]["download_consent"] is True
        assert current["settings"]["intent"]["speaker"]["enabled"] is False
        install = await control.submit("speaker.install", {}, request_id="speaker-no-source")
        assert (await control.wait(install["operation_id"], timeout=5))["status"] == "applied"
        enable = await control.submit(
            "speaker.configure", {"enabled": True}, request_id="speaker-enable"
        )
        assert (await control.wait(enable["operation_id"], timeout=5))["status"] == "degraded"
        assert (await control.query("speaker.status", {}))["speaker"]["state"] == "unavailable"
        disable = await control.submit(
            "speaker.configure", {"enabled": False}, request_id="speaker-disable"
        )
        assert (await control.wait(disable["operation_id"], timeout=5))["status"] == "applied"
    finally:
        await app.stop()
    restored = from_dict(json.loads(config.read_text(encoding="utf-8")))
    assert restored.intent.speaker == SpeakerIntent(
        profile="0.64", vulkan_device_id="0", download_consent=True
    )
