from types import SimpleNamespace

from puripuly_heart.app.services.application_after_launch import ApplicationAfterLaunchOwner


async def test_prepare_starts_only_runtime_probes() -> None:
    scheduled = []
    gpu_calls = []

    async def preload_saved_device_discovery():
        gpu_calls.append(True)
        return ()

    owner = ApplicationAfterLaunchOwner(
        vrchat_presence=SimpleNamespace(schedule=lambda *, force=False: scheduled.append(force)),
        gpu=SimpleNamespace(preload_saved_device_discovery=preload_saved_device_discovery),
    )

    await owner.prepare()

    assert scheduled == [True]
    assert gpu_calls == [True]
