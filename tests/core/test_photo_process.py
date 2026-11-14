from __future__ import annotations

import asyncio

from puripuly_heart.core.photo.bridge import PhotoBridge
from puripuly_heart.core.photo.process import PhotoProcessSupervisor


class UnavailableRunner:
    def __init__(self):
        self.prepares = 0

    def prepare(self, manifest):
        self.prepares += 1
        raise FileNotFoundError("private synthetic path must not be logged")


async def test_missing_native_runtime_is_bounded_and_does_not_throw_into_application_start(
    tmp_path,
):
    runner = UnavailableRunner()
    diagnostics = []
    generations = []
    bridge = PhotoBridge(control=lambda value, generation: None)
    supervisor = PhotoProcessSupervisor(
        bridge=bridge,
        renderer_changed=generations.append,
        diagnostic=lambda stage, fields: diagnostics.append((stage, fields)),
        runner=runner,
        restart_limit=2,
        temporary_root=tmp_path,
    )
    await supervisor.start()
    await asyncio.wait_for(asyncio.shield(supervisor._task), timeout=2)
    assert runner.prepares == 3
    assert generations == [1, 2, 3]
    assert supervisor.failure_code == "photo_child_unavailable"
    assert supervisor.process is None
    assert all("private" not in str(item) for item in diagnostics)
    assert list(tmp_path.iterdir()) == []
    await supervisor.close()
    await supervisor.close()


async def test_shutdown_before_start_prevents_child_admission(tmp_path):
    runner = UnavailableRunner()
    bridge = PhotoBridge(control=lambda value, generation: None)
    supervisor = PhotoProcessSupervisor(
        bridge=bridge,
        renderer_changed=lambda generation: None,
        diagnostic=lambda stage, fields: None,
        runner=runner,
        temporary_root=tmp_path,
    )
    await supervisor.close()
    await supervisor.start()
    assert runner.prepares == 0
    assert supervisor.process is None
