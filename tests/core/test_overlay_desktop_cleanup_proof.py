from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from puripuly_heart.core.overlay.process import OverlayProcessManager
from puripuly_heart.core.overlay.process_adapter import OverlayProcessEvent


class _FakeManagedProcess:
    def __init__(self) -> None:
        self._events: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        self._exit_future: asyncio.Future[int | None] = asyncio.get_running_loop().create_future()
        self.pid = 4321
        self.terminated = False

    @property
    def returncode(self) -> int | None:
        if not self._exit_future.done() or self._exit_future.cancelled():
            return None
        return self._exit_future.result()

    async def next_event(self) -> OverlayProcessEvent:
        return OverlayProcessEvent(await self._events.get(), "process_pipe")

    async def wait_for_exit(self) -> int | None:
        return await asyncio.shield(self._exit_future)

    async def finish_readers(self) -> None:
        return None

    async def terminate(self) -> None:
        self.terminated = True
        if not self._exit_future.done():
            self._exit_future.set_result(0)

    def drain_events(self) -> list[OverlayProcessEvent]:
        drained: list[OverlayProcessEvent] = []
        while True:
            try:
                drained.append(OverlayProcessEvent(self._events.get_nowait(), "process_pipe"))
            except asyncio.QueueEmpty:
                return drained

    def attach_diagnostics(self, diagnostics: object, *, overlay_instance_id: str) -> None:
        return None

    def attach_lifecycle_sink(self, sink: object | None) -> None:
        return None


class _FakeRunner:
    def __init__(self, process: _FakeManagedProcess) -> None:
        self._process = process

    def configure_runtime(self, *, quiet_tail_profile: str, handoff_experiment: str) -> None:
        _ = (quiet_tail_profile, handoff_experiment)

    def prepare(self, manifest: object) -> Path:
        return Path("overlay.exe")

    async def spawn(self, executable_path: Path, manifest_path: Path) -> _FakeManagedProcess:
        return self._process


def _starting_manager(
    tmp_path: Path, instance_id: str = "overlay-proof", target: str | None = "desktop"
) -> OverlayProcessManager:
    manager = OverlayProcessManager(
        overlay_instance_id=instance_id,
        selected_target=target,
        diagnostics_dir=tmp_path,
    )
    manager.state = "starting"
    manager._current_phase = "startup"  # noqa: SLF001
    return manager


@pytest.mark.asyncio
async def test_cleanup_proof_requires_ack_plus_outer_exit(tmp_path: Path) -> None:
    process = _FakeManagedProcess()
    process._events.put_nowait(
        {"type": "shutdown_complete", "overlay_instance_id": "overlay-proof"}
    )
    process._exit_future.set_result(0)
    manager = OverlayProcessManager(
        process_runner=_FakeRunner(process),  # type: ignore[arg-type]
        startup_timeout_ms=5000,
        overlay_instance_id="overlay-proof",
        selected_target="desktop",
        diagnostics_dir=tmp_path,
    )
    await manager.start()
    assert manager.state == "failed"
    assert manager._shutdown_acknowledged is True  # noqa: SLF001
    assert manager.desktop_cleanup_complete is True
    assert manager._process is None  # noqa: SLF001
    await manager.stop()
    assert manager.desktop_cleanup_complete is True


@pytest.mark.asyncio
async def test_outer_exit_without_ack_leaves_proof_false(tmp_path: Path) -> None:
    process = _FakeManagedProcess()
    process._exit_future.set_result(0)
    manager = OverlayProcessManager(
        process_runner=_FakeRunner(process),  # type: ignore[arg-type]
        startup_timeout_ms=5000,
        overlay_instance_id="overlay-proof",
        selected_target="desktop",
        diagnostics_dir=tmp_path,
    )
    await manager.start()
    assert manager.state == "failed"
    assert manager._shutdown_acknowledged is False  # noqa: SLF001
    assert manager.desktop_cleanup_complete is False
    assert manager._process is None  # noqa: SLF001


@pytest.mark.asyncio
async def test_foreign_instance_ack_is_rejected(tmp_path: Path) -> None:
    process = _FakeManagedProcess()
    process._events.put_nowait(
        {"type": "shutdown_complete", "overlay_instance_id": "overlay-other"}
    )
    process._exit_future.set_result(0)
    manager = OverlayProcessManager(
        process_runner=_FakeRunner(process),  # type: ignore[arg-type]
        startup_timeout_ms=5000,
        overlay_instance_id="overlay-proof",
        selected_target="desktop",
        diagnostics_dir=tmp_path,
    )
    await manager.start()
    assert manager._shutdown_acknowledged is False  # noqa: SLF001
    assert manager.desktop_cleanup_complete is False


@pytest.mark.asyncio
async def test_first_visible_latches_before_callback_without_ready(tmp_path: Path) -> None:
    fired: list[None] = []
    manager = OverlayProcessManager(
        overlay_instance_id="overlay-visible",
        selected_target="desktop",
        diagnostics_dir=tmp_path,
        first_visible_callback=lambda: fired.append(None),
    )
    manager.state = "starting"
    manager._current_phase = "startup"  # noqa: SLF001
    outcome = await manager._handle_lifecycle_event(  # noqa: SLF001
        {
            "type": "desktop_first_visible",
            "overlay_instance_id": "overlay-visible",
            "generation": 3,
        },
        allow_ready=True,
        trusted_process_event=True,
    )
    assert outcome == "ignored"
    assert manager.desktop_first_visible is True
    assert fired == [None]
    assert manager.state == "starting"
    assert manager.failure_reason is None
    assert manager._accepted_ready_generation is None  # noqa: SLF001


@pytest.mark.asyncio
async def test_first_visible_rejects_stale_duplicate_and_post_terminal(
    tmp_path: Path,
) -> None:
    manager = _starting_manager(tmp_path, instance_id="overlay-visible")
    fired: list[None] = []
    manager.first_visible_callback = lambda: fired.append(None)
    valid = {
        "type": "desktop_first_visible",
        "overlay_instance_id": "overlay-visible",
        "generation": 2,
    }
    cases: list[tuple[dict[str, object], bool]] = [
        (dict(valid, overlay_instance_id="overlay-other"), True),
        (dict(valid, generation=0), True),
        (dict(valid, generation="2"), True),
        ({k: v for k, v in valid.items() if k != "generation"}, True),
        (dict(valid), False),
    ]
    for event, trusted in cases:
        outcome = await manager._handle_lifecycle_event(  # noqa: SLF001
            event, allow_ready=True, trusted_process_event=trusted
        )
        assert outcome == "ignored"
    assert manager.desktop_first_visible is False
    assert fired == []
    outcome = await manager._handle_lifecycle_event(  # noqa: SLF001
        dict(valid), allow_ready=True, trusted_process_event=True
    )
    assert manager.desktop_first_visible is True
    assert fired == [None]
    outcome = await manager._handle_lifecycle_event(  # noqa: SLF001
        dict(valid), allow_ready=True, trusted_process_event=True
    )
    assert outcome == "ignored"
    assert fired == [None]
    manager.state = "failed"
    manager.desktop_first_visible = False
    manager._accepted_first_visible_generation = None  # noqa: SLF001
    outcome = await manager._handle_lifecycle_event(  # noqa: SLF001
        dict(valid), allow_ready=True, trusted_process_event=True
    )
    assert outcome == "ignored"
    assert manager.desktop_first_visible is False
    assert fired == [None]
