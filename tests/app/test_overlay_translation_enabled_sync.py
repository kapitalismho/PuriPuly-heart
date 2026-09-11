from __future__ import annotations

import asyncio
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

import pytest

from puripuly_heart.app.services.overlay.overlay_application import (
    OverlayApplicationOwner,
    OverlayApplicationState,
)
from puripuly_heart.app.services.peer_application import PeerApplicationSnapshot
from puripuly_heart.app.services.translation_enable import (
    TranslationEnableOwner,
    TranslationEnableState,
)
from puripuly_heart.app.wiring import wiring_managed_account
from puripuly_heart.app.wiring.wiring_managed_account import ManagedTranslationRuntimeAccess
from puripuly_heart.app.wiring.wiring_translation_runtime_configuration import (
    replace_translation_runtime_enabled,
)
from puripuly_heart.composition.application_runtime import compose_application_runtime
from puripuly_heart.composition.application_settings import load_application_settings
from puripuly_heart.config.overlay_calibration import OverlayCalibration
from puripuly_heart.config.resolved import ResolvedOverlayConfig
from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.orchestrator.configuration import (
    TranslationRuntimeConfig,
    TranslationRuntimeConfigurationOwner,
)
from puripuly_heart.core.overlay.manifest import (
    OVERLAY_EXECUTION_CONTRACT,
    OVERLAY_NATIVE_RETRY_CONTRACT,
)
from puripuly_heart.core.overlay.presenter import OverlayPresenter
from puripuly_heart.core.overlay.sink import OverlayEventAdapter
from puripuly_heart.core.runtime.provider_handle import ProviderRuntimeHandle
from puripuly_heart.domain.models import Transcript
from puripuly_heart.ui.presentation_adapter import FletUiPresentationAdapter


async def _noop_async() -> None:
    return None


async def _noop_renderer(queue: Any, overlay_instance_id: str) -> None:
    _ = queue, overlay_instance_id


class RecordingBridge:
    def __init__(self) -> None:
        self.snapshots: list[object] = []

    async def replace_snapshot(
        self, snapshot: object, *, block_expirations: object | None = None
    ) -> None:
        _ = block_expirations
        self.snapshots.append(snapshot)

    async def broadcast_shutdown(self) -> None:
        return None


class _NullOutput:
    overlay_sink: object | None = None

    async def replace_overlay_sink(
        self,
        _sink: object | None,
        *,
        expected_current: object | None = None,
        require_match: bool = False,
    ) -> bool:
        _ = (expected_current, require_match)
        return False

    async def reset_overlay_preview(self) -> None:
        return None


def _overlay_config(target: str = "steamvr") -> ResolvedOverlayConfig:
    return ResolvedOverlayConfig(
        enabled=True,
        target=target,
        show_translation=True,
        show_peer_original=True,
        calibration={},
        desktop_overlay_options={},
    )


def _peer_snapshot() -> PeerApplicationSnapshot:
    return PeerApplicationSnapshot(
        intent_enabled=False,
        activation_requested=False,
        effective_enabled=False,
        desired_active=False,
        activation_generation=0,
        activation_starting=False,
        model_loading=False,
        process_warning_reason=None,
        runtime_signature=None,
        provider_signature=None,
    )


def make_owner(
    config_owner: TranslationRuntimeConfigurationOwner,
) -> OverlayApplicationOwner:
    return OverlayApplicationOwner(
        state_provider=lambda: OverlayApplicationState(
            settings_available=True,
            overlay_intent_enabled=True,
            configured_target="steamvr",
            locale="en",
        ),
        config_provider=lambda: _overlay_config(),
        overlay_intent_sink=lambda _enabled: None,
        output_provider=lambda: cast(Any, _NullOutput()),
        diagnostics_provider=lambda: None,
        peer_snapshot_provider=_peer_snapshot,
        disable_peer_intent=lambda: None,
        sync_peer_effective=lambda: None,
        cancel_peer_activation=lambda: None,
        refresh_peer_dependencies=_noop_async,
        presentation_sink=lambda _state: None,
        state_sink=lambda _state, _reason: None,
        fallback_notice_sink=lambda _active: None,
        cancel_bounds_persistence=_noop_async,
        clear_bounds_suppressed=lambda: None,
        calibration_provider=lambda: cast(OverlayCalibration, OverlayCalibration()),
        logging_mode_provider=lambda: "basic",
        log_dir_provider=lambda: "",
        desktop_controls_factory=lambda _config: [],
        interaction_mode_sink=lambda _mode: None,
        bounds_control_sink=lambda _control: None,
        renderer_event_consumer=_noop_renderer,
        edit_interaction_mode="edit",
        clock=FakeClock(_now=0.0),
        log_basic=lambda _message, _level: None,
        log_detailed=lambda _message, _level, _exception: False,
        translation_enabled_provider=(lambda: config_owner.snapshot().value.translation_enabled),
    )


def attach_live_presenter(
    owner: OverlayApplicationOwner,
    config_owner: TranslationRuntimeConfigurationOwner,
) -> tuple[Any, OverlayPresenter, RecordingBridge]:
    bridge = RecordingBridge()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=FakeClock(_now=10.0),
        translation_enabled=config_owner.snapshot().value.translation_enabled,
        peer_presentation_refresh_burst=False,
        self_presentation_refresh_burst=False,
    )
    runtime = owner.new_runtime()
    runtime.set_overlay_instance_id("overlay-test")
    runtime.adopt_presenter(cast(Any, presenter))
    presenter.attach_bridge(cast(Any, bridge))
    owner.state = "connected"
    return runtime, presenter, bridge


def make_enable_owner(
    config_owner: TranslationRuntimeConfigurationOwner,
    overlay_owner: OverlayApplicationOwner,
    llm_available: bool,
) -> tuple[TranslationEnableOwner, list[bool]]:
    dashboard_values: list[bool] = []
    state_box = {"llm_available": llm_available}

    def state_provider() -> TranslationEnableState:
        return TranslationEnableState(
            runtime_available=True,
            translation_enabled=config_owner.snapshot().value.translation_enabled,
            llm_available=state_box["llm_available"],
            settings_available=True,
            provider_name="gemini",
            qwen_region=None,
            managed_selected=False,
            managed_china=False,
            managed_local_key_available=False,
            managed_release_service_available=False,
            ingress_frozen=False,
        )

    access = ManagedTranslationRuntimeAccess(
        llm_runtime_provider=lambda: None,
        context_provider=lambda: None,
        translation_runtime_configuration_provider=lambda: config_owner,
        rebuild_llm=_noop_async,
    )

    def runtime_sink(enabled: bool) -> None:
        wiring_managed_account._set_runtime_enabled(
            access,
            enabled,
            overlay_owner.notify_translation_runtime_state_changed,
        )

    async def warmup() -> None:
        return None

    async def teardown() -> None:
        return None

    owner = TranslationEnableOwner(
        state_provider=state_provider,
        managed_prepare=_unexpected_prepare,
        founder_route=_unexpected_founder_route,
        pending_sink=lambda _pending: None,
        runtime_ensurer=_unexpected_ensurer,
        usage_refresh_sink=lambda: None,
        usage_refresh_now=_noop_async,
        runtime_sink=runtime_sink,
        dashboard_sink=dashboard_values.append,
        clear_context=lambda: None,
        warmup=warmup,
        message_sink=lambda _key, _values: None,
        qq_dialog_sink=lambda: None,
        result_sink=lambda _result: None,
        log_basic=lambda _message: None,
        log_detailed=lambda _message: None,
        log_error=lambda _message: None,
        founder_letter_sink=lambda: None,
        teardown=teardown,
    )
    return owner, dashboard_values


async def _unexpected_prepare() -> Any:
    raise AssertionError("managed prepare must not run when managed is not selected")


async def _unexpected_founder_route() -> bool:
    raise AssertionError("founder route must not run in these tests")


async def _unexpected_ensurer(_mode: str) -> bool:
    raise AssertionError("runtime ensurer must not run in these tests")


async def _drain() -> None:
    await asyncio.sleep(0)
    await asyncio.sleep(0)


@dataclass(slots=True)
class _TestManagedProcess:
    ready: bool = True
    overlay_instance_id: str = "overlay-test"
    failure_reason: str | None = None
    _events: asyncio.Queue[dict[str, object]] = dataclass_field(default_factory=asyncio.Queue)
    _exit_future: asyncio.Future[int | None] | None = dataclass_field(default=None, init=False)
    terminated: bool = dataclass_field(default=False, init=False)

    def __post_init__(self) -> None:
        loop = asyncio.get_running_loop()
        self._exit_future = loop.create_future()
        if self.failure_reason is not None:
            self._events.put_nowait(
                {"type": "startup_error", "failure_reason": self.failure_reason}
            )
        elif self.ready:
            self._events.put_nowait(
                {
                    "type": "overlay_ready",
                    "overlay_instance_id": self.overlay_instance_id,
                    "runtime_generation": 1,
                    "capabilities": {
                        "execution_contract": OVERLAY_EXECUTION_CONTRACT,
                        "native_presentation_retry": OVERLAY_NATIVE_RETRY_CONTRACT,
                    },
                }
            )

    async def next_event(self) -> dict[str, object]:
        return await self._events.get()

    async def wait(self) -> int | None:
        assert self._exit_future is not None
        return await asyncio.shield(self._exit_future)

    async def wait_for_exit(self) -> int | None:
        assert self._exit_future is not None
        return await asyncio.shield(self._exit_future)

    async def finish_readers(self) -> None:
        return None

    async def terminate(self) -> None:
        self.terminated = True
        if self._exit_future is not None and not self._exit_future.done():
            self._exit_future.set_result(0)

    @property
    def returncode(self) -> int | None:
        if self._exit_future is None or not self._exit_future.done():
            return None
        if self._exit_future.cancelled():
            return None
        return self._exit_future.result()


@dataclass(slots=True)
class _TestProcessRunner:
    failure_reason: str | None = None
    ready: bool = True
    last_process: _TestManagedProcess | None = dataclass_field(default=None, init=False)
    overlay_instance_id: str = "overlay-test"

    def prepare(self, manifest: object) -> Path:
        self.overlay_instance_id = str(getattr(manifest, "overlay_instance_id"))
        return Path("C:/fake/PuriPulyHeartOverlay.exe")

    async def spawn(self, executable_path: object, manifest_path: object) -> _TestManagedProcess:
        _ = (executable_path, manifest_path)
        self.last_process = _TestManagedProcess(
            ready=self.ready,
            failure_reason=self.failure_reason,
            overlay_instance_id=self.overlay_instance_id,
        )
        return self.last_process


@dataclass(slots=True)
class _GatedProcessRunner:
    entered: asyncio.Event = dataclass_field(default_factory=asyncio.Event)
    gate: asyncio.Event = dataclass_field(default_factory=asyncio.Event)
    last_process: _TestManagedProcess | None = dataclass_field(default=None, init=False)
    overlay_instance_id: str = "overlay-test"

    def prepare(self, manifest: object) -> Path:
        self.overlay_instance_id = str(getattr(manifest, "overlay_instance_id"))
        return Path("C:/fake/PuriPulyHeartOverlay.exe")

    async def spawn(self, executable_path: object, manifest_path: object) -> _TestManagedProcess:
        _ = (executable_path, manifest_path)
        self.entered.set()
        await self.gate.wait()
        self.last_process = _TestManagedProcess(
            ready=True,
            overlay_instance_id=self.overlay_instance_id,
        )
        return self.last_process


async def _start_via_begin_start(
    owner: OverlayApplicationOwner,
    runner: object,
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    monkeypatch.setattr(
        OverlayApplicationOwner,
        "process_runner",
        staticmethod(lambda _target, _task_factory: runner),
    )
    await owner.begin_start()
    runtime = owner._runtime
    assert runtime is not None
    assert runtime.start_task is not None
    await runtime.start_task
    return runtime


async def _restart_via_transition(
    owner: OverlayApplicationOwner,
    runner: object,
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    monkeypatch.setattr(
        OverlayApplicationOwner,
        "process_runner",
        staticmethod(lambda _target, _task_factory: runner),
    )
    owner.on_runtime_crashed()
    await _drain()
    await owner.begin_start()
    runtime = owner._runtime
    assert runtime is not None
    if runtime.start_task is not None:
        await runtime.start_task
    return runtime


async def test_disabled_startup_first_snapshot_is_source_primary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_owner = TranslationRuntimeConfigurationOwner(
        TranslationRuntimeConfig(translation_enabled=False)
    )
    owner = make_owner(config_owner)
    runtime = await _start_via_begin_start(owner, _TestProcessRunner(), monkeypatch)
    try:
        assert owner.state == "connected"
        presenter = cast(OverlayPresenter, runtime.presenter)
        assert presenter.translation_enabled is False
        bridge = runtime.bridge
        assert bridge is not None
        assert bridge.snapshot() == presenter.snapshot()
        adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
        live_id = uuid4()
        await presenter.emit(
            adapter.peer_active_update(
                text="peer live caption",
                utterance_id=live_id,
                occupant_key=f"peer:{live_id}",
                source_language="en",
                target_language="ko",
                created_at=10.0,
            )
        )
        live_block = presenter.snapshot().blocks[0]
        assert live_block.block_variant == "active_peer"
        assert live_block.primary_text == "peer live caption"
        assert live_block.secondary_text == ""
        assert live_block.secondary_enabled is False
        final_id = uuid4()
        await presenter.emit(
            adapter.transcript_final(
                Transcript(
                    utterance_id=final_id,
                    channel="peer",
                    text="peer finalized source",
                    is_final=True,
                    created_at=10.1,
                ),
                source_language="en",
                target_language="ko",
            )
        )
        blocks = {block.id: block for block in presenter.snapshot().blocks}
        final_block = blocks[f"peer:{final_id}"]
        assert final_block.block_variant == "finalized"
        assert final_block.primary_text == "peer finalized source"
        assert final_block.secondary_text == ""
        assert final_block.secondary_enabled is False
        assert bridge.snapshot() == presenter.snapshot()
    finally:
        await owner.teardown(preserve_presenter_state=False)


async def test_restart_reuse_rebuilds_visible_row_as_source_primary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_owner = TranslationRuntimeConfigurationOwner(
        TranslationRuntimeConfig(translation_enabled=True)
    )
    owner = make_owner(config_owner)
    first_runtime = await _start_via_begin_start(owner, _TestProcessRunner(), monkeypatch)
    try:
        assert owner.state == "connected"
        first_presenter = cast(OverlayPresenter, first_runtime.presenter)
        assert first_presenter.translation_enabled is True
        adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
        peer_turn_id = uuid4()
        await first_presenter.emit(
            adapter.transcript_final(
                Transcript(
                    utterance_id=peer_turn_id,
                    channel="peer",
                    text="peer source toggle",
                    is_final=True,
                    created_at=10.0,
                ),
                source_language="en",
                target_language="ko",
            )
        )
        await first_presenter.emit(
            adapter.translation_final(
                utterance_id=peer_turn_id,
                channel="peer",
                text="peer translation toggle",
                source_language="en",
                target_language="ko",
                applied_context_mode=None,
                created_at=10.1,
            )
        )
        before = first_presenter.snapshot().blocks[0]
        assert before.primary_text == "peer translation toggle"
        replace_translation_runtime_enabled(config_owner, False)
        second_runtime = await _restart_via_transition(owner, _TestProcessRunner(), monkeypatch)
        assert owner.state == "connected", owner.failure_reason
        reused = cast(OverlayPresenter, second_runtime.presenter)
        assert reused is first_presenter
        assert reused.translation_enabled is False
        second_bridge = second_runtime.bridge
        assert second_bridge is not None
        assert second_bridge.snapshot() == reused.snapshot()
        after = reused.snapshot().blocks[0]
        assert after.primary_text == "peer source toggle"
        assert after.secondary_text == ""
        assert after.secondary_enabled is False
    finally:
        await owner.teardown(preserve_presenter_state=False)


async def test_paused_start_applies_latest_translation_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_owner = TranslationRuntimeConfigurationOwner(
        TranslationRuntimeConfig(translation_enabled=True)
    )
    owner = make_owner(config_owner)
    enable_owner, _dashboard = make_enable_owner(config_owner, owner, llm_available=True)
    runner = _GatedProcessRunner()
    monkeypatch.setattr(
        OverlayApplicationOwner,
        "process_runner",
        staticmethod(lambda _target, _task_factory: runner),
    )
    await owner.begin_start()
    runtime = owner._runtime
    assert runtime is not None
    assert runtime.start_task is not None
    start_task = runtime.start_task
    try:
        await runner.entered.wait()
        assert await enable_owner.set_enabled(False) is False
        runner.gate.set()
        await start_task
        await _drain()
        assert owner.state == "connected"
        presenter = cast(OverlayPresenter, runtime.presenter)
        assert presenter.translation_enabled is False
        adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
        peer_turn_id = uuid4()
        await presenter.emit(
            adapter.transcript_final(
                Transcript(
                    utterance_id=peer_turn_id,
                    channel="peer",
                    text="peer source during paused start",
                    is_final=True,
                    created_at=10.0,
                ),
                source_language="en",
                target_language="ko",
            )
        )
        block = presenter.snapshot().blocks[0]
        assert block.primary_text == "peer source during paused start"
        assert block.secondary_enabled is False
    finally:
        if not start_task.done():
            start_task.cancel()
            await asyncio.gather(start_task, return_exceptions=True)
        await owner.teardown(preserve_presenter_state=False)


async def test_generation_failure_preserves_source_primary_truth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_owner = TranslationRuntimeConfigurationOwner(
        TranslationRuntimeConfig(translation_enabled=False)
    )
    owner = make_owner(config_owner)
    runtime = await _start_via_begin_start(
        owner, _TestProcessRunner(failure_reason="startup_timeout"), monkeypatch
    )
    try:
        assert owner.state == "failed"
        assert owner.failure_reason == "startup_timeout"
        presenter = cast(OverlayPresenter, runtime.presenter)
        assert presenter.translation_enabled is False
    finally:
        await owner.teardown(preserve_presenter_state=False)


async def test_composed_app_translation_disable_rebuilds_overlay_source_primary(
    tmp_path: Path,
) -> None:
    presentation = FletUiPresentationAdapter(SimpleNamespace(debug_ui_preview=False))
    app = compose_application_runtime(
        presentation=presentation,
        config_path=tmp_path / "settings.json",
    )
    load_application_settings(settings=app._settings.settings)
    pipeline = app._input_runtime.pipeline
    config_owner = TranslationRuntimeConfigurationOwner(
        TranslationRuntimeConfig(translation_enabled=True)
    )
    pipeline.translation_runtime_configuration = config_owner
    pipeline.llm_runtime = ProviderRuntimeHandle(name="test-llm", provider=object())
    overlay_owner = app._overlay.overlay
    runtime, presenter, bridge = attach_live_presenter(overlay_owner, config_owner)
    try:
        adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
        peer_turn_id = uuid4()
        await presenter.emit(
            adapter.transcript_final(
                Transcript(
                    utterance_id=peer_turn_id,
                    channel="peer",
                    text="peer source toggle",
                    is_final=True,
                    created_at=10.0,
                ),
                source_language="en",
                target_language="ko",
            )
        )
        await presenter.emit(
            adapter.translation_final(
                utterance_id=peer_turn_id,
                channel="peer",
                text="peer translation toggle",
                source_language="en",
                target_language="ko",
                applied_context_mode=None,
                created_at=10.1,
            )
        )
        assert presenter.snapshot().blocks[0].primary_text == "peer translation toggle"
        snapshots_before = len(bridge.snapshots)
        assert await app._input_runtime.translation.set_enabled(False) is False
        await _drain()
        assert config_owner.snapshot().value.translation_enabled is False
        off_block = presenter.snapshot().blocks[0]
        assert off_block.primary_text == "peer source toggle"
        assert off_block.secondary_text == ""
        assert off_block.secondary_enabled is False
        assert len(bridge.snapshots) == snapshots_before + 1
    finally:
        await runtime.close(
            preserve_presenter_state=False,
            overlay_sink_detach=None,
            preview_reset=None,
            diagnostics_detach=None,
            emit_shutdown=False,
        )


async def test_accepted_toggle_round_trip_rebuilds_without_new_speech() -> None:
    config_owner = TranslationRuntimeConfigurationOwner(
        TranslationRuntimeConfig(translation_enabled=True)
    )
    owner = make_owner(config_owner)
    _runtime, presenter, bridge = attach_live_presenter(owner, config_owner)
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    peer_turn_id = uuid4()
    await presenter.emit(
        adapter.transcript_final(
            Transcript(
                utterance_id=peer_turn_id,
                channel="peer",
                text="peer source toggle",
                is_final=True,
                created_at=10.0,
            ),
            source_language="en",
            target_language="ko",
        )
    )
    await presenter.emit(
        adapter.translation_final(
            utterance_id=peer_turn_id,
            channel="peer",
            text="peer translation toggle",
            source_language="en",
            target_language="ko",
            applied_context_mode=None,
            created_at=10.1,
        )
    )
    enable_owner, _dashboard = make_enable_owner(config_owner, owner, llm_available=True)
    snapshots_before = len(bridge.snapshots)
    assert await enable_owner.set_enabled(False) is False
    await _drain()
    off_block = presenter.snapshot().blocks[0]
    assert off_block.primary_text == "peer source toggle"
    assert off_block.secondary_text == ""
    assert off_block.secondary_enabled is False
    assert len(bridge.snapshots) == snapshots_before + 1
    assert await enable_owner.set_enabled(True) is True
    await _drain()
    on_block = presenter.snapshot().blocks[0]
    assert on_block.primary_text == "peer translation toggle"
    assert on_block.secondary_text == "peer source toggle"
    assert on_block.secondary_enabled is True


async def test_rejected_enable_keeps_source_primary_without_republish() -> None:
    config_owner = TranslationRuntimeConfigurationOwner(
        TranslationRuntimeConfig(translation_enabled=False)
    )
    owner = make_owner(config_owner)
    _runtime, presenter, bridge = attach_live_presenter(owner, config_owner)
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    peer_turn_id = uuid4()
    await presenter.emit(
        adapter.transcript_final(
            Transcript(
                utterance_id=peer_turn_id,
                channel="peer",
                text="peer finalized source",
                is_final=True,
                created_at=10.0,
            ),
            source_language="en",
            target_language="ko",
        )
    )
    assert presenter.snapshot().blocks[0].primary_text == "peer finalized source"
    enable_owner, _dashboard = make_enable_owner(config_owner, owner, llm_available=False)
    snapshots_before = len(bridge.snapshots)
    assert await enable_owner.set_enabled(True) is False
    assert config_owner.snapshot().value.translation_enabled is False
    await _drain()
    block = presenter.snapshot().blocks[0]
    assert block.primary_text == "peer finalized source"
    assert block.secondary_text == ""
    assert block.secondary_enabled is False
    assert len(bridge.snapshots) == snapshots_before


async def test_managed_exhaustion_rebuilds_visible_row_as_source_primary() -> None:
    config_owner = TranslationRuntimeConfigurationOwner(
        TranslationRuntimeConfig(translation_enabled=True)
    )
    owner = make_owner(config_owner)
    _runtime, presenter, bridge = attach_live_presenter(owner, config_owner)
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    peer_turn_id = uuid4()
    await presenter.emit(
        adapter.transcript_final(
            Transcript(
                utterance_id=peer_turn_id,
                channel="peer",
                text="peer source toggle",
                is_final=True,
                created_at=10.0,
            ),
            source_language="en",
            target_language="ko",
        )
    )
    await presenter.emit(
        adapter.translation_final(
            utterance_id=peer_turn_id,
            channel="peer",
            text="peer translation toggle",
            source_language="en",
            target_language="ko",
            applied_context_mode=None,
            created_at=10.1,
        )
    )
    enable_owner, _dashboard = make_enable_owner(config_owner, owner, llm_available=True)
    snapshots_before = len(bridge.snapshots)
    enable_owner.disable_for_managed_exhaustion(reopen_founder_letter=False)
    assert config_owner.snapshot().value.translation_enabled is False
    await _drain()
    block = presenter.snapshot().blocks[0]
    assert block.primary_text == "peer source toggle"
    assert block.secondary_text == ""
    assert block.secondary_enabled is False
    assert len(bridge.snapshots) == snapshots_before + 1


async def test_rapid_toggles_converge_to_latest_snapshot() -> None:
    config_owner = TranslationRuntimeConfigurationOwner(
        TranslationRuntimeConfig(translation_enabled=True)
    )
    owner = make_owner(config_owner)
    _runtime, presenter, _bridge = attach_live_presenter(owner, config_owner)
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    peer_turn_id = uuid4()
    await presenter.emit(
        adapter.transcript_final(
            Transcript(
                utterance_id=peer_turn_id,
                channel="peer",
                text="peer finalized source",
                is_final=True,
                created_at=10.0,
            ),
            source_language="en",
            target_language="ko",
        )
    )
    replace_translation_runtime_enabled(config_owner, False)
    owner.notify_translation_runtime_state_changed()
    replace_translation_runtime_enabled(config_owner, True)
    owner.notify_translation_runtime_state_changed()
    await _drain()
    assert presenter.translation_enabled is True
    block = presenter.snapshot().blocks[0]
    assert block.primary_text == ""
    assert block.secondary_text == "peer finalized source"
    assert block.secondary_enabled is True


async def test_stale_and_closed_updates_are_ignored() -> None:
    config_owner = TranslationRuntimeConfigurationOwner(
        TranslationRuntimeConfig(translation_enabled=True)
    )
    owner = make_owner(config_owner)
    runtime, old_presenter, _bridge = attach_live_presenter(owner, config_owner)
    replace_translation_runtime_enabled(config_owner, False)
    owner.notify_translation_runtime_state_changed()
    new_presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=FakeClock(_now=10.0),
        translation_enabled=True,
        peer_presentation_refresh_burst=False,
        self_presentation_refresh_burst=False,
    )
    runtime.adopt_presenter(cast(Any, new_presenter))
    runtime.set_overlay_instance_id("overlay-next")
    await _drain()
    assert old_presenter.translation_enabled is True
    assert new_presenter.translation_enabled is True
    await runtime.close(
        preserve_presenter_state=True,
        overlay_sink_detach=None,
        preview_reset=None,
        diagnostics_detach=None,
        emit_shutdown=False,
    )
    replace_translation_runtime_enabled(config_owner, False)
    owner.notify_translation_runtime_state_changed()
    await _drain()
    assert new_presenter.translation_enabled is True
