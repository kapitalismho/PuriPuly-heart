from __future__ import annotations

import asyncio
from typing import cast

import pytest
from puripuly_heart.app.services.overlay_application import (
    OverlayApplicationOwner,
    OverlayApplicationState,
)

from puripuly_heart.config.overlay_calibration import OverlayCalibration
from puripuly_heart.config.resolved import ResolvedOverlayConfig
from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.orchestrator.configuration import (
    TranslationRuntimeConfig,
    TranslationRuntimeConfigurationOwner,
)
from puripuly_heart.core.orchestrator.translation_diagnostics import (
    TranslationLatencyDiagnosticsOwner,
)
from puripuly_heart.core.runtime.overlay import OverlayRuntimeHandle


class OverlayDiagnostics:
    def record_translation(self, event: str, **fields: object) -> None:
        _ = event, fields


class OutputProjection:
    def __init__(self, overlay_sink: object | None = None) -> None:
        self.overlay_sink = overlay_sink
        self.reset_overlay_preview_calls = 0

    async def replace_overlay_sink(
        self,
        overlay_sink: object | None,
        *,
        expected_current: object | None = None,
        require_match: bool = False,
    ) -> bool:
        if require_match and self.overlay_sink is not expected_current:
            return False
        self.overlay_sink = overlay_sink
        return True

    async def reset_overlay_preview(self) -> None:
        self.reset_overlay_preview_calls += 1


class FailingOutputProjection(OutputProjection):
    async def replace_overlay_sink(
        self,
        overlay_sink: object | None,
        *,
        expected_current: object | None = None,
        require_match: bool = False,
    ) -> bool:
        _ = overlay_sink, expected_current, require_match
        raise RuntimeError("output detach failed")


class Presenter:
    def __init__(self) -> None:
        self.clear_calls = 0
        self.detach_calls = 0
        self.reset_calls = 0

    async def broadcast_shutdown(self) -> None:
        return None

    async def clear_for_runtime_detach(self) -> None:
        self.clear_calls += 1

    def detach_bridge(self) -> None:
        self.detach_calls += 1

    def reset_scene(self) -> None:
        self.reset_calls += 1


class Manager:
    def __init__(self) -> None:
        self.stop_calls = 0
        self.shutdown_calls = 0

    def mark_shutdown_requested(self) -> None:
        self.shutdown_calls += 1

    async def stop(self) -> None:
        self.stop_calls += 1


class Bridge:
    def __init__(self) -> None:
        self.stop_calls = 0

    async def stop(self) -> None:
        self.stop_calls += 1


class FailingDiagnosticsPort:
    def __init__(self) -> None:
        self.calls: list[object | None] = []

    def replace_overlay_diagnostics(
        self,
        diagnostics: object | None,
        *,
        expected_current: object | None = None,
        require_match: bool = False,
    ) -> bool:
        _ = diagnostics, require_match
        self.calls.append(expected_current)
        raise RuntimeError("diagnostics detach failed")


async def _noop_async() -> None:
    return None


async def _noop_renderer(
    queue: asyncio.Queue[dict[str, object]],
    overlay_instance_id: str,
) -> None:
    _ = queue, overlay_instance_id


def make_application(
    diagnostics_port: object,
    *,
    output_projection: OutputProjection | None = None,
    detailed_logs: list[tuple[str, int, Exception | None]] | None = None,
) -> OverlayApplicationOwner:
    return OverlayApplicationOwner(
        state_provider=lambda: OverlayApplicationState(
            settings_available=False,
            overlay_intent_enabled=False,
            configured_target="steamvr",
            locale="en",
        ),
        config_provider=lambda: cast(ResolvedOverlayConfig, object()),
        overlay_intent_sink=lambda _enabled: None,
        output_provider=lambda: output_projection,
        diagnostics_provider=lambda: cast(object, diagnostics_port),
        peer_snapshot_provider=lambda: cast(object, object()),
        disable_peer_intent=lambda: None,
        sync_peer_effective=lambda: None,
        cancel_peer_activation=lambda: None,
        refresh_peer_dependencies=_noop_async,
        presentation_sink=lambda _state: None,
        state_sink=lambda _state, _reason: None,
        fallback_notice_sink=lambda _active: None,
        cancel_bounds_persistence=_noop_async,
        clear_bounds_suppressed=lambda: None,
        calibration_provider=lambda: cast(OverlayCalibration, object()),
        logging_mode_provider=lambda: "basic",
        log_dir_provider=lambda: "",
        desktop_controls_factory=lambda _config: [],
        interaction_mode_sink=lambda _mode: None,
        bounds_control_sink=lambda _control: None,
        renderer_event_consumer=_noop_renderer,
        edit_interaction_mode="edit",
        clock=FakeClock(_now=0.0),
        log_basic=lambda _message, _level: None,
        log_detailed=lambda message, level, exception: (
            detailed_logs.append((message, level, exception))
            if detailed_logs is not None
            else False
        ),
    )


def make_diagnostics_owner() -> TranslationLatencyDiagnosticsOwner:
    configuration = TranslationRuntimeConfigurationOwner(TranslationRuntimeConfig())
    return TranslationLatencyDiagnosticsOwner(
        clock=FakeClock(_now=0.0),
        config_snapshot=configuration.snapshot,
    )


def test_application_attaches_and_identity_detaches_production_diagnostics_owner() -> None:
    diagnostics_owner = make_diagnostics_owner()
    application = make_application(diagnostics_owner)
    first = OverlayDiagnostics()
    second = OverlayDiagnostics()

    application.attach_translation_diagnostics(first)
    application.attach_translation_diagnostics(second)

    assert not application.detach_translation_diagnostics(first)
    assert diagnostics_owner.overlay_diagnostics is second
    assert application.detach_translation_diagnostics(second)
    assert diagnostics_owner.overlay_diagnostics is None


@pytest.mark.asyncio
async def test_stale_start_cleanup_does_not_detach_current_diagnostics() -> None:
    diagnostics_owner = make_diagnostics_owner()
    current = OverlayDiagnostics()
    stale = OverlayDiagnostics()
    output_projection = OutputProjection(overlay_sink=object())
    application = make_application(
        diagnostics_owner,
        output_projection=output_projection,
    )
    application.attach_translation_diagnostics(current)
    runtime = OverlayRuntimeHandle(shutdown_grace_s=0)
    runtime.attach_presenter(Presenter())
    runtime.attach_diagnostics(stale)

    await application.close_stale_start(runtime)

    assert diagnostics_owner.overlay_diagnostics is current


@pytest.mark.asyncio
async def test_stale_start_cleanup_contains_diagnostics_detach_failure() -> None:
    diagnostics_port = FailingDiagnosticsPort()
    detailed_logs: list[tuple[str, int, Exception | None]] = []
    stale = OverlayDiagnostics()
    output_projection = OutputProjection(overlay_sink=object())
    application = make_application(
        diagnostics_port,
        output_projection=output_projection,
        detailed_logs=detailed_logs,
    )
    runtime = OverlayRuntimeHandle(shutdown_grace_s=0)
    runtime.attach_presenter(Presenter())
    runtime.attach_diagnostics(stale)

    await application.close_stale_start(runtime)

    assert diagnostics_port.calls == [stale]
    assert len(detailed_logs) == 1
    message, _level, exception = detailed_logs[0]
    assert message == "[Overlay] Stale diagnostics detach reported failure"
    assert isinstance(exception, RuntimeError)


@pytest.mark.asyncio
async def test_stale_start_cleanup_contains_output_detach_failure() -> None:
    diagnostics_owner = make_diagnostics_owner()
    detailed_logs: list[tuple[str, int, Exception | None]] = []
    presenter = Presenter()
    output_projection = FailingOutputProjection(overlay_sink=presenter)
    application = make_application(
        diagnostics_owner,
        output_projection=output_projection,
        detailed_logs=detailed_logs,
    )
    runtime = OverlayRuntimeHandle(shutdown_grace_s=0)
    runtime.attach_presenter(presenter)

    await application.close_stale_start(runtime)

    assert output_projection.overlay_sink is presenter
    assert [message for message, _level, _exception in detailed_logs] == [
        "[Overlay] Stale overlay start cleanup reported failure",
        "[Overlay] Stale output ingress detach reported failure",
    ]
    assert all(isinstance(exception, RuntimeError) for _message, _level, exception in detailed_logs)


@pytest.mark.asyncio
async def test_diagnostics_detach_failure_does_not_block_overlay_resource_teardown() -> None:
    diagnostics_port = FailingDiagnosticsPort()
    presenter = Presenter()
    manager = Manager()
    bridge = Bridge()
    diagnostics = OverlayDiagnostics()
    output_projection = OutputProjection(overlay_sink=presenter)
    application = make_application(
        diagnostics_port,
        output_projection=output_projection,
    )
    runtime = OverlayRuntimeHandle(shutdown_grace_s=0)
    runtime.attach_presenter(presenter)
    runtime.attach_process_manager(manager)
    runtime.attach_bridge(bridge)
    runtime.attach_diagnostics(diagnostics)
    application.runtime = runtime

    succeeded = await application.teardown(preserve_presenter_state=False)

    assert not succeeded
    assert diagnostics_port.calls == [diagnostics]
    assert manager.stop_calls == 1
    assert bridge.stop_calls == 1
    assert presenter.clear_calls == 1
    assert presenter.detach_calls == 1
    assert presenter.reset_calls == 1
