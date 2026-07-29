from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from puripuly_heart.app.services.application_runtime_shutdown import (
    compose_application_runtime_shutdown_callbacks,
)
from puripuly_heart.app.services.application_shutdown import (
    ApplicationShutdownCoordinator,
)
from puripuly_heart.app.wiring_runtime_composition import RuntimeCompositionComponents
from puripuly_heart.app.wiring_runtime_pipeline import (
    RuntimePipelineLauncher,
    RuntimePipelineResources,
)
from puripuly_heart.core.clock import SystemClock
from puripuly_heart.ui.controller import GuiController


@pytest.mark.asyncio
async def test_application_shutdown_retries_retained_pipeline_cleanup() -> None:
    class RetainedSelfCapture:
        close_calls = 1

        async def close(self) -> None:
            self.close_calls += 1

    retained = RetainedSelfCapture()
    resources = RuntimePipelineResources()
    resources.self_capture = retained
    launcher = RuntimePipelineLauncher(
        config_path=Path("settings.json"),
        clock=SystemClock(),
        runtime_logging=object(),
        managed_release=object(),
        managed_delegate_ready=lambda: None,
        local_asr_factory=lambda _secrets: object(),
        self_capture_factory=lambda _hub, _gate: object(),
        peer_capture_factory=lambda _hub: object(),
        previous_self_capture=lambda: None,
        component_sink=lambda _components: None,
        peer_application=lambda: object(),
        configure_vrc_mic=lambda **_kwargs: None,
        stt_failure_sink=lambda _message: None,
        cleanup_failure_sink=lambda _message, _exception: None,
    )
    launcher.failed_resources = resources
    controller = GuiController(
        page=SimpleNamespace(),
        app=SimpleNamespace(),
        config_path=Path("settings.json"),
    )
    controller.install_runtime_composition(
        RuntimeCompositionComponents(
            self_capture_owner=lambda: cast(Any, object()),
            provider_runtime=cast(Any, object()),
            managed_account=cast(Any, object()),
            provider_application=cast(Any, object()),
            pipeline_launcher=launcher,
        )
    )
    callback = next(
        callback
        for callback in compose_application_runtime_shutdown_callbacks(controller)
        if callback.owner_name == "RuntimePipelineLauncher"
    )

    snapshot = await ApplicationShutdownCoordinator((callback,)).shutdown()

    assert snapshot.state == "completed"
    assert retained.close_calls == 2
    assert launcher.failed_resources is None
    assert controller.runtime_composition.pipeline_launcher is launcher
