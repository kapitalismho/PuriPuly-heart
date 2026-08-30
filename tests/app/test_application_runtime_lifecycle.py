from __future__ import annotations

import asyncio

import pytest

from puripuly_heart.app.ports.application_startup import ApplicationStartupState
from puripuly_heart.app.services.application_shutdown import (
    ApplicationShutdownContext,
    ApplicationShutdownDiagnostic,
    application_shutdown_callback,
)
from puripuly_heart.app.services.application_startup import ApplicationStartupOwner
from puripuly_heart.core.lifecycle import (
    LIFECYCLE_SHUTDOWN_PHASE_ORDER,
    SHUTDOWN_PHASE_CLOSE_LOGGING_DIAGNOSTICS,
    SHUTDOWN_PHASE_FREEZE_INGRESS,
    SHUTDOWN_PHASE_STOP_EXTERNAL_PRODUCERS,
)
from puripuly_heart.ui.app import TranslatorApp
from tests.helpers.ui_application import (
    ApplicationRuntimeShutdownStub,
    compose_test_ui_application_boundary,
)


class RecordingShutdownRuntime(ApplicationRuntimeShutdownStub):
    def __init__(self) -> None:
        super().__init__(object())
        self.events: list[str] = []

    def freeze_application_ingress(self) -> None:
        self.events.append("freeze")

    def stop_github_star_prompt_ingress(self) -> None:
        self.events.append("runtime-prompt-stop")

    async def release_manual_typing(self) -> None:
        self.events.append("manual")

    async def close_clipboard_runtime(self) -> None:
        self.events.append("clipboard")

    async def cancel_vrchat_osc_presence_probe(self) -> None:
        self.events.append("osc-presence")

    async def stop_self_capture_ingress(self) -> None:
        self.events.append("self-ingress")

    async def close_vrc_mic_receiver_runtime(self) -> None:
        self.events.append("vrc-mic")

    async def close_overlay_runtime(self) -> None:
        self.events.append("overlay-failed")
        raise RuntimeError("overlay close failed")

    async def close_peer_runtime(self) -> None:
        self.events.append("peer-after-failure")

    async def close_github_star_prompt_owner(self) -> None:
        self.events.append("runtime-prompt-close")

    async def close_openrouter_oauth_runtime(self) -> None:
        self.events.append("openrouter-oauth")

    async def close_local_asr_provisioning(self) -> None:
        self.events.append("local-asr")

    async def close_microphone_test_runtime(self) -> None:
        self.events.append("microphone-test")

    async def close_self_capture_owner(self) -> None:
        self.events.append("self-close")

    async def close_runtime_logging_background_tasks(self) -> None:
        self.events.append("logging-background")

    async def close_managed_auth_owner(self) -> None:
        self.events.append("managed-auth")

    async def close_translation_enable_owner(self) -> None:
        self.events.append("translation-enable")

    async def close_managed_usage_owner(self) -> None:
        self.events.append("managed-usage")

    async def close_runtime_pipeline_launcher(self) -> None:
        self.events.append("pipeline")

    async def close_peer_capture_owner(self) -> None:
        self.events.append("peer-close")

    async def close_self_translation_ingress(self) -> None:
        self.events.append("self-translation-ingress")

    async def close_peer_translation_ingress(self) -> None:
        self.events.append("peer-translation-ingress")

    async def close_translation_turns(self) -> None:
        self.events.append("translation-turns")

    async def close_output_runtime(self) -> None:
        self.events.append("output")

    async def close_self_channel_runtime(self) -> None:
        self.events.append("self-channel")

    async def close_peer_channel_runtime(self) -> None:
        self.events.append("peer-channel")

    async def close_local_asr_runtime(self) -> None:
        self.events.append("local-asr-runtime")

    async def close_llm_runtime(self) -> None:
        self.events.append("llm-runtime")

    async def close_managed_gemma_runtime(self) -> None:
        self.events.append("managed-gemma")

    def close_vrchat_sender(self) -> None:
        self.events.append("vrchat-sender")

    async def close_managed_openrouter_release_service(self) -> None:
        self.events.append("managed-release")

    def emit_final_application_shutdown_diagnostics(
        self,
        context: ApplicationShutdownContext,
    ) -> None:
        self.events.append(f"final:{len(context.failures)}")

    def close_runtime_logging(
        self,
        context: ApplicationShutdownContext,
    ) -> None:
        self.events.append(f"logging:{len(context.failures)}")

    def emit_application_shutdown_diagnostic(
        self,
        diagnostic: ApplicationShutdownDiagnostic,
    ) -> None:
        self.events.append(f"diagnostic:{diagnostic.owner_name}:{diagnostic.callback_name}")


def _position_of(events: list[str], marker: str) -> int:
    return events.index(marker)


@pytest.mark.asyncio
async def test_runtime_shutdown_graph_preserves_order_and_logging_last_after_failure() -> None:
    runtime = RecordingShutdownRuntime()
    boundary = compose_test_ui_application_boundary(
        object(),
        runtime_shutdown=runtime,
    )

    class PromptRuntime:
        def stop_ingress(self) -> None:
            runtime.events.append("boundary-prompt-stop")

        async def close(self) -> None:
            runtime.events.append("boundary-prompt-close")

    class AuthRuntime:
        async def close(self) -> None:
            runtime.events.append("boundary-auth-close")

    class FoundationRuntime:
        def application_shutdown_callbacks(self):
            return (
                application_shutdown_callback(
                    phase=SHUTDOWN_PHASE_STOP_EXTERNAL_PRODUCERS,
                    owner_name="FletFoundationRuntime",
                    callback_name="cancel_page_tasks",
                    callback=self.close,
                ),
            )

        async def close(self) -> None:
            runtime.events.append("foundation-page-tasks")

        def bind_application_lifecycle(self, lifecycle) -> None:
            self.lifecycle = lifecycle

    class ApplicationShim:
        """Duck-typed application boundary for lifecycle composition."""

        def __init__(self) -> None:
            self.application = boundary

    class UiHooks:
        def freeze_ui_ingress(self) -> None:
            runtime.events.append("ui-freeze")

        async def close_after_launch_ui_tasks(self) -> None:
            runtime.events.append("ui-after-launch")

        async def close_managed_auth_ui_tasks(self) -> None:
            runtime.events.append("ui-managed-auth")

    ui = UiHooks()
    boundary._github_star_prompt_runtime = PromptRuntime()
    boundary._managed_auth_runtime = AuthRuntime()
    foundation_runtime = FoundationRuntime()
    callbacks = (
        application_shutdown_callback(
            phase=SHUTDOWN_PHASE_FREEZE_INGRESS,
            owner_name="TranslatorApp",
            callback_name="freeze_ui_ingress",
            callback=ui.freeze_ui_ingress,
        ),
        *foundation_runtime.application_shutdown_callbacks(),
        application_shutdown_callback(
            phase=SHUTDOWN_PHASE_STOP_EXTERNAL_PRODUCERS,
            owner_name="TranslatorApp",
            callback_name="close_after_launch_tasks",
            callback=ui.close_after_launch_ui_tasks,
        ),
        application_shutdown_callback(
            phase=SHUTDOWN_PHASE_STOP_EXTERNAL_PRODUCERS,
            owner_name="TranslatorApp",
            callback_name="close_managed_auth_tasks",
            callback=ui.close_managed_auth_ui_tasks,
        ),
    )
    lifecycle = TranslatorApp.compose_application_lifecycle(
        ApplicationShim(),
        foundation_runtime,
        callbacks,
    )
    foundation_runtime.bind_application_lifecycle(lifecycle)

    with pytest.raises(RuntimeError, match="overlay close failed"):
        await lifecycle.shutdown()

    events = runtime.events

    def position(marker: str) -> int:
        return _position_of(events, marker)

    # Freeze semantics: UI ingress and runtime ingress freeze first, before
    # any producer stop or owner close.
    for marker in (
        "ui-freeze",
        "boundary-prompt-stop",
        "freeze",
        "runtime-prompt-stop",
    ):
        assert position(marker) < position("foundation-page-tasks")

    # External producers stop before owner drain, and boundary producers close
    # in the same phase.
    for marker in ("foundation-page-tasks", "ui-after-launch", "ui-managed-auth"):
        assert position(marker) < position("manual")
    for marker in ("boundary-prompt-close", "boundary-auth-close"):
        assert position(marker) < position("manual")

    # Owner drain and provider close proceed in phase order after producers.
    assert position("manual") < position("overlay-failed") < position("pipeline")

    # Overlay close failure surfaces as a diagnostic but does not stop later
    # closes: the peer owner still ran after the failure.
    assert position("diagnostic:OverlayApplicationOwner:close") < position("peer-after-failure")
    assert position("peer-after-failure") < position("managed-gemma")
    assert position("managed-gemma") < position("vrchat-sender")
    assert position("vrchat-sender") < position("managed-release")

    # Final diagnostics run before logging closes, and both record the single
    # failure; logging is the last event of the shutdown.
    assert position("final:1") < position("logging:1")
    assert events[-2:] == ["final:1", "logging:1"]

    snapshot = lifecycle.snapshot
    assert snapshot.failure_count == 1
    assert snapshot.terminal is True

    # The lifecycle phase order starts at ingress freeze and ends with logging.
    assert LIFECYCLE_SHUTDOWN_PHASE_ORDER[0] == SHUTDOWN_PHASE_FREEZE_INGRESS
    assert LIFECYCLE_SHUTDOWN_PHASE_ORDER[-1] == SHUTDOWN_PHASE_CLOSE_LOGGING_DIAGNOSTICS


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_type", [RuntimeError, asyncio.CancelledError])
async def test_startup_partial_allocation_failure_runs_boundary_cleanup(
    failure_type: type[BaseException],
) -> None:
    events: list[str] = []
    shutdown = ApplicationRuntimeShutdownStub(object())

    class Settings:
        async def prepare_startup_settings(self) -> ApplicationStartupState:
            events.append("settings")
            return ApplicationStartupState(
                settings=object(),
                fallback_channels=(),
                installation_fallback=False,
            )

    class Presentation:
        def apply_startup_presentation(self, state: ApplicationStartupState) -> None:
            _ = state
            events.append("presentation")

    class Runtime:
        async def launch_startup_runtime(self, state: ApplicationStartupState) -> None:
            _ = state
            events.append("runtime-allocated")
            raise failure_type("startup interrupted")

    class Events:
        async def start_application_events(self) -> None:
            events.append("events")

    class ApplicationRuntime:
        def __init__(self, startup: ApplicationStartupOwner) -> None:
            self._startup = startup

        async def start(self) -> None:
            await self._startup.start()

    class RecordingCleanup(ApplicationRuntimeShutdownStub):
        async def close_runtime_pipeline_launcher(self) -> None:
            events.append("cleanup-runtime")

        def close_runtime_logging(self, context: ApplicationShutdownContext) -> None:
            _ = context
            events.append("cleanup-logging")

    startup = ApplicationStartupOwner(
        settings=Settings(),
        presentation=Presentation(),
        runtime=Runtime(),
        events=Events(),
    )
    cleanup = RecordingCleanup(shutdown)
    boundary = compose_test_ui_application_boundary(
        ApplicationRuntime(startup),
        runtime_shutdown=cleanup,
    )

    with pytest.raises(failure_type, match="startup interrupted"):
        await boundary.start()

    assert events == [
        "settings",
        "presentation",
        "runtime-allocated",
        "cleanup-runtime",
        "cleanup-logging",
    ]
    assert boundary.application_lifecycle().is_terminal
