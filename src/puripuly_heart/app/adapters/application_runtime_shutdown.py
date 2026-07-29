from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from puripuly_heart.app.services.application_ingress import ApplicationIngressGate
from puripuly_heart.app.services.application_runtime_logging import (
    ApplicationRuntimeLoggingOwner,
)
from puripuly_heart.app.services.application_shutdown import (
    ApplicationShutdownContext,
    ApplicationShutdownDiagnostic,
)
from puripuly_heart.app.services.clipboard_auto_translation import (
    ClipboardAutoTranslationOwner,
)
from puripuly_heart.app.services.github_star_prompt import GithubStarPromptOwner
from puripuly_heart.app.services.overlay_application import OverlayApplicationOwner
from puripuly_heart.app.services.peer_application import PeerApplicationOwner
from puripuly_heart.app.services.vrc_mic_sync import VrcMicSyncOwner
from puripuly_heart.app.wiring_managed_account import ManagedAccountComponents
from puripuly_heart.app.wiring_microphone_test import MicrophoneTestRuntime
from puripuly_heart.app.wiring_runtime_pipeline import (
    RuntimePipelineHandle,
    RuntimePipelineLauncher,
)
from puripuly_heart.core.runtime.vrchat_osc_presence import (
    VrchatOscPresenceProbeOwner,
)


@dataclass(slots=True)
class ApplicationRuntimeShutdownAdapter:
    ingress: ApplicationIngressGate
    pipeline: RuntimePipelineHandle
    runtime_logging: ApplicationRuntimeLoggingOwner
    managed: ManagedAccountComponents
    pipeline_launcher: RuntimePipelineLauncher
    stop_self_capture: Callable[[], Awaitable[None]]
    release_manual_typing_owner: Callable[[], Awaitable[None]]
    close_local_asr_provisioning_owner: Callable[[], Awaitable[None]]
    close_openrouter_oauth_owner: Callable[[], Awaitable[None]]
    clear_ui_event_runtime: Callable[[], None]
    peer: Callable[[], PeerApplicationOwner | None]
    overlay: Callable[[], OverlayApplicationOwner | None]
    vrchat_presence: Callable[[], VrchatOscPresenceProbeOwner | None]
    vrc_mic_sync: Callable[[], VrcMicSyncOwner | None]
    github_prompt: Callable[[], GithubStarPromptOwner | None]
    clipboard: Callable[[], ClipboardAutoTranslationOwner | None]
    microphone: Callable[[], MicrophoneTestRuntime | None]

    def freeze_application_ingress(self) -> None:
        self.ingress.freeze()
        self_capture = self.pipeline.self_capture
        if self_capture is not None:
            self_capture.invalidate_intent()
        peer = self.peer()
        if peer is not None:
            peer.stop_ingress()
        presence = self.vrchat_presence()
        if presence is not None:
            presence.stop_ingress()
        overlay = self.overlay()
        if overlay is not None:
            overlay.stop_ingress()
        vrc_mic_sync = self.vrc_mic_sync()
        if vrc_mic_sync is not None:
            vrc_mic_sync.stop_ingress()
        self.runtime_logging.stop_ingress()
        self.managed.usage.stop_ingress()
        self.managed.auth.stop_ingress()
        self.managed.translation.stop_ingress()

    def stop_github_star_prompt_ingress(self) -> None:
        owner = self.github_prompt()
        if owner is not None:
            owner.stop_ingress()

    async def release_manual_typing(self) -> None:
        await self.release_manual_typing_owner()

    async def close_clipboard_runtime(self) -> None:
        owner = self.clipboard()
        if owner is not None:
            await owner.close(strict_runtime_errors=True)

    async def cancel_vrchat_osc_presence_probe(self) -> None:
        owner = self.vrchat_presence()
        if owner is not None:
            await owner.cancel()

    async def stop_self_capture_ingress(self) -> None:
        await self.stop_self_capture()

    async def close_vrc_mic_receiver_runtime(self) -> None:
        owner = self.vrc_mic_sync()
        if owner is not None:
            await owner.close()

    async def close_overlay_runtime(self) -> None:
        owner = self.overlay()
        if owner is None:
            return
        owner.stop_ingress()
        await owner.shutdown(preserve_failure_reason=True)
        owner.clear_fallback()
        await owner.fallback_owner.close()

    async def close_peer_runtime(self) -> None:
        owner = self.peer()
        if owner is not None:
            await owner.close()

    async def close_github_star_prompt_owner(self) -> None:
        owner = self.github_prompt()
        if owner is not None:
            await owner.close()

    async def close_openrouter_oauth_runtime(self) -> None:
        await self.close_openrouter_oauth_owner()

    async def close_local_asr_provisioning(self) -> None:
        await self.close_local_asr_provisioning_owner()

    async def close_microphone_test_runtime(self) -> None:
        runtime = self.microphone()
        if runtime is not None:
            await runtime.close()

    async def close_self_capture_owner(self) -> None:
        owner = self.pipeline.self_capture
        if owner is None:
            return
        await owner.close()
        if self.pipeline.self_capture is owner:
            self.pipeline.self_capture = None

    async def close_runtime_logging_background_tasks(self) -> None:
        await self.runtime_logging.close_background_tasks()

    async def close_managed_auth_owner(self) -> None:
        await self.managed.auth.close()

    async def close_translation_enable_owner(self) -> None:
        await self.managed.translation.close()

    async def close_managed_usage_owner(self) -> None:
        await self.managed.usage.close()

    async def close_runtime_pipeline_launcher(self) -> None:
        await self.pipeline_launcher.close()

    async def stop_hub_owned_runtimes(self) -> None:
        hub = self.pipeline.hub
        if hub is None:
            return
        await hub.stop()
        if self.pipeline.hub is hub:
            self.pipeline.hub = None
            self.clear_ui_event_runtime()

    def close_vrchat_sender(self) -> None:
        sender = self.pipeline.sender
        if sender is not None:
            sender.close()
            if self.pipeline.sender is sender:
                self.pipeline.sender = None
        self.pipeline.osc = None

    async def close_managed_openrouter_release_service(self) -> None:
        await self.managed.release.close()

    def emit_final_application_shutdown_diagnostics(
        self,
        context: ApplicationShutdownContext,
    ) -> None:
        self.runtime_logging.emit_terminal_summary(context)

    def close_runtime_logging(self, context: ApplicationShutdownContext) -> None:
        self.runtime_logging.close_after_producers_stop(context)

    def emit_application_shutdown_diagnostic(
        self,
        diagnostic: ApplicationShutdownDiagnostic,
    ) -> None:
        self.runtime_logging.emit_shutdown_diagnostic(diagnostic)
