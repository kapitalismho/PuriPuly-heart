from __future__ import annotations

import inspect
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
from puripuly_heart.app.services.osc.control_runtime import OscControlIntegrationOwner
from puripuly_heart.app.services.overlay_application import OverlayApplicationOwner
from puripuly_heart.app.services.peer_application import PeerApplicationOwner
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
    vrc_mic_sync: Callable[[], OscControlIntegrationOwner | None]
    github_prompt: Callable[[], GithubStarPromptOwner | None]
    clipboard: Callable[[], ClipboardAutoTranslationOwner | None]
    microphone: Callable[[], MicrophoneTestRuntime | None]
    close_managed_gemma_owner: Callable[[], Awaitable[None]] | None = None

    def effective_osc_ports(self) -> tuple[int | None, int | None]:
        owner = self.vrc_mic_sync()
        if owner is None or owner.receiver is None:
            return (None, None)
        send_port = owner.effective_send_port
        receive_port = owner.effective_receive_port
        return (
            send_port if isinstance(send_port, int) and send_port > 0 else None,
            receive_port if isinstance(receive_port, int) and receive_port > 0 else None,
        )

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
        components = self.pipeline.current
        if components is None:
            return
        await self._invoke_pipeline_close(components.close_callbacks.close_self_capture)
        if components.resource_owner.self_capture is None:
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

    async def close_peer_capture_owner(self) -> None:
        components = self.pipeline.current
        if components is None:
            return
        await self._invoke_pipeline_close(components.close_callbacks.close_peer_capture)

    async def close_self_translation_ingress(self) -> None:
        components = self.pipeline.current
        if components is not None:
            await self._invoke_pipeline_close(components.close_callbacks.close_self_ingress)

    async def close_peer_translation_ingress(self) -> None:
        components = self.pipeline.current
        if components is not None:
            await self._invoke_pipeline_close(components.close_callbacks.close_peer_ingress)

    async def close_translation_turns(self) -> None:
        components = self.pipeline.current
        if components is None:
            return
        await self._invoke_pipeline_close(components.close_callbacks.close_translation_turns)
        if components.resource_owner.translation_turns is None:
            self.pipeline.translation_turns = None

    async def close_output_runtime(self) -> None:
        components = self.pipeline.current
        if components is None:
            return
        await self._invoke_pipeline_close(components.close_callbacks.close_output)
        if components.resource_owner.output_runtime is None:
            self.pipeline.output_runtime = None
            self.clear_ui_event_runtime()

    async def close_self_channel_runtime(self) -> None:
        components = self.pipeline.current
        if components is None:
            return
        await self._invoke_pipeline_close(components.close_callbacks.close_self_channel)
        if components.resource_owner.self_runtime is None:
            self.pipeline.self_runtime = None

    async def close_peer_channel_runtime(self) -> None:
        components = self.pipeline.current
        if components is None:
            return
        await self._invoke_pipeline_close(components.close_callbacks.close_peer_channel)
        if components.resource_owner.peer_runtime is None:
            self.pipeline.peer_runtime = None

    async def close_local_asr_runtime(self) -> None:
        components = self.pipeline.current
        if components is None:
            return
        await self._invoke_pipeline_close(components.close_callbacks.close_local_asr)
        if components.resource_owner.local_asr_runtime is None:
            self.pipeline.local_asr_runtime = None

    async def close_llm_runtime(self) -> None:
        components = self.pipeline.current
        if components is None:
            return
        await self._invoke_pipeline_close(components.close_callbacks.close_llm)
        if components.resource_owner.llm_runtime is None:
            self.pipeline.llm_runtime = None

    async def close_managed_gemma_runtime(self) -> None:
        if self.close_managed_gemma_owner is not None:
            await self.close_managed_gemma_owner()

    def close_vrchat_sender(self) -> None:
        components = self.pipeline.current
        if components is None:
            return
        result = components.close_callbacks.close_sender()
        if inspect.isawaitable(result):
            raise RuntimeError("VRChat sender close callback must be synchronous")
        if components.resource_owner.sender is None:
            self.pipeline.sender = None
            self.pipeline.osc = None
        if not components.resource_owner.has_resources:
            self.pipeline.clear(components)

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

    @staticmethod
    async def _invoke_pipeline_close(
        callback: Callable[[], Awaitable[None] | None],
    ) -> None:
        result = callback()
        if inspect.isawaitable(result):
            await result
