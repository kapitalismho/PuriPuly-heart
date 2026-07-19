from __future__ import annotations

import inspect

from puripuly_heart.core.local_asr_provider_runtime import (
    LocalASRProviderRuntimeCallbacks,
    LocalASRProviderRuntimeSnapshot,
    ProviderRuntimeBuildRequest,
    ProviderRuntimeChannel,
    ProviderRuntimeChannelSnapshot,
    ProviderRuntimeDiagnostic,
    ProviderRuntimeGpuSnapshot,
    ProviderRuntimeMutationResult,
    ProviderRuntimeReleaseMode,
    ProviderRuntimeTerminalFailureSink,
)
from puripuly_heart.core.runtime.local_asr_transition import LocalASRSessionOptions
from puripuly_heart.core.runtime.provider_handle import ProviderRuntimeHandle


class PrebuiltLocalASRProviderRuntime:
    def __init__(
        self,
        *,
        self_provider: object | None,
        peer_provider: object | None,
        callbacks: LocalASRProviderRuntimeCallbacks,
    ) -> None:
        self._handles = {
            "self": ProviderRuntimeHandle(
                name="prebuilt_self_stt",
                provider=self_provider,
                event_handler=callbacks.self_event_handler,
                retired_event_handler=callbacks.retired_event_handler,
                exception_handler=callbacks.self_exception_handler,
            ),
            "peer": ProviderRuntimeHandle(
                name="prebuilt_peer_stt",
                provider=peer_provider,
                event_handler=callbacks.peer_event_handler,
                retired_event_handler=callbacks.retired_event_handler,
                exception_handler=callbacks.peer_exception_handler,
            ),
        }
        self._pending_candidates: dict[ProviderRuntimeChannel, object] = {}
        self._closed = False

    @property
    def snapshot(self) -> LocalASRProviderRuntimeSnapshot:
        return LocalASRProviderRuntimeSnapshot(
            channels=tuple(self._channel_snapshot(channel) for channel in ("self", "peer")),
            gpu=ProviderRuntimeGpuSnapshot(
                phase="inactive",
                devices=(),
                active_channels=frozenset(),
                pending_count=0,
                worker_pid=None,
                configured_device_id=None,
                model_resident=False,
                retry_required=False,
                failure_code=None,
            ),
        )

    @property
    def diagnostics(self) -> tuple[ProviderRuntimeDiagnostic, ...]:
        return ()

    async def start(self) -> None:
        self._require_open()
        await self._handles["self"].start()
        await self._handles["peer"].start()

    async def replace_prebuilt_provider(
        self,
        channel: ProviderRuntimeChannel,
        provider: object | None,
        *,
        start: bool,
    ) -> object | None:
        self._require_open()
        return await self._handles[channel].replace_provider(provider, start=start)

    async def handoff_prebuilt_provider(
        self,
        channel: ProviderRuntimeChannel,
        provider: object,
        *,
        start: bool,
    ) -> object | None:
        self._require_open()
        self._pending_candidates[channel] = provider
        try:
            return await self._handles[channel].handoff_provider_at_boundary(
                provider,
                start=start,
            )
        finally:
            if self._pending_candidates.get(channel) is provider:
                self._pending_candidates.pop(channel, None)

    async def replace_provider(
        self,
        request: ProviderRuntimeBuildRequest,
        *,
        start: bool,
        on_terminal_failure: ProviderRuntimeTerminalFailureSink | None = None,
    ) -> ProviderRuntimeMutationResult:
        _ = request, start, on_terminal_failure
        raise RuntimeError("prebuilt provider runtime cannot construct providers")

    async def handoff_provider(
        self,
        request: ProviderRuntimeBuildRequest,
        *,
        start: bool,
        on_terminal_failure: ProviderRuntimeTerminalFailureSink | None = None,
    ) -> ProviderRuntimeMutationResult:
        _ = request, start, on_terminal_failure
        raise RuntimeError("prebuilt provider runtime cannot construct providers")

    async def commit_handoff(self, channel: ProviderRuntimeChannel) -> None:
        self._require_open()
        await self._handles[channel].commit_pending_handoff()

    async def cancel_handoff(self, channel: ProviderRuntimeChannel) -> bool:
        self._require_open()
        provider = self._pending_candidates.get(channel)
        if provider is None:
            return False
        cancelled = await self._handles[channel].cancel_pending_handoff(provider)
        if cancelled:
            self._pending_candidates.pop(channel, None)
            close = getattr(provider, "close", None)
            if callable(close):
                result = close()
                if inspect.isawaitable(result):
                    await result
        return cancelled

    async def release_channel(
        self,
        channel: ProviderRuntimeChannel,
        *,
        mode: ProviderRuntimeReleaseMode,
        release_backend_after: float | None = None,
    ) -> None:
        self._require_open()
        handle = self._handles[channel]
        if mode == "abort":
            await handle.abort_and_release()
        elif mode == "dormant":
            provider = handle.provider
            if provider is not None:
                await handle.retire_for_dormant_reuse(provider)
        else:
            await handle.drain_for_toggle_off(release_backend_after=release_backend_after)

    async def start_channel(self, channel: ProviderRuntimeChannel) -> None:
        self._require_open()
        await self._handles[channel].start()

    async def warmup_channel(self, channel: ProviderRuntimeChannel) -> None:
        self._require_open()
        provider = self._handles[channel].provider
        warmup = getattr(provider, "warmup", None)
        if callable(warmup):
            result = warmup()
            if inspect.isawaitable(result):
                await result

    async def reconfigure_channel(
        self,
        channel: ProviderRuntimeChannel,
        options: LocalASRSessionOptions,
    ) -> None:
        self._require_open()
        provider = self._handles[channel].provider
        reconfigure = getattr(provider, "reconfigure_session_options", None)
        if callable(reconfigure):
            result = reconfigure(options)
            if inspect.isawaitable(result):
                await result

    async def handle_vad_event(self, channel: ProviderRuntimeChannel, event: object) -> None:
        self._require_open()
        provider = self._handles[channel].provider
        handler = getattr(provider, "handle_vad_event", None)
        if callable(handler):
            result = handler(event)
            if inspect.isawaitable(result):
                await result

    async def discover_gpu(self, *, force: bool = False) -> LocalASRProviderRuntimeSnapshot:
        _ = force
        self._require_open()
        return self.snapshot

    async def inspect_gpu_readiness(
        self,
        *,
        explicit_intent: bool,
        device_id: str,
    ) -> LocalASRProviderRuntimeSnapshot:
        _ = explicit_intent, device_id
        self._require_open()
        return self.snapshot

    async def retry_gpu(
        self,
        desired_channels: tuple[ProviderRuntimeChannel, ...],
    ) -> LocalASRProviderRuntimeSnapshot:
        _ = desired_channels
        self._require_open()
        raise RuntimeError("prebuilt provider runtime has no GPU worker")

    async def close(self) -> None:
        if self._closed:
            return
        failures: list[Exception] = []
        for channel in ("self", "peer"):
            try:
                await self._handles[channel].close()
            except Exception as exc:
                failures.append(exc)
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise ExceptionGroup("prebuilt provider runtime close failed", failures)
        self._closed = True

    def _channel_snapshot(
        self,
        channel: ProviderRuntimeChannel,
    ) -> ProviderRuntimeChannelSnapshot:
        handle = self._handles[channel]
        provider = handle.provider
        provider_name = getattr(provider, "stt_provider_name", None)
        provider_id = getattr(provider_name, "value", provider_name)
        if provider is not None and not isinstance(provider_id, str):
            provider_id = "prebuilt"
        model_id = getattr(getattr(provider, "backend", None), "model_id", None)
        return ProviderRuntimeChannelSnapshot(
            channel=channel,
            provider_id=provider_id,
            model_id=model_id,
            phase=(
                "inactive"
                if provider is None
                else "running" if handle.event_task is not None else "ready"
            ),
            generation=handle.generation,
            pending_handoff=channel in self._pending_candidates,
            has_resources=handle.has_resources,
        )

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("prebuilt provider runtime is closed")


class PrebuiltLocalASRProviderRuntimeFactory:
    def __init__(self, *, self_provider: object | None, peer_provider: object | None) -> None:
        self._self_provider = self_provider
        self._peer_provider = peer_provider

    def create(
        self,
        callbacks: LocalASRProviderRuntimeCallbacks,
    ) -> PrebuiltLocalASRProviderRuntime:
        return PrebuiltLocalASRProviderRuntime(
            self_provider=self._self_provider,
            peer_provider=self._peer_provider,
            callbacks=callbacks,
        )


__all__ = ["PrebuiltLocalASRProviderRuntimeFactory"]
