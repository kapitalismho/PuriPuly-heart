from __future__ import annotations

import asyncio
from dataclasses import replace

from puripuly_heart.core.clock import SystemClock
from puripuly_heart.core.local_asr_provider_runtime import (
    LocalASRProviderRuntimeCallbacks,
)
from puripuly_heart.core.orchestrator.channel_runtime import ChannelRuntime
from puripuly_heart.core.orchestrator.configuration import (
    TranslationRuntimeConfig,
    TranslationRuntimeConfigurationOwner,
)
from puripuly_heart.core.orchestrator.context import ContextResolver
from puripuly_heart.core.orchestrator.hub import ClientHub
from puripuly_heart.core.orchestrator.hub_callbacks import (
    ClientHubDurableOwnerCallbacks,
)
from puripuly_heart.core.orchestrator.translation_diagnostics import (
    TranslationLatencyDiagnosticsOwner,
)
from puripuly_heart.core.orchestrator.translation_output_projection import (
    TranslationOutputProjectionOwner,
    TranslationUiMessageQueue,
)
from puripuly_heart.core.orchestrator.translation_request import TranslationRequestOwner
from puripuly_heart.core.orchestrator.translation_turn import (
    TranslationTurnLifecycleOwner,
)
from puripuly_heart.core.runtime.output import OutputRuntime
from puripuly_heart.core.runtime.prebuilt_local_asr_provider_runtime import (
    PrebuiltLocalASRProviderRuntimeFactory,
)
from puripuly_heart.core.runtime.provider_handle import ProviderRuntimeHandle
from puripuly_heart.core.runtime.stt_session_projection import SttSessionStateProjection


class ClientHubTestHarness:
    __slots__ = (
        "_hub",
        "_llm_runtime",
        "_local_asr_runtime",
        "_output_runtime",
        "_output_projection",
        "_osc",
        "_ui_events",
        "_stt_sessions",
        "_started",
    )

    def __init__(
        self,
        *,
        hub: ClientHub,
        llm_runtime: ProviderRuntimeHandle,
        local_asr_runtime: object,
        output_runtime: OutputRuntime,
        output_projection: TranslationOutputProjectionOwner,
        osc: object,
        ui_events: asyncio.Queue,
        stt_sessions: SttSessionStateProjection,
    ) -> None:
        object.__setattr__(self, "_hub", hub)
        object.__setattr__(self, "_llm_runtime", llm_runtime)
        object.__setattr__(self, "_local_asr_runtime", local_asr_runtime)
        object.__setattr__(self, "_output_runtime", output_runtime)
        object.__setattr__(self, "_output_projection", output_projection)
        object.__setattr__(self, "_osc", osc)
        object.__setattr__(self, "_ui_events", ui_events)
        object.__setattr__(self, "_stt_sessions", stt_sessions)
        object.__setattr__(self, "_started", False)

    def __getattr__(self, name: str) -> object:
        if name == "llm":
            return self._llm_runtime.provider
        if name in {"stt", "peer_stt"}:
            return None
        if name == "local_asr_provider_runtime":
            return self._local_asr_runtime
        if name == "output_runtime":
            return self._output_runtime
        if name == "output_projection":
            return self._output_projection
        if name == "overlay_event_adapter":
            return self._output_projection.overlay_event_adapter
        if name == "overlay_sink":
            return self._output_projection.overlay_sink
        if name == "ui_events":
            return self._ui_events
        if name == "osc":
            return self._osc
        if name == "provider_runtime_handles":
            return {"llm": self._llm_runtime}
        if name == "_running":
            return self._started
        return getattr(self._hub, name)

    def __setattr__(self, name: str, value: object) -> None:
        if name in ClientHubTestHarness.__slots__:
            object.__setattr__(self, name, value)
            return
        if name == "llm":
            self._llm_runtime.attach_provider_reference(value)
            return
        if name in {"stt", "peer_stt"}:
            if value is not None:
                raise RuntimeError("concrete STT assignment is disabled")
            return
        if name == "_running":
            object.__setattr__(self, "_started", bool(value))
            return
        setattr(self._hub, name, value)

    async def start(self, *, auto_flush_osc: bool = False) -> None:
        if self._started:
            return
        await self._output_runtime.start(auto_flush_chatbox=auto_flush_osc)
        await self._hub.translation_turns.open_channel_ingress("self")
        await self._hub.translation_turns.open_channel_ingress("peer")
        await self._hub.translation_turns.start()
        await self._local_asr_runtime.start()
        object.__setattr__(self, "_started", True)

    async def stop(self) -> None:
        failures: list[BaseException] = []
        was_started = self._started
        object.__setattr__(self, "_started", False)
        for callback in (
            lambda: self._hub.translation_turns.close_channel_ingress("self"),
            lambda: self._hub.translation_turns.close_channel_ingress("peer"),
            self._hub.translation_turns.close,
            self._output_runtime.close,
        ):
            try:
                await callback()
            except BaseException as exc:
                failures.append(exc)
        if was_started:
            for callback in (
                self._output_projection.reset_overlay_preview,
                self._hub._reset_stt_runtime_state,
            ):
                try:
                    await callback()
                except BaseException as exc:
                    failures.append(exc)
        for callback in (
            self._local_asr_runtime.close,
            self._llm_runtime.close,
        ):
            try:
                await callback()
            except BaseException as exc:
                failures.append(exc)
        _raise_failures(failures)

    def has_stt_provider(self, channel: str) -> bool:
        return self._local_asr_runtime.snapshot.channel_for(channel).provider_id is not None

    def stt_session_state(self, channel: str = "self") -> object | None:
        return self._stt_sessions.state(channel)

    async def replace_stt_provider_request(
        self,
        request: object,
        *,
        start: bool | None = None,
        on_terminal_failure=None,
    ) -> object:
        await self._hub.reset_provider_channel("self")
        return await self._local_asr_runtime.replace_provider(
            request,
            start=self._started if start is None else start,
            on_terminal_failure=on_terminal_failure,
        )

    async def handoff_stt_provider_request(
        self,
        request: object,
        *,
        start: bool | None = None,
        on_terminal_failure=None,
    ) -> object:
        return await self._local_asr_runtime.handoff_provider(
            request,
            start=self._started if start is None else start,
            on_terminal_failure=on_terminal_failure,
        )

    async def cancel_stt_provider_request_handoff(self) -> bool:
        return await self._local_asr_runtime.cancel_handoff("self")

    async def replace_peer_stt_provider_request(
        self,
        request: object,
        *,
        start: bool | None = None,
        on_terminal_failure=None,
    ) -> object:
        await self._hub.reset_provider_channel("peer")
        return await self._local_asr_runtime.replace_provider(
            request,
            start=self._started if start is None else start,
            on_terminal_failure=on_terminal_failure,
        )

    async def handoff_peer_stt_provider_request(
        self,
        request: object,
        *,
        start: bool | None = None,
        on_terminal_failure=None,
    ) -> object:
        return await self._local_asr_runtime.handoff_provider(
            request,
            start=self._started if start is None else start,
            on_terminal_failure=on_terminal_failure,
        )

    async def cancel_peer_stt_provider_request_handoff(self) -> bool:
        return await self._local_asr_runtime.cancel_handoff("peer")

    async def start_peer_stt_provider_ingress(self) -> None:
        if self._started:
            await self._local_asr_runtime.start_channel("peer")

    async def abort_peer_stt_for_toggle_off(self) -> None:
        await self._hub.reset_provider_channel("peer")
        await self._local_asr_runtime.release_channel("peer", mode="abort")

    async def replace_llm_provider(self, llm: object | None) -> object | None:
        return await self._llm_runtime.replace_provider(llm, start=False)

    async def drain_self_stt_for_toggle_off(
        self,
        *,
        release_backend_after: float | None = None,
    ) -> None:
        await self._local_asr_runtime.release_channel(
            "self",
            mode="drain",
            release_backend_after=release_backend_after,
        )

    async def abort_self_stt_for_toggle_off(self) -> None:
        await self._hub.reset_provider_channel("self")
        await self._local_asr_runtime.release_channel("self", mode="abort")

    async def schedule_self_stt_idle_release(self, *, release_backend_after: float) -> None:
        await self._local_asr_runtime.release_channel(
            "self",
            mode="drain",
            release_backend_after=release_backend_after,
        )

    async def resume_self_stt_after_toggle_on(self) -> None:
        await self._local_asr_runtime.start_channel("self")

    async def warmup_stt_channel(self, channel: str) -> None:
        await self._local_asr_runtime.warmup_channel(channel)

    async def reconfigure_stt_channel(self, channel: str, options: object) -> None:
        await self._local_asr_runtime.reconfigure_channel(channel, options)


def compose_client_hub(**values: object) -> ClientHubTestHarness:
    stt = values.pop("stt", None)
    peer_stt = values.pop("peer_stt", None)
    llm = values.pop("llm", None)
    osc = values.pop("osc")
    clock = values.pop("clock", None) or SystemClock()
    overlay_sink = values.pop("overlay_sink", None)
    overlay_diagnostics = values.pop("overlay_diagnostics", None)
    runtime_logging = values.pop("runtime_logging", None)
    runtime_factory = values.pop("local_asr_provider_runtime_factory", None)
    config_owner = values.pop("translation_runtime_configuration", None)
    config_fields = TranslationRuntimeConfig.__dataclass_fields__
    config_overrides = {
        name: values.pop(name)
        for name in tuple(values)
        if name in config_fields and values[name] is not None
    }
    if config_owner is None:
        config_owner = TranslationRuntimeConfigurationOwner(
            replace(TranslationRuntimeConfig(), **config_overrides)
        )
    elif config_overrides:
        config_owner.replace(replace(config_owner.snapshot().value, **config_overrides))
    stt_sessions = SttSessionStateProjection()
    callbacks = ClientHubDurableOwnerCallbacks(stt_sessions)
    output_runtime = OutputRuntime(
        chatbox=osc,
        clock=clock,
        overlay_sink=overlay_sink,
    )
    self_runtime = ChannelRuntime(channel="self")
    peer_runtime = ChannelRuntime(channel="peer")
    context_resolver = ContextResolver(
        clock=clock,
        config_snapshot=config_owner.snapshot,
    )
    translation_diagnostics = TranslationLatencyDiagnosticsOwner(
        clock=clock,
        config_snapshot=config_owner.snapshot,
        runtime_logging=runtime_logging,
        overlay_diagnostics=overlay_diagnostics,
    )
    ui_events = asyncio.Queue()
    translation_output_projection = TranslationOutputProjectionOwner(
        output_runtime=output_runtime,
        ui_messages=TranslationUiMessageQueue(ui_events),
        diagnostics=translation_diagnostics,
        clock=clock,
    )
    translation_turns = TranslationTurnLifecycleOwner(
        on_child_created=callbacks.child_created,
        on_child_started=callbacks.child_started,
        process_child=callbacks.process_child,
        on_child_terminal=callbacks.child_terminal,
        on_parent_closed=callbacks.parent_closed,
        on_parent_rejected=callbacks.parent_rejected,
        output=callbacks,
        config_snapshot=config_owner.snapshot,
    )
    factory = runtime_factory or PrebuiltLocalASRProviderRuntimeFactory(
        self_provider=stt,
        peer_provider=peer_stt,
    )
    local_asr_runtime = factory.create(
        LocalASRProviderRuntimeCallbacks(
            self_event_handler=callbacks.self_event_handler,
            peer_event_handler=callbacks.peer_event_handler,
            retired_event_handler=callbacks.retired_event_handler,
            self_exception_handler=callbacks.self_exception_handler,
            peer_exception_handler=callbacks.peer_exception_handler,
        )
    )
    llm_runtime = ProviderRuntimeHandle(name="llm", provider=llm)
    translation_requests = TranslationRequestOwner(
        config_snapshot=config_owner.snapshot,
        self_runtime=self_runtime,
        peer_runtime=peer_runtime,
        context_resolver=context_resolver,
        provider_runtime=llm_runtime,
        diagnostics=translation_diagnostics,
        presentation=translation_output_projection,
        clock=clock,
    )
    hub = ClientHub(
        translation_runtime_configuration=config_owner,
        direct_self_runtime=self_runtime,
        direct_peer_runtime=peer_runtime,
        direct_translation_turns=translation_turns,
        direct_local_asr_runtime=local_asr_runtime,
        direct_translation_diagnostics=translation_diagnostics,
        direct_output_projection=translation_output_projection,
        direct_translation_requests=translation_requests,
        clock=clock,
        **values,
    )
    callbacks.bind(hub)
    return ClientHubTestHarness(
        hub=hub,
        llm_runtime=llm_runtime,
        local_asr_runtime=local_asr_runtime,
        output_runtime=output_runtime,
        output_projection=translation_output_projection,
        osc=osc,
        ui_events=ui_events,
        stt_sessions=stt_sessions,
    )


def _raise_failures(failures: list[BaseException]) -> None:
    if not failures:
        return
    if len(failures) == 1:
        raise failures[0]
    if all(isinstance(failure, Exception) for failure in failures):
        raise ExceptionGroup(
            "ClientHub test lifecycle cleanup failed",
            [failure for failure in failures if isinstance(failure, Exception)],
        )
    raise BaseExceptionGroup("ClientHub test lifecycle cleanup failed", failures)


__all__ = ["ClientHubTestHarness", "compose_client_hub"]
