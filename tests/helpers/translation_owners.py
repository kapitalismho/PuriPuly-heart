from __future__ import annotations

import asyncio
from dataclasses import replace
from uuid import UUID, uuid4

from puripuly_heart.core.local_asr_provider_runtime import (
    LocalASRProviderRuntimeCallbacks,
    LocalASRProviderRuntimePort,
)

from puripuly_heart.core.clock import Clock, SystemClock
from puripuly_heart.core.orchestrator.channel_runtime import (
    ChannelRuntime,
    _SpeculativeAttempt,
    _SpeculativeAttemptStatus,
)
from puripuly_heart.core.orchestrator.configuration import (
    TranslationRuntimeConfig,
    TranslationRuntimeConfigSnapshot,
    TranslationRuntimeConfigurationOwner,
)
from puripuly_heart.core.orchestrator.context import ContextResolver
from puripuly_heart.core.orchestrator.peer_translation_channel import (
    PeerTranslationChannelOwner,
)
from puripuly_heart.core.orchestrator.self_translation_channel import (
    SelfTranslationChannelOwner,
)
from puripuly_heart.core.orchestrator.translation_channel_callbacks import (
    TranslationChannelOwnerCallbacks,
)
from puripuly_heart.core.orchestrator.translation_diagnostics import (
    TranslationLatencyDiagnosticsOwner,
)
from puripuly_heart.core.orchestrator.translation_output_projection import (
    TranslationOutputProjectionOwner,
    TranslationUiMessageQueue,
)
from puripuly_heart.core.orchestrator.translation_request import (
    TranslationProcessRequest,
    TranslationRequestOwner,
)
from puripuly_heart.core.orchestrator.translation_turn import (
    TranslationTurnLifecycleOwner,
)
from puripuly_heart.core.runtime.output import OutputRuntime
from puripuly_heart.core.runtime.prebuilt_local_asr_provider_runtime import (
    PrebuiltLocalASRProviderRuntimeFactory,
)
from puripuly_heart.core.runtime.provider_handle import ProviderRuntimeHandle
from puripuly_heart.core.runtime.stt_session_projection import SttSessionStateProjection
from puripuly_heart.core.translation_backend import LlmTranslationBackend, TranslationBackend
from puripuly_heart.domain.events import STTFinalEvent
from puripuly_heart.domain.models import Transcript


def make_speculative_attempt(
    *,
    source_text: str = "",
    config_snapshot: TranslationRuntimeConfigSnapshot | None = None,
    provider_generation: int = 0,
    sequence: int = 1,
    task: asyncio.Task[None] | None = None,
    result: object | None = None,
    started_at: float | None = None,
    completed_at: float | None = None,
    latency_stage_times: dict[str, float] | None = None,
    status: _SpeculativeAttemptStatus | None = None,
) -> _SpeculativeAttempt:
    snapshot = config_snapshot or TranslationRuntimeConfigSnapshot(
        revision=0,
        value=TranslationRuntimeConfig(),
    )
    return _SpeculativeAttempt(
        source_text=source_text,
        normalized_text=source_text.strip(),
        config_snapshot=snapshot,
        provider_generation=provider_generation,
        sequence=sequence,
        status=status
        or (
            _SpeculativeAttemptStatus.READY
            if result is not None
            else _SpeculativeAttemptStatus.RUNNING
        ),
        task=task,
        result=result,
        started_at=started_at,
        completed_at=completed_at,
        latency_stage_times=dict(latency_stage_times or {}),
    )


class TranslationOwnersTestHarness:
    __slots__ = (
        "_peer_owner",
        "_self_owner",
        "_translation_runtime_configuration",
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
        peer_owner: PeerTranslationChannelOwner,
        self_owner: SelfTranslationChannelOwner,
        translation_runtime_configuration: TranslationRuntimeConfigurationOwner,
        llm_runtime: ProviderRuntimeHandle,
        local_asr_runtime: LocalASRProviderRuntimePort,
        output_runtime: OutputRuntime,
        output_projection: TranslationOutputProjectionOwner,
        osc: object,
        ui_events: asyncio.Queue,
        stt_sessions: SttSessionStateProjection,
    ) -> None:
        object.__setattr__(self, "_peer_owner", peer_owner)
        object.__setattr__(self, "_self_owner", self_owner)
        object.__setattr__(
            self,
            "_translation_runtime_configuration",
            translation_runtime_configuration,
        )
        object.__setattr__(self, "_llm_runtime", llm_runtime)
        object.__setattr__(self, "_local_asr_runtime", local_asr_runtime)
        object.__setattr__(self, "_output_runtime", output_runtime)
        object.__setattr__(self, "_output_projection", output_projection)
        object.__setattr__(self, "_osc", osc)
        object.__setattr__(self, "_ui_events", ui_events)
        object.__setattr__(self, "_stt_sessions", stt_sessions)
        object.__setattr__(self, "_started", False)

    @property
    def self_owner(self) -> SelfTranslationChannelOwner:
        return self._self_owner

    @property
    def peer_owner(self) -> PeerTranslationChannelOwner:
        return self._peer_owner

    @property
    def configuration(self) -> TranslationRuntimeConfigurationOwner:
        return self._translation_runtime_configuration

    @property
    def llm_runtime(self) -> ProviderRuntimeHandle:
        return self._llm_runtime

    @property
    def local_asr_runtime(self) -> LocalASRProviderRuntimePort:
        return self._local_asr_runtime

    @property
    def output_runtime(self) -> OutputRuntime:
        return self._output_runtime

    @property
    def output_projection(self) -> TranslationOutputProjectionOwner:
        return self._output_projection

    @property
    def osc(self) -> object:
        return self._osc

    @property
    def ui_events(self) -> asyncio.Queue:
        return self._ui_events

    @property
    def stt_sessions(self) -> SttSessionStateProjection:
        return self._stt_sessions

    @property
    def self_runtime(self) -> ChannelRuntime:
        return self._self_owner.runtime

    @property
    def peer_runtime(self) -> ChannelRuntime:
        return self._peer_owner.runtime

    @property
    def translation_turns(self) -> TranslationTurnLifecycleOwner:
        return self._peer_owner.translation_turns

    @property
    def translation_requests(self) -> TranslationRequestOwner:
        return self._self_owner.translation_requests

    @property
    def translation_diagnostics(self) -> TranslationLatencyDiagnosticsOwner:
        return self._peer_owner.diagnostics

    @property
    def clock(self) -> Clock:
        return self._self_owner.clock

    @property
    def started(self) -> bool:
        return self._started

    def replace_configuration(self, **changes: object) -> None:
        self._translation_runtime_configuration.transform(
            lambda current: replace(current, **changes)
        )

    def set_clock(self, clock: Clock) -> None:
        self._self_owner.set_clock(clock)
        self._peer_owner.set_clock(clock)
        self._output_projection.set_clock(clock)
        self._peer_owner.diagnostics.clock = clock
        self._peer_owner.translation_requests.set_clock(clock)

    def set_started_for_test(self, started: bool) -> None:
        object.__setattr__(self, "_started", started)

    def replace_translation_turn_owner_for_test(self, owner: object) -> None:
        self._self_owner.translation_turns = owner
        self._peer_owner.translation_turns = owner

    async def dispatch_stt_event(self, event: object) -> None:
        if getattr(event, "channel", "self") == "self":
            await self._self_owner.handle_stt_event(event)
            return
        await self._peer_owner.handle_stt_event(event)

    async def dispatch_retired_stt_event(self, event: object) -> None:
        if getattr(event, "channel", "self") == "self":
            await self._self_owner.handle_retired_stt_event(event)
            return
        await self._peer_owner.handle_retired_stt_event(event)

    async def dispatch_stt_failure(
        self,
        exc: Exception,
        *,
        channel: str = "self",
    ) -> None:
        if channel == "self":
            await self._self_owner.handle_stt_event_loop_exception(exc)
            return
        await self._peer_owner.handle_stt_event_loop_exception(exc, channel="peer")

    async def dispatch_transcript(self, *args: object, **kwargs: object) -> None:
        transcript = args[0] if args else kwargs.get("transcript")
        if getattr(transcript, "channel", "self") == "self":
            await self._self_owner._handle_transcript(*args, **kwargs)
            return
        await self._peer_owner._handle_transcript(*args, **kwargs)

    async def ensure_translation(self, *args: object, **kwargs: object) -> None:
        transcript = args[0] if args else kwargs.get("transcript")
        if getattr(transcript, "channel", "self") == "self":
            await self._self_owner._ensure_translation(*args, **kwargs)
            return
        await self._peer_owner._ensure_translation(*args, **kwargs)

    async def process_translation(
        self,
        utterance_id: object,
        text: str,
        *,
        runtime: object | None = None,
        detected_language: str | None = None,
        cancellation_requested=None,
    ) -> None:
        runtime = runtime or self._self_owner.runtime
        if getattr(runtime, "channel", "self") == "self":
            await self._self_owner.translate_and_enqueue(
                utterance_id,
                text,
                cancellation_requested=cancellation_requested,
            )
            return
        config_snapshot = self._translation_runtime_configuration.snapshot()
        source = runtime.get_source(utterance_id) or "Peer"
        result = await self._peer_owner.translation_requests.process(
            TranslationProcessRequest(
                parent_utterance_id=utterance_id,
                utterance_id=utterance_id,
                sequence=0,
                text=text,
                channel="peer",
                source=source,
                target_language=self._peer_owner.translation_requests.target_language_for(
                    "peer",
                    config_snapshot.value,
                ),
                context_policy=self._peer_owner.translation_turns.policy.context_policy,
                detected_language=detected_language,
                config_snapshot=config_snapshot,
            ),
            cancellation_requested=cancellation_requested,
        )
        if result.output is not None:
            await self._peer_owner.submit_translation_output(result.output)

    async def handle_peer_transcript_final_for_test(
        self,
        text: str,
        source: str = "Peer",
    ) -> UUID:
        _ = source
        parent_utterance_id = uuid4()
        runtime = self._peer_owner.runtime
        existing_peer_utterance_ids = set(runtime.utterances)
        await self._peer_owner.handle_stt_event(
            STTFinalEvent(
                utterance_id=parent_utterance_id,
                transcript=Transcript(
                    utterance_id=parent_utterance_id,
                    text=text,
                    is_final=True,
                    created_at=self._peer_owner.clock.now(),
                    channel="peer",
                ),
            )
        )
        if (
            not self._peer_owner.translation_requests.provider_available
            or not self._peer_owner._translation_enabled_for_runtime(runtime)
        ):
            await self._peer_owner.translation_turns.wait_for_idle()
        for utterance_id, bundle in runtime.utterances.items():
            if utterance_id in existing_peer_utterance_ids:
                continue
            if bundle.final is not None and bundle.final.text == text:
                return utterance_id
        raise AssertionError("peer test helper did not produce a peer logical turn")

    async def translate_peer_text_for_test(self, text: str) -> UUID:
        utterance_id = await self.handle_peer_transcript_final_for_test(text=text)
        await self._peer_owner.translation_turns.wait_for_idle()
        return utterance_id

    async def reset_provider_channel(self, channel: str) -> None:
        if channel == "self":
            await self._self_owner.reset_provider_channel(channel)
            return
        await self._peer_owner.reset_provider_channel(channel)

    async def clear_channel_language_state(self, *, channel: str) -> None:
        if channel == "self":
            await self._self_owner.clear_language_runtime_state()
            return
        await self._peer_owner.clear_language_runtime_state(channel=channel)

    async def reset_runtime_state(self) -> None:
        await self._self_owner.runtime.reset_runtime_state()
        await self._peer_owner.reset_provider_channel("peer")

    def clear_context(self) -> None:
        self._self_owner.translation_requests.clear_context()

    def get_valid_context(self) -> object:
        return self._self_owner.translation_requests.get_valid_context()

    def format_context(self, context: object) -> str:
        return self._self_owner.translation_requests.format_context(context)

    def remember_context(
        self,
        text: str,
        timestamp: float,
        *,
        config_snapshot=None,
        runtime=None,
        source_language: str | None = None,
    ) -> None:
        runtime = runtime or self._self_owner.runtime
        self._self_owner.translation_requests.remember_context(
            text,
            timestamp,
            channel=runtime.channel,
            config_snapshot=config_snapshot,
            source_language=source_language,
        )

    def prepare_translation_request(
        self,
        text: str,
        *,
        runtime=None,
        detected_language: str | None = None,
        context_policy: str = "integrated_preferred",
        config_snapshot=None,
    ) -> tuple[str, str, float]:
        runtime = runtime or self._self_owner.runtime
        prepared = self._self_owner.translation_requests.prepare(
            text,
            channel=runtime.channel,
            detected_language=detected_language,
            context_policy=context_policy,
            config_snapshot=config_snapshot,
        )
        return prepared.system_prompt, prepared.context, prepared.requested_at

    def prepare_translation_request_with_mode(
        self,
        text: str,
        *,
        runtime=None,
        detected_language: str | None = None,
        context_policy: str = "integrated_preferred",
        config_snapshot=None,
    ) -> tuple[str, str, float, object]:
        runtime = runtime or self._self_owner.runtime
        prepared = self._self_owner.translation_requests.prepare(
            text,
            channel=runtime.channel,
            detected_language=detected_language,
            context_policy=context_policy,
            config_snapshot=config_snapshot,
        )
        return (
            prepared.system_prompt,
            prepared.context,
            prepared.requested_at,
            prepared.applied_context_mode,
        )

    def bundle_for(self, utterance_id: object, *, channel: str = "self") -> object:
        runtime = self._self_owner.runtime if channel == "self" else self._peer_owner.runtime
        return runtime.get_or_create_bundle(utterance_id)

    async def start(self, *, auto_flush_osc: bool = False) -> None:
        if self._started:
            return
        await self._output_runtime.start(auto_flush_chatbox=auto_flush_osc)
        await self._self_owner.open_ingress()
        await self._peer_owner.open_ingress()
        await self._peer_owner.translation_turns.open_channel_ingress("self")
        await self._peer_owner.translation_turns.open_channel_ingress("peer")
        await self._peer_owner.translation_turns.start()
        await self._local_asr_runtime.start()
        object.__setattr__(self, "_started", True)

    async def stop(self) -> None:
        failures: list[BaseException] = []
        was_started = self._started
        object.__setattr__(self, "_started", False)
        for callback in (
            self._self_owner.close_ingress,
            self._peer_owner.close_ingress,
            lambda: self._peer_owner.translation_turns.close_channel_ingress("self"),
            lambda: self._peer_owner.translation_turns.close_channel_ingress("peer"),
            self._peer_owner.translation_turns.close,
            self._output_runtime.close,
        ):
            try:
                await callback()
            except BaseException as exc:
                failures.append(exc)
        if was_started:
            for callback in (
                self._output_projection.reset_overlay_preview,
                self._self_owner.close,
                self._peer_owner.close,
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
        await self._self_owner.reset_provider_channel("self")
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
        await self._peer_owner.reset_provider_channel("peer")
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
        await self._peer_owner.reset_provider_channel("peer")
        await self._local_asr_runtime.release_channel("peer", mode="abort")

    async def replace_llm_provider(self, llm: object | None) -> object | None:
        backend = (
            llm
            if llm is None or isinstance(llm, TranslationBackend)
            else LlmTranslationBackend(llm)
        )
        return await self._llm_runtime.replace_provider(backend, start=False)

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
        await self._self_owner.reset_provider_channel("self")
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


def compose_translation_test_harness(**values: object) -> TranslationOwnersTestHarness:
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
    callbacks = TranslationChannelOwnerCallbacks(stt_sessions)
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
        predecessor_wait_observer=translation_diagnostics.record_translation_wait,
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
    backend = (
        llm if llm is None or isinstance(llm, TranslationBackend) else LlmTranslationBackend(llm)
    )
    llm_runtime = ProviderRuntimeHandle(name="llm", provider=backend)
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
    self_owner = SelfTranslationChannelOwner(
        runtime=self_runtime,
        config_snapshot=config_owner.snapshot,
        translation_turns=translation_turns,
        local_asr_runtime=local_asr_runtime,
        translation_requests=translation_requests,
        output_projection=translation_output_projection,
        diagnostics=translation_diagnostics,
        clock=clock,
    )
    peer_owner = PeerTranslationChannelOwner(
        runtime=peer_runtime,
        config_snapshot=config_owner.snapshot,
        translation_turns=translation_turns,
        local_asr_runtime=local_asr_runtime,
        translation_requests=translation_requests,
        output_projection=translation_output_projection,
        diagnostics=translation_diagnostics,
        clock=clock,
    )
    callbacks.bind_self(self_owner)
    callbacks.bind_peer(peer_owner)
    return TranslationOwnersTestHarness(
        peer_owner=peer_owner,
        self_owner=self_owner,
        translation_runtime_configuration=config_owner,
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
            "translation-owner test lifecycle cleanup failed",
            [failure for failure in failures if isinstance(failure, Exception)],
        )
    raise BaseExceptionGroup("translation-owner test lifecycle cleanup failed", failures)


__all__ = ["TranslationOwnersTestHarness", "compose_translation_test_harness"]
