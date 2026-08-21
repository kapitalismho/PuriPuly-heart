from __future__ import annotations

from dataclasses import replace
from uuid import uuid4

from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.orchestrator.configuration import (
    TranslationRuntimeConfig,
    TranslationRuntimeConfigurationOwner,
)
from puripuly_heart.core.orchestrator.context import ContextMode
from puripuly_heart.core.orchestrator.translation_diagnostics import (
    ContextApplicationDiagnostic,
    ContextModeDiagnostic,
    LatencyInheritanceDiagnostic,
    LatencyStageDiagnostic,
    OverlayEmitDiagnostic,
    RuntimeDiagnostic,
    SelfOverlayDecisionDiagnostic,
    SttEventLoopFailureDiagnostic,
    TranslationLatencyDiagnosticsOwner,
    TranslationReadyDiagnostic,
    TranslationSkipDiagnostic,
)


class RuntimeLogging:
    def __init__(self, *, detailed: bool = True) -> None:
        self.mode = "detailed" if detailed else "basic"
        self.basic: list[str] = []
        self.detailed: list[str] = []

    def emit_basic(self, message: str, *, level: int = 20) -> None:
        _ = level
        self.basic.append(message)

    def emit_detailed(self, message: str, *, level: int = 20) -> bool:
        _ = level
        if self.mode != "detailed":
            return False
        self.detailed.append(message)
        return True

    def emit_detailed_lazy(self, build_message, *, level: int = 20) -> bool:
        _ = level
        if self.mode != "detailed":
            return False
        self.detailed.append(build_message())
        return True


class OverlayDiagnostics:
    def __init__(self) -> None:
        self.records: list[tuple[str, dict[str, object]]] = []

    def record_translation(self, event: str, **fields: object) -> None:
        self.records.append((event, fields))


class ExplodingString:
    def __str__(self) -> str:
        raise AssertionError("disabled Detailed logging evaluated the message")


class SttProvider:
    stt_provider_name = "soniox"
    channel = "peer"


class ExplodingSttProvider:
    @property
    def stt_provider_name(self) -> str:
        raise AssertionError("fallback-only STT diagnostics read provider metadata")


def make_owner(
    *,
    clock: FakeClock | None = None,
    runtime_logging: RuntimeLogging | None = None,
    overlay_diagnostics: OverlayDiagnostics | None = None,
) -> TranslationLatencyDiagnosticsOwner:
    config = TranslationRuntimeConfigurationOwner(
        replace(
            TranslationRuntimeConfig(),
            hangover_s=0.4,
            peer_hangover_s=0.9,
        )
    )
    return TranslationLatencyDiagnosticsOwner(
        clock=clock or FakeClock(_now=10.0),
        config_snapshot=config.snapshot,
        runtime_logging=runtime_logging,
        overlay_diagnostics=overlay_diagnostics,
    )


def test_owner_emits_latency_contract_once_with_channel_hangover_and_cause() -> None:
    logging = RuntimeLogging()
    owner = make_owner(runtime_logging=logging)
    utterance_id = uuid4()
    stages = (
        ("speech_end", 10.0),
        ("stt_final", 10.1),
        ("llm_request_start", 10.2),
        ("llm_first_chunk", 10.4),
        ("llm_done", 10.5),
        ("self_chatbox_enqueue", 10.6),
    )

    for stage, timestamp in stages:
        owner.record_latency_stage(
            LatencyStageDiagnostic(
                channel="self",
                utterance_id=utterance_id,
                stage=stage,
                timestamp=timestamp,
            )
        )
    owner.record_latency_stage(
        LatencyStageDiagnostic(
            channel="self",
            utterance_id=utterance_id,
            stage="self_chatbox_enqueue",
            timestamp=10.7,
            overwrite=False,
        )
    )

    assert logging.basic == ["[Basic][Latency] channel=self e2e_ms=1000"]
    assert sum("[Detailed][Latency]" in message for message in logging.detailed) == 6
    cause = next(message for message in logging.detailed if "latency_cause" in message)
    assert "provider=llm" in cause
    assert "dominant_stage=llm_request_to_llm_done" in cause
    assert "llm_request_to_llm_done_ms=300" in cause


def test_owner_inherits_and_clears_only_the_selected_timeline() -> None:
    owner = make_owner()
    source_id = uuid4()
    output_id = uuid4()
    peer_id = uuid4()
    owner.record_latency_stage(
        LatencyStageDiagnostic(
            channel="self",
            utterance_id=source_id,
            stage="speech_end",
            timestamp=10.0,
            publish_now=False,
        )
    )
    owner.record_latency_stage(
        LatencyStageDiagnostic(
            channel="peer",
            utterance_id=peer_id,
            stage="speech_end",
            timestamp=10.0,
            publish_now=False,
        )
    )
    owner.inherit_latency(
        LatencyInheritanceDiagnostic(
            channel="self",
            output_utterance_id=output_id,
            source_utterance_ids=(source_id,),
        )
    )

    assert owner.snapshot().timeline_keys == frozenset(
        {
            ("self", source_id),
            ("self", output_id),
            ("peer", peer_id),
        }
    )
    owner.clear_latency_state("self")
    assert owner.snapshot().timeline_keys == frozenset({("peer", peer_id)})


def test_owner_suppresses_duplicate_context_mode_and_logs_metadata_only() -> None:
    logging = RuntimeLogging()
    owner = make_owner(runtime_logging=logging)
    mode: ContextMode = "integrated"
    owner.record_context_mode(ContextModeDiagnostic(channel="self", applied_mode=mode))
    owner.record_context_mode(ContextModeDiagnostic(channel="self", applied_mode=mode))
    owner.record_context_application(
        ContextApplicationDiagnostic(
            channel="self",
            request_chars=17,
            context_lines=("- [self] first", "- [peer] second"),
            context_chars=35,
        )
    )

    assert sum("Context mode" in message for message in logging.basic) == 1
    application = next(message for message in logging.basic if "Context apply" in message)
    assert "entries=2" in application
    assert "self_entries=1" in application
    assert "peer_entries=1" in application
    assert "first" not in application
    assert "second" not in application


def test_owner_suppresses_runtime_and_overlay_decision_duplicates_independently() -> None:
    logging = RuntimeLogging()
    overlay = OverlayDiagnostics()
    owner = make_owner(
        runtime_logging=logging,
        overlay_diagnostics=overlay,
    )
    diagnostic = SelfOverlayDecisionDiagnostic.create(
        merge_id=uuid4(),
        source="spec",
        active_text="active value",
        secondary_text="subtitle",
        spec_text_len=12,
        spec_translation_len=8,
        cached_secondary_len=0,
        reuse_mode="exact",
        resume_pending=False,
        resume_confirmed=False,
    )

    owner.record_self_overlay_decision(diagnostic)
    owner.record_self_overlay_decision(diagnostic)

    assert sum("active_self_secondary" in message for message in logging.detailed) == 1
    assert [event for event, _fields in overlay.records] == ["active_self_secondary"]


def test_owner_does_not_suppress_same_length_overlay_text_changes() -> None:
    logging = RuntimeLogging()
    overlay = OverlayDiagnostics()
    owner = make_owner(
        runtime_logging=logging,
        overlay_diagnostics=overlay,
    )
    merge_id = uuid4()

    for active_text, secondary_text in (("alpha", "beta"), ("bravo", "zeta")):
        owner.record_self_overlay_decision(
            SelfOverlayDecisionDiagnostic.create(
                merge_id=merge_id,
                source="spec",
                active_text=active_text,
                secondary_text=secondary_text,
                spec_text_len=5,
                spec_translation_len=4,
                cached_secondary_len=0,
                reuse_mode=None,
                resume_pending=False,
                resume_confirmed=False,
            )
        )

    assert sum("active_self_secondary" in message for message in logging.detailed) == 2
    assert [event for event, _fields in overlay.records] == [
        "active_self_secondary",
        "active_self_secondary",
    ]


def test_owner_preserves_overlay_suppression_across_detach_and_replacement() -> None:
    first = OverlayDiagnostics()
    second = OverlayDiagnostics()
    owner = make_owner(overlay_diagnostics=first)
    merge_id = uuid4()
    unchanged = SelfOverlayDecisionDiagnostic.create(
        merge_id=merge_id,
        source="spec",
        active_text="alpha",
        secondary_text="beta",
        spec_text_len=5,
        spec_translation_len=4,
        cached_secondary_len=0,
        reuse_mode=None,
        resume_pending=False,
        resume_confirmed=False,
    )
    changed = SelfOverlayDecisionDiagnostic.create(
        merge_id=merge_id,
        source="spec",
        active_text="bravo",
        secondary_text="zeta",
        spec_text_len=5,
        spec_translation_len=4,
        cached_secondary_len=0,
        reuse_mode=None,
        resume_pending=False,
        resume_confirmed=False,
    )

    owner.record_self_overlay_decision(unchanged)
    assert owner.replace_overlay_diagnostics(
        None,
        expected_current=first,
        require_match=True,
    )
    assert owner.replace_overlay_diagnostics(second)
    owner.record_self_overlay_decision(unchanged)
    owner.record_self_overlay_decision(changed)

    assert [event for event, _fields in first.records] == ["active_self_secondary"]
    assert [event for event, _fields in second.records] == ["active_self_secondary"]


def test_owner_keeps_detailed_message_building_lazy_in_basic_mode() -> None:
    logging = RuntimeLogging(detailed=False)
    owner = make_owner(runtime_logging=logging)

    assert not owner.emit(
        RuntimeDiagnostic(
            message="[Translation] detail=%s",
            args=(ExplodingString(),),
            detailed=True,
        )
    )
    assert not owner.emit_translation_ready(
        TranslationReadyDiagnostic(
            channel="self",
            utterance_id=uuid4(),
            update_id=ExplodingString(),
            origin_wall_clock_ms=None,
            session_scope=None,
            source_text_hash=None,
            source_text_len=None,
            logical_turn_key=None,
            translation_len=3,
        )
    )
    assert logging.detailed == []


def test_owner_sanitizes_stt_failure_and_tracks_overlay_failure_state() -> None:
    logging = RuntimeLogging()
    owner = make_owner(runtime_logging=logging)
    owner.record_stt_event_loop_failure(
        SttEventLoopFailureDiagnostic(
            exception=RuntimeError("private speech secret-token"),
            provider=SttProvider(),
            default_channel="self",
        )
    )
    owner.record_overlay_sink_failure("RuntimeError")

    combined = "\n".join(logging.basic + logging.detailed)
    assert "private speech" not in combined
    assert "secret-token" not in combined
    assert "code=stt.unknown" in combined
    assert owner.snapshot().last_error_source == "overlay_sink"


def test_owner_fallback_stt_failure_does_not_read_provider_metadata(caplog) -> None:
    owner = make_owner()

    owner.record_stt_event_loop_failure(
        SttEventLoopFailureDiagnostic(
            exception=RuntimeError("private speech secret-token"),
            provider=ExplodingSttProvider(),
            default_channel="self",
        )
    )

    assert "RuntimeError" in caplog.text
    assert "private speech" not in caplog.text
    assert "secret-token" not in caplog.text


def test_owner_derives_translation_skip_reason_from_runtime_state() -> None:
    logging = RuntimeLogging()
    owner = make_owner(runtime_logging=logging)

    owner.record_translation_skip(
        TranslationSkipDiagnostic(
            stage="final",
            channel="peer",
            publish_chatbox=False,
            llm_available=True,
            configuration=replace(
                TranslationRuntimeConfig(),
                peer_translation_enabled=False,
            ),
        )
    )

    assert any("peer translation disabled" in message for message in logging.detailed)


def test_owner_replaces_overlay_diagnostics_by_expected_identity() -> None:
    first = OverlayDiagnostics()
    second = OverlayDiagnostics()
    owner = make_owner(overlay_diagnostics=first)

    assert not owner.replace_overlay_diagnostics(
        second,
        expected_current=second,
        require_match=True,
    )
    assert owner.overlay_diagnostics is first
    assert owner.replace_overlay_diagnostics(
        second,
        expected_current=first,
        require_match=True,
    )
    owner.record_overlay_emit(
        OverlayEmitDiagnostic(
            event_kind="translation_final",
            utterance_id=uuid4(),
            channel="peer",
            secondary_len=7,
            sink_type="OverlayPresenter",
        )
    )

    assert owner.overlay_diagnostics is second
    assert [event for event, _fields in second.records] == ["overlay_emit"]
