from __future__ import annotations

from dataclasses import replace
from uuid import uuid4

import pytest

from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.orchestrator.configuration import (
    TranslationRuntimeConfig,
    TranslationRuntimeConfigurationOwner,
)
from puripuly_heart.core.orchestrator.translation_diagnostics import (
    LatencyInheritanceDiagnostic,
    LatencyStageDiagnostic,
    OverlayEmitDiagnostic,
    SelfOverlayDecisionDiagnostic,
    SttEventLoopFailureDiagnostic,
    TranslationLatencyDiagnosticsOwner,
)
from puripuly_heart.core.osc.chatbox_paginator import ChatboxPaginator
from puripuly_heart.domain.models import OSCMessage
from tests.helpers.fakes import FakeSender


class RuntimeLogging:
    def __init__(self) -> None:
        self.basic: list[str] = []
        self.detailed: list[str] = []

    def emit_basic(self, message: str, *, level: int = 20) -> None:
        _ = level
        self.basic.append(message)

    def emit_diagnostic(self, message: str, *, level: int = 20) -> bool:
        _ = level
        self.detailed.append(message)
        return True

    def emit_diagnostic_lazy(self, build_message, *, level: int = 20) -> bool:
        _ = level
        self.detailed.append(build_message())
        return True


class OverlayDiagnostics:
    def __init__(self) -> None:
        self.records: list[tuple[str, dict[str, object]]] = []

    def record_translation(self, event: str, **fields: object) -> None:
        self.records.append((event, fields))


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


def test_owner_summarizes_actual_delayed_chatbox_send_from_last_speech() -> None:
    clock = FakeClock(_now=10.0)
    logging = RuntimeLogging()
    configuration = TranslationRuntimeConfig(hangover_s=0.4)
    owner = TranslationLatencyDiagnosticsOwner(
        clock=clock,
        config_snapshot=lambda: TranslationRuntimeConfigurationOwner(configuration).snapshot(),
        runtime_logging=logging,
    )
    paginator = ChatboxPaginator(
        sender=FakeSender(),
        clock=clock,
        max_chars=4,
        page_interval_s=3.0,
        stage_recorder=owner.record_chatbox_stage,
    )
    paginator.enqueue(OSCMessage(utterance_id=uuid4(), text="abcdefgh", created_at=clock.now()))
    utterance_id = uuid4()
    owner.record_latency_stage(
        LatencyStageDiagnostic(
            channel="self",
            utterance_id=utterance_id,
            stage="last_speech",
            timestamp=8.0,
        )
    )
    owner.record_latency_stage(
        LatencyStageDiagnostic(
            channel="self",
            utterance_id=utterance_id,
            stage="speech_end",
            timestamp=9.0,
        )
    )
    paginator.enqueue(OSCMessage(utterance_id=utterance_id, text="done", created_at=clock.now()))
    owner.retain_latency_until_output("self", utterance_id)
    owner.clear_latency_timeline("self", utterance_id)

    assert logging.basic == []

    configuration = replace(configuration, hangover_s=9.0)
    clock.advance(3.0)
    paginator.process_due()

    assert logging.basic == [
        "[Basic][Latency] channel=self endpoint=chatbox_send last_speech_to_chatbox_send_ms=5000"
    ]
    assert owner.snapshot().timeline_keys == frozenset()


def test_first_successful_page_keeps_latency_after_earlier_page_failures() -> None:
    class SelectiveSender(FakeSender):
        def send_chatbox(self, text: str) -> None:
            if text != "ijkl":
                raise OSError("send failed")
            super().send_chatbox(text)

    clock = FakeClock(_now=10.0)
    logging = RuntimeLogging()
    owner = make_owner(clock=clock, runtime_logging=logging)
    sender = SelectiveSender()
    paginator = ChatboxPaginator(
        sender=sender, clock=clock, max_chars=4, stage_recorder=owner.record_chatbox_stage
    )
    utterance_id = uuid4()
    owner.record_latency_stage(
        LatencyStageDiagnostic(
            channel="self", utterance_id=utterance_id, stage="last_speech", timestamp=9.0
        )
    )
    owner.retain_latency_until_output("self", utterance_id)
    paginator.enqueue(OSCMessage(utterance_id=utterance_id, text="abcdefghijkl", created_at=10.0))
    owner.clear_latency_timeline("self", utterance_id)
    clock.advance(3.0)
    paginator.process_due()
    assert logging.basic == []
    clock.advance(3.0)
    paginator.process_due()

    assert sender.sent == ["ijkl"]
    assert len(logging.basic) == 1
    assert "last_speech_to_chatbox_send_ms=7000" in logging.basic[0]
    assert owner.snapshot().timeline_keys == frozenset()

    failed_id = uuid4()
    owner.record_latency_stage(
        LatencyStageDiagnostic(
            channel="self", utterance_id=failed_id, stage="last_speech", timestamp=16.0
        )
    )
    owner.retain_latency_until_output("self", failed_id)
    paginator.enqueue(OSCMessage(utterance_id=failed_id, text="abcdefgh", created_at=16.0))
    clock.advance(3.0)
    paginator.process_due()
    assert len(logging.basic) == 1
    assert owner.snapshot().timeline_keys == frozenset()


@pytest.mark.parametrize("same_parent", [False, True])
def test_pending_self_replacement_retires_only_superseded_timing(same_parent: bool) -> None:
    clock = FakeClock(_now=10.0)
    logging = RuntimeLogging()
    owner = make_owner(clock=clock, runtime_logging=logging)
    sender = FakeSender()
    paginator = ChatboxPaginator(
        sender=sender, clock=clock, max_chars=4, stage_recorder=owner.record_chatbox_stage
    )
    paginator.enqueue(OSCMessage(utterance_id=uuid4(), text="abcdefgh", created_at=10.0))
    old_id = uuid4()
    new_id = old_id if same_parent else uuid4()
    for revision, utterance_id, text in ((1, old_id, "old"), (2, new_id, "new")):
        owner.record_latency_stage(
            LatencyStageDiagnostic(
                channel="self", utterance_id=utterance_id, stage="last_speech", timestamp=9.0
            )
        )
        owner.retain_latency_until_output("self", utterance_id)
        paginator.enqueue(
            OSCMessage(
                utterance_id=utterance_id,
                text=text,
                created_at=10.0,
                self_speech=True,
                turn_generation=1,
                turn_order=1 if same_parent else revision,
                presentation_revision=revision,
            )
        )
        owner.clear_latency_timeline("self", utterance_id)
    clock.advance(3.0)
    paginator.process_due()

    assert sender.sent == ["abcd", "efgh", "new"]
    assert len(logging.basic) == 1
    assert "last_speech_to_chatbox_send_ms=4000" in logging.basic[0]
    assert owner.snapshot().timeline_keys == frozenset()


def test_late_source_end_reaches_committed_outputs_after_source_cleanup() -> None:
    clock = FakeClock(_now=13.0)
    logging = RuntimeLogging()
    owner = make_owner(clock=clock, runtime_logging=logging)
    source_id, publication_id, first_id, second_id = (uuid4() for _ in range(4))
    owner.record_latency_stage(
        LatencyStageDiagnostic("self", source_id, "stt_final", timestamp=9.0)
    )
    owner.inherit_latency(LatencyInheritanceDiagnostic("self", publication_id, (source_id,)))
    for output_id in (first_id, second_id):
        owner.inherit_latency(LatencyInheritanceDiagnostic("self", output_id, (publication_id,)))
        owner.retain_latency_until_output("self", output_id)
        owner.clear_latency_timeline("self", output_id)
    owner.clear_latency_timeline("self", publication_id)
    owner.clear_latency_timeline("self", source_id)
    owner.record_output_latency_stage(
        LatencyStageDiagnostic("self", first_id, "self_chatbox_send", timestamp=13.0)
    )
    assert logging.basic == []
    owner.record_latency_stage(
        LatencyStageDiagnostic("self", source_id, "last_speech", timestamp=9.5, publish_now=False)
    )
    owner.record_latency_stage(
        LatencyStageDiagnostic("self", source_id, "speech_end", timestamp=10.1)
    )
    owner.record_output_latency_stage(
        LatencyStageDiagnostic("self", second_id, "self_chatbox_send", timestamp=15.0)
    )

    assert len(logging.basic) == 2
    assert "last_speech_to_chatbox_send_ms=3500" in logging.basic[0]
    assert "last_speech_to_chatbox_send_ms=5500" in logging.basic[1]
    assert owner.snapshot().timeline_keys == frozenset()


@pytest.mark.parametrize("late_last_speech,expected_ms", [(9.5, 500), (11.0, None)])
def test_merged_output_waits_for_latest_source_without_fabricating_zero_latency(
    late_last_speech: float, expected_ms: int | None
) -> None:
    logging = RuntimeLogging()
    owner = make_owner(runtime_logging=logging)
    first_id, late_id, output_id = (uuid4() for _ in range(3))
    owner.record_latency_stage(LatencyStageDiagnostic("self", late_id, "stt_final", timestamp=9.0))
    owner.record_latency_stage(
        LatencyStageDiagnostic("self", first_id, "last_speech", timestamp=8.0)
    )
    owner.record_latency_stage(
        LatencyStageDiagnostic("self", first_id, "speech_end", timestamp=8.5)
    )
    owner.inherit_latency(LatencyInheritanceDiagnostic("self", output_id, (first_id, late_id)))
    owner.retain_latency_until_output("self", output_id)
    owner.clear_latency_timeline("self", first_id)
    owner.clear_latency_timeline("self", late_id)
    owner.clear_latency_timeline("self", output_id)
    owner.record_output_latency_stage(
        LatencyStageDiagnostic("self", output_id, "self_chatbox_send", timestamp=10.0)
    )
    assert logging.basic == []
    owner.record_latency_stage(
        LatencyStageDiagnostic(
            "self", late_id, "last_speech", timestamp=late_last_speech, publish_now=False
        )
    )
    owner.record_latency_stage(
        LatencyStageDiagnostic("self", late_id, "speech_end", timestamp=late_last_speech + 0.5)
    )

    if expected_ms is None:
        assert logging.basic == []
    else:
        assert len(logging.basic) == 1
        assert f"last_speech_to_chatbox_send_ms={expected_ms}" in logging.basic[0]
    assert owner.snapshot().timeline_keys == frozenset()


def test_owner_does_not_measure_output_without_last_speech_origin() -> None:
    logging = RuntimeLogging()
    owner = make_owner(runtime_logging=logging)
    utterance_id = uuid4()
    owner.record_latency_stage(
        LatencyStageDiagnostic(
            channel="self",
            utterance_id=utterance_id,
            stage="speech_end",
            timestamp=10.0,
        )
    )
    owner.record_output_latency_stage(
        LatencyStageDiagnostic(
            channel="self",
            utterance_id=utterance_id,
            stage="self_chatbox_send",
            timestamp=10.5,
        )
    )

    assert logging.basic == []


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


def test_owner_suppresses_duplicate_overlay_decisions() -> None:
    overlay = OverlayDiagnostics()
    owner = make_owner(overlay_diagnostics=overlay)
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

    assert [event for event, _fields in overlay.records] == ["active_self_secondary"]


def test_owner_does_not_suppress_changed_overlay_text_with_same_length() -> None:
    overlay = OverlayDiagnostics()
    owner = make_owner(overlay_diagnostics=overlay)
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

    assert "private speech" not in caplog.text
    assert any(record.levelname == "ERROR" for record in caplog.records)
    assert "secret-token" not in caplog.text


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
