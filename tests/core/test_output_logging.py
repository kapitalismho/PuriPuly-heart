from __future__ import annotations

import asyncio
import io
import json
import logging
import socket
from uuid import uuid4

import pytest
from pythonosc.osc_message import OscMessage

from puripuly_heart.app.services.manual_typing import ManualTypingOwner
from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.orchestrator.self_translation_channel import TextSubmissionAdmission
from puripuly_heart.core.osc.chatbox_paginator import ChatboxPaginator
from puripuly_heart.core.osc.udp_sender import VrchatOscUdpSender
from puripuly_heart.core.overlay.sink import (
    OverlayApplicationReceipt,
    OverlayEventAdapter,
    UtteranceClosed,
)
from puripuly_heart.core.runtime.output import OutputRuntime
from puripuly_heart.core.runtime_logging import RuntimeLoggingSinks, SessionRuntimeLoggingService
from tests.helpers.fakes import FakeSender, RecordingOscQueue
from tests.helpers.translation_owners import compose_translation_test_harness


@pytest.fixture
def output_logs(tmp_path, monkeypatch):
    live = io.StringIO()
    log_file = tmp_path / "output.log"
    stream_handler = logging.StreamHandler(live)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    root = logging.getLogger("puripuly_heart")
    monkeypatch.setattr(root, "handlers", [])
    monkeypatch.setattr(root, "propagate", False)
    monkeypatch.setattr(root, "level", logging.INFO)
    session = logging.getLogger(f"test.output.logging.{uuid4().hex}")
    service = SessionRuntimeLoggingService(
        root_logger=root,
        session_logger=session,
        sinks=RuntimeLoggingSinks(
            stream_handler=stream_handler,
            file_handler=file_handler,
            log_file=log_file,
        ),
    )
    service.configure_diagnostics(instance_id="output-logging-test")
    try:
        yield service, live, log_file
    finally:
        service.close()
        file_handler.close()


def _packets(receiver: socket.socket) -> list[tuple[str, list[object]]]:
    packets = []
    while True:
        try:
            data, _ = receiver.recvfrom(65535)
        except TimeoutError:
            return packets
        message = OscMessage(data)
        packets.append((message.address, message.params))


def _event(channel: str, event_id: str) -> UtteranceClosed:
    return UtteranceClosed(
        event_id=event_id,
        seq=1,
        utterance_id=uuid4(),
        channel=channel,
        created_at=0.0,
        is_final=True,
    )


@pytest.mark.asyncio
async def test_manual_typing_has_no_publications_but_submission_reaches_udp_and_logs(output_logs):
    service, live, log_file = output_logs
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(0.01)
    sender = VrchatOscUdpSender(port=receiver.getsockname()[1])
    clock = FakeClock()
    output = OutputRuntime(
        chatbox=ChatboxPaginator(sender=sender, clock=clock, runtime_logging=service),
        clock=clock,
        routing_observer=service.record_output_routing_decision,
    )
    typing = ManualTypingOwner(
        output_provider=lambda: output,
        completion_provider=lambda _: None,
        log_diagnostic=service.emit_diagnostic,
        idle_timeout_seconds=10.0,
        submit_timeout_seconds=1.0,
    )
    try:
        typing.set_input_activity(True)
        first_idle = typing.idle_task
        for _ in range(9):
            typing.set_input_activity(True)
        assert typing.idle_task is not first_idle
        output.set_self_chatbox_typing_reason("self_speech_pending", True)
        submit_reason = typing.begin_submit()
        typing.set_input_activity(False)
        typing.clear_submit(submit_reason)
        output.set_self_chatbox_typing_reason("self_speech_pending", False)
        output.clear_self_chatbox_typing_reasons()

        assert _packets(receiver) == [
            ("/chatbox/typing", [True]),
            ("/chatbox/typing", [False]),
        ]
        assert output.routing_decisions == ()
        assert live.getvalue() == ""
        assert log_file.read_text(encoding="utf-8") == ""

        publication_id = uuid4()

        async def submit() -> TextSubmissionAdmission:
            await output.publish_chatbox(
                publication_id=publication_id,
                channel="self",
                transcript_text="synthetic manual message",
                translation_text=None,
                include_source=True,
            )
            return TextSubmissionAdmission("admitted", publication_id, ())

        admission = await typing.submit(submit)

        assert admission.status == "admitted"
        assert _packets(receiver) == [
            ("/chatbox/typing", [True]),
            ("/chatbox/input", ["synthetic manual message", True, False]),
            ("/chatbox/typing", [False]),
        ]
        assert len(output.routing_decisions) == 1
        decision = output.routing_decisions[0]
        assert decision.publication_id == str(publication_id)
        assert decision.metadata["stage"] == "paginator_handoff"
        assert decision.metadata["physical_ack"] is False
        for rendered in (live.getvalue(), log_file.read_text(encoding="utf-8")):
            assert len(rendered.splitlines()) == 1
            assert str(publication_id) in rendered
            assert "paginator_handoff" in rendered
            assert "physical_ack=false" in rendered
            assert "synthetic manual message" not in rendered
    finally:
        await typing.release()
        await output.close()
        sender.close()
        receiver.close()


@pytest.mark.asyncio
async def test_failed_udp_typing_is_visible_without_false_publication_and_retries(output_logs):
    service, live, log_file = output_logs

    class RetrySender(FakeSender):
        fail_next = True

        def send_typing(self, active: bool) -> None:
            if self.fail_next:
                self.fail_next = False
                raise OSError(5, "private send detail")
            super().send_typing(active)

    sender = RetrySender()
    clock = FakeClock()
    output = OutputRuntime(
        chatbox=ChatboxPaginator(sender=sender, clock=clock, runtime_logging=service),
        clock=clock,
        routing_observer=service.record_output_routing_decision,
    )
    output.set_self_chatbox_typing_reason("manual_input", True)
    output.set_self_chatbox_typing_reason("manual_input", True)
    sender.fail_next = True
    output.clear_self_chatbox_typing_reasons()
    output.clear_self_chatbox_typing_reasons()

    assert sender.typing == [True, False]
    assert output.routing_decisions == ()
    records = service.diagnostic_history({})["records"]
    assert [record["operation"] for record in records] == ["typing_reason", "typing_clear"]
    assert all(record["code"] == "osc_local_send_failed" for record in records)
    assert all(record["fields"]["remote_delivery"] == "unacknowledged" for record in records)
    assert all("publication_id" not in record["fields"] for record in records)
    for rendered in (live.getvalue(), log_file.read_text(encoding="utf-8")):
        assert len(rendered.splitlines()) == 2
        assert "destination_result" not in rendered
        assert "private send detail" not in rendered
    await output.close()


@pytest.mark.asyncio
async def test_normal_policy_decisions_remain_owned_without_rendered_noise(output_logs):
    service, live, log_file = output_logs
    clock = FakeClock()
    sender = FakeSender()
    output = OutputRuntime(
        chatbox=ChatboxPaginator(sender=sender, clock=clock, runtime_logging=service),
        clock=clock,
        activation_notice_enabled=False,
        routing_observer=service.record_output_routing_decision,
    )
    publication_id = uuid4()
    await output.publish_chatbox(
        publication_id=publication_id,
        channel="self",
        transcript_text="synthetic",
        translation_text=None,
        include_source=True,
    )
    await output.publish_chatbox(
        publication_id=publication_id,
        channel="self",
        transcript_text="synthetic",
        translation_text=None,
        include_source=True,
    )
    await output.publish_chatbox(
        publication_id=uuid4(),
        channel="peer",
        transcript_text="peer",
        translation_text=None,
        include_source=True,
    )
    output.publish_system_disclosure_chatbox(text="notice")
    await output.publish_overlay_event(_event("self", "optional-overlay"))

    assert sender.sent == ["synthetic"]
    assert [decision.reason for decision in output.routing_decisions] == [
        None,
        "duplicate_publication",
        "peer_chatbox_denied",
        "activation_notice_disabled",
        "destination_unconfigured",
    ]
    for rendered in (live.getvalue(), log_file.read_text(encoding="utf-8")):
        assert len(rendered.splitlines()) == 1
        assert str(publication_id) in rendered

    class RejectingSink:
        async def emit(self, event):
            return OverlayApplicationReceipt(
                stage="application_accepted",
                outcome="not_applied",
                publication_id=event.event_id,
                scene_revision=0,
                cause="unexpected_rejection",
            )

    await output.replace_overlay_sink(RejectingSink())
    await output.publish_overlay_event(_event("self", "unexpected"))
    await output.close()
    await output.publish_overlay_event(_event("self", "closed-misuse"))
    for rendered in (live.getvalue(), log_file.read_text(encoding="utf-8")):
        assert "unexpected_rejection" in rendered
        assert "output_runtime_closed" in rendered
        assert "backlog dropped" not in rendered


@pytest.mark.asyncio
async def test_peer_overlay_wait_is_file_only_and_terminal_application_remains_live(output_logs):
    service, live, log_file = output_logs
    started, release = asyncio.Event(), asyncio.Event()

    class PacedSink:
        async def emit_peer_when_admissible(self, event, *, on_wait):
            on_wait("protected_rows")
            started.set()
            await release.wait()
            return OverlayApplicationReceipt(
                stage="application_accepted",
                outcome="applied",
                publication_id=event.event_id,
                scene_revision=1,
            )

    output = OutputRuntime(
        chatbox=ChatboxPaginator(sender=FakeSender(), clock=FakeClock(), runtime_logging=service),
        overlay_sink=PacedSink(),
        routing_observer=service.record_output_routing_decision,
    )
    output.activate_peer_generation(1)
    try:
        await output.publish_overlay_event(
            OverlayEventAdapter().utterance_closed(
                utterance_id=uuid4(), channel="peer", is_final=True
            ),
            publication_generation=1,
            source_order=0,
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        assert live.getvalue() == ""
        pending_file = log_file.read_text(encoding="utf-8")
        assert len(pending_file.splitlines()) == 2
        assert "admission_accepted" in pending_file
        assert "logical_pacing" in pending_file
        assert "decision=published" not in pending_file

        release.set()
        await output.wait_for_peer_output_idle()
        assert [decision.reason for decision in output.routing_decisions] == [
            "accepted_handoff",
            "logical_pacing_wait",
            "application_applied",
        ]
        assert len(live.getvalue().splitlines()) == 1
        assert "application_applied" in live.getvalue()
        assert "physical_ack=false" in live.getvalue()
        assert len(log_file.read_text(encoding="utf-8").splitlines()) == 3
    finally:
        release.set()
        await output.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["self", "peer"])
async def test_overlay_failure_has_one_typed_cause_and_preserves_translation_state(
    output_logs, channel
):
    service, live, log_file = output_logs

    class FailedSink:
        async def emit(self, event):
            raise RuntimeError("private overlay payload")

    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        overlay_sink=FailedSink(),
        runtime_logging=service,
    )

    def observe(decision):
        service.record_output_routing_decision(decision)
        if decision.diagnostics is not None:
            raise ValueError("private observer detail")

    harness.output_runtime.routing_observer = observe
    event = _event(channel, f"failed-{channel}")
    if channel == "peer":
        harness.output_runtime.activate_peer_generation(1)
    try:
        await harness.output_projection.publish_overlay_event(
            event,
            publication_generation=1 if channel == "peer" else None,
            source_order=0 if channel == "peer" else None,
        )
        await harness.output_runtime.wait_for_peer_output_idle()
        records = service.diagnostic_history({})["records"]
        failures = [record for record in records if record["code"] == "destination_publish_failed"]
        assert len(failures) == 1
        failure = failures[0]
        decision = harness.output_runtime.routing_decisions[-1]
        assert decision.diagnostics.diagnostic_id == failure["diagnostic_id"]
        assert failure["fields"]["destination"] == "subtitle_overlay"
        assert failure["fields"]["exception_class"] == "RuntimeError"
        assert failure["channel"] == channel
        observer = next(record for record in records if record["code"] == "output_observer_failed")
        assert observer["cause_id"] == failure["diagnostic_id"]
        health = harness.output_runtime.diagnostic_delivery_health
        assert health["observer"] == "degraded"
        assert health["observer_failure_count"] == 1
        assert health["observer_failure_diagnostic_id"] == observer["diagnostic_id"]
        assert harness.translation_diagnostics.snapshot().last_error_source == "overlay_sink"
        assert len(live.getvalue().splitlines()) == 1
        persisted_failures = [
            json.loads(line.removeprefix("[Failure] "))
            for line in log_file.read_text(encoding="utf-8").splitlines()
            if line.startswith("[Failure] ")
        ]
        assert len(persisted_failures) == 2
        assert (
            sum(
                record["diagnostic_id"] == failure["diagnostic_id"] for record in persisted_failures
            )
            == 1
        )
        for rendered in (
            live.getvalue(),
            log_file.read_text(encoding="utf-8"),
            json.dumps(records),
        ):
            assert "destination_result" not in rendered
            assert "Traceback (most recent call last):" not in rendered
            assert "private overlay payload" not in rendered
            assert "private observer detail" not in rendered
    finally:
        await harness.output_runtime.close()


@pytest.mark.parametrize("has_backlog", [False, True])
def test_shutdown_backlog_logs_only_real_discarded_work(output_logs, has_backlog):
    service, live, log_file = output_logs
    clock = FakeClock()
    output = OutputRuntime(
        chatbox=ChatboxPaginator(
            sender=FakeSender(), clock=clock, max_chars=5, runtime_logging=service
        ),
        clock=clock,
    )
    if has_backlog:
        output.publish_system_disclosure_chatbox(text="abcdefghijk")
        output.publish_system_disclosure_chatbox(text="queued")
    output.chatbox.drop_pending()
    for rendered in (live.getvalue(), log_file.read_text(encoding="utf-8")):
        if has_backlog:
            assert len(rendered.splitlines()) == 1
            assert "pages=2 messages=1" in rendered
        else:
            assert rendered == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason",
    [
        "publication_generation_retired",
        "stale_source_order",
        "stale_retired",
        "stale",
        "preview_superseded",
        "destination_replaced",
        "cancelled_local",
        "output_runtime_closing",
    ],
)
async def test_expected_overlay_cancellation_is_retained_without_rendered_noise(
    output_logs, reason
):
    service, live, log_file = output_logs

    class CancelledSink:
        async def emit(self, event):
            return OverlayApplicationReceipt(
                stage="application_accepted",
                outcome="not_applied",
                publication_id=event.event_id,
                scene_revision=0,
                cause=reason,
            )

    output = OutputRuntime(
        chatbox=ChatboxPaginator(sender=FakeSender(), clock=FakeClock(), runtime_logging=service),
        overlay_sink=CancelledSink(),
        routing_observer=service.record_output_routing_decision,
    )
    try:
        result = await output.publish_overlay_event(_event("self", "cancelled"))
        assert result.decision.reason == reason
        assert output.routing_decisions == (result.decision,)
        assert live.getvalue() == ""
        assert log_file.read_text(encoding="utf-8") == ""
    finally:
        await output.close()


@pytest.mark.parametrize(
    "field",
    [
        "decision=private_value",
        "route=private_value",
        "reason=private_value",
        "stage=private_value",
        "outcome=private_value",
        "physical_ack=true",
        "wait_reason=private_value",
        "pending_batches=private_value",
        "private_field=private_value",
    ],
)
def test_file_only_output_stages_reject_untrusted_metadata(output_logs, field):
    service, live, log_file = output_logs
    service.emit_diagnostic(f"[Output] destination_stage {field}")
    persisted = log_file.read_text(encoding="utf-8")
    assert live.getvalue() == ""
    assert "untrusted_record_redacted" in persisted
    assert field not in persisted
