from __future__ import annotations

import json
import asyncio
from dataclasses import dataclass, replace
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest

from puripuly_heart.app.ports.agent_chat import AgentServiceError, AgentTransportError
from puripuly_heart.app.services.agent_chat.context import AgentActivity, collect_agent_context
from puripuly_heart.app.services.agent_chat.execution import project_query, safe_receipt
from puripuly_heart.app.services.agent_chat.modules import model_context, model_receipt, operation_header
from puripuly_heart.app.services.agent_chat.owner import AgentChatOwner
from puripuly_heart.app.services.agent_chat.report import build_support_bundle, recent_diagnostics
from puripuly_heart.app.services.agent_chat.tools import AgentTools, project_operation_receipt
from puripuly_heart.core.diagnostics import (
    DIAGNOSTIC_CAUSE_MAX_COUNT,
    DiagnosticHistory,
    exception_diagnostics,
    serialize_diagnostics,
)
from puripuly_heart.core.llm.fallback_racing import LLMProviderRaceError
from puripuly_heart.core.messages import ErrorDiagnostics
from puripuly_heart.domain.models import Translation
from tests.app.test_agent_chat import Records, Responses, answer
from tests.app.test_agent_chat_report import contents
from tests.app.test_translation_probe import ControlledBackend, control_for, submit_probe

PRIVATE = "PRIVATE-CREDENTIAL-BODY-CONTENT-C:/Users/private"


def failure(*, causes=()):
    return ErrorDiagnostics(
        component="translation_request", operation="translation_probe", code="provider.auth",
        category="auth", visibility="diagnostic_only", content_policy="metadata_only",
        status_code=403, retry_after_ms=None,
        fields={"provider": "openrouter", "body": PRIVATE, "path": PRIVATE}, causes=causes,
    )


class HistorySink:
    def __init__(self, instance_id):
        self.history = DiagnosticHistory()
        self.history.bind_instance(instance_id)

    def emit_failure(self, diagnostic, **kwargs):
        return self.history.retain(diagnostic, **kwargs)[0]

    def diagnostic_history(self, arguments):
        return self.history.query(arguments)

    def diagnostic_health(self):
        return self.history.health()


@pytest.mark.asyncio
async def test_compound_probe_cause_survives_receipt_history_tools_summary_and_bundle():
    errors = tuple(
        httpx.HTTPStatusError(
            PRIVATE,
            request=httpx.Request("POST", "https://example.invalid/private", headers={"authorization": PRIVATE}),
            response=httpx.Response(status, json={"message": PRIVATE}),
        )
        for status in (403, 403, 404)
    )
    backend = ControlledBackend(failure=LLMProviderRaceError(errors))
    control, runtime = control_for(backend)
    sink = HistorySink(control.instance_id)
    control.application._runtime_logging = sink
    try:
        receipt = await submit_probe(control)
        assert receipt["status"] == "failed" and receipt["terminal"]
        cause = receipt["diagnostics"]
        assert cause["code"] == "provider.compound_failure"
        assert cause["disposition"] == "failed"
        assert [item["status_code"] for item in cause["causes"]] == [403, 403, 404]
        assert cause["operation_id"] == receipt["operation_id"]
        assert cause["request_id"] == receipt["request_id"]
        assert cause["turn_id"] == str(backend.requests[0].utterance_id)
        assert receipt["target_results"][0]["diagnostic_id"] == cause["diagnostic_id"]
        history = await control.query("diagnostics.history", {"operation_id": receipt["operation_id"]})
        assert len(history["records"]) == 1
        assert history["records"][0] == cause
        context = await collect_agent_context(
            control, activity=AgentActivity(), topic="diagnostics", operation_id=receipt["operation_id"]
        )
        tool_receipt = project_operation_receipt(receipt, "probe_translation")
        summary = operation_header({"tool": "probe_translation", "result": {"receipt": tool_receipt}})
        direct = safe_receipt(receipt)
        read = await project_query("diagnostics.history", history, control, str)
        evidence = await recent_diagnostics(control)
        bundle = build_support_bundle(
            ticket="PP-0123456789ABCDEF", app_version="synthetic", summary="Synthetic failure",
            diagnostics=evidence, archive=(), log_snapshot=None,
        )
        meta = json.loads(contents(bundle, missing_log=True)["meta.json"])
        projected = [
            context["recent_diagnostics"][0], tool_receipt["diagnostics"],
            summary["diagnostics"], direct["diagnostics"], read["records"][0],
            meta["diagnostics"]["records"][0],
        ]
        for value in projected:
            assert value["diagnostic_id"] == cause["diagnostic_id"]
            assert value["operation_id"] == receipt["operation_id"]
            assert [item["status_code"] for item in value["causes"]] == [403, 403, 404]
        assert PRIVATE not in json.dumps([receipt, history, context, projected, meta])
        assert "configured_provider" not in receipt and "model" not in cause["fields"]
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_failed_sink_keeps_original_failure_and_truthful_unretained_reference():
    original = failure()
    error = RuntimeError(PRIVATE)
    error.failure_diagnostics = original
    control, runtime = control_for(ControlledBackend(failure=error))

    def broken(*args, **kwargs):
        raise OSError(PRIVATE)

    control.application._runtime_logging = SimpleNamespace(emit_failure=broken)
    try:
        receipt = await submit_probe(control)
        assert receipt["status"] == "failed"
        assert receipt["error"]["code"] == "translation_probe_failed"
        assert receipt["diagnostics"]["diagnostic_id"] == original.diagnostic_id
        assert receipt["diagnostics"]["status_code"] == 403
        assert receipt["diagnostics"]["disposition"] == "failed"
        assert receipt["diagnostics"]["file_evidence"] == "unavailable"
        assert "diagnostic_sequence" not in receipt["diagnostics"]
        assert PRIVATE not in json.dumps(receipt)
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_compound_overflow_is_not_reported_as_absence_and_missing_history_is_explicit():
    original = replace(failure(), code="provider.compound_failure", causes=tuple(failure() for _ in range(100)))
    error = RuntimeError(PRIVATE)
    error.failure_diagnostics = original
    control, runtime = control_for(ControlledBackend(failure=error))
    try:
        receipt = await submit_probe(control)
        assert receipt["diagnostics"]["causes"]
        assert receipt["diagnostics"]["causes_truncated"] is True
        assert receipt["diagnostics"]["detail"]["truncated"] is True
        assert len(json.dumps(receipt["diagnostics"], separators=(",", ":")).encode()) <= 4096
        records = Records()
        empty = await recent_diagnostics(records)
        assert empty["availability"] == "retained_current_run" and empty["records"] == []
        records.fail.add("diagnostics.history")
        unavailable = await recent_diagnostics(records)
        assert unavailable == {"availability": "query_failed", "records": []}
        bundle = build_support_bundle(
            ticket="PP-0123456789ABCDEF", app_version="synthetic", summary="Synthetic failure",
            diagnostics=unavailable, archive=(), log_snapshot=None,
        )
        meta = json.loads(contents(bundle, missing_log=True)["meta.json"])
        assert meta["missing_material"]["diagnostics"] == "query_failed"
        assert PRIVATE not in json.dumps([receipt, meta])
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_successful_fallback_keeps_recovered_cause_without_failed_probe():
    recovered = failure()

    class RecoveredBackend(ControlledBackend):
        async def translate(self, request):
            self.requests.append(request)
            return Translation(
                utterance_id=request.utterance_id, text="Synthetic translated text",
                recovered_failures=(serialize_diagnostics(recovered),),
            )

    control, runtime = control_for(RecoveredBackend())
    sink = HistorySink(control.instance_id)
    control.application._runtime_logging = sink
    try:
        receipt = await submit_probe(control)
        assert receipt["status"] == "applied" and receipt["translation_result"] == "succeeded"
        assert "diagnostics" not in receipt and "error" not in receipt
        (record,) = receipt["recovered_diagnostics"]
        assert record["diagnostic_id"] == recovered.diagnostic_id
        assert record["disposition"] == "recovered"
        assert sink.diagnostic_history({})["records"] == [record]
        projected = project_operation_receipt(receipt, "probe_translation")
        assert projected["recovered_diagnostics"][0]["diagnostic_id"] == recovered.diagnostic_id
        assert PRIVATE not in json.dumps(projected)
    finally:
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error,code",
    [
        (AgentTransportError(PRIVATE), "agent_transport_failed"),
        (AgentServiceError("service_allowance_exhausted"), "agent_service_failed"),
    ],
)
async def test_chat_response_failure_returns_safe_typed_cause_without_changing_allowance(error, code):
    owner = AgentChatOwner(Records(), Responses(error), localize=str)
    await owner.start()
    try:
        result = await owner.send("Synthetic application question")
        assert result.status == ("service_unavailable" if isinstance(error, AgentServiceError) else "failed")
        assert isinstance(result.diagnostics, ErrorDiagnostics)
        assert result.diagnostics.code == code
        assert owner._allowance_exhausted is isinstance(error, AgentServiceError)
        serialized = serialize_diagnostics(result.diagnostics)
        assert PRIVATE not in json.dumps(serialized)
        if isinstance(error, AgentServiceError):
            assert serialized["fields"]["service_code"] == "service_allowance_exhausted"
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_verification_failure_notice_preserves_safe_cause_without_automatic_request():
    transport = Responses(answer("Synthetic diagnosis"))
    owner = AgentChatOwner(Records(), transport, localize=str)
    owner.bind_verification_providers(frozenset({"openai"}))
    await owner.start()
    try:
        owner._watched_provider = "openai"
        original = failure()
        owner.key_verification_completed("openai", False, "verification_failed", original)
        await asyncio.sleep(0)
        assert not transport.requests and not transport.classifications
        result = await owner.send("What was the verification result?")
        assert result.status == "completed"
        model_input = json.dumps(transport.requests[-1]["input"])
        assert original.diagnostic_id in model_input and "provider.auth" in model_input
        assert "403" in model_input and PRIVATE not in model_input
        assert "failed" in model_input and "configured_provider" not in model_input
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("raises", (False, True))
async def test_recognition_primary_survives_failed_cleanup_with_retained_session(raises):
    from puripuly_heart.core.stt.backend import STTProviderTurnTerminal
    from puripuly_heart.core.stt.recognition_probe import recognize_probe_audio
    from tests.core.test_recognition_probe_sessions import WATCHDOGS
    from tests.core.test_stt_scoped_engine import settings

    original = failure()
    error = RuntimeError(PRIVATE)
    error.failure_diagnostics = original

    class Session:
        async def begin_turn(self, request):
            self.request = request

        async def send_turn_audio(self, *args, **kwargs):
            pass

        async def seal_turn(self, *args, **kwargs):
            pass

        async def close(self):
            raise OSError(PRIVATE)

        async def turn_events(self):
            if raises:
                raise error
            yield STTProviderTurnTerminal(
                self.request.identity, "failed", "", failure_diagnostics=original
            )

    session = Session()

    async def factory(snapshot, epoch):
        return session, "deepgram"

    retained = []
    if raises:
        with pytest.raises(RuntimeError) as raised:
            await recognize_probe_audio(
                factory, settings("deepgram"), "self", b"\\x01\\x00" * 160, WATCHDOGS, retained
            )
        assert raised.value is error
        secondary = raised.value.secondary_diagnostics
    else:
        result = await recognize_probe_audio(
            factory, settings("deepgram"), "self", b"\\x01\\x00" * 160, WATCHDOGS, retained
        )
        assert result.result == "failed" and result.diagnostics is original
        assert result.cleanup_complete is False
        secondary = result.secondary_diagnostics
    assert retained == [session]
    assert len(secondary) == 1 and secondary[0].code == "recognition_probe_cleanup_failed"
    assert PRIVATE not in json.dumps([serialize_diagnostics(item) for item in secondary])


def test_control_error_event_keeps_failure_when_recorder_fails():
    from puripuly_heart.app.services.application_control_events import ControlEvents
    from puripuly_heart.core.error_messages import provider_failure_report
    from puripuly_heart.domain.events import UIEvent, UIEventType

    original = replace(failure(), causes=(failure(),))

    def broken(*args, **kwargs):
        raise OSError(PRIVATE)

    events = ControlEvents(instance_id="synthetic-event", diagnostic_recorder=broken)
    events.publish_ui(
        UIEvent(
            UIEventType.ERROR,
            payload=provider_failure_report(original, provider="translation", operation="translation_probe"),
            channel="self", utterance_id=uuid4(),
        )
    )
    (event,) = events.history
    assert event["diagnostic_id"] == original.diagnostic_id
    assert event["error"]["diagnostics"]["causes"][0]["status_code"] == 403
    assert event["error"]["diagnostics"]["file_evidence"] == "unavailable"
    assert PRIVATE not in json.dumps(event)


@pytest.mark.asyncio
async def test_audio_diagnosis_keeps_failed_route_and_successful_route_in_shared_consumers(monkeypatch):
    from puripuly_heart.app.services import audio_test
    from tests.app.test_audio_diagnosis import candidate, diagnose
    from tests.app.test_audio_test_session_owner import FeedSource, Harness, Opener

    monkeypatch.setattr(audio_test, "AUDIO_DIAGNOSE_CANDIDATE_TIMEOUT_S", 0.025)
    monkeypatch.setattr(audio_test, "DIAGNOSTIC_CLEANUP_TIMEOUT_S", 0.1)
    harness = Harness()
    source = FeedSource()
    source.feed(0.5, chunks=4)
    result = await diagnose(
        harness,
        (candidate(0, current=True), candidate(1)),
        {0: Opener(PermissionError(13, PRIVATE), {}), 1: Opener(source, {})},
    )
    assert result["status"] == "applied" and result["settings_changed"] is False
    failed, recovered = result["diagnosis"]["candidates"]
    assert failed["failure_code"] == "source_open_failed"
    assert failed["diagnostics"]["fields"]["exception_class"] == "PermissionError"
    assert failed["diagnostics"]["fields"]["errno"] == 13
    assert recovered["speech_detected"] and recovered["cleanup_complete"]
    assert result["recognition"] == "not_exercised"
    secondary = replace(failure(), code="provider.cleanup_failed", category="lifecycle")
    recovered_cause = failure()
    failed["secondary_diagnostics"] = [serialize_diagnostics(secondary)]
    failed["recovery_diagnostics"] = [serialize_diagnostics(recovered_cause)]
    failed["secondary_truncated"] = True
    failed["secondary_omitted_count"] = 2
    failed["secondary_diagnostics_truncated"] = True
    failed["recovery_diagnostics_truncated"] = True
    control, runtime = control_for(ControlledBackend())
    sink = HistorySink(control.instance_id)
    control.application._runtime_logging = sink
    try:
        operation_id, request_id = uuid4().hex, uuid4().hex
        control._retain_audio_diagnosis(result, operation_id, request_id)
        projected = project_operation_receipt(result, "diagnose_microphone")
        retained = projected["diagnosis"]["candidates"][0]["diagnostics"]
        assert retained["diagnostic_id"] == failed["diagnostics"]["diagnostic_id"]
        assert retained["operation_id"] == operation_id and retained["request_id"] == request_id
        history = sink.diagnostic_history({"diagnostic_id": retained["diagnostic_id"]})
        assert len(history["records"]) == 1
        assert history["records"][0]["fields"]["exception_class"] == "PermissionError"
        reply = model_receipt({"status": result["status"], "receipt": projected})
        action = {"tool": "diagnose_microphone", "result": reply}
        summary = operation_header(action)
        context = model_context(
            {"recent_agent_actions": [action], "agent_operations": [action]}, modules=("core",)
        )
        for view in (
            reply["receipt"], summary, context["core"]["recent_actions"][0],
            context["core"]["operations"][0],
        ):
            candidate_view = view["diagnosis"]["candidates"][0]
            assert candidate_view["diagnostic_id"] == retained["diagnostic_id"]
            assert candidate_view["diagnostics"]["diagnostic_id"] == retained["diagnostic_id"]
            assert candidate_view["diagnostics"]["fields"]["exception_class"] == "PermissionError"
            assert candidate_view["diagnostics"]["fields"]["errno"] == 13
            assert candidate_view["diagnostics"]["operation_id"] == operation_id
            assert candidate_view["diagnostics"]["request_id"] == request_id
            assert candidate_view["failure_code"] == "source_open_failed"
            assert candidate_view["secondary_diagnostics"][0]["diagnostic_id"] == secondary.diagnostic_id
            assert candidate_view["recovery_diagnostics"][0]["diagnostic_id"] == recovered_cause.diagnostic_id
            assert candidate_view["secondary_truncated"] is True
            assert candidate_view["secondary_diagnostics_truncated"] is True
            assert candidate_view["recovery_diagnostics_truncated"] is True
            assert view["diagnosis"]["selected_candidate_id"] == recovered["id"]
            assert view["status"] == "applied"
            assert "observed_speech_probabilities" not in json.dumps(view)
        assert PRIVATE not in json.dumps([reply, summary, context])
        assert PRIVATE not in json.dumps([result, projected, history])
    finally:
        await harness.owner.stop()
        await runtime.close()


@pytest.mark.asyncio
async def test_generic_operation_sink_failure_keeps_original_exception_and_cause(monkeypatch):
    import puripuly_heart.app.services.application_control as application_control

    original = failure()
    error = RuntimeError(PRIVATE)
    error.failure_diagnostics = original

    async def fail_probe(*args, **kwargs):
        raise error

    def broken_sink(*args, **kwargs):
        raise OSError("PRIVATE-SINK")

    monkeypatch.setattr(application_control, "run_translation_probe", fail_probe)
    control, runtime = control_for(ControlledBackend())
    control.application._runtime_logging = SimpleNamespace(emit_failure=broken_sink)
    try:
        receipt = await submit_probe(control)
        assert receipt["terminal"] and receipt["status"] == "failed"
        assert receipt["error"] == {
            "code": "application_operation_failed", "type": "RuntimeError"
        }
        assert receipt["cause_diagnostics"]["diagnostic_id"] == original.diagnostic_id
        assert receipt["diagnostics"]["causes"][0]["diagnostic_id"] == original.diagnostic_id
        assert receipt["diagnostics"]["operation_id"] == receipt["operation_id"]
        assert receipt["diagnostics"]["request_id"] == receipt["request_id"]
        assert receipt["diagnostics"]["file_evidence"] == "unavailable"
        assert PRIVATE not in json.dumps(receipt)
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_presentation_secondary_survives_operation_without_transaction():
    presentation = replace(
        failure(), component="application_notification", operation="snapshot_notification",
        code="presentation_sync_failed", category="lifecycle", status_code=None, fields={},
    )

    class Backend(ControlledBackend):
        async def translate(self, request):
            control.results.add_secondary(presentation)
            return Translation(utterance_id=request.utterance_id, text="Synthetic translation")

    control, runtime = control_for(Backend())
    try:
        receipt = await submit_probe(control)
        assert receipt["status"] == "applied" and "transaction" not in receipt
        assert receipt["secondary_diagnostics"][0]["diagnostic_id"] == presentation.diagnostic_id
        assert receipt["secondary_omitted_count"] == 0
        assert receipt["secondary_truncated"] is False
        projected = project_operation_receipt(receipt, "probe_translation")
        assert projected["secondary_diagnostics"][0]["code"] == "presentation_sync_failed"
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_recognition_cleanup_cancellation_keeps_known_failure_without_failed_outcome():
    from puripuly_heart.core.stt.recognition_probe import recognize_probe_audio
    from tests.core.test_recognition_probe_sessions import WATCHDOGS
    from tests.core.test_stt_scoped_engine import settings

    original = failure()
    error = RuntimeError(PRIVATE)
    error.failure_diagnostics = original
    closing, release = asyncio.Event(), asyncio.Event()

    class Session:
        async def begin_turn(self, request):
            pass

        async def send_turn_audio(self, *args, **kwargs):
            pass

        async def seal_turn(self, *args, **kwargs):
            pass

        async def close(self):
            closing.set()
            await release.wait()

        async def turn_events(self):
            raise error
            yield

    session = Session()

    async def factory(snapshot, epoch):
        return session, "deepgram"

    retained = []
    task = asyncio.create_task(
        recognize_probe_audio(
            factory, settings("deepgram"), "self", b"\\x01\\x00" * 160, WATCHDOGS, retained
        )
    )
    try:
        await asyncio.wait_for(closing.wait(), 1)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError) as raised:
            await task
        diagnostics = serialize_diagnostics(raised.value.failure_diagnostics)
        assert diagnostics["causes"][0]["diagnostic_id"] == original.diagnostic_id
        assert retained == []
        assert PRIVATE not in json.dumps(diagnostics)
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_successful_probe_keeps_cleanup_secondaries_distinct_from_recovered_attempts():
    class Backend(ControlledBackend):
        def __init__(self):
            super().__init__()
            self.recovered = []
            self.secondary = []

        async def translate(self, request):
            self.requests.append(request)
            recovered = failure()
            secondary = replace(
                failure(), code="provider.cleanup_failed", category="lifecycle",
                operation="cleanup", status_code=None,
            )
            self.recovered.append(recovered)
            self.secondary.append(secondary)
            return Translation(
                utterance_id=request.utterance_id,
                text="Synthetic translation",
                recovered_failures=(serialize_diagnostics(recovered),),
                secondary_diagnostics=(serialize_diagnostics(secondary),),
            )

    backend = Backend()
    control, runtime = control_for(backend, secondary_target_language="ja")
    sink = HistorySink(control.instance_id)
    control.application._runtime_logging = sink
    try:
        receipt = await submit_probe(control)
        assert receipt["status"] == "applied" and receipt["translation_result"] == "succeeded"
        assert "error" not in receipt and "diagnostics" not in receipt
        assert [item["diagnostic_id"] for item in receipt["recovered_diagnostics"]] == [
            item.diagnostic_id for item in backend.recovered
        ]
        assert [item["diagnostic_id"] for item in receipt["secondary_diagnostics"]] == [
            item.diagnostic_id for item in backend.secondary
        ]
        assert {item["disposition"] for item in receipt["recovered_diagnostics"]} == {"recovered"}
        assert {item["disposition"] for item in receipt["secondary_diagnostics"]} == {"degraded"}
        for target, request in zip(receipt["target_results"], backend.requests, strict=True):
            assert target["secondary_diagnostics"][0]["turn_id"] == str(request.utterance_id)
        projected = project_operation_receipt(receipt, "probe_translation")
        assert [item["code"] for item in projected["secondary_diagnostics"]] == [
            "provider.cleanup_failed", "provider.cleanup_failed"
        ]
        assert len(sink.diagnostic_history({})["records"]) == 4
        assert PRIVATE not in json.dumps([receipt, projected])
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_probe_preserves_empty_but_truncated_producer_collections_across_targets():
    class Backend(ControlledBackend):
        async def translate(self, request):
            return Translation(
                utterance_id=request.utterance_id,
                text="Synthetic translation",
                recovered_failures_truncated=request.target_language == "en",
                secondary_diagnostics_truncated=request.target_language == "en",
            )

    control, runtime = control_for(Backend(), secondary_target_language="ja")
    try:
        receipt = await submit_probe(control)
        assert receipt["status"] == "applied" and receipt["translation_result"] == "succeeded"
        assert receipt["recovered_diagnostics"] == receipt["secondary_diagnostics"] == []
        assert receipt["recovered_failures_truncated"] is True
        assert receipt["secondary_diagnostics_truncated"] is True
        assert [item["secondary_diagnostics_truncated"] for item in receipt["target_results"]] == [
            True, False
        ]
        projected = project_operation_receipt(receipt, "probe_translation")
        summary = operation_header({"tool": "probe_translation", "result": {"receipt": projected}})
        for value in (projected, summary):
            assert value["recovered_failures_truncated"] is True
            assert value["secondary_diagnostics_truncated"] is True
            assert value["recovered_diagnostics"] == value["secondary_diagnostics"] == []
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_model_projection_keeps_truncation_of_empty_recovery_and_secondary_collections():
    source = {
        "gemma": {
            "state": "ready",
            "recovery_diagnostics": [],
            "secondary_diagnostics": [],
            "recovery_diagnostics_truncated": True,
            "secondary_diagnostics_truncated": True,
        }
    }
    projected = await project_query("models.status", source, Records(), str)
    assert projected["gemma"]["state"] == "ready"
    assert projected["gemma"]["recovery_diagnostics"] == []
    assert projected["gemma"]["secondary_diagnostics"] == []
    assert projected["gemma"]["recovery_diagnostics_truncated"] is True
    assert projected["gemma"]["secondary_diagnostics_truncated"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("cause_count", "known_omitted", "broken_sink", "linked_origin"),
    [
        (1, 1, False, False),
        (1, 1, True, False),
        (0, 2, False, False),
        (0, 2, True, False),
        (0, 0, False, False),
        (6, 2, False, False),
        (1, 0, False, True),
    ],
)
async def test_managed_activation_cleanup_survives_actual_cancelled_probe_and_shared_consumers(
    monkeypatch, cause_count, known_omitted, broken_sink, linked_origin
):
    from puripuly_heart.core.openrouter.managed_openrouter_release import (
        ManagedOpenRouterLLMProvider,
        ManagedOpenRouterReleaseBehavior,
        ManagedOpenRouterReleaseResult,
    )
    from puripuly_heart.core.translation_backend import LlmTranslationBackend

    cleanups = tuple(
        exception_diagnostics(
            PermissionError(13, PRIVATE),
            component="managed_openrouter_release",
            operation="managed_operation_clear",
            code="managed_release_failed",
        )
        for _ in range(cause_count)
    )
    if linked_origin:
        cleanups = tuple(
            replace(
                diagnostic, operation_id=uuid4().hex, request_id=uuid4().hex,
                turn_id=uuid4().hex, generation=19,
            )
            for diagnostic in cleanups
        )
    activation = ManagedOpenRouterReleaseResult(
        ManagedOpenRouterReleaseBehavior.READY,
        api_key=PRIVATE,
        secondary_diagnostics=cleanups,
        secondary_omitted_count=known_omitted,
    )
    activation_calls = []

    async def ready():
        activation_calls.append(True)
        return activation

    class Delegate:
        def __init__(self):
            self.entered = asyncio.Event()
            self.requests = []
            self.closed = False

        async def translate(self, **kwargs):
            self.requests.append(kwargs)
            if len(self.requests) == 1:
                self.entered.set()
                await asyncio.Event().wait()
            return Translation(utterance_id=kwargs["utterance_id"], text="Synthetic translation")

        async def close(self):
            self.closed = True

    delegate = Delegate()
    provider = ManagedOpenRouterLLMProvider(
        release_service=SimpleNamespace(ensure_key_for_llm_start=ready),
        delegate_factory=lambda _key: delegate,
    )
    await provider._ensure_delegate()
    control, runtime = control_for(LlmTranslationBackend(provider))
    sink = HistorySink(control.instance_id)
    if broken_sink:
        def broken(*args, **kwargs):
            raise OSError(PRIVATE)

        monkeypatch.setattr(sink, "emit_failure", broken)
    control.application._runtime_logging = sink
    actions = []
    tools = AgentTools(
        control, changed=lambda: pytest.fail("probe changed settings"),
        record_action=actions.append,
        emit_event=lambda _: pytest.fail("probe requested confirmation"),
    )
    task = asyncio.create_task(tools.execute("probe_translation", {}))
    try:
        await asyncio.wait_for(delegate.entered.wait(), 1)
        operation_id = actions[-1]["receipt"]["operation_id"]
        receipt = await control.cancel(operation_id)
        reply = await asyncio.wait_for(task, 2)
        assert reply["status"] == "cancelled"
        assert receipt["terminal"] and receipt["status"] == "cancelled"
        assert receipt["translation_result"] == "cancelled"
        assert receipt["output_delivery"] == receipt["display"] == "not_exercised"
        assert receipt["captured_generation"] == receipt["current_generation"] == 0
        assert activation_calls == [True] and len(delegate.requests) == 1
        assert not delegate.closed
        assert [item["phase"] for item in actions] == ["submitted", "completed"]
        assert actions[-1]["result"] == reply and tools.confirmation is None
        model_reply = model_receipt(reply)
        summary = operation_header(actions[-1])
        history = await control.query("diagnostics.history", {})
        read = await project_query("diagnostics.history", history, control, str)
        context = await collect_agent_context(control, activity=AgentActivity(), topic="diagnostics")
        model = model_context(
            {
                **context, "recent_agent_actions": actions[-1:],
                "agent_operations": actions[-1:],
            },
            modules=("core",),
        )
        views = (
            receipt, reply["receipt"], model_reply["receipt"], summary,
            model["core"]["recent_actions"][0], model["core"]["operations"][0],
        )
        expected = activation.secondary_diagnostics
        expected_ids = [diagnostic.diagnostic_id for diagnostic in expected]
        omitted = activation.secondary_omitted_count
        for view in views:
            assert view["status"] == view["translation_result"] == "cancelled"
            assert not {"error", "diagnostics", "diagnostic_id", "failure_diagnostics"} & view.keys()
            assert view["secondary_omitted_count"] == omitted
            assert view["secondary_truncated"] is (omitted > 0)
            assert view["secondary_diagnostics_truncated"] is (omitted > 0)
            target = view["target_results"][0]
            assert target["translation_result"] == "cancelled"
            assert target["secondary_omitted_count"] == omitted
            assert target["secondary_truncated"] is (omitted > 0)
            assert target["secondary_diagnostics_truncated"] is (omitted > 0)
            for collection in (view["secondary_diagnostics"], target["secondary_diagnostics"]):
                assert [record["diagnostic_id"] for record in collection] == expected_ids
                for record, diagnostic in zip(collection, expected, strict=True):
                    assert record["fields"]["exception_class"] == "PermissionError"
                    assert record["fields"]["errno"] == 13
                    assert record["operation_id"] == (diagnostic.operation_id or operation_id)
                    assert record["request_id"] == (diagnostic.request_id or receipt["request_id"])
                    assert record["turn_id"] == (
                        diagnostic.turn_id or str(delegate.requests[0]["utterance_id"])
                    )
                    assert record["generation"] == (
                        diagnostic.generation if diagnostic.generation is not None else 0
                    )
                    assert record["disposition"] == "degraded"
                    if broken_sink:
                        assert record["file_evidence"] == "unavailable"
                        assert "diagnostic_sequence" not in record
        retained_ids = set() if broken_sink else set(expected_ids)
        for collection in (history["records"], read["records"], context["recent_diagnostics"]):
            assert {record["diagnostic_id"] for record in collection} == retained_ids
        evidence = await recent_diagnostics(control)
        bundle = build_support_bundle(
            ticket="PP-0123456789ABCDEF", app_version="synthetic", summary="Synthetic cancellation",
            diagnostics=evidence, archive=tuple(actions), log_snapshot=None,
        )
        members = contents(bundle, missing_log=True)
        meta = json.loads(members["meta.json"])
        archive = json.loads(members["puri_chat.json"])
        assert {record["diagnostic_id"] for record in meta["diagnostics"]["records"]} == retained_ids
        assert meta["missing_material"]["runtime.log"] == "collection_unavailable"
        assert archive[-1]["result"]["receipt"]["secondary_diagnostics"] == reply["receipt"]["secondary_diagnostics"]
        assert archive[-1]["result"]["receipt"]["secondary_omitted_count"] == omitted
        following = await tools.execute("probe_translation", {})
        assert following["status"] == "applied"
        assert following["receipt"]["translation_result"] == "succeeded"
        assert following["receipt"]["secondary_diagnostics"] == []
        assert following["receipt"]["secondary_omitted_count"] == 0
        assert following["receipt"]["secondary_truncated"] is False
        assert following["receipt"]["secondary_diagnostics_truncated"] is False
        assert activation_calls == [True] and len(delegate.requests) == 2
        assert PRIVATE not in json.dumps([views, history, read, context, model, meta, archive, following])
        assert "Synthetic translation" not in json.dumps(following)
    finally:
        await tools.close()
        control.freeze_ingress()
        await control.drain_operations()
        await runtime.close()
    assert delegate.closed


@pytest.mark.asyncio
async def test_cancelled_probe_bounds_aggregate_without_losing_failed_target_cleanup():
    from puripuly_heart.core.openrouter.managed_openrouter_release import (
        ManagedOpenRouterLLMProvider,
        ManagedOpenRouterReleaseBehavior,
        ManagedOpenRouterReleaseResult,
    )
    from puripuly_heart.core.translation_backend import LlmTranslationBackend

    preceding = tuple(
        serialize_diagnostics(replace(failure(), code="provider.cleanup_failed", category="lifecycle"))
        for _ in range(DIAGNOSTIC_CAUSE_MAX_COUNT)
    )
    cleanup = exception_diagnostics(
        PermissionError(13, PRIVATE), component="managed_openrouter_release",
        operation="managed_operation_clear", code="managed_release_failed",
    )
    entered = asyncio.Event()
    requests = []

    async def ready():
        return ManagedOpenRouterReleaseResult(
            ManagedOpenRouterReleaseBehavior.READY, api_key=PRIVATE
        )

    async def translate(**kwargs):
        requests.append(kwargs)
        if len(requests) == 1:
            return Translation(
                utterance_id=kwargs["utterance_id"], text="Synthetic translation",
                secondary_diagnostics=preceding,
            )
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as exc:
            exc.secondary_diagnostics = (cleanup,)
            exc.secondary_omitted_count = 2
            raise

    async def close():
        return None

    provider = ManagedOpenRouterLLMProvider(
        SimpleNamespace(ensure_key_for_llm_start=ready),
        lambda _key: SimpleNamespace(translate=translate, close=close),
    )
    await provider._ensure_delegate()
    control, runtime = control_for(LlmTranslationBackend(provider), secondary_target_language="ja")
    sink = HistorySink(control.instance_id)
    control.application._runtime_logging = sink
    try:
        accepted = await control.submit("translation.probe", {}, request_id=uuid4().hex)
        await asyncio.wait_for(entered.wait(), 1)
        receipt = await control.cancel(accepted["operation_id"])
        assert receipt["status"] == receipt["translation_result"] == "cancelled"
        assert len(receipt["secondary_diagnostics"]) == DIAGNOSTIC_CAUSE_MAX_COUNT
        assert receipt["secondary_omitted_count"] == 3
        assert receipt["secondary_truncated"] and receipt["secondary_diagnostics_truncated"]
        assert [target["translation_result"] for target in receipt["target_results"]] == [
            "succeeded", "cancelled"
        ]
        projected = project_operation_receipt(receipt, "probe_translation")
        reply = model_receipt({"status": "cancelled", "receipt": projected})
        summary = operation_header({"tool": "probe_translation", "result": reply})
        context = model_context(
            {"agent_operations": [{"tool": "probe_translation", "result": reply}]},
            modules=("core",),
        )
        for view in (receipt, projected, reply["receipt"], summary, context["core"]["operations"][0]):
            assert view["status"] == view["translation_result"] == "cancelled"
            target = view["target_results"][1]
            assert target["secondary_diagnostics"][0]["diagnostic_id"] == cleanup.diagnostic_id
            assert target["secondary_diagnostics"][0]["disposition"] == "degraded"
            assert target["secondary_diagnostics"][0]["turn_id"] == str(requests[1]["utterance_id"])
            assert target["secondary_diagnostics"][0]["operation_id"] == accepted["operation_id"]
            assert target["secondary_omitted_count"] == 2
            assert target["secondary_truncated"] and target["secondary_diagnostics_truncated"]
        assert projected["secondary_omitted_count"] == 7
        assert [record["diagnostic_id"] for record in projected["secondary_diagnostics"]] == [
            record["diagnostic_id"] for record in preceding[:4]
        ]
        history = await control.query("diagnostics.history", {"diagnostic_id": cleanup.diagnostic_id})
        retained = history["records"]
        read = await project_query("diagnostics.history", history, control, str)
        assert len(retained) == 1 and retained[0]["fields"]["errno"] == 13
        assert retained[0]["disposition"] == read["records"][0]["disposition"] == "degraded"
        assert PRIVATE not in json.dumps([receipt, projected, reply, summary, context, retained, read])
    finally:
        control.freeze_ingress()
        await control.drain_operations()
        await runtime.close()


@pytest.mark.asyncio
async def test_nested_install_result_and_snapshot_preserve_owning_evidence_roles_and_context():
    from puripuly_heart.core.local_asr.local_asr_provisioning import (
        LocalASRInstallResult,
        LocalASRModelProvisioningState,
        LocalASRProvisioningSnapshot,
    )

    @dataclass(frozen=True, slots=True)
    class DetailedModelState(LocalASRModelProvisioningState):
        recovered_diagnostics: tuple[ErrorDiagnostics, ...] = ()
        recovery_diagnostics: tuple[ErrorDiagnostics, ...] = ()
        secondary_diagnostics: tuple[ErrorDiagnostics, ...] = ()

    source_context = {
        "operation_id": uuid4().hex, "request_id": uuid4().hex,
        "turn_id": uuid4().hex, "generation": 7,
    }
    primary, recovered, recovery, secondary = (
        replace(failure(), **source_context) for _ in range(4)
    )
    snapshot = LocalASRProvisioningSnapshot(
        (
            DetailedModelState(
                "synthetic-model", "cpu", "ready", operation="cancelled",
                failure_diagnostics=primary, recovered_diagnostics=(recovered,),
                recovery_diagnostics=(recovery,), secondary_diagnostics=(secondary,),
            ),
        ),
        ("synthetic-model",),
        "synthetic-gpu",
    )

    class Provisioning:
        def __init__(self):
            self.snapshot = snapshot
            self.requests = []

        def start_install(self, request):
            self.requests.append(request)

            async def complete():
                return LocalASRInstallResult(request, (), (), True, snapshot)

            return asyncio.create_task(complete())

    control, runtime = control_for(ControlledBackend())
    provisioning = Provisioning()
    control.provisioning = lambda: provisioning
    try:
        standalone = control._snapshot_json(snapshot)
        sink = HistorySink(control.instance_id)
        control.application._runtime_logging = sink
        accepted = await control.submit("models.install", {"backend": "cpu"}, request_id=uuid4().hex)
        receipt = await control.wait(accepted["operation_id"], timeout=2)
        assert receipt["terminal"] and receipt["status"] == "cancelled"
        assert len(provisioning.requests) == 1
        assert receipt["recovery_diagnostics"] == receipt["secondary_diagnostics"] == []
        for view, fallback in ((standalone, "unknown"), (receipt["models"], "cancelled")):
            row = view["models"][0]
            assert row["integrity"] == "ready" and row["operation"] == "cancelled"
            for name, diagnostic, disposition in (
                ("failure_diagnostics", primary, fallback),
                ("recovered_diagnostics", recovered, "recovered"),
                ("recovery_diagnostics", recovery, "recovered"),
                ("secondary_diagnostics", secondary, "degraded"),
            ):
                record = row[name] if name == "failure_diagnostics" else row[name][0]
                assert record["diagnostic_id"] == diagnostic.diagnostic_id
                assert record["disposition"] == disposition
                for key, value in source_context.items():
                    assert record[key] == value
                assert record["operation_id"] != accepted["operation_id"]
                assert record["request_id"] != accepted["request_id"]
        history = await control.query("diagnostics.history", {})
        read = await project_query("diagnostics.history", history, control, str)
        expected = {
            primary.diagnostic_id: "cancelled",
            recovered.diagnostic_id: "recovered",
            recovery.diagnostic_id: "recovered",
            secondary.diagnostic_id: "degraded",
        }
        for records in (history["records"], read["records"]):
            assert {record["diagnostic_id"]: record["disposition"] for record in records} == expected
            for record in records:
                assert record["instance_id"] == control.instance_id
                for key, value in source_context.items():
                    assert record[key] == value
        assert PRIVATE not in json.dumps([standalone, receipt, history, read])
    finally:
        control.freeze_ingress()
        await control.drain_operations()
        await runtime.close()
