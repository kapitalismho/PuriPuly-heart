from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, replace
from uuid import uuid4

import pytest

from puripuly_heart.app.services.agent_chat.context import AgentActivity, collect_agent_context
from puripuly_heart.app.services.agent_chat.execution import project_query
from puripuly_heart.app.services.agent_chat.modules import (
    model_context,
    model_receipt,
    operation_header,
)
from puripuly_heart.app.services.agent_chat.owner import AgentChatOwner
from puripuly_heart.app.services.agent_chat.tools import project_operation_receipt
from puripuly_heart.core.diagnostics import DIAGNOSTIC_CAUSE_MAX_COUNT
from tests.app.test_agent_chat import Records, Responses, answer, context_of
from tests.app.test_agent_chat_tools import composed_osc_agent as composed_osc_agent
from tests.app.test_agent_chat_unified import batch, call
from tests.app.test_diagnostic_consumers import PRIVATE, HistorySink, failure
from tests.app.test_translation_probe import ControlledBackend, control_for


@pytest.mark.asyncio
async def test_controlled_osc_timeout_survives_retry_query_context_and_model_receipts(
    composed_osc_agent, monkeypatch
):
    fixture = composed_osc_agent
    control = fixture.app.control()
    before = await control.query("settings.current", {})
    persisted = fixture.path.read_bytes()

    async def timeout(*args, **kwargs):
        raise TimeoutError(PRIVATE)

    monkeypatch.setattr(fixture.service, "discover_vrchat", timeout)
    result = await fixture.tools.execute("retry_app", {"target": "osc_discovery"})
    assert result["status"] == "degraded"
    receipt = result["receipt"]
    assert receipt["reason"] == "discovery_timeout"
    assert receipt["retry_started"] is True
    diagnostic_id = receipt["diagnostics"]["diagnostic_id"]
    raw = await control.query("osc.status", {})
    direct = await project_query("osc.status", raw, control, str)
    chat = AgentChatOwner(
        control, Responses(batch(call("output", "read_app", "output")), answer()), localize=str
    )
    try:
        await chat.start()
        turn = await chat.send("Inspect the retained OSC failure.")
        (read,) = turn.tool_results
        assert read.status == "unchanged"
        assert "output" in read.result["context"]["unchanged"]
        retained_output = context_of(chat._transport.requests[0])["output"]["osc"]
    finally:
        await chat.close()
    context = await collect_agent_context(control, activity=AgentActivity(), topic="output")
    model = model_context(context, "output")
    reply = model_receipt(result)
    summary = operation_header({"tool": "osc.discovery.retry", "result": reply})
    for output in (
        receipt["output"],
        direct["output"],
        retained_output,
        context["output"]["osc"],
        model["output"]["osc"],
        summary["output"],
    ):
        assert output["query_diagnostics"]["diagnostic_id"] == diagnostic_id
        assert output["query_diagnostics"]["fields"]["exception_class"] == "TimeoutError"
        assert (
            output["query_diagnostics"]["operation_id"]
            == raw["output"]["query_diagnostics"]["operation_id"]
        )
        assert output["remote_delivery"] == "unacknowledged"
    assert reply["receipt"]["diagnostics"]["diagnostic_id"] == diagnostic_id
    assert summary["diagnostics"]["diagnostic_id"] == diagnostic_id
    assert summary["operation_id"] == receipt["operation_id"]
    assert summary["request_id"] == receipt["request_id"]
    assert (await control.query("settings.current", {})) == before
    assert fixture.path.read_bytes() == persisted
    assert PRIVATE not in json.dumps([result, direct, read.result, context, model, summary])


@pytest.mark.asyncio
async def test_output_and_capture_diagnostics_survive_all_chatbot_read_views():
    records = Records()

    def retain(diagnostic, disposition="failed"):
        return records.history.retain(diagnostic, severity="error", disposition=disposition)[0]

    primary = retain(replace(failure(), causes=(failure(),)))
    recovered = retain(failure(), "recovered")
    native = retain(
        replace(
            failure(),
            component="overlay_application",
            operation="startup",
            code="worker_startup_failed",
            category="lifecycle",
            status_code=None,
            fields={"exit_code": 7, "body": PRIVATE},
            causes=(failure(),),
        )
    )
    runtime = [retain(failure()) for _ in range(DIAGNOSTIC_CAUSE_MAX_COUNT + 1)]
    osc = {
        "query_failure": "oscquery_unavailable",
        "query_diagnostics": primary,
        "discovery_diagnostics": primary,
        "avatar_diagnostics": primary,
        "discovery_observation": "unavailable",
        "avatar_observation": "unavailable",
        "runtime_failures": runtime,
        "runtime_failures_truncated": True,
        "runtime_failures_omitted_count": 2,
        "service_failures": [],
        "service_failures_truncated": True,
        "dispatch_failures": [primary],
        "configured_host": PRIVATE,
        "discovered_remote": {"host": PRIVATE},
        "remote_delivery": "received",
    }
    overlay = {
        "failure_reason": "render_failed",
        "failure_diagnostics": native,
        "desired_enabled": True,
        "effective_target": None,
        "fallback_active": True,
        "private_path": PRIVATE,
    }
    channels = {
        "self": {
            "desired_active": True,
            "effective_active": True,
            "capture": {
                "state": "running",
                "generation": 3,
                "has_source": True,
                "source_open_diagnostics": recovered,
                "source_open_recovered": True,
                "runtime_signature": [PRIVATE],
            },
        },
        "peer": {
            "desired_active": True,
            "effective_active": False,
            "capture": {
                "state": "faulted",
                "generation": 4,
                "has_source": False,
                "failure_diagnostics": primary,
                "source_open_recovered": False,
            },
        },
    }
    records.values["osc.status"] = {"output": osc}
    records.values["overlay.status"] = {"output": overlay, "locked": False}
    records.values["app.status"]["channels"] = channels
    for name in ("capture.channels", "providers.status"):
        records.values[name] = {"channels": channels}
    views = []
    chat = AgentChatOwner(records, None, localize=str)
    try:
        for name in (
            "osc.status",
            "overlay.status",
            "app.status",
            "capture.channels",
            "providers.status",
        ):
            raw = await records.query(name, {})
            projected = await project_query(name, raw, records, str)
            views.append(model_receipt({"result": projected})["result"])
        output_read = await chat._execute_call(
            call("output", "read_app", "output"), {"topic": "output"}, chat._generation
        )
        overview = await chat._execute_call(
            call("overview", "read_app", "overview"), {"topic": "overview"}, chat._generation
        )
        assert output_read["status"] == overview["status"] == "observed"
        assert "core" in overview["context"]["unchanged"]
        snapshot = await collect_agent_context(records, activity=AgentActivity())
        context = model_context(snapshot)
        osc_views = [
            views[0]["output"],
            snapshot["output"]["osc"],
            context["output"]["osc"],
            output_read["context"]["output"]["osc"],
        ]
        for output in osc_views:
            for key in ("query_diagnostics", "discovery_diagnostics", "avatar_diagnostics"):
                assert output[key]["diagnostic_id"] == primary["diagnostic_id"]
                assert output[key]["causes"][0]["status_code"] == 403
            assert len(output["runtime_failures"]) == DIAGNOSTIC_CAUSE_MAX_COUNT
            assert output["runtime_failures_omitted_count"] == 3
            assert output["runtime_failures_truncated"] is True
            assert output["service_failures"] == [] and output["service_failures_truncated"] is True
            assert output["dispatch_failures"][0]["diagnostic_id"] == primary["diagnostic_id"]
            assert output["discovery_observation"] == output["avatar_observation"] == "unavailable"
            assert output["remote_delivery"] == "unacknowledged"
        for output in (
            views[1]["output"],
            snapshot["output"]["overlay"],
            context["output"]["overlay"],
            output_read["context"]["output"]["overlay"],
        ):
            assert output["failure_diagnostics"]["diagnostic_id"] == native["diagnostic_id"]
            assert output["failure_diagnostics"]["fields"]["exit_code"] == 7
            assert output["fallback_active"] is True and output["effective_target"] is None
        for value in (
            views[2],
            views[3],
            views[4],
            snapshot,
            context["core"],
            output_read["context"]["core"],
        ):
            current = value["channels"]
            assert current["self"]["effective_active"] is True
            assert current["self"]["capture"]["source_open_recovered"] is True
            assert (
                current["self"]["capture"]["source_open_diagnostics"]["diagnostic_id"]
                == recovered["diagnostic_id"]
            )
            assert (
                current["peer"]["capture"]["failure_diagnostics"]["diagnostic_id"]
                == primary["diagnostic_id"]
            )
            assert current["peer"]["effective_active"] is False
        for receipt, tool in (
            ({"status": "degraded", "output": overlay}, "overlay.set"),
            ({"status": "degraded", "channel": channels["peer"]}, "capture.set"),
        ):
            reply = model_receipt({"receipt": project_operation_receipt(receipt, tool)})
            summary = operation_header({"tool": tool, "result": reply})
            retained = (
                summary["output"]["failure_diagnostics"]
                if "output" in summary
                else summary["channel"]["capture"]["failure_diagnostics"]
            )
            assert retained["diagnostic_id"] == (
                native["diagnostic_id"] if "output" in summary else primary["diagnostic_id"]
            )
        assert PRIVATE not in json.dumps([views, snapshot, context, output_read, overview])
    finally:
        await chat.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ("recovered", "cleanup", "cancelled"))
async def test_models_install_preserves_owning_result_evidence_in_terminal_and_chatbot_views(
    outcome,
):
    from puripuly_heart.core.local_asr.local_asr_provisioning import (
        LocalASRInstallResult,
        LocalASRModelProvisioningState,
        LocalASRProvisioningSnapshot,
    )

    @dataclass(frozen=True, slots=True)
    class ResultWithOmission(LocalASRInstallResult):
        recovery_diagnostics_truncated: bool = False
        secondary_diagnostics_truncated: bool = False

    primary = replace(failure(), component="model_assets", operation="download")
    recovered = replace(failure(), component="model_assets", operation="download")
    secondary = replace(
        failure(),
        component="model_assets",
        operation="cleanup",
        code="local_asr.cleanup_failed",
        category="lifecycle",
        status_code=None,
    )
    snapshot = LocalASRProvisioningSnapshot(
        (LocalASRModelProvisioningState("synthetic-model", "cpu", "ready"),),
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
                return ResultWithOmission(
                    request=request,
                    installed_model_ids=() if outcome == "cancelled" else ("synthetic-model",),
                    failed_model_ids=(),
                    cancelled=outcome == "cancelled",
                    snapshot=snapshot,
                    failure_diagnostics=primary if outcome == "cancelled" else None,
                    recovery_diagnostics=(recovered,) if outcome != "cleanup" else (),
                    secondary_diagnostics=(secondary,) if outcome != "recovered" else (),
                    recovery_diagnostics_truncated=outcome == "cleanup",
                    secondary_diagnostics_truncated=outcome == "recovered",
                )

            return asyncio.create_task(complete())

    provisioning = Provisioning()
    control, runtime = control_for(ControlledBackend())
    control.provisioning = lambda: provisioning
    sink = HistorySink(control.instance_id)
    control.application._runtime_logging = sink
    try:
        accepted = await control.submit(
            "models.install", {"backend": "cpu"}, request_id=uuid4().hex
        )
        receipt = await control.wait(accepted["operation_id"], timeout=2)
        expected_status = "cancelled" if outcome == "cancelled" else "applied"
        assert receipt["terminal"] and receipt["status"] == expected_status
        assert len(provisioning.requests) == 1
        assert receipt["failed_model_ids"] == []
        assert "error" not in receipt and "request" not in receipt
        projected = project_operation_receipt(receipt, "models.install")
        reply = model_receipt({"status": expected_status, "receipt": projected})
        action = {"tool": "models.install", "result": reply}
        summary = operation_header(action)
        context = model_context({"agent_operations": [action]}, modules=("core",))
        for value in (
            receipt,
            projected,
            reply["receipt"],
            summary,
            context["core"]["operations"][0],
        ):
            assert value["status"] == expected_status
            assert value["recovery_diagnostics_truncated"] is (outcome == "cleanup")
            assert value["secondary_diagnostics_truncated"] is (outcome == "recovered")
            if outcome == "cancelled":
                assert value["failure_diagnostics"]["diagnostic_id"] == primary.diagnostic_id
            for key, diagnostic, present, disposition in (
                ("recovery_diagnostics", recovered, outcome != "cleanup", "recovered"),
                ("secondary_diagnostics", secondary, outcome != "recovered", "degraded"),
            ):
                if present:
                    assert value[key][0]["diagnostic_id"] == diagnostic.diagnostic_id
                    assert value[key][0]["operation_id"] == accepted["operation_id"]
                    assert value[key][0]["request_id"] == accepted["request_id"]
                    assert value[key][0]["disposition"] == disposition
                else:
                    assert value[key] == []
        history = sink.diagnostic_history({"operation_id": accepted["operation_id"]})
        expected_ids = {
            diagnostic.diagnostic_id
            for diagnostic, present in (
                (primary, outcome == "cancelled"),
                (recovered, outcome != "cleanup"),
                (secondary, outcome != "recovered"),
            )
            if present
        }
        assert {item["diagnostic_id"] for item in history["records"]} == expected_ids
        assert PRIVATE not in json.dumps([receipt, reply, summary, context, history])
    finally:
        control.freeze_ingress()
        await control.drain_operations()
        await runtime.close()
