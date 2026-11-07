from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest

from puripuly_heart.app.services.agent_chat.tools import AgentTools
from puripuly_heart.app.services.application_control import ApplicationControlOwner
from puripuly_heart.app.services.application_control_events import ControlEvents
from puripuly_heart.app.services.settings.settings_transaction_result import (
    SettingsTransactionResultOwner,
)
from puripuly_heart.core.local_translation.runtime import ManagedGemmaRuntimeOwner
from puripuly_heart.core.openrouter.managed_openrouter_release import ManagedOpenRouterLLMProvider
from puripuly_heart.core.runtime.provider_handle import ProviderRuntimeHandle
from puripuly_heart.core.translation_backend import LlmTranslationBackend, TranslationBackend
from puripuly_heart.domain.models import Translation
from puripuly_heart.providers.llm.managed_gemma import ManagedGemmaLLMProvider


class ControlledBackend(TranslationBackend):
    def __init__(self, *, text="Hola, que tengas un buen día.", failure=None, hold=False):
        self.text = text
        self.failure = failure
        self.hold = hold
        self.requests = []
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.closed = False
        self.ignore_cancellation = False

    async def translate(self, request):
        assert not self.closed
        self.requests.append(request)
        self.entered.set()
        if self.hold:
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                if not self.ignore_cancellation:
                    raise
                await self.release.wait()
        assert not self.closed
        if self.failure is not None:
            raise self.failure
        return Translation(utterance_id=request.utterance_id, text=self.text)

    async def close(self):
        self.closed = True


def control_for(backend):
    runtime = ProviderRuntimeHandle(name="llm", provider=backend)
    control = ApplicationControlOwner(
        application=SimpleNamespace(),
        settings=SimpleNamespace(),
        pipeline=SimpleNamespace(llm_runtime=runtime),
        results=SettingsTransactionResultOwner(),
        events=ControlEvents(instance_id="translation-probe-test"),
        provisioning=lambda: pytest.fail("probe accessed model provisioning"),
        gpu=lambda: pytest.fail("probe accessed GPU discovery"),
        peer=lambda: pytest.fail("probe accessed peer capture"),
        calibration=lambda: None,
        gemma=lambda: pytest.fail("probe accessed Gemma activation"),
        sync_ui=lambda: pytest.fail("probe changed presentation"),
        locale_choices=("en",),
        localize=str,
        gemma_selection=lambda _: pytest.fail("probe read selected configuration"),
        instance_id="translation-probe-test",
    )
    return control, runtime


async def submit_probe(control, arguments=None):
    accepted = await control.submit(
        "translation.probe", {} if arguments is None else arguments, request_id=uuid4().hex
    )
    assert accepted["status"] == "accepted"
    assert accepted["terminal"] is False
    return await control.wait(accepted["operation_id"], timeout=2)


def assert_no_output(result):
    assert result["terminal"] is True
    assert result["output_delivery"] == "not_exercised"
    assert result["display"] == "not_exercised"
    assert not {"text", "translation", "prompt", "context", "provider", "model"} & result.keys()


async def test_probe_invokes_current_backend_and_retains_metadata_only_terminal_receipt():
    backend = ControlledBackend()
    control, runtime = control_for(backend)
    try:
        result = await submit_probe(control)
        assert_no_output(result)
        assert result["status"] == "applied"
        assert result["translation_result"] == "succeeded"
        assert result["captured_generation"] == result["current_generation"] == 0
        assert result["source_language"] == "en"
        assert result["target_language"] == "es"
        assert len(backend.requests) == 1
        request = backend.requests[0]
        assert request.text == "Hello, have a nice day."
        assert request.system_prompt == (
            "Translate from English to Spanish. Return only the translation."
        )
        assert request.context == ""
        assert request.scene_participant_count is None
        assert request.source_language == "en"
        assert request.target_language == "es"
        assert await control.operation(result["operation_id"]) == result
        assert [event["topic"] for event in control.events.history] == ["operation"]
        assert backend.text not in json.dumps(result)
    finally:
        control.freeze_ingress()
        await control.drain_operations()
        await runtime.close()


@pytest.mark.parametrize(
    ("backend", "status", "code"),
    [
        (None, "action_required", "translation_backend_unavailable"),
        (ControlledBackend(text=" \n"), "failed", "translation_probe_invalid_response"),
        (
            ControlledBackend(failure=RuntimeError("PRIVATE-KEY-AND-USER-TEXT")),
            "failed",
            "translation_probe_failed",
        ),
        (
            ControlledBackend(failure=httpx.ConnectError("PRIVATE-ENDPOINT")),
            "failed",
            "translation_probe_failed",
        ),
    ],
)
async def test_probe_unavailable_empty_and_failed_backend_are_not_success(backend, status, code):
    control, runtime = control_for(backend)
    try:
        result = await submit_probe(control)
        assert_no_output(result)
        assert result["status"] == status
        assert result["translation_result"] != "succeeded"
        assert result["error"]["code"] == code
        assert "PRIVATE" not in json.dumps(result)
        if isinstance(getattr(backend, "failure", None), httpx.ConnectError):
            assert result["failure_category"] == "network"
    finally:
        await runtime.close()


@pytest.mark.parametrize("failure", [None, RuntimeError("PRIVATE-RETIRED-FAILURE")])
async def test_replaced_backend_completion_is_stale_and_resources_drain(failure):
    backend = ControlledBackend(hold=True, failure=failure)
    replacement = ControlledBackend()
    control, runtime = control_for(backend)
    try:
        accepted = await control.submit("translation.probe", {}, request_id=uuid4().hex)
        await asyncio.wait_for(backend.entered.wait(), timeout=1)
        replacing = asyncio.create_task(runtime.replace_provider(replacement, start=False))
        await asyncio.sleep(0)
        assert runtime.provider is replacement
        assert backend.closed is False
        assert replacing.done() is False
        backend.release.set()
        result = await control.wait(accepted["operation_id"], timeout=2)
        await replacing
        assert_no_output(result)
        assert result["status"] == "failed"
        assert result["translation_result"] == "stale"
        assert result["error"]["code"] == "stale_provider_completion"
        assert result["captured_generation"] == 0
        assert result["current_generation"] == 1
        assert backend.closed is True
        recovered = await submit_probe(control)
        assert recovered["translation_result"] == "succeeded"
        assert recovered["captured_generation"] == recovered["current_generation"] == 1
    finally:
        backend.release.set()
        control.freeze_ingress()
        await control.drain_operations()
        await runtime.close()


@pytest.mark.parametrize("shutdown", [False, True])
async def test_cancellation_and_shutdown_reject_late_success_and_drain_owned_work(shutdown):
    backend = ControlledBackend(hold=True)
    backend.ignore_cancellation = True
    control, runtime = control_for(backend)
    accepted = await control.submit("translation.probe", {}, request_id=uuid4().hex)
    await asyncio.wait_for(backend.entered.wait(), timeout=1)
    if shutdown:
        control.freeze_ingress()
        cancelled = asyncio.create_task(control.drain_operations())
    else:
        cancelled = asyncio.create_task(control.cancel(accepted["operation_id"]))
    await asyncio.wait_for(backend.cancelled.wait(), timeout=1)
    closing = asyncio.create_task(runtime.close())
    await asyncio.sleep(0)
    assert backend.closed is False
    assert closing.done() is False
    assert cancelled.done() is False
    backend.release.set()
    await cancelled
    await closing
    result = await control.operation(accepted["operation_id"])
    assert_no_output(result)
    assert result["status"] == ("interrupted" if shutdown else "cancelled")
    assert result["translation_result"] == result["status"]
    assert backend.closed is True
    assert runtime.has_resources is False


async def test_probe_strict_arguments_reject_before_backend_invocation():
    backend = ControlledBackend()
    control, runtime = control_for(backend)
    try:
        result = await submit_probe(control, {"text": "do not send"})
        assert result["status"] == "rejected"
        assert result["error"]["code"] == "invalid_arguments"
        assert backend.requests == []
    finally:
        await runtime.close()


async def test_probe_does_not_prepare_managed_local_model_or_release_managed_credentials(tmp_path):
    calls = []

    async def provision(**kwargs):
        calls.append("provision")
        raise AssertionError("probe provisioned assets")

    local_runtime = ManagedGemmaRuntimeOwner(
        install_dir=tmp_path,
        provisioner=provision,
        transport_factory=lambda _: pytest.fail("probe created local transport"),
    )
    local_backend = LlmTranslationBackend(
        ManagedGemmaLLMProvider(runtime=local_runtime, backend="cpu")
    )

    async def release():
        calls.append("release")
        raise AssertionError("probe released credentials")

    managed_backend = LlmTranslationBackend(
        ManagedOpenRouterLLMProvider(
            release_service=SimpleNamespace(ensure_key_for_llm_start=release),
            delegate_factory=lambda _: pytest.fail("probe created managed delegate"),
        )
    )
    for backend in (local_backend, managed_backend):
        control, runtime = control_for(backend)
        try:
            result = await submit_probe(control)
            assert_no_output(result)
            assert result["status"] == "action_required"
            assert result["translation_result"] == "unavailable"
            assert result["error"]["code"] == "translation_preparation_required"
            assert calls == []
        finally:
            await runtime.close()
    await local_runtime.close()


async def test_agent_probe_waits_for_actual_terminal_receipt_without_settings_actions():
    backend = ControlledBackend(hold=True)
    control, runtime = control_for(backend)
    actions = []
    tools = AgentTools(
        control,
        changed=lambda: pytest.fail("probe changed settings UI"),
        record_action=actions.append,
        emit_event=lambda _: pytest.fail("probe emitted a confirmation"),
    )
    try:
        invalid = await tools.execute("probe_translation", {"text": "private"})
        assert invalid["status"] == "rejected"
        assert backend.requests == []
        task = asyncio.create_task(tools.execute("probe_translation", {}))
        await asyncio.wait_for(backend.entered.wait(), timeout=1)
        assert task.done() is False
        assert actions[0]["phase"] == "submitted"
        backend.release.set()
        result = await task
        assert result["status"] == "applied"
        assert_no_output(result["receipt"])
        assert result["receipt"]["translation_result"] == "succeeded"
        assert [action["phase"] for action in actions] == ["submitted", "completed"]
        assert tools.confirmation is None
        assert backend.text not in json.dumps(actions)
    finally:
        backend.release.set()
        await tools.close()
        await runtime.close()


@pytest.mark.parametrize("interruption", ["stale", "cancelled"])
async def test_agent_probe_reports_interruption_before_recovery_on_current_runtime(interruption):
    backend = ControlledBackend(hold=True)
    backend.ignore_cancellation = True
    replacement = ControlledBackend()
    control, runtime = control_for(backend)
    actions = []
    tools = AgentTools(
        control,
        changed=lambda: pytest.fail("probe changed settings presentation"),
        record_action=actions.append,
        emit_event=lambda _: pytest.fail("probe requested confirmation"),
    )
    task = asyncio.create_task(tools.execute("probe_translation", {}))
    settling = None
    try:
        await asyncio.wait_for(backend.entered.wait(), 1)
        operation_id = actions[-1]["receipt"]["operation_id"]
        assert actions[-1]["phase"] == "submitted" and not task.done()
        if interruption == "stale":
            settling = asyncio.create_task(runtime.replace_provider(replacement, start=False))
            await asyncio.sleep(0)
            assert runtime.provider is replacement
        else:
            settling = asyncio.create_task(control.cancel(operation_id))
            await asyncio.wait_for(backend.cancelled.wait(), 1)
        assert not task.done() and not backend.closed and not settling.done()
        backend.release.set()
        await settling
        result = await asyncio.wait_for(task, 2)
        assert_no_output(result["receipt"])
        assert result["status"] == ("failed" if interruption == "stale" else "cancelled")
        assert result["receipt"]["translation_result"] == interruption
        assert actions[-1]["phase"] == "completed" and actions[-1]["result"] == result
        if interruption == "cancelled":
            await runtime.replace_provider(replacement, start=False)
        assert backend.closed
        recovered = await tools.execute("probe_translation", {})
        assert recovered["status"] == "applied"
        assert recovered["receipt"]["translation_result"] == "succeeded"
        assert recovered["receipt"]["captured_generation"] == runtime.generation
        assert_no_output(recovered["receipt"])
        assert len(replacement.requests) == 1 and not replacement.closed
        assert backend.text not in json.dumps(actions)
        assert tools.confirmation is None
    finally:
        backend.release.set()
        if settling is not None:
            await settling
        await asyncio.gather(task, return_exceptions=True)
        await tools.close()
        control.freeze_ingress()
        await control.drain_operations()
        await runtime.close()


def test_relay_probe_schema_is_strict_empty_and_matches_generator():
    import ast

    root = Path(__file__).resolve().parents[2]
    generated = (root / "agent-relay/src/action-tools.ts").read_text(encoding="utf-8")
    tools = json.loads(generated.removeprefix("export const actionTools = ").removesuffix(";\n"))
    probe = next(tool for tool in tools if tool["name"] == "probe_translation")
    tree = ast.parse((root / "agent-relay/generate-tools.py").read_text(encoding="utf-8"))
    declaration = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        and any(
            isinstance(value, ast.Constant) and value.value == "probe_translation"
            for value in node.values
        )
    )
    assert ast.literal_eval(declaration) == probe
    assert probe["strict"] is True
    assert probe["parameters"] == {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }


async def test_probe_cancelled_before_execution_never_invokes_backend():
    backend = ControlledBackend()
    control, runtime = control_for(backend)
    try:
        accepted = await control.submit("translation.probe", {}, request_id=uuid4().hex)
        cancelled = await control.cancel(accepted["operation_id"])
        assert_no_output(cancelled)
        assert cancelled["status"] == "cancelled"
        assert cancelled["translation_result"] == "cancelled"
        assert backend.requests == []
        assert [event["topic"] for event in control.events.history] == ["operation"]
    finally:
        await runtime.close()


async def test_cancelled_replacement_retains_backend_until_probe_settles_and_shutdown():
    backend = ControlledBackend(hold=True)
    replacement = ControlledBackend()
    control, runtime = control_for(backend)
    try:
        accepted = await control.submit("translation.probe", {}, request_id=uuid4().hex)
        await asyncio.wait_for(backend.entered.wait(), timeout=1)
        replacing = asyncio.create_task(runtime.replace_provider(replacement, start=False))
        await asyncio.sleep(0)
        replacing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await replacing
        assert runtime.provider is replacement
        assert backend.closed is False
        backend.release.set()
        result = await control.wait(accepted["operation_id"], timeout=2)
        assert result["translation_result"] == "stale"
        await runtime.close()
        assert backend.closed is True
        assert replacement.closed is True
        assert runtime.has_resources is False
    finally:
        backend.release.set()
        control.freeze_ingress()
        await control.drain_operations()
        await runtime.close()


@pytest.mark.parametrize("replacement_attached", [False, True])
async def test_pipeline_runtime_replacement_or_detachment_fences_late_probe(replacement_attached):
    backend = ControlledBackend(hold=True)
    control, runtime = control_for(backend)
    replacement = (
        ProviderRuntimeHandle(name="llm", provider=ControlledBackend())
        if replacement_attached
        else None
    )
    try:
        accepted = await control.submit("translation.probe", {}, request_id=uuid4().hex)
        await asyncio.wait_for(backend.entered.wait(), timeout=1)
        control.pipeline.llm_runtime = replacement
        backend.release.set()
        result = await control.wait(accepted["operation_id"], timeout=2)
        assert_no_output(result)
        assert result["status"] == "failed"
        assert result["translation_result"] == "stale"
        assert result["captured_generation"] == 0
        assert result["current_generation"] == (0 if replacement_attached else None)
    finally:
        backend.release.set()
        control.freeze_ingress()
        await control.drain_operations()
        await runtime.close()
        if replacement is not None:
            await replacement.close()
