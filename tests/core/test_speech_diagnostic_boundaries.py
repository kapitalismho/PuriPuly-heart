from __future__ import annotations

import asyncio
import logging
from dataclasses import replace

import pytest

from puripuly_heart.core.diagnostics import DIAGNOSTIC_RECORD_ATTRIBUTE
from puripuly_heart.core.gpu_worker import GpuWorkerRequestError
from puripuly_heart.core.local_asr.local_asr_provider_runtime import ProviderRuntimeBuildRequest
from puripuly_heart.core.self_capture import SelfCaptureSessionState
from tests.core.runtime.test_local_asr_provider_runtime import (
    FakeProvider,
    FakeProviderFactory,
    _owner,
    _resolved_config,
)
from tests.core.runtime.test_self_capture_session import RecordingProvider, build_owner, config


def native_records(caplog):
    return [
        getattr(record, DIAGNOSTIC_RECORD_ATTRIBUTE)
        for record in caplog.records
        if hasattr(record, DIAGNOSTIC_RECORD_ATTRIBUTE)
    ]


@pytest.mark.asyncio
async def test_failed_build_warmup_retains_first_cause_and_cleanup_debt(caplog) -> None:
    primary = GpuWorkerRequestError("heartbeat_timeout", {"raw_payload": "PRIVATE_NATIVE_BODY"})

    class Candidate(FakeProvider):
        async def warmup(self):
            raise primary

        async def close(self):
            self.close_calls += 1
            if self.close_calls == 1:
                raise OSError("PRIVATE_SECONDARY_CLEANUP")

    class Factory(FakeProviderFactory):
        async def create(self, request, *, gpu_runtime, on_terminal_failure=None):
            candidate = Candidate(
                request.provider_id,
                gpu_runtime=gpu_runtime,
                channel=request.channel,
                gpu_device_id=request.gpu_device_id,
            )
            self.providers.append(candidate)
            return candidate

    owner, _, _, factory = _owner(provider_factory=Factory())
    caplog.set_level(logging.WARNING)
    try:
        result = await owner.replace_provider(
            ProviderRuntimeBuildRequest(config=_resolved_config("self", "soniox"), warmup=True),
            start=True,
        )
        assert result.status == "failed"
        assert result.failure_code == "heartbeat_timeout"
        assert result.failure_diagnostics.category == "timeout"
        assert result.failure_diagnostics.fields["native_classification"] == "heartbeat_timeout"
        records = native_records(caplog)
        secondary = [
            item
            for item, _, disposition, _, _ in records
            if disposition == "degraded"
            and item.cause_id == result.failure_diagnostics.diagnostic_id
        ]
        assert len(secondary) == 1
        assert secondary[0].fields["cleanup_confirmed"] is False
        assert owner._pending_candidates["self"] is factory.providers[0]
        assert "PRIVATE_" not in repr([item for item, *_ in records])
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_attached_warmup_failure_preserves_actual_phase_and_retryability(caplog) -> None:
    class Candidate(FakeProvider):
        async def warmup(self):
            raise GpuWorkerRequestError("heartbeat_timeout", {})

    class Factory(FakeProviderFactory):
        async def create(self, request, *, gpu_runtime, on_terminal_failure=None):
            candidate = Candidate(
                request.provider_id,
                gpu_runtime=gpu_runtime,
                channel=request.channel,
                gpu_device_id=request.gpu_device_id,
            )
            self.providers.append(candidate)
            return candidate

    owner, _, _, _ = _owner(provider_factory=Factory())
    caplog.set_level(logging.WARNING)
    try:
        await owner.replace_provider(
            ProviderRuntimeBuildRequest(config=_resolved_config("self")), start=True
        )
        generation = owner.snapshot.channel_for("self").generation
        with pytest.raises(GpuWorkerRequestError):
            await owner.warmup_channel("self")
        assert owner.snapshot.channel_for("self").phase == "running"
        assert owner.snapshot.channel_for("self").generation == generation
        assert owner.snapshot.channel_for("self").provider_live is True
        records = native_records(caplog)
        assert any(
            item.fields.get("warmup_usable") is True
            and disposition == "degraded"
            and item.category == "timeout"
            for item, _, disposition, _, _ in records
        )
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_self_capture_failed_warmup_keeps_actual_capture_running() -> None:
    class Provider(RecordingProvider):
        async def warmup(self, *, operation_id=None, request_id=None):
            raise TimeoutError("PRIVATE_WARMUP_DETAIL")

    diagnostics = []
    owner, _, _, sources, _, _ = build_owner(provider=Provider(), diagnostics=diagnostics)
    try:
        await owner.apply_intent(config(local_gpu=True), enabled=True)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert owner.snapshot.state is SelfCaptureSessionState.RUNNING
        assert owner.snapshot.effective_active is True
        assert sources[0].close_calls == 0
        failures = [item for item in diagnostics if item.event.value == "failure"]
        assert failures
        assert failures[-1].failure_diagnostics.category == "timeout"
        assert "PRIVATE_" not in repr(failures[-1].failure_diagnostics)
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_local_probe_primary_status_survives_secondary_close(monkeypatch, caplog) -> None:
    from puripuly_heart.providers.llm import local_openai
    from puripuly_heart.providers.llm.error_details import LLMResponseError

    class Client:
        def __init__(self, **kwargs):
            pass

        async def translate(self, **kwargs):
            raise LLMResponseError("PRIVATE_PROBE_BODY", provider="local_llm", status_code=503)

        async def close(self):
            raise OSError("PRIVATE_PROBE_CLOSE")

    monkeypatch.setattr(local_openai, "HttpxLocalOpenAIClient", Client)
    caplog.set_level(logging.WARNING)
    assert await local_openai.LocalOpenAICompatibleLLMProvider.verify_connection() is False
    records = native_records(caplog)
    primary = next(item for item, _, _, stage, _ in records if stage == "probe")
    secondary = next(item for item, _, _, stage, _ in records if stage == "cleanup")
    assert primary.status_code == 503
    assert primary.category == "service_unavailable"
    assert secondary.cause_id == primary.diagnostic_id
    assert "PRIVATE_" not in repr([item for item, *_ in records])


@pytest.mark.asyncio
@pytest.mark.parametrize("primary_stage", ["admission", "child_creation", "child_execution"])
async def test_translation_callbacks_retain_primary_and_link_retirement(
    primary_stage, caplog
) -> None:
    from uuid import uuid4

    from puripuly_heart.core.orchestrator.translation_turn import TranslationTurnLifecycleOwner
    from tests.core.test_translation_turn_owner import _request

    caplog.set_level(logging.WARNING)
    operation_id, request_id = uuid4().hex, uuid4().hex
    parent_id = uuid4()

    async def child_created(child):
        if primary_stage == "child_creation":
            raise ConnectionError("PRIVATE_CREATION_FAILURE")

    async def child_started(child, task):
        pass

    async def process_child(child, cancellation_requested):
        raise TimeoutError("PRIVATE_PROCESS_FAILURE")

    async def child_terminal(child, outcome):
        raise OSError("PRIVATE_CHILD_RETIREMENT")

    async def parent_closed(parent):
        raise OSError("PRIVATE_PARENT_RETIREMENT")

    async def parent_rejected(parent):
        pass

    async def parent_admitted(children):
        if primary_stage == "admission":
            raise ConnectionError("PRIVATE_ADMISSION_FAILURE")

    owner = TranslationTurnLifecycleOwner(
        on_child_created=child_created,
        on_child_started=child_started,
        process_child=process_child,
        on_child_terminal=child_terminal,
        on_parent_closed=parent_closed,
        on_parent_rejected=parent_rejected,
        on_parent_admitted=parent_admitted,
    )
    try:
        request = replace(
            _request(parent_id=parent_id, turn_kind="manual"),
            operation_id=operation_id,
            request_id=request_id,
        )
        admitted_ids = await owner.submit(request, wait_for_parent=True)
        assert admitted_ids
        assert owner.is_parent_closed(parent_id)
        records = native_records(caplog)
        primary = next(
            item
            for item, _, disposition, stage, _ in records
            if disposition == "failed" and stage == primary_stage
        )
        retirements = [
            item
            for item, _, disposition, stage, _ in records
            if disposition == "degraded" and stage == "retirement"
        ]
        assert len(retirements) == 2
        assert all(item.cause_id == primary.diagnostic_id for item in retirements)
        assert all(
            item.operation_id == operation_id and item.request_id == request_id
            for item, *_ in records
        )
        assert "PRIVATE_" not in repr([item for item, *_ in records])
    finally:
        await owner.close()
