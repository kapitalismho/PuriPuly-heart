from __future__ import annotations

from uuid import uuid4

import pytest

from puripuly_heart.app.adapters.peer_capture.peer_capture_provider import (
    PeerCaptureProviderAdapter,
)
from puripuly_heart.core.diagnostics import DiagnosticHistory, diagnostic_operation
from puripuly_heart.core.local_asr.local_asr_provider_runtime import ProviderRuntimeBuildRequest
from puripuly_heart.core.peer_capture import PeerCaptureSessionState
from tests.core.runtime.test_local_asr_provider_runtime import (
    FakeProviderFactory,
    _owner,
    _resolved_config,
)
from tests.core.runtime.test_peer_capture_session import make_config, make_owner


@pytest.mark.asyncio
@pytest.mark.parametrize("admission", ["inactive", "startup", "handoff"])
@pytest.mark.parametrize("stage", ["provider_build", "provider_warmup"])
async def test_peer_admission_retains_initiating_command_and_exact_native_cause(
    monkeypatch, admission, stage
):
    from puripuly_heart.core.runtime import local_asr_provider_runtime, peer_channel

    class Factory(FakeProviderFactory):
        armed = False

        async def create(self, request, *, gpu_runtime, on_terminal_failure=None):
            if self.armed and stage == "provider_build":
                raise ConnectionError("PRIVATE_PEER_PROVIDER_CREATION")
            provider = await super().create(
                request, gpu_runtime=gpu_runtime, on_terminal_failure=on_terminal_failure
            )
            if self.armed:

                async def fail_warmup():
                    raise TimeoutError("PRIVATE_PEER_PROVIDER_WARMUP")

                provider.warmup = fail_warmup
            return provider

    factory = Factory()
    runtime, *_ = _owner(provider_factory=factory)

    class Reset:
        async def reset_provider_channel(self, channel):
            return None

    capture, *_ = make_owner(provider=PeerCaptureProviderAdapter(runtime, Reset()))
    capture._provider_request_factory = lambda config, warmup: ProviderRuntimeBuildRequest(
        config=_resolved_config("peer", config.provider_id),
        warmup=stage == "provider_warmup",
        provider_signature=config.provider_signature,
        runtime_signature=config.runtime_signature,
    )
    instance, initial_operation, initial_request = uuid4().hex, uuid4().hex, uuid4().hex
    operation, request = uuid4().hex, uuid4().hex
    history = DiagnosticHistory()
    history.bind_instance(instance)

    def retain(logger, diagnostics, *, severity="error", disposition="failed", **kwargs):
        history.retain(diagnostics, severity=severity, disposition=disposition, **kwargs)

    monkeypatch.setattr(local_asr_provider_runtime, "emit_failure", retain)
    monkeypatch.setattr(peer_channel, "emit_failure", retain)
    initial = make_config(provider_id="soniox")
    changed = make_config(provider_id="deepgram")
    try:
        if admission == "handoff":
            with diagnostic_operation(instance, initial_operation, initial_request):
                await capture.apply_intent(initial, enabled=True)
            assert capture.snapshot.state is PeerCaptureSessionState.RUNNING
        generation_before = capture.snapshot.generation
        factory.armed = True
        with diagnostic_operation(instance, operation, request):
            if admission == "inactive":
                snapshot = await capture.prepare_provider(changed)
            else:
                snapshot = await capture.apply_intent(changed, enabled=True)
        all_records = history.query({})["records"]
        native = next(record for record in all_records if record["stage"] == stage)
        assert native["operation_id"] == operation
        assert native["request_id"] == request
        assert native["channel"] == "peer"
        assert native["category"] == ("network" if stage == "provider_build" else "timeout")
        assert snapshot.failure_diagnostics is not None
        assert snapshot.failure_diagnostics.diagnostic_id == native["diagnostic_id"]
        assert snapshot.failure_diagnostics.operation_id == operation
        assert snapshot.failure_diagnostics.request_id == request
        assert snapshot.failure_diagnostics.generation == native["generation"]
        assert len(history.query({"operation_id": operation})["records"]) == 1
        assert not history.query({"operation_id": initial_operation})["records"]
        if admission == "handoff":
            assert snapshot.generation == generation_before
            assert snapshot.state is PeerCaptureSessionState.RUNNING
            assert snapshot.effective_active
            factory.armed = False
            with diagnostic_operation(instance, uuid4().hex, uuid4().hex):
                recovered = await capture.apply_intent(changed, enabled=True)
            assert recovered.generation == generation_before
            assert recovered.effective_active
            assert recovered.failure_diagnostics is None
            assert (
                history.query({"operation_id": operation})["records"][0]["diagnostic_id"]
                == native["diagnostic_id"]
            )
        else:
            assert snapshot.state is PeerCaptureSessionState.FAULTED
            assert not snapshot.effective_active
        assert "PRIVATE_" not in repr(all_records)
    finally:
        await capture.close()
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("transition_failed", [False, True])
async def test_peer_application_convergence_preserves_exact_native_cause(transition_failed):
    from types import SimpleNamespace

    from puripuly_heart.app.services.provider.provider_runtime_apply import (
        ProviderRuntimeConvergenceError,
    )
    from puripuly_heart.core.stt.diagnostics import stt_failure_report
    from tests.app.test_peer_application import Harness, Runtime

    native = stt_failure_report(
        ConnectionError("PRIVATE_PEER_CONVERGENCE"),
        provider="local_qwen",
        channel="peer",
        operation="provider_build",
    ).diagnostics

    class FailedRuntime(Runtime):
        @property
        def snapshot(self):
            snapshot = super().snapshot
            return SimpleNamespace(
                desired_active=snapshot.desired_active,
                effective_active=snapshot.effective_active,
                provider_id=snapshot.provider_id,
                runtime_signature=snapshot.runtime_signature,
                provider_status=snapshot.provider_status,
                failure_reason=snapshot.failure_reason,
                failure_diagnostics=native,
            )

    harness = Harness()
    harness.settings.ui.peer_translation_enabled = True
    harness.settings.ui.peer_translation_eula_accepted = True
    owner = harness.owner()
    runtime = FailedRuntime(
        current_signature=("previous-runtime",),
        provider_id="stale-provider",
        runtime_signature=("stale-runtime",),
        apply_converges=False,
        last_local_asr_transition_status="failed" if transition_failed else "idle",
    )
    owner.bind_runtime(runtime)
    with pytest.raises(ProviderRuntimeConvergenceError) as failure:
        await owner.refresh_runtime()
    assert failure.value.failure_diagnostics is native
