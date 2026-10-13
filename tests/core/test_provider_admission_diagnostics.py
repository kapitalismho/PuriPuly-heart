from __future__ import annotations

from dataclasses import replace
from uuid import uuid4

import pytest

from puripuly_heart.app.adapters.self_capture.self_capture_provider import (
    SelfCaptureProviderAdapter,
)
from puripuly_heart.core.diagnostics import DiagnosticHistory, diagnostic_operation
from puripuly_heart.core.local_asr.local_asr_provider_runtime import ProviderRuntimeBuildRequest
from tests.core.runtime.test_local_asr_provider_runtime import (
    FakeProviderFactory,
    _owner,
    _resolved_config,
)
from tests.core.runtime.test_self_capture_session import build_owner, config


@pytest.mark.asyncio
@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("stage", ["provider_build", "provider_warmup"])
async def test_provider_admission_first_retention_has_exact_command_identity(
    monkeypatch, active, stage
):
    from puripuly_heart.core.runtime import local_asr_provider_runtime, self_capture

    class Factory(FakeProviderFactory):
        armed = False

        async def create(self, request, *, gpu_runtime, on_terminal_failure=None):
            if self.armed and stage == "provider_build":
                raise ConnectionError("PRIVATE_PROVIDER_CREATION")
            provider = await super().create(
                request, gpu_runtime=gpu_runtime, on_terminal_failure=on_terminal_failure
            )
            if self.armed:

                async def fail_warmup():
                    raise TimeoutError("PRIVATE_PROVIDER_WARMUP")

                provider.warmup = fail_warmup
            return provider

    factory = Factory()
    runtime, *_ = _owner(provider_factory=factory)

    class Reset:
        async def reset_provider_channel(self, channel):
            return None

    capture, *_ = build_owner(provider=SelfCaptureProviderAdapter(runtime, Reset()))
    capture._provider_request_factory = lambda current, warmup: ProviderRuntimeBuildRequest(
        config=_resolved_config("self", "soniox"),
        warmup=stage == "provider_warmup",
        provider_signature=current.provider_signature,
        runtime_signature=current.runtime_signature,
    )
    instance, operation, request = uuid4().hex, uuid4().hex, uuid4().hex
    history = DiagnosticHistory()
    history.bind_instance(instance)

    def retain(logger, diagnostics, *, severity="error", disposition="failed", **kwargs):
        history.retain(diagnostics, severity=severity, disposition=disposition, **kwargs)

    monkeypatch.setattr(local_asr_provider_runtime, "emit_failure", retain)
    monkeypatch.setattr(self_capture, "emit_failure", retain)
    initial = replace(config(), provider_id="soniox")
    changed = replace(config("two"), provider_id="soniox")
    try:
        if active:
            await capture.apply_intent(initial, enabled=True)
        factory.armed = True
        with diagnostic_operation(instance, operation, request):
            if active:
                snapshot = await capture.apply_intent(changed, enabled=True, restart=True)
            else:
                snapshot = await capture.prepare_provider(changed)
        records = history.query({"operation_id": operation})["records"]
        assert records
        assert any(item["stage"] == stage for item in records), [
            (item["stage"], item["fields"].get("exception_class"), item["code"]) for item in records
        ]
        native = next(item for item in records if item["stage"] == stage)
        assert native["request_id"] == request
        assert native["channel"] == "self"
        assert native["diagnostic_id"] == snapshot.failure_diagnostics.diagnostic_id
        assert native["category"] == ("network" if stage == "provider_build" else "timeout")
        assert "PRIVATE_" not in repr(records)
    finally:
        await capture.close()
        await runtime.close()
