from __future__ import annotations

import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
from puripuly_heart.app.services.provider_runtime_apply import ProviderRuntimeApplyPlan
from puripuly_heart.app.services.provider_settings import ProviderApplicationOwner
from puripuly_heart.app.services.settings_transaction_result import (
    SettingsTransactionResultOwner,
)

from puripuly_heart.config.provider_values import STTProviderName
from puripuly_heart.config.settings_vnext import serialization
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.core.messages import (
    RUNTIME_APPLY_STATUS_FAILED,
    TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
    TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED,
    RuntimeApplyResult,
    TransactionResult,
)


class FakeSettingsOwner:
    def __init__(self, current: AppSettingsVNext, events: list[object]) -> None:
        self.canonical = current
        self.projection_snapshot = copy.deepcopy(current)
        self.events = events
        self.mutation_depth = 0
        self.rollback_pending = False

    @staticmethod
    def legacy_snapshot_values(settings: AppSettingsVNext) -> dict[str, object]:
        return serialization.to_dict(settings)

    def project_legacy_delta(
        self,
        _base: AppSettingsVNext,
        _next: AppSettingsVNext,
    ) -> object:
        return object()

    def create_legacy_patch_repository(self, **_kwargs) -> object:
        return object()

    def begin(self, **_kwargs) -> None:
        self.mutation_depth += 1
        self.rollback_pending = True
        self.events.append("begin")

    def apply_legacy_delta(self, _base: AppSettingsVNext, _next: AppSettingsVNext) -> None:
        self.events.append("delta")

    def persist(self) -> None:
        self.events.append("persist")

    def save_current(self, **_kwargs: object) -> bool:
        self.events.append("persist_current")
        return True

    def rollback(self) -> None:
        self.mutation_depth = 0
        self.rollback_pending = False
        self.events.append("rollback")

    def remember_projection(self, _settings: AppSettingsVNext) -> None:
        self.events.append("projection")

    def complete(self) -> None:
        self.mutation_depth = max(0, self.mutation_depth - 1)
        self.rollback_pending = self.mutation_depth > 0
        self.events.append("complete")


class FakeRuntimeOwner:
    def __init__(self, events: list[object], *, mode: str = "success") -> None:
        self.events = events
        self.mode = mode

    def build_plan(self, _settings: object, **_kwargs) -> ProviderRuntimeApplyPlan:
        self.events.append("plan")
        return ProviderRuntimeApplyPlan(False, False, False)

    async def apply(
        self,
        _settings: object,
        _plan: ProviderRuntimeApplyPlan,
    ) -> None:
        self.events.append("runtime")
        if self.mode == "exception":
            raise RuntimeError("runtime failed")

    def unavailable_result(
        self,
        _settings: object,
        _plan: ProviderRuntimeApplyPlan,
        **_kwargs,
    ) -> RuntimeApplyResult | None:
        if self.mode != "unavailable":
            return None
        return RuntimeApplyResult(
            status=RUNTIME_APPLY_STATUS_FAILED,
            message=None,
            diagnostics=SimpleNamespace(code="provider_runtime_apply_unavailable"),
        )


class RecordingMutationService:
    def __init__(self, requests: list[object]) -> None:
        self.requests = requests

    async def mutate(self, request) -> TransactionResult:
        self.requests.append(request)
        return _applied_result()


def _applied_result() -> TransactionResult:
    return TransactionResult(
        status=TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
        message=None,
        diagnostics=None,
    )


def _owner(
    *,
    settings: FakeSettingsOwner,
    runtime: FakeRuntimeOwner,
    service: RecordingMutationService,
    results: SettingsTransactionResultOwner,
    order24_patch_provider=lambda _settings: None,
    apply_order24=None,
    events: list[object],
) -> ProviderApplicationOwner:
    async def preserve(_settings: AppSettingsVNext) -> None:
        events.append("preserve")

    async def no_order24(_settings: AppSettingsVNext) -> bool:
        return False

    return ProviderApplicationOwner(
        settings=settings,
        runtime=runtime,
        merge_settings=copy.deepcopy,
        preserve_before_replace=preserve,
        sync_ui=lambda: events.append("sync_ui"),
        order24_patch_provider=order24_patch_provider,
        apply_order24=apply_order24 or no_order24,
        remember_order22=lambda _settings: events.append("remember"),
        mutation_service_provider=lambda: service,
        save_failure_sink=lambda message: events.append(("failure", message)),
        results=results,
        sync_memory=lambda _settings: events.append("sync_memory"),
        capture_runtime_signatures=lambda: events.append("capture"),
        sync_signatures=lambda _settings: events.append("signatures"),
        consume_superseded_settings=lambda _settings: False,
        active_local_asr_change=lambda _base, _next: False,
        compensate_local_asr=lambda **_kwargs: SimpleNamespace(),
        llm_retry_pending=lambda: False,
        mark_llm_retry=lambda: events.append("retry"),
    )


@pytest.mark.asyncio
async def test_provider_application_owner_routes_translation_patch_through_mutation_service() -> (
    None
):
    events: list[object] = []
    requests: list[object] = []
    results = SettingsTransactionResultOwner()
    baseline = AppSettingsVNext()
    pending = replace(
        baseline,
        intent=replace(
            baseline.intent,
            translation=replace(
                baseline.intent.translation,
                concurrency_limit=baseline.intent.translation.concurrency_limit + 1,
            ),
        ),
    )
    settings = FakeSettingsOwner(baseline, events)
    owner = _owner(
        settings=settings,
        runtime=FakeRuntimeOwner(events),
        service=RecordingMutationService(requests),
        results=results,
        events=events,
    )

    result = await owner.apply(pending)

    assert result is True
    assert len(requests) == 1
    assert requests[0].values == {
        "intent.translation.concurrency_limit": pending.intent.translation.concurrency_limit,
    }
    assert (
        settings.canonical.intent.translation.concurrency_limit
        == pending.intent.translation.concurrency_limit
    )
    assert events[0] == "preserve"
    assert events[-1] == "sync_ui"


@pytest.mark.asyncio
async def test_provider_application_owner_routes_combined_surfaces_in_order() -> None:
    events: list[object] = []
    requests: list[object] = []
    results = SettingsTransactionResultOwner()
    baseline = AppSettingsVNext()
    pending = replace(
        baseline,
        intent=replace(
            baseline.intent,
            translation=replace(
                baseline.intent.translation,
                model="local_llm",
                connection="ollama",
            ),
            stt=replace(baseline.intent.stt, provider=STTProviderName.SONIOX.value),
            ui=replace(baseline.intent.ui, locale="ja"),
        ),
    )
    settings = FakeSettingsOwner(baseline, events)

    async def apply_order24(next_settings: AppSettingsVNext) -> bool:
        events.append("order24")
        settings.canonical = next_settings
        results.set(_applied_result())
        return True

    owner = _owner(
        settings=settings,
        runtime=FakeRuntimeOwner(events),
        service=RecordingMutationService(requests),
        results=results,
        order24_patch_provider=lambda _settings: (
            copy.deepcopy(settings.canonical),
            {"intent.ui.locale": "ja"},
        ),
        apply_order24=apply_order24,
        events=events,
    )

    result = await owner.apply(pending)

    assert result is True
    assert [request.reason for request in requests] == [
        "settings.translation_provider",
        "settings.stt_language_audio",
    ]
    assert events.count("order24") == 1
    assert settings.canonical.intent.translation.model == "local_llm"
    assert settings.canonical.intent.stt.provider == STTProviderName.SONIOX.value
    assert settings.canonical.intent.ui.locale == "ja"


@pytest.mark.asyncio
async def test_provider_application_owner_force_rebuild_owns_direct_apply_sequence() -> None:
    cases = [
        (
            "success",
            TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
            None,
        ),
        (
            "exception",
            TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED,
            "provider_runtime_apply_exception",
        ),
        (
            "unavailable",
            TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED,
            "provider_runtime_apply_unavailable",
        ),
    ]
    for runtime_mode, expected_status, expected_code in cases:
        events: list[object] = []
        results = SettingsTransactionResultOwner()
        settings = FakeSettingsOwner(AppSettingsVNext(), events)
        owner = _owner(
            settings=settings,
            runtime=FakeRuntimeOwner(events, mode=runtime_mode),
            service=RecordingMutationService([]),
            results=results,
            events=events,
        )

        result = await owner.apply(force_rebuild_llm=True)

        assert result is True
        assert events[:7] == [
            "preserve",
            "begin",
            "capture",
            "delta",
            "plan",
            "persist_current",
            "runtime",
        ]
        assert "remember" in events
        assert events[-2:] == ["complete", "sync_ui"]
        assert settings.mutation_depth == 0
        assert settings.rollback_pending is False
        assert results.current is not None
        assert results.current.status == expected_status
        assert (
            None if results.current.diagnostics is None else results.current.diagnostics.code
        ) == expected_code


@pytest.mark.asyncio
async def test_provider_application_owner_runtime_only_apply_does_not_persist() -> None:
    events: list[object] = []
    owner = _owner(
        settings=FakeSettingsOwner(AppSettingsVNext(), events),
        runtime=FakeRuntimeOwner(events),
        service=RecordingMutationService([]),
        results=SettingsTransactionResultOwner(),
        events=events,
    )

    result = await owner.apply(persist_settings=False)

    assert result is True
    assert events == ["preserve", "capture", "plan", "runtime", "sync_ui"]


@pytest.mark.asyncio
async def test_provider_application_owner_runtime_only_apply_can_preserve_ui_draft() -> None:
    events: list[object] = []
    owner = _owner(
        settings=FakeSettingsOwner(AppSettingsVNext(), events),
        runtime=FakeRuntimeOwner(events),
        service=RecordingMutationService([]),
        results=SettingsTransactionResultOwner(),
        events=events,
    )

    result = await owner.apply(persist_settings=False, refresh_ui=False)

    assert result is True
    assert events == ["preserve", "capture", "plan", "runtime"]
