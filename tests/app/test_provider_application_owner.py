from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from puripuly_heart.app.services.provider_runtime_apply import ProviderRuntimeApplyPlan
from puripuly_heart.app.services.provider_settings import ProviderApplicationOwner
from puripuly_heart.config.settings import (
    AppSettings,
    LLMProviderName,
    STTProviderName,
    to_dict,
)
from puripuly_heart.core.messages import (
    RUNTIME_APPLY_STATUS_FAILED,
    TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
    TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED,
    RuntimeApplyResult,
    TransactionResult,
)


class FakeSettingsOwner:
    def __init__(self, current: AppSettings, events: list[object]) -> None:
        self.current = current
        self.projection_snapshot = copy.deepcopy(current)
        self.events = events
        self.mutation_depth = 0
        self.rollback_pending = False

    @staticmethod
    def legacy_snapshot_values(settings: AppSettings) -> dict[str, object]:
        return to_dict(settings)

    def project_legacy_delta(
        self,
        _base: AppSettings,
        _next: AppSettings,
    ) -> object:
        return object()

    def create_legacy_patch_repository(self, **_kwargs) -> object:
        return object()

    def begin(self, **_kwargs) -> None:
        self.mutation_depth += 1
        self.rollback_pending = True
        self.events.append("begin")

    def apply_legacy_delta(self, _base: AppSettings, _next: AppSettings) -> None:
        self.events.append("delta")

    def persist(self) -> None:
        self.events.append("persist")

    def rollback(self) -> None:
        self.mutation_depth = 0
        self.rollback_pending = False
        self.events.append("rollback")

    def remember_projection(self, _settings: AppSettings) -> None:
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
    external_result: list[TransactionResult | None],
    order24_patch_provider=lambda _settings: None,
    apply_order24=None,
    events: list[object],
) -> ProviderApplicationOwner:
    async def preserve(_settings: AppSettings) -> None:
        events.append("preserve")

    async def no_order24(_settings: AppSettings) -> bool:
        return False

    def set_result(result: TransactionResult) -> None:
        external_result[0] = result

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
        persist_current_settings=lambda: (
            events.append("persist_current"),
            True,
        )[1],
        save_failure_sink=lambda message: events.append(("failure", message)),
        result_sink=set_result,
        result_provider=lambda: external_result[0],
        sync_memory=lambda _settings: events.append("sync_memory"),
        capture_runtime_signatures=lambda: events.append("capture"),
        sync_signatures=lambda _settings: events.append("signatures"),
        consume_superseded_settings=lambda _settings: False,
        active_local_asr_change=lambda _base, _next: False,
        compensate_local_asr=lambda **_kwargs: SimpleNamespace(),
        copy_runtime_only_ui_state=lambda source, target: (
            setattr(target.ui, "overlay_enabled", source.ui.overlay_enabled),
            setattr(
                target.ui,
                "peer_translation_enabled",
                source.ui.peer_translation_enabled,
            ),
        ),
        llm_retry_pending=lambda: False,
        mark_llm_retry=lambda: events.append("retry"),
    )


@pytest.mark.asyncio
async def test_provider_application_owner_routes_translation_patch_through_mutation_service() -> (
    None
):
    events: list[object] = []
    requests: list[object] = []
    external_result: list[TransactionResult | None] = [None]
    baseline = AppSettings()
    pending = copy.deepcopy(baseline)
    pending.llm.concurrency_limit = baseline.llm.concurrency_limit + 1
    settings = FakeSettingsOwner(baseline, events)
    owner = _owner(
        settings=settings,
        runtime=FakeRuntimeOwner(events),
        service=RecordingMutationService(requests),
        external_result=external_result,
        events=events,
    )

    result = await owner.apply(pending)

    assert result is True
    assert len(requests) == 1
    assert requests[0].values == {
        "llm.concurrency_limit": pending.llm.concurrency_limit,
    }
    assert settings.current.llm.concurrency_limit == pending.llm.concurrency_limit
    assert events[0] == "preserve"
    assert events[-1] == "sync_ui"


@pytest.mark.asyncio
async def test_provider_application_owner_routes_combined_surfaces_in_order() -> None:
    events: list[object] = []
    requests: list[object] = []
    external_result: list[TransactionResult | None] = [None]
    baseline = AppSettings()
    pending = copy.deepcopy(baseline)
    pending.provider.llm = LLMProviderName.LOCAL_LLM
    pending.provider.stt = STTProviderName.SONIOX
    pending.ui.locale = "ja"
    settings = FakeSettingsOwner(baseline, events)

    async def apply_order24(next_settings: AppSettings) -> bool:
        events.append("order24")
        settings.current = next_settings
        external_result[0] = _applied_result()
        return True

    owner = _owner(
        settings=settings,
        runtime=FakeRuntimeOwner(events),
        service=RecordingMutationService(requests),
        external_result=external_result,
        order24_patch_provider=lambda _settings: (
            copy.deepcopy(settings.current),
            {"ui.locale": "ja"},
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
    assert settings.current.provider.llm == LLMProviderName.LOCAL_LLM
    assert settings.current.provider.stt == STTProviderName.SONIOX
    assert settings.current.ui.locale == "ja"


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
        external_result: list[TransactionResult | None] = [None]
        settings = FakeSettingsOwner(AppSettings(), events)
        owner = _owner(
            settings=settings,
            runtime=FakeRuntimeOwner(events, mode=runtime_mode),
            service=RecordingMutationService([]),
            external_result=external_result,
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
        assert external_result[0] is not None
        assert external_result[0].status == expected_status
        assert (
            None if external_result[0].diagnostics is None else external_result[0].diagnostics.code
        ) == expected_code
