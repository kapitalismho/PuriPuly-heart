from __future__ import annotations

import asyncio
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
from puripuly_heart.app.services.settings_application import SettingsApplicationOwner
from puripuly_heart.app.services.settings_projection import SettingsProjectionOwner

from puripuly_heart.app.adapters.settings_vnext_canonical_persistence import (
    SettingsVNextCanonicalPersistenceAdapter,
)
from puripuly_heart.app.ports.settings_runtime_effects import (
    SettingsRuntimeState,
    SettingsRuntimeTransition,
)
from puripuly_heart.app.services.canonical_settings_persistence import SettingsOwner
from puripuly_heart.app.services.manual_local_asr_fallback import (
    ManualLocalASRFallbackOwner,
)
from puripuly_heart.app.services.settings import (
    settings_application as settings_application_module,
)
from puripuly_heart.config.settings import AppSettings, to_dict
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.core.messages import (
    TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
    TransactionResult,
)


class FakeSettingsOwner:
    def __init__(
        self,
        current: AppSettings,
        events: list[str] | None = None,
    ) -> None:
        self.current = current
        self.projection_snapshot = copy.deepcopy(current)
        self.completed = 0
        self.events = events

    @staticmethod
    def legacy_snapshot_values(settings: AppSettings) -> dict[str, object]:
        return to_dict(settings)

    @staticmethod
    def normalize_compatibility(settings: AppSettings) -> None:
        settings.stt.low_latency_mode = True
        settings.ui.integrated_context_enabled = True

    @staticmethod
    def create_legacy_patch_repository(**_kwargs: object) -> object:
        return object()

    def complete(self) -> None:
        self.completed += 1
        if self.events is not None:
            self.events.append("complete")

    def begin(self, **_kwargs: object) -> None:
        if self.events is not None:
            self.events.append("begin")

    def apply_legacy_delta(self, _base: AppSettings, _next: AppSettings) -> None:
        if self.events is not None:
            self.events.append("delta")

    def save_current(self, **_kwargs: object) -> bool:
        if self.events is not None:
            self.events.append("save")
        return True

    def persist(self) -> None:
        if self.events is not None:
            self.events.append("persist")

    def rollback(self) -> None:
        if self.events is not None:
            self.events.append("rollback")

    def remember_projection(self, settings: AppSettings) -> None:
        self.projection_snapshot = copy.deepcopy(settings)
        if self.events is not None:
            self.events.append("remember")


class RecordingMutationService:
    def __init__(self) -> None:
        self.requests: list[object] = []

    async def mutate(self, request: object) -> TransactionResult:
        self.requests.append(request)
        return TransactionResult(
            status=TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
            message=None,
            diagnostics=None,
        )


class FakeRuntimeEffects:
    def __init__(
        self,
        events: list[str],
        failure: BaseException | None = None,
    ) -> None:
        self.events = events
        self.failure = failure
        self.reload_settings_view: list[bool] = []

    async def preserve_before_replace(self, _settings: AppSettings) -> None:
        self.events.append("preserve")

    def capture_runtime_signatures(self) -> None:
        self.events.append("capture")

    async def prepare(
        self,
        current_settings: AppSettings | None,
        next_settings: AppSettings,
    ) -> SettingsRuntimeTransition[AppSettings]:
        self.events.append("prepare")
        return SettingsRuntimeTransition(
            settings=next_settings,
            previous_settings=current_settings,
            previous_locale="en",
            previous_overlay_enabled=False,
            previous_self_signature=None,
            previous_peer_signature=None,
            previous_peer_translation_enabled=False,
            previous_peer_activation_requested=False,
            source_language_changed=False,
            target_language_changed=False,
            effective_peer_source_changed=False,
            effective_peer_target_changed=False,
            peer_source_language_changed=False,
            peer_target_language_changed=False,
            peer_source_mode_changed=False,
            desktop_runtime_controls=(),
        )

    def activate_before_persist(
        self,
        _transition: SettingsRuntimeTransition[AppSettings],
    ) -> None:
        self.events.append("activate")

    async def prepare_overlay_persistence(
        self,
        _previous_settings: AppSettings,
        _next_settings: AppSettings,
    ) -> None:
        return None

    def restore_memory(self, _settings: AppSettings) -> None:
        return None

    def sync_signatures(self, _settings: AppSettings) -> None:
        return None

    def state(self, _settings: AppSettings) -> SettingsRuntimeState:
        return SettingsRuntimeState(
            runtime_available=False,
            self_stt_desired=False,
            self_stt_available=False,
            peer_stt_desired=False,
            peer_stt_available=False,
            qwen_llm_desired=False,
            llm_available=False,
        )

    async def apply_after_persist(
        self,
        _transition: SettingsRuntimeTransition[AppSettings],
        **kwargs: object,
    ) -> None:
        self.events.append("runtime")
        self.reload_settings_view.append(bool(kwargs.get("reload_settings_view", True)))
        if self.failure is not None:
            raise self.failure


@pytest.mark.asyncio
async def test_settings_application_owner_routes_mixed_surfaces_in_transaction_order() -> None:
    settings = FakeSettingsOwner(AppSettings())
    service = RecordingMutationService()
    projection = SettingsProjectionOwner(
        presentation=SimpleNamespace(render_settings=lambda *_args, **_kwargs: True),
        config_path=Path("settings.json"),
        current_settings=lambda: settings.current,
    )
    projection.remember_all(settings.current)

    async def inspect_cpu() -> None:
        return None

    owner = SettingsApplicationOwner(
        settings=settings,
        projection=projection,
        runtime_effects=FakeRuntimeEffects([]),
        manual_fallback=ManualLocalASRFallbackOwner(),
        cpu_auto_available=lambda: True,
        inspect_cpu=inspect_cpu,
        fallback_sink=lambda _channels, _installation: None,
        sync_ui=lambda: None,
        fallback_log_sink=lambda _previous, _normalized, _channels: None,
        mutation_service_provider=lambda: service,
        consume_superseded_settings=lambda _settings: False,
        active_local_asr_change=lambda _base, _next: False,
        failure_sink=lambda _message: None,
    )
    pending = copy.deepcopy(settings.current)
    pending.languages.source_language = "ja"
    pending.overlay.show_translation = False
    pending.ui.locale = "ko"
    pending.system_prompt = "mixed prompt"

    assert await owner.apply(pending)
    assert [request.reason for request in service.requests] == [
        "settings.stt_language_audio",
        "settings.overlay_osc_output",
        "settings.ui_prompt_clipboard_state",
    ]
    assert service.requests[0].values == {"languages.source_language": "ja"}
    assert service.requests[1].values == {"overlay.show_translation": False}
    assert service.requests[2].values == {
        "ui.locale": "ko",
        "system_prompt": "mixed prompt",
    }
    assert settings.current.languages.source_language == "ja"
    assert settings.current.overlay.show_translation is False
    assert settings.current.ui.locale == "ko"
    assert settings.current.system_prompt == "mixed prompt"
    assert settings.completed == 3
    assert owner.results.current is not None
    assert (
        owner.results.current.status == TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED
    )


@pytest.mark.asyncio
async def test_settings_application_owner_can_suppress_language_view_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = FakeSettingsOwner(AppSettings())
    renders: list[object] = []
    projection = SettingsProjectionOwner(
        presentation=SimpleNamespace(
            render_settings=lambda *args, **_kwargs: renders.append(args[0])
        ),
        config_path=Path("settings.json"),
        current_settings=lambda: settings.current,
    )
    projection.remember_all(settings.current)
    effects = FakeRuntimeEffects([])

    class ApplyingMutationService:
        def __init__(self, *, runtime_apply: object, **_kwargs: object) -> None:
            self.runtime_apply = runtime_apply

        async def mutate(self, _request: object) -> TransactionResult:
            await self.runtime_apply.apply_runtime(SimpleNamespace())
            return TransactionResult(
                status=TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
                message=None,
                diagnostics=None,
            )

    monkeypatch.setattr(
        settings_application_module,
        "SettingsMutationService",
        ApplyingMutationService,
    )
    owner = SettingsApplicationOwner(
        settings=settings,
        projection=projection,
        runtime_effects=effects,
        manual_fallback=ManualLocalASRFallbackOwner(),
        cpu_auto_available=lambda: True,
        inspect_cpu=lambda: None,
        fallback_sink=lambda _channels, _installation: None,
        sync_ui=lambda: None,
        fallback_log_sink=lambda _previous, _normalized, _channels: None,
        mutation_service_provider=lambda: None,
        consume_superseded_settings=lambda _settings: False,
        active_local_asr_change=lambda _base, _next: False,
        failure_sink=lambda _message: None,
    )
    pending = copy.deepcopy(settings.current)
    pending.languages.source_language = "ja"

    assert await owner.apply(pending, reload_settings_view=False)

    assert effects.reload_settings_view == [False]
    assert renders == []


@pytest.mark.asyncio
async def test_settings_application_owner_owns_direct_persistence_sequence() -> None:
    events: list[str] = []
    settings = FakeSettingsOwner(AppSettings(), events)
    projection = SettingsProjectionOwner(
        presentation=SimpleNamespace(render_settings=lambda *_args, **_kwargs: True),
        config_path=Path("settings.json"),
        current_settings=lambda: settings.current,
    )
    projection.remember_all(settings.current)
    owner = SettingsApplicationOwner(
        settings=settings,
        projection=projection,
        runtime_effects=FakeRuntimeEffects(events),
        manual_fallback=ManualLocalASRFallbackOwner(),
        cpu_auto_available=lambda: True,
        inspect_cpu=lambda: None,
        fallback_sink=lambda _channels, _installation: None,
        sync_ui=lambda: None,
        fallback_log_sink=lambda _previous, _normalized, _channels: None,
        mutation_service_provider=lambda: None,
        consume_superseded_settings=lambda _settings: False,
        active_local_asr_change=lambda _base, _next: False,
        failure_sink=lambda _message: None,
    )
    pending = copy.deepcopy(settings.current)
    pending.ui.locale = "ja"

    await owner.apply_direct(pending)

    assert events == [
        "preserve",
        "begin",
        "capture",
        "delta",
        "prepare",
        "activate",
        "save",
        "runtime",
        "complete",
    ]
    assert settings.current is pending


@pytest.mark.asyncio
async def test_direct_runtime_failure_finalizes_commit_and_preserves_it_on_next_failed_save(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for index, failure in enumerate((RuntimeError("runtime failed"), asyncio.CancelledError())):
        current = AppSettings()
        persistence = SettingsVNextCanonicalPersistenceAdapter()
        settings = SettingsOwner(
            path=tmp_path / f"settings-{index}.json",
            persistence=persistence,
            canonical=AppSettingsVNext(),
            current=current,
            authoritative=True,
            projection_snapshot=copy.deepcopy(current),
        )
        projection = SettingsProjectionOwner(
            presentation=SimpleNamespace(render_settings=lambda *_args, **_kwargs: True),
            config_path=settings.path,
            current_settings=lambda: settings.current,
        )
        projection.remember_all(current)
        effects = FakeRuntimeEffects([], failure)
        owner = SettingsApplicationOwner(
            settings=settings,
            projection=projection,
            runtime_effects=effects,
            manual_fallback=ManualLocalASRFallbackOwner(),
            cpu_auto_available=lambda: True,
            inspect_cpu=lambda: None,
            fallback_sink=lambda _channels, _installation: None,
            sync_ui=lambda: None,
            fallback_log_sink=lambda _previous, _normalized, _channels: None,
            mutation_service_provider=lambda: None,
            consume_superseded_settings=lambda _settings: False,
            active_local_asr_change=lambda _base, _next: False,
            failure_sink=lambda _message: None,
        )
        committed = copy.deepcopy(current)
        committed.ui.locale = "ja"

        with pytest.raises(type(failure)):
            await owner.apply_direct(
                committed,
                strict_runtime_errors=True,
            )

        assert settings.mutation_depth == 0
        assert settings.rollback_pending is False
        assert settings.current is committed
        assert settings.current.ui.locale == "ja"
        assert settings.projection_snapshot is not None
        assert settings.projection_snapshot.ui.locale == "ja"

        effects.failure = None
        rejected = copy.deepcopy(committed)
        rejected.system_prompt = "must roll back"

        def fail_persist(_path: Path, _settings: AppSettingsVNext) -> None:
            raise OSError("save failed")

        with monkeypatch.context() as patch:
            patch.setattr(persistence, "persist", fail_persist)
            await owner.apply_direct(rejected)

        assert settings.mutation_depth == 0
        assert settings.rollback_pending is False
        assert settings.current is committed
        assert settings.current.ui.locale == "ja"
        assert settings.current.system_prompt != "must roll back"
        assert settings.projection_snapshot is not None
        assert settings.projection_snapshot.ui.locale == "ja"
        assert settings.projection_snapshot.system_prompt != "must roll back"


@pytest.mark.asyncio
async def test_routed_mutation_cancellation_finalizes_committed_settings(
    tmp_path: Path,
) -> None:
    current = AppSettings()
    settings = SettingsOwner(
        path=tmp_path / "settings.json",
        persistence=SettingsVNextCanonicalPersistenceAdapter(),
        canonical=AppSettingsVNext(),
        current=current,
        authoritative=True,
        projection_snapshot=copy.deepcopy(current),
    )
    projection = SettingsProjectionOwner(
        presentation=SimpleNamespace(render_settings=lambda *_args, **_kwargs: True),
        config_path=settings.path,
        current_settings=lambda: settings.current,
    )
    projection.remember_all(current)
    owner = SettingsApplicationOwner(
        settings=settings,
        projection=projection,
        runtime_effects=FakeRuntimeEffects([], asyncio.CancelledError()),
        manual_fallback=ManualLocalASRFallbackOwner(),
        cpu_auto_available=lambda: True,
        inspect_cpu=lambda: None,
        fallback_sink=lambda _channels, _installation: None,
        sync_ui=lambda: None,
        fallback_log_sink=lambda _previous, _normalized, _channels: None,
        mutation_service_provider=lambda: None,
        consume_superseded_settings=lambda _settings: False,
        active_local_asr_change=lambda _base, _next: False,
        failure_sink=lambda _message: None,
    )
    committed = copy.deepcopy(current)
    committed.languages.source_language = "ja"

    with pytest.raises(asyncio.CancelledError):
        await owner.apply(committed)

    assert settings.mutation_depth == 0
    assert settings.rollback_pending is False
    assert settings.compatibility_projection().languages.source_language == "ja"
