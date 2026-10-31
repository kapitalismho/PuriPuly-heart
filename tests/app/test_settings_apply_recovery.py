import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from puripuly_heart.app.adapters.ui_runtime import UiSettingsRuntimeAdapter
from puripuly_heart.app.ports.settings_view import LocaleSettingsIntent
from puripuly_heart.app.services.canonical_settings_persistence import compose_settings_owner
from puripuly_heart.app.services.settings.settings_application import SettingsApplicationOwner
from puripuly_heart.app.services.settings.settings_projection import SettingsProjectionOwner
from puripuly_heart.app.services.settings.settings_transaction_result import (
    SettingsTransactionResultOwner,
)
from puripuly_heart.config.settings_vnext.facade import load_vnext_settings
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.core.messages import TransactionResult
from puripuly_heart.ui.settings_queue import SettingsQueue
from tests.app.test_settings_application_owner import FakeRuntimeEffects
from tests.helpers.ui_application import compose_test_ui_application_boundary


@pytest.mark.parametrize(
    ("status", "saved", "applied"),
    (
        ("settings_commit_failed", False, False),
        ("settings_commit_success_runtime_degraded", True, False),
        ("settings_commit_success_runtime_applied", True, True),
    ),
)
async def test_ui_receives_actual_save_and_apply_result(status, saved, applied):
    results = SettingsTransactionResultOwner()

    async def apply_settings(_settings):
        results.set(TransactionResult(status, None, None))
        return applied

    app = compose_test_ui_application_boundary(
        SimpleNamespace(settings=AppSettingsVNext(), apply_settings=apply_settings)
    )
    app._settings_results = results
    result = await app.apply_settings_change(LocaleSettingsIntent("ja"))
    assert result.saved is saved
    assert result.applied is applied
    assert result.transaction.status == status


async def test_ui_result_is_scoped_to_the_attempt_not_the_last_global_failure():
    results = SettingsTransactionResultOwner()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def apply_settings(_settings):
        results.set(TransactionResult("settings_commit_failed", None, None))
        entered.set()
        await release.wait()
        return False

    app = compose_test_ui_application_boundary(
        SimpleNamespace(settings=AppSettingsVNext(), apply_settings=apply_settings)
    )
    app._settings_results = results
    task = asyncio.create_task(app.apply_settings_change(LocaleSettingsIntent("ja")))
    await entered.wait()
    results.set(TransactionResult("settings_commit_success_runtime_degraded", None, None))
    release.set()
    result = await task
    assert result.saved is False


async def test_retry_uses_current_saved_settings_and_does_not_replay_the_edit():
    calls = []

    async def retry_settings_runtime():
        calls.append("retry_runtime")
        return True

    async def apply_settings(_settings):
        raise AssertionError("runtime retry must not resave the old draft")

    app = compose_test_ui_application_boundary(
        SimpleNamespace(
            settings=AppSettingsVNext(),
            apply_settings=apply_settings,
            retry_settings_runtime=retry_settings_runtime,
        )
    )
    result = await app.apply_settings_change(None)
    assert calls == ["retry_runtime"]
    assert result.saved is True
    assert result.applied is True


def persistent_application(tmp_path, effects):
    settings = compose_settings_owner(tmp_path / "settings.json")
    settings.persistence.persist(settings.path, AppSettingsVNext())
    settings.start()
    settings.set_peer_translation_enabled(True)
    projection = SettingsProjectionOwner(
        presentation=SimpleNamespace(render_settings=lambda **_kwargs: True),
        config_path=settings.path,
        current_settings=lambda: settings.canonical,
    )
    projection.remember_all(settings.canonical)
    owner = SettingsApplicationOwner(
        settings=settings,
        projection=projection,
        runtime_effects=effects,
        sync_ui=lambda: None,
        mutation_service_provider=lambda: None,
        consume_superseded_settings=lambda _: False,
        active_local_asr_change=lambda *_: False,
    )
    app = compose_test_ui_application_boundary(SimpleNamespace(settings=settings.canonical))
    app._settings_results = owner.results
    app._settings = UiSettingsRuntimeAdapter(
        settings=settings,
        projection=projection,
        application=owner,
        merge_provider_settings=lambda value: value,
        telemetry_enabled_settings=lambda value, _: value,
    )
    return app, settings


async def test_runtime_recovery_reloads_saved_configuration_without_another_disk_write(tmp_path):
    effects = FakeRuntimeEffects([], RuntimeError("controlled runtime failure"))
    app, settings = persistent_application(tmp_path, effects)
    result = await app.apply_settings_change(LocaleSettingsIntent("ja"))
    assert result.saved is True
    assert result.applied is False
    assert load_vnext_settings(settings.path).settings.intent.ui.locale == "ja"

    current = settings.require_committed()
    latest = replace(
        current, intent=replace(current.intent, ui=replace(current.intent.ui, locale="ko"))
    )
    settings.commit_delta(current, latest)
    contents = settings.path.read_bytes()
    modified = settings.path.stat().st_mtime_ns
    effects.failure = None
    result = await app.apply_settings_change(None)
    assert result.saved and result.applied
    assert settings.path.read_bytes() == contents
    assert settings.path.stat().st_mtime_ns == modified
    assert settings.require_committed().intent.ui.locale == "ko"
    assert settings.peer_translation_enabled() is True


async def test_unsaved_edit_requires_explicit_resubmission_and_survives_reload(
    tmp_path, monkeypatch
):
    app, settings = persistent_application(tmp_path, FakeRuntimeEffects([]))
    original = settings.path.read_bytes()
    persist = settings.persistence.persist
    available = False

    def controlled_persist(*args, **kwargs):
        if not available:
            raise OSError("controlled storage failure")
        return persist(*args, **kwargs)

    monkeypatch.setattr(settings.persistence, "persist", controlled_persist)
    queue = SettingsQueue(app.apply_settings_change, lambda _: None)
    intent = LocaleSettingsIntent("ja")
    queue.submit(intent)
    await queue.drain()
    assert queue.has_pending
    assert settings.path.read_bytes() == original
    assert settings.require_committed().intent.ui.locale != "ja"
    assert settings.peer_translation_enabled() is True
    available = True
    await queue.drain()
    assert settings.path.read_bytes() == original
    queue.submit(intent)
    await queue.drain()
    assert not queue.has_pending
    assert load_vnext_settings(settings.path).settings.intent.ui.locale == "ja"
    assert settings.peer_translation_enabled() is True


async def test_runtime_retry_rebuilds_failed_llm_without_restarting_healthy_capture(tmp_path):
    from puripuly_heart.app.services.provider.provider_runtime_apply import ProviderRuntimeState
    from tests.app.test_provider_runtime_owner import _runtime_owner

    app, settings = persistent_application(tmp_path, FakeRuntimeEffects([]))
    current = settings.require_committed()
    events = []
    runtime = _runtime_owner(
        events=events,
        state=ProviderRuntimeState(True, True, True, True, False, True),
        current=current,
        cache=(("self", current), ("peer", current, None), ()),
        convergence=lambda _: True,
        peer_convergence=lambda _: True,
    )
    runtime.gpu_restart_decision = lambda *_: False
    app._settings.application.provider_runtime = runtime
    result = await app.apply_settings_change(None)
    assert result.saved and result.applied
    assert "llm" in events
    assert "peer" not in events
    assert "self" not in events
    assert settings.peer_translation_enabled() is True


@pytest.mark.parametrize("failure", [RuntimeError("controlled failure"), asyncio.CancelledError()])
async def test_missing_receipt_is_unknown_and_cancellation_is_not_a_failed_save(failure):
    async def apply_settings(_settings):
        raise failure

    app = compose_test_ui_application_boundary(
        SimpleNamespace(settings=AppSettingsVNext(), apply_settings=apply_settings)
    )
    if isinstance(failure, asyncio.CancelledError):
        with pytest.raises(asyncio.CancelledError):
            await app.apply_settings_change(LocaleSettingsIntent("ja"))
    else:
        result = await app.apply_settings_change(LocaleSettingsIntent("ja"))
        assert result.saved is None
        assert result.applied is False


@pytest.mark.parametrize("failure", [None, "save", "runtime"])
async def test_capture_target_change_reports_saved_and_applied_separately(
    tmp_path, monkeypatch, failure
):
    from puripuly_heart.app.adapters.ui_runtime import UiPeerCaptureRuntimeAdapter
    from puripuly_heart.app.ports.settings_view import LoopbackCaptureSettingsIntent
    from puripuly_heart.app.services.capture.peer_capture_target_application import (
        PeerCaptureTargetApplicationOwner,
    )

    app, settings = persistent_application(tmp_path, FakeRuntimeEffects([]))

    async def apply_runtime(_settings):
        if failure == "runtime":
            raise RuntimeError("controlled capture restart failure")
        return True

    target = PeerCaptureTargetApplicationOwner(
        settings=settings,
        localize=lambda key: key,
        processes=SimpleNamespace(candidates=lambda: ()),
        devices=SimpleNamespace(names=lambda: ("Synthetic Speakers",)),
        runtime_effects=SimpleNamespace(apply_capture_target=apply_runtime),
        settings_presentation_sink=lambda _: None,
        warning_reset=lambda: None,
        stop_audio_test=lambda: asyncio.sleep(0),
        transaction_result_sink=app._settings_results.set,
    )
    app._peer_capture = UiPeerCaptureRuntimeAdapter(
        peer=SimpleNamespace(target=target),
        overlay=SimpleNamespace(),
    )
    if failure == "save":

        def fail_persist(*_args, **_kwargs):
            raise OSError("controlled storage failure")

        monkeypatch.setattr(settings.persistence, "persist", fail_persist)

    result = await app.apply_settings_change(
        LoopbackCaptureSettingsIntent("device:Synthetic Speakers")
    )
    assert result.saved is (failure != "save")
    assert result.applied is (failure is None)
    assert settings.peer_translation_enabled() is True


@pytest.mark.parametrize("active", [False, True])
async def test_transient_local_asr_failure_preserves_requested_change_after_compensation(
    tmp_path, active
):
    from puripuly_heart.app.language_selection import LanguageSelectionChange
    from puripuly_heart.app.services.settings.settings_application import (
        materialize_language_selection,
    )
    from puripuly_heart.composition.ui_application import compose_ui_application
    from puripuly_heart.ui.presentation_adapter import FletUiPresentationAdapter

    composed = compose_ui_application(
        presentation=FletUiPresentationAdapter(SimpleNamespace(debug_ui_preview=False)),
        config_path=tmp_path / "composition.json",
    )
    predicate = composed._settings.application.active_local_asr_change
    closure = dict(
        zip(predicate.__code__.co_freevars, (cell.cell_contents for cell in predicate.__closure__))
    )
    closure["pipeline"].self_capture = SimpleNamespace(
        snapshot=SimpleNamespace(desired_active=active)
    )

    class TransientFailure(FakeRuntimeEffects):
        async def apply_after_persist(self, transition, **kwargs):
            self.events.append("runtime")
            if self.events.count("runtime") == 1:
                raise RuntimeError("controlled capture apply failure")

    app, settings = persistent_application(tmp_path / "settings", TransientFailure([]))
    app._settings.application.active_local_asr_change = predicate
    original = settings.require_committed()
    language = original.intent.languages
    intent = LanguageSelectionChange(
        source_code="ja",
        target_code=language.target_language,
        secondary_target_code=language.secondary_target_language,
        peer_source_code=language.peer_source_language,
        peer_target_code=language.peer_target_language,
        peer_source_mode=language.peer_source_mode,
        recent_source_codes=(),
        recent_target_codes=(),
    )
    assert predicate(original, materialize_language_selection(original, intent)) is active
    attempts = []
    committed = []

    async def apply(value):
        result = await app.apply_settings_change(value)
        attempts.append((value, result.saved, result.applied))
        return result

    queue = SettingsQueue(apply, committed.append)
    queue.submit(intent)
    await queue.drain()
    await queue.drain()
    assert attempts == [(intent, not active, False)]
    assert committed == ([] if active else [intent])
    assert queue.has_pending is active
    assert load_vnext_settings(settings.path).settings.intent.languages.source_language == (
        original.intent.languages.source_language if active else "ja"
    )
    queue.submit(intent)
    await queue.drain()
    assert load_vnext_settings(settings.path).settings.intent.languages.source_language == "ja"
    assert attempts == [(intent, not active, False), (intent, True, True)]
    assert committed == ([intent] if active else [intent, intent])
    assert not queue.has_pending
    assert settings.peer_translation_enabled()


async def test_failed_compensation_keeps_saved_intent_for_runtime_retry(tmp_path, monkeypatch):
    from puripuly_heart.app.ports.settings_view import (
        AudioSettingsIntent,
        DesktopAudioOutputSettingsIntent,
    )

    app, settings = persistent_application(
        tmp_path, FakeRuntimeEffects([], RuntimeError("controlled runtime failure"))
    )
    app._settings.application.active_local_asr_change = lambda *_: True
    original_commit = type(settings).commit_delta_async

    async def fail_rollback(owner, base, next_settings):
        if next_settings.intent.desktop_audio.output_device == "":
            raise OSError("controlled rollback storage failure")
        return await original_commit(owner, base, next_settings)

    monkeypatch.setattr(type(settings), "commit_delta_async", fail_rollback)
    result = await app.apply_settings_change(
        AudioSettingsIntent((DesktopAudioOutputSettingsIntent("Synthetic output"),))
    )
    assert result.saved is True
    assert result.applied is False
    assert not result.reverted
    assert (
        load_vnext_settings(settings.path).settings.intent.desktop_audio.output_device
        == "Synthetic output"
    )


async def test_reverted_outcome_is_scoped_to_the_current_attempt():
    results = SettingsTransactionResultOwner()
    attempts = 0

    async def apply_settings(_settings):
        nonlocal attempts
        attempts += 1
        results.set(
            TransactionResult(
                (
                    "settings_commit_success_runtime_degraded"
                    if attempts == 1
                    else "settings_commit_success_runtime_applied"
                ),
                None,
                None,
            )
        )
        if attempts == 1:
            results.mark_settings_reverted()
        return True

    app = compose_test_ui_application_boundary(
        SimpleNamespace(settings=AppSettingsVNext(), apply_settings=apply_settings)
    )
    app._settings_results = results
    first = await app.apply_settings_change(LocaleSettingsIntent("ja"))
    second = await app.apply_settings_change(LocaleSettingsIntent("ja"))
    assert first.reverted and first.saved is False and first.applied is False
    assert not second.reverted and second.saved is True and second.applied is True
