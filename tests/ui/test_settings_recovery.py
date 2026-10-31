import asyncio
from types import SimpleNamespace

import pytest

from puripuly_heart.app.ports.settings_view import (
    LocaleSettingsIntent,
    PeerVadHangoverIntent,
    PromptApplyIntent,
    ProviderApplyIntent,
    SettingsApplyOutcome,
    SonioxSpeakerDiarizationEdit,
    SystemPromptEdit,
)
from puripuly_heart.app.services.settings.settings_transaction_result import (
    SettingsTransactionResultOwner,
)
from puripuly_heart.config.settings_vnext.facade import load_vnext_settings
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.core.messages import TransactionResult
from puripuly_heart.ui.app import TranslatorApp
from puripuly_heart.ui.settings_recovery import SettingsRecovery
from tests.helpers.ui_application import compose_test_ui_application_boundary


async def test_failed_save_retries_automatically_and_keeps_the_edit_until_saved():
    calls = []
    committed = []
    intent = LocaleSettingsIntent("ja")

    async def apply(value):
        calls.append(value)
        return SettingsApplyOutcome(saved=len(calls) > 1, applied=len(calls) > 1)

    recovery = SettingsRecovery(apply, committed.append, retry_delays=(0,))
    recovery.submit(intent)
    await recovery.drain()
    assert calls == [intent, intent]
    assert committed == [intent]
    assert not recovery.has_pending


async def test_saved_but_unapplied_retries_current_runtime_without_resaving_old_intent():
    calls = []
    committed = []
    intent = PeerVadHangoverIntent(800)

    async def apply(value):
        calls.append(value)
        return SettingsApplyOutcome(saved=True, applied=value is None)

    recovery = SettingsRecovery(apply, committed.append, retry_delays=(0,))
    recovery.submit(intent)
    await recovery.drain()
    assert calls == [intent, None]
    assert committed == [intent]
    assert not recovery.has_pending


async def test_exhausted_save_failure_stays_pending_until_a_newer_edit():
    calls = []
    available = False

    async def apply(value):
        calls.append(value)
        return SettingsApplyOutcome(saved=available, applied=available)

    recovery = SettingsRecovery(apply, lambda _: None, retry_delays=(0,))
    intent = LocaleSettingsIntent("ja")
    recovery.submit(intent)
    await recovery.drain()
    await recovery.drain()
    assert calls == [intent, intent]
    assert recovery.has_pending
    available = True
    latest = LocaleSettingsIntent("ko")
    recovery.submit(latest)
    await recovery.drain()
    assert calls == [intent, intent, latest]
    assert not recovery.has_pending


async def test_newer_edit_supersedes_failed_retry_and_preserves_other_provider_fields():
    calls = []
    first = ProviderApplyIntent((SonioxSpeakerDiarizationEdit(False), SystemPromptEdit("old")))

    async def apply(value):
        calls.append(value)
        return SettingsApplyOutcome(saved=len(calls) > 1, applied=len(calls) > 1)

    async def sleep(_delay):
        recovery.submit(PromptApplyIntent("new"))

    recovery = SettingsRecovery(apply, lambda _: None, sleep=sleep)
    recovery.submit(first)
    await recovery.drain()
    assert len(calls) == 2
    assert calls[1] == ProviderApplyIntent(
        (SonioxSpeakerDiarizationEdit(False), SystemPromptEdit("new"))
    )
    assert not recovery.has_pending


async def test_failed_runtime_recovery_does_not_hide_an_independent_unsaved_edit():
    calls = []
    committed = []

    async def apply(value):
        calls.append(value)
        if value is None:
            return SettingsApplyOutcome(saved=True, applied=True)
        return SettingsApplyOutcome(saved=isinstance(value, LocaleSettingsIntent), applied=False)

    recovery = SettingsRecovery(apply, committed.append, retry_delays=(0,))
    saved = LocaleSettingsIntent("ja")
    unsaved = PeerVadHangoverIntent(900)
    recovery.submit(saved)
    recovery.submit(unsaved)
    await recovery.drain()
    assert recovery.has_pending
    assert committed == [saved]
    assert calls.count(unsaved) == 2
    assert calls.count(None) == 1


async def test_shutdown_during_retry_never_reapplies():
    calls = []

    async def apply(value):
        calls.append(value)
        return SettingsApplyOutcome(saved=False, applied=False)

    async def sleep(_delay):
        recovery.stop()

    recovery = SettingsRecovery(apply, lambda _: None, sleep=sleep)
    recovery.submit(LocaleSettingsIntent("ja"))
    await recovery.drain()
    assert len(calls) == 1
    assert not recovery.has_pending


async def test_submission_during_inflight_apply_is_not_acknowledged_or_lost():
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = []
    committed = []

    async def apply(value):
        calls.append(value)
        if len(calls) == 1:
            entered.set()
            await release.wait()
        return SettingsApplyOutcome(saved=True, applied=True)

    recovery = SettingsRecovery(apply, committed.append)
    first = LocaleSettingsIntent("ja")
    latest = LocaleSettingsIntent("en")
    recovery.submit(first)
    task = asyncio.create_task(recovery.drain())
    await entered.wait()
    recovery.submit(latest)
    release.set()
    await task
    assert calls == [first, latest]
    assert committed == [first, latest]
    assert not recovery.has_pending


@pytest.mark.parametrize("saved", [False, True])
async def test_settings_failures_remain_pending_without_notifications(saved):
    results = SettingsTransactionResultOwner()
    calls = []
    notifications = []
    workers = []

    async def apply_settings(_settings):
        calls.append("save")
        results.set(
            TransactionResult(
                ("settings_commit_success_runtime_degraded" if saved else "settings_commit_failed"),
                None,
                None,
            )
        )
        return False

    async def retry_settings_runtime():
        calls.append("runtime")
        return False

    app = TranslatorApp.__new__(TranslatorApp)
    app._ui_application = compose_test_ui_application_boundary(
        SimpleNamespace(
            settings=AppSettingsVNext(),
            apply_settings=apply_settings,
            retry_settings_runtime=retry_settings_runtime,
        )
    )
    app.application._settings_results = results
    app._run_page_task = workers.append
    app._show_snackbar = lambda *args: notifications.append(args)
    app._ensure_settings_recovery()._retry_delays = (0,)
    app._on_settings_changed(LocaleSettingsIntent("ja"))
    await workers[0]()

    assert app._settings_recovery.has_pending
    assert calls == ["save", "runtime" if saved else "save"]
    assert notifications == []


async def test_unknown_outcome_stops_automatic_retries_and_retains_the_edit():
    calls = []

    async def apply(value):
        calls.append(value)
        return SettingsApplyOutcome(saved=None, applied=False)

    recovery = SettingsRecovery(
        apply, lambda _: pytest.fail("unknown edit acknowledged"), retry_delays=(0, 0)
    )
    intent = LocaleSettingsIntent("ja")
    recovery.submit(intent)
    await recovery.drain()
    await recovery.drain()
    assert calls == [intent]
    assert recovery.has_pending


async def test_shutdown_during_apply_does_not_acknowledge_late_result():
    committed = []

    async def apply(_value):
        recovery.stop()
        return SettingsApplyOutcome(saved=True, applied=True)

    recovery = SettingsRecovery(apply, committed.append)
    recovery.submit(LocaleSettingsIntent("ja"))
    await recovery.drain()
    assert committed == []
    assert not recovery.has_pending


async def test_new_successful_change_rechecks_existing_runtime_failure():
    available = False
    calls = []

    async def apply(value):
        calls.append(value)
        return SettingsApplyOutcome(saved=True, applied=available)

    recovery = SettingsRecovery(apply, lambda _: None, retry_delays=())
    recovery.submit(LocaleSettingsIntent("ja"))
    await recovery.drain()
    assert recovery.has_pending
    available = True
    latest = LocaleSettingsIntent("ko")
    recovery.submit(latest)
    await recovery.drain()
    assert calls[-2:] == [latest, None]
    assert not recovery.has_pending


@pytest.mark.parametrize("merged", [False, True])
@pytest.mark.parametrize("replacement", ["committed prompt", "newer prompt"])
async def test_prompt_reversion_supersedes_failed_autosave_through_editor(
    monkeypatch, tmp_path, merged, replacement
):
    from dataclasses import replace

    from puripuly_heart.ui.views.settings import SettingsView
    from tests.app.test_settings_application_owner import FakeRuntimeEffects
    from tests.app.test_settings_apply_recovery import persistent_application

    monkeypatch.setattr(SettingsView, "_load_secrets", lambda *args: None)
    boundary, settings = persistent_application(tmp_path, FakeRuntimeEffects([]))
    boundary._provider._backend.apply_providers = boundary._settings.application.apply
    current = settings.require_committed()
    initial = replace(
        current,
        intent=replace(
            current.intent,
            prompts=replace(current.intent.prompts, system_prompt_override="committed prompt"),
        ),
    )
    settings.commit_delta(current, initial)
    boundary._settings.projection.remember_all(settings.canonical)
    view = SettingsView()
    boundary._settings.projection.presentation = SimpleNamespace(
        render_settings=lambda **kwargs: view.load_from_settings(**kwargs)
    )
    boundary.refresh_settings_projection()
    ui = TranslatorApp.__new__(TranslatorApp)
    ui._ui_application = boundary
    ui.view_settings = view
    workers = []
    ui._run_page_task = workers.append
    view.on_prompt_apply_settings = ui._on_prompt_apply_settings
    view.on_pending_prompt_changed = ui._on_pending_prompt_changed
    attempts = []
    persist = settings.persistence.persist
    failures_left = 1

    def persist_with_one_failure(*args, **kwargs):
        nonlocal failures_left
        if failures_left:
            failures_left -= 1
            raise OSError("controlled storage fault")
        return persist(*args, **kwargs)

    monkeypatch.setattr(settings.persistence, "persist", persist_with_one_failure)

    def edit_and_blur(text):
        editor = view._prompt_editor
        editor._text_field.value = text
        editor._handle_change(None)
        editor._handle_blur(None)

    async def apply(intent):
        attempts.append(intent)
        return await boundary.apply_settings_change(intent)

    async def retry_delay(_delay):
        edit_and_blur(replacement)

    recovery = SettingsRecovery(
        apply, ui._on_settings_committed, retry_delays=(0,), sleep=retry_delay
    )
    ui._settings_recovery = recovery
    edit_and_blur("discarded prompt")
    if merged:
        recovery.submit(ProviderApplyIntent((SonioxSpeakerDiarizationEdit(False),)))
    await workers[0]()
    assert (
        load_vnext_settings(settings.path).settings.intent.prompts.system_prompt_override
        == replacement
    )
    assert view._prompt_editor.value == replacement
    assert not recovery.has_pending
    if merged:
        assert settings.require_committed().intent.stt.soniox.enable_speaker_diarization is False


def test_unchanged_prompt_blur_does_not_start_autosave(monkeypatch, tmp_path):
    from puripuly_heart.app.services.settings.settings_application import (
        settings_view_surface_snapshots,
    )
    from puripuly_heart.ui.views.settings import SettingsView

    monkeypatch.setattr(SettingsView, "_load_secrets", lambda *args: None)
    view = SettingsView()
    provider, general, prompt, overlay = settings_view_surface_snapshots(AppSettingsVNext())
    view.load_from_settings(
        provider=provider,
        general=general,
        prompt=prompt,
        overlay=overlay,
        config_path=tmp_path / "settings.json",
    )
    ui = TranslatorApp.__new__(TranslatorApp)
    workers = []
    ui._queue_settings_mutation_task = workers.append
    view.on_prompt_apply_settings = ui._on_prompt_apply_settings
    view.on_pending_prompt_changed = ui._on_pending_prompt_changed
    view._prompt_editor._handle_blur(None)
    assert workers == []
    assert getattr(ui, "_settings_recovery", None) is None


async def test_repeated_compensation_retains_edit_without_acknowledging_it():
    intent = LocaleSettingsIntent("ja")
    attempts = []

    async def apply(value):
        attempts.append(value)
        return SettingsApplyOutcome(saved=False, applied=False, reverted=True)

    recovery = SettingsRecovery(
        apply, lambda _: pytest.fail("reverted edit acknowledged"), retry_delays=(0,)
    )
    recovery.submit(intent)
    await recovery.drain()
    assert attempts == [intent, intent]
    assert recovery.has_pending


@pytest.mark.parametrize("merged", [False, True])
async def test_prompt_reversion_during_inflight_apply_is_saved_after_stale_completion(merged):
    entered = asyncio.Event()
    release = asyncio.Event()
    prompt = "committed"
    calls = []

    async def apply(intent):
        nonlocal prompt
        calls.append(intent)
        if len(calls) == 1:
            entered.set()
            await release.wait()
        prompt = (
            intent.value
            if isinstance(intent, PromptApplyIntent)
            else next(edit.value for edit in intent.edits if isinstance(edit, SystemPromptEdit))
        )
        return SettingsApplyOutcome(saved=True, applied=True)

    recovery = SettingsRecovery(apply, lambda _: None)
    recovery.submit(PromptApplyIntent("discarded"))
    if merged:
        recovery.submit(ProviderApplyIntent((SonioxSpeakerDiarizationEdit(False),)))
    task = asyncio.create_task(recovery.drain())
    await entered.wait()
    assert recovery.replace_pending_prompt(PromptApplyIntent("committed"))
    release.set()
    await task
    assert prompt == "committed"
    assert len(calls) == 2
    assert not recovery.has_pending
