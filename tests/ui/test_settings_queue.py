import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from puripuly_heart.app.ports.settings_view import (
    AudioInputSettingsIntent,
    AudioSettingsIntent,
    CustomVocabularySettingsIntent,
    DesktopAudioOutputSettingsIntent,
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
from puripuly_heart.ui.settings_queue import SettingsQueue
from tests.helpers.ui_application import compose_test_ui_application_boundary


@pytest.mark.parametrize(
    "outcome",
    [
        SettingsApplyOutcome(saved=False, applied=False),
        SettingsApplyOutcome(saved=True, applied=False),
        SettingsApplyOutcome(saved=None, applied=False),
        SettingsApplyOutcome(saved=False, applied=False, reverted=True),
    ],
)
async def test_failed_generation_is_attempted_once_and_unrelated_edits_do_not_revive_it(outcome):
    calls = []
    committed = []
    intent = LocaleSettingsIntent("ja")
    unrelated = PeerVadHangoverIntent(900)

    async def apply(value):
        calls.append(value)
        return outcome if value == intent else SettingsApplyOutcome(saved=True, applied=True)

    queue = SettingsQueue(apply, committed.append)
    queue.submit(intent)
    await queue.drain()
    await queue.drain()
    assert calls == [intent]
    assert queue.has_pending is (not outcome.saved)
    queue.submit(unrelated)
    await queue.drain()
    await queue.drain()
    assert calls == [intent, unrelated]
    assert committed == ([intent, unrelated] if outcome.saved else [unrelated])
    queue.submit(intent)
    await queue.drain()
    assert calls == [intent, unrelated, intent]


async def test_explicit_runtime_only_submission_remains_available_without_automatic_replay():
    calls = []

    async def apply(value):
        calls.append(value)
        return SettingsApplyOutcome(saved=True, applied=False)

    queue = SettingsQueue(apply, lambda _: pytest.fail("runtime-only edit acknowledged"))
    queue.submit(None)
    await queue.drain()
    await queue.drain()
    assert calls == [None]
    assert not queue.has_pending
    queue.submit(None)
    await queue.drain()
    assert calls == [None, None]


async def test_newer_prompt_preserves_other_fields_of_failed_provider_edit():
    calls = []
    first = ProviderApplyIntent((SonioxSpeakerDiarizationEdit(False), SystemPromptEdit("old")))

    async def apply(value):
        calls.append(value)
        return SettingsApplyOutcome(saved=len(calls) > 1, applied=len(calls) > 1)

    queue = SettingsQueue(apply, lambda _: None)
    queue.submit(first)
    await queue.drain()
    queue.submit(PromptApplyIntent("new"))
    await queue.drain()
    assert calls == [
        first,
        ProviderApplyIntent((SonioxSpeakerDiarizationEdit(False), SystemPromptEdit("new"))),
    ]
    assert not queue.has_pending


async def test_audio_fields_coalesce_and_vocabulary_languages_remain_independent():
    calls = []

    async def apply(value):
        calls.append(value)
        return SettingsApplyOutcome(saved=True, applied=True)

    queue = SettingsQueue(apply, lambda _: None)
    queue.submit(AudioSettingsIntent((AudioInputSettingsIntent("Synthetic API", "old"),)))
    queue.submit(AudioSettingsIntent((DesktopAudioOutputSettingsIntent("speakers"),)))
    queue.submit(AudioSettingsIntent((AudioInputSettingsIntent("Synthetic API", "latest"),)))
    japanese = CustomVocabularySettingsIntent("ja", ("Japanese vocabulary",))
    korean = CustomVocabularySettingsIntent("ko", ("Korean vocabulary",))
    queue.submit(CustomVocabularySettingsIntent("ja", ("old vocabulary",)))
    queue.submit(korean)
    queue.submit(japanese)
    await queue.drain()
    assert calls == [
        AudioSettingsIntent(
            (
                AudioInputSettingsIntent("Synthetic API", "latest"),
                DesktopAudioOutputSettingsIntent("speakers"),
            )
        ),
        japanese,
        korean,
    ]


@pytest.mark.parametrize("saved", [False, True])
async def test_new_generation_survives_old_inflight_result_and_committed_acknowledgment(saved):
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = []
    committed = []
    first = LocaleSettingsIntent("ja")
    latest = LocaleSettingsIntent("en")

    async def apply(value):
        calls.append(value)
        if len(calls) == 1:
            entered.set()
            await release.wait()
            return SettingsApplyOutcome(saved=saved, applied=False)
        return SettingsApplyOutcome(saved=True, applied=True)

    queue = SettingsQueue(apply, committed.append)
    queue.submit(first)
    task = asyncio.create_task(queue.drain())
    await entered.wait()
    queue.submit(latest)
    await queue.drain()
    release.set()
    await task
    await queue.drain()
    assert calls == [first, latest]
    assert committed == ([first, latest] if saved else [latest])
    assert not queue.has_pending


async def test_shutdown_during_apply_drops_queued_work_and_late_acknowledgment():
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = []
    committed = []

    async def apply(value):
        calls.append(value)
        entered.set()
        await release.wait()
        return SettingsApplyOutcome(saved=True, applied=True)

    queue = SettingsQueue(apply, committed.append)
    first = LocaleSettingsIntent("ja")
    queue.submit(first)
    task = asyncio.create_task(queue.drain())
    await entered.wait()
    queue.submit(LocaleSettingsIntent("en"))
    queue.stop()
    queue.submit(LocaleSettingsIntent("ko"))
    release.set()
    await task
    await queue.drain()
    assert calls == [first]
    assert committed == []
    assert not queue.has_pending


async def test_unexpected_apply_exception_is_not_replayed_by_a_later_drain():
    calls = []

    async def apply(value):
        calls.append(value)
        raise RuntimeError("controlled unexpected failure")

    queue = SettingsQueue(apply, lambda _: None)
    intent = LocaleSettingsIntent("ja")
    queue.submit(intent)
    with pytest.raises(RuntimeError, match="controlled unexpected failure"):
        await queue.drain()
    await queue.drain()
    assert calls == [intent]
    assert queue.has_pending


@pytest.mark.parametrize("saved", [False, True])
async def test_settings_failures_do_not_notify_or_block_audio_and_telemetry(saved, monkeypatch):
    results = SettingsTransactionResultOwner()
    calls = []
    notifications = []
    workers = []

    async def apply_settings(_settings):
        calls.append("save")
        results.set(
            TransactionResult(
                "settings_commit_success_runtime_degraded" if saved else "settings_commit_failed",
                None,
                None,
            )
        )
        return False

    async def apply_telemetry_enabled(enabled):
        assert enabled is False
        calls.append("telemetry_off")
        results.set(TransactionResult("settings_commit_success_runtime_applied", None, None))
        return True

    async def retry_settings_runtime():
        pytest.fail("implicit runtime retry")

    async def no_sleep(_delay):
        pytest.fail("settings queue must not wait before progressing")

    async def audio():
        calls.append("audio")

    app = TranslatorApp.__new__(TranslatorApp)
    app._ui_application = compose_test_ui_application_boundary(
        SimpleNamespace(
            settings=AppSettingsVNext(),
            apply_settings=apply_settings,
            apply_telemetry_enabled=apply_telemetry_enabled,
            retry_settings_runtime=retry_settings_runtime,
        )
    )
    app.application._settings_results = results
    app._run_page_task = workers.append
    app._show_snackbar = lambda *args: notifications.append(args)
    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    app._on_settings_changed(LocaleSettingsIntent("ja"))
    app._queue_settings_mutation_task(audio)
    app._on_telemetry_enabled_change(False)
    await workers[0]()
    await app._ensure_settings_queue().drain()
    assert calls == ["save", "telemetry_off", "audio"]
    assert app._settings_queue.has_pending is (not saved)
    assert notifications == []


@pytest.mark.parametrize("merged", [False, True])
@pytest.mark.parametrize("phase", ["queued", "inflight", "failed"])
@pytest.mark.parametrize("replacement", ["committed prompt", "newer prompt"])
async def test_latest_prompt_through_real_editor_replaces_queued_inflight_or_failed_edit(
    monkeypatch, tmp_path, merged, phase, replacement
):
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
    entered = asyncio.Event()
    release = asyncio.Event()
    persist = settings.persistence.persist
    failures_left = int(phase == "failed")

    def controlled_persist(*args, **kwargs):
        nonlocal failures_left
        if failures_left:
            failures_left -= 1
            raise OSError("controlled storage fault")
        return persist(*args, **kwargs)

    monkeypatch.setattr(settings.persistence, "persist", controlled_persist)

    def edit_and_blur(text):
        editor = view._prompt_editor
        editor._text_field.value = text
        editor._handle_change(None)
        editor._handle_blur(None)

    async def apply(intent):
        attempts.append(intent)
        if phase == "inflight" and len(attempts) == 1:
            entered.set()
            await release.wait()
        return await boundary.apply_settings_change(intent)

    queue = SettingsQueue(apply, ui._on_settings_committed)
    ui._settings_queue = queue
    edit_and_blur("discarded prompt")
    if merged:
        queue.submit(ProviderApplyIntent((SonioxSpeakerDiarizationEdit(False),)))
    if phase == "inflight":
        task = asyncio.create_task(workers.pop(0)())
        await entered.wait()
    elif phase == "failed":
        await workers.pop(0)()
        await queue.drain()
        assert len(attempts) == 1
    edit_and_blur(replacement)
    if phase == "inflight":
        release.set()
        await task
    while workers:
        await workers.pop(0)()
    assert (
        load_vnext_settings(settings.path).settings.intent.prompts.system_prompt_override
        == replacement
    )
    assert settings.require_committed().intent.prompts.system_prompt_override == replacement
    assert view._prompt_editor.value == replacement
    assert view._prompt_snapshot.system_prompt == replacement
    assert SystemPromptEdit not in view._provider_edits
    assert not view.has_pending_prompt_changes
    assert len(attempts) == (1 if phase == "queued" else 2)
    assert not queue.has_pending
    if merged:
        assert settings.require_committed().intent.stt.soniox.enable_speaker_diarization is False
    view._prompt_editor._handle_blur(None)
    assert workers == []
    assert len(attempts) == (1 if phase == "queued" else 2)


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
    assert getattr(ui, "_settings_queue", None) is None


@pytest.mark.parametrize("provider", [False, True])
async def test_leaving_settings_does_not_resubmit_failed_autosave_but_first_staged_edit_applies(
    provider,
):
    calls = []
    workers = []
    intent = (
        ProviderApplyIntent((SonioxSpeakerDiarizationEdit(False),))
        if provider
        else PromptApplyIntent("unsaved")
    )

    async def apply(value):
        calls.append(value)
        return SettingsApplyOutcome(saved=False, applied=False)

    ui = TranslatorApp.__new__(TranslatorApp)
    ui._ui_application = compose_test_ui_application_boundary(
        SimpleNamespace(settings=AppSettingsVNext())
    )
    ui.page = SimpleNamespace(pop_dialog=lambda: None)
    ui.content_area = SimpleNamespace(update=lambda: None)
    ui.view_about = object()
    ui.view_settings = SimpleNamespace(
        has_provider_changes=provider,
        has_pending_prompt_changes=not provider,
        build_provider_apply_settings=lambda: intent,
        _prompt_editor=SimpleNamespace(value="unsaved"),
    )
    ui._settings_queue = SettingsQueue(apply, lambda _: None)
    ui._run_page_task = workers.append
    ui._current_tab = 1
    ui._on_nav_change(3)
    await workers.pop(0)()
    assert calls == [intent]
    ui._current_tab = 1
    ui._on_nav_change(3)
    assert workers == []
    assert calls == [intent]
    ui._queue_settings_apply(intent)
    await workers.pop(0)()
    assert calls == [intent, intent]


def test_settings_ingress_stays_frozen_after_shutdown():
    ui = TranslatorApp.__new__(TranslatorApp)
    ui._freeze_ui_ingress()
    ui._on_settings_changed(LocaleSettingsIntent("ja"))
    ui._queue_settings_apply(PromptApplyIntent("late"))
    ui._on_pending_prompt_changed(PromptApplyIntent("late"))
    assert ui._settings_mutation_queue == []
    assert getattr(ui, "_settings_queue", None) is None
