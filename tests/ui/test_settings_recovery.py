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
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.core.messages import TransactionResult
from puripuly_heart.ui.app import TranslatorApp
from puripuly_heart.ui.i18n import t
from puripuly_heart.ui.settings_recovery import SettingsRecovery
from tests.helpers.ui_application import compose_test_ui_application_boundary


async def test_failed_save_retries_automatically_and_keeps_the_edit_until_saved():
    calls = []
    committed = []
    states = []
    intent = LocaleSettingsIntent("ja")

    async def apply(value):
        calls.append(value)
        return SettingsApplyOutcome(saved=len(calls) > 1, applied=len(calls) > 1)

    recovery = SettingsRecovery(apply, committed.append, states.append, retry_delays=(0,))
    recovery.submit(intent)
    await recovery.drain()
    assert calls == [intent, intent]
    assert committed == [intent]
    assert recovery.phase == "idle"
    assert not recovery.has_pending
    assert "retrying_save" in states


async def test_saved_but_unapplied_retries_current_runtime_without_resaving_old_intent():
    calls = []
    committed = []
    intent = PeerVadHangoverIntent(800)

    async def apply(value):
        calls.append(value)
        return SettingsApplyOutcome(saved=True, applied=value is None)

    recovery = SettingsRecovery(apply, committed.append, lambda _: None, retry_delays=(0,))
    recovery.submit(intent)
    await recovery.drain()
    assert calls == [intent, None]
    assert committed == [intent]
    assert recovery.phase == "idle"


async def test_exhausted_save_failure_can_be_retried_without_a_listen_toggle():
    calls = []
    available = False

    async def apply(value):
        calls.append(value)
        return SettingsApplyOutcome(saved=available, applied=available)

    recovery = SettingsRecovery(apply, lambda _: None, lambda _: None, retry_delays=(0,))
    intent = LocaleSettingsIntent("ja")
    recovery.submit(intent)
    await recovery.drain()
    assert calls == [intent, intent]
    assert recovery.phase == "save_failed"
    assert recovery.has_pending
    available = True
    recovery.retry()
    await recovery.drain()
    assert calls == [intent, intent, intent]
    assert recovery.phase == "idle"


async def test_newer_edit_supersedes_failed_retry_and_preserves_other_provider_fields():
    calls = []
    first = ProviderApplyIntent((SonioxSpeakerDiarizationEdit(False), SystemPromptEdit("old")))

    async def apply(value):
        calls.append(value)
        return SettingsApplyOutcome(saved=len(calls) > 1, applied=len(calls) > 1)

    async def sleep(_delay):
        recovery.submit(PromptApplyIntent("new"))

    recovery = SettingsRecovery(apply, lambda _: None, lambda _: None, sleep=sleep)
    recovery.submit(first)
    await recovery.drain()
    assert len(calls) == 2
    assert calls[1] == ProviderApplyIntent(
        (SonioxSpeakerDiarizationEdit(False), SystemPromptEdit("new"))
    )
    assert not recovery.has_pending


async def test_failed_runtime_recovery_does_not_hide_an_independent_unsaved_edit():
    async def apply(value):
        if value is None:
            return SettingsApplyOutcome(saved=True, applied=True)
        return SettingsApplyOutcome(saved=isinstance(value, LocaleSettingsIntent), applied=False)

    recovery = SettingsRecovery(apply, lambda _: None, lambda _: None, retry_delays=(0,))
    recovery.submit(LocaleSettingsIntent("ja"))
    recovery.submit(PeerVadHangoverIntent(900))
    await recovery.drain()
    assert recovery.phase == "save_failed"
    assert recovery.has_pending


async def test_shutdown_during_retry_never_reapplies_or_updates_presentation():
    calls = []
    states = []

    async def apply(value):
        calls.append(value)
        return SettingsApplyOutcome(saved=False, applied=False)

    async def sleep(_delay):
        recovery.stop()
        states.clear()

    recovery = SettingsRecovery(apply, lambda _: None, states.append, sleep=sleep)
    recovery.submit(LocaleSettingsIntent("ja"))
    await recovery.drain()
    assert len(calls) == 1
    assert states == []


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

    recovery = SettingsRecovery(apply, committed.append, lambda _: None)
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
async def test_visible_failure_offers_retry_and_clears_only_after_success(saved):
    results = SettingsTransactionResultOwner()
    available = False
    calls = []
    workers = []

    async def apply_settings(_settings):
        calls.append("save")
        results.set(
            TransactionResult(
                (
                    "settings_commit_success_runtime_applied"
                    if available
                    else (
                        "settings_commit_success_runtime_degraded"
                        if saved
                        else "settings_commit_failed"
                    )
                ),
                None,
                None,
            )
        )
        return available

    async def retry_settings_runtime():
        calls.append("runtime")
        return available

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
    app._build_settings_status()
    app._ensure_settings_recovery()._retry_delays = ()
    app._on_settings_changed(LocaleSettingsIntent("ja"))
    await workers[0]()

    assert app._settings_status.visible
    assert app._settings_status_text.value == t(
        "settings.recovery.apply_failed" if saved else "settings.recovery.save_failed"
    )
    assert app._settings_retry_button.visible
    assert not app._settings_retry_button.disabled
    available = True
    app._settings_retry_button.on_click(None)
    await workers[-1]()
    assert not app._settings_status.visible
    assert calls == ["save", "runtime" if saved else "save"]


async def test_unknown_outcome_requires_manual_retry_without_claiming_save_failure():
    calls = []

    async def apply(value):
        calls.append(value)
        return SettingsApplyOutcome(saved=None, applied=False)

    recovery = SettingsRecovery(apply, lambda _: None, lambda _: None, retry_delays=(0, 0))
    recovery.submit(LocaleSettingsIntent("ja"))
    await recovery.drain()
    assert len(calls) == 1
    assert recovery.phase == "unknown_failed"
    recovery.retry()
    await recovery.drain()
    assert len(calls) == 2


async def test_shutdown_during_apply_does_not_acknowledge_or_publish_late_result():
    committed = []
    phases = []

    async def apply(_value):
        recovery.stop()
        phases.clear()
        return SettingsApplyOutcome(saved=True, applied=True)

    recovery = SettingsRecovery(apply, committed.append, phases.append)
    recovery.submit(LocaleSettingsIntent("ja"))
    await recovery.drain()
    assert committed == []
    assert phases == []
    assert not recovery.has_pending


async def test_new_successful_change_rechecks_existing_runtime_failure():
    available = False
    calls = []

    async def apply(value):
        calls.append(value)
        return SettingsApplyOutcome(saved=True, applied=available)

    recovery = SettingsRecovery(apply, lambda _: None, lambda _: None, retry_delays=())
    recovery.submit(LocaleSettingsIntent("ja"))
    await recovery.drain()
    assert recovery.phase == "apply_failed"
    available = True
    latest = LocaleSettingsIntent("ko")
    recovery.submit(latest)
    await recovery.drain()
    assert calls[-2:] == [latest, None]
    assert recovery.phase == "idle"
