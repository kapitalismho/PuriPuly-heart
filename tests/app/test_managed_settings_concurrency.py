from __future__ import annotations

import asyncio
import multiprocessing
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from puripuly_heart.app.adapters.settings_vnext_canonical_persistence import (
    SettingsVNextCanonicalPersistenceAdapter,
)
from puripuly_heart.app.ports.settings_repository import SettingsCommitRequest
from puripuly_heart.app.ports.settings_runtime_effects import SettingsRuntimeTransition
from puripuly_heart.app.services.canonical_settings_persistence import SettingsOwner
from puripuly_heart.app.services.settings.settings_application import SettingsApplicationOwner
from puripuly_heart.app.services.settings.settings_projection import SettingsProjectionOwner
from puripuly_heart.app.services.settings.settings_runtime_effects import (
    SettingsRuntimeEffectsAdapter,
)
from puripuly_heart.app.wiring.wiring_managed_auth_factory import (
    CanonicalManagedIdentityStateAdapter,
)
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.core.messages import TransactionResult
from puripuly_heart.core.owned_thread import run_owned_thread_call

REFERENCE = "synthetic-managed-reference-258"


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "home"
    root.mkdir()
    for key in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "TEMP", "TMP"):
        monkeypatch.setenv(key, str(root))


class GatedPersistence(SettingsVNextCanonicalPersistenceAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()
        self.settled = threading.Event()
        self.armed = False
        self.fail_at: int | None = None
        self.attempts = 0
        self.threads: list[int] = []

    def persist(self, path: Path, settings: AppSettingsVNext) -> None:
        if not self.armed:
            super().persist(path, settings)
            return
        self.attempts += 1
        attempt = self.attempts
        self.threads.append(threading.get_ident())
        try:
            if attempt == 1:
                self.started.set()
                if not self.release.wait(10):
                    raise TimeoutError("synthetic persistence gate was not released")
            if attempt == self.fail_at:
                raise OSError("synthetic canonical persistence failure")
            super().persist(path, settings)
        finally:
            if attempt == 1:
                self.settled.set()


def _owner(path: Path) -> tuple[SettingsOwner, GatedPersistence]:
    persistence = GatedPersistence()
    persistence.persist(path, AppSettingsVNext())
    owner = SettingsOwner(path=path, persistence=persistence)
    owner.start()
    owner.remember_projection(owner.require_canonical())
    persistence.armed = True
    return owner, persistence


def _locale(settings: AppSettingsVNext, locale: str) -> AppSettingsVNext:
    return replace(
        settings, intent=replace(settings.intent, ui=replace(settings.intent.ui, locale=locale))
    )


def _launch(owner: SettingsOwner, kind: str):
    admitted = asyncio.Event()
    base = owner.require_canonical()
    repository = owner.create_canonical_patch_repository(
        base_settings=base,
        committed_settings=base,
        surface="ui_prompt_clipboard_state",
    )
    identity = CanonicalManagedIdentityStateAdapter(owner)
    identity.active_managed_credential_ref = REFERENCE

    async def operation():
        admitted.set()
        if kind == "managed":
            await run_owned_thread_call(identity.persist)
            return None
        return await repository.save(
            SettingsCommitRequest(
                values={"intent.ui.locale": "ja"},
                expected_revision=None,
                reason="settings.ui_prompt_clipboard_state",
            )
        )

    return asyncio.create_task(operation()), admitted, repository


async def _entered(persistence: GatedPersistence) -> None:
    assert await asyncio.to_thread(persistence.started.wait, 5)


def _assert_state(owner: SettingsOwner, *, locale: str, reference: str | None) -> None:
    fresh = SettingsVNextCanonicalPersistenceAdapter().load_active(owner.path).canonical_settings
    for settings in (
        owner.require_canonical(),
        owner.committed_settings,
        owner.projection_snapshot,
        fresh,
    ):
        assert settings is not None
        assert settings.intent.ui.locale == locale
        assert settings.state.managed_connection.active_managed_credential_ref == reference
    assert owner.mutation_depth == 0
    assert owner.rollback_pending is False


async def _progress(owner: SettingsOwner, locale: str = "ko") -> None:
    base = owner.require_canonical()
    committed = await owner.commit_delta_async(base, _locale(base, locale))
    assert committed.intent.ui.locale == locale
    _assert_state(owner, locale=locale, reference=REFERENCE)


@pytest.mark.asyncio
async def test_sequential_ordinary_and_retained_managed_changes_remain_durable(tmp_path: Path):
    owner, persistence = _owner(tmp_path / "settings.json")
    persistence.release.set()
    identity = CanonicalManagedIdentityStateAdapter(owner)
    ordinary, _, _ = _launch(owner, "ordinary")
    assert (await ordinary).succeeded
    identity.active_managed_credential_ref = REFERENCE
    await run_owned_thread_call(identity.persist)
    _assert_state(owner, locale="ja", reference=REFERENCE)
    await _progress(owner)
    identity.founder_letter_seen_credential_ref = REFERENCE
    await run_owned_thread_call(identity.persist)
    _assert_state(owner, locale="ko", reference=REFERENCE)
    fresh = SettingsVNextCanonicalPersistenceAdapter().load_active(owner.path).canonical_settings
    assert fresh.state.managed_connection.founder_letter_seen_credential_ref == REFERENCE


@pytest.mark.asyncio
async def test_retained_patch_repository_can_restore_baseline_without_losing_managed_state(
    tmp_path: Path,
):
    owner, persistence = _owner(tmp_path / "settings.json")
    persistence.release.set()
    baseline = owner.require_canonical()
    repository = owner.create_canonical_patch_repository(
        base_settings=baseline, committed_settings=baseline
    )
    managed, _, _ = _launch(owner, "managed")
    await managed
    for locale in ("ja", "en", "ko", "en"):
        result = await repository.save(
            SettingsCommitRequest(
                values={"intent.ui.locale": locale},
                expected_revision=None,
                reason="settings.ui_prompt_clipboard_state",
            )
        )
        assert result.succeeded
        assert repository.committed_settings.intent.ui.locale == locale
        _assert_state(owner, locale=locale, reference=REFERENCE)


@pytest.mark.asyncio
@pytest.mark.parametrize("first_kind", ["managed", "ordinary"])
async def test_overlapping_physical_commits_preserve_both_changes(tmp_path: Path, first_kind: str):
    owner, persistence = _owner(tmp_path / "settings.json")
    loop_thread = threading.get_ident()
    first, _, _ = _launch(owner, first_kind)
    await _entered(persistence)
    second, admitted, _ = _launch(owner, "ordinary" if first_kind == "managed" else "managed")
    try:
        await admitted.wait()
        await asyncio.sleep(0)
        assert not second.done()
        assert persistence.attempts == 1
    finally:
        persistence.release.set()
    outcomes = await asyncio.wait_for(asyncio.gather(first, second), 5)
    assert all(result is None or result.succeeded for result in outcomes)
    assert len(persistence.threads) == 2
    assert loop_thread not in persistence.threads
    _assert_state(owner, locale="ja", reference=REFERENCE)
    await _progress(owner)


@pytest.mark.asyncio
@pytest.mark.parametrize("first_kind", ["managed", "ordinary"])
@pytest.mark.parametrize("fail_at", [1, 2])
async def test_failed_overlapping_commit_preserves_other_commit_and_retry(
    tmp_path: Path, first_kind: str, fail_at: int
):
    owner, persistence = _owner(tmp_path / "settings.json")
    persistence.fail_at = fail_at
    first, _, _ = _launch(owner, first_kind)
    await _entered(persistence)
    second_kind = "ordinary" if first_kind == "managed" else "managed"
    second, admitted, _ = _launch(owner, second_kind)
    await admitted.wait()
    persistence.release.set()
    outcomes = await asyncio.wait_for(asyncio.gather(first, second, return_exceptions=True), 5)
    failed_kind = first_kind if fail_at == 1 else second_kind
    failed = outcomes[fail_at - 1]
    successful = outcomes[2 - fail_at]
    assert successful is None or successful.succeeded
    if failed_kind == "managed":
        assert isinstance(failed, OSError)
    else:
        assert not failed.succeeded
        assert failed.diagnostics.code == "settings_save_failed"
    _assert_state(
        owner,
        locale="en" if failed_kind == "ordinary" else "ja",
        reference=None if failed_kind == "managed" else REFERENCE,
    )
    persistence.fail_at = None
    retried, _, _ = _launch(owner, failed_kind)
    result = await asyncio.wait_for(retried, 5)
    assert result is None or result.succeeded
    _assert_state(owner, locale="ja", reference=REFERENCE)
    await _progress(owner)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_kind", ["managed", "ordinary"])
@pytest.mark.parametrize("cancel_queued", [False, True])
async def test_cancelled_canonical_commit_settles_before_caller_and_preserves_progress(
    tmp_path: Path, cancel_kind: str, cancel_queued: bool
):
    owner, persistence = _owner(tmp_path / "settings.json")
    first_kind = (
        ("ordinary" if cancel_kind == "managed" else "managed") if cancel_queued else cancel_kind
    )
    first, _, first_repository = _launch(owner, first_kind)
    await _entered(persistence)
    second, admitted, second_repository = _launch(
        owner, "ordinary" if first_kind == "managed" else "managed"
    )
    await admitted.wait()
    cancelled = second if cancel_queued else first
    remaining = first if cancel_queued else second
    repository = second_repository if cancel_queued else first_repository
    try:
        cancelled.cancel()
        await asyncio.sleep(0)
        cancelled.cancel()
        await asyncio.sleep(0)
        assert not cancelled.done()
        assert not persistence.settled.is_set()
    finally:
        persistence.release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(cancelled, 5)
    result = await asyncio.wait_for(remaining, 5)
    assert result is None or result.succeeded
    if cancel_kind == "ordinary":
        assert repository.commit_succeeded
    assert persistence.settled.is_set()
    assert persistence.attempts == 2
    _assert_state(owner, locale="ja", reference=REFERENCE)
    await _progress(owner)


async def _observer_scenario(path: Path, reentrant: bool) -> None:
    owner, persistence = _owner(path)
    observed: list[tuple[str, str | None]] = []
    observer_threads: list[int] = []
    reentered = False

    def observe(settings: AppSettingsVNext) -> None:
        nonlocal reentered
        observer_threads.append(threading.get_ident())
        observed.append(
            (
                settings.intent.ui.locale,
                settings.state.managed_connection.active_managed_credential_ref,
            )
        )
        if reentrant and not reentered and settings.intent.ui.locale == "ja":
            reentered = True
            owner.commit_delta(settings, _locale(settings, "ko"))
            raise OSError("synthetic observer notification failure")

    owner.observe_commits(observe)
    observed.clear()
    observer_threads.clear()
    first, _, _ = _launch(owner, "managed")
    await _entered(persistence)
    base = owner.require_canonical()
    if reentrant:
        second, admitted, repository = _launch(owner, "ordinary")
        await admitted.wait()
        persistence.release.set()
        await asyncio.wait_for(asyncio.gather(first, second), 5)
        assert repository.secondary_diagnostics
        assert repository.secondary_diagnostics[0].code == "settings_commit_notification_failed"
    else:
        start_sync = threading.Event()

        def release_from_thread() -> None:
            assert start_sync.wait(5)
            persistence.release.set()

        releaser = threading.Thread(target=release_from_thread)
        releaser.start()
        try:
            start_sync.set()
            owner.commit_delta(base, _locale(base, "ja"))
            await asyncio.wait_for(first, 5)
        finally:
            persistence.release.set()
            await asyncio.to_thread(releaser.join, 5)
        assert not releaser.is_alive()
    await asyncio.sleep(0)
    assert observed[:2] == [("en", REFERENCE), ("ja", REFERENCE)]
    if reentrant:
        assert observed[2] == ("ko", REFERENCE)
    assert observer_threads == [threading.get_ident()] * len(observed)
    _assert_state(owner, locale="ko" if reentrant else "ja", reference=REFERENCE)
    await _progress(owner, "ru")
    print(f"observer reentrant={reentrant}: ordered notifications, durable state, later progress")


def _observer_child(path: str, reentrant: bool) -> None:
    asyncio.run(_observer_scenario(Path(path), reentrant))


@pytest.mark.parametrize("reentrant", [False, True])
def test_observer_publication_does_not_hold_worker_commit_lock(tmp_path: Path, reentrant: bool):
    process = multiprocessing.get_context("spawn").Process(
        target=_observer_child, args=(str(tmp_path / "settings.json"), reentrant)
    )
    process.start()
    try:
        process.join(15)
        if process.is_alive():
            process.terminate()
            process.join(5)
            pytest.fail("canonical observer/worker transaction deadlocked")
        assert process.exitcode == 0
    finally:
        process.close()


class RuntimeEffects:
    def __init__(self, owner: SettingsOwner):
        self.fail_runtime = False
        self.restored: list[AppSettingsVNext] = []
        self.applied: list[AppSettingsVNext] = []
        self.calibrated: list[AppSettingsVNext] = []
        self.output = SimpleNamespace(activation_notice_enabled=False)
        self.adapter = object.__new__(SettingsRuntimeEffectsAdapter)
        self.adapter._settings = owner
        self.adapter._pipeline = SimpleNamespace(
            output_runtime=self.output, translation_runtime_configuration=None
        )
        self.adapter._calibration = SimpleNamespace(sync_from_settings=self.calibrated.append)
        self.adapter._sync_stt_signatures = self.restored.append

    async def preserve_before_replace(self, settings):
        return settings

    def capture_runtime_signatures(self):
        pass

    async def prepare(self, current, settings):
        return SettingsRuntimeTransition(
            settings,
            current,
            "en",
            False,
            None,
            None,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            (),
        )

    def activate_before_persist(self, transition):
        pass

    async def apply_after_persist(self, transition, **kwargs):
        self.applied.append(transition.settings)
        if self.fail_runtime:
            raise RuntimeError("synthetic runtime application failure")

    def restore_memory(self, settings):
        self.adapter.restore_memory(settings)


@pytest.mark.asyncio
async def test_runtime_failure_compensation_preserves_overlapping_managed_commit(tmp_path: Path):
    owner, persistence = _owner(tmp_path / "settings.json")
    persistence.release.set()
    base = owner.require_canonical()
    committed = owner.commit_delta(base, _locale(base, "ja"))
    effects = RuntimeEffects(owner)
    projection = SettingsProjectionOwner(
        presentation=SimpleNamespace(render_settings=lambda *args, **kwargs: True),
        config_path=owner.path,
        current_settings=lambda: owner.canonical,
    )
    projection.remember_all(committed)
    application = SettingsApplicationOwner(
        settings=owner,
        projection=projection,
        runtime_effects=effects,
        sync_ui=lambda: None,
        mutation_service_provider=lambda: None,
        consume_superseded_settings=lambda settings: False,
        active_local_asr_change=lambda *args: False,
    )
    effects.fail_runtime = True
    with pytest.raises(RuntimeError, match="synthetic runtime"):
        await application.apply_direct(committed, persist=False, strict_runtime_errors=True)
    persistence.attempts = 0
    persistence.started.clear()
    persistence.release.clear()
    managed, _, _ = _launch(owner, "managed")
    await _entered(persistence)
    compensation = asyncio.create_task(
        application.compensate_failed_local_asr_settings_apply(
            base_settings=base,
            committed_settings=committed,
        )
    )
    effects.fail_runtime = False
    persistence.release.set()
    await asyncio.wait_for(asyncio.gather(managed, compensation), 5)
    _assert_state(owner, locale="en", reference=REFERENCE)
    effects.fail_runtime = True
    failure = TransactionResult(status="settings_commit_failed", message=None, diagnostics=None)
    await application._resync_committed_runtime(
        base_settings=base,
        committed_settings=committed,
        failure_result=failure,
    )
    assert effects.restored
    assert effects.calibrated == effects.restored
    assert all(settings.intent.ui.locale == "en" for settings in effects.restored)
    assert all(
        settings.state.managed_connection.active_managed_credential_ref == REFERENCE
        for settings in effects.restored
    )
    assert (
        effects.output.activation_notice_enabled
        == owner.require_canonical().intent.osc.activation_notice_enabled
    )
    assert (
        application.results.current.secondary_diagnostics[0].code
        == "settings_runtime_resync_failed"
    )
    effects.adapter.restore_memory(committed)
    _assert_state(owner, locale="en", reference=REFERENCE)
    await _progress(owner)
