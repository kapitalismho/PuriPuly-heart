from __future__ import annotations

import asyncio
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest
from puripuly_heart.core.openrouter_metadata import OpenRouterKeyMetadata

from puripuly_heart.app.ports.settings_repository import SettingsCommitRequest
from puripuly_heart.app.services.canonical_settings_persistence import compose_settings_owner
from puripuly_heart.app.services.index_selection import (
    IndexManagedAllowanceObservation,
    IndexSelectionOwner,
)
from puripuly_heart.app.wiring import wiring_index_selection as wiring
from puripuly_heart.config.settings_vnext.facade import load_vnext_settings
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.core.llm.index_translate import (
    IndexRequestLimitError,
    IndexServiceUnavailableError,
)
from puripuly_heart.core.messages import TransactionResult
from puripuly_heart.core.storage.secrets import InMemorySecretStore


def synthetic_store(values):
    store = InMemorySecretStore()
    for key, value in values.items():
        store.set(key, value)
    return store


def index_settings(settings=None):
    settings = settings or AppSettingsVNext()
    return replace(
        settings,
        intent=replace(
            settings.intent,
            translation=replace(
                settings.intent.translation,
                model="index_translate_35b",
                connection="public_api",
                openrouter_selected_source="none",
                openrouter_selection_alias=None,
            ),
        ),
    )


def make_owner(tmp_path, probe, *, allowance="missing", status="applied", before_commit=None):
    settings = compose_settings_owner(tmp_path / "settings.json")
    settings.canonical = index_settings()
    settings.remember_projection(settings.canonical)
    messages = []
    diagnostics = []
    commits = []
    owner = None

    async def observe():
        return IndexManagedAllowanceObservation(allowance)

    async def commit(observation, guard, _runtime_guard):
        if before_commit is not None:
            await before_commit()
        connection = "openrouter" if observation.state == "exhausted" else "managed"
        current = settings.canonical
        target = wiring.gemma_fallback_settings(current, connection)
        repository = settings.create_canonical_patch_repository(
            base_settings=current, committed_settings=target, commit_guard=guard
        )
        saved = await repository.save(SettingsCommitRequest({}, None, "index_fallback"))
        if not saved.succeeded:
            return TransactionResult("settings_commit_failed", None, saved.diagnostics)
        settings.complete()
        commits.append(connection)
        return TransactionResult(f"settings_commit_success_runtime_{status}", None, None)

    owner = IndexSelectionOwner(
        settings_provider=lambda: settings.canonical,
        probe=probe,
        observe_allowance=observe,
        commit_fallback=commit,
        message_sink=messages.append,
        diagnostic_sink=diagnostics.append,
    )
    settings.observe_commits(owner.selection_changed)
    return SimpleNamespace(
        owner=owner, settings=settings, messages=messages, diagnostics=diagnostics, commits=commits
    )


@pytest.mark.asyncio
async def test_start_selection_refresh_and_concurrent_trans_share_one_probe(tmp_path):
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = []

    async def probe():
        calls.append("probe")
        entered.set()
        await release.wait()

    harness = make_owner(tmp_path, probe)
    owner = harness.owner
    owner.start()
    await entered.wait()
    waiters = [asyncio.create_task(owner.prepare_enable()) for _ in range(3)]
    owner.start()
    unrelated = replace(
        harness.settings.canonical,
        intent=replace(
            harness.settings.canonical.intent,
            ui=replace(harness.settings.canonical.intent.ui, locale="ja"),
        ),
    )
    harness.settings.canonical = unrelated
    owner.selection_changed(unrelated)
    await asyncio.sleep(0)
    assert not any(task.done() for task in waiters)
    assert calls == ["probe"]
    release.set()
    assert await asyncio.gather(*waiters) == [True, True, True]
    assert await owner.prepare_enable() is True
    assert calls == ["probe"]
    assert harness.messages == harness.commits == []
    await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "allowance,connection",
    [
        ("missing", "managed"),
        ("remaining", "managed"),
        ("unknown", "managed"),
        ("exhausted", "openrouter"),
    ],
)
async def test_failure_commits_materialized_selection_and_toasts_once(
    tmp_path, allowance, connection
):
    async def probe():
        raise IndexServiceUnavailableError("transport")

    harness = make_owner(tmp_path, probe, allowance=allowance)
    harness.owner.start()
    assert await harness.owner.prepare_enable() is True
    assert await harness.owner.prepare_enable() is True
    assert harness.commits == [connection]
    assert harness.messages == [f"messages.index_api_fallback_{connection}"]
    persisted = load_vnext_settings(harness.settings.path).settings
    translation = persisted.intent.translation
    assert (translation.model, translation.connection) == ("gemma4_26b_31b", connection)
    assert translation.openrouter_selected_source == (
        "byok" if connection == "openrouter" else "managed"
    )
    assert (
        translation.openrouter_selection_alias
        == f"gemma4_26b_31b_{translation.openrouter_selected_source}"
    )
    assert translation.openrouter_provider_routing == "gemma4_26b_31b_latency"
    assert harness.owner.failure_diagnostics.fields["reason"] == "transport"
    await harness.owner.close()


@pytest.mark.asyncio
async def test_waiting_trans_joins_commit_and_preserves_current_unrelated_settings(tmp_path):
    committing = asyncio.Event()
    release = asyncio.Event()

    async def probe():
        raise IndexServiceUnavailableError("http_status", status_code=503)

    async def before_commit():
        committing.set()
        await release.wait()

    harness = make_owner(tmp_path, probe, before_commit=before_commit)
    harness.owner.start()
    waiter = asyncio.create_task(harness.owner.prepare_enable())
    await committing.wait()
    settings = harness.settings.canonical
    harness.settings.canonical = replace(
        settings,
        intent=replace(
            settings.intent,
            prompts=replace(settings.intent.prompts, system_prompt_override="synthetic override"),
            ui=replace(settings.intent.ui, locale="ko"),
        ),
    )
    await asyncio.sleep(0)
    assert not waiter.done()
    assert harness.messages == []
    release.set()
    assert await waiter is True
    persisted = load_vnext_settings(harness.settings.path).settings
    assert persisted.intent.prompts.system_prompt_override == "synthetic override"
    assert persisted.intent.ui.locale == "ko"
    await harness.owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("shutdown", [False, True])
async def test_late_probe_cannot_mutate_after_retirement_or_shutdown(tmp_path, shutdown):
    entered = asyncio.Event()
    release = asyncio.Event()
    settled = asyncio.Event()

    async def probe():
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()
        settled.set()
        raise IndexServiceUnavailableError("transport")

    harness = make_owner(tmp_path, probe)
    harness.owner.start()
    await entered.wait()
    if shutdown:
        harness.owner.stop_ingress()
    else:
        harness.settings.canonical = wiring.gemma_fallback_settings(
            harness.settings.canonical, "managed"
        )
        harness.owner.selection_changed(harness.settings.canonical)
    release.set()
    await settled.wait()
    await harness.owner.close()
    assert harness.commits == harness.messages == []
    assert harness.owner.allowance_observation is None


@pytest.mark.asyncio
async def test_commit_fence_rejects_newer_selection(tmp_path):
    entered = asyncio.Event()
    release = asyncio.Event()

    async def probe():
        raise IndexServiceUnavailableError("transport")

    async def before_commit():
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()

    harness = make_owner(tmp_path, probe, before_commit=before_commit)
    harness.owner.start()
    await entered.wait()
    newer = wiring.gemma_fallback_settings(harness.settings.canonical, "openrouter")
    harness.settings.canonical = newer
    harness.owner.selection_changed(newer)
    release.set()
    await harness.owner.close()
    assert harness.settings.canonical == newer
    assert harness.commits == harness.messages == []
    assert not harness.settings.path.exists()


@pytest.mark.asyncio
async def test_failed_save_has_no_switched_toast_and_degraded_commit_is_not_ready(
    tmp_path, monkeypatch
):
    async def probe():
        raise IndexServiceUnavailableError("transport")

    harness = make_owner(tmp_path, probe)

    def fail_save(*_args):
        raise OSError("synthetic persistence failure")

    with monkeypatch.context() as patch:
        patch.setattr(type(harness.settings.persistence), "persist", fail_save)
        assert await harness.owner.prepare_enable() is False
    assert harness.messages == harness.commits == []
    assert harness.settings.canonical.intent.translation.model == "index_translate_35b"
    await harness.owner.close()

    degraded = make_owner(tmp_path, probe, status="degraded")
    assert await degraded.owner.prepare_enable() is False
    assert degraded.messages == ["messages.index_api_fallback_managed"]
    assert degraded.owner.transition_result.status == "settings_commit_success_runtime_degraded"
    await degraded.owner.close()


@pytest.mark.asyncio
async def test_reselection_creates_new_probe_and_only_settled_service_failure_transitions(tmp_path):
    probes = []

    async def probe():
        probes.append(1)

    harness = make_owner(tmp_path, probe)
    owner = harness.owner
    assert await owner.prepare_enable() is True
    owner.request_failed(asyncio.CancelledError())
    owner.request_failed(IndexRequestLimitError("output_limit"))
    await asyncio.sleep(0)
    assert harness.commits == []
    away = wiring.gemma_fallback_settings(harness.settings.canonical, "managed")
    owner.selection_changed(away)
    harness.settings.canonical = index_settings(away)
    owner.selection_changed(harness.settings.canonical)
    assert await owner.prepare_enable() is True
    assert len(probes) == 2
    failure = IndexServiceUnavailableError("timeout")
    owner.request_failed(failure)
    owner.request_failed(failure)
    assert await owner.prepare_enable() is True
    assert harness.commits == ["managed"]
    assert len(probes) == 2
    await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "secrets,remaining,expected,lookup",
    [
        ({}, None, "missing", False),
        ({"openrouter_api_key": "personal"}, 0, "missing", False),
        ({"openrouter_managed_qq_api_key": "qq"}, 0, "unknown", False),
        ({"openrouter_managed_api_key": "standard"}, 0.1, "remaining", True),
        ({"openrouter_managed_api_key": "standard"}, 0.0007, "exhausted", True),
        ({"openrouter_managed_api_key": "standard"}, 0.00071, "remaining", True),
        ({"openrouter_managed_api_key": "standard"}, None, "unknown", True),
        ({"openrouter_managed_api_key": "standard"}, float("nan"), "unknown", True),
        ({"openrouter_managed_api_key": "standard"}, float("inf"), "unknown", True),
        (
            {"openrouter_managed_api_key": "standard", "openrouter_managed_qq_api_key": "qq"},
            0,
            "unknown",
            False,
        ),
    ],
)
async def test_allowance_observes_gemma_standard_credential_independent_of_index_ui(
    tmp_path, monkeypatch, secrets, remaining, expected, lookup
):
    store = synthetic_store(secrets)
    monkeypatch.setattr(wiring, "create_secret_store", lambda *_args, **_kwargs: store)
    settings = compose_settings_owner(tmp_path / "settings.json")
    settings.canonical = index_settings()
    calls = []

    async def metadata(key):
        calls.append(key)
        return OpenRouterKeyMetadata(1, remaining, 0)

    observer = wiring.IndexManagedAllowanceObserver(
        settings,
        settings.path,
        SimpleNamespace(fetch_openrouter_key_metadata=metadata),
        lambda: False,
    )
    observation = await observer.observe()
    assert observation.state == expected
    assert calls == (["standard"] if lookup else [])
    assert {key: store.get(key) for key in secrets} == secrets
    if expected == "unknown":
        assert observation.diagnostics.code == "index_managed_allowance_unknown"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["lookup", "key_changed", "identity_changed"])
async def test_allowance_unknown_retains_diagnostic_when_lookup_or_identity_not_current(
    tmp_path, monkeypatch, failure
):
    store = synthetic_store({"openrouter_managed_api_key": "standard"})
    monkeypatch.setattr(wiring, "create_secret_store", lambda *_args, **_kwargs: store)
    settings = compose_settings_owner(tmp_path / "settings.json")
    settings.canonical = index_settings()

    async def metadata(key):
        assert key == "standard"
        if failure == "lookup":
            raise OSError("synthetic metadata failure")
        if failure == "key_changed":
            store.set("openrouter_managed_api_key", "replacement")
        else:
            settings.canonical = replace(
                settings.canonical,
                state=replace(
                    settings.canonical.state,
                    managed_connection=replace(
                        settings.canonical.state.managed_connection,
                        active_managed_credential_ref="replacement",
                    ),
                ),
            )
        return OpenRouterKeyMetadata(1, 0, 1)

    observer = wiring.IndexManagedAllowanceObserver(
        settings,
        settings.path,
        SimpleNamespace(fetch_openrouter_key_metadata=metadata),
        lambda: False,
    )
    result = await observer.observe()
    assert result.state == "unknown"
    assert result.diagnostics.code == "index_managed_allowance_unknown"


@pytest.mark.asyncio
async def test_owned_blocking_credential_observation_does_not_block_event_loop(
    tmp_path, monkeypatch
):
    entered = threading.Event()
    release = threading.Event()

    class Store:
        def get(self, key):
            entered.set()
            assert release.wait(3)
            return None

    monkeypatch.setattr(wiring, "create_secret_store", lambda *_args, **_kwargs: Store())
    settings = compose_settings_owner(tmp_path / "settings.json")
    settings.canonical = index_settings()
    observer = wiring.IndexManagedAllowanceObserver(settings, settings.path, None, lambda: False)
    task = asyncio.create_task(observer.observe())
    async with asyncio.timeout(3):
        while not entered.is_set():
            await asyncio.sleep(0)
    progress = []
    for _ in range(5):
        await asyncio.sleep(0)
        progress.append(1)
    assert len(progress) == 5
    assert not task.done()
    release.set()
    assert (await task).state == "missing"


@pytest.mark.asyncio
async def test_exhaustion_evidence_revalidates_key_before_commit(tmp_path, monkeypatch):
    store = synthetic_store({"openrouter_managed_api_key": "standard"})
    monkeypatch.setattr(wiring, "create_secret_store", lambda *_args, **_kwargs: store)
    settings = compose_settings_owner(tmp_path / "settings.json")
    settings.canonical = index_settings()

    async def metadata(_key):
        return OpenRouterKeyMetadata(1, 0, 1)

    observer = wiring.IndexManagedAllowanceObserver(
        settings,
        settings.path,
        SimpleNamespace(fetch_openrouter_key_metadata=metadata),
        lambda: False,
    )
    observation = await observer.observe()
    assert observation.state == "exhausted"
    assert await observation.confirm_current() is True
    store.set("openrouter_managed_api_key", "replacement")
    assert await observation.confirm_current() is False


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_cancel_the_shared_probe(tmp_path):
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = []

    async def probe():
        calls.append(1)
        entered.set()
        await release.wait()

    harness = make_owner(tmp_path, probe)
    waiter = asyncio.create_task(harness.owner.prepare_enable())
    await entered.wait()
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    release.set()
    assert await harness.owner.prepare_enable() is True
    assert calls == [1]
    assert harness.commits == []
    await harness.owner.close()


@pytest.mark.asyncio
async def test_probe_and_metadata_deadlines_are_terminal_without_polling(tmp_path, monkeypatch):
    async def pending_probe():
        await asyncio.Event().wait()

    harness = make_owner(tmp_path, pending_probe)
    harness.owner.deadline_seconds = 0.01
    assert await harness.owner.prepare_enable() is True
    assert harness.commits == ["managed"]
    assert harness.messages == ["messages.index_api_fallback_managed"]
    await harness.owner.close()

    store = synthetic_store({"openrouter_managed_api_key": "standard"})
    monkeypatch.setattr(wiring, "create_secret_store", lambda *_args, **_kwargs: store)
    calls = []

    async def metadata(_key):
        calls.append(1)
        await asyncio.Event().wait()

    observer = wiring.IndexManagedAllowanceObserver(
        harness.settings,
        harness.settings.path,
        SimpleNamespace(fetch_openrouter_key_metadata=metadata),
        lambda: False,
        metadata_timeout_seconds=0.01,
    )
    observation = await observer.observe()
    assert observation.state == "unknown"
    assert observation.diagnostics.fields["reason"] == "metadata_unavailable"
    assert calls == [1]


@pytest.mark.asyncio
async def test_total_selection_deadline_bounds_a_pending_fallback_transition(tmp_path):
    async def probe():
        raise IndexServiceUnavailableError("transport")

    async def pending_commit():
        await asyncio.Event().wait()

    harness = make_owner(tmp_path, probe, before_commit=pending_commit)
    harness.owner.total_deadline_seconds = 0.01
    assert await harness.owner.prepare_enable() is False
    assert harness.owner.failure_diagnostics.code == "index_selection_deadline"
    assert harness.messages == harness.commits == []
    await harness.owner.close()


@pytest.mark.asyncio
async def test_successful_probe_does_not_enable_an_unapplied_index_runtime(tmp_path):
    calls = []
    applied = False

    async def probe():
        calls.append(1)

    harness = make_owner(tmp_path, probe)
    harness.owner.selected_runtime_applied = lambda: applied
    assert await harness.owner.prepare_enable() is False
    assert harness.owner.availability == "available"
    applied = True
    assert await harness.owner.prepare_enable() is True
    assert calls == [1]
    await harness.owner.close()


@pytest.mark.asyncio
async def test_total_deadline_returns_while_owned_storage_drains_and_shutdown_joins_it(
    tmp_path, monkeypatch
):
    entered = threading.Event()
    release = threading.Event()

    class Store:
        def get(self, _key):
            entered.set()
            assert release.wait(5)
            return None

    async def probe():
        raise IndexServiceUnavailableError("transport")

    monkeypatch.setattr(wiring, "create_secret_store", lambda *_args, **_kwargs: Store())
    harness = make_owner(tmp_path, probe)
    observer = wiring.IndexManagedAllowanceObserver(
        harness.settings, harness.settings.path, None, lambda: False
    )
    harness.owner.observe_allowance = observer.observe
    harness.owner.total_deadline_seconds = 0.03
    waiting = asyncio.create_task(harness.owner.prepare_enable())
    try:
        async with asyncio.timeout(1):
            while not entered.is_set():
                await asyncio.sleep(0)
        heartbeat = []
        for _ in range(5):
            await asyncio.sleep(0)
            heartbeat.append(1)
        assert await asyncio.wait_for(waiting, 0.5) is False
        assert len(heartbeat) == 5
        assert release.is_set() is False
        assert harness.owner.failure_diagnostics.code == "index_selection_deadline"
        assert await harness.owner.prepare_enable() is False
        closing = asyncio.create_task(harness.owner.close())
        await asyncio.sleep(0)
        assert not closing.done()
    finally:
        release.set()
    await closing
    assert harness.commits == harness.messages == []
    assert harness.owner.allowance_observation is None
    assert harness.settings.canonical.intent.translation.model == "index_translate_35b"
