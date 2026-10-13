from __future__ import annotations

import errno
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest

from puripuly_heart.app.adapters.settings_vnext_canonical_persistence import (
    SettingsVNextCanonicalPersistenceAdapter,
)
from puripuly_heart.app.ports.settings_repository import SettingsCommitRequest
from puripuly_heart.app.services.application_shutdown import (
    ApplicationShutdownCoordinator,
    application_shutdown_callback,
)
from puripuly_heart.app.services.canonical_settings_persistence import (
    CanonicalSettingsPatchRepository,
    SettingsOwner,
)
from puripuly_heart.app.services.chatgpt_account import ChatGptAccountOwner
from puripuly_heart.app.services.manual_typing import ManualTypingOwner
from puripuly_heart.app.services.secret_settings_transaction import (
    SecretSetRequest,
    SecretSettingsTransaction,
)
from puripuly_heart.app.services.settings.settings_mutation import (
    SettingsMutationRequest,
    SettingsMutationService,
    SettingsMutationValidationResult,
)
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.core.chatgpt.oauth import ChatGptReauthRequired
from puripuly_heart.core.chatgpt.session import (
    CHATGPT_CLIENT_ID_SECRET,
    CHATGPT_REFRESH_TOKEN_SECRET,
    ChatGptSession,
)
from puripuly_heart.core.lifecycle import (
    SHUTDOWN_PHASE_CLOSE_PROVIDERS_OUTPUT_ADAPTERS,
    SHUTDOWN_PHASE_OWNER_DRAIN_CANCEL,
)
from puripuly_heart.core.orchestrator.self_translation_channel import TextSubmissionAdmission
from puripuly_heart.core.storage.secrets import InMemorySecretStore
from puripuly_heart.core.vrchat_scene_service import VrchatSceneService
from puripuly_heart.ui.app import TranslatorApp

FORBIDDEN = "synthetic_credential_private_response_personal_path"


class AllowValidation:
    async def validate(self, request):
        return SettingsMutationValidationResult(True, None, None)


@pytest.mark.asyncio
async def test_disk_replacement_cause_survives_all_typed_settings_boundaries(tmp_path, monkeypatch):
    persistence = SettingsVNextCanonicalPersistenceAdapter()
    settings = AppSettingsVNext()
    owner = SettingsOwner(tmp_path / "settings.json", persistence, canonical=settings)
    owner.persist()
    owner.remember_projection(settings)
    repository = CanonicalSettingsPatchRepository(
        owner, settings, base_settings=settings, surface="ui_prompt_clipboard_state"
    )
    original = Path.replace

    def deny(path, target):
        if Path(target) == owner.path:
            failure = PermissionError(errno.EACCES, FORBIDDEN)
            failure.winerror = 32
            raise failure
        return original(path, target)

    monkeypatch.setattr(Path, "replace", deny)
    result = await repository.save(SettingsCommitRequest({"intent.ui.locale": "ja"}, None, None))
    assert result.succeeded is False
    assert result.diagnostics is not None
    assert result.diagnostics.diagnostic_id == owner.last_failure.diagnostic_id
    assert result.diagnostics.fields["phase"] == "settings_replace"
    assert result.diagnostics.fields["exception_class"] == "PermissionError"
    assert result.diagnostics.fields["errno"] == errno.EACCES
    assert result.diagnostics.fields["winerror"] == 32
    assert owner.canonical.intent.ui.locale == settings.intent.ui.locale
    assert (
        persistence.load_active(owner.path).canonical_settings.intent.ui.locale
        == settings.intent.ui.locale
    )
    assert FORBIDDEN not in repr(result)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", [TypeError(FORBIDDEN), ValueError(FORBIDDEN), KeyError(FORBIDDEN)]
)
async def test_internal_runtime_defect_keeps_durable_intent_and_diagnosable_phase(
    tmp_path, failure
):
    persistence = SettingsVNextCanonicalPersistenceAdapter()
    settings = AppSettingsVNext()
    owner = SettingsOwner(tmp_path / "settings.json", persistence, canonical=settings)
    owner.persist()
    repository = CanonicalSettingsPatchRepository(owner, settings, base_settings=settings)

    class Runtime:
        async def apply_runtime(self, request):
            raise failure

    result = await SettingsMutationService(repository, Runtime(), AllowValidation()).mutate(
        SettingsMutationRequest({"intent.ui.locale": "ja"}, None, None, uuid4().hex)
    )
    assert result.status == "settings_commit_success_runtime_degraded"
    assert result.diagnostics.code == "runtime_apply_exception"
    assert result.diagnostics.fields["exception_class"] == type(failure).__name__
    assert owner.committed_settings.intent.ui.locale == "ja"
    assert persistence.load_active(owner.path).canonical_settings.intent.ui.locale == "ja"
    assert FORBIDDEN not in repr(result)
    owner.complete()


@pytest.mark.asyncio
async def test_secret_commit_primary_is_not_replaced_by_failed_restore():
    class Store:
        async def snapshot_secret(self, key):
            from puripuly_heart.app.ports.secret_store import SecretSnapshot

            return SecretSnapshot(key, "old", None, True)

        async def set_secret(self, key, value):
            from puripuly_heart.app.ports.secret_store import SecretWriteResult

            return SecretWriteResult(True, key, None, None, None)

        async def restore_secret(self, snapshot):
            raise PermissionError(errno.EACCES, FORBIDDEN)

    class Repository:
        async def save(self, request):
            raise OSError(errno.ENOSPC, FORBIDDEN)

    result = await SecretSettingsTransaction(Store(), Repository()).set_provider_secret(
        SecretSetRequest("openrouter_api_key", FORBIDDEN, {}, None, None, uuid4().hex)
    )
    assert result.status == "settings_commit_failed_secret_restore_failed"
    assert result.diagnostics.fields["errno"] == errno.ENOSPC
    assert result.secondary_diagnostics[0].fields["secret_restore_succeeded"] is False
    assert result.secondary_diagnostics[1].fields["errno"] == errno.EACCES
    assert all(
        cause.cause_id == result.diagnostics.diagnostic_id for cause in result.secondary_diagnostics
    )
    assert FORBIDDEN not in repr(result)


@pytest.mark.asyncio
async def test_unexpected_sign_in_failure_has_owned_phase_without_auth_url_or_credential():
    session = ChatGptSession(secret_store=InMemorySecretStore)

    def fail_listener(**kwargs):
        raise TypeError(FORBIDDEN)

    owner = ChatGptAccountOwner(session, listener_factory=fail_listener)
    result = await owner.connect(open_browser=False)
    assert result.succeeded is False
    assert result.diagnostics.operation == "loopback_bind"
    assert result.diagnostics.fields["exception_class"] == "TypeError"
    assert owner.in_progress is False
    assert FORBIDDEN not in repr(result)
    await owner.close()


@pytest.mark.asyncio
async def test_refresh_rejection_remains_primary_when_local_delete_fails():
    class Store(InMemorySecretStore):
        def delete(self, key):
            raise PermissionError(errno.EACCES, FORBIDDEN)

    store = Store()
    store.set(CHATGPT_CLIENT_ID_SECRET, "synthetic_client")
    store.set(CHATGPT_REFRESH_TOKEN_SECRET, FORBIDDEN)
    session = ChatGptSession(
        secret_store=lambda: store,
        http_factory=lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    400, json={"error": "invalid_grant", "body": FORBIDDEN}
                )
            )
        ),
    )
    with pytest.raises(ChatGptReauthRequired) as caught:
        await session.access_token()
    primary = caught.value.diagnostics
    assert primary.status_code == 400
    assert primary.code == "chatgpt_invalid_grant"
    assert primary.category == "auth"
    assert primary.operation == "refresh_rejected"
    assert primary.fields["local_credentials_removed"] is False
    secondary = caught.value.secondary_diagnostics[0]
    assert secondary.cause_id == primary.diagnostic_id
    assert secondary.operation == "local_credential_delete"
    assert FORBIDDEN not in repr(primary) + repr(secondary)
    assert store.get(CHATGPT_REFRESH_TOKEN_SECRET) == FORBIDDEN
    await session.close()


@pytest.mark.asyncio
async def test_scene_enumeration_failure_is_unknown_not_confirmed_empty(tmp_path):
    class Snapshots:
        failure = True

        def snapshots(self):
            if self.failure:
                raise PermissionError(errno.EACCES, FORBIDDEN)
            return ()

    snapshots = Snapshots()
    scene = VrchatSceneService(snapshots=snapshots, log_directory=tmp_path)
    scene._started = True
    await scene._poll_once(scene.generation)
    failed = scene.lifecycle_owner_snapshot()
    assert failed["process_observation_succeeded"] is False
    assert failed["process_present"] is None
    assert failed["observation_diagnostic_id"] is not None
    assert scene.snapshot().participant_count is None
    assert FORBIDDEN not in repr(failed)
    snapshots.failure = False
    await scene._poll_once(scene.generation)
    empty = scene.lifecycle_owner_snapshot()
    assert empty["process_observation_succeeded"] is True
    assert empty["process_present"] is False
    assert empty["observation_diagnostic_id"] is None
    await scene.close()


@pytest.mark.asyncio
async def test_shutdown_continues_with_first_and_secondary_cause_identity():
    order = []

    async def first():
        order.append("first")
        raise OSError(errno.ENOSPC, FORBIDDEN)

    async def second():
        order.append("second")
        raise TypeError(FORBIDDEN)

    coordinator = ApplicationShutdownCoordinator(
        [
            application_shutdown_callback(
                phase=SHUTDOWN_PHASE_OWNER_DRAIN_CANCEL,
                owner_name="ApplicationRuntime",
                callback_name="close",
                callback=first,
            ),
            application_shutdown_callback(
                phase=SHUTDOWN_PHASE_CLOSE_PROVIDERS_OUTPUT_ADAPTERS,
                owner_name="ApplicationRuntime",
                callback_name="close",
                callback=second,
            ),
        ]
    )
    with pytest.raises(ExceptionGroup):
        await coordinator.shutdown()
    assert order == ["first", "second"]
    failures = coordinator.snapshot.failures
    assert failures[0].diagnostics.fields["errno"] == errno.ENOSPC
    assert failures[1].diagnostics.cause_id == failures[0].diagnostics.diagnostic_id
    assert all(failure.diagnostics.fields["cleanup_confirmed"] is False for failure in failures)
    assert FORBIDDEN not in repr(coordinator.snapshot)


@pytest.mark.asyncio
async def test_gui_queue_contains_failed_task_and_runs_next_without_acknowledging_failure(caplog):
    app = TranslatorApp.__new__(TranslatorApp)
    workers = []
    notifications = []
    acknowledgements = []
    app._ui_application = SimpleNamespace(diagnostic_instance_id=uuid4().hex)
    app._run_page_task = workers.append
    app._show_snackbar = lambda message, color: notifications.append(message)

    async def failed():
        raise TypeError(FORBIDDEN)

    async def next_task():
        acknowledgements.append("next")
        return True

    app._queue_settings_mutation_task(failed)
    app._queue_settings_mutation_task(next_task)
    assert len(workers) == 1
    await workers[0]()
    assert acknowledgements == ["next"]
    assert len(notifications) == 1
    assert app._settings_mutation_worker_active is False
    diagnostics = next(
        record._puripuly_heart_failure[0]
        for record in caplog.records
        if hasattr(record, "_puripuly_heart_failure")
    )
    assert diagnostics.code == "settings_background_task_failed"
    assert diagnostics.operation_id is not None
    assert diagnostics.fields["exception_class"] == "TypeError"
    assert FORBIDDEN not in repr(diagnostics)


@pytest.mark.asyncio
async def test_manual_typing_preserves_native_admission_and_parent_wait_identity():
    admission = TextSubmissionAdmission("admitted", uuid4(), (uuid4(),))
    waited = []
    owner = ManualTypingOwner(
        lambda: None, lambda turn_id: waited.append(turn_id), lambda message: None, 1.0, 1.0
    )

    async def submit():
        return admission

    assert await owner.submit(submit) is admission
    assert waited == [admission.turn_id]
    assert await owner.submit(None) is None


@pytest.mark.asyncio
async def test_account_retains_failed_listener_cleanup_and_closes_session():
    events = []

    class Listener:
        redirect_uri = "http://127.0.0.1:1/callback"

        def wait(self, timeout):
            raise TypeError(FORBIDDEN)

        def close(self):
            events.append("listener")
            if events.count("listener") == 1:
                raise PermissionError(errno.EACCES, FORBIDDEN)

    async def close_session():
        events.append("session")

    listener = Listener()
    owner = ChatGptAccountOwner(
        session=SimpleNamespace(
            client_id=lambda: None, host_id=lambda: "synthetic-host", close=close_session
        ),
        listener_factory=lambda **kwargs: listener,
    )
    result = await owner.connect(open_browser=False)
    assert not result.succeeded
    assert result.diagnostics.category == "unknown"
    assert result.diagnostics.fields["exception_class"] == "TypeError"
    assert result.secondary_diagnostics[0].cause_id == result.diagnostics.diagnostic_id
    assert result.secondary_diagnostics[0].fields["errno"] == errno.EACCES
    assert id(listener) in owner._pending_listeners
    assert FORBIDDEN not in repr(result)
    await owner.close()
    assert events == ["listener", "listener", "session"]
    assert owner._pending_listeners == {}


@pytest.mark.asyncio
async def test_existing_account_flow_respects_no_browser_and_preserves_notification_failure():
    def fail_browser(url):
        raise TypeError(FORBIDDEN)

    owner = ChatGptAccountOwner(session=SimpleNamespace(), browser_opener=fail_browser)
    owner._listener = SimpleNamespace()
    owner._authorization_url = "https://example.invalid/synthetic"
    result = await owner.connect(open_browser=False)
    assert result.failure_code == "in_progress"
    assert result.secondary_diagnostics == ()
    result = await owner.connect(open_browser=True)
    assert result.failure_code == "in_progress"
    assert owner.in_progress
    assert result.secondary_diagnostics[0].code == "chatgpt_browser_open_failed"
    assert FORBIDDEN not in repr(result)


@pytest.mark.asyncio
async def test_settings_resync_and_memory_restore_preserve_save_primary(monkeypatch):
    from puripuly_heart.app.services.settings.settings_application import SettingsApplicationOwner
    from puripuly_heart.app.services.settings.settings_transaction_result import (
        SettingsTransactionResultOwner,
    )
    from puripuly_heart.config.settings_vnext.compat import (
        SettingsPersistenceStatus,
        safe_persistence_error,
    )
    from puripuly_heart.core.messages import TransactionResult

    primary = safe_persistence_error(
        SettingsPersistenceStatus.SAVE_FAILED, PermissionError(errno.EACCES, FORBIDDEN)
    ).diagnostics
    failure_result = TransactionResult("settings_commit_failed", None, primary)
    restores = []

    def restore(settings):
        restores.append(settings)
        if len(restores) == 2:
            raise PermissionError(errno.EACCES, FORBIDDEN)

    async def fail_apply(self, settings, **kwargs):
        raise TypeError(FORBIDDEN)

    monkeypatch.setattr(SettingsApplicationOwner, "apply_direct", fail_apply)
    owner = SettingsApplicationOwner.__new__(SettingsApplicationOwner)
    owner.runtime_effects = SimpleNamespace(restore_memory=restore)
    owner.results = SettingsTransactionResultOwner()
    await owner._resync_committed_runtime(
        base_settings=AppSettingsVNext(),
        committed_settings=AppSettingsVNext(),
        failure_result=failure_result,
    )
    result = owner.results.current
    assert result.status == "settings_commit_failed"
    assert result.diagnostics is primary
    assert len(result.secondary_diagnostics) == 2
    resync, memory = result.secondary_diagnostics
    assert resync.cause_id == primary.diagnostic_id
    assert resync.fields["exception_class"] == "TypeError"
    assert memory.cause_id == resync.diagnostic_id
    assert memory.fields["errno"] == errno.EACCES
    assert FORBIDDEN not in repr(result)


@pytest.mark.asyncio
async def test_unexpected_refresh_type_error_is_unknown_not_auth_rejection():
    from puripuly_heart.core.chatgpt.oauth import ChatGptAuthError

    store = InMemorySecretStore()
    store.set(CHATGPT_CLIENT_ID_SECRET, "synthetic_client")
    store.set(CHATGPT_REFRESH_TOKEN_SECRET, FORBIDDEN)

    def fail(request):
        raise TypeError(FORBIDDEN)

    session = ChatGptSession(
        secret_store=lambda: store,
        http_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(fail)),
    )
    with pytest.raises(ChatGptAuthError) as captured:
        await session.access_token()
    diagnostic = captured.value.diagnostics
    assert diagnostic.code == "chatgpt_refresh_failed"
    assert diagnostic.category == "unknown"
    assert diagnostic.operation == "refresh_request"
    assert diagnostic.fields["exception_class"] == "TypeError"
    assert FORBIDDEN not in repr(diagnostic)
    assert store.get(CHATGPT_REFRESH_TOKEN_SECRET) == FORBIDDEN
    await session.close()
