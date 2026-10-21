from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from puripuly_heart.app.adapters.settings_vnext_canonical_persistence import (
    SettingsVNextCanonicalPersistenceAdapter,
)
from puripuly_heart.app.ports.broker_client import (
    ManagedKeyDeliveryAckMetadata,
    ManagedKeyDeliveryAckResult,
    ManagedOperationAttemptSnapshot,
    ManagedOperationDeliverySnapshot,
    ManagedOperationStatusResult,
    QqManagedAssertionResult,
    QqManagedEntitlementSnapshot,
)
from puripuly_heart.app.services import qq_managed_auth
from puripuly_heart.app.services.canonical_settings_persistence import SettingsOwner
from puripuly_heart.app.services.managed import managed_connection_auth
from puripuly_heart.app.services.managed.managed_operation import (
    MANAGED_OPERATION_RESUME_TOKEN_SECRET,
    new_managed_operation_id,
    new_managed_operation_resume_token,
)
from puripuly_heart.app.wiring import wiring_managed_auth_factory as factory
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.config.settings_vnext.serialization import from_dict, to_dict
from puripuly_heart.core.managed_credential import (
    ManagedCredentialBinding,
    credential_binding_secret,
    managed_key_secret,
)
from puripuly_heart.core.managed_identity import ensure_managed_identity_bundle
from puripuly_heart.core.openrouter.managed_openrouter_broker_client import (
    HttpManagedOpenRouterBrokerClient,
)
from puripuly_heart.core.openrouter.managed_openrouter_release import (
    ManagedOpenRouterDiscordStartSuccess,
    ManagedOpenRouterFingerprintSalt,
    ManagedOpenRouterIssueSuccess,
    ManagedOpenRouterReleaseBehavior,
    ManagedOpenRouterReleaseService,
)
from puripuly_heart.core.storage.secrets import EncryptedFileSecretStore

FUTURE = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
PAST = (datetime.now(UTC) - timedelta(hours=1)).isoformat()


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    for name in ("HOME", "APPDATA", "LOCALAPPDATA", "USERPROFILE", "TEMP", "TMP"):
        directory = tmp_path / name
        directory.mkdir()
        monkeypatch.setenv(name, str(directory))
    return tmp_path


class Broker:
    def __init__(self, source, owner, store):
        self.source = source
        self.owner = owner
        self.store = store
        self.calls = []
        self.statuses = []
        self.resumes = []
        self.ack_observations = []
        self.delivery = "delivery-new"
        self.ref = "credential-new"
        self.key = "synthetic-new-key"
        self.status_seen = False
        self.ack_status = "acknowledged"

    def metadata(self):
        return ManagedKeyDeliveryAckMetadata(
            source=self.source,
            delivery_id=self.delivery,
            managed_credential_ref=self.ref,
            expires_at=FUTURE,
            delivery_ack_token="synthetic-ack",
        )

    def resumed_delivery(self):
        return ManagedOperationStatusResult(
            succeeded=True,
            operation_status="DELIVERY_PENDING",
            client_action="acknowledge_delivery",
            managed_secret_key=self.key,
            managed_credential_ref=self.ref,
            expires_at=FUTURE,
            delivery_ack=self.metadata(),
        )

    def pending_delivery(self):
        return ManagedOperationStatusResult(
            succeeded=True,
            operation_status="DELIVERY_PENDING",
            client_action="acknowledge_delivery",
            attempt=ManagedOperationAttemptSnapshot(managed_credential_ref=self.ref),
            delivery=ManagedOperationDeliverySnapshot(self.delivery, "pending", FUTURE),
        )

    async def start_discord_oauth(self, **kwargs):
        self.calls.append(("oauth", kwargs))
        return ManagedOpenRouterDiscordStartSuccess(
            authorization_url="https://discord.example/authorize?state=current-state",
            redirect_uri=kwargs["redirect_uri"],
            oauth_session_expires_at=FUTURE,
            issue_nonce="synthetic-nonce",
            fingerprint_salt=ManagedOpenRouterFingerprintSalt(1, "salt"),
            fingerprint_salt_version=1,
        )

    async def issue_discord_managed_key(self, request):
        self.calls.append(("issue", request))
        return ManagedOpenRouterIssueSuccess(
            openrouter_api_key=self.key,
            managed_credential_ref=self.ref,
            expires_at=FUTURE,
            delivery_ack_required=True,
            delivery_id=self.delivery,
            delivery_ack_token="synthetic-ack",
            delivery_ack_expires_at=FUTURE,
        )

    async def assert_qq_managed_identity(self, request):
        self.calls.append(("assert", request))
        return QqManagedAssertionResult(
            succeeded=True,
            managed_secret_key=self.key,
            entitlement=QqManagedEntitlementSnapshot("synthetic-subject", self.ref, FUTURE),
            failure_subcode=None,
            retry_after_ms=None,
            message=None,
            diagnostics=None,
            delivery_ack=self.metadata(),
        )

    async def get_managed_operation_status(self, request):
        self.calls.append(("status", request))
        self.status_seen = True
        return self.statuses.pop(0)

    async def resume_managed_operation(self, request):
        self.calls.append(("resume", request))
        return self.resumes.pop(0)

    async def acknowledge_managed_key_delivery(self, request):
        self.calls.append(("ack", request))
        fresh = EncryptedFileSecretStore(self.store.path, passphrase="synthetic-passphrase")
        binding = ManagedCredentialBinding.decode(fresh.get(credential_binding_secret(self.source)))
        current = self.owner.require_canonical().state.managed_connection
        disk = (
            SettingsVNextCanonicalPersistenceAdapter()
            .load_active(self.owner.path)
            .canonical_settings
        )
        exact = binding is not None and binding.matches(
            credential=fresh.get(managed_key_secret(self.source)),
            source=self.source,
            installation_id=current.installation_id,
            delivery_id=request.delivery_id,
            managed_credential_ref=request.managed_credential_ref,
        )
        self.ack_observations.append((exact, disk.state.managed_connection == current))
        assert exact
        return ManagedKeyDeliveryAckResult(
            succeeded=self.ack_status == "acknowledged", status=self.ack_status
        )


class StoreBoundary:
    def __init__(self, store):
        self.store = store
        self.fail_read = None
        self.fail_write = None
        self.after_status = None

    def get(self, key):
        if key == self.fail_read and (self.after_status is None or self.after_status.status_seen):
            self.fail_read = None
            raise OSError("synthetic transient storage observation")
        return self.store.get(key)

    def set(self, key, value):
        if key == self.fail_write:
            self.fail_write = None
            raise OSError("synthetic interrupted persistence")
        self.store.set(key, value)

    def delete(self, key):
        self.store.delete(key)


async def callback(listener, authorization_url, expires_at):
    return "synthetic-code", "current-state"


def component(root, source, *, existing=False):
    settings = AppSettingsVNext()
    settings = replace(
        settings,
        intent=replace(
            settings.intent,
            translation=replace(
                settings.intent.translation,
                connection="managed_china" if source == "qq" else "managed",
                openrouter_selected_source="managed",
            ),
        ),
    )
    persistence = SettingsVNextCanonicalPersistenceAdapter()
    if existing:
        settings = persistence.load_active(root / "settings.json").canonical_settings
    owner = SettingsOwner(
        path=root / "settings.json", persistence=persistence, canonical=settings, authoritative=True
    )
    if not existing:
        owner.persist()
    store = EncryptedFileSecretStore(root / "secrets.json", passphrase="synthetic-passphrase")
    if not existing:
        store.set("unrelated", "synthetic-unrelated")
    boundary = StoreBoundary(store)
    broker = Broker(source, owner, store)
    release = ManagedOpenRouterReleaseService(
        openrouter_config=factory.build_openrouter_release_runtime_config_from_vnext(settings),
        managed_state=factory.managed_identity_state_port_from_vnext(owner, settings),
        secrets=boundary,
        client=broker,
        app_version="synthetic",
        hardware_hash_provider=lambda: "hardware",
        discord_oauth_callback_runner=callback,
    )
    adapter = factory.ManagedAuthRuntimeAdapter(
        config_path=owner.path,
        secret_store_factory=lambda *args, **kwargs: boundary,
        settings=owner,
        release_service_provider=lambda: release,
        settings_repository_factory=lambda base, committed, surface: owner.create_canonical_patch_repository(
            base_settings=base, committed_settings=committed, surface=surface
        ),
        runtime_presence_provider=lambda: (True, True),
        ingress_provider=lambda: False,
    )
    return owner, store, boundary, broker, release, adapter


async def authorize(adapter, source):
    if source == "qq":
        return await adapter.execute_qq("synthetic-account-B", "b" * 64)
    return await adapter.execute_discord(None)


def pending(owner, store, source, *, ack=None, key=None, bound=False, empty_installation=False):
    state = factory.managed_identity_state_port_from_vnext(owner, owner.require_canonical())
    identity = ensure_managed_identity_bundle(state, store)
    operation = new_managed_operation_id()
    state.pending_managed_operation_id = operation
    state.pending_managed_operation_source = source
    state.pending_managed_operation_installation_id = identity.installation_id
    state.pending_managed_operation_state = "DELIVERY_PENDING"
    store.set(MANAGED_OPERATION_RESUME_TOKEN_SECRET, new_managed_operation_resume_token())
    if key is not None:
        store.set(managed_key_secret(source), key)
    if ack is not None:
        state.pending_delivery_ack_source = source
        state.pending_delivery_ack_delivery_id = ack.delivery_id
        state.pending_delivery_ack_managed_credential_ref = ack.managed_credential_ref
        state.pending_delivery_ack_expires_at = ack.expires_at
        store.set(
            f"openrouter_managed_{'qq_' if source == 'qq' else ''}delivery_ack_token",
            ack.delivery_ack_token,
        )
        if bound:
            store.set(
                credential_binding_secret(source),
                ManagedCredentialBinding.create(
                    source=source,
                    installation_id=identity.installation_id,
                    operation_id=operation,
                    delivery_id=ack.delivery_id,
                    managed_credential_ref=ack.managed_credential_ref,
                    credential=key,
                ).encode(),
            )
    if empty_installation:
        state.installation_id = ""
    state.persist()
    return operation, identity.installation_id


def observe_fresh_process(owner, store, source, key, installation):
    script = """
import json, sys
from pathlib import Path
from puripuly_heart.app.adapters.settings_vnext_canonical_persistence import SettingsVNextCanonicalPersistenceAdapter
from puripuly_heart.core.storage.secrets import EncryptedFileSecretStore
from puripuly_heart.core.managed_credential import ManagedCredentialBinding, credential_binding_secret, managed_key_secret
p=json.load(sys.stdin)
s=SettingsVNextCanonicalPersistenceAdapter().load_active(Path(p['settings'])).canonical_settings.state.managed_connection
store=EncryptedFileSecretStore(Path(p['secrets']),passphrase='synthetic-passphrase')
b=ManagedCredentialBinding.decode(store.get(credential_binding_secret(p['source'])))
print(json.dumps({'same_installation':s.installation_id==p['installation'],'key_matches':store.get(managed_key_secret(p['source']))==p['key'],'binding_matches':b is not None and b.matches(credential=store.get(managed_key_secret(p['source'])),source=p['source'],installation_id=s.installation_id,managed_credential_ref=s.active_managed_credential_ref),'pending':bool(s.pending_managed_operation_id or s.pending_delivery_ack_source),'unrelated_preserved':store.get('unrelated')=='synthetic-unrelated'}))
"""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(path) for path in sys.path if path))
    result = subprocess.run(
        [sys.executable, "-c", script],
        input=json.dumps(
            {
                "settings": str(owner.path),
                "secrets": str(store.path),
                "source": source,
                "key": key,
                "installation": installation,
            }
        ),
        text=True,
        capture_output=True,
        env=env,
        timeout=30,
        check=True,
    )
    return json.loads(result.stdout)


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["discord", "qq"])
async def test_actual_factory_cold_success_is_bound_and_retained_consumer_is_current(
    isolated_home, source
):
    owner, store, boundary, broker, release, adapter = component(isolated_home, source)
    try:
        result = await authorize(adapter, source)
        assert result.succeeded
        assert result.runtime_rebuild == "always"
        assert broker.ack_observations == [(True, True)]
        installation = owner.require_canonical().state.managed_connection.installation_id
        assert installation
        ready = await release.prepare_for_translation(allow_authorization=False)
        assert ready.behavior == ManagedOpenRouterReleaseBehavior.READY
        assert ready.api_key == broker.key
        assert observe_fresh_process(owner, store, source, broker.key, installation) == {
            "same_installation": True,
            "key_matches": True,
            "binding_matches": True,
            "pending": False,
            "unrelated_preserved": True,
        }
        assert store.get("openrouter_managed_qq_status_auth") is None
        assert store.get("openrouter_managed_qq_operation_auth") is None
    finally:
        await release.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["discord", "qq"])
@pytest.mark.parametrize(
    "bad_state", ["old_key", "missing_key", "missing_token", "expired", "empty_installation"]
)
async def test_impossible_or_old_unbound_ack_reconciles_same_operation(
    isolated_home, source, bad_state
):
    owner, store, boundary, broker, release, adapter = component(isolated_home, source)
    metadata = broker.metadata()
    if bad_state == "expired":
        metadata = replace(metadata, expires_at=PAST)
    operation, installation = pending(
        owner,
        store,
        source,
        ack=metadata,
        key=None if bad_state == "missing_key" else "old-unrelated-key",
        bound=bad_state in {"missing_token", "expired"},
        empty_installation=bad_state == "empty_installation",
    )
    if bad_state == "missing_token":
        store.delete(f"openrouter_managed_{'qq_' if source == 'qq' else ''}delivery_ack_token")
    broker.statuses = [ManagedOperationStatusResult(True, "RETRY_READY", "retry_authorized")]
    broker.resumes = [broker.resumed_delivery()]
    try:
        result = await authorize(adapter, source)
        assert result.succeeded
        assert [name for name, _ in broker.calls] == ["status", "resume", "ack"]
        assert all(
            request.operation_id == operation
            for name, request in broker.calls
            if name in {"status", "resume"}
        )
        assert owner.require_canonical().state.managed_connection.installation_id == installation
        assert store.get(managed_key_secret(source)) == broker.key
        assert broker.ack_observations == [(True, True)]
    finally:
        await release.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["discord", "qq"])
async def test_exact_bound_ack_precedes_status_and_new_identity_input(isolated_home, source):
    owner, store, boundary, broker, release, adapter = component(isolated_home, source)
    pending(owner, store, source, ack=broker.metadata(), key=broker.key, bound=True)
    try:
        assert (await authorize(adapter, source)).succeeded
        assert [name for name, _ in broker.calls] == ["ack"]
        assert store.get("openrouter_managed_qq_status_auth") is None
    finally:
        await release.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["discord", "qq"])
@pytest.mark.parametrize("observation", ["resume", "active_key", "active_unbound"])
async def test_unknown_or_unbound_active_observation_preserves_authority(
    isolated_home, source, observation
):
    owner, store, boundary, broker, release, adapter = component(isolated_home, source)
    operation, _ = pending(owner, store, source, ack=broker.metadata(), key=broker.key, bound=True)
    state = factory.managed_identity_state_port_from_vnext(owner, owner.require_canonical())
    state.pending_delivery_ack_source = None
    state.pending_delivery_ack_delivery_id = None
    state.pending_delivery_ack_managed_credential_ref = None
    state.pending_delivery_ack_expires_at = None
    state.persist()
    if observation == "resume":
        boundary.fail_read = MANAGED_OPERATION_RESUME_TOKEN_SECRET
    else:
        broker.statuses = [
            ManagedOperationStatusResult(True, "ACTIVE", "none", managed_credential_ref=broker.ref)
        ]
        if observation == "active_key":
            boundary.fail_read = managed_key_secret(source)
            boundary.after_status = broker
        else:
            store.set(managed_key_secret(source), "unrelated-old-key")
    try:
        result = await authorize(adapter, source)
        assert not result.succeeded
        assert (
            owner.require_canonical().state.managed_connection.pending_managed_operation_id
            == operation
        )
        assert store.get(MANAGED_OPERATION_RESUME_TOKEN_SECRET)
        assert not any(name in {"assert", "issue", "ack", "resume"} for name, _ in broker.calls)
        if observation != "active_unbound":
            broker.status_seen = False
            broker.statuses = [
                ManagedOperationStatusResult(
                    True, "ACTIVE", "none", managed_credential_ref=broker.ref
                )
            ]
            assert (await authorize(adapter, source)).succeeded
            assert (
                owner.require_canonical().state.managed_connection.active_managed_credential_ref
                == broker.ref
            )
    finally:
        await release.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["discord", "qq"])
@pytest.mark.parametrize("failed_slot", ["key", "binding", "ack_token"])
async def test_interrupted_delivery_never_acks_old_key_and_fresh_factory_recovers(
    isolated_home, source, failed_slot
):
    owner, store, boundary, broker, release, adapter = component(isolated_home, source)
    pending(owner, store, source, key="old-key")
    operation = owner.require_canonical().state.managed_connection.pending_managed_operation_id
    broker.statuses = [ManagedOperationStatusResult(True, "RETRY_READY", "retry_authorized")]
    broker.resumes = [broker.resumed_delivery()]
    boundary.fail_write = {
        "key": managed_key_secret(source),
        "binding": credential_binding_secret(source),
        "ack_token": f"openrouter_managed_{'qq_' if source == 'qq' else ''}delivery_ack_token",
    }[failed_slot]
    try:
        assert not (await authorize(adapter, source)).succeeded
        assert not broker.ack_observations
        disk = SettingsVNextCanonicalPersistenceAdapter().load_active(owner.path).canonical_settings
        assert disk.state.managed_connection.pending_managed_operation_id == operation
        owner.canonical = disk
        broker.statuses = [ManagedOperationStatusResult(True, "RETRY_READY", "retry_authorized")]
        broker.resumes = [broker.resumed_delivery()]
        assert (await authorize(adapter, source)).succeeded
        assert broker.ack_observations == [(True, True)]
    finally:
        await release.close()


@pytest.mark.asyncio
async def test_live_release_preserves_mixed_source_authority_without_qq_key(isolated_home):
    owner, store, boundary, broker, release, adapter = component(isolated_home, "discord")
    operation, _ = pending(owner, store, "discord", key="existing-discord-key")
    state = factory.managed_identity_state_port_from_vnext(owner, owner.require_canonical())
    state.pending_delivery_ack_source = "qq"
    state.pending_delivery_ack_delivery_id = "qq-delivery"
    state.pending_delivery_ack_managed_credential_ref = "qq-ref"
    state.pending_delivery_ack_expires_at = FUTURE
    state.persist()
    store.set("openrouter_managed_qq_delivery_ack_token", "qq-token")
    try:
        result = await release.prepare_for_translation(allow_authorization=False)
        assert result.behavior != ManagedOpenRouterReleaseBehavior.READY
        assert broker.calls == []
        assert (
            owner.require_canonical().state.managed_connection.pending_managed_operation_id
            == operation
        )
        assert store.get(MANAGED_OPERATION_RESUME_TOKEN_SECRET)
        assert store.get("openrouter_managed_qq_delivery_ack_token") == "qq-token"
    finally:
        await release.close()


def test_canonical_stale_adapters_commit_only_their_changes(isolated_home):
    owner, store, boundary, broker, release, adapter = component(isolated_home, "qq")
    first = factory.managed_identity_state_port_from_vnext(owner, owner.require_canonical())
    second = factory.managed_identity_state_port_from_vnext(owner, owner.require_canonical())
    old = first.snapshot()
    second.active_managed_credential_ref = "current-ref"
    second.persist()
    first.founder_letter_seen_credential_ref = "seen-ref"
    first.restore(old)
    first.persist()
    assert first.active_managed_credential_ref == "current-ref"
    assert (
        owner.require_canonical().state.managed_connection.active_managed_credential_ref
        == "current-ref"
    )
    assert (
        SettingsVNextCanonicalPersistenceAdapter()
        .load_active(owner.path)
        .canonical_settings.state.managed_connection.active_managed_credential_ref
        == "current-ref"
    )


def test_retirement_prunes_known_extensions_without_changing_authority():
    raw = to_dict(AppSettingsVNext())
    raw["state"]["managed_connection"].update(
        referral_id="OLD123",
        referral_source="qq",
        installation_id="preserved",
        local_managed_claim_sources=["qq"],
        future_option="keep",
    )
    raw["managed_identity"] = {"referral_id": "OLD123", "future_legacy": "keep"}
    encoded = to_dict(from_dict(raw))
    managed = encoded["state"]["managed_connection"]
    assert "referral_id" not in managed and "referral_source" not in managed
    assert managed["installation_id"] == "preserved" and managed["local_managed_claim_sources"] == (
        "qq",
    )
    assert managed["future_option"] == "keep"
    assert encoded["managed_identity"] == {"future_legacy": "keep"}


@pytest.mark.asyncio
async def test_cancelled_identity_work_settles_before_task_completion(isolated_home, monkeypatch):
    owner, store, boundary, broker, release, adapter = component(isolated_home, "qq")
    entered, finish = threading.Event(), threading.Event()
    real_ensure = factory.ensure_managed_identity_bundle

    def gated(state, secrets):
        entered.set()
        assert finish.wait(10)
        return real_ensure(state, secrets)

    monkeypatch.setattr(factory, "ensure_managed_identity_bundle", gated)
    identity = factory.ManagedIdentityPreflightAdapter(release.managed_state, boundary)
    task = asyncio.create_task(identity.ensure_bundle())
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        saved = (
            SettingsVNextCanonicalPersistenceAdapter().load_active(owner.path).canonical_settings
        )
        assert saved.state.managed_connection.installation_id
        assert store.get("managed_device_private_key")
    finally:
        finish.set()
        await asyncio.gather(task, return_exceptions=True)
        await release.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["discord", "qq"])
async def test_actual_http_fallback_terminal_is_observed_once_and_clears_only_owned_state(
    isolated_home, source
):
    owner, store, boundary, broker, release, adapter = component(isolated_home, source)
    pending(owner, store, source, ack=broker.metadata(), key="unbound-old-key")
    calls = []

    async def handler(request):
        calls.append(request.url.path)
        return httpx.Response(410, json={"code": "authorization_expired"})

    client = HttpManagedOpenRouterBrokerClient(
        base_url="https://broker.invalid", transport=httpx.MockTransport(handler)
    )
    release.client = client
    try:
        result = await authorize(adapter, source)
        assert not result.succeeded
        assert len(calls) == 1
        assert result.transaction_result.diagnostics.status_code == 410
        state = owner.require_canonical().state.managed_connection
        assert state.pending_managed_operation_id is None
        assert state.pending_delivery_ack_source is None
        assert store.get(managed_key_secret(source)) == "unbound-old-key"
        assert store.get("unrelated") == "synthetic-unrelated"
    finally:
        await release.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["discord", "qq"])
async def test_retained_core_consumer_recovers_exact_same_operation(isolated_home, source):
    owner, store, boundary, broker, release, adapter = component(isolated_home, source)
    operation, installation = pending(
        owner, store, source, ack=broker.metadata(), key="old-unbound-key"
    )
    broker.statuses = [ManagedOperationStatusResult(True, "RETRY_READY", "retry_authorized")]
    broker.resumes = [broker.resumed_delivery()]
    try:
        ready = await release.prepare_for_translation(allow_authorization=False)
        assert ready.behavior == ManagedOpenRouterReleaseBehavior.READY
        assert ready.api_key == broker.key
        assert broker.ack_observations == [(True, True)]
        assert [name for name, _ in broker.calls] == ["status", "resume", "ack"]
        assert all(
            request.operation_id == operation
            for name, request in broker.calls
            if name in {"status", "resume"}
        )
        assert owner.require_canonical().state.managed_connection.installation_id == installation
    finally:
        await release.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["discord", "qq"])
async def test_mismatched_resumed_contract_cannot_replace_existing_key(isolated_home, source):
    owner, store, boundary, broker, release, adapter = component(isolated_home, source)
    operation, _ = pending(owner, store, source, key="preserved-key")
    broker.statuses = [ManagedOperationStatusResult(True, "RETRY_READY", "retry_authorized")]
    delivery = broker.resumed_delivery()
    broker.resumes = [
        replace(
            delivery,
            delivery_ack=replace(
                delivery.delivery_ack, source="qq" if source == "discord" else "discord"
            ),
        )
    ]
    try:
        assert not (await authorize(adapter, source)).succeeded
        assert store.get(managed_key_secret(source)) == "preserved-key"
        assert (
            owner.require_canonical().state.managed_connection.pending_managed_operation_id
            == operation
        )
        assert not broker.ack_observations
    finally:
        await release.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["discord", "qq"])
async def test_interrupted_storage_recovers_in_fresh_process(isolated_home, source):
    owner, store, boundary, broker, release, adapter = component(isolated_home, source)
    operation, installation = pending(owner, store, source, key="preserved-old-key")
    broker.statuses = [ManagedOperationStatusResult(True, "RETRY_READY", "retry_authorized")]
    broker.resumes = [broker.resumed_delivery()]
    boundary.fail_write = credential_binding_secret(source)
    try:
        assert not (await authorize(adapter, source)).succeeded
        assert not broker.ack_observations
    finally:
        await release.close()
    script = """
import asyncio, json, os, sys
from pathlib import Path
from tests.app.test_managed_auth_reliability_258 import component, authorize, ManagedOperationStatusResult, factory
p=json.load(sys.stdin)
async def run():
    owner, store, boundary, broker, release, adapter=component(Path(p['root']),p['source'],existing=True)
    broker.statuses=[ManagedOperationStatusResult(True,'RETRY_READY','retry_authorized')]
    broker.resumes=[broker.resumed_delivery()]
    try:
        result=await authorize(adapter,p['source'])
        state=owner.require_canonical().state.managed_connection
        print(json.dumps({'succeeded':result.succeeded,'same_operation':all(request.operation_id==p['operation'] for name,request in broker.calls if name in {'status','resume'}),'same_installation':state.installation_id==p['installation'],'acks_exact':broker.ack_observations==[(True,True)],'pending':bool(state.pending_managed_operation_id or state.pending_delivery_ack_source),'unrelated_preserved':store.get('unrelated')=='synthetic-unrelated','intended_import':str(Path(factory.__file__).resolve())==p['factory'],'fresh_process':os.getpid()!=p['parent']}))
    finally:
        await release.close()
asyncio.run(run())
"""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(path) for path in sys.path if path))
    child = subprocess.run(
        [sys.executable, "-c", script],
        input=json.dumps(
            {
                "root": str(isolated_home),
                "source": source,
                "operation": operation,
                "installation": installation,
                "factory": str(Path(factory.__file__).resolve()),
                "parent": os.getpid(),
            }
        ),
        text=True,
        capture_output=True,
        env=env,
        timeout=30,
        check=True,
    )
    assert json.loads(child.stdout) == {
        "succeeded": True,
        "same_operation": True,
        "same_installation": True,
        "acks_exact": True,
        "pending": False,
        "unrelated_preserved": True,
        "intended_import": True,
        "fresh_process": True,
    }


@pytest.mark.asyncio
async def test_actual_legacy_non_ack_partial_write_rolls_back_and_retires_temporary_authority(
    isolated_home,
):
    owner, store, boundary, broker, release, adapter = component(isolated_home, "discord")
    state = release.managed_state
    ensure_managed_identity_bundle(state, store)
    state.release_token = "synthetic-legacy-release"
    state.release_token_expires_at = FUTURE
    state.verified_hardware_hash = "synthetic-hash"
    state.verified_hardware_hash_salt_version = 1
    state.persist()

    async def issue(request):
        return ManagedOpenRouterIssueSuccess(
            openrouter_api_key="legacy-new-key",
            managed_credential_ref="legacy-ref",
            expires_at=FUTURE,
        )

    broker.issue = issue
    original_set = boundary.set

    def set_then_fail(key, value):
        original_set(key, value)
        if key == managed_key_secret("discord"):
            raise OSError("synthetic post-mutation storage failure")

    boundary.set = set_then_fail
    try:
        result = await release.prepare_for_translation(allow_authorization=False)
        assert result.behavior == ManagedOpenRouterReleaseBehavior.STOP
        fresh = EncryptedFileSecretStore(store.path, passphrase="synthetic-passphrase")
        assert fresh.get(managed_key_secret("discord")) is None
        saved = (
            SettingsVNextCanonicalPersistenceAdapter().load_active(owner.path).canonical_settings
        )
        assert saved.state.managed_connection.release_token is None
        assert saved.state.managed_connection.active_managed_credential_ref is None
        assert fresh.get("unrelated") == "synthetic-unrelated"
    finally:
        await release.close()


@pytest.mark.asyncio
async def test_actual_legacy_prepare_and_ensure_join_one_owned_issue(isolated_home):
    owner, store, boundary, broker, release, adapter = component(isolated_home, "discord")
    state = release.managed_state
    ensure_managed_identity_bundle(state, store)
    state.release_token = "synthetic-legacy-release"
    state.release_token_expires_at = FUTURE
    state.verified_hardware_hash = "synthetic-hash"
    state.verified_hardware_hash_salt_version = 1
    state.persist()
    entered, finish = asyncio.Event(), asyncio.Event()
    count = 0

    async def issue(request):
        nonlocal count
        count += 1
        entered.set()
        await finish.wait()
        return ManagedOpenRouterIssueSuccess(
            openrouter_api_key="legacy-key", managed_credential_ref="legacy-ref", expires_at=FUTURE
        )

    broker.issue = issue
    prepare = asyncio.create_task(release.prepare_for_translation())
    try:
        await asyncio.wait_for(entered.wait(), 5)
        ensure = asyncio.create_task(release.ensure_key_for_llm_start())
        await asyncio.sleep(0)
        finish.set()
        results = await asyncio.gather(prepare, ensure)
        assert count == 1
        assert all(result.behavior == ManagedOpenRouterReleaseBehavior.READY for result in results)
        assert all(result.api_key == "legacy-key" for result in results)
        assert results[1].single_flight_reused
        assert (
            EncryptedFileSecretStore(store.path, passphrase="synthetic-passphrase").get(
                managed_key_secret("discord")
            )
            == "legacy-key"
        )
    finally:
        finish.set()
        await asyncio.gather(prepare, return_exceptions=True)
        await release.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["discord", "qq"])
async def test_modern_missing_delivery_contract_cannot_overwrite_or_activate(isolated_home, source):
    owner, store, boundary, broker, release, adapter = component(isolated_home, source)
    ensure_managed_identity_bundle(release.managed_state, store)
    store.set(managed_key_secret(source), "preserved-existing-key")
    if source == "qq":
        original = broker.assert_qq_managed_identity

        async def missing_delivery(request):
            return replace(await original(request), delivery_ack=None)

        broker.assert_qq_managed_identity = missing_delivery
    else:
        original = broker.issue_discord_managed_key

        async def missing_delivery(request):
            return replace(await original(request), delivery_ack_required=False)

        broker.issue_discord_managed_key = missing_delivery
    try:
        result = await authorize(adapter, source)
        assert not result.succeeded
        assert result.transaction_result.diagnostics.code == "managed_delivery_contract_missing"
        assert store.get(managed_key_secret(source)) == "preserved-existing-key"
        assert owner.require_canonical().state.managed_connection.pending_managed_operation_id
        assert not broker.ack_observations
    finally:
        await release.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["discord", "qq"])
async def test_factory_recovery_preserves_safe_http_cause_and_route(isolated_home, source):
    from puripuly_heart.core.messages import ErrorDiagnostics

    owner, store, boundary, broker, release, adapter = component(isolated_home, source)
    operation, _ = pending(owner, store, source, key="preserved-key")
    cause = ErrorDiagnostics(
        component="broker",
        operation="managed_operation_status",
        code="access_rejected",
        category="auth",
        visibility="basic",
        content_policy="metadata_only",
        status_code=403,
        retry_after_ms=7000,
        fields={
            "proxy_source": "env",
            "tls_source": "explicit_file",
            "tls_backend": "openssl",
            "stage": "response_body",
            "dispatch_certainty": "response_received",
            "configuration_error": "unknown",
        },
    )
    broker.statuses = [
        ManagedOperationStatusResult(
            succeeded=False,
            operation_status="CREATE_UNKNOWN",
            client_action="action_required",
            diagnostics=cause,
        )
    ]
    try:
        assert not adapter.state().local_key_available
        result = await authorize(adapter, source)
        assert not result.succeeded
        diagnostic = result.transaction_result.diagnostics
        assert diagnostic.status_code == 403 and diagnostic.retry_after_ms == 7000
        assert cause.diagnostic_id in {diagnostic.diagnostic_id, diagnostic.cause_id}
        assert diagnostic.fields["proxy_source"] == "env"
        assert diagnostic.fields["tls_source"] == "explicit_file"
        assert diagnostic.fields["stage"] == "response_body"
        assert (
            owner.require_canonical().state.managed_connection.pending_managed_operation_id
            == operation
        )
        assert [name for name, _ in broker.calls] == ["status"]
    finally:
        await release.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("consumer", ["factory", "core"])
async def test_concrete_oauth_expiry_requires_fresh_authorization_without_same_code_reissue(
    isolated_home, consumer
):
    from puripuly_heart.core.openrouter.managed_openrouter_release import (
        ManagedOpenRouterReleaseError,
    )

    owner, store, boundary, broker, release, adapter = component(isolated_home, "discord")
    ensure_managed_identity_bundle(release.managed_state, store)
    private_key = store.get("managed_device_private_key")
    error = ManagedOpenRouterReleaseError(
        code="trial_unavailable",
        error_class="retryable",
        subcode="discord_oauth_session_expired",
        operation="discord_issue",
        http_status=409,
        message="synthetic expired authorization",
        retry_after_ms=0,
    )

    async def expired_issue(request):
        broker.calls.append(("issue", request))
        raise error

    broker.issue_discord_managed_key = expired_issue
    broker.statuses = [ManagedOperationStatusResult(False, "UNKNOWN_OPERATION", "wait")]
    try:
        if consumer == "factory":
            result = await adapter.execute_discord(None)
            assert not result.succeeded
            assert result.message_key == "discord_auth.error.expired"
            assert [name for name, _ in broker.calls] == ["oauth", "issue", "status"]
            diagnostic = result.transaction_result.diagnostics
        else:
            result = await release.prepare_for_translation()
            assert result.behavior == ManagedOpenRouterReleaseBehavior.RESTART
            assert result.message_key == "discord_auth.error.expired"
            assert [name for name, _ in broker.calls] == ["oauth", "issue"]
            diagnostic = result.diagnostics.failure_diagnostics
        assert diagnostic.status_code == 409
        assert (
            owner.require_canonical().state.managed_connection.pending_managed_operation_id is None
        )
        assert store.get("managed_device_private_key") == private_key
        assert store.get(managed_key_secret("discord")) is None
        assert store.get("unrelated") == "synthetic-unrelated"
        assert not broker.ack_observations
    finally:
        await release.close()


def assert_wire_ack_is_durable(owner, store, source, operation, body):
    fresh = EncryptedFileSecretStore(store.path, passphrase="synthetic-passphrase")
    current = owner.require_canonical().state.managed_connection
    saved = SettingsVNextCanonicalPersistenceAdapter().load_active(owner.path).canonical_settings
    binding = ManagedCredentialBinding.decode(fresh.get(credential_binding_secret(source)))
    assert binding is not None
    assert binding.matches(
        credential=fresh.get(managed_key_secret(source)),
        source=source,
        installation_id=current.installation_id,
        operation_id=operation,
        delivery_id=body["delivery_id"],
        managed_credential_ref=body["managed_credential_ref"],
    )
    assert saved.state.managed_connection == current
    assert current.pending_managed_operation_id == operation
    assert body["delivery_ack_token"] == fresh.get(
        f"openrouter_managed_{'qq_' if source == 'qq' else ''}delivery_ack_token"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["discord", "qq"])
@pytest.mark.parametrize("consumer", ["factory", "core"])
@pytest.mark.parametrize(
    ("rejection", "http_status"), [("invalid", 404), ("expired", 410), ("mismatched", 409)]
)
async def test_actual_http_ack_rejection_recovers_same_operation_to_durable_success(
    isolated_home, source, consumer, rejection, http_status
):
    owner, store, boundary, broker, release, adapter = component(isolated_home, source)
    operation, installation = pending(
        owner, store, source, ack=broker.metadata(), key=broker.key, bound=True
    )
    resume_token = store.get(MANAGED_OPERATION_RESUME_TOKEN_SECRET)
    calls = []
    acked_deliveries = []
    new_key, new_ref, new_delivery = "synthetic-wire-key", "wire-ref", "wire-delivery"

    def handler(request):
        body = json.loads(request.content)
        calls.append(request.url.path)
        if request.url.path.endswith("/managed-key-delivery/ack"):
            assert_wire_ack_is_durable(owner, store, source, operation, body)
            acked_deliveries.append(body["delivery_id"])
            if body["delivery_id"] == broker.delivery:
                return httpx.Response(
                    http_status,
                    json={
                        "error": {
                            "code": "invalid_request",
                            "class": "terminal",
                            "subcode": f"delivery_ack_{rejection}",
                            "message": "synthetic stale delivery",
                        }
                    },
                )
            assert body["delivery_id"] == new_delivery
            assert body["managed_credential_ref"] == new_ref
            assert store.get(managed_key_secret(source)) == new_key
            return httpx.Response(200, json={"ok": True, "status": "acknowledged"})
        assert body == {
            "operation_id": operation,
            "installation_id": installation,
            "resume_token": resume_token,
        }
        if request.url.path.endswith("/managed-operation/status"):
            assert store.get(managed_key_secret(source)) == broker.key
            return httpx.Response(
                200,
                json={
                    "operation_id": operation,
                    "installation_id": installation,
                    "issue_source": source,
                    "state": "RETRY_READY",
                    "client_action": "retry_authorized",
                },
            )
        assert request.url.path.endswith("/managed-operation/resume")
        return httpx.Response(
            200,
            json={
                "operation_id": operation,
                "installation_id": installation,
                "issue_source": source,
                "openrouter_api_key": new_key,
                "managed_credential_ref": new_ref,
                "expires_at": FUTURE,
                "delivery_ack_required": True,
                "delivery_id": new_delivery,
                "delivery_ack_token": "synthetic-wire-ack",
                "delivery_ack_expires_at": FUTURE,
            },
        )

    release.client = HttpManagedOpenRouterBrokerClient(
        base_url="https://broker.invalid", transport=httpx.MockTransport(handler)
    )
    try:
        if consumer == "factory":
            result = await authorize(adapter, source)
            assert result.succeeded
            assert adapter.state().local_key_available
        else:
            result = await release.prepare_for_translation(allow_authorization=False)
            assert result.behavior == ManagedOpenRouterReleaseBehavior.READY
            assert result.api_key == new_key
        assert calls == [
            "/v1/providers/openrouter/managed-key-delivery/ack",
            "/v1/providers/openrouter/managed-operation/status",
            "/v1/providers/openrouter/managed-operation/resume",
            "/v1/providers/openrouter/managed-key-delivery/ack",
        ]
        assert acked_deliveries == [broker.delivery, new_delivery]
        state = owner.require_canonical().state.managed_connection
        assert state.installation_id == installation
        assert state.active_managed_credential_ref == new_ref
        assert state.pending_managed_operation_id is None
        assert state.pending_delivery_ack_source is None
        assert store.get(MANAGED_OPERATION_RESUME_TOKEN_SECRET) is None
        assert store.get("unrelated") == "synthetic-unrelated"
    finally:
        await release.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["discord", "qq"])
@pytest.mark.parametrize("consumer", ["factory", "core"])
async def test_actual_http_ack_retryable_failed_retains_exact_authority(
    isolated_home, source, consumer
):
    owner, store, boundary, broker, release, adapter = component(isolated_home, source)
    operation, installation = pending(
        owner, store, source, ack=broker.metadata(), key=broker.key, bound=True
    )
    initial = owner.require_canonical().state.managed_connection
    receipt = store.get(credential_binding_secret(source))
    resume_token = store.get(MANAGED_OPERATION_RESUME_TOKEN_SECRET)
    calls = []

    def handler(request):
        calls.append(request.url.path)
        assert request.url.path == "/v1/providers/openrouter/managed-key-delivery/ack"
        assert_wire_ack_is_durable(owner, store, source, operation, json.loads(request.content))
        return httpx.Response(
            409,
            json={
                "error": {
                    "code": "invalid_request",
                    "class": "retryable",
                    "subcode": "delivery_ack_failed",
                    "message": "synthetic-sensitive upstream detail",
                }
            },
        )

    release.client = HttpManagedOpenRouterBrokerClient(
        base_url="https://broker.invalid", transport=httpx.MockTransport(handler)
    )
    try:
        for _ in range(2):
            if consumer == "factory":
                result = await authorize(adapter, source)
                assert not result.succeeded
                diagnostic = result.transaction_result.diagnostics
                assert not adapter.state().local_key_available
            else:
                result = await release.prepare_for_translation(allow_authorization=False)
                assert result.behavior == ManagedOpenRouterReleaseBehavior.RETRY
                assert not result.local_key_available
                diagnostic = result.diagnostics.failure_diagnostics
            assert diagnostic is not None
            assert diagnostic.status_code == 409
            assert diagnostic.fields["broker_subcode"] == "delivery_ack_failed"
            assert "synthetic-sensitive" not in repr(diagnostic)
            state = owner.require_canonical().state.managed_connection
            assert state == initial
            assert state.installation_id == installation
            assert store.get(managed_key_secret(source)) == broker.key
            assert store.get(credential_binding_secret(source)) == receipt
            assert store.get(MANAGED_OPERATION_RESUME_TOKEN_SECRET) == resume_token
        assert len(calls) == 2
        assert store.get("unrelated") == "synthetic-unrelated"
    finally:
        await release.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("ack_mode", ["missing", "false", "full", "incomplete", "malformed"])
@pytest.mark.parametrize("concurrent_credential", [False, True])
async def test_actual_http_legacy_delivery_contract_is_rejected_before_replacement(
    isolated_home, ack_mode, concurrent_credential
):
    owner, store, boundary, broker, release, adapter = component(isolated_home, "discord")
    state = release.managed_state
    identity = ensure_managed_identity_bundle(state, store)
    state.release_token = "synthetic-legacy-release"
    state.release_token_expires_at = FUTURE
    state.verified_hardware_hash = "synthetic-hash"
    state.verified_hardware_hash_salt_version = 1
    state.persist()
    private_key = store.get("managed_device_private_key")
    old_binding = ManagedCredentialBinding.create(
        source="discord",
        installation_id=identity.installation_id,
        operation_id=new_managed_operation_id(),
        delivery_id="existing-delivery",
        managed_credential_ref="existing-ref",
        credential="existing-key",
    ).encode()
    calls = []

    def handler(request):
        calls.append(request.url.path)
        assert request.url.path == "/v1/providers/openrouter/issue"
        assert "operation_id" not in json.loads(request.content)
        if concurrent_credential:
            store.set(managed_key_secret("discord"), "existing-key")
            store.set(credential_binding_secret("discord"), old_binding)
            state.active_managed_credential_ref = "existing-ref"
            state.persist()
        payload = {
            "openrouter_api_key": "synthetic-sensitive-legacy-key",
            "managed_credential_ref": "legacy-new-ref",
            "expires_at": FUTURE,
        }
        if ack_mode != "missing":
            payload["delivery_ack_required"] = ack_mode != "false"
        if ack_mode == "full":
            payload.update(
                delivery_id="legacy-delivery",
                delivery_ack_token="synthetic-sensitive-legacy-ack",
                delivery_ack_expires_at=FUTURE,
            )
        elif ack_mode == "malformed":
            payload.update(delivery_id=7, delivery_ack_token=None)
        return httpx.Response(200, json=payload)

    release.client = HttpManagedOpenRouterBrokerClient(
        base_url="https://broker.invalid", transport=httpx.MockTransport(handler)
    )
    try:
        result = await release.prepare_for_translation(allow_authorization=False)
        saved = (
            SettingsVNextCanonicalPersistenceAdapter().load_active(owner.path).canonical_settings
        )
        fresh = EncryptedFileSecretStore(store.path, passphrase="synthetic-passphrase")
        current = saved.state.managed_connection
        assert current.installation_id == identity.installation_id
        assert current.release_token is None
        assert current.verified_hardware_hash is None
        assert current.pending_managed_operation_id is None
        assert current.pending_delivery_ack_source is None
        assert fresh.get("managed_device_private_key") == private_key
        assert fresh.get("unrelated") == "synthetic-unrelated"
        if ack_mode in {"missing", "false"}:
            assert result.behavior == ManagedOpenRouterReleaseBehavior.READY
            assert result.api_key == "synthetic-sensitive-legacy-key"
            assert fresh.get(managed_key_secret("discord")) == result.api_key
            assert current.active_managed_credential_ref == "legacy-new-ref"
        else:
            assert result.behavior == ManagedOpenRouterReleaseBehavior.STOP
            assert not result.local_key_available
            assert result.api_key is None
            assert result.message_key == "discord_auth.error.action_required"
            assert result.diagnostics.code == "managed_delivery_protocol_mismatch"
            assert result.diagnostics.failure_diagnostics.status_code == 200
            assert result.diagnostics.failure_diagnostics.fields["stage"] == "protocol_parse"
            assert "synthetic-sensitive" not in repr(result.diagnostics)
            assert fresh.get(managed_key_secret("discord")) == (
                "existing-key" if concurrent_credential else None
            )
            assert fresh.get(credential_binding_secret("discord")) == (
                old_binding if concurrent_credential else None
            )
            assert current.active_managed_credential_ref == (
                "existing-ref" if concurrent_credential else None
            )
            again = await release.prepare_for_translation(allow_authorization=False)
            assert again.api_key != "synthetic-sensitive-legacy-key"
        assert calls == ["/v1/providers/openrouter/issue"]
    finally:
        await release.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode", ["timeout", "typed_bind_failure", "bind_oserror", "hardware_oserror", "success"]
)
async def test_actual_oauth_adapter_scopes_callback_timeout_and_listener_failures(
    isolated_home, mode
):
    from dataclasses import asdict
    from types import SimpleNamespace

    from puripuly_heart.app.ports.discord_auth import DiscordAuthRequest
    from puripuly_heart.core.diagnostics import sanitize_diagnostics
    from puripuly_heart.core.discord.discord_managed_oauth import run_discord_oauth_callback_flow
    from puripuly_heart.core.discord.discord_oauth_loopback import (
        DiscordOAuthLoopbackListener,
        DiscordOAuthLoopbackUnavailableError,
    )
    from puripuly_heart.core.runtime.oauth import OAuthRuntime

    state = factory.ManagedIdentityStateAdapter(
        SimpleNamespace(**asdict(AppSettingsVNext().state.managed_connection)), lambda: None
    )
    store = EncryptedFileSecretStore(
        isolated_home / "callback-secrets.json", passphrase="synthetic-passphrase"
    )
    runtime = OAuthRuntime()
    listeners = []
    calls = []

    def create_listener():
        if mode == "typed_bind_failure":
            try:
                raise OSError(10048, "synthetic-sensitive bind detail")
            except OSError as exc:
                raise DiscordOAuthLoopbackUnavailableError("synthetic-sensitive listener") from exc
        if mode == "bind_oserror":
            raise OSError(10048, "synthetic-sensitive bind detail")
        listener = DiscordOAuthLoopbackListener.bind(0)
        listener.port = listener._server.server_port
        listeners.append(listener)
        return listener

    def start_response(request):
        calls.append(request.url.path)
        assert request.url.path == "/v1/auth/discord/start"
        return httpx.Response(
            200,
            json={
                "authorization_url": "https://discord.invalid/authorize?state=current-state",
                "redirect_uri": json.loads(request.content)["redirect_uri"],
                "oauth_session_expires_at": PAST if mode == "timeout" else FUTURE,
                "issue_nonce": "synthetic-sensitive-nonce",
                "fingerprint_salt": {"version": 1, "salt": "synthetic"},
                "fingerprint_salt_version": 1,
            },
        )

    async def callback_flow(listener, authorization_url, expires_at):
        if mode != "timeout":
            async with httpx.AsyncClient(trust_env=False) as local:
                response = await local.get(
                    listener.redirect_uri,
                    params={"code": "synthetic-sensitive-code", "state": "current-state"},
                )
                assert response.status_code == 200
        return await run_discord_oauth_callback_flow(
            listener, authorization_url, expires_at, open_browser=False
        )

    def hardware():
        if mode == "hardware_oserror":
            raise OSError(5, "synthetic-sensitive hardware detail")
        return "synthetic-hardware"

    client = HttpManagedOpenRouterBrokerClient(
        base_url="https://broker.invalid", transport=httpx.MockTransport(start_response)
    )
    auth = factory.DiscordOAuthAuthAdapter(
        identity=factory.ManagedIdentityPreflightAdapter(state, store),
        client=client,
        app_version="synthetic",
        raw_hardware_fingerprint_provider=None,
        hardware_hash_provider=hardware,
        oauth_runtime=runtime,
        listener_factory=create_listener,
        callback_runner=callback_flow,
    )
    try:
        result = await auth.start_discord_auth(DiscordAuthRequest(None, {}))
        if mode == "success":
            assert result.succeeded
            assert result.authorization_code == "synthetic-sensitive-code"
            assert result.oauth_state == "current-state"
            assert result.diagnostics is None
        else:
            assert not result.succeeded
            diagnostic = sanitize_diagnostics(result.diagnostics)
            assert diagnostic.component == "managed_auth"
            assert diagnostic.operation != "provider_build"
            assert diagnostic.code != "provider.unknown"
            assert "synthetic-sensitive" not in repr(diagnostic)
            if mode == "timeout":
                assert result.message.key == "discord_auth.error.retry"
                assert diagnostic.code == "discord_oauth_timeout"
                assert diagnostic.operation == "discord_callback"
                assert diagnostic.fields["stage"] == "callback_wait"
                assert diagnostic.fields["exception_class"] == "TimeoutError"
            elif mode in {"typed_bind_failure", "bind_oserror"}:
                assert result.message.key == "discord_auth.error.loopback_unavailable"
                assert diagnostic.code == "discord_loopback_unavailable"
                assert diagnostic.operation == "discord_start"
                assert diagnostic.fields["stage"] == "listener_start"
                assert diagnostic.fields["errno"] == 10048
                assert (
                    diagnostic.fields[
                        "cause_class_1" if mode == "typed_bind_failure" else "exception_class"
                    ]
                    == "OSError"
                )
            else:
                assert result.message.key == "discord_auth.error.retry"
                assert diagnostic.code == "discord_auth_exception"
                assert diagnostic.operation == "discord_auth"
                assert diagnostic.fields["exception_class"] == "OSError"
                assert diagnostic.fields["errno"] == 5
        assert not runtime.active_listener_names
        assert all(listener._closed and not listener._thread.is_alive() for listener in listeners)
        assert len(listeners) == (0 if mode in {"typed_bind_failure", "bind_oserror"} else 1)
        assert len(calls) == len(listeners)
    finally:
        await client.close()
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["discord", "qq"])
@pytest.mark.parametrize("issuance", ["initial", "resumed", "core"])
@pytest.mark.parametrize("consumer", ["factory", "core"])
async def test_physical_final_settings_failure_recovers_bound_delivery_after_reload(
    isolated_home, monkeypatch, source, issuance, consumer
):
    monkeypatch.setattr(managed_connection_auth, "DEFAULT_MAX_STATUS_POLLS", 1)
    monkeypatch.setattr(qq_managed_auth, "DEFAULT_MAX_STATUS_POLLS", 1)
    owner, store, boundary, broker, release, adapter = component(isolated_home, source)
    if issuance != "initial":
        pending(owner, store, source, key="preserved-old-key")
        broker.statuses = [ManagedOperationStatusResult(True, "RETRY_READY", "retry_authorized")]
        broker.resumes = [broker.resumed_delivery()]
    replace_file = Path.replace
    failures = []

    def fail_final_replace(path, target):
        if Path(target) == owner.path:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if raw["state"]["managed_connection"].get("pending_delivery_ack_source"):
                failures.append("settings_replace")
                raise PermissionError(13, "synthetic final settings replacement denied")
        return replace_file(path, target)

    try:
        with monkeypatch.context() as failure:
            failure.setattr(Path, "replace", fail_final_replace)
            if issuance == "core":
                result = await release.prepare_for_translation(allow_authorization=False)
                assert result.behavior == ManagedOpenRouterReleaseBehavior.RETRY
                assert result.diagnostics.code == "managed_delivery_store_failed"
            else:
                result = await authorize(adapter, source)
                assert not result.succeeded
                assert result.transaction_result.diagnostics is not None
        assert failures
        assert not broker.ack_observations
        saved = (
            SettingsVNextCanonicalPersistenceAdapter().load_active(owner.path).canonical_settings
        )
        before = saved.state.managed_connection
        operation, installation = (
            before.pending_managed_operation_id,
            before.pending_managed_operation_installation_id,
        )
        assert operation and installation == before.installation_id
        assert before.pending_delivery_ack_source is None
        assert before.active_managed_credential_ref is None
        assert before == owner.require_canonical().state.managed_connection
        assert release.managed_state.active_managed_credential_ref is None
        fresh = EncryptedFileSecretStore(store.path, passphrase="synthetic-passphrase")
        assert fresh.get(managed_key_secret(source)) == broker.key
        binding = ManagedCredentialBinding.decode(fresh.get(credential_binding_secret(source)))
        assert binding is not None and binding.matches(
            credential=broker.key,
            source=source,
            installation_id=installation,
            operation_id=operation,
            delivery_id=broker.delivery,
            managed_credential_ref=broker.ref,
        )
        ack_slot = f"openrouter_managed_{'qq_' if source == 'qq' else ''}delivery_ack_token"
        assert fresh.get(ack_slot) == "synthetic-ack"
    finally:
        await release.close()
    owner, store, boundary, broker, release, adapter = component(
        isolated_home, source, existing=True
    )
    broker.statuses = [broker.pending_delivery()]
    broker.resumes = [broker.pending_delivery()]
    try:
        if consumer == "factory":
            assert (await authorize(adapter, source)).succeeded
        else:
            ready = await release.prepare_for_translation(allow_authorization=False)
            assert ready.behavior == ManagedOpenRouterReleaseBehavior.READY
            assert ready.api_key == broker.key
        assert [name for name, _ in broker.calls] == ["status", "ack"]
        assert broker.calls[0][1].operation_id == operation
        assert broker.ack_observations == [(True, True)]
        assert observe_fresh_process(owner, store, source, broker.key, installation) == {
            "same_installation": True,
            "key_matches": True,
            "binding_matches": True,
            "pending": False,
            "unrelated_preserved": True,
        }
    finally:
        await release.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["discord", "qq"])
@pytest.mark.parametrize("consumer", ["factory", "core"])
@pytest.mark.parametrize(
    "boundary",
    [
        "missing_key",
        "wrong_key",
        "missing_token",
        "missing_receipt",
        "wrong_source",
        "wrong_installation",
        "wrong_operation",
        "wrong_delivery",
        "wrong_ref",
        "unknown_key",
        "unknown_receipt",
        "unknown_token",
        "expired",
        "terminal",
        "recovery_write",
    ],
)
async def test_missing_ack_metadata_requires_exact_proof_and_broker_truth(
    isolated_home, monkeypatch, source, consumer, boundary
):
    monkeypatch.setattr(managed_connection_auth, "DEFAULT_MAX_STATUS_POLLS", 1)
    monkeypatch.setattr(qq_managed_auth, "DEFAULT_MAX_STATUS_POLLS", 1)
    monkeypatch.setattr(managed_connection_auth, "DEFAULT_STATUS_POLL_INTERVAL_MS", 0)
    monkeypatch.setattr(qq_managed_auth, "DEFAULT_STATUS_POLL_INTERVAL_MS", 0)
    owner, store, secrets, broker, release, adapter = component(isolated_home, source)
    operation, installation = pending(
        owner, store, source, ack=broker.metadata(), key=broker.key, bound=True
    )
    state = release.managed_state
    state.pending_delivery_ack_source = None
    state.pending_delivery_ack_delivery_id = None
    state.pending_delivery_ack_managed_credential_ref = None
    state.pending_delivery_ack_expires_at = None
    state.persist()
    ack_slot = f"openrouter_managed_{'qq_' if source == 'qq' else ''}delivery_ack_token"
    if boundary == "missing_key":
        store.delete(managed_key_secret(source))
    elif boundary == "wrong_key":
        store.set(managed_key_secret(source), "unrelated-old-key")
    elif boundary == "missing_token":
        store.delete(ack_slot)
    elif boundary == "missing_receipt":
        store.delete(credential_binding_secret(source))
    elif boundary.startswith("wrong_"):
        binding = ManagedCredentialBinding.decode(store.get(credential_binding_secret(source)))
        field = {
            "wrong_source": "source",
            "wrong_installation": "installation_id",
            "wrong_operation": "operation_id",
            "wrong_delivery": "delivery_id",
            "wrong_ref": "managed_credential_ref",
        }[boundary]
        value = ("qq" if source == "discord" else "discord") if field == "source" else "other"
        store.set(credential_binding_secret(source), replace(binding, **{field: value}).encode())
    elif boundary.startswith("unknown_"):
        secrets.fail_read = {
            "unknown_key": managed_key_secret(source),
            "unknown_receipt": credential_binding_secret(source),
            "unknown_token": ack_slot,
        }[boundary]
        secrets.after_status = broker
    probe = broker.pending_delivery()
    if boundary == "expired":
        probe = replace(probe, delivery=replace(probe.delivery, expires_at=PAST))
    elif boundary == "terminal":
        probe = ManagedOperationStatusResult(
            True, "FAILED", "action_required", failed_reason="authorization_expired"
        )
    broker.statuses = [probe, probe]
    broker.resumes = [probe, probe]
    replace_file = Path.replace

    def deny_recovery_write(path, target):
        if Path(target) == owner.path:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if raw["state"]["managed_connection"].get("pending_delivery_ack_source"):
                raise PermissionError(13, "synthetic recovery replacement denied")
        return replace_file(path, target)

    if boundary == "recovery_write":
        monkeypatch.setattr(Path, "replace", deny_recovery_write)
    try:
        if consumer == "factory":
            assert not (await authorize(adapter, source)).succeeded
        else:
            result = await release.prepare_for_translation(allow_authorization=False)
            assert result.behavior != ManagedOpenRouterReleaseBehavior.READY
        assert not broker.ack_observations
        assert all(name in {"status", "resume"} for name, _ in broker.calls)
        assert all(request.operation_id == operation for _, request in broker.calls)
        saved = (
            SettingsVNextCanonicalPersistenceAdapter().load_active(owner.path).canonical_settings
        )
        current = saved.state.managed_connection
        assert current == owner.require_canonical().state.managed_connection
        assert current.installation_id == installation
        assert current.active_managed_credential_ref is None
        assert current.pending_delivery_ack_source is None
        if boundary != "terminal":
            assert current.pending_managed_operation_id == operation
            assert store.get(MANAGED_OPERATION_RESUME_TOKEN_SECRET)
        else:
            assert current.pending_managed_operation_id is None
        assert store.get("unrelated") == "synthetic-unrelated"
    finally:
        await release.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["discord", "qq"])
@pytest.mark.parametrize("consumer", ["factory", "core"])
async def test_reconstructed_ack_metadata_commit_settles_before_cancellation(
    isolated_home, monkeypatch, source, consumer
):
    owner, store, boundary, broker, release, adapter = component(isolated_home, source)
    operation, installation = pending(
        owner, store, source, ack=broker.metadata(), key=broker.key, bound=True
    )
    state = release.managed_state
    state.pending_delivery_ack_source = None
    state.pending_delivery_ack_delivery_id = None
    state.pending_delivery_ack_managed_credential_ref = None
    state.pending_delivery_ack_expires_at = None
    state.persist()
    broker.statuses = [broker.pending_delivery()]
    entered, finish = threading.Event(), threading.Event()
    replace_file = Path.replace

    def hold_recovery_replace(path, target):
        if Path(target) == owner.path:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if raw["state"]["managed_connection"].get("pending_delivery_ack_source"):
                entered.set()
                assert finish.wait(5)
        return replace_file(path, target)

    monkeypatch.setattr(Path, "replace", hold_recovery_replace)
    task = asyncio.create_task(
        authorize(adapter, source)
        if consumer == "factory"
        else release.prepare_for_translation(allow_authorization=False)
    )
    closing = None
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        if consumer == "core":
            closing = asyncio.create_task(release.close())
        else:
            task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        if closing is not None:
            await closing
        assert not broker.ack_observations
        fresh = (
            SettingsVNextCanonicalPersistenceAdapter().load_active(owner.path).canonical_settings
        )
        assert fresh.state.managed_connection.pending_delivery_ack_delivery_id == broker.delivery
        assert fresh.state.managed_connection.pending_managed_operation_id == operation
        assert fresh.state.managed_connection.installation_id == installation
        assert fresh.state.managed_connection.active_managed_credential_ref is None
    finally:
        finish.set()
        await asyncio.gather(task, return_exceptions=True)
        if closing is not None:
            await asyncio.gather(closing, return_exceptions=True)
        await release.close()
    owner, store, boundary, broker, release, adapter = component(
        isolated_home, source, existing=True
    )
    try:
        assert (await authorize(adapter, source)).succeeded
        assert [name for name, _ in broker.calls] == ["ack"]
        assert broker.ack_observations == [(True, True)]
    finally:
        await release.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["discord", "qq"])
@pytest.mark.parametrize("consumer", ["factory", "core"])
@pytest.mark.parametrize("metadata", ["missing", "staged"])
async def test_terminal_delivery_never_promotes_retained_key_on_repeat_or_restart(
    isolated_home, source, consumer, metadata
):
    owner, store, boundary, broker, release, adapter = component(isolated_home, source)
    operation, installation = pending(
        owner, store, source, ack=broker.metadata(), key=broker.key, bound=True
    )
    state = release.managed_state
    if metadata == "missing":
        state.pending_delivery_ack_source = None
        state.pending_delivery_ack_delivery_id = None
        state.pending_delivery_ack_managed_credential_ref = None
        state.pending_delivery_ack_expires_at = None
    else:
        state.pending_delivery_ack_expires_at = PAST
        state.active_managed_credential_ref = broker.ref
        state.active_managed_expires_at = FUTURE
        state.founder_letter_seen_credential_ref = broker.ref
    state.persist()
    receipt = store.get(credential_binding_secret(source))
    broker.statuses = [
        ManagedOperationStatusResult(
            True, "FAILED", "action_required", failed_reason="authorization_expired"
        )
    ]
    try:
        if consumer == "factory":
            assert not (await authorize(adapter, source)).succeeded
        else:
            first = await release.prepare_for_translation(allow_authorization=False)
            assert first.behavior == ManagedOpenRouterReleaseBehavior.STOP
        for _ in range(2):
            repeated = await release.prepare_for_translation(allow_authorization=False)
            assert repeated.behavior != ManagedOpenRouterReleaseBehavior.READY
            assert not repeated.local_key_available
            assert not adapter.state().local_key_available
        assert [name for name, _ in broker.calls] == ["status"]
        assert broker.calls[0][1].operation_id == operation
        assert not broker.ack_observations
        saved = (
            SettingsVNextCanonicalPersistenceAdapter().load_active(owner.path).canonical_settings
        )
        current = saved.state.managed_connection
        assert current == owner.require_canonical().state.managed_connection
        assert current.installation_id == installation
        assert current.pending_managed_operation_id is None
        assert current.pending_delivery_ack_source is None
        assert current.active_managed_credential_ref is None
        assert current.active_managed_expires_at is None
        assert current.founder_letter_seen_credential_ref is None
        assert store.get(managed_key_secret(source)) == broker.key
        assert store.get(credential_binding_secret(source)) == receipt
        assert store.get("unrelated") == "synthetic-unrelated"
    finally:
        await release.close()
    script = """
import asyncio, json, sys
from pathlib import Path
from tests.app.test_managed_auth_reliability_258 import component, ManagedOpenRouterReleaseBehavior
async def run():
    p=json.load(sys.stdin)
    owner,store,boundary,broker,release,adapter=component(Path(p['root']),p['source'],existing=True)
    try:
        first=await release.prepare_for_translation(allow_authorization=False)
        second=await release.prepare_for_translation(allow_authorization=False)
        print(json.dumps({'first_ready':first.behavior==ManagedOpenRouterReleaseBehavior.READY,'second_ready':second.behavior==ManagedOpenRouterReleaseBehavior.READY,'local_key_available':adapter.state().local_key_available,'broker_calls':len(broker.calls)}))
    finally:
        await release.close()
asyncio.run(run())
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        input=json.dumps({"root": str(isolated_home), "source": source}),
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(str(path) for path in sys.path if path)),
        text=True,
        capture_output=True,
        timeout=30,
        check=True,
    )
    assert json.loads(result.stdout) == {
        "first_ready": False,
        "second_ready": False,
        "local_key_available": False,
        "broker_calls": 0,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["discord", "qq"])
@pytest.mark.parametrize("consumer", ["factory", "core"])
@pytest.mark.parametrize("credential_kind", ["legacy", "acknowledged"])
async def test_terminal_unrelated_operation_preserves_legitimate_existing_credential(
    isolated_home, source, consumer, credential_kind
):
    owner, store, boundary, broker, release, adapter = component(isolated_home, source)
    try:
        if credential_kind == "acknowledged":
            assert (await authorize(adapter, source)).succeeded
            assert broker.ack_observations == [(True, True)]
        operation, installation = pending(owner, store, source, key=broker.key)
        receipt = store.get(credential_binding_secret(source))
        broker.calls.clear()
        broker.ack_observations.clear()
        broker.statuses = [
            ManagedOperationStatusResult(
                True, "FAILED", "action_required", failed_reason="authorization_expired"
            )
        ]
        if consumer == "factory":
            assert not (await authorize(adapter, source)).succeeded
        else:
            first = await release.prepare_for_translation(allow_authorization=False)
            assert first.behavior == ManagedOpenRouterReleaseBehavior.STOP
        ready = await release.prepare_for_translation(allow_authorization=False)
        assert ready.behavior == ManagedOpenRouterReleaseBehavior.READY
        assert ready.api_key == broker.key
        assert adapter.state().local_key_available
        assert [name for name, _ in broker.calls] == ["status"]
        assert broker.calls[0][1].operation_id == operation
        assert not broker.ack_observations
        current = owner.require_canonical().state.managed_connection
        assert current.installation_id == installation
        assert current.pending_managed_operation_id is None
        assert current.active_managed_credential_ref == (
            broker.ref if credential_kind == "acknowledged" else None
        )
        assert store.get(managed_key_secret(source)) == broker.key
        assert store.get(credential_binding_secret(source)) == receipt
        assert store.get("unrelated") == "synthetic-unrelated"
    finally:
        await release.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["discord", "qq"])
@pytest.mark.parametrize("consumer", ["factory", "core"])
@pytest.mark.parametrize("observation", ["key", "receipt"])
async def test_terminal_projection_observation_failure_retains_recovery_authority(
    isolated_home, source, consumer, observation
):
    owner, store, boundary, broker, release, adapter = component(isolated_home, source)
    operation, installation = pending(
        owner, store, source, ack=broker.metadata(), key=broker.key, bound=True
    )
    state = release.managed_state
    state.pending_delivery_ack_source = None
    state.pending_delivery_ack_delivery_id = None
    state.pending_delivery_ack_managed_credential_ref = None
    state.pending_delivery_ack_expires_at = None
    state.active_managed_credential_ref = broker.ref
    state.persist()
    resume = store.get(MANAGED_OPERATION_RESUME_TOKEN_SECRET)
    receipt = store.get(credential_binding_secret(source))
    terminal = ManagedOperationStatusResult(
        True, "FAILED", "action_required", failed_reason="authorization_expired"
    )
    broker.statuses = [terminal, terminal]
    boundary.fail_read = (
        managed_key_secret(source) if observation == "key" else credential_binding_secret(source)
    )
    boundary.after_status = broker
    try:
        if consumer == "factory":
            assert not (await authorize(adapter, source)).succeeded
        else:
            result = await release.prepare_for_translation(allow_authorization=False)
            assert result.behavior == ManagedOpenRouterReleaseBehavior.STOP
        current = owner.require_canonical().state.managed_connection
        assert current.pending_managed_operation_id == operation
        assert current.installation_id == installation
        assert store.get(MANAGED_OPERATION_RESUME_TOKEN_SECRET) == resume
        assert not adapter.state().local_key_available
        repeated = await release.prepare_for_translation(allow_authorization=False)
        assert repeated.behavior == ManagedOpenRouterReleaseBehavior.STOP
        assert not adapter.state().local_key_available
        current = owner.require_canonical().state.managed_connection
        assert current.pending_managed_operation_id is None
        assert current.active_managed_credential_ref is None
        assert store.get(managed_key_secret(source)) == broker.key
        assert store.get(credential_binding_secret(source)) == receipt
        assert [name for name, _ in broker.calls] == ["status", "status"]
        assert not broker.ack_observations
        assert store.get("unrelated") == "synthetic-unrelated"
    finally:
        await release.close()
