from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from puripuly_heart.app.adapters.settings_vnext_canonical_persistence import (
    SettingsVNextCanonicalPersistenceAdapter,
)
from puripuly_heart.app.services.managed.managed_operation import new_managed_operation_id
from puripuly_heart.app.wiring import wiring_llm_factory
from puripuly_heart.app.wiring.wiring_managed_auth_factory import (
    managed_identity_state_port_from_vnext,
)
from puripuly_heart.app.wiring.wiring_secrets_factory import create_secret_store
from puripuly_heart.config.settings_vnext.defaults import new_settings_for_reset
from puripuly_heart.core.managed_credential import (
    ManagedCredentialBinding,
    credential_binding_secret,
    managed_key_secret,
)
from puripuly_heart.core.managed_identity import ensure_managed_identity_bundle
from puripuly_heart.ui.app import TranslatorApp
from tests.app.test_active_byok_secret_application import (
    NEW_CREDENTIAL,
    ROUTES,
    _assert_credential,
    _assert_disabled,
    _assert_live_request,
    _presentation,
    _runtime,
)
from tests.app.test_active_byok_secret_application import (
    composed as composed,
)
from tests.ui.test_dashboard_view_branches import _make_dashboard


def _saved_managed_credential(app, store, source):
    owner = app.control().settings
    state = managed_identity_state_port_from_vnext(owner, owner.require_canonical())
    identity = ensure_managed_identity_bundle(state, store)
    reference = f"ph-mcred-retry-{source}"
    binding = ManagedCredentialBinding.create(
        source=source,
        installation_id=identity.installation_id,
        operation_id=new_managed_operation_id(),
        delivery_id=f"retry-delivery-{source}",
        managed_credential_ref=reference,
        credential=NEW_CREDENTIAL,
    ).encode()
    store.set(managed_key_secret(source), NEW_CREDENTIAL)
    store.set(credential_binding_secret(source), binding)
    state.active_managed_credential_ref = reference
    state.persist()
    return identity.installation_id, reference, binding


def _assert_retained(app, path, source, expected):
    installation, reference, binding = expected
    owner = app.control().settings
    fresh = SettingsVNextCanonicalPersistenceAdapter().load_active(path).canonical_settings
    for settings in (owner.require_canonical(), owner.require_committed(), fresh):
        state = settings.state.managed_connection
        assert state.installation_id == installation
        assert state.active_managed_credential_ref == reference
        assert state.pending_managed_operation_id is None
        assert state.pending_delivery_ack_source is None
    store = create_secret_store(fresh.intent.secrets, config_path=path)
    _assert_credential(store.get(managed_key_secret(source)), NEW_CREDENTIAL)
    assert store.get(credential_binding_secret(source)) == binding


def _native_callback_dashboard(app, monkeypatch):
    dashboard = _make_dashboard(monkeypatch)
    translator = TranslatorApp.__new__(TranslatorApp)
    translator._ui_application = app
    translator.view_dashboard = dashboard
    tasks = []
    dialogs = []

    def schedule(factory, *args):
        task = asyncio.create_task(factory(*args))
        tasks.append(task)
        return task

    translator._run_page_task = schedule
    translator.show_discord_managed_auth_dialog = lambda **kwargs: dialogs.append("discord")
    translator.show_qq_managed_auth_dialog = lambda: dialogs.append("qq")
    dashboard.on_toggle_translation = translator._on_translation_toggle
    presentation = _presentation(app)
    original_needs_key = presentation.set_dashboard_translation_needs_key
    original_enabled = presentation.set_dashboard_translation_enabled

    def needs_key(value):
        original_needs_key(value)
        dashboard.set_translation_needs_key(value)

    def enabled(value):
        original_enabled(value)
        dashboard.set_translation_enabled(value)

    monkeypatch.setattr(presentation, "set_dashboard_translation_needs_key", needs_key)
    monkeypatch.setattr(presentation, "set_dashboard_translation_enabled", enabled)
    dashboard.set_translation_needs_key(
        presentation.projections.get("translation_needs_key", False)
    )
    return dashboard, tasks, dialogs


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ("discord", "qq"))
async def test_one_native_enable_retries_failed_runtime_with_retained_managed_credential(
    composed, monkeypatch, source
):
    profile = new_settings_for_reset("zh-CN" if source == "qq" else "en").intent.translation
    route = replace(
        ROUTES[-1],
        model=profile.model,
        connection=profile.connection,
        secret_key=managed_key_secret(source),
    )
    async with composed(route, translation=profile) as (app, store, requests, path):
        expected = _saved_managed_credential(app, store, source)
        authority = app._managed.managed.auth_runtime.state()
        assert authority.managed_selected, app.compatibility_settings().intent.translation
        assert authority.local_key_available
        dashboard, tasks, dialogs = _native_callback_dashboard(app, monkeypatch)
        original_constructor = wiring_llm_factory.OpenRouterLLMProvider
        held = True
        constructed = []
        attempted = []

        def constructor(**kwargs):
            attempted.append(held)
            if held:
                raise RuntimeError("controlled provider construction unavailable")
            provider = original_constructor(**kwargs)
            constructed.append(provider)
            return provider

        monkeypatch.setattr(wiring_llm_factory, "OpenRouterLLMProvider", constructor)
        await app.apply_providers(force_rebuild_llm=True, persist_settings=False)
        assert attempted
        assert await app.set_translation_enabled(True) is False
        assert _runtime(app).provider is None
        await _assert_disabled(app)
        assert app.dashboard_managed_auth_action() == "continue"
        _assert_retained(app, path, source, expected)
        assert _presentation(app).projections["translation_needs_key"] is False
        assert dashboard.translation_needs_key is False
        held = False
        dashboard._toggle_translation()
        await asyncio.wait_for(asyncio.gather(*tasks), 5)
        status = await app.control().query("app.status", {})
        assert status["translation_enabled"] is True
        assert status["translation_runtime_ready"] is True
        assert constructed, app._managed.managed.translation.state_provider()
        assert dashboard.is_translation_on is True
        assert not dashboard._translation_showing_warning
        assert dialogs == []
        await _assert_live_request(app, route, NEW_CREDENTIAL, requests)
        _assert_retained(app, path, source, expected)
        assert all(host == route.host for host, _method, _path, _credential in requests)
        assert all(
            path.endswith("/key") or path.endswith("/chat/completions")
            for _host, _method, path, _credential in requests
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ("missing-key", "rejected-local-claim", "read-error"))
async def test_native_retry_retains_managed_credential_admission_guards(
    composed, monkeypatch, failure
):
    source = "discord"
    profile = new_settings_for_reset("en").intent.translation
    route = replace(
        ROUTES[-1],
        model=profile.model,
        connection=profile.connection,
        secret_key=managed_key_secret(source),
    )
    async with composed(route, translation=profile) as (app, store, requests, _path):
        _saved_managed_credential(app, store, source)
        dashboard, tasks, _dialogs = _native_callback_dashboard(app, monkeypatch)
        original_get = type(store).get
        unreadable = False
        attempted = asyncio.Event()
        original_constructor = wiring_llm_factory.OpenRouterLLMProvider
        held = True
        constructed = []

        def get_secret(self, key):
            if unreadable and key == managed_key_secret(source):
                raise OSError("controlled managed credential read failure")
            return original_get(self, key)

        def constructor(**kwargs):
            nonlocal unreadable
            if not held:
                provider = original_constructor(**kwargs)
                constructed.append(provider)
                return provider
            attempted.set()
            if failure == "missing-key":
                store.delete(managed_key_secret(source))
            elif failure == "rejected-local-claim":
                store.set(managed_key_secret("qq"), "synthetic-opposite-source-credential")
            else:
                unreadable = True
            raise RuntimeError("controlled provider construction unavailable")

        with monkeypatch.context() as boundary:
            boundary.setattr(type(store), "get", get_secret)
            boundary.setattr(wiring_llm_factory, "OpenRouterLLMProvider", constructor)
            await app.apply_providers(force_rebuild_llm=True, persist_settings=False)
            assert attempted.is_set()
            assert _runtime(app).provider is None
            assert dashboard.translation_needs_key is True
            held = False
            dashboard._toggle_translation()
            await asyncio.wait_for(asyncio.gather(*tasks), 5)
            await _assert_disabled(app)
            assert dashboard.is_translation_on is False
            assert _runtime(app).provider is None
            assert constructed == []
            assert not any(method == "POST" for _host, method, _path, _credential in requests)


@pytest.mark.asyncio
@pytest.mark.parametrize("verification", (None, False), ids=("unverified", "rejected"))
async def test_native_retry_preserves_selected_byok_verification_gate(
    composed, monkeypatch, verification
):
    route = ROUTES[-1]
    async with composed(route, initial=NEW_CREDENTIAL) as (app, _store, requests, _path):
        dashboard, tasks, dialogs = _native_callback_dashboard(app, monkeypatch)
        if verification is not None:
            app.persist_api_key_verification(route.provider, NEW_CREDENTIAL, verification)
        await app.apply_providers(force_rebuild_llm=True, persist_settings=False)
        assert dashboard.translation_needs_key is True
        request_count = len(requests)
        dashboard._toggle_translation()
        await asyncio.wait_for(asyncio.gather(*tasks), 5)
        await _assert_disabled(app)
        assert dashboard.is_translation_on is False
        assert dialogs == []
        assert not any(
            method == "POST" for _host, method, _path, _credential in requests[request_count:]
        )
