from __future__ import annotations

import asyncio
import json
import logging
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
from puripuly_heart.app.services.settings_transaction_result import SettingsTransactionResultOwner

from puripuly_heart.app.services.capture.peer_capture_target_application import (
    PeerCaptureTargetUnavailable,
)
from puripuly_heart.app.services.overlay.overlay_application import OverlayApplicationOwner
from puripuly_heart.app.wiring import wiring_microphone_test
from puripuly_heart.composition import application_runtime
from puripuly_heart.composition.headless_application import (
    HeadlessApplicationPresentation,
    compose_headless_application,
)
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext, SecretsIntent
from puripuly_heart.config.settings_vnext.serialization import to_dict
from puripuly_heart.core.audio.format import AudioFrameF32
from puripuly_heart.core.audio.source import (
    MicrophoneTestRouteObservation,
    SelfMicCaptureChannelDecision,
    SoundDeviceInputMetadata,
)
from puripuly_heart.core.http_extensions import http_extension_secret_key
from puripuly_heart.core.messages import TransactionResult


def isolated_settings(path):
    settings = AppSettingsVNext()
    settings = replace(
        settings,
        intent=replace(
            settings.intent,
            secrets=SecretsIntent(backend="encrypted_file", encrypted_file_path="secrets.json"),
            osc=replace(settings.intent.osc, connection_mode="off"),
        ),
    )
    path.write_text(json.dumps(to_dict(settings)), encoding="utf-8")


@pytest.mark.asyncio
async def test_cli_model_only_selection_restores_saved_luna_connection(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("PURIPULY_HEART_SECRETS_PASSPHRASE", "isolated-test-passphrase")
    path = tmp_path / "settings.json"
    isolated_settings(path)
    app = compose_headless_application(path)
    try:
        await app.start()
        control = app.control()
        control.bind_instance("luna-history-control")
        for index, changes in enumerate(
            (
                {"translation.model": "gpt_6_luna"},
                {"translation.connection": "official_byok"},
                {"translation.model": "gemma4_26b_31b"},
                {"translation.model": "gpt_6_luna"},
            )
        ):
            submitted = await control.submit(
                "settings.apply",
                {"changes": changes},
                request_id=f"luna-history-{index}",
            )
            result = await control.wait(submitted["operation_id"], timeout=10)
            assert result["status"] in {"applied", "degraded"}
        translation = (await control.query("settings.current", {}))["settings"]["intent"][
            "translation"
        ]
        assert translation["model"] == "gpt_6_luna"
        assert translation["connection"] == "official_byok"
        assert translation["connection_history"]["gpt_6_luna"] == "official_byok"
        assert translation["openrouter_selection_alias"] is None
    finally:
        await app.stop()
    restarted = compose_headless_application(path)
    try:
        await restarted.start()
        restored = restarted.control()
        restored.bind_instance("luna-history-restarted")
        translation = (await restored.query("settings.current", {}))["settings"]["intent"][
            "translation"
        ]
        assert translation["model"] == "gpt_6_luna"
        assert translation["connection"] == "official_byok"
        assert translation["connection_history"]["gpt_6_luna"] == "official_byok"
    finally:
        await restarted.stop()


def test_headless_runtime_error_is_localized_in_dashboard_state(caplog) -> None:
    from puripuly_heart.ui.i18n import t

    presentation = HeadlessApplicationPresentation()
    presentation.set_locale("en")
    with caplog.at_level(logging.ERROR):
        presentation.show_message("local_stt.download_failed", is_error=True)

    assert presentation.dashboard.issue == t("local_stt.download_failed")
    assert [(record.levelno, record.getMessage()) for record in caplog.records] == [
        (logging.ERROR, "Application message: local_stt.download_failed")
    ]


@pytest.mark.asyncio
async def test_declared_extension_secret_roundtrip_is_private_and_revision_safe(
    tmp_path, monkeypatch
):
    extensions = tmp_path / "extensions"
    extensions.mkdir()
    (extensions / "sample.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "id": "sample",
                "name": "Sample",
                "url": "https://example.test/translate",
                "headers": {"Authorization": "Bearer {{secret:key}}"},
                "request": {"query": {}, "body": {"type": "json", "value": {"text": "{{text}}"}}},
                "response": {"type": "text"},
                "secrets": [{"id": "key", "label": "API Key"}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(application_runtime, "default_http_extensions_dir", lambda: extensions)
    monkeypatch.setenv("PURIPULY_HEART_SECRETS_PASSPHRASE", "isolated-test-passphrase")
    path = tmp_path / "settings.json"
    isolated_settings(path)
    app = compose_headless_application(path)
    try:
        await app.start()
        control = app.control()
        control.bind_instance("isolated-control")
        key = http_extension_secret_key("sample", "key")
        from puripuly_heart.ui.i18n import t

        catalog = await control.query("settings.choices", {})
        assert "sample" in catalog["choices"]["translation.http_extension_id"]
        terms = await control.query("consent.peer_translation", {})
        assert terms["accepted"] is False
        assert terms["terms"] == t("peer_translation_eula.body")
        current = await control.query("settings.current", {})
        current_locale = current["settings"]["intent"]["ui"]["locale"]
        valid_locale = next(
            (locale for locale in catalog["choices"]["locale"] if locale != current_locale),
            current_locale,
        )
        accepted = await control.submit(
            "settings.apply",
            {"changes": {"locale": valid_locale}},
            request_id="catalog-valid-locale",
        )
        applied = await control.wait(accepted["operation_id"], timeout=5)
        assert applied["status"] == "applied"
        assert (await control.query("settings.current", {}))["settings"]["intent"]["ui"][
            "locale"
        ] == valid_locale
        unchanged = await control.query("settings.current", {})
        malformed = await control.submit(
            "settings.apply",
            {
                "changes": {
                    "locale": valid_locale,
                    "translation.model": "not-a-model",
                }
            },
            request_id="catalog-invalid-model",
        )
        rejected = await control.wait(malformed["operation_id"], timeout=5)
        assert rejected["status"] == "rejected"
        after = await control.query("settings.current", {})
        assert after["settings"] == unchanged["settings"]
        assert after["revision"] == unchanged["revision"]
        assert current["revision"] <= unchanged["revision"]
        bad_operations = (
            (
                "capture.set",
                {"channel": "self", "enabled": True, "accept_terms": True},
            ),
            (
                "settings.apply",
                {"changes": {"locale": valid_locale, "languages": {"source": "not-a-language"}}},
            ),
            (
                "settings.apply",
                {"changes": {"osc.connection": {"mode": "off", "receive_port": True}}},
            ),
            (
                "settings.apply",
                {"changes": {"translation.connection_history": {"not-a-model": "direct"}}},
            ),
            ("settings.apply", {"changes": {"stt.cloud_free_tier_providers": []}}),
            ("provider.apply", {"channel": "self", "provider": "not-a-provider"}),
            ("models.install", {"model_ids": "not-a-list"}),
            (
                "overlay.calibrate",
                {"action": "change", "field": "unknown", "value": 1},
            ),
            ("overlay.calibrate", {"action": "change", "field": "distance", "value": True}),
            ("overlay.calibrate", {"action": "begin", "field": "anchor", "value": "top"}),
            ("secrets.set", {"name": key, "value": 1}),
            (
                "auth.login",
                {
                    "provider": "discord",
                    "referral_id": 5,
                    "open_browser": False,
                },
            ),
            (
                "auth.login",
                {
                    "provider": "qq",
                    "qq_identity": "offline",
                    "credential": "offline",
                    "open_browser": True,
                },
            ),
        )
        for index, (command, arguments) in enumerate(bad_operations):
            pending = await control.submit(
                command,
                arguments,
                request_id=f"invalid-{index}",
            )
            invalid = await control.wait(pending["operation_id"], timeout=5)
            assert invalid["status"] == "rejected"
        after_invalid = await control.query("settings.current", {})
        assert after_invalid["settings"] == after["settings"]
        assert (await control.query("secrets.presence", {}))["presence"][key] is False
        assert key in control.capabilities()["secret_keys"]
        before = await control.query("secrets.presence", {})
        assert before["presence"][key] is False
        secret = "sensitive-control-test-value"
        accepted = await control.submit(
            "secrets.set", {"name": key, "value": secret}, request_id="set-once"
        )
        assert accepted["terminal"] is False
        stored = await control.wait(accepted["operation_id"])
        assert stored["status"] == "applied" and stored["terminal"] is True
        assert stored["transaction"]["status"] == "settings_commit_success_runtime_applied"
        assert secret not in json.dumps(stored)
        assert secret not in json.dumps(await control.query("settings.current", {}))
        assert (await control.query("secrets.presence", {}))["presence"][key] is True
        duplicate = await control.submit(
            "secrets.set", {"name": key, "value": secret}, request_id="set-once"
        )
        assert duplicate == stored
        unknown = await control.submit(
            "secrets.set",
            {"name": "http_extension.sample.undeclared", "value": secret},
            request_id="undeclared",
        )
        rejected = await control.wait(unknown["operation_id"])
        assert rejected["status"] == "rejected" and rejected["terminal"] is True
        assert (await control.query("secrets.presence", {}))["presence"][key] is True
        unverified = await control.submit(
            "secrets.verify", {"name": key, "value": secret}, request_id="verify"
        )
        verdict = await control.wait(unverified["operation_id"])
        assert verdict["status"] == "action_required" and verdict["verification"] == "unavailable"
        assert verdict["terminal"] is True
        removed = await control.submit("secrets.delete", {"name": key}, request_id="delete")
        deleted = await control.wait(removed["operation_id"])
        assert deleted["status"] == "applied" and deleted["terminal"] is True
        assert (await control.query("secrets.presence", {}))["presence"][key] is False
    finally:
        await app.stop()


@pytest.mark.asyncio
async def test_control_microphone_reports_route_failure_pending_frame_meter_and_released_stop(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PURIPULY_HEART_SECRETS_PASSPHRASE", "isolated-test-passphrase")
    monkeypatch.setattr(
        application_runtime, "default_http_extensions_dir", lambda: tmp_path / "extensions"
    )
    path = tmp_path / "settings.json"
    isolated_settings(path)
    app = compose_headless_application(path)
    try:
        await app.start()
        control = app.control()
        control.bind_instance("isolated-microphone")
        microphone = app._microphone.microphone
        route = MicrophoneTestRouteObservation(
            saved_host_api="",
            actual_host_api="",
            requested_device="controlled",
            hostapi_index=None,
            resolved_device_idx=None,
            resolved_device_name=None,
            resolution_exception_class="DeviceUnavailable",
            resolution_exception_message=None,
            should_attempt_open=False,
            wasapi_auto_convert=False,
            wasapi_exclusive=False,
        )
        monkeypatch.setattr(
            wiring_microphone_test, "observe_microphone_test_route", lambda **_kwargs: route
        )
        receipt = await control.submit(
            "microphone.test", {"enabled": True}, request_id="missing-mic"
        )
        failed = await control.wait(receipt["operation_id"], timeout=5)
        assert failed["status"] == "action_required"
        assert failed["microphone_test"]["failure_reason"] == "input_route_unavailable"
        assert failed["microphone_test"]["effective_active"] is False
        assert (await control.query("app.status", {}))["microphone_test"][
            "failure_reason"
        ] == "input_route_unavailable"

        class ControlledSource:
            def __init__(self):
                self.frame = asyncio.Event()
                self.closed = 0

            async def frames(self):
                await self.frame.wait()
                yield AudioFrameF32(
                    samples=np.asarray([0.25, -0.75], dtype=np.float32), sample_rate_hz=16000
                )
                await asyncio.Event().wait()

            async def close(self):
                self.closed += 1

        source = ControlledSource()
        microphone.source_factory = lambda **_kwargs: source
        monkeypatch.setattr(
            wiring_microphone_test,
            "observe_microphone_test_route",
            lambda **_kwargs: replace(
                route,
                should_attempt_open=True,
                resolved_device_idx=2,
            ),
        )
        monkeypatch.setattr(
            wiring_microphone_test,
            "determine_self_mic_capture_channels",
            lambda **_kwargs: SelfMicCaptureChannelDecision(
                device_idx=2,
                internal_channels=1,
                preferred_capture_channels=1,
                metadata=SoundDeviceInputMetadata(
                    device_idx=2,
                    name="controlled",
                    max_input_channels=1,
                    default_samplerate=16000.0,
                    metadata_status="ok",
                ),
            ),
        )
        await microphone.close()
        microphone._owner = None
        pending_receipt = await control.submit(
            "microphone.test", {"enabled": True}, request_id="pending-stop-mic"
        )
        pending_start = await control.wait(pending_receipt["operation_id"], timeout=0.01)
        assert pending_start["terminal"] is False
        off_receipt = await control.submit(
            "microphone.test", {"enabled": False}, request_id="stop-pending-mic"
        )
        pending_off = await control.wait(off_receipt["operation_id"], timeout=5)
        assert pending_off["status"] == "applied"
        assert pending_off["microphone_test"]["state"] == "off"
        assert source.closed == 1
        assert (await control.wait(pending_receipt["operation_id"], timeout=5))[
            "status"
        ] == "cancelled"
        source = ControlledSource()
        receipt = await control.submit("microphone.test", {"enabled": True}, request_id="ready-mic")
        pending = await control.wait(receipt["operation_id"], timeout=0.01)
        assert pending["terminal"] is False
        assert (await control.query("app.status", {}))["microphone_test"]["state"] == "pending"
        source.frame.set()
        ready = await control.wait(receipt["operation_id"], timeout=5)
        assert ready["status"] == "applied"
        status = (await control.query("app.status", {}))["microphone_test"]
        assert status["state"] == "ready"
        assert status["effective_active"] is True
        assert status["meter_level"] == 0.75
        off = await control.submit("microphone.test", {"enabled": False}, request_id="stop-mic")
        stopped = await control.wait(off["operation_id"], timeout=5)
        assert stopped["status"] == "applied"
        assert stopped["microphone_test"]["state"] == "off"
        assert source.closed == 1
    finally:
        await app.stop()


@pytest.mark.asyncio
async def test_auth_challenge_stays_pending_until_owner_finishes_and_cancel_does_not_fake_success(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PURIPULY_HEART_SECRETS_PASSPHRASE", "isolated-test-passphrase")
    monkeypatch.setattr(
        application_runtime, "default_http_extensions_dir", lambda: tmp_path / "extensions"
    )
    path = tmp_path / "settings.json"
    isolated_settings(path)
    app = compose_headless_application(path)
    try:
        await app.start()
        control = app.control()
        control.bind_instance("isolated-auth")
        for outcome in (True, False):
            finish = asyncio.Event()
            entered = asyncio.Event()

            async def authorization(*, authorization_url_sink, open_browser, **_kwargs):
                assert open_browser is False
                authorization_url_sink("https://example.test/authorize?challenge=public")
                entered.set()
                await finish.wait()
                return outcome

            monkeypatch.setattr(app, "start_discord_managed_auth_from_dialog", authorization)
            receipt = await control.submit(
                "auth.login",
                {"provider": "discord", "open_browser": False},
                request_id=f"auth-{outcome}",
            )
            await asyncio.wait_for(entered.wait(), 2)
            pending = await control.operation(receipt["operation_id"])
            assert pending["status"] == "running" and pending["terminal"] is False
            assert (await control.wait(receipt["operation_id"], timeout=0.01))["terminal"] is False
            challenge = (await control.query("auth.status", {}))["challenge"]
            assert challenge["operation_id"] == receipt["operation_id"]
            assert challenge["authorization_url"].startswith("https://example.test/authorize")
            cancellation = await control.cancel(receipt["operation_id"])
            assert cancellation["cancellation"] == "unsupported"
            assert cancellation["terminal"] is False
            finish.set()
            finished = await control.wait(receipt["operation_id"], timeout=2)
            assert finished["status"] == ("applied" if outcome else "failed")
            assert finished["terminal"] is True
            assert (await control.query("auth.status", {}))["challenge"] is None
            assert "authorization_url" not in finished
    finally:
        await app.stop()


@pytest.mark.asyncio
async def test_interleaved_transactions_retain_per_operation_outcome() -> None:
    results = SettingsTransactionResultOwner()
    first_ready = asyncio.Event()
    second_done = asyncio.Event()
    applied = TransactionResult("settings_commit_success_runtime_applied", None, None)
    degraded = TransactionResult("settings_commit_success_runtime_degraded", None, None)

    async def first_operation() -> str:
        with results.capture():
            results.set(applied)
            first_ready.set()
            await second_done.wait()
            return results.current.status

    async def second_operation() -> str:
        await first_ready.wait()
        with results.capture():
            results.set(degraded)
            second_done.set()
            return results.current.status

    assert await asyncio.gather(first_operation(), second_operation()) == [
        applied.status,
        degraded.status,
    ]


@pytest.mark.asyncio
async def test_control_translation_enable_never_starts_implicit_authorization(
    tmp_path, monkeypatch
):
    import webbrowser

    from puripuly_heart.app.wiring import wiring_managed_account

    authorization_attempts = []

    class OfflineBroker:
        def __init__(self, **_kwargs):
            pass

        def __getattr__(self, name):
            async def forbidden_request(*_args, **_kwargs):
                if name == "start_discord_oauth":
                    authorization_attempts.append("broker")
                raise AssertionError("offline test forbids external broker requests")

            return forbidden_request

        async def close(self):
            pass

    def forbidden_authorization(*_args, **_kwargs):
        authorization_attempts.append("browser_or_listener")
        raise AssertionError("control translation enable initiated browser or listener")

    monkeypatch.setenv("PURIPULY_HEART_SECRETS_PASSPHRASE", "isolated-test-passphrase")
    monkeypatch.setattr(
        application_runtime, "default_http_extensions_dir", lambda: tmp_path / "extensions"
    )
    monkeypatch.setattr(wiring_managed_account, "HttpManagedOpenRouterBrokerClient", OfflineBroker)
    monkeypatch.setattr(webbrowser, "open", forbidden_authorization)
    path = tmp_path / "settings.json"
    isolated_settings(path)
    app = compose_headless_application(path)
    try:
        await app.start()
        release = app._managed.managed.release.service
        assert release is not None
        release.discord_oauth_listener_factory = forbidden_authorization
        app.control().bind_instance("offline-control")
        receipt = await app.control().submit(
            "translation.set", {"enabled": True}, request_id="enable-without-auth"
        )
        result = await app.control().wait(receipt["operation_id"], timeout=5)
        assert result["status"] == "action_required"
        assert result["terminal"] is True
        assert (await app.control().query("app.status", {}))["translation_enabled"] is False
        osc_runtime = app._runtime_shutdown.vrc_mic_sync()
        assert osc_runtime is not None
        await osc_runtime.router._application.set_translation(True)
        assert (await app.control().query("app.status", {}))["translation_enabled"] is False
        assert authorization_attempts == []
    finally:
        await app.stop()


@pytest.mark.asyncio
async def test_gui_cli_and_osc_mutations_share_revision_and_preserve_focused_edits(
    tmp_path, monkeypatch
):
    from puripuly_heart.core.managed_openrouter_release import (
        UnavailableManagedOpenRouterReleaseClient,
    )

    from puripuly_heart.app.ports.settings_view import LocaleSettingsIntent
    from puripuly_heart.app.wiring import wiring_managed_account

    monkeypatch.setenv("PURIPULY_HEART_SECRETS_PASSPHRASE", "isolated-test-passphrase")
    monkeypatch.setattr(
        application_runtime, "default_http_extensions_dir", lambda: tmp_path / "extensions"
    )
    monkeypatch.setattr(
        wiring_managed_account,
        "HttpManagedOpenRouterBrokerClient",
        lambda **_kwargs: UnavailableManagedOpenRouterReleaseClient(),
    )
    path = tmp_path / "settings.json"
    isolated_settings(path)
    app = compose_headless_application(path)
    try:
        await app.start()
        control = app.control()
        control.bind_instance("concurrent-control")
        initial = await control.query("settings.current", {})
        osc_runtime = app._runtime_shutdown.vrc_mic_sync()
        assert osc_runtime is not None
        osc = osc_runtime.router._application

        await control._lock.acquire()
        try:
            gui = asyncio.create_task(app.apply_settings_intent(LocaleSettingsIntent("ja")))
            await asyncio.sleep(0)
            cli = await control.submit(
                "settings.apply",
                {"changes": {"locale": "en"}},
                request_id="stale-locale",
                expected_revision=initial["revision"],
            )
            await asyncio.sleep(0)
            osc_mutation = asyncio.create_task(osc.set_secondary_target_language("zh-CN"))
            await asyncio.sleep(0)
            assert not gui.done() and not osc_mutation.done()
        finally:
            control._lock.release()
        await gui
        cli_result = await control.wait(cli["operation_id"], timeout=5)
        await osc_mutation
        final = await control.query("settings.current", {})
        assert cli_result["status"] == "rejected"
        assert cli_result["error"]["code"] == "revision_conflict"
        assert final["settings"]["intent"]["ui"]["locale"] == "ja"
        assert final["settings"]["intent"]["languages"]["secondary_target_language"] == "zh-CN"
        assert final["revision"] > initial["revision"]
    finally:
        await app.stop()


@pytest.mark.asyncio
async def test_overlay_command_waits_for_owned_start_and_distinguishes_intent_from_output(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PURIPULY_HEART_SECRETS_PASSPHRASE", "isolated-test-passphrase")
    monkeypatch.setattr(
        application_runtime, "default_http_extensions_dir", lambda: tmp_path / "extensions"
    )
    path = tmp_path / "settings.json"
    isolated_settings(path)
    app = compose_headless_application(path)
    try:
        await app.start()
        control = app.control()
        control.bind_instance("isolated-overlay")
        owner = app._overlay.overlay
        monkeypatch.setattr(owner, "output_provider", lambda: None)
        gate = asyncio.Event()
        entered = asyncio.Event()
        original_start = OverlayApplicationOwner.run_start
        attempts = []

        async def delayed_start(self, runtime=None):
            if self is owner:
                attempts.append(self.active_target)
                if len(attempts) == 1:
                    await self.handle_start_failure("steamvr_not_running")
                    return
                entered.set()
                await gate.wait()
            return await original_start(self, runtime)

        monkeypatch.setattr(OverlayApplicationOwner, "run_start", delayed_start)
        try:
            receipt = await control.submit(
                "overlay.set", {"enabled": True}, request_id="overlay-on"
            )
            await asyncio.wait_for(entered.wait(), 5)
            assert (await control.wait(receipt["operation_id"], timeout=0.01))["terminal"] is False
            starting = await control.query("overlay.status", {})
            assert starting["output"]["desired_enabled"] is True
            assert starting["output"]["lifecycle"] == "starting"
            assert starting["output"]["presentation_ready"] is False
            assert starting["output"]["fallback_active"] is True
            gate.set()
            result = await control.wait(receipt["operation_id"], timeout=5)
            assert attempts == ["steamvr", "desktop"]
            assert result["status"] == "action_required"
            assert result["reason"] == "output_unavailable"
            assert result["intent"] == {"requested_enabled": True, "desired_enabled": True}
            assert result["output"]["failure_reason"] == "output_unavailable"
            assert result["output"]["effective_target"] is None
            assert result["output"]["presentation_ready"] is False

            off_receipt = await control.submit(
                "overlay.set", {"enabled": False}, request_id="overlay-off"
            )
            off = await control.wait(off_receipt["operation_id"], timeout=5)
            observed = (await control.query("overlay.status", {}))["output"]
            assert off["status"] == "applied"
            assert off["intent"] == {"requested_enabled": False, "desired_enabled": False}
            assert observed["lifecycle"] == "off"
            assert observed["runtime_active"] is False
            assert observed["process_state"] is None
            assert owner.runtime is None or not owner.runtime.has_resources()
        finally:
            gate.set()
    finally:
        await app.stop()


@pytest.mark.asyncio
async def test_peer_terms_are_explicit_and_shared_acceptance_is_isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("PURIPULY_HEART_SECRETS_PASSPHRASE", "isolated-test-passphrase")
    monkeypatch.setattr(
        application_runtime, "default_http_extensions_dir", lambda: tmp_path / "extensions"
    )
    path = tmp_path / "settings.json"
    isolated_settings(path)
    app = compose_headless_application(path)
    try:
        await app.start()
        control = app.control()
        control.bind_instance("isolated-consent")
        from puripuly_heart.ui.i18n import t

        terms = await control.query("consent.peer_translation", {})
        assert terms["accepted"] is False
        assert terms["terms"] == t("peer_translation_eula.body")
        request = await control.submit(
            "capture.set",
            {"channel": "peer", "enabled": True},
            request_id="peer-without-consent",
        )
        denied = await control.wait(request["operation_id"], timeout=5)
        assert denied["status"] == "action_required"
        assert denied["action"] == "accept_peer_translation_terms"
        assert (await control.query("consent.peer_translation", {}))["accepted"] is False

        async def unavailable_peer_capture(_enabled):
            raise PeerCaptureTargetUnavailable("isolated capture unavailable")

        monkeypatch.setattr(app, "set_peer_translation_enabled", unavailable_peer_capture)
        request = await control.submit(
            "capture.set",
            {"channel": "peer", "enabled": True, "accept_terms": True},
            request_id="peer-explicit-consent",
        )
        unavailable = await control.wait(request["operation_id"], timeout=5)
        assert unavailable["status"] == "action_required"
        assert unavailable["error"]["code"] == "capture_target_unavailable"
        assert (await control.query("consent.peer_translation", {}))["accepted"] is True
        assert app.compatibility_settings().state.peer_translation.eula_accepted is True
    finally:
        await app.stop()


@pytest.mark.asyncio
async def test_activation_notice_cli_catalog_apply_and_durable_reload(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PURIPULY_HEART_SECRETS_PASSPHRASE", "isolated-test-passphrase")
    path = tmp_path / "settings.json"
    isolated_settings(path)
    app = compose_headless_application(path)
    key = "chatbox.activation_notice.enabled"
    try:
        await app.start()
        control = app.control()
        control.bind_instance("activation-notice-control")
        catalog = await control.query("settings.choices", {})
        assert key in control.capabilities()["settings_fields"]
        assert key in catalog["fields"]
        assert catalog["field_types"][key] == "boolean"
        assert catalog["field_schemas"][key] == {"type": "boolean", "free_form": False}
        initial = await control.query("settings.current", {})
        assert initial["settings"]["intent"]["osc"]["activation_notice_enabled"] is True

        submitted = await control.submit(
            "settings.apply",
            {"changes": {key: False}},
            request_id="disable-activation-notice",
        )
        result = await control.wait(submitted["operation_id"], timeout=5)
        assert result["status"] == "applied"
        disabled = await control.query("settings.current", {})
        expected = deepcopy(initial["settings"])
        expected["intent"]["osc"]["activation_notice_enabled"] = False
        assert disabled["settings"] == expected
        for index, invalid in enumerate((None, 0, 1, "false", "true", [], {})):
            submitted = await control.submit(
                "settings.apply",
                {"changes": {key: invalid}},
                request_id=f"invalid-activation-notice-{index}",
            )
            result = await control.wait(submitted["operation_id"], timeout=5)
            assert result["status"] == "rejected"
            assert await control.query("settings.current", {}) == disabled
    finally:
        await app.stop()
    restarted = compose_headless_application(path)
    try:
        await restarted.start()
        control = restarted.control()
        control.bind_instance("activation-notice-restarted")
        assert (await control.query("settings.current", {}))["settings"]["intent"]["osc"][
            "activation_notice_enabled"
        ] is False
    finally:
        await restarted.stop()
