from __future__ import annotations

import asyncio
import io
import json
import logging
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

from puripuly_heart.app.ports.settings_view import LocaleSettingsIntent
from puripuly_heart.app.services.canonical_settings_persistence import (
    CanonicalSettingsPatchRepository,
)
from puripuly_heart.cli.host import HostedApplication
from puripuly_heart.composition import application_runtime
from puripuly_heart.composition.headless_application import compose_headless_application
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext, SecretsIntent
from puripuly_heart.config.settings_vnext.serialization import to_dict
from puripuly_heart.core.runtime_logging import RuntimeLoggingSinks


def isolated_settings(path):
    settings = AppSettingsVNext()
    settings = replace(
        settings,
        intent=replace(
            settings.intent,
            secrets=SecretsIntent(backend="encrypted_file", encrypted_file_path="secrets.json"),
            osc=replace(settings.intent.osc, connection_mode="off"),
            stt=replace(settings.intent.stt, custom_terms={"ko": ["private phrase"]}),
        ),
    )
    path.write_text(json.dumps(to_dict(settings)), encoding="utf-8")


@pytest.fixture(autouse=True)
async def isolated_composed_logging(tmp_path, monkeypatch):
    root = logging.getLogger()
    level = root.level
    monkeypatch.setattr(root, "handlers", list(root.handlers))
    monkeypatch.setattr(root, "filters", list(root.filters))
    monkeypatch.setattr(root, "level", level)
    for handler in root.handlers:
        monkeypatch.setattr(handler, "filters", list(handler.filters))
    compose = compose_headless_application
    applications = []

    def tracked_compose(path):
        log_file = tmp_path / f"headless-{len(applications)}.log"
        sinks = RuntimeLoggingSinks(
            logging.StreamHandler(io.StringIO()),
            logging.FileHandler(log_file, encoding="utf-8"),
            log_file,
        )
        try:
            application = compose(path, runtime_logging_sinks=sinks)
        except BaseException:
            sinks.close()
            raise
        applications.append((application, sinks))
        return application

    monkeypatch.setitem(globals(), "compose_headless_application", tracked_compose)
    try:
        yield
    finally:
        try:
            for application, sinks in reversed(applications):
                try:
                    if application.control().pipeline.self_translation_channel is not None:
                        await application.stop()
                    else:
                        service = application._runtime_logging.installed_service
                        if service is not None:
                            service.close()
                finally:
                    sinks.close()
                    sinks.stream_handler.close()
        finally:
            root.setLevel(level)


@pytest.fixture
def offline(tmp_path, monkeypatch):
    monkeypatch.setenv("PURIPULY_HEART_SECRETS_PASSPHRASE", "isolated-test-passphrase")
    monkeypatch.setattr(
        application_runtime, "default_http_extensions_dir", lambda: tmp_path / "extensions"
    )
    path = tmp_path / "settings.json"
    isolated_settings(path)
    return path


@pytest.mark.asyncio
async def test_committed_gui_event_is_unsolicited_and_staged_failure_is_invisible(
    offline, monkeypatch
):
    app = compose_headless_application(offline)
    try:
        await app.start()
        control = app.control()
        control.bind_instance("offline-commit")
        initial = await control.query("settings.current", {})
        assert initial["settings"]["intent"]["stt"]["custom_terms"] == "<redacted>"
        assert "private phrase" not in json.dumps(initial)
        stream = control.subscribe(topics=["settings"])
        event_task = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        await app.apply_settings_intent(LocaleSettingsIntent("ja"))
        event = await asyncio.wait_for(event_task, 5)
        updated = await control.query("settings.current", {})
        assert event["revision"] == updated["revision"] == initial["revision"] + 1
        assert updated["settings"]["intent"]["ui"]["locale"] == "ja"
        await app.apply_settings_intent(LocaleSettingsIntent("ja"))
        assert (await control.query("settings.current", {}))["revision"] == updated["revision"]
        await stream.aclose()

        settings_app = app._settings.application
        entered = asyncio.Event()
        release = asyncio.Event()
        original = type(settings_app.runtime_effects).prepare

        async def staged(self, *args, **kwargs):
            entered.set()
            await release.wait()
            return await original(self, *args, **kwargs)

        monkeypatch.setattr(type(settings_app.runtime_effects), "prepare", staged)
        before = await control.query("settings.current", {})
        candidate = replace(
            app.compatibility_settings(),
            intent=replace(
                app.compatibility_settings().intent,
                ui=replace(app.compatibility_settings().intent.ui, locale="en"),
            ),
        )
        original_persist = type(control.settings.persistence).persist

        def fail_persist(self, *_args):
            raise OSError("isolated save failure")

        monkeypatch.setattr(type(control.settings.persistence), "persist", fail_persist)
        pending = asyncio.create_task(settings_app.apply_direct(candidate))
        await asyncio.wait_for(entered.wait(), 5)
        staged_query = await control.query("settings.current", {})
        assert staged_query == before
        release.set()
        assert await pending is False
        assert await control.query("settings.current", {}) == before
        monkeypatch.setattr(type(control.settings.persistence), "persist", original_persist)
        assert not [
            item
            for item in control.events.history
            if item["topic"] == "settings" and item["revision"] > before["revision"]
        ]
    finally:
        await app.stop()


@pytest.mark.asyncio
async def test_gui_commit_reaches_idle_subscriber_without_any_prior_query(offline):
    app = compose_headless_application(offline)
    try:
        await app.start()
        control = app.control()
        control.bind_instance("offline-subscriber")
        stream = control.subscribe(topics=["settings"])
        event_task = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        await app.apply_settings_intent(LocaleSettingsIntent("ja"))
        event = await asyncio.wait_for(event_task, 5)
        current = await control.query("settings.current", {})
        assert event["revision"] == current["revision"]
        assert current["settings"]["intent"]["ui"]["locale"] == "ja"
        await stream.aclose()
    finally:
        await app.stop()


@pytest.mark.asyncio
async def test_shutdown_before_accepted_task_first_runs_reports_terminal_interruption(offline):
    app = compose_headless_application(offline)
    control = app.control()
    control.bind_instance("offline-early-freeze")
    accepted = await control.submit(
        "settings.apply", {"changes": {"locale": "ja"}}, request_id="early"
    )
    control.freeze_ingress()
    await control.drain_operations()
    receipt = await control.operation(accepted["operation_id"])
    assert receipt["status"] == "interrupted"
    assert receipt["terminal"] is True
    assert "transaction" not in receipt
    assert any(
        event.get("topic") == "operation"
        and event.get("operation_id") == accepted["operation_id"]
        and event.get("status") == "interrupted"
        for event in control.events.history
    )
    assert json.loads(offline.read_text(encoding="utf-8"))["intent"]["ui"]["locale"] == "en"


@pytest.mark.asyncio
async def test_completed_settings_receipt_retains_applied_transaction(offline):
    app = compose_headless_application(offline)
    try:
        await app.start()
        control = app.control()
        control.bind_instance("offline-completed-settings")
        initial = await control.query("settings.current", {})
        submitted = await control.submit(
            "settings.apply",
            {"changes": {"locale": "ja"}},
            request_id="completed-locale",
        )
        receipt = await control.wait(submitted["operation_id"], timeout=5)
        assert receipt["status"] == "applied"
        assert receipt["terminal"] is True
        assert receipt["transaction"]["status"] == "settings_commit_success_runtime_applied"
        assert receipt["revision"] == initial["revision"] + 1
        assert (await control.query("settings.current", {}))["revision"] == receipt["revision"]
    finally:
        await app.stop()
    assert (await control.operation(submitted["operation_id"])) == receipt


@pytest.mark.asyncio
async def test_host_shutdown_interrupts_active_and_queued_settings_without_late_commit(
    offline, monkeypatch
):
    app = compose_headless_application(offline)

    class Lease:
        def publish(self, **_kwargs):
            pass

        def close(self):
            pass

    host = HostedApplication(app, Lease())
    entered = asyncio.Event()
    release = asyncio.Event()
    original = CanonicalSettingsPatchRepository.save

    async def slow_save(self, request):
        entered.set()
        await release.wait()
        return await original(self, request)

    try:
        await host.start()
        monkeypatch.setattr(CanonicalSettingsPatchRepository, "save", slow_save)
        control = host.control
        initial = await control.query("settings.current", {})
        active = await control.submit(
            "settings.apply", {"changes": {"locale": "ja"}}, request_id="slow-active"
        )
        await asyncio.wait_for(entered.wait(), 5)
        queued = await control.submit(
            "settings.apply", {"changes": {"locale": "en"}}, request_id="queued"
        )
        await asyncio.sleep(0)
        stop = await control.submit("app.stop", {}, request_id="stop")
        assert (await control.wait(stop["operation_id"], timeout=5))["status"] == "applied"
        await asyncio.wait_for(host.wait_for_stop(), 6)
        await asyncio.wait_for(host.close(), 6)
        active_receipt = await control.operation(active["operation_id"])
        queued_receipt = await control.operation(queued["operation_id"])
        assert active_receipt["status"] == queued_receipt["status"] == "interrupted"
        assert "transaction" not in active_receipt
        assert "transaction" not in queued_receipt
        assert queued_receipt["revision"] == initial["revision"]
        assert (await control.query("settings.current", {})) == initial
        assert not [
            operation.task
            for operation in control._operations.values()
            if operation.task is not None and not operation.task.done()
        ]
    finally:
        release.set()
        await host.close()


@pytest.mark.asyncio
async def test_noninterruptible_persistence_drains_before_host_closes(offline, monkeypatch):
    app = compose_headless_application(offline)

    class Lease:
        def publish(self, **_kwargs):
            pass

        def close(self):
            pass

    host = HostedApplication(app, Lease())
    entered = threading.Event()
    release = threading.Event()
    original = type(host.control.settings.persistence).persist

    def slow_save(self, path, settings):
        entered.set()
        if not release.wait(5):
            raise TimeoutError("save gate not released")
        return original(self, path, settings)

    monkeypatch.setattr(type(host.control.settings.persistence), "persist", slow_save)
    try:
        await host.start()
        control = host.control
        initial = await control.query("settings.current", {})
        stream = control.subscribe(topics=["settings"])
        event_task = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        active = await control.submit(
            "settings.apply", {"changes": {"locale": "ja"}}, request_id="persist-active"
        )
        assert await asyncio.to_thread(entered.wait, 5)
        assert (await control.query("settings.current", {})) == initial
        assert not event_task.done()
        queued = await control.submit(
            "settings.apply", {"changes": {"locale": "en"}}, request_id="persist-queued"
        )
        stop = await control.submit("app.stop", {}, request_id="persist-stop")
        await control.wait(stop["operation_id"], timeout=5)
        closing = asyncio.create_task(host.close())
        await asyncio.sleep(0.05)
        assert not closing.done()
        release.set()
        await asyncio.wait_for(closing, 6)
        delivered = await asyncio.wait_for(event_task, 5)
        assert delivered["revision"] == initial["revision"] + 1
        await stream.aclose()
        active_receipt = await control.operation(active["operation_id"])
        queued_receipt = await control.operation(queued["operation_id"])
        current = await control.query("settings.current", {})
        assert active_receipt["status"] == "interrupted"
        assert active_receipt["terminal"] is True
        assert active_receipt["revision"] == delivered["revision"] == current["revision"]
        assert active_receipt["transaction"]["status"] in {
            "settings_commit_success_runtime_interrupted",
            "settings_commit_success_runtime_applied",
            "settings_commit_success_runtime_degraded",
        }
        assert queued_receipt["status"] == "interrupted"
        assert queued_receipt["terminal"] is True
        assert queued_receipt["revision"] == initial["revision"]
        assert "transaction" not in queued_receipt
        assert current["settings"]["intent"]["ui"]["locale"] == "ja"
        assert not [
            operation.task
            for operation in control._operations.values()
            if operation.task is not None and not operation.task.done()
        ]
        assert json.loads(offline.read_text(encoding="utf-8"))["intent"]["ui"]["locale"] == "ja"
    finally:
        release.set()
        await host.close()


@pytest.mark.asyncio
async def test_shutdown_after_commit_before_runtime_reports_interrupted_transaction(
    offline, monkeypatch
):
    from puripuly_heart.app.services.provider_runtime_apply import (
        UiPromptClipboardStateRuntimeApplyAdapter,
    )

    app = compose_headless_application(offline)

    class Lease:
        def publish(self, **_kwargs):
            pass

        def close(self):
            pass

    host = HostedApplication(app, Lease())
    persist_entered = threading.Event()
    persist_release = threading.Event()
    runtime_entered = asyncio.Event()
    original_persist = type(host.control.settings.persistence).persist

    def slow_save(self, path, settings):
        persist_entered.set()
        if not persist_release.wait(5):
            raise TimeoutError("save gate not released")
        return original_persist(self, path, settings)

    async def blocked_runtime(self, request):
        runtime_entered.set()
        await asyncio.Event().wait()

    try:
        await host.start()
        monkeypatch.setattr(type(host.control.settings.persistence), "persist", slow_save)
        monkeypatch.setattr(
            UiPromptClipboardStateRuntimeApplyAdapter, "apply_runtime", blocked_runtime
        )
        control = host.control
        initial = await control.query("settings.current", {})
        active = await control.submit(
            "settings.apply",
            {"changes": {"locale": "ja"}},
            request_id="commit-before-runtime",
        )
        assert await asyncio.to_thread(persist_entered.wait, 5)
        queued = await control.submit(
            "settings.apply",
            {"changes": {"locale": "en"}},
            request_id="uncommitted-queued",
        )
        closing = asyncio.create_task(host.close())
        await asyncio.sleep(0.05)
        assert not closing.done()
        persist_release.set()
        await asyncio.wait_for(runtime_entered.wait(), 5)
        control._operations[active["operation_id"]].task.cancel()
        await asyncio.wait_for(closing, 6)
        active_receipt = await control.operation(active["operation_id"])
        queued_receipt = await control.operation(queued["operation_id"])
        current = await control.query("settings.current", {})
        assert active_receipt["status"] == "interrupted"
        assert active_receipt["terminal"] is True
        assert (
            active_receipt["transaction"]["status"] == "settings_commit_success_runtime_interrupted"
        )
        assert active_receipt["revision"] == current["revision"] == initial["revision"] + 1
        assert "private phrase" not in json.dumps(active_receipt)
        assert queued_receipt["status"] == "interrupted"
        assert queued_receipt["revision"] == initial["revision"]
        assert "transaction" not in queued_receipt
        assert current["settings"]["intent"]["ui"]["locale"] == "ja"
        assert json.loads(offline.read_text(encoding="utf-8"))["intent"]["ui"]["locale"] == "ja"
        assert not [
            operation.task
            for operation in control._operations.values()
            if operation.task is not None and not operation.task.done()
        ]
    finally:
        persist_release.set()
        if "active" in locals():
            operation = host.control._operations[active["operation_id"]]
            if operation.task is not None and not operation.task.done():
                operation.task.cancel()
        await host.close()


@pytest.mark.asyncio
async def test_model_install_wait_releases_global_lock_but_keeps_backend_order(offline):
    app = compose_headless_application(offline)
    started = asyncio.Event()
    release = asyncio.Event()
    installs = []
    snapshot = SimpleNamespace(
        required_cpu_model_ids=("cpu-test",),
        gpu_model_id="gpu-test",
        models=(SimpleNamespace(model_id="cpu-test", backend="cpu"),),
    )

    class Provisioning:
        def __init__(self):
            self.snapshot = snapshot

        def start_install(self, request):
            installs.append(request)

            async def finish():
                started.set()
                await release.wait()
                return SimpleNamespace(cancelled=False, failed_model_ids=(), snapshot=snapshot)

            return asyncio.create_task(finish())

    try:
        await app.start()
        control = app.control()
        control.bind_instance("isolated-model-wait")
        control.provisioning = lambda: Provisioning()
        first = await control.submit(
            "models.install",
            {"backend": "cpu"},
            request_id="first-install",
        )
        await asyncio.wait_for(started.wait(), 3)
        second = await control.submit(
            "models.install",
            {"backend": "cpu"},
            request_id="second-install",
        )
        initial = (await control.query("settings.current", {}))["revision"]
        settings = await control.submit(
            "settings.apply",
            {"changes": {"locale": "ja"}},
            request_id="during-install",
            expected_revision=initial,
        )
        committed = await control.wait(settings["operation_id"], timeout=5)
        assert committed["status"] == "applied"
        assert committed["transaction"]["status"] == "settings_commit_success_runtime_applied"
        assert (await control.wait(first["operation_id"], timeout=0.01))["terminal"] is False
        assert len(installs) == 1
        off = await control.submit(
            "capture.set",
            {"channel": "self", "enabled": False},
            request_id="off-during-install",
        )
        assert (await control.wait(off["operation_id"], timeout=5))["status"] == "applied"
        release.set()
        assert (await control.wait(first["operation_id"], timeout=5))["status"] == "applied"
        assert (await control.wait(second["operation_id"], timeout=5))["status"] == "applied"
        assert len(installs) == 2
    finally:
        release.set()
        await app.stop()


@pytest.mark.asyncio
async def test_freeze_cancels_active_install_and_never_starts_queued_backend_work(offline):
    app = compose_headless_application(offline)
    started = asyncio.Event()
    release = asyncio.Event()
    installs = []
    snapshot = SimpleNamespace(
        required_cpu_model_ids=("cpu-test",),
        gpu_model_id="gpu-test",
        models=(SimpleNamespace(model_id="cpu-test", backend="cpu"),),
    )

    class Provisioning:
        def __init__(self):
            self.snapshot = snapshot

        def start_install(self, request):
            installs.append(request)

            async def finish():
                started.set()
                await release.wait()
                return SimpleNamespace(cancelled=False, failed_model_ids=(), snapshot=snapshot)

            return asyncio.create_task(finish())

    try:
        await app.start()
        control = app.control()
        control.bind_instance("isolated-model-freeze")
        owner = Provisioning()
        control.provisioning = lambda: owner
        first = await control.submit(
            "models.install", {"backend": "cpu"}, request_id="active-install"
        )
        await asyncio.wait_for(started.wait(), 3)
        second = await control.submit(
            "models.install", {"backend": "cpu"}, request_id="queued-install"
        )
        await asyncio.sleep(0)
        control.freeze_ingress()
        await asyncio.wait_for(control.drain_operations(), 5)
        assert (await control.operation(first["operation_id"]))["status"] == "interrupted"
        queued = await control.operation(second["operation_id"])
        assert queued["status"] == "interrupted"
        assert "transaction" not in queued
        assert len(installs) == 1
    finally:
        release.set()
        await app.stop()


@pytest.mark.asyncio
async def test_install_cancel_still_targets_started_owner_work(offline):
    app = compose_headless_application(offline)
    started = asyncio.Event()
    cancelled = []
    snapshot = SimpleNamespace(
        required_cpu_model_ids=("cpu-test",),
        gpu_model_id="gpu-test",
        models=(SimpleNamespace(model_id="cpu-test", backend="cpu"),),
    )

    class Provisioning:
        def __init__(self):
            self.snapshot = snapshot

        def start_install(self, _request):
            async def finish():
                started.set()
                await asyncio.Event().wait()

            return asyncio.create_task(finish())

        async def cancel_install(self, backend):
            cancelled.append(backend)

    try:
        await app.start()
        control = app.control()
        control.bind_instance("isolated-model-cancel")
        owner = Provisioning()
        control.provisioning = lambda: owner
        submitted = await control.submit(
            "models.install",
            {"backend": "cpu"},
            request_id="cancel-install",
        )
        await asyncio.wait_for(started.wait(), 3)
        receipt = await asyncio.wait_for(control.cancel(submitted["operation_id"]), 5)
        assert receipt["status"] == "cancelled"
        assert receipt["terminal"] is True
        assert cancelled == ["cpu"]
    finally:
        await app.stop()


@pytest.mark.asyncio
async def test_manual_output_wait_allows_safe_mutations_but_orders_provider_transition(
    offline, monkeypatch
):
    app = compose_headless_application(offline)
    entered = asyncio.Event()
    release = asyncio.Event()
    translated = []

    from uuid import uuid4

    from puripuly_heart.core.orchestrator.self_translation_channel import TextSubmissionAdmission

    async def slow_output(_text):
        entered.set()
        await release.wait()
        return TextSubmissionAdmission("admitted", uuid4(), ())

    async def translation(enabled, **_kwargs):
        translated.append(enabled)
        return True

    try:
        await app.start()
        control = app.control()
        control.bind_instance("isolated-manual-wait")
        monkeypatch.setattr(app, "submit_text", slow_output)
        monkeypatch.setattr(app, "set_translation_enabled", translation)
        manual = await control.submit("text.submit", {"text": "safe text"}, request_id="manual")
        await asyncio.wait_for(entered.wait(), 3)
        old_revision = (await control.query("settings.current", {}))["revision"]
        safe = await control.submit(
            "settings.apply",
            {"changes": {"locale": "ja"}},
            request_id="manual-safe",
        )
        committed = await control.wait(safe["operation_id"], timeout=5)
        assert committed["status"] == "applied"
        off = await control.submit(
            "capture.set",
            {"channel": "self", "enabled": False},
            request_id="manual-off",
        )
        assert (await control.wait(off["operation_id"], timeout=5))["status"] == "applied"
        stale = await control.submit(
            "translation.set",
            {"enabled": False},
            request_id="manual-stale",
            expected_revision=old_revision,
        )
        conflicting = await control.submit(
            "translation.set",
            {"enabled": False},
            request_id="manual-translation",
        )
        assert (await control.wait(conflicting["operation_id"], timeout=0.01))["terminal"] is False
        assert (await control.wait(stale["operation_id"], timeout=0.01))["terminal"] is False
        osc_runtime = app._runtime_shutdown.vrc_mic_sync()
        assert osc_runtime is not None
        osc_change = asyncio.create_task(
            osc_runtime.router._application.set_secondary_target_language("zh-CN")
        )
        await asyncio.sleep(0)
        assert not osc_change.done()
        assert translated == []
        assert (await control.wait(manual["operation_id"], timeout=0.01))["terminal"] is False
        release.set()
        assert (await control.wait(manual["operation_id"], timeout=5))["status"] == "applied"
        rejected = await control.wait(stale["operation_id"], timeout=5)
        assert rejected["status"] == "rejected"
        assert rejected["error"] == {
            "code": "revision_conflict",
            "current_revision": committed["revision"],
        }
        assert (await control.wait(conflicting["operation_id"], timeout=5))["status"] == "applied"
        assert translated == [False]
        await asyncio.wait_for(osc_change, 5)
        assert (await control.query("settings.current", {}))["settings"]["intent"]["languages"][
            "secondary_target_language"
        ] == "zh-CN"
    finally:
        release.set()
        await app.stop()


@pytest.mark.asyncio
async def test_long_output_wait_orders_cli_overlay_audio_and_gui_settings_capture(
    offline, monkeypatch
):
    app = compose_headless_application(offline)
    entered = asyncio.Event()
    release = asyncio.Event()

    from uuid import uuid4

    from puripuly_heart.core.orchestrator.self_translation_channel import TextSubmissionAdmission

    async def slow_output(_text):
        entered.set()
        await release.wait()
        return TextSubmissionAdmission("admitted", uuid4(), ())

    async def no_audio_effects(_settings):
        pass

    try:
        await app.start()
        control = app.control()
        control.bind_instance("isolated-complete-resource-order")
        monkeypatch.setattr(app, "submit_text", slow_output)
        target = app._peer_capture.peer.target
        target.devices = SimpleNamespace(names=lambda: ["Simulated A", "Simulated B"])
        target.processes = SimpleNamespace(candidates=lambda: ())
        target.runtime_effects = SimpleNamespace(apply_capture_target=no_audio_effects)
        app.begin_overlay_calibration()
        app.set_overlay_calibration_field("offset_x", 37)
        before = await control.query("settings.current", {})
        assert app.current_loopback_capture_option_value() == "device:"
        assert before["settings"]["intent"]["telemetry"]["enabled"] is True

        manual = await control.submit(
            "text.submit", {"text": "gated output"}, request_id="order-manual"
        )
        await asyncio.wait_for(entered.wait(), 3)
        mutations = [
            ("overlay.size", {"preset": "large"}),
            ("overlay.lock", {"locked": True}),
            ("overlay.position.reset", {}),
            ("overlay.calibrate", {"action": "apply"}),
            ("audio.target.set", {"value": "device:Simulated A"}),
        ]
        pending = [
            await control.submit(command, arguments, request_id=f"order-{index}")
            for index, (command, arguments) in enumerate(mutations)
        ]
        await asyncio.sleep(0)
        assert all(
            not receipt["terminal"]
            for receipt in [await control.operation(item["operation_id"]) for item in pending]
        )
        assert (await control.query("settings.current", {})) == before
        assert app.current_loopback_capture_option_value() == "device:"
        release.set()
        assert (await control.wait(manual["operation_id"], timeout=5))["status"] == "applied"
        outcomes = [await control.wait(item["operation_id"], timeout=5) for item in pending]
        assert [item["status"] for item in outcomes] == [
            "applied",
            "action_required",
            "action_required",
            "applied",
            "applied",
        ]
        after = await control.query("settings.current", {})
        assert after["revision"] > before["revision"]
        assert after["settings"]["intent"]["overlay"]["desktop_flet"]["size_preset"] == "large"
        assert after["settings"]["intent"]["overlay"]["calibration"]["offset_x"] == 37
        assert app.current_loopback_capture_option_value() == "device:Simulated A"

        entered.clear()
        release = asyncio.Event()
        second = await control.submit(
            "text.submit", {"text": "second gated output"}, request_id="order-manual-2"
        )
        await asyncio.wait_for(entered.wait(), 3)
        gui_capture = asyncio.create_task(app.apply_loopback_capture_option("device:Simulated B"))
        gui_telemetry = asyncio.create_task(app.apply_telemetry_enabled(False))
        await asyncio.sleep(0)
        assert not gui_capture.done() and not gui_telemetry.done()
        assert app.current_loopback_capture_option_value() == "device:Simulated A"
        assert (await control.query("settings.current", {}))["settings"]["intent"]["telemetry"][
            "enabled"
        ] is True
        release.set()
        assert (await control.wait(second["operation_id"], timeout=5))["status"] == "applied"
        await asyncio.wait_for(asyncio.gather(gui_capture, gui_telemetry), 5)
        assert app.current_loopback_capture_option_value() == "device:Simulated B"
        assert (await control.query("settings.current", {}))["settings"]["intent"]["telemetry"][
            "enabled"
        ] is False
    finally:
        release.set()
        await app.stop()


@pytest.mark.asyncio
async def test_freeze_interrupts_manual_output_and_queued_provider_without_late_dispatch(
    offline, monkeypatch
):
    app = compose_headless_application(offline)
    entered = asyncio.Event()
    release = asyncio.Event()
    transitions = []

    async def slow_output(_text):
        entered.set()
        await release.wait()

    async def translation(enabled, **_kwargs):
        transitions.append(enabled)
        return True

    try:
        await app.start()
        control = app.control()
        control.bind_instance("isolated-manual-freeze")
        monkeypatch.setattr(app, "submit_text", slow_output)
        monkeypatch.setattr(app, "set_translation_enabled", translation)
        manual = await control.submit(
            "text.submit", {"text": "pending"}, request_id="freeze-manual"
        )
        await asyncio.wait_for(entered.wait(), 3)
        queued = await control.submit(
            "translation.set",
            {"enabled": False},
            request_id="freeze-transition",
        )
        await asyncio.sleep(0)
        control.freeze_ingress()
        await asyncio.wait_for(control.drain_operations(), 5)
        assert (await control.operation(manual["operation_id"]))["status"] == "interrupted"
        assert (await control.operation(queued["operation_id"]))["status"] == "interrupted"
        assert transitions == []
    finally:
        release.set()
        await app.stop()


@pytest.mark.asyncio
async def test_control_overlay_receipts_preserve_owner_outcomes_and_captured_transaction(
    offline, monkeypatch
):
    from puripuly_heart.config.resolved import OVERLAY_TARGET_DESKTOP
    from puripuly_heart.config.settings_vnext.schema import DesktopFletOverlayPositionIntent

    app = compose_headless_application(offline)
    owner = app._settings.settings
    owner.start()
    app._settings.projection.remember_all(owner.canonical)
    control = app.control()
    control.bind_instance("offline-overlay-outcomes")

    async def command(name, args):
        accepted = await control.submit(name, args, request_id=f"{name}-{len(control._operations)}")
        return await control.wait(accepted["operation_id"], timeout=5)

    absent = await command("overlay.lock", {"locked": True})
    assert absent["status"] == "action_required"
    reset_absent = await command("overlay.position.reset", {})
    assert reset_absent["status"] == "action_required"

    def locked_file(_path, _settings):
        raise PermissionError("isolated persistence failure")

    persist = owner.persistence.persist
    monkeypatch.setattr(owner.persistence, "persist", locked_file)
    failed = await command("overlay.size", {"preset": "large"})
    assert failed["status"] == "persistence_failed"
    assert failed["transaction"]["status"] == "settings_commit_failed"
    monkeypatch.setattr(owner.persistence, "persist", persist)

    owner.canonical = replace(
        owner.canonical,
        intent=replace(
            owner.canonical.intent,
            overlay=replace(
                owner.canonical.intent.overlay,
                target=OVERLAY_TARGET_DESKTOP,
                desktop_flet=replace(
                    owner.canonical.intent.overlay.desktop_flet,
                    position=DesktopFletOverlayPositionIntent(x=140, y=85),
                ),
            ),
        ),
    )
    owner.persist_current()
    app._settings.projection.remember_all(owner.canonical)
    overlay = app._overlay.overlay
    overlay.active_target = OVERLAY_TARGET_DESKTOP
    overlay.state = "connected"

    class Bridge:
        def __init__(self):
            self.calls = 0

        async def broadcast_desktop_runtime_control(self, _payload):
            self.calls += 1
            if self.calls == 2:
                raise OSError("isolated bridge failure")

    bridge = Bridge()
    overlay.runtime = SimpleNamespace(
        current_bridge_for_runtime_command=lambda: bridge,
        renderer_events_or_none=lambda: None,
    )
    reset = await command("overlay.position.reset", {})
    assert reset["status"] == "failed"
    assert reset["reason"] == "desktop_broadcast_failed"
    assert reset["transaction"]["status"] == "settings_commit_success_runtime_degraded"


@pytest.mark.asyncio
async def test_gpu_discovery_receipt_follows_owner_snapshot(offline):
    from puripuly_heart.core.gpu_worker import GpuWorkerDevice
    from puripuly_heart.core.local_asr.local_asr_provider_runtime import (
        LocalASRProviderRuntimeSnapshot,
        ProviderRuntimeGpuSnapshot,
    )

    app = compose_headless_application(offline)
    try:
        await app.start()
        control = app.control()
        control.bind_instance("offline-gpu")
        owner = control.gpu()
        device = GpuWorkerDevice("gpu:0", 0, "isolated", "fake GPU", "discrete", 1024, 512)
        phase = "unsupported"

        async def discover_gpu(*, force):
            assert force is False
            return LocalASRProviderRuntimeSnapshot(
                channels=(),
                gpu=ProviderRuntimeGpuSnapshot(
                    phase=phase,
                    devices=(device,) if phase == "available" else (),
                    active_channels=frozenset(),
                    pending_count=0,
                    worker_pid=None,
                    configured_device_id=None,
                    model_resident=False,
                    retry_required=False,
                    failure_code="isolated_failure" if phase == "failed" else None,
                ),
            )

        owner.runtime_provider = lambda: SimpleNamespace(discover_gpu=discover_gpu)
        for phase, expected in (
            ("unsupported", "action_required"),
            ("failed", "failed"),
            ("available", "applied"),
        ):
            accepted = await control.submit("gpu.discover", {}, request_id=f"gpu-{phase}")
            receipt = await control.wait(accepted["operation_id"], timeout=5)
            assert receipt["status"] == expected
            assert receipt["gpu"]["discovery_failed"] is (phase != "available")
            queried = await control.query("gpu.status", {})
            assert queried["state"]["discovery_failed"] is (phase != "available")
    finally:
        await app.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "model", "connection", "source", "should_stop"),
    [
        ("qq", "local_llm", "ollama", "byok", False),
        ("discord", "gemma4_26b_31b", "openrouter", "byok", False),
        ("qq", "deepseek_v4_flash", "managed", "managed", False),
        ("discord", "deepseek_v4_flash", "managed_china", "managed", False),
        ("qq", "deepseek_v4_flash", "managed_china", "managed", True),
        ("discord", "gemma4_26b_31b", "managed", "managed", True),
        ("openrouter", "gemma4_26b_31b", "openrouter", "byok", True),
    ],
)
async def test_logout_affects_only_selected_credential_route(
    offline,
    provider,
    model,
    connection,
    source,
    should_stop,
):
    app = compose_headless_application(offline)
    control = app.control()
    control.bind_instance("offline-logout")
    base = AppSettingsVNext()
    settings = replace(
        base,
        intent=replace(
            base.intent,
            translation=replace(
                base.intent.translation,
                model=model,
                connection=connection,
                openrouter_selected_source=source,
            ),
        ),
    )
    state = SimpleNamespace(translation_enabled=True)
    effects = []

    async def enabled(value):
        effects.append(("translation", value))
        state.translation_enabled = value

    async def logout(_provider):
        effects.append(("logout", _provider))
        return {"status": "applied", "provider": _provider}

    async def secret(_key, _value):
        effects.append(("secret", _key))
        return True

    async def rebuild(**kwargs):
        effects.append(("rebuild", kwargs))
        return True

    control.application = SimpleNamespace(
        compatibility_settings=lambda: settings,
        state=lambda: state,
        set_translation_enabled=enabled,
        logout_local_managed=logout,
        persist_provider_secret_change=secret,
        apply_providers=rebuild,
    )
    control.sync_ui = lambda: None
    result = await control._logout({"provider": provider})
    assert result["status"] == "applied"
    assert state.translation_enabled is (not should_stop)
    assert (("translation", False) in effects) is should_stop
    assert any(effect[0] == "rebuild" for effect in effects) is should_stop


@pytest.mark.asyncio
async def test_logout_restores_active_translation_after_failed_persistence_and_honors_pending(
    offline,
):
    app = compose_headless_application(offline)
    control = app.control()
    control.bind_instance("offline-logout-failure")
    base = AppSettingsVNext()
    settings = replace(
        base,
        intent=replace(
            base.intent,
            translation=replace(
                base.intent.translation,
                model="deepseek_v4_flash",
                connection="managed_china",
                openrouter_selected_source="managed",
            ),
        ),
    )
    state = SimpleNamespace(translation_enabled=True)
    effects = []

    async def enabled(value):
        effects.append(value)
        state.translation_enabled = value

    async def logout(_provider):
        return {"status": "persistence_failed", "provider": _provider}

    control.application = SimpleNamespace(
        compatibility_settings=lambda: settings,
        state=lambda: state,
        set_translation_enabled=enabled,
        logout_local_managed=logout,
    )
    control.sync_ui = lambda: None
    result = await control._logout({"provider": "qq"})
    assert result["status"] == "persistence_failed"
    assert effects == [False, True]
    assert state.translation_enabled is True

    pending = replace(
        settings.state.managed_connection,
        pending_delivery_ack_source="qq",
        pending_delivery_ack_delivery_id="offline-delivery",
        pending_delivery_ack_managed_credential_ref="offline-reference",
    )
    settings = replace(settings, state=replace(settings.state, managed_connection=pending))
    effects.clear()
    result = await control._logout({"provider": "qq"})
    assert result == {
        "status": "action_required",
        "action": "resolve_pending_managed_authorization",
    }
    assert effects == []

    settings = base
    result = await control._logout({"provider": "qq"})
    assert result["status"] == "persistence_failed"
    assert effects == []


@pytest.mark.asyncio
async def test_managed_logout_preserves_other_account_metadata_and_compensates_failed_save(
    offline, monkeypatch
):
    from puripuly_heart.core.openrouter_credentials import (
        OPENROUTER_MANAGED_API_KEY_SECRET,
        OPENROUTER_MANAGED_QQ_API_KEY_SECRET,
    )

    from puripuly_heart.app.wiring.wiring_managed_auth_factory import ManagedAuthRuntimeAdapter

    app = compose_headless_application(offline)
    owner = app._settings.settings
    owner.start()
    original = owner.canonical
    identity = replace(
        original.state.managed_connection,
        active_managed_credential_ref="discord-reference",
        active_managed_expires_at="2030-01-01",
        founder_letter_seen_credential_ref="discord-reference",
    )
    owner.canonical = replace(original, state=replace(original.state, managed_connection=identity))
    owner.persist_current()
    values = {
        OPENROUTER_MANAGED_API_KEY_SECRET: "offline-discord",
        OPENROUTER_MANAGED_QQ_API_KEY_SECRET: "offline-qq",
    }

    class MemorySecrets:
        def get(self, key):
            return values.get(key)

        def set(self, key, value):
            values[key] = value

        def delete(self, key):
            values.pop(key, None)

    adapter = ManagedAuthRuntimeAdapter(
        config_path=offline,
        secret_store_factory=lambda *_args, **_kwargs: MemorySecrets(),
        settings=owner,
        release_service_provider=lambda: None,
        settings_repository_factory=lambda **_kwargs: None,
        runtime_presence_provider=lambda: (False, False),
        ingress_provider=lambda: False,
    )
    assert (await adapter.logout_local_managed("qq"))["status"] == "applied"
    assert (
        owner.canonical.state.managed_connection.active_managed_credential_ref
        == "discord-reference"
    )
    assert (
        owner.canonical.state.managed_connection.founder_letter_seen_credential_ref
        == "discord-reference"
    )
    assert OPENROUTER_MANAGED_QQ_API_KEY_SECRET not in values
    assert values[OPENROUTER_MANAGED_API_KEY_SECRET] == "offline-discord"
    assert (await adapter.logout_local_managed("qq"))["reason"] == "not_authorized"

    def locked_file(_path, _settings):
        raise PermissionError("controlled persistence failure")

    monkeypatch.setattr(owner.persistence, "persist", locked_file)
    assert (await adapter.logout_local_managed("discord"))["status"] == "persistence_failed"
    assert values[OPENROUTER_MANAGED_API_KEY_SECRET] == "offline-discord"
    assert (
        owner.canonical.state.managed_connection.active_managed_credential_ref
        == "discord-reference"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["timeout", "proxy", "tls", "dns", "network"])
async def test_qq_network_recovery_login_remains_action_required_without_reassert(offline, kind):
    import socket
    import ssl

    import httpx
    from puripuly_heart.app.services.managed_auth import (
        ManagedAuthExecutionResult,
        ManagedAuthOwner,
        ManagedAuthState,
    )

    from puripuly_heart.core import messages
    from tests.app.test_qq_managed_auth import (
        RAW_QQ_CREDENTIAL,
        RAW_QQ_IDENTITY,
        _request,
        _service,
    )

    cause = {
        "timeout": httpx.ReadTimeout("private-timeout"),
        "proxy": httpx.ProxyError("http://private:password@proxy"),
        "tls": ssl.SSLCertVerificationError("private-certificate"),
        "dns": socket.gaierror(-2, "private-host"),
        "network": httpx.ConnectError("private-network"),
    }[kind]
    service, broker, store, state = _service(cause)
    app = compose_headless_application(offline)
    control = app.control()

    async def execute_qq(_identity, _credential, _referral_id):
        transaction = await service.authenticate(_request())
        return ManagedAuthExecutionResult(
            succeeded=False,
            transaction_result=transaction,
            delivery_ack_pending=(
                transaction.status == messages.TRANSACTION_STATUS_REMOTE_DELIVERY_ACK_PENDING
            ),
            message_key=transaction.message.key,
        )

    async def unexpected(*_args, **_kwargs):
        raise AssertionError("QQ recovery must not start another auth flow or rebuild")

    owner = ManagedAuthOwner(
        state_provider=lambda: ManagedAuthState(
            settings_available=True,
            managed_selected=True,
            managed_china=True,
            local_key_available=False,
            release_service_available=True,
            runtime_available=False,
            ingress_frozen=False,
        ),
        pending_sink=lambda _pending: None,
        qq_executor=execute_qq,
        discord_executor=unexpected,
        runtime_ensurer=unexpected,
        usage_view_sink=lambda *_args: None,
        usage_refresh_sink=lambda: None,
        message_sink=lambda *_args: None,
        result_sink=control.results.set,
        log_sink=lambda _message: None,
    )
    control.application = SimpleNamespace(start_qq_managed_auth_from_dialog=owner.start_qq)

    result = await control._login(
        {"provider": "qq", "qq_identity": RAW_QQ_IDENTITY, "credential": RAW_QQ_CREDENTIAL},
        "network-recovery-login",
    )

    assert result["status"] == "action_required"
    assert result["reason"] == "resolve_pending_authorization"
    assert result["provider"] == "qq"
    assert (
        result["transaction"]["status"] == messages.TRANSACTION_STATUS_REMOTE_DELIVERY_ACK_PENDING
    )
    assert result["transaction"]["message"] == f"qq_managed_auth.network_{kind}"
    assert state.pending_managed_operation_id is not None
    assert store.values["openrouter_managed_operation_resume_token"]
    assert len(broker.requests) == 1
    assert broker.resume_requests == []
    assert broker.ack_requests == []
    for private in (RAW_QQ_IDENTITY, RAW_QQ_CREDENTIAL, "private", "password"):
        assert private not in json.dumps(result)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "returned", "failure_kind", "pending", "expected_status", "expected_reason"),
    [
        (
            "qq",
            ("qq_auth.error.credential_mismatch", {}),
            None,
            False,
            "rejected",
            "authorization_failed",
        ),
        ("discord", True, None, False, "applied", None),
        ("discord", False, "recovering", True, "action_required", "authorization_failed"),
        ("discord", False, "failed", True, "failed", "authorization_failed"),
        (
            "discord",
            ("discord_auth.error.network_timeout", {}),
            None,
            True,
            "rejected",
            "authorization_failed",
        ),
        ("openrouter", True, None, False, "applied", None),
        (
            "openrouter",
            ("openrouter.error.recovery_pending", {}),
            None,
            True,
            "action_required",
            "resolve_pending_authorization",
        ),
        (
            "openrouter",
            ("openrouter.error.retry", {}),
            None,
            True,
            "rejected",
            "authorization_failed",
        ),
    ],
)
async def test_auth_login_keeps_other_existing_outcomes(
    offline, provider, returned, failure_kind, pending, expected_status, expected_reason
):
    from puripuly_heart.core import messages

    app = compose_headless_application(offline)
    control = app.control()

    async def login(**_kwargs):
        if pending:
            control.results.set(
                messages.TransactionResult(
                    status=messages.TRANSACTION_STATUS_REMOTE_DELIVERY_ACK_PENDING,
                    message=None,
                    diagnostics=None,
                )
            )
        return returned

    control.application = SimpleNamespace(
        start_qq_managed_auth_from_dialog=login,
        start_discord_managed_auth_from_dialog=login,
        build_managed_openrouter_byok_target=lambda: "selected-target",
        connect_openrouter_via_pkce=login,
        managed_auth_last_failure_kind=lambda: failure_kind,
    )
    result = await control._login(
        {
            "provider": provider,
            "qq_identity": "synthetic-identity",
            "credential": "synthetic-credential",
        },
        "existing-login-outcome",
    )

    assert result["status"] == expected_status
    assert result.get("reason") == expected_reason
    assert result["provider"] == provider


@pytest.mark.asyncio
async def test_settings_failure_is_joinable_after_event_retention_and_state_inspection(
    offline, monkeypatch
):
    from uuid import uuid4

    monkeypatch.setenv("LOCALAPPDATA", str(offline.parent / "isolated-diagnostics-data"))
    app = compose_headless_application(offline)
    try:
        await app.start()
        control = app.control()
        control.bind_instance(str(uuid4()))
        before = await control.query("settings.current", {})

        def denied(_self, *_args):
            raise PermissionError(13, "FORBIDDEN_SETTINGS_EXCEPTION_SENTINEL")

        monkeypatch.setattr(type(control.settings.persistence), "persist", denied)
        accepted = await control.submit(
            "settings.apply",
            {"changes": {"locale": "ja"}},
            request_id=uuid4().hex,
        )
        receipt = await control.wait(accepted["operation_id"], timeout=5)
        assert receipt["status"] == "persistence_failed"
        cause = receipt["transaction"]["diagnostics"]
        assert cause["code"] == "settings_save_failed"
        assert cause["operation_id"] == receipt["operation_id"]
        assert cause["instance_id"] == receipt["instance_id"]
        assert (await control.query("settings.current", {}))["revision"] == before["revision"]
        retained = await control.query(
            "diagnostics.history", {"operation_id": receipt["operation_id"]}
        )
        assert any(item["diagnostic_id"] == cause["diagnostic_id"] for item in retained["records"])
        stream = control.subscribe(topics=["logs", "operation"], after=0)
        async with asyncio.timeout(5):
            while True:
                event = await anext(stream)
                if event.get("diagnostic", {}).get("diagnostic_id") == cause["diagnostic_id"]:
                    assert event["diagnostic"] == cause
                    break
        await stream.aclose()
        operation_event = next(
            item
            for item in control.events.history
            if item["topic"] == "operation" and item["operation_id"] == receipt["operation_id"]
        )
        assert operation_event["diagnostic_id"] == cause["diagnostic_id"]
        assert "FORBIDDEN" not in json.dumps(retained)
    finally:
        await app.stop()


@pytest.mark.parametrize("exception_type", [ValueError, TypeError, KeyError])
@pytest.mark.asyncio
async def test_execution_defects_are_not_argument_rejections_and_keep_safe_causes(
    offline, monkeypatch, exception_type
):
    from uuid import uuid4

    monkeypatch.setenv("LOCALAPPDATA", str(offline.parent / "isolated-control-errors"))
    app = compose_headless_application(offline)
    try:
        await app.start()
        control = app.control()
        control.bind_instance(str(uuid4()))
        invalid = await control.submit(
            "translation.set", {"enabled": "wrong"}, request_id=uuid4().hex
        )
        rejected = await control.wait(invalid["operation_id"], timeout=5)
        assert rejected["status"] == "rejected" and rejected["error"]["code"] == "invalid_arguments"

        async def broken(_self, _command, _arguments, **_kwargs):
            try:
                raise PermissionError(13, "FORBIDDEN_NATIVE_SENTINEL")
            except PermissionError as native:
                raise exception_type("FORBIDDEN_INTERNAL_SENTINEL") from native

        monkeypatch.setattr(type(control), "_dispatch", broken)
        accepted = await control.submit(
            "translation.set", {"enabled": False}, request_id=uuid4().hex
        )
        receipt = await control.wait(accepted["operation_id"], timeout=5)
        assert (
            receipt["status"] == "failed"
            and receipt["error"]["code"] == "application_operation_failed"
        )
        record = receipt["diagnostics"]
        assert record["operation_id"] == receipt["operation_id"]
        assert record["stage"] == "dispatch" and record["category"] == "unknown"
        assert record["fields"]["exception_class"] == exception_type.__name__
        assert record["fields"]["cause_class_1"] == "PermissionError"
        assert record["fields"]["errno"] == 13
        assert "FORBIDDEN" not in json.dumps(receipt)
        assert (
            await control.query("diagnostics.history", {"diagnostic_id": record["diagnostic_id"]})
        )["records"] == [record]
    finally:
        await app.stop()


@pytest.mark.asyncio
async def test_runtime_error_report_crosses_composed_event_bridge_with_source_identity(
    offline, monkeypatch
):
    from uuid import uuid4

    from puripuly_heart.core.messages import ErrorDiagnostics, UserErrorReport, UserMessageRef
    from puripuly_heart.domain.events import UIEvent, UIEventType

    monkeypatch.setenv("LOCALAPPDATA", str(offline.parent / "isolated-report-bridge"))
    app = compose_headless_application(offline)
    try:
        await app.start()
        control = app.control()
        control.bind_instance(str(uuid4()))
        turn = uuid4()
        cause = ErrorDiagnostics(
            component="provider.stt",
            operation="receive",
            code="provider_receive_failed",
            category="network",
            visibility="basic",
            content_policy="metadata_only",
            status_code=503,
            retry_after_ms=None,
            fields={"message": "FORBIDDEN_PROVIDER_BODY_SENTINEL", "provider": "soniox"},
            channel="self",
            turn_id=str(turn),
            generation=5,
        )
        report = UserErrorReport(UserMessageRef("stt.failure", {}, "error"), cause)
        stream = control.subscribe(topics=["error"], after=control.events.sequence)
        pending = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        await control.pipeline.ui_events.put(
            UIEvent(
                UIEventType.ERROR, payload=report, channel="self", source="stt", utterance_id=turn
            )
        )
        event = await asyncio.wait_for(pending, 5)
        await stream.aclose()
        record = event["error"]["diagnostics"]
        assert record["diagnostic_id"] == cause.diagnostic_id
        assert record["code"] == "provider_receive_failed"
        assert record["turn_id"] == str(turn) and record["generation"] == 5
        assert record["operation_id"] is None
        assert record["detail"]["state"] == "rejected"
        assert "FORBIDDEN" not in json.dumps(event)
        assert (await control.query("diagnostics.history", {"diagnostic_id": cause.diagnostic_id}))[
            "records"
        ] == [record]
    finally:
        await app.stop()


def test_native_manual_admission_preserves_ids_without_output_ack():
    from uuid import uuid4

    from puripuly_heart.app.services.application_control import ApplicationControlOwner
    from puripuly_heart.core.orchestrator.self_translation_channel import TextSubmissionAdmission

    turn, child = uuid4(), uuid4()
    result = ApplicationControlOwner._text_admission_data(
        TextSubmissionAdmission("admitted", turn, (child,))
    )
    assert result == {
        "status": "applied",
        "admission": "admitted",
        "turn_id": str(turn),
        "child_turn_ids": [str(child)],
        "output_delivery": "unobserved",
    }
    rejected = ApplicationControlOwner._text_admission_data(None)
    assert rejected["status"] == "rejected" and rejected["turn_id"] is None
    with pytest.raises(TypeError):
        ApplicationControlOwner._text_admission_data(True)


def test_unknown_account_observation_preserves_owning_diagnostic():
    from puripuly_heart.app.services.application_control import _chatgpt_status
    from puripuly_heart.core.messages import ErrorDiagnostics

    cause = ErrorDiagnostics(
        component="chatgpt_account",
        operation="read",
        code="chatgpt_secret_store_failed",
        category="auth",
        visibility="basic",
        content_policy="metadata_only",
        status_code=None,
        retry_after_ms=None,
        fields={},
    )
    observed = []

    def retain(diagnostic, **_):
        observed.append(diagnostic)
        return {"diagnostic_id": diagnostic.diagnostic_id}

    app = SimpleNamespace(
        chatgpt_account_snapshot=lambda: SimpleNamespace(
            signed_in=None, in_progress=False, diagnostics=cause
        )
    )
    result = _chatgpt_status(app, retain)
    assert result["signed_in"] is None and result["sign_in_required"] is None
    assert result["diagnostic_id"] == cause.diagnostic_id and observed == [cause]


@pytest.mark.asyncio
async def test_real_peer_fault_queries_project_safe_original_cause(offline):
    from uuid import uuid4

    from puripuly_heart.core.messages import ErrorDiagnostics
    from puripuly_heart.core.peer_capture import (
        PeerCaptureFailureReason,
        PeerCaptureLanguageFacts,
        PeerCaptureSessionConfig,
        PeerCaptureSessionState,
        PeerCaptureTargetIntent,
    )

    app = compose_headless_application(offline)
    try:
        await app.start()
        control = app.control()
        control.bind_instance(str(uuid4()))
        owner = control.pipeline.current.peer_capture
        config = PeerCaptureSessionConfig(
            provider_id="soniox",
            provider_signature=("soniox",),
            runtime_signature=("soniox",),
            capture_signature=("fixture",),
            capture_target=PeerCaptureTargetIntent(kind="default_output_device"),
            language=PeerCaptureLanguageFacts(source_mode="manual", source_language="en"),
            target_sample_rate_hz=16000,
            vad_speech_threshold=0.4,
            vad_hangover_ms=500,
            vad_pre_roll_ms=500,
        )
        cause = ErrorDiagnostics(
            component="capture.peer",
            operation="receive",
            code="capture.peer.provider_failed",
            category="network",
            visibility="diagnostic_only",
            content_policy="metadata_only",
            status_code=999,
            retry_after_ms=-1,
            fields={"errno": 13, "body": "PRIVATE_PEER_FIELD_SENTINEL"},
            channel="PRIVATE_PEER_CHANNEL_SENTINEL",
            generation=-1,
        )
        await owner._fault_current_generation(
            owner.snapshot.generation,
            config=config,
            reason=PeerCaptureFailureReason.PROVIDER_FAILED,
            exception=cause,
        )
        assert owner.snapshot.state is PeerCaptureSessionState.FAULTED
        assert owner.snapshot.failure_diagnostics.diagnostic_id == cause.diagnostic_id
        for query in ("capture.channels", "providers.status", "app.status"):
            result = await control.query(query, {})
            text = json.dumps(result)
            assert cause.diagnostic_id in text and "PRIVATE_PEER" not in text
        retained = (
            await control.query("diagnostics.history", {"diagnostic_id": cause.diagnostic_id})
        )["records"][0]
        assert retained["code"] == cause.code
        assert retained["channel"] is None and retained["generation"] is None
        assert retained["status_code"] is None and retained["retry_after_ms"] is None
        assert retained["detail"]["rejected_fields"] == 5
        self_snapshot = replace(control.pipeline.self_capture.snapshot, failure_diagnostics=cause)
        safe_self = control._snapshot_json(self_snapshot)["failure_diagnostics"]
        assert safe_self["diagnostic_id"] == cause.diagnostic_id
        assert safe_self["detail"]["rejected_fields"] == 5
        assert "PRIVATE_PEER" not in json.dumps(safe_self)
    finally:
        await app.stop()


def test_transaction_receipt_reports_owner_secondary_omissions():
    from puripuly_heart.app.services.application_control import ApplicationControlOwner
    from puripuly_heart.core.messages import ErrorDiagnostics, TransactionResult

    causes = tuple(
        ErrorDiagnostics(
            component="settings_repository",
            operation="save",
            code="settings_save_failed",
            category="transaction",
            visibility="diagnostic_only",
            content_policy="metadata_only",
            status_code=None,
            retry_after_ms=None,
            fields={},
        )
        for _ in range(6)
    )
    transaction = TransactionResult(
        "settings_commit_failed",
        None,
        causes[0],
        secondary_diagnostics=causes,
        secondary_omitted_count=2,
    )
    owner = SimpleNamespace(
        _retain_diagnostic=lambda diagnostic, **_: {"diagnostic_id": diagnostic.diagnostic_id}
    )
    receipt = ApplicationControlOwner._transaction_receipt(owner, transaction, "persistence_failed")
    assert [item["diagnostic_id"] for item in receipt["secondary_diagnostics"]] == [
        cause.diagnostic_id for cause in causes[:4]
    ]
    assert receipt["secondary_omitted_count"] == 4 and receipt["secondary_truncated"] is True


@pytest.mark.parametrize("carrier", ["provider_convergence", "self_runtime", "legacy_diagnostics"])
@pytest.mark.asyncio
async def test_execution_receipt_preserves_approved_native_failure_carrier(
    offline, monkeypatch, carrier
):
    from uuid import uuid4

    from puripuly_heart.app.services.capture.self_capture_application import (
        SelfCaptureRuntimeApplyError,
    )
    from puripuly_heart.app.services.provider.provider_runtime_apply import (
        ProviderRuntimeConvergenceError,
    )
    from puripuly_heart.core.messages import ErrorDiagnostics

    app = compose_headless_application(offline)
    request_id = uuid4().hex
    native = []
    try:
        await app.start()
        control = app.control()
        control.bind_instance(str(uuid4()))

        async def fail_dispatch(_self, _command, _arguments, *, operation_id):
            cause = ErrorDiagnostics(
                component="capture.peer",
                operation="provider_build",
                code="capture.peer.provider_failed",
                category="lifecycle",
                visibility="diagnostic_only",
                content_policy="metadata_only",
                status_code=None,
                retry_after_ms=None,
                fields={"errno": 13},
                channel="peer",
                generation=1,
                operation_id=operation_id,
                request_id=request_id,
            )
            native.append(cause)
            control._retain_diagnostic(cause, disposition="failed")
            if carrier == "provider_convergence":
                error = ProviderRuntimeConvergenceError(
                    "PRIVATE_WRAPPER_MARKER", failure_diagnostics=cause
                )
                error.diagnostics = replace(cause, diagnostic_id=uuid4().hex)
            elif carrier == "self_runtime":
                error = SelfCaptureRuntimeApplyError(
                    "PRIVATE_WRAPPER_MARKER", failure_diagnostics=cause
                )
            else:
                error = RuntimeError("PRIVATE_WRAPPER_MARKER")
                error.failure_diagnostics = "PRIVATE_NON_TYPED_MARKER"
                error.diagnostics = cause
            raise error

        monkeypatch.setattr(type(control), "_dispatch", fail_dispatch)
        admitted = await control.submit(
            "translation.set", {"enabled": False}, request_id=request_id
        )
        receipt = await control.wait(admitted["operation_id"], timeout=5)
        cause = native[0]
        assert (
            receipt["status"] == "failed"
            and receipt["error"]["code"] == "application_operation_failed"
        )
        assert receipt["cause_diagnostics"]["diagnostic_id"] == cause.diagnostic_id
        assert receipt["diagnostics"]["cause_id"] == cause.diagnostic_id
        assert receipt["diagnostics"]["diagnostic_id"] != cause.diagnostic_id
        assert receipt["cause_diagnostics"]["operation_id"] == admitted["operation_id"]
        assert receipt["cause_diagnostics"]["request_id"] == request_id
        assert receipt["cause_diagnostics"]["category"] == "lifecycle"
        assert "PRIVATE_" not in json.dumps(receipt)
        records = (
            await control.query("diagnostics.history", {"operation_id": admitted["operation_id"]})
        )["records"]
        assert receipt["cause_diagnostics"] in records and receipt["diagnostics"] in records
    finally:
        await app.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["cancelled", "failed", "interrupted"])
@pytest.mark.parametrize("command", ["gemma.prepare", "translation.set"])
async def test_gemma_prepare_cancellation_receipt_tracks_settled_provisioning(
    offline, monkeypatch, outcome, command
):
    import socket

    from puripuly_heart.core.local_asr.local_stt_download_port import (
        HuggingFaceDownloadProgress,
        LocalSTTDownloadPortCancelled,
    )
    from puripuly_heart.core.local_translation import assets
    from puripuly_heart.core.local_translation.provisioning import ensure_gemma_installed
    from puripuly_heart.core.local_translation.runtime import ManagedGemmaRuntimeOwner

    monkeypatch.setenv("LOCALAPPDATA", str(offline.parent / "isolated-gemma-data"))
    monkeypatch.setenv("APPDATA", str(offline.parent / "isolated-gemma-roaming"))
    settings = json.loads(offline.read_text(encoding="utf-8"))
    settings["intent"]["translation"].update(model="managed_gemma_26b", connection="gpu")
    settings["intent"]["stt"]["provider"] = "soniox"
    settings["intent"]["peer_stt"]["provider"] = "soniox"
    settings["intent"]["telemetry"]["enabled"] = False
    settings["intent"]["integrated_context"]["enabled"] = False
    sink = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sink.bind(("127.0.0.1", 0))
    settings["intent"]["osc"].update(
        host="127.0.0.1",
        port=sink.getsockname()[1],
        send_port=sink.getsockname()[1],
        activation_notice_enabled=False,
    )
    offline.write_text(json.dumps(settings), encoding="utf-8")
    spec = assets.resolve_gemma_spec(assets.GEMMA_26B_MODEL_ID)
    spec = replace(
        spec,
        assets=(replace(spec.assets[0], size_bytes=100, sha256="0" * 64),),
    )
    monkeypatch.setattr(assets, "gemma_26b_spec", lambda: spec)
    entered = asyncio.Event()
    settled = asyncio.Event()
    staging_paths = []

    class Downloader:
        async def download(self, request, *, cancel_event, on_progress):
            staging_paths.append(request.local_dir)
            (request.local_dir / request.remote_path).write_bytes(b"partial synthetic target")
            on_progress(HuggingFaceDownloadProgress(24, 100))
            entered.set()
            await asyncio.to_thread(cancel_event.wait)
            settled.set()
            if outcome == "failed":
                raise PermissionError(5, "PRIVATE_DOWNLOAD_CLEANUP_FAILURE")
            raise LocalSTTDownloadPortCancelled("synthetic download cancelled")

    def reject_launch(*_args, **_kwargs):
        raise AssertionError("cancelled provisioning must not launch a model process")

    async def provision(**kwargs):
        kwargs["downloader"] = Downloader()
        return await ensure_gemma_installed(**kwargs)

    install_dir = offline.parent / "gemma-models" / spec.install_dirname
    runtime = ManagedGemmaRuntimeOwner(
        install_dir=install_dir,
        provisioner=provision,
        process_factory=reject_launch,
        transport_factory=reject_launch,
    )
    monkeypatch.setattr(
        application_runtime, "create_managed_gemma_runtime", lambda **_kwargs: runtime
    )
    app = compose_headless_application(offline)
    try:
        await app.start()
        control = app.control()
        control.bind_instance(f"isolated-gemma-{outcome}")
        owner = control.gemma()
        arguments = {"enabled": True} if command == "translation.set" else {}
        admitted = await control.submit(command, arguments, request_id="prepare-cancel")
        await asyncio.wait_for(entered.wait(), 5)
        assert owner.snapshot.state == "downloading"
        assert staging_paths and staging_paths[0].is_dir()
        assert (await control.operation(admitted["operation_id"]))["terminal"] is False

        if outcome == "interrupted":
            await asyncio.wait_for(app.stop(), 5)
        else:
            cancellation = await control.submit("gemma.cancel", {}, request_id="cancel-preparation")
            cancel_receipt = await control.wait(cancellation["operation_id"], timeout=5)
            assert cancel_receipt["status"] == "cancelled"
            assert cancel_receipt["terminal"] is True
        receipt = await control.wait(admitted["operation_id"], timeout=5)
        assert receipt["status"] == outcome
        assert receipt["terminal"] is True
        if command == "translation.set" and outcome != "interrupted":
            status = await control.query("app.status", {})
            assert status["translation_enabled"] is False
            assert status["translation_runtime_ready"] is True
            assert (
                control.pipeline.translation_runtime_configuration.snapshot().value.translation_enabled
                is False
            )
        assert settled.is_set()
        assert not install_dir.exists()
        assert all(not path.exists() for path in staging_paths)
        if outcome == "interrupted":
            assert owner.snapshot.state == "closed"
            assert runtime.readiness is None
            assert control.pipeline.llm_runtime is None
        if outcome == "failed":
            assert owner.snapshot.state == "failed"
            assert owner.snapshot.failure_code == "filesystem_failed"
            assert owner.snapshot.cause_type == "PermissionError"
            assert receipt["error"]["code"] == "application_operation_failed"
            assert "diagnostics" in receipt
            assert "PRIVATE_DOWNLOAD_CLEANUP_FAILURE" not in json.dumps(receipt)
        elif outcome == "cancelled":
            assert owner.snapshot.state == "cancelled"
            assert "error" not in receipt
            assert "diagnostics" not in receipt
        if outcome != "interrupted":
            next_operation = await control.submit(
                "settings.apply", {"changes": {"locale": "ja"}}, request_id="after-cancel"
            )
            next_receipt = await control.wait(next_operation["operation_id"], timeout=5)
            assert next_receipt["terminal"] is True
            assert next_receipt["status"] == "degraded"
            assert (
                next_receipt["transaction"]["status"] == "settings_commit_success_runtime_degraded"
            )
            assert (await control.query("settings.current", {}))["settings"]["intent"]["ui"][
                "locale"
            ] == "ja"
    finally:
        await runtime.close()
        await app.stop()
        sink.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "backend", "gpu_only", "failure_phase", "failure_key"),
    [
        ("managed_gemma", "cpu", False, "runtime", "dashboard.managed_gemma_notice.failed"),
        ("managed_gemma", "gpu", False, "runtime", "dashboard.managed_gemma_notice.failed"),
        (
            "managed_gemma_26b",
            "gpu",
            True,
            "runtime",
            "dashboard.managed_gemma_notice.gpu_only_failed",
        ),
        ("managed_gemma_26b", "gpu", True, "download", "dashboard.managed_gemma_notice.failed"),
    ],
)
async def test_composed_gemma_notices_carry_presentation_policy_without_backend_identity(
    offline, monkeypatch, model, backend, gpu_only, failure_phase, failure_key
):
    import socket

    from puripuly_heart.app.ports.ui_models import ManagedGemmaDashboardNotice
    from puripuly_heart.composition import headless_application
    from puripuly_heart.core.local_asr.local_stt_download_port import LocalSTTDownloadPortError
    from puripuly_heart.core.local_translation.provisioning import GemmaProvisioningUpdate
    from puripuly_heart.core.local_translation.runtime import ManagedGemmaRuntimeError
    from puripuly_heart.ui.dashboard.capture_notices import managed_gemma_capture_notice
    from puripuly_heart.ui.i18n import t

    monkeypatch.setenv("LOCALAPPDATA", str(offline.parent / "isolated-notice-data"))
    monkeypatch.setenv("APPDATA", str(offline.parent / "isolated-notice-roaming"))
    settings = json.loads(offline.read_text(encoding="utf-8"))
    settings["intent"]["translation"].update(model=model, connection=backend)
    settings["intent"]["stt"]["provider"] = "soniox"
    settings["intent"]["peer_stt"]["provider"] = "soniox"
    settings["intent"]["telemetry"]["enabled"] = False
    settings["intent"]["integrated_context"]["enabled"] = False
    sink = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sink.bind(("127.0.0.1", 0))
    settings["intent"]["osc"].update(
        host="127.0.0.1",
        port=sink.getsockname()[1],
        send_port=sink.getsockname()[1],
        activation_notice_enabled=False,
    )
    offline.write_text(json.dumps(settings), encoding="utf-8")
    presentation = headless_application.HeadlessApplicationPresentation()
    monkeypatch.setattr(
        headless_application, "HeadlessApplicationPresentation", lambda: presentation
    )
    progress_notices = []
    terminal_error = (
        ManagedGemmaRuntimeError("controlled runtime failure")
        if failure_phase == "runtime"
        else LocalSTTDownloadPortError("controlled download failure")
    )

    class Runtime:
        async def prepare(self, **kwargs):
            kwargs["provision_kwargs"]["on_status"](
                GemmaProvisioningUpdate("downloading", 25, 100, 25)
            )
            progress_notices.append(presentation.projections["managed_gemma"])
            raise terminal_error

        async def release(self):
            return None

        async def close(self):
            return None

    monkeypatch.setattr(
        application_runtime, "create_managed_gemma_runtime", lambda **_kwargs: Runtime()
    )
    app = compose_headless_application(offline)
    try:
        await app.start()
        control = app.control()
        owner = control.gemma()
        with pytest.raises(type(terminal_error)):
            await owner.prepare(control.gemma_selection(app.compatibility_settings()))
        progress = progress_notices[-1]
        assert isinstance(progress, ManagedGemmaDashboardNotice)
        assert progress.gpu_only is gpu_only
        assert progress.progress_percent == 25
        assert progress.action == "cancel"
        assert managed_gemma_capture_notice(progress).text == t(
            "dashboard.managed_gemma_notice.downloading", percent=25
        )
        failure = presentation.projections["managed_gemma"]
        assert isinstance(failure, ManagedGemmaDashboardNotice)
        assert failure.gpu_only is gpu_only
        assert failure.failure_phase == failure_phase
        assert failure.action is None
        rendered = managed_gemma_capture_notice(failure)
        assert rendered.tone == "error"
        assert rendered.text == t(failure_key)
        if gpu_only and failure_phase == "runtime":
            assert "No CPU or cloud fallback" in rendered.text
    finally:
        await app.stop()
        sink.close()


@pytest.mark.asyncio
async def test_failed_26b_enable_stays_off_without_implicit_retry_and_recovers_explicitly(
    offline, monkeypatch
):
    import hashlib

    from puripuly_heart.composition import headless_application
    from puripuly_heart.core.local_asr.local_stt_download_port import HuggingFaceDownloadProgress
    from puripuly_heart.core.local_translation import assets
    from puripuly_heart.core.local_translation.provisioning import ensure_gemma_installed
    from puripuly_heart.core.local_translation.runtime import (
        ManagedGemmaMetrics,
        ManagedGemmaResponse,
        ManagedGemmaRuntimeOwner,
    )
    from puripuly_heart.core.local_translation.runtime_profile import GemmaRuntimePaths

    monkeypatch.setenv("LOCALAPPDATA", str(offline.parent / "isolated-enable-data"))
    monkeypatch.setenv("APPDATA", str(offline.parent / "isolated-enable-roaming"))
    settings = json.loads(offline.read_text(encoding="utf-8"))
    settings["intent"]["translation"].update(model="managed_gemma_26b", connection="gpu")
    settings["intent"]["stt"]["provider"] = "soniox"
    settings["intent"]["peer_stt"]["provider"] = "soniox"
    settings["intent"]["telemetry"]["enabled"] = False
    settings["intent"]["integrated_context"]["enabled"] = False
    offline.write_text(json.dumps(settings), encoding="utf-8")
    presentation = headless_application.HeadlessApplicationPresentation()
    monkeypatch.setattr(
        headless_application, "HeadlessApplicationPresentation", lambda: presentation
    )
    content = b"synthetic 26B asset"
    spec = assets.resolve_gemma_spec(assets.GEMMA_26B_MODEL_ID)
    spec = replace(
        spec,
        assets=(
            replace(
                spec.assets[0],
                size_bytes=len(content),
                sha256=hashlib.sha256(content).hexdigest(),
            ),
        ),
    )
    monkeypatch.setattr(assets, "gemma_26b_spec", lambda: spec)
    downloads = []

    class Downloader:
        async def download(self, request, *, cancel_event, on_progress):
            downloads.append(request.remote_path)
            destination = request.local_dir / request.remote_path
            destination.write_bytes(content)
            on_progress(HuggingFaceDownloadProgress(len(content), len(content)))
            return destination

    async def provision(**kwargs):
        kwargs["downloader"] = Downloader()
        return await ensure_gemma_installed(**kwargs)

    class Process:
        def __init__(self):
            self.returncode = None
            self.exited = asyncio.Event()

        def terminate(self):
            self.returncode = 0
            self.exited.set()

        def kill(self):
            self.returncode = -9
            self.exited.set()

        async def wait(self):
            await self.exited.wait()
            return self.returncode

    commands = []
    processes = []
    translations = []
    prefixes = []
    closed = []
    failure = OSError(5, "PRIVATE_CONTROLLED_VULKAN_FAILURE")

    async def spawn(command, _cwd):
        commands.append(command)
        if len(commands) == 1:
            raise failure
        process = Process()
        processes.append(process)
        return process

    class Transport:
        async def wait_until_ready(self, *, timeout_s):
            assert timeout_s > 0

        async def prepare_prefix(self, *, system_prompt, slot_id):
            prefixes.append((system_prompt, slot_id))

        async def wait_slot_idle(self, *, slot_id):
            assert slot_id in range(3)

        async def translate(self, **kwargs):
            translations.append(kwargs)
            return ManagedGemmaResponse(
                text="controlled recovered translation",
                metrics=ManagedGemmaMetrics(24, 20, 4, 5.0, 100.0, 40.0),
            )

        async def close(self):
            closed.append(True)

    executable = offline.parent / "controlled-llama-server.exe"
    executable.write_bytes(b"never executed")

    async def validate_version(path):
        assert path == executable

    runtime = ManagedGemmaRuntimeOwner(
        install_dir=offline.parent / "models" / spec.install_dirname,
        runtime_paths=GemmaRuntimePaths(executable, executable),
        provisioner=provision,
        process_factory=spawn,
        transport_factory=lambda _url: Transport(),
        version_validator=validate_version,
    )
    monkeypatch.setattr(
        application_runtime, "create_managed_gemma_runtime", lambda **_kwargs: runtime
    )
    app = compose_headless_application(offline)
    try:
        await app.start()
        control = app.control()
        control.bind_instance("controlled-failed-26b-enable")
        before = await control.query("app.status", {})
        assert before["translation_enabled"] is False
        assert before["translation_runtime_ready"] is True
        presentation.set_dashboard_translation_enabled(True)
        enabled = await control.submit(
            "translation.set", {"enabled": True}, request_id="first-enable"
        )
        receipt = await control.wait(enabled["operation_id"], timeout=5)
        assert receipt["terminal"] is True
        assert receipt["status"] == "failed"
        assert receipt["error"]["code"] == "application_operation_failed"
        assert "PRIVATE_" not in json.dumps(receipt)
        status = await control.query("app.status", {})
        assert status["translation_enabled"] is False
        assert status["translation_runtime_ready"] is True
        assert (
            control.pipeline.translation_runtime_configuration.snapshot().value.translation_enabled
            is False
        )
        assert presentation.dashboard_translation_enabled() is False
        assert presentation.projections["translation_starting"] is False
        snapshot = (await control.query("models.status", {}))["gemma"]
        assert snapshot["state"] == "failed"
        assert snapshot["model_id"] == spec.model_id
        assert snapshot["backend"] == "gpu"
        assert snapshot["failure_code"] == "filesystem_failed"
        assert snapshot["cause_type"] == "OSError"
        assert runtime.readiness is None
        assert len(commands) == 1
        assert not processes

        disabled_text = await control.submit(
            "text.submit", {"text": "synthetic input while disabled"}, request_id="off-text"
        )
        disabled_receipt = await control.wait(disabled_text["operation_id"], timeout=5)
        assert disabled_receipt["terminal"] is True
        assert disabled_receipt["admission"] == "admitted"
        await asyncio.wait_for(control.pipeline.translation_turns.wait_for_idle(), 5)
        assert len(commands) == 1
        assert translations == []
        assert (await control.query("models.status", {}))["gemma"] == snapshot

        recovery = await control.submit(
            "translation.set", {"enabled": True}, request_id="explicit-retry"
        )
        recovered = await control.wait(recovery["operation_id"], timeout=5)
        assert recovered["terminal"] is True
        assert recovered["status"] == "applied"
        assert (await control.query("app.status", {}))["translation_enabled"] is True
        ready = (await control.query("models.status", {}))["gemma"]
        assert ready["state"] == "ready"
        assert ready["model_id"] == snapshot["model_id"]
        assert ready["backend"] == "gpu"
        assert runtime.readiness.effective_backend == "vulkan"
        assert len(commands) == 2
        assert len(processes) == 1
        assert downloads == [spec.assets[0].filename]
        for command in commands:
            assert command[command.index("--alias") + 1] == "puripuly-gemma-4-26b-a4b-q4"
            assert command[command.index("--device") + 1].startswith("Vulkan")
            assert command[command.index("--model") + 1].endswith(spec.assets[0].filename)
        selected = (await control.query("settings.current", {}))["settings"]["intent"][
            "translation"
        ]
        assert selected["model"] == "managed_gemma_26b"
        assert selected["connection"] == "gpu"
        stream = control.subscribe(topics=["translation"], include_translations=True)

        async def translated_event():
            async for event in stream:
                return event

        observed = asyncio.create_task(translated_event())
        await asyncio.sleep(0)
        text = await control.submit(
            "text.submit", {"text": "synthetic recovered input"}, request_id="on-text"
        )
        text_receipt = await control.wait(text["operation_id"], timeout=5)
        assert text_receipt["admission"] == "admitted"
        event = await asyncio.wait_for(observed, 5)
        await stream.aclose()
        await asyncio.wait_for(control.pipeline.translation_turns.wait_for_idle(), 5)
        assert event["utterance_id"] in [text_receipt["turn_id"], *text_receipt["child_turn_ids"]]
        assert event["channel"] == "self"
        assert event["text"] == "controlled recovered translation"
        assert len(translations) == 1
        assert prefixes
        assert len(commands) == 2
    finally:
        await app.stop()
        await runtime.close()
    assert all(process.returncode == 0 for process in processes)
    assert closed == [True]
