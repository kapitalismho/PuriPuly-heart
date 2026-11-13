from __future__ import annotations

import errno
import json
import sys
from types import SimpleNamespace

import pytest

from puripuly_heart.core.messages import ErrorDiagnostics


def test_audio_partial_observation_and_cleanup_retain_safe_causes_despite_sink_failure(monkeypatch):
    from puripuly_heart.ui.components.settings import audio_settings as module

    class Manager:
        def get_loopback_device_info_generator(self):
            yield {"name": "Synthetic output"}
            raise PermissionError(errno.EACCES, "private device details")

        def terminate(self):
            raise OSError(errno.EIO, "private cleanup path")

    monkeypatch.setitem(sys.modules, "pyaudiowpatch", SimpleNamespace(PyAudio=Manager))

    def broken_sink(*args, **kwargs):
        raise RuntimeError("private sink failure")

    monkeypatch.setattr(module.logger, "log", broken_sink)
    settings = module.AudioSettings()
    options = settings._get_desktop_output_options()
    assert [item.value for item in options] == ["", "Synthetic output"]
    failures = settings.enumeration_diagnostics
    assert [item.code for item in failures] == [
        "audio_device_enumeration_failed",
        "audio_device_cleanup_failed",
    ]
    assert [item.fields["errno"] for item in failures] == [errno.EACCES, errno.EIO]
    assert "private" not in repr(failures)

    monkeypatch.setitem(
        sys.modules,
        "pyaudiowpatch",
        SimpleNamespace(
            PyAudio=lambda: SimpleNamespace(
                get_loopback_device_info_generator=lambda: (), terminate=lambda: None
            )
        ),
    )
    assert [item.value for item in settings._get_desktop_output_options()] == [""]
    assert settings.enumeration_diagnostics == ()


def test_localization_recovery_retains_resource_and_format_failures_without_params(
    monkeypatch, tmp_path
):
    from puripuly_heart.ui import i18n

    (tmp_path / "en.json").write_text(json.dumps({"greeting": "Hello {name}"}), encoding="utf-8")
    (tmp_path / "ja.json").write_text("invalid private bundle content", encoding="utf-8")
    monkeypatch.setattr(i18n, "_bundles", {})
    monkeypatch.setattr(i18n, "_failures", {})
    monkeypatch.setattr(i18n, "_locale_cache", None)
    monkeypatch.setattr(i18n, "_current_locale", "ja")
    monkeypatch.setattr(
        i18n,
        "current_runtime_layout",
        lambda: SimpleNamespace(
            package_resource=lambda directory, *parts: tmp_path.joinpath(*parts)
        ),
    )
    assert i18n.t("greeting", name="Synthetic") == "Hello Synthetic"
    assert i18n.t("greeting", secret="private format value") == "Hello {name}"
    assert i18n.t_for_locale("ja", "greeting", secret="private format value") == "Hello {name}"
    failures = i18n.localization_diagnostics()
    assert [item.code for item in failures] == [
        "localization_resource_failed",
        "localization_format_failed",
    ]
    assert failures[1].fields["exception_class"] == "KeyError"
    assert "private" not in repr(failures)
    assert i18n.available_locales() == ("en", "ja")


@pytest.mark.asyncio
async def test_model_dialog_consumes_owned_failure_without_recording_it_again(monkeypatch):
    from puripuly_heart.app.services.model_assets import ModelAssetsError
    from puripuly_heart.ui.components import model_assets_dialog as module

    diagnostics = ErrorDiagnostics(
        component="model_assets",
        operation="download",
        code="download_failed",
        category="network",
        visibility="basic",
        content_policy="metadata_only",
        status_code=None,
        retry_after_ms=None,
        fields={},
    )
    failure = ModelAssetsError("private source path", failure_code="download_failed")
    failure.failure_diagnostics = diagnostics
    tasks = []
    records = []
    monkeypatch.setattr(module.logger, "log", lambda *args, **kwargs: records.append(args))
    dialog = module.ModelAssetsDialog.__new__(module.ModelAssetsDialog)
    dialog.run_task = tasks.append
    dialog.render = lambda: None
    dialog._is_open = False
    dialog.failure_diagnostics = None

    async def fail():
        raise failure

    dialog._run(fail)
    await tasks.pop()()
    assert dialog.failure_diagnostics is diagnostics
    assert records == []
    assert "private" not in repr(dialog.failure_diagnostics)

    cancelled = ModelAssetsError("cancelled", failure_code="cancelled")
    dialog.failure_diagnostics = None
    dialog._retain_failure("asset_operation", cancelled)
    assert dialog.failure_diagnostics is None


@pytest.mark.asyncio
async def test_update_unavailable_retains_owned_evidence_without_notification(monkeypatch):
    from puripuly_heart.ui import app

    diagnostics = ErrorDiagnostics(
        component="updater",
        operation="update_check",
        code="update_check_failed",
        category="network",
        visibility="basic",
        content_policy="metadata_only",
        status_code=None,
        retry_after_ms=None,
        fields={},
    )

    class Unavailable(RuntimeError):
        failure_diagnostics = diagnostics

    async def load():
        raise Unavailable("private network details")

    shown = []
    retained = await app._check_and_notify_update(
        SimpleNamespace(show_dialog=shown.append), load_update_info=load
    )
    assert retained is diagnostics
    assert shown == []


def test_launcher_preserves_original_when_logging_close_fails(monkeypatch, tmp_path):
    from puripuly_heart import main

    original = PermissionError(errno.EACCES, "private startup path")

    def launch(*args, **kwargs):
        raise original

    def close(**kwargs):
        raise OSError(errno.EIO, "private sink path")

    monkeypatch.setattr(
        main, "configure_main_logging", lambda **kwargs: SimpleNamespace(close=close)
    )
    monkeypatch.setattr(main, "_run_gui", launch)
    with pytest.raises(PermissionError) as caught:
        main.main(["--config", str(tmp_path / "settings.json"), "run-gui"])
    assert caught.value is original
    diagnostics = original.failure_diagnostics
    assert diagnostics.fields["errno"] == errno.EACCES
    assert diagnostics.causes[-1].code == "gui_cleanup_failed"
    assert diagnostics.causes[-1].fields["errno"] == errno.EIO
    assert "private" not in repr(diagnostics)


def test_error_presented_observer_failure_is_secondary_and_original_survives_failed_sink(
    monkeypatch,
):
    from puripuly_heart.core.diagnostics import exception_diagnostics
    from puripuly_heart.core.messages import UserErrorReport, UserMessageRef
    from puripuly_heart.ui import event_dispatch

    original = exception_diagnostics(
        PermissionError(13, "private original content"),
        component="ui_application",
        operation="discord_auth",
        code="ui_boundary_failed",
    )
    report = UserErrorReport(
        message=UserMessageRef(key="error.unknown", params={}, severity="error"),
        diagnostics=original,
    )

    def fail(*args, **kwargs):
        raise OSError(5, "private observer path")

    monkeypatch.setattr(event_dispatch.logger, "log", fail)
    bridge = event_dispatch.UIEventBridge.__new__(event_dispatch.UIEventBridge)
    bridge._error_presented_callback = fail
    bridge.presentation_observer_failure = None
    bridge._notify_error_presented("private localized text", report)
    assert report.diagnostics is original
    secondary = bridge.presentation_observer_failure
    assert secondary.cause_id == original.diagnostic_id
    assert secondary.operation == "error_presented"
    assert secondary.fields["errno"] == 5
    assert "private" not in repr(secondary)
