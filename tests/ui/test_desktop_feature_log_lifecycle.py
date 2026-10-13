import logging
from uuid import uuid4

import pytest

from puripuly_heart.core import feature_diagnostics
from puripuly_heart.core.runtime_logging import (
    RealtimeLogHandler,
    SessionRuntimeLoggingService,
    configure_main_logging,
)
from puripuly_heart.ui.components.microphone_test_dialog import MicrophoneTestDialog
from puripuly_heart.ui.components.model_assets_dialog import ModelAssetsDialog
from puripuly_heart.ui.i18n import get_locale, set_locale
from puripuly_heart.ui.views.logs import LogsView
from tests.helpers.flet_page import DialogTrackingPage


@pytest.mark.parametrize(
    "module,stage,code",
    [
        ("ModelDownload", "download", "download_timeout"),
        ("ModelDownload", "verifying", "checksum_failed"),
        ("QuickTranslate", "submission", "backend_not_ready"),
        ("QuickTranslate", "translation", "translation_failed"),
        ("AudioTest", "device_open", "source_open_failed"),
        ("AudioTest", "playback", "playback_failed"),
    ],
)
def test_normal_logs_survive_surface_close_reopen_and_file_flush(tmp_path, module, stage, code):
    logger = feature_diagnostics.logger
    previous_handlers, previous_level = list(logger.handlers), logger.level
    sinks = configure_main_logging(root_logger=logger, log_dir=tmp_path)
    logging_service = SessionRuntimeLoggingService(
        root_logger=logger,
        session_logger=logging.getLogger(f"test.preview.{uuid4()}"),
        sinks=sinks,
        ui_handler_factory=RealtimeLogHandler,
    )
    view = LogsView()
    logging_service.attach_realtime_sink(view)
    page = DialogTrackingPage()
    dialog = MicrophoneTestDialog(page, on_close=lambda: None)
    try:
        dialog.open()
        feature_diagnostics.log_feature(
            module,
            stage,
            code,
            error=OSError(
                5,
                "QQ-123456789 credential-secret API-key token-secret https://proxy-user:password@proxy.invalid",
            ),
        )
        dialog.close()
        view._flush_logs()
        assert f"[{module}] failure stage={stage} code={code}" in view._log_text.value
        logging_service.detach_realtime_sink()
        view.will_unmount()
        logging_service.attach_realtime_sink(view)
        view._flush_logs()
        assert f"code={code}" in view._log_text.value
    finally:
        logging_service.detach_realtime_sink()
        sinks.close()
        for handler in tuple(logger.handlers):
            if handler not in previous_handlers:
                logger.removeHandler(handler)
        logger.setLevel(previous_level)
    persisted = sinks.log_file.read_text(encoding="utf-8")
    assert f"[{module}] failure stage={stage} code={code}" in persisted
    assert "[ERROR]" in persisted and "os_code=5" in persisted
    for secret in (
        "123456789",
        "credential-secret",
        "API-key",
        "token-secret",
        "password",
        "proxy-user",
    ):
        assert secret not in persisted


async def test_model_dialog_reports_actual_failure_without_raw_response(caplog):
    from types import SimpleNamespace

    from puripuly_heart.core.local_translation.provisioning import GemmaProvisioningError

    async def configure(source):
        return None

    async def fail(**kwargs):
        raise GemmaProvisioningError(
            "signed-uri-with-secret", failure_code="mirror_redirect_external"
        )

    locale = get_locale()
    set_locale("zh-CN")
    try:
        application = SimpleNamespace(
            selected_model_asset_ids=lambda: ("gemma-4-e4b-it-qat-ud-q4-k-xl",),
            gemma_download_source=lambda: "mirror",
            configure_gemma_source=configure,
            provision_gemma_assets=fail,
        )
        page = DialogTrackingPage()
        page.update = lambda: None
        dialog = ModelAssetsDialog(page, application, lambda f: None)
        dialog.open()
        await dialog.install()
        assert "镜像跳回" in dialog.results["gemma-4-e4b-it-qat-ud-q4-k-xl"].value
        dialog.close()
        assert "code=mirror_redirect_external" in caplog.text
        assert "signed-uri-with-secret" not in caplog.text
    finally:
        set_locale(locale)
