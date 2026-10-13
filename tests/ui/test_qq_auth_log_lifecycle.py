from __future__ import annotations

import logging
from dataclasses import replace
from uuid import uuid4

import httpx
import pytest

from puripuly_heart.core.openrouter.managed_openrouter_broker_client import (
    HttpManagedOpenRouterBrokerClient,
)
from puripuly_heart.core.runtime_logging import (
    RealtimeLogHandler,
    SessionRuntimeLoggingService,
    configure_main_logging,
)
from puripuly_heart.ui.components.qq_managed_auth_dialog import QqManagedAuthDialog
from puripuly_heart.ui.i18n import get_locale, set_locale
from puripuly_heart.ui.views.logs import LogsView
from tests.app.test_qq_managed_auth import (
    RAW_MANAGED_KEY,
    RAW_QQ_CREDENTIAL,
    RAW_QQ_IDENTITY,
    _failure_result,
    _request,
    _service,
)
from tests.helpers.flet_page import DialogTrackingPage


@pytest.mark.asyncio
async def test_qq_failure_survives_dialog_close_log_page_reopen_and_file_drain(tmp_path):
    previous_locale = get_locale()
    set_locale("zh-CN")
    logger = logging.getLogger()
    previous_handlers = list(logger.handlers)
    previous_level = logger.level
    sinks = configure_main_logging(root_logger=logger, log_dir=tmp_path)
    logging_service = SessionRuntimeLoggingService(
        root_logger=logger,
        session_logger=logging.getLogger(f"test.qq.{uuid4()}"),
        sinks=sinks,
        ui_handler_factory=RealtimeLogHandler,
    )
    logging_service.configure_diagnostics(instance_id=uuid4().hex)
    view = LogsView()
    logging_service.attach_realtime_sink(view)

    def respond(request):
        if request.url.path.endswith("/status"):
            return httpx.Response(404, json={"ok": False})
        return httpx.Response(
            401,
            json={
                "error": {
                    "code": "invalid_request",
                    "class": "security_fail",
                    "subcode": "qq_credential_invalid",
                    "message": RAW_QQ_CREDENTIAL + RAW_MANAGED_KEY,
                }
            },
        )

    client = HttpManagedOpenRouterBrokerClient(
        base_url="https://mock.invalid", transport=httpx.MockTransport(respond)
    )
    try:
        service, broker, store, state = _service(_failure_result("invalid_credential"))
        service = replace(service, broker_client=client)
        result = await service.authenticate(_request())
        page = DialogTrackingPage()
        dialog = QqManagedAuthDialog(page, on_continue=lambda: None, on_close=lambda: None)
        dialog.open()
        dialog.set_error("qq_auth.error.credential_mismatch")
        assert "凭证不正确" in dialog._error_text.value
        dialog.close()
        assert not dialog._is_open
        view._flush_logs()
        rendered = view._log_text.value
        assert rendered
        records = logging_service.diagnostic_history({})["records"]
        primary = [
            record
            for record in records
            if record["diagnostic_id"] == result.diagnostics.diagnostic_id
        ]
        assert len(primary) == 1
        assert primary[0]["status_code"] == 401
        assert result.diagnostics.visibility == "basic"
        assert primary[0]["occurred_at"] == result.diagnostics.occurred_at
        logging_service.detach_realtime_sink()
        view.will_unmount()
        logging_service.attach_realtime_sink(view)
        view._flush_logs()
        assert view._log_text.value == rendered
        assert result.message.key == "qq_managed_auth.invalid_credential"
    finally:
        await client.close()
        logging_service.detach_realtime_sink()
        logging_service.close_terminal_owner()
        for handler in tuple(logger.handlers):
            if handler not in previous_handlers:
                logger.removeHandler(handler)
        logger.setLevel(previous_level)
        set_locale(previous_locale)
    persisted = sinks.log_file.read_text(encoding="utf-8")
    assert persisted
    for sensitive in (RAW_QQ_CREDENTIAL, RAW_MANAGED_KEY, RAW_QQ_IDENTITY):
        assert sensitive not in persisted
