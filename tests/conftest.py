from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_PATH))

_GLOBAL_LOGGER_NAMES = ("", "puripuly_heart", "puripuly_heart.runtime.session")


@pytest.fixture
def memory_osc_transports(monkeypatch):
    from types import SimpleNamespace

    from puripuly_heart.app.ports.oscquery import OscQueryServiceInfo
    from puripuly_heart.app.services.osc.control_runtime import OscControlIntegrationOwner
    from puripuly_heart.app.wiring import wiring_runtime_pipeline
    from puripuly_heart.composition import application_runtime
    from tests.app.test_osc_control_runtime import (
        FakeReceiverOwner,
        FakeSender,
        RetryDiscoveryService,
    )

    receiver = FakeReceiverOwner()
    receiver.runtime = None
    service = RetryDiscoveryService(
        OscQueryServiceInfo(
            service_id="controlled-VRChat",
            host="127.0.0.1",
            osc_send_port=19000,
            is_vrchat=True,
        )
    )
    senders = []

    class MemorySender(FakeSender):
        def __init__(self, **options):
            super().__init__()
            self.host = options["host"]
            self.port = options["port"]
            self.chatbox_address = options["chatbox_address"]
            self.chatbox_send = options["chatbox_send"]
            self.chatbox_clear = options["chatbox_clear"]
            self.chatboxes = []
            self.closed = False
            senders.append(self)

        def set_destination(self, host, port):
            super().set_destination(host, port)
            self.host = host
            self.port = port

        def send_chatbox(self, text):
            self.chatboxes.append(text)

        def close(self):
            self.closed = True

    def compose_memory_osc(**options):
        options.pop("state_provider")
        options.pop("gate_provider")
        options.pop("log_diagnostic")
        options["state_provider"] = options.pop("osc_state_provider")
        return OscControlIntegrationOwner(
            receiver_owner=receiver,
            query_service=service,
            **options,
        )

    monkeypatch.setattr(wiring_runtime_pipeline, "VrchatOscUdpSender", MemorySender)
    monkeypatch.setattr(application_runtime, "compose_vrc_mic_sync", compose_memory_osc)

    async def enable_manual(application):
        control = application.control()
        submitted = await control.submit(
            "settings.apply",
            {
                "changes": {
                    "osc.connection": {
                        "mode": "manual",
                        "send_port": 19000,
                        "receive_port": 19001,
                    }
                }
            },
            request_id="memory-osc-manual",
        )
        assert (await control.wait(submitted["operation_id"], timeout=5))["status"] == "applied"
        owner = application._runtime_shutdown.vrc_mic_sync()
        assert owner._receiver_owner is receiver
        assert owner.query_runtime.service is service
        assert control.pipeline.sender in senders
        return owner

    return SimpleNamespace(
        receiver=receiver,
        service=service,
        senders=senders,
        enable_manual=enable_manual,
    )


@pytest.fixture(autouse=True)
def _isolated_user_data_dirs(tmp_path_factory, monkeypatch):
    root = tmp_path_factory.mktemp("user-data")
    for name in ("LOCALAPPDATA", "APPDATA", "XDG_CONFIG_HOME"):
        path = root / name.lower()
        path.mkdir()
        monkeypatch.setenv(name, str(path))


@pytest.fixture(autouse=True)
def _restore_global_logging_and_locale():
    from puripuly_heart.ui.i18n import get_locale, set_locale

    loggers = [logging.getLogger(name or None) for name in _GLOBAL_LOGGER_NAMES]
    handlers = [tuple(logger.handlers) for logger in loggers]
    levels = [logger.level for logger in loggers]
    locale = get_locale()
    yield
    set_locale(locale)
    for logger, level in zip(loggers, levels, strict=True):
        logger.setLevel(level)
    for logger, original in zip(loggers, handlers, strict=True):
        for handler in tuple(logger.handlers):
            if handler not in original:
                logger.removeHandler(handler)
