from __future__ import annotations

import logging
import socket

import httpx
import pytest

from puripuly_heart.core.qq_auth_diagnostics import exception_fields, log_qq_auth


@pytest.mark.parametrize(
    "exception,kind",
    [
        (httpx.ReadTimeout("token=do-not-log"), "timeout"),
        (httpx.ProxyError("http://user:password@proxy"), "proxy"),
        (socket.gaierror(-2, "identity=123456789"), "dns"),
    ],
)
def test_qq_network_classification_does_not_render_exception(exception, kind, caplog):
    with caplog.at_level(logging.ERROR):
        log_qq_auth("broker_request", "broker_exception", fields=exception_fields(exception))
    assert f"network_error_kind={kind}" in caplog.text
    assert "category=network" in caplog.text
    for sensitive in ("do-not-log", "password", "123456789"):
        assert sensitive not in caplog.text


def test_qq_log_fields_are_closed_allowlists(caplog):
    with caplog.at_level(logging.ERROR):
        log_qq_auth(
            "secret-token",
            "secret-code",
            fields={
                "broker_subcode": "qq_secret-token",
                "proxy_source": "http://secret@host",
                "exception_type": "token=secret",
                "qq_identity": "123456789",
                "credential": "a" * 64,
                "openrouter_api_key": "sk-or-actual-secret",
                "http_status": 401,
                "retry_after_ms": 5000,
            },
        )
    assert "code=unknown_error" in caplog.text
    assert "http_status=401" in caplog.text
    assert "retry_after_ms=5000" in caplog.text
    for value in ("secret-token", "qq_secret", "123456789", "a" * 64, "sk-or-actual"):
        assert value not in caplog.text
