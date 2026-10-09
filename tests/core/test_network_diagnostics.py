from __future__ import annotations

import asyncio
import ssl
from collections.abc import Mapping

import httpx
import pytest

from puripuly_heart.core.error_messages import provider_failure_report, stt_failure_report
from puripuly_heart.core.network_diagnostics import classify_transport_error, safe_transport_fields

SECRET = "https://user:injected-secret@host/path?key=injected-secret"


def certificate_failure(code: int) -> ssl.SSLCertVerificationError:
    failure = ssl.SSLCertVerificationError(1, SECRET)
    failure.verify_code = code
    return failure


@pytest.mark.parametrize("code", [20, 62, 0x800B0109])
def test_certificate_codes_preserve_namespace_without_mislabeling_ssl_errno(code: int) -> None:
    failure = certificate_failure(code)
    fields = safe_transport_fields(failure)
    assert fields["tls_verify_code"] == code
    assert fields["tls_backend"] == "unknown"
    assert "os_errno" not in fields
    assert SECRET not in repr(fields)


@pytest.mark.parametrize(
    ("failure", "kind", "category"),
    [
        (certificate_failure(20), "tls", "network"),
        (httpx.ProxyError("neutral"), "proxy", "network"),
        (httpx.ConnectError("neutral"), "network", "network"),
        (httpx.ReadTimeout("neutral"), "timeout", "timeout"),
    ],
)
def test_typed_causes_classify_authentication_translation_and_stt_consistently(
    failure: Exception, kind: str, category: str
) -> None:
    from puripuly_heart.core.openrouter.authentication import OpenRouterAuthenticationError

    wrapper = RuntimeError("neutral")
    wrapper.__cause__ = failure
    assert classify_transport_error(wrapper) == kind
    assert OpenRouterAuthenticationError.from_exception(wrapper, stage="key_verification").reason == kind
    assert provider_failure_report(wrapper, provider="openrouter", operation="translate").diagnostics.category == category
    assert stt_failure_report(wrapper, provider="soniox", operation="open_session", channel="self").diagnostics.category == category


def test_allowlisted_metadata_and_numeric_os_fields_only() -> None:
    failure = OSError(10061, SECRET)
    failure.winerror = 10061
    failure.connection_diagnostics = {
        "transport": "wss", "tls_source": "windows", "tls_backend": "native",
        "proxy_source": "system", "proxy_url": SECRET, "ca_path": SECRET,
    }
    fields = safe_transport_fields(failure)
    assert fields == {
        "transport": "wss", "tls_source": "windows", "tls_backend": "native",
        "proxy_source": "system", "os_errno": 10061, "winerror": 10061,
    }


@pytest.mark.parametrize("value", [True, SECRET, object()])
def test_hostile_metadata_and_numeric_attributes_are_not_serialized(value: object) -> None:
    failure = certificate_failure(20)
    failure.verify_code = value
    failure.connection_diagnostics = {key: value for key in ("transport", "tls_source", "tls_backend", "proxy_source")}
    failure.diagnostic_transport_fields = {"tls_verify_code": value, "winerror": value, "raw_exception": SECRET}
    fields = safe_transport_fields(failure)
    assert set(fields.values()) == {"unknown"}
    assert SECRET not in repr(fields)


def test_hostile_attribute_access_and_mapping_reads_are_ignored() -> None:
    class HostileMapping(Mapping):
        def __getitem__(self, key):
            raise RuntimeError(SECRET)

        def __iter__(self):
            raise RuntimeError(SECRET)

        def __len__(self):
            raise RuntimeError(SECRET)

    class HostileError(RuntimeError):
        @property
        def diagnostic_transport_fields(self):
            raise RuntimeError(SECRET)

    failure = HostileError(SECRET)
    failure.connection_diagnostics = HostileMapping()
    assert set(safe_transport_fields(failure).values()) == {"unknown"}


def test_cyclic_and_overlong_cause_chains_are_bounded() -> None:
    first = RuntimeError("neutral")
    second = certificate_failure(62)
    first.__context__ = second
    second.__cause__ = first
    assert safe_transport_fields(first)["tls_verify_code"] == 62
    assert classify_transport_error(first) == "tls"
    root = current = RuntimeError("neutral")
    for _ in range(100):
        current.__cause__ = RuntimeError("neutral")
        current = current.__cause__
    current.__cause__ = certificate_failure(20)
    assert classify_transport_error(root) is None
    assert "tls_verify_code" not in safe_transport_fields(root)


def test_cancellation_is_not_a_transport_failure() -> None:
    assert classify_transport_error(asyncio.CancelledError()) is None


@pytest.mark.parametrize("code", [20, 62])
def test_requests_retry_reason_and_ssl_argument_preserve_verification_code(code: int) -> None:
    from requests import exceptions as requests_errors
    from urllib3 import exceptions as urllib3_errors

    from puripuly_heart.core.error_messages import format_error_report_for_log

    ssl_wrapper = urllib3_errors.SSLError(certificate_failure(code))
    retry = urllib3_errors.MaxRetryError(None, SECRET, reason=ssl_wrapper)
    failure = requests_errors.SSLError(retry)
    failure.connection_diagnostics = {
        "transport": "https", "tls_source": "explicit_file",
        "proxy_source": "direct", "tls_backend": "openssl",
    }
    assert failure.__cause__ is failure.__context__ is None
    report = provider_failure_report(failure, provider="dashscope", operation="recognize")
    assert classify_transport_error(failure) == "tls"
    assert report.diagnostics.category == "network"
    assert report.diagnostics.fields["tls_verify_code"] == code
    assert report.diagnostics.fields["tls_backend"] == "openssl"
    assert "os_errno" not in report.diagnostics.fields
    assert SECRET not in repr(report)
    assert f"tls_verify_code={code}" in format_error_report_for_log(report)


def test_exception_graph_identity_limit_ignores_raw_args_and_cyclic_reason() -> None:
    from puripuly_heart.core.network_diagnostics import transport_exception_chain

    root = RuntimeError(SECRET, *(RuntimeError(SECRET) for _ in range(100)))
    root.reason = root
    nodes = tuple(transport_exception_chain(root))
    assert len(nodes) == 32
    assert len({id(node) for node in nodes}) == 32
    assert SECRET not in repr(safe_transport_fields(root))


@pytest.mark.parametrize(
    ("failure", "kind"),
    [
        ("ssl", "tls"),
        ("proxy", "proxy"),
        ("timeout", "timeout"),
        ("connection", "network"),
    ],
)
def test_requests_and_urllib3_typed_wrappers_classify_without_message_heuristics(
    failure: str, kind: str
) -> None:
    from requests import exceptions as requests_errors
    from urllib3 import exceptions as urllib3_errors

    wrappers = {
        "ssl": (requests_errors.SSLError("neutral"), urllib3_errors.SSLError("neutral")),
        "proxy": (
            requests_errors.ProxyError("neutral"),
            urllib3_errors.ProxyError("neutral", RuntimeError("neutral")),
        ),
        "timeout": (requests_errors.Timeout("neutral"), urllib3_errors.TimeoutError("neutral")),
        "connection": (
            requests_errors.ConnectionError("neutral"),
            urllib3_errors.NewConnectionError(None, "neutral"),
        ),
    }
    for wrapper in wrappers[failure]:
        assert classify_transport_error(wrapper) == kind
