from __future__ import annotations

import socket
import threading
import urllib.error
import urllib.parse
import urllib.request

import pytest
from puripuly_heart.core.discord_oauth_loopback import (
    DISCORD_OAUTH_LOOPBACK_PATH,
    DISCORD_OAUTH_LOOPBACK_PORTS,
    DiscordOAuthCallbackError,
    DiscordOAuthCallbackResult,
    DiscordOAuthLoopbackClosedError,
    _DiscordOAuthHTTPServer,
    _send_success_callback_response,
    bind_first_available,
)

from puripuly_heart.core import discord_oauth_loopback as loopback


def _callback_url(listener: object, **params: str) -> str:
    return f"{listener.redirect_uri}?{urllib.parse.urlencode(params)}"


def _get_status(url: str) -> int:
    try:
        with urllib.request.urlopen(url, timeout=2.0) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code


def _get_response(url: str) -> tuple[int, str, str]:
    try:
        with urllib.request.urlopen(url, timeout=2.0) as response:
            body = response.read().decode("utf-8")
            return response.status, response.headers.get("content-type", ""), body
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        return exc.code, exc.headers.get("content-type", ""), body


def test_bind_first_available_uses_fixed_loopback_redirect_uri() -> None:
    listener = bind_first_available()
    try:
        assert listener.port in DISCORD_OAUTH_LOOPBACK_PORTS
        assert listener.redirect_uri == (
            f"http://127.0.0.1:{listener.port}{DISCORD_OAUTH_LOOPBACK_PATH}"
        )
    finally:
        listener.close()


def test_success_callback_returns_completion_page_and_wait_returns_code_state() -> None:
    listener = bind_first_available(locale="ko")
    try:
        status, content_type, body = _get_response(
            _callback_url(listener, code="discord-code-1", state="state-1")
        )

        assert status == 200
        assert "text/html" in content_type
        assert "charset=utf-8" in content_type.lower()
        assert "인증 정보를 받았어요" in body
        assert "PuriPuly 앱에서 연결을 마무리하고 있어요." in body
        assert "이 탭은 닫아도 괜찮아요." in body
        assert "Discord 인증" not in body
        assert "Managed 키" not in body
        assert "window.close" not in body
        assert "<script" not in body
        assert "<button" not in body
        assert "border-radius" not in body
        assert "box-shadow" not in body
        assert "Arial Rounded MT Bold" not in body
        assert "<title>PuriPuly</title>" in body

        result = listener.wait(timeout=2.0)

        assert result.code == "discord-code-1"
        assert result.state == "state-1"
    finally:
        listener.close()


class FakeSuccessResponseHandler:
    def __init__(self) -> None:
        self.wfile = self
        self.status: int | None = None
        self.headers: list[tuple[str, str]] = []
        self.ended = False

    def send_response(self, status: int) -> None:
        self.status = status

    def send_header(self, name: str, value: str) -> None:
        self.headers.append((name, value))

    def end_headers(self) -> None:
        self.ended = True

    def write(self, _body: bytes) -> int:
        raise OSError("browser disconnected")


class FakeSuccessResponseListener:
    locale = "en"

    def __init__(self) -> None:
        self.completed: DiscordOAuthCallbackResult | None = None
        self.closed_async = False

    def _complete(self, *, result: DiscordOAuthCallbackResult | None = None) -> None:
        self.completed = result

    def _close_async(self) -> None:
        self.closed_async = True


def test_success_callback_delivers_result_even_when_browser_disconnects() -> None:
    handler = FakeSuccessResponseHandler()
    listener = FakeSuccessResponseListener()
    result = DiscordOAuthCallbackResult(code="discord-code-1", state="state-1")

    with pytest.raises(OSError, match="browser disconnected"):
        _send_success_callback_response(handler, listener, result)

    assert listener.completed == result
    assert listener.closed_async is True


@pytest.mark.parametrize(
    ("locale", "expected_text", "expected_font"),
    [
        ("en", "We received your authentication", "system-ui"),
        ("ko", "인증 정보를 받았어요", "NanumSquareRound"),
        ("ja", "認証情報を受け取りました", "M PLUS Rounded 1c"),
        ("zh-CN", "已收到认证信息", "ResourceHanRoundedCN"),
    ],
)
def test_success_callback_completion_page_matches_locale(
    locale: str,
    expected_text: str,
    expected_font: str,
) -> None:
    listener = bind_first_available(locale=locale)
    try:
        status, _content_type, body = _get_response(
            _callback_url(listener, code="discord-code-1", state="state-1")
        )

        assert status == 200
        assert expected_text in body
        assert expected_font in body
    finally:
        listener.close()


def test_error_callback_returns_204_and_wait_raises_callback_error() -> None:
    listener = bind_first_available()
    try:
        assert _get_status(_callback_url(listener, error="access_denied", state="state-2")) == 204

        with pytest.raises(DiscordOAuthCallbackError) as exc_info:
            listener.wait(timeout=2.0)

        assert exc_info.value.error == "access_denied"
        assert exc_info.value.state == "state-2"
    finally:
        listener.close()


def test_wrong_path_and_missing_parameters_do_not_complete_callback() -> None:
    listener = bind_first_available()
    try:
        wrong_path = f"http://127.0.0.1:{listener.port}/wrong?code=code&state=state"
        assert _get_status(wrong_path) == 404
        assert _get_status(f"{listener.redirect_uri}?code=code-only") == 400
        assert _get_status(f"{listener.redirect_uri}?state=state-only") == 400

        listener.close()
        with pytest.raises(DiscordOAuthLoopbackClosedError):
            listener.wait(timeout=2.0)
    finally:
        listener.close()


def test_close_unblocks_wait_and_stops_listener_thread() -> None:
    listener = bind_first_available()
    started = threading.Event()
    outcome: dict[str, BaseException | object] = {}

    def wait_for_callback() -> None:
        started.set()
        try:
            outcome["result"] = listener.wait(timeout=10.0)
        except BaseException as exc:  # noqa: BLE001 - test captures thread outcome
            outcome["error"] = exc

    thread = threading.Thread(target=wait_for_callback)
    thread.start()
    assert started.wait(timeout=1.0)

    listener.close()
    thread.join(timeout=2.0)

    assert thread.is_alive() is False
    assert isinstance(outcome.get("error"), DiscordOAuthLoopbackClosedError)


def _free_loopback_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


def test_server_bind_does_not_enable_reuseaddr_and_uses_windows_exclusive_bind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exclusive_addr_use = 0x1004
    monkeypatch.setattr(socket, "SO_EXCLUSIVEADDRUSE", exclusive_addr_use, raising=False)
    calls: list[tuple[str, int, int, int] | tuple[str, tuple[str, int]]] = []

    class FakeSocket:
        def setsockopt(self, level: int, option: int, value: int) -> None:
            calls.append(("setsockopt", level, option, value))

        def bind(self, address: tuple[str, int]) -> None:
            calls.append(("bind", address))

        def getsockname(self) -> tuple[str, int]:
            return ("127.0.0.1", DISCORD_OAUTH_LOOPBACK_PORTS[0])

    server = _DiscordOAuthHTTPServer.__new__(_DiscordOAuthHTTPServer)
    server.socket = FakeSocket()
    server.server_address = ("127.0.0.1", DISCORD_OAUTH_LOOPBACK_PORTS[0])

    server.server_bind()

    assert _DiscordOAuthHTTPServer.allow_reuse_address is False
    assert calls == [
        ("setsockopt", socket.SOL_SOCKET, exclusive_addr_use, 1),
        ("bind", ("127.0.0.1", DISCORD_OAUTH_LOOPBACK_PORTS[0])),
    ]


def test_occupied_configured_port_cannot_be_reused_and_falls_back_to_next_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    occupied = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    occupied.bind(("127.0.0.1", 0))
    occupied.listen(1)
    occupied_port = int(occupied.getsockname()[1])
    fallback_port = _free_loopback_port()
    monkeypatch.setattr(
        loopback,
        "DISCORD_OAUTH_LOOPBACK_PORTS",
        (occupied_port, fallback_port),
    )

    try:
        listener = bind_first_available()
        try:
            assert listener.port == fallback_port
        finally:
            listener.close()
    finally:
        if occupied is not None:
            occupied.close()
