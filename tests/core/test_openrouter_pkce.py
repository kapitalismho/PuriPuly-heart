from __future__ import annotations

import asyncio
import socket
import threading

import httpx
import pytest
from puripuly_heart.core.openrouter_pkce import (
    OpenRouterPKCEClient,
    OpenRouterPKCEExchangeResult,
    OpenRouterPKCESession,
)

from puripuly_heart.ui.i18n import get_locale, set_locale


def _get_unused_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_build_authorize_url_uses_loopback_callback_and_s256() -> None:
    client = OpenRouterPKCEClient(callback_origin="http://127.0.0.1:43123")

    session = client.build_session()

    assert session.authorization_url.startswith("https://openrouter.ai/auth?")
    assert "callback_url=http%3A%2F%2F127.0.0.1%3A43123%2Fcallback" in session.authorization_url
    assert "code_challenge_method=S256" in session.authorization_url
    assert session.code_verifier
    assert session.state


@pytest.mark.asyncio
async def test_exchange_code_posts_expected_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    class DummyResponse:
        status_code = 200

        def json(self) -> dict[str, str]:
            return {"key": "sk-or-v1-user", "user_id": "user_123"}

        def raise_for_status(self) -> None:
            return None

    class DummyAsyncClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def __aenter__(self) -> DummyAsyncClient:
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def post(
            self,
            url: str,
            *,
            json: dict[str, str],
            headers: dict[str, str],
        ) -> DummyResponse:
            seen["url"] = url
            seen["json"] = json
            seen["headers"] = headers
            return DummyResponse()

    monkeypatch.setattr("httpx.AsyncClient", DummyAsyncClient)

    client = OpenRouterPKCEClient(callback_origin="http://127.0.0.1:43123")
    result = await client.exchange_code(
        code="code_123",
        code_verifier="verifier_123",
        code_challenge_method="S256",
    )

    assert result.api_key == "sk-or-v1-user"
    assert result.user_id == "user_123"
    assert seen["url"] == "https://openrouter.ai/api/v1/auth/keys"
    assert seen["json"] == {
        "code": "code_123",
        "code_verifier": "verifier_123",
        "code_challenge_method": "S256",
    }
    assert seen["headers"] == {"Content-Type": "application/json"}


def test_extract_callback_code_accepts_missing_state_but_rejects_mismatch() -> None:
    client = OpenRouterPKCEClient(callback_origin="http://127.0.0.1:43123")
    session = client.build_session()

    assert client._extract_callback_code("/callback?code=code_123", session) == "code_123"

    with pytest.raises(ValueError, match="state"):
        client._extract_callback_code("/callback?code=code_123&state=wrong-state", session)


def test_reopen_authorization_url_reopens_current_pkce_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[str] = []
    client = OpenRouterPKCEClient(callback_origin="http://localhost:3000")

    assert client.reopen_authorization_url() is False

    client.current_authorization_url = "https://openrouter.ai/auth?callback_url=callback"
    monkeypatch.setattr(
        "puripuly_heart.core.openrouter_pkce.webbrowser.open",
        lambda url: opened.append(url) or True,
    )

    assert client.reopen_authorization_url() is True
    assert opened == ["https://openrouter.ai/auth?callback_url=callback"]


@pytest.mark.asyncio
async def test_run_desktop_flow_binds_listener_before_opening_browser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = OpenRouterPKCEClient(callback_origin="http://127.0.0.1:43123")
    session = OpenRouterPKCESession(
        code_verifier="verifier_123",
        code_challenge="challenge_123",
        state="state_123",
        callback_url="http://127.0.0.1:43123/callback",
        authorization_url="https://openrouter.ai/auth?callback_url=callback",
    )
    seen: list[str] = []

    class DummyListener:
        def wait_for_code(self) -> str:
            seen.append("wait")
            return "code_123"

        def close(self) -> None:
            seen.append("close")

    async def fake_to_thread(func, *args):  # type: ignore[no-untyped-def]
        seen.append("to_thread")
        return func(*args)

    async def fake_exchange_code(**_kwargs: str) -> OpenRouterPKCEExchangeResult:
        seen.append("exchange")
        return OpenRouterPKCEExchangeResult(api_key="sk-or-v1-user", user_id="user_123")

    def fake_create_callback_listener(session_arg: OpenRouterPKCESession) -> DummyListener:
        assert session_arg == session
        seen.append("bind")
        return DummyListener()

    def fake_open(url: str) -> bool:
        assert seen == ["bind"]
        assert url == session.authorization_url
        seen.append("open")
        return True

    monkeypatch.setattr(client, "build_session", lambda: session)
    monkeypatch.setattr(client, "_create_callback_listener", fake_create_callback_listener)
    monkeypatch.setattr(client, "exchange_code", fake_exchange_code)
    monkeypatch.setattr("puripuly_heart.core.openrouter_pkce.asyncio.to_thread", fake_to_thread)
    monkeypatch.setattr("puripuly_heart.core.openrouter_pkce.webbrowser.open", fake_open)

    result = await client.run_desktop_flow()

    assert result == OpenRouterPKCEExchangeResult(api_key="sk-or-v1-user", user_id="user_123")
    assert seen == ["bind", "open", "to_thread", "wait", "close", "exchange"]


def test_callback_listener_ignores_invalid_requests_until_valid_code_arrives() -> None:
    port = _get_unused_loopback_port()
    client = OpenRouterPKCEClient(callback_origin=f"http://127.0.0.1:{port}")
    session = client.build_session()
    listener = client._create_callback_listener(session)
    result: dict[str, object] = {}
    previous_locale = get_locale()

    def run_listener() -> None:
        result["code"] = listener.wait_for_code()

    worker = threading.Thread(target=run_listener, daemon=True)
    worker.start()
    try:
        set_locale("ko")

        unrelated = httpx.get(f"http://127.0.0.1:{port}/", timeout=1.0)
        assert unrelated.status_code == 404
        assert unrelated.text == ""

        mismatched_state = httpx.get(
            f"http://127.0.0.1:{port}/callback?code=bad_code&state=wrong-state",
            timeout=1.0,
        )
        assert mismatched_state.status_code == 400
        assert mismatched_state.text == ""

        valid = httpx.get(f"http://127.0.0.1:{port}/callback?code=code_123", timeout=1.0)
        assert valid.status_code == 200
        assert "text/html" in valid.headers.get("content-type", "")
        assert "charset=utf-8" in valid.headers.get("content-type", "").lower()
        assert valid.headers.get("cache-control") == "no-store"
        assert "인증 정보를 받았어요" in valid.text
        assert "PuriPuly 앱에서 연결을 마무리하고 있어요." in valid.text
        assert "이 탭은 닫아도 괜찮아요." in valid.text
        assert "<script" not in valid.text
        assert "window.close" not in valid.text

        worker.join(timeout=2.0)
        assert worker.is_alive() is False
        assert result == {"code": "code_123"}
    finally:
        set_locale(previous_locale)
        listener.close()


@pytest.mark.asyncio
async def test_run_desktop_flow_closes_listener_and_skips_exchange_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = OpenRouterPKCEClient(callback_origin="http://127.0.0.1:43123")
    session = OpenRouterPKCESession(
        code_verifier="verifier_123",
        code_challenge="challenge_123",
        state="state_123",
        callback_url="http://127.0.0.1:43123/callback",
        authorization_url="https://openrouter.ai/auth?callback_url=callback",
    )
    seen: list[str] = []

    class DummyListener:
        def wait_for_code(self) -> str:
            seen.append("wait")
            raise TimeoutError("timed out waiting for OpenRouter callback")

        def close(self) -> None:
            seen.append("close")

    async def fake_to_thread(func, *args):  # type: ignore[no-untyped-def]
        seen.append("to_thread")
        return func(*args)

    async def fake_exchange_code(**_kwargs: str) -> OpenRouterPKCEExchangeResult:
        seen.append("exchange")
        raise AssertionError("exchange_code should not be called on timeout")

    def fake_create_callback_listener(session_arg: OpenRouterPKCESession) -> DummyListener:
        assert session_arg == session
        seen.append("bind")
        return DummyListener()

    def fake_open(url: str) -> bool:
        assert seen == ["bind"]
        assert url == session.authorization_url
        seen.append("open")
        return True

    monkeypatch.setattr(client, "build_session", lambda: session)
    monkeypatch.setattr(client, "_create_callback_listener", fake_create_callback_listener)
    monkeypatch.setattr(client, "exchange_code", fake_exchange_code)
    monkeypatch.setattr("puripuly_heart.core.openrouter_pkce.asyncio.to_thread", fake_to_thread)
    monkeypatch.setattr("puripuly_heart.core.openrouter_pkce.webbrowser.open", fake_open)

    with pytest.raises(TimeoutError, match="timed out waiting for OpenRouter callback"):
        await client.run_desktop_flow()

    assert seen == ["bind", "open", "to_thread", "wait", "close"]


@pytest.mark.asyncio
async def test_run_desktop_flow_cancellation_unblocks_waiting_listener(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    port = _get_unused_loopback_port()
    client = OpenRouterPKCEClient(callback_origin=f"http://127.0.0.1:{port}")
    session = OpenRouterPKCESession(
        code_verifier="verifier_123",
        code_challenge="challenge_123",
        state="state_123",
        callback_url=f"http://127.0.0.1:{port}/callback",
        authorization_url="https://openrouter.ai/auth?callback_url=callback",
    )
    wait_started = threading.Event()
    wait_finished = threading.Event()
    listener_closed = threading.Event()
    real_create_callback_listener = client._create_callback_listener

    class TrackingListener:
        def __init__(self, inner: object) -> None:
            self._inner = inner

        def wait_for_code(self) -> str:
            wait_started.set()
            try:
                return self._inner.wait_for_code()
            finally:
                wait_finished.set()

        def close(self) -> None:
            listener_closed.set()
            self._inner.close()

    async def fake_exchange_code(**_kwargs: str) -> OpenRouterPKCEExchangeResult:
        raise AssertionError("exchange_code should not be called after cancellation")

    def fake_create_callback_listener(session_arg: OpenRouterPKCESession) -> TrackingListener:
        assert session_arg == session
        return TrackingListener(real_create_callback_listener(session_arg))

    def fake_open(url: str) -> bool:
        assert url == session.authorization_url
        return True

    monkeypatch.setattr(client, "build_session", lambda: session)
    monkeypatch.setattr(client, "_create_callback_listener", fake_create_callback_listener)
    monkeypatch.setattr(client, "exchange_code", fake_exchange_code)
    monkeypatch.setattr("puripuly_heart.core.openrouter_pkce.webbrowser.open", fake_open)

    task = asyncio.create_task(client.run_desktop_flow())
    started = await asyncio.to_thread(wait_started.wait, 1.0)
    assert started is True

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert listener_closed.is_set() is True
    finished = await asyncio.to_thread(wait_finished.wait, 1.0)
    assert finished is True
