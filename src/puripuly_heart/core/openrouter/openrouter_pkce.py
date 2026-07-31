from __future__ import annotations

import asyncio
import base64
import hashlib
import http
import queue
import secrets
import threading
import urllib.parse
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx

from puripuly_heart.core.oauth_callback_page import render_oauth_callback_completion_page

OPENROUTER_AUTH_URL = "https://openrouter.ai/auth"
OPENROUTER_AUTH_EXCHANGE_URL = "https://openrouter.ai/api/v1/auth/keys"
PKCE_CHALLENGE_METHOD = "S256"
CALLBACK_TIMEOUT_SECONDS = 180
_LISTENER_CLOSED_SENTINEL = object()


def _base64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _code_challenge(verifier: str) -> str:
    return _base64url(hashlib.sha256(verifier.encode("ascii")).digest())


@dataclass(slots=True)
class OpenRouterPKCESession:
    code_verifier: str
    code_challenge: str
    state: str
    callback_url: str
    authorization_url: str


@dataclass(slots=True)
class OpenRouterPKCEExchangeResult:
    api_key: str
    user_id: str | None


@dataclass(slots=True)
class _OpenRouterPKCECallbackListener:
    server: ThreadingHTTPServer
    result_queue: queue.Queue[str | object]
    timeout_seconds: float
    _worker: threading.Thread = field(init=False, repr=False)
    _closed: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        self._worker = threading.Thread(
            target=self.server.serve_forever,
            kwargs={"poll_interval": 0.1},
            daemon=True,
        )
        self._worker.start()

    def wait_for_code(self) -> str:
        try:
            result = self.result_queue.get(timeout=self.timeout_seconds)
        except queue.Empty as exc:
            raise TimeoutError("timed out waiting for OpenRouter callback") from exc

        if result is _LISTENER_CLOSED_SENTINEL:
            raise RuntimeError("OpenRouter callback listener was closed")
        return str(result)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.result_queue.put_nowait(_LISTENER_CLOSED_SENTINEL)
        except queue.Full:
            pass
        self.server.shutdown()
        self.server.server_close()
        self._worker.join(timeout=5.0)


class OpenRouterPKCEClient:
    def __init__(self, *, callback_origin: str):
        self.callback_origin = callback_origin.rstrip("/")
        self.current_authorization_url: str | None = None

    def build_session(self) -> OpenRouterPKCESession:
        code_verifier = secrets.token_urlsafe(64)
        code_challenge = _code_challenge(code_verifier)
        state = secrets.token_urlsafe(24)
        callback_url = f"{self.callback_origin}/callback"
        authorization_url = (
            f"{OPENROUTER_AUTH_URL}?"
            f"{urllib.parse.urlencode({
                'callback_url': callback_url,
                'code_challenge': code_challenge,
                'code_challenge_method': PKCE_CHALLENGE_METHOD,
                'state': state,
            })}"
        )
        return OpenRouterPKCESession(
            code_verifier=code_verifier,
            code_challenge=code_challenge,
            state=state,
            callback_url=callback_url,
            authorization_url=authorization_url,
        )

    async def exchange_code(
        self,
        *,
        code: str,
        code_verifier: str,
        code_challenge_method: str,
    ) -> OpenRouterPKCEExchangeResult:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                OPENROUTER_AUTH_EXCHANGE_URL,
                json={
                    "code": code,
                    "code_verifier": code_verifier,
                    "code_challenge_method": code_challenge_method,
                },
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            payload = response.json()

        user_id = payload.get("user_id")
        return OpenRouterPKCEExchangeResult(
            api_key=str(payload["key"]),
            user_id=user_id if isinstance(user_id, str) else None,
        )

    def _extract_callback_code(self, path: str, session: OpenRouterPKCESession) -> str:
        parsed_path = urllib.parse.urlsplit(path)
        query = urllib.parse.parse_qs(parsed_path.query)
        state = query.get("state", [""])[0]
        if state and state != session.state:
            raise ValueError("callback state did not match")

        code = query.get("code", [""])[0]
        if not code:
            raise ValueError("callback code was missing")
        return code

    def _create_callback_listener(
        self, session: OpenRouterPKCESession
    ) -> _OpenRouterPKCECallbackListener:
        result_queue: queue.Queue[str | object] = queue.Queue(maxsize=1)
        parsed_callback = urllib.parse.urlsplit(session.callback_url)
        host = parsed_callback.hostname or "127.0.0.1"
        port = parsed_callback.port or 80
        callback_path = parsed_callback.path or "/callback"
        client = self

        class CallbackHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                request_path = urllib.parse.urlsplit(self.path).path or "/"
                if request_path != callback_path:
                    self.send_response(http.HTTPStatus.NOT_FOUND)
                    self.end_headers()
                    return

                try:
                    code = client._extract_callback_code(self.path, session)
                except ValueError:
                    self.send_response(http.HTTPStatus.BAD_REQUEST)
                    self.end_headers()
                    return

                try:
                    result_queue.put_nowait(code)
                except queue.Full:
                    pass

                body = render_oauth_callback_completion_page()
                self.send_response(http.HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:  # noqa: A003
                return

        server = ThreadingHTTPServer((host, port), CallbackHandler)
        return _OpenRouterPKCECallbackListener(
            server=server,
            result_queue=result_queue,
            timeout_seconds=CALLBACK_TIMEOUT_SECONDS,
        )

    async def run_desktop_flow(self) -> OpenRouterPKCEExchangeResult:
        session = self.build_session()
        self.current_authorization_url = session.authorization_url
        listener = self._create_callback_listener(session)
        try:
            webbrowser.open(session.authorization_url)
            code = await asyncio.to_thread(listener.wait_for_code)
        finally:
            listener.close()

        return await self.exchange_code(
            code=code,
            code_verifier=session.code_verifier,
            code_challenge_method=PKCE_CHALLENGE_METHOD,
        )

    def reopen_authorization_url(self) -> bool:
        if not self.current_authorization_url:
            return False
        return bool(webbrowser.open(self.current_authorization_url))
