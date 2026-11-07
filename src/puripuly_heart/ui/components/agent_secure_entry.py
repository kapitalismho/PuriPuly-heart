from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable

from puripuly_heart.app.ports.agent_chat import AgentSecureEntryRequest
from puripuly_heart.app.services.agent_chat.provider_setup import PROVIDER_SETUPS
from puripuly_heart.ui.components.settings.api_key_verification_controller import (
    ApiKeyVerificationController,
    SaveHandler,
    VerifyHandler,
)


class AgentSecureEntryController:
    def __init__(
        self,
        *,
        save: SaveHandler,
        verify: VerifyHandler,
        persist_verification: Callable[[str, str, bool], None],
    ) -> None:
        self._save = save
        self._verify = verify
        self._persist_verification = persist_verification
        self._controller: ApiKeyVerificationController | None = None
        self._pending: asyncio.Task[None] | None = None
        self._value = ""
        self._epoch = 0

    def cancel(self) -> None:
        self._epoch += 1
        self._value = ""
        controller, self._controller = self._controller, None
        if controller is not None:
            controller.bind_completion(None)
            controller.set_value_getter(lambda: "")
        pending, self._pending = self._pending, None
        if pending is not None and not pending.done() and pending is not asyncio.current_task():
            pending.cancel()

    async def submit(
        self,
        request: AgentSecureEntryRequest,
        value: str,
        *,
        is_current: Callable[[], bool],
        complete: Callable[[str, bool, str | None], None],
    ) -> dict:
        setup = PROVIDER_SETUPS.get(request.provider)
        if setup is None or not value or not is_current():
            return {"status": "rejected", "code": "stale_request"}
        self.cancel()
        epoch = self._epoch
        self._value = value
        completion: dict = {}

        def current() -> bool:
            return epoch == self._epoch and is_current()

        async def save(secret_key: str, credential: str) -> object:
            if not current():
                raise asyncio.CancelledError
            result = self._save(secret_key, credential)
            if inspect.isawaitable(result):
                result = await result
            if not current():
                raise asyncio.CancelledError
            return result

        async def verify(provider: str, credential: str):
            if not current():
                raise asyncio.CancelledError
            result = self._verify(provider, credential)
            if inspect.isawaitable(result):
                result = await result
            if not current():
                raise asyncio.CancelledError
            if result.provider != provider:
                raise ValueError("verification_provider_mismatch")
            self._persist_verification(provider, credential, result.status == "verified")
            return result

        def on_complete(provider: str, success: bool, code: str | None) -> None:
            if not current() or completion:
                return
            completion.update(status="verified" if success else "failed")
            complete(provider, success, code)

        controller = ApiKeyVerificationController(
            secret_key=setup.secret_key.value,
            provider=request.provider,
            on_save=save,
            on_verify=verify,
            on_complete=on_complete,
        )
        self._controller = controller
        controller.set_value_getter(lambda: self._value if current() else "")
        controller.notify_edit()
        controller.handle_blur(value)
        pending = controller.run_pending()
        task = asyncio.ensure_future(pending) if pending is not None else None
        self._pending = task
        try:
            if task is not None:
                await task
            if epoch != self._epoch:
                return {"status": "cancelled", "request_id": request.id}
            return {"status": completion.get("status", "cancelled"), "request_id": request.id}
        except asyncio.CancelledError:
            if epoch == self._epoch:
                self.cancel()
            current_task = asyncio.current_task()
            if current_task is not None and current_task.cancelling():
                raise
            return {"status": "cancelled", "request_id": request.id}
        finally:
            controller.bind_completion(None)
            controller.set_value_getter(lambda: "")
            if epoch == self._epoch:
                self._value = ""
                self._controller = None
                self._pending = None
