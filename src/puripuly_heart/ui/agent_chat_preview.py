from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import replace
from typing import Literal

from puripuly_heart.app.ports.agent_chat import (
    AgentChatPort,
    AgentChatSnapshot,
    AgentReportOffer,
    AgentSecureEntryRequest,
    AgentSetupAction,
)

PreviewFamily = Literal["report", "change", "key", "account"]

PREVIEW_DELAY_S = 1.0
PREVIEW_PROVIDER = "deepgram"
PREVIEW_ACCOUNT_PROVIDER = "chatgpt"
PREVIEW_TICKET = "PP-0123456789ABCDEF"
PREVIEW_KEY_ERROR = "error.api_key_invalid"

_REPORT_STEPS = ("offered", "sending", "failed", "sent")
_CHANGE_STEPS = (
    ({"stt.provider": "gemini_transcribe"}, {"stt.provider": "deepgram"}),
    (
        {"languages": {"target": "ja"}, "overlay.background_alpha": 0.6},
        {"languages": {"target": "en"}, "overlay.background_alpha": 0.35},
    ),
    (
        {"audio.input_device": "Headset Microphone (Realtek(R) Audio) - Virtual Cable Input 2"},
        {"audio.input_device": ""},
    ),
    ({"overlay.translation.enabled": True}, {}),
    ({"secrets.deepgram_api_key": False}, {"secrets.deepgram_api_key": True}),
    ({"accounts.chatgpt": False}, {"accounts.chatgpt": True}),
)
_KEY_STEPS = ("create", "entry", "verifying", "failed")
_ACCOUNT_STEPS = ("account_setup", "public_usage")
_STEP_COUNTS = {
    "report": len(_REPORT_STEPS),
    "change": len(_CHANGE_STEPS),
    "key": len(_KEY_STEPS),
    "account": len(_ACCOUNT_STEPS),
}


class AgentChatPreview:
    """Shows synthetic Agent Chat cards over the live conversation without side effects."""

    def __init__(self, live: AgentChatPort, on_finish: Callable[[], None]) -> None:
        self._live = live
        self._on_finish = on_finish
        self._observers: list[Callable[[AgentChatSnapshot], None]] = []
        self._live_unsubscribe: Callable[[], None] | None = None
        self._overlay: dict = {}
        self._steps: dict[str, int] = {}
        self._serial = 0

    def __getattr__(self, name: str):
        return getattr(self._live, name)

    @property
    def snapshot(self) -> AgentChatSnapshot:
        cards = {
            "confirmation": None,
            "confirmation_previous": None,
            "report_offers": (),
            "setup_actions": (),
            "secure_entry_requests": (),
        }
        return replace(self._live.snapshot, **{**cards, **self._overlay})

    def observe(self, callback: Callable[[AgentChatSnapshot], None]) -> Callable[[], None]:
        self._observers.append(callback)
        if self._live_unsubscribe is None:
            self._live_unsubscribe = self._live.observe(lambda _snapshot: self._publish())

        def unsubscribe() -> None:
            if callback in self._observers:
                self._observers.remove(callback)
            if not self._observers and self._live_unsubscribe is not None:
                self._live_unsubscribe()
                self._live_unsubscribe = None

        return unsubscribe

    def cycle(self, family: PreviewFamily) -> None:
        step = (self._steps.get(family, -1) + 1) % _STEP_COUNTS[family]
        self._steps[family] = step
        self._serial += 1
        identity = f"preview-{self._serial}"
        if family == "report":
            self._show_report(identity, _REPORT_STEPS[step])
        elif family == "change":
            changes, previous = _CHANGE_STEPS[step]
            self._overlay = {
                "confirmation": {"id": identity, "changes": changes},
                "confirmation_previous": previous,
            }
        elif family == "key":
            self._show_key(identity, _KEY_STEPS[step])
        else:
            action = _setup(identity, _ACCOUNT_STEPS[step], PREVIEW_ACCOUNT_PROVIDER)
            self._overlay = {"setup_actions": (action,)}
        self._publish()

    def finish(self) -> None:
        self._overlay = {}
        self._publish()
        self._on_finish()

    # Card actions stay inside the preview

    def mark_report_displayed(self, offer_id: str) -> dict:
        return {"status": "displayed", "offer_id": offer_id}

    def mark_setup_action_displayed(self, action_id: str) -> dict:
        return {"status": "displayed", "action_id": action_id}

    def mark_secure_entry_displayed(self, request_id: str) -> dict:
        return {"status": "displayed", "request_id": request_id}

    async def confirm(self, card_id: str, approved: bool) -> dict:
        self.finish()
        return {"status": "declined", "card_id": card_id}

    def activate_setup_action(self, action_id: str) -> dict:
        actions = self._overlay.get("setup_actions", ())
        if any(action.id == action_id and action.kind == "key_console" for action in actions):
            return {"status": "opened", "action_id": action_id}
        self.finish()
        return {"status": "opened", "action_id": action_id}

    def cancel_secure_entry(self, request_id: str) -> dict:
        self.finish()
        return {"status": "cancelled", "request_id": request_id}

    async def submit_secure_entry(self, request_id: str, value: str) -> dict:
        self._update_secure(request_id, status="verifying", code=None)
        await asyncio.sleep(PREVIEW_DELAY_S)
        self._update_secure(request_id, status="failed", code=PREVIEW_KEY_ERROR)
        return {"status": "failed", "request_id": request_id}

    async def send_report(self, offer_id: str) -> dict:
        offers = self._overlay.get("report_offers", ())
        offer = next((item for item in offers if item.id == offer_id), None)
        if offer is None or offer.status not in {"offered", "failed"}:
            return {"status": "rejected", "code": "unavailable_offer"}
        outcome = "sent" if offer.status == "failed" else "failed"
        self._show_report(offer_id, "sending")
        self._publish()
        await asyncio.sleep(PREVIEW_DELAY_S)
        self._show_report(offer_id, outcome)
        self._publish()
        return {"status": outcome}

    def reject_report(self, offer_id: str) -> dict:
        self.finish()
        return {"status": "rejected", "offer_id": offer_id}

    def _show_report(self, identity: str, status: str) -> None:
        ticket = PREVIEW_TICKET if status == "sent" else None
        self._overlay = {"report_offers": (AgentReportOffer(identity, 0, status, ticket),)}

    def _show_key(self, identity: str, step: str) -> None:
        request = AgentSecureEntryRequest(
            identity,
            PREVIEW_PROVIDER,
            "preview",
            0,
            message_index=0,
            available=True,
            status={"verifying": "verifying", "failed": "failed"}.get(step, "pending"),
            code=PREVIEW_KEY_ERROR if step == "failed" else None,
        )
        actions = ()
        if step == "create":
            setup = _setup(f"{identity}-setup", "key_console", PREVIEW_PROVIDER)
            actions = (replace(setup, secure_request_id=identity),)
        self._overlay = {"setup_actions": actions, "secure_entry_requests": (request,)}

    def _update_secure(self, request_id: str, *, status: str, code: str | None) -> None:
        requests = tuple(
            replace(request, status=status, code=code) if request.id == request_id else request
            for request in self._overlay.get("secure_entry_requests", ())
        )
        self._overlay = {**self._overlay, "secure_entry_requests": requests}
        self._publish()

    def _publish(self) -> None:
        snapshot = self.snapshot
        for callback in tuple(self._observers):
            callback(snapshot)


def _setup(identity: str, kind: str, provider: str) -> AgentSetupAction:
    return AgentSetupAction(identity, kind, provider, kind, 0, message_index=0, available=True)
