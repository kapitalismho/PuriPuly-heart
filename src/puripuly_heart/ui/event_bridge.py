from __future__ import annotations

import asyncio
import logging

from puripuly_heart.domain.events import UIEvent, UIEventType
from puripuly_heart.domain.models import OSCMessage, Transcript, Translation

logger = logging.getLogger(__name__)


class UIEventBridge:
    def __init__(self, *, app: object, event_queue: asyncio.Queue[UIEvent]):
        self.app = app
        self.event_queue = event_queue
        self._running = False

    async def run(self) -> None:
        self._running = True
        logger.info("UI Event Bridge started")
        try:
            while self._running:
                event = await self.event_queue.get()
                try:
                    await self._handle_event(event)
                except Exception:
                    logger.exception("Error handling UI event")
                finally:
                    self.event_queue.task_done()
        except asyncio.CancelledError:
            logger.info("UI Event Bridge cancelled")
            raise

    async def _handle_event(self, event: UIEvent) -> None:
        if event.type == UIEventType.SESSION_STATE_CHANGED:
            state = event.payload
            connected = getattr(state, "name", "") == "STREAMING"
            dash = getattr(self.app, "view_dashboard", None)
            if dash is not None:
                dash.set_status(connected)
            return

        if event.type in (UIEventType.TRANSCRIPT_PARTIAL, UIEventType.TRANSCRIPT_FINAL):
            transcript = event.payload
            if not isinstance(transcript, Transcript):
                return
            source = event.source or "Mic"

            dash = getattr(self.app, "view_dashboard", None)
            if dash is not None:
                dash.hero_text.value = transcript.text
                if dash.hero_text.page:
                    dash.hero_text.update()

            if event.type == UIEventType.TRANSCRIPT_FINAL:
                add_history = getattr(self.app, "add_history_entry", None)
                if add_history is not None:
                    add_history(source, transcript.text)
            return

        if event.type == UIEventType.TRANSLATION_DONE:
            translation = event.payload
            if not isinstance(translation, Translation):
                return
            source = event.source or "Mic"
            add_history = getattr(self.app, "add_history_entry", None)
            if add_history is not None:
                add_history(f"{source} (Translated)", translation.text)
            return

        if event.type == UIEventType.OSC_SENT:
            msg = event.payload
            if not isinstance(msg, OSCMessage):
                return
            dash = getattr(self.app, "view_dashboard", None)
            if dash is not None:
                dash.hero_text.value = msg.text
                if dash.hero_text.page:
                    dash.hero_text.update()
            add_history = getattr(self.app, "add_history_entry", None)
            if add_history is not None:
                add_history("VRChat", msg.text)
            return

        if event.type == UIEventType.ERROR:
            payload = event.payload
            text = str(payload) if payload is not None else "Unknown error"
            logs = getattr(self.app, "view_logs", None)
            if logs is not None:
                logs.append_log(f"ERROR: {text}")
            return
