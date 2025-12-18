import asyncio
import logging
from ajebal_daera_translator.domain.events import (
    STTFinalEvent,
    STTPartialEvent,
    STTSessionStateEvent,
    STTErrorEvent
)
from ajebal_daera_translator.ui.app import TranslatorApp

logger = logging.getLogger(__name__)

class UIEventBridge:
    def __init__(self, app: TranslatorApp, event_queue: asyncio.Queue):
        self.app = app
        self.event_queue = event_queue
        self._running = False

    async def run(self):
        self._running = True
        logger.info("UI Event Bridge started")
        try:
            while self._running:
                event = await self.event_queue.get()
                try:
                    await self._handle_event(event)
                except Exception as exc:
                    logger.exception(f"Error handling UI event: {exc}")
                finally:
                    self.event_queue.task_done()
        except asyncio.CancelledError:
            logger.info("UI Event Bridge cancelled")

    async def _handle_event(self, event):
        # STT Events
        if isinstance(event, STTPartialEvent):
            # Update Dashboard logic for partial
            # TODO: Implement partial update in DashboardView
            pass

        elif isinstance(event, STTFinalEvent):
            transcript = event.transcript
            self.app.view_dashboard.add_history("Mic", transcript.text)
            self.app.view_dashboard.hero_text.value = transcript.text
            self.app.view_dashboard.hero_text.update()

        elif isinstance(event, STTSessionStateEvent):
            connected = (event.state.name == "STREAMING")
            self.app.view_dashboard.set_status(connected)

        elif isinstance(event, STTErrorEvent):
             self.app.view_logs.append_log(f"ERROR: {event.error}")

        # Add more event handlers here (Translation, OSC, etc.)
