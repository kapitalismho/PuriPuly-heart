from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from uuid import uuid4

from ajebal_daera_translator.config.settings import AppSettings
from ajebal_daera_translator.core.clock import SystemClock
from ajebal_daera_translator.core.llm.provider import LLMProvider
from ajebal_daera_translator.core.osc.smart_queue import SmartOscQueue
from ajebal_daera_translator.core.osc.udp_sender import VrchatOscUdpSender
from ajebal_daera_translator.domain.models import OSCMessage


@dataclass(slots=True)
class HeadlessStdinRunner:
    settings: AppSettings
    llm: LLMProvider | None = None
    clock: SystemClock = SystemClock()

    async def run(self) -> int:
        sender = VrchatOscUdpSender(
            host=self.settings.osc.host,
            port=self.settings.osc.port,
            chatbox_address=self.settings.osc.chatbox_address,
            chatbox_send=self.settings.osc.chatbox_send,
            chatbox_clear=self.settings.osc.chatbox_clear,
        )
        osc = SmartOscQueue(
            sender=sender,
            clock=self.clock,
            max_chars=self.settings.osc.chatbox_max_chars,
            cooldown_s=self.settings.osc.cooldown_s,
            ttl_s=self.settings.osc.ttl_s,
        )

        flush_task = asyncio.create_task(self._flush_loop(osc))
        try:
            await self._stdin_loop(osc)
        except KeyboardInterrupt:
            return 0
        finally:
            flush_task.cancel()
            await asyncio.gather(flush_task, return_exceptions=True)
            sender.close()

        return 0

    async def _stdin_loop(self, osc: SmartOscQueue) -> None:
        loop = asyncio.get_running_loop()
        while True:
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if not line:
                return
            text = line.strip()
            if not text:
                continue
            utterance_id = uuid4()

            if self.llm is not None:
                translation = await self.llm.translate(
                    utterance_id=utterance_id,
                    text=text,
                    system_prompt=self.settings.system_prompt,
                    source_language=self.settings.languages.source_language,
                    target_language=self.settings.languages.target_language,
                )
                merged = f"{text} ({translation.text})"
            else:
                merged = text

            osc.enqueue(
                OSCMessage(utterance_id=utterance_id, text=merged, created_at=self.clock.now())
            )

    async def _flush_loop(self, osc: SmartOscQueue) -> None:
        try:
            while True:
                osc.process_due()
                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            raise
