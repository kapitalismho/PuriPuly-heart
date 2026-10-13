from __future__ import annotations

import asyncio
import json
from collections import OrderedDict
from uuid import UUID

from puripuly_heart.app.services.manual_typing import ManualTranslationFailed
from puripuly_heart.app.services.quick_chat import QuickChatHistory
from puripuly_heart.core.feature_diagnostics import log_feature

FRAME_PREFIX = "PURIPULY_QUICK:"
MAX_FRAME_BYTES = 65536


class QuickTranslateOwner:
    def __init__(
        self, *, spawn, submit, preferences, save_preferences, stop=None, on_visibility=None
    ):
        self.spawn = spawn
        self.submit = submit
        self.preferences = preferences
        self.save_preferences = save_preferences
        self.stop = stop
        self.process = None
        self.reader_task = None
        self.tasks = set()
        self.closed = False
        self.visible = False
        self.visibility_revision = 0
        self.lock = asyncio.Lock()
        self.write_lock = asyncio.Lock()
        self.seen = OrderedDict()
        self.on_visibility = on_visibility
        self.history = QuickChatHistory()
        self.outbox = OrderedDict()
        self.delivery_task = None

    def set_visible_state(self, visible):
        self.visible = visible
        if self.on_visibility is not None:
            self.on_visibility(visible)

    def publish_event(self, event):
        if self.closed:
            return
        preferences = self.preferences()
        row = self.history.accept(
            event,
            translation_enabled=bool(
                preferences.get(
                    (
                        "peer_translation_enabled"
                        if event.channel == "peer"
                        else "translation_enabled"
                    ),
                    False,
                )
            ),
        )
        if row is not None:
            if row["state"] == "failed":
                log_feature("QuickTranslate", "translation", "translation_failed")
            self.enqueue(row["id"], dict(type="chat", record=row))

    def sync_context(self):
        if not self.closed:
            self.enqueue("context", dict(type="context", values=self.preferences()))

    def enqueue(self, key, message):
        if self.process is None or self.process.returncode is not None:
            return
        self.outbox[key] = message
        while len(self.outbox) > 101:
            self.outbox.popitem(last=False)
        if self.delivery_task is None or self.delivery_task.done():
            self.delivery_task = asyncio.create_task(self.deliver(self.process))

    async def deliver(self, process):
        try:
            while self.outbox and self.process is process and not self.closed:
                _, message = self.outbox.popitem(last=False)
                await asyncio.wait_for(self.send(message), 3)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            log_feature("QuickTranslate", "window", "window_connection_failed", error=error)

    async def open(self, visible=None):
        async with self.lock:
            if self.closed:
                return
            if visible is False and (self.process is None or self.process.returncode is not None):
                self.set_visible_state(False)
                return
            try:
                if self.process is None or self.process.returncode is not None:
                    if self.reader_task is not None:
                        await asyncio.gather(self.reader_task, return_exceptions=True)
                    if self.delivery_task is not None:
                        self.delivery_task.cancel()
                        await asyncio.gather(self.delivery_task, return_exceptions=True)
                    self.process = await self.spawn()
                    self.set_visible_state(False)
                    self.outbox.clear()
                    self.reader_task = asyncio.create_task(self.read_messages(self.process))
                    for row in self.history.records.values():
                        self.enqueue(row["id"], dict(type="chat", record=row))
                requested_visible = not self.visible if visible is None else bool(visible)
                self.visibility_revision += 1
                message = dict(
                    type="show" if requested_visible else "hide",
                    visibility_revision=self.visibility_revision,
                )
                if requested_visible:
                    message["preferences"] = self.preferences()
                await asyncio.wait_for(self.send(message), 3)
                self.set_visible_state(requested_visible)
            except Exception as error:
                log_feature("QuickTranslate", "window", "window_start_failed", error=error)
                raise

    async def send(self, message):
        process = self.process
        if process is None or process.returncode is not None:
            return
        data = (json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8")
        if len(data) > MAX_FRAME_BYTES:
            raise ValueError("Window message too large")
        async with self.write_lock:
            process.stdin.write(data)
            await process.stdin.drain()

    async def read_messages(self, process):
        try:
            while not self.closed:
                line = await process.stdout.readline()
                if not line:
                    break
                if len(line) > MAX_FRAME_BYTES:
                    raise ValueError("Window message too large")
                line = line.decode("utf-8")
                if not line.startswith(FRAME_PREFIX):
                    continue
                message = json.loads(line[len(FRAME_PREFIX) :])
                if message.get("type") == "submit":
                    identity = str(UUID(message["id"]))
                    text = message["text"]
                    if (
                        not isinstance(text, str)
                        or not text.strip()
                        or len(text) > 8000
                        or identity in self.seen
                    ):
                        continue
                    if len(self.tasks) >= 8:
                        await self.send(dict(type="result", id=identity, code="busy"))
                        continue
                    self.seen[identity] = None
                    if len(self.seen) > 512:
                        self.seen.popitem(last=False)
                    task = asyncio.create_task(self.translate(identity, text, process))
                    self.tasks.add(task)
                    task.add_done_callback(self.tasks.discard)
                elif message.get("type") == "visibility":
                    if (
                        self.process is process
                        and message.get("visibility_revision") == self.visibility_revision
                        and isinstance(message.get("visible"), bool)
                    ):
                        self.set_visible_state(message["visible"])
                elif message.get("type") == "preferences":
                    try:
                        await self.save_preferences(message["values"])
                        self.sync_context()
                    except Exception as error:
                        log_feature(
                            "QuickTranslate", "settings", "preferences_save_failed", error=error
                        )
                elif message.get("type") == "failure":
                    code = (
                        "scroll_failed"
                        if message.get("code") == "scroll_failed"
                        else "window_failed"
                    )
                    log_feature("QuickTranslate", "presentation", code)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            log_feature("QuickTranslate", "window", "window_connection_failed", error=error)
        finally:
            if self.process is process:
                self.set_visible_state(False)
            if process.returncode is None and not self.closed:
                log_feature("QuickTranslate", "window", "window_closed_unexpectedly")
                if self.stop is not None:
                    await self.stop(process)
                else:
                    process.terminate()
            await process.wait()

    async def translate(self, identity, text, process):
        try:
            utterance_id, translation = await self.submit(text)
            result = dict(
                type="result", id=identity, utterance_id=utterance_id, translation=translation
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            code = (
                "backend_not_ready"
                if isinstance(error, RuntimeError) and str(error) == "backend_not_ready"
                else "translation_failed"
            )
            log_feature(
                "QuickTranslate",
                "submission" if code == "backend_not_ready" else "translation",
                code,
                error=error,
            )
            result = dict(type="result", id=identity, code=code)
            if isinstance(error, ManualTranslationFailed):
                result["utterance_id"] = error.utterance_id
        if self.process is process and not self.closed:
            try:
                await self.send(result)
            except Exception as error:
                log_feature("QuickTranslate", "window", "window_connection_failed", error=error)

    async def close(self):
        async with self.lock:
            self.closed = True
        self.set_visible_state(False)
        if self.delivery_task is not None:
            self.delivery_task.cancel()
            await asyncio.gather(self.delivery_task, return_exceptions=True)
        self.outbox.clear()
        process = self.process
        if process is not None and process.returncode is None:
            try:
                await asyncio.wait_for(self.send(dict(type="exit")), 3)
                await asyncio.wait_for(process.wait(), 5)
            except Exception as error:
                log_feature("QuickTranslate", "shutdown", "window_connection_failed", error=error)
                if self.stop is not None:
                    await self.stop(process)
                else:
                    process.kill()
                    await process.wait()
        if self.reader_task is not None:
            await asyncio.gather(self.reader_task, return_exceptions=True)
        for task in tuple(self.tasks):
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks.clear()
