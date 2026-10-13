from __future__ import annotations

from collections import OrderedDict

from puripuly_heart.domain.events import UIEventType
from puripuly_heart.domain.models import Transcript, Translation


class QuickChatHistory:
    def __init__(self, limit=100):
        self.limit = limit
        self.records = OrderedDict()
        self.retired = OrderedDict()
        self.sequence = 0

    def accept(self, event, *, translation_enabled):
        identity = event.utterance_id or getattr(event.payload, "utterance_id", None)
        if identity is None:
            return None
        key = f"{event.channel}:{identity}"
        if key in self.retired:
            return None
        existing = self.records.get(key)
        row = (
            dict(existing)
            if existing
            else dict(id=key, role=event.channel, original="", translation="", state="original")
        )
        payload = event.payload
        if event.type == UIEventType.TRANSCRIPT_FINAL and isinstance(payload, Transcript):
            if not payload.text.strip():
                return None
            row["original"] = payload.text[:8000]
            if not existing:
                row["state"] = "pending" if translation_enabled else "original"
        elif event.type == UIEventType.TRANSLATION_DONE and isinstance(payload, Translation):
            row.update(
                original=(payload.source_text or row["original"])[:8000],
                translation=payload.translated_text[:8000],
                state="complete",
            )
        elif event.type == UIEventType.ERROR and existing and row["state"] == "pending":
            row["state"] = "failed"
        else:
            return None
        if row == existing:
            return None
        if not existing:
            self.sequence += 1
            row["sequence"] = self.sequence
        self.records[key] = row
        while len(self.records) > self.limit:
            removed, _ = self.records.popitem(last=False)
            self.retired[removed] = None
        while len(self.retired) > 512:
            self.retired.popitem(last=False)
        return row


def chat_text(row, *, peer_mode, translation_enabled):
    original = row["original"]
    translated = row.get("translation", "")
    if not translation_enabled or row.get("state") == "original":
        return original
    if not translated:
        return original
    mode = peer_mode if row["role"] == "peer" else "both"
    if mode == "original":
        return original
    if mode == "translation":
        return translated
    return original + "\n→ " + translated
