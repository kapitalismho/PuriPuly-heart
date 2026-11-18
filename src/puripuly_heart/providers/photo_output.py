from __future__ import annotations

from dataclasses import dataclass, field

from puripuly_heart.core.photo.protocol import PhotoProviderError


def _invalid() -> PhotoProviderError:
    return PhotoProviderError("photo_stream_invalid", retryable=False)


def _phase(item: dict) -> str | None:
    phase = item.get("phase")
    if phase not in (None, "commentary", "final_answer"):
        raise _invalid()
    return phase


def _identity(index: object, identifier: object) -> tuple[int, str]:
    if type(index) is not int or index < 0 or not isinstance(identifier, str) or not identifier:
        raise _invalid()
    return index, identifier


@dataclass(slots=True)
class _Item:
    identifier: str
    kind: str
    phase: str | None
    done: bool = False
    pending: list[str] = field(default_factory=list)


class PhotoResponseOutput:
    def __init__(self) -> None:
        self._items: dict[int, _Item] = {}
        self._selected: int | None = None
        self._anonymous: list[str] = []
        self._anonymous_identity: tuple[int, str] | None = None
        self._anonymous_seen = False

    def accept(self, event: dict) -> str:
        kind = event.get("type")
        if kind in ("response.output_item.added", "response.output_item.done"):
            item = event.get("item")
            if not isinstance(item, dict) or not isinstance(item.get("type"), str):
                raise _invalid()
            index, identifier = _identity(event.get("output_index"), item.get("id"))
            phase = _phase(item) if item["type"] == "message" else None
            if item["type"] == "message" and item.get("role") != "assistant":
                raise _invalid()
            if kind == "response.output_item.added":
                if (
                    self._anonymous_seen
                    or index in self._items
                    or any(value.identifier == identifier for value in self._items.values())
                    or self._items
                    and index <= max(self._items)
                ):
                    raise _invalid()
                self._items[index] = _Item(identifier, item["type"], phase)
            else:
                previous = self._items.get(index)
                if (
                    previous is None
                    or previous.done
                    or previous.identifier != identifier
                    or previous.kind != item["type"]
                    or previous.phase is not None
                    and phase is not None
                    and previous.phase != phase
                ):
                    raise _invalid()
                phase = previous.phase if phase is None else phase
                previous.phase = phase
                previous.done = True
            if phase == "final_answer":
                if self._selected is not None and self._selected != index:
                    raise _invalid()
                self._selected = index
                pending = self._items[index].pending
                text = "".join(pending)
                pending.clear()
                return text
            return ""
        if kind != "response.output_text.delta":
            return ""
        delta = event.get("delta")
        if not isinstance(delta, str):
            raise _invalid()
        if "content_index" in event and (
            type(event["content_index"]) is not int or event["content_index"] != 0
        ):
            raise _invalid()
        identity = (
            _identity(event.get("output_index"), event.get("item_id"))
            if "output_index" in event or "item_id" in event
            else None
        )
        if not self._items:
            if self._anonymous_seen and identity != self._anonymous_identity:
                raise _invalid()
            self._anonymous_seen = True
            self._anonymous_identity = identity
            self._anonymous.append(delta)
            return ""
        if identity is None:
            raise _invalid()
        index, identifier = identity
        item = self._items.get(index)
        if item is None or item.identifier != identifier or item.kind != "message" or item.done:
            raise _invalid()
        if item.phase == "final_answer":
            return delta
        if item.phase is None:
            item.pending.append(delta)
        return ""

    def finish(self, response: dict) -> str:
        output = response.get("output")
        if output is not None and not isinstance(output, list):
            raise _invalid()
        messages: list[tuple[int, dict]] = []
        for index, item in enumerate(output or []):
            if not isinstance(item, dict):
                raise _invalid()
            if item.get("type") == "message":
                if item.get("role") != "assistant":
                    raise _invalid()
                _phase(item)
                messages.append((index, item))
        if self._items and output:
            terminal_messages = {index for index, _ in messages}
            if terminal_messages != {
                index for index, item in self._items.items() if item.kind == "message"
            }:
                raise _invalid()
            for index, terminal in messages:
                item = self._items[index]
                phase = _phase(terminal)
                if item.identifier != terminal.get("id") or (
                    item.phase is not None and phase is not None and item.phase != phase
                ):
                    raise _invalid()
                item.phase = item.phase if phase is None else phase
        if self._items:
            candidates = [
                index for index, item in self._items.items() if item.phase == "final_answer"
            ]
            if not candidates:
                candidates = [
                    index
                    for index, item in self._items.items()
                    if item.kind == "message" and item.phase is None
                ]
                if len([item for item in self._items.values() if item.kind == "message"]) != 1:
                    raise _invalid()
            if len(candidates) != 1 or (
                self._selected is not None and candidates[0] != self._selected
            ):
                raise _invalid()
            pending = self._items[candidates[0]].pending
            text = "".join(pending)
            pending.clear()
            return text
        if messages:
            if len(messages) != 1 or _phase(messages[0][1]) == "commentary":
                raise _invalid()
            index, item = messages[0]
            if self._anonymous_identity is not None and self._anonymous_identity != (
                index,
                item.get("id"),
            ):
                raise _invalid()
        text = "".join(self._anonymous)
        self._anonymous.clear()
        return text
