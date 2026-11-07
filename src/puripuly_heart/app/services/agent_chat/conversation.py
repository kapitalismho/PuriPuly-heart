from __future__ import annotations

import json
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Literal

from puripuly_heart.app.ports.agent_chat import AgentMessage


@dataclass(slots=True)
class _Group:
    item_count: int = 0
    characters: int = 0
    messages: list[int] = field(default_factory=list)


class AgentConversation:
    def __init__(self, *, token_limit: int = 48_000) -> None:
        self.items: list[dict] = []
        self.messages: list[AgentMessage] = []
        self.archive: list[dict] = []
        self._groups: deque[_Group] = deque()
        self._token_limit = token_limit

    @property
    def active_messages(self) -> list[AgentMessage]:
        return [self.messages[index] for group in self._groups for index in group.messages]

    def begin(self, *, continuation: bool = False) -> None:
        if not continuation or not self._groups:
            self._groups.append(_Group())

    def record(self, item: dict) -> None:
        if item.get("type") not in {"reasoning", "configuration_update"}:
            self.archive.append(deepcopy(item))

    def append(self, item: dict, *, archive: bool = True) -> None:
        if not self._groups:
            self.begin()
        self.items.append(deepcopy(item))
        group = self._groups[-1]
        group.item_count += 1
        group.characters += len(json.dumps(item, ensure_ascii=False))
        if archive:
            self.record(item)

    def display(
        self, role: Literal["user", "assistant", "notice"], text: str, *, grouped: bool = True
    ) -> int:
        index = len(self.messages)
        self.messages.append(AgentMessage(role, text))
        if grouped and self._groups:
            self._groups[-1].messages.append(index)
        return index

    def attach(self, index: int) -> None:
        self._groups[-1].messages.append(index)

    def trim(self) -> None:
        characters = sum(group.characters for group in self._groups)
        while len(self._groups) > 1 and characters > self._token_limit * 4:
            group = self._groups.popleft()
            del self.items[: group.item_count]
            characters -= group.characters
            for index in group.messages:
                self.messages[index] = replace(self.messages[index], archived=True)

    def reset(self, separator: str) -> None:
        self.items.clear()
        self._groups.clear()
        self.messages[:] = [replace(message, archived=True) for message in self.messages]
        self.display("notice", separator)
        self.record({"role": "system", "content": separator, "type": "conversation_separator"})
