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
    def __init__(self, *, token_limit: int = 48_000, retained_tokens: int = 18_000) -> None:
        self.items: list[dict] = []
        self.messages: list[AgentMessage] = []
        self.archive: list[dict] = []
        self._groups: deque[_Group] = deque()
        self._token_limit = token_limit
        self._retained_tokens = retained_tokens
        self._module_contexts: dict[str, tuple[dict, _Group]] = {}
        self._core_fields: dict[str, tuple[object, _Group]] = {}
        self._catalogs: dict[str, tuple[dict, _Group]] = {}

    @property
    def active_messages(self) -> list[AgentMessage]:
        return [self.messages[index] for group in self._groups for index in group.messages]

    def begin(self, *, continuation: bool = False) -> None:
        if not continuation or not self._groups:
            self._groups.append(_Group())

    def record(self, item: dict) -> None:
        if item.get("type") not in {"reasoning", "configuration_update"}:
            self.archive.append(deepcopy(item))

    def project_context(self, context: dict, *, force_modules: tuple[str, ...] = ()) -> dict:
        if context.get("missing") or context.get("view") == "diagnostics":
            return deepcopy(context)
        result = {
            key: deepcopy(context[key])
            for key in ("observed_at", "observed_monotonic", "modules", "missing", "view")
            if key in context
        }
        catalogs = {
            module: deepcopy(catalog)
            for module, catalog in context.get("catalogs", {}).items()
            if module not in self._catalogs or catalog != self._catalogs[module][0]
        }
        if catalogs:
            result["catalogs"] = catalogs
        unchanged = []
        for module in context["modules"]:
            if module not in context:
                continue
            if module == "core":
                changes = self._core_changes(context["core"], forced=module in force_modules)
                if changes:
                    result["core"] = changes
                else:
                    unchanged.append(module)
                continue
            baseline = self._module_contexts.get(module)
            if (
                module not in force_modules
                and baseline is not None
                and context[module] == baseline[0]
            ):
                unchanged.append(module)
            else:
                result[module] = deepcopy(context[module])
        if unchanged:
            result["unchanged"] = unchanged
        return result

    def _core_changes(self, core: dict, *, forced: bool) -> dict:
        changes = {
            key: deepcopy(value)
            for key, value in core.items()
            if forced or key not in self._core_fields or value != self._core_fields[key][0]
        }
        changes.update(
            {
                key: None
                for key, (value, _) in self._core_fields.items()
                if key not in core and value is not None
            }
        )
        return changes

    def append(self, item: dict, *, archive: bool = True, context: dict | None = None) -> None:
        if not self._groups:
            self.begin()
        self.items.append(deepcopy(item))
        group = self._groups[-1]
        group.item_count += 1
        group.characters += len(json.dumps(item, ensure_ascii=False))
        if archive:
            self.record(item)
        if context is not None:
            self._remember_context(context, group)

    def _remember_context(self, context: dict, group: _Group) -> None:
        if context.get("view") == "diagnostics":
            return
        if context.get("missing"):
            for module in context["modules"]:
                self._module_contexts.pop(module, None)
                self._catalogs.pop(module, None)
            if "core" in context["modules"]:
                self._core_fields.clear()
            return
        for module, catalog in context.get("catalogs", {}).items():
            self._catalogs[module] = (deepcopy(catalog), group)
        for module in context["modules"]:
            if module == "core" and module in context:
                for key, value in context["core"].items():
                    self._core_fields[key] = (deepcopy(value), group)
            elif module in context:
                self._module_contexts[module] = (deepcopy(context[module]), group)

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
        if characters <= self._token_limit * 4:
            return
        retained = min(self._retained_tokens, self._token_limit) * 4
        while len(self._groups) > 1 and characters > retained:
            group = self._groups.popleft()
            del self.items[: group.item_count]
            characters -= group.characters
            for baselines in (self._module_contexts, self._core_fields, self._catalogs):
                for module in [
                    module for module, (_, anchor) in baselines.items() if anchor is group
                ]:
                    del baselines[module]
            for index in group.messages:
                self.messages[index] = replace(self.messages[index], archived=True)

    def clear_model_context(self) -> None:
        self.items.clear()
        self._groups.clear()
        self._module_contexts.clear()
        self._core_fields.clear()
        self._catalogs.clear()

    def reset(self, separator: str) -> None:
        self.clear_model_context()
        self.messages[:] = [replace(message, archived=True) for message in self.messages]
        self.display("notice", separator)
        self.record({"role": "system", "content": separator, "type": "conversation_separator"})
