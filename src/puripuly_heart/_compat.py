"""Private import-compatibility support for moved modules."""

from __future__ import annotations

import sys
from importlib import import_module
from types import ModuleType

__all__ = ["MovedModuleAlias", "install_moved_module_aliases"]


class MovedModuleAlias(ModuleType):
    _targets: dict[str, tuple[str, str, str]] = {}

    def _materialize(self) -> ModuleType:
        new_name, parent_name, short = type(self)._targets[self.__name__]
        real = import_module(new_name)
        sys.modules[self.__name__] = real
        parent = sys.modules.get(parent_name)
        if parent is not None:
            setattr(parent, short, real)
        return real

    def __getattr__(self, item: str) -> object:
        return getattr(self._materialize(), item)

    def __setattr__(self, key: str, value: object) -> None:
        setattr(self._materialize(), key, value)

    def __delattr__(self, key: str) -> None:
        delattr(self._materialize(), key)

    def __dir__(self) -> list[str]:
        return dir(self._materialize())


def install_moved_module_aliases(parent_name: str, mapping: dict[str, str]) -> None:
    parent = sys.modules[parent_name]
    for short, new_name in mapping.items():
        old_name = f"{parent_name}.{short}"
        MovedModuleAlias._targets[old_name] = (new_name, parent_name, short)
        alias = MovedModuleAlias(old_name)
        sys.modules[old_name] = alias
        setattr(parent, short, alias)
