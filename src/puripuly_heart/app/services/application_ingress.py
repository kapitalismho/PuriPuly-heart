from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ApplicationIngressGate:
    frozen: bool = False

    def freeze(self) -> None:
        self.frozen = True


__all__ = ["ApplicationIngressGate"]
