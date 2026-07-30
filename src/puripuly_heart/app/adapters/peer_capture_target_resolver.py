from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from puripuly_heart.config.settings_vnext.schema import ProcessCaptureTargetIntent
from puripuly_heart.core.peer_capture import (
    PeerCaptureResolvedTarget,
    PeerCaptureTargetIntent,
    PeerCaptureTargetResolution,
    PeerCaptureTargetStatus,
)


class PeerCaptureProcessResolution(Protocol):
    identity: object | None
    unavailable_reason: str | None


class PeerCaptureProcessResolver(Protocol):
    def resolve_for_start(
        self,
        target: ProcessCaptureTargetIntent,
    ) -> PeerCaptureProcessResolution: ...


PeerCaptureProcessResolverFactory = Callable[[], PeerCaptureProcessResolver]


@dataclass(frozen=True, slots=True)
class PeerCaptureTargetResolverAdapter:
    resolver_factory: PeerCaptureProcessResolverFactory

    async def resolve(
        self,
        target: PeerCaptureTargetIntent,
    ) -> PeerCaptureTargetResolution:
        if target.kind != "process":
            return PeerCaptureTargetResolution(
                PeerCaptureTargetStatus.RESOLVED,
                target=PeerCaptureResolvedTarget(intent=target),
            )
        process_target = self._process_target(target)
        resolution = await asyncio.to_thread(
            lambda: self.resolver_factory().resolve_for_start(process_target)
        )
        if resolution.identity is None:
            return PeerCaptureTargetResolution(
                PeerCaptureTargetStatus.UNAVAILABLE,
                reason=resolution.unavailable_reason,
            )
        return PeerCaptureTargetResolution(
            PeerCaptureTargetStatus.RESOLVED,
            target=PeerCaptureResolvedTarget(
                intent=target,
                capture_descriptor=resolution,
            ),
        )

    @staticmethod
    def _process_target(target: PeerCaptureTargetIntent) -> ProcessCaptureTargetIntent:
        if target.kind != "process" or target.process_kind is None:
            raise ValueError("process peer source requires a process capture target")
        if target.process_kind == "discord":
            return ProcessCaptureTargetIntent.discord(target.discord_channel or "")
        if target.process_kind == "vrchat":
            return ProcessCaptureTargetIntent.vrchat(target.executable_identity or "")
        return ProcessCaptureTargetIntent.generic_executable(target.executable_identity or "")


__all__ = [
    "PeerCaptureProcessResolution",
    "PeerCaptureProcessResolver",
    "PeerCaptureProcessResolverFactory",
    "PeerCaptureTargetResolverAdapter",
]
