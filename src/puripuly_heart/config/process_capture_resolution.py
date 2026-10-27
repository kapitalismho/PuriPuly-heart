from __future__ import annotations

import ntpath
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal, Protocol

from puripuly_heart.config.process_capture_platform import (
    ProcessCapturePlatformAvailability,
    get_process_capture_platform_availability,
)
from puripuly_heart.config.settings_vnext.schema import ProcessCaptureTargetIntent

ProcessCaptureUnavailableReason = Literal[
    "unsupported_platform",
    "no_process",
    "ambiguous",
    "ineligible",
]


_DISCORD_CHANNEL_BY_BASENAME = {
    "discord.exe": "stable",
    "discordptb.exe": "ptb",
    "discordcanary.exe": "canary",
}


@dataclass(frozen=True, slots=True)
class ProcessSnapshot:
    pid: int
    parent_pid: int | None
    is_current_user: bool
    executable_path: str | None
    instance_id: str | None


class CurrentUserProcessSnapshotPort(Protocol):
    def snapshots(self) -> Iterable[ProcessSnapshot]: ...


@dataclass(frozen=True, slots=True)
class ProcessCaptureCandidate:
    name: str
    target: ProcessCaptureTargetIntent
    enabled: bool


@dataclass(frozen=True, slots=True)
class ResolvedProcessCaptureIdentity:
    pid: int
    target: ProcessCaptureTargetIntent
    instance_id: str

    def __post_init__(self) -> None:
        if self.pid <= 0:
            raise ValueError("resolved process PID must be positive")
        if not self.instance_id:
            raise ValueError("resolved process instance identity must be non-empty")


@dataclass(frozen=True, slots=True)
class ProcessCaptureResolution:
    identity: ResolvedProcessCaptureIdentity | None
    unavailable_reason: ProcessCaptureUnavailableReason | None

    @property
    def available(self) -> bool:
        return self.identity is not None

    @property
    def pid(self) -> int | None:
        return self.identity.pid if self.identity is not None else None


@dataclass(frozen=True, slots=True)
class ProcessCaptureResolver:
    snapshots: CurrentUserProcessSnapshotPort
    platform_availability: Callable[[], ProcessCapturePlatformAvailability] = (
        get_process_capture_platform_availability
    )

    def enumerate_candidates(self) -> tuple[ProcessCaptureCandidate, ...]:
        if not self.platform_availability().available:
            return ()
        inventory = _inventory(self.snapshots.snapshots())
        return tuple(
            ProcessCaptureCandidate(
                name=_candidate_name(target, roots),
                target=target,
                enabled=len(roots) == 1,
            )
            for target, roots in sorted(inventory.eligible_roots.items(), key=_candidate_sort_key)
        )

    def resolve_for_start(self, target: ProcessCaptureTargetIntent) -> ProcessCaptureResolution:
        return self._resolve(target)

    def resolve_for_retry(self, target: ProcessCaptureTargetIntent) -> ProcessCaptureResolution:
        return self._resolve(target)

    def _resolve(self, target: ProcessCaptureTargetIntent) -> ProcessCaptureResolution:
        if not self.platform_availability().available:
            return ProcessCaptureResolution(
                identity=None, unavailable_reason="unsupported_platform"
            )
        inventory = _inventory(self.snapshots.snapshots())
        roots = inventory.eligible_roots.get(target, ())
        if len(roots) > 1:
            return ProcessCaptureResolution(identity=None, unavailable_reason="ambiguous")
        if len(roots) == 1:
            root = roots[0]
            return ProcessCaptureResolution(
                identity=ResolvedProcessCaptureIdentity(
                    pid=root.pid,
                    target=target,
                    instance_id=root.instance_id or "",
                ),
                unavailable_reason=None,
            )
        if target in inventory.ineligible_targets:
            return ProcessCaptureResolution(identity=None, unavailable_reason="ineligible")
        return ProcessCaptureResolution(identity=None, unavailable_reason="no_process")


@dataclass(frozen=True, slots=True)
class _Inventory:
    eligible_roots: dict[ProcessCaptureTargetIntent, tuple[ProcessSnapshot, ...]]
    ineligible_targets: frozenset[ProcessCaptureTargetIntent]


def _inventory(snapshots: Iterable[ProcessSnapshot]) -> _Inventory:
    all_snapshots = tuple(snapshot for snapshot in snapshots if snapshot.pid > 0)
    by_pid = {snapshot.pid: snapshot for snapshot in all_snapshots}
    identities = {snapshot.pid: _target_for_snapshot(snapshot) for snapshot in all_snapshots}
    ineligible_targets: set[ProcessCaptureTargetIntent] = set()
    eligible: list[tuple[ProcessSnapshot, ProcessCaptureTargetIntent]] = []

    for snapshot in all_snapshots:
        target = identities[snapshot.pid]
        if target is None:
            continue
        if (
            not snapshot.is_current_user
            or not snapshot.instance_id
            or _is_excluded(snapshot.executable_path)
        ):
            ineligible_targets.add(target)
            continue
        if _same_identity_ancestor(snapshot, target, by_pid, identities):
            continue
        eligible.append((snapshot, target))

    grouped: dict[ProcessCaptureTargetIntent, list[ProcessSnapshot]] = {}
    for snapshot, target in eligible:
        grouped.setdefault(target, []).append(snapshot)
    return _Inventory(
        eligible_roots={target: tuple(roots) for target, roots in grouped.items()},
        ineligible_targets=frozenset(ineligible_targets),
    )


def _same_identity_ancestor(
    snapshot: ProcessSnapshot,
    target: ProcessCaptureTargetIntent,
    by_pid: dict[int, ProcessSnapshot],
    identities: dict[int, ProcessCaptureTargetIntent | None],
) -> bool:
    parent_pid = snapshot.parent_pid
    visited = {snapshot.pid}
    while parent_pid is not None and parent_pid not in visited:
        visited.add(parent_pid)
        parent = by_pid.get(parent_pid)
        if parent is None:
            return False
        if identities[parent.pid] == target:
            return True
        parent_pid = parent.parent_pid
    return False


def _target_for_snapshot(snapshot: ProcessSnapshot) -> ProcessCaptureTargetIntent | None:
    path = snapshot.executable_path
    if not isinstance(path, str) or not path.strip():
        return None
    basename = ntpath.basename(path).casefold()
    try:
        if basename == "vrchat.exe":
            return ProcessCaptureTargetIntent.vrchat(path)
        channel = _DISCORD_CHANNEL_BY_BASENAME.get(basename)
        if channel is not None:
            return ProcessCaptureTargetIntent.discord(channel)
        return ProcessCaptureTargetIntent.generic_executable(path)
    except ValueError:
        return None


def _is_excluded(executable_path: str | None) -> bool:
    if not isinstance(executable_path, str):
        return True
    basename = ntpath.basename(executable_path).casefold()
    if not basename:
        return True
    if basename.startswith("discord") and ("development" in basename or "internal" in basename):
        return True
    if basename in {
        "install.exe",
        "launch.exe",
        "steam.exe",
        "steamservice.exe",
        "steamwebhelper.exe",
    }:
        return True
    if basename.endswith("launcher.exe"):
        return True
    return any(
        token in basename for token in ("updater", "update", "installer", "setup", "uninstall")
    )


def _candidate_name(
    target: ProcessCaptureTargetIntent,
    roots: tuple[ProcessSnapshot, ...],
) -> str:
    if target.kind == "vrchat":
        name = "VRChat"
    elif target.kind == "discord":
        labels = {"stable": "Discord Stable", "ptb": "Discord PTB", "canary": "Discord Canary"}
        name = labels[target.discord_channel or "stable"]
    else:
        name = ntpath.splitext(ntpath.basename(roots[0].executable_path or ""))[0]
    return f"{name} ({len(roots)})" if len(roots) > 1 else name


def _candidate_sort_key(
    item: tuple[ProcessCaptureTargetIntent, tuple[ProcessSnapshot, ...]],
) -> tuple[int, str, str]:
    target, _roots = item
    if target.kind == "vrchat":
        return (0, "", "")
    if target.kind == "discord":
        order = {"stable": "0", "ptb": "1", "canary": "2"}
        return (1, order[target.discord_channel or "stable"], "")
    return (
        2,
        _candidate_name(target, _roots).casefold(),
        target.executable_identity or "",
    )
