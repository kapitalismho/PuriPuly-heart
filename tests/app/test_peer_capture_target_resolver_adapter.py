from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from puripuly_heart.app.adapters.peer_capture_target_resolver import (
    PeerCaptureTargetResolverAdapter,
)
from puripuly_heart.app.wiring import create_peer_capture_target_resolver_adapter
from puripuly_heart.core.peer_capture import (
    PeerCaptureTargetIntent,
    PeerCaptureTargetStatus,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "target",
    [
        PeerCaptureTargetIntent(kind="default_output_device"),
        PeerCaptureTargetIntent(
            kind="named_output_device",
            device_name="Named Speakers",
        ),
    ],
)
async def test_adapter_resolves_device_target_without_process_discovery(
    target: PeerCaptureTargetIntent,
) -> None:
    adapter = PeerCaptureTargetResolverAdapter(
        resolver_factory=lambda: pytest.fail("process resolver constructed")
    )

    resolution = await adapter.resolve(target)

    assert resolution.status is PeerCaptureTargetStatus.RESOLVED
    assert resolution.target is not None
    assert resolution.target.intent is target
    assert resolution.target.capture_descriptor is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("target", "expected"),
    [
        (
            PeerCaptureTargetIntent(
                kind="process",
                process_kind="generic_executable",
                executable_identity=r"c:\games\game.exe",
            ),
            ("generic_executable", r"c:\games\game.exe", None),
        ),
        (
            PeerCaptureTargetIntent(
                kind="process",
                process_kind="vrchat",
                executable_identity=r"c:\vrchat\vrchat.exe",
            ),
            ("vrchat", r"c:\vrchat\vrchat.exe", None),
        ),
        (
            PeerCaptureTargetIntent(
                kind="process",
                process_kind="discord",
                discord_channel="canary",
            ),
            ("discord", None, "canary"),
        ),
    ],
)
async def test_adapter_maps_process_intent_and_preserves_resolution(
    target: PeerCaptureTargetIntent,
    expected: tuple[str, str | None, str | None],
) -> None:
    identity = object()
    process_targets: list[object] = []
    process_resolution = SimpleNamespace(identity=identity, unavailable_reason=None)

    class Resolver:
        def resolve_for_start(self, process_target: object) -> object:
            process_targets.append(process_target)
            return process_resolution

    adapter = PeerCaptureTargetResolverAdapter(resolver_factory=Resolver)

    resolution = await adapter.resolve(target)

    assert resolution.status is PeerCaptureTargetStatus.RESOLVED
    assert resolution.target is not None
    assert resolution.target.intent is target
    assert resolution.target.capture_descriptor is process_resolution
    process_target = process_targets[0]
    assert (
        process_target.kind,
        process_target.executable_identity,
        process_target.discord_channel,
    ) == expected


@pytest.mark.asyncio
async def test_adapter_constructs_fresh_resolver_and_runs_each_resolution_off_thread() -> None:
    main_thread = threading.get_ident()
    factory_calls: list[int] = []
    resolver_threads: list[int] = []
    target = PeerCaptureTargetIntent(
        kind="process",
        process_kind="vrchat",
        executable_identity=r"c:\vrchat\vrchat.exe",
    )

    class Resolver:
        def __init__(self, sequence: int) -> None:
            self.sequence = sequence

        def resolve_for_start(self, _process_target: object) -> object:
            resolver_threads.append(threading.get_ident())
            return SimpleNamespace(identity=self.sequence, unavailable_reason=None)

    def resolver_factory() -> Resolver:
        sequence = len(factory_calls) + 1
        factory_calls.append(sequence)
        return Resolver(sequence)

    adapter = PeerCaptureTargetResolverAdapter(resolver_factory=resolver_factory)

    first = await adapter.resolve(target)
    second = await adapter.resolve(target)

    assert factory_calls == [1, 2]
    assert resolver_threads
    assert all(thread_id != main_thread for thread_id in resolver_threads)
    assert first.target is not None
    assert second.target is not None
    assert first.target.capture_descriptor.identity == 1
    assert second.target.capture_descriptor.identity == 2


@pytest.mark.asyncio
async def test_adapter_preserves_unavailable_reason_without_resolved_target() -> None:
    target = PeerCaptureTargetIntent(
        kind="process",
        process_kind="discord",
        discord_channel="stable",
    )

    class Resolver:
        def resolve_for_start(self, _process_target: object) -> object:
            return SimpleNamespace(identity=None, unavailable_reason="ambiguous")

    resolution = await PeerCaptureTargetResolverAdapter(resolver_factory=Resolver).resolve(target)

    assert resolution.status is PeerCaptureTargetStatus.UNAVAILABLE
    assert resolution.target is None
    assert resolution.reason == "ambiguous"


def test_wiring_factory_composes_fresh_process_resolver_adapter() -> None:
    adapter = create_peer_capture_target_resolver_adapter()

    assert isinstance(adapter, PeerCaptureTargetResolverAdapter)
    first = adapter.resolver_factory()
    second = adapter.resolver_factory()
    assert first is not second
    assert type(first).__name__ == "ProcessCaptureResolver"
    assert type(first.snapshots).__name__ == "PsutilCurrentUserProcessSnapshots"
