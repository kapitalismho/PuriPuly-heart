from __future__ import annotations

import ast
from pathlib import Path

from puripuly_heart.core.peer_capture import (
    PeerCaptureAdmissionPort,
    PeerCaptureProviderPort,
    PeerCaptureSessionConfig,
    PeerCaptureSessionSnapshot,
    PeerCaptureTargetResolverPort,
)
from puripuly_heart.core.runtime.peer_channel import PeerCaptureSessionOwner
from puripuly_heart.core.runtime.self_capture import SelfCaptureSessionOwner

ROOT = Path(__file__).resolve().parents[2]
OWNER_PATH = ROOT / "src" / "puripuly_heart" / "core" / "runtime" / "peer_channel.py"
CONTROLLER_PATH = ROOT / "src" / "puripuly_heart" / "ui" / "controller.py"
SOURCE_ADAPTER_PATH = (
    ROOT / "src" / "puripuly_heart" / "app" / "adapters" / "peer_capture_source.py"
)
TARGET_RESOLVER_PATH = (
    ROOT / "src" / "puripuly_heart" / "app" / "adapters" / "peer_capture_target_resolver.py"
)
VAD_ADAPTER_PATH = ROOT / "src" / "puripuly_heart" / "app" / "adapters" / "peer_capture_vad.py"


def test_peer_capture_owner_exposes_explicit_dto_port_and_lifecycle_contracts() -> None:
    assert PeerCaptureSessionConfig.__dataclass_params__.frozen is True
    assert PeerCaptureSessionSnapshot.__dataclass_params__.frozen is True
    assert PeerCaptureAdmissionPort.__module__ == "puripuly_heart.core.peer_capture"
    assert PeerCaptureTargetResolverPort.__module__ == "puripuly_heart.core.peer_capture"
    assert PeerCaptureProviderPort.__module__ == "puripuly_heart.core.peer_capture"
    assert PeerCaptureSessionOwner is not SelfCaptureSessionOwner
    assert PeerCaptureSessionOwner.lifecycle_owner_snapshot


def test_peer_capture_owner_has_no_ui_hub_or_settings_dependency() -> None:
    tree = ast.parse(OWNER_PATH.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.add(node.module)

    assert not any(module.startswith("puripuly_heart.ui") for module in imports)
    assert "puripuly_heart.core.orchestrator.hub" not in imports
    assert "puripuly_heart.config.settings" not in imports


def test_production_controller_composes_one_peer_owner_through_ports() -> None:
    tree = ast.parse(CONTROLLER_PATH.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "compose_peer_capture_session_owner"
    ]
    assert len(calls) == 1
    keywords = {keyword.arg for keyword in calls[0].keywords}
    assert {
        "admission",
        "target_resolver",
        "provider_request_factory",
        "source_factory",
        "vad_factory",
        "run_audio_loop",
        "vad_sink",
        "state_changed",
        "diagnostic_sink",
    }.issubset(keywords)
    source = CONTROLLER_PATH.read_text(encoding="utf-8")
    assert "target_resolver=create_peer_capture_target_resolver_adapter()" in source
    assert "vad_factory=create_peer_capture_vad_adapter(" in source
    assert "PeerChannelRuntime" not in source
    assert "PeerRuntimeConfig" not in source


def test_controller_does_not_construct_peer_owner_resources_outside_adapters() -> None:
    tree = ast.parse(CONTROLLER_PATH.read_text(encoding="utf-8"))
    controller = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "GuiController"
    )
    resource_methods = {
        node.name
        for node in controller.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name
        in {
            "_run_peer_audio_vad_loop",
        }
    }
    assert resource_methods == {
        "_run_peer_audio_vad_loop",
    }


def test_controller_composes_peer_capture_source_adapter_without_source_algorithm() -> None:
    source = CONTROLLER_PATH.read_text(encoding="utf-8")

    assert "source_factory=create_peer_capture_source_adapter(" in source
    assert "_create_peer_audio_source_from_runtime_config" not in source
    assert "_create_process_peer_audio_source" not in source
    assert "DesktopLoopbackAudioSource" not in source
    assert "ProcessAudioCaptureSource" not in source
    assert "DesktopPeerPipeline" not in source


def test_peer_capture_source_adapter_has_no_ui_resolution_or_lifecycle_ownership() -> None:
    source = SOURCE_ADAPTER_PATH.read_text(encoding="utf-8")

    assert "puripuly_heart.ui" not in source
    assert "GuiController" not in source
    assert "AppSettings" not in source
    assert "ProcessCaptureResolver" not in source
    assert "asyncio" not in source
    assert "async def close(" not in source
    assert "PeerCaptureSessionConfig" in source
    assert "PeerCaptureResolvedTarget" in source


def test_peer_capture_session_owner_remains_source_lifecycle_owner() -> None:
    source = OWNER_PATH.read_text(encoding="utf-8")

    assert '"_audio_source"' in source
    assert "self._retired_sources" in source
    assert "await self._close_if_possible(source)" in source
    assert "source_factory: PeerCaptureSourceFactory" in source


def test_controller_composes_peer_target_resolver_without_resolution_algorithm() -> None:
    source = CONTROLLER_PATH.read_text(encoding="utf-8")

    assert "target_resolver=create_peer_capture_target_resolver_adapter()" in source
    assert "_PeerCaptureTargetResolverAdapter" not in source
    assert "_resolve_peer_capture_target_for_owner" not in source
    assert "_process_target_from_capture_target" not in source


def test_peer_target_resolver_has_no_ui_source_or_session_lifecycle_ownership() -> None:
    source = TARGET_RESOLVER_PATH.read_text(encoding="utf-8")

    assert "puripuly_heart.ui" not in source
    assert "GuiController" not in source
    assert "AppSettings" not in source
    assert "DesktopLoopbackAudioSource" not in source
    assert "ProcessAudioCaptureSource" not in source
    assert "PeerCaptureSessionOwner" not in source
    assert "async def close(" not in source
    assert "asyncio.to_thread" in source
    assert "PeerCaptureTargetResolverPort" not in source


def test_peer_capture_session_owner_remains_target_resolution_lifecycle_caller() -> None:
    source = OWNER_PATH.read_text(encoding="utf-8")

    assert "target_resolver: PeerCaptureTargetResolverPort" in source
    assert "await self._target_resolver.resolve(config.capture_target)" in source
    assert "self._is_superseded(generation)" in source


def test_controller_composes_peer_vad_adapter_without_vad_algorithm() -> None:
    source = CONTROLLER_PATH.read_text(encoding="utf-8")

    assert "vad_factory=create_peer_capture_vad_adapter(" in source
    assert "_create_peer_vad_from_runtime_config" not in source
    assert "_peer_capture_sample_rate" not in source
    assert "create_peer_vad_gating" not in source


def test_peer_vad_adapter_has_no_ui_self_vad_or_session_lifecycle_ownership() -> None:
    source = VAD_ADAPTER_PATH.read_text(encoding="utf-8")

    assert "puripuly_heart.ui" not in source
    assert "GuiController" not in source
    assert "AppSettings" not in source
    assert "SelfCaptureSessionConfig" not in source
    assert "run_audio_vad_loop" not in source
    assert "asyncio" not in source
    assert "async def close(" not in source
    assert "PeerCaptureSessionConfig" in source


def test_peer_capture_session_owner_remains_vad_and_loop_lifecycle_owner() -> None:
    source = OWNER_PATH.read_text(encoding="utf-8")

    assert '"_vad"' in source
    assert "vad_factory: PeerCaptureVadFactory" in source
    assert "vad = self._vad_factory(config)" in source
    assert "self._run_audio_loop" in source
