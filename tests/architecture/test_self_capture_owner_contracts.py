from __future__ import annotations

import ast
from pathlib import Path

from puripuly_heart.core.runtime.self_capture import SelfCaptureSessionOwner
from puripuly_heart.core.self_capture import (
    SelfCaptureAdmissionPort,
    SelfCaptureProviderPort,
    SelfCaptureSessionConfig,
    SelfCaptureSessionSnapshot,
)

ROOT = Path(__file__).resolve().parents[2]
OWNER_PATH = ROOT / "src" / "puripuly_heart" / "core" / "runtime" / "self_capture.py"
CONTROLLER_PATH = ROOT / "src" / "puripuly_heart" / "ui" / "controller.py"
LEGACY_OWNER_PATH = ROOT / "src" / "puripuly_heart" / "core" / "runtime" / "self_audio.py"


def test_self_capture_owner_exposes_explicit_dto_port_and_lifecycle_contracts() -> None:
    assert SelfCaptureSessionConfig.__dataclass_params__.frozen is True
    assert SelfCaptureSessionSnapshot.__dataclass_params__.frozen is True
    assert SelfCaptureAdmissionPort.__module__ == "puripuly_heart.core.self_capture"
    assert SelfCaptureProviderPort.__module__ == "puripuly_heart.core.self_capture"
    assert SelfCaptureSessionOwner.resource_fields == (
        "_source",
        "_vad",
        "_loop_task",
        "_transition_task",
        "_fault_tasks",
        "_retired_sources",
        "_generation",
    )
    snapshot = SelfCaptureSessionOwner.lifecycle_owner_snapshot
    assert callable(snapshot)


def test_self_capture_owner_has_no_ui_or_hub_dependency() -> None:
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


def test_controller_has_no_legacy_self_capture_lifecycle() -> None:
    tree = ast.parse(CONTROLLER_PATH.read_text(encoding="utf-8"))
    controller = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "GuiController"
    )
    methods = {
        node.name
        for node in controller.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert not LEGACY_OWNER_PATH.exists()
    assert methods.isdisjoint(
        {
            "_get_self_audio_runtime",
            "_sync_self_audio_runtime_aliases",
            "_adopt_self_audio_legacy_aliases",
            "_start_mic_loop",
            "_stop_mic_loop",
            "_run_mic_loop",
            "_drain_self_stt_for_toggle_off",
        }
    )
