from __future__ import annotations

import ast

from tests.helpers.paths import SOURCE_ROOT


def test_output_projection_owner_is_the_only_channel_output_side_effect_boundary() -> None:
    channel_source = (
        SOURCE_ROOT / "core" / "orchestrator" / "peer_translation_channel.py"
    ).read_text(encoding="utf-8")
    composition_source = (SOURCE_ROOT / "composition" / "application_runtime.py").read_text(
        encoding="utf-8"
    )
    overlay_source = (
        SOURCE_ROOT / "app" / "services" / "overlay" / "overlay_application.py"
    ).read_text(encoding="utf-8")
    settings_source = (
        SOURCE_ROOT / "app" / "services" / "settings" / "settings_runtime_effects.py"
    ).read_text(encoding="utf-8")
    output_source = (SOURCE_ROOT / "core" / "runtime" / "output.py").read_text(encoding="utf-8")
    projection_source = (
        SOURCE_ROOT / "core" / "orchestrator" / "translation_output_projection.py"
    ).read_text(encoding="utf-8")

    assert "self.overlay_sink.emit(" not in channel_source
    for side_effect in (
        "enqueue",
        "send_immediate",
        "send_typing",
        "set_typing_reason",
        "clear_typing_reasons",
        "process_due",
        "drop_pending",
    ):
        assert f"self.osc.{side_effect}(" not in channel_source
        assert f"self.osc.{side_effect}(" not in composition_source
    assert "OutputRuntime" not in channel_source
    assert "OverlayEventAdapter" not in channel_source
    assert "UIEvent(" not in channel_source
    for output_call in (
        "publish_overlay_event(",
        "replace_overlay_sink(",
        "publish_chatbox(",
        "publish_system_disclosure_chatbox(",
        "publish_system_immediate_chatbox(",
        "set_self_chatbox_typing_reason(",
        "clear_self_chatbox_typing_reasons(",
    ):
        assert f"self.output_runtime.{output_call}" in projection_source
    assert "overlay_sink.emit(" in output_source
    assert "self.chatbox.enqueue(" in output_source
    assert "self.chatbox.send_immediate(" in output_source
    assert "self.chatbox.set_typing_reason(" in output_source
    assert "self.chatbox.clear_typing_reasons(" in output_source
    assert "self.output_provider()" in overlay_source
    assert "await output.replace_overlay_sink(" in overlay_source
    assert "hub_provider" not in overlay_source
    assert "self._pipeline.translation_output_projection" in settings_source
    assert "output_projection.overlay_sink is presenter" in settings_source
    assert "hub.overlay_sink" not in settings_source
    assert "getattr(hub, 'overlay_sink'" not in settings_source


def test_flet_composition_uses_owner_without_importing_output_implementation() -> None:
    composition_source = (SOURCE_ROOT / "composition" / "application_runtime.py").read_text(
        encoding="utf-8"
    )
    ui_composition_source = (SOURCE_ROOT / "composition" / "ui_application.py").read_text(
        encoding="utf-8"
    )
    pipeline_source = (SOURCE_ROOT / "app" / "wiring" / "wiring_runtime_pipeline.py").read_text(
        encoding="utf-8"
    )
    imported_modules: set[str] = set()
    for source_file in sorted((SOURCE_ROOT / "ui").rglob("*.py")):
        tree = ast.parse(source_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported_modules.add(node.module)

    assert "_init_pipeline" not in composition_source
    for retired_assembly in (
        "_get_managed_account_components",
        "_get_provider_application_owner",
        "_get_provider_runtime_components",
        "_get_runtime_pipeline_launcher",
        "_get_self_capture_owner",
        "_get_capture_owner_factory",
    ):
        assert retired_assembly not in composition_source
    for extracted_construction in (
        "compose_managed_account(",
        "compose_provider_runtime(",
        "CaptureOwnerFactory(",
        "RuntimePipelineLauncher(",
    ):
        assert composition_source.count(extracted_construction) == 1
    assert "compose_runtime_pipeline(" in pipeline_source
    assert "PeerTranslationChannelOwner(" in pipeline_source
    assert "ClientHub" not in pipeline_source
    assert "compose_application_runtime(" in ui_composition_source
    assert "compose_managed_account(" in composition_source
    assert "compose_provider_runtime(" in composition_source
    assert "CaptureOwnerFactory(" in composition_source
    assert "RuntimePipelineLauncher(" in composition_source
    assert "RuntimeCompositionComponents(" in composition_source
    assert "output_runtime.start_ui_event_bridge(" in composition_source
    assert "hub.output_runtime.start_ui_event_bridge(" not in composition_source
    assert "puripuly_heart.core.runtime.output" not in imported_modules
    assert "puripuly_heart.core.output.router" not in imported_modules
