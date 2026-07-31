from tests.helpers.ast_sources import method_source as _method_source
from tests.helpers.paths import REPO_ROOT

COMPOSITION_PATH = REPO_ROOT / "src" / "puripuly_heart" / "composition" / "application_runtime.py"
UI_RUNTIME_PATH = REPO_ROOT / "src" / "puripuly_heart" / "app" / "adapters" / "ui_runtime.py"


def test_application_manual_typing_owner_is_only_factory_composition() -> None:
    source = COMPOSITION_PATH.read_text(encoding="utf-8")

    assert "def _begin_manual_submit_typing(" not in source
    assert "def _manual_typing_idle_task(" not in source
    assert source.count("create_manual_typing_owner(") == 1
    assert "return pipeline.translation_output_projection" in source
    assert 'getattr(hub, "set_self_chatbox_typing_reason", None)' not in source
    assert '"clear_self_chatbox_typing_reasons",' not in source
    assert 'getattr(runtime, "translation_tasks", None)' in source
    assert "idle_timeout_seconds=MANUAL_INPUT_TYPING_IDLE_TIMEOUT_S" in source
    assert "submit_timeout_seconds=MANUAL_SUBMIT_TYPING_TIMEOUT_S" in source


def test_ui_manual_submit_preserves_self_source() -> None:
    method = _method_source(UI_RUNTIME_PATH, "UiInputRuntimeAdapter", "submit_text")

    assert 'owner.submit_text(text, source="You")' in method
