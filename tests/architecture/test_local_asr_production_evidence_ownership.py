import inspect

from puripuly_heart.composition.local_asr_production_evidence import (
    compose_local_asr_production_evidence,
)
from tests.helpers.ast_sources import assert_no_forbidden_imports
from tests.helpers.paths import SOURCE_ROOT


def test_local_asr_evidence_driver_and_port_do_not_import_ui() -> None:
    for module in (
        "app/ports/local_asr_production_evidence.py",
        "release_evidence/local_asr_production_composition.py",
    ):
        assert_no_forbidden_imports(SOURCE_ROOT / module, ("flet", "puripuly_heart.ui"))


def test_local_asr_evidence_composition_does_not_require_a_gui_page() -> None:
    assert "page" not in inspect.signature(compose_local_asr_production_evidence).parameters
    assert_no_forbidden_imports(
        SOURCE_ROOT / "composition" / "local_asr_production_evidence.py",
        ("flet", "puripuly_heart.ui.app", "puripuly_heart.ui.views"),
    )
