from __future__ import annotations

from tests.helpers.ast_sources import assert_no_forbidden_imports
from tests.helpers.paths import SOURCE_ROOT


def test_ui_does_not_import_output_delivery_implementation() -> None:
    for source_file in sorted((SOURCE_ROOT / "ui").rglob("*.py")):
        assert_no_forbidden_imports(
            source_file,
            ("puripuly_heart.core.runtime.output", "puripuly_heart.core.output.router"),
        )
