from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_osc_documentation_describes_persistent_integer_menu_controls() -> None:
    documentation = (ROOT / "docs" / "vrchat-osc.md").read_text(encoding="utf-8")

    assert "PuriPuly_Menu_" in documentation
    assert "one `Bool` proxy Expression Parameter per selectable ID" in documentation
    assert "Avatar Parameter Driver" in documentation
    assert "Enable **Local Only**" in documentation
    assert "sets ID `0` explicitly" in documentation
    assert "Add reverse states" in documentation
    assert "desktop or OSC change" in documentation
    assert "matching proxy Bool to `true`" in documentation
    assert "Puppet controls are not suitable" in documentation
