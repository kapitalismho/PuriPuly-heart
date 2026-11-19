from __future__ import annotations

import ast
from pathlib import Path

import pytest

from puripuly_heart.core.stt import local_qwen_hallucination as detector


def test_known_local_qwen_hallucination_set_contains_only_selected_artifacts() -> None:
    assert detector.KNOWN_LOCAL_QWEN_HALLUCINATIONS == frozenset({"leşme", "acia"})


@pytest.mark.parametrize("text", ["leşme", "acia", "  leşme  ", "\tacia\r\n"])
def test_known_local_qwen_hallucination_detector_accepts_exact_artifacts_after_strip(
    text: str,
) -> None:
    assert detector.is_known_local_qwen_hallucination(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "的答案",
        "虚构",
        "夫",
        "夫夫",
        "格力",
        "",
        "   ",
        "Leşme",
        "LEŞME",
        "lesme",
        "leşm",
        "leşmeler",
        "xleşmex",
        "AcIa",
        "acíá",
        "aci",
        "acia.",
        "xaciax",
    ],
)
def test_known_local_qwen_hallucination_detector_rejects_other_text(text: str) -> None:
    assert detector.is_known_local_qwen_hallucination(text) is False


def test_known_local_qwen_hallucination_detector_has_no_ui_settings_provider_or_flet_imports() -> (
    None
):
    module_path = Path(detector.__file__)
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imports: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.append(node.module)

    disallowed_imports = (
        "flet",
        "puripuly_heart.ui",
        "puripuly_heart.config",
        "puripuly_heart.providers",
    )
    assert not [
        imported
        for imported in imports
        if any(
            imported == disallowed or imported.startswith(f"{disallowed}.")
            for disallowed in disallowed_imports
        )
    ]
