from __future__ import annotations

import ast
from pathlib import Path

from tests.helpers.paths import REPO_ROOT

STT_PROVIDER_ROOT = REPO_ROOT / "src" / "puripuly_heart" / "providers" / "stt"

LOGGER_METHODS = {"debug", "info", "warning", "error", "exception", "critical"}
UNSAFE_TEXT_NAMES = {
    "preview",
    "source_text",
    "stash",
    "text",
    "transcript",
    "transcript_text",
    "translation",
    "translation_text",
}


def _repo_path(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _is_logger_call(node: ast.Call) -> bool:
    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr in LOGGER_METHODS
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "logger"
    )


def _unsafe_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name) and node.id in UNSAFE_TEXT_NAMES:
        return node.id
    if isinstance(node, ast.Attribute) and node.attr in UNSAFE_TEXT_NAMES:
        return node.attr
    return None


def _unsafe_logger_text_args(node: ast.Call) -> list[str]:
    offenders: list[str] = []
    for arg in node.args:
        name = _unsafe_name(arg)
        if name is not None:
            offenders.append(name)
        if isinstance(arg, ast.JoinedStr):
            for value in arg.values:
                if isinstance(value, ast.FormattedValue):
                    name = _unsafe_name(value.value)
                    if name is not None:
                        offenders.append(name)
    for keyword in node.keywords:
        if keyword.value is None:
            continue
        name = _unsafe_name(keyword.value)
        if name is not None:
            offenders.append(name)
    return offenders


def test_stt_provider_logs_do_not_emit_raw_transcript_translation_or_source_text() -> None:
    offenders: list[str] = []
    for source_file in sorted(STT_PROVIDER_ROOT.rglob("*.py")):
        tree = ast.parse(source_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not _is_logger_call(node):
                continue
            for name in _unsafe_logger_text_args(node):
                offenders.append(f"{_repo_path(source_file)}:{node.lineno}:{name}")

    assert offenders == []
