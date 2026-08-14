from __future__ import annotations

import ast
from pathlib import Path
from textwrap import dedent

from tests.helpers.paths import REPO_ROOT
from tests.helpers.paths import SOURCE_ROOT as SRC_ROOT


def test_new_ui_error_events_do_not_publish_raw_string_payloads() -> None:
    violations: list[str] = []
    for path in SRC_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        _RawErrorPayloadVisitor(path, violations).visit(tree)

    assert violations == []


def test_guard_fixture_catches_positional_ui_error_payload() -> None:
    violations = _collect_violations_for_source("""
        from puripuly_heart.domain.events import UIEvent, UIEventType

        def publish(exc: Exception) -> None:
            UIEvent(UIEventType.ERROR, None, str(exc))
        """)

    assert len(violations) == 1
    assert "publishes raw string UIEventType.ERROR payload" in violations[0]


def test_guard_fixture_catches_raw_stt_error_event_messages() -> None:
    violations = _collect_violations_for_source("""
        from puripuly_heart.domain.events import STTErrorEvent

        def publish(exc: Exception) -> None:
            STTErrorEvent(message="raw failure")
            STTErrorEvent(message=f"raw failure: {exc}")
            STTErrorEvent(str(exc))
        """)

    assert len(violations) == 3
    assert all("publishes raw string STTErrorEvent message" in item for item in violations)


def test_guard_fixture_catches_stt_error_event_message_assigned_from_raw_text() -> None:
    violations = _collect_violations_for_source("""
        from puripuly_heart.domain.events import STTErrorEvent

        def publish(exc: Exception) -> None:
            message = str(exc)
            STTErrorEvent(message=message)
        """)

    assert len(violations) == 1
    assert "publishes raw string STTErrorEvent message" in violations[0]


def _collect_violations_for_source(source: str) -> list[str]:
    violations: list[str] = []
    path = SRC_ROOT / "synthetic_guard_fixture.py"
    tree = ast.parse(dedent(source), filename=str(path))
    _RawErrorPayloadVisitor(path, violations).visit(tree)
    return violations


class _RawErrorPayloadVisitor(ast.NodeVisitor):
    def __init__(self, path: Path, violations: list[str]) -> None:
        self._path = path
        self._violations = violations
        self._raw_assignments: list[set[str]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function_body(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function_body(node)

    def _visit_function_body(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._raw_assignments.append(set())
        self.generic_visit(node)
        self._raw_assignments.pop()

    def visit_Assign(self, node: ast.Assign) -> None:
        if self._raw_assignments and _is_raw_user_visible_text_expr(node.value):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self._raw_assignments[-1].add(target.id)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if (
            self._raw_assignments
            and isinstance(node.target, ast.Name)
            and node.value is not None
            and _is_raw_user_visible_text_expr(node.value)
        ):
            self._raw_assignments[-1].add(node.target.id)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if _is_ui_event_call(node) and _is_error_event_call(node):
            payload = _ui_event_payload_value(node)
            if payload is not None and self._is_forbidden_payload(payload):
                self._add_violation(
                    node,
                    "publishes raw string UIEventType.ERROR payload",
                )
        if _is_stt_error_event_call(node):
            message = _stt_error_event_message_value(node)
            if message is not None and self._is_forbidden_payload(message):
                self._add_violation(
                    node,
                    "publishes raw string STTErrorEvent message",
                )
        self.generic_visit(node)

    def _add_violation(self, node: ast.AST, message: str) -> None:
        relative_path = self._path.relative_to(REPO_ROOT).as_posix()
        self._violations.append(f"{relative_path}:{node.lineno} {message}")

    def _is_forbidden_payload(self, node: ast.AST) -> bool:
        if _is_raw_user_visible_text_expr(node):
            return True
        return (
            isinstance(node, ast.Name)
            and bool(self._raw_assignments)
            and node.id in self._raw_assignments[-1]
        )


def _is_ui_event_call(node: ast.Call) -> bool:
    func = node.func
    return isinstance(func, ast.Name) and func.id == "UIEvent"


def _is_stt_error_event_call(node: ast.Call) -> bool:
    func = node.func
    return isinstance(func, ast.Name) and func.id == "STTErrorEvent"


def _is_error_event_call(node: ast.Call) -> bool:
    type_arg = _keyword_value(node, "type")
    if type_arg is None and node.args:
        type_arg = node.args[0]
    return _is_ui_event_type_error(type_arg)


def _ui_event_payload_value(node: ast.Call) -> ast.AST | None:
    payload = _keyword_value(node, "payload")
    if payload is not None:
        return payload
    if len(node.args) > 2:
        return node.args[2]
    return None


def _stt_error_event_message_value(node: ast.Call) -> ast.AST | None:
    message = _keyword_value(node, "message")
    if message is not None:
        return message
    if node.args:
        return node.args[0]
    return None


def _keyword_value(node: ast.Call, keyword_name: str) -> ast.AST | None:
    for keyword in node.keywords:
        if keyword.arg == keyword_name:
            return keyword.value
    return None


def _is_ui_event_type_error(node: ast.AST | None) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "ERROR"
        and isinstance(node.value, ast.Name)
        and node.value.id == "UIEventType"
    )


def _is_raw_user_visible_text_expr(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return True
    if isinstance(node, ast.JoinedStr):
        return True
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "str":
        return True
    return False
