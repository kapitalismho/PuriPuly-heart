from __future__ import annotations

import ast
import os
from collections.abc import Iterator
from pathlib import Path


class SourceScanError(AssertionError):
    pass


def parse_python_source(source: str, module: str) -> ast.Module:
    try:
        return ast.parse(source, filename=module)
    except (SyntaxError, ValueError, TypeError) as exc:
        raise SourceScanError(f"Cannot parse protected source {module}: {exc}") from exc


def first_party_ast_sources(
    repo_root: Path, protected_roots: tuple[str, ...]
) -> Iterator[tuple[str, ast.Module]]:
    if not protected_roots:
        raise SourceScanError("No protected source roots declared")

    def discovery_error(error: OSError) -> None:
        raise error

    for relative_root in protected_roots:
        root = repo_root / relative_root
        if not root.is_dir():
            raise SourceScanError(f"Missing protected source root: {relative_root}")
        paths: list[Path] = []
        try:
            for directory, _directories, filenames in os.walk(root, onerror=discovery_error):
                paths.extend(Path(directory) / name for name in filenames if name.endswith(".py"))
        except OSError as exc:
            raise SourceScanError(
                f"Cannot discover protected source {relative_root}: {exc}"
            ) from exc
        if not paths:
            raise SourceScanError(f"Empty protected source root: {relative_root}")
        for path in sorted(paths):
            module = path.relative_to(repo_root).as_posix()
            try:
                source = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                raise SourceScanError(f"Cannot read protected source {module}: {exc}") from exc
            yield module, parse_python_source(source, module)


def imported_modules_from_source(source: str) -> set[str]:
    tree = ast.parse(source)
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def imported_modules(path: Path) -> set[str]:
    return imported_modules_from_source(path.read_text(encoding="utf-8"))


def method_source(path: Path, class_name: str, method_name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if (
                    isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name == method_name
                ):
                    return ast.get_source_segment(source, item) or ""
    raise AssertionError(f"{class_name}.{method_name} not found in {path}")


def method_source_unscoped(path: Path, method_name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name:
            return ast.get_source_segment(source, node) or ""
    raise AssertionError(f"method not found: {method_name}")


def call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def assert_no_forbidden_imports(path: Path, forbidden_prefixes: tuple[str, ...]) -> None:
    imports = imported_modules(path)
    assert not {
        imported
        for imported in imports
        for forbidden in forbidden_prefixes
        if imported == forbidden or imported.startswith(f"{forbidden}.")
    }


def find_constructions(class_name: str, root: Path) -> list[str]:
    results: list[str] = []
    for source_file in sorted(root.rglob("*.py")):
        tree = ast.parse(source_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and call_name(node) == class_name:
                results.append(source_file.relative_to(root).as_posix())
    return results
