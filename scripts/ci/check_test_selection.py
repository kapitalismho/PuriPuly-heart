from __future__ import annotations

import ast
import os
import re
import shlex
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = Path(".github/workflows/pr-ci.yml")
INTENTIONAL_EXCLUSIONS = {
    "tests/integration": (
        "Mixed live-provider, credential, model-download, and other opt-in integration prerequisites; "
        "not authorized for ordinary offline PR execution."
    ),
}
MATRIX_ARGS = "${{ matrix.pytest-args }}"


class SelectionError(ValueError):
    pass


class _UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, (str, bool)) or key in mapping:
                raise SelectionError(f"Unsupported or duplicate workflow key: {key!r}")
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


@dataclass(frozen=True)
class SelectionReport:
    selected: frozenset[str]
    excluded: frozenset[str]


def _mapping(value: object, label: str) -> dict:
    if not isinstance(value, dict) or not value:
        raise SelectionError(f"Missing or invalid {label}")
    return value


def _pytest_commands(workflow: object) -> list[str]:
    jobs = _mapping(_mapping(workflow, "workflow").get("jobs"), "workflow jobs")
    commands = []
    for job_name, job in jobs.items():
        job = _mapping(job, f"job {job_name}")
        steps = job.get("steps")
        if not isinstance(steps, list) or not steps:
            raise SelectionError(f"Missing steps for {job_name}")
        for step in steps:
            if not isinstance(step, dict):
                raise SelectionError(f"Invalid step in {job_name}")
            command = step.get("run", "")
            if not isinstance(command, str):
                raise SelectionError(f"Invalid run source in {job_name}")
            if not re.search(r"\bpytest\b", command):
                continue
            if MATRIX_ARGS not in command:
                commands.append(command)
                continue
            strategy = _mapping(job.get("strategy"), f"strategy for {job_name}")
            matrix = _mapping(strategy.get("matrix"), f"matrix for {job_name}")
            if set(matrix) != {"include"}:
                raise SelectionError(
                    f"Only explicit matrix include selections are supported: {job_name}"
                )
            rows = matrix["include"]
            if not isinstance(rows, list) or not rows:
                raise SelectionError(f"Empty matrix selections: {job_name}")
            for row in rows:
                args = _mapping(row, f"matrix row for {job_name}").get("pytest-args")
                if not isinstance(args, str) or not args.strip():
                    raise SelectionError(f"Missing pytest-args in {job_name}")
                commands.append(command.replace(MATRIX_ARGS, args))
    if not commands:
        raise SelectionError("Workflow declares no pytest selections")
    return commands


def _targets(command: str) -> list[str]:
    command = command.strip()
    if "${{" in command or any(char in command for char in "\n\r;&|`<>"):
        raise SelectionError(f"Unsupported pytest command source: {command!r}")
    try:
        tokens = shlex.split(command)
    except ValueError as error:
        raise SelectionError(f"Cannot parse pytest command: {error}") from error
    if tokens[:4] != ["uv", "run", "--locked", "pytest"]:
        raise SelectionError(f"Expected direct uv run --locked pytest selection: {command!r}")
    targets = []
    for token in tokens[4:]:
        if token == "-ra" or token.startswith("--junitxml="):
            continue
        if token.startswith("-"):
            raise SelectionError(f"Unsupported pytest selection option: {token}")
        targets.append(token)
    if not targets:
        raise SelectionError("Empty pytest selection")
    return targets


def _discover(root: Path) -> dict[str, ast.Module]:
    protected = root / "tests"
    if not protected.is_dir():
        raise SelectionError(f"Missing protected test root: {protected}")
    sources = {}

    def fail(error: OSError) -> None:
        raise error

    for directory, children, filenames in os.walk(protected, onerror=fail):
        for child in children:
            if (Path(directory) / child).is_symlink():
                raise SelectionError(f"Symlinked test directory is unsupported: {directory}/{child}")
        for name in filenames:
            if not (name.startswith("test_") or name.endswith("_test.py")) or not name.endswith(
                ".py"
            ):
                continue
            path = Path(directory) / name
            if path.is_symlink():
                raise SelectionError(f"Symlinked test source is unsupported: {path}")
            with tokenize.open(path) as source:
                tree = ast.parse(source.read(), filename=str(path))
            sources[path.relative_to(root).as_posix()] = tree
    if not sources:
        raise SelectionError("Protected test root is unexpectedly empty")
    return sources


def _target_files(root: Path, target: str, sources: dict[str, ast.Module]) -> set[str]:
    filename, *nodes = target.split("::")
    path = PurePosixPath(filename)
    if (
        not filename
        or path.is_absolute()
        or path.parts[0] != "tests"
        or ".." in path.parts
        or "\\" in filename
        or any(char in filename for char in "*?[]")
    ):
        raise SelectionError(f"Unsupported test target: {target}")
    actual = root / filename
    if actual.is_dir() and not nodes:
        matches = {name for name in sources if PurePosixPath(name).is_relative_to(path)}
    elif actual.is_file() and filename in sources:
        matches = {filename}
    else:
        raise SelectionError(f"Missing or invalid declared test target: {target}")
    if not matches:
        raise SelectionError(f"Declared test target is unexpectedly empty: {target}")
    if nodes:
        body = sources[filename].body
        for node_name in nodes:
            matches_node = [
                node
                for node in body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and node.name == node_name
            ]
            if len(matches_node) != 1:
                raise SelectionError(f"Missing or unsupported declared test node: {target}")
            body = matches_node[0].body
        return set()
    return matches


def check_selection(root: Path = REPO_ROOT) -> SelectionReport:
    try:
        with (root / WORKFLOW).open(encoding="utf-8") as stream:
            workflow = yaml.load(stream, Loader=_UniqueKeyLoader)
        sources = _discover(root)
        selected = set()
        for command in _pytest_commands(workflow):
            for target in _targets(command):
                selected.update(_target_files(root, target, sources))
        excluded = set()
        for directory, reason in INTENTIONAL_EXCLUSIONS.items():
            if not reason.strip():
                raise SelectionError(f"Missing exclusion rationale: {directory}")
            excluded.update(_target_files(root, directory, sources))
        if overlap := selected & excluded:
            raise SelectionError(f"Intentional integration exclusion was selected: {sorted(overlap)}")
        if omitted := sources.keys() - selected - excluded:
            raise SelectionError(f"Offline test files absent from PR selections: {sorted(omitted)}")
        return SelectionReport(frozenset(selected), frozenset(excluded))
    except (OSError, UnicodeError, SyntaxError, yaml.YAMLError) as error:
        raise SelectionError(f"Cannot inspect workflow/test source: {error}") from error


def main() -> int:
    try:
        report = check_selection()
    except SelectionError as error:
        print(f"Test selection check failed: {error}", file=sys.stderr)
        return 1
    print(f"PR test selection covers {len(report.selected)} offline files")
    for directory, reason in INTENTIONAL_EXCLUSIONS.items():
        print(f"Excluded {directory}: {reason}")
    print(f"Intentionally excluded files: {len(report.excluded)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
