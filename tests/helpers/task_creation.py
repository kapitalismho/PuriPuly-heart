from __future__ import annotations

import ast
from collections import ChainMap, Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tests.helpers.ast_sources import first_party_ast_sources, parse_python_source

ASYNCIO_CREATE_TASK = "asyncio.create_task"
LOOP_CREATE_TASK = "loop.create_task"
BARE_RUN_TASK = "run_task(...)"
RUN_TASK = ".run_task"
ASYNCIO_TASK = "asyncio.Task"
ASYNCIO_ENSURE_FUTURE = "asyncio.ensure_future"
TaskClassification = Literal["owner_primitive", "named_owner", "legacy_exception", "raw_rejected"]


@dataclass(frozen=True, order=True)
class TaskIdentity:
    module: str
    function: str
    family: str


@dataclass(frozen=True)
class TaskCreationSite:
    identity: TaskIdentity
    line: int
    callee: str


@dataclass(frozen=True)
class TaskDisposition:
    classification: TaskClassification
    max_count: int
    rationale: str


_PAGE = "<flet.Page>"
_LOOP = "<asyncio.loop>"
_RUNNER = "<injected.run_task>"
_CONTROL_PAGE = "puripuly_heart.ui.flet_runtime.control_page"
KNOWN_PAGE_BINDINGS = {
    (
        "src/puripuly_heart/ui/desktop_overlay.py",
        "FletDesktopRendererWindow",
        "self._page",
    ): "Flet app_runner injects its Page into _handle_page/_handle_preview_page",
    (
        "src/puripuly_heart/ui/presentation_adapter.py",
        "FletUiPresentationAdapter",
        "self._app.page",
    ): "Flet presentation adapter receives the application's injected page",
}


class _LocalBindings(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.names.add(node.id)

    def visit_Import(self, node: ast.Import) -> None:
        self.names.update(alias.asname or alias.name.split(".")[0] for alias in node.names)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.names.update(alias.asname or alias.name for alias in node.names)

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.names.add(node.name)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.names.add(node.name)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name is not None:
            self.names.add(node.name)
        self.generic_visit(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        pass

    def visit_Global(self, node: ast.Global) -> None:
        self.names.difference_update(node.names)

    visit_Nonlocal = visit_Global

    def visit_ListComp(self, node: ast.ListComp) -> None:
        pass

    visit_SetComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp
    visit_DictComp = visit_ListComp


@dataclass(frozen=True)
class _DeferredCallable:
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda
    parameters: dict[str, str | None]
    lexical_bindings: ChainMap[str, str | None]
    owners: tuple[str, ...]
    classes: tuple[str, ...]
    instance_bindings: tuple[dict[str, str], ...]


class _TaskScanner(ast.NodeVisitor):
    def __init__(self, module: str) -> None:
        self.module = module
        self.module_name = module.removeprefix("src/").removesuffix(".py").replace("/", ".")
        self.bindings: ChainMap[str, str | None] = ChainMap()
        self.lexical_bindings = self.bindings
        self.instance_bindings: list[dict[str, str]] = []
        self.owners: list[str] = []
        self.classes: list[str] = []
        self.sites: list[TaskCreationSite] = []
        self.pending_callables: list[_DeferredCallable] = []

    def visit_Module(self, node: ast.Module) -> None:
        self.scan_body(node.body)

    def scan_body(self, body: Iterable[ast.AST]) -> None:
        previous_pending = self.pending_callables
        self.pending_callables = []
        for node in body:
            self.visit(node)
        completed = self.pending_callables
        self.pending_callables = previous_pending
        for pending in completed:
            self.scan_callable(pending)

    def defer_callable(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda,
        parameters: dict[str, str | None],
    ) -> None:
        self.pending_callables.append(
            _DeferredCallable(
                node,
                parameters,
                self.lexical_bindings,
                tuple(self.owners),
                tuple(self.classes),
                tuple(self.instance_bindings),
            )
        )

    def scan_callable(self, pending: _DeferredCallable) -> None:
        previous = (
            self.bindings,
            self.lexical_bindings,
            self.owners,
            self.classes,
            self.instance_bindings,
        )
        self.bindings = pending.lexical_bindings.new_child()
        self.lexical_bindings = self.bindings
        self.owners = list(pending.owners)
        self.classes = list(pending.classes)
        self.instance_bindings = list(pending.instance_bindings)
        node = pending.node
        body = [node.body] if isinstance(node, ast.Lambda) else node.body
        local_bindings = _LocalBindings()
        for statement in body:
            local_bindings.visit(statement)
        self.bindings.update(dict.fromkeys(local_bindings.names))
        self.bindings.update(pending.parameters)
        self.scan_body(body)
        (
            self.bindings,
            self.lexical_bindings,
            self.owners,
            self.classes,
            self.instance_bindings,
        ) = previous

    def resolve(self, node: ast.AST | None) -> str | None:
        if isinstance(node, ast.Name):
            return self.bindings.get(node.id)
        if isinstance(node, ast.Attribute):
            expression = ast.unparse(node)
            if self.classes and (self.module, self.classes[-1], expression) in KNOWN_PAGE_BINDINGS:
                return _PAGE
            if expression in self.bindings:
                return self.bindings[expression]
            if self.instance_bindings and expression in self.instance_bindings[-1]:
                return self.instance_bindings[-1][expression]
            value = self.resolve(node.value)
            if value == _PAGE and node.attr == "run_task":
                return _RUNNER
            if value == _LOOP and node.attr == "create_task":
                return LOOP_CREATE_TASK
            return f"{value}.{node.attr}" if value is not None else None
        if isinstance(node, ast.Call):
            callee = self.resolve(node.func)
            if callee in {
                "asyncio.get_running_loop",
                "asyncio.get_event_loop",
                "asyncio.new_event_loop",
            }:
                return _LOOP
            if callee in {_CONTROL_PAGE, "flet.Page"}:
                return _PAGE
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) >= 2
            ):
                attribute = node.args[1]
                if isinstance(attribute, ast.Constant) and isinstance(attribute.value, str):
                    return self.resolve(
                        ast.Attribute(value=node.args[0], attr=attribute.value, ctx=ast.Load())
                    )
        return None

    def annotation_expression(self, annotation: ast.AST | None) -> ast.AST | None:
        if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
            tree = parse_python_source(annotation.value, self.module)
            if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
                raise AssertionError(
                    f"Invalid binding annotation in protected source {self.module}"
                )
            return tree.body[0].value
        return annotation

    def annotation_binding(self, annotation: ast.AST | None) -> str | None:
        annotation = self.annotation_expression(annotation)
        resolved = self.resolve(annotation)
        if resolved == "flet.Page":
            return _PAGE
        if resolved in {"asyncio.AbstractEventLoop", "asyncio.BaseEventLoop"}:
            return _LOOP
        alternatives: tuple[ast.AST, ...] = ()
        if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
            alternatives = (annotation.left, annotation.right)
        elif isinstance(annotation, ast.Subscript) and self.resolve(annotation.value) in {
            "typing.Optional",
            "typing.Union",
        }:
            alternatives = (
                tuple(annotation.slice.elts)
                if isinstance(annotation.slice, ast.Tuple)
                else (annotation.slice,)
            )
        bindings = {self.annotation_binding(item) for item in alternatives} - {None}
        return next(iter(bindings)) if len(bindings) == 1 else None

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.bindings[alias.asname or alias.name.split(".")[0]] = (
                alias.name if alias.asname else alias.name.split(".")[0]
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level:
            package = self.module_name.split(".")[:-1]
            package = package[: len(package) - node.level + 1]
            module = ".".join((*package, *((node.module or "").split("."))))
        else:
            module = node.module or ""
        for alias in node.names:
            self.bindings[alias.asname or alias.name] = f"{module}.{alias.name}"

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        binding = self.resolve(node.value)
        for target in node.targets:
            self.bind_target(target, binding)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(node.value)
        self.bind_target(
            node.target, self.annotation_binding(node.annotation) or self.resolve(node.value)
        )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        previous = self.bindings
        self.owners.append(node.name)
        for expression in (*node.decorator_list, *node.bases):
            self.visit(expression)
        for keyword in node.keywords:
            self.visit(keyword.value)
        flet_control = any((self.resolve(base) or "").startswith("flet.") for base in node.bases)
        self.bindings = self.lexical_bindings.new_child()
        self.classes.append(node.name)
        inferred_attributes: dict[str, str] = {}
        self.instance_bindings.append(inferred_attributes)
        if flet_control:
            inferred_attributes["self.page"] = _PAGE
        for statement in node.body:
            if isinstance(statement, ast.AnnAssign):
                binding = self.annotation_binding(statement.annotation)
                if binding and isinstance(statement.target, ast.Name):
                    inferred_attributes[f"self.{statement.target.id}"] = binding
            if (
                isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
                and statement.name == "__init__"
            ):
                parameter_bindings = {
                    argument.arg: self.annotation_binding(argument.annotation)
                    for argument in (
                        *statement.args.posonlyargs,
                        *statement.args.args,
                        *statement.args.kwonlyargs,
                    )
                }
                for assignment in ast.walk(statement):
                    if isinstance(assignment, ast.Assign) and isinstance(
                        assignment.value, ast.Name
                    ):
                        for target in assignment.targets:
                            if isinstance(target, ast.Attribute) and parameter_bindings.get(
                                assignment.value.id
                            ):
                                inferred_attributes[ast.unparse(target)] = parameter_bindings[
                                    assignment.value.id
                                ]
        for statement in node.body:
            self.visit(statement)
        self.classes.pop()
        self.instance_bindings.pop()
        self.owners.pop()
        self.bindings = previous
        self.bindings[node.name] = f"{self.module_name}.{node.name}"

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.owners.append(node.name)
        for expression in (*node.decorator_list, *node.args.defaults):
            self.visit(expression)
        for expression in node.args.kw_defaults:
            if expression is not None:
                self.visit(expression)
        parameter_bindings = {
            argument.arg: self.annotation_binding(argument.annotation)
            for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
        }
        for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
            if argument.arg == "run_task":
                annotation = self.annotation_expression(argument.annotation)
                if isinstance(annotation, ast.Subscript):
                    annotation = annotation.value
                if self.resolve(annotation) in {"typing.Callable", "collections.abc.Callable"}:
                    parameter_bindings[argument.arg] = _RUNNER
        for argument in (node.args.vararg, node.args.kwarg):
            if argument is not None:
                parameter_bindings[argument.arg] = None
        self.defer_callable(node, parameter_bindings)
        self.owners.pop()
        self.bindings[node.name] = f"{self.module_name}.{node.name}"

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node: ast.Lambda) -> None:
        for expression in node.args.defaults:
            self.visit(expression)
        for expression in node.args.kw_defaults:
            if expression is not None:
                self.visit(expression)
        parameter_bindings = {
            argument.arg: None
            for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
        }
        for argument in (node.args.vararg, node.args.kwarg):
            if argument is not None:
                parameter_bindings[argument.arg] = None
        self.defer_callable(node, parameter_bindings)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is not None:
            self.visit(node.type)
        if node.name is not None:
            self.bindings[node.name] = None
        for statement in node.body:
            self.visit(statement)

    def visit_For(self, node: ast.For | ast.AsyncFor) -> None:
        self.visit(node.iter)
        self.bind_target(node.target)
        for statement in (*node.body, *node.orelse):
            self.visit(statement)

    visit_AsyncFor = visit_For

    def bind_target(self, target: ast.AST, binding: str | None = None) -> None:
        if isinstance(target, ast.Name):
            self.bindings[target.id] = binding
        elif isinstance(target, ast.Attribute):
            self.visit(target.value)
            self.bindings[ast.unparse(target)] = binding
        elif isinstance(target, ast.Subscript):
            self.visit(target.value)
            self.visit(target.slice)
        elif isinstance(target, ast.Starred):
            self.bind_target(target.value)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for item in target.elts:
                self.bind_target(item)

    def visit_With(self, node: ast.With | ast.AsyncWith) -> None:
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self.bind_target(item.optional_vars)
        for statement in node.body:
            self.visit(statement)

    visit_AsyncWith = visit_With

    def visit_ListComp(
        self, node: ast.ListComp | ast.SetComp | ast.GeneratorExp | ast.DictComp
    ) -> None:
        previous = self.bindings
        self.visit(node.generators[0].iter)
        previous_lexical = self.lexical_bindings
        self.bindings = previous_lexical.new_child()
        self.lexical_bindings = self.bindings
        for index, generator in enumerate(node.generators):
            if index:
                self.visit(generator.iter)
            self.bind_target(generator.target)
            for condition in generator.ifs:
                self.visit(condition)
        if isinstance(node, ast.DictComp):
            self.visit(node.key)
            self.visit(node.value)
        else:
            self.visit(node.elt)
        self.bindings = previous
        self.lexical_bindings = previous_lexical

    visit_SetComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp
    visit_DictComp = visit_ListComp

    def visit_Call(self, node: ast.Call) -> None:
        resolved = self.resolve(node.func)
        family = resolved
        if resolved == _RUNNER:
            family = RUN_TASK if isinstance(node.func, ast.Attribute) else BARE_RUN_TASK
        if family in {
            ASYNCIO_CREATE_TASK,
            LOOP_CREATE_TASK,
            ASYNCIO_TASK,
            ASYNCIO_ENSURE_FUTURE,
            BARE_RUN_TASK,
            RUN_TASK,
        }:
            self.sites.append(
                TaskCreationSite(
                    TaskIdentity(self.module, ".".join(self.owners) or "<module>", family),
                    node.lineno,
                    ast.unparse(node.func),
                )
            )
        self.generic_visit(node)


def task_creation_sites_from_source(source: str, module: str) -> list[TaskCreationSite]:
    scanner = _TaskScanner(module)
    scanner.visit(parse_python_source(source, module))
    return scanner.sites


def scan_task_creation_sites(
    repo_root: Path, protected_roots: tuple[str, ...]
) -> list[TaskCreationSite]:
    sites: list[TaskCreationSite] = []
    for module, tree in first_party_ast_sources(repo_root, protected_roots):
        scanner = _TaskScanner(module)
        scanner.visit(tree)
        sites.extend(scanner.sites)
    return sites


def assert_task_ownership(
    sites: list[TaskCreationSite], reviewed: Mapping[TaskIdentity, TaskDisposition]
) -> list[tuple[TaskCreationSite, TaskClassification]]:
    counts = Counter(site.identity for site in sites)
    violations: list[str] = []
    classifications: list[tuple[TaskCreationSite, TaskClassification]] = []
    for identity, disposition in reviewed.items():
        if disposition.classification not in {"owner_primitive", "named_owner", "legacy_exception"}:
            raise AssertionError(f"Invalid task disposition: {identity}")
        if disposition.max_count < 1 or not disposition.rationale.strip():
            raise AssertionError(f"Missing task bound or lifetime rationale: {identity}")
    for site in sites:
        disposition = reviewed.get(site.identity)
        classification: TaskClassification = (
            disposition.classification if disposition else "raw_rejected"
        )
        classifications.append((site, classification))
        if disposition is None or counts[site.identity] > disposition.max_count:
            identity = site.identity
            reason = (
                "unreviewed raw creation"
                if disposition is None
                else (
                    f"multiplicity {counts[identity]} exceeds reviewed upper bound {disposition.max_count}"
                )
            )
            violations.append(
                f"{identity.module}:{site.line} {identity.function} {identity.family} "
                f"({site.callee}): {reason}; use LifecycleScope/start_lifecycle_task or "
                "an explicitly reviewed named owner with a cancellation/draining lifetime rationale"
            )
    assert not violations, "\n".join(violations)
    return classifications
