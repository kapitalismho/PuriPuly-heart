from __future__ import annotations

import importlib
import importlib.machinery
import importlib.resources
import importlib.util
import inspect
import marshal
import os
import pkgutil
import shutil
import subprocess
import sys
import traceback
import zipfile
from pathlib import Path

import pytest

_HOST_ROOT = Path(__file__).resolve().parents[1] / "native" / "windows_host"
_PREFIX = "_native_test_"


def _bytecode(source: str, filename: str) -> bytes:
    code = compile(source, filename, "exec")
    return importlib.util.MAGIC_NUMBER + b"\x01\x00\x00\x00" + bytes(8) + marshal.dumps(code)


def _stage_runtime(tmp_path: Path, modules: dict[str, str], *, payloads=None, index=None):
    runtime_root = tmp_path / "runtime 설치 with spaces"
    dependency_root = runtime_root / "site-packages"
    archive_path = runtime_root / "app" / "python.zip"
    dependency_root.mkdir(parents=True)
    archive_path.parent.mkdir()
    entries = {}
    with zipfile.ZipFile(archive_path, "w") as archive:
        for relative, source in modules.items():
            physical_source = dependency_root / relative
            physical_source.parent.mkdir(parents=True, exist_ok=True)
            physical_source.write_text(source, encoding="utf-8")
            name = relative.removesuffix(".py").replace("/", ".")
            if name.endswith(".__init__"):
                name = name.removesuffix(".__init__")
            entries[name] = relative
            payload = _bytecode(source, "site-packages/" + relative)
            if payloads is not None and relative in payloads:
                payload = payloads[relative]
            if payload is not None:
                archive.writestr("_native_dependencies/" + relative[:-3] + ".pyc", payload)
        archive.writestr(
            "_native_dependencies.index",
            marshal.dumps({"version": 1, "modules": entries} if index is None else index),
        )
    return runtime_root, dependency_root, archive_path


@pytest.fixture
def runtime_loader(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "_native_runtime_under_test", _HOST_ROOT / "_puripuly_native_runtime.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delenv("PURIPULY_HEART_NATIVE_RUNTIME_ROOT", raising=False)
    monkeypatch.delenv("PURIPULY_HEART_NATIVE_RESOURCE_ROOT", raising=False)
    yield module
    for name in tuple(sys.modules):
        if name.startswith(_PREFIX):
            del sys.modules[name]


def _activate(runtime_loader, monkeypatch, runtime_root, dependency_root):
    monkeypatch.syspath_prepend(str(dependency_root))
    runtime_loader.install(runtime_root)


def _copy_socket_extension(directory: Path):
    spec = importlib.util.find_spec("_socket")
    if spec is None or spec.origin is None or not any(
        spec.origin.endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES
    ):
        pytest.skip("the interpreter does not provide _socket as a native extension")
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / Path(spec.origin).name
    shutil.copyfile(spec.origin, destination)
    return destination


@pytest.fixture
def mixed_runtime(tmp_path, monkeypatch, runtime_loader):
    package_source = (
        "from . import _socket, logic\n"
        "from pathlib import Path\n"
        "address = _socket.inet_aton('127.0.0.1')\n"
        "eager_value = logic.VALUE\n"
        "payload = (Path(__file__).parent / 'payload.txt').read_text(encoding='utf-8')\n"
    )
    logic_source = (
        "VALUE = 'archived'\n"
        "def explode():\n"
        "    raise RuntimeError('archived failure')\n"
    )
    runtime_root, dependency_root, archive_path = _stage_runtime(
        tmp_path,
        {
            "_native_test_mixed/__init__.py": package_source,
            "_native_test_mixed/logic.py": logic_source,
        },
    )
    package_root = dependency_root / "_native_test_mixed"
    (package_root / "payload.txt").write_text("physical resource 한글", encoding="utf-8")
    extension = _copy_socket_extension(package_root)
    _activate(runtime_loader, monkeypatch, runtime_root, dependency_root)
    return package_root, extension, logic_source


def test_mixed_package_eagerly_loads_archived_python_and_physical_extension(mixed_runtime):
    package_root, extension, _ = mixed_runtime

    package = importlib.import_module("_native_test_mixed")

    assert package.address == b"\x7f\x00\x00\x01"
    assert package.eager_value == "archived"
    assert package.payload == "physical resource 한글"
    assert package.__file__ == str(package_root / "__init__.py")
    assert package.__path__ == [str(package_root)]
    assert package.__spec__.origin == package.__file__
    assert isinstance(package.__loader__, importlib.machinery.SourceFileLoader)
    assert Path(package._socket.__file__) == extension
    assert isinstance(package._socket.__loader__, importlib.machinery.ExtensionFileLoader)
    assert not list(package_root.rglob("*.pyc"))


def test_package_resources_discovery_source_and_tracebacks_remain_physical(mixed_runtime):
    package_root, _, logic_source = mixed_runtime
    package = importlib.import_module("_native_test_mixed")
    logic = package.logic

    assert pkgutil.get_data(package.__name__, "payload.txt") == "physical resource 한글".encode()
    resource = importlib.resources.files(package).joinpath("payload.txt")
    assert resource.read_text(encoding="utf-8") == "physical resource 한글"
    with importlib.resources.as_file(resource) as physical_resource:
        assert physical_resource == package_root / "payload.txt"
    assert {item.name for item in pkgutil.iter_modules(package.__path__)} >= {"_socket", "logic"}
    assert logic.__loader__.get_source(logic.__name__) == logic_source
    assert inspect.getsource(logic.explode) == (
        "def explode():\n    raise RuntimeError('archived failure')\n"
    )
    assert inspect.getsourcefile(logic.explode) == str(package_root / "logic.py")
    assert logic.explode.__code__.co_filename == str(package_root / "logic.py")
    with pytest.raises(RuntimeError, match="archived failure") as caught:
        logic.explode()
    frames = traceback.extract_tb(caught.value.__traceback__)
    assert frames[-1].filename == str(package_root / "logic.py")
    assert frames[-1].line == "raise RuntimeError('archived failure')"


def test_indexed_import_uses_archived_code_without_source_checks_or_cache_writes(
    tmp_path, monkeypatch, runtime_loader
):
    runtime_root, dependency_root, _ = _stage_runtime(
        tmp_path,
        {
            "_native_test_archived/__init__.py": "from .child import VALUE\n",
            "_native_test_archived/child.py": "VALUE = 'archive'\n",
            "_native_test_changed.py": "VALUE = 'archive'\n",
        },
    )
    _activate(runtime_loader, monkeypatch, runtime_root, dependency_root)
    (dependency_root / "_native_test_archived" / "__init__.py").unlink()
    (dependency_root / "_native_test_archived" / "child.py").unlink()
    (dependency_root / "_native_test_changed.py").write_text("VALUE = 'loose'\n")

    package = importlib.import_module("_native_test_archived")
    changed = importlib.import_module("_native_test_changed")

    assert package.VALUE == "archive"
    assert changed.VALUE == "archive"
    assert not list(dependency_root.rglob("*.pyc"))


@pytest.mark.parametrize(
    "payload",
    [
        None,
        b"truncated",
        b"BAD!" + bytes(12) + marshal.dumps(compile("VALUE = 2", "bad.py", "exec")),
        importlib.util.MAGIC_NUMBER + b"\x04\x00\x00\x00" + bytes(8),
        importlib.util.MAGIC_NUMBER + b"\x01\x00\x00\x00" + bytes(8) + b"\xff",
        importlib.util.MAGIC_NUMBER + b"\x01\x00\x00\x00" + bytes(8) + marshal.dumps({}),
    ],
    ids=["missing", "truncated", "wrong-magic", "wrong-flags", "invalid-marshal", "not-code"],
)
def test_indexed_code_failure_never_executes_loose_source(
    tmp_path, monkeypatch, runtime_loader, payload
):
    marker = tmp_path / "source-executed"
    source = f"from pathlib import Path\nPath({str(marker)!r}).write_text('fallback')\n"
    runtime_root, dependency_root, _ = _stage_runtime(
        tmp_path,
        {"_native_test_broken.py": source},
        payloads={"_native_test_broken.py": payload},
    )
    _activate(runtime_loader, monkeypatch, runtime_root, dependency_root)

    with pytest.raises(ImportError) as caught:
        importlib.import_module("_native_test_broken")

    assert caught.value.name == "_native_test_broken"
    assert caught.value.path == str(dependency_root / "_native_test_broken.py")
    assert not marker.exists()
    assert "_native_test_broken" not in sys.modules
    assert not list(dependency_root.rglob("*.pyc"))


@pytest.mark.parametrize("external_first", [True, False])
def test_top_level_search_path_precedence_is_preserved(
    tmp_path, monkeypatch, runtime_loader, external_first
):
    runtime_root, dependency_root, _ = _stage_runtime(
        tmp_path, {"_native_test_choice.py": "VALUE = 'archive'\n"}
    )
    external = tmp_path / "external"
    external.mkdir()
    (external / "_native_test_choice.py").write_text("VALUE = 'external'\n")
    _activate(runtime_loader, monkeypatch, runtime_root, dependency_root)
    if external_first:
        monkeypatch.syspath_prepend(str(external))
    else:
        sys.path.insert(1, str(external))

    module = importlib.import_module("_native_test_choice")

    assert module.VALUE == ("external" if external_first else "archive")
    expected_root = external if external_first else dependency_root
    assert module.__file__ == str(expected_root / "_native_test_choice.py")


@pytest.mark.parametrize("include_native_path", [True, False])
def test_package_search_path_precedence_and_replacement_are_preserved(
    tmp_path, monkeypatch, runtime_loader, include_native_path
):
    runtime_root, dependency_root, _ = _stage_runtime(
        tmp_path,
        {
            "_native_test_paths/__init__.py": "VALUE = 'package'\n",
            "_native_test_paths/child.py": "VALUE = 'archive'\n",
        },
        payloads={"_native_test_paths/child.py": None},
    )
    external = tmp_path / "external-package"
    external.mkdir()
    (external / "child.py").write_text("VALUE = 'external'\n")
    _activate(runtime_loader, monkeypatch, runtime_root, dependency_root)
    package = importlib.import_module("_native_test_paths")
    if include_native_path:
        package.__path__.insert(0, str(external))
    else:
        package.__path__[:] = [str(external)]

    child = importlib.import_module("_native_test_paths.child")

    assert child.VALUE == "external"
    assert child.__file__ == str(external / "child.py")


def test_namespace_paths_native_extensions_and_physical_resources_are_preserved(
    tmp_path, monkeypatch, runtime_loader
):
    runtime_root, dependency_root, _ = _stage_runtime(
        tmp_path, {"_native_test_namespace/child.py": "VALUE = 'archive'\n"}
    )
    native_namespace = dependency_root / "_native_test_namespace"
    (native_namespace / "child.py").unlink()
    (native_namespace / "native.txt").write_text("native resource")
    extension = _copy_socket_extension(native_namespace)
    external = tmp_path / "external"
    external_namespace = external / "_native_test_namespace"
    external_namespace.mkdir(parents=True)
    (external_namespace / "other.py").write_text("VALUE = 'external'\n")
    (external_namespace / "external.txt").write_text("external resource")
    _activate(runtime_loader, monkeypatch, runtime_root, dependency_root)
    monkeypatch.syspath_prepend(str(external))

    namespace = importlib.import_module("_native_test_namespace")
    child = importlib.import_module("_native_test_namespace.child")
    other = importlib.import_module("_native_test_namespace.other")
    native = importlib.import_module("_native_test_namespace._socket")

    assert namespace.__spec__.origin is None
    assert list(namespace.__path__) == [str(external_namespace), str(native_namespace)]
    assert child.VALUE == "archive"
    assert other.VALUE == "external"
    assert Path(native.__file__) == extension
    assert native.inet_aton("127.0.0.1") == b"\x7f\x00\x00\x01"
    resources = importlib.resources.files(namespace)
    assert resources.joinpath("native.txt").read_text() == "native resource"
    assert resources.joinpath("external.txt").read_text() == "external resource"


def test_earlier_namespace_does_not_shadow_indexed_regular_package(
    tmp_path, monkeypatch, runtime_loader
):
    runtime_root, dependency_root, _ = _stage_runtime(
        tmp_path, {"_native_test_regular/__init__.py": "VALUE = 'archive'\n"}
    )
    external = tmp_path / "external"
    (external / "_native_test_regular").mkdir(parents=True)
    _activate(runtime_loader, monkeypatch, runtime_root, dependency_root)
    monkeypatch.syspath_prepend(str(external))

    package = importlib.import_module("_native_test_regular")

    assert package.VALUE == "archive"
    assert package.__path__ == [str(dependency_root / "_native_test_regular")]


def test_install_is_idempotent_and_honors_distinct_resource_root(
    tmp_path, monkeypatch, runtime_loader
):
    runtime_root, dependency_root, archive_path = _stage_runtime(
        tmp_path, {"_native_test_repeat.py": "VALUE = 'archive'\n"}
    )
    resource_root = tmp_path / "resources separate"
    resource_root.mkdir()
    archive_path.rename(resource_root / "python.zip")
    monkeypatch.setenv("PURIPULY_HEART_NATIVE_RESOURCE_ROOT", str(resource_root))
    _activate(runtime_loader, monkeypatch, runtime_root, dependency_root)

    runtime_loader.install(str(runtime_root / "."))
    runtime_loader.install(runtime_root)
    module = importlib.import_module("_native_test_repeat")

    assert module.VALUE == "archive"
    assert module.__file__ == str(dependency_root / "_native_test_repeat.py")


def _child_environment(runtime_root, dependency_root, archive_path, *, configured):
    environment = os.environ.copy()
    environment.pop("PURIPULY_HEART_NATIVE_RESOURCE_ROOT", None)
    environment.pop("PURIPULY_HEART_NATIVE_RUNTIME_ROOT", None)
    environment["PYTHONPATH"] = os.pathsep.join([str(archive_path), str(dependency_root)])
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    if configured:
        environment["PURIPULY_HEART_NATIVE_RUNTIME_ROOT"] = str(runtime_root)
        environment["PURIPULY_HEART_NATIVE_RESOURCE_ROOT"] = str(archive_path.parent)
    return environment


def _add_bootstrap(archive_path):
    with zipfile.ZipFile(archive_path, "a") as archive:
        for name in ("_puripuly_native_runtime.py", "sitecustomize.py"):
            archive.write(_HOST_ROOT / name, name)


@pytest.mark.parametrize("mode", ["command", "module", "worker"])
def test_configured_python_children_activate_archived_code(tmp_path, runtime_loader, mode):
    runtime_root, dependency_root, archive_path = _stage_runtime(
        tmp_path,
        {
            "_native_test_child/__init__.py": "VALUE = 'archive'\n",
            "_native_test_child/__main__.py": "from . import VALUE\nprint(VALUE)\n",
        },
    )
    (dependency_root / "_native_test_child" / "__init__.py").write_text("VALUE = 'loose'\n")
    resource_root = tmp_path / "configured child resources"
    resource_root.mkdir()
    moved_archive = resource_root / "python.zip"
    archive_path.rename(moved_archive)
    _add_bootstrap(moved_archive)
    environment = _child_environment(
        runtime_root, dependency_root, moved_archive, configured=True
    )
    script = "import _native_test_child; print(_native_test_child.VALUE)"
    if mode == "module":
        arguments = ["-m", "_native_test_child"]
    elif mode == "worker":
        arguments = [
            "-c",
            "import subprocess, sys; "
            f"raise SystemExit(subprocess.call([sys.executable, '-c', {script!r}]))",
        ]
    else:
        arguments = ["-c", script]

    child = subprocess.run(
        [sys.executable, *arguments],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert child.returncode == 0, child.stderr
    assert child.stdout.strip() == "archive"


def test_builtin_and_frozen_imports_keep_precedence_in_configured_children(
    tmp_path, runtime_loader
):
    runtime_root, dependency_root, archive_path = _stage_runtime(
        tmp_path,
        {
            "_ast.py": "raise RuntimeError('dependency shadowed builtin')\n",
            "__hello__.py": "raise RuntimeError('dependency shadowed frozen module')\n",
        },
    )
    _add_bootstrap(archive_path)

    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "import _ast, __hello__; print(_ast.__spec__.origin, __hello__.__spec__.origin)",
        ],
        cwd=tmp_path,
        env=_child_environment(runtime_root, dependency_root, archive_path, configured=True),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert child.returncode == 0, child.stderr
    assert child.stdout.strip().endswith("built-in frozen")


def test_root_init_source_is_an_archived_module_not_a_package(tmp_path, runtime_loader):
    runtime_root, dependency_root, archive_path = _stage_runtime(
        tmp_path, {"__init__.py": "VALUE = 'archive'\n"}
    )
    (dependency_root / "__init__.py").write_text("VALUE = 'loose'\n")
    _add_bootstrap(archive_path)

    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "import __init__; "
            "assert __init__.__spec__.submodule_search_locations is None; "
            "assert not hasattr(__init__, '__path__'); "
            "print(__init__.VALUE)",
        ],
        cwd=tmp_path,
        env=_child_environment(runtime_root, dependency_root, archive_path, configured=True),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert child.returncode == 0, child.stderr
    assert child.stdout.strip() == "archive"


def test_unconfigured_python_does_not_activate_loader(tmp_path, runtime_loader):
    runtime_root, dependency_root, archive_path = _stage_runtime(
        tmp_path,
        {"_native_test_unconfigured.py": "VALUE = 'archive'\n"},
        payloads={"_native_test_unconfigured.py": None},
        index={"version": 999, "modules": {}},
    )
    (dependency_root / "_native_test_unconfigured.py").write_text("VALUE = 'loose'\n")
    _add_bootstrap(archive_path)

    child = subprocess.run(
        [sys.executable, "-c", "import _native_test_unconfigured; print(_native_test_unconfigured.VALUE)"],
        cwd=tmp_path,
        env=_child_environment(runtime_root, dependency_root, archive_path, configured=False),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert child.returncode == 0, child.stderr
    assert child.stdout.strip() == "loose"


@pytest.mark.parametrize("failure", ["missing-archive", "invalid-index", "missing-code"])
def test_configured_activation_and_indexed_import_failures_stop_children(
    tmp_path, runtime_loader, failure
):
    runtime_root, dependency_root, archive_path = _stage_runtime(
        tmp_path,
        {"_native_test_failed_child.py": "VALUE = 'loose'\n"},
        payloads={"_native_test_failed_child.py": None},
        index={"version": 999, "modules": {}} if failure == "invalid-index" else None,
    )
    _add_bootstrap(archive_path)
    environment = _child_environment(runtime_root, dependency_root, archive_path, configured=True)
    if failure == "missing-archive":
        archive_path.unlink()
        environment["PYTHONPATH"] = os.pathsep.join([str(_HOST_ROOT), str(dependency_root)])
    script = "import _native_test_failed_child; print('application executed')"

    child = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert child.returncode != 0
    assert "application executed" not in child.stdout
    assert child.stderr
    if failure != "missing-code":
        assert "native Python runtime activation failed" in child.stderr
    else:
        assert "native dependency bytecode" in child.stderr
