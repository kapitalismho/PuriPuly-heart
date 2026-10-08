import _imp
import importlib.machinery
import importlib.util
import marshal
import os
import sys
import types
import zipimport


def _path_key(path):
    return os.path.normcase(os.path.abspath(path))


class _DependencyLoader(importlib.machinery.SourceFileLoader):
    def __init__(self, fullname, source, archive, member):
        super().__init__(fullname, source)
        self._archive = archive
        self._member = member

    def get_code(self, fullname):
        source = self.get_filename(fullname)
        try:
            data = self._archive.get_data(self._member)
        except OSError as exc:
            raise ImportError(
                f"missing native dependency bytecode: {self._member}",
                name=fullname,
                path=source,
            ) from exc
        if (
            len(data) < 16
            or data[:4] != importlib.util.MAGIC_NUMBER
            or int.from_bytes(data[4:8], "little") & ~3
        ):
            raise ImportError(
                f"invalid native dependency bytecode: {self._member}",
                name=fullname,
                path=source,
            )
        try:
            code = marshal.loads(memoryview(data)[16:])
        except (EOFError, ValueError, TypeError) as exc:
            raise ImportError(
                f"invalid native dependency code: {self._member}",
                name=fullname,
                path=source,
            ) from exc
        if not isinstance(code, types.CodeType):
            raise ImportError(
                f"native dependency bytecode contains no code: {self._member}",
                name=fullname,
                path=source,
            )
        _imp._fix_co_filename(code, source)
        return code


class _DependencyFinder:
    def __init__(self, runtime_root, archive_path):
        self.runtime_root = runtime_root
        self.archive_path = archive_path
        self._archive = zipimport.zipimporter(archive_path)
        index = marshal.loads(self._archive.get_data("_native_dependencies.index"))
        if (
            not isinstance(index, dict)
            or index.get("version") != 1
            or not isinstance(index.get("modules"), dict)
        ):
            raise ValueError("invalid native dependency index")
        self._modules = index["modules"]
        dependency_root = os.path.join(runtime_root, "site-packages")
        for fullname, relative_source in self._modules.items():
            if not isinstance(fullname, str) or not isinstance(relative_source, str):
                raise ValueError("invalid native dependency index entry")
            source = os.path.join(dependency_root, relative_source.replace("/", os.sep))
            directory = os.path.dirname(source)
            if relative_source.endswith("/__init__.py"):
                directory = os.path.dirname(directory)
            self._modules[fullname] = (
                _path_key(directory),
                source,
                "_native_dependencies/" + relative_source[:-3] + ".pyc",
            )

    def find_spec(self, fullname, path=None, target=None):
        entry = self._modules.get(fullname)
        if entry is None:
            return None
        directory, source, member = entry
        preceding = None
        for search_path in sys.path if path is None else path:
            if not isinstance(search_path, str):
                continue
            if _path_key(search_path) == directory:
                if preceding is not None:
                    spec = importlib.machinery.PathFinder.find_spec(fullname, preceding, target)
                    if spec is not None and spec.loader is not None:
                        return spec
                loader = _DependencyLoader(fullname, source, self._archive, member)
                return importlib.util.spec_from_file_location(fullname, source, loader=loader)
            if preceding is None:
                preceding = []
            preceding.append(search_path)
        return None


def install(runtime_root):
    runtime_root = os.fspath(runtime_root)
    if not runtime_root:
        raise ValueError("native runtime root is empty")
    runtime_root = os.path.abspath(runtime_root)
    resource_root = os.environ.get("PURIPULY_HEART_NATIVE_RESOURCE_ROOT")
    if resource_root is None:
        resource_root = os.path.join(runtime_root, "app")
    elif not resource_root:
        raise ValueError("native resource root is empty")
    archive_path = os.path.abspath(os.path.join(resource_root, "python.zip"))
    for finder in sys.meta_path:
        if isinstance(finder, _DependencyFinder):
            if _path_key(finder.runtime_root) != _path_key(runtime_root) or _path_key(
                finder.archive_path
            ) != _path_key(archive_path):
                raise RuntimeError(
                    "native dependency loader is already configured for another runtime"
                )
            return
    position = sys.meta_path.index(importlib.machinery.PathFinder)
    finder = _DependencyFinder(runtime_root, archive_path)
    sys.meta_path.insert(position, finder)
