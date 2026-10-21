from __future__ import annotations

import importlib.util
import json
import marshal
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from puripuly_heart.release_evidence.native_distribution import bundle_runtime, compile_runtime
from tests.helpers.paths import REPO_ROOT


@pytest.mark.skipif(sys.version_info[:2] != (3, 14), reason="native SDK uses CPython 3.14")
def test_stdlib_archive_supports_early_interpreter_bootstrap_without_loose_lib(
    tmp_path: Path, staged_archive_runtime: tuple[Path, Path, dict[str, object]]
) -> None:
    import encodings

    layout, bytecode, _ = staged_archive_runtime
    shutil.copytree(
        Path(encodings.__file__).parent,
        tmp_path / "Lib/encodings",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    bytecode.write_text(json.dumps(compile_runtime(tmp_path, layout)), encoding="utf-8")
    bundle_runtime(tmp_path, layout, bytecode)
    shutil.rmtree(tmp_path / "Lib")
    archive = tmp_path / "python314.zip"
    executable = sys.executable
    if sys.platform == "win32":
        executable = str(tmp_path / "python.exe")
        shutil.copy2(sys._base_executable, executable)
        shutil.copy2(Path(sys.base_prefix) / "python314.dll", tmp_path / "python314.dll")
    else:
        (tmp_path / "lib").mkdir()
        shutil.copy2(archive, tmp_path / "lib/python314.zip")
        archive = tmp_path / "lib/python314.zip"
    environment = dict(os.environ)
    for name in tuple(environment):
        if name.startswith("PYTHON") or name.startswith("PURIPULY_HEART_NATIVE_"):
            del environment[name]
    environment.update(PYTHONHOME=str(tmp_path), PYTHONUTF8="1")
    script = (
        "import encodings, sys, sdk_probe\n"
        f"assert encodings.__file__.startswith({str(archive)!r})\n"
        "assert sdk_probe.VALUE == 77\n"
        "assert sys.flags.no_site == 1\n"
        "print('stdlib archive bootstrap')\n"
    )
    script_path = tmp_path / "bootstrap_probe.py"
    script_path.write_text(script, encoding="utf-8")
    result = subprocess.run(
        [executable, "-S", "-B", str(script_path)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
        timeout=30,
    )
    assert result.stdout.strip() == "stdlib archive bootstrap"
    assert not (tmp_path / "Lib").exists()


def test_embedded_bootstrap_only_persists_bounded_uncaught_error_diagnostics(
    tmp_path: Path,
) -> None:
    root = REPO_ROOT
    template = (root / "native/windows_host/python_bootstrap.py.in").read_text(encoding="utf-8")
    modules = tmp_path / "site-packages"
    modules.mkdir()
    bridge_output = tmp_path / "bridge.json"
    (modules / "certifi.py").write_text("def where(): return __file__\n", encoding="utf-8")
    (modules / "dart_bridge.py").write_text(
        "import os\n"
        "def send_bytes(port, payload):\n"
        "    open(os.environ['BRIDGE_OUTPUT'], 'wb').write(payload)\n",
        encoding="utf-8",
    )
    app = tmp_path / "app"
    app.mkdir()
    for filename in ("_puripuly_native_runtime.py", "sitecustomize.py"):
        shutil.copy2(root / "native/windows_host" / filename, app / filename)
    archive_path = app / "python.zip"

    def invoke(
        module_source: str,
        *,
        profile: Path | None = None,
    ) -> tuple[subprocess.CompletedProcess[str], dict[str, object], Path]:
        (modules / "probe.py").write_text(module_source, encoding="utf-8")
        compiled = compile_runtime(tmp_path, root / "native/windows_host/artifact-layout.json")
        index = {}
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for entry in compiled["bytecode"]:
                source = Path(entry["source"])
                if source.parts[0] == "site-packages":
                    relative = source.relative_to("site-packages").as_posix()
                    index[source.stem] = relative
                    member = "_native_dependencies/" + str(
                        Path(relative).with_suffix(".pyc")
                    ).replace("\\", "/")
                else:
                    member = source.with_suffix(".pyc").name
                archive.write(tmp_path / entry["path"], member)
            archive.writestr(
                "_native_dependencies.index", marshal.dumps({"version": 1, "modules": index})
            )
        script = (
            template.replace("{argv}", "['PuriPulyHeart']")
            .replace("{host_executable}", repr(str(tmp_path / "PuriPulyHeart.exe")))
            .replace("{module_name}", repr("probe"))
            .replace("{error_exit_code}", "255")
        )
        bridge_output.unlink(missing_ok=True)
        profile = profile or tmp_path / "profile"
        environment = {
            "PYTHONPATH": os.pathsep.join((str(archive_path), str(modules), str(root / "src"))),
            "LOCALAPPDATA": str(profile),
            "PURIPULY_HEART_NATIVE_RESOURCE_ROOT": str(app),
            "PURIPULY_HEART_NATIVE_RUNTIME_ROOT": str(tmp_path),
            "PURIPULY_HEART_NATIVE_HOST_EXECUTABLE": str(tmp_path / "PuriPulyHeart.exe"),
            "PURIPULY_HEART_NATIVE_PYTHON_EXECUTABLE": sys.executable,
            "FLET_DART_BRIDGE_EXIT_PORT": "7",
            "BRIDGE_OUTPUT": str(bridge_output),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf-8",
        }
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=tmp_path,
            env=environment,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=True,
        )
        payload = json.loads(bridge_output.read_text(encoding="utf-8"))
        return (
            completed,
            payload,
            profile / "puripuly-heart" / "logs" / "native-bootstrap-error.log",
        )

    normal, normal_payload, error_path = invoke("print('ordinary output')\n")
    assert normal.stdout == "ordinary output\n"
    assert normal_payload == {"code": 0, "error": ""}
    assert not error_path.exists()

    system_exit, system_exit_payload, error_path = invoke("raise SystemExit(23)\n")
    assert system_exit.stderr == ""
    assert system_exit_payload == {"code": 23, "error": ""}
    assert not error_path.exists()

    failed, failed_payload, error_path = invoke(
        "def fail():\n    raise RuntimeError('한' * 100000)\nfail()\n"
    )
    encoded = failed_payload["error"].encode("utf-8")
    assert failed_payload["code"] == 255
    assert len(encoded) <= 65536
    assert "builtins.RuntimeError" in failed_payload["error"]
    assert "probe.py" in failed_payload["error"]
    assert error_path.read_bytes() == encoded
    assert "closed file" not in failed.stderr.lower()

    blocked_profile = tmp_path / "blocked-profile"
    blocked_profile.write_text("not a directory", encoding="utf-8")
    _, blocked_payload, blocked_error_path = invoke(
        "raise LookupError('bridge survives log failure')\n",
        profile=blocked_profile,
    )
    assert blocked_payload["code"] == 255
    assert "builtins.LookupError" in blocked_payload["error"]
    assert not blocked_error_path.exists()


@pytest.fixture
def bootstrap_https_server(tmp_path: Path):
    import datetime
    import ipaddress
    import ssl
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "bootstrap-test")])
    now = datetime.datetime.now(datetime.UTC)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(
            x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    ca_path = tmp_path / "ca.pem"
    ca_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path = tmp_path / "key.pem"
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"verified")

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(ca_path, key_path)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield ca_path, f"https://127.0.0.1:{server.server_port}/"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize(
    "policy,httpx_result,requests_result",
    [
        ("file", "verified", "peer_rejected"),
        ("directory", "peer_rejected", "peer_rejected"),
        ("requests", "peer_rejected", "verified"),
        ("curl", "peer_rejected", "verified"),
        ("unset", "peer_rejected", "peer_rejected"),
        ("invalid-file", "configuration_failed", "peer_rejected"),
        ("invalid-requests", "peer_rejected", "configuration_failed"),
    ],
)
def test_embedded_bootstrap_preserves_independent_ca_policy_and_tls_verification(
    tmp_path: Path,
    bootstrap_https_server,
    policy: str,
    httpx_result: str,
    requests_result: str,
) -> None:
    root = REPO_ROOT
    ca_path, url = bootstrap_https_server
    ca_directory = tmp_path / "empty-ca-directory"
    ca_directory.mkdir()
    policies = {
        "file": {"SSL_CERT_FILE": str(ca_path)},
        "directory": {"SSL_CERT_DIR": str(ca_directory)},
        "requests": {"REQUESTS_CA_BUNDLE": str(ca_path)},
        "curl": {"CURL_CA_BUNDLE": str(ca_path)},
        "unset": {},
        "invalid-file": {"SSL_CERT_FILE": str(tmp_path / "missing-ca.pem")},
        "invalid-requests": {"REQUESTS_CA_BUNDLE": str(tmp_path / "missing-ca.pem")},
    }
    inherited = policies[policy]
    modules = tmp_path / "startup"
    modules.mkdir()
    (modules / "dart_bridge.py").write_text(
        "import json\n"
        "def send_bytes(port, payload):\n"
        "    print(json.dumps({'bootstrap': json.loads(payload)}))\n",
        encoding="utf-8",
    )
    (modules / "_puripuly_native_runtime.py").write_text(
        "def install(root): pass\n", encoding="utf-8"
    )
    ca_keys = ("SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE")
    (modules / "probe.py").write_text(
        "import json, os\n"
        "import httpx, requests\n"
        "from puripuly_heart.core.network_clients import external_client\n"
        "from puripuly_heart.core.network_requests import ExternalRequestsSession\n"
        f"result = {{'environment': {{key: os.environ.get(key) for key in {ca_keys!r}}}}}\n"
        "for name, factory in [('httpx', external_client), ('requests', ExternalRequestsSession)]:\n"
        "    try:\n"
        "        client = factory()\n"
        "    except OSError:\n"
        "        result[name] = 'configuration_failed'\n"
        "        continue\n"
        "    with client:\n"
        "        try:\n"
        f"            response = client.get({url!r}, timeout=5)\n"
        "            response.raise_for_status()\n"
        "            result[name] = response.text\n"
        "        except (httpx.ConnectError, requests.exceptions.SSLError):\n"
        "            result[name] = 'peer_rejected'\n"
        "print(json.dumps(result))\n",
        encoding="utf-8",
    )
    template = (root / "native/windows_host/python_bootstrap.py.in").read_text(encoding="utf-8")
    script = (
        template.replace("{argv}", "['PuriPulyHeart']")
        .replace("{host_executable}", repr(str(tmp_path / "PuriPulyHeart.exe")))
        .replace("{module_name}", repr("probe"))
        .replace("{error_exit_code}", "255")
    )
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in ca_keys and not key.upper().endswith("_PROXY")
    }
    environment.update(
        inherited,
        PYTHONPATH=os.pathsep.join((str(modules), str(root / "src"))),
        LOCALAPPDATA=str(tmp_path / "profile"),
        PURIPULY_HEART_NATIVE_RUNTIME_ROOT=str(tmp_path),
        FLET_DART_BRIDGE_EXIT_PORT="7",
        PYTHONDONTWRITEBYTECODE="1",
        NO_PROXY="*",
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
    )
    result, bridge = map(json.loads, completed.stdout.splitlines())
    expected = dict.fromkeys(ca_keys)
    expected.update(inherited)
    assert result["environment"] == expected
    assert result["httpx"] == httpx_result
    assert result["requests"] == requests_result
    assert bridge == {"bootstrap": {"code": 0, "error": ""}}


def test_bundled_runtime_imports_code_and_flet_resources_without_recompilation(
    tmp_path: Path, staged_archive_runtime: tuple[Path, Path, dict[str, object]]
) -> None:
    layout, bytecode, _ = staged_archive_runtime
    bundled = bundle_runtime(tmp_path, layout, bytecode)
    app = tmp_path / "app"
    archive = app / "python.zip"
    dependencies = tmp_path / "site-packages"
    socket_spec = importlib.util.find_spec("_socket")
    assert socket_spec is not None and socket_spec.origin is not None
    assert (dependencies / Path(socket_spec.origin).name).read_bytes() == Path(
        socket_spec.origin
    ).read_bytes()
    assert (app / "product_bootstrap.pyc").is_file()
    assert not (app / "product_bootstrap.py").exists()
    assert list(app.rglob("*.py")) == []
    assert list(app.rglob("*.pyc")) == [app / "product_bootstrap.pyc"]
    assert not (dependencies / "flet").exists()
    assert (dependencies / "dependency.py").is_file()
    assert list(dependencies.glob("flet-*.dist-info"))
    assert list(dependencies.rglob("*.pyc")) == []
    assert list((tmp_path / "Lib").rglob("*.py")) == []
    assert list((tmp_path / "Lib").rglob("*.pyc")) == []
    assert (dependencies / "shared_ns/empty").is_dir()
    with zipfile.ZipFile(archive) as packed:
        names = packed.namelist()
        assert names == sorted(names)
        assert "puripuly_heart/ui/" in names
        assert "puripuly_heart/ui/consumer.pyc" in names
        assert "puripuly_heart/ui/consumer.py" in names
        assert "puripuly_heart/data/payload.txt" not in names
        assert not any(name.startswith("product_bootstrap.") for name in names)
        index = marshal.loads(packed.read("_native_dependencies.index"))["modules"]
        assert index["mixed"] == "mixed/__init__.py"
        assert index["choice"] == "choice/__init__.py"
        assert "mixed._socket" not in index
        assert "masked.child" not in index
        assert index["shared_ns.child"] == "shared_ns/child.py"
    stale = app / "puripuly_heart"
    (stale / "__init__.py").write_text(
        "raise AssertionError('stale app selected')\n", encoding="utf-8"
    )
    (stale / "ui" / "consumer.py").write_text(
        "raise AssertionError('stale module selected')\n", encoding="utf-8"
    )
    (dependencies / "flet").mkdir()
    (dependencies / "flet" / "__init__.py").write_text(
        "raise AssertionError('stale Flet selected')\n", encoding="utf-8"
    )
    script = (
        "import importlib.machinery, importlib.metadata, importlib.resources, importlib.util, inspect, json, os, pkgutil, sys, traceback, zipimport\n"
        f"sys.path[:0] = [{str(archive)!r}, {str(tmp_path / 'python314.zip')!r}, {str(app)!r}, {str(dependencies)!r}]\n"
        "from _puripuly_native_runtime import install\n"
        f"install({str(tmp_path)!r})\n"
        f"os.environ['PURIPULY_HEART_NATIVE_RESOURCE_ROOT'] = {str(app)!r}\n"
        "def reject_source_compilation(*args, **kwargs):\n"
        "    raise AssertionError('archive source compilation')\n"
        "zipimport._compile_source = reject_source_compilation\n"
        "original_get_data = importlib.machinery.SourceFileLoader.get_data\n"
        "def reject_source_reads(self, path):\n"
        f"    if path.endswith('.py') and (path.startswith({str(app)!r}) or path.startswith({str(dependencies)!r})):\n"
        "        raise AssertionError('loose runtime source read')\n"
        "    return original_get_data(self, path)\n"
        "importlib.machinery.SourceFileLoader.get_data = reject_source_reads\n"
        "import flet\n"
        "from puripuly_heart.ui import consumer\n"
        "import mixed, mixed.code, mixed._socket, shared_ns.child, choice, masked, sdk_probe, source_stdlib\n"
        "assert mixed.VALUE == 23 and mixed._socket.socket is not None\n"
        "assert shared_ns.child.VALUE == 5 and choice.VALUE == 31 and masked.VALUE == 17\n"
        "assert sdk_probe.VALUE == 77 and source_stdlib.VALUE == 12\n"
        "assert importlib.import_module('hyphen-module').VALUE == 7\n"
        "assert importlib.metadata.version('mixed') == '1.0'\n"
        "assert importlib.resources.files(mixed).joinpath('payload.txt').read_text() == 'mixed resource'\n"
        "assert importlib.resources.files(source_stdlib).joinpath('payload.txt').read_text() == 'stdlib resource'\n"
        f"assert mixed.__file__ == {str(dependencies / 'mixed/__init__.py')!r}\n"
        f"assert list(mixed.__path__) == [{str(dependencies / 'mixed')!r}]\n"
        f"assert mixed._socket.__file__ == {str(dependencies / 'mixed' / Path(socket_spec.origin).name)!r}\n"
        "assert {'code', '_socket'} <= {entry.name for entry in pkgutil.iter_modules(mixed.__path__)}\n"
        "importlib.machinery.SourceFileLoader.get_data = original_get_data\n"
        "assert \"raise RuntimeError('physical source traceback')\" in inspect.getsource(mixed.code.fail)\n"
        "try:\n"
        "    mixed.code.fail()\n"
        "except RuntimeError:\n"
        "    formatted = traceback.format_exc()\n"
        f"    assert {str(dependencies / 'mixed/code.py')!r} in formatted\n"
        "    assert \"raise RuntimeError('physical source traceback')\" in formatted\n"
        "from puripuly_heart.runtime_layout import current_runtime_layout\n"
        "layout = current_runtime_layout()\n"
        "icons = json.loads(importlib.resources.files('flet.controls.material').joinpath('icons.json').read_text())\n"
        "assert int(flet.Icons.ADD) == icons['ADD']\n"
        f"assert flet.__file__.startswith({str(archive)!r})\n"
        f"assert consumer.__file__.startswith({str(archive)!r})\n"
        f"spec = importlib.util.spec_from_file_location('bootstrap', {str(app / 'product_bootstrap.pyc')!r})\n"
        "bootstrap = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(bootstrap)\n"
        "print(json.dumps([consumer.RESULT, bootstrap.RESULT, "
        "layout.package_resource('data', 'payload.txt').read_text(), "
        "layout.resource('prompts', 'system.txt').read_text()]))\n"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script], check=True, capture_output=True, text=True
    )
    assert json.loads(result.stdout) == [42, 9, "filesystem package data", "filesystem prompt"]
    assert {entry["source"] for entry in bundled["bytecode"]} == {"app/product_bootstrap.py"}
