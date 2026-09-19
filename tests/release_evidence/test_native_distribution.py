from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

from puripuly_heart.release_evidence.native_distribution import (
    NativeArtifactLayout,
    compile_application,
    filter_requirements,
    finalize_soxr_wheel,
    stage_product_metadata,
    verify_installed_soxr_record,
    verify_wheel_record,
)


def _record_digest(data: bytes) -> str:
    import base64

    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


def test_native_layout_generates_the_host_runtime_paths() -> None:
    root = Path(__file__).resolve().parents[2]
    layout = NativeArtifactLayout.load(root / "native/windows_host/artifact-layout.json")

    header = layout.render_cpp_header()

    assert layout.values["host_executable"] == "PuriPulyHeart.exe"
    assert layout.values["python_executable"] == "python.exe"
    assert layout.values["dependency_root"] == "site-packages"
    assert 'kApplicationRoot[] = L"app"' in header
    assert 'kDependencyRoot[] = L"site-packages"' in header


def test_requirement_filter_removes_viewer_and_replaces_soxr_as_whole_stanzas(
    tmp_path: Path,
) -> None:
    source = tmp_path / "requirements.txt"
    destination = tmp_path / "native.txt"
    source.write_text(
        "# generated\n"
        "flet==1.0.0 \\\n    --hash=sha256:flet\n"
        "flet-desktop==1.0.0 \\\n    --hash=sha256:viewer\n"
        "six==1.17.0 \\\n    --hash=sha256:six\n"
        "proc-tap==1.1.1 ; platform_machine == 'AMD64' \\\n    --hash=sha256:proc\n"
        "psutil==7.2.2 ; sys_platform == 'win32' \\\n    --hash=sha256:psutil\n"
        "soxr==1.1.0 \\\n    --hash=sha256:soxr\n"
        "    # via product\n",
        encoding="utf-8",
    )

    assert sorted(filter_requirements(source, destination)) == ["flet-desktop", "soxr"]
    filtered = destination.read_text(encoding="utf-8")
    assert "flet==1.0.0" in filtered
    assert "six==1.17.0" in filtered
    assert "viewer" not in filtered
    assert "proc-tap==1.1.1 \\" in filtered
    assert "psutil==7.2.2 \\" in filtered
    assert "platform_machine" not in filtered
    assert "sys_platform" not in filtered
    assert "soxr==" not in filtered


def test_finalized_soxr_wheel_owns_both_native_runtime_files(tmp_path: Path) -> None:
    source = tmp_path / "source.whl"
    destination = tmp_path / "soxr-1.1.0-cp312-abi3-win_amd64.whl"
    dll = tmp_path / "soxr.dll"
    pyd = b"native-extension"
    dll.write_bytes(b"runtime-dll")
    record_name = "soxr-1.1.0.dist-info/RECORD"
    members = {
        "soxr/__init__.py": b"",
        "soxr/soxr_ext.pyd": pyd,
        "soxr-1.1.0.dist-info/METADATA": b"Name: soxr\nVersion: 1.1.0\n",
    }
    rows = [[name, _record_digest(data), str(len(data))] for name, data in members.items()]
    rows.append([record_name, "", ""])
    with zipfile.ZipFile(source, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
        output = __import__("io").StringIO()
        writer = csv.writer(output, lineterminator="\n")
        writer.writerows(rows)
        archive.writestr(record_name, output.getvalue())

    finalize_soxr_wheel(source, dll, destination)
    verify_wheel_record(destination)
    installed = tmp_path / "site-packages"
    with zipfile.ZipFile(destination) as archive:
        archive.extractall(installed)
    installed_identities = verify_installed_soxr_record(installed)
    assert set(installed_identities) == {"soxr/soxr.dll", "soxr/soxr_ext.pyd"}

    with zipfile.ZipFile(destination) as archive:
        assert archive.read("soxr/soxr_ext.pyd") == pyd
        assert archive.read("soxr/soxr.dll") == b"runtime-dll"


def test_product_metadata_is_non_editable_and_record_owned(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "puripuly-heart"\nversion = "2.7.0"\ndescription = "Product"\n',
        encoding="utf-8",
    )
    site_packages = tmp_path / "site-packages"

    result = stage_product_metadata(site_packages, pyproject)

    metadata = site_packages / result["metadata"]
    assert (metadata / "INSTALLER").read_text(encoding="utf-8") == "native-experimental\n"
    assert not (metadata / "direct_url.json").exists()
    rows = list(csv.reader((metadata / "RECORD").read_text(encoding="utf-8").splitlines()))
    assert {row[0] for row in rows} == {
        f"{metadata.name}/INSTALLER",
        f"{metadata.name}/METADATA",
        f"{metadata.name}/RECORD",
        f"{metadata.name}/WHEEL",
    }


def test_application_bytecode_is_checked_hash_optimization_zero_and_declared_build_derived(
    tmp_path: Path,
) -> None:
    app = tmp_path / "app"
    package = app / "example"
    package.mkdir(parents=True)
    (app / "product_bootstrap.py").write_text("VALUE = 1\n", encoding="utf-8")
    (package / "__init__.py").write_text("VALUE = 2\n", encoding="utf-8")

    result = compile_application(app)

    assert result["python_optimize"] == 0
    assert (app / "product_bootstrap.pyc").is_file()
    assert {entry["provenance"] for entry in result["bytecode"]} == {"build-derived"}
    assert {entry["invalidation_mode"] for entry in result["bytecode"]} == {"checked-hash"}
    for entry in result["bytecode"]:
        payload = (tmp_path / entry["path"]).read_bytes()
        assert payload[4:8] == b"\x03\x00\x00\x00"


def test_embedded_bootstrap_only_persists_bounded_uncaught_error_diagnostics(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[2]
    template = (root / "native/windows_host/python_bootstrap.py.in").read_text(encoding="utf-8")
    modules = tmp_path / "modules"
    modules.mkdir()
    bridge_output = tmp_path / "bridge.json"
    (modules / "certifi.py").write_text("def where(): return __file__\n", encoding="utf-8")
    (modules / "dart_bridge.py").write_text(
        "import os\n"
        "def send_bytes(port, payload):\n"
        "    open(os.environ['BRIDGE_OUTPUT'], 'wb').write(payload)\n",
        encoding="utf-8",
    )

    def invoke(
        module_source: str,
        *,
        profile: Path | None = None,
    ) -> tuple[subprocess.CompletedProcess[str], dict[str, object], Path]:
        (modules / "probe.py").write_text(module_source, encoding="utf-8")
        script = (
            template.replace("{argv}", "['PuriPulyHeart']")
            .replace("{host_executable}", repr(str(tmp_path / "PuriPulyHeart.exe")))
            .replace("{module_name}", repr("probe"))
            .replace("{error_exit_code}", "255")
        )
        bridge_output.unlink(missing_ok=True)
        profile = profile or tmp_path / "profile"
        environment = {
            "PYTHONPATH": str(root / "src") + __import__("os").pathsep + str(modules),
            "LOCALAPPDATA": str(profile),
            "PURIPULY_HEART_NATIVE_RESOURCE_ROOT": str(root),
            "PURIPULY_HEART_NATIVE_RUNTIME_ROOT": str(root),
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
