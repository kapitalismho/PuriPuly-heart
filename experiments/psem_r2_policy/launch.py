from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping, Sequence

REVISION = "PSEM-R2-CAPSULE-LAUNCHER-1"
EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
PIN_PATH = EXP / "RUNTIME_PIN.json"
_CHUNK = 1 << 20
_PACKAGE_INIT = "from __future__ import annotations\n"
_MODULE_NAMES = (
    "experiments.psem_r2_policy.run",
    "experiments.psem_r2_policy.budget",
    "experiments.psem_r2_policy.phase",
    "experiments.psem_r2_policy.metrics",
    "tests.helpers.translation_owners",
)
_EPILOG = """\
runtime:
  the capsule is the pinned af26d1d3 product runtime archive (src, prompts,
  tests, pyproject.toml, LICENSE) plus the current canonical R2 harness/config
  and its four PSEM tests, extracted once per input fingerprint into
  experiments/psem_r2_policy/.capsule/<fingerprint>. run.py is the entry, so its
  ROOT/src bootstrap selects the pinned runtime, test helpers, and prompts.

modes:
  (default)     forward --smoke, --offline-replay, --paid, --phase, --meeting,
                and --wav unchanged to the capsule entry run.py
  --tests       run only the four PSEM tests inside the capsule source and
                pinned test support
  --prepare     build or reuse the capsule and print its build manifest JSON

bindings inside the capsule process:
  cash ledger  -> experiments/psem_r2_policy/artifacts/budget_ledger.json
  case outputs -> experiments/psem_r2_policy/artifacts
  run artifacts-> capsule experiments/psem_r2_policy/artifacts

examples:
  .venv/Scripts/python.exe experiments/psem_r2_policy/launch.py --prepare
  .venv/Scripts/python.exe experiments/psem_r2_policy/launch.py --smoke
  .venv/Scripts/python.exe experiments/psem_r2_policy/launch.py --tests
  .venv/Scripts/python.exe experiments/psem_r2_policy/launch.py --paid --phase dev --meeting ES2009a
"""


class LaunchError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalized_prompt_text(raw: str) -> str:
    return raw.replace("\r\n", "\n").strip()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_pin() -> dict[str, Any]:
    if not PIN_PATH.is_file():
        raise LaunchError(f"runtime pin is missing: {PIN_PATH}")
    try:
        return json.loads(PIN_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LaunchError(f"runtime pin is not valid JSON: {PIN_PATH}: {exc}") from exc


def _require_interpreter(pin: Mapping[str, Any]) -> None:
    required = str(pin["python"]["requires"])
    current = f"{sys.version_info.major}.{sys.version_info.minor}"
    if current != required:
        raise LaunchError(f"python {required} is required by the runtime pin; running {current}")
    canonical = (ROOT / str(pin["python"]["canonical"])).resolve()
    if Path(sys.executable).resolve() != canonical:
        relative = Path(__file__).resolve().relative_to(ROOT).as_posix()
        raise LaunchError(f"run with the canonical interpreter: {pin['python']['canonical']} {relative}")


def _package_dir(pin: Mapping[str, Any]) -> str:
    return str(pin["capsule"]["package_dir"])


def _overlay_plan(pin: Mapping[str, Any]) -> list[tuple[str, Path]]:
    canonical = pin["canonical"]
    plan: list[tuple[str, Path]] = []
    for name in canonical["harness_files"]:
        plan.append((f"{_package_dir(pin)}/{name}", EXP / str(name)))
    for name in canonical["config_files"]:
        plan.append((f"{_package_dir(pin)}/{name}", EXP / str(name)))
    for source in sorted(canonical["tests"]):
        plan.append((str(canonical["tests"][source]), EXP / str(source)))
    return plan


def _fingerprint(
    pin: Mapping[str, Any],
    archive_sha: str,
    overlay: Sequence[Mapping[str, Any]],
    generated: Sequence[Mapping[str, Any]],
) -> str:
    payload = {
        "revision": REVISION,
        "pin_revision": pin["revision"],
        "entry_module": pin["entry_module"],
        "archive": {
            "file": pin["runtime_archive"]["file"],
            "sha256": archive_sha,
        },
        "overlay": [{"capsule": item["capsule"], "sha256": item["sha256"]} for item in overlay],
        "generated": [{"capsule": item["capsule"], "sha256": item["sha256"]} for item in generated],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _capsule_state(
    manifest_path: Path,
    fingerprint: str,
    overlay: Sequence[Mapping[str, Any]],
    generated: Sequence[Mapping[str, Any]],
    capsule_root: Path,
) -> tuple[dict[str, Any] | None, str]:
    if not manifest_path.is_file():
        return None, "manifest is missing"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None, "manifest is not valid JSON"
    if manifest.get("revision") != REVISION:
        return None, "manifest revision mismatch"
    if manifest.get("fingerprint") != fingerprint:
        return None, "manifest fingerprint mismatch"
    if manifest.get("overlay") != [dict(item) for item in overlay]:
        return None, "manifest overlay mismatch"
    if manifest.get("generated") != [dict(item) for item in generated]:
        return None, "manifest generated-entry mismatch"
    for item in [*manifest["overlay"], *manifest["generated"]]:
        path = capsule_root / str(item["capsule"])
        if not path.is_file():
            return None, f"capsule file is missing: {item['capsule']}"
        if _sha256_file(path) != str(item["sha256"]):
            return None, f"capsule file hash mismatch: {item['capsule']}"
    if not (capsule_root / str(manifest["prompt"]["file"])).is_file():
        return None, "capsule prompt is missing"
    return manifest, "ok"


def _verify_prompt_file(path: Path, pin: Mapping[str, Any]) -> dict[str, str]:
    if not path.is_file():
        raise LaunchError(f"pinned prompt is missing in the capsule: {path}")
    expected_file = str(pin["prompt"]["file_sha256"])
    observed_file = _sha256_file(path)
    if observed_file != expected_file:
        raise LaunchError(f"pinned prompt file sha256 mismatch: {path}: {observed_file} != {expected_file}")
    expected_text = str(pin["prompt"]["stripped_sha256"])
    observed_text = _sha256_text(_normalized_prompt_text(path.read_text(encoding="utf-8")))
    if observed_text != expected_text:
        raise LaunchError(f"pinned prompt sha256 mismatch: {path}: {observed_text} != {expected_text}")
    return {"file": str(path), "file_sha256": observed_file, "stripped_sha256": observed_text}


def _publish(pin: Mapping[str, Any], capsule_root: Path, archive: Path, manifest: Mapping[str, Any]) -> None:
    parent = capsule_root.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"build-{capsule_root.name[:8]}-", dir=str(parent)))
    try:
        with tarfile.open(archive, "r:gz") as handle:
            handle.extractall(staging, filter="data")
        for item in manifest["overlay"]:
            destination = staging / str(item["capsule"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / str(item["source"]), destination)
        for item in manifest["generated"]:
            destination = staging / str(item["capsule"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(_PACKAGE_INIT.encode("utf-8"))
        _verify_prompt_file(staging / str(pin["prompt"]["file"]), pin)
        _write_json(staging / str(pin["capsule"]["manifest"]), manifest)
        if capsule_root.exists():
            shutil.rmtree(staging, ignore_errors=True)
            return
        try:
            os.replace(staging, capsule_root)
        except OSError:
            if not capsule_root.is_dir():
                raise
            shutil.rmtree(staging, ignore_errors=True)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _resolve_capsule(pin: Mapping[str, Any]) -> dict[str, Any]:
    archive = EXP / str(pin["runtime_archive"]["file"])
    if not archive.is_file():
        raise LaunchError(f"runtime archive is missing: {archive}")
    observed_archive = _sha256_file(archive)
    expected_archive = str(pin["runtime_archive"]["sha256"])
    if observed_archive != expected_archive:
        raise LaunchError(
            f"runtime archive sha256 mismatch: {archive.name}: {observed_archive} != {expected_archive}"
        )
    plan = _overlay_plan(pin)
    for capsule_rel, source in plan:
        if not source.is_file():
            raise LaunchError(f"canonical input is missing: {source} (capsule {capsule_rel})")
    overlay = [
        {
            "capsule": capsule_rel,
            "source": source.relative_to(ROOT).as_posix(),
            "sha256": _sha256_file(source),
        }
        for capsule_rel, source in plan
    ]
    generated = [
        {
            "capsule": f"{pin['capsule']['experiments_dir']}/__init__.py",
            "sha256": _sha256_text(_PACKAGE_INIT),
        }
    ]
    fingerprint = _fingerprint(pin, observed_archive, overlay, generated)
    capsule_root = EXP / str(pin["capsule"]["dir"]) / fingerprint[: int(pin["capsule"]["name_length"])]
    manifest_path = capsule_root / str(pin["capsule"]["manifest"])
    stored, reason = _capsule_state(manifest_path, fingerprint, overlay, generated, capsule_root)
    reused = stored is not None
    if stored is None:
        manifest = {
            "revision": REVISION,
            "fingerprint": fingerprint,
            "runtime_archive": {
                "file": pin["runtime_archive"]["file"],
                "sha256": observed_archive,
                "commit": pin["runtime_archive"]["commit"],
            },
            "prompt": dict(pin["prompt"]),
            "entry_module": pin["entry_module"],
            "expected_tests": int(pin["expected_tests"]),
            "overlay": overlay,
            "generated": generated,
        }
        _publish(pin, capsule_root, archive, manifest)
        stored, reason = _capsule_state(manifest_path, fingerprint, overlay, generated, capsule_root)
        if stored is None:
            raise LaunchError(f"capsule is not reusable ({reason}): {capsule_root}; delete it and rerun")
    return {
        "root": capsule_root,
        "manifest_path": manifest_path,
        "manifest": stored,
        "fingerprint": fingerprint,
        "reused": reused,
        "archive": archive,
        "archive_sha256": observed_archive,
    }


def _capsule_modules(capsule_root: Path) -> dict[str, Any]:
    resolved_root = capsule_root.resolve()
    if str(resolved_root) not in sys.path:
        sys.path.insert(0, str(resolved_root))
    modules: dict[str, Any] = {}
    for name in _MODULE_NAMES:
        module = importlib.import_module(name)
        origin = Path(str(getattr(module, "__file__", ""))).resolve()
        if not origin.is_relative_to(resolved_root):
            raise LaunchError(f"{name} resolved outside the capsule: {origin}")
        modules[name] = module
    return modules


def _module_paths(modules: Mapping[str, Any]) -> dict[str, str]:
    return {name: str(Path(str(module.__file__)).resolve()) for name, module in modules.items()}


def _prompt_probe(pin: Mapping[str, Any], capsule_root: Path) -> dict[str, str]:
    resolved_root = capsule_root.resolve()
    module = importlib.import_module("puripuly_heart.config.prompts")
    origin = Path(str(getattr(module, "__file__", ""))).resolve()
    if not origin.is_relative_to(resolved_root):
        raise LaunchError(f"prompt loader resolved outside the capsule: {origin}")
    directory = Path(module.get_prompts_dir()).resolve()
    if not directory.is_relative_to(resolved_root):
        raise LaunchError(f"prompt directory resolved outside the capsule: {directory}")
    digest = _sha256_text(_normalized_prompt_text(module.get_default_prompt()))
    expected = str(pin["prompt"]["stripped_sha256"])
    if digest != expected:
        raise LaunchError(f"default prompt sha256 mismatch: {digest} != {expected}")
    return {
        "prompt_dir": str(directory),
        "prompt_loader": str(origin),
        "default_prompt_sha256": digest,
    }


def _artifact_paths(pin: Mapping[str, Any], capsule_root: Path) -> tuple[Path, Path, Path]:
    artifacts = EXP / str(pin["case_outputs"])
    ledger = EXP / str(pin["ledger"])
    runtime_artifacts = capsule_root / _package_dir(pin) / "artifacts"
    return artifacts, ledger, runtime_artifacts


def _uses_ledger(argv: Sequence[str]) -> bool:
    for item in argv:
        if item in {"--paid", "--phase"} or item.startswith("--phase="):
            return True
    return False


def _execute_capsule(pin: Mapping[str, Any], capsule: Mapping[str, Any], argv: Sequence[str]) -> int:
    capsule_root = Path(capsule["root"])
    prompt = _verify_prompt_file(capsule_root / str(pin["prompt"]["file"]), pin)
    modules = _capsule_modules(capsule_root)
    probe = _prompt_probe(pin, capsule_root)
    artifacts, ledger, runtime_artifacts = _artifact_paths(pin, capsule_root)
    artifacts.mkdir(parents=True, exist_ok=True)
    if _uses_ledger(argv) and not ledger.is_file():
        raise LaunchError(f"canonical budget ledger is missing: {ledger}")
    run_module = modules["experiments.psem_r2_policy.run"]
    phase_module = modules["experiments.psem_r2_policy.phase"]
    run_module.LEDGER_PATH = ledger
    phase_module.ARTIFACTS = artifacts
    print(
        json.dumps(
            {
                "mode": "run",
                "capsule_root": str(capsule_root),
                "fingerprint": capsule["fingerprint"],
                "ledger_path": str(run_module.LEDGER_PATH),
                "case_output_dir": str(phase_module.ARTIFACTS),
                "runtime_artifact_dir": str(runtime_artifacts),
                "modules": _module_paths(modules),
                "prompt": prompt,
                "prompt_probe": probe,
            },
            ensure_ascii=False,
        ),
        file=sys.stderr,
    )
    return int(run_module.main(list(argv)))


def _read_junit(path: Path) -> dict[str, int]:
    if not path.is_file():
        return {}
    root = ET.parse(path).getroot()
    counts = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    for case in root.iter("testcase"):
        counts["tests"] += 1
        tags = {child.tag for child in case}
        if "failure" in tags:
            counts["failures"] += 1
        elif "error" in tags:
            counts["errors"] += 1
        elif "skipped" in tags:
            counts["skipped"] += 1
    return counts


def _run_tests(pin: Mapping[str, Any], capsule: Mapping[str, Any]) -> int:
    capsule_root = Path(capsule["root"])
    tests = [
        str(item["capsule"])
        for item in capsule["manifest"]["overlay"]
        if str(item["capsule"]).startswith("tests/")
    ]
    if not tests:
        raise LaunchError(f"capsule has no PSEM tests: {capsule_root}")
    report_dir = Path(tempfile.mkdtemp(prefix="psem-r2-junit-"))
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        f"--junitxml={report_dir / 'report.xml'}",
        *tests,
    ]
    try:
        completed = subprocess.run(command, cwd=str(capsule_root))
        counts = _read_junit(report_dir / "report.xml")
    finally:
        shutil.rmtree(report_dir, ignore_errors=True)
    payload: dict[str, Any] = {
        "mode": "tests",
        "capsule_root": str(capsule_root),
        "fingerprint": capsule["fingerprint"],
        "returncode": int(completed.returncode),
        "expected_tests": int(pin["expected_tests"]),
        "tests": tests,
        **counts,
    }
    print(json.dumps(payload, ensure_ascii=False))
    if counts.get("tests") != int(pin["expected_tests"]):
        print(
            f"launch warning: collected {counts.get('tests')} tests, pin expects {pin['expected_tests']}",
            file=sys.stderr,
        )
    return int(completed.returncode)


def _prepare(pin: Mapping[str, Any], capsule: Mapping[str, Any]) -> int:
    capsule_root = Path(capsule["root"])
    prompt = _verify_prompt_file(capsule_root / str(pin["prompt"]["file"]), pin)
    modules = _module_paths(_capsule_modules(capsule_root))
    probe = _prompt_probe(pin, capsule_root)
    artifacts, ledger, runtime_artifacts = _artifact_paths(pin, capsule_root)
    tests = [
        str(item["capsule"])
        for item in capsule["manifest"]["overlay"]
        if str(item["capsule"]).startswith("tests/")
    ]
    payload = {
        "mode": "prepare",
        "revision": REVISION,
        "fingerprint": capsule["fingerprint"],
        "reused": bool(capsule["reused"]),
        "capsule_root": str(capsule_root),
        "capsule_manifest": str(capsule["manifest_path"]),
        "entry_module": pin["entry_module"],
        "runtime_archive": {
            "file": str(pin["runtime_archive"]["file"]),
            "sha256": capsule["archive_sha256"],
            "verified": capsule["archive_sha256"] == str(pin["runtime_archive"]["sha256"]),
            "commit": str(pin["runtime_archive"]["commit"]),
        },
        "prompt": prompt,
        "prompt_probe": probe,
        "python": {
            "executable": sys.executable,
            "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "requires": str(pin["python"]["requires"]),
        },
        "modules": modules,
        "bindings": {
            "ledger_path": str(ledger),
            "case_output_dir": str(artifacts),
            "runtime_artifact_dir": str(runtime_artifacts),
        },
        "tests": tests,
        "expected_tests": int(pin["expected_tests"]),
        "target": {"experiment_dir": str(EXP), "root": str(ROOT)},
        "manifest": capsule["manifest"],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="launch.py",
        description="Run the pinned PSEM R2 experiment runtime from its generated capsule.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--tests",
        action="store_true",
        help="run only the four PSEM tests inside the capsule source and pinned test support",
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="build or reuse the capsule and print its build manifest JSON",
    )
    args, forwarded = parser.parse_known_args(list(argv) if argv is not None else None)
    try:
        if args.tests and args.prepare:
            raise LaunchError("--tests and --prepare cannot be combined")
        if (args.tests or args.prepare) and forwarded:
            raise LaunchError(f"--tests/--prepare accept no further arguments: {' '.join(forwarded)}")
        pin = _load_pin()
        _require_interpreter(pin)
        os.environ.pop("PURIPULY_HEART_PROMPTS_DIR", None)
        capsule = _resolve_capsule(pin)
        if args.prepare:
            return _prepare(pin, capsule)
        if args.tests:
            return _run_tests(pin, capsule)
        return _execute_capsule(pin, capsule, forwarded)
    except LaunchError as exc:
        print(f"launch error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
