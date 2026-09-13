from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.psem_r2_policy import launch, phase


def test_freeze_bindings_cover_every_runtime_overlay() -> None:
    runtime_pin = json.loads(phase.RUNTIME_PIN_PATH.read_text(encoding="utf-8"))
    bindings = phase._pin_bindings(runtime_pin)
    package_dir = runtime_pin["capsule"]["package_dir"]
    expected = {
        *(f"{package_dir}/{name}" for name in runtime_pin["canonical"]["harness_files"]),
        *(
            f"{package_dir}/{name}"
            for name in runtime_pin["canonical"]["config_files"]
            if name != "PIN_MANIFEST.json"
        ),
        *runtime_pin["canonical"]["runtime_overrides"],
        *runtime_pin["canonical"]["tests"].values(),
    }

    assert set(bindings) == expected
    assert all(path.is_file() for path in bindings.values())
    identity = phase._runtime_identity(runtime_pin)
    assert (
        identity["runtime_pin_sha256"]
        == hashlib.sha256(phase.RUNTIME_PIN_PATH.read_bytes()).hexdigest()
    )
    assert identity["archive"] == {
        key: runtime_pin["runtime_archive"][key] for key in ("file", "sha256", "commit")
    }
    assert identity["prompt"] == runtime_pin["prompt"]


def test_write_pin_manifest_refuses_missing_holdout_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(phase, "build_pin_manifest", lambda: {"missing_inputs": ["audio:ES2004a"]})

    with pytest.raises(FileNotFoundError, match="audio:ES2004a"):
        phase.write_pin_manifest(tmp_path / "PIN_MANIFEST.json")
    assert not (tmp_path / "PIN_MANIFEST.json").exists()


def test_holdout_unlock_requires_exact_complete_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current = {
        "revision": phase.PIN_REVISION,
        "files": {"runtime.py": "abc"},
        "missing_inputs": [],
    }
    pin_path = tmp_path / "PIN_MANIFEST.json"
    pin_path.write_text(json.dumps(current), encoding="utf-8")
    monkeypatch.setattr(phase, "PIN_PATH", pin_path)
    monkeypatch.setattr(phase, "load_holdout_gate", lambda: {"frozen": True})
    monkeypatch.setattr(phase, "build_pin_manifest", lambda: current)

    assert phase.holdout_unlock_error() is None

    pin_path.write_text(
        json.dumps({**current, "files": {"runtime.py": "changed"}}), encoding="utf-8"
    )
    assert (
        phase.holdout_unlock_error()
        == "holdout pin manifest does not match current frozen runtime and inputs"
    )


def test_prepare_pin_refuses_existing_pin_and_nonfinal_billing_before_capsule_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing = tmp_path / "PIN_MANIFEST.json"
    original = b"{invalid preserved pin"
    existing.write_bytes(original)
    monkeypatch.setattr(launch, "EXP", tmp_path)
    monkeypatch.setattr(launch, "_load_pin", lambda: {})
    monkeypatch.setattr(launch, "_require_interpreter", lambda pin: None)

    def capsule_build_forbidden(*args: object, **kwargs: object) -> dict:
        raise AssertionError("capsule build must not run before preparation checks")

    monkeypatch.setattr(launch, "_resolve_capsule", capsule_build_forbidden)

    assert launch.main(["--prepare-pin"]) == 2
    assert existing.read_bytes() == original

    existing.unlink()
    (tmp_path / "HOLD_OUT_GATE.json").write_text(
        json.dumps({"frozen": True, "pin_required": True}), encoding="utf-8"
    )
    (tmp_path / "BILLING_BOUNDS.json").write_text(
        json.dumps({"paid_ready": False}), encoding="utf-8"
    )
    assert launch.main(["--prepare-pin"]) == 2
    assert not existing.exists()


def test_overlay_allows_absent_pin_only_before_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(launch, "EXP", tmp_path)
    pin = {
        "capsule": {"package_dir": "experiments/psem_r2_policy"},
        "canonical": {
            "harness_files": [],
            "config_files": ["HOLD_OUT_GATE.json", "RUNTIME_PIN.json", "PIN_MANIFEST.json"],
            "runtime_overrides": {},
            "tests": {},
        },
    }
    (tmp_path / "HOLD_OUT_GATE.json").write_text(
        json.dumps({"frozen": False, "pin_required": True}), encoding="utf-8"
    )

    plan = launch._overlay_plan(pin)
    assert [source.name for _, source in plan] == ["HOLD_OUT_GATE.json", "RUNTIME_PIN.json"]

    (tmp_path / "HOLD_OUT_GATE.json").write_text(
        json.dumps({"frozen": True, "pin_required": True}), encoding="utf-8"
    )
    with pytest.raises(launch.LaunchError, match="PIN_MANIFEST.json is required"):
        launch._overlay_plan(pin)
    preparation_plan = launch._overlay_plan(pin, allow_missing_pin=True)
    assert [source.name for _, source in preparation_plan] == [
        "HOLD_OUT_GATE.json",
        "RUNTIME_PIN.json",
    ]


def test_capsule_state_checks_carried_archive_and_actual_overlay_bytes(tmp_path: Path) -> None:
    pin = {
        "runtime_archive": {"file": "runtime.tar.gz", "sha256": "archive-sha", "commit": "abc"},
        "prompt": {"file": "prompts/prompt.md", "file_sha256": "p", "stripped_sha256": "s"},
        "entry_module": "experiments.psem_r2_policy.run",
        "expected_tests": 1,
    }
    overlay_path = tmp_path / "runtime.py"
    overlay_path.write_text("runtime = 1\n", encoding="utf-8")
    prompt_path = tmp_path / "prompts" / "prompt.md"
    prompt_path.parent.mkdir()
    prompt_path.write_text("prompt\n", encoding="utf-8")
    overlay = [
        {
            "capsule": "runtime.py",
            "source": "canonical.py",
            "sha256": launch._sha256_file(overlay_path),
        }
    ]
    manifest = {
        "revision": launch.REVISION,
        "fingerprint": "fingerprint",
        "runtime_archive": {"file": "runtime.tar.gz", "sha256": "archive-sha", "commit": "abc"},
        "prompt": pin["prompt"],
        "entry_module": pin["entry_module"],
        "expected_tests": 1,
        "overlay": overlay,
        "generated": [],
    }
    manifest_path = tmp_path / "capsule_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    stored, reason = launch._capsule_state(
        manifest_path, "fingerprint", pin, "archive-sha", overlay, [], tmp_path
    )
    assert stored == manifest
    assert reason == "ok"

    overlay_path.write_text("runtime = 2\n", encoding="utf-8")
    stored, reason = launch._capsule_state(
        manifest_path, "fingerprint", pin, "archive-sha", overlay, [], tmp_path
    )
    assert stored is None
    assert reason == "capsule file hash mismatch: runtime.py"
