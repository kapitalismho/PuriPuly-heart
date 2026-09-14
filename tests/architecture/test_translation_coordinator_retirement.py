from __future__ import annotations

import re

from tests.helpers.paths import REPO_ROOT, SOURCE_ROOT

TEST_ROOT = REPO_ROOT / "tests"


def test_retired_coordinator_and_dynamic_fixture_surfaces_are_absent() -> None:
    assert not (SOURCE_ROOT / "core" / "orchestrator" / "hub.py").exists()
    assert not (SOURCE_ROOT / "core" / "orchestrator" / "hub_callbacks.py").exists()
    assert not (TEST_ROOT / "helpers" / "client_hub.py").exists()

    helper_source = (TEST_ROOT / "helpers" / "translation_owners.py").read_text(encoding="utf-8")
    assert "ClientHubTestHarness" not in helper_source
    assert "compose_client_hub" not in helper_source
    assert "def __getattr__(" not in helper_source
    assert "def __setattr__(" not in helper_source
    assert "TranslationOwnersTestHarness" in helper_source
    assert "compose_translation_test_harness" in helper_source


def test_production_has_no_retired_coordinator_residue() -> None:
    residue = re.compile(r"\bClientHub\b|\bHub[A-Z]|\bhub\b|_HUB_")
    violations: list[str] = []
    for path in SOURCE_ROOT.rglob("*.py"):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            lowered = line.lower()
            if not residue.search(line):
                continue
            if "github" in lowered or "huggingface" in lowered or "hf_hub_" in lowered:
                continue
            violations.append(f"{path.relative_to(REPO_ROOT)}:{line_number}:{line.strip()}")
    assert violations == []


def test_windows_process_evidence_uses_current_peer_capture_owner_contract() -> None:
    source = (SOURCE_ROOT / "release_evidence" / "windows_process_isolation.py").read_text(
        encoding="utf-8"
    )
    constructor = source[source.index("runtime = PeerCaptureSessionOwner(") :]
    assert "admission=Admission()" in constructor
    assert "target_resolver=TargetResolver()" in constructor
    assert "provider=Provider(events)" in constructor
    assert "provider_request_factory=" in constructor
    assert "vad_sink=VadSink()" in constructor
    assert "hub=" not in constructor
    assert "PeerChannelRuntime(" not in source
