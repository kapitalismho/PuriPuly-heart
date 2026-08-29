from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.psem_sortformer_adaptation_depth import authority_registry
from experiments.psem_sortformer_adaptation_depth.authority_registry import (
    register_execution,
    require_registered_execution,
)
from experiments.psem_sortformer_adaptation_depth.protocol import bind_payload


def test_authority_execution_registry_is_content_addressed_and_fail_closed(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "registry"
    monkeypatch.setattr(authority_registry, "authority_registry_root", lambda: root)
    payload = bind_payload(
        {
            "schema_version": 1,
            "artifact_role": "test_runtime_receipt",
            "value": 1,
        }
    )
    descriptor = register_execution("test-runtime", payload)
    assert require_registered_execution("test-runtime", payload)["payload"] == payload
    path = descriptor["authority_registry_record"]
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    record["payload"]["value"] = 2
    Path(path).write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(Exception, match="registered execution payload differs"):
        require_registered_execution("test-runtime", payload)
