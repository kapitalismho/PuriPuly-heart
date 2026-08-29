import importlib.metadata
import json
import platform
from types import SimpleNamespace

import pytest
from torch import nn

from experiments.psem_sortformer_adaptation_depth import nemo_adapter
from experiments.psem_sortformer_adaptation_depth.nemo_adapter import (
    NEMO_REVISION,
    PINNED_CONTAINER_IMAGE_IDENTITY,
    REQUIRED_LOCK_PACKAGES,
    TrainableSortformerPSEM,
    _temporary_causal_attention,
    _validate_state_reset_lifecycle,
    validate_dependency_lock,
)


def _lock() -> dict:
    return {
        "schema_version": 1,
        "artifact_role": "nemo_dependency_lock",
        "nemo_revision": NEMO_REVISION,
        "python_version": platform.python_version(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "container_image_identity": "sha256:" + "a" * 64,
        "accelerator": {
            "cuda_device_count": 1,
            "device_name": "test accelerator",
            "device_total_memory_bytes": 80000000000,
            "nvidia_driver_version": "test-driver",
            "torch_cuda_version": "test-cuda",
        },
        "lock_kind": "complete_installed_distribution_inventory",
        "packages": [
            {"name": name, "version": importlib.metadata.version(name)}
            for name in sorted(REQUIRED_LOCK_PACKAGES)
        ],
    }


def test_dependency_lock_requires_exact_installed_versions_and_complete_package_set(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "1.0")
    monkeypatch.setattr(
        nemo_adapter,
        "_installed_dependency_inventory",
        lambda: _lock()["packages"],
    )
    monkeypatch.setattr(
        nemo_adapter,
        "_container_image_identity",
        lambda: _lock()["container_image_identity"],
    )
    monkeypatch.setattr(
        nemo_adapter,
        "_accelerator_identity",
        lambda: _lock()["accelerator"],
    )
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(_lock()), encoding="utf-8")
    receipt = validate_dependency_lock(path)
    assert receipt["sha256"]
    value = _lock()
    value["packages"][0]["version"] = "0.invalid"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(Exception, match="differs from lock"):
        validate_dependency_lock(path)
    value = _lock()
    value["packages"] = value["packages"][1:]
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(Exception, match="incomplete"):
        validate_dependency_lock(path)


def test_container_identity_requires_the_pinned_ngc_manifest_digest(monkeypatch) -> None:
    monkeypatch.setenv("PSEM_CONTAINER_IMAGE_IDENTITY", PINNED_CONTAINER_IMAGE_IDENTITY)
    assert nemo_adapter._container_image_identity() == PINNED_CONTAINER_IMAGE_IDENTITY
    monkeypatch.setenv("PSEM_CONTAINER_IMAGE_IDENTITY", "sha256:" + "a" * 64)
    with pytest.raises(Exception, match="differs from the pinned NVIDIA PyTorch image digest"):
        nemo_adapter._container_image_identity()


def test_psem_head_moves_with_the_sortformer_wrapper() -> None:
    wrapped = TrainableSortformerPSEM(nn.Linear(1, 1), lambda *args, **kwargs: None).to("meta")
    assert next(wrapped.sortformer.parameters()).device.type == "meta"
    assert next(wrapped.psem_head.parameters()).device.type == "meta"


def test_random_causal_attention_restores_the_exact_prior_context() -> None:
    model = SimpleNamespace(
        encoder=SimpleNamespace(att_context_size=[3, 5]),
        transformer_encoder=SimpleNamespace(diag={"prior": 7}),
        sortformer_modules=SimpleNamespace(causal_attn_rc=11),
    )
    with _temporary_causal_attention(model, True):
        assert model.encoder.att_context_size == [-1, 11]
        assert model.transformer_encoder.diag == 11
    assert model.encoder.att_context_size == [3, 5]
    assert model.transformer_encoder.diag == {"prior": 7}


def test_state_reset_evidence_matches_actual_sequence_initialization() -> None:
    import torch

    _validate_state_reset_lifecycle(torch.tensor([[[True], [False]]]), batch_size=1, frame_count=2)
    with pytest.raises(Exception, match="actual sequence initialization"):
        _validate_state_reset_lifecycle(
            torch.tensor([[[False], [False]]]), batch_size=1, frame_count=2
        )
    with pytest.raises(Exception, match="actual sequence initialization"):
        _validate_state_reset_lifecycle(
            torch.tensor([[[True], [True]]]), batch_size=1, frame_count=2
        )
