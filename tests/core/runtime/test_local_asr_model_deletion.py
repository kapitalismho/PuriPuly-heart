from __future__ import annotations

import asyncio

import pytest
from puripuly_heart.core.local_asr_provisioning import LocalASRInstallRequest
from puripuly_heart.core.local_stt_assets import (
    LOCAL_STT_MODEL_ID,
    load_local_stt_asset_manifest,
)
from puripuly_heart.core.local_stt_runtime_installer import LocalSTTRuntimeInstallError

from tests.core.runtime.test_local_asr_provisioning import (
    MutableProvisioningBackend,
    _all_states,
    _owner,
    _state,
)


@pytest.mark.asyncio
async def test_delete_removes_the_install_and_republishes_integrity(tmp_path) -> None:
    backend = MutableProvisioningBackend(_all_states())
    owner = _owner(backend, model_root=tmp_path)
    install = tmp_path / load_local_stt_asset_manifest(LOCAL_STT_MODEL_ID).install_dirname
    install.mkdir()
    (install / "model.onnx").write_bytes(b"weights")
    try:
        await owner.inspect_cpu((LOCAL_STT_MODEL_ID,))
        assert owner.snapshot.state_for(LOCAL_STT_MODEL_ID).integrity == "ready"
        backend.states[LOCAL_STT_MODEL_ID] = _state("missing", LOCAL_STT_MODEL_ID)

        snapshot = await owner.delete_model(LOCAL_STT_MODEL_ID)

        assert not install.exists()
        assert snapshot.state_for(LOCAL_STT_MODEL_ID).integrity == "missing"
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_delete_is_refused_while_the_backend_is_installing(tmp_path) -> None:
    backend = MutableProvisioningBackend(_all_states("missing"), release_install=asyncio.Event())
    owner = _owner(backend, model_root=tmp_path)
    install = tmp_path / load_local_stt_asset_manifest(LOCAL_STT_MODEL_ID).install_dirname
    install.mkdir()
    try:
        task = owner.start_install(
            LocalASRInstallRequest("cpu", (LOCAL_STT_MODEL_ID,), None, "activation")
        )
        await backend.install_started.wait()

        with pytest.raises(LocalSTTRuntimeInstallError) as caught:
            await owner.delete_model(LOCAL_STT_MODEL_ID)

        assert caught.value.failure_code == "task_busy"
        assert install.exists()
        backend.release_install.set()
        await task
    finally:
        await owner.close()
