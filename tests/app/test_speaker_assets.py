from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from puripuly_heart.app.adapters import nemotron
from puripuly_heart.app.adapters.nemotron import assets as speaker_assets
from puripuly_heart.app.adapters.nemotron.assets import NativeSpeakerAssets


@pytest.mark.asyncio
async def test_local_install_promotes_verified_artifact_and_manifest_independently(
    tmp_path, monkeypatch
):
    destination = tmp_path / "speaker" / "model.gguf"
    destination.parent.mkdir()
    destination.write_bytes(b"previous")
    (destination.parent / "model.manifest.json").write_text("old", encoding="utf-8")
    calls = []

    async def acquire(path, *, download_consent, source_path, cancel_event):
        calls.append((path, download_consent, source_path))
        assert path != destination
        path.parent.mkdir()
        path.write_bytes(b"verified-F16")
        (path.parent / "model.manifest.json").write_text(
            json.dumps(
                {
                    "sha256": "test-digest",
                    "publisher": {"original_checkpoint_lineage": "not_independently_reproduced"},
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(nemotron, "acquire_artifact", acquire)
    monkeypatch.setattr(
        nemotron,
        "inspect_artifact",
        lambda path: {"size_bytes": path.stat().st_size, "sha256": "test-digest"},
    )
    monkeypatch.setattr(nemotron, "validate_runtime", lambda path: {"ready": True})
    monkeypatch.setattr(
        nemotron,
        "validate_artifact_provenance",
        lambda path, metadata=None: json.loads(
            path.with_name("model.manifest.json").read_text(encoding="utf-8")
        ),
    )
    assets = NativeSpeakerAssets(artifact_path=destination, runtime_dir=tmp_path / "runtime")
    result = await assets.install(download_consent=False, source_path=Path("explicit-local.gguf"))
    assert result.ready
    assert destination.read_bytes() == b"verified-F16"
    provenance = json.loads(
        (destination.parent / "model.manifest.json").read_text(encoding="utf-8")
    )
    assert provenance["sha256"] == "test-digest"
    assert (
        result.metadata["provenance"]["publisher"]["original_checkpoint_lineage"]
        == "not_independently_reproduced"
    )
    assert calls[0][1:] == (False, Path("explicit-local.gguf"))
    assert not list(tmp_path.glob("speaker.staging-*"))
    assert not (tmp_path / "speaker.backup").exists()
    await assets.close()


@pytest.mark.asyncio
async def test_cancel_preserves_previous_install_and_waits_for_conversion_cleanup(
    tmp_path, monkeypatch
):
    destination = tmp_path / "speaker" / "model.gguf"
    destination.parent.mkdir()
    destination.write_bytes(b"previous")
    started = asyncio.Event()

    async def acquire(path, *, download_consent, source_path, cancel_event):
        path.parent.mkdir()
        path.write_bytes(b"incomplete")
        started.set()
        while not cancel_event.is_set():
            await asyncio.sleep(0)
        raise InterruptedError("cancelled conversion")

    monkeypatch.setattr(nemotron, "acquire_artifact", acquire)
    assets = NativeSpeakerAssets(artifact_path=destination, runtime_dir=tmp_path / "runtime")
    install = asyncio.create_task(
        assets.install(download_consent=False, source_path=Path("local.gguf"))
    )
    await started.wait()
    await assets.cancel()
    with pytest.raises(InterruptedError):
        await install
    assert destination.read_bytes() == b"previous"
    assert not list(tmp_path.glob("speaker.staging-*"))
    await assets.close()


@pytest.mark.asyncio
async def test_download_requires_independent_consent_before_owned_work(tmp_path, monkeypatch):
    async def unexpected(**kwargs):
        pytest.fail("download started without independent speaker consent")

    monkeypatch.setattr(speaker_assets, "ensure_local_stt_installed", unexpected)
    assets = NativeSpeakerAssets(
        artifact_path=tmp_path / "speaker" / "model.gguf", runtime_dir=tmp_path / "runtime"
    )
    with pytest.raises(ValueError, match="consent"):
        await assets.install(download_consent=False)
    assert assets.phase == "idle"
    assert assets._install_task is None
    await assets.close()


@pytest.mark.asyncio
async def test_consented_download_reuses_shared_installer_and_pinned_native_manifest(
    tmp_path, monkeypatch
):
    destination = tmp_path / "speaker" / "model.gguf"
    calls = []

    async def download(*, model_root, manifest, preferred_source, cancel_event):
        calls.append((model_root, manifest, preferred_source))
        assert (
            manifest.sources["huggingface"].revision == "25cf0d6b0c91b679376df0f6ef3087b4fd96af1f"
        )
        assert (
            manifest.files[0].sha256
            == "9b277e1dcec60b2c771a1dcb45dc8e36f7285ff7c1619d880d3926a373036b38"
        )
        assert manifest.files[0].size_bytes == 201832928
        staged = model_root / manifest.install_dirname
        staged.mkdir(parents=True)
        (staged / "model.gguf").write_bytes(b"qualified-fixture")
        (staged / "installed-manifest.json").write_text("{}", encoding="utf-8")

    def adopt(path):
        assert path.read_bytes() == b"qualified-fixture"
        path.with_name("model.manifest.json").write_text("{}", encoding="utf-8")
        return {}

    monkeypatch.setattr(speaker_assets, "ensure_local_stt_installed", download)
    monkeypatch.setattr(nemotron, "adopt_preconverted_artifact", adopt)
    assets = NativeSpeakerAssets(artifact_path=destination, runtime_dir=tmp_path / "runtime")
    await assets.install(download_consent=True)
    assert destination.read_bytes() == b"qualified-fixture"
    assert (destination.parent / "installed-manifest.json").is_file()
    assert calls[0][2] == "huggingface"
    assert not list(tmp_path.glob("speaker.download-*"))
    await assets.install(download_consent=True)
    assert len(calls) == 1
    await assets.close()


@pytest.mark.asyncio
async def test_incompatible_download_keeps_previous_install(tmp_path, monkeypatch):
    destination = tmp_path / "speaker" / "model.gguf"
    destination.parent.mkdir()
    destination.write_bytes(b"previous")

    async def download(*, model_root, manifest, preferred_source, cancel_event):
        staged = model_root / manifest.install_dirname
        staged.mkdir(parents=True)
        (staged / "model.gguf").write_bytes(b"incompatible")

    def reject(path):
        raise ValueError("not the qualified pinned F16 artifact")

    monkeypatch.setattr(speaker_assets, "ensure_local_stt_installed", download)
    monkeypatch.setattr(nemotron, "adopt_preconverted_artifact", reject)
    assets = NativeSpeakerAssets(artifact_path=destination, runtime_dir=tmp_path / "runtime")
    with pytest.raises(ValueError, match="qualified"):
        await assets.install(download_consent=True)
    assert destination.read_bytes() == b"previous"
    assert not list(tmp_path.glob("speaker.download-*"))
    assert assets.phase == "idle"
    await assets.close()


@pytest.mark.asyncio
async def test_unverified_sidecar_cannot_create_product_readiness(tmp_path, monkeypatch):
    artifact = tmp_path / "model.gguf"
    artifact.write_bytes(b"probe")
    artifact.with_name("model.manifest.json").write_text(
        json.dumps(
            {"sha256": "probe", "conversion": {"original_checkpoint_lineage": "unverified"}}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(nemotron, "inspect_artifact", lambda path: {"sha256": "probe"})
    monkeypatch.setattr(
        nemotron,
        "validate_runtime",
        lambda path: pytest.fail("runtime inspected for rejected provenance"),
    )
    assets = NativeSpeakerAssets(artifact_path=artifact, runtime_dir=tmp_path / "runtime")
    result = await assets.inspect()
    assert not result.ready
    assert result.reason == "ValueError"
    await assets.close()


@pytest.mark.asyncio
async def test_cancel_download_waits_for_shared_installer_and_cleans_staging(tmp_path, monkeypatch):
    started = asyncio.Event()

    async def download(*, model_root, manifest, preferred_source, cancel_event):
        staged = model_root / manifest.install_dirname
        staged.mkdir(parents=True)
        (staged / "model.gguf").write_bytes(b"partial")
        started.set()
        while not cancel_event.is_set():
            await asyncio.sleep(0)
        raise InterruptedError("cancelled download")

    monkeypatch.setattr(speaker_assets, "ensure_local_stt_installed", download)
    assets = NativeSpeakerAssets(
        artifact_path=tmp_path / "speaker" / "model.gguf", runtime_dir=tmp_path / "runtime"
    )
    install = asyncio.create_task(assets.install(download_consent=True))
    await started.wait()
    await assets.cancel()
    with pytest.raises(InterruptedError):
        await install
    assert not list(tmp_path.glob("speaker.download-*"))
    assert assets.phase == "idle"
    await assets.close()
