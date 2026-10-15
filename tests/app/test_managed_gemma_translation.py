from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import httpx
import pytest

from puripuly_heart.app.ports.managed_gemma_translation import (
    ManagedGemmaTranslationSelection,
)
from puripuly_heart.app.services.managed_gemma_translation import (
    ManagedGemmaTranslationOwner,
)
from puripuly_heart.app.wiring.wiring_managed_gemma import managed_gemma_selection
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.config.translation_values import TranslationConnection, TranslationModel
from puripuly_heart.core.local_asr.local_stt_download_port import (
    HuggingFaceDownloadProgress,
    LocalSTTDownloadPortError,
)
from puripuly_heart.core.local_translation import assets, provisioning
from puripuly_heart.core.local_translation.provisioning import (
    GemmaProvisioningCancelled,
    GemmaProvisioningUpdate,
)
from puripuly_heart.core.local_translation.runtime import ManagedGemmaReadiness


def _selection(backend: str = "cpu") -> ManagedGemmaTranslationSelection:
    return ManagedGemmaTranslationSelection(
        backend=backend,
        source_language="ko",
        target_language="en",
        system_prompt="translate {source_language} to {target_language}",
        locale="en",
    )


def _readiness(backend: str) -> ManagedGemmaReadiness:
    effective = "vulkan" if backend == "gpu" else "cpu"
    return ManagedGemmaReadiness(
        requested_backend=backend,
        effective_backend=effective,
        source_language="ko",
        target_language="en",
        prefix_identity=f"prefix-{backend}",
    )


class RecordingRuntime:
    def __init__(self, prepare: Callable[..., Any]) -> None:
        self._prepare = prepare
        self.calls: list[dict[str, object]] = []
        self.release_count = 0
        self.close_count = 0

    async def prepare(self, **kwargs: object) -> ManagedGemmaReadiness:
        self.calls.append(kwargs)
        result = self._prepare(**kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result

    async def release(self) -> None:
        self.release_count += 1

    async def close(self) -> None:
        self.close_count += 1


@pytest.mark.asyncio
async def test_prepare_reports_progress_and_gates_activation_on_runtime_readiness() -> None:
    ready = asyncio.Event()
    progress_emitted = asyncio.Event()
    statuses = []

    async def prepare(**kwargs: object) -> ManagedGemmaReadiness:
        callback = kwargs["provision_kwargs"]["on_status"]
        callback(GemmaProvisioningUpdate("downloading", 25, 100, 25))
        progress_emitted.set()
        await ready.wait()
        callback(GemmaProvisioningUpdate("ready", 100, 100, 100))
        return _readiness("gpu")

    runtime = RecordingRuntime(prepare)
    owner = ManagedGemmaTranslationOwner(runtime=runtime, status_sink=statuses.append)

    task = asyncio.create_task(owner.prepare(_selection("gpu")))
    await progress_emitted.wait()

    assert task.done() is False
    assert [snapshot.state for snapshot in statuses] == ["checking", "downloading"]
    assert statuses[-1].progress_percent == 25

    ready.set()
    activation = await task

    assert activation.backend == "gpu"
    assert activation.readiness.effective_backend == "vulkan"
    assert [snapshot.state for snapshot in statuses] == [
        "checking",
        "downloading",
        "preparing",
        "ready",
    ]


@pytest.mark.asyncio
async def test_cancel_interrupts_prepare_and_exposes_retryable_cancelled_state() -> None:
    started = asyncio.Event()

    async def prepare(**_kwargs: object) -> ManagedGemmaReadiness:
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    runtime = RecordingRuntime(prepare)
    owner = ManagedGemmaTranslationOwner(runtime=runtime)
    task = asyncio.create_task(owner.prepare(_selection()))
    await started.wait()

    assert owner.cancel() is True
    with pytest.raises(GemmaProvisioningCancelled):
        await task

    assert owner.snapshot.state == "cancelled"
    assert owner.cancel() is False


@pytest.mark.asyncio
async def test_failed_prepare_can_be_retried_and_activated() -> None:
    attempts = 0

    def prepare(**kwargs: object) -> ManagedGemmaReadiness:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("download failed")
        return _readiness(str(kwargs["backend"]))

    runtime = RecordingRuntime(prepare)
    owner = ManagedGemmaTranslationOwner(runtime=runtime)

    with pytest.raises(RuntimeError, match="download failed"):
        await owner.prepare(_selection())
    assert owner.snapshot.state == "failed"

    activation = await owner.prepare(_selection())

    assert activation.backend == "cpu"
    assert owner.snapshot.state == "ready"
    assert attempts == 2


@pytest.mark.asyncio
async def test_failed_prepare_exposes_bounded_download_diagnostics() -> None:
    def prepare(**_kwargs: object) -> ManagedGemmaReadiness:
        download_failure = LocalSTTDownloadPortError(
            "sensitive worker detail",
            failure_code="download_failed",
            cause_type="RuntimeError",
            worker_exit_code=1,
            status_code=503,
        )
        raise RuntimeError("provisioning wrapper") from download_failure

    owner = ManagedGemmaTranslationOwner(runtime=RecordingRuntime(prepare))

    with pytest.raises(RuntimeError, match="provisioning wrapper"):
        await owner.prepare(_selection())

    assert owner.snapshot.failure_phase == "download"
    assert owner.snapshot.failure_code == "download_failed"
    assert owner.snapshot.cause_type == "RuntimeError"
    assert owner.snapshot.worker_exit_code == 1
    assert owner.snapshot.status_code == 503
    assert "sensitive worker detail" not in repr(owner.snapshot)


@pytest.mark.asyncio
async def test_combined_cancellation_and_cleanup_failure_is_terminal_and_bounded() -> None:
    def prepare(**_kwargs: object) -> ManagedGemmaReadiness:
        raise BaseExceptionGroup(
            "combined failure",
            [
                asyncio.CancelledError(),
                PermissionError(5, "sensitive staging path"),
            ],
        )

    statuses = []
    owner = ManagedGemmaTranslationOwner(
        runtime=RecordingRuntime(prepare),
        status_sink=statuses.append,
    )

    with pytest.raises(BaseExceptionGroup) as caught:
        await owner.prepare(_selection())

    assert any(
        isinstance(exception, asyncio.CancelledError) for exception in caught.value.exceptions
    )
    assert any(isinstance(exception, PermissionError) for exception in caught.value.exceptions)
    assert owner.snapshot.state == "failed"
    assert owner.snapshot.failure_phase == "provisioning"
    assert owner.snapshot.failure_code == "filesystem_failed"
    assert owner.snapshot.cause_type == "PermissionError"
    assert all(snapshot.state != "ready" for snapshot in statuses)
    assert "sensitive staging path" not in repr(owner.snapshot)


@pytest.mark.asyncio
async def test_failed_prepare_diagnostic_does_not_prevent_terminal_runtime_close() -> None:
    diagnostics = []

    def prepare(**_kwargs: object) -> ManagedGemmaReadiness:
        raise RuntimeError("download failed")

    runtime = RecordingRuntime(prepare)
    owner = ManagedGemmaTranslationOwner(
        runtime=runtime,
        lifecycle_diagnostic_sink=diagnostics.append,
    )

    with pytest.raises(RuntimeError, match="download failed"):
        await owner.prepare(_selection())

    await owner.close()

    assert runtime.close_count == 1
    assert len(diagnostics) == 1
    assert diagnostics[0].fields["exception_class"] == "RuntimeError"


@pytest.mark.asyncio
async def test_lifecycle_diagnostic_sink_failure_still_closes_runtime() -> None:
    def prepare(**_kwargs: object) -> ManagedGemmaReadiness:
        raise RuntimeError("download failed")

    def fail_diagnostic(_event: object) -> None:
        raise RuntimeError("diagnostic sink failed")

    runtime = RecordingRuntime(prepare)
    owner = ManagedGemmaTranslationOwner(
        runtime=runtime,
        lifecycle_diagnostic_sink=fail_diagnostic,
    )

    with pytest.raises(RuntimeError, match="download failed"):
        await owner.prepare(_selection())
    with pytest.raises(RuntimeError, match="diagnostic sink failed"):
        await owner.close()

    assert runtime.close_count == 1


@pytest.mark.asyncio
async def test_retired_activation_cannot_release_newer_shared_runtime() -> None:
    runtime = RecordingRuntime(lambda **kwargs: _readiness(str(kwargs["backend"])))
    owner = ManagedGemmaTranslationOwner(runtime=runtime)

    first = await owner.prepare(_selection("cpu"))
    second = await owner.prepare(_selection("gpu"))

    await first.release()
    assert runtime.release_count == 0

    await second.release()
    assert runtime.release_count == 1
    assert owner.snapshot.state == "idle"


@pytest.mark.asyncio
async def test_linger_deactivate_keeps_runtime_until_timer_elapses() -> None:
    runtime = RecordingRuntime(lambda **kwargs: _readiness(str(kwargs["backend"])))
    owner = ManagedGemmaTranslationOwner(runtime=runtime, idle_linger_s=0.05)
    await owner.prepare(_selection())

    await owner.deactivate(linger=True)

    assert runtime.release_count == 0
    assert owner.snapshot.state == "idle"
    await asyncio.sleep(0.12)
    assert runtime.release_count == 1


@pytest.mark.asyncio
async def test_prepare_during_linger_cancels_shutdown_and_reuses_runtime() -> None:
    runtime = RecordingRuntime(lambda **kwargs: _readiness(str(kwargs["backend"])))
    owner = ManagedGemmaTranslationOwner(runtime=runtime, idle_linger_s=0.2)
    await owner.prepare(_selection())
    await owner.deactivate(linger=True)

    await owner.prepare(_selection())

    assert runtime.release_count == 0
    assert owner.snapshot.state == "ready"
    await asyncio.sleep(0.25)
    assert runtime.release_count == 0


@pytest.mark.asyncio
async def test_immediate_deactivate_and_close_skip_linger() -> None:
    runtime = RecordingRuntime(lambda **kwargs: _readiness(str(kwargs["backend"])))
    owner = ManagedGemmaTranslationOwner(runtime=runtime, idle_linger_s=1.0)
    await owner.prepare(_selection())

    await owner.deactivate()
    assert runtime.release_count == 1

    await owner.prepare(_selection())
    await owner.deactivate(linger=True)
    await owner.close()

    assert runtime.release_count == 1
    assert runtime.close_count == 1
    assert owner.snapshot.state == "closed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("locale", "source_language", "expected_source"),
    [
        ("zh-CN", "ko", "modelscope"),
        ("en", "zh-CN", "huggingface"),
    ],
)
async def test_preparation_download_source_comes_from_ui_locale_not_translation_language(
    tmp_path, monkeypatch, locale, source_language, expected_source
) -> None:
    contents = {
        assets.GEMMA_MODEL_FILENAME: b"target-model",
        assets.GEMMA_DRAFT_FILENAME: b"draft-model",
    }
    monkeypatch.setattr(
        assets,
        "GEMMA_ASSETS",
        tuple(
            assets.GemmaAsset(name, len(content), hashlib.sha256(content).hexdigest())
            for name, content in contents.items()
        ),
    )
    settings = AppSettingsVNext()
    settings = replace(
        settings,
        intent=replace(
            settings.intent,
            ui=replace(settings.intent.ui, locale=locale),
            languages=replace(
                settings.intent.languages,
                source_language=source_language,
                target_language="en",
            ),
            translation=replace(
                settings.intent.translation,
                model=TranslationModel.MANAGED_GEMMA.value,
                connection=TranslationConnection.CPU.value,
            ),
        ),
    )
    selected_sources = []

    class Downloader:
        async def download(self, request, *, cancel_event, on_progress):
            if expected_source != "huggingface":
                raise AssertionError("Chinese UI must select ModelScope")
            selected_sources.append("huggingface")
            destination = request.local_dir / request.remote_path
            destination.write_bytes(contents[request.remote_path])
            return destination

    def serve(request):
        if expected_source != "modelscope":
            raise AssertionError("Chinese translation language must not select ModelScope")
        selected_sources.append("modelscope")
        return httpx.Response(200, content=contents[request.url.params["FilePath"]])

    monkeypatch.setattr(
        provisioning.network_clients,
        "external_client",
        lambda **kwargs: httpx.Client(transport=httpx.MockTransport(serve), **kwargs),
    )
    install_dir = tmp_path / "gemma"

    async def prepare(**kwargs):
        await provisioning.ensure_gemma_installed(
            install_dir=install_dir,
            **kwargs["provision_kwargs"],
        )
        return _readiness(str(kwargs["backend"]))

    owner = ManagedGemmaTranslationOwner(
        runtime=RecordingRuntime(prepare),
        downloader=Downloader(),
    )
    try:
        activation = await owner.prepare(managed_gemma_selection(settings))

        assert activation.backend == "cpu"
        assert owner.snapshot.state == "ready"
        assert {name: (install_dir / name).read_bytes() for name in contents} == contents
        assert selected_sources == [expected_source] * len(contents)
        assert assets.validate_gemma_install(install_dir).revision == assets.GEMMA_REVISION
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_modelscope_http_failure_preserves_download_status_metadata(
    tmp_path, monkeypatch
) -> None:
    class FailedDownloader:
        async def download(self, request, *, cancel_event, on_progress):
            raise LocalSTTDownloadPortError("primary download unavailable")

    monkeypatch.setattr(
        provisioning.network_clients,
        "external_client",
        lambda **kwargs: httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(503)),
            **kwargs,
        ),
    )

    async def prepare(**kwargs):
        await provisioning.ensure_gemma_installed(
            install_dir=tmp_path / "gemma",
            **kwargs["provision_kwargs"],
        )

    owner = ManagedGemmaTranslationOwner(
        runtime=RecordingRuntime(prepare),
        downloader=FailedDownloader(),
    )
    try:
        with pytest.raises(provisioning.GemmaProvisioningError):
            await owner.prepare(_selection())

        assert owner.snapshot.state == "failed"
        assert owner.snapshot.failure_phase == "download"
        assert owner.snapshot.failure_code == "download_failed"
        assert owner.snapshot.cause_type == "HTTPStatusError"
        assert owner.snapshot.status_code == 503
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_26b_selection_installs_only_its_verified_target_and_reuses_it(
    tmp_path, monkeypatch
) -> None:
    contents = b"synthetic-26b-target"
    real_spec = assets.resolve_gemma_spec(assets.GEMMA_26B_MODEL_ID)
    spec = replace(
        real_spec,
        assets=(
            assets.GemmaAsset(
                real_spec.model_filename,
                len(contents),
                hashlib.sha256(contents).hexdigest(),
            ),
        ),
    )
    monkeypatch.setattr(assets, "gemma_26b_spec", lambda: spec)
    downloads = []

    class Downloader:
        async def download(self, request, *, cancel_event, on_progress):
            downloads.append(request.remote_path)
            destination = request.local_dir / request.remote_path
            destination.write_bytes(contents)
            on_progress(HuggingFaceDownloadProgress(len(contents), len(contents)))
            return destination

    settings = AppSettingsVNext()
    settings = replace(
        settings,
        intent=replace(
            settings.intent,
            translation=replace(
                settings.intent.translation, model="managed_gemma_26b", connection="gpu"
            ),
        ),
    )
    selection = managed_gemma_selection(settings)
    install_dir = tmp_path / spec.install_dirname

    async def prepare(**kwargs):
        await provisioning.ensure_gemma_installed(
            install_dir=install_dir,
            **kwargs["provision_kwargs"],
        )
        return _readiness(str(kwargs["backend"]))

    statuses = []
    owner = ManagedGemmaTranslationOwner(
        runtime=RecordingRuntime(prepare),
        downloader=Downloader(),
        status_sink=statuses.append,
    )
    try:
        activation = await owner.prepare(selection)
        assert activation.readiness.effective_backend == "vulkan"
        assert owner.snapshot.model_id == assets.GEMMA_26B_MODEL_ID
        assert install_dir.joinpath(spec.model_filename).read_bytes() == contents
        assert assets.validate_gemma_install(install_dir, spec=spec).model_id == spec.model_id
        assert not tmp_path.joinpath(assets.GEMMA_MODEL_ID).exists()
        assert downloads == [spec.model_filename]
        assert any(snapshot.state == "downloading" for snapshot in statuses)
        await owner.prepare(selection)
        assert downloads == [spec.model_filename]
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_old_model_release_cannot_overtake_new_model_preparation() -> None:
    release_started = asyncio.Event()
    release_finish = asyncio.Event()

    class Runtime(RecordingRuntime):
        async def release(self):
            release_started.set()
            await release_finish.wait()
            await super().release()

    runtime = Runtime(lambda **kwargs: _readiness(str(kwargs["backend"])))
    owner = ManagedGemmaTranslationOwner(runtime=runtime)
    first = await owner.prepare(_selection("cpu"))
    releasing = asyncio.create_task(first.release())
    await release_started.wait()
    second = asyncio.create_task(
        owner.prepare(replace(_selection("gpu"), model_id=assets.GEMMA_26B_MODEL_ID))
    )
    await asyncio.sleep(0)
    assert len(runtime.calls) == 1
    release_finish.set()
    await releasing
    await second
    assert runtime.calls[-1]["spec"].model_id == assets.GEMMA_26B_MODEL_ID
    assert owner.snapshot.state == "ready"
    assert owner.snapshot.model_id == assets.GEMMA_26B_MODEL_ID
    await first.release()
    assert runtime.release_count == 1
    await owner.close()


def test_26b_application_selection_rejects_cpu_without_substitution() -> None:
    with pytest.raises(ValueError, match="requires the GPU backend"):
        replace(_selection("cpu"), model_id=assets.GEMMA_26B_MODEL_ID)
