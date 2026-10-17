from __future__ import annotations

import asyncio
import json

import pytest

from puripuly_heart.core.local_asr.local_stt_download_port import (
    HuggingFaceDownloadProgress,
    LocalSTTDownloadPortCancelled,
)
from puripuly_heart.core.local_translation import assets, provisioning
from tests.app.test_managed_gemma_provider_runtime_integration import (
    _composed_signature_application,
)
from tests.core.local_translation.test_gemma_provisioning import CoordinatedDownloader
from tests.core.test_model_asset_reliability import _small_gemma


@pytest.mark.parametrize("finish", ["success", "cancel", "shutdown"])
async def test_composed_provider_cutover_and_locale_complete_during_gated_asset_transfer(
    tmp_path, monkeypatch, finish
):
    spec, contents = _small_gemma(monkeypatch, tmp_path)
    started, release, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()
    runtime_released = asyncio.Event()
    cancellation = []

    class Downloader:
        async def download(self, request, *, cancel_event, on_progress):
            cancellation.append(cancel_event)
            started.set()
            while not release.is_set():
                if cancel_event.is_set():
                    cancelled.set()
                    raise LocalSTTDownloadPortCancelled("Explicit asset cancelled")
                try:
                    await asyncio.wait_for(release.wait(), timeout=0.01)
                except TimeoutError:
                    pass
            if cancel_event.is_set():
                cancelled.set()
                raise LocalSTTDownloadPortCancelled("Explicit asset cancelled")
            path = request.local_dir / request.remote_path
            path.parent.mkdir(parents=True, exist_ok=True)
            data = contents[request.remote_path]
            path.write_bytes(data)
            on_progress(HuggingFaceDownloadProgress(len(data), len(data)))
            return path

    async with _composed_signature_application(tmp_path, monkeypatch) as fixture:
        control = fixture.control
        owner = control.gemma()
        monkeypatch.setattr(owner, "_downloader", Downloader())
        actual_release = fixture.native.release

        async def observe_release():
            await actual_release()
            runtime_released.set()

        monkeypatch.setattr(fixture.native, "release", observe_release)
        original_provider = control.pipeline.llm_runtime.provider
        download = await control.submit(
            "models.download",
            {"model_id": spec.model_id, "source": "huggingface"},
            request_id="independent-asset",
        )
        try:
            await asyncio.wait_for(started.wait(), timeout=5)
            switch = await control.submit(
                "settings.apply",
                {"changes": {"translation.model": "local_llm", "translation.connection": "ollama"}},
                request_id="non-gemma-cutover",
            )
            locale = await control.submit(
                "settings.apply", {"changes": {"locale": "ja"}}, request_id="following-locale"
            )
            switched = await control.wait(switch["operation_id"], timeout=5)
            localized = await control.wait(locale["operation_id"], timeout=5)
            assert switched["terminal"] and switched["status"] == "applied"
            assert localized["terminal"] and localized["status"] == "applied"
            assert runtime_released.is_set()
            assert not release.is_set() and not cancelled.is_set()
            assert all(not event.is_set() for event in cancellation)
            assert not (await control.wait(download["operation_id"], timeout=0))["terminal"]
            assert control.pipeline.llm_runtime.provider is not None
            assert control.pipeline.llm_runtime.provider is not original_provider
            current = (await control.query("settings.current", {}))["settings"]
            assert current["intent"]["translation"]["model"] == "local_llm"
            assert current["intent"]["translation"]["connection"] == "ollama"
            assert current["intent"]["ui"]["locale"] == "ja"
            saved = json.loads(fixture.settings_path.read_text(encoding="utf-8"))
            assert saved["intent"]["translation"]["model"] == "local_llm"
            assert saved["intent"]["ui"]["locale"] == "ja"
            install = assets.default_gemma_install_dir(spec)
            if finish == "success":
                release.set()
                downloaded = await control.wait(download["operation_id"], timeout=5)
                assert downloaded["terminal"] and downloaded["status"] == "applied"
                assets.validate_gemma_install(install, spec=spec)
            elif finish == "cancel":
                cancel = await control.submit(
                    "models.cancel", {"model_id": spec.model_id}, request_id="explicit-asset-cancel"
                )
                receipt = await control.wait(cancel["operation_id"], timeout=5)
                assert receipt["terminal"] and receipt["status"] == "cancelled"
                downloaded = await control.wait(download["operation_id"], timeout=5)
                assert downloaded["terminal"] and downloaded["status"] == "cancelled"
                assert cancelled.is_set()
                assert not release.is_set() and not install.exists()
            else:
                await asyncio.wait_for(fixture.app.stop(), timeout=5)
                downloaded = await control.wait(download["operation_id"], timeout=5)
                assert downloaded["terminal"] and downloaded["status"] == "interrupted"
                assert cancelled.is_set()
                assert not release.is_set() and not install.exists()
            assert not tuple(install.parent.glob("*.staging-*"))
            assert fixture.commands == []
        finally:
            release.set()


async def test_composed_prepare_conflict_reports_busy_then_recovers_without_masking_other_errors(
    tmp_path, monkeypatch
):
    spec, contents = _small_gemma(monkeypatch, tmp_path)
    downloader = CoordinatedDownloader(contents)
    async with _composed_signature_application(tmp_path, monkeypatch) as fixture:
        control = fixture.control
        owner = control.gemma()
        monkeypatch.setattr(owner, "_downloader", downloader)
        download = await control.submit(
            "models.download",
            {"model_id": spec.model_id, "source": "huggingface"},
            request_id="asset-before-prepare",
        )
        try:
            await asyncio.wait_for(downloader.first_request_started.wait(), timeout=5)
            conflict = await control.submit("gemma.prepare", {}, request_id="conflicting-prepare")
            blocked = await control.wait(conflict["operation_id"], timeout=5)
            assert blocked["terminal"] and blocked["status"] == "failed"
            assert blocked["error"]["code"] == "task_busy"
            assert fixture.commands == []
            assert not downloader.release_first_request.is_set()
            assert not (await control.wait(download["operation_id"], timeout=0))["terminal"]
            downloader.release_first_request.set()
            installed = await control.wait(download["operation_id"], timeout=5)
            assert installed["terminal"] and installed["status"] == "applied"
            preparation = await control.submit("gemma.prepare", {}, request_id="legitimate-prepare")
            prepared = await control.wait(preparation["operation_id"], timeout=5)
            assert prepared["terminal"] and prepared["status"] == "applied"
            assert len(fixture.commands) == 1
            assert fixture.native.readiness is None
            assert all(process.returncode == 0 for process in fixture.processes)
            deletion = await control.submit(
                "models.delete", {"model_id": spec.model_id}, request_id="legitimate-delete"
            )
            deleted = await control.wait(deletion["operation_id"], timeout=5)
            assert deleted["terminal"] and deleted["status"] == "applied"
            assert deleted["asset"]["status"] == "missing"

            async def failed_prepare(_selection):
                raise provisioning.GemmaProvisioningError(
                    "Controlled non-busy failure", failure_code="checksum_failed"
                )

            monkeypatch.setattr(owner, "prepare", failed_prepare)
            failed = await control.submit("gemma.prepare", {}, request_id="non-busy-failure")
            failure = await control.wait(failed["operation_id"], timeout=5)
            assert failure["terminal"] and failure["status"] == "failed"
            assert failure["error"]["code"] == "application_operation_failed"
            assert failure["diagnostic_id"] == failure["diagnostics"]["diagnostic_id"]
        finally:
            downloader.release_first_request.set()
