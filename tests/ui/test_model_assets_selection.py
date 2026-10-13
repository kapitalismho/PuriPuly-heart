import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from puripuly_heart.core.local_stt_assets import LOCAL_STT_MODEL_ID, PARAKEET_V3_MODEL_ID
from puripuly_heart.core.local_stt_runtime_installer import (
    LocalSTTRuntimeInstallCancelled,
    LocalSTTRuntimeInstallError,
)

from puripuly_heart.core.local_translation.assets import GEMMA_MODEL_ID
from puripuly_heart.ui.components.model_assets_dialog import ModelAssetsDialog
from puripuly_heart.ui.i18n import get_locale, set_locale
from tests.helpers.flet_page import DialogTrackingPage


def make_dialog(selected=(LOCAL_STT_MODEL_ID,)):
    application = SimpleNamespace(
        selected_model_asset_ids=lambda: selected,
        gemma_download_source=lambda: "mirror",
        configure_gemma_source=AsyncMock(),
        provision_gemma_assets=AsyncMock(),
        provision_asr_assets=AsyncMock(),
        cancel_asr_assets=AsyncMock(),
        handle_managed_gemma_notice_action=AsyncMock(),
    )
    page = DialogTrackingPage()
    page.update = lambda: None
    dialog = ModelAssetsDialog(page, application, lambda f: None)
    dialog.open()
    return dialog, application


def test_model_names_and_sources_use_readable_explicit_foreground():
    from puripuly_heart.ui.theme import COLOR_NEUTRAL_DARK

    dialog, _ = make_dialog()
    assert all(
        control.label_style.color == COLOR_NEUTRAL_DARK for control in dialog.checkboxes.values()
    )
    assert all(
        control.text_style.color == COLOR_NEUTRAL_DARK for control in dialog.sources.values()
    )
    assert dialog.dialog.title.color == COLOR_NEUTRAL_DARK


async def test_qwen_notice_selection_downloads_qwen_not_gemma():
    dialog, application = make_dialog()
    assert dialog.checkboxes[LOCAL_STT_MODEL_ID].value
    assert not dialog.checkboxes[GEMMA_MODEL_ID].value
    await dialog.install()
    application.provision_asr_assets.assert_awaited_once()
    kwargs = application.provision_asr_assets.call_args.kwargs
    assert kwargs["model_id"] == LOCAL_STT_MODEL_ID
    assert kwargs["source"] == "modelscope_only"
    application.configure_gemma_source.assert_not_awaited()
    application.provision_gemma_assets.assert_not_awaited()


async def test_nothing_checked_never_downloads():
    dialog, application = make_dialog(())
    await dialog.install()
    application.provision_asr_assets.assert_not_awaited()
    application.provision_gemma_assets.assert_not_awaited()


@pytest.mark.parametrize("source", ["official", "mirror", "modelscope_only"])
async def test_per_model_source_is_forwarded_without_settings_change(source):
    dialog, application = make_dialog()
    dialog.sources[LOCAL_STT_MODEL_ID].value = source
    await dialog.install()
    assert application.provision_asr_assets.call_args.kwargs["source"] == source
    application.configure_gemma_source.assert_not_awaited()


async def test_modelscope_not_offered_for_gemma_or_other_asr_models():
    dialog, _ = make_dialog()
    for model_id, source in dialog.sources.items():
        values = [o.key for o in source.options]
        assert ("modelscope_only" in values) == (model_id == LOCAL_STT_MODEL_ID)


async def test_cancelled_first_download_does_not_start_second():
    dialog, application = make_dialog((LOCAL_STT_MODEL_ID, GEMMA_MODEL_ID))
    application.provision_asr_assets.side_effect = LocalSTTRuntimeInstallCancelled("cancelled")
    await dialog.install()
    application.provision_asr_assets.assert_awaited_once()
    application.provision_gemma_assets.assert_not_awaited()
    assert not dialog.busy


async def test_close_stops_queue_even_if_current_install_committed():
    dialog, application = make_dialog((LOCAL_STT_MODEL_ID, GEMMA_MODEL_ID))

    async def finish(**kwargs):
        dialog.close()

    application.provision_asr_assets.side_effect = finish
    await dialog.install()
    application.provision_gemma_assets.assert_not_awaited()
    assert dialog.closed and dialog.cancel_requested


async def test_duplicate_click_is_ignored_and_only_owned_asr_cancelled():
    dialog, application = make_dialog((LOCAL_STT_MODEL_ID, GEMMA_MODEL_ID))
    entered, release = asyncio.Event(), asyncio.Event()

    async def hold(**kwargs):
        entered.set()
        await release.wait()
        raise LocalSTTRuntimeInstallCancelled("cancelled")

    application.provision_asr_assets.side_effect = hold
    task = asyncio.create_task(dialog.install())
    await entered.wait()
    await dialog.install()
    await dialog.cancel()
    application.cancel_asr_assets.assert_awaited_once_with(model_id=LOCAL_STT_MODEL_ID)
    application.handle_managed_gemma_notice_action.assert_not_awaited()
    release.set()
    await task
    application.provision_asr_assets.assert_awaited_once()
    application.provision_gemma_assets.assert_not_awaited()


async def test_failed_row_keeps_safe_details_and_other_selected_model_can_finish(caplog):
    locale = get_locale()
    set_locale("zh-CN")
    try:
        dialog, application = make_dialog((LOCAL_STT_MODEL_ID, PARAKEET_V3_MODEL_ID))
        application.provision_asr_assets.side_effect = [
            LocalSTTRuntimeInstallError(
                "secret-token-and-signed-url", failure_code="download_timeout", status_code=503
            ),
            None,
        ]
        await dialog.install()
        assert "超时" in dialog.results[LOCAL_STT_MODEL_ID].value
        assert dialog.results[PARAKEET_V3_MODEL_ID].value == "完成"
        dialog.close()
        assert "code=download_timeout" in caplog.text and "http_status=503" in caplog.text
        assert "secret-token" not in caplog.text
        assert "超时" in dialog.status.value
    finally:
        set_locale(locale)
