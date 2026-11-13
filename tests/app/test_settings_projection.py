from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from puripuly_heart.app.ports.ui_presentation import SettingsRenderResult
from puripuly_heart.app.services.settings.settings_application import (
    settings_view_surface_snapshots,
)
from puripuly_heart.app.services.settings.settings_projection import SettingsProjectionOwner
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext


def _with_source_language(settings: AppSettingsVNext, language: str) -> AppSettingsVNext:
    return replace(
        settings,
        intent=replace(
            settings.intent,
            languages=replace(settings.intent.languages, source_language=language),
        ),
    )


def _with_show_translation(settings: AppSettingsVNext, enabled: bool) -> AppSettingsVNext:
    return replace(
        settings,
        intent=replace(
            settings.intent,
            overlay=replace(settings.intent.overlay, show_translation=enabled),
        ),
    )


def test_projection_rebases_view_change_onto_current_settings() -> None:
    displayed = _with_source_language(AppSettingsVNext(), "en")
    current = copy.deepcopy(displayed)
    rendered: list[dict[str, object]] = []
    presentation = SimpleNamespace(
        render_settings=lambda **kwargs: rendered.append(kwargs) or SettingsRenderResult(True),
    )
    owner = SettingsProjectionOwner(
        presentation=presentation,
        config_path=Path("settings.json"),
        current_settings=lambda: current,
    )

    assert owner.render(displayed).loaded is True
    pending = _with_show_translation(displayed, False)
    change = owner.capture(pending)
    current = _with_source_language(current, "ja")

    merged = owner.merge_with_current(change)

    assert rendered[0]["general"].effective_peer_source_language == "en"
    assert merged.intent.languages.source_language == "ja"
    assert merged.intent.overlay.show_translation is False


def test_failed_render_preserves_previous_projection_baseline() -> None:
    displayed = _with_source_language(AppSettingsVNext(), "en")
    current = copy.deepcopy(displayed)

    def fail_render(*_args, **_kwargs) -> bool:
        raise RuntimeError("render failed")

    owner = SettingsProjectionOwner(
        presentation=SimpleNamespace(render_settings=fail_render),
        config_path=Path("settings.json"),
        current_settings=lambda: current,
    )
    owner.remember_all(displayed)
    current = _with_source_language(current, "ja")

    result = owner.render(current)
    assert result.loaded is False
    assert result.diagnostics.code == "settings_render_failed"
    assert result.diagnostics.fields["exception_class"] == "RuntimeError"
    assert owner.last_render_result is result
    pending = _with_show_translation(displayed, False)
    merged = owner.merge_with_current(owner.capture(pending))

    assert merged.intent.languages.source_language == "ja"
    assert merged.intent.overlay.show_translation is False


def test_pkce_refresh_uses_specialized_projection_path() -> None:
    settings = AppSettingsVNext()
    refreshes: list[tuple[object, object, Path]] = []

    def refresh(*, provider, prompt, config_path: Path) -> SettingsRenderResult:
        refreshes.append((provider, prompt, config_path))
        return SettingsRenderResult(True)

    owner = SettingsProjectionOwner(
        presentation=SimpleNamespace(
            refresh_settings_after_openrouter_pkce_success=refresh,
        ),
        config_path=Path("settings.json"),
        current_settings=lambda: settings,
    )

    provider, _general, prompt, _overlay = settings_view_surface_snapshots(settings)
    assert (
        owner.refresh_after_openrouter_pkce_success(
            provider=provider,
            prompt=prompt,
            compatibility_settings=settings,
        ).loaded
        is True
    )
    captured = owner.capture(copy.deepcopy(settings))

    assert refreshes == [(provider, prompt, Path("settings.json"))]
    assert captured.can_rebase is True
    assert captured.values_by_path == {}


def test_render_failure_is_secondary_without_changing_committed_outcome():
    from puripuly_heart.app.services.settings.settings_application import SettingsApplicationOwner
    from puripuly_heart.app.services.settings.settings_transaction_result import (
        SettingsTransactionResultOwner,
    )
    from puripuly_heart.core.messages import (
        TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
        TransactionResult,
    )

    settings = AppSettingsVNext()

    def fail_render(**_kwargs):
        raise PermissionError(13, "synthetic-private-path")

    projection = SettingsProjectionOwner(
        presentation=SimpleNamespace(render_settings=fail_render),
        config_path=Path("settings.json"),
        current_settings=lambda: settings,
    )
    results = SettingsTransactionResultOwner()
    SettingsApplicationOwner(
        settings=SimpleNamespace(),
        projection=projection,
        runtime_effects=SimpleNamespace(),
        sync_ui=lambda: None,
        mutation_service_provider=lambda: None,
        consume_superseded_settings=lambda _settings: False,
        active_local_asr_change=lambda *_args: False,
        results=results,
    )
    with results.capture() as scope:
        results.set(
            TransactionResult(
                status=TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
                message=None,
                diagnostics=None,
            )
        )
        rendered = projection.render(settings)
        assert rendered.loaded is False
        assert (
            scope.current.secondary_diagnostics[0].diagnostic_id
            == rendered.diagnostics.diagnostic_id
        )
        results.set(
            TransactionResult(
                status=TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
                message=None,
                diagnostics=None,
            )
        )
    assert scope.current.status == TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED
    assert (
        scope.current.secondary_diagnostics[0].diagnostic_id == rendered.diagnostics.diagnostic_id
    )
    assert scope.current.secondary_diagnostics[0].fields["errno"] == 13
    assert "synthetic-private" not in str(scope.current)


def test_pending_presentation_evidence_transfers_into_next_capture_without_replay():
    from puripuly_heart.app.services.settings.settings_transaction_result import (
        SettingsTransactionResultOwner,
    )
    from puripuly_heart.core.diagnostics import exception_diagnostics
    from puripuly_heart.core.messages import (
        TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
        TransactionResult,
    )

    owner = SettingsTransactionResultOwner()
    causes = tuple(
        exception_diagnostics(
            PermissionError(13, "synthetic-private-path"),
            component="settings_projection",
            operation="render_settings",
            code="settings_render_failed",
        )
        for _ in range(5)
    )
    for cause in causes:
        owner.add_secondary(cause)
    with owner.capture() as empty:
        assert empty.current is None
    import asyncio

    import pytest

    with pytest.raises(asyncio.CancelledError):
        with owner.capture():
            raise asyncio.CancelledError
    success = TransactionResult(
        status=TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
        message=None,
        diagnostics=None,
    )
    with owner.capture() as captured:
        owner.set(success)
        assert captured.current.secondary_diagnostics == causes[:4]
        assert captured.current.secondary_omitted_count == 1
        assert captured.current.status == success.status
    assert owner.current.secondary_diagnostics == causes[:4]
    with owner.capture() as following:
        owner.set(success)
        assert following.current.secondary_diagnostics == ()
        assert following.current.secondary_omitted_count == 0


def test_empty_nested_capture_preserves_evidence_for_parent_transaction():
    from puripuly_heart.app.services.settings.settings_transaction_result import (
        SettingsTransactionResultOwner,
    )
    from puripuly_heart.core.diagnostics import exception_diagnostics
    from puripuly_heart.core.messages import (
        TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
        TransactionResult,
    )

    owner = SettingsTransactionResultOwner()
    cause = exception_diagnostics(
        PermissionError(13, "synthetic-private-path"),
        component="settings_projection",
        operation="render_settings",
        code="settings_render_failed",
    )
    with owner.capture() as parent:
        with owner.capture():
            owner.add_presentation_result(
                SettingsRenderResult(
                    True, secondary_diagnostics=(cause,), secondary_omitted_count=2
                )
            )
        owner.set(
            TransactionResult(
                status=TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
                message=None,
                diagnostics=None,
            )
        )
        assert parent.current.secondary_diagnostics == (cause,)
        assert parent.current.secondary_omitted_count == 2
    assert owner.current.secondary_diagnostics == (cause,)
    assert owner.current.secondary_omitted_count == 2
