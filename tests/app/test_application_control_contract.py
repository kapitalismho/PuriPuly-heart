from __future__ import annotations

import asyncio
import json
import logging
import socket
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest
from puripuly_heart.app.services.settings_transaction_result import SettingsTransactionResultOwner

from puripuly_heart.app.services.capture.peer_capture_target_application import (
    PeerCaptureTargetUnavailable,
)
from puripuly_heart.app.services.overlay.overlay_application import OverlayApplicationOwner
from puripuly_heart.app.wiring import wiring_audio_test
from puripuly_heart.composition import application_runtime
from puripuly_heart.composition.headless_application import (
    HeadlessApplicationPresentation,
    compose_headless_application,
)
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext, SecretsIntent
from puripuly_heart.config.settings_vnext.serialization import to_dict
from puripuly_heart.core.audio.format import AudioFrameF32
from puripuly_heart.core.audio.source import (
    MicrophoneTestRouteObservation,
    SelfMicCaptureChannelDecision,
    SoundDeviceInputMetadata,
)
from puripuly_heart.core.diagnostics import DIAGNOSTIC_RECORD_ATTRIBUTE
from puripuly_heart.core.http_extensions import http_extension_secret_key
from puripuly_heart.core.messages import TransactionResult
from puripuly_heart.core.peer_capture import (
    PeerCaptureResolvedTarget,
    PeerCaptureTargetResolution,
    PeerCaptureTargetStatus,
)
from puripuly_heart.domain.events import (
    SelfTurnPresentation,
    SelfTurnTargetPresentation,
    UIEvent,
    UIEventType,
)
from puripuly_heart.domain.models import Transcript, Translation


def isolated_settings(path):
    settings = AppSettingsVNext()
    settings = replace(
        settings,
        intent=replace(
            settings.intent,
            secrets=SecretsIntent(backend="encrypted_file", encrypted_file_path="secrets.json"),
            osc=replace(settings.intent.osc, connection_mode="off"),
            telemetry=replace(settings.intent.telemetry, enabled=False),
        ),
    )
    path.write_text(json.dumps(to_dict(settings)), encoding="utf-8")


@pytest.mark.asyncio
async def test_cli_model_only_selection_restores_saved_luna_connection(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("PURIPULY_HEART_SECRETS_PASSPHRASE", "isolated-test-passphrase")
    path = tmp_path / "settings.json"
    isolated_settings(path)
    app = compose_headless_application(path)
    try:
        await app.start()
        control = app.control()
        control.bind_instance("luna-history-control")
        for index, changes in enumerate(
            (
                {"translation.model": "gpt_6_luna"},
                {"translation.connection": "official_byok"},
                {"translation.model": "gemma4_26b_31b"},
                {"translation.model": "gpt_6_luna"},
            )
        ):
            submitted = await control.submit(
                "settings.apply",
                {"changes": changes},
                request_id=f"luna-history-{index}",
            )
            result = await control.wait(submitted["operation_id"], timeout=10)
            assert result["status"] in {"applied", "degraded"}
        translation = (await control.query("settings.current", {}))["settings"]["intent"][
            "translation"
        ]
        assert translation["model"] == "gpt_6_luna"
        assert translation["connection"] == "official_byok"
        assert translation["connection_history"]["gpt_6_luna"] == "official_byok"
        assert translation["openrouter_selection_alias"] is None
    finally:
        await app.stop()
    restarted = compose_headless_application(path)
    try:
        await restarted.start()
        restored = restarted.control()
        restored.bind_instance("luna-history-restarted")
        translation = (await restored.query("settings.current", {}))["settings"]["intent"][
            "translation"
        ]
        assert translation["model"] == "gpt_6_luna"
        assert translation["connection"] == "official_byok"
        assert translation["connection_history"]["gpt_6_luna"] == "official_byok"
    finally:
        await restarted.stop()


def test_headless_runtime_error_is_localized_in_dashboard_state(caplog) -> None:
    from puripuly_heart.ui.i18n import t

    presentation = HeadlessApplicationPresentation()
    presentation.set_locale("en")
    with caplog.at_level(logging.ERROR):
        presentation.show_message("local_stt.download_failed", is_error=True)

    assert presentation.dashboard.issue == t("local_stt.download_failed")
    assert [(record.levelno, record.getMessage()) for record in caplog.records] == [
        (logging.ERROR, "Application message: local_stt.download_failed")
    ]


@pytest.fixture
def headless_event_bridge():
    presentation = HeadlessApplicationPresentation()
    presentation.application = SimpleNamespace(
        get_event_language_codes=lambda: ("fr", "en"),
        state=lambda: SimpleNamespace(translation_enabled=True, stt_state=None),
        clear_managed_auth_pending_state=lambda: None,
        schedule_github_star_prompt_translation_success_observed=lambda: None,
    )
    bridge = presentation.create_ui_event_bridge(event_queue=asyncio.Queue(), runtime_logging=None)
    yield presentation, bridge
    bridge.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", ("cancelled", "expired"))
async def test_headless_self_turn_progress_rejects_stale_state_and_recovers(
    headless_event_bridge, caplog, terminal
):
    presentation, bridge = headless_event_bridge
    dashboard = presentation.dashboard
    source = SelfTurnPresentation(
        utterance_id=uuid4(),
        turn_generation=1,
        turn_order=1,
        revision=1,
        source_revision=1,
        source_text="synthetic progressive source",
        source_language="fr",
        source_closed=False,
        status="active",
        targets=(
            SelfTurnTargetPresentation("en", 1, "pending"),
            SelfTurnTargetPresentation("de", 1, "pending"),
        ),
        created_at=0.0,
    )

    async def publish(snapshot):
        await bridge._handle_event(
            UIEvent(
                UIEventType.SELF_TURN_UPDATED,
                utterance_id=snapshot.utterance_id,
                payload=snapshot,
                source="Mic",
                channel="self",
            )
        )

    def ready(snapshot, language, text):
        return SelfTurnTargetPresentation(
            language,
            snapshot.source_revision,
            "ready",
            Translation(
                snapshot.utterance_id,
                text,
                source_text=snapshot.source_text,
                source_language=snapshot.source_language,
                target_language=language,
            ),
        )

    await publish(source)
    assert dashboard.transcript == source.source_text
    assert dashboard.translation is None
    first = replace(
        source,
        revision=2,
        targets=(ready(source, "en", "synthetic first translation"), source.targets[1]),
    )
    await publish(first)
    assert dashboard.translation == "synthetic first translation"
    both = replace(
        first,
        revision=3,
        targets=(first.targets[0], ready(source, "de", "synthetic second translation")),
    )
    await publish(both)
    assert dashboard.translation.splitlines() == [
        "synthetic first translation",
        "synthetic second translation",
    ]
    extended = replace(both, revision=4, source_revision=2, source_text="synthetic extended source")
    await publish(extended)
    assert dashboard.transcript == extended.source_text
    assert dashboard.translation is None
    current = replace(
        extended,
        revision=5,
        targets=(
            ready(extended, "en", "synthetic current translation"),
            replace(both.targets[1], status="failed"),
        ),
    )
    await publish(current)
    assert dashboard.translation == "synthetic current translation"
    await publish(both)
    await publish(replace(both, revision=6))
    assert dashboard.transcript == current.source_text
    assert dashboard.translation == "synthetic current translation"
    ended = replace(current, revision=6, status=terminal)
    await publish(ended)
    assert dashboard.transcript == current.source_text
    assert dashboard.translation is None
    await publish(replace(current, revision=99, source_closed=True, status="complete"))
    assert dashboard.translation is None
    successor = replace(
        source,
        utterance_id=uuid4(),
        turn_order=2,
        source_text="synthetic successor source",
    )
    successor = replace(
        successor,
        targets=(
            ready(successor, "en", "synthetic successor translation"),
            replace(ready(successor, "de", "synthetic invalid target"), status="cancelled"),
        ),
    )
    await publish(successor)
    assert dashboard.transcript == successor.source_text
    assert dashboard.translation == "synthetic successor translation"
    await publish(replace(ended, revision=100))
    assert dashboard.transcript == successor.source_text
    assert dashboard.translation == "synthetic successor translation"
    generation = replace(
        source,
        utterance_id=uuid4(),
        turn_generation=2,
        source_text="synthetic next generation source",
    )
    generation = replace(
        generation, targets=(ready(generation, "en", "synthetic generation translation"),)
    )
    await publish(generation)
    await publish(replace(successor, revision=100, turn_order=100))
    assert dashboard.transcript == generation.source_text
    assert dashboard.translation == "synthetic generation translation"
    failures = [
        value[0]
        for record in caplog.records
        if (value := getattr(record, DIAGNOSTIC_RECORD_ATTRIBUTE, None)) is not None
        and value[0].code == "destination_publish_failed"
    ]
    assert failures == []


@pytest.mark.asyncio
@pytest.mark.parametrize("channel,source", (("self", "You"), ("peer", "Peer")))
async def test_headless_legacy_manual_and_peer_events_remain_supported(
    headless_event_bridge, caplog, channel, source
):
    presentation, bridge = headless_event_bridge
    utterance_id = uuid4()
    transcript = Transcript(utterance_id, "synthetic legacy source", True, channel=channel)
    translation = Translation(
        utterance_id,
        "synthetic legacy translation",
        source_text=transcript.text,
        source_language="fr",
        target_language="en",
        channel=channel,
    )
    for event_type, payload in (
        (UIEventType.TRANSCRIPT_FINAL, transcript),
        (UIEventType.TRANSLATION_DONE, translation),
    ):
        await bridge._handle_event(
            UIEvent(
                event_type,
                utterance_id=utterance_id,
                payload=payload,
                source=source,
                channel=channel,
            )
        )
    assert presentation.dashboard.transcript == transcript.text
    assert presentation.dashboard.translation == translation.text
    assert not any(
        value[0].code == "destination_publish_failed"
        for record in caplog.records
        if (value := getattr(record, DIAGNOSTIC_RECORD_ATTRIBUTE, None)) is not None
    )


@pytest.mark.parametrize("route", ("mixed", "provider_only", "general_only"))
@pytest.mark.parametrize("synthetic_provider", (False, True))
async def test_settings_routes_converge_provider_and_general_effects(
    tmp_path, monkeypatch, route, synthetic_provider
):
    from puripuly_heart.app.wiring import wiring_provider_runtime
    from puripuly_heart.ui.i18n import get_locale

    monkeypatch.setenv("PURIPULY_HEART_SECRETS_PASSPHRASE", "isolated-test-passphrase")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    path = tmp_path / "settings.json"
    isolated_settings(path)
    app = compose_headless_application(path)
    try:
        await app.start()
        control = app.control()
        control.bind_instance("mixed-settings")
        previous = await control.query("settings.current", {})
        initial_provider = control.pipeline.llm_runtime.provider
        assert initial_provider is not None
        replacement = SimpleNamespace()
        replacements = []
        audio = app._settings.application.runtime_effects._audio_test
        stop_calls = []
        stop_audio = type(audio).stop

        async def observe_stop(runtime):
            if runtime is audio:
                stop_calls.append(True)
            await stop_audio(runtime)

        monkeypatch.setattr(type(audio), "stop", observe_stop)
        from puripuly_heart.app.services import application_audio_devices

        monkeypatch.setattr(
            application_audio_devices,
            "enumerate_audio_devices",
            lambda _settings: {
                "host_apis": ["Windows WASAPI"],
                "microphones": [{"name": "isolated microphone", "host_api": "Windows WASAPI"}],
                "loopback_outputs": [],
            },
        )
        if synthetic_provider:

            def create_backend(**kwargs):
                replacements.append(kwargs)
                return replacement

            monkeypatch.setattr(
                wiring_provider_runtime, "create_translation_backend", create_backend
            )
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
            receiver.bind(("127.0.0.1", 0))
            send_port = receiver.getsockname()[1]
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as reservation:
                reservation.bind(("127.0.0.1", 0))
                receive_port = reservation.getsockname()[1]
            provider_changes = {
                "translation.model": "deepseek_v4_flash_41",
                "translation.connection": "official_byok",
            }
            general_changes = {
                "locale": "ja",
                "languages": {"source": "ja", "target": "en"},
                "osc.connection": {
                    "mode": "manual",
                    "send_port": send_port,
                    "receive_port": receive_port,
                },
                "chatbox.activation_notice.enabled": False,
                "self_vad.speech_threshold": 0.65,
                "audio.input_device": "isolated microphone",
            }
            changes = {}
            if route != "general_only":
                changes.update(provider_changes)
            if route != "provider_only":
                changes.update(general_changes)
            submitted = await control.submit(
                "settings.apply",
                {"changes": changes},
                request_id="mixed-settings",
                expected_revision=previous["revision"],
            )
            result = await control.wait(submitted["operation_id"], timeout=10)
            current = await control.query("settings.current", {})
            intent = current["settings"]["intent"]
            if route != "general_only":
                assert intent["translation"]["model"] == "deepseek_v4_flash_41"
                assert intent["translation"]["connection"] == "official_byok"
                assert control.pipeline.llm_runtime.provider is (
                    replacement if synthetic_provider else None
                )
                assert current["revision"] == previous["revision"] + 1
                assert result["status"] == ("applied" if synthetic_provider else "degraded")
                presentation = app._settings.application.runtime_effects._presentation
                assert presentation.projections["translation_needs_key"] is True
                assert (await control.query("secrets.presence", {}))["presence"][
                    "deepseek_api_key"
                ] is False
                if not synthetic_provider:
                    assert result["transaction"]["diagnostics"]["code"] == (
                        "provider_runtime_apply_exception"
                    )
            else:
                assert result["status"] == "applied"
                assert control.pipeline.llm_runtime.provider is initial_provider
                assert not replacements
            assert (
                json.loads(path.read_text(encoding="utf-8"))["intent"]
                == to_dict(app.compatibility_settings())["intent"]
            )
            if route != "provider_only":
                assert intent["languages"]["source_language"] == "ja"
                assert intent["languages"]["target_language"] == "en"
                config = control.pipeline.translation_runtime_configuration.snapshot().value
                assert config.source_language == "ja"
                assert config.target_language == "en"
                assert get_locale() == "ja"
                assert app.effective_osc_ports() == (send_port, receive_port)
                assert control.pipeline.current.output_runtime.activation_notice_enabled is False
                assert intent["stt"]["vad_speech_threshold"] == 0.65
                assert audio.vad_settings("self").speech_threshold == 0.65
                assert stop_calls
            else:
                assert intent["languages"] == previous["settings"]["intent"]["languages"]
                assert intent["ui"] == previous["settings"]["intent"]["ui"]
                assert app.effective_osc_ports() == (None, None)
            assert control.settings.mutation_depth == 0
            assert control.settings.rollback_pending is False
    finally:
        await app.stop()


@pytest.mark.parametrize("retry_route", ("mixed", "provider_only"))
@pytest.mark.parametrize("failure_mode", ("exception", "unavailable"))
async def test_failed_mixed_settings_explicit_retry_converges_without_general_only_retry(
    tmp_path, monkeypatch, retry_route, failure_mode
):
    from puripuly_heart.app.wiring import wiring_provider_runtime
    from puripuly_heart.ui.i18n import get_locale

    monkeypatch.setenv("PURIPULY_HEART_SECRETS_PASSPHRASE", "isolated-test-passphrase")
    path = tmp_path / "settings.json"
    isolated_settings(path)
    app = compose_headless_application(path)
    try:
        await app.start()
        control = app.control()
        control.bind_instance("mixed-provider-retry")
        previous = await control.query("settings.current", {})
        assert control.pipeline.llm_runtime.provider is not None
        replacement = SimpleNamespace()
        attempts = []
        available = False

        def create_backend(**_kwargs):
            attempts.append(available)
            if available:
                return replacement
            if failure_mode == "exception":
                raise ValueError("controlled provider construction failure")
            return None

        monkeypatch.setattr(wiring_provider_runtime, "create_translation_backend", create_backend)
        provider_changes = {
            "translation.model": "deepseek_v4_flash_41",
            "translation.connection": "official_byok",
        }
        mixed_changes = {
            **provider_changes,
            "locale": "ja",
            "languages": {"source": "ja", "target": "en"},
        }
        expected_code = (
            "provider_runtime_apply_exception"
            if failure_mode == "exception"
            else "provider_runtime_apply_unavailable"
        )
        for attempt in range(2):
            submitted = await control.submit(
                "settings.apply",
                {"changes": mixed_changes},
                request_id=f"mixed-failed-attempt-{attempt}",
            )
            result = await control.wait(submitted["operation_id"], timeout=10)
            assert result["terminal"] is True
            assert result["status"] == "degraded"
            assert result["transaction"]["status"] == "settings_commit_success_runtime_degraded"
            assert result["transaction"]["diagnostics"]["code"] == expected_code
            assert len(attempts) == attempt + 1
            assert control.pipeline.llm_runtime.provider is None
            current = await control.query("settings.current", {})
            assert current["revision"] == previous["revision"] + (attempt == 0)
            if attempt == 0:
                selected = deepcopy(current["settings"])
            else:
                assert current["settings"] == selected
            previous = current
            assert get_locale() == "ja"
            config = control.pipeline.translation_runtime_configuration.snapshot().value
            assert config.source_language == "ja"
            assert config.target_language == "en"
            assert control.settings.mutation_depth == 0
            assert control.settings.rollback_pending is False

        available = True
        for locale in ("en", "ja"):
            submitted = await control.submit(
                "settings.apply",
                {"changes": {"locale": locale}},
                request_id=f"general-only-no-provider-retry-{locale}",
            )
            general = await control.wait(submitted["operation_id"], timeout=10)
            assert general["status"] == "applied"
            assert get_locale() == locale
            assert len(attempts) == 2
            assert control.pipeline.llm_runtime.provider is None
        previous = await control.query("settings.current", {})
        submitted = await control.submit(
            "settings.apply",
            {"changes": mixed_changes if retry_route == "mixed" else provider_changes},
            request_id="explicit-provider-recovery",
        )
        recovered = await control.wait(submitted["operation_id"], timeout=10)
        assert recovered["terminal"] is True
        assert recovered["status"] == "applied"
        assert recovered["transaction"]["status"] == "settings_commit_success_runtime_applied"
        assert len(attempts) == 3
        assert control.pipeline.llm_runtime.provider is replacement
        current = await control.query("settings.current", {})
        assert current["revision"] == previous["revision"]
        assert current["settings"] == selected
        assert get_locale() == "ja"

        submitted = await control.submit(
            "settings.apply",
            {"changes": {**provider_changes, "locale": "ko", "languages": {"target": "ja"}}},
            request_id="healthy-provider-independent-general-effects",
        )
        healthy = await control.wait(submitted["operation_id"], timeout=10)
        assert healthy["status"] == "applied"
        assert len(attempts) == 3
        assert control.pipeline.llm_runtime.provider is replacement
        assert (await control.query("settings.current", {}))["revision"] == current["revision"] + 1
        assert get_locale() == "ko"
        config = control.pipeline.translation_runtime_configuration.snapshot().value
        assert config.source_language == "ja"
        assert config.target_language == "ja"
        assert (
            json.loads(path.read_text(encoding="utf-8"))["intent"]
            == to_dict(app.compatibility_settings())["intent"]
        )
        assert control.settings.mutation_depth == 0
        assert control.settings.rollback_pending is False
    finally:
        await app.stop()


@pytest.mark.parametrize(
    ("setup", "changes"),
    (
        (
            {},
            {"translation.model": "managed_gemma", "translation.connection": "cpu"},
        ),
        (
            {"translation.model": "local_llm", "translation.connection": "ollama"},
            {"local_llm.base_url": "http://127.0.0.1:12345/v1", "local_llm.model": "isolated"},
        ),
        (
            {"translation.model": "qwen38_flash", "translation.connection": "official_byok"},
            {"translation.qwen.region": "singapore"},
        ),
    ),
)
@pytest.mark.parametrize("provider_fails", (False, True))
async def test_mixed_llm_options_and_managed_switch_replace_once(
    tmp_path, monkeypatch, setup, changes, provider_fails
):
    from puripuly_heart.app.wiring import wiring_provider_runtime
    from puripuly_heart.ui.i18n import get_locale

    monkeypatch.setenv("PURIPULY_HEART_SECRETS_PASSPHRASE", "isolated-test-passphrase")
    path = tmp_path / "settings.json"
    isolated_settings(path)
    app = compose_headless_application(path)
    try:
        await app.start()
        control = app.control()
        control.bind_instance("mixed-options")
        assert control.pipeline.llm_runtime.provider is not None
        replacements = []

        def create_backend(**kwargs):
            provider = SimpleNamespace()
            replacements.append((provider, kwargs))
            return provider

        monkeypatch.setattr(wiring_provider_runtime, "create_translation_backend", create_backend)
        if setup:
            submitted = await control.submit(
                "settings.apply", {"changes": setup}, request_id="setup-options"
            )
            assert (await control.wait(submitted["operation_id"], timeout=10))[
                "status"
            ] == "applied"
        initial_provider = control.pipeline.llm_runtime.provider
        replacements.clear()
        if provider_fails:

            def fail_backend(**kwargs):
                replacements.append((None, kwargs))
                raise ValueError("controlled provider construction failure")

            monkeypatch.setattr(wiring_provider_runtime, "create_translation_backend", fail_backend)
        previous = await control.query("settings.current", {})
        submitted = await control.submit(
            "settings.apply",
            {"changes": {**changes, "locale": "ja", "languages": {"source": "ja", "target": "en"}}},
            request_id="mixed-options",
        )
        result = await control.wait(submitted["operation_id"], timeout=10)
        assert result["status"] == ("degraded" if provider_fails else "applied")
        assert len(replacements) == 1
        assert control.pipeline.llm_runtime.provider is replacements[0][0]
        assert control.pipeline.llm_runtime.provider is not initial_provider
        current = await control.query("settings.current", {})
        assert current["revision"] == previous["revision"] + 1
        assert current["settings"]["intent"]["ui"]["locale"] == "ja"
        assert get_locale() == "ja"
        config = control.pipeline.translation_runtime_configuration.snapshot().value
        assert config.source_language == "ja"
        assert config.target_language == "en"
        if provider_fails:
            assert (
                result["transaction"]["diagnostics"]["code"] == "provider_runtime_apply_exception"
            )
        assert control.settings.mutation_depth == 0
        assert control.settings.rollback_pending is False
    finally:
        await app.stop()


@pytest.mark.parametrize("provider_fails", (False, True))
async def test_mixed_general_failure_still_applies_provider_and_retains_causes(
    tmp_path, monkeypatch, provider_fails
):
    from puripuly_heart.app.wiring import wiring_provider_runtime
    from puripuly_heart.ui.i18n import get_locale

    monkeypatch.setenv("PURIPULY_HEART_SECRETS_PASSPHRASE", "isolated-test-passphrase")
    path = tmp_path / "settings.json"
    isolated_settings(path)
    app = compose_headless_application(path)
    try:
        await app.start()
        control = app.control()
        control.bind_instance("mixed-failure")
        initial_provider = control.pipeline.llm_runtime.provider
        assert initial_provider is not None
        replacement = SimpleNamespace()
        replacements = []

        def create_backend(**kwargs):
            replacements.append(kwargs)
            if provider_fails:
                raise ValueError("controlled provider construction failure")
            return replacement

        def fail_locale():
            raise RuntimeError("controlled locale presentation failure")

        monkeypatch.setattr(wiring_provider_runtime, "create_translation_backend", create_backend)
        effects = app._settings.application.runtime_effects
        monkeypatch.setattr(effects._presentation, "apply_locale", fail_locale)
        previous = await control.query("settings.current", {})
        submitted = await control.submit(
            "settings.apply",
            {
                "changes": {
                    "translation.model": "deepseek_v4_flash_41",
                    "translation.connection": "official_byok",
                    "locale": "ja",
                    "languages": {"source": "ja", "target": "en"},
                }
            },
            request_id="mixed-failure",
        )
        result = await control.wait(submitted["operation_id"], timeout=10)
        assert result["status"] == "degraded"
        assert len(replacements) == 1
        assert control.pipeline.llm_runtime.provider is (None if provider_fails else replacement)
        assert get_locale() == "ja"
        assert (
            control.pipeline.translation_runtime_configuration.snapshot().value.source_language
            == "ja"
        )
        current = await control.query("settings.current", {})
        assert current["revision"] == previous["revision"] + 1
        assert (
            json.loads(path.read_text(encoding="utf-8"))["intent"]
            == to_dict(app.compatibility_settings())["intent"]
        )
        transaction = result["transaction"]
        assert transaction["diagnostics"]["code"] == "settings_runtime_apply_exception"
        if provider_fails:
            secondary = transaction["secondary_diagnostics"]
            assert [item["code"] for item in secondary] == ["provider_runtime_apply_exception"]
            assert secondary[0]["cause_id"] == transaction["diagnostics"]["diagnostic_id"]
        assert control.settings.mutation_depth == 0
        assert control.settings.rollback_pending is False
        committed_provider = control.pipeline.llm_runtime.provider

        def fail_save(*_args):
            raise OSError("controlled next persistence failure")

        with monkeypatch.context() as patch:
            patch.setattr(control.settings.persistence, "persist", fail_save)
            submitted = await control.submit(
                "settings.apply",
                {"changes": {"locale": "en"}},
                request_id="next-save-failure",
            )
            failed = await control.wait(submitted["operation_id"], timeout=10)
        assert failed["status"] == "persistence_failed"
        assert await control.query("settings.current", {}) == current
        assert control.pipeline.llm_runtime.provider is committed_provider
    finally:
        await app.stop()


async def test_mixed_save_failure_keeps_binding_settings_and_revision_then_recovers(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PURIPULY_HEART_SECRETS_PASSPHRASE", "isolated-test-passphrase")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    path = tmp_path / "settings.json"
    isolated_settings(path)
    app = compose_headless_application(path)
    try:
        await app.start()
        control = app.control()
        control.bind_instance("mixed-save")
        previous = await control.query("settings.current", {})
        initial_provider = control.pipeline.llm_runtime.provider
        assert initial_provider is not None
        previous_file = path.read_bytes()

        def fail_save(*_args):
            raise OSError("controlled persistence failure")

        changes = {
            "translation.model": "deepseek_v4_flash_41",
            "translation.connection": "official_byok",
            "locale": "ja",
            "languages": {"source": "ja", "target": "en"},
        }
        with monkeypatch.context() as patch:
            patch.setattr(control.settings.persistence, "persist", fail_save)
            submitted = await control.submit(
                "settings.apply", {"changes": changes}, request_id="mixed-save-failure"
            )
            result = await control.wait(submitted["operation_id"], timeout=10)
        assert result["status"] == "persistence_failed"
        assert await control.query("settings.current", {}) == previous
        assert path.read_bytes() == previous_file
        assert control.pipeline.llm_runtime.provider is initial_provider
        assert control.settings.mutation_depth == 0
        assert control.settings.rollback_pending is False
        submitted = await control.submit(
            "settings.apply",
            {"changes": changes},
            request_id="mixed-save-recovery",
            expected_revision=previous["revision"],
        )
        result = await control.wait(submitted["operation_id"], timeout=10)
        assert result["status"] == "degraded"
        assert control.pipeline.llm_runtime.provider is None
        assert (await control.query("settings.current", {}))["revision"] == previous["revision"] + 1
    finally:
        await app.stop()


async def test_provider_only_full_draft_does_not_overwrite_newer_general_settings(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PURIPULY_HEART_SECRETS_PASSPHRASE", "isolated-test-passphrase")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    path = tmp_path / "settings.json"
    isolated_settings(path)
    app = compose_headless_application(path)
    try:
        await app.start()
        control = app.control()
        control.bind_instance("stale-provider-draft")
        stale = deepcopy(app.compatibility_settings())
        initial_provider = control.pipeline.llm_runtime.provider
        submitted = await control.submit(
            "settings.apply",
            {"changes": {"locale": "ja", "languages": {"target": "ja"}}},
            request_id="current-general",
        )
        assert (await control.wait(submitted["operation_id"], timeout=10))["status"] == "applied"
        general = deepcopy(app.compatibility_settings().intent)
        assert await app.apply_providers(stale) is True
        assert app.compatibility_settings().intent.ui == general.ui
        assert app.compatibility_settings().intent.languages == general.languages
        assert control.pipeline.llm_runtime.provider is initial_provider
    finally:
        await app.stop()


async def test_mixed_provider_interruption_retains_commit_without_claiming_runtime_applied(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PURIPULY_HEART_SECRETS_PASSPHRASE", "isolated-test-passphrase")
    path = tmp_path / "settings.json"
    isolated_settings(path)
    app = compose_headless_application(path)
    try:
        await app.start()
        control = app.control()
        control.bind_instance("mixed-interrupted")
        previous = await control.query("settings.current", {})
        initial_provider = control.pipeline.llm_runtime.provider
        entered = asyncio.Event()

        async def block_rebuild():
            entered.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(
            app._provider.provider_application.runtime, "rebuild_llm", block_rebuild
        )
        submitted = await control.submit(
            "settings.apply",
            {
                "changes": {
                    "translation.model": "deepseek_v4_flash_41",
                    "translation.connection": "official_byok",
                    "locale": "ja",
                }
            },
            request_id="mixed-interrupted",
        )
        await asyncio.wait_for(entered.wait(), timeout=10)
        control.freeze_ingress()
        await control.drain_operations()
        result = await control.operation(submitted["operation_id"])
        assert result["status"] == "interrupted"
        assert result["transaction"]["status"] == "settings_commit_success_runtime_interrupted"
        assert result["revision"] == previous["revision"] + 1
        assert json.loads(path.read_text(encoding="utf-8"))["intent"]["ui"]["locale"] == "ja"
        assert control.pipeline.llm_runtime.provider is initial_provider
        assert control.settings.mutation_depth == 0
        assert control.settings.rollback_pending is False
    finally:
        await app.stop()


@pytest.mark.asyncio
async def test_declared_extension_secret_roundtrip_is_private_and_revision_safe(
    tmp_path, monkeypatch
):
    extensions = tmp_path / "extensions"
    extensions.mkdir()
    (extensions / "sample.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "id": "sample",
                "name": "Sample",
                "url": "https://example.test/translate",
                "headers": {"Authorization": "Bearer {{secret:key}}"},
                "request": {"query": {}, "body": {"type": "json", "value": {"text": "{{text}}"}}},
                "response": {"type": "text"},
                "secrets": [{"id": "key", "label": "API Key"}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(application_runtime, "default_http_extensions_dir", lambda: extensions)
    monkeypatch.setenv("PURIPULY_HEART_SECRETS_PASSPHRASE", "isolated-test-passphrase")
    path = tmp_path / "settings.json"
    isolated_settings(path)
    app = compose_headless_application(path)
    try:
        await app.start()
        control = app.control()
        control.bind_instance("isolated-control")
        key = http_extension_secret_key("sample", "key")
        from puripuly_heart.ui.i18n import t

        catalog = await control.query("settings.choices", {})
        assert "sample" in catalog["choices"]["translation.http_extension_id"]
        terms = await control.query("consent.peer_translation", {})
        assert terms["accepted"] is False
        assert terms["terms"] == t("peer_translation_eula.body")
        current = await control.query("settings.current", {})
        current_locale = current["settings"]["intent"]["ui"]["locale"]
        valid_locale = next(
            (locale for locale in catalog["choices"]["locale"] if locale != current_locale),
            current_locale,
        )
        accepted = await control.submit(
            "settings.apply",
            {"changes": {"locale": valid_locale}},
            request_id="catalog-valid-locale",
        )
        applied = await control.wait(accepted["operation_id"], timeout=5)
        assert applied["status"] == "applied"
        assert (await control.query("settings.current", {}))["settings"]["intent"]["ui"][
            "locale"
        ] == valid_locale
        unchanged = await control.query("settings.current", {})
        malformed = await control.submit(
            "settings.apply",
            {
                "changes": {
                    "locale": valid_locale,
                    "translation.model": "not-a-model",
                }
            },
            request_id="catalog-invalid-model",
        )
        rejected = await control.wait(malformed["operation_id"], timeout=5)
        assert rejected["status"] == "rejected"
        after = await control.query("settings.current", {})
        assert after["settings"] == unchanged["settings"]
        assert after["revision"] == unchanged["revision"]
        assert current["revision"] <= unchanged["revision"]
        bad_operations = (
            (
                "capture.set",
                {"channel": "self", "enabled": True, "accept_terms": True},
            ),
            (
                "settings.apply",
                {"changes": {"locale": valid_locale, "languages": {"source": "not-a-language"}}},
            ),
            (
                "settings.apply",
                {"changes": {"osc.connection": {"mode": "off", "receive_port": True}}},
            ),
            (
                "settings.apply",
                {"changes": {"translation.connection_history": {"not-a-model": "direct"}}},
            ),
            ("settings.apply", {"changes": {"stt.cloud_free_tier_providers": []}}),
            ("provider.apply", {"channel": "self", "provider": "not-a-provider"}),
            ("models.install", {"model_ids": "not-a-list"}),
            (
                "overlay.calibrate",
                {"action": "change", "field": "unknown", "value": 1},
            ),
            ("overlay.calibrate", {"action": "change", "field": "distance", "value": True}),
            ("overlay.calibrate", {"action": "begin", "field": "anchor", "value": "top"}),
            ("secrets.set", {"name": key, "value": 1}),
            (
                "auth.login",
                {
                    "provider": "discord",
                    "referral_id": 5,
                    "open_browser": False,
                },
            ),
            (
                "auth.login",
                {
                    "provider": "qq",
                    "qq_identity": "offline",
                    "credential": "offline",
                    "open_browser": True,
                },
            ),
        )
        for index, (command, arguments) in enumerate(bad_operations):
            pending = await control.submit(
                command,
                arguments,
                request_id=f"invalid-{index}",
            )
            invalid = await control.wait(pending["operation_id"], timeout=5)
            assert invalid["status"] == "rejected"
        after_invalid = await control.query("settings.current", {})
        assert after_invalid["settings"] == after["settings"]
        assert (await control.query("secrets.presence", {}))["presence"][key] is False
        assert key in control.capabilities()["secret_keys"]
        before = await control.query("secrets.presence", {})
        assert before["presence"][key] is False
        secret = "sensitive-control-test-value"
        accepted = await control.submit(
            "secrets.set", {"name": key, "value": secret}, request_id="set-once"
        )
        assert accepted["terminal"] is False
        stored = await control.wait(accepted["operation_id"])
        assert stored["status"] == "applied" and stored["terminal"] is True
        assert stored["transaction"]["status"] == "settings_commit_success_runtime_applied"
        assert secret not in json.dumps(stored)
        assert secret not in json.dumps(await control.query("settings.current", {}))
        assert (await control.query("secrets.presence", {}))["presence"][key] is True
        duplicate = await control.submit(
            "secrets.set", {"name": key, "value": secret}, request_id="set-once"
        )
        assert duplicate == stored
        unknown = await control.submit(
            "secrets.set",
            {"name": "http_extension.sample.undeclared", "value": secret},
            request_id="undeclared",
        )
        rejected = await control.wait(unknown["operation_id"])
        assert rejected["status"] == "rejected" and rejected["terminal"] is True
        assert (await control.query("secrets.presence", {}))["presence"][key] is True
        unverified = await control.submit(
            "secrets.verify", {"name": key, "value": secret}, request_id="verify"
        )
        verdict = await control.wait(unverified["operation_id"])
        assert verdict["status"] == "action_required" and verdict["verification"] == "unavailable"
        assert verdict["terminal"] is True
        removed = await control.submit("secrets.delete", {"name": key}, request_id="delete")
        deleted = await control.wait(removed["operation_id"])
        assert deleted["status"] == "applied" and deleted["terminal"] is True
        assert (await control.query("secrets.presence", {}))["presence"][key] is False
    finally:
        await app.stop()


class LoudnessEngine:
    def speech_probability(self, samples, *, sample_rate_hz):
        return 0.9 if float(np.max(np.abs(samples))) > 0.1 else 0.05

    def reset(self):
        return None


class ControlledSource:
    def __init__(self):
        self.frame = asyncio.Event()
        self.closed = 0

    async def frames(self):
        await self.frame.wait()
        samples = np.full(512 * 4, 0.75, dtype=np.float32)
        samples[::2] = -0.75
        yield AudioFrameF32(samples=samples, sample_rate_hz=16000)
        await asyncio.Event().wait()

    async def close(self):
        self.closed += 1


class ControlledTargetResolver:
    def __init__(self):
        self.available = False

    async def resolve(self, target):
        if not self.available:
            return PeerCaptureTargetResolution(
                status=PeerCaptureTargetStatus.UNAVAILABLE,
                reason="process_not_running",
            )
        return PeerCaptureTargetResolution(
            status=PeerCaptureTargetStatus.RESOLVED,
            target=PeerCaptureResolvedTarget(intent=target),
        )


@pytest.mark.asyncio
async def test_control_audio_test_reports_channel_failures_pending_stop_and_live_observations(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PURIPULY_HEART_SECRETS_PASSPHRASE", "isolated-test-passphrase")
    monkeypatch.setattr(
        application_runtime, "default_http_extensions_dir", lambda: tmp_path / "extensions"
    )
    path = tmp_path / "settings.json"
    isolated_settings(path)
    app = compose_headless_application(path)
    try:
        await app.start()
        control = app.control()
        control.bind_instance("isolated-audio-test")
        audio_test = app._audio_test.audio_test
        resolver = ControlledTargetResolver()
        audio_test.target_resolver = resolver
        audio_test.vad_engine_factory = LoudnessEngine
        route = MicrophoneTestRouteObservation(
            saved_host_api="",
            actual_host_api="",
            requested_device="controlled",
            hostapi_index=None,
            resolved_device_idx=None,
            resolved_device_name=None,
            resolution_exception_class="DeviceUnavailable",
            resolution_exception_message=None,
            should_attempt_open=False,
            wasapi_auto_convert=False,
            wasapi_exclusive=False,
        )
        monkeypatch.setattr(
            wiring_audio_test, "observe_microphone_test_route", lambda **_kwargs: route
        )
        receipt = await control.submit("audio.test", {"enabled": True}, request_id="missing")
        failed = await control.wait(receipt["operation_id"], timeout=5)
        assert failed["status"] == "action_required"
        channels = failed["audio_test"]["channels"]
        assert channels["self"]["failure_reason"] == "input_route_unavailable"
        assert channels["peer"]["failure_reason"] == "target_unavailable"
        assert channels["peer"]["failure_type"] == "process_not_running"
        assert failed["audio_test"]["effective_active"] is False
        assert (await control.query("app.status", {}))["audio_test"]["state"] == "failed"

        mic = ControlledSource()
        loopback = ControlledSource()
        audio_test.microphone_source_factory = lambda **_kwargs: mic
        audio_test.loopback_source_factory = lambda _target, _device: loopback
        resolver.available = True
        monkeypatch.setattr(
            wiring_audio_test,
            "observe_microphone_test_route",
            lambda **_kwargs: replace(
                route,
                should_attempt_open=True,
                resolved_device_idx=2,
                resolved_device_name="controlled",
            ),
        )
        monkeypatch.setattr(
            wiring_audio_test,
            "determine_self_mic_capture_channels",
            lambda **_kwargs: SelfMicCaptureChannelDecision(
                device_idx=2,
                internal_channels=1,
                preferred_capture_channels=1,
                metadata=SoundDeviceInputMetadata(
                    device_idx=2,
                    name="controlled",
                    max_input_channels=1,
                    default_samplerate=16000.0,
                    metadata_status="ok",
                ),
            ),
        )
        pending_receipt = await control.submit(
            "audio.test", {"enabled": True}, request_id="pending-stop"
        )
        pending_start = await control.wait(pending_receipt["operation_id"], timeout=0.05)
        assert pending_start["terminal"] is False
        off_receipt = await control.submit(
            "audio.test", {"enabled": False}, request_id="stop-pending"
        )
        pending_off = await control.wait(off_receipt["operation_id"], timeout=5)
        assert pending_off["status"] == "applied"
        assert pending_off["audio_test"]["state"] == "off"
        assert (mic.closed, loopback.closed) == (1, 1)
        assert (await control.wait(pending_receipt["operation_id"], timeout=5))[
            "status"
        ] == "cancelled"

        mic = ControlledSource()
        loopback = ControlledSource()
        receipt = await control.submit("audio.test", {"enabled": True}, request_id="ready")
        pending = await control.wait(receipt["operation_id"], timeout=0.05)
        assert pending["terminal"] is False
        assert (await control.query("app.status", {}))["audio_test"]["state"] == "pending"
        mic.frame.set()
        loopback.frame.set()
        ready = await control.wait(receipt["operation_id"], timeout=5)
        assert ready["status"] == "applied"
        for _ in range(100):
            status = (await control.query("app.status", {}))["audio_test"]
            if status["channels"]["peer"]["meter_level"] > 0:
                break
            await asyncio.sleep(0.01)
        assert status["state"] == "ready"
        assert status["effective_active"] is True
        for channel in ("self", "peer"):
            observed = status["channels"][channel]
            assert observed["meter_level"] == pytest.approx(0.75)
            assert observed["speech_probability"] == pytest.approx(0.9)
            assert observed["speech_detected"] is True
            assert observed["speech_threshold"] is not None
        assert status["channels"]["self"]["device"]["name"] == "controlled"
        off = await control.submit("audio.test", {"enabled": False}, request_id="stop")
        stopped = await control.wait(off["operation_id"], timeout=5)
        assert stopped["status"] == "applied"
        assert stopped["audio_test"]["state"] == "off"
        assert (mic.closed, loopback.closed) == (1, 1)

        resolver.available = False
        mic = ControlledSource()
        receipt = await control.submit("audio.test", {"enabled": True}, request_id="degraded")
        mic.frame.set()
        degraded = await control.wait(receipt["operation_id"], timeout=5)
        assert degraded["status"] == "degraded"
        assert degraded["audio_test"]["state"] == "degraded"
        assert degraded["audio_test"]["channels"]["self"]["state"] == "ready"
        end = await control.submit("audio.test", {"enabled": False}, request_id="end")
        assert (await control.wait(end["operation_id"], timeout=5))["status"] == "applied"
    finally:
        await app.stop()


@pytest.mark.asyncio
async def test_auth_challenge_stays_pending_until_owner_finishes_and_cancel_does_not_fake_success(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PURIPULY_HEART_SECRETS_PASSPHRASE", "isolated-test-passphrase")
    monkeypatch.setattr(
        application_runtime, "default_http_extensions_dir", lambda: tmp_path / "extensions"
    )
    path = tmp_path / "settings.json"
    isolated_settings(path)
    app = compose_headless_application(path)
    try:
        await app.start()
        control = app.control()
        control.bind_instance("isolated-auth")
        for outcome in (True, False):
            finish = asyncio.Event()
            entered = asyncio.Event()

            async def authorization(*, authorization_url_sink, open_browser, **_kwargs):
                assert open_browser is False
                authorization_url_sink("https://example.test/authorize?challenge=public")
                entered.set()
                await finish.wait()
                return outcome

            monkeypatch.setattr(app, "start_discord_managed_auth_from_dialog", authorization)
            receipt = await control.submit(
                "auth.login",
                {"provider": "discord", "open_browser": False},
                request_id=f"auth-{outcome}",
            )
            await asyncio.wait_for(entered.wait(), 2)
            pending = await control.operation(receipt["operation_id"])
            assert pending["status"] == "running" and pending["terminal"] is False
            assert (await control.wait(receipt["operation_id"], timeout=0.01))["terminal"] is False
            challenge = (await control.query("auth.status", {}))["challenge"]
            assert challenge["operation_id"] == receipt["operation_id"]
            assert challenge["authorization_url"].startswith("https://example.test/authorize")
            cancellation = await control.cancel(receipt["operation_id"])
            assert cancellation["cancellation"] == "unsupported"
            assert cancellation["terminal"] is False
            finish.set()
            finished = await control.wait(receipt["operation_id"], timeout=2)
            assert finished["status"] == ("applied" if outcome else "failed")
            assert finished["terminal"] is True
            assert (await control.query("auth.status", {}))["challenge"] is None
            assert "authorization_url" not in finished
    finally:
        await app.stop()


@pytest.mark.asyncio
async def test_interleaved_transactions_retain_per_operation_outcome() -> None:
    results = SettingsTransactionResultOwner()
    first_ready = asyncio.Event()
    second_done = asyncio.Event()
    applied = TransactionResult("settings_commit_success_runtime_applied", None, None)
    degraded = TransactionResult("settings_commit_success_runtime_degraded", None, None)

    async def first_operation() -> str:
        with results.capture():
            results.set(applied)
            first_ready.set()
            await second_done.wait()
            return results.current.status

    async def second_operation() -> str:
        await first_ready.wait()
        with results.capture():
            results.set(degraded)
            second_done.set()
            return results.current.status

    assert await asyncio.gather(first_operation(), second_operation()) == [
        applied.status,
        degraded.status,
    ]


@pytest.mark.asyncio
async def test_control_translation_enable_never_starts_implicit_authorization(
    tmp_path, monkeypatch
):
    import webbrowser

    from puripuly_heart.app.wiring import wiring_managed_account

    authorization_attempts = []

    class OfflineBroker:
        def __init__(self, **_kwargs):
            pass

        def __getattr__(self, name):
            async def forbidden_request(*_args, **_kwargs):
                if name == "start_discord_oauth":
                    authorization_attempts.append("broker")
                raise AssertionError("offline test forbids external broker requests")

            return forbidden_request

        async def close(self):
            pass

    def forbidden_authorization(*_args, **_kwargs):
        authorization_attempts.append("browser_or_listener")
        raise AssertionError("control translation enable initiated browser or listener")

    monkeypatch.setenv("PURIPULY_HEART_SECRETS_PASSPHRASE", "isolated-test-passphrase")
    monkeypatch.setattr(
        application_runtime, "default_http_extensions_dir", lambda: tmp_path / "extensions"
    )
    monkeypatch.setattr(wiring_managed_account, "HttpManagedOpenRouterBrokerClient", OfflineBroker)
    monkeypatch.setattr(webbrowser, "open", forbidden_authorization)
    path = tmp_path / "settings.json"
    isolated_settings(path)
    app = compose_headless_application(path)
    try:
        await app.start()
        release = app._managed.managed.release.service
        assert release is not None
        release.discord_oauth_listener_factory = forbidden_authorization
        app.control().bind_instance("offline-control")
        receipt = await app.control().submit(
            "translation.set", {"enabled": True}, request_id="enable-without-auth"
        )
        result = await app.control().wait(receipt["operation_id"], timeout=5)
        assert result["status"] == "action_required"
        assert result["terminal"] is True
        assert (await app.control().query("app.status", {}))["translation_enabled"] is False
        osc_runtime = app._runtime_shutdown.vrc_mic_sync()
        assert osc_runtime is not None
        await osc_runtime.router._application.set_translation(True)
        assert (await app.control().query("app.status", {}))["translation_enabled"] is False
        assert authorization_attempts == []
    finally:
        await app.stop()


@pytest.mark.asyncio
async def test_gui_cli_and_osc_mutations_share_revision_and_preserve_focused_edits(
    tmp_path, monkeypatch
):
    from puripuly_heart.core.managed_openrouter_release import (
        UnavailableManagedOpenRouterReleaseClient,
    )

    from puripuly_heart.app.ports.settings_view import LocaleSettingsIntent
    from puripuly_heart.app.wiring import wiring_managed_account

    monkeypatch.setenv("PURIPULY_HEART_SECRETS_PASSPHRASE", "isolated-test-passphrase")
    monkeypatch.setattr(
        application_runtime, "default_http_extensions_dir", lambda: tmp_path / "extensions"
    )
    monkeypatch.setattr(
        wiring_managed_account,
        "HttpManagedOpenRouterBrokerClient",
        lambda **_kwargs: UnavailableManagedOpenRouterReleaseClient(),
    )
    path = tmp_path / "settings.json"
    isolated_settings(path)
    app = compose_headless_application(path)
    try:
        await app.start()
        control = app.control()
        control.bind_instance("concurrent-control")
        initial = await control.query("settings.current", {})
        osc_runtime = app._runtime_shutdown.vrc_mic_sync()
        assert osc_runtime is not None
        osc = osc_runtime.router._application

        await control._lock.acquire()
        try:
            gui = asyncio.create_task(app.apply_settings_intent(LocaleSettingsIntent("ja")))
            await asyncio.sleep(0)
            cli = await control.submit(
                "settings.apply",
                {"changes": {"locale": "en"}},
                request_id="stale-locale",
                expected_revision=initial["revision"],
            )
            await asyncio.sleep(0)
            osc_mutation = asyncio.create_task(osc.set_secondary_target_language("zh-CN"))
            await asyncio.sleep(0)
            assert not gui.done() and not osc_mutation.done()
        finally:
            control._lock.release()
        await gui
        cli_result = await control.wait(cli["operation_id"], timeout=5)
        await osc_mutation
        final = await control.query("settings.current", {})
        assert cli_result["status"] == "rejected"
        assert cli_result["error"]["code"] == "revision_conflict"
        assert final["settings"]["intent"]["ui"]["locale"] == "ja"
        assert final["settings"]["intent"]["languages"]["secondary_target_language"] == "zh-CN"
        assert final["revision"] > initial["revision"]
    finally:
        await app.stop()


@pytest.mark.asyncio
async def test_overlay_command_waits_for_owned_start_and_distinguishes_intent_from_output(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PURIPULY_HEART_SECRETS_PASSPHRASE", "isolated-test-passphrase")
    monkeypatch.setattr(
        application_runtime, "default_http_extensions_dir", lambda: tmp_path / "extensions"
    )
    path = tmp_path / "settings.json"
    isolated_settings(path)
    app = compose_headless_application(path)
    try:
        await app.start()
        control = app.control()
        control.bind_instance("isolated-overlay")
        owner = app._overlay.overlay
        monkeypatch.setattr(owner, "output_provider", lambda: None)
        gate = asyncio.Event()
        entered = asyncio.Event()
        original_start = OverlayApplicationOwner.run_start
        attempts = []

        async def delayed_start(self, runtime=None):
            if self is owner:
                attempts.append(self.active_target)
                if len(attempts) == 1:
                    await self.handle_start_failure("steamvr_not_running")
                    return
                entered.set()
                await gate.wait()
            return await original_start(self, runtime)

        monkeypatch.setattr(OverlayApplicationOwner, "run_start", delayed_start)
        try:
            receipt = await control.submit(
                "overlay.set", {"enabled": True}, request_id="overlay-on"
            )
            await asyncio.wait_for(entered.wait(), 5)
            assert (await control.wait(receipt["operation_id"], timeout=0.01))["terminal"] is False
            starting = await control.query("overlay.status", {})
            assert starting["output"]["desired_enabled"] is True
            assert starting["output"]["lifecycle"] == "starting"
            assert starting["output"]["presentation_ready"] is False
            assert starting["output"]["fallback_active"] is True
            gate.set()
            result = await control.wait(receipt["operation_id"], timeout=5)
            assert attempts == ["steamvr", "desktop"]
            assert result["status"] == "action_required"
            assert result["reason"] == "output_unavailable"
            assert result["intent"] == {"requested_enabled": True, "desired_enabled": True}
            assert result["output"]["failure_reason"] == "output_unavailable"
            assert result["output"]["effective_target"] is None
            assert result["output"]["presentation_ready"] is False

            off_receipt = await control.submit(
                "overlay.set", {"enabled": False}, request_id="overlay-off"
            )
            off = await control.wait(off_receipt["operation_id"], timeout=5)
            observed = (await control.query("overlay.status", {}))["output"]
            assert off["status"] == "applied"
            assert off["intent"] == {"requested_enabled": False, "desired_enabled": False}
            assert observed["lifecycle"] == "off"
            assert observed["runtime_active"] is False
            assert observed["process_state"] is None
            assert owner.runtime is None or not owner.runtime.has_resources()
        finally:
            gate.set()
    finally:
        await app.stop()


@pytest.mark.asyncio
async def test_peer_terms_are_explicit_and_shared_acceptance_is_isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("PURIPULY_HEART_SECRETS_PASSPHRASE", "isolated-test-passphrase")
    monkeypatch.setattr(
        application_runtime, "default_http_extensions_dir", lambda: tmp_path / "extensions"
    )
    path = tmp_path / "settings.json"
    isolated_settings(path)
    app = compose_headless_application(path)
    try:
        await app.start()
        control = app.control()
        control.bind_instance("isolated-consent")
        from puripuly_heart.ui.i18n import t

        terms = await control.query("consent.peer_translation", {})
        assert terms["accepted"] is False
        assert terms["terms"] == t("peer_translation_eula.body")
        request = await control.submit(
            "capture.set",
            {"channel": "peer", "enabled": True},
            request_id="peer-without-consent",
        )
        denied = await control.wait(request["operation_id"], timeout=5)
        assert denied["status"] == "action_required"
        assert denied["action"] == "accept_peer_translation_terms"
        assert (await control.query("consent.peer_translation", {}))["accepted"] is False

        async def unavailable_peer_capture(_enabled):
            raise PeerCaptureTargetUnavailable("isolated capture unavailable")

        monkeypatch.setattr(app, "set_peer_translation_enabled", unavailable_peer_capture)
        request = await control.submit(
            "capture.set",
            {"channel": "peer", "enabled": True, "accept_terms": True},
            request_id="peer-explicit-consent",
        )
        unavailable = await control.wait(request["operation_id"], timeout=5)
        assert unavailable["status"] == "action_required"
        assert unavailable["error"]["code"] == "capture_target_unavailable"
        assert (await control.query("consent.peer_translation", {}))["accepted"] is True
        assert app.compatibility_settings().state.peer_translation.eula_accepted is True
    finally:
        await app.stop()


@pytest.mark.asyncio
async def test_activation_notice_cli_catalog_apply_and_durable_reload(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("PURIPULY_HEART_SECRETS_PASSPHRASE", "isolated-test-passphrase")
    path = tmp_path / "settings.json"
    isolated_settings(path)
    app = compose_headless_application(path)
    key = "chatbox.activation_notice.enabled"
    try:
        await app.start()
        control = app.control()
        control.bind_instance("activation-notice-control")
        catalog = await control.query("settings.choices", {})
        assert key in control.capabilities()["settings_fields"]
        assert key in catalog["fields"]
        assert catalog["field_types"][key] == "boolean"
        assert catalog["field_schemas"][key] == {"type": "boolean", "free_form": False}
        initial = await control.query("settings.current", {})
        assert initial["settings"]["intent"]["osc"]["activation_notice_enabled"] is True

        submitted = await control.submit(
            "settings.apply",
            {"changes": {key: False}},
            request_id="disable-activation-notice",
        )
        result = await control.wait(submitted["operation_id"], timeout=5)
        assert result["status"] == "applied"
        disabled = await control.query("settings.current", {})
        expected = deepcopy(initial["settings"])
        expected["intent"]["osc"]["activation_notice_enabled"] = False
        assert disabled["settings"] == expected
        for index, invalid in enumerate((None, 0, 1, "false", "true", [], {})):
            submitted = await control.submit(
                "settings.apply",
                {"changes": {key: invalid}},
                request_id=f"invalid-activation-notice-{index}",
            )
            result = await control.wait(submitted["operation_id"], timeout=5)
            assert result["status"] == "rejected"
            assert await control.query("settings.current", {}) == disabled
    finally:
        await app.stop()
    restarted = compose_headless_application(path)
    try:
        await restarted.start()
        control = restarted.control()
        control.bind_instance("activation-notice-restarted")
        assert (await control.query("settings.current", {}))["settings"]["intent"]["osc"][
            "activation_notice_enabled"
        ] is False
    finally:
        await restarted.stop()


@pytest.mark.asyncio
async def test_control_model_commands_download_import_delete_cancel_and_report_assets(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from puripuly_heart.app.services.model_assets import ModelAssetsApplication
    from puripuly_heart.app.services.model_assets_catalog import ModelAssetChoice

    monkeypatch.setenv("PURIPULY_HEART_SECRETS_PASSPHRASE", "isolated-test-passphrase")
    monkeypatch.setattr(
        application_runtime, "default_http_extensions_dir", lambda: tmp_path / "extensions"
    )
    path = tmp_path / "settings.json"
    isolated_settings(path)

    class Provisioning:
        def __init__(self):
            self.requests = []
            self.release = asyncio.Event()
            self.started = asyncio.Event()
            self.cancelled = False
            self.snapshot = SimpleNamespace(activity_for=lambda backend: None)

        async def start_install(self, request, *, on_status=None):
            self.requests.append(request)
            self.started.set()
            await self.release.wait()
            return SimpleNamespace(cancelled=self.cancelled, failed_model_ids=(), failure_codes=())

        async def cancel_install(self, backend, *, origin=None):
            self.cancelled = True
            self.release.set()

        async def delete_model(self, model_id):
            return None

    class Gemma:
        status = "missing"

        def install_status(self, model_id):
            return self.status

        async def install_assets(self, **kwargs):
            self.kwargs = kwargs
            self.status = "ready"

        async def delete_assets(self, model_id):
            self.status = "missing"

    provisioning, gemma = Provisioning(), Gemma()
    app = compose_headless_application(path)
    try:
        await app.start()
        control = app.control()
        control.bind_instance("isolated-models")
        app._provider.model_assets = ModelAssetsApplication(
            provisioning=lambda: provisioning,
            gemma=lambda: gemma,
            locale=lambda: "en",
            model_root=tmp_path / "models",
            choices=(
                ModelAssetChoice("qwen3-asr-0.6b-int8-sherpa", "Qwen", "d", "asr", 1),
                ModelAssetChoice("gemma-x", "Gemma", "d", "translation", 2, True, ("model.gguf",)),
            ),
        )

        status = await control.query("models.status", {})
        assert [item["status"] for item in status["assets"]] == ["missing", "missing"]

        download = await control.submit(
            "models.download",
            {"model_id": "qwen3-asr-0.6b-int8-sherpa", "source": "hf-mirror"},
            request_id="download",
        )
        await provisioning.started.wait()
        assert (await control.query("models.status", {}))["assets"][0]["status"] == "downloading"
        cancel = await control.submit(
            "models.cancel",
            {"model_id": "qwen3-asr-0.6b-int8-sherpa"},
            request_id="cancel",
        )
        assert (await control.wait(cancel["operation_id"], timeout=5))["status"] == "cancelled"
        cancelled = await control.wait(download["operation_id"], timeout=5)
        assert cancelled["status"] == "cancelled"
        assert provisioning.requests[0].preferred_source == "hf_mirror"

        missing = await control.submit(
            "models.import",
            {"model_id": "gemma-x", "directory": str(tmp_path)},
            request_id="import-missing",
        )
        failed = await control.wait(missing["operation_id"], timeout=5)
        assert failed["status"] == "failed" and failed["error"]["code"] == "missing_file"

        (tmp_path / "model.gguf").write_bytes(b"x")
        imported = await control.submit(
            "models.import",
            {"model_id": "gemma-x", "directory": str(tmp_path)},
            request_id="import",
        )
        done = await control.wait(imported["operation_id"], timeout=5)
        assert done["status"] == "applied" and done["asset"]["status"] == "installed"
        assert gemma.kwargs["source"] == "local"

        deleted = await control.submit(
            "models.delete", {"model_id": "gemma-x"}, request_id="delete"
        )
        removed = await control.wait(deleted["operation_id"], timeout=5)
        assert removed["status"] == "applied" and removed["asset"]["status"] == "missing"

        bad = await control.submit(
            "models.download",
            {"model_id": "gemma-x", "source": "mirror"},
            request_id="bad-source",
        )
        rejected = await control.wait(bad["operation_id"], timeout=5)
        assert rejected["status"] != "applied"
        assert rejected["error"]["code"] == "invalid_arguments"
    finally:
        await app.stop()
