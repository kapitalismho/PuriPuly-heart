from __future__ import annotations

import json

import pytest

from puripuly_heart.app.services.agent_chat.context import AgentActivity, collect_agent_context
from puripuly_heart.app.services.agent_chat.execution import project_catalog
from puripuly_heart.app.services.agent_chat.modules import model_context
from puripuly_heart.app.services.agent_chat.tools import LANGUAGE_FIELD_PATHS, SETTINGS_FIELD_PATHS
from puripuly_heart.core.messages import ErrorDiagnostics
from tests.app.test_agent_chat import Records


def _snapshot(*, values=None, choices=None, localize=str):
    paths = {
        **SETTINGS_FIELD_PATHS,
        **{f"languages.{name}": path for name, path in LANGUAGE_FIELD_PATHS.items()},
    }
    snapshot = {
        "observed_at": "2026-11-07T12:00:00+00:00",
        "revision": 17,
        "missing": [],
        "settings": {paths[field]: value for field, value in (values or {}).items()},
    }
    if choices is not None:
        snapshot["settings_catalog"] = project_catalog(
            {
                "fields": [*SETTINGS_FIELD_PATHS, "languages"],
                "choices": choices,
                "field_schemas": {
                    "languages": {
                        "properties": {
                            name: {"empty_allowed": True}
                            for name in ("secondary_target", "peer_source", "peer_target")
                        }
                    }
                },
            },
            localize,
        )
    return snapshot


def _catalog(view, module, field):
    return next(group[field] for group in view["catalogs"][module].values() if field in group)


def _choices(view, module, field):
    if field in view[module].get("options", {}):
        return view[module]["options"][field]
    entry = _catalog(view, module, field)
    while "options_from" in entry:
        entry = _catalog(view, module, entry["options_from"])
    return entry.get("options")


def test_projection_preserves_values_labels_units_and_nested_static_choices():
    translations = {
        "settings.subtab.general": "General",
        "settings.presentation.languages.source": "Source language",
        "settings.description.languages.source": "Language spoken by Self.",
        "settings.presentation.languages.target": "Target language",
        "settings.description.languages.target": "Target language",
        "language.ko": "Korean",
        "language.en": "English",
        "dashboard.language.secondary_target.none": "No secondary language",
        "settings.translation_connection.official_byok": "Own API key",
    }

    def localize(key):
        return translations.get(key, key)

    values = {
        "audio.input_device": "",
        "self_vad.speech_threshold": 0.62,
        "peer_vad.hangover_ms": 240,
        "languages.source": "ko",
        "languages.secondary_target": "",
        "peer_expected_languages": ["ko", "en"],
        "overlay.translation.enabled": False,
        "overlay.background_alpha": 0.6,
        "osc.connection": {"mode": "manual", "send_port": 9000, "receive_port": 9001},
        "translation.model": "gpt_6_luna",
        "translation.connection_history": {"gpt_6_luna": "official_byok"},
        "clipboard.enabled": False,
    }
    snapshot = _snapshot(
        values=values,
        choices={
            "languages": {"codes": ["ko", "en"]},
            "peer_expected_languages": ["ko", "en"],
            "translation.connection": {"gpt_6_luna": ["official_byok"]},
        },
        localize=localize,
    )

    view = model_context(snapshot, localize=localize)

    for field, value in values.items():
        module = next(
            module for module in view["modules"] if field in view[module].get("settings", {})
        )
        assert view[module]["settings"][field] == value
    source = view["catalogs"]["languages"]["General"]["languages.source"]
    assert source["label"] == "Source language"
    assert source["description"] == "Language spoken by Self."
    assert _catalog(view, "languages", "languages.target") == {
        "label": "Target language",
        "options_from": "languages.source",
    }
    assert _choices(view, "languages", "languages.source") == {"ko": "Korean", "en": "English"}
    assert _catalog(view, "languages", "languages.secondary_target")["empty_label"] == (
        "No secondary language"
    )
    assert _catalog(view, "audio", "peer_vad.hangover_ms")["unit"] == "ms"
    assert _choices(view, "translation", "translation.connection") == {
        "gpt_6_luna": {"official_byok": "Own API key"}
    }
    assert view["core"]["revision"] == 17
    assert view["observed_at"] == snapshot["observed_at"]


def test_static_choices_share_module_local_references_and_keep_subset_labels():
    translations = {
        "provider.deepgram": "Deepgram",
        "language.ko": "Korean",
        "language.en": "English",
    }

    def localize(key):
        return translations.get(key, key)

    snapshot = _snapshot(
        choices={
            "stt.provider": ["deepgram", "local_cpu_auto"],
            "peer_stt.provider": ["deepgram", "local_cpu_auto"],
            "stt.cloud_free_tier_providers": ["deepgram"],
            "translation.model": ["gpt_6_luna"],
            "translation.previous_llm_model": ["gpt_6_luna"],
            "translation.connection": {"gpt_6_luna": ["official_byok"]},
            "translation.connection_history": {"gpt_6_luna": ["official_byok"]},
            "languages": {"codes": ["ko", "en"]},
            "peer_expected_languages": ["ko", "en"],
        },
        localize=localize,
    )

    view = model_context(snapshot, localize=localize)

    for module, field, source in (
        ("recognition", "peer_stt.provider", "stt.provider"),
        ("translation", "translation.previous_llm_model", "translation.model"),
        ("translation", "translation.connection_history", "translation.connection"),
        ("languages", "languages.target", "languages.source"),
        ("languages", "peer_expected_languages", "languages.source"),
    ):
        assert _catalog(view, module, field)["options_from"] == source
        assert _choices(view, module, field) == _choices(view, module, source)
    assert _choices(view, "recognition", "stt.cloud_free_tier_providers") == {
        "deepgram": "Deepgram"
    }
    assert _choices(view, "languages", "peer_expected_languages") == {
        "ko": "Korean",
        "en": "English",
    }


def test_private_settings_expose_only_presence_and_never_private_values_or_choices():
    snapshot = _snapshot(values={"translation.model": "gpt_6_luna"})
    snapshot["settings"].update(
        {
            "prompts.system_prompt_override": "PRIVATE-PROMPT",
            "local_llm.base_url": "https://PRIVATE-HOST",
            "stt.custom.endpoint": "https://PRIVATE-ENDPOINT",
        }
    )
    snapshot["local_settings"] = {
        "prompts.system_prompt_override": {"presence": True, "value": "PRIVATE-PROMPT"},
        "local_llm.base_url": {"presence": False, "value": "PRIVATE-URL"},
        "stt.custom.endpoint": {"presence": None, "value": "PRIVATE-ENDPOINT"},
        "stt.custom.model": {"presence": 1},
    }
    snapshot["settings_catalog"] = {
        "prompts.system_prompt_override": {"choices": [{"value": "PRIVATE-PROMPT"}]}
    }
    snapshot["secret_presence"] = {"openai_api_key": True}

    view = model_context(snapshot)

    assert view["translation"]["private"]["prompts.system_prompt_override"] == "saved"
    assert view["translation"]["private"]["local_llm.base_url"] == "empty"
    assert view["recognition"]["private"]["stt.custom.endpoint"] == "unknown"
    assert view["recognition"]["private"]["stt.custom.model"] == "unknown"
    assert view["translation"]["private"]["local_llm.model"] == "unknown"
    assert view["translation"]["keys"]["openai_api_key"] is True
    for module in ("recognition", "translation"):
        assert not view[module]["settings"].keys() & view[module]["private"].keys()
    assert "PRIVATE-" not in json.dumps(view)
    assert _choices(view, "translation", "prompts.system_prompt_override") is None


@pytest.mark.parametrize(
    ("module", "field"),
    (
        ("audio", "audio.input_host_api"),
        ("audio", "audio.input_device"),
        ("audio", "audio.output_device"),
        ("recognition", "stt.gpu_device_id"),
        ("translation", "translation.gpu_device_id"),
        ("translation", "translation.http_extension_id"),
    ),
)
def test_identity_choices_live_in_module_state_even_without_runtime_flags(module, field):
    snapshot = _snapshot()
    snapshot["settings_catalog"] = {
        field: {"choices": [{"value": "identity-1", "label": "First identity"}]}
    }
    first = model_context(snapshot, modules=(module,))
    snapshot["settings_catalog"][field]["choices"].append(
        {"value": "identity-2", "label": "Second identity"}
    )

    second = model_context(snapshot, modules=(module,))

    assert first[module]["options"][field] == {"identity-1": "First identity"}
    assert second[module]["options"][field] == {
        "identity-1": "First identity",
        "identity-2": "Second identity",
    }
    assert not {"options", "options_from"} & _catalog(second, module, field).keys()
    assert first["catalogs"] == second["catalogs"]


def test_device_availability_and_selectability_changes_are_observable_module_state():
    choices = {
        "audio.input_device": [
            {"value": "Desk microphone", "host_api": "MME", "available": True, "selectable": True}
        ]
    }
    first = model_context(
        _snapshot(values={"audio.input_device": "Desk microphone"}, choices=choices), "audio"
    )
    choices["audio.input_device"][0].update(available=False, selectable=False)

    second = model_context(
        _snapshot(values={"audio.input_device": "Desk microphone"}, choices=choices), "audio"
    )

    first_device = first["audio"]["options"]["audio.input_device"][0]
    second_device = second["audio"]["options"]["audio.input_device"][0]
    assert first_device["available"] is True and first_device["selectable"] is True
    assert second_device["available"] is False and second_device["selectable"] is False
    assert second_device["value"] == "Desk microphone"
    assert second_device["host_api"] == "MME"
    assert first["audio"]["settings"] == second["audio"]["settings"]
    assert first["catalogs"] == second["catalogs"]


@pytest.mark.parametrize(
    "runtime",
    (
        {"available": False},
        {"selectable": False},
        {"runtime": {"status": "loading", "progress": 0.5}},
    ),
)
def test_runtime_bearing_choices_are_state_even_for_otherwise_static_fields(runtime):
    snapshot = _snapshot()
    snapshot["settings_catalog"] = {
        "translation.connection": {
            "choices": {
                "gpt_6_luna": [{"value": "official_byok", "label": "Own API key", **runtime}]
            }
        }
    }

    view = model_context(snapshot, "translation")

    choice = view["translation"]["options"]["translation.connection"]["gpt_6_luna"][0]
    assert choice["value"] == "official_byok"
    assert choice["label"] == "Own API key"
    for key, value in runtime.items():
        assert choice[key] == value
    assert (
        not {"options", "options_from"}
        & _catalog(view, "translation", "translation.connection").keys()
    )


def test_static_references_never_target_choices_that_have_become_dynamic_state():
    snapshot = _snapshot()
    snapshot["settings_catalog"] = {
        field: {"choices": [{"value": "deepgram", "label": "Deepgram"}]}
        for field in ("stt.provider", "peer_stt.provider")
    }
    first = model_context(snapshot, "recognition")
    snapshot["settings_catalog"]["stt.provider"]["choices"][0]["available"] = False

    second = model_context(snapshot, "recognition")

    assert _catalog(first, "recognition", "peer_stt.provider")["options_from"] == "stt.provider"
    assert second["recognition"]["options"]["stt.provider"][0]["available"] is False
    assert not {"options", "options_from"} & _catalog(second, "recognition", "stt.provider").keys()
    assert "options_from" not in _catalog(second, "recognition", "peer_stt.provider")
    assert _choices(second, "recognition", "peer_stt.provider") == {"deepgram": "Deepgram"}


def test_locale_and_static_catalog_changes_are_visible_without_a_revision_change():
    locale = "en"
    translations = {
        "en": {
            "settings.subtab.general": "General",
            "settings.presentation.languages.source": "Source",
            "settings.description.languages.source": "Source language",
            "language.ko": "Korean",
        },
        "fr": {
            "settings.subtab.general": "Général",
            "settings.presentation.languages.source": "Source française",
            "settings.description.languages.source": "Langue source",
            "language.ko": "Coréen",
        },
    }

    def localize(key):
        return translations[locale].get(key, key)

    choices = {"languages": {"codes": ["ko"]}}
    snapshot = _snapshot(values={"languages.source": "ko"}, choices=choices, localize=localize)
    first = model_context(snapshot, "languages", localize)
    locale = "fr"
    snapshot["settings_catalog"] = _snapshot(choices=choices, localize=localize)["settings_catalog"]

    second = model_context(snapshot, "languages", localize)
    choices["languages"]["codes"].append("en")
    snapshot["settings_catalog"] = _snapshot(choices=choices, localize=localize)["settings_catalog"]
    third = model_context(snapshot, "languages", localize)

    assert first["catalogs"]["languages"]["General"]["languages.source"]["label"] == "Source"
    translated = second["catalogs"]["languages"]["Général"]["languages.source"]
    assert translated["label"] == "Source française"
    assert translated["description"] == "Langue source"
    assert _choices(first, "languages", "languages.source")["ko"] == "Korean"
    assert _choices(second, "languages", "languages.source") == {"ko": "Coréen"}
    assert "en" in _choices(third, "languages", "languages.source")
    assert (
        first["languages"]["settings"]
        == second["languages"]["settings"]
        == third["languages"]["settings"]
    )
    assert first["core"]["revision"] == second["core"]["revision"] == third["core"]["revision"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failed",
    ({"settings.current"}, {"settings.choices"}, {"settings.current", "settings.choices"}),
)
async def test_failed_queries_are_not_empty_values_or_options_and_omitted_modules_are_unobserved(
    failed,
):
    records = Records()
    records.fail.update(failed)
    snapshot = await collect_agent_context(records, activity=AgentActivity(), topic="languages")

    view = model_context(snapshot, "languages")

    assert set(view["missing"]) == failed
    assert "languages" in view["modules"]
    assert "audio" not in view["modules"] and "audio" not in view
    assert "audio" not in view["catalogs"]
    assert "audio.devices" not in view["missing"]
    if "settings.current" in failed:
        assert "languages.source" not in view["languages"]["settings"]
        assert "languages.secondary_target" not in view["languages"]["settings"]
    else:
        assert view["languages"]["settings"]["languages.source"] == "ko"
        assert view["languages"]["settings"]["languages.secondary_target"] == ""
    if "settings.choices" in failed:
        assert _choices(view, "languages", "languages.source") is None
        assert "empty_label" not in _catalog(view, "languages", "languages.secondary_target")
    else:
        assert {"ko", "en"} <= _choices(view, "languages", "languages.source").keys()


def test_observed_empty_choices_remain_distinct_from_a_failed_choices_query():
    snapshot = _snapshot(choices={"translation.model": [], "translation.http_extension_id": []})
    observed = model_context(snapshot, "translation")
    snapshot["missing"] = ["settings.choices"]

    failed = model_context(snapshot, "translation")

    assert _choices(observed, "translation", "translation.model") == {}
    assert observed["translation"]["options"]["translation.http_extension_id"] == {}
    assert _choices(failed, "translation", "translation.model") is None
    assert "translation.http_extension_id" not in failed["translation"].get("options", {})
    assert failed["missing"] == ["settings.choices"]


@pytest.mark.asyncio
async def test_filtered_diagnostics_are_a_marked_query_view_not_an_ordinary_core_baseline():
    records = Records()
    operation_id = "b" * 32
    retained = []
    for index in range(3):
        diagnostic, _ = records.history.retain(
            ErrorDiagnostics(
                component="translation_request",
                operation="translate",
                code="provider_receive_failed",
                category="provider",
                visibility="basic",
                content_policy="metadata_only",
                operation_id=operation_id if index != 1 else "c" * 32,
                status_code=401,
                retry_after_ms=None,
                fields={},
            ),
            severity="error",
            disposition="failed",
        )
        retained.append(diagnostic)
    baseline = model_context(
        await collect_agent_context(records, activity=AgentActivity(), topic="output"), "output"
    )
    cursor = retained[0]["diagnostic_sequence"]
    filtered = model_context(
        await collect_agent_context(
            records,
            activity=AgentActivity(),
            topic="diagnostics",
            operation_id=operation_id,
            after=cursor,
        ),
        "diagnostics",
    )

    assert "view" not in baseline
    assert filtered["view"] == "diagnostics"
    assert filtered["modules"] == []
    assert not {"core", "output", "catalogs"} & filtered.keys()
    assert {item["diagnostic_id"] for item in baseline["core"]["diagnostics"]["records"]} == {
        item["diagnostic_id"] for item in retained
    }
    assert [item["diagnostic_id"] for item in filtered["diagnostics"]["records"]] == [
        retained[2]["diagnostic_id"]
    ]
    assert filtered["diagnostics"]["coverage"]["filters"] == {
        "operation_id": operation_id,
        "after": cursor,
    }
