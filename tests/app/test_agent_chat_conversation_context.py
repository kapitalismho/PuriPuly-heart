from __future__ import annotations

import json
from copy import deepcopy

import pytest

from puripuly_heart.app.services.agent_chat.conversation import AgentConversation


def canonical_context(*, modules=("core", "audio", "accounts"), observed_at="first"):
    state = {
        "core": {
            "revision": 7,
            "status": {"running": False},
            "diagnostics": {"records": [], "coverage": {"complete": True}},
        },
        "audio": {
            "settings": {"audio.input_device": "default", "audio.input_host_api": None},
            "options": {
                "audio.input_device": [{"value": "default", "label": "Default microphone"}]
            },
            "devices": {"inputs": ["default"]},
        },
        "accounts": {"puri": {"authenticated": True, "allowance": {"remaining": 12}}},
    }
    context = {
        "observed_at": observed_at,
        "modules": list(modules),
        "missing": [],
        **{module: deepcopy(state[module]) for module in modules},
    }
    if "audio" in modules:
        context["catalogs"] = {
            "audio": {
                "音频": {
                    "audio.input_device": {
                        "label": "Microphone",
                        "description": "Select an input device",
                        "empty_label": "System default",
                        "options_from": "audio.input_device",
                    },
                    "audio.input_host_api": {
                        "label": "Host API",
                        "unit": "backend",
                        "options": [{"value": None, "label": "Automatic"}],
                    },
                }
            }
        }
    return context


def context_item(context, *, receipt=False):
    if receipt:
        return {
            "type": "function_call_output",
            "call_id": "context-read",
            "output": json.dumps({"context": context}, ensure_ascii=False),
        }
    return {
        "role": "developer",
        "content": json.dumps({"app_context": context}, ensure_ascii=False),
    }


def transmit(conversation, context, *, force_modules=(), receipt=False, archive=True):
    projected = conversation.project_context(context, force_modules=force_modules)
    conversation.append(
        context_item(projected, receipt=receipt), archive=archive, context=projected
    )
    return projected


def reconstruct(conversation):
    result = {"state": {}, "catalogs": {}, "missing": []}
    for item in conversation.items:
        content = item.get("content", "")
        if isinstance(content, str) and content.startswith('{"app_context":'):
            context = json.loads(content)["app_context"]
        elif item.get("type") == "function_call_output":
            context = json.loads(item["output"]).get("context")
            if context is None:
                continue
        else:
            continue
        if context.get("view") == "diagnostics":
            result["diagnostics"] = context
            continue
        result["observed_at"] = context["observed_at"]
        result["missing"] = context["missing"]
        if context["missing"]:
            for module in context["modules"]:
                result["state"].pop(module, None)
                result["catalogs"].pop(module, None)
        result["catalogs"].update(deepcopy(context.get("catalogs", {})))
        for module in context["modules"]:
            if module == "core" and module in context:
                result["state"]["core"] = {
                    **result["state"].get("core", {}),
                    **deepcopy(context["core"]),
                }
            elif module in context:
                result["state"][module] = deepcopy(context[module])
    return result


def effective(state):
    return {
        module: (
            {key: item for key, item in value.items() if item is not None}
            if module == "core"
            else value
        )
        for module, value in state.items()
    }


def assert_reconstructed(conversation, context):
    restored = reconstruct(conversation)
    assert effective(restored["state"]) == effective(
        {module: context[module] for module in context["modules"]}
    )
    assert restored["catalogs"] == context.get("catalogs", {})
    assert restored["missing"] == context["missing"]
    assert restored["observed_at"] == context["observed_at"]


def test_projection_is_read_only_until_the_exact_envelope_is_appended():
    conversation = AgentConversation()
    context = canonical_context()
    original = deepcopy(context)

    first = conversation.project_context(context)
    assert first == context
    assert conversation.project_context(context) == context
    assert not conversation.items and not conversation.archive

    first["audio"]["settings"]["audio.input_device"] = "another"
    first["catalogs"]["audio"]["音频"]["audio.input_device"]["label"] = "Changed"
    first["modules"].clear()
    assert context == original
    assert conversation.project_context(context) == original

    conversation.append(context_item(context), context=context)
    context["audio"]["devices"]["inputs"].append("mutated caller input")
    context["catalogs"]["audio"]["音频"].clear()
    assert conversation.project_context(original)["unchanged"] == original["modules"]
    assert "catalogs" not in conversation.project_context(original)
    assert_reconstructed(conversation, original)


@pytest.mark.parametrize("receipt", [False, True])
def test_unchanged_turns_preserve_fresh_observations_without_rewriting_history(receipt):
    conversation = AgentConversation()
    context = canonical_context()
    assert transmit(conversation, context, receipt=receipt) == context
    original_items = deepcopy(conversation.items)
    conversation.begin()
    fresh = canonical_context(observed_at="second")

    projected = transmit(conversation, fresh, receipt=receipt)

    assert projected == {
        "observed_at": "second",
        "modules": fresh["modules"],
        "missing": [],
        "unchanged": fresh["modules"],
    }
    assert conversation.items[: len(original_items)] == original_items
    assert len(conversation.items) == 2
    assert_reconstructed(conversation, fresh)


def test_catalog_and_complete_module_changes_are_independent():
    conversation = AgentConversation()
    context = canonical_context()
    transmit(conversation, context)
    state_changed = deepcopy(context)
    state_changed["observed_at"] = "state changed"
    state_changed["audio"]["settings"]["audio.input_device"] = "usb"
    state_changed["audio"]["options"]["audio.input_device"].append(
        {"value": "usb", "label": "USB microphone"}
    )
    state_changed["audio"]["devices"]["inputs"].append("usb")
    state_changed["audio"]["private"] = {"audio.input_device": "unknown"}

    state_projection = transmit(conversation, state_changed, receipt=True)

    assert state_projection["audio"] == state_changed["audio"]
    assert state_projection["unchanged"] == ["core", "accounts"]
    assert "catalogs" not in state_projection
    assert_reconstructed(conversation, state_changed)

    catalog_changed = deepcopy(state_changed)
    catalog_changed["observed_at"] = "catalog changed"
    section = catalog_changed["catalogs"]["audio"]["音频"]
    section["audio.input_device"]["label"] = "Input microphone"
    del section["audio.input_host_api"]
    catalog_projection = transmit(conversation, catalog_changed)

    assert catalog_projection["catalogs"] == catalog_changed["catalogs"]
    assert catalog_projection["unchanged"] == catalog_changed["modules"]
    assert "audio" not in catalog_projection
    assert_reconstructed(conversation, catalog_changed)

    both_changed = deepcopy(catalog_changed)
    both_changed["audio"]["devices"]["inputs"] = []
    both_changed["catalogs"]["audio"] = {}
    both_projection = transmit(conversation, both_changed)
    assert both_projection["audio"] == both_changed["audio"]
    assert both_projection["catalogs"] == {"audio": {}}
    assert_reconstructed(conversation, both_changed)


def test_core_sends_only_changed_fields_and_nulls_removed_ones():
    conversation = AgentConversation()
    original = canonical_context()
    transmit(conversation, original)
    changed = deepcopy(original)
    changed["observed_at"] = "second"
    changed["core"]["status"]["running"] = True
    del changed["core"]["diagnostics"]

    projected = transmit(conversation, changed)

    assert projected["core"] == {"status": {"running": True}, "diagnostics": None}
    assert projected["unchanged"] == ["audio", "accounts"]
    assert_reconstructed(conversation, changed)
    assert conversation.project_context(changed)["unchanged"] == changed["modules"]
    changed["core"]["diagnostics"] = {"records": [], "coverage": {"complete": False}}
    assert conversation.project_context(changed)["core"] == {
        "diagnostics": changed["core"]["diagnostics"]
    }


def test_trim_resends_only_core_fields_whose_anchor_was_trimmed():
    conversation = AgentConversation(token_limit=0)
    context = canonical_context()
    transmit(conversation, context)
    conversation.begin()
    context["core"]["status"]["running"] = True
    assert transmit(conversation, context)["core"] == {"status": {"running": True}}

    conversation.trim()

    projected = conversation.project_context(context)
    assert projected["core"] == {
        "revision": context["core"]["revision"],
        "diagnostics": context["core"]["diagnostics"],
    }
    transmit(conversation, context)
    assert_reconstructed(conversation, context)


def test_trim_cuts_history_deeply_once_over_the_limit_so_later_turns_keep_the_prefix():
    conversation = AgentConversation(token_limit=100, retained_tokens=50)
    trimmed_at = []
    for turn in range(1, 9):
        before = deepcopy(conversation.items)
        conversation.begin()
        conversation.append({"role": "user", "content": str(turn) * 69})
        conversation.trim()
        if conversation.items[: len(before)] != before:
            trimmed_at.append(turn)
            assert sum(len(json.dumps(item)) for item in conversation.items) <= 200

    assert trimmed_at == [5, 8]
    assert [item["content"][0] for item in conversation.items] == ["7", "8"]


def test_module_replacement_preserves_nulls_and_removes_deleted_properties():
    conversation = AgentConversation()
    original = canonical_context()
    original["audio"]["private"] = {"audio.input_device": "saved"}
    transmit(conversation, original)
    changed = deepcopy(original)
    changed["audio"]["settings"]["audio.input_device"] = None
    del changed["audio"]["settings"]["audio.input_host_api"]
    del changed["audio"]["options"]
    del changed["audio"]["private"]
    del changed["audio"]["devices"]

    projected = transmit(conversation, changed)

    assert projected["audio"] == {"settings": {"audio.input_device": None}}
    assert_reconstructed(conversation, changed)
    assert conversation.project_context(changed)["unchanged"] == changed["modules"]


def test_modules_outside_an_observation_do_not_lose_their_baselines():
    conversation = AgentConversation()
    complete = canonical_context()
    transmit(conversation, complete)
    accounts = canonical_context(modules=("core", "accounts"), observed_at="accounts only")
    accounts["accounts"]["puri"]["allowance"]["remaining"] = 11

    projected = transmit(conversation, accounts)

    assert projected["modules"] == ["core", "accounts"]
    assert projected["unchanged"] == ["core"]
    assert "audio" not in projected and "catalogs" not in projected
    complete["observed_at"] = "overview again"
    complete["accounts"] = deepcopy(accounts["accounts"])
    resumed = transmit(conversation, complete)
    assert resumed["unchanged"] == complete["modules"]
    assert "catalogs" not in resumed
    assert_reconstructed(conversation, complete)


def test_missing_projection_is_full_and_only_invalidates_when_appended():
    conversation = AgentConversation()
    complete = canonical_context()
    transmit(conversation, complete)
    missing = canonical_context(modules=("core", "audio"), observed_at="missing")
    missing["missing"] = ["settings.current"]
    missing["audio"]["settings"] = {}
    missing["catalogs"]["audio"] = {}

    assert conversation.project_context(missing) == missing
    assert "unchanged" not in conversation.project_context(missing)
    assert conversation.project_context(complete)["unchanged"] == complete["modules"]

    transmit(conversation, missing, receipt=True)
    recovered = canonical_context(observed_at="recovered")
    projection = conversation.project_context(recovered)
    assert projection["core"] == recovered["core"]
    assert projection["audio"] == recovered["audio"]
    assert projection["catalogs"] == recovered["catalogs"]
    assert projection["unchanged"] == ["accounts"]
    transmit(conversation, recovered)
    assert_reconstructed(conversation, recovered)


def test_partial_payloads_from_missing_queries_never_become_reusable_baselines():
    conversation = AgentConversation()
    missing = canonical_context(modules=("core", "audio"))
    missing["missing"] = ["settings.choices", "audio.devices"]
    missing["audio"] = {"settings": {"audio.input_device": None}}
    missing["catalogs"]["audio"] = {}
    transmit(conversation, missing)
    assert conversation.project_context(missing) == missing

    recovered = deepcopy(missing)
    recovered["missing"] = []
    assert conversation.project_context(recovered) == recovered
    transmit(conversation, recovered)
    assert_reconstructed(conversation, recovered)


def test_diagnostics_are_complete_query_scoped_views_without_overview_cache_updates():
    conversation = AgentConversation()
    overview = canonical_context()
    transmit(conversation, overview)
    diagnostic = canonical_context(modules=("core",), observed_at="filtered")
    diagnostic["view"] = "diagnostics"
    diagnostic["core"]["diagnostics"] = {
        "records": [{"operation_id": "operation-one", "sequence": 9}],
        "filters": {"operation_id": "operation-one"},
    }

    assert transmit(conversation, diagnostic, receipt=True) == diagnostic
    assert transmit(conversation, diagnostic) == diagnostic
    assert "unchanged" not in conversation.project_context(diagnostic)
    assert conversation.project_context(overview)["unchanged"] == overview["modules"]
    assert_reconstructed(conversation, overview)
    assert reconstruct(conversation)["diagnostics"] == diagnostic

    next_filter = deepcopy(diagnostic)
    next_filter["core"]["diagnostics"]["filters"] = {"after": 9}
    next_filter["core"]["diagnostics"]["records"] = []
    assert transmit(conversation, next_filter) == next_filter
    assert conversation.project_context(overview)["unchanged"] == overview["modules"]


def test_diagnostics_without_an_overview_do_not_seed_ordinary_core_state():
    conversation = AgentConversation()
    diagnostic = canonical_context(modules=("core",))
    diagnostic["view"] = "diagnostics"
    transmit(conversation, diagnostic)
    overview = canonical_context(modules=("core",))
    assert conversation.project_context(overview) == overview


def test_forcing_a_module_reemits_its_complete_state_but_not_unchanged_catalogs():
    conversation = AgentConversation()
    context = canonical_context()
    transmit(conversation, context)
    forced = transmit(conversation, context, force_modules=("accounts", "not-observed"))

    assert forced["accounts"] == context["accounts"]
    assert forced["unchanged"] == ["core", "audio"]
    assert "catalogs" not in forced
    assert_reconstructed(conversation, context)
    assert conversation.project_context(context)["unchanged"] == context["modules"]


def test_trim_invalidates_original_anchors_despite_later_unchanged_receipts():
    conversation = AgentConversation(token_limit=0)
    context = canonical_context()
    first = transmit(conversation, context)
    old_message = conversation.display("user", "old turn")
    conversation.begin()
    conversation.append({"role": "user", "content": "current turn"})
    unchanged = transmit(conversation, context, receipt=True)
    reasoning = {"type": "reasoning", "encrypted_content": "native-reasoning"}
    call = {"type": "function_call", "call_id": "native", "name": "inspect"}
    output = {
        "type": "function_call_output",
        "call_id": "native",
        "output": json.dumps({"status": "completed"}),
    }
    for item in (reasoning, call, output):
        conversation.append(item)
    retained = deepcopy(conversation.items[1:])
    assert unchanged["unchanged"] == context["modules"]

    conversation.trim()

    assert conversation.items == retained
    assert conversation.messages[old_message].archived
    assert context_item(first) in conversation.archive
    assert reasoning not in conversation.archive
    assert conversation.project_context(context) == context
    transmit(conversation, context)
    assert_reconstructed(conversation, context)


@pytest.mark.parametrize("new_anchor", ["state", "catalog"])
def test_trim_keeps_independent_newer_state_or_catalog_anchors(new_anchor):
    conversation = AgentConversation(token_limit=0)
    context = canonical_context()
    transmit(conversation, context)
    conversation.begin()
    if new_anchor == "state":
        staged = transmit(conversation, context, force_modules=tuple(context["modules"]))
        assert "catalogs" not in staged
    else:
        context["catalogs"]["audio"]["音频"]["audio.input_device"]["label"] = "New label"
        staged = transmit(conversation, context)
        assert staged["unchanged"] == context["modules"]

    conversation.trim()
    projected = conversation.project_context(context)

    if new_anchor == "state":
        assert projected["unchanged"] == context["modules"]
        assert projected["catalogs"] == context["catalogs"]
        assert "audio" not in projected
    else:
        assert "unchanged" not in projected
        assert projected["audio"] == context["audio"]
        assert "catalogs" not in projected
    transmit(conversation, context)
    assert_reconstructed(conversation, context)


def test_continuation_retains_the_payload_anchor_in_the_same_group():
    conversation = AgentConversation(token_limit=0)
    context = canonical_context()
    transmit(conversation, context)
    conversation.begin(continuation=True)
    transmit(conversation, context, receipt=True)
    retained = deepcopy(conversation.items)

    conversation.trim()

    assert conversation.items == retained
    assert conversation.project_context(context)["unchanged"] == context["modules"]
    assert_reconstructed(conversation, context)


def test_reset_clears_all_baselines_without_erasing_the_archive():
    conversation = AgentConversation()
    context = canonical_context()
    transmit(conversation, context)
    index = conversation.display("user", "before reset")
    old_archive = deepcopy(conversation.archive)

    conversation.reset("New conversation")

    assert not conversation.items
    assert conversation.messages[index].archived
    assert conversation.messages[-1].text == "New conversation"
    assert conversation.archive[:-1] == old_archive
    assert conversation.project_context(context) == context
    transmit(conversation, context)
    assert_reconstructed(conversation, context)


@pytest.mark.parametrize("receipt", [False, True])
def test_append_and_record_without_explicit_context_never_infer_a_baseline(receipt):
    conversation = AgentConversation()
    context = canonical_context()
    item = context_item(context, receipt=receipt)
    conversation.record(item)
    conversation.append(item)

    assert conversation.project_context(context) == context
    assert conversation.items == [item]
    assert conversation.archive == [item, item]


def test_nonarchived_context_is_still_anchored_to_retained_model_history():
    conversation = AgentConversation()
    context = canonical_context()
    transmit(conversation, context, archive=False)

    assert not conversation.archive
    assert conversation.project_context(context)["unchanged"] == context["modules"]
    assert_reconstructed(conversation, context)
