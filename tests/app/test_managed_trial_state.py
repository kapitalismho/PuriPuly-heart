from __future__ import annotations

import pytest

from puripuly_heart.app.services.agent_chat.modules import model_context, topic_queries
from puripuly_heart.app.services.agent_chat.tools import project_accounts
from puripuly_heart.app.wiring.wiring_managed_auth_factory import _managed_trial
from puripuly_heart.config.settings_vnext.schema import ManagedConnectionState

PENDING = ManagedConnectionState(
    pending_managed_operation_id="ph-mop-v1_synthetic",
    pending_managed_operation_source="discord",
    pending_managed_operation_installation_id="installation",
)
CLAIMED = ManagedConnectionState(local_managed_claim_sources=("discord",))


@pytest.mark.parametrize(
    ("state", "selected", "key_available", "expected"),
    [
        (ManagedConnectionState(), True, False, "not_claimed"),
        (ManagedConnectionState(), False, False, "not_claimed"),
        (ManagedConnectionState(), True, None, "unknown"),
        (ManagedConnectionState(), False, None, "unknown"),
        (ManagedConnectionState(), False, True, "claimed"),
        (PENDING, True, False, "sign_in_pending"),
        (CLAIMED, True, True, "active"),
        (CLAIMED, True, False, "claimed_without_key"),
        (CLAIMED, False, False, "claimed"),
    ],
)
def test_trial_state_follows_local_managed_evidence(state, selected, key_available, expected):
    assert (
        _managed_trial(
            state,
            managed_china=False,
            managed_selected=selected,
            key_available=key_available,
        )
        == expected
    )


@pytest.mark.parametrize(("claimed", "managed_china"), [("discord", True), ("qq", False)])
def test_trial_is_blocked_after_a_claim_through_the_other_sign_in(claimed, managed_china):
    managed = ManagedConnectionState(local_managed_claim_sources=(claimed,))
    assert (
        _managed_trial(
            managed, managed_china=managed_china, managed_selected=True, key_available=None
        )
        == "blocked_by_other_sign_in"
    )


@pytest.mark.parametrize(
    ("trial", "expected"),
    [
        ("not_claimed", "not_claimed"),
        ("exhausted", "exhausted"),
        ("blocked_by_other_sign_in", "blocked_by_other_sign_in"),
        ("expired_soon", "unknown"),
    ],
)
def test_puri_accounts_context_carries_only_known_trial_states(trial, expected):
    projected = project_accounts({"managed": {"provider": "discord", "trial": trial}})
    assert projected["managed"]["trial"] == expected


@pytest.mark.parametrize("selected", [True, False])
def test_translation_context_carries_trial_only_for_selected_managed_connection(selected):
    accounts = project_accounts(
        {
            "managed": {
                "provider": "discord",
                "available": True,
                "selected": selected,
                "credential_present": False,
                "trial": "claimed_without_key",
            }
        }
    )

    context = model_context({"accounts": accounts}, modules=("translation",))

    assert "accounts" not in context
    if selected:
        assert context["translation"]["managed"]["trial"] == "claimed_without_key"
        assert context["translation"]["managed"]["credential_present"] is False
    else:
        assert "managed" not in context["translation"]


def test_translation_topic_reads_account_status():
    assert "accounts.status" in topic_queries("translation")
