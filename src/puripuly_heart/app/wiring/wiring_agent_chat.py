from __future__ import annotations

import os
from pathlib import Path
from typing import Final

from puripuly_heart import __version__
from puripuly_heart.app.adapters.agent_relay_auth import AgentRelayTokenOwner
from puripuly_heart.app.adapters.agent_relay_transport import AgentRelayTransport
from puripuly_heart.app.services.canonical_settings_persistence import SettingsOwner
from puripuly_heart.app.wiring.wiring_managed_auth_factory import (
    managed_identity_state_port_from_vnext,
)
from puripuly_heart.app.wiring.wiring_secrets_factory import create_secret_store
from puripuly_heart.core.managed_identity import (
    ManagedIdentityBundle,
    ensure_managed_identity_bundle,
)
from puripuly_heart.core.owned_thread import run_owned_thread_call
from puripuly_heart.runtime_layout import current_runtime_layout

AGENT_RELAY_URL: Final = "https://puripuly-heart-agent-relay.kapitalismho.workers.dev/v1/respond"


def compose_agent_transport(
    *, settings: SettingsOwner, config_path: Path
) -> AgentRelayTransport | None:
    source = current_runtime_layout().host_kind == "source"
    endpoint = os.environ.get("PURIPULY_AGENT_RELAY_URL", AGENT_RELAY_URL)
    if not endpoint:
        return None
    broker_override = os.environ.get("PURIPULY_AGENT_BROKER_URL") or None

    def broker_url() -> str:
        return (
            broker_override
            or settings.require_canonical().intent.translation.openrouter_broker_base_url
        )

    def identity_bundle() -> ManagedIdentityBundle:
        current = settings.require_canonical()
        state = managed_identity_state_port_from_vnext(settings, current)
        secrets = create_secret_store(current.intent.secrets, config_path=config_path)
        return ensure_managed_identity_bundle(state, secrets, require_empty_installation_id=True)

    async def identity() -> ManagedIdentityBundle:
        return await run_owned_thread_call(identity_bundle)

    token_owner = AgentRelayTokenOwner(
        broker_url, identity, app_version=__version__, allow_loopback_http=source
    )
    return AgentRelayTransport(endpoint, token_owner, allow_loopback_http=source)
