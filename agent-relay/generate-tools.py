from __future__ import annotations

import argparse
import json
from pathlib import Path

from puripuly_heart.app.services.agent_chat.context import TOPICS
from puripuly_heart.app.services.agent_chat.execution import app_tool_schemas, request_schemas
from puripuly_heart.app.services.agent_chat.provider_setup import NAVIGATION_PROVIDERS
from puripuly_heart.app.services.agent_chat.tools import settings_tool_schema


def _openai_schema(value: object) -> object:
    if isinstance(value, dict):
        return {key: _openai_schema(item) for key, item in value.items() if key != "uniqueItems"}
    if isinstance(value, list):
        return [_openai_schema(item) for item in value]
    return value


def unified_tools(catalog: dict) -> list[dict]:
    schemas = app_tool_schemas(catalog)
    proposal = schemas["propose_app"]
    changes = proposal["properties"]["request"]["anyOf"][0]["properties"]["changes"]
    groups = {}
    for field, schema in changes["properties"].items():
        groups.setdefault(json.dumps(schema, sort_keys=True), []).append(field)
    definitions = {}
    for fields in groups.values():
        if len(fields) < 2:
            continue
        identity = f"value{len(definitions) + 1}"
        definitions[identity] = changes["properties"][fields[0]]
        for field in fields:
            changes["properties"][field] = {"$ref": f"#/$defs/{identity}"}
    if definitions:
        proposal["$defs"] = definitions
    descriptions = {
        "read_app": (
            "Observe app context, allowlisted queries or provider setup metadata without UI or runtime effects. "
            "Independent reads may share a batch; results are not an atomic snapshot. "
            "Use observed diagnostic IDs/cursors for retained details; not_retained is not an operation failure."
        ),
        "execute_app": (
            "Run listed immediate operations or explicitly stage existing UI controls. "
            "ui.provider_setup returns metadata and stages links/secure entry; ui.accounts stages account buttons. "
            "Staged is not displayed, clicked, verified or ready. Only users enter credentials and authorize accounts. "
            "Probes are active, scoped synthetic checks, not proof of live capture, quality or display. "
            "audio.diagnose exercises local capture/VAD, not recognition, and never changes settings. "
            "OSC target application does not prove remote receipt. Inspect terminal receipts; do not replay uncertain work. "
            "confirmation.withdraw removes only the exact pending proposal, never saved changes."
        ),
        "propose_app": (
            "Propose sparse typed settings, key deletion, logout, position reset or a support report. "
            "Settings require revision-bound user approval; supply only changed fields, never null placeholders. "
            "Submit translation model and connection together. Coupled audio routes retain validation. "
            "The app queues cards with at most two blocks; explain and end the turn without resubmitting. "
            "Key deletion is local, not provider revocation. Reports upload only after the user's explicit click. "
            "Approval, required user action, pending work or blocking failure stops later effects."
        ),
    }
    return [
        {
            "type": "function",
            "name": name,
            "description": descriptions[name],
            "strict": False,
            "parameters": _openai_schema(schema),
        }
        for name, schema in schemas.items()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("choices", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--protocol", type=int, choices=(5, 6), default=6)
    args = parser.parse_args()
    catalog = json.loads(args.choices.read_text(encoding="utf-8"))
    if args.protocol == 6:
        output = args.output or Path("agent-relay/src/app-tools.ts")
        output.write_text(
            "export const appTools = "
            + json.dumps(unified_tools(catalog), ensure_ascii=False, indent=2)
            + ";\n",
            encoding="utf-8",
        )
        return
    args.output = args.output or Path("agent-relay/src/action-tools.ts")
    tools = [
        {
            "type": "function",
            "name": "get_app_context",
            "description": (
                "Read app_context modules omitted from the initial context or refresh stale facts. "
                "A read returns core plus the same-name topic module; overview returns all. "
                "An omitted module is not an absent or disabled feature. "
                "An explicit accounts read also stages user-click account actions; "
                "initial facts alone do not show controls. No identities, secrets or live balance "
                "refresh. Diagnostics accepts observed diagnostic_id/operation_id or next_after, "
                "returning at most 10 records in core.diagnostics; missing/evicted records do not "
                "prove recovery. Other topics require null filters. No raw logs or runtime changes."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "enum": sorted(TOPICS)},
                    "diagnostic_id": {"type": ["string", "null"], "minLength": 1, "maxLength": 64},
                    "operation_id": {"type": ["string", "null"], "minLength": 1, "maxLength": 64},
                    "after": {"type": ["integer", "null"], "minimum": 0},
                },
                "required": ["topic", "diagnostic_id", "operation_id", "after"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "change_settings",
            "description": (
                "Propose supported settings changes within the user's request. Every change "
                "requires a user-approved card; this tool never applies settings directly. "
                "Null means leave unchanged. Use actual settings choices; submit translation "
                "model and connection together. The app queues cards with at most two setting "
                "blocks, keeping coupled settings together, and returns needs_confirmation. "
                "Explain the proposal and end the turn; do not resubmit queued changes. "
                "Each user approval applies only that card before the next appears. "
                "Do not retry a revision conflict without reconsidering current state."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {"changes": _openai_schema(settings_tool_schema(catalog))},
                "required": ["changes"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "control_app",
            "description": (
                "Read current-instance status or run a few non-settings app actions. "
                "query.* returns projected metadata only, never private settings, raw content "
                "or secret values. overlay.set, audio.retry and models.retry run immediately; "
                "inspect the terminal receipt. secrets.delete, auth.logout and "
                "overlay.position.reset only create a user-approval card like change_settings; "
                "explain the effect and end the turn. Deleting a stored key does not revoke it "
                "at the provider. Model downloads or imports, key entry, account login, manual "
                "text, restart and overlay calibration are not available here; point the user "
                "to the existing app screen."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {"request": {"anyOf": _openai_schema(request_schemas(catalog))}},
                "required": ["request"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "withdraw_confirmation",
            "description": (
                "Withdraw the exact current pending settings card identified by "
                "app_context.core.pending_confirmation.id and its remaining unapproved cards when "
                "the user cancels that proposal. A question about the card or an unrelated "
                "message does not withdraw it. Stale or unknown IDs are rejected. This never "
                "approves, applies or undoes settings, and creates no app-notice continuation."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {"card_id": {"type": "string", "minLength": 1}},
                "required": ["card_id"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "set_capture",
            "description": (
                "Set Self or Peer capture through the existing application owner. "
                "This cannot accept Peer terms; action_required means the user must "
                "complete that existing consent step. Inspect the terminal receipt."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "channel": {"type": "string", "enum": ["self", "peer"]},
                    "enabled": {"type": "boolean"},
                },
                "required": ["channel", "enabled"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "set_translation",
            "description": (
                "Set translation through the existing application owner without authorizing "
                "accounts or changing providers. Inspect the terminal receipt."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {"enabled": {"type": "boolean"}},
                "required": ["enabled"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "diagnose_microphone",
            "description": (
                "Run bounded local microphone capture/VAD diagnosis on the current route "
                "and actual enumerated alternatives. No network recognizer or saved settings "
                "changes. Inspect candidates, cleanup and restoration in the terminal receipt. "
                "Frames or speech activity are not recognition success."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "retry_osc_discovery",
            "description": (
                "Run one bounded automatic OSC discovery requery/recovery using the current "
                "connection settings and await its terminal internal application result. "
                "Does not enable disabled OSC, change manual mode or saved destinations, "
                "or send chatbox text. Target application does not prove VRChat receipt or display."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "get_provider_setup",
            "description": (
                "Read app-owned official provider setup links and the existing secure-field "
                "identity. No navigation, browser opening, provider switching or key entry. "
                "Returned metadata does not mean a link or key input is currently visible."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "provider": {"type": "string", "enum": sorted(NAVIGATION_PROVIDERS)},
                },
                "required": ["provider"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "open_settings",
            "description": (
                "Open a supported settings section without changing settings. For key help "
                "open api with the known provider to expose its existing secure input, otherwise null. "
                "This does not select that provider or start browser authorization. "
                "Navigation never proves key verification or user-step completion."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "section": {"type": "string", "enum": ["api", "general", "overlay", "prompts"]},
                    "provider": {
                        "type": ["string", "null"],
                        "enum": [*sorted(NAVIGATION_PROVIDERS), None],
                    },
                },
                "required": ["section", "provider"],
                "additionalProperties": False,
            },
        },
        *(
            {
                "type": "function",
                "name": f"probe_{channel}_recognition",
                "description": (
                    f"Check {channel.title()} recognition independently with app-owned synthetic "
                    "Korean speech (about one second) in an isolated session of its configured "
                    "backend. No arguments, live capture, playback, provider switching, settings "
                    "changes or transcript/history/translation/output publication. Inspect the "
                    "terminal receipt. Success means nonempty final synthetic recognition only, "
                    "not accuracy, other-language quality, microphone/loopback capture or display. "
                    "The existing configured connection may incur ordinary recognition cost."
                ),
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            }
            for channel in ("self", "peer")
        ),
        {
            "type": "function",
            "name": "probe_translation",
            "description": (
                "Run one synthetic translation through the current attached usable backend. "
                "No arguments, live conversation, custom prompt, history, capture activation, "
                "provider provisioning or output publication. Inspect the terminal receipt; "
                "success tests translation only, not output delivery or display. Existing "
                "runtime routing policy remains in effect; this does not prove primary-model recovery."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    ]
    args.output.write_text(
        "export const actionTools = " + json.dumps(tools, ensure_ascii=False, indent=2) + ";\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
