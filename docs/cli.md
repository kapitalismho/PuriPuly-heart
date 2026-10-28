# Command-line control

The CLI controls the same application owners as the GUI. It can start a headless host or attach to a running GUI/headless instance; it is not a second translation runtime and does not edit settings JSON directly.

This guide covers the usage model and contracts needed for safe automation. Discover command syntax with `--help`, supported operations with `capabilities`, and current values with `settings choices`. Provider internals, policy constants, and regression details belong in [the implementation](#source-map), not this guide.

## Entry Point and Host Lifecycle

Installed Windows command:

```powershell
puripuly.exe --help
puripuly.exe app start --background
puripuly.exe app status
puripuly.exe app stop
```

In a development checkout, use the same CLI through the Python entry point:

```powershell
uv run python -m puripuly_heart.main cli --help
uv run python -m puripuly_heart.main cli --config C:\work\puripuly-dev.json app start --background
uv run python -m puripuly_heart.main cli --config C:\work\puripuly-dev.json app status
uv run python -m puripuly_heart.main cli --config C:\work\puripuly-dev.json app stop
```

- Use the same `--config` path for host and clients. It identifies canonical settings-file ownership; omission selects the user's normal settings. Use isolated synthetic settings for probes, not a live user's configuration.
- Only one host may own that settings identity. `app start` defaults to foreground; `--background` outlives the launching terminal. Ordinary no-command application launch still opens the GUI.
- `app list` lists current-user instances. `app discover` and `app status` target the selected settings identity; `app restart --background` replaces that host.
- Queries and mutations never start a missing host implicitly: `instance_not_found` requires an explicit start. A PID is observation, not control identity; the CLI never kills by PID.
- `app stop` waits for the original endpoint and observed process identity to disappear. `cleanup_outcome: unobserved` does not certify every teardown callback or resource release. `--no-wait` returns only shutdown acceptance.
- `--timeout <seconds>` bounds client waiting, not operation lifetime, and never retries a mutation. `Ctrl+C` exits 130; a background host remains independent.

## Discover Before Acting

| Need | Authority |
| --- | --- |
| CLI syntax and flags | `puripuly.exe <domain> --help` and subcommand help |
| Commands, arguments, queries, secret identifiers, event topics | `capabilities` |
| Setting types, provider/model combinations, finite values | `settings choices` |
| Committed settings and revision | `settings current` |
| Effective resources and failures | `app status` and domain status queries |

```powershell
puripuly.exe capabilities
puripuly.exe settings choices
puripuly.exe settings current
puripuly.exe capture status
puripuly.exe asr status
puripuly.exe models status
puripuly.exe query providers.status
```

Choices can depend on extensions and available devices. Query the running host rather than copying a catalog into a script. Queries are side-effect-free observations; enumeration may inspect the desktop/audio environment but does not enable recording.

Convenience commands cover common controls:

```powershell
puripuly.exe capture set self on
puripuly.exe translation set on
puripuly.exe audio test on
puripuly.exe overlay set on
```

For less common operations, use the finite named catalog, not private Python APIs:

```powershell
puripuly.exe query <query-name> --file .\query.json
puripuly.exe command <command-name> --file .\arguments.json --no-wait
```

`--arguments '<JSON object>'` is an alternative for non-sensitive payloads. Unknown names/fields are rejected; this is not arbitrary RPC reflection. Use dedicated `secrets`, `auth`, and `text` commands for protected operations: generic dispatch rejects `secrets.set`, `secrets.verify`, `auth.login`, and `text.submit`.

## Settings and Runtime State

Read the current revision, then submit focused changes rather than rewriting the settings file:

```json
{
  "changes": {
    "telemetry.enabled": false
  }
}
```

```powershell
puripuly.exe settings current
puripuly.exe settings apply --file .\changes.json --expected-revision <current-revision>
```

- `--expected-revision` is optional; use it to reject a concurrent edit rather than overwrite it. Focused changes preserve unrelated settings.
- Settings/provider changes commit together, then runtime owners apply them. A runtime failure can return `degraded` while the intent remains saved. Cancellation does not undo a durable commit.
- Apply a translation model and its connection together using a combination from `settings choices`. For ASR, `asr set --channel self|peer|both --provider <choice>` is the convenience form.
- Explicit provider fields can retry pending provider application; unrelated general settings edits do not. Observe committed settings and effective provider state separately, including after restart when checking persistence.

### Interpret Readiness at Its Owner

| Observation | What it does not establish |
| --- | --- |
| Saved provider/model selection | Attached or usable runtime provider |
| `app.status.translation_runtime_ready` | Translation enabled or managed model prepared; it means a provider is attached |
| Active audio capture | Recognition availability or successful translation |
| Configured overlay/OSC target | Effective resource readiness, remote receipt, or physical display |
| Applied manual-text admission | Completed translation or destination delivery |

Use the terminal operation receipt together with `providers.status`, `models status`, `overlay status`, `osc status`, or the relevant owner's query. Correct the reported cause before explicitly retrying. A failed/cancelled current translation warmup turns effective translation off while retaining the selection and failure details.

Matching `translation set off` or `capture set <channel> off` can revoke pending startup. Shared model demand may keep resources alive for another channel. Runtime stops are distinct from cancelling an explicit model download/import.

### Peer Capture and Audio Tests

Read `capture terms` first. Only an informed user's explicit choice authorizes `capture set peer on --accept-peer-terms`; never auto-supply consent. The flag is valid only for peer-on. Peer speech is excluded from the VRChat chatbox.

`audio test on|off` runs VAD without recognition or translation. It pauses normal capture and restores only still-valid prior intent; inspect `app status` → `audio_test` for readiness, failures, and VAD readings. Stopping waits for owned input release, not just a visual toggle.

### Manual Text

Manual text bypasses microphone/STT and follows the Self translation/output path. Keep conversation text out of process arguments:

```powershell
puripuly.exe text submit --file .\message.txt
Get-Content -Raw .\message.txt | puripuly.exe text submit --stdin
```

The receipt provides admission and turn identities with `output_delivery: unobserved`. Correlate subsequent translation events and destination evidence; the bounded typing-completion wait is not an output acknowledgment.

### Managed Local Translation

Select the model/backend through `settings choices` and `settings apply`. `gemma prepare` prepares the selected model; `gemma cancel` cancels preparation. Inspect `models status` for Gemma, not ASR-specific `gpu status`.

- Assets download on demand and are validated before atomic installation. `models status` exposes model IDs, asset state, and import requirements; use `models download|import|cancel|delete --help` for explicit asset operations.
- Asset operations preserve provider selection. On cancellation, wait for the original operation to settle before retrying a conflict; runtime demand changes do not cancel transfers. In-use deletion is rejected.
- E4B supports CPU/GPU and may fall back internally from GPU to CPU. 26B is Vulkan-only and never substitutes CPU, E4B, or a remote provider after failure. Resolve the cause or explicitly choose another supported selection.
- Self, Peer, and manual requests share the managed server's bounded capacity. Model/backend changes drain owned work before replacement. Preparation success is not translation success.
- Managed Windows servers run offline and ignore inherited llama.cpp environment settings and user/system configuration. Model provisioning remains owned by the application; this is not an operating-system network sandbox.

Fixed deployment profiles are not CLI tuning settings. Exact model identities/hashes, runtime pins, and execution options live in [assets.py](../src/puripuly_heart/core/local_translation/assets.py), [runtime_profile.py](../src/puripuly_heart/core/local_translation/runtime_profile.py), and the [runtime notice](../third_party/llama.cpp/README.md).

## Operation Results and Retries

Mutations wait for a terminal result by default. Where supported, `--no-wait` returns an accepted operation identity, not completion.

| Status | Interpretation |
| --- | --- |
| `accepted`, `running` | Nonterminal: queued or executing |
| `applied` | Terminal: the reported operation effect completed, not every downstream effect |
| `degraded` | Terminal: incomplete runtime application; saved intent may remain |
| `rejected`, `persistence_failed`, `failed` | Terminal rejection, failed persistence, or failed execution |
| `action_required` | Read `terminal`: a final missing prerequisite or an interim authorization challenge |
| `cancelled`, `interrupted` | Terminal interruption; already committed/applied effects remain |

```powershell
puripuly.exe operation status <operation-id>
puripuly.exe operation wait <operation-id>
puripuly.exe operation cancel <operation-id>
```

- Always inspect `terminal`, status, instance/operation identity, and effective state before proceeding. Unsupported cancellation exits 4 and may leave the original operation running.
- An OAuth challenge can return an authorization URL with `action_required`, `operation_status: running`, and `terminal: false` (exit 5). Complete authorization, then inspect/wait for that same operation.
- A timeout or lost response can leave execution unknown. Inspect host/operation state before deciding whether to retry; the CLI never automatically replays an ambiguous mutation.
- `command --request-id <id>` supplies a stable request identity. Matching identities/payloads are deduplicated only while retained; different content with the same ID is rejected. Retention is bounded and resets with the host: old operation IDs cannot resume on a new instance.

## Credentials, Authorization, and Consent

### Secrets

Use hidden prompts or stdin, never ordinary command arguments:

```powershell
puripuly.exe secrets set <secret-id>
puripuly.exe secrets verify <secret-id>
puripuly.exe secrets presence
```

`set` and `verify` accept `--stdin`; discover identifiers through `capabilities`. Verification takes protected input and is separate from storage. Secret values are never returned in settings, receipts, queries, or logs.

Saving an active BYOK translation key applies it to subsequent work without enabling capture/translation. A saved-but-unapplied key can report `degraded`; keep storage success distinct from runtime adoption. A verification probe is not proof of live recognition/translation health, and unavailable verification reports `action_required` rather than success.

### Authentication and Recovery

```powershell
puripuly.exe auth status
puripuly.exe auth login discord --open-browser
puripuly.exe auth login chatgpt --open-browser
puripuly.exe auth logout chatgpt
```

- Login is explicit authorization. Translation/settings changes do not implicitly initiate OAuth. Browser opening requires `--open-browser`; consent is never auto-accepted.
- QQ uses hidden identity/credential prompts or `auth login qq --stdin` with a protected JSON object containing `qq_identity` and `credential`. Never place those values in generic arguments.
- Authorization, local credential persistence, remote delivery acknowledgment, and runtime activation are distinct. A pending managed delivery or a network failure is not permission to claim another key.
- For `resolve_pending_authorization` / `remote_delivery_ack_pending`, preserve settings/secrets and explicitly retry the same source's login after correcting the cause. Saved authority resumes reconciliation; it does not prove a recovery task is running. A retained unacknowledged modern key is not automatically usable after restart.
- Storage-read failure is unknown state (`signed_in: null`), not logout. Do not delete settings, encrypted secrets, or their `.lock` file as a recovery shortcut. If authentication succeeded but provider activation failed, retry translation enablement after correcting the runtime cause.
- Logout is local-only except ChatGPT, which also requests refresh-token revocation. Inspect the reported scope; local deletion alone does not prove remote revocation.
- Browser authorization and app-to-service traffic can use different routes. Correct proxy/TLS configuration rather than disabling certificate verification. Share only sanitized diagnostic fields, never OAuth codes, keys, tokens, proxy URLs, or full settings/secret files.

## Events and Privacy

Normal CLI results are UTF-8 JSON with `output_version: 1` on stdout; diagnostics/errors use stderr. Help is human-readable. Follow commands emit one JSON event per line. The local protocol authenticates same-user instance identity and token over loopback; it is not an unauthenticated management API.

```powershell
puripuly.exe events follow --topic operation --topic settings
puripuly.exe logs follow --level warning
puripuly.exe events follow --topic translation --channel peer --include-translations
```

- Default query/event/log views exclude credentials and conversation content. Content subscriptions require the corresponding `transcript`/`translation` topic and explicit `--include-transcripts`/`--include-translations`. Keep them off unless needed and protect the output.
- Subscribers have independent bounded buffers. Slow consumers cannot block producers; gaps require refreshing owner snapshots. Ordering is per owner/topic/channel, not global.
- Use `--after <event-sequence> --instance-id <originating-instance>` for replay. Without `--after`, follow observes new events only.
- CLI logs are sanitized metadata, not a copy of every GUI log row or the main file. Main logs may contain permitted conversation records: review/redact them before sharing.

## Retained Diagnostics and Delivery Health

Inspect a past failure without having subscribed before it happened:

```powershell
puripuly.exe query diagnostics.history
puripuly.exe query diagnostics.history --arguments '{"operation_id":"<operation-id>","instance_id":"<originating-instance-id>"}'
puripuly.exe query diagnostics.health
```

Use returned identities, not invented placeholders. `capabilities.query_arguments` describes supported history filters and limits.

- Records preserve safe typed causes and known relationships. Missing relationships remain unknown; secondary cleanup failures do not replace the primary cause. Owner state and terminal receipts remain authoritative.
- History is bounded, in-memory, and current-instance only. Inspect gaps, eviction/truncation, and lookup outcomes. `not_retained`, `foreign_instance`, or `invalid_cursor` does not mean an operation succeeded or never failed.
- History's `next_after` is a diagnostic cursor. An event's `sequence` is a replay cursor. They are not interchangeable; retain the originating instance with either.
- `diagnostics.health` reports history/event/file/output delivery health independently of a working file sink. Enqueued is not written; written is not durable. Aggregate counters do not prove persistence of an individual record.
- Terminal file records trigger delivery without waiting for ordinary batch expiry; their severity is unchanged. This does not bypass a blocked sink or the bounded close deadline: a timeout remains visible and cleanup stays listener-owned.
- Diagnostics cannot establish remote UDP receipt or physical display. They exclude raw exceptions, arbitrary child output, personal paths, credentials, audio, and conversation content.

Exact schemas, bounds, and projections live in [diagnostics](../src/puripuly_heart/core/diagnostics.py), [runtime logging](../src/puripuly_heart/core/runtime_logging.py), and [control events](../src/puripuly_heart/app/services/application_control_events.py).

## Exit Codes

| Code | Meaning |
| ---: | --- |
| 0 | Applied result, successful query/start/stop, or accepted non-waiting request |
| 2 | Syntax/argument parsing error; help exits 0 |
| 3 | Invalid input, protocol/transport error, or unavailable instance |
| 4 | Rejected/degraded/failed operation, persistence failure, or unsupported cancellation |
| 5 | Action required, including a nonterminal OAuth challenge |
| 6 | Cancelled or interrupted terminal operation |
| 7 | Unknown execution/startup outcome, nonterminal wait, or ambiguous host disappearance |
| 130 | Keyboard interrupt |

An exit code does not replace the JSON receipt or owner-state observation.

## Source Map

| Question | Read |
| --- | --- |
| Syntax, input protection, receipt-to-exit mapping | [CLI entry point](../src/puripuly_heart/cli/main.py) |
| Command/query catalog, schemas, settings choices, operation ordering | [Application control owner](../src/puripuly_heart/app/services/application_control.py) |
| Endpoint, wire protocol, discovery, instance lifetime | [Host](../src/puripuly_heart/cli/host.py), [transport](../src/puripuly_heart/cli/transport.py), [instance ownership](../src/puripuly_heart/core/control_instance.py) |
| Events, privacy projection, replay | [Control events](../src/puripuly_heart/app/services/application_control_events.py) |
| Owner boundaries and downstream implementation | [Architecture](architecture.md) |
| Application behavior contracts | [Control contract tests](../tests/app/test_application_control_contract.py) |
