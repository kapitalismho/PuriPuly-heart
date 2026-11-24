# Command-line control

The CLI starts a headless host or controls a running GUI/headless instance. Both use the same application runtime. Change settings through the CLI, not by editing JSON.

This guide covers usage and result interpretation. Discover syntax with `--help`, operations with `capabilities`, and settings with `settings choices`. See the [source map](#source-map) for implementation details.

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

- Use the same `--config` path for host and clients. Omission selects the user's normal settings. For probes, isolate settings and writable runtime roots; a separate settings file alone does not isolate secrets or models.
- One host owns each settings identity. `app start` runs in the foreground; `--background` outlives the terminal. Launching the application without a CLI command opens the GUI.
- `app list` lists current-user instances. `app discover` and `app status` target the selected settings identity; `app restart --background` replaces that host.
- Commands never start a missing host implicitly. On `instance_not_found`, start it explicitly.
- `app stop` waits for host exit. `--no-wait` returns shutdown acceptance, not completed cleanup.
- `--timeout <seconds>` bounds client waiting, not operation lifetime, and never retries a mutation. `Ctrl+C` exits 130; a background host remains independent.

Restart source hosts after code changes. Installed builds require a rebuilt payload.

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

Choices depend on extensions and devices. Query the host instead of hard-coding catalogs. Queries do not enable recording.

Convenience commands cover common controls:

```powershell
puripuly.exe capture set self on
puripuly.exe translation set on
puripuly.exe audio test on
puripuly.exe overlay set on
```

Use named commands and queries for other operations:

```powershell
puripuly.exe query <query-name> --file .\query.json
puripuly.exe command <command-name> --file .\arguments.json --no-wait
```

Use `--arguments '<JSON object>'` only for non-sensitive input. Unknown names and fields are rejected. Protected operations require the dedicated `secrets`, `auth`, and `text` commands.

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

- Use `--expected-revision` to reject concurrent edits. Focused changes preserve unrelated settings.
- Settings commit before runtime application. `degraded` can mean saved but not applied; cancellation does not undo a commit.
- Apply a translation model and connection together using `settings choices`. For ASR, use `asr set --channel self|peer|both --provider <choice>`.
- Retry failed provider application explicitly. Unrelated settings edits do not retry it.
- SteamVR caption compatibility: apply `{"changes":{"overlay.vr_compatibility_mode":true}}` through the settings command above, or select VR (Compatibility Mode) in Settings → Overlay → Overlay mode. Default `false` avoids periodic retries; `true` keeps Flush and adds bounded redraws/resubmissions. The setting persists without enabling a disabled overlay; switching is manual.
- Check `overlay status` → `effective_vr_compatibility_mode` for the active mode (`null` unless SteamVR captions are ready), separately from saved settings.
- Check `query photo.status` for VR photo translation. `availability` is `off`, `not_connected` (the chosen route's ChatGPT sign-in or OpenAI/Gemini key is missing), `allowance_exhausted`, `allowance_unknown` or `ready`, with today's Managed `remaining`/`allowance` when known. Pass `{"refresh": true}` to re-read the Managed allowance from the service; without it the query uses only today's last answer.

### Readiness

| Observation | What it does not establish |
| --- | --- |
| Saved provider/model selection | Attached or usable runtime provider |
| `app.status.translation_runtime_ready` | Translation enabled or managed model prepared; it means a provider is attached |
| Active audio capture | Recognition availability or successful translation |
| Configured overlay/OSC target | Effective resource readiness, remote receipt, or physical display |
| Applied manual-text admission | Completed translation or destination delivery |

Read the terminal receipt and the relevant status query: `providers.status`, `models status`, `overlay status`, or `osc status`. Fix the reported cause before retrying. Capture, recognition, translation, and output have separate readiness states.

### Independent Component Checks

These commands use application-owned synthetic inputs, not the microphone, loopback, speaker playback, or normal conversation/output pipeline:

| Command | Arguments | Successful terminal result establishes |
| --- | --- | --- |
| `recognition.probe` | `{"channel":"self"}` or `{"channel":"peer"}` | Final recognition from an isolated synthetic sample |
| `translation.probe` | `{"channel":"self"}` or `{"channel":"peer"}`; default Self | Synthetic translation through the attached backend in that channel's language direction |

```powershell
puripuly.exe command recognition.probe --arguments '{"channel":"self"}'
puripuly.exe command recognition.probe --arguments '{"channel":"peer"}'
puripuly.exe command translation.probe
puripuly.exe command translation.probe --arguments '{"channel":"peer"}'
```

Read the terminal result. Probes do not enable capture, publish conversation text, install models, or sign in. Provider requests may incur normal costs. Prepare missing prerequisites separately.

These checks cover individual components, not microphone input, accuracy, output delivery, or physical display.

### Peer Capture and Audio Tests

Read `capture terms` before enabling Peer capture. Use `capture set peer on --accept-peer-terms` only with the user's explicit consent. Peer speech never reaches the chatbox.

`audio test on|off` checks input and VAD without recognition or translation. `audio diagnose` compares available microphone routes and suggests a separate settings change. Both pause normal capture and restore still-valid intent afterward.

```powershell
puripuly.exe audio diagnose
```

Inspect input, VAD, and cleanup results. Silence or an enumerated device alone does not show that speech input works.

### Manual Text

Manual text bypasses microphone/STT and follows the Self translation/output path. Keep conversation text out of process arguments:

```powershell
puripuly.exe text submit --file .\message.txt
Get-Content -Raw .\message.txt | puripuly.exe text submit --stdin
```

The receipt confirms admission, not translation or delivery. Use its turn identity to follow subsequent events.

### Local Speech Recognition

Use `settings choices` to select ASR and `models status` to inspect assets. CPU Auto routes by the configured source language, not detected speech. It requires the current CPU model set.

Model routing and assets are defined in the [ASR catalog](../src/puripuly_heart/core/local_asr/local_stt_catalog.py) and [asset registry](../src/puripuly_heart/core/local_asr/local_stt_assets.py).

### Managed Local Translation

Select the model and backend through `settings choices` and `settings apply`. Run `gemma prepare` to prepare the selected model, or `gemma cancel` to cancel preparation. Inspect readiness with `models status`.

Use `models download|import|cancel|delete --help` for asset operations. These do not change provider selection. After cancellation, wait for the operation to settle before retrying a conflicting operation.

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

- Check `terminal`, status, operation identity, and effective state. Unsupported cancellation can leave the operation running.
- An OAuth challenge can return `action_required` with `terminal: false`. Complete authorization, then wait for the same operation.
- Timeout or lost response leaves execution uncertain. Inspect state before retrying; the CLI never automatically replays a mutation.
- `command --request-id <id>` deduplicates matching requests while retained. Reusing the ID with different input is rejected. Request and operation history resets with the host.

## Credentials, Authorization, and Consent

### Secrets

Use hidden prompts or stdin, never ordinary command arguments:

```powershell
puripuly.exe secrets set <secret-id>
puripuly.exe secrets verify <secret-id>
puripuly.exe secrets presence
```

`set` and `verify` accept `--stdin`. Discover secret identifiers through `capabilities`. Storage, verification, and runtime adoption are separate results; secret values are never returned.

### Authentication and Recovery

```powershell
puripuly.exe auth status
puripuly.exe auth login discord --open-browser
puripuly.exe auth login chatgpt --open-browser
puripuly.exe auth logout chatgpt
```

- Login requires explicit authorization. Use `--open-browser` to open the browser.
- QQ accepts hidden prompts or `auth login qq --stdin` with protected `qq_identity` and `credential` fields.
- For pending managed delivery, correct the cause and retry the same source's login. Do not request a replacement key or delete settings/secrets to clear uncertainty.
- Logout reports local removal and any remote revocation separately. Storage-read failure means unknown state, not logout.
- Correct proxy/TLS configuration instead of disabling certificate verification. Existing connections may need restarting after proxy changes.

## Events and Privacy

Results are UTF-8 JSON on stdout with `output_version: 1`; errors use stderr. Follow commands emit one JSON event per line. The local endpoint requires same-user authentication over loopback.

```powershell
puripuly.exe events follow --topic operation --topic settings
puripuly.exe logs follow --level warning
puripuly.exe events follow --topic translation --channel peer --include-translations
```

- Conversation content requires explicit `--include-transcripts` or `--include-translations` on the matching topic.
- Buffers are bounded. Refresh status after gaps; events have no global ordering.
- Replay with `--after <event-sequence> --instance-id <originating-instance>`. Without `--after`, only new events are observed.
- CLI logs are sanitized metadata. Console and main-file logs can contain conversation text; redact them before sharing. See [local information](../PRIVACY.md#5-local-information-and-deletion).

## Retained Diagnostics and Delivery Health

Inspect a past failure without having subscribed before it happened:

```powershell
puripuly.exe query diagnostics.history
puripuly.exe query diagnostics.history --arguments '{"operation_id":"<operation-id>","instance_id":"<originating-instance-id>"}'
puripuly.exe query diagnostics.health
```

Use returned identities and the filters in `capabilities.query_arguments`.

- History is bounded and current-instance only. Missing records do not imply success.
- Diagnostic `next_after` and event `sequence` are different cursors. Keep the originating instance with each.
- `diagnostics.health` reports delivery failures. Queued records are not proof of file persistence or remote display.

Read owner state and terminal receipts for the operation outcome. See the [diagnostic schema](../src/puripuly_heart/core/diagnostics.py) for record details.

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
