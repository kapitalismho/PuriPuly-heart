# Command-line control

PuriPuly's CLI controls the same application owners used by the desktop GUI. It can start a GUI-free host, attach to an already-running GUI or headless host, inspect the active runtime, and submit typed application operations. It is not a second ASR/translation runtime and it does not edit the settings JSON directly.

## Choose the entry point

### Installed Windows command

Use the installed console command `puripuly.exe` (or `puripuly` when its install directory is on `PATH`):

```powershell
puripuly.exe --help
puripuly.exe app start --background
puripuly.exe app status
puripuly.exe app stop
```

The package declares the `puripuly` console entry point. Its CLI contract is JSON on stdout, diagnostics on stderr, and documented process exit codes.

### Development checkout

Run the same CLI through the Python module entry point:

```powershell
python -m puripuly_heart.main cli --help
python -m puripuly_heart.main cli --config C:\work\puripuly-dev.json app start --background
python -m puripuly_heart.main cli --config C:\work\puripuly-dev.json app status
python -m puripuly_heart.main cli --config C:\work\puripuly-dev.json app stop
```

Use the same `--config` path for a host and each client that should reach it. The path identifies canonical settings-file ownership; without it the installed or development CLI uses the current user's normal settings file. A running GUI and a headless host use the same settings identity and control endpoint.

## Host lifecycle

`app start` is the explicit GUI-free host launcher. It runs in the foreground by default; `--foreground` makes that choice explicit. `--background` starts a detached service that outlives its launching terminal:

```powershell
puripuly.exe app start --foreground
puripuly.exe app start --background
puripuly.exe app list
puripuly.exe app discover
puripuly.exe app status
puripuly.exe app restart --background
puripuly.exe app stop
```

- Only one host may own a canonical settings-file identity for the current user. A duplicate start is an explicit error, not a second competing capture or settings writer.
- `app list` enumerates discoverable current-user instances. `app discover` and `app status` query the instance matching `--config` and report its instance identity, process ID, channel and translation state. The PID is observation only; it is not the control identity or a signal target.
- `app restart` stops the instance for that settings identity before starting it again.
- A query or mutation does not launch a missing host implicitly. A missing instance is an explicit `instance_not_found` error; start one deliberately with `app start`.
- `app stop` waits until the original instance endpoint is gone and the observed original process identity has exited. `app stop --no-wait` returns the accepted shutdown receipt instead. It never kills a process by PID.
- `Ctrl+C` interrupts a foreground host/client with exit code 130. A background host is independent of the short-lived launch client. On Windows, the host retains its kill-on-close child-process Job Object; the CLI client does not inherit that host containment.

For direct host startup, `python -m puripuly_heart.main run-headless --config <settings-file>` is the explicit GUI-free runtime command. The CLI form `app start --foreground` is equivalent for control use and establishes local discovery. Do not use the legacy native `--headless` launcher dispatch as a service contract: it does not itself define an application-control host, and Python's no-command fallback starts the GUI. The ordinary no-command application launch remains GUI startup.

## Discover commands, values, and state

JSON is the default CLI output; there is no `--json` switch. Use `capabilities` to get the active host's command argument descriptions, query names, setting fields, secret-key identifiers, and event topics. Use `settings choices` for runtime-supported provider/model/connection/region/overlay values and other finite choices exposed by the catalog. These values may depend on loaded extensions or available devices, so discover them from the running host rather than copying a stale list into a script.

```powershell
puripuly.exe capabilities
puripuly.exe settings current
puripuly.exe settings choices
puripuly.exe capture status
puripuly.exe capture terms
puripuly.exe asr status
puripuly.exe audio devices list
puripuly.exe audio processes list
puripuly.exe audio target status
puripuly.exe models status
puripuly.exe gpu status
puripuly.exe overlay status
puripuly.exe osc status
puripuly.exe auth status
puripuly.exe secrets presence
```

The corresponding finite named-query interface is:

```powershell
puripuly.exe query app.status
puripuly.exe query providers.status
puripuly.exe query settings.current
puripuly.exe query settings.choices
puripuly.exe query consent.peer_translation
```

Use `capabilities` to discover the supported query names. Queries are side-effect-free observations; discovery commands such as audio-device enumeration may read the current desktop/audio environment, but do not enable capture or start recording.

## Common controls

Dedicated commands are explicit setters where a state change is meaningful:

```powershell
puripuly.exe capture set self on
puripuly.exe capture set peer off
puripuly.exe translation set on
puripuly.exe asr set --channel both --provider soniox
puripuly.exe microphone test on
puripuly.exe audio target set <value-from-audio-target-status>
puripuly.exe audio retry
puripuly.exe overlay set on
puripuly.exe overlay lock off
puripuly.exe overlay size <preset-from-settings-choices>
puripuly.exe overlay reset-position
puripuly.exe overlay calibrate begin
```

Capture/provider/output states in command receipts are runtime observations, not just persisted selections. Peer capture requires informed, explicit acceptance of the existing terms. Read them with `capture terms` (query `consent.peer_translation`) first. If consent is not already stored, the user may then explicitly run `capture set peer on --accept-peer-terms`; this flag is valid only for enabling the peer channel, routes through the existing GUI acceptance owner, and is never auto-supplied by the CLI.

The finite fallback is equivalent only when it carries the same explicit consent field: `puripuly.exe command capture.set --arguments '{"channel":"peer","enabled":true,"accept_terms":true}'`. `accept_terms` is invalid for self capture or a peer-off request. Inspect the terms first and pass `true` only for the user's explicit informed choice.

`overlay status` returns the owner-backed projection in its top-level `output` object: desired/configured/effective/attempting target, lifecycle, runtime/process state, presentation readiness, desktop visibility, generation, failure/recovery and ingress-stop state. `overlay set` waits for the explicit owner transition; a configured target alone is not readiness, and future automatic recovery does not make a failed attempt successful. `desktop_visible` is `null` for non-desktop targets. `osc status` also returns a top-level `output` object, separating configured from effective local mode/ports (effective values are `null` without their runtime resource) and reporting receiver/sender/service availability and discovery/failure state. OSC/UDP `remote_delivery` is always unacknowledged; it does not prove that VRChat received a packet.

`overlay calibrate change --field FIELD --value VALUE` accepts `anchor` as a choice from `settings choices` (`overlay.calibration_fields.anchor`) and parses the remaining supported fields as finite numbers: `offset_x`, `offset_y`, `distance`, `text_scale`, or `background_alpha`. Distance and text scale must be positive; alpha is between 0 and 1. For example, begin a calibration, run `overlay calibrate change --field offset_x --value 0.05`, then apply or cancel it. Numeric CLI values are sent as JSON numbers; field-specific bounds are validated by the application owner.

`microphone test on` remains in progress until the owner receives an actual input frame, not merely until a capture task is scheduled. `app status` returns a `microphone_test` object with `state`, `desired_active`, `effective_active`, normalized `meter_level` (0–1), `failure_reason`, and `failure_type`; no audio content is returned. An unavailable input produces `action_required`. `microphone test off` can interrupt pending startup and waits for owned input resources to close. Testing takes exclusive self-input ownership and can disable normal self capture; re-enable that capture explicitly afterward if needed.


Manual text goes through the normal self transcript/translation/output pipeline, without microphone capture or STT. Do not put conversation text in a process argument:

```powershell
puripuly.exe text submit --file .\message.txt
Get-Content -Raw .\message.txt | puripuly.exe text submit --stdin
```

## Typed settings and provider changes

Use `settings.current` to read the current `revision`. `settings apply` accepts one UTF-8 JSON object from `--file` (or a non-sensitive object from `--arguments`) and an optional `--expected-revision`. The JSON may wrap focused updates in `changes`; unknown fields and invalid types are rejected. The application materializes those updates as typed settings/provider intents against canonical settings, persists them through the settings owner, and applies runtime effects through the owning runtime boundary. It never treats a direct JSON file rewrite as a settings operation.

Revisions and settings events follow successfully committed changes, including GUI-only edits. Queries return the committed snapshot without advancing its revision. Staged preparation and failed persistence are not reported as persisted settings.

`chatbox.activation_notice.enabled` is a boolean preference, enabled by default, persisted as `intent.osc.activation_notice_enabled`. Setting it to `false` suppresses the Talk `PuriPuly ON!` message and Listen audio-capture disclosure without disabling capture or normal translation output. It applies immediately without a runtime restart, and changing it to `true` does not itself send or replay a notice. Peer capture still requires explicit acceptance of its terms.

Example `changes.json` (supported field/type shape; provider availability and credentials still determine runtime success):

```json
{
  "changes": {
    "stt.provider": "soniox",
    "peer_stt.provider": "qwen_audio",
    "translation.model": "gemma4_26b_31b",
    "translation.connection": "openrouter",
    "chatbox.activation_notice.enabled": false,
    "telemetry.enabled": false
  }
}
```

Apply it against the revision just read:

```powershell
puripuly.exe settings current
puripuly.exe settings apply --file .\changes.json --expected-revision 12
```

`12` is an example revision, not a fixed value; substitute the current result's revision. If another settings mutation changes that revision first, the operation is rejected with a revision conflict. Focused changes preserve unrelated settings. For ASR-only changes, use the convenience form or the command fallback:

```powershell
puripuly.exe asr set --channel both --provider soniox
puripuly.exe command provider.apply --arguments '{"channel":"both","provider":"soniox"}'
```

A file-based command can also express both channels explicitly:

```json
{
  "changes": {
    "stt.provider": "soniox",
    "peer_stt.provider": "qwen_audio"
  }
}
```

Use `asr set` with `--channel self|peer|both` for the same STT provider on the requested channel(s), or settings fields when the two channels need different providers. `settings choices` is the authority for currently available values and valid model/connection combinations. A successful persistence receipt is not proof that a provider is attached or active; inspect `asr status` / `providers.status` and its selected, runtime, capture-attached, activity, pending-handoff, and failure fields.

The finite command catalog also includes Soniox diarization, managed referral settings, custom and local provider settings, GPU device selection, both-channel provider updates, free-tier provider choices, Qwen region, model/connection history, prompts, languages, vocabulary, VAD, audio devices/host APIs, overlay values, OSC and telemetry settings. Use `capabilities` for field names and types, and `settings choices` for supported values rather than reproducing dynamic lists.

In an already-running GUI, focused external changes preserve unrelated provider/prompt drafts; overlapping changes require explicit conflict resolution. A successful pending GUI apply acknowledges only its submitted values. Newer edits made while it was pending remain staged, and a failed apply retains its draft for retry.

## Named query and command fallback

Convenience domains cover the common operations. When scripting a less common supported operation, use the finite catalog rather than private Python APIs:

```powershell
puripuly.exe query <query-name> [--arguments '<JSON object>' | --file <query.json>]
puripuly.exe command <command-name> [--arguments '<JSON object>' | --file <arguments.json>] [--expected-revision <revision>] [--no-wait]
```

The catalog rejects unknown names; it is not RPC reflection. Query names currently take no arguments. `command` accepts only names and argument shapes returned by `capabilities`. Use `--file` for JSON payloads that include sensitive or private content. Protected operations have dedicated `secrets`, `auth`, and `text` commands; the generic command form deliberately rejects `secrets.set`, `secrets.verify`, `auth.login`, and `text.submit`.

Global options are `--config <settings-file>` and `--timeout <seconds>` (greater than 0 and at most 3600). Mutations wait for a terminal result by default. `--no-wait` returns an accepted operation identity without claiming completion.

## Operation results and retries

Every operation receipt has an explicit `terminal` boolean:

| Receipt status | `terminal` | Meaning |
| --- | --- | --- |
| `accepted`, `running` | `false` | The operation is queued or still executing. |
| `applied` | `true` | The operation reached its reported applied result. |
| `degraded` | `true` | Settings may be persisted, but runtime application/convergence is incomplete. This is not full success. |
| `rejected` | `true` | Input, revision, precondition, or operation was rejected. |
| `persistence_failed` | `true` | Requested persistence did not complete. |
| `failed` | `true` | The operation failed. |
| `action_required` | `true` for a final precondition result | Human action or an unavailable prerequisite remains. |
| `cancelled`, `interrupted` | `true` | Cancellation or shutdown interrupted the operation; already persisted/applied work is not undone. |

An OAuth challenge is an important interim exception: an explicitly requested Discord/OpenRouter `auth login` can print an authorization URL with `status: action_required`, `operation_status: running`, and `terminal: false`. Exit code 5 does not mean that login finished. Complete the human authorization separately, then inspect or wait for the same operation:

```powershell
puripuly.exe operation status <operation-id>
puripuly.exe operation wait <operation-id>
```

Only a terminal receipt establishes the final login result. A final `action_required` is terminal and means the workflow cannot proceed without its stated action. Never treat an authorization URL as a credential; the URL may be returned only by an explicitly requested auth command/query/subscription, and the PKCE verifier is never returned.

Use operation commands to inspect or await work, or request supported cancellation:

```powershell
puripuly.exe operation status <operation-id>
puripuly.exe operation wait <operation-id>
puripuly.exe operation cancel <operation-id>
```

Active model installation and Gemma preparation are the cancellable long-running operations. Other operation types may report `cancellation: unsupported`; the CLI exits 4 and leaves a still-running operation marked nonterminal. Cancellation is not a rollback: a settings commit or already-applied runtime change remains committed. The operation receipt describes what has already completed.

If shutdown interrupts a settings operation after a durable save, its receipt retains the committed transaction and that commit's revision. `transaction.status: settings_commit_success_runtime_interrupted` means settings persisted but runtime completion was not established before interruption; a known runtime-applied or runtime-degraded transaction is retained instead when available. A queued operation that never committed has no committed transaction. These fields report completion metadata, not private setting values. Shutdown does not rewrite an already completed receipt.

Mutation requests have a caller `request_id`, which the client generates unless `--request-id` is supplied on `command`. The host deduplicates a matching request identity and payload while that host retains it; reusing an ID with different content is rejected. Up to 256 operation/request identities are retained per host and terminal records may be evicted when that bound is reached. The instance UUID changes on restart, so operation IDs from a previous host cannot be resumed against the new one. If a response is lost or the client times out, execution may be unknown; inspect host and operation state before deciding what to do. The CLI never automatically replays an ambiguous mutation.

## Credentials, authorization, and consent

Provider secrets are supplied via hidden terminal prompts or stdin, never as regular command-line arguments:

```powershell
puripuly.exe secrets set soniox_api_key
Get-Content -Raw .\secret.txt | puripuly.exe secrets set soniox_api_key --stdin
puripuly.exe secrets presence
puripuly.exe secrets verify soniox_api_key
```

Verification reads a secret using the same hidden-input/stdin rule and is separate from storage. The declared secret identifiers are listed in `capabilities` and `secrets presence`; loaded HTTP-extension declarations are included. A key without a supported verification protocol returns terminal `action_required` (`verification: unavailable`), not a fabricated verification success. Secret values are never returned in settings, query results, operation receipts, or logs. Delete is local storage deletion.

Authentication commands are explicit:

```powershell
puripuly.exe auth login qq
puripuly.exe auth login discord
puripuly.exe auth login openrouter
puripuly.exe auth login chatgpt --open-browser
puripuly.exe auth login discord --open-browser
puripuly.exe auth logout discord
puripuly.exe auth logout chatgpt
```

Browser opening is off by default. `--open-browser` is the sole CLI permission to launch a browser. QQ uses hidden prompts for its identity and credential, or accepts a protected JSON object on stdin with the supported fields `qq_identity`, `credential`, and optional `referral_id`. Discord/OpenRouter accept optional `referral_id` only where allowed; OpenRouter does not accept one. For automation, use the dedicated `auth login ... --stdin` protected JSON input rather than generic arguments. A login request itself is the explicit authorization action; translation toggles, settings/provider changes, generic commands, and OSC translation commands must not initiate OAuth implicitly. Existing authorization may still be used for normal operation.

ChatGPT login authorizes GPT 6 Luna on the `chatgpt` translation connection with the user's ChatGPT plan. It does not change the selected translation model or connection. `auth status` reports `chatgpt.signed_in`, `chatgpt.in_progress`, and `chatgpt.sign_in_required` without the account email.

Account logout is local-only and does not revoke a remote provider grant, except ChatGPT logout, which also requests revocation of the refresh token and reports `scope: remote_revoked` or `scope: local_only` when revocation was not confirmed. Logging out an inactive account preserves unrelated BYOK, local, or other-account translation; only the affected active route is stopped/rebuilt. Failed logout persistence does not silently leave that route's translation disabled. Authentication and model operations can require a human step or missing entitlement and report `action_required`. The CLI never auto-accepts consent. Peer terms are available through `capture terms`; an informed user can explicitly accept them when enabling peer capture as described above.

## Output, events, logs, and privacy

- CLI stdout is UTF-8 JSON with `output_version: 1`; diagnostics and structured errors are emitted as JSON on stderr. Follow commands emit one JSON event per line.
- The local transport is protocol v1 newline-delimited UTF-8 JSON (not pickle) over ephemeral IPv4 loopback (`127.0.0.1`). It authenticates the discovered instance UUID and capability token (constant-time token comparison) before catalog dispatch; no arbitrary method/reflection or code execution is exposed.
- Windows instance discovery uses a current-user ACL-verified record under the user's local application data and an exclusive `LockFileEx` lease for the canonical settings identity. The endpoint/token are not an unauthenticated management service. Requests are limited to 65,536 bytes including newline; responses are limited to 1,048,576 bytes. The transport validates protocol, instance and request correlation.
- Queries and default event/log views omit credentials and conversation content. Recognition and translation content subscriptions require both the corresponding event topic and explicit `--include-transcripts` / `--include-translations` selection. Non-content topics expose typed metadata only; `osc_sent` never includes chatbox text, even with both content flags enabled. Keep those flags off unless the caller needs content and can protect the output.
- Subscribers have bounded independent buffers. Slow consumers do not block capture/translation; event sequence gaps are explicit. Use `--after` to resume and refresh a snapshot with queries after a gap. Ordering is per owner/topic/channel; do not infer one global order across self, peer, UI, and provider streams.
- `events follow` defaults to operation, settings, session-state, error and gap topics. Topics can be selected with repeatable `--topic`; `--channel` filters self or peer events. For content, explicitly add `--topic transcript` or `--topic translation` and the corresponding include flag.
- `logs follow` defaults to warning and follows sanitized application metadata; choose `--level debug|info|warning|error` to change the threshold. Logs are bounded diagnostics, not a durable audit trail, and do not contain transcript/translation bodies or credentials.

```powershell
puripuly.exe events follow --topic operation --topic settings
puripuly.exe events follow --topic transcript --channel self --include-transcripts
puripuly.exe events follow --topic translation --channel peer --include-translations
puripuly.exe logs follow --level warning
```

## Exit codes

| Code | Meaning |
| ---: | --- |
| 0 | Applied result, successful query/start/stop, or accepted non-waiting request. |
| 2 | Command-line syntax/argument parsing error; help itself exits 0. |
| 3 | Invalid input, protocol/transport error, or unavailable instance. |
| 4 | Rejected, degraded, persistence failure, operation failure, or unsupported cancellation. |
| 5 | Action required, including an interim OAuth challenge (`terminal:false`). |
| 6 | Cancelled or interrupted terminal operation. |
| 7 | Execution or startup outcome is unknown, operation is still nonterminal when waiting, or host identity disappeared ambiguously. |
| 130 | Keyboard interrupt (`Ctrl+C`). |

An exit code does not replace the JSON receipt: inspect `terminal`, `status`, operation/instance IDs, and runtime state before automation proceeds.
