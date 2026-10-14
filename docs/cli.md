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
- `app stop` waits until the original instance endpoint is gone and the observed original process identity has exited. Its `cleanup_outcome: unobserved` distinguishes that process-stop observation from successful completion of every teardown callback, file flush, or resource release. `app stop --no-wait` returns the accepted shutdown receipt instead, also without a cleanup-success claim. It never kills a process by PID.
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
puripuly.exe query diagnostics.history
puripuly.exe query diagnostics.health
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

The receipt reports `admission`, the pipeline's existing `turn_id` and `child_turn_ids`, and `output_delivery: unobserved`. The existing bounded completion wait manages typing; it is not a destination acknowledgment and does not establish translation/output success. Later translation/output failures retain the explicitly admitted operation/request relationship, so query diagnostic history for the original operation and inspect the affected destination separately. An applied admission is not retroactively rewritten as successful translation or display.

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

GPT 6 Luna (`translation.model: gpt_6_luna`) sends `temperature: 0.6` with reasoning effort `none` on the `openrouter` and `official_byok` connections. The `chatgpt` connection omits `temperature`. These request parameters are fixed, not editable CLI settings.

The finite command catalog also includes Soniox diarization, managed referral settings, custom and local provider settings, GPU device selection, both-channel provider updates, free-tier provider choices, Qwen region, model/connection history, prompts, languages, vocabulary, VAD, audio devices/host APIs, overlay values, OSC and telemetry settings. Use `capabilities` for field names and types, and `settings choices` for supported values rather than reproducing dynamic lists.

In an already-running GUI, focused external changes preserve unrelated provider/prompt drafts; overlapping changes require explicit conflict resolution. A successful pending GUI apply acknowledges only its submitted values. Newer edits made while it was pending remain staged, and a failed apply retains its draft for retry.

## Named query and command fallback

Convenience domains cover the common operations. When scripting a less common supported operation, use the finite catalog rather than private Python APIs:

```powershell
puripuly.exe query <query-name> [--arguments '<JSON object>' | --file <query.json>]
puripuly.exe command <command-name> [--arguments '<JSON object>' | --file <arguments.json>] [--expected-revision <revision>] [--no-wait]
```

The catalog rejects unknown names; it is not RPC reflection. `capabilities.query_arguments` describes the finite argument shapes for named queries; queries without declared arguments reject them. `command` accepts only names and argument shapes returned by `capabilities`. Use `--file` for JSON payloads that include sensitive or private content. Protected operations have dedicated `secrets`, `auth`, and `text` commands; the generic command form deliberately rejects `secrets.set`, `secrets.verify`, `auth.login`, and `text.submit`.

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

Verification is a limited key/protocol probe, not proof of healthy live recognition: diagnostic metadata distinguishes `verification_scope: key_probe` from `recognition_readiness: unobserved`. A failed probe preserves known typed transport/status causes; an unsuccessful boolean alone is not evidence of invalid credentials. Probe timeout-success behavior, where established by a provider protocol, does not establish successful recognition.

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

QQ authorization that is still awaiting managed-operation recovery reports `action_required` with `reason: resolve_pending_authorization` and `transaction.status: remote_delivery_ack_pending`, including when its message identifies a timeout, proxy, TLS, DNS, or other network failure. Inspect the transaction status rather than interpreting a network message as rejected credentials. Pending delivery acknowledgment does not prove that a usable key was saved locally: a missing or unreadable local key receives separate storage feedback and is not acknowledged to the Broker. Neither pending condition is a reason to claim another key.

ChatGPT login authorizes GPT 6 Luna on the `chatgpt` translation connection with the user's ChatGPT plan. It does not change the selected translation model or connection. `auth status` reports `chatgpt.signed_in`, `chatgpt.in_progress`, and `chatgpt.sign_in_required` without the account email.

When secret observation fails, `signed_in` is `null` rather than a fabricated signed-out result; inspect its diagnostic reference. Local credential deletion and confirmed remote revocation remain separate outcomes. Secondary listener, storage, or notification failures do not replace the original authentication cause or retroactively invalidate a completed durable settings transition.

Account logout is local-only and does not revoke a remote provider grant, except ChatGPT logout, which also requests revocation of the refresh token and reports `scope: remote_revoked` or `scope: local_only` when revocation was not confirmed. Logging out an inactive account preserves unrelated BYOK, local, or other-account translation; only the affected active route is stopped/rebuilt. Failed logout persistence does not silently leave that route's translation disabled. Authentication and model operations can require a human step or missing entitlement and report `action_required`. The CLI never auto-accepts consent. Peer terms are available through `capture terms`; an informed user can explicitly accept them when enabling peer capture as described above.

## Output, events, logs, and privacy

- CLI stdout is UTF-8 JSON with `output_version: 1`; diagnostics and structured errors are emitted as JSON on stderr. Follow commands emit one JSON event per line.
- The local transport is protocol v1 newline-delimited UTF-8 JSON (not pickle) over ephemeral IPv4 loopback (`127.0.0.1`). It authenticates the discovered instance UUID and capability token (constant-time token comparison) before catalog dispatch; no arbitrary method/reflection or code execution is exposed.
- Windows instance discovery uses a current-user ACL-verified record under the user's local application data and an exclusive `LockFileEx` lease for the canonical settings identity. The endpoint/token are not an unauthenticated management service. Requests are limited to 65,536 bytes including newline; responses are limited to 1,048,576 bytes. The transport validates protocol, instance and request correlation.
- Queries and default event/log views omit credentials and conversation content. Recognition and translation content subscriptions require both the corresponding event topic and explicit `--include-transcripts` / `--include-translations` selection. Non-content topics expose typed metadata only; `osc_sent` never includes chatbox text, even with both content flags enabled. Keep those flags off unless the caller needs content and can protect the output.
- Subscribers have bounded independent buffers. Slow consumers do not block capture/translation; event sequence gaps are explicit. Use `--after` to resume and refresh a snapshot with queries after a gap. Ordering is per owner/topic/channel; do not infer one global order across self, peer, UI, and provider streams.
- `events follow` defaults to operation, settings, session-state, error and gap topics. Topics can be selected with repeatable `--topic`; `--channel` filters self or peer events. For content, explicitly add `--topic transcript` or `--topic translation` and the corresponding include flag.
- `logs follow` defaults to warning and follows sanitized application metadata; choose `--level debug|info|warning|error` to change the threshold. Failure events carry the same safe diagnostic record available from retained history. Use `--after <event-sequence> --instance-id <originating-host>` for scoped replay; without `--after`, only new events are followed. `events follow` also accepts `--instance-id`. Neither interface returns mixed-content main-file bytes.
- CLI `logs` events are not a transcript of every GUI Basic row or main-file line: typed retained diagnostics publish metadata events, while ordinary Basic/file-only observations do not automatically enter that stream. Compare those audiences separately.
- Basic output logs omit typing-control calls, expected policy/duplicate/stale decisions, and duplicate summaries of typed failures. Typing transitions and message submission still run; failures remain available through the owning diagnostics. Overlay admission/pacing records are file-only; surviving output milestones name the observed handoff, local send, or application stage, never remote receipt or physical display. Owner overlay transitions and nonempty discarded chatbox backlogs remain visible.
- QQ authentication also emits bounded `[QQAuth]` diagnostics through the normal live and persisted application logs. Successful committed recovery is logged as success even without a user-facing message; local-key absence is distinguished from a pending remote delivery acknowledgment. These records use allowlisted stages, codes, and available HTTP/transport classifications, not raw exceptions, response bodies, QQ identities, credentials, keys, tokens, or proxy URLs.

```powershell
puripuly.exe events follow --topic operation --topic settings
puripuly.exe events follow --topic transcript --channel self --include-transcripts
puripuly.exe events follow --topic translation --channel peer --include-translations
puripuly.exe logs follow --level warning
```

## Retained diagnostics and delivery health

An independently attached client can inspect a prior failure without subscribing before it happened:

```powershell
puripuly.exe query diagnostics.history
puripuly.exe query diagnostics.history --arguments '{"operation_id":"<operation-id>","instance_id":"<originating-instance-id>","limit":10}'
puripuly.exe query diagnostics.history --arguments '{"diagnostic_id":"<diagnostic-id>"}'
puripuly.exe query diagnostics.history --arguments '{"after":42,"limit":100}'
puripuly.exe query diagnostics.health
puripuly.exe operation status <operation-id>
```

Use real identities and cursors returned by the host, not the example placeholders. `diagnostics.history` accepts only optional `operation_id`, `diagnostic_id`, `instance_id`, `after`, and `limit`. `diagnostics.health` takes no arguments. Both use the existing authenticated same-user local connection; they neither grant mutation authority nor enable conversation access.

Operation and diagnostic identity payloads use 8–64 ASCII hexadecimal digits or UUID-shaped `8-4-4-4-12` groups, optionally with an existing native prefix such as `overlay-`. Short native hexadecimal identities remain valid; identities are not restricted to UUIDs. Hyphen-only payloads or trailing separators such as `--------` and `dead----` are malformed, not missing retained evidence. Host `instance_id` uses its separate instance-name syntax.

### Identity and interpretation

Schema version 1 records contain a distinct `diagnostic_id`, UTC `occurred_at`, current `instance_id`, `diagnostic_sequence`, repository-owned component/operation/stage, severity, actual disposition, stable code/category, and known cause/operation/request/channel/turn/generation identities. Unavailable relationships are `null`, not inferred from nearby events. Allowlisted native/status/provider details and bounded exception-class cause relationships may be present; raw exception messages, stack traces, arbitrary child JSON/stderr, response bodies, credentials, URLs, personal paths, audio, and conversation content are excluded.

`detail.state`, `detail.rejected_fields`, and `detail.truncated` disclose incomplete optional evidence. A rejected detail must not hide an otherwise permitted code or identity. Repeated projection of the same owning diagnostic is deduplicated; a new operation or generation is not silently joined to the previous failure. Unknown defects remain explicitly unknown and identify the safe owning boundary rather than a guessed diagnosis.

Disposition is one of `failed`, `degraded`, `recovered`, `cancelled`, `superseded`, `disabled`, `unknown`, or `action_required`. It describes an observation, not permission to retry. Owner state and terminal receipts remain authoritative: persistence failure, committed-but-degraded runtime application, cancellation, expected empty recognition, and disabled destinations are not interchangeable. Primary and bounded secondary cleanup causes remain separate. Local UDP send success does not acknowledge remote delivery; output admission does not establish physical display.

Receipt projections retain the first four ordered `secondary_diagnostics`. `secondary_omitted_count` adds known omissions from the owning result and any later receipt or notification projection; `secondary_truncated` is true when that count is nonzero. The primary diagnostic remains separate and is not replaced by a secondary failure or truncation. Unobserved or unavailable detail is not counted as a known omitted cause. These cause-list bounds are distinct from an individual diagnostic's `detail.truncated` field.

The same record is available as `diagnostic` on `logs` events with `kind: failure`; operation events carry diagnostic references. **Diagnostic history cursors and event replay cursors are different sequences.** Use history's `next_after` only for `diagnostics.history`; use an event's `sequence` only for follow/replay.

`gpu.status.runtime.startup_cleanup` reports pending startup cleanup, confirmed completion, and bounded safe cleanup failures. Missing owner evidence is unavailable, not proof that an absent worker was fully closed. `app.status.scene_observation` distinguishes a successful process probe from a failed observation; a failed enumeration cannot establish that VRChat is not running. Neither query exposes process command lines, participant identities, or raw scene logs.

Existing ASR model installation failures retain `local_asr_provisioning` / `models.install` / `local_asr_install_failed` evidence with the validated model ID, installation generation, and originating operation/request IDs when known. Exhausting the existing source fallback preserves the first native cause; typed HTTP status and transport classification are retained without parsing exception prose. Query by the install operation ID or replay `logs` to inspect this file-only technical evidence. Basic installation summaries, model integrity, terminal install status, source fallback and cancellation behavior are unchanged; the diagnostic does not authorize a retry.

### Bounds, gaps, and restart

| Surface | Bound |
| --- | --- |
| One diagnostic record | 4,096 encoded metadata bytes; at most 32 optional producer fields |
| Secondary cause projection | At most four; additional causes report truncation |
| Current-instance diagnostic history | At most 256 records and 512 KiB encoded metadata |
| History query | Default 50 records, maximum 100 |
| Event replay history | 256 entries |
| Subscriber wake buffer | 32 entries; wakeups coalesced independently per subscriber |
| Cross-thread diagnostic dispatch | 256 pending metadata events |
| Existing main-file queue | 2,048 records, including 64 priority slots |
| Existing file retention | 20 MiB main file and one backup |

History returns records in ascending diagnostic sequence order. Inspect `oldest_sequence`, `latest_sequence`, `next_after`, `gap`, `evicted_records`, `truncated`, and `retention`; a bounded result is not proof of complete evidence. History is in memory for the current host, resets on restart, and is not a database or durable audit. Retain the originating instance alongside any operation or diagnostic ID.

Malformed arguments, negative cursors, invalid identity syntax, unknown keys, and limits outside 1–100 produce `invalid_arguments` and CLI exit 3. A validly shaped future cursor returns lookup `invalid_cursor`; a different host identity returns `foreign_instance`. Both are successful queries (exit 0) with empty records and an explicit gap, not evidence that no failure occurred. `not_retained` means no matching retained evidence, not that an operation succeeded or never failed. Unavailable history is explicit. Follow/replay gaps identify eviction, foreign-instance, or invalid-cursor conditions and request a fresh snapshot.

### Delivery health is not persistence acknowledgment

`diagnostics.health` reports `history`, `events`, `file`, and available output-owner delivery observations independently of successful file writing:

- History reports availability, retained bytes/records, eviction, rejected detail/records, duplicate projections, subscriber/file projection failures, and bounded cross-thread dispatch pressure.
- Events report retained sequences, subscriber count, eviction, gaps, and disconnected consumers. Slow consumers have independent bounded work and do not backpressure application producers.
- File reports writer availability, queue capacity and pending work, enqueued/written/dropped/rejected records, delivery failures, writer liveness, and time since progress. `shutdown_attempt_complete` is distinct from confirmed `shutdown_complete`; cleanup failures remain visible. Pending work includes in-flight/buffered unfinished records. `stalled` means a pending episode without observed writer progress for at least two seconds; it is not a guessed disk diagnosis.
- Output reports bounded owner-history retention/eviction and coalesced observer failures, including the first `observer_failure_diagnostic_id` and current failure count. A failed optional observer does not retroactively fail translation or acknowledge a destination.

File counters cover the existing whole main-file stream, including its permitted conversation records. `persistence` distinguishes `pending`, `incomplete`, `unknown`, and `written_not_durable`; `durable` is false. Enqueued is not written, and written is not an fsync/durability claim. A record's `file_evidence` is `unknown` unless unavailability is established; aggregate counts cannot establish persistence of that individual record. Failed-file and overflow episodes expose `delivery_diagnostic_id` or `loss_diagnostic_id` through retained history without recursively writing another record to the failed sink. Their file evidence is unavailable. Bounded shutdown, queue loss, failed storage, and uninstrumented sinks must not be interpreted as lossless logging.

The live Basic/Logs audience remains concise and factual; selected technical metadata is file-only and available through the metadata CLI. Detailed mode is not restored. Existing explicitly permitted conversation logging remains separate, so the main log still requires review and redaction before sharing.


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
