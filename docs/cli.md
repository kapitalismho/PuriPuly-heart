# Command-line control

PuriPuly's CLI controls the same application owners used by the desktop GUI. Use it to:

- Start a GUI-free host.
- Attach to an already-running GUI or headless host.
- Inspect the active runtime.
- Submit typed application operations.

The CLI is not a second ASR/translation runtime. It does not edit the settings JSON directly.

## Choose the entry point

### Installed Windows command

Use the installed console command `puripuly.exe` (or `puripuly` when its install directory is on `PATH`):

```powershell
puripuly.exe --help
puripuly.exe app start --background
puripuly.exe app status
puripuly.exe app stop
```

The package declares the `puripuly` console entry point. Its CLI contract is:

- JSON on stdout.
- Diagnostics on stderr.
- Documented process exit codes.

### Development checkout

Run the same CLI through the Python module entry point:

```powershell
python -m puripuly_heart.main cli --help
python -m puripuly_heart.main cli --config C:\work\puripuly-dev.json app start --background
python -m puripuly_heart.main cli --config C:\work\puripuly-dev.json app status
python -m puripuly_heart.main cli --config C:\work\puripuly-dev.json app stop
```

- Use the same `--config` path for a host and each client that should reach it.
- The path identifies canonical settings-file ownership.
- Without it, the installed or development CLI uses the current user's normal settings file.
- A running GUI and a headless host use the same settings identity and control endpoint.

## Host lifecycle

`app start` explicitly launches a GUI-free host:

- It runs in the foreground by default. `--foreground` makes that choice explicit.
- `--background` starts a detached service that outlives its launching terminal.

```powershell
puripuly.exe app start --foreground
puripuly.exe app start --background
puripuly.exe app list
puripuly.exe app discover
puripuly.exe app status
puripuly.exe app restart --background
puripuly.exe app stop
```

- **Instance ownership**
  - Only one host may own a canonical settings-file identity for the current user.
  - A duplicate start is an explicit error, not a second competing capture or settings writer.
- **Discovery and status**
  - `app list` enumerates discoverable current-user instances.
  - `app discover` and `app status` query the instance matching `--config`.
  - They report its instance identity, process ID, channel state, and translation state.
  - The PID is observation only. It is not the control identity or a signal target.
- **Restart and missing hosts**
  - `app restart` stops the instance for that settings identity before starting it again.
  - A query or mutation does not launch a missing host implicitly.
  - A missing instance returns `instance_not_found`. Start one deliberately with `app start`.
- **Stopping**
  - `app stop` waits until the original instance endpoint is gone and the observed original process identity has exited.
  - `cleanup_outcome: unobserved` reports that process-stop observation. It does not confirm completion of every teardown callback, file flush, or resource release.
  - `app stop --no-wait` returns the accepted shutdown receipt without a cleanup-success claim.
  - The CLI never kills a process by PID.
- **Terminal and child-process lifetime**
  - `Ctrl+C` interrupts a foreground host/client with exit code 130.
  - A background host is independent of the short-lived launch client.
  - On Windows, the host retains its kill-on-close child-process Job Object. The CLI client does not inherit that containment.

Direct startup and launcher behavior:

- `python -m puripuly_heart.main run-headless --config <settings-file>` explicitly starts the GUI-free runtime.
- The CLI form `app start --foreground` is equivalent for control use and establishes local discovery.
- The legacy native `--headless` launcher dispatch is not a service contract. It does not itself define an application-control host.
- Python's no-command fallback starts the GUI. Ordinary no-command application launch remains GUI startup.

## Discover commands, values, and state

JSON is the default CLI output. There is no `--json` switch.

- Use `capabilities` for the active host's command argument descriptions, query names, setting fields, secret-key identifiers, and event topics.
- Use `settings choices` for runtime-supported provider/model/connection/region/overlay values and other finite catalog choices.
- Values may depend on loaded extensions or available devices. Discover them from the running host rather than copying a stale list into a script.

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

Use `capabilities` to discover supported query names.

- Queries are side-effect-free observations.
- Discovery commands such as audio-device enumeration may read the current desktop/audio environment. They do not enable capture or start recording.

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

Capture/provider/output states in command receipts are runtime observations, not just persisted selections.

In `app.status`, `translation_runtime_ready` means a provider is attached; it can be true while translation is disabled or a managed model has not loaded. Check the operation receipt and `models.status` for managed preparation success or failure. If the current translation-enable attempt fails or is cancelled during warmup, effective translation turns off while the selected model and failure details remain. Superseded work does not change newer intent. After correcting the cause, explicitly run `translation set on` again.

### CPU model repair

To repair an outdated Qwen3-ASR 0.6B CPU installation and verify the downloaded files:

```powershell
puripuly.exe models install --backend cpu --model-id qwen3-asr-0.6b-int8-sherpa
puripuly.exe models prepare --backend cpu --verify-checksums
puripuly.exe models status
```

- The corrected decoder uses a new pinned source revision.
- An older installed revision is invalid until model repair replaces it. Restarting alone does not update the files.
- Installation reports progress through the normal application owner.
- It publishes `ready` only after validation and staged replacement complete.

### Peer capture and consent

Peer capture requires informed, explicit acceptance of the existing terms:

1. Read the terms with `capture terms` (query `consent.peer_translation`).
2. If consent is not already stored, the user may explicitly run `capture set peer on --accept-peer-terms`.

The flag is valid only when enabling the peer channel. It routes through the existing GUI acceptance owner and is never auto-supplied by the CLI.

The finite fallback is equivalent only when it carries the same explicit consent field:

`puripuly.exe command capture.set --arguments '{"channel":"peer","enabled":true,"accept_terms":true}'`

- `accept_terms` is invalid for self capture or a peer-off request.
- Inspect the terms first. Pass `true` only for the user's explicit informed choice.

### Qwen Audio handover and Peer recovery

- **Local completion and publication**
  - Directly selected Qwen Audio assigns each local segment to one native task. A local end sends `finish-task`; the next segment can use the other connection without waiting for the preceding task's result. Its PCM waits for its own `task-started`.
  - Peer retains its four/five-second pause opportunities and six-second hard boundary. It collects distinct native final sentences and drain tails into one completed local-segment result before translation admission. Native sentence endings, partials and `fixed` values do not independently trigger Peer translations.
  - SELF shares the handover mechanism but retains its own VAD endpoint and stable-contribution/low-latency merge policy. It does not gain Peer's six-second cut. Successor contributions remain ordered behind their predecessor; manual input does not wait for ASR drain.
  - The native silence parameters remain 800 ms for Peer and 6000 ms for SELF. They are not the app's local endpoint policy or an end-to-end latency guarantee. Handover removes predecessor waiting where a slot is available; it does not remove initial readiness, recognition, merge or translation time.
- **Audio scope and limits**
  - Each channel has at most two connections and two in-flight local turns; using both channels can use four connections. The connection limit includes current, retained and retiring Qwen owners: healthy matching-scope sockets may be reused, while another scope or a replacement after retirement waits for confirmed old cleanup before admitting native tasks. Their queues, settings and result authority are separate. If both slots are occupied, subsequent input waits within the existing bounded capture queue and operation/retention limits.
  - Qwen receives locally selected speech/context/tail audio rather than an all-capture stream between local segments. Capture permission, source selection and mute/activation gates still apply. Selected context and tails can contain background audio or pauses.
  - Existing onset context and applicable 300 ms Peer continuation context are retained. Context can overlap preceding input, but handover adds no second replay, whole-stream copy or fabricated keepalive audio. Normal recognition starts tasks on demand, not empty standby tasks. Explicit key verification retains a bounded no-PCM permission probe.
  - Each task retains at most 256 native sentences, 4096 source-map records and 256 KiB of text/identity/timing/map evidence. A native readiness attempt, PCM/finish write and final completion each retain their five-second defaults. Readiness retries have a separate existing budget, described below; five seconds is not an aggregate retry guarantee. These are failure bounds, not added healthy-path delays. Source/VAD-idle retirement remains 60 seconds. Failure or expiry is not successful empty recognition, and uncertain writes are not replayed.
  - Equal locally selected PCM and task/request counts do not establish equal token charges. The restored local boundaries/context differ from the previous continuous Peer candidate; real-service concurrency, latency and billing require separate qualification.
  - Other providers retain their own policies. Peer output remains excluded from the VRChat chatbox.

- **Transient connection failures**
  - Initial Qwen session admission on either channel retains up to three safe pre-PCM readiness attempts within its 30-second budget, with 0.8/1.6-second retry backoffs.
  - SELF retains its existing stop-on-failure behavior after initial retries are exhausted. It does not gain Peer's cooldown or automatic retries for later task failures.
  - Peer also applies its readiness recovery to successor tasks. Capture remains enabled when temporary connection timeouts, network failures, service unavailability, or rate limits exhaust its three short attempts.
  - A failed attempt must be safely retired before retry; unresolved cleanup is not permission to reuse its socket or native task identity. An unresolved session remains owned and contributes cleanup debt. That channel must reclaim old Qwen resources before constructing or admitting another Qwen owner, including explicit re-enable and return from another provider. Failed reclamation retains the typed runtime cause `provider_resource_quarantined`; the public capture receipt keeps its existing generic degraded-provider-failure projection. Other-provider, other-channel and manual work remain independent. Actual audio/finish writes retain their separate limits, and capture queue/expiry limits remain active during readiness.
  - Peer waits 30 seconds before allowing another bounded attempt sequence on the next utterance.
  - Silence does not trigger background reconnects.
  - Failed or expired utterances are not replayed after recovery.
- **Stopping and invalidation**
  - Authentication, configuration, certificate, and unclassified failures stop the channel rather than enter this recovery loop.
  - Turning capture off or applying an invalidating provider configuration revokes the affected channel's old Qwen authority, including both slots and queued updates. Late old-task text cannot publish under the replacement scope. Drain-intended EOF/idle completion remains distinct from abort.
  - A capture discontinuity also cancels pre-gap queued input and revokes both old slots. Its control notification is delivered independently of blocked audio dispatch; this does not enable all-capture PCM submission for Qwen. Deferred results recheck authority before consumption, so not-yet-admitted old text cannot enter the merge or translation owner. Previously admitted speech remains subject to the existing channel policies. Fresh post-gap input uses new recognition authority.
- **Recovery observations**
  - During the wait, `capture status` can report active audio capture without implying that recognition is available.
  - Retained recognition diagnostics expose `recovery_pending` and `retry_after_ms`.
  - Inspect them through `query diagnostics.history` or `logs follow --level warning`.
  - Successful recognition ends the recovery episode and preserves its cause link.

### Overlay and OSC state

**Overlay**

`overlay status` returns the owner-backed projection in its top-level `output` object:

- Desired, configured, effective, and attempting target.
- Lifecycle and runtime/process state.
- Presentation readiness and desktop visibility.
- Generation, failure/recovery, and ingress-stop state.

Interpretation:

- `overlay set` waits for the explicit owner transition.
- A configured target alone does not establish readiness.
- Future automatic recovery does not make a failed attempt successful.
- `desktop_visible` is `null` for non-desktop targets.

**OSC**

`osc status` also returns a top-level `output` object:

- Configured and effective local mode/ports are separate.
- Effective values are `null` without their runtime resource.
- The projection reports receiver/sender/service availability and discovery/failure state.
- OSC/UDP `remote_delivery` is always unacknowledged. It does not prove that VRChat received a packet.

### Overlay calibration

`overlay calibrate change --field FIELD --value VALUE` accepts:

- `anchor`: a choice from `settings choices` (`overlay.calibration_fields.anchor`).
- Numeric fields: `offset_x`, `offset_y`, `distance`, `text_scale`, and `background_alpha`.

The numeric fields are parsed as finite numbers. Distance and text scale must be positive; alpha must be between 0 and 1.

For example:

1. Begin a calibration.
2. Run `overlay calibrate change --field offset_x --value 0.05`.
3. Apply or cancel the calibration.

Numeric CLI values are sent as JSON numbers. The application owner validates field-specific bounds.

### Microphone test

- `microphone test on` stays in progress until the owner receives an actual input frame. Scheduling a capture task is not enough.
- `app status` returns a `microphone_test` object with:
  - `state`, `desired_active`, and `effective_active`.
  - Normalized `meter_level` (0–1).
  - `failure_reason` and `failure_type`.
- No audio content is returned.
- An unavailable input produces `action_required`.
- `microphone test off` can interrupt pending startup. It waits for owned input resources to close.
- Testing takes exclusive self-input ownership and can disable normal self capture. Re-enable that capture explicitly afterward if needed.

### Manual text

Manual text goes through the normal self transcript/translation/output pipeline, without microphone capture or STT. Do not put conversation text in a process argument:

```powershell
puripuly.exe text submit --file .\message.txt
Get-Content -Raw .\message.txt | puripuly.exe text submit --stdin
```

- The receipt reports `admission`, the pipeline's existing `turn_id` and `child_turn_ids`, and `output_delivery: unobserved`.
- The existing bounded completion wait manages typing. It is not a destination acknowledgment and does not establish translation/output success.
- Later translation/output failures retain the explicitly admitted operation/request relationship.
- Query diagnostic history for the original operation and inspect the affected destination separately.
- An applied admission is not retroactively rewritten as successful translation or display.

## Typed settings and provider changes

- Read the current `revision` through `settings.current`.
- `settings apply` accepts one UTF-8 JSON object from `--file`, or a non-sensitive object from `--arguments`.
- It also accepts an optional `--expected-revision`.
- The JSON may wrap focused updates in `changes`. Unknown fields and invalid types are rejected.

The application:

1. Materializes updates as typed settings/provider intents against canonical settings.
2. Persists them through the settings owner.
3. Applies runtime effects through the owning runtime boundary.

A direct JSON file rewrite is not a settings operation.

- Revisions and settings events follow successfully committed changes, including GUI-only edits.
- Queries return the committed snapshot without advancing its revision.
- Staged preparation and failed persistence are not reported as persisted settings.

### Activation notice

- `chatbox.activation_notice.enabled` is a boolean preference, enabled by default.
- It is persisted as `intent.osc.activation_notice_enabled`.
- Setting it to `false` suppresses the Talk `PuriPuly ON!` message and Listen audio-capture disclosure.
- This does not disable capture or normal translation output.
- The preference applies immediately without a runtime restart.
- Changing it to `true` does not itself send or replay a notice.
- Peer capture still requires explicit acceptance of its terms.

### Settings examples

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

- `12` is an example revision, not a fixed value. Substitute the current result's revision.
- If another settings mutation changes that revision first, the operation is rejected with a revision conflict.
- Focused changes preserve unrelated settings.

### ASR provider changes

For ASR-only changes, use the convenience form or the command fallback:

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

- Use `asr set` with `--channel self|peer|both` for the same STT provider on the requested channels.
- Use settings fields when the two channels need different providers.
- `settings choices` is the authority for available values and valid model/connection combinations.
- A successful persistence receipt does not prove that a provider is attached or active.
- Inspect `asr status` / `providers.status` and its selected, runtime, capture-attached, activity, pending-handoff, and failure fields.

### Index public translation

Apply `translation.model: index_translate_35b` and `translation.connection: public_api` together through `settings apply`. This connection needs no API key and is the default for new installations; existing selections are preserved.

- Startup and Index selection check availability in the background. `translation set on` waits for a pending check.
- If Index is unavailable, the app saves a switch to `gemma4_26b_31b` / `managed`. It selects `openrouter` instead only when the applicable Managed allowance is confirmed exhausted. The saved selection survives restart; selecting Index again is explicit.
- Inspect `settings current`, `providers.status` and the terminal operation result after a switch. The destination may require authentication or key setup; noninteractive callers can receive `action_required`. A saved selection does not establish translation readiness.

### Translation request parameters

GPT 6 Luna (`translation.model: gpt_6_luna`) uses fixed request parameters:

- On `openrouter` and `official_byok`, it sends `temperature: 0.6` with reasoning effort `none`.
- The `chatgpt` connection omits `temperature`.
- These parameters are not editable CLI settings.

### Managed local translation

Discover current values with `settings choices` before applying them:

| `translation.model` | Supported `translation.connection` | Managed model |
| --- | --- | --- |
| `managed_gemma` | `cpu`, `gpu` | Gemma 4 E4B IT QAT; existing selection/defaults retained |
| `managed_gemma_26b` | `gpu` only | Gemma 4 26B A4B IT QAT |

Apply model and connection together through `settings apply`. The normal settings owner persists the selection; observe it again after a fresh process restart. `gemma.prepare` prepares the selected model, and `gemma.cancel` cancels ongoing preparation. Use `models status` for the managed Gemma owner, not ASR-specific `gpu status`. A completed preparation is not evidence of a completed translation; correlate `text submit` admission with the translation events.

Models download on demand with progress, cancellation, checksum validation and atomic installation. No weights are bundled with the installer. The 26B target is `unsloth/gemma-4-26B-A4B-it-qat-GGUF` revision `7b92b5b28818151e8669af2e45e88d6086f490dd`, file `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf`, 14249047104 bytes, SHA-256 `a7c5bc715f5ff8e99a3e8901ce7d2b42b402c669bf24f7c5250747633d0f5891`. Chinese UI locales download that file from ModelScope revision `f5f82bed79bc0bc49b01e0e577bb3dc949ebc564` first. Other locales try that mirror after Hugging Face fails. The installed manifest keeps the Hugging Face revision. It has no draft model or multimodal projector. E4B retains its existing pinned target and CPU MTP drafter.

The managed runtime is pinned to llama.cpp **b11537 / `8e2d31e0eb658d0382d3106a61369dff81f0c080`**. Source runtime files live under `build/llama.cpp/llama.cpp-b11537/{cpu,vulkan}`; installed files live under `_runtime/llama.cpp-b11537/{cpu,vulkan}`. `PURIPULY_HEART_LLAMA_CPP_ROOT` may select a compatible root with those backend subdirectories, but an incompatible executable build/commit is rejected. A missing or incompatible executable fails before launch; `models status` reports `failure_phase: runtime_files` with `failure_code` `runtime_missing` or `runtime_incompatible`, distinct from a native load failure (`runtime`). Distribution archive identities are in [the runtime notice](../third_party/llama.cpp/README.md).

These are fixed deployment profiles, not CLI tuning settings:

| Concern | E4B CPU | E4B GPU | 26B GPU |
| --- | --- | --- | --- |
| Loading | `mmap` | `mmap` | `none` |
| Device / GPU layers | `none` / 0 | Selected Vulkan device / 99 | Selected Vulkan device / 99 |
| Target / batch threads | Persisted detected profile | 4 / 4 | 4 / 4 |
| Flash attention | Off | On | On |
| MTP | E4B drafter, on | Off | Off |
| Expert placement / cache | Not applied | Not applied | `--n-cpu-moe 22 --moe-cache-mib 2132` |
| Additional placement controls | Unchanged | Unchanged | `--swa-full --cache-ram 0 --fit off` |
| Active translations | 1 | Up to 2 | Up to 2 |

All profiles keep total context 4608, three server slots (1536 tokens each), batch/ubatch 512, f16 K/V cache, prompt reuse, reasoning off with budget 0, priority -1, warmup/performance/metrics, no web UI, one HTTP thread and an owned dynamic loopback port. CPU detection uses target `max(1,min(ceil(physical_cores*0.5),6))` and batch `max(1,min(ceil(logical_cores*0.75),12))`; an 8C/16T system detects 4/12. CPU draft threads are 1/1, device `none`, GPU layers 0, f16 K/V, `draft-mtp`, draft maximum/minimum 3/2 and minimum probability 0.0.

Self and Peer share one model/server without sharing mutable generation state. Three retained prefix slots do not enable three simultaneous generations. Manual translations use the same active capacity. Cancellation retains its slot until owned server work is settled; model/backend changes and shutdown drain or cancel owned work before replacement.

E4B may fall back internally from GPU to its serialized CPU/MTP profile. **26B never falls back to CPU or E4B.** Resolve the reported Vulkan/device/resource problem or explicitly select another supported model/backend. Neither managed model uses a remote translation-provider fallback.


### Other settings

The finite command catalog also includes:

- Providers and accounts: Soniox diarization, managed referral settings, custom and local provider settings, both-channel provider updates, free-tier provider choices, and Qwen region.
- Models and devices: GPU device selection, model/connection history, and audio devices/host APIs.
- Translation inputs: prompts, languages, and vocabulary.
- Capture and output: VAD, overlay values, and OSC settings.
- Telemetry settings.

Use `capabilities` for field names and types. Use `settings choices` for supported values rather than reproducing dynamic lists.

### GUI draft conflicts

- In an already-running GUI, focused external changes preserve unrelated provider/prompt drafts.
- Overlapping changes require explicit conflict resolution.
- A successful pending GUI apply acknowledges only its submitted values.
- Newer edits made while it was pending remain staged.
- A failed apply retains its draft for retry.

## Named query and command fallback

Convenience domains cover the common operations. When scripting a less common supported operation, use the finite catalog rather than private Python APIs:

```powershell
puripuly.exe query <query-name> [--arguments '<JSON object>' | --file <query.json>]
puripuly.exe command <command-name> [--arguments '<JSON object>' | --file <arguments.json>] [--expected-revision <revision>] [--no-wait]
```

- The catalog rejects unknown names. It is not RPC reflection.
- `capabilities.query_arguments` describes the finite argument shapes for named queries.
- Queries without declared arguments reject them.
- `command` accepts only names and argument shapes returned by `capabilities`.
- Use `--file` for JSON payloads that include sensitive or private content.
- Protected operations have dedicated `secrets`, `auth`, and `text` commands.
- The generic command form rejects `secrets.set`, `secrets.verify`, `auth.login`, and `text.submit`.

Global options:

- `--config <settings-file>`
- `--timeout <seconds>`: greater than 0 and at most 3600.

Mutations wait for a terminal result by default. `--no-wait` returns an accepted operation identity without claiming completion.

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

### OAuth challenges

An explicitly requested Discord/OpenRouter `auth login` can return an interim authorization challenge:

- An authorization URL.
- `status: action_required`
- `operation_status: running`
- `terminal: false`

Exit code 5 does not mean that login finished. Complete the human authorization separately, then inspect or wait for the same operation:

```powershell
puripuly.exe operation status <operation-id>
puripuly.exe operation wait <operation-id>
```

- Only a terminal receipt establishes the final login result.
- A final `action_required` is terminal. The workflow cannot proceed without its stated action.
- An authorization URL is not a credential.
- The URL may be returned only by an explicitly requested auth command, query, or subscription.
- The PKCE verifier is never returned.

### Waiting and cancellation

Use operation commands to inspect or await work, or request supported cancellation:

```powershell
puripuly.exe operation status <operation-id>
puripuly.exe operation wait <operation-id>
puripuly.exe operation cancel <operation-id>
```

- Active model installation and Gemma preparation are the cancellable long-running operations.
- Other operation types may report `cancellation: unsupported`. The CLI exits 4 and leaves a still-running operation marked nonterminal.
- Cancellation is not a rollback. A settings commit or already-applied runtime change remains committed.
- The operation receipt describes what has already completed.

### Shutdown during settings operations

- If shutdown interrupts a settings operation after a durable save, the receipt retains the committed transaction and its revision.
- `transaction.status: settings_commit_success_runtime_interrupted` means settings persisted, but runtime completion was not established before interruption.
- A known runtime-applied or runtime-degraded transaction is retained instead when available.
- A queued operation that never committed has no committed transaction.
- These fields report completion metadata, not private setting values.
- Shutdown does not rewrite an already completed receipt.

### Request identity and retries

- Mutation requests have a caller `request_id`. The client generates it unless `--request-id` is supplied on `command`.
- The host deduplicates a matching request identity and payload while it retains them.
- Reusing an ID with different content is rejected.
- Each host retains up to 256 operation/request identities. Terminal records may be evicted when that bound is reached.
- The instance UUID changes on restart. Operation IDs from a previous host cannot be resumed against the new one.
- If a response is lost or the client times out, execution may be unknown. Inspect host and operation state before deciding what to do.
- The CLI never automatically replays an ambiguous mutation.

## Credentials, authorization, and consent

### Secrets

Provider secrets are supplied via hidden terminal prompts or stdin, never as regular command-line arguments:

```powershell
puripuly.exe secrets set soniox_api_key
Get-Content -Raw .\secret.txt | puripuly.exe secrets set soniox_api_key --stdin
puripuly.exe secrets presence
puripuly.exe secrets verify soniox_api_key
```

Saving the active DeepSeek, OpenAI, Gemini, or OpenRouter BYOK translation key applies it to subsequent translations without restarting, including first registration. This does not enable translation or capture. Saving an inactive key does not rebuild the active translation provider, and managed-account routes do not switch to BYOK.

- Verification reads a secret through hidden input or stdin. It is separate from storage; saving a key resets its verification status to `unknown` and does not establish service health.
- `capabilities` and `secrets presence` list the declared secret identifiers, including loaded HTTP-extension declarations.
- A key without a supported verification protocol returns terminal `action_required` with `verification: unavailable`. It does not report verification success.
- Secret values are never returned in settings, query results, operation receipts, or logs.
- Delete removes the secret from local storage.
- If a key is persisted but runtime application fails, the operation reports `degraded` with the retained failure cause. The stored key remains saved and can still be verified separately; neither a successful save nor a successful key probe claims that the live provider adopted it.
- If independent STT/GPU application also fails, its cause is retained as a linked secondary diagnostic. The missing translation backend remains eligible for recovery through an ordinary provider apply.

Verification is a limited key/protocol probe, not proof of healthy live recognition:

- Diagnostic metadata distinguishes `verification_scope: key_probe` from `recognition_readiness: unobserved`.
- A failed probe preserves known typed transport/status causes. An unsuccessful boolean alone is not evidence of invalid credentials.
- Some provider protocols treat a probe timeout as success. That result does not establish successful recognition.

### Authentication

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

- **Browser access**
  - Browser opening is off by default.
  - `--open-browser` is the sole CLI permission to launch a browser.
- **Protected input**
  - QQ uses hidden prompts for its identity and credential.
  - It also accepts a protected JSON object on stdin with `qq_identity`, `credential`, and optional `referral_id`.
  - Discord accepts optional `referral_id` where allowed. OpenRouter does not accept one.
  - For automation, use the dedicated `auth login ... --stdin` input rather than generic arguments.
- **Authorization**
  - A login request is the explicit authorization action.
  - Translation toggles, settings/provider changes, generic commands, and OSC translation commands must not initiate OAuth implicitly.
  - Existing authorization may still be used for normal operation.

### QQ authorization recovery

QQ authorization awaiting managed-operation recovery reports:

- `action_required`
- `reason: resolve_pending_authorization`
- `transaction.status: remote_delivery_ack_pending`

This also applies when the message identifies a timeout, proxy, TLS, DNS, or other network failure.

- Inspect the transaction status. A network message alone does not establish rejected credentials.
- Pending delivery acknowledgment does not prove that a usable key was saved locally.
- A missing or unreadable local key receives separate storage feedback and is not acknowledged to the Broker.
- Neither pending condition is a reason to claim another key.

### ChatGPT login

- ChatGPT login authorizes GPT 6 Luna on the `chatgpt` translation connection with the user's ChatGPT plan.
- It does not change the selected translation model or connection.
- `auth status` reports `chatgpt.signed_in`, `chatgpt.in_progress`, and `chatgpt.sign_in_required`. It does not return the account email.

### Authentication outcomes and logout

- **Observation and failure causes**
  - When secret observation fails, `signed_in` is `null`, not a fabricated signed-out result. Inspect its diagnostic reference.
  - Local credential deletion and confirmed remote revocation are separate outcomes.
  - Secondary listener, storage, or notification failures do not replace the original authentication cause.
  - They also do not retroactively invalidate a completed durable settings transition.
- **Logout scope**
  - Account logout is local-only and does not revoke a remote provider grant, except for ChatGPT.
  - ChatGPT logout also requests refresh-token revocation.
  - It reports `scope: remote_revoked` when revocation is confirmed, or `scope: local_only` when it is not.
- **Active route**
  - Logging out an inactive account preserves unrelated BYOK, local, or other-account translation.
  - Only the affected active route is stopped/rebuilt.
  - Failed logout persistence does not silently leave that route's translation disabled.
- **Required user action**
  - Authentication and model operations can require a human step or missing entitlement and report `action_required`.
  - The CLI never auto-accepts consent.
  - Read peer terms through `capture terms`. An informed user can explicitly accept them when enabling peer capture, as described above.

## Output, events, logs, and privacy

### Output and transport

- CLI stdout is UTF-8 JSON with `output_version: 1`.
- Diagnostics and structured errors are JSON on stderr.
- Follow commands emit one JSON event per line.
- The local transport uses protocol v1 newline-delimited UTF-8 JSON, not pickle, over ephemeral IPv4 loopback (`127.0.0.1`).
- Before catalog dispatch, it authenticates the discovered instance UUID and capability token. Token comparison is constant-time.
- The transport exposes no arbitrary method/reflection or code execution.

Windows discovery and transport bounds:

- Discovery uses a current-user ACL-verified record under the user's local application data.
- An exclusive `LockFileEx` lease protects the canonical settings identity.
- The endpoint/token are not an unauthenticated management service.
- Requests are limited to 65,536 bytes including the newline. Responses are limited to 1,048,576 bytes.
- The transport validates protocol, instance, and request correlation.

### Event subscriptions and content

- **Content access**
  - Queries and default event/log views omit credentials and conversation content.
  - Recognition and translation content subscriptions require the corresponding event topic and explicit `--include-transcripts` / `--include-translations` selection.
  - Non-content topics expose typed metadata only.
  - `osc_sent` never includes chatbox text, even with both content flags enabled.
  - Keep content flags off unless the caller needs content and can protect the output.
- **Buffers and ordering**
  - Subscribers have bounded independent buffers. Slow consumers do not block capture or translation.
  - Event sequence gaps are explicit.
  - Use `--after` to resume. Refresh a snapshot with queries after a gap.
  - Ordering is per owner/topic/channel. Do not infer one global order across self, peer, UI, and provider streams.
- **Topic selection**
  - `events follow` defaults to operation, settings, session-state, error, and gap topics.
  - Select topics with repeatable `--topic`. Use `--channel` to filter self or peer events.
  - For content, add `--topic transcript` or `--topic translation` and the corresponding include flag.

### Log subscriptions and audiences

- **Following logs**
  - `logs follow` defaults to warning and follows sanitized application metadata.
  - Use `--level debug|info|warning|error` to change the threshold.
  - Failure events carry the same safe diagnostic record available from retained history.
  - Use `--after <event-sequence> --instance-id <originating-host>` for scoped replay. Without `--after`, only new events are followed.
  - `events follow` also accepts `--instance-id`.
  - Neither interface returns mixed-content main-file bytes.
- **CLI and GUI audiences**
  - CLI `logs` events are not a transcript of every GUI Basic row or main-file line.
  - Typed retained diagnostics publish metadata events. Ordinary Basic/file-only observations do not automatically enter that stream.
  - Compare those audiences separately.
- **Basic output logs**
  - Basic logs omit typing-control calls, expected policy/duplicate/stale decisions, and duplicate summaries of typed failures.
  - Typing transitions and message submission still run. Failures remain available through the owning diagnostics.
  - Overlay admission/pacing records are file-only.
  - Surviving output milestones name the observed handoff, local send, or application stage. They never claim remote receipt or physical display.
  - Owner overlay transitions and nonempty discarded chatbox backlogs remain visible.
- **QQ authentication logs**
  - QQ authentication emits bounded `[QQAuth]` diagnostics through the normal live and persisted application logs.
  - Successful committed recovery is logged as success even without a user-facing message.
  - Local-key absence is distinguished from pending remote delivery acknowledgment.
  - Records use allowlisted stages, codes, and available HTTP/transport classifications.
  - They exclude raw exceptions, response bodies, QQ identities, credentials, keys, tokens, and proxy URLs.

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

- Use real identities and cursors returned by the host, not the example placeholders.
- `diagnostics.history` accepts only optional `operation_id`, `diagnostic_id`, `instance_id`, `after`, and `limit`.
- `diagnostics.health` takes no arguments.
- Both queries use the existing authenticated same-user local connection. They neither grant mutation authority nor enable conversation access.

Operation and diagnostic identity syntax:

- Payloads use 8–64 ASCII hexadecimal digits or UUID-shaped `8-4-4-4-12` groups.
- An existing native prefix such as `overlay-` is optional.
- Short native hexadecimal identities are valid. Identities are not restricted to UUIDs.
- Hyphen-only payloads or trailing separators such as `--------` and `dead----` are malformed. They do not mean retained evidence is missing.
- Host `instance_id` uses its separate instance-name syntax.

### Identity and interpretation

Schema version 1 records contain:

- A distinct `diagnostic_id`, UTC `occurred_at`, current `instance_id`, and `diagnostic_sequence`.
- Repository-owned component, operation, and stage.
- Severity, actual disposition, and stable code/category.
- Known cause, operation, request, channel, turn, and generation identities.

Unavailable relationships are `null`; they are not inferred from nearby events.

- Allowlisted native/status/provider details and bounded exception-class cause relationships may be present.
- Records exclude raw exception messages, stack traces, arbitrary child JSON/stderr, response bodies, credentials, URLs, personal paths, audio, and conversation content.

**Incomplete evidence and deduplication**

- `detail.state`, `detail.rejected_fields`, and `detail.truncated` disclose incomplete optional evidence.
- A rejected detail must not hide an otherwise permitted code or identity.
- Repeated projections of the same owning diagnostic are deduplicated.
- A new operation or generation is not silently joined to the previous failure.
- Unknown defects remain explicitly unknown. They identify the safe owning boundary rather than a guessed diagnosis.

**Disposition and authority**

Disposition is one of `failed`, `degraded`, `recovered`, `cancelled`, `superseded`, `disabled`, `unknown`, or `action_required`.

- It describes an observation, not permission to retry.
- Owner state and terminal receipts remain authoritative.
- Persistence failure, committed-but-degraded runtime application, cancellation, expected empty recognition, and disabled destinations are not interchangeable.
- Primary and bounded secondary cleanup causes remain separate.
- Local UDP send success does not acknowledge remote delivery.
- Output admission does not establish physical display.

**Secondary causes**

- Receipt projections retain the first four ordered `secondary_diagnostics`.
- `secondary_omitted_count` adds known omissions from the owning result and any later receipt or notification projection.
- `secondary_truncated` is true when that count is nonzero.
- The primary diagnostic remains separate. A secondary failure or truncation does not replace it.
- Transaction receipts preserve the transaction's validated cause linkage even when a diagnostic was emitted earlier; diagnostic history and failure events retain the immutable first record and identity.
- Unobserved or unavailable detail is not counted as a known omitted cause.
- These cause-list bounds are distinct from an individual diagnostic's `detail.truncated` field.

**Events and cursors**

- The same record is available as `diagnostic` on `logs` events with `kind: failure`.
- Operation events carry diagnostic references.
- **Diagnostic history cursors and event replay cursors are different sequences.**
  - Use history's `next_after` only for `diagnostics.history`.
  - Use an event's `sequence` only for follow/replay.

**Owner-specific observations**

- `gpu.status.runtime.startup_cleanup` reports pending startup cleanup, confirmed completion, and bounded safe cleanup failures.
  - Missing owner evidence is unavailable. It does not prove that an absent worker was fully closed.
- `app.status.scene_observation` distinguishes a successful process probe from a failed observation.
  - A failed enumeration cannot establish that VRChat is not running.
- Neither query exposes process command lines, participant identities, or raw scene logs.

**ASR model installation failures**

- Existing failures retain `local_asr_provisioning` / `models.install` / `local_asr_install_failed` evidence.
- That evidence includes the validated model ID, installation generation, and originating operation/request IDs when known.
- Exhausting the existing source fallback preserves the first native cause.
- Typed HTTP status and transport classification are retained without parsing exception prose.
- Query by the install operation ID or replay `logs` to inspect this file-only technical evidence.
- Basic installation summaries, model integrity, terminal install status, source fallback, and cancellation behavior are unchanged.
- The diagnostic does not authorize a retry.

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

- History returns records in ascending diagnostic sequence order.
- Inspect `oldest_sequence`, `latest_sequence`, `next_after`, `gap`, `evicted_records`, `truncated`, and `retention`. A bounded result is not proof of complete evidence.
- History stays in memory for the current host and resets on restart. It is not a database or durable audit.
- Retain the originating instance alongside any operation or diagnostic ID.

Query errors and lookup outcomes:

- Malformed arguments, negative cursors, invalid identity syntax, unknown keys, and limits outside 1–100 produce `invalid_arguments` and CLI exit 3.
- A validly shaped future cursor returns lookup `invalid_cursor`.
- A different host identity returns `foreign_instance`.
- Both lookup outcomes are successful queries (exit 0) with empty records and an explicit gap. They do not prove that no failure occurred.
- `not_retained` means no matching retained evidence. It does not mean an operation succeeded or never failed.
- Unavailable history is explicit.
- Follow/replay gaps identify eviction, foreign-instance, or invalid-cursor conditions and request a fresh snapshot.

### Delivery health is not persistence acknowledgment

`diagnostics.health` reports `history`, `events`, `file`, and available output-owner delivery observations independently of successful file writing:

- **History**
  - Reports availability, retained bytes/records, eviction, rejected detail/records, and duplicate projections.
  - Reports subscriber/file projection failures and bounded cross-thread dispatch pressure.
- **Events**
  - Report retained sequences, subscriber count, eviction, gaps, and disconnected consumers.
  - Slow consumers have independent bounded work. They do not backpressure application producers.
- **File**
  - Reports writer availability, queue capacity and pending work, and enqueued/written/dropped/rejected records.
  - Reports delivery failures, writer liveness, and time since progress.
  - `shutdown_attempt_complete` is distinct from confirmed `shutdown_complete`. Cleanup failures remain visible.
  - Pending work includes in-flight/buffered unfinished records.
  - `stalled` means a pending episode without observed writer progress for at least two seconds. It is not a guessed disk diagnosis.
- **Output**
  - Reports bounded owner-history retention/eviction and coalesced observer failures.
  - Includes the first `observer_failure_diagnostic_id` and current failure count.
  - A failed optional observer does not retroactively fail translation or acknowledge a destination.

**Persistence boundaries**

- File counters cover the existing whole main-file stream, including its permitted conversation records.
- `persistence` distinguishes `pending`, `incomplete`, `unknown`, and `written_not_durable`. `durable` is false.
- Enqueued is not written. Written is not an fsync/durability claim.
- A record's `file_evidence` is `unknown` unless unavailability is established.
- Aggregate counts cannot establish persistence of an individual record.

**Delivery failure evidence**

- Failed-file and overflow episodes expose `delivery_diagnostic_id` or `loss_diagnostic_id` through retained history.
- They do not recursively write another record to the failed sink. Their file evidence is unavailable.
- Bounded shutdown, queue loss, failed storage, and uninstrumented sinks must not be interpreted as lossless logging.

**Log audiences and sharing**

- The live Basic/Logs audience remains concise and factual.
- Selected technical metadata is file-only and available through the metadata CLI. Detailed mode is not restored.
- Explicitly permitted conversation logging remains separate. Review and redact the main log before sharing.

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

An exit code does not replace the JSON receipt. Before automation proceeds, inspect `terminal`, `status`, operation/instance IDs, and runtime state.
