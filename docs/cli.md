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

- Use the same `--config` path for host and clients. It identifies canonical settings-file ownership; omission selects the user's normal settings. Use isolated synthetic settings for probes, not a live user's configuration. A separate settings path alone does not isolate model, secret-store or other writable runtime roots.
- Only one host may own that settings identity. `app start` defaults to foreground; `--background` outlives the launching terminal. Ordinary no-command application launch still opens the GUI.
- `app list` lists current-user instances. `app discover` and `app status` target the selected settings identity; `app restart --background` replaces that host.
- Queries and mutations never start a missing host implicitly: `instance_not_found` requires an explicit start. A PID is observation, not control identity; the CLI never kills by PID.
- `app stop` waits for the original endpoint and observed process identity to disappear. `cleanup_outcome: unobserved` does not certify every teardown callback or resource release. `--no-wait` returns only shutdown acceptance.
- `--timeout <seconds>` bounds client waiting, not operation lifetime, and never retries a mutation. `Ctrl+C` exits 130; a background host remains independent.

Source changes are not hot-reloaded. Restart a running source host after changing code or bundled resource names; installed builds require a rebuilt application payload.

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

### GUI Settings Selection

- An enabled modal option applies on its first click. Single-select dialogs close after selection; multi-select dialogs stay open and retain at least one selection.
- Puri can still open settings and reveal a provider's secure key field without changing the selected provider. Settings no longer show the temporary Puri guidance banner or its return button; use the main navigation to return to Puri.

### Agent Chat Control

- Puri admits up to 30 submitted prompts per installation per UTC day, resetting at 00:00 UTC (09:00 KST). Classification and tool continuations share their prompt's allowance; starting a new conversation or restarting the app does not reset it.
- Puri uses the same application-control owners as the CLI through finite typed tools, not shell commands. It controls only the current application instance and exposes a subset of the CLI: settings changes, capture/translation/overlay switches, recognition/translation probes, microphone diagnosis, OSC discovery retry, Peer capture retry, GPU activation retry, stored-key deletion, account sign-out, desktop overlay position reset, and read-only status queries.
- CLI-only actions remain available from the CLI but not from Puri: shutdown and restart, model download/install/import/deletion and cancellation, Gemma preparation, GPU discovery, standalone credential save/verify, account login, manual text submission, Audio Test, Peer target selection, overlay size/lock/calibration, operation status/wait/cancel, and event/log observation.
- Puri receives app state as modules: `core` (switches, channels, pending card, operations, recent diagnostics) plus `audio`, `recognition`, `translation`, `accounts`, `languages`, `output`, and `general`. For work requests, D independently selects relevant modules and whether the task continues; `core` is always observed. Missing, ambiguous or invalid selection and classification timeouts select all modules. Omitted modules are not absent or disabled; `missing` records failed local observations. A `get_app_context` read observes `core` plus the requested module, and `overview` observes all of them.
- Up to four valid `get_app_context` calls may run together; other tools must be requested individually. Within the existing limits, a completed response with valid call identities but mixed tools or invalid batch arguments executes no tools and returns `invalid_tool_batch` with correction guidance to the model instead of ending the reply. The model can correct its calls within the same submitted prompt; rejected calls count toward the tool limit. Approval requirements, incomplete-response rejection and model-request bounds remain unchanged.
- Localized setting labels, descriptions and static options are sent in per-module `catalogs`; current values remain in module `settings`, private presence in `private`, and device/host/GPU/extension choices and availability in `options`. Later contexts append only new or changed catalogs and complete module states; `unchanged` confirms fresh observations matching retained evidence. Included module states and catalogs replace their earlier versions, rather than merging omitted fields. Failed observations are never reported as unchanged. When supporting history is trimmed or the conversation resets, needed evidence is sent again. Filtered diagnostics remain separate query-scoped evidence. An explicit accounts read still stages current account-action controls even when the account facts are already known.
- Same-task follow-ups retain selected and additionally read modules; new tasks, conversation reset/expiry and trimmed task history clear the previous selection. Approval and key-verification continuations use code-selected modules and their originating task, without another D call. Social requests receive display dialogue only; out-of-scope requests receive only the current message. Neither receives app snapshots, tool results or reasoning history.
- Simulator `run` and `smoke` accept `--context-mode selected|full` (default `selected`) for comparison, and record the mode and effective context decisions. Both modes use the same relay instructions and tools. Regenerate the classifier's module list from the Python module registry with `uv run python agent-relay/generate-context.py`; rebuild the relay after generation. This does not authorize deployment.
- Option labels, short descriptions, choices, and approval values come from a shared catalog localized to the current UI language (English, Korean, Japanese, Simplified Chinese, or Russian). The response language does not rename UI controls. `chatbox.activation_notice.enabled` controls the `PuriPuly ON!` notice; `stt.soniox.enable_speaker_diarization` controls Soniox speaker separation.
- CLI/API `overlay.background_alpha` remains opacity in `[0, 1]`. Both Settings and Puri's approval card show background **transparency** as `(1 - alpha) × 100%`; `0.6` therefore displays as `40%`.
- Settings changes, stored-key deletion, account sign-out, and desktop overlay position reset use the same revision-bound change card with before and after values. Settings cards show at most two logical changes; coupled model/connection and microphone host/device selections remain grouped. After approval the app waits for the terminal result and then shows a fixed result line; pending operations are not reported as completed.
- Recognition/translation originals, user-authored prompts, vocabulary, custom endpoints, free-text settings, and secret values are not provided to the agent model. Private settings editing and raw-content inspection use existing app screens; there is no separate agent settings editor or local input form. Key entry uses the existing provider-setup save-and-verify card. Turning on Peer capture before its terms are accepted returns `action_required`; the terms are accepted through the dashboard LISTEN switch.
- `settings.current` includes `private_presence` boolean metadata for protected settings. A missing presence observation is unknown, not evidence that a value is empty.

### GUI Autosave

- GUI settings edits are coalesced and attempted once per submitted edit. There are no timed retries or automatic runtime-only replays after failure. Saved-but-unapplied edits remain saved; persistence and effective runtime state are separate results.
- Expected save/application failures do not show a settings failure toast, a persistent recovery row, or a dedicated Retry button. Diagnostics and unrelated operation notifications remain available.
- Unsaved, reverted, or unknown-outcome edits remain only in the current page's memory; closing the app does not preserve them for restart. Explicitly submitting the affected setting again permits another attempt. Unrelated edits, repeated queue drains, and navigation without a new draft do not replay a failed submission.
- This settings queue belongs to the GUI. CLI operations use the application owners and their receipts. Explicit runtime-only application and existing provider, telemetry-reporting, and model-installation recovery are unchanged. Actual persistence and resource application can still delay other ordered operations.

### Interpret Readiness at Its Owner

| Observation | What it does not establish |
| --- | --- |
| Saved provider/model selection | Attached or usable runtime provider |
| `app.status.translation_runtime_ready` | Translation enabled or managed model prepared; it means a provider is attached |
| Active audio capture | Recognition availability or successful translation |
| Configured overlay/OSC target | Effective resource readiness, remote receipt, or physical display |
| Applied manual-text admission | Completed translation or destination delivery |

The GUI dashboard's “Ready!” means Self/TALK capture is active and its recognition provider accepts input. It does not require a first recognition result, certify an open remote connection, or guarantee recognition/translation success. Startup is shown as preparing, shutdown as stopping, and unavailable capture as failed. Readiness refreshes preserve existing conversation content and specific error/model notices; they do not clear the display.

Use the terminal operation receipt together with `providers.status`, `models status`, `overlay status`, `osc status`, or the relevant owner's query. Correct the reported cause before explicitly retrying. A failed/cancelled current translation warmup turns effective translation off while retaining the selection and failure details.

An explicit SteamVR runtime shutdown turns CAPTIONS off after native cleanup, without a Desktop fallback or automatic restart. The saved VR target and calibration remain unchanged; capture and translation continue independently. Start SteamVR and use `overlay set on` to enable captions again. Initial enable without SteamVR retains the existing Desktop fallback; genuine crashes and cleanup failures remain failures. If runtime shutdown interrupts an enable operation, its receipt can be `degraded` because ON was not reached; use `overlay status` to distinguish the resulting clean OFF state from an overlay failure.

Matching `translation set off` or `capture set <channel> off` can revoke pending startup. Shared model demand may keep resources alive for another channel. Runtime stops are distinct from cancelling an explicit model download/import.

`osc discovery retry` performs one bounded automatic-discovery attempt using the saved connection. It refreshes/reapplies target information for a running discovery service or restarts failed discovery resources; it does not rebuild a healthy receiver or change saved destinations. Manual/off, in-flight, changed connections and stopped lifecycles are rejected. The terminal receipt distinguishes `applied` with `discovery_target_applied` from `degraded` with `discovery_unavailable`; acceptance or scheduling alone is not recovery. `osc status` exposes the current destination source and a retained, timestamped local-listener observation. Unknown ownership/access remains unknown; the narrow local probe is not applicable to non-loopback destinations. Finding/applying a target or matching a listener does not prove VRChat health, receipt or display. Remote delivery remains unacknowledged; the operation submits no chatbox text and replays no old translations.

### Independent Component Checks

These commands use application-owned synthetic inputs, not the microphone, loopback, speaker playback, or normal conversation/output pipeline:

| Command | Arguments | Successful terminal result establishes |
| --- | --- | --- |
| `recognition.probe` | `{"channel":"self"}` or `{"channel":"peer"}` | A nonempty authoritative final recognition result for that configured recognizer's isolated test session |
| `translation.probe` | `{}` | A nonempty synthetic translation through the currently attached shared translation backend |

```powershell
puripuly.exe command recognition.probe --arguments '{"channel":"self"}'
puripuly.exe command recognition.probe --arguments '{"channel":"peer"}'
puripuly.exe command translation.probe
```

Read the terminal receipt, not admission or readiness. Recognition reports `recognition_result`, final-event authority, configured/exercised provider, and captured/current revision and generation. It uses the bundled 1.194-second `data/diagnostics/eung.wav` sample without changing the configured language. `configured_language` reports the language applied to that recognition session, not a detected sample language or an accuracy guarantee. For `rolling_free`, `recognition_scope: single_eligible_rolling_member` identifies only the tested member, not full rolling-route health.

An inactive configured recognition channel can use a probe-owned backend without enabling capture. An attached channel keeps its live sessions/model capacity; the probe owns only its isolated session. Missing assets/authentication/readiness return a prerequisite or failure, without provisioning, login, provider switching, or settings changes. Translation requires an already attached usable backend and reports `translation_result`; managed preparation is not silently performed. Existing selected-provider routing can involve more than one transport attempt and may incur its normal cost.

Cold recognition probe success also includes temporary-backend cleanup. Pending or failed cleanup remains runtime-owned and reserves that channel's construction capacity until reclaimed; a cancelled operation does not by itself prove resource release.

Empty, failed, cancelled, or stale results are not successful tests. Receipts omit recognized/translated text and explicitly mark untested stages: real capture, recognition accuracy, translation after recognition, output delivery, and display as applicable. A successful component check does not certify the whole Self/Listen path. The sample duration is not a completion-time guarantee.


### Peer Capture and Audio Tests

Read `capture terms` first. Only an informed user's explicit choice authorizes `capture set peer on --accept-peer-terms`; never auto-supply consent. The flag is valid only for peer-on. Peer speech is excluded from the VRChat chatbox.

A capture-target change that is saved while requested Peer capture remains pending reports `degraded`, not `applied`, through both the GUI application boundary and `audio.target.set` control receipts. With Peer capture intentionally off, successful target application does not require an active capture source.

After a capture failure, explicitly applying a valid target reports `applied` once the requested provider and capture configuration are ready and active. Resubmitting that unchanged healthy target also reports `applied`. Prior safe failure evidence remains available through bounded current-instance `diagnostics.history`; it does not make recovered capture degraded. Pending recovery and new current failures still prevent successful convergence.

`audio test on|off` runs VAD without recognition or translation. It pauses normal capture and restores only still-valid prior intent; inspect `app status` → `audio_test` for readiness, failures, and VAD readings. Stopping waits for owned input release, not just a visual toggle.

`audio diagnose` runs the cancellable `audio.diagnose {}` operation. It tests up to four actually enumerated microphone/host pairs sequentially, current route first, with a bounded three-second observation window per candidate. Present Virtual Desktop, Steam Link, Meta Link/Air Link and PICO microphone-driver routes are considered before unrelated alternatives; distinct driver routes precede duplicate host-API views of the same driver. Names only affect test order, never the measured choice or saved settings. Playback-only endpoints are excluded. A driver can remain installed while its headset session, microphone permission or passthrough is unavailable, so enumeration alone does not establish working input. The operation pauses normal capture through the same owner and restores only still-authorized intent; it never tests loopback, performs recognition, calls a cloud service, saves audio or changes the selected route.

```powershell
puripuly.exe audio diagnose
puripuly.exe audio diagnose --no-wait
```

Read each candidate's open/readiness, frame count, signal level, VAD evidence, failure and cleanup fields. The diagnosis owner alone ends the observation window; reaching that bound preserves measured frames without inventing a no-audio failure. The best measured candidate without a source or cleanup failure is a suggestion for a separate settings change, not a recovery certificate. Equal evidence preserves test order, including the current route; silence-only frames do not prove microphone speech works. Actual source and cleanup failures remain failures even after frames arrive and exclude that candidate from selection. Silence, missing frames and recognition failure are distinct. A cancelled operation can retain unresolved native cleanup debt, which blocks conflicting capture until release. A later explicit capture-off is never undone by restoration.

Diagnostic history records bounded numeric evidence without raw audio, device labels or exception prose. Audio-test logging adds phase, frame/level/VAD and cleanup observations. These establish only the exercised input/VAD path, not recognition, translation or physical caption display.

### Manual Text

Manual text bypasses microphone/STT and follows the Self translation/output path. Keep conversation text out of process arguments:

```powershell
puripuly.exe text submit --file .\message.txt
Get-Content -Raw .\message.txt | puripuly.exe text submit --stdin
```

The receipt provides admission and turn identities with `output_delivery: unobserved`. Correlate subsequent translation events and destination evidence; the bounded typing-completion wait is not an output acknowledgment.

### Local Speech Recognition

Use current `settings choices` and `models status`; retired CPU models are not installable choices. CPU Auto is the default for both Self and Peer. It routes by the configured source language, not automatic spoken-language detection, with this priority:

| Selected CPU model | Source language codes |
| --- | --- |
| Standard Fun-ASR-Nano | `zh`, including `zh-CN` and `zh-TW` |
| Parakeet Japanese CTC | `ja` |
| Granite Speech 5.0 Multilingual | `en`, `fr`, `de`, `es`, `pt`, `it` |
| GigaAM-v3 e2e-CTC | `ru` |
| Fun-ASR-MLT-Nano | `yue`, `ko`, `vi`, `id`, `th`, `ms`, `tl`, `ar`, `hi`, `bg`, `hr`, `cs`, `da`, `nl`, `et`, `fi`, `el`, `hu`, `ga`, `lv`, `lt`, `mt`, `pl`, `ro`, `sk`, `sl`, `sv` |

Direct Granite selection supports seven languages: `en`, `fr`, `de`, `es`, `pt`, `it`, `ja`; Auto retains Japanese CTC for `ja`. GigaAM supports Russian only. Direct [MLT selection](https://github.com/handy-computer/transcribe.cpp/blob/8facde6fcd8fb9caa585a5788ab450df6e8ec7db/docs/models/fun-asr-nano.md#choosing-a-variant) supports its full declared 31-language scope, including `zh`, `en`, `ja` and `pt`, which Auto routes elsewhere. Regional codes are normalized to their base language. The union covers 36 base language codes; this does not add languages to the application's language picker. Ukrainian (`uk`) and languages outside that scope are explicitly unsupported, without silently switching to cloud, GPU or retired providers. Auto requires all five current CPU models installed.

- Each active transcribe.cpp CPU channel owns its native Model/Session and requests two inference threads. Talk and Listen can compute concurrently with 2+2 threads, including on the same checkpoint, without preloading an inactive channel. This is not a total process-thread cap.
- Granite, GigaAM, standard Nano and MLT use Q8_0 weight quantization, not all-INT8 execution. They use the existing CPU worker and unchanged transcribe.cpp revision `8facde6fcd8fb9caa585a5788ab450df6e8ec7db`; Japanese CTC retains sherpa-onnx.
- Granite and GigaAM receive no language hint. Granite marker-only no-speech output becomes an empty transcript, including surrounding whitespace or terminal punctuation; real mixed lexical output is retained.
- Fun-ASR Nano and MLT keep ISO language codes at the application/worker boundary and convert them to the checkpoint's documented Chinese language names inside the native prompt (`zh` → `中文`, `ko` → `韩文`). This is language conditioning, not a strict decoder constraint: short, ambiguous or noisy speech can still be misrecognized or produce another language.
- GPU ASR keeps one resident Model shared by independent channel Sessions and serializes inference. Switching drains and frees the old Model before loading its replacement; it never preloads a second GPU Model.
- Channel stops or changes revoke that channel's results and retain started native work until settlement. Another channel's unrelated resource remains owned. An admitted stop is not proof of native cleanup.
- Legacy settings and control inputs `local_orukeet`, `local_parakeet_v3` and `local_qwen` normalize to `local_cpu_auto` on both channels and are persisted canonically through the existing recoverable migration. Queries, saved settings and choices use canonical values, including `local_granite` and `local_gigaam` for direct selection. See [OSC numeric compatibility](vrchat-osc.md#asr-ids) for legacy numeric inputs.
- Only initial application startup, after successful recoverable canonical settings loading, removes recognized retired CPU assets and their known residue before constructing native providers. Ordinary settings reloads or telemetry verification do not authorize deletion. Locked/deferred cleanup retries on a later application startup. Unknown contents, reparse targets, Japanese/GPU assets and settings backups are preserved.
- All five CPU models have alternate download paths. Downloads use pinned sizes and SHA-256 with the normal staged installer. The derived `hf_mirror` source serves the same files through `hf-mirror.net`, including Granite and GigaAM; source fallback changes neither the selected model nor its integrity contract. Source reachability is distinct from recognition readiness.
- Windows Setup automatically provisions these five CPU ASR models. The optional **Re-download AI models** task forces their replacement even when installed files pass validation; leaving it unchecked still installs missing or invalid required models. Smart Turn and Silero VAD are bundled application resources, installed with the package rather than downloaded by this task. They are not loaded from the writable ASR model cache. Managed Gemma weights remain a separate on-demand download.

Exact model/source pins live in the [ASR asset registry](../src/puripuly_heart/core/local_asr/local_stt_assets.py) and its packaged manifests. Granite is Apache-2.0, GigaAM is MIT, and Fun-ASR retains its model agreement; see the [third-party notices](../src/puripuly_heart/data/THIRD_PARTY_NOTICES.txt).

The original [#353 comparison report](https://github.com/kapitalismho/PuriPuly-heart/blob/0246def7b/docs/experiments/issue-353/results/report.md), [saved summary](https://github.com/kapitalismho/PuriPuly-heart/blob/0246def7b/docs/experiments/issue-353/results/summary.json) and [execution evidence](https://github.com/kapitalismho/PuriPuly-heart/blob/0246def7b/docs/experiments/issue-353/execution-evidence.json) remain pinned to the merged experiment commit. That bounded English/Japanese comparison supports the English Granite and retained Japanese CTC decision; it does not benchmark-verify every declared language or Russian GigaAM.

The reusable [CPU ASR benchmark harness](../benchmarks/asr/README.md) keeps scripts, frozen inputs, local model files, audio/cache data and generated results together under `benchmarks/asr/`. Large/local/generated directories are ignored; model downloads reuse the application installer and its alternate-source integrity checks.

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
- ChatGPT logout waits for already-started credential mutations before confirming local removal. A cancelled refresh cannot restore the removed credential. If a later successful explicit login supersedes removal, its credentials are preserved and the earlier logout reports `local_credentials_removed: false`; remote revocation is reported independently. A failed login with rolled-back credentials does not cancel removal.
- Browser authorization and app-to-service traffic can use different routes. Correct proxy/TLS configuration rather than disabling certificate verification. Share only sanitized diagnostic fields, never OAuth codes, keys, tokens, proxy URLs, or full settings/secret files.

### Proxy Changes

- Shared external HTTP clients read the currently visible proxy settings when dispatching each request. Generic WebSocket connectors read them when opening a connection, including redirected connection attempts. Changing the proxy does not migrate an existing request or socket, or automatically restart recognition.
- Gemini Live and ordinary Hugging Face/Xet worker downloads retain a proxy snapshot selected during session or operation setup. Start a new session or operation to pick up changed settings in those paths.
- Environment settings are read from the running process; editing operating-system environment variables does not necessarily update that process. Proxy refresh does not reload an existing client's explicit CA-file configuration.

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
- History is bounded, in-memory, and current-instance only. Inspect gaps, eviction/truncation, and lookup outcomes. An unfiltered cursor at the current tail returns `no_new_records`: no newer retained diagnostic at that snapshot, not proof of recovery. Missing requested identities remain `not_retained`; neither that nor `foreign_instance` or `invalid_cursor` proves an operation succeeded or never failed.
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
