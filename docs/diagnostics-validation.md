# Diagnostic failure verification — #228

Implementation authority: [#228](https://github.com/kapitalismho/PuriPuly-heart/issues/228), with child outcomes #229–#232. Source baseline: `75c4485a6787d504f5335d1602973f63b4d0c234`. This record distinguishes executed software observations from hardware or external-service claims. The implementation commit and independent review range are identified in the delivery record; working-source fingerprints below identify the original live observations.

## Environment and isolation

Windows x64, Python 3.14.7, Flet/flet-desktop 1.0.0. Every Python execution explicitly selected this worktree's `src` through `PYTHONPATH`, rather than the adjacent checkout's editable install. Each composed host used independent temporary settings, `LOCALAPPDATA`, discovery and empty encrypted-file secrets with a fresh passphrase. Inherited provider credentials and native-layout overrides were removed. Telemetry was off; OSC/HTTP/WebSocket fixtures used loopback only. No real account actions, paid provider requests, ambient audio, or new peer-terms acceptance were used.

CLI observations came from separately attached processes using the existing authenticated local transport. GUI observations used actual owned Flet windows and owned-window input. Screenshots were not diagnostic payloads. Owned process identities were checked after shutdown; temporary application state and throwaway probes were removed after retaining sanitized evidence.

## Reusable composed commands

`<python>` denotes the configured Python 3.14 interpreter; `<report>` is an explicitly selected local evidence destination. Both probes construct isolated application state, launch the real composed host, run independent CLI clients, and stop their owned processes:

```powershell
<python> -m puripuly_heart.release_evidence.diagnostic_cli_failure --report <report>
<python> -m puripuly_heart.release_evidence.output_diagnostics --report <report>
```

The actual CLI command prefix was:

```powershell
<python> -m puripuly_heart.main cli --config <isolated-config> --timeout 30
```

With that prefix, the executed command families included:

```text
capabilities
query capture.channels
query settings.current
settings apply --file <isolated-change.json>
operation status <operation-id>
query diagnostics.history --arguments '{"operation_id":"<operation-id>","instance_id":"<instance-id>","limit":10}'
query diagnostics.history --arguments '{"diagnostic_id":"<diagnostic-id>"}'
query diagnostics.health
logs follow --level info --after 0 --instance-id <instance-id>
auth login chatgpt
translation set on
text submit --stdin
query osc.status
overlay set on
query overlay.status
app stop
```

Settings-denial input was `{"changes":{"locale":"ja"}}`. Manual text was synthetic and passed on stdin, not a command-line argument. A protected authentication operation used dedicated `auth login chatgpt`, without `--open-browser`; a rejected generic-command attempt was not counted as authentication proof.

## Settings denial, retained history and restart

The real Windows settings file was held with read/write sharing but without delete sharing. The application could not replace it. Observed selected fields:

```json
{
  "instance_id": "98bba849-736e-4002-be57-52c909e7bb04",
  "operation_id": "39dc786dfd424da2a6f3ca084fe601bc",
  "request_id": "ccf3b2ece48418a54372e40124f84908",
  "cli_exit": 4,
  "status": "persistence_failed",
  "transaction_status": "settings_commit_failed",
  "diagnostic": {
    "diagnostic_id": "9a78be9ee88b404ea40f90bf78109f00",
    "component": "settings_repository",
    "operation": "save",
    "stage": "settings_replace",
    "code": "settings_save_failed",
    "fields": {"exception_class": "PermissionError", "errno": 13, "winerror": 5},
    "detail": {"rejected_fields": 0, "truncated": false}
  }
}
```

The receipt, independently queried operation/history, separately attached follow/replay and selected main-file JSON retained the same diagnostic identity and safe meaning. Settings bytes and effective locale remained unchanged; both capture channels remained inactive. File health eventually reported six enqueued/six written, no drops/failures, and `durable: false`.

After restart, querying the original instance returned `foreign_instance`, no records, and an explicit gap. It did not claim that the old operation had succeeded or never failed. Normal and restart stops returned CLI 0/host 0. A disconnected follower was removed from subscriber ownership.

A separate real failed-file scenario used a directory as the main-file destination. Live history/health queries still exited 0. `log_file_delivery_failed` retained `PermissionError`/errno 13 and `file_evidence: unavailable`; health reported four enqueued, zero written and four delivery failures, with incomplete persistence. Stop returned CLI 0 and `cleanup_outcome: unobserved`; the host exited 1. Process disappearance was not reported as successful resource cleanup.

## Application, authentication, GUI and cleanup

The final composed application run used instance `eb377a2e-9c30-4681-ad56-f7ba6550d8d9`:

- Valid locale input durably advanced revision 2 to 3, while an injected owning runtime `TypeError` left effective locale `en` and persisted locale `ja`. CLI exit 4 reported committed/degraded, not invalid arguments. Diagnostic `4e21d0cab4c14bfc81a5375d43e56a02` was associated with operation `7cbf3930d249494b9a7b0f615bbc2a4f`, component `settings_application`, operation `apply_ui_prompt_clipboard_state_runtime`, stage `runtime_apply`, and no rejected detail.
- Deliberately invalid locale input was rejected without advancing the committed revision. Permanent tests distinguish execution `TypeError`, `ValueError`, and `KeyError` from genuine validation rejection.
- Dedicated ChatGPT login failed at controlled `loopback_bind` with a safe `TypeError` and unknown category before creating an authorization URL, opening a browser, or making an account/network request. A successful empty local credential observation remained signed-out; failed observation is separately tested as unknown/null.
- Process enumeration failure appeared as `observation_succeeded: false`, `process_present: null`, and an owning diagnostic reference. It did not claim that VRChat was absent.
- Two ordered teardown callbacks failed after their real callbacks: a primary permission error followed by a linked I/O error. All six lifecycle phases continued, with zero outstanding tasks and a completed-with-failures result. Host exit 1 and CLI process-stop exit 0/`cleanup_outcome: unobserved` remained distinct.
- Startup fallback under settings replacement denial retained the loaded canonical state and linked the failure-to-persist observation to the original disk failure; it did not claim the fallback was installed.

Actual GUI instance `377a6189-22f2-4a79-8d60-b4d546fbf172` exercised a staged synthetic prompt edit. Under replacement denial, revision 2 remained unchanged and the edit remained visibly pending/unacknowledged. Releasing the owned lock allowed the next actual queued GUI action to persist revision 3. Basic/Logs showed one factual summary per distinct failure, without the removed duplicate summary, troubleshooting advice, or Detailed mode. The same native permission fields survived into independent CLI replay. Both owned Python/Flet process identities were absent after normal stop.

Secret write, verification, commit, restore and QQ compensation are covered by deterministic behavioral regressions: native primary identity survives separate state/secret restoration failures; bounded secondary causes are linked rather than replacing primary authority. Refresh rejection, internal refresh defects, unconfirmed remote revocation, local credential removal, listener cleanup debt and notification failure are distinct. These tests do not claim real remote login, refresh or revocation.

## Speech, capture and provider software path

The permanent source-composed regression was run with:

```powershell
<python> -m pytest tests/core/test_speech_diagnostics_composed.py -o addopts= -q
```

It launched production `run-headless`, control, logging, settings/secrets/discovery, recognition and translation owners. Only the capture-source/VAD seams supplied paced generated audio and a deterministic VAD sequence; real loopback HTTP and WebSocket servers returned controlled failures. No microphone, process-audio device, model installation or paid provider was used.

Instance `2fdf4a16-7bd7-4925-8200-f0f4225c1e00` produced these selected observations:

```json
{
  "manual_operation_id": "32f9bd374a664f288383d94fb1935aa3",
  "manual_request_id": "355537c23f8b372c9ffae18fee8e693b",
  "admitted_turn_id": "b3f74cd6-f86b-4d66-b841-d5684ef3f1c4",
  "translation_failure": {
    "diagnostic_id": "5602fa4d423a44c0a3d694d5dff329b4",
    "code": "provider.service_unavailable",
    "status_code": 503,
    "channel": "self",
    "generation": 1
  },
  "recognition_exhaustion": {
    "diagnostic_id": "9ed62dc615a04645ba4fbfa68d787bcb",
    "cause_id": "faccaedd02724e83bb81259dc9d432f3",
    "status_code": 503,
    "attempts": 3,
    "generation": 2,
    "final_provider_epoch": "77dacb6ef3634a10b673bbc3230341ee"
  }
}
```

Operation-filtered history connected the manual HTTP 503 cause to the original request and existing child turn, while admission still reported output delivery unobserved. Three actual WebSocket connections exercised established Soniox recovery attempts with generated audio; later attempts and final exhaustion linked the first cause. Source/provider epochs stayed distinct and did not borrow a detached command context.

Final Self capture was faulted/inactive with provider detached and no source/VAD/loop/cleanup debt; Peer remained stopped at generation 0. Retained history contained five distinct failures, no rejected fields/records and no subscriber/file projection failures. Seven repeated projections were deduplicated, not retained as extra failures. The sampled file was pending rather than declared durable; after shutdown the actual file contained all five diagnostic IDs and no private sentinel/key/body. The host exited 0.

The final 39-suite speech regression passed 1,249 tests; the independent composed fixture passed. Deepgram malformed Results versus valid no-text/ignorable messages, provider write/read/status boundaries, peer source-open/cleanup, attached-provider warmup usability, stale/cancelled work, translation source-only outcomes, probe-close masking and local CPU/GPU typed request causes have permanent behavioral coverage. Known native metadata is preserved; arbitrary prose remains unknown. These tests do not establish real device capture, paid-provider service health or physical GPU recognition.

## Output, OSC and native software path

Final composed output instance: `120a64fd-7455-4054-97a7-706ab4d1fc70`. Selected independently observed admission and later destination-failure fields:

```json
{
  "admission": {
    "operation_id": "d3ce538c55aa494bb448c94bce6c1a81",
    "request_id": "1075c87efc729be79aa4802701123b44",
    "status": "applied",
    "admission": "admitted",
    "turn_id": "f98119d0-0404-4c9a-a230-c16002cfab53",
    "child_turn_ids": ["f98119d0-0404-4c9a-a230-c16002cfab53"],
    "output_delivery": "unobserved"
  },
  "later_diagnostic": {
    "diagnostic_id": "4954ee43035e4a3ca94a5f6c69de9dbb",
    "operation_id": "d3ce538c55aa494bb448c94bce6c1a81",
    "request_id": "1075c87efc729be79aa4802701123b44",
    "turn_id": "f98119d0-0404-4c9a-a230-c16002cfab53",
    "code": "destination_publish_failed",
    "stage": "ui_event_delivery",
    "disposition": "degraded",
    "fields": {
      "destination": "dashboard",
      "publication_id": "f98119d0-0404-4c9a-a230-c16002cfab53",
      "exception_class": "OSError",
      "errno": 5
    }
  }
}
```

Exactly two loopback HTTP requests accompanied two admissions: the failed destination did not replay translation. OSC failed after two of fifteen local sends; only the successful prefix was cached. A subsequent thirteen-of-thirteen delta recovery linked the original failure. Remote delivery remained unacknowledged.

A real local subprocess, authenticated loopback WebSocket and pipe carried a native-protocol primary `render_failed`, linked `openvr_failed` cleanup, and linked `native_diagnostic_loss` under the original enable operation and generation. Seven diagnostic drops, one reliable-delivery failure, rejected detail and incomplete evidence remained observable. Stale-generation input was rejected. This subprocess was a deterministic protocol fixture, not an HMD runtime.

Six selected failure records were read from the actual main file after shutdown; all seven retained records were received by independent follow/replay. Runtime `file_evidence` stayed unknown rather than claiming durability from aggregate counts. Forbidden synthetic credential/body/text/path/stderr values were absent from exported diagnostic and selected-file evidence. Normal stop returned CLI 0/host 0 and the pinned child process identity was absent.

Separate actual GUI evidence showed the factual Basic output failure and a real desktop overlay rendering synthetic source and translation. Unaffected history/destinations continued, three admissions caused three local HTTP requests, and all seven recorded process identities exited. Final operation correlation was proved by the later CLI run; earlier GUI presentation is not substituted for that proof. Changed Flet cleanup additionally exercised actual child/PID-file lifecycle tests.

### Native build identity

Current changed Rust sources passed 289 tests, with one ignored test, and built successfully under MSVC 2022. Software loopback/runtime tests include primary render failure plus secondary OpenVR cleanup and terminal-delivery loss. The built debug executable reported:

```json
{
  "sha256": "bc4dc87f3a72e0452ab04975b8f2966c48766fd903ffc05cc0bd4f0312dd5c0d",
  "version_exit": 0,
  "version": "2.8.2",
  "startup_contract_exit": 0,
  "contract_version": 15,
  "execution_contract": {"version": 1, "revision": "r2"}
}
```

Executed native commands were `cargo test` and `cargo build` with `--manifest-path native/overlay/Cargo.toml --locked --target-dir <owned-temp-target>`, followed by the built executable's `--version` and `--check-startup-contract`. This proves compiled software contracts, not actual OpenVR/GPU/HMD rendering or recognition.

## Slow consumers, stalled storage and bounded shutdown

A separate actual composed Windows host, instance `b83c499a-5bf4-4dda-bc63-87a55873b787`, retained standard owners and control. A throwaway fixture delayed the actual file batch write only in its worker thread. Two independently authenticated metadata subscribers did not read their 1,024-byte receive buffers; a third real CLI follower continuously read. Sixteen focused assertions passed.

| Observation | Measured result |
| --- | --- |
| Actual writer stall | 22.187387 seconds |
| Real settings apply during stall | Exit 0, applied/runtime-applied in 0.215655 seconds |
| Healthy follower | 4,004 diagnostic events, no parse errors or gaps in this run |
| Slow consumers | Both disconnected independently; healthy subscriber remained |
| File queue loss | 2,017 dropped records, zero write failures during pressure |
| Retained diagnostic history | 256 records, 3,747 evicted |
| Retained event history | 256 records, 3,752 evicted |
| After releasing writer | 1,991 enqueued/written, zero pending; persistence still incomplete due prior loss |
| Bounded close under a second stall | Actual close raised timeout after 2.258681 seconds |
| CLI stop under that stall | Exit 0 in 4.744090 seconds; cleanup unobserved; host exit 1 |

The settings operation's receipt, independently queried operation state, persisted locale/revision and effective capture/translation state agreed while the writer was blocked. History explicitly reported gaps/truncation; replay reported eviction and snapshot-required. `log_queue_overflow` retained unavailable file evidence without recursively writing to the failed path.

Forbidden optional values and nested exception prose were rejected while safe code, identity and builtin cause classes survived. The marker was absent from captured CLI/follow/replay and actual main-file bytes. Post-close writer health was observed by the task-local wrapper around the real close, not by a fictional CLI query after endpoint shutdown. No post-stop durable health guarantee is claimed. All probe scripts, temporary state, sockets and owned processes were cleaned up.

## Audited failure-family dispositions

These are scoped owning-boundary decisions, not a mechanical conversion of every `except` block. Expected cancellation, no-speech, disabled output, stale generations and established fallback policy remain normal control flow.

| Owner/families | Disposition |
| --- | --- |
| Shared ingress, projections, root/owner logging | Normalize typed safe meaning before retention/rendering; preserve IDs; reject unknown optional detail; remove rendered-string reconstruction and duplicate projections. |
| File writer, queue, subscriber and shutdown delivery | Keep bounded asynchronous delivery; retain first loss/write cause independently; expose incomplete/unknown persistence and cleanup. |
| Application control and query validation | Validate at the argument boundary; execution defects remain execution failures; queries remain finite, metadata-only and instance-scoped. |
| Canonical load/migration/write/replace/validate/restore | Preserve native phase/primary cause; bounded compensation and notification secondaries cannot rewrite commit authority. |
| Settings/provider apply, full-draft resync and interruption | Separate persisted revision from effective state; retain failed/degraded drafts and actual restoration/interruption outcomes. |
| Secret backend, partial read, write, verification, commit, restore | Typed stage-specific cause; limited probe observation; no credential/body/path leakage or fabricated authentication diagnosis. |
| ChatGPT/managed Discord/QQ workflows | Distinct sign-in, refresh, local removal, remote revocation, compensation and cleanup; existing recovery policy retained. |
| Managed background status/usage and optional publishers | Preserve explicit admission identity and lifecycle policy; report delivery secondaries without undoing completed actions. |
| Startup fallback, GUI queue, ordered shutdown, scene observation | Truthful fallback persistence, pending edits, continued queue/cleanup, timeout/abandonment and unknown observation. |
| Self/Peer admission, source/VAD, provider build/handoff and warmup | Actual native cause and generation association; first cause before cleanup; attached retryability, effective capture and GPU readiness separated. |
| Scoped recognition begin/write/read/seal/epoch, recovery/exhaustion and drains | Typed terminals, exact source scope, bounded linked recovery/cleanup; existing attempts/backoff and cancellation preserved. |
| Deepgram, Soniox, Gemini, Scribe and Qwen protocol adapters | Known transport/status/protocol cause before reduction; malformed versus legitimate no-text/cancellation distinguished; bounded callback/cleanup evidence. |
| Custom STT and local CPU/GPU request/decode | Safe HTTP/protocol/decode cause, actual provider/channel identity, no body/audio logging or recognition algorithm change. |
| Rolling selection and probes | Existing exclusion/retry policy retained but not used as diagnostic truth; key/protocol observation does not establish live readiness. |
| Translation request/turn/LLM adapters and managed wrappers | Typed status/response cause; same identity across wrappers; explicit manual parent/child/operation links; stale/callback/retirement observations preserve primary. |
| Output fanout, router, chatbox and optional observers | Independent destinations; actual publication identity; bounded observer loss separate from action outcome. |
| OSC state/cache/discovery | Successful-prefix cache, actual local partial/recovered outcomes; intentional OSCQuery fallback retained; no remote acknowledgment. |
| GPU startup/client/runtime/teardown | Primary versus cleanup cause; explicit factory/runtime debt; retain ownership until resource exit/close confirmed; no utterance retry. |
| Overlay bridge/process/readers/native status | Finite child ingress, stale rejection before coalescing, separate primary/cleanup/delivery loss, reserved control capacity and bounded artifacts. |
| Desktop renderer and Flet child lifecycle | Independent bounded cleanup, retained unfinished resources, safe original cause, actual child identity checks. |
| Optional non-operational UI notifications/probes | Preserve deliberate best effort; do not retroactively fail completed publication or invent resource/remote-delivery authority. |

Permanent regression coverage includes `tests/core/test_diagnostics.py`, `test_diagnostic_delivery.py`, `test_runtime_logging.py`, `test_speech_diagnostic_causes.py`, `test_speech_diagnostic_boundaries.py`, `test_speech_diagnostics_composed.py`, `tests/cli/test_transport.py`, `tests/app/test_application_diagnostic_causes.py`, settings/secret/account/lifecycle suites, scoped/provider/capture/translation suites, output/OSC/GPU/overlay/UI lifecycle suites, and `native/overlay/tests/runtime.rs`. Assertions cover consumer outcomes, identity, privacy, precedence and lifecycle; obsolete summary/API-membership-only assertions were removed rather than re-pinned.

## Original live-source fingerprints and limits

The settings/failed-file and pressure runs exercised `core/runtime_logging.py` SHA-256 `a0ea46cc6968bcf4ddcd32e09544dddf10fa6b94c1b7e32c403c16e17da8e1ac` and `app/services/application_control.py` SHA-256 `95d1efa54441bbb295b0f03adc5da333d3b9bffd679deec457633401257a8a7b`. The settings probe was `1fefafbe6546d89a09f72df476b4d92a1014c1bd3dd0fca4390ae90edf7a2166`; the output probe was `b4bc926cdea196502e0c7318a161100cad1f22e16d30ed2dd7fc4e6ff930458d`. The removed application fixture's UTF-8 source digest was `8b74ed348e6de30dc487962dd57c0d31987e6befffa38b587f2f5e647bc895e1`. Later finite provider-label additions have their own focused ingress checks; earlier live proof is not represented as exercising those newly added labels.

No real provider account login/refresh/revocation, paid request, ambient capture, physical GPU recognition failure, actual OpenVR session or HMD presentation was exercised. Installed hardware/tooling is not treated as execution evidence. Deterministic local failures establish the stated software behavior only. History remains finite and restart-scoped, main-file evidence is not a lossless durable audit, and process exit is not universal cleanup confirmation.
