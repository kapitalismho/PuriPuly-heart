# Diagnostic failure verification — #228

Implementation authority: [#228](https://github.com/kapitalismho/PuriPuly-heart/issues/228), with child outcomes #229–#232. Source baseline: `75c4485a6787d504f5335d1602973f63b4d0c234`. This record distinguishes executed software observations from hardware or external-service claims. Original live observations appear first; the reviewed repair checkpoints and their later evidence are recorded below.

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

The original 39-suite speech regression passed 1,249 tests; the independent composed fixture passed. Provider write/read/status boundaries, peer source-open/cleanup, attached-provider warmup usability, stale/cancelled work, translation source-only outcomes, probe-close masking and local CPU/GPU typed request causes have permanent behavioral coverage. The original evidence did not establish malformed Deepgram callback-to-epoch disposition; the bounded terminal-audit closure is recorded separately below. Known native metadata is preserved; arbitrary prose remains unknown. These tests do not establish real device capture, paid-provider service health or physical GPU recognition.

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

## Reviewed repair checkpoints

The first integrated checkpoint was `48d8a08bf84460c24ebf8c8566d57e323c86382e`. Independent whole-code and focused-privacy reviews found eleven material issues. None was waived. Repairs were pinned at `02657c0f679e650bf7208deec0aea298330f1d93`; a bounded follow-up identified remaining Peer admission paths, repaired at `a665c294b5cc8bcb270ddbb0b869f2026b10bc05`. Both originating reviewers verified the remaining repairs and retained unaffected coverage.

The repairs cover:

- The remaining GitHub prompt persistence caller, native lazy-builder failure handling, and truthful bounded secondary-cause omission counts.
- Explicit Self and Peer provider admission identity before first retention, including inactive preparation and a second provider command that reuses the running capture generation. Startup and handoff preserve the native cause through convergence and control receipts.
- Failed OSC full-publication obligations: empty or unrelated deltas do not establish recovery.
- GPU activation failure followed by actual temporary-directory cleanup denial: primary cause, settled state and resource ownership survive.
- Fixed CLI input-error metadata, finite renderer event/disposition validation before artifact retention, known desktop startup codes and visible top-level detail rejection.
- Safe Self/Peer snapshot projection without `MappingProxyType` deep-copy failure or raw diagnostic-field serialization.

The last Peer/control delta changes neither the shared normalizer nor file/subscriber delivery. The exact-`02657c0f` foundation and pressure observations below remain applicable to their exercised paths; they are not relabeled as new executions on the later commit.

### Completed regression coverage

One monolithic Python run timed out after 1,800 seconds at approximately 92%, with failures and no final XML. It is **not** a completed passing run. Its obsolete consumers and concrete defects were repaired, then complete, bounded owner partitions and cross-cutting partitions were executed.

The final command inventory covers all **462 nonpaid Python pytest files**. Nine paid/external/model integration files are explicitly excluded. The three deterministic nonpaid integration files are included, with `INTEGRATION=1` restricted to the required files. Whole-file/directory commands establish file coverage; targeted repairs do not receive whole-file credit. Overlapping memberships are deduplicated.

| Verification group | Observed result |
|---|---|
| Foundation's complete 21-file area | 328 passed |
| Application, UI and application-owned core partitions | 2,380 passed; later affected convergence subset 154 passed |
| Speech's complete bounded partitions | 2,362 passing executions, including overlapping cases |
| Output's complete area | 1,583 passed, zero failed or skipped |
| Director cross-cutting partition | 1,234 passed, three skipped, two failures subsequently closed individually |
| Director remaining HTTP-extension file | 31 passed |
| Final Peer affected whole-file suite | 112 passed |
| Final Peer composed startup/handoff file | Four passed |
| Final typed control-carrier regression | Three passed |
| Native Rust software suite | 289 passed, one ignored; unchanged native source |
| Repository Python lint | `ruff check src tests` passed at the final repair pin |

These are separate suite results, **not an additive unique-test total**. The interrupted monolithic run is not hidden by the partition results. Conditional hardware/model skips and the ignored native test are not passes.

### Refreshed Windows delivery and control observations

On `02657c0f`, the standard composed settings-denial/file/restart probe exited zero in 15.627 seconds. Seventeen relevant source hashes were identical before and after.

- Actual settings replacement denial retained `settings_save_failed`, `PermissionError`, errno 13 and winerror 5. The CLI exited four; settings bytes and effective locale remained unchanged. Receipt, later operation lookup, independent history, follow replay and the single selected main-file record shared diagnostic `56bd1a5424534e2c8dff7a1c8a45f51c`, operation `a2234894125d48eabd123033bcc5728d` and request `5cf27edb331ba64d07bcdfe04f724885`.
- Healthy delivery observed six enqueued/six written, no pending records or loss, and explicitly `written_not_durable`. Restart lookup reported `foreign_instance`, empty results and a gap.
- An actual directory at the log-file target produced independently queryable `log_file_delivery_failed`, zero written/four delivery failures and unavailable file evidence. CLI stop exited zero with cleanup unobserved; the host exited one.

The same original pressure scenario was refreshed on that pin: all sixteen assertions passed with sixteen identical before/after source fingerprints.

- The actual file worker stalled for 22.092593 seconds while a locale mutation completed in 0.225119 seconds; CLI state and committed bytes independently agreed.
- Two authenticated no-read consumers disconnected independently. A healthy CLI follower received 4,004 events without parse errors or gaps.
- The bounded queue dropped 2,017 records; history/replay disclosed eviction gaps. Rejected optional detail preserved failure identity, and the actual file later contained safe metadata, not the forbidden values.
- Bounded logging close raised `TimeoutError`; host exit one remained distinct from CLI stop zero/cleanup unobserved. All recorded owned process identities exited.

The refreshed pressure report SHA-256 is `5963912020c33bc8aecdcf56e2fac2acf7811b163d04102f5d91492921b382d8`. Post-close health is a direct observation around close, not a fabricated CLI query after endpoint freeze.

### Final admission and output closure observations

Four actual Self CLI cases cover active/inactive provider creation and warmup failure. Operation-filtered history retains the first native diagnostic, and the degraded transaction refers to that same identity.

Four actual Peer CLI cases cover startup and running handoff, each with creation and warmup failure. Startup receipts preserve the native `cause_diagnostics` and an explicitly linked command wrapper. Handoff transactions retain the native diagnostic ID, the second command's identity and the unchanged capture generation; the previous attachment remains active. Inactive Peer preparation has native-owner regression coverage, not an invented CLI observation.

Peer fixtures preseed terms acceptance only in isolated initial test settings; no command accepts real-user terms. Their overlay prerequisite is fixture-connected, and audio/VAD are generated. These cases prove native ownership, correlation and outcomes, not ambient capture, physical overlay readiness or model/GPU execution.

Output closure additionally exercised real loopback OSC sends after a failed unchanged-state resynchronization, a real native-protocol child whose invalid renderer strings were excluded from its automatically written failure artifact, and an actual Windows delete-sharing denial during GPU-protocol-child cleanup. The OOM primary remained distinct from linked cleanup `PermissionError`, and later explicit close released the owned process and directory.

Latest output evidence contains 113 exact source/test hash matches and three explicitly reconstructed import-only differences. The known pre-format import blocks reproduce the recorded hashes exactly; production AST and relevant test bodies are unchanged. All eight separately recorded composed-output source hashes match the repair pin. This is bounded evidence applicability, not a new test execution.

### Application refresh and GUI evidence limit

The later production-composed headless refresh exercised the real public GitHub prompt mutation without opening a browser and observed its durable clicked flag. A separate real settings mutation committed revision three to four with locale `ja`, while an explicitly synthetic runtime-apply failure left effective locale `en`. Its receipt retained four native secondary causes, reported four known omissions and set truncation true. Five retained history records shared the initiating operation/request and expected identities, with zero rejected detail. Normal stop and host both exited zero; process disappearance still did not claim universal cleanup.

The original actual GUI observation remains the evidence for visible Basic presentation, retained pending draft and the next successful queued action. Its recorded `ui/app.py` hash, `4d6a505ef0c182d679322a0656da528d8273398e9f7a382b82c0c77c5f0352b0`, exactly matches the historical first checkpoint. A bounded comparison to the later source finds only the queue's data-only application-instance/request identity change; other UI application functions and the settings renderer are unchanged.

Two later actual native GUI hosts, including one controlled restart on `a665c294`, attached the Basic view and owned their Flet windows, but bounded owned-window capture returned a complete black surface. Visibility, restored state, DPI, cloak, capture affinity and typed `PrintWindow` handling did not establish a fresh observable frame. Window-ready and `ui_attached` are not rendering acknowledgments. **No fresh visible Basic, pending-draft or next-mutation pass is claimed for those attempts.** The prior GUI observation and its source-applicability map remain separate evidence, not fabricated fresh pixels.

Both later GUI hosts stopped normally, and all six recorded owned GUI process identities were absent afterward. Temporary settings, secrets, discovery, logs, screenshots and execution scaffolds were removed; non-application captures were discarded immediately. The unresolved fresh-capture prerequisite is recorded rather than replaced by a headless or model-only GUI claim.

### Terminal-audit Deepgram disposition closure

The final audit found a verification gap, not a new production defect: the earlier parser probe did not prove malformed input's observable scoped outcome. This was closed without production changes by extending `tests/providers/test_deepgram_reuse.py`. The whole file passed **20 tests in 30.63 seconds**, including twelve new installed-SDK callback cases; its Ruff check passed.

`test_installed_sdk_malformed_callback_fails_scoped_turn_and_epoch` covers missing/invalid Results structure and invalid JSON, both without and with a prior fragment. The actual installed Deepgram SDK 5.3.4 consumes in-memory wire messages through the production session's registered callbacks. Malformed Results reaches `_handle_message`; invalid JSON raises the SDK's `JSONDecodeError` and reaches the registered error callback instead. Consumers observe a failed or degraded terminal and an unordered ended epoch, with `deepgram_protocol_failed`, retired source identity and the same native diagnostic across both events—not absent output, ordinary empty recognition or a later finalize timeout.

`test_installed_sdk_valid_callbacks_preserve_empty_and_cancellation_semantics` distinguishes valid empty-without-ack, whitespace, ignorable Metadata and authoritative empty-with-ack. Whitespace may produce a blank transcript update; that is not a protocol failure. Existing acknowledged-empty reuse coverage remains, and explicit cancellation is still cancellation.

A separate throwaway native-callback smoke exited zero. With a thirty-second finalize drain armed, malformed Results produced the failed terminal and epoch in 1.8035 ms; invalid JSON did so in 0.2944 ms. Their native diagnostic IDs were respectively `65537b305f32451ea1ba020450e31c7d` and `7728a0e11abb4f2fbb98d7def7f95e49`, with code `stt.deepgram_protocol_failed`, category `invalid_response` and matching channel/turn/generation/provider identities. The original exceptions were `ValueError` and `JSONDecodeError`; the strict safe class catalog intentionally projects the latter as `unclassified` without losing the protocol code/category or widening the metadata schema.

Private payload, transcript and key markers were absent from the safe diagnostic projections. Production and test hashes were unchanged across the probe; its script/directory and owned tasks were cleaned up. This proves installed-SDK/native-consumer software disposition using an in-memory wire, not network service health, paid-provider operation or audio hardware. Existing SDK/Python compatibility warnings were retained rather than suppressed.

### PR CI publication and formatting repair

PR #235's first CI run (`38459324503`) failed Black's formatting gate on 144 files and one Runtime artifact-access assertion; Runtime otherwise reported 1,587 passed and three skipped. Earlier Ruff success did not establish Black compliance. The locked Black 26.5.1 formatter was applied, after which `black --check .` reported all 1,055 files unchanged and `ruff check .` passed.

The Runtime test had treated process state `failed` as proof that its diagnostic artifact was published. A controlled Windows probe showed that this state precedes completion of the failure monitor and writer; calling `stop()` then legitimately records `abandoned` / `dump_cancelled` while the bounded daemon writer remains unfinished. Completed pipe-reader cleanup does not establish file publication. The original CI `PermissionError` records errno 13 but no Windows denial subtype; it does not establish a particular ACL or sharing-violation defect.

The real-pipe/loopback test now awaits its owned failure monitor and requires a `written` receipt before reading the matching artifact. It additionally verifies the published byte count and SHA-256, retaining its original cause, stale-generation and privacy assertions. Production cancellation, abandonment and ACL behavior are unchanged. The complete `tests/app/test_overlay_process_manager.py` passed 118 tests before formatting and again in 3.90 seconds afterward, including the existing cancellation/abandonment coverage.

A separate actual Windows smoke awaited publication of a 139,629-byte artifact and matched its receipt hash through both `Path.read_bytes` and Win32 `CreateFileW` / `ReadFile`. Its DACL fingerprint was unchanged, private markers were absent, and the primary/cleanup causes and loss counters remained intact. The smoke used explicit event/WebSocket synchronization before releasing the terminal fault, rather than assuming diagnostic ordering from elapsed time. Its child, manifest, temporary files and native handle were cleaned up, and its closed bridge port was successfully rebound. This proves native-protocol and filesystem access, not physical OpenVR/HMD execution. Hosted CI results for the follow-up commit remain separate from these local observations.

### Bounded PR review feedback

The CI repair commit `73f3628e` passed all nine hosted jobs in run `38463418919`, including Black, Ruff and Runtime. The following review-feedback changes are a separate candidate, not implicitly covered by that run.

Three comments were assessed against their actual impact and repair complexity. Nine newly added explanatory comment lines violated the repository rule and were removed without changing behavior; existing comments, attributes and type-checker directives remain. The OpenRouter label mismatch was reproduced through the actual handoff producer and retention, but that service has no caller in the inspected current application composition, so a current-user P1 incident was not established. Its three known labels were added to the existing finite catalogs without new wiring, registry machinery or optional private fields.

Malformed operation/diagnostic filters previously returned `not_retained`. The shared payload pattern now distinguishes bounded continuous hexadecimal and UUID-shaped identities from the reported hyphen-only/trailing-separator values. Existing native prefix syntax, short overlay hexadecimal identities, admission sentinel, managed-operation identities, integer capture epochs and host instance syntax remain unchanged. Some strings can still validly parse as an existing prefix plus hexadecimal payload; this is not a UUID-only redesign.

The two affected whole test files passed 50 cases under the project Python environment. Global Black and Ruff checks passed. A separate isolated real headless host and CLI returned `invalid_arguments` with exit three for `operation_id=--------` and `diagnostic_id=dead----`, while a valid unknown hexadecimal ID returned `not_retained` with exit zero. An actual handoff-service result retained the same native identity, specific component/code and degraded disposition through history, event and file projections; Basic used the specific labels, while unsupported private detail still reported rejection. No account, browser, paid-provider, capture, GPU/HMD or GUI action was exercised. The probe host stopped, logging closed and its temporary directory was removed; no stronger graceful-host-cleanup claim is made.

## PR #234 diagnostic selection — #237

### Inspected revisions and selection boundary

- PR #234 head: [`86d0e698de7cefb8b6360f8e119ec63beb4ae2b0`](https://github.com/kapitalismho/PuriPuly-heart/commit/86d0e698de7cefb8b6360f8e119ec63beb4ae2b0), based on `75c4485a6787d504f5335d1602973f63b4d0c234`.
- Merged #235, current GitHub `dev`, and this worktree's starting HEAD: [`e4c77a272f5f1867b5cc7753c6071c20ba943339`](https://github.com/kapitalismho/PuriPuly-heart/commit/e4c77a272f5f1867b5cc7753c6071c20ba943339). The final #235 premerge head was `3b185120f633f1117ce9dde0e1e12f54a97ee306`, not the earlier `0a4fec7d` probe baseline.
- The 77-file #234 comparison was inventoried; review was limited to diagnostic additions and immediate producers. Model management, new source/download/import behavior, floating chat/windows/IPC/preferences, recording/playback, BYOK fixes and Beta packaging remain excluded. None is a dependency or acceptance obligation.

### Dispositions

| #234 change and concrete producer | Disposition against merged upstream |
| --- | --- |
| [ASR installer source exhaustion](https://github.com/kapitalismho/PuriPuly-heart/blob/86d0e698de7cefb8b6360f8e119ec63beb4ae2b0/src/puripuly_heart/core/local_asr/local_stt_runtime_installer.py): native HTTP/transport cause and status before the install wrapper; existing provisioning catch adds a `ModelDownload` summary. | **Independently useful; retained narrowly.** Existing CPU/GPU install and repair producers already use this installer. The controlled CPU failure below established that the original owner/composition exposed only a Basic exception-type summary and failed model state, not retained cause/status. Preserve the first native cause and classify it with existing typed error machinery, then emit one owner-authored `ErrorDiagnostics`. Do not copy #234's last-attempt-only reduction, feature helper, `failure_codes` result protocol or new source policies. |
| [Managed Gemma provisioning logs](https://github.com/kapitalismho/PuriPuly-heart/blob/86d0e698de7cefb8b6360f8e119ec63beb4ae2b0/src/puripuly_heart/core/local_translation/provisioning.py): `ModelDownload` failure summaries for existing download/verification. | **Useful metadata already covered.** `ManagedGemmaTranslationOwner._failure_metadata` and composed Basic/diagnostic sinks already expose phase/code, safe cause type and available HTTP/OS codes for the existing producer. Do not add a second generic summary. The proposed `GemmaInstallInvalidError` → `checksum_failed` mapping is not justified: that exception also represents missing files and size mismatches, so it would assert a cause not established by the type. New disk preflight is operational policy, not a logging correction. |
| Model-assets dialog, `install_assets`, local Gemma import/file picker, public mirror downloader and explicit-source callbacks. | **Dependent on excluded functionality.** Their picker/import/source/task codes describe producers not being adopted. Catalog/download support without a logging call is not an independent logging improvement. Do not import their callbacks, codes, labels or tests. |
| `QuickTranslateOwner` translation/backend, preferences, window/IPC and presentation logs; optional quick-history callback in `event_dispatch`; child presentation reports. | **Feature-dependent, with ordinary upstream failures already covered.** The new window/history/IPC producers are excluded. Existing manual submission and translation failures retain `manual_typing` and translation-owner diagnostics; actual dashboard/history destination exceptions retain `output_runtime` publication identity through the existing dispatcher. No floating admission or request-correlation adaptation is retained. |
| `AudioTest` device-open/capture/preflight logs in the existing microphone test. | **Duplicate existing observations; no demonstrated independent benefit.** Existing capture callbacks retain failed readiness/state, while the composed `MicTest` Basic path reports source-open/capture and failed recognition-stop causes. The extra generic feature records do not preserve an additional typed cause or identity. The new recording timeout, channel fallback and device-metadata callbacks change or depend on excluded recording behavior. |
| `AudioTest` playback/start/completion/restore logs in microphone owner, UI and application boundary. | **Dependent on excluded recording/playback and capture-restoration behavior.** These producers do not exist in the retained upstream feature set. |
| [Shared `log_feature`](https://github.com/kapitalismho/PuriPuly-heart/blob/86d0e698de7cefb8b6360f8e119ec63beb4ae2b0/src/puripuly_heart/core/feature_diagnostics.py) and desktop-feature log-lifecycle tests. | **No independent helper/test import.** The helper sends ordinary Basic text through existing `emit_basic_log`; its stage/code check is lexical, not a finite catalog, and it has no owning diagnostic identity. Existing session logging already owns Basic/file delivery and view replay. Synthetic helper calls surviving a dialog close do not prove a missing upstream failure record. Preserve the merged typed ingress, privacy filters and bounded retention instead. |
| GPU worker `CREATE_NO_WINDOW`, packaging changes and other non-diagnostic edits. | **Outside the diagnostic selection.** The worker flag is a process-window behavior change, not a new diagnostic producer. |

### Retained implementation and controlled before/after evidence

The existing `LocalASRProvisioningOwner` now captures the operation/request identity at installation admission and emits `local_asr_install_failed`, component `local_asr_provisioning`, operation `models.install`, stage `installation`. It reuses `provider_failure_report` for typed status/category and safe exception/transport metadata. The validated existing model ID and source-owner generation travel with that fact. Only the live producer's component, stage, code and finite existing ASR model IDs were registered; no arbitrary feature vocabulary was admitted.

The existing installer still attempts its existing sources and produces the same aggregate error/result behavior. On exhausted fallback, its wrapper chains the first native failure rather than replacing it with the last source's error or inferring a cause from rendered prose. This is one terminal install observation, not a per-source failure audit. Unknown causes remain unknown. Expected cancellation bypasses failure emission. The existing Basic summary is unchanged; the additional record is technical file/CLI evidence only. No GUI logging widget or interaction changed, and diagnostic projection does not change model integrity, installation state or retry authority.

The throwaway Windows probe used the established Python 3.14.7 environment with this worktree's `src` on `PYTHONPATH`. It launched the real `run-headless` composition and independent authenticated CLI clients. Only controlled fixture inputs were supplied at the existing provisioning factory: a temporary model root and manifest source URLs directed to a loopback HTTP server returning 503 with private URL/body sentinels. The normal installer, provisioning owner, control host, logging ingress and CLI were exercised. Settings, discovery/data and empty encrypted-file secrets were temporary; provider credential fallbacks were removed, telemetry and OSC were off, and no capture was requested.

| Observation | Merged baseline | Retained correction |
| --- | --- | --- |
| `models.install` for CPU `qwen3-asr-0.6b-int8-sherpa` | Terminal `degraded`, CLI exit 4 | Same terminal status and exit code |
| Effective model state | Integrity `missing`, operation `download_failed` | Unchanged |
| Operation-filtered `diagnostics.history` | `not_retained`, no records; unfiltered history also empty | One `local_asr_install_failed` record, category `service_unavailable`, HTTP status 503 |
| Failure identity and projections | No retained diagnostic identity | Admission operation/request and generation 1 retained; authenticated `logs follow` replay and the selected main-file JSON equal the history record |
| Privacy and host shutdown | Private URL/body markers absent; owned host exit 0 | Private URL/body markers and personal paths absent from selected diagnostic projections; owned host exit 0 |

Baseline instance: `606b949a-bb9f-4317-9c1e-59cc9d502229`, operation `b23498816be746169728a779649cc742`. Final source-composed instance: `77be846c-4675-47d1-b519-fb9098327449`, operation `e465794c444b43b0b29cc1f79a8cb7e3`, request `76a8c78d9f56a0d17c225fe218b4fb92`, diagnostic `ccd2a19552174d09b1ea79efd4ee8b1d`. A diagnostic-ID lookup returned that same record. Cause/exception classes unsupported by the merged finite catalog intentionally remain `unclassified`; the typed HTTP status/category is not lost and no exception-class catalog was widened. Observed selected file bytes establish the matching projection, not fsync or durable persistence; `file_evidence` remains `unknown`.

### Regression coverage and limits

The eight affected/adjacent behavior suites passed **155 tests in 50.64 seconds** using the established interpreter and this worktree's source:

```text
tests/core/test_local_stt_runtime_installer.py
tests/core/runtime/test_local_asr_provisioning.py
tests/core/test_diagnostics.py
tests/core/test_diagnostic_delivery.py
tests/core/test_runtime_logging.py
tests/core/test_speech_diagnostic_causes.py
tests/app/test_application_control_contract.py
tests/app/test_local_asr_gpu_provisioning.py
```

The installer regression distinguishes the first failing source from the fallback source. Owner regressions cover native HTTP versus misleading arbitrary exception prose, private metadata exclusion across history/event/file, preserved admission identity after its context expires while an unrelated operation is active, validated model identity/generation, unchanged failed integrity, and normal cancellation without a failure record. The existing delivery suite covers repeated projections retaining one failure identity. An initial cancellation-test attempt wrongly assumed an empty file must exist; that test-only assumption was removed without changing runtime behavior.

No GUI run is claimed because the Basic/view surface is unchanged. Real model downloads, disk exhaustion, physical microphone/GPU/OpenVR/HMD execution, paid-provider health and excluded desktop/Beta behavior were not exercised or required. Verification above was completed in the source worktree before committing or publishing. No merge, deployment or release is implied by this evidence. Logging ownership and architecture boundaries remain unchanged.

After the final test-only import-order correction, the whole provisioning-owner module passed **19 tests in 0.48 seconds**, and Ruff passed for all six affected Python files. Black formatted the changed owner file and left the other five affected Python files unchanged.

A pre-publication Black check also reported all six affected Python files unchanged.

## SELF mini-turn captions and latest-scene protection — #265

Authority: [issue #265](https://github.com/kapitalismho/PuriPuly-heart/issues/265).
This is a historical verification record. The SELF publication policy below is
superseded by [#279](#readable-self-translated-units--279); its native freshness
and physical-observation limitations remain relevant.
Implementation started from `639144cb27c3f5fa69a90375e49360c505dbc7b3`.
The initial integrated product was `458b1faa39f5c728d7e4ca7d20d83899637020b7`;
`eba9abb81404019e8912d37dc47240599a7bc0f3` changed only a superseded
same-episode test assertion and blocked-fixture cleanup. These local checkpoints
are not a published PR, release, or issue-closure receipt.

### Controlled source and output preservation

The locked Python 3.14.7 environment used this checkout's editable package,
NumPy 2.5.1 and websockets 16.1.1. Actual source, translation, projection,
output, paginator and presenter owners consumed synthetic input and a held
local provider. The recording sender was in-process: these results do not claim
UDP reception or VRChat display.

The baseline/candidate comparison fixed canonical UUID allocation, input order,
provider releases and the production 400 ms grace. Its schedule was
`alpha` → `alpha beta`, speech end, translation release, compatible
`alpha beta.` punctuation reuse, 450 ms timer service, then `bravo`,
speech end/result release and another 450 ms timer service.

| Route | Overlay delivery | Requests, baseline/candidate | Local-sender payloads, baseline/candidate | Canonical identity, revisions, closure and target coverage |
| --- | --- | --- | --- | --- |
| Stable append | Normal | 3 / 3 | 3 / 3 | Identical |
| Stable append | Blocked | 3 / 3 | 3 / 3 before sink release | Identical |
| Stable append | Disabled | 3 / 3 | 3 / 3 | Identical |
| Final only | Normal | 2 / 2 | 3 / 3 | Identical |
| Final only | Blocked | 2 / 2 | 3 / 3 before sink release | Identical |
| Final only | Disabled | 2 / 2 | 3 / 3 | Identical |

Payload order was `alpha beta (T(alpha beta))`,
`alpha beta. (T(alpha beta))`, then `bravo (T(bravo))`.
All cases ended with zero unsent destination batches. Blocked cases applied no
overlay blocks before release while requests and sender progression completed.
This is schedule-controlled preservation, not identical wall-clock timing.

### Initial actual desktop observation

The production Flet desktop child received the actual source-owner → translation
owner → output → presenter → bridge/mailbox path. Only source/VAD input,
translation responses and the external sender were controlled. This was the
actual application overlay, not the full main GUI or a live ASR/LLM service.
SystemClock, 400 ms grace, contract 16 and Flet/flet-desktop 1.0.0 were recorded.
Private manifests and ephemeral loopback ports isolated the run. Captures
covered only the owned synthetic backdrop and overlay region.

Observed pixels showed no caption during a held-translation stable burst, source
at the first boundary, matching translation attachment, unchanged resumed
source, second-boundary growth with the prior translation, and newer completed
source/translation before another boundary. Ready, pending, disabled and failed
closure paths were observed. A retained predecessor patch changed only its
translation region; the successor's pixels stayed identical. Mixed PEER input,
displacement without revival and independent final-only source-first display
were also observed. Four fast completed translations arrived about 46–47 ms
apart; the latest mixed scene remained correct.

Actual edit-mode drag moved the owned window from `(0, 0)` to `(40, 40)`.
Pass-through delivered two clicks to the owned backdrop while caption pixels
remained unchanged. Earlier launcher/Pillow/privacy-guard failures and a
synthetic-backdrop activation failure were retained, not counted as passes.
The final backdrop-only correction enabled the successful interaction check.
All owned processes exited and private manifests were deleted.

### Native execution, work bounds and the discovered empty-scene failure

The initial Windows release SHA-256 was
`cce0d5a4a35488baff369a71eabd3b1bda623dca4e47c2981b06e5b00f9ae04b`.
Its startup contract was bridge 16, execution r2, exclusive native retry v2 and
speaker identity v2. P05 and handoff `off` were explicit.

Actual Python projection/mailbox JSON crossed a loopback WebSocket into the
native parser, reducer and presentation owner. Controlled renderer/submitter
checks required active/due schedules and matching fresh-render, producer-ready
and successful-submission witnesses. They covered wall/count exhaustion,
sub-100 ms coalescing, retained patches/removal, transport held 800 ms,
send-time expiry, delayed readiness, no-progress, clear coalescing and fresh epochs.

| Same-caption workload | Fresh renders / submissions | Observed interval |
| --- | --- | --- |
| Six milestones 600 ms apart | 28 / 28 | 4,313 ms |
| Six milestones 30 ms apart, then silence | 11 / 11 | 1,476 ms |

Intervals start before the loopback owner connection and end after joined
shutdown, including one empty frame and 600 ms settled-empty service. They
exclude Python fixture production and compilation. These controlled software
counts do not measure GPU utilization or HMD latency. Sparse rearmed episodes
can perform more total work per caption; the preservation claim is about
per-episode and simultaneous ownership bounds, not identical total work.
The pressure scenario retained one pending scene and bounded receipt storage.

A separate actual D3D11/OpenVR run with a unique owned overlay key recorded
31 correlated completions. Sparse rearming, count-exhausted growth, fast latest
tail and SELF removal with unchanged translated PEER passed. Fully empty
revision 13 then failed with `openvr_failed`, native exit 1 and no graceful
shutdown. Handle disappearance before 500 ms was error teardown, not successful
Hide. The original binary discarded the exact API/event payload; its original
cause remains unknown.

One separately instrumented archive of the pinned candidate distinguished the
failure stages. Empty revision 2 rendered a transparent frame, became
producer-ready and completed `SetOverlayTexture`. An overlay `ProcessQuit`
event (701) for a different process was then classified fatal with 377 ms of
Hide grace remaining. No texture/calibration/Hide API error was observed.
This proves a reachable PID-unaware event-classification defect; it does not
recover the original run's missing payload. The repair preserves process
identity through both event sources and ignores unrelated process quits while
retaining genuine global/owned-process termination. Renderer, P05, event budget
and the 500 ms Hide contract are unchanged.

### Independent findings and verification limits

Independent whole-checkpoint and focused-native review identified three
additional defects: queued first admission could be lost behind a blocked
predecessor; a chatbox-only setting change could blank compatible sticky text;
and secondary-only PEER survivors could lose SELF scene protection. Repairs
preserve the queued admission's position and appearance authority, reuse the
translation owner's configuration-compatibility comparison, and use effective
drawability for scene-level eligibility. Translated-primary and secondary-only
survivor controls are both retained.

A subsequent recovery-boundary review found that a fresh native epoch still
dropped applicable SELF scene coverage over an unchanged secondary-only PEER
survivor. Fresh-epoch seeding now preserves that eligible scene at trigger 1,
episode 1 and the current phase without renewing the original caption deadline.
Actual-presenter checks preserve identity/order and expire at that original
deadline. Ordinary PEER without prior SELF intent, hidden/empty/expired scenes,
disabled epochs and cleared scenes remain negative controls. Both secondary-only
and translated-primary recovery cases cross Python JSON into a fresh native
owner and require active/due schedules plus correlated render/readiness/submit
success. An initial test-only field-name compilation error was corrected to the
existing wire field, not a product change.

### Final software verification

The repaired caption/admission, drawability and event checkpoint was
`82f37eabf150e4ae7780ad810de13379eb858727`. The final product checkpoint is
`cc08aee64517989e0e0b883850fb60b420d8940d`: its only additional product change
is fresh-epoch seeding in `NativeRetryIntentProjection.begin`. Independent
whole-checkpoint and focused-native repair reviews verified all accepted
findings, retaining unaffected original review coverage. They are not an
issue-closure or physical-observation approval.

| Executed selection | Applicable checkpoint | Result |
| --- | --- | --- |
| Source, translation, caption, output and pressure regressions | `82f37eab` | 499 passed |
| Overlay, desktop and configuration regressions | `82f37eab` | 372 passed; 42 existing deprecation warnings |
| Architecture contracts | `82f37eab` | 378 passed |
| Full native default-parallel suite | `82f37eab` | 302 passed; one opt-in memory probe ignored by default |
| Contract/process compatibility selection | Initial contract-16/retry-2 implementation, unchanged afterward | 215 passed; no skips |
| README real-child integration selection | Initial integrated checkpoint, unchanged contracts afterward | Five passed; no skips |
| Scene protection and active freshness | `cc08aee6` | 26 passed |
| Native Python cross-language selection, including two new recovery cases | `cc08aee6` | Eight passed; 102 other runtime tests filtered out |

These selections overlap and are not a summed suite total. The six fixed-schedule
baseline comparisons were repeated after the admission/configuration repairs
and remained identical. Global Ruff and Black checks passed at `82f37eab`;
Black checked 1,123 files. The three Python files changed afterward passed their
focused Black/Ruff checks. Changed native files passed rustfmt. Whole-tree
rustfmt still reports differences in unchanged `logging.rs`,
`renderer/backend.rs`, `renderer/layout.rs` and `tests/renderer.rs`; those files
were not reformatted. The default-ignored memory probe was separately executed:
8,192 offscreen render/readiness operations passed, without HMD observation.

An earlier parallel native run had four `overlay control write blocked` failures.
The serial run passed, and the final default-parallel run passed after other
native jobs had stopped. An environmental cause was not proven; the failures
are retained, with no timeout relaxation or weakened product assertion.

Actual desktop pixels at `82f37eab` confirmed repaired queued A/B admission,
appearance order and focus, plus unchanged sticky translation/provider work
across a compatible chatbox-only setting change. Safe representative frames
also covered mini-turns, ready/pending closure, retained updates and mixed PEER.
The extended driver later exited 1 during a capture safety interruption.
Six late images and their derived sheet were deleted and excluded; no final
late-terminal/fast-frame or new interaction pass is claimed. The earlier
unchanged terminal and drag/pass-through evidence remains separately attributed.
No further capture was attempted. Owned GUI resources, authenticated manifests,
runner scripts and generated profiles/caches were removed.

### Final actual OpenVR continuation

The matched release SHA-256 is
`a4d79740556ec358a3c900f473ee53b23bab7217890142a8418c456c094bd3d9`.
It was built at `82f37eab`; native production code is unchanged at `cc08aee6`.
Its recorded startup contract is bridge 16, execution r2, exclusive native
retry v2 and speaker identity v2. Actual runs used D3D11/OpenVR, P05 and handoff
`off`, unique owned overlay keys and synthetic local inputs.

The first repaired run at `82f37eab` recorded 26 correlated successful
fresh-render/readiness/submission completions for sparse updates, exhaustion,
fast latest-tail protection and normal SELF removal leaving secondary-only
PEER. It then stopped before empty publication because the probe incorrectly
equated the Windows Python launcher PID with the helper's actual PID. Native
remained healthy and shut down gracefully. The failed harness attempt remains
recorded; the correction distinguishes the two exclusively owned identities.

The authorized bounded continuation at `cc08aee6` passed 22 checks without
replaying the sparse/fast workload. Minimal setup supplied nine correlated
completions, including five final opportunities over the secondary-only PEER
survivor. An existing owner receipt confirmed the successful current empty
handoff while the overlay remained visible. Only then was the owned background
helper shut down. An independent observer received its foreign process-quit
event during the empty grace, while the native-owned overlay remained healthy.
This proves event generation, not that this child consumed that exact event;
raw-ingress tests and the actual-owner 17-event, 8/8/1-budget regression provide
the separate consumption/classification evidence.

Visibility sampling bracketed Hide at 512.572–528.109 ms after Python empty
publication. This is not an exact native submission-to-Hide measurement.
The handle remained valid; later samples stayed hidden, the owner reported
confirmed Hide, and no empty-scene retry episode appeared. Explicit shutdown
returned `shutdown_complete` and exit 0; lookup of the exact owned key returned
`UnknownOverlay`. All helper/native/observer/bridge resources closed and the
private manifest was removed. No user process or runtime setting was changed.
The original failed run's missing payload remains unknown; this successful
continuation does not retroactively identify its cause.

### Physical evidence limits and closure scope

Physical HMD consumption, comfort, sparse-update freshness on an affected
headset and long-session behavior are **not run**. An installed runtime,
successful API submission or desktop pixels cannot substitute for those
observations. The finite shared caption window still permits displacement
before a late translation, and fast completed translations can still cause
closely spaced visible changes. No cross-turn grouping or new dwell timer was
introduced.

The maintainer subsequently approved closing #265 on software verification,
excluding physical HMD and long-session observation from this issue's closure
criteria. The [recorded scope decision](https://github.com/kapitalismho/PuriPuly-heart/issues/265#issuecomment-6255868549)
supersedes only that earlier acceptance blocker. Physical results remain
**not run**, not passed; no historical HMD-freshness fix or P05 equivalence is
certified. All R1–R6 software requirements and their preserved contracts remain
in scope. PR CI, review and normal merge are separate publication obligations.

### PR #267 CI integration repairs

The first [PR CI run](https://github.com/kapitalismho/PuriPuly-heart/actions/runs/39171950809)
used head `68b80e7c94cd2751099ad2da8a3dfffd1653fcb4` and synthetic merge
`b28099c1a67f17e582028e4b58250f1cde9da4ff` against `639144cb`.
Its Windows Python 3.14.7 jobs reported four Contracts, twelve UI and six
Runtime failures. Lint, Architecture, Providers, Tooling, Broker and Overlay
passed, including deterministic/graphics native tests and all five required
real-child integration cases. Codex completed its review of that exact head
without findings. The failed jobs were not treated as passes or retried
unchanged.

The four Contracts failures were stale schedules in
`test_self_ui_isolation.py`: merged stable and provider-final-only input was
expected to display or send before any eligible boundary. Those cases now
prove provider/ingress progress and preboundary suppression while UI is held,
then publish actual source-owner `SpeechEnd` and observe the source-only
caption before releasing UI or translation. Transport cases establish a
genuinely blocked send before proving translation progress and exact final
delivery. Independent recognition-unit controls remain immediate and receive
no invented VAD event. Existing deadlines, source-closure isolation,
request-count, identity/order and cleanup assertions remain. The full file,
four neighboring pressure/isolation cases and caption-policy suite passed
28 tests. An initial new assertion used `None` instead of the defined zero
boundary sentinel; it was corrected without changing the visible-behavior
assertions or production code.

The eighteen UI/Runtime failures came from an inherited test fixture.
Baseline commit `6591a4d2` added runtime-toggle admission to the application
boundary, but the shared test helper supplied a minimal `SimpleNamespace`
without `runtime_toggle_intent`. An isolated baseline source import reproduced
that exception before the backend was reached, and a baseline Discord success
case failed at the same enabled-state assertion. The helper now constructs the
real `ApplicationControlOwner` and `ControlEvents`, not no-op admission hooks.
The original EULA, capture-takeover and four gated audio-start ordering cases
are unchanged. Two QQ negative cases now prove their intended activation
backend was reached; four owner-backed QQ/Discord warmup cases additionally
cover held progress, repeated-submit rejection, success and rollback.

Local validation passed 1,299 UI cases with the two existing opt-in process
skips, plus 45 application-boundary/translation-enable cases. The controlled
checks execute production control/enable owners with synthetic dependencies
and Flet controls on a test page; they do not claim a new native GUI or live
authentication observation. These CI repairs change tests and fixtures only,
not the accepted application or native implementation.


## Readable SELF translated units — #279

Authority: [issue #279](https://github.com/kapitalismho/PuriPuly-heart/issues/279).
The implementation baseline and refreshed `origin/dev` were
`a52fcc97a93d34e040545f388e77a55724cf3c91`. The historical presentation reference
was `f1ab9a47d9caaf67bd7666b7d5f5e380da46b185`.

The maintainer requested actual PC-overlay verification in place of agent-run
HMD observation and retained physical headset checking personally. Physical
HMD readability, consumption, historical stale-headset behavior and sustained
headset-session exposure remain **NOT RUN / maintainer-owned**. Desktop pixels,
offscreen graphics and controlled native submissions do not certify them.
This is a task-specific disposition, not reuse of #265's closure amendment.

### Implemented display unit and preserved boundaries

SELF overlay publication selects closed source or an opportunistic complete
sentence within an open cumulative source. Source and matching translation
advance together. A new speculative result or grace restart alone is no longer
a display boundary. The obsolete source-boundary publication machinery was
removed; shared source closure remains independent of translation completion.

A pending source-only successor does not displace a visible translated SELF
unit merely to fill the shared window. Available space still permits source
display. Bounded staging admits the latest fallback when capacity becomes
available, including predecessor expiry. Failed, disabled and unavailable
translation retain source-only behavior. Ready successors can advance without
waiting for an older translation. Retired or displaced identities cannot return
as new captions through a late result.

Blocked overlay admission retains the latest two translated units and a latest
coherent source fallback, plus applicable admitted-caption patches, rather than
playing back obsolete intermediate admissions. Queued first-admission authority
survives its later retained-only patch. These are overlay decisions; progressive
chatbox eligibility, corrections, ordering, pagination, compatible-final
deduplication, UI, manual input and PEER policies remain independent.

This is not a literal restoration of the old LLM-dependent source merge.
Sentence publication is opportunistic: a newer incompatible source can cancel
the earlier sentence's speculative request. Punctuation-free uninterrupted
speech can wait for source closure. A continuing sentence milestone can replace
the full cumulative translation, not a separately translated suffix. Finite
shared slots do not guarantee a reading interval under arbitrary arrival rates,
particularly with active PEER occupying a slot. No fixed dwell or caption
playback queue was added.

### Verification boundaries and corrected timing premise

Actual production source, translation, output, paginator, presenter, bridge and
Flet desktop-child paths consumed synthetic source and controlled translation
responses. The chatbox destination was a recording local sender, not VRChat.
The process used isolated manifests, loopback endpoints and owned synthetic
backdrops. Only guarded owned-window regions were captured; no live audio,
private settings, provider credentials or user conversations were required.
Python 3.14.7, Flet/flet-desktop 1.0.0 and websockets 16.1.1 were recorded.
Historical and baseline imports came from pinned source exports; candidate
source identities were recorded around execution.

An initial short-turn experiment injected `SpeechStart`, then full source and
immediate `SpeechEnd` 40 ms later, with starts 650 ms apart. It bypassed acoustic
hangover. It is evidence for already-separated callback turns, **not** evidence
that a real 650 ms pause splits historically. The maintainer identified this
distinction; the initial explanation was corrected rather than used for
acceptance.

For the ordinary local-VAD low-latency path, 500 ms hangover rounds to sixteen
512-sample chunks at 16 kHz, approximately 512 ms. The existing 400 ms source
grace follows `SpeechEnd`: about 912 ms from modeled speech stop when source is
already available. Late source can restart grace; provider-owned independent
recognition does not use this equation. Speculation runs concurrently, not as
another delay added after those intervals. Historically, a still-running
translation could additionally keep the source merge open.

Corrective probes used real `VadGating`, controlled speech probabilities and
synthetic 512-sample chunks, then the production audio-segment ledger and
`OwnedVadEvent` identity matching the STT updates. These model stop-to-onset
timing; they are not recorded speech or a real classifier-accuracy test.
Three genuine resumed-speech chunks preceded resumed recognition.
A no-GUI producer check first established ordinary source closure about 400 ms
after the emitted end event and correct final idle state.

All six final actual-desktop comparisons then completed with guarded pixels
and clean child exits. At a modeled 650 ms stop-to-onset gap, historical,
baseline and candidate each retained one canonical turn. At a 1.1 s gap with
two-second translation latency, historical retained one joined A+B turn while
baseline and candidate closed two turns. The candidate does not claim to
restore that historical cross-turn translation grouping.

An earlier six-run corrective attempt was invalid for grouping despite process
exit success: plain VAD UUID ownership did not match scoped STT ownership and
left continuation waiting for a watchdog. Those observations are excluded from
grouping acceptance. The repaired ledger-based probes established closure and
identity invariants before another GUI comparison. The final pixel observer
ran in a separate process so screenshot work did not serialize producer events.
Actual scheduling jitter and result cancellation remain distinct from intended
input offsets.

### Readable translated coverage and observed timing

The corrected fast scenario used modeled speech onsets at 0, 0.970 and 1.940 s
and stops at 0.320, 1.290 and 2.260 s. Full stable source was scheduled 100 ms
before each stop. A was “The blue door is open.”, B “Please bring your ticket.”
and C “We leave after sunset.” Their supplied Korean translations were
“파란 문이 열려 있어요.”, “표를 가져오세요.” and “해가 진 뒤에 떠나요.”
The candidate kept one caption identity and advanced A → cumulative AB →
cumulative ABC, rather than exposing translated word-prefix revisions.
Historical and baseline also merged these speech segments, but exposed
intermediate prefixes in the controlled fast-result schedule.

Times below are seconds from scenario start. Capture intervals bracket the
actual screenshot operation that contained the matching text; they are not
exact panel transition times. Readiness means the controlled provider completed
the matching source response. The last column is a sample-supported unchanged
display interval, not an enforced dwell or comfort guarantee.

| Candidate coverage, 650 ms modeled pause | Translation ready | Matching pixel capture | Ready to capture, ms | Observed unchanged interval, s |
| --- | --- | --- | --- | --- |
| A | 0.255371 | 0.296919–0.328223 | 41.548–72.852 | 0.878480 |
| AB | 1.220885 | 1.242300–1.277607 | 21.415–56.722 | 0.898791 |
| ABC | 2.201549 | 2.225811–2.261050 | 24.262–59.501 | 1.747542 |

Actual candidate speech-end events were 0.846647, 1.795769 and 2.794708 s;
source closed at 3.199326 s, versus baseline 3.197153 s. A coherent translated
sentence could therefore be read before the final acoustic/source boundary.
These are translated-content observations; first-source appearance was not
used as the completion metric. Cumulative sentence growth still changes layout
and can wrap the final text.

The slower scenario used modeled A onset/stop 0/0.320 s and B 1.420/1.740 s,
with two-second translation requests. Candidate A source closed at 1.248485 s,
before actual B onset at 1.450188 s; B source closed at 2.650470 s.

| Slow scenario unit | Translation ready, s | Matching pixel capture, s |
| --- | --- | --- |
| Candidate A | 2.230398 | 2.264915–2.302039 |
| Candidate B | 3.663085 | 3.696517–3.715466 |
| Historical joint AB | 3.659080 | 3.683241–3.717328 |

The candidate exposed A about 1.4 s before the historical joined translation,
and retained A when B appeared. It did not recreate the historical joint
translation. That difference preserves current canonical chatbox/source
semantics while offering earlier complete translated content.

Separate callback-boundary desktop cases observed source before/after end,
resumed speech, slow/pending/failed/disabled translation, late results, expiry
and mixed SELF/PEER. With two available slots, successive complete short units
retained their predecessor for approximately 1.3 s in the observed schedule.
With a held active PEER occupying one slot, complete SELF units could replace
each other about 650 ms apart. The first Korean target had 13 characters.
The Director retained this finite shared-capacity tradeoff rather than adding
unsupported dwell or changing PEER selection; it is not a universal readability
claim. The PEER fixture entered at the production presenter boundary.

A blocked-release desktop burst reached the latest G+H scene without captured
intermediate B–F playback. Producer-level admission checks separately establish
coalescing; finite-rate screenshots alone cannot prove that no intermediate
frame existed.

### Baseline chatbox preservation

A separate controlled-release matrix completed twelve cases: baseline and
candidate, stable-append and final-only input, each with normal, blocked or
disabled overlay delivery. Normal delivery used the real presenter and bridge
without a desktop child. Every case made four translation requests and recorded
four successful local-sender payloads. Normalized request, payload and canonical
closure sequences matched baseline/candidate exactly; stable append yielded one
canonical final, and final-only input yielded four. Source idle and completed
chatbox progress were observed before blocked delivery was released. Compatible
finalization did not add a duplicate payload.

Blocked matrix teardown released and drained both versions before stopping.
It is not a baseline immediate-stop pass; that earlier runner encountered the
shutdown deadlock recorded below. The repaired candidate's independent
immediate-stop smoke supplies the distinct cancellation-safety evidence.

### Deterministic, native and process verification

| Executed selection | Result and limits |
| --- | --- |
| Integrated domain, configuration, core and architecture suites at `766ae006` | 4,289 passed; four opt-in real-process skips; 210 existing websockets deprecation warnings |
| Repaired streaming and producer-isolation suites | 94 passed; obsolete publication/wiring/empty diagnostic assertions removed, behavioral coverage retained |
| Focused source, translation, chatbox, caption and retention neighbors | 463 passed; overlapping selection, not an additional suite total |
| Output shutdown and caption-policy regressions after repair | 105 passed |
| Native full serial suite, including Python cross-language scenarios | 304 passed; one existing opt-in renderer memory probe ignored |
| Explicit D3D11 style/replay check | One passed; real offscreen GPU, no OpenVR hardware session |
| README real-process overlay integration selection | Five passed; zero skips |
| Repository Ruff, Black and offline test selection | Passed; Black checked 1,132 files, selection covered 516 offline files |

The native release built successfully and the assembled executable's actual
`--check-startup-contract` returned app 2.9.0, bridge 16, execution v1/r2,
exclusive native retry v2 and immutable-first-readable speaker identity v2.
Executable SHA-256:
`eca11de223a283cb0588d9eb56557a742d366b49940cb10f33774278ee9e6d62`.
Native product sources and protocol were unchanged.

Actual Python presenter/mailbox JSON crossed loopback WebSocket parsing into
the native presentation owner. This tier used a controlled renderer and
submitter, separately from real D3D11 offscreen checks. Sparse 600 ms updates
produced 28 renders/submissions; 30 ms bursts followed by silence produced 11.
Measured owner-connection-through-joined-shutdown intervals were 4,317 ms and
1,494 ms respectively; they are not HMD latency or GPU-utilization measurements.
All 131 completed rows among 431 parsed retry rows carried fresh-render,
producer-ready and successful-submit witnesses. Assertions correlated scene
and episode identity, increasing execution IDs and existing per-episode limits.

Coverage included latest scenes after exhaustion, retained patches, send-time
expiry, SELF removal leaving primary-translated or secondary-only PEER,
coalesced clear, fresh epochs without renewed caption age, empty handoff/Hide
grace, OFF-side retirement and no-progress timeout despite scene churn.
Metadata scheduling alone was not counted as execution.

### Preserved failures and bounded shutdown repair

The first broad integration run reported 13 failures, 4,274 passes and four
skips. Tests expecting arbitrary speculative or grace-boundary display were
replaced with coherent-coverage behavior checks. Provider language provenance,
caption identity, pending-to-ready/failure transitions, progressive chatbox
progress and producer isolation remain asserted. The final broad run above
passed after those repairs.

A blocked-output fixture exposed a real shutdown deadlock: the cancelling owner
held the admission lock while awaiting a caption task whose cancellation cleanup
tried to reacquire that lock. A two-file run timed out at 300.21 seconds; a
single-case diagnostic timed out at 30.04 seconds. These are not test passes.
Terminal-batch cleanup now avoids that reacquisition while preserving receipt
identity; joined replacement-task markers are released after successful join.
Seven regressions cover blocked and release-immediate close, applied-on-cancel
receipts, live-waiter accounting and replacement-generation progress.

A production-owner synthetic smoke then exercised both blocked-to-stop and
release-to-immediate-stop without pre-stop delivery draining. Both exited
successfully, closed providers, cleared reservations and left no live tasks or
post-stop replay. This is cleanup evidence, not a performance guarantee.

Earlier GUI harness attempts stopped on focus/occlusion guards, incompatible
historical callbacks or an invalid mixed-source fixture; incomplete runs were
not promoted to complete observations. Capture safety required the exact owned
overlay and opaque backdrop with pre/post-capture region/z-order checks.
Initial native shell quoting and log-decoding failures were corrected before
interpreting successful checks. Initial Black failures were repaired using the
repository formatter; final global checks passed.

### Independent mixed-scene selection repair

Whole-checkpoint review of `a52fcc97` → `766ae006` found one material mixed-window
defect. A full scene containing an older finalized PEER and a newer translated
SELF unnecessarily staged the next untranslated SELF, even though ordinary
selection would replace only the older PEER. The independent production-owner
probe observed the failed successor hidden until predecessor expiry. The
Director accepted this P2 finding.

Both staging and later promotion now forecast the actual next selection using
the existing shared selector and current effective display preferences.
Forecasts do not mutate stored entries, retirement, live pointers or appearance
order. The translated SELF is protected only when it would actually lose its
slot. Older finalized PEER can yield normally; newer finalized PEER and drawable
active PEER still preserve their existing ordering/protection.

Eight added cases cover both chronology orders, active/secondary-only/hidden
PEER, translation-disabled drawability and promotion after preference or active
protection changes. The affected caption/presenter/freshness suites passed 158
tests. A fresh-process production-owner smoke observed immediate pending-source
admission beside the retained translation at t=1.0, terminal failure without
hiding it, and unchanged truthful fallback at t=7.9. Chatbox progression and
appearance identity remained intact.

The repaired integrated selection passed 705 caption, presenter, freshness,
output, streaming, producer-isolation and architecture tests. Global Ruff and
Black passed again. Earlier source/translation and native evidence is retained
only where the bounded selection repair does not change its exercised path;
these overlapping selections are not summed.

Actual repaired PC pixels confirmed the reversed-order fix: after successor
source arrived at 1.059477 s, the capture at 1.102059–1.138162 s showed the
retained SELF translation beside successor source, without the older finalized
PEER. The same content remained after both deliberately failed translation
attempts. Inverse-order pixels retained the previous SELF until the new matching
translation was ready. Active-PEER pixels retained the previous SELF during
pending/failure and then progressed directly to a later ready successor.
All three selected runs used identical repaired product hashes, ended with
source idle and exited successfully.

Two earlier control captures stopped on an owned-window identity/bounds/
visibility guard: the inverse case saved no pixels; the active case saved only
two initial safe frames. Their original exact causes remain unclassified,
not product-selection failures or successful complete observations. Bounded
continuations first established a populated owned window and guarded matching
pixels before starting the scenario. Neither relaxed capture safety nor
changed foreign windows. These completed both control observations.

The repaired Python-to-native selection also passed all nine selected cases
(296 other cases filtered, zero failures/skips). Its controlled sparse and
burst/silence workloads again completed 28 and 11 fresh renders/submissions.
All 120 completed rows among 292 retry rows carried matching fresh-render,
producer-ready and submit-success witnesses. Native file hashes remained
unchanged, so earlier release/startup and real offscreen GPU evidence remains
applicable. This continuation does not add OpenVR or physical HMD evidence.
