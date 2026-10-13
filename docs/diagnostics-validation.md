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
