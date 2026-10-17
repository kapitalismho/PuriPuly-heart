# SELF-STREAM-2 isolated runtime evidence

This task-local record is separate from the historical [SELF-GRACE-1 record](RESULTS.md). Product sources, other tests, maintained project guidance, Git history/index, and external user logs are outside this worker's mutation scope. These observations do not constitute whole-issue acceptance.

The checks intentionally require both legitimate progress and preserved source accounting: all six real local-ASR terminals, controlled translation/result snapshots, normal source closure, and actual received OSC payloads are observed separately. A passing admission receipt or an overlay pixel alone could hide missing speech or absent chatbox delivery, so neither substitutes for those observations. Fresh native timing runs are not causal same-ASR comparisons.

## Final `_04` Director-authorized runtime evidence

**All four assigned commands exited 0 on the explicitly released, frozen candidate `27e93797dd9ab8b6a2a7969eac73e409ccaffcda`.** The authoritative current runtime set is [fast native `_04`](final_candidate_native_04.json), [slow/no-overlay native `_04`](final_candidate_no_overlay_04.json), [actual isolated CLI `_04`](final_candidate_cli_04.json), and the [correlated summary](final_candidate_summary_04.json). All prior `_03` facts below remain precursor evidence at their original source pin; they are not relabeled as current executions.

All **556 product Python source files** matched at entry/exit in each run and matched across all three runs. The canonical source-hash-map SHA-256 is `7003f8d9467812cb4b2fcb4343b92701961d692967368dca11f771dfb7ea4582`. This is the runtime probe's product-Python scope; the separately reported native-GUI code/helper manifest has its own broader scope and is not silently equated with this manifest. Executed runner/support hashes, actual interpreter/packages, settings-profile hashes, and original-local JSON hashes remain in each report.

| Current route | Real local CPU ASR | Controlled LLM | Source-set accounting | Actual received chatbox packets |
| --- | --- | --- | --- | --- |
| Fast, presenter attached, 50 ms LLM | 6 invocations / 6 final terminals / **0 stable updates** | 6 invoked / 6 ready / 0 cancelled | 6 episodes in 3 cases → 3 closed turns; 0 omitted / 0 duplicated | 6 previews; 0 fresh identical-final packets |
| Slow, **no overlay sink**, 1 s LLM | 6 invocations / 6 final terminals / **0 stable updates** | 6 invoked / 5 ready / 1 obsolete-source cancellation | Same six-episode denominator → 3 closed turns; 0 omitted / 0 duplicated | 2 previews + 3 final packets |
| Actual headless CLI | Capture/ASR not exercised | Translation disabled/unattached | One invented manual source-only input | 1 source-only chatbox packet; 2 typing-control packets counted separately |

Both native runs used the existing installed Parakeet v3 CPU model, revision `2bda32ec70b097a55adaa07d9a7173915b43cc78`, with two CPU threads. All four model-file hashes matched the current packaged manifest at each relevant execution. Effective profile remained English → Japanese, concurrency 2, integrated context enabled, low-latency enabled, configured grace **400 ms**. The frozen WAV denominator remains six invented offline-SAPI episodes, 12.14525 seconds. Each native trace retains three accepted source-context records. No cloud translation quality, microphone, VAD-detector, stable-ASR or population-performance claim is made.

### Current clocks and progression

All times below are compatible monotonic probe-relative seconds; source-close entry/return brackets the delegated production path rather than inventing one exact internal timestamp:

| Fast turn order | First usable snapshot | First local page transmission | First actual loopback packet | Source-close entry → return |
| --- | ---: | ---: | ---: | --- |
| 0 | 5.680260 | 5.684276 | 5.685203 | 6.146621 → 6.151795 |
| 1 | 10.515180 | 10.516920 | 10.517578 | 13.140164 → 13.145245 |
| 2 | 17.571217 | 17.573727 | 17.574506 | 20.039420 → 20.043473 |

All three fast turns transmitted and received usable speculative output before source closure. Their final promotions retained the prior result update identities and emitted no extra identical final packets. The six actual transmissions are classified as previews, not silently converted into new final-send measurements.

With no overlay sink, two turns also sent/received earlier-source previews before closure. All three current-source operations remained running through closure: existing invocations **2, 4 and 6** became ready at **6.014278, 13.610633 and 20.531578 s** after close-entry times **5.408338, 13.005364 and 19.923672 s**, respectively. The same invocations subsequently produced their results without closure-triggered duplicate calls. The first turn had no usable current-source result until after closure and correctly contributed only a final packet. Preview, final local transmission and actual UDP reception remain distinct. Fresh native ASR arrival times were not held identical across fast/slow runs; this is not a deterministic LLM-only causal comparison.

### Current actual-app operations and cleanup

The isolated CLI instance was `24efeae6-b8ac-48a8-a056-d864d2203749`, with interpreter/module resolving `<checkout>/.venv/Scripts/python.exe` and `<checkout>/src/puripuly_heart/__init__.py`. All 11 supported queries/operations exited 0: current capabilities/effective-state/settings/models/OSC queries, terminal applied translation-off/self-capture-off, admitted invented manual text, and ordered stop. Effective self/peer capture and microphone test stayed off; translation stayed disabled/unattached. Its admitted manual turn `7efc13f7-ba54-4c54-9bb1-02d042d98dbb` retained `output_delivery: unobserved`; a separately observed owned UDP packet establishes the source-only output, not the admission receipt.

All owned CLI hosts exited 0; native application close reported no error; UDP receivers were closed and receiver tasks cancelled/awaited. The CLI stop receipt retains `cleanup_outcome: unobserved`, without a fabricated all-callback/durability guarantee. Ephemeral settings and empty invented encrypted-secret stores were removed for all fifteen completed exploratory/frozen runs; exact original-local trace JSON and synthetic logs are retained. The final isolated control directory retained only its empty ownership lock, not an advisory endpoint. No canonical settings, shared model installation, external user logs or user process was changed.

The process environment isolated settings, default logging, extension/control discovery and scene-log roots through owned `LOCALAPPDATA`/`APPDATA`/`USERPROFILE`/`HOME`/temporary directories. Telemetry was disabled and only invented encrypted-file secret state was used. The unchanged Python guard was configured to reject non-loopback destinations and default OSC ports 9000/9001; actual output used only owned ephemeral loopback UDP. No unsafe-destination attempt was made merely to demonstrate that rejection. These controls are not described as an OS network sandbox. No provider/model download, spending, private-data transfer or physical-session send occurred.

| Current sanitized export | SHA-256 |
| --- | --- |
| `final_candidate_native_04.json` | `4239be8107b3543449e691f1f3236a18ce23f4481bb15b8bb8f87ca522e7897d` |
| `final_candidate_no_overlay_04.json` | `9a8b63bb5b2e76c034318bf4e7a87fbf01b5e68c92364d8a9c424486947d77a4` |
| `final_candidate_cli_04.json` | `868b5cca47a2819429775e76450b36f02d5fe7159aed2d8f4b8dfe60e4f15235` |
| `final_candidate_summary_04.json` | `55e4dee0be76b8168588958278cfe5df7c01f10d87f7429bfaa8fddce6c628ef` |

**Remaining boundaries:** native ASR here is final-only; required deterministic stable-contribution evidence and the broader failure/resource/dual-target matrix belong to the Director's separate current-owner evidence. Actual GUI presentation is separately owned and must not be inferred from these application/presenter receipts. Local packet reception is observed, but physical VRChat chatbox/HMD visibility remains **blocked/unperformed**: no isolated authorized physical session was available. This runtime success does not by itself mark the issue wholly complete.

## Retained `_03` precursor runtime set

**The `_03` executable runtime checks passed on their recorded source content and remain precursor/applicability-limited evidence only.** The retained set is [fast native `_03`](final_candidate_native_03.json), [slow/no-overlay native `_03`](final_candidate_no_overlay_03.json), [actual CLI `_03`](final_candidate_cli_03.json), and the [correlated summary](final_candidate_summary_03.json). After completion, stream-core invalidated freeze for the Director's first-successful-page latency summary failure after earlier send failures; these successful-send routes did not exercise that failure/recovery case. Subsequent core/UI repairs and independent reviews preceded the separately authorized `_04` executions above. Original `_03` observations/hashes remain unchanged and are not post-repair final acceptance.

The Director required consolidation with independent whole/resource reviews; core-only restored-freeze messages did not release another run. The worker held until the Director explicitly authorized frozen candidate `27e93797dd9ab8b6a2a7969eac73e409ccaffcda` for `_04`.

All **556 product Python files** matched at entry and exit in each `_03` execution and matched across all three executions. This includes source/turn/request/projection/diagnostics/output/output-batch/domain/paginator and UI modules. The canonical sorted source-hash-map SHA-256 is:

`d9143fb16ab725ac60d13a3eb8095baa5e69c13bb5d5ab39b6688455e9f8bff4`

The fast run recorded HEAD `f1ab9a47d9caaf67bd7666b7d5f5e380da46b185`; the slower native and CLI runs recorded `5087173cb3f8fe7b1312c59337e5ec7f41e6b81e` after the Director's commit. **The executed product source bytes are identical across that label change.** No original report is relabeled, and this worker performed no Git mutation. The [pin-transition record](runtime_pin_transition.json) independently identifies only the three UI source-file hash changes from `_02` to `_03`; import-only semantics are reported by the UI owner/Director, not inferred from hashes.

| Observed `_03` route | Real CPU ASR | Controlled LLM | Source accounting | Actual chatbox UDP |
| --- | --- | --- | --- | --- |
| Fast, presenter attached | 6 calls / 6 final terminals / 0 stable updates | 6 invoked / 6 ready / 0 cancelled | 6 episodes in 3 cases → 3 closed turns; none omitted/duplicated | 6 preview packets; zero new final packets |
| Slow, **no overlay sink** | 6 calls / 6 final terminals / 0 stable updates | 6 invoked / 5 ready / 1 obsolete-source cancellation | Same six-episode denominator → 3 closed turns; none omitted/duplicated | 2 preview + 3 final packets |
| Actual headless CLI | Not exercised; capture kept off | Disabled/unattached | One invented manual source-only input | 1 source-only packet, separately from 2 typing-control packets |

Both native runs used the current installed `local_parakeet_v3` CPU backend with two threads and verified all four model-file hashes at execution. Source language was English, target Japanese, configured concurrency 2, low-latency enabled, integrated context enabled, and grace remained **400 ms**. Each native run retained exactly three accepted source-context records, one per closed source unit. No translation-quality or cloud-provider result is claimed: LLM strings are controlled scheduling results.

### Separate source, preview, send and receiver clocks

The following are compatible monotonic probe-relative seconds. Source-close entry/return brackets the delegated normal source-close path; it is not fabricated as one exact internal instruction timestamp. Raw traces separately retain source VAD seals/scheduled clocks, handler receipts, actual ASR result readiness/terminal receipts, LLM invocation/readiness, source/presentation revisions and application receipts.

| Route / turn order | First usable snapshot | First local page send | First actual loopback receipt | Source-close entry → return |
| --- | ---: | ---: | ---: | --- |
| Fast / 0 | 5.613145 | 5.615652 | 5.616468 | 6.062176 → 6.067139 |
| Fast / 1 | 10.466924 | 10.469369 | 10.470158 | 13.097898 → 13.103472 |
| Fast / 2 | 17.509420 | 17.511506 | 17.512077 | 20.000372 → 20.005124 |
| Slow, no overlay / 0 | 6.364527 | 6.366057 | 6.368600 | 5.773019 → 5.775450 |
| Slow, no overlay / 1 | 11.496753 | 11.498405 | 11.500527 | 13.129543 → 13.131323 |
| Slow, no overlay / 2 | 18.514083 | 18.515871 | 18.516595 | 20.054674 → 20.056816 |

All three fast turns sent/received usable previews before source closure. Each subsequently completed with the same final-result update identity and no identical extra final packet. The paginator's `self_turn_final` metadata classifies all six transmissions as previews: final promotion retains the earlier preview-transmission relationship rather than manufacturing a new final-send measurement.

In the no-overlay run, the first turn closed before its first current-source translation became ready, while the other two turns had earlier-source previews before closure. Existing current-source invocations **2, 4 and 6** were still running at source-close entry and became ready at **6.363352, 13.736985 and 20.647362 s**, respectively. Their invocation times precede closure and no duplicate invocation is introduced by closure. Final packets are classified separately from previews. This establishes observed running-work handoff and overlay-independent production UDP on this sample; fresh native ASR arrivals are **not held identical** between runs, so it is not the issue's deterministic LLM-only causal comparison.

### Actual application operation and cleanup

The final isolated CLI host instance was `4e8c82cb-3182-442d-9e61-4f963207f9dc`. Its recorded interpreter/module resolve `<checkout>/.venv/Scripts/python.exe` and `<checkout>/src/puripuly_heart/__init__.py`. All 11 supported queries/operations exited 0, including current `capabilities`, effective status/settings/models/OSC queries, terminal applied translation-off and self-capture-off operations, invented manual text admission, and ordered stop. After the operations, self/peer capture were effectively off, microphone test was off, and translation was disabled/unattached. The manual receipt's `output_delivery: unobserved` remains unchanged; actual owned receiver observation separately proves its source-only packet.

The owned host exited 0, UDP receivers were closed, native receiver tasks were cancelled and awaited, and native application close completed without a reported error. The CLI's `cleanup_outcome: unobserved` is not upgraded into a guarantee about every teardown callback. Ephemeral settings and all empty invented encrypted-secret stores were removed for all twelve completed exploratory/frozen runs, including the obsolete root-level residue. Original local trace JSON and synthetic logs remain for hash provenance; no external user logs, canonical profile, shared model installation, or user process was changed.

### Recorded `_03` artifact identities and remaining boundaries

| Sanitized export | SHA-256 |
| --- | --- |
| `final_candidate_native_03.json` | `7f706ac1cbedffab0038511549ee8530b8bb6638d55e588238154b6c107171ef` |
| `final_candidate_no_overlay_03.json` | `8c1785bcd284af244ecb6274e0b771563da87bf7590929d372a557b7e30279af` |
| `final_candidate_cli_03.json` | `e1006db817b45528dbb4748ea60541e827347d73b40d00b31538429ac84b02f6` |
| `final_candidate_summary_03.json` | `5d42b09aadf76bb6711f9784b76461e34ff0f5af402543d648f1b21866df717d` |

The native/local-ASR production route and actual local packet portion of C9/C10 are observed. These final-only native runs do **not** establish deterministic stable-contribution coverage, all resource/failure/dual-target cases, or whole-issue completion. Actual native GUI presentation/interaction is a [separate UI-owner proof](../gui/REPORT.md) using controlled snapshots; it is not inferred from this audio/headless probe. Physical VRChat chatbox and HMD visibility remain **blocked/unperformed** because no isolated authorized physical session is established. No private audio, microphone, cloud call, spending, model download, or default OSC-port send was used.

## Environment and safety

- `uv sync --extra dev` created the **worktree-local** `.venv` with CPython 3.14.7 and installed 106 packages. No parent/shared virtual environment was selected and no models were installed or downloaded.
- Explicit imports of `puripuly_heart` and `compose_application_runtime` resolved `<checkout>/src/puripuly_heart`. The runnable interpreter is `.venv/Scripts/python.exe` from the checkout root. Core and UI owners received this readiness information directly.
- Observed package versions: PuriPuly Heart 2.8.2, Flet/Flet Desktop 1.0.0, NumPy 2.5.1, sherpa-onnx 1.13.8, ONNX Runtime 1.28.0, soxr 1.1.0, sounddevice 0.5.5, python-osc 1.10.2, pytest 9.1.1, pytest-asyncio 1.4.0, Ruff 0.16.1.
- Installed Parakeet v3 metadata reported ready at the existing `<model-root>/parakeet-tdt-0.6b-v3-int8-sherpa`, selected revision `2bda32ec70b097a55adaa07d9a7173915b43cc78`. Each native run independently checked all four file SHA-256 values against the current package manifest before decoding.
- Frozen inputs remain the six invented offline-SAPI WAVs in [inputs.json](inputs.json): 12.14525 seconds, mono 16 kHz PCM16. No microphone, private recording, cloud ASR/LLM, real translation service, paid provider, or model installation was used.
- Each run owns a new state directory below `<checkout>/.venv/self-stream-233/`. `LOCALAPPDATA`, `APPDATA`, `USERPROFILE`, `HOME`, `TEMP`, and `TMP` point there. Settings use an encrypted-file secret store with an invented passphrase, telemetry is disabled, and account state is empty. Native decoding reads the existing installed model through the selected CPU factory root only; provisioning/model-write roots stay isolated.
- Python network audit guards reject non-loopback connections and default OSC ports 9000/9001. This is an opt-in Python safety guard, not an OS network sandbox. Production OSC sends use an owned ephemeral loopback UDP receiver, never a user's VRChat receiver.
- Reports in this directory are **sanitized exports**, not original local bytes. Their `trace_export.original_local_sha256` preserves the exact original JSON identity. Original local trace JSON/log bytes remain under `<isolated-state>`; ephemeral settings and empty invented secret stores are removed after the completed evidence sets. Exported path substitutions do not alter executed source, audio, model, or settings hashes. New report names are required; retained reports are not overwritten.

## Launch prerequisites and side effects

Inspection used `AGENTS.md`, current `docs/architecture.md`, current `docs/cli.md`, the issue body, and the required implementation-workstream prompt.

| Boundary | Observed implementation or actual result | Safe probe control |
| --- | --- | --- |
| Settings identity | Explicit `--config` selects the owned settings file/control identity; absent configuration uses canonical user state | Always pass the isolated settings path |
| Main logging | Explicit GUI configuration chooses its directory; `run-headless --config` still defaults main logging to the normal user-config root | Isolated `LOCALAPPDATA` plus an explicit task-local logging sink for the audio seam |
| Extension registry | Composition reloads the default user-config `http_extensions` directory independently of `--config` | Empty isolated `LOCALAPPDATA` registry; controlled LLM injected before native composition |
| Discovery | Same-user control records/ownership locks live below `LOCALAPPDATA/puripuly-heart/control`; the directory must exist | Create isolated base before launch; never enumerate/attach canonical instances |
| Scene context | Startup can begin a process-lifetime scene owner whose default log root uses `USERPROFILE/AppData/LocalLow/VRChat/VRChat` | Isolate `USERPROFILE`; no user VRChat logs are read |
| Secrets | Default settings use keyring; an isolated settings path alone is insufficient | Explicit encrypted-file backend, relative owned secret file, invented passphrase |
| Telemetry | New default intent enables telemetry | Disable intent and clear anonymous telemetry identity before composition |
| Capture/providers | Startup may construct/attach a lazy ASR provider without capture; attachment is not model preparation | Capture remains off; native probe supplies only owned WAV ingress; CLI does not enable capture |
| OSC | Actual CLI run received chatbox UDP even while `osc.status` reported connection mode off and no control sender | Explicit owned loopback send port is mandatory; off mode or `chatbox_send=false` is not a packet-suppression safety boundary |
| Flet desktop | Installed Python package has no bundled client archive; an existing cached Flet 1.0.0 executable was found | GUI owner can use `FLET_VIEW_PATH` for that existing client with isolated home/state; no download or shared-cache mutation is necessary |

After actual headless shutdown, inspected owned startup side effects were its settings/ready/identity files, an invented encrypted secret file, isolated main log, empty extension directory, control lock, and provisioning lock. No model files were installed. The control advisory endpoint was removed; lock files are retained as ordinary owned state. External user logs were not inspected, deleted, or modified.

## Historical pin warning

The old audio replay imports `tests/fixtures/self_grace_233/experiment.py`. Its `baseline_owner_module()` obtains the SELF owner with `git show 75c4485a...`; the default `SELF_GRACE_233_OWNER_SOURCE` is `baseline`. Its result revision remains the historical constant even in worktree mode. Therefore neither the old bridge nor the old `pin75`-era observations are current-candidate acceptance. The exact old native runner is archived without replacing the historical traces. Elapsed/no-op policies remain unselected; the current issue supersedes historical G2 structural deferral.

## Exploratory current-checkout observations

The [machine-readable exploratory summary](exploratory_runtime_summary_02.json), produced by [the current summarizer](summarize_runtime_evidence.py), correlates each delegated paginator first-page send with the actual loopback packet's address/parameters and reports its separate receiver timestamp. It also records calls still running at source-close entry and their actual later result/cancellation times. The earlier summary is retained rather than overwritten.

The core owner first allowed the seam to run, then explicitly reported further core repairs in progress. The reports below are useful **exploratory observations**, not a final candidate pin. They record HEAD `f1ab9a47d9caaf67bd7666b7d5f5e380da46b185` plus their observed dirty source hashes. The initial runner version fingerprinted only selected relevant files; the prepared final runner now fingerprints **all product Python files at entry and exit**. The original executed exploratory runner versions are retained alongside the current runner.

| Report | Actual result | Limit |
| --- | --- | --- |
| [Native 01](integrated_f1ab9a4_native_01.json) | Exit 0; six native CPU decodes/terminals, six controlled LLM calls/results, two source closures covering all six episodes exactly once, six actual chatbox packets on owned UDP port 57232 | OverlayPresenter application receipts, not GUI pixels; cold native preparation/handling caused a different observed grouping, so this is not a causal baseline comparison |
| [Native no-overlay 02](integrated_f1ab9a4_native_no_overlay_02.json) | Exit 0; six decodes/terminals, six calls, five ready results and one cancelled obsolete call, three closures, five actual chatbox packets | One-second controlled LLM delay and fresh native arrivals differ from Native 01; do not attribute grouping differences solely to LLM timing |
| [Actual CLI 01](integrated_f1ab9a4_cli_01.json) | Exit 0; 11 queries/operations including shutdown; one isolated instance, explicit package/import identity, terminal applied translation-off/capture-off receipts, synthetic manual admission, actual loopback source-only packet, host exit 0 | Manual source-only output is not progressive speech/ASR evidence; `output_delivery: unobserved` in admission is not a send receipt; packet reception was separately observed |

Native 01 observed one logical turn's first usable snapshot at 8.258823 s, first local page transmission at 8.261051 s, and source-closed snapshot at 13.282008 s. A second turn's corresponding observations were 17.713219 s, 17.714875 s, and 20.220765 s. These are monotonic probe-relative times, not physical-display latency. Final promotion retained result update identities without an extra final packet in this short one-target sample. Native final-only ASR produced zero stable updates; it cannot satisfy the required deterministic stable-contribution route by itself.

Actual CLI host instance `b18038c5-36ee-4607-9812-c701a676a7b8` reported the expected checkout module, translation disabled/unattached, capture stopped, and an isolated missing-model catalog. A lazy ASR provider attachment was visible before the explicit off operation; no native decode/capture occurred in this CLI run. The typed manual operation returned `terminal=true`, `status=applied`, `admission=admitted`; the receiver separately observed `/chatbox/input` containing only the invented CLI sentence. Ordered `app stop` returned stopped, and the owned host process exited 0. Its `cleanup_outcome: unobserved` is not upgraded into a guarantee about every teardown callback.

## Retained pre-repair final-pin attempt

The first freeze produced successful [fast native](final_candidate_native.json), [slow/no-overlay native](final_candidate_no_overlay.json), and [actual CLI](final_candidate_cli.json) runs, plus a [correlated summary](final_candidate_summary.json). All 556 product Python files matched at entry/exit and across those runs: manifest SHA-256 `3c6f7b850e5dae48fb4336350a04d0eefd8fa8a53dde182c584491b02d6916f0`.

**These filenames do not make them final acceptance.** The core owner subsequently explicitly invalidated freeze for the asynchronous overlay destination-failure diagnostics callback repair in output/projection. The successful native behavior observations retain bounded relevance, but this exact source pin was superseded. All original attempts, hashes, and summary bytes are retained without overwriting; [attempt dispositions](attempt_dispositions.json) records the distinction. Fresh `_02` executions follow the restored freeze.

The Director requested removal of obsolete settings/secret-store residue. The old `audio/isolated-settings.json` and `audio/unused-isolated-secrets.json`, and completed prior runs' ephemeral settings/empty invented secret stores under owned state, were removed. Frozen profile intent and executed hashes remain in reports/runner code. Original local trace JSON and synthetic-only logs are retained; external user logs were not touched.

## Retained pre-import-only-delta attempt

The restored core freeze produced successful `_02` [fast native](final_candidate_native_02.json), [slow/no-overlay native](final_candidate_no_overlay_02.json), and [actual CLI](final_candidate_cli_02.json) runs. Their [summary](final_candidate_summary_02.json) pins 556 stable product Python files under manifest SHA-256 `c423cb0b9fbd9cb84b046960107458209e3c849d6b28123c05f65febbd5c4073`.

After that set completed, the Director requested only import ordering in `ui/event_dispatch.py`, `ui/event_mapping.py`, and `ui/event_projection.py`. The UI owner reported that only `SelfTurnPresentation`/`STTSessionState` order changed and restored freeze after checks. These prior runtime observations remain behaviorally applicable across the reported nonsemantic import-only delta, but they are not relabeled as having executed the newer source bytes. Fresh `_03` runs supplied the next recorded all-source pin, which later also became precursor evidence; `_04` above is the current Director-authorized execution.

## Physical destination boundary

A safe process-name/PID snapshot initially showed Steam and `OVRServer_x64.exe`, but no VRChat/SteamVR compositor/runtime processes. Present-device metadata matched only Oculus virtual gamepad/audio devices, not a physical HMD. The first PowerShell metadata attempt failed because the shell expanded its pipeline variable; the property-filter retry succeeded. No sessions, command lines, process handles, user logs, microphone data, or external network endpoints were inspected.

A later actual-app scene query reported `process_present=true`. These are separate observations at different times. Neither process existence nor a virtual driver establishes an isolated authorized VRChat social session, a connected HMD, or permission to send into the user's session. Physical VRChat chatbox presentation and HMD visibility remain **blocked/unperformed**, not passed. Actual GUI presentation/interaction proof belongs to the UI owner and is not inferred from the audio or headless reports.

## Runnable integration commands

The current `_04` command spellings below were exercised only after the Director's explicit consolidated release. Run from the checkout; future repetitions must choose fresh report names rather than overwrite these retained attempts:

```powershell
.venv/Scripts/python.exe -B tests/fixtures/self_grace_233/audio/run_production_audio.py --report final_candidate_native_04.json --llm-delay 0.05
.venv/Scripts/python.exe -B tests/fixtures/self_grace_233/audio/run_production_audio.py --report final_candidate_no_overlay_04.json --llm-delay 1.0 --no-overlay
.venv/Scripts/python.exe -B tests/fixtures/self_grace_233/audio/run_isolated_app.py --report final_candidate_cli_04.json
.venv/Scripts/python.exe -B tests/fixtures/self_grace_233/audio/summarize_runtime_evidence.py --reports final_candidate_native_04.json final_candidate_no_overlay_04.json final_candidate_cli_04.json --output final_candidate_summary_04.json
```

These commands preserve frozen WAVs and historical reports. The final report's source-fingerprint stability check rejects a changing product tree. Core/deterministic stable-contribution, bounded-resource, failure/cancellation, secondary-target, UI, architecture, formatter, and repository-wide checks remain with their assigned owners/Director; this worker has not run them or claimed them passed.
