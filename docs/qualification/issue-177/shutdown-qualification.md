# Issue 177 — Outcome C shutdown qualification

## Scope and disposition

This record covers the existing application shutdown coordinator, the Python/Flet application boundary, retained native/download children, and the Flet desktop-viewer containment required by R2, R5, R6, and R8. It does not change product ownership, introduce a supervisor or backend service, or claim native-package cutover.

The source repair is committed at `651f0c3fecc1451f5511c0f3b098a117d2fada9c` (lifecycle source `e2807563`); the final source/evidence commit used for retained runs is `107ed1820005872561c98809c0d4734fe8374cfd`. The final rebuilt package used below contains that product source; its `PuriPulyHeart.exe` SHA-256 is `affc6412869f7a24b669932e5cecca0bd2f899b49309e82b8bfe1c91d55c72f1`. The pre-repair package with executable SHA-256 `40b31d85497755447ed7b81f7bb678d768f488d15615e789b04a231686785cd3` does not contain this repair and is not evidence for it.

## On-demand shutdown diagnostic

`UiApplicationPort.capture_application_shutdown_stall_diagnostic()` reaches the existing `ApplicationShutdownCoordinator`; it does not create another lifecycle manager. Production composition binds `ApplicationRuntimeShutdownAdapter.application_shutdown_runtime_states` as the supplier. Its bounded projection reports:

- Self and Peer capture generation and owned source/VAD/loop state;
- Self and Peer Local ASR provider generation and current native/provider phase;
- GPU worker PID, phase, pending work, and active channels;
- active model-provisioning generation plus the owned Hugging Face/Xet helper PID;
- named asyncio task/await graphs;
- coordinator state, terminal flag, failure count, active shutdown owner/callback; and
- the explicit limit `native_stack_available=false`.

A callback timeout captures this state before callback cancellation and sends it through the existing shutdown-diagnostic logging path. Call graphs are bounded. The projection does not serialize arbitrary owner objects, settings, credentials, transcripts, provider payloads, or native stacks. Snapshot/call-graph failure is represented only by a safe exception class.

A CPython 3.14.7 source composition probe initialized the real application pipeline and observed these live states rather than a caller-supplied map: `SelfCaptureSessionOwner` generation 1, `PeerCaptureSessionOwner` generation 0, `LocalASRProviderRuntimeOwner:self` generation 1 in `provider:dormant`, and `LocalASRProviderRuntimeOwner:peer` generation 0. Repeated concurrent and subsequent `UiApplicationBoundary.stop()` calls converged on the same terminal `completed` coordinator snapshot.

A separate real blocked-shutdown probe used the actual composed boundary and production runtime-state adapter while its composed `LocalASRProvisioningOwner` owned a spawned CPython Xet helper. The retained timeout run named `QualificationBlockedOwner`, included `LocalASRProvisioningOwner:gpu` generation 1 with `model-download`, reported controlled helper `hf-xet-worker:pid=11132:running`, and contained the coordinator/callback/download named-task await graph. The timeout logging record included failure count 1 and `native_stack_available=false`; subsequent ordered shutdown reaped the helper. This used an isolated synthetic helper and no network or model download.

An earlier version of that throwaway blocked-shutdown probe attempted to replace a slotted adapter method and failed before shutdown. Its two isolated helper/launcher PIDs were identified by the unique temporary request path and explicitly killed; that failed harness run is not counted as lifecycle evidence. The corrected probe used the real runtime logger and completed without survivors.

### Native-host close orchestration repair

A later real combined native-product-host run exposed a shutdown task-ownership defect that the retained PyInstaller evidence did not exercise. The accepted native window close was scheduled as an ordinary foundation page task. During `stop_external_producers`, `FletFoundationRuntime.cancel_page_tasks` therefore cancelled the same page task that was awaiting the shielded shutdown coordinator. The page task received `CancelledError` with the coordinator still non-terminal in `stop_external_producers`, then its existing `finally` destroyed the native window. Native host exit `0` and an empty survivor set masked the incomplete critical teardown.

The frozen pre-repair evidence is under `.tmp/issue-177-native-product/evidence/pre-repair-r5-self-cancellation/`; its manifest binds the baseline and artifact hashes. The source repair gives application-shutdown orchestration a dedicated foundation task ownership path that is excluded from ordinary page-job cancellation. The deduplicated window-close request alone uses that path. Ordinary UI jobs remain cancel-on-close, and the existing coordinator remains the only lifecycle manager.

A deterministic concurrency regression uses the actual foundation ownership and application shutdown graph. It failed before the repair with `CancelledError` at `ApplicationShutdownCoordinator.shutdown()`. After the repair it observes the ordinary UI job cancelled, one critical application stop completing, a terminal coordinator, and window destruction last; a repeated close request does not schedule a second teardown. Focused UI, foundation, coordinator, and application-runtime lifecycle suites and the complete automated suite pass under the isolated CPython 3.14 environment.

The separate combined native-host rerun passed after restaging the repaired source with the retained probe-only adaptations. Both normal and repeated-close runs returned coordinator state `completed`, `terminal=true`, zero failures, no outstanding tasks, and all six shutdown phases in order before `page.window.destroy()`. The repeated close was deduplicated while shutdown was active. Both hosts exited `0` with no survivors or manual cleanup. The run rechecked the explicit encrypted-file secret store and did not use the production keyring; the native headless version, telemetry persistence, and failure exit `23` checks also remained passing. The immutable evidence bundle and hashed manifest are under `.tmp/issue-177-native-product/evidence/post-repair-r5-native-validation/`; the pre-repair failure bundle remains unchanged beside it.

### Rebuilt source/PyInstaller validation of the same repair

The shared repair is committed at `459e00306d8512d7f91163be7f3084a3c9fcfb2d`. A new clean archive contains 1,308 tracked-file entries, all byte-identical after extraction; the handoff's initial count of 1,325 was a transcription error, not a different inventory. Archive SHA-256: `9a19eb4808bb15fac21dc1fd6d1cfc42459b004045e07f99255fb11313b717d7`. PyInstaller analysis resolves the main entry, UI app and foundation runtime to that extracted source.

The retained manifest is `C:/c177/r5-459e/validation-manifest.json`, SHA-256 `8273318915b17e22ad9121a8b0ad23a29da9bfc6fbaeda8bde7fc51992ee5a0a`. The new package is `C:/c177/r5-459e/배포 space/PuriPulyHeart`; its executable SHA-256 is `419799ea6d66598b960fd258460c23f81a7a3054dd54ef9ada4ea3a812886d7b`. This supersedes neither the identity nor the workload results of the historical package described elsewhere in this record.

| Execution | Observed result and limit |
| --- | --- |
| Actual source GUI, normal and repeated close | Both `run-gui --debug-ui-preview` runs exited `0` with no host/viewer survivor. An external probe-only launcher added an inert ordinary UI task and lifecycle observations, without changing tracked source. Ordinary cleanup cancelled that task, while one shutdown task remained owned; the coordinator reached `completed`, `terminal=true`, zero failures, all six phases and no outstanding tasks before window destruction. Normal after-launch tasks were not suppressed. |
| Clean packaged GUI, normal and repeated close | Both runs used the same debug-preview command with isolated explicit configuration and unrelated Unicode CWD, rendered the dashboard with the expected missing-model notice, exited `0` and left no host/viewer survivor. The selected HWND was bound to the launched host's exact isolated cached-viewer child. No lifecycle instrumentation was shipped, so these package runs do not directly observe coordinator internals. |
| Clean packaged main-host loss | Deliberate loss of only the owned host returned `15`; its identified viewer disappeared with no survivors or manual cleanup. This proves containment, not graceful teardown. |
| Clean packaged headless/payload checks | `--version` returned `2.7.0`/`0`; explicit encrypted-file telemetry disable and persisted reload returned `0`; malformed explicit configuration returned `23`. The merged license gate passed with 26 payloads and the retained soxr source bundle `159af704190e6c0f604e3d0c11f0b6a92f3ef082caf103cc8a187156dd42d50e`. Inventory: 1,146 files / 585,734,848 bytes; build-identical Python DLL, custom soxr and standard PortAudio present, ASIO absent. |

Source reports/screenshots are under `C:/c177/r5-459e/source smoke 한글/{normal-passed,repeated-passed}`; clean-package reports/screenshots under `package smoke 한글/{normal,repeated,host-loss-passed}`; headless results under `headless inventory 한글`. The complete-suite output is retained at `.tmp/issue-177-r5-pyinstaller/full-suite-output.txt`, byte SHA-256 `c492265b5aac61b18317a0c109340f10945adbd79f59c2b458a939a88d0597f1`; the isolated run returned success. Its output contains warnings, including an unawaited API-key-verification drain warning also present before this repair; no warning-free or unrecorded test-count claim is made. The initial record mistakenly hashed the newline-normalized string rather than the retained Windows file bytes.

Independent review found that the original source-smoke launcher was not hash-bound and its reports omitted `PYTHONPATH` and imported module origins. New normal/repeated source runs repair that gap rather than retrofitting historical hashes. Their supplemental manifest is `C:/c177/r5-459e-source-bound/supplemental-manifest.json`, byte SHA-256 `ef54676481e3a554ab37d54332a288162b31f3c3834703320b7db26780e1823d`; it binds the original archive/manifest, exact probe scripts before and after execution, and entry/UI/foundation/coordinator origins and byte identities under `C:/c177/r5-459e/source/src`.

The new reports are `C:/c177/r5-459e-source-bound/normal/report.json` (`e276e6b0fcc521a68ed150a8765a219d6051eb51ab4acdb79e9bb65b4b0e190d`) and `repeated/report.json` (`efd8c9ea912dd9b0f9f60ff418df97efdc77ee5d983c30d4db4e71f4bf77e32c`). Both actual debug-preview GUIs rendered, cancelled the injected ordinary task, completed all six phases with zero failures/outstanding tasks before destruction, exited `0`, and left no owned host/viewer survivor. Two close messages still produced one teardown. Frozen launcher, runner and safe-configuration generator are under `probes/`; the launcher byte SHA-256 is `dfdc66bf8ff09e33bd6541daa7c11d79feff260dae88413c40ba5b2317567d07`. After-launch tasks were not suppressed.

These new runs used explicit encrypted-file configuration, isolated writable roots and `HF_HOME`, SDK offline mode and disabled bytecode writing; their Flet viewer came from the earlier isolated profile, not the user's cache. No OS-level network interception was used, so absence of metadata-request text is not proof that every possible non-SDK network path was blocked. This is source evidence, not direct clean-package coordinator instrumentation.

The build was not network-free: retained output `.tmp/issue-177-r5-pyinstaller/build-output.txt`, byte SHA-256 `66f3fcac4aa9677e5a9aa39691de37bca15bb83e2fb8bc41e3f50c07cda7662b`, records a public `GET https://huggingface.co/api/agent-harnesses` returning `200` during PyInstaller spec evaluation. No cloud workload was intentionally invoked and credentials were not inspected, but the request's ambient authentication-header status is unknown. The manifest's broad exclusion statement describes intended application-validation scope, not proof of a network-free build or absent authentication. This incident was disclosed to the user and its source-only investigation is separate from the shutdown repair.

No new installer, physical audio, model workload, VR/HMD, published upgrade/rollback or release qualification is claimed by these runs.

### Real native in-flight close

A source production-composition probe used the public `qwen-de.wav` fixture (SHA-256 `80bb10c44085a7ce01a17abaf6a2095ed37e1695fca41cc0ea9733f1f24a749c`) and the pinned public Qwen3 ASR Vulkan model (SHA-256 `c75a961b7134a6c952d89797865cb0d0376876185aee04ef6d12c31c2952e4e1`) under isolated application roots. It selected the discovered physical AMD Radeon RX 7900 XTX device `vulkan-index-0`, activated Self through the actual composed `LocalASRProviderRuntimeOwner`, and submitted audio through the production owned-VAD path rather than invoking the worker directly.

The observed transition was `available` (no worker), `validating`, `loading`, `warming`, then `ready`; worker PID 31520 was bound by executable image to `build/gpu_worker/PuriPulyHeartGpuWorker.exe`. At the close trigger the native runtime reported `pending_count=1`, proving an actual native inference request was in flight. Ordered application close returned in 5.029 seconds. One second after terminal close, GPU phase was `closed`, worker PID was absent, pending count was zero, both channels were `closed`, no Local ASR/GPU qualification task remained, and the owned worker PID no longer existed. Event counts remained unchanged before close, at close, and after the delay, so the retired request produced no late publication or resurrection.

The VAD dispatch coroutine itself had returned after submitting the request; `pending_count=1` in the production GPU runtime, not coroutine liveness, is the in-flight native-work evidence. The probe establishes bounded owner close, worker teardown, terminal state, and stale-result suppression. It does not claim that asyncio cancellation interrupted an executing native thread, nor does it substitute for the separate packaged-artifact run.


## Retained Flet viewer containment

Checkpoint C still requires the Flet desktop viewer. The desktop-overlay renderer already assigned its viewer to a Windows Job Object with kill-on-close. The main GUI launcher now uses the same private-launch interception and process owner. Normal close remains application-driven and ordered before window destruction. Abrupt host loss closes the host's Job Object handle and terminates the assigned viewer; this is containment, not graceful-cleanup evidence.

A retained source CPython 3.14.7 host-loss probe bound the controlled child by PID, executable image, and direct ancestry before terminating only the isolated host. Host PID 12320 owned child PID 32448; Job Object assignment succeeded with no failure reason. The host exited with code 15 after deliberate loss, `child_survived_host_loss=false`, and no manual child cleanup was required. Separate real-process integration exercised Job Object close and ten repeated owner close cycles.

## Locally executed adversarial scenarios

| Scenario | Source result | Meaning |
| --- | --- | --- |
| Active initialization close | Passed | An actual composed-application initialization task was active when two application close requests were issued. Initialization and both closes completed normally; the coordinator reached terminal `completed` with zero failures and no outstanding named tasks. Focused Peer and CPU owner races also rejected late source/delegate adoption. |
| Active download close | Passed | A controlled isolated CPython Hugging Face/Xet helper PID 32308 was observable while active. `LocalSTTDownloadRuntime.close()` cancelled the operation, stopped the helper, cleared `child_states`, and left no `.hf-xet-*` residue; the worker PID did not survive. No network or model download ran. |
| Repeated real-application close | Passed | Actual composed application pipeline exposed live owner generations; concurrent close plus a later close returned one terminal state. The active-initialization run independently repeated the concurrent-close result. |
| Child hang | Passed | A controlled owned fake-GPU-worker process deliberately ignored cooperative shutdown. Bounded escalation completed in 0.066 seconds with return code 1; owned PID 21024 did not survive and no manual cleanup was required. This is escalation evidence, not real native inference evidence. |
| Main-host loss | Passed for the source containment boundary | Controlled PID/image/ancestry-bound isolated host loss removed child PID 32448 without manual recovery. |
| Real native inference close | Passed for source production composition | Public audio traversed the actual composed owned-VAD/provider path. With the real Vulkan worker PID alive and native `pending_count=1`, application close reached terminal state in 5.029 seconds; the pending request, worker PID, and relevant tasks were absent afterward, and no late event appeared during the post-close observation window. |
| Packaged repaired GUI normal/repeated close | Passed on final rebuilt artifact | Three isolated-profile cycles selected the visible Flet viewer by PID and launched-host ancestry, posted `WM_CLOSE` twice, and observed host exit code 0. Each host and cache-resident viewer exited; there were no survivors and no forced/manual cleanup. |
| Packaged repaired GUI main-host loss | Passed on final rebuilt artifact | The selected visible viewer was the exact child of the launched host. Killing only host PID 7680 yielded host exit code 15 and caused cache-resident viewer PID 9924 to disappear through Job Object containment; there were no survivors and no forced/manual cleanup. |
| Physical microphone/loopback, VRChat, SteamVR/HMD, cloud calls | Not run | Required controlled hardware/session/credentials were not supplied; no claim is made. |

Cancellation remains distinct from termination of an executing native thread. The real native in-flight close above verifies the owner lifetime, stale-generation fence, and owned-worker outcome without claiming cancellation preempted native code.

## Commands and results

All source checks used ordinary-GIL CPython 3.14.7 from the isolated Outcome C environment with the worktree `src` first on `PYTHONPATH`.

- Focused coordinator, logging, UI boundary, main launcher, viewer owner, and Xet adapter suite: passed.
- Real-process viewer Job Object integration (`INTEGRATION=1`): passed, including repeated cycles.
- Actual composed active-initialization plus concurrent repeated-close probe: passed (`active_at_close=true`, all three operations returned normally, terminal with zero failures and no outstanding named tasks).
- Actual isolated active Xet-helper close probe: passed (PID present while active, `CancelledError` at the download boundary, no survivor/state/residue).
- Controlled hung fake-GPU-child process probe: passed (cooperative refusal followed by bounded termination in 0.066 seconds; PID 21024 did not survive).
- Actual composed blocked-shutdown diagnostic probe with a controlled synthetic Xet helper: passed with live owner generation, `model-download`, helper PID 11132, task graph, logged failure count 1, and native-stack limit.
- Source abrupt-host-loss Job Object probe: passed for controlled CPython host PID 12320 and child PID 32448; `child_survived_host_loss=false`.
- Actual production-composition native in-flight close probe: passed with physical RX 7900 XTX selection, public pinned model/audio, real worker PID/image, native `pending_count=1` at close, 5.029-second ordered close, terminal closed channels, no worker/task survivor, and no delayed event publication.
- Final repaired packaged GUI probe: passed against executable SHA-256 `affc6412869f7a24b669932e5cecca0bd2f899b49309e82b8bfe1c91d55c72f1`. Three fresh isolated Unicode-CWD/profile/config/cache cycles each posted repeated `WM_CLOSE`, returned exit code 0, and left neither host nor Flet viewer. A fourth isolated cycle killed only the host; the PID/image/ancestry-bound Flet viewer was removed with no survivor or manual cleanup.

Packaged evidence recorded the exact rebuilt executable identity, isolated profile/cache/config roots, PID/image/ancestry-bound HWND selection without title matching, close mechanism, exit codes, and the complete observed owned-process inventory. The superseding retained normal runs used host/viewer PID pairs 20368/31448, 23884/9252, and 28152/28984. The abrupt host-loss run used host/viewer PID pair 7680/9924 with ancestry `[9924, 7680]`. Every Flet viewer image was under that run's isolated profile cache (`.flet/client/flet-desktop-full-1.0.0-758f21506fbb/flet/flet.exe`). All four survivor sets were empty and `manual_cleanup=false`.

## Retained evidence artifacts

The independently assessable evidence root is `.tmp/issue-177-shutdown-final/`. Its manifest is `.tmp/issue-177-shutdown-final/manifest.json` (SHA-256 `c9666fac140bd6f9f8591475bf4bdff256b503fd91867980d879d962a5ffce7b`). The retained runner is `.tmp/issue-177-shutdown-final/run_evidence.py` (SHA-256 `999b8b0ec8e958839332b49fc92f76f5dd014284fb46f8b3ab7f431e1d4daef6`). Every wrapper report binds the command, CPython version, source commits `651f0c3f`/`107ed182`, explicit `USERPROFILE`, `APPDATA`, `LOCALAPPDATA`, `TEMP`, and `TMP`, probe hash, UTC times, exit code, parsed observation, and paths plus hashes for raw `stdout.txt` and `stderr.txt`.

All retained reruns used the isolated runtime root `C:\c177\shutdown-evidence-runtime\final-107ed`; no production config, log, or window was read. Each packaged child received an explicit unique `--config` below `C:\c177\shutdown-evidence-runtime\final-107ed\packaged-r5\temp\puri-c177-shutdown-vmmft805`, and ownership/survival classification was limited to the launched host's PID ancestry. The final executable path and SHA-256 are bound in the wrapper and each packaged-cycle report.

| Retained report | SHA-256 | Exact retained observation |
| --- | --- | --- |
| `.tmp/issue-177-shutdown-final/source-active-initialization/report.json` | `3c87892a4652918c8753e0587d97dfd659c90ef3503b11ed9cf9160c5d679e1d` | Exit 0; `active_at_close=true`; initialization and two closes returned normally; terminal `completed`; zero failures/tasks. |
| `.tmp/issue-177-shutdown-final/source-active-download/report.json` | `d0426c701b0ee307b8543b7635e653148080a02569af33f4574ac7895e6abc59` | Exit 0; controlled Xet helper PID 32308 active; cancellation result; no state, residue, or survivor. |
| `.tmp/issue-177-shutdown-final/source-child-hang-controlled/report.json` | `774c5a5aefa8f698d90e2566eee268349fb39738461c7240e421e3393943e487` | Exit 0; controlled fake GPU child PID 21024; 0.066-second escalation; return code 1; no survivor. |
| `.tmp/issue-177-shutdown-final/source-blocked-timeout-controlled/report.json` | `99253d6c7ce299906f35b3d4e3ff22661ce4b30fbd3888fff09046b7e2e5561d` | Exit 0; controlled Xet helper PID 11132, live `model-download`, timeout/task graph, persisted failure count 1, no survivor. |
| `.tmp/issue-177-shutdown-final/source-job-loss/report.json` | `1334b29f2bbb8091bb7fd47ff15f3ad20524a5cd2053f7360cfc8c040d2efd77` | Exit 0; controlled host/child PIDs 12320/32448 with Python image and direct ancestry; Job assignment true; child did not survive. |
| `.tmp/issue-177-shutdown-final/packaged-r5/report.json` | `1e21dbd15dc11e75979cdbc4bfec063dd6fde53edb257282c84ceda9a633dd59` | Exit 0 wrapper containing all three repeated normal closes and actual packaged host loss; executable SHA-256 `affc6412…`; all survivor sets empty; no manual cleanup. |
| `.tmp/issue-177-shutdown-final/normal-repeated-1/report.json` | `6cdec5b0abc536bafc3a89a474488ea67a05bb400888f8b56f875b255d236fb4` | Explicit config; host/viewer PIDs 20368/31448; host exit 0; no survivors/manual cleanup. |
| `.tmp/issue-177-shutdown-final/normal-repeated-2/report.json` | `2ef6dde925bcd2d311418ca6e9ef864ed6b0a9ad7087fa9f91cfafb9dcbebcbe` | Explicit config; host/viewer PIDs 23884/9252; host exit 0; no survivors/manual cleanup. |
| `.tmp/issue-177-shutdown-final/normal-repeated-3/report.json` | `fdd128ee0b14c97dd7e64fcc98c703eb2fc5f91f77b47ac34c35af8ba3e615f0` | Explicit config; host/viewer PIDs 28152/28984; host exit 0; no survivors/manual cleanup. |
| `.tmp/issue-177-shutdown-final/main-host-loss/report.json` | `dc95b3c4b9897a1445d88732812a0889a87315d84e83caa4dd68e3b2b6909f18` | Explicit config; killed only host PID 7680; viewer PID 9924, ancestry `[9924, 7680]`; host exit 15; no survivors/manual cleanup. |

The real native GPU in-flight report remains separately retained at `.tmp/issue-177-native-workloads/native-inflight-shutdown.json` (SHA-256 `036bf5942251b9f2c421c3d23bf90a9e7c2e47c4ad9c9dfe10d2ab56fbdf0918`) and was not rerun. It is the real-native evidence; the retained Xet-helper and fake-hang runs above are controlled process-lifecycle evidence only.
