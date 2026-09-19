# Issue 177 — Outcome C shutdown qualification

## Scope and disposition

This record covers the existing application shutdown coordinator, the Python/Flet application boundary, retained native/download children, and the Flet desktop-viewer containment required by R2, R5, R6, and R8. It does not change product ownership, introduce a supervisor or backend service, or claim native-package cutover.

The source checkpoint is repaired and locally executable. Final packaged-C results remain separate until a package containing these source changes is assembled. The pre-repair package with `PuriPulyHeart.exe` SHA-256 `40b31d85497755447ed7b81f7bb678d768f488d15615e789b04a231686785cd3` does not contain this repair and is not evidence for it.

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

## Retained Flet viewer containment

Checkpoint C still requires the Flet desktop viewer. The desktop-overlay renderer already assigned its viewer to a Windows Job Object with kill-on-close. The main GUI launcher now uses the same private-launch interception and process owner. Normal close remains application-driven and ordered before window destruction. Abrupt host loss closes the host's Job Object handle and terminates the assigned viewer; this is containment, not graceful-cleanup evidence.

A source CPython 3.14.7 host-loss probe bound the child by PID, executable image, and direct ancestry before terminating only the isolated host. Host PID 27080 owned child PID 5928; Job Object assignment succeeded with no failure reason. The host exited after deliberate loss, the child disappeared within the bounded observation window, and no manual child cleanup was required. Separate real-process integration exercised Job Object close and ten repeated owner close cycles.

## Locally executed adversarial scenarios

| Scenario | Source result | Meaning |
| --- | --- | --- |
| Active initialization close | Passed | Peer capture close raced a pending admission/start, then repeated close remained stopped with no source or loop task. CPU local-provider close during validation and delegate open rejected late creation/adoption. |
| Active download close | Passed | A real isolated CPython Hugging Face/Xet helper was observable in the adapter snapshot while active. Owner close stopped the helper, cleared the child snapshot, emitted no late status, promoted no model, and left no `.hf-xet-*` files. |
| Repeated real-application close | Passed | Actual composed application pipeline exposed live owner generations; concurrent close plus a later close returned one terminal state. |
| Child hang | Passed | The real fake-GPU-worker process suite exercised cooperative-shutdown refusal and bounded escalation; focused suite completed without a survivor. |
| Main-host loss | Passed for the source containment boundary | PID/image/ancestry-bound isolated host loss removed the assigned child without manual recovery. |
| Packaged repaired GUI normal/repeated close | Pending final rebuilt C artifact | The existing package predates the repair. |
| Packaged repaired GUI main-host loss | Pending final rebuilt C artifact | Must account for the viewer even when its executable is under the Flet cache rather than the package root. |
| Physical microphone/loopback, VRChat, SteamVR/HMD, cloud calls | Not run | Required controlled hardware/session/credentials were not supplied; no claim is made. |

Cancellation remains distinct from termination of an executing native thread. Existing native in-flight lifetime and stale-generation fences are unchanged.

## Commands and results

All source checks used ordinary-GIL CPython 3.14.7 from the isolated Outcome C environment with the worktree `src` first on `PYTHONPATH`.

- Focused coordinator, logging, UI boundary, main launcher, viewer owner, and Xet adapter suite: passed.
- Real-process viewer Job Object integration (`INTEGRATION=1`): passed, including repeated cycles.
- Active Peer/CPU initialization and GPU worker hang/termination selection: passed.
- Actual composed-application diagnostic/close probe: passed.
- Source abrupt-host-loss Job Object probe: passed; `child_survived_host_loss=false`.

Final packaged evidence must record exact rebuilt executable identity, the isolated profile/cache/config roots, PID/image/ancestry-bound HWND selection, close mechanism, exit codes, owned process inventory (including Flet cache paths), and whether any forced/manual cleanup was required. Forced or manual cleanup must be recorded as failure, not graceful success.
