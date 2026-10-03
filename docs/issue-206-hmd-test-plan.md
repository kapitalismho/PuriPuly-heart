# Issue 206: SteamVR worn-HMD test plan

## Purpose and authority

The reported delay/staleness occurred in **SteamVR**. Desktop captures, Presenter acceptance, successful OpenVR API calls, and native health are not proof that the wearer saw current pixels.

The maintainer requested a test plan and complete preparation up to the test start, then a joint session after returning and putting on the HMD. Preparation must not start a live overlay, SteamVR, VRChat, microphone/loopback capture, or paid provider requests. The initial [short physical session](#short-physical-session-2026-10-03-utc) and subsequent [resumed batch](#resumed-batch-2026-10-03-utc) retain the actual observations and failures below; full physical acceptance remains incomplete. See [software evidence](issue-206-verification.md).

Compare product baseline `8666bb57935b7c6da0c3c8aeaec9d116762d9f3d` with candidate `c501b83350d4c39129cf61dd3e582b6ebbea04a5`. Use separate pinned Python/source exports, not two labels pointing to the candidate Python code. The native production implementation is unchanged between these revisions, so use one shared, hash-verified native assembly and record its actual build provenance. Full native source trees are not identical: the candidate adds tests.

For the resumed session, the maintainer explicitly requested batch execution
instead of slow per-case questioning, then confirmed **“착용 중, 일괄 시작”**
for the remaining short checks and isolated application ON/manual/OFF complement.
This session-specific agreement supersedes the per-case readiness/observation
pauses below, not the stop, ownership, identity or sustained-admission rules.
Run one owned overlay at a time, stop the batch on request/error/incomplete
cleanup, retain ordered per-run receipts, and obtain the wearer's attributable
batch observation afterward. Uncertain or unwatched cases remain unobserved.
Reuse the earlier two stable/head-locked observations; run the remaining 42
short arm/anchor cases. Sustained runs still require observed successful short
prerequisites. No simultaneous baseline/candidate overlays or automatic retry
after failure is authorized.

## Operator commands and resume point

Run from the development checkout in PowerShell. The prepared-session location
for resumed work is `C:/pph206-hmd-kit/issue206-resume-fixed`; use its manifest and reports,
after repair verification and renewed wearer readiness. The prior
`issue206-ready-fixed` stage retains the original live receipts, including failure.

Safe before the wearer returns:

```powershell
$stage = 'C:/pph206-hmd-kit/issue206-resume-fixed'
uv run --frozen python -B scripts/bench_ovr_hmd_measurement.py inspect --stage $stage
uv run --frozen python -B scripts/bench_ovr_hmd_measurement.py dry-run --stage $stage --arm candidate --scenario stable --anchor head_locked
```

`inspect` (also named `preflight`) does not initialize OpenVR or execute the
native binary. A dry run exercises the real source-specific owners and bridge,
not the HMD. To create a separate session rather than overwrite the retained
comparison, use `prepare --stage <new-unused-stage>`; preparation validates the
native executable with `--check-startup-contract`, which returns before VR
initialization. The default build input is
`C:/pph206-native/release/PuriPulyHeartOverlay.exe`.

**Stop here until the wearer returns.** Ask for SteamVR readiness, headset and
connection details, comfort, and confirmation that the ordinary overlay is off.
Then run just the short candidate source case:

```powershell
uv run --frozen python -B scripts/bench_ovr_hmd_measurement.py live --stage $stage --arm candidate --scenario stable --anchor head_locked --confirm-hmd-ready
```

Add `--device`, `--firmware` and `--connection` with the wearer's actual supplied
values. Unavailable values stay unknown. The confirmation is renewed on each
live command; the command does not start SteamVR/VRChat, enable capture, or call
a paid provider. It prints the run ID and report location. Keep a second
operator terminal available:

```powershell
uv run --frozen python -B scripts/bench_ovr_hmd_measurement.py stop --stage $stage --run-id <printed-run-id>
```

After completion and the wearer's report, record the actual observation against
that run, not an offline receipt:

```powershell
uv run --frozen python -B scripts/bench_ovr_hmd_measurement.py observe --run-report <printed-report-path> --result <observed-result> --note <wearer-description> --uncertainty <observation-uncertainty>
```

Result choices are `no_issue`, `stale`, `missing`, `wrong_text`, `placement`,
`flicker`, and `discomfort`. Angle-bracket values above are operator inputs from
the actual run, not prerecorded successful observations. Do not enter
`no_issue` on behalf of an absent or uncertain observer.

Select `--arm baseline` or `--arm candidate`, and `--anchor head_locked` or
`--anchor spatial_locked`. Available short scenarios are `independent`, `stable`,
`final_only`, `stable_burst`, `resume`, `mixed_active`, `finalized_eviction`,
`sticky`, `expiry`, `clear_off`, and `restart_reconnect`. Each command runs exactly
one case. The three source scenarios include the capacity-one UI barrier:
one second live, 0.2 seconds offline, followed by explicit release. Baseline
independent ingress is expected to block; baseline stable and final-only active
paths can already progress. The release is a comparison probe, not a new
production dwell or UI policy.

`sustained` is separate, defaults to ten seconds and accepts `--duration 1..60`
at a 0.5-second update cadence (at most 120 updates). Live sustained admission
requires successful software receipts and correlated `no_issue` observations
for all eleven short cases on the selected arm and anchor. It does not run
those prerequisites automatically. `--timeout 1..120` bounds a run (default
90 seconds); choosing a timeout shorter than the scenario requires records a
failure rather than truncating it into a pass.

The stage retains its own `control/bench_ovr_hmd_measurement.py` and interpreter
identity. If the checkout harness later changes, use the staged entrypoint with
the recorded interpreter; do not weaken the harness/source identity checks to
reuse an incompatible command.

## Conditions before the wearer starts

1. The wearer starts the intended SteamVR/VRChat environment and confirms the same HMD/connection path in which the delay was observed. Do not infer the model, firmware, connection path or historical affected setup from an OpenVR presence flag.
2. Record Windows/GPU driver, SteamVR/VRChat versions, HMD model/firmware, wired/wireless connection, refresh rate if known, and whether this is the previously affected setup. Unknown values remain unknown, not fabricated defaults. Avoid serial numbers, account identifiers, private conversation and credentials.
3. Stop the ordinary PuriPuly overlay through its own UI/CLI before using the isolated harness. The harness refuses a pre-existing native overlay; it must not kill or replace it. Do not run baseline and candidate simultaneously.
4. Keep the same SteamVR environment, placement, text scale, native binary, production P05/off profile and scenario settings within each pair. Record any intervening driver/runtime/configuration changes and repeat the affected pair rather than combining incomparable results.
5. The wearer confirms readiness for **one scenario**, with the operator present to request stop. No automatic sequence advances from a short case into a sustained run.

## Stop conditions and bounded exposure

Stop immediately on discomfort, distracting flicker, unexpected placement, persistent old text, a failed runtime/cleanup receipt, or the wearer's request. The wearer can remove the HMD or close the overlay/SteamVR independently; the operator also has a run-correlated stop command and Ctrl+C. Software cleanup is not a guarantee that old pixels have disappeared: ask the wearer separately whether the overlay is gone.

Every run has a finite deadline. No flashing pattern, full-screen strobe, forced head movement, forced original-only dwell, indefinite stress loop, or automatic restart-after-failure is part of this plan. Begin with short, readable synthetic captions. Only after the short cases have acceptable software receipts and the wearer is comfortable may the operator explicitly select a bounded sustained burst, at most 60 seconds. This is an engineering presentation check, not a medical safety certification or 4–6 h stability test.

A failed/aborted software run remains failed/aborted even if the wearer reports no visible problem. An offline run cannot receive a physical pass. Lack of a report is `not_observed`, not `no_issue`.

## Session sequence

Run one case, wait for owned cleanup, ask for the wearer's observation, and record it before choosing the next case. Use baseline then candidate for the first pair; repeat a suspicious difference in reverse order to expose order/warm-up bias. Do not average away a stale frame or claim a universal fix from a short run.

| Stage | Input and boundary | Wearer observation | Software check |
| --- | --- | --- | --- |
| 1. Short placement/stop check | Candidate, head-locked, one controlled source/translation sequence; then clear/OFF | Readable placement; source and later translation; overlay disappears on stop | Correct staged identity/profile, native readiness, owned child exit/cleanup |
| 2. Stable source pair | Baseline/candidate normalized stable contributions, deliberately delayed synthetic translation | When source first becomes readable; retained older secondary while pending; same-caption update without reanchor | Real SELF owner, stream freshness intent, same logical ID, no extra provider attempt |
| 3. Independent/final-only pair | Independent Gemini-shaped result and scoped terminal-only result, each with gated synthetic translation | Source-first once trustworthy text exists; no claim of preterminal source for final-only | No artificial Gemini VAD endpoint; actual eligible provider invocation; record source-ready/application separately |
| 4. UI-pressure pair | Capacity-one UI consumer barrier while event loop runs; release barrier after the comparison interval | Does source appear before dashboard consumption resumes? | Baseline may reproduce blocking; candidate must progress. Record that expected baseline behavior rather than labeling it candidate success |
| 5. Stable bursts/resume | Finite stable contributions and real scoped endpoint/terminal/resume flow | Flicker, repeated reanchors, growing update age, sticky-secondary continuity | Production grace/speculation unchanged; bounded retry accounting and request counts |
| 6. Mixed SELF/PEER | Active SELF, Peer-1, paced Peer-2, late SELF translation; finalized eviction control | Active SELF remains protected; PEER styling/selection; no old caption resurrection | Two-slot window, one-second PEER gate, identity/slot/appearance and expected stale late result |
| 7. Expiry/clear/OFF | Idle expiry, late result, semantic clear, owned runtime shutdown | Old pixels disappear; late text does not return | Python expiry authority, original deadlines, clear/OFF state and shutdown receipt |
| 8. Spatial-locked repeat | Repeat selected stable/mixed cases with spatial lock | Source/final/translation remain at the same anchor; natural head movement only if comfortable | Same caption identity and native spatial bookkeeping |
| 9. Reconnect/recovery | Bounded, explicitly selected owned-runtime restart/reconnect; separate application control check below | No stale replay or persistent old pair after return | Original deadlines, fresh generation and owned cleanup. A harness restart is not proof of application auto-recovery |
| 10. Sustained pair | Optional separate baseline/candidate bounded burst, only after prior checks | Any progressive lag/flicker or old pair outside documented sticky policy | Existing bounded native diagnostics/resource counters; duration and evidence capacity recorded |

Synthetic external provider responses make the comparison reproducible and avoid paid requests or private speech. They exercise production translation/output/overlay owners but are not live-provider timing measurements. A later real speech check needs a separate explicit decision to enable capture/provider use; it is not silently enabled by this kit.

## Application control complement

The synthetic owner harness must not be described as a complete installed-application test. Separately use the documented [CLI](cli.md) against an explicitly isolated settings identity to inspect `app status`, `settings current` and `overlay status`, enable the SteamVR target only after wearer confirmation, submit a public synthetic source with translation/capture off, then run `overlay set off` and inspect effective state. Keep user settings and the installed application untouched.

Application automatic recovery acceptance requires observing the actual application recovery owner and effective generation/recovery state. An owned harness process restart proves only its declared process/bridge seam. Do not force-kill SteamVR, VRChat or unrelated processes to manufacture recovery. If the intended recovery condition is not available during the session, record that case as not run instead of broadening the fault injection.

This complement is staged separately at `C:/pph206-hmd-kit/app-control`.
Its `command-sheet.md` contains the exact safe and later-live commands;
`session.ps1` selects the recorded interpreter, pinned candidate source,
isolated settings and empty isolated secret store in a dedicated PowerShell
terminal. It starts nothing when loaded. Do not use that redirected terminal
for ordinary applications. Never co-run this host's overlay with the synthetic
harness.

Safe host setup and inspection:

For a later live check, capture the original Windows shell OpenVR registry path
in a fresh dedicated terminal **before** changing the profile environment, and
bind it before starting the host. The registry contains runtime discovery
metadata, not application settings or credentials; do not copy or edit it.
All application/home/secret-store isolation remains enabled. If a host was
already started without this binding, stop it normally and start a new isolated
host; changing only a later CLI client's environment cannot repair its parent.

```powershell
$originalShellLocalAppData = [Environment]::GetFolderPath([Environment+SpecialFolder]::LocalApplicationData)
$openvrRegistryPath = Join-Path $originalShellLocalAppData 'openvr/openvrpaths.vrpath'
. C:/pph206-hmd-kit/app-control/session.ps1
$env:VR_PATHREG_OVERRIDE = $openvrRegistryPath
pph app start --background
pph capture set self off
pph capture set peer off
pph translation set off
pph overlay set off
$current = pph settings current | ConvertFrom-Json
pph settings apply --file "$appStage/config/safe-settings.json" --expected-revision $current.revision
pph app status
pph settings current
pph capture status
pph overlay status
pph osc status
```

Check the JSON effective OFF state before proceeding. Live admission also
requires the captured OpenVR registry file to exist, the wearer to be ready,
and no other native overlay. Execute each live command in order, inspecting
its effective receipt before continuing: `pph overlay set on`, `pph overlay status`, then
`pph text submit --file "$appStage/config/manual-source.txt"`. Stop with
`pph overlay set off`, inspect `pph overlay status`, and finish with
`pph app stop`, including after a failed ON attempt. Ask the wearer separately
about disappearance; an OFF receipt alone does not prove blank HMD pixels.

Offline preparation exercised the actual production CLI background host,
settings mutation, public manual submission with outputs OFF, effective-state
queries and ordered stop. SELF/PEER desired/effective capture and source/loop
ownership were false/absent; translation, microphone test, telemetry, consent
and OSC were off. Overlay was configured SteamVR but desired false, lifecycle
off, runtime inactive, with no effective target/process/recovery. The host is
now stopped; the following status returned `instance_not_found`. Initial
uncredentialed provider initialization reported `provider_failed` without a
capture source/loop; this is not a provider-readiness pass.

`preparation.json` SHA256:
`738e3d89aa928675ba0d4eca07063f4599a6b157f9dc70f8205c465ac9089dfc`.
Full effective JSON is in `effective-off-state.json`, with raw argv/stdout/stderr
and exit codes under `records/`. All 1,378 source-export files were hash-verified.
Ordinary no-argument native resolution/prepare selects the verified assembly at
`source/candidate/build/overlay/PuriPulyHeartOverlay.exe`; no executable override
or production patch is used. Its ordinary fresh staging-copy timestamp is
recorded separately from the existing build provenance, not called a new build.
The source/DLL/font bytes match the comparison kit. No native `--config` or
SteamVR overlay ON command was executed during preparation.

## Evidence and interpretation

Keep these stages separate:

- trustworthy stable/final source availability;
- source Presenter application and scene revision;
- actual translation invocation/readiness and translated application;
- bridge/native render, readiness and API submission diagnostics;
- wearer's physical observation and observation method.

Record the scenario, anchor, arm, source revision, harness/native/DLL/resource identity, effective protocol/profile, exposure, shutdown result and per-scenario observation. Use qualitative results such as current, stale, missing, wrong text, flicker or placement, with uncertainty. Visual timing estimates are not instrumented latency. Cross-process/GPU/HMD clocks need valid correlation before arithmetic; do not add overlapping stages as independent gains. SELF E2E remains the first successful chatbox-send metric, not source-overlay latency.

The current secondary line may intentionally retain the translation of an older stable prefix while source grows. Report a pair persisting beyond the expected sticky interval separately from that preserved policy. An expired/evicted caption legitimately rejects late translation; do not extend lifetime or resurrect it to make a test look successful.

The final conclusion must distinguish: prepared/offline verified, software live pass/failure, physical observation, previously affected environment comparison, and untested long-session/device scope. No HMD latency-fix claim is permitted solely from the preparation receipts.

## Preparation evidence

The old protocol-8 OV01/off-versus-cached-handoff harness was replaced in place.
No production runtime, backend, retry profile, settings policy or provider code
was changed for this kit. The developer scenario helper composes the existing
translation/output/overlay owners; it is not a second production scheduler or an
installed-application execution claim.

Ready stage: `C:/pph206-hmd-kit/issue206-ready-fixed`.
Preparation receipt SHA256:
`bc9a21ca1f78334525154d3c0c43fa0bf154e5395671a5061d41618d46d5fb63`.
Harness aggregate SHA256:
`a43d0d9f83993732bdb766e0ecc7fed1e404497ed1bcc5f46aac24edafca72b0`.
The stage records and checks complete source trees, archive/lock hashes, harness,
native/resource hashes, interpreter and installed-distribution identity.

The shared native executable SHA256 is
`5a4c14864dd2c8662bcfdb77c1076812f6a58fa45ad998c56d66e2b3880184ee`.
Source and staged `--check-startup-contract` executions exited zero and reported
app 2.7.0, protocol 15, r2/version 1, exclusive native retry/version 1 and immutable
speaker identity/version 2. They returned before OpenVR initialization.
Production profile remains P05/off. No new native build or physical run occurred.

| Check | Observed result |
| --- | --- |
| Focused harness/owner/bridge/runtime suite | Initially 115 passed; after preparation-abort repair, 127 passed. Two existing opt-in real-subprocess tests skipped in those invocations; separately enabled and passed below |
| Both arms × both anchors × all 12 actual offline scenarios | 48 passed, zero failed; real controller/worker processes and owner engine, not a substituted short sequence |
| Both maximum offline burst cases | 60-second requested scheduling window per arm, 119 meaningful updates/provider calls per arm, cleanup complete |
| Actual run-correlated stop after source/provider start | Controller exited 2, `operator_stop`, cleanup complete; not a software pass |
| Actual timeout after source/provider start | Controller exited 2, `run_timeout`, cleanup complete; not a software pass |
| Post-format checkpoint | Both arms' independent/stable cases passed (four runs), startup contract and read-only identity inspection passed |
| Repaired ready stage | Ten actual runs passed: both arms × independent/stable/final-only/mixed-active/restart-reconnect; immediate and post-source stop/timeout failed truthfully, completed cleanup and released owned locks |
| Previously skipped shutdown integrations, explicitly enabled | Both passed with real synthetic Python subprocesses; no native OpenVR process |
| Ruff/Black | Ruff passed after unused-local/import cleanup; Black formatted the three tool/test files |

The full 48-case matrix, maximum bursts and stop/timeout receipts are retained in
the pre-format stage `C:/pph206-hmd-kit/issue206-final/offline-verification.json`;
its preparation SHA256 is
`f4135cff509c6a7c4c3ca778d9e587efdd9b4c4aa9b9377888713f04ffaf7113`.
The intermediate `issue206-reviewed/offline-verification.json` identifies four
post-format runs and documents retained evidence applicability. Formatting removed only an unused
local binding (preserving validation), an unused import and reordered imports;
normalized operational AST comparisons matched. The 48-case matrix is retained,
not mislabeled as rerun against different harness bytes.

Independent review of preparation checkpoint `5869d04a4c0c1dd00d5216539ba0fd7872b1f1b3`
found a medium-severity abort gap before scenario entry: cancellable environment
inventory was outside the receipt finalizer, so cancellation could leave a
controller lock without a report even though native had not started. The
finding was accepted. Four focused regressions failed before the repair.

The repaired worker covers preparation, inventory and partial initialization
with its monitored failure/receipt boundary. The controller also covers
confirmed pre-spawn failures, and worker bootstrap failures receive an explicit
pre-runtime receipt. Stop is checked before initialization and native start.
Unknown worker/native termination or ambiguous spawn cancellation remains
fail-closed rather than claiming cleanup. Read-only inventory work already
running in an executor can finish under its own ten-second timeout; cancellation
is not falsely described as terminating that work. The controller waits for
worker exit before releasing ownership.

`issue206-ready-fixed/repair-verification.json` records the 127-pass affected
suite, ten fresh owner scenarios, four actual early/active stop/timeout receipts,
current source/native/harness identities and remaining limits. Its scenario
engine and provider/presenter helpers match the prior checkpoint's operational
AST; the full earlier 48-case matrix and maximum bursts retain scenario
applicability, but are not substituted for the refreshed abort/cleanup proof.
The two separately passed production-runtime subprocess integrations and the
isolated application-control complement are unaffected by this tooling repair.

Observed baseline independent ingress had no source/provider call before UI
release; candidate had source and one provider call. Stable and final-only
baseline routes already progressed. Four stable extensions used four provider
calls in each arm; a resumed-speech case used two. The kit does not manufacture a
one-request claim for an evolving burst. The 60-second burst wakeups overshot
their scheduling window by about 0.012 seconds, and complete invocation exposure
was about 62.7 seconds. Neither is an HMD latency metric or hard real-time bound.

Focused suite command:

```powershell
uv run --frozen pytest tests/scripts/test_ovr_hmd_measurement.py tests/core/test_self_ui_isolation.py tests/core/test_overlay_active_freshness.py tests/core/test_overlay_bridge.py tests/core/runtime/test_overlay_runtime.py -o addopts= -q --tb=short
```

The two safe subprocess integration checks were subsequently run explicitly:

```powershell
$env:INTEGRATION = '1'
uv run --frozen pytest tests/core/runtime/test_overlay_runtime.py::test_overlay_runtime_receives_real_subprocess_shutdown_ack_before_reader_cleanup tests/core/runtime/test_overlay_runtime.py::test_runtime_real_bridge_writer_delivers_one_shutdown_before_delayed_child_exit -o addopts= -q --tb=short
Remove-Item Env:INTEGRATION
```

At the final preparation inspection, identity verification passed, SteamVR's
process pair and a pre-existing native overlay were absent, and
`vr_initialized=false` / physical `not_observed` were reported. Recheck process
and identity prerequisites when the wearer returns; these observations are not
a reservation of the future environment or a live-safety pass.

## Short physical session: 2026-10-03 UTC

The wearer confirmed that the HMD was worn, authorized each run separately,
selected Virtual Desktop and confirmed this was the same environment in which
the earlier delay occurred. Headset model, firmware and refresh rate were not
supplied; SteamVR/VRChat version fields returned null and remain unknown.
Windows build 22631 and AMD Radeon RX 7900 XTX driver 32.0.31041.1004 were
recorded. Kit/source/native identities matched the prepared pins above.

Only `stable` with `head_locked` placement ran, candidate first then baseline,
once each. External translation was the gated synthetic provider, not a paid
service or live STT. The wearer chose **“여기서 마치기”** after the baseline
observation. No subsequent scenario or overlay launch was authorized or run.

| Run | UTC start | Worker elapsed, not display latency | Software and cleanup | Wearer observation |
| --- | --- | --- | --- | --- |
| `candidate-stable-90d9620b` | `2026-10-03T18:08:24Z` | 7.792 s | Pass; source applied and one provider invocation while UI queue full; identity preserved; native graceful exit 0, no forced termination, readers/cleanup complete | Original first, translation added, disappeared after shutdown; no discomfort |
| `baseline-stable-9bd7aad4` | `2026-10-03T18:10:56Z` | 7.658 s | Pass; source applied and one provider invocation while UI queue full; identity preserved; native graceful exit 0, no forced termination, readers/cleanup complete | Both versions normal; no perceived difference; no discomfort |

The observation method was the wearer's qualitative HMD report through the
conversation, not a mirror screenshot or instrumented latency measurement.
The baseline stable active path already supports source-first publication;
these two normal observations neither demonstrate an improvement nor establish
that the historical SteamVR delay is fixed.

Raw software receipts and their sidecars are under
`C:/pph206-hmd-kit/issue206-ready-fixed/runs/<run-id>/report.json`.
Separate `observation.json` records preserve the wearer's `no_issue` reports and
uncertainty; the immutable software report's initial `physical_hmd` field is not
rewritten into a physical pass. Session setup/termination is retained in
`live-session-candidate-stable-90d9620b.json` at the stage root.

Both owned test native processes exited normally. A later final process
inspection detected another native overlay from the ordinary repository
`build/overlay` path, not the kit's pinned runtime path. It was not terminated
or modified. No claim is made that every user's overlay process was stopped.

### Native-stage evidence limit

Both reports retained bridge/process lifecycle evidence but had empty detailed
native stage/outcome counters and unknown loss-counter samples. This is
**unavailable detailed evidence**, not proof that no rendering happened.
Read-only inspection of the pinned native source found presentation records
collected internally but no production export to the Python parser's
`presentation_diagnostics` marker. Python measurement capture was already on;
no supported settings-only switch or early-report race was found to explain
the gap. Empty native log directories are consistent with the logger's
stdout/stderr behavior.

Consequently these runs do not establish per-caption native render/submit
counts, retry opportunity completion/deadlines, GPU timing, diagnostic loss-free
delivery or sustained native resource behavior. No instrumentation, production
code, retry profile or staged executable was changed during this session.
Detailed native-stage acceptance needs a separately authorized, bounded export
and repinned executable/evidence; repeating this exact build alone is not a
supported way to obtain those missing counters.

### Remaining physical scope

Independent/final-only, bursts/resume, mixed-channel/eviction, sticky/expiry,
dedicated clear/OFF, spatial lock, restart/reconnect, sustained exposure and the
isolated real-application ON/manual/OFF check are **not run in this session**.
Automatic application recovery and long-session/device-wide stability remain
unverified. Observed disappearance on the two ordinary shutdowns is not relabeled
as completion of those dedicated scenarios. The user-ended session is complete
as a two-case record, not full issue-206 physical acceptance or issue closure.

## Resumed batch: 2026-10-03 UTC

The wearer requested resumption without per-case questions and explicitly
authorized a sequential batch in the same Virtual Desktop setup. Identity and
SteamVR prerequisites passed; no other native overlay was present at admission.
The plan reused the earlier stable/head-locked pair and ordered the other
42 short cases as baseline/candidate pairs, head lock before spatial lock.
Sustained runs remained gated by successful observed short cases.

`issue206-ready-fixed/resumed-batch-01/state.json` retains the ordered plan and
receipts. The batch ran from `19:07:06Z` to `19:10:49Z` (223.053 seconds,
not a caption-latency measurement). The first 18 runs passed software and
cleanup: both arms of independent, final-only, stable burst, resume, mixed
active, finalized eviction, sticky, expiry and clear/OFF, all head-locked.

Case 19, `baseline-restart_reconnect-b19d60e9`, failed with
`bridge_auth_failed`. The batch automatically stopped and did not launch the
remaining 23 planned cases or the separate actual-application complement.
The first native process acknowledged shutdown and exited 0. Its replacement
exited 12 with readers complete; aggregate cleanup correctly remained failed.
No automatic retry, failure-to-pass rewrite or background continuation occurred.

The wearer answered **“계속 지켜봤고 이상 없었음”** and
**“완전히 사라짐”** after this stop. Nineteen `observation.json` sidecars
correlate that continuous batch-level qualitative observation to the attempted
runs. They are not individual latency or baseline/candidate difference reports.
The failed reconnect remains failed despite no observed display anomaly.
`resumed-batch-01/wearer-observation.json` records this distinction and the
unexecuted scope. With the earlier stable pair, 20 short head-locked runs have
software passes and wearer observations; no spatial case has run yet.

Read-only diagnosis found a harness contract error: each fresh native process
authenticates with wire runtime generation 1, while the old harness used its
run-wide restart ordinal as the replacement bridge's wire generation.
The replacement bridge therefore required 2 and rejected native generation 1.
It was not reuse of a consumed token: the bridge and token were both fresh.
Production generation construction uses a fresh instance identity and the
bridge's default wire generation 1. This receipt does not demonstrate an
application automatic-recovery regression; replay/deadline checks after the
first restart were never reached.

The harness repair keeps restart ordinals as evidence metadata, creates a
fresh instance identity per owner and uses the existing per-process wire
contract. Offline transport must authenticate through the real bridge using
that native-compatible contract, rather than bypassing authentication.
Production source pins, native executable, protocol and retry policy are
unchanged. The corrected stage is `C:/pph206-hmd-kit/issue206-resume-fixed`;
old receipts retain their original harness identity and are not relabeled.

After the controller and both native PIDs were confirmed absent, with no
native overlay present, the failed run's live guard was preserved as
`resumed-batch-01/failed-live-guard.json` rather than silently discarded.
The original failure and cleanup result are unchanged. Further HMD exposure
requires completed repair verification and renewed wearer readiness.
The remaining short scope is the repaired baseline reconnect, candidate
reconnect and 22 spatial cases, followed by the separate application complement
and only eligible optional sustained cases. Detailed native-stage/resource
evidence remains unavailable as described above.

### Corrected-kit nonphysical verification

The corrected helper SHA256 is
`144aa0663f246b25dea6c4bd42ab680a8ef6bf0fef727ee291e6044c0328c646`;
controller and shared native hashes are unchanged. The affected harness suite
passed **61 tests** (`uv run --no-sync pytest tests/scripts/test_ovr_hmd_measurement.py`).
Regression coverage authenticates three fresh owners against real bridges and
rejects a native-compatible wire-1 client when a bridge incorrectly requires 2.

The corrected staged CLI also executed actual offline `restart_reconnect`
scenarios against both pinned source arms:

- `baseline-restart_reconnect-df97b515`;
- `candidate-restart_reconnect-6bb41da3`.

Both report software pass and complete cleanup, three authenticated/detached
bridge sessions with distinct instance IDs, restart ordinals 1/2/3 and wire
generation 1, preserved original/replayed deadlines, stale late translation
and empty final blocks. These are real WebSocket/owner-path checks, not native
or HMD runs. They do not replace the still-required live reconnect comparison.

### Corrected live batch and short-case coverage

After the repair checkpoint `c8e19ad4657b8c96bbb2d510c97bc5112dfb8077`
passed independent whole-checkpoint review, the wearer renewed readiness with
**“껐고 착용 중, 나머지 일괄 시작”**. The corrected stage's
`resumed-batch-02/state.json` records **24 software/cleanup passes**:
the remaining head-locked baseline/candidate reconnect pair, then all eleven
short scenarios on both arms with spatial lock. The command wall time was
349.11 seconds, not a display-latency measurement.

The wearer reported **“계속 봤고 모두 정상”**, including normal disappearance,
after this entire corrected batch. Twenty-four new observation sidecars retain
that ordered batch-level qualitative observation and its limits.
`short-coverage.json` verifies **44 unique passing observed short combinations**:
11 per arm/anchor, each with live mode, software pass, complete cleanup,
`no_issue` observation and matching report SHA256.

Twenty records retain the original harness aggregate identity
`a43d0d9f83993732bdb766e0ecc7fed1e404497ed1bcc5f46aac24edafca72b0`;
24 use the corrected aggregate
`c61ba7691fed792b32715ea07353da4bf5e0d5191a885212fb89234a225b755b`.
`carryover-index.json` lists the exact copied original evidence and file hashes.
Independent review accepted retention because those earlier single-owner
scenarios already used wire generation 1 and their product/native bytes,
scenario semantics and retry policy were unchanged. Copies are not new runs.
The original failed reconnect remains at the old stage and does not qualify.

The wearer separately authorized four ten-second sustained comparisons followed
by the application complement. Two initial sustained admissions were blocked
before any run/native launch because the ordinary installed application was
present; their zero-completion states are retained as `sustained-batch-01` and
`sustained-batch-02`. After the wearer confirmed application closure, the
remaining shutdown interval ended and targeted inspection found no guarded
application/native process. The Director retained that renewed readiness for
a new admission, without relaxing the guard or adding an automatic retry loop.
These admission blocks are neither physical executions nor passes.

### Sustained observations and actual-application boundary

`sustained-batch-03/state.json` records four successful native software/cleanup
runs, each requesting a ten-second update interval:

- head-locked baseline `baseline-sustained-d1107f27`;
- head-locked candidate `candidate-sustained-e0d90687`;
- spatial-locked baseline `baseline-sustained-cf0e422d`;
- spatial-locked candidate `candidate-sustained-dd655f30`.

The wearer answered **“계속 봤고 모두 정상”** and **“완전히 사라짐”**.
Four correlated observation sidecars retain those qualitative results.
The session therefore has **48 passing observed synthetic-owner/native cases**
(44 short plus four sustained), not 48 real-application tests.
`final-physical-observation.json` retains the final physical scope and limits.

The separate actual-application attempt is **failed**, not included in that
count. `app-complement-02/14-overlay-set-on.json` has an applied terminal ON
receipt, but `configured_target=steamvr`, `effective_target=desktop` and
`fallback_active=true`. The operator rejected this state before manual text
submission. An applied ON receipt and connected desktop fallback do not prove
SteamVR presentation. No manual-source HMD observation is claimed.

The app log at
`C:/pph206-hmd-kit/app-control/localappdata/puripuly-heart/puripuly_heart.log`
records the original SteamVR generation failure as `steamvr_not_installed`.
Its retained diagnostic file
`diagnostics/overlay/overlay-diagnostics-failure-20261004-050215-561115000-overlay-a71aeae56cdc918f.jsonl`
records native startup failure; child PID 204 exited 20 with readers complete
and no force. In the pinned native implementation this means
`VR_IsRuntimeInstalled()` returned false before OpenVR background initialization,
HMD discovery or rendering. The same pinned bytes passed the preceding native
suite; discoverability in the application's separately isolated environment
must not be inferred from that success.

OFF returned applied/off/runtime-inactive, app stop returned terminal stopped,
and post-stop status returned `instance_not_found`. The final process check
found owned host PID 33576 absent and no native overlays. The wearer was told
the HMD could be removed; further live application exposure requires renewed
readiness, not an automatic retry.

The application operator's Windows PowerShell 5.1 subprocess handling was
validated before this attempt using actual safe Python stderr/exit probes
(`app-native-smoke-04/result.json`), not application launches. Separate stdout
and stderr capture preserves expected exit-3 absence JSON; typed state checks
and failed outcomes remain strict. The identity verifier checks all 1,378
source files and 31 native assembly files even under Python optimization.
Four seconds is the intended post-admission hold, not a measured ON-to-OFF
duration; this failed attempt never reached that hold.

These results do not establish detailed native render/retry/resource accounting,
application automatic recovery, exact HMD latency, long-session stability,
device-wide safety or resolution of the historical SteamVR delay.

### Resolved app discovery and final physical check

`openvr-discovery-proof.json` isolates the original app failure without VR
initialization. With the pinned `openvr_api.dll`, `VR_IsRuntimeInstalled()` was
true in the regular environment and false under the exact app profile
isolation. Restoring only `USERPROFILE` restored discovery; keeping all profile
isolation and setting `VR_PATHREG_OVERRIDE` to the original Windows shell
OpenVR registry also restored discovery. The redirected profile changed shell
LocalAppData to `app-control/home/AppData/Local`, where no registry existed.
Only path/existence/API booleans were recorded, not registry contents.

The live operator now captures the original shell registry before isolation and
binds that single runtime-discovery prerequisite before host startup. It does
not expose the real home/profile or change application settings, secrets,
product/native bytes, or the user's registry. The exact binding fragment passed
a no-initialization probe (`app03-environment-binding-proof.json`).

The wearer then explicitly selected **“착용 중, 마지막 앱 확인 진행”**.
The separately recorded `app-complement-03` attempt passed:

- instance `711d02f2-fe82-4ac7-82e9-5b7eaedadcda`;
- ON applied/terminal, configured and effective target `steamvr`, connected
  process/runtime, presentation ready, `fallback_active=false`;
- public manual-source submission applied/terminal with capture and translation
  still disabled;
- OFF applied/terminal, desired false, lifecycle off, effective target null and
  runtime inactive;
- app stop terminal/stopped, then exit-3 `instance_not_found`;
- final owned host PID 7984 absent and no native overlay processes.

The wearer answered **“정상 표시됐고 완전히 사라짐”**, including no
placement/flicker/residual-text issue or discomfort. `app-complement-03/observation.json`
correlates this actual-application HMD observation with report SHA256
`d17c98e7d23f74fae56ce62034ca97c606244afe11640daf71c99ef5436c1ca8`.
The earlier fallback attempt and both prelaunch admission blocks remain intact;
none is rewritten into success.

The bounded session is complete: **44 short native scenarios, four ten-second
sustained scenarios, and one corrected-environment actual-application
ON/manual-source/OFF observation** passed their declared software/physical
checks. This is not full issue-206 remediation, a long-session guarantee,
native-cost measurement, or application automatic-recovery acceptance.
