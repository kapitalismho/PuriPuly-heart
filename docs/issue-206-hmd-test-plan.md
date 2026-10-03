# Issue 206: SteamVR worn-HMD test plan

## Purpose and authority

The reported delay/staleness occurred in **SteamVR**. Desktop captures, Presenter acceptance, successful OpenVR API calls, and native health are not proof that the wearer saw current pixels.

The maintainer requested a test plan and complete preparation up to the test start, then a joint session after returning and putting on the HMD. Preparation must not start a live overlay, SteamVR, VRChat, microphone/loopback capture, or paid provider requests. Physical acceptance remains **not run** until that session. See [software evidence](issue-206-verification.md).

Compare product baseline `8666bb57935b7c6da0c3c8aeaec9d116762d9f3d` with candidate `c501b83350d4c39129cf61dd3e582b6ebbea04a5`. Use separate pinned Python/source exports, not two labels pointing to the candidate Python code. The native production implementation is unchanged between these revisions, so use one shared, hash-verified native assembly and record its actual build provenance. Full native source trees are not identical: the candidate adds tests.

## Operator commands and resume point

Run from the development checkout in PowerShell. The prepared-session location
for this work is `C:/pph206-hmd-kit/issue206-reviewed`; use its manifest and reports,
not an installed executable or a newer unrecorded Python source tree.

Safe before the wearer returns:

```powershell
$stage = 'C:/pph206-hmd-kit/issue206-reviewed'
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

```powershell
. C:/pph206-hmd-kit/app-control/session.ps1
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

Check the JSON effective OFF state before proceeding. Only later, with the
wearer's renewed confirmation and no other native overlay, execute each live
command separately: `pph overlay set on`, `pph overlay status`, then
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

Ready stage: `C:/pph206-hmd-kit/issue206-reviewed`.
Preparation receipt SHA256:
`e92c78c91aa578b27daae09a5bad8ecb28d1bbf9cb73319323564af495792dcf`.
Harness aggregate SHA256:
`dab10c5640c8997f9aa7c04a9a1b7ea6e68345213edd24ad4388f9622122de46`.
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
| Focused harness/owner/bridge/runtime suite | 115 passed; two existing real-subprocess tests initially skipped because `INTEGRATION` was unset |
| Both arms × both anchors × all 12 actual offline scenarios | 48 passed, zero failed; real controller/worker processes and owner engine, not a substituted short sequence |
| Both maximum offline burst cases | 60-second requested scheduling window per arm, 119 meaningful updates/provider calls per arm, cleanup complete |
| Actual run-correlated stop after source/provider start | Controller exited 2, `operator_stop`, cleanup complete; not a software pass |
| Actual timeout after source/provider start | Controller exited 2, `run_timeout`, cleanup complete; not a software pass |
| Post-format ready stage | Both arms' independent/stable cases passed (four runs), startup contract and read-only identity inspection passed |
| Previously skipped shutdown integrations, explicitly enabled | Both passed with real synthetic Python subprocesses; no native OpenVR process |
| Ruff/Black | Ruff passed after unused-local/import cleanup; Black formatted the three tool/test files |

The full 48-case matrix, maximum bursts and stop/timeout receipts are retained in
the pre-format stage `C:/pph206-hmd-kit/issue206-final/offline-verification.json`;
its preparation SHA256 is
`f4135cff509c6a7c4c3ca778d9e587efdd9b4c4aa9b9377888713f04ffaf7113`.
The ready-stage `offline-verification.json` identifies the four fresh runs and
documents retained evidence applicability. Formatting removed only an unused
local binding (preserving validation), an unused import and reordered imports;
normalized operational AST comparisons matched. The 48-case matrix is retained,
not mislabeled as rerun against different harness bytes.

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
