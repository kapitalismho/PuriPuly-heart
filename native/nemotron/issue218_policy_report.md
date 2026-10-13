# Issue 218 practical-policy non-VR validation

## Disposition

Measured product candidate: `dd3cad278dc683840c6eabaae9777159cb75a710` (starting baseline `b2cfc0c42bfa07d1570c243fb17ab2e547b8da1e`). The approved **30 actual local GPU cases and 12 fixed-native-input Deepgram replays executed**, with real local Gemma Vulkan translations. This is a completed measurement, **not speaker-feature acceptance, profile selection, or production enablement**.

Required speaker gaps remain: clean local B and published returning-A identities are gray; Deepgram short A–B–A is not independently attributed. The original Deepgram matrix's visual timing failed: all owned screenshots were blank despite dispatch ACKs. A separate renderer-only investigation now establishes representative actual surfaces and a task-owned visibility/layout prerequisite; it does not backfill that matrix's pixels or E2E latency. No gray, nonempty result, first token, ACK, or fewer-job count is substituted for acceptance.

The worker changed only new `issue218_policy*` drivers/reports/archives/PNGs and their `build/issue218_policy*` outputs. Product, maintained docs, Git/GitHub, settings and default profiles were untouched. No provider network call, credential read, ambient capture, VRChat or SteamVR launch occurred.

## Real paths and preserved conditions

- GPU Qwen worker/model, Nemotron F16 artifact/runtime and Gemma server/model hashes were checked. Exact artifacts/dependencies are in the [structured report](issue218_policy_report.json) and execution traces.
- Qwen activated `vulkan-index-0`, registry index 0, AMD Radeon RX 7900 XTX. Gemma reported effective backend `vulkan`, with the production managed owner, current prompt, prefix preparation/cache and concurrency 3. Nemotron diagnostics reported Vulkan, neural CPU fallback forbidden, eight slots, complete geometries `9/4/0/264/264/222` and `6/2/0/264/264/222`.
- The evidence pump used the production 50 ms sleep, not the old 5 ms pump. Actual Windows/event-loop median intervals were approximately 62 ms; maximum was 170.127 ms. Actual timings remain in the traces.
- Genuine 500 ms onset recovery, 300 ms rollover context, exact PCM/native options, trimming OFF and no synthetic native tail were retained.
- The driver delegates actual `TranslationChannelOwnerCallbacks`, production peer update/boundary/terminal handlers and actual capture ready-receipt methods. It does not convert old admitted sources or decide attribution. The local snapshot observer delegates the candidate's `_attribute` unchanged.

## Local GPU: 30 cases, 62 real translations

Each row ran twice, with rotated matched baseline/full 1.04/full 0.64 arms.

| Case | Changed-path content | Identity and actual pixels |
| --- | --- | --- |
| `clean_A_B` | Both full sentences retained in both profiles; matched no-earlycut baseline loses A. | A is gold; B is gray in all four new runs. Clear A/B acceptance fails here. |
| `distinct_repeated_A_B_A` | Initial A, full B, returning-A request and native rollover tail retained. Native rollover fragments/repeated `from his` remain. | Initial A/ B are distinct same-epoch 0/1, gold/cyan. Returning A is gray in all four new runs. |
| `short_A_B_A` | Three independent sources: full A, `No.`, full returning A. Real `아니요.` translation retained. | A gold, actual short No cyan, returning A gray. Short B is not absorbed or discarded. |
| `linguistic_correction` | `No, no, three.` retained in every arm. | Full 1.04 unavailable; full 0.64 identified/gold. |
| `six_second_rollover` | Required continuation words retained in all arms; the native `with the` repetition/translated continuation fragment is not repaired. | Full 1.04 both sources gray; full 0.64 both same-epoch slot 0/gold. |

Strict ordered required-word availability fails **4/30** cases, all matched-baseline clean/repeated A/B cases. It passes all 20 new-arm cases, but is not a perfect-verbatim or semantic-quality certificate. Complete-turn latency is null on omitted content.

Identified sources: baseline 0/14; full 1.04 10/24; full 0.64 16/24. All 62 timed local owned readbacks contain caption-color pixels. Useful captions are distinct from ACKs: for repeated A/B, first-A stable readback precedes the six-second enclosing seal by approximately 1.65–2.13 s; B by 0.45–0.87 s. The summary retains each ACK/readback bound separately.

### Actual local PNGs inspected

- [Gold A / cyan B, coherent full sentences](issue218_policy_gpu_valid/commit_23.png)
- [Gold A / cyan short No](issue218_policy_gpu_valid/commit_51.png)
- [Cyan No / gray returning A](issue218_policy_gpu_valid/commit_52.png)
- [Full 0.64 six-second continuation, both gold](issue218_policy_gpu_valid/commit_59.png)

These are owned Win32 client crops over the owned opaque checkerboard. They are software pixel observations, not photon onset, HMD readability or VR coexistence.

## Own-final gap evidence

The actual driver records the candidate's native-result delivery, dequeue-time attribution snapshot/frontier and later wrapper/source release separately. There are **48 snapshots: 26 identified, 21 unavailable, 1 uncertain**. The Director independently accepted the candidate's F1 final-emission/dequeue eligibility race and F2 repeated support-scan findings for subsequent repair; this measurement does not claim either is repaired.

The source owner's [bounded read-only disposition](issue218_speaker_policy_report.json) explains the observed gray outcomes without model reruns or retuning:

| Source / profile | Missing eligible suffix at current freeze |
| --- | --- |
| Clean B, both profiles/repeats | `[70848,77312)`: 404 ms; includes scoring speech. |
| Repeated returning A, 1.04 | `[93888,105472)`: 724 ms. |
| Repeated returning A, 0.64 | `[101568,105472)`: 244 ms. |
| Short returning A, 1.04 | `[82368,92160)`: 612 ms. |
| Short returning A, 0.64 | `[86208,92160)`: 372 ms. |
| Numeric correction, 1.04 | `[35264,41984)`: 420 ms; no scoring speech overlap. |
| Six-second first source, 1.04 | `[116672,116736)`: 4 ms. |
| Six-second successor, 1.04 | `[128192,136704)`: 532 ms. |

Returning-A delivered prefixes and later suffix evidence support the same slot 0; that does **not** qualify published returning-A identity or permit future recoloring. One complete-coverage 0.64 repeated-rollover snapshot is uncertain: 120 ms uncertainty versus 236 ms unique clear support (50.8%), exceeding the 20% ratio; replay cannot enlarge the denominator. No reset/epoch loss or hard-cache exhaustion was observed. Snapshot counts were 62–840, below 2048. Actual GPU trace does not retain per-frame Silero probabilities/`speech_observed`, so missing suffixes cannot retrospectively be called VAD-known quiet.

## Deepgram: 12 fixed-native-input replays, 23 real translations

All arms use the same canonical historical baseline-rep1 tape for each case. Submitted PCM is reconstructed through the original source maps and checked against payload/request hashes. Original successful write cursors and controls are retained; raw receipt cursors match exactly. One chronological capture/native-input scheduler verifies that every submitted mapped endpoint has already been captured. Only host origin is rebased; no guessed cursor, future capture or delayed native final is introduced.

The actual current `_DeepgramSDKSession` / `_build_transcript_event`, scoped projection, `STTScopedTurnNormalizer`, changed streaming wrapper, production peer handlers, real current Nemotron and Gemma consume these tapes. Native timings are **historical ASR evidence**, not fresh cloud latency, fresh VAD qualification or service qualification.

| Tape | Observed changed source composition / limitation |
| --- | --- |
| Clean A/B: 1 Results, 4.724 s submitted | `Gentlemen of the press.` is one identified source and one real useful translation in each new arm. Whole A is absent from the historical native tape. |
| Repeated A/B/A: 3 Results, 7.744 s; controls 103744 / 123904 | `Washed the detritus from his hair. ` and `Gentlemen of the press.` remain coherent separate A/B units, not gray/gold word fragments. Native returning fragments/duplication remain; final `from his hair.` preserves A0 metadata, preceding return fragment is unavailable. |
| Short A/B/A: 2 Results, 5.652 s | Exact `No.` is retained inside a mixed gray already-finalized native phrase. It is not assigned to majority A or deleted, but independent short-B attribution is not qualified. |
| Numeric: 1 Results, 2.58 s | Exact `No. No. Three.` and real `아니. 아니. 셋.` retained as one coherent source. |

Translation jobs: 7 baseline, 8 full 1.04, 8 full 0.64. This is meaning/source preservation evidence, not a fewer-jobs result. Strict required content fails 9/12 cases because native omissions/glue remain; all authoritative received native text is conserved exactly once in all 12.

### Actual replay presentation failure

Flet acknowledged nonempty committed snapshots, but **all 23 timed owned readbacks and 12 final pictures are blank** with zero gold/cyan/gray caption pixels. The renderer's reported slot count and ACK therefore do not establish useful visible output. Original replay pixel/complete-turn readback latency stays null. The later bounded investigation below reproduced a concrete task-window/compositor prerequisite; the historical run did not record cloak state, so attributing every original blank to that mechanism remains **[INFERENCE]**. Publication generation 0 appears in both working and blank runs and is not a demonstrated cause.

Original blank PNGs actually inspected: [A/B request](issue218_policy_deepgram_final/commit_20.png), [return request](issue218_policy_deepgram_final/commit_22.png), [short No request](issue218_policy_deepgram_final/commit_25.png), [numeric request](issue218_policy_deepgram_final/commit_45.png). Requested texts/styles, real translations, HWND/PID/crop and software ACKs remain correlated in the unchanged raw trace. No additional frozen-candidate ASR/translation matrix was run.

Read-only inspection also found actual gray `기자 여러분.` / `Gentlemen of the press.` pixels in the first PCM/cursor-verified cleanAB subcase of `issue218_policy_invalid_live_vad.zip:commit_1.png`. The whole attempt is still invalid because its later ABA cursor mismatch aborted it. This is bounded renderer compatibility evidence, not a matrix-arm replacement or rich-color qualification, and it does not demonstrate universal Deepgram/Flet incompatibility. The temporary extracted crop was removed after inspection.

### Bounded renderer-only investigation and task-harness correction

The current actual Flet renderer consumed the **exact previously observed real source/translation snapshots**, preserving text, palette, scope, generation, IDs, revision and event order. Presentation file hashes matched the original execution manifest in every probe; current F1/F2 backend changes were bypassed. This work made **zero ASR, Gemma or provider-network calls** and does not qualify new E2E latency.

- Unguarded diagnostic: working GPU-derived control **4/4**; Deepgram **20/23** positive. All 20 positives had DWM cloak 0 before/after; all three blanks (41, 42, 45) had cloak **2, `DWM_CLOAKED_SHELL`**, despite visible/nonminimized native/Flet state, correct retained text and ACKs. Microsoft defines a cloaked window as not visible to the user ([primary reference](https://learn.microsoft.com/en-us/windows/win32/api/dwmapi/ne-dwmapi-dwmwindowattribute)); cloak 2 alone does not prove which shell feature hid it.
- The task-only guard checks owned PID/HWND, visible/nonminimized state and cloak 0 before **and** after pixels, plus requested palette pixels and stable revision. It permits at most two owned show/restore/current-desktop attempts per capture and one same-revision public-owner bounds reapply/recapture. It never switches global desktops/settings or manipulates unrelated windows. ACK is never readback validity.
- Full guarded replay: **22/23** valid, including previously blank 41/42/45. Numeric **29 remained failed**: the owned window was minimized, native restore succeeded, but the retained locked-mode layout stayed 160×28 instead of 1344×336. An auxiliary task-driver cloak-injection check failed with `E_ACCESSDENIED`; it was removed, retained as failed evidence, and is not a product failure.
- Changed-guard working control: **4/4** valid. The first numeric-only correction still failed because the owned opaque backdrop did not cover the restored client. The next correctly reapplied the existing public `apply_window_bounds` owner control, but its recapture was blocked by shell cloak 2; two owned COM desktop-move attempts returned `E_ACCESSDENIED`. These attempts remain failed, not pixel passes.
- The **final, single bounded owned-ShowWindow-correction attempt** restored the owned minimized window, reapplied the same public-owner layout bounds and captured exact numeric **29 successfully (1/1)**: both windows uncloaked before/after, 2,228 gold pixels, actual `아니. 아니. 셋.` / `No. No. Three.`. The final recapture was already uncloaked; this does **not** demonstrate that the new show-on-cloak branch or COM migration can recover every shell-cloaked window. No further desktop/COM experiment or model matrix was run.

Thus the evidence is **22/23 full guarded replay + 4/4 changed-guard countercontrol + 1/1 targeted numeric recovery**, not a relabeled full E2E matrix pass. It covers all distinct retained semantic/palette paths. No product source, palette or renderer defect was demonstrated; the observed correction is task-owned window visibility/layout management. Shell-cloaked recovery remains permission/environment-dependent; unavailable visibility explicitly fails without a pixel claim.

Actual renderer-only PNGs inspected:

- [Coherent Washed sentence gold / Gentlemen sentence cyan](issue218_policy_renderer_guarded/Deepgram_original_submissions/commit_20.png)
- [Gray returning-A prefix / gold same-A tail](issue218_policy_renderer_guarded/Deepgram_original_submissions/commit_22.png)
- [Gray native mixed phrase preserving No](issue218_policy_renderer_guarded/Deepgram_original_submissions/commit_41.png) and [gray continuation](issue218_policy_renderer_guarded/Deepgram_original_submissions/commit_42.png)
- [Validated gold numeric 45](issue218_policy_renderer_guarded/Deepgram_original_submissions/commit_45.png) and [recovered exact numeric 29](issue218_policy_renderer_owned_show_final/Deepgram_original_submissions/guard_recovered_29.png)
- [Changed-guard gold/cyan control](issue218_policy_renderer_repair_verify/GPU_working_control/commit_23.png) and [cyan B / gray return control](issue218_policy_renderer_repair_verify/GPU_working_control/commit_24.png)

Numeric 29 and validated 45 have identical requested visible text, languages, gold style, calibration, one-card finalized layout and publication scope/generation/order/index; only nonvisual identities/time differ. Their positive PNGs are **byte-identical**, SHA-256 `285cfe00456f8a88967bba0baa869b285ec29f8e89574baf2a4b4d2fae2e09be`. Numeric coverage was therefore already representative before the final recovery; the failed 29 attempts are still retained separately. Gray mixed No and gray return remain real attribution limitations, not recolored substitutes.

The [renderer findings ledger](issue218_policy_renderer_findings.json) records every historical probe command, execution source/helper hashes, per-revision result, scope and current archive-member reference. The first diagnostic source snapshot was reconstructed from the exact known edit delta and verified against its recorded execution driver hash; subsequent snapshots were saved before execution. All failed attempts remain archived; direct PNGs remain readable. Redundant direct traces/source snapshots were removed only after archive equivalence verification.

Historical renderer-only commands exercised, in order (execution-time paths; not new execution requests):

```text
.venv/Scripts/python.exe -B native/nemotron/issue218_policy_renderer_probe.py --output native/nemotron/issue218_policy_renderer_probe_run1
.venv/Scripts/python.exe -B native/nemotron/issue218_policy_renderer_probe.py --output native/nemotron/issue218_policy_renderer_guarded
.venv/Scripts/python.exe -B native/nemotron/issue218_policy_renderer_probe.py --repair-verify --output native/nemotron/issue218_policy_renderer_repair_verify
.venv/Scripts/python.exe -B native/nemotron/issue218_policy_renderer_probe.py --repair-verify --numeric-only --output native/nemotron/issue218_policy_renderer_numeric_recovered
.venv/Scripts/python.exe -B native/nemotron/issue218_policy_renderer_probe.py --repair-verify --numeric-only --output native/nemotron/issue218_policy_renderer_owned_show_final
```

Reproduction uses each archived execution driver/helper, not a current driver mislabeled as the historical one. The current validation driver delegates this same task-only guard for any separately authorized later measurement; no post-guard ASR/translation run has occurred.

## Conservation, bounds and durability

Across **42/42** valid-input cases: capture and declared native PCM mismatches 0; exact authoritative text conserved once; source history registered once; source/publication order preserved. These are changed-path observations, not replacement claims for unchanged reset/cancel/Unicode/TTL suites.

Speaker peaks: pending 18,432 samples / 17 items; retained map pieces 38; evidence queue 148; evidence evictions/overloads 0; maximum copy 512 samples; pending 0 at drain. The 300 Windows 1 s counter samples had no invalid values. Adapter dedicated-memory sampled peak was 11,329,855,488 bytes; shared peak 2,308,300,800 bytes. Counters include surrounding applications; they are not isolated inference GPU timestamps or whole-host peak-VRAM qualification. Per-process samples, exact queues and native output are retained.

Four invalid task-driver attempts are separated in archives: missing local backend `provider_id`, new-live-VAD/historical receipt cursor mismatch, parallel capture availability mismatch, and omitted current scoped normalizer. None is reported as a provider/product failure or a valid acoustic case. All 14 owned ZIPs passed container SHA-256, exact member-set, member SHA-256/byte-count and CRC verification, including 14 nested execution-source ZIPs. After current readers finished, **18 byte-equivalent non-PNG direct copies (131,109,893 bytes)** were removed; all ZIPs, every direct PNG, root reports, saved summary and drivers remain. Earlier issue evidence was untouched. The [cleanup ledger](issue218_policy_cleanup.json) records exact removed paths and archive-member replacements.

- [Per-source/latency/provenance summary](issue218_policy_summary.json)
- [Complete compact outcome ledger](issue218_policy_report.json)
- [Commands and failed attempts](issue218_policy_attempts.json)
- [Exact archive/current-driver/source-snapshot hashes](issue218_policy_archive_inventory.json)

Valid raw trace SHA-256:

- GPU: `66148f92fff4a80b95c626f356e944f326adfcdd0aa1c3330ba19693fcc6c734`
- Replay: `484ac97a1e851a88c29758119b4bcd191f9f21a7d23fb36be7a7d44bc02d9fd3`

Raw archive SHA-256:

- GPU: `4f1573c87550cf209f32898e63a8c677a146ccf9631d6fe6cc7b31579c902d32`
- Replay: `7af7b60e0560889295b4b63193123f047db9f99f26d701d33acccf50f415507d`

Actual UTF-8 LF sanitized trace bytes and execution source snapshots are hashed; current summary-driver hashes are separate. The two valid runs' complete `src/` manifests are identical. No model/cloud response is prerecorded and then relabeled as fresh Qwen/Gemma execution.

## Historical commands exercised

```text
.venv/Scripts/python.exe -B native/nemotron/issue218_policy_validation.py --route gpu --candidate dd3cad278dc683840c6eabaae9777159cb75a710 --output native/nemotron/issue218_policy_gpu_valid
.venv/Scripts/python.exe -B native/nemotron/issue218_policy_validation.py --route deepgram-native-replay --candidate dd3cad278dc683840c6eabaae9777159cb75a710 --output native/nemotron/issue218_policy_deepgram_final
.venv/Scripts/python.exe -B native/nemotron/issue218_policy_summary.py --trace native/nemotron/issue218_policy_gpu_valid/trace.json --trace native/nemotron/issue218_policy_deepgram_final/trace.json --output native/nemotron/issue218_policy_summary.json
```

These are historical execution commands. Their original direct trace paths, including paths embedded in saved summaries/raw traces, describe execution-time locations now represented by archive members. Do not rerun the frozen candidate's matrix merely to reconfirm the negative observations. Director-owned F1/F2 repair and any later acoustic/presentation checks need explicit changed-path evidence applicability. No broad parameter sweep, production default, VR, HMD or photon qualification is implied.

## Current archive references and offline reproduction

- GPU trace: `native/nemotron/issue218_policy_gpu_valid.zip:trace.json`
- Replay trace: `native/nemotron/issue218_policy_deepgram_final.zip:trace.json`
- Each run's counters, preflight and source snapshot are the same ZIP's `gpu_counters.jsonl`, `preflight.json` and `execution_sources.zip` members where present.
- Each renderer attempt's trace/source snapshot is `native/nemotron/<attempt-name>.zip:trace.json` / `:execution_sources.zip`; exact names and hashes are in the findings/inventory. Direct PNG links above remain unchanged.

The existing filesystem-based drivers are retained unchanged; archive members must be explicitly extracted before supplying their `--trace`, `--gpu-trace` or `--recorded-trace` paths. Their historical default input paths require equivalent extraction to those original locations. No renderer/model execution is authorized by this reproduction metadata.

Offline summary reproduction from the two archived valid traces:

```text
.venv/Scripts/python.exe -m zipfile -e native/nemotron/issue218_policy_gpu_valid.zip build/issue218_policy_archive_gpu
.venv/Scripts/python.exe -m zipfile -e native/nemotron/issue218_policy_deepgram_final.zip build/issue218_policy_archive_replay
.venv/Scripts/python.exe -B native/nemotron/issue218_policy_summary.py --trace build/issue218_policy_archive_gpu/trace.json --trace build/issue218_policy_archive_replay/trace.json --output build/issue218_policy_reproduced_summary.json
```

Extraction-path/command metadata may differ; archived measurement inputs are identical. The Director already reproduced the current summary before this artifact cleanup: **all measurement fields identical to the saved summary**, with only output-command element 6 and the updated summary-driver hash differing. This is Director-reported reused evidence, not another worker summary/model/probe run.
