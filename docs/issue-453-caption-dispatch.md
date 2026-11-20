# #453: SteamVR caption dispatch and P05 decision

[Issue #453](https://github.com/kapitalismho/PuriPuly-heart/issues/453) · [Prior findings in #347](https://github.com/kapitalismho/PuriPuly-heart/issues/347#issuecomment-6961005964)

## Implementation and decision

After a successful caption `SetOverlayTexture`, [the native caption runtime](../native/overlay/src/runtime.rs) calls `Flush` once on [the producing renderer's immediate context](../native/overlay/src/renderer/backend.rs), before handling visibility. The pre-submit GPU readiness check stays in place. Failed or preempted handoffs do not reach the added call. Flush is not a completion wait, permission to reuse a texture, or a display acknowledgment. Photo and desktop paths are unchanged.

The matched fixed-binary comparison found no additional protection from P05 in the tested cases. The implementation therefore removes periodic fresh renders and resubmissions from ordinary captions. [Python's generation start](../src/puripuly_heart/app/services/overlay/overlay_generation_start.py) no longer requests retry intent; it still refreshes the current scene and rearms original expiry deadlines on a new runtime. [The launcher](../src/puripuly_heart/core/overlay/process_runners.py) does not pass retry or experiment overrides on ordinary starts. [Native selection](../native/overlay/src/manifest.rs) defaults to `no_retry` and ignores retry-profile input unless cached rehandoff is explicitly selected. The cached rehandoff experiment remains available; readiness, cancellation, failure recovery, clear/Hide and lifetime rules remain in force. The bridge contract is unchanged.

## Build and environment

The retained comparison used baseline [`fc8b0b1`](https://github.com/kapitalismho/PuriPuly-heart/commit/fc8b0b1eb2312cc1bf56ddd5c3def7d379f1da65) and stage-one local checkpoint `5cf10a0d7097eb7545a799e88e112c13546ee9b9`, with the latter's app 2.11.2 Python source for both. The exact A executable was saved before the stage-one edit. The final default used app 2.11.3 Python and native source on synchronized local base `faf4e2317ed0063caef597fe91203aaf821ad4b9` (merging stage one with upstream [`b568179`](https://github.com/kapitalismho/PuriPuly-heart/commit/b56817921fa5eeee8a74018feee751b269c9cbe4)), plus the P05 cutover in this change. That composition includes upstream long-open SELF expiry and PEER lineage behavior. All caption binaries are locked MSVC x64 release builds, Rust 1.99.1, overlay contract 17/r2. Both live runs observed SteamVR 2.18.4, Meta Quest Pro at 72 Hz and the Windows D3D11 renderer (RX 7900 XTX, driver 32.0.31041.3013, Windows build 26300.9711).

| Lane | App | Effective configuration | Executable SHA-256 |
| --- | --- | --- | --- |
| A | 2.11.2 | baseline, P05, cached off | `5615779737ccf297246c601dc64ff163792947ae289edb6fc742352c0ac7adca` |
| B | 2.11.2 | Flush, P05, cached off | `1c2235031872f33291212b8748f631e3e71bb1e0973de7f1165234ec421d8327` |
| C | 2.11.2 | Flush, `no_retry`, cached off | same as B |
| A′ | 2.11.2 | traced A | `88d42ee6048604fc97124a3b5072a854528e143c9d51336fddaeb57d7776b88e` |
| B′ | 2.11.2 | traced B | `abdb09fe0dbd766c6c8b3b6022449c34a2dceaf06924a47855eaf009246ecd32` |
| C′ | 2.11.2 | traced C | same as B′ |
| D | 2.11.3 | exact final default, no child overrides | `eb2c9e40f3df700c5511fa37fc1b06242a03fca575f5607bd23271335a425e67` |
| D′ | 2.11.3 | traced final default, no child overrides | `287d6e454a43b2457a4a7a6f492e453a539bb6710f09762258a39f16ad2904e6` |

Shared OpenVR DLL SHA-256 `bab8ac6ef64e68a9ca53315b0014d131088584b2efdfa6db511d67ec03cfcb4a`; bundled TTC `197d5e1e019faca33a4d55931c7d68b8056f3b97cb862049f5cb8de9efdfb8ce`. Independent D3D reference executables: 2.11.2 `f929a761ff9a2c3788df8ca547342dc85c1dbf9ce414fdbc3e39b145a5224581`; 2.11.3 `64c6e9d15b5a67fc2f33f6484a73b4c9105495bbd6f489bbc433285d51666826`.

Trace builds are separate from exact binaries. Their only caption runtime additions print the resolved profile/experiment, adapter, accepted revision, QPC draw start/end, submit call start/return/dispatch boundary, success, and cached status. They do not add a GPU operation, readiness query, draw, delay, or alternate submit. Exact delta SHA-256s: A′ `21e1c706fb95de91816d3a747a7d28957f58836d43fec5311030a356d7db73ff`; B′/C′ `e1edb37d1020aa97e5f350aba61f228868d11369fe4853cc39a563736e3337e5`; D′ `3c08528eaafb3707e111666ce0b1a8ae19259d3b84bfbca1f3b3bddbc12be79e`. Readback and CPU tracing can perturb timing; no instrumented timing is used as exact-binary cost evidence.

The retained final product-source SHA-256 inventory is `final-product.sources.sha256` (SHA-256 `249c68eff8e6372b53c316fd554e9e9b3cedbaa35cf13adb1712a22593debea0`). The later test-fixture fixes did not change the product sources in D or D′.

## Destination pixels, visibility and recovery

Synthetic captions followed the ordinary Python presenter, authenticated loopback bridge and native child launcher. Only the unique test overlay and its rendering PID were inspected. Full 4096 × 1056 BGRA destination bytes were compared with independent reference renders assembled from the accepted scenes. Traced renderer and reference adapters matched. QPC measurements bracket the entire read; early samples with an intervening or intersecting draw/submit do not qualify. SteamVR visibility was queried separately. Python caption time was controlled for expiry, while real time governed authenticated health and QPC sampling.

| Lane | Qualified early whole-image matches | Early read range after handoff (ms) | Late whole-image matches | Late visibility matches | Draws / successful submits |
| --- | ---: | ---: | ---: | ---: | ---: |
| A′ | 1/11 | 23.611–42.250 | 8/12 | 12/12 | 55 / 52 |
| B′ | 12/12 | 22.995–39.823 | 12/12 | 12/12 | 55 / 52 |
| C′ | 12/12 | 22.979–40.984 | 12/12 | 12/12 | 28 / 25 |
| Exact B, stage-one default | not sampled early | — | 12/12 | 12/12 | not directly counted |
| Exact D, final default | not sampled early | — | 12/12 | 12/12 | not directly counted |
| D′, final default | 12/12 | 22.197–36.974 | 12/12 | 12/12 | 28 / 25 |

The only qualified early baseline match was a surface already blank after a clear-and-restart. One additional A′ sample (first caption, 26.036–112.940 ms) matched but crossed draw and submit 3, so it is inconclusive for first delivery. Four baseline clears retained the previous pixels after roughly 0.8 s even though the overlay was hidden. The fixed builds produced zero bytes on clear before Hide and remained transparent after Hide. Both fixed profiles matched first and final content, redisplay, and the latest accepted overlapping SELF/PEER scene. On a fresh owned child, the still-current SELF caption returned without its expired PEER; a second restart stayed clear until a fresh caption appeared. C′ had one submission per submitted revision, no cached submits and no C-only failure to reverse against B′.

All exact D and traced D′ child environments lacked both retry-profile and handoff experiment variables, no launcher overrides were supplied, and none of the wire scenes contained retry intent. D′ resolved `no_retry/off` in all three epochs. It recorded one successful submit per submitted revision and zero scheduled retry audit records. Two explicit idle windows (visible and clear) lasted 3,210.056 and 3,207.544 ms with **zero draws or submits** before the next logical caption. Three additional draws during rapid content churn did not submit. Exact D independently delivered the current image and visibility after idle and recovery. The runs restarted only their own test children/overlays; all live child exits were confirmed and test surfaces removed. No shared runtime was started, stopped or restarted for these runs. Full-image readback does not establish physical scanout or readability; no wearer observation was made.

## Matched load: work counts, not frame-time savings

A separate run segment used 12 synthetic SELF+PEER burst/quiet cycles per lane, without image reads. Each of nine lanes completed all 12 challenge-qualified final revisions (71–72 wire writes, 20.92–20.96 s of workload, 1,485–1,505 unique compositor frame rows and no frame-index gaps). Exact A/B/C/C/B/A supplied frame timing; matched trace-only lanes supplied counts. A′ and B′ each drew 120 times and submitted 108 times; C′ drew 72 times and submitted 60 times. All traced count lanes had 72 writes, no failed or cached submits. With Flush in both B′ and C′, removing ordinary retries avoided 48 draws and 48 submits in this load. That is not a game FPS or GPU-cost claim.

**Qualified frame-time comparison unavailable.** The first exact A and B and all three count-lane control windows violated the predeclared caption-empty background comparability limits. Controls required stable focus, continuous compositor indices, CPU utilization within 10 percentage points and compositor render CPU/GPU and interval p95 within the greater of 0.5 ms or 20%. First A's blank-window compositor GPU p95 moved from 1.5804 to 0.1552 ms; first B's CPU fraction moved from 0.429 to 0.280. Raw workload compositor-GPU p95 in exact A/B/C/C/B/A was 2.70844/1.97572/0.15868/0.15880/0.15652/0.15768 ms. These recorded observations are not attributable to the policy; later individually valid legs do not rescue the planned comparison. The smallest missing cost observation is a complete matched series with stable caption-empty controls. No frame-time savings are inferred from the submission counts.

## Retained checks and limitations

Before the runtime was available, background OpenVR init returned 121 and three actual A/B/C startup attempts exited 20 (`steamvr_not_running`) before authentication; child and bridge resources were released. Those attempts are not pixel evidence. One initial staging attempt failed for a missing Windows version source; another used a shared build target and reused an identical trace binary. Both were discarded; the comparison used distinct trace artifacts. An initial GPU test selector ran zero tests and a subsequent run failed with invalid UV path syntax; the corrected existing real-D3D11 fixture executed with producer reads and a substituted submitter, not SteamVR destination evidence.

The final release executable reported app 2.11.3 and overlay contract 17/r2. The affected Python suite ran 395 passing cases and two existing real-subprocess integration skips; the 14 Scribe integration cases passed again with explicit diagnostic retry intent, preserving their metadata assertions. The native integration target ran 102 cases: 100 passed and two old-default fixture assumptions failed (invalid profile must now be tested in cached mode; the scene-churn case expects scheduled P05 evidence). Both assertions were preserved by explicitly selecting their diagnostic input, and both cases passed when selected again. The production-default idle/new-caption regression and three focused native units covering cached identity, retry accounting and diagnostic records also passed. The earlier native fixture compilation error for an unmigrated profile resolver call was corrected without changing product code or result requirements.

Preparatory offline probes qualified transport health across frozen synthetic caption time and a seven-second jump, pruned an expired queued scene, and rejected incomplete workloads, focus drift and CPU drift. All live evidence above came from real SteamVR, not those substituted peers or timing inputs. The raw `trace.json`, `images.json`, synthetic pixel artifacts, `compositor.json`, `cost-comparison.json`, run IDs `live-20261121-042525-43a3ea` and `final-live-20261121-043835-6f6d4b`, matching source/hash lists, and temporary experiment scripts are retained in the task evidence kit.
