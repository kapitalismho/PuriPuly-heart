# #441 Soniox input-selection diagnosis handoff

## Owner decision and scope

**Analysis/comparison only; preserve production unchanged.** The owner selected this outcome and requested an English handoff. The acoustic-state preservation candidate below is **unimplemented and unapproved**. This diagnosis does not adopt Scribe turn prediction, retune thresholds, or change SmartTurn, provider, padding, caption, or rollover policy.

The observed problem is **source PCM omitted from S1 input**, not an established Soniox server-side ASR/diarization error. Native S0 assigned the Japanese 30.08 s anchor correctly; S1's completed source ledger did not cover it. No new cloud or quality runs were made to reconfirm that omission.

## Pins and reproducible input

Relevant production owners were unchanged between the [planning/product pin `9f7474b94b885a7a9a4777b43a3a05026e3e248e`](https://github.com/kapitalismho/PuriPuly-heart/commit/9f7474b94b885a7a9a4777b43a3a05026e3e248e), actual S1 capture pin `ff7f0f04319cfd63809732440a3db960aeceb2c3`, and diagnostic checkout `63c8be10c174390cb6c450bc8b9cf0292517f5ac`. Read-only source comparisons established that equivalence. Upstream `a5933e7b6c62d57a97fe4c05262b81067279e222` was not merged. The probe imported this worktree using the shared `../../../.venv/Scripts/python.exe`.

Before finalization, upstream `cb47ada43dcd8c9deda22ee447ca6bf9637dd566` was fetched and inspected, but not merged. The complete latest runtime is **not equivalent** to the frozen product: `ae941bde3` changes recognition metadata logging, `4a795b669` changes final-timeout recovery/provider-epoch retirement, and `3e22f9421` changes recognition-wait/retention timing. The default/Self wait becomes 12 s, while production Soniox Peer watchdog wiring remains 5 s; this is not a universal 12 s Soniox timeout.

A direct source comparison from the diagnostic checkout to that upstream pin found the relevant `ListenDeliveryController`, VAD/Silero reset, audio VAD loop, Soniox provider and Peer channel paths unchanged. The affected original turn completed normally and did not exercise timeout recovery. The newer changes therefore do not overturn the pinned diagnosis, but no latest-upstream ASR execution or quality benefit was measured. Revalidate any separately approved repair against the then-current timeout/recovery and provider-epoch lifecycle.

The maintained [frozen manifest](issue441-inputs.json) records these inputs; audio is not redistributed:

| Item | Identity |
| --- | --- |
| Public source | [IngCrowd Japanese themed dialogue](https://huggingface.co/datasets/IngCrowd/japanese-themed-dialogue), revision `e2eed8769400269f20821deb0e47e68b40012e8b`; IngCrowd Inc.; CC-BY-NC-4.0, noncommercial evaluation |
| Original audio | `japanese-themed-dialogue01_audio.wav`; SHA-256 `ea1138f2b05e6b3cc9580ed1196748ed67c5847f222c1cf4a54de6bf4d9d3282` |
| Annotation | `japanese-themed-dialogue01.eaf`, same revision; SHA-256 `c59daa71d0ffa34e50b4c532d33f5c0d2a51ceaa8aea4435a68a87d924f61b32` |
| Selected clip | Original offset **86.04 s**, duration **120 s**; continuous original mixed mono audio, 16 kHz PCM16, including gaps and overlap |
| Clip WAV SHA-256 | `8cdbe6868df6e141f80a14ce7e49b501cd371e185e2899a2b1a659dc22d06a2b` |
| Frozen manifest SHA-256 | `dc9b9e08ddc55c2f1e501de007ceaf8d7c1b24cd529c9c5e0c2f0dcc81acf294` |
| Local VAD model | Production `src/puripuly_heart/data/vad/silero_vad.onnx`, CPU ONNX inference; SHA-256 `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3` |

The frozen `ja-anchor-10` is at clip time 30.08 s in annotated span `ja-a28`, speaker `speaker_1`. Its independent continuous-model proxy was VAD 0.922957, RMS −25.97 dBFS. Selection used corpus annotation and automatic signal checks, not new listening; neither proxy establishes lexical or speaker truth on its own.

## Original timeline and responsible owners

1. Segment 5 owned source **24.096–29.984 s**. At controller event 173, source frontier **479744 samples / 29.984 s**, the observed wall-clock segment age was **5.9044 s**, pause **192 ms**, with no pending SmartTurn decision. This selects the **age ≥5 s / 192 ms cloud opportunity branch**, not ordinary 512/800 ms SmartTurn termination or the 10 s cap.
2. Completed audio send 891 covered **29.952–29.984 s**. Send 892 added **200 ms generated zero padding**; send 893 sent `finalize`. Padding has no source correspondence and cannot cover the anchor. Native `<fin>` arrived at 30.299977 s; the final receipt was admitted at 30.310974 s. Segment 5 had zero failed ledger samples.
3. The next content onset was **31.232 s**, committed after three frames at **31.328 s**. Context send 894 covered **30.732–31.232 s**; content send 895 began **31.232–31.264 s**. The 500 ms pre-roll was therefore too late to recover 30.08 s. Segment 6 also recorded zero failed samples. The original run used one Soniox session and recorded no execution errors.

Production ownership at the product pin:

| Owner | Relevant behavior |
| --- | --- |
| [`ListenDeliveryController`](../../src/puripuly_heart/core/audio/listen_delivery.py), `observe_acoustic_chunk` lines 215–218; `_seal_locked` 434–459 | Selects the age ≥5 s / 192 ms pause seal and calls `vad.seal_active`, without rollover. |
| [`VadGating`](../../src/puripuly_heart/core/vad/gating.py), `seal_active` 486–497; `reset` 216–233 | Resets both logical selection state and the acoustic engine; clears ring and pending candidate state. |
| [`SileroVadOnnx.reset`](../../src/puripuly_heart/core/vad/silero.py), lines 46–50 | Resets recurrent tensors, the 64-sample acoustic context, and sample-rate/batch bookkeeping. |
| `VadGating`, lines 317–323, 371–445, 561–586 | New onset requires three consecutive qualifying 32 ms frames; onset threshold 0.4, continuation approximately 0.3, 500 ms pre-roll. |
| [`run_audio_vad_loop`](../../src/puripuly_heart/core/runtime/audio_vad_loop.py), lines 390–488 | Processes and dispatches each frame before the acoustic controller can seal it. |
| [`ScopedRecognitionEngine`](../../src/puripuly_heart/core/stt/scoped_engine.py), lines 1194–1224, 1352–1427 | Converts and writes only admitted pre-roll/content, retaining source correspondence. |
| [`_SonioxSession`](../../src/puripuly_heart/providers/stt/soniox.py), lines 1228–1290, 477–516 | Queues admitted PCM, writes it unchanged, and separately queues padding/finalize. |

## Bounded offline causal comparison

The production-reset track reproduced the original selected source ledger through 32 s. A second track changed **exactly one operation**: at the target 29.984 s seal, it suppressed the **whole `SileroVadOnnx.reset` call**, preserving recurrent tensors **and** acoustic context/rate/batch state. Logical segment, pending-candidate and ring resets still occurred; earlier pause seals retained production behavior. This was not a recurrent-tensors-only intervention or a change to every pause boundary.

| Source frame interval, seconds | Production acoustic reset | Whole acoustic state preserved at target only |
| --- | ---: | ---: |
| [29.984, 30.016) | 0.084291 | 0.727811 |
| [30.016, 30.048) | 0.381173 | 0.981659 |
| [30.048, 30.080) | 0.197179 | 0.929641 |
| **[30.080, 30.112)** | **0.053745** | **0.790226** |

The 0.053745 value belongs to the frame **starting at 30.080 s**, not the frame ending there. The independent uninterrupted-model probability for this same frame was 0.922957, reproducing the manifest proxy; it is not the reset production model's probability.

Threshold **settings** were unchanged. The counterfactual's first three frames satisfied the onset threshold 0.4 and opened a new segment at frontier 30.080 s, owning source from 29.984 s. Subsequently it normally used continuation threshold ≈0.3; production remained outside speech at onset threshold 0.4. At the anchor, production had no pending onset candidate—not merely a candidate rejected for being shorter than three frames.

The counterfactual admitted source **30.080–30.112 s**, samples **[481280, 481792)**, as one 512-sample content payload. Its production-converted PCM SHA-256 was `b5dd2025424f08b1b82ce5e0c6b3d0b6e650cdee3a8c62b6d88a66732ffe179f` at the replaced local wire.

**Evidence boundaries:**

- All **910/910 original source signatures** through 32 s matched the production-reset track: payload kind, source ranges and sample count. This is not original wire-hash equality.
- The completed tracks admitted 910 production-reset and 948 counterfactual payloads. All **1,858 offline payloads** matched the exact mapped-source production float32→PCM conversion and completed the real native queue/writer unchanged. Queues were empty, writers settled, and synthetic input terminals completed without failure.
- Original per-frame probabilities and wire PCM hashes were not retained. New probabilities are controlled offline observations, not retroactively measured original probabilities. Transport completion is not Soniox server acceptance.
- The result establishes a reachable production input-selection bottleneck caused by acoustic cold start, rather than supporting a send-ledger observer defect or dropping of admitted PCM in the native queue. It does **not** establish improved server recognition, general benefit, or acceptable false-positive tradeoffs.

## Reproduction procedure without cloud access

The following records the procedure independently of ignored local scripts:

1. Obtain the revision-pinned public audio/annotation above, reconstruct the continuous 120 s clip at offset 86.04 s, and verify clip/model/manifest hashes. Use only the first **32 s** for this diagnostic; no private capture or application bootstrap is needed.
2. Feed 512-sample frames from clip sample zero into the real `SileroVadOnnx` and `create_peer_vad_gating`. Give every frame exact source/normalized `AudioCaptureSpan` ranges. Use source sample frontier /16000 as the monotonic clock; disable wall timers.
3. Use real `PeerAudioSegmentLedger`, `ListenDeliveryController`, `ScopedRecognitionEngine` and Soniox queue/writer. Settings are 16 kHz, onset 0.4, continuation ≈0.3, three-frame onset commitment, 900 ms configured hangover and 500 ms pre-roll. The original delivery snapshot was on/ready, threshold 0.75. Do not infer SmartTurn: replay earlier seals, then use the real target opportunity branch. Controlled post-target counterfactual SmartTurn requests report unavailable explicitly.
4. After processing the corresponding frames, replay the first four original non-rollover `delivery_pause` seals at these source frontiers:

   | Samples | Clip seconds |
   | ---: | ---: |
   | 110080 | 6.880 |
   | 188416 | 11.776 |
   | 262144 | 16.384 |
   | 384000 | 24.000 |

   From source time 29 s, call the real `observe_acoustic_chunk` for each frame. It selects the target seal at **479744 / 29.984 s**, with 192 ms pause. Source-clock age is **5.888 s**, versus the original wall age **5.9044 s**; both select the same opportunity branch. Do not conflate these ages.
5. Replace transport with an in-memory recording wire; do not start network transport or load credentials/settings. Use synthetic authoritative empty `<fin>` receipts to settle input, explicitly forwarding `event.text_authority` to the ledger. This controls recognition completion, not ASR quality. Deny remote network access after the local Windows asyncio self-pipe setup.
6. Execute the production-reset track and the otherwise identical track suppressing the whole acoustic reset at **only frontier 479744**. End input at 32 s and settle/close owned resources. Record per-frame probabilities, candidate/admission state, content/context source ranges, PCM hashes and completed native writes. An independent uninterrupted Silero pass explains the frozen signal proxy only; it is not another provider/quality arm.

## Retained attempts and local evidence

Three failed probe attempts remain failures, not hidden passes:

1. An overbroad network guard blocked Windows asyncio's local self-pipe before PCM ingest or inference.
2. A probe terminal sink omitted `event.text_authority`; the unchanged ownership guard correctly rejected non-authoritative empty completion. **197 admitted payloads** had already completed byte-identically, with no network attempts and settled cleanup.
3. The production track completed 32 s and the counterfactual recovered the target, then hit the controlled SmartTurn stub's missing `submit`. It was corrected to return unavailable explicitly; no SmartTurn inference or production guard weakening occurred.

The final corrected execution passed the declared source-ledger, PCM, queue, boundary, terminal, cleanup and no-network checks. Earlier partial evidence is not added to its denominators. No cloud calls, private audio/settings/secrets/discovery/telemetry/OSC, MT/GUI checks, model-quality reruns or score changes occurred.

These links are **local-only, ignored artifacts, not published evidence URLs** and may not resolve in a clean checkout. The provenance, procedure and decisive observations needed to understand this result are embedded above:

- [Derived diagnosis](../../.tmp/issue441/soniox-input-diagnosis/diagnosis.json), [completed probe result](../../.tmp/issue441/soniox-input-diagnosis/probe-result.json), [probe script](../../.tmp/issue441/soniox-input-diagnosis/probe.py), [original evidence extraction](../../.tmp/issue441/soniox-input-diagnosis/original-evidence.json).
- Failed records: [startup](../../.tmp/issue441/soniox-input-diagnosis/startup-failure.json), [terminal authority](../../.tmp/issue441/soniox-input-diagnosis/terminal-authority-failure.json), [counterfactual SmartTurn stub](../../.tmp/issue441/soniox-input-diagnosis/counterfactual-smartturn-failure.json). Corresponding consoles are `probe-console.log`, `probe-console-authorized.log`, `probe-console-corrected.log`; the completed console is `probe-console-completed.log`, in the same local directory.
- Original capture: `.tmp/issue441/selected-runs/S1/ja.json` and its retained native/send/receipt records. It was not modified or recaptured.

## Future contract question—not an implemented fix

If separately approved, investigate preserving the **whole acoustic state across logical `delivery_pause` seals only for a continuous, compatible Peer source**. Keep logical segment/candidate/ring boundaries distinct. Retain full acoustic reset for actual source end, discontinuity and explicit reset. This is not a change to hard-rollover semantics and is not justified as a universal reset removal.

Required regressions before any repair is accepted:

- Positive resumed speech still opens a distinct segment through the unchanged three-frame gate; fewer than three qualifying frames remain rejected.
- Silence/noise/false-onset cases measure the risk of preserved acoustic history; recovery of this single interval is insufficient.
- Preserve settings application and onset/continuation threshold-state transitions; do not silently carry incompatible authority/settings state.
- Exact PCM/source correspondence, once-only content ownership, bounded pre-roll and no lost/duplicated admitted ranges across seals.
- Hard-rollover overlap/context and continuity behavior remain unchanged.
- EOF/source end, discontinuity and explicit reset still clear the full acoustic state and settle owned work.
- Preserve current timeout recovery/provider-epoch retirement behavior; a pause-only preservation change must not carry acoustic state across incompatible capture/source-reset boundaries.

No broad threshold, SmartTurn, provider or boundary-policy retuning is proposed. **Current production remains unchanged.**
