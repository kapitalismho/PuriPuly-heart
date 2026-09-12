# Issue #157 Soniox fixed-boundary experiment preparation

## Status

**Preparation complete; every live comparison is blocked and no provider result is claimed.**

No appropriately licensed or consented recording, human-checked speaker reference, external-processing approval, or explicit paid API budget (including the second stream in C) is available in this worktree. The checked-in manifest therefore contains no recording and keeps every approval field false or null. No audio was uploaded and no paid API was invoked.

The durable deliverables are:

- `profile.json`: frozen source-boundary, Soniox, arm, pacing, lifetime, and pricing snapshot;
- `manifest.json`: blocked input/approval manifest and the exact fields needed for selected recordings;
- `replay.py`: offline validation/accounting, plan generation, live access guards, and a direct Soniox WebSocket replay/trace path;
- `.gitignore`: prevents selected audio, human references, raw provider traces, and transcripts from entering Git.

This is an isolated experiment surface. It does not import or modify production runtime code.

## Baseline and accessible evidence

- Authorized baseline and current checkout: `9b95293b2041760401d82d80da4daf07c30f6fb9`.
- Local change known before this work: `AGENTS.md`; it was not edited.
- The repository had no `experiments/` tree at this baseline. No old experimental branch or #51 artifact was treated as an implementation base.
- The only in-tree Soniox integration smoke (`tests/integration/test_soniox_stt_integration.py`) streams silence to check connectivity. It is not diarization, accuracy, five-minute-lifetime, or human-reference evidence.
- Issue #51 is contextual evidence only. Its later reported local-detector investigations do not establish a Soniox result and are not inherited here.
- Issue #66 confirms that production Soniox silence injection and the undocumented `trailing_silence_ms` finalize field were removed. This experiment sends synthetic padding as PCM and keeps the control message exactly `{"type":"finalize"}`.

## Production-shaped contracts preserved by the profile

At the baseline:

- `ListenDeliveryController` owns the LISTEN policy constants: four-second step age, 224 ms pause seal, and six-second hard seal.
- `PeerAudioSegmentLedger` separates content ranges, prefix/context ranges, failed ranges, and synthetic-context counts. It assigns capture epoch, source order, segment identity, frozen settings, seal, and terminal ownership.
- VAD `SpeechStart.pre_roll` is recorded as context separately from content. The replay manifest therefore requires explicit `prefix_spans`; the plan counts them as duplicated, real source-mapped input rather than hiding them as new content.
- The production Soniox adapter uses mono PCM16LE at 16 kHz, `stt-rt-v5`, endpoint detection disabled, repeated manual finalize messages, an empty frame for graceful stream end, and keepalives on idle connections. Its current configuration does not enable diarization; this probe freezes `enable_speaker_diarization=true` only inside the experiment.
- Production currently sends source/context metadata to the scoped adapter boundary but Soniox receives PCM frames. The experiment preserves the metadata in its plan and raw trace rather than changing production.
- Production `ScopedRecognitionEngine._handle_end` sends the seal and then awaits the scoped provider terminal before finishing the turn. Its frozen final timeout is 20 seconds. The experiment now mirrors that FIFO terminal gate; it does not use an idealized fire-and-continue baseline.
- No five-minute cutoff was found in the Soniox adapter. Five minutes is the user-reported behavior and issue authority, so the experiment profile explicitly holds every primary comparison stream open to at least 300 seconds. It does not infer or propose a production session-lifetime change.

## Current Soniox contract snapshot

Official documentation read on 2026-09-12 states:

- manual finalize is `{"type":"finalize"}`; `<fin>` is the completion marker; finalize may repeat and audio may continue in the same connection;
- manual finalization can significantly reduce diarization accuracy, and Soniox recommends approximately 200 ms of transmitted post-speech silence before finalizing;
- realtime speaker attribution can switch temporarily before stabilizing; each token may contain `speaker`, timestamps, confidence, and finality;
- responses expose `final_audio_proc_ms` and `total_audio_proc_ms` processing frontiers;
- current WebSocket documentation says a stream supports up to 300 minutes of audio. This newer provider maximum does not replace the issue-authorized five-minute comparison profile;
- current pricing is token based, with a published realtime equivalent of about $0.12/hour, $2.00 per million input-audio tokens, $4.00 per million input-text tokens, and $4.00 per million output-text tokens. Diarization is listed as included.

Provider documentation supports protocol preparation, not an accuracy or latency outcome. Actual API acceptance, response timing, speaker behavior, connection lifetime, and cost remain unmeasured until an approved run.

## Replay and accounting design

Each selected WAV must be uncompressed mono 16 kHz PCM16 and must expose one exact 4,800,000-sample session span. Its byte SHA-256, human-reference file/hash/checker/date, consent or license basis, coverage tags, and frozen source-ordered segments are mandatory. Current acoustic/VAD observations arrive in 512-sample (32 ms) frames.

Each segment records:

- source content `[start, end)` in normalized 16 kHz samples;
- a required `prefix_spans` field containing zero or more real, source-mapped spans (use `[]` explicitly when no prefix exists);
- capture/session epoch through the containing recording session;
- pause (224 ms after the four-second step age) or exact six-second hard boundary;
- already-transmitted contiguous VAD-classified trailing silence and its limitation note: a pause seal must record exactly 3,584 samples (seven frames), while a hard cut must record fewer than 3,584 samples or the pause policy would have sealed first;
- an optional human/acoustic speech-end source sample.

Validation uses the policy annotation and frame coordinate, not a scan for zero-valued PCM. Zero samples are not proof of acoustic silence, and nonzero samples are not proof of speech.

The plan assigns every PCM transmission a provider `[start, end)` sample range and one of:

- `real_content`, with an exact source span;
- `prefix_context`, with an exact source span and a duplicated-input count;
- `synthetic_silence`, with no source span;
- `continuous_observation`, with an exact source span in C.

Synthetic samples advance provider time only. They never advance or rewrite source coordinates. For example, S200 adds 3,200 provider samples per cut; 50 cuts add 160,000 samples, or ten provider-audio seconds, and zero source seconds. T200 uses `max(0, 3,200 - transmitted_trailing_silence_samples)` for each cut. Pause and hard cuts remain distinguishable.

The source-availability clock progresses independently of the sender clock. After each primary finalize send, transmission of the next segment is gated on the preceding scoped `<fin>` receipt for up to the production-shaped 20-second timeout. Source availability continues during that wait; immutable WAV source samples are not dropped. A provider error, connection end, or timeout fails the gate and aborts the stream before any following segment can be transmitted. Provider receipts are attributed to the still-active scoped segment, the active attribution is cleared at `<fin>`, and only then can the following segment become active. This prevents final text or speaker attribution from being assigned across the fixed boundary.

Static plans label their send schedule and backlog as a lower bound that excludes the unmeasured terminal receipt wait; they are not claimed as an idealized B0 result. Live traces record each gate's release/failure and wait, and every sent PCM frame records actual backlog from its independent source-availability time. Thus W200, realtime-paced padding, provider processing, and the scoped terminal wait all remain visible in next-turn delay. Send start/finish, finalize send, provider receipt, processing frontier, token text/time/speaker/finality, connection error, request ID, and session lifetime are written to local JSONL. The report consumer must distinguish source seal, finalize send, final transcript receipt, and speaker-label availability.

This file replay conserves pending audio on disk. It does **not** reproduce the production bounded-retention envelope (eight wholly unsent segments plus the active recognition and open source segments), memory pressure, or queue failure behavior. No new dropping policy is introduced. A live result must disclose this retention difference and cannot claim that the replay proves production backpressure behavior.

C starts two concurrent, independent WebSocket sessions for each recording:

1. an authoritative B0 primary stream with the fixed boundaries;
2. a continuously paced observer with no padding and no intermediate finalize, finalized only at recording end.

The streams retain independent request/session and speaker namespaces. `replay.py` deliberately does not align observer tokens or substitute observer text. Human-reviewed source/text alignment, including mixed/unknown/unalignable spans and primary-ready versus later availability, remains required analysis after execution.

## Arm disposition

| Arm | Current disposition | Reason |
| --- | --- | --- |
| B0 | Blocked, unmeasured | No selected/approved audio, references, or budget |
| S200 | Blocked, unmeasured | Same; PCM acceptance/timing not assumed |
| T200 | Blocked, unmeasured | Same; actual transmitted trailing-silence annotations absent |
| W200 | Blocked, unmeasured | Same; waiting is a control, not assumed latency matched |
| S200-paced | Blocked, unmeasured | Same; scheduling comparison prepared |
| T200-paced | Blocked, unmeasured | Same; scheduling comparison prepared |
| C primary + observer | Blocked, unmeasured | Same; observer upload and spend explicitly unapproved |
| S100 / S400 | Conditional, not opened | Open only if S200 has a useful measured effect |
| T100 / T400 | Conditional, not opened | Open only if T200 has a useful measured effect |

Conditional arms have complete configurations but `opened=false`. If a measured 200 ms result could change the decision, record that decision by changing only the applicable arm(s) to `opened=true`, then name them explicitly with `--arms`. A negative initial result is a reason not to open them. This prevents a silent broad sweep.

No result supports decisions about whether padding helps, top-up is sufficient, waiting explains an effect, or C adds useful headroom. Those decision fields are intentionally **unmeasured**.

## Commands

Run from the repository root with the locked project environment:

```text
uv run python experiments/soniox_fixed_boundaries/replay.py check
uv run python experiments/soniox_fixed_boundaries/replay.py self-check
uv run python experiments/soniox_fixed_boundaries/replay.py plan --arms all --output experiments/soniox_fixed_boundaries/run_artifacts/plan.json
```

`check` is expected to return a structured blocked disposition while the checked-in manifest remains empty. `plan` requires real selected inputs and does not contact Soniox.

A live run remains inaccessible unless all manifest input/reference checks pass, all required coverage is present, recording approval includes C, the explicit budget covers the selected arm estimate, `SONIOX_API_KEY` exists, and the caller supplies the separate command guard:

```text
uv run python experiments/soniox_fixed_boundaries/replay.py live --arms all --output experiments/soniox_fixed_boundaries/run_artifacts/<new-run-id> --authorize-paid-run ISSUE-157
```

`--arms C` opens both streams concurrently. Raw artifacts can contain private transcript text and therefore stay under the ignored local `run_artifacts/` directory. The script never writes the API key to its plan or trace.

## Offline verification completed

Environment recorded for the reachable smoke:

- Python 3.12.10 through `uv run`;
- Windows 11 build 22631;
- `websockets` 16.1.1;
- code revision `9b95293b2041760401d82d80da4daf07c30f6fb9`.

Commands and results:

```text
uv run python experiments/soniox_fixed_boundaries/replay.py self-check
# passed: five-minute WAV/lifetime; fixed pause/hard coordinates; rejection
# of pause tails 0/1600, hard tail 8000, and missing prefix_spans;
# source/provider epochs and B0/S200/T200/W200/pacing/C accounting;
# source progression with next-segment transmission gated on scoped <fin>;
# terminal failure/timeout safety; lower-bound and actual-backlog design;
# nested plan output creation; duplicate-arm rejection; observer conservation;
# approval/budget guard. Accuracy claim: none.

uv run python experiments/soniox_fixed_boundaries/replay.py check
# status: blocked; all 13 required coverage tags and every approval/input
# prerequisite are reported; all initial and conditional arm dispositions emitted.

uv run python experiments/soniox_fixed_boundaries/replay.py check --arms B0,B0
# status: invalid; duplicate arm ids rejected before planning or spend.

uv run ruff check experiments/soniox_fixed_boundaries/replay.py
# All checks passed.
```

The self-check uses generated zero PCM only to test accounting and guards. It is not mock accuracy, acoustic, provider, or reference evidence.

## Required owner choices and remaining blockers

Before any live arm can run, the owner must supply or select:

1. A small frozen set of recordings with explicit authorization for Soniox external processing. Each must provide a continuous five-minute session span and collectively cover every manifest coverage tag. Private recordings may stay local.
2. Human-checked speaker/reference annotations with durable local file hashes, checker evidence, sequential/overlap/mixed/unknown regions, and actual speech ends where latency from speech end is evaluated. Prior branch labels are not silently accepted.
3. Frozen source boundaries produced by the current 4 s / 224 ms / 6 s policy, including each segment's following segment, actual transmitted trailing silence, prefix spans, capture epoch, and VAD limitation note.
4. An explicit paid API budget that includes all initial primary streams, the second C stream, text/context token charges, and an allowed retry margin.
5. A post-S200/T200 decision on whether any 100/400 arm is opened.

For one recording with `K` cuts, the prepared initial run's conservative realtime-equivalent authorization estimate is:

```text
2,420 seconds                              # 40 stream-minutes + C observer final gate
+ 140.6 * K seconds                        # seven primary 20 s gates + S200 twice + W200
+ 2 * sum(T200 top-up seconds per cut)     # immediate and paced T200
```

The 20-second terms reserve the configured worst case for every scoped terminal gate because the script does not implement a separate mid-run billing cutoff. At zero transmitted trailing silence, the expression is `2,420 + 141*K` seconds. With 50 cuts, the published $0.12/hour equivalent is approximately **$0.3157 per recording** before context/output tokens or retries. Multiply by the number of selected recordings. This conservative streaming-duration estimate is still not a calculable dollar maximum because output/context token counts, provider tokenization, failures, and authorized retries are not yet known. The owner must choose a larger explicit cap rather than treating $0.3157 as sufficient authorization.

After execution, a human-grounded evaluator/report pass must calculate one fixed speaker-to-reference mapping per session, concrete source-referenced failures, transcript loss/duplication/substitution/leakage, current and following-segment effects, sequential versus overlap metrics, primary-ready and later observer availability, connection failures, actual usage, and uncertainty/repeats. Until then, the experiment supports preparation only and no production recommendation.
