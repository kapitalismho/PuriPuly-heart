# Issue #157 Soniox fixed-boundary experiment: corrected pre-execution report

## Status and decision barrier

**READY FOR DIRECTOR COMMIT; DO NOT START A PAID RUN FROM THE CURRENT UNCOMMITTED TREE.**

The preparation, replay, and evaluator have been repaired offline. The next evidence must be generated only after the Director commits the stable executable, then regenerates each plan so its Git revision and replay/profile/manifest hashes identify that commit. No paid provider call was made during this repair.

The earlier qualifying-only runs are retained as **superseded diagnostics**. They do not answer the experiment because they omitted 47 of 63 emitted ES2002a baseline segments and 50 of 72 emitted IS1004a baseline segments (including natural short speech-end seals), and S100/S400 lacked same-batch B0 controls. Their original report also tokenized every Soniox token piece as a word, which inverted the text ranking. No production recommendation survives those results.

## Corrected frozen inputs and schedule

AMI Meeting Corpus manual annotations 1.6.2 are CC BY 4.0. The controlling license is archive-root `LICENCE.txt`. Transcripts are AMI two/three-pass human transcription, one participant channel per speaker. Word boundaries are forced-alignment estimates. Human turn spans are parsed directly from `segments/<meeting>.<agent>.segments.xml` in the same NXT archive and ordered by `transcriber_start`; the reference provenance embeds the annotation archive SHA-256. Cross-speaker word/turn intersections are derived overlap/interruption evidence because the release has no manual overlap layer.

| Meeting | Window | Full WAV SHA-256 | Normalized window SHA-256 |
| --- | --- | --- | --- |
| ES2002a | `[165,465)` s | `9c76866990fcc8b84006dc32d273ad99df439090b748ebe72103bb78c3216ee7` | `0b247f9a53edd074d9ed49dac71be2e994bfe6ec6baaaffddbb69b27137a5b69` |
| IS1004a | `[300,600)` s | `c37050e46bf3d339e896cc75baf31b191f4a3c491109d6faa44d9d55cd79666b` | `34c39320bb18da70c928727236deed1f0f9aa58fbdaf0020f18fd245540edb2a` |

The annotation archive SHA-256 is `b56e5babb2496b8795deeeda7e71178d7fbc9963f94276cf2a3f4b56ebbc9f9d`.

`prepare_ami.py` replays every 512-sample source frame through the bundled Silero peer VAD, `create_peer_vad_gating`, `PeerAudioSegmentLedger`, and `ListenDeliveryController`. The freeze now retains **every emitted profile-off baseline segment**, without changing the 4 s step, 224 ms pause, or first frame at/after 6 s hard threshold and without appending subsequent speech to a preceding segment.

| Meeting | All emitted | Natural hangover `<4s` | Pause `4–6s` | Hard `>=6s` | EOF | Emitted audio/window | Human turns `<1s` covered |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ES2002a | 63 | 47 | 11 | 5 | 0 | 55.4% | 20/27 |
| IS1004a | 72 | 49 | 19 | 3 | 1 | 66.0% | 7/9 |

The former qualifying-only freeze covered only 27.0% and 35.1% of the two windows and only 9 and 2 in-scope short turns. That 65–73% source omission made short-response and following-segment attribution incomplete. The corrected schedule has no fabricated gaps and no filtered emitted short segment. Source intervals between emitted segments remain genuine VAD-non-speech, not replayed audio.

Required available coverage is present: A-B-A, A-B-C, brief response, laughter, silence, natural/pause/hard boundaries, continuation beyond six seconds, overlap, interruption proxy, and same-speaker continuation. `voice_chat_codec_noise` and `similar_voices` were decided pre-execution as optional where available; they are absent, disclosed, and not blockers.

## Corrected evaluator contract

Soniox tokens are pieces. Production constructs text by concatenating their exact `text` fields before word normalization (`src/puripuly_heart/providers/stt/soniox.py`). The repaired evaluator does the same. It never treats each token piece as an independent word.

Primary output ownership is receipt-scoped:

1. Every final non-`<fin>` token piece is owned by the segment whose scoped receipt carried it.
2. Text is concatenated within that receipt segment, then compared with human words centered in that segment's authoritative real-content span.
3. Provider timestamp overshoot cannot move a token into the next segment, synthetic padding, or a dropped bucket. A piece outside its receipt's provider span remains emitted segment text and is counted separately as `timestamp_outside_receipt_scope`; pieces timestamped in synthetic audio are counted as synthetic-timestamp uncertainty. Neither condition creates a fake padding benefit.
4. Prefix context is source-mapped but excluded from new-content text scoring.
5. An explicitly warned unscoped-provider-timestamp diagnostic is retained only to reproduce the old analysis; it must never drive per-segment ownership.

For each episode and arm, the evaluator emits:

- receipt-owned WER with reference/hypothesis words and insertions, deletions, substitutions;
- every segment and aggregate natural-hangover, pause, hard, and EOF stratum;
- the actual next emitted segment's source span, gap, duration, human-turn count, and short-turn count;
- fixed-map speaker accuracy with overlap/sequential denominators;
- unknown, mixed, unalignable, and missing fixed-map speaker coverage;
- provider-label merge candidates and reference-speaker split candidates with explicit label/speaker denominators;
- A-B-A returning-speaker preservation/correctness;
- `<1s` human-turn WER and its turn/reference denominators;
- adjacent duplicate hypothesis words and word-pair denominator;
- receipt gate waits, source-to-ready timing, backlog, and unmapped reason counts.

Merge/split figures are temporal association diagnostics: a provider label associated with multiple forced-aligned human speakers is a merge candidate, and a human speaker associated with multiple provider labels is a split candidate. They are not manual adjudication. Forced timing and overlaps can make them mixed or unalignable, which is why those categories remain explicit.

## C observer alignment

C uses independent primary and observer requests, sessions, and speaker namespaces. Observer output is annotation-only and is never substituted for primary text.

For every primary segment, the evaluator uses the authoritative primary real-content source span and primary `<fin>` gate-release time. It reports observer final token pieces and speaker states available at that instant, the later complete state, partial text diagnostics, and last-label delay. Labels are classified `correct`, `incorrect`, `unknown`, `mixed`, or `unalignable` under a fixed observer-session mapping. Observer text is compared only on the same primary source spans.

The superseded traces validate this alignment path:

| Meeting | Labels available at primary-ready | Later token pieces | Last label after primary-ready min/median/max | Observer same-span WER | Primary receipt-owned same-span WER |
| --- | ---: | ---: | ---: | ---: | ---: |
| ES2002a | 16/481 | 481 | 3,687 / 4,695 / 6,047 ms | 0.1918 | 0.2260 |
| IS1004a | 11/466 | 466 | 3,297 / 5,023.5 / 5,968 ms | 0.1721 | 0.1494 |

The observer text was better on the old ES span and worse on the old IS span under authoritative receipt-owned primary scoring, contrary to the superseded blanket claim that it had no text headroom. Nearly all observer labels arrived after primary readiness and observer diarization was weaker/mixed. This is diagnostic validation, not evidence that observer text may replace primary output.

## Superseded diagnostics: corrected interpretation

The original broken per-piece metric reported aggregate WER around 1.02–1.08 and falsely selected S400. Exact text concatenation on the retained old traces gives the following unscoped-timestamp diagnostic across the two initial episodes:

| Arm | Corrected diagnostic WER |
| --- | ---: |
| B0 | 0.2117 |
| S200 | 0.2217 |
| T200 | 0.2150 |
| W200 | **0.2083** |
| S200 paced | 0.2217 |
| T200 paced | 0.2183 |
| C primary | 0.2100 |

Thus B0/W200, not S400, were best on corrected old text. Receipt-owned emitted-text scoring also reports timestamp uncertainty rather than cross-boundary “leakage”; the old ES counts include 37 timestamp overruns for B0 versus 12 overruns plus 11 synthetic-timestamp pieces for S200, so lower mapped counts cannot be presented as a treatment benefit.

Old B0 receipt-owned boundary diagnostics demonstrate the rebuilt denominators but remain invalid for the intact-schedule outcome:

| Meeting / stratum | Segments | Ref / hyp words | WER | Ins / del / sub | Short turns in segment | Actual following segments / following short turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ES pause | 11 | 168 / 158 | 0.2679 | 10 / 20 / 15 | 5 | 10 / 5 |
| ES hard | 5 | 124 / 119 | 0.1774 | 4 / 9 / 9 | 4 | 5 / 4 |
| IS pause | 19 | 243 / 239 | 0.1481 | 9 / 13 / 14 | 2 | 18 / 2 |
| IS hard | 3 | 65 / 61 | 0.1538 | 1 / 5 / 4 | 0 | 3 / 0 |

On the same old B0 scope, A-B-A diagnostics were ES `12` human triplets, `12` assessable, `6` same-label returns, `4` fixed-map-correct on both appearances; IS `2/2/2/1`. Short-turn scoring had ES 9 turns but only 3 with word references (3/3 deletions) and IS 2 turns with no centered reference word. Adjacent duplicate counts were ES `1/261` and IS `8/278` within-segment word pairs. These denominators expose why the old schedule cannot support a general short-response conclusion.

S100/S400 results are not causal evidence: they ran later, under different concurrency, without same-batch B0, after the now-invalid S200 interpretation. Both conditional arms are closed again. There is no valid conclusion that padding, top-up, pacing, or waiting helps until the intact schedule is run.

## Production applicability and retention

This experiment uses direct Soniox WebSockets, five-minute primary session lifetimes, endpoint detection disabled, and provider diarization enabled. Production's scoped engine has `healthy_reset_age_s=180` and performs healthy rotation; this replay intentionally does not. Production recognition retention is bounded by `STTRetentionProfile` and the LISTEN retained-segment envelope (`src/puripuly_heart/app/wiring/wiring_local_asr_provider_runtime.py` and `src/puripuly_heart/core/audio/listen_delivery.py`). The experiment retains immutable WAVs and raw traces locally for analysis. Therefore experiment latency, memory/retention behavior, and diarization applicability are not direct production equivalence claims.

The user authorized public AMI input, external Soniox processing including C, useful bounded follow-ups, and a cumulative **US$3 cap**. The local application credential is read in process without printing or persistence. F10/F11 review verified budget and secret handling.

Conservative admission estimates for all prior created plans total about **$1.293694**. The corrected two-episode initial-arm execution preview is **$0.794147**, making the conservative cumulative preview about **$2.087841**, before unknown output/context token billing. That leaves about $0.912159 under the cap, but it is not pre-authorized for waste: only a same-arm repeat or S100/S400 may be considered after intact evidence shows it is useful. T100/T400 remain closed.

## Economical execution after commit

Run the two independent episodes separately to avoid the failed 16-session concurrency pattern. Each episode includes all initial arms: B0, S200, T200, W200, S200-paced, T200-paced, and C primary plus observer.

```text
# After Director commit only; regenerate plans after the commit.
uv run python experiments/soniox_fixed_boundaries/replay.py plan --recordings ES2002a --arms all --output <ES-plan>
uv run python experiments/soniox_fixed_boundaries/replay.py live --recordings ES2002a --arms all --credential-source local-app --authorize-paid-run I_APPROVE_SONIOX_PAID_RUN --output <ES-run>
uv run python experiments/soniox_fixed_boundaries/evaluate_run.py <ES-run-dir> --output <ES-evaluation>

uv run python experiments/soniox_fixed_boundaries/replay.py plan --recordings IS1004a --arms all --output <IS-plan>
uv run python experiments/soniox_fixed_boundaries/replay.py live --recordings IS1004a --arms all --credential-source local-app --authorize-paid-run I_APPROVE_SONIOX_PAID_RUN --output <IS-run>
uv run python experiments/soniox_fixed_boundaries/evaluate_run.py <IS-run-dir> --output <IS-evaluation>
```

The second AMI episode supplies independent meeting/speaker variation. Do not open a repeat or S100/S400 in advance. Decide only after the two complete initial episodes, require same-batch controls for any follow-up, and keep the total conservative admission accounting below $3.

## Focused offline verification

The repaired flow is verified by:

```text
uv run python experiments/soniox_fixed_boundaries/prepare_ami.py
# prepared ES2002a and IS1004a; genuine words + NXT turns regenerated

uv run python experiments/soniox_fixed_boundaries/replay.py check
# ready; two recordings; all required coverage; S100/S400/T100/T400 closed
# corrected initial estimate $0.794147

uv run python experiments/soniox_fixed_boundaries/replay.py self-check
# passed schedule/accounting/gate/identity/budget guard checks

uv run python experiments/soniox_fixed_boundaries/evaluate_run.py <retained-run> --output <repaired-evaluation>
# retained ES, IS, and targeted diagnostics rebuilt successfully

uv run ruff check experiments/soniox_fixed_boundaries/replay.py \
  experiments/soniox_fixed_boundaries/prepare_ami.py \
  experiments/soniox_fixed_boundaries/evaluate_run.py
uv run python -m py_compile experiments/soniox_fixed_boundaries/replay.py \
  experiments/soniox_fixed_boundaries/prepare_ami.py \
  experiments/soniox_fixed_boundaries/evaluate_run.py
# passed
```

Raw audio, human references, provider traces, schedule audit, plan previews, and repaired detailed evaluations remain ignored under `selected_audio/`, `human_references/`, and `run_artifacts/`. Production code, Git/GitHub state, `AGENTS.md`, and secrets are outside this repair and were not changed.
