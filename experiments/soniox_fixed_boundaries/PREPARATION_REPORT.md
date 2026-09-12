# Issue #157 Soniox fixed-boundary measured outcome

## Outcome

The authorized experiment completed against Soniox `stt-rt-v5` with endpoint detection disabled and diarization enabled. The evidence supports a narrow conclusion:

- adding transmitted PCM silence changed speaker attribution and boundary leakage, so the effect is real enough to reject “waiting alone explains it”;
- no single treatment won every metric or both episodes;
- immediate 400 ms had the best aggregate text result, immediate 200 ms reduced late/cross-boundary final tokens most, and paced 200 ms had the best aggregate speaker accuracy;
- waiting 200 ms without audio did not help and increased backlog;
- the continuous C observer was later and materially worse for speaker attribution, so its text was never substituted for the primary result.

This is experiment evidence, not a production-change recommendation. The direct five-minute sessions, immutable-WAV retention, fixed qualifying-boundary schedule, and Soniox diarization setting differ from current production in the ways documented below.

## Authorization, spend, and credentials

The user authorized public AMI recordings, external Soniox processing including C, targeted useful repeats, and a cumulative **US$3 cap**. The configured local application keyring credential was read in process without printing or writing it. No secret appears in plans, traces, summaries, or this report.

The completed initial evidence and targeted S100/S400 plans had conservative admission estimates of $0.339527 and $0.091300. Counting the full admission estimate of every created live plan—including the failed teardown run and two early cancelled/connection-limited attempts—totals **$1.293694**, comfortably below $3 and deliberately overstates what was transmitted. Completed trace frontiers reported 2,030.64 initial plus 422.76 targeted provider-audio seconds, about **$0.0818** at the published $0.12/hour equivalent. That is not an invoice: Soniox bills tokens, and failed-attempt/output/context token charges were not available.

## Dataset and immutable inputs

AMI Meeting Corpus manual annotations 1.6.2 are CC BY 4.0. The controlling license is the archive-root `LICENCE.txt`, not the stale directory-level license page. Transcription provenance is AMI's two/three-pass human transcription, one participant channel per speaker; word times are forced alignments of that human transcript and remain estimates. No new listening pass or invented human check is claimed. Cross-speaker word/turn intersections are derived overlap/interruption evidence because the public manual release has no manual overlap layer.

Two independent Mix-Headset windows were used:

| Meeting | Meeting window | Normalized span | Full WAV SHA-256 | Window WAV SHA-256 |
| --- | --- | --- | --- | --- |
| ES2002a | `[165,465)` s | 4,800,000 mono 16 kHz PCM16 samples | `9c76866990fcc8b84006dc32d273ad99df439090b748ebe72103bb78c3216ee7` | `0b247f9a53edd074d9ed49dac71be2e994bfe6ec6baaaffddbb69b27137a5b69` |
| IS1004a | `[300,600)` s | 4,800,000 mono 16 kHz PCM16 samples | `c37050e46bf3d339e896cc75baf31b191f4a3c491109d6faa44d9d55cd79666b` | `34c39320bb18da70c928727236deed1f0f9aa58fbdaf0020f18fd245540edb2a` |

The annotation ZIP SHA-256 is `b56e5babb2496b8795deeeda7e71178d7fbc9963f94276cf2a3f4b56ebbc9f9d`. Normalized audio, canonical references, and raw transcripts remain in ignored local directories.

Coverage present across the two windows: A-B-A, A-B-C, brief response, laughter, silence, qualifying pause and hard boundaries, continuation beyond six seconds, overlap, interruption proxy, and same-speaker continuation. The schedule contained **38 qualifying boundaries**: ES2002a 11 pause + 5 hard; IS1004a 19 pause + 3 hard. `voice_chat_codec_noise` and `similar_voices` were absent and are reported as optional gaps, not fabricated labels or execution blockers.

## Source-derived boundary schedule

`prepare_ami.py` cut the exact windows, parsed AMI's per-speaker forced-aligned words, and replayed every normalized 512-sample frame through the repository's bundled Silero peer VAD, `create_peer_vad_gating`, `PeerAudioSegmentLedger`, and `ListenDeliveryController`. Manifest content ranges, prefix/context spans, seal reasons, and trailing VAD-classified samples come from those emitted ownership records—not hand-authored intervals. Coverage tags were cross-checked against the researcher's independent NXT turn parsing; the current preparation script also parses those turn spans when regenerating references.

The controlled schedule retains only issue-authorized qualifying boundaries: pause seals from four seconds up to the hard threshold with at least 3,584 aligned trailing-silence samples, and hard seals on the first 512-sample frame at or after six seconds with at most 3,584 trailing samples. Earlier default-OFF 500 ms hangover seals and smart-turn pre-four-second paths are audited locally as exclusions. This is not a claim that production emits only the retained schedule.

At the six-second threshold the observed source duration is frame-aligned (for example 96,256 samples), not an invented exact 96,000-sample endpoint. A hard/pause trailing-silence tie at 3,584 is accepted according to the controller-emitted seal reason.

## Protocol and causal controls

Every primary stream used exact `{"type":"finalize"}` controls. The next segment was withheld until the preceding scoped final `<fin>` arrived, up to the production-shaped 20-second timeout. Source availability continued during that wait; actual send backlog includes it. Provider tokens were attributed to the pending segment until `<fin>`, then the next segment became active. Synthetic padding advanced provider time only; source coordinates were never shifted.

Arms were B0, S200 immediate, T200 immediate top-up, W200 wait-only, S200 paced, T200 paced, and C (B0 primary plus independent continuous observer). S100/S400 were opened only after S200 showed decision-relevant but episode-dependent changes. T100/T400 stayed closed; there was no broad sweep.

One full ES2002a run produced every scoped final receipt but the first replay version incorrectly waited for the server to close after the production-style empty stop frame; all eight traces therefore ended with a local teardown timeout after complete evidence. The replay was fixed to close locally after all scoped finals. A 16-session attempt was stopped after one connection closed normally under excess concurrency, and an early sequential repeat was cancelled to avoid needless spend. The completed IS1004a and targeted runs had zero stream errors. These attempts are retained locally and included in the conservative cap accounting, but only receipt-complete ES2002a plus completed IS1004a and targeted traces supply metrics.

## Aggregate primary results

Word error is Levenshtein distance divided by human reference words whose centers fall inside the planned real-content spans. Prefix-context tokens are counted separately rather than scored as new content. Speaker accuracy uses one fixed maximum-overlap mapping per provider session; it is conditional on source-mapped tokens that overlap a forced-aligned human word. The two B0 streams scored 600 reference words, 905 hypothesis words, and 955 speaker-attributed tokens. High absolute WER reflects substantial insertions/substitutions under the selected fragmented schedule; relative treatment comparisons are more informative than the absolute score.

| Arm | Aggregate WER | Fixed-map speaker accuracy | Sequential / overlap accuracy | Cross-boundary receipt tokens | Max actual backlog | Mean episode p95 token availability |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | 1.0767 | 87.43% | 88.06% / 79.10% (`n=888/67`) | 37 | 255 ms | 5,720.5 ms |
| S100 immediate | 1.0750 | 87.87% | 88.61% / 78.57% (`n=878/70`) | 29 | 286 ms | 5,560.0 ms |
| S200 immediate | 1.0633 | 88.52% | 89.39% / 78.38% (`n=858/74`) | **25** | 318 ms | 5,686.0 ms |
| S400 immediate | **1.0217** | 88.45% | 89.13% / 80.56% (`n=837/72`) | 31 | 364 ms | **5,481.5 ms** |
| T200 immediate | 1.0700 | 88.27% | not separately decisive | **25** | 303 ms | 5,766.0 ms |
| W200 wait-only | 1.0783 | 87.27% | not separately decisive | 37 | **418 ms** | 5,807.5 ms |
| S200 paced | 1.0633 | **90.74%** | **91.60%** / 80.28% (`n=869/71`) | 26 | 396 ms | 5,695.5 ms |
| T200 paced | 1.0717 | 88.16% | not separately decisive | **25** | 402 ms | 5,702.0 ms |

Immediate S200 versus B0 improved aggregate WER by 0.0134, fixed-map speaker accuracy by 1.09 percentage points, and reduced cross-boundary receipt tokens 37→25. The effect varied: ES2002a speaker accuracy improved 88.1%→92.7% and leakage 20→6, while IS1004a speaker accuracy fell 86.9%→84.6% and leakage rose 17→19 even as WER improved 1.068→1.045. This episode dependence motivated the targeted sizes.

S400 gave the best aggregate WER but was not uniformly safer: IS1004a leakage rose to 27 and speaker accuracy was 84.5%. S100 was the lowest-backlog targeted size and reduced aggregate leakage to 29, but did not materially change WER. Paced S200's speaker result was strongest, but its maximum backlog was 141 ms above B0. W200 matched B0 leakage, slightly worsened WER/speaker accuracy, and added 163 ms maximum backlog. Audio transmission—not elapsed waiting alone—is therefore the plausible treatment carrier, but the trade-off is not monotonic.

## Boundary, following-segment, and latency evidence

Per-segment metrics and source spans are preserved in ignored `evaluation.json` artifacts. Concrete examples show both gains and regressions:

- ES2002a S200 regressed current `segment-0051` `[3,687,424,3,751,936)` by +0.286 WER with no measured change in following `segment-0053`.
- ES2002a S200 improved `segment-0059` `[4,441,088,4,537,344)` by -0.154 and its following `segment-0060` by -0.545.
- IS1004a S200 regressed `segment-0014` `[1,063,424,1,159,680)` by +0.278 with no following-segment change, while `segment-0032` `[2,399,232,2,465,280)` improved by -0.154 and following `segment-0034` by -0.200.
- IS1004a S400 regressed `segment-0066` `[4,445,696,4,516,864)` by +0.154 despite improving the episode aggregate.

For the 38 initial boundaries, median/p95 scoped `<fin>` gate wait was 265/297 ms for B0, 328/359 ms S200, 250/344 ms T200, 235/250 ms W200, 234/266 ms paced S200, and 234/266 ms paced T200. Median/p95 speech-end-to-primary-ready was 1,337/1,425 ms B0; 1,366/1,547 ms S200; 1,322/1,413 ms T200; 1,481/1,638 ms W200; 1,467/1,631 ms paced S200; and 1,215/1,378 ms paced T200. First speaker-label availability was effectively the same receipt as primary readiness for scored pause boundaries. Hard boundaries without a human speech end are excluded from speech-end latency denominators.

Targeted S100/S400 ran later with only four concurrent streams, so their absolute latency must not be compared causally to the more concurrent initial batches. Within that targeted batch, median/p95 speech-end-to-ready was 1,150/1,272 ms S100 versus 1,252/1,394 ms S400.

## C observer and alignment coverage

C retained independent request/session/speaker namespaces. No observer token replaced primary text.

| Meeting | Stream | Reference / hypothesis words | Fixed-map speaker accuracy | Token availability p50 / p95 / max |
| --- | --- | ---: | ---: | ---: |
| ES2002a | C primary | selected scope | 88.1% | 3,428 / 5,515 / 6,413 ms |
| ES2002a | C observer | 568 / 903 | 78.6% | 6,539 / 7,419 / 7,572 ms |
| IS1004a | C primary | selected scope | 86.5% | 3,037 / 5,725 / 6,793 ms |
| IS1004a | C observer | 530 / 816 | 57.8% | 6,488 / 7,534 / 7,780 ms |

The observer covered the continuous five-minute source (965 and 841 source-mapped final tokens), but its final-token availability was roughly three seconds later at the median and speaker accuracy was worse. It supplied no useful production headroom in these episodes.

Soniox exposed only three provider speaker labels for four ES2002a participants and two labels for four IS1004a participants in the inspected primary mappings. Reported speaker accuracy is therefore conditional coverage, not proof that all human speakers were recovered. Mixed/unknown and tokens with no forced-aligned word were excluded with numerator/denominator counts retained. Overlap scores are based on cross-speaker forced-word intersections and inherit alignment uncertainty.

## Reproducibility and retained artifacts

Each plan, raw session trace, and successful run summary records actual Git `HEAD` plus SHA-256 of the executed replay, profile, and manifest. Successful IS1004a ran at Git `1d82cad2cc48e739f687fd4829793258c4dd0f81` with replay hash `82a4b130d076f5267d1672ea6100a7df2c52c74ef5cf4a10284f1c4188542e70`; targeted S100/S400 additionally records profile hash `14a1e4e92435d3815bd27dc190876da380f92d863d14c1b6a65f65b3e57d8058`. Evaluation output records its own evaluator hash.

Focused verification:

```text
uv run python experiments/soniox_fixed_boundaries/prepare_ami.py
uv run python experiments/soniox_fixed_boundaries/replay.py check
# ready; 2 recordings; 11 required tags present; 2 optional gaps reported
uv run python experiments/soniox_fixed_boundaries/replay.py self-check
# passed offline accounting, boundary, gate, identity, guard, and coverage checks
uv run ruff check experiments/soniox_fixed_boundaries/replay.py \
  experiments/soniox_fixed_boundaries/prepare_ami.py \
  experiments/soniox_fixed_boundaries/evaluate_run.py
# passed
```

Raw audio, canonical human words, provider transcripts, schedule audit, plans, summaries, and detailed evaluations remain under ignored `selected_audio/`, `human_references/`, and `run_artifacts/`. Only the reproducible scripts, frozen profile/manifest metadata, and this aggregate report are durable. Production code, `AGENTS.md`, Git, GitHub, and secrets were not changed.
