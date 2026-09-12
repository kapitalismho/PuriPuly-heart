# Issue #157 Soniox fixed-boundary measured outcome

## Outcome

Two independent five-minute AMI episodes completed against Soniox `stt-rt-v5` with every emitted profile-off baseline segment, all initial arms, and the independent C observer. The result is **negative/inconclusive for a production treatment**:

- no padding, top-up, pacing, or wait treatment improved both episodes and both text and speaker attribution;
- immediate S200 slightly worsened aggregate receipt-owned WER (`0.2102 → 0.2112`) and speaker accuracy (`79.20% → 77.67%`);
- T200's aggregate WER change was small (`0.2102 → 0.2083`) while speaker accuracy fell (`79.20% → 77.52%`);
- W200 had the best aggregate WER (`0.1974`), entirely from an IS1004a improvement; it was slightly worse on ES2002a and added about 181–192 ms maximum backlog, so this does not establish that waiting helps;
- paced and immediate 200 ms variants did not separate consistently;
- the continuous observer was better than its C primary on ES2002a text/speaker and worse on IS1004a. Almost all observer labels arrived after primary readiness. Its text was never substituted.

The episode-dependent W200 result and small arm differences are compatible with provider/session variation. A negative/inconclusive result is sufficient; no repeat, S100/S400, T100/T400, or broad sweep was run.

## Authorization, execution, spend, and provenance

The user authorized public AMI input, external Soniox processing including C, and a cumulative **US$3 cap**. Both intact episodes ran separately after commit `9d444fae6f7191fef34651a94ddb9f9e8d58a9fe`. Every plan and trace records:

- replay SHA-256 `92d41ddbfd1d7e703e46c754ba52acaf25a33d40a0609e43905d9522067a16a5`;
- profile SHA-256 `c6e120656f1bb38ae6e9d4d52b7063c7fba472ad669d05c56c7a2ebb87690b8a`;
- manifest SHA-256 `9ac061522ac1756907e5b37544ad38d51489e47e4229b5b2fc424d05a6db2d8c`.

Each episode produced eight complete streams: six treatment primaries, C primary, and C observer. ES primaries received 63 scoped finals each; IS primaries received 72 each; each observer received its terminal final. There were no missing traces or connection errors.

| Episode | Conservative admission estimate | Completed provider audio | Streams |
| --- | ---: | ---: | ---: |
| ES2002a | $0.375987 | 1,665.428 s | 8 |
| IS1004a | $0.418160 | 1,898.716 s | 8 |
| Intact total | **$0.794147** | **3,564.144 s** | 16 |

Prior conservative created-plan estimates were $1.293694, so the cumulative conservative accounting is **$2.087841**, below $3. Completed intact audio is about $0.118805 at the published $0.12/hour equivalent. This is not an invoice: Soniox bills tokens, and output/context charges are unavailable. No secret was printed or persisted; the configured local application credential was read only in process.

## Dataset, human provenance, and intact baseline

AMI Meeting Corpus manual annotations 1.6.2 are CC BY 4.0 under archive-root `LICENCE.txt`. Transcripts are AMI two/three-pass human transcription, one participant channel per speaker. Word times are forced-alignment estimates. Human turns are parsed from `segments/<meeting>.<agent>.segments.xml` in the same NXT archive and ordered by `transcriber_start`. Cross-speaker intersections are derived overlap/interruption evidence; AMI does not supply a manual overlap layer.

The annotation archive SHA-256 is `b56e5babb2496b8795deeeda7e71178d7fbc9963f94276cf2a3f4b56ebbc9f9d`.

| Meeting | Window | Full WAV SHA-256 | Normalized window SHA-256 |
| --- | --- | --- | --- |
| ES2002a | `[165,465)` s | `9c76866990fcc8b84006dc32d273ad99df439090b748ebe72103bb78c3216ee7` | `0b247f9a53edd074d9ed49dac71be2e994bfe6ec6baaaffddbb69b27137a5b69` |
| IS1004a | `[300,600)` s | `c37050e46bf3d339e896cc75baf31b191f4a3c491109d6faa44d9d55cd79666b` | `34c39320bb18da70c928727236deed1f0f9aa58fbdaf0020f18fd245540edb2a` |

The freeze replayed every 512-sample frame through the bundled Silero peer VAD, `create_peer_vad_gating`, `PeerAudioSegmentLedger`, and `ListenDeliveryController`. It retained every emitted profile-off segment without changing the 4 s step, 224 ms pause, or first frame at/after 6 s hard threshold and without attaching subsequent speech to a preceding segment.

| Meeting | All emitted | Natural `<4s` | Pause `4–6s` | Hard `>=6s` | EOF | Emitted audio/window | `<1s` turns covered |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ES2002a | 63 | 47 | 11 | 5 | 0 | 55.4% | 20/27 |
| IS1004a | 72 | 49 | 19 | 3 | 1 | 66.0% | 7/9 |

By forced-word center, real-content spans cover 90.3% of ES words and 95.3% of IS words. An additional 34/14 words fall in prefix context; 21/11 fall outside both and are reported as unscored missing coverage, mostly disfluencies or edge-stretched alignments rather than reclassified silence. Legacy audit names mean `qualifying_segments` = all emitted and `excluded_segments` = unsupported controller emissions; unsupported count is zero.

## Evaluation and P2 prefix-context repair

Soniox outputs token pieces. The evaluator concatenates their exact text within each receipt segment before word normalization, matching production text assembly. Provider timestamp overrun cannot transfer a token to the next segment or turn padding into a false benefit: the emitted piece remains receipt-owned text and its timestamp condition is counted separately.

The first intact ES evaluation found one prefix-context token piece (one word) in B0, T200, W200, T200-paced, and C; S200 and S200-paced had zero. IS had zero in every primary. The offline evaluator was repaired as required:

- prefix pieces are concatenated and counted separately;
- they are excluded from new-content WER;
- excluding a prefix run creates a word boundary, so genuine pieces on either side are never concatenated across the removed context;
- all non-prefix emitted pieces, including timestamp-overrun and synthetic-timestamp uncertainty, remain receipt-owned text.

The final evaluator SHA-256 is `8f184ae6df136de1461d7e8a1feb83b9fe9927db7572e602ac9d63babf87a91d` in both evaluations.

## Primary text and same-source speaker results

WER is total receipt-segment Levenshtein distance divided by human words centered in authoritative real-content spans. Speaker accuracy uses one fixed maximum-overlap mapping per session and source-mapped provider pieces overlapping a forced-aligned human word.

| Arm | Ref / hyp words | Ins / del / sub | Aggregate WER | ES WER | IS WER | Speaker correct / total | Speaker accuracy | Overlap correct / total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | 1018 / 1031 | 79 / 66 / 69 | 0.2102 | 0.2203 | 0.2000 | 636 / 803 | 79.20% | 64 / 75 |
| S200 | 1018 / 1034 | 78 / 62 / 75 | 0.2112 | 0.2281 | 0.1941 | 619 / 797 | 77.67% | 67 / 74 |
| T200 | 1018 / 1029 | 78 / 67 / 67 | 0.2083 | 0.2203 | 0.1960 | 631 / 814 | 77.52% | 70 / 78 |
| W200 | 1018 / 1046 | 80 / 52 / 69 | **0.1974** | 0.2222 | **0.1723** | 647 / 815 | 79.39% | 62 / 73 |
| S200 paced | 1018 / 1033 | 78 / 63 / 73 | 0.2102 | 0.2261 | 0.1941 | 619 / 793 | 78.06% | 67 / 74 |
| T200 paced | 1018 / 1029 | 77 / 66 / 70 | 0.2092 | 0.2242 | 0.1941 | 630 / 813 | 77.49% | 70 / 78 |
| C primary | 1018 / 1047 | 80 / 51 / 75 | 0.2024 | 0.2261 | 0.1782 | 554 / 815 | 67.98% | 60 / 73 |

Overlap denominators are small and inherit forced-alignment uncertainty. Higher overlap accuracy for some padded arms did not accompany higher total speaker accuracy. C primary's IS speaker mapping varied markedly from B0 despite the same treatment, direct evidence that independent provider sessions can vary.

## Boundary and actual following-segment strata

Cells are `WER (hypothesis words)`; segment/reference denominators are fixed by meeting and shown in the headers. Full per-stratum insertions, deletions, substitutions, short turns, actual next source spans, gaps, durations, and following-turn counts are retained in each `evaluation.json`.

| ES arm | Natural `n=47, ref=221` | Pause `n=11, ref=168` | Hard `n=5, ref=124` |
| --- | ---: | ---: | ---: |
| B0 | 0.2308 (243) | 0.2500 (163) | 0.1613 (122) |
| S200 | 0.2308 (243) | 0.2619 (163) | 0.1774 (119) |
| T200 | 0.2308 (243) | 0.2500 (163) | 0.1613 (122) |
| W200 | 0.2308 (243) | 0.2500 (163) | 0.1694 (121) |
| S200 paced | 0.2308 (243) | 0.2560 (162) | 0.1774 (119) |
| T200 paced | 0.2308 (244) | 0.2619 (163) | 0.1613 (122) |

| IS arm | Natural `n=49, ref=188` | Pause `n=19, ref=243` | Hard `n=3, ref=65` | EOF `n=1, ref=9` |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.2394 (201) | 0.1317 (241) | 0.2923 (53) | 0.5556 (8) |
| S200 | 0.2447 (204) | 0.1235 (242) | 0.2615 (55) | 0.5556 (8) |
| T200 | 0.2394 (199) | 0.1235 (241) | 0.2923 (53) | 0.5556 (8) |
| W200 | 0.2074 (208) | 0.1193 (244) | 0.2154 (59) | 0.5556 (8) |
| S200 paced | 0.2447 (204) | 0.1235 (242) | 0.2615 (55) | 0.5556 (8) |
| T200 paced | 0.2394 (199) | 0.1193 (240) | 0.2923 (53) | 0.5556 (8) |

Against B0, assessable current segments and their actual following emitted segments changed as follows (`improved / unchanged / regressed`):

| Episode / arm | Current segments | Actual following segments |
| --- | ---: | ---: |
| ES S200 | 3 / 54 / 5 | 3 / 53 / 5 |
| ES T200 | 0 / 62 / 0 | 0 / 61 / 0 |
| ES W200 | 0 / 61 / 1 | 0 / 60 / 1 |
| ES S200 paced | 3 / 54 / 5 | 3 / 53 / 5 |
| ES T200 paced | 0 / 61 / 1 | 0 / 60 / 1 |
| IS S200 | 4 / 63 / 2 | 4 / 62 / 2 |
| IS T200 | 2 / 66 / 1 | 2 / 65 / 1 |
| IS W200 | 11 / 57 / 1 | 11 / 56 / 1 |
| IS S200 paced | 4 / 63 / 2 | 4 / 62 / 2 |
| IS T200 paced | 3 / 65 / 1 | 3 / 64 / 1 |

Each following segment also received its own arm treatment, so following differences cannot be isolated as carryover from only the previous boundary. The near-identity of current/following counts reflects the ordered one-to-one schedule, not independent evidence.

## Short responses, returning speakers, merge/split, missing, and duplication

B0 illustrates the fixed denominators; every arm's corresponding data is in `evaluation.json`:

| Metric | ES2002a B0 | IS1004a B0 |
| --- | ---: | ---: |
| `<1s` human turns in scope | 20 | 7 |
| Short turns with centered reference words | 16 | 2 |
| Short ref / hyp words | 16 / 1 | 2 / 1 |
| Short-turn WER | 1.000 | 1.500 |
| Human A-B-A triplets | 19 | 6 |
| Assessable A-B-A | 11 | 3 |
| Same provider label on return | 7 | 2 |
| Fixed-map correct on both appearances | 4 | 2 |
| Adjacent duplicate words / within-segment pairs | 1 / 466 | 2 / 433 |

Across initial primaries, provider `unknown` speaker pieces were zero. Source-mapped but human-unalignable pieces ranged 34–42 on ES and 10–14 on IS. Every non-observer primary exposed two provider labels for four human speakers: both labels were temporal merge candidates, three of four human speakers were split candidates, and two human speakers were absent from the fixed mapping. IS C primary split all four. These are temporal association diagnostics, not manual merge/split adjudication; mixed overlap and forced timing are explicitly excluded or reported.

Receipt timestamp uncertainty was substantial with short intact segments: ES B0 had 453 pieces outside its receipt's provider-time span; padding arms additionally had pieces timestamped in synthetic audio. These pieces remained receipt-owned text and were not called cross-boundary leakage or silently dropped. Speaker scoring cannot assign a source speaker to such pieces, so the C comparison below reports them unalignable.

## C observer: primary-ready and later

C observer output is annotation-only. Primary source spans and receipt-owned primary text remain authoritative.

| Meeting | Observer pieces at primary-ready / later | Ready status `correct/incorrect/mixed/unalignable` | Last label after ready min/median/max | Observer same-span WER | C-primary WER |
| --- | ---: | ---: | ---: | ---: | ---: |
| ES2002a | 17 / 849 | 10 / 5 / 0 / 2 | 1,531 / 4,797 / 6,016 ms | **0.1969** | 0.2261 |
| IS1004a | 27 / 808 | 8 / 3 / 16 / 0 | -62 / 4,890 / 6,500 ms | 0.2277 | **0.1782** |

One IS segment's final observer label preceded primary-ready by 62 ms; the median label remained about 4.9 s later. Later same-source speaker states were:

| Meeting / stream | Correct / incorrect / mixed / unalignable | Accuracy excluding unknown/mixed/unalignable |
| --- | ---: | ---: |
| ES C primary | 274 / 113 / 45 / 491 | 70.80% |
| ES observer | 605 / 154 / 79 / 11 | 79.71% |
| IS C primary | 220 / 135 / 28 / 537 | 61.97% |
| IS observer | 432 / 298 / 63 / 15 | 59.18% |

Unknown was zero for all four. The observer's continuous source makes more pieces source-alignable than segmented primary output; accuracy denominators therefore differ. Observer text/speakers improved ES but regressed IS, and 98%/97% of its token pieces were unavailable at primary-ready. There is no demonstrated production headroom.

## Latency and backlog

Speech-end-to-primary-ready median/p95 and maximum actual backlog:

| Arm | ES ready ms | IS ready ms | ES backlog | IS backlog |
| --- | ---: | ---: | ---: | ---: |
| B0 | 1,619 / 1,662 | 1,414 / 1,470 | 246 ms | 234 ms |
| S200 | 1,506 / 1,544 | 1,520.5 / 1,552 | 287 ms | 297 ms |
| T200 | 1,562 / 1,605 | 1,553.5 / 1,596 | 433 ms | 328 ms |
| W200 | 1,750 / 1,773 | 1,789.5 / 1,818 | 427 ms | 426 ms |
| S200 paced | 1,782 / 1,814 | 1,757 / 1,794 | 474 ms | 469 ms |
| T200 paced | 1,564 / 1,616 | 1,569 / 1,615 | 442 ms | 453 ms |

ES used 59 and IS 68 human-speech-end denominators; hard boundaries without a human speech end were excluded. First speaker label arrived in effectively the same receipt as primary-ready. W200 predictably increased readiness latency and backlog. S200 latency improved ES and regressed IS. Pacing increased backlog and did not consistently improve output.

## Superseded diagnostic history

The old qualifying-only schedule retained only 16 ES and 22 IS segments, covering 27.0%/35.1% of source windows and only 9/2 short turns. It omitted natural short seals. Its first evaluator treated each Soniox token piece as a word, producing WER around 1.02–1.08 and falsely choosing S400. Exact concatenation corrected old aggregate unscoped-timestamp WER to B0 `0.2117`, W200 `0.2083`, S200 `0.2217`; S400 had no same-batch B0 and was not causal evidence. Those artifacts remain retained as superseded diagnostics only.

## Production applicability and final decision

The experiment uses direct five-minute Soniox WebSockets, endpoint detection disabled, and provider diarization enabled. Production's scoped engine rotates healthy sessions at `healthy_reset_age_s=180`; the replay intentionally does not. Production recognition retention is bounded by `STTRetentionProfile` and the LISTEN retained-segment envelope, whereas this experiment retains immutable WAVs and raw traces locally. Latency, retention, diarization, and five-minute behavior are therefore not direct production-equivalence claims.

**Decision:** do not carry a silence, top-up, pacing, waiting, or observer-substitution treatment into production from this evidence. The two intact episodes show no consistent cross-episode benefit, and the remaining budget does not justify a sweep. If a future decision specifically depends on the IS W200 anomaly, the appropriate next evidence is a separately authorized same-batch B0/W200 repeat—not S100/S400 extrapolation.

## End-to-end verification and retained artifacts

```text
uv run python experiments/soniox_fixed_boundaries/prepare_ami.py
uv run python experiments/soniox_fixed_boundaries/replay.py check
uv run python experiments/soniox_fixed_boundaries/replay.py self-check
uv run ruff check experiments/soniox_fixed_boundaries/replay.py \
  experiments/soniox_fixed_boundaries/prepare_ami.py \
  experiments/soniox_fixed_boundaries/evaluate_run.py
uv run python -m py_compile experiments/soniox_fixed_boundaries/replay.py \
  experiments/soniox_fixed_boundaries/prepare_ami.py \
  experiments/soniox_fixed_boundaries/evaluate_run.py
# passed

# Paid runs completed separately, all initial arms:
replay.py live --recordings ES2002a ... --output run_artifacts/intact-es-9d444fa
replay.py live --recordings IS1004a ... --output run_artifacts/intact-is-9d444fa
# 16 complete streams, 0 missing traces, 0 connection errors

evaluate_run.py run_artifacts/intact-es-9d444fa --output .../evaluation.json
evaluate_run.py run_artifacts/intact-is-9d444fa --output .../evaluation.json
# completed; P2 prefix separation and all required metric families present
```

Raw audio, human references, provider traces, plans, summaries, schedule audit, and detailed evaluations remain ignored under `selected_audio/`, `human_references/`, and `run_artifacts/`. Production code, Git/GitHub state, `AGENTS.md`, and secrets were not changed.
