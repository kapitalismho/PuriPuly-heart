# #204 S200-only bounded native/text/readiness diagnosis

Status: **completed**. Diagnostic version: **S200-only v1**.

## Scope and unchanged evidence

Two user-approved public-AMI calls only: brief_interjections and long_return_gap, D-minus-S200 paired against immutable original D. No provider calls are made by this diagnostic. Original A-D live_identity_evidence.json and live_identity_report.md, retained diagnose.py, public references and baseline artifacts are not rewritten.

The exact frozen lexicalization, conservative timing-independent anchors, one whole-case optimal native-label mapping, reference coverage/strata and provider-arrival exposure helpers are reused. Native token widths never provide identity support. Missing, mixed, ambiguous, unmatched and not-yet-final words stay explicit; a lower conditional error alone is not a text/quality win.

## Bounded outcomes

| Case | Arm | DER | Known anchored native error | Correct reference-word coverage | Recognized words | Sent PCM seconds | Post-input final-word median s | Sender/source-availability median s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| brief_interjections | D | 83.26% | 36.23% | 38.68% | 369 | 125.304 | 3.938 | 22.772 |
| brief_interjections | D_no_S200 | 69.81% | 22.96% | 45.71% | 353 | 121.504 | 3.641 | 20.162 |
| long_return_gap | D | 43.66% | 9.45% | 80.71% | 433 | 160.656 | 3.726 | 14.106 |
| long_return_gap | D_no_S200 | 44.03% | 9.50% | 80.27% | 432 | 156.256 | 3.585 | 14.240 |

## Quality, overlap and short-interjection strata

Coverage is correct reference words divided by all published reference words in the stratum, not just recognized matches. Conditional native error uses only known anchored words. Every residual category is retained below.

### brief_interjections

D-minus-S200 minus D: DER -13.45 points; native conditional error -13.27 points; correct reference coverage +7.03 points; recognized words -16; transmitted PCM -3.800s.

| Stratum | Arm | Reference words | Native error | Correct coverage | Explicit outcomes |
|---|---|---:|---:|---:|---|
| all | D | 455 | 36.23% | 38.68% | `{'correct': 176, 'unmatched_reference': 139, 'ambiguous_alignment': 40, 'wrong': 100}` |
| all | D_no_S200 | 455 | 22.96% | 45.71% | `{'correct': 208, 'unmatched_reference': 145, 'wrong': 62, 'ambiguous_alignment': 40}` |
| overlap | D | 238 | 32.08% | 30.25% | `{'unmatched_reference': 102, 'correct': 72, 'ambiguous_alignment': 30, 'wrong': 34}` |
| overlap | D_no_S200 | 238 | 14.42% | 37.39% | `{'unmatched_reference': 106, 'correct': 89, 'ambiguous_alignment': 28, 'wrong': 15}` |
| nonoverlap | D | 217 | 38.82% | 47.93% | `{'correct': 104, 'unmatched_reference': 37, 'wrong': 66, 'ambiguous_alignment': 10}` |
| nonoverlap | D_no_S200 | 217 | 28.31% | 54.84% | `{'correct': 119, 'unmatched_reference': 39, 'wrong': 47, 'ambiguous_alignment': 12}` |
| short_interjection | D | 17 | 33.33% | 11.76% | `{'unmatched_reference': 11, 'correct': 2, 'wrong': 1, 'ambiguous_alignment': 3}` |
| short_interjection | D_no_S200 | 17 | 0.00% | 5.88% | `{'unmatched_reference': 14, 'correct': 1, 'ambiguous_alignment': 2}` |
| turn_reference_missing | D | 0 | unavailable | unavailable | `{}` |
| turn_reference_missing | D_no_S200 | 0 | unavailable | unavailable | `{}` |

### Complete final-word arrivals and cut exposure

A word is received only when every contributing final token is received. Separate source-cut and actual-finalize-write deadlines use the unchanged retrospective whole-case mapping; there is no added fixed color wait. Post-input delay subtracts the actual write containing the last published word sample; replay ambiguities are excluded only from that decomposition. This combined provider/wire/finalization observation is not pure model processing or application publication.

| Deadline | Arm | Reference words | Correct available coverage | Explicit exposure outcomes |
|---|---|---:|---:|---|
| frozen_source_cut_time | D | 455 | 0.00% | `{'not_yet_final_word': 276, 'unmatched_reference': 139, 'ambiguous_alignment': 40}` |
| actual_finalize_write_time | D | 455 | 4.84% | `{'not_yet_final_word': 239, 'unmatched_reference': 139, 'correct': 22, 'ambiguous_alignment': 40, 'wrong': 15}` |
| frozen_source_cut_time | D_no_S200 | 455 | 0.00% | `{'not_yet_final_word': 270, 'unmatched_reference': 145, 'ambiguous_alignment': 40}` |
| actual_finalize_write_time | D_no_S200 | 455 | 3.30% | `{'not_yet_final_word': 241, 'unmatched_reference': 145, 'wrong': 14, 'correct': 15, 'ambiguous_alignment': 40}` |

Actual-wire S200-only verification:

```json
{
  "status": "passed",
  "checks": {
    "no_actual_S200_packets": true,
    "same_source_replay_packets_order_bytes_boundaries_availability": true,
    "actual_provider_positions_equal_frozen_packet_remap": true,
    "same_finalize_source_availability_terminal_sequence": true,
    "actual_control_positions_equal_frozen_remap": true,
    "same_source_timeline_and_replay": true,
    "same_source_cuts_and_reasons": true,
    "same_provider_configuration": true,
    "same_requested_and_completed_finalization_counts": true,
    "actual_audio_equals_shortened_transmitted_support": true,
    "exact_planned_S200_sample_removal": true
  },
  "removed_samples": 60800,
  "removed_seconds": 3.8,
  "original_pcm_packets": 2609,
  "retained_source_packets": 2533,
  "actual_ablation_pcm_packets": 2533,
  "retained_source_packet_signature_sha256": "b3e7f2f13d9e3c4cdca23cf96ce817f924385161d03f2568717e54dbd354c898",
  "actual_source_packet_signature_sha256": "b3e7f2f13d9e3c4cdca23cf96ce817f924385161d03f2568717e54dbd354c898",
  "limit": "S200-only byte/availability/cut isolation, not identical elapsed pacing or provider model arrival times. Removing paced padding can advance controls and successors. Raw sender does not gate intermediate ACKs."
}
```

Literal final processed-counter differences (minus-S200 minus D): `{'final_audio_proc_ms': -3960, 'total_audio_proc_ms': -3960}`. Requested-finalize difference: `0`. Different PCM provider positions/duration mechanically affect transcript native offsets; no guessed correction or clock calibration is derived.

### long_return_gap

D-minus-S200 minus D: DER +0.37 points; native conditional error +0.05 points; correct reference coverage -0.44 points; recognized words -1; transmitted PCM -4.400s.

| Stratum | Arm | Reference words | Native error | Correct coverage | Explicit outcomes |
|---|---|---:|---:|---:|---|
| all | D | 451 | 9.45% | 80.71% | `{'correct': 364, 'unmatched_reference': 46, 'wrong': 38, 'ambiguous_alignment': 3}` |
| all | D_no_S200 | 451 | 9.50% | 80.27% | `{'correct': 362, 'unmatched_reference': 47, 'wrong': 38, 'ambiguous_alignment': 4}` |
| overlap | D | 52 | 39.39% | 38.46% | `{'wrong': 13, 'unmatched_reference': 17, 'correct': 20, 'ambiguous_alignment': 2}` |
| overlap | D_no_S200 | 52 | 40.62% | 36.54% | `{'wrong': 13, 'unmatched_reference': 17, 'correct': 19, 'ambiguous_alignment': 3}` |
| nonoverlap | D | 399 | 6.78% | 86.22% | `{'correct': 344, 'unmatched_reference': 29, 'wrong': 25, 'ambiguous_alignment': 1}` |
| nonoverlap | D_no_S200 | 399 | 6.79% | 85.96% | `{'correct': 343, 'unmatched_reference': 30, 'wrong': 25, 'ambiguous_alignment': 1}` |
| short_interjection | D | 1 | unavailable | 0.00% | `{'unmatched_reference': 1}` |
| short_interjection | D_no_S200 | 1 | unavailable | 0.00% | `{'unmatched_reference': 1}` |
| turn_reference_missing | D | 0 | unavailable | unavailable | `{}` |
| turn_reference_missing | D_no_S200 | 0 | unavailable | unavailable | `{}` |

### Complete final-word arrivals and cut exposure

A word is received only when every contributing final token is received. Separate source-cut and actual-finalize-write deadlines use the unchanged retrospective whole-case mapping; there is no added fixed color wait. Post-input delay subtracts the actual write containing the last published word sample; replay ambiguities are excluded only from that decomposition. This combined provider/wire/finalization observation is not pure model processing or application publication.

| Deadline | Arm | Reference words | Correct available coverage | Explicit exposure outcomes |
|---|---|---:|---:|---|
| frozen_source_cut_time | D | 451 | 0.00% | `{'not_yet_final_word': 402, 'unmatched_reference': 46, 'ambiguous_alignment': 3}` |
| actual_finalize_write_time | D | 451 | 5.99% | `{'not_yet_final_word': 373, 'unmatched_reference': 46, 'correct': 27, 'wrong': 2, 'ambiguous_alignment': 3}` |
| frozen_source_cut_time | D_no_S200 | 451 | 0.00% | `{'not_yet_final_word': 400, 'unmatched_reference': 47, 'ambiguous_alignment': 4}` |
| actual_finalize_write_time | D_no_S200 | 451 | 4.21% | `{'not_yet_final_word': 381, 'unmatched_reference': 47, 'correct': 19, 'ambiguous_alignment': 4}` |

Actual-wire S200-only verification:

```json
{
  "status": "passed",
  "checks": {
    "no_actual_S200_packets": true,
    "same_source_replay_packets_order_bytes_boundaries_availability": true,
    "actual_provider_positions_equal_frozen_packet_remap": true,
    "same_finalize_source_availability_terminal_sequence": true,
    "actual_control_positions_equal_frozen_remap": true,
    "same_source_timeline_and_replay": true,
    "same_source_cuts_and_reasons": true,
    "same_provider_configuration": true,
    "same_requested_and_completed_finalization_counts": true,
    "actual_audio_equals_shortened_transmitted_support": true,
    "exact_planned_S200_sample_removal": true
  },
  "removed_samples": 70400,
  "removed_seconds": 4.4,
  "original_pcm_packets": 3346,
  "retained_source_packets": 3258,
  "actual_ablation_pcm_packets": 3258,
  "retained_source_packet_signature_sha256": "71e02676db3c073330585c80cfc6b6e04f2c0fa968af2ee4e841ea0b77ba6ba3",
  "actual_source_packet_signature_sha256": "71e02676db3c073330585c80cfc6b6e04f2c0fa968af2ee4e841ea0b77ba6ba3",
  "limit": "S200-only byte/availability/cut isolation, not identical elapsed pacing or provider model arrival times. Removing paced padding can advance controls and successors. Raw sender does not gate intermediate ACKs."
}
```

Literal final processed-counter differences (minus-S200 minus D): `{'final_audio_proc_ms': -4080, 'total_audio_proc_ms': -4080}`. Requested-finalize difference: `0`. Different PCM provider positions/duration mechanically affect transcript native offsets; no guessed correction or clock calibration is derived.

## Returning-speaker witness

Original published turns and a single whole-case native-label mapping are retained. Full word/token/receipt witnesses are in the JSON.

**D**, reference speaker `ES2004b.D`: all-reference outcomes `{'wrong': 38, 'unmatched_reference': 13, 'ambiguous_alignment': 1}`, anchored native IDs `{'2': 23, '1': 15}`, global mapping `{'1': 'ES2004b.B', '2': 'ES2004b.C'}`.
- `ES2004b.sync.722`: `{'wrong': 23, 'unmatched_reference': 2}`; native IDs `{'2': 23}`.
- `ES2004b.sync.724`: `{'unmatched_reference': 11, 'wrong': 15, 'ambiguous_alignment': 1}`; native IDs `{'1': 15}`.

**D_no_S200**, reference speaker `ES2004b.D`: all-reference outcomes `{'wrong': 38, 'unmatched_reference': 13, 'ambiguous_alignment': 1}`, anchored native IDs `{'2': 23, '1': 15}`, global mapping `{'1': 'ES2004b.B', '2': 'ES2004b.C'}`.
- `ES2004b.sync.722`: `{'wrong': 23, 'unmatched_reference': 2}`; native IDs `{'2': 23}`.
- `ES2004b.sync.724`: `{'unmatched_reference': 11, 'wrong': 15, 'ambiguous_alignment': 1}`; native IDs `{'1': 15}`.

## Supported recommendation and limits

- **brief_interjections**: S200 removal improves both conditional native attribution and all-reference correct coverage. Correct all-reference coverage changes +7.03 points; short-interjection correct coverage changes -5.88 points. Single-source-occurrence post-input complete-final-word median is 3.938s → 3.641s. These are bounded final-text/provider-readiness observations, not caption readiness.
- **brief_interjections, readiness tradeoff:** correctly attributed complete-final words already available at actual finalize writes cover 4.84% → 3.30% of the same reference windows. Controls can advance after S200 removal, so a lower post-input median does not imply better availability at those seals. These remain provider-arrival exposures, not consumer publication.
- **brief_interjections, short-word/text caution:** correct short-interjection words fall 2 → 1 of 17; conditional native error 33.33% → 0.00% must not be sold as a short-interjection win. Recognized words change -16; all-reference unmatched words change +6. Correct attribution coverage is not a complete ASR/WER score.
- **long_return_gap**: S200 removal has a mixed or non-improving native-attribution/reference-coverage result. Correct all-reference coverage changes -0.44 points; short-interjection correct coverage changes +0.00 points. Single-source-occurrence post-input complete-final-word median is 3.726s → 3.585s. These are bounded final-text/provider-readiness observations, not caption readiness.
- **long_return_gap, readiness tradeoff:** correctly attributed complete-final words already available at actual finalize writes cover 5.99% → 4.21% of the same reference windows. Controls can advance after S200 removal, so a lower post-input median does not imply better availability at those seals. These remain provider-arrival exposures, not consumer publication.
- **Returning-speaker outcome:** wrong anchored-word counts are 38 → 38. Published initial/return turn outcomes and native-ID/global-mapping witnesses remain explicit above; S200 removal does not erase the observed native returning-speaker failure merely by changing timestamp support.
- **Decision supported by these two runs:** do not treat S200-off as a blanket quality/text/readiness improvement; the case-specific and stratum outcomes above are the supported result.
- S200-only isolation is verified against actual original/new packet hashes, boundaries, source availability, replay, source cuts, provider configuration and controls. Removing paced S200 necessarily also permits earlier controls/input; this is the intended bounded intervention, not a wall-clock-matched pure acoustic estimate.
- Single new runs against earlier immutable D results have no repeat/significance or independent-holdout claim. Actual processed counters and native timing remain literal; neither changing PCM duration nor transcript matching calibrates the provider clock. No fitted timestamp correction is justified.
- **No production cutover/readiness certification:** this raw sender independently receives but ignores intermediate ACK gates. It does not exercise production socket reuse, pending-finalization ownership, canonical colors known at publication, acknowledged consumer attachment, or the first-readable caption. Neither disabling S200 in production nor adding a fixed color wait follows from these recordings alone. No further provider calls were made by this diagnostic.

## Receipts and reproducibility

Execution UTC: `2026-09-30T17:05:04.081030+00:00`.

```json
{
  "original_live_evidence": {
    "path": "experiments/ami_benchmark/providers/soniox/gap/live_identity_evidence.json",
    "sha256": "4672ed43819f858a8d6f0a447b0abd395001c550d89ef68b65478a8163f79608",
    "bytes": 31763396
  },
  "ablation_plan": {
    "path": "experiments/ami_benchmark/providers/soniox/gap/factorial/ablation/run_plan.json",
    "sha256": "a454485e811757e413706a86bdbfc1487ffced8377ee02be847008ebcf5f29e8",
    "bytes": 9752249
  },
  "ablation_configuration": {
    "path": "experiments/ami_benchmark/providers/soniox/gap/factorial/ablation/config.json",
    "sha256": "c7254b19241967c7c4184c569eaaa3e920e5d671a00e6a6bdc809216b4ac7eb7",
    "bytes": 2366
  },
  "ablation_runner": {
    "path": "experiments/ami_benchmark/providers/soniox/gap/factorial/ablation/frozen/ablation_sender.py",
    "sha256": "047dfbba7c0633e9cd8bcada589299f94a4b71d538f8319bc04159f0c08a441f",
    "bytes": 11879
  },
  "shared_sender": {
    "path": "experiments/ami_benchmark/providers/soniox/gap/factorial/ablation/frozen/shared_sender.py",
    "sha256": "e5a654687b1d48e099b3a0da25064fbb255c027e7e42d756d11d76ec8567ce6c",
    "bytes": 33212
  },
  "scores": {
    "path": "experiments/ami_benchmark/providers/soniox/gap/factorial/ablation/scores.json",
    "sha256": "3d72bc36d1dd736c626b6a67456756d0465a50c78ab3bc23d1a189efba213b65",
    "bytes": 1009
  },
  "ablation_quota": {
    "path": "experiments/ami_benchmark/providers/soniox/gap/factorial/ablation/quota_ledger.json",
    "sha256": "8a144b7d484131411e476f84ce1f155a19a21ede2b9fe4f8a12277636fe12de4",
    "bytes": 695
  }
}
```

Frozen helper SHA-256: `fa6b2393ebebd9db60a4661be78cbeae9e71c520987a93b97337f04b96fc01b7`. Frozen live-helper SHA-256: `bff019b88be22a6e4adadef799440c58d07ca3fba694ae8abef2ba7110810522`.

Input result/events/responses/SDK/RTTM receipts, all per-word alignment outcomes, published-turn arrivals, every cut exposure, actual counters, literal native times and source-write associations are in ablation_identity_evidence.json. Public AMI annotations are independently published references, not newly listener-calibrated acoustic anchors. These two exposed clips are not holdout validation or independent participant sampling.
