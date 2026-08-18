# Annotation-only calibration

This report uses accepted natural source annotations only. No model prediction, model score, official model result, or model training participated.

Contract: `psem-handoff-v0` (`frozen_after_annotation_only_calibration`)

## Decision

All provisional constants are retained and the operational contract is frozen without a version bump.

| Constant | Retained value (ms) |
|---|---:|
| `reliable_solo_min_duration` | 200 |
| `annotation_boundary_jitter` | 50 |
| `gap_topology_min_duration` | 100 |
| `overlap_topology_min_duration` | 100 |
| `local_continuity_max_gap` | 1200 |
| `short_backchannel_min_duration` | 200 |
| `short_backchannel_max_duration` | 1000 |

## Candidate set

| Scope | Sessions | Scored hours | Solo intervals | Internal gaps | Overlap intervals |
|---|---:|---:|---:|---:|---:|
| All | 28 | 15.886353 | 13747 | 5248 | 13887 |
| AMI | 20 | 11.694456 | 9413 | 3553 | 9206 |
| AliMeeting | 8 | 4.191897 | 4334 | 1695 | 4681 |

## Distribution quantiles

| Scope | Distribution | p01 ms | p05 ms | p10 ms | p50 ms | p90 ms | p95 ms | p99 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| All | `solo_segment_duration` | 25.0 | 110.0 | 238.0 | 1495.0 | 6765.8 | 9648.0 | 18630.72 |
| All | `silence_gap_duration` | 16.0 | 78.35 | 150.0 | 768.0 | 2667.1 | 4308.95 | 9469.2 |
| All | `overlap_duration` | 12.0 | 50.0 | 101.0 | 520.0 | 1570.4 | 2060.0 | 3385.56 |
| All | `intervening_speaker_duration` | 208.62 | 291.45 | 340.0 | 1089.0 | 6005.2 | 8320.1 | 15307.68 |
| AMI | `solo_segment_duration` | 25.0 | 116.0 | 245.0 | 1480.0 | 7113.2 | 10412.6 | 20852.44 |
| AMI | `silence_gap_duration` | 16.0 | 80.0 | 150.0 | 1008.0 | 3433.4 | 5582.8 | 11284.44 |
| AMI | `overlap_duration` | 12.0 | 50.0 | 99.0 | 560.0 | 1637.0 | 2202.25 | 3658.45 |
| AMI | `intervening_speaker_duration` | 281.42 | 323.4 | 384.2 | 1097.0 | 5241.6 | 7494.4 | 18750.44 |
| AliMeeting | `solo_segment_duration` | 30.0 | 100.0 | 220.0 | 1520.0 | 6257.0 | 8347.0 | 12883.4 |
| AliMeeting | `silence_gap_duration` | 10.0 | 70.0 | 160.0 | 500.0 | 1196.0 | 1563.0 | 2250.0 |
| AliMeeting | `overlap_duration` | 18.0 | 50.0 | 110.0 | 440.0 | 1440.0 | 1880.0 | 2852.0 |
| AliMeeting | `intervening_speaker_duration` | 200.0 | 254.0 | 280.0 | 980.0 | 6372.0 | 8402.0 | 11387.2 |

## Threshold audit

- Solo bins: `{"fragment_below_200ms": 1136, "reliable_at_or_above_200ms": 12611}`
- Silence-gap bins: `{"continuity_unknown_above_1200ms": 1640, "jitter_at_or_below_50ms": 168, "micro_above_50ms_below_100ms": 181, "official_100ms_through_1200ms": 3259}`
- Overlap bins: `{"jitter_at_or_below_50ms": 699, "micro_above_50ms_below_100ms": 633, "official_at_or_above_100ms": 12555}`
- Intervening-speaker bins: `{"above_1000ms": 173, "below_200ms": 1, "short_backchannel_200ms_through_1000ms": 158}`
- Micro-gap fraction: `0.03448933`
- Micro-overlap fraction: `0.0455822`
- Ambiguous sample fraction: `0.0`
- Unknown-identity sample fraction: `0.0`
- Masked transition fraction: `0.31654904`
- Masked transition reasons: `{"complex_overlap_transition": 1512, "continuity_unknown": 1450, "mixed_unresolved_transition": 1030}`
- Diagnostic masked region counts: `{"complex_overlap_region": 3057, "complex_overlap_transition": 5428, "overlap_to_silence_unresolved": 2}`

## Per-corpus annotation granularity

| Corpus | GCD quantum (ms) | Minimum positive step (ms) | 1 ms aligned | 10 ms aligned | 50 ms aligned |
|---|---:|---:|---:|---:|---:|
| AMI | 1.0 | 1.0 | 1.0 | 0.13542136 | 0.02807571 |
| AliMeeting | 10.0 | 10.0 | 1.0 | 1.0 | 0.19341295 |

## Rationale

- The 50 ms reconciliation tolerance remains above observed per-corpus annotation granularity and micro-gap/micro-overlap rates are limited rather than dominant.
- The 200 ms reliable-solo threshold removes short annotation fragments while retaining the large majority of known singleton intervals in both corpora.
- The 100 ms topology minima leave substantial natural gap and overlap coverage in both corpora rather than eliminating nearly all short events.
- Masked transitions and diagnostic masked regions are explicitly attributable to v0 complex-overlap, continuity-unknown, mixed-unresolved, or overlap-to-silence rules rather than hidden ambiguous coercion.
- The 1200 ms continuity maximum is the hard inherited #76 value and was not eligible for calibration change.
