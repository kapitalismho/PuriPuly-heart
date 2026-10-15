# Split native prefill / decode timings — all tested options

Native timings are **not audio end-to-end latency**. App wall time is separately CLI text-submit launch → matched translation-completion event, excluding audio collection, endpointing, STT and GUI rendering. Failed/unavailable fields are N/A, never invented zero.

**Tokens:** P = actual uncached prompt evaluations; C = preexisting prompt KV hits; O = API predicted output; D = native perf line's **printed** eval-token count. API tokens_evaluated is total prompt; tokens_cached is final KV occupancy, not C. Do not divide cached prompt by prefill time. Printed D can equal O while native TPS accounts O−1: observed21 output/printed tokens,423.682ms,47.205215tokens/s (rate numerator20). We retain reported TPS; D is not a traced pure forward-pass counter. Generic logs lack C/O/total; stock managed stderr lacks D. JSON retains actual source conventions and spread.

Throughput medians are per-observation rates, not ratios of medians. Pressure-qualified/repeat identities remain; consult RESULTS.md/repair-matrix.json before ranking. Standalone concurrent native logs are phase-only, not false per-request joins. Generic app timings are ordered; managed structured diagnostics additionally join host/request ID→utterance UUID.

**Reference labels:** `e4b-product` in standalone tables is the historical standalone E4B control; in app tables it is a **generic user-managed endpoint reference, not current managed product latency**. `managed-e4b-stock` is the true managed owner/prefix-slot reference.

App hot_same_pair and pair_alternation group actual adjacent same-pair versus changed-pair submissions within each fresh app process (n4/n24 for successful30-text runs). App three_slots is N/A: submissions were sequential; three-slot occupancy is separately measured in the standalone matrix.

## Standalone: first_prefix

| Run | n | prefill_ms med / p95 | prefill_tps med | decode_ms med / p95 | decode_tps med | P eval / C hit / O output / D printed med |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| followup-placement3072-n18-mmap-a1 | 3 | 1519.30 / 7855.34 | 597.65 | 295.05 / 560.59 | 40.67 | 908.00 / 0.00 / 13.00 / 13.00 |
| followup-placement3072-n18-read-b1 | 3 | 646.54 / 769.14 | 1404.40 | 218.27 / 390.48 | 54.98 | 908.00 / 0.00 / 13.00 / 13.00 |
| followup-placement3072-n24-mmap-a1 | 3 | 1691.76 / 4714.59 | 533.76 | 489.37 / 695.25 | 28.77 | 908.00 / 0.00 / 13.00 / 13.00 |
| followup-placement3072-n24-read-b1 | 3 | 801.76 / 900.77 | 1126.28 | 279.32 / 456.16 | 43.84 | 908.00 / 0.00 / 13.00 / 13.00 |

## Standalone: hot_same_pair

| Run | n | prefill_ms med / p95 | prefill_tps med | decode_ms med / p95 | decode_tps med | P eval / C hit / O output / D printed med |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| followup-placement3072-n18-mmap-a1 | 5 | 189.45 / 223.03 | 125.54 | 241.74 / 439.93 | 56.25 | 26.00 / 884.00 / 15.00 / 15.00 |
| followup-placement3072-n18-read-b1 | 5 | 92.99 / 96.21 | 273.64 | 235.05 / 358.98 | 59.56 | 26.00 / 884.00 / 15.00 / 15.00 |
| followup-placement3072-n24-mmap-a1 | 5 | 332.45 / 470.24 | 75.32 | 355.90 / 574.81 | 39.34 | 26.00 / 884.00 / 15.00 / 15.00 |
| followup-placement3072-n24-read-b1 | 5 | 133.59 / 170.54 | 191.95 | 278.05 / 425.74 | 50.35 | 26.00 / 884.00 / 15.00 / 15.00 |

## Standalone: pair_alternation

| Run | n | prefill_ms med / p95 | prefill_tps med | decode_ms med / p95 | decode_tps med | P eval / C hit / O output / D printed med |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| followup-placement3072-n18-mmap-a1 | 20 | 249.11 / 527.20 | 153.31 | 338.67 / 438.51 | 47.10 | 28.00 / 878.00 / 16.50 / 16.50 |
| followup-placement3072-n18-read-b1 | 20 | 104.82 / 267.01 | 308.62 | 275.48 / 366.53 | 56.13 | 28.00 / 878.00 / 16.50 / 16.50 |
| followup-placement3072-n24-mmap-a1 | 20 | 415.56 / 734.43 | 83.55 | 475.14 / 626.95 | 31.12 | 28.00 / 878.00 / 16.50 / 16.50 |
| followup-placement3072-n24-read-b1 | 20 | 155.31 / 346.33 | 214.51 | 344.73 / 446.06 | 44.88 | 28.00 / 878.00 / 16.50 / 16.50 |

## Standalone: three_slots

| Run | n | prefill_ms med / p95 | prefill_tps med | decode_ms med / p95 | decode_tps med | P eval / C hit / O output / D printed med |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| followup-placement3072-n18-mmap-a1 | 15 | 260.81 / 734.81 | 99.84 | 543.59 / 919.07 | 24.14 | 26.00 / 882.00 / 15.00 / 15.00 |
| followup-placement3072-n18-read-b1 | 15 | 154.72 / 419.37 | 187.91 | 394.02 / 552.56 | 36.16 | 26.00 / 882.00 / 15.00 / 15.00 |
| followup-placement3072-n24-mmap-a1 | 15 | 499.48 / 1206.40 | 51.87 | 897.55 / 1477.58 | 15.05 | 26.00 / 882.00 / 15.00 / 15.00 |
| followup-placement3072-n24-read-b1 | 15 | 192.60 / 568.64 | 145.11 | 548.70 / 679.94 | 28.81 | 26.00 / 882.00 / 15.00 / 15.00 |

## Actual app native inference: all_text_submissions

First text per app instance is not a guaranteed cold-prefix observation; explicit managed preparation/restore is outside the text-submit wall timer.

| Run | n | prefill_ms med / p95 | prefill_tps med | decode_ms med / p95 | decode_tps med | P eval / C hit / O output / D printed med |
| --- | ---: | ---: | ---: | ---: | ---: | --- |

## Actual app native inference: first_text_per_fresh_app

First text per app instance is not a guaranteed cold-prefix observation; explicit managed preparation/restore is outside the text-submit wall timer.

| Run | n | prefill_ms med / p95 | prefill_tps med | decode_ms med / p95 | decode_tps med | P eval / C hit / O output / D printed med |
| --- | ---: | ---: | ---: | ---: | ---: | --- |

## Actual app native inference: subsequent_text_submissions

First text per app instance is not a guaranteed cold-prefix observation; explicit managed preparation/restore is outside the text-submit wall timer.

| Run | n | prefill_ms med / p95 | prefill_tps med | decode_ms med / p95 | decode_tps med | P eval / C hit / O output / D printed med |
| --- | ---: | ---: | ---: | ---: | ---: | --- |

## Actual app native inference: hot_same_pair

First text per app instance is not a guaranteed cold-prefix observation; explicit managed preparation/restore is outside the text-submit wall timer.

| Run | n | prefill_ms med / p95 | prefill_tps med | decode_ms med / p95 | decode_tps med | P eval / C hit / O output / D printed med |
| --- | ---: | ---: | ---: | ---: | ---: | --- |

## Actual app native inference: pair_alternation

First text per app instance is not a guaranteed cold-prefix observation; explicit managed preparation/restore is outside the text-submit wall timer.

| Run | n | prefill_ms med / p95 | prefill_tps med | decode_ms med / p95 | decode_tps med | P eval / C hit / O output / D printed med |
| --- | ---: | ---: | ---: | ---: | ---: | --- |

## Actual app native inference: three_slots

First text per app instance is not a guaranteed cold-prefix observation; explicit managed preparation/restore is outside the text-submit wall timer.

| Run | n | prefill_ms med / p95 | prefill_tps med | decode_ms med / p95 | decode_tps med | P eval / C hit / O output / D printed med |
| --- | ---: | ---: | ---: | ---: | ---: | --- |

## Bounded memory-attribution diagnostic work (not ranking runs)

| Run | n | prefill_ms med / p95 | prefill_tps med | decode_ms med / p95 | decode_tps med | P eval / C hit / O output / D printed med |
| --- | ---: | ---: | ---: | ---: | ---: | --- |

## Missing / failed records


Per-run sidecars are under results/timings/; originals are not rewritten. Complete distributions/raw native observations are in results/timing-summary.json. True managed prefix-preparation/restore wall windows and any available native metric snapshots remain in its original app record, separately from translation durations.
