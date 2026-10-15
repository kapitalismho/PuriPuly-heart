# Split native prefill / decode timings — all tested options

Native timings are **not audio end-to-end latency**. App wall time is separately CLI text-submit launch → matched translation-completion event, excluding audio collection, endpointing, STT and GUI rendering. Failed/unavailable fields are N/A, never invented zero.

**Tokens:** P = actual uncached prompt evaluations; C = preexisting prompt KV hits; O = API predicted output; D = native perf line's **printed** eval-token count. API tokens_evaluated is total prompt; tokens_cached is final KV occupancy, not C. Do not divide cached prompt by prefill time. Printed D can equal O while native TPS accounts O−1: observed21 output/printed tokens,423.682ms,47.205215tokens/s (rate numerator20). We retain reported TPS; D is not a traced pure forward-pass counter. Generic logs lack C/O/total; stock managed stderr lacks D. JSON retains actual source conventions and spread.

Throughput medians are per-observation rates, not ratios of medians. Pressure-qualified/repeat identities remain; consult RESULTS.md/repair-matrix.json before ranking. Standalone concurrent native logs are phase-only, not false per-request joins. Generic app timings are ordered; managed structured diagnostics additionally join host/request ID→utterance UUID.

**Reference labels:** `e4b-product` in standalone tables is the historical standalone E4B control; in app tables it is a **generic user-managed endpoint reference, not current managed product latency**. `managed-e4b-stock` is the true managed owner/prefix-slot reference.

App hot_same_pair and pair_alternation group actual adjacent same-pair versus changed-pair submissions within each fresh app process (n4/n24 for successful30-text runs). App three_slots is N/A: submissions were sequential; three-slot occupancy is separately measured in the standalone matrix.

## Standalone: first_prefix

| Run | n | prefill_ms med / p95 | prefill_tps med | decode_ms med / p95 | decode_tps med | P eval / C hit / O output / D printed med |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| followup-allhost3072-mmap-a1 | 1 | 21303.53 / 21303.53 | 42.81 | 1394.55 / 1394.55 | 14.34 | 912.00 / 0.00 / 21.00 / 21.00 |
| followup-allhost3072-read-b1 | 3 | 958.94 / 1029.80 | 941.66 | 350.24 / 485.10 | 34.26 | 908.00 / 0.00 / 13.00 / 13.00 |
| followup-n24cache2048-mmap-a1 | 3 | 1902.71 / 3662.22 | 474.59 | 582.19 / 735.62 | 23.93 | 908.00 / 0.00 / 13.00 / 13.00 |
| followup-n24cache2048-mmap-a2-bracket | 3 | 2064.84 / 3653.41 | 439.74 | 576.33 / 798.88 | 24.71 | 908.00 / 0.00 / 13.00 / 13.00 |
| followup-n24cache2048-read-b1 | 3 | 860.30 / 897.55 | 1049.63 | 311.19 / 444.52 | 41.64 | 908.00 / 0.00 / 13.00 / 13.00 |

## Standalone: hot_same_pair

| Run | n | prefill_ms med / p95 | prefill_tps med | decode_ms med / p95 | decode_tps med | P eval / C hit / O output / D printed med |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| followup-allhost3072-mmap-a1 | 0 | N/A | N/A | N/A | N/A | N/A / N/A / N/A / N/A |
| followup-allhost3072-read-b1 | 5 | 176.92 / 212.45 | 142.09 | 338.39 / 518.84 | 41.56 | 26.00 / 884.00 / 15.00 / 15.00 |
| followup-n24cache2048-mmap-a1 | 5 | 457.80 / 537.88 | 53.74 | 516.57 / 799.06 | 28.04 | 26.00 / 884.00 / 15.00 / 15.00 |
| followup-n24cache2048-mmap-a2-bracket | 5 | 452.42 / 550.27 | 56.44 | 450.14 / 690.91 | 31.10 | 26.00 / 884.00 / 15.00 / 15.00 |
| followup-n24cache2048-read-b1 | 5 | 164.25 / 192.56 | 154.07 | 303.52 / 446.98 | 46.13 | 26.00 / 884.00 / 15.00 / 15.00 |

## Standalone: pair_alternation

| Run | n | prefill_ms med / p95 | prefill_tps med | decode_ms med / p95 | decode_tps med | P eval / C hit / O output / D printed med |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| followup-allhost3072-mmap-a1 | 0 | N/A | N/A | N/A | N/A | N/A / N/A / N/A / N/A |
| followup-allhost3072-read-b1 | 20 | 204.54 / 418.38 | 153.71 | 400.63 / 522.94 | 36.55 | 28.00 / 878.00 / 16.50 / 16.50 |
| followup-n24cache2048-mmap-a1 | 20 | 529.05 / 829.71 | 61.03 | 630.97 / 854.85 | 22.79 | 28.00 / 878.00 / 16.50 / 16.50 |
| followup-n24cache2048-mmap-a2-bracket | 20 | 492.03 / 806.21 | 62.77 | 610.84 / 809.50 | 23.44 | 28.00 / 878.00 / 16.50 / 16.50 |
| followup-n24cache2048-read-b1 | 20 | 184.96 / 378.45 | 168.34 | 363.51 / 504.88 | 40.52 | 28.00 / 878.00 / 16.50 / 16.50 |

## Standalone: three_slots

| Run | n | prefill_ms med / p95 | prefill_tps med | decode_ms med / p95 | decode_tps med | P eval / C hit / O output / D printed med |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| followup-allhost3072-mmap-a1 | 0 | N/A | N/A | N/A | N/A | N/A / N/A / N/A / N/A |
| followup-allhost3072-read-b1 | 15 | 243.00 / 419.14 | 110.10 | 700.35 / 953.65 | 21.00 | 26.00 / 882.00 / 15.00 / 15.00 |
| followup-n24cache2048-mmap-a1 | 15 | 573.27 / 1089.57 | 45.35 | 1350.27 / 2412.81 | 10.62 | 26.00 / 882.00 / 15.00 / 15.00 |
| followup-n24cache2048-mmap-a2-bracket | 15 | 568.11 / 871.48 | 44.01 | 1296.40 / 1837.74 | 11.01 | 26.00 / 882.00 / 15.00 / 15.00 |
| followup-n24cache2048-read-b1 | 15 | 234.80 / 573.81 | 109.01 | 583.59 / 1020.30 | 23.51 | 26.00 / 882.00 / 15.00 / 15.00 |

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

- followup-allhost3072-mmap-a1: Owned conservative safety termination, NOT a runtime OOM

Per-run sidecars are under results/timings/; originals are not rewritten. Complete distributions/raw native observations are in results/timing-summary.json. True managed prefix-preparation/restore wall windows and any available native metric snapshots remain in its original app record, separately from translation durations.
