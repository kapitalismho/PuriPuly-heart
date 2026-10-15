# Split native prefill / decode timings — all tested options

Native timings are **not audio end-to-end latency**. App wall time is separately CLI text-submit launch → matched translation-completion event, excluding audio collection, endpointing, STT and GUI rendering. Failed/unavailable fields are N/A, never invented zero.

**Tokens:** P = actual uncached prompt evaluations; C = preexisting prompt KV hits; O = API predicted output; D = native perf line's **printed** eval-token count. API tokens_evaluated is total prompt; tokens_cached is final KV occupancy, not C. Do not divide cached prompt by prefill time. Printed D can equal O while native TPS accounts O−1: observed21 output/printed tokens,423.682ms,47.205215tokens/s (rate numerator20). We retain reported TPS; D is not a traced pure forward-pass counter. Generic logs lack C/O/total; stock managed stderr lacks D. JSON retains actual source conventions and spread.

Throughput medians are per-observation rates, not ratios of medians. Pressure-qualified/repeat identities remain; consult RESULTS.md/repair-matrix.json before ranking. Standalone concurrent native logs are phase-only, not false per-request joins. Generic app timings are ordered; managed structured diagnostics additionally join host/request ID→utterance UUID.

**Reference labels:** `e4b-product` in standalone tables is the historical standalone E4B control; in app tables it is a **generic user-managed endpoint reference, not current managed product latency**. `managed-e4b-stock` is the true managed owner/prefix-slot reference.

App hot_same_pair and pair_alternation group actual adjacent same-pair versus changed-pair submissions within each fresh app process (n4/n24 for successful30-text runs). App three_slots is N/A: submissions were sequential; three-slot occupancy is separately measured in the standalone matrix.

## Standalone: first_prefix

| Run | n | prefill_ms med / p95 | prefill_tps med | decode_ms med / p95 | decode_tps med | P eval / C hit / O output / D printed med |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| e4b-product | 3 | 1086.18 / 1123.24 | 835.96 | 150.34 / 282.25 | 79.44 | 904.00 / 0.00 / 13.00 / 13.00 |
| new-cpu-moe-cache1024 | 3 | 3343.78 / 5430.38 | 271.55 | 1274.95 / 1699.43 | 10.72 | 908.00 / 0.00 / 13.00 / 13.00 |
| new-cpu-moe-cache2048-swa | 3 | 2761.70 / 4797.22 | 328.78 | 847.27 / 1052.80 | 16.71 | 908.00 / 0.00 / 13.00 / 13.00 |
| new-cpu-moe-cache2048 | 3 | 2477.59 / 6315.11 | 366.48 | 839.56 / 912.98 | 15.59 | 908.00 / 0.00 / 13.00 / 13.00 |
| new-cpu-moe-cache3072-swa-nocram | 3 | 1988.96 / 4220.92 | 454.01 | 686.55 / 766.96 | 22.64 | 908.00 / 0.00 / 13.00 / 13.00 |
| new-cpu-moe-cache3072-swa | 3 | 1991.05 / 4210.76 | 453.53 | 680.97 / 754.19 | 22.80 | 908.00 / 0.00 / 13.00 / 13.00 |
| new-cpu-moe-cache3072 | 3 | 2784.87 / 5153.67 | 326.05 | 730.70 / 826.95 | 21.02 | 908.00 / 0.00 / 13.00 / 13.00 |
| new-cpu-moe-cache4096 | 0 | N/A | N/A | N/A | N/A | N/A / N/A / N/A / N/A |
| new-cpu-moe-off-swa-nocram | 3 | 2262.50 / 4434.24 | 399.12 | 493.01 / 828.10 | 24.80 | 908.00 / 0.00 / 13.00 / 13.00 |
| new-cpu-moe-off-swa | 3 | 2284.41 / 4388.16 | 397.07 | 500.38 / 828.28 | 25.21 | 908.00 / 0.00 / 13.00 / 13.00 |
| new-cpu-moe-off | 3 | 3032.77 / 5212.34 | 299.40 | 500.88 / 835.68 | 23.96 | 908.00 / 0.00 / 13.00 / 13.00 |
| new-full-nocram | 3 | 1311.16 / 1334.55 | 688.70 | 129.67 / 225.45 | 92.61 | 908.00 / 0.00 / 13.00 / 13.00 |
| new-full-swa-nocram | 3 | 1243.53 / 1264.30 | 728.10 | 131.72 / 220.44 | 90.73 | 908.00 / 0.00 / 13.00 / 13.00 |
| new-full-swa | 3 | 1242.52 / 1258.49 | 729.99 | 129.89 / 214.93 | 92.39 | 908.00 / 0.00 / 13.00 / 13.00 |
| new-full | 3 | 1321.45 / 1342.83 | 685.32 | 129.65 / 225.98 | 92.56 | 908.00 / 0.00 / 13.00 / 13.00 |
| new-ncpu12-cache1024-swa-nocram-repair2 | 3 | 1063.63 / 1993.95 | 848.98 | 357.43 / 468.38 | 36.94 | 908.00 / 0.00 / 13.00 / 13.00 |
| new-ncpu12-cache1024-swa-nocram | 3 | 1073.82 / 2147.83 | 840.93 | 366.49 / 525.47 | 36.03 | 908.00 / 0.00 / 13.00 / 13.00 |
| new-ncpu12-cache2048-swa-nocram-repair-control | 3 | 1044.67 / 8198.60 | 864.39 | 236.85 / 455.11 | 50.67 | 908.00 / 0.00 / 13.00 / 13.00 |
| new-ncpu12-cache2048-swa-nocram-repair2-control | 3 | 910.81 / 1956.61 | 996.92 | 234.48 / 431.33 | 51.18 | 908.00 / 0.00 / 13.00 / 13.00 |
| new-ncpu12-cache2048-swa-nocram | 3 | 918.38 / 1969.62 | 988.69 | 241.17 / 443.82 | 49.76 | 908.00 / 0.00 / 13.00 / 13.00 |
| new-ncpu12-cache2048-swa | 3 | 914.27 / 1962.67 | 993.15 | 240.79 / 441.51 | 49.84 | 908.00 / 0.00 / 13.00 / 13.00 |
| new-ncpu12-cache2048 | 3 | 1139.35 / 2144.86 | 796.94 | 241.26 / 418.69 | 50.16 | 908.00 / 0.00 / 13.00 / 13.00 |
| new-ncpu12-cache3072-swa-nocram | 3 | 921.73 / 2038.21 | 985.11 | 241.17 / 423.68 | 49.76 | 908.00 / 0.00 / 13.00 / 13.00 |
| new-ncpu12-off-swa-nocram-repair2 | 3 | 1150.19 / 2080.86 | 785.09 | 291.06 / 482.62 | 41.23 | 908.00 / 0.00 / 13.00 / 13.00 |
| new-ncpu12-off-swa-nocram | 3 | 1224.37 / 2245.15 | 741.61 | 272.85 / 534.32 | 43.45 | 908.00 / 0.00 / 13.00 / 13.00 |
| new-ncpu12-off | 3 | 1517.35 / 2479.39 | 595.31 | 286.39 / 505.26 | 41.90 | 908.00 / 0.00 / 13.00 / 13.00 |
| new-ncpu24-cache2048-swa-nocram | 3 | 1751.29 / 3542.92 | 515.62 | 566.68 / 676.97 | 25.29 | 908.00 / 0.00 / 13.00 / 13.00 |
| new-ncpu24-cache2048-swa | 3 | 1768.98 / 3500.74 | 510.46 | 563.07 / 696.50 | 25.01 | 908.00 / 0.00 / 13.00 / 13.00 |
| new-ncpu24-cache2048 | 3 | 2414.07 / 4223.92 | 376.13 | 635.18 / 780.91 | 23.79 | 908.00 / 0.00 / 13.00 / 13.00 |
| new-ncpu24-off | 3 | 2689.08 / 4495.02 | 337.66 | 434.99 / 775.28 | 27.59 | 908.00 / 0.00 / 13.00 / 13.00 |
| new-partial14-swa-nocram | 3 | 1657.82 / 2857.47 | 544.69 | 374.91 / 615.69 | 32.48 | 908.00 / 0.00 / 13.00 / 13.00 |
| new-partial14-swa | 3 | 1829.49 / 3222.19 | 496.31 | 367.32 / 630.39 | 32.34 | 908.00 / 0.00 / 13.00 / 13.00 |
| new-partial14 | 3 | 2302.21 / 3493.97 | 394.40 | 395.45 / 659.59 | 30.35 | 908.00 / 0.00 / 13.00 / 13.00 |
| new-partial20 | 3 | 1544.41 / 2439.13 | 584.69 | 307.10 / 507.96 | 39.37 | 908.00 / 0.00 / 13.00 / 13.00 |
| old-full | 3 | 3063.79 / 4686.75 | 296.37 | 122.52 / 409.05 | 97.94 | 908.00 / 0.00 / 13.00 / 13.00 |

## Standalone: hot_same_pair

| Run | n | prefill_ms med / p95 | prefill_tps med | decode_ms med / p95 | decode_tps med | P eval / C hit / O output / D printed med |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| e4b-product | 5 | 94.93 / 520.55 | 212.19 | 187.04 / 187.91 | 79.98 | 22.00 / 884.00 / 16.00 / 16.00 |
| new-cpu-moe-cache1024 | 5 | 1550.49 / 1905.99 | 314.30 | 794.04 / 1384.30 | 17.63 | 511.00 / 399.00 / 15.00 / 15.00 |
| new-cpu-moe-cache2048-swa | 5 | 579.05 / 663.31 | 44.66 | 586.57 / 904.02 | 23.87 | 26.00 / 884.00 / 15.00 / 15.00 |
| new-cpu-moe-cache2048 | 5 | 1472.22 / 5777.63 | 197.03 | 893.14 / 1463.67 | 22.39 | 511.00 / 399.00 / 15.00 / 15.00 |
| new-cpu-moe-cache3072-swa-nocram | 5 | 478.06 / 601.80 | 53.11 | 472.06 / 729.94 | 29.66 | 26.00 / 884.00 / 15.00 / 15.00 |
| new-cpu-moe-cache3072-swa | 5 | 484.69 / 583.83 | 52.61 | 469.36 / 732.84 | 29.83 | 26.00 / 884.00 / 15.00 / 15.00 |
| new-cpu-moe-cache3072 | 5 | 1279.89 / 1439.43 | 389.49 | 392.64 / 698.73 | 35.66 | 511.00 / 399.00 / 15.00 / 15.00 |
| new-cpu-moe-cache4096 | 0 | N/A | N/A | N/A | N/A | N/A / N/A / N/A / N/A |
| new-cpu-moe-off-swa-nocram | 5 | 334.09 / 359.15 | 77.40 | 545.15 / 796.84 | 25.77 | 26.00 / 884.00 / 15.00 / 15.00 |
| new-cpu-moe-off-swa | 5 | 341.95 / 361.52 | 76.03 | 557.54 / 796.00 | 25.32 | 26.00 / 884.00 / 15.00 / 15.00 |
| new-cpu-moe-off | 5 | 1405.16 / 1476.38 | 348.83 | 553.99 / 813.87 | 25.31 | 511.00 / 399.00 / 15.00 / 15.00 |
| new-full-nocram | 5 | 844.42 / 857.96 | 602.52 | 150.47 / 215.37 | 93.04 | 511.00 / 399.00 / 15.00 / 15.00 |
| new-full-swa-nocram | 5 | 77.68 / 80.97 | 325.13 | 156.89 / 218.19 | 91.08 | 26.00 / 884.00 / 15.00 / 15.00 |
| new-full-swa | 5 | 77.22 / 685.23 | 331.76 | 151.37 / 218.68 | 92.49 | 26.00 / 884.00 / 15.00 / 15.00 |
| new-full | 5 | 846.57 / 860.82 | 603.11 | 151.45 / 215.35 | 92.55 | 511.00 / 399.00 / 15.00 / 15.00 |
| new-ncpu12-cache1024-swa-nocram-repair2 | 5 | 265.62 / 294.67 | 97.00 | 311.06 / 467.89 | 45.01 | 26.00 / 884.00 / 15.00 / 15.00 |
| new-ncpu12-cache1024-swa-nocram | 5 | 266.53 / 302.96 | 93.55 | 312.45 / 472.29 | 44.00 | 26.00 / 884.00 / 15.00 / 15.00 |
| new-ncpu12-cache2048-swa-nocram-repair-control | 5 | 173.70 / 199.47 | 141.42 | 212.47 / 353.63 | 64.58 | 26.00 / 884.00 / 15.00 / 15.00 |
| new-ncpu12-cache2048-swa-nocram-repair2-control | 5 | 153.58 / 170.07 | 164.64 | 210.06 / 351.16 | 65.05 | 26.00 / 884.00 / 15.00 / 15.00 |
| new-ncpu12-cache2048-swa-nocram | 5 | 158.47 / 180.72 | 155.10 | 217.04 / 355.99 | 64.02 | 26.00 / 884.00 / 15.00 / 15.00 |
| new-ncpu12-cache2048-swa | 5 | 149.29 / 176.40 | 161.52 | 213.35 / 352.19 | 64.51 | 26.00 / 884.00 / 15.00 / 15.00 |
| new-ncpu12-cache2048 | 5 | 622.33 / 714.68 | 764.70 | 223.71 / 368.47 | 62.58 | 511.00 / 399.00 / 15.00 / 15.00 |
| new-ncpu12-cache3072-swa-nocram | 5 | 57.63 / 108.84 | 416.43 | 215.30 / 297.87 | 67.14 | 26.00 / 884.00 / 15.00 / 15.00 |
| new-ncpu12-off-swa-nocram-repair2 | 5 | 166.27 / 187.60 | 149.25 | 323.36 / 478.13 | 43.43 | 26.00 / 884.00 / 15.00 / 15.00 |
| new-ncpu12-off-swa-nocram | 5 | 173.13 / 188.16 | 140.75 | 322.05 / 499.90 | 43.67 | 26.00 / 884.00 / 15.00 / 15.00 |
| new-ncpu12-off | 5 | 759.88 / 815.03 | 650.47 | 331.12 / 486.96 | 42.87 | 511.00 / 399.00 / 15.00 / 15.00 |
| new-ncpu24-cache2048-swa-nocram | 5 | 444.85 / 517.05 | 57.82 | 451.86 / 684.44 | 30.98 | 26.00 / 884.00 / 15.00 / 15.00 |
| new-ncpu24-cache2048-swa | 5 | 440.74 / 520.82 | 58.53 | 446.20 / 681.86 | 31.38 | 26.00 / 884.00 / 15.00 / 15.00 |
| new-ncpu24-cache2048 | 5 | 1171.75 / 1331.57 | 415.71 | 419.14 / 672.11 | 33.40 | 511.00 / 399.00 / 15.00 / 15.00 |
| new-ncpu24-off | 5 | 1185.91 / 1282.94 | 412.82 | 517.08 / 747.89 | 28.34 | 511.00 / 399.00 / 15.00 / 15.00 |
| new-partial14-swa-nocram | 5 | 359.80 / 406.73 | 72.16 | 431.16 / 644.68 | 32.57 | 26.00 / 884.00 / 15.00 / 15.00 |
| new-partial14-swa | 5 | 361.35 / 392.56 | 71.33 | 436.83 / 654.32 | 32.29 | 26.00 / 884.00 / 15.00 / 15.00 |
| new-partial14 | 5 | 1083.04 / 1157.21 | 461.56 | 433.14 / 638.52 | 32.50 | 511.00 / 399.00 / 15.00 / 15.00 |
| new-partial20 | 5 | 822.99 / 863.81 | 614.16 | 348.06 / 511.91 | 41.02 | 511.00 / 399.00 / 15.00 / 15.00 |
| old-full | 5 | 1797.16 / 2051.24 | 280.94 | 144.62 / 235.45 | 96.81 | 511.00 / 399.00 / 15.00 / 15.00 |

## Standalone: pair_alternation

| Run | n | prefill_ms med / p95 | prefill_tps med | decode_ms med / p95 | decode_tps med | P eval / C hit / O output / D printed med |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| e4b-product | 20 | 98.51 / 348.55 | 250.94 | 187.84 / 237.24 | 80.06 | 24.00 / 878.00 / 16.00 / 16.00 |
| new-cpu-moe-cache1024 | 20 | 1600.67 / 1935.02 | 301.77 | 1270.95 / 1631.38 | 11.50 | 511.50 / 401.00 / 16.50 / 16.50 |
| new-cpu-moe-cache2048-swa | 20 | 649.59 / 991.88 | 47.59 | 870.49 / 1182.71 | 16.83 | 28.00 / 878.00 / 16.50 / 16.50 |
| new-cpu-moe-cache2048 | 20 | 1588.53 / 3579.23 | 290.70 | 1004.30 / 1416.02 | 14.81 | 511.50 / 401.00 / 16.50 / 16.50 |
| new-cpu-moe-cache3072-swa-nocram | 20 | 557.21 / 926.38 | 58.69 | 667.52 / 842.57 | 21.33 | 28.00 / 878.00 / 16.50 / 16.50 |
| new-cpu-moe-cache3072-swa | 20 | 560.32 / 906.53 | 57.66 | 669.74 / 831.23 | 21.17 | 28.00 / 878.00 / 16.50 / 16.50 |
| new-cpu-moe-cache3072 | 20 | 1368.10 / 1795.52 | 363.43 | 665.37 / 861.57 | 21.45 | 511.50 / 401.00 / 16.50 / 16.50 |
| new-cpu-moe-cache4096 | 0 | N/A | N/A | N/A | N/A | N/A / N/A / N/A / N/A |
| new-cpu-moe-off-swa-nocram | 20 | 384.08 / 1085.92 | 77.50 | 629.62 / 835.27 | 25.02 | 28.00 / 878.00 / 16.50 / 16.50 |
| new-cpu-moe-off-swa | 20 | 356.38 / 1078.47 | 79.39 | 604.86 / 824.31 | 25.36 | 28.00 / 878.00 / 16.50 / 16.50 |
| new-cpu-moe-off | 20 | 1423.05 / 1612.21 | 351.11 | 604.47 / 821.68 | 25.56 | 511.50 / 401.00 / 16.50 / 16.50 |
| new-full-nocram | 20 | 841.00 / 863.96 | 601.71 | 167.18 / 226.18 | 92.64 | 511.50 / 401.00 / 16.50 / 16.50 |
| new-full-swa-nocram | 20 | 84.85 / 427.62 | 340.15 | 169.27 / 229.66 | 91.20 | 28.00 / 878.00 / 16.50 / 16.50 |
| new-full-swa | 20 | 82.28 / 428.06 | 345.96 | 166.91 / 225.51 | 92.77 | 28.00 / 878.00 / 16.50 / 16.50 |
| new-full | 20 | 851.72 / 882.73 | 597.12 | 167.23 / 226.16 | 92.52 | 511.50 / 401.00 / 16.50 / 16.50 |
| new-ncpu12-cache1024-swa-nocram-repair2 | 20 | 284.70 / 472.46 | 109.63 | 408.86 / 536.43 | 37.37 | 28.00 / 878.00 / 16.50 / 16.50 |
| new-ncpu12-cache1024-swa-nocram | 20 | 293.86 / 478.55 | 111.46 | 411.54 / 549.16 | 36.85 | 28.00 / 878.00 / 16.50 / 16.50 |
| new-ncpu12-cache2048-swa-nocram-repair-control | 20 | 202.24 / 428.64 | 186.28 | 281.50 / 366.33 | 55.60 | 28.00 / 878.00 / 16.50 / 16.50 |
| new-ncpu12-cache2048-swa-nocram-repair2-control | 20 | 183.17 / 397.40 | 197.70 | 281.44 / 361.89 | 55.97 | 28.00 / 878.00 / 16.50 / 16.50 |
| new-ncpu12-cache2048-swa-nocram | 20 | 186.70 / 398.33 | 191.93 | 286.36 / 369.76 | 54.96 | 28.00 / 878.00 / 16.50 / 16.50 |
| new-ncpu12-cache2048-swa | 20 | 186.37 / 396.49 | 190.55 | 287.14 / 365.88 | 55.22 | 28.00 / 878.00 / 16.50 / 16.50 |
| new-ncpu12-cache2048 | 20 | 665.41 / 764.96 | 756.07 | 294.53 / 379.28 | 53.68 | 511.50 / 401.00 / 16.50 / 16.50 |
| new-ncpu12-cache3072-swa-nocram | 20 | 92.55 / 335.50 | 392.31 | 226.01 / 298.06 | 68.04 | 28.00 / 878.00 / 16.50 / 16.50 |
| new-ncpu12-off-swa-nocram-repair2 | 20 | 178.51 / 531.69 | 158.75 | 354.24 / 488.62 | 44.04 | 28.00 / 878.00 / 16.50 / 16.50 |
| new-ncpu12-off-swa-nocram | 20 | 198.96 / 566.46 | 149.24 | 368.94 / 497.81 | 42.11 | 28.00 / 878.00 / 16.50 / 16.50 |
| new-ncpu12-off | 20 | 783.04 / 878.24 | 646.78 | 373.83 / 489.77 | 42.41 | 511.50 / 401.00 / 16.50 / 16.50 |
| new-ncpu24-cache2048-swa-nocram | 20 | 504.90 / 795.52 | 63.59 | 609.56 / 824.63 | 23.41 | 28.00 / 878.00 / 16.50 / 16.50 |
| new-ncpu24-cache2048-swa | 20 | 503.72 / 795.66 | 63.53 | 601.11 / 791.29 | 23.70 | 28.00 / 878.00 / 16.50 / 16.50 |
| new-ncpu24-cache2048 | 20 | 1183.88 / 1599.67 | 420.67 | 601.99 / 831.35 | 24.76 | 511.50 / 401.00 / 16.50 / 16.50 |
| new-ncpu24-off | 20 | 1206.78 / 1410.13 | 411.36 | 539.53 / 745.74 | 28.53 | 511.50 / 401.00 / 16.50 / 16.50 |
| new-partial14-swa-nocram | 20 | 408.96 / 783.01 | 73.73 | 469.69 / 637.03 | 32.71 | 28.00 / 878.00 / 16.50 / 16.50 |
| new-partial14-swa | 20 | 391.23 / 777.89 | 72.85 | 473.87 / 641.50 | 32.35 | 28.00 / 878.00 / 16.50 / 16.50 |
| new-partial14 | 20 | 1135.27 / 1335.51 | 431.89 | 494.20 / 642.80 | 32.22 | 511.50 / 401.00 / 16.50 / 16.50 |
| new-partial20 | 20 | 866.12 / 966.48 | 578.54 | 378.67 / 528.73 | 40.74 | 511.50 / 401.00 / 16.50 / 16.50 |
| old-full | 20 | 1834.58 / 2045.92 | 277.92 | 157.77 / 203.55 | 98.00 | 511.50 / 401.00 / 16.50 / 16.50 |

## Standalone: three_slots

| Run | n | prefill_ms med / p95 | prefill_tps med | decode_ms med / p95 | decode_tps med | P eval / C hit / O output / D printed med |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| e4b-product | 15 | 223.99 / 523.16 | 95.26 | 318.62 / 409.71 | 44.30 | 22.00 / 882.00 / 16.00 / 16.00 |
| new-cpu-moe-cache1024 | 15 | 4532.02 / 7981.45 | 111.87 | 4026.37 / 7061.50 | 3.59 | 509.00 / 401.00 / 15.00 / 15.00 |
| new-cpu-moe-cache2048-swa | 15 | 787.05 / 1905.38 | 33.56 | 1828.71 / 2644.67 | 7.82 | 26.00 / 882.00 / 15.00 / 15.00 |
| new-cpu-moe-cache2048 | 15 | 5806.05 / 10989.90 | 87.32 | 2394.41 / 7192.06 | 6.43 | 506.00 / 404.00 / 15.00 / 15.00 |
| new-cpu-moe-cache3072-swa-nocram | 15 | 593.01 / 917.54 | 44.71 | 1360.92 / 2069.03 | 9.95 | 26.00 / 882.00 / 15.00 / 15.00 |
| new-cpu-moe-cache3072-swa | 15 | 687.56 / 1703.92 | 38.31 | 1315.54 / 2234.47 | 9.87 | 26.00 / 882.00 / 15.00 / 15.00 |
| new-cpu-moe-cache3072 | 15 | 3862.07 / 6356.19 | 131.79 | 1506.15 / 3250.76 | 9.94 | 507.00 / 403.00 / 15.00 / 15.00 |
| new-cpu-moe-cache4096 | 0 | N/A | N/A | N/A | N/A | N/A / N/A / N/A / N/A |
| new-cpu-moe-off-swa-nocram | 15 | 702.28 / 1952.58 | 37.02 | 1064.93 / 1808.08 | 13.15 | 26.00 / 882.00 / 15.00 / 15.00 |
| new-cpu-moe-off-swa | 15 | 825.26 / 1955.08 | 31.52 | 1073.79 / 1623.46 | 12.94 | 26.00 / 882.00 / 15.00 / 15.00 |
| new-cpu-moe-off | 15 | 4597.98 / 6884.61 | 108.70 | 1335.53 / 3643.87 | 10.67 | 506.00 / 404.00 / 15.00 / 15.00 |
| new-full-nocram | 15 | 1781.39 / 2581.05 | 279.42 | 290.59 / 1184.62 | 45.53 | 506.00 / 404.00 / 15.00 / 15.00 |
| new-full-swa-nocram | 15 | 134.73 / 500.27 | 192.98 | 279.57 / 688.31 | 54.63 | 26.00 / 882.00 / 15.00 / 15.00 |
| new-full-swa | 15 | 137.03 / 530.64 | 193.55 | 279.71 / 359.47 | 55.22 | 26.00 / 882.00 / 15.00 / 15.00 |
| new-full | 15 | 1803.91 / 2611.01 | 278.81 | 315.13 / 1207.03 | 46.31 | 509.00 / 401.00 / 15.00 / 15.00 |
| new-ncpu12-cache1024-swa-nocram-repair2 | 15 | 318.93 / 748.69 | 78.39 | 943.60 / 1456.03 | 15.64 | 26.00 / 882.00 / 15.00 / 15.00 |
| new-ncpu12-cache1024-swa-nocram | 15 | 344.05 / 852.77 | 72.66 | 802.92 / 1549.94 | 16.61 | 26.00 / 882.00 / 15.00 / 15.00 |
| new-ncpu12-cache2048-swa-nocram-repair-control | 15 | 206.62 / 408.34 | 121.22 | 464.11 / 743.92 | 30.02 | 26.00 / 882.00 / 15.00 / 15.00 |
| new-ncpu12-cache2048-swa-nocram-repair2-control | 15 | 244.82 / 605.17 | 110.68 | 519.61 / 745.45 | 26.38 | 26.00 / 882.00 / 15.00 / 15.00 |
| new-ncpu12-cache2048-swa-nocram | 15 | 234.05 / 414.21 | 110.83 | 518.12 / 764.35 | 31.60 | 26.00 / 882.00 / 15.00 / 15.00 |
| new-ncpu12-cache2048-swa | 15 | 269.28 / 665.83 | 93.56 | 437.80 / 732.82 | 31.55 | 26.00 / 882.00 / 15.00 / 15.00 |
| new-ncpu12-cache2048 | 15 | 1742.29 / 2444.01 | 289.85 | 571.19 / 1271.51 | 28.00 | 507.00 / 403.00 / 15.00 / 15.00 |
| new-ncpu12-cache3072-swa-nocram | 15 | 155.44 / 380.03 | 172.35 | 325.29 / 592.98 | 46.22 | 26.00 / 882.00 / 15.00 / 15.00 |
| new-ncpu12-off-swa-nocram-repair2 | 15 | 362.83 / 913.37 | 70.43 | 602.80 / 911.65 | 24.71 | 26.00 / 882.00 / 15.00 / 15.00 |
| new-ncpu12-off-swa-nocram | 15 | 400.57 / 735.23 | 65.62 | 617.41 / 1185.93 | 23.60 | 26.00 / 882.00 / 15.00 / 15.00 |
| new-ncpu12-off | 15 | 1873.41 / 2751.22 | 266.42 | 708.59 / 1662.20 | 22.58 | 509.00 / 401.00 / 15.00 / 15.00 |
| new-ncpu24-cache2048-swa-nocram | 15 | 1035.15 / 1374.66 | 39.79 | 1333.11 / 2505.66 | 10.76 | 26.00 / 882.00 / 15.00 / 15.00 |
| new-ncpu24-cache2048-swa | 15 | 623.30 / 1466.53 | 43.43 | 1229.68 / 2461.33 | 11.20 | 26.00 / 882.00 / 15.00 / 15.00 |
| new-ncpu24-cache2048 | 15 | 4029.84 / 5888.24 | 120.85 | 1554.84 / 3169.70 | 10.22 | 506.00 / 404.00 / 15.00 / 15.00 |
| new-ncpu24-off | 15 | 2912.25 / 4338.65 | 172.73 | 1107.70 / 2778.75 | 14.44 | 506.00 / 403.00 / 15.00 / 15.00 |
| new-partial14-swa-nocram | 15 | 527.08 / 1463.92 | 48.70 | 1277.45 / 1599.18 | 10.25 | 26.00 / 882.00 / 15.00 / 15.00 |
| new-partial14-swa | 15 | 677.06 / 1471.59 | 37.33 | 1192.79 / 1657.39 | 12.19 | 26.00 / 882.00 / 15.00 / 15.00 |
| new-partial14 | 15 | 2946.89 / 4068.25 | 172.72 | 1293.81 / 2549.22 | 10.98 | 506.00 / 403.00 / 15.00 / 15.00 |
| new-partial20 | 15 | 2171.53 / 2986.07 | 234.40 | 957.95 / 1877.13 | 16.19 | 506.00 / 404.00 / 15.00 / 15.00 |
| old-full | 15 | 3812.58 / 7343.91 | 131.66 | 274.51 / 2728.08 | 45.26 | 507.00 / 402.00 / 15.00 / 15.00 |

## Actual app native inference: all_text_submissions

First text per app instance is not a guaranteed cold-prefix observation; explicit managed preparation/restore is outside the text-submit wall timer.

| Run | n | prefill_ms med / p95 | prefill_tps med | decode_ms med / p95 | decode_tps med | P eval / C hit / O output / D printed med |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| e4b-product / attempt1 | 30 | 348.83 / 378.37 | 626.55 | 183.00 / 220.37 | 82.00 | 218.50 / N/A / N/A / 15.00 |
| managed-e4b-stock / attempt1 | 30 | 1035.38 / 1099.24 | 805.38 | 196.41 / 268.34 | 81.80 | 863.00 / 0.00 / 17.00 / N/A |
| new-cpu-moe-cache3072-swa / attempt1 | 30 | 870.30 / 1374.74 | 256.34 | 713.33 / 795.02 | 20.57 | 222.50 / N/A / N/A / 17.00 |
| new-cpu-moe-cache3072-swa-nocram / attempt1 | 30 | 860.40 / 1333.39 | 259.15 | 696.56 / 754.44 | 21.31 | 222.50 / N/A / N/A / 17.00 |
| new-cpu-moe-off-swa-nocram / attempt1 | 30 | 1027.79 / 1066.34 | 217.78 | 625.33 / 786.59 | 25.51 | 222.50 / N/A / N/A / 17.00 |
| new-full-swa-nocram / attempt1 | 30 | 363.75 / 851.94 | 625.48 | 172.13 / 216.91 | 92.59 | 222.50 / N/A / N/A / 17.00 |
| new-ncpu12-cache1024-swa-nocram-repair2 / attempt1 | 30 | 443.69 / 1053.81 | 502.65 | 446.88 / 509.98 | 35.36 | 222.50 / N/A / N/A / 17.00 |
| new-ncpu12-cache2048-swa-nocram / attempt1 | 30 | 360.16 / 959.04 | 625.98 | 278.40 / 336.92 | 57.11 | 222.50 / N/A / N/A / 17.00 |
| new-ncpu12-cache2048-swa-nocram-repair2-control / attempt1 | 30 | 365.08 / 945.25 | 611.86 | 281.95 / 340.48 | 56.13 | 222.50 / N/A / N/A / 17.00 |
| new-ncpu12-cache3072-swa-nocram / attempt1 | 30 | 285.18 / 987.10 | 806.20 | 229.17 / 342.39 | 67.39 | 222.50 / N/A / N/A / 17.00 |
| new-ncpu12-off-swa-nocram-repair2 / attempt1 | 30 | 513.50 / 1010.53 | 430.53 | 378.69 / 495.69 | 41.59 | 222.50 / N/A / N/A / 17.00 |
| new-ncpu24-cache2048-swa-nocram / attempt1 | 30 | 750.16 / 1261.37 | 297.80 | 664.00 / 691.33 | 24.03 | 222.50 / N/A / N/A / 17.00 |
| new-partial14-swa-nocram / attempt1 | 0 | N/A | N/A | N/A | N/A | N/A / N/A / N/A / N/A |
| new-partial14-swa-nocram / attempt2 | 30 | 741.97 / 1196.59 | 305.22 | 486.96 / 617.94 | 32.56 | 222.50 / N/A / N/A / 17.00 |

## Actual app native inference: first_text_per_fresh_app

First text per app instance is not a guaranteed cold-prefix observation; explicit managed preparation/restore is outside the text-submit wall timer.

| Run | n | prefill_ms med / p95 | prefill_tps med | decode_ms med / p95 | decode_tps med | P eval / C hit / O output / D printed med |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| e4b-product / attempt1 | 2 | 1465.46 / 2830.98 | 274.13 | 208.53 / 270.80 | 63.18 | 434.50 / N/A / N/A / 13.00 |
| managed-e4b-stock / attempt1 | 2 | 818.08 / 1057.41 | 421.55 | 208.42 / 268.85 | 61.00 | 435.00 / 409.00 / 12.50 / N/A |
| new-cpu-moe-cache3072-swa / attempt1 | 2 | 3444.20 / 5513.67 | 87.45 | 696.76 / 729.53 | 17.26 | 438.50 / N/A / N/A / 13.00 |
| new-cpu-moe-cache3072-swa-nocram / attempt1 | 2 | 3410.66 / 5487.94 | 88.14 | 691.81 / 734.03 | 17.41 | 438.50 / N/A / N/A / 13.00 |
| new-cpu-moe-off-swa-nocram / attempt1 | 2 | 3367.74 / 5710.93 | 88.40 | 485.23 / 491.29 | 24.73 | 438.50 / N/A / N/A / 13.00 |
| new-full-swa-nocram / attempt1 | 2 | 1639.66 / 2427.39 | 191.70 | 129.82 / 130.35 | 92.44 | 438.50 / N/A / N/A / 13.00 |
| new-ncpu12-cache1024-swa-nocram-repair2 / attempt1 | 2 | 2177.03 / 3300.26 | 142.23 | 378.66 / 382.01 | 31.69 | 438.50 / N/A / N/A / 13.00 |
| new-ncpu12-cache2048-swa-nocram / attempt1 | 2 | 2123.59 / 3288.13 | 144.07 | 285.75 / 332.07 | 43.13 | 438.50 / N/A / N/A / 13.00 |
| new-ncpu12-cache2048-swa-nocram-repair2-control / attempt1 | 2 | 2133.33 / 3321.41 | 143.00 | 286.45 / 330.39 | 42.90 | 438.50 / N/A / N/A / 13.00 |
| new-ncpu12-cache3072-swa-nocram / attempt1 | 2 | 2238.28 / 3489.45 | 136.20 | 325.65 / 464.33 | 45.01 | 438.50 / N/A / N/A / 13.00 |
| new-ncpu12-off-swa-nocram-repair2 / attempt1 | 2 | 2895.14 / 4779.75 | 103.06 | 290.47 / 292.83 | 41.31 | 438.50 / N/A / N/A / 13.00 |
| new-ncpu24-cache2048-swa-nocram / attempt1 | 2 | 3056.78 / 4852.18 | 98.88 | 581.52 / 593.14 | 20.64 | 438.50 / N/A / N/A / 13.00 |
| new-partial14-swa-nocram / attempt1 | 0 | N/A | N/A | N/A | N/A | N/A / N/A / N/A / N/A |
| new-partial14-swa-nocram / attempt2 | 2 | 2717.54 / 4238.49 | 112.16 | 371.10 / 372.94 | 32.34 | 438.50 / N/A / N/A / 13.00 |

## Actual app native inference: subsequent_text_submissions

First text per app instance is not a guaranteed cold-prefix observation; explicit managed preparation/restore is outside the text-submit wall timer.

| Run | n | prefill_ms med / p95 | prefill_tps med | decode_ms med / p95 | decode_tps med | P eval / C hit / O output / D printed med |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| e4b-product / attempt1 | 28 | 348.83 / 377.76 | 628.60 | 183.00 / 219.45 | 82.00 | 218.50 / N/A / N/A / 16.00 |
| managed-e4b-stock / attempt1 | 28 | 1035.38 / 1099.24 | 806.77 | 196.41 / 257.07 | 81.82 | 866.50 / 0.00 / 17.00 / N/A |
| new-cpu-moe-cache3072-swa / attempt1 | 28 | 868.81 / 911.51 | 257.48 | 713.33 / 795.02 | 20.79 | 222.50 / N/A / N/A / 17.00 |
| new-cpu-moe-cache3072-swa-nocram / attempt1 | 28 | 858.86 / 891.29 | 260.66 | 696.56 / 754.44 | 21.31 | 222.50 / N/A / N/A / 17.00 |
| new-cpu-moe-off-swa-nocram / attempt1 | 28 | 1027.79 / 1055.12 | 218.39 | 625.97 / 786.59 | 25.56 | 222.50 / N/A / N/A / 17.00 |
| new-full-swa-nocram / attempt1 | 28 | 354.88 / 391.41 | 626.10 | 172.27 / 216.91 | 92.59 | 222.50 / N/A / N/A / 17.00 |
| new-ncpu12-cache1024-swa-nocram-repair2 / attempt1 | 28 | 442.97 / 466.02 | 503.98 | 448.72 / 509.98 | 35.36 | 222.50 / N/A / N/A / 17.00 |
| new-ncpu12-cache2048-swa-nocram / attempt1 | 28 | 359.12 / 375.73 | 627.92 | 278.40 / 336.92 | 57.19 | 222.50 / N/A / N/A / 17.00 |
| new-ncpu12-cache2048-swa-nocram-repair2-control / attempt1 | 28 | 364.50 / 383.99 | 616.07 | 281.95 / 340.48 | 56.28 | 222.50 / N/A / N/A / 17.00 |
| new-ncpu12-cache3072-swa-nocram / attempt1 | 28 | 284.72 / 314.80 | 812.37 | 229.17 / 309.55 | 67.46 | 222.50 / N/A / N/A / 17.00 |
| new-ncpu12-off-swa-nocram-repair2 / attempt1 | 28 | 510.38 / 590.85 | 433.66 | 379.14 / 495.69 | 41.67 | 222.50 / N/A / N/A / 17.00 |
| new-ncpu24-cache2048-swa-nocram / attempt1 | 28 | 749.69 / 783.61 | 298.22 | 664.94 / 691.33 | 24.05 | 222.50 / N/A / N/A / 17.00 |
| new-partial14-swa-nocram / attempt1 | 0 | N/A | N/A | N/A | N/A | N/A / N/A / N/A / N/A |
| new-partial14-swa-nocram / attempt2 | 28 | 740.00 / 761.25 | 306.06 | 488.15 / 617.94 | 32.60 | 222.50 / N/A / N/A / 17.00 |

## Actual app native inference: hot_same_pair

First text per app instance is not a guaranteed cold-prefix observation; explicit managed preparation/restore is outside the text-submit wall timer.

| Run | n | prefill_ms med / p95 | prefill_tps med | decode_ms med / p95 | decode_tps med | P eval / C hit / O output / D printed med |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| e4b-product / attempt1 | 4 | 136.69 / 146.42 | 398.65 | 134.41 / 134.74 | 81.84 | 55.00 / N/A / N/A / 12.00 |
| managed-e4b-stock / attempt1 | 4 | 136.39 / 144.70 | 400.45 | 189.81 / 208.04 | 81.81 | 55.00 / 834.00 / 16.50 / N/A |
| new-cpu-moe-cache3072-swa / attempt1 | 4 | 561.68 / 583.22 | 103.91 | 615.92 / 627.66 | 19.48 | 59.00 / N/A / N/A / 13.00 |
| new-cpu-moe-cache3072-swa-nocram / attempt1 | 4 | 559.88 / 582.33 | 103.96 | 614.29 / 627.75 | 19.54 | 59.00 / N/A / N/A / 13.00 |
| new-cpu-moe-off-swa-nocram / attempt1 | 4 | 751.38 / 767.56 | 78.48 | 476.29 / 488.10 | 25.20 | 59.00 / N/A / N/A / 13.00 |
| new-full-swa-nocram / attempt1 | 4 | 123.47 / 131.74 | 477.53 | 129.63 / 130.99 | 92.56 | 59.00 / N/A / N/A / 13.00 |
| new-ncpu12-cache1024-swa-nocram-repair2 / attempt1 | 4 | 285.87 / 296.62 | 204.12 | 353.79 / 361.60 | 33.92 | 59.00 / N/A / N/A / 13.00 |
| new-ncpu12-cache2048-swa-nocram / attempt1 | 4 | 219.91 / 222.46 | 269.38 | 210.13 / 212.56 | 57.11 | 59.00 / N/A / N/A / 13.00 |
| new-ncpu12-cache2048-swa-nocram-repair2-control / attempt1 | 4 | 220.53 / 223.58 | 266.60 | 212.87 / 217.92 | 56.38 | 59.00 / N/A / N/A / 13.00 |
| new-ncpu12-cache3072-swa-nocram / attempt1 | 4 | 167.66 / 169.97 | 356.17 | 173.00 / 179.04 | 69.42 | 59.00 / N/A / N/A / 13.00 |
| new-ncpu12-off-swa-nocram-repair2 / attempt1 | 4 | 363.83 / 377.06 | 161.44 | 302.84 / 312.73 | 39.66 | 59.00 / N/A / N/A / 13.00 |
| new-ncpu24-cache2048-swa-nocram / attempt1 | 4 | 478.35 / 486.82 | 123.20 | 521.71 / 522.25 | 23.00 | 59.00 / N/A / N/A / 13.00 |
| new-partial14-swa-nocram / attempt1 | 0 | N/A | N/A | N/A | N/A | N/A / N/A / N/A / N/A |
| new-partial14-swa-nocram / attempt2 | 4 | 541.36 / 556.66 | 108.34 | 369.22 / 369.66 | 32.50 | 59.00 / N/A / N/A / 13.00 |

## Actual app native inference: pair_alternation

First text per app instance is not a guaranteed cold-prefix observation; explicit managed preparation/restore is outside the text-submit wall timer.

| Run | n | prefill_ms med / p95 | prefill_tps med | decode_ms med / p95 | decode_tps med | P eval / C hit / O output / D printed med |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| e4b-product / attempt1 | 24 | 360.72 / 377.76 | 630.79 | 194.51 / 219.45 | 82.09 | 225.50 / N/A / N/A / 17.00 |
| managed-e4b-stock / attempt1 | 24 | 1052.15 / 1099.24 | 811.03 | 196.41 / 257.07 | 81.82 | 873.50 / 0.00 / 17.00 / N/A |
| new-cpu-moe-cache3072-swa / attempt1 | 24 | 871.85 / 911.51 | 259.82 | 728.83 / 795.02 | 20.96 | 229.50 / N/A / N/A / 17.00 |
| new-cpu-moe-cache3072-swa-nocram / attempt1 | 24 | 861.78 / 891.29 | 262.16 | 727.51 / 754.44 | 21.34 | 229.50 / N/A / N/A / 17.00 |
| new-cpu-moe-off-swa-nocram / attempt1 | 24 | 1032.42 / 1055.12 | 219.70 | 629.05 / 786.59 | 25.59 | 229.50 / N/A / N/A / 17.00 |
| new-full-swa-nocram / attempt1 | 24 | 367.05 / 391.41 | 628.03 | 172.56 / 216.91 | 92.63 | 229.50 / N/A / N/A / 17.00 |
| new-ncpu12-cache1024-swa-nocram-repair2 / attempt1 | 24 | 443.99 / 466.02 | 508.34 | 451.64 / 509.98 | 35.54 | 229.50 / N/A / N/A / 17.00 |
| new-ncpu12-cache2048-swa-nocram / attempt1 | 24 | 362.06 / 375.73 | 631.85 | 279.38 / 336.92 | 57.27 | 229.50 / N/A / N/A / 17.00 |
| new-ncpu12-cache2048-swa-nocram-repair2-control / attempt1 | 24 | 366.19 / 383.99 | 626.02 | 284.35 / 340.48 | 56.27 | 229.50 / N/A / N/A / 17.00 |
| new-ncpu12-cache3072-swa-nocram / attempt1 | 24 | 285.78 / 314.80 | 821.73 | 238.38 / 309.55 | 67.41 | 229.50 / N/A / N/A / 17.00 |
| new-ncpu12-off-swa-nocram-repair2 / attempt1 | 24 | 517.71 / 590.85 | 446.18 | 385.65 / 495.69 | 41.91 | 229.50 / N/A / N/A / 17.00 |
| new-ncpu24-cache2048-swa-nocram / attempt1 | 24 | 751.20 / 783.61 | 299.74 | 667.32 / 691.33 | 24.09 | 229.50 / N/A / N/A / 17.00 |
| new-partial14-swa-nocram / attempt1 | 0 | N/A | N/A | N/A | N/A | N/A / N/A / N/A / N/A |
| new-partial14-swa-nocram / attempt2 | 24 | 742.61 / 761.25 | 307.73 | 489.58 / 617.94 | 32.63 | 229.50 / N/A / N/A / 17.00 |

## Actual app native inference: three_slots

First text per app instance is not a guaranteed cold-prefix observation; explicit managed preparation/restore is outside the text-submit wall timer.

| Run | n | prefill_ms med / p95 | prefill_tps med | decode_ms med / p95 | decode_tps med | P eval / C hit / O output / D printed med |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| e4b-product / attempt1 | 0 | N/A | N/A | N/A | N/A | N/A / N/A / N/A / N/A |
| managed-e4b-stock / attempt1 | 0 | N/A | N/A | N/A | N/A | N/A / N/A / N/A / N/A |
| new-cpu-moe-cache3072-swa / attempt1 | 0 | N/A | N/A | N/A | N/A | N/A / N/A / N/A / N/A |
| new-cpu-moe-cache3072-swa-nocram / attempt1 | 0 | N/A | N/A | N/A | N/A | N/A / N/A / N/A / N/A |
| new-cpu-moe-off-swa-nocram / attempt1 | 0 | N/A | N/A | N/A | N/A | N/A / N/A / N/A / N/A |
| new-full-swa-nocram / attempt1 | 0 | N/A | N/A | N/A | N/A | N/A / N/A / N/A / N/A |
| new-ncpu12-cache1024-swa-nocram-repair2 / attempt1 | 0 | N/A | N/A | N/A | N/A | N/A / N/A / N/A / N/A |
| new-ncpu12-cache2048-swa-nocram / attempt1 | 0 | N/A | N/A | N/A | N/A | N/A / N/A / N/A / N/A |
| new-ncpu12-cache2048-swa-nocram-repair2-control / attempt1 | 0 | N/A | N/A | N/A | N/A | N/A / N/A / N/A / N/A |
| new-ncpu12-cache3072-swa-nocram / attempt1 | 0 | N/A | N/A | N/A | N/A | N/A / N/A / N/A / N/A |
| new-ncpu12-off-swa-nocram-repair2 / attempt1 | 0 | N/A | N/A | N/A | N/A | N/A / N/A / N/A / N/A |
| new-ncpu24-cache2048-swa-nocram / attempt1 | 0 | N/A | N/A | N/A | N/A | N/A / N/A / N/A / N/A |
| new-partial14-swa-nocram / attempt1 | 0 | N/A | N/A | N/A | N/A | N/A / N/A / N/A / N/A |
| new-partial14-swa-nocram / attempt2 | 0 | N/A | N/A | N/A | N/A | N/A / N/A / N/A / N/A |

## True managed E4B: separate prefix evaluation / zero-eval restore windows

Native counter deltas bracket completed language-apply operations only; CLI preparation wall includes provider/settings/restore overhead. Unbracketed initial enable is explicitly N/A. Zero-evaluation windows can be resident or restored prefixes; TPS unavailable rather than inflated. The prefix one-output-token request has no normal decode loop; native timer floors are not throughput benchmarks.

| Run | n | prefill_ms med / p95 | prefill_tps med | decode_ms med / p95 | decode_tps med | P eval / C hit / O output / D printed med |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| evaluating_prefix_windows | 3 | 979.38 / 1004.02 | 821.70 | 0.00 / 0.00 | N/A | 819.00 / 0.00 / 1.00 / N/A |
| zero_eval_apply_or_restore_windows | 28 | 0.00 / 0.00 | N/A | 0.00 / 0.00 | N/A | 0.00 / 0.00 / 0.00 / N/A |

## Bounded memory-attribution diagnostic work (not ranking runs)

| Run | n | prefill_ms med / p95 | prefill_tps med | decode_ms med / p95 | decode_tps med | P eval / C hit / O output / D printed med |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| new-ncpu12-off-swa-nocram-repair2 | 3 | 1349.40 / 2376.92 | 672.89 | 280.52 / 518.31 | 42.78 | 908.00 / 0.00 / 13.00 / 13.00 |
| new-ncpu12-cache2048-swa-nocram-repair2-control | 3 | 923.47 / 1960.72 | 983.24 | 238.75 / 428.68 | 50.26 | 908.00 / 0.00 / 13.00 / 13.00 |

## Missing / failed records

- new-cpu-moe-cache4096: RuntimeError: Server exited before readiness: 1
- new-partial14-swa-nocram: CLI stop observed original process exit/endpoint disappearance; no separate host-release snapshot was recorded on this failed attempt. Raw full probe evidence is preserved only in the external owned artifact directory. Unrelated process/device catalogs and personal paths are intentionally omitted.

Per-run sidecars are under results/timings/; originals are not rewritten. Complete distributions/raw native observations are in results/timing-summary.json. True managed prefix-preparation/restore wall windows and any available native metric snapshots remain in its original app record, separately from translation durations.
