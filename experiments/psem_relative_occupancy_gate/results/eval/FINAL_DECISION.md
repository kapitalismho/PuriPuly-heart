# PSEM relative-occupancy EVAL decision

## Decision

`psem-relative-occupancy-v0` is information-sufficient, and Streaming Sortformer should receive the first true V3 fine-tuning run.

This is a representation-and-training recommendation, not a production-readiness decision. The frozen Sortformer pipeline does not yet establish a production win: its causal lower-bound contamination estimate improves on the VAD-only proxy only by accepting hundreds of speaker-induced cuts per active-speech hour, while its fail-closed unknown-time upper bound remains worse than the VAD-only upper bound. LS-EEND is faster, but the frozen checkpoint exposes too little usable `other_present` information and its causal anchor is slower and less safe.

All PSEM-STRATEGY-DATA-v2 roles are now development-known for the next program. V3 must create a fresh DEV/EVAL holdout; V2 may only be folded into V3 TRAIN.

## Evidence boundary

- EVAL was opened once with the frozen nineteen-source manifest, SHA-256 `38260d1dddcebc546d9b9b5672d3185849fc433a9027bdf2831094196c2a2b30`.
- The comparison covers 36,085.618 seconds of audio and 6.725 active-speech hours.
- DEV selection remained frozen at SHA-256 `f181563ec55908570487f50817c1dfff47b3678af35b590e70b394282bb5d489`.
- The corrected DEV diagnostic retained the exact selected settings for both families. Its summary file SHA-256 is `911c47873c61b518ee51d79a4007d4321d529188ac791863c6cb16fb29202c59`; no DEV artifact, setting, or EVAL selection was mutated.
- Sortformer used the pinned `transcribe.cpp` commit `d42c3bbdfa2f63c37e5891e27de47a612d62f221`, model revision `7ef0c15dc8f9d717e9d24fac29a6e6551e9c6ddf`, Q8 model SHA-256 `a5dacdc650790266c7a362e54e6bf51952015487edaa606c4e11632bc32442a9`, and the experimental Vulkan backend.
- LS-EEND used revision `cc40a1e1242c148fbbc15c132e43b8ac15056e53`, ONNX model SHA-256 `5a2b813ffe41170e40d0fc08a6eb1699e579e377af30c7962d07885608a6aa77`, and `CPUExecutionProvider`.
- The owner-authorized terminal recovery consumed recovery_4 exactly once and regenerated the derived aggregates from unchanged traces and frozen settings. This V2 EVAL is issue-local recovery-qualified evidence, not a pristine fresh holdout.
- Canonical Gate 1 and Gate 2 ledgers contain 152 rows each, SHA-256 `920a967b507ce3ad10f6e44b05497fd5388bc382644a6cef588c6fade932c359` and `135b103d26e6ebfd262686d809e874c2282018ebe8341801d8b192586ec2c337`. Event-level compute lag is explicitly unavailable because the frozen receipts bind only aggregate runtime; no event timing was inferred from it.
- `eval_verification.json`, SHA-256 `1a3e7b445c981d9e113df2c902521ed563f0b48a5fe176e6a6582c6b2a47f6c7`, independently regenerated all six canonical artifacts, replayed lifecycle opportunities, ordering, annotations, topology and fail-closed exposure, and reports `passed: true`.

The product tables report the measured lower estimate and, where relevant, the upper bound that treats fail-closed unknown active-speech time as contaminated. No scalar contamination-versus-cut cost was authorized.

## Frozen posterior summary

Perfect GT relative occupancy is the primitive upper bound of 1.0. Gate 2 primitive metrics are conditional on periods in which the causal system is anchored, so their smaller evaluation set must not be interpreted as an unconditional improvement over Gate 1.

| Family and gate | Anchor AUPRC | Other AUPRC | Four-state macro-F1 | False `OTHER_ONLY` inside GT `ANCHOR_ONLY` | Missed `OTHER_ONLY` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sortformer Gate 1, oracle anchor | 0.9855 | 0.8441 | 0.7925 | 70.2 s | 317.7 s |
| Sortformer Gate 2, causal anchor | 0.9510 | 0.8249 | 0.8024 | 165.5 s | 796.1 s |
| LS-EEND Gate 1, oracle anchor | 0.7372 | 0.1243 | 0.2546 | 459.8 s | 827.9 s |
| LS-EEND Gate 2, causal anchor | 0.6848 | 0.3819 | 0.3103 | 53.9 s | 1,562.1 s |

Relative to the perfect primitive upper bound, Sortformer Gate 1 loses 0.0145 anchor AUPRC, 0.1559 other AUPRC, and 0.2075 macro-F1. LS-EEND loses 0.2628 anchor AUPRC, 0.8757 other AUPRC, and 0.7454 macro-F1. The decisive difference is `other_present`: the Sortformer posterior contains useful relative-occupancy evidence, while this LS-EEND checkpoint largely does not.

High primitive quality does not by itself imply good logical segmentation. With a 100 ms replacement confirmation, the GT oracle produces 75.1 contamination seconds per active-speech hour. Sortformer Gate 1 produces 2,004.3 and LS-EEND Gate 1 produces 2,124.2. Their lower cut rates are accompanied by 3,570 and 4,044 missed replacements respectively, so fewer cuts are not evidence of a better product result.

## Causal product frontier

| Family | Confirm | Contamination s / active-speech h | Upper-bound contamination s / h | Cuts / h | Replacement delay p50 / p90 | False cuts | Missed replacements |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sortformer | 100 ms | 1,175.3 | 2,255.5 | 575.3 | 922.5 / 1,154.5 ms | 634 | 1,626 |
| Sortformer | 200 ms | 1,055.0 | 2,060.2 | 531.0 | 1,010.5 / 1,236.1 ms | 624 | 1,304 |
| Sortformer | 300 ms | 1,036.7 | 1,967.4 | 484.1 | 1,090.5 / 1,332.5 ms | 693 | 1,046 |
| Sortformer | 500 ms | 1,042.9 | 1,836.2 | 378.7 | 1,322.5 / 1,572.5 ms | 531 | 782 |
| LS-EEND | 100 ms | 2,418.8 | 4,526.2 | 220.2 | 180.0 / 552.4 ms | 1,114 | 4,494 |
| LS-EEND | 200 ms | 2,380.8 | 4,467.3 | 209.7 | 260.0 / 640.6 ms | 1,115 | 3,956 |
| LS-EEND | 300 ms | 2,438.2 | 4,509.0 | 200.7 | 355.0 / 747.0 ms | 1,099 | 3,358 |
| LS-EEND | 500 ms | 2,570.9 | 4,621.7 | 183.5 | 630.5 / 953.1 ms | 1,012 | 2,576 |

The VAD-only lifecycle proxy is 1,418.7 lower-estimate and 1,486.3 upper-bound contamination seconds per active-speech hour with zero speaker-induced cuts. The GT oracle spans 75.1–294.9 contamination seconds and 722.8–416.0 cuts per hour over the fixed 100–500 ms confirmation family.

There is no strict cross-family dominance without a cut cost. Sortformer occupies the low-contamination end; its 300 and 500 ms points are its useful causal Pareto points. LS-EEND occupies the lower-cut end, but every LS-EEND causal point has worse lower and upper contamination than the no-speaker-cut baseline. Sortformer at 300 ms preserves 71.8% of overlap-return episodes and succeeds on 49.3% of overlap-takeover episodes; LS-EEND at 300 ms preserves 85.4% but succeeds on only 3.8% of takeovers. This is a fragmentation-versus-replacement trade-off, not a hidden scalar win.

## Anchor availability and safety

| Family | Enrollment p50 | Enrollment p90 | Within 1.0 s | Within 1.5 s | Wrong-anchor rate | Failure rate | Cascade maximum / p90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sortformer | 1,003.5 ms | 1,631.9 ms | 47.88% | 85.46% | 15.66% at 100 ms; 12.78% at 500 ms | 3.71–4.23% | 2–3 / about 1 |
| LS-EEND | 1,343.5 ms | 3,200.0 ms | 25.58% | 47.89% | 14.83–15.38% | 28.64–29.64% | 5–6 / 2 |

Approximately one-second Sortformer availability is plausible at the median but not reliable enough to claim a one-second product guarantee. Counting every lifecycle opportunity exposes a 3.71–4.23% no-anchor timeout rate. LS-EEND does not make a practical one-second case: almost three quarters of eligible enrollments miss one second, fewer than half complete within 1.5 seconds, and 28.64–29.64% fail to enroll.

Both adapters report zero explicit slot-loss events in this run. That does not erase wrong enrollment. Sortformer produces 110–171 false cuts after wrong anchors, while LS-EEND produces 203–223, and the LS-EEND cascade maximum is twice the worst Sortformer maximum. Sortformer is therefore the safer causal family under this baseline. The lower corrected wrong-anchor percentages do not mean the systems improved; previously omitted failed lifecycle opportunities are now present in the denominator and reported separately.

## Timing and compute

Sortformer has 80 ms native frames and 1,040 ms recorded algorithmic buffering. Its causal model-evidence availability is 792.5 ms p50 and 992.5 ms p90, and the experimental Vulkan pass measured wall RTF 0.0477. LS-EEND has 100 ms frames, no separately recorded algorithmic buffer, 54 ms p50 and 66 ms p90 evidence availability, and CPU wall RTF 0.0252. LS-EEND is faster, but its representation and anchor failures outweigh that advantage for the V3 family decision.

The Vulkan figures establish only this experiment's frozen inference backend. They do not establish final Vulkan deployment, quantization, device coverage, or production readiness. Third-party model and runtime licensing was not adjudicated by this experiment and must be cleared before a V3 or production commitment.

## Answers to the ten required questions

1. **Did Gate 0 validate the ontology?** Yes. All synthetic fixtures and all mandatory natural DEV topologies passed with no event boundary inside a mask. The two occupancies, explicit anchor lifecycle, and short causal history are information-sufficient.
2. **What remains an ontology or policy problem?** No observed mandatory topology requires another primitive. The unresolved policy choices are the contamination-versus-fragmentation cost, replacement confirmation duration, and integration with the actual production VAD lifecycle. Product-VAD replay remains deferred; GT speech/non-speech was only the authorized lifecycle proxy.
3. **How far did Sortformer Gate 1 fall below GT?** Primitive loss is modest for `anchor_present` and material for `other_present`: AUPRC 0.9855 and 0.8441, with macro-F1 0.7925. Product loss is much larger: at 100 ms, contamination rises from 75.1 to 2,004.3 s/h and 3,570 of 4,861 reference replacements are missed.
4. **How far did LS-EEND Gate 1 fall below GT?** Severely: anchor AUPRC 0.7372, other AUPRC 0.1243, macro-F1 0.2546. At 100 ms it produces 2,124.2 contamination s/h and misses 4,044 of 4,861 replacements.
5. **What additional loss came from causal anchoring?** Sortformer's conditionally scored primitive AUPRC is 0.9510/0.8249, but wrong anchors are 12.78–15.66%, no-anchor timeouts are 3.71–4.23%, false `OTHER_ONLY` inside true anchor-only speech is 165.5 s, and the causal frontier trades roughly twice the oracle cut rate for lower measured contamination. LS-EEND's causal path has 14.83–15.38% wrong anchors, 28.64–29.64% no-anchor timeouts, a 3.200 s enrollment p90, 1,562.1 s of missed `OTHER_ONLY` on anchored intervals, and far more fail-closed unknown time. Gate 2 is not a monotonic degradation statistic because it changes anchor coverage and cut frequency; the full Pareto and safety metrics are the authoritative comparison.
6. **Is roughly one-second anchor availability plausible?** Borderline for Sortformer, no for LS-EEND. Sortformer is 1,003.5/1,631.9 ms p50/p90 with 47.88% within 1 s and 85.46% within 1.5 s. LS-EEND is 1,343.5/3,200.0 ms with 25.58% and 47.89%.
7. **Which family has the best causal contamination-versus-cut frontier?** Neither strictly dominates without a scalar cut cost. Sortformer is the only family that lowers measured contamination below the VAD-only lower estimate, at a high fragmentation cost. LS-EEND offers fewer cuts but worse contamination than the no-speaker-cut baseline. For the issue's priority ordering, the useful frontier is Sortformer.
8. **Which family is safer under wrong or uncertain anchors?** Sortformer. It has lower wrong-anchor and failure rates, shorter cascades, and fewer false cuts after wrong anchors. Both systems fail closed, but LS-EEND spends substantially more time without a trustworthy anchor.
9. **Which family should receive the first true V3 run?** Streaming Sortformer. The recommendation is based on primitive headroom, causal contamination capability, enrollment availability, and cascade safety. It is not a deployment approval, and it remains subject to licensing and deployment-feasibility checks.
10. **What should V3 train?** Fine-tune the load-bearing online Sortformer encoder/activity/slot path with a causal anchor-conditioned relative-occupancy objective that emits `anchor_present` and `other_present` on the source-time-aligned, mask-aware grid. Train stable anchor enrollment/association and explicit fail-closed uncertainty as part of that path. Keep persistence and hysteresis in the deterministic decoder. Do not train a direct `handoff_confirmed` target, a tiny frozen-backbone event verifier, global speaker identity, or a scratch model.

## Final disposition

Proceed to one evidence-bounded V3 Sortformer fine-tuning program after creating a fresh split and clearing licensing/runtime constraints. Do not spend the first V3 run on this LS-EEND checkpoint, do not deploy either frozen pipeline from this result, and do not reuse V2 as V3 DEV or EVAL.
