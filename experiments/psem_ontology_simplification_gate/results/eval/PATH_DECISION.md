# PSEM ontology simplification path decision

## Decision

Carry `psem-simple-anchor-v0` forward as the current neural contract, with an explicit valid-anchor lifecycle and a speech gate whose live confirmation support is distinct from ASR pre-roll. The perfect-state replay proves that `speech && !anchor_present` preserves every required issue-97 Gate 0 replacement action. The extra `anchor_overlap_present` state is information-valid, but the current frozen `min(p_anchor, p_nonanchor_max)` proxy does not produce a better product frontier than Simple Anchor.

The next ML path is Sortformer task adaptation before compression. Direct distillation of the frozen Sortformer is premature because its oracle-anchor Simple Anchor result remains far from the perfect-state product oracle, dangerous overlap dropout is material, and the production-VAD sensitivity exposes an independent integration failure. The current LS-EEND checkpoint should not be revived.

This is a development-path decision on development-known V2 evidence, not a production-readiness claim.

## Evidence boundary

- All challenger views were derived offline from the exact issue-97 cached posterior traces. No new speaker-model inference was performed.
- DEV contains 10 sources and EVAL contains 19 sources. EVAL covers 36,085.618 seconds of audio and 6.725 active-speech hours.
- Sortformer retains the issue-97 Q8 Vulkan trace pin. LS-EEND retains the issue-97 `L-AMI` CPU `CPUExecutionProvider` trace pin.
- The fixed persistence family is 100, 200, 300, and 500 ms. Candidate B's primary thresholds are Sortformer `anchor=0.5, overlap=0.35` and LS-EEND `anchor=0.7, overlap=0.3`.
- EVAL is recovery-qualified and development-known. It is comparative evidence, not a fresh holdout.
- Production VAD was replayed deterministically on all 29 exact V2 audio sources with bundled Silero VAD 6.2.1, model SHA-256 `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`, and `CPUExecutionProvider`. No VAD setting was tuned.
- The repaired artifact set contains 192 frontier rows per role, 720 DEV and 1,368 EVAL paired-session rows, 72 bootstrap rows per role, 16 exact R0 reconstruction checks per role, and 32 production-VAD sensitivity rows per role.
- Oracle-anchor mapping covers 2,992 of 3,068 DEV episodes and 5,786 of 5,972 EVAL episodes for both families. The 76 DEV and 186 EVAL unmapped episodes are explicit fail-closed uncertainty, covering 58.428/148.955 seconds of episode support and 13.202/33.536 seconds of unmasked active speech.

## Ontology and proxy verdicts

| Verdict | Disposition | Evidence |
| --- | --- | --- |
| Simple Anchor ontology | GREEN | Zero action mismatches against the shared Gate 0 oracle at every persistence point. |
| B-ONTOLOGY | GREEN | Anchor+Overlap also has zero action mismatches; the state is semantically valid. |
| B-PROXY | YELLOW | The cheap proxy provides a conservative false-cut trade but does not dominate Simple Anchor or rescue a perfect-state loss. |
| Current neural contract | Simple Anchor | `other_present` and an overlap head are not product-action requirements under perfect state. |

DEV Gate S0 covered 3,420 anchor episodes and 2,726 reference actions at 100 ms, 3,068/2,374 at 200 ms, 2,732/2,037 at 300 ms, and 2,306/1,592 at 500 ms. Both challengers produced zero action mismatches in every cell.

## Primitive evidence

The S2 rows below are conditional on the already-realized issue-97 lifecycle and must not be read as native simplified-runtime quality.

| Family / arm | Anchor AUPRC | A-only recall | A+other recall | Anchor-absence recall | Anchor FP on absent active speech |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sortformer S1 oracle anchor | 0.9855 | 96.4% | 89.9% | 82.7% | 155.6 s |
| Sortformer S2 fixed lifecycle, 300 ms | 0.9489 | 95.4% | 84.0% | 76.2% | 661.5 s |
| LS-EEND S1 oracle anchor | 0.7372 | 43.5% | 49.6% | 60.3% | 357.9 s |
| LS-EEND S2 fixed lifecycle, 300 ms | 0.6906 | 94.9% | 88.2% | 26.4% | 1,280.2 s |

Sortformer S1 duration-weighted anchor-dropout inside GT A+other is 10.06%, 4.49%, and 1.68% for runs of at least 100, 300, and 500 ms. The corresponding affected-episode fractions are 30.60%, 6.97%, and 1.77%. In the fixed-lifecycle S2 300 ms arm, the duration-weighted values rise to 15.96%, 7.56%, and 3.19%.

LS-EEND S1 duration-weighted A+other dropout is 50.41%, 43.73%, and 30.54% at those horizons. Its apparently smaller conditional S2 dropout is selection-biased by the old lifecycle and does not repair the weak oracle-anchor acoustic result.

The anchor-independent global-overlap diagnostic reinforces the family split. At threshold 0.5, Sortformer has AUPRC 0.785, precision 0.813, recall 0.629, short-backchannel recall 0.680, and 280.6 false-overlap seconds. LS-EEND has AUPRC 0.112, precision 0.236, recall 0.142, short-backchannel recall 0.119, and 891.5 false-overlap seconds. LS-EEND global overlap is not stronger than its issue-97 `other_present` evidence.

Global-overlap scoring is complete for Sortformer. LS-EEND exposes 84 invalid DEV cells and 171 invalid EVAL cells, representing 8.4/17.1 seconds of support and 0.7/2.1 seconds of active speech; the scored-support fractions remain 99.960% and 99.951%. These cells are reported as invalid coverage rather than silently dropped.

## Product frontier

`R0` is issue-97 relative occupancy, `A0` is Simple Anchor, and `B0` is the primary fail-closed Anchor+Overlap proxy. The full 100–500 ms frontier is authoritative; 300 ms is shown first only as a compact representative comparison. `Unknown` is the unanchored-or-uncertain fraction of unmasked active speech. Upper-bound contamination treats fail-closed unknown active speech as contaminated.

| Family / arm / candidate | Contam s/h | Upper s/h | Cuts/h | False | Missed | Delay p50/p90 ms | Boundary error p50/p90 ms | Takeover | Return | Unknown | Wrong anchor / cascade max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sortformer S1 R0 | 2,036.0 | 2,488.1 | 218.9 | 561 | 2,698 | 972.5/1,212.5 | -36/0 | 2.2% | 92.0% | 13.0% | n/a |
| Sortformer S1 A0 | 1,113.2 | 1,607.1 | 410.5 | 549 | 1,397 | 1,072.5/1,284.5 | 0/0 | 11.6% | 74.0% | 14.2% | n/a |
| Sortformer S1 B0 | 1,168.8 | 1,646.5 | 382.4 | 457 | 1,494 | 1,075.5/1,288.5 | 0/0 | 12.0% | 76.5% | 13.7% | n/a |
| Sortformer S2 R0 | 1,036.7 | 1,967.4 | 484.1 | 693 | 1,046 | 1,090.5/1,332.5 | 40/200 | 49.3% | 71.8% | 26.7% | 14.4% / 3 |
| Sortformer S2 A0 | 1,272.6 | 2,203.2 | 416.2 | 540 | 1,350 | 1,094.5/1,312.5 | 0/140 | 43.7% | 69.5% | 26.7% | 14.4% / 2 |
| Sortformer S2 B0 | 1,291.8 | 2,234.6 | 402.2 | 493 | 1,397 | 1,107/1,330.3 | 0/162.5 | 48.1% | 71.1% | 27.0% | 14.4% / 2 |
| LS-EEND S1 R0 | 2,339.2 | 2,919.1 | 150.9 | 632 | 3,226 | 819/888 | 500/500 | 7.2% | 92.7% | 16.6% | n/a |
| LS-EEND S1 A0 | 1,212.6 | 3,822.1 | 593.3 | 2,255 | 1,874 | 836/890 | 500/500 | 42.9% | 62.6% | 74.9% | n/a |
| LS-EEND S1 B0 | 1,276.4 | 3,854.8 | 571.9 | 2,213 | 1,976 | 840/890 | 500/500 | 40.5% | 63.6% | 74.0% | n/a |
| LS-EEND S2 R0 | 2,438.2 | 4,509.0 | 200.7 | 1,099 | 3,358 | 355/747 | 0/428 | 3.8% | 85.4% | 59.4% | 15.3% / 6 |
| LS-EEND S2 A0 | 2,363.5 | 4,434.3 | 155.5 | 706 | 3,269 | 451/768 | 132/446.6 | 3.8% | 86.1% | 59.4% | 15.3% / 3 |
| LS-EEND S2 B0 | 2,444.7 | 4,530.0 | 104.5 | 445 | 3,351 | 446/774.6 | 115/444.6 | 2.4% | 93.0% | 59.8% | 15.3% / 3 |

The perfect-state oracle at 300 ms is 196.3 contamination s/h and 536.6 cuts/h. Sortformer S1 A0 therefore improves substantially over frozen R0 but still has 5.7 times the oracle contamination and misses 1,397 replacements.

### All persistence points

| Family / arm | Candidate | 100 ms contam/cuts/false/missed | 200 ms | 300 ms | 500 ms |
| --- | --- | --- | --- | --- | --- |
| Sortformer S1 | R0 | 2,004.3 / 269.6 / 522 / 3,570 | 1,983.4 / 235.8 / 494 / 3,159 | 2,036.0 / 218.9 / 561 / 2,698 | 2,068.6 / 167.0 / 403 / 2,078 |
| Sortformer S1 | A0 | 1,454.4 / 607.0 / 1,579 / 2,358 | 1,220.3 / 491.6 / 847 / 1,792 | 1,113.2 / 410.5 / 549 / 1,397 | 1,086.9 / 311.2 / 297 / 1,002 |
| Sortformer S1 | B0 | 1,495.1 / 579.3 / 1,498 / 2,463 | 1,268.8 / 462.1 / 760 / 1,903 | 1,168.8 / 382.4 / 457 / 1,494 | 1,165.7 / 288.5 / 219 / 1,077 |
| Sortformer S2 fixed | R0 | 1,175.3 / 575.3 / 634 / 1,626 | 1,055.0 / 531.0 / 624 / 1,304 | 1,036.7 / 484.1 / 693 / 1,046 | 1,042.9 / 378.7 / 531 / 782 |
| Sortformer S2 fixed | A0 | 1,671.2 / 547.2 / 1,358 / 2,539 | 1,383.3 / 476.7 / 780 / 1,825 | 1,272.6 / 416.2 / 540 / 1,350 | 1,164.5 / 327.1 / 363 / 961 |
| Sortformer S2 fixed | B0 | 1,687.6 / 534.3 / 1,310 / 2,578 | 1,402.0 / 461.7 / 737 / 1,883 | 1,291.8 / 402.2 / 493 / 1,397 | 1,165.8 / 317.0 / 316 / 982 |
| LS-EEND S1 | R0 | 2,124.2 / 243.7 / 822 / 4,044 | 2,181.9 / 192.7 / 733 / 3,688 | 2,339.2 / 150.9 / 632 / 3,226 | 2,660.3 / 106.3 / 601 / 2,684 |
| LS-EEND S1 | A0 | 1,027.6 / 779.3 / 2,562 / 2,182 | 1,131.6 / 685.8 / 2,523 / 2,162 | 1,212.6 / 593.3 / 2,255 / 1,874 | 2,423.1 / 444.9 / 2,618 / 2,424 |
| LS-EEND S1 | B0 | 1,097.3 / 751.9 / 2,513 / 2,317 | 1,210.1 / 659.8 / 2,458 / 2,272 | 1,276.4 / 571.9 / 2,213 / 1,976 | 2,477.7 / 427.8 / 2,549 / 2,470 |
| LS-EEND S2 fixed | R0 | 2,418.8 / 220.2 / 1,114 / 4,494 | 2,380.8 / 209.7 / 1,115 / 3,956 | 2,438.2 / 200.7 / 1,099 / 3,358 | 2,570.9 / 183.5 / 1,012 / 2,576 |
| LS-EEND S2 fixed | A0 | 2,351.8 / 190.3 / 823 / 4,404 | 2,317.2 / 170.1 / 746 / 3,853 | 2,363.5 / 155.5 / 706 / 3,269 | 2,451.7 / 133.2 / 593 / 2,495 |
| LS-EEND S2 fixed | B0 | 2,416.9 / 136.6 / 552 / 4,494 | 2,378.3 / 116.7 / 471 / 3,937 | 2,444.7 / 104.5 / 445 / 3,351 | 2,562.1 / 87.3 / 368 / 2,579 |

The cells are `contamination s/h / cuts/h / false cuts / missed replacements`. No scalar contamination-versus-fragmentation cost was authorized.

At 300 ms, the paired 10,000-resample source-session bootstrap for Sortformer S1 A0 minus R0 is contamination `[-1110.9, -766.8]` s/h, false cuts per session `[-6.7, +4.6]`, missed replacements per session `[-87.1, -52.2]`, and overlap-takeover rate `[+0.071, +0.117]`. The improvement is not a pooled-number accident. For the fixed-lifecycle S2 counterfactual, A0 minus R0 is worse on contamination `[+168.7, +319.5]` s/h and missed replacements `[+11.3, +21.1]` per session while reducing false cuts `[-13.2, -3.5]` per session.

The directly paired EVAL S1 cross-family intervals also reject treating LS-EEND simplification as competitive. For LS-EEND minus Sortformer at 300 ms, A0 contamination spans `[-154.2, +341.7]` s/h, but false cuts per session are `[+61.3, +122.9]` and missed replacements are `[+4.7, +47.4]`. B0 contamination similarly spans `[-172.7, +373.6]`, while false cuts are `[+62.8, +125.5]` and misses are `[+4.3, +47.9]` per session.

## Production-VAD sensitivity

The exact production peer-VAD support includes 500 ms pre-roll plus committed chunks through speech end and excludes trailing hangover. At 300 ms on EVAL:

| Family / arm / candidate | GT-gate contam s/h | Production-VAD contam s/h | Delta | Production cuts/h | Cut delta | False delta | Missed delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sortformer S1 A0 | 1,113.2 | 1,939.7 | +826.5 | 546.6 | +136.1 | +2,029 | +1,114 |
| Sortformer S1 B0 | 1,168.8 | 1,979.7 | +810.9 | 532.5 | +150.0 | +2,073 | +1,064 |
| Sortformer S2 fixed A0 | 1,272.6 | 1,962.1 | +689.5 | 521.9 | +105.7 | +1,696 | +985 |
| Sortformer S2 fixed B0 | 1,291.8 | 1,964.6 | +672.8 | 517.9 | +115.7 | +1,719 | +941 |
| LS-EEND S1 A0 | 1,212.6 | 1,292.9 | +80.3 | 599.2 | +5.9 | +37 | -3 |
| LS-EEND S1 B0 | 1,276.4 | 1,349.7 | +73.3 | 578.4 | +6.5 | +24 | -20 |

The Sortformer degradation repeats on DEV and across all persistence points. Candidate B does not repair it. This is evidence against treating the current pre-roll-as-speech integration as product-ready, not evidence that the Simple Anchor ontology is information-insufficient. Production-VAD replay on development-known V2 is still not a deployment claim.

## Required answers

1. **With perfect states, is Simple Anchor information-sufficient?** Yes. It exactly reproduces the shared issue-97 Gate 0 product actions at all four persistence points with zero mismatches.
2. **How much product information is lost by collapsing A and A+B?** Zero required replacement-action information was lost in the tested ontology. The overlap label itself is discarded, but it is not needed to decide whether qualifying non-anchor solo speech should cut.
3. **How often does frozen Sortformer sustain dangerous false-negative runs inside GT overlap?** In S1, duration-weighted A+other dropout is 10.06%, 4.49%, and 1.68% for runs at least 100, 300, and 500 ms. The affected-episode fractions are 30.60%, 6.97%, and 1.77%. This is material at 300 ms and not negligible at 500 ms.
4. **Does Simple Anchor improve the frozen Sortformer frontier relative to issue 97?** Under oracle anchor and GT speech, yes. At 300 ms it cuts contamination from 2,036.0 to 1,113.2 s/h, missed replacements from 2,698 to 1,397, and takeover success rises from 2.2% to 11.6%, with nearly unchanged false cuts. The fixed-lifecycle S2 counterfactual is a different trade: A0 reduces cuts and false cuts but raises contamination and misses relative to R0.
5. **Does adding only anchor-overlap recover any perfect-state Simple Anchor loss?** No recovery is needed: both are exactly sufficient. Anchor+Overlap retains a semantically valid state but adds no required Gate 0 action information.
6. **Does the frozen `min()` proxy realize Anchor+Overlap utility?** Only as a conservative trade, not as a superior frontier. At Sortformer S1/300 ms, B0 reduces false cuts from 549 to 457 and cuts/h from 410.5 to 382.4, but raises contamination from 1,113.2 to 1,168.8 and missed replacements from 1,397 to 1,494. B-PROXY is YELLOW, while B-ONTOLOGY remains GREEN.
7. **How much LS-EEND issue-97 failure disappears without `other_present`?** A substantial oracle-anchor contamination/miss component disappears, but it converts into severe fragmentation and unknown exposure. At 300 ms, contamination improves from 2,339.2 to 1,212.6 s/h and misses from 3,226 to 1,874, while cuts rise from 150.9 to 593.3/h, false cuts from 632 to 2,255, and unknown exposure from 16.6% to 74.9%.
8. **Is LS-EEND global-overlap evidence stronger than its old `other_present` evidence?** No. Global-overlap AUPRC is 0.112 versus issue-97 `other_present` AUPRC 0.124 in S1, with only 0.142 recall at threshold 0.5 and 0.119 short-backchannel recall.
9. **Under oracle anchor, is LS-EEND's main remaining problem acoustic anchor detection or causal association?** Acoustic anchor detection is already inadequate: anchor AUPRC is 0.737, A-only recall 43.5%, overlap recall 49.6%, and 300 ms overlap dropout occupies 43.73% of support. Causal association is also poor in issue 97, but it is not the sole explanation.
10. **Did the reused lifecycle depend on the old ontology?** Yes. Enrollment rejects a singleton candidate when another alive slot exceeds `other_low_threshold`, and episode termination uses old `OTHER_ONLY`. Every S2 conclusion here is therefore a fixed-issue-97-lifecycle counterfactual, not a native end-to-end Simple Anchor or Anchor+Overlap runtime claim.
11. **Was production VAD unavailable?** No. It was replayed deterministically on every DEV and EVAL source. Its severe Sortformer degradation means the current integration is not end-to-end viable; even so, the replay remains development-known sensitivity evidence rather than production readiness.
12. **Which ontology should be carried forward?** Simple Anchor, with an explicit valid-anchor lifecycle and a confirmed-speech gate. Keep Anchor+Overlap as a viable fallback ontology, not as the current frozen proxy contract. Do not retain `other_present` as a neural requirement solely for segmentation.
13. **Which path is GREEN / YELLOW / RED?** Path A, frozen Sortformer to scratch compact-student KD: **RED**. Path B, Sortformer task adaptation before KD: **GREEN**, subject to first resolving the speech-support integration boundary. Path C1, reuse this frozen LS-EEND checkpoint: **RED**. Path C2, future attractor-family ideas: **YELLOW** because this checkpoint lowers the prior but does not test a purpose-built compact architecture.
14. **What is the single cheapest next discriminative experiment?** Reuse the same cached speaker posteriors and recorded production-VAD events for one deterministic integration replay that treats ASR pre-roll as context but allows replacement confirmation only on live committed-VAD speech support. Attribute false cuts to pre-roll, committed speech, and VAD false-positive spans. This requires no neural or VAD inference, no threshold tuning, and decides whether the immediate blocker is the integration contract or the frozen Sortformer representation before spending a training run.

## Limits

This experiment does not prove that a trained overlap head is easy, that Sortformer adaptation will succeed, that a student can match its teacher, that any frozen system generalizes beyond V2, or that a poor LS-EEND checkpoint rules out attractor-family architectures. It does not authorize training, deployment, or a scalar product cost.
