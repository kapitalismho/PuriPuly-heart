# Bounded translation quality assessment

## Initial matrix: independent inspection

A separate read-only reviewer inspected every stored input/output in these immutable initial records: `new-full`, `new-cpu-moe-off`, `new-cpu-moe-cache1024`, `new-cpu-moe-cache2048`, `new-partial14`, `new-partial20`, `old-full`, and `e4b-product` under `results/`.

- **344 completed outputs:** 301 fixed-target 26B QAT Q4 outputs across seven profiles, plus 43 E4B reference outputs.
- **Only 12 distinct synthetic source sentences.** Repetition is serving/stability evidence, not 344 independent quality cases.
- Each profile contains three first-prefix requests, five same-pair requests, 20 alternating-pair requests, and 15 requests at three-slot occupancy.
- All stored outputs ended with `eos`; visible text was also inspected for completeness, meaning, negation, names, quantities, malformed text, repetition, and reasoning leakage.

**Verdict:** no clear placement-, cache-, or old-to-new-runtime-specific material semantic regression was observed in this initial corpus. The tested central negations, ownership corrections, recipient restrictions, times, dosage frequency, fee, currency, exclusion, rain condition, and emotional contrast were retained. No visible reasoning, pathological repetition, malformed text, or obvious truncation was found. This is a bounded spot-check, not a language-wide quality pass, exact-equivalence proof, or evidence that 26B generally outperforms E4B.

## Fidelity caveats retained

| Case | Observation | Interpretation |
| --- | --- | --- |
| Chinese `小林没有取消预约，只是把时间改到了星期五。` | Every profile renders `小林` as **Kobayashi**. | The source does not establish whether this is the Japanese surname or Chinese Xiao Lin. Exact name reading is unresolved; agreement across profiles is not an exact-name pass. Cancellation negation and Friday rescheduling are preserved. |
| `I will be back in 5 minutes.` | 11 of 42 QAT occurrences use `5분 안에 돌아올게요` (**within** five minutes), rather than `5분 뒤에` (**in/after** five minutes). | Minor temporal precision drift, also present in the old-runtime control and not monotonic with cache or placement. Do not claim uniformly exact translations. |
| E4B Chinese appointment case | E4B adds **he** where the source has no gendered pronoun; QAT uses **they**. | Unsupported gender specificity in 2/43 E4B outputs, one unique source. Not a broad model ranking. |
| E4B ticket correction | `미나는 12장이 아니라 2장만 샀어` omits explicit **tickets**. | Name, negation, and quantities remain, but standalone object specificity is reduced in 7/43 E4B outputs, one unique source. |
| Japanese dosage `一日2回 ... 3回ではありません` | Correct English says **twice a day, not three times**. | The retained automated `numbers_present=false` is a literal-digit false alarm, not a semantic failure. Automatic digit matching is not the quality gate. |
| Red-door and rain cases | E4B uses **through** rather than **to**, and **can meet** rather than an asserted alternative plan. | Small route/modality differences, not reversals of the central instruction or condition. |

### Exact initial “within five minutes” locations

Indices below are phase-local indices stored in the run records, not array offsets.

- `new-full`: `hot_same_pair[4]`.
- `new-cpu-moe-off`: `hot_same_pair[1,4]`, `three_slots[3]`.
- `new-cpu-moe-cache1024`: `hot_same_pair[4]`.
- `new-cpu-moe-cache2048`: `hot_same_pair[4]`.
- `new-partial14`: `hot_same_pair[4]`, `pair_alternation[4]`, `three_slots[3]`.
- `new-partial20`: none observed.
- `old-full`: `hot_same_pair[4]`, `three_slots[12]`.
- `e4b-product`: none observed in its six occurrences.

The ambiguous-name case occurs at `pair_alternation[3,15]` in every initial profile. E4B's ticket-object omission occurs at `first_prefix[0]`, `hot_same_pair[0,3]`, `pair_alternation[0,12]`, and `three_slots[0,9]`.

## Boundaries

This initial review does not establish Vulkan execution, internal reasoning controls, memory efficiency, application completion, restart behavior, or resource release. Those require the separately recorded runtime/app observations. Refinement and actual-app outputs are outside the independent review's denominator and must be assessed separately before selection.

## Complete refinement corpus: execution-owner inspection

The execution owner inspected all unique stored input/output pairs after the original profiles and accepted-F1 refinements completed. `results/quality-unique.json` now contains **1,462 output occurrences, 29 unique input/output pairs and 12 distinct source sentences**: 1,419 fixed-target QAT outputs from 33 successful run identities and 43 E4B outputs. Repeated-control identities are not new source cases. This extends the observed serving configurations; it does not expand the independent review's 344-output denominator or source-language coverage.

The same main meaning, negation, quantity and ownership checks remained intact across these outputs. No new placement/cache-specific semantic collapse, wrong-language response, pathological repetition, visible reasoning, or obvious truncation was observed. Every successful standalone record reports normal EOS; content inspection, not EOS alone, supports the bounded completeness observation.

The current QAT temporal-precision count is **61/198 occurrences** of the five-minute source using `안에` (within), out of 204 occurrences including the six E4B references. The original checkpoint **44/156** and initial independent **11/42** results remain historical findings, not overwritten run data. This drift must not be described as uniformly exact time fidelity or attributed specifically to one cache budget. All standalone target profiles retain the ambiguous **Kobayashi** reading of `小林`. The F1 repair added301 translations, including pressure-qualified repeats; every output variant was already represented in the prior29-pair corpus, with central negations/quantities preserved and no new visible reasoning, repetition or truncation.

## Actual application: execution-owner inspection

All unique variants representing **390 matched completed app translations** were assessed across13 run identities: **330 QAT outputs and60 E4B outputs** (30 generic-endpoint reference plus30 true managed stock reference), using five distinct sources and actual rolling, pair-matched source history. Each completed run supplied30 translations across two fresh app hosts. The true managed run also restarted its owned model process; generic servers survived app-only restart. This is separate from the fixed-history matrix corpus.

- The QAT app profiles preserve the explicit ticket noun, Mina's 12-versus-two correction, Yuna's four-not-three time, Tanaka's “not yet,” and Minsu's ownership restriction. No visible reasoning, repetition, wrong-language output or obvious truncation was observed.
- The Chinese QAT app output is “Xiao Lin didn't cancel the appointment, they just moved it to Friday.” This differs from the standalone fixed-history name rendering, without reversing cancellation or Friday rescheduling. The prompt/source still does not determine the intended name reading, so neither spelling is an exact-name certification.
- E4B app outputs preserve the ticket noun in these probes, unlike its standalone omission. The generic reference alternates Kobayashi/he and Xiaolin/she for the ungendered Chinese source; the true managed stock run also varies Kobayashi/Xiaolin and he/she/they. Unsupported gender specificity and unresolved name reading remain; negation and Friday rescheduling are preserved.
- Completed content was matched to admission IDs and checked after language changes and across fresh app instances. The first failed app attempt submitted no text and contributes no quality sample.
- Accepted-F1 OFF/1024/fresh2048/3072 probes added120 completions with the same five QAT output variants and no new material semantic change. True managed E4B added30 normal-product-sampling outputs (no probe seed/output override); stylistic variants were checked separately, not treated as equivalent-quality/model superiority evidence.
- The six bounded memory-attribution texts preserve Mina's12-versus-two correction, Yuna's four-not-three time and Japanese twice-not-three dosage. They are diagnostic observations, not new independent source cases or additions to the1,462-output ranking corpus.
- Native prefill/decode counters and wall time do not certify audio quality or speech end-to-end latency. Text wall starts at CLI submit, after any separately measured prefix preparation; audio collection/endpointing/STT and GUI are outside the controlled boundary.

This is a synthetic spot-check, not bilingual professional adjudication, a model-quality leaderboard, live audio/ASR quality evidence, or proof that QAT 26B justifies replacing E4B. Runtime-resource conclusions and the controlled inactive-STT boundary are recorded separately in `RESULTS.md`.
