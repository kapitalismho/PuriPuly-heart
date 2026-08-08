# Phase 3 experiment design

## Decision question

The experiment asks whether an online speaker-change detector adds useful turn boundaries to the existing VAD segmentation policy. B0 is the current VAD-only segmentation policy. It is not a speaker-change detector. Its apparent speaker-change recall is incidental coverage from a VAD boundary near an annotated change.

The experiment compares LS-EEND and ERes2NetV2 as boundary-event producers, both alone and after adding their non-coalesced events to B0. It does not wire either family into production.

## Hypotheses

- H1: a detector can recover B0 speaker-change misses without an unacceptable number of extra logical cuts.
- H2: at matched actual extra-cut counts, LS-EEND has enough additional boundary value to justify its larger streaming and compute footprint.
- H3: the selected development frontier generalizes to held-out synthetic, AMI, and AliMeeting data.

No single false-cut price is assumed. The result preserves the whole empirical trade-off frontier.

## Data and access order

Development uses `mixed_dev_pool`. Held-out evaluation uses `ls_held_out_clean`, `ls_held_out_other`, `ami_held_out_pilot`, and `alimeeting_eval_pilot`.

All LS-EEND checkpoints and both ERes2NetV2 checkpoints are evaluated on development. The full adjacent and stable-anchor policy matrices are evaluated. One representative development-frontier finalist per family is frozen before held-out manifests are opened by the Phase 3 runner.

The representative maximizes gross 500 ms B0-miss recoveries per incremental false cut. A zero-false-cut recovery has infinite efficiency. Ties prefer more 250 ms recoveries, more 500 ms recoveries, fewer false cuts, lower causal delay, and then lexical profile ID. This ratio is a parameter-free empirical efficiency rule rather than a false-cut cap. The full frontier remains a development result; the held-out run checks generalization of the frozen family representatives and does not retune them.

## Canonical timeline

All boundary, observation, and ground-truth coordinates are canonical 16 kHz source samples inside one audio epoch.

Every detector event carries:

- `boundary_source_sample`: where the split belongs;
- `observed_source_sample_at_emit`: how much source audio was available when the event became usable;
- `emitted_monotonic_ns`: measured execution time when available;
- `confidence`: increasing speaker-change strength.

An event is never allowed to use an observation coordinate earlier than its boundary coordinate. State resets at an audio-epoch boundary. LS-EEND state otherwise remains continuous across VAD utterances. ERes2NetV2 policies reset their anchor or confirmation state at each VAD utterance because the policy is explicitly VAD-scoped.

## ERes2NetV2 policies

Adjacent-window profiles compare the two windows touching a proposed boundary. Confirmation 1 emits immediately after the first low-similarity probe is available. Confirmation 2 emits the first candidate only after the next consecutive probe is also low-similarity. Each low-similarity run emits at most once and rearms after a failed probe. Change confidence is one minus the mean confirmed cosine similarity.

Stable-anchor profiles initialize an anchor from the first complete post-`SpeechStart` window. Confirmation 1 immediately emits and promotes the candidate probe. Confirmation 2 requires a second consecutive low-anchor-similarity probe and the pinned mutual-similarity threshold, emits the first candidate at second-probe availability, and promotes the first candidate embedding. A failed confirmation starts a new candidate only if the current probe is itself below the anchor threshold. EMA updates occur only on stable non-candidate probes and use the profile's frozen alpha.

No window is padded. A probe exists only when its complete right edge has been observed.

## Causal matching contract

Boundary localization and causal availability are separate constraints.

A cut can match a ground-truth speaker change only when all of the following hold:

1. both belong to the same audio epoch;
2. absolute boundary localization error is at most 500 ms;
3. the event was not available before the ground-truth change;
4. the event was available no later than the reported deadline after the change.

Matching is deterministic, one-to-one, ordered within each epoch, and maximizes matched changes before minimizing total observation delay and localization error. Recall is reported at 250, 500, 1000, 1500, and 2000 ms.

Detector-only matching uses all detector events. Product matching first locks B0 matches at each deadline and then matches only non-coalesced detector cuts to the remaining B0 misses. This directly identifies gross recovered B0 misses and prevents a detector from receiving credit by reassigning an existing VAD success.

At 2000 ms, any logical cut left unmatched is an operational false cut. Product incremental false cuts equal added detector logical cuts minus recovered B0 misses at 2000 ms. Coalesced detector events add no logical cut and therefore no false-cut cost.

## Reported quantities

Raw integer counts are primary:

- ground-truth changes;
- B0, detector-only, and product matches at every deadline;
- gross recovered B0 misses;
- B0 matches lost by the product, which must be zero under the locked policy;
- added detector logical cuts;
- incremental false cuts;
- remaining misses at 2000 ms.

Rates are descriptive views of the same counts. Both annotated-active-speech hours and source/session hours are reported. A five-minute-session estimate always uses source/session time, never active-speech time. Exact Poisson 95% intervals accompany false-cut rates, and Wilson 95% intervals accompany recall.

Timing reports signed localization error, non-negative causal audio delay, event lookback, and positive late-cut leakage. Negative causal delay is a contract violation.

Condition breakdowns cover gap/overlap class, following-turn duration, clean versus codec/noise stress, dataset, language, domain, and ground-truth transition kind.

## Comparison and freeze rule

Profiles are dominated only when another profile has no more incremental false cuts, no fewer recovered B0 misses at every deadline, and at least one strict improvement. Rates such as 0.5, 1, 2, and 5 false cuts per hour are shown only as reference slices. They never eliminate a profile before the frontier is computed.

Development operating curves use actual integer incremental false-cut allowances. Family comparison uses the union of achieved integer costs so LS-EEND and ERes2NetV2 can be viewed at matched observed cost.

Held-out results are decision-grade only if the frozen artifact hash, manifest hashes, model hashes, cache contracts, causal invariants, and independent aggregate recomputation all pass. Because no authorized D4 product-domain audio is available, any family preference remains provisional.

## Cache and provenance boundary

Neural caches are reusable only when they bind the checkpoint hash, frontend contract, manifest hash, case ID, WAV hash, window coordinates or LS capture arrays, and cache schema. Legacy ERes embeddings may be imported only after deterministic sampled recomputation agrees numerically. Legacy metric rows, shortlists, frozen files, and held-out conclusions are never imported.

Every result artifact is canonical JSON with a content hash. Row files have direct byte SHA-256 provenance. Partial family runs cannot overwrite or silently stand in for a complete summary.

## Validity limits

The development pool is small enough that one false cut materially changes an hourly rate. Therefore the experiment does not claim fine-grained precision from rate differences. Corpus diversity, clustered examples, synthetic construction, VAD-scoped ERes operation, and the absence of D4 audio remain explicit external-validity limits.
