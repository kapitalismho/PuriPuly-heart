# Evaluator contract

This contract is part of `psem-relative-occupancy-v0` and is frozen before V2 EVAL access.

## Coordinates and inclusion

All source coordinates are zero-based, half-open, unsnapped 16 kHz samples. Primitive cross-model metrics use 1600-sample cells and the GT state at each cell center. A final partial cell is included only when its center is inside the scored range and is weighted by its exact duration.

Masked cells are excluded from primitive state metrics. Masked spans pause enrollment and replacement evidence and cannot enroll, confirm, clear, or cut. Invalid trace support and `UNANCHORED` or `ANCHOR_UNCERTAIN` time are not scored as valid anchor-conditioned primitive observations. Product metrics report active speech during masked, uncertain, and unanchored time separately as fail-closed unknown exposure. They also report a conservative upper bound that adds unanchored and uncertain active speech to exact anchored `OTHER_ONLY`; those unknown spans are not mislabeled as known contamination.

Sortformer native 80 ms posterior supports are treated as half-open source intervals and held at 100 ms cell centers. LS-EEND native 100 ms outputs use their exact source-frame support and center mapping. A posterior is available only at its recorded evidence frontier.

## Primitive metrics

`anchor_present` and `other_present` each use duration-weighted PR curves over the predeclared 0.05 threshold grid and duration-weighted average precision. Threshold comparisons use `probability >= threshold`.

At each threshold pair, the two binary decisions form the four canonical states. The evaluator reports a duration-weighted 4×4 confusion matrix, per-state precision/recall/F1, macro-F1, and binary FP/FN duration. `p_other` is the maximum posterior among alive non-anchor slots, or zero if none is alive.

Gate 1 chooses one anchor slot per logical GT anchor episode. Its support score is the duration-weighted mean posterior over unmasked GT anchor-active cells. Ties select the lowest slot index. No frame-wise oracle remapping is allowed.

## Product metrics

The no-speaker-cut baseline and every decoded arm use the same 200 ms reliable-singleton GT eligibility rule and the separate 1200 ms GT speech/non-speech final-lifecycle proxy. The proxy does not define `other_present`.

Exclusive-other contamination is the exact unmasked source duration for which the current logical segment's GT anchor is absent and at least one other GT speaker is active before a speaker-induced cut or final lifecycle reset. Active-speech hours use the union duration where at least one authoritative GT speaker is active, including overlap once.

For a fixed replacement duration, the corresponding Gate 0 oracle event sequence is the primary reference for model cut alignment. Alignment is per source, monotonic, one-to-one, within ±500 ms, maximizing match count and then minimizing total absolute boundary displacement. Unmatched predicted events are unnecessary/false speaker cuts; unmatched oracle events are missed replacements. Historical V2 `handoff_confirmed` alignment uses the same rule only as a separately labeled diagnostic.

Contamination per true replacement uses the exact interval from the oracle replacement boundary until the predicted cut, final lifecycle reset, or next oracle replacement, with zero floor for an early aligned cut. Backdated boundary error is predicted boundary minus the aligned exact oracle boundary. Replacement emit delay is decoder emit sample minus the exact oracle boundary.

Overlap-return preservation means no speaker-induced cut in a coverage-eligible `overlap_return` episode. Overlap-takeover success means an aligned cut after the exact `OTHER_ONLY` onset in a coverage-eligible `overlap_takeover` episode. All topology labels remain derived evaluation slices rather than occupancy targets.

## Anchor safety and timing

Enrollment delay begins at the first unmasked GT reliable-singleton opportunity for the eventual anchor and ends at the causal decoder's anchor emission. The report includes p50, p90, fractions within 1.0 s and 1.5 s, failure, and wrong-anchor rates. Wrong-anchor cascades count consecutive false speaker cuts attributable to one wrong/unstable enrollment until a correct fresh enrollment or final lifecycle reset. Uncertain time is retained in product denominators.

Every event preserves the backdated boundary, model evidence frontier, decoder emit sample, and measured compute lag when available. The model evidence frontier is the scalar availability time attached to the qualifying observation and is never interpolated backward. Native frame duration, algorithmic buffering/lookahead, model frontier delay, enrollment confirmation, replacement confirmation, and compute runtime are reported separately and are not added twice.

## Role and freeze discipline

TRAIN is adapter smoke only. DEV selects thresholds and one causal enrollment configuration from `config.json` without changing the grid. At Gate 0, EVAL derivation is unconditionally rejected. EVAL remains inaccessible until a frozen-selection receipt implementation binds the config, ontology, decoder, evaluator, trace schema, model pins, DEV manifest, selected settings, and authorization state by SHA-256. The same cached posterior pass feeds Gate 1 and Gate 2. EVAL is then derived and decoded once without selection changes.

Gate 0 acceptance independently reconstructs the exact DEV manifest from the pinned V2 artifacts and reference checkout, replays the decoder, checks exact boundary and qualification samples for every anchor episode, and compares the regenerated event, topology, result, and semantic-metric artifacts. Worktree Git state in preflight is informational; all load-bearing experiment files are content-hashed.
