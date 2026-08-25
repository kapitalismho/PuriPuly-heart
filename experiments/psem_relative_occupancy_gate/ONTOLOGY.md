# `psem-relative-occupancy-v0`

For a valid transient anchor `a` and the authoritative active-speaker set `S(t)`:

```text
anchor_present(t) = 1 iff a is in S(t)
other_present(t)  = 1 iff any speaker other than a is in S(t)
```

The joint states are `NONE`, `ANCHOR_ONLY`, `ANCHOR_PLUS_OTHER`, and `OTHER_ONLY`. Overlap is derived as `anchor_present AND other_present`. VAD is not an occupancy input.

Anchor lifecycle is separate: `UNANCHORED`, `ANCHORED`, or `ANCHOR_UNCERTAIN`. Speaker-induced cuts are enabled only while `ANCHORED`. Uncertainty fails closed to VAD-only behavior.

`OTHER_ONLY` begins replacement evidence. A fixed decoder confirms after 100, 200, 300, or 500 ms of unmasked evidence, backdates the logical boundary to the first qualifying source sample, and keeps boundary time distinct from evidence availability and emission time. Each model observation carries one scalar evidence frontier; the decoder never interpolates an earlier frontier inside that observation. Gate 0 splits exact GT spans at the qualification sample, so its frontier and emission remain exact. Masked cells pause evidence. `ANCHOR_ONLY`, `ANCHOR_PLUS_OTHER`, and `NONE` clear a pending run.

After a confirmed replacement, the primary lifecycle returns to `UNANCHORED`. It never inherits a non-anchor slot automatically.

Gate 0 uses a deterministic GT anchor after 200 ms of an unmasked singleton interval. The GT lifecycle proxy preserves that anchor through ordinary pauses and discards it after 1200 ms of accumulated unmasked silence; masked time pauses this control timer. The 1200 ms value is the frozen V2 local-continuity maximum and is not part of the occupancy ontology. Gate 1 selects one model slot per logical GT anchor episode with an episode-level support integral and lowest-slot tie-break. Gate 2 uses only past/current model state plus the same GT speech/non-speech lifecycle proxy. It requires one strong slot, all others low, and a fixed 400/600/800/1000/1200 ms confirmation interval.

Canonical scoring uses 100 ms cells centered on the exact 16 kHz source timeline. Exact source intervals remain authoritative for boundary and duration calculations. Sortformer 80 ms native posteriors are held over their half-open source support and sampled at 100 ms cell centers. LS-EEND is already 100 ms and is mapped by its exact output-frame source center. Invalid sampled cells carry `trace_valid=false` and an evidence-frontier sentinel of `-1`; downstream decoders must exclude them.

The historical V2 `handoff_confirmed` rows are used only as a derived diagnostic. Event alignment uses a predeclared ±500 ms monotonic one-to-one window that maximizes match count and then minimizes total absolute displacement. It does not define occupancy, anchor lifecycle, or a successful product cut.
