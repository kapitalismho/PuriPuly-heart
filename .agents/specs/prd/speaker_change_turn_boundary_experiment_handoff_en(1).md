# Handoff for Resuming the Speaker-Change-Based STT Turn-Boundary Experiment

## 0. Document Status and Authority

This document is the **execution handoff for resuming the speaker-change-based STT/translation turn-boundary experiment from the designated earlier checkpoint**.

The experiment authority order is:

1. This handoff
2. GitHub issue #51: **Experiment v2: speaker-change turn boundaries (LS-EEND vs ERes2NetV2) and streaming ASR policies**
3. Raw development artifacts and result rows created before the resume checkpoint whose provenance has been verified
4. Historical experiment descriptions and analysis notes — but only as historical evidence when they conflict with the corrected contract in this document

This is not a general diarization benchmark.

The purpose of the experiment is to **determine whether speaker-change evidence can be used in the existing PuriPuly STT lifecycle to create logical STT/translation turn boundaries, and which detector and policy are suitable for that purpose**.

Production wiring itself is out of scope. At experiment completion, the handoff must be specific enough that a separate implementation agent can implement the chosen detector/policy without reopening the research.

---

# 1. Original Product Problem

## 1.1 The Failure Users Actually Experience

The current product primarily ends an STT turn when VAD detects the end of a speech region.

The problem appears when multiple people speak **one after another without silence between them**.

Example:

```text
Speaker A: ...
Speaker B: ...
Speaker C: ...

Speech energy remains continuous
        ↓
VAD does not release the utterance
        ↓
[A + B + C] end up in one logical STT/translation turn
```

This is expensive because:

- Messages from multiple people are mixed into one subtitle/translation turn.
- It becomes difficult to understand who said what.
- Even when STT transcription is correct, the turn structure does not match the actual conversation structure.
- Sentences from different speakers are mixed into the same translation input unit.
- The current subtitle system preserves context across turns, so there is no need to keep speech from multiple speakers inside one oversized turn just to preserve context.

The product requirement is therefore much narrower than general speaker identification.

> **When the speaker changes, create a logical STT/translation turn boundary at the best recoverable audio position.**

Not required:

- Identifying the real-world person by name
- Enrollment
- Persistent speaker IDs
- Remembering speaker identity across unrelated sessions
- General diarization UI
- Source separation

The minimum signal needed is closer to:

```text
Is this the same current speaker?
            vs
Has it changed to a different speaker?
```

In other words, the core feature is not a `speaker identity system` itself, but **speaker-change-triggered logical turn segmentation**.

---

# 2. Error Costs Are Asymmetric

One of the most important parts of the original product definition in issue #51 is that false merges and false splits do not have equal cost.

## 2.1 False Merge — Expensive Error

```text
A -> B
```

The speaker actually changes, but both speakers remain inside the same logical turn.

This reproduces the exact product failure we are trying to solve.

## 2.2 False Split — Undesirable, but Relatively More Tolerable

```text
A -> A
```

Speech from the same speaker is unnecessarily divided into two or more logical turns.

False splits cannot be allowed without limit. However, they must not be assigned the same cost as false merges.

In particular, the following two cases have very different user impact.

```text
A: "I think tha|t is correct"
              ^ cut inside a word
```

```text
A: "I think that is correct." | "Next..."
                            ^ natural pause / sentence boundary
```

Both may be extra boundaries that do not match a reference speaker change, but their operational harm is not the same.

The original product assumption that a typical session is around five minutes is also retained. Product-facing rates must therefore be reported not only per speech-hour, but also in **values that are directly interpretable for a five-minute session**.

---

# 3. Questions This Experiment Must Answer

The following is not the primary question:

> Does splitting turns on speaker change help?

That behavior is a product requirement intended to address an already observed failure mode.

The following is also not the primary question:

> Which model has the best DER/EER or diarization performance?

We are not building general diarization.

## 3.1 Primary Product Question

> **When VAD and speaker-change evidence are used together, how much can we reduce mixed-speaker logical turns while staying within acceptable same-speaker false-split and causal-delay levels?**

## 3.2 Model Question

> Does each detector family provide speaker-change evidence that is accurate and fast enough to drive this product behavior?

## 3.3 Policy Question

> Can continuous model output be converted through debounce / confirmation / clustering / VAD fusion into one stable logical turn decision?

These questions must preserve the perspective order **product -> policy -> model diagnostics**.

---

# 4. Primary Product Construct

The previous experiment focused on counting boundary events. The resumed experiment changes direction so that the product failure users actually experience is measured more directly.

## 4.1 Primary Benefit: Reduced Speaker Contamination

Clean handoff example:

```text
Ground truth
A ----------------| B ----------------
                  ^ actual speaker change

VAD-only
A ------------------------------------|
                                     ^ late finalize
```

If B's speech remains in the previous logical turn after the true speaker change, that interval is **previous-turn speaker contamination**.

Recommended primary benefit metrics:

1. `wrong-speaker speech milliseconds assigned to previous logical turn`
2. contamination duration per five-minute session
3. fraction of finalized logical turns containing substantive speech from two or more speakers
4. number/rate of clean or gap handoffs that remain merged in final logical segmentation
5. contamination reduction relative to B0

Speaker-change recall remains useful, but should be treated as a **proxy/diagnostic for product benefit**.

## 4.2 Primary Harm: Same-Speaker False Logical Split

The opposing product cost is creating a logical turn boundary when the speaker did not actually change.

Recommended metrics:

- same-speaker false logical splits / five-minute session
- same-speaker false logical splits / active-speech hour
- number and duration distribution of excessively short fragments
- degree to which continuous same-speaker speech is fragmented across multiple turns
- separate count of cuts inside lexical material where word timing is reliable
- unmatched splits at natural pauses or sentence boundaries classified separately

Core principle:

> **Do not label every extra boundary as an equally harmful false split merely because it does not match a reference transition.**

## 4.3 Causal Delay

A detector may estimate the historical boundary location accurately, while only being able to emit the decision after observing some amount of future audio.

Always report separately:

- `boundary_source_sample`: source position where the detector estimates the transition occurred
- `observed_source_sample_at_emit`: source frontier already observed when the event became usable
- boundary localization error
- causal availability delay relative to GT
- event lookback
- wall/scheduling delay

An accurate boundary and a boundary that is **available in time to use** are different things.

---

# 5. Scope Actually Covered by the Existing Development Experiment

The completed development sweep contains 1,369 candidate rows in total.

- B0: 1
- LS-EEND: 432
- ERes2NetV2: 936

Candidate diversity was therefore not the main problem.

The problem was **how this large candidate set was interpreted through the chosen metric and selection rule**.

---

# 6. Exact Meaning of B0

B0 is not a speaker-change detector.

B0 uses only the current VAD segmentation.

A reference match occurs only when silence causes VAD to create a turn boundary that happens to fall near the actual speaker transition.

Development results:

| causal deadline | B0 match | recall |
|---|---:|---:|
| 250 ms | 57 / 374 | 15.2% |
| 500 ms | 86 / 374 | 23.0% |
| 1000 ms | 89 / 374 | 23.8% |
| 2000 ms | 89 / 374 | 23.8% |

Do not interpret this as:

> VAD detects 23.8% of speaker changes.

The correct interpretation is:

> About 23.8% of all reference transitions were already close to a logical boundary created by VAD.

The residual problem the detector must solve is therefore relatively difficult:

- immediate handoffs with little or no silence
- zero-gap handoffs
- speaker changes inside VAD hangover
- short utterances
- some overlap/interruption onsets

This distinction must be preserved when interpreting model capability.

---

# 7. Single-Ratio Selection Distorted Family Capability

At one stage, one representative per family was selected using this ratio:

```text
number of B0 misses recovered within 500 ms
------------------------------------------
number of additional unmatched boundaries
```

Applying this rule to the completed development rows selected a very conservative LS candidate and a relatively aggressive ERes candidate.

## 7.1 Previously Selected LS Candidate

Approximately:

- profile: `L-AMI:dominant_replacement@thr0.70-p1-med11`
- 500 ms additional recovery: 4
- 2 s additional recovery: 4
- legacy additional unmatched count: 4
- B0 500 ms recall: 86/374 = 23.0%
- VAD+LS 500 ms recall: 90/374 = 24.1%
- improvement: about +1.1%p

Looking only at this point makes the LS family appear almost ineffective.

However, this was **not the family's maximum capability; it was an extremely conservative operating point that happened to look favorable under the ratio**.

## 7.2 Previously Selected ERes Candidate

Approximately:

- profile: `E-standard:anchor-W0.5-s0.1-thr0.30-c2-ema`
- 500 ms additional recovery: 43
- 2 s additional recovery: 75
- legacy additional unmatched count: 82
- B0 500 ms recall: 86/374 = 23.0%
- VAD+ERes 500 ms recall: 129/374 = 34.5%
- 2 s final recall: 164/374 = 43.9%
- 82 legacy unmatched events over about 43.2 minutes of development source audio
- under the previous normalization, about 9.5 per five minutes

This point recovered far more speaker transitions than the selected LS point, but it also produced too many additional reactions to be considered product-ready.

---

# 8. Looking at the Full Families Gives a Different Picture

High-recovery operating points in the full development rows show:

| configuration | additional recovery within 2 s | final recall | legacy additional unmatched |
|---|---:|---:|---:|
| B0 | 0 | 89/374 = 23.8% | 0 |
| LS conservative efficiency point | 4 | 93/374 = 24.9% | 4 |
| LS high-recovery point | 96 | 185/374 = 49.5% | 782 |
| ERes conservative efficiency point | 75 | 164/374 = 43.9% | 82 |
| ERes high-recovery point | 220 | 309/374 = 82.6% | 4,877 |

Facts already supported by this table:

### LS

- It is not a family that "only catches four".
- With aggressive reducers/thresholds, it recovers many more B0 residual transitions.
- However, repeated/unstable events increase sharply as recovery is pushed higher.

### ERes

- Strong speaker-change signal is clearly present.
- At the high-recovery point, it recovered 220 of the 285 B0 misses within two seconds.
- However, the raw policy can become so unstable that it reacts at nearly every inspection point.

The correct conclusion is therefore:

> **Both families contain useful signal, but converting that signal into one stable product-boundary decision is likely a central bottleneck.**

And:

> **Using a single recovery/unmatched-ratio winner as the representative of a family can severely distort family capability.**

---

# 9. Interpretation of LS-EEND That Must Be Preserved

The selected LS profile used a very conservative `dominant_replacement` reducer.

This policy roughly required:

- the previous dominant speaker to weaken or disappear
- another speaker track to become sufficiently strong
- the new speaker to actually become dominant

This helps suppress false reactions, but it may discard the following cases:

- a new speaker appears while the previous speaker remains active
- short interruptions
- both speaker posteriors are temporarily high
- transient slot instability
- evidence of a new speaker exists, but dominant replacement has not yet happened

Conversely, a more sensitive policy such as `new_speaker_onset` improves recovery, but unstable track activations can also become boundary proposals.

The evidence therefore supports this conclusion:

> **There is no basis for claiming that LS lacks speaker-change signal internally. The strong tradeoff appears in the step that converts posterior/activity tracks into product boundaries.**

The resumed experiment must separate:

1. raw/posterior signal quality
2. onset/replacement reducer quality
3. proposal clustering quality
4. VAD fusion quality

---

# 10. Interpretation of ERes2NetV2 That Must Be Preserved

ERes2NetV2 is an embedding extractor.

The model itself does not directly output a `boundary event`.

Representative policies used in the experiment:

### Adjacent

Compare speech-window embeddings immediately before and after a candidate boundary `t`.

```text
left embedding
     vs
right embedding
```

Low similarity is treated as a speaker-change candidate.

### Stable Anchor

Create an anchor embedding for the current speaker and continuously compare it to the latest probe embedding.

```text
stable_anchor
     vs
latest probe
```

A new speaker is confirmed when the probe differs from the current speaker while consecutive candidates remain mutually similar.

Development evidence indicates that ERes contains strong speaker-change signal.

However, aggressive profiles produced too many events.

The supported conclusion is:

> **ERes embedding signal is highly promising, but threshold/window/anchor policy alone is too sensitive to repeated reactions and ordinary acoustic variation to directly serve as a stable product boundary.**

The resumed experiment must explicitly place the following between `model output -> product action`:

- confirmation
- candidate mutual similarity
- debounce
- refractory
- detector-detector clustering
- VAD interaction

---

# 11. Legacy "Additional False Cuts" Were Not Operational Harm

The previous calculation was roughly:

```text
number of new logical cuts created by the detector
-
number of B0 misses recovered within the chosen deadline
```

This quantity is reproducible as bookkeeping, but it is not equivalent to product harm.

A single count could include any of the following:

- a cut inside a word spoken by the same speaker
- a cut at a natural pause by the same speaker
- a cut at a sentence boundary
- an event that slightly accelerates a logical boundary VAD would create shortly afterward
- multiple detector reactions around one true transition
- an event whose product harm cannot be determined because annotation is insufficient

Historical values must therefore be referred to only as:

> **legacy additional/unmatched boundary count**

Do not use the phrase:

> harmful false cut count

unless product harm has actually been classified using an observable definition.

---

# 12. One of the Largest Omissions: Detector Events Were Not Coalesced with Each Other

The previous implementation roughly behaved as follows:

- detector event <-> VAD boundary: coalesce when close
- detector event <-> another detector event: may remain independent even when both are repeated reactions to the same transition

With this structure, if one real speaker transition produces:

```text
10.10 change
10.20 change
10.30 change
10.40 change
```

only one event can match the GT transition and the rest become unmatched.

An exploratory 500 ms refractory diagnostic reduced action counts approximately as follows:

| family high-recovery point | original added actions | approximate count after 500 ms refractory |
|---|---:|---:|
| LS | 878 | 795 |
| ERes | 5,097 | 1,766 |

These numbers must not be treated as corrected results.

The diagnostic did not:

- use a frozen causal representative-selection rule
- fully rerun matching
- use clustering defined as a product policy

However, it strongly suggests:

> **Especially for ERes, a substantial portion of the legacy unmatched count may represent repeated reactions around one underlying change rather than independent user-visible false decisions.**

Mandatory correction for the resumed experiment:

1. Treat raw detector output as proposals
2. Apply causal debounce/refractory/clustering
3. Select the event representing one underlying logical decision
4. Then perform VAD fusion
5. Score only final logical actions

---

# 13. A Proposal Is Not a Product Cut

The resumed experiment uses the following vocabulary explicitly.

## Detector Proposal

A speaker-change hypothesis produced by a model or reducer.

It does not yet cut an STT turn.

## Logical Boundary Cluster

A unit that groups multiple proposals and VAD events that refer to the same underlying transition into one logical decision.

## Fusion Action

The actual product-level decision.

Possible actions include:

- retain an existing VAD boundary
- accelerate a VAD boundary using speaker evidence
- let a detector boundary replace a later VAD boundary
- add a new hard boundary missed by VAD
- suppress duplicate/repeated proposals
- record overlap-related events separately as diagnostic events

**Do not use experiment semantics in which every model event immediately becomes a cut.**

---

# 14. VAD Must Not Be Treated Only as "Existing Boundaries + Detector Additions"

The original issue #51 expressed the product combination as:

```text
existing VAD boundaries
+
new speaker-change boundaries that VAD did not already create
```

This was reasonable for measuring incremental value over B0.

However, the real product lifecycle has another important case.

## 14.1 Acceleration / Replacement

```text
A -------- B ---------------- silence
          ^ speaker evidence      ^ VAD finalize
```

If speaker evidence allows the turn to end near B onset, the detector may not be adding a new logical turn at all. It may be **making the same logical boundary that VAD would eventually create available earlier**.

In the previous evaluation, an event near a VAD boundary could be absorbed so that it received:

- no extra false cost, but also
- no proper latency/contamination-reduction benefit

The resumed scoring must distinguish at least:

1. `recovered B0 miss`
2. `accelerated/replaced B0 hit`
3. `duplicate logical boundary`
4. `independent same-speaker false split`
5. `unscorable / annotation-insufficient`

Without this distinction, part of the detector's core product value may remain invisible.

---

# 15. Do Not Collapse Transition Taxonomy into One Headline Recall

Approximate composition of the 374 development GT transitions:

- gap speaker change: 183
- interruption onset: 162
- clean handoff: 29

Interruption onset accounts for roughly 43%.

This matters.

## 15.1 Clean Handoff

```text
{A} -> {B}
```

Primary product case.

Desired action:

> Finalize A's turn near B onset and begin B's turn.

## 15.2 Gap Handoff

```text
{A} -> {} -> {B}
```

This is a true speaker change, but VAD is likely to handle many such cases already.

The detector's product value may be either:

- recovering a case VAD missed, or
- accelerating a case delayed by VAD hangover

## 15.3 Interruption / Overlap Onset

```text
{A} -> {A,B}
```

Detecting B's appearance is a success from the perspective of speaker-change signal.

But in a mono mixed waveform:

```text
A A A A A A
      B B B B B
```

cutting at B onset does not make the following audio B-only.

Therefore:

- report it separately as new-speaker-onset detection
- do not call it clean turn-separation success
- do not combine it into the clean/gap headline recall
- do not imply source-separation capability

Because a substantial fraction of high-recovery ERes recoveries came from interruption onset, the overall 82.6% recall must not be interpreted as a **clean multi-speaker turn-resolution rate**.

## 15.4 Same-Speaker Pause

```text
{A} -> {} -> {A}
```

This is not a speaker-change target.

VAD may create an ordinary turn boundary, but it must not receive speaker-detector benefit credit.

---

# 16. Limitations of the Existing Development Data

Completed development sweep:

- cases: 204
- reference speaker changes: 374
- source duration: about 2,591.9 s = 43.2 min
- active speech: about 1,822.6 s = 30.4 min

Composition:

- LibriSpeech-based synthetic cases: 202
- AMI real source sessions: 2

Reference transitions:

- synthetic: 164
- AMI: 210

Critical point:

> The 210 AMI transitions were not 210 independent real sessions. They were **clustered inside only two source sessions**.

Transition-level pooled intervals therefore overstate the effective independent sample size.

For example, one unusual acoustic/session pattern in a single AMI session could heavily influence a family result while appearing numerically like more than 100 independent transitions.

## 16.1 Role of Synthetic Cases

Synthetic data remains useful for:

- exact zero-gap generation
- sharply defined boundary timestamps
- same-speaker negatives
- short A/B duration stress
- codec/noise/gain stress
- basic threshold/reducer development

However, product generalization must not be claimed from short synthetic splice cases alone.

## 16.2 Role of Public Real Conversational Data

Private or newly recorded product-like conversation data is **not available for this experiment and must not be assumed as a prerequisite**.

Real-conversation coverage should therefore come only from authorized public conversational corpora already compatible with the experiment, such as the AMI/AliMeeting-style sources defined by the experiment authority. The goal is not simply to accumulate more transitions. The goal is to increase the number of **independent source sessions** and reduce the extent to which a family result is determined by one or two meetings.

When expanding public real-conversation coverage, prioritize sessions that expose:

- natural no-gap handoffs
- short reactions
- microphone/gain differences
- prosody variation from the same speaker
- natural pauses
- interruption/overlap
- multiple speakers within one source session
- language/acoustic diversity available from the authorized public corpora

The inventory must report both transition counts and independent-session counts. Hundreds of transitions from a small number of meetings must not be treated as equivalent to broad independent-session coverage.

## 16.3 Product-Domain Limitation

Because representative newly recorded PuriPuly-like conversational audio cannot be created for this experiment, **product-domain acceptance audio is not a gate and must not be silently assumed to exist**.

Consequences:

- final family selection may be made from synthetic and authorized public conversational evidence;
- any claim about exact PuriPuly-domain generalization remains provisional;
- the experiment may still select the strongest engineering candidate for follow-up implementation/conformance work;
- the absence of private product-like audio must be stated as an external-validity limit rather than compensated for by pretending a public corpus is product-identical.

## 16.4 Coverage-Driven Data Expansion

Additional data should be added to fill known coverage gaps, **not to hit an arbitrary hour or transition-count target**.

The preferred expansion order is:

1. increase independent public conversational sessions;
2. increase zero-gap and near-zero-gap different-speaker handoffs;
3. increase same-speaker hard negatives;
4. increase short-turn and acoustic-stress coverage;
5. add only the strata that remain underrepresented after an inventory pass.

### Zero-gap and near-zero-gap emphasis

The detector exists primarily to solve transitions that VAD does not naturally separate. Synthetic expansion should therefore oversample the conditions most likely to remain inside one VAD utterance:

- exact zero-gap A -> B
- approximately 0-100 ms gap
- approximately 100-300 ms gap
- short B turns immediately after A
- handoffs occurring inside VAD hangover

Longer-gap A -> B cases remain useful as controls, but they should not dominate the development pool because VAD often already owns those boundaries.

### Same-speaker hard negatives

False-split evaluation requires substantially more same-speaker exposure than a small set of simple A1 -> A2 negatives. Add controlled negatives covering, where source material allows:

- brief pauses followed by the same speaker
- large gain changes
- prosody/emotion changes
- laughter or non-speech vocalization followed by resumed speech
- codec/noise perturbation
- bandwidth changes
- short silence/noise spans
- same-speaker utterance concatenations with gap distributions matched to positive cases

These negatives are especially important for embedding-based policies, because acoustic change can otherwise be mistaken for speaker change.

### Coverage inventory before further expansion

Before generating or acquiring more cases, create a metadata-only inventory for every development/held-out pool with at least:

- source-session count
- source duration
- active-speech duration
- number of speakers
- clean zero-gap handoffs
- near-zero-gap handoffs
- longer-gap handoffs
- overlap/interruption onsets
- same-speaker pause/continuation cases
- same-speaker active-speech exposure
- B0-missed speaker changes
- B0-already-separated speaker changes
- short-turn counts
- codec/noise/gain stress counts

The most important exposure quantity for detector value is **how many speaker changes B0 actually misses**. A large corpus dominated by transitions that VAD already separates provides little information about the incremental product value of a speaker-change detector.

### Data-pool roles

Keep the purpose of each pool explicit:

**Controlled diagnostic pool**
- synthetic cases
- exact known boundary positions
- controlled gap/overlap/turn-length/acoustic factors
- used for failure analysis and reducer development

**Public conversational benchmark**
- independent natural conversation sessions from authorized public corpora
- used to test whether operating behavior survives natural speech, recurrence, overlap, microphone variation, and session-level clustering

**Held-out evaluation pool**
- session/speaker-disjoint from development wherever source metadata permits
- all thresholds, reducer parameters, clustering, and fusion rules frozen before access

There is no private/product-recorded acceptance pool in the current experiment. This limitation must remain visible in the final conclusion.

---

# 17. Statistical Independence Contract

The primary uncertainty unit is not an individual speaker transition.

Required:

- paired comparisons within source session
- session-level bootstrap or stronger grouping
- recurring participants / related recording families kept in the same uncertainty block where applicable
- raw integer counts alongside every rate
- transition-pooled micro counts used only as descriptive evidence

Forbidden interpretation:

> Treating 210 transitions inside two real meetings as 210 independent real-world samples

Small counts such as LS `4 recovery / 4 unmatched` must be explicitly recognized as high-variance evidence.

---

# 18. Correct Candidate Selection

Do not freeze a single-ratio winner.

Do not automatically freeze the maximum-recall winner either.

## 18.1 Development Frontier

For each family, preserve a non-dominated frontier over at least:

- speaker contamination reduction / false-merge reduction
- same-speaker false splits
- causal delay
- duplicate/repeated logical boundaries
- compute/runtime cost

Also retain as diagnostics:

- speaker-change recall at deadlines
- localization error

## 18.2 Matched Product-Harm Budget

Compare families at the same product-harm budget.

Example:

```text
same-speaker false logical splits
0.5 / 1 / 2 / 5 per speech-hour
```

or an equivalent budget directly interpretable per five-minute session.

Exact numeric budgets may be frozen after corrected product data is available.

The key invariant is **comparison at the same budget**.

## 18.3 Frozen Operating Panel

Before held-out evaluation, freeze multiple tradeoff points per family.

For example:

- low false-split point
- medium point
- high contamination-recovery point

The purpose of this panel is:

> to test whether the shape of the development frontier persists on unseen data

not to compress an entire family into one point.

## 18.4 Held-Out Discipline

- run held-out only after threshold freeze
- freeze reducer
- freeze clustering/refractory
- freeze VAD fusion rule
- freeze action taxonomy
- freeze scoring
- do not move profiles after seeing held-out results
- report poor frozen points as well

---

# 19. Diagnostics for Separating Model-Signal Failure from Policy Failure

Low boundary results do not automatically imply a weak model.

Possible failure causes include:

- weak speaker-discriminative signal in the model itself
- useful signal exists but the reducer discards it
- raw proposals repeat but clustering is absent
- confirmation is too conservative or too aggressive
- localization is correct but causal availability is late
- VAD fusion mishandles valid proposals

Therefore, retain lightweight signal diagnostics.

## 19.1 LS Diagnostics

Around GT transitions, inspect:

- speaker posterior changes
- new-track onset timing
- dominant replacement timing
- track flicker
- event sequence before and after confirmation
- actual causal timing through the streaming frontend
- oracle/near-oracle reducer upper bound using only causal model output

Question:

> **Does the model lack the signal, or is the reducer discarding it?**

## 19.2 ERes Diagnostics

Inspect:

- same-speaker cosine distribution
- different-speaker cosine distribution
- sensitivity to window length
- transition mixed-window behavior
- anchor drift
- mutual similarity among consecutive candidates
- sensitivity to gain/noise/prosody variation
- oracle/near-oracle upper bound using only available embeddings

Question:

> **Are the embeddings weak, or is the rolling policy oversensitive to ordinary acoustic variation?**

These diagnostics do not replace product-decision metrics.

---

# 20. Causal Timing Contract

Retain the source-time contract from issue #51.

Canonical identity is the continuous normalized source-audio timeline.

Recommended event:

```text
SpeakerBoundaryEvent {
    audio_epoch: int
    boundary_source_sample: int
    observed_source_sample_at_emit: int
    emitted_monotonic_ns: int
    confidence: float | null
    source: str
    debug: dict
}
```

Required semantics:

- `boundary_source_sample`: estimated change location
- `observed_source_sample_at_emit`: source frontier observed when the event became usable
- `emitted_monotonic_ns`: runtime/scheduling timing
- stale epoch events must not be applied to the current epoch

A detector must not receive causal credit as though it made an immediate decision if it actually used future audio that did not exist before the transition.

## 20.1 Frontend Timing

Measuring neural-model output latency alone is insufficient.

Causal availability must include delays introduced by:

- resampling buffers
- feature windows
- group/filter delay
- chunk processing
- flush behavior

When LS-EEND converts product 16 kHz audio through another sample rate/frontend, source-sample mapping must be measured using the actual streaming path.

Likewise, neural-output parity alone is insufficient for ERes. Frontend/export-path parity must be verified separately.

---

# 21. Correct Purpose of Oracle Tests

Oracle boundaries should be used, but for the correct reason.

The purpose of the oracle test is not:

> to prove again that speaker-separated turns are beneficial

The purpose is:

> **to validate, independently of detector quality, whether the provider/STT lifecycle can actually implement the intended logical split when given a perfect or controlled boundary event**

Use oracle traces with controlled parameters such as:

- emission delay: 250 / 500 / 750 / 1000 / 1250 / 1500 / 2000 ms
- boundary offset: -200 / -100 / 0 / +100 / +200 ms

Verify:

- how much wrong-speaker audio remains in the old turn
- whether any audio is lost
- whether any audio is duplicated
- whether retrospective logical repair is possible
- the point after which a late boundary becomes unrecoverable
- how much finalization latency is introduced

This separates:

```text
detector failure
vs
provider policy failure
```

---

# 22. Separate Provider Policy from Detector Quality

The provider-neutral detector output is a source-timeline `SpeakerBoundaryEvent`.

Provider integration is a separate layer that consumes that event.

The experiment must separate:

1. detector boundary accuracy
2. detector causal/frontend delay
3. provider audio-input shaping
4. provider finalization policy
5. provider transcript/timestamp/speaker-metadata behavior
6. reconnect behavior

Do not change multiple factors simultaneously and then attribute the result to the detector.

## 22.1 Deepgram Family

If provider timestamps are relative to transmitted audio, do not compare them directly to the source timeline.

An explicit source <-> provider span mapping is required.

If reconnect/bridging audio exists:

- separate provider epochs
- represent that the same source span may be resent into a new provider epoch
- remove duplicate logical words/cuts after source normalization

## 22.2 Qwen Realtime Family

If provider word timestamps do not exist, do not invent a mapping.

Use a local PCM holdback/ring keyed by canonical source samples.

When the boundary is still inside unsent PCM:

- send the old-speaker prefix to the current turn
- commit at the boundary
- retain the suffix as the start of the next turn

Choose holdback `H` based on:

> measured event lookback + runtime scheduling margin

not on guessed model latency.

## 22.3 Soniox Family

If native diarization/speaker metadata sufficiently solves the product requirement, it is acceptable to conclude that the local detector should not participate in the Soniox path.

Compare provider-native solutions and the local detector fairly and separately.

---

# 23. Resume Experiment Phase Plan

## Phase 0 — Verify Resume Checkpoint and Invariants

Deliverables:

- record exact restart SHA
- verify provenance of retained artifacts
- preserve historical artifacts as read-only evidence
- verify canonical source sample/epoch semantics
- verify B0 replay
- document current VAD logical-finalize semantics
- freeze proposal -> clustering -> fusion -> logical-action contract

Gate:

> Do not run new held-out evaluation until the corrected action/scoring contract is frozen.

---

## Phase 1 — Corrected Re-Score of Existing Development Evidence

Reuse raw detector output/proposal caches where possible.

Avoid unnecessary model reruns.

Deliverables:

- causal detector-detector clustering
- debounce/refractory
- representative proposal selection
- detector <-> VAD coalescing
- retain / accelerate-or-replace / add / suppress action accounting
- clean / gap / overlap separation
- historical legacy counts reported alongside corrected metrics
- contamination / same-speaker false-split calculation where data permits
- full LS/ERes operating frontiers

Gate:

> Do not draw a family-level conclusion from one profile.

---

## Phase 2 — Fill Development-Data Gaps

No newly recorded/private product conversation data is assumed or required. Expansion uses controlled synthetic data plus authorized public conversational corpora.

Deliverables:

- metadata-only coverage inventory before expansion
- more independent public conversational source sessions, not merely more transitions from the same meetings
- expanded exact zero-gap / 0-100 ms / 100-300 ms synthetic handoffs
- expanded same-speaker hard negatives matched to positive gap/acoustic conditions
- short-turn, codec/noise, gain, and other controlled stress cases where source material permits
- explicit counts of B0-missed versus B0-already-separated speaker changes
- synthetic + public-real mixed development pool with clear pool roles
- session/speaker-disjoint held-out construction wherever source metadata permits
- exact channel/codec/frontend recipe
- immutable manifests and hashes

Gate:

> Do not freeze a production threshold until the coverage inventory shows adequate exposure to the actual target condition: speaker changes that B0 fails to separate, together with enough same-speaker negative exposure to estimate false-split behavior. Product-domain generalization remains provisional because no newly recorded product-like acceptance data is available.

---

## Phase 3 — Freeze Matched Operating Panel

Deliverables:

- low / medium / high tradeoff point per family
- comparison at common product-harm budgets
- frozen threshold/reducer
- frozen clustering/refractory
- frozen VAD fusion
- frozen timing/scoring contract
- model/config/artifact hashes

Gate:

> Every held-out candidate must be chosen before viewing held-out audio.

---

## Phase 4 — Held-Out Detector/Fusion Evaluation

Deliverables:

- paired per-session results
- contamination reduction
- same-speaker false splits
- causal availability delay
- localization error
- clean/gap/overlap breakdown
- compute/RAM/RTF
- failure examples selected by a predeclared rule
- session-level uncertainty

Gate:

> Mark model selection as provisional if representative product-domain evidence is unavailable.

---

## Phase 5 — Provider-Policy Oracle Validation

Deliverables:

- controlled oracle delay/error replay
- audio/text contamination
- transcript loss/duplication
- unrecoverable late-boundary cases
- finalization latency
- reconnect/time-domain correctness

Gate:

> Demonstrate that the provider policy itself behaves correctly independent of real detector error.

---

## Phase 6 — End-to-End Replay with Frozen Real Detector Traces

Deliverables:

- exactly the same provider-policy code used in Phase 5
- frozen detector-trace replay
- retained real-detector benefit relative to oracle ceiling
- residual contamination
- same-speaker false splits
- finalization latency
- transcript integrity

Gate:

> Recommend implementation only when sufficient product benefit remains after including actual detector error and latency.

---

# 24. Required Falsification Tests

The preferred explanation is:

> **Speaker-specific evidence exists, and a stable causal policy can convert that evidence into logical boundaries that reduce mixed-speaker turns relative to VAD-only.**

The experiment must be capable of falsifying this explanation.

## F1. Same-Proposal Policy Ablation

Replay the same raw proposal stream through:

1. naive actionization
2. + detector clustering/refractory
3. + VAD coalescing
4. + VAD replacement/acceleration
5. full fusion

Purpose:

> Separate whether improvement/degradation comes from model signal or policy.

## F2. Same-Speaker Acoustic Negative Control

Measure detector reaction to the same speaker under:

- gain changes
- prosody changes
- codec changes
- noise changes

If the detector reacts nearly the same way as it does to true different-speaker transitions, it is difficult to claim that the product signal is speaker-specific.

## F3. Segmentation-Frequency Control

Create heuristic/pseudo boundaries without speaker information at a similar cut frequency.

If downstream benefit is similar, the improvement may come from:

> simple segment shortening rather than speaker recognition

## F4. Add vs Accelerate Decomposition

Separate:

- benefit from newly recovering B0 misses
- benefit from finalizing the same logical boundary earlier than VAD

If most benefit comes from acceleration, reflect that fact in the center of the product design.

## F5. Overlap-Separated Headline

Recompare families after removing interruption/overlap from the clean-handoff headline.

If the family advantage disappears, the earlier headline depended heavily on overlap signal rather than product-separable cases.

## F6. Session Robustness

On the real conversational subset, check:

- leave-one-session-out sensitivity
- session-block bootstrap

If removing one meeting reverses the conclusion, family-selection evidence is not yet sufficient.

---

# 25. Questions That Must Be Answered by Experiment Completion

1. How much mixed-speaker logical-turn contamination occurs under B0?
2. At the same same-speaker false-split budget, how much contamination does LS reduce and how much does ERes reduce?
3. How much of the benefit comes from true B0-miss recovery versus VAD acceleration/replacement?
4. After causal clustering, how many independent bad logical decisions remain?
5. How do results differ across clean handoff / gap handoff / interruption onset?
6. Does LS provide enough additional product value to justify stateful inference/runtime complexity?
7. After stabilizing proposal policy, does ERes provide enough product value with a simpler structure?
8. What are the finalist's p50/p90/p95 localization, causal availability, and scheduling delays?
9. Do the results hold across independent public real sessions and the controlled synthetic strata that target B0-missed handoffs and same-speaker negatives?
10. Can each provider lifecycle implement the boundary as a real logical turn without audio loss or duplication?
11. Which exact detector + policy should be handed to the follow-up implementation issue, or should no local detector be selected?

---

# 26. Levels of Conclusions That Are Allowed

The following are separate conclusions:

1. a model family contains useful speaker-change signal
2. a reducer/proposal policy converts that signal into stable proposals
3. VAD fusion converts proposals into better logical turns
4. the provider lifecycle implements those logical turns correctly in audio/text semantics
5. the overall system is strong enough to be a product implementation candidate

Success at one stage does not automatically imply success at the next.

---

# 27. Forbidden Interpretations

The following conclusions are not allowed.

### Forbidden 1

> LS only catches four changes.

Four is the result of the conservative ratio-selected point. The LS high-recovery point recovered 96 B0 misses.

### Forbidden 2

> ERes reaches 82.6%, so it is product-ready.

The aggressive point produced a very large legacy unmatched/repeated-reaction count.

### Forbidden 3

> 4,877 unmatched = 4,877 harmful user-visible false splits.

The count mixes repeated detector events, pause cuts, acceleration-like events, and other categories.

### Forbidden 4

> Interruption-onset detection = clean mono speaker separation.

Overlap waveforms are not source-separated.

### Forbidden 5

> 210 AMI transitions = 210 independent real samples.

They are clustered in two source sessions.

### Forbidden 6

> One ratio-selected profile = family capability.

Use the operating frontier.

### Forbidden 7

> Provider integration failure = detector failure.

Separate them with oracle provider-policy validation.

### Forbidden 8

> Higher detector boundary recall = proven improvement in product translation/turn quality.

Final logical-turn contamination and provider behavior must also be validated.

---

# 28. Minimum Evidence Required for a Production Recommendation

All of the following must be bound into the result.

## Detector Identity

- exact family
- checkpoint/model ID
- artifact hash
- model provenance/license

## Frontend

- sample rate
- resampler
- feature frontend
- buffering/group delay
- streaming mapping

## Reducer / Proposal Policy

- threshold
- window
- step
- persistence/confirmation
- anchor update if applicable
- raw proposal semantics

## Fusion

- debounce
- refractory
- detector-detector clustering
- detector-VAD coalescing
- retain/replace/accelerate/add/suppress semantics

## Timing

- source epoch/sample contract
- boundary location
- causal availability
- scheduling delay

## Product Results

- per-session raw counts
- contamination reduction
- same-speaker false splits
- clean/gap/overlap breakdown
- causal-delay distribution
- representative-domain result or provisional label

## Runtime

- CPU
- RAM
- RTF
- threads
- concurrency/backpressure smoke evidence

## Provider

- oracle-policy result
- actual detector-trace result
- reconnect behavior
- audio/text loss/duplication invariant

## Follow-Up Implementation

- known failure modes
- exact acceptance-test cases
- telemetry required in production

Without this information, do not claim:

> `production selection complete`

The outcome must instead be:

> `no selection yet` or `provisional`

---

# 29. Immediate Resume Procedure

1. Return to the intended earlier checkpoint.
2. Record the exact SHA.
3. Preserve existing development raw artifacts as historical evidence; do not overwrite them.
4. Do not freeze the previously ratio-selected LS/ERes pair directly for held-out evaluation.
5. Verify whether existing raw detector outputs or proposal caches are sufficient for corrected scoring.
6. Define detector-detector causal clustering/refractory policy first.
7. Freeze VAD fusion in terms of `retain / accelerate-or-replace / add / suppress`.
8. Separate clean / gap / overlap scoring.
9. Recompute contamination / same-speaker false splits to the extent supported by existing data.
10. Mark annotation-insufficient cases as unscorable rather than inferring missing labels.
11. Rebuild the full LS/ERes development frontiers.
12. Build the coverage inventory before adding data: independent public sessions, zero/near-zero-gap handoffs, same-speaker hard negatives, short turns, and B0-missed transitions.
13. Fill only the underrepresented strata using controlled synthetic generation and authorized public conversational corpora; do not require newly recorded/private product conversations.
14. Compare families at the same product-harm budget.
15. Only then decide whether additional model execution is necessary.
16. Run held-out only after threshold/reducer/fusion/scoring freeze.
17. Validate provider integration with oracle traces before connecting real detector traces.

---

# 30. Compact Handoff for the Implementation/Experiment Agent

```text
Resume the speaker-change-based STT turn-boundary experiment from GitHub issue #51
at the designated earlier checkpoint.

The product goal is to reduce the case where sequential speech from multiple speakers
is mixed into one STT/translation turn because VAD continues to see uninterrupted
speech. General diarization and real-world person identity are not the goal.

The primary product tradeoff is false merge / speaker contamination versus same-speaker
false split, with causal boundary delay as the third major axis.

Preserve the existing 1,369-profile development sweep as evidence. However, do not use
the rule that selects one family representative by 500 ms recovery / legacy unmatched
boundary ratio. The conservative LS 4-recovery point does not represent LS family
capability. The LS high-recovery point recovered 96 B0 misses within two seconds.
ERes recovered up to 220 B0 residual misses and therefore showed strong speaker-change
signal, but aggressive policy produced an explosion of repeated/unmatched events.

Legacy additional/unmatched boundary count is not operational harm. Raw model/reducer
events are proposals, not product cuts. Apply detector-detector clustering, debounce,
and refractory causally, then score only final logical actions.

Do not restrict the VAD-detector relationship to 'existing VAD + extra cut'. If speaker
evidence can finalize the same logical turn earlier than a later VAD boundary, explicitly
credit that as acceleration/replacement benefit.

Do not combine clean handoff, gap handoff, and interruption/overlap onset into one
headline. Overlap-onset detection is evidence of change signal, not success at mono
source separation.

Primary product evidence is wrong-speaker speech contamination or multi-speaker
logical-turn rate, same-speaker false logical splits, causal delay, and per-session
uncertainty. Keep speaker-change recall as a diagnostic.

Use session-level uncertainty for public real-conversation statistics. The existing
development set had 202 synthetic cases and only two AMI source sessions; the 210 AMI
transitions are not 210 independent real samples. Do not require newly recorded/private
product conversation data. Instead, expand independent public sessions and controlled
synthetic coverage, especially zero/near-zero-gap handoffs and same-speaker hard negatives.
Inventory B0-missed transitions explicitly; data volume alone is not the objective.

Compare LS and ERes on operating frontiers at the same product-harm budget. Freeze the
reducer, clustering, VAD fusion, timing, scoring, and candidate panel before held-out
evaluation.

Separate provider-policy failure from detector failure. Validate the provider lifecycle
first with oracle boundary traces having controlled delay/location error, then replay
frozen real detector traces through exactly the same policy code.

A production recommendation must bind the exact model/profile/frontend, causal timing,
clustering/fusion rules, contamination/false-split results, session-level held-out
evidence, runtime cost, provider-policy evidence, and acceptance tests. If evidence is
insufficient, conclude explicitly with no-selection or provisional status.
```

---

# 31. Source Provenance

Primary planning source:

- GitHub issue #51
- `Experiment v2: speaker-change turn boundaries (LS-EEND vs ERes2NetV2) and streaming ASR policies`
- https://github.com/kapitalismho/PuriPuly-heart/issues/51

Historical quantitative evidence preserved in this document:

- 1,369 candidates
- B0 1 / LS-EEND 432 / ERes2NetV2 936
- B0 recall at each deadline
- LS conservative / high-recovery points
- ERes conservative / high-recovery points
- 204 cases / 374 transitions / 43.2 min source duration / 30.4 min active speech
- 202 synthetic cases / 2 AMI source sessions
- transition composition: gap 183 / interruption 162 / clean 29
- exploratory detector-event refractory diagnostic
- analysis showing that legacy unmatched boundary count does not accurately represent operational product harm
- analysis showing that VAD acceleration/replacement was not sufficiently credited by the previous additive scoring

The product metrics, corrected action taxonomy, and clustering/fusion scoring newly proposed in this document are **a corrected contract for the resumed experiment, not already observed results**.
