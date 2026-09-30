# #204 R2 application consumption and R5 feasibility

## Disposition

**Defer a production candidate; retain current behavior pending causal/live evidence.** The model-free application ownership path is executable and preserves the tested native attribution and meaningful text. It does not measure Soniox accuracy or show that continuous recognition is a safe configuration toggle. No finalize controls were removed and sealed-turn overlap was not enabled. A quality–latency operating point is **inconclusive/not run**, not empirically rejected.

Authority: [#204](https://github.com/kapitalismho/PuriPuly-heart/issues/204), especially R2/R5/R6, and [#190 including amendments](https://github.com/kapitalismho/PuriPuly-heart/issues/190). Director-supplied clean baseline: `4dc46fdac8c771d3bb58e46bd44304b86fde9c47`, `soniox_diar_test`. No Git inspection or mutation was performed. The companion [consumption_evidence.json](consumption_evidence.json) freezes SHA-256 identities of the exercised production owners, helper composition, and probe itself. These identify executed bytes rather than inventing a new research commit; integration/research commit ownership remains with the Director.

### Scope and execution environment

Only three new task-local files are owned: [consumption_probe.py](consumption_probe.py), [consumption_evidence.json](consumption_evidence.json), and this report. No production code, existing tests, maintained documentation, GUI, provider network call, credential read, or private recording was changed/used.

Executed on Windows 11 build 22631, Python 3.14.7, NumPy 2.3.5. This is a replacement headless application environment, not the archived Linux/WSL scoring environment. Installed NumPy differs from the current `pyproject.toml` pin (`2.5.1`); the evidence does not assert a locked full application environment. Neither the Soniox JavaScript SDK nor `pyannote.metrics` is invoked by this model-free consumer probe; the corrected SDK/scorer baseline is the separate R1 evidence, not a result reproduced here.

From repository root:

```text
python experiments/ami_benchmark/providers/soniox/gap/consumption_probe.py --output experiments/ami_benchmark/providers/soniox/gap/consumption_evidence.json
python -m ruff check experiments/ami_benchmark/providers/soniox/gap/consumption_probe.py
```

Final smoke receipt: `{"passed_checks": 16, "disposition": "defer; current serial finalize consumer verified, no supported quality-latency production candidate"}`. Focused Ruff: `All checks passed!`. The executable is an investigation harness, not a new maintained regression suite. Its `--output` argument rewrites only the chosen task-local evidence file; without it, JSON is printed. No provider connection is opened, and unavailable physical-capture operations raise if accidentally invoked.

## Exercised actual application owners

The integrated fixture path is:

```text
scripted source spans + synthetic PCM + owned SpeechStart/SpeechEnd
→ PeerAudioSegmentLedger
→ ScopedRecognitionEngine
→ actual _SonioxSession ordered writer / receive loop
→ actual STTScopedTurnNormalizer
→ TranslationChannelOwnerCallbacks.peer_event_handler
→ PeerCaptureSessionOwner.admit_provider_terminal
→ PeerTranslationChannelOwner
→ TranslationTurnLifecycleOwner + PeerSpeakerIdentityAllocator
→ actual OutputRuntime / output projection
→ actual OverlayPresenter snapshot at readable admission
```

The real capture owner's ledger/publication generation are seeded with the fixture ledger; its OS capture/VAD factories are deliberately unavailable. Existing `compose_translation_test_harness` is used solely to compose the real translation/output owners, not to reimplement them. Its direct peer-test STT admission path is not used here: real `TranslationChannelOwnerCallbacks` and real capture-terminal admission are explicitly connected. Invalid publication generations and capture epochs are rejected before real terminal admission; a duplicate terminal produces no second admission.

The WebSocket boundary is an in-memory recording transport; its `send` and `recv` execute the real adapter loops. Native token text/speakers/times and receive opportunities are scripted. There is no acoustic model, actual network, OS capture loop, SmartTurn inference, native renderer, CLI-host launch, or HMD exercise. The repeated-context fixture adds one explicit adapter `context_only` write outside the engine to expose that boundary's handling; it is not claimed to reproduce production context scheduling. Translation uses a controlled deterministic provider double while real concurrent execution, source correspondence, publication order, output, and presentation run unchanged.

This is headless **publication-owner** evidence, not a whole physical microphone/VRChat pipeline certification. Architecture orientation: [system data handoffs](../../../../../docs/architecture.md#data-handoffs).

## R2 findings and traceable examples

### Native timing is retained literally, not canonicalized by its field name

`FinalSpeakerRun.source_start_ms/source_end_ms` originate directly from Soniox `start_ms/end_ms`. `send_turn_audio` validates ownership/sequence but discards `source_ranges` and `context_only`; it does not construct a canonical time map. The normalizer trims outer whitespace, conserves run text, and merges adjacent compatible runs while retaining native run envelopes. Translation segmentation copies the envelopes into child transcripts. Source-text character ranges—not these native times—bind children to translation results and speaker assignments. Presentation chooses styles from attribution/scope/order, not native timestamp position.

The evidence includes both fixture source spans and explicit transmitted-provider sample maps:

| Piece | Provider samples | Modeled canonical source samples (16 kHz) |
| --- | --- | --- |
| Initial real-context fixture | 0–800 | 159200–160000 |
| Content fixture | 800–2400 | 160000–161600 |
| Explicit repeated context, when present | 2400–3200 | 159200–160000 again |
| S200, when present | 3200–6400 | none (synthetic padding) |

These half-open intervals are declared fixture maps whose transmitted byte counts are asserted against the real writer receipt. PCM contents are synthetic, not speech; no reference labels enter production inference. Source spans are intentionally displaced from stream time. Native tokens are deliberately arbitrary, including overrun values, and do not pretend to be service outputs acoustically supported by 150 ms of fixture audio.

Examples from the evidence:

- `A_B_A`: native `Alice ` A `[100,190]`, `Beto ` B `[200,290]`, `Alice` A `[300,390]` survive the normalizer and real translation-child owner. Character ranges are `[0,6)`, `[6,11)`, `[11,16)`; readable styles are gold/cyan/gold. Equal A keys do not merge across B.
- `A_missing_A`: `one unknown again` is conserved; the missing-speaker middle span remains a separate unknown child and renders gold/gray/gold. Missing attribution does not erase the A mapping.
- `overlap_and_zero_duration`: native `first ` A `[100,250]`, `brief ` B `[200,260]`, `yes` A `[260,260]` retain all text, literal times, and terminal overlap flags `[false,true,false]`. Zero-width native times do not delete readable `yes`. Importantly, cumulative **stable updates** here carry overlap flags all false, while terminal token aggregation recognizes the overlap; early stable evidence is not equivalent to terminal evidence.
- `repeated_context_and_padding`: two same-speaker `echo` contributions with identical native `[9000,9090]` become `echo echo`, not one `echo`. The preserved child envelope remains `[9000,9090]`, despite the fixture source being near 10 seconds and transmitted support only 400 ms. No guessed clock correction, clipping, VAD repair, or reference fitting is applied.
- `boundary_whitespace`: raw outer whitespace is stripped to `alpha beta`; internal word separation, speaker separation, and literal native times remain intact.
- `legitimate_repeated_text_and_reuse`: two ordered turns each say `same` with `[100,100]`; both are published (combined evidence string `samesame` represents two event texts, not a fused caption). The turn-local buffer clears and native clock reset-like fixture values do not deduplicate a legitimate successor.

**Classification:** these fixtures establish text conservation, missing-metadata exposure, literal timing propagation, repeated-context susceptibility, and consumer ownership. They cannot diagnose native speaker merge/split, wrong returning-person attribution, ASR omissions, timestamp drift, SDK DER envelopes, or full recognized-word accuracy in actual audio. R1's reference-aligned recognized-word diagnostic and R3/R4's causal/timestamp evidence remain separate responsibilities. Scripted recognized-word identity is not reported as provider accuracy.

### Current consumption is terminal-driven

The native adapter drops non-final tokens. Final native tokens do generate stable scoped updates and usable attribution before the manual completion; the integrated peer callback drops these updates rather than publishing them. `<end>` does not discharge the active turn or its one pending finalize. Only an accepted pending `<fin>` after local seal yields `final`/`empty` with `reuse`, then clears scoped/pending token buffers, pending completion count, last native end time, diagnostics, and active ownership.

Thus “native stable text has arrived,” “application seal,” “actual finalize write,” “terminal receipt,” and “first readable caption” are distinct events. Stable-token availability alone does not demonstrate live caption readiness on the current peer route.

### Completion, cancellation and scope isolation

All following scenarios execute adapter or engine owners, not a modeled state machine:

| Check | Observed outcome |
| --- | --- |
| Textless `<fin>` after seal | `empty` / `reuse`; fresh successor is admitted with clean text |
| Failed textless native request | `failed` / `retire`, epoch ends; old output does not seed replacement |
| Cancelled active turn and late completion | `cancelled` / `retire`; later final text/`<fin>` produces no output |
| Source-discontinuity adapter abort | cancelled retired scope; explicit new connection scope differs |
| Actual engine source discontinuity | ledger fails open source, engine emits textless cancellation, retention returns to zero, old result is suppressed; new capture epoch 2 gets a new provider epoch/speaker scope and only `after loss` |
| Detectable idle late text | epoch ends with `soniox_idle_authoritative_text`; no successor seeded |
| Detectable idle extra `<fin>` | epoch ends with protocol ambiguity; no successor seeded |
| Cross-epoch normalized terminal | `provider_turn_identity_mismatch`, never accepted |
| Wrong publication generation / capture epoch | real capture-terminal admission rejects it |
| Duplicate terminal | real capture-terminal admission returns empty, no duplicate text |
| Late usable speaker after readable gray | remains gray; readable `visible_since` unchanged |
| Same opaque speaker ID in new scope while old colored caption is readable | new caption gray; old caption gold, not cleared/recolored |

A deliberately adversarial `late_after_successor_active` injection also verifies a **limitation**, not a success guarantee: once the successor is active, an unkeyed old-looking native token carrying a stream `request_id` is attached to that active successor. There is no observable per-turn native key allowing the adapter to classify it as old. This is a modeled ambiguity demonstration, **not proof Soniox actually emits old tokens after its accepted barrier**. Known identity/epoch and detectable idle fences work; they do not magically correlate unkeyed tokens after admission. A continuous consumer cannot assume this ambiguity away or turn stream `request_id` into a per-turn identifier.

### First-readable immutable colors and translation opportunity

A separate actual presenter probe first publishes missing attribution in gray, then supplies identified A for the same logical caption. It stays gray without changing readable start time. In the integrated translated A→B→A fixture, three real child tasks reach the translation double concurrently (observed peak 3). The third translation completes first and cannot overtake the first; releasing the first yields an output event while the second is still pending. After release, source text is exactly `Alice Beto Alice` in authoritative order with gold/cyan/gold and three—not speaker-driven duplicate—translation calls.

This exercises existing translation opportunity but does **not** prove that newly arriving STT metadata is reconciled into an already submitted child during a translation wait. That update path is not established here. Metadata arriving after readable admission cannot improve that caption's color under #190. Waiting after a native wrong ID has already finalized does not itself create new evidence.

## R5 measured application path versus modeled/provider time

The following milliseconds are measured `perf_counter` deltas from the beginning of each independent fixture; source/token coordinates and the scheduling of native input are modeled. `first_stable_text` and `first_usable_speaker` coincide for these identified-first fixtures. The per-message native receive times, all PCM/control writes, all normalized events, child runs, and each first-readable snapshot are retained in JSON.

| Fixture | Stable/usable speaker | Local seal | Actual finalize write | Terminal at application sink | First readable | PCM seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A→B→A | 5.023 | 5.409 | 5.519 | 5.854 | 8.760 | 0.150 |
| A→missing→A | 1.363 | 1.708 | 1.803 | 2.111 | 4.657 | 0.150 |
| Repeated text/reuse | 1.447 | 1.608 | 1.713 | 30.531 | 32.454 | 0.250 |
| Overlap/zero duration | 1.324 | 1.950 | 2.045 | 2.346 | 4.720 | 0.150 |
| Repeated context/S200 | 1.251 | 1.481 | 1.595 | 1.874 | 3.321 | 0.400 |
| Whitespace boundary | 1.298 | 1.604 | 1.697 | 2.006 | 4.097 | 0.150 |
| Concurrent translated A→B→A | 1.304 | 1.644 | 1.734 | 2.027 | 4.575 | 0.150 |

These values are illustrative receipts from one finite smoke, **not a provider latency benchmark, repeated estimate, or no-regression claim**. No artificial color wait was added. Existing presenter/output admission runs as-is; this compact smoke does not measure a sustained saturated caption-pacing queue or establish the application's 1.0-second replacement frontier.

In the successor scenario, input admission was attempted at 1.943 ms, the probe deliberately held `<fin>` injection (requested sleep 20 ms; actual scheduler-delayed injection 30.074 ms), and no successor PCM was written during that wait. Its actual PCM write occurs at 32.925 ms after predecessor completion. The 28.818-ms finalize-to-terminal delta contains the **injected barrier hold plus local scheduling/processing**, not Soniox computation. Likewise the translated fixture's first output event at 4.313 ms precedes completion of the later gated translation; first-readable observation is 4.575 ms. Translation text is a deterministic double, so no live translation latency is inferred.

The modeled source speech end is 10,100 ms in fixture source coordinates. It is **not on the measured application origin**, and must not be subtracted from the table to manufacture speech-end latency.

### Coverage, resources and cost

- Exact meaningful text conservation is asserted at normalized terminal, child transcript concatenation, and source-corresponding output. All expected fixture words survive; no A→B→A fragmentation is collapsed into an A-only caption. The missing span stays visible gray, and overlap/zero-width interjections are not dropped. These are conditional software fixture results, not DER/WER/color accuracy on recordings.
- Seven integrated fixture runs transmit **1.400 synthetic PCM seconds total** to in-memory transport doubles. The repeated-context/S200 fixture transmits 0.400 seconds: 0.050 initial context + 0.100 content + 0.050 replay + 0.200 synthetic padding. The other lifecycle/engine probes also perform finite local writes; the 1.400-second figure explicitly covers the seven integrated fixtures only, not all probes or any bill.
- Actual adapter PCM/control ordering is recorded. S200 is exactly 6,400 all-zero bytes at 16 kHz mono PCM16 and immediately precedes finalize. Serial input ACK completion and pending `<fin>` gating are exercised; no future waveform is claimed.
- Adapter queue capacity observed: 258 entries. Each integrated engine fixture's retention high water is 1,600 samples / 3,200 bytes, ending at zero. These finite high waters are not sustained-load or maximum-product-memory proofs. Recorded context writes bypassing the engine are separately identified above.
- Whole probe after imports: 0.131877 seconds; Python `tracemalloc` current 542,396 bytes, peak 756,072 bytes, including evidence capture and all harness bookkeeping. This is not process RSS, isolated product allocation, provider memory, or a production processing overhead estimate.
- Provider calls: **0**. Actual active WebSocket duration: **unavailable/null**, not zero-valued live evidence. The transport is an in-memory double. No billing savings, invoice amount, provider processing time, or extra-model cost is inferred from PCM volume. Full-stream billing assumptions require the bounded authorized live run and vendor billing semantics; no concurrent observer is introduced.

Required R5 arrival/resource measurements still **unavailable**: independently anchored physical speech end; measured live stable-token/speaker arrival; actual service processing and network latency; real socket lifetime; sustained successor backlog/retained maps and PCM under the candidate; candidate-versus-control DER/word identity/WER/colored coverage/unknown and mixed exposure; live translation/caption-pacing added waiting; supported billing/quota effects. Final-only transcripts cannot replace these traces.

## Smallest supported feasibility conclusion

No quality-latency candidate can be adopted from this probe. The smallest **currently executable** contract-preserving path remains the current serial manual-finalize barrier. A narrowly less aggressive experimental finalization/PCM policy can only be shortlisted after the matched R3 comparison establishes a useful effect and R4 bounds timing; this report selects no arbitrary interval or production default.

If later evidence supports structurally continuous consumption rather than a bounded finalize-policy intervention, its minimal ownership amendment is **not** “remove finalize” or “enable overlap”:

1. Account for stable native contributions and speaker evidence against locally owned, sealed source pieces with a protocol-supported stream-to-source map, retaining overlap/unknown states. Current stable and terminal overlap metadata differ.
2. Specify a bounded, independently terminating textless/failed/late local completion contract, since the current adapter only ends the active turn at pending `<fin>` and clears it on `reuse`.
3. Keep serial wire/control ordering, source/capture/publication/provider epochs, cancellation authority, bounded PCM/map/queue retention, and successor admission; do not infer a native turn key from stream `request_id`.
4. Preserve existing text-run/translation-child correspondence and source-order color allocation. Use existing translation/pacing opportunities only if an actual late-evidence-before-publication path is demonstrated. No fixed color wait or readable recoloring.
5. Measure matched acoustic quality, correctly colored coverage, gray/mixed exposure, omissions/duplications, startup/returning-speaker continuity, arrival backlog, and resource/cost effects together. Gray or missing short turns cannot count as a quality improvement.

Whether such an amendment, later publication deadline, or background transmission is acceptable is a maintainer decision. It is neither implemented nor authorized here.

## Completion status and Director documentation proposal

**Passed:** seven integrated application fixture scenarios; nine lifecycle/presentation/ambiguity checks (16 total assertion-backed checks); isolated Ruff. The active-successor ambiguity check passes by reproducing the bounded known limitation, not by claiming late-native-text isolation that is unobservable. No maintained tests were added or rerun.

**Failed:** no outstanding smoke assertion or lint failure. Earlier import-format diagnostics were repaired before the final receipt.

**Blocked/inconclusive:** R5 quality–latency candidate and provider arrival/resource conclusions require causal comparison, authorized waveform/runtime, configured service, and explicit finite quota authorization. These are not supplied/authorized for this worker; no credential discovery or provider contact was attempted.

**Not run:** paid Soniox matrix, scorer/SDK reproduction, acoustic accuracy/DER/WER evaluation, independent drift calibration, sustained live resource benchmark, physical microphone/GUI/native renderer/VRChat/HMD checks, commit/merge/deployment. R1/R3/R4 have other designated owners; this report does not certify their outcomes or claim #204 complete.

**Proposed maintained documentation update, for Director only:** link the new issue-local consumption report from the final #204 experiment completion record and explicitly separate stable-token arrival, terminal admission, and first-readable publication. State that `FinalSpeakerRun.source_*_ms` is currently native stream time on this Soniox path, not a canonical source conversion, and that detectable idle-output fencing does not provide per-turn native correlation. Do not amend `docs/architecture.md`, endpoint defaults, #190, or production behavior merely to record this investigation.

## Source citations (frozen baseline)

- [Soniox receive/drop/native-time handling, lines 464–633](https://github.com/kapitalismho/PuriPuly-heart/blob/4dc46fdac8c771d3bb58e46bd44304b86fde9c47/src/puripuly_heart/providers/stt/soniox.py#L464-L633)
- [Manual completion, run aggregation, reuse clearing, lines 635–754](https://github.com/kapitalismho/PuriPuly-heart/blob/4dc46fdac8c771d3bb58e46bd44304b86fde9c47/src/puripuly_heart/providers/stt/soniox.py#L635-L754)
- [Ordered writes, padding/finalize, lines 360–413](https://github.com/kapitalismho/PuriPuly-heart/blob/4dc46fdac8c771d3bb58e46bd44304b86fde9c47/src/puripuly_heart/providers/stt/soniox.py#L360-L413) and [source-range discard/seal/S200, lines 992–1077](https://github.com/kapitalismho/PuriPuly-heart/blob/4dc46fdac8c771d3bb58e46bd44304b86fde9c47/src/puripuly_heart/providers/stt/soniox.py#L992-L1077)
- [Normalizer terminal conservation, lines 149–215](https://github.com/kapitalismho/PuriPuly-heart/blob/4dc46fdac8c771d3bb58e46bd44304b86fde9c47/src/puripuly_heart/core/stt/scoped_normalizer.py#L149-L215), [timed speaker runs, lines 350–412](https://github.com/kapitalismho/PuriPuly-heart/blob/4dc46fdac8c771d3bb58e46bd44304b86fde9c47/src/puripuly_heart/core/stt/scoped_normalizer.py#L350-L412), and [identity fence, lines 488–490](https://github.com/kapitalismho/PuriPuly-heart/blob/4dc46fdac8c771d3bb58e46bd44304b86fde9c47/src/puripuly_heart/core/stt/scoped_normalizer.py#L488-L490)
- [Actual peer callbacks drop stable updates and admit terminals, lines 105–116](https://github.com/kapitalismho/PuriPuly-heart/blob/4dc46fdac8c771d3bb58e46bd44304b86fde9c47/src/puripuly_heart/core/orchestrator/translation_channel_callbacks.py#L105-L116)
- [Capture generation/identity/duplicate/source-order terminal admission, lines 702–758](https://github.com/kapitalismho/PuriPuly-heart/blob/4dc46fdac8c771d3bb58e46bd44304b86fde9c47/src/puripuly_heart/core/runtime/peer_channel.py#L702-L758)
- [Peer terminal to transcript, lines 415–459](https://github.com/kapitalismho/PuriPuly-heart/blob/4dc46fdac8c771d3bb58e46bd44304b86fde9c47/src/puripuly_heart/core/orchestrator/peer_translation_channel.py#L415-L459)
- [Source-run segmentation, lines 78–172](https://github.com/kapitalismho/PuriPuly-heart/blob/4dc46fdac8c771d3bb58e46bd44304b86fde9c47/src/puripuly_heart/core/orchestrator/translation_turn.py#L78-L172) and [child timed-run and text-range propagation, lines 809–842](https://github.com/kapitalismho/PuriPuly-heart/blob/4dc46fdac8c771d3bb58e46bd44304b86fde9c47/src/puripuly_heart/core/orchestrator/translation_turn.py#L809-L842)
- [Speaker allocation in source order, lines 28–66](https://github.com/kapitalismho/PuriPuly-heart/blob/4dc46fdac8c771d3bb58e46bd44304b86fde9c47/src/puripuly_heart/core/speaker_identity.py#L28-L66)
- [First-readable style freeze/scope handoff, lines 821–874](https://github.com/kapitalismho/PuriPuly-heart/blob/4dc46fdac8c771d3bb58e46bd44304b86fde9c47/src/puripuly_heart/core/overlay/presenter.py#L821-L874)
- [Engine cancellation authority, lines 438–463](https://github.com/kapitalismho/PuriPuly-heart/blob/4dc46fdac8c771d3bb58e46bd44304b86fde9c47/src/puripuly_heart/core/stt/scoped_engine.py#L438-L463), [seal/barrier wait, lines 624–662](https://github.com/kapitalismho/PuriPuly-heart/blob/4dc46fdac8c771d3bb58e46bd44304b86fde9c47/src/puripuly_heart/core/stt/scoped_engine.py#L624-L662), and [epoch/generation event fences, lines 893–987](https://github.com/kapitalismho/PuriPuly-heart/blob/4dc46fdac8c771d3bb58e46bd44304b86fde9c47/src/puripuly_heart/core/stt/scoped_engine.py#L893-L987)
- [Ledger source-loss terminal, lines 381–398](https://github.com/kapitalismho/PuriPuly-heart/blob/4dc46fdac8c771d3bb58e46bd44304b86fde9c47/src/puripuly_heart/core/audio/ownership.py#L381-L398)
- [Existing real-owner test composition, lines 731–860](https://github.com/kapitalismho/PuriPuly-heart/blob/4dc46fdac8c771d3bb58e46bd44304b86fde9c47/tests/helpers/translation_owners.py#L731-L860)
