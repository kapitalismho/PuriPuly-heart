# Scribe v3 integration and matched follow-up

## Authority and status

The owner requested implementing the discussed Scribe v3 integration, repeating the same experiment against the retained results, investigating observed problems, and explicitly checking readability when speaker information arrives late, including expired captions reappearing. This authorizes local product implementation and bounded public-audio/service verification. It supersedes the earlier analysis-only boundary for this follow-up, not the historical findings in [the original report](issue441-results.md).

The integration, bounded repair review and owner-reduced comparison are complete locally: one six-clip recognition pass, three synthetic Desktop scenarios, and three selected actual-MT/Desktop cases. This record does not claim release, deployment, PR publication or issue closure.

- Implementation baseline: `4dab89126bcee83bc501fd95f5bbd742ef861d37`, a clean merge of the retained experiment head `b28e92edfdb06dca7d74f2815a3e964feb2a3002` with `dev` at `fc8b0b1eb2312cc1bf56ddd5c3def7d379f1da65`.
- The existing Scribe route moves to v3. Do not change the default provider, add a v2 compatibility mode, or modify the user's installed settings.
- Preserve existing audio boundary control, preview timing, source grouping, continuity, bounded context, output isolation and caption reading protection. Add no speaker-triggered cut, attribution wait timer, arbitrary speaker smoothing, or reference-derived correction.
- Turn prediction remains explicitly rejected. Do not implement or rerun its policies.
- [Soniox input loss and measurement follow-up #454](https://github.com/kapitalismho/PuriPuly-heart/issues/454) remains separate. Do not silently apply its acoustic-reset counterfactual here.
- [SteamVR handoff and P05 work #453](https://github.com/kapitalismho/PuriPuly-heart/issues/453) remains separate; do not change native retry or dispatch policy here.
- No remote publication, push, merge, release or tracking transition is authorized by this local implementation request. The existing #441 Project item remains Backlog rather than claiming a remote scope amendment.

## Owner amendment: bounded verification

The owner explicitly requested reducing rigor and verification to a reasonable level rather than expanding parallelization. This supersedes the exhaustive planned scenario matrix and the requirement to execute all seven natural MT/display intervals. It does not waive a working integration, the same six-clip comparison, fixing demonstrated in-scope bugs, or direct readability observations.

Run one fresh candidate ASR pass over the six frozen clips and compare with retained results. Limit actual MT/natural Desktop checks to `mt-01`, `mt-02`, and `mt-07`: a single-speaker case, a mixed-speaker case, and the known Japanese fragmentation case. The other four selected paid/display intervals and a complete historical E0 replay campaign are superseded, not pending completion gates. Reuse existing checks and add or rerun only checks affected by a concrete failure or the remaining implementation delta. Do not extend stress, retention-churn, all-provider, or combinatorial protocol matrices.

The bounded connected Desktop proof covers a successful visible late identity update, a live source-group split/replacement, and a late update/split after expiry that cannot revive the old lineage while a fresh caption still appears. Observe ordering, visible disruption and fragment behavior within those cases. Existing clear/eviction/other synthetic evidence may be retained but is not a reason to start another campaign. Keep sensible isolation, source/candidate identity, original evidence, failure reporting and cleanup.

## Required integration outcomes

- Preserve normal text availability and start translation at the existing policy's admission point without waiting for speaker information. Use already available metadata before dispatch rather than deliberately issuing a redundant whole-text translation.
- Correlate committed text and delayed diarization by actual provider-session scope and `segment_index`. Text equality, event adjacency, reference labels and guessed timestamps are not identity.
- Separate recognition completion from permitted caption-attribution revisions. Do not block subsequent input, reopen retired recognition turns, or let a previous session update current captions.
- An identity-only update must not trigger translation or independently restart fragment lifetime, registration, ordering or presentation. A changed source partition may require translation of the changed complete source groups; never slice translated strings by source offsets.
- Preserve exact source coverage, once-only history/context contributions, admitted source order, frozen request contexts and stale-result rejection across regrouping, cancellation and translation failure. Logical source order and ready-overlay registration order are separate contracts.
- Expired, cleared, evicted or superseded caption lineages must stay dead, including new child identities produced by late grouping. Positive late updates to still-eligible captions must also work; dropping every update is not a solution.
- Canonicalize received speaker merges within the actual session without reviving past presentation or launching broad historical retranslation. Unknown/mixed attribution must not discard otherwise usable text.
- Preserve Self and unrelated-provider behavior, provider timeout/recovery, bounded state and complete shutdown.

Current presentation authority is [#403](https://github.com/kapitalismho/PuriPuly-heart/issues/403) and [#421](https://github.com/kapitalismho/PuriPuly-heart/issues/421), not the superseded three-second paging description in #356. Preserve ordinary capacity-safe two-line fragments, eight-second visible-content idle expiry, reading protection for replacement, Self priority and the one-second maximum new-Peer wait. Eight seconds is not minimum dwell. Correct surviving visible/pending registrations in place without resetting their order or wait origin; genuinely new tails append after existing waiters. Do not add contiguous-group priority or reserve places for unready translations. The complete original appears only on display fragment zero and never moves to a surviving continuation. Full retranslations may repeat previously read meaning; semantic deduplication is not part of the existing contract.

## Comparison and presentation obligations

Use the same six frozen 120-second public clips and existing annotation/scoring contracts. Preserve all original inputs, selection records, raw provider events, fixed-point results and the separately frozen lexical supplement. A new result is not a rewrite of the historical result.

Run the candidate Scribe production path with its declared product input policy and actual controller. Distinguish it from the original continuous-input E0 arm: same source audio is not identical transmitted audio. Use retained native events where appropriate to isolate application-policy changes, clearly separating controlled replay from fresh provider execution. Reuse unchanged N/S0/S1/E0 evidence rather than rerunning rejected or unrelated arms.

Compare native attribution, application source/caption composition, text availability, metadata delay, translation request counts and actual translation/display outcomes. Retain per-language denominators and online/eventual distinctions. A score change alone is not proof of recognized-content or translation-quality improvement.

Use current [verification guidance](../../docs/verification.md) and supported application controls. Observe the intended candidate and actual Desktop surface with isolated public/synthetic input. Follow the same source/segment/caption lineage from provider or replay ingress through production owners to the screen. Record replaced boundaries; screenshots of independently reconstructed cards do not establish integration.

Readability coverage follows the bounded owner amendment above, under the current fragment and registration policy. Do not revive the retired #373 harness or desktop reproduction command as a maintained dependency.

Investigate concrete failures and repair in-scope product defects before final review. Preserve failed attempts and limits. Provider-native attribution errors and any proposed new smoothing/input policy require explicit interpretation rather than a hidden heuristic or weakened acceptance.

## Evidence and completion

### Implementation and repairs

Implementation commit: `48b0db31e58e0a2d06c3ed72734a77afbd455057`. Repaired application candidate: `00ee2630083f36b4771a635bb8bdb00969472c96`.

- The existing route uses `scribe_v3_live`; saved settings migrate to version 56. Peer requests native diarization/timestamps; Self and rolling capture remain non-diarized. The default provider and installed user settings were not changed.
- Raw WebSocket transport replaces the SDK wrapper that did not expose the needed diarization/events. Text completion and post-commit speaker evidence are separate. Native association uses actual session and segment index, with confirmed local writes and bounded retention.
- Identity-only changes update attribution without another MT request or lifetime reset. Changed partitions retranslate complete source groups. Caption lineage prevents expired families from returning through newly created child IDs.
- Independent whole-checkpoint review found three concrete defects, all repaired and independently verified: accepted split translations could be lost on clean socket retirement; unknown continuation could follow the split head rather than the adjacent tail; experimental replay omitted the required merge keyword and lost merge semantics. Publication now retains accepted source authority across normal socket retirement, continuation follows the source boundary, and replay retains typed merge evidence.
- The live experiment also exposed an observer counting error: an authoritative empty recognition result was classified as a missing indexed terminal. The recorder now distinguishes authoritative empty settlement from missing/failed/wrong-session evidence. Original failed status and process exit remain preserved.

### Same-source recognition comparison

One pass over the same six 120-second public clips; no quality retry, new corpus, Soniox rerun or turn-prediction run. Capture used commit `48b0db31e58e`; the later repairs do not change its native recognition path. All six source EOFs and normal WebSocket closes were observed. The cohort exited 1 because EN diagnostic was labeled partial by the observer: all 26 commits actually settled, including one authoritative empty result without `native_segment`. This was not a provider timeout or dropped terminal.

C/W/U means correct / wrong / unassigned at the unchanged 120 point anchors. Online is the original reference-span end plus one second; eventual includes later native evidence.

| Arm | Online C/W/U | Eventual C/W/U |
|---|---:|---:|
| N, retained activity-only model | 119/0/1 | 119/0/1 |
| S0, retained Soniox continuous input | 49/2/69 | 50/3/67 |
| S1, retained Soniox product input | 46/4/70 | 46/4/70 |
| E0, retained Scribe continuous input | 86/1/33 | 98/0/22 |
| I441, integrated Scribe product input | 93/2/25 | 99/1/20 |

N does not recognize text. Soniox's native timestamp coverage and application envelopes are different representations; its unassigned points are not a count of missing words or omitted input. Those distinctions and the confirmed Soniox input-loss investigation remain in #454. This table is not a blanket provider-quality ranking.

I441's final result gains one correct point but also introduces one wrong point relative to E0. This is not persuasive evidence of a quality improvement. I441 eventual unassigned points comprise 16 without timed coverage and four without a speaker; neither category alone proves deleted text. Per-language eventual C/W/U: EN 35/1/4 of 40, JA 33/0/7 of 40, KO 15/0/5 of 20, ZH 16/0/4 of 20. The 24 fixed change/re-entry edges had 15 available observations in each view; only 2 online and 3 eventual pairs were correct.

The independently frozen lexical supplement also shows no clear improvement:

| Eligible lexical units | E0 eventual C/W/U | I441 eventual C/W/U |
|---|---:|---:|
| EN: 40 words | 38/0/2 | 37/1/2 |
| KO: 20 words | 17/0/3 | 17/0/3 |
| JA: 36 phrases | 17/0/19 | 17/0/19 |

These heterogeneous, partly overlapping units are not WER/CER or DER. There is no Chinese lexical ground truth.

Input is deliberately not identical at the provider boundary: I441 used current Silero/SmartTurn/manual product control, while E0 sent continuous PCM. Of 720 source seconds, 40.468 were not transmitted; 3.000 seconds were replayed context, yielding 682.532 seconds of completed local PCM writes without generated padding. These are input-policy measurements, not recognized-word loss, server admission or billing. I441 used auto-language ASR; its application harness used an explicit per-clip MT source language. The server echoed language detection disabled, unlike E0's explicit enabled request. These differences prevent attributing score changes solely to integration.

Commit-to-first timed native identity was 45.3 ms median / 71.3 ms p95 over 125 of 127 indexed commits, versus retained E0's 60.9 / 277.2 ms over 126 of 128. This is metadata delay after committed text, not speech-to-caption latency or a causal speedup estimate.

### Application workload and end-of-input limit

At the capture candidate, all 121 ordinary nonempty indexed commits received current Peer speaker evidence. Five source-EOF commits received native diarization only after source authority ended, so none entered current Peer attribution. Do not equate eventual native score with eventual on-screen speaker coverage, or add an unapproved metadata wait to hide this boundary.

The 121 admitted updates produced 71 identity-only changes without new backend work, 49 partition changes that started 151 repair requests, and one no-change update. Overall, the controlled backend returned 349 requests: 57 final, 49 continuation, 72 preview and 171 repair. This establishes substantial retranslation work, not paid cost or a universal multiplier. The new strategy preserves current no-wait policy; it does not eliminate provider fragmentation or make regrouping free.

### Readability and verification

The actual production Desktop/Flet surface ran three controlled cases at `48b0db31e58e`, each once and with exit 0:

- Late identity: the same readable card changed gray to gold; one MT request total, unchanged registration and exact eight-second expiry deadline.
- Live split: a combined card became two readable source-group captions with their own originals, in the observed order. Ten transition samples over 1.44 seconds showed only initial and settled states, with no sampled blank, clipping, reorder or old return. This does not prove absence of sub-sample transients.
- Expiry: late identity and split evidence caused no new old-group MT, pending child or scene. Seven samples stayed blank; a fresh caption then appeared alone.

These cases used synthetic owned input, controlled raw Scribe frames and deterministic MT through the real capture ledger, adapter, scoped callbacks, caption/output owners, authenticated bridge and Desktop renderer. They are not live-provider-to-GUI latency or headset proof. Owned renderer trees exited gracefully, and manifests, scratch roots and backdrops were removed.

Focused product evidence: final initial gate 43 passed; earlier overlapping provider/caption gate 323 passed and settings/factory gate 460 passed. After repair, 37 distinct caption checks passed across the composed Scribe and native-caption files; the two new continuation cases were rerun after correcting probe assertions about an empty test prompt and serialized UUIDs. The accepted-text retirement test, expired-family rejection and fresh-socket successor passed. Comparison tooling's final focused gate: 10 passed. Scoped Ruff/Black checks passed; no full-suite or type-check claim. Independent repair review returned `repair_verified`.

### Selected actual translation

At repaired candidate `00ee2630083f`, the three frozen intervals selected 15 complete application requests: 1 for `mt-01`, 5 for `mt-02`, and 9 for `mt-07`. All effective bodies were checked with the production request builder, including explicit source/target/channel, bound prompt variables and exact model/provider options. Repaired-candidate replay matched every selected lineage and body before paid dispatch. A compressed-clock probe missed two Japanese requests; ordinary recorded pacing matched all nine. That probe scheduling difference did not justify changing the product or rerunning recognition.

The original `google/gemma-4-26b-a4b-it` profile was retained: LATENCY/DEFAULT, temperature 0.6, reasoning none, default 100 output tokens and 30-second timeout. One exact historical body/response was reused for `mt-01`; 14 fresh calls all completed with HTTP 200 and the requested model, served by NextBit/Makora. Fresh call durations were 0.396–1.520 seconds. Returned usage totals: 13,357 input tokens, 244 output tokens and USD 0.00115853 reported cost. This is the response-reported charge for these calls, not an invoice or extrapolated product cost.

| Case | Observed translation outcome | Comparison limit |
|---|---|---|
| `mt-01`, short single speaker | `Okay.` → `알겠어요.`; exact valid historical response reused. | E0's complete selected group was longer. This is not a matched sentence-quality improvement. |
| `mt-02`, English speaker change | Complete groups remain understandable: the presentation/borrowing request and the affirmative reply are translated separately. Later joining/regrouping changes the reply wording again. | Five application requests rather than E0's three prepared whole/group units; source context and later group coverage differ. |
| `mt-07`, Japanese fragmentation | Native boundaries split `やっぱ楽しみ方` into `やっ / ぱ楽しみ / 方…`, producing `앗 / 아, 기대된다 / 방식…`. Another group ends at `健` and leaves that character untranslated. | E0 already fragmented `楽しみ方` and `なんか`; the current policy has not fixed that problem. Native recognized text also changes `ラーメン` to `ラム`; whole/group MT interpret the latter as rum/RAM respectively. |

The result supports integration feasibility, not a translation-quality win. The product faithfully uses native speaker partitions, but those partitions can fall inside words. Preserving source coverage and translating each complete *speaker group* does not restore a broken linguistic unit. A new smoothing rule, a metadata wait, or using speaker IDs only for styling would change policy and needs an owner decision; none was silently added.

### Connected natural Desktop observations

All three cases ran serially at `00ee2630083f` through the repaired admission replay, real caption/output owners, authenticated bridge and actual Desktop/Flet child. The same selected native-capture lineage and effective request bodies reached 1/5/9 retained real replies. Full captured warm-up was preserved through the last selected admission. Each selected response was delayed by its measured service duration; nonselected warm-up/surrounding requests used controlled source echoes. Those echoes are not translation-quality evidence. This is paced replay, not live ASR-to-MT-to-GUI latency.

The three runs exited 0; every renderer reported graceful acknowledged exit 0, with scratch, manifests and owned backdrops removed. There were 17/27/37 pixel samples respectively, targeting 150 ms spacing during selected output, plus independently inspected PNGs.

- `mt-01`: one readable `알겠어요.` / `Okay.` card.
- `mt-02`: both real speaker groups were readable without clipping, but the affirmative reply appeared before the presentation/borrowing request. Its translation completed faster; current first-renderable registration order is not source-speech order. The reply then visibly changed wording after another group revision.
- `mt-07`: the actual screen displayed short fragments such as `앗`, `아, 기대된다`, and `, 뭐`, followed by the RAM interpretation. At two adjacent samples approximately 0.19 seconds apart, both cards changed. This is a rapid-content/reordering readability problem, even though no intermediate blank was sampled. The existing ready-first/two-slot/one-second-wait policy does not turn erroneous within-word speaker boundaries into readable sentences.
- No departed caption identity returned in the sampled selected transitions. The deliberate late-after-expiry/no-revival proof remains the separate synthetic case above. Absence of a sampled blank is not absence of every possible transient.

Actual visual references: `display/natural-2/mt-01/natural-005.png`, `mt-02/natural-013.png`, and `mt-07/natural-011.png` / `natural-014.png`. Per-case `receipt.json` retains request matches, presenter/bridge state, pixel timestamps and cleanup. The initial Desktop wrapper attempt stopped before renderer startup because it read the capture pin from the wrong JSON level; that probe error was corrected, with no extra MT calls.

**Decision:** keep the default provider unchanged. Native speaker IDs are usable for live attribution, but immediate source splitting is not yet a quality recommendation for difficult Japanese dialogue. Changing word-boundary treatment, queue/order policy, or attribution waiting is a separate product choice rather than an unreported fix to this experiment.

### Evidence locations and documentation

Local retained evidence under `.tmp/issue441/scribe-integration/`:

- `experiment/source-method-v4.json`: file SHA-256 `2bf749d22da6f5265872a4525c6cf3ea3500565c2a16c77a41aa9835338d602b`; frozen original inputs and method.
- `capture/live-1/`, `experiment/capture-live-1.stdout`, `experiment/live-result-1.md`, `experiment/live-summary-1.json`: raw capture, original failure receipt and interpreted boundary evidence.
- `scores/comparison-1.json`: SHA-256 `3449c784664c5cfa78d6c846dedd7863f064acd07180f589de8b449eaff7332b`; unchanged fixed-point/lexical scoring and retained baselines.
- `replay/i441-1/selection-index.json`: three selected packs, with separate capture and repaired-application pins.
- `implementation/evidence.md`, `display/candidate-synthetic-results.md`, and per-case PNG/receipt directories: executed checks and actual Desktop observations.
- `display/natural-2/prepared.json`, `mt-result.json`, and `mt-07-gate-paced.json`: validated effective bodies, actual MT responses/usage, and repaired-candidate request matching. The earlier compressed-clock gate attempts remain distinct.

`docs/verification.md` now gives a bounded delayed-speaker scenario. Protected documentation was not edited: `docs/architecture.md` does not yet describe the new bounded post-commit speaker-evidence path; `native/overlay/README.md` omits the new still-live lineage rule in its abbreviated Python-ownership summary. Neither implies a native protocol change. User-visible changes are the Scribe v3 route/label and live Peer attribution/group corrections; the default provider and user's saved settings were not switched.

### Completion receipt

Independent terminal evidence audit accepted candidate `7acc9268d9c7eb9d9bb13c5678497e28bdb9d14f` under the owner's reduced scope. No additional product changes followed. The task-owned temporary Desktop/replay/extraction helpers were removed after verification; raw inputs, captures, scores, selected bodies, service responses, PNGs and receipts remain. Maintained benchmark entrypoints remain in the repository. No owned renderer process remains from the completed runs, and no remote publication or installed-setting change was performed.
