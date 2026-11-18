# Scribe v3 integration and matched follow-up

## Authority and status

The owner requested implementing the discussed Scribe v3 integration, repeating the same experiment against the retained results, investigating observed problems, and explicitly checking readability when speaker information arrives late, including expired captions reappearing. This authorizes local product implementation and bounded public-audio/service verification. It supersedes the earlier analysis-only boundary for this follow-up, not the historical findings in [the original report](issue441-results.md).

Implementation and verification are pending. This record does not claim a working integration, successful comparison, release, deployment, PR publication, or issue closure.

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

Pending: implementation, focused deterministic behavior checks, the declared same-input comparison, connected actual Desktop observations, independent checkpoint review and terminal completion audit. Later sections will record executed evidence, findings, repairs, user-visible changes and protected-document drift. No completion claim is made while these obligations remain pending.
