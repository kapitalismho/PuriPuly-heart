# Exact-b399 partial-success row-status adjudication

```json
{
  "RESULT": "repair_failed",
  "report_type": "Additional adjudication within the existing managed target-row presentation finding; no rerun",
  "candidate_applicability": "Exact retained GUI06/b399 product evidence already inspected.",
  "judgment": "Yes. In a target-specific English row, the unqualified header 'Mic → English · Translation failed' falsely attributes aggregate failure to the valid ready English target. The dashboard's aggregate failure is correct; repeating that aggregate as the English row's apparent outcome loses the preserved independent-target success semantics.",
  "decisive_basis": {
    "record": "The primary record is disposition=ready, has nonempty valid English translation, translation_source_revision=source_revision=1 and its real accepted primary logger identity. The Japanese target failed separately.",
    "visible_evidence": "The retained14-native-partial-target-failure.png and rendered conversation show 'Mic → English · Translation failed'. This is visible row text, unlike the hidden manual dashboard scalar.",
    "source": "ConversationViewModel.render():247-253 derives every managed row header from record.turn_status, which is the aggregate status, without considering the row's target disposition/current-source coverage."
  },
  "smallest_semantic_correction": {
    "scope": "Rendering only: for this terminal aggregate-failed case, a ready target with nonempty current-source translation should retain a successful target header, using the existing Complete label rather than the aggregate Translation failed label. Keep the stored aggregate turn_status and all accepted logger/model identities unchanged.",
    "positive_condition": "Aggregate status failed AND target disposition ready AND genuine translated text AND translation_source_revision equals source_revision. This is a target-specific successful terminal result despite a different target's failure.",
    "preserve": [
      "Failed target rows remain failed and must identify their actual target language, as previously adjudicated.",
      "Whole-turn cancelled and expired states remain cancelled/expired even if an earlier ready translation is retained; never override them with Complete.",
      "A translation for an earlier source revision must not become current successful completion; preserve stale-source warning and invalidation semantics.",
      "Active/pending/preview and fallback-in-progress phases retain their existing presentation; readiness alone must not promote an active turn to Complete.",
      "True source-only and legacy rows retain their existing semantics. Do not infer accepted logging or create an accepted record from presentation readiness."
    ],
    "not_required": "No new status taxonomy, lifecycle fact, provider retry, aggregate-state mutation or DTO is needed. An alternative explicitly scoped whole-turn summary could avoid ambiguity, but broad UI restructuring is unnecessary for this repair."
  },
  "minimum_regression_protection": [
    "Actual partial-failure composition: English ready/current row presents successful completion, Japanese row presents Japanese failure, dashboard remains aggregate failed, and accepted log keys stay unchanged.",
    "Negative cases: cancelled/expired retained-ready rows do not show Complete; stale-revision ready data does not show current successful completion.",
    "Active source-open and pending secondary/fallback phases do not acquire premature Complete labels.",
    "Retain prior source-only/legacy header tests and real before/after accepted-log correlation protections."
  ],
  "relationship": "This is a second aspect of the same target-row semantic rendering finding, directly related to F4's preservation of a still-valid successful target. It does not invalidate the already verified dashboard primary-retention repair.",
  "execution": "No checks or user-reported observation were rerun, and no files were changed."
}
```
