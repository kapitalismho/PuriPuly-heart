# Exact-b399 managed target-header finding

```json
{
  "RESULT": "repair_failed",
  "report_type": "Bounded R8/F4 target-presentation adjudication addendum; no scenario rerun",
  "candidate_applicability": "b3995754cd7f63e1ca7bf446f0eb46c00bdd1366. The retained GUI06 product manifest matches b399's565 product files; this Logs renderer is unchanged by the inspected continuation delta.",
  "finding": {
    "id": "RV3-target-header-follow-on",
    "severity": "P2",
    "title": "Pending/failed managed target rows omit their target identity when no translation text exists",
    "classification": "Supported user-visible semantic presentation defect, not merely preferred wording.",
    "contract_judgment": "Yes: when the actual conversation view presents separate managed parent/target rows and attaches pending/failed status to each, the user must be able to identify the configured target that owns that status. Internal parent/target correlation alone does not make the presented failure unambiguous. This follows the R6/R8 target-specific provenance/status and identity-aware application presentation contract; it does not require a particular decorative header format.",
    "location": "src/puripuly_heart/ui/views/logs.py:239-253",
    "cause": "Header selection uses record.translated_text truthiness as the prerequisite for showing target_language. A correctly identified managed target row with translated_text=None falls back to source_language, even when target_index/target_language and failed/pending disposition are authoritative.",
    "decisive_existing_evidence": {
      "record": "GUI06 partial-target-failure has configured targets en/ja; the second managed record has target_index=1, target_language='ja', disposition='failed', translated_text=null, source_language='fr'.",
      "rendered_conversation": "Its header is 'Mic · French · Translation failed', while the ready primary row is 'Mic → English · Translation failed'. The Japanese target is absent from its own failed row.",
      "pixels": "Inspected the retained14-native-partial-target-failure.png: the rendered conversation visibly shows that French failure header, not Japanese. Unlike the prior hidden dashboard scalar, this text is actually visible.",
      "source_defense_checked": "The correct target identity is already retained by update_self_turn in the ConversationRecord. No missing provider fact, new metadata or new lifecycle authority is needed; render() discards its usefulness when translation text is absent."
    },
    "impact": "Users cannot reliably tell which configured target is pending or failed from the target-specific row. Pending dual rows can both be labeled only with the same source language; a failed secondary appears only as a source-language failure rather than an identified Japanese-target outcome."
  },
  "smallest_semantic_correction": {
    "scope": "Use existing managed target identity to label target-work rows even before/without successful translated text. Keep the current successful target labeling and per-target statuses.",
    "preserve": "Do not blanket-label every record with target_language. Genuine source-only rows and legacy source/accepted-record headers retain their existing source-language semantics. Distinguish managed target-work rows using their existing turn/target identity and disposition.",
    "not_needed": "No DTO, logger identity, accepted-record key, row correlation, provider, scheduler, admission, retry or source-lifecycle change."
  },
  "required_positive_negative_protection": [
    "Managed dual pending rows identify English and Japanese separately, with no speculative accepted-log entries.",
    "Secondary-first rendering still identifies both the ready secondary and pending primary in configured row order.",
    "Partial failure identifies the failed Japanese row without translated text while preserving the valid English row and accurate failure status.",
    "Real accepted-before/after-snapshot correlation still yields one row per parent/target and unchanged legitimate accepted logger identities.",
    "True managed source-only and legacy source/translation records retain their existing headers; do not turn source-only status into an implied failed target attempt."
  ],
  "execution_note": "No user-reported case was rerun to confirm it. This adjudication uses the supplied native observation, retained rows/rendered text, inspected pixels and the decisive renderer condition.",
  "handoff": "Add this bounded renderer finding to the pending repair wave if adjudicated. Previously verified closures and the two other accepted RV3 follow-ons remain unchanged. No edits made; no whole-issue acceptance or physical C10 claim."
}
```
