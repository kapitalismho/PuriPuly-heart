# Exact-b399 legacy status timeout finding

```json
{
  "RESULT": "repair_failed",
  "report_type": "Complete RV3-status-follow-on addendum; no rerun performed for this request",
  "candidate": "b3995754cd7f63e1ca7bf446f0eb46c00bdd1366",
  "finding": {
    "id": "RV3-status-follow-on",
    "severity": "P2",
    "classification": "Reachable software-consumer presentation defect under UI source-delivery failure. Separate from the correctly hidden raw status values in native07/18.",
    "title": "A legacy translation arriving after its source UI times out retains the previous managed turn's visible status",
    "contract": "SELF-STREAM-2 R8 requires current-turn presentation/status to remain accurate across bounded delivery and retained manual paths. A completed current manual translation must not display an unrelated older managed turn's translating state.",
    "locations": [
      "src/puripuly_heart/ui/components/display_card.py:254-278 — legacy source setter hides managed status",
      "src/puripuly_heart/ui/components/display_card.py:280-296 — legacy translation setter updates text but does not hide/reset managed status",
      "src/puripuly_heart/ui/components/display_card.py:298-324 — managed snapshot setter stores status and makes the status control visible",
      "src/puripuly_heart/core/orchestrator/translation_output_projection.py — existing bounded SELF UI writer timeout and consumption-time legacy completion authority"
    ],
    "cause": "Normal manual source consumption invokes set_display(), which hides the old managed label. That source event is not guaranteed to reach the consumer: the existing bounded writer can reject it with destination_write_timeout. Translation computation/admission appropriately continues independently. Its later valid current TRANSLATION_DONE calls set_display_translation(), which has no corresponding status reset, so the previously visible managed label remains visible alongside the new manual translation."
  },
  "execution_boundary": {
    "pinning": "A fresh .venv/Scripts/python.exe -B child used an in-memory git archive import loader for every imported repository Python module. PIN printed b3995754cd7f63e1ca7bf446f0eb46c00bdd1366; missing repository modules were rejected instead of falling back to worktree source.",
    "composition": "Tracked tests.ui.test_self_turn_presentation.presentation fixture and tracked OwnerPresentation; real SelfTranslationChannelOwner, TranslationTurnLifecycleOwner, request/output/projection owners, TranslationUiMessageQueue, FletUiPresentationAdapter, UIEventBridge, DashboardView and LogsView.",
    "controlled_boundaries": "Synthetic source text, controlled local provider, in-memory logging/NullHandler, RecordingChatbox and local UI queue. No real audio, native GUI, network, remote OSC or private state.",
    "timeout": "The existing TranslationUiMessageQueue.write_timeout_s was set to0.02 seconds for the isolated probe, rather than waiting its default5 seconds. No timeout behavior, admission, retry or production source was changed. A UI blockage lasting the configured production timeout follows the same branch."
  },
  "reproduction_steps": [
    "Create OwnerPresentation with the normal four-slot harness UI destination queue.",
    "Hold the controlled provider for ('older speech','en'); submit/close that speech source. Wait until the actual dashboard status control is visible and equals t('display.turn.translating'). Its displayed source is 'older speech'.",
    "Clear OwnerPresentation.delivery_gate, put one valid OSC_SENT history-only event so the real bridge has dequeued it and is waiting on the gate, then fill all four queue slots with additional valid OSC_SENT history-only events. These backlog events do not alter dashboard content.",
    "Set the UI message owner's existing write_timeout_s to0.02. Hold the provider for ('manual after dropped source','en'), then call the real Self owner submit_text(..., source='You').",
    "Wait for the actual UI message owner to become idle. Inspect its public OutputRuntime.routing_decisions: the manual TRANSCRIPT_FINAL received destination_write_timeout and accepted_handoff=false.",
    "Release the bridge delivery gate and join the real UI queue. Assert the dashboard still displays 'older speech': the new manual source was not consumed, and history-only backlog did not clear the prior managed status.",
    "Release the manual provider. Wait for that admitted parent to finish and for the actual bridge/dashboard to display 'en:manual after dropped source'. Join the queue and observe its status control."
  ],
  "observed_result": {
    "child_exit_code": 0,
    "dashboard_text": "en:manual after dropped source",
    "status_control_visible": true,
    "status_control_value": "The previous managed t('display.turn.translating') label: Source complete · translating",
    "retained_card_source_value": "older speech",
    "manual_source_delivery": {
      "reason": "destination_write_timeout",
      "event_type": "TRANSCRIPT_FINAL",
      "accepted_handoff": false,
      "ui_queue_submitted": false,
      "outcome": "rejected",
      "physical_ack": false
    },
    "interpretation": "The new current manual translated text is applied, while the older managed translating label is still configured visible. This is an actual production-view consumer-state observation, not just a hidden stale Text.value and not a claim of newly observed native pixels."
  },
  "contrast_with_native07_18": {
    "classification": "No visible product defect established by those original stages.",
    "basis": "Their normal manual source delivery ran set_display(), hiding _turn_state_text. Existing07-native-manual-complete.png and18-native-future-manual-complete.png show no old managed status label. The collectors serialized raw .value without .visible.",
    "why_the_new_case_differs": "The genuine source UI timeout removes the source-setter call that normally hides the label. The later current translation setter alone leaves visible=True."
  },
  "defenses_and_setup_limits": {
    "existing_currentness_fence": "The manual completion is legitimately current and should be displayed; rejecting that translation would solve the wrong problem. The old managed provider remains held throughout the observation, so no new old-turn completion re-shows its label.",
    "initial_invalid_setup": "An earlier exploratory setup used SESSION_STATE_CHANGED backlog. Those events legitimately changed the dashboard, so the intended prior-label prerequisite failed before the novel observation. That setup exited1 and was discarded. Replacing the filler with valid OSC_SENT history-only events preserved the prerequisite; the corrected observation exited0. No product edit or desired-result assertion was weakened.",
    "cleanup": "Provider gates were released by OwnerPresentation teardown; harness/output/bridge/logging owners closed and the child exited. No probe file or persistent resource was created."
  },
  "narrow_repair_assessment": {
    "justified": "Reset/hide managed status in the legacy translation setter, not only when legacy source happens to arrive. The proposed small consumer reset in both legacy setters—_turn_status=None, empty status text, visible=False—is consistent with existing legacy blank-status presentation.",
    "not_authorized_by_finding": "No new manual lifecycle/status taxonomy, provider replay, destination retry, scheduler, admission change, or currentness-fence weakening is required."
  },
  "positive_negative_requirements": [
    "Exercise a real bounded UI source timeout followed by a valid current manual translation through the actual owners/bridge/view. Require manual translated text, no visible inherited managed status, and preserved accepted logging/history.",
    "Retain the ordinary manual-source path: it hides prior managed status before completion; raw/effective-status observation should distinguish hidden state.",
    "Keep managed snapshots' own preview/translating/complete/failed/cancelled/expired labels and partial-target behavior unchanged.",
    "After legacy presentation, a newer managed turn must show its own valid status and complete normally.",
    "Preserve old-turn rejection/current-parent guards, source/history delivery, manual chatbox bypass and no provider/output replay. A direct DisplayCard setter unit check may supplement, but must not replace, the timeout composition."
  ],
  "relationship_to_other_review_results": "This finding supplements, not replaces, the preceding b399 full report's RV3-follow-on segmented-source defect. RV1/RV2 and other verified closures remain retained. No new source pin or source changes were reviewed in this addendum.",
  "delivery": "Complete evidence returned inline for Director persistence; no report file or accessible artifact file creation is claimed."
}
```
