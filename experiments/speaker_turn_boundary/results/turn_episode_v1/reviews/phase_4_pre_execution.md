# Phase 4 pre-execution review

Status: **approved**.

## Review identity

- Phase: 4, raw signal diagnostics.
- Normative plan: `.agents/specs/prd/bounded_turn_episode_speaker_change_fusion_experiment_review_gated.md`.
- Plan Git blob: `cbcf1455651d144df808027183ec8e360752b432`.
- Plan SHA-256: `ecd16c765072504f9970cb17fd5cba4f9967715954adead6ebc386fffb8d4f8c`.
- Accepted entry candidate: `aad402be25b02a54aba7ae1ce3f4066eeaf460a0`.
- Review bundle: `reviews/phase_4_review_bundle.md`, revision 5.
- Candidate under review: `working-tree` based on `aad402be25b02a54aba7ae1ce3f4066eeaf460a0`.
- Final reviewed bundle byte SHA-256:
  `a6afa3dc946815c162ee18d09b1c7ad3ad08e252f7286c110b37a685fe2b1759`.
- Design ledger byte SHA-256:
  `0a86788a4817d4a205d92b0afb6ee05dc97d11da3e99d4c0501d74be30473691`.
- Design ledger content SHA-256:
  `c8336c2665b28047b1a169fc9605a6c6a3c400afe553dc3a2d35ca9b20b41536`.
- Reviewer: fresh read-only Implementation Reviewer
  `/root/phase4_pre_execution_review`.
- Final review date: 2026-08-09.

## Execution barrier

No Phase 4 large neural inference, full diagnostic sweep, or scientific result existed
before this approval. Confirmatory held-out access, Phase 5 policy replay, provider
activity, production wiring, merge, push, and deployment remain outside this approval.

## Review history

- Revision 1 returned `repair_required` for an invalid candidate pin,
  under-specified ERes coordinates and state, absent pre-signal coordinate/pair and
  runtime ledgers, incomplete acoustic matching, ambiguous low-block precedence, and
  incomplete LS terminal/parity/export provenance.
- Revision 2 resolved every revision-1 finding and returned `repair_required` for a
  center-bound LS acoustic comparator with zero valid primary 500 ms pairs and terminal
  wording that incorrectly forbade committed frontend analysis padding and decode-only
  flush behavior.
- Revision 3 freezes candidate-aligned 250/500/1000 ms acoustic supports with
  295/313/247 valid matched pairs, a fail-closed zero-count guard, and terminal semantics
  that distinguish forbidden appended source audio from permitted STFT/context padding
  and `ingest=0, decode=1` flushing. Same-reviewer verification returned `accepted`.
- A pre-full-run audit then proved revision 3 omitted 744,442 ERes public source-prefix
  windows and falsely declared unexecuted state classes. Revision 4 expands the frozen
  universe to 1,217,509 coordinate declarations, 895,656 unique windows, and 2,848
  snapshot declarations; it adds bounded binary-v2 caches, executed LS/ERes state
  traces, source-prefix fallback, and a reviewed 13.222169-hour parallel forecast.
- Revision 4 returned `repair_required` because its verifier only shallow-checked state
  receipts, the ERes cache-hit runner unpacked the loader incorrectly, and exact cosine
  0.50 had inconsistent state/proposal semantics. Revision 5 independently reconstructs
  all state receipts and rejects a coherently rehashed mutation, fixes and exercises
  LS/ERes first-write/all-hit/mixed-hit retries, and freezes change score `>0.50`.

## Final verdict

**accepted** with no material findings. The reviewer independently recomputed the
expanded ledger/window/snapshot identities and confirmed that revision 5 closes every
revision-4 finding. Black/Ruff, 25 focused Phase 4 fixtures, all 340 experiment tests,
and two consecutive complete smoke runs pass. The 13.222169-hour,
4,006,266,688-byte-cache, 8-GiB-RSS forecast remains within the frozen 16-hour,
8-GiB-cache, and 16-GiB-RSS ceilings. The candidate changes only experiment-tree files.
Acceptance authorizes one full Phase 4 execution under bundle revision 5; it does not
accept Phase 4 scientific results or authorize Phase 5, merge, push, or deployment.
