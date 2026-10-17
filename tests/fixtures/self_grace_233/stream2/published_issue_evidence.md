## SELF-STREAM-2 implementation and final software evidence

The selected structural implementation is complete locally. **The issue is not fully complete: physical VRChat/HMD observation under C10 remains blocked/unperformed.** Keep this issue open and In progress.

- Implementation baseline: `f1ab9a47d9caaf67bd7666b7d5f5e380da46b185`.
- Final product/test/current-probe candidate: `27e93797dd9ab8b6a2a7969eac73e409ccaffcda`.
- Consolidated evidence/audit checkpoint: `69b67ed8be9370ea4112f1833b45f4e3a198ac7e`; subsequent contents are documentation/evidence, not another product change.
- These commits are **local and unpublished**, not GitHub commit links. No push, PR, merge, deployment or release occurred. This accessible summary is published under the issue's authorized evidence-record scope.

### Implemented behavior

Usable speculative SELF translations now flow through one maintained turn lifecycle to chatbox as well as overlay/application presentation. Later results revise the same logical turn. Source closure is independent of LLM completion and retains compatible running primary/secondary work rather than cancelling and reissuing it.

Chatbox delivery replaces obsolete unsent same-turn revisions, preserves an earlier required speech correction's normal first-page opportunity before a newer speech preview, and keeps manual bypass and normal long-caption supersession. An identical successfully emitted compatible preview is promoted without another final packet or resetting pagination. Queued/failed output is not mistaken for delivery; equal text in different turns remains distinct.

Dashboard and actual conversation rows are revision-aware. Late old source/results retain legitimate history without stealing the current dashboard. Real secondary child IDs correlate with their parent/target row without changing accepted logger keys. Pending/failed rows identify their target; successful current-source targets remain visible and correctly labelled when another target fails. Preview display does not create accepted conversation/context duplicates.

The original 400-ms default and handler-receipt grace/rearm policy are unchanged. Elapsed credit and no-op rearm suppression remain unselected. No new scheduler, provider retry budget, fuzzy reuse or token-streaming policy was introduced. The maintained architecture follows the selected owner boundaries; no architecture drift was identified.

### Executed verification, with distinct candidate attribution

| Observation | Candidate | Result |
| --- | --- | --- |
| Complete non-integration breadth | `b3995754` | 8,464 passed, 9 process-dependent skips, 256 warnings; 1,091.38 s. Retained earlier breadth, **not** a final-pin full-suite claim. |
| All UI/architecture plus affected SELF/turn/request/output/paginator/diagnostics suites | `27e93797` | 2,160 passed, 2 process-dependent skips, 43 existing deprecation warnings; 95.48 s. |
| Static checks | `27e93797` | Maintained source/tests and current GUI-probe Ruff passed; Black left 57 maintained/current-probe Python files unchanged. |
| Windows-native production views and real owners | `27e93797` | 32 owner-driven stages and both production Logs-mode interactions passed; actual importing entry/exit HEAD and all 565 product Python/JSON files matched. |
| Real local CPU ASR, fast controlled LLM | `27e93797` | Six synthetic episodes decoded into six final terminals and three source turns, none omitted/duplicated; six actual preview UDP packets, no extra identical-final packets. |
| Real local CPU ASR, slow controlled LLM, no overlay sink | `27e93797` | Six calls, five ready results and one obsolete-source cancellation; three source turns; two preview plus three final UDP packets. Existing current-source calls 2/4/6 survived source closure without reissue. |
| Actual isolated headless application / CLI | `27e93797` | Eleven supported queries/operations exited 0; effective capture/translation off, invented manual source-only UDP observed separately, ordered stop and owned host exit 0. |

The native-audio denominator is six invented offline-SAPI episodes / three cases / 12.14525 seconds **per run**. Both runs used the installed Parakeet v3 CPU model, two threads, four verified model-file hashes, English→Japanese, concurrency 2, integrated context and unchanged 400-ms grace. Translation strings were controlled scheduling results, not translation-quality evidence. Native ASR produced zero stable updates; deterministic stable-prefix/suffix and final-only coverage is separate.

The final GUI proof includes same-text final promotion, both manual/speech completion orders, queued and pre-admission sources, real child-ID acceptance in both orders, pending/secondary-first/partial-target states, delayed old language segments, and actual bounded source-UI timeout recovery. It is an isolated native production-component/real-owner composition, not full main-GUI bootstrap, settings navigation or arbitrary responsive-layout validation. Final07 does not freshly repeat native OFF/cancel/expiry; their software checks and older native observations retain their own applicability boundaries.

### Criterion index

| Criterion | Status and decisive evidence |
| --- | --- |
| C1 | Supported. Published SELF-STREAM-2 decision and archived prior contract; 424 original historical traces retain 53 schedules / 91 episodes per policy across four policies and two variants. Two isolated representative replays each executed 24 cases. No new full historical replay or elapsed/no-op adoption is claimed. |
| C2 | Supported. Real owned ingress protects unresolved/prebuffer B from delayed A; matching empty/error/cancel/expiry/watchdog and B-ended/C-false-start paths progress. The original Qwen gap case preserves exact A+C+tail, native cancellations, provenance and fresh-source recovery. |
| C3 | Supported. One translation-work lifecycle owns draft→sealed→final and preserves compatible ready/running primary/secondary operations, context and generation/configuration fencing. Relevant replacement/OFF/shutdown and no-duplicate-call checks passed. |
| C4 | Supported within controlled comparison. Fixed alpha/bravo source schedules differing only in LLM timing retain the same two closures, two calls, ordered context and packets, without cancellation. Resource/retiring-work exceptions are separately exercised. Fresh native ASR timing is not presented as an LLM-only causal comparison. |
| C5 | Supported. Actual output/paginator checks and received local UDP cover preview-before-close, no-overlay operation, queued correction ordering, successful-identical-final promotion, failed/queued delivery truth and distinct repeated turns. |
| C6 | Supported. Dual targets, one speculation/two canonical running/eight waiting budgets, original waiting deadlines, actual prepared context, exactly-once canonical context, manual bypass and cancellation with outstanding physical cleanup are retained. |
| C7 | Supported within the recorded GUI boundary. Final native views/owners/logging, actual Logs interactions and all-UI checks demonstrate revision-aware presentation, legitimate accepted history and current-slot fencing. |
| C8 | Supported. Preserved overlay/Peer/expiry/source-only/manual/independent-input and pagination behavior remains covered by the checkpoint chain and integrated suites. Destination failures do not replay providers. |
| C9 | Supported. Final `_04` real CPU-ASR production composition, controlled local boundaries, deterministic stable/final ingress and actual CLI/effective-state observations are retained at exact source fingerprints. |
| C10 | **Local packet portion supported; physical VRChat/HMD portion blocked/unperformed.** No isolated authorized physical session was established. No messages were sent to default OSC ports or a live social session. |
| C11 | Supported within recorded software boundaries. Maintained architecture/user guidance, complete cutover, integrated checks and independent checkpoint plus terminal audits are present. The fresh terminal audit established no remaining product defect; it did not accept whole-issue completion because physical C10 and then-pending C12 remained. |
| C12 | This record provides the shared final software evidence index, explicit remaining blocker and adoption/revert recommendation. It does not mark the issue fully complete or authorize code publication. |

### Independent review and retained failures

Independent whole/resource checkpoints reviewed the pinned baseline implementation and bounded repair ranges. The final whole continuation returned `repair_verified`, directly executing 28 asserted cases from exact Git-archive imports; this is not a pytest count. The resource continuation confirmed prior resource/source/paginator evidence remains applicable without relabelling its earlier executions. The whole reviewer independently reconstructed the final native product/probe fingerprints.

A fresh independent terminal audit of `69b67ed8` inspected the requirements, underlying records, meaningful assertions, selected native pixels and exact Git-to-GUI/audio/CLI manifests. It supported C1–C9 and C11 within their stated boundaries and found no remaining material product defect. Its initial verdict was **`repair_required`**, distinguishing reachable C12 publication from blocked physical C10, plus a low-severity wording clarification about historical logging isolation. This record supplies the requested public evidence and carries that historical limit explicitly. A bounded publication/readback continuation will retain C10 rather than converting it into a pass.

Initial integrated failures remain recorded: 6,337 passed / 9 failed, 2,060 passed / 21 failed, and later 8,455 passed / 2 fixture-protocol failures. The seven defects in the previous checkpoint comment and subsequent preview-latency/current-slot/target-rendering follow-ons were repaired rather than hidden by weaker outcomes. The final broad/scoped passes above do not erase those original results. Final GUI output retains one plain `[Failure] structured_failure` marker during controlled failure/timeout scenarios; its formatter does not identify the originating controlled failure. No traceback or additional failed stage is reported.

Local retained index: `tests/fixtures/self_grace_233/stream2/COMPLETION.md`; full change inventory: `stream2/final_change_inventory.json`. Supporting records include root `REPORT.md`, `audio/RESULTS.md`, `audio/RUNTIME_RESULTS.md`, `audio/final_candidate_summary_04.json`, `gui/REPORT.md`, `gui/evidence_owner_repairs_07/`, `stream2/integration_verification.md`, and the pinned independent review reports. These are local artifact locations, not currently accessible GitHub blob links. This comment supplies the public outcomes and limits until code publication is separately authorized.

### Remaining limits, owners and reversible adoption

- **Maintainer-owned C10 prerequisite:** an explicitly authorized isolated VRChat/chatbox session and corresponding headset observation. Local send, local UDP reception, app pixels and physical visibility are distinct. The issue remains open/In progress.
- Final probes used invented audio/text and isolated settings, secrets, logging, discovery and scene roots; all owned hosts/receivers/tasks closed and ephemeral settings/secrets were removed. Python loopback guards are not an OS sandbox. The older historical preliminary runner had a possible synthetic-only default-log append; that inferred limit is retained, and user logs were neither inspected nor removed.
- CLI admission `output_delivery=unobserved` and stop `cleanup_outcome=unobserved` remain unchanged; separate packet/process observations do not rewrite those receipts. No live-cloud integration, microphone/private recording, actual translation-quality, build/type-check, fresh persistence/restart or population-performance claim is made.
- Previews can change. Not every intermediate revision/page is guaranteed to appear; already-read remote text cannot be retracted. Final-only recognition still requires its first recognition result. No universal latency saving is promised.
- Recommend an explicitly authorized isolated adoption/physical probe, **not an unapproved release**. No persisted settings-schema migration is introduced. Review a final-tree publication/squash plan before an authorized push, because local evidence history retains earlier pre-export representations.
- If reversal is needed, use the synchronized pre-feature build or revert the complete task-owned cutover together after authorization. Do not selectively restore the old RUNNING/source-close coupling inside the new lifecycle. A full rollback loses progressive chatbox behavior and restores the known predecessor-obligation defect. Preserve the evidence record. Push/merge/release and physical-session decisions remain with the maintainer.
