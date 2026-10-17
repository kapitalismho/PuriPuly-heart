# SELF grace #233 — retained historical comparison evidence

Current authority: [#233 SELF-STREAM-2](https://github.com/kapitalismho/PuriPuly-heart/issues/233). The original SELF-GRACE-1 authority is [archived in the issue](https://github.com/kapitalismho/PuriPuly-heart/issues/233#issuecomment-6236992003). This task-local record preserves the old observations; it is not current-product acceptance, maintained architecture guidance, or an integration approval.

## SELF-STREAM-2 C1 — historical isolation receipt (2026-10-18)

### Current decision and scope

- **Elapsed-time credit and no-op rearm suppression are unselected.** SELF-STREAM-2 preserves the existing configured 400 ms handler-receipt grace and rearm policy.
- **The original G2 structural deferral below is superseded.** The selected current design is the single translation-work lifecycle and independent source closure in SELF-STREAM-2 R3–R8. The old no-change endpoint and prior integration-check request are also superseded.
- The retained `results/` and `correctness_results/` observations are historical SELF-GRACE-1 evidence, not the current SELF-STREAM-2 implementation. Current-owner acceptance belongs to the maintained `tests/core/test_self_stream_233.py` suite and the separately owned `stream2/` proof.
- This isolation changed only historical fixture/evidence surfaces. It did not add compatibility DTOs, change product source, touch maintained tests/helpers, or change `audio/`, `gui/`, or `stream2/`.

### Exact source and harness reconstruction

The old runner pinned only the SELF owner while importing the rest of the worktree. That cannot preserve a historical comparison after the current product removes its private speculative-attempt DTOs. The runner now uses read-only `git archive` to materialize **all `src/`, `tests/`, `prompts/`, and `pyproject.toml` at `75c4485a6787d504f5335d1602973f63b4d0c234`** into an automatically removed temporary directory. It loads the existing opt-in experiment there and reuses the pinned low-latency `compose_translation_test_harness`; there is no second translation/recognition pipeline.

The child interpreter starts with `-B -I`, prepends only the archived product/harness roots, disables unrelated pytest plugin autoload, and explicitly loads pytest-asyncio. Every loaded `puripuly_heart`/`tests` module file must belong to that temporary tree; relative import origins are retained in each fresh summary. Direct imports of the historical experiment outside this runner are rejected before the old private DTO import. Only controlled scoped recognition, scheduled LLM responses, synthetic PCM levels, and recording sinks run. This harness does not start application settings/secrets/discovery/provider composition, real audio/ASR, GUI, or network destinations. No dependency installation, model download, cloud/provider access, or user-log inspection was performed.

| Source | Exact pin |
| --- | --- |
| Baseline owner SHA-256 | `e29fb972bd8174d7f8fbe8d91a42b4da519f2912e5b58e6c97183ae90430a007` |
| Historical correctness owner SHA-256 | `ec77ef5e91ddde932de59ed8a036e101da72373042b000b516ce831159e58e2f` |
| Baseline source/harness archive SHA-256 | `53079e8fa23b4e315b187ad63ffb02185321859d0873067e093c37cb1bf1e406` |
| Original experiment SHA-256 before isolation edits | `ccdc8b7cf00982a118339ec6edd29ec2172b0add031769062bd224921f858ebb` |

The historical correctness source **was reconstructable**: the local historical Git object `ed92da54c32bcac4add53be4f53a119308fea361` contains the exact owner hash recorded in the original receipt. Read-only comparison with the baseline showed only that owner and the historical characterization additions in the low-latency test changed among `src/`, `tests/`, and `pyproject.toml`. The exact owner bytes are now retained at [`correctness_results/source/self_translation_channel.py`](correctness_results/source/self_translation_channel.py). Replay replaces only that owner in the otherwise pinned baseline tree, under the explicit `historical-correctness` label. The stored source removes any future dependence on the lifetime of that local Git object; it is not a published implementation commit or the current worktree.

Historical characterization assertions were recovered from the same archived test and moved into the opt-in experiment. Their expected timings, counts, continuation failures, source parts, and provenance protections were retained; only helper names and the candidate label changed. The original assertion-function SHA-256 was `d739598a1e1299abb196f6521b1f8597b333940c71b87cdf1e25b410b53b5781`. Pinned module, experiment, runner, and collection-adapter hashes are recorded in the new [baseline source manifest](results/isolation_replay_20261018/source_manifest.json) and [candidate source manifest](correctness_results/isolation_replay_20261018/source_manifest.json).

### Runnable commands and observed representative proof

Run from the checkout root with the existing worktree virtual environment. Git must already contain the baseline revision, and the installed Python packages must be available; the runner does not fetch or install them. `--report-dir` **must be new**. The historical `--owner-source worktree` selector is deliberately rejected, not repointed to the current product.

The two commands below were executed using the worktree `.venv/Scripts/python.exe`; the interpreter path is portable notation. To repeat them, choose different fresh output-directory names:

```text
<checkout>/.venv/Scripts/python.exe -B tests/fixtures/self_grace_233/run.py --report-dir tests/fixtures/self_grace_233/results/isolation_replay_20261018 --case "final_after_end or stable_before_end_terminal_early or B_before_buffer_final_only or watchdog_prefix_suffix_terminal or G2_fixed_three"
<checkout>/.venv/Scripts/python.exe -B tests/fixtures/self_grace_233/run.py --owner-source historical-correctness --report-dir tests/fixtures/self_grace_233/correctness_results/isolation_replay_20261018 --case "final_after_end or stable_before_end_terminal_early or B_before_buffer_final_only or watchdog_prefix_suffix_terminal or G2_fixed_three"
```

Check rationale: a passing test that still imports today's harness could silently change the historical result. A fresh process, pinned whole-source archive, checked owner fingerprint, recorded import origins, and comparison with the original traces close that gap. The selected schedules cover legitimate final-only/stable progress, the prebuffer continuation failure and its correction, consumed-prefix/suffix publication, and fast/slow translation-dependent historical grouping.

| Evidence set | Collection/execution | Source-set denominator | Continuation disposition |
| --- | --- | --- | --- |
| Original baseline | **passed**: 212 characterizations, exit 0 | 53 schedules / 91 acoustic episodes per policy | **failed** on eight original schedules per policy; these are expected historical defects, not product passes |
| Original historical correctness candidate | **passed**: 212 characterizations, exit 0 | Same 53 / 91 per policy | Zero premature continuation schedules in original reports |
| Isolated baseline representative replay | **passed**: 24 collected, 24 executed/passed, exit 0 | Six schedules / 11 episodes per policy, four policies | `B_before_buffer_final_only` remains **failed** under all four policies |
| Isolated historical correctness representative replay | **passed**: 24 collected, 24 executed/passed, exit 0 | Same six / 11 per policy | All 24 reports have **passed** continuation status |
| Full 53-schedule comparison through the new isolated runner | **not run** | Full original 53 / 91 remains available | No new full-run claim |
| Current SELF-STREAM-2 acceptance or physical output | **not run by this worker** | Not the historical comparison | Owned by current implementation/runtime/UI workers |

Observed stdout for **each** isolated run included `pytest_exit_code: 0`, `collected_case_policy_samples: 24`, `test_outcomes: {"passed": 24}`, and `case_policy_samples: 24`. Full outputs, profiles, and relative import origins are retained in [baseline output](results/isolation_replay_20261018/pytest_output.txt), [baseline summary](results/isolation_replay_20261018/summary.json), [candidate output](correctness_results/isolation_replay_20261018/pytest_output.txt), and [candidate summary](correctness_results/isolation_replay_20261018/summary.json).

Both fresh environments were Python 3.14.7 / Windows 11 10.0.22631; NumPy 2.5.1, pytest 9.1.1, pytest-asyncio 1.4.0, onnxruntime 1.28.0. **sherpa-onnx is now 1.13.8, not the originally recorded 1.13.4**; this controlled-recognition replay does not perform ASR and is not a fresh native-ASR replication. Other profile settings and recorded numerical evidence were not changed.

The 24 fresh reports in each group match the corresponding original reports across owner fingerprint, frozen profile/schedule, source counts, commits/parts, premature commits, qualified waits, metrics, invocations, physical retirement, finals, and full timelines after **consistent aliases for random UUIDs and contribution UUID prefixes only**. Identity equality and contribution sequence relationships are preserved by that comparison. The candidate's original `worktree` label is intentionally not compared as a current source identity. See [baseline comparison receipt](results/isolation_replay_20261018/comparison.json) and [candidate comparison receipt](correctness_results/isolation_replay_20261018/comparison.json). This representative agreement is not an assertion that the unrerun 188 reports in each group were exercised now.

For a fresh **full** historical comparison, omit `--case`:

```text
<checkout>/.venv/Scripts/python.exe -B tests/fixtures/self_grace_233/run.py --report-dir <new-baseline-report-dir>
<checkout>/.venv/Scripts/python.exe -B tests/fixtures/self_grace_233/run.py --owner-source historical-correctness --report-dir <new-correctness-report-dir>
```

Both rejection boundaries were also exercised: an existing `results/` report directory exited **2** with `--report-dir must not exist`; `--owner-source worktree` exited **2** as an invalid choice. Neither ran pytest or replaced evidence. Both successful replay [execution receipts](results/isolation_replay_20261018/execution.json) / [candidate receipt](correctness_results/isolation_replay_20261018/execution.json) report temporary source-tree removal and no original evidence overwrite.

### Preservation and limits

All **424 original per-case JSON traces** remain unchanged. Original summary/analysis interpreter paths were explicitly sanitized to `<checkout>/.venv/Scripts/python.exe`, and summary provenance annotations were added; no original recorded numerical evidence was rewritten. The original candidate's raw `owner_source: worktree` value is retained as a historical label, with the exact preserved owner/hash identified here and in its summary. Original per-case set fingerprints and pre-sanitization summary fingerprints are retained in [analysis.json](analysis.json).

The current isolation proof is representative, not a claim of complete historical-environment/package replication or current integrated acceptance. Prior failed fixture-authoring attempts and unperformed/unknown boundaries remain in the archived receipt below. No authoritative original evidence was deleted. If storage consolidation is wanted, the Director should decide whether to archive the original traces once with a hash index and retain only representative fresh replay receipts; this worker did not remove or replace them.

## Original SELF-GRACE-1 receipt (historical, superseded decisions)

The following sections describe the earlier experiment and correctness candidate only. Their references to “worktree,” “now,” blocked elapsed selection, structural deferral, and final checks belong to that historical receipt, not the current checkout or active SELF-STREAM-2 decision.

## Decision and authority

- **Correctness: implemented in the worktree under the Director's explicit authorization.** Delayed predecessor results no longer settle unrelated active/unresolved acoustic successors; obligations also exist before a merge buffer does. This is not a translation-speed claim.
- **Elapsed-time credit: blocked pending the user/maintainer's segmentation decision. Not implemented in production.** A demonstrated isolated qualified wait exists, and real final-only local ASR ingress is available, but earlier deadlines also change grouping and translation context on reachable fixed schedules.
- **No-op rearm suppression: not selected by the Director.** Boundary-period replay gains do not independently establish semantic acceptability. True native duplicates/same snapshots already disappear before source reconciliation.
- **Structural handoff: deferred to the maintainer.** Grouping depends on LLM timing even after the correctness repair and accounting for elapsed credit. Adoption would change source/context/presentation/ownership contracts. No source/task transfer or RUNNING-guard deletion was implemented.
- Production grace remains configured **400 ms from handler receipt**. Provider/model, Peer policy, VAD endpoint policy, retries/hedges, output pacing, and the deliberate one-second presentation policy were not changed by this worker.

The Director owns maintained-doc/changelog updates, final integrated validation, the durable adoption receipt, and publication of accessible versioned artifact links. This work did not mutate Git/GitHub, amend other issues, merge, deploy, or make paid/provider-upload calls.

## Pins, environment, and reproduction

Execution baseline: `75c4485a6787d504f5335d1602973f63b4d0c234` (`git rev-parse HEAD` observed). The user supplied the clean-branch/remote-dev relationship; it was not reinterpreted as an executed remote check.

SELF owner baseline SHA-256: `e29fb972bd8174d7f8fbe8d91a42b4da519f2912e5b58e6c97183ae90430a007`.

Correctness-candidate SELF owner SHA-256 exercised here: `ec77ef5e91ddde932de59ed8a036e101da72373042b000b516ce831159e58e2f`. The candidate was uncommitted; the base commit is not a substitute for this candidate source pin.

Interpreter used: the parent project's existing `.venv/Scripts/python.exe` (public commands below use portable `<checkout>/.venv/Scripts/python.exe` notation), Python 3.14.7, Windows 11 / 10.0.22631. Observed package versions: NumPy 2.5.1, pytest 9.1.1, pytest-asyncio 1.4.0, sherpa-onnx 1.13.4, onnxruntime 1.28.0. The historical runner inserted that worktree's `src` before the parent environment's editable installation and set `PYTHONPATH` accordingly. Bare system Python was not used for the experiments.

Commands actually executed for the final complete comparison sets:

```text
<checkout>/.venv/Scripts/python.exe -B tests/fixtures/self_grace_233/run.py --report-dir tests/fixtures/self_grace_233/results
<checkout>/.venv/Scripts/python.exe -B tests/fixtures/self_grace_233/run.py --owner-source worktree --report-dir tests/fixtures/self_grace_233/correctness_results
```

Both exited **0**, each collecting **53 schedules × 4 opt-in policies = 212 passing characterization assertions**. Passing the baseline characterization tests does **not** mean the baseline continuation invariant passes: eight schedules deliberately assert the reproduced failure, and their reports say `failed`. Candidate product assertions require no such premature commits.

Earlier fixture-authoring runs exited 1 for three fixture errors (duplicate `event` trace argument, incorrect positional `AudioCaptureSpan` construction, and missing accepted provider configuration scope). Those were corrected before the final measurements. They were not classified as product-policy failures. Intermediate 140-, 188-, and 200-sample runs were superseded by the complete 212-sample receipts above.

Links within the versioned task artifact:

- [Runnable fixtures and frozen schedules](experiment.py)
- [Bounded runner](run.py)
- [Measured aggregate record](analysis.json)
- [Pinned baseline summary and individual traces](results/summary.json)
- [Correctness candidate summary and individual traces](correctness_results/summary.json)
- [Independent real synthetic-audio production evidence](audio/RESULTS.md)
- [Real baseline ASR/source timeline](audio/baseline_translation_asr_schedule.json)

## Composition, profiles, clocks, and provenance

The deterministic fixture extends `tests/core/test_self_translation_low_latency.py` and its existing `compose_translation_test_harness`. It uses the actual `SelfCaptureSessionOwner` generation-guarded serial dispatcher and source ledger, `OwnedVadEvent`, `LocalASRProviderRuntimeOwner`/provider handle, `ScopedRecognitionEngine`/normalizer, channel callbacks, SELF owner, translation-request/turn owners, output projection, and output runtime. Only the source/provider/LLM/sink boundaries and opt-in grace-policy hooks are controlled. No competing translation or recognition pipeline was constructed.

The baseline owner is loaded from the exact pinned Git blob with read-only `git show`, using the same existing composition. This permits unchanged comparisons after the worktree owner is repaired without a copied legacy production implementation or changing the production default.

Frozen deterministic profile: en → ko; the dual-target case adds ja; concurrency 10; integrated context disabled; ordinary SELF context window 30 s / three entries; finalize grace 400 ms; merge gap 600 ms; spec retry max 1; awaiting/resume watchdog 3 s except explicit shortened watchdog cases. The test capture profile advertises 400 ms hangover while the translation profile retains its independent 1 s value. These injected observations do not measure the VAD detector or retune either profile. Initial context is empty. Provider overlap is an explicit capability parameter; production local overlap and a serial/non-overlap counterexample are kept distinct. All sinks are controllable, bounded test boundaries without physical destination access.

`Step.at` is handler schedule time. Optional `Step.source_at` stamps the source-ledger VAD seal independently. `delayed_VAD_receipt` observes a seal at .100 s and handler receipt at .300 s. The future-source-anchor fixture demonstrates candidate fallback to baseline; it is an invalid-clock mechanism probe, not a real capture observation.

Timer bodies use explicitly registered fake-clock futures. The scheduler advances to each deadline, wakes its corresponding future, and drains the event loop before injecting the next observation. Numeric clock advancement is never treated as an elapsed `asyncio.sleep`. Real provider write/final waiting remains a distinct owner boundary; an injected `expired` terminal is not claimed to be an executed native-resource timeout.

Traces link accepted PCM source ranges and full activation/order/capture-epoch identities to request/contribution offsets, publication UUIDs, and merge parts. Scoped terminal snapshots carry included contribution ranges; they do not append already consumed prefixes. A publication UUID after the early-prefix watchdog is deliberately different from its acoustic episode. Random IDs alone are not acoustic or semantic proof. The deterministic PCM level arrays are not speech; the sibling's separately hashed synthetic WAV set supplies real speech/ASR evidence.

The production `_emit_metric` remains a no-op. Only the opt-in probe's actually executed override collects bounded counters. Timeline events record source/handler endpoints, stable and terminal receipt, grace rules/deadlines, evaluated blockers, LLM invocation/readiness/logical cancel/physical completion, normal source commit, and accepted overlay projections. No accepted projection is called an actual chatbox send, GUI/HMD display, or physical delivery.

### Qualified grace-only waiting

Wait is integrated only over intervals with a READY compatible primary translation, matching current provider/configuration, no awaiting-end/resume blocker, and no actual active or unresolved other acoustic continuation. Applicable lifecycle release (matching false start or executed watchdog) is respected. The baseline deliberately permits stable-prefix publication without a full provider terminal; full terminal readiness is a separate elapsed-candidate eligibility fact. Slow service, valid continuation, stale configuration/provider, and failed speculation are not labeled grace-only delay. No overlapping stage durations are added.

## G0 — reproduced reachable ordering cases

The strongest deterministic final-only cases traverse the production serial capture dispatcher and shared request engine with local-style sealed-turn overlap, not a forced private merge buffer.

- `delayed_A_during_B_pending` / `delayed_A_during_B_confirmed`: P final .150 s opens a buffer; A is captured .200–.300; B begins .400 and remains active until 1.300. A terminal at .500 changes `resume_before=B` to `resume_after=null` in the baseline. P+A commits .900 while B remains active/unresolved. Three B chunks distinguish confirmation from a merely pending onset.
- `B_before_buffer_final_only`: B starts .200 before delayed A terminal creates a buffer at .300. Baseline commits A at .700 while B is active until 1.100. No missing flag proves silence. The repaired owner holds both source parts and commits normally at 1.600 after B's terminal.
- `B_unresolved_C_empty` / `B_unresolved_C_final`: B ends .500 without a terminal; newer C begins .600. A C false-start end at .700 does not settle B. In the baseline P can commit at .700 with A/B unresolved, and late A can commit at 1.300 while B remains unresolved until 1.800. C's terminal emitted .800 is actually held by ordered engine terminal draining until 1.800; the report does not pretend C completed at owner receipt .800. Repair produces one normal source commit at 2.200.
- `prebuffer_B_ended_pending` and `B_watchdog_survives_C_false_start` cover no-buffer end obligations and retaining B's own watchdog through a newer C false start.
- `same_acoustic_suffix_during_B_pending` covers a new publication UUID for A's remaining suffix while B exists. Its stable-plus-overlap provider boundary is explicitly **not demonstrated as a concrete real provider route**; it is a provenance/mechanism regression, not additional provider promotion.
- `serial_provider_B_queued` is the reachability counterexample: the real serial dispatcher cannot deliver B's owner onset while A's non-overlap provider end is blocking; B is delivered afterward and its obligation is preserved.

All four experimental grace policies retained the eight baseline failure schedules. After the production correctness repair, all four report zero premature continuation commits on the same inputs. This is a correctness receipt, not a general no-loss or incident-frequency certification.

The independent [audio receipt](audio/RESULTS.md) observes six fresh `local_parakeet_v3` CPU decodes through production composition, six scoped terminals, six controlled LLM calls, and four baseline source commits over the same six synthetic inputs. It also reproduces A committing while B, which began before A's buffer existed, is still active. This confirms a real final-only merge route, not every local or cloud route. That native baseline was completed and source hashes pinned before any product mutation.

## G1 — isolated gains and material boundary consequences

Experimental elapsed rule: where every contributing acoustic episode has an actual successful final terminal and compatible finite VAD-end anchor, and no active/unresolved continuation remains, use `max(now, latest_VAD_seal + configured_grace)`. Otherwise retain the full handler-receipt grace. The broad deterministic experiment also investigates a modeled stable request becoming fully terminal; **that is not promotion of a real stable route**. A possible production selection would have to be narrower: only the actual verified final-only route, no early-stable/watchdog publication, exact source/terminal provenance, and explicit clock fallback. No route currently receives production elapsed credit.

| Frozen case (one sample/policy) | Unchanged commit | Elapsed commit | No-op commit | Combined commit | Meaning of measurement |
| --- | ---: | ---: | ---: | ---: | --- |
| End .100, terminal .500, LLM ready .550 | .900 | .550 | .900 | .550 | 350 ms qualified grace-only wait removed; one invocation each |
| End .100, terminal .200, LLM ready .250 | .600 | .500 | .600 | .500 | 100 ms qualified wait removed |
| Same source/ASR, LLM ready 1.300 | 1.300 | 1.300 | 1.300 | 1.300 | No service-limited gain; zero qualified grace-only wait |
| Stable .200, terminal adds permitted period .400 | .800 | .500 | .600 | .500 | No-op isolates 200 ms rearm wait, but semantic policy not approved |
| Native duplicate and identical snapshot | .600 | .500 | .600 | .500 | Duplicate/snapshot do not reach reconciliation; one invocation, one `final_phase` |
| Stable .300, terminal only at 1.000 | .700 | .700 | .700 | .700 | Stable is not terminal; candidate retains prefix policy before completion |
| Terminal emitted before end (.100 / .300) | .700 | .700 | .700 | .700 | Engine withholds actual terminal until local seal; LLM starts .300 |

Questions, number additions, and negation additions require a new speculative request; all cases exercise that invalidation. Contradicting an already stable number produces explicit failed recognition and retains usable prior stable text, not fuzzy repair. Different acoustic episodes saying identical text remain two source parts.

### Decision-critical grouping/context comparison

Fixed A: start .000, VAD end .100, terminal .200, fast LLM ready .250. Earlier elapsed deadline .500 versus unchanged deadline .600.

- B onset .490: both policies merge A+B.
- B onset .500 (defined timer-before-input tie) or .550: **unchanged merges A+B; elapsed splits A | B**.
- B onset .600 (timer-before-input tie) or .610: both split.
- With only LLM timing changed to .800 s service, all five continuation schedules merge A+B under both policies: running work retains the open source buffer.

Exact .550 example:

- Unchanged: requests `one` at .200 and `one two` at .750, both with initially empty context; one source unit commits 1.150.
- Elapsed: `one` commits .500; request `two` starts .750 with context `- [self] "one"`; second source unit commits 1.050.
- Both make two provider invocations over the same two inputs, but request grouping/context/output identity differs.

This is a material source/translation-context tradeoff, not merely a subtraction from postterminal waiting. It is why elapsed remains blocked despite an isolated 350 ms wait and the sibling's smaller observed-event replay gains. No absolute latency target or acceptable quality loss was inferred.

## G2 — LLM timing alone changes grouping

`G2_fixed_three_fast` and `G2_fixed_three_slow` freeze A/B/C audio/ASR arrivals; none waits for the prior translation before injecting successor speech.

With elapsed accounting: .050 s LLM service yields three source units at .500 / 1.300 / 2.100. With only service changed to 1.000 s, the same source set yields one merged unit `one two three` at 2.800. Slow execution cancels the logically current attempts at .830 / 1.630; physical controlled calls still finish at 1.200 / 2.000. The replacement calls overlap that retiring work, with a measured maximum of two physically active boundary calls. Both schedules make three invocations over three source inputs, but only one slow attempt is adopted; two pieces of physical work finish after logical cancellation. The correctness repair does not erase this separate result.

**Disposition: deferred, maintainer-owned.** If structural handoff is later selected, its contract must state which eligible source unit closes, preserve the existing running primary/secondary attempts without cancel/reissue, preserve their original prepared context/provider/configuration/admission authority, establish exactly-once eventual result ownership and ordered output, and keep source closure separate from final presentation/late SELF protection under Peer traffic. The current commit path can cancel running work; deleting the RUNNING guard is not such a contract. No structural prototype that crosses those ownership/presentation boundaries was implemented without the maintainer decision.

## Aggregate counters and interpretation

Each policy uses the same **53 schedules / 91 acoustic input episodes**, plus explicitly separate manual/independent-unit inputs. The denominator is not only committed units. Empty/error/timeout/OFF/shutdown, legitimate early-prefix splits, revisions, and different grouping are all retained in the set.

| Owner / policy | Source units | Provider invocations | Spec starts | Adopted primary spec | Logical spec cancels | Qualified wait sum, ms | Failed continuation schedules |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline unchanged | 65 | 101 | 86 | 54 | 28 | 14240 | 8 |
| Baseline elapsed | 67 | 101 | 86 | 56 | 26 | 8840 | 8 |
| Baseline no-op | 65 | 101 | 86 | 54 | 28 | 14040 | 8 |
| Baseline combined | 67 | 101 | 86 | 56 | 26 | 8840 | 8 |
| Correctness unchanged | 58 | 99 | 86 | 49 | 33 | 14240 | 0 |
| Correctness elapsed probe | 60 | 99 | 86 | 51 | 31 | 8840 | 0 |
| Correctness no-op probe | 58 | 99 | 86 | 49 | 33 | 14040 | 0 |
| Correctness combined probe | 60 | 99 | 86 | 51 | 31 | 8840 | 0 |

All observations end with zero physically active **controlled LLM** boundary calls. This does not establish that a logically expired native ASR request was physically complete; pending recognition remains owned and is closed by the existing engine during teardown.

Baseline per-schedule qualified-wait distributions (n=53; deterministic, heterogeneous schedule sums, **not population latency**): unchanged min/median/mean/max = 0 / 350 / 268.679 / 1050 ms; no-op = 0 / 350 / 264.906 / 1050; elapsed and combined = 0 / 0 / 166.792 / 750. Per-case commit times, counts, statuses, and complete timelines are in the linked JSON reports. One sample per fixed schedule does not justify statistical confidence or a production p95 claim.

Correctness reduces premature source splits in the reproduced cases and changes the resulting grouped requests; it does not alter the grace-only interval or claim a speed gain. Exactly-once text/provenance assertions include repeated identical speech and consumed-prefix/suffix terminal reconciliation; exceptional no-text/abort cases are not counted as successful speech output. Semantic equivalence of fake translation strings is unknown, not a pass.

## Correctness implementation transitions and resource lifetime

The SELF owner now records source obligations keyed by full `OwnedVadEvent` acoustic identity, independently of the merge buffer. A new onset adds an obligation; it does not supersede unresolved older episodes. Three subsequent chunks confirm it. An unconfirmed end releases only its own false-start obligation. A confirmed end starts its own existing configured resume-end watchdog even when no buffer exists. Newer onsets cannot cancel that older watchdog.

Matching usable scoped text settles that acoustic continuation, while the existing publication awaiting-end/watchdog path remains independent. Matching scoped empty/failed/cancelled/expired/final terminal settles only its own obligation. A suffix publication UUID is related back to the same acoustic identity by the scoped exchange, not by textual or UUID overlap guessing. All unresolved source obligations participate in the continuation blocker; buffer resume fields are the current presentation mirror, not the authoritative set. Each watchdog releases only its own logical continuation and reevaluates the normal commit path; it does not terminalize or free provider/native work.

A recognition configuration barrier releases old matching settings-scope obligations according to the existing barrier policy. OFF/reset/language clear/shutdown cancel owned watchdogs and clear speech source bookkeeping; manual and Peer work retain their existing scopes. Native independent-input release retires only the matching local source observation; manual/recognition-unit translation still bypasses the merge.

No arbitrary registry capacity, oldest-source eviction, replacement provider call, new recognition retry, or admission fallback was introduced. The bookkeeping lifetime is a subset of actual owned source requests plus bounded terminal handoff:

- Non-overlap streaming input dispatch is serial and waits on the previous request end/terminal.
- Current overlap local batch routes retain their reservation until **ordered** `_finish_turn`; C's completed native work behind B does not release that source envelope early. Existing SELF retained-PCM capacity is 2,880,000 normalized sample-equivalents. Actual 16 kHz production VAD emits fixed 512-sample windows; a confirmed episode includes its start and three chunks (at least 2048 samples), bounding charged confirmed episodes by 1406, plus the in-flight onset/admission transition. The independent 256-event engine handoff bounds already-finished terminal controls awaiting SELF settlement. This is an envelope-derived bound for current actual routes, not a new selected cap or a bound for arbitrary fabricated zero-length ingress.
- False starts are removed at their own end. Prebuffer watchdog expiry removes only logical continuation bookkeeping, leaving recognition/native lifetime with its existing owner. Native independent recognition releases input observations through its existing typed input-terminal path.
- Local decode's `8` backlog value is a **warning threshold**, not a demonstrated hard queue cap; it was not used to claim this map is bounded.

Maintained architecture/changelog proposal for the Director: document that SELF's per-acoustic continuation obligations survive absence of a publication buffer and release only through matching contribution/terminal or the existing scoped watchdog, with source completion/presentation/native-resource retirement remaining distinct. No shared recognition exchange/interface change was required.

## Required matrix status and remaining gaps

| Required boundary | Deterministic status | Real route / limits |
| --- | --- | --- |
| Stable/final before/after VAD, terminal before/after timer, source seal versus receipt | **passed** mechanism assertions, including actual engine withholding pre-end terminal | Fresh final-only local route observed; real stable provider **blocked** without authorized provider access |
| Native duplicate/same snapshot, permitted period, number/negation/question revisions, identical disjoint speech | **passed** provenance/invalidation assertions | Translation/segmentation semantic quality **unknown**; no real stable promotion |
| Delayed A during B pending/confirmed, before buffer, B ended then C false-start/empty/final | Baseline **failed**; correctness candidate **passed** | Actual local final-only prebuffer failure observed by sibling; stable+overlap suffix case is mechanism-only, concrete route **unknown** |
| Empty/error/cancelled/expired, false-start and own watchdog settlement; consumed-prefix terminal reconciliation | **passed** reachable owner/engine assertions | Real provider resource timeout/quarantine conformance **not run** in this worker; injected outcome is not physical completion |
| Fast/slow LLM, continuing multi-source fixed schedule, grouping/adoption/cancel/retiring work | **passed** observations; elapsed selection **blocked**, G2 **deferred** | Fresh ASR/frozen observed replay available; no real-model translation quality or population performance claim |
| One/two targets, secondary prestart, failed speculation, provider/config replacement, recognition config barrier, OFF/shutdown | **passed** bounded cases | Secondary-first/failing-secondary/resource-pressure cross-products and full runtime replacement acceptance **not run here**; Main must run existing focused regressions |
| Active SELF with intervening Peer, accepted late translation, manual/recognition-unit bypass | **passed** projection/owner observations | Actual one-second presenter expiry/protection, stale/sink-pressure regressions and physical destinations **not run here** |
| Actual chatbox send / OSC / HMD visibility | **not run** | Access required only for claims at those destinations; none claimed |
| G3 final integrated receipt / public implementation link / reversible deployment path | **not run / blocked** on Director integration and final pin | Worker did not commit, publish, merge or release |

### Historical final-check request (superseded)

The earlier request was to rerun the historical deterministic commands on a final integrated owner and verify its related owner/overlay/runtime suites. That approach is superseded by SELF-STREAM-2: do **not** repin the historical expectations to the new lifecycle. The current candidate's proof belongs to the maintained SELF-STREAM-2 tests and separate `stream2/` scenario, while this runner preserves historical behavior.

A final fresh/local observed-schedule repair replay should use the candidate owner, not the retained native baseline. Retain the unchanged 400 ms policy as the reversible fallback; correctness reversion would restore the reproduced obligation defect and is not a latency recommendation. Any future elapsed or structural adoption requires its own recorded segmentation/context/presentation decision and new final-candidate acceptance evidence.
