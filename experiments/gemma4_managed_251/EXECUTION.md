# Execution ledger

This ledger separates completed observations from unperformed checks. Functional smoke is not the loading-mode experiment and establishes no speedup, cold-start or memory SLA.

## Completed before the native-build barrier

| Observation | Outcome | Durable record |
| --- | --- | --- |
| Exact retained model identities | Fresh sizes/SHA-256 matched E4B target, E4B MTP drafter and the required 26B QAT target. Originals were not modified. | `results/asset-identities.json` |
| Source runtime closure | Fresh SHA-256 for all 47 selected CPU/Vulkan runtime files; actual application launches subsequently reported b11537 / `8e2d31e0e`. | Asset record and native identity in smoke records |
| Capture-free composition preflight | Actual CLI capabilities/settings/status queries completed. Self/Peer capture desired/effective false, microphone test off, OSC receiver/query off. Expected inactive keyless-STT failure retained. No managed process launched; owned tasks drained. | Raw local preflight journal; summarized here |
| Initial E4B attempt A | Four translations completed, including a server-overlapping same-pair Self/Peer pair, before the fixture incorrectly treated a known inactive-STT degraded settings commit as fatal. This failed attempt is retained, not replaced. | `results/initial-smoke-a.json` |
| Repaired E4B attempt B | Six translations completed on one managed Vulkan server: Self-only, Peer-only, same-pair overlap and opposite-pair overlap. Native processing intervals and `/slots` both demonstrated two active requests. Positive Self chatbox packets and three explicit Peer chatbox denials were recorded. | `results/initial-smoke-b.json` |
| Initial 26B smoke | Six translations completed on one managed Vulkan server, covering the same single/same/opposite paths. Actual native startup reported the selected RX 7900 XTX, host-expert allocation, 24 expert layers, 2045.07 MiB actual MoE cache allocation under the requested 2048-MiB budget, and three context sequences. The original/effective command records preserve the complete fixed profile. | `results/initial-smoke26b-a.json` |
| Functional synthetic quality readback | Human review of the twelve outputs in the two successful smoke records found preserved names, quantities, negation, corrected meeting times and destinations. E4B's English battery sentence has minor idiomatic awkwardness; quality is not described as perfect. Surface-check calibration flagged none of these twelve outputs, illustrating that such flags do not replace human review. These unforced-smoke observations are not a loading-mode comparison. | Complete synthetic source/output pairs in the two records |
| Resource release after smoke | Owned native process exit and closed server port observed for both successful lifecycles; no remaining fixture tasks. Windows requested termination returned exit code 1; this is recorded with the owned termination request, not labeled an unexplained model crash. | Cleanup and process-shutdown events in smoke records |
| Offline correlation check | All 16 retained completed request records matched an actual native slot interval and rendered-template prompt hash. For every record, native total prompt tokens equaled `cache_n + prompt_n`; output token counts matched; EOS, no truncation and channel association were observed. Every retained simultaneous pair had positive native overlap. | Executed assertion check; underlying request-level records above |
| Sanitization check | Search of durable result files and README found no personal path/user-root matches. Raw native logs and full local diagnostics remain only in owned temporary storage. | Search observation |
| Scoped lint/format | Experiment-only Ruff check and format checks passed after fixture preparation. No product test/build result is claimed by this worker. | Executed Ruff commands |
| Comparison rejection check | Feeding the actual non-comparative E4B smoke record to the comparator returned `complete_comparison: false`, zero comparative outputs and missing-comparison validation errors. Smoke cannot accidentally become an ABBA pass. | Executed no-model aggregation check |
| Structured metadata check | Mapping-proxy channel/turn metadata now serializes to its actual mapping, rather than a type-name placeholder; deterministic no-model check passed. Initial smoke publication IDs, routes, reasons and result IDs were retained, but their additional mapping metadata was not captured and is not reconstructed. | Executed serializer check; initial raw evidence preserved |

The bounded fixture repair accepts only a terminal degraded **settings commit** whose typed diagnostic is `provider.stt / stt.provider_not_ready`. It records that receipt, reads back committed fields, verifies physical capture remains off, and subsequently checks actual effective language projection and translation toggles. It does not convert the application to fully healthy, ignore arbitrary nonzero exit codes, or replace a provider.

Receipt acceptance proves committed intent and the stated limited fields only. It does not prove effective backend convergence: the later E4B GPU-return failures below demonstrate why actual native profile/process/completion observations remain necessary.

Native response `timings.cache_n` is cached **input**. Native top-level `tokens_cached` is the slot's post-completion count and must not be substituted for cached input. Both original fields are retained. Working set/private commitment and process/device GPU counters retain their separate meanings.

Initial smoke used unforced random sampling. Final ABBA used the approved fixed seed2512026 and checked actual outgoing/native-effective parameters and both outgoing/raw-echo prompt hashes. Its observer introduced no extra prefix/completion request. The exact successful comparison fixture is frozen in `results/abba-harness-v3.zip`; subsequent safety-observation and after-repair assertion changes do not replace it.

## Executed at frozen product8786

| Case | Actual result | Durable record |
| --- | --- | --- |
|`abba`|Four fresh mmap/none/none/mmap lifecycles,44 translations each;176 completed,48 native-overlapping pairs, maximum2 including prefixes, one process/model per arm. Strict comparison0 errors/0 warnings. All children exited, ports closed and tasks drained.|`results/e4b-abba-runs.json`, `results/e4b-loading-comparison.json`, `results/e4b-startup.json`|
|Failed ABBA fixtures|Attempts1/2 failed at startup-prefix raw echo equality, zero application translations. Attempt2 retained the response before assertion and showed a native-added leading `<bos>` with exactly equal remaining bytes. All four arms restarted fresh after approved bounded observation correction; no pooling.|`results/abba-attempt1-failure.json`, `results/abba-attempt2-failure.json`|
|`26b`|Exit0:44-request workload plus natural history/order/cancellation/response-fault/manual/prefix/transition/shutdown sequence.69 recognition-request completions,76 native translation completions retained. Original26B GPU admission2; transitioned E4B CPU/MTP1; returned E4B GPU2. Three child lifetimes, never more than one resident managed process observed.|`results/26b-execution.json`, `results/26b-audit.json`|
|`cpu`|Exit0:44 completions on actual b11537 CPU/MTP. Detected8C/16T; target/batch4/12, draft1/1; unchanged CPU profile, native maximum1 despite simultaneous ingress and1,566 native draft tokens.|`results/cpu-execution.json`, `results/cpu-audit.json`|
|`fallback`|Exit0: one injected Vulkan spawn rejection, actual product-owned CPU/MTP fallback;44 completions, native maximum1 and1,566 native draft tokens. Not a hardware-failure claim.|`results/fallback-execution.json`, `results/fallback-audit.json`|
|`26b-failure` original|Assertion failed and preserved. Managed26B/GPU failed and no native/fallback process existed; effective translation incorrectly remained enabled after warmup failure. Historical provider-binding/native-readiness assertion was also overstrict; adjudication below preserves rather than hides this distinction.|`results/26b-no-fallback-attempt1.json`, `results/26b-no-fallback-audit.json`|
|E4B safety attempt1|70 recognition-request completions; ordering/cancellation/fault/manual paths completed, actual CPU transition occurred. GPU intent then committed with inactive-STT degradation while actual CPU remained. Two-active GPU shutdown check timed out.|`results/e4b-safety-attempt1.json`, `results/e4b-safety-attempt1-audit.json`|
|E4B supported recovery attempt2|66 recognition-request completions. Actual off/on receipts applied and prepared new Vulkan child62788; first Self/Peer translations then failed, child exited, no post-recovery native completion. Timeout retained; not an effective GPU-return pass.|`results/e4b-safety-attempt2.json`, `results/e4b-safety-attempt2-audit.json`, `results/recovery-diagnosis.json`|
|`supported`|Stock sourceCLI start/capabilities/settings/manual/stop, explicit26B selection and fresh-host restart persistence/manual completion. Two distinct hosts and receipt-correlated public translation events plus actual owned OSC packets. Three native children released.|`results/supported-source-cli.json`|
|Current cleanup inspection|One read-only inspection:33 recorded owned PIDs all absent,19 recorded server ports all closed. No unrelated process terminated; model/GPU/CPU resources released.|`results/final-cleanup.json`|
|Offline quality/resource analysis|All13 ABBA cards/176 outputs plus175 matrix cards/293 recognition completions read. Eight ABBA correction inversions, seven additional matrix factual errors and lexical concerns retained.17 additional native-only/manual/shutdown outputs read separately. Severe transient available-memory minima and exact sampling/threshold bounds documented; no RAM-safety claim.|`results/e4b-human-quality-review.json`, `results/matrix-human-quality-review.json`, `RESOURCE_PRESSURE.md`|

During ordered26B-scenario shutdown one Peer translation legitimately completed and published; the admitted waiting manual native result was not published. All explicitly cancelled IDs and response-fault IDs had no translation publication anywhere in their final journals. These are distinct observations; no “zero publication after close request” assertion is invented.

The Director **accepted RETAIN mmap**, with no default change or retuning. See `RESULTS.md` for timing/resource tradeoffs, quality limitations and separate installed/GUI proofs. Comparison acquisition completeness is not whole-issue acceptance.

## Adjudicated repairs and after-proof contract

`app.status.translation_runtime_ready` means provider binding, not native model readiness; it was already true while disabled. The actual failed-enable bug is effective translation remaining enabled after managed warmup failure. Original failed assertions/records remain untouched. Prepared targeted protection requires enabled=false, provider binding=true, managed26B/GPU failed, actual native owner readiness=None and zero native/fallback processes, followed by explicit successful retry and six real smoke completions. This replacement preserves native-readiness protection instead of merely dropping a boolean assertion.

The E4B source-supported diagnosis is a partial-apply signature hole: successful LLM rebinding precedes STT convergence, but signature publication follows it. Subsequent GPU intent may compare against a stale GPU signature and leave the existing CPU-bound provider. Connection-only prefix refresh is also skipped. Canonical enable demand can prepare a GPU child without rebinding that provider. Private cache/backend state and the original rejected argv were not directly captured; do not label a post-recovery CPU spawn as observed.

The Director authorized targeted source-native after-proof at frozen671 after the accepted app/provider/signature repairs and review barrier. Both cases completed as recorded below; no original failure was rerun merely to confirm it and no older completed matrix was repeated. True prelaunch memory baselines were not captured in the originals; earliest postspawn proxies remain explicitly labeled.

## Offline replacement handoff and gated after-proof

The original worker ended after provider-dispatch INTERNAL failures; those are not product failures. The replacement recovered its last recorded state and the existing asset bundle without executing a model or repeating preparation. The original raw/sanitized failures, comparison snapshot and all original pressure/quality records remain unchanged. The retained asset manifest names the source b11537 root and exactly the E4B target/drafter plus required26B target; its preparation candidate is provenance, not the candidate used for a later lifecycle. Each new lifecycle records its own actual import/head/source hashes.

The7cff native build was intentionally cancelled for an accepted remaining review repair: generic combined-signature publication could erase the failed-LLM retry marker during unrelated locale success. The source owner completed generic/STT-only writer and initial actual-LLM binding migration in frozen671; the Director supplied targeted-test/review readiness and explicit exclusive model GO. This worker did not change product sources, profiles or defaults. The source26B failure/retry and E4B safety window ran **before** the final full native build/installed proof; afterward all owned resources were released and the Director was explicitly notified. Original33-PID/19-port cleanup evidence remains separate and was not rechecked.

Prepared fixture protection closes a concrete coverage gap: successful automatic GPU rebinding previously received only one post-return pair because the repeat was conditional on off/on recovery. Two pairs/four native completions are now mandatory in either path. The fixture retains ready/effective GPU state and PID-linked provider requests; the offline audit requires positive native overlap in both pairs plus same-PID two-active/admitted-manual shutdown. The26B retry additionally checks unchanged26B/GPU selection, effective enabled/provider binding, ready managed/native state, one live exact26B child and six actual native translation completions. No recovery provider/runtime override or profile-guard exception was introduced.

The following is the executed case/aggregation protocol, not authorization for another run. Existing prepared assets were reused in two new isolated roots. Placeholders deliberately sanitize public paths; the actual local asset manifest/interpreter identities were recovered rather than guessed. No `prepare`, ABBA, CPU/fallback or sourceCLI rerun was performed.

```powershell
$python = "<dependency-capable-python>"
$assets = "<retained-verified-assets>"
$after = Join-Path $env:TEMP ("puripuly-managed-251-afterproof-" + [guid]::NewGuid().ToString("N"))
& $python -B experiments/gemma4_managed_251/run.py run --case 26b-failure --execute-approved --python $python --source . --assets $assets --root "$after/26b-afterproof"
& $python -B experiments/gemma4_managed_251/run.py run --case safety --execute-approved --python $python --source . --assets $assets --root "$after/e4b-afterproof"
```

Run the cases sequentially, not together. Retain every supervisor exit/failure and raw journal; a failed case is not silently rerun or treated as proof. Inspect any demonstrated product defect before proceeding with affected work. The first case injects one26B Vulkan spawn rejection, checks disabled/failed/native-not-ready/no-fallback while preserving provider-binding availability, then retries the **same** selection through supported `translation set on` and completes the real six-request smoke. The second runs the full44-request E4B GPU workload and safety lifecycle, including GPU→CPU/MTP→GPU under preserved independent inactive-STT degraded receipts, cancellation/fault-survivor/manual/LRU paths, repeated return pairs and two-active/admitted-waiting-manual shutdown.

After each case settles, aggregate into **new** result files; never pool these outputs with ABBA or overwrite the8786 failure records:

```powershell
& $python -B experiments/gemma4_managed_251/summarize.py "$after/26b-afterproof" --output experiments/gemma4_managed_251/results/26b-afterproof.json
& $python -B experiments/gemma4_managed_251/audit.py "$after/26b-afterproof/26b-failure/events.jsonl" --output experiments/gemma4_managed_251/results/26b-afterproof-audit.json
& $python -B experiments/gemma4_managed_251/summarize.py "$after/e4b-afterproof" --output experiments/gemma4_managed_251/results/e4b-afterproof.json
& $python -B experiments/gemma4_managed_251/audit.py "$after/e4b-afterproof/safety/events.jsonl" --output experiments/gemma4_managed_251/results/e4b-afterproof-audit.json
& $python -B experiments/gemma4_managed_251/pressure.py "$after/26b-afterproof/26b-failure/events.jsonl" "$after/e4b-afterproof/safety/events.jsonl" --output experiments/gemma4_managed_251/results/afterproof-memory-pressure.json
```

Every additional synthetic native/application output was read, including rejected/cancelled/manual/shutdown outputs; separate factual/lexical failures and counts are recorded below. Actual candidate/import/runtime/profile identities, effective GPU request PID/slot records, both native-overlap intervals, preserved degraded receipts, cleanup/task/port observations and per-PID pressure are retained. Four new prelaunch system-memory observations were exercised; they cannot fill the missing original baseline. Scoped sanitization and the Director's integration checks remain separate from runtime proof.

## Executed at frozen product671

Exact candidate: `671ca2ac68a184a2d6915e9efd36464b52d5b0f5`; actual source-tree SHA256 `cbd004e0bd448c1b6efb20f0836b247a5bb0c116adf3b5d736a878a81eb496c5` matched in both fresh source lifecycles. Runtime remained b11537/`8e2d31e0e` with unchanged exact model assets/profile guards and identical fixture hashes between cases. Summary: [afterproof-evidence.json](results/afterproof-evidence.json).

Exact executed fixture: [afterproof-harness-671.zip](results/afterproof-harness-671.zip), SHA256 `6412fa95e7b99276cbc6280d14d6cd2e527c3dbf8aff412f2ab76ceb8df49307`. All seven member digests match the in-run record. Subsequent CI-required Black formatting changed four source representations but no parsed AST; no native evidence was rerun or relabeled. [Bounded evidence repair checks](results/evidence-repair-verification.json).

| Case | Actual outcome | Durable records |
| --- | --- | --- |
|26B failure/explicit retry|Exit0. One injected pre-spawn Vulkan rejection; effective enabled=false/provider binding=true, exact26B/GPU failed, native readiness=None and zero native/fallback processes. Explicit same-selection retry produced one exact26B Vulkan child PID69864 and six real native/application translations; native maximum2.|`26b-afterproof.json`, `26b-afterproof-audit.json`|
|E4B full safety|Exit0. Original44 workload plus safety yielded70 recognition completions and78 native translation completions. GPU50884→CPU/MTP72040→GPU69380; native maxima2/1/2, CPU draft tokens6 for the short transition requests. Seven inactive-STT degraded receipts remain degraded, including both backend commits; effective GPU return succeeded without off/on recovery.|`e4b-afterproof.json`, `e4b-afterproof-audit.json`, typed receipts in `afterproof-evidence.json`|
|Repeated GPU return/shutdown|Four actual Self/Peer completions remained on GPU PID69380, slots0/1. Return-pair native overlaps1.036750s and1.289447s. Same PID then had two fresh active native requests plus admitted waiting manual turn at shutdown.|`gpu_return_native_pairs`, `backend_return_outcome`, `shutdown_waiting_manual_admission` in audit|
|Ordering/cancellation/fault/manual/LRU|Both channel-order cases retained other-channel progress; both cancellation directions retained survivor/slot reuse and no cancelled publication; injected post-native response rejection retained survivor with rejected ID unpublished. Manual shared max2, prefix LRU and active backend transition completed.|Raw-controlled outcomes and independent final-journal audit|
|Owned release|All four native children exited after owned termination; four ports closed, both fixture task registries0 and supervisors completed. Hardware release explicitly notified before offline analysis. Native exit1 is paired with requested termination, not labeled an unexplained crash.|`afterproof-release.json`|
|Quality/pressure|All59 unique cards representing76 application completions plus8 additional native outputs read. Two new factual errors and2 lexical concerns, no blanket quality pass. Severe26B loading-memory minimum155,648B; no8s cutoff, not RAM safety. Four true new prelaunch baselines captured.|`afterproof-human-quality-review.json`, `afterproof-memory-pressure.json`, `RESOURCE_PRESSURE.md`|

The eight additional native outputs include two publications: manual-capacity completion and one Peer completion during ordered shutdown. Six remained unpublished, including both explicit cancellations, injected rejected response, retired-backend result, long Self shutdown result and queued manual shutdown result. Legitimate ordered-shutdown progress is preserved rather than asserting zero publication from close request.

Original8786/ABBA evidence and every historical failure remain immutable and separately attributed. Final full native build/installed proof, GUI boundaries, integration and issue acceptance remain with their assigned owners; this worker claims completed source-native after-proof, not whole-issue acceptance.

## Final distribution boundary and authorized skip

The full671 native pipeline completed exit0 with552/552 packaged source hashes matching unchanged input. A genuine unique-identity smoke installer compiled/installed exit0, and exact26B nonmodel hardlink/checksum staging passed. [Final distribution record](results/distribution-final-evidence.json) retains the commands, artifact identities and original records.

Post-staging physical headroom remained about6.9GB; a later Director observation was about4.0GB. After the explicit memory-risk question, the user requested skipping only latest installed actual inference. No final control-host/model probe or prepared2s watchdog ran. Current installed child/profile/completion/fresh-host persistence is therefore unverified; original8786 two-host success is not relabeled. The original issue is not declared complete.

Both unique test installations and registrations were removed; normal user-Temp model aliases/state were cleaned after raw-state evidence preservation. Six permission-denied Inno/Windows temporary files totaling8,986,024bytes remain, without ACL escalation. Separately, [GUI resource cleanup](results/gui-resource-cleanup.json) removed the task-created target copy/cache hardlink while preserving the original model and historical evidence. No further model execution or resource-safety claim follows from cleanup.

## Explicit limits

- Controlled recognition and Peer effective-state projection are not audio, consent, capture-activation or ASR evidence.
- Headless UI events and owned loopback OSC receipt are not actual GUI or VRChat presentation evidence.
- Source launch evidence is not installed immutable-payload evidence. GUI/installer/installed-runtime observations remain with the assigned owners.
- Smoke ran in the authorized functional integration window, alongside permitted deterministic integration activity. Its timings are not final comparative measurements.
- This workstation's 7900 XTX observations do not establish an 8-GB-card or VRChat frame-time guarantee.
- Available-memory/commit watchdog cutoff is a safety failure; sampled maxima are highwater lower bounds, private commit is not physical RAM, and shared GPU accounting alone is not uncontrolled spill.
