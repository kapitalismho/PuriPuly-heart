# Managed Gemma issue 251 evidence

Task-local validation fixture for [issue 251](https://github.com/kapitalismho/PuriPuly-heart/issues/251). Product sources, live user state, repository history and remotes are not modified by this fixture.

## Boundaries

- `session.py` starts the real application composition, headless presentation/event consumer, settings/secrets owners, control host, Self/Peer translation owners, production provider factory/admission, managed model/runtime and real output routing.
- Synthetic `STTRecognitionUnit` objects enter the actual Self/Peer recognition-unit handlers. There is no audio capture, live ASR, remote translation endpoint or mock inference. This deliberately does not test Self's audio/VAD low-latency merge path.
- Physical capture, microphone testing, overlays, clipboard translation, telemetry, live accounts and remote fallback remain off. Keyless inactive Soniox can report the expected `stt.provider_not_ready` degradation; this is not full-app health.
- There is no supported capture-free Peer activation command. The Director authorized task-local Peer intent through the existing settings owner plus `replace_translation_runtime_effective_flags`, and generation activation on the actual output owner. This substitutes the capture-admission projection, not translation admission or inference. It is **not** evidence of capture activation or acceptance of audio terms.
- The managed process-factory observation retains native stdout/stderr and original/effective arguments. It adds `--verbosity 4` only. In the E4B comparison, it additionally replaces only the existing `--load-mode` value. This is a task-local experiment, not a product switch. Product runtime version validation, model integrity/provisioning owners and server lifecycle stay active.
- A transport observation records rendered messages, actual completion payload parameters/prompt hashes, native response token/cache counts, timings and request/slot identities. It delegates exactly once to the real HTTP/native inference. **Only the comparison** sets completion `seed: 2512026`, identically in all four arms (including prefix preparation), while retaining product temperature/top-p/top-k/min-p/repetition/output-limit behavior. Native response generation settings verify the effective seed. Ordinary product smoke remains unforced sampling. This is a disclosed measurement control, not a product setting. Prefix one-token preparation is labeled separately from translations; no extra prefix/completion request is introduced by observation.
- The Director separately authorized one rejected Vulkan spawn to exercise E4B's genuine CPU fallback and 26B's no-fallback outcome, and one response rejection **after real native completion**. These are injected software failures, not observed hardware/network failures. The response fault does not kill the shared server or substitute the other request.
- Output uses the actual OSC sender and a worker-owned loopback UDP receiver. Positive Self chatbox delivery and Peer routing denial are recorded. Headless UI events establish application event delivery, not GUI/VRChat/HMD presentation.

## Isolation and identity

Each lifecycle gets new environment/config/data/discovery/secrets/temp/cache/prefix roots. Credentials and inherited native/llama/GGML overrides are removed. Secrets use a new encrypted file and process-local passphrase. The scene observer resolves beneath the isolated profile. No live application endpoint is attached.

`prepare` hashes the exact retained E4B target/draft, 26B target and runtime DLL closure, and hardlinks model assets without modifying originals. Every lifecycle rehashes the selected model before use, stages only its selected model, validates its managed manifest, and records the actual candidate import, Python version, source-tree hashes, runtime executable hash and arguments. Model transition staging is explicit. The source runtime is resolved normally; an alternate verified root is an explicitly recorded override, not installed-distribution proof.

Integrity hashing warms the filesystem cache before each lifecycle. This is matched across arms; startup observations are not a cold-disk startup comparison. No cache flush or working-set trim is performed.

The original Windows collector is from commit `c3da5bb5f97ede9b2cfb31ce1d0f1f590e7e2ec6`, `experiments/gemma4_26b_qat/windows_metrics.py`. Original byte SHA-256: `b0a91d610bbcd5905e68470c538d15106f40991c2539026e44f759a7a08aa775`. Its counter/ctypes implementation is reused, not the missing historical experiment checkout. The harness discovers the adapter from owned-process counter instances rather than retaining the historical default LUID. Ruff import sorting/formatting was applied to this task-local copy; the formatted collector SHA-256 is `d8d9782c7743be4ea5d121078317f38a725275f33d7f9c72e4e6fe5c553d9534`. Counter definitions and the historical module's unused default adapter argument are retained; the harness always supplies its own discovery selection.

## Workload and measurement

The E4B comparison uses four fresh managed lifecycles in ABBA order: **mmap A1, none B1, none B2, mmap A2**. All other E4B profile arguments remain product-generated: full offload, flash attention, 4/4 target/batch, no MTP; no 26B cache/offload/SWA controls are copied into E4B.

Each comparison lifecycle has 44 translations:

| Phase | Same pair | Opposite pair | Total translations |
| --- | ---: | ---: | ---: |
| First-prefix, one/channel | 2 | 2 | 4 |
| Retained single/channel | 4 | 4 | 8 |
| Two serial pairs | 4 | 4 | 8 |
| Six simultaneous pairs | 12 | 12 | 24 |

Inputs cover names, numbers, negation, corrections and instructions. Synthetic source history is seeded through `clear_context`/`remember_context` only between fully settled groups, in fixed Self/Peer order. Real resolver and prompt rendering are retained. The comparison freezes seed `2512026` through the approved task-local completion payload seam; all other product sampling/output-limit values are preserved and native response generation settings are recorded. Request-level native prompt hashes, effective payload parameters and actual token/cache counts permit observed equivalence checks. A separate natural-history sequence is retained in the safety lifecycle; controlled history is not ordinary-history correctness proof.

Startup/model loading and first-prefix work are excluded from retained-request distributions. Request latency starts just before recognition-unit ingress and ends at the correlated completed translation event, not admission. Pair makespan starts at the common release and ends after both correlated events. Native processing/stop intervals are joined by request identity, slot and observation window; prefix work is excluded from actual translation overlap. Open client tasks alone do not prove server overlap.

Resource samples are nominally 250 ms across process creation/load, prefixes, active requests and shutdown. `/slots` observations are nominally 25 ms, supplemented by native interval logs. Actual sample gaps and unavailable counters are retained. Report process dedicated/shared GPU, working set/private commit, CPU seconds, available RAM, commit headroom and paging separately from device-wide GPU usage. Sampled maxima are highwater **lower bounds**; private commit is not physical RAM, and shared GPU accounting alone is not spill evidence. No forced cache flush or working-set trim is used.

The original monitor starts after native spawn. No true prelaunch system-memory baseline was captured; earliest postspawn values are explicitly proxies. `pressure.py` uses actual timestamps/gaps and above-threshold brackets to distinguish sampled spans from continuous-duration certainty. Its physical512MiB analysis threshold is separate from the watchdog's512MiB commit-headroom threshold.

The after-proof observer additionally records a read-only system-memory observation immediately before native creation, using the retained collector's `MemoryStatus` layout. Four true prelaunch observations were exercised in the671 source-native after-proof; they do not retroactively fill the missing original baseline. No model was launched solely for a pressure recheck.

The owned-only watchdog requests shutdown if available physical memory is below 256 MiB or commit headroom below 512 MiB continuously for eight seconds. It never terminates unrelated processes. A cutoff is a retained failed/safety outcome, not success or a diagnosed OOM.

Safety execution includes both cancellation directions with other-channel progress and later slot reuse, injected one-channel response failure with a surviving real request, a third manual contender, more than three language prefixes, active settings/backend/model transition as applicable, and active/waiting shutdown. E4B CPU and injected-fallback runs retain real CPU/MTP inference and serial admission. 26B's fixed profile has no CPU fallback.

For targeted after-repair `26b-failure`, `app.status.translation_runtime_ready` is treated as provider binding, not native model readiness. Protection requires effective translation disabled after failed warmup, the provider binding preserved, managed26B/GPU failed, actual owner native readiness absent and no fallback process; then an explicit retry must produce six actual managed smoke completions. This passed at frozen671; historical8786 failures remain unchanged.

The safety after-proof requires two completed Self/Peer pairs after return to GPU even when the repaired settings owner converges without off/on recovery. It records each actual native request's PID/slot/channel and requires four completions on the retained live E4B GPU child before the two-active shutdown/manual-admission boundary. `audit.py` independently requires positive native overlap for both return pairs and joins the admitted manual shutdown to that same PID. Attempted argv is journaled before the strict profile guard and sanitized into the audit, including a rejected CPU request if one recurs. These guards passed in the671 after-proof without an extra off/on recovery.

Stock sourceCLI observation uses Python `-u` for the event follower and retained cursor0, without replacing its event stream. Recorded host argv0 names the dependency-environment interpreter; Windows reports the underlying base interpreter as the OS executable. Both identities remain distinct. The candidate source import root and host working directory were recorded; source import was not inferred from executable basename alone.

## Invocation

Use the dependency-capable interpreter and an explicit candidate source path. Paths below are placeholders, not machine-specific instructions:

```text
<python> experiments/gemma4_managed_251/run.py prepare --python <python> --source <candidate> --assets <new-owned-assets> --e4b-source <retained-e4b-directory> --model26-source <retained-26b-file> --runtime-root <verified-b11537-root>
<python> experiments/gemma4_managed_251/run.py preflight --python <python> --source <candidate> --assets <owned-assets> --root <new-owned-preflight>
```

Model execution is gated by the Director's integrated-ready/hardware ownership grant and `--execute-approved`. Functional smoke and performance windows are separately coordinated; smoke results are not comparative performance evidence.

```text
<python> experiments/gemma4_managed_251/run.py run --case smoke --execute-approved --python <python> --assets <owned-assets> --root <new-smoke-root>
<python> experiments/gemma4_managed_251/run.py run --case smoke26b --execute-approved --python <python> --assets <owned-assets> --root <new-smoke26b-root>
<python> experiments/gemma4_managed_251/run.py run --case abba --execute-approved --python <python> --assets <owned-assets> --root <new-comparison-root>
<python> experiments/gemma4_managed_251/run.py run --case 26b --execute-approved --python <python> --assets <owned-assets> --root <new-26b-root>
<python> experiments/gemma4_managed_251/run.py run --case safety --execute-approved --python <python> --assets <owned-assets> --root <new-e4b-safety-root>
<python> experiments/gemma4_managed_251/run.py run --case cpu --execute-approved --python <python> --assets <owned-assets> --root <new-cpu-root>
<python> experiments/gemma4_managed_251/run.py run --case fallback --execute-approved --python <python> --assets <owned-assets> --root <new-fallback-root>
<python> experiments/gemma4_managed_251/run.py run --case 26b-failure --execute-approved --python <python> --assets <owned-assets> --root <new-failure-root>
<python> experiments/gemma4_managed_251/run.py supported --execute-approved --python <python> --assets <owned-assets> --root <new-supported-cli-root>
<python> experiments/gemma4_managed_251/summarize.py <raw-root> --output experiments/gemma4_managed_251/results/<sanitized-result>.json
<python> -B experiments/gemma4_managed_251/compare.py experiments/gemma4_managed_251/results/<comparison-runs>.json --output experiments/gemma4_managed_251/results/<comparison-analysis>.json
<python> -B experiments/gemma4_managed_251/audit.py <raw-events.jsonl> --output experiments/gemma4_managed_251/results/<sanitized-audit>.json
<python> -B experiments/gemma4_managed_251/pressure.py <raw-events.jsonl> [<additional-raw-events.jsonl> ...] --output experiments/gemma4_managed_251/results/<sanitized-pressure>.json
```

Raw journals/native logs/configs remain local beneath the unique owned temporary root. Durable evidence under this directory must omit private absolute paths, credentials and unrelated transcripts. Failures are retained rather than silently rerun or omitted. Candidate GUI and rebuilt installed-distribution observations belong to their separately coordinated owners and are not inferred from this source/headless fixture.

After offline recovery and the Director's explicit source-first GO, both targeted cases completed on frozen `671ca2ac68a184a2d6915e9efd36464b52d5b0f5`, before the final full native build/installed proof. See `results/afterproof-evidence.json` and `EXECUTION.md`:26B failure/explicit retry6 completions, E4B full safety70 application/78 native completions, both GPU-return pairs overlapping, and owned resource release. The source-native window is finished and hardware release was explicitly notified. This is not installed-proof or whole-issue acceptance; severe26B loading-memory depletion and new semantic errors remain explicit. No original setup/matrix was repeated, and the old ABBA archive and original result records were not overwritten.

The exact seven-module fixture executed for the671 after-proof is preserved in [afterproof-harness-671.zip](results/afterproof-harness-671.zip), SHA256 `6412fa95e7b99276cbc6280d14d6cd2e527c3dbf8aff412f2ab76ceb8df49307`. Every member matches the recorded in-run fixture digest. The current working fixture subsequently received the repository's Black formatting in four files; all seven modules have identical parsed ASTs before/after, without exclusions. No native result was regenerated. See [evidence repair verification](results/evidence-repair-verification.json). Use the archived fixture for exact historical byte reproduction, separately from the older ABBA snapshot.

Final671 full build, isolated installation and nonmodel integrity staging passed. The user then explicitly skipped latest installed actual inference after the memory warning; no control-host/model probe or prepared watchdog was executed. That boundary remains unverified, not a pass. [Completion and remaining criterion](COMPLETION.md) separates current source proof, older installed proof, final build evidence, cleanup and publication authority.

`compare.py` requires four complete comparable arms, unchanged candidate/harness content, matched actual native profiles except mode/owned paths/port, fixed native seed and otherwise unchanged effective sampling, identical native request prompt hashes, positive server overlap and the required resource counters. It produces separate per-arm timing/resource distributions and ordering-bracket differences; it does not invent a speedup threshold or automatically adopt a mode. Original sanitized resource samples and completed synthetic outputs remain in the run summaries. Unique source/target/output quality cards group every completed output for human review, with lexical/numeric/negation/truncation flags as aids rather than an automatic quality pass.

## Decision rule

Propose non-mmap only for a repeated useful benefit without unacceptable quality, stability, memory-pressure or latency tradeoffs. Otherwise retain mmap; equal/better mmap or inconclusive measurements are valid negative outcomes. Bring an unresolved material tradeoff to the Director. The Director owns adoption, integration and publication.

For the original8786 ABBA176 evidence, the Director accepted **RETAIN mmap**. See `RESULTS.md`, `RESOURCE_PRESSURE.md` and `EXECUTION.md` for separately attributed8786 comparison/matrix observations and671 after-proof, semantic failures, resource limits and separate-owner native/installed boundaries. This is not whole-issue acceptance.
