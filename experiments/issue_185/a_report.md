# Issue 185 — Outcome A: Self speech translation execution overlap

**Status:** implementation verified in controlled production-owner scenarios. Execution baseline HEAD `78d90ca9722d3c88e05448bbe7c958892c5b11ec` (includes intervening shutdown-window change after issue publication `b6dab6fccf843fd86824bb6b9dd8770c0d9a6214`). Result is the working-tree change to `translation_turn.py`, focused tests, and this experiment; no integration commit is claimed.

## Mechanism and boundary

`TranslationTurnLifecycleOwner.submit()` serializes channel admission; `SelfTranslationChannelOwner.on_parent_admitted()` calls `TranslationRequestOwner.admit()` before parent execution. `admit()` prepares request context then remembers source text. Its `process()` receives the already-prepared request when invoking the provider: B does not need A's translated response for its context. Previously `_run_parent()` awaited predecessor semantic completion before acquiring an execution slot for every non-Peer, non-dual-target parent. A new deterministic speech-parent regression failed on the baseline logic: B's provider never started during A's held response (the one-second test gate timed out). The same regression passes after excluding **only Self speech** from this pre-execution dependency.

`_execute_child()` still waits on the predecessor's **closed event after computation** before output submission. Admission still uses the existing serialized Self owner, slots still limit speech to two running parents, and output destination batches still enforce their own order. The non-speech/manual pre-execution condition, Peer path, dual-target path, speculative selection, generation fences, and configured policy timers are untouched. The old test requiring Self speech to execute serially was deleted rather than preserved as a conflicting implementation test.

## Runnable handoff evidence

Run `python experiments/issue_185/a_probe.py` at repository root with the project's Python 3.14 environment and dependencies installed (for example, `uv run python experiments/issue_185/a_probe.py`). The retained run used an existing project environment's CPython 3.14.7 interpreter. [`a_probe.py`](a_probe.py) uses the production `TranslationTurnLifecycleOwner` → Self `on_parent_admitted()` → `TranslationRequestOwner` → provider backend → production output callback, assembled by the repository's translation-owner composition helper. It injects two **synthetic** speech parents directly at the translation-owner boundary, holds A's translation provider, observes B's actual provider invocation, and records prepared context, provider readiness, and ordered output handoff. It does not directly await an output callback from a capture dispatcher or pretend to model capture/STT. Output observation wraps the production callback; it does not stand in for an overlay or physical delivery.

Retained [`a_trace.json`](a_trace.json): one Windows 11 / CPython 3.14.7 run, Self en→ja, configured `${sourceName}|${targetName}` prompt, context max 3 entries, single target, requested 20 ms gate on A, no paid provider, capture device, GPU, renderer, or HMD. The trace verifies that prepared context, prompt hash, provider generation 0, and configuration revision 0 correspond to invocation. Relative loop-monotonic timestamps (µs from scenario start):

| Fact | t (µs) | Interpretation |
| --- | ---: | --- |
| A ordered preparation / provider invocation | 2,750 / 2,866 | A provider remains gated |
| B ordered preparation / provider invocation | 2,975 / 3,038 | 63 µs observed post-preparation scheduling interval in this one run |
| B provider result ready | 3,041 | A has not completed; B context includes synthetic A source, not A translation |
| A provider result ready | 36,349 | A gated for this scenario, not a service latency measurement |
| A / B ordered output handoff | 36,476 / 36,747 | B waits for A's legitimate order despite early computation |

B result-ready → B ordered handoff was 33,706 µs in this deliberately gated scenario; it is **legitimate ordering**, not removable display lag. A ready → A handoff was 127 µs and A ready → B handoff 398 µs in the same one run. Baseline provides a deterministic no-start result while A unresolved, not a corresponding measured provider/context-to-invocation distribution. These intervals demonstrate a removed execution dependency, **not** an absolute or universal latency, output application, remote chatbox delivery, overlay speedup, or HMD presentation gain. No change to the 400 ms Self post-end grace or one-second overlay replacement policy.

## Checks and scope

- Before fix: new production-lifecycle gate test `test_self_speech_executes_after_ordered_admission_before_predecessor_completes` failed because B's start timed out with A held.
- After fix: `tests/core/test_translation_turn_owner.py`, `tests/core/test_dual_target_translation_lifecycle.py`, `tests/core/test_translation_request_owner.py` — passed; Self single/dual exact two-running/eight-waiting capacity and pending expiry coverage, empty/failure/source-only release, speech-only cancellation and new-generation progress, ordered output, Peer and manual/dual behavior.
- `tests/core/test_self_translation_low_latency.py` — passed, including speculative and merge behavior.
- `tests/core/test_output_owner_wiring.py` and `tests/core/test_translation_output_streaming.py` — passed, including output and mixed-channel owner behavior.
- `a_probe.py` — passed and wrote eight correlated events in `a_trace.json`.

The deterministic tests cover empty/failure, source-only/expiry, cancellation and stale-generation order release through existing owner boundaries. The probe uses synthetic requests rather than microphone/STT, network service or physical display; real provider service-time effects and HMD-visible latency remain **unknown**. No claim of a hardware/field speedup, paid-provider conformance, or assembled A–H integration acceptance. Architecture drift relative to `docs/architecture.md`: **none observed**; existing translation lifecycle and output owners retain their documented responsibilities.
