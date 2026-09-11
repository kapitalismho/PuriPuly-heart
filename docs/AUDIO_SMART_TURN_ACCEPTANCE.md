# AUDIO-A3 Smart Turn acceptance receipt

## Baseline and scope

- Contract source: [#134](https://github.com/kapitalismho/PuriPuly-heart/issues/134), including its four finalized design comments.
- Canonical policy identity: AUDIO-LISTEN-1 was finalized on 2026-09-09; the #134 issue body was last updated at `2026-09-09T07:04:53Z`.
- Implementation issue: [#136](https://github.com/kapitalismho/PuriPuly-heart/issues/136).
- Accepted LISTEN base: #135, accepted commit `41210e2` and `docs/AUDIO_CORE_ACCEPTANCE.md`.
- Working-tree baseline: `cf5d038ee9e23d1daab367a00343142e8e504f6a`.
- Reviewed prior implementation candidate: `7167f0b207bbaad090b9b72e5c3002a845287dd6` (`7167f0b2`). This receipt also describes the subsequent uncommitted consolidated repairs; it does not require or claim a self-referential implementation commit.
- Environment: Windows 11 x64, Python 3.12.10, ONNX Runtime 1.28.0, CPU execution provider.

This receipt is evidence for the optional LISTEN Smart Turn endpoint policy only. It makes no claim that the model is superior to persisted VAD hangover, does not enable the feature by default, and does not add speaker inference, policy tuning, retrospective partitioning, a second recognition engine, or a SELF migration.

## Shipped surfaces

- `core/audio/smart_turn_features.py` reproduces the pinned input revision and produces an `80 x 800` Whisper log-mel tensor from the latest eight seconds ending at the actual 224 ms frontier, with left zero padding when context is short.
- `core/audio/smart_turn.py` owns the exact optional model artifact, SHA-256 verification, CPU-only ONNX Runtime session, two intra-op threads, one inter-op thread, sequential execution, and one actual executing inference with no pending queue or replacement. A cached artifact remains truthfully `unloaded` until an owned prepare starts; the complete optional preparation path has a 60 s logical readiness watchdog while late physical work remains owned and reclaimable.
- `core/audio/listen_delivery.py`, `core/runtime/audio_vad_loop.py`, and `core/runtime/peer_channel.py` implement the OFF/ON endpoint policy in the existing LISTEN owner. They bind activation, segment, pause, context revision, source frontier, policy snapshot, and input revision; latch the 512 ms early/fallback choice at the observed boundary; preserve independent four- and six-second source-time deadlines; clip probe input to the exact 224 ms frontier; and invalidate context on capture discontinuity.
- The persisted `intent.desktop_audio.smart_turn_enabled` setting defaults to `false`. The Flet peer VAD card and `/avatar/parameters/PuriPuly_SmartTurn` OSC boolean both read and mutate that same canonical field.
- Runtime snapshots distinguish requested settings from the effective open-segment truth. Requested language/profile/hangover can move immediately, while effective language/profile/hangover and availability remain bound to the current segment until its successor opens.
- `ARCHITECTURE.md`, `docs/vrchat-osc.md`, and the bundled third-party notice describe the runtime boundary, control ABI, and optional model license/source.

## Canonical contract consumed

The canonical policy contract is the finalized [#134 C5 contract](https://github.com/kapitalismho/PuriPuly-heart/issues/134): AUDIO-LISTEN-1 finalized on 2026-09-09, with the issue body last updated at `2026-09-09T07:04:53Z`. This policy identity is separate from implementation input lineage. The feature construction lineage is pinned independently at input authority revision `8dd248b8f73556ac32d24c00223b4b413d4aca98`. This receipt intentionally does not duplicate C5's editable timing or threshold table. Future policy revisions belong in #134; future input changes require an explicitly updated lineage revision.

## Model and input identity

- Artifact: `smart-turn-v3.2-cpu.onnx`
- Source: `https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx`
- Model SHA-256: `2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f`
- License recorded by the upstream model card: BSD-2-Clause.
- Consumed input revision: `8dd248b8f73556ac32d24c00223b4b413d4aca98`
- Authoritative feature source: `https://raw.githubusercontent.com/kapitalismho/PuriPuly-heart/8dd248b8f73556ac32d24c00223b4b413d4aca98/src/puripuly_heart/core/vad/smart_turn_features.py`
- Authoritative source SHA-256: `1a7fde0a790c17c7ca78abe2bd5904c227279a77b02c251fd6c4907a53809b00`
- Runtime configuration: `CPUExecutionProvider`, two intra-op threads, one inter-op thread, sequential graph execution.

The retained parity fixture is a one-second, 16 kHz, float32, 0.25-amplitude 220 Hz sine generated from a float64 sample index. Its raw audio SHA-256 is `b30dde97f347b9f64623089f9395a253f69c4213fda97bb045a558cf734a0755`; its eight-second left-padded input SHA-256 is `de37fbd4b1f1b91ea182fccb6e281065ac5304f7e3d932941a8d8e074daab4c7`; and its `80 x 800` feature SHA-256 is `619865d13db4e64a0640e3d613e021f6971b7e53a03b76a8eaa65a54879f4d52`.

`scripts/bench_smart_turn_input_revision.py` downloads the source at the pinned commit, rejects a source-hash mismatch, executes both the authoritative and production feature construction on that identified fixture, verifies the pinned model, and reports JSON. The retained golden-hash test is `test_pinned_input_fixture_identity_matches_authoritative_revision`. The executed script reported exact feature-array equality and maximum absolute difference `0.0`. Two real CPU model executions on that same identified input both scored `0.8170837163925171`, with measured script durations `48.4918 ms` and `46.3850 ms`. This is input parity against the independent pinned historical construction plus model repeatability; identical scores alone are not described as parity.

A separate production-owner smoke measured model setup at `144.8765 ms`, then one completed inference at `47.0000 ms`. Immediate concurrent submission returned `busy`; the runtime snapshot reported `inference_count=1`, `busy_skip_count=1`, `late_count=0`, and `availability=ready` before close. Controlled late-result tests, rather than this timely real-model input, exercise late-count and authority rejection.

The synthetic tone is not a labeled endpoint-content corpus. These results establish artifact identity, input construction, deterministic repeatability, execution-provider setup, and lifecycle accounting only. They do not establish multilingual accuracy, content regression, precision/recall, or superiority over persisted hangover.

## Verification

- Current focused policy/resource/owner verification passed all 124 cases in 2.88 seconds across Smart Turn delivery/runtime, the actual peer capture owner, peer application replacement, application shutdown, and peer ownership architecture.
- Controlled real-thread barriers cover cancellation during native setup and native inference. Logical close rejects new submission immediately, cancellation of the close waiter does not cancel owned cleanup, an occupied native worker is not replaced, release produces exactly one physical resource reclamation, and later close joins the same cleanup. Failed or timed-out setup remains exactly one attempt.
- Post-validation repair regressions exercise the production settings builder and actual peer owner. Requested `auto` remains `unsupported_auto` even when `local_cpu_auto` resolves the provider to manual English, so Smart Turn performs zero prepare/infer calls and the segment seals at its persisted 480 ms hangover. A live ON/ko/900 segment changed to OFF/ja/1200 continues to project ON/ko/900 with ready availability until sealing; its successor projects OFF/ja/1200 with disabled availability.
- The application-facing cleanup regression uses the real `ApplicationShutdownCoordinator` callback composed for `PeerCaptureSessionOwner`. Its production deadline is finite at 30 seconds; a 50 ms test deadline returns a truthful timeout in under 500 ms while blocked native setup remains owned by the peer/Smart Turn cleanup tasks. Releasing the controlled native barrier lets a later `close()` join that same cleanup and reclaim the one late-created session exactly once.
- Independent validation-only rechecks observed provider facts `manual/en` alongside endpoint-delivery facts `auto/en`, effective profile `unsupported_auto`, and `prepare=submit=infer=0`. During an ON/ko/900 segment with pending OFF/ja/1200 settings, the snapshot remained effectively ON/ko/900 with ready availability until the successor transition. Controlled native setup and inference shutdowns each returned `TimeoutError` under a shortened 0.15 s coordinator bound after `0.16624 s` and `0.16639 s`, respectively; publication was retired, the named close task remained owned, release joined that same task, and exactly one physical close occurred. The production 30 s bound was verified as configured but was not waited out.
- Consolidated C5 repairs are covered by observable regressions: once the 512 ms boundary is observed without a result, a completion carrying a pre-deadline timestamp cannot cause a 544 ms intermediate cut and the segment waits to 800 ms; nonuniform frames no larger than 32 ms clip the probe input and identity to the exact 224 ms frontier; and the independent hard timer seals only the actual owned range while inference remains pending.
- Actual-owner known-loss and unknown-loss gap cases fail/retire the old segment, reject its pending stamp-timely result, and build the successor probe from only post-gap PCM. The known gap preserves its representable epoch and advanced coordinates; the unknown gap starts the supplied new epoch. Neither concatenates pre-gap context nor pads missing time into owned input.
- A cached artifact now projects `unloaded` during supported-ON idle, moves to `loading` only when speech starts real preparation, and reaches `ready` only after checksum and resource construction complete.
- Resource residual investigation found two real unbounded logical waits and repaired both without freeing busy resources or authorizing setup retry. The entire optional Smart Turn prepare path—not merely each HTTP operation—now has the canonical 60 s local-preparation readiness bound. Peer runtime replacement reuses the existing 30 s application cleanup convention, keeps the previous runtime authoritative on timeout, and retains its named cleanup task instead of adopting a replacement while the old resource is occupied. Physical native/network cleanup may intentionally outlive either logical deadline in its single owned quarantine. Tests use shortened injected bounds; the production 60 s and 30 s durations were verified as configured, not waited out.
- The reproducible pinned-input/model command is `uv run --frozen python scripts/bench_smart_turn_input_revision.py <model-path> --repeats 2`; its detailed data and result identity are recorded above.
- Final current-tree baseline-scoped command `uv run pytest --ignore=tests/integration`: `5894 passed, 8 skipped, 267 warnings in 158.26s`.
- The earlier broader `uv run pytest` result belongs to prior candidate `7167f0b2` and is not reused as current-tree proof. The final current-tree command above covers the accepted baseline scope; its eight skips remain three tests requiring real subprocesses, three installer tests requiring unavailable Windows PowerShell, and two Flet process-owner tests requiring real processes.
- Changed-file `ruff format --check` and `ruff check` passed across all 34 changed Python files.
- Actual Flet control smoke began with canonical `false`, changed the real `Switch` to `true`, updated the projected snapshot, and emitted `SmartTurnEnabledIntent(enabled=True)`. The noninteractive harness cannot mount a desktop window for a visual screenshot; the Flet control surface and focused UI tests are the supported equivalent.
- OSC schema/router/application/publisher/OSCQuery tests exercise the real `/avatar/parameters/PuriPuly_SmartTurn` boolean surface, canonical persistence, idempotence, full-state republish, delta publication, and UI projection. No live VRChat client was attached.

## Timing separation

| Stage | Evidence on #136's declared scope |
| --- | --- |
| Source to acoustic seal | Controlled source coordinates verify the canonical C5 probe, strict early-completion boundary, early seal, fallback, age step, and hard deadline exactly. These are source-time boundary checks, not wall-clock output latency. |
| Smart Turn setup/inference | Identified real CPU setup and inference measurements are recorded above; model execution does not block acoustic progress. |
| Provider | The blocked-provider integration proves capture, a pending Smart Turn request, and fallback sealing continue while provider dispatch is stalled. No provider wire-latency benchmark was added. |
| Translation | Translation starts after recognition terminality and is unchanged by this acoustic policy. Existing finite-owner regression suites passed; no new translation wall-clock measurement or quality claim is made. |
| Output | Existing bounded handoff/publication regressions passed. No live sink acknowledgement or source-to-visible-output timing is inferred. |

The independent validation-only OFF pipeline measured both first and second source-time seals at `0.768 s`. Wall-clock observations were: first local seal `0.0021866 s`, second local seal `0.0044042 s`, provider-B handler `0.0044930 s`, provider-A injection `0.0044932 s`, translation idle `0.0052395 s`, and output idle `0.0053444 s`; derived intervals were seal-A to seal-B `0.0022176 s`, provider-A to translation idle `0.0007463 s`, and translation idle to output idle `0.0001049 s`. Provider B was injected before A, the overlay remained empty after B alone, and final publication order was first then second. These observations are explicitly OFF-only scheduling evidence; they are not an ON-path Smart Turn wall-clock benchmark.

The same validation pass exposed the pre-repair native-cleanup condition: blocked setup/inference occupied a worker beyond 350 ms and direct peer close waited for physical completion. The repaired ownership test above now proves the application shutdown boundary is finite while physical native work remains owned and reclaimable.

## Implementation acceptance checks

The formal review wave completed against prior committed candidate `7167f0b2` and supplied the findings addressed by the consolidated repairs above. The table remains an implementation-check claim for the current uncommitted repaired candidate; reviewer verdicts are not treated as acceptance authority, and final Director integration acceptance remains pending.

| Acceptance | Implementation check and evidence |
| --- | --- |
| AC03 | Pass implementation check: default OFF, supported/unsupported language profiles, per-pause probe identity, exact 224 ms input frontier, latched strict 512/800 ms choice with no delayed-callback intermediate cut, immediate age step, callback-independent hard timer, resumption/PSEM precedence, natural/synthetic context, actual-owner known/unknown gap invalidation, and next-segment setting snapshots are covered. |
| AC08 | Pass regression: Smart Turn does not move provider readiness into capture or alter scoped retry/age/config/idle-end behavior. |
| AC09 | Pass regression: OFF/restart and #141 lifecycle quarantine remain authoritative; late Smart Turn results cannot publish or rewrite a terminal segment. |
| AC10 | Pass implementation check: the single native owner has no pending inference queue or repeated setup, total preparation and runtime replacement have finite logical bounds, occupied work remains owned instead of freed/replaced, and existing capture/provider/output capacity and cleanup bounds remain intact. |
| AC11 | Pass regression: language normalization and source ordering remain downstream of an immutable acoustic boundary; regional supported tags resolve only to the frozen threshold profile. |
| AC12 | Pass regression: translation and output streaming/projection contracts are unchanged; Smart Turn controls only LISTEN acoustic sealing. |
| AC13 | Pass regression: a valid prospective speaker boundary and a valid Smart Turn completion arbitrate through the same one-seal controller; no speaker producer was introduced. |
| AC14 | Pass regression: provider setup, handoff, serialization, and endpoint snapshot behavior remain independent from Smart Turn model readiness. |
| AC15 retained #141 portion | Pass regression: Gemini lifecycle/ownership tests remain in the full suite; no provider task or publication authority is transferred to the model worker. Broader simultaneous-route/device stress remains #139 scope. |

## Touched-file inventory

- Architecture, evidence, and control documentation: `ARCHITECTURE.md`; `docs/AUDIO_SMART_TURN_ACCEPTANCE.md`; `docs/vrchat-osc.md`; `scripts/bench_smart_turn_input_revision.py`.
- Model, acoustic policy, and ownership: `src/puripuly_heart/core/audio/smart_turn.py`; `src/puripuly_heart/core/audio/smart_turn_features.py`; `src/puripuly_heart/core/audio/listen_delivery.py`; `src/puripuly_heart/core/audio/ownership.py`; `src/puripuly_heart/core/runtime/audio_vad_loop.py`; `src/puripuly_heart/core/runtime/peer_channel.py`; `src/puripuly_heart/core/peer_capture.py`; `src/puripuly_heart/app/services/peer_application.py`.
- Persisted settings and wiring: `src/puripuly_heart/config/settings_vnext/schema.py`; `src/puripuly_heart/app/wiring/wiring_stt_factory.py`; `src/puripuly_heart/app/ports/settings_view.py`; `src/puripuly_heart/app/services/settings/settings_application.py`; `src/puripuly_heart/app/services/settings/settings_mutation_legacy.py`; `src/puripuly_heart/app/services/canonical_settings_persistence.py`.
- OSC and UI: `src/puripuly_heart/core/osc/control_schema.py`; `src/puripuly_heart/app/ports/osc_control.py`; `src/puripuly_heart/app/ports/ui_models.py`; `src/puripuly_heart/app/services/osc/control_application.py`; `src/puripuly_heart/app/services/osc/control_router.py`; `src/puripuly_heart/app/services/osc/state_publisher.py`; `src/puripuly_heart/ui/views/settings.py`.
- Product data: `src/puripuly_heart/data/THIRD_PARTY_NOTICES.txt`; `src/puripuly_heart/data/i18n/en.json`; `src/puripuly_heart/data/i18n/ja.json`; `src/puripuly_heart/data/i18n/ko.json`; `src/puripuly_heart/data/i18n/ru.json`; `src/puripuly_heart/data/i18n/zh-CN.json`.
- Regression coverage: `tests/core/test_smart_turn_delivery.py`; `tests/core/test_smart_turn_runtime.py`; `tests/core/runtime/test_peer_capture_session.py`; `tests/app/test_peer_application.py`; `tests/core/test_osc_control_protocol.py`; `tests/core/test_osc_state_publisher.py`; `tests/core/test_oscquery_service.py`; `tests/app/test_osc_control_application.py`; `tests/app/test_osc_control_router.py`; `tests/app/test_osc_control_runtime.py`; `tests/app/test_settings_mutation_legacy.py`; `tests/app/test_settings_view_boundary.py`.

The pre-existing user change in `AGENTS.md` remains untouched and is not part of this inventory.


Validation-only temporary probes are absent from the repository. Required architecture, control ABI, third-party, and acceptance documentation is complete, and the changed implementation/evidence surfaces contain no scaffold, placeholder, `TODO`, or `FIXME` markers.
No paid provider call, physical microphone/device scheduling certification, labeled multilingual quality benchmark, or live VRChat session is claimed by this receipt.
