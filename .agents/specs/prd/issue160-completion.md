# Issue 160 completion evidence

## Source identity

- Branch: `listen-add-soniox-speaker-segmentation-and-provi`
- Branch/local base: `150e3980276f8610ed632d7e04a530f34f0bb15c`
- Integrated overlay #162 source: `55a2ac3991ba55173e3e09f3da8b8fa9a8d52c2e`
- Designated #158 source: `b4ff4a2b90e3b688a7f85bd775ac620ca108e24a`
- Inherited #159 authority: `LISTEN-ENDPOINT-7S-HYST010`, amendment comment 5653348843
- Overlay contract: protocol 8 / execution contract r2

## Implemented result

Soniox final-token speaker attribution now survives independent final speaker-run normalization with provider-session scope. Language/speaker/unknown changes produce conserved ordered children. LLM-backed peer translation submits one whole-parent logical batch with explicit child identities and maps responses by UUID. The common OutputRuntime/OverlayPresenter path fills vacancies immediately and gates logical replacements at 1.0 second, using shared SELF/PEER occupant state, existing expiry, authority, and bounded output ownership.

The retained policy remains latest-conversation-first. Pacing can increase explicit overload retirement. There is no new presentation-wait TTL, no larger queue, and no physical display acknowledgement claim.

## E09 result-order frontier

Command:

```text
uv run --no-project python -m pytest -q tests/core/test_translation_turn_owner.py::test_peer_result_availability_waits_for_predecessor_terminal_and_failure_releases_frontier tests/core/test_translation_turn_owner.py::test_peer_cancellation_releases_frontier_for_new_generation tests/core/runtime/test_output_runtime.py::test_retired_peer_generation_cancels_active_and_rejects_late_output --tb=short
```

Result: 3 passed. The production `TranslationTurnLifecycleOwner` scenario makes the second peer result available while the predecessor output is still blocked/nonterminal, observes no second output until predecessor completion, then emits `first`, `second`. A failed predecessor releases the next result. Channel cancellation releases a new generation while cancelling the old waiting frontier. The production `OutputRuntime` rejects a late retired-generation result as `publication_generation_retired`.

This is distinct from E04's shuffled item mapping inside one LLM batch.

## E14 real desktop/native smoke

The smoke used the production `OverlayRuntimeHandle`, `OverlayBridge`, `OverlayProcessManager`, `DesktopFletOverlayRunner`, `OutputRuntime`, and `OverlayPresenter`; the manager wrote the protocol-8 launch manifest with the live Python owner PID and held its auth/lifetime through graceful shutdown. Synthetic captions only were used.

The first instrumented run isolated a product regression: the real desktop emitted `desktop_first_visible`, but `OverlayBridge` omitted that established process-manager event from `_REVERSE_KNOWN_TYPES`, retired the authenticated connection as `unknown_reverse_message_type`, and caused `runtime_disconnected`. `core/overlay/bridge.py` now accepts that lifecycle event; a regression test keeps the connection authenticated after it. Probe-only faults encountered separately were an unavailable Pillow import and an incorrect probe import; neither implicated product code. One later startup attempt reported the existing recoverable `window_reveal_lost`; a clean retry completed.

Successful command:

```text
PYTHONPATH=src uv run --no-project python .tmp_issue160_desktop_smoke.py
```

Observed real-time logical selections on the live Flet desktop surface:

```text
1.484s  captions 1,2 selected immediately
2.484s  captions 2,3 selected
3.484s  captions 3,4 selected
4.484s  captions 4,5 selected
4.984s  application_applied=5, process state=connected
6.078s  graceful stop: exit_code=0, acknowledged=true, forced=false, cleanup_succeeded=true
```

Durable evidence:

- `evidence/issue160-paced-desktop-smoke.jsonl` — production owner/process/bridge events and exact pacing observations.
- Local-only `evidence/issue160-paced-desktop-final.png` — real Windows desktop capture visibly showing synthetic captions 4 and 5 in the final two selected rows. SHA-256: `0eaa9c92b9820eae4f395fe4aac46df09311aa3e6dd2fc8511342b613c701953`. Not committed because the transparent overlay capture includes unrelated desktop content.

The successful smoke proves application-owned logical admission and actual desktop transport/rendering. It does not claim exactly one second of physical exposure, lossless intermediate transport, HMD visibility, or deployment certification.

## Final automated verification

Final Python suite command:

```text
PYTHONPATH=src uv run --no-project python -m pytest -q --tb=short --junitxml=.agents/specs/prd/evidence/issue160-pytest.xml
```

Result: **6,029 passed, 37 skipped, 0 failed, 0 errors; 6,066 total**, 186.297 seconds in JUnit / 187.59 seconds wall time.

Final lint command:

```text
PYTHONPATH=src uv run --no-project python -m ruff check src/puripuly_heart tests/architecture/test_translation_request_ownership.py tests/core/runtime/test_output_runtime.py tests/core/test_overlay_bridge.py tests/core/test_overlay_presenter.py tests/core/test_soniox_multilingual_release_readiness.py tests/core/test_translation_output_streaming.py tests/core/test_translation_request_owner.py tests/core/test_translation_turn_owner.py tests/helpers/translation_owners.py tests/providers/test_soniox_backend.py
```

Result: all checks passed.

Native commands ran inside Visual Studio 2022 Build Tools Developer Command Prompt 17.14.22 with `CARGO_TARGET_DIR=C:\tmp\ph160-target`:

```text
cargo test --locked --manifest-path native\overlay\Cargo.toml
cargo build --locked --release --bin PuriPulyHeartOverlay --manifest-path native\overlay\Cargo.toml
C:\tmp\ph160-target\release\PuriPulyHeartOverlay.exe --check-startup-contract
certutil -hashfile C:\tmp\ph160-target\release\PuriPulyHeartOverlay.exe SHA256
```

Result: **235 passed, 1 ignored, 0 failed** (lib 129/1 ignored, renderer 94, state 12); release build passed. Startup identity: app `2.6.1`, contract `8`, execution `r2` version 1, native presentation retry `exclusive` version 1. Exercised release executable SHA-256: `697fec416a4a54fa94ac56eae6baf2462709938a9fc4aa91682e8e4ef8888f93`.

## Remaining limits

- No HMD was available; physical HMD exposure/freshness is unperformed, not passed.
- No paid Soniox request or private-audio upload was performed.
- The real desktop smoke used authorized synthetic content and validates the changed software/UI path, not diarization accuracy or provider latency.
