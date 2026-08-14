# Execution Record — Removal of ASR finalization latency logic (reverts complete)

- Target branch: `Cloud-ASR-Finalization`
- Revert baseline: `dev` (branch start `31008444`)
- Purpose: restore the original behavior for logic that was introduced for accuracy but only increased latency, complexity, and malfunction risk.
- Status: **Reverts complete (2026-08-13)** — only final review and merge/push approval remain

## Background

The `Cloud-ASR-Finalization` branch was created to ensure cloud ASR (speech recognition) finalization results are not lost, duplicated, or reordered. Investigation found that several mechanisms introduced for accuracy only **increased latency, complexity, and malfunction risk**. The four policies below have been reverted in code; this document records that decision and its execution.

## Withdrawn policies (reverts complete)

1. **Deepgram fence wait** → `dab0d7a7`
   Serialization logic that sent Finalize and waited up to 1 second for the `from_finalize` response (fence). Per Deepgram's official docs, this response is not guaranteed ("may not arrive if the buffer has little audio"). On short sentences it killed and reconnected sessions even in normal operation, adding up to 1 second of latency per sentence.
   → Restored to fire-and-forget: results are emitted immediately on `is_final`/`speech_final` arrival.

2. **Ordered flush gate** → `0700e1a7`
   A gate that only released final results in speech-ended order. If one sentence stalled, all later captions were blocked (head-of-line blocking).
   → Restored to a single FIFO queue: finals are emitted immediately in arrival order.

3. **20-second timeout + session recovery** → `0700e1a7`
   Active background timer and session reconnect. Previously (`dev`) stale results were quietly dropped "when another event arrived" (lazy), with no session recovery.
   → Restored to lazy stale-drop: active timers, loss declarations, and reconnect triggers removed.

4. **Soniox trailing silence** → `4fef2f81`
   Logic that sent a `trailing_silence_ms` parameter on the finalize message and injected silence bytes. Per Soniox's official docs, the current finalize only accepts `{"type":"finalize"}`; the parameter is absent from the current API (legacy). Silence injection was also unnecessary because VAD hangover (500–1000 ms) already sends real silence (Soniox recommends "~200 ms silence before finalize", already satisfied).
   → Restored to sending only `{"type":"finalize"}`; silence injection removed. The config surface stays inert (option B).

Regression test: `e51b7e3b` — added a test proving no head-of-line blocking across sessions.

## Kept policies (not reverted)

- **Qwen item_id correlation** (`a67e2756`) — async matching that improves accuracy without latency.
- **Soniox 1:1 token mapping** (`4b330b26`) — prevents token loss.
- **Soft-pause VAD boundary** (`ab5640c0`) — `SpeechBoundaryReason`, `boundary_wait_ms`, and the `reason` parameter on `on_speech_end`.
- **Qwen/Deepgram padding removal** (`51a44e32`) — confirmed correct by investigation.
- **`self_capture.py` `_release_plan` change** (cloud toggle-off → abort, `42b31d70`).

## Principles

- `dev` (`git show dev:<file>`) is the revert baseline, but the "kept" items above are re-applied.
- Reverts are manual restores, not `git revert` (kept items are interleaved in related commits).
- The Soniox `trailing_silence_ms` **config surface is not touched (option B)** — only behavior is removed; the setting field stays inert.

---

## Unit 1 — Revert controller finalization handling — complete (`0700e1a7`)

### Context

The controller (`core/stt/controller.py`) receives VAD events (speech start/chunk/end), forwards them to provider sessions, and emits the provider's final/interim results externally. This branch replaced finalization tracking here with new structures (`_PendingBoundary`/`_SessionContext`/`_accepted_boundaries`), introducing ② (ordered flush) and ③ (20 s timeout + recovery). Both behaviors were entangled with those new structures, so the revert centered on **restoring finalization tracking to `dev`'s simple FIFO design**.

### Changes

- Finals emitted immediately in arrival order (ordered-flush queue gate removed).
- Active 20-second timer, loss declarations, and session-reconnect triggers removed → restored lazy dropping of stale finals when another event arrives.
- `_PendingBoundary`/`_SessionContext`/`_accepted_boundaries`/boundary timeout and fencing methods removed.
- Soft-pause `reason` passthrough and session lifecycle semantics (generation invalidation, bridging, reconnects) preserved.

### Completion criteria

- [x] Controller-related tests pass.
- [x] Reverted structures/methods are no longer referenced in code.
- [x] Finals emit immediately in arrival order; unresolved speech does not block later results (proven by `e51b7e3b` regression test).
- [x] `reason` reaches `on_speech_end` (proven by tests).

---

## Unit 2 — Restore Deepgram fire-and-forget finalize — complete (`dab0d7a7`)

### Context

`providers/stt/deepgram.py` sends audio over a websocket via the official SDK and sends a `Finalize` control message at speech end to request finalization. This branch "serialized" it: after Finalize, it waited up to 1 second for the `from_finalize` fence before continuing, pausing audio transmission while waiting. The problem was the non-guaranteed fence response and the resulting per-sentence latency of up to 1 second.

### Changes

- Finalize message is sent and processing moves on immediately (synchronous waits, timeouts, and fence-based session-failure handling removed).
- Provider `is_final` or `speech_final` results are emitted as final events immediately.
- `_PendingFinalize`, `_all_finalized`, `_fail_finalize_session`, `_segment_buffer`, and `from_finalize`-related state/methods removed.
- `on_speech_end`'s `reason` signature and padding-free body preserved.
- Connection errors/closes propagate to the event stream as exceptions as in `dev`, so the upper layer can detect session failure and reconnect.

### Completion criteria

- [x] Deepgram session tests pass.
- [x] Fence wait and serialization state removed from code.
- [x] Returns immediately after Finalize; results emitted on arrival (proven by tests).

---

## Unit 3 — Remove Soniox trailing-silence behavior — complete (`4fef2f81`)

### Context

`providers/stt/soniox.py` sends a `finalize` control message over a raw websocket for finalization. On this branch `on_speech_end` (a) injected silence bytes and (b) sent a `trailing_silence_ms` parameter on finalize. Investigation showed (b) is a legacy field absent from the current Soniox API, and (a) was unnecessary because VAD hangover already sends 500–1000 ms of real silence (Soniox recommends "~200 ms silence before finalize", already satisfied).

### Changes

- Finalize control message contains only the `type` field (no `trailing_silence_ms`).
- Silence byte synthesis/sending at speech end removed.
- `<fin>` marker flush and 1:1 token mapping preserved.
- `on_speech_end`'s `reason` signature preserved.
- Config surface unchanged (option B): `trailing_silence_ms` setting fields, wiring, and migrations stay as-is; the value is ignored (inert).

### Completion criteria

- [x] Soniox tests pass.
- [x] Finalize payload contains only `{"type":"finalize"}` (proven by tests).
- [x] `<fin>` flush and token-mapping behavior unchanged (proven by tests).
- [x] No changes to the config surface (wiring/migrations).

---

## Remaining steps

- [ ] Final review of `git diff dev...HEAD` (opencode subagent) → no merge/push before approval.

## Sequence

```
Unit1 → Unit2 → Unit3 → regression test (e51b7e3b) → complete
Remaining: final overall review → approval → merge/push
```
