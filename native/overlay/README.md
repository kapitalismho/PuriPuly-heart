# Native Overlay

Windows Rust runtime for the VR subtitle overlay.

## Ownership

- Rust implementation: `native/overlay/src`
- Rust tests: `native/overlay/tests`
- Python protocol and process integration:
  - `src/puripuly_heart/core/overlay`
  - `src/puripuly_heart/core/runtime/overlay.py`
  - `src/puripuly_heart/ui/desktop_overlay.py`

The shared bridge is version 16 with execution contract r2 and exclusive native
presentation retries. Authentication and readiness require both
`speaker_identity_presentation: {"version":3,"policy":"bounded_late_speaker_assignment"}`
and `caption_card_presentation: {"version":1,"policy":"source_bound_accumulation"}`.
Version 15 and earlier are incompatible. Bridge-16 peers advertising the older
version-2 immutable-style capability are also rejected; there is no compatibility
alias. The snapshot/card schema is unchanged.

Python resolves eligible speaker metadata before the first readable card choice.
Text and translation never wait for a speaker. An unresolved caption can receive
one identified style within 1.000 seconds of its first logically readable original
or translation; 500 ms is a preferred observation band, not a deliberate wait or
guarantee. An established speaker cannot change. Python sends only the resulting
`speaker_style`, not speaker identity, and owns the deadline/assignment policy. SELF
remains white. With speaker tracking disabled, the existing plain Peer presentation
uses gold `#FFD700` without consuming speaker slots. Nemotron is the speaker authority
when tracking is enabled, independently of ASR provider: peers use four stable
per-epoch styles (gold, cyan `#40DBFF`, coral `#FF7F5C`, blue `#7593FF`), or gray
`#B4B4B4` when unidentified, mixed, unavailable, handed off, or past the fourth key.
The canonical style colors live in `src/renderer/types.rs`. The same selected
color applies to source and translation and participates in native
render-cache invalidation and replay.

For an already-readable unresolved source, evidence availability is not snapshot
application. If eligible proof arrives before the cap but cannot be applied until
after it, the source remains unresolved across targets and retries. Late overflow
identities still update their gray disposition/divider in place without allocating
an additional color.

Card `id` and `occupant_key` remain stable across accumulation. `appearance_seq`
is the mutable latestness rank; both native and desktop apply its order rather than
keeping an existing occupant in a sticky slot. `card_revision` participates in
snapshot, frame and card-cache invalidation alongside text and style; unchanged
line/layout/glyph resources remain reusable. Desktop width retention follows card
identity, not its rank.

Python binds each admitted source once. Only a new source with a verified matching
display group, speaker epoch/key and compatible presentation can append to a
currently visible card. It moves that card newest and restarts its whole lifetime
at readable application. Source/context order remains unchanged; expired or evicted
accumulated history is not restored. Same-source completion, targets, retries,
metadata and replay neither append an utterance nor renew its lifetime. Caption
aggregation never changes OSC/chatbox payloads.
An already-readable unknown card that resolves changes style and revision in place;
it does not merge into an older same-speaker card or change position, text or TTL.
The existing returning-speaker append path remains available when identity is known
before the initial readable card choice. An earlier unresolved source may defer
later colors to preserve source-order allocation, but never delays their text.
This no-renew rule also covers finalized SELF cards: late translation does not add
the former four-second minimum or restart eight seconds. Active SELF preview and
slot protection remain separate from finalized-card lifetime.

The snapshot's `speaker_divider` flag asks native to draw a gray band between
the two caption slots. Python sets it only when both visible blocks are peer
captions frozen as palette overflow for different speakers in the same scope;
native draws it only while both slots are occupied. The band is centered in the
slot gap and on the caption center, 1320 × 12 surface px (8 px `#E6E6E6` fill,
2 px black outline, rounded ends), fixed in surface pixels, and is part of frame
identity and damage tracking.

Python owns caption expiry and send-time pruning. Native renders the blocks in
each accepted replacement snapshot without filtering them by semantic retirement
frontiers, exchanging validity leases, or autonomously expiring captions.
Equal or older snapshot revisions remain rejected.
Reconnect replay uses the original Python-held card deadlines, not a fresh lifetime.
The artificial one-second occupant replacement delay is removed; source-order,
active-source/SELF admission protection, bounded delivery and watchdogs remain.
Old captions can remain if removal cannot be delivered or replacement rendering
fails. Health reports presentation progress, not caption freshness. Runtime
generation retirement, bounded recovery, OFF/shutdown, and the 500 ms empty-frame
hide grace remain unchanged.

Changed, visible SELF active-source captions use the existing stream-phase
fresh-render episode without becoming semantic finals. Normal scene-update
rendering remains immediate. Same-target updates may advance trigger generation
but retain the episode deadline and completed count, including after exhaustion.
Semantic finalization changes phase; a changed final translation retains its
distinct final episode. Unchanged content does not establish a new trigger.

The production P05 bounds remain 100 ms retry cadence, a 500 ms scheduling
deadline, at most four stream and five final opportunities, and a separate
2 s readiness no-progress timeout. These are scheduling bounds, not display
latency guarantees. Retry metadata does not refresh caption age or reanchor a
current identity. Desktop presentation does not schedule native retries.

Python's semantic retirement compares `(publication_order, publication_index)`
within a scope and generation to reject late publications. It does not invalidate
a different entry that is still current. Native uses those coordinates and
frontiers only to reclaim spatial reanchor history, never to remove caption blocks.

Spatial history protects all currently drawable IDs and refreshes ordering
metadata on an already-seen ID without reanchoring it. Only absent IDs with known
scope, generation, order, and index can be reclaimed behind a matching frontier.
An ID returning after its history was reclaimed may reanchor; history is not an
unbounded record of every previously displayed ID. Source-only IDs that never
receive ordering metadata remain non-reclaimable under this policy. The existing
64-ID history limit remains: if unreclaimable IDs fill it, additional spatial
identities are not admitted until capacity is freed or the history is reset.

## Caption Sizing

Desktop and VR captions keep their selected preset or calibration font size for
ordinary text. Each primary and secondary region independently uses 100%, 90%,
or 80% of that size when approximate text width exceeds its reserved line budget,
with 10% overflow tolerance. Explicit newlines consume that same line budget.
Fitting is intentionally approximate; text that still exceeds the budget at 80%
keeps the existing clipping or ellipsis behavior. There is no pagination.

For a continuing caption, each region can shrink but does not grow again on
content updates or active-to-final promotion. A new occupant resets its fitting
history, and retired occupants are removed. This text-only adjustment does not
change full caption strings, window size, VR world placement, persisted settings,
or Python-owned caption lifetime.

The VR texture preserves its normal dimensions when sufficient and grows in
height when the selected calibration requires more room for two complete slots.
Slot anchors, reserved regions, and spacing continue to use the unshrunk
calibration size; a long caption does not move neighboring captions.

Run commands from the repository root. On Windows, use a short `--target-dir`
path if the checkout is deep enough to exceed MSBuild's tracking-file path limit.

## Verification

The cross-language active-source tests invoke `uv run --frozen python` to produce
snapshots through the actual Python Presenter before exercising native reducer
and retry ownership. Install `uv` and the locked Python test environment before
running the Rust suite.

The source-card cross-language fixture also runs the actual Python presenter:

```powershell
uv run --frozen python -m tests.helpers.overlay_source_cards
uv run --frozen python -m puripuly_heart.ui.desktop_overlay_card_repro --output-dir <new-empty-directory> --dwell-ms 2000
```

The second command opens a finite, synthetic Flet window covering accumulation,
replay, style changes, gray/divider cases, SELF and removal. Its JSON records describe
logical state; actual window captures or native texture readback are required for
rendering claims. Neither proves HMD readability, transparency compositing or photon
latency. The sanitized #217 records are `issue217_render_report.json`,
`issue217_desktop_report.json` and `issue217_evidence/`.

```powershell
cargo test --locked --manifest-path native/overlay/Cargo.toml
cargo test --locked --manifest-path native/overlay/Cargo.toml windows_graphics_style_changes_repaint_both_rows_and_replay_identically -- --nocapture
cargo build --manifest-path native/overlay/Cargo.toml --locked --release --bin PuriPulyHeartOverlay --target-dir target

New-Item -ItemType Directory -Force -Path build/overlay | Out-Null
Copy-Item target/release/PuriPulyHeartOverlay.exe build/overlay/PuriPulyHeartOverlay.exe -Force
Copy-Item third_party/openvr/win64/openvr_api.dll build/overlay/openvr_api.dll -Force

.\build\overlay\PuriPulyHeartOverlay.exe --check-startup-contract
```

The deterministic, graphics, and Python integration tiers run sequentially in
the single Windows-only `Overlay` job (`native-overlay`) in
`.github/workflows/pr-ci.yml`. The five real-process Python integration tests
run with `INTEGRATION=1` and zero skips enforced:

```powershell
$env:INTEGRATION = "1"
uv run --frozen pytest tests/app/test_desktop_overlay_runner.py::test_import_run_desktop_overlay_dispatch_is_provider_secret_and_stt_free tests/app/test_desktop_overlay_runner.py::test_import_preview_dispatch_is_provider_secret_and_stt_free tests/core/runtime/test_overlay_runtime.py::test_overlay_runtime_receives_real_subprocess_shutdown_ack_before_reader_cleanup tests/ui/test_flet_desktop_view_process_owner.py::test_windows_kill_on_close_job_reaps_assigned_real_process tests/ui/test_flet_desktop_view_process_owner.py::test_owner_reaps_real_process_and_pid_file_across_ten_cycles --junitxml=overlay-integration-results.xml
```


Shared protocol or startup changes also run:

```powershell
python -m pytest tests/core/test_overlay_protocol.py tests/core/test_overlay_manifest.py tests/app/test_desktop_overlay_runner.py

```

## Completion

Overlay behavior, protocol, or startup changes complete with:

1. Rust tests
2. Windows release build
3. Runtime assembly
4. Startup-contract verification
5. Python integration tests when the shared boundary changes

