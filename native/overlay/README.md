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
presentation retries (`native_presentation_retry: {"version":2,"ownership":"exclusive"}`).
Authentication and readiness also require
`speaker_identity_presentation: {"version":2,"policy":"immutable_first_readable_style"}`.
Version 15 and earlier are incompatible with the scene-coverage retry contract.
Python fixes a caption's `speaker_style` at first
readable publication and sends only that style, not speaker identity. SELF
remains white. Peers without diarization (unsupported providers or Soniox with
diarization OFF) use gold `#FFD700` without consuming or resetting speaker slots.
With diarization enabled, peers use four stable per-scope styles (gold, cyan
`#40DBFF`, coral `#FF7F5C`, blue `#7593FF`), or gray `#B4B4B4` when
unidentified, malformed, not ready, handed off, or past the fourth speaker in a scope.
The canonical style colors live in `src/renderer/types.rs`. The same selected
color applies to source and translation and participates in native
render-cache invalidation and replay.

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
Old captions can remain if removal cannot be delivered or replacement rendering
fails. Health reports presentation progress, not caption freshness. Runtime
generation retirement, bounded recovery, OFF/shutdown, and the 500 ms empty-frame
hide grace remain unchanged.

Changed, visible SELF scenes renew both trigger and episode generations under the
existing native retry owner. Ordinary scene-update rendering remains immediate.
The reserved `scene:self` target covers the currently accepted nonempty scene,
including a retained-caption patch and the surviving scene after SELF removal.
It is not a caption, does not change PEER admission, and cannot revive removed
content. The phase stays stream while an active SELF caption survives; otherwise
the scene uses final protection. Source closure alone does not end an active
caption's pending-translation protection.

Eligibility compares effective visible identity, text and style. Undisplayed
stable source, unchanged boundary metadata and hidden translations do not rearm
or erase intent. Coalescing protects the latest scene rather than replaying
intermediate scenes. SELF publication uses odd generation identities; a nonempty
send-time expiry change uses the following even identity, so it cannot collide
with the next presenter milestone. Fully empty output cancels scene protection
and retains the successful-empty-frame and Hide-grace contract.
Same-epoch clear retains internal generation high-water marks so a later scene
remains distinct even if the empty snapshot coalesces away. Fresh runtime epochs
reset and seed only currently valid content; they do not replay old retry ticks.

Each P05 window starts when native reconciles the accepted eligible scene, not
at Python publication time. The bounds remain 100 ms cadence, a 500 ms scheduling
wall, at most four stream and five final credited opportunities, and a separate
2 s due-work no-progress timeout. New episodes do not reset that timeout or
release unfinished GPU work. Fresh rendering and readiness checking precede
submission; delayed, superseded, cancelled and exhausted work is not a completed
opportunity. More milestones can increase total work per caption despite bounded
per-episode and simultaneous ownership. These limits do not guarantee physical
HMD display. Retry metadata does not refresh caption age or reanchor an identity.
Desktop presentation does not schedule native retries.

The bounded `native_retry` diagnostic stream correlates runtime, scene, trigger,
episode, phase and hashed target identity with active/due state, deadline and
disposition. Completed rows retain matching fresh-render, producer-ready and
submit-success witnesses. No caption text is logged. A scheduled row alone is
not execution evidence; an obsolete completion cannot satisfy a newer episode.

Scene-level nonempty eligibility includes enabled secondary-only text. A PEER
awaiting translation can therefore retain SELF-driven scene protection without
gaining a separate PEER retry episode or changing its caption policy.

OpenVR process-quit events retain their process identity. A quit notification
for another process cannot terminate this overlay or interrupt its empty-frame
Hide grace. Owned-process quit, global Quit and DriverRequestedQuit remain fatal;
ignored events still consume the existing bounded event-poll budget.

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

