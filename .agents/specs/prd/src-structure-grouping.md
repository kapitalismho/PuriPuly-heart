---
id: PRD-SRC-STRUCTURE-GROUPING-001
status: reviewed
source: .agents/specs/prd/drafts/src-structure-grouping.source.r1.md
baseline_ref: refactor/flet-ui-migration@a6f08b10be0526eef839b732f19aa0117ab82819
integration_target: dev
document_review_verdict: ready
blocking_open_decisions: 0
---

# Outcome

PuriPuly Heart's `src/puripuly_heart` package gains domain-aligned folder
grouping that reflects the owner boundaries established by the G02–G19 cutover
sequence. Flat prefix-based file clusters in `core/`, `app/services/`,
`app/adapters/`, and the `app/` composition root are reorganized into cohesive
packages. Every existing import path continues to resolve through re-export
shims. No production behavior, persisted data, settings schema, secret key,
provider alias, prompt, locale string, output channel, or UI surface changes.

# Established Baseline

## Code baseline

- Analysis baseline is `refactor/flet-ui-migration@a6f08b10be0526eef839b732f19aa0117ab82819`
  with a clean working tree. Execution starts from integrated `dev` after G19
  integration approval; the executor re-verifies the file inventory at that SHA
  before beginning.
- `core/` contains approximately 25 loose `.py` files alongside 11 subpackages.
  Three domain clusters are identifiable by prefix: `local_stt_*` /
  `local_asr_*` / `local_gpu_*` / `local_qwen_*` (9 files, 70 importers),
  `openrouter_*` / `managed_openrouter_*` (6 files, 41 importers), and
  `discord_*` / `oauth_callback_page` (3 files, 9 importers).
- `core/runtime/` contains 26 flat files including names that duplicate loose
  `core/` files (`local_asr_provider_runtime.py`, `local_asr_provisioning.py`,
  `self_capture.py`), creating discoverability confusion.
- `app/services/` contains 61 flat files with at least six identifiable domain
  clusters (settings, managed account, overlay, local ASR, capture, provider).
- `app/` root contains 20 `wiring_*.py` files alongside `adapters/`, `ports/`,
  and `services/` subpackages.
- `app/adapters/` contains 27 flat files including 8 `peer_capture_*` and 6
  `self_capture_*` files.
- 181 architecture checks pass. 4,694 non-Discord tests pass with 0 failures.

## User-visible surfaces

- None. This contract changes no user-visible surface.

## Actual product entrypoints

- Production GUI: `python -m puripuly_heart.main run-gui`.
- All other established subcommands of `puripuly_heart.main`.
- Entrypoints are unchanged by this contract.

## Platform and environment

- Windows with the repository `.venv`, Python 3.12.
- Python application surface only. The Broker service and the Rust VR overlay
  are unaffected and require no verification for this contract.

## Compatibility baseline

- Every module that is moved remains importable at its pre-move path through
  re-export shims in the destination package `__init__.py`. No downstream
  consumer, test, architecture check, or packaging reference breaks.
- Settings serialization stays round-trippable with no new key and no new
  persisted field. No forward migration and no persisted-data backup are
  required.
- Secrets continue to load through SecretStore with unchanged key
  compatibility.
- Provider aliases, prompts, output routing, and channel separation remain
  unchanged.
- The installed package structure (`pyproject.toml`, `setup.cfg`, or
  equivalent) continues to discover all packages after the move.

# Scope

## Included

- Grouping the three identified `core/` prefix clusters into `core/discord/`,
  `core/openrouter/`, and `core/local_asr/` packages.
- Grouping the six identified `app/services/` domain clusters into
  `app/services/settings/`, `app/services/managed/`,
  `app/services/overlay/`, `app/services/local_asr/`,
  `app/services/capture/`, and `app/services/provider/` packages.
- Grouping the 20 `app/wiring_*.py` files into an `app/wiring/` package.
- Grouping the 14 capture adapter files into `app/adapters/peer_capture/`
  and `app/adapters/self_capture/` packages.
- Re-export shims in every destination `__init__.py` preserving pre-move
  import paths.
- Architecture check rule updates required by the moves, committed together
  with each move.

## Non-goals

### NG-001 — `core/runtime/` internal regrouping
The 26 flat files inside `core/runtime/` are not reorganized. The duplicate
name confusion with `core/` loose files is partially resolved by the
`core/local_asr/` move but no further `core/runtime/` restructuring is in
scope.

### NG-002 — `app/ports/` grouping
The 36 flat port files remain flat. Ports are interface definitions and flat
layout is conventional.

### NG-003 — `ui/` grouping
The `ui/` package already has adequate subpackage structure. No `ui/` file
moves.

### NG-004 — `config/` grouping
The `config/` package (16 entries, already containing `settings_vnext/`) is
not reorganized.

### NG-005 — Re-export shim removal
Shim removal is a separate future task. This contract creates shims and
verifies they resolve; it does not schedule or perform their deletion.

### NG-006 — Behavioral, schema, or surface changes
No production behavior, settings key, secret key, provider alias, prompt,
locale string, output channel, UI layout, or packaging metadata changes.

### NG-007 — `composition/`, `domain/`, `data/`, `providers/` changes
These packages are already clean and are not touched.

# Requirements

## REQ-001 — Domain-aligned package grouping
Each identified prefix cluster is moved into a dedicated subpackage whose name
reflects the domain boundary established by the owner cutover sequence. The
destination package contains an `__init__.py` and the moved modules. File
contents are unchanged except for intra-cluster relative imports that must be
adjusted to the new package location.

## REQ-002 — Import-path backward compatibility
Every pre-move module path remains importable after the move. The destination
package `__init__.py` re-exports every public name that the pre-move module
exposed. No consumer, test, or architecture check is required to change its
import statement as a condition of this contract, though consumers may
optionally adopt the new paths.

## REQ-003 — Architecture check alignment
Architecture check rules that reference moved module paths are updated in the
same commit as the move they affect. The total architecture check count does
not decrease. No check is weakened, silenced, or removed; only path references
change.

## REQ-004 — Commit-per-step discipline
Each of the 11 grouping steps is a separate commit. Each commit is
self-contained: the test suite, architecture checks, lint, and format checks
pass at every commit boundary. No commit depends on a later commit to reach a
green state.

## REQ-005 — Package discovery
The build and packaging configuration continues to discover all new
subpackages. `python -m compileall src` succeeds. An editable install
resolves every moved module at both its old and new path.

## REQ-006 — Blame preservation
All file moves use `git mv` (or equivalent rename-tracking mechanism) so that
`git log --follow` and `git blame` trace through the move.

# Protected Invariants

## Product invariants

### INV-P-001 — Zero behavioral delta
No production behavior changes. The application produces identical outputs,
identical settings round-trips, identical secret loading, identical output
channel routing, and identical locale rendering before and after the moves.

### INV-P-002 — Channel separation
Peer utterances never route to the VRChat chatbox. Self, peer, and system
outputs remain separate product channels.

### INV-P-003 — Settings and secret compatibility
Settings serialization remains round-trippable with no new key. SecretStore
key compatibility is unchanged. No forward migration is introduced.

## Durable architecture invariants

### INV-A-001 — Architecture check integrity
All 181 architecture checks pass at every commit boundary. The check count
does not decrease. No check is weakened or removed.

### INV-A-002 — Test suite integrity
The full test suite passes at every commit boundary with no new skips, no
new xfails, and no removed tests.

### INV-A-003 — Import resolution
Every pre-move import path resolves through re-export shims. No
`ImportError` or `ModuleNotFoundError` is introduced for any established
consumer.

### INV-A-004 — Packaging integrity
The installed package discovers all subpackages. `compileall` succeeds. No
module is orphaned or unreachable from the package root.

# Approved Decisions

- **D1** Execution starts from integrated `dev` on a separate short-lived
  branch, not on the evidence-pinned `refactor/flet-ui-migration` branch.
- **D2** Scope is Tier 1 (steps 1–3: `core/` clusters) plus Tier 2 (steps
  4–11: `app/services/` clusters, `app/wiring/`, `app/adapters/capture/`).
  Tier 3 candidates are excluded.
- **D3** Steps execute in ascending blast-radius order within each tier,
  one commit per step.
- **D4** Re-export shims preserve all pre-move import paths. Shim removal
  is deferred.
- **D5** `git mv` for all moves; architecture rules updated in the same
  commit as the affected move.
- **D6** Zero behavioral change is a hard constraint, not an aspiration.

# Open Product Decisions

`None`

# Acceptance Criteria

| AC | Verifies | Evidence class | Required environment | Pass condition |
|---|---|---|---|---|
| AC-001 | REQ-001, REQ-006, INV-P-001 | automated | Windows `.venv`; `git log --oneline`, `git diff --stat` per commit | Exactly 11 grouping commits exist, each using rename-tracked moves. No file content changes beyond intra-cluster relative import adjustments and `__init__.py` re-export shims. |
| AC-002 | REQ-002, INV-A-003 | automated | Windows `.venv`; `python -c "import <old_path>"` for every moved module | Every pre-move module path imports successfully and exposes the same public names as before the move. |
| AC-003 | REQ-003, INV-A-001 | automated | Windows `.venv`; `python -m pytest tests/architecture/` | All 181 architecture checks pass at every one of the 11 commit boundaries. The check count equals or exceeds 181 at each boundary. No check is weakened or removed. |
| AC-004 | REQ-004, INV-A-002 | automated | Windows `.venv`; `python -m pytest` at each commit boundary | The full test suite passes with 0 failures and 0 errors at every one of the 11 commit boundaries. No new skips or xfails are introduced. |
| AC-005 | REQ-004 | automated | Windows `.venv`; `ruff check src tests`, `black --check src tests`, `python -m compileall src` at each commit boundary | Lint, format, and compilation checks report no findings at every commit boundary. |
| AC-006 | REQ-005, INV-A-004 | automated + manual | Windows `.venv`; editable install, `python -m compileall src`, `pip show -f puripuly-heart` | All new subpackages are discovered by the packaging configuration. `compileall` succeeds. An editable install resolves every moved module at both old and new paths. |
| AC-007 | INV-P-001, INV-P-002, INV-P-003, NG-006 | automated + manual | Windows `.venv`; `python -m puripuly_heart.main run-gui` | The application starts, renders the dashboard at fixed 1136x850, loads settings, and shuts down cleanly. Settings round-trip produces a byte-identical file. No output channel routing changes. |
| AC-008 | NG-001 through NG-007 | manual diff review | Candidate diff against baseline | No file outside the 11 grouping steps is moved, renamed, or deleted. No `core/runtime/`, `app/ports/`, `ui/`, `config/`, `composition/`, `domain/`, `data/`, or `providers/` file is touched. No behavioral, schema, locale, or packaging metadata change appears in the diff. |

# Decision Authority

## Executor may decide

- reversible implementation details
- private types and internal APIs
- exact `__init__.py` re-export style (wildcard vs explicit names)
- file and helper placement within destination packages
- implementation sequence within the approved step order
- tests and diagnostics

## Independent review required

- durable boundary reliance
- architecture check rule changes beyond path updates
- any change that decreases the architecture check count
- packaging configuration changes
- terminal completion

## User decision required

- observable product behavior
- scope or non-goal change
- compatibility break
- shim removal timing
- required evidence weakening

# Completion Rule

Every acceptance criterion must be directly proven in its required environment
and evidence class. Automated tests alone cannot replace the manual diff
review of AC-008 or the production-composition startup evidence of AC-007.
Each of the 11 commits must independently satisfy AC-003, AC-004, and AC-005
before the next commit begins.
