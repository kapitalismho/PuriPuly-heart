---
artifact_kind: prd_contract
version: 1
artifact_ref: test-suite-refactor
status: reviewed
source_snapshot: .agents/specs/prd/drafts/test-suite-refactor.source.md
review_rounds: 1
---

# Test Suite Refactoring

## Product Intent

### Problem

The test suite grew to ~121K lines / ~300 files during a production refactoring cycle. The cycle is complete, leaving refactoring residue: duplicated helpers across 60+ files, 44 architecture tests that share 4 repeating patterns, 9 HIGH-severity wasteful tests, ~50 MEDIUM-severity over-specified or redundant tests, and lifecycle/geometry assertions coupled to internal naming rather than observable contracts. This inflates maintenance cost — a single private-field rename breaks 9 files; a legitimate design-token change breaks tests in two places.

### Desired Outcome

A leaner test suite (≤117K lines, targeting ~115K) that preserves all behavioral coverage, eliminates redundant and meaningless assertions, consolidates repeating patterns into parameterized tests and shared helpers, and decouples assertions from internal implementation naming where no production contract exists.

### Actors and Key Journeys

1. **Maintainer renaming an internal field** — expects zero test breakage unless a public contract changes.
2. **Maintainer changing a design token** — expects one-place update in the token source, not a second hardcoded literal in tests.
3. **Maintainer adding a new lifecycle owner** — expects a clear structural contract to satisfy, not 9 files of verbatim prose to copy.

## Scope

### In Scope

- Consolidate 44 architecture tests into ~15 files via parametrize and shared helpers.
- Delete or fix 9 HIGH-severity wasteful tests.
- Refactor ~50 MEDIUM-severity over-specified, duplicated, or bloated tests.
- Extract shared test helpers (AST walkers, path constants, DummyPage, repeated fixtures).
- Transition lifecycle_owner_snapshot assertions from verbatim prose to structural checks.
- Transition overlay geometry tests from hardcoded literals to contract-token references.
- Delete undocumented dialog magic-number pins.
- Delete redundant source-scan test file (test_controller_api_verification.py).
- Remove 15-line wiring/unit test duplicate.

### Out of Scope

#### NG-001 — Production source changes
- Statement: No production source code is modified. This is a test-only refactoring.
- Source refs: D-001

#### NG-002 — Behavioral coverage reduction
- Statement: No existing behavioral test coverage is removed. Only meaningless, redundant, or over-specified assertions are eliminated.
- Source refs: D-002

#### NG-003 — Architecture enforcement rule changes
- Statement: The rules enforced by architecture tests remain unchanged. Only their encoding (file structure, parametrize) changes.
- Source refs: D-003

#### NG-004 — Provider and config test restructuring
- Statement: tests/providers/ and tests/config/ are clean (0 HIGH, 1 MEDIUM). Only the single Deepgram fixture extraction is in scope; no structural changes.
- Source refs: none

#### NG-005 — LOW-severity item remediation
- Statement: ~40 LOW-severity items (private attribute assertions, hasattr guards, frozen-dataclass language-feature tests, runner.__name__ wiring guards) are accepted as-is per user approval. They are stylistic brittleness, not waste, and do not warrant refactoring effort.
- Source refs: none

## Requirements

### R-001 — Architecture test consolidation
- Type: quality
- Statement: The 29 ownership architecture tests sharing patterns O1–O4 are consolidated into parameterized tests. Standalone tests (guards, dependency engine, UI-boundary, flet-runtime, coordinator-retirement) remain unchanged. All 44 original enforcement rules continue to pass.
- Source refs: D-003
- Rationale: 29 files with near-identical logic create maintenance overhead without additional safety. Parametrize preserves every rule with one code path per pattern.

### R-002 — HIGH-severity waste elimination
- Type: quality
- Statement: Tests that provide no meaningful regression detection — including stdlib-only exercises, constructor echo, exact duplicates, self-asserting values, brittle source-text matching, and over-mocked stubs that verify only mock wiring — are deleted, fixed, or refactored. The broken test (test_unattended_runtime.py:334) is repaired to call the actual validation function. The over-mocked settings stub factory (test_settings_view_branches.py:173-285) is refactored toward a real or fake component fixture. The single 15-line wiring duplicate (test_wiring_providers.py:2176) is removed; remaining wiring tests are unchanged.
- Source refs: D-002
- Rationale: These tests provide near-zero regression detection value while adding noise and false confidence.

### R-003 — Shared helper extraction
- Type: quality
- Statement: Repeated test utilities (AST import-walker ×17 copies, path constants ×60 files, _method_source ×6, _assert_no_forbidden_imports ×3, DummyPage ×5, repeated fixtures ×3) are extracted into shared modules under tests/helpers/ or tests/conftest.py. All consuming tests import from the shared source.
- Source refs: none
- Rationale: Byte-near-identical copies across 60+ files make global changes error-prone and inflate line count.

### R-004 — Lifecycle snapshot structural assertions
- Type: boundary
- Statement: lifecycle_owner_snapshot test assertions verify structural contract only: required keys exist, resource_fields is a tuple of str, prose fields are non-empty str. Verbatim prose strings and private field names are not asserted.
- Source refs: D-004
- Rationale: No production code consumes the prose strings. They are executable documentation, not a public contract. Private field renames are safe refactors that should not break tests.

### R-005 — Overlay geometry token references
- Type: boundary
- Statement: Overlay renderer tests assert geometry values by referencing the defining constants in the overlay surface contract module, not by hardcoding the same literals independently.
- Source refs: D-005
- Rationale: Double-pinning (contract module + test literal) means a legitimate preset change breaks in two places and the test cannot distinguish intentional change from regression.

### R-006 — Undocumented magic-number pin removal
- Type: quality
- Statement: Dialog and settings widget tests that assert exact pixel values (border_radius, content_padding) with no corresponding design token and no documented rationale are deleted.
- Source refs: D-005
- Rationale: These pins guard no documented contract, couple tests to arbitrary implementation values, and create refactoring friction without regression value.

### R-007 — Redundant source-scan test removal
- Type: quality
- Statement: test_controller_api_verification.py is deleted. Its i18n-key assertions are fully subsumed by the unused-key scanner; its snackbar-absence assertion is subsumed by architecture boundary tests.
- Source refs: D-003
- Rationale: Both assertions are redundant with stronger, more general tests that already exist and continue to run.

### R-008 — MEDIUM-severity consolidation
- Type: quality
- Statement: Duplicated test scaffolding (test_app_branches.py FakeApp pairs, test_loopback_process_capture_ui.py ×7 setup, test_deepgram_session.py ×2 scaffolding) is extracted into shared fixtures. Duplicate behavioral assertions (dashboard subsumed tests, debug-preview locale refresh, capture-controls contract in two files) are merged or deleted. The loopback fixture extraction is classified here rather than R-002 because its action is extraction (preserving coverage) rather than deletion or repair.
- Source refs: D-002
- Rationale: Repeated scaffolding inflates line count and diverges silently; duplicate assertions add maintenance cost without coverage gain.

## Acceptance Contract

### AC-001 — Full suite passes
- Verifies: R-001, R-002, R-003, R-004, R-005, R-006, R-007, R-008
- Observable outcome: `pytest` exits 0 with no failures, no errors, and no skipped tests that were previously passing.

### AC-002 — Architecture rules preserved
- Verifies: R-001
- Observable outcome: Every enforcement rule that existed before consolidation continues to detect its target violation. Introducing a known violation (e.g., constructing an owner outside composition) still fails the corresponding test.

### AC-003 — Line count reduction
- Verifies: R-001, R-003, R-008
- Observable outcome: Total test line count decreases by at least 4,000 lines from the pre-refactoring baseline (~121K → ≤117K).

### AC-004 — No behavioral coverage loss
- Verifies: R-002, R-006, R-007, R-008
- Observable outcome: No production code mutation that was previously caught by a deleted test goes undetected. Verified by confirming that every deleted test either (a) tested no application code, (b) was fully subsumed by a remaining test, or (c) was replaced by an equivalent assertion.

### AC-005 — Rename resilience
- Verifies: R-004, R-005
- Observable outcome: Renaming a private field in any lifecycle owner (e.g., `_presenter` → `_overlay_presenter`) with no behavioral change produces zero test failures. Changing a design token in the overlay contract module requires updating only that module, not test literals.

## Approved Boundary Decisions

### D-001 — Test-only refactoring
- Decision: No production source files under src/ are modified.
- Rationale: Separates test cleanup from product behavior changes; eliminates risk of user-visible regression.
- Reconsider when: A test consolidation reveals a production bug that must be fixed to make tests pass.

### D-002 — Behavioral coverage invariant
- Decision: Every remaining production code path that had test coverage before retains equivalent coverage after.
- Rationale: The refactoring reduces waste, not safety.
- Reconsider when: A deleted test is found to have been the sole detector of a real regression class.

### D-003 — Architecture enforcement stability
- Decision: The set of enforced architectural rules is unchanged. Only encoding (file count, parametrize structure) changes.
- Rationale: Architecture tests encode product invariants (peer isolation, output routing, composition exclusivity). Their rules are not refactoring targets.
- Reconsider when: A rule is identified as enforcing a convention that no longer applies.

### D-004 — Lifecycle snapshot is not a public contract
- Decision: lifecycle_owner_snapshot prose strings and private field names are internal documentation, not a stable API. Tests assert structure only.
- Rationale: No production consumer reads the prose. No Protocol/ABC defines the shape. The one production call site reads a boolean key not present in the general shape.
- Reconsider when: A diagnostics UI or telemetry system begins consuming snapshot prose for user-visible output.

### D-005 — Design tokens are the single source of geometric truth
- Decision: Where a centralized token/constant defines a visual value, tests reference that constant. Where no token exists and no documented rationale supports a pin, the pin is deleted.
- Rationale: Double-pinning creates phantom coupling. Undocumented pins guard nothing identifiable.
- Reconsider when: A visual regression testing system (screenshot comparison) is adopted, making numeric pins redundant.

## Verification Intent

- Highest observable seam: Full pytest run (unit + architecture + integration-gated).
- Mandatory platform or manual evidence: Windows `.venv` pytest run; mutation spot-check on 2–3 deleted tests to confirm no coverage gap.
- Critical regression boundaries: Architecture ownership rules (composition exclusivity, peer isolation, output routing); overlay rendering correctness; i18n key parity.

## Assumptions and Deferrals

### A-001 — Refactoring cycle is complete
- Assumption: The production refactoring that caused test bloat is finished. No further large-scale production renames are imminent.
- Invalidated when: A new production refactoring wave begins that would immediately re-break consolidated tests.

### A-002 — Flet version remains pinned at 0.8.6.1
- Assumption: The pinned-Flet compatibility tests remain valid because the Flet version does not change during this refactoring.
- Invalidated when: Flet is upgraded, requiring a new compatibility test pass.

### DD-001 — Screenshot-based visual regression testing
- Deferred decision: Whether to adopt screenshot comparison for overlay/dialog geometry, replacing numeric pins entirely.
- Owner/stage: Product owner / post-refactoring evaluation.
- Resolve before: Next overlay visual redesign.

### DD-002 — Lifecycle snapshot Protocol formalization
- Deferred decision: Whether to promote lifecycle_owner_snapshot to a formal Protocol with documented stability guarantees.
- Owner/stage: Architecture owner / when a second consumer appears.
- Resolve before: Any system begins consuming snapshot output beyond the existing boolean read.

## Planning Defaults

### PD-001 — Consolidation order
- Default: Extract shared helpers first (R-003), then consolidate architecture tests (R-001), then delete/fix HIGH items (R-002), then MEDIUM consolidation (R-008), then boundary transitions (R-004, R-005, R-006), then redundant deletion (R-007).
- Override when: A dependency between items requires reordering.
- Preserves: R-001, R-003, AC-001

### PD-002 — Parametrize structure for architecture tests
- Default: One parametrize test per pattern family (O1, O2, O3, O4) with a data table of (owner_path, forbidden_imports, retired_names, expected_paths). Bespoke secondary tests in the same file remain as individual test functions.
- Override when: A pattern family has fewer than 3 members, making parametrize overhead exceed savings.
- Preserves: R-001, AC-002

### PD-003 — Shared helper module placement
- Default: tests/helpers/ast_sources.py for AST walkers, tests/helpers/paths.py for REPO_ROOT/SOURCE_ROOT. Fixtures go in tests/conftest.py.
- Override when: An existing helper module naturally absorbs the extraction.
- Preserves: R-003, AC-003

---

## Appendix A — Architecture Test Consolidation Detail

Governs: R-001, AC-002, PD-002

### A.1 Pattern O1 — "Controller no longer owns X" (3 files → 1 parametrized test)

**Files to delete after extraction:**
- `tests/architecture/test_managed_auth_ownership.py` (74 lines)
- `tests/architecture/test_managed_usage_ownership.py` (72 lines)
- `tests/architecture/test_translation_enable_ownership.py` (62 lines)

**New file:** `tests/architecture/test_controller_retirement_ownership.py`

**Shared helper** (move to `tests/architecture/conftest.py` or `_helpers.py`):
```python
def imports_of(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names.add(node.module or "")
    return names
```

**Parametrize data table:**

| id | owner_paths | forbidden_prefix | controller_path | retired_methods | retired_fields |
|----|-------------|-----------------|-----------------|-----------------|----------------|
| managed_auth | `app/services/managed_auth.py`, `app/wiring_managed_auth_factory.py`, `app/wiring_managed_account.py` | `puripuly_heart.ui` | `composition/application_runtime.py` | 10 methods: `_discord_auth_message_key`, `_discord_release_service_supports_transaction_auth`, `_start_discord_managed_auth_via_release_service`, `_get_managed_auth_runtime_adapter`, `_ensure_managed_auth_runtime`, `_create_managed_openrouter_release_service`, `_replace_managed_openrouter_release_service`, `_managed_openrouter_release_settings`, `_create_openrouter_pkce_client`, `_on_discord_managed_auth_callback_received` | 13 fields: `_managed_trial_pending_auth`, `_discord_managed_auth_in_progress`, `_discord_managed_auth_callback_received_hook`, `last_discord_managed_auth_referral_bonus_applied`, `telemetry_client`, `_managed_openrouter_release_service`, `_managed_auth_runtime_adapter`, `_managed_auth_owner`, `_managed_translation_runtime_adapter`, `_translation_enable_owner`, `_managed_usage_owner`, `_openrouter_pkce_flow_owner`, `_openrouter_pkce_application_owner` |
| managed_usage | `app/services/managed_usage.py` | `puripuly_heart.ui`, `puripuly_heart.config.settings` | `composition/application_runtime.py` | 19 methods: `_managed_identity_scope`, `_current_owned_referral_id`, `_talk_together_pass_cache_key`, `_clear_talk_together_pass_status_cache`, `_cached_talk_together_pass_status_for`, `_managed_key_card_visible_from_settings`, `_refresh_managed_status_best_effort`, `_schedule_owned_referral_id_status_refresh`, `_get_managed_status_refresh_owner`, `_clear_managed_trial_usage_metadata_cache`, `_sync_managed_trial_usage_metadata_scope`, `_schedule_managed_trial_usage_refresh`, `_refresh_managed_trial_usage_state`, `_refresh_managed_trial_usage_state_impl`, `_set_managed_usage_view_state`, `_managed_usage_state`, `_fetch_managed_usage_metadata`, `_managed_usage_auto_show_founder_letter`, `_managed_usage_warning_sink` | 5 fields: `_managed_status_refresh_owner`, `_managed_trial_usage_metadata`, `_managed_trial_usage_metadata_entitlement_ref`, `_talk_together_pass_status`, `_talk_together_pass_status_key` |
| translation_enable | `app/services/translation_enable.py`, `app/wiring_managed_auth_factory.py` | `puripuly_heart.ui` | `composition/application_runtime.py` | 10 methods: `_handle_managed_translation_enable`, `_show_founder_letter_dialog`, `_disable_translation_for_managed_exhaustion`, `_should_route_managed_trans_to_founder_letter`, `_record_translation_toggle_intent`, `_translation_toggle_intent_matches`, `_should_show_managed_auth_pending_before_prepare`, `_managed_auth_claim_guard_for_settings`, `_managed_china_auth_relevant_for_translation_enable`, `_show_qq_managed_auth_dialog` | 2 fields: `_translation_toggle_intent_enabled`, `_translation_toggle_generation` |

**Test structure:**
```python
@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_owner_does_not_import_ui_or_controller(case): ...

@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_controller_no_longer_owns_retired_state(case): ...
```

### A.2 Pattern O2 — "Constructed only by composition" (5 files → 1 parametrized test + bespoke tails)

**Files to consolidate (first test only):**
- `tests/architecture/test_local_asr_cpu_repair_ownership.py` (51 lines)
- `tests/architecture/test_local_asr_diagnostics_ownership.py`
- `tests/architecture/test_local_asr_gpu_provisioning_ownership.py`
- `tests/architecture/test_local_asr_provisioning_ownership.py`
- `tests/architecture/test_output_routing_ownership.py` (153 lines)

**New file:** `tests/architecture/test_construction_exclusivity.py`

**Parametrize data:**

| id | constructor_name | allowed_construction_file |
|----|-----------------|--------------------------|
| cpu_repair | `LocalASRCpuRepairOwner` | `src/puripuly_heart/app/wiring_composition.py` |
| diagnostics | `LocalASRDiagnosticsOwner` | `src/puripuly_heart/app/wiring_composition.py` |
| gpu_provisioning | `LocalASRGpuProvisioningOwner` | `src/puripuly_heart/app/wiring_composition.py` |
| provisioning | `LocalASRProvisioningOwner` | `src/puripuly_heart/app/wiring_composition.py` |
| output_runtime | `OutputRuntime` | `src/puripuly_heart/app/wiring_runtime_pipeline.py` |
| output_projection | `TranslationOutputProjectionOwner` | `src/puripuly_heart/app/wiring_runtime_pipeline.py` |

**Shared helper:**
```python
def find_constructions(class_name: str, root: Path) -> list[str]:
    results = []
    for f in sorted(root.rglob("*.py")):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == class_name:
                results.append(str(f.relative_to(root)))
    return results
```

**Bespoke tails remain in original files** (output_routing's channel/projection/settings assertions, cpu_repair's wiring-content assertions). Only the construction-count first test is parametrized.

### A.3 Pattern O3 — "UI adapter method is only an owner delegate" (6 files → 1 parametrized test)

**Files to delete after extraction:**
- `tests/architecture/test_manual_typing_composition_ownership.py` (40 lines)
- `tests/architecture/test_microphone_test_session_ownership.py`
- `tests/architecture/test_peer_process_capture_retry_ownership.py`
- `tests/architecture/test_provider_credential_verification_ownership.py`
- `tests/architecture/test_provider_secret_change_ownership.py`
- `tests/architecture/test_managed_openrouter_settings_ownership.py`

**New file:** `tests/architecture/test_ui_adapter_delegation.py`

**Shared helper:**
```python
def method_source(path: Path, class_name: str, method_name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == method_name:
                    return ast.get_source_segment(source, item)
    raise AssertionError(f"{class_name}.{method_name} not found in {path}")
```

**Parametrize data:**

| id | adapter_class | method | required_snippet | forbidden_snippets |
|----|--------------|--------|-----------------|-------------------|
| manual_typing | `UiInputRuntimeAdapter` | `submit_text` | `owner.submit_text(text, source="You")` | `hub`, `set_self_chatbox_typing_reason` |
| mic_test | `UiInputRuntimeAdapter` | `start_microphone_test` | `owner.` | `self._session`, `asyncio.create_task` |
| peer_retry | `UiInputRuntimeAdapter` | `retry_process_capture` | `owner.retry` | `asyncio`, `AppSettings` |
| credential_verify | `UiInputRuntimeAdapter` | `verify_provider_credential` | `owner.verify` | `httpx`, `aiohttp` |
| secret_change | `UiInputRuntimeAdapter` | `on_provider_secret_changed` | `owner.` | `SecretStore`, `openrouter` |
| managed_settings | `UiInputRuntimeAdapter` | `apply_managed_openrouter_settings` | `owner.` | `broker`, `webbrowser` |

### A.4 Pattern O4 — "Composition composes exactly one X" (7 files → 1 parametrized test)

**Files to delete after extraction:**
- `tests/architecture/test_github_star_prompt_settings_ownership.py` (19 lines)
- `tests/architecture/test_gpu_provider_recovery_ownership.py`
- `tests/architecture/test_vrc_mic_sync_ownership.py` (32 lines)
- `tests/architecture/test_vrchat_osc_presence_composition_ownership.py`
- `tests/architecture/test_sync_secret_store_adapter_ownership.py`
- `tests/architecture/test_overlay_session_transition_ownership.py`
- `tests/architecture/test_overlay_generation_start_ownership.py`

**New file:** `tests/architecture/test_composition_factory_exclusivity.py`

**Parametrize data:**

| id | factory_call | required_snippets | retired_names |
|----|-------------|-------------------|---------------|
| github_star_prompt | `compose_github_star_prompt_owner(` | `settings=settings`, `transaction_result_sink=` | `build_ui_prompt_clipboard_state_settings_path_patch`, `_persist_order24_state_mutation`, `_github_star_prompt_translation_connection_for`, `_github_star_prompt_has_managed_connection`, `_github_star_prompt_has_user_owned_cloud_connection` |
| vrc_mic_sync | `compose_vrc_mic_sync(` | `configure_vrc_mic=lambda *, enabled: (require_vrc_mic_sync().configure(enabled=enabled))` | `def _stop_vrc_mic_receiver(` |
| gpu_recovery | `compose_gpu_provider_recovery(` | (file-specific) | (file-specific) |
| osc_presence | `compose_vrchat_osc_presence(` | (file-specific) | (file-specific) |
| secret_store | `compose_sync_secret_store(` | (file-specific) | (file-specific) |
| overlay_session | `compose_overlay_session_transition(` | (file-specific) | (file-specific) |
| overlay_generation | `compose_overlay_generation_start(` | (file-specific) | (file-specific) |

**Test structure:**
```python
@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_composition_calls_factory_exactly_once(case):
    source = COMPOSITION_PATH.read_text(encoding="utf-8")
    assert source.count(case["factory_call"]) == 1

@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_composition_contains_required_snippets(case): ...

@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_composition_has_no_retired_names(case): ...
```

### A.5 Files that remain standalone (no changes)

- `test_dependency_boundaries.py` (1,759 lines) — layered import-rule engine
- `test_ui_boundary_architecture.py` (278 lines) — inspect-based contract conformance
- `test_flet_desktop_runtime_boundary.py` (142 lines) — live runtime behavior
- `test_translation_coordinator_retirement.py` (84 lines) — regex residue scan
- `test_lifecycle_task_guard.py` (539 lines) — task-creation inventory
- `test_raw_transcript_logging_guard.py` (74 lines)
- `test_raw_user_visible_error_guard.py` (196 lines)
- `test_self_capture_owner_contracts.py` / `test_peer_capture_owner_contracts.py` / `test_self_capture_source_ownership.py` — consolidate self/peer into channel-parametrize (optional, lower priority)

---

## Appendix B — HIGH-Severity File Actions

Governs: R-002, AC-004

### B.1 DELETE — `tests/core/test_file_logging.py` lines 17–52

Two tests (`test_rotating_file_handler_creates_log_file`, `test_rotating_handler_with_backup_count_zero`) exercise Python's `RotatingFileHandler`. No application code is called. Delete these two test functions. Keep the remaining 6 tests (L80–311) which test `SessionRuntimeLoggingService`.

### B.2 DELETE — `tests/core/test_clipboard_watcher.py` lines 25–28

`test_create_clipboard_watcher_returns_runtime` asserts `isinstance(watcher, WindowsClipboardWatcher)` on a freshly constructed object. No logic exercised. Delete this test function. Keep `test_cleanup_window_unregisters_window_class` (L31+).

### B.3 DELETE ONE — duplicate between `tests/core/test_context_memory.py:404` and `tests/core/test_orchestrator_pipeline.py:39`

Both named `test_translation_fixture_uses_local_context_when_peer_translation_is_off`. The context_memory version calls the resolver directly; the orchestrator version goes through `submit_text`. **Delete the orchestrator version** (L39–63) — the context_memory version tests the actual resolution logic; the orchestrator version only re-confirms the same fixture wiring.

### B.4 DELETE — `tests/core/test_messages_contracts.py` lines 76–90

`test_result_statuses_cover_settings_runtime_secret_and_compensation_flows` constructs `RuntimeApplyResult(message=message)` then asserts `runtime_result.message is message`. Pure constructor echo. Delete this test function. Keep L93–132 (field-name/type-alias assertions have marginal contract value).

### B.5 DELETE — `tests/ui/test_dashboard_view_branches.py` lines 585–606

Three tests fully subsumed by `test_dashboard_builds_4x3_friendly_shell_without_managed_trial_row` (L547–582):
- `test_dashboard_bottom_row_uses_trans_and_subtitles_labels` (L585–590) — subsumed by L548
- `test_dashboard_overlay_button_uses_subtitles_icon` (L593–596) — subsumed by L570
- `test_dashboard_peer_trans_overlay_buttons_use_default_on_color` (L599–606) — subsumed by L571–573

Delete all three.

### B.6 REFACTOR — `tests/ui/test_settings_view_branches.py` lines 173–285

`_make_llm_selection_view` builds a 110-line `SimpleNamespace` stub tree via `SettingsView.__new__`, replacing the entire real widget tree. Tests using it verify logic against fake controls.

**Action:** Replace with a fixture that constructs a real `SettingsView` with minimal real dependencies (the codebase already has `compose_test_ui_application_boundary` in `tests/helpers/ui_application.py`). If full construction is infeasible, extract the stub into a named fixture class in `tests/helpers/` so it is shared and its divergence from reality is visible in one place.

### B.7 DELETE — `tests/ui/test_flet_foundation.py` lines 351–358

`test_production_app_composes_foundation_adapter_runtime_and_application_callback` asserts `source.count("FletFoundationAdapter(") == 2` and exact substring presence via `inspect.getsource`. Pure source-text matching; breaks on any refactor with no behavior change. Delete. The composition is already verified by the architecture boundary tests and the foundation adapter's own behavioral tests.

### B.8 FIX — `tests/release_evidence/test_unattended_runtime.py` lines 334–342

`test_qwen_stale_schema_requires_observed_facts_for_pass` builds a `report` dict, sets `report["stages"]["stale_result"] = {...}`, then asserts `report[...]["rejected"] == 2` — the value it literally just assigned. Never calls `validate_report` (defined at `src/puripuly_heart/release_evidence/unattended_runtime.py:337`, already imported at test L28).

**Action:** Rewrite to construct the report input, call `validate_report(report)`, and assert on the validation result (pass/fail/rejected count).

### B.9 DELETE — `tests/app/test_wiring_providers.py` lines 2176–2181

Asserts `LocalQwenSherpaSTTBackend(model_dir=..., sample_rate_hz=8000)` raises `ValueError` matching `"16000"`. Identical assertion exists at `tests/providers/test_local_qwen_sherpa.py:619–621`. Delete the wiring copy.

---

## Appendix C — Shared Helper Extraction Detail

Governs: R-003, AC-003, PD-003

### C.1 New file: `tests/helpers/paths.py`

```python
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "src" / "puripuly_heart"
```

**Consumers to update (~60 files):** Every file containing `REPO_ROOT = Path(__file__).resolve().parents[2]` or `ROOT = Path(__file__).resolve().parents[2]` or `SOURCE_ROOT = ...`. Replace local definition with `from tests.helpers.paths import REPO_ROOT, SOURCE_ROOT`.

### C.2 New file: `tests/helpers/ast_sources.py`

```python
def imported_modules(path: Path) -> set[str]: ...
def imported_modules_from_source(source: str) -> set[str]: ...
def method_source(path: Path, class_name: str, method_name: str) -> str: ...
def method_source_unscoped(path: Path, method_name: str) -> str: ...
def call_name(node: ast.Call) -> str | None: ...
def assert_no_forbidden_imports(path: Path, forbidden_prefixes: tuple[str, ...]) -> None: ...
def find_constructions(class_name: str, root: Path) -> list[str]: ...
```

**Consumers to update:**

| Helper | Current copies (files) |
|--------|----------------------|
| `imported_modules` | `tests/core/output/test_router.py:33`, `tests/core/test_diagnostic_validator_contract.py:29`, `tests/core/test_observability_output_contracts.py:27`, `tests/app/test_service_ports_contracts.py:27`, `tests/architecture/test_logs_about_contract_boundary.py:27`, `test_app_shell_contract_boundary.py:31`, `test_local_asr_provider_runtime_contracts.py:13`, `test_desktop_overlay_surface_boundary.py:53`, `test_dashboard_contract_boundary.py:18`, `test_settings_contract_boundary.py:95`, `test_dependency_boundaries.py:661` |
| `imported_modules_from_source` | `tests/config/test_resolved_runtime_dtos.py:70`, `tests/config/test_runtime_resolution.py:63` |
| `_imports` (same logic, different name) | `tests/architecture/test_managed_usage_ownership.py:9`, `test_managed_auth_ownership.py:9`, `test_translation_enable_ownership.py:9`, `test_ui_boundary_architecture.py:68` |
| `method_source` | `test_manual_typing_composition_ownership.py:9`, `test_microphone_test_session_ownership.py:11`, `test_peer_process_capture_retry_ownership.py:12`, `test_provider_credential_verification_ownership.py:14` |
| `method_source_unscoped` | `test_vrc_mic_sync_ownership.py:10`, `test_local_asr_gpu_provisioning_ownership.py:14` |
| `assert_no_forbidden_imports` | `tests/core/output/test_router.py:44`, `tests/core/test_observability_output_contracts.py:38`, `tests/app/test_service_ports_contracts.py:38` |

### C.3 DummyPage consolidation

`tests/helpers/flet_page.py` already provides `DummyPage` (dialog-shaped: `opened`/`closed`/`show_dialog`/`pop_dialog`).

**Delete local redefinitions in (dialog-shaped, true duplicates):**
- `tests/ui/test_app_branches.py:79`
- `tests/ui/test_founder_letter_dialog.py:35`
- `tests/ui/test_discord_managed_auth_dialog.py:29`
- `tests/ui/test_language_modal.py:15`
- `tests/ui/test_github_star_snackbar.py:37`

**Keep local (window-shaped, different interface):**
- `tests/ui/test_title_bar.py:19` (has `window`/`update` — different contract)
- `tests/ui/test_qq_managed_auth_dialog.py:20` (verify shape; if dialog-shaped, consolidate)

### C.4 Repeated fixtures → `tests/conftest.py`

| Fixture | Current locations | Action |
|---------|-------------------|--------|
| `restore_locale_after_test` | `test_discord_managed_auth_dialog.py:21`, `test_founder_letter_dialog.py:29`, `test_qq_managed_auth_dialog.py:12` | Move to `tests/conftest.py` as autouse=False fixture |
| `reset_prompt_cache` | `tests/config/test_public_compatibility_surfaces.py:241`, `tests/config/test_prompt_loader.py:24` | Move to `tests/config/conftest.py` |
| `file_server` | `tests/scripts/test_install_local_stt_model.py:38`, `tests/core/test_local_stt_huggingface_xet_adapter.py:47`, `tests/core/test_local_stt_runtime_installer.py:150` | Bodies differ partially; extract common HTTP server scaffold to `tests/helpers/http_server.py`, keep per-test customization |

---

## Appendix D — Lifecycle Snapshot Transition Detail

Governs: R-004, AC-005, D-004

### D.1 Current state

9 test files assert `lifecycle_owner_snapshot()` output. The production method returns:
```python
{"owner": str, "resource_fields": tuple[str, ...], "stop_ingress": str, "shutdown_policy": str, "late_callback_rule": str}
```
Variant: `LocalASRProviderRuntimeOwner` adds `"provider_handles": dict`.

No production code consumes prose strings. Only `local_asr_provider_runtime.py:1088` reads `lifecycle["pending_handoff"]` (boolean).

### D.2 Files to modify

| File | Test function | Lines |
|------|--------------|-------|
| `tests/core/runtime/test_overlay_runtime.py` | `test_overlay_runtime_handle_exposes_lifecycle_inventory_and_policy` | 153–183 |
| `tests/core/runtime/test_output_runtime.py` | `test_output_runtime_exposes_lifecycle_inventory_and_policy` | 237–256 |
| `tests/core/runtime/test_receiver_runtime.py` | `test_receiver_runtimes_expose_lifecycle_inventory` | 41–56 |
| `tests/core/runtime/test_mic_test_runtime.py` | `test_mic_test_runtime_exposes_lifecycle_inventory_and_policy` | 25–40 |
| `tests/core/runtime/test_local_stt_download_runtime.py` | `test_local_stt_download_runtime_exposes_lifecycle_inventory_and_policy` | 12–26 |
| `tests/core/runtime/test_github_star_prompt_runtime.py` | `test_github_star_prompt_runtime_exposes_lifecycle_inventory` | 15–28 |
| `tests/core/runtime/test_runtime_logging_service.py` | `test_runtime_logging_service_exposes_lifecycle_inventory` | 175–187 |
| `tests/core/runtime/test_local_asr_provisioning.py` | `test_owner_lifecycle_inventory_names_all_provisioning_resources` | 587–605 |
| `tests/core/runtime/test_local_asr_provider_runtime.py` | `test_owner_lifecycle_inventory_names_provider_and_gpu_resources` | 1018–1029 |

### D.3 Replacement assertion pattern

Replace all verbatim prose and exact-tuple assertions with:

```python
def assert_lifecycle_structure(snapshot: dict[str, object]) -> None:
    assert isinstance(snapshot["owner"], str) and snapshot["owner"]
    assert isinstance(snapshot["resource_fields"], tuple)
    assert all(isinstance(f, str) and f for f in snapshot["resource_fields"])
    assert len(snapshot["resource_fields"]) > 0
    assert isinstance(snapshot["stop_ingress"], str) and snapshot["stop_ingress"]
    assert isinstance(snapshot["shutdown_policy"], str) and snapshot["shutdown_policy"]
    assert isinstance(snapshot["late_callback_rule"], str) and snapshot["late_callback_rule"]
```

Place in `tests/helpers/lifecycle.py`. Each test calls `assert_lifecycle_structure(snapshot)` plus any owner-specific structural checks (e.g., `len(resource_fields) >= 3`, `"provider_handles" in snapshot` for ASR).

### D.4 Assertions to REMOVE (examples)

| File | Remove | Reason |
|------|--------|--------|
| test_overlay_runtime.py:178 | `== "broadcast shutdown and reject new overlay commands"` | Verbatim prose |
| test_overlay_runtime.py:160-177 | `"_presenter" in ...`, `"OverlayPresenter._expiration_tasks" in ...` (×15) | Private field names |
| test_mic_test_runtime.py:31 | `== ("_session_task", "_source", "_pending_frame_task", ...)` | Exact tuple with private names |
| test_receiver_runtime.py:50 | `== "stop receiver before runtime shutdown"` | Verbatim prose |
| test_runtime_logging_service.py:180 | `== "flush final shutdown summary, then close handlers"` | Verbatim prose |
| test_local_asr_provisioning.py:592 | `== ("_cpu_install_runtime", "_gpu_install_runtime", ...)` | Exact tuple |

### D.5 Assertions to KEEP (structural)

| File | Keep | Reason |
|------|------|--------|
| test_overlay_runtime.py:159 | `snapshot["resource_fields"] == OverlayRuntimeHandle.resource_fields` | Identity with class attr (structural) |
| test_local_asr_provider_runtime.py:1023 | `inventory["provider_handles"].keys() == {"self", "peer"}` | Structural key set |
| test_output_runtime.py:244-249 | `"overlay_event_adapter" in ...` (descriptive names, not private) | Borderline — keep if name is part of owner's public vocabulary |

---

## Appendix E — Overlay Geometry Token Transition Detail

Governs: R-005, R-006, AC-005, D-005

### E.1 `tests/ui/test_desktop_overlay_renderer.py` — replace literals with contract imports

**Add import:**
```python
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    _DESKTOP_CAPTION_GOLD,
    _DESKTOP_CAPTION_WHITE,
    _DESKTOP_CAPTION_LINE_HEIGHT,
    _DESKTOP_CAPTION_TEXT_STACK_ALIGNMENT_Y,
    _DESKTOP_CAPTION_PRIMARY_REGION_ALIGNMENT_Y,
    _DESKTOP_CAPTION_CONTACT_SHADOW_COLOR,
    _DESKTOP_CAPTION_CONTACT_SHADOW_OFFSET,
    _DESKTOP_CAPTION_CONTACT_SHADOW_BLUR,
    _DESKTOP_CAPTION_AMBIENT_SHADOW_COLOR,
    _DESKTOP_CAPTION_AMBIENT_SHADOW_OFFSET,
    _DESKTOP_CAPTION_AMBIENT_SHADOW_BLUR,
    _DESKTOP_CAPTION_MIN_DYNAMIC_CARD_WIDTH,
    _DESKTOP_CAPTION_MAX_VISIBLE_SLOTS,
    _DESKTOP_CAPTION_MAX_VISIBLE_LINES,
    _DESKTOP_CAPTION_SIZE_PRESETS,
    _DESKTOP_CAPTION_TRANSPARENT,
    _DESKTOP_PREVIEW_BACKGROUND_ALPHA_PRESETS,
)
```

**Replacement table (representative — apply to all ~50 assertions):**

| Line | Current | Replace with |
|------|---------|-------------|
| 89, 91, 110, 115, 284, 287 | `== "#FFFFFF"` | `== _DESKTOP_CAPTION_WHITE` |
| 96, 98, 99, 105, 291, 293, 296, 427 | `== "#FFD700"` | `== _DESKTOP_CAPTION_GOLD` |
| 394 | `plan.primary_font_size == 41` | `== _DESKTOP_CAPTION_SIZE_PRESETS["medium"].primary_font_size` |
| 395 | `plan.secondary_font_size == 25` | `== _DESKTOP_CAPTION_SIZE_PRESETS["medium"].secondary_font_size` |
| 398 | `plan.padding_horizontal == 22` | `== _DESKTOP_CAPTION_SIZE_PRESETS["medium"].padding_horizontal` |
| 399 | `plan.padding_vertical == 10` | `== _DESKTOP_CAPTION_SIZE_PRESETS["medium"].padding_vertical` |
| 401, 405, 413, 470, 890 | `border_radius == 16` | `== _DESKTOP_CAPTION_SIZE_PRESETS["medium"].border_radius` |
| 430 | `style.height == pytest.approx(1.24)` | `== pytest.approx(_DESKTOP_CAPTION_LINE_HEIGHT)` |
| 609 | `contact_shadow.color == "#C0000000"` | `== _DESKTOP_CAPTION_CONTACT_SHADOW_COLOR` |
| 610 | `contact_shadow.offset == (0, 1)` | `== _DESKTOP_CAPTION_CONTACT_SHADOW_OFFSET` |
| 611 | `blur_radius == pytest.approx(1.0)` | `== pytest.approx(_DESKTOP_CAPTION_CONTACT_SHADOW_BLUR)` |
| 612 | `ambient_shadow.color == "#66000000"` | `== _DESKTOP_CAPTION_AMBIENT_SHADOW_COLOR` |
| 613 | `ambient_shadow.offset == (0, 0)` | `== _DESKTOP_CAPTION_AMBIENT_SHADOW_OFFSET` |
| 614 | `blur_radius == pytest.approx(3.0)` | `== pytest.approx(_DESKTOP_CAPTION_AMBIENT_SHADOW_BLUR)` |
| 725, 808 | `alignment.y == pytest.approx(-0.08)` | `== pytest.approx(_DESKTOP_CAPTION_TEXT_STACK_ALIGNMENT_Y)` |
| 735, 811 | `alignment.y == pytest.approx(-0.5)` | `== pytest.approx(_DESKTOP_CAPTION_PRIMARY_REGION_ALIGNMENT_Y)` |
| 457 | `card_width == pytest.approx(320.0)` | `== pytest.approx(_DESKTOP_CAPTION_MIN_DYNAMIC_CARD_WIDTH)` |
| 652 | `plan.max_visible_slots == 2` | `== _DESKTOP_CAPTION_MAX_VISIBLE_SLOTS` |
| 968 | `sum(...) == 6` | `== _DESKTOP_CAPTION_MAX_VISIBLE_LINES` |
| 1001 | `plan.primary_font_size == 35` | `== _DESKTOP_CAPTION_SIZE_PRESETS["small"].primary_font_size` |
| 1002 | `plan.secondary_font_size == 21` | `== _DESKTOP_CAPTION_SIZE_PRESETS["small"].secondary_font_size` |
| 1621 | `== (0.35, 0.5, 0.6, 0.8)` | `== _DESKTOP_PREVIEW_BACKGROUND_ALPHA_PRESETS` |
| 2028–2029 | `width == 1600`, `height == 400` | `== _DESKTOP_CAPTION_SIZE_PRESETS["large"].window_width/height` |
| 2865–2866 | `width == 1344`, `height == 336` | `== _DESKTOP_CAPTION_SIZE_PRESETS["medium"].window_width/height` |

**Keep as-is (derived values, no constant):**
- L400: `plan.text_width == 1300` (derived: 1344 − 2×22)
- L360–361: `background_alpha == 0.6`, `background_color == "#99000000"` (edit-mode default, not in contract)
- L397: `background_color == "#61000000"` (derived from alpha 0.38)
- L469, 889, 922: `bgcolor == "#80000000"` (derived from alpha 0.5)

### E.2 `tests/ui/test_baseline_control_geometry.py` — already correct

This file imports `TEXT_BUTTON_PADDING` and `PROMPT_FIELD_CONTENT_PADDING` from source and asserts both the token value and its propagation to widgets. **No changes needed.** This is the reference pattern.

### E.3 `tests/ui/test_custom_vocabulary_tag_editor.py` — replace with source constants

**Add import:**
```python
from puripuly_heart.ui.components.settings.custom_vocabulary_tag_editor import (
    _CHIP_TEXT_SIZE,
    _CHIP_HORIZONTAL_PADDING,
    _CHIP_VERTICAL_PADDING,
    _INPUT_FIELD_RADIUS,
)
```

| Line | Current | Replace with |
|------|---------|-------------|
| 68 | `term_text.size == 22` | `== _CHIP_TEXT_SIZE` |
| 69–70 | `padding.left == 20`, `padding.right == 20` | `== _CHIP_HORIZONTAL_PADDING` |
| 71–72 | `padding.top == 14`, `padding.bottom == 14` | `== _CHIP_VERTICAL_PADDING` |
| 109 | `border_radius == 12` | `== _INPUT_FIELD_RADIUS` |

**Delete (no constant, no rationale):**
- L108: `_input_field.text_size == 28` — no named constant in source

### E.4 `tests/ui/test_discord_managed_auth_dialog.py` — partial token, partial delete

**Replace with source constants:**

| Line | Current | Replace with |
|------|---------|-------------|
| 154 | `modal_content.width == 720` | `== warm_document_dialog.DIALOG_WIDTH` |
| 163 | `body_text.size == 24` | `== warm_document_dialog.BODY_TEXT_SIZE` |
| 173 | `[_button_text_size(...)] == [26, 26]` | `== [warm_document_dialog.BUTTON_TEXT_SIZE] * 2` |

**Delete (inline literals, no token, no documented rationale):**
- L244: `field.text_size == 22`
- L246–249: `content_padding.left == 16`, `.right == 16`, `.top == 20`, `.bottom == 20`
- L251: `field.border_radius == 14`
- L264: `body_column.spacing == 44`
- L266: `action_spacer.height == 24`
- L182: `animation_duration == [0, 0]`

### E.5 `tests/ui/test_settings_view_branches.py` — delete undocumented pins

**Delete all bare magic-number assertions** (no token exists for any):
- L534–536: `_setting_action_text_size(...)` threshold assertions
- L1535: `_local_llm_extra_body_helper.size == 15`
- L1552: `field.text_size == 24`
- L1554: `field.label_style.size == 18`
- L2242: `_gpu_device_card.height == 228`
- L2246, 4501, 4648, 6225, 6306–6309: `content.size == 28` / `== 22`
- L7410–7412: `body_region.padding == ft.Padding.only(left=16, top=16, right=16)`
- L7508: `_subtab_text_size(button) == 20`
- L7540: `subtab_bar.border_radius == 24`

**Keep:**
- L1551: `field.border_radius == api_field.border_radius` (relative assertion, no magic number)
- L7497: `subtab_bar.border_radius is None` (behavioral: conditional logic)
- L7503: `subtab_bar.padding is None` (behavioral)

---

## Appendix F — MEDIUM-Severity Consolidation Detail

Governs: R-008, AC-004

### F.1 `tests/ui/test_loopback_process_capture_ui.py` — extract dashboard warning fixture

Lines 600–1033 contain 7 tests each repeating ~15–20 lines of identical setup:
```python
view = DashboardView.__new__(DashboardView)
view._process_capture_warning_text = ...
view._process_capture_warning_visible = ...
view._peer_process_capture_warning_text = ...
view._peer_process_capture_warning_visible = ...
view.set_display_text = lambda *a: None
view._sync_overlay_peer_buttons = lambda: None
```

**Action:** Extract to a `@pytest.fixture` named `dashboard_warning_view` at module level. Each test receives the fixture and only sets the fields specific to its scenario.

### F.2 `tests/providers/test_deepgram_session.py` — extract fake module scaffold

Lines 319–394 and 397–472 each define an identical 6-class fake `deepgram` module (`FakeEventType`, `FakeControlMessage`, `FakeConnection`, `FakeV1`, `FakeListen`, `FakeClient`) + 6 `monkeypatch.setitem(sys.modules, ...)` calls.

**Action:** Extract to a module-level fixture:
```python
@pytest.fixture
def fake_deepgram_modules(monkeypatch):
    # ... class definitions ...
    for name, mod in [("deepgram", fake_dg), ("deepgram.v1", fake_v1), ...]:
        monkeypatch.setitem(sys.modules, name, mod)
    return fake_dg
```

Each test receives `fake_deepgram_modules` and only customizes `_make_session(...)` args.

### F.3 `tests/ui/test_app_branches.py` lines 1498–1638 — extract FakeApp/FakeController

Two tests (`test_main_gui_routes_update_check_through_app_log_helper`, `test_main_gui_forwards_debug_ui_preview_flag`) each define a ~60-line `FakeController` + `FakeApp` pair differing only by one flag.

**Action:** Extract to a module-level helper function:
```python
def _make_fake_app_and_controller(*, debug_ui_preview: bool = False):
    ...
    return fake_app, fake_controller
```

### F.4 `tests/ui/test_debug_preview_panel.py` lines 162–275 — merge duplicate locale assertions

`test_debug_preview_panel_apply_locale_refreshes_labels` (L162–202) and the second half of `test_debug_preview_panel_uses_flet_086_text_button_content_api` (L245–275) assert the same per-button locale label refresh.

**Action:** Delete the locale-refresh assertions from the content-API test (L245–275). The dedicated locale test already covers this.

### F.5 `tests/ui/test_dashboard_capture_controls.py` + `test_dashboard_surface_contract.py` — consolidate

Both assert `DashboardSurfaceSlots.from_capture_provider` maps `self_capture`/`peer_capture`/`overlay` correctly.

**Action:** Keep the assertion in `test_dashboard_surface_contract.py` (broader contract test). Delete the duplicate from `test_dashboard_capture_controls.py:45–57` if it adds no additional provider-shape coverage.

### F.6 `tests/ui/test_controller_api_verification.py` — DELETE entire file (26 lines)

- L14–16: `show_action_snackbar` absence — subsumed by architecture boundary tests policing `app.py` surface.
- L20–26: obsolete i18n keys — fully subsumed by `test_i18n_key_usage.py:522` unused-key scanner.

---

## Appendix G — Source-Scan Tests Disposition

Governs: R-007

| File | Lines | Disposition | Rationale |
|------|-------|-------------|-----------|
| `test_controller_api_verification.py` | 26 | **DELETE** | Redundant (see F.6) |
| `test_debug_preview_panel.py:288–304` | 17 | KEEP | Guards AGENTS.md product invariant (preview isolation) beyond AST import checks; catches stdlib `webbrowser`, string-level `SecretStore` |
| `test_flet_pinned_compatibility.py:59–72` | 14 | KEEP | Pinned-SDK compat; `page.open()/close()` removed in Flet 0.8.6; no runtime test path without full Flet page |
| `test_flet_086_interaction_equivalence.py:57–64` | 8 | KEEP | Exhaustive negative-existence for `data == "true"` string comparison; silent failure mode |
| `test_flet_086_interaction_equivalence.py:120–134` | 15 | KEEP | Static layout lint; TextField without width is invisible at paint time; no unit-level detection |
| `test_i18n_key_usage.py:522–543` | 22 | KEEP | Canonical dead-resource lint; negative-existence property |
