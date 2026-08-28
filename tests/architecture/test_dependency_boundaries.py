from __future__ import annotations

import ast
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from tests.helpers.paths import REPO_ROOT

SOURCE_PACKAGE_ROOT = REPO_ROOT / "src" / "puripuly_heart"
PACKAGE_NAME = "puripuly_heart"

SCHEMA_VALUES = "schema values"
MIGRATION_SERIALIZATION = "migration/serialization"
RESOLVED_DTOS = "resolved DTOs"
RUNTIME_RESOLUTION = "runtime resolution"
DOMAIN = "domain"
RUNTIME_OWNERS = "runtime owners"
ORCHESTRATOR = "orchestrator"
OVERLAY_CORE = "overlay core"
APP_SERVICES = "app services"
APP_COMPOSITION = "app composition"
SERVICE_PORTS = "service ports"
ADAPTERS = "adapters"
SETTINGS_PERSISTENCE_ADAPTERS = "settings persistence adapters"
OUTPUT_MESSAGE_OBSERVABILITY_PORTS = "output/message/observability ports"
PROVIDERS = "providers"
UI_ADAPTERS_RENDERERS = "UI adapters/renderers"

REQUIRED_LAYER_VOCABULARY = (
    SCHEMA_VALUES,
    MIGRATION_SERIALIZATION,
    RESOLVED_DTOS,
    RUNTIME_RESOLUTION,
    DOMAIN,
    RUNTIME_OWNERS,
    ORCHESTRATOR,
    OVERLAY_CORE,
    APP_SERVICES,
    APP_COMPOSITION,
    SERVICE_PORTS,
    ADAPTERS,
    SETTINGS_PERSISTENCE_ADAPTERS,
    OUTPUT_MESSAGE_OBSERVABILITY_PORTS,
    PROVIDERS,
    UI_ADAPTERS_RENDERERS,
)


@dataclass(frozen=True, order=True, slots=True)
class ImportViolation:
    rule_id: str
    importer: str
    imported: str
    importer_layer: str
    imported_layer: str
    reason: str


@dataclass(frozen=True, order=True, slots=True)
class SettingsRuntimeConfinementViolation:
    category: str
    path: str
    symbol: str
    rationale: str


@dataclass(frozen=True, slots=True)
class LayerRule:
    layer: str
    prefixes: tuple[str, ...]
    forbidden_layers: frozenset[str]
    rule_id: str
    reason: str


LAYER_RULES = (
    LayerRule(
        layer=SCHEMA_VALUES,
        prefixes=(
            "puripuly_heart.config.desktop_overlay_values",
            "puripuly_heart.config.overlay_calibration",
            "puripuly_heart.config.settings_vnext.schema",
            "puripuly_heart.config.audio_host_api",
            "puripuly_heart.config.llm_profiles",
        ),
        forbidden_layers=frozenset(
            {
                UI_ADAPTERS_RENDERERS,
                APP_SERVICES,
                ADAPTERS,
                PROVIDERS,
                OVERLAY_CORE,
                OUTPUT_MESSAGE_OBSERVABILITY_PORTS,
            }
        ),
        rule_id="schema-values-stay-pure",
        reason="schema/default value modules must not depend on UI, services, adapters, providers, overlay runtime, or observability ports",
    ),
    LayerRule(
        layer=MIGRATION_SERIALIZATION,
        prefixes=(
            "puripuly_heart.config.settings",
            "puripuly_heart.config.settings_vnext.migration",
            "puripuly_heart.config.settings_vnext.serialization",
            "puripuly_heart.config.settings_vnext.compat",
            "puripuly_heart.config.settings_vnext.facade",
        ),
        forbidden_layers=frozenset(
            {
                UI_ADAPTERS_RENDERERS,
                APP_SERVICES,
                ADAPTERS,
                PROVIDERS,
                RUNTIME_OWNERS,
            }
        ),
        rule_id="migration-serialization-stays-compatible-and-pure",
        reason="settings migration and serialization must not import UI, app services, provider construction, SecretStore/Broker adapters, provider internals, or runtime state owners",
    ),
    LayerRule(
        layer=RESOLVED_DTOS,
        prefixes=("puripuly_heart.config.resolved",),
        forbidden_layers=frozenset(
            {
                UI_ADAPTERS_RENDERERS,
                APP_SERVICES,
                ADAPTERS,
                PROVIDERS,
                MIGRATION_SERIALIZATION,
            }
        ),
        rule_id="resolved-dtos-stay-pure",
        reason="resolved runtime DTOs must stay immutable/pure and avoid file I/O, SecretStore, providers, UI, Broker, or migration internals",
    ),
    LayerRule(
        layer=RUNTIME_RESOLUTION,
        prefixes=("puripuly_heart.config.runtime_resolution",),
        forbidden_layers=frozenset(
            {
                UI_ADAPTERS_RENDERERS,
                APP_SERVICES,
                ADAPTERS,
                PROVIDERS,
                MIGRATION_SERIALIZATION,
            }
        ),
        rule_id="runtime-resolution-stays-pure",
        reason="runtime resolution must consume canonical settings and resolved DTOs without file I/O, SecretStore, concrete providers, Flet UI, Broker HTTP, or migration internals",
    ),
    LayerRule(
        layer=DOMAIN,
        prefixes=("puripuly_heart.domain",),
        forbidden_layers=frozenset(
            {
                MIGRATION_SERIALIZATION,
                RUNTIME_RESOLUTION,
                UI_ADAPTERS_RENDERERS,
                APP_SERVICES,
                ADAPTERS,
                PROVIDERS,
            }
        ),
        rule_id="domain-stays-independent",
        reason="domain modules must not depend on config migration, UI, app services, adapters, runtime resolution, or concrete providers",
    ),
    LayerRule(
        layer=RUNTIME_OWNERS,
        prefixes=(
            "puripuly_heart.core.lifecycle",
            "puripuly_heart.core.runtime",
        ),
        forbidden_layers=frozenset(
            {
                MIGRATION_SERIALIZATION,
                UI_ADAPTERS_RENDERERS,
                APP_SERVICES,
                ADAPTERS,
                PROVIDERS,
            }
        ),
        rule_id="runtime-owners-use-ports",
        reason="runtime owners must coordinate through domain events, resolved DTOs, lifecycle/message/observability protocols, not app wiring, Flet UI, provider config parsing, or concrete adapters",
    ),
    LayerRule(
        layer=ORCHESTRATOR,
        prefixes=("puripuly_heart.core.orchestrator",),
        forbidden_layers=frozenset(
            {
                MIGRATION_SERIALIZATION,
                UI_ADAPTERS_RENDERERS,
                APP_SERVICES,
                ADAPTERS,
                PROVIDERS,
            }
        ),
        rule_id="orchestrator-avoids-product-adapters",
        reason="orchestrator modules must avoid Flet UI, concrete provider construction, settings migration internals, services, and product-output adapters",
    ),
    LayerRule(
        layer=OVERLAY_CORE,
        prefixes=("puripuly_heart.core.overlay",),
        forbidden_layers=frozenset({UI_ADAPTERS_RENDERERS}),
        rule_id="overlay-core-avoids-ui-renderers",
        reason="overlay core may use overlay protocol/value objects and observability ports, but not Flet controls, views, or desktop renderer defaults except through adapters",
    ),
    LayerRule(
        layer=APP_SERVICES,
        prefixes=("puripuly_heart.app.services",),
        forbidden_layers=frozenset(
            {
                MIGRATION_SERIALIZATION,
                UI_ADAPTERS_RENDERERS,
                ADAPTERS,
                PROVIDERS,
            }
        ),
        rule_id="app-services-use-ports",
        reason="app services own transactions through ports and DTOs, not UI controls, localized text, concrete providers, adapters, or migration internals",
    ),
    LayerRule(
        layer=APP_COMPOSITION,
        prefixes=(
            "puripuly_heart.app.services.canonical_settings_persistence",
            "puripuly_heart.app.services.capture.capture_target_settings",
            "puripuly_heart.app.services.github_star_prompt_settings",
            "puripuly_heart.app.services.manual_local_asr_fallback",
            "puripuly_heart.app.services.openrouter_pkce_flow",
            "puripuly_heart.app.services.capture.peer_capture_target_application",
            "puripuly_heart.app.services.provider.provider_settings",
            "puripuly_heart.app.services.settings.settings_application",
            "puripuly_heart.app.services.settings.settings_runtime_effects",
        ),
        forbidden_layers=frozenset(
            {
                UI_ADAPTERS_RENDERERS,
                ADAPTERS,
                PROVIDERS,
            }
        ),
        rule_id="app-composition-owns-settings-persistence-assembly",
        reason="app composition may assemble the explicit settings persistence port and reference its public settings types, but must not absorb UI or unrelated adapter behavior",
    ),
    LayerRule(
        layer=SERVICE_PORTS,
        prefixes=(
            "puripuly_heart.app.ports",
            "puripuly_heart.core.osc.control_schema",
            "puripuly_heart.core.osc.control_codec",
            "puripuly_heart.core.osc.oscquery_contract",
            "puripuly_heart.core.osc.receiver_contract",
        ),
        forbidden_layers=frozenset(
            {
                MIGRATION_SERIALIZATION,
                UI_ADAPTERS_RENDERERS,
                ADAPTERS,
                PROVIDERS,
            }
        ),
        rule_id="service-ports-stay-abstract",
        reason="service ports and stable external contracts define protocols, DTOs, and ABI codecs only; they must not import concrete files, keyring/encrypted-file implementations, Flet, provider SDKs, adapters, or migration internals",
    ),
    LayerRule(
        layer=ADAPTERS,
        prefixes=(
            "puripuly_heart.app.adapters",
            "puripuly_heart.app.wiring",
            "puripuly_heart.core.openrouter.managed_openrouter_broker_client",
            "puripuly_heart.core.osc",
            "puripuly_heart.core.runtime_logging",
            "puripuly_heart.core.storage",
        ),
        forbidden_layers=frozenset(
            {
                MIGRATION_SERIALIZATION,
                UI_ADAPTERS_RENDERERS,
            }
        ),
        rule_id="adapters-avoid-ui-and-migration-internals",
        reason="adapters may wrap concrete resources but must not depend on settings migration internals or UI controls unless explicitly UI-owned",
    ),
    LayerRule(
        layer=SETTINGS_PERSISTENCE_ADAPTERS,
        prefixes=("puripuly_heart.app.adapters.settings_vnext_canonical_persistence",),
        forbidden_layers=frozenset(
            {
                UI_ADAPTERS_RENDERERS,
                APP_SERVICES,
                APP_COMPOSITION,
                ADAPTERS,
                PROVIDERS,
            }
        ),
        rule_id="settings-persistence-adapter-owns-canonical-settings-internals",
        reason="the settings persistence adapter is the sole concrete boundary permitted to use facade, migration, and serialization internals",
    ),
    LayerRule(
        layer=OUTPUT_MESSAGE_OBSERVABILITY_PORTS,
        prefixes=(
            "puripuly_heart.core.diagnostic_validation",
            "puripuly_heart.core.messages",
            "puripuly_heart.core.observability",
            "puripuly_heart.core.output",
        ),
        forbidden_layers=frozenset(
            {
                MIGRATION_SERIALIZATION,
                UI_ADAPTERS_RENDERERS,
                ADAPTERS,
                PROVIDERS,
            }
        ),
        rule_id="output-message-observability-ports-stay-abstract",
        reason="output/message/observability ports must avoid UI widgets, concrete OSC/overlay/log adapters, provider HTTP clients, and settings migration internals",
    ),
    LayerRule(
        layer=PROVIDERS,
        prefixes=("puripuly_heart.providers",),
        forbidden_layers=frozenset(
            {
                MIGRATION_SERIALIZATION,
                UI_ADAPTERS_RENDERERS,
                APP_SERVICES,
                ADAPTERS,
            }
        ),
        rule_id="providers-avoid-ui-settings-and-runtime-log-concretes",
        reason="providers may use provider ports, SDKs, and message/observability protocols, but not Flet UI, settings migration internals, app services, or concrete SessionRuntimeLoggingService-style adapters",
    ),
    LayerRule(
        layer=UI_ADAPTERS_RENDERERS,
        prefixes=("puripuly_heart.ui",),
        forbidden_layers=frozenset(
            {
                MIGRATION_SERIALIZATION,
                ADAPTERS,
                SETTINGS_PERSISTENCE_ADAPTERS,
                PROVIDERS,
            }
        ),
        rule_id="ui-adapters-avoid-provider-construction",
        reason="UI adapters/renderers may depend on app services, snapshots, i18n, and rendered log entries, not migration internals, provider construction, or concrete resource wiring",
    ),
)

EXTERNAL_MODULE_LAYERS = {
    "flet": UI_ADAPTERS_RENDERERS,
    "keyring": ADAPTERS,
}

KNOWN_ALLOWED_VIOLATIONS: frozenset[ImportViolation] = frozenset(
    {
        *(
            ImportViolation(
                rule_id="adapters-avoid-ui-and-migration-internals",
                importer=f"src/puripuly_heart/app/wiring/{importer_name}",
                imported="puripuly_heart.config.settings",
                importer_layer="adapters",
                imported_layer="migration/serialization",
                reason="adapters may wrap concrete resources but must not depend on settings migration internals or UI controls unless explicitly UI-owned",
            )
            for importer_name in (
                "wiring_composition.py",
                "wiring_managed_account.py",
                "wiring_managed_auth_factory.py",
                "wiring_secrets_factory.py",
            )
        ),
    }
)

SETTINGS_COMPATIBILITY_SOURCE_PATHS = frozenset(
    {
        "src/puripuly_heart/config/settings.py",
        "src/puripuly_heart/config/settings_vnext/compat.py",
        "src/puripuly_heart/config/settings_vnext/canonical_persistence.py",
        "src/puripuly_heart/config/settings_vnext/facade.py",
        "src/puripuly_heart/config/settings_vnext/migration.py",
        "src/puripuly_heart/config/settings_vnext/serialization.py",
    }
)

SETTINGS_PUBLIC_COMPATIBILITY_FACADE_PATHS = frozenset(
    {
        "src/puripuly_heart/app/wiring/root.py",
    }
)

SETTINGS_PERSISTENCE_COMPOSITION_PATHS = frozenset(
    {
        "src/puripuly_heart/app/adapters/settings_vnext_canonical_persistence.py",
        "src/puripuly_heart/app/services/canonical_settings_persistence.py",
    }
)

SETTINGS_LEGACY_COMPATIBILITY_ADAPTER_PATHS = frozenset(
    {
        "src/puripuly_heart/app/services/settings/settings_mutation_legacy.py",
        "src/puripuly_heart/app/services/github_star_prompt_settings.py",
        "src/puripuly_heart/app/services/manual_local_asr_fallback.py",
        "src/puripuly_heart/app/services/openrouter_pkce_flow.py",
        "src/puripuly_heart/app/services/settings/settings_application.py",
        "src/puripuly_heart/app/services/settings/settings_runtime_effects.py",
        "src/puripuly_heart/app/wiring/wiring_managed_auth_factory.py",
        "src/puripuly_heart/app/wiring/wiring_managed_account.py",
    }
)

LEGACY_SETTINGS_API_NAMES = frozenset(
    {
        "AppSettings",
        "from_dict",
        "load_settings",
        "load_settings_with_result",
        "save_settings",
        "save_settings_with_result",
        "to_dict",
        "to_legacy_dict",
    }
)

FLAT_SETTINGS_PATCH_SYMBOLS = frozenset(
    {
        "ORDER21_TRANSLATION_PROVIDER_SETTINGS_PATHS",
        "ORDER22_STT_LANGUAGE_AUDIO_SETTINGS_PATHS",
        "ORDER23_OVERLAY_OSC_OUTPUT_SETTINGS_PATHS",
        "ORDER24_UI_PROMPT_CLIPBOARD_STATE_SETTINGS_PATHS",
        "SettingsPathMutationValidator",
        "SettingsPathPatch",
    }
)

CONTROLLER_FLAT_SETTINGS_PATCH_HELPERS = frozenset(
    {
        "_apply_settings_path_patch",
        "_build_settings_path_patch",
        "_get_settings_path_value",
        "_set_settings_path_value",
    }
)

LEGACY_SETTINGS_VALUE_PAYLOAD_KEYS = frozenset(
    {
        "api_key_verified",
        "openrouter.llm_model",
        "openrouter.selected_source",
        "openrouter.selection_alias",
        "provider.llm",
    }
)

LEGACY_SETTINGS_VALUE_PAYLOAD_PREFIXES = (
    "state.managed_connection.",
    "state.managed_identity.",
    "state.provider_verification.",
)

UNKNOWN_SETTINGS_RUNTIME_CONFINEMENT_RATIONALE = "unclassified order-11 settings runtime debt"

KNOWN_SETTINGS_RUNTIME_CONFINEMENT_DEBT: frozenset[SettingsRuntimeConfinementViolation] = frozenset(
    {
        SettingsRuntimeConfinementViolation(
            "legacy-settings-api-import",
            "src/puripuly_heart/composition/application_runtime.py",
            "AppSettings",
            "The production composition root uses the AppSettings compatibility DTO only to connect focused owners while SettingsOwner exclusively owns load, normalization, migration, backup, and persistence.",
        ),
        SettingsRuntimeConfinementViolation(
            "legacy-settings-api-import",
            "src/puripuly_heart/core/telemetry.py",
            "AppSettings",
            "Translation-success telemetry service mutates and persists consent/anonymous identity through the public AppSettings compatibility model until a dedicated telemetry state port is extracted.",
        ),
    }
)


def _known_allowed_violation_gate6_rationale(violation: ImportViolation) -> str:
    if (
        violation.importer
        == "src/puripuly_heart/core/openrouter/managed_openrouter_broker_client.py"
    ):
        return "managed OpenRouter broker adapter still consumes public settings compatibility values at the adapter boundary"
    if violation.importer_layer == ADAPTERS and "/app/wiring/" in violation.importer:
        return (
            "wiring composition modules retain public settings compatibility imports; grouping the "
            "wiring package makes the pre-existing debt layer-visible, so it is explicitly recorded "
            "until the settings compatibility facade is retired"
        )
    if violation.importer_layer == RUNTIME_OWNERS:
        return "runtime owner currently wraps a concrete adapter while preserving explicit lifecycle ownership; adapter-port extraction remains deferred work"
    if violation.importer_layer == PROVIDERS:
        return "provider modules retain concrete runtime logging/settings compatibility imports until provider observation ports replace adapter logging"
    if violation.importer_layer == UI_ADAPTERS_RENDERERS:
        if violation.imported == "puripuly_heart.config.settings":
            return "UI boundary uses the public settings compatibility facade for user settings load/edit/save surfaces"
        if violation.imported == "puripuly_heart.app.wiring":
            return "UI composition still enters through the preserved public wiring facade while split factories remain behind it"
        return "UI boundary still wires concrete adapter seams for user-facing runtime controls; concrete port extraction is deferred and explicitly guarded"
    return UNKNOWN_SETTINGS_RUNTIME_CONFINEMENT_RATIONALE


KNOWN_ALLOWED_VIOLATION_GATE6_RATIONALES = {
    violation: _known_allowed_violation_gate6_rationale(violation)
    for violation in KNOWN_ALLOWED_VIOLATIONS
}


def _module_name_for_path(path: Path) -> str:
    relative = path.relative_to(SOURCE_PACKAGE_ROOT).with_suffix("")
    parts = (PACKAGE_NAME, *relative.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _package_parts_for_importer(importer_module: str, importer_path: Path) -> list[str]:
    parts = importer_module.split(".")
    if importer_path.name != "__init__.py":
        parts = parts[:-1]
    return parts


def _absolute_import_from_module(
    importer_module: str,
    importer_path: Path,
    node: ast.ImportFrom,
) -> str | None:
    if node.level == 0:
        return node.module

    package_parts = _package_parts_for_importer(importer_module, importer_path)
    if node.level > len(package_parts) + 1:
        return None

    base_parts = package_parts[: len(package_parts) - node.level + 1]
    if node.module:
        base_parts.extend(node.module.split("."))
    return ".".join(base_parts)


def _internal_module_names() -> frozenset[str]:
    return frozenset(_module_name_for_path(path) for path in SOURCE_PACKAGE_ROOT.rglob("*.py"))


def _is_internal_module(module: str) -> bool:
    return module == PACKAGE_NAME or module.startswith(f"{PACKAGE_NAME}.")


def _layer_root_module_names() -> frozenset[str]:
    return frozenset(prefix for rule in LAYER_RULES for prefix in rule.prefixes)


def _imported_modules(
    importer_module: str,
    importer_path: Path,
    internal_modules: frozenset[str],
) -> Iterator[str]:
    tree = ast.parse(importer_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
            continue

        if not isinstance(node, ast.ImportFrom):
            continue

        module = _absolute_import_from_module(importer_module, importer_path, node)
        if module is None:
            continue

        if not _is_internal_module(module):
            yield module
            continue

        for alias in node.names:
            candidate = f"{module}.{alias.name}"
            if candidate in internal_modules or candidate in _layer_root_module_names():
                yield candidate
            else:
                yield module


def _layer_for_module(module: str) -> str | None:
    for external_module, layer in EXTERNAL_MODULE_LAYERS.items():
        if module == external_module or module.startswith(f"{external_module}."):
            return layer

    for prefix, layer in _layer_prefixes_by_specificity():
        if module == prefix or module.startswith(f"{prefix}."):
            return layer

    return None


def _layer_prefixes_by_specificity() -> tuple[tuple[str, str], ...]:
    prefixes = [(prefix, rule.layer) for rule in LAYER_RULES for prefix in rule.prefixes]
    return tuple(sorted(prefixes, key=lambda entry: len(entry[0]), reverse=True))


def _rule_for_layer(layer: str) -> LayerRule:
    for rule in LAYER_RULES:
        if rule.layer == layer:
            return rule
    raise AssertionError(f"no dependency rule for layer {layer!r}")


def _relative_repo_path(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _dependency_violations() -> frozenset[ImportViolation]:
    internal_modules = _internal_module_names()
    violations: set[ImportViolation] = set()

    for importer_path in sorted(SOURCE_PACKAGE_ROOT.rglob("*.py")):
        importer_module = _module_name_for_path(importer_path)
        importer_layer = _layer_for_module(importer_module)
        if importer_layer is None:
            continue

        rule = _rule_for_layer(importer_layer)
        for imported_module in sorted(
            set(_imported_modules(importer_module, importer_path, internal_modules))
        ):
            imported_layer = _layer_for_module(imported_module)
            if imported_layer is None:
                continue
            if imported_layer not in rule.forbidden_layers:
                continue

            violations.add(
                ImportViolation(
                    rule_id=rule.rule_id,
                    importer=_relative_repo_path(importer_path),
                    imported=imported_module,
                    importer_layer=importer_layer,
                    imported_layer=imported_layer,
                    reason=rule.reason,
                )
            )

    return frozenset(violations)


def _format_violations(violations: list[ImportViolation]) -> str:
    if not violations:
        return "  <none>"

    return "\n".join(
        "  ImportViolation(\n"
        f'      rule_id="{violation.rule_id}",\n'
        f'      importer="{violation.importer}",\n'
        f'      imported="{violation.imported}",\n'
        f'      importer_layer="{violation.importer_layer}",\n'
        f'      imported_layer="{violation.imported_layer}",\n'
        f'      reason="{violation.reason}",\n'
        "  ),"
        for violation in violations
    )


def _format_settings_runtime_violations(
    violations: list[SettingsRuntimeConfinementViolation],
) -> str:
    if not violations:
        return "  <none>"

    return "\n".join(
        "  SettingsRuntimeConfinementViolation(\n"
        f'      category="{violation.category}",\n'
        f'      path="{violation.path}",\n'
        f'      symbol="{violation.symbol}",\n'
        f'      rationale="{violation.rationale}",\n'
        "  ),"
        for violation in violations
    )


def _known_settings_runtime_rationale(
    *,
    category: str,
    path: str,
    symbol: str,
) -> str:
    for violation in KNOWN_SETTINGS_RUNTIME_CONFINEMENT_DEBT:
        if violation.category == category and violation.path == path and violation.symbol == symbol:
            return violation.rationale
    return UNKNOWN_SETTINGS_RUNTIME_CONFINEMENT_RATIONALE


def _settings_runtime_violation(
    *,
    category: str,
    path: str,
    symbol: str,
) -> SettingsRuntimeConfinementViolation:
    return SettingsRuntimeConfinementViolation(
        category=category,
        path=path,
        symbol=symbol,
        rationale=_known_settings_runtime_rationale(
            category=category,
            path=path,
            symbol=symbol,
        ),
    )


def _settings_runtime_confinement_violations() -> frozenset[SettingsRuntimeConfinementViolation]:
    violations: set[SettingsRuntimeConfinementViolation] = set()
    for source_path in sorted(SOURCE_PACKAGE_ROOT.rglob("*.py")):
        relative_path = _relative_repo_path(source_path)
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        violations.update(_legacy_settings_api_import_violations(tree, relative_path))
        violations.update(_dynamic_settings_shape_violations(tree, relative_path))
        violations.update(_flat_settings_patch_violations(tree, relative_path))
        violations.update(_legacy_settings_value_payload_key_violations(tree, relative_path))
    return frozenset(violations)


def _legacy_settings_api_import_violations(
    tree: ast.AST,
    relative_path: str,
) -> set[SettingsRuntimeConfinementViolation]:
    if relative_path in SETTINGS_COMPATIBILITY_SOURCE_PATHS:
        return set()
    if relative_path in SETTINGS_PUBLIC_COMPATIBILITY_FACADE_PATHS:
        return set()
    if relative_path in SETTINGS_PERSISTENCE_COMPOSITION_PATHS:
        return set()
    if relative_path in SETTINGS_LEGACY_COMPATIBILITY_ADAPTER_PATHS:
        return set()

    violations: set[SettingsRuntimeConfinementViolation] = set()
    migration_module_aliases = _settings_vnext_migration_module_aliases(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            if _uses_qualified_to_legacy_dict(node, migration_module_aliases):
                violations.add(
                    _settings_runtime_violation(
                        category="legacy-settings-migration-projection",
                        path=relative_path,
                        symbol="to_legacy_dict",
                    )
                )
            continue
        if node.module == "puripuly_heart.config.settings":
            imported_legacy_symbols = {
                alias.name for alias in node.names if alias.name in LEGACY_SETTINGS_API_NAMES
            }
        elif node.module == "puripuly_heart.config.settings_vnext.migration":
            imported_legacy_symbols = {
                alias.name for alias in node.names if alias.name == "to_legacy_dict"
            }
        else:
            continue
        for symbol in sorted(imported_legacy_symbols):
            violations.add(
                _settings_runtime_violation(
                    category="legacy-settings-api-import",
                    path=relative_path,
                    symbol=symbol,
                )
            )
    return violations


def _settings_vnext_migration_module_aliases(tree: ast.AST) -> frozenset[str]:
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "puripuly_heart.config.settings_vnext.migration":
                    aliases.add(alias.asname or "puripuly_heart")
        elif isinstance(node, ast.ImportFrom):
            if node.module == "puripuly_heart.config.settings_vnext":
                aliases.update(
                    alias.asname or alias.name for alias in node.names if alias.name == "migration"
                )
    return frozenset(aliases)


def _uses_qualified_to_legacy_dict(
    node: ast.AST,
    migration_module_aliases: frozenset[str],
) -> bool:
    if not isinstance(node, ast.Attribute) or node.attr != "to_legacy_dict":
        return False
    parts = _attribute_parts(node)
    if len(parts) < 2:
        return False
    module_parts = parts[:-1]
    if module_parts[0] in migration_module_aliases:
        return True
    return module_parts == ["puripuly_heart", "config", "settings_vnext", "migration"]


def _attribute_parts(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        return [*_attribute_parts(node.value), node.attr]
    return []


def _dynamic_settings_shape_violations(
    tree: ast.AST,
    relative_path: str,
) -> set[SettingsRuntimeConfinementViolation]:
    if relative_path in SETTINGS_COMPATIBILITY_SOURCE_PATHS:
        return set()
    if relative_path in SETTINGS_LEGACY_COMPATIBILITY_ADAPTER_PATHS:
        return set()

    violations: set[SettingsRuntimeConfinementViolation] = set()
    for node, qualified_name in _function_nodes_with_qualified_names(tree):
        function_symbol = _settings_shape_function_symbol(node, qualified_name)
        if function_symbol is None:
            continue
        violations.add(
            _settings_runtime_violation(
                category="dynamic-legacy-settings-shape-read",
                path=relative_path,
                symbol=function_symbol,
            )
        )
    return violations


def _function_nodes_with_qualified_names(
    tree: ast.AST,
) -> Iterator[tuple[ast.FunctionDef | ast.AsyncFunctionDef, str]]:
    parents = {
        id(child): parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        parent = parents.get(id(node))
        if isinstance(parent, ast.ClassDef):
            yield node, f"{parent.name}.{node.name}"
        else:
            yield node, node.name


def _settings_shape_function_symbol(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    qualified_name: str,
) -> str | None:
    calls_dynamic_settings = False
    reads_legacy_settings_leaf = False
    calls_settings_identity_helper = False
    for child in ast.walk(node):
        if _is_getattr_settings_call(child):
            calls_dynamic_settings = True
        if isinstance(child, ast.Attribute) and child.attr in {"languages", "openrouter"}:
            reads_legacy_settings_leaf = True
        if _is_getattr_legacy_settings_leaf_call(child):
            reads_legacy_settings_leaf = True
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
            if child.func.attr == "_settings_identity":
                calls_settings_identity_helper = True

    if calls_dynamic_settings and (reads_legacy_settings_leaf or calls_settings_identity_helper):
        return qualified_name
    if node.name == "_settings_identity" and reads_legacy_settings_leaf:
        return qualified_name
    return None


def _is_getattr_settings_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == "settings"
    )


def _is_getattr_legacy_settings_leaf_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value in {"languages", "llm_model", "openrouter", "selected_source"}
    )


def _flat_settings_patch_violations(
    tree: ast.AST,
    relative_path: str,
) -> set[SettingsRuntimeConfinementViolation]:
    if relative_path in SETTINGS_LEGACY_COMPATIBILITY_ADAPTER_PATHS:
        return set()

    violations: set[SettingsRuntimeConfinementViolation] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in {
            "puripuly_heart.app.services.settings_mutation",
            "puripuly_heart.app.services.settings.settings_mutation",
        }:
            for alias in node.names:
                if alias.name in FLAT_SETTINGS_PATCH_SYMBOLS:
                    violations.add(
                        _settings_runtime_violation(
                            category="legacy-flat-settings-patch-import",
                            path=relative_path,
                            symbol=alias.name,
                        )
                    )
        if isinstance(node, ast.ClassDef) and node.name in FLAT_SETTINGS_PATCH_SYMBOLS:
            violations.add(
                _settings_runtime_violation(
                    category="legacy-flat-settings-patch-definition",
                    path=relative_path,
                    symbol=node.name,
                )
            )
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name in (
            CONTROLLER_FLAT_SETTINGS_PATCH_HELPERS
        ):
            violations.add(
                _settings_runtime_violation(
                    category="legacy-flat-settings-patch-helper",
                    path=relative_path,
                    symbol=node.name,
                )
            )
        if isinstance(node, ast.Assign | ast.AnnAssign):
            for target_name in _assignment_target_names(node):
                if target_name in FLAT_SETTINGS_PATCH_SYMBOLS:
                    violations.add(
                        _settings_runtime_violation(
                            category="legacy-flat-settings-patch-definition",
                            path=relative_path,
                            symbol=target_name,
                        )
                    )
    return violations


def _assignment_target_names(node: ast.Assign | ast.AnnAssign) -> Iterator[str]:
    targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
    for target in targets:
        if isinstance(target, ast.Name):
            yield target.id


def _legacy_settings_value_payload_key_violations(
    tree: ast.AST,
    relative_path: str,
) -> set[SettingsRuntimeConfinementViolation]:
    if not relative_path.startswith("src/puripuly_heart/app/services/"):
        return set()
    if relative_path in SETTINGS_LEGACY_COMPATIBILITY_ADAPTER_PATHS:
        return set()

    violations: set[SettingsRuntimeConfinementViolation] = set()
    for node in ast.walk(tree):
        symbol = _legacy_settings_value_payload_symbol(node)
        if symbol is None:
            continue
        violations.add(
            _settings_runtime_violation(
                category="legacy-settings-value-payload-key",
                path=relative_path,
                symbol=symbol,
            )
        )
    return violations


def _legacy_settings_value_payload_symbol(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return _legacy_settings_value_payload_text_symbol(node.value)
    if isinstance(node, ast.JoinedStr):
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                symbol = _legacy_settings_value_payload_text_symbol(value.value, dynamic=True)
                if symbol is not None:
                    return symbol
    return None


def _legacy_settings_value_payload_text_symbol(
    text: str,
    *,
    dynamic: bool = False,
) -> str | None:
    if text in LEGACY_SETTINGS_VALUE_PAYLOAD_KEYS:
        return text
    for prefix in LEGACY_SETTINGS_VALUE_PAYLOAD_PREFIXES:
        if text.startswith(prefix):
            return f"{prefix}*"
        if dynamic and prefix.startswith(text):
            return f"{prefix}*"
    return None


def test_dependency_rule_vocabulary_distinguishes_required_layers() -> None:
    assert tuple(rule.layer for rule in LAYER_RULES) == REQUIRED_LAYER_VOCABULARY
    assert {rule.layer for rule in LAYER_RULES} == set(REQUIRED_LAYER_VOCABULARY)
    assert _layer_for_module("keyring") == ADAPTERS


def test_migration_serialization_forbids_runtime_owner_dependencies() -> None:
    assert RUNTIME_OWNERS in _rule_for_layer(MIGRATION_SERIALIZATION).forbidden_layers


def test_concrete_osc_modules_classify_as_adapters() -> None:
    assert _layer_for_module("puripuly_heart.core.osc.chatbox_paginator") == ADAPTERS
    assert _layer_for_module("puripuly_heart.core.osc.receiver") == ADAPTERS
    assert _layer_for_module("puripuly_heart.core.osc.udp_sender") == ADAPTERS


def test_osc_receiver_contract_classifies_as_service_port() -> None:
    assert _layer_for_module("puripuly_heart.core.osc.receiver_contract") == SERVICE_PORTS


def test_overlay_calibration_value_object_has_config_schema_ownership() -> None:
    assert _layer_for_module("puripuly_heart.config.overlay_calibration") == SCHEMA_VALUES
    assert _layer_for_module("puripuly_heart.ui.overlay_calibration") == UI_ADAPTERS_RENDERERS

    forbidden_imports = {
        ("src/puripuly_heart/config/settings.py", "puripuly_heart.ui.overlay_calibration"),
        ("src/puripuly_heart/core/overlay/presenter.py", "puripuly_heart.ui.overlay_calibration"),
    }
    actual_imports = {
        (violation.importer, violation.imported) for violation in _dependency_violations()
    }

    assert not (forbidden_imports & actual_imports)


def test_settings_public_facade_delegates_persistence_helpers_to_vnext_facade() -> None:
    assert (
        _layer_for_module("puripuly_heart.config.settings_vnext.facade") == MIGRATION_SERIALIZATION
    )

    settings_path = SOURCE_PACKAGE_ROOT / "config" / "settings.py"
    tree = ast.parse(settings_path.read_text(encoding="utf-8"))
    delegated_names = {
        "FacadeSettingsLoadResult",
        "load_settings",
        "load_settings_with_result",
        "save_settings",
        "save_settings_with_result",
        "load_vnext_settings",
        "save_vnext_settings",
    }
    helper_definitions = delegated_names | {"_atomic_write_text"}
    definitions = {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    }
    facade_imports = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "puripuly_heart.config.settings_vnext.facade"
        for alias in node.names
    }

    assert definitions.isdisjoint(helper_definitions)
    assert delegated_names <= facade_imports


def test_controller_consumes_only_settings_owner_and_public_binding_contract() -> None:
    controller_path = SOURCE_PACKAGE_ROOT / "composition" / "application_runtime.py"
    tree = ast.parse(controller_path.read_text(encoding="utf-8"))
    imports = {
        node.module: {alias.name for alias in node.names}
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert "puripuly_heart.app.ports.canonical_settings_persistence" not in imports
    assert imports["puripuly_heart.app.services.canonical_settings_persistence"] == {
        "compose_settings_owner",
    }
    forbidden_modules = {
        "puripuly_heart.app.adapters.settings_vnext_canonical_persistence",
        "puripuly_heart.config.settings_vnext.facade",
        "puripuly_heart.config.settings_vnext.migration",
        "puripuly_heart.config.settings_vnext.serialization",
    }
    assert forbidden_modules.isdisjoint(imports)
    assert "save_vnext_settings" not in {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    assert {"load_settings", "save_settings", "persist_desktop_audio_capture_target"}.isdisjoint(
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    )


def test_canonical_settings_persistence_layers_are_explicit() -> None:
    assert (
        _layer_for_module("puripuly_heart.app.adapters.settings_vnext_canonical_persistence")
        == SETTINGS_PERSISTENCE_ADAPTERS
    )
    assert (
        _layer_for_module("puripuly_heart.app.services.canonical_settings_persistence")
        == APP_COMPOSITION
    )


def test_canonical_settings_persistence_composition_uses_only_public_settings_types() -> None:
    composition_path = (
        SOURCE_PACKAGE_ROOT / "app" / "services" / "canonical_settings_persistence.py"
    )
    tree = ast.parse(composition_path.read_text(encoding="utf-8"))
    imports = {
        node.module: {alias.name for alias in node.names}
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert imports["puripuly_heart.config.settings"] == {"AppSettings"}
    assert imports["puripuly_heart.config.settings_vnext.defaults"] == {
        "new_settings_for_first_run"
    }
    assert imports["puripuly_heart.config.settings_vnext.schema"] == {
        "AppSettingsVNext",
        "CaptureTargetIntent",
        "with_capture_target",
    }
    assert {
        "puripuly_heart.config.settings_vnext.facade",
        "puripuly_heart.config.settings_vnext.migration",
        "puripuly_heart.config.settings_vnext.serialization",
    }.isdisjoint(imports)


def test_capture_target_compatibility_service_delegates_to_settings_owner() -> None:
    service_path = (
        SOURCE_PACKAGE_ROOT / "app" / "services" / "capture" / "capture_target_settings.py"
    )
    tree = ast.parse(service_path.read_text(encoding="utf-8"))
    imports = {
        node.module: {alias.name for alias in node.names}
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert imports["puripuly_heart.app.services.canonical_settings_persistence"] == {
        "compose_settings_owner"
    }
    assert {
        "puripuly_heart.config.settings_vnext.compat",
        "puripuly_heart.config.settings_vnext.facade",
        "puripuly_heart.config.settings_vnext.migration",
        "puripuly_heart.config.settings_vnext.serialization",
    }.isdisjoint(imports)


def test_settings_persistence_calls_are_confined_to_owner_mechanics() -> None:
    allowed_paths = {
        "app/adapters/settings_vnext_canonical_persistence.py",
        "config/settings_vnext/compat.py",
        "config/settings_vnext/facade.py",
    }
    persistence_calls = {
        "load_settings",
        "load_settings_with_result",
        "load_vnext_settings",
        "save_settings",
        "save_vnext_settings",
    }
    violations: list[tuple[str, str]] = []
    for path in SOURCE_PACKAGE_ROOT.rglob("*.py"):
        relative_path = path.relative_to(SOURCE_PACKAGE_ROOT).as_posix()
        if relative_path in allowed_paths or relative_path == "main.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            callee = node.func.id if isinstance(node.func, ast.Name) else None
            if isinstance(node.func, ast.Attribute):
                callee = node.func.attr
            if callee in persistence_calls:
                violations.append((relative_path, callee))

    assert violations == []

    main_tree = ast.parse((SOURCE_PACKAGE_ROOT / "main.py").read_text(encoding="utf-8"))
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_load_settings_or_default"
        for node in ast.walk(main_tree)
    )

    assert not (SOURCE_PACKAGE_ROOT / "config" / "profile_bootstrap.py").exists()


def test_internal_source_imports_canonical_overlay_calibration_not_ui_facade() -> None:
    facade_module = "puripuly_heart.ui.overlay_calibration"
    internal_modules = _internal_module_names()
    offenders: set[str] = set()

    for importer_path in sorted(SOURCE_PACKAGE_ROOT.rglob("*.py")):
        importer_module = _module_name_for_path(importer_path)
        imported_modules = set(_imported_modules(importer_module, importer_path, internal_modules))
        if facade_module not in imported_modules:
            continue

        offenders.add(_relative_repo_path(importer_path))

    assert offenders == set()


def test_application_composition_uses_adapter_seam_instead_of_provider_imports() -> None:
    composition_path = SOURCE_PACKAGE_ROOT / "composition" / "application_runtime.py"
    imported_modules = set(
        _imported_modules(
            "puripuly_heart.composition.application_runtime",
            composition_path,
            _internal_module_names(),
        )
    )

    assert not {
        imported_module
        for imported_module in imported_modules
        if imported_module == "puripuly_heart.providers"
        or imported_module.startswith("puripuly_heart.providers.")
    }


def test_application_composition_avoids_legacy_overlay_resource_mirrors() -> None:
    composition_path = SOURCE_PACKAGE_ROOT / "composition" / "application_runtime.py"
    tree = ast.parse(composition_path.read_text(encoding="utf-8"))
    legacy_mirror_fields = {
        "_overlay_presenter",
        "_overlay_bridge",
        "_overlay_manager",
        "_overlay_diagnostics",
        "_overlay_start_task",
        "_overlay_monitor_task",
        "_desktop_renderer_events",
        "_desktop_renderer_events_task",
    }
    removed_private_alias_shims = legacy_mirror_fields | {
        "_overlay_runtime_for_private_alias",
    }
    offenders: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if node.name in removed_private_alias_shims:
            offenders.append(f"{node.name}:{node.lineno}:definition")
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Attribute)
                and child.attr in legacy_mirror_fields
                and isinstance(child.value, ast.Name)
                and child.value.id == "self"
            ):
                offenders.append(f"{node.name}:{child.lineno}:{child.attr}")

    assert offenders == []


def test_application_composition_has_no_concrete_osc_import_violations() -> None:
    orchestrator_rule = _rule_for_layer(ORCHESTRATOR)
    assert not any(
        violation.importer == "src/puripuly_heart/composition/application_runtime.py"
        and ".core.osc." in violation.imported
        for violation in _dependency_violations()
    )
    assert (
        ImportViolation(
            rule_id=orchestrator_rule.rule_id,
            importer="src/puripuly_heart/core/orchestrator/peer_translation_channel.py",
            imported="puripuly_heart.core.osc.chatbox_paginator",
            importer_layer=ORCHESTRATOR,
            imported_layer=ADAPTERS,
            reason=orchestrator_rule.reason,
        )
        not in _dependency_violations()
    )


def test_a03_runtime_owner_adapter_imports_are_retired() -> None:
    runtime_owner_rule = _rule_for_layer(RUNTIME_OWNERS)
    inherited_runtime_owner_violations = {
        ImportViolation(
            rule_id=runtime_owner_rule.rule_id,
            importer="src/puripuly_heart/core/runtime/logging.py",
            imported="puripuly_heart.core.runtime_logging",
            importer_layer=RUNTIME_OWNERS,
            imported_layer=ADAPTERS,
            reason=runtime_owner_rule.reason,
        ),
        ImportViolation(
            rule_id=runtime_owner_rule.rule_id,
            importer="src/puripuly_heart/core/runtime/receiver.py",
            imported="puripuly_heart.core.osc.receiver",
            importer_layer=RUNTIME_OWNERS,
            imported_layer=ADAPTERS,
            reason=runtime_owner_rule.reason,
        ),
    }

    assert inherited_runtime_owner_violations.isdisjoint(_dependency_violations())
    assert inherited_runtime_owner_violations.isdisjoint(KNOWN_ALLOWED_VIOLATIONS)


def test_gate1_existing_replacement_private_shims_are_removed() -> None:
    controller_overlay_alias_shims = {
        "_overlay_presenter",
        "_overlay_bridge",
        "_overlay_manager",
        "_overlay_diagnostics",
        "_overlay_start_task",
        "_overlay_monitor_task",
        "_desktop_renderer_events",
        "_desktop_renderer_events_task",
    }
    disallowed_private_shims = {
        "src/puripuly_heart/composition/application_runtime.py": (
            "_overlay_runtime_for_private_alias",
            "def _get_github_star_prompt_runtime(",
            "def _github_star_prompt_runtime(",
            "@_github_star_prompt_runtime.setter",
            "def _drain_github_star_prompt_translation_success_observation(",
            "def _sync_github_star_prompt_runtime_aliases",
            "def _github_star_prompt_initial_launch_gate_satisfied(",
            "def _get_github_star_prompt_persistence_lock(",
            "def _github_star_prompt_state_snapshot(",
            "def _restore_github_star_prompt_state_snapshot(",
            "def _persist_github_star_prompt_mutation(",
            "def _run_github_star_prompt_persistence_sync(",
            "def record_github_star_prompt_opened(",
            "def record_github_star_prompt_clicked(",
            "def record_github_star_prompt_translation_success_observed(",
            "def _persist_desktop_bounds_after_debounce(",
            "def _desktop_bounds_persist_task(",
            "def _pending_desktop_bounds(",
            "@_pending_desktop_bounds.setter",
            "def _build_initial_desktop_runtime_controls_from_resolved_config(",
            "def _desktop_dimensions_for_size_preset(",
            "def _desktop_launch_bounds_for_current_launch(",
            "def _desktop_centered_bounds_for_dimensions(",
            "def _broadcast_desktop_runtime_control(",
            "def _broadcast_desktop_window_bounds_control(",
            "def _handle_desktop_renderer_event(",
            "def _handle_desktop_window_bounds_changed(",
            "def _persist_desktop_bounds(",
            "def _handle_desktop_overlay_reset_requested(",
            "def _get_desktop_overlay_bounds_owner(",
            "def _prepare_desktop_runtime_settings_update(",
            "def _apply_desktop_size_preset_persistence_adjustment(",
            "def _persist_overlay_calibration(",
            "def _sync_overlay_calibration_cache(",
            "def _emit_overlay_calibration_to_runtime(",
            "def _get_overlay_calibration_owner(",
            "_desktop_overlay_bounds_owner:",
            "_overlay_calibration_owner:",
            "def _overlay_runtime(",
            "def _active_overlay_target(",
            "def failure_reason(",
            "def auto_restart_scheduled(",
            "@overlay_calibration.setter",
            "def _overlay_calibration_draft(",
            "def desktop_overlay_interaction_mode(",
            "def _normalized_overlay_target(",
            "def _overlay_target_for_settings(",
            "def _effective_overlay_target_for_start(",
            "def _clear_overlay_session_desktop_fallback(",
            "def _set_overlay_session_fallback_notice_active(",
            "def _should_session_fallback_overlay_to_desktop(",
            "def _replace_hub_overlay_sink(",
            "def _overlay_process_runner_for_target(",
            "def _new_overlay_runtime_handle(",
            "def _ensure_overlay_runtime_handle(",
            "def _overlay_runtime_is_current(",
            "def _overlay_runtime_has_resources(",
            "def _close_stale_overlay_start_runtime(",
            "def _overlay_runtime_is_active(",
            "def _current_overlay_presenter_for_direct_runtime_command(",
            "def _current_overlay_bridge_for_direct_runtime_command(",
            "def _previous_overlay_target_for_apply(",
            "def _begin_overlay_start(",
            "def _run_overlay_start(",
            "def _shutdown_overlay_runtime(",
            "def _teardown_overlay_runtime(",
            "def on_overlay_start_failed(",
            "def on_overlay_runtime_disconnected(",
            "def on_overlay_runtime_crashed(",
            "def _overlay_session_desktop_fallback_active(",
            "def _set_vrchat_osc_notice_active(",
            "def _run_vrchat_osc_presence_probe_loop(",
            "def _vrchat_osc_probe_task(",
            "def _local_stt_pending_enable_generation(",
            "@_local_stt_pending_enable_generation.setter",
            "def _local_stt_pending_peer_enable_after_install(",
            "@_local_stt_pending_peer_enable_after_install.setter",
            "def _apply_overlay_calibration_persistence(",
            "def _schedule_overlay_calibration_persistence(",
            "def _emit_overlay_calibration_update(",
            "def _schedule_overlay_calibration_emit(",
            "def _emit_overlay_shutdown(",
            "def _managed_openrouter_fallback_branch_settings_for(",
            "def _managed_openrouter_branch_settings_for(",
            "_MANAGED_OPENROUTER_MODEL_BY_TRANSLATION_MODEL =",
            "def _log_audio_environment_snapshot_async(",
            "def _refresh_owned_referral_id_from_managed_status_best_effort(",
            "def _request_local_asr_install(",
            "def _show_local_stt_download_prompt(",
            "def _on_local_stt_download_action(",
            "def _transition_active_self_local_asr(",
            "def _build_stt_runtime_signature(",
            "def _peer_stt_runtime_custom_vocabulary_signature(",
            "def _update_canonical_settings_from_compatibility_mutation(",
            "def effective_context_mode(",
            "def peer_warning_action_is_retry(",
            "def _build_initial_desktop_runtime_controls(",
            "def discord_managed_auth_in_progress(",
            "def effective_peer_translation_enabled(",
            "def _microphone_test_meter_level(",
            "@_microphone_test_meter_level.setter",
            "def microphone_test_meter_level(",
            "def _verify_and_update_status(",
            "def _rebuild_pipeline(",
            "def _get_provider_status_verification_owner(",
            "def _schedule_provider_status_verification(",
            "def _build_provider_status_verification_request(",
            "def _apply_provider_status_verification_result(",
            "def _close_provider_status_verification_owner(",
            "def _get_qwen_key_and_base_url(",
            "_provider_status_verification_owner:",
            'owner_name="ProviderStatusVerificationOwner"',
            "_self_local_asr_transition:",
            "_self_asr_model_loading:",
            "class _LocalASRTransitionSuperseded(",
            'owner_name="SelfLocalASRTransitionCoordinator"',
            "def _sync_local_stt_download_runtime_aliases",
            "def _sync_clipboard_runtime_aliases",
            "def _sync_microphone_test_runtime_aliases",
            "_github_star_prompt_translation_success_task",
            "_local_stt_download_task",
            "_local_stt_download_cancel_event",
            "_local_stt_download_origin",
            "def _get_clipboard_watcher_lock(",
            "def _get_clipboard_runtime(",
            "def _clipboard_runtime(",
            "@_clipboard_runtime.setter",
            "def _stop_clipboard_watcher(",
            "def _close_clipboard_runtime(",
            "def _on_clipboard_text_from_thread(",
            "def _schedule_clipboard_submit(",
            "def _submit_clipboard_text(",
            "_clipboard_watcher: ClipboardWatcherRuntime",
            "_clipboard_loop: asyncio.AbstractEventLoop",
            "_clipboard_runtime: ClipboardRuntime",
            "_clipboard_watcher_lock:",
            "_strict_runtime_errors_for_clipboard_watcher",
            "def _get_oauth_runtime(",
            "def _oauth_runtime(",
            "@_oauth_runtime.setter",
            "_oauth_runtime: OAuthRuntime",
            "def _openrouter_pkce_client(",
            "@_openrouter_pkce_client.setter",
            "def _close_oauth_runtime_for_release(",
            "_openrouter_pkce_client: OpenRouterPKCEClient",
            "receiver: VrcOscReceiver",
            "def _vrc_mic_receiver_runtime(",
            "@_vrc_mic_receiver_runtime.setter",
            "def _get_vrc_mic_receiver_runtime(",
            "def _sync_vrc_mic_receiver_runtime_aliases(",
            "_vrc_mic_receiver_runtime: VrcMicReceiverRuntime",
            "_vrc_receiver_lock:",
            "_last_vrc_mic_sync_enabled:",
            "_microphone_test_task",
        ),
        "src/puripuly_heart/ui/app.py": (
            "def _sync_github_star_prompt_runtime_aliases",
            "_github_star_prompt_launch_task",
        ),
        "src/puripuly_heart/app/services/settings/settings_runtime_effects.py": (
            "def _prepare_desktop_runtime_settings_update(",
            "def _sync_overlay_calibration_cache(",
            "def _sync_desktop_overlay_interaction_mode_from_settings(",
            "def _apply_desktop_size_preset_persistence_adjustment(",
            "def _broadcast_desktop_runtime_control_payloads(",
        ),
        "src/puripuly_heart/core/orchestrator/peer_translation_channel.py": (
            "_osc_flush_task",
            "def _run_osc_flush_loop",
        ),
        "src/puripuly_heart/core/runtime/local_stt_download.py": ("def adopt_legacy_state",),
        "src/puripuly_heart/core/runtime/clipboard.py": ("def adopt_legacy_state",),
        "src/puripuly_heart/core/runtime/mic_test.py": ("def adopt_legacy_state",),
        "src/puripuly_heart/core/runtime/output.py": ("_osc_flush_task",),
        "src/puripuly_heart/core/runtime/overlay.py": (
            "_overlay_presenter",
            "_overlay_bridge",
            "_overlay_manager",
            "_overlay_start_task",
            "_overlay_monitor_task",
            "_desktop_renderer_events",
            "_desktop_renderer_events_task",
        ),
    }

    offenders: list[str] = []
    for relative_path, forbidden_tokens in disallowed_private_shims.items():
        path = REPO_ROOT / relative_path
        if not path.exists():
            continue
        source = path.read_text(encoding="utf-8")
        offenders.extend(
            f"{relative_path}: {token}" for token in forbidden_tokens if token in source
        )

    composition_path = REPO_ROOT / "src/puripuly_heart/composition/application_runtime.py"
    composition_tree = ast.parse(composition_path.read_text(encoding="utf-8"))
    for node in ast.walk(composition_tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if node.name in controller_overlay_alias_shims:
                offenders.append(
                    f"src/puripuly_heart/composition/application_runtime.py: def {node.name} "
                    f"at line {node.lineno}"
                )
            continue
        if (
            isinstance(node, ast.Attribute)
            and node.attr in controller_overlay_alias_shims
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        ):
            offenders.append(
                "src/puripuly_heart/composition/application_runtime.py: "
                f"self.{node.attr} at line {node.lineno}"
            )
        if isinstance(node, ast.Assign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id in controller_overlay_alias_shims:
                    offenders.append(
                        "src/puripuly_heart/composition/application_runtime.py: "
                        f"field {target.id} at line {target.lineno}"
                    )

    assert not (
        REPO_ROOT / "src/puripuly_heart/app/services/provider_status_verification.py"
    ).exists()
    assert offenders == []


def test_absolute_from_import_resolves_layer_root_namespace_candidates(
    tmp_path: Path,
) -> None:
    importer_path = tmp_path / "importer.py"
    importer_path.write_text("from puripuly_heart import ui\n", encoding="utf-8")

    imported_modules = set(
        _imported_modules(
            "puripuly_heart.config.settings",
            importer_path,
            _internal_module_names(),
        )
    )

    assert "puripuly_heart.ui" in imported_modules
    assert _layer_for_module("puripuly_heart.ui") == UI_ADAPTERS_RENDERERS


def test_dependency_boundary_allowlist_matches_current_violations() -> None:
    actual = _dependency_violations()

    unexpected = sorted(actual - KNOWN_ALLOWED_VIOLATIONS)
    stale = sorted(KNOWN_ALLOWED_VIOLATIONS - actual)

    assert not unexpected and not stale, (
        "Dependency boundary allowlist mismatch. Add only current known exceptions "
        "to KNOWN_ALLOWED_VIOLATIONS, and remove entries as refactors eliminate them.\n"
        "Unexpected violations:\n"
        f"{_format_violations(unexpected)}\n"
        "Stale allowlist entries:\n"
        f"{_format_violations(stale)}"
    )


def test_r00_legacy_settings_reachability_census_matches_the_pinned_baseline() -> None:
    census_path = REPO_ROOT / "tests" / "architecture" / "settings_legacy_reachability_r00.json"
    census = json.loads(census_path.read_text(encoding="utf-8"))
    entries = census["entries"]
    assert census["baseline_sha"] == "a4aeacccf17194bbf607266f037d16e680234eef"
    assert len(entries) == 39
    assert len({entry["path"] for entry in entries}) == 39
    assert all(
        entry["symbols"] and entry["classifications"] and entry["owners"] for entry in entries
    )
    assert {classification for entry in entries for classification in entry["classifications"]} <= {
        "flat persistence ingress",
        "runtime AppSettings projection/mutation",
        "pure enum/constant/value misplaced in settings.py",
        "first-run/default policy",
        "test-only compatibility",
        "proven external compatibility",
    }
    assert {owner for entry in entries for owner in entry["owners"]} <= {
        "R00",
        "A02",
        "A04",
        "A06",
        "A07",
        "A08",
        "A09",
        "A10",
    }

    expected_current = {entry["path"]: set(entry["symbols"]) for entry in entries}
    expected_current.pop("src/puripuly_heart/main.py")
    expected_current.pop("src/puripuly_heart/providers/llm/openrouter.py")
    expected_current.pop("src/puripuly_heart/ui/desktop_overlay.py")
    expected_current.pop("src/puripuly_heart/ui/desktop_overlay_surface/contract.py")
    expected_current.pop("src/puripuly_heart/ui/desktop_overlay_surface/renderer.py")
    expected_current.pop("src/puripuly_heart/ui/views/settings.py")
    expected_current.pop("src/puripuly_heart/core/openrouter/managed_openrouter_broker_client.py")
    expected_current.pop("src/puripuly_heart/app/wiring/wiring_llm_factory.py")
    expected_current.pop("src/puripuly_heart/app/wiring/wiring_local_asr_provider_runtime.py")
    expected_current.pop("src/puripuly_heart/app/wiring/wiring_provider_runtime.py")
    expected_current.pop("src/puripuly_heart/app/wiring/wiring_provider_runtime_policy.py")
    expected_current.pop("src/puripuly_heart/app/wiring/wiring_runtime_pipeline.py")
    expected_current.pop("src/puripuly_heart/app/wiring/wiring_stt_factory.py")
    expected_current.pop("src/puripuly_heart/app/services/provider/provider_settings.py")
    expected_current.pop("src/puripuly_heart/app/wiring/wiring_capture_runtime.py")
    expected_current.pop("src/puripuly_heart/app/wiring/wiring_local_asr_application.py")
    expected_current.pop("src/puripuly_heart/app/wiring/wiring_microphone_test.py")
    expected_current.pop("src/puripuly_heart/app/wiring/wiring_overlay_factory.py")
    expected_current.pop("src/puripuly_heart/app/wiring/wiring_peer_application.py")
    expected_current.pop("src/puripuly_heart/app/services/capture/capture_target_settings.py")
    expected_current.pop(
        "src/puripuly_heart/app/services/capture/peer_capture_target_application.py"
    )
    expected_current["src/puripuly_heart/app/services/canonical_settings_persistence.py"] = {
        "AppSettings"
    }
    expected_current["src/puripuly_heart/app/services/settings/settings_application.py"] = {
        "AppSettings",
        "TranslationFallbackSettings",
        "materialize_translation_settings",
    }
    expected_current["src/puripuly_heart/app/services/openrouter_pkce_flow.py"] = {"AppSettings"}
    actual_current: dict[str, set[str]] = {}
    for source_path in SOURCE_PACKAGE_ROOT.rglob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        symbols = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "puripuly_heart.config.settings"
            for alias in node.names
        }
        if symbols:
            actual_current[_relative_repo_path(source_path)] = symbols

    assert actual_current == expected_current


def test_r00_retires_stable_profile_import_and_secret_copy_surfaces() -> None:
    forbidden = {
        "_extract_unknown_legacy_values",
        "_fallback_intent_from_legacy_raw_dict",
        "_migrate_legacy_local_qwen_providers",
        "_telemetry_consent_from_legacy_raw_dict",
        "_telemetry_state_from_legacy_raw_dict",
        "allow_stable_settings_import",
        "copy_stable_secrets_to_vnext_namespace",
        "import_stable_settings_if_missing",
        "is_legacy_shape_dict",
    }
    occurrences = {
        (path.relative_to(REPO_ROOT).as_posix(), symbol)
        for path in SOURCE_PACKAGE_ROOT.rglob("*.py")
        for symbol in forbidden
        if symbol in path.read_text(encoding="utf-8")
    }

    assert occurrences == set()


def test_a07_managed_broker_boundary_reduces_dependency_debt_to_16() -> None:
    assert not any(
        violation.importer.endswith("managed_openrouter_broker_client.py")
        for violation in KNOWN_ALLOWED_VIOLATIONS
    )
    assert len(KNOWN_SETTINGS_RUNTIME_CONFINEMENT_DEBT) == 2
    assert len(_settings_runtime_confinement_violations()) == 2


def test_a08_provider_runtime_wiring_reduces_dependency_debt_to_9() -> None:
    assert len(KNOWN_SETTINGS_RUNTIME_CONFINEMENT_DEBT) == 2
    assert len(_settings_runtime_confinement_violations()) == 2
    a08_paths = (
        SOURCE_PACKAGE_ROOT / "app" / "wiring" / "wiring_llm_factory.py",
        SOURCE_PACKAGE_ROOT / "app" / "wiring" / "wiring_local_asr_provider_runtime.py",
        SOURCE_PACKAGE_ROOT / "app" / "wiring" / "wiring_provider_runtime.py",
        SOURCE_PACKAGE_ROOT / "app" / "wiring" / "wiring_provider_runtime_policy.py",
        SOURCE_PACKAGE_ROOT / "app" / "wiring" / "wiring_runtime_pipeline.py",
        SOURCE_PACKAGE_ROOT / "app" / "wiring" / "wiring_stt_factory.py",
        SOURCE_PACKAGE_ROOT / "app" / "services" / "provider" / "provider_settings.py",
    )
    for path in a08_paths:
        imported_modules = set(
            _imported_modules(
                _module_name_for_path(path),
                path,
                _internal_module_names(),
            )
        )
        assert "puripuly_heart.config.settings" not in imported_modules
        assert "puripuly_heart.config.settings_vnext.migration" not in imported_modules
    assert {violation.importer for violation in KNOWN_ALLOWED_VIOLATIONS}.isdisjoint(
        {_relative_repo_path(path) for path in a08_paths}
    )


def test_a09_capture_overlay_application_wiring_reduces_dependency_debt_to_4() -> None:
    assert len(KNOWN_ALLOWED_VIOLATIONS) == 4
    assert len(_dependency_violations()) == 4
    assert len(KNOWN_SETTINGS_RUNTIME_CONFINEMENT_DEBT) == 2
    assert len(_settings_runtime_confinement_violations()) == 2
    a09_paths = (
        SOURCE_PACKAGE_ROOT / "app" / "wiring" / "wiring_capture_runtime.py",
        SOURCE_PACKAGE_ROOT / "app" / "wiring" / "wiring_local_asr_application.py",
        SOURCE_PACKAGE_ROOT / "app" / "wiring" / "wiring_microphone_test.py",
        SOURCE_PACKAGE_ROOT / "app" / "wiring" / "wiring_overlay_factory.py",
        SOURCE_PACKAGE_ROOT / "app" / "wiring" / "wiring_peer_application.py",
        SOURCE_PACKAGE_ROOT / "app" / "services" / "capture" / "capture_target_settings.py",
        SOURCE_PACKAGE_ROOT / "app" / "services" / "capture" / "peer_capture_target_application.py",
    )
    for path in a09_paths:
        imported_modules = set(
            _imported_modules(
                _module_name_for_path(path),
                path,
                _internal_module_names(),
            )
        )
        assert "puripuly_heart.config.settings" not in imported_modules
    assert {violation.importer for violation in KNOWN_ALLOWED_VIOLATIONS}.isdisjoint(
        {_relative_repo_path(path) for path in a09_paths}
    )


def test_a07_managed_broker_uses_the_pure_provider_value_owner() -> None:
    broker_path = (
        SOURCE_PACKAGE_ROOT / "core" / "openrouter" / "managed_openrouter_broker_client.py"
    )
    imported_modules = set(
        _imported_modules(
            _module_name_for_path(broker_path),
            broker_path,
            _internal_module_names(),
        )
    )

    assert "puripuly_heart.config.settings" not in imported_modules
    assert "puripuly_heart.config.provider_values" in imported_modules


def test_a04_overlay_ui_uses_the_pure_canonical_value_owner() -> None:
    owned_paths = (
        SOURCE_PACKAGE_ROOT / "ui" / "desktop_overlay.py",
        SOURCE_PACKAGE_ROOT / "ui" / "desktop_overlay_surface" / "contract.py",
        SOURCE_PACKAGE_ROOT / "ui" / "desktop_overlay_surface" / "renderer.py",
    )
    for path in owned_paths:
        imported_modules = set(
            _imported_modules(
                _module_name_for_path(path),
                path,
                _internal_module_names(),
            )
        )
        assert "puripuly_heart.config.settings" not in imported_modules
        assert "puripuly_heart.config.desktop_overlay_values" in imported_modules

    settings_view_path = SOURCE_PACKAGE_ROOT / "ui" / "views" / "settings.py"
    settings_view_tree = ast.parse(settings_view_path.read_text(encoding="utf-8"))
    imports_by_module = {
        node.module: {alias.name for alias in node.names}
        for node in ast.walk(settings_view_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    a04_symbols = {
        "DESKTOP_FLET_DEFAULT_BACKGROUND_ALPHA",
        "DESKTOP_FLET_SIZE_PRESET_DISPLAY_ORDER",
        "DESKTOP_FLET_SIZE_PRESET_ORDER",
    }
    assert a04_symbols <= imports_by_module["puripuly_heart.config.desktop_overlay_values"]
    assert a04_symbols.isdisjoint(imports_by_module.get("puripuly_heart.config.settings", set()))
    assert _layer_for_module("puripuly_heart.config.desktop_overlay_values") == SCHEMA_VALUES


def test_a05_settings_view_uses_composed_typed_secret_ports() -> None:
    settings_view_path = SOURCE_PACKAGE_ROOT / "ui" / "views" / "settings.py"
    settings_view_source = settings_view_path.read_text(encoding="utf-8")
    settings_view_tree = ast.parse(settings_view_source)
    imported_modules = set(
        _imported_modules(
            _module_name_for_path(settings_view_path),
            settings_view_path,
            _internal_module_names(),
        )
    )
    imported_secret_port_symbols = {
        alias.name
        for node in ast.walk(settings_view_tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "puripuly_heart.app.ports.settings_secrets"
        for alias in node.names
    }

    assert "puripuly_heart.app.wiring" not in imported_modules
    assert "create_secret_store" not in settings_view_source
    assert "SecretStore" not in settings_view_source
    assert {
        "SettingsSecretKey",
        "SettingsSecretMutation",
        "SettingsSecretSnapshot",
        "SettingsSecretsPort",
    } <= imported_secret_port_symbols

    composition_source = (SOURCE_PACKAGE_ROOT / "composition" / "application_runtime.py").read_text(
        encoding="utf-8"
    )
    assert "settings_secrets=SettingsSecretsOwner(" in composition_source
    assert "secret_store_factory=create_settings_secret_store" in composition_source


def test_r00_canonical_migration_loader_has_no_flat_ingress() -> None:
    migration_path = SOURCE_PACKAGE_ROOT / "config" / "settings_vnext" / "migration.py"
    tree = ast.parse(migration_path.read_text(encoding="utf-8"))
    loader = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "from_dict"
    )
    referenced_names = {node.id for node in ast.walk(loader) if isinstance(node, ast.Name)} | {
        node.attr for node in ast.walk(loader) if isinstance(node, ast.Attribute)
    }
    imported_modules = {
        node.module
        for node in ast.walk(loader)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {
        alias.name
        for node in ast.walk(loader)
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    assert "is_vnext_settings_dict" in referenced_names
    assert referenced_names.isdisjoint(
        {
            "_extract_unknown_legacy_values",
            "_migrate_settings_dict",
            "from_legacy_app_settings",
            "is_legacy_shape_dict",
        }
    )
    assert "puripuly_heart.config.settings" not in imported_modules


def test_dependency_boundary_allowlist_entries_have_gate6_rationale() -> None:
    assert set(KNOWN_ALLOWED_VIOLATION_GATE6_RATIONALES) == set(KNOWN_ALLOWED_VIOLATIONS)
    assert all(
        rationale and rationale != UNKNOWN_SETTINGS_RUNTIME_CONFINEMENT_RATIONALE
        for rationale in KNOWN_ALLOWED_VIOLATION_GATE6_RATIONALES.values()
    )


def test_settings_runtime_confinement_guard_tracks_current_debt() -> None:
    actual = _settings_runtime_confinement_violations()

    unexpected = sorted(actual - KNOWN_SETTINGS_RUNTIME_CONFINEMENT_DEBT)
    stale = sorted(KNOWN_SETTINGS_RUNTIME_CONFINEMENT_DEBT - actual)

    assert not unexpected and not stale, (
        "Settings runtime confinement guard mismatch. Legacy settings APIs, "
        "dynamic settings-shape reads, and legacy flat payload keys must be "
        "confined to compatibility/migration locations or listed as current "
        "order-11 debt with an explicit rationale.\n"
        "Unexpected violations:\n"
        f"{_format_settings_runtime_violations(unexpected)}\n"
        "Stale allowlist entries:\n"
        f"{_format_settings_runtime_violations(stale)}"
    )


def test_settings_runtime_confinement_debt_has_current_gate6_rationale() -> None:
    assert all(
        violation.rationale
        and "Gate 2" not in violation.rationale
        and "order-11" not in violation.rationale
        and "resolved by" not in violation.rationale.lower()
        for violation in KNOWN_SETTINGS_RUNTIME_CONFINEMENT_DEBT
    )


def test_settings_runtime_confinement_guard_flags_qualified_to_legacy_dict_usage() -> None:
    tree = ast.parse(
        "import puripuly_heart.config.settings_vnext.migration as migration\n"
        "migration.to_legacy_dict({})\n"
    )

    violations = _legacy_settings_api_import_violations(
        tree,
        "src/puripuly_heart/app/services/example.py",
    )
    violation_keys = {(item.category, item.path, item.symbol) for item in violations}

    assert (
        "legacy-settings-migration-projection",
        "src/puripuly_heart/app/services/example.py",
        "to_legacy_dict",
    ) in violation_keys
