from __future__ import annotations

import ast
import asyncio
import base64
import importlib
import inspect
import json
import re
import shutil
import textwrap
from collections.abc import Iterator, Mapping
from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest

from puripuly_heart.app import wiring
from puripuly_heart.app.wiring import root as wiring_root
from puripuly_heart.config import llm_profiles, runtime_resolution
from puripuly_heart.config import prompts as prompts_module
from puripuly_heart.config import resolved as resolved_config
from puripuly_heart.config.prompts import (
    TRANSLATION_PROMPT_NAME,
    load_prompt,
    load_prompt_for_provider,
)
from puripuly_heart.config.settings import (
    AppSettings,
    OpenRouterCredentialSource,
    SecretsBackend,
    SecretsSettings,
    to_dict,
)
from puripuly_heart.core import (
    managed_identity,
    openrouter_credentials,
)
from puripuly_heart.core import managed_openrouter_broker_client as broker_client
from puripuly_heart.core.openrouter.managed_openrouter_release import (
    ManagedOpenRouterChallengeSuccess,
    ManagedOpenRouterDiscordStartSuccess,
    ManagedOpenRouterFingerprintSalt,
    ManagedOpenRouterIssueSuccess,
    ManagedOpenRouterTrialStatusSuccess,
    ManagedOpenRouterVerifySuccess,
    TalkTogetherPassStatus,
)
from puripuly_heart.core.overlay import manifest as overlay_manifest_module
from puripuly_heart.core.overlay import process as overlay_process_module
from puripuly_heart.core.overlay.manifest import OVERLAY_CONTRACT_VERSION
from puripuly_heart.core.overlay.protocol import (
    OverlayPresentationBlock,
    OverlayPresentationCalibration,
    OverlayPresentationSnapshot,
)
from puripuly_heart.core.storage.secrets import (
    EncryptedFileSecretStore,
    InMemorySecretStore,
    KeyringSecretStore,
)
from tests.config.settings_migration_fixtures import (
    maximal_v24_settings_fixture,
    serialized_field_paths,
)
from tests.helpers.paths import REPO_ROOT

SOURCE_PACKAGE_ROOT = REPO_ROOT / "src" / "puripuly_heart"
PACKAGE_NAME = "puripuly_heart"
SNAPSHOT_PATH = Path(__file__).with_name("public_compatibility_surfaces_snapshot.json")
INVENTORY_PATH = Path(__file__).with_name("compatibility_surface_inventory.json")

REQUIRED_SURFACES = (
    "secret_store",
    "broker_v1",
    "overlay",
    "installer_identity",
    "prompts",
    "provider_aliases",
    "provider_runtime_public_config",
    "guard_coverage",
    "blockers",
)
REQUIRED_SOURCE_PUBLIC_COMPATIBILITY_SURFACES = {
    "SecretStore keys": ("secret_store",),
    "Broker /v1": ("broker_v1",),
    "overlay protocol and startup contract": ("overlay",),
    "prompt fallback behavior": ("prompt_loader",),
    "provider aliases": ("provider_aliases",),
    "i18n key parity": ("i18n_parity",),
    "installer identity": ("packaging",),
}
REQUIRED_INVENTORY_SURFACES = (
    "public_import_facades",
    "settings_runtime_compatibility",
    "persisted_operational_state",
    "secret_store",
    "provider_aliases",
    "provider_runtime_public_config",
    "prompt_fallback",
    "broker_v1",
    "overlay_protocol_startup_snapshot",
    "i18n_key_parity",
    "installer_identity",
    "rust_overlay_startup",
)
REQUIRED_INVENTORY_SOURCE_RULES = {
    "__all__",
    "lazy __getattr__",
    "documented entry point",
    "existing public test or packaging import",
    "approved facade",
}
DOCUMENTED_ENTRY_POINT_OR_TEST_IMPORT_MODULES = frozenset(
    {
        "puripuly_heart.app.wiring",
        "puripuly_heart.config.llm_profiles",
        "puripuly_heart.config.prompts",
        "puripuly_heart.config.settings",
        "puripuly_heart.core.managed_identity",
        "puripuly_heart.core.managed_openrouter_broker_client",
        "puripuly_heart.core.managed_openrouter_release",
        "puripuly_heart.core.openrouter_credentials",
        "puripuly_heart.core.overlay.manifest",
        "puripuly_heart.core.overlay.protocol",
        "puripuly_heart.core.storage.secrets",
        "puripuly_heart.domain.events",
        "puripuly_heart.domain.models",
        "puripuly_heart.main",
        "puripuly_heart.providers.llm.deepseek",
        "puripuly_heart.providers.llm.gemini",
        "puripuly_heart.providers.llm.local_openai",
        "puripuly_heart.providers.llm.openrouter",
        "puripuly_heart.providers.llm.qwen",
        "puripuly_heart.providers.llm.qwen_async",
        "puripuly_heart.providers.stt.deepgram",
        "puripuly_heart.providers.stt.local_qwen_sherpa",
        "puripuly_heart.providers.stt.qwen_asr",
        "puripuly_heart.providers.stt.soniox",
        "puripuly_heart.ui.event_bridge",
    }
)
REQUIRED_SECRET_KEYS = (
    "google_api_key",
    "openrouter_api_key",
    "openrouter_managed_api_key",
    "openrouter_managed_qq_api_key",
    "openrouter_managed_user_id",
    "openrouter_managed_user_installation_id",
    "deepseek_api_key",
    "cerebras_api_key",
    "deepgram_api_key",
    "soniox_api_key",
    "alibaba_api_key_beijing",
    "alibaba_api_key_singapore",
    "alibaba_api_key",
    "local_llm_api_key",
    "managed_device_private_key",
    "managed_device_public_key",
    "managed_identity_binding",
)
WIRING_SECRET_KEYS = (
    "google_api_key",
    "deepseek_api_key",
    "deepgram_api_key",
    "soniox_api_key",
    "alibaba_api_key_beijing",
    "alibaba_api_key_singapore",
    "alibaba_api_key",
    "local_llm_api_key",
)
EXPECTED_SECRET_ENV_LOOKUP_PATHS = (
    {
        "owner": "gemini_llm",
        "lookup": "require_secret",
        "key": "google_api_key",
        "env_vars": ["GOOGLE_API_KEY"],
        "legacy_keys": [],
    },
    {
        "owner": "openrouter_byok_llm",
        "lookup": "openrouter_byok",
        "key": "openrouter_api_key",
        "env_vars": ["OPENROUTER_API_KEY"],
        "legacy_keys": [],
    },
    {
        "owner": "deepseek_llm",
        "lookup": "require_secret",
        "key": "deepseek_api_key",
        "env_vars": ["DEEPSEEK_API_KEY"],
        "legacy_keys": [],
    },
    {
        "owner": "qwen_beijing_llm_stt_peer_stt",
        "lookup": "require_secret_any",
        "key": "alibaba_api_key_beijing",
        "env_vars": ["ALIBABA_API_KEY_BEIJING", "ALIBABA_API_KEY", "DASHSCOPE_API_KEY"],
        "legacy_keys": ["alibaba_api_key"],
    },
    {
        "owner": "qwen_singapore_llm_stt_peer_stt",
        "lookup": "require_secret_any",
        "key": "alibaba_api_key_singapore",
        "env_vars": ["ALIBABA_API_KEY_SINGAPORE", "ALIBABA_API_KEY", "DASHSCOPE_API_KEY"],
        "legacy_keys": ["alibaba_api_key"],
    },
    {
        "owner": "deepgram_stt_peer_stt",
        "lookup": "require_secret",
        "key": "deepgram_api_key",
        "env_vars": ["DEEPGRAM_API_KEY"],
        "legacy_keys": [],
    },
    {
        "owner": "soniox_stt_peer_stt",
        "lookup": "require_secret",
        "key": "soniox_api_key",
        "env_vars": ["SONIOX_API_KEY"],
        "legacy_keys": [],
    },
    {
        "owner": "local_llm_optional",
        "lookup": "optional_secret_store_only",
        "key": "local_llm_api_key",
        "env_vars": [],
        "legacy_keys": [],
        "ignored_env_vars": ["LOCAL_LLM_API_KEY"],
    },
)
EXPECTED_PROMPT_FALLBACK_ORDER = (
    "{name}.md",
    "{name}.txt",
    "default.md",
    "default.txt",
)
EXPECTED_ENCRYPTED_FILE_PASSPHRASE = "fixture-passphrase-not-secret"


def _load_snapshot() -> dict[str, Any]:
    return json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))


def _load_inventory() -> dict[str, Any]:
    return json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))


def _render_command_template(parts: list[str], **values: str) -> tuple[str, ...]:
    return tuple(part.format(**values) for part in parts)


def _inno_define_literals(script: str) -> dict[str, str]:
    return dict(re.findall(r'^\s*#define\s+([A-Za-z0-9_]+)\s+"([^"]*)"', script, re.M))


def _powershell_string_variable(script: str, variable_name: str) -> str:
    match = re.search(rf'^\s*\${re.escape(variable_name)}\s*=\s+"([^"]*)"', script, re.M)
    assert match is not None, variable_name
    return match.group(1)


def _test_function_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }


def _module_name_for_source_path(path: Path) -> str:
    relative = path.relative_to(SOURCE_PACKAGE_ROOT).with_suffix("")
    parts = (PACKAGE_NAME, *relative.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _is_non_private_source_module(path: Path) -> bool:
    relative = path.relative_to(SOURCE_PACKAGE_ROOT)
    parts = relative.with_suffix("").parts
    return not any(part.startswith("_") and part != "__init__" for part in parts)


def _module_public_export_signals(path: Path) -> frozenset[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    signals: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
        ):
            try:
                exported_names = ast.literal_eval(node.value)
            except (SyntaxError, ValueError):
                signals.add("__all__")
                continue
            if exported_names:
                signals.add("__all__")
        elif (
            isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == "__getattr__"
        ):
            signals.add("lazy __getattr__")
    return frozenset(signals)


def _relevant_non_private_inventory_modules() -> dict[str, frozenset[str]]:
    relevant: dict[str, frozenset[str]] = {}
    for path in SOURCE_PACKAGE_ROOT.rglob("*.py"):
        if not _is_non_private_source_module(path):
            continue

        module_name = _module_name_for_source_path(path)
        signals = set(_module_public_export_signals(path))
        if module_name in DOCUMENTED_ENTRY_POINT_OR_TEST_IMPORT_MODULES:
            signals.add("documented/test import")
        if signals:
            relevant[module_name] = frozenset(signals)
    return relevant


def _inventory_accounts_for_module(inventory: dict[str, Any], module_name: str) -> bool:
    public_modules = {entry["module"] for entry in inventory["public_imports"]}
    if module_name in public_modules:
        return True

    for classification in inventory["non_public_module_classifications"]:
        if module_name in classification.get("modules", []):
            return True
        if any(
            module_name == prefix or module_name.startswith(f"{prefix}.")
            for prefix in classification.get("module_prefixes", [])
        ):
            return True
    return False


def _leaf_values(value: object) -> Iterator[object]:
    if isinstance(value, dict):
        for child in value.values():
            yield from _leaf_values(child)
        return
    if isinstance(value, list):
        for child in value:
            yield from _leaf_values(child)
        return
    yield value


def _assert_source_ref_exists(ref: str) -> None:
    path_text = ref.split("::", maxsplit=1)[0]
    assert path_text, ref
    assert (REPO_ROOT / path_text).exists(), ref


def _assert_inventory_export_resolves(module_name: str, export: dict[str, str]) -> None:
    export_name = export["name"]
    if export["kind"] == "submodule":
        importlib.import_module(f"{module_name}.{export_name}")
        return

    module = importlib.import_module(module_name)
    assert hasattr(module, export_name), f"{module_name}.{export_name}"


def _clear_entry_env(monkeypatch: pytest.MonkeyPatch, entry: dict[str, Any]) -> None:
    for env_var in [*entry["env_vars"], *entry.get("ignored_env_vars", [])]:
        monkeypatch.delenv(env_var, raising=False)


def _broker_source(relative_path: str) -> str:
    return (REPO_ROOT / "broker" / "src" / relative_path).read_text(encoding="utf-8")


def _broker_app_v1_routes() -> tuple[dict[str, str], ...]:
    app_source = _broker_source("app.ts")
    return tuple(
        {"method": method.upper(), "path": path}
        for method, path in re.findall(r"app\.(get|post)\('(/v1/[^']+)'", app_source)
    )


def _typescript_string_union_literals(source: str, type_name: str) -> tuple[str, ...]:
    match = re.search(rf"export type {re.escape(type_name)} =(?P<body>.*?);", source, re.S)
    assert match is not None, type_name
    return tuple(re.findall(r"'([^']+)'", match.group("body")))


def _typescript_interface_fields(source: str, interface_name: str) -> tuple[str, ...]:
    match = re.search(rf"interface {re.escape(interface_name)} \{{(?P<body>.*?)\n\}}", source, re.S)
    assert match is not None, interface_name
    return tuple(re.findall(r"^\s+([A-Za-z_][A-Za-z0-9_]*)\??:", match.group("body"), re.M))


def _typescript_const_object_keys(source: str, const_name: str) -> tuple[str, ...]:
    match = re.search(
        rf"export const {re.escape(const_name)} = \{{(?P<body>.*?)\n\}} as const;",
        source,
        re.S,
    )
    assert match is not None, const_name
    return tuple(re.findall(r"^\s{2}([A-Za-z_][A-Za-z0-9_]*):", match.group("body"), re.M))


def _matching_delimiter_index(
    source: str,
    open_index: int,
    *,
    open_char: str = "{",
    close_char: str = "}",
) -> int:
    depth = 0
    quote: str | None = None
    escaped = False
    for index in range(open_index, len(source)):
        char = source[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"', "`"}:
            quote = char
        elif char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return index
    raise AssertionError(f"unmatched {open_char!r} in TypeScript source")


def _balanced_delimited_body(
    source: str,
    open_index: int,
    *,
    open_char: str = "{",
    close_char: str = "}",
) -> str:
    close_index = _matching_delimiter_index(
        source,
        open_index,
        open_char=open_char,
        close_char=close_char,
    )
    return source[open_index + 1 : close_index]


def _typescript_function_source(source: str, function_name: str) -> str:
    match = re.search(rf"\bfunction\s+{re.escape(function_name)}\s*\(", source)
    assert match is not None, function_name
    parameter_open_index = match.end() - 1
    parameter_close_index = _matching_delimiter_index(
        source,
        parameter_open_index,
        open_char="(",
        close_char=")",
    )
    open_index = source.find("{", parameter_close_index)
    assert open_index != -1, function_name
    return source[open_index : _matching_delimiter_index(source, open_index) + 1]


def _managed_state_response_fields() -> tuple[str, ...]:
    return _typescript_return_object_fields(
        _broker_source("managed-state.ts"),
        "normalizeManagedState",
    )


def _top_level_object_fields(
    object_body: str,
    *,
    spread_expansions: Mapping[str, tuple[str, ...]] | None = None,
) -> tuple[str, ...]:
    expansions = spread_expansions or {}
    lines = [line for line in object_body.splitlines() if line.strip()]
    if not lines:
        return ()
    root_indent = min(len(line) - len(line.lstrip(" ")) for line in lines)
    fields_: list[str] = []
    for line in lines:
        if len(line) - len(line.lstrip(" ")) != root_indent:
            continue
        stripped = line.strip().rstrip(",")
        if stripped.startswith("..."):
            expanded = False
            for spread_name, field_names in expansions.items():
                if re.match(rf"\.\.\.{re.escape(spread_name)}(?:\b|\()", stripped):
                    fields_.extend(field_names)
                    expanded = True
                    break
            if not expanded:
                fields_.extend(re.findall(r"\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*:", stripped))
            continue
        key_match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*:", stripped)
        if key_match:
            fields_.append(key_match.group(1))
            continue
        shorthand_match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)$", stripped)
        if shorthand_match:
            fields_.append(shorthand_match.group(1))
    return tuple(fields_)


def _typescript_c_json_object_fields(source: str, function_name: str) -> tuple[str, ...]:
    function_source = _typescript_function_source(source, function_name)
    call_index = function_source.find("c.json(")
    assert call_index != -1, function_name
    object_index = function_source.find("{", call_index)
    assert object_index != -1, function_name
    object_body = _balanced_delimited_body(function_source, object_index)
    return _top_level_object_fields(
        object_body,
        spread_expansions={"normalizeManagedState": _managed_state_response_fields()},
    )


def _typescript_return_object_fields(
    source: str,
    function_name: str,
    *,
    spread_expansions: Mapping[str, tuple[str, ...]] | None = None,
) -> tuple[str, ...]:
    function_source = _typescript_function_source(source, function_name)
    return_index = function_source.find("return")
    assert return_index != -1, function_name
    object_index = function_source.find("{", return_index)
    assert object_index != -1, function_name
    object_body = _balanced_delimited_body(function_source, object_index)
    return _top_level_object_fields(object_body, spread_expansions=spread_expansions)


def _typescript_c_json_nested_object_fields(
    source: str,
    function_name: str,
    object_key: str,
) -> tuple[str, ...]:
    function_source = _typescript_function_source(source, function_name)
    call_index = function_source.find("c.json(")
    assert call_index != -1, function_name
    object_index = function_source.find("{", call_index)
    assert object_index != -1, function_name
    object_body = _balanced_delimited_body(function_source, object_index)
    key_match = re.search(rf"\b{re.escape(object_key)}\s*:\s*\{{", object_body)
    assert key_match is not None, object_key
    nested_open_index = object_body.find("{", key_match.start())
    nested_body = _balanced_delimited_body(object_body, nested_open_index)
    return _top_level_object_fields(nested_body)


def _broker_success_response_fields(source_ref: Mapping[str, str]) -> tuple[str, ...]:
    source = _broker_source(source_ref["file"])
    kind = source_ref["kind"]
    if kind == "const_object":
        return _typescript_const_object_keys(source, source_ref["name"])
    if kind == "c_json_object":
        return _typescript_c_json_object_fields(source, source_ref["function"])
    if kind == "return_object":
        return _typescript_return_object_fields(
            source,
            source_ref["function"],
            spread_expansions={"managedState": _managed_state_response_fields()},
        )
    raise AssertionError(f"unknown Broker success response source kind: {kind!r}")


def _return_dict_keys(function: object) -> tuple[str, ...]:
    source = textwrap.dedent(inspect.getsource(function))
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            return tuple(
                key.value
                for key in node.value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            )
    raise AssertionError(f"{function!r} does not return a dict literal")


def _client_keyword_request_fields(function: object) -> tuple[tuple[str, ...], tuple[str, ...]]:
    required: list[str] = []
    optional: list[str] = []
    for name, parameter in inspect.signature(function).parameters.items():
        if name == "self":
            continue
        if parameter.kind is not inspect.Parameter.KEYWORD_ONLY:
            continue
        target = optional if parameter.default is not inspect.Parameter.empty else required
        target.append(name)
    return tuple(required), tuple(optional)


def _dataclass_field_names(cls: type[object]) -> tuple[str, ...]:
    return tuple(field.name for field in fields(cls))


def _assert_required_secret_env_lookup(
    monkeypatch: pytest.MonkeyPatch,
    entry: dict[str, Any],
) -> None:
    key = entry["key"]
    env_vars = entry["env_vars"]
    assert len(env_vars) == 1
    env_var = env_vars[0]
    _clear_entry_env(monkeypatch, entry)
    monkeypatch.setenv(env_var, "fake-env-secret")

    assert wiring.require_secret(InMemorySecretStore(), key=key, env_var=env_var) == (
        "fake-env-secret"
    )

    store = InMemorySecretStore()
    store.set(key, "fake-store-secret")
    assert wiring.require_secret(store, key=key, env_var=env_var) == "fake-store-secret"


def _assert_required_any_secret_env_lookup(
    monkeypatch: pytest.MonkeyPatch,
    entry: dict[str, Any],
) -> None:
    key = entry["key"]
    env_vars = tuple(entry["env_vars"])
    legacy_keys = tuple(entry["legacy_keys"])

    store = InMemorySecretStore()
    store.set(key, "fake-store-secret")
    for legacy_key in legacy_keys:
        store.set(legacy_key, "fake-legacy-secret")
    for env_var in env_vars:
        monkeypatch.setenv(env_var, f"fake-{env_var}")
    assert (
        wiring.require_secret_any(
            store,
            key=key,
            env_vars=env_vars,
            legacy_keys=legacy_keys,
        )
        == "fake-store-secret"
    )

    legacy_store = InMemorySecretStore()
    for legacy_key in legacy_keys:
        legacy_store.set(legacy_key, "fake-legacy-secret")
    assert (
        wiring.require_secret_any(
            legacy_store,
            key=key,
            env_vars=env_vars,
            legacy_keys=legacy_keys,
        )
        == "fake-legacy-secret"
    )
    assert legacy_store.get(key) == "fake-legacy-secret"

    for selected_env_var in env_vars:
        _clear_entry_env(monkeypatch, entry)
        monkeypatch.setenv(selected_env_var, f"fake-{selected_env_var}")
        assert (
            wiring.require_secret_any(
                InMemorySecretStore(),
                key=key,
                env_vars=env_vars,
                legacy_keys=legacy_keys,
            )
            == f"fake-{selected_env_var}"
        )


def _assert_openrouter_byok_env_lookup(
    monkeypatch: pytest.MonkeyPatch,
    entry: dict[str, Any],
) -> None:
    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.BYOK
    env_var = entry["env_vars"][0]
    _clear_entry_env(monkeypatch, entry)
    monkeypatch.setenv(env_var, "fake-env-openrouter")

    credential_config = openrouter_credentials.OpenRouterCredentialRuntimeConfig(
        selected_source=settings.openrouter.selected_source,
        installation_id=settings.managed_identity.installation_id,
    )
    env_resolution = openrouter_credentials.resolve_openrouter_credentials(
        credential_config,
        secrets=InMemorySecretStore(),
    )
    assert env_resolution.api_key == "fake-env-openrouter"

    store = InMemorySecretStore()
    store.set(entry["key"], "fake-store-openrouter")
    store_resolution = openrouter_credentials.resolve_openrouter_credentials(
        credential_config, secrets=store
    )
    assert store_resolution.api_key == "fake-store-openrouter"


def _assert_local_llm_optional_secret_store_only(entry: dict[str, Any]) -> None:
    llm_factory = importlib.import_module("puripuly_heart.app.wiring.wiring_llm_factory")
    source = "\n".join(
        (
            inspect.getsource(wiring.create_llm_provider),
            inspect.getsource(wiring._base_llm_provider_from_resolved_config),
            inspect.getsource(llm_factory._provider_from_resolved_target),
        )
    )

    assert entry["env_vars"] == []
    assert entry["ignored_env_vars"] == ["LOCAL_LLM_API_KEY"]
    assert 'secrets.get("local_llm_api_key")' in source
    assert "LOCAL_LLM_API_KEY" not in source


def _assert_qwen_resolved_credential_helper_preserves_legacy_fallbacks() -> None:
    llm_factory = importlib.import_module("puripuly_heart.app.wiring.wiring_llm_factory")
    helper_source = inspect.getsource(wiring._qwen_api_key_for_resolved_credential)
    base_llm_source = inspect.getsource(wiring._base_llm_provider_from_resolved_config)
    provider_target_source = inspect.getsource(llm_factory._provider_from_resolved_target)
    stt_source = inspect.getsource(wiring.create_stt_backend_from_resolved_config)
    peer_resolved_source = inspect.getsource(wiring.create_peer_stt_backend_from_resolved_config)

    assert 'key="alibaba_api_key_beijing"' in helper_source
    assert 'key="alibaba_api_key_singapore"' in helper_source
    assert '"ALIBABA_API_KEY_BEIJING"' in helper_source
    assert '"ALIBABA_API_KEY_SINGAPORE"' in helper_source
    assert '"ALIBABA_API_KEY"' in helper_source
    assert '"DASHSCOPE_API_KEY"' in helper_source
    assert helper_source.count('legacy_keys=("alibaba_api_key",)') == 2
    assert "_provider_from_resolved_target(" in base_llm_source
    assert "_qwen_api_key_for_resolved_credential(target.credential" in provider_target_source
    assert "_qwen_api_key_for_resolved_credential(config.credential" in stt_source
    assert "create_stt_backend_from_resolved_config(" in peer_resolved_source


def test_public_compatibility_snapshot_declares_all_required_surfaces() -> None:
    snapshot = _load_snapshot()

    assert tuple(snapshot) == REQUIRED_SURFACES
    assert snapshot["blockers"] == []


def test_gate_zero_compatibility_inventory_declares_required_surfaces() -> None:
    inventory = _load_inventory()

    assert inventory["bundle"]["ref"] == "vnext-internal-cutover"
    assert inventory["bundle"]["sha256"] == (
        "033912b4bf580cd5d488066dd9139c3ff0896d962df94e5de405bb3c694dd1b6"
    )
    assert inventory["source_spec"]["sha256"] == (
        "42e4a691eacdbbc495eb5ede54c0c3511b1c0c78b67159ead94d2bad35c0c1f9"
    )
    assert tuple(inventory["compatibility_surfaces"]) == REQUIRED_INVENTORY_SURFACES
    assert inventory["ambiguous_public_private_surfaces"] == []

    for surface_name, surface in inventory["compatibility_surfaces"].items():
        assert surface["classification"] == "public_compatibility_preserve", surface_name
        assert surface["source_of_truth_refs"], surface_name
        assert surface["fixture_or_guard_refs"], surface_name
        for ref in [*surface["source_of_truth_refs"], *surface["fixture_or_guard_refs"]]:
            _assert_source_ref_exists(ref)


def test_public_import_inventory_smoke_imports_every_exported_name() -> None:
    public_imports = _load_inventory()["public_imports"]

    assert public_imports
    for entry in public_imports:
        module_name = entry["module"]
        source_rule = entry["source_rule"]
        exports = entry["exports"]

        assert source_rule in REQUIRED_INVENTORY_SOURCE_RULES, module_name
        assert exports, module_name
        for ref in entry["source_of_truth_refs"]:
            _assert_source_ref_exists(ref)

        module = importlib.import_module(module_name)
        if entry.get("runtime_all_matches_exports"):
            assert tuple(module.__all__) == tuple(export["name"] for export in exports)
        for export in exports:
            _assert_inventory_export_resolves(module_name, export)


def test_public_facade_inventory_classifies_thin_delegate_targets() -> None:
    inventory = _load_inventory()
    public_imports = inventory["public_imports"]
    facade_entries = [entry for entry in public_imports if entry["facade_contract"]]

    assert facade_entries
    for entry in facade_entries:
        contract = entry["facade_contract"]
        assert contract["preserve_import_path"] is True, entry["module"]
        assert contract["target_role"] in {
            "thin_delegate",
            "thin_reexport",
            "lazy_reexport",
            "compatibility_boundary_owner",
        }, entry["module"]
        assert contract["implementation_owner"] in {
            "canonical_vnext_owner",
            "compatibility_boundary",
            "adapter_owner_until_split",
            "public_contract_owner",
        }, entry["module"]

    for classification in inventory["non_public_module_classifications"]:
        assert classification["classification"] != "ambiguous", classification
        assert classification["rationale"], classification


def test_relevant_non_private_modules_are_inventoried_or_classified() -> None:
    inventory = _load_inventory()
    relevant_modules = _relevant_non_private_inventory_modules()

    missing = {
        module_name: sorted(signals)
        for module_name, signals in sorted(relevant_modules.items())
        if not _inventory_accounts_for_module(inventory, module_name)
    }

    assert missing == {}


def test_guard_coverage_references_existing_tests_for_every_surface() -> None:
    coverage = _load_snapshot()["guard_coverage"]
    expected_surfaces = {
        "secret_store",
        "broker_v1",
        "overlay",
        "prompt_loader",
        "provider_aliases",
        "provider_runtime_public_config",
        "i18n_parity",
        "packaging",
    }

    assert set(coverage) == expected_surfaces
    for surface, refs in coverage.items():
        assert refs, surface
        for ref in refs:
            file_name, separator, test_name = ref.partition("::")
            assert separator == "::", ref
            test_path = REPO_ROOT / file_name
            assert test_path.is_file(), ref
            assert test_name in _test_function_names(test_path), ref


def test_source_named_public_compatibility_surfaces_have_guard_evidence() -> None:
    snapshot = _load_snapshot()
    coverage = snapshot["guard_coverage"]
    inventory = _load_inventory()["compatibility_surfaces"]

    assert REQUIRED_SOURCE_PUBLIC_COMPATIBILITY_SURFACES.keys() == {
        "SecretStore keys",
        "Broker /v1",
        "overlay protocol and startup contract",
        "prompt fallback behavior",
        "provider aliases",
        "i18n key parity",
        "installer identity",
    }
    assert "prompt_fallback" in inventory
    assert "installer_identity" in inventory
    assert inventory["prompt_fallback"]["fixture_or_guard_refs"]
    assert inventory["installer_identity"]["fixture_or_guard_refs"]

    for source_surface, coverage_surfaces in REQUIRED_SOURCE_PUBLIC_COMPATIBILITY_SURFACES.items():
        refs = [ref for surface in coverage_surfaces for ref in coverage[surface]]
        assert refs, source_surface
        for ref in refs:
            file_name, separator, test_name = ref.partition("::")
            assert separator == "::", ref
            assert test_name in _test_function_names(REPO_ROOT / file_name), ref


def test_guard_coverage_includes_overlay_rust_and_installer_freeze_refs() -> None:
    coverage = _load_snapshot()["guard_coverage"]

    assert (
        "tests/config/test_public_compatibility_surfaces.py::"
        "test_overlay_startup_contract_snapshot_matches_python_runners_and_manifest_handoff"
        in coverage["overlay"]
    )
    assert (
        "tests/config/test_public_compatibility_surfaces.py::"
        "test_rust_overlay_startup_contract_snapshot_matches_native_sources" in coverage["overlay"]
    )
    assert (
        "tests/config/test_public_compatibility_surfaces.py::"
        "test_installer_identity_snapshot_matches_inno_and_smoke_guard_contract"
        in coverage["packaging"]
    )


def test_secret_store_key_registry_snapshot_matches_current_public_keys() -> None:
    snapshot = _load_snapshot()["secret_store"]
    registry_keys = tuple(snapshot["registry_keys"])

    assert registry_keys == REQUIRED_SECRET_KEYS
    assert KeyringSecretStore().service_name == snapshot["keyring_service_name"]
    assert openrouter_credentials.OPENROUTER_BYOK_API_KEY_SECRET == "openrouter_api_key"
    assert openrouter_credentials.OPENROUTER_MANAGED_API_KEY_SECRET == (
        "openrouter_managed_api_key"
    )
    assert openrouter_credentials.OPENROUTER_MANAGED_QQ_API_KEY_SECRET == (
        "openrouter_managed_qq_api_key"
    )
    assert openrouter_credentials.OPENROUTER_MANAGED_USER_ID_SECRET == (
        "openrouter_managed_user_id"
    )
    assert openrouter_credentials.OPENROUTER_MANAGED_USER_INSTALLATION_ID_SECRET == (
        "openrouter_managed_user_installation_id"
    )
    assert managed_identity.MANAGED_DEVICE_PRIVATE_KEY_SECRET == "managed_device_private_key"
    assert managed_identity.MANAGED_DEVICE_PUBLIC_KEY_SECRET == "managed_device_public_key"
    assert managed_identity.MANAGED_IDENTITY_BINDING_SECRET == "managed_identity_binding"

    wiring_source = inspect.getsource(wiring_root)
    for key in WIRING_SECRET_KEYS:
        assert f'"{key}"' in wiring_source
    _assert_qwen_resolved_credential_helper_preserves_legacy_fallbacks()


def test_secret_store_env_lookup_snapshot_matches_current_fallback_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = tuple(_load_snapshot()["secret_store"]["env_lookup_paths"])

    assert entries == EXPECTED_SECRET_ENV_LOOKUP_PATHS
    for entry in entries:
        _clear_entry_env(monkeypatch, entry)
        lookup = entry["lookup"]
        if lookup == "require_secret":
            _assert_required_secret_env_lookup(monkeypatch, entry)
        elif lookup == "require_secret_any":
            _assert_required_any_secret_env_lookup(monkeypatch, entry)
        elif lookup == "openrouter_byok":
            _assert_openrouter_byok_env_lookup(monkeypatch, entry)
        elif lookup == "optional_secret_store_only":
            _assert_local_llm_optional_secret_store_only(entry)
        else:  # pragma: no cover - snapshot guard should fail before this branch matters.
            raise AssertionError(f"unknown SecretStore lookup snapshot type: {lookup!r}")


@pytest.mark.parametrize(
    "current_key",
    ["alibaba_api_key_beijing", "alibaba_api_key_singapore"],
)
def test_secret_store_accepts_legacy_alibaba_key_and_backfills_current_key(
    current_key: str,
) -> None:
    store = InMemorySecretStore()
    store.set("alibaba_api_key", "fake-legacy-alibaba-key")

    value = wiring.require_secret_any(
        store,
        key=current_key,
        env_vars=(),
        legacy_keys=("alibaba_api_key",),
    )

    assert value == "fake-legacy-alibaba-key"
    assert store.get(current_key) == "fake-legacy-alibaba-key"


def test_settings_serialization_excludes_secret_store_registry_keys() -> None:
    registry_keys = set(_load_snapshot()["secret_store"]["registry_keys"])
    serialized_settings = (
        to_dict(AppSettings()),
        maximal_v24_settings_fixture(),
    )

    for data in serialized_settings:
        paths = set(serialized_field_paths(data))
        forbidden_paths = sorted(
            path
            for path in paths
            if not path.startswith("api_key_verified.")
            and any(part in registry_keys for part in path.split("."))
        )
        string_values = {value for value in _leaf_values(data) if isinstance(value, str)}

        assert forbidden_paths == []
        assert registry_keys.isdisjoint(string_values)


def test_secret_store_encrypted_file_fixture_decrypts_and_freezes_wire_format(
    tmp_path: Path,
) -> None:
    encrypted_snapshot = _load_snapshot()["secret_store"]["encrypted_file"]
    fixture_path = REPO_ROOT / encrypted_snapshot["golden_fixture"]
    raw_fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

    assert tuple(raw_fixture) == ("version", "salt", "items")
    assert raw_fixture["version"] == encrypted_snapshot["version"] == 1
    assert raw_fixture["salt"] == encrypted_snapshot["salt_b64"]
    assert len(base64.b64decode(raw_fixture["salt"])) == encrypted_snapshot["salt_bytes"] == 16
    assert tuple(raw_fixture["items"]) == tuple(encrypted_snapshot["item_keys"])
    assert all(
        isinstance(token, str) and token.startswith("gAAAAA")
        for token in raw_fixture["items"].values()
    )

    rendered_fixture = json.dumps(raw_fixture, ensure_ascii=False)
    for raw_secret in encrypted_snapshot["expected_fake_values"].values():
        assert raw_secret not in rendered_fixture

    store = EncryptedFileSecretStore(
        fixture_path,
        passphrase=encrypted_snapshot["passphrase"],
    )
    assert {key: store.get(key) for key in encrypted_snapshot["item_keys"]} == encrypted_snapshot[
        "expected_fake_values"
    ]

    wrong = EncryptedFileSecretStore(fixture_path, passphrase="wrong-fixture-passphrase")
    with pytest.raises(ValueError, match="invalid passphrase"):
        wrong.get(encrypted_snapshot["item_keys"][0])

    working_path = tmp_path / "secrets.json"
    shutil.copyfile(fixture_path, working_path)
    working = EncryptedFileSecretStore(working_path, passphrase=encrypted_snapshot["passphrase"])
    working.set("deepseek_api_key", "fixture-deepseek-api-key")
    updated = json.loads(working_path.read_text(encoding="utf-8"))

    assert updated["version"] == raw_fixture["version"]
    assert updated["salt"] == raw_fixture["salt"]
    assert set(updated["items"]) == {*encrypted_snapshot["item_keys"], "deepseek_api_key"}
    assert "fixture-deepseek-api-key" not in json.dumps(updated, ensure_ascii=False)
    assert working.get("deepseek_api_key") == "fixture-deepseek-api-key"


def test_secret_store_encrypted_file_path_resolution_and_env_passphrase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config" / "settings.json"
    config_path.parent.mkdir()
    settings = SecretsSettings(
        backend=SecretsBackend.ENCRYPTED_FILE,
        encrypted_file_path="fixture-secrets.json",
    )

    monkeypatch.setenv(wiring.SECRETS_PASSPHRASE_ENV, EXPECTED_ENCRYPTED_FILE_PASSPHRASE)
    store = wiring.create_secret_store(settings, config_path=config_path)

    assert isinstance(store, EncryptedFileSecretStore)
    assert store.path == config_path.parent / "fixture-secrets.json"
    raw = json.loads(store.path.read_text(encoding="utf-8"))
    assert tuple(raw) == ("version", "salt", "items")
    assert raw["version"] == 1
    assert isinstance(raw["salt"], str)
    assert raw["items"] == {}

    monkeypatch.delenv(wiring.SECRETS_PASSPHRASE_ENV, raising=False)
    with pytest.raises(ValueError, match=wiring.SECRETS_PASSPHRASE_ENV):
        wiring.create_secret_store(settings, config_path=config_path)


def test_broker_v1_snapshot_matches_client_paths_and_public_error_vocabulary() -> None:
    snapshot = _load_snapshot()["broker_v1"]
    client_source = inspect.getsource(broker_client.HttpManagedOpenRouterBrokerClient)
    path_literals = tuple(sorted(set(re.findall(r'path="([^"]+)"', client_source))))
    broker_error_source = _broker_source("broker-error.ts")

    assert path_literals == tuple(snapshot["paths"])
    assert all(path.startswith("/v1/") for path in path_literals)
    assert "/v2/" not in inspect.getsource(broker_client)
    assert tuple(snapshot["source_routes"]) == _broker_app_v1_routes()
    assert set(path_literals).issubset({route["path"] for route in snapshot["source_routes"]})
    assert managed_identity.DISCORD_OPENROUTER_ISSUE_PATH == (
        "/v1/providers/openrouter/discord/issue"
    )
    assert tuple(sorted(broker_client.PUBLIC_ERROR_CODES)) == tuple(snapshot["public_error_codes"])
    assert tuple(sorted(broker_client.PUBLIC_ERROR_CLASSES)) == tuple(
        snapshot["public_error_classes"]
    )
    assert tuple(snapshot["public_error_codes"]) == tuple(
        sorted(_typescript_string_union_literals(broker_error_source, "PublicErrorCode"))
    )
    assert tuple(snapshot["public_error_classes"]) == tuple(
        sorted(_typescript_string_union_literals(broker_error_source, "PublicErrorClass"))
    )


def test_broker_v1_snapshot_freezes_request_success_and_error_envelopes() -> None:
    snapshot = _load_snapshot()["broker_v1"]
    operations = snapshot["operations"]
    source_routes = {(route["method"], route["path"]) for route in snapshot["source_routes"]}
    operation_routes = {
        (operation["method"], operation["path"]) for operation in operations.values()
    }
    expected_operation_names = (
        "foundation",
        "challenge",
        "discord_start",
        "qq_auth_assert",
        "verify",
        "issue",
        "discord_issue",
        "trial_status",
        "managed_key_delivery_ack",
        "telemetry_translation_success_day",
    )

    assert tuple(operations) == expected_operation_names
    assert source_routes - operation_routes == set()
    assert tuple(snapshot.get("source_route_operation_exclusions", ())) == ()
    for operation_name, operation in operations.items():
        assert "success_response_source" in operation, operation_name
    assert "source" in snapshot["error_envelope"]

    broker_contract_source = _broker_source("contract.ts")
    assert tuple(operations["foundation"]["query_fields"]) == ()
    assert tuple(operations["foundation"]["header_fields"]) == ()
    assert tuple(operations["foundation"]["success_response_fields"]) == (
        _typescript_const_object_keys(broker_contract_source, "FOUNDATION_RESPONSE")
    )

    challenge_required, challenge_optional = _client_keyword_request_fields(
        broker_client.HttpManagedOpenRouterBrokerClient.challenge
    )
    assert tuple(operations["challenge"]["request_body_fields"]) == challenge_required
    assert tuple(operations["challenge"]["optional_request_body_fields"]) == challenge_optional
    assert tuple(operations["challenge"]["client_success_fields"]) == _dataclass_field_names(
        ManagedOpenRouterChallengeSuccess
    )

    discord_required, discord_optional = _client_keyword_request_fields(
        broker_client.HttpManagedOpenRouterBrokerClient.start_discord_oauth
    )
    assert tuple(operations["discord_start"]["request_body_fields"]) == discord_required
    assert tuple(operations["discord_start"]["optional_request_body_fields"]) == discord_optional
    assert tuple(operations["discord_start"]["client_success_fields"]) == _dataclass_field_names(
        ManagedOpenRouterDiscordStartSuccess
    )

    qq_auth_source = _broker_source("qq-auth.ts")
    assert tuple(operations["qq_auth_assert"]["request_body_fields"]) == (
        _typescript_interface_fields(qq_auth_source, "QqAuthAssertRequestBody")
    )

    assert tuple(operations["verify"]["request_body_fields"]) == _return_dict_keys(
        managed_identity.ManagedIdentityBundle.sign_verify_request
    )
    assert tuple(operations["verify"]["client_success_fields"]) == _dataclass_field_names(
        ManagedOpenRouterVerifySuccess
    )

    assert tuple(operations["issue"]["request_body_fields"]) == _return_dict_keys(
        managed_identity.ManagedIdentityBundle.sign_issue_request
    )
    assert tuple(operations["issue"]["client_success_fields"]) == _dataclass_field_names(
        ManagedOpenRouterIssueSuccess
    )

    assert tuple(operations["discord_issue"]["request_body_fields"]) == _return_dict_keys(
        managed_identity.ManagedIdentityBundle.sign_discord_issue_request
    )
    assert operations["discord_issue"]["path"] == managed_identity.DISCORD_OPENROUTER_ISSUE_PATH
    assert tuple(operations["discord_issue"]["client_success_fields"]) == _dataclass_field_names(
        ManagedOpenRouterIssueSuccess
    )

    trial_status_required, trial_status_optional = _client_keyword_request_fields(
        broker_client.HttpManagedOpenRouterBrokerClient.get_trial_status
    )
    assert trial_status_required == ("installation_id", "timestamp", "signature")
    assert trial_status_optional == ()
    assert tuple(operations["trial_status"]["query_fields"]) == ("installation_id",)
    assert tuple(operations["trial_status"]["header_fields"]) == (
        "X-Puripuly-Timestamp",
        "X-Puripuly-Signature",
    )
    assert tuple(operations["trial_status"]["client_success_fields"]) == _dataclass_field_names(
        ManagedOpenRouterTrialStatusSuccess
    )

    for operation_name, operation in operations.items():
        assert operation["path"] in {route["path"] for route in snapshot["source_routes"]}
        assert tuple(operation["success_response_fields"]) == _broker_success_response_fields(
            operation["success_response_source"]
        ), operation_name

    nested_shapes = snapshot["nested_shapes"]
    assert tuple(nested_shapes["fingerprint_salt"]) == _dataclass_field_names(
        ManagedOpenRouterFingerprintSalt
    )
    assert tuple(nested_shapes["talk_together_pass"]) == _dataclass_field_names(
        TalkTogetherPassStatus
    )
    assert tuple(nested_shapes["managed_state"]) == ("lifecycle", "managed_availability")
    assert tuple(nested_shapes["current_entitlement"]) == (
        "provider",
        "budget_usd",
        "issued_at",
        "expires_at",
    )

    broker_error_source = _broker_source("broker-error.ts")
    assert tuple(snapshot["error_envelope"]["top_level_fields"]) == (
        _typescript_c_json_object_fields(
            _broker_source(snapshot["error_envelope"]["source"]["file"]),
            snapshot["error_envelope"]["source"]["function"],
        )
    )
    assert tuple(snapshot["error_envelope"]["error_fields"]) == tuple(
        _typescript_c_json_nested_object_fields(
            broker_error_source,
            snapshot["error_envelope"]["source"]["function"],
            "error",
        )
    )


def test_overlay_contract_snapshot_matches_manifest_and_protocol_wire_shape() -> None:
    snapshot = _load_snapshot()["overlay"]
    block = OverlayPresentationBlock(
        id="self:1",
        occupant_key="self:1",
        appearance_seq=1,
        channel="self",
        block_variant="finalized",
        primary_text="hello",
        secondary_text="안녕",
        secondary_enabled=True,
        primary_language="en",
        secondary_language="ko",
        update_id="update-1",
        origin_wall_clock_ms=1712345678901,
        session_scope="session:self",
        source_text_hash="abc123",
        source_text_len=5,
        logical_turn_key="self:1",
    )
    presentation_snapshot = OverlayPresentationSnapshot(blocks=[block]).to_dict()

    assert OVERLAY_CONTRACT_VERSION == snapshot["contract_version"]
    assert tuple(sorted(overlay_manifest_module._MANIFEST_FIELDS)) == tuple(
        snapshot["manifest_fields"]
    )
    assert tuple(presentation_snapshot) == tuple(snapshot["presentation_snapshot_fields"])
    assert tuple(OverlayPresentationCalibration().to_dict()) == tuple(
        snapshot["calibration_fields"]
    )
    assert tuple(presentation_snapshot["blocks"][0]) == tuple(snapshot["presentation_block_fields"])
    assert tuple(snapshot["channels"]) == ("self", "peer")
    assert tuple(snapshot["block_variants"]) == ("active_self", "active_peer", "finalized")


@pytest.mark.asyncio
async def test_overlay_startup_contract_snapshot_matches_python_runners_and_manifest_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup = _load_snapshot()["overlay"]["startup_contract"]
    manifest_path = tmp_path / "overlay-manifest.json"
    python_executable = tmp_path / "python.exe"
    app_executable = tmp_path / "PuriPulyHeart.exe"
    overlay_executable = tmp_path / "PuriPulyHeartOverlay.exe"

    source_runner = overlay_process_module.DesktopFletOverlayRunner(
        frozen=False,
        python_executable=python_executable,
    )
    frozen_runner = overlay_process_module.DesktopFletOverlayRunner(
        frozen=True,
        app_executable=app_executable,
    )

    assert source_runner.build_command(manifest_path) == _render_command_template(
        startup["desktop_source_runner_command"],
        python_executable=str(python_executable),
        manifest_path=str(manifest_path),
    )
    assert frozen_runner.build_command(manifest_path) == _render_command_template(
        startup["desktop_frozen_runner_command"],
        app_executable=str(app_executable),
        manifest_path=str(manifest_path),
    )

    captured_command: tuple[str, ...] | None = None
    captured_stdio: tuple[object, object] | None = None

    class FakeSubprocess:
        stdout = None
        stderr = None
        returncode = 0

        async def wait(self) -> int:
            return 0

        def terminate(self) -> None:
            self.returncode = 0

        def kill(self) -> None:
            self.returncode = -9

    async def fake_create_subprocess_exec(
        *args: str,
        stdout: object | None = None,
        stderr: object | None = None,
        env: dict[str, str] | None = None,
    ) -> FakeSubprocess:
        nonlocal captured_command, captured_stdio
        captured_command = tuple(args)
        captured_stdio = (stdout, stderr)
        assert env is not None
        assert env[overlay_process_module.QUIET_TAIL_PROFILE_ENV] == "p05"
        return FakeSubprocess()

    monkeypatch.setattr(
        overlay_process_module.asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    await overlay_process_module.DefaultOverlayProcessRunner().spawn(
        overlay_executable,
        manifest_path,
    )

    assert captured_command == _render_command_template(
        startup["native_runner_command"],
        overlay_executable=str(overlay_executable),
        manifest_path=str(manifest_path),
    )
    assert captured_stdio == (asyncio.subprocess.PIPE, asyncio.subprocess.PIPE)

    manifest = overlay_manifest_module.OverlayLaunchManifest(
        contract_version=OVERLAY_CONTRACT_VERSION,
        app_version="test",
        overlay_instance_id="overlay-test",
        bridge_url="ws://127.0.0.1:8765",
        session_token="session-token",
        parent_pid=1234,
        startup_deadline_ms=3000,
        log_dir="logs",
        log_level="INFO",
        locale="en",
        logging_mode="basic",
    )
    manager = overlay_process_module.OverlayProcessManager()
    written_manifest_path = manager._write_manifest(manifest)
    try:
        assert written_manifest_path.name.startswith(startup["manifest_temp_prefix"])
        assert written_manifest_path.name.endswith(startup["manifest_temp_suffix"])
        assert json.loads(written_manifest_path.read_text(encoding="utf-8")) == manifest.to_dict()
    finally:
        written_manifest_path.unlink(missing_ok=True)

    manager_handler_source = inspect.getsource(
        overlay_process_module.OverlayProcessManager._handle_lifecycle_event
    )
    for key in (
        "ready_event_type",
        "startup_failure_event_type",
        "runtime_failure_event_type",
        "renderer_event_type",
    ):
        assert f'"{startup[key]}"' in manager_handler_source

    default_spawn_source = inspect.getsource(
        overlay_process_module.DefaultOverlayProcessRunner.spawn
    )
    desktop_spawn_source = inspect.getsource(overlay_process_module.DesktopFletOverlayRunner.spawn)
    assert startup["explicit_env_overrides"] == [overlay_process_module.QUIET_TAIL_PROFILE_ENV]
    assert "env=child_env" in default_spawn_source
    assert "env=" not in desktop_spawn_source


def test_rust_overlay_startup_contract_snapshot_matches_native_sources() -> None:
    rust_startup = _load_snapshot()["overlay"]["rust_startup_behavior"]
    manifest_source = (REPO_ROOT / "native" / "overlay" / "src" / "manifest.rs").read_text(
        encoding="utf-8"
    )
    runtime_source = (REPO_ROOT / "native" / "overlay" / "src" / "runtime.rs").read_text(
        encoding="utf-8"
    )
    runtime_tests = (REPO_ROOT / "native" / "overlay" / "tests" / "runtime.rs").read_text(
        encoding="utf-8"
    )
    startup_check_match = re.search(
        r'args\[1\] == "--check-startup-contract".*?json!\(\{(?P<body>.*?)\}\)',
        runtime_source,
        re.S,
    )
    assert startup_check_match is not None
    startup_output_fields = tuple(
        re.findall(r'"([A-Za-z0-9_]+)"\s*:', startup_check_match.group("body"))
    )

    assert (
        f"pub const EXPECTED_CONTRACT_VERSION: u32 = {rust_startup['expected_contract_version']};"
        in manifest_source
    )
    assert f'.arg("{rust_startup["startup_check_arg"]}")' in runtime_tests
    assert tuple(rust_startup["startup_check_output_fields"]) == startup_output_fields
    assert 'payload["contract_version"]' in runtime_tests
    assert f'event["type"] == "{rust_startup["startup_error_event_type"]}"' in runtime_tests
    assert (
        rust_startup["startup_error_event_type"]
        == _load_snapshot()["overlay"]["startup_contract"]["startup_failure_event_type"]
    )


def test_installer_identity_snapshot_matches_inno_and_smoke_guard_contract() -> None:
    snapshot = _load_snapshot()["installer_identity"]
    installer_script = (REPO_ROOT / "installer.iss").read_text(encoding="utf-8")
    release_script = (REPO_ROOT / "scripts" / "ci" / "build-release-artifacts.ps1").read_text(
        encoding="utf-8"
    )
    defines = _inno_define_literals(installer_script)

    for name, value in snapshot["defines"].items():
        assert defines[name] == value
    for setup_line in snapshot["setup_lines"]:
        assert setup_line in installer_script

    assert snapshot["production_installer_build"] in release_script
    assert (
        _powershell_string_variable(release_script, "InstallerTestAppId")
        == snapshot["smoke_alternate_app_id"]
    )
    for guard_line in snapshot["smoke_guard_lines"]:
        assert guard_line in release_script


def test_prompt_loader_snapshot_freezes_fallback_order_and_translation_prompt_requirement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _load_snapshot()["prompts"]
    prompt_name = "surface"
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    files = {
        f"{prompt_name}.md": "name-md",
        f"{prompt_name}.txt": "name-txt",
        "default.md": "default-md",
        "default.txt": "default-txt",
    }
    for file_name, content in files.items():
        (prompts_dir / file_name).write_text(content, encoding="utf-8")
    monkeypatch.setenv("PURIPULY_HEART_PROMPTS_DIR", str(prompts_dir))

    assert snapshot["translation_prompt_name"] == TRANSLATION_PROMPT_NAME
    assert tuple(snapshot["load_prompt_fallback_order"]) == EXPECTED_PROMPT_FALLBACK_ORDER
    assert tuple(snapshot["llm_provider_prompt_keys"]) == tuple(
        sorted(prompts_module._LLM_PROVIDER_PROMPT_KEYS)
    )
    assert load_prompt(prompt_name) == "name-md"
    (prompts_dir / f"{prompt_name}.md").unlink()
    assert load_prompt(prompt_name) == "name-txt"
    (prompts_dir / f"{prompt_name}.txt").unlink()
    assert load_prompt(prompt_name) == "default-md"
    (prompts_dir / "default.md").unlink()
    assert load_prompt(prompt_name) == "default-txt"
    (prompts_dir / "default.txt").unlink()
    assert load_prompt(prompt_name) == ""

    with pytest.raises(FileNotFoundError):
        load_prompt_for_provider("gemini")


def test_provider_alias_snapshot_matches_current_aliases_and_legacy_acceptance() -> None:
    snapshot = _load_snapshot()["provider_aliases"]

    assert tuple(snapshot["openrouter_main_selection_aliases"]) == (
        llm_profiles.OPENROUTER_MAIN_SELECTION_ALIASES
    )
    assert tuple(snapshot["openrouter_fallback_selection_aliases"]) == (
        llm_profiles.OPENROUTER_FALLBACK_SELECTION_ALIASES
    )
    assert tuple(snapshot["legacy_selection_aliases"]) == tuple(
        sorted(llm_profiles.LEGACY_PROFILE_BY_ALIAS)
    )
    assert snapshot["legacy_fallback_aliases"] == llm_profiles.LEGACY_FALLBACK_ALIAS_TO_ALIAS

    for alias in snapshot["legacy_selection_aliases"]:
        assert llm_profiles.get_openrouter_llm_profile(alias) is not None
    for legacy_alias, canonical_alias in snapshot["legacy_fallback_aliases"].items():
        assert llm_profiles.normalize_openrouter_fallback_selection_alias(legacy_alias) == (
            canonical_alias
        )


def test_provider_runtime_public_config_snapshot_matches_resolved_contracts() -> None:
    snapshot = _load_snapshot()["provider_runtime_public_config"]

    assert tuple(snapshot["runtime_resolution_input_fields"]) == tuple(
        field.name for field in fields(runtime_resolution.RuntimeResolutionInput)
    )
    assert tuple(snapshot["translation_runtime_intent_fields"]) == tuple(
        field.name for field in fields(runtime_resolution.TranslationRuntimeIntent)
    )
    assert tuple(snapshot["openrouter_runtime_intent_fields"]) == tuple(
        field.name for field in fields(runtime_resolution.OpenRouterRuntimeIntent)
    )
    assert tuple(snapshot["resolved_llm_config_fields"]) == tuple(
        field.name for field in fields(resolved_config.ResolvedLLMConfig)
    )
    assert tuple(snapshot["resolved_credential_requirement_fields"]) == tuple(
        field.name for field in fields(resolved_config.ResolvedCredentialRequirement)
    )
    assert tuple(snapshot["credential_sources"]) == resolved_config.CREDENTIAL_SOURCES
    assert snapshot["current_credential_source_key"] == "selected_source"
    assert tuple(snapshot["old_credential_source_keys"]) == (
        "credential_source",
        "selected_credential_source",
    )
    assert snapshot["legacy_alias_normalization_boundary"] == (
        "normalize_openrouter_runtime_intent"
    )
    assert hasattr(runtime_resolution, snapshot["legacy_alias_normalization_boundary"])

    resolved_llm_fields = {field.name for field in fields(resolved_config.ResolvedLLMConfig)}
    for field_name in snapshot["resolved_runtime_forbidden_alias_fields"]:
        assert field_name not in resolved_llm_fields
