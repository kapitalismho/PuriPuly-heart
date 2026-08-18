"""Prompt file loader utility.

Loads system prompts from files in the prompts/ directory.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

TRANSLATION_PROMPT_NAME = "translation_prompt"
_LLM_PROVIDER_PROMPT_KEYS = {
    "gemini",
    "qwen",
    "deepseek",
    "openrouter",
    "local_llm",
    "managed_gemma",
}


@dataclass(frozen=True)
class PromptAssemblyCache:
    """Cached prompt pieces used to assemble translation prompts."""

    template: str
    target_language_rules: Mapping[str, str]
    language_pair_examples: Mapping[str, str]
    fallback_examples: str


_PROMPT_CACHE: PromptAssemblyCache | None = None

_ENGLISH_KEY = "english"
_FALLBACK_EXAMPLES_KEY = "fallback"
_TARGET_LANGUAGE_RULE_KEYS = {"english", "japanese", "korean"}
_CHINESE_SIMPLIFIED_KEYS = {
    "chinese-simplified",
    "simplified-chinese",
    "zh-cn",
    "zh-hans",
    "zh-sg",
}
_CHINESE_TRADITIONAL_KEYS = {
    "chinese-traditional",
    "traditional-chinese",
    "zh-tw",
    "zh-hant",
    "zh-hk",
    "zh-mo",
}
_CHINESE_BASE_KEYS = {"chinese", "zh", "mandarin"}
_CHINESE_SIMPLIFIED_MARKERS = {"simplified", "hans", "cn", "sg"}
_CHINESE_TRADITIONAL_MARKERS = {
    "traditional",
    "hant",
    "tw",
    "taiwan",
    "hk",
    "hong-kong",
    "mo",
    "macau",
}


def get_prompts_dir() -> Path:
    """Get the prompts directory path."""
    env_dir = os.getenv("PURIPULY_HEART_PROMPTS_DIR")
    if env_dir:
        env_path = Path(env_dir)
        if env_path.exists():
            return env_path

    # PyInstaller frozen app: use _MEIPASS
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        meipass_prompts = Path(sys._MEIPASS) / "prompts"
        if meipass_prompts.exists():
            return meipass_prompts

    # Try relative to the project root first
    candidates = [
        Path(__file__).parent.parent.parent.parent
        / "prompts",  # src/puripuly_heart.../config -> project root
        Path.cwd() / "prompts",
        Path(__file__).parent / "prompts",
    ]

    for path in candidates:
        if path.exists():
            return path

    # Walk up from cwd to find project root (pyproject.toml) with prompts/
    for parent in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        candidate = parent / "prompts"
        if (parent / "pyproject.toml").exists() and candidate.exists():
            return candidate

    # Walk up from cwd to find any prompts/ directory (e.g., when running from .venv)
    for parent in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        candidate = parent / "prompts"
        if candidate.exists():
            return candidate

    # Default: relative to cwd
    return Path.cwd() / "prompts"


def list_prompts() -> list[str]:
    """List available prompt file names (without extension)."""
    prompts_dir = get_prompts_dir()
    if not prompts_dir.exists():
        return []

    return sorted([f.stem for f in prompts_dir.glob("*.md")])


def load_prompt(name: str = "default") -> str:
    """Load a prompt from file.

    Args:
        name: Prompt file name (without .txt extension)

    Returns:
        Prompt content, or empty string if not found
    """
    prompts_dir = get_prompts_dir()

    # Try .md first
    prompt_file = prompts_dir / f"{name}.md"
    if prompt_file.exists():
        return prompt_file.read_text(encoding="utf-8").strip()

    # Fallback to .txt
    prompt_file = prompts_dir / f"{name}.txt"
    if prompt_file.exists():
        return prompt_file.read_text(encoding="utf-8").strip()

    # Fallback to default
    default_file = prompts_dir / "default.md"
    if default_file.exists():
        return default_file.read_text(encoding="utf-8").strip()

    # Legacy default.txt
    default_file = prompts_dir / "default.txt"
    if default_file.exists():
        return default_file.read_text(encoding="utf-8").strip()

    return ""


def _read_prompt_text(path: Path) -> str:
    """Read and normalize prompt text from a prompt file."""
    return path.read_text(encoding="utf-8").strip()


def _load_markdown_files(directory: Path) -> dict[str, str]:
    """Load all Markdown files in a directory keyed by filename stem."""
    if not directory.exists():
        return {}

    return {
        prompt_file.stem: _read_prompt_text(prompt_file)
        for prompt_file in sorted(directory.glob("*.md"))
        if prompt_file.is_file()
    }


def _load_prompt_cache() -> PromptAssemblyCache:
    """Load all translation-prompt assembly files into an immutable cache."""
    prompts_dir = get_prompts_dir()
    template_path = prompts_dir / f"{TRANSLATION_PROMPT_NAME}.md"
    if not template_path.exists():
        raise FileNotFoundError(f"Required translation prompt not found: {template_path}")

    target_language_rules = _load_markdown_files(prompts_dir / "prompt-rules" / "target-language")
    language_pair_examples = _load_markdown_files(prompts_dir / "prompt-examples" / "language-pair")

    return PromptAssemblyCache(
        template=_read_prompt_text(template_path),
        target_language_rules=MappingProxyType(target_language_rules),
        language_pair_examples=MappingProxyType(language_pair_examples),
        fallback_examples=language_pair_examples.get(_FALLBACK_EXAMPLES_KEY, ""),
    )


def _get_prompt_cache() -> PromptAssemblyCache:
    """Return the process-wide translation prompt assembly cache."""
    global _PROMPT_CACHE
    if _PROMPT_CACHE is None:
        _PROMPT_CACHE = _load_prompt_cache()
    return _PROMPT_CACHE


def warm_prompt_cache() -> None:
    """Preload translation prompt files into the process cache."""
    _get_prompt_cache()


def _reset_prompt_cache_for_tests() -> None:
    """Reset cached prompt pieces for tests that swap prompt directories."""
    global _PROMPT_CACHE
    _PROMPT_CACHE = None


def _normalize_language_name(language_name: str) -> str:
    """Normalize a display language name into a prompt filename key."""
    normalized = language_name.strip().lower()
    for old, new in (("_", "-"), ("/", "-"), ("(", " "), (")", " ")):
        normalized = normalized.replace(old, new)
    normalized = "-".join(normalized.split())
    while "--" in normalized:
        normalized = normalized.replace("--", "-")
    return normalized.strip("-")


def _resolve_language_pair_key(language_name: str) -> str:
    """Resolve a language display name to a language-pair example key."""
    language_key = _normalize_language_name(language_name)
    language_tokens = set(language_key.split("-"))
    if language_key in _CHINESE_SIMPLIFIED_KEYS:
        return "chinese-simplified"
    if language_key in _CHINESE_TRADITIONAL_KEYS:
        return "chinese-traditional"
    if language_key in _CHINESE_BASE_KEYS:
        return "chinese"
    if (
        "chinese" in language_tokens
        or "mandarin" in language_tokens
        or language_key.startswith("zh-")
    ):
        if any(marker in language_key for marker in _CHINESE_TRADITIONAL_MARKERS):
            return "chinese-traditional"
        if any(marker in language_tokens for marker in _CHINESE_SIMPLIFIED_MARKERS):
            return "chinese-simplified"
        return "chinese"
    return language_key


def _resolve_target_language_rules_key(target_name: str) -> str:
    """Resolve a target language display name to a target-rule filename key."""
    language_key = _resolve_language_pair_key(target_name)
    if language_key in _TARGET_LANGUAGE_RULE_KEYS:
        return language_key
    if language_key in {"chinese", "chinese-simplified", "chinese-traditional"}:
        return "chinese"
    return ""


def _select_translation_examples(
    cache: PromptAssemblyCache, source_name: str, target_name: str
) -> str:
    """Select exact language-pair examples or target-English fallback examples."""
    source_key = _resolve_language_pair_key(source_name)
    target_key = _resolve_language_pair_key(target_name)
    exact_key = f"{source_key}-to-{target_key}"
    exact_examples = cache.language_pair_examples.get(exact_key)
    if exact_examples is not None:
        return exact_examples
    if target_key == _ENGLISH_KEY:
        return cache.fallback_examples
    return ""


def build_translation_prompt_variables(source_name: str, target_name: str) -> dict[str, str]:
    """Build dynamic variables for rendering the shared translation prompt."""
    cache = _get_prompt_cache()
    target_rules_key = _resolve_target_language_rules_key(target_name)
    target_language_rules = (
        cache.target_language_rules.get(target_rules_key, "") if target_rules_key else ""
    )

    return {
        "sourceName": source_name,
        "targetName": target_name,
        "targetLanguageRules": target_language_rules,
        "translationExamples": _select_translation_examples(cache, source_name, target_name),
    }


def render_translation_prompt_template(template: str, *, source_name: str, target_name: str) -> str:
    """Render the shared translation prompt template with dynamic variables."""
    rendered = template
    for key, value in build_translation_prompt_variables(source_name, target_name).items():
        rendered = rendered.replace(f"${{{key}}}", value)
    return rendered


def get_translation_prompt_template() -> str:
    """Load the shared translation prompt template."""
    return _get_prompt_cache().template


def get_default_prompt() -> str:
    """Load the default prompt."""
    return get_translation_prompt_template()


def load_prompt_for_provider(provider: str) -> str:
    """Load the prompt for a specific LLM provider.

    Args:
        provider: Provider name ('gemini' or 'qwen')

    Returns:
        Prompt content for the provider, or default if not found
    """
    provider_lower = provider.lower()
    if provider_lower in _LLM_PROVIDER_PROMPT_KEYS:
        return get_translation_prompt_template()

    return load_prompt(provider_lower)
