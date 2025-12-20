"""Prompt file loader utility.

Loads system prompts from files in the prompts/ directory.
"""

from __future__ import annotations

import os
from pathlib import Path

import sys


def get_prompts_dir() -> Path:
    """Get the prompts directory path."""
    # PyInstaller frozen app: use _MEIPASS
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        meipass_prompts = Path(sys._MEIPASS) / "prompts"
        if meipass_prompts.exists():
            return meipass_prompts
    
    # Try relative to the project root first
    candidates = [
        Path(__file__).parent.parent.parent.parent / "prompts",  # src/ajebal.../config -> project root
        Path.cwd() / "prompts",
        Path(__file__).parent / "prompts",
    ]
    
    for path in candidates:
        if path.exists():
            return path
    
    # Default: relative to cwd
    return Path.cwd() / "prompts"


def list_prompts() -> list[str]:
    """List available prompt file names (without extension)."""
    prompts_dir = get_prompts_dir()
    if not prompts_dir.exists():
        return []
    
    return sorted([
        f.stem for f in prompts_dir.glob("*.txt")
    ])


def load_prompt(name: str = "default") -> str:
    """Load a prompt from file.
    
    Args:
        name: Prompt file name (without .txt extension)
        
    Returns:
        Prompt content, or empty string if not found
    """
    prompts_dir = get_prompts_dir()
    prompt_file = prompts_dir / f"{name}.txt"
    
    if prompt_file.exists():
        return prompt_file.read_text(encoding="utf-8").strip()
    
    # Fallback to default
    default_file = prompts_dir / "default.txt"
    if default_file.exists():
        return default_file.read_text(encoding="utf-8").strip()
    
    return ""


def get_default_prompt() -> str:
    """Load the default prompt."""
    return load_prompt("default")


def load_prompt_for_provider(provider: str) -> str:
    """Load the prompt for a specific LLM provider.

    Args:
        provider: Provider name ('gemini' or 'qwen')

    Returns:
        Prompt content for the provider, or default if not found
    """
    provider_lower = provider.lower()
    prompts_dir = get_prompts_dir()

    # Try provider-specific prompt first
    prompt_file = prompts_dir / f"{provider_lower}.txt"
    if prompt_file.exists():
        return prompt_file.read_text(encoding="utf-8").strip()

    # Fallback to default
    return load_prompt("default")
