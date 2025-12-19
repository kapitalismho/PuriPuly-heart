"""Prompt file loader utility.

Loads system prompts from files in the prompts/ directory.
"""

from __future__ import annotations

import os
from pathlib import Path


def get_prompts_dir() -> Path:
    """Get the prompts directory path."""
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
