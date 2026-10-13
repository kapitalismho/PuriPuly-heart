from __future__ import annotations

import os
import sys
from pathlib import Path

STABLE_APP_DIR_NAME = "puripuly-heart"
APP_DIR_NAME = STABLE_APP_DIR_NAME
SETTINGS_FILENAME = "settings.json"
MODELS_DIRNAME = "models"
HTTP_EXTENSIONS_DIRNAME = "http_extensions"


def test_profile_name() -> str | None:
    profile = os.getenv("PURIPULY_HEART_TEST_PROFILE")
    if not profile:
        return None
    if not profile.isascii() or not profile.replace("-", "").isalnum() or len(profile) > 64:
        raise ValueError("Invalid test profile name")
    return profile


def user_config_dir(*, app_dir_name: str = APP_DIR_NAME) -> Path:
    profile = test_profile_name()
    if profile is not None:
        directory = os.getenv("PURIPULY_HEART_TEST_DATA_DIR")
        if not directory or not Path(directory).is_absolute():
            raise ValueError("Test profile requires an absolute isolated data directory")
        return Path(directory)
    if sys.platform.startswith("win"):
        base = os.getenv("LOCALAPPDATA") or os.getenv("APPDATA")
        if base:
            return Path(base) / app_dir_name
        return Path.home() / "AppData" / "Local" / app_dir_name

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / app_dir_name

    base = os.getenv("XDG_CONFIG_HOME")
    if base:
        return Path(base) / app_dir_name
    return Path.home() / ".config" / app_dir_name


def default_settings_path() -> Path:
    return user_config_dir() / SETTINGS_FILENAME


def default_models_dir() -> Path:
    return user_config_dir() / MODELS_DIRNAME


def default_http_extensions_dir() -> Path:
    return user_config_dir() / HTTP_EXTENSIONS_DIRNAME
