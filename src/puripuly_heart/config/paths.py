from __future__ import annotations

import os
import sys
from pathlib import Path

APP_DIR_NAME = "puripuly-heart"
SETTINGS_FILENAME = "settings.json"
VAD_MODEL_FILENAME = "silero_vad.onnx"
MODELS_DIRNAME = "models"

_DATA_DIR_ENV = "PURIPULY_HEART_DATA_DIR"


def is_portable() -> bool:
    """Return True when running in portable mode (data dir env var is set)."""
    return _DATA_DIR_ENV in os.environ


def portable_data_dir() -> Path:
    """Return the portable data directory from the environment variable."""
    return Path(os.environ[_DATA_DIR_ENV])


def user_config_dir(*, app_dir_name: str = APP_DIR_NAME) -> Path:
    env_dir = os.getenv(_DATA_DIR_ENV)
    if env_dir:
        return Path(env_dir)

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


def default_vad_model_path() -> Path:
    return user_config_dir() / VAD_MODEL_FILENAME


def default_models_dir() -> Path:
    return user_config_dir() / MODELS_DIRNAME
