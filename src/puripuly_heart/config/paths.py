from __future__ import annotations

import os
import sys
from pathlib import Path

STABLE_APP_DIR_NAME = "puripuly-heart"
APP_DIR_NAME = STABLE_APP_DIR_NAME
SETTINGS_FILENAME = "settings.json"
VAD_MODEL_FILENAME = "silero_vad.onnx"
MODELS_DIRNAME = "models"
HTTP_EXTENSIONS_DIRNAME = "http_extensions"


def user_config_dir(*, app_dir_name: str = APP_DIR_NAME) -> Path:
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


def default_http_extensions_dir() -> Path:
    return user_config_dir() / HTTP_EXTENSIONS_DIRNAME
