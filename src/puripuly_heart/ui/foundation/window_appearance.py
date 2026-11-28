from pathlib import Path
from tempfile import NamedTemporaryFile


def _preference_path(config_path: Path) -> Path:
    return config_path.with_name(f"{config_path.name}.theme")


def load_dark_mode(config_path: Path) -> bool:
    try:
        return _preference_path(config_path).read_text(encoding="utf-8").strip() == "dark"
    except OSError, UnicodeError:
        return False


def save_dark_mode(config_path: Path, dark: bool) -> None:
    path = _preference_path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8", delete=False) as file:
            temporary_path = Path(file.name)
            file.write("dark\n" if dark else "light\n")
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
