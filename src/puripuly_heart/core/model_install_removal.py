from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4


class ModelInstallInUseError(RuntimeError):
    failure_code = "model_in_use"


def remove_model_install_dir(install_dir: Path) -> bool:
    for residue in install_dir.parent.glob(f"{install_dir.name}.deleting-*"):
        shutil.rmtree(residue, ignore_errors=True)
    if not install_dir.exists():
        return False
    trash = install_dir.with_name(f"{install_dir.name}.deleting-{uuid4().hex}")
    try:
        install_dir.rename(trash)
    except OSError as error:
        raise ModelInstallInUseError("Model files are in use") from error
    shutil.rmtree(trash, ignore_errors=True)
    return True


__all__ = ["ModelInstallInUseError", "remove_model_install_dir"]
