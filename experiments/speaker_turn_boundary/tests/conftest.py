from __future__ import annotations

import pathlib
import shutil
import tempfile
from collections.abc import Iterator

import pytest

from puripuly_heart.core.vad.bundled import bundled_silero_vad_onnx_path


@pytest.fixture(scope="session")
def silero_model_path() -> Iterator[pathlib.Path]:
    tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="stb_silero_"))
    model_path = tmp_dir / "silero_vad.onnx"
    with bundled_silero_vad_onnx_path().open("rb") as source:
        with model_path.open("wb") as target:
            shutil.copyfileobj(source, target)
    yield model_path
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture()
def tmp_dir() -> Iterator[pathlib.Path]:
    tmp_path = pathlib.Path(tempfile.mkdtemp(prefix="stb_test_"))
    yield tmp_path
    shutil.rmtree(tmp_path, ignore_errors=True)
