from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "prepare_special_thanks_avatar.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("prepare_special_thanks_avatar", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normalize_avatar_preserves_aspect_ratio_and_strips_exif(tmp_path: Path) -> None:
    module = _load_script_module()
    source = tmp_path / "source.jpg"
    destination = tmp_path / "out.png"

    image = Image.new("RGB", (600, 400), color=(255, 0, 0))
    exif = Image.Exif()
    exif[271] = "TestCamera"
    exif[274] = 6
    image.save(source, format="JPEG", exif=exif)

    width, height, mode = module.normalize_avatar(source, destination)

    assert (width, height, mode) == (171, 256, "RGB")
    with Image.open(destination) as saved:
        assert saved.format == "PNG"
        assert saved.size == (171, 256)
        assert saved.mode == "RGB"
        assert len(saved.getexif()) == 0


def test_normalize_avatar_preserves_alpha_only_when_transparent(tmp_path: Path) -> None:
    module = _load_script_module()
    transparent_src = tmp_path / "transparent.png"
    opaque_src = tmp_path / "opaque.png"
    transparent_dst = tmp_path / "transparent_out.png"
    opaque_dst = tmp_path / "opaque_out.png"

    Image.new("RGBA", (300, 300), color=(10, 20, 30, 128)).save(transparent_src)
    Image.new("RGBA", (300, 300), color=(10, 20, 30, 255)).save(opaque_src)

    assert module.normalize_avatar(transparent_src, transparent_dst) == (256, 256, "RGBA")
    assert module.normalize_avatar(opaque_src, opaque_dst) == (256, 256, "RGB")


def test_normalize_avatar_does_not_upscale_or_crop_small_images(tmp_path: Path) -> None:
    module = _load_script_module()
    source = tmp_path / "small.png"
    destination = tmp_path / "small_out.png"

    Image.new("RGB", (160, 120), color=(0, 128, 255)).save(source)

    assert module.normalize_avatar(source, destination) == (160, 120, "RGB")
    with Image.open(destination) as saved:
        assert saved.size == (160, 120)


def test_main_writes_explicit_output_and_rejects_invalid_names(tmp_path: Path) -> None:
    module = _load_script_module()
    source = tmp_path / "input.png"
    output = tmp_path / "nested" / "avatar.png"
    Image.new("RGB", (400, 400), color=(20, 40, 60)).save(source)

    assert module.main([str(source), "koz39", "--output", str(output)]) == 0
    assert output.is_file()

    with pytest.raises(ValueError, match="Invalid avatar name"):
        module.main([str(source), "../escape"])
