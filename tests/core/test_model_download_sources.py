from __future__ import annotations

import pytest

from puripuly_heart.core.model_download_sources import (
    model_download_source_candidates,
    model_download_source_order,
)
from puripuly_heart.core.model_install_removal import (
    ModelInstallInUseError,
    remove_model_install_dir,
)


@pytest.mark.parametrize(
    ("locale", "expected"),
    [
        ("zh-CN", ("modelscope", "hf_mirror", "huggingface")),
        ("zh_HK", ("modelscope", "hf_mirror", "huggingface")),
        ("ko", ("huggingface", "modelscope", "hf_mirror")),
        (None, ("huggingface", "modelscope", "hf_mirror")),
    ],
)
def test_source_order_follows_ui_locale(locale, expected) -> None:
    assert model_download_source_order(locale) == expected


def test_candidates_skip_unregistered_sources_and_derive_the_mirror_from_huggingface() -> None:
    assert model_download_source_candidates(("huggingface",), locale="zh-CN") == (
        "hf_mirror",
        "huggingface",
    )
    assert model_download_source_candidates(("modelscope",), locale="en") == ("modelscope",)
    assert model_download_source_candidates(
        ("huggingface", "modelscope"), locale="en", explicit="hf_mirror"
    ) == ("hf_mirror",)
    assert (
        model_download_source_candidates(("huggingface",), locale="en", explicit="modelscope") == ()
    )


def test_removal_deletes_the_install_and_previous_residue(tmp_path) -> None:
    install = tmp_path / "model"
    (install / "nested").mkdir(parents=True)
    (install / "nested" / "weights.bin").write_bytes(b"x")
    residue = tmp_path / "model.deleting-old"
    residue.mkdir()

    assert remove_model_install_dir(install) is True

    assert not install.exists()
    assert not residue.exists()
    assert list(tmp_path.iterdir()) == []
    assert remove_model_install_dir(install) is False


def test_removal_keeps_files_when_the_install_cannot_be_moved(tmp_path, monkeypatch) -> None:
    install = tmp_path / "model"
    install.mkdir()
    (install / "weights.bin").write_bytes(b"keep")

    def refuse(self, target):
        raise PermissionError(32, "in use")

    monkeypatch.setattr(type(install), "rename", refuse)

    with pytest.raises(ModelInstallInUseError):
        remove_model_install_dir(install)

    assert (install / "weights.bin").read_bytes() == b"keep"
