import pytest

from puripuly_heart.ui import log_folder


@pytest.mark.parametrize(
    ("platform", "launcher"),
    (("win32", "explorer"), ("darwin", "open"), ("linux", "xdg-open")),
)
def test_agent_folder_action_opens_the_user_log_directory(
    monkeypatch, tmp_path, platform, launcher
):
    commands = []
    monkeypatch.setattr(log_folder, "user_config_dir", lambda: tmp_path)
    monkeypatch.setattr(log_folder.sys, "platform", platform)
    monkeypatch.setattr(log_folder.subprocess, "Popen", commands.append)
    log_folder.open_log_folder()
    assert commands == [[launcher, str(tmp_path)]]
