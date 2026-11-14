import subprocess
import sys

from puripuly_heart.config.paths import user_config_dir


def open_log_folder() -> None:
    log_dir = user_config_dir()
    if sys.platform == "win32":
        subprocess.Popen(["explorer", str(log_dir)])
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(log_dir)])
    else:
        subprocess.Popen(["xdg-open", str(log_dir)])
