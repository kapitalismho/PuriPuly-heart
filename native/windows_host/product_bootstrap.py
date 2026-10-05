from __future__ import annotations

import os
import sys

from _puripuly_native_runtime import install

install(os.environ["PURIPULY_HEART_NATIVE_RUNTIME_ROOT"])

from puripuly_heart.core.windows_process_ownership import retain_current_process_job
from puripuly_heart.main import main

if len(sys.argv) < 2 or sys.argv[1] != "cli":
    retain_current_process_job()
raise SystemExit(main(sys.argv[1:]))
