from __future__ import annotations

import sys

from puripuly_heart.core.windows_process_ownership import retain_current_process_job
from puripuly_heart.main import main

retain_current_process_job()
raise SystemExit(main(sys.argv[1:]))
