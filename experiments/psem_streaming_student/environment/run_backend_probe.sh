#!/usr/bin/env bash
set -euo pipefail

readonly VENV=/opt/psem-streaming-student/rocm-10.0.0-pytorch-2.13.0
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export PATH="$VENV/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="/opt/rocm/lib:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec "$VENV/bin/python" "$SCRIPT_DIR/gpu_backend_probe.py" "$@"
