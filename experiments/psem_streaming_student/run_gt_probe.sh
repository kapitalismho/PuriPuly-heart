#!/usr/bin/env bash
set -euo pipefail

readonly VENV=/opt/psem-streaming-student/rocm-10.0.0-pytorch-2.13.0
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
readonly STAGE="${1:?stage is required}"
readonly CONFIG="${2:?config path is required}"
RUN_ROOT="${3:?run root is required}"
if [[ "$RUN_ROOT" != /* || "$RUN_ROOT" == "/" ]]; then
    printf 'run root must be a non-root absolute WSL path: %q\n' "$RUN_ROOT" >&2
    exit 2
fi
readonly RUN_ROOT

export PATH="$VENV/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="/opt/rocm/lib:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$REPO_ROOT"

exec "$VENV/bin/python" "$SCRIPT_DIR/gt_probe.py" "$STAGE" --config "$CONFIG" --run-root "$RUN_ROOT"
