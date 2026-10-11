import os
import sys
import time

import numpy as np

from puripuly_heart.app.adapters.nemotron.protocol import encode, read_message

cursor = emitted = epoch = 0
chunk = right = 0
while message := read_message(sys.stdin.buffer):
    request, payload = message
    op = request["op"]
    response = {"id": request["id"], "ok": True}
    data = b""
    if op == "start":
        if request["device"] == 99:
            sys.stdout.buffer.write(
                encode({**response, "ok": False, "error": "no selected Vulkan GPU"})
            )
            sys.stdout.buffer.flush()
            break
        chunk, right = (9, 4) if request["profile"] == "1.04" else (6, 2)
        bound = (chunk + right) * 1280 + 512
        response.update(
            tail_padding_samples=bound,
            max_revision_samples=bound,
            diagnostics=[["fixture_pid", str(os.getpid())]],
        )
        sys.stderr.buffer.write(b"fixture stderr" * 5000)
        sys.stderr.buffer.flush()
    elif op == "reset":
        epoch, cursor, emitted = request["epoch"], 0, 0
    elif op in ("feed", "pause"):
        first = emitted
        if op == "feed":
            values = np.frombuffer(payload, dtype=np.float32)
            if values[0] == 99:
                time.sleep(1)
            cursor += len(values)
            while (emitted // 8 + chunk + right) * 1280 + 256 <= cursor:
                emitted += chunk * 8
        count = emitted - first
        probabilities = np.full((count, 8), 0.125, dtype=np.float32)
        response.update(
            first_frame=first,
            count=count,
            processed=emitted * 160,
            immutable=emitted * 160,
            epoch=epoch,
        )
        data = probabilities.tobytes()
    elif op == "stop":
        break
    sys.stdout.buffer.write(encode(response, data))
    sys.stdout.buffer.flush()
