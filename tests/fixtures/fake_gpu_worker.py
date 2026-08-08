from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time

config_path = sys.argv[sys.argv.index("--config") + 1]
with open(config_path, encoding="utf-8") as config_file:
    config = json.load(config_file)

if os.environ.get("FAKE_GPU_WORKER_EXIT_BEFORE_AUTH") == "1":
    raise SystemExit(23)

connection = socket.create_connection((config["connect_host"], config["connect_port"]))
writer_lock = threading.Lock()
stopping = threading.Event()
active_request_id = None


def send(payload):
    frame = {
        **payload,
        "contract_version": config["contract_version"],
        "session_id": config["session_id"],
    }
    encoded = json.dumps(frame, separators=(",", ":")).encode("utf-8") + b"\n"
    with writer_lock:
        connection.sendall(encoded)


token = config["auth_token"]
if os.environ.get("FAKE_GPU_WORKER_BAD_AUTH") == "1":
    token = "0" * len(token)
send(
    {
        "type": "authenticate",
        "auth_token": token,
        "worker": "PuriPulyHeartGpuWorker",
        "pid": os.getpid(),
        "mode": config["mode"],
    }
)
if token != config["auth_token"]:
    connection.close()
    raise SystemExit(12)


def heartbeats():
    while not stopping.wait(config["heartbeat_interval_ms"] / 1000):
        send({"type": "heartbeat"})


threading.Thread(target=heartbeats, daemon=True).start()
send({"type": "event", "event": "startup", "request_id": None, "fields": {}})

reader = connection.makefile("rb")
for raw in reader:
    request = json.loads(raw)
    request_type = request["type"]
    request_id = request["request_id"]
    if request_type == "discover":
        send(
            {
                "type": "response",
                "request_id": request_id,
                "status": "ok",
                "payload": {
                    "devices": [
                        {
                            "device_id": "vulkan:0",
                            "registry_index": 0,
                            "name": "Fake GPU",
                            "description": "Fake Vulkan GPU",
                            "device_type": "discrete",
                            "memory_total_bytes": 8000000000,
                            "memory_free_bytes": 4000000000,
                        }
                    ]
                },
            }
        )
    elif request_type == "activate":
        send(
            {
                "type": "response",
                "request_id": request_id,
                "status": "ok",
                "payload": {
                    "activation": {
                        "device": {
                            "device_id": "vulkan:0",
                            "registry_index": 0,
                            "name": "Fake GPU",
                            "description": "Fake Vulkan GPU",
                            "device_type": "discrete",
                            "memory_total_bytes": 8000000000,
                            "memory_free_bytes": 4000000000,
                        },
                        "model_load_seconds": 0.2,
                        "warmup_seconds": 0.1,
                    }
                },
            }
        )
    elif request_type == "transcribe":
        active_request_id = request_id
        if os.environ.get("FAKE_GPU_WORKER_PRESTART_AUDIO_INVALID") == "1":
            send(
                {
                    "type": "response",
                    "request_id": request_id,
                    "status": "failed",
                    "error_code": "audio_invalid",
                    "attempt_started": False,
                    "payload": {"channel": request["channel"], "backend": "Vulkan"},
                }
            )
            active_request_id = None
            continue
        send(
            {
                "type": "event",
                "event": "transcribe_started",
                "request_id": request_id,
                "fields": {
                    "channel": request["channel"],
                    "backend": "Vulkan",
                    "audio_seconds": 1.0,
                },
            }
        )
        if os.environ.get("FAKE_GPU_WORKER_STARTED_FAILURE") == "1":
            if os.environ.get("FAKE_GPU_WORKER_STDERR_ON_FAILURE") == "1":
                print(
                    "native decoder rejected peer frame: invalid token state",
                    file=sys.stderr,
                    flush=True,
                )
            send(
                {
                    "type": "response",
                    "request_id": request_id,
                    "status": "failed",
                    "error_code": "decode_failure",
                    "attempt_started": True,
                    "payload": {
                        "audio_seconds": 1.0,
                        "decode_seconds": 0.25,
                        "rtf": 0.25,
                    },
                }
            )
            active_request_id = None
        elif os.environ.get("FAKE_GPU_WORKER_STARTED_SUCCESS") == "1":
            send(
                {
                    "type": "response",
                    "request_id": request_id,
                    "status": "ok",
                    "payload": {
                        "transcription": {
                            "text": request.get("language_hint") or "fixture",
                            "detected_language": "en",
                            "audio_seconds": 1.0,
                            "decode_seconds": 0.2,
                            "rtf": 0.2,
                        }
                    },
                }
            )
            active_request_id = None
    elif request_type == "cancel":
        target = request["target_request_id"]
        send(
            {
                "type": "event",
                "event": "cancellation_requested",
                "request_id": request_id,
                "fields": {"target_request_id": target, "active": target == active_request_id},
            }
        )
        if target == active_request_id:
            send(
                {
                    "type": "response",
                    "request_id": target,
                    "status": "failed",
                    "error_code": "cancelled",
                    "attempt_started": True,
                    "payload": {
                        "audio_seconds": 1.0,
                        "decode_seconds": 0.25,
                        "rtf": 0.25,
                    },
                }
            )
            active_request_id = None
    elif request_type == "shutdown":
        if os.environ.get("FAKE_GPU_WORKER_IGNORE_SHUTDOWN") == "1":
            while True:
                time.sleep(1)
        stopping.set()
        send(
            {
                "type": "event",
                "event": "shutdown",
                "request_id": request_id,
                "fields": {"outcome": "completed"},
            }
        )
        break

reader.close()
connection.close()
