import asyncio
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
from PIL import Image, ImageDraw, ImageFont

from puripuly_heart.core.photo.bridge import PhotoBridge
from puripuly_heart.core.photo.image import load_prepared_photo
from puripuly_heart.providers.photo import RelayPhotoConnection


def generate(path: Path) -> None:
    size = (3840, 2160)
    noise_size = (160, 90)
    random_bytes = random.Random(499).randbytes(noise_size[0] * noise_size[1] * 3)
    texture = Image.frombytes("RGB", noise_size, random_bytes).resize(size, Image.Resampling.BICUBIC)
    image = Image.blend(Image.new("RGB", size, (47, 68, 94)), texture, 0.25)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=92)
    for index in range(12):
        left = 160 + index % 3 * 1230
        top = 175 + index // 3 * 485
        draw.rounded_rectangle((left, top, left + 1020, top + 170), 26, fill=(242, 238, 225))
        draw.text((left + 42, top + 27), f"Synthetic label {index + 1:02}", fill=(23, 24, 27), font=font)
    image.save(path)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


async def prepare(executable: Path, output: Path, root: Path) -> dict:
    source = root / "capture.png"
    generate(source)
    prepared_dir = root / "private"
    prepared_dir.mkdir()
    stat = source.stat()
    identity = (stat.st_dev, stat.st_ino)
    photo_id = "issue-499-synthetic"
    pulled = asyncio.Event()

    def control(value: dict, generation: int) -> None:
        if (generation == 1 and value["type"] == "photo_pull" and value["photo_id"] == photo_id
                and value["event_id"] == "synthetic-pull"):
            pulled.set()

    bridge = PhotoBridge(control=control)
    child = None
    try:
        await bridge.start()
        manifest = await bridge.activate(1, backend="probe")
        manifest_path = root / "launch.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        env = {name: os.environ[name] for name in ("SystemRoot", "WINDIR", "PATH") if name in os.environ}
        env.update(TEMP=str(root), TMP=str(root), USERPROFILE=str(root), LOCALAPPDATA=str(root), APPDATA=str(root))
        child = await asyncio.create_subprocess_exec(
            str(executable), "--role", "photo", "--config", str(manifest_path), env=env, cwd=root,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        exited = asyncio.create_task(child.communicate())
        ready = asyncio.create_task(bridge.ready.wait())
        await asyncio.wait((exited, ready), return_when=asyncio.FIRST_COMPLETED, timeout=10)
        if not ready.done() or exited.done():
            ready.cancel()
            raise RuntimeError(f"photo child not ready (exit={child.returncode})")
        photo = {
            "photo_id": photo_id, "path": str(source), "expected_file_id": list(map(str, identity)),
            "file_size": stat.st_size, "shutter_unix_ms": time.time_ns() // 1_000_000,
            "attempt_id": 0, "phase": "offer", "offer_open": True, "visible": False, "image": None,
            "preparations": [], "blocks": [], "excluded_blocks": [], "failure_code": None,
            "first_block_drawn": False,
        }
        bridge.publish({"type": "photo_snapshot", "revision": 1, "photo": photo})
        assert bridge.send({"type": "photo_probe_pull", "photo_id": photo_id, "event_id": "synthetic-pull"})
        await asyncio.wait_for(pulled.wait(), 5)
        bridge.publish({"type": "photo_snapshot", "revision": 2, "photo": {
            **photo, "attempt_id": 1, "phase": "preparing", "offer_open": False, "visible": True,
        }})
        result = await bridge.prepare(photo_id, 1, source, prepared_dir, expected_file_id=identity)
        prepared = await load_prepared_photo(result, directory=prepared_dir)
        jpeg_path = Path(result["jpeg_path"])
        with Image.open(jpeg_path) as jpeg:
            assert jpeg.size == prepared.send_size == (1280, 720)
        async with httpx.AsyncClient(trust_env=False) as client:
            def forbidden_credential():
                raise AssertionError("no app credentials permitted")

            connection = RelayPhotoConnection("http://127.0.0.1/v1/photo", forbidden_credential,
                                              client=client, allow_loopback_http=True)
            body = connection._body(prepared, "ko")
        assert bridge.send({"type": "shutdown"})
        await asyncio.wait_for(asyncio.shield(exited), 10)
        assert child.returncode == 0 and bridge.shutdown_complete.is_set()
        shutil.copyfile(jpeg_path, output / "app-prepared.jpg")
        (output / "app-request.json").write_text(json.dumps(body, separators=(",", ":")), encoding="utf-8")
        return {
            "source_size": [prepared.source_width, prepared.source_height], "source_bytes": stat.st_size,
            "source_sha256": sha256(source.read_bytes()), "jpeg_size": list(prepared.send_size),
            "jpeg_bytes": jpeg_path.stat().st_size, "jpeg_sha256": sha256(jpeg_path.read_bytes()),
            "native_sha256": sha256(executable.read_bytes()), "native_probe_shutdown_acknowledged": True,
            "native_preparation_ms": prepared.preparation_ms,
        }
    finally:
        if child is not None and child.returncode is None:
            child.kill()
            await child.wait()
        await bridge.close()


def main() -> None:
    executable, output = map(lambda arg: Path(arg).resolve(strict=True), sys.argv[1:])
    if (output / "app-request.json").exists():
        raise SystemExit("Use a fresh output directory")
    with tempfile.TemporaryDirectory(prefix="prepare-", dir=output) as directory:
        result = asyncio.run(prepare(executable, output, Path(directory)))
    (output / "app-fixture.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
