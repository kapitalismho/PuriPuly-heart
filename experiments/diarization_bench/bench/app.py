"""In-process development host driven through the application control catalog."""

from __future__ import annotations

import asyncio
import itertools
import json
import time
from pathlib import Path

from bench import env

_request_ids = itertools.count(1)


class CommandFailed(RuntimeError):
    def __init__(self, name: str, result: dict) -> None:
        super().__init__(f"{name}: {result.get('status')} {json.dumps(result, default=str)[:800]}")
        self.result = result


class BenchApp:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.environment = env.prepare(self.root)
        self.control_dir = Path(self.environment["control"])
        self.config_path = self.root / "settings.json"
        self.app = None
        self.control = None

    async def start(self) -> None:
        from bench import observe, source

        observe.install(self.control_dir / "observations.jsonl")
        from puripuly_heart.core.audio import desktop_source

        desktop_source.DesktopLoopbackAudioSource = source.BenchLoopbackSource
        env.write_isolated_settings(self.config_path)
        from puripuly_heart.composition.headless_application import compose_headless_application

        self.app = compose_headless_application(self.config_path)
        await self.app.start()
        self.control = self.app.control()
        self.control.bind_instance(f"diarization-bench-{int(time.time())}")

    async def stop(self) -> None:
        if self.app is not None:
            await self.app.stop()
            self.app = None

    async def query(self, name: str, arguments: dict | None = None) -> dict:
        return await self.control.query(name, arguments or {})

    async def command(
        self,
        name: str,
        arguments: dict | None = None,
        *,
        timeout: float = 300,
        accept: tuple[str, ...] = ("applied",),
    ) -> dict:
        submitted = await self.control.submit(
            name, arguments or {}, request_id=f"bench-{next(_request_ids)}-{time.time_ns()}"
        )
        operation_id = submitted.get("operation_id")
        result = submitted
        if operation_id and not submitted.get("terminal"):
            result = await self.control.wait(operation_id, timeout=timeout)
        if result.get("status") not in accept:
            raise CommandFailed(name, result)
        return result

    async def settings_apply(self, changes: dict, **kwargs) -> dict:
        return await self.command("settings.apply", {"changes": changes}, **kwargs)

    async def wait_for(self, predicate, *, timeout: float, interval: float = 0.25):
        deadline = time.monotonic() + timeout
        while True:
            value = await predicate()
            if value:
                return value
            if time.monotonic() > deadline:
                raise TimeoutError("benchmark host condition not reached")
            await asyncio.sleep(interval)
