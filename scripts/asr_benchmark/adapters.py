from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from puripuly_heart.app.adapters.asr_worker_process import DefaultASRWorkerProcessFactory
from puripuly_heart.core.lifecycle import start_lifecycle_task
from puripuly_heart.providers.stt.local_cpp import LocalCppSTTBackend
from puripuly_heart.providers.stt.local_cpu import create_local_cpu_backend


class ObservedClient:
    def __init__(self, client, owner):
        self.client = client
        self.owner = owner

    def __getattr__(self, name):
        return getattr(self.client, name)

    async def activate(self, **kwargs):
        result = await self.client.activate(**kwargs)
        self.owner.activation = asdict(result)
        return result

    async def transcribe(self, **kwargs):
        self.owner.last_native = None
        result = await self.client.transcribe(**kwargs)
        self.owner.last_native = asdict(result)
        return result


class ObservedFactory:
    def __init__(self, worker: Path):
        self.factory = DefaultASRWorkerProcessFactory(executable_path=worker)
        self.activation = None
        self.last_native = None

    def __getattr__(self, name):
        return getattr(self.factory, name)

    async def start(self, **kwargs):
        return ObservedClient(await self.factory.start(**kwargs), self)


@dataclass(slots=True)
class ExplicitGGUFBackend(LocalCppSTTBackend):
    gguf_path: Path | None = None

    def _validate_runtime_assets(self):
        if self.gguf_path is None or not self.gguf_path.is_file():
            raise FileNotFoundError(self.gguf_path)

    async def _load_resource(self):
        client = await self.process_factory.start(mode="persistent")
        try:
            await self._settle(
                client.activate(
                    model_path=self.gguf_path,
                    device_id="cpu",
                    backend="cpu",
                    channel="self",
                    n_threads=2,
                )
            )
            start_lifecycle_task(self._scope, self._drain_events(client), name="events")
            return client
        except BaseException:
            await client.close()
            raise


class ModelAdapter:
    def __init__(self, config, language, app_root: Path, model_root: Path, worker: Path):
        self.observed = ObservedFactory(worker)
        if config["adapter"] == "app":
            self.backend = create_local_cpu_backend(
                config["model_id"],
                model_root=app_root,
                source_language=language,
                sample_rate_hz=16000,
                stream_label="self",
                process_factory=self.observed,
            )
        elif config["adapter"] == "gguf":
            self.backend = ExplicitGGUFBackend(
                model_dir=model_root,
                model_id="experiment-explicit-gguf",
                provider_id="experiment_gguf",
                stream_label="self",
                process_factory=self.observed,
                gguf_path=model_root / config["files"][0]["path"],
                language_hint=None,
            )
        else:
            raise ValueError(f"Unknown adapter: {config['adapter']}")

    async def initialize(self):
        await self.backend._ensure_recognizer()

    async def decode(self, pcm: bytes):
        self.observed.last_native = None
        return await self.backend.decode_pcm16le(pcm)

    async def close(self):
        await self.backend.close()
