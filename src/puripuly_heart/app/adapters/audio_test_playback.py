from __future__ import annotations

import asyncio

import numpy as np


async def play_recording(samples, sample_rate, cancellation):
    def play():
        import sounddevice as sd

        pcm = np.asarray(samples, dtype=np.float32).reshape(len(samples), -1)
        with sd.OutputStream(
            samplerate=sample_rate, channels=pcm.shape[1], dtype="float32"
        ) as stream:
            block = max(1, sample_rate // 10)
            for offset in range(0, len(pcm), block):
                if cancellation.is_set():
                    stream.abort()
                    break
                stream.write(pcm[offset : offset + block])

    task = asyncio.create_task(asyncio.to_thread(play))
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        cancellation.set()
        await asyncio.gather(task, return_exceptions=True)
        raise
