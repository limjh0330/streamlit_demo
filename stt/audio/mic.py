"""로컬 마이크 입력 (sounddevice InputStream -> Queue).

개발문서의 "마이크 -> sounddevice callback -> 50~100 ms block -> Queue" 경로.
InputStream 만 사용해 입력 전용으로 연다.
"""
from __future__ import annotations

import queue
from typing import Callable

import numpy as np

from ..config import BLOCK_SIZE, CHANNELS, SAMPLE_RATE


def list_input_devices() -> list[dict]:
    """입력 채널이 있는 디바이스 목록. sounddevice 미설치 시 빈 리스트."""
    try:
        import sounddevice as sd
    except Exception:
        return []
    devices = []
    for idx, dev in enumerate(sd.query_devices()):
        if dev.get("max_input_channels", 0) > 0:
            devices.append(
                {
                    "index": idx,
                    "name": dev["name"],
                    "channels": dev["max_input_channels"],
                    "default_samplerate": dev.get("default_samplerate"),
                }
            )
    return devices


class MicrophoneStream:
    """sounddevice 콜백에서 큐로만 밀어넣는(=논블로킹) 마이크 스트림."""

    def __init__(
        self,
        device: int | str | None = None,
        sample_rate: int = SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        on_block: Callable[[np.ndarray], None] | None = None,
    ) -> None:
        self.device = device
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.on_block = on_block
        self.queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self.status_messages: list[str] = []
        self._stream = None

    def _callback(self, indata, frames, time_info, status):  # PortAudio 스레드
        if status:
            self.status_messages.append(str(status))
        block = indata[:, 0].copy() if indata.ndim == 2 else indata.copy()
        if self.on_block is not None:
            self.on_block(block)
        else:
            self.queue.put(block)

    def start(self) -> None:
        import sounddevice as sd

        self._stream = sd.InputStream(
            device=self.device,
            samplerate=self.sample_rate,
            channels=CHANNELS,
            dtype="float32",
            blocksize=self.block_size,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None

    @property
    def active(self) -> bool:
        return self._stream is not None and self._stream.active

    def __enter__(self) -> "MicrophoneStream":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
