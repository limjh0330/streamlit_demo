"""수음된 오디오를 실시간으로 raw WAV 에 저장.

세션이 진행되는 동안 계속 append 하므로, 도중에 프로세스가 죽어도
헤더만 고쳐 쓰면 그때까지의 오디오가 남는다.
"""
from __future__ import annotations

import threading
import wave
from pathlib import Path

import numpy as np

from ..config import CHANNELS, RECORDINGS_DIR, SAMPLE_RATE
from .resampler import float32_to_pcm16


class WavRecorder:
    def __init__(
        self,
        session_id: str,
        directory: Path = RECORDINGS_DIR,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.path = Path(directory) / f"{session_id}.wav"
        self.sample_rate = sample_rate
        self._lock = threading.Lock()
        self._frames = 0

        self._wf = wave.open(str(self.path), "wb")
        self._wf.setnchannels(CHANNELS)
        self._wf.setsampwidth(2)                 # PCM16
        self._wf.setframerate(sample_rate)

    def write(self, audio: np.ndarray | bytes) -> None:
        pcm = audio if isinstance(audio, (bytes, bytearray)) else float32_to_pcm16(audio)
        with self._lock:
            if self._wf is None:
                return
            self._wf.writeframes(pcm)
            self._frames += len(pcm) // 2

    def close(self) -> Path:
        with self._lock:
            if self._wf is not None:
                self._wf.close()
                self._wf = None
        return self.path

    @property
    def duration_sec(self) -> float:
        return self._frames / self.sample_rate
