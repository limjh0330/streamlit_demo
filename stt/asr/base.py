"""ASR 어댑터 공통 인터페이스.

개발문서의 ASR Adapter 계층. Whisper / Zipformer / SenseVoice 를
같은 인터페이스 뒤에 두어 세션 코드가 엔진에 의존하지 않게 한다.

  - 윈도우형 엔진(Whisper, SenseVoice): `transcribe(audio, t0)` 로 구간을 통째로 인식
  - 진짜 스트리밍 엔진(Zipformer): `accept_waveform()` + `partial()` 사용
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from ..config import SAMPLE_RATE, StreamConfig


@dataclass
class Word:
    """절대 시각(세션 시작 기준, 초)이 붙은 단어."""

    text: str
    start: float
    end: float
    prob: float = 1.0


@dataclass
class ASRResult:
    text: str = ""
    words: list[Word] = field(default_factory=list)
    language: str | None = None
    audio_sec: float = 0.0
    infer_sec: float = 0.0
    is_final: bool = False

    @property
    def rtf(self) -> float:
        """Real-Time Factor = 추론시간 / 오디오길이. 1.0 미만이어야 실시간."""
        return self.infer_sec / self.audio_sec if self.audio_sec > 0 else 0.0


class ASREngine(ABC):
    """모든 ASR 백엔드의 베이스."""

    name: str = "base"
    #: True 면 세션이 sliding window 대신 accept_waveform/partial 경로를 쓴다.
    native_streaming: bool = False
    #: 단어 단위 타임스탬프를 주는지(병합기의 정렬 전략이 달라진다).
    has_word_timestamps: bool = False

    def __init__(self, config: StreamConfig) -> None:
        self.config = config
        self.sample_rate = SAMPLE_RATE

    @abstractmethod
    def transcribe(
        self, audio: np.ndarray, t0: float = 0.0, is_final: bool = False
    ) -> ASRResult:
        """`audio` 구간을 인식. t0 는 이 구간의 절대 시작 시각(초)."""

    # --- 네이티브 스트리밍 엔진만 구현 -------------------------------------
    def accept_waveform(self, audio: np.ndarray) -> None:
        raise NotImplementedError

    def partial(self) -> ASRResult:
        raise NotImplementedError

    def is_endpoint(self) -> bool:
        return False

    def reset_stream(self) -> None:
        pass

    # --- 공통 -------------------------------------------------------------
    def warmup(self) -> float:
        """모델 첫 추론 지연을 미리 소진. 소요 시간(초) 반환."""
        t = time.perf_counter()
        silence = np.zeros(int(self.sample_rate * 0.5), dtype=np.float32)
        try:
            if self.native_streaming:
                self.accept_waveform(silence)
                self.partial()
                self.reset_stream()
            else:
                self.transcribe(silence)
        except Exception:
            pass
        return time.perf_counter() - t

    def close(self) -> None:
        pass

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} name={self.name!r}>"


def create_engine(config: StreamConfig) -> ASREngine:
    """config.engine 이름으로 어댑터를 생성한다."""
    engine = (config.engine or "whisper").lower()
    if engine == "whisper":
        from .whisper import WhisperEngine

        return WhisperEngine(config)
    if engine == "zipformer":
        from .zipformer import ZipformerEngine

        return ZipformerEngine(config)
    if engine == "sensevoice":
        from .sensevoice import SenseVoiceEngine

        return SenseVoiceEngine(config)
    raise ValueError(f"알 수 없는 엔진: {config.engine!r}")
