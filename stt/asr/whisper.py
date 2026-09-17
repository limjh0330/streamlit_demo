"""faster-whisper 백엔드 (한국어 baseline).

CTranslate2 기반이라 CPU int8 에서도 small 모델이 실시간에 근접한다.
스트리밍에서는 beam_size=1, condition_on_previous_text=False 로 두어
지연과 hallucination 을 함께 줄인다.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from ..config import StreamConfig
from ..transcript.medical_terms import build_initial_prompt
from .base import ASREngine, ASRResult, Word

_MODEL_CACHE: dict[tuple, object] = {}
_CACHE_LOCK = threading.Lock()


def load_whisper_model(size: str, device: str, compute_type: str):
    """프로세스 전역 모델 캐시. Streamlit 리런/다중 세션에서 재로드를 막는다."""
    key = (size, device, compute_type)
    with _CACHE_LOCK:
        if key not in _MODEL_CACHE:
            from faster_whisper import WhisperModel

            _MODEL_CACHE[key] = WhisperModel(
                size, device=device, compute_type=compute_type
            )
        return _MODEL_CACHE[key]


class WhisperEngine(ASREngine):
    name = "whisper"
    native_streaming = False
    has_word_timestamps = True

    def __init__(self, config: StreamConfig) -> None:
        super().__init__(config)
        self.model = load_whisper_model(
            config.model_size, config.device, config.compute_type
        )
        self._prompt = build_initial_prompt() if config.use_initial_prompt else None
        self._lock = threading.Lock()

    def transcribe(
        self, audio: np.ndarray, t0: float = 0.0, is_final: bool = False
    ) -> ASRResult:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        started = time.perf_counter()

        with self._lock:                       # 모델 인스턴스는 스레드 안전하지 않다
            segments, info = self.model.transcribe(
                audio,
                language=self.config.language or None,
                beam_size=(
                    self.config.final_beam_size if is_final else self.config.beam_size
                ),
                word_timestamps=True,
                condition_on_previous_text=False,
                initial_prompt=self._prompt,
                vad_filter=False,              # VAD 는 세션 단에서 이미 수행
                no_speech_threshold=0.6,
            )
            words: list[Word] = []
            texts: list[str] = []
            for seg in segments:               # generator -> lock 안에서 소비
                texts.append(seg.text)
                for w in (seg.words or []):
                    token = w.word.strip()
                    if token:
                        words.append(
                            Word(token, t0 + w.start, t0 + w.end, float(w.probability))
                        )

        return ASRResult(
            text="".join(texts).strip(),
            words=words,
            language=getattr(info, "language", self.config.language),
            audio_sec=audio.size / self.sample_rate,
            infer_sec=time.perf_counter() - started,
            is_final=is_final,
        )

    def warmup(self) -> float:
        """첫 추론 지연을 미리 소진한다.

        주의: `initial_prompt` 를 붙인 채 무음을 디코딩하면 Whisper 가 프롬프트를
        이어 쓰려고 헛돌아 small 기준 10 초 넘게 걸린다(실측). warmup 은
        프롬프트 없이 돌린다.
        """
        started = time.perf_counter()
        silence = np.zeros(int(self.sample_rate * 0.5), dtype=np.float32)
        with self._lock:
            segments, _ = self.model.transcribe(
                silence,
                language=self.config.language or None,
                beam_size=1,
                word_timestamps=False,
                condition_on_previous_text=False,
            )
            list(segments)
        return time.perf_counter() - started

    def transcribe_file(self, path: str, beam_size: int = 5):
        """파일 전사용 경로. (segments, info) 를 그대로 돌려준다."""
        return self.model.transcribe(
            path,
            language=self.config.language or None,
            beam_size=beam_size,
            initial_prompt=self._prompt,
        )
