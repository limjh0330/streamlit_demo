"""Fun-ASR-MLT-Nano-2512 어댑터.

공식 FunASR ``AutoModel.generate`` API를 사용한다. 이 체크포인트는 현재
타임스탬프를 반환하지 않으므로, SenseVoice와 마찬가지로 발화 전체를 주기적으로
재인식해 최신 가설로 교체한다.
"""
from __future__ import annotations

import os
import tempfile
import threading
import time
import wave

import numpy as np

from ..config import MODELS_DIR, StreamConfig
from .base import ASREngine, ASRResult

MODEL_ID = "FunAudioLLM/Fun-ASR-MLT-Nano-2512"

_HINT = (
    "Fun-ASR-MLT-Nano-2512에는 FunASR가 필요합니다.\n"
    '  pip install -U "funasr>=1.4.1"\n'
    "  모델은 첫 실행 시 Hugging Face 캐시에 자동으로 내려받습니다.\n"
    "  모델: https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512"
)

_MODEL_CACHE: dict[str, object] = {}
_CACHE_LOCK = threading.Lock()
#: 모델(장치)별 추론 lock. 세션마다 엔진 인스턴스는 따로지만 모델은 하나이므로,
#: `AutoModel.generate()` 동시 호출을 막으려면 lock 도 모델 단위로 공유해야 한다.
_INFER_LOCKS: dict[str, threading.Lock] = {}

# 모델 카드의 언어명 API와 기존 UI의 ISO 언어 코드를 모두 지원한다.
_LANGUAGE_NAMES = {
    "ko": "한국어", "korean": "한국어", "한국어": "한국어",
    "en": "영어", "english": "영어", "영어": "영어",
    "zh": "중국어", "zh-cn": "중국어", "chinese": "중국어", "중국어": "중국어",
    "ja": "일본어", "japanese": "일본어", "일본어": "일본어",
    "yue": "광둥어", "cantonese": "광둥어", "광둥어": "광둥어",
}


def _funasr_device(device: str) -> str:
    return "cuda:0" if device == "cuda" else "cpu"


def load_model(device: str):
    """프로세스당 장치별 한 번만 모델을 로드한다."""
    resolved = _funasr_device(device)
    with _CACHE_LOCK:
        if resolved not in _MODEL_CACHE:
            try:
                from funasr import AutoModel
            except ImportError as e:  # pragma: no cover - environment dependent
                raise RuntimeError(_HINT) from e
            model_dir = MODELS_DIR / "funasr_mlt_nano"
            model = str(model_dir) if (model_dir / "model.pt").is_file() else MODEL_ID
            kwargs = {
                "model": model,
                "trust_remote_code": True,
                "remote_code": "./model.py",
                "device": resolved,
            }
            if model == MODEL_ID:
                kwargs["hub"] = "hf"
            _MODEL_CACHE[resolved] = AutoModel(**kwargs)
            _INFER_LOCKS[resolved] = threading.Lock()
        return _MODEL_CACHE[resolved]


def inference_lock(device: str) -> threading.Lock:
    """`load_model(device)` 로 올린 모델을 쓰는 모든 세션이 공유하는 추론 lock."""
    resolved = _funasr_device(device)
    with _CACHE_LOCK:
        return _INFER_LOCKS.setdefault(resolved, threading.Lock())


def is_loaded(device: str) -> bool:
    return _funasr_device(device) in _MODEL_CACHE


class FunASRMLTNanoEngine(ASREngine):
    name = "funasr_mlt_nano"
    native_streaming = False
    has_word_timestamps = False
    decodes_full_utterance = True
    shared_model = True

    def __init__(self, config: StreamConfig) -> None:
        super().__init__(config)
        self.model = load_model(config.device)
        # 세션 간 공유: 동시 세션의 추론은 이 lock 으로 한 번에 하나씩 돈다
        self._lock = inference_lock(config.device)

    def transcribe(
        self, audio: np.ndarray, t0: float = 0.0, is_final: bool = False
    ) -> ASRResult:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        started = time.perf_counter()
        path: str | None = None
        try:
            # 공식 API 예제는 파일 경로 입력을 사용한다. 외부 오디오 코덱 의존 없이
            # 16-kHz PCM WAV를 만들어 동일한 경로로 실시간 버퍼도 처리한다.
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                path = f.name
            pcm = np.clip(audio, -1.0, 1.0)
            with wave.open(path, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(self.sample_rate)
                wav.writeframes((pcm * 32767.0).astype("<i2").tobytes())

            language = _LANGUAGE_NAMES.get((self.config.language or "").lower())
            kwargs = {
                "input": [path],
                "cache": {},
                "batch_size": 1,
                "itn": True,
                "llm_kwargs": {"do_sample": False},
            }
            if language:
                kwargs["language"] = language
            with self._lock:
                result = self.model.generate(**kwargs)
            text = str(result[0].get("text", "")).strip() if result else ""
        finally:
            if path:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass

        return ASRResult(
            text=text,
            language=self.config.language,
            audio_sec=audio.size / self.sample_rate,
            infer_sec=time.perf_counter() - started,
            is_final=is_final,
        )
