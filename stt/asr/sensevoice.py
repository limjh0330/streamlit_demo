"""sherpa-onnx SenseVoice 백엔드 (빠른 non-autoregressive ASR).

autoregressive 디코딩이 없어 한 chunk 를 통째로 밀어 넣으면 매우 빠르게
결과가 나온다. 대신 스트리밍 상태가 없으므로 VAD 로 자른 chunk 단위 추론에 쓴다.

추론이 워낙 빨라(RTF ~0.02) 윈도우를 잘라 넣는 대신 발화 전체를 매번 다시
인식한다. 잘린 오디오를 인식할 때 생기는 오류와 윈도우 간 중복이 함께 사라진다.

모델 준비:
    python -m scripts.fetch_models sensevoice
    → models/sensevoice/ (sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17)
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

from ..config import MODELS_DIR, StreamConfig
from .base import ASREngine, ASRResult, Word

_HINT = (
    "sherpa-onnx 와 SenseVoice 모델이 필요합니다.\n"
    "  pip install sherpa-onnx\n"
    "  python -m scripts.fetch_models sensevoice\n"
    "  모델: https://github.com/k2-fsa/sherpa-onnx/releases (sense-voice)"
)


class SenseVoiceEngine(ASREngine):
    name = "sensevoice"
    native_streaming = False
    has_word_timestamps = False        # 단어 타임스탬프 없음 -> 텍스트 기반 병합
    decodes_full_utterance = True      # 발화 전체를 매번 다시 인식 (RTF 0.02)

    def __init__(self, config: StreamConfig) -> None:
        super().__init__(config)
        try:
            import sherpa_onnx  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(_HINT) from e

        model_dir = Path(config.model_dir or (MODELS_DIR / "sensevoice"))
        if not model_dir.is_dir():
            raise FileNotFoundError(f"모델 디렉터리가 없습니다: {model_dir}\n{_HINT}")

        models = sorted(model_dir.glob("*.onnx"), key=lambda p: (0 if "int8" in p.name else 1))
        tokens = model_dir / "tokens.txt"
        if not models or not tokens.exists():
            raise FileNotFoundError(f"{model_dir} 에 model(.onnx)/tokens.txt 가 없습니다.\n{_HINT}")

        self.recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(models[0]),
            tokens=str(tokens),
            num_threads=2,
            use_itn=True,
            language=self.config.language or "",
            provider="cuda" if config.device == "cuda" else "cpu",
        )
        self._lock = threading.Lock()

    def transcribe(
        self, audio: np.ndarray, t0: float = 0.0, is_final: bool = False
    ) -> ASRResult:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        started = time.perf_counter()
        with self._lock:
            stream = self.recognizer.create_stream()
            stream.accept_waveform(self.sample_rate, audio)
            self.recognizer.decode_stream(stream)
            text = stream.result.text.strip()
        infer = time.perf_counter() - started

        audio_sec = audio.size / self.sample_rate
        pieces = text.split()
        step = audio_sec / len(pieces) if pieces else 0.0
        words = [
            Word(p, t0 + i * step, t0 + (i + 1) * step) for i, p in enumerate(pieces)
        ]
        return ASRResult(
            text=text,
            words=words,
            language=self.config.language,
            audio_sec=audio_sec,
            infer_sec=infer,
            is_final=is_final,
        )
