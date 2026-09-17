"""sherpa-onnx streaming Zipformer 백엔드 (진짜 streaming ASR).

프레임 단위로 accept_waveform 하며 매 호출마다 부분 결과를 갱신하고,
sherpa-onnx 내장 endpoint 검출을 그대로 쓴다. 슬라이딩 윈도우가 필요 없어
first-partial latency 가 Whisper 대비 크게 낮다.

모델 준비:
    models/zipformer/ 아래에 encoder/decoder/joiner .onnx 와 tokens.txt 배치
    (k2-fsa/sherpa-onnx 릴리스의 streaming zipformer 모델)
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..config import MODELS_DIR, StreamConfig
from .base import ASREngine, ASRResult, Word

_HINT = (
    "sherpa-onnx 와 streaming zipformer 모델이 필요합니다.\n"
    "  pip install sherpa-onnx\n"
    "  models/zipformer/ 에 encoder/decoder/joiner(.onnx) 와 tokens.txt 를 두세요.\n"
    "  모델: https://github.com/k2-fsa/sherpa-onnx/releases (streaming-zipformer)"
)


def _find(directory: Path, *keywords: str) -> str:
    """디렉터리에서 키워드가 모두 들어간 파일을 찾는다(int8 우선)."""
    cands = [
        p for p in sorted(directory.glob("*"))
        if p.is_file() and all(k in p.name for k in keywords)
    ]
    if not cands:
        raise FileNotFoundError(
            f"{directory} 에서 {'/'.join(keywords)} 파일을 찾지 못했습니다.\n{_HINT}"
        )
    cands.sort(key=lambda p: (0 if "int8" in p.name else 1, len(p.name)))
    return str(cands[0])


class ZipformerEngine(ASREngine):
    name = "zipformer"
    native_streaming = True
    has_word_timestamps = True

    def __init__(self, config: StreamConfig) -> None:
        super().__init__(config)
        try:
            import sherpa_onnx  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(_HINT) from e

        model_dir = Path(config.model_dir or (MODELS_DIR / "zipformer"))
        if not model_dir.is_dir():
            raise FileNotFoundError(f"모델 디렉터리가 없습니다: {model_dir}\n{_HINT}")

        self._sherpa = sherpa_onnx
        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=_find(model_dir, "tokens"),
            encoder=_find(model_dir, "encoder"),
            decoder=_find(model_dir, "decoder"),
            joiner=_find(model_dir, "joiner"),
            num_threads=2,
            sample_rate=self.sample_rate,
            feature_dim=80,
            decoding_method="greedy_search",
            provider="cuda" if config.device == "cuda" else "cpu",
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=2.4,
            rule2_min_trailing_silence=max(0.4, config.silence_sec),
            rule3_min_utterance_length=config.max_utterance_sec,
        )
        self.stream = self.recognizer.create_stream()
        self._t0 = 0.0          # 현재 발화 시작 절대 시각
        self._fed = 0           # 현재 발화에 넣은 샘플 수
        self._infer_sec = 0.0

    # ------------------------------------------------------------ streaming
    def accept_waveform(self, audio: np.ndarray) -> None:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        started = time.perf_counter()
        self.stream.accept_waveform(self.sample_rate, audio)
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)
        self._infer_sec += time.perf_counter() - started
        self._fed += audio.size

    def partial(self) -> ASRResult:
        res = self.recognizer.get_result(self.stream)
        text = (res if isinstance(res, str) else getattr(res, "text", "")).strip()
        words = self._to_words(text)
        result = ASRResult(
            text=text,
            words=words,
            language=self.config.language,
            audio_sec=self._fed / self.sample_rate,
            infer_sec=self._infer_sec,
        )
        self._infer_sec = 0.0
        return result

    def is_endpoint(self) -> bool:
        return bool(self.recognizer.is_endpoint(self.stream))

    def reset_stream(self) -> None:
        self.recognizer.reset(self.stream)
        self._t0 += self._fed / self.sample_rate
        self._fed = 0

    def set_stream_origin(self, t0: float) -> None:
        self._t0 = t0

    def _to_words(self, text: str) -> list[Word]:
        """토큰 타임스탬프가 있으면 쓰고, 없으면 균등 분배로 근사."""
        try:
            timestamps = list(self.recognizer.get_result(self.stream).timestamps)
            tokens = list(self.recognizer.get_result(self.stream).tokens)
        except Exception:
            timestamps, tokens = [], []

        if tokens and len(tokens) == len(timestamps):
            words: list[Word] = []
            for tok, ts in zip(tokens, timestamps):
                tok = tok.replace("▁", " ")
                if not tok.strip():
                    continue
                words.append(Word(tok.strip(), self._t0 + ts, self._t0 + ts + 0.05))
            return words

        pieces = text.split()
        if not pieces:
            return []
        dur = self._fed / self.sample_rate
        step = dur / len(pieces)
        return [
            Word(p, self._t0 + i * step, self._t0 + (i + 1) * step)
            for i, p in enumerate(pieces)
        ]

    # ------------------------------------------------------------ windowed
    def transcribe(
        self, audio: np.ndarray, t0: float = 0.0, is_final: bool = False
    ) -> ASRResult:
        """비스트리밍 호출(파일 전사 등)을 위한 일회성 디코딩."""
        stream = self.recognizer.create_stream()
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        started = time.perf_counter()
        stream.accept_waveform(self.sample_rate, audio)
        tail = np.zeros(int(self.sample_rate * 0.5), dtype=np.float32)
        stream.accept_waveform(self.sample_rate, tail)
        stream.input_finished()
        while self.recognizer.is_ready(stream):
            self.recognizer.decode_stream(stream)
        res = self.recognizer.get_result(stream)
        text = (res if isinstance(res, str) else getattr(res, "text", "")).strip()
        return ASRResult(
            text=text,
            words=[],
            language=self.config.language,
            audio_sec=audio.size / self.sample_rate,
            infer_sec=time.perf_counter() - started,
            is_final=is_final,
        )
