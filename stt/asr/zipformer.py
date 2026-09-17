"""sherpa-onnx streaming Zipformer 백엔드 (진짜 streaming ASR).

프레임 단위로 accept_waveform 하며 매 호출마다 부분 결과를 갱신하고,
sherpa-onnx 내장 endpoint 검출을 그대로 쓴다. 슬라이딩 윈도우가 필요 없어
first-partial latency 가 Whisper 대비 크게 낮다.

모델 준비:
    python -m scripts.fetch_models zipformer
    → models/zipformer/ (sherpa-onnx-streaming-zipformer-korean-2024-06-16)
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
    "  python -m scripts.fetch_models zipformer\n"
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
        self._fed = 0           # 현재 발화에 넣은 샘플 수 (누적)
        self._fed_delta = 0     # 직전 partial() 이후 넣은 샘플 수
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
        self._fed_delta += audio.size

    def partial(self) -> ASRResult:
        words = self._read_result()
        # RTF 는 "이번에 넣은 오디오"와 "이번에 쓴 시간"의 비여야 한다.
        # 누적 오디오로 나누면 발화가 길어질수록 RTF 가 0 으로 수렴해 버린다.
        result = ASRResult(
            text=_join(words),
            words=words,
            language=self.config.language,
            audio_sec=self._fed_delta / self.sample_rate,
            infer_sec=self._infer_sec,
        )
        self._infer_sec = 0.0
        self._fed_delta = 0
        return result

    def is_endpoint(self) -> bool:
        return bool(self.recognizer.is_endpoint(self.stream))

    def reset_stream(self) -> None:
        self.recognizer.reset(self.stream)
        self._t0 += self._fed / self.sample_rate
        self._fed = 0
        self._fed_delta = 0

    def set_stream_origin(self, t0: float) -> None:
        self._t0 = t0

    # ------------------------------------------------------------ 결과 해석
    def _read_result(self, stream=None, t0: float | None = None) -> list[Word]:
        """스트림의 현재 가설을 어절 단위 Word 리스트로 읽는다."""
        stream = self.stream if stream is None else stream
        t0 = self._t0 if t0 is None else t0

        # get_result() 는 공백이 지워진 문자열만 준다. 어절 경계를 살리려면
        # 토큰/타임스탬프가 함께 들어 있는 전체 결과 객체가 필요하다.
        try:
            res = self.recognizer.get_result_all(stream)
            tokens, timestamps = list(res.tokens), list(res.timestamps)
        except Exception:
            tokens, timestamps = [], []

        if tokens and len(tokens) == len(timestamps):
            return _words_from_tokens(tokens, timestamps, t0)

        # 폴백: 토큰을 못 얻으면 통짜 문자열이라도 돌려준다.
        text = str(self.recognizer.get_result(stream)).strip()
        return [Word(text, t0, t0 + self._fed / self.sample_rate)] if text else []

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
        stream.accept_waveform(self.sample_rate, tail)   # 마지막 토큰을 밀어낸다
        stream.input_finished()
        while self.recognizer.is_ready(stream):
            self.recognizer.decode_stream(stream)
        words = self._read_result(stream, t0)
        return ASRResult(
            text=_join(words),
            words=words,
            language=self.config.language,
            audio_sec=audio.size / self.sample_rate,
            infer_sec=time.perf_counter() - started,
            is_final=is_final,
        )


def _words_from_tokens(tokens: list[str], timestamps: list[float], t0: float) -> list[Word]:
    """BPE 토큰열을 어절 단위로 묶는다.

    한국어 zipformer 의 tokens.txt 는 어절 시작을 `▁` 로 표시하고, sherpa-onnx 는
    이를 **선행 공백**으로 바꿔서 준다. 그래서 공백으로 시작하는 토큰이 새 어절이다.

        [' 걔는', ' 괜찮은', ' 척', '하', '려', '구', ...]
          → ['걔는', '괜찮은', '척하려구', ...]

    이 경계를 살리지 않으면 "걔는괜찮은척하려구" 처럼 공백 없이 붙어 나온다.
    """
    words: list[Word] = []
    for token, ts in zip(tokens, timestamps):
        piece = token.replace("▁", " ")
        text = piece.strip()
        if not text:
            continue
        if piece.startswith(" ") or not words:
            words.append(Word(text, t0 + ts, t0 + ts))
        else:                                   # 같은 어절의 뒤따르는 토큰
            words[-1].text += text
            words[-1].end = t0 + ts

    # end 는 다음 어절의 시작으로 메운다(마지막 어절만 약간의 여유를 준다).
    for i, word in enumerate(words):
        nxt = words[i + 1].start if i + 1 < len(words) else word.end + 0.2
        word.end = max(word.end, nxt)
    return words


def _join(words: list[Word]) -> str:
    return " ".join(w.text for w in words)
