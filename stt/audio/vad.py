"""에너지 기반 VAD + endpoint 검출.

추가 의존성 없이 동작하도록 적응형 노이즈 플로어를 쓰는 RMS VAD 를 기본으로 하고,
`webrtcvad` 가 설치돼 있으면 그쪽을 사용한다(더 정확함).

endpoint = "말이 있었고, 그 뒤로 silence_sec 이상 무음" -> 발화 확정(final) 신호.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import SAMPLE_RATE
from .resampler import float32_to_pcm16, rms_dbfs

try:
    import webrtcvad  # type: ignore
except Exception:      # pragma: no cover
    webrtcvad = None


@dataclass
class VadState:
    is_speech: bool = False
    level_db: float = -120.0
    noise_db: float = -60.0
    speech_sec: float = 0.0
    silence_sec: float = 0.0


class EnergyVAD:
    """적응형 노이즈 플로어를 쓰는 RMS VAD.

    노이즈 플로어는 무음 구간에서만 천천히 따라가고(attack 느림),
    말이 시작되면 갱신을 멈춰 플로어가 음성에 끌려 올라가지 않게 한다.
    """

    #: 노이즈 플로어가 아무리 낮아도 이보다 조용하면 음성으로 보지 않는다(dBFS).
    ABSOLUTE_FLOOR_DB = -55.0

    def __init__(
        self,
        threshold_db: float = 12.0,
        sample_rate: int = SAMPLE_RATE,
        use_webrtc: bool = True,
        aggressiveness: int = 2,
    ) -> None:
        self.threshold_db = threshold_db
        self.sample_rate = sample_rate
        self.noise_db = -60.0

        self._webrtc = None
        if use_webrtc and webrtcvad is not None and sample_rate in (8000, 16000, 32000, 48000):
            self._webrtc = webrtcvad.Vad(aggressiveness)

    def _webrtc_is_speech(self, block: np.ndarray) -> bool:
        """webrtcvad 는 10/20/30 ms 프레임만 받으므로 20 ms 로 잘라 다수결."""
        frame = int(self.sample_rate * 0.02)
        if block.size < frame:
            return False
        votes = []
        for i in range(0, block.size - frame + 1, frame):
            pcm = float32_to_pcm16(block[i:i + frame])
            try:
                votes.append(self._webrtc.is_speech(pcm, self.sample_rate))
            except Exception:
                return False
        return bool(votes) and sum(votes) / len(votes) >= 0.4

    def __call__(self, block: np.ndarray) -> tuple[bool, float]:
        """(is_speech, level_db) 반환."""
        level = rms_dbfs(block)
        gate = max(self.noise_db + self.threshold_db, self.ABSOLUTE_FLOOR_DB)

        if self._webrtc is not None:
            speech = self._webrtc_is_speech(
                np.asarray(block, dtype=np.float32).reshape(-1)
            ) and level > self.ABSOLUTE_FLOOR_DB
        else:
            speech = level > gate

        if not speech:                      # 무음일 때만 플로어 추종
            self.noise_db = 0.95 * self.noise_db + 0.05 * level
        self.noise_db = float(np.clip(self.noise_db, -90.0, -20.0))
        return speech, level


class EndpointDetector:
    """VAD 결과를 누적해 발화 시작/종료(endpoint)를 판정."""

    def __init__(
        self,
        silence_sec: float = 0.7,
        min_speech_sec: float = 0.3,
        max_utterance_sec: float = 20.0,
        threshold_db: float = 12.0,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self.vad = EnergyVAD(threshold_db=threshold_db, sample_rate=sample_rate)
        self.silence_sec = silence_sec
        self.min_speech_sec = min_speech_sec
        self.max_utterance_sec = max_utterance_sec
        self.sample_rate = sample_rate
        self.state = VadState()

    def reset(self) -> None:
        self.state = VadState(noise_db=self.vad.noise_db)

    def push(self, block: np.ndarray) -> tuple[bool, VadState]:
        """블록을 넣고 (endpoint 발생 여부, 상태) 반환."""
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        dur = block.size / self.sample_rate
        speech, level = self.vad(block)

        st = self.state
        st.is_speech = speech
        st.level_db = level
        st.noise_db = self.vad.noise_db

        if speech:
            st.speech_sec += dur
            st.silence_sec = 0.0
        else:
            st.silence_sec += dur

        endpoint = (
            st.speech_sec >= self.min_speech_sec
            and st.silence_sec >= self.silence_sec
        ) or (st.speech_sec >= self.max_utterance_sec)

        if endpoint:
            self.reset()
        return endpoint, st
