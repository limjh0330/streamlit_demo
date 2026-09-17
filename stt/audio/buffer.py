"""sliding window + overlap 오디오 버퍼.

    ── window_sec ──┐
    [==============]                 t0
           [==============]          t0 + hop
                  [==============]   t0 + 2*hop
           └ overlap ┘

`push()` 로 블록을 넣고, `pop_window()` 가 추론할 구간을 돌려준다.
반환되는 t0 는 **세션 시작 기준 절대 시각(초)** 이라, 워드 타임스탬프를
그대로 절대 시각으로 환산해 병합기(merger)에 넘길 수 있다.
"""
from __future__ import annotations

import numpy as np

from ..config import SAMPLE_RATE


class SlidingWindowBuffer:
    def __init__(
        self,
        window_sec: float = 5.0,
        overlap_sec: float = 1.5,
        min_window_sec: float = 1.0,
        first_hop_sec: float = 1.5,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self.sample_rate = sample_rate
        self.window = int(window_sec * sample_rate)
        self.overlap = int(min(overlap_sec, window_sec * 0.9) * sample_rate)
        self.hop = max(int(0.2 * sample_rate), self.window - self.overlap)
        self.min_window = int(min_window_sec * sample_rate)
        # 발화 시작 직후 첫 윈도우만 짧게 끊어 first partial latency 를 줄인다.
        self.first_hop = max(self.min_window, min(int(first_hop_sec * sample_rate), self.hop))

        self._buf = np.zeros(0, dtype=np.float32)
        self._start_sample = 0        # _buf[0] 의 절대 샘플 인덱스
        self._total = 0               # 지금까지 push 된 총 샘플 수
        self._since_emit = 0          # 마지막 emit 이후 새로 들어온 샘플 수
        self._emitted = False         # 현재 발화에서 한 번이라도 emit 했는지

    # ------------------------------------------------------------------ 입력
    def push(self, block: np.ndarray) -> None:
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        if block.size == 0:
            return
        self._buf = np.concatenate([self._buf, block])
        self._total += block.size
        self._since_emit += block.size

    # ------------------------------------------------------------------ 출력
    def ready(self) -> bool:
        """추론을 돌릴 만큼 새 오디오가 쌓였는지."""
        needed = self.hop if self._emitted else self.first_hop
        return self._since_emit >= needed and self._buf.size >= self.min_window

    def pop_window(self, force: bool = False) -> tuple[np.ndarray, float] | None:
        """(audio, t0) 반환. force=True 면 hop 미달이어도 남은 걸 꺼낸다."""
        if self._buf.size == 0:
            return None
        if not force and not self.ready():
            return None
        if force and self._buf.size < int(0.2 * self.sample_rate):
            return None

        take = self._buf[-self.window:] if self._buf.size > self.window else self._buf
        offset = self._buf.size - take.size
        t0 = (self._start_sample + offset) / self.sample_rate

        self._since_emit = 0
        self._emitted = True
        self._trim()
        return take.copy(), t0

    # ---------------------------------------------------------------- 내부/제어
    def _trim(self) -> None:
        """window 를 넘는 앞부분은 버린다(overlap 은 자연히 남는다)."""
        if self._buf.size > self.window:
            drop = self._buf.size - self.window
            self._buf = self._buf[drop:]
            self._start_sample += drop

    def reset_to_now(self) -> None:
        """발화 확정(endpoint) 후 버퍼를 비운다."""
        self._start_sample = self._total
        self._buf = np.zeros(0, dtype=np.float32)
        self._since_emit = 0
        self._emitted = False

    @property
    def elapsed_sec(self) -> float:
        return self._total / self.sample_rate

    @property
    def buffered_sec(self) -> float:
        return self._buf.size / self.sample_rate
