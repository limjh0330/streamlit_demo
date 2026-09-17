"""지연/자원 사용 지표 수집.

개발문서 평가 metric 중 실시간으로 측정 가능한 항목:
  - Real-Time Factor (RTF)          = 추론시간 / 오디오길이
  - First partial latency           = 발화 시작 -> 첫 partial 이 뜨기까지
  - Final transcript latency        = 발화 종료(endpoint) -> final 확정까지
  - Partial transcript revision rate= partial 이 뒤집힌 비율 (merger 가 집계)
  - CPU usage / memory              = psutil 샘플링
"""
from __future__ import annotations

import statistics
import threading
import time
from dataclasses import dataclass, field

try:
    import psutil
except Exception:                       # pragma: no cover
    psutil = None


def _q(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(p * (len(ordered) - 1)))))
    return ordered[idx]


@dataclass
class LatencyTracker:
    """세션 하나의 지연 지표."""

    rtfs: list[float] = field(default_factory=list)
    infer_secs: list[float] = field(default_factory=list)
    first_partial: list[float] = field(default_factory=list)
    final_latency: list[float] = field(default_factory=list)

    _speech_started_at: float | None = None
    _partial_seen: bool = False
    _endpoint_at: float | None = None

    # -------------------------------------------------------------- 이벤트
    def on_speech_start(self) -> None:
        if self._speech_started_at is None:
            self._speech_started_at = time.perf_counter()
            self._partial_seen = False

    def on_partial(self, text: str) -> None:
        if text and not self._partial_seen and self._speech_started_at is not None:
            self.first_partial.append(time.perf_counter() - self._speech_started_at)
            self._partial_seen = True

    def on_endpoint(self) -> None:
        self._endpoint_at = time.perf_counter()

    def on_final(self) -> None:
        if self._endpoint_at is not None:
            self.final_latency.append(time.perf_counter() - self._endpoint_at)
            self._endpoint_at = None
        self._speech_started_at = None
        self._partial_seen = False

    def on_inference(self, audio_sec: float, infer_sec: float) -> None:
        self.infer_secs.append(infer_sec)
        if audio_sec > 0:
            self.rtfs.append(infer_sec / audio_sec)

    # -------------------------------------------------------------- 요약
    def summary(self) -> dict:
        return {
            "rtf_mean": round(statistics.fmean(self.rtfs), 3) if self.rtfs else 0.0,
            "rtf_p95": round(_q(self.rtfs, 0.95), 3),
            "infer_mean_ms": round(statistics.fmean(self.infer_secs) * 1000, 1)
            if self.infer_secs else 0.0,
            "first_partial_ms": round(statistics.fmean(self.first_partial) * 1000, 1)
            if self.first_partial else 0.0,
            "first_partial_p95_ms": round(_q(self.first_partial, 0.95) * 1000, 1),
            "final_latency_ms": round(statistics.fmean(self.final_latency) * 1000, 1)
            if self.final_latency else 0.0,
            "final_latency_p95_ms": round(_q(self.final_latency, 0.95) * 1000, 1),
            "inferences": len(self.infer_secs),
        }


class ResourceSampler:
    """백그라운드 스레드로 CPU/메모리를 주기적으로 샘플링."""

    def __init__(self, interval: float = 1.0) -> None:
        self.interval = interval
        self.cpu: list[float] = []
        self.rss_mb: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc = psutil.Process() if psutil else None

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            if self._proc is None:
                continue
            try:
                self.cpu.append(self._proc.cpu_percent())
                self.rss_mb.append(self._proc.memory_info().rss / (1024 * 1024))
            except Exception:
                break

    def start(self) -> None:
        if self._proc is None or self._thread is not None:
            return
        self._proc.cpu_percent()             # 첫 호출은 0.0 이므로 버린다
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def summary(self) -> dict:
        return {
            "cpu_mean_pct": round(statistics.fmean(self.cpu), 1) if self.cpu else 0.0,
            "cpu_max_pct": round(max(self.cpu), 1) if self.cpu else 0.0,
            "rss_mean_mb": round(statistics.fmean(self.rss_mb), 1) if self.rss_mb else 0.0,
            "rss_max_mb": round(max(self.rss_mb), 1) if self.rss_mb else 0.0,
            "samples": len(self.cpu),
        }


def gpu_memory_mb() -> float | None:
    """CUDA 가 있으면 할당된 GPU 메모리(MB)."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
    except Exception:
        pass
    return None
