"""PCM16 <-> float32 변환 및 리샘플링.

브라우저(AudioWorklet)는 PCM16 / 16 kHz mono 를 보내지만,
디바이스에 따라 다른 샘플레이트가 올라올 수 있으므로 방어적으로 변환한다.
"""
from __future__ import annotations

import numpy as np

try:                                       # 있으면 품질 좋은 폴리페이즈 리샘플 사용
    from scipy.signal import resample_poly
except Exception:                          # pragma: no cover
    resample_poly = None


def pcm16_to_float32(data: bytes | np.ndarray) -> np.ndarray:
    """little-endian int16 바이트 -> [-1, 1] float32 mono 배열."""
    if isinstance(data, (bytes, bytearray, memoryview)):
        arr = np.frombuffer(bytes(data), dtype="<i2")
    else:
        arr = np.asarray(data, dtype=np.int16)
    return (arr.astype(np.float32) / 32768.0).copy()


def float32_to_pcm16(audio: np.ndarray) -> bytes:
    """[-1, 1] float32 -> little-endian int16 바이트."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def to_mono(audio: np.ndarray) -> np.ndarray:
    """(n, ch) 또는 (n,) 입력을 (n,) mono float32 로."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]
    return np.ascontiguousarray(audio, dtype=np.float32)


def resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """mono float32 리샘플링."""
    audio = to_mono(audio)
    if src_rate == dst_rate or audio.size == 0:
        return audio

    if resample_poly is not None:
        g = np.gcd(int(src_rate), int(dst_rate))
        up, down = int(dst_rate // g), int(src_rate // g)
        return np.ascontiguousarray(
            resample_poly(audio, up, down).astype(np.float32)
        )

    # scipy 가 없을 때의 선형 보간 폴백
    n_out = int(round(audio.size * dst_rate / src_rate))
    x_old = np.linspace(0.0, 1.0, num=audio.size, endpoint=False, dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False, dtype=np.float64)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def rms_dbfs(audio: np.ndarray) -> float:
    """블록의 RMS 레벨(dBFS). 무음은 -120 으로 클램프."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    return max(-120.0, 20.0 * np.log10(rms + 1e-12))
