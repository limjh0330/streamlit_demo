"""전역 오디오/스트리밍 설정.

개발문서 기준값:
  - PCM16 / 16 kHz / mono
  - 100~500 ms audio chunk (브라우저 AudioWorklet)
  - 5~6 s sliding window + 1~2 s overlap
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ---------------------------------------------------------------- 오디오 규격
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "float32"
BLOCK_DURATION = 0.1                      # 100 ms
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION)
CHUNK_MS_CHOICES = (100, 200, 250, 500)   # 브라우저가 보내는 청크 길이

# ---------------------------------------------------------------- 백엔드 주소
# Streamlit 과 STT 백엔드는 같은 RunPod 인스턴스에서 서로 다른 포트로 돈다.
#   - Python → 백엔드(상태 조회): 같은 호스트이므로 localhost
#   - 브라우저 → 백엔드(오디오):  외부에서 접근 가능한 주소가 필요
# RunPod 은 포트마다 별도 호스트(<podId>-<port>.proxy.runpod.net)를 주므로
# WS_URL 이 비어 있으면 브라우저가 주소창에서 유도한다.
BACKEND_PORT = int(os.getenv("STT_BACKEND_PORT", "8000"))
API_URL = os.getenv("STT_API_URL", f"http://127.0.0.1:{BACKEND_PORT}")
WS_URL = os.getenv("STT_WS_URL", "")

# ---------------------------------------------------------------- 경로
ROOT = Path(__file__).resolve().parent.parent
RECORDINGS_DIR = ROOT / "recordings"
TRANSCRIPTS_DIR = ROOT / "transcripts"
MODELS_DIR = ROOT / "models"

for _d in (RECORDINGS_DIR, TRANSCRIPTS_DIR, MODELS_DIR):
    _d.mkdir(exist_ok=True)


# ---------------------------------------------------------------- 실행 장치
def default_device() -> str:
    """RunPod GPU 인스턴스면 cuda, 로컬 Mac 이면 cpu.

    faster-whisper 는 CTranslate2 위에 있으므로 torch 대신 CTranslate2 에 묻는다.
    """
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"


def default_compute_type(device: str | None = None) -> str:
    """GPU 는 float16, CPU 는 int8 이 기본."""
    return "float16" if (device or default_device()) == "cuda" else "int8"


@dataclass
class StreamConfig:
    """세션 하나의 스트리밍 파라미터."""

    # ASR 엔진
    engine: str = "whisper"               # whisper | zipformer | sensevoice
    model_size: str = "small"             # whisper 전용
    model_dir: str | None = None          # zipformer / sensevoice 모델 디렉터리
    device: str = field(default_factory=default_device)          # cpu | cuda
    compute_type: str = field(default_factory=default_compute_type)  # ct2 quantization
    language: str | None = "ko"
    beam_size: int = 1                    # partial 윈도우: 지연을 위해 1
    final_beam_size: int = 5              # 발화 확정 시 1회만: 품질을 위해 5

    # sliding window
    window_sec: float = 5.0               # 개발문서: 5~6 s
    overlap_sec: float = 1.5              # 개발문서: 1~2 s
    min_window_sec: float = 1.0           # 이보다 짧으면 추론하지 않음
    first_hop_sec: float = 1.5            # 발화 시작 직후 첫 윈도우 (first partial latency)

    # VAD / endpoint
    vad_enabled: bool = True
    silence_sec: float = 0.7              # 이만큼 무음이면 발화 종료(final)
    max_utterance_sec: float = 20.0       # 무음이 안 와도 강제 확정
    vad_threshold_db: float = 12.0        # 노이즈 플로어 대비 dB 마진

    # 후처리
    medical_correction: bool = True
    use_initial_prompt: bool = True

    # 저장
    save_wav: bool = True

    @property
    def hop_sec(self) -> float:
        return max(0.2, self.window_sec - self.overlap_sec)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["hop_sec"] = self.hop_sec
        return d


ENGINE_CHOICES = ("whisper", "zipformer", "sensevoice")
WHISPER_SIZES = ("tiny", "base", "small", "medium", "large-v3")
