"""전역 오디오/스트리밍 설정.

개발문서 기준값:
  - PCM16 / 16 kHz / mono
  - 50~100 ms audio block
  - 5~6 s sliding window + 1~2 s overlap
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

# ---------------------------------------------------------------- 오디오 규격
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "float32"
BLOCK_DURATION = 0.1                      # 100 ms
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION)

# ---------------------------------------------------------------- 경로
ROOT = Path(__file__).resolve().parent.parent
RECORDINGS_DIR = ROOT / "recordings"
TRANSCRIPTS_DIR = ROOT / "transcripts"
MODELS_DIR = ROOT / "models"

for _d in (RECORDINGS_DIR, TRANSCRIPTS_DIR, MODELS_DIR):
    _d.mkdir(exist_ok=True)


@dataclass
class StreamConfig:
    """세션 하나의 스트리밍 파라미터."""

    # ASR 엔진
    engine: str = "whisper"               # whisper | zipformer | sensevoice
    model_size: str = "small"             # whisper 전용
    model_dir: str | None = None          # zipformer / sensevoice 모델 디렉터리
    device: str = "cpu"                   # cpu | cuda | auto
    compute_type: str = "int8"            # ctranslate2 quantization
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
