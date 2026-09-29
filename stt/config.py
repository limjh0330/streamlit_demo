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
# Streamlit 이 STT 백엔드에 닿는 주소는 두 가지다.
#   - Python → 백엔드(REST, 엔진 목록 조회) : API_URL
#   - 브라우저 → 백엔드(마이크 WebSocket)    : WS_URL
# 두 구성을 지원한다.
#   A. Streamlit 도 RunPod 에서 실행 : REST 는 localhost, 브라우저는 RunPod 공개 프록시(RUNPOD_POD_ID 로 계산)
#   B. Streamlit 은 로컬 PC 에서 실행 : STT_SERVER_URL=https://<POD_ID>-8000.proxy.runpod.net 하나만 주면
#                                     REST·WebSocket 모두 공개 프록시로 간다(VS Code 포트 포워딩 불필요)
# RunPod 은 포트마다 별도 호스트(<podId>-<port>.proxy.runpod.net)를 준다.
BACKEND_PORT = int(os.getenv("STT_BACKEND_PORT", "8000"))
#: 원격 STT 서버의 기본 주소 (구성 B). 예: https://druyf5wybhan4k-8000.proxy.runpod.net
SERVER_URL = os.getenv("STT_SERVER_URL", "").strip().rstrip("/")
API_URL = os.getenv("STT_API_URL", "").strip() or SERVER_URL or f"http://127.0.0.1:{BACKEND_PORT}"

# ---------------------------------------------------------------- 엔드포인트 경로
# REST 는 API_BASE, WebSocket 은 WS_BASE 아래에 둔다(서버·Streamlit·브라우저 공용).
API_BASE = "/api/v1/stt"
WS_BASE = "/ws/v1/stt"
WS_STREAM_PATH = f"{WS_BASE}/stream"      # External Backend (PCM16 → transcript)
WS_BROWSER_PATH = f"{WS_BASE}/browser"    # Streamlit 실시간 전사 페이지

#: RunPod 이 컨테이너에 넣어 주는 Pod ID (로컬에서는 빈 값)
RUNPOD_POD_ID = os.getenv("RUNPOD_POD_ID", "").strip()


def public_ws_url(path: str = WS_BROWSER_PATH) -> str:
    """RunPod 공개 프록시의 WebSocket 주소. Pod 밖이면 빈 문자열."""
    if not RUNPOD_POD_ID:
        return ""
    return f"wss://{RUNPOD_POD_ID}-{BACKEND_PORT}.proxy.runpod.net{path}"


def ws_url_from(http_url: str, path: str = WS_BROWSER_PATH) -> str:
    """http(s)://host → ws(s)://host + path."""
    if not http_url:
        return ""
    scheme, sep, rest = http_url.partition("://")
    ws_scheme = "wss" if scheme == "https" else "ws"
    return f"{ws_scheme}{sep}{rest.rstrip('/')}{path}" if sep else ""


# 브라우저 → 백엔드(오디오) 주소.
# 우선순위: STT_WS_URL > STT_SERVER_URL(구성 B) > RunPod 공개 프록시(구성 A) > 주소창에서 유도(빈 값).
# 주소창에서 유도하면 Streamlit 을 VS Code 포트 포워딩(http://localhost:8501)으로 연 경우
# ws://localhost:8000 이 되어, VS Code 창을 닫는 순간 포워딩이 끊기고 마이크 연결이 실패한다.
WS_URL = os.getenv("STT_WS_URL", "").strip() or ws_url_from(SERVER_URL) or public_ws_url()

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


#: 설정을 주지 않았을 때 쓰는 엔진 (/ws/v1/stt/stream 기본값 포함)
DEFAULT_ENGINE = "funasr_mlt_nano"


@dataclass
class StreamConfig:
    """세션 하나의 스트리밍 파라미터."""

    # ASR 엔진
    engine: str = DEFAULT_ENGINE          # funasr_mlt_nano | whisper | zipformer | sensevoice
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
    refresh_sec: float = 0.8              # 발화 전체 재인식 엔진(SenseVoice)의 갱신 간격

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


ENGINE_CHOICES = ("whisper", "zipformer", "sensevoice", "funasr_mlt_nano")
WHISPER_SIZES = ("tiny", "base", "small", "medium", "large-v3")
