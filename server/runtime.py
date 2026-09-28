"""STT 서버 런타임 — 활성 설정(source of truth) + 모델 preload + readiness.

서버 프로세스당 하나(`RUNTIME`). 서버가 시작할 때 환경변수를 **한 번** 읽어 활성
설정을 고정하고, 그 설정으로 엔진을 미리 올려 warm-up 한다. preload, `/stream`
세션, readiness/engines API 가 모두 이 객체의 설정을 보므로 서로 다른 엔진을
쓰는 일이 없다.

    FastAPI startup → configure() → start_preload()
        → create_engine(active config)   모델 로드 · GPU 적재
        → engine.warmup(strict=True)     실제 추론 1회
        → READY

환경변수
    STT_ENGINE       funasr_mlt_nano(기본) | whisper | zipformer | sensevoice
    STT_MODEL_SIZE   whisper 모델 크기
    STT_CONFIG       StreamConfig 필드 JSON (예: {"silence_sec":0.6,"save_wav":false})
    STT_TIMESTAMPS   /stream final 에 start_time/end_time 포함 (기본 1)
    STT_PRELOAD      시작 시 모델 preload (기본 1). 0 이면 첫 연결에서 로드(개발용)
"""
from __future__ import annotations

import copy
import json
import logging
import os
import threading
import time

from stt.asr.base import ASREngine, create_engine
from stt.config import StreamConfig

log = logging.getLogger("stt")

LOADING = "LOADING"
READY = "READY"
NOT_READY = "NOT_READY"


def _config_from(payload: dict) -> StreamConfig:
    """start 페이로드 / STT_CONFIG 같은 dict 에서 StreamConfig 를 만든다."""
    cfg = StreamConfig()
    for key, value in payload.items():
        if key in ("type", "sample_rate"):
            continue
        if hasattr(cfg, key) and value is not None:
            current = getattr(cfg, key)
            try:
                setattr(cfg, key, type(current)(value) if current is not None else value)
            except (TypeError, ValueError):
                setattr(cfg, key, value)
    return cfg


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def load_stream_config() -> StreamConfig:
    """환경변수(STT_CONFIG → STT_ENGINE/STT_MODEL_SIZE 순으로 우선)에서 설정을 만든다."""
    overrides: dict = {}
    raw = os.getenv("STT_CONFIG", "").strip()
    if raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            overrides.update(parsed)
        else:
            log.warning("[STT] STT_CONFIG 는 JSON 객체여야 합니다 — 무시합니다")
    for env, key in (("STT_ENGINE", "engine"), ("STT_MODEL_SIZE", "model_size")):
        if os.getenv(env):
            overrides[key] = os.environ[env]
    return _config_from(overrides)


class Runtime:
    """활성 설정과 preload 상태. 모든 필드는 `_lock` 아래에서 바뀐다."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._settled = threading.Event()   # preload 가 끝났다(성공이든 실패든)
        self._settled.set()
        self.config = load_stream_config()
        self.timestamps = True
        self.preload = True
        self.status = NOT_READY
        self.error: str | None = "server not started"
        self.model_loaded = False
        self.load_sec: float | None = None
        self.warmup_sec: float | None = None
        self._engine: ASREngine | None = None

    # ------------------------------------------------------------ 설정
    def configure(self) -> None:
        """환경변수에서 활성 설정을 읽어 고정한다. 서버 시작 시 한 번 부른다."""
        config = load_stream_config()
        with self._lock:
            self.config = config
            self.timestamps = _flag("STT_TIMESTAMPS", True)
            self.preload = _flag("STT_PRELOAD", True)
            self.model_loaded = False
            self.load_sec = self.warmup_sec = None
            self._engine = None
            if self.preload:
                self.status, self.error = LOADING, None
                self._settled.clear()
            else:
                self.status, self.error = NOT_READY, "preload disabled (STT_PRELOAD=0)"
                self._settled.set()
        log.info("[STT] Active engine: %s", config.engine)
        log.info("[STT] Device: %s (compute_type=%s)", config.device, config.compute_type)
        log.info("[STT] Timestamps: %s, preload: %s", self.timestamps, self.preload)

    def stream_config(self) -> StreamConfig:
        """세션용 활성 설정 사본(세션이 바꿔도 원본에 영향 없음)."""
        with self._lock:
            return copy.deepcopy(self.config)

    # ------------------------------------------------------------ preload
    def start_preload(self) -> None:
        """preload 를 백그라운드 스레드로 시작한다. liveness 는 그동안에도 응답한다."""
        if not self.preload:
            log.info("[STT] Preload disabled — model loads on first connection")
            return
        threading.Thread(target=self._preload, name="stt-preload", daemon=True).start()

    def _preload(self) -> None:
        config = self.stream_config()
        try:
            log.info("[STT] Loading model... (engine=%s)", config.engine)
            started = time.perf_counter()
            engine = create_engine(config)
            load_sec = time.perf_counter() - started
            log.info("[STT] Model loaded (%.1f s)", load_sec)

            log.info("[STT] Warm-up started")
            warmup_sec = engine.warmup(strict=True)
            log.info("[STT] Warm-up completed (%.1f s)", warmup_sec)
        except Exception as e:
            # 상세 원인은 로그에만. API 에는 예외 종류만 알린다(경로 등 비노출).
            log.exception("[STT] Model preload failed — server is NOT_READY")
            with self._lock:
                self.status = NOT_READY
                self.error = f"{type(e).__name__}: model initialization failed (see server log)"
        else:
            with self._lock:
                self._engine = engine            # 모델 참조 유지
                self.model_loaded = True
                self.load_sec, self.warmup_sec = round(load_sec, 2), round(warmup_sec, 2)
                self.status, self.error = READY, None
            log.info("[STT] Server READY")
        finally:
            self._settled.set()

    def wait_settled(self, timeout: float | None = None) -> bool:
        """preload 가 끝날 때까지 기다린다. 첫 연결이 모델을 한 번 더 올리지 않게 한다."""
        return self._settled.wait(timeout)

    def is_warm(self, config: StreamConfig, engine: ASREngine) -> bool:
        """이 세션 엔진이 preload·warm-up 된 공유 모델을 그대로 쓰는지."""
        with self._lock:
            active = self.config
            return (
                self.status == READY
                and engine.shared_model
                and config.engine == active.engine
                and config.device == active.device
                and config.compute_type == active.compute_type
                and (config.engine != "whisper" or config.model_size == active.model_size)
            )

    # ------------------------------------------------------------ 조회
    @property
    def ready(self) -> bool:
        return self.status == READY

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "status": self.status,
                "engine": self.config.engine,
                "device": self.config.device,
                "model_loaded": self.model_loaded,
                "error": self.error,
                "load_sec": self.load_sec,
                "warmup_sec": self.warmup_sec,
            }


#: 프로세스 전역 런타임
RUNTIME = Runtime()
