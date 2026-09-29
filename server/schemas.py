"""REST 응답 스키마 — Swagger UI(`/api/v1/stt/docs`)에 표시되는 모델과 예시.

응답 형식 자체는 바꾸지 않는다. 기존 dict 응답의 필드를 그대로 모델로 옮겨
문서화·검증에만 쓴다.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """liveness — 프로세스가 요청을 받을 수 있으면 항상 200."""

    status: Literal["OK"] = Field(description="항상 `OK`")
    sample_rate: int = Field(description="오디오 규격 샘플레이트 (Hz)")
    active_sessions: list[str] = Field(description="진행 중인 STT 세션 ID")
    gpu_memory_mb: float | None = Field(description="torch 가 할당한 GPU 메모리 (MB). GPU 없으면 null")

    model_config = {"json_schema_extra": {"examples": [
        {"status": "OK", "sample_rate": 16000, "active_sessions": [], "gpu_memory_mb": 2500.0}
    ]}}


class ReadyResponse(BaseModel):
    """readiness — 활성 엔진의 로드·warm-up 상태."""

    status: Literal["READY", "LOADING", "NOT_READY"] = Field(
        description="`READY`(200) 일 때만 트래픽을 보낸다. `LOADING`/`NOT_READY` 는 503")
    engine: str = Field(description="서버 활성 엔진")
    device: str = Field(description="`cuda` 또는 `cpu`")
    model_loaded: bool = Field(description="모델 로드 + warm-up 추론까지 끝났는지")
    gpu_available: bool
    gpu_memory_mb: float | None
    sample_rate: int
    active_sessions: int = Field(description="진행 중인 세션 수")
    load_sec: float | None = Field(description="모델 로드 소요 시간 (초)")
    warmup_sec: float | None = Field(description="warm-up 추론 소요 시간 (초)")
    output_time: float = Field(description="응답 생성 시각 (Unix epoch, 초)")
    error: str | None = Field(None, description="`NOT_READY` 원인. 예외 종류만 담고 상세는 서버 로그에")

    model_config = {"json_schema_extra": {"examples": [
        {"status": "READY", "engine": "funasr_mlt_nano", "device": "cuda", "model_loaded": True,
         "gpu_available": True, "gpu_memory_mb": 2500.0, "sample_rate": 16000,
         "active_sessions": 0, "load_sec": 15.0, "warmup_sec": 1.2, "output_time": 1788095605.9}
    ]}}


READY_RESPONSES: dict[int | str, dict[str, Any]] = {
    200: {"model": ReadyResponse, "description": "READY — 추론을 받을 수 있음"},
    503: {
        "model": ReadyResponse,
        "description": "LOADING(모델 로딩 중) 또는 NOT_READY(초기화 실패·preload 비활성)",
        "content": {"application/json": {"examples": {
            "loading": {"summary": "로딩 중", "value": {
                "status": "LOADING", "engine": "funasr_mlt_nano", "device": "cuda",
                "model_loaded": False, "gpu_available": True, "gpu_memory_mb": 0.0,
                "sample_rate": 16000, "active_sessions": 0, "load_sec": None,
                "warmup_sec": None, "output_time": 1788095590.1}},
            "not_ready": {"summary": "초기화 실패", "value": {
                "status": "NOT_READY", "engine": "funasr_mlt_nano", "device": "cuda",
                "model_loaded": False, "gpu_available": True, "gpu_memory_mb": 0.0,
                "sample_rate": 16000, "active_sessions": 0, "load_sec": None,
                "warmup_sec": None, "output_time": 1788095590.1,
                "error": "FileNotFoundError: model initialization failed (see server log)"}},
        }}},
    },
}


class EngineInfo(BaseModel):
    ready: bool = Field(description="패키지·모델 파일이 준비돼 선택 가능한지")
    detail: str
    loaded: bool = Field(description="서버 활성 엔진으로 메모리에 올라가 있는지")


class EnginesResponse(BaseModel):
    active_engine: str = Field(description="서버가 실제로 쓰는 엔진 (= preload 엔진 = /stream 세션 엔진)")
    active_status: Literal["READY", "LOADING", "NOT_READY"]
    active_config: dict[str, Any] = Field(description="서버 활성 StreamConfig")
    engines: dict[str, EngineInfo]
    whisper_sizes: list[str]
    defaults: dict[str, Any] = Field(description="설정을 주지 않았을 때의 코드 기본값")


class SessionsResponse(BaseModel):
    active: list[str] = Field(description="진행 중인 세션 ID")
    saved: list[str] = Field(description="저장된 전사 세션 ID (최근 50개, 최신순)")

    model_config = {"json_schema_extra": {"examples": [
        {"active": [], "saved": ["20260928-215345-d8a74a", "20260927-172618-6ed053"]}
    ]}}


class ErrorResponse(BaseModel):
    detail: str


TRANSCRIPT_RESPONSES: dict[int | str, dict[str, Any]] = {
    200: {"description": "세션 전사 결과 (`transcripts/<session_id>.json`)",
          "content": {"application/json": {"example": {
              "session_id": "20260928-215345-d8a74a", "engine": "funasr_mlt_nano",
              "stable": "…", "partial": "",
              "utterances": [{"text": "…", "start": 0.0, "end": 7.0}],
              "metrics": {"rtf_mean": 0.42, "first_partial_ms": 967.7, "final_latency_ms": 1639.1},
              "wav": None, "error": None}}}},
    404: {"model": ErrorResponse, "description": "해당 세션의 전사 결과가 없음"},
}

AUDIO_RESPONSES: dict[int | str, dict[str, Any]] = {
    200: {"description": "세션 녹음 WAV (PCM16 / 16 kHz / mono)", "content": {"audio/wav": {}}},
    404: {"model": ErrorResponse, "description": "해당 세션의 녹음 파일이 없음 (`save_wav=false` 등)"},
}
