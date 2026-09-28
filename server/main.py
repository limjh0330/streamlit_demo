"""STT 백엔드 진입점 (FastAPI) — RunPod 에서 Streamlit 과 나란히 돈다.

    ./start_server.sh                                   # RunPod 운영
    python -m server.main --host 0.0.0.0 --port 8000    # 직접 실행

시작하면 활성 엔진(기본 funasr_mlt_nano)을 백그라운드로 preload · warm-up 한다.
    GET /api/v1/stt/health         liveness  — 프로세스가 살아 있으면 200
    GET /api/v1/stt/health/ready   readiness — 모델이 추론 준비를 마쳤으면 200, 아니면 503

External Backend 는 `/ws/v1/stt/stream` 으로 붙는다(규격은 `server/backend_ws.py`).
브라우저는 `/ws/v1/stt/browser` 로 PCM16 오디오를 밀어넣고 partial/final 전사를 받는다.
Streamlit 은 `/api/v1/stt/*` 로 상태만 조회한다(같은 인스턴스이므로 localhost).

RunPod 프록시가 TLS 를 끝내주므로 여기서는 평문 HTTP 로 띄우면 된다.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, HTTPException, WebSocket
from fastapi.responses import FileResponse, JSONResponse

from stt.config import (
    API_BASE,
    ENGINE_CHOICES,
    RECORDINGS_DIR,
    SAMPLE_RATE,
    TRANSCRIPTS_DIR,
    WHISPER_SIZES,
    WS_BASE,
    StreamConfig,
)
from stt.metrics.latency import gpu_available, gpu_memory_mb

from .backend_ws import backend_stt_endpoint
from .runtime import READY, RUNTIME
from .websocket import SESSIONS, stt_endpoint

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
)
log = logging.getLogger("stt")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """시작: 활성 설정 고정 → 모델 preload(백그라운드). 종료: 로그만 남긴다."""
    RUNTIME.configure()
    RUNTIME.start_preload()
    yield
    log.info("[STT] Shutting down (active sessions: %d)", len(SESSIONS))


app = FastAPI(title="ER STT Backend", version="2.0", lifespan=lifespan)
api = APIRouter(prefix=API_BASE)     # /api/v1/stt/*
ws_api = APIRouter(prefix=WS_BASE)   # /ws/v1/stt/*


@ws_api.websocket("/stream")
async def backend_websocket_route(ws: WebSocket) -> None:
    await backend_stt_endpoint(ws)


@ws_api.websocket("/browser")
async def websocket_route(ws: WebSocket) -> None:
    await stt_endpoint(ws)


@api.get("/health")
async def health() -> dict:
    """liveness: 프로세스가 요청을 받을 수 있는지만 본다(모델 상태와 무관, 항상 200)."""
    return {
        "status": "OK",
        # 아래는 기존 클라이언트 호환용 필드
        "sample_rate": SAMPLE_RATE,
        "active_sessions": list(SESSIONS),
        "gpu_memory_mb": gpu_memory_mb(),
    }


@api.get("/health/ready")
async def health_ready() -> JSONResponse:
    """readiness: 활성 엔진이 로드·warm-up 되어 추론을 받을 수 있으면 200(READY), 아니면 503."""
    state = RUNTIME.snapshot()
    memory = gpu_memory_mb()
    body = {
        "status": state["status"],                 # READY | LOADING | NOT_READY
        "engine": state["engine"],
        "device": state["device"],
        "model_loaded": state["model_loaded"],
        "gpu_available": gpu_available(),
        "gpu_memory_mb": round(memory, 1) if memory is not None else None,
        "sample_rate": SAMPLE_RATE,
        "active_sessions": len(SESSIONS),
        "load_sec": state["load_sec"],
        "warmup_sec": state["warmup_sec"],
        "output_time": round(time.time(), 3),
    }
    if state["error"]:
        body["error"] = state["error"]
    return JSONResponse(body, status_code=200 if state["status"] == READY else 503)


@api.get("/engines")
async def engines() -> dict:
    """설치/모델 준비 상태까지 확인해서 선택 가능한 엔진을 알려준다."""
    available = {}
    for name in ENGINE_CHOICES:
        try:
            if name == "whisper":
                import faster_whisper  # noqa: F401

                available[name] = {"ready": True, "detail": "faster-whisper"}
            elif name == "funasr_mlt_nano":
                import funasr  # noqa: F401

                from stt.config import MODELS_DIR

                local = (MODELS_DIR / name / "model.pt").is_file()
                available[name] = {
                    "ready": True,
                    "detail": "funasr (로컬 models/funasr_mlt_nano)" if local
                    else "funasr (모델은 첫 로드 시 Hugging Face에서 다운로드)",
                }
            else:
                import sherpa_onnx  # noqa: F401

                from stt.config import MODELS_DIR

                d = MODELS_DIR / name
                ready = d.is_dir() and any(d.glob("*.onnx"))
                available[name] = {
                    "ready": ready,
                    "detail": str(d) if ready else f"{d} 에 모델 파일이 없습니다",
                }
        except ImportError as e:
            available[name] = {"ready": False, "detail": str(e)}
    # 실제로 서버가 쓰고 있는 엔진(= preload 한 엔진, /stream 세션 엔진)
    state = RUNTIME.snapshot()
    active = state["engine"]
    for name, meta in available.items():
        meta["loaded"] = name == active and state["model_loaded"]
    return {
        "active_engine": active,
        "active_status": state["status"],
        "active_config": RUNTIME.stream_config().to_dict(),
        "engines": available,
        "whisper_sizes": list(WHISPER_SIZES),
        "defaults": StreamConfig().to_dict(),   # 코드 기본값(설정 미지정 시)
    }


@api.get("/sessions/{session_id}/transcript")
async def transcript(session_id: str) -> JSONResponse:
    path = TRANSCRIPTS_DIR / f"{session_id}.json"
    if not path.exists():
        raise HTTPException(404, "전사 결과가 없습니다")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@api.get("/sessions/{session_id}/audio")
async def audio(session_id: str) -> FileResponse:
    path = RECORDINGS_DIR / f"{session_id}.wav"
    if not path.exists():
        raise HTTPException(404, "녹음 파일이 없습니다")
    return FileResponse(path, media_type="audio/wav", filename=path.name)


@api.get("/sessions")
async def sessions() -> dict:
    saved = sorted(
        (p.stem for p in TRANSCRIPTS_DIR.glob("*.json")), reverse=True
    )
    return {"active": list(SESSIONS), "saved": saved[:50]}


app.include_router(api)
app.include_router(ws_api)


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run("server.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
