"""STT 백엔드 진입점 (FastAPI) — RunPod 에서 Streamlit 과 나란히 돈다.

    ./start_server.sh                                   # RunPod 운영
    python -m server.main --host 0.0.0.0 --port 8000    # 직접 실행

시작하면 활성 엔진(기본 funasr_mlt_nano)을 백그라운드로 preload · warm-up 한다.
    GET /api/v1/stt/health         liveness  — 프로세스가 살아 있으면 200
    GET /api/v1/stt/health/ready   readiness — 모델이 추론 준비를 마쳤으면 200, 아니면 503

REST API 는 브라우저에서 바로 시험할 수 있다(STT_DOCS=0 이면 끔).
    /api/v1/stt/docs          Swagger UI  (/docs 는 여기로 리다이렉트)
    /api/v1/stt/redoc         ReDoc
    /api/v1/stt/openapi.json  OpenAPI 명세

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

from fastapi import APIRouter, FastAPI, HTTPException, Path, WebSocket
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

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
from .runtime import READY, RUNTIME, _flag
from .schemas import (
    AUDIO_RESPONSES,
    READY_RESPONSES,
    TRANSCRIPT_RESPONSES,
    EnginesResponse,
    HealthResponse,
    SessionsResponse,
)
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


DOCS_ENABLED = _flag("STT_DOCS", True)     # 운영에서 문서를 숨기려면 STT_DOCS=0

DESCRIPTION = f"""
응급실 예진 대화용 실시간 한국어 STT 서버입니다. 이 페이지에서 REST API 를 바로 실행해 볼 수 있습니다
(각 API 의 **Try it out → Execute**).

**권장 확인 순서**: `GET /health/ready` 가 `READY`(200) → `GET /engines` 로 `active_engine` 확인 →
WebSocket 으로 오디오 전송.

### 실시간 전사는 WebSocket 입니다 (이 페이지에서는 실행되지 않음)

OpenAPI/Swagger 는 WebSocket 을 표현하지 못해 아래에 규격만 적습니다.
시험은 `python -m scripts.stt_client <wav> --url ws://<host>:<port>{WS_BASE}/stream` 로 하세요.

| 경로 | 쓰는 쪽 |
|---|---|
| `{WS_BASE}/stream` | External Backend |
| `{WS_BASE}/browser` | Streamlit 실시간 전사 페이지 |

`{WS_BASE}/stream` 규격:

- Backend → STT: binary **PCM16 LE / 16 kHz / mono**, 100 ms = 3,200 bytes · 종료는 text `close` 또는 `{{"type":"close"}}`
- STT → Backend:
  - `{{"type":"transcript","text":"...","is_final":false}}` — partial (화면 표시용)
  - `{{"type":"transcript","text":"...","is_final":true,"start_time":3.2,"end_time":5.8}}` — final (발화 확정)
  - `{{"type":"error","message":"..."}}`
  - `{{"type":"done"}}` — `close` 후 남은 오디오 flush 가 끝나면 마지막으로 전송
"""

TAGS = [
    {"name": "Health", "description": "liveness / readiness"},
    {"name": "Engines", "description": "활성 엔진과 엔진별 준비 상태"},
    {"name": "Sessions", "description": "저장된 세션의 전사·녹음 조회 (운영·디버깅용)"},
]

app = FastAPI(
    title="ER STT API",
    version="2.0",
    description=DESCRIPTION,
    openapi_tags=TAGS,
    lifespan=lifespan,
    # 문서도 API base path 아래에 둔다
    docs_url=f"{API_BASE}/docs" if DOCS_ENABLED else None,
    redoc_url=f"{API_BASE}/redoc" if DOCS_ENABLED else None,
    openapi_url=f"{API_BASE}/openapi.json" if DOCS_ENABLED else None,
    swagger_ui_parameters={
        "tryItOutEnabled": True,          # Try it out 을 누르지 않아도 바로 Execute
        "displayRequestDuration": True,   # 응답 시간 표시
        "defaultModelsExpandDepth": 0,
    },
)
api = APIRouter(prefix=API_BASE)     # /api/v1/stt/*
ws_api = APIRouter(prefix=WS_BASE)   # /ws/v1/stt/*


@ws_api.websocket("/stream")
async def backend_websocket_route(ws: WebSocket) -> None:
    await backend_stt_endpoint(ws)


@ws_api.websocket("/browser")
async def websocket_route(ws: WebSocket) -> None:
    await stt_endpoint(ws)


if DOCS_ENABLED:
    @app.get("/docs", include_in_schema=False)
    async def docs_redirect() -> RedirectResponse:
        """버전 없는 /docs 로 들어와도 Swagger UI 로 보낸다."""
        return RedirectResponse(f"{API_BASE}/docs")


@api.get("/health", tags=["Health"], response_model=HealthResponse,
         summary="Liveness — 프로세스 생존 확인")
async def health() -> dict:
    """liveness: 프로세스가 요청을 받을 수 있는지만 본다(모델 상태와 무관, 항상 200)."""
    return {
        "status": "OK",
        # 아래는 기존 클라이언트 호환용 필드
        "sample_rate": SAMPLE_RATE,
        "active_sessions": list(SESSIONS),
        "gpu_memory_mb": gpu_memory_mb(),
    }


@api.get("/health/ready", tags=["Health"], responses=READY_RESPONSES,
         summary="Readiness — 모델 추론 준비 상태 (READY=200, 그 외 503)")
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


@api.get("/engines", tags=["Engines"], response_model=EnginesResponse,
         summary="활성 엔진(active_engine)과 엔진별 준비·로드 상태")
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


@api.get("/sessions/{session_id}/transcript", tags=["Sessions"], responses=TRANSCRIPT_RESPONSES,
         summary="저장된 전사 결과 (JSON)")
async def transcript(session_id: str = Path(description="STT 내부 세션 ID (`GET /sessions` 의 값)",
                           examples=["20260928-215345-d8a74a"])) -> JSONResponse:
    path = TRANSCRIPTS_DIR / f"{session_id}.json"
    if not path.exists():
        raise HTTPException(404, "전사 결과가 없습니다")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@api.get("/sessions/{session_id}/audio", tags=["Sessions"], responses=AUDIO_RESPONSES,
         response_class=FileResponse, summary="저장된 녹음 (WAV 다운로드)")
async def audio(session_id: str = Path(description="STT 내부 세션 ID (`GET /sessions` 의 값)",
                           examples=["20260928-215345-d8a74a"])) -> FileResponse:
    path = RECORDINGS_DIR / f"{session_id}.wav"
    if not path.exists():
        raise HTTPException(404, "녹음 파일이 없습니다")
    return FileResponse(path, media_type="audio/wav", filename=path.name)


@api.get("/sessions", tags=["Sessions"], response_model=SessionsResponse,
         summary="진행 중 / 저장된 세션 ID 목록")
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
