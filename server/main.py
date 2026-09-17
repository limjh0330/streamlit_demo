"""STT 백엔드 진입점 (FastAPI) — RunPod 에서 Streamlit 과 나란히 돈다.

    python -m server.main --host 0.0.0.0 --port 8000

브라우저는 `/ws` 로 PCM16 오디오를 밀어넣고 partial/final 전사를 받는다.
Streamlit 은 `/api/*` 로 상태만 조회한다(같은 인스턴스이므로 localhost).

RunPod 프록시가 TLS 를 끝내주므로 여기서는 평문 HTTP 로 띄우면 된다.
"""
from __future__ import annotations

import argparse
import json
import logging

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import FileResponse, JSONResponse

from stt.config import (
    ENGINE_CHOICES,
    RECORDINGS_DIR,
    SAMPLE_RATE,
    TRANSCRIPTS_DIR,
    WHISPER_SIZES,
    StreamConfig,
)
from stt.metrics.latency import gpu_memory_mb

from .websocket import SESSIONS, stt_endpoint

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
)

app = FastAPI(title="ER STT Backend", version="2.0")


@app.websocket("/ws")
async def websocket_route(ws: WebSocket) -> None:
    await stt_endpoint(ws)


@app.get("/")
@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "sample_rate": SAMPLE_RATE,
        "active_sessions": list(SESSIONS),
        "gpu_memory_mb": gpu_memory_mb(),
    }


@app.get("/api/engines")
async def engines() -> dict:
    """설치/모델 준비 상태까지 확인해서 선택 가능한 엔진을 알려준다."""
    available = {}
    for name in ENGINE_CHOICES:
        try:
            if name == "whisper":
                import faster_whisper  # noqa: F401

                available[name] = {"ready": True, "detail": "faster-whisper"}
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
    return {
        "engines": available,
        "whisper_sizes": list(WHISPER_SIZES),
        "defaults": StreamConfig().to_dict(),
    }


@app.get("/api/sessions/{session_id}/transcript")
async def transcript(session_id: str) -> JSONResponse:
    path = TRANSCRIPTS_DIR / f"{session_id}.json"
    if not path.exists():
        raise HTTPException(404, "전사 결과가 없습니다")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@app.get("/api/sessions/{session_id}/audio")
async def audio(session_id: str) -> FileResponse:
    path = RECORDINGS_DIR / f"{session_id}.wav"
    if not path.exists():
        raise HTTPException(404, "녹음 파일이 없습니다")
    return FileResponse(path, media_type="audio/wav", filename=path.name)


@app.get("/api/sessions")
async def sessions() -> dict:
    saved = sorted(
        (p.stem for p in TRANSCRIPTS_DIR.glob("*.json")), reverse=True
    )
    return {"active": list(SESSIONS), "saved": saved[:50]}


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
