"""STT 서버 진입점 (FastAPI).

    python -m server.main --host 0.0.0.0 --port 8000
    # 태블릿에서 http://<서버IP>:8000 접속

주의: 브라우저 getUserMedia() 는 보안 컨텍스트에서만 동작한다.
localhost 는 예외이지만, 태블릿에서 IP 로 접속하려면 HTTPS 가 필요하다.
  python -m server.main --certfile cert.pem --keyfile key.pem
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

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

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(title="ER STT Server", version="1.0")


@app.websocket("/ws")
async def websocket_route(ws: WebSocket) -> None:
    await stt_endpoint(ws)


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


# 웹 클라이언트(index.html, recorder.js, pcm-worklet.js) 를 루트에 서빙
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--certfile", help="HTTPS 인증서 (태블릿 마이크 접근에 필요)")
    parser.add_argument("--keyfile", help="HTTPS 키")
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run(
        "server.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        ssl_certfile=args.certfile,
        ssl_keyfile=args.keyfile,
    )


if __name__ == "__main__":
    main()
