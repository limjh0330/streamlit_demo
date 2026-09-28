"""WebSocket 엔드포인트 — 브라우저 <-> RunPod STT 백엔드.

프로토콜
--------
client -> server
    {"type":"start", "engine":"whisper", "model_size":"small", "language":"ko",
     "sample_rate":16000, ...}      : 세션 시작 (JSON 텍스트)
    <binary>                        : PCM16 little-endian mono 오디오 청크
    {"type":"stop"}                 : 세션 종료 요청

server -> client (JSON 텍스트)
    {"type":"ready",   "session_id":..., "engine":..., "config":{...}}
    {"type":"partial", "stable":"...", "partial":"...", "committed":"..."}
    {"type":"final",   "text":"...", "start":.., "end":.., "index":..}
    {"type":"metrics", "rtf_mean":.., "first_partial_ms":.., ...}
    {"type":"error",   "message":"..."}
    {"type":"closed",  "summary":{...}, "transcript":"...", "wav":"..."}

External Backend 용 `/ws/stt` 는 `backend_ws.py` 에 있다. 세션 생성/정리와
이벤트 브리지는 아래 공용 헬퍼(`event_bridge`, `open_session`, `close_session`)를
두 엔드포인트가 함께 쓴다.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Callable

from fastapi import WebSocket, WebSocketDisconnect

from stt.config import SAMPLE_RATE, TRANSCRIPTS_DIR, StreamConfig
from stt.session import Event, StreamingSession

log = logging.getLogger("stt.ws")

#: 살아있는 세션 (session_id -> StreamingSession)
SESSIONS: dict[str, StreamingSession] = {}


def _config_from(payload: dict) -> StreamConfig:
    """클라이언트가 보낸 start 페이로드에서 StreamConfig 를 만든다."""
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


# ---------------------------------------------------------------- 공용 헬퍼
def event_bridge(
    loop: asyncio.AbstractEventLoop,
) -> tuple[asyncio.Queue, Callable[[Event], None]]:
    """(outbox, on_event). 워커 스레드의 이벤트를 이벤트 루프 큐로 넘긴다."""
    outbox: asyncio.Queue = asyncio.Queue()

    def on_event(event: Event) -> None:
        loop.call_soon_threadsafe(outbox.put_nowait, event)

    return outbox, on_event


async def open_session(
    config: StreamConfig, on_event: Callable[[Event], None]
) -> StreamingSession:
    """세션 생성 + 모델 로드 + warmup. 실패하면 워커까지 정리하고 다시 던진다."""
    session = StreamingSession(config, on_event=on_event)
    try:
        # 모델 로드는 수 초 걸릴 수 있으므로 이벤트 루프를 막지 않는다
        await asyncio.to_thread(session.start)
        await asyncio.to_thread(session.engine.warmup)
    except Exception:
        await asyncio.to_thread(session.stop)
        raise
    SESSIONS[session.session_id] = session
    return session


async def close_session(session: StreamingSession) -> tuple[dict, Path | None]:
    """잔여 오디오 flush → recorder 정리 → 전사 저장. (summary, 전사 경로)."""
    try:
        summary = await asyncio.to_thread(session.stop)
        try:
            path = await asyncio.to_thread(session.save_transcript, TRANSCRIPTS_DIR)
        except Exception:
            log.exception("transcript save failed: %s", session.session_id)
            path = None
    finally:
        SESSIONS.pop(session.session_id, None)
    return summary, path


# ---------------------------------------------------------------- /ws
async def stt_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    loop = asyncio.get_running_loop()
    outbox, on_event = event_bridge(loop)
    session: StreamingSession | None = None
    sender: asyncio.Task | None = None
    session_sample_rate = SAMPLE_RATE

    async def pump() -> None:
        while True:
            event = await outbox.get()
            try:
                await ws.send_text(event.to_json())
            except Exception:
                return

    try:
        while True:
            message = await ws.receive()

            if message["type"] == "websocket.disconnect":
                break

            # ---------------------------------------------------- 바이너리 오디오
            if (chunk := message.get("bytes")) is not None:
                if session is None:
                    continue
                session.feed(chunk, sample_rate=session_sample_rate)
                continue

            # ---------------------------------------------------- 제어 메시지
            text = message.get("text")
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                await ws.send_text(json.dumps({"type": "error", "message": "invalid JSON"}))
                continue

            kind = payload.get("type")

            if kind == "start":
                if session is not None:
                    continue
                session_sample_rate = int(payload.get("sample_rate") or SAMPLE_RATE)
                config = _config_from(payload)
                if sender is None:
                    sender = asyncio.create_task(pump())
                try:
                    session = await open_session(config, on_event)
                except Exception as e:
                    await ws.send_text(json.dumps({"type": "error", "message": str(e)},
                                                  ensure_ascii=False))
                    session = None
                    continue
                log.info("session started: %s (%s)", session.session_id, config.engine)

            elif kind == "stop":
                break

            elif kind == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))

    except WebSocketDisconnect:
        pass
    except Exception as e:                       # pragma: no cover
        log.exception("websocket error")
        try:
            await ws.send_text(json.dumps({"type": "error", "message": str(e)},
                                          ensure_ascii=False))
        except Exception:
            pass
    finally:
        if session is not None:
            summary, path = await close_session(session)
            try:
                await ws.send_text(
                    json.dumps(
                        {
                            "type": "closed",
                            "summary": summary["metrics"],
                            "text": summary["stable"],
                            "utterances": summary["utterances"],
                            "wav": summary["wav"],
                            "transcript": str(path),
                            "session_id": session.session_id,
                        },
                        ensure_ascii=False,
                    )
                )
            except Exception:
                pass
            log.info("session closed: %s", session.session_id)
        if sender is not None:
            sender.cancel()
        try:
            await ws.close()
        except Exception:
            pass
