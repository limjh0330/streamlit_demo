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
"""
from __future__ import annotations

import asyncio
import json
import logging

from fastapi import WebSocket, WebSocketDisconnect

from stt.config import SAMPLE_RATE, StreamConfig
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


async def stt_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    loop = asyncio.get_running_loop()
    outbox: asyncio.Queue[Event] = asyncio.Queue()
    session: StreamingSession | None = None
    sender: asyncio.Task | None = None
    session_sample_rate = SAMPLE_RATE

    def on_event(event: Event) -> None:
        """워커 스레드에서 호출되므로 이벤트 루프로 안전하게 넘긴다."""
        loop.call_soon_threadsafe(outbox.put_nowait, event)

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
                session = StreamingSession(config, on_event=on_event)
                sender = asyncio.create_task(pump())
                try:
                    # 모델 로드는 수 초 걸릴 수 있으므로 이벤트 루프를 막지 않는다
                    await asyncio.to_thread(session.start)
                    await asyncio.to_thread(session.engine.warmup)
                except Exception as e:
                    await ws.send_text(json.dumps({"type": "error", "message": str(e)},
                                                  ensure_ascii=False))
                    session = None
                    continue
                SESSIONS[session.session_id] = session
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
            summary = await asyncio.to_thread(session.stop)
            path = await asyncio.to_thread(session.save_transcript)
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
            SESSIONS.pop(session.session_id, None)
            log.info("session closed: %s", session.session_id)
        if sender is not None:
            sender.cancel()
        try:
            await ws.close()
        except Exception:
            pass
