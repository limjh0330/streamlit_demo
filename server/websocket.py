"""WebSocket 엔드포인트 `/ws/v1/stt/browser` — 브라우저 <-> RunPod STT 백엔드.

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

External Backend 용 `/ws/v1/stt/stream` 은 `backend_ws.py` 에 있다. 세션 생성/정리와
이벤트 브리지는 아래 공용 헬퍼(`event_bridge`, `open_session`, `close_session`)를
두 엔드포인트가 함께 쓴다.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import threading
from pathlib import Path
from typing import Callable

from fastapi import WebSocket, WebSocketDisconnect

from stt.config import SAMPLE_RATE, TRANSCRIPTS_DIR, StreamConfig
from stt.session import Event, StreamingSession

from .runtime import RUNTIME, _config_from  # noqa: F401  (_config_from: 기존 import 경로 유지)

log = logging.getLogger("stt.ws")

#: 살아있는 세션 (session_id -> StreamingSession)
SESSIONS: dict[str, StreamingSession] = {}


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
    """세션 생성 + 모델 로드 + warmup. 실패하면 워커까지 정리하고 다시 던진다.

    서버 시작 시 preload 가 진행 중이면 끝날 때까지 기다린다(같은 모델을 두 번
    올리지 않게). preload·warm-up 된 공유 모델을 쓰는 세션은 warm-up 을 건너뛴다.
    """
    await asyncio.to_thread(RUNTIME.wait_settled)
    session = StreamingSession(config, on_event=on_event)
    try:
        # 모델 로드는 수 초 걸릴 수 있으므로 이벤트 루프를 막지 않는다
        await asyncio.to_thread(session.start)
        if not RUNTIME.is_warm(config, session.engine):
            await asyncio.to_thread(session.engine.warmup)
    except Exception:
        await asyncio.to_thread(session.stop)
        raise
    SESSIONS[session.session_id] = session
    return session


async def close_session(session: StreamingSession) -> tuple[dict, Path | None]:
    """잔여 오디오 flush → recorder 정리 → 전사 저장. (summary, 전사 경로).

    정리는 전용 스레드에서 순서대로 끝내고, SESSIONS 에서는 맨 마지막에 뺀다.
    `asyncio.to_thread()` 를 쓰면 핸들러 코루틴이 취소될 때(연결 끊김 중 서버 종료,
    테스트 클라이언트 등) 아직 시작 전인 작업이 스레드풀 큐에서 함께 취소돼
    정리가 아예 안 되거나 recorder 가 열린 채 남을 수 있다. 그래서 작업을 먼저
    RUNNING 상태로 만든 뒤 스레드를 띄운다 — 기다리던 쪽이 취소돼도 정리는 끝까지 돈다.
    """
    directory = TRANSCRIPTS_DIR
    future: concurrent.futures.Future = concurrent.futures.Future()
    future.set_running_or_notify_cancel()            # 이후 cancel() 은 효과 없음

    def _close() -> None:
        try:
            summary = session.stop()
            try:
                path = session.save_transcript(directory)
            except Exception:
                log.exception("transcript save failed: %s", session.session_id)
                path = None
            future.set_result((summary, path))
        except BaseException as e:                   # noqa: BLE001 — 호출자에게 그대로 전달
            future.set_exception(e)
        finally:
            SESSIONS.pop(session.session_id, None)

    threading.Thread(target=_close, name=f"stt-close-{session.session_id}").start()
    return await asyncio.wrap_future(future)


# ---------------------------------------------------------------- /ws/v1/stt/browser
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
                log.info("[STT] Session opened (browser): stt=%s engine=%s",
                         session.session_id, config.engine)

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
            log.info("[STT] Session closed (browser): stt=%s", session.session_id)
        if sender is not None:
            sender.cancel()
        try:
            await ws.close()
        except Exception:
            pass
