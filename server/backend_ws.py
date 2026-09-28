"""External Backend 용 WebSocket 어댑터 — `/ws/v1/stt/stream`.

STT 코어(StreamingSession → VAD → ASR → TranscriptMerger)는 그대로 두고,
그 앞뒤의 입출력만 Backend 규격으로 바꾼다. 브라우저용 `/ws/v1/stt/browser` 와 세션 생성/정리
헬퍼를 공유한다(`websocket.py`).

프로토콜
--------
Backend -> STT
    <binary>   PCM16 little-endian / 16 kHz / mono. 보통 100 ms = 3200 bytes
    "close"    오디오 전송 종료 (text frame). {"type":"close"} 도 받는다

STT -> Backend (JSON 텍스트)
    {"type":"transcript", "text":"...", "is_final":false}          : 현재 발화의 partial
    {"type":"transcript", "text":"...", "is_final":true,
     "start_time":12.4, "end_time":15.6}                           : 발화 확정
    {"type":"error", "message":"..."}                              : 안전한 메시지만
    {"type":"done"}                                                : flush 끝, 마지막 메시지

책임 범위는 Audio → Transcript 까지다. conversation/turn 번호, KTAS 판단,
LLM 호출, 화자 역할(환자/의료진) 추정은 Backend 가 하므로 여기서 만들지 않는다.

엔진/윈도우 같은 ASR 설정은 Backend 가 몰라도 되도록 서버 환경변수로 정한다.
    STT_ENGINE       whisper | zipformer | sensevoice | funasr_mlt_nano
    STT_MODEL_SIZE   whisper 모델 크기
    STT_CONFIG       StreamConfig 필드 JSON (예: {"silence_sec":0.6,"save_wav":false})
    STT_TIMESTAMPS   final 에 start_time/end_time 포함 여부 (기본 1)
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
import uuid

from fastapi import WebSocket, WebSocketDisconnect

from stt.config import SAMPLE_RATE, StreamConfig
from stt.session import Event

from .websocket import _config_from, close_session, event_bridge, open_session

log = logging.getLogger("stt.ws.backend")

#: 100 ms PCM16 mono 16 kHz = 1600 samples * 2 bytes
FRAME_BYTES = int(SAMPLE_RATE * 0.1) * 2
#: 이보다 큰 프레임은 명백한 프로토콜 위반으로 보고 버린다(1 s).
MAX_FRAME_BYTES = SAMPLE_RATE * 2

# Backend 에 돌려주는 오류 문구. 예외 내용/경로/traceback 은 로그에만 남긴다.
ERR_INIT = "STT engine initialization failed"
ERR_INFERENCE = "STT inference failed"
ERR_FRAME = "Invalid audio frame"
ERR_CONTROL = "Unknown control message"
ERR_INTERNAL = "Internal STT server error"

_DONE = object()      # outbox sentinel: 남은 전사를 다 보낸 뒤 done 전송
_STOP = object()      # outbox sentinel: done 없이 종료(비정상 연결 끊김)


def backend_config() -> StreamConfig:
    """`/ws/v1/stt/stream` 세션 설정. Backend 요청이 아니라 서버 환경변수에서 정한다."""
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
            log.warning("STT_CONFIG 는 JSON 객체여야 합니다 — 무시합니다")
    for env, key in (("STT_ENGINE", "engine"), ("STT_MODEL_SIZE", "model_size")):
        if os.getenv(env):
            overrides[key] = os.environ[env]
    return _config_from(overrides)


def _include_timestamps() -> bool:
    return os.getenv("STT_TIMESTAMPS", "1").strip().lower() not in ("0", "false", "no")


def _triage_session_id(ws: WebSocket) -> str | None:
    """로그 상관관계용 Backend 예진 세션 UUID(선택). STT 세션 ID 와는 별개다."""
    raw = ws.query_params.get("triage_session_id")
    if not raw:
        return None
    try:
        return str(uuid.UUID(raw))
    except ValueError:
        log.warning("invalid triage_session_id query parameter — ignored")
        return None


def _is_close(text: str) -> bool:
    if text.strip().lower() == "close":
        return True
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return False
    return isinstance(payload, dict) and payload.get("type") == "close"


class PCMFrameAssembler:
    """WebSocket 프레임 경계와 PCM 샘플 경계를 분리한다.

    정상 프레임(3200 B)은 그대로 통과한다. 크기가 다른 프레임도 받아들이되,
    홀수 길이라 샘플이 쪼개지면 남는 1 바이트를 다음 프레임 앞에 붙여 PCM16
    정렬을 지킨다. 세션은 블록 크기에 의존하지 않으므로 더 모을 필요는 없다.
    """

    def __init__(self) -> None:
        self._carry = b""
        self.frames = 0
        self.irregular = 0

    def push(self, data: bytes) -> bytes:
        self.frames += 1
        if len(data) != FRAME_BYTES:
            self.irregular += 1
        if self._carry:
            data, self._carry = self._carry + data, b""
        if len(data) % 2:
            data, self._carry = data[:-1], data[-1:]
        return data

    @property
    def leftover(self) -> int:
        return len(self._carry)


class BackendEventWriter:
    """세션 Event 를 Backend 규격 메시지로 바꿔 보낸다(송신은 이 태스크 하나로)."""

    #: 같은 추론 오류가 블록마다 반복돼도 Backend 에는 이 간격으로만 알린다.
    ERROR_INTERVAL_SEC = 1.0

    def __init__(self, ws: WebSocket, timestamps: bool = True) -> None:
        self.ws = ws
        self.timestamps = timestamps
        self._last_partial = ""
        self._last_error = -math.inf

    def translate(self, event: Event) -> dict | None:
        data = event.data
        if event.type == "partial":
            text = data.get("utterance", "")
            if not text or text == self._last_partial:
                return None
            self._last_partial = text
            return {"type": "transcript", "text": text, "is_final": False}

        if event.type == "final":
            self._last_partial = ""
            text = data.get("text", "")
            if not text:
                return None
            msg = {"type": "transcript", "text": text, "is_final": True}
            if self.timestamps and "start" in data and "end" in data:
                msg["start_time"] = data["start"]
                msg["end_time"] = data["end"]
            return msg

        if event.type == "error":
            log.error("stt inference error: %s", data.get("message"))
            now = time.monotonic()
            if now - self._last_error < self.ERROR_INTERVAL_SEC:
                return None
            self._last_error = now
            return {"type": "error", "message": ERR_INFERENCE}

        return None                               # ready / metrics: Backend 규격 밖

    async def run(self, outbox: asyncio.Queue) -> None:
        while True:
            item = await outbox.get()
            if item is _STOP:
                return
            if item is _DONE:
                await self._send({"type": "done"})
                return
            msg = item if isinstance(item, dict) else self.translate(item)
            if msg is not None and not await self._send(msg):
                return                            # 상대가 이미 끊었다

    async def _send(self, msg: dict) -> bool:
        try:
            await self.ws.send_text(json.dumps(msg, ensure_ascii=False))
            return True
        except Exception:
            return False


async def backend_stt_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    triage_id = _triage_session_id(ws)
    loop = asyncio.get_running_loop()
    outbox, on_event = event_bridge(loop)

    try:
        session = await open_session(backend_config(), on_event)
    except Exception:
        log.exception("stt session init failed (triage=%s)", triage_id)
        try:
            await ws.send_text(json.dumps({"type": "error", "message": ERR_INIT}))
            await ws.close(code=1011)
        except Exception:
            pass
        return

    stt_id = session.session_id
    log.info("backend session started: stt=%s triage=%s engine=%s",
             stt_id, triage_id, session.engine.name)
    writer = BackendEventWriter(ws, timestamps=_include_timestamps())
    sender = asyncio.create_task(writer.run(outbox))
    frames = PCMFrameAssembler()
    graceful = False          # Backend 가 "close" 로 끝냈는지
    failed = False

    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break

            # ------------------------------------------------ 바이너리 오디오
            if (chunk := message.get("bytes")) is not None:
                if not chunk:
                    continue
                if len(chunk) > MAX_FRAME_BYTES:
                    log.warning("stt=%s oversized audio frame dropped: %d bytes",
                                stt_id, len(chunk))
                    outbox.put_nowait({"type": "error", "message": ERR_FRAME})
                    continue
                if len(chunk) != FRAME_BYTES and frames.irregular == 0:
                    log.warning("stt=%s audio frame of %d bytes (expected %d)",
                                stt_id, len(chunk), FRAME_BYTES)
                pcm = frames.push(chunk)
                if pcm:
                    session.feed(pcm, sample_rate=SAMPLE_RATE)
                continue

            # ------------------------------------------------ 제어 메시지
            text = message.get("text")
            if text is None:
                continue
            if _is_close(text):
                graceful = True
                break
            outbox.put_nowait({"type": "error", "message": ERR_CONTROL})

    except WebSocketDisconnect:
        pass
    except Exception:
        failed = True
        log.exception("backend websocket error: stt=%s", stt_id)
        outbox.put_nowait({"type": "error", "message": ERR_INTERNAL})

    finally:
        # close → 남은 오디오/디코더/병합기 flush(final 이 outbox 로 나간다)
        #       → recorder 정리 → 그 뒤에야 done.
        # stop() 안에서 워커가 emit 한 final 은 stop() 이 돌아오기 전에 outbox 에
        # 들어가 있으므로, 그 뒤에 넣는 sentinel 은 항상 마지막이 된다.
        await close_session(session)
        if frames.leftover:
            log.warning("stt=%s dropped %d trailing byte(s) of a split sample",
                        stt_id, frames.leftover)
        if graceful:
            outbox.put_nowait(_DONE)
            await sender
            try:
                await ws.close(code=1000)
            except Exception:
                pass
        else:
            outbox.put_nowait(_STOP)
            if failed:
                await sender                      # error 메시지까지는 보낸다
                try:
                    await ws.close(code=1011)
                except Exception:
                    pass
            else:
                sender.cancel()                   # 이미 끊긴 소켓에 보낼 필요 없음
        log.info("backend session closed: stt=%s triage=%s graceful=%s frames=%d irregular=%d",
                 stt_id, triage_id, graceful, frames.frames, frames.irregular)
