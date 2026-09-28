"""External Backend 용 `/ws/stt` 어댑터를 가짜 엔진으로 검증한다.

    pytest tests/test_backend_ws.py

실제 모델 대신 `stt.session.create_engine` 을 결정적 엔진으로 바꿔 끼운다.
세션/VAD/버퍼/병합기는 실제 코드가 그대로 돈다.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

import server.websocket as ws_module
import stt.session as session_module
from server.backend_ws import FRAME_BYTES, MAX_FRAME_BYTES
from server.main import app
from server.websocket import SESSIONS
from stt.asr.base import ASREngine, ASRResult
from stt.audio.recorder import WavRecorder
from stt.config import SAMPLE_RATE
from stt.session import StreamingSession

SCRIPT = "가슴이 답답해요 어제부터 계속".split()
SPEECH_FRAMES = 20          # 2.0 s 발화
SILENCE_FRAMES = 12         # 1.2 s 무음 -> endpoint


class ScriptedEngine(ASREngine):
    """SenseVoice 처럼 발화 전체를 재인식한다. 들은 길이만큼 SCRIPT 를 앞에서부터 준다."""

    name = "scripted"
    decodes_full_utterance = True

    def transcribe(self, audio, t0: float = 0.0, is_final: bool = False) -> ASRResult:
        dur = audio.size / SAMPLE_RATE
        n = min(len(SCRIPT), math.ceil(dur / 0.5))
        text = " ".join(SCRIPT[:n])
        return ASRResult(text=text, audio_sec=dur, infer_sec=0.001, is_final=is_final)


class FailingEngine(ScriptedEngine):
    def transcribe(self, audio, t0: float = 0.0, is_final: bool = False) -> ASRResult:
        raise RuntimeError("/secret/path/model.bin exploded")


def _pcm(amplitude: float, seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    audio = np.clip(amplitude * rng.standard_normal(FRAME_BYTES // 2), -1, 1)
    return (audio * 32767).astype("<i2").tobytes()


def _utterance_frames() -> list[bytes]:
    speech = [_pcm(0.15, i) for i in range(SPEECH_FRAMES)]
    silence = [_pcm(1e-5, 100 + i) for i in range(SILENCE_FRAMES)]
    return speech + silence


@pytest.fixture
def stt(monkeypatch, tmp_path):
    """가짜 엔진 + 임시 저장 경로 + feed 호출 기록."""
    monkeypatch.setenv(
        "STT_CONFIG", json.dumps({"save_wav": True, "medical_correction": False})
    )
    monkeypatch.setattr(session_module, "create_engine", lambda cfg: ScriptedEngine(cfg))
    monkeypatch.setattr(ws_module, "TRANSCRIPTS_DIR", tmp_path / "transcripts")
    monkeypatch.setattr(
        session_module, "WavRecorder", lambda sid: WavRecorder(sid, directory=tmp_path)
    )

    state = {"fed": [], "sessions": []}
    original_feed = StreamingSession.feed

    def spy_feed(self, audio, sample_rate=SAMPLE_RATE):
        if self not in state["sessions"]:
            state["sessions"].append(self)
        state["fed"].append(bytes(audio))
        return original_feed(self, audio, sample_rate=sample_rate)

    monkeypatch.setattr(StreamingSession, "feed", spy_feed)
    with TestClient(app) as client:
        state["client"] = client
        yield state


def _receive_until_done(ws, limit: int = 200) -> list[dict]:
    messages = []
    for _ in range(limit):
        msg = ws.receive_json()
        messages.append(msg)
        if msg["type"] == "done":
            return messages
    raise AssertionError(f"done 을 받지 못함: {messages}")


def _run_utterance(client, frames: list[bytes]) -> list[dict]:
    with client.websocket_connect("/ws/stt") as ws:
        for frame in frames:
            ws.send_bytes(frame)
        ws.send_text("close")
        return _receive_until_done(ws)


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


# ---------------------------------------------------------------- tests
def test_single_3200_byte_frame_reaches_session(stt) -> None:
    frame = _pcm(0.15, 0)
    assert len(frame) == FRAME_BYTES == 3200
    messages = _run_utterance(stt["client"], [frame])

    assert stt["fed"] == [frame]
    assert messages[-1] == {"type": "done"}


def test_consecutive_frames_are_fed_in_order(stt) -> None:
    frames = _utterance_frames()
    _run_utterance(stt["client"], frames)
    assert stt["fed"] == frames
    assert len(stt["sessions"]) == 1


def test_partial_transcript_format(stt) -> None:
    messages = _run_utterance(stt["client"], _utterance_frames())
    partials = [m for m in messages if m["type"] == "transcript" and not m["is_final"]]

    assert partials, messages
    for m in partials:
        assert set(m) == {"type", "text", "is_final"}
        assert isinstance(m["text"], str) and m["text"]
    # 같은 partial 을 연달아 반복하지 않는다
    texts = [m["text"] for m in partials]
    assert all(a != b for a, b in zip(texts, texts[1:])), texts
    # 발화 단위 텍스트여야 한다(시작이 SCRIPT 첫 어절)
    assert texts[0].startswith(SCRIPT[0])


def test_final_transcript_format(stt) -> None:
    messages = _run_utterance(stt["client"], _utterance_frames())
    finals = [m for m in messages if m["type"] == "transcript" and m["is_final"]]

    assert len(finals) == 1, messages
    final = finals[0]
    assert final["text"] == " ".join(SCRIPT)
    assert set(final) <= {"type", "text", "is_final", "start_time", "end_time"}
    assert final["end_time"] >= final["start_time"] >= 0
    # conversation/turn/화자 역할은 STT 가 만들지 않는다
    for key in ("turn_id", "conversation_id", "speaker", "role"):
        assert key not in final


def test_timestamps_can_be_disabled(stt, monkeypatch) -> None:
    monkeypatch.setenv("STT_TIMESTAMPS", "0")
    messages = _run_utterance(stt["client"], _utterance_frames())
    final = next(m for m in messages if m.get("is_final"))
    assert set(final) == {"type", "text", "is_final"}


def test_close_flushes_remaining_final_before_done(stt) -> None:
    """endpoint 전에 close 가 와도 남은 발화를 final 로 내보낸 뒤 done 을 보낸다."""
    speech_only = [_pcm(0.15, i) for i in range(SPEECH_FRAMES)]   # 무음 없음 -> endpoint 없음
    messages = _run_utterance(stt["client"], speech_only)

    types = [(m["type"], m.get("is_final")) for m in messages]
    assert types[-1] == ("done", None), types
    assert types.count(("done", None)) == 1
    final_idx = [i for i, t in enumerate(types) if t == ("transcript", True)]
    assert final_idx and final_idx[-1] < len(types) - 1
    assert messages[final_idx[-1]]["text"] == " ".join(SCRIPT)

    # done 시점에 세션/recorder 는 이미 정리돼 있다
    session = stt["sessions"][0]
    assert session._worker is None
    assert session.recorder is not None and session.recorder._wf is None
    assert session.session_id not in SESSIONS


def test_invalid_frames_do_not_crash_server(stt) -> None:
    frames = _utterance_frames()
    odd_a, odd_b = frames[0][:1601], frames[0][1601:]     # 샘플이 쪼개진 프레임
    with stt["client"].websocket_connect("/ws/stt") as ws:
        ws.send_bytes(b"")                                  # 빈 프레임: 무시
        ws.send_bytes(b"\x00" * (MAX_FRAME_BYTES + 2))      # 과대 프레임: error 후 버림
        assert ws.receive_json() == {"type": "error", "message": "Invalid audio frame"}
        ws.send_text("hello")                               # 모르는 제어 메시지
        assert ws.receive_json() == {"type": "error", "message": "Unknown control message"}

        ws.send_bytes(odd_a)
        ws.send_bytes(odd_b)
        for frame in frames[1:]:
            ws.send_bytes(frame)
        ws.send_text("close")
        messages = _receive_until_done(ws)

    # 쪼개진 샘플도 정렬을 지켜 원래 오디오 그대로 들어간다
    assert b"".join(stt["fed"]) == b"".join(frames)
    assert all(len(b) % 2 == 0 for b in stt["fed"])
    assert any(m.get("is_final") for m in messages), messages


def test_inference_error_is_sanitized(stt, monkeypatch) -> None:
    monkeypatch.setattr(session_module, "create_engine", lambda cfg: FailingEngine(cfg))
    messages = _run_utterance(stt["client"], _utterance_frames())

    errors = [m for m in messages if m["type"] == "error"]
    assert errors, messages
    for m in errors:
        assert m == {"type": "error", "message": "STT inference failed"}
    assert "secret" not in json.dumps(messages)
    assert messages[-1] == {"type": "done"}


def test_engine_init_failure_returns_safe_error(stt, monkeypatch) -> None:
    def broken(cfg):
        raise RuntimeError("/models/zipformer 에 모델 파일이 없습니다")

    monkeypatch.setattr(session_module, "create_engine", broken)
    with stt["client"].websocket_connect("/ws/stt") as ws:
        assert ws.receive_json() == {
            "type": "error", "message": "STT engine initialization failed"
        }
    assert not SESSIONS


def test_abrupt_disconnect_releases_resources(stt) -> None:
    with stt["client"].websocket_connect("/ws/stt") as ws:
        for frame in _utterance_frames()[:SPEECH_FRAMES]:
            ws.send_bytes(frame)
        assert _wait_until(lambda: len(stt["fed"]) == SPEECH_FRAMES)
        session = stt["sessions"][0]
        assert session.session_id in SESSIONS
        # close 없이 연결을 끊는다
    assert _wait_until(lambda: session.session_id not in SESSIONS)
    assert _wait_until(lambda: session._worker is None)
    assert session.recorder._wf is None                 # WAV 는 닫혀 있다
    assert session.recorder.path.exists()


def test_legacy_browser_ws_protocol_still_works(stt) -> None:
    """Streamlit/브라우저용 /ws 는 기존 start/stop/closed 규격 그대로다."""
    with stt["client"].websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "start", "engine": "sensevoice", "sample_rate": 16000}))
        assert ws.receive_json()["type"] == "ready"
        for frame in _utterance_frames():
            ws.send_bytes(frame)
        ws.send_text(json.dumps({"type": "stop"}))
        messages = []
        while True:
            msg = ws.receive_json()
            messages.append(msg)
            if msg["type"] == "closed":
                break

    kinds = {m["type"] for m in messages}
    assert "final" in kinds
    closed = messages[-1]
    assert closed["text"] == " ".join(SCRIPT)
    assert {"summary", "utterances", "wav", "transcript", "session_id"} <= set(closed)
