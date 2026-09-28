"""[Unit test] 서버 런타임 — liveness/readiness, 활성 엔진, preload, 모델 공유.

    pytest tests/test_runtime.py

실제 모델은 쓰지 않는다. `server.runtime.create_engine`(preload)과
`stt.session.create_engine`(세션)을 가짜 엔진으로 바꿔 끼운다.
실제 Fun-ASR/GPU 검증은 `python -m scripts.stt_client` (Real E2E) 로 한다.
"""
from __future__ import annotations

import json
import threading
import time
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

import server.runtime as runtime_module
import server.websocket as ws_module
import stt.asr.funasr_mlt_nano as funasr_module
import stt.session as session_module
from server.main import app
from server.runtime import RUNTIME
from stt.config import API_BASE, SAMPLE_RATE, WS_STREAM_PATH, StreamConfig
from tests.test_backend_ws import SCRIPT, ScriptedEngine, _receive_until_done, _utterance_frames

READY_URL = f"{API_BASE}/health/ready"


class SharedScriptedEngine(ScriptedEngine):
    """프로세스 공유 모델을 쓰는 엔진 흉내(Fun-ASR/Whisper 처럼). warm-up 횟수를 센다."""

    shared_model = True
    warmups = 0

    def warmup(self, strict: bool = False) -> float:
        type(self).warmups += 1
        return super().warmup(strict=strict)


class Recorder:
    """create_engine 대역: 받은 설정을 기록하고 엔진을 만든다."""

    def __init__(self, cls=SharedScriptedEngine, gate: threading.Event | None = None,
                 error: Exception | None = None) -> None:
        self.cls, self.gate, self.error = cls, gate, error
        self.configs: list[StreamConfig] = []

    def __call__(self, cfg: StreamConfig):
        self.configs.append(cfg)
        if self.gate is not None:
            assert self.gate.wait(10), "gate 가 열리지 않음"
        if self.error is not None:
            raise self.error
        return self.cls(cfg)


@pytest.fixture
def server(monkeypatch, tmp_path):
    """환경변수·가짜 엔진을 정한 뒤 `start()` 로 서버(TestClient)를 띄운다."""
    SharedScriptedEngine.warmups = 0
    monkeypatch.setenv("STT_CONFIG", json.dumps({"save_wav": False, "medical_correction": False}))
    monkeypatch.setenv("STT_ENGINE", "funasr_mlt_nano")
    monkeypatch.delenv("STT_PRELOAD", raising=False)
    monkeypatch.setattr(ws_module, "TRANSCRIPTS_DIR", tmp_path / "transcripts")
    state = {"preload": Recorder(), "session": Recorder()}

    def start():
        monkeypatch.setattr(runtime_module, "create_engine", state["preload"])
        monkeypatch.setattr(session_module, "create_engine", state["session"])
        return TestClient(app)

    state["start"] = start
    yield state


def _wait_ready(client, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(READY_URL).json()
        if body["status"] != "LOADING":
            return body
        time.sleep(0.02)
    raise AssertionError("readiness 가 LOADING 에서 벗어나지 않음")


def _stream(client, frames) -> list[dict]:
    with client.websocket_connect(WS_STREAM_PATH) as ws:
        for frame in frames:
            ws.send_bytes(frame)
        ws.send_text(json.dumps({"type": "close"}))
        return _receive_until_done(ws)


# ---------------------------------------------------------------- health
def test_liveness_is_ok_regardless_of_model_state(server) -> None:
    gate = threading.Event()
    server["preload"] = Recorder(gate=gate)                  # 모델 로드가 끝나지 않은 상태
    with server["start"]() as client:
        r = client.get(f"{API_BASE}/health")
        assert r.status_code == 200
        assert r.json()["status"] == "OK"
        assert client.get(READY_URL).json()["status"] == "LOADING"
        gate.set()


def test_readiness_loading_then_ready(server) -> None:
    gate = threading.Event()
    server["preload"] = Recorder(gate=gate)
    with server["start"]() as client:
        loading = client.get(READY_URL)
        assert loading.status_code == 503
        assert loading.json()["status"] == "LOADING"
        assert loading.json()["model_loaded"] is False
        assert loading.json()["engine"] == "funasr_mlt_nano"

        gate.set()
        body = _wait_ready(client)
        r = client.get(READY_URL)
        assert r.status_code == 200
        assert body["status"] == "READY"
        assert body["engine"] == "funasr_mlt_nano"
        assert body["model_loaded"] is True
        assert body["sample_rate"] == SAMPLE_RATE
        assert body["active_sessions"] == 0
        assert isinstance(body["gpu_available"], bool)
        assert isinstance(body["output_time"], float)
        assert "error" not in body


def test_readiness_not_ready_when_model_load_fails(server) -> None:
    server["preload"] = Recorder(error=RuntimeError("/secret/models/funasr/model.pt 가 없습니다"))
    with server["start"]() as client:
        body = _wait_ready(client)
        assert client.get(READY_URL).status_code == 503
        assert body["status"] == "NOT_READY"
        assert body["model_loaded"] is False
        assert body["error"].startswith("RuntimeError")
        assert "secret" not in body["error"]                 # 경로·예외 내용은 로그에만


def test_readiness_not_ready_when_warmup_inference_fails(server) -> None:
    class BrokenInference(SharedScriptedEngine):
        def transcribe(self, audio, t0=0.0, is_final=False):
            raise RuntimeError("CUDA error")

    server["preload"] = Recorder(cls=BrokenInference)
    with server["start"]() as client:
        body = _wait_ready(client)
        assert body["status"] == "NOT_READY"                 # 로드만 되고 추론이 안 되면 READY 아님


def test_preload_disabled_reports_not_ready_but_sessions_work(server, monkeypatch) -> None:
    monkeypatch.setenv("STT_PRELOAD", "0")
    with server["start"]() as client:
        body = client.get(READY_URL).json()
        assert body["status"] == "NOT_READY" and "preload disabled" in body["error"]
        assert server["preload"].configs == []               # preload 하지 않음
        messages = _stream(client, _utterance_frames())
        assert messages[-1] == {"type": "done"}              # 첫 연결에서 lazy 로드


# ---------------------------------------------------------------- 활성 엔진
@pytest.mark.parametrize("engine", ["funasr_mlt_nano", "sensevoice"])
def test_engines_reports_active_engine_from_env(server, monkeypatch, engine) -> None:
    monkeypatch.setenv("STT_ENGINE", engine)
    with server["start"]() as client:
        _wait_ready(client)
        body = client.get(f"{API_BASE}/engines").json()
        assert body["active_engine"] == engine
        assert body["active_status"] == "READY"
        assert body["active_config"]["engine"] == engine
        assert body["engines"][engine]["loaded"] is True
        assert all(not m["loaded"] for n, m in body["engines"].items() if n != engine)
        assert "defaults" in body                            # 기존 필드 유지


def test_preload_and_sessions_share_one_active_config(server, monkeypatch) -> None:
    """startup 은 funasr, 세션은 whisper 같은 불일치가 없어야 한다."""
    with server["start"]() as client:
        _wait_ready(client)
        monkeypatch.setenv("STT_ENGINE", "whisper")           # 시작 후 바뀐 env 는 무시
        monkeypatch.setenv("STT_TIMESTAMPS", "0")
        messages = _stream(client, _utterance_frames())

    assert [c.engine for c in server["preload"].configs] == ["funasr_mlt_nano"]
    assert [c.engine for c in server["session"].configs] == ["funasr_mlt_nano"]
    final = next(m for m in messages if m.get("is_final"))
    assert "start_time" in final                             # 시작 시 STT_TIMESTAMPS=1 그대로


def test_timestamps_env_read_at_startup(server, monkeypatch) -> None:
    monkeypatch.setenv("STT_TIMESTAMPS", "0")
    with server["start"]() as client:
        _wait_ready(client)
        final = next(m for m in _stream(client, _utterance_frames()) if m.get("is_final"))
    assert set(final) == {"type", "text", "is_final"}


# ---------------------------------------------------------------- preload · 공유 모델
def test_preloaded_shared_model_skips_per_session_warmup(server) -> None:
    with server["start"]() as client:
        _wait_ready(client)
        assert SharedScriptedEngine.warmups == 1             # preload 에서 한 번
        for _ in range(3):
            assert _stream(client, _utterance_frames())[-1] == {"type": "done"}
    assert SharedScriptedEngine.warmups == 1                 # 세션마다 다시 하지 않음
    assert len(server["session"].configs) == 3               # 세션 상태(엔진 인스턴스)는 각자


def test_non_shared_engine_still_warms_each_session(server) -> None:
    class PerSession(SharedScriptedEngine):
        shared_model = False                                 # sherpa 엔진처럼 연결마다 로드

    server["preload"] = Recorder(cls=PerSession)
    server["session"] = Recorder(cls=PerSession)
    with server["start"]() as client:
        _wait_ready(client)
        _stream(client, _utterance_frames())
    assert PerSession.warmups == 2                           # preload 1 + 세션 1


def test_connection_during_loading_waits_then_streams(server) -> None:
    """모델 로딩 중에 연결해도 오류 없이 기다렸다가 기존 순서대로 응답한다."""
    gate = threading.Event()
    server["preload"] = Recorder(gate=gate)
    with server["start"]() as client:
        threading.Timer(0.3, gate.set).start()
        speech_only = _utterance_frames()[:20]               # endpoint 없이 close
        messages = _stream(client, speech_only)

    kinds = [(m["type"], m.get("is_final")) for m in messages]
    assert ("error", None) not in kinds
    assert kinds[-1] == ("done", None)
    finals = [m for m in messages if m.get("is_final")]
    assert finals and finals[-1]["text"] == " ".join(SCRIPT)  # close → flush → final → done
    assert kinds.index(("transcript", True)) < len(kinds) - 1


# ---------------------------------------------------------------- Fun-ASR 공유 lock
def test_funasr_sessions_share_model_and_serialize_inference(monkeypatch) -> None:
    """세션별 엔진 인스턴스가 하나의 모델을 공유하고, 추론은 한 번에 하나씩 돈다."""
    active = {"now": 0, "max": 0}
    guard = threading.Lock()

    class FakeAutoModel:
        def generate(self, input, **kwargs):
            with guard:
                active["now"] += 1
                active["max"] = max(active["max"], active["now"])
            time.sleep(0.05)
            with wave.open(input[0]) as wf:                  # 입력별 결과가 섞이지 않았는지 확인용
                frames = wf.getnframes()
            with guard:
                active["now"] -= 1
            return [{"text": f"frames={frames}"}]

    model = FakeAutoModel()
    monkeypatch.setattr(funasr_module, "load_model", lambda device: model)
    cfg = StreamConfig(engine="funasr_mlt_nano", device="cpu")
    engines = [funasr_module.FunASRMLTNanoEngine(cfg) for _ in range(3)]
    assert all(e.model is model for e in engines)
    assert engines[0]._lock is engines[1]._lock is engines[2]._lock

    results: dict[int, str] = {}

    def run(i: int) -> None:
        audio = np.zeros(SAMPLE_RATE // 10 * (i + 1), dtype=np.float32)
        results[i] = engines[i].transcribe(audio).text

    threads = [threading.Thread(target=run, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert active["max"] == 1                                # 동시 generate() 없음
    assert results == {i: f"frames={SAMPLE_RATE // 10 * (i + 1)}" for i in range(3)}


# ---------------------------------------------------------------- 세션 정리
def test_close_session_finishes_cleanup_even_if_handler_is_cancelled(tmp_path, monkeypatch) -> None:
    """연결 처리 코루틴이 정리 도중 취소돼도 워커·recorder 정리와 SESSIONS 제거가 끝까지 된다.

    `asyncio.to_thread()` 로 정리하면 취소 시 아직 시작 전인 작업이 큐에서 빠져
    세션이 정리되지 않은 채 남는 경우가 있었다.
    """
    import asyncio

    from stt.audio.recorder import WavRecorder
    from stt.session import StreamingSession

    monkeypatch.setattr(ws_module, "TRANSCRIPTS_DIR", tmp_path)
    monkeypatch.setattr(session_module, "WavRecorder",
                        lambda sid: WavRecorder(sid, directory=tmp_path))
    cfg = StreamConfig(save_wav=True, medical_correction=False)

    async def scenario() -> list:
        from concurrent.futures import ThreadPoolExecutor

        # 기본 스레드풀을 1개로 두고 막아 둔다 → 스레드풀에 맡긴 작업은 큐에서 대기하다
        # 취소되는 최악의 경우가 항상 재현된다(예전 to_thread 구현은 여기서 정리를 잃었다).
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
        gate = threading.Event()
        blocker = loop.run_in_executor(None, gate.wait)

        sessions, tasks = [], []
        for _ in range(5):
            s = StreamingSession(cfg, engine=ScriptedEngine(cfg))
            s.start()
            for frame in _utterance_frames()[:10]:
                s.feed(frame)
            ws_module.SESSIONS[s.session_id] = s
            tasks.append(asyncio.create_task(ws_module.close_session(s)))
            sessions.append(s)
        await asyncio.sleep(0.05)
        for task in tasks:
            task.cancel()                                    # 정리를 기다리던 쪽이 취소됨
        await asyncio.gather(*tasks, return_exceptions=True)
        gate.set()
        await blocker
        return sessions

    sessions = asyncio.run(scenario())
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and any(s.session_id in ws_module.SESSIONS for s in sessions):
        time.sleep(0.02)
    for s in sessions:
        assert s.session_id not in ws_module.SESSIONS
        assert s._worker is None
        assert s.recorder._wf is None                        # WAV 헤더까지 닫혔다
