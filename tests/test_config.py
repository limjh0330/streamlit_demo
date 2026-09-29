"""[Unit test] 브라우저 → STT WebSocket 주소 결정 규칙 (stt.config.WS_URL).

RunPod 에서는 공개 프록시 주소가 기본이어야 한다. 비어 있으면 브라우저가 주소창에서
유도하는데, Streamlit 을 VS Code 포트 포워딩(localhost)으로 열면 ws://localhost:8000 이
되어 VS Code 창을 닫는 순간 마이크 연결이 끊긴다.

설정은 import 시 읽으므로 환경변수를 바꿔 새 프로세스에서 확인한다.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

from stt.config import ROOT


def _urls(**env: str) -> tuple[str, str]:
    """(API_URL, WS_URL) — 주어진 환경변수만 켠 새 프로세스에서 읽는다."""
    clean = {k: v for k, v in os.environ.items()
             if k not in ("RUNPOD_POD_ID", "STT_WS_URL", "STT_BACKEND_PORT",
                          "STT_SERVER_URL", "STT_API_URL")}
    out = subprocess.run(
        [sys.executable, "-c", "from stt.config import API_URL, WS_URL; print(API_URL); print(WS_URL)"],
        cwd=ROOT, env={**clean, **env}, capture_output=True, text=True, timeout=60,
    )
    assert out.returncode == 0, out.stderr
    api, ws = (out.stdout.splitlines() + ["", ""])[:2]
    return api.strip(), ws.strip()


def _ws_url(**env: str) -> str:
    return _urls(**env)[1]


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, ""),                                            # 로컬: 브라우저가 주소창에서 유도
        ({"RUNPOD_POD_ID": "druyf5wybhan4k"},
         "wss://druyf5wybhan4k-8000.proxy.runpod.net/ws/v1/stt/browser"),
        ({"RUNPOD_POD_ID": "abc", "STT_BACKEND_PORT": "8001"},
         "wss://abc-8001.proxy.runpod.net/ws/v1/stt/browser"),
        ({"RUNPOD_POD_ID": "abc", "STT_WS_URL": "wss://example.org/ws"},  # 명시값이 우선
         "wss://example.org/ws"),
    ],
)
def test_browser_ws_url(env: dict, expected: str) -> None:
    assert _ws_url(**env) == expected


def test_local_streamlit_uses_one_server_url_for_rest_and_websocket() -> None:
    """구성 B: 로컬 PC 의 Streamlit 이 RunPod STT 를 쓸 때 STT_SERVER_URL 하나로 둘 다 정해진다."""
    api, ws = _urls(STT_SERVER_URL="https://druyf5wybhan4k-8000.proxy.runpod.net/")
    assert api == "https://druyf5wybhan4k-8000.proxy.runpod.net"
    assert ws == "wss://druyf5wybhan4k-8000.proxy.runpod.net/ws/v1/stt/browser"


def test_server_url_beats_pod_id_and_explicit_values_win() -> None:
    _, ws = _urls(STT_SERVER_URL="http://10.0.0.5:8000", RUNPOD_POD_ID="abc")
    assert ws == "ws://10.0.0.5:8000/ws/v1/stt/browser"
    api, ws = _urls(STT_SERVER_URL="https://a.example", STT_API_URL="http://127.0.0.1:9000",
                    STT_WS_URL="wss://b.example/ws")
    assert (api, ws) == ("http://127.0.0.1:9000", "wss://b.example/ws")


def test_default_rest_is_localhost() -> None:
    assert _urls()[0] == "http://127.0.0.1:8000"
