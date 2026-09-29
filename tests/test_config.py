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


def _ws_url(**env: str) -> str:
    clean = {k: v for k, v in os.environ.items()
             if k not in ("RUNPOD_POD_ID", "STT_WS_URL", "STT_BACKEND_PORT")}
    out = subprocess.run(
        [sys.executable, "-c", "from stt.config import WS_URL; print(WS_URL)"],
        cwd=ROOT, env={**clean, **env}, capture_output=True, text=True, timeout=60,
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


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
