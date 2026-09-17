"""브라우저 마이크 스트리밍 컴포넌트 (Streamlit Custom Component v2).

    Browser mic → getUserMedia() → AudioWorklet → PCM16 16 kHz chunk
      → WebSocket → RunPod 백엔드(streaming ASR) → partial transcript → Browser

CCv2 는 iframe 을 쓰지 않고 앱 문서 안에서 실행되므로 마이크 권한을 위임할
필요가 없다. 부분 전사는 브라우저에서 바로 그리고, 세션이 끝날 때만
`result` state 를 Python 으로 올린다(= 리런 1회).
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

_DIR = Path(__file__).resolve().parent

#: 컴포넌트 등록은 import 시 한 번만. 마운트 함수 안에서 등록하면 재등록된다.
_LIVE_MIC = st.components.v2.component(
    "er_live_mic",
    html=(_DIR / "live_mic.html").read_text(encoding="utf-8"),
    css=(_DIR / "live_mic.css").read_text(encoding="utf-8"),
    js=(_DIR / "live_mic.js").read_text(encoding="utf-8"),
)


def live_mic(
    *,
    key: str = "live_mic",
    ws_url: str = "",
    backend_port: int = 8000,
    chunk_ms: int = 100,
    engine: str = "whisper",
    model_size: str = "small",
    language: str | None = "ko",
    window_sec: float = 5.0,
    overlap_sec: float = 1.5,
    silence_sec: float = 0.7,
    vad_threshold_db: float = 12.0,
    medical_correction: bool = True,
    save_wav: bool = True,
) -> dict | None:
    """마이크 캡처 UI 를 그리고, 세션이 끝나면 결과 요약을 돌려준다.

    `ws_url` 이 비어 있으면 브라우저가 주소창과 `backend_port` 로 유도한다.
    설정은 다음 **녹음 시작** 때 적용된다(진행 중인 세션은 바뀌지 않는다).
    """
    result = _LIVE_MIC(
        key=key,
        data={
            "ws_url": ws_url,
            "backend_port": backend_port,
            "chunk_ms": chunk_ms,
            "engine": engine,
            "model_size": model_size,
            "language": language,
            "window_sec": window_sec,
            "overlap_sec": overlap_sec,
            "silence_sec": silence_sec,
            "vad_threshold_db": vad_threshold_db,
            "medical_correction": medical_correction,
            "save_wav": save_wav,
        },
        on_result_change=lambda: None,
    )
    return result.result
