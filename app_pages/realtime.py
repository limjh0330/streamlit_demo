"""실시간 전사 — 브라우저 마이크를 RunPod 백엔드로 스트리밍한다.

    Browser microphone
        ↓ getUserMedia()
    100~500 ms PCM16 chunk (AudioWorklet)
        ↓ WebSocket
    RunPod: StreamingSession → VAD → sliding window → ASR → merger
        ↓ partial / final
    Browser

부분 전사는 브라우저에서 직접 갱신되고, 세션이 끝나면 요약이 Python 으로 올라온다.
"""
import json
import urllib.request

import streamlit as st

from stt.config import (
    API_URL,
    BACKEND_PORT,
    CHUNK_MS_CHOICES,
    ENGINE_CHOICES,
    WHISPER_SIZES,
    WS_URL,
    StreamConfig,
)
from web.live_mic import live_mic

KEY = "live_mic"

st.title("실시간 전사")
st.caption(
    "브라우저 마이크로 수음해 RunPod STT 백엔드로 스트리밍합니다. "
    "마이크는 **HTTPS 또는 localhost** 에서만 열립니다."
)


@st.cache_data(ttl="30s", show_spinner=False)
def backend_engines(api_url: str) -> dict:
    """백엔드에 설치·준비된 엔진을 물어본다. 같은 인스턴스이므로 localhost."""
    with urllib.request.urlopen(f"{api_url}/api/engines", timeout=3) as r:
        return json.load(r)


defaults = StreamConfig()

# ------------------------------------------------------------------ 설정
with st.sidebar:
    st.subheader("백엔드")
    api_url = st.text_input("REST 주소 (Python → 백엔드)", API_URL)
    ws_url = st.text_input(
        "WebSocket 주소 (브라우저 → 백엔드)",
        WS_URL,
        placeholder=f"비우면 주소창에서 유도 (포트 {BACKEND_PORT})",
        help="RunPod 은 포트마다 호스트가 달라 자동 유도합니다. "
        "다르게 노출했다면 wss://… /ws 를 직접 적으세요.",
    )

    try:
        info = backend_engines(api_url)
    except Exception as e:
        info = None
        st.error(f"백엔드에 연결하지 못했습니다: {e}", icon=":material/error:")
        st.code("python -m server.main --host 0.0.0.0 --port %d" % BACKEND_PORT,
                language="bash")

    ready = {name for name, meta in (info or {}).get("engines", {}).items() if meta["ready"]}

    st.subheader("엔진")
    engine = st.selectbox(
        "ASR 엔진",
        ENGINE_CHOICES,
        format_func=lambda n: n if not info or n in ready else f"{n} (모델 없음)",
    )
    if info and engine not in ready:
        st.warning(info["engines"][engine]["detail"], icon=":material/warning:")
    model_size = st.selectbox("Whisper 모델", WHISPER_SIZES, index=2,
                              disabled=engine != "whisper")
    language = st.text_input("언어 코드", "ko", help="비우면 자동 감지")

    st.subheader("스트리밍")
    chunk_ms = st.select_slider("청크 길이 (ms)", CHUNK_MS_CHOICES, value=100,
                                help="브라우저가 한 번에 보내는 오디오 길이")
    window_sec = st.slider("window (초)", 2.0, 10.0, defaults.window_sec, 0.5)
    overlap_sec = st.slider("overlap (초)", 0.0, 4.0, defaults.overlap_sec, 0.5)
    silence_sec = st.slider("발화 종료 무음 (초)", 0.2, 3.0, defaults.silence_sec, 0.1)
    vad_db = st.slider("VAD 민감도 (dB)", 4.0, 24.0, defaults.vad_threshold_db, 1.0,
                       help="낮을수록 민감. 노이즈 플로어 대비 마진입니다.")

    st.subheader("처리")
    correction = st.toggle("의료 용어 교정", True)
    save_wav = st.toggle("원본 WAV 저장", True)

if info:
    st.caption(
        f"백엔드 `{engine}`"
        + (f" · `{model_size}`" if engine == "whisper" else "")
        + f" · {info['defaults']['device']} · 청크 {chunk_ms} ms"
    )

# ------------------------------------------------------------------ 라이브 화면
# 설정은 다음 "녹음 시작" 때 적용된다. 녹음 중 슬라이더를 움직여도 끊기지 않는다.
result = live_mic(
    key=KEY,
    ws_url=ws_url.strip(),
    backend_port=BACKEND_PORT,
    chunk_ms=int(chunk_ms),
    engine=engine,
    model_size=model_size,
    language=language.strip() or None,
    window_sec=window_sec,
    overlap_sec=overlap_sec,
    silence_sec=silence_sec,
    vad_threshold_db=vad_db,
    medical_correction=correction,
    save_wav=save_wav,
)

# ------------------------------------------------------------------ 결과 화면
if not result:
    st.stop()

st.subheader("마지막 세션 결과")
m = result["metrics"]
cols = st.columns(5)
cols[0].metric("오디오", f"{m.get('audio_sec', 0):.0f} s")
cols[1].metric("RTF (평균)", f"{m.get('rtf_mean', 0):.2f}")
cols[2].metric("첫 partial", f"{m.get('first_partial_ms', 0):.0f} ms")
cols[3].metric("final 지연", f"{m.get('final_latency_ms', 0):.0f} ms")
cols[4].metric("revision rate", f"{m.get('revision_rate', 0) * 100:.0f} %")

full_text = " ".join(u["text"] for u in result["utterances"]) or result["text"]
st.text_area("전체 텍스트", full_text, height=180)

with st.container(horizontal=True):
    st.download_button("TXT 다운로드", full_text.encode("utf-8"),
                       f"{result['session_id']}.txt", icon=":material/download:")
    if result.get("wav"):
        st.caption(f"원본 WAV: `{result['wav']}`")

st.caption("전사 결과는 백엔드의 `transcripts/` 에 저장되어 **성능 비교** 페이지에 나타납니다.")

with st.expander("세부 지표"):
    st.json(m)
