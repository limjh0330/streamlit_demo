"""파일 전사 — 업로드 또는 브라우저 녹음을 통째로 전사.

    오디오 업로드/녹음 → bytes → 임시 WAV → faster-whisper → 구간별 결과 → TXT
"""
import json
import os
import tempfile
import time

import streamlit as st

from stt.config import WHISPER_SIZES, StreamConfig
from stt.metrics.evaluator import evaluate
from stt.transcript.medical_terms import correct

st.title("파일 전사")
st.caption("전체 오디오를 한 번에 전사합니다. 실시간 결과와 비교할 정답(reference) 만들기에도 씁니다.")


@st.cache_resource(show_spinner="모델 로드 중…")
def get_engine(size: str, language: str, device: str, compute_type: str):
    from stt.asr.whisper import WhisperEngine

    return WhisperEngine(
        StreamConfig(model_size=size, language=language or None,
                     device=device, compute_type=compute_type)
    )


with st.sidebar:
    size = st.selectbox("모델 크기", WHISPER_SIZES, index=2)
    lang = st.text_input("언어 코드", "ko", help="비우면 자동 감지")
    beam_size = st.slider("beam size", 1, 10, 5,
                          help="후보를 여러 개 유지하는 beam search 폭. 클수록 정확·느림")
    device = st.selectbox("device", ["cpu", "cuda"])
    compute_type = st.selectbox("compute type", ["int8", "int8_float16", "float16", "float32"])
    correction = st.toggle("의료 용어 교정", True)

source = st.segmented_control("입력", ["업로드", "녹음"], default="업로드")

audio_bytes = None
if source == "업로드":
    uploaded = st.file_uploader("오디오 파일", type=["wav", "mp3", "m4a", "ogg", "flac", "webm"])
    if uploaded:
        audio_bytes = uploaded.getvalue()
else:
    recorded = st.audio_input("녹음하세요")
    if recorded:
        audio_bytes = recorded.getvalue()

if not audio_bytes:
    st.info("오디오를 업로드하거나 녹음하세요.", icon=":material/upload_file:")
    st.stop()

st.audio(audio_bytes)

if not st.button("전사", type="primary", icon=":material/edit_note:"):
    st.stop()

with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
    f.write(audio_bytes)
    path = f.name

try:
    engine = get_engine(size, lang, device, compute_type)
    started = time.perf_counter()
    with st.status("전사 중…", expanded=True) as status:
        segments, info = engine.transcribe_file(path, beam_size=beam_size)
        rows, text = [], ""
        for seg in segments:                       # generator: 나오는 대로 스트리밍 표시
            line = correct(seg.text) if correction else seg.text
            st.write(f"`{seg.start:5.1f}s → {seg.end:5.1f}s`  {line.strip()}")
            rows.append({"start": round(seg.start, 2), "end": round(seg.end, 2),
                         "text": line.strip()})
            text += line + " "
        elapsed = time.perf_counter() - started
        status.update(label=f"완료 · {elapsed:.1f}초", state="complete")
finally:
    os.unlink(path)

text = text.strip()
duration = getattr(info, "duration", 0.0) or 0.0

cols = st.columns(4)
cols[0].metric("오디오 길이", f"{duration:.1f} s")
cols[1].metric("추론 시간", f"{elapsed:.1f} s")
cols[2].metric("RTF", f"{elapsed / duration:.2f}" if duration else "—")
cols[3].metric("감지 언어", f"{info.language} ({info.language_probability:.0%})")

st.subheader("전체 텍스트")
st.text_area("전체 텍스트", text, height=200, label_visibility="collapsed")

with st.container(horizontal=True):
    st.download_button("TXT 다운로드", text.encode("utf-8"), "transcript.txt",
                       icon=":material/download:")
    st.download_button(
        "구간 JSON 다운로드",
        json.dumps(rows, ensure_ascii=False, indent=2).encode("utf-8"),
        "segments.json",
        icon=":material/data_object:",
    )

st.session_state["reference_text"] = text
st.caption("이 결과는 **성능 비교** 페이지에서 정답(reference)으로 바로 불러올 수 있습니다.")

with st.expander("정답과 비교(WER / CER)"):
    ref = st.text_area("정답 전사문", "", height=120,
                       placeholder="사람이 작성한 정답 전사문을 붙여넣으세요")
    if ref.strip():
        st.json(evaluate(ref, text))
