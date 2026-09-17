"""초기 단일 파일 데모. 현재는 streamlit_app.py (app_pages/file_stt.py) 로 대체되었습니다.

    streamlit run streamlit_app.py
"""

import streamlit as st
import tempfile, os
from faster_whisper import WhisperModel

# Streamlit UI settings
st.set_page_config(page_title="🎤 STT", page_icon="🎤")
st.title("🎤 음성 인식 (faster-whisper)")

@st.cache_resource(show_spinner="🧠 모델 로드 중...") # Model load and reuse
def get_model(size: str):
    return WhisperModel(size, device="cpu", compute_type="int8") # Use "int8" for faster inference on CPU

with st.sidebar:
    size = st.selectbox("모델 크기", ["tiny", "base", "small", "medium"], index=1)
    lang = st.text_input("언어 코드 (자동: 빈칸)", "ko")

src = st.radio("입력", ["업로드", "녹음"], horizontal=True)
audio_bytes = None # voice data
if src == "업로드":
    up = st.file_uploader("오디오", type=["mp3", "wav", "m4a", "ogg"])
    if up: audio_bytes = up.getvalue()
else:
    rec = st.audio_input("녹음하세요")
    if rec: audio_bytes = rec.getvalue()

if not audio_bytes:
    st.stop()

st.audio(audio_bytes) # play audio

if st.button("📝 전사", type="primary"):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes); path = f.name # 임시 오디오 파일 생성

    # Load model and transcribe
    model = get_model(size)
    with st.status("전사 중...", expanded=True) as status:
        # Transcribe audio(segments: 음성 구간별 전사 결과, info: dict with metadata)
        segments, info = model.transcribe(path, language=lang or None, beam_size=5) 
        # beam search decoding
        # 한 번에 가장 가능성이 높은 token 하나만 고르는 대신 여러 후보를 유지하면서 탐색
        text = ""
        for seg in segments:
            line = f"[{seg.start:.1f}s → {seg.end:.1f}s] {seg.text}"
            st.write(line)
            text += seg.text + " "
        status.update(label="완료", state="complete")

    os.unlink(path)
    st.subheader("📄 전체 텍스트")
    st.text_area("", text.strip(), height=200)
    st.download_button("TXT 다운로드", text.strip().encode("utf-8"), "transcript.txt")
