"""실시간 전사 — 이 컴퓨터의 마이크(sounddevice)를 직접 사용.

    마이크 → sounddevice callback → 100 ms block → Queue
    → 2~5초 누적(sliding window) → ASR → 부분 전사 → 화면 갱신
"""

import streamlit as st

from stt.audio.mic import MicrophoneStream, list_input_devices
from stt.config import ENGINE_CHOICES, WHISPER_SIZES, StreamConfig
from stt.session import StreamingSession

st.title("실시간 전사")
st.caption("이 앱이 실행 중인 컴퓨터의 마이크로 수음합니다. 태블릿 마이크를 쓰려면 "
           "**태블릿 · 브라우저** 페이지를 사용하세요.")

running = st.session_state.session is not None

# ------------------------------------------------------------------ 설정
with st.sidebar:
    st.subheader("엔진")
    engine = st.selectbox("ASR 엔진", ENGINE_CHOICES, disabled=running,
                          help="zipformer / sensevoice 는 sherpa-onnx 와 모델 파일이 필요합니다.")
    model_size = st.selectbox("Whisper 모델", WHISPER_SIZES, index=2,
                              disabled=running or engine != "whisper")
    language = st.text_input("언어 코드", "ko", disabled=running,
                             help="비우면 자동 감지")

    st.subheader("스트리밍")
    window_sec = st.slider("window (초)", 2.0, 10.0, 5.0, 0.5, disabled=running)
    overlap_sec = st.slider("overlap (초)", 0.0, 4.0, 1.5, 0.5, disabled=running)
    silence_sec = st.slider("발화 종료 무음 (초)", 0.2, 3.0, 0.7, 0.1, disabled=running)
    vad_db = st.slider("VAD 민감도 (dB)", 4.0, 24.0, 12.0, 1.0, disabled=running,
                       help="낮을수록 민감. 노이즈 플로어 대비 마진입니다.")

    st.subheader("입출력")
    devices = list_input_devices()
    device_labels = ["기본 장치"] + [f"[{d['index']}] {d['name']}" for d in devices]
    device_pick = st.selectbox("마이크", device_labels, disabled=running)
    device = None if device_pick == "기본 장치" else devices[device_labels.index(device_pick) - 1]["index"]
    correction = st.toggle("의료 용어 교정", True, disabled=running)
    save_wav = st.toggle("원본 WAV 저장", True, disabled=running)
    refresh = st.select_slider("화면 갱신 주기", ["0.3s", "0.5s", "1s", "2s"], value="0.5s")

# ------------------------------------------------------------------ 제어
with st.container(horizontal=True):
    start = st.button("녹음 시작", type="primary", icon=":material/mic:", disabled=running)
    stop = st.button("정지", icon=":material/stop:", disabled=not running)

if start:
    config = StreamConfig(
        engine=engine, model_size=model_size, language=language or None,
        window_sec=window_sec, overlap_sec=overlap_sec, silence_sec=silence_sec,
        vad_threshold_db=vad_db, medical_correction=correction, save_wav=save_wav,
    )
    with st.spinner("모델 로드 중… 최초 1회는 수십 초 걸릴 수 있습니다."):
        try:
            session = StreamingSession(config)
            session.start()
            session.engine.warmup()
            mic = MicrophoneStream(device=device, on_block=session.feed)
            mic.start()
        except Exception as e:
            st.error(f"시작하지 못했습니다: {e}")
            st.stop()
    st.session_state.session = session
    st.session_state.mic = mic
    st.rerun()

if stop and st.session_state.session is not None:
    with st.spinner("남은 오디오를 전사하는 중…"):
        st.session_state.mic.stop()
        summary = st.session_state.session.stop()
        path = st.session_state.session.save_transcript()
    st.session_state.last_result = {**summary, "transcript_path": str(path)}
    st.session_state.session = None
    st.session_state.mic = None
    st.rerun()

# ------------------------------------------------------------------ 라이브 화면
session = st.session_state.session

if session is not None:
    @st.fragment(run_every=refresh)
    def live_view():
        session.drain_events()          # 큐가 무한히 자라지 않게 비워준다
        metrics = session.metrics()

        level = max(0.0, min(1.0, (metrics["level_db"] + 60) / 60))
        st.progress(level, text="녹음 중" if metrics["is_speech"] else "무음")

        cols = st.columns(4)
        cols[0].metric("오디오", f"{metrics['audio_sec']:.0f} s")
        cols[1].metric("RTF", f"{metrics['rtf_mean']:.2f}")
        cols[2].metric("첫 partial", f"{metrics['first_partial_ms']:.0f} ms")
        cols[3].metric("발화", metrics["utterances"])

        for utt in session.merger.utterances:
            st.markdown(f"`{utt.start:5.1f}s → {utt.end:5.1f}s`  {session.postprocess(utt.text)}")

        stable = session.postprocess(session.merger.stable_text)
        partial = session.postprocess(session.merger.partial_text)
        if stable or partial:
            st.markdown(f"**{stable}** :gray[{partial}]")
        elif not session.merger.utterances:
            st.info("말을 시작하면 부분 전사가 여기에 나타납니다.", icon=":material/hearing:")

        if session.error:
            st.warning(session.error, icon=":material/warning:")

    live_view()
    st.stop()

# ------------------------------------------------------------------ 결과 화면
result = st.session_state.last_result
if result is None:
    st.info("설정을 고른 뒤 **녹음 시작** 을 누르세요.", icon=":material/info:")
    st.stop()

st.subheader("마지막 세션 결과")
m = result["metrics"]
cols = st.columns(5)
cols[0].metric("오디오", f"{m['audio_sec']:.0f} s")
cols[1].metric("RTF (평균)", f"{m['rtf_mean']:.2f}")
cols[2].metric("첫 partial", f"{m['first_partial_ms']:.0f} ms")
cols[3].metric("final 지연", f"{m['final_latency_ms']:.0f} ms")
cols[4].metric("revision rate", f"{m['revision_rate'] * 100:.0f} %")

for utt in result["utterances"]:
    st.markdown(f"`{utt['start']:5.1f}s → {utt['end']:5.1f}s`  {utt['text']}")

full_text = " ".join(u["text"] for u in result["utterances"]) or result["stable"]
st.text_area("전체 텍스트", full_text, height=180)

with st.container(horizontal=True):
    st.download_button("TXT 다운로드", full_text.encode("utf-8"),
                       f"{result['session_id']}.txt", icon=":material/download:")
    if result.get("wav"):
        st.caption(f"원본 WAV: `{result['wav']}`")

with st.expander("세부 지표"):
    st.json(m)
