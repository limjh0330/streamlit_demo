"""ER 실시간 음성 전사 데모 — Streamlit 진입점.

    streamlit run streamlit_app.py
"""
import streamlit as st

from stt.config import StreamConfig

st.set_page_config(
    page_title="ER 실시간 음성 전사",
    page_icon=":material/graphic_eq:",
    layout="wide",
)

# 페이지 간 공유하는 상태는 여기서 한 번만 초기화한다.
st.session_state.setdefault("config", StreamConfig())
st.session_state.setdefault("session", None)
st.session_state.setdefault("mic", None)
st.session_state.setdefault("last_result", None)

page = st.navigation(
    [
        st.Page(
            "app_pages/realtime.py",
            title="실시간 전사",
            icon=":material/mic:",
            default=True,
        ),
        st.Page("app_pages/file_stt.py", title="파일 전사", icon=":material/audio_file:"),
        st.Page("app_pages/browser.py", title="태블릿 · 브라우저", icon=":material/tablet:"),
        st.Page("app_pages/metrics.py", title="성능 비교", icon=":material/speed:"),
    ],
    position="top",
)

page.run()
