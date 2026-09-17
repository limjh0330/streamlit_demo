"""ER 실시간 음성 전사 데모 — Streamlit 진입점 (RunPod 에서 실행).

    streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0

실시간 전사는 같은 인스턴스에서 도는 STT 백엔드가 필요하다.

    python -m server.main --host 0.0.0.0 --port 8000
"""
import streamlit as st

st.set_page_config(
    page_title="ER 실시간 음성 전사",
    page_icon=":material/graphic_eq:",
    layout="wide",
)

page = st.navigation(
    [
        st.Page(
            "app_pages/realtime.py",
            title="실시간 전사",
            icon=":material/mic:",
            default=True,
        ),
        st.Page("app_pages/file_stt.py", title="파일 전사", icon=":material/audio_file:"),
        st.Page("app_pages/metrics.py", title="성능 비교", icon=":material/speed:"),
    ],
    position="top",
)

page.run()
