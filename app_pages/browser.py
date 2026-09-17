"""태블릿·브라우저 경로 안내 및 서버 상태 확인.

Streamlit 은 서버 프로세스가 도는 컴퓨터의 마이크만 읽는다. 태블릿 마이크를
쓰려면 브라우저에서 AudioWorklet 으로 PCM 을 떠서 WebSocket 으로 보내는
별도 STT 서버(`server/main.py`)가 필요하다.
"""
import socket

import streamlit as st

st.title("태블릿 · 브라우저 수음")

st.markdown(
    """
```
Tablet / Browser                     STT Server
─────────────────                    ──────────────────────────
Microphone                           WebSocket /ws
  ↓ getUserMedia()                     ↓
AudioWorklet (pcm-worklet.js)        Session Manager
  ↓ PCM16 / 16 kHz / mono              ├→ Audio Recorder → recordings/*.wav
WebSocket  ──── binary chunk ──────→   └→ Audio Buffer → VAD → ASR → Merger
                                           ↓
Browser UI ←──── JSON partial/final ──── transcript
```
"""
)


def local_ip() -> str:
    """LAN 에서 태블릿이 접속할 이 컴퓨터의 IP 를 추정한다."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))      # 패킷을 실제로 보내지는 않는다
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


ip = local_ip()
port = st.number_input("서버 포트", 1024, 65535, 8000, step=1)
url = f"http://{ip}:{port}"

st.subheader("1. STT 서버 실행")
st.code("python -m server.main --host 0.0.0.0 --port %d" % port, language="bash")

st.subheader("2. 태블릿에서 접속")
with st.container(horizontal=True):
    st.code(url, language=None)
    st.link_button("이 컴퓨터에서 열기", f"http://localhost:{port}",
                   icon=":material/open_in_new:")

st.warning(
    "브라우저는 **보안 컨텍스트(HTTPS)** 또는 `localhost` 에서만 마이크를 허용합니다. "
    "태블릿에서 IP 로 접속하려면 HTTPS 가 필요합니다.",
    icon=":material/lock:",
)

st.subheader("HTTPS 로 실행하기")
st.code(
    """# 자체 서명 인증서 생성 (태블릿에서 '신뢰할 수 없음' 경고를 한 번 통과해야 합니다)
openssl req -x509 -newkey rsa:2048 -nodes -days 365 \\
  -keyout key.pem -out cert.pem -subj "/CN=%s" \\
  -addext "subjectAltName=IP:%s"

python -m server.main --host 0.0.0.0 --port %d --certfile cert.pem --keyfile key.pem"""
    % (ip, ip, port),
    language="bash",
)

st.subheader("서버 상태")
if st.button("연결 확인", icon=":material/sync:"):
    try:
        import json
        import urllib.request

        with urllib.request.urlopen(f"http://localhost:{port}/api/engines", timeout=3) as r:
            info = json.load(r)
        st.success("서버 응답 정상", icon=":material/check_circle:")
        for name, meta in info["engines"].items():
            icon = ":material/check_circle:" if meta["ready"] else ":material/cancel:"
            st.markdown(f"{icon} **{name}** — {meta['detail']}")
    except Exception as e:
        st.error(f"서버에 연결하지 못했습니다: {e}", icon=":material/error:")
