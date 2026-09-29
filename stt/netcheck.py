"""STT 서버 도달 가능 여부 진단 (Streamlit 사이드바용).

브라우저가 붙을 WebSocket 주소(ws/wss)를 같은 호스트의 liveness(`/api/v1/stt/health`)로
바꿔 GET 해 본다. WebSocket 이 실패하면 브라우저는 원인을 알려 주지 않으므로, 여기서
흔한 원인을 구분해 안내한다.

  - RunPod 프록시가 본문 없는 404 를 준다 → 그 포트가 Pod 의 "Expose HTTP Ports" 에 없음
  - 연결 자체가 안 된다 → 주소 오타, 서버 중지, (localhost 라면) VS Code 포트 포워딩 끊김
"""
from __future__ import annotations

import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

from .config import API_BASE


@dataclass
class Probe:
    ok: bool
    message: str
    health_url: str = ""


def health_url_for(ws_url: str) -> str:
    """ws(s)://host/ws/v1/stt/browser → http(s)://host/api/v1/stt/health"""
    u = urllib.parse.urlparse(ws_url.strip())
    if u.scheme not in ("ws", "wss") or not u.netloc:
        return ""
    scheme = "https" if u.scheme == "wss" else "http"
    return f"{scheme}://{u.netloc}{API_BASE}/health"


def probe(ws_url: str, timeout: float = 5.0) -> Probe:
    url = health_url_for(ws_url)
    if not url:
        return Probe(False, "ws:// 또는 wss:// 로 시작하는 주소가 아닙니다.")
    host = urllib.parse.urlparse(url).hostname or ""
    # Cloudflare(RunPod 프록시)는 Python 기본 User-Agent 를 봇으로 보고 403 을 주므로 이름을 붙인다
    req = urllib.request.Request(url, headers={"User-Agent": "er-stt-netcheck/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if r.status == 200:
                return Probe(True, "STT 서버 연결 확인됨", url)
            return Probe(False, f"STT 서버가 HTTP {r.status} 를 반환했습니다.", url)
    except urllib.error.HTTPError as e:
        body = e.read() or b""
        via_runpod = host.endswith(".proxy.runpod.net")
        if e.code == 404 and via_runpod and not body.strip():
            port = host.split(".")[0].rsplit("-", 1)[-1]
            return Probe(False,
                         f"RunPod 프록시가 포트 {port} 를 연결하지 않습니다(404). "
                         f"Pod 설정의 **Expose HTTP Ports** 에 `{port}` 를 추가하세요.", url)
        return Probe(False, f"STT 서버에 닿았지만 HTTP {e.code} 를 받았습니다.", url)
    except (urllib.error.URLError, OSError) as e:
        reason = getattr(e, "reason", e)
        hint = (" localhost 주소는 VS Code 포트 포워딩이 있을 때만 연결됩니다."
                if host in ("localhost", "127.0.0.1") else "")
        return Probe(False, f"STT 서버에 연결할 수 없습니다({reason}).{hint}", url)
