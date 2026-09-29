"""[Unit test] Streamlit 사이드바의 STT 연결 진단 (stt.netcheck)."""
from __future__ import annotations

import io
import urllib.error

import pytest

import stt.netcheck as netcheck
from stt.netcheck import health_url_for, probe


@pytest.mark.parametrize(("ws", "http"), [
    ("wss://abc-8000.proxy.runpod.net/ws/v1/stt/browser", "https://abc-8000.proxy.runpod.net/api/v1/stt/health"),
    ("ws://localhost:8000/ws/v1/stt/browser", "http://localhost:8000/api/v1/stt/health"),
    ("https://abc/ws", ""),
    ("", ""),
])
def test_health_url_for(ws: str, http: str) -> None:
    assert health_url_for(ws) == http


class _Resp:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _raise(code: int, body: bytes):
    def fake(req, timeout):
        raise urllib.error.HTTPError(req.full_url, code, "x", {}, io.BytesIO(body))
    return fake


def test_ok_and_user_agent_is_set(monkeypatch) -> None:
    seen = {}

    def fake(req, timeout):
        seen["ua"] = req.get_header("User-agent")
        return _Resp()

    monkeypatch.setattr(netcheck.urllib.request, "urlopen", fake)
    assert probe("wss://abc-8000.proxy.runpod.net/ws/v1/stt/browser").ok
    assert seen["ua"] and "Python-urllib" not in seen["ua"]   # Cloudflare 403 회피


def test_runpod_unexposed_port_is_explained(monkeypatch) -> None:
    monkeypatch.setattr(netcheck.urllib.request, "urlopen", _raise(404, b""))
    r = probe("wss://druyf5wybhan4k-8000.proxy.runpod.net/ws/v1/stt/browser")
    assert not r.ok and "Expose HTTP Ports" in r.message and "8000" in r.message


def test_other_404_is_not_misreported_as_unexposed_port(monkeypatch) -> None:
    monkeypatch.setattr(netcheck.urllib.request, "urlopen", _raise(404, b'{"detail":"Not Found"}'))
    r = probe("wss://abc-8888.proxy.runpod.net/ws/v1/stt/browser")
    assert not r.ok and "Expose HTTP Ports" not in r.message


def test_localhost_failure_mentions_port_forwarding(monkeypatch) -> None:
    def refuse(req, timeout):
        raise urllib.error.URLError(ConnectionRefusedError(61, "Connection refused"))
    monkeypatch.setattr(netcheck.urllib.request, "urlopen", refuse)
    r = probe("ws://localhost:8000/ws/v1/stt/browser")
    assert not r.ok and "포트 포워딩" in r.message


def test_rejects_non_websocket_url() -> None:
    assert not probe("https://example.org").ok
