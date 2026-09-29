"""[Unit test] Swagger UI / OpenAPI 문서.

    pytest tests/test_api_docs.py
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from server.main import app
from stt.config import API_BASE, ROOT

REST_PATHS = [
    "/health", "/health/ready", "/engines", "/sessions",
    "/sessions/{session_id}/transcript", "/sessions/{session_id}/audio",
]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("STT_PRELOAD", "0")                   # 문서 확인에 모델은 필요 없다
    with TestClient(app) as c:
        yield c


def test_swagger_ui_and_redoc_served_under_api_base(client) -> None:
    docs = client.get(f"{API_BASE}/docs")
    assert docs.status_code == 200
    assert "swagger-ui" in docs.text
    assert f"{API_BASE}/openapi.json" in docs.text          # UI 가 base path 아래 명세를 읽는다
    assert '"tryItOutEnabled": true' in docs.text            # 바로 Execute 가능
    assert client.get(f"{API_BASE}/redoc").status_code == 200


def test_unversioned_docs_redirects_to_swagger(client) -> None:
    r = client.get("/docs", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert r.headers["location"] == f"{API_BASE}/docs"


def test_openapi_lists_every_rest_endpoint_with_tags(client) -> None:
    spec = client.get(f"{API_BASE}/openapi.json").json()
    assert [t["name"] for t in spec["tags"]] == ["Health", "Engines", "Sessions"]
    for path in REST_PATHS:
        op = spec["paths"][f"{API_BASE}{path}"]["get"]
        assert op.get("tags") and op.get("summary"), path
    # readiness 의 503, 세션 조회의 404 가 문서에 드러난다
    assert {"200", "503"} <= set(spec["paths"][f"{API_BASE}/health/ready"]["get"]["responses"])
    assert "404" in spec["paths"][f"{API_BASE}/sessions/{{session_id}}/transcript"]["get"]["responses"]
    # WebSocket 은 OpenAPI 로 표현할 수 없어 설명에 규격을 싣는다
    assert "/ws/v1/stt/stream" in spec["info"]["description"]


def test_response_models_keep_existing_fields(client) -> None:
    """문서용 response_model 이 기존 응답 필드를 걸러내지 않는다."""
    assert set(client.get(f"{API_BASE}/health").json()) == {
        "status", "sample_rate", "active_sessions", "gpu_memory_mb"}
    engines = client.get(f"{API_BASE}/engines").json()
    assert {"active_engine", "active_status", "active_config", "engines",
            "whisper_sizes", "defaults"} <= set(engines)
    assert all({"ready", "detail", "loaded"} <= set(m) for m in engines["engines"].values())


def test_docs_can_be_disabled_with_env() -> None:
    """STT_DOCS=0 이면 문서·명세를 노출하지 않는다(설정은 import 시 읽으므로 새 프로세스로)."""
    code = (
        "from fastapi.testclient import TestClient; from server.main import app;"
        "c = TestClient(app);"
        "print([c.get(u, follow_redirects=False).status_code for u in"
        " ('/api/v1/stt/docs', '/api/v1/stt/openapi.json', '/docs', '/api/v1/stt/health')])"
    )
    env = {**os.environ, "STT_DOCS": "0", "STT_PRELOAD": "0"}
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                         capture_output=True, text=True, timeout=120)
    assert out.stdout.strip().splitlines()[-1] == "[404, 404, 404, 200]", out.stderr[-2000:]
