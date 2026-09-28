"""Streamlit 페이지가 예외 없이 렌더링되는지 확인한다(백엔드 없이).

    pytest tests/test_streamlit_pages.py
"""
from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from stt.config import ROOT, WS_URL

PAGES = ["realtime.py", "file_stt.py", "metrics.py"]


def _run(page: str) -> AppTest:
    # 상대 경로는 이 테스트 파일 기준으로 풀리므로 프로젝트 루트 기준 절대 경로를 준다
    return AppTest.from_file(str(ROOT / "app_pages" / page), default_timeout=60).run()


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exception(page: str) -> None:
    at = _run(page)
    assert not at.exception, [e.value for e in at.exception]


def test_realtime_ws_url_defaults_to_config() -> None:
    """WebSocket 주소 입력칸의 기본값은 STT_WS_URL(비우면 브라우저가 유도)이다."""
    at = _run("realtime.py")
    fields = {t.label: t.value for t in at.sidebar.text_input}
    assert fields["WebSocket 주소 (브라우저 → 백엔드)"] == WS_URL
