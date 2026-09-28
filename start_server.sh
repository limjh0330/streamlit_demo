#!/bin/bash
# STT 서버 실행 (RunPod 운영용)
#
#   cd /workspace/streamlit_demo && ./start_server.sh
#
# 시작하면 활성 엔진을 preload → warm-up 한 뒤 READY 가 된다.
#   curl -s http://localhost:8000/api/v1/stt/health/ready
#
# 밖에서 지정한 환경변수가 항상 우선한다. 예:
#   STT_ENGINE=sensevoice ./start_server.sh
#   STT_CONFIG='{"save_wav":false}' ./start_server.sh
set -euo pipefail

cd "$(dirname "$0")"

export STT_ENGINE="${STT_ENGINE:-funasr_mlt_nano}"
export STT_TIMESTAMPS="${STT_TIMESTAMPS:-1}"
export STT_PRELOAD="${STT_PRELOAD:-1}"
export STT_BACKEND_PORT="${STT_BACKEND_PORT:-8000}"
export PYTHONUNBUFFERED=1                 # 로그를 바로 내보낸다

# 인터프리터: $PYTHON > 프로젝트 .venv > 시스템 python
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x .venv/bin/python ]]; then PYTHON=.venv/bin/python; else PYTHON=python; fi
fi

echo "[STT] Starting server: engine=${STT_ENGINE} port=${STT_BACKEND_PORT} python=${PYTHON}"

# exec: 셸 대신 python 이 PID 를 이어받아 SIGTERM 을 직접 받는다(정상 종료)
exec "${PYTHON}" -m server.main \
  --host "${STT_HOST:-0.0.0.0}" \
  --port "${STT_BACKEND_PORT}" \
  "$@"
