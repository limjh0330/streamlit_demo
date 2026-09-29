#!/bin/bash
# STT 서버 실행 (포그라운드). 터미널을 닫으면 함께 종료된다.
#
#   cd /workspace/streamlit_demo && ./start_server.sh
#
# VS Code·SSH 창을 닫아도 계속 돌게 하려면 이 스크립트를 직접 쓰지 말고
#   ./serverctl.sh start        (setsid nohup 으로 분리 실행 · PID/로그/워치독 관리)
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

# exec: 셸 대신 python(uvicorn) 이 PID 를 이어받아 SIGTERM 을 직접 받는다(정상 종료).
#       serverctl.sh 가 저장한 PID 도 그대로 uvicorn 프로세스가 된다.
# --host 0.0.0.0 : RunPod 프록시가 접근하려면 필수 (127.0.0.1 이면 외부 접근 불가)
# --workers 1    : GPU 모델이 워커 수만큼 중복 로드되지 않도록 (세션 상태도 프로세스 내부)
exec "${PYTHON}" -m uvicorn server.main:app \
  --host "${STT_HOST:-0.0.0.0}" \
  --port "${STT_BACKEND_PORT}" \
  --workers 1 \
  "$@"
