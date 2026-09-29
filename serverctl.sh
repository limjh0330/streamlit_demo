#!/bin/bash
# STT 서버 상시 구동 관리 — uvicorn 을 setsid nohup 으로 터미널 세션에서 분리해 실행한다.
# (Claude outputs/RunPod_서버_상시구동_가이드.md 의 serverctl.sh 템플릿을 이 프로젝트에 맞춘 것)
#
#   ./serverctl.sh start [--wait]   분리 기동 (--wait: READY 까지 기다림)
#   ./serverctl.sh stop             점검 모드로 전환 → 워치독 정지 → 서버 정상 종료(시간 초과 시 강제)
#   ./serverctl.sh restart [--wait] 종료 완료를 확인한 뒤 재기동
#   ./serverctl.sh status           PID/PPID/SID, 워치독, readiness, 외부 URL
#   ./serverctl.sh logs             서버 로그 따라 보기 (Ctrl+C 해도 서버는 계속 동작)
#   ./serverctl.sh watchdog-start   헬스체크 실패 시 자동 재기동하는 워치독 시작 (역시 분리 실행)
#   ./serverctl.sh watchdog-stop    워치독 정지
#
# VS Code·SSH 창을 닫아도 서버가 계속 도는 이유: setsid 로 새 세션을 만들어 터미널 세션과의
# 소속을 끊고(SIGHUP 이 오지 않음), nohup 으로 SIGHUP 을 한 번 더 막는다. 기동한 셸이 끝나면
# uvicorn 은 PID 1 에 입양된다(PPID=1, SID=자기 PID).
#
# ⚠ 기동 명령은 터미널에 직접 붙여 넣지 말고 반드시 이 스크립트로 실행한다(가이드 §6-2).
#   대화형 셸에서는 setsid 가 한 번 더 fork 해 $! 가 엉뚱한 PID 가 된다.
set -uo pipefail

# ============================ 설정 (환경변수로 덮어쓸 수 있음) ============================
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"   # 코드 루트 = 이 스크립트 위치
SELF="$APP_DIR/$(basename "${BASH_SOURCE[0]}")"

# 로그·PID 는 Pod 재생성에도 남는 /workspace 에 둔다(없으면 코드 루트의 run/).
if [[ -z "${STT_RUN_DIR:-}" ]]; then
  if [[ -d /workspace && -w /workspace ]]; then STT_RUN_DIR=/workspace; else STT_RUN_DIR="$APP_DIR/run"; fi
fi
mkdir -p "$STT_RUN_DIR"

# 영속 설정 파일: 여기에 적은 값은 수동 기동과 워치독 재기동 모두에 적용된다(가이드 §6-4).
#   예) /workspace/stt-server.env
#       STT_ENGINE=funasr_mlt_nano
#       STT_CONFIG={"save_wav":false}
ENV_FILE="${STT_ENV_FILE:-$STT_RUN_DIR/stt-server.env}"
if [[ -f "$ENV_FILE" ]]; then set -a; # shellcheck disable=SC1090
  source "$ENV_FILE"; set +a; fi

PORT="${STT_BACKEND_PORT:-8000}"                                   # RunPod 에 노출한 HTTP 포트
HEALTH_URL="${STT_HEALTH_URL:-http://127.0.0.1:$PORT/api/v1/stt/health/ready}"   # READY 면 200
LIVE_URL="http://127.0.0.1:$PORT/api/v1/stt/health"                # 프로세스 생존(항상 200)
STARTUP_WAIT="${STT_STARTUP_WAIT:-300}"      # 모델 로드·warm-up 최대 대기(초). Fun-ASR 첫 다운로드 고려
STOP_TIMEOUT="${STT_STOP_TIMEOUT:-30}"       # 정상 종료 대기(초). 넘기면 강제 종료
WATCH_INTERVAL="${STT_WATCH_INTERVAL:-30}"   # 워치독 헬스체크 간격(초)
WATCH_FAILS="${STT_WATCH_FAILS:-3}"          # 연속 실패 몇 번이면 재기동
LOG_MAX_MB="${STT_LOG_MAX_MB:-100}"          # 기동 시 로그가 이보다 크면 .1 로 넘긴다

LOG="$STT_RUN_DIR/stt-server.log"
PIDF="$STT_RUN_DIR/stt-server.pid"
WLOG="$STT_RUN_DIR/stt-watchdog.log"
WPIDF="$STT_RUN_DIR/stt-watchdog.pid"
MAINT="$STT_RUN_DIR/.stt-maintenance"        # 있으면 워치독이 재기동하지 않는다

# 파이썬은 절대경로로 고정한다(워치독은 activate 된 셸 환경을 물려받지 않음, 가이드 §6-3).
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$APP_DIR/.venv/bin/python" ]]; then PYTHON="$APP_DIR/.venv/bin/python"
  else PYTHON="$(command -v python || command -v python3)"; fi
fi
export PYTHON STT_BACKEND_PORT="$PORT"
# ========================================================================================

ts()      { date '+%F %T'; }
pid_of()  { [[ -f "$1" ]] && cat "$1"; }
alive()   { local p; p="$(pid_of "$1")" && [[ -n "$p" ]] && kill -0 "$p" 2>/dev/null; }
code_of() { curl -s -o /dev/null -w '%{http_code}' -m 5 "$1" 2>/dev/null; }
healthy() { [[ "$(code_of "$HEALTH_URL")" == "200" ]]; }

# 저장된 PID 가 정말 이 서버인지(재사용된 PID 를 죽이지 않도록)
is_ours() {
  local cmd; cmd="$(ps -o command= -p "$1" 2>/dev/null)"
  [[ "$cmd" == *"server.main"* || "$cmd" == *"start_server.sh"* || "$cmd" == *"serverctl.sh"* ]]
}

# setsid 로 분리 실행. setsid 명령이 없으면(macOS 등) 같은 일을 하는 파이썬으로 대신한다.
# `detach ... &` 는 백그라운드 서브셸에서 돌므로 반드시 exec 으로 그 서브셸을 대체한다.
# 그래야 $! (서브셸 PID) = nohup = setsid = start_server.sh = uvicorn 이 모두 같은 PID 가 된다.
detach() {
  if command -v setsid >/dev/null 2>&1; then
    exec setsid nohup "$@"
  else
    exec nohup "$PYTHON" -c 'import os, sys; os.setsid(); os.execv(sys.argv[1], sys.argv[1:])' "$@"
  fi
}

sid_of() { "$PYTHON" -c 'import os, sys; print(os.getsid(int(sys.argv[1])))' "$1" 2>/dev/null; }

rotate_log() {
  [[ -f "$LOG" ]] || return 0
  local size_mb=$(( $(wc -c <"$LOG") / 1024 / 1024 ))
  if (( size_mb >= LOG_MAX_MB )); then mv -f "$LOG" "$LOG.1"; echo "로그가 ${size_mb} MB 라 $LOG.1 로 넘겼습니다"; fi
}

wait_ready() {
  echo "READY 대기 중 (최대 ${STARTUP_WAIT}s) — $HEALTH_URL"
  local start=$SECONDS
  while (( SECONDS - start < STARTUP_WAIT )); do
    if ! alive "$PIDF"; then
      echo "서버 프로세스가 종료됐습니다. 로그 마지막 부분:"; tail -n 30 "$LOG"; return 1
    fi
    if healthy; then echo "READY ($(( SECONDS - start ))s)"; return 0; fi
    sleep 2
  done
  echo "${STARTUP_WAIT}s 안에 READY 가 되지 않았습니다. 현재 상태:"
  curl -s -m 5 "$HEALTH_URL"; echo; return 1
}

start() {
  rm -f "$MAINT"
  if alive "$PIDF"; then echo "이미 실행 중 (PID $(pid_of "$PIDF"))"; return 0; fi
  if [[ "$(code_of "$LIVE_URL")" == "200" ]]; then
    echo "포트 $PORT 에 이미 다른 STT 서버가 떠 있습니다(PID 파일 없음)."
    echo "포그라운드로 띄운 ./start_server.sh 가 있다면 먼저 종료하세요."; return 1
  fi
  cd "$APP_DIR" || { echo "APP_DIR 이 없습니다: $APP_DIR"; return 1; }
  rotate_log
  echo "$(ts) ===== serverctl start (port=$PORT) =====" >>"$LOG"
  detach "$APP_DIR/start_server.sh" </dev/null >>"$LOG" 2>&1 &
  echo $! >"$PIDF"
  disown "$!" 2>/dev/null || true            # 이 셸의 작업 목록에서 뺀다(종료 알림·대기 없음)
  echo "기동 시작 (PID $(pid_of "$PIDF")) — 로그: $LOG"
  if [[ "${1:-}" == "--wait" ]]; then wait_ready; fi
}

# 종료 신호(SIGTERM)를 보내고 실제로 끝날 때까지 기다린다. 끝나지 않으면 강제 종료한다.
kill_server() {
  if ! alive "$PIDF"; then rm -f "$PIDF"; return 0; fi
  local pid; pid="$(pid_of "$PIDF")"
  if ! is_ours "$pid"; then
    echo "PID $pid 는 STT 서버가 아닙니다(재사용된 PID). PID 파일만 지웁니다."; rm -f "$PIDF"; return 0
  fi
  kill "$pid" 2>/dev/null
  for _ in $(seq "$STOP_TIMEOUT"); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  if kill -0 "$pid" 2>/dev/null; then
    echo "정상 종료되지 않아 강제 종료합니다 (PID $pid)"
    kill -9 "$pid" 2>/dev/null; sleep 1
  fi
  rm -f "$PIDF"
}

stop()    { touch "$MAINT"; watchdog_stop; kill_server; echo "서버 종료"; }
restart() { touch "$MAINT"; kill_server; start "${1:-}"; }

status() {
  if alive "$PIDF"; then
    local pid; pid="$(pid_of "$PIDF")"
    local ppid; ppid="$(ps -o ppid= -p "$pid" | tr -d ' ')"
    local sid; sid="$(sid_of "$pid")"
    local mark="분리됨"; [[ "$sid" == "$pid" ]] || mark="⚠ 터미널 세션에 속해 있음"
    echo "서버   : 실행 중 PID=$pid PPID=$ppid SID=$sid ($mark)"
  else
    echo "서버   : 중지됨"
  fi
  if alive "$WPIDF"; then echo "워치독 : 실행 중 PID=$(pid_of "$WPIDF")"; else echo "워치독 : 중지됨"; fi
  [[ -f "$MAINT" ]] && echo "상태   : 점검 모드 (워치독 재기동 보류)"
  echo "로그   : $LOG"
  [[ -f "$ENV_FILE" ]] && echo "설정   : $ENV_FILE"
  if [[ -n "${RUNPOD_POD_ID:-}" ]]; then
    echo "외부   : https://${RUNPOD_POD_ID}-${PORT}.proxy.runpod.net/api/v1/stt/health/ready"
    echo "         wss://${RUNPOD_POD_ID}-${PORT}.proxy.runpod.net/ws/v1/stt/stream"
  fi
  local out
  if out="$(curl -s -m 5 "$HEALTH_URL" -w '\nHTTP %{http_code}')"; then echo "$out"
  else echo "헬스   : 응답 없음 ($HEALTH_URL)"; fi
}

# WATCH_INTERVAL 간격 헬스체크. WATCH_FAILS 회 연속 실패하면 (살아 있는 프로세스까지 정리한 뒤) 재기동.
# 1시간에 5번 넘게 재기동하면 자동 복구를 멈추고 수동 확인을 기다린다.
watchdog_loop() {
  local fail=0 restarts=0 window=$(( $(date +%s) + 3600 ))
  echo "$(ts) 워치독 시작 (interval=${WATCH_INTERVAL}s fails=${WATCH_FAILS} url=$HEALTH_URL)"
  # 막 기동한 서버가 모델을 올리는 동안은 실패로 세지 않는다
  if alive "$PIDF" && ! healthy; then sleep "$STARTUP_WAIT" & wait $!; fi
  while true; do
    sleep "$WATCH_INTERVAL"
    [[ -f "$MAINT" ]] && { fail=0; continue; }
    if healthy; then fail=0; continue; fi
    fail=$((fail + 1))
    echo "$(ts) 헬스체크 실패 ${fail}/${WATCH_FAILS} (HTTP $(code_of "$HEALTH_URL"))"
    (( fail < WATCH_FAILS )) && continue
    (( $(date +%s) > window )) && { restarts=0; window=$(( $(date +%s) + 3600 )); }
    if (( restarts >= 5 )); then
      echo "$(ts) 1시간 내 5회 재기동 — 자동 복구 중단, 수동 확인 필요 (로그: $LOG)"
      sleep 600; continue
    fi
    restarts=$((restarts + 1))
    echo "$(ts) 서버 재기동 (${restarts}회차)"
    kill_server                 # 응답 없이 살아만 있는 프로세스도 정리
    start
    sleep "$STARTUP_WAIT"       # 모델 로드 동안 기다린다
    fail=0
  done
}

watchdog_start() {
  if alive "$WPIDF"; then echo "워치독 이미 실행 중 (PID $(pid_of "$WPIDF"))"; return 0; fi
  detach "$SELF" _loop </dev/null >>"$WLOG" 2>&1 &
  echo $! >"$WPIDF"
  disown "$!" 2>/dev/null || true
  echo "워치독 시작 (PID $(pid_of "$WPIDF")) — 로그: $WLOG"
}

watchdog_stop() {
  if alive "$WPIDF"; then
    local wpid; wpid="$(pid_of "$WPIDF")"
    pkill -P "$wpid" 2>/dev/null            # 대기 중인 sleep 까지 정리
    kill "$wpid" 2>/dev/null && echo "워치독 종료"
  fi
  rm -f "$WPIDF"
}

case "${1:-}" in
  start)          start "${2:-}" ;;
  stop)           stop ;;
  restart)        restart "${2:-}" ;;
  status)         status ;;
  logs)           exec tail -n 100 -f "$LOG" ;;
  watchdog-start) watchdog_start ;;
  watchdog-stop)  watchdog_stop ;;
  _loop)          watchdog_loop ;;
  *) echo "사용법: $0 {start [--wait]|stop|restart [--wait]|status|logs|watchdog-start|watchdog-stop}"; exit 2 ;;
esac
