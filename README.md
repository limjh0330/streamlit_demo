# streamlit_demo
# ER 실시간 음성 전사 (STT)

응급실 진료 대화를 실시간으로 전사하는 시스템입니다.
같은 코어(`stt/`) 위에 **두 개의 입력 경로**가 올라가 있습니다.

| 경로 | 마이크 위치 | 진입점 |
|---|---|---|
| Streamlit 데모 | 앱이 실행 중인 **컴퓨터** (sounddevice) | `streamlit run streamlit_app.py` |
| STT 서버 | **태블릿/브라우저** (getUserMedia + AudioWorklet) | `python -m server.main` |

ASR 백엔드는 어댑터로 분리되어 있어 같은 오디오로 세 엔진을 비교할 수 있습니다.

- **Whisper** (`faster-whisper`) — 높은 범용성, 안정적인 한국어 baseline. sliding window 방식.
- **Zipformer** (`sherpa-onnx`) — 진짜 streaming ASR. 프레임 단위 디코딩 + 내장 endpoint 검출.
- **SenseVoice** (`sherpa-onnx`) — 빠른 non-autoregressive ASR. VAD 로 자른 chunk 단위 추론.

---

## 1. 설치

anaconda base 환경에는 NumPy 1.x 로 빌드된 패키지가 섞여 있어 충돌이 납니다.
**프로젝트 전용 가상환경**을 권장합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Zipformer / SenseVoice 까지 쓰려면:

```bash
pip install sherpa-onnx
# models/zipformer/, models/sensevoice/ 에 모델 파일 배치 (models/README.md 참고)
```

## 2. 실행

### Streamlit 데모 (로컬 마이크)

```bash
streamlit run streamlit_app.py
```

| 페이지 | 하는 일 |
|---|---|
| 실시간 전사 | 로컬 마이크 → sliding window → partial/final 전사, 라이브 지표 |
| 파일 전사 | 업로드·녹음 파일을 통째로 전사. 정답(reference) 만들기용 |
| 태블릿 · 브라우저 | STT 서버 실행법, 접속 URL, HTTPS 인증서 생성 안내 |
| 성능 비교 | 저장된 세션의 RTF/지연/WER/CER/의료용어 recall 비교 |

### STT 서버 (태블릿 마이크)

```bash
python -m server.main --host 0.0.0.0 --port 8000
```

브라우저는 **HTTPS 또는 localhost** 에서만 마이크를 허용합니다.
태블릿에서 IP 로 접속하려면 인증서가 필요합니다.

```bash
openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
  -keyout key.pem -out cert.pem -subj "/CN=192.168.0.10" \
  -addext "subjectAltName=IP:192.168.0.10"

python -m server.main --host 0.0.0.0 --port 8000 --certfile cert.pem --keyfile key.pem
```

### 테스트

```bash
python -m tests.test_pipeline     # pytest 없이도 실행됨
pytest tests/
```

---

## 3. 구조

```
streamlit_demo/
├── streamlit_app.py          # Streamlit 진입점 (st.navigation)
├── app_pages/
│   ├── realtime.py           # 로컬 마이크 실시간 전사
│   ├── file_stt.py           # 파일/녹음 전사 (기존 app.py 계승)
│   ├── browser.py            # 태블릿 경로 안내
│   └── metrics.py            # 엔진 성능 비교
│
├── stt/                      # ── 공용 코어 (서버·Streamlit 공유) ──
│   ├── config.py             # 오디오 규격 + StreamConfig
│   ├── session.py            # Session Manager (파이프라인 전체)
│   ├── audio/
│   │   ├── buffer.py         # sliding window + overlap
│   │   ├── recorder.py       # raw WAV 실시간 저장
│   │   ├── resampler.py      # PCM16 ↔ float32, 리샘플
│   │   ├── vad.py            # 에너지 VAD + endpoint 검출
│   │   └── mic.py            # sounddevice InputStream
│   ├── asr/
│   │   ├── base.py           # ASREngine 인터페이스 + 팩토리
│   │   ├── whisper.py        # faster-whisper
│   │   ├── zipformer.py      # sherpa-onnx streaming
│   │   └── sensevoice.py     # sherpa-onnx offline
│   ├── transcript/
│   │   ├── merger.py         # stable prefix / unstable suffix / overlap dedup
│   │   └── medical_terms.py  # ER 용어 사전 + 후처리
│   └── metrics/
│       ├── latency.py        # RTF, first partial, final latency, CPU/메모리
│       └── evaluator.py      # WER / CER / 의료용어 recall
│
├── server/
│   ├── main.py               # FastAPI 앱 + REST + 정적 파일
│   └── websocket.py          # /ws 엔드포인트
│
├── web/
│   ├── index.html            # 브라우저 UI
│   ├── recorder.js           # getUserMedia → WebSocket
│   └── pcm-worklet.js        # AudioWorklet: PCM16 / 16 kHz 변환
│
├── recordings/               # 세션별 원본 WAV
├── transcripts/              # 세션별 전사 결과 (.txt / .json)
├── models/                   # sherpa-onnx 모델
├── examples/                 # sounddevice 참고 예제 (원본)
└── app.py                    # 초기 단일 파일 데모 (streamlit_app.py 로 대체됨)
```

---

## 4. 파이프라인

```
마이크 / 브라우저
   ↓ PCM16 16 kHz mono, 100 ms block
StreamingSession.feed()          ← 논블로킹 (PortAudio 콜백 / WS 핸들러)
   ↓ Queue
워커 스레드
   ├→ WavRecorder                → recordings/<session>.wav
   ├→ EndpointDetector (VAD)     → 발화 시작 / 종료 판정
   ├→ SlidingWindowBuffer        → 5 s window, 1.5 s overlap
   ↓
ASREngine.transcribe(audio, t0)  → 절대 시각이 붙은 단어열
   ↓
TranscriptMerger                 → stable(확정) + unstable(부분)
   ↓
Event(partial / final / metrics) → Streamlit fragment · WebSocket JSON
```

### 중복 단어 제거 (핵심)

윈도우가 겹치므로 같은 단어가 여러 번 나옵니다. `TranscriptMerger` 는 3중으로 거릅니다.

1. **시각 기준** — 확정 시각 이전의 단어는 버린다.
2. **텍스트 기준** — 확정된 꼬리와 새 가설의 머리가 겹치면 잘라낸다
   (타임스탬프가 윈도우마다 수십 ms 흔들리므로 1) 만으로는 부족).
3. **LocalAgreement-2** — 두 번 연속 같게 나온 접두사만 확정하고,
   나머지 꼬리는 `unstable` 로 두어 다음 윈도우에서 다시 판단한다.

윈도우 경계에 걸쳐 다음 가설이 이어받지 못하는 단어는 **버리지 않고 그 시점에 확정**합니다.
이 부분이 누락·중복의 주된 원인이라 `tests/test_pipeline.py` 에서
window/overlap/지터 조합을 훑는 회귀 테스트로 고정해 두었습니다.

---

## 5. 지표

| 지표 | 위치 | 의미 |
|---|---|---|
| RTF | `metrics/latency.py` | 추론시간 / 오디오길이. **1.0 미만**이어야 실시간 |
| First partial latency | 〃 | 발화 시작 → 첫 부분 전사 |
| Final transcript latency | 〃 | endpoint → 발화 확정 |
| Partial revision rate | `transcript/merger.py` | 부분 전사가 뒤집힌 비율 |
| CPU / 메모리 / GPU | `metrics/latency.py` | psutil, torch |
| WER / CER | `metrics/evaluator.py` | 한국어는 CER 이 더 신뢰할 만함 |
| 의료용어 recall | 〃 | 정답 속 도메인 용어를 얼마나 살렸는지 |

### 실측 (Apple Silicon, CPU int8, 5.4 s 한국어 발화, window 5 s / overlap 1.5 s)

| 모델 | RTF | 첫 partial | final 지연 | 전사 |
|---|---|---|---|---|
| tiny | 0.10 | 1.75 s | 0.14 s | 오늘 아침부터 **개가** … (오인식) |
| base | 0.24 | 4.19 s → 2.1 s | 0.71 s | 환각 문장 삽입("고맙습니다") |
| small | 0.44 | 2.66 s | 0.72 s | 정답과 일치 |

CPU 에서는 `small` 까지가 실시간(RTF < 1)입니다. `medium` 이상은 GPU 를 쓰세요.

첫 partial 지연은 `first_hop_sec`(기본 1.5 s)로 조절합니다. 발화 시작 직후
첫 윈도우만 짧게 끊어 내보내고, 이후에는 `window - overlap`(=3.5 s) 간격으로
갱신합니다. 값을 줄이면 반응이 빨라지지만 추론 횟수와 CPU 사용이 늘어납니다.

### 튜닝하며 알게 된 것

- **무음에 `initial_prompt` 를 붙여 디코딩하면 안 됩니다.** Whisper 가 프롬프트를
  이어 쓰려고 헛돌아 `small` 기준 한 번에 **10 초** 넘게 걸립니다(0.9 s → 10.7 s).
  그래서 `WhisperEngine.warmup()` 은 프롬프트 없이 돌리고, 세션은
  노이즈 플로어 수준인 윈도우의 추론을 아예 건너뜁니다(`_is_silent`).
  이 처리로 `small` 의 warmup 13.3 s → 0.9 s, final 지연 9.1 s → 0.8 s 가 됐습니다.
- 무음 구간 추론을 건너뛰면 없는 말을 지어내는 환각도 함께 줄어듭니다.
- `initial_prompt` 자체는 실제 음성에서 +0.2 s 수준이라 유지할 만합니다.

---

## 6. WebSocket 프로토콜

```jsonc
// client → server
{"type":"start","engine":"whisper","model_size":"small","language":"ko",
 "sample_rate":16000,"window_sec":5,"overlap_sec":1.5,"silence_sec":0.7}
<binary>                       // PCM16 little-endian mono
{"type":"stop"}

// server → client
{"type":"ready",   "session_id":"...", "engine":"whisper", "config":{...}}
{"type":"partial", "stable":"...", "partial":"...", "committed":"..."}
{"type":"final",   "text":"...", "start":0.0, "end":3.2, "index":0}
{"type":"metrics", "rtf_mean":0.42, "first_partial_ms":1830, ...}
{"type":"closed",  "summary":{...}, "text":"...", "transcript":"...", "wav":"..."}
```

## 7. 개발 순서 대비 현황

| 단계 | 상태 |
|---|---|
| 1. Web Audio + WebSocket (PCM16 / 16 kHz / WAV 저장) | 구현 완료 |
| 2. Whisper baseline (sliding window + overlap, partial) | 구현 완료 |
| 3. Transcript merge (stable/unstable, overlap dedup) | 구현 완료 · 회귀 테스트 |
| 4. Metrics logging (RTF, latency, 자원 사용) | 구현 완료 |
| 5. Zipformer backend (sherpa-onnx, native streaming, endpoint) | 코드 완료 · **모델 파일 필요** |
| 6. SenseVoice backend (VAD + chunk 추론) | 코드 완료 · **모델 파일 필요** |
| 7. 동일 의료 데이터셋 비교 | 비교 화면 완료 · 데이터셋 필요 |
