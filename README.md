# streamlit_demo
# ER 실시간 음성 전사 (STT)

응급실 진료 대화를 실시간으로 전사하는 시스템입니다.
**수음은 브라우저, 추론은 RunPod** 에서 합니다. 클라이언트에는 아무것도 설치하지 않습니다.

```
┌──────────── Mac / Client ─────────────┐
│                                       │
│ Microphone                            │
│      ↓ getUserMedia()                 │
│ Browser                               │
│      ↓ AudioWorklet (PCM16 / 16 kHz)  │
│ Streamlit audio_input / WebSocket     │
└────────────────┬──────────────────────┘
                 │  100~500 ms audio chunk
                 ↓
┌──────────── RunPod ───────────────────┐
│                                       │
│ Streamlit (8501) / Backend (8000)     │
│       ↓                               │
│ STT model (streaming ASR)             │
│       ↓                               │
│ Korean transcript (partial → final)   │
└────────────────┬──────────────────────┘
                 │  JSON partial / final
                 ↓
              Browser
```

| 경로 | 브라우저가 하는 일 | RunPod 이 하는 일 |
|---|---|---|
| **실시간 전사** | getUserMedia → AudioWorklet → WebSocket 으로 청크 전송 | streaming ASR → partial/final 을 WS 로 회신 |
| **파일 전사** | `st.audio_input` / 파일 업로드 | 전체 오디오를 한 번에 전사 |

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

RunPod 인스턴스에서 **두 프로세스**를 띄웁니다. 포트 8501(Streamlit)과
8000(STT 백엔드)을 모두 노출해 두세요.

```bash
# 1) STT 백엔드 — 브라우저 오디오를 받는 WebSocket 서버
python -m server.main --host 0.0.0.0 --port 8000

# 2) Streamlit UI
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

Mac 에서는 RunPod 이 준 Streamlit 주소만 열면 됩니다.

| 페이지 | 하는 일 |
|---|---|
| 실시간 전사 | 브라우저 마이크 → WebSocket → streaming ASR → partial/final, 라이브 지표 |
| 파일 전사 | `st.audio_input`·업로드 파일을 통째로 전사. 정답(reference) 만들기용 |
| 성능 비교 | 저장된 세션의 RTF/지연/WER/CER/의료용어 recall 비교 |

### 백엔드 주소

브라우저는 Streamlit(8501)과 **다른 호스트**의 백엔드(8000)로 오디오를 보냅니다.
RunPod 은 포트마다 호스트를 따로 주므로(`<podId>-<port>.proxy.runpod.net`)
기본값은 주소창에서 자동으로 유도합니다. 다르게 노출했다면 환경변수나
사이드바에서 직접 지정하세요.

| 환경변수 | 기본값 | 쓰는 쪽 |
|---|---|---|
| `STT_BACKEND_PORT` | `8000` | 주소 유도에 쓰는 백엔드 포트 |
| `STT_API_URL` | `http://127.0.0.1:8000` | Python → 백엔드 (상태 조회, 같은 인스턴스) |
| `STT_WS_URL` | (비움 → 자동 유도) | 브라우저 → 백엔드 (오디오 스트림) |

브라우저는 **HTTPS 또는 localhost** 에서만 마이크를 엽니다. RunPod 프록시는
HTTPS 라 그대로 되고, 로컬 개발은 `localhost` 라 그대로 됩니다.

### 로컬에서 돌려보기

```bash
python -m server.main --host 127.0.0.1 --port 8000
streamlit run streamlit_app.py --server.port 8501
# http://localhost:8501
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
│   ├── realtime.py           # 브라우저 마이크 실시간 전사
│   ├── file_stt.py           # audio_input / 업로드 파일 전사
│   └── metrics.py            # 엔진 성능 비교
│
├── web/                      # ── 브라우저에서 도는 코드 ──
│   ├── live_mic.py           # CCv2 컴포넌트 등록 + Python 래퍼
│   ├── live_mic.html         # 컴포넌트 마크업
│   ├── live_mic.css          # Streamlit 테마 토큰(--st-*) 기반 스타일
│   └── live_mic.js           # getUserMedia → AudioWorklet → WebSocket
│
├── stt/                      # ── 공용 코어 (백엔드·Streamlit 공유) ──
│   ├── config.py             # 오디오 규격 + 백엔드 주소 + StreamConfig
│   ├── session.py            # Session Manager (파이프라인 전체)
│   ├── audio/
│   │   ├── buffer.py         # sliding window + overlap
│   │   ├── recorder.py       # raw WAV 실시간 저장
│   │   ├── resampler.py      # PCM16 ↔ float32, 리샘플
│   │   └── vad.py            # 에너지 VAD + endpoint 검출
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
│   ├── main.py               # FastAPI 앱 + REST (상태 조회)
│   └── websocket.py          # /ws 엔드포인트 (오디오 인입 / 전사 회신)
│
├── recordings/               # 세션별 원본 WAV
├── transcripts/              # 세션별 전사 결과 (.txt / .json)
└── models/                   # sherpa-onnx 모델
```

### 마이크 캡처가 Streamlit 안에서 도는 이유

Streamlit Custom Component **v2** 는 iframe 이 아니라 앱 문서 안에서 실행됩니다.
따라서 마이크 권한을 iframe 으로 위임할 필요 없이 `getUserMedia()`,
`AudioWorklet`, `WebSocket` 을 그대로 쓸 수 있습니다. AudioWorklet 은 URL 로만
로드되므로 워클릿 소스를 Blob URL 로 만들어 넘깁니다(`web/live_mic.js`).

부분 전사는 초당 여러 번 갱신되므로 Python 으로 올리지 않고 브라우저에서 직접
그립니다. 세션이 끝날 때만 `setStateValue("result", …)` 로 요약을 한 번 올려
리런을 1회로 억제합니다.

---

## 4. 파이프라인

```
브라우저 마이크 (getUserMedia → AudioWorklet)
   ↓ PCM16 16 kHz mono, 100~500 ms chunk over WebSocket
StreamingSession.feed()          ← 논블로킹 (WS 핸들러에서 호출)
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
Event(partial / final / metrics) → WebSocket JSON → 브라우저 화면
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

### 실측 (Apple Silicon 로컬, CPU int8, 5.4 s 한국어 발화, window 5 s / overlap 1.5 s)

| 모델 | RTF | 첫 partial | final 지연 | 전사 |
|---|---|---|---|---|
| tiny | 0.10 | 1.75 s | 0.14 s | 오늘 아침부터 **개가** … (오인식) |
| base | 0.24 | 4.19 s → 2.1 s | 0.71 s | 환각 문장 삽입("고맙습니다") |
| small | 0.44 | 2.66 s | 0.72 s | 정답과 일치 |

CPU 에서는 `small` 까지가 실시간(RTF < 1)입니다. `medium` 이상은 GPU 를 쓰세요.
RunPod GPU 인스턴스에서는 `stt/config.py` 의 `default_device()` 가 CUDA 를 감지해
`device=cuda` / `compute_type=float16` 을 자동으로 고릅니다.

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
| 1-b. RunPod 배포 (브라우저 수음 + 서버 추론 분리) | 구현 완료 |
| 2. Whisper baseline (sliding window + overlap, partial) | 구현 완료 |
| 3. Transcript merge (stable/unstable, overlap dedup) | 구현 완료 · 회귀 테스트 |
| 4. Metrics logging (RTF, latency, 자원 사용) | 구현 완료 |
| 5. Zipformer backend (sherpa-onnx, native streaming, endpoint) | 코드 완료 · **모델 파일 필요** |
| 6. SenseVoice backend (VAD + chunk 추론) | 코드 완료 · **모델 파일 필요** |
| 7. 동일 의료 데이터셋 비교 | 비교 화면 완료 · 데이터셋 필요 |
