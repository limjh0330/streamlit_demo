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

ASR 백엔드는 어댑터로 분리되어 있어 같은 오디오로 네 엔진을 비교할 수 있습니다.
모두 한국어를 지원하며, 엔진마다 오디오를 넣는 방식이 다릅니다.

| 엔진 | 패키지 | 오디오 공급 방식 | 특징 |
|---|---|---|---|
| **Whisper** | `faster-whisper` | 5 s sliding window + 1.5 s overlap | 정확도 baseline. autoregressive 라 느림 |
| **Zipformer** | `sherpa-onnx` | 프레임 단위 `accept_waveform` | 진짜 streaming. 내장 endpoint 검출, 첫 partial 이 가장 빠름 |
| **SenseVoice** | `sherpa-onnx` | 발화 전체를 0.8 s 마다 재인식 | non-autoregressive. RTF 0.02 로 매우 빠름 |
| **Fun-ASR-MLT-Nano-2512** | `funasr` | 발화 전체를 0.8 s 마다 재인식 | 800M·31개 언어 다국어 ASR. 현재 타임스탬프 미지원 |

SenseVoice 는 추론이 워낙 빨라 윈도우를 잘라 넣는 대신 **발화 전체를 매번 다시
인식**합니다(`decodes_full_utterance`). 잘린 오디오를 인식할 때 생기는 오류와
윈도우 간 중복이 함께 사라집니다.

---

## 1. 설치

anaconda base 환경에는 NumPy 1.x 로 빌드된 패키지가 섞여 있어 충돌이 납니다.
**프로젝트 전용 가상환경**을 권장합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Whisper 모델은 최초 실행 시 자동으로 받습니다. Zipformer / SenseVoice 는
모델 파일을 먼저 내려받아야 합니다.

```bash
python -m scripts.fetch_models        # 약 360 MB (int8)
```

Fun-ASR-MLT-Nano-2512는 `funasr>=1.4.1` 설치 후 첫 전사 때 Hugging Face
캐시에 자동으로 내려받습니다. 모델이 약 800M 파라미터이므로 GPU 실행을 권장합니다.

| 디렉터리 | 릴리스 |
|---|---|
| `models/zipformer/` | `sherpa-onnx-streaming-zipformer-korean-2024-06-16` (한국어 전용) |
| `models/sensevoice/` | `sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17` (한국어 포함 다국어) |

RunPod GPU 에서는 sherpa-onnx 도 CUDA 휠로 바꾸고 fp32 가중치를 받으세요.
CPU 휠에 `provider="cuda"` 를 주면 경고만 내고 CPU 로 떨어집니다.

```bash
pip install sherpa-onnx -f https://k2-fsa.github.io/sherpa/onnx/cuda.html
python -m scripts.fetch_models --keep all --force
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

### 엔진 비교 (CLI)

Streamlit 없이 같은 오디오를 네 엔진에 통과시켜 지표를 뽑습니다. 결과는
`transcripts/` 에 저장되어 **성능 비교** 페이지에 그대로 나타납니다.

```bash
python -m scripts.benchmark recordings/sample.wav --realtime --reference ref.txt
```

`--realtime` 은 오디오를 실제 속도로 흘려 넣습니다. **지연 지표(first partial /
final latency)는 이때만 의미가 있습니다** — 한 번에 밀어 넣으면 큐에 쌓인 채로
측정돼 실제보다 훨씬 작게 나옵니다.

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
│   │   ├── sensevoice.py     # sherpa-onnx offline
│   │   └── funasr_mlt_nano.py # Fun-ASR-MLT-Nano-2512 (FunASR)
│   ├── transcript/
│   │   ├── merger.py         # stable prefix / unstable suffix / overlap dedup
│   │   └── medical_terms.py  # ER 용어 사전 + 후처리
│   └── metrics/
│       ├── latency.py        # RTF, first partial, final latency, CPU/메모리
│       └── evaluator.py      # WER / CER / 의료용어 recall
│
├── server/
│   ├── main.py               # FastAPI 앱 + REST (상태 조회)
│   ├── websocket.py          # /ws 엔드포인트 (브라우저) + 세션 공용 헬퍼
│   └── backend_ws.py         # /ws/stt 엔드포인트 (External Backend 어댑터)
│
├── scripts/
│   ├── fetch_models.py       # Zipformer / SenseVoice 모델 다운로드
│   └── benchmark.py          # 같은 오디오로 엔진 비교 (CLI)
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

### 엔진별 병합 경로

엔진이 주는 정보가 달라서 병합 전략도 셋으로 갈립니다.

| 엔진 | 병합기 진입점 | 이유 |
|---|---|---|
| Whisper, Zipformer | `update(words)` | 단어 타임스탬프가 있어 시각 기준 정렬이 가능 |
| SenseVoice, Fun-ASR-MLT-Nano | `replace_text(text)` | 매번 발화 전체를 다시 인식하므로 최신 가설로 교체 |
| (타임스탬프 없는 윈도우형) | `update_text(text)` | 텍스트 겹침만 제거하고 한 윈도우 늦게 확정 |

여기서 걸렸던 것들 — 모두 회귀 테스트로 고정했습니다.

- **네이티브 스트리밍 엔진은 같은 가설을 반복해서 준다.** Zipformer 는 100 ms
  블록마다 결과를 주는데 대부분 직전과 같습니다. 그대로 병합기에 넣으면
  LocalAgreement-2 의 "두 번 연속 같았다" 가 저절로 성립해, 아직 자라는 중인
  어절이 확정돼 버립니다("척" 확정 → 다음 블록에서 "척할려고" 가 또 확정 →
  `척 척할려고`). **가설이 실제로 바뀌었을 때만** 넘깁니다.
- **BPE 토큰을 그대로 이으면 띄어쓰기가 사라진다.** sherpa-onnx 의
  `get_result()` 는 공백이 지워진 문자열을 줍니다(`걔는괜찮은척하려구`).
  `get_result_all()` 로 토큰열을 받아 선행 공백(= `▁`)을 어절 경계로 삼아
  다시 묶습니다.
- **endpoint 직후에는 마지막 어절이 잘린다.** 디코더 안에 토큰이 남아 있어
  "같았다" 가 "같았" 로 끝납니다. 0.3 s 무음을 흘려 넣어 끌어냅니다.
- **발화 전체를 재인식하는 엔진에 겹침 제거를 쓰면 중복된다.** 잘린 오디오의
  부정확한 인식("괜찮찮은 척 하려")이 뒤의 정확한 인식과 텍스트가 달라
  `_strip_overlap` 을 빠져나옵니다. 교체 경로(`replace_text`)로 분리했습니다.

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

### 엔진 비교 (Apple Silicon CPU, 6.9 s 한국어 진료 발화, `--realtime`)

`python -m scripts.benchmark recordings/... --realtime --reference ref.txt`

| 엔진 | RTF | 첫 partial | final 지연 | WER | CER | 의료용어 recall |
|---|---|---|---|---|---|---|
| Whisper `small` | 0.45 | 2638 ms | 783 ms | **0.00** | **0.00** | 1.00 |
| Zipformer (streaming) | **0.06** | **537 ms** | 17 ms | 0.73 | 0.40 | 0.00 |
| SenseVoice | 0.03 | 973 ms | 64 ms | 0.18 | 0.03 | 1.00 |

- **Zipformer** 는 첫 partial 이 Whisper 의 1/5 로 압도적으로 빠릅니다. 다만 받아
  쓴 모델은 일상 대화(KsponSpeech) 로 학습된 것이라 의료 용어에서 많이 틀립니다
  ("배가 아팠습니다" → "걔가 했습니다"). 의료 도메인에 쓰려면 fine-tuning 이 필요합니다.
- **SenseVoice** 가 지연·정확도 균형이 가장 좋습니다. CER 0.03 으로 Whisper 에
  근접하면서 RTF 는 1/15 입니다. 다만 단어 타임스탬프가 없어 발화 단위 시각만 나옵니다.
- **Whisper** 는 정확도 기준점이지만 첫 partial 이 2.6 s 로 가장 느립니다.

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

브라우저/Streamlit 용 `/ws` 규격입니다. External Backend 는 아래
[Backend WebSocket API](#backend-websocket-api) 의 `/ws/stt` 를 쓰세요.

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

## Backend WebSocket API

External Backend 가 STT 서버에 붙는 인터페이스입니다. STT 서버는 **Audio →
Transcript** 까지만 책임지며 conversation/turn 번호, KTAS 판단, LLM 호출, 화자
역할(환자/의료진) 추정은 하지 않습니다. 연결 하나 = 오디오 스트림 하나이고,
Backend 가 연결과 예진 세션(UUID)을 스스로 매핑합니다.

### Endpoint

```
ws://<host>:8000/ws/stt
wss://<podId>-8000.proxy.runpod.net/ws/stt       # RunPod
```

선택: `?triage_session_id=<uuid>` 를 붙이면 서버 **로그에만** 함께 남습니다
(STT 내부 세션 ID 와는 별개이며 응답에는 포함되지 않습니다).

### Backend → STT

**Binary** — 오디오

| 항목 | 값 |
|---|---|
| format | raw PCM, signed **PCM16**, **little-endian** |
| sample rate | **16 kHz** |
| channels | **mono** |
| chunk | **100 ms** = 1,600 samples = **3,200 bytes** |

- 3200 B 가 아닌 프레임도 받습니다(경고 로그). 홀수 길이로 샘플이 쪼개지면
  남는 바이트를 다음 프레임에 이어 붙입니다.
- 빈 프레임은 무시합니다. 1 s(32,000 B)를 넘는 프레임은 버리고
  `{"type":"error","message":"Invalid audio frame"}` 를 보냅니다. 연결은 유지됩니다.

**Close** — 오디오 전송 종료 (text frame)

```
close
```

`close` 를 받으면 남은 오디오 → ASR 디코더 → TranscriptMerger 를 flush 하고,
남은 final 을 모두 보낸 뒤 **마지막에** `done` 을 보내고 연결을 닫습니다(1000).

### STT → Backend

**Partial** — 현재 발화의 중간 결과. 화면 표시용이며 DB 저장용이 아닙니다.
직전과 같은 텍스트는 다시 보내지 않습니다.

```json
{"type": "transcript", "text": "가슴이", "is_final": false}
```

**Final** — 발화 하나의 확정 결과. Backend 의 conversation row 하나에 해당합니다.

```json
{"type": "transcript", "text": "가슴이 답답해요", "is_final": true,
 "start_time": 12.4, "end_time": 15.6}
```

`start_time` / `end_time` 은 선택 필드로, **이 연결에 들어온 오디오 기준 초**입니다
(`STT_TIMESTAMPS=0` 이면 빠집니다). SenseVoice/Fun-ASR 은 발화 단위 시각입니다.

**Error**

```json
{"type": "error", "message": "..."}
```

| message | 상황 | 이후 |
|---|---|---|
| `STT engine initialization failed` | 모델 로드 실패 | 연결 종료 (1011) |
| `STT inference failed` | 추론 중 오류 (초당 최대 1회) | 계속 동작 |
| `Invalid audio frame` | 1 s 초과 프레임 | 해당 프레임만 버림 |
| `Unknown control message` | `close` 가 아닌 text frame | 계속 동작 |
| `Internal STT server error` | 예기치 못한 서버 오류 | 연결 종료 (1011) |

traceback·파일 경로·예외 내용은 서버 로그에만 남습니다.

**Done** — flush 가 끝났다는 신호. 항상 마지막 메시지입니다.

```json
{"type": "done"}
```

### 흐름 예시

```
Backend                                STT
  ── connect /ws/stt ─────────────────▶
  ── <3200 B> <3200 B> <3200 B> … ────▶
  ◀── {"type":"transcript","text":"가슴이","is_final":false}
  ◀── {"type":"transcript","text":"가슴이 답답해요","is_final":false}
  ◀── {"type":"transcript","text":"가슴이 답답해요","is_final":true,…}
  ── <3200 B> … ──────────────────────▶
  ── "close" ─────────────────────────▶
  ◀── {"type":"transcript","text":"(남은 발화)","is_final":true,…}
  ◀── {"type":"done"}
  ◀── close(1000)
```

Backend 가 `close` 없이 끊으면 STT 는 flush·WAV/전사 저장·세션 정리만 하고
아무것도 보내지 않습니다.

### 서버 설정

엔진·윈도우 같은 ASR 설정은 Backend 가 보내지 않고 STT 서버 환경변수로 정합니다.
엔진을 바꿔도 Backend 프로토콜은 그대로입니다.

| 환경변수 | 예 | 의미 |
|---|---|---|
| `STT_ENGINE` | `sensevoice` | `whisper` · `zipformer` · `sensevoice` · `funasr_mlt_nano` |
| `STT_MODEL_SIZE` | `small` | Whisper 모델 크기 |
| `STT_CONFIG` | `{"silence_sec":0.6,"save_wav":false}` | `StreamConfig` 필드 덮어쓰기 (JSON) |
| `STT_TIMESTAMPS` | `1` | final 에 `start_time`/`end_time` 포함 (기본 1) |

```bash
STT_ENGINE=sensevoice python -m server.main --host 0.0.0.0 --port 8000
```

연결마다 엔진을 새로 로드하므로(브라우저 `/ws` 와 같음) 첫 오디오가 모델 로드
시간만큼 늦게 처리될 수 있습니다. 그동안 보낸 프레임은 버려지지 않고 처리됩니다.

## 7. 개발 순서 대비 현황

| 단계 | 상태 |
|---|---|
| 1. Web Audio + WebSocket (PCM16 / 16 kHz / WAV 저장) | 구현 완료 |
| 1-b. RunPod 배포 (브라우저 수음 + 서버 추론 분리) | 구현 완료 |
| 2. Whisper baseline (sliding window + overlap, partial) | 구현 완료 |
| 3. Transcript merge (stable/unstable, overlap dedup) | 구현 완료 · 회귀 테스트 |
| 4. Metrics logging (RTF, latency, 자원 사용) | 구현 완료 |
| 5. Zipformer backend (sherpa-onnx, native streaming, endpoint) | 구현 완료 · 한국어 모델 연결 |
| 6. SenseVoice backend (발화 전체 재인식) | 구현 완료 · 한국어 모델 연결 |

## 8. API 사용 방법

STT 서버를 다른 서비스에서 API 로 쓰는 방법입니다. 실시간 전사는 WebSocket
`/ws/stt`, 상태·결과 조회는 REST `/api/*` 를 씁니다. 메시지 규격의 세부 사항은
[Backend WebSocket API](#backend-websocket-api) 를 참고하세요.

### 8.1 서버 실행

```bash
source .venv/bin/activate
STT_ENGINE=sensevoice python -m server.main --host 0.0.0.0 --port 8000
```

엔진은 서버 쪽 환경변수로 고릅니다(`STT_ENGINE`, `STT_MODEL_SIZE`, `STT_CONFIG`).
클라이언트는 엔진을 몰라도 되고, 엔진을 바꿔도 클라이언트 코드는 그대로입니다.

### 8.2 엔드포인트

| 종류 | 경로 | 용도 |
|---|---|---|
| WebSocket | `/ws/stt` | **외부 서비스용 실시간 전사** (PCM16 → transcript) |
| WebSocket | `/ws` | Streamlit/브라우저 UI 전용 (6장 규격) — 외부 연동에는 쓰지 마세요 |
| GET | `/api/health` | 서버 상태, 활성 세션 수 |
| GET | `/api/engines` | 엔진별 설치·모델 준비 상태, 기본 설정 |
| GET | `/api/sessions` | 활성 세션 / 저장된 세션 ID 목록(최근 50개) |
| GET | `/api/sessions/{session_id}/transcript` | 저장된 전사 결과(JSON) |
| GET | `/api/sessions/{session_id}/audio` | 저장된 녹음(WAV) |

### 8.3 상태 확인

```bash
curl http://localhost:8000/api/health
# {"status":"ok","sample_rate":16000,"active_sessions":[],"gpu_memory_mb":null}

curl http://localhost:8000/api/engines
# {"engines":{"whisper":{"ready":true,...},"sensevoice":{"ready":true,...}, ...},
#  "whisper_sizes":[...], "defaults":{...}}
```

배포 후 헬스체크나 준비 상태 확인(readiness probe)에는 `/api/health` 를 쓰면 됩니다.

### 8.4 오디오 준비

`/ws/stt` 는 **헤더 없는 raw PCM16 / little-endian / 16 kHz / mono** 만 받습니다.
WAV 파일이라면 헤더를 떼고 샘플만 보내야 합니다.

```bash
# 임의의 오디오 → raw PCM (ffmpeg)
ffmpeg -i input.m4a -ac 1 -ar 16000 -f s16le sample.pcm
```

```python
# 16 kHz / mono / 16-bit WAV → raw PCM (Python 표준 라이브러리)
import wave
with wave.open("sample.wav") as wf:
    pcm = wf.readframes(wf.getnframes())
```

마이크 입력을 중계할 때도 같은 형식으로 100 ms(3,200 bytes) 단위로 보내면 됩니다.

### 8.5 Python 클라이언트

`websockets` 패키지를 씁니다(`uvicorn[standard]` 설치 시 함께 설치됨).

```python
import asyncio
import json
import sys
import wave

import websockets

URL = "ws://localhost:8000/ws/stt"
FRAME_BYTES = 3200                     # 100 ms = 1600 samples × 2 bytes


async def stream(path: str) -> list[str]:
    with wave.open(path) as wf:
        assert (wf.getframerate(), wf.getnchannels(), wf.getsampwidth()) == (16000, 1, 2)
        pcm = wf.readframes(wf.getnframes())

    finals: list[str] = []
    async with websockets.connect(URL) as ws:

        async def send_audio() -> None:
            for i in range(0, len(pcm), FRAME_BYTES):
                await ws.send(pcm[i:i + FRAME_BYTES])   # bytes → binary frame
                await asyncio.sleep(0.1)                # 실시간 속도로 전송
            await ws.send("close")                      # str → text frame

        sender = asyncio.create_task(send_audio())
        async for raw in ws:
            msg = json.loads(raw)
            if msg["type"] == "transcript":
                if msg["is_final"]:
                    finals.append(msg["text"])          # ← DB 저장 대상
                    print("[final]  ", msg["text"], msg.get("start_time"), msg.get("end_time"))
                else:
                    print("[partial]", msg["text"])     # ← 화면 표시용
            elif msg["type"] == "error":
                print("[error]  ", msg["message"])
            elif msg["type"] == "done":                 # 마지막 메시지
                break
        await sender
    return finals


if __name__ == "__main__":
    print(asyncio.run(stream(sys.argv[1])))
```

```text
$ python stt_client.py recordings/sample.wav
[partial] 오늘 아침부터.
[partial] 오늘 아침 부터 배가 아팠 습니다.
...
[final]   오늘 아침부터 배가 아팠습니다 구토도 두 번습니다 열도 조금 났습니다. 0.0 6.1
['오늘 아침부터 배가 아팠습니다 구토도 두 번습니다 열도 조금 났습니다.']
```

### 8.6 Node.js 클라이언트

Node 22 이상은 `WebSocket` 이 내장돼 있어 추가 패키지가 필요 없습니다
(그 이하 버전은 `ws` 패키지를 쓰세요).

```js
// node stt_client.mjs sample.pcm
import { readFileSync } from "node:fs";

const URL = "ws://localhost:8000/ws/stt";
const FRAME_BYTES = 3200;                        // 100 ms
const pcm = readFileSync(process.argv[2]);       // raw PCM16 LE / 16 kHz / mono

const ws = new WebSocket(URL);
ws.binaryType = "arraybuffer";

ws.onopen = async () => {
  for (let i = 0; i < pcm.length; i += FRAME_BYTES) {
    ws.send(pcm.subarray(i, i + FRAME_BYTES));   // binary frame
    await new Promise((r) => setTimeout(r, 100));
  }
  ws.send("close");                              // text frame
};

ws.onmessage = ({ data }) => {
  const msg = JSON.parse(data);
  if (msg.type === "transcript") {
    console.log(msg.is_final ? "[final]  " : "[partial]", msg.text);
  } else if (msg.type === "error") {
    console.error("[error]  ", msg.message);
  } else if (msg.type === "done") {
    ws.close();
  }
};
```

### 8.7 연동 시 지켜야 할 것

- **연결 하나 = 오디오 스트림 하나.** 예진 세션을 시작할 때 연결을 열고 끝날 때
  닫습니다. 여러 세션의 오디오를 한 연결에 섞지 마세요. 로그 추적이 필요하면
  `?triage_session_id=<uuid>` 를 붙입니다(응답에는 포함되지 않습니다).
- **final 만 저장합니다.** `is_final: true` 하나가 발화 하나(= conversation row
  하나)입니다. partial 은 같은 발화의 중간 결과라 계속 바뀌므로 화면 표시에만 씁니다.
- **`close` 를 보낸 뒤 `done` 을 받을 때까지 연결을 유지하세요.** 마지막 발화의
  final 은 `close` 이후에 나옵니다. `done` 전에 끊으면 마지막 발화를 잃습니다.
- **연결 직후에는 모델 로드 시간만큼 응답이 늦을 수 있습니다.** 그동안 보낸
  오디오는 버려지지 않으므로 기다리지 않고 바로 보내도 됩니다.
- **실시간 속도로 보내는 것을 권장합니다.** 파일을 한 번에 밀어 넣어도 전사는
  되지만 서버 큐에 쌓여 partial 이 늦게 몰려 옵니다.
- **`start_time` / `end_time` 은 연결 기준 상대 시각(초)입니다.** 재연결하면
  0 부터 다시 셉니다. 절대 시각이 필요하면 Backend 가 연결 시작 시각을 더하세요.
- **오류 처리.** `STT engine initialization failed` 나 `Internal STT server error`
  를 받으면 서버가 연결을 끊으므로(1011) 새로 연결합니다. 나머지 오류는 연결이
  유지되니 기록만 하고 계속 보내면 됩니다.

### 8.8 전사·녹음 조회 (REST)

세션이 끝나면 서버가 전사 결과와 녹음을 `transcripts/`, `recordings/` 에
저장합니다(`STT_CONFIG='{"save_wav":false}'` 이면 녹음은 생략). 여기서 쓰는
`session_id` 는 STT 서버 내부 ID 라 `/ws/stt` 응답에는 나오지 않으므로,
`/api/sessions` 목록에서 찾아 운영·디버깅 용도로 씁니다.

```bash
curl http://localhost:8000/api/sessions
# {"active":[],"saved":["20260927-172618-6ed053", ...]}

curl http://localhost:8000/api/sessions/20260927-172618-6ed053/transcript
# {"session_id":"...","engine":"sensevoice","stable":"...",
#  "utterances":[{"text":"...","start":0.0,"end":6.1}],
#  "metrics":{"rtf_mean":...,"first_partial_ms":...}, "wav":"...", "error":null}

curl -o session.wav http://localhost:8000/api/sessions/20260927-172618-6ed053/audio
```
