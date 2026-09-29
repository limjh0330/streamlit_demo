# ER 실시간 음성 전사 (STT)

응급실 예진 대화를 **실시간으로 한국어 전사**하는 STT 서버입니다.
RunPod GPU 인스턴스에서 모델을 띄우고, 외부 Backend 나 브라우저가 WebSocket 으로
PCM 오디오를 보내면 partial / final 전사를 실시간으로 돌려줍니다.

- **4개 ASR 엔진**을 같은 인터페이스로 지원합니다 — Whisper · Zipformer · SenseVoice · Fun-ASR-MLT-Nano
- 엔진이 달라도 **API 규격은 동일**합니다. 엔진은 서버 설정으로만 바꿉니다.
- 연구·비교용 **Streamlit UI**(실시간 전사 / 파일 전사 / 성능 비교)가 함께 들어 있습니다.

## 목차

1. [시스템 개요](#1-시스템-개요)
2. [빠른 시작](#2-빠른-시작)
3. [지원 STT 모델](#3-지원-stt-모델)
4. [파이프라인 구조](#4-파이프라인-구조)
5. [STT API](#5-stt-api)
6. [설정](#6-설정)
7. [Streamlit UI](#7-streamlit-ui)
8. [평가와 벤치마크](#8-평가와-벤치마크)
9. [프로젝트 구조](#9-프로젝트-구조)
10. [테스트](#10-테스트)
11. [RunPod 배포 참고](#11-runpod-배포-참고)
12. [현재 제약과 남은 과제](#12-현재-제약과-남은-과제)

---

## 1. 시스템 개요

```
┌──────── Client ────────┐        ┌──────── External Backend ────────┐
│ Browser (Streamlit UI) │        │ 예진 세션 · turn · DB · LLM · KTAS │
│ getUserMedia           │        └───────────────┬──────────────────┘
│ → AudioWorklet (PCM16) │                        │ PCM16 16 kHz mono, 100 ms
└───────────┬────────────┘                        │ WS /ws/v1/stt/stream
            │ WS /ws/v1/stt/browser               │
            ▼                                     ▼
┌──────────────────────── RunPod ───────────────────────────────────┐
│  Streamlit :8501            STT server (FastAPI) :8000            │
│                               │                                   │
│                     StreamingSession (연결 1개 = 세션 1개)         │
│                     VAD/endpoint → ASR Engine → TranscriptMerger  │
│                               │                                   │
│                     transcript (partial / final)                  │
└───────────────────────────────┬───────────────────────────────────┘
                                ▼
                  Backend / Browser 로 JSON 회신
```

### 책임 범위

| STT 서버가 하는 일 | STT 서버가 **하지 않는** 일 (Backend 책임) |
|---|---|
| PCM 오디오 수신 · 검증 | conversation / turn 번호 관리 |
| VAD · 발화 끝(endpoint) 판정 | 화자 구분, 환자·의료진 역할 추정 |
| ASR 추론 · partial / final 전사 | KTAS · Major/Minor 분류 |
| 의료 용어 후처리 | LLM 호출 |
| WAV 녹음 · 지연/RTF 지표 기록 | DB 저장 |

### 두 개의 WebSocket

| 경로 | 쓰는 쪽 | 특징 |
|---|---|---|
| `/ws/v1/stt/stream` | **External Backend** | 오디오와 `close` 만 보낸다. 엔진 설정은 서버가 정한다 |
| `/ws/v1/stt/browser` | Streamlit 실시간 전사 페이지 | `start` 메시지로 엔진·윈도우를 직접 고른다(실험용) |

두 경로는 입출력 형식만 다르고 뒤의 `StreamingSession` 파이프라인은 같습니다.

---

## 2. 빠른 시작

### 2.1 설치

anaconda base 환경은 NumPy 1.x 로 빌드된 패키지와 충돌하므로 **전용 가상환경**을 씁니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2.2 모델 준비

```bash
python -m scripts.fetch_models                    # zipformer + sensevoice (int8) + funasr_mlt_nano
python -m scripts.fetch_models sensevoice         # 하나만
python -m scripts.fetch_models --keep all --force # fp32 까지 (GPU 권장)
```

Whisper 는 첫 실행 때 Hugging Face 캐시(`~/.cache/huggingface`)로 자동 다운로드됩니다.
모델별 위치와 용량은 [3.1](#31-한눈에-보기) 을 보세요.

### 2.3 실행

```bash
# STT 서버 (API) — 기본 엔진 Fun-ASR-MLT-Nano 를 preload → warm-up → READY
./serverctl.sh start --wait                      # 운영: VS Code·SSH 창을 닫아도 계속 동작 (11.1 참고)
./start_server.sh                                # 개발: 포그라운드 실행 (터미널을 닫으면 종료)
STT_ENGINE=funasr_mlt_nano ./start_server.sh          # 다른 엔진 (밖에서 준 환경변수가 우선)
python -m server.main --host 0.0.0.0 --port 8000 # 스크립트 없이 직접 실행해도 같다

# Streamlit UI (선택)
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

### 2.4 동작 확인

```bash
curl -s http://localhost:8000/api/v1/stt/health          # liveness: {"status":"OK", ...}
curl -s http://localhost:8000/api/v1/stt/health/ready    # readiness: LOADING(503) → READY(200)
curl -s http://localhost:8000/api/v1/stt/engines         # active_engine, 엔진별 준비 상태

python -m scripts.stt_client recordings/sample.wav       # 실제 모델로 E2E (READY 를 기다렸다가 전송)
```

브라우저로 확인하려면 **Swagger UI** `http://localhost:8000/api/v1/stt/docs` 를 여세요
(RunPod: `https://<POD_ID>-8000.proxy.runpod.net/api/v1/stt/docs`). 사용법은 [5.3](#53-rest-api--swagger-ui-로-브라우저에서-테스트).

---

## 3. 지원 STT 모델

### 3.1 한눈에 보기

| `STT_ENGINE` | 모델 | 패키지 | 모델 위치 | 입력 방식 | 단어 타임스탬프 | 권장 장치 |
|---|---|---|---|---|---|---|
| `whisper` | faster-whisper `tiny`~`large-v3` (기본 `small`) | `faster-whisper` | HF 캐시 (자동 다운로드) | sliding window 5 s / overlap 1.5 s | 있음 | CPU(small 까지) / GPU |
| `zipformer` | sherpa-onnx streaming zipformer korean 2024-06-16 | `sherpa-onnx` | `models/zipformer/` (127 MB, int8) | 100 ms 블록 native streaming | 있음 (토큰 기반) | CPU |
| `sensevoice` | sherpa-onnx SenseVoice zh-en-ja-ko-yue 2024-07-17 | `sherpa-onnx` | `models/sensevoice/` (229 MB, int8) | 발화 전체를 0.8 s 마다 재인식 | 없음 (균등 분할 추정) | CPU / GPU |
| `funasr_mlt_nano` **(기본)** | FunAudioLLM/Fun-ASR-MLT-Nano-2512 | `funasr`, `torch` | `models/funasr_mlt_nano/` (1.9 GB) | 발화 전체를 **5 s** 마다 재인식 | 없음 | **GPU** |

모든 엔진은 `stt/asr/base.py` 의 `ASREngine` 을 구현하며, 세션은 아래 세 속성만 보고 처리 경로를 고릅니다.

| 속성 | 의미 | 해당 엔진 |
|---|---|---|
| `native_streaming` | `accept_waveform()` / `partial()` 로 블록마다 가설 갱신 | Zipformer |
| `decodes_full_utterance` | 윈도우를 자르지 않고 발화 시작부터 현재까지 매번 재인식 | SenseVoice, Fun-ASR |
| `has_word_timestamps` | 단어 시각으로 겹침 제거·확정 | Whisper, Zipformer |

### 3.2 Whisper (`stt/asr/whisper.py`)

- **정확도 기준점(baseline)**. autoregressive 라 느리고 첫 partial 이 늦습니다.
- partial 윈도우는 `beam_size=1`, 발화 확정 시 1회만 `final_beam_size=5` 로 다시 디코딩합니다.
- `initial_prompt` 에 ER 용어 사전(`medical_terms.TERMS`)을 넣어 도메인 어휘 쪽으로 편향시킵니다.
- `condition_on_previous_text=False`, `vad_filter=False`(VAD 는 세션에서 수행).
- 모델은 프로세스 전역 캐시(`_MODEL_CACHE`)로 **한 번만 로드**되고 세션 간에 공유됩니다.
- CPU int8 에서는 `small` 까지가 실시간입니다. `medium` 이상은 GPU 를 쓰세요.

### 3.3 Zipformer (`stt/asr/zipformer.py`)

- **유일한 진짜 streaming 엔진**. 100 ms 마다 가설을 갱신하므로 첫 partial 이 가장 빠릅니다.
- sherpa-onnx 내장 endpoint 규칙을 세션 VAD 와 함께 씁니다
  (`rule2_min_trailing_silence = max(0.4, silence_sec)`, `rule3 = max_utterance_sec`).
- BPE 토큰의 선행 공백(`▁`)을 어절 경계로 삼아 단어를 다시 묶습니다
  (`get_result()` 는 공백이 지워진 문자열을 주기 때문).
- endpoint 직후 마지막 어절이 잘리지 않도록 0.3 s 무음을 흘려 넣어 꼬리 토큰을 끌어냅니다.
- 받아 쓴 모델은 일상 대화(KsponSpeech)로 학습돼 **의료 용어 정확도가 매우 낮습니다**. 의료 도메인에는 fine-tuning 이 필요합니다.
- 한국어 전용 모델입니다.

### 3.4 SenseVoice (`stt/asr/sensevoice.py`)

- non-autoregressive 라 RTF ≈ 0.02 로 매우 빠릅니다. 그래서 윈도우를 자르지 않고
  **발화 전체를 `refresh_sec`(0.8 s) 마다 다시 인식**하고 최신 가설로 교체합니다.
  잘린 오디오의 오인식과 윈도우 간 중복이 함께 사라집니다.
- 단어 타임스탬프가 없어, 발화 구간을 어절 수로 **균등 분할한 추정 시각**을 씁니다.
  final 의 `start_time`/`end_time` 은 발화 단위로만 신뢰하세요.
- 다국어(zh/en/ja/ko/yue) 모델이며 `language` 로 언어를 지정합니다. ITN(숫자 정규화) 사용.

### 3.5 Fun-ASR-MLT-Nano-2512 (`stt/asr/funasr_mlt_nano.py`) — 기본 엔진

- FunASR `AutoModel.generate` 를 쓰는 LLM 기반 다국어 ASR 입니다
  (`models/funasr_mlt_nano/` 에 Qwen3-0.6B 디코더 포함).
- `models/funasr_mlt_nano/model.pt` 가 있으면 로컬 모델을, 없으면 Hugging Face 에서 받습니다.
- 모델이 커서 CPU 에서 0.8 s 갱신을 따라가지 못하므로 **갱신 주기를 최소 5 s** 로 둡니다
  (`StreamingSession.start()`). partial 이 드물게 갱신되는 것은 정상입니다.
- 추론마다 16 kHz 임시 WAV 를 만들어 파일 경로로 넘깁니다(공식 API 입력 형식).
- `language` 는 `ko`/`en`/`zh`/`ja`/`yue` 를 모델의 언어명으로 매핑합니다. 타임스탬프 미지원.
- 데이터셋 평가에서 **정확도가 가장 높아 기본 엔진**으로 씁니다([8.2](#82-데이터셋-평가-결과)).
  기본값은 `stt/config.py` 의 `DEFAULT_ENGINE` 에서 바꿉니다.

### 3.6 엔진 선택 가이드

| 목적 | 추천 | 이유 |
|---|---|---|
| 실시간 표시 + 적당한 정확도 | **SenseVoice** | 빠르고(RTF 0.02) 실시간 여유가 가장 큼 |
| 정확도 우선 (GPU 있음) | **Fun-ASR-MLT-Nano** | CER·의료용어 정확도 최고. partial 갱신은 5 s 간격 |
| 비교 기준점 | Whisper `small` | 단어 타임스탬프가 가장 정확. CPU 에서는 실시간 한계 근처 |
| 지연 실험 | Zipformer | 가장 빠르지만 현재 모델로는 의료 대화 인식 불가 수준 |

---

## 4. 파이프라인 구조

### 4.1 전체 흐름

```
WebSocket binary (PCM16 LE, 16 kHz, mono)
   │
   ▼
StreamingSession.feed()              ← 논블로킹. PCM16 → float32, 필요 시 리샘플
   │  queue
   ▼  ─────────────── 세션 전용 워커 스레드 ───────────────
   ├─→ WavRecorder                   → recordings/<stt_session_id>.wav
   ├─→ EndpointDetector (VAD)        → 발화 시작 / 끝(endpoint) 판정
   │
   ├─ windowed 엔진 ──→ SlidingWindowBuffer ──→ ASREngine.transcribe(audio, t0)
   └─ native 엔진   ──→ ASREngine.accept_waveform() / partial() / is_endpoint()
   │
   ▼
TranscriptMerger                     → stable(확정) + unstable(흔들리는 꼬리)
   │
   ▼
medical_terms.correct()              → 의료 용어 후처리 (medical_correction=True)
   │
   ▼
Event(partial / final / metrics / error)
   │
   ▼
WebSocket 어댑터                      → /stream: transcript 규격 / /browser: UI 규격
```

### 4.2 모듈별 역할

| 단계 | 모듈 | 하는 일 |
|---|---|---|
| 입력 변환 | `stt/audio/resampler.py` | PCM16 ↔ float32, 리샘플, RMS(dBFS) |
| 녹음 | `stt/audio/recorder.py` | 수신 오디오를 WAV 로 실시간 append |
| VAD / endpoint | `stt/audio/vad.py` | 적응형 노이즈 플로어 RMS VAD(`webrtcvad` 있으면 사용). 무음 `silence_sec` 이상이면 endpoint, `max_utterance_sec` 넘으면 강제 확정 |
| 윈도우 | `stt/audio/buffer.py` | sliding window + overlap. 발화 첫 윈도우는 `first_hop_sec` 로 짧게 끊어 첫 partial 을 앞당김 |
| ASR | `stt/asr/*.py` | 엔진 어댑터 ([3장](#3-지원-stt-모델)) |
| 병합 | `stt/transcript/merger.py` | 겹침 제거, stable/unstable 관리, 발화 확정 |
| 후처리 | `stt/transcript/medical_terms.py` | 오인식 사전 치환 + 유사도 교정, Whisper `initial_prompt` 생성 |
| 지표 | `stt/metrics/latency.py` | RTF, 첫 partial, final 지연, CPU/메모리 |
| 세션 | `stt/session.py` | 위 단계를 묶는 스레드 안전한 파이프라인 |

### 4.3 엔진별 처리 경로

| 경로 | 엔진 | 동작 | 병합기 진입점 |
|---|---|---|---|
| sliding window | Whisper | 5 s 윈도우를 3.5 s 마다 인식(첫 윈도우 1.5 s) | `update(words)` — 단어 시각 기준 정렬 |
| 발화 전체 재인식 | SenseVoice, Fun-ASR | 발화 시작~현재를 `refresh_sec` 마다 다시 인식 | `replace_text(text)` — 최신 가설로 교체 |
| native streaming | Zipformer | 100 ms 마다 가설 갱신, **가설이 바뀐 경우에만** 병합 | `update(words)` |

노이즈 플로어 수준인 윈도우는 추론을 건너뜁니다(`_is_silent`). 무음에서 Whisper 가
없는 말을 지어내거나 `initial_prompt` 때문에 수 초씩 헛도는 것을 막기 위해서입니다.

### 4.4 TranscriptMerger — 중복·누락 없이 합치기

윈도우가 겹치므로 같은 단어가 여러 번 나오고, Whisper 는 앞 단어를 고쳐 쓰기도 합니다.

1. **시각 기준 제거** — 이미 확정된 시각 이전의 단어는 버린다.
2. **텍스트 기준 제거** — 확정된 꼬리와 새 가설의 머리가 겹치면 잘라낸다(타임스탬프가 수십 ms 씩 흔들리기 때문).
3. **LocalAgreement-2** — 두 번 연속 같게 나온 접두사만 stable 로 확정, 나머지는 unstable.
4. 다음 윈도우가 이어받지 못하는 단어는 버리지 않고 그 시점에 확정한다.
5. endpoint 에서 unstable 까지 모두 확정해 **발화(Utterance) 하나**를 만든다 → final.

이 규칙들은 `tests/test_pipeline.py` 의 window/overlap/지터 조합 회귀 테스트로 고정돼 있습니다.

### 4.5 세션 수명과 종료(flush)

```
서버 시작        RUNTIME.configure()  환경변수 → 활성 설정 고정 (source of truth)
                 RUNTIME.start_preload()  create_engine(활성 설정) → warmup(strict) → READY
open_session()   preload 가 끝날 때까지 대기 → StreamingSession 생성 → start()(엔진 인스턴스, 워커)
                 → 공유 모델이 이미 warm 이면 warmup() 생략
feed() × N       오디오 블록 투입
close_session()  stop(): 큐 잔여 처리 → 버퍼/디코더 flush → 마지막 final emit
                        → recorder close → 요약 반환
                 save_transcript(): transcripts/<id>.json, .txt 저장
```

`/ws/v1/stt/stream` 은 `close` 를 받으면 위 flush 가 **끝난 뒤**에 `done` 을 보냅니다.
`stop()` 안에서 나온 final 은 `done` 보다 항상 먼저 전송됩니다.

`close_session()` 은 정리를 전용 스레드에서 끝까지 수행하고 `SESSIONS` 에서는 맨 마지막에 뺍니다.
연결 처리 코루틴이 취소되더라도(서버 종료 등) 워커·WAV recorder 정리가 빠지지 않습니다.

#### 모델 공유와 동시성

| 엔진 | 가중치 | 세션별로 따로인 것 | 동시 추론 |
|---|---|---|---|
| Fun-ASR | 프로세스당 1회 로드(`_MODEL_CACHE`) | 엔진 인스턴스, VAD, 버퍼, 병합기 | 모델 단위 공유 lock 으로 **한 번에 하나씩** |
| Whisper | 프로세스당 1회 로드 | 〃 | 인스턴스별 lock (CTranslate2 내부에서 직렬 처리) |
| SenseVoice / Zipformer | **연결마다 로드** | recognizer 포함 전부 | 연결별 독립 |

`AutoModel.generate()` 의 동시 호출 안전성이 보장되지 않아, 세션 A/B/C 의 Fun-ASR 추론은
공유 lock 으로 직렬화합니다. 세션 하나일 때는 기다림이 없어 기존 성능과 같고, 세션이 늘면
추론이 순서대로 처리되므로 **동시 세션 수 × RTF < 1** 안에서 운영해야 실시간이 유지됩니다.

### 4.6 세션 이벤트 (내부)

| type | 필드 | 발생 |
|---|---|---|
| `ready` | `session_id`, `engine`, `config` | 세션 시작 |
| `partial` | `stable`(세션 전체 확정문), `partial`, `committed`, `utterance`(현재 발화) | 가설이 바뀔 때 |
| `final` | `text`, `start`, `end`, `index`, `stable` | endpoint · 종료 flush |
| `metrics` | RTF, 지연, 큐 길이, 레벨 등 | 약 2 s 마다 |
| `error` | `message` | 추론 실패(세션은 계속 동작) |

WebSocket 어댑터가 이 이벤트를 각 API 규격으로 변환합니다.

---

## 5. STT API

### 5.1 엔드포인트

REST 는 **`/api/v1/stt`**, WebSocket 은 **`/ws/v1/stt`** 를 base path 로 씁니다
(`stt/config.py` 의 `API_BASE` / `WS_BASE`).

| 종류 | 경로 | 용도 |
|---|---|---|
| WebSocket | `/ws/v1/stt/stream` | **외부 서비스용 실시간 전사** |
| WebSocket | `/ws/v1/stt/browser` | Streamlit 실시간 전사 페이지 전용 |
| GET | `/api/v1/stt/health` | **liveness** — 프로세스가 살아 있으면 항상 200 |
| GET | `/api/v1/stt/health/ready` | **readiness** — 모델 로드·warm-up 완료 시 200(`READY`), 아니면 503 |
| GET | `/api/v1/stt/engines` | 실제 사용 중인 엔진(`active_engine`), 엔진별 설치·로드 상태 |
| GET | `/api/v1/stt/sessions` | 활성 / 저장된 세션 ID (최근 50개) |
| GET | `/api/v1/stt/sessions/{session_id}/transcript` | 저장된 전사 결과(JSON) |
| GET | `/api/v1/stt/sessions/{session_id}/audio` | 저장된 녹음(WAV) |
| 문서 | `/api/v1/stt/docs` | **Swagger UI** — REST API 를 브라우저에서 바로 실행 (`/docs` 는 여기로 리다이렉트) |
| 문서 | `/api/v1/stt/redoc` · `/api/v1/stt/openapi.json` | ReDoc 문서 · OpenAPI 명세(JSON) |

```
ws://<host>:8000/ws/v1/stt/stream
wss://<podId>-8000.proxy.runpod.net/ws/v1/stt/stream     # RunPod 프록시
```

### 5.2 실시간 전사 WebSocket — `/ws/v1/stt/stream`

연결 하나가 오디오 스트림 하나(STT 세션 하나)입니다. Backend 는 연결과 예진 세션을 스스로 매핑합니다.
로그 추적이 필요하면 `?triage_session_id=<uuid>` 를 붙이세요 — **서버 로그에만** 남고 응답에는 포함되지 않습니다.

#### Backend → STT

| 메시지 | 형식 | 내용 |
|---|---|---|
| 오디오 | binary frame | raw **PCM16 little-endian / 16 kHz / mono**, 100 ms = 1,600 samples = **3,200 bytes** |
| 종료 | text frame | `close` (`{"type":"close"}` 도 허용) |

프레임 처리 규칙:

- 3,200 B 가 아닌 프레임도 받습니다(첫 1회 경고 로그).
- 홀수 길이로 샘플이 쪼개지면 남는 1 바이트를 다음 프레임 앞에 붙여 정렬을 유지합니다.
- 빈 프레임은 무시합니다.
- 1 s(32,000 B)를 넘는 프레임은 버리고 `Invalid audio frame` 을 보냅니다. 연결은 유지됩니다.

#### STT → Backend

모든 응답은 JSON text frame 입니다.

```jsonc
// partial — 현재 발화의 중간 결과. 화면 표시용이며 DB 저장용이 아님. 직전과 같으면 보내지 않음
{"type": "transcript", "text": "가슴이", "is_final": false}

// final — 발화 하나의 확정 결과 = Backend conversation row 하나
{"type": "transcript", "text": "가슴이 답답해요", "is_final": true,
 "start_time": 12.4, "end_time": 15.6}

// error — 안전한 문구만 전달(traceback·경로·예외 내용은 서버 로그에만)
{"type": "error", "message": "STT inference failed"}

// done — close 이후 flush 완료. 항상 마지막 메시지
{"type": "done"}
```

- `start_time` / `end_time` 은 선택 필드로, **이 연결로 받은 오디오 기준 상대 시각(초)** 입니다.
  `STT_TIMESTAMPS=0` 이면 빠집니다.
- `turn_id`, `conversation_id`, 화자/역할, 세션 ID 는 보내지 않습니다.

#### 오류 메시지

| message | 상황 | 이후 |
|---|---|---|
| `STT engine initialization failed` | 세션 엔진 초기화 실패 (preload 가 실패해 첫 연결에서 다시 로드하다 실패한 경우 등) | 연결 종료 (1011) |
| `STT inference failed` | 추론 중 오류 (초당 최대 1회 통지) | 계속 동작 |
| `Invalid audio frame` | 1 s 초과 프레임 | 해당 프레임만 버림 |
| `Unknown control message` | `close` 가 아닌 text frame | 계속 동작 |
| `Internal STT server error` | 예기치 못한 서버 오류 | 연결 종료 (1011) |

#### 흐름

```
Backend                                     STT
  ── connect /ws/v1/stt/stream ────────────▶  세션 생성 (모델은 서버 시작 시 preload 완료)
  ── <3200 B> <3200 B> <3200 B> … ─────────▶
  ◀── {"type":"transcript","text":"가슴이","is_final":false}
  ◀── {"type":"transcript","text":"가슴이 답답해요","is_final":false}
  ◀── {"type":"transcript","text":"가슴이 답답해요","is_final":true,…}   ← 무음 → endpoint
  ── <3200 B> … ───────────────────────────▶
  ── "close" ──────────────────────────────▶  남은 오디오 · 디코더 · 병합기 flush
  ◀── {"type":"transcript","text":"(남은 발화)","is_final":true,…}
  ◀── {"type":"done"}
  ◀── close(1000)
```

`close` 없이 연결이 끊기면 STT 는 flush · WAV/전사 저장 · 세션 정리만 하고 아무것도 보내지 않습니다.

### 5.3 REST API · Swagger UI 로 브라우저에서 테스트

#### Swagger UI

서버가 떠 있으면 별도 설치 없이 브라우저에서 REST API 를 실행해 볼 수 있습니다.

| 환경 | 주소 |
|---|---|
| 로컬 | `http://localhost:8000/api/v1/stt/docs` |
| RunPod | `https://<POD_ID>-8000.proxy.runpod.net/api/v1/stt/docs` |

1. 위 주소를 엽니다(`/docs` 로 들어가도 자동으로 이동합니다).
2. 확인할 API 줄(예: `GET /api/v1/stt/health/ready`)을 눌러 펼칩니다.
3. **Execute** 를 누릅니다(Try it out 은 기본으로 켜져 있음). 아래에 실제 요청 URL(`curl` 명령 포함),
   응답 코드, 응답 JSON, 소요 시간이 표시됩니다.
4. 세션 조회처럼 경로 값이 필요한 API 는 입력칸에 `session_id` 를 넣고 Execute 합니다.
   값은 먼저 `GET /api/v1/stt/sessions` 를 실행해 얻습니다. `/audio` 는 응답에 **Download file** 링크가 나옵니다.

| 태그 | API | 확인할 것 |
|---|---|---|
| Health | `GET /health` | 항상 200 `{"status":"OK"}` |
| Health | `GET /health/ready` | `READY` 면 200, 로딩 중·실패면 503 (예시 응답은 Responses 의 드롭다운에서 볼 수 있음) |
| Engines | `GET /engines` | `active_engine` 이 의도한 엔진(기본 `funasr_mlt_nano`)인지, `loaded: true` 인지 |
| Sessions | `GET /sessions` → `/sessions/{id}/transcript` · `/audio` | 저장된 전사·녹음 |

- 각 API 의 응답 스키마와 필드 설명은 펼친 화면의 **Schema** 탭, 페이지 맨 아래 **Schemas** 에 있습니다.
- 같은 문서를 읽기용으로 보려면 ReDoc(`/api/v1/stt/redoc`), 코드 생성·Postman 가져오기에는
  OpenAPI 명세(`/api/v1/stt/openapi.json`)를 씁니다.
- **WebSocket(`/ws/v1/stt/stream`)은 Swagger 에서 실행되지 않습니다.** OpenAPI 가 WebSocket 을 표현하지 못해
  페이지 상단 설명에 규격만 적어 두었습니다. 실시간 전사는 `python -m scripts.stt_client` 로 시험하세요([10장](#10-테스트)).
- Swagger UI 화면 파일(JS/CSS)은 `cdn.jsdelivr.net` 에서 받아 오므로, 브라우저가 인터넷에 연결돼 있어야 합니다.
- 인증이 없어 문서 주소를 아는 누구나 세션 목록·녹음을 받을 수 있습니다. 외부에 노출된 운영 환경에서는
  `STT_DOCS=0` 으로 문서를 끄세요(API 자체는 그대로 동작).

#### Liveness / Readiness

```bash
curl -s http://localhost:8000/api/v1/stt/health
# 200 {"status":"OK","sample_rate":16000,"active_sessions":[],"gpu_memory_mb":null}
```

`/health` 는 모델 상태와 무관하게 프로세스만 봅니다. 트래픽을 보내도 되는지는 `/health/ready` 로 판단하세요.

```jsonc
// GET /api/v1/stt/health/ready
// 200 — 모델 로드 + warm-up 추론까지 완료
{"status":"READY","engine":"funasr_mlt_nano","device":"cuda","model_loaded":true,
 "gpu_available":true,"gpu_memory_mb":2500.0,"sample_rate":16000,"active_sessions":0,
 "load_sec":15.0,"warmup_sec":1.2,"output_time":1788095605.9}

// 503 — 로딩 중 (이때 연결해도 오류 없이 READY 까지 기다렸다가 처리됨)
{"status":"LOADING","engine":"funasr_mlt_nano","model_loaded":false,"gpu_available":true, ...}

// 503 — 초기화 실패 (상세 원인은 서버 로그에만)
{"status":"NOT_READY","engine":"funasr_mlt_nano","model_loaded":false,"gpu_available":true,
 "error":"FileNotFoundError: model initialization failed (see server log)", ...}
```

#### 엔진

```bash
curl -s http://localhost:8000/api/v1/stt/engines
# {"active_engine":"funasr_mlt_nano","active_status":"READY","active_config":{...},
#  "engines":{"funasr_mlt_nano":{"ready":true,"loaded":true,...},"whisper":{"ready":true,"loaded":false,...}, ...},
#  "whisper_sizes":[...], "defaults":{...}}
```

`active_engine` / `active_config` 는 서버가 **실제로** 쓰는 설정(= preload 엔진 = `/stream` 세션 엔진),
`defaults` 는 설정을 주지 않았을 때의 코드 기본값입니다. `ready` 는 패키지·모델 파일 준비, `loaded` 는 메모리 적재 여부입니다.

#### 세션 조회

```bash

curl http://localhost:8000/api/v1/stt/sessions
# {"active":[],"saved":["20260927-172618-6ed053", ...]}

curl http://localhost:8000/api/v1/stt/sessions/20260927-172618-6ed053/transcript
# {"session_id":"...","engine":"sensevoice","stable":"...","partial":"",
#  "utterances":[{"text":"...","start":0.0,"end":6.1}],
#  "metrics":{"rtf_mean":...,"first_partial_ms":...,"final_latency_ms":...}, "wav":"...", "error":null}

curl -o session.wav http://localhost:8000/api/v1/stt/sessions/20260927-172618-6ed053/audio
```

`session_id` 는 STT 내부 ID(`YYYYMMDD-HHMMSS-xxxxxx`)입니다. `/ws/v1/stt/stream` 응답에는
나오지 않으므로 이 API 들은 운영·디버깅 용도입니다.

### 5.4 클라이언트 예제

#### 오디오 준비

`/ws/v1/stt/stream` 은 **헤더 없는 raw PCM** 만 받습니다.

```bash
ffmpeg -i input.m4a -ac 1 -ar 16000 -f s16le sample.pcm          # 임의 오디오 → raw PCM
```

```python
import wave                                                      # 16 kHz/mono/16-bit WAV → raw PCM
with wave.open("sample.wav") as wf:
    pcm = wf.readframes(wf.getnframes())
```

#### Python (`websockets`, `uvicorn[standard]` 설치 시 포함)

```python
import asyncio
import json
import sys
import wave

import websockets

URL = "ws://localhost:8000/ws/v1/stt/stream"
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
$ python stt_client.py recordings/sample.wav          # STT_ENGINE=sensevoice
[partial] 오늘 아침부터.
[partial] 오늘 아침 부터 배가 아팠 습니다.
...
[final]   오늘 아침부터 배가 아팠습니다 구토도 두 번습니다 열도 조금 났습니다. 0.0 6.1
```

#### Node.js (22 이상은 `WebSocket` 내장, 그 이하는 `ws` 패키지)

```js
// node stt_client.mjs sample.pcm
import { readFileSync } from "node:fs";

const URL = "ws://localhost:8000/ws/v1/stt/stream";
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

### 5.5 연동 체크리스트

- **연결 하나 = 오디오 스트림 하나.** 여러 세션의 오디오를 한 연결에 섞지 않습니다.
- **final 만 저장합니다.** partial 은 같은 발화의 중간 결과라 계속 바뀝니다.
- **`close` 후 `done` 을 받을 때까지 연결을 유지합니다.** 마지막 발화의 final 은 `close` 이후에 옵니다.
- **연결 전에 `/api/v1/stt/health/ready` 가 `READY`(200)인지 확인합니다.** 로딩 중에 연결해도 오류 없이 READY 까지 기다렸다가 처리하며, 그동안 보낸 오디오는 버려지지 않습니다.
- **실시간 속도(100 ms 간격)로 보내는 것을 권장합니다.** 한 번에 밀어 넣으면 서버 큐에 쌓여 partial 이 몰려 옵니다.
- **시각은 연결 기준 상대값입니다.** 재연결하면 0 부터 다시 셉니다.
- **`initialization failed` / `Internal STT server error` 는 연결이 끊기므로 재연결**하고, 나머지 오류는 기록만 하고 계속 보냅니다.

### 5.6 브라우저 WebSocket — `/ws/v1/stt/browser`

Streamlit 실시간 전사 페이지(`web/live_mic.js`) 전용입니다. 외부 연동에는 `/stream` 을 쓰세요.

```jsonc
// client → server
{"type":"start","engine":"whisper","model_size":"small","language":"ko","sample_rate":16000,
 "window_sec":5,"overlap_sec":1.5,"silence_sec":0.7,"vad_threshold_db":12,
 "medical_correction":true,"save_wav":true}
<binary>                       // PCM16 little-endian mono (100~500 ms)
{"type":"stop"}
{"type":"ping"}                // → {"type":"pong"}

// server → client
{"type":"ready",   "session_id":"...", "engine":"whisper", "config":{...}}
{"type":"partial", "stable":"...", "partial":"...", "committed":"...", "utterance":"..."}
{"type":"final",   "text":"...", "start":0.0, "end":3.2, "index":0, "stable":"..."}
{"type":"metrics", "rtf_mean":0.42, "first_partial_ms":1830, ...}
{"type":"error",   "message":"..."}
{"type":"closed",  "summary":{...}, "text":"...", "utterances":[...], "wav":"...", "transcript":"...", "session_id":"..."}
```

`start` 의 키는 `StreamConfig` 필드 이름과 같으며, 여기서는 클라이언트가 엔진을 고를 수 있습니다.

---

## 6. 설정

### 6.1 환경변수

| 환경변수 | 기본값 | 쓰는 곳 | 의미 |
|---|---|---|---|
| `STT_ENGINE` | `funasr_mlt_nano` | `/stream` | `funasr_mlt_nano` · `whisper` · `zipformer` · `sensevoice` |
| `STT_MODEL_SIZE` | `small` | `/stream` | Whisper 모델 크기 |
| `STT_CONFIG` | (없음) | `/stream` | `StreamConfig` 필드 덮어쓰기 JSON. 예: `{"silence_sec":0.6,"save_wav":false}` |
| `STT_TIMESTAMPS` | `1` | `/stream` | final 에 `start_time`/`end_time` 포함 |
| `STT_DOCS` | `1` | 서버 | Swagger UI·ReDoc·OpenAPI 명세 제공. `0` 이면 `/api/v1/stt/docs` 등이 404 |
| `STT_PRELOAD` | `1` | 서버 | 시작 시 활성 엔진 preload·warm-up. `0` 이면 첫 연결에서 로드(개발용, readiness 는 `NOT_READY`) |
| `STT_HOST` | `0.0.0.0` | `start_server.sh` | 바인드 주소 |
| `PYTHON` | `.venv/bin/python` → `python` | `start_server.sh` | 사용할 인터프리터 |
| `STT_BACKEND_PORT` | `8000` | Streamlit | 브라우저 WS 주소 유도에 쓰는 포트 |
| `STT_API_URL` | `http://127.0.0.1:8000` | Streamlit | Python → STT 서버 (같은 인스턴스) |
| `STT_WS_URL` | (비움 → 자동 유도) | Streamlit | 브라우저 → STT 서버. RunPod 은 `<podId>-<port>.proxy.runpod.net` 으로 유도 |

서버 설정은 **서버가 시작할 때 한 번** 환경변수에서 읽어 고정합니다(`server/runtime.py` 의 `RUNTIME`).
preload, `/stream` 세션, `/engines`·`/health/ready` 가 모두 이 설정을 보므로 "preload 는 funasr, 세션은 whisper"
같은 불일치가 생기지 않습니다. 설정을 바꾸려면 서버를 재시작하세요. Backend 는 엔진 설정을 보내지 않습니다.

### 6.2 `StreamConfig` (`stt/config.py`)

| 분류 | 필드 | 기본값 | 설명 |
|---|---|---|---|
| 엔진 | `engine` | `funasr_mlt_nano` | 사용할 엔진 (`DEFAULT_ENGINE`) |
| | `model_size` | `small` | Whisper 크기 |
| | `model_dir` | `None` | sherpa 모델 디렉터리 (기본 `models/<engine>`) |
| | `device` | 자동 | CUDA 감지 시 `cuda`, 아니면 `cpu` |
| | `compute_type` | 자동 | GPU `float16`, CPU `int8` (Whisper) |
| | `language` | `ko` | 인식 언어 |
| | `beam_size` / `final_beam_size` | `1` / `5` | Whisper partial / final 빔 크기 |
| 윈도우 | `window_sec` / `overlap_sec` | `5.0` / `1.5` | sliding window (Whisper) |
| | `min_window_sec` | `1.0` | 이보다 짧으면 추론 안 함 |
| | `first_hop_sec` | `1.5` | 발화 첫 윈도우 길이 (첫 partial 지연) |
| | `refresh_sec` | `0.8` | 발화 전체 재인식 주기 (Fun-ASR 은 최소 5 s) |
| VAD | `silence_sec` | `0.7` | 이만큼 무음이면 발화 확정(final) |
| | `max_utterance_sec` | `20.0` | 무음이 없어도 강제 확정 |
| | `vad_threshold_db` | `12.0` | 노이즈 플로어 대비 음성 판정 마진 |
| 후처리 | `medical_correction` | `True` | 의료 용어 교정 |
| | `use_initial_prompt` | `True` | Whisper 에 ER 용어 프롬프트 |
| 저장 | `save_wav` | `True` | `recordings/` 에 WAV 저장 |

튜닝 팁:

- 첫 partial 을 빠르게 → `first_hop_sec` ↓ (추론 횟수·CPU ↑)
- 발화가 너무 잘게 끊김 → `silence_sec` ↑ / 발화 확정이 늦음 → `silence_sec` ↓
- 시끄러운 환경에서 무음을 음성으로 인식 → `vad_threshold_db` ↑

---

## 7. Streamlit UI

```bash
streamlit run streamlit_app.py --server.port 8501
```

| 페이지 | 파일 | 하는 일 |
|---|---|---|
| 실시간 전사 | `app_pages/realtime.py` | 브라우저 마이크 → `/ws/v1/stt/browser` → partial/final, 라이브 지표. 엔진·윈도우·VAD 를 사이드바에서 선택 |
| 파일 전사 | `app_pages/file_stt.py` | 업로드 / `st.audio_input` 녹음을 Whisper 로 통째로 전사. 정답(reference) 만들기용 |
| 성능 비교 | `app_pages/metrics.py` | `transcripts/*.json` 을 모아 엔진별 RTF·지연·WER/CER 비교 |

- 브라우저는 **HTTPS 또는 localhost** 에서만 마이크를 엽니다. RunPod 프록시는 HTTPS 라 그대로 동작합니다.
- 마이크 컴포넌트는 Streamlit Custom Component **v2**(`web/live_mic.*`)라 iframe 없이 앱 문서 안에서
  `getUserMedia` · `AudioWorklet` · `WebSocket` 을 씁니다. 부분 전사는 브라우저가 직접 그리고,
  세션이 끝날 때만 결과를 Python 으로 올려 리런을 1회로 줄입니다.

---

## 8. 평가와 벤치마크

### 8.1 지표

| 지표 | 계산 위치 | 의미 |
|---|---|---|
| RTF | `metrics/latency.py` | 추론 시간 / 오디오 길이. **1.0 미만**이어야 실시간 |
| First partial latency | 〃 | 발화 시작 → 첫 partial |
| Final latency | 〃 | endpoint → final 확정 |
| Revision rate | `transcript/merger.py` | 이미 보여준 partial 이 뒤집힌 비율 |
| CPU / 메모리 / GPU | `metrics/latency.py` | psutil, torch |
| WER / CER | `metrics/evaluator.py` | 한국어는 CER 이 더 신뢰할 만함 |
| 의료용어 정확도 | 〃 | 정답 속 **의학용어 · 영문 약어** 중 전사에 살아남은 비율 |

### 8.2 데이터셋 평가 결과

`dataset/sound_data/` 응급실 예진 낭독 음성 **50개(총 43분, 파일당 31~75 s)** 를 4개 엔진에 통과시킨 결과입니다
(`dataset/stt_all_summary.csv`, `dataset/evaluation.csv`). Whisper 는 `small`, `language=ko` 이며
100 ms 청크를 **가능한 한 빠르게** 투입하는 accelerated 모드로 측정했습니다.

| 엔진 | WER | CER | 의료용어 정확도 | 평균 RTF | 실시간 통과 파일 |
|---|---|---|---|---|---|
| **Fun-ASR-MLT-Nano** | **0.54** | **0.44** | **0.65** (330/504) | 0.42 | 50/50 |
| Whisper `small` | 0.61 | 0.48 | 0.54 (273/504) | 0.67 | 42/50 |
| SenseVoice | 0.70 | 0.47 | 0.40 (200/504) | **0.02** | 50/50 |
| Zipformer | 0.96 | 0.93 | 0.01 (6/504) | 0.03 | 50/50 |

- accelerated 모드에서는 큐 대기가 섞이므로 **지연(ms) 값은 실제 라이브 지연이 아닙니다.** 라이브 지연은 `--paced` 로 측정하세요.
- 측정 장치 정보는 결과 파일에 기록돼 있지 않습니다. 장치를 바꾸면 RTF 는 크게 달라집니다.
- 절대 정확도가 모두 낮은 편이라, 의료 도메인 적용에는 용어 사전 보강이나 fine-tuning 이 필요합니다.

### 8.3 스크립트

| 명령 | 용도 |
|---|---|
| `python -m scripts.benchmark rec.wav --realtime --reference ref.txt` | WAV 하나를 여러 엔진으로 비교. 결과는 `transcripts/` → 성능 비교 페이지 |
| `python -m scripts.evaluate_dataset [--paced] [--engines ...] [--file-ids ...]` | 데이터셋 전체 평가 → `dataset/evaluation.csv` |
| `python -m scripts.merge_evaluations a.csv b.csv --output out.csv` | 엔진별로 나눠 돌린 평가 결과 병합 |
| `python -m scripts.build_stt_test_csv` | 정답 CSV + 엔진별 전사·지표를 한 장의 표(`STT_test_50.csv`)로 |
| `python -m scripts.reevaluate_stt_test_csv` | ASR 재실행 없이 저장된 전사를 현재 평가 규칙으로 재채점 |
| `python -m scripts.build_stt_all_summary` | 모델별 요약(`stt_all_summary.csv`) |

`--realtime` / `--paced` 는 오디오를 실제 속도로 흘려 넣습니다. **지연 지표는 이때만 의미가 있습니다.**

### 8.4 튜닝하며 알게 된 것

- **무음에 `initial_prompt` 를 붙여 디코딩하면 Whisper 가 10 s 넘게 헛돕니다.** warmup 은 프롬프트 없이 돌리고,
  세션은 무음 윈도우 추론을 건너뜁니다. 이로써 `small` warmup 13.3 s → 0.9 s, final 지연 9.1 s → 0.8 s.
- **네이티브 스트리밍 엔진은 같은 가설을 반복해서 줍니다.** 그대로 병합하면 LocalAgreement 가 저절로 성립해
  자라는 중인 어절이 중복 확정됩니다(`척 척할려고`). 가설이 바뀔 때만 병합합니다.
- **발화 전체를 재인식하는 엔진에 겹침 제거를 쓰면 중복됩니다.** 교체 경로(`replace_text`)로 분리했습니다.
- Whisper 모델 크기(Apple Silicon CPU int8, 5.4 s 발화): `tiny` RTF 0.10 오인식 / `base` 0.24 환각 / `small` 0.44 정답 일치.

---

## 9. 프로젝트 구조

```
streamlit_demo/
├── server/                      # ── STT 서버 (FastAPI) ──
│   ├── main.py                  # 앱, lifespan(preload), REST /api/v1/stt/*, WS 라우터 등록
│   ├── runtime.py               # 활성 설정(source of truth) · preload · readiness 상태
│   ├── schemas.py               # REST 응답 스키마 (Swagger UI 문서·예시)
│   ├── backend_ws.py            # /ws/v1/stt/stream — External Backend 어댑터
│   └── websocket.py             # /ws/v1/stt/browser + 세션 공용 헬퍼(open/close_session)
│
├── stt/                         # ── STT 코어 (서버·Streamlit·스크립트 공용) ──
│   ├── config.py                # 오디오 규격, 경로, API base path, StreamConfig
│   ├── session.py               # StreamingSession (파이프라인 전체)
│   ├── asr/
│   │   ├── base.py              # ASREngine 인터페이스 + create_engine()
│   │   ├── whisper.py           # faster-whisper
│   │   ├── zipformer.py         # sherpa-onnx streaming zipformer
│   │   ├── sensevoice.py        # sherpa-onnx SenseVoice
│   │   └── funasr_mlt_nano.py   # Fun-ASR-MLT-Nano-2512
│   ├── audio/                   # buffer(윈도우) · recorder(WAV) · resampler · vad
│   ├── transcript/              # merger(병합) · medical_terms(용어 사전·교정)
│   └── metrics/                 # latency(RTF·지연·자원) · evaluator(WER/CER/의료용어)
│
├── streamlit_app.py             # Streamlit 진입점
├── app_pages/                   # realtime · file_stt · metrics
├── web/                         # live_mic.{py,js,html,css} — 브라우저 마이크 컴포넌트
│
├── serverctl.sh                 # 상시 구동 관리 (setsid nohup · PID · 로그 · 워치독)
├── start_server.sh              # 포그라운드 실행 (uvicorn server.main:app --workers 1)
├── scripts/                     # stt_client(Real E2E) · fetch_models · benchmark · evaluate_dataset 외
├── tests/                       # test_pipeline · test_backend_ws · test_runtime · test_api_docs · test_streamlit_pages
│
├── models/                      # 모델 파일 (zipformer · sensevoice · funasr_mlt_nano)
├── recordings/                  # 세션별 WAV
├── transcripts/                 # 세션별 전사 (.json / .txt)
└── dataset/                     # 평가 음성·정답·결과 CSV
```

### 새 엔진 추가하기

1. `stt/asr/<name>.py` 에 `ASREngine` 을 상속해 `transcribe()` 를 구현합니다
   (스트리밍 엔진이면 `accept_waveform` / `partial` / `is_endpoint` / `reset_stream` 도).
2. `native_streaming` · `decodes_full_utterance` · `has_word_timestamps` · `shared_model` 을 엔진 특성에 맞게 설정합니다
   (`shared_model=True` 는 가중치를 프로세스 캐시로 공유할 때만. 동시 추론 보호용 lock 도 공유해야 합니다).
3. `stt/asr/base.py` 의 `create_engine()` 과 `stt/config.py` 의 `ENGINE_CHOICES` 에 등록합니다.
4. `server/main.py` 의 `/engines` 준비 상태 확인에 분기를 추가합니다.

세션·병합기·API 는 수정할 필요가 없습니다.

---

## 10. 테스트

테스트는 두 종류로 나뉩니다.

| 종류 | 실행 | 엔진 | 확인하는 것 |
|---|---|---|---|
| **Unit / Protocol** | `pytest tests/` (45개) | 가짜 엔진 — 모델·GPU 불필요 | 프로토콜, 순서, readiness, 설정, 자원 정리 |
| **Real E2E** | `python -m scripts.stt_client <wav>` | 서버의 활성 엔진(기본 `funasr_mlt_nano`) | 실제 모델·GPU 추론, partial/final/timestamps/done |

```bash
pip install pytest httpx
pytest tests/
python -m tests.test_pipeline    # pytest 없이 파이프라인 테스트만
```

| 파일 | 검증 내용 |
|---|---|
| `tests/test_pipeline.py` | 병합기 중복·누락 회귀(window/overlap/지터 조합), 세션 end-to-end, 엔진별 병합 경로, WER/CER, 의료 용어 교정 |
| `tests/test_backend_ws.py` | `/stream` 프레임 수신·순서, partial/final 형식, `close → final → done` 순서, 잘못된 프레임, 오류 메시지 비노출, 비정상 종료 시 자원 정리, `/browser` 호환, base path |
| `tests/test_runtime.py` | liveness/readiness(LOADING→READY, 로드·warm-up 실패 시 NOT_READY), `active_engine`, preload·세션 설정 일치, 공유 모델 warm-up 생략, 로딩 중 연결, Fun-ASR 공유 lock 직렬화, 취소돼도 세션 정리 완료 |
| `tests/test_api_docs.py` | Swagger UI·ReDoc 제공 경로, `/docs` 리다이렉트, 모든 REST API 의 태그·요약·503/404 문서화, 응답 필드 유지, `STT_DOCS=0` |
| `tests/test_streamlit_pages.py` | Streamlit 세 페이지가 백엔드 없이 예외 없이 렌더링되는지, 실시간 전사 페이지의 WebSocket 주소 기본값 |

Unit 테스트는 `stt.session.create_engine`(세션)과 `server.runtime.create_engine`(preload)을 가짜 엔진으로
바꿔 끼워 모델 없이 돕니다. 실제 모델 다운로드·GPU 가 필요한 검증은 일반 `pytest` 에 넣지 않았습니다.

#### Real E2E — `scripts/stt_client.py`

```bash
./start_server.sh &                                          # 서버 (다른 터미널)
python -m scripts.stt_client recordings/sample.wav           # READY 대기 → 100 ms 실시간 전송 → close → done
python -m scripts.stt_client sample.wav \
  --url wss://<POD_ID>-8000.proxy.runpod.net/ws/v1/stt/stream  # RunPod 원격
```

partial/final 을 시각과 함께 출력하고, 마지막에 요약(`partials`, `finals`, `errors`, `done`)을 보여 줍니다.
`done` 과 final 을 하나 이상 받으면 `PASS`(exit 0)입니다. 16 kHz mono 가 아닌 WAV 는 변환해서 보냅니다.

---

## 11. RunPod 배포 참고

### 11.1 상시 구동 — VS Code·SSH 창을 닫아도 계속 동작

VS Code 터미널에서 `./start_server.sh` 를 그대로 실행하면 서버가 그 터미널 세션에 속해, 창을 닫거나
SSH 가 끊길 때 함께 종료됩니다(`&` 로 백그라운드에 보내도 같은 세션이라 정리될 수 있습니다).
운영에서는 **`serverctl.sh`** 로 띄웁니다. uvicorn 을 `setsid nohup` 으로 터미널 세션에서 분리해
실행하고, PID·로그·워치독을 관리합니다(`Claude outputs/RunPod_서버_상시구동_가이드.md` 기반).

```
프론트/백엔드 ──HTTPS/WSS──▶ RunPod 프록시  https://<POD_ID>-8000.proxy.runpod.net
                                     │
                                     ▼
                          Pod 컨테이너 :8000
                          └─ uvicorn server.main:app (0.0.0.0, --workers 1, setsid 분리, PPID=1)
                               └─ FastAPI 앱 (Fun-ASR 은 기동 시 1회 preload)

VS Code / SSH ──▶ Pod   ← 관리용 통로일 뿐, 위 요청 경로에 포함되지 않음
```

#### 실행

```bash
cd /workspace/streamlit_demo
./serverctl.sh start --wait         # 분리 기동 → READY 까지 대기 (모델 로드 로그는 logs 로)
./serverctl.sh watchdog-start       # (권장) 응답이 없으면 자동 재기동
./serverctl.sh status               # 상태 확인
```

이제 VS Code 창을 닫아도 서버는 계속 동작합니다. 정상이면 `status` 가 아래처럼 나옵니다.

```
서버   : 실행 중 PID=12345 PPID=1 SID=12345 (분리됨)
워치독 : 실행 중 PID=12400
로그   : /workspace/stt-server.log
외부   : https://<POD_ID>-8000.proxy.runpod.net/api/v1/stt/health/ready
         wss://<POD_ID>-8000.proxy.runpod.net/ws/v1/stt/stream
{"status":"READY","engine":"funasr_mlt_nano","device":"cuda","model_loaded":true, ...}
HTTP 200
```

`SID` 가 자기 `PID` 와 같으면 터미널에서 분리된 것입니다(워치독이 재기동한 서버는 `PPID` 가 워치독 PID).
`외부` 줄은 RunPod 이 넣어 주는 `RUNPOD_POD_ID` 가 있을 때 표시됩니다.

| 명령 | 동작 |
|---|---|
| `start [--wait]` | `setsid nohup` 으로 분리 기동. 이미 실행 중이면 아무것도 안 함. 포그라운드 서버가 포트를 쓰고 있으면 거부 |
| `stop` | 점검 모드 표시 → 워치독 정지 → SIGTERM 후 종료 확인(`STOP_TIMEOUT` 초과 시 강제 종료) |
| `restart [--wait]` | 종료가 끝난 것을 확인한 뒤 재기동 (포트·GPU 메모리 충돌 방지) |
| `status` | PID·PPID·SID·분리 여부, 워치독, 점검 모드, readiness 응답, 외부 URL |
| `logs` | `tail -f` 로 서버 로그 보기. Ctrl+C 하거나 창을 닫아도 서버에는 영향 없음 |
| `watchdog-start` / `watchdog-stop` | 30 초 간격으로 `/health/ready` 를 확인해 3 회 연속 실패하면 프로세스를 정리하고 재기동. 1 시간에 5 회 넘게 재기동하면 자동 복구를 멈춤 |

#### 창을 닫아도 살아 있는지 확인

1. `./serverctl.sh start --wait` 로 기동합니다.
2. VS Code 창을 완전히 닫습니다(또는 SSH 연결을 끊습니다).
3. **로컬 PC** 에서 호출합니다.

   ```bash
   curl -s https://<POD_ID>-8000.proxy.runpod.net/api/v1/stt/health/ready
   python -m scripts.stt_client sample.wav --url wss://<POD_ID>-8000.proxy.runpod.net/ws/v1/stt/stream
   ```

4. `"status":"READY"` 가 오고 전사가 `PASS` 면 성공입니다.

워치독 확인: `kill -9 $(cat /workspace/stt-server.pid)` 후 `tail -f /workspace/stt-watchdog.log` 에서
`헬스체크 실패 1/3 → 2/3 → 3/3 → 서버 재기동` 이 기록되고 READY 로 돌아오는지 봅니다
(30 s × 3 회 + 모델 로드 시간).

#### 설정 — 워치독 재기동에도 유지하려면 env 파일에

`STT_ENGINE=sensevoice ./serverctl.sh start` 처럼 명령 앞에 붙인 값은 **그 기동에만** 적용되고,
워치독은 자신이 시작될 때의 환경으로 재기동합니다. 계속 유지할 설정은 `/workspace/stt-server.env` 에 적습니다.
`serverctl.sh` 는 실행할 때마다 이 파일을 읽으므로 수동 기동·워치독 재기동 모두에 적용됩니다.

```bash
# /workspace/stt-server.env
STT_ENGINE=funasr_mlt_nano
STT_TIMESTAMPS=1
STT_CONFIG={"save_wav":false}
```

| 환경변수 | 기본값 | 의미 |
|---|---|---|
| `STT_BACKEND_PORT` | `8000` | 수신 포트. RunPod **Expose HTTP Ports** 와 같아야 함 |
| `STT_RUN_DIR` | `/workspace` (없으면 `./run`) | 로그 `stt-server.log`, PID `stt-server.pid`, 워치독 로그·PID, 점검 표시 파일 위치 |
| `STT_ENV_FILE` | `$STT_RUN_DIR/stt-server.env` | 영속 설정 파일 |
| `PYTHON` | `.venv/bin/python` → `python` (절대경로로 고정) | 서버를 실행할 인터프리터 |
| `STT_STARTUP_WAIT` | `300` | `--wait`·워치독 재기동 후 READY 를 기다리는 최대 시간(초) |
| `STT_STOP_TIMEOUT` | `30` | 정상 종료 대기(초). 넘기면 강제 종료 |
| `STT_WATCH_INTERVAL` / `STT_WATCH_FAILS` | `30` / `3` | 워치독 확인 간격(초) / 재기동까지 연속 실패 횟수 |
| `STT_HEALTH_URL` | `http://127.0.0.1:$PORT/api/v1/stt/health/ready` | 워치독·`--wait` 이 보는 주소. `STT_PRELOAD=0` 으로 운영하면 `/api/v1/stt/health` 로 바꿀 것 |
| `STT_LOG_MAX_MB` | `100` | 기동 시 로그가 이보다 크면 `stt-server.log.1` 로 넘김 |

#### 주의

- **기동 명령을 터미널에 직접 붙여 넣지 말고 반드시 `serverctl.sh` 로 실행합니다.** 대화형 셸에서는
  `setsid` 가 한 번 더 fork 해 PID 파일이 틀어지고 `status`·`stop`·워치독이 오작동합니다.
- **Pod 을 stop/재시작하면 서버는 자동으로 뜨지 않습니다.** `setsid nohup` 은 "창을 닫아도 유지"까지만
  보장합니다. 재시작 후 다시 실행하거나, Pod 의 **Container Start Command** 에 등록해 두세요.

  ```bash
  cd /workspace/streamlit_demo && ./serverctl.sh start && ./serverctl.sh watchdog-start
  ```

- 로그·PID·env 파일은 Pod 재생성에도 남는 `/workspace` 에 둡니다. 새 Pod 을 만들면 **Pod ID 가 바뀌어
  공개 URL 도 바뀌므로** 연동 상대(프론트·백엔드)에게 새 주소를 알려야 합니다.
- 재기동하면 진행 중이던 WebSocket 세션은 끊깁니다(`done` 없이 종료). 연동 쪽은 `/health/ready` 가 READY 가
  된 뒤 다시 연결하면 됩니다.
- 같은 Pod 에서 다른 서버(예: LLM 서버 `8000`)와 함께 돌리면 `STT_BACKEND_PORT=8001` 처럼 포트를 나누고
  그 포트도 HTTP 로 노출하세요. GPU 메모리 합계가 VRAM 을 넘지 않는지도 확인합니다.
- RunPod 컨테이너에는 보통 systemd 가 없어 서비스 등록 대신 이 스크립트 + 워치독 방식을 씁니다.
  `./start_server.sh` 는 개발용 포그라운드 실행으로 남겨 두었습니다(`serverctl.sh` 도 내부에서 이 스크립트를 실행).

### 11.2 그 밖의 배포 참고

- **Pod 로 운영합니다.** 장시간 양방향 WebSocket 과 모델 상주가 필요하기 때문입니다.
- 기동하면 `[STT] Active engine` → `Loading model...` → `Model loaded` → `Warm-up started` →
  `Warm-up completed` → `Server READY` 로그가 차례로 나옵니다(`./serverctl.sh logs`). 모델 초기화가 실패하면
  프로세스는 살아 있되 readiness 가 `NOT_READY`(503) 를 보고하고 원인은 `[STT] Model preload failed` 로그에 남습니다.
  워치독을 켜 두었다면 이 경우 재기동을 시도합니다(1 시간 5 회까지).
- 포트 **8000**(STT 서버), 필요하면 **8501**(Streamlit)을 HTTP 포트로 노출합니다.
  프록시 주소는 `https://<podId>-<port>.proxy.runpod.net` 이며 TLS 를 대신 처리합니다.
- **모델을 영속 볼륨에 둡니다.** 저장소를 `/workspace` 아래에 두면 `models/` 가 재시작 후에도 남습니다.
  Whisper 는 HF 캐시를 쓰므로 `HF_HOME=/workspace/hf_cache` 를 지정하세요.
- 모델은 배포 단계에서 미리 받습니다(`python -m scripts.fetch_models`).
- GPU 에서 sherpa-onnx 를 쓰려면 CUDA 휠로 바꿉니다. CPU 휠에 `provider="cuda"` 를 주면 경고만 내고 CPU 로 돕니다.

  ```bash
  pip install sherpa-onnx -f https://k2-fsa.github.io/sherpa/onnx/cuda.html
  python -m scripts.fetch_models --keep all --force
  ```

- uvicorn 은 **worker 1개**로 띄웁니다. 여러 개면 모델이 GPU 에 중복으로 올라가고 세션 목록이 프로세스별로 갈립니다.
  확장은 Pod 를 늘려서 합니다.
- 헬스체크: 프로세스 생존은 `/api/v1/stt/health`, 트래픽 투입 판단은 `/api/v1/stt/health/ready` 를 씁니다.
- 배포 직후 브라우저로 `https://<POD_ID>-8000.proxy.runpod.net/api/v1/stt/docs` 를 열어 `health/ready`,
  `engines` 를 Execute 하면 상태를 바로 확인할 수 있습니다.

---

## 12. 현재 제약과 남은 과제

| 항목 | 현재 상태 | 영향 |
|---|---|---|
| 인증 | `/ws/v1/stt/*`, `/api/v1/stt/*`, Swagger UI 모두 **없음** | 프록시 URL 을 알면 누구나 전사·**녹음 다운로드** 가능. 운영 전 필수 (당장은 `STT_DOCS=0`) |
| 녹음·전사 저장 | WAV 는 `save_wav` 로 끌 수 있으나 전사 JSON 은 항상 저장 | 환자 음성·대화가 서버 디스크에 남음 |
| 모델 로드 | 활성 엔진은 시작 시 preload. Whisper·Fun-ASR 은 공유, **sherpa 엔진은 연결마다 새로 로드** | sherpa 엔진은 연결 직후 지연, 동시 접속 시 메모리 증가 |
| 동시 접속 | 세션 수 제한 없음. Fun-ASR 추론은 공유 lock 으로 직렬화 | 세션이 많으면 순서 대기로 partial/final 이 늦어짐 |
| VAD 노이즈 플로어 | 적응형 플로어가 조용한 음성(−30 dB 안팎)에 끌려 올라갈 수 있음 | 긴 녹음 후반의 음성을 무음으로 보고 건너뜀. 마이크 입력이 작으면 `vad_threshold_db` 를 낮추거나 `webrtcvad` 설치 |
| backpressure | 입력 큐 무제한 | 추론이 실시간보다 느리면 지연이 계속 늘어남 |
| 재연결 | 이어 받기 없음 | 끊기면 진행 중 발화 유실, 시각 0 부터 재시작 |
| 화자 분리 | 범위 밖 | 의료진·환자가 한 마이크를 공유 |
| 도메인 정확도 | CER 0.44~0.93 | 용어 사전 보강 · fine-tuning 필요 |

### 개발 현황

| 단계 | 상태 |
|---|---|
| Web Audio + WebSocket (PCM16 / 16 kHz / WAV 저장) | 완료 |
| RunPod 배포 (브라우저 수음 + 서버 추론 분리) | 완료 |
| Whisper baseline (sliding window + overlap) | 완료 |
| Transcript merge (stable/unstable, overlap dedup) | 완료 · 회귀 테스트 |
| Metrics (RTF, latency, 자원) | 완료 |
| Zipformer / SenseVoice / Fun-ASR-MLT-Nano 엔진 | 완료 |
| 의료 데이터셋 50건 비교 평가 | 완료 (`dataset/`) |
| External Backend API (`/ws/v1/stt/stream`) | 완료 · 통합 테스트 |
| 엔진 사전 로드 · liveness/readiness 분리 · `start_server.sh` | 완료 · 테스트 |
| 인증 · 동시 접속 제한 | 예정 |
