"""Session Manager — 오디오 한 줄기를 전사문으로 바꾸는 파이프라인.

    feed(audio)
        ├─→ Audio Recorder ──→ recordings/<id>.wav
        └─→ Audio Queue
                ↓   (워커 스레드)
            VAD / Endpoint
                ↓
            Sliding Window Buffer      (Whisper / SenseVoice)
              또는 accept_waveform      (Zipformer)
                ↓
            ASR Engine
                ↓
            Transcript Merger  →  stable / partial
                ↓
            events (partial / final / metrics)

`feed()` 는 PortAudio 콜백이나 WebSocket 핸들러에서 호출되므로 절대 블록하지 않는다.
무거운 일은 전부 워커 스레드에서 한다.
"""
from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .asr.base import ASREngine, ASRResult, create_engine
from .audio.buffer import SlidingWindowBuffer
from .audio.recorder import WavRecorder
from .audio.resampler import pcm16_to_float32, resample, rms_dbfs, to_mono
from .audio.vad import EndpointDetector
from .config import SAMPLE_RATE, TRANSCRIPTS_DIR, StreamConfig
from .metrics.latency import LatencyTracker, ResourceSampler
from .transcript.medical_terms import correct
from .transcript.merger import TranscriptMerger, Utterance


@dataclass
class Event:
    """세션이 밖으로 내보내는 메시지."""

    type: str                       # ready | partial | final | metrics | error
    data: dict = field(default_factory=dict)
    t: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps({"type": self.type, **self.data}, ensure_ascii=False)


class StreamingSession:
    """스레드 안전한 실시간 전사 세션."""

    def __init__(
        self,
        config: StreamConfig | None = None,
        session_id: str | None = None,
        engine: ASREngine | None = None,
        on_event: Callable[[Event], None] | None = None,
    ) -> None:
        self.config = config or StreamConfig()
        self.session_id = session_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.on_event = on_event

        self.engine: ASREngine | None = engine
        self.merger = TranscriptMerger()
        self.latency = LatencyTracker()
        self.resources = ResourceSampler()
        self.endpointer = EndpointDetector(
            silence_sec=self.config.silence_sec,
            max_utterance_sec=self.config.max_utterance_sec,
            threshold_db=self.config.vad_threshold_db,
        )
        self.buffer = SlidingWindowBuffer(
            window_sec=self.config.window_sec,
            overlap_sec=self.config.overlap_sec,
            min_window_sec=self.config.min_window_sec,
            first_hop_sec=self.config.first_hop_sec,
        )
        self.recorder: WavRecorder | None = None

        self._in: "queue.Queue[np.ndarray | None]" = queue.Queue()
        self.events: "queue.Queue[Event]" = queue.Queue()
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

        self.level_db = -120.0
        self.is_speech = False
        self.started_at: float | None = None
        self.error: str | None = None
        self._last_partial = ""
        self._last_hypothesis: tuple[str, ...] | str | None = None

    # ------------------------------------------------------------------ 수명
    def start(self) -> None:
        if self._worker is not None:
            return
        try:
            if self.engine is None:
                self.engine = create_engine(self.config)
        except Exception as e:
            self.error = str(e)
            self._emit(Event("error", {"message": str(e)}))
            raise

        if self.engine.decodes_full_utterance:
            # 발화 전체를 매번 다시 인식하는 엔진: 윈도우를 최대 발화 길이만큼
            # 열어 두고(= 잘라내지 않고) refresh_sec 마다 갱신한다.
            self.buffer = SlidingWindowBuffer(
                window_sec=self.config.max_utterance_sec,
                overlap_sec=0.0,
                min_window_sec=self.config.min_window_sec,
                first_hop_sec=self.config.first_hop_sec,
                hop_sec=self.config.refresh_sec,
            )

        if self.config.save_wav:
            self.recorder = WavRecorder(self.session_id)

        self.started_at = time.time()
        self.resources.start()
        self._worker = threading.Thread(target=self._run, name=f"stt-{self.session_id}", daemon=True)
        self._worker.start()
        self._emit(
            Event(
                "ready",
                {
                    "session_id": self.session_id,
                    "engine": self.engine.name,
                    "config": self.config.to_dict(),
                },
            )
        )

    def stop(self, timeout: float = 20.0) -> dict:
        """남은 오디오를 마저 처리하고 결과 요약을 돌려준다."""
        if self._worker is None:
            return self.snapshot()
        self._in.put(None)                       # sentinel: 잔여 처리 후 종료
        self._worker.join(timeout=timeout)
        self._stop.set()
        self._worker = None
        self.resources.stop()
        if self.recorder is not None:
            self.recorder.close()
        # 종료 이벤트는 내보내지 않는다. stop() 이 요약을 그대로 반환하므로
        # 여기서 emit 하면 WebSocket 핸들러가 보내는 closed 와 중복된다.
        return self.snapshot()

    # ------------------------------------------------------------------ 입력
    def feed(self, audio, sample_rate: int = SAMPLE_RATE) -> None:
        """PCM16 바이트 또는 float32 배열을 넣는다. 논블로킹."""
        if isinstance(audio, (bytes, bytearray, memoryview)):
            block = pcm16_to_float32(audio)
        else:
            block = to_mono(np.asarray(audio, dtype=np.float32))
        if block.size == 0:
            return
        if sample_rate != SAMPLE_RATE:
            block = resample(block, sample_rate, SAMPLE_RATE)
        self._in.put(block)

    # ------------------------------------------------------------------ 워커
    def _run(self) -> None:
        assert self.engine is not None
        native = self.engine.native_streaming
        last_metrics = time.perf_counter()

        while True:
            block = self._in.get()
            if block is None:                    # 종료 요청: 잔여분 flush
                self._flush()
                break
            if self._stop.is_set():
                break

            if self.recorder is not None:
                self.recorder.write(block)

            self.level_db = rms_dbfs(block)
            endpoint, vad = self.endpointer.push(block)
            self.is_speech = vad.is_speech
            if vad.is_speech:
                self.latency.on_speech_start()

            try:
                if native:
                    endpoint = self._step_native(block) or endpoint
                else:
                    self._step_windowed(block, endpoint)
                if endpoint:
                    self._finalize()
            except Exception as e:                # 한 스텝 실패로 세션을 죽이지 않는다
                self.error = f"{type(e).__name__}: {e}"
                self._emit(Event("error", {"message": self.error}))

            now = time.perf_counter()
            if now - last_metrics > 2.0:
                last_metrics = now
                self._emit(Event("metrics", self.metrics()))

    # --- Whisper / SenseVoice: sliding window ------------------------------
    def _step_windowed(self, block: np.ndarray, endpoint: bool) -> None:
        self.buffer.push(block)
        if endpoint:
            return                                # _finalize 가 force 로 처리
        if not self.buffer.ready():
            return
        window = self.buffer.pop_window()
        if window is None:
            return
        audio, t0 = window
        self._infer(audio, t0, is_final=False)

    # --- Zipformer: native streaming --------------------------------------
    def _step_native(self, block: np.ndarray) -> bool:
        assert self.engine is not None
        self.engine.accept_waveform(block)
        result = self.engine.partial()
        self.latency.on_inference(result.audio_sec, result.infer_sec)
        self._merge_native(result)
        return self.engine.is_endpoint()

    def _merge_native(self, result: ASRResult) -> None:
        """스트리밍 가설을 병합기에 넣는다. 단, 바뀌었을 때만.

        네이티브 스트리밍 엔진은 블록마다(100 ms) 같은 가설을 다시 준다. 그대로
        넣으면 LocalAgreement-2 가 "두 번 연속 같았다"를 자동으로 만족해 아직
        자라는 중인 어절이 확정돼 버린다("척" 확정 후 "척할려고" 가 또 확정).
        가설이 실제로 바뀐 경우에만 넘겨서 합의 횟수가 의미를 갖게 한다.
        """
        if result.words:
            hypothesis = tuple(w.text for w in result.words)
            if hypothesis == self._last_hypothesis:
                return
            self._last_hypothesis = hypothesis
            self._publish(*self.merger.update(result.words))
        elif result.text:
            if result.text == self._last_hypothesis:
                return
            self._last_hypothesis = result.text
            self._publish(*self.merger.update_text(result.text))

    # --- 공통 --------------------------------------------------------------
    def _is_silent(self, audio: np.ndarray) -> bool:
        """윈도우 전체가 노이즈 플로어 수준이면 True.

        무음 구간에 ASR 을 돌리면 비용만 드는 게 아니라 Whisper 가 없는 말을
        지어낸다("고맙습니다" 등). 특히 initial_prompt 가 붙어 있으면 디코딩이
        수 초씩 헛돌기 때문에 아예 건너뛴다.
        """
        vad = self.endpointer.vad
        gate = max(vad.noise_db + vad.threshold_db, vad.ABSOLUTE_FLOOR_DB)
        return rms_dbfs(audio) < gate

    def _infer(self, audio: np.ndarray, t0: float, is_final: bool) -> None:
        assert self.engine is not None
        if self._is_silent(audio):
            return
        result: ASRResult = self.engine.transcribe(audio, t0=t0, is_final=is_final)
        self.latency.on_inference(result.audio_sec, result.infer_sec)
        if self.engine.has_word_timestamps and result.words:
            committed, partial = self.merger.update(result.words, window_start=t0)
        elif self.engine.decodes_full_utterance:
            committed, partial = self.merger.replace_text(
                result.text, start=t0, end=t0 + result.audio_sec
            )
        else:
            committed, partial = self.merger.update_text(
                result.text, start=t0, end=t0 + result.audio_sec
            )
        self._publish(committed, partial)

    def _publish(self, committed: str, partial: str) -> None:
        if committed or partial != self._last_partial:
            self._last_partial = partial
            self.latency.on_partial(partial or committed)
            self._emit(
                Event(
                    "partial",
                    {
                        "stable": self._post(self.merger.stable_text),
                        "partial": self._post(partial),
                        "committed": self._post(committed),
                    },
                )
            )

    def _finalize(self) -> None:
        """endpoint: 버퍼에 남은 오디오까지 인식하고 발화를 확정."""
        self.latency.on_endpoint()
        if self.engine is not None and self.engine.native_streaming:
            self._flush_native()
            self.engine.reset_stream()
        else:
            leftover = self.buffer.pop_window(force=True)
            if leftover is not None:
                audio, t0 = leftover
                self._infer(audio, t0, is_final=True)
            self.buffer.reset_to_now()

        utt: Utterance | None = self.merger.finalize()
        self.latency.on_final()
        self._last_partial = ""
        self._last_hypothesis = None
        if utt is None:
            return
        text = self._post(utt.text)
        self._emit(
            Event(
                "final",
                {
                    "text": text,
                    "start": round(utt.start, 2),
                    "end": round(utt.end, 2),
                    "index": len(self.merger.utterances) - 1,
                    "stable": self._post(self.merger.stable_text),
                },
            )
        )

    def _flush_native(self) -> None:
        """네이티브 스트리밍 엔진의 꼬리 토큰을 끌어낸다.

        endpoint 시점에는 마지막 어절의 토큰이 아직 디코더 안에 남아 있다
        ("같았다" 가 "같았" 로 잘린다). 짧은 무음을 흘려 넣으면 나온다.
        """
        assert self.engine is not None
        tail = np.zeros(int(SAMPLE_RATE * 0.3), dtype=np.float32)
        self.engine.accept_waveform(tail)
        self._merge_native(self.engine.partial())

    def _flush(self) -> None:
        """스트림 종료 시 남은 오디오/미확정 텍스트를 모두 확정."""
        while True:                               # 큐에 남은 블록 소진
            try:
                block = self._in.get_nowait()
            except queue.Empty:
                break
            if block is None:
                continue
            if self.recorder is not None:
                self.recorder.write(block)
            if self.engine is not None and self.engine.native_streaming:
                self.engine.accept_waveform(block)
            else:
                self.buffer.push(block)
        try:
            self._finalize()
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"

    def postprocess(self, text: str) -> str:
        """의료 용어 교정을 적용한 표시용 텍스트."""
        if not text:
            return ""
        return correct(text) if self.config.medical_correction else text

    #: 내부 호출용 별칭
    _post = postprocess

    def _emit(self, event: Event) -> None:
        self.events.put(event)
        if self.on_event is not None:
            try:
                self.on_event(event)
            except Exception:
                pass

    # ------------------------------------------------------------------ 조회
    def drain_events(self, limit: int = 200) -> list[Event]:
        out: list[Event] = []
        for _ in range(limit):
            try:
                out.append(self.events.get_nowait())
            except queue.Empty:
                break
        return out

    def metrics(self) -> dict:
        m = self.latency.summary()
        m.update(self.resources.summary())
        m["revision_rate"] = round(self.merger.revision_rate, 3)
        m["utterances"] = len(self.merger.utterances)
        m["audio_sec"] = round(self.buffer.elapsed_sec, 1)
        m["queue"] = self._in.qsize()
        m["level_db"] = round(self.level_db, 1)
        m["is_speech"] = bool(self.is_speech)
        if self.started_at:
            m["wall_sec"] = round(time.time() - self.started_at, 1)
        return m

    def snapshot(self) -> dict:
        return {
            "session_id": self.session_id,
            "engine": self.engine.name if self.engine else self.config.engine,
            "stable": self._post(self.merger.stable_text),
            "partial": self._post(self.merger.partial_text),
            "utterances": [
                {"text": self._post(u.text), "start": round(u.start, 2), "end": round(u.end, 2)}
                for u in self.merger.utterances
            ],
            "metrics": self.metrics(),
            "wav": str(self.recorder.path) if self.recorder else None,
            "error": self.error,
        }

    @property
    def full_text(self) -> str:
        return self._post(self.merger.full_text)

    def save_transcript(self, directory: Path = TRANSCRIPTS_DIR) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        snap = self.snapshot()
        (directory / f"{self.session_id}.json").write_text(
            json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        path = directory / f"{self.session_id}.txt"
        lines = [f"[{u['start']:.1f}s → {u['end']:.1f}s] {u['text']}" for u in snap["utterances"]]
        body = "\n".join(lines) or snap["stable"]
        path.write_text(body + "\n\n---\n" + self.full_text + "\n", encoding="utf-8")
        return path
