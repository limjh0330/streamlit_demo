"""같은 오디오를 여러 엔진으로 돌려 지표를 비교한다 (개발문서 7단계).

    python -m scripts.benchmark recordings/sample.wav
    python -m scripts.benchmark sample.wav --engines whisper sensevoice
    python -m scripts.benchmark sample.wav --realtime      # 실제 속도로 흘려 넣기
    python -m scripts.benchmark sample.wav --reference ref.txt

결과는 `transcripts/` 에 저장되므로 Streamlit **성능 비교** 페이지에서 바로
보입니다. 지연 지표(first partial / final latency)는 오디오를 실제 속도로
흘려 넣을 때만 의미가 있으므로 `--realtime` 을 쓰세요.
"""
from __future__ import annotations

import argparse
import time
import wave
from pathlib import Path

import numpy as np

from stt.config import SAMPLE_RATE, ENGINE_CHOICES, StreamConfig
from stt.metrics.evaluator import evaluate
from stt.session import StreamingSession


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"PCM16 WAV 만 지원합니다: {path}")
        rate, channels = w.getframerate(), w.getnchannels()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    audio = pcm.astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, rate


def run(engine: str, audio: np.ndarray, rate: int, *, model_size: str,
        language: str, realtime: bool, save_wav: bool) -> dict | None:
    config = StreamConfig(engine=engine, model_size=model_size,
                          language=language or None, save_wav=save_wav)
    session = StreamingSession(config)
    try:
        session.start()
        session.engine.warmup()
    except Exception as e:
        print(f"  {engine:11s} 건너뜀 — {type(e).__name__}: {e}")
        return None

    chunk = rate // 10                      # 100 ms (브라우저가 보내는 크기)
    started = time.perf_counter()
    for i in range(0, audio.size, chunk):
        session.feed(audio[i:i + chunk], sample_rate=rate)
        if realtime:
            time.sleep(chunk / rate)
    snapshot = session.stop()
    snapshot["wall_sec"] = round(time.perf_counter() - started, 2)
    session.save_transcript()
    return snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("wav", type=Path, help="PCM16 WAV 파일")
    parser.add_argument("--engines", nargs="+", default=list(ENGINE_CHOICES),
                        choices=list(ENGINE_CHOICES))
    parser.add_argument("--model-size", default="small", help="whisper 모델 크기")
    parser.add_argument("--language", default="ko")
    parser.add_argument("--realtime", action="store_true",
                        help="실제 속도로 흘려 넣기 (지연 지표를 재려면 필요)")
    parser.add_argument("--reference", type=Path, help="정답 전사문 (WER/CER 계산)")
    parser.add_argument("--no-save-wav", action="store_true")
    args = parser.parse_args()

    audio, rate = load_wav(args.wav)
    reference = args.reference.read_text(encoding="utf-8").strip() if args.reference else ""
    print(f"{args.wav}  {rate} Hz  {audio.size / rate:.1f} s"
          f"{'  (실시간 재생)' if args.realtime else ''}\n")

    rows = []
    for engine in args.engines:
        snapshot = run(engine, audio, rate, model_size=args.model_size,
                       language=args.language, realtime=args.realtime,
                       save_wav=not args.no_save_wav)
        if snapshot is None:
            continue
        m = snapshot["metrics"]
        row = {
            "engine": engine,
            "RTF": m["rtf_mean"],
            "wall(s)": snapshot["wall_sec"],
            "first_partial(ms)": m["first_partial_ms"],
            "final(ms)": m["final_latency_ms"],
            "revision": f"{m['revision_rate'] * 100:.0f}%",
            "발화": m["utterances"],
        }
        text = " ".join(u["text"] for u in snapshot["utterances"]) or snapshot["stable"]
        if reference:
            scores = evaluate(reference, text)
            row["WER"] = round(scores["wer"], 3)
            row["CER"] = round(scores["cer"], 3)
            row["의료용어"] = round(scores["medical"]["recall"], 3)
        rows.append(row)

        print(f"  {engine:11s} " + "  ".join(f"{k}={v}" for k, v in row.items()
                                             if k != "engine"))
        for u in snapshot["utterances"]:
            print(f"              [{u['start']:5.1f}→{u['end']:5.1f}] {u['text']}")
        if not snapshot["utterances"] and snapshot["stable"]:
            print(f"              {snapshot['stable']}")
        print()

    if not rows:
        print("돌릴 수 있는 엔진이 없습니다. python -m scripts.fetch_models 를 먼저 실행하세요.")
        return 1
    print("전사 결과를 transcripts/ 에 저장했습니다 — 성능 비교 페이지에서 볼 수 있습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
