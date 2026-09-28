"""Evaluate every dataset/sound_data audio file with the three streaming ASR engines.

The CSV uses the Korean narration column (``한글 낭독본``) as the ground truth.
Audio is delivered to :class:`StreamingSession` in 100-ms chunks, the same size
used by the browser client.  By default chunks are supplied as quickly as the
engine can consume them; this makes a complete reproducible dataset run
practical while RTF remains the real-time suitability measure.  Use
``--paced`` to also sleep for each chunk and measure wall-clock live latency.

    .venv/bin/python -m scripts.evaluate_dataset
    .venv/bin/python -m scripts.evaluate_dataset --paced
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import av
import numpy as np

from stt.config import BLOCK_SIZE, ENGINE_CHOICES, SAMPLE_RATE, StreamConfig
from stt.metrics.evaluator import evaluate, levenshtein, normalize_text
from stt.session import StreamingSession


ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"


def load_audio(path: Path) -> np.ndarray:
    """Decode an m4a to 16-kHz mono float32 without depending on ffmpeg CLI."""
    parts: list[np.ndarray] = []
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        resampler = av.audio.resampler.AudioResampler(
            format="fltp", layout="mono", rate=SAMPLE_RATE
        )
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                parts.append(np.asarray(out.to_ndarray(), dtype=np.float32).reshape(-1))
        for out in resampler.resample(None):
            parts.append(np.asarray(out.to_ndarray(), dtype=np.float32).reshape(-1))
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)


def references(path: Path) -> dict[str, str]:
    with path.open(encoding="cp949", newline="") as f:
        return {row["index"].strip(): row["한글 낭독본"].strip() for row in csv.DictReader(f)}


def transcript(snapshot: dict) -> str:
    return " ".join(u["text"] for u in snapshot["utterances"]).strip() or snapshot["stable"]


def run_one(engine: str, audio: np.ndarray, *, paced: bool) -> tuple[dict, float]:
    session = StreamingSession(
        StreamConfig(engine=engine, language="ko", save_wav=False),
        session_id=f"evaluation-{engine}",
    )
    session.start()
    session.engine.warmup()
    started = time.perf_counter()
    for start in range(0, audio.size, BLOCK_SIZE):
        block = audio[start : start + BLOCK_SIZE]
        session.feed(block, sample_rate=SAMPLE_RATE)
        if paced:
            time.sleep(block.size / SAMPLE_RATE)
    snapshot = session.stop(timeout=120.0)
    return snapshot, time.perf_counter() - started


def summary_row(engine: str, rows: list[dict]) -> dict:
    refs = [normalize_text(row["reference_text"]).split() for row in rows]
    hyps = [normalize_text(row["hypothesis_text"]).split() for row in rows]
    ref_chars = [list(normalize_text(row["reference_text"], keep_space=False)) for row in rows]
    hyp_chars = [list(normalize_text(row["hypothesis_text"], keep_space=False)) for row in rows]
    word_ops = levenshtein([x for xs in refs for x in xs], [x for xs in hyps for x in xs])
    char_ops = levenshtein([x for xs in ref_chars for x in xs], [x for xs in hyp_chars for x in xs])
    durations = sum(float(row["audio_sec"]) for row in rows)
    wall = sum(float(row["processing_wall_sec"]) for row in rows)
    return {
        "row_type": "summary",
        "engine": engine,
        "file_id": "__ALL__",
        "files": len(rows),
        "audio_sec": round(durations, 3),
        "processing_wall_sec": round(wall, 3),
        "end_to_end_rtf": round(wall / durations, 4) if durations else "",
        "wer": round(word_ops.error_rate, 4),
        "cer": round(char_ops.error_rate, 4),
        "mean_rtf": round(sum(float(r["mean_rtf"]) for r in rows) / len(rows), 4),
        "realtime_capable": "yes" if wall <= durations else "no",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paced", action="store_true", help="Replay every chunk in real time (about 129 min total).")
    parser.add_argument("--engines", nargs="+", choices=ENGINE_CHOICES, default=list(ENGINE_CHOICES))
    parser.add_argument("--output", type=Path, default=DATASET / "evaluation.csv")
    parser.add_argument("--file-ids", nargs="+", help="Evaluate only these numeric dataset IDs (smoke test / resume).")
    args = parser.parse_args()

    ref_by_id = references(DATASET / "STT_text_50.csv")
    files = sorted((DATASET / "sound_data").glob("*.m4a"), key=lambda p: int(p.stem))
    if set(p.stem for p in files) != set(ref_by_id):
        raise ValueError("Audio file IDs and STT_text_50.csv indexes do not match")
    if args.file_ids:
        wanted = set(args.file_ids)
        unknown = wanted - set(ref_by_id)
        if unknown:
            raise ValueError(f"Unknown dataset IDs: {sorted(unknown)}")
        files = [path for path in files if path.stem in wanted]

    rows: list[dict] = []
    for engine in args.engines:
        print(f"[{engine}] loading model", flush=True)
        for num, path in enumerate(files, start=1):
            audio = load_audio(path)
            snapshot, wall = run_one(engine, audio, paced=args.paced)
            hyp = transcript(snapshot)
            scores = evaluate(ref_by_id[path.stem], hyp)
            metrics = snapshot["metrics"]
            row = {
                "row_type": "file",
                "test_mode": "paced_realtime" if args.paced else "accelerated_streaming_100ms",
                "engine": engine,
                "file_id": path.stem,
                "audio_file": str(path.relative_to(ROOT)),
                "files": 1,
                "audio_sec": round(audio.size / SAMPLE_RATE, 3),
                "processing_wall_sec": round(wall, 3),
                "end_to_end_rtf": round(wall / (audio.size / SAMPLE_RATE), 4),
                "mean_rtf": metrics["rtf_mean"],
                "first_partial_ms": metrics["first_partial_ms"],
                "final_latency_ms": metrics["final_latency_ms"],
                "revision_rate": metrics["revision_rate"],
                "utterances": metrics["utterances"],
                "wer": scores["wer"],
                "cer": scores["cer"],
                "medical_term_recall": scores["medical"]["recall"],
                "reference_text": ref_by_id[path.stem],
                "hypothesis_text": hyp,
                "error": snapshot["error"] or "",
                "realtime_capable": "yes" if wall <= audio.size / SAMPLE_RATE else "no",
            }
            rows.append(row)
            print(f"[{engine}] {num:02d}/{len(files)} {path.name}: CER={scores['cer']:.4f}, RTF={row['end_to_end_rtf']:.3f}", flush=True)

    output_rows = rows + [summary_row(engine, [r for r in rows if r["engine"] == engine]) for engine in args.engines]
    fields = sorted({key for row in output_rows for key in row})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {args.output} ({len(output_rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
