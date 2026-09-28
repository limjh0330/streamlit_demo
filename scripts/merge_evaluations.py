"""Merge per-engine outputs made by ``evaluate_dataset`` into one CSV."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from stt.metrics.evaluator import levenshtein, normalize_text
from stt.transcript.medical_terms import find_medical_accuracy_terms


def summary(engine: str, rows: list[dict[str, str]]) -> dict[str, str | float | int]:
    ref_words = [normalize_text(r["reference_text"]).split() for r in rows]
    hyp_words = [normalize_text(r["hypothesis_text"]).split() for r in rows]
    ref_chars = [list(normalize_text(r["reference_text"], keep_space=False)) for r in rows]
    hyp_chars = [list(normalize_text(r["hypothesis_text"], keep_space=False)) for r in rows]
    word_ops = levenshtein([x for group in ref_words for x in group], [x for group in hyp_words for x in group])
    char_ops = levenshtein([x for group in ref_chars for x in group], [x for group in hyp_chars for x in group])
    ref_terms = [set(find_medical_accuracy_terms(r["reference_text"])) for r in rows]
    hyp_terms = [set(find_medical_accuracy_terms(r["hypothesis_text"])) for r in rows]
    term_total = sum(len(x) for x in ref_terms)
    term_match = sum(len(ref & hyp) for ref, hyp in zip(ref_terms, hyp_terms))
    duration = sum(float(r["audio_sec"]) for r in rows)
    wall = sum(float(r["processing_wall_sec"]) for r in rows)
    realtime_passes = sum(float(r["end_to_end_rtf"]) <= 1 for r in rows)
    return {
        "row_type": "summary", "engine": engine, "file_id": "__ALL__", "files": len(rows),
        "audio_sec": round(duration, 3), "processing_wall_sec": round(wall, 3),
        "end_to_end_rtf": round(wall / duration, 4),
        "mean_rtf": round(sum(float(r["mean_rtf"]) for r in rows) / len(rows), 4),
        "wer": round(word_ops.error_rate, 4), "cer": round(char_ops.error_rate, 4),
        "medical_term_recall": round(term_match / term_total, 4) if term_total else "",
        "realtime_file_pass_rate": round(realtime_passes / len(rows), 4),
        "realtime_capable": "yes" if wall <= duration else "no",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    fields: set[str] = set()
    for path in args.inputs:
        with path.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if row["row_type"] == "file":
                    rows.append(row)
                    fields.update(row)
    engines = sorted({row["engine"] for row in rows})
    summaries = [summary(engine, [r for r in rows if r["engine"] == engine]) for engine in engines]
    fields.update(key for row in summaries for key in row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(fields))
        writer.writeheader()
        writer.writerows(rows + summaries)
    print(f"Wrote {args.output} ({len(rows) + len(summaries)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
