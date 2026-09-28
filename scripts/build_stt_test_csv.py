"""Build a wide, human-readable 50-file ASR comparison CSV.

The original ``STT_text_50.csv`` columns are retained and each engine's
transcript and metrics are appended.  ``evaluation.csv`` is the normalized
long-form source; this file is the requested spreadsheet-friendly final view.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
METRICS = ("hypothesis_text", "wer", "cer", "medical_term_recall", "end_to_end_rtf", "mean_rtf", "realtime_capable", "error")


def load_csv(path: Path, encoding: str) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames or []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, default=DATASET / "STT_text_50.csv")
    parser.add_argument("--evaluation", type=Path, default=DATASET / "evaluation.csv")
    parser.add_argument("--output", type=Path, default=DATASET / "STT_test_50.csv")
    args = parser.parse_args()

    references, original_fields = load_csv(args.reference, "cp949")
    evaluated, _ = load_csv(args.evaluation, "utf-8-sig")
    files = [row for row in evaluated if row["row_type"] == "file"]
    engines = sorted({row["engine"] for row in files})
    index = {(row["file_id"], row["engine"]): row for row in files}

    appended = [f"{engine}_{metric}" for engine in engines for metric in METRICS]
    output_rows: list[dict[str, str]] = []
    for row in references:
        out = dict(row)
        for engine in engines:
            result = index.get((row["index"], engine), {})
            for metric in METRICS:
                out[f"{engine}_{metric}"] = result.get(metric, "")
        output_rows.append(out)

    # A final summary line makes aggregate model comparisons visible without
    # opening the normalized evaluation CSV.
    summaries = {row["engine"]: row for row in evaluated if row["row_type"] == "summary"}
    total = {field: "" for field in original_fields}
    total["index"] = "__ALL__"
    total["챗GPT와 대화한 내용"] = "전체 50개 파일 성능 요약"
    for engine in engines:
        summary = summaries.get(engine, {})
        for metric in METRICS:
            key = f"{engine}_{metric}"
            # Summary metrics whose per-file names differ are intentionally
            # mapped here to retain a uniform spreadsheet column layout.
            total[key] = summary.get(metric, "")
    output_rows.append(total)

    fields = original_fields + appended
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {args.output} ({len(output_rows)} rows; engines: {', '.join(engines)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
