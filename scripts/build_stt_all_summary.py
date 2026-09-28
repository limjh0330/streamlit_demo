"""Create a per-model summary from stored index-level STT evaluations."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"


def engines(fields: list[str]) -> list[str]:
    suffix = "_hypothesis_text"
    return sorted(field.removesuffix(suffix) for field in fields if field.endswith(suffix))


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DATASET / "STT_test_50.csv")
    parser.add_argument("--output", type=Path, default=DATASET / "stt_all_summary.csv")
    args = parser.parse_args()

    with args.input.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        all_rows = list(reader)
    rows = [row for row in all_rows if row.get("index") != "__ALL__"]
    overall = next((row for row in all_rows if row.get("index") == "__ALL__"), {})
    model_names = engines(fields)
    if not rows or not model_names:
        raise ValueError("STT_test_50.csv must contain file rows and *_hypothesis_text columns")

    output_rows: list[dict[str, str | int]] = []
    for engine in model_names:
        accuracies = [float(row[f"{engine}_medical_term_accuracy"])
                      for row in rows if row.get(f"{engine}_medical_term_accuracy", "")]
        matched = sum(int(row[f"{engine}_medical_term_matched"]) for row in rows)
        total = sum(int(row[f"{engine}_medical_term_total"]) for row in rows)
        output_rows.append({
            "model": engine,
            "indexes_evaluated": len(rows),
            "indexes_with_medical_terms": len(accuracies),
            "hypotheses_present": sum(bool(row[f"{engine}_hypothesis_text"]) for row in rows),
            "wer_index_mean": f"{mean([float(row[f'{engine}_wer']) for row in rows]):.4f}",
            "cer_index_mean": f"{mean([float(row[f'{engine}_cer']) for row in rows]):.4f}",
            "wer_micro": overall.get(f"{engine}_wer", ""),
            "cer_micro": overall.get(f"{engine}_cer", ""),
            "medical_accuracy_index_mean": f"{mean(accuracies):.4f}" if accuracies else "",
            "medical_accuracy_micro": f"{matched / total:.4f}" if total else "",
            "medical_terms_matched": matched,
            "medical_terms_total": total,
        })

    fields = list(output_rows[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {args.output} ({len(output_rows)} models)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
