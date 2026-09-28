"""Re-evaluate stored hypothesis texts in ``dataset/STT_test_50.csv``.

This avoids re-running ASR.  The Korean narration column is the reference and
each ``*_hypothesis_text`` column is evaluated with the current WER/CER and
medical-term rules.  It is intended to be run after changes to the glossary or
evaluation policy.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from stt.metrics.evaluator import evaluate, levenshtein, normalize_text


ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
DETAIL_METRICS = (
    "wer", "cer", "medical_term_recall", "medical_term_accuracy",
    "medical_term_matched", "medical_term_total", "medical_term_missed",
)


def engines(fields: list[str]) -> list[str]:
    suffix = "_hypothesis_text"
    return sorted(field.removesuffix(suffix) for field in fields if field.endswith(suffix))


def aggregate(rows: list[dict[str, str]], engine: str) -> dict[str, str]:
    refs = [normalize_text(row["한글 낭독본"]).split() for row in rows]
    hyps = [normalize_text(row[f"{engine}_hypothesis_text"]).split() for row in rows]
    ref_chars = [list(normalize_text(row["한글 낭독본"], keep_space=False)) for row in rows]
    hyp_chars = [list(normalize_text(row[f"{engine}_hypothesis_text"], keep_space=False)) for row in rows]
    word_ops = levenshtein([word for row in refs for word in row], [word for row in hyps for word in row])
    char_ops = levenshtein([char for row in ref_chars for char in row], [char for row in hyp_chars for char in row])
    matched = sum(int(row[f"{engine}_medical_term_matched"]) for row in rows)
    total = sum(int(row[f"{engine}_medical_term_total"]) for row in rows)
    return {
        f"{engine}_wer": f"{word_ops.error_rate:.4f}",
        f"{engine}_cer": f"{char_ops.error_rate:.4f}",
        f"{engine}_medical_term_recall": f"{matched / total:.4f}" if total else "",
        f"{engine}_medical_term_accuracy": f"{matched / total:.4f}" if total else "",
        f"{engine}_medical_term_matched": str(matched),
        f"{engine}_medical_term_total": str(total),
        f"{engine}_medical_term_missed": "",  # file-level column holds the audit detail
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DATASET / "STT_test_50.csv")
    parser.add_argument("--output", type=Path, help="Default: overwrite --input")
    args = parser.parse_args()
    output = args.output or args.input

    with args.input.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        rows = list(reader)
    model_names = engines(fields)
    if not model_names:
        raise ValueError("No *_hypothesis_text columns found")

    for engine in model_names:
        for metric in DETAIL_METRICS:
            field = f"{engine}_{metric}"
            if field not in fields:
                fields.append(field)

    data_rows = [row for row in rows if row.get("index") != "__ALL__"]
    if not data_rows:
        raise ValueError("No file rows found")
    for row in data_rows:
        reference = row["한글 낭독본"]
        for engine in model_names:
            scores = evaluate(reference, row[f"{engine}_hypothesis_text"])
            medical = scores["medical"]
            row[f"{engine}_wer"] = f"{scores['wer']:.4f}"
            row[f"{engine}_cer"] = f"{scores['cer']:.4f}"
            row[f"{engine}_medical_term_recall"] = "" if medical["recall"] is None else f"{medical['recall']:.4f}"
            row[f"{engine}_medical_term_accuracy"] = row[f"{engine}_medical_term_recall"]
            row[f"{engine}_medical_term_matched"] = str(medical["matched"])
            row[f"{engine}_medical_term_total"] = str(medical["total"])
            row[f"{engine}_medical_term_missed"] = ", ".join(medical["missed"])

    summary = next((row for row in rows if row.get("index") == "__ALL__"), {field: "" for field in fields})
    summary["index"] = "__ALL__"
    summary["챗GPT와 대화한 내용"] = "저장된 모델 전사문 기준 전체 50개 파일 성능 요약"
    for engine in model_names:
        summary.update(aggregate(data_rows, engine))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(data_rows + [summary])
    print(f"Wrote {output} ({len(data_rows)} files; engines: {', '.join(model_names)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
