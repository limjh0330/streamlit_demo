"""WER / CER / 의료 용어 정확도 평가.

동일 데이터셋으로 Whisper / Zipformer / SenseVoice 를 비교할 때 사용한다.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from ..transcript.medical_terms import find_terms

_PUNCT = re.compile(r"[^\w\s가-힣]", re.UNICODE)
_SPACE = re.compile(r"\s+")


def normalize_text(text: str, keep_space: bool = True) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    text = _PUNCT.sub(" ", text)
    text = _SPACE.sub(" ", text).strip()
    return text if keep_space else text.replace(" ", "")


@dataclass
class EditOps:
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    hits: int = 0

    @property
    def total_ref(self) -> int:
        return self.substitutions + self.deletions + self.hits

    @property
    def error_rate(self) -> float:
        if self.total_ref == 0:
            return 0.0
        return (self.substitutions + self.deletions + self.insertions) / self.total_ref


def levenshtein(ref: list, hyp: list) -> EditOps:
    """S/D/I 를 각각 세는 Levenshtein 정렬 (백트래킹 대신 op 카운트 전파)."""
    n, m = len(ref), len(hyp)
    # dp[j] = (cost, sub, dele, ins, hit)
    prev = [(j, 0, 0, j, 0) for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [(i, 0, i, 0, 0)] + [(0, 0, 0, 0, 0)] * m
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                c, s, d, ins, h = prev[j - 1]
                cur[j] = (c, s, d, ins, h + 1)
                continue
            sub = prev[j - 1]
            dele = prev[j]
            insr = cur[j - 1]
            best = min(sub[0], dele[0], insr[0]) + 1
            if sub[0] <= dele[0] and sub[0] <= insr[0]:
                cur[j] = (best, sub[1] + 1, sub[2], sub[3], sub[4])
            elif dele[0] <= insr[0]:
                cur[j] = (best, dele[1], dele[2] + 1, dele[3], dele[4])
            else:
                cur[j] = (best, insr[1], insr[2], insr[3] + 1, insr[4])
        prev = cur
    _, s, d, ins, h = prev[m]
    return EditOps(substitutions=s, deletions=d, insertions=ins, hits=h)


def wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate (어절 단위)."""
    return levenshtein(
        normalize_text(reference).split(), normalize_text(hypothesis).split()
    ).error_rate


def cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate. 한국어는 CER 이 더 신뢰할 만한 지표다."""
    return levenshtein(
        list(normalize_text(reference, keep_space=False)),
        list(normalize_text(hypothesis, keep_space=False)),
    ).error_rate


def medical_term_accuracy(reference: str, hypothesis: str) -> dict:
    """정답에 등장한 도메인 용어를 가설이 얼마나 살렸는지(recall)."""
    ref_terms = set(find_terms(reference))
    hyp_terms = set(find_terms(hypothesis))
    if not ref_terms:
        return {"recall": None, "matched": 0, "total": 0, "missed": []}
    matched = ref_terms & hyp_terms
    return {
        "recall": round(len(matched) / len(ref_terms), 3),
        "matched": len(matched),
        "total": len(ref_terms),
        "missed": sorted(ref_terms - matched),
    }


def evaluate(reference: str, hypothesis: str) -> dict:
    return {
        "wer": round(wer(reference, hypothesis), 4),
        "cer": round(cer(reference, hypothesis), 4),
        "medical": medical_term_accuracy(reference, hypothesis),
        "ref_words": len(normalize_text(reference).split()),
        "hyp_words": len(normalize_text(hypothesis).split()),
    }
