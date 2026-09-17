"""가짜 ASR 엔진으로 전사 파이프라인을 검증한다.

    python -m tests.test_pipeline      (pytest 없이도 실행 가능)
    pytest tests/
"""
from __future__ import annotations

import time

import numpy as np

from stt.asr.base import ASREngine, ASRResult, Word
from stt.config import SAMPLE_RATE, StreamConfig
from stt.metrics.evaluator import cer, evaluate, levenshtein, wer
from stt.session import StreamingSession
from stt.transcript.medical_terms import correct
from stt.transcript.merger import TranscriptMerger

GROUND_TRUTH = (
    "오늘 아침부터 배가 아팠습니다 구토도 두 번 했습니다 "
    "열도 조금 났습니다 어제 저녁부터 그랬습니다"
).split()
STEP = 0.45
TIMELINE = [Word(t, i * STEP, (i + 1) * STEP) for i, t in enumerate(GROUND_TRUTH)]


class MockEngine(ASREngine):
    """주어진 구간에 완전히 포함되는 단어만 돌려주는 결정적 엔진."""

    name = "mock"
    has_word_timestamps = True

    def transcribe(self, audio, t0: float = 0.0, is_final: bool = False) -> ASRResult:
        dur = audio.size / SAMPLE_RATE
        words = [
            Word(w.text, w.start, w.end)
            for w in TIMELINE
            if w.start >= t0 - 1e-6 and w.end <= t0 + dur + 1e-6
        ]
        return ASRResult(
            text=" ".join(w.text for w in words),
            words=words,
            audio_sec=dur,
            infer_sec=0.01,
            is_final=is_final,
        )


def _window(t0: float, t1: float, jitter: float = 0.0) -> list[Word]:
    return [
        Word(w.text, w.start + jitter, w.end + jitter)
        for w in TIMELINE
        if w.start >= t0 - 1e-6 and w.end <= t1 + 1e-6
    ]


def test_merger_reconstructs_transcript() -> None:
    """겹치는 윈도우 + 타임스탬프 흔들림에서도 중복/누락 없이 복원한다."""
    merger = TranscriptMerger()
    for t0, t1, jitter in [(0.0, 5.0, 0.0), (3.5, 8.5, 0.05), (7.0, 12.0, -0.04)]:
        merger.update(_window(t0, t1, jitter), window_start=t0)
    merger.finalize()
    assert merger.full_text == " ".join(GROUND_TRUTH), merger.full_text


def test_session_end_to_end() -> None:
    """마이크 블록 -> VAD -> 윈도우 -> ASR -> 병합 -> final 까지."""
    config = StreamConfig(silence_sec=0.6, save_wav=False, medical_correction=False)
    session = StreamingSession(config, session_id="__test__", engine=MockEngine(config))
    session.start()

    rng = np.random.default_rng(0)
    speech_blocks = int(len(GROUND_TRUTH) * STEP / 0.1)
    for _ in range(speech_blocks):
        session.feed((0.15 * rng.standard_normal(1600)).astype(np.float32))
    for _ in range(12):                       # 1.2 s 무음 -> endpoint
        session.feed((1e-5 * rng.standard_normal(1600)).astype(np.float32))
    time.sleep(0.3)
    snapshot = session.stop()

    assert snapshot["stable"] == " ".join(GROUND_TRUTH), snapshot["stable"]
    assert len(snapshot["utterances"]) >= 1
    assert snapshot["metrics"]["inferences"] > 0
    types = [e.type for e in session.drain_events(500)]
    assert "ready" in types and "final" in types


def test_merger_sweep_no_dropped_or_duplicated_words() -> None:
    """window/overlap/타임스탬프 지터 조합을 훑어 중복·누락 회귀를 잡는다."""
    import itertools
    import random

    long_gt = (
        "환자는 오늘 아침부터 심한 복통을 호소했고 구토를 세 번 했습니다 "
        "혈압은 백사십에 구십 이었고 산소포화도는 구십육 퍼센트 입니다 "
        "심전도 검사와 혈액검사를 시행했습니다"
    ).split()
    step = 0.42
    timeline = [Word(t, i * step, (i + 1) * step) for i, t in enumerate(long_gt)]
    total = len(long_gt) * step

    def run(win: float, overlap: float, jitter: float, seed: int) -> str:
        rng = random.Random(seed)
        merger = TranscriptMerger()
        t0 = 0.0
        while t0 < total:
            j = rng.uniform(-jitter, jitter)
            hypothesis = [
                Word(w.text, w.start + j, w.end + j)
                for w in timeline
                if w.start >= t0 - 1e-6 and w.end <= t0 + win + 1e-6
            ]
            merger.update(hypothesis, window_start=t0)
            t0 += win - overlap
        merger.update([], window_start=total + win)   # 스트림 종료 flush
        merger.finalize()
        return merger.full_text

    expected = " ".join(long_gt)
    failures = [
        combo
        for combo in itertools.product([4.0, 5.0, 6.0], [1.0, 1.5, 2.0], [0.0, 0.06, 0.15], range(5))
        if run(*combo) != expected
    ]
    assert not failures, failures[:3]


def test_revision_rate_counts_only_real_revisions() -> None:
    """이미 보여준 단어가 바뀌면 revision, 뒤에 덧붙기만 하면 revision 아님."""

    def hyp(spec: str, step: float = 0.4) -> list[Word]:
        return [Word(t, i * step, (i + 1) * step) for i, t in enumerate(spec.split())]

    # 아직 unstable 인 '구톄도' 가 다음 가설에서 '구토도' 로 정정된다
    revised = TranscriptMerger()
    for text in (
        "오늘 아침부터 배가 아팠습니다 구톄도 두 번",
        "오늘 아침부터 배가 아팠습니다 구토도 두 번 했습니다",
        "오늘 아침부터 배가 아팠습니다 구토도 두 번 했습니다 열도",
    ):
        revised.update(hyp(text), window_start=0.0)
    assert revised.revisions == 1, revised.revisions

    # 뒤에 덧붙기만 하는 경우
    grown = TranscriptMerger()
    for text in ("오늘 아침부터", "오늘 아침부터 배가", "오늘 아침부터 배가 아팠습니다"):
        grown.update(hyp(text), window_start=0.0)
    assert grown.revisions == 0, grown.revisions


def test_error_rates() -> None:
    reference = "오늘 아침부터 배가 아팠습니다"
    assert wer(reference, reference) == 0.0
    assert cer(reference, reference) == 0.0
    ops = levenshtein(list("abc"), list("axcd"))
    assert (ops.substitutions, ops.insertions, ops.deletions) == (1, 1, 0)
    scores = evaluate("복통과 구토가 있었습니다", "복통과 구토가 있었습니다")
    assert scores["medical"]["recall"] == 1.0


def test_medical_correction_keeps_particles() -> None:
    """어절 통째 치환으로 조사가 사라지면 안 된다."""
    assert correct("심금경색 의심되며 호흡 곤란과") == "심근경색 의심되며 호흡곤란과"
    assert correct("오늘 아침부터 배가 아팠습니다") == "오늘 아침부터 배가 아팠습니다"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  PASS  {name}")
    print("\n전체 통과")
