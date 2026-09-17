"""Sliding window 결과를 하나의 전사문으로 병합.

핵심 문제: window 가 1~2 s 겹치므로 같은 단어가 두 번 나온다. 그리고 Whisper 는
윈도우가 바뀌면 앞 단어를 뒤집어 쓰기도 한다(revision).

전략 — LocalAgreement-2 + 타임스탬프 중복 제거:

  1. 새 가설의 단어 중 이미 확정(commit)된 시각 이전 것은 버린다  → overlap dedup
  2. 남은 단어열을 직전 가설의 남은 단어열과 비교해 **공통 접두사**를 구한다
  3. 두 번 연속 같게 나온 접두사만 stable 로 확정(commit)
  4. 나머지는 unstable(partial) 로 화면에만 보여주고 다음 윈도우에서 다시 판단

     stable ────────────────┐  unstable ──────┐
     "오늘 아침부터 배가 아팠"   "습니다 구토도"

단어 타임스탬프가 없는 엔진(SenseVoice 등)은 텍스트 접미/접두 최장 일치로
overlap 을 제거하는 폴백 경로를 쓴다.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from ..asr.base import Word

_PUNCT = re.compile(r"[^\w가-힣]+", re.UNICODE)


def normalize(token: str) -> str:
    """비교 전용 정규화(구두점/대소문자/유니코드 형태 차이 제거)."""
    token = unicodedata.normalize("NFC", token)
    return _PUNCT.sub("", token).lower()


def join_words(words: list[Word]) -> str:
    """한국어는 어절 사이 공백, 붙는 구두점은 그대로 이어붙인다."""
    out = ""
    for w in words:
        t = w.text
        if out and not t.startswith(("'", ",", ".", "?", "!", "…")):
            out += " "
        out += t
    return out.strip()


@dataclass
class Utterance:
    """endpoint 로 확정된 하나의 발화."""

    text: str
    start: float
    end: float
    words: list[Word] = field(default_factory=list)

    def as_line(self) -> str:
        return f"[{self.start:.1f}s → {self.end:.1f}s] {self.text}"


class TranscriptMerger:
    """윈도우 가설들을 stable + unstable 로 병합한다."""

    #: 같은 단어로 볼 수 있는 윈도우 간 타임스탬프 오차 (초)
    ALIGN_TOLERANCE = 0.6

    def __init__(self, agreement: int = 2, commit_margin: float = 0.06) -> None:
        self.agreement = agreement          # 몇 번 연속 일치해야 확정할지
        self.commit_margin = commit_margin  # 타임스탬프 흔들림 허용치(초)

        self.committed: list[Word] = []     # 현재 발화의 확정 단어
        self.unstable: list[Word] = []      # 아직 흔들리는 꼬리
        self.utterances: list[Utterance] = []
        self.committed_time = 0.0

        self._prev: list[Word] = []         # 직전 가설(공통 접두사 비교용)
        self._prev_text = ""                # 타임스탬프 없는 엔진용 폴백
        self._last_display: list[str] = []  # 직전에 화면에 보여준 단어열
        self.revisions = 0                  # 이미 보여준 텍스트가 바뀐 횟수(지표)
        self.updates = 0

    # ------------------------------------------------------------------ API
    def update(
        self, words: list[Word], window_start: float | None = None
    ) -> tuple[str, str]:
        """새 윈도우 가설을 반영. (newly_committed, unstable) 반환.

        `window_start` 는 이 가설이 커버하는 구간의 절대 시작 시각. 가설이 비어
        있을 때(무음 구간 등) 그 앞의 unstable 을 흘려보내는 데 쓴다.
        """
        self.updates += 1
        if not words:
            flushed = self._commit(self._cut_by_time(window_start))
            return join_words(flushed), join_words(self.unstable)
        return self._update_with_words(words)

    # ------------------------------------------------------- orphan 처리
    def _commit(self, cut: int) -> list[Word]:
        """unstable 앞 `cut` 개를 확정으로 옮긴다."""
        if cut <= 0:
            return []
        expired, self.unstable = self.unstable[:cut], self.unstable[cut:]
        self.committed.extend(expired)
        self.committed_time = max(self.committed_time, expired[-1].end)
        self._prev = [
            w for w in self._prev if w.end > self.committed_time + self.commit_margin
        ]
        return expired

    def _cut_by_time(self, boundary: float | None) -> int:
        """`boundary` 보다 앞서 시작한 unstable 단어 수."""
        if boundary is None:
            return 0
        cut = 0
        for w in self.unstable:
            if w.start >= boundary - self.commit_margin:
                break
            cut += 1
        return cut

    def _orphan_cut(self, fresh: list[Word]) -> int:
        """새 가설이 이어받지 못하는 unstable 앞부분의 길이.

        새 윈도우는 fresh[0] 부터 시작하므로, unstable 에서 fresh[0] 에 해당하는
        위치 앞의 단어들은 다시 확인될 기회가 없다 -> 지금 확정해야 한다.
        위치는 **텍스트로** 찾는다. 시각만 보면 윈도우마다 수십 ms 씩 흔들리는
        타임스탬프 때문에 같은 단어를 앞 단어로 오인해 중복이 생긴다.
        """
        if not self.unstable or not fresh:
            return 0
        head = normalize(fresh[0].text)
        for i, w in enumerate(self.unstable):
            if normalize(w.text) == head and abs(w.start - fresh[0].start) <= self.ALIGN_TOLERANCE:
                return i
        return self._cut_by_time(fresh[0].start)

    def update_text(
        self, text: str, start: float | None = None, end: float | None = None
    ) -> tuple[str, str]:
        """단어 타임스탬프가 없는 엔진용: 텍스트 겹침만 제거해 이어 붙인다.

        타임스탬프가 없으니 시각으로 정렬할 수 없다. 대신 **한 윈도우 늦게**
        확정한다 — 새 윈도우가 도착하면 직전 윈도우의 꼬리는 더 바뀌지 않는다고
        보고 commit 하고, 이번에 새로 나온 부분만 unstable 로 둔다.

        `start`/`end` 는 이 가설이 커버하는 구간의 절대 시각. 없으면 0 이 되어
        발화의 시작/끝 시각을 표시할 수 없다.
        """
        self.updates += 1
        text = text.strip()
        if not text:
            return "", join_words(self.unstable)

        new_part = _strip_overlap(self._prev_text, text)
        self._prev_text = text
        if not new_part:
            return "", join_words(self.unstable)

        # 직전 윈도우의 꼬리를 확정한다. 이걸 빠뜨리면 unstable 을 덮어쓰면서
        # 발화 하나에서 마지막 윈도우의 텍스트만 남는다.
        promoted = self.unstable
        if promoted:
            self.committed.extend(promoted)
            self.committed_time = max(self.committed_time, promoted[-1].end)

        w_start = self.committed_time if start is None else start
        w_end = w_start if end is None else max(w_start, end)
        self.unstable = [Word(t, w_start, w_end) for t in new_part.split()]

        self._record_revision()
        return join_words(promoted), join_words(self.unstable)

    def replace_text(
        self, text: str, start: float | None = None, end: float | None = None
    ) -> tuple[str, str]:
        """발화 전체를 매번 다시 인식하는 엔진용: 가설을 통째로 교체한다.

        매번 같은 구간(발화 시작~현재)을 인식하므로 겹침 제거가 필요 없고,
        가장 최근 가설이 곧 현재까지의 최선이다. 이어 붙이려 하면 오히려
        앞 윈도우의 부정확한 인식이 남아 중복된다.
        """
        self.updates += 1
        text = text.strip()
        self._prev_text = text
        start = self.committed_time if start is None else start
        end = start if end is None else max(start, end)
        self.unstable = [Word(t, start, end) for t in text.split()]
        self._record_revision()
        return "", join_words(self.unstable)

    def _update_with_words(self, words: list[Word]) -> tuple[str, str]:
        # 1) 이미 확정된 구간의 단어는 버린다 (overlap dedup)
        fresh = [w for w in words if w.end > self.committed_time + self.commit_margin]
        # 1b) 타임스탬프가 흔들려 1) 을 빠져나온 중복은 텍스트로 한 번 더 거른다
        fresh = _drop_repeated_head(self.committed, fresh)
        # 1c) 남은 게 없으면 unstable 을 건드리지 않는다.
        #     여기서 rest 로 덮어쓰면 아직 확정 안 된 꼬리가 통째로 사라진다.
        if not fresh:
            return "", join_words(self.unstable)

        # 2) 새 가설이 이어받지 못하는 앞부분을 확정
        orphans = self._commit(self._orphan_cut(fresh))

        # 3) 직전 가설과의 공통 접두사 = 두 번 연속 같게 나온 부분만 확정
        agreed = _common_prefix(self._prev, fresh)
        self._prev = fresh
        if self.agreement <= 1:
            newly, rest = fresh, []
        else:
            newly, rest = agreed, fresh[len(agreed):]

        if newly:
            self.committed.extend(newly)
            self.committed_time = max(self.committed_time, newly[-1].end)
        self.unstable = rest

        self._record_revision()
        committed_text = " ".join(
            t for t in (join_words(orphans), join_words(newly)) if t
        )
        return committed_text, join_words(self.unstable)

    def _record_revision(self) -> None:
        """이미 화면에 보여준 텍스트가 뒤집혔는지 센다.

        partial transcript revision rate = 새로 보여줄 단어열이 직전 단어열의
        단순한 '연장'이 아닌 업데이트의 비율. 뒤에 덧붙기만 하면 revision 이
        아니고, 앞서 보여준 단어가 다른 단어로 바뀌면 revision 이다.
        """
        display = [normalize(w.text) for w in (self.committed + self.unstable)]
        if self._last_display and display[: len(self._last_display)] != self._last_display:
            self.revisions += 1
        self._last_display = display

    def finalize(self) -> Utterance | None:
        """endpoint 도달: unstable 까지 전부 확정하고 발화 하나를 닫는다."""
        words = self.committed + self.unstable
        self.committed, self.unstable = [], []
        self._prev, self._prev_text = [], ""
        self._last_display = []
        if not words:
            return None

        utt = Utterance(
            text=join_words(words),
            start=words[0].start,
            end=words[-1].end,
            words=words,
        )
        self.committed_time = utt.end
        self.utterances.append(utt)
        return utt

    # --------------------------------------------------------------- 조회용
    @property
    def stable_text(self) -> str:
        """확정된 발화들 + 현재 발화의 확정 부분."""
        past = " ".join(u.text for u in self.utterances)
        current = join_words(self.committed)
        return " ".join(p for p in (past, current) if p).strip()

    @property
    def partial_text(self) -> str:
        return join_words(self.unstable)

    @property
    def full_text(self) -> str:
        return " ".join(p for p in (self.stable_text, self.partial_text) if p).strip()

    @property
    def revision_rate(self) -> float:
        return self.revisions / self.updates if self.updates else 0.0

    def reset(self) -> None:
        self.__init__(self.agreement, self.commit_margin)


# ---------------------------------------------------------------- helpers
def _common_prefix(a: list[Word], b: list[Word]) -> list[Word]:
    """두 단어열의 공통 접두사(정규화 비교). 값은 새 쪽(b)을 쓴다."""
    out: list[Word] = []
    for wa, wb in zip(a, b):
        if normalize(wa.text) != normalize(wb.text) or not normalize(wb.text):
            break
        out.append(wb)
    return out


def _drop_repeated_head(committed: list[Word], fresh: list[Word], max_words: int = 12) -> list[Word]:
    """committed 의 꼬리와 fresh 의 머리가 최장으로 겹치면 fresh 앞을 잘라낸다."""
    if not committed or not fresh:
        return fresh
    tail = [normalize(w.text) for w in committed[-max_words:]]
    head = [normalize(w.text) for w in fresh[:max_words]]
    for k in range(min(len(tail), len(head)), 0, -1):
        if tail[-k:] == head[:k]:
            return fresh[k:]
    return fresh


def _strip_overlap(prev: str, new: str, max_words: int = 40) -> str:
    """prev 의 접미사와 new 의 접두사가 최장으로 겹치는 만큼 new 앞을 잘라낸다."""
    if not prev:
        return new
    p, n = prev.split(), new.split()
    limit = min(len(p), len(n), max_words)
    for k in range(limit, 0, -1):
        if [normalize(x) for x in p[-k:]] == [normalize(x) for x in n[:k]]:
            return " ".join(n[k:])
    return new
