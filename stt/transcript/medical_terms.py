"""응급실(ER) 도메인 용어 사전 및 전사 후처리.

두 곳에서 쓴다.
  1. Whisper `initial_prompt` — 디코딩을 도메인 어휘 쪽으로 편향시킨다.
  2. 후처리 `correct()` — 자주 나오는 오인식을 사전/유사도 기반으로 교정한다.

용어를 추가할 때는 TERMS 에 넣으면 프롬프트와 유사도 교정에 동시에 반영된다.
CONFUSIONS 는 "확실히 틀린" 표기만 넣는다(무조건 치환되므로).
"""
from __future__ import annotations

import difflib
import re
from functools import lru_cache

#: 도메인 어휘 (initial_prompt + 유사도 교정 후보)
TERMS: dict[str, list[str]] = {
    "증상": [
        "복통", "흉통", "두통", "호흡곤란", "발열", "오한", "구토", "구역",
        "설사", "혈변", "토혈", "객혈", "어지럼증", "실신", "경련", "발작",
        "마비", "저림", "부종", "발진", "가려움", "기침", "가래", "인후통",
        "혈뇨", "배뇨통", "요통", "관절통", "근육통", "전신쇠약",
    ],
    "진단": [
        "심근경색", "협심증", "부정맥", "심부전", "뇌경색", "뇌출혈", "뇌졸중",
        "폐렴", "폐색전증", "기흉", "천식", "만성폐쇄성폐질환", "패혈증",
        "충수염", "췌장염", "담낭염", "담석증", "장폐색", "위장관출혈",
        "요로결석", "신우신염", "당뇨병성케톤산증", "저혈당", "아나필락시스",
        "골절", "탈구", "열상", "타박상", "화상", "뇌진탕",
    ],
    "활력징후": [
        "혈압", "수축기혈압", "이완기혈압", "맥박", "심박수", "호흡수",
        "체온", "산소포화도", "의식수준", "지에스씨", "GCS",
    ],
    "검사": [
        "심전도", "흉부엑스레이", "씨티", "CT", "엠알아이", "MRI", "초음파",
        "혈액검사", "동맥혈가스분석", "소변검사", "트로포닌", "디다이머",
        "씨알피", "CRP", "백혈구수치", "헤모글로빈", "혈당",
    ],
    "처치·약물": [
        "정맥주사", "수액", "생리식염수", "산소투여", "기관삽관", "심폐소생술",
        "제세동", "니트로글리세린", "아스피린", "모르핀", "에피네프린",
        "아세트아미노펜", "이부프로펜", "세프트리악손", "헤파린", "인슐린",
        "진통제", "해열제", "항생제", "항응고제", "진정제",
    ],
    "과거력": [
        "고혈압", "당뇨", "고지혈증", "심방세동", "만성신부전", "간경화",
        "천식", "결핵", "암", "수술력", "알레르기", "복용약",
    ],
}

#: 반드시 틀린 표기 -> 올바른 표기 (정확 치환)
CONFUSIONS: dict[str, str] = {
    "심금경색": "심근경색",
    "심근겨색": "심근경색",
    "협신증": "협심증",
    "호흡 곤란": "호흡곤란",
    "호흡곤한": "호흡곤란",
    "부정막": "부정맥",
    "폐새전증": "폐색전증",
    "충소염": "충수염",
    "췌장년": "췌장염",
    "패혈정": "패혈증",
    "산소 포화도": "산소포화도",
    "산소포하도": "산소포화도",
    "심페소생술": "심폐소생술",
    "기관 삽관": "기관삽관",
    "생리 식염수": "생리식염수",
    "정맥 주사": "정맥주사",
    "뇌족중": "뇌졸중",
    "요로 결석": "요로결석",
    "혈액 검사": "혈액검사",
    "씨 티": "CT",
    "시티": "CT",
}


@lru_cache(maxsize=1)
def all_terms() -> tuple[str, ...]:
    return tuple(dict.fromkeys(t for group in TERMS.values() for t in group))


@lru_cache(maxsize=1)
def build_initial_prompt(max_chars: int = 220) -> str:
    """Whisper initial_prompt. 너무 길면 디코딩이 프롬프트에 끌려가므로 짧게."""
    head = (
        "응급실 진료 대화입니다. 환자 증상과 활력징후, 진단명, 처치를 기록합니다. "
    )
    picked: list[str] = []
    budget = max_chars - len(head)
    for term in all_terms():
        if budget - (len(term) + 2) < 0:
            break
        picked.append(term)
        budget -= len(term) + 2
    return head + ", ".join(picked) + "."


_TOKEN = re.compile(r"[가-힣A-Za-z]+")


def correct(text: str, cutoff: float = 0.86) -> str:
    """오인식 교정. (1) 확정 치환 -> (2) 어절 단위 유사도 교정."""
    if not text:
        return text

    for wrong, right in CONFUSIONS.items():
        text = text.replace(wrong, right)

    terms = all_terms()
    lookup = {t.lower(): t for t in terms}

    def fix(match: re.Match) -> str:
        """어절에서 '어간'만 교정하고 뒤에 붙은 조사/어미는 그대로 둔다.

        한국어는 교착어라 어절 전체를 사전 표제어로 치환하면 '호흡 곤란과'가
        '호흡곤란'이 되어 조사가 사라진다. 그래서 뒤에서부터 잘라가며
        어간 후보를 찾고, 길이가 같은 표제어에만 교정을 적용한다.
        """
        token = match.group(0)
        if len(token) < 2 or token.lower() in lookup:
            return token

        for cut in range(len(token), max(1, len(token) - 3), -1):
            stem, suffix = token[:cut], token[cut:]
            if len(stem) < 2:
                break
            if stem.lower() in lookup:
                return lookup[stem.lower()] + suffix
            if len(stem) < 3:
                continue
            close = difflib.get_close_matches(stem, terms, n=1, cutoff=cutoff)
            if close and len(close[0]) == len(stem):
                return close[0] + suffix
        return token

    return _TOKEN.sub(fix, text)


def find_terms(text: str) -> list[str]:
    """텍스트에 등장한 도메인 용어 목록(의료 용어 정확도 평가용)."""
    lowered = text.lower()
    return [t for t in all_terms() if t.lower() in lowered]
