"""응급실(ER) 도메인 용어 사전 및 전사 후처리.

두 곳에서 쓴다.
  1. Whisper `initial_prompt` — 디코딩을 도메인 어휘 쪽으로 편향시킨다.
  2. 후처리 `correct()` — 자주 나오는 오인식을 사전/유사도 기반으로 교정한다.

용어를 추가할 때는 TERMS 에 넣으면 프롬프트와 유사도 교정에 동시에 반영된다.
CONFUSIONS 는 "확실히 틀린" 표기만 넣는다(무조건 치환되므로).

``dataset/STT_test_50_info.xlsx``의 선별근거 시트에 정의된 네 평가
분류를 그대로 사용한다. 단, ``medical_term_accuracy``는 활력징후와
통증 점수를 별도 지표로 유지하기 위해 **의학용어**와 **영문_약어 표기**만
평가한다.
"""
from __future__ import annotations

import difflib
import re
from functools import lru_cache

#: STT_test_50_info.xlsx의 확정 평가 어휘 (initial_prompt + 유사도 교정 후보)
TERMS: dict[str, list[str]] = {
    "활력징후 항목": ["혈압", "맥박", "호흡수", "체온", "산소포화도", "혈당", "GCS"],
    # 해당 항목이 포함된 파일은 23건이다. 실제로 낭독되는 표기는 NRS/FPRS다.
    "통증 점수 유무": ["통증점수", "통증", "NRS", "FPRS"],
    "의학용어": [
        "NRS", "mmHg", "감각", "과거력", "근력", "디스크", "맥박", "산소포화도", "소변", "수술",
        "저림", "체온", "혈압", "호흡", "활력징후", "SPO2", "복용", "복통", "생리", "의식",
        "진통제", "구토", "외상", "인후통", "BST", "당뇨", "발적", "부종", "화상", "고혈압",
        "GCS", "고지혈증", "지혈", "경련", "기면", "발작", "가슴통증", "호흡곤란", "발진", "쌕쌕",
        "알러지", "양수", "임신", "질출혈", "출혈", "태동", "설사", "V/S", "시야", "약물",
        "두통", "메스꺼", "당뇨약", "보행", "혈압약", "이물감", "공황", "처방", "월경", "발열",
        "고열", "청력", "투석", "기침", "구음장애", "마비", "명료", "편마비", "흉통", "골절",
        "어지럼", "옆구리", "와파린", "혈뇨", "복부", "뇌졸중", "수액", "실신", "기저질환", "간경화", "가려움",
    ],
    "영문_약어 표기": [
        "MOTOR", "NRS", "POWER", "mmHg", "Alert", "SPO2", "FPRS", "BST", "mg/dL", "GCS",
        "grade", "motor", "stupor", "V/S", "kg", "ESRD", "Mental", "cc",
    ],
}

# medical_term_accuracy의 평가 범위. 활력징후·통증점수는 TERMS에 남겨
# 프롬프트/교정에는 쓰되 이 지표의 정답·가설 비교에서는 제외한다.
MEDICAL_ACCURACY_CATEGORIES = ("의학용어", "영문_약어 표기")

# 낭독본은 영문 표기를 한국어 음가로 읽는다. 정확도 평가는 엑셀의 원 표기를
# canonical 값으로 유지하면서 이 별칭들 중 하나가 전사되면 해당 항목을 맞춘다.
TERM_ALIASES: dict[str, tuple[str, ...]] = {
    "MOTOR": ("MOTOR", "motor", "모터"),
    "NRS": ("NRS", "엔알에스"),
    "POWER": ("POWER", "파워"),
    "mmHg": ("mmHg", "엠엠에이치지"),
    "Alert": ("Alert", "얼럿"),
    "SPO2": ("SPO2", "SpO2", "에스피오투"),
    "FPRS": ("FPRS", "에프피알에스"),
    "BST": ("BST", "비에스티"),
    "mg/dL": ("mg/dL", "밀리그램 퍼 데시리터"),
    "GCS": ("GCS", "지씨에스"),
    "grade": ("grade", "그레이드"),
    "stupor": ("stupor", "스투퍼"),
    "V/S": ("V/S", "브이에스"),
    "kg": ("kg", "킬로그램"),
    "ESRD": ("ESRD", "이에스알디"),
    "Mental": ("Mental", "멘탈"),
    "cc": ("cc", "씨씨"),
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
def medical_accuracy_terms() -> tuple[str, ...]:
    """의료용어 정확도에 포함할 canonical 항목(중복 제거)."""
    return tuple(dict.fromkeys(
        term
        for category in MEDICAL_ACCURACY_CATEGORIES
        for term in TERMS[category]
    ))


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
    """텍스트에 등장한 전체 도메인 용어 목록(프롬프트/후처리용)."""
    lowered = text.lower()
    return [t for t in all_terms() if t.lower() in lowered]


def find_medical_accuracy_terms(text: str) -> list[str]:
    """의학용어·영문 약어 표기만 대상으로 canonical 항목을 찾는다."""
    lowered = text.lower()
    found: list[str] = []
    for term in medical_accuracy_terms():
        aliases = TERM_ALIASES.get(term, (term,))
        if any(alias.lower() in lowered for alias in aliases):
            found.append(term)
    return found
