"""성능 비교 — 저장된 세션의 지표를 모아 보고, 정답 대비 WER/CER 을 계산한다.

개발문서 7단계("동일 의료 음성 데이터셋으로 비교")용 화면.
"""
import json

import pandas as pd
import streamlit as st

from stt.config import TRANSCRIPTS_DIR
from stt.metrics.evaluator import evaluate

st.title("성능 비교")
st.caption("Whisper / Zipformer / SenseVoice 를 같은 오디오로 돌린 뒤 지표를 비교합니다.")


@st.cache_data(ttl="30s")
def load_sessions() -> pd.DataFrame:
    """transcripts/*.json 을 한 장의 표로."""
    rows = []
    for path in sorted(TRANSCRIPTS_DIR.glob("*.json"), reverse=True):
        try:
            snap = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        m = snap.get("metrics", {})
        rows.append(
            {
                "session_id": snap.get("session_id", path.stem),
                "engine": snap.get("engine"),
                "오디오(s)": m.get("audio_sec"),
                "RTF": m.get("rtf_mean"),
                "첫 partial(ms)": m.get("first_partial_ms"),
                "final 지연(ms)": m.get("final_latency_ms"),
                "revision": m.get("revision_rate"),
                "CPU(%)": m.get("cpu_mean_pct"),
                "메모리(MB)": m.get("rss_mean_mb"),
                "발화": m.get("utterances"),
                "텍스트": snap.get("stable", ""),
            }
        )
    return pd.DataFrame(rows)


df = load_sessions()

if df.empty:
    st.info(
        "아직 저장된 세션이 없습니다. **실시간 전사** 페이지에서 녹음하거나 "
        "STT 서버로 세션을 만들면 `transcripts/` 에 쌓입니다.",
        icon=":material/inbox:",
    )
    st.stop()

with st.container(horizontal=True, horizontal_alignment="right"):
    if st.button("새로고침", icon=":material/refresh:"):
        load_sessions.clear()
        st.rerun()

st.subheader("세션별 지표")
event = st.dataframe(
    df.drop(columns=["텍스트"]),
    hide_index=True,
    on_select="rerun",
    selection_mode="multi-row",
    key="sessions",
)

selected = df.iloc[event.selection.rows] if event.selection.rows else df.head(0)

numeric = ["RTF", "첫 partial(ms)", "final 지연(ms)", "revision", "CPU(%)"]
engine_summary = (
    df.groupby("engine", dropna=True)[numeric].mean(numeric_only=True).round(3).reset_index()
)
if len(engine_summary) > 1:
    st.subheader("엔진별 평균")
    st.dataframe(engine_summary, hide_index=True)
    st.bar_chart(engine_summary, x="engine", y="RTF", horizontal=True)

# ------------------------------------------------------------------ 정확도
st.subheader("정확도 (WER / CER)")
default_ref = st.session_state.get("reference_text", "")
reference = st.text_area(
    "정답 전사문 (reference)",
    default_ref,
    height=140,
    placeholder="사람이 작성한 정답 전사문. 파일 전사 페이지의 결과가 자동으로 채워집니다.",
)

if not reference.strip():
    st.info("정답 전사문을 입력하면 선택한 세션들의 WER / CER 을 계산합니다.",
            icon=":material/rule:")
    st.stop()

targets = selected if not selected.empty else df
results = []
for _, row in targets.iterrows():
    scores = evaluate(reference, row["텍스트"] or "")
    results.append(
        {
            "session_id": row["session_id"],
            "engine": row["engine"],
            "WER": scores["wer"],
            "CER": scores["cer"],
            "의료용어 recall": scores["medical"]["recall"],
            "누락 용어": ", ".join(scores["medical"]["missed"][:5]),
            "RTF": row["RTF"],
        }
    )

acc = pd.DataFrame(results)
st.dataframe(acc, hide_index=True)
st.scatter_chart(acc, x="RTF", y="CER", color="engine",
                 x_label="Real-Time Factor (낮을수록 빠름)", y_label="CER (낮을수록 정확)")

if selected.empty:
    st.caption("위 표에서 행을 선택하면 해당 세션만 비교합니다.")
