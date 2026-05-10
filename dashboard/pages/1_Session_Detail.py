"""Session detail page — step timeline + emotion chart."""
from __future__ import annotations

import os

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

DB_PATH = os.environ.get("DB_PATH", "ego_crawler.duckdb")

st.set_page_config(page_title="Детали сессии", page_icon="🔍", layout="wide")
st.title("🔍 Детали сессии")


@st.cache_resource
def get_conn():
    return duckdb.connect(DB_PATH, read_only=True)


conn = get_conn()

# ── Session picker ─────────────────────────────────────────────────────────────

sessions = conn.execute(
    "SELECT id, persona_name, start_time FROM sessions ORDER BY start_time DESC"
).fetchall()

if not sessions:
    st.info("Нет сохранённых сессий.", icon="💡")
    st.stop()

options = {f"{r[1]}  ·  {str(r[2])[:16]}  ·  {r[0][:8]}…": r[0] for r in sessions}
selected_label = st.selectbox("Выберите сессию", list(options.keys()))
session_id = options[selected_label]

# ── Session header ─────────────────────────────────────────────────────────────

s = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
(
    sid, persona_name, persona_archetype, persona_age, model, status,
    budget_total, budget_remaining, start_time, end_time,
    mood, mood_intensity, step_number, current_url,
) = s

used_sec  = budget_total * 60 - budget_remaining
used_min, used_sec_r = divmod(used_sec, 60)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Персона",   f"{persona_name}, {persona_age} лет")
col2.metric("Статус",    status)
col3.metric("Шагов",     step_number)
col4.metric("Израсходовано", f"{used_min}:{used_sec_r:02d} / {budget_total}:00")

st.caption(f"Модель: `{model}`  ·  Настроение: **{mood}** ({mood_intensity:.2f})")

if current_url:
    st.caption(f"Последний URL: {current_url}")

st.divider()

# ── Steps data ─────────────────────────────────────────────────────────────────

steps_rows = conn.execute(
    """
    SELECT step_number, step_type, timestamp,
           thought_content, action_tool, action_params,
           observation_raw, observation_summary,
           latency_ms, token_count_input, token_count_output
    FROM steps
    WHERE session_id = ?
    ORDER BY step_number, timestamp
    """,
    (session_id,),
).fetchall()

if not steps_rows:
    st.info("Шагов ещё нет.")
    st.stop()

df = pd.DataFrame(steps_rows, columns=[
    "шаг", "тип", "время",
    "мысль", "инструмент", "параметры",
    "наблюдение", "резюме",
    "мс", "токены↑", "токены↓",
])

# ── Emotion timeline ───────────────────────────────────────────────────────────

thought_df = df[df["тип"] == "THOUGHT"].copy()
if not thought_df.empty:
    st.subheader("Таймлайн токенов")
    thought_df["токены"] = thought_df["токены↑"].fillna(0) + thought_df["токены↓"].fillna(0)
    fig = px.bar(
        thought_df, x="шаг", y="токены",
        labels={"шаг": "Шаг", "токены": "Токены"},
        color_discrete_sequence=["#4C9BE8"],
    )
    fig.update_layout(margin=dict(t=20, b=20), height=220)
    st.plotly_chart(fig, use_container_width=True)

# ── Latency chart ──────────────────────────────────────────────────────────────

    if thought_df["мс"].notna().any():
        st.subheader("Задержка LLM (мс)")
        fig2 = px.line(
            thought_df, x="шаг", y="мс",
            markers=True,
            labels={"шаг": "Шаг", "мс": "Задержка (мс)"},
            color_discrete_sequence=["#F4A261"],
        )
        fig2.update_layout(margin=dict(t=20, b=20), height=200)
        st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Step-by-step log ───────────────────────────────────────────────────────────

st.subheader("Лог шагов")

for _, row in df.iterrows():
    step_num = row["шаг"]
    stype    = row["тип"]

    if stype == "THOUGHT":
        with st.expander(f"🧠 Шаг {step_num} · МЫСЛЬ  ({row['мс'] or 0} мс)", expanded=False):
            st.markdown(row["мысль"] or "—")
            cols = st.columns(2)
            cols[0].metric("Токены ↑", int(row["токены↑"] or 0))
            cols[1].metric("Токены ↓", int(row["токены↓"] or 0))

    elif stype == "ACTION":
        with st.expander(f"⚡ Шаг {step_num} · ДЕЙСТВИЕ  `{row['инструмент'] or '—'}`", expanded=False):
            st.code(row["параметры"] or "{}", language="json")

    elif stype == "OBSERVATION":
        with st.expander(f"👁 Шаг {step_num} · НАБЛЮДЕНИЕ", expanded=False):
            if row["резюме"]:
                st.caption(f"Резюме: {row['резюме']}")
            raw = (row["наблюдение"] or "")[:2000]
            st.text(raw + ("…" if len(row["наблюдение"] or "") > 2000 else ""))
