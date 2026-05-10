"""Thought Patterns — как агент думает: токены, темп, ключевые слова."""
from __future__ import annotations

import os
import re
from collections import Counter

import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DB_PATH = os.environ.get("DB_PATH", "ego_crawler.duckdb")

st.set_page_config(page_title="Thought Patterns", page_icon="💭", layout="wide")
st.title("💭 Thought Patterns — как агент думает")


@st.cache_resource
def get_conn():
    return duckdb.connect(DB_PATH, read_only=True)


conn = get_conn()

# ── Session picker ─────────────────────────────────────────────────────────────

sessions = conn.execute(
    "SELECT id, persona_name, start_time FROM sessions ORDER BY start_time DESC"
).fetchall()

if not sessions:
    st.info("Нет данных.", icon="💡")
    st.stop()

all_opt = "— все сессии —"
options = {all_opt: None}
options.update({f"{r[1]}  ·  {str(r[2])[:16]}  ·  {r[0][:8]}…": r[0] for r in sessions})
chosen = st.selectbox("Сессия", list(options.keys()))
session_filter = options[chosen]

sid_clause = "AND session_id = ?" if session_filter else ""
params = (session_filter,) if session_filter else ()

# ── Fetch thought steps ────────────────────────────────────────────────────────

rows = conn.execute(
    f"""
    SELECT step_number, thought_content, latency_ms,
           token_count_input, token_count_output
    FROM steps
    WHERE step_type = 'THOUGHT' {sid_clause}
    ORDER BY step_number
    """,
    params,
).fetchall()

if not rows:
    st.info("Мыслей ещё нет.")
    st.stop()

df = pd.DataFrame(rows, columns=["шаг", "мысль", "мс", "токены↑", "токены↓"])
df["токены"] = df["токены↑"].fillna(0) + df["токены↓"].fillna(0)
df["слов"] = df["мысль"].fillna("").apply(lambda t: len(t.split()))

# ── Top-level stats ────────────────────────────────────────────────────────────

c1, c2, c3, c4 = st.columns(4)
c1.metric("Всего мыслей",    len(df))
c2.metric("Ср. токенов",     int(df["токены"].mean()))
c3.metric("Ср. задержка мс", int(df["мс"].dropna().mean()) if df["мс"].notna().any() else "—")
c4.metric("Ср. слов в мысли", int(df["слов"].mean()))

st.divider()

# ── Token trend ────────────────────────────────────────────────────────────────

col1, col2 = st.columns(2)

with col1:
    st.subheader("Токены по шагам")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["шаг"], y=df["токены↑"].fillna(0),
                         name="входящие", marker_color="#4C9BE8"))
    fig.add_trace(go.Bar(x=df["шаг"], y=df["токены↓"].fillna(0),
                         name="исходящие", marker_color="#F4A261"))
    fig.update_layout(barmode="stack", margin=dict(t=10, b=10),
                      height=260, legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Задержка LLM по шагам (мс)")
    lat_df = df[df["мс"].notna()]
    if not lat_df.empty:
        fig2 = px.line(lat_df, x="шаг", y="мс", markers=True,
                       color_discrete_sequence=["#F4A261"])
        fig2.add_hline(y=lat_df["мс"].mean(), line_dash="dot",
                       annotation_text=f"среднее {int(lat_df['мс'].mean())} мс",
                       line_color="gray")
        fig2.update_layout(margin=dict(t=10, b=10), height=260)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.caption("Нет данных о задержках.")

st.divider()

# ── Depth of thought: words per step ──────────────────────────────────────────

st.subheader("Длина мысли (слов) — насколько глубоко рассуждает агент")
fig3 = px.bar(df, x="шаг", y="слов", color="слов",
              color_continuous_scale="Teal",
              labels={"слов": "слов в мысли"})
fig3.update_layout(coloraxis_showscale=False, margin=dict(t=10, b=10), height=220)
st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ── Keyword frequency ──────────────────────────────────────────────────────────

st.subheader("Топ слов в мыслях")

STOPWORDS = {
    "и", "в", "на", "что", "это", "я", "не", "с", "а", "но", "как", "по",
    "из", "за", "у", "к", "о", "так", "то", "же", "он", "она", "они", "мне",
    "мы", "вы", "ты", "для", "от", "до", "ещё", "уже", "да", "нет", "если",
    "бы", "то", "при", "или", "the", "to", "a", "of", "and", "is", "in",
    "that", "it", "for", "on", "are", "with", "as", "at", "be", "by", "an",
    "нужно", "надо", "можно", "хочу", "буду", "мой", "мне", "свой",
}

all_words: list[str] = []
for text in df["мысль"].dropna():
    words = re.findall(r"[а-яёА-ЯЁa-zA-Z]{3,}", text.lower())
    all_words.extend(w for w in words if w not in STOPWORDS)

if all_words:
    top_words = pd.DataFrame(
        Counter(all_words).most_common(25),
        columns=["слово", "частота"],
    )
    fig4 = px.bar(
        top_words, x="частота", y="слово", orientation="h",
        color="частота", color_continuous_scale="Purples",
    )
    fig4.update_layout(
        yaxis={"categoryorder": "total ascending"},
        coloraxis_showscale=False,
        margin=dict(t=10, b=10),
        height=max(250, len(top_words) * 26),
    )
    st.plotly_chart(fig4, use_container_width=True)

st.divider()

# ── Full thought log ───────────────────────────────────────────────────────────

st.subheader("Лог мыслей")
for _, row in df.iterrows():
    label = f"Шаг {int(row['шаг'])}  ·  {int(row['слов'])} слов  ·  {int(row['токены'])} токенов"
    with st.expander(label, expanded=False):
        st.markdown(row["мысль"] or "—")
