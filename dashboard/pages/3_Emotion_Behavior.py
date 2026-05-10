"""Emotion & Behavior — как эмоции влияют на поведение агента."""
from __future__ import annotations

import json
import os

import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DB_PATH = os.environ.get("DB_PATH", "ego_crawler.duckdb")

st.set_page_config(page_title="Emotion & Behavior", page_icon="🎭", layout="wide")
st.title("🎭 Emotion & Behavior — эмоции и поведение")
st.caption("Как настроение меняется со временем и влияет на выбор действий")


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

# ── Mood arc (per session, ordered by step) ────────────────────────────────────
# We reconstruct mood per step: after each OBSERVATION the session emotion is
# updated. We read session snapshots from sessions table (current state only),
# but the richer signal comes from correlating action SUCCESS/FAIL with emotion.
#
# Pragmatic approach: collect (step_number, observation_success) and annotate
# mood changes: success → CURIOUS/EXCITED, fail → FRUSTRATED.

sid_clause = "AND st.session_id = ?" if session_filter else ""
params = (session_filter,) if session_filter else ()

obs_rows = conn.execute(
    f"""
    SELECT st.session_id, st.step_number,
           st.observation_success,
           s.current_mood, s.current_mood_intensity
    FROM steps st
    JOIN sessions s ON s.id = st.session_id
    WHERE st.step_type = 'OBSERVATION' {sid_clause}
    ORDER BY st.session_id, st.step_number
    """,
    params,
).fetchall()

action_rows = conn.execute(
    f"""
    SELECT st.session_id, st.step_number, st.action_tool,
           ob.observation_success
    FROM steps st
    LEFT JOIN steps ob ON ob.session_id = st.session_id
                       AND ob.step_number = st.step_number
                       AND ob.step_type = 'OBSERVATION'
    WHERE st.step_type = 'ACTION' {sid_clause}
    ORDER BY st.session_id, st.step_number
    """,
    params,
).fetchall()

if not obs_rows and not action_rows:
    st.info("Действий ещё нет — запустите сессию.")
    st.stop()

obs_df     = pd.DataFrame(obs_rows, columns=["session", "шаг", "успех", "mood", "intensity"])
action_df  = pd.DataFrame(action_rows, columns=["session", "шаг", "инструмент", "успех"])

# ── Success/fail rate ─────────────────────────────────────────────────────────

st.subheader("Результативность действий")

col1, col2 = st.columns(2)

with col1:
    known = obs_df[obs_df["успех"].notna()]
    if not known.empty:
        counts = known["успех"].map({True: "✅ успех", False: "❌ ошибка"}).value_counts()
        fig = px.pie(
            values=counts.values, names=counts.index,
            color=counts.index,
            color_discrete_map={"✅ успех": "#57CC99", "❌ ошибка": "#E63946"},
            hole=0.45,
        )
        fig.update_layout(margin=dict(t=10, b=10), height=260,
                          legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("Нет данных о результатах действий.")

with col2:
    if not action_df.empty:
        tool_ok = action_df.groupby("инструмент")["успех"].agg(
            total="count",
            successes=lambda x: (x == True).sum(),
        ).reset_index()
        tool_ok["% успеха"] = (tool_ok["successes"] / tool_ok["total"] * 100).round(1)

        fig2 = px.bar(
            tool_ok, x="инструмент", y="% успеха",
            color="% успеха",
            color_continuous_scale=["#E63946", "#FFD166", "#57CC99"],
            range_color=[0, 100],
            text="% успеха",
        )
        fig2.update_traces(texttemplate="%{text}%", textposition="outside")
        fig2.update_layout(coloraxis_showscale=False,
                           margin=dict(t=10, b=10), height=260,
                           yaxis_range=[0, 110])
        st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Mood arc over steps ────────────────────────────────────────────────────────

st.subheader("Настроение по шагам — реконструкция через исходы действий")
st.caption(
    "Точные данные о настроении не сохраняются пошагово (только финальное). "
    "Здесь мы аппроксимируем: ✅ → позитив, ❌ → негатив."
)

if not obs_df.empty:
    arc_df = obs_df.copy()
    arc_df["тональность"] = arc_df["успех"].map(
        lambda v: 0.7 if v is True else (-0.6 if v is False else 0.0)
    )
    # running mean for smoothing
    arc_df["сглаженная"] = arc_df["тональность"].rolling(3, min_periods=1).mean()
    arc_df["цвет"] = arc_df["тональность"].map(
        lambda v: "#57CC99" if v > 0 else ("#E63946" if v < 0 else "#AAAAAA")
    )

    fig3 = go.Figure()
    fig3.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)
    fig3.add_trace(go.Bar(
        x=arc_df["шаг"], y=arc_df["тональность"],
        marker_color=arc_df["цвет"],
        name="исход шага",
        opacity=0.6,
    ))
    fig3.add_trace(go.Scatter(
        x=arc_df["шаг"], y=arc_df["сглаженная"],
        mode="lines+markers",
        name="скользящее среднее",
        line=dict(color="#4C9BE8", width=2),
    ))
    fig3.update_layout(
        margin=dict(t=10, b=10), height=280,
        yaxis_title="тональность (условная)",
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ── Tool choice by outcome of previous step ────────────────────────────────────

st.subheader("Следующий инструмент после успеха vs ошибки")
st.caption("Меняет ли агент стратегию, если предыдущее действие не сработало?")

if len(action_df) >= 2:
    action_df_sorted = action_df.sort_values(["session", "шаг"]).copy()
    action_df_sorted["prev_success"] = action_df_sorted.groupby("session")["успех"].shift(1)
    reaction_df = action_df_sorted[action_df_sorted["prev_success"].notna()].copy()
    reaction_df["контекст"] = reaction_df["prev_success"].map(
        {True: "после успеха", False: "после ошибки"}
    )

    if not reaction_df.empty:
        heat = reaction_df.groupby(["контекст", "инструмент"]).size().reset_index(name="раз")
        fig4 = px.density_heatmap(
            heat, x="инструмент", y="контекст", z="раз",
            color_continuous_scale="Blues",
            text_auto=True,
        )
        fig4.update_layout(margin=dict(t=10, b=10), height=220,
                           coloraxis_showscale=False)
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.caption("Недостаточно данных для анализа реакции.")

st.divider()

# ── Final mood per session ─────────────────────────────────────────────────────

st.subheader("Финальное настроение по сессиям")

mood_rows = conn.execute("""
    SELECT persona_name, current_mood, current_mood_intensity, start_time
    FROM sessions
    ORDER BY start_time DESC
""").fetchall()

if mood_rows:
    mood_df = pd.DataFrame(mood_rows, columns=["персона", "настроение", "интенсивность", "дата"])
    mood_df["дата"] = pd.to_datetime(mood_df["дата"]).dt.strftime("%Y-%m-%d %H:%M")

    fig5 = px.bar(
        mood_df, x="дата", y="интенсивность",
        color="настроение",
        color_discrete_map={
            "CURIOUS": "#4C9BE8", "EXCITED": "#57CC99",
            "HAPPY": "#95D44A",
            "ANXIOUS": "#FFD166", "BORED": "#AAAAAA",
            "FRUSTRATED": "#E63946",
        },
        text="настроение",
    )
    fig5.update_traces(textposition="inside")
    fig5.update_layout(
        yaxis_range=[0, 1.1],
        margin=dict(t=10, b=10), height=260,
        legend_title="настроение",
    )
    st.plotly_chart(fig5, use_container_width=True)
