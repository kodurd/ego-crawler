"""Analytics page — aggregated stats across all sessions."""
from __future__ import annotations

import os

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

DB_PATH = os.environ.get("DB_PATH", "ego_crawler.duckdb")

st.set_page_config(page_title="Аналитика", page_icon="📊", layout="wide")
st.title("📊 Аналитика")


@st.cache_resource
def get_conn():
    return duckdb.connect(DB_PATH, read_only=True)


conn = get_conn()

# ── Guard ──────────────────────────────────────────────────────────────────────

try:
    n = conn.execute("SELECT COUNT(*) FROM steps").fetchone()[0]
except Exception:
    n = 0

if n == 0:
    st.info("Нет данных для анализа. Запустите хотя бы одну сессию.", icon="💡")
    st.stop()

# ── Tool usage ─────────────────────────────────────────────────────────────────

st.subheader("Использование инструментов")

tool_rows = conn.execute("""
    SELECT action_tool, COUNT(*) AS uses
    FROM steps
    WHERE step_type = 'ACTION' AND action_tool IS NOT NULL
    GROUP BY action_tool
    ORDER BY uses DESC
""").fetchall()

if tool_rows:
    tool_df = pd.DataFrame(tool_rows, columns=["инструмент", "вызовов"])
    fig = px.bar(
        tool_df, x="инструмент", y="вызовов",
        color="инструмент",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig.update_layout(showlegend=False, margin=dict(t=20, b=20), height=280)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.caption("Действий ещё не было.")

st.divider()

# ── Per-session cost table ─────────────────────────────────────────────────────

st.subheader("Стоимость по сессиям")

cost_rows = conn.execute("""
    SELECT
        s.persona_name,
        s.model,
        s.start_time,
        COALESCE(SUM(st.token_count_input),  0) AS tok_in,
        COALESCE(SUM(st.token_count_output), 0) AS tok_out
    FROM sessions s
    LEFT JOIN steps st ON st.session_id = s.id AND st.step_type = 'THOUGHT'
    GROUP BY s.id, s.persona_name, s.model, s.start_time
    ORDER BY s.start_time DESC
""").fetchall()

cost_df = pd.DataFrame(cost_rows, columns=["персона", "модель", "дата", "токены↑", "токены↓"])
cost_df["дата"] = pd.to_datetime(cost_df["дата"]).dt.strftime("%Y-%m-%d %H:%M")
cost_df["стоимость $"] = (
    (cost_df["токены↑"] / 1000 * 0.0004) + (cost_df["токены↓"] / 1000 * 0.0012)
).map("{:.4f}".format)

st.dataframe(cost_df, use_container_width=True, hide_index=True)

st.divider()

# ── Tokens per step distribution ───────────────────────────────────────────────

st.subheader("Распределение токенов на шаг")

tok_rows = conn.execute("""
    SELECT token_count_input, token_count_output
    FROM steps
    WHERE step_type = 'THOUGHT'
      AND token_count_input IS NOT NULL
""").fetchall()

if tok_rows:
    tok_df = pd.DataFrame(tok_rows, columns=["входящие", "исходящие"])
    fig2 = px.histogram(
        tok_df.melt(var_name="тип", value_name="токены"),
        x="токены", color="тип", barmode="overlay",
        nbins=30, opacity=0.7,
        color_discrete_map={"входящие": "#4C9BE8", "исходящие": "#F4A261"},
    )
    fig2.update_layout(margin=dict(t=20, b=20), height=280)
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Latency distribution ───────────────────────────────────────────────────────

st.subheader("Задержка LLM (мс)")

lat_rows = conn.execute("""
    SELECT latency_ms
    FROM steps
    WHERE step_type = 'THOUGHT' AND latency_ms IS NOT NULL
""").fetchall()

if lat_rows:
    lat_df = pd.DataFrame(lat_rows, columns=["мс"])
    fig3 = px.box(lat_df, y="мс", points="all", color_discrete_sequence=["#57CC99"])
    fig3.update_layout(margin=dict(t=20, b=20), height=300)
    st.plotly_chart(fig3, use_container_width=True)
