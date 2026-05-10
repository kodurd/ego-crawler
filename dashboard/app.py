"""Streamlit dashboard — ego-crawler.

Run:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import os
from pathlib import Path

import duckdb
import streamlit as st

# ── Config ─────────────────────────────────────────────────────────────────────

DB_PATH = os.environ.get("DB_PATH", "ego_crawler.duckdb")

st.set_page_config(
    page_title="ego-crawler dashboard",
    page_icon="🧠",
    layout="wide",
)

# ── Helpers ────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_conn():
    return duckdb.connect(DB_PATH, read_only=True)


def sessions_exist() -> bool:
    try:
        conn = get_conn()
        n = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        return n > 0
    except Exception:
        return False


# ── Main page ──────────────────────────────────────────────────────────────────

st.title("🧠 ego-crawler")
st.caption("Симулятор поведения пользователя · аналитика сессий")

if not Path(DB_PATH).exists() or not sessions_exist():
    st.info(
        "База данных пуста. Запустите `py run_anya.py` чтобы начать первую сессию.",
        icon="💡",
    )
    st.stop()

conn = get_conn()

# ── Summary metrics ────────────────────────────────────────────────────────────

totals = conn.execute("""
    SELECT
        COUNT(*)                                         AS total_sessions,
        SUM(CASE WHEN status = 'ACTIVE' THEN 1 ELSE 0 END) AS active,
        SUM(CASE WHEN status = 'COMPLETED' THEN 1 ELSE 0 END) AS completed
    FROM sessions
""").fetchone()

total_tokens = conn.execute("""
    SELECT
        COALESCE(SUM(token_count_input),  0) AS tok_in,
        COALESCE(SUM(token_count_output), 0) AS tok_out
    FROM steps
    WHERE step_type = 'THOUGHT'
""").fetchone()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Сессий всего",      totals[0])
col2.metric("Активных",          totals[1])
col3.metric("Завершённых",       totals[2])
tok_in, tok_out = total_tokens
approx_usd = (tok_in / 1000 * 0.0004) + (tok_out / 1000 * 0.0012)
col4.metric("Примерная стоимость", f"${approx_usd:.4f}")

st.divider()

# ── Sessions table ─────────────────────────────────────────────────────────────

st.subheader("Список сессий")

rows = conn.execute("""
    SELECT
        s.id,
        s.persona_name,
        s.persona_archetype,
        s.model,
        s.status,
        s.start_time,
        s.budget_total_minutes,
        s.budget_remaining_seconds,
        COUNT(st.id)                              AS steps_count,
        COALESCE(SUM(st.token_count_input),  0)   AS tok_in,
        COALESCE(SUM(st.token_count_output), 0)   AS tok_out
    FROM sessions s
    LEFT JOIN steps st ON st.session_id = s.id
    GROUP BY s.id, s.persona_name, s.persona_archetype, s.model, s.status,
             s.start_time, s.budget_total_minutes, s.budget_remaining_seconds
    ORDER BY s.start_time DESC
""").fetchall()

import pandas as pd

df = pd.DataFrame(rows, columns=[
    "id", "персона", "архетип", "модель", "статус", "старт",
    "бюджет (мин)", "остаток (сек)", "шагов", "токены↑", "токены↓",
])

df["старт"] = pd.to_datetime(df["старт"]).dt.strftime("%Y-%m-%d %H:%M")
df["израсходовано"] = df.apply(
    lambda r: f"{(r['бюджет (мин)'] * 60 - r['остаток (сек)']) // 60}:"
              f"{(r['бюджет (мин)'] * 60 - r['остаток (сек)']) % 60:02d}",
    axis=1,
)
df["стоимость $"] = (
    (df["токены↑"] / 1000 * 0.0004) + (df["токены↓"] / 1000 * 0.0012)
).map("{:.4f}".format)

display_df = df[["id", "персона", "архетип", "модель", "статус", "старт",
                  "шагов", "израсходовано", "стоимость $"]]

st.dataframe(display_df, use_container_width=True, hide_index=True)

# ── Navigation hint ────────────────────────────────────────────────────────────

st.caption("👈 Используйте боковое меню для перехода к деталям сессии или аналитике.")
