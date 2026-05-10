"""ego-crawler dashboard — Session Feed.

Run:
    uv run streamlit run dashboard/app.py
"""
from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st

DB_PATH = os.environ.get("DB_PATH", "ego_crawler.duckdb")

st.set_page_config(
    page_title="ego-crawler",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧠 ego-crawler · Session Feed")
st.caption("Как агент ведёт себя в сети — поведенческая аналитика")


@st.cache_resource
def get_conn():
    return duckdb.connect(DB_PATH, read_only=True)


def db_ok() -> bool:
    if not Path(DB_PATH).exists():
        return False
    try:
        return get_conn().execute("SELECT COUNT(*) FROM sessions").fetchone()[0] > 0
    except Exception:
        return False


if not db_ok():
    st.info(
        "База данных пуста. Запустите `uv run py run_anya.py` чтобы начать сессию.",
        icon="💡",
    )
    st.stop()

conn = get_conn()

# ── Global metrics ─────────────────────────────────────────────────────────────

m = conn.execute("""
    SELECT
        COUNT(DISTINCT s.id)                                          AS sessions,
        COALESCE(SUM(st.token_count_input),  0)                       AS tok_in,
        COALESCE(SUM(st.token_count_output), 0)                       AS tok_out,
        COUNT(CASE WHEN st.step_type = 'ACTION' THEN 1 END)           AS actions,
        COUNT(CASE WHEN st.observation_success = FALSE THEN 1 END)    AS failures
    FROM sessions s
    LEFT JOIN steps st ON st.session_id = s.id
""").fetchone()

sessions_n, tok_in, tok_out, actions_n, failures_n = m
cost = (tok_in / 1000 * 0.0004) + (tok_out / 1000 * 0.0012)
success_rate = round((1 - failures_n / actions_n) * 100, 1) if actions_n else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Сессий",          sessions_n)
c2.metric("Действий",        actions_n)
c3.metric("Успех действий",  f"{success_rate}%")
c4.metric("Токены ↑/↓",     f"{tok_in:,} / {tok_out:,}")
c5.metric("Стоимость",       f"${cost:.4f}")

st.divider()

# ── Session list ───────────────────────────────────────────────────────────────

st.subheader("Сессии")

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
        s.current_mood,
        s.current_mood_intensity,
        COUNT(CASE WHEN st.step_type = 'THOUGHT' THEN 1 END)           AS thoughts,
        COUNT(CASE WHEN st.step_type = 'ACTION'  THEN 1 END)           AS actions,
        COUNT(CASE WHEN st.observation_success = FALSE THEN 1 END)     AS errors,
        COALESCE(SUM(st.token_count_input),  0)                        AS tok_in,
        COALESCE(SUM(st.token_count_output), 0)                        AS tok_out
    FROM sessions s
    LEFT JOIN steps st ON st.session_id = s.id
    GROUP BY s.id, s.persona_name, s.persona_archetype, s.model, s.status,
             s.start_time, s.budget_total_minutes, s.budget_remaining_seconds,
             s.current_mood, s.current_mood_intensity
    ORDER BY s.start_time DESC
""").fetchall()

df = pd.DataFrame(rows, columns=[
    "id", "персона", "архетип", "модель", "статус", "старт",
    "бюджет_мин", "остаток_сек",
    "настроение", "интенсивность",
    "мыслей", "действий", "ошибок",
    "токены↑", "токены↓",
])

df["дата"] = pd.to_datetime(df["старт"]).dt.strftime("%Y-%m-%d %H:%M")
used = df["бюджет_мин"] * 60 - df["остаток_сек"]
df["израсх."] = (used // 60).astype(str) + ":" + (used % 60).astype(int).map("{:02d}".format)
df["$"] = ((df["токены↑"] / 1000 * 0.0004) + (df["токены↓"] / 1000 * 0.0012)).map("{:.4f}".format)
df["mood"] = df["настроение"] + " " + df["интенсивность"].map("{:.2f}".format)

display = df[["дата", "персона", "архетип", "модель", "статус",
              "мыслей", "действий", "ошибок", "израсх.", "$", "mood", "id"]]

st.dataframe(display, use_container_width=True, hide_index=True)
st.caption("👈 Выберите страницу в боковом меню для углублённого анализа.")
