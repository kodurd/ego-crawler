"""Web Trail — куда ходит агент: домены, запросы, навигационный поток."""
from __future__ import annotations

import json
import os
import re
from urllib.parse import urlparse

import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DB_PATH = os.environ.get("DB_PATH", "ego_crawler.duckdb")

st.set_page_config(page_title="Web Trail", page_icon="🗺️", layout="wide")
st.title("🗺️ Web Trail — куда ходит агент")


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

all_opt = "— все сессии —"
options = {all_opt: None}
options.update({f"{r[1]}  ·  {str(r[2])[:16]}  ·  {r[0][:8]}…": r[0] for r in sessions})
chosen = st.selectbox("Сессия", list(options.keys()))
session_filter = options[chosen]

sid_clause = "AND st.session_id = ?" if session_filter else ""
params_base = (session_filter,) if session_filter else ()

# ── Fetch action steps ─────────────────────────────────────────────────────────

actions = conn.execute(
    f"""
    SELECT st.session_id, st.step_number, st.action_tool, st.action_params,
           ob.observation_success
    FROM steps st
    LEFT JOIN steps ob ON ob.session_id = st.session_id
                       AND ob.step_number = st.step_number
                       AND ob.step_type = 'OBSERVATION'
    WHERE st.step_type = 'ACTION' {sid_clause}
    ORDER BY st.session_id, st.step_number
    """,
    params_base,
).fetchall()

if not actions:
    st.info("Действий ещё нет в выбранной сессии.")
    st.stop()

# Parse params JSON
def _parse_params(raw):
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}

rows = []
for sess_id, step_n, tool, params_raw, success in actions:
    p = _parse_params(params_raw)
    url  = p.get("url", "")
    query = p.get("query", "")
    domain = ""
    if url:
        try:
            domain = urlparse(url).netloc or url
        except Exception:
            domain = url
    rows.append({
        "session": sess_id[:8] + "…",
        "шаг": step_n,
        "инструмент": tool or "",
        "url": url,
        "домен": domain,
        "запрос": query,
        "успех": success,
    })

df = pd.DataFrame(rows)

# ── Tool distribution ──────────────────────────────────────────────────────────

col1, col2 = st.columns(2)

with col1:
    st.subheader("Использование инструментов")
    tool_counts = df["инструмент"].value_counts().reset_index()
    tool_counts.columns = ["инструмент", "вызовов"]
    fig = px.bar(
        tool_counts, x="инструмент", y="вызовов",
        color="инструмент",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig.update_layout(showlegend=False, margin=dict(t=10, b=10), height=280)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Успех / ошибка по инструменту")
    sdf = df[df["успех"].notna()].copy()
    if not sdf.empty:
        sdf["результат"] = sdf["успех"].map({True: "✅ успех", False: "❌ ошибка"})
        fig2 = px.histogram(
            sdf, x="инструмент", color="результат",
            barmode="group",
            color_discrete_map={"✅ успех": "#57CC99", "❌ ошибка": "#E63946"},
        )
        fig2.update_layout(margin=dict(t=10, b=10), height=280, legend_title="")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.caption("Нет данных о результатах.")

st.divider()

# ── Top domains ────────────────────────────────────────────────────────────────

nav_df = df[df["домен"] != ""].copy()
if not nav_df.empty:
    st.subheader("Топ доменов")
    dom = nav_df["домен"].value_counts().head(15).reset_index()
    dom.columns = ["домен", "переходов"]
    fig3 = px.bar(
        dom, x="переходов", y="домен", orientation="h",
        color="переходов", color_continuous_scale="Blues",
    )
    fig3.update_layout(
        yaxis={"categoryorder": "total ascending"},
        margin=dict(t=10, b=10), height=max(200, len(dom) * 32),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.divider()

# ── Search queries ─────────────────────────────────────────────────────────────

search_df = df[(df["инструмент"] == "web_search") & (df["запрос"] != "")]
if not search_df.empty:
    st.subheader("Поисковые запросы")
    st.dataframe(
        search_df[["шаг", "запрос", "успех"]].rename(columns={"шаг": "шаг №"}),
        use_container_width=True,
        hide_index=True,
    )
    st.divider()

# ── Navigation sequence ────────────────────────────────────────────────────────

st.subheader("Последовательность действий (по шагам)")

if not df.empty:
    seq_df = df[["шаг", "инструмент", "url", "запрос", "успех"]].copy()
    seq_df["детали"] = seq_df.apply(
        lambda r: r["запрос"] if r["запрос"] else (r["url"][:60] + "…" if len(r["url"]) > 60 else r["url"]),
        axis=1,
    )
    seq_df["ok"] = seq_df["успех"].map(
        lambda v: "✅" if v is True else ("❌" if v is False else "—")
    )
    st.dataframe(
        seq_df[["шаг", "инструмент", "детали", "ok"]],
        use_container_width=True,
        hide_index=True,
    )

# ── Sankey: tool flow ──────────────────────────────────────────────────────────

if len(df) >= 2:
    st.divider()
    st.subheader("Поток инструментов (Sankey)")
    tools_seq = df["инструмент"].tolist()
    pairs = [(tools_seq[i], tools_seq[i + 1]) for i in range(len(tools_seq) - 1)]
    from collections import Counter
    pair_counts = Counter(pairs)

    all_tools = list(dict.fromkeys(tools_seq))
    tool_idx = {t: i for i, t in enumerate(all_tools)}

    sources = [tool_idx[a] for a, _ in pair_counts]
    targets = [tool_idx[b] for _, b in pair_counts]
    values  = list(pair_counts.values())

    fig_s = go.Figure(go.Sankey(
        node=dict(label=all_tools, pad=15, thickness=20),
        link=dict(source=sources, target=targets, value=values),
    ))
    fig_s.update_layout(margin=dict(t=10, b=10), height=320)
    st.plotly_chart(fig_s, use_container_width=True)
