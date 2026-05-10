from __future__ import annotations

import duckdb

_DDL_CREATE = """
CREATE TABLE IF NOT EXISTS sessions (
    id                      TEXT PRIMARY KEY,
    persona_name            TEXT NOT NULL,
    persona_archetype       TEXT NOT NULL,
    persona_age             INTEGER NOT NULL,
    model                   TEXT NOT NULL,
    status                  TEXT NOT NULL,
    budget_total_minutes    INTEGER NOT NULL,
    budget_remaining_seconds INTEGER NOT NULL,
    start_time              TIMESTAMPTZ NOT NULL,
    end_time                TIMESTAMPTZ,
    current_mood            TEXT NOT NULL,
    current_mood_intensity  REAL NOT NULL,
    current_step_number     INTEGER NOT NULL DEFAULT 0,
    current_url             TEXT
);

CREATE TABLE IF NOT EXISTS steps (
    id                   TEXT PRIMARY KEY,
    session_id           TEXT NOT NULL REFERENCES sessions(id),
    step_number          INTEGER NOT NULL,
    step_type            TEXT NOT NULL,
    timestamp            TIMESTAMPTZ NOT NULL,
    thought_content      TEXT,
    action_tool          TEXT,
    action_params        TEXT,
    observation_raw      TEXT,
    observation_summary  TEXT,
    observation_success  BOOLEAN,
    latency_ms           INTEGER,
    token_count_input    INTEGER,
    token_count_output   INTEGER
);
"""

# Run once after CREATE to add columns introduced in later versions
_DDL_MIGRATE = """
ALTER TABLE steps ADD COLUMN IF NOT EXISTS observation_success BOOLEAN;
"""


def ensure_schema(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(_DDL_CREATE)
    conn.execute(_DDL_MIGRATE)
