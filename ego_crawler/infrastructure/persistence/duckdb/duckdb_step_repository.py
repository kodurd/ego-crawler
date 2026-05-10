from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import UUID

import duckdb

from ego_crawler.domain.entities.agent_step import AgentStep
from ego_crawler.domain.repositories.step_repository import StepRepository
from ego_crawler.domain.value_objects.step_type import StepType
from ego_crawler.infrastructure.persistence.duckdb.schema import ensure_schema


class DuckDBStepRepository(StepRepository):
    """Persists AgentStep records to DuckDB."""

    def __init__(self, db_path: str | Path = "ego_crawler.duckdb") -> None:
        self._conn = duckdb.connect(str(db_path))
        ensure_schema(self._conn)

    # ── Write ──────────────────────────────────────────────────────────────────

    def save(self, step: AgentStep) -> None:
        self._conn.execute(
            """
            INSERT INTO steps
                (id, session_id, step_number, step_type, timestamp,
                 thought_content, action_tool, action_params,
                 observation_raw, observation_summary, observation_success,
                 latency_ms, token_count_input, token_count_output)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (id) DO UPDATE SET
                observation_raw     = excluded.observation_raw,
                observation_summary = excluded.observation_summary,
                observation_success = excluded.observation_success,
                latency_ms          = excluded.latency_ms,
                token_count_input   = excluded.token_count_input,
                token_count_output  = excluded.token_count_output
            """,
            (
                str(step.id),
                str(step.session_id),
                step.step_number,
                step.step_type.name,
                step.timestamp,
                step.thought_content,
                step.action_tool,
                json.dumps(step.action_params, ensure_ascii=False) if step.action_params else None,
                step.observation_raw,
                step.observation_summary,
                step.observation_success,
                step.latency_ms,
                step.token_count_input,
                step.token_count_output,
            ),
        )

    # ── Read ───────────────────────────────────────────────────────────────────

    def get_by_session(self, session_id: UUID, limit: Optional[int] = None) -> list[AgentStep]:
        q = "SELECT * FROM steps WHERE session_id = ? ORDER BY step_number, timestamp"
        params = (str(session_id),)
        if limit:
            q += f" LIMIT {int(limit)}"
        rows = self._conn.execute(q, params).fetchall()
        return [self._row_to_step(r) for r in rows]

    def get_latest(self, session_id: UUID, n: int = 1) -> list[AgentStep]:
        rows = self._conn.execute(
            "SELECT * FROM steps WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
            (str(session_id), n),
        ).fetchall()
        return [self._row_to_step(r) for r in reversed(rows)]

    # ── Mapping ────────────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_step(row: tuple) -> AgentStep:
        (
            id_, session_id, step_number, step_type, timestamp,
            thought_content, action_tool, action_params_json,
            observation_raw, observation_summary, observation_success,
            latency_ms, token_count_input, token_count_output,
        ) = row

        action_params = None
        if action_params_json:
            try:
                action_params = json.loads(action_params_json)
            except (json.JSONDecodeError, TypeError):
                action_params = {}

        ts = timestamp
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        if ts and ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        return AgentStep(
            id=UUID(id_),
            session_id=UUID(session_id),
            step_number=step_number,
            step_type=StepType[step_type],
            timestamp=ts,
            thought_content=thought_content,
            action_tool=action_tool,
            action_params=action_params,
            observation_raw=observation_raw,
            observation_summary=observation_summary,
            observation_success=observation_success,
            latency_ms=latency_ms,
            token_count_input=token_count_input,
            token_count_output=token_count_output,
        )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "DuckDBStepRepository":
        return self

    def __exit__(self, *_) -> None:
        self.close()
