from __future__ import annotations

from pathlib import Path
from typing import Optional
from uuid import UUID

import duckdb

from ego_crawler.domain.entities.session import Session, SessionStatus
from ego_crawler.domain.entities.persona import Persona
from ego_crawler.domain.repositories.session_repository import SessionRepository
from ego_crawler.domain.value_objects.budget import Budget
from ego_crawler.domain.value_objects.emotion import Emotion, Mood
from ego_crawler.infrastructure.persistence.duckdb.schema import ensure_schema


class DuckDBSessionRepository(SessionRepository):
    """Persists Session aggregates to a DuckDB file."""

    def __init__(self, db_path: str | Path = "ego_crawler.duckdb") -> None:
        self._conn = duckdb.connect(str(db_path))
        ensure_schema(self._conn)

    # ── Write ──────────────────────────────────────────────────────────────────

    def save(self, session: Session) -> None:
        self._conn.execute(
            """
            INSERT INTO sessions
                (id, persona_name, persona_archetype, persona_age, model, status,
                 budget_total_minutes, budget_remaining_seconds,
                 start_time, end_time,
                 current_mood, current_mood_intensity,
                 current_step_number, current_url)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (id) DO UPDATE SET
                status                   = excluded.status,
                budget_remaining_seconds = excluded.budget_remaining_seconds,
                end_time                 = excluded.end_time,
                current_mood             = excluded.current_mood,
                current_mood_intensity   = excluded.current_mood_intensity,
                current_step_number      = excluded.current_step_number,
                current_url              = excluded.current_url
            """,
            (
                str(session.id),
                session.persona.name,
                session.persona.archetype,
                session.persona.age,
                session.persona.context_vars.get("model", "qwen-plus"),
                session.status.name,
                session.budget.total_minutes,
                session.budget.remaining_seconds,
                session.start_time,
                session.end_time,
                session.current_emotion.mood.name,
                session.current_emotion.intensity,
                session.current_step_number,
                session.current_url,
            ),
        )

    # ── Read ───────────────────────────────────────────────────────────────────

    def get_by_id(self, session_id: UUID) -> Optional[Session]:
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (str(session_id),)
        ).fetchone()
        return self._row_to_session(row) if row else None

    def get_active(self) -> Optional[Session]:
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE status = 'ACTIVE' ORDER BY start_time DESC LIMIT 1"
        ).fetchone()
        return self._row_to_session(row) if row else None

    # ── Mapping ────────────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_session(row: tuple) -> Session:
        (
            id_, persona_name, persona_archetype, persona_age, model, status,
            budget_total, budget_remaining,
            start_time, end_time,
            mood_name, mood_intensity,
            step_number, current_url,
        ) = row

        persona = Persona(
            name=persona_name,
            archetype=persona_archetype,
            age=persona_age,
            context_vars={"model": model},
        )
        return Session(
            id=UUID(id_),
            persona=persona,
            status=SessionStatus[status],
            budget=Budget(
                total_minutes=budget_total,
                remaining_seconds=budget_remaining,
            ),
            start_time=start_time,
            end_time=end_time,
            current_emotion=Emotion(mood=Mood[mood_name], intensity=mood_intensity),
            current_step_number=step_number,
            current_url=current_url,
        )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "DuckDBSessionRepository":
        return self

    def __exit__(self, *_) -> None:
        self.close()
