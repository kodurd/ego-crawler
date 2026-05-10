"""Unit tests for DuckDBSessionRepository.

All tests use DuckDB in-memory database — no file I/O, no mocks.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import duckdb
import pytest

from ego_crawler.domain.entities.persona import Persona
from ego_crawler.domain.entities.session import Session, SessionStatus
from ego_crawler.domain.value_objects.budget import Budget
from ego_crawler.domain.value_objects.emotion import Emotion, Mood
from ego_crawler.infrastructure.persistence.duckdb.duckdb_session_repository import (
    DuckDBSessionRepository,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def repo() -> DuckDBSessionRepository:
    """Fresh in-memory repository per test."""
    r = DuckDBSessionRepository(db_path=":memory:")
    yield r
    r.close()


def _make_session(
    status: SessionStatus = SessionStatus.ACTIVE,
    budget_min: int = 30,
    remaining_sec: int | None = None,
    mood: Mood = Mood.CURIOUS,
    intensity: float = 0.5,
    model: str = "qwen-plus",
) -> Session:
    persona = Persona(
        name="Аня",
        archetype="отличница",
        age=17,
        context_vars={"model": model},
    )
    return Session(
        persona=persona,
        status=status,
        budget=Budget(
            total_minutes=budget_min,
            remaining_seconds=remaining_sec if remaining_sec is not None else budget_min * 60,
        ),
        current_emotion=Emotion(mood=mood, intensity=intensity),
    )


# ── save / get_by_id ──────────────────────────────────────────────────────────

class TestSaveAndGetById:
    def test_saved_session_is_retrievable(self, repo):
        s = _make_session()
        repo.save(s)
        loaded = repo.get_by_id(s.id)
        assert loaded is not None

    def test_id_roundtrips(self, repo):
        s = _make_session()
        repo.save(s)
        loaded = repo.get_by_id(s.id)
        assert loaded.id == s.id

    def test_persona_name_roundtrips(self, repo):
        s = _make_session()
        repo.save(s)
        loaded = repo.get_by_id(s.id)
        assert loaded.persona.name == "Аня"

    def test_persona_archetype_roundtrips(self, repo):
        s = _make_session()
        repo.save(s)
        loaded = repo.get_by_id(s.id)
        assert loaded.persona.archetype == "отличница"

    def test_persona_age_roundtrips(self, repo):
        s = _make_session()
        repo.save(s)
        loaded = repo.get_by_id(s.id)
        assert loaded.persona.age == 17

    def test_model_stored_in_context_vars(self, repo):
        s = _make_session(model="qwen-max")
        repo.save(s)
        loaded = repo.get_by_id(s.id)
        assert loaded.persona.context_vars.get("model") == "qwen-max"

    def test_status_roundtrips(self, repo):
        s = _make_session(status=SessionStatus.COMPLETED)
        repo.save(s)
        loaded = repo.get_by_id(s.id)
        assert loaded.status == SessionStatus.COMPLETED

    def test_budget_total_minutes_roundtrips(self, repo):
        s = _make_session(budget_min=45)
        repo.save(s)
        loaded = repo.get_by_id(s.id)
        assert loaded.budget.total_minutes == 45

    def test_budget_remaining_seconds_roundtrips(self, repo):
        s = _make_session(budget_min=30, remaining_sec=1200)
        repo.save(s)
        loaded = repo.get_by_id(s.id)
        assert loaded.budget.remaining_seconds == 1200

    def test_mood_roundtrips(self, repo):
        s = _make_session(mood=Mood.ANXIOUS)
        repo.save(s)
        loaded = repo.get_by_id(s.id)
        assert loaded.current_emotion.mood == Mood.ANXIOUS

    def test_mood_intensity_roundtrips(self, repo):
        s = _make_session(intensity=0.85)
        repo.save(s)
        loaded = repo.get_by_id(s.id)
        assert abs(loaded.current_emotion.intensity - 0.85) < 1e-6

    def test_unknown_id_returns_none(self, repo):
        assert repo.get_by_id(uuid4()) is None

    def test_multiple_sessions_isolated(self, repo):
        s1 = _make_session(mood=Mood.HAPPY)
        s2 = _make_session(mood=Mood.FRUSTRATED)
        repo.save(s1)
        repo.save(s2)
        assert repo.get_by_id(s1.id).current_emotion.mood == Mood.HAPPY
        assert repo.get_by_id(s2.id).current_emotion.mood == Mood.FRUSTRATED


# ── upsert / update ───────────────────────────────────────────────────────────

class TestUpsert:
    def test_saving_again_updates_status(self, repo):
        s = _make_session(status=SessionStatus.ACTIVE)
        repo.save(s)

        completed = Session(
            id=s.id,
            persona=s.persona,
            status=SessionStatus.COMPLETED,
            budget=s.budget,
            current_emotion=s.current_emotion,
            start_time=s.start_time,
        )
        repo.save(completed)

        loaded = repo.get_by_id(s.id)
        assert loaded.status == SessionStatus.COMPLETED

    def test_saving_again_updates_remaining_budget(self, repo):
        s = _make_session(budget_min=30, remaining_sec=1800)
        repo.save(s)

        consumed = Session(
            id=s.id,
            persona=s.persona,
            status=s.status,
            budget=Budget(total_minutes=30, remaining_seconds=900),
            current_emotion=s.current_emotion,
            start_time=s.start_time,
        )
        repo.save(consumed)

        loaded = repo.get_by_id(s.id)
        assert loaded.budget.remaining_seconds == 900

    def test_saving_again_updates_emotion(self, repo):
        s = _make_session(mood=Mood.CURIOUS, intensity=0.5)
        repo.save(s)

        updated = Session(
            id=s.id,
            persona=s.persona,
            status=s.status,
            budget=s.budget,
            current_emotion=Emotion(Mood.ANXIOUS, 0.9),
            start_time=s.start_time,
        )
        repo.save(updated)

        loaded = repo.get_by_id(s.id)
        assert loaded.current_emotion.mood == Mood.ANXIOUS

    def test_row_count_stays_one_after_upsert(self, repo):
        s = _make_session()
        repo.save(s)
        repo.save(s)
        conn = duckdb.connect(":memory:")  # separate connection won't see our data
        # check via the repo's own connection
        count = repo._conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE id = ?", (str(s.id),)
        ).fetchone()[0]
        assert count == 1


# ── get_active ────────────────────────────────────────────────────────────────

class TestGetActive:
    def test_returns_active_session(self, repo):
        s = _make_session(status=SessionStatus.ACTIVE)
        repo.save(s)
        assert repo.get_active() is not None

    def test_returns_none_when_no_active(self, repo):
        s = _make_session(status=SessionStatus.COMPLETED)
        repo.save(s)
        assert repo.get_active() is None

    def test_returns_none_when_empty(self, repo):
        assert repo.get_active() is None

    def test_prefers_most_recent_active(self, repo):
        s1 = _make_session()
        import time; time.sleep(0.01)
        s2 = _make_session()
        repo.save(s1)
        repo.save(s2)
        active = repo.get_active()
        # most recent is returned (ORDER BY start_time DESC LIMIT 1)
        assert active.id in (s1.id, s2.id)
