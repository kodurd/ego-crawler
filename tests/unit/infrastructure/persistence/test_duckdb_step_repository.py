"""Unit tests for DuckDBStepRepository.

All tests use DuckDB in-memory database with a pre-seeded session row
so the foreign-key constraint on steps.session_id is satisfied.
"""
from __future__ import annotations

from uuid import uuid4

import pytest

from ego_crawler.domain.entities.agent_step import AgentStep
from ego_crawler.domain.entities.persona import Persona
from ego_crawler.domain.entities.session import Session, SessionStatus
from ego_crawler.domain.value_objects.budget import Budget
from ego_crawler.domain.value_objects.emotion import Emotion, Mood
from ego_crawler.domain.value_objects.step_type import StepType
from ego_crawler.infrastructure.persistence.duckdb.duckdb_session_repository import (
    DuckDBSessionRepository,
)
from ego_crawler.infrastructure.persistence.duckdb.duckdb_step_repository import (
    DuckDBStepRepository,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def repos():
    """Pair of (session_repo, step_repo) sharing the same in-memory DB."""
    session_repo = DuckDBSessionRepository(db_path=":memory:")
    # Share the same connection so FK constraint is satisfied
    step_repo = DuckDBStepRepository.__new__(DuckDBStepRepository)
    step_repo._conn = session_repo._conn
    yield session_repo, step_repo
    session_repo.close()


@pytest.fixture
def session_id(repos):
    """Pre-seeded session; returns its UUID."""
    session_repo, _ = repos
    persona = Persona(name="Аня", archetype="отличница", age=17)
    s = Session(
        persona=persona,
        status=SessionStatus.ACTIVE,
        budget=Budget.from_minutes(30),
        current_emotion=Emotion(Mood.CURIOUS, 0.5),
    )
    session_repo.save(s)
    return s.id


def _thought(session_id, step_number=1, content="Надо проверить", **kw) -> AgentStep:
    return AgentStep.create_thought(
        session_id=session_id,
        step_number=step_number,
        content=content,
        **kw,
    )


def _action(session_id, step_number=2, tool="web_search",
            params=None, **kw) -> AgentStep:
    return AgentStep.create_action(
        session_id=session_id,
        step_number=step_number,
        tool=tool,
        params=params or {"query": "ЕГЭ 2026"},
        **kw,
    )


def _observation(session_id, step_number=3, raw="<html>…</html>",
                 summary=None, observation_success=None, **kw) -> AgentStep:
    return AgentStep.create_observation(
        session_id=session_id,
        step_number=step_number,
        raw_data=raw,
        summary=summary,
        observation_success=observation_success,
        **kw,
    )


# ── save / get_by_session ─────────────────────────────────────────────────────

class TestSaveAndGetBySession:
    def test_saved_step_is_retrievable(self, repos, session_id):
        _, step_repo = repos
        step = _thought(session_id)
        step_repo.save(step)
        results = step_repo.get_by_session(session_id)
        assert len(results) == 1

    def test_step_id_roundtrips(self, repos, session_id):
        _, step_repo = repos
        step = _thought(session_id)
        step_repo.save(step)
        loaded = step_repo.get_by_session(session_id)[0]
        assert loaded.id == step.id

    def test_session_id_roundtrips(self, repos, session_id):
        _, step_repo = repos
        step = _thought(session_id)
        step_repo.save(step)
        loaded = step_repo.get_by_session(session_id)[0]
        assert loaded.session_id == session_id

    def test_step_number_roundtrips(self, repos, session_id):
        _, step_repo = repos
        step = _thought(session_id, step_number=7)
        step_repo.save(step)
        loaded = step_repo.get_by_session(session_id)[0]
        assert loaded.step_number == 7

    def test_step_type_thought_roundtrips(self, repos, session_id):
        _, step_repo = repos
        step = _thought(session_id)
        step_repo.save(step)
        loaded = step_repo.get_by_session(session_id)[0]
        assert loaded.step_type == StepType.THOUGHT

    def test_step_type_action_roundtrips(self, repos, session_id):
        _, step_repo = repos
        step = _action(session_id)
        step_repo.save(step)
        loaded = step_repo.get_by_session(session_id)[0]
        assert loaded.step_type == StepType.ACTION

    def test_step_type_observation_roundtrips(self, repos, session_id):
        _, step_repo = repos
        step = _observation(session_id)
        step_repo.save(step)
        loaded = step_repo.get_by_session(session_id)[0]
        assert loaded.step_type == StepType.OBSERVATION

    def test_thought_content_roundtrips(self, repos, session_id):
        _, step_repo = repos
        step = _thought(session_id, content="Надо посмотреть расписание")
        step_repo.save(step)
        loaded = step_repo.get_by_session(session_id)[0]
        assert loaded.thought_content == "Надо посмотреть расписание"

    def test_action_tool_roundtrips(self, repos, session_id):
        _, step_repo = repos
        step = _action(session_id, tool="navigate")
        step_repo.save(step)
        loaded = step_repo.get_by_session(session_id)[0]
        assert loaded.action_tool == "navigate"

    def test_action_params_roundtrips(self, repos, session_id):
        _, step_repo = repos
        params = {"url": "https://example.com", "timeout": 5000}
        step = _action(session_id, params=params)
        step_repo.save(step)
        loaded = step_repo.get_by_session(session_id)[0]
        assert loaded.action_params == params

    def test_action_params_unicode_roundtrips(self, repos, session_id):
        _, step_repo = repos
        params = {"query": "ЕГЭ математика 2026"}
        step = _action(session_id, params=params)
        step_repo.save(step)
        loaded = step_repo.get_by_session(session_id)[0]
        assert loaded.action_params["query"] == "ЕГЭ математика 2026"

    def test_observation_raw_roundtrips(self, repos, session_id):
        _, step_repo = repos
        step = _observation(session_id, raw="<div>Результат поиска</div>")
        step_repo.save(step)
        loaded = step_repo.get_by_session(session_id)[0]
        assert loaded.observation_raw == "<div>Результат поиска</div>"

    def test_observation_summary_roundtrips(self, repos, session_id):
        _, step_repo = repos
        step = _observation(session_id, summary="Страница загружена")
        step_repo.save(step)
        loaded = step_repo.get_by_session(session_id)[0]
        assert loaded.observation_summary == "Страница загружена"

    def test_latency_ms_roundtrips(self, repos, session_id):
        _, step_repo = repos
        step = _thought(session_id, latency_ms=1234)
        step_repo.save(step)
        loaded = step_repo.get_by_session(session_id)[0]
        assert loaded.latency_ms == 1234

    def test_token_counts_roundtrip(self, repos, session_id):
        _, step_repo = repos
        step = _thought(session_id, token_count_input=800, token_count_output=120)
        step_repo.save(step)
        loaded = step_repo.get_by_session(session_id)[0]
        assert loaded.token_count_input == 800
        assert loaded.token_count_output == 120

    def test_none_token_counts_stay_none(self, repos, session_id):
        _, step_repo = repos
        step = _thought(session_id)
        step_repo.save(step)
        loaded = step_repo.get_by_session(session_id)[0]
        assert loaded.token_count_input is None
        assert loaded.token_count_output is None

    def test_empty_for_unknown_session(self, repos, session_id):
        _, step_repo = repos
        results = step_repo.get_by_session(uuid4())
        assert results == []

    def test_returns_all_steps_ordered_by_step_number(self, repos, session_id):
        _, step_repo = repos
        step_repo.save(_thought(    session_id, step_number=1))
        step_repo.save(_action(     session_id, step_number=2))
        step_repo.save(_observation(session_id, step_number=3))
        results = step_repo.get_by_session(session_id)
        assert [r.step_number for r in results] == [1, 2, 3]

    def test_limit_parameter_respected(self, repos, session_id):
        _, step_repo = repos
        for i in range(1, 6):
            step_repo.save(_thought(session_id, step_number=i))
        results = step_repo.get_by_session(session_id, limit=3)
        assert len(results) == 3

    def test_steps_from_different_sessions_isolated(self, repos, session_id):
        session_repo, step_repo = repos
        persona = Persona(name="Другая", archetype="test", age=20)
        s2 = Session(
            persona=persona,
            status=SessionStatus.ACTIVE,
            budget=Budget.from_minutes(10),
            current_emotion=Emotion(Mood.HAPPY, 0.6),
        )
        session_repo.save(s2)

        step_repo.save(_thought(session_id, step_number=1))
        step_repo.save(_thought(s2.id,      step_number=1))

        assert len(step_repo.get_by_session(session_id)) == 1
        assert len(step_repo.get_by_session(s2.id))      == 1


# ── upsert ────────────────────────────────────────────────────────────────────

class TestUpsert:
    def test_saving_same_step_twice_does_not_duplicate(self, repos, session_id):
        _, step_repo = repos
        step = _thought(session_id)
        step_repo.save(step)
        step_repo.save(step)
        results = step_repo.get_by_session(session_id)
        assert len(results) == 1

    def test_upsert_updates_latency(self, repos, session_id):
        _, step_repo = repos
        step = _thought(session_id, latency_ms=100)
        step_repo.save(step)

        updated = AgentStep(
            id=step.id,
            session_id=step.session_id,
            step_number=step.step_number,
            step_type=step.step_type,
            thought_content=step.thought_content,
            latency_ms=999,
        )
        step_repo.save(updated)

        loaded = step_repo.get_by_session(session_id)[0]
        assert loaded.latency_ms == 999


# ── get_latest ────────────────────────────────────────────────────────────────

class TestGetLatest:
    def test_returns_last_n_steps(self, repos, session_id):
        _, step_repo = repos
        for i in range(1, 6):
            step_repo.save(_thought(session_id, step_number=i))
        latest = step_repo.get_latest(session_id, n=2)
        assert len(latest) == 2

    def test_latest_steps_in_chronological_order(self, repos, session_id):
        _, step_repo = repos
        for i in range(1, 6):
            step_repo.save(_thought(session_id, step_number=i))
        latest = step_repo.get_latest(session_id, n=3)
        assert latest[0].step_number <= latest[-1].step_number

    def test_get_latest_1_returns_single(self, repos, session_id):
        _, step_repo = repos
        for i in range(1, 4):
            step_repo.save(_thought(session_id, step_number=i))
        latest = step_repo.get_latest(session_id, n=1)
        assert len(latest) == 1

    def test_empty_when_no_steps(self, repos, session_id):
        _, step_repo = repos
        assert step_repo.get_latest(session_id) == []

    def test_empty_for_unknown_session(self, repos):
        _, step_repo = repos
        assert step_repo.get_latest(uuid4()) == []


# ── observation_success ───────────────────────────────────────────────────────

class TestObservationSuccess:
    def test_success_true_roundtrips(self, repos, session_id):
        _, step_repo = repos
        step = _observation(session_id, observation_success=True)
        step_repo.save(step)
        loaded = step_repo.get_by_session(session_id)[0]
        assert loaded.observation_success is True

    def test_success_false_roundtrips(self, repos, session_id):
        _, step_repo = repos
        step = _observation(session_id, observation_success=False)
        step_repo.save(step)
        loaded = step_repo.get_by_session(session_id)[0]
        assert loaded.observation_success is False

    def test_success_none_by_default(self, repos, session_id):
        _, step_repo = repos
        step = _observation(session_id)
        step_repo.save(step)
        loaded = step_repo.get_by_session(session_id)[0]
        assert loaded.observation_success is None

    def test_thought_step_success_is_none(self, repos, session_id):
        _, step_repo = repos
        step = _thought(session_id)
        step_repo.save(step)
        loaded = step_repo.get_by_session(session_id)[0]
        assert loaded.observation_success is None

    def test_upsert_updates_success(self, repos, session_id):
        _, step_repo = repos
        step = _observation(session_id, observation_success=None)
        step_repo.save(step)

        updated = AgentStep(
            id=step.id,
            session_id=step.session_id,
            step_number=step.step_number,
            step_type=step.step_type,
            observation_raw=step.observation_raw,
            observation_success=True,
        )
        step_repo.save(updated)

        loaded = step_repo.get_by_session(session_id)[0]
        assert loaded.observation_success is True
